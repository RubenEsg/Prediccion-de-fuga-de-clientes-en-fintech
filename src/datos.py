"""Carga, auditoría, variables derivadas, partición y etiqueta.

Orden obligatorio del flujo (corrección solicitada en la Entrega 1):

1. ``cargar_tablas`` → ``limpiar_transacciones`` → ``derivar_variables_transaccionales`` → ``unir``
2. ``partir``: se separan los índices de entrenamiento y prueba **antes** de mirar el objetivo.
3. ``diagnosticos_objetivo(df, particion)``: toda estadística que involucre al objetivo se
   calcula solo con el índice de entrenamiento (la función lo impone por sí misma).
4. ``depurar`` → ``separar`` → ``construir_etiqueta`` (umbral estimado solo con entrenamiento).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ETIQUETA_METODO, ETIQUETA_Q, OBJETIVO, SEED, TEST_SIZE, localizar_datos

# ------------------------------------------------------------------ constantes del dominio
COLUMNAS_A_ELIMINAR: list[str] = [
    'customer_id',
    'active_products',
    'customer_lifetime_value', 'clv_segment',
    'average_transaction_value', 'total_transaction_volume',
    'avg_daily_transactions', 'monthly_transaction_count',
    'first_transaction_date', 'last_transaction_date',
    'nps_score',
    'occupation',
    'complaint_topics', 'feature_requests',
    'last_survey_date', 'first_tx', 'last_tx',
]
"""Columnas descartadas con la evidencia de ``diagnosticos_objetivo`` (ver Entrega 2, §3)."""

BOOLEANAS: list[str] = ['savings_account', 'credit_card', 'personal_loan', 'investment_account',
                        'insurance_product', 'bill_payment_user', 'auto_savings_enabled']
"""Tenencia de productos. Se conservan: no reconstruyen ``active_products``."""

PARES_DUPLICADOS: list[tuple[str, str]] = [
    ('avg_tx_value', 'average_transaction_value'),
    ('total_tx_volume', 'total_transaction_volume'),
    ('transaction_frequency', 'avg_daily_transactions'),
    ('transaction_frequency', 'monthly_transaction_count'),
]


# ------------------------------------------------------------------------------ carga
def cargar_tablas(carpeta: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee ``customer_data.csv`` y ``transactions_data.csv``.

    Parameters
    ----------
    carpeta : Path or str, optional
        Carpeta con los CSV. Si se omite se localiza con ``config.localizar_datos``.

    Returns
    -------
    clientes : DataFrame
        Un registro por cliente (48.723 × 54 en la versión pública).
    tx : DataFrame
        Detalle de transacciones con ``date`` ya interpretada como fecha.
    """
    carpeta = Path(carpeta) if carpeta is not None else localizar_datos()
    clientes = pd.read_csv(carpeta / 'customer_data.csv')
    tx = pd.read_csv(carpeta / 'transactions_data.csv', parse_dates=['date'])
    return clientes, tx


def auditar_transacciones(clientes: pd.DataFrame, tx: pd.DataFrame) -> dict[str, int]:
    """Comprueba integridad entre las dos tablas.

    Parameters
    ----------
    clientes, tx : DataFrame
        Tablas tal como las devuelve :func:`cargar_tablas`.

    Returns
    -------
    dict
        ``duplicadas``, ``montos_no_positivos``, ``fechas_nulas``, ``huerfanas``
        (transacciones sin cliente) y ``clientes_sin_tx``.
    """
    return {
        'duplicadas': int(tx.duplicated().sum()),
        'montos_no_positivos': int((tx['amount'] <= 0).sum()),
        'fechas_nulas': int(tx['date'].isna().sum()),
        'huerfanas': int((~tx['customer_id'].isin(clientes['customer_id'])).sum()),
        'clientes_sin_tx': int((~clientes['customer_id'].isin(tx['customer_id'])).sum()),
    }


def limpiar_transacciones(tx: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas exactamente duplicadas del detalle transaccional.

    Parameters
    ----------
    tx : DataFrame

    Returns
    -------
    DataFrame
        Sin duplicados y con el índice reiniciado.
    """
    return tx.drop_duplicates().reset_index(drop=True)


def derivar_variables_transaccionales(tx: pd.DataFrame) -> pd.DataFrame:
    """Agrega el detalle por cliente en doce variables de comportamiento.

    Parameters
    ----------
    tx : DataFrame
        Transacciones limpias con ``customer_id``, ``date``, ``amount`` y ``type``.

    Returns
    -------
    DataFrame
        Indexado por ``customer_id`` con ``n_tx``, ``monto_total``, ``monto_medio``,
        ``monto_mediano``, ``recencia_dias``, ``dias_activo``, ``volatilidad_monto``,
        ``meses_activos`` y ``prop_<tipo>`` por tipo de operación.

    Notes
    -----
    Ninguna de estas variables usa el objetivo, de modo que pueden calcularse sobre el
    total de clientes sin riesgo de fuga. La recencia se mide contra la última fecha del
    detalle completo.
    """
    tx = tx.copy()
    fecha_corte = tx['date'].max()
    base = tx.groupby('customer_id').agg(
        n_tx=('amount', 'size'), monto_total=('amount', 'sum'),
        monto_medio=('amount', 'mean'), monto_mediano=('amount', 'median'),
        monto_desv=('amount', 'std'), ultima_tx=('date', 'max'), primera_tx=('date', 'min'))
    base['recencia_dias'] = (fecha_corte - base['ultima_tx']).dt.days
    base['dias_activo'] = (base['ultima_tx'] - base['primera_tx']).dt.days
    base['volatilidad_monto'] = base['monto_desv'] / base['monto_medio']
    base = base.drop(columns=['ultima_tx', 'primera_tx', 'monto_desv'])
    tx['mes'] = tx['date'].dt.to_period('M')
    meses = tx.groupby('customer_id')['mes'].nunique().rename('meses_activos')
    mix = pd.crosstab(tx['customer_id'], tx['type'], normalize='index').add_prefix('prop_')
    return base.join([meses, mix])


def unir(clientes: pd.DataFrame, variables_tx: pd.DataFrame) -> pd.DataFrame:
    """Une la tabla de clientes con las variables derivadas.

    Parameters
    ----------
    clientes : DataFrame
    variables_tx : DataFrame
        Indexado por ``customer_id`` (salida de :func:`derivar_variables_transaccionales`).

    Returns
    -------
    DataFrame
        ``left join`` sobre ``customer_id``; conserva el índice de ``clientes``.
    """
    return clientes.merge(variables_tx, left_on='customer_id', right_index=True, how='left')


def columnas_transaccionales(df: pd.DataFrame) -> list[str]:
    """Nombres de las variables derivadas del detalle presentes en ``df``.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    list of str
    """
    fijas = ['n_tx', 'monto_total', 'monto_medio', 'monto_mediano', 'recencia_dias',
             'dias_activo', 'volatilidad_monto', 'meses_activos']
    return [c for c in df.columns if c in fijas or c.startswith('prop_')]


# --------------------------------------------------------------------------- partición
@dataclass
class Particion:
    """Índices de las particiones. Se crean **antes** de cualquier uso del objetivo.

    Attributes
    ----------
    idx_train, idx_test : pandas.Index
        Índices de entrenamiento y prueba.
    idx_val : pandas.Index or None
        Índice de validación explícita, si se pidió ``val_size > 0``.
    seed : int
        Semilla usada para el reparto.
    """
    idx_train: pd.Index
    idx_test: pd.Index
    idx_val: pd.Index | None = None
    seed: int = SEED
    test_size: float = TEST_SIZE
    val_size: float = 0.0

    def __post_init__(self) -> None:
        conjuntos = [self.idx_train, self.idx_test] + ([self.idx_val] if self.idx_val is not None else [])
        total = sum(len(c) for c in conjuntos)
        union = conjuntos[0]
        for c in conjuntos[1:]:
            union = union.union(c)
        if len(union) != total:
            raise ValueError('las particiones se solapan')


def partir(df: pd.DataFrame, test_size: float = TEST_SIZE, val_size: float = 0.0,
           seed: int = SEED) -> Particion:
    """Reparte los índices de ``df`` en entrenamiento, (validación) y prueba.

    Parameters
    ----------
    df : DataFrame
        Tabla completa; solo se usa su índice.
    test_size : float
        Fracción de prueba sobre el total.
    val_size : float
        Fracción de validación sobre el total (0 para no crearla). Útil para el
        early stopping de XGBoost y para la gráfica entrenamiento/validación/prueba.
    seed : int
        Semilla del reparto.

    Returns
    -------
    Particion

    Notes
    -----
    El reparto se hace sobre el objetivo **continuo**, sin estratificar: la etiqueta
    binaria todavía no existe, porque su umbral se estimará solo con entrenamiento.
    """
    idx_tr, idx_te = train_test_split(df.index, test_size=test_size, random_state=seed)
    idx_va = None
    if val_size > 0:
        frac = val_size / (1 - test_size)
        idx_tr, idx_va = train_test_split(idx_tr, test_size=frac, random_state=seed)
    return Particion(pd.Index(idx_tr), pd.Index(idx_te), None if idx_va is None else pd.Index(idx_va),
                     seed=seed, test_size=test_size, val_size=val_size)


# ----------------------------------------------------------- diagnósticos (solo train)
def _eta2(categorica: pd.Series, continua: pd.Series) -> float:
    """Tamaño de efecto η² (varianza entre grupos / varianza total) de una categórica."""
    g = pd.DataFrame({'c': categorica.astype(str), 'y': continua}).dropna()
    media = g['y'].mean()
    entre = g.groupby('c')['y'].apply(lambda s: len(s) * (s.mean() - media) ** 2).sum()
    total = ((g['y'] - media) ** 2).sum()
    return float(entre / total) if total > 0 else float('nan')


def diagnosticos_objetivo(df: pd.DataFrame, particion: Particion, objetivo: str = OBJETIVO) -> dict:
    """Recalcula la evidencia que justifica cada eliminación de columna.

    Parameters
    ----------
    df : DataFrame
        Tabla unida completa (con ``active_products``, ``customer_lifetime_value``, etc.
        todavía presentes).
    particion : Particion
        La función restringe por sí misma al índice de entrenamiento todo estadístico que
        involucre al objetivo: no es posible pasarle filas de prueba por descuido.
    objetivo : str
        Nombre del objetivo continuo.

    Returns
    -------
    dict
        ``r_active_products``, ``r2_active_products``, ``escalera`` (DataFrame por nivel de
        ``active_products`` con media, desviación, conteo y salto), ``desv_saltos``,
        ``rho_clv_volumen``, ``pares_duplicados`` (DataFrame), ``r_satisfaccion_nps``,
        ``eta2`` (DataFrame), ``faltantes`` (DataFrame), ``contingencia_credito`` (DataFrame),
        ``ausencia_vs_objetivo`` (dict), ``booleanas_vs_active`` (dict) y ``corr_tx`` (Series).

    Notes
    -----
    Las estadísticas que miran el objetivo (correlaciones, escalera, η², ausencia frente al
    objetivo, derivadas transaccionales) se calculan **solo con entrenamiento**: si usaran
    las filas de prueba, la selección de variables incorporaría información de prueba. Las
    comprobaciones estructurales que no involucran al objetivo —pares de columnas
    duplicadas y recuento de faltantes— se hacen sobre la tabla completa, porque una
    columna permutada entre filas solo se reconoce como copia cuando se comparan todos sus
    valores; el diccionario indica en ``ambito`` qué se calculó con qué.
    """
    df_train = df.loc[particion.idx_train]
    obj = df_train[objetivo]
    r_ap = float(obj.corr(df_train['active_products']))
    escalera = df_train.groupby('active_products')[objetivo].agg(['count', 'mean', 'std'])
    escalera['salto'] = escalera['mean'].diff()

    filas = []
    for a, b in PARES_DUPLICADOS:                       # estructura de la tabla: sobre el total
        por_fila = float(np.isclose(df[a], df[b], rtol=1e-6).mean() * 100)
        mismos = float(np.isclose(np.sort(df[a].values), np.sort(df[b].values), rtol=1e-6).mean() * 100)
        filas.append([f'{a} / {b}', por_fila, mismos, float(df[a].corr(df[b]))])
    pares = pd.DataFrame(filas, columns=['par', 'coincide_por_fila_pct', 'mismos_valores_pct', 'r'])

    candidatas = ['occupation', 'last_survey_date', 'first_tx', 'last_tx']
    eta = pd.DataFrame({'variable': candidatas,
                        'niveles': [df_train[c].nunique() for c in candidatas],
                        'eta2': [_eta2(df_train[c], obj) for c in candidatas]})

    falt = df.isna().sum()                              # estructura de la tabla: sobre el total
    falt = falt[falt > 0].to_frame('faltantes').assign(pct=lambda d: d['faltantes'] / len(df) * 100)
    contingencia = pd.crosstab(df['credit_utilization_ratio'].isna(), df['credit_card'], normalize='index')
    ausencia = {}
    for col in ['complaint_topics', 'feature_requests']:
        a = df_train[col].isna().astype(int)
        ausencia[col] = {'r_con_objetivo': float(a.corr(obj)),
                         'objetivo_medio_presente': float(obj[a == 0].mean()),
                         'objetivo_medio_ausente': float(obj[a == 1].mean())}

    suma_bool = df_train[BOOLEANAS].sum(axis=1)
    booleanas = {'coincidencia_pct': float((suma_bool == df_train['active_products']).mean() * 100),
                 'r': float(suma_bool.corr(df_train['active_products']))}
    cols_tx = columnas_transaccionales(df_train)
    corr_tx = df_train[cols_tx].corrwith(obj).sort_values(key=abs, ascending=False)

    return {
        'n_train': int(len(df_train)),
        'n_total': int(len(df)),
        'ambito': {'entrenamiento': ['r_active_products', 'r2_active_products', 'escalera', 'desv_saltos',
                                     'rho_clv_volumen', 'r_satisfaccion_nps', 'eta2', 'ausencia_vs_objetivo',
                                     'booleanas_vs_active', 'corr_tx'],
                   'total': ['pares_duplicados', 'faltantes', 'contingencia_credito']},
        'r_active_products': r_ap,
        'r2_active_products': r_ap ** 2,
        'escalera': escalera,
        'desv_saltos': float(escalera['salto'].dropna().std()),
        'rho_clv_volumen': float(df_train['customer_lifetime_value'].corr(df_train['total_tx_volume'], method='spearman')),
        'pares_duplicados': pares,
        'r_satisfaccion_nps': float(df_train['satisfaction_score'].corr(df_train['nps_score'])),
        'eta2': eta,
        'faltantes': falt,
        'contingencia_credito': contingencia,
        'ausencia_vs_objetivo': ausencia,
        'booleanas_vs_active': booleanas,
        'corr_tx': corr_tx,
    }


# ------------------------------------------------------------------------- depuración
def depurar(df: pd.DataFrame, columnas_a_eliminar: list[str] | None = None) -> pd.DataFrame:
    """Elimina las columnas descartadas e imputa el faltante estructural.

    Parameters
    ----------
    df : DataFrame
        Tabla unida.
    columnas_a_eliminar : list of str, optional
        Por defecto ``COLUMNAS_A_ELIMINAR``, decidida en la Entrega 2 y re-validada aquí
        con :func:`diagnosticos_objetivo` sobre entrenamiento.

    Returns
    -------
    DataFrame
        Copia sin esas columnas y sin faltantes.

    Notes
    -----
    ``credit_utilization_ratio`` falta exactamente cuando no hay tarjeta de crédito: se
    imputa 0 por definición del dominio, no por un estadístico de los datos, y por eso
    puede aplicarse a todas las filas sin fuga.
    """
    datos = df.drop(columns=columnas_a_eliminar or COLUMNAS_A_ELIMINAR).copy()
    datos['credit_utilization_ratio'] = datos['credit_utilization_ratio'].fillna(0)
    return datos


@dataclass
class Conjuntos:
    """Matrices de predictores y objetivo continuo por partición."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_cont: pd.Series
    y_test_cont: pd.Series
    X_val: pd.DataFrame | None = None
    y_val_cont: pd.Series | None = None


def separar(datos: pd.DataFrame, particion: Particion, objetivo: str = OBJETIVO) -> Conjuntos:
    """Aplica la partición a la tabla depurada y separa predictores y objetivo continuo.

    Parameters
    ----------
    datos : DataFrame
        Salida de :func:`depurar`.
    particion : Particion
    objetivo : str

    Returns
    -------
    Conjuntos
    """
    X = datos.drop(columns=[objetivo])
    y = datos[objetivo]
    c = Conjuntos(X.loc[particion.idx_train], X.loc[particion.idx_test],
                  y.loc[particion.idx_train], y.loc[particion.idx_test])
    if particion.idx_val is not None:
        c.X_val, c.y_val_cont = X.loc[particion.idx_val], y.loc[particion.idx_val]
    return c


# ------------------------------------------------------------------------------ etiqueta
@dataclass
class Etiqueta:
    """Umbral de binarización y su procedencia.

    Attributes
    ----------
    umbral : float
        Valor de corte; ``y = 1`` si el objetivo continuo lo supera.
    metodo : str
        ``'mediana'`` o ``'percentil'``.
    q : float or None
        Percentil usado cuando ``metodo == 'percentil'``.
    n_train : int
        Número de observaciones de entrenamiento con las que se estimó.
    """
    umbral: float
    metodo: str
    q: float | None
    n_train: int

    def binarizar(self, y_cont: pd.Series) -> pd.Series:
        """Aplica el umbral, ya fijo, a cualquier conjunto.

        Parameters
        ----------
        y_cont : Series
            Objetivo continuo.

        Returns
        -------
        Series
            ``riesgo_alto`` (1 si ``y_cont > umbral``).
        """
        return (y_cont > self.umbral).astype(int).rename('riesgo_alto')


def construir_etiqueta(y_train_cont: pd.Series, metodo: str = ETIQUETA_METODO,
                       q: float | None = ETIQUETA_Q) -> Etiqueta:
    """Estima el umbral de binarización **solo con entrenamiento**.

    Parameters
    ----------
    y_train_cont : Series
        Objetivo continuo de las filas de entrenamiento.
    metodo : {'percentil', 'mediana'}
        Con ``'percentil'`` (por defecto, ``config.ETIQUETA_METODO``) se marca como riesgo
        alto la fracción ``1 - q`` superior (``q = 0.75`` → 25 % de positivos), lo que
        produce un desbalanceo real sobre el que tienen sentido las técnicas de balanceo
        y la calibración. Con ``'mediana'`` las clases quedan ~50/50 por construcción y
        el sobremuestreo no tiene nada que generar.
    q : float, optional
        Percentil en (0, 1) cuando ``metodo == 'percentil'`` (por defecto
        ``config.ETIQUETA_Q``); se ignora con ``'mediana'``.

    Returns
    -------
    Etiqueta
    """
    if metodo == 'mediana':
        umbral = float(y_train_cont.median())
        q = None
    elif metodo == 'percentil':
        if q is None or not 0 < q < 1:
            raise ValueError("con metodo='percentil' hay que dar q en (0, 1)")
        umbral = float(y_train_cont.quantile(q))
    else:
        raise ValueError(f'metodo desconocido: {metodo!r}')
    return Etiqueta(umbral=umbral, metodo=metodo, q=q, n_train=int(len(y_train_cont)))


def preparar_todo(carpeta: Path | str | None = None, test_size: float = TEST_SIZE,
                  val_size: float = 0.0, metodo_etiqueta: str = ETIQUETA_METODO,
                  q: float | None = ETIQUETA_Q, seed: int = SEED) -> dict:
    """Ejecuta el flujo completo en el orden correcto y devuelve todas las piezas.

    Parameters
    ----------
    carpeta : Path or str, optional
        Carpeta de los CSV; por defecto se localiza automáticamente.
    test_size, val_size : float
        Fracciones de prueba y de validación explícita (0 para no crearla).
    metodo_etiqueta : {'percentil', 'mediana'}
    q : float, optional
        Percentil del umbral (ver :func:`construir_etiqueta`).
    seed : int

    Returns
    -------
    dict
        ``clientes``, ``tx``, ``auditoria``, ``df`` (tabla unida), ``particion``,
        ``diagnosticos`` (calculados sobre entrenamiento), ``datos`` (depurada),
        ``conjuntos``, ``etiqueta``, ``y_train``, ``y_test`` (y ``y_val`` si aplica).
    """
    clientes, tx = cargar_tablas(carpeta)
    auditoria = auditar_transacciones(clientes, tx)
    tx = limpiar_transacciones(tx)
    df = unir(clientes, derivar_variables_transaccionales(tx))
    particion = partir(df, test_size=test_size, val_size=val_size, seed=seed)
    diagnosticos = diagnosticos_objetivo(df, particion)
    datos = depurar(df)
    conjuntos = separar(datos, particion)
    etiqueta = construir_etiqueta(conjuntos.y_train_cont, metodo=metodo_etiqueta, q=q)
    salida = {
        'clientes': clientes, 'tx': tx, 'auditoria': auditoria, 'df': df,
        'particion': particion, 'diagnosticos': diagnosticos, 'datos': datos,
        'conjuntos': conjuntos, 'etiqueta': etiqueta,
        'y_train': etiqueta.binarizar(conjuntos.y_train_cont),
        'y_test': etiqueta.binarizar(conjuntos.y_test_cont),
    }
    if conjuntos.y_val_cont is not None:
        salida['y_val'] = etiqueta.binarizar(conjuntos.y_val_cont)
    return salida
