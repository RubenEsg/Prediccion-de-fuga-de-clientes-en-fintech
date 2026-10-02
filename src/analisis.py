"""Lectura de la tabla maestra y evaluación en prueba de los modelos finales (guía §5).

Las funciones devuelven tablas listas para mostrarse en el libro. La prueba se toca aquí por
primera vez para los modelos de la tabla maestra: cada modelo final (reajustado sobre todo el
entrenamiento con sus hiperparámetros finales) se evalúa **una sola vez**, y sus puntajes se
guardan en caché para que la calibración, los contrastes y las gráficas no tengan que volver a
predecir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from . import evaluacion
from .config import CARPETA_RESULTADOS
from .registro import Registro

CLAVE: list[str] = ['tarea', 'modelo', 'balanceo', 'optimizador']
"""Columnas que identifican una corrida (la semilla es única en este proyecto)."""

NOMBRES_MODELO: dict[str, str] = {
    'knn': 'k-NN', 'naive_bayes': 'Naive Bayes', 'logistica': 'Regresión logística',
    'arbol': 'Árbol de decisión', 'random_forest': 'Random Forest', 'xgboost': 'XGBoost',
    'svm': 'SVM lineal', 'ridge': 'Ridge', 'lasso': 'Lasso', 'svr': 'SVR lineal',
    'ebm': 'EBM'}
"""Nombre legible de cada modelo para tablas y gráficas."""


def clave(fila) -> str:
    """Identificador ``tarea__modelo__balanceo__optimizador`` de una fila de la tabla."""
    return '__'.join(str(fila[c]) for c in CLAVE)


def cargar_tabla(ruta: Path | str | None = None) -> pd.DataFrame:
    """Tabla maestra con las columnas JSON decodificadas y la métrica en sentido natural.

    Parameters
    ----------
    ruta : Path or str, optional
        Por defecto ``resultados/experimentos.parquet``.

    Returns
    -------
    DataFrame
        La tabla con ``clave`` y ``metrica``: AUC media externa en clasificación y RMSE medio
        externo (positivo) en regresión; ``NaN`` en las combinaciones que no aplican.
    """
    df = Registro(Path(ruta) if ruta is not None else CARPETA_RESULTADOS / 'experimentos.parquet').cargar(
        decodificar=True)
    df['clave'] = df.apply(clave, axis=1)
    media = pd.to_numeric(df['media_ext'], errors='coerce')
    df['metrica'] = np.where(df['tarea'] == 'regresion', -media, media)
    return df


def mejores_por_modelo(tabla: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """La mejor combinación (balanceo × optimizador) de cada modelo según el bucle externo.

    Parameters
    ----------
    tabla : DataFrame
        Salida de :func:`cargar_tabla`.
    tarea : {'clasificacion', 'regresion'}

    Returns
    -------
    DataFrame
        Una fila por modelo, ordenada de mejor a peor (``media_ext`` es *mayor es mejor* en las
        dos tareas). Es la selección que se evalúa en prueba: se decide sin mirarla.
    """
    t = tabla[(tabla['tarea'] == tarea) & (tabla['estado'] == 'ok')]
    t = t.loc[t.groupby('modelo')['media_ext'].idxmax()]
    return t.sort_values('media_ext', ascending=False).reset_index(drop=True)


def predecir_prueba(tabla: pd.DataFrame, X_test: pd.DataFrame, ruta_cache: Path | str,
                    informar: Callable[[str], None] = print) -> pd.DataFrame:
    """Puntajes de prueba de todos los modelos finales, con caché en disco.

    Parameters
    ----------
    tabla : DataFrame
        Salida de :func:`cargar_tabla`; se usan las filas ``ok`` y su ``ruta_modelo``.
    X_test : DataFrame
        Predictores de prueba (el índice se conserva en la salida).
    ruta_cache : Path or str
        Parquet con una columna por clave. Las claves ya presentes no se recalculan.
    informar : callable

    Returns
    -------
    DataFrame
        Columna ``<clave>``: probabilidad de la clase positiva (o función de decisión si el
        modelo no da probabilidades, p. ej. el SVM lineal) en clasificación y pronóstico en
        regresión. Columna ``<clave>__clase``: la clase que predice el modelo.
    """
    ruta_cache = Path(ruta_cache)
    cache = pd.read_parquet(ruta_cache) if ruta_cache.exists() else pd.DataFrame(index=X_test.index)
    if not cache.index.equals(X_test.index):
        raise ValueError('la caché de predicciones no corresponde a estas filas de prueba')
    pendientes = [f for _, f in tabla[tabla['estado'] == 'ok'].iterrows() if f['clave'] not in cache]
    for i, f in enumerate(pendientes, 1):
        modelo = joblib.load(f['ruta_modelo'])
        if f['tarea'] == 'clasificacion':
            pred, puntaje, _ = evaluacion.puntajes(modelo, X_test)
            cache[f['clave']] = puntaje
            cache[f['clave'] + '__clase'] = pred
        else:
            cache[f['clave']] = modelo.predict(X_test)
        cache.to_parquet(ruta_cache)
        informar(f'[{i}/{len(pendientes)}] {f["clave"]}')
    return cache


def es_probabilidad(modelo: str) -> bool:
    """El puntaje guardado del modelo es una probabilidad (todos menos el SVM lineal)."""
    return modelo != 'svm'


def metricas_prueba(tabla: pd.DataFrame, predicciones: pd.DataFrame, y_bin: pd.Series,
                    y_cont: pd.Series) -> pd.DataFrame:
    """Métricas de prueba de cada modelo final junto a su métrica de validación externa.

    Parameters
    ----------
    tabla : DataFrame
        Salida de :func:`cargar_tabla`.
    predicciones : DataFrame
        Salida de :func:`predecir_prueba`.
    y_bin, y_cont : Series
        Etiqueta y objetivo continuo de prueba, con el índice de ``predicciones``.

    Returns
    -------
    DataFrame
        Clave de la corrida, ``metrica_cv`` (media del bucle externo), ``desv_cv`` y las
        métricas de prueba de la tarea (``accuracy`` … ``brier`` o ``rmse``, ``mae``, ``r2``).
    """
    filas = []
    for _, f in tabla[tabla['estado'] == 'ok'].iterrows():
        k = f['clave']
        if k not in predicciones:
            continue
        base = {c: f[c] for c in CLAVE} | {'metrica_cv': f['metrica'], 'desv_cv': f['desv_ext']}
        if f['tarea'] == 'clasificacion':
            m = evaluacion.metricas_clasificacion(y_bin.to_numpy(), predicciones[k + '__clase'].to_numpy(),
                                                  predicciones[k].to_numpy(), es_probabilidad(f['modelo']))
        else:
            m = evaluacion.metricas_regresion(y_cont.to_numpy(), predicciones[k].to_numpy())
        filas.append(base | m)
    return pd.DataFrame(filas)


def forma_ebm(pipeline, variable: str) -> pd.DataFrame:
    """Función de forma de la EBM para una variable numérica, en sus unidades originales.

    Parameters
    ----------
    pipeline : Pipeline
        Ajustado, con pasos ``preprocesado`` (escalado + indicadoras) y ``modelo`` (EBM, o
        :class:`modelos.EBMConPesos`, que guarda la EBM en ``ebm_``).
    variable : str
        Nombre de una columna numérica original.

    Returns
    -------
    DataFrame
        Un tramo por fila: ``desde`` y ``hasta`` (en las unidades de la variable), ``efecto``
        (contribución al logit en clasificación o al pronóstico en regresión, centrada en la media
        del entrenamiento) e ``inf``/``sup`` (banda entre bolsas externas).

    Notes
    -----
    Los cortes de la EBM se aprenden sobre la variable estandarizada; se devuelven a la escala
    original con la media y la desviación del ``StandardScaler`` del pipeline.
    """
    from .preprocesamiento import nombres_variables
    pre = pipeline.named_steps['preprocesado']
    ebm = pipeline.named_steps['modelo']
    ebm = getattr(ebm, 'ebm_', ebm)
    j = nombres_variables(pre).index(variable)
    termino = next(i for i, f in enumerate(ebm.term_features_) if tuple(f) == (j,))
    d = ebm.explain_global().data(termino)
    bordes = np.asarray(d['names'], dtype=float)
    escalador = pre.named_transformers_['num']
    k = list(pre.transformers_[0][2]).index(variable)
    x = bordes * escalador.scale_[k] + escalador.mean_[k]
    return pd.DataFrame({'desde': x[:-1], 'hasta': x[1:], 'efecto': np.asarray(d['scores'], dtype=float),
                         'inf': np.asarray(d['lower_bounds'], dtype=float),
                         'sup': np.asarray(d['upper_bounds'], dtype=float)})


def efecto_lineal(pipeline, variable: str, x: np.ndarray) -> np.ndarray:
    """Contribución de una variable numérica en un modelo lineal del pipeline, en ``x`` original.

    Es ``β · (x − media) / desviación``: la misma escala (logit o pronóstico) y el mismo centrado
    que la función de forma de la EBM, para compararlas en una gráfica.
    """
    from .preprocesamiento import nombres_variables
    pre = pipeline.named_steps['preprocesado']
    coef = np.ravel(pipeline.named_steps['modelo'].coef_)
    j = nombres_variables(pre).index(variable)
    escalador = pre.named_transformers_['num']
    k = list(pre.transformers_[0][2]).index(variable)
    return coef[j] * (np.asarray(x, dtype=float) - escalador.mean_[k]) / escalador.scale_[k]


# ------------------------------------------------------------- optimizadores (§3.4)
OPTIMIZADORES: list[str] = ['grid', 'random', 'bayesiana', 'genetica']


def _bloques_completos(tabla: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """Filas ``ok`` de la tarea cuyos bloques (modelo × balanceo) tienen los 4 optimizadores."""
    ok = tabla[(tabla['tarea'] == tarea) & (tabla['estado'] == 'ok')]
    n = ok.groupby(['modelo', 'balanceo'])['optimizador'].transform('nunique')
    return ok[n == len(OPTIMIZADORES)]


def resumen_optimizadores(tabla: pd.DataFrame) -> pd.DataFrame:
    """Comparación agregada de los cuatro optimizadores por tarea (guía §3.4).

    Parameters
    ----------
    tabla : DataFrame
        Salida de :func:`cargar_tabla`.

    Returns
    -------
    DataFrame
        Índice (tarea, optimizador) con: ``corridas``; ``rango_medio`` y ``victorias`` dentro de
        cada bloque modelo × balanceo (rango 1 = mejor métrica externa; los empates reparten);
        ``metrica_media`` (AUC o RMSE externos); ``evaluaciones`` por corrida;
        ``s_por_evaluacion`` (tiempo de búsqueda entre evaluaciones); ``min_por_corrida``;
        ``horas_totales``; ``optimismo`` medio (interno − externo, en la escala *mayor es mejor*).
    """
    partes = []
    for tarea in ('clasificacion', 'regresion'):
        ok = _bloques_completos(tabla, tarea).copy()
        if ok.empty:
            continue
        ok['rango'] = ok.groupby(['modelo', 'balanceo'])['media_ext'].rank(ascending=False)
        ok['victoria'] = ok.groupby(['modelo', 'balanceo'])['media_ext'].rank(ascending=False, method='min') == 1
        g = ok.groupby('optimizador').agg(
            corridas=('clave', 'size'), rango_medio=('rango', 'mean'), victorias=('victoria', 'sum'),
            metrica_media=('metrica', 'mean'), evaluaciones=('n_evaluaciones', 'mean'),
            s_por_evaluacion=('tiempo_por_evaluacion_s', 'mean'),
            min_por_corrida=('tiempo_total_s', lambda s: s.mean() / 60),
            horas_totales=('tiempo_total_s', lambda s: s.sum() / 3600),
            optimismo=('optimismo_seleccion', 'mean'))
        partes.append(g.reindex(OPTIMIZADORES).assign(tarea=tarea).set_index('tarea', append=True)
                      .reorder_levels(['tarea', 'optimizador']))
    return pd.concat(partes)


def matriz_optimizadores(tabla: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """Métrica externa (*mayor es mejor*) por bloque modelo × balanceo y optimizador.

    Es la entrada de :func:`estadistica.friedman` para contrastar si los optimizadores difieren.
    """
    ok = _bloques_completos(tabla, tarea)
    return ok.pivot_table(index=['modelo', 'balanceo'], columns='optimizador',
                          values='media_ext')[OPTIMIZADORES]


def diferencias_con_grid(tabla: pd.DataFrame) -> pd.DataFrame:
    """Diferencia de métrica externa de Random, Bayesiana y Genética frente a Grid, por bloque.

    Returns
    -------
    DataFrame
        Por tarea y optimizador: media, mínimo y máximo de la diferencia (positiva = mejor que
        Grid; en AUC o en RMSE con el signo cambiado), fracción de bloques en que supera a Grid y
        p-valor de Wilcoxon de rangos con signo.
    """
    from scipy.stats import wilcoxon

    filas = []
    for tarea in ('clasificacion', 'regresion'):
        m = matriz_optimizadores(tabla, tarea)
        for o in OPTIMIZADORES[1:]:
            d = (m[o] - m['grid']).to_numpy()
            filas.append({'tarea': tarea, 'optimizador': o, 'bloques': len(d), 'media': d.mean(),
                          'minimo': d.min(), 'maximo': d.max(), 'supera_a_grid': float(np.mean(d > 0)),
                          'p_wilcoxon': float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
    return pd.DataFrame(filas).set_index(['tarea', 'optimizador'])


def curvas_anytime(tabla: pd.DataFrame, tarea: str, puntos: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arrepentimiento normalizado de cada optimizador en función de la fracción de presupuesto.

    Parameters
    ----------
    tabla : DataFrame
    tarea : {'clasificacion', 'regresion'}
    puntos : int
        Resolución de la rejilla común de fracciones de presupuesto.

    Returns
    -------
    curvas : DataFrame
        ``optimizador``, ``fraccion``, ``media``, ``q25`` y ``q75`` del arrepentimiento.
    por_busqueda : DataFrame
        Una fila por búsqueda (bloque × pliegue externo × optimizador): ``area`` bajo la curva de
        arrepentimiento (menor es mejor), ``final`` (arrepentimiento al agotar el presupuesto) y
        ``evals_al_95`` (evaluaciones hasta cerrar el 95 % de la brecha, NaN si no lo logra).

    Notes
    -----
    Para cada bloque modelo × balanceo y cada pliegue externo, ``mejor`` es el mejor puntaje
    interno que alcanzó cualquiera de los cuatro optimizadores y ``peor`` el menor primer
    puntaje. El arrepentimiento tras *t* evaluaciones es ``(mejor − traza_t)/(mejor − peor)``:
    0 si ya encontró lo mejor que se encontró en ese pliegue, 1 si no mejoró desde el peor punto
    de partida. Normalizar así permite promediar modelos con escalas de métrica distintas.
    """
    ok = _bloques_completos(tabla, tarea)
    rejilla = np.linspace(1 / puntos, 1.0, puntos)
    filas, resumen = [], []
    for (modelo, balanceo), g in ok.groupby(['modelo', 'balanceo']):
        trazas_por_opt = {o: g.loc[g['optimizador'] == o, 'trazas'].iloc[0] for o in OPTIMIZADORES}
        for f in range(len(trazas_por_opt['grid'])):
            tr = {o: np.asarray(trazas_por_opt[o][f], dtype=float) for o in OPTIMIZADORES}
            finitos = {o: v[np.isfinite(v)] for o, v in tr.items()}
            mejor = max(v.max() for v in finitos.values())
            peor = min(v[0] for v in finitos.values())
            if mejor - peor <= 0:
                continue
            for o, v in tr.items():
                v = np.where(np.isfinite(v), v, peor)
                arrep = (mejor - v) / (mejor - peor)
                idx = np.minimum(np.ceil(rejilla * len(v)).astype(int) - 1, len(v) - 1)
                filas += [{'optimizador': o, 'fraccion': fr, 'arrepentimiento': arrep[i]}
                          for fr, i in zip(rejilla, idx)]
                alcanzado = np.flatnonzero(arrep <= 0.05)
                resumen.append({'modelo': modelo, 'balanceo': balanceo, 'pliegue': f, 'optimizador': o,
                                'presupuesto': len(v), 'area': float(arrep.mean()), 'final': float(arrep[-1]),
                                'evals_al_95': float(alcanzado[0] + 1) if len(alcanzado) else np.nan})
    df = pd.DataFrame(filas)
    curvas = df.groupby(['optimizador', 'fraccion'])['arrepentimiento'].agg(
        media='mean', q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
    return curvas, pd.DataFrame(resumen)


def diversidad_genetica(tabla: pd.DataFrame) -> pd.DataFrame:
    """Diversidad, mejor aptitud y aptitud media por generación de cada búsqueda genética.

    Returns
    -------
    DataFrame
        ``tarea``, ``modelo``, ``balanceo``, ``pliegue`` (``'final'`` para la búsqueda sobre todo
        el entrenamiento), ``generacion``, ``diversidad`` (desviación típica media de los genes en
        [0, 1]), ``mejor`` y ``media`` (puntaje interno).
    """
    filas = []
    ok = tabla[(tabla['estado'] == 'ok') & (tabla['optimizador'] == 'genetica')]
    for _, f in ok.iterrows():
        detalles = list(enumerate(f['detalles'])) + [('final', f['detalle'])]
        for pliegue, d in detalles:
            for gen, (div, mej, med) in enumerate(zip(d['diversidad_por_generacion'], d['mejor_por_generacion'],
                                                       d['media_por_generacion'])):
                filas.append({'tarea': f['tarea'], 'modelo': f['modelo'], 'balanceo': f['balanceo'],
                              'pliegue': pliegue, 'generacion': gen, 'diversidad': div, 'mejor': mej,
                              'media': med})
    return pd.DataFrame(filas)
