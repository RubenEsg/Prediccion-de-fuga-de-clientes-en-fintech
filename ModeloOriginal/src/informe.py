"""Tablas, figuras y textos del informe del proyecto final, generados a partir de los resultados.

Cada función produce una pieza (una o varias tablas en ``ProyectoFinal/_generado``, o una figura en
``ProyectoFinal/figuras``) leyendo los archivos de ``ModeloOriginal/resultados``: ninguna cifra del informe se escribe
a mano. Hay dos formas de usarlo:

- desde los cuadernos del libro, con :func:`tabla`, :func:`texto` y :func:`figura`, que regeneran la pieza y la
  muestran en la celda;
- desde la línea de órdenes, con ``python experiments/08_tablas_libro.py``, que las regenera todas.

Una pieza a la que todavía le falta un resultado avisa y no interrumpe a las demás.
"""
import json
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')                                                     # las figuras se guardan en disco y se muestran como imagen
import matplotlib.pyplot as plt                                            # noqa: E402
import matplotlib.ticker                                                   # noqa: E402

from . import conjuntos, plan                                              # noqa: E402
from .config import CARPETA_RESULTADOS, SEED                               # noqa: E402
from .registro import resolver                                             # noqa: E402

warnings.filterwarnings('ignore')
VERBOSO = True                                                            # los cuadernos lo apagan: solo interesa la pieza

LIBRO = plan.RAIZ.parent / 'ProyectoFinal'
GEN, FIG = LIBRO / '_generado', LIBRO / 'figuras'
R = CARPETA_RESULTADOS
E3 = plan.RAIZ.parent / 'Entrega3' / 'resultados'
AZUL, NARANJA, GRIS, VERDE = '#1f4e79', '#d9822b', '#8a8f98', '#2e7d5b'
BASES = {'openalex': 'OpenAlex', 'arxiv': 'arXiv', 'crossref': 'Crossref'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130,
                     'axes.grid': True, 'grid.alpha': 0.25, 'savefig.bbox': 'tight'})


class _FormatoComa(matplotlib.ticker.ScalarFormatter):
    """El formato numérico por defecto de los ejes, con coma decimal como en el texto del informe."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', ',')

    def get_offset(self):
        return super().get_offset().replace('.', ',')


def _guardar(fig, nombre: str) -> None:
    """Guarda la figura en ``ProyectoFinal/figuras`` con coma decimal en los ejes numéricos que usan el formato por defecto."""
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            if type(eje.get_major_formatter()) is matplotlib.ticker.ScalarFormatter:
                eje.set_major_formatter(_FormatoComa())
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / nombre)


MENOS = '−'                                                               # signo menos tipográfico
SUPERINDICES = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def num(v, d=4) -> str:
    """Número con coma decimal y signo menos tipográfico."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{d}f}'.replace('.', ',').replace('-', MENOS)


def entero(v) -> str:
    return f'{int(round(float(v))):,}'.replace(',', '.').replace('-', MENOS)


def estad(v) -> str:
    """Estadístico de una prueba: entero si lo es y notación científica si es muy pequeño."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    v = float(v)
    if v == int(v):
        return entero(v)
    if abs(v) < 0.001:
        mantisa, exponente = f'{v:.2e}'.split('e')
        return f'{mantisa.replace(".", ",").replace("-", MENOS)} × 10{str(int(exponente)).translate(SUPERINDICES)}'
    return num(v, 4 if abs(v) < 100 else 1)


def pvalor(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '—'
    return '< 0,001' if p < 0.001 else num(p, 3)


def tabla_md(encabezado: list[str], filas: list[list], alineacion: str | None = None) -> str:
    alineacion = alineacion or 'l' + 'r' * (len(encabezado) - 1)
    marcas = {'l': ':---', 'r': '---:', 'c': ':---:'}
    lineas = ['| ' + ' | '.join(encabezado) + ' |', '| ' + ' | '.join(marcas[a] for a in alineacion) + ' |']
    lineas += ['| ' + ' | '.join(str(c) for c in f) + ' |' for f in filas]
    return '\n'.join(lineas) + '\n'


def _aviso(mensaje: str) -> None:
    if VERBOSO:
        print(mensaje)


def escribir(nombre: str, texto: str) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / nombre).write_text(texto, encoding='utf-8', newline='\n')
    _aviso('  ' + nombre)


@lru_cache(maxsize=1)
def _problema() -> dict:
    """Los datos del problema principal, preparados una sola vez para todas las piezas."""
    return plan.problema()


def pieza(funcion):
    """Ejecuta una función generadora y avisa si le falta un insumo."""
    try:
        funcion()
    except FileNotFoundError as e:
        _aviso(f'   [pendiente] {funcion.__name__}: falta {Path(str(e.filename)).name}')
    except Exception as e:                                               # noqa: BLE001
        _aviso(f'   [error] {funcion.__name__}: {type(e).__name__}: {e}')


# ------------------------------------------------------------------ capítulo 4: literatura
MESES = ('enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre')
ALGORITMICOS = ('Hou', 'Zhou', 'Xue', 'Jobst', 'März', 'Hajjem', 'Sela', 'Sigrist')   # los antecedentes algorítmicos más cercanos


def _fecha_larga(iso: str) -> str:
    """'2026-11-18' → '18 de noviembre de 2026'."""
    anio, mes, dia = (int(x) for x in iso.split('-'))
    return f'{dia} de {MESES[mes - 1]} de {anio}'


def _recortar(texto: str, largo: int) -> str:
    """Corta un título en un espacio y lo marca con puntos suspensivos."""
    texto = ' '.join(str(texto).split())
    return texto if len(texto) <= largo else texto[:largo].rsplit(' ', 1)[0].rstrip(' ,;:') + '…'


def literatura() -> None:
    carpeta = R / 'literatura'
    d = json.loads((carpeta / 'busqueda.json').read_text(encoding='utf-8'))
    rel = pd.read_csv(plan.RAIZ / 'experiments' / 'literatura_relevancia.csv').set_index(['base', 'cadena'])
    anteriores = {f: json.loads((carpeta / f'busqueda_{f}.json').read_text(encoding='utf-8'))
                  for f in d.get('consultas_anteriores', []) if (carpeta / f'busqueda_{f}.json').exists()}
    filas, nuevas, entradas_nuevas, celdas_comunes = [], [], 0, 0
    for clave, c in d['cadenas'].items():
        previas = [a['cadenas'][clave] for a in anteriores.values() if clave in a['cadenas']]
        if anteriores and not previas:
            nuevas.append(clave)
        for base in BASES:
            r = c[base]
            filas.append([BASES[base], f'{clave}. {c["tema"]}', entero(r['total']) if r.get('total') is not None else '—',
                          min(len(r.get('primeros', [])), d['revisados_por_cadena']), int(rel.loc[(base, clave), 'relevantes']),
                          rel.loc[(base, clave), 'observacion']])
            for previa in previas:
                if previa.get(base, {}).get('total') is None:                     # esa consulta falló: no hay con qué comparar
                    continue
                antes = {(x.get('titulo') or '').strip().lower() for x in previa[base].get('primeros', [])}
                entradas_nuevas += sum((x.get('titulo') or '').strip().lower() not in antes for x in r.get('primeros', []))
                celdas_comunes += 1
    cabecera = f'**Consulta del {_fecha_larga(d["fecha_de_consulta"])}.**'
    if anteriores:
        comunes = [k for k in d['cadenas'] if k not in nuevas]
        cabecera += (f' Es la repetición que pide la guía: la primera consulta fue el {_fecha_larga(min(anteriores))}, con las cadenas '
                     f'{comunes[0]} a {comunes[-1]}')
        if nuevas:
            cabecera += f'; las cadenas {nuevas[0]} a {nuevas[-1]} se añadieron en esta'
        cabecera += (f'. Entre las dos fechas, en las {celdas_comunes} combinaciones de base y cadena que se pueden comparar entraron '
                     f'{entradas_nuevas} resultados que no estaban entre los primeros {d["revisados_por_cadena"]}; se leyeron y no cambian '
                     f'ninguna observación.')
    escribir('tabla_matriz.md', cabecera + '\n\n' + tabla_md(['Base', 'Cadena', 'Result.', 'Revis.', 'Relev.', 'Observación'], filas, 'llrrrl')
             + f'\nResult.: registros que devuelve la base. Revis.: los primeros por relevancia, cuyo título y resumen se leyeron (hasta '
               f'{d["revisados_por_cadena"]}; Crossref no da resumen y su búsqueda no admite operadores, así que su total cuenta todo registro '
               f'con alguno de los términos). Relev.: de los revisados, los trabajos cercanos, es decir, los que comparten con la propuesta la '
               f'familia del modelo o el problema con un componente latente o probabilístico; son los que nombra la observación. Un trabajo '
               f'cercano no es una coincidencia: ninguno aplica el modelo propuesto a la fuga de clientes.\n')
    escribir('tabla_cadenas.md', tabla_md(['Cadena', 'OpenAlex', 'arXiv', 'Crossref'],
                                          [[clave] + [f'`{c[base]["cadena"]}`' for base in BASES] for clave, c in d['cadenas'].items()], 'llll'))
    citacion = {n: s for n, s in d['citacion'].items() if 'citado_por' in s}
    adelante = [[n, entero(s['citado_por']), s['citan_y_tratan_fuga'],
                 '; '.join(f'{_recortar(w["titulo"], 90)} ({w["anio"]})' for w in s['primeros'][:2]) or '—'] for n, s in citacion.items()]
    cercanos = {n: s for n, s in citacion.items() if n.startswith(ALGORITMICOS)}
    atras = [[n, s['referencias'], s['sobre_mezclas_o_clases_latentes'], s['sobre_fuga'], s['que_combinan_mezcla_y_fuga']]
             for n, s in d['citacion_hacia_atras'].items() if 'referencias' in s]
    repos = [[f'`{q}`', entero(r['total']), _recortar(f'{r["primeros"][0]["nombre"]}: {r["primeros"][0]["descripcion"]}', 110) if r.get('primeros') else '—']
             for q, r in d.get('repositorios', {}).items() if r.get('total') is not None]
    texto = ('**Hacia adelante: trabajos que citan a cada antecedente y mencionan «churn» en el título o el resumen.**\n\n'
             + tabla_md(['Antecedente', 'Citado por', 'De ellos, mencionan «churn»', 'Primeros'], adelante, 'lrrl')
             + '\nLa cuenta es automática (OpenAlex) y cada resultado se leyó: no todos tratan de fuga de clientes (la palabra también '
               'nombra la rotación de personal).')
    if cercanos:
        texto += (f' De los {entero(sum(s["citado_por"] for s in cercanos.values()))} trabajos que citan a los {len(cercanos)} antecedentes '
                  f'algorítmicos más cercanos (las {len(cercanos)} últimas filas), mencionan la fuga '
                  f'{sum(s["citan_y_tratan_fuga"] for s in cercanos.values())}.')
    texto += ('\n\n**Hacia atrás: referencias de los trabajos más cercanos.**\n\n'
              + tabla_md(['Trabajo cercano', 'Referencias', 'Sobre mezclas o clases latentes', 'Sobre fuga', 'Mezcla y fuga a la vez'], atras))
    if repos:
        texto += ('\n**Repositorios de código: consultas a GitHub, por número de estrellas.**\n\n'
                  + tabla_md(['Consulta', 'Repositorios', 'El primero'], repos, 'lrl'))
    escribir('tabla_citacion.md', texto)


# ------------------------------------------------------------------ capítulo 2: ETL y EDA
def datos_eda() -> None:
    """ETL y EDA del capítulo 2, a partir de los resúmenes del entrenamiento que usa el dashboard (``Dashboard/datos/eda``)."""
    eda = plan.RAIZ.parent / 'Dashboard' / 'datos' / 'eda'
    e = json.loads((eda / 'etl.json').read_text(encoding='utf-8'))
    a = e['auditoria_transacciones']
    tipos = e['tipos']
    codigo = lambda nombres: ', '.join(f'`{x}`' for x in nombres)          # noqa: E731
    filas = [
        ['1. Carga', f'{entero(e["n_clientes"])} clientes con {e["n_columnas"]} columnas; el detalle de transacciones, en una tabla aparte'],
        ['2. Auditoría de las transacciones',
         f'{entero(a["duplicadas"])} filas exactamente duplicadas, que se eliminan (quedan {entero(e["n_transacciones_limpias"])}); '
         f'{a["montos_no_positivos"]} montos no positivos, {a["fechas_nulas"]} fechas nulas, {a["huerfanas"]} transacciones sin cliente y '
         f'{a["clientes_sin_tx"]} clientes sin transacciones'],
        ['3. Partición', 'Entrenamiento y prueba (80/20, semilla 42), antes de mirar el objetivo y antes de derivar variables'],
        ['4. Variables derivadas', f'{len(e["derivadas"])}, del detalle de transacciones y sin usar el objetivo: {codigo(e["derivadas"])}'],
        ['5. Columnas eliminadas', f'{len(e["eliminadas"])}: {codigo(e["eliminadas"])}'],
        ['6. Predictores', f'{len(e["predictores"])}: {len(tipos["numericas"])} numéricos, {len(tipos["booleanas"])} booleanos y '
                           f'{len(tipos["categoricas"])} categóricos'],
    ]
    escribir('tabla_etl.md', tabla_md(['Paso', 'Resultado'], filas, 'll'))
    c = pd.read_csv(eda / 'cohortes.csv')
    c = c[c.conjunto == 'entrenamiento']
    escribir('tabla_cohortes.md', tabla_md(['Cohorte de edad', 'Clientes', 'Puntaje medio', 'Riesgo alto'],
                                           [[f'{f.cohorte} años', entero(f.n), num(f.media_objetivo, 3), f'{num(100 * f.tasa_riesgo_alto, 1)} %']
                                            for f in c.itertuples()], 'lrrr')
             + f'\nEl riesgo alto va del {num(100 * c.tasa_riesgo_alto.min(), 1)} % al {num(100 * c.tasa_riesgo_alto.max(), 1)} % según la cohorte, '
               'y no crece con la edad: la relación no es monótona.\n')
    s = pd.read_csv(eda / 'asociacion.csv')
    s['fuerza'] = s['valor'].abs()
    numericas = s[s.medida == 'spearman'].sort_values('fuerza', ascending=False)
    categoricas = s[s.medida == 'eta2'].sort_values('fuerza', ascending=False)
    primeras = numericas.head(6)
    filas = [[f'`{f.variable}`', 'ρ de Spearman', num(f.valor, 3), num(f.pearson, 3),
              'Excluida: genera el puntaje' if f.variable in e['eliminadas'] else 'Predictor'] for f in primeras.itertuples()]
    filas += [[f'`{f.variable}`', 'η²', num(f.valor, 4), '—', 'Excluida' if f.variable in e['eliminadas'] else 'Predictor']
              for f in categoricas.head(2).itertuples()]
    en_modelo = numericas[~numericas.variable.isin(e['eliminadas'])]
    debiles = en_modelo[en_modelo.fuerza < 0.02]
    escribir('tabla_asociacion.md', tabla_md(['Variable', 'Medida', 'Asociación con el puntaje', 'r de Pearson', 'Uso'], filas, 'llrrl')
             + f'\nAsociación de cada variable con el puntaje, calculada solo en entrenamiento: las {len(primeras)} numéricas de mayor |ρ| y las dos '
               f'categóricas de mayor η². De las {len(en_modelo)} numéricas que entran al modelo, {len(debiles)} tienen |ρ| menor que 0,02; ninguna '
               f'categórica pasa de η² = {num(categoricas.fuerza.max(), 3)}.\n')
    f = pd.read_csv(eda / 'faltantes.csv')
    f = f[f.faltantes > 0].sort_values('fraccion', ascending=False)
    escribir('tabla_faltantes.md', tabla_md(['Columna', 'Faltantes', 'Parte de las filas', '¿Entra al modelo?'],
                                            [[f'`{x.columna}`', entero(x.faltantes), f'{num(100 * x.fraccion, 1)} %', 'Sí' if x.en_modelo else 'No']
                                             for x in f.itertuples()], 'lrrl'))

    # declaración de los datos: procedencia, licencia, composición y sesgos, con las cifras del entrenamiento
    tr = pd.read_parquet(eda / 'entrenamiento.parquet', columns=['age', 'churn_probability', 'credit_card', 'credit_utilization_ratio'])
    edades = sorted(int(x) for x in tr['age'].unique())
    derivadas = s[s.variable.isin(e['derivadas'])]
    productos = s[s.variable == 'active_products'].iloc[0]
    # en el resumen del entrenamiento el uso del cupo ya está imputado con 0: se compara con los faltantes del archivo original
    sin_tarjeta = tr['credit_card'].astype(float) == 0
    cupo = f[f.columna == 'credit_utilization_ratio'].iloc[0]
    solo_sin_tarjeta = int(cupo.faltantes) == int(sin_tarjeta.sum()) and bool((tr.loc[sin_tarjeta, 'credit_utilization_ratio'] == 0).all())
    falta_cupo = (f'el uso del cupo falta exactamente cuando el cliente no tiene tarjeta de crédito ({num(100 * cupo.fraccion, 1)} % de los '
                  'clientes de entrenamiento): es un «no aplica», no un dato perdido, y se imputa con 0' if solo_sin_tarjeta
                  else f'el uso del cupo falta en el {num(100 * cupo.fraccion, 1)} % de los clientes de entrenamiento')
    textos = f[~f.en_modelo.astype(bool)]
    filas = [
        ['Conjunto', 'COFINFAD: *Colombian Fintech Financial Analytics Dataset*'],
        ['Procedencia', 'Muñoz Guerrero, Ceballos y Trejos Rojas; publicado en Mendeley Data, versión 1, 2025; DOI '
                        '[10.17632/mhb4zn3258.1](https://doi.org/10.17632/mhb4zn3258.1)'],
        ['Licencia', 'CC BY 4.0: se puede usar y redistribuir citando a los autores'],
        ['Composición', f'{entero(e["n_clientes"])} clientes, uno por fila, con {e["n_columnas"]} columnas, y '
                        f'{entero(e["n_transacciones_limpias"] + a["duplicadas"])} transacciones de 2023 en una tabla aparte'],
        ['Naturaleza', f'**Sintético**, generado con reglas. Sus huellas: la edad solo toma {len(edades)} valores ({_lista(edades)}); el puntaje baja '
                       f'en escalones casi idénticos con el número de productos activos (r de Pearson = {num(productos.pearson, 3)}); y las '
                       f'{len(derivadas)} variables derivadas de las transacciones casi no se asocian con el puntaje (ρ de Spearman de a lo sumo '
                       f'{num(derivadas.fuerza.max(), 3)} en valor absoluto)'],
        ['Objetivo', f'`churn_probability`, un puntaje de propensión entre {num(tr["churn_probability"].min(), 3)} y '
                     f'{num(tr["churn_probability"].max(), 3)} en entrenamiento: nunca vale 0 ni 1 y no es una fuga observada'],
        ['Sesgos conocidos', f'Las cohortes de edad tienen tasas de riesgo alto muy distintas (del {num(100 * c.tasa_riesgo_alto.min(), 1)} % al '
                             f'{num(100 * c.tasa_riesgo_alto.max(), 1)} %); el archivo está ordenado por cohortes, y por eso la partición es '
                             f'aleatoria y no por posición; {falta_cupo}; las columnas de texto ({codigo(textos.columna)}) tienen entre el '
                             f'{num(100 * textos.fraccion.min(), 0)} % y el {num(100 * textos.fraccion.max(), 0)} % de faltantes y se excluyen'],
        ['Usos adecuados', 'Enseñar y comparar métodos de aprendizaje supervisado con un protocolo reproducible; estudiar fugas de información, '
                           'calibración y modelos legibles'],
        ['Usos no recomendados', 'Sacar conclusiones sobre clientes reales, estimar tasas reales de abandono (el objetivo es un puntaje, no la fuga '
                                 'observada) o tomar decisiones sobre personas'],
    ]
    escribir('tabla_declaracion.md', tabla_md(['Aspecto', 'Declaración'], filas, 'll'))


# ------------------------------------------------------------------ capítulo 2: auditoría de fuga
def fuga() -> None:
    a = json.loads((R / 'auditoria_fuga.json').read_text(encoding='utf-8'))
    f, p, pl = a['fecha_de_corte'], a['particion'], a['pliegues']
    inv = max(a['invarianza_de_la_recencia_estandarizada'].values())
    filas = [
        ['La partición es la del Entregable 3', 'Sí' if p['igual_que_entregable_3'] else 'No', f'{entero(p["n_train"])} de entrenamiento y {entero(p["n_test"])} de prueba; {p["clientes_en_ambas"]} clientes en las dos'],
        ['Fecha de corte con todo el detalle (Entregable 3)', f['entregable_3_todo_el_detalle'], 'Incluía a los clientes de prueba'],
        ['Fecha de corte solo con entrenamiento (corrección)', f['correccion_solo_entrenamiento'], 'La misma: hay ' + entero(pl['clientes_de_entrenamiento_con_transaccion_en_la_fecha_de_corte']) + ' clientes de entrenamiento con una transacción ese día'],
        ['Celdas de recencia que cambian', f['celdas_de_recencia_que_cambian'], f'Clientes con la última transacción posterior al corte: {f["clientes_con_ultima_transaccion_posterior_al_corte"]}'],
        ['Predictores idénticos antes y después', 'Sí' if f['predictores_identicos'] else 'No', f'Huella de entrenamiento `{f["huella_X_train_ahora"]}`; de prueba `{f["huella_X_test_ahora"]}`'],
        ['Pliegues de la validación anidada con otra fecha de corte', f'{pl["pliegues_con_otra_fecha_de_corte"]} de {pl["pliegues_revisados"]}', 'Tres repeticiones, externos e internos, las dos tareas'],
        ['Diferencia máxima de la recencia estandarizada al mover el corte entre −30 y +365 días', f'{inv:.1e}'.replace('.', ','), 'La fecha de corte se cancela al estandarizar'],
    ]
    escribir('tabla_fuga.md', tabla_md(['Comprobación', 'Resultado', 'Detalle'], filas, 'lll'))


# ------------------------------------------------------------------ capítulo 5: la limitación, a la vista
def figura_escalones() -> None:
    oof = pd.read_parquet(E3 / 'complementos' / 'fuera_de_pliegue.parquet')
    r = (oof['y_cont'] - oof['regresion__xgboost__ninguno__bayesiana__fuera_de_pliegue']).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.hist(r, bins=240, color=AZUL, alpha=0.9)
    ax.set_xlabel('Residuo fuera de pliegue de XGBoost: puntaje observado − pronóstico')
    ax.set_ylabel('Clientes')
    ax.set_title('El error del mejor benchmark de regresión no es una campana: tiene cinco escalones', loc='left', fontsize=10.5)
    FIG.mkdir(parents=True, exist_ok=True)
    _guardar(fig, 'residuos_escalones.png')
    plt.close(fig)
    _aviso('   figuras/residuos_escalones.png')


def figuras_eda() -> None:
    """Las dos figuras del capítulo 2, solo con entrenamiento (active_products se usa aquí para explicar el EDA)."""
    p = _problema()
    c, df, part = p['conjuntos'], p['df'], p['particion']
    s = c.y_train_cont.to_numpy()
    ap = df.loc[part.idx_train, 'active_products'].to_numpy()
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    niveles = sorted(np.unique(ap))
    cajas = ax.boxplot([s[ap == k] for k in niveles], positions=niveles, widths=0.55, patch_artist=True, showfliers=False,
                       medianprops={'color': 'white', 'linewidth': 1.6})
    for caja in cajas['boxes']:
        caja.set(facecolor=AZUL, edgecolor=AZUL, alpha=0.9)
    medias = [s[ap == k].mean() for k in niveles]
    ax.plot(niveles, medias, color=NARANJA, marker='o', linewidth=1.4, label='media')
    for k, m in zip(niveles, medias):
        ax.annotate(num(m, 3), (k, m), textcoords='offset points', xytext=(16, 6), fontsize=8.5, color=NARANJA)
    ax.axhline(p['etiqueta'].umbral, color=GRIS, linestyle='--', linewidth=1)
    ax.annotate('umbral de riesgo alto', (niveles[-1] + 0.45, p['etiqueta'].umbral), fontsize=8.5, color=GRIS, va='bottom', ha='right')
    ax.set_xlabel('Productos activos (variable excluida por fuga de información)')
    ax.set_ylabel('Puntaje de fuga')
    ax.set_title('Cada producto activo baja el puntaje un escalón de 0,05', loc='left', fontsize=10.5)
    ax.legend(frameon=False, loc='upper right')
    _guardar(fig, 'escalera_productos.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.hist(s, bins=120, color=AZUL, alpha=0.9)
    ax.axvline(p['etiqueta'].umbral, color=NARANJA, linewidth=1.6)
    ax.annotate(f'umbral = {num(p["etiqueta"].umbral, 4)}\n({num(100 * float((s > p["etiqueta"].umbral).mean()), 1)} % de riesgo alto)',
                (p['etiqueta'].umbral, ax.get_ylim()[1] * 0.86), textcoords='offset points', xytext=(8, 0), fontsize=8.5, color=NARANJA)
    ax.set_xlabel('Puntaje de fuga (churn_probability), entrenamiento')
    ax.set_ylabel('Clientes')
    ax.set_title('El umbral de riesgo alto cae en la zona más poblada del puntaje', loc='left', fontsize=10.5)
    _guardar(fig, 'objetivo.png')
    plt.close(fig)
    _aviso('   figuras/escalera_productos.png, figuras/objetivo.png')


# ------------------------------------------------------------------ utilidades de resultados
MAYOR = {'clasificacion': True, 'regresion': False}
METRICA = {'clasificacion': 'AUC', 'regresion': 'RMSE'}
DEC = {'clasificacion': 4, 'regresion': 5}
OPT = {'grid': 'rejilla (Grid)', 'random': 'aleatoria (Random)', 'bayesiana': 'bayesiana (TPE)', 'genetica': 'genética'}
BAL = {'ninguno': 'sin balanceo', 'class_weight': 'ponderación de clases'}
TAREA = {'regresion': 'Regresión (RMSE, menor es mejor)', 'clasificacion': 'Clasificación (AUC, mayor es mejor)'}
N = plan.NOMBRES


def _json(v):
    return json.loads(v) if isinstance(v, str) else v


def _fila(tabla: pd.DataFrame, tarea: str, modelo: str, semilla: int = SEED, combinacion=None):
    balanceo, optimizador = combinacion or plan.combinacion(tarea, modelo)
    f = tabla[(tabla.tarea == tarea) & (tabla.modelo == modelo) & (tabla.balanceo == balanceo)
              & (tabla.optimizador == optimizador) & (tabla.semilla == semilla) & (tabla.estado == 'ok')]
    return None if f.empty else f.iloc[0]


def _mas_menos(v, d) -> str:
    v = np.asarray(v, dtype=float)
    if d == 0:                                                          # cifras grandes: con separador de miles
        return f'{entero(v.mean())} ± {entero(v.std(ddof=1))}'
    return f'{num(v.mean(), d)} ± {num(v.std(ddof=1), d)}'


def _final(tarea: str, modelo: str = 'bdl'):
    """Tubería final (semilla 42) de un modelo nuevo, cargada de resultados/modelos."""
    import joblib
    f = _fila(plan.registro().cargar(), tarea, modelo)
    if f is None:
        raise FileNotFoundError(2, 'falta la corrida', f'{tarea}__{modelo}')
    return joblib.load(resolver(f['ruta_modelo']))


def _nombres_columnas(tuberia) -> list[str]:
    return [n.split('__', 1)[-1] for n in tuberia[:-1].get_feature_names_out()]


# ------------------------------------------------------------------ capítulo 3: benchmarks
def _mcs_regresion() -> dict:
    """*Model Confidence Set* al 90 % de los diez modelos de regresión, con sus errores cuadráticos en prueba (se guarda)."""
    from . import estadistica
    ruta, origen = R / 'mcs_regresion.json', R / 'predicciones_prueba.parquet'
    if ruta.exists() and ruta.stat().st_mtime > origen.stat().st_mtime:
        return json.loads(ruta.read_text(encoding='utf-8'))
    pred = pd.read_parquet(origen)
    y = _problema()['conjuntos'].y_test_cont.to_numpy()
    perdidas = pd.DataFrame({m: (y - pred[f'regresion__{m}'].to_numpy()) ** 2 for m in plan.modelos('regresion')})
    salida = {'alfa': 0.10, 'replicas': 1000}
    for clave, columnas in (('diez', list(perdidas.columns)), ('benchmarks', [m for m in perdidas.columns if m != plan.ORIGINAL])):
        r = estadistica.conjunto_confianza_modelos(perdidas[columnas], alfa=0.10, n_boot=1000)
        salida[clave] = {'bloque': int(r['bloque']), 'incluidos': [str(m) for m in r['incluidos']],
                         'p_mcs': {str(m): float(x) for m, x in r['p_mcs'].items()}}
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding='utf-8')
    return salida


def benchmarks() -> None:
    tabla = plan.registro().cargar()
    e3 = plan.tabla_e3()
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    partes, lectura = [], []
    for tarea in ('clasificacion', 'regresion'):
        d = DEC[tarea]
        M = pd.read_csv(R / f'pliegues_{tarea}.csv', index_col=0)
        filas = []
        for modelo in sorted(plan.modelos(tarea), key=lambda m: M[m].mean(), reverse=MAYOR[tarea]):
            f = _fila(tabla, tarea, modelo)
            balanceo, optimizador = plan.combinacion(tarea, modelo)
            if modelo in plan.BENCHMARKS_E3[tarea]:                      # la mejor de las combinaciones del Entregable 3
                candidatas = e3[(e3.tarea == tarea) & (e3.modelo == modelo) & (e3.semilla == SEED) & (e3.estado == 'ok')]['media_ext'].dropna().abs()   # el RMSE se guardó con signo negativo
                candidatas = candidatas.sort_values(ascending=not MAYOR[tarea])
                motivo = (f'La mejor de {len(candidatas)}; margen sobre la segunda: {num(abs(candidatas.iloc[0] - candidatas.iloc[1]), d + 1)}'
                          if len(candidatas) > 1 else 'La única evaluada')
            else:
                motivo = 'La única, fijada en el protocolo'
            nombre = f'**{N[modelo]}**' if modelo == plan.ORIGINAL else N[modelo] + (' *' if modelo in ('ebm', 'lightgbm') else '')
            combinacion = f'{BAL[balanceo]}, {OPT[optimizador]}' if tarea == 'clasificacion' else OPT[optimizador]
            filas.append([nombre, combinacion, motivo, int(f['presupuesto']), int(f['n_evaluaciones']) + int(f.get('n_evaluaciones_final') or 0),
                          num(float(f['tiempo_total_s']) / 60, 1), _mas_menos(M[modelo], d)])
        partes.append(f'**{TAREA[tarea]}, {len(M)} pliegues externos.**\n\n'
                      + tabla_md(['Modelo', 'Combinación', 'Por qué esa combinación', 'Presupuesto', 'Evaluaciones', 'Tiempo (min)',
                                  f'{METRICA[tarea]} externo'], filas, 'lllrrrr'))
        # mejor combinación global y finalistas de la comparación estadística
        fr, mejor = v[tarea]['friedman'], v[tarea]['mejor_benchmark']
        rangos = pd.Series(fr['rangos_medios']).sort_values()
        dc = float(fr['diferencia_critica_nemenyi'])
        finalistas = [m for m, r in rangos.items() if r - rangos.iloc[0] <= dc]
        balanceo, optimizador = plan.combinacion(tarea, mejor)
        combinacion = f'{BAL[balanceo]}, {OPT[optimizador]}' if tarea == 'clasificacion' else OPT[optimizador]
        frase = (f'En {NOMBRE_TAREA[tarea]}, la mejor combinación global entre los benchmarks es {N[mejor]} con '
                 f'{combinacion.replace("sin balanceo, ", "búsqueda ").replace("ponderación de clases, ", "ponderación de clases y búsqueda ") if tarea == "clasificacion" else "búsqueda " + combinacion}: '
                 f'{METRICA[tarea]} externo de {_mas_menos(M[mejor], d)}. La prueba de Friedman rechaza que los diez modelos rindan igual '
                 f'(p {"< 0,001" if fr["p_iman_davenport"] < 0.001 else "= " + pvalor(fr["p_iman_davenport"])} con la corrección de Iman-Davenport) y son '
                 f'finalistas de Nemenyi los {len(finalistas)} modelos a menos de la diferencia crítica ({num(dc, 2)} rangos) del mejor rango medio: '
                 f'{_lista(N[m] for m in finalistas)}. {N[mejor]} {"está" if mejor in finalistas else "no está"} entre ellos'
                 + (f', con rango medio {num(rangos[mejor], 1)}.' if mejor in rangos.index else '.'))
        if tarea == 'regresion':
            try:
                mcs = _mcs_regresion()
                entre, con = mcs['benchmarks'], mcs['diez']
                frase += (f' El *Model Confidence Set* al {int(round(100 * (1 - mcs["alfa"])))} % sobre los errores cuadráticos de prueba: entre los '
                          f'nueve benchmarks conserva a {_lista(N[m] for m in entre["incluidos"])}, de modo que {N[mejor]} '
                          f'{"pertenece al conjunto" if mejor in entre["incluidos"] else "queda fuera del conjunto"}; con el modelo original '
                          f'incluido conserva a {_lista(N[m] for m in con["incluidos"])}'
                          + (f' ({N[mejor]}: p = {pvalor(con["p_mcs"][mejor])})' if mejor in con['p_mcs'] and mejor not in con['incluidos'] else '')
                          + '. El *Model Confidence Set* no es una de las cuatro pruebas del protocolo: se reporta porque la guía lo pide para '
                            'la selección de combinaciones, y no cuenta para el veredicto.')
            except Exception as e:                                       # noqa: BLE001 - el paquete arch puede no estar instalado
                frase += f' (El *Model Confidence Set* no se pudo calcular: {type(e).__name__}.)'
        lectura.append(frase)
    escribir('tabla_benchmarks.md', '\n'.join(partes) + '\n\\* Método reciente. En negrita, el modelo original, que no es un benchmark: se incluye '
             'para comparar el presupuesto. Presupuesto: evaluaciones por búsqueda. Evaluaciones y tiempo: los de la repetición de la semilla 42 '
             '(cinco búsquedas externas y la final); los de los ocho modelos del Entregable 3 se midieron en ese entregable y los de LightGBM y el '
             'original, en este: son ejecuciones distintas, comparables en orden de magnitud y no al segundo. Margen: diferencia de la métrica externa entre la combinación elegida y la siguiente del '
             'mismo modelo.\n\n**Mejor combinación global y finalistas.** ' + ' '.join(lectura) + '\n')


# ------------------------------------------------------------------ capítulo 6: resumen, pruebas, veredicto
def _ic_prueba() -> pd.DataFrame:
    """Intervalo BCa del 95 % de la métrica de prueba de cada modelo (se calcula una vez y se guarda)."""
    from . import estadistica
    ruta = R / 'ic_prueba.csv'
    pred = pd.read_parquet(R / 'predicciones_prueba.parquet')
    if ruta.exists():
        ic = pd.read_csv(ruta)
        if len(ic) == sum(len(plan.modelos(t)) for t in MAYOR):
            return ic
    p = _problema()
    y = {'clasificacion': p['y_test'].to_numpy(), 'regresion': p['conjuntos'].y_test_cont.to_numpy()}
    filas = []
    for tarea in MAYOR:
        for modelo in plan.modelos(tarea):
            v = pred[f'{tarea}__{modelo}'].to_numpy()
            if tarea == 'clasificacion':
                b = estadistica.bootstrap_auc(y[tarea], v)
            else:
                b = estadistica.bootstrap_regresion(y[tarea], v)['rmse']
            filas.append({'tarea': tarea, 'modelo': modelo, 'estimacion': b['estimacion'], 'ic_bajo': b['ic_bca'][0], 'ic_alto': b['ic_bca'][1]})
    ic = pd.DataFrame(filas)
    ic.to_csv(ruta, index=False)
    return ic


def resumen() -> None:
    ic = _ic_prueba()
    ruta_semillas = R / 'semillas.csv'
    sem = pd.read_csv(ruta_semillas) if ruta_semillas.exists() else None
    for tarea in ('regresion', 'clasificacion'):
        M = pd.read_csv(R / f'pliegues_{tarea}.csv', index_col=0)
        d = DEC[tarea]
        filas = []
        for modelo in sorted(M.columns, key=lambda m: M[m].mean(), reverse=MAYOR[tarea]):
            i = ic[(ic.tarea == tarea) & (ic.modelo == modelo)].iloc[0]
            fila = [f'**{N[modelo]}**' if modelo == plan.ORIGINAL else N[modelo], _mas_menos(M[modelo], d), num(i['estimacion'], d),
                    f'[{num(i["ic_bajo"], d)}; {num(i["ic_alto"], d)}]']
            if sem is not None:
                s = sem[(sem.tarea == tarea) & (sem.modelo == modelo)]
                fila += [num(s['ajuste_s'].mean(), 1), num(s['inferencia_s'].mean(), 2), entero(s['tamano_kb'].mean())] if len(s) else ['—'] * 3
            filas.append(fila)
        encabezado = ['Modelo', f'{METRICA[tarea]} externo ({len(M)} pliegues)', f'{METRICA[tarea]} en prueba', 'IC 95 % (BCa)']
        if sem is not None:
            encabezado += ['Ajuste (s)', 'Inferencia (s)', 'Tamaño (KB)']
        escribir(f'tabla_resumen_{tarea}.md', tabla_md(encabezado, filas))


def _ic_diferencia_auc(mejor: str) -> dict:
    """Intervalo BCa de la diferencia de AUC en prueba, original menos mejor benchmark (bootstrap pareado; se guarda)."""
    from . import estadistica
    ruta, origen = R / 'ic_diferencia_auc.json', R / 'predicciones_prueba.parquet'
    if ruta.exists() and ruta.stat().st_mtime > origen.stat().st_mtime:
        guardado = json.loads(ruta.read_text(encoding='utf-8'))
        if guardado.get('benchmark') == mejor:
            return guardado
    pred = pd.read_parquet(origen)
    b = estadistica.bootstrap_auc(_problema()['y_test'].to_numpy(), pred[f'clasificacion__{mejor}'].to_numpy(),
                                  pred[f'clasificacion__{plan.ORIGINAL}'].to_numpy())
    salida = {'benchmark': mejor, 'estimacion': float(b['estimacion']), 'ic_bca': [float(x) for x in b['ic_bca']], 'n_boot': int(b['n_boot'])}
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding='utf-8')
    return salida


def orden_modelos() -> None:
    """Orden de los modelos antes y después de la corrección (guía, §3): si cambia, se dice."""
    e3 = plan.tabla_e3()
    partes = []
    for tarea in ('regresion', 'clasificacion'):
        M = pd.read_csv(R / f'pliegues_{tarea}.csv', index_col=0)
        antes = {}
        for modelo in plan.BENCHMARKS_E3[tarea]:
            balanceo, optimizador = plan.combinacion(tarea, modelo)
            f = e3[(e3.tarea == tarea) & (e3.modelo == modelo) & (e3.balanceo == balanceo) & (e3.optimizador == optimizador) & (e3.semilla == SEED)]
            antes[modelo] = abs(float(f['media_ext'].iloc[0]))            # el Entregable 3 guardó el RMSE con signo negativo
        orden_antes = sorted(antes, key=antes.get, reverse=MAYOR[tarea])
        orden_ahora = sorted(M.columns, key=lambda m: M[m].mean(), reverse=MAYOR[tarea])
        comunes = [m for m in orden_ahora if m in antes]
        nuevos = [f'{N[m]} en el puesto {orden_ahora.index(m) + 1}' for m in orden_ahora if m not in antes]
        frase = (f'**{NOMBRE_TAREA[tarea].capitalize()}.** Antes de la corrección (ocho modelos, semilla 42): {", ".join(N[m] for m in orden_antes)}. '
                 f'Ahora (diez modelos, {len(M)} pliegues externos): {", ".join(N[m] for m in orden_ahora)}. ')
        if comunes == orden_antes:
            frase += 'El orden relativo de los ocho modelos anteriores no cambia'
        else:
            movidos = [N[m] for m in comunes if comunes.index(m) != orden_antes.index(m)]
            frase += f'Entre los ocho modelos anteriores cambian de puesto {_lista(movidos)}'
        frase += f'; entran {_lista(nuevos)}.'
        partes.append(frase)
    partes.append('La corrección de la recencia no modificó ningún dato (sección 2.5), así que las corridas de la semilla 42 de los ocho modelos '
                  'anteriores son las mismas; lo que puede moverlos son las repeticiones nuevas de la validación anidada.')
    escribir('orden_modelos.md', '\n\n'.join(partes) + '\n')


def error_indice() -> None:
    """Con qué precisión estima cada modelo el índice: la limitación que motiva el modelo, medida en prueba."""
    p = _problema()
    c = p['conjuntos']
    te = json.loads((R / 'techo.json').read_text(encoding='utf-8'))
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    pred = pd.read_parquet(R / 'predicciones_prueba.parquet')
    mejor = v['regresion']['mejor_benchmark']
    y = c.y_test_cont.to_numpy()
    productos = p['df'].loc[c.X_test.index, 'active_products'].to_numpy()   # solo para medir el error del índice
    real = y + te['delta'] * (productos - 1)
    desplazamiento_medio = float(np.array([te['pesos'][str(k)] for k in range(1, 6)]) @ (-te['delta'] * np.arange(5)))
    del_benchmark = float(np.std(pred[f'regresion__{mejor}'].to_numpy() - desplazamiento_medio - real))
    texto = (f'**Evidencia en la partición de prueba.** El pronóstico de un regresor de mínimos cuadrados es el índice más el desplazamiento '
             f'medio; restándole ese desplazamiento queda el índice que el modelo aprendió. El de {N[mejor]} se aparta del índice real con una '
             f'desviación de {num(del_benchmark, 5)}')
    if f'regresion__{plan.ORIGINAL}__indice' in pred:
        del_original = float(np.std(pred[f'regresion__{plan.ORIGINAL}__indice'].to_numpy() - real))
        texto += (f'; el del modelo original, que separa el escalón, con una de {num(del_original, 5)} '
                  f'({num(del_benchmark / del_original, 1)} veces menor)')
    texto += ('. El índice real se reconstruye con `active_products`, que ningún modelo recibe: se usa solo para esta medida.')
    escribir('error_indice.md', texto + '\n')


def _supuestos(tarea: str, M: pd.DataFrame, mejor: str, dm: dict, n_pos: int, n_neg: int) -> dict:
    from scipy import stats
    d = (M[plan.ORIGINAL] - M[mejor]).to_numpy()
    sw = stats.shapiro(d).pvalue if len(d) >= 3 else float('nan')
    asim = float(stats.skew(d))
    return {
        'P1': f'Pares por pliegue; asimetría de las diferencias {num(asim, 2)}',
        'P2': f'Normalidad (Shapiro-Wilk, p = {pvalor(sw)}); varianza corregida por 1/J + 1/4',
        'P3': (f'Filas independientes: {entero(n_pos)} positivos y {entero(n_neg)} negativos'
               if tarea == 'clasificacion' else 'Filas independientes; sin supuesto de distribución'),
        'P4': (f'{entero(dm.get("P4", {}).get("solo_original", 0) + dm.get("P4", {}).get("solo_benchmark", 0))} pares discordantes'
               if tarea == 'clasificacion' else 'Filas independientes; intervalo BCa'),
    }


def pruebas() -> None:
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    t = pd.read_csv(R / 'pruebas_hipotesis.csv')
    p = _problema()
    n_pos = int(p['y_test'].sum())
    n_neg = int(len(p['y_test']) - n_pos)
    nombres = {'P1': 'P1. Wilcoxon de rangos con signo', 'P2': 'P2. *t* corregida de Nadeau-Bengio',
               'P3c': 'P3. DeLong (prueba)', 'P3r': 'P3. Permutación pareada (prueba)',
               'P4c': 'P4. McNemar, campaña del 25 % (prueba)', 'P4r': 'P4. Bootstrap pareado BCa (prueba)'}
    nula = {'P1': 'La mediana de las diferencias por pliegue es cero', 'P2': 'La diferencia media por pliegue es cero',
            'P3c': 'Los dos AUC son iguales', 'P3r': 'El error cuadrático medio es igual', 'P4c': 'Los dos aciertan con la misma frecuencia',
            'P4r': 'La diferencia de RMSE es cero'}
    for tarea in ('regresion', 'clasificacion'):
        r = v[tarea]
        M = pd.read_csv(R / f'pliegues_{tarea}.csv', index_col=0)
        mejor, dm, fr = r['mejor_benchmark'], r['frente_al_mejor'], r['friedman']
        sup = _supuestos(tarea, M, mejor, dm, n_pos, n_neg)
        letra = 'c' if tarea == 'clasificacion' else 'r'
        efecto = f'δ = {num(dm["cliff"], 2)}; *d* = {num(dm["cohen"], 2)}'
        filas = [['Friedman con Iman-Davenport (global)', 'Los diez modelos rinden igual', f'{r["pliegues"]} bloques completos, 10 modelos',
                  f'F = {num(fr["F_iman_davenport"], 1)}', pvalor(fr['p_iman_davenport']), '—', '—',
                  'Rechaza' if fr['p_iman_davenport'] < 0.05 else 'No rechaza']]
        for k in ('P1', 'P2', 'P3', 'P4'):
            if k not in dm:
                continue
            clave = k if k in ('P1', 'P2') else k + letra
            rechaza = r['rechazos_frente_al_mejor'].get(k, False)
            significativa = dm[k].get('p_holm', 1.0) < 0.05 and (k != 'P1' or fr['p_iman_davenport'] < 0.05)
            conclusion = ('Rechaza, a favor del original' if rechaza else
                          'Rechaza, a favor del benchmark' if significativa and not dm[k]['a_favor'] else 'No rechaza')
            filas.append([nombres[clave], nula[clave], sup[k], estad(dm[k]['estadistico']),
                          pvalor(dm[k]['p']), pvalor(dm[k].get('p_holm')), efecto if k in ('P1', 'P2') else '—', conclusion])
        texto = (f'**{TAREA[tarea]}. Modelo original frente al mejor benchmark, {N[mejor]}.** Ventaja media en los {r["pliegues"]} pliegues: '
                 f'{num(r["ventaja_sobre_el_mejor"], DEC[tarea] + 1)} (a favor del original en {dm["pliegues_a_favor"]}). '
                 f'Los p ajustados son de Holm sobre las {r["comparaciones_por_prueba"]} comparaciones de cada prueba.\n\n'
                 + tabla_md(['Prueba', 'Hipótesis nula', 'Supuestos verificados', 'Estadístico', 'p crudo', 'p ajustado', 'Tamaño del efecto', 'Conclusión'],
                            filas, 'lllrrrll'))
        texto += ('\nEstadísticos: en P1, la menor de las dos sumas de rangos; en P2, *t* de la diferencia original menos benchmark; en P3, '
                  + ('*z* de DeLong (positivo: el AUC del original es mayor)' if tarea == 'clasificacion'
                     else 'la diferencia de error cuadrático medio, original menos benchmark (negativo: el original se equivoca menos)')
                  + '; en P4, '
                  + ('los aciertos que solo tiene el original menos los que solo tiene el benchmark' if tarea == 'clasificacion'
                     else 'la diferencia de RMSE, original menos benchmark')
                  + f'. Tamaño del efecto: δ de Cliff y *d* de Cohen entre los valores por pliegue de los dos modelos, con signo positivo a favor del '
                    f'original; la *d* pareada de las diferencias es {num(dm["cohen_pareada"], 2)}. La diferencia crítica de Nemenyi es '
                    f'{num(fr["diferencia_critica_nemenyi"], 2)} rangos. Los pliegues de una validación cruzada comparten filas de entrenamiento, '
                    f'de modo que el p de Wilcoxon es optimista; la *t* corregida lo compensa.\n')
        if 2 / 2 ** r['pliegues'] >= 0.05:
            texto += (f'\nCon {r["pliegues"]} pliegues, el menor p-valor que puede dar la prueba de Wilcoxon es {num(2 / 2 ** r["pliegues"], 4)}: '
                      'P1 no puede rechazar con una sola repetición, sea cual sea el resultado.\n')
        sub = t[t.tarea == tarea]
        filas = []
        for b in sorted(sub.benchmark.unique(), key=lambda m: M[m].mean(), reverse=MAYOR[tarea]):
            s = sub[sub.benchmark == b].set_index('prueba')
            uno = s.iloc[0]
            filas.append([N[b], num(uno['ventaja_media_cv'], DEC[tarea] + 1), f'{int(uno["pliegues_a_favor"])}/{len(M)}']
                         + [f'{pvalor(s.loc[k, "p_crudo"])} / {pvalor(s.loc[k, "p_holm"])}' if k in s.index else '—' for k in ('P1', 'P2', 'P3', 'P4')]
                         + [num(uno['cliff'], 2), num(uno['cohen'], 2)])
        pruebas_hechas = int(sub['prueba'].nunique())
        texto += (f'\n**El original frente a cada benchmark** (ventaja positiva: a favor del original; en cada prueba, p crudo / p ajustado por '
                  f'Holm). Son {len(filas) * pruebas_hechas} comparaciones en esta tarea ({len(filas)} benchmarks por {pruebas_hechas} pruebas) y '
                  f'{len(t)} en las dos, además de la prueba global de cada una; el ajuste de Holm se hace dentro de cada prueba, sobre sus '
                  f'{len(filas)} comparaciones, como fija el protocolo.\n\n'
                  + tabla_md(['Benchmark', 'Ventaja media', 'Pliegues a favor', 'P1', 'P2', 'P3', 'P4', 'δ de Cliff', '*d* de Cohen'], filas))
        escribir(f'tabla_pruebas_{tarea}.md', texto)


def veredicto() -> None:
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    ruta4 = R / 'condicion4.json'
    c4 = json.loads(ruta4.read_text(encoding='utf-8')) if ruta4.exists() else {}
    si = lambda b: 'Sí' if b else 'No'                                   # noqa: E731
    filas, provisional = [], []
    for tarea in ('regresion', 'clasificacion'):
        r = v[tarea]
        n_rech = sum(r['rechazos_frente_al_mejor'].values())
        cuatro = c4.get(tarea, {})
        tres = r['condicion_1_media_mejor'] and r['condicion_2_tres_pruebas'] and r['condicion_3_efecto']
        completa = bool(cuatro.get('completo'))
        if not cuatro:
            celda4 = 'Pendiente'
        elif not completa:
            celda4 = f'Provisional, con {cuatro["conjuntos"]} de {cuatro.get("esperados", 4)} conjuntos: {cuatro["resumen"]}'
        else:
            celda4 = f'{si(cuatro["cumple"])} ({cuatro["resumen"]})'
        if tres and completa and cuatro['cumple']:
            final = '**Superior**'
        elif tres and not completa:
            final = '**Pendiente de la condición 4**'
        else:
            final = '**No superior**'
        if len(r['repeticiones']) < 3:
            provisional.append(f'{TAREA[tarea].split(" (")[0].lower()}, {len(r["repeticiones"])} de 3 repeticiones ({r["pliegues"]} pliegues)')
        filas.append([METRICA[tarea] + (' (regresión)' if tarea == 'regresion' else ' (clasificación)'), N[r['mejor_benchmark']],
                      f'{si(r["condicion_1_media_mejor"])} ({num(r["ventaja_sobre_el_mejor"], DEC[tarea] + 1)})',
                      f'{si(r["condicion_2_tres_pruebas"])} ({n_rech} de {len(r["rechazos_frente_al_mejor"])})',
                      f'{si(r["condicion_3_efecto"])} (δ = {num(r["frente_al_mejor"]["cliff"], 2)}; *d* = {num(r["frente_al_mejor"]["cohen"], 2)})',
                      celda4, final])
    nota = ('\nEn las condiciones 1 y 3, el número entre paréntesis es la ventaja del original (positiva: a su favor) y el tamaño del efecto '
            'con ese mismo signo. En la 2, las pruebas que rechazan a favor del original con p ajustado < 0,05.\n')
    if provisional:
        nota += f'\n**Resultado provisional:** {"; ".join(provisional)}. El protocolo fija el veredicto con las tres repeticiones.\n'
    escribir('tabla_veredicto.md', tabla_md(['Tarea', 'Mejor benchmark', '1. Media externa mejor', '2. Al menos tres pruebas rechazan',
                                             '3. Efecto no despreciable', '4. Conjuntos adicionales', 'Veredicto R3'], filas, 'lllllll') + nota)


def figura_cd() -> None:
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    for tarea in ('regresion', 'clasificacion'):
        fr = v[tarea]['friedman']
        rangos = sorted(fr['rangos_medios'].items(), key=lambda kv: kv[1])
        dc = fr['diferencia_critica_nemenyi']
        k = len(rangos)
        fig, ax = plt.subplots(figsize=(8.8, 3.3))
        margen = 3.7                                                    # sitio para los nombres, fuera del eje
        ax.set_xlim(1 - margen, k + margen)
        ax.set_ylim(-1.75, 1.2)
        ax.axis('off')
        ax.hlines(0, 1, k, color='black', linewidth=1)
        for r in range(1, k + 1):
            ax.vlines(r, -0.06, 0.06, color='black', linewidth=1)
            ax.text(r, 0.13, str(r), ha='center', va='bottom', fontsize=8.5)
        mitad = (k + 1) // 2
        for i, (m, r) in enumerate(rangos):
            izquierda = i < mitad
            nivel = -0.3 - 0.26 * (i if izquierda else k - 1 - i)
            x_codo = 0.7 if izquierda else k + 0.3
            color = NARANJA if m == plan.ORIGINAL else 'black'
            ax.plot([r, r, x_codo], [0, nivel, nivel], color=color, linewidth=1.3 if m == plan.ORIGINAL else 0.8)
            ax.text(x_codo + (-0.1 if izquierda else 0.1), nivel, f'{N[m]} ({num(r, 2)})', ha='right' if izquierda else 'left', va='center',
                    fontsize=8.5, color=color, fontweight='bold' if m == plan.ORIGINAL else 'normal')
        ax.hlines(0.95, 1, 1 + dc, color=AZUL, linewidth=2.2)
        ax.text(1 + dc / 2, 1.0, f'diferencia crítica = {num(dc, 2)}', ha='center', va='bottom', fontsize=8.5, color=AZUL)
        valores = [r for _, r in rangos]
        grupos, y0 = [], 0.42
        for i in range(k):                                              # grupos máximos de modelos que no difieren
            j = i
            while j + 1 < k and valores[j + 1] - valores[i] < dc:
                j += 1
            if j > i and not any(a <= i and j <= b for a, b in grupos):
                grupos.append((i, j))
        for n_g, (i, j) in enumerate(grupos):
            ax.hlines(y0 + 0.11 * (n_g % 4), valores[i] - 0.05, valores[j] + 0.05, color=GRIS, linewidth=3)
        ax.set_title(f'{TAREA[tarea].split(" (")[0]}: rango medio de cada modelo en {v[tarea]["pliegues"]} pliegues (1 = mejor)', loc='left', fontsize=10.5)
        FIG.mkdir(parents=True, exist_ok=True)
        _guardar(fig, f'cd_{tarea}.png')
        plt.close(fig)
    _aviso('   figuras/cd_regresion.png, figuras/cd_clasificacion.png')


def techo() -> None:
    te = json.loads((R / 'techo.json').read_text(encoding='utf-8'))
    m = pd.read_csv(R / 'metricas_prueba.csv')
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    filas = []
    for tarea, col, clave, d in (('regresion', 'rmse', 'rmse_minimo', 5), ('regresion', 'mae', 'mae_minimo', 5),
                                 ('clasificacion', 'auc', 'auc_maximo', 4), ('clasificacion', 'logloss_isotonica', 'logloss_minima', 4)):
        mejor = v[tarea]['mejor_benchmark']
        sub = m[m.tarea == tarea].set_index('modelo')
        limite = te['prueba'][clave]
        bench, orig = float(sub.loc[mejor, col]), float(sub.loc[plan.ORIGINAL, col if col != 'logloss_isotonica' else 'logloss'])
        if col == 'mae' and 'mae_mediana' in sub.columns and not np.isnan(sub.loc[plan.ORIGINAL, 'mae_mediana']):
            orig = float(sub.loc[plan.ORIGINAL, 'mae_mediana'])
        cerrado = (bench - orig) / (bench - limite) if abs(bench - limite) > 1e-12 else float('nan')
        etiqueta = {'rmse': 'RMSE', 'mae': 'MAE (el original predice la mediana)', 'auc': 'AUC',
                    'logloss_isotonica': 'Pérdida logarítmica (benchmark recalibrado)'}[col]
        filas.append([etiqueta, N[mejor], num(bench, d), num(orig, d), num(limite, d), f'{num(100 * cerrado, 0)} %'])
    escribir('tabla_techo.md', tabla_md(['Métrica en prueba', 'Mejor benchmark', 'Benchmark', 'Original', 'Límite teórico',
                                         'Parte de la distancia al límite que cierra el original'], filas, 'llrrrr')
             + f'\nLímite teórico: el valor que obtiene en la muestra de prueba el predictor óptimo, el que conoce el índice real de cada '
               f'cliente (δ = {num(te["delta"], 3)} y los pesos de los cinco niveles estimados en entrenamiento: '
               f'{", ".join(num(x, 3) for x in te["pesos"].values())}). En esperanza ningún modelo lo mejora; en una muestra concreta un modelo '
               'puede quedar del otro lado por azar' + _variante_bajo_el_limite(te) + '. La última columna describe esta muestra (un porcentaje '
               'negativo indica que el original queda más lejos del límite que el benchmark)'
             + (': las diferencias en prueba entre el original y el benchmark no son significativas tras el ajuste de Holm (tablas 6.7 y 6.8), '
                'así que no mide un aporte demostrado.\n'
                if not any(v[t]['rechazos_frente_al_mejor'].get(k) for t in ('regresion', 'clasificacion') for k in ('P3', 'P4')) else '.\n'))


def _variante_bajo_el_limite(te: dict) -> str:
    """Si una variante de la ablación queda en prueba por debajo del RMSE del predictor óptimo, se dice: el límite no es una cota."""
    ruta = R / 'ablacion.csv'
    if not ruta.exists():
        return ''
    a = pd.read_csv(ruta)
    a = a[(a.tarea == 'regresion') & (a.prueba < te['prueba']['rmse_minimo'])]
    if a.empty:
        return ''
    f = a.sort_values('prueba').iloc[0]
    return (f' (le ocurre a la variante «{f["nombre"]}», con un RMSE de {num(f["prueba"], 6)} frente a {num(te["prueba"]["rmse_minimo"], 6)})')


# ------------------------------------------------------------------ capítulo 5: convergencia, ablación, interpretación
def convergencia() -> None:
    tuberia = _final('regresion')
    b = tuberia.named_steps['modelo']
    h = b.historial_
    fig, ejes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    v = np.array(h['verosimilitud']) / b.n_ajuste_
    ejes[0].plot(np.arange(1, len(v) + 1), v, color=AZUL)
    ejes[0].set_xlabel('Vuelta de EM')
    ejes[0].set_ylabel('Log-verosimilitud por cliente (ajuste)')
    ejes[0].set_title('La verosimilitud nunca baja', loc='left', fontsize=10)
    cal, em = np.array(h['error_calentamiento']), np.array(h['error_validacion'])
    ejes[1].plot(np.arange(1, len(cal) + 1), cal, color=GRIS, label='calentamiento: RMSE')
    x_em = b.n_calentamiento_ + np.arange(1, len(em) + 1) * int(b.arboles_por_vuelta)
    ejes[1].plot(x_em, em, color=NARANJA, label='EM: error dentro de clase')
    ejes[1].axvline(b.n_arboles_, color=VERDE, linestyle='--', linewidth=1)
    ejes[1].set_yscale('log')
    ejes[1].set_xlabel('Árboles')
    ejes[1].set_ylabel('Error en validación')
    ejes[1].set_title(f'Al separar el escalón, el error cae {entero(round(float(cal.min() / em.min())))} veces', loc='left', fontsize=10)
    ejes[1].legend(frameon=False, fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    _guardar(fig, 'convergencia.png')
    plt.close(fig)
    filas = [[f['K'], entero(round(f['bic'])), f['utiles']] for f in h['bic']]
    escribir('tabla_bic.md', tabla_md(['K', 'BIC', 'Clases que quedan tras la regla de parsimonia'], filas, 'rrr')
             + f'\nModelo final de regresión: K = {b.n_componentes_}; pesos {", ".join(num(x, 3) for x in b.pi_)}; '
               f'separación media entre desplazamientos {num(float(np.diff(b.centros_).mean()), 4)}; {entero(b.n_arboles_)} árboles '
               f'({b.n_calentamiento_} de calentamiento); dispersión dentro de clase en validación {num(b.sigma_validacion_, 5)} '
               f'(robusta: {num(b.sigma_robusta_, 5)}).\n')
    _aviso('   figuras/convergencia.png')


def ablacion() -> None:
    t = pd.read_csv(R / 'ablacion.csv')
    partes = []
    for tarea in ('regresion', 'clasificacion'):
        d = DEC[tarea]
        filas = []
        for _, f in t[t.tarea == tarea].iterrows():
            if f['variante'] == 'bdl':
                filas.append([f'**{f["nombre"]}**', f'{num(f["cv_media"], d)} ± {num(f["cv_desv"], d)}', num(f['prueba'], d), '—', '—', '—', '—'])
            else:
                filas.append([f['nombre'], f'{num(f["cv_media"], d)} ± {num(f["cv_desv"], d)}', num(f['prueba'], d),
                              f'{num(f["aporte_cv"], d + 1)} ({int(f["pliegues_a_favor_del_completo"])}/5)', pvalor(f.get('p_t_corregida_holm')),
                              num(f['aporte_prueba'], d + 1), pvalor(f.get('p_prueba_holm'))])
        partes.append(f'**{TAREA[tarea]}.**\n\n' + tabla_md(['Variante', f'{METRICA[tarea]} externo', f'{METRICA[tarea]} en prueba',
                                                               'Aporte del componente (externo)', 'p (*t* corregida, Holm)',
                                                               'Aporte en prueba', 'p (prueba, Holm)'], filas))
    escribir('tabla_ablacion.md', '\n'.join(partes) + '\nAporte: lo que mejora el modelo completo respecto de la variante (positivo: el componente ayuda). '
             'En prueba, la comparación es DeLong en clasificación y permutación pareada en regresión.\n\n' + _lectura_ablacion(t) + '\n')


QUITA = {'bdl_sin_mezcla': 'retirar la mezcla ($K = 1$)', 'bdl_sin_puntaje': 'aprender de la etiqueta en lugar del puntaje',
         'bdl_conforme': 'cambiar la salida de la mezcla por la conforme empírica'}


def _lectura_ablacion(t: pd.DataFrame) -> str:
    """Lectura de la tabla de ablación: cuánto cuesta retirar cada componente y si la diferencia es significativa."""
    partes, sig_cv, sig_prueba = [], [], []
    for tarea in ('regresion', 'clasificacion'):
        d = DEC[tarea]
        frases = []
        for _, f in t[(t.tarea == tarea) & (t.variante != 'bdl')].iterrows():
            que = QUITA.get(f['variante'], f['nombre'])
            cv = (f'cuesta {num(f["aporte_cv"], d + 1)}' if f['aporte_cv'] > 0 else f'mejora en {num(abs(f["aporte_cv"]), d + 1)}')
            prueba = (f'en prueba cuesta {num(f["aporte_prueba"], d + 1)}' if f['aporte_prueba'] > 0 else
                      f'en prueba la variante queda {num(abs(f["aporte_prueba"]), d + 1)} por delante')
            frases.append(f'{que} {cv} en la media externa (el modelo completo va por delante en '
                          f'{int(f["pliegues_a_favor_del_completo"])} de 5 pliegues; p = {pvalor(f["p_t_corregida_holm"])}) y '
                          f'{prueba} (p = {pvalor(f["p_prueba_holm"])})')
            if f['p_t_corregida_holm'] < 0.05 and f['aporte_cv'] > 0:
                sig_cv.append(f'{que} en {NOMBRE_TAREA[tarea]}')
            if f['p_prueba_holm'] < 0.05 and f['aporte_prueba'] > 0:
                sig_prueba.append(f'{que} en {NOMBRE_TAREA[tarea]}')
        if frases:
            partes.append(f'En {NOMBRE_TAREA[tarea]} ({METRICA[tarea]}), ' + '; '.join(frases) + '.')
    texto = '**Lectura.** ' + ' '.join(partes)
    texto += (' Tras el ajuste de Holm, ' + (f'en los pliegues externos solo es significativo el costo de {_lista(sig_cv)}' if sig_cv else
                                              'ningún aporte es significativo en los pliegues externos')
              + (f'; en prueba, {_lista(sig_prueba)}.' if sig_prueba else '; en prueba, ninguno.'))
    c = t[t.tarea == 'clasificacion'].set_index('variante')
    if {'bdl_sin_puntaje', 'bdl_sin_mezcla'} <= set(c.index) and c.loc['bdl_sin_puntaje', 'aporte_prueba'] > c.loc['bdl_sin_mezcla', 'aporte_prueba']:
        texto += (' En clasificación, lo que más pesa en prueba es aprender del puntaje continuo y no de su dicotomización: es la diferencia entre '
                  f'un AUC de {num(c.loc["bdl", "prueba"], 4)} y uno de {num(c.loc["bdl_sin_puntaje", "prueba"], 4)}.')
    texto += (' Que los aportes sean tan pequeños es coherente con la sección 6.4: con todas las filas, un boosting sin mezcla ya está casi en '
              'el límite del conjunto. En regresión la mezcla no cambia lo que el modelo estima, que es la media condicional del puntaje'
              + (', sino con cuántos datos lo consigue (sección 7.5)' if (R / 'robustez' / 'muestra_sin_mezcla.csv').exists() else '')
              + '. Lo que añade es lo que el modelo sabe del error: cuántos niveles hay, dónde están y con qué probabilidad (sección 7.8).')
    return texto


def importancia() -> None:
    p = _problema()
    tuberia = _final('regresion')
    b = tuberia.named_steps['modelo']
    nombres = _nombres_columnas(tuberia)
    X = tuberia[:-1].transform(p['conjuntos'].X_test.iloc[:2000])
    aportes, _ = b.contribuciones(X)
    imp = pd.Series(np.abs(aportes).mean(axis=0), index=nombres).sort_values(ascending=False)
    filas = [[i + 1, f'`{v}`', num(x, 5), f'{num(100 * x / imp.sum(), 1)} %'] for i, (v, x) in enumerate(imp.head(8).items())]
    texto = tabla_md(['N.º', 'Variable', 'Aporte medio absoluto al puntaje', 'Parte del total'], filas, 'rlrr')
    ruta_shap = plan.RAIZ.parent / 'Dashboard' / 'datos' / 'shap' / 'regresion__xgboost.parquet'
    if ruta_shap.exists():
        sx = pd.read_parquet(ruta_shap)
        sx = sx[[c for c in sx.columns if c != 'fila' and not c.startswith('__')]].abs().mean().sort_values(ascending=False)
        texto += (f'\nLas cinco variables de más peso en BDL son {", ".join(f"`{v}`" for v in imp.head(5).index)}; en el mejor benchmark de '
                  f'regresión (XGBoost, valores SHAP de 500 clientes de prueba), {", ".join(f"`{v}`" for v in sx.head(5).index)}.\n')
    escribir('tabla_importancia.md', texto)
    if b.formas_ is not None:
        pre = tuberia.named_steps['preprocesado']
        escala = {n: (pre.named_transformers_['num'].mean_[i], pre.named_transformers_['num'].scale_[i])
                  for i, n in enumerate(pre.transformers_[0][2])}
        principales = [v for v in imp.index if v in escala][:3]
        ruta_variables = plan.RAIZ.parent / 'Dashboard' / 'datos' / 'variables.csv'
        dic = pd.read_csv(ruta_variables).set_index('variable') if ruta_variables.exists() else None
        fig, ejes = plt.subplots(1, len(principales), figsize=(9.2, 3.2))
        for ax, v in zip(np.atleast_1d(ejes), principales):
            j = nombres.index(v)
            media, desv = escala[v]
            cortes = b.cortes_[j] * desv + media
            forma = b.formas_[j] - b.formas_[j] @ (b.conteo_bins_[j] / b.conteo_bins_[j].sum())
            crudo = p['conjuntos'].X_train[v]
            bordes = np.concatenate(([crudo.min()], cortes, [crudo.max()]))
            ax.step(bordes, np.concatenate((forma, [forma[-1]])), where='post', color=AZUL)
            ax.axhline(0, color=GRIS, linewidth=0.8)
            etiqueta = v
            if dic is not None and v in dic.index:                # nombre y unidad del diccionario de variables del dashboard
                unidad = dic.loc[v, 'unidad']
                etiqueta = str(dic.loc[v, 'nombre']) + (f' ({unidad})' if isinstance(unidad, str) and unidad else '') + f'\n{v}'
            ax.set_xlabel(etiqueta)
            if crudo.max() > 50 * max(crudo.median(), 1):
                ax.set_xscale('log')
        np.atleast_1d(ejes)[0].set_ylabel('Aporte al puntaje')
        fig.suptitle('Curvas del modelo original: lo que cada variable suma o resta al puntaje', x=0.01, ha='left', fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _guardar(fig, 'formas_bdl.png')
        plt.close(fig)
        _aviso('   figuras/formas_bdl.png')


# ------------------------------------------------------------------ capítulo 7: conjuntos adicionales
def adicionales() -> None:
    from scipy import stats
    from . import estadistica, pruebas as pr
    c4, consolidado = {}, {}
    partes = []
    fichas = []
    import joblib
    for tarea in ('regresion', 'clasificacion'):
        filas, mejores, peores_sig, incompletos, detalle = [], 0, [], [], {}
        medias_por_conjunto = {}
        for nombre in conjuntos.CARGADORES:
            carpeta = R / 'adicionales' / nombre
            if not all((carpeta / a).exists() for a in ('experimentos.parquet', 'metricas_prueba.csv', 'ficha.json')):   # en curso
                continue
            ficha = json.loads((carpeta / 'ficha.json').read_text(encoding='utf-8'))
            if tarea == 'regresion':
                fichas.append([ficha['nombre'], entero(ficha['n']), ficha['p'], ficha['omitida'] or '—', num(100 * ficha['prevalencia_train'], 1) + ' %',
                               f'`{ficha["sha256"][:12]}`'])
            tabla = plan.registro(nombre).cargar()
            pliegues, clases, configurado, sin_mezcla, n_pliegues = {}, None, None, 0, 0
            for modelo in plan.modelos(tarea):
                f = _fila(tabla, tarea, modelo)
                if f is not None:
                    pliegues[modelo] = plan.puntajes_por_pliegue(f)
                    if modelo == plan.ORIGINAL:                          # qué eligió la búsqueda: la mezcla ('auto') o sin mezcla (1)
                        configurado = _json(f['hiperparametros']).get('modelo__n_componentes')
                        por_pliegue = [x.get('modelo__n_componentes') for x in _json(f['params_por_pliegue'])]
                        sin_mezcla, n_pliegues = sum(str(x) == '1' for x in por_pliegue), len(por_pliegue)
                    if modelo == plan.ORIGINAL and f['ruta_modelo'] and resolver(f['ruta_modelo']).is_file():
                        final = joblib.load(resolver(f['ruta_modelo'])).named_steps['modelo']
                        if getattr(final, 'modo_', None) == 'directo':
                            clases = 'sin puntaje'
                        else:
                            clases = getattr(getattr(final, 'regresor_', final), 'n_componentes_', None)
                            clases = None if clases is None else int(clases)
            if plan.ORIGINAL not in pliegues or len(pliegues) < 3:
                continue
            if len(pliegues) < len(plan.modelos(tarea)):
                incompletos.append(ficha['nombre'])
            medias = {m: v.mean() for m, v in pliegues.items()}
            medias_por_conjunto[nombre] = medias
            otros = {m: v for m, v in medias.items() if m != plan.ORIGINAL}
            mejor = (max if MAYOR[tarea] else min)(otros, key=otros.get)
            signo = 1.0 if MAYOR[tarea] else -1.0
            ventaja = signo * (medias[plan.ORIGINAL] - medias[mejor])
            t = pr.t_corregida(pliegues[plan.ORIGINAL], pliegues[mejor])
            es_mejor = ventaja > 0
            peor_sig = (not es_mejor) and t['p'] < 0.05
            mejores += int(es_mejor)
            if peor_sig:
                peores_sig.append(ficha['nombre'])
            mp = pd.read_csv(carpeta / 'metricas_prueba.csv')
            col = 'auc' if tarea == 'clasificacion' else 'rmse'
            mp = mp[mp.tarea == tarea].set_index('modelo')[col]
            d = 4 if tarea == 'clasificacion' else (5 if medias[mejor] < 1 else (3 if medias[mejor] < 100 else 0))
            rango = pd.Series(medias).rank(ascending=not MAYOR[tarea])[plan.ORIGINAL]
            filas.append([ficha['nombre'] + (' †' if nombre in conjuntos.CON_VARIABLE_OMITIDA else ''), N[mejor], _mas_menos(pliegues[mejor], d),
                          _mas_menos(pliegues[plan.ORIGINAL], d), f'{int(rango)} de {len(medias)}',
                          f'{num(100 * ventaja / abs(medias[mejor]), 2)} %', pvalor(t['p']),
                          'Mejor' if es_mejor else ('Peor, significativo' if peor_sig else 'Peor, no significativo'),
                          f'{entero(mp[mejor])} / {entero(mp[plan.ORIGINAL])}' if d == 0 else f'{num(float(mp[mejor]), d)} / {num(float(mp[plan.ORIGINAL]), d)}',
                          '—' if clases is None else clases if not isinstance(clases, int) else
                          f'{clases} (sin mezcla)' if str(configurado) == '1' else f'{clases} (BIC)'])
            detalle[nombre] = {'nombre': ficha['nombre'], 'con_variable_omitida': nombre in conjuntos.CON_VARIABLE_OMITIDA, 'mejor_benchmark': mejor,
                               'ventaja_relativa': float(ventaja / abs(medias[mejor])), 'p': float(t['p']), 'es_mejor': bool(es_mejor),
                               'peor_significativo': bool(peor_sig), 'puesto': int(rango), 'modelos': len(medias), 'clases': clases,
                               'n_componentes_configurado': None if configurado is None else str(configurado),
                               'pliegues_sin_mezcla': int(sin_mezcla), 'pliegues': int(n_pliegues),
                               'prueba_benchmark': float(mp[mejor]), 'prueba_original': float(mp[plan.ORIGINAL])}
        if not filas:
            continue
        partes.append(f'**{TAREA[tarea]}.**\n\n' + tabla_md(['Conjunto', 'Mejor benchmark', 'Benchmark (externo)', 'Original (externo)', 'Puesto del original',
                                                               'Ventaja relativa', 'p (*t* corregida)', 'Resultado', 'Prueba: benchmark / original',
                                                               '$K$ del modelo final'], filas, 'llrrrrrlrr'))
        cumple = mejores * 2 >= len(filas) and not peores_sig
        completo = len(filas) == len(conjuntos.CARGADORES) and not incompletos
        c4[tarea] = {'conjuntos': len(filas), 'esperados': len(conjuntos.CARGADORES), 'completo': bool(completo), 'con_modelos_pendientes': incompletos,
                     'mejor_en': mejores, 'peor_significativo_en': peores_sig, 'cumple': bool(cumple), 'detalle': detalle,
                     'resumen': f'mejor en {mejores} de {len(filas)}' + (f'; peor de forma significativa en {", ".join(peores_sig)}' if peores_sig else '; nunca peor de forma significativa')}
        # prueba consolidada: rangos de los diez modelos por conjunto (los adicionales y COFINFAD)
        M = pd.read_csv(R / f'pliegues_{tarea}.csv', index_col=0) if (R / f'pliegues_{tarea}.csv').exists() else None
        if M is not None:
            medias_por_conjunto['cofinfad'] = M.mean().to_dict()
        D = pd.DataFrame(medias_por_conjunto).T.dropna(axis=1)
        if len(D) >= 3 and D.shape[1] >= 3:
            fr = estadistica.friedman(D, mayor_es_mejor=MAYOR[tarea])
            crudos = {}
            for m in D.columns:
                if m != plan.ORIGINAL:
                    crudos[m] = pr.wilcoxon_pareado(D[plan.ORIGINAL], D[m])['p']
            ajust = dict(zip(crudos, estadistica.ajustar_pvalores(list(crudos.values()), 'holm')))
            consolidado[tarea] = {'conjuntos': list(D.index), 'modelos': int(D.shape[1]), 'F': fr['F'], 'p_iman_davenport': fr['p_iman_davenport'],
                                  'rangos': {m: float(x) for m, x in fr['rangos_medios'].items()},
                                  'p_holm_minimo': float(min(ajust.values())), 'p_wilcoxon_minimo_posible': float(2 / 2 ** len(D))}
    if not partes:
        raise FileNotFoundError(2, 'no hay resultados', 'adicionales')
    (R / 'condicion4.json').write_text(json.dumps(c4, ensure_ascii=False, indent=2), encoding='utf-8')
    escribir('tabla_conjuntos.md', tabla_md(['Conjunto', 'Filas', 'Predictores', 'Variable omitida', 'Clase positiva (entrenamiento)', 'Huella del archivo'],
                                            fichas, 'lrrlrl'))
    escribir('tabla_adicionales.md', '\n'.join(partes) + '\n† Conjunto con una variable omitida. Externo: media ± desviación de los cinco pliegues externos. '
             'Ventaja relativa: positiva si el original es mejor. La prueba *t* corregida compara al original con el mejor benchmark del conjunto. '
             '$K$: clases latentes del modelo final. «Sin mezcla»: la búsqueda de hiperparámetros eligió `n_componentes = 1` y el modelo es un '
             'boosting de mínimos cuadrados con árboles propios. «BIC»: la búsqueda eligió la mezcla y el criterio BIC fijó el número de clases. '
             '«Sin puntaje»: en clasificación, la búsqueda eligió aprender de la etiqueta.\n')
    filas = []
    for tarea, c in consolidado.items():
        orden = sorted(c['rangos'].items(), key=lambda kv: kv[1])
        puesto = [m for m, _ in orden].index(plan.ORIGINAL) + 1
        filas.append([TAREA[tarea].split(' (')[0], len(c['conjuntos']), f'F = {num(c["F"], 2)}', pvalor(c['p_iman_davenport']),
                      f'{num(c["rangos"][plan.ORIGINAL], 2)} (puesto {puesto} de {c["modelos"]})', f'{N[orden[0][0]]} ({num(orden[0][1], 2)})',
                      pvalor(c['p_holm_minimo']), pvalor(c['p_wilcoxon_minimo_posible'])])
    if filas:
        escribir('tabla_consolidada.md', tabla_md(['Tarea', 'Conjuntos', 'Friedman con Iman-Davenport', 'p', 'Rango medio del original', 'Mejor rango medio',
                                                   'Menor p de Wilcoxon-Holm (original contra cada benchmark)', 'Menor p posible con estos conjuntos'], filas, 'lrrrllrr'))


# ------------------------------------------------------------------ capítulo 7: robustez
TRES = ('bdl', 'xgboost', 'lightgbm')                                      # los modelos de los estudios de robustez


def _referencia_anidada(tarea: str) -> tuple[str, float]:
    """Mejor benchmark de la tarea y su media en los pliegues externos de la semilla 42 (validación anidada)."""
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    M = pd.read_csv(R / f'pliegues_{tarea}.csv', index_col=0)
    mejor = v[tarea]['mejor_benchmark']
    return mejor, float(M.loc[[f for f in M.index if f.startswith(f'{SEED}-')], mejor].mean())


def _referencia_homogenea(tarea: str) -> tuple[str, float, float] | None:
    """El mejor benchmark de boosting con sus hiperparámetros finales, en los cinco pliegues externos de la semilla 42.

    Es la referencia homogénea de los estudios de robustez: la misma receta (hiperparámetros finales, sin búsqueda) y los
    mismos pliegues que se usan para el modelo original. Sale de ``robustez/muestra.csv`` con todas las filas. Devuelve
    (modelo, media, desviación) o ``None`` si ese estudio todavía no existe.
    """
    ruta = R / 'robustez' / 'muestra.csv'
    if not ruta.exists():
        return None
    mu = pd.read_csv(ruta)
    mu = mu[(mu.tarea == tarea) & (mu.fraccion == 1.0) & (mu.modelo != plan.ORIGINAL)].set_index('modelo')
    if mu.empty:
        return None
    mejor = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))[tarea]['mejor_benchmark']
    if mejor not in mu.index:
        mejor = mu['media'].idxmax() if MAYOR[tarea] else mu['media'].idxmin()
    return mejor, float(mu.loc[mejor, 'media']), float(mu.loc[mejor, 'desv'])


def sensibilidad() -> None:
    s = pd.read_csv(R / 'robustez' / 'sensibilidad.csv')
    parametros = list(dict.fromkeys(s['parametro']))
    columnas = 4                                                         # cuatro paneles por fila: legibles al ancho del libro
    por_tarea = int(np.ceil((len(parametros) + 1) / columnas))           # un hueco por tarea para el rótulo
    fig, ejes = plt.subplots(2 * por_tarea, columnas, figsize=(9.6, 2.3 * 2 * por_tarea), squeeze=False)
    filas = []
    for i, tarea in enumerate(('regresion', 'clasificacion')):
        homogenea = _referencia_homogenea(tarea)
        mejor, referencia = (homogenea[0], homogenea[1]) if homogenea else _referencia_anidada(tarea)
        bloque = ejes[i * por_tarea:(i + 1) * por_tarea].ravel()
        for ax, par in zip(bloque, parametros):
            sub = s[(s.tarea == tarea) & (s.parametro == par)]
            x = np.arange(len(sub))
            ax.errorbar(x, sub['media'], yerr=sub['desv'], color=AZUL, marker='o', markersize=3.5, linewidth=1, capsize=2)
            ax.axhline(referencia, color=NARANJA, linestyle='--', linewidth=1)
            final = np.flatnonzero((sub['es_el_final'].astype(str) == 'True').to_numpy())
            if len(final):
                ax.plot(x[final], sub['media'].to_numpy()[final], marker='o', markersize=7, markerfacecolor='none', markeredgecolor=AZUL, linestyle='none')
            ax.set_xticks(x)
            ax.set_xticklabels([_nivel(v) for v in sub['valor']], fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_title(par, fontsize=9, loc='left', family='monospace')
            ax.set_ylabel(METRICA[tarea], fontsize=8.5)
        for ax in bloque[len(parametros):]:                               # el hueco: qué tarea es y qué significa cada marca
            ax.axis('off')
            ax.text(0.0, 0.92, TAREA[tarea].split(' (')[0], fontsize=11, fontweight='bold', va='top', transform=ax.transAxes)
            ax.text(0.0, 0.70, f'{METRICA[tarea]} ({"mayor" if MAYOR[tarea] else "menor"} es mejor),\nmedia ± desviación en cinco pliegues',
                    fontsize=8.5, va='top', transform=ax.transAxes)
            ax.plot([0.02, 0.16], [0.36, 0.36], color=NARANJA, linestyle='--', linewidth=1, transform=ax.transAxes, clip_on=False)
            ax.text(0.20, 0.36, f'{N[mejor]} con sus\nhiperparámetros finales', fontsize=8.5, va='center', transform=ax.transAxes)
            ax.plot([0.09], [0.12], marker='o', markersize=7, markerfacecolor='none', markeredgecolor=AZUL, linestyle='none',
                    transform=ax.transAxes, clip_on=False)
            ax.text(0.20, 0.12, 'configuración final', fontsize=8.5, va='center', transform=ax.transAxes)
        sub = s[s.tarea == tarea]
        gana = (sub['media'] > referencia) if MAYOR[tarea] else (sub['media'] < referencia)
        peor, mejor_valor = (sub['media'].min(), sub['media'].max()) if MAYOR[tarea] else (sub['media'].max(), sub['media'].min())
        final = sub[sub['es_el_final'].astype(str) == 'True']
        filas.append([TAREA[tarea].split(' (')[0], len(sub), num(final.iloc[0]['media'], DEC[tarea]) if len(final) else '—',
                      f'{num(referencia, DEC[tarea])} ({N[mejor]})', int(gana.sum()), num(peor, DEC[tarea]), num(mejor_valor, DEC[tarea])])
    fig.suptitle('Sensibilidad del modelo original a cada hiperparámetro propio', x=0.01, ha='left', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _guardar(fig, 'sensibilidad.png')
    plt.close(fig)
    escribir('tabla_sensibilidad.md', tabla_md(['Tarea', 'Configuraciones', 'Configuración final', 'Mejor benchmark, con sus hiperparámetros finales',
                                                'Configuraciones mejores que ese benchmark', 'Peor valor', 'Mejor valor'], filas, 'lrrrrrr')
             + '\nCada configuración cambia un solo hiperparámetro respecto de la final y se evalúa en los cinco pliegues externos de la semilla 42, '
               'sin nueva búsqueda. La referencia es el mejor benchmark evaluado igual: con sus hiperparámetros finales y en los mismos pliegues.\n\n'
             + _lectura_sensibilidad(s) + '\n')
    _aviso('   figuras/sensibilidad.png')


def _nivel(v) -> str:
    """Valor de un hiperparámetro tal como está en ``sensibilidad.csv`` ('0.05', '6', 'auto'), con coma decimal."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(x)) if x == int(x) else str(v).replace('.', ',')


def _lectura_sensibilidad(s: pd.DataFrame) -> str:
    """Cuánto pierde el modelo al mover cada hiperparámetro, cómo queda frente al benchmark y qué no dice este estudio."""
    partes = []
    for tarea in ('regresion', 'clasificacion'):
        d, signo = DEC[tarea], (1.0 if MAYOR[tarea] else -1.0)
        homogenea = _referencia_homogenea(tarea)
        mejor, referencia = (homogenea[0], homogenea[1]) if homogenea else _referencia_anidada(tarea)
        sub = s[s.tarea == tarea]
        gana = signo * (sub['media'] - referencia) > 0
        final = sub[sub['es_el_final'].astype(str) == 'True']
        frase = f'En {NOMBRE_TAREA[tarea]}, '
        if gana.sum() == 0:
            mejor_fila = sub.loc[(signo * sub['media']).idxmax()]
            frase += (f'ninguna de las {len(sub)} configuraciones queda por delante de {N[mejor]} con sus hiperparámetros finales '
                      f'({num(referencia, d)}): la mejor obtiene {num(mejor_fila["media"], d)}. ')
        else:
            frase += (f'{int(gana.sum())} de las {len(sub)} configuraciones quedan por delante de {N[mejor]} con sus hiperparámetros finales '
                      f'({num(referencia, d)}). ')
        if len(final):
            base, tolerancia = float(final.iloc[0]['media']), float(final.iloc[0]['desv'])
            caen = sub[signo * (base - sub['media']) > tolerancia]
            agrupadas = {par: [_nivel(v) for v in g['valor']] for par, g in caen.groupby('parametro', sort=False)}
            firmes = [f'`{par}`' for par in dict.fromkeys(sub['parametro']) if par not in agrupadas]
            frase += f'La configuración final obtiene {num(base, d)} ± {num(tolerancia, d)}. '
            if agrupadas:
                frase += ('Pierden más de una desviación entre pliegues respecto de ella: '
                          + '; '.join(f'`{par}` = {_lista(vs)}' for par, vs in agrupadas.items()) + '. ')
            if firmes:
                frase += f'Los demás valores, y todos los de {_lista(firmes)}, quedan dentro de esa desviación.'
        partes.append(frase.strip())
    texto = '**Lectura.** ' + ' '.join(partes)
    texto += (' La comparación con el benchmark es homogénea: los dos modelos usan hiperparámetros finales, que se eligieron con toda la '
              'partición de entrenamiento, y por eso estas cifras son algo optimistas para ambos; la estimación del desempeño es la de la '
              'validación anidada (tablas 6.5 y 6.6).')
    try:
        usa = _uso_del_puntaje(_hechos())
        if usa['sin']:
            texto += (f' En clasificación, además, aquí el modelo original siempre usa el puntaje, mientras que en la validación anidada la '
                      f'búsqueda eligió no usarlo en {usa["sin"]} de {usa["sin"] + usa["con"]} pliegues (sección 6.8).')
    except (FileNotFoundError, KeyError):
        pass
    return texto


def ruido() -> None:
    r = pd.read_csv(R / 'robustez' / 'ruido.csv')
    partes = []
    for tarea in ('regresion', 'clasificacion'):
        d = DEC[tarea]
        piv = r[r.tarea == tarea].pivot_table(index=['escenario', 'nivel'], columns='modelo', values='media', sort=False)
        mejores = [(f.idxmax() if MAYOR[tarea] else f.idxmin()) for _, f in piv.iterrows()]
        filas = [[esc, num(niv, 3)] + [num(f[m], d) for m in TRES] + [N[mej]] for ((esc, niv), f), mej in zip(piv.iterrows(), mejores)]
        partes.append(f'**{TAREA[tarea]}.** El original es el mejor de los tres en {sum(m == plan.ORIGINAL for m in mejores)} de {len(mejores)} '
                      f'escenarios.\n\n' + tabla_md(['Contaminación', 'Nivel'] + [N[m] for m in TRES] + ['Mejor'], filas, 'lrrrrl'))
    escribir('tabla_ruido.md', '\n'.join(partes) + '\nNivel: desviación del ruido añadido al puntaje de entrenamiento; fracción de puntajes de entrenamiento '
             'reemplazados por un valor al azar; o desviación del ruido en los predictores numéricos, en desviaciones típicas de cada uno. '
             'La evaluación siempre es contra el puntaje y la etiqueta sin contaminar.\n\n' + _lectura_ruido(r) + '\n')


def _lectura_ruido(r: pd.DataFrame) -> str:
    """Con qué contaminación el original deja de ser el mejor de los tres y si las diferencias superan la variación entre pliegues."""
    partes = []
    escenarios = dentro = favor = 0
    contra = []
    otros = [m for m in TRES if m != plan.ORIGINAL]
    for tarea in ('regresion', 'clasificacion'):
        d, sub = DEC[tarea], r[r.tarea == tarea]
        piv = sub.pivot_table(index=['escenario', 'nivel'], columns='modelo', values='media', sort=False)
        desv = sub[sub.modelo == plan.ORIGINAL].set_index(['escenario', 'nivel'])['desv']
        rival = piv[otros].max(axis=1) if MAYOR[tarea] else piv[otros].min(axis=1)
        ventaja = (piv[plan.ORIGINAL] - rival) * (1.0 if MAYOR[tarea] else -1.0)
        escenarios += len(piv)
        dentro += int((ventaja.abs() < desv.reindex(piv.index)).sum())
        favor += int((ventaja >= desv.reindex(piv.index)).sum())
        contra += [f'{esc} de {num(niv, 3)} en {NOMBRE_TAREA[tarea]}' for (esc, niv), x in (-ventaja >= desv.reindex(piv.index)).items() if x]
        frases = []
        for esc in dict.fromkeys(piv.index.get_level_values(0)):
            if esc == 'sin contaminar':
                continue
            g, v = piv.loc[esc], ventaja.loc[esc]
            alto = g.index.max()
            quien = g.loc[alto, otros].idxmax() if MAYOR[tarea] else g.loc[alto, otros].idxmin()
            if (v > 0).all():
                cuando = f'es el mejor de los tres en los {len(g)} niveles'
            elif not (v > 0).any():
                cuando = f'no es el mejor en ninguno de los {len(g)} niveles'
            else:
                cuando = (f'es el mejor con {_lista(num(n, 3) for n in g.index[(v > 0).to_numpy()])} y no con '
                          f'{_lista(num(n, 3) for n in g.index[(v <= 0).to_numpy()])}')
            frases.append(f'con {esc}, {cuando} (en el nivel más alto, {num(g.loc[alto, plan.ORIGINAL], d)} frente a {num(g.loc[alto, quien], d)} de {N[quien]})')
        partes.append(f'En {NOMBRE_TAREA[tarea]}, el original, ' + '; '.join(frases) + '.')
    texto = '**Lectura.** ' + ' '.join(partes)
    texto += (f' En {dentro} de los {escenarios} escenarios de las dos tareas la diferencia entre el original y el mejor de los otros dos es menor que '
              f'la desviación del original entre pliegues; la supera a favor del original en {favor} y en contra en {len(contra)}'
              + (f' ({_lista(contra)})' if contra else '') + '. El estudio muestra que una contaminación moderada no rompe al modelo, no que sea '
              'más robusto que los benchmarks.')
    reg = r[(r.tarea == 'regresion') & (r.escenario == 'ruido gaussiano en el objetivo')].pivot(index='nivel', columns='modelo', values='media')
    pierde = reg.index[(reg[plan.ORIGINAL] >= reg[otros].min(axis=1)).to_numpy()]
    ruta = R / 'techo.json'
    if len(pierde) and ruta.exists():
        delta = float(json.loads(ruta.read_text(encoding='utf-8'))['delta'])
        texto += (f' En regresión la ventaja se pierde cuando el ruido en el puntaje llega a {num(pierde.min(), 3)}: los niveles del escalón están '
                  f'separados {num(delta, 2)}, y la Proposición 4 solo promete ganancia mientras esa separación sea grande frente al ruido.')
    return texto


SIN_MEZCLA = 'BDL sin mezcla (K = 1)'


def _lectura_sin_mezcla(mu: pd.DataFrame, k1: pd.DataFrame) -> str:
    """Qué parte de la ventaja con pocas filas se debe a la clase latente (curva de aprendizaje sin la mezcla)."""
    frases, como_benchmark = [], []
    for tarea in ('regresion', 'clasificacion'):
        d, signo = DEC[tarea], (1.0 if MAYOR[tarea] else -1.0)
        piv = mu[mu.tarea == tarea].pivot(index='n', columns='modelo', values='media').sort_index()
        s = k1[k1.tarea == tarea].set_index('n')['media'].sort_index()
        poco, todo = piv.index.min(), piv.index.max()
        otros = piv[[m for m in TRES if m != plan.ORIGINAL]]
        rival = otros.loc[poco].idxmax() if MAYOR[tarea] else otros.loc[poco].idxmin()
        completo, sin, bench = piv.loc[poco, plan.ORIGINAL], s[poco], piv.loc[poco, rival]
        como_benchmark.append(abs(sin - bench) < abs(sin - completo))
        frase = (f'en {NOMBRE_TAREA[tarea]}, con {entero(poco)} filas el modelo completo obtiene {num(completo, d)}; sin la mezcla, {num(sin, d)}, '
                 f'y {N[rival]}, {num(bench, d)}')
        alcanza = [n for n in piv.index if signo * (piv.loc[n, plan.ORIGINAL] - s[todo]) >= 0]
        if alcanza and min(alcanza) < todo:
            frase += (f' (con las {entero(todo)}, la variante sin mezcla llega a {num(s[todo], d)}, que el modelo completo ya supera con '
                      f'{entero(min(alcanza))})')
        frases.append(frase)
    texto = '**Qué parte se debe a la mezcla.** La misma curva para el modelo original con $K = 1$ (línea discontinua): ' + '; '.join(frases) + '. '
    if all(como_benchmark):
        texto += ('Sin la clase latente, el modelo se comporta como los benchmarks: la ventaja con pocas filas la da la mezcla, que es lo que '
                  'predice la Proposición 4.')
    elif any(como_benchmark):
        texto += 'Sin la clase latente la ventaja con pocas filas se pierde en una de las dos tareas, no en las dos.'
    else:
        texto += ('Sin la clase latente el modelo conserva la mayor parte de la ventaja con pocas filas: no se debe a la mezcla, sino al resto '
                  'del modelo.')
    ruta = R / 'ablacion.csv'
    if ruta.exists():
        a = pd.read_csv(ruta)
        a = a[a.variante == 'bdl_sin_mezcla'].set_index('tarea')['aporte_cv']
        if {'regresion', 'clasificacion'} <= set(a.index):
            texto += (f' La variante usa aquí los hiperparámetros del modelo completo; en la ablación de la sección 5.9, donde tiene su propia '
                      f'búsqueda, la diferencia con todas las filas es de {num(a["regresion"], 6)} de RMSE y {num(a["clasificacion"], 5)} de AUC. '
                      f'Este análisis se añadió después de ver los resultados y es exploratorio (sección 6.2).')
    return texto


def muestra() -> None:
    carpeta = R / 'robustez'
    mu = pd.read_csv(carpeta / 'muestra.csv')
    k1 = pd.read_csv(carpeta / 'muestra_sin_mezcla.csv') if (carpeta / 'muestra_sin_mezcla.csv').exists() else None
    fig, ejes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    colores = {'bdl': NARANJA, 'xgboost': AZUL, 'lightgbm': VERDE}
    for ax, tarea in zip(ejes[:2], ('regresion', 'clasificacion')):
        for modelo in TRES:
            sub = mu[(mu.tarea == tarea) & (mu.modelo == modelo)]
            ax.errorbar(sub['n'], sub['media'], yerr=sub['desv'], marker='o', markersize=3.5, linewidth=1.2, capsize=2, color=colores[modelo], label=N[modelo])
        if k1 is not None:
            sub = k1[k1.tarea == tarea]
            ax.plot(sub['n'], sub['media'], marker='s', markersize=3, linewidth=1.1, linestyle='--', color=NARANJA, label=SIN_MEZCLA)
        ax.set_xscale('log')
        ax.set_xlabel('Filas de entrenamiento')
        ax.set_ylabel(METRICA[tarea])
        ax.set_title(TAREA[tarea].split(' (')[0], loc='left', fontsize=10)
    manijas, rotulos = ejes[0].get_legend_handles_labels()
    orden = sorted(range(len(rotulos)), key=lambda i: rotulos[i] == SIN_MEZCLA)          # la variante, al final
    ejes[0].legend([manijas[i] for i in orden], [rotulos[i] for i in orden], frameon=False, fontsize=7.5)
    for modelo in TRES:
        sub = mu[(mu.tarea == 'regresion') & (mu.modelo == modelo)]
        ejes[2].plot(sub['n'], sub['ajuste_s'], marker='o', markersize=3.5, color=colores[modelo], label=N[modelo])
    ejes[2].set_xscale('log')
    ejes[2].set_yscale('log')
    ejes[2].set_xlabel('Filas de entrenamiento')
    ejes[2].set_ylabel('Tiempo de ajuste (s)')
    ejes[2].set_title('Costo (regresión)', loc='left', fontsize=10)
    from matplotlib.ticker import NullFormatter
    filas_n = sorted(mu['n'].unique())
    for ax in ejes:                                                     # marcas solo en los tamaños medidos
        ax.set_xticks(filas_n)
        ax.set_xticklabels([f'{n / 1000:.0f} mil' for n in filas_n], fontsize=8)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='x', which='minor', length=0)
    marcas = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if mu['ajuste_s'].min() / 1.5 <= t <= mu['ajuste_s'].max() * 1.5]
    ejes[2].set_yticks(marcas)
    ejes[2].set_yticklabels([num(t, 1) for t in marcas])
    ejes[2].yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    _guardar(fig, 'aprendizaje.png')
    plt.close(fig)
    filas, filas_tiempo, notas = [], [], []
    for tarea in ('regresion', 'clasificacion'):
        d = DEC[tarea]
        sub_t = mu[mu.tarea == tarea]
        for n, sub in sub_t.groupby('n'):
            sub = sub.set_index('modelo')
            sin = [] if k1 is None else [(lambda f: f'{num(f["media"], d)} ± {num(f["desv"], d)}')(k1[(k1.tarea == tarea) & (k1.n == n)].iloc[0])]
            filas.append([TAREA[tarea].split(' (')[0], entero(n)] + [f'{num(sub.loc[m, "media"], d)} ± {num(sub.loc[m, "desv"], d)}' for m in TRES] + sin)
            filas_tiempo.append([TAREA[tarea].split(' (')[0], entero(n)] + [num(sub.loc[m, 'ajuste_s'], 1) for m in TRES])
        piv = sub_t.pivot(index='n', columns='modelo', values='media').sort_index()
        todo, poco = piv.index.max(), piv.index.min()
        rival = (piv.loc[todo, ['xgboost', 'lightgbm']].idxmax() if MAYOR[tarea] else piv.loc[todo, ['xgboost', 'lightgbm']].idxmin())
        meta = piv.loc[todo, rival]
        alcanza = [n for n in piv.index if (piv.loc[n, 'bdl'] >= meta if MAYOR[tarea] else piv.loc[n, 'bdl'] <= meta)]
        frase = (f'**{TAREA[tarea].split(" (")[0]}.** Con {entero(poco)} filas (el {num(100 * poco / todo, 0)} %), el original obtiene {num(piv.loc[poco, "bdl"], d)} '
                 f'y {N[rival]}, {num(piv.loc[poco, rival], d)}; con las {entero(todo)}, {num(piv.loc[todo, "bdl"], d)} y {num(meta, d)}. ')
        if alcanza and min(alcanza) < todo:
            frase += (f'El original alcanza con {entero(min(alcanza))} filas el valor que {N[rival]} obtiene con todas: la ventaja del modelo es de '
                      f'eficiencia muestral y se reduce al crecer la muestra, cuando todos se acercan al límite.')
        elif alcanza:
            frase += 'La ventaja del original se mantiene con todas las filas.'
        else:
            frase += f'Con todas las filas, {N[rival]} queda por delante: la ventaja del original aparece con muestras pequeñas y desaparece al crecer la muestra.'
        notas.append(frase)
    tiempos = mu[mu.tarea == 'regresion'].pivot(index='n', columns='modelo', values='ajuste_s').sort_index()
    poco, todo = tiempos.index.min(), tiempos.index.max()
    notas.append(f'**Costo.** Al multiplicar las filas por {num(todo / poco, 0)}, el tiempo de ajuste del original se multiplica por '
                 f'{num(tiempos.loc[todo, "bdl"] / tiempos.loc[poco, "bdl"], 1)} (el de XGBoost, por {num(tiempos.loc[todo, "xgboost"] / tiempos.loc[poco, "xgboost"], 1)}): '
                 f'crece como mucho de forma lineal, como predice la sección 5.7. Con todas las filas el original tarda '
                 f'{num(tiempos.loc[todo, "bdl"] / tiempos.loc[todo, "xgboost"], 1)} veces lo que XGBoost.')
    if k1 is not None:
        notas.insert(len(notas) - 1, _lectura_sin_mezcla(mu, k1))
    escribir('tabla_muestra.md', '**Métrica: RMSE en regresión y AUC en clasificación.**\n\n'
             + tabla_md(['Tarea', 'Filas'] + [N[m] for m in TRES] + ([SIN_MEZCLA] if k1 is not None else []), filas, 'lr' + 'r' * (len(filas[0]) - 2))
             + '\n**Tiempo de ajuste, en segundos.**\n\n' + tabla_md(['Tarea', 'Filas'] + [N[m] for m in TRES], filas_tiempo, 'lrrrr')
             + '\nMedia ± desviación en los cinco pliegues externos de la semilla 42, con los hiperparámetros finales; las filas se submuestrean '
               'dentro de cada pliegue de entrenamiento y la evaluación usa el pliegue de prueba completo. Ningún modelo se reajusta para cada '
               'tamaño: parte de la diferencia con pocas filas puede deberse a que los hiperparámetros se eligieron con todas.\n\n'
             + '\n\n'.join(notas) + '\n')
    co = pd.read_csv(carpeta / 'columnas.csv').pivot(index='columnas', columns='modelo', values='ajuste_s').sort_index()
    poco, todo = co.index.min(), co.index.max()
    escribir('tabla_columnas.md', tabla_md(['Columnas originales', f'Ajuste {N["bdl"]} (s)', 'Ajuste XGBoost (s)'],
                                           [[int(k), num(f['bdl'], 1), num(f['xgboost'], 1)] for k, f in co.iterrows()], 'rrr')
             + f'\nAl multiplicar las columnas por {num(todo / poco, 0)}, el tiempo del original se multiplica por {num(co.loc[todo, "bdl"] / co.loc[poco, "bdl"], 1)} '
               f'y el de XGBoost, por {num(co.loc[todo, "xgboost"] / co.loc[poco, "xgboost"], 1)}.\n')
    ruta = R / 'inferencia_filas.csv'
    if ruta.exists():
        inf_filas = pd.read_csv(ruta)
        piv = inf_filas.pivot(index='filas', columns='modelo', values='inferencia_s').sort_index()
        modelos = [m for m in (plan.ORIGINAL, *[x for x in piv.columns if x != plan.ORIGINAL]) if m in piv.columns]
        poco, todo = piv.index.min(), piv.index.max()
        escribir('tabla_inferencia.md', tabla_md(['Filas'] + [f'Inferencia {N[m]} (s)' for m in modelos] + [f'{N[m]}: microsegundos por fila' for m in modelos],
                                                 [[entero(n)] + [num(f[m], 3) for m in modelos] + [num(1e6 * f[m] / n, 1) for m in modelos]
                                                  for n, f in piv.iterrows()], 'r' * (1 + 2 * len(modelos)))
                 + f'\nMediana de cinco medidas; la tubería incluye el preprocesamiento. Al multiplicar las filas por {num(todo / poco, 0)}, el '
                   f'tiempo de inferencia del original se multiplica por {num(piv.loc[todo, plan.ORIGINAL] / piv.loc[poco, plan.ORIGINAL], 1)}: '
                   'crece de forma lineal o menos, como predice la sección 5.7.\n')
    _aviso('   figuras/aprendizaje.png')


def ciudades() -> None:
    ci = pd.read_csv(R / 'robustez' / 'ciudades.csv')
    ruta = R / 'robustez' / 'muestra.csv'
    dentro = pd.read_csv(ruta) if ruta.exists() else None                # misma configuración, pliegues externos al azar
    partes, lectura, degradados, comparados = [], [], [], False
    for tarea in ('regresion', 'clasificacion'):
        d = DEC[tarea]
        piv = ci[ci.tarea == tarea].pivot(index='ciudad', columns='modelo', values='metrica')
        n = ci[ci.tarea == tarea].groupby('ciudad')['n'].first()
        otros = [m for m in TRES if m != plan.ORIGINAL]
        gana = int(((piv['bdl'] > piv[otros].max(axis=1)) if MAYOR[tarea] else (piv['bdl'] < piv[otros].min(axis=1))).sum())
        filas = [[c, entero(n[c])] + [num(f[m], d) for m in TRES] for c, f in piv.iterrows()]
        filas.append(['**Media**', '—'] + [f'**{num(piv[m].mean(), d)}**' for m in TRES])
        partes.append(f'**{TAREA[tarea]}.** El original es el mejor de los tres en {gana} de {len(piv)} ciudades.\n\n'
                      + tabla_md(['Ciudad que se deja fuera', 'Clientes'] + [N[m] for m in TRES], filas, 'lrrrr'))
        medias = piv[list(TRES)].mean()
        quien = medias[otros].idxmax() if MAYOR[tarea] else medias[otros].idxmin()
        frase = (f'En {NOMBRE_TAREA[tarea]}, la media simple de las {len(piv)} ciudades es {num(medias[plan.ORIGINAL], d)} para el original y '
                 f'{num(medias[quien], d)} para {N[quien]}')
        if dentro is not None:
            ref = dentro[(dentro.tarea == tarea) & (dentro.fraccion == 1.0)].set_index('modelo')
            if set(TRES) <= set(ref.index):
                peor = {m: bool((medias[m] < ref.loc[m, 'media'] - ref.loc[m, 'desv']) if MAYOR[tarea] else
                                (medias[m] > ref.loc[m, 'media'] + ref.loc[m, 'desv'])) for m in TRES}
                degradados.extend(N[m] for m, p in peor.items() if p)
                comparados = True
                frase += (f'; con pliegues al azar y la misma configuración, el original obtiene {num(ref.loc[plan.ORIGINAL, "media"], d)} ± '
                          f'{num(ref.loc[plan.ORIGINAL, "desv"], d)}'
                          + (': evaluar en una ciudad que no se vio empeora su resultado' if peor[plan.ORIGINAL] else
                             ': evaluar en una ciudad que no se vio no empeora su resultado'))
        lectura.append(frase + '.')
    pequenas = ci.groupby('ciudad')['n'].first()
    cierre = (f' Las ciudades pequeñas pesan lo mismo que las grandes en esa media, y en las de pocos cientos de clientes '
              f'({int((pequenas < 500).sum())} de {len(pequenas)} tienen menos de 500) las diferencias entre modelos son ruido de muestreo.')
    if comparados and not degradados:
        cierre += (' Como dejar fuera una ciudad no degrada a ninguno de los tres modelos, en COFINFAD la relación entre los predictores y el '
                   'puntaje no parece cambiar de una ciudad a otra: este estudio no es una prueba exigente de cambio de distribución.')
    elif comparados:
        cierre += f' Dejar fuera una ciudad degrada a {_lista(dict.fromkeys(degradados))} más de una desviación entre pliegues.'
    escribir('tabla_ciudades.md', '\n'.join(partes) + '\n**Lectura.** ' + ' '.join(lectura) + cierre + '\n')


def semillas() -> None:
    s = pd.read_csv(R / 'semillas.csv')
    fig, ejes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    filas = []
    from scipy import stats
    memoria = pd.read_csv(R / 'memoria.csv') if (R / 'memoria.csv').exists() else None
    filas_computo = []
    for ax, tarea in zip(ejes, ('regresion', 'clasificacion')):
        sub = s[s.tarea == tarea]
        orden = sub.groupby('modelo')['metrica'].mean().sort_values(ascending=not MAYOR[tarea]).index.tolist()
        principales = orden[:6]
        cajas = ax.boxplot([sub[sub.modelo == m]['metrica'] for m in principales], vert=False, patch_artist=True, widths=0.6,
                           medianprops={'color': 'white'})
        for caja, m in zip(cajas['boxes'], principales):
            caja.set(facecolor=NARANJA if m == plan.ORIGINAL else AZUL, edgecolor='none', alpha=0.9)
        ax.set_yticklabels([N[m] for m in principales], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(f'{METRICA[tarea]} en prueba, diez semillas')
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title(TAREA[tarea].split(' (')[0], loc='left', fontsize=10)
        for m in orden:
            x = sub[sub.modelo == m]
            media, desv, n = x['metrica'].mean(), x['metrica'].std(ddof=1), len(x)
            mitad = float(stats.t.ppf(0.975, n - 1) * desv / np.sqrt(n)) if n > 1 else 0.0
            d1 = DEC[tarea] + 1
            filas.append([TAREA[tarea].split(' (')[0], N[m], num(media, d1), num(desv, d1), f'[{num(media - mitad, d1)}; {num(media + mitad, d1)}]',
                          num(x['metrica'].min(), d1), num(x['metrica'].max(), d1)])
            mm = memoria[(memoria.tarea == tarea) & (memoria.modelo == m)] if memoria is not None else []
            filas_computo.append([TAREA[tarea].split(' (')[0], N[m], num(x['ajuste_s'].mean(), 1), num(x['inferencia_s'].mean(), 2),
                                  entero(mm.iloc[0]['memoria_mb']) if len(mm) else '—', entero(x['tamano_kb'].mean())])
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    _guardar(fig, 'semillas.png')
    plt.close(fig)
    escribir('tabla_semillas.md', '**Estabilidad: métrica en prueba con diez semillas.**\n\n'
             + tabla_md(['Tarea', 'Modelo', 'Media', 'Desviación', 'IC 95 % de la media', 'Mínimo', 'Máximo'], filas, 'llrrrrr')
             + '\n**Cómputo de cada modelo final.**\n\n'
             + tabla_md(['Tarea', 'Modelo', 'Ajuste (s)', 'Inferencia (s)', 'Memoria (MB)', 'Tamaño (KB)'], filas_computo, 'llrrrr')
             + '\nDiez semillas (42 a 51) con los hiperparámetros finales; solo cambia la semilla del modelo, no la partición ni la búsqueda. IC 95 % '
               'de la media: con la *t* de Student sobre las diez semillas. Los modelos deterministas dan el mismo valor las diez veces. Ajuste e '
               'inferencia: media de las diez semillas; la inferencia es sobre los 9.745 clientes de prueba. Memoria: aumento máximo del conjunto de '
               'trabajo del proceso al ajustar el modelo y predecir la prueba, medido en un proceso aparte (`experiments/03c_memoria.py`). Tamaño: '
               'el modelo serializado y comprimido.\n')
    _aviso('   figuras/semillas.png')


def _nota_empates_calibracion() -> str:
    """Aviso de la tabla de calibración: dos de sus columnas dependen de los empates de la probabilidad del original."""
    ruta = R / 'empates_campana.json'
    if not ruta.exists():
        return ''
    e = json.loads(ruta.read_text(encoding='utf-8'))
    o = e['modelos'][plan.ORIGINAL]
    if o['empatados'] <= 1:
        return ''
    return (f'\n**Dos columnas dependen de los empates.** La probabilidad del original toma {entero(o["valores_distintos"])} valores distintos, y '
            f'{entero(o["empatados"])} clientes comparten la del corte de la campaña (Tabla 6.11). Su captura al 25 % es la del desempate del '
            f'protocolo; con un desempate al azar se espera {num(100 * o["captura_esperada"], 1)} % ± {num(100 * o["captura_desviacion"], 1)} '
            f'puntos. Su precisión media es la de la convención de scikit-learn, que cuenta a los empatados como un solo punto; con los empates '
            f'rotos al azar es {num(o["precision_media_azar"], 4)} ± {num(o["precision_media_azar_desviacion"], 4)}.\n')


def calibracion() -> None:
    m = pd.read_csv(R / 'metricas_prueba.csv')
    m = m[m.tarea == 'clasificacion'].sort_values('auc', ascending=False)
    filas = [[f'**{N[f.modelo]}**' if f.modelo == plan.ORIGINAL else N[f.modelo], num(f.auc, 4), num(f.ap, 4), num(100 * f.captura_25, 1) + ' %',
              num(f.logloss, 4), num(f.brier, 4), num(f.ece, 4), num(f.logloss_isotonica, 4), num(f.brier_isotonica, 4), num(f.ece_isotonica, 4)]
             for f in m.itertuples()]
    escribir('tabla_calibracion.md', tabla_md(['Modelo', 'AUC', 'Precisión media', 'Captura al 25 %', 'Pérdida log.', 'Brier', 'ECE',
                                               'Pérdida log. (isotónica)', 'Brier (isotónica)', 'ECE (isotónica)'], filas)
             + '\nECE: error de calibración esperado con diez tramos de igual número de clientes; el dashboard lo calcula con diez tramos de igual ancho, como el Entregable 3, y por eso sus cifras no coinciden exactamente con las de esta tabla. Isotónica: recalibración ajustada con las '
               'predicciones fuera de pliegue de entrenamiento. El SVM lineal no da probabilidades: solo se reporta recalibrado.\n'
             + _nota_empates_calibracion())
    pred = pd.read_parquet(R / 'predicciones_prueba.parquet')
    y = _problema()['y_test'].to_numpy()
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    mejor = v['clasificacion']['mejor_benchmark']
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GRIS, linewidth=0.8)
    for clave, nombre, color in ((f'clasificacion__{plan.ORIGINAL}', N[plan.ORIGINAL], NARANJA),
                                 (f'clasificacion__{mejor}__isotonica', f'{N[mejor]} (isotónica)', AZUL),
                                 (f'clasificacion__{mejor}', f'{N[mejor]} (sin recalibrar)', GRIS)):
        prob = pred[clave].to_numpy()
        orden = np.argsort(prob)
        xs = [prob[t].mean() for t in np.array_split(orden, 10)]
        ys = [y[t].mean() for t in np.array_split(orden, 10)]
        ax.plot(xs, ys, marker='o', markersize=3.5, linewidth=1.2, color=color, label=nombre)
    ax.set_xlabel('Probabilidad predicha (media del tramo)')
    ax.set_ylabel('Frecuencia observada de riesgo alto')
    ax.set_title('Calibración en prueba', loc='left', fontsize=10.5)
    ax.legend(frameon=False, fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    _guardar(fig, 'calibracion.png')
    plt.close(fig)
    _aviso('   figuras/calibracion.png')


def incertidumbre() -> None:
    u = pd.read_csv(R / 'incertidumbre_prueba.csv')
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    mejor = v['regresion']['mejor_benchmark']
    filas = []
    for cob in sorted(u.cobertura_nominal.unique()):
        sub = u[u.cobertura_nominal == cob]
        for modelo, metodo, etiqueta in ((mejor, 'intervalo conforme (pronóstico ± q)', f'{N[mejor]}: intervalo conforme'),
                                         (plan.ORIGINAL, 'intervalo conforme (pronóstico ± q)', 'BDL: intervalo conforme con la misma receta'),
                                         (plan.ORIGINAL, 'intervalo central del modelo', 'BDL: intervalo central'),
                                         (plan.ORIGINAL, 'conjunto de predicción del modelo', 'BDL: conjunto de predicción')):
            f = sub[(sub.modelo == modelo) & (sub.metodo == metodo)]
            if len(f):
                f = f.iloc[0]
                filas.append([f'{int(100 * cob)} %', etiqueta, num(100 * f['cobertura'], 1) + ' %', num(f['largo'], 4),
                              '—' if pd.isna(f.get('tramos')) else int(f['tramos'])])
    base = u[(u.modelo == mejor) & (u.cobertura_nominal == 0.9) & (u.metodo == 'intervalo conforme (pronóstico ± q)')]['largo'].iloc[0]
    conj = u[(u.modelo == plan.ORIGINAL) & (u.cobertura_nominal == 0.9) & (u.metodo == 'conjunto de predicción del modelo')]['largo'].iloc[0]
    n = len(_problema()['y_test'])
    nota = (f'\nAl 90 %, el conjunto de predicción del modelo original mide {num(conj, 4)} y el intervalo del mejor benchmark, {num(base, 4)}: '
            f'**{num(base / conj, 0)} veces más corto** con la misma cobertura nominal. Con {entero(n)} clientes de prueba, el error de '
            f'muestreo de una cobertura del 90 % es de ±{num(100 * 1.96 * np.sqrt(0.9 * 0.1 / n), 1)} puntos.\n')
    receta = u[(u.modelo == plan.ORIGINAL) & (u.metodo == 'intervalo conforme (pronóstico ± q)')].set_index('cobertura_nominal')['cobertura']
    desviadas = [(c, x) for c, x in receta.items() if abs(x - c) > 3 * np.sqrt(c * (1 - c) / n)]
    if desviadas:
        nota += ('\n**La receta «pronóstico ± q» no sirve para el modelo original.** Su cobertura medida fue '
                 + ' y '.join(f'{num(100 * x, 1)} % para el {int(round(100 * c))} % nominal' for c, x in desviadas)
                 + '. La causa no está en el modelo sino en la receta: cuando el error tiene escalones, el cuantil *q* cae justo sobre uno de '
                   'ellos, y que ese nivel entero quede dentro o fuera del intervalo depende de diferencias de diezmilésimas entre el modelo '
                   'final y los modelos de los pliegues con los que se calculó *q*. En los benchmarks no se nota porque su propio error '
                   'difumina los escalones. Los conjuntos de predicción del modelo no tienen ese problema: se construyen con la densidad '
                   'del error, no con una distancia al pronóstico.\n')
    homogenea = _referencias_homogeneas()
    if homogenea and homogenea['benchmark'] == mejor:
        nuevas = []
        for fila in filas:                                               # tras el intervalo del benchmark, su conjunto por densidad
            nuevas.append(fila)
            if fila[1] == f'{N[mejor]}: intervalo conforme':
                x = homogenea['conjuntos']['por_cobertura'][f'{int(fila[0].split()[0]) / 100:.2f}']['benchmark_conjunto']
                nuevas.append([fila[0], f'{N[mejor]}: conjunto por densidad de sus residuos (exploratorio)', num(100 * x['cobertura'], 1) + ' %',
                               num(x['largo'], 4), x['tramos']])
        filas = nuevas
        x = homogenea['conjuntos']['por_cobertura']['0.90']['benchmark_conjunto']
        nota += (f'\n**Con la misma receta para los dos.** El conjunto del original es un conjunto por densidad y el intervalo del benchmark es '
                 f'simétrico, así que parte de la diferencia es de receta. Una mezcla ajustada a los residuos fuera de pliegue de {N[mejor]} '
                 f'también encuentra {homogenea["conjuntos"]["clases_de_la_mezcla_del_benchmark"]} niveles, y el conjunto por densidad que resulta '
                 f'mide {num(x["largo"], 4)} al 90 % (cobertura en prueba: {num(100 * x["cobertura"], 1)} %). Frente a esa referencia, el conjunto '
                 f'del original es {num(x["largo"] / conj, 0)} veces más corto: esa es la parte atribuible a que estima mejor el índice. Es una '
                 f'comparación exploratoria, añadida después de ver los resultados (`experiments/04e_referencias_homogeneas.py`).\n')
    escribir('tabla_incertidumbre.md', tabla_md(['Cobertura nominal', 'Método', 'Cobertura en prueba', 'Largo', 'Tramos'], filas, 'llrrr') + nota)
    p = _problema()
    tuberia = _final('regresion')
    b = tuberia.named_steps['modelo']
    fila = p['conjuntos'].X_test.iloc[[0]]
    F = float(b.indice(tuberia[:-1].transform(fila))[0])
    s_real = float(p['conjuntos'].y_test_cont.iloc[0])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    ax.hist(F + b.residuos_calibracion_, bins=220, color=AZUL, alpha=0.9)
    techo_y = ax.get_ylim()[1]
    for a, c in b.conjunto_residual(0.9):
        ax.axvspan(F + a, F + c, color=NARANJA, alpha=0.45, linewidth=0)
        ax.plot([F + (a + c) / 2], [techo_y * 0.99], marker='v', color=NARANJA, markersize=6, clip_on=False)
    q = float(u[(u.modelo == mejor) & (u.cobertura_nominal == 0.9) & (u.metodo == 'intervalo conforme (pronóstico ± q)')]['largo'].iloc[0]) / 2
    centro = float(pd.read_parquet(R / 'predicciones_prueba.parquet').loc[fila.index[0], f'regresion__{mejor}'])   # pronóstico del benchmark
    ax.hlines(techo_y * 0.88, centro - q, centro + q, color=GRIS, linewidth=3)
    ax.axvline(s_real, color=VERDE, linewidth=1.4)
    ax.axvline(p['etiqueta'].umbral, color='black', linestyle='--', linewidth=0.9)
    ax.set_xlabel('Puntaje de fuga')
    ax.set_ylabel('Clientes de calibración')
    ax.set_title('Distribución predictiva del modelo original para un cliente de prueba', loc='left', fontsize=10.5)
    ax.legend(handles=[Patch(facecolor=AZUL, label='niveles posibles del puntaje\n(residuos de calibración)'),
                       Line2D([0], [0], color=NARANJA, marker='v', linewidth=5, alpha=0.6, label='conjunto de predicción al 90 %'),
                       Line2D([0], [0], color=GRIS, linewidth=3, label=f'intervalo al 90 % de {N[mejor]}'),
                       Line2D([0], [0], color=VERDE, linewidth=1.4, label='valor real'),
                       Line2D([0], [0], color='black', linestyle='--', linewidth=0.9, label='umbral de riesgo alto')],
              frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    FIG.mkdir(parents=True, exist_ok=True)
    _guardar(fig, 'distribucion_predictiva.png')
    plt.close(fig)
    _aviso('   figuras/distribucion_predictiva.png')


def errores() -> None:
    p = _problema()
    c = p['conjuntos']
    pred = pd.read_parquet(R / 'predicciones_prueba.parquet')
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    y_bin, y_cont, X = p['y_test'].to_numpy(), c.y_test_cont.to_numpy(), c.X_test
    from sklearn.metrics import roc_auc_score
    indice = pred['regresion__bdl__indice'].to_numpy() if 'regresion__bdl__indice' in pred else None
    grupos = {'Todos': np.ones(len(X), dtype=bool)}
    for edad in sorted(X['age'].unique()):
        grupos[f'Edad {int(edad)}'] = (X['age'] == edad).to_numpy()
    grupos['tx_count > 500 (cola)'] = (X['tx_count'] > 500).to_numpy()
    grupos['tx_count ≤ 500'] = (X['tx_count'] <= 500).to_numpy()
    for seg in sorted(X['customer_segment'].unique()):
        grupos[f'Segmento {seg}'] = (X['customer_segment'] == seg).to_numpy()
    mr, mc = v['regresion']['mejor_benchmark'], v['clasificacion']['mejor_benchmark']
    filas, registro = [], []
    for nombre, g in grupos.items():
        rb = float(np.sqrt(np.mean((y_cont[g] - pred[f'regresion__{plan.ORIGINAL}'].to_numpy()[g]) ** 2)))
        rm = float(np.sqrt(np.mean((y_cont[g] - pred[f'regresion__{mr}'].to_numpy()[g]) ** 2)))
        if 0 < y_bin[g].sum() < g.sum():
            ab = roc_auc_score(y_bin[g], pred[f'clasificacion__{plan.ORIGINAL}'].to_numpy()[g])
            am = roc_auc_score(y_bin[g], pred[f'clasificacion__{mc}'].to_numpy()[g])
        else:
            ab = am = float('nan')
        filas.append([nombre, entero(g.sum()), num(rb, 5), num(rm, 5), 'original' if rb < rm else 'benchmark', num(ab, 4), num(am, 4),
                      '—' if np.isnan(ab) else ('original' if ab > am else 'benchmark')])
        registro.append({'subgrupo': nombre, 'clientes': int(g.sum()), 'rmse_original': rb, 'rmse_benchmark': rm, 'benchmark_regresion': mr,
                         'auc_original': ab, 'auc_benchmark': am, 'benchmark_clasificacion': mc})
    pd.DataFrame(registro).to_csv(R / 'errores_subgrupos.csv', index=False)
    escribir('tabla_errores.md', tabla_md(['Subgrupo de prueba', 'Clientes', 'RMSE original', f'RMSE {N[mr]}', 'Mejor', 'AUC original', f'AUC {N[mc]}', 'Mejor'],
                                          filas, 'lrrrlrrl'))


# ------------------------------------------------------------------ textos que dependen de los resultados
CORTO = {'regresion': {'P1': 'Wilcoxon', 'P2': '*t* corregida', 'P3': 'permutación pareada', 'P4': 'bootstrap pareado'},
         'clasificacion': {'P1': 'Wilcoxon', 'P2': '*t* corregida', 'P3': 'DeLong', 'P4': 'McNemar'}}
NOMBRE_TAREA = {'regresion': 'regresión', 'clasificacion': 'clasificación'}
RECETA = 'intervalo conforme (pronóstico ± q)'


def _lista(xs) -> str:
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' y ' + xs[-1]


def _cuantas(k: int, n: int) -> str:
    """«ninguna de las 4», «1 de las 4», «las 4»."""
    return f'ninguna de las {n}' if k == 0 else f'las {n}' if k == n else f'{k} de las {n}'


def _p_ajustado(r: dict) -> str:
    p = r.get('p_holm')
    return 'p ajustado < 0,001' if p is not None and p < 0.001 else f'p ajustado = {pvalor(p)}'


def _hechos() -> dict:
    """Las cifras que citan los textos, leídas de los archivos de resultados."""
    v = json.loads((R / 'veredicto.json').read_text(encoding='utf-8'))
    m = pd.read_csv(R / 'metricas_prueba.csv')
    te = json.loads((R / 'techo.json').read_text(encoding='utf-8'))
    u = pd.read_csv(R / 'incertidumbre_prueba.csv')
    c4 = json.loads((R / 'condicion4.json').read_text(encoding='utf-8')) if (R / 'condicion4.json').exists() else {}
    h = {'v': v, 'm': m, 'techo': te, 'u': u, 'c4': c4, 'n_prueba': int(te['prueba']['n']), 'n_train': int(te['entrenamiento']['n'])}
    for tarea in MAYOR:
        r, col = v[tarea], ('auc' if tarea == 'clasificacion' else 'rmse')
        dm, mejor = r['frente_al_mejor'], r['mejor_benchmark']
        sub = m[m.tarea == tarea].set_index('modelo')
        orden = sub[col].sort_values(ascending=not MAYOR[tarea]).index.tolist()
        cuatro = c4.get(tarea) or {}
        t = {'r': r, 'dm': dm, 'mejor': mejor, 'nb': N[mejor], 'd': DEC[tarea], 'metrica': METRICA[tarea], 'sub': sub,
             'cv_o': r['medias'][plan.ORIGINAL], 'cv_b': r['medias'][mejor], 'ventaja': r['ventaja_sobre_el_mejor'],
             'puesto_cv': sorted(r['medias'], key=r['medias'].get, reverse=MAYOR[tarea]).index(plan.ORIGINAL) + 1,
             'pr_o': float(sub.loc[plan.ORIGINAL, col]), 'pr_b': float(sub.loc[mejor, col]), 'puesto_prueba': orden.index(plan.ORIGINAL) + 1,
             'limite': float(te['prueba']['auc_maximo' if tarea == 'clasificacion' else 'rmse_minimo']),
             'rechazan': [k for k, x in r['rechazos_frente_al_mejor'].items() if x], 'pruebas': [k for k in ('P1', 'P2', 'P3', 'P4') if k in dm],
             'c1': bool(r['condicion_1_media_mejor']), 'c2': bool(r['condicion_2_tres_pruebas']), 'c3': bool(r['condicion_3_efecto']),
             'c4': cuatro, 'c4_completa': bool(cuatro.get('completo')), 'repeticiones': len(r['repeticiones'])}
        t['tres'] = t['c1'] and t['c2'] and t['c3']
        t['superior'] = bool(t['tres'] and t['c4_completa'] and cuatro.get('cumple'))
        t['pendiente'] = bool(t['tres'] and not t['c4_completa'])
        h[tarea] = t
    # hipótesis secundarias
    mr = h['regresion']['mejor']
    al90 = u[u.cobertura_nominal == 0.9]
    base = al90[(al90.modelo == mr) & (al90.metodo == RECETA)].iloc[0]
    conj = al90[(al90.modelo == plan.ORIGINAL) & (al90.metodo == 'conjunto de predicción del modelo')].iloc[0]
    error = 1.96 * np.sqrt(0.9 * 0.1 / h['n_prueba'])
    h['h3'] = {'largo_conjunto': float(conj['largo']), 'largo_benchmark': float(base['largo']), 'cobertura': float(conj['cobertura']),
               'cobertura_benchmark': float(base['cobertura']), 'error': float(error), 'tramos': int(conj['tramos']),
               'cumple': bool(conj['largo'] < base['largo'] and conj['cobertura'] >= 0.9 - error)}
    sc = h['clasificacion']['sub']
    mc = h['clasificacion']['mejor']
    h['h4'] = {'original': float(sc.loc[plan.ORIGINAL, 'logloss']), 'benchmark': float(sc.loc[mc, 'logloss_isotonica']),
               'limite': float(te['prueba']['logloss_minima']), 'ece_original': float(sc.loc[plan.ORIGINAL, 'ece']),
               'ece_benchmark_crudo': float(sc.loc[mc, 'ece']), 'cumple': bool(sc.loc[plan.ORIGINAL, 'logloss'] < sc.loc[mc, 'logloss_isotonica'])}
    return h


def _uso_del_puntaje(h: dict) -> dict:
    """En qué pliegues externos la búsqueda del clasificador original eligió no usar el puntaje, y con qué AUC."""
    tabla = plan.registro().cargar()
    M = pd.read_csv(R / 'pliegues_clasificacion.csv', index_col=0)
    con, sin = [], []
    for semilla in h['clasificacion']['r']['repeticiones']:
        f = _fila(tabla, 'clasificacion', plan.ORIGINAL, semilla=semilla)
        for i, pr in enumerate(_json(f['params_por_pliegue'])):
            (con if pr.get('modelo__usar_puntaje', True) else sin).append(f'{semilla}-{i + 1}')
    final = _json(_fila(tabla, 'clasificacion', plan.ORIGINAL)['hiperparametros'])
    mejor = h['clasificacion']['mejor']
    medias = lambda filas, modelo: float(M.loc[filas, modelo].mean()) if filas else float('nan')   # noqa: E731
    return {'con': len(con), 'sin': len(sin), 'original_con': medias(con, plan.ORIGINAL), 'original_sin': medias(sin, plan.ORIGINAL),
            'benchmark_con': medias(con, mejor), 'benchmark_sin': medias(sin, mejor),
            'gana_con': int((M.loc[con, plan.ORIGINAL] > M.loc[con, mejor]).sum()) if con else 0,
            'final_usa_puntaje': bool(final.get('modelo__usar_puntaje', True))}


def _parrafo_veredicto(tarea: str, h: dict) -> str:
    t = h[tarea]
    r, dm, d, nb = t['r'], t['dm'], t['d'], t['nb']
    n = r['pliegues']
    rel = 100 * abs(t['ventaja']) / abs(t['cv_b'])
    texto = f'**{NOMBRE_TAREA[tarea].capitalize()} ({"H1" if tarea == "regresion" else "H2"}).** '
    if t['c1']:
        texto += (f'El {t["metrica"]} medio del modelo original en los {n} pliegues externos es {num(t["cv_o"], d)}, frente a {num(t["cv_b"], d)} de '
                  f'{nb}, el mejor benchmark: una ventaja de {num(t["ventaja"], d + 1)} ({num(rel, 2)} %), con el original por delante en '
                  f'{dm["pliegues_a_favor"]} de los {n} pliegues. La condición 1 se cumple. ')
    else:
        texto += (f'El {t["metrica"]} medio del modelo original en los {n} pliegues externos es {num(t["cv_o"], d)} y el de {nb}, el mejor benchmark, '
                  f'{num(t["cv_b"], d)}: el original queda en el puesto {t["puesto_cv"]} de 10, a {num(abs(t["ventaja"]), d + 1)} del primero, y va por '
                  f'delante en {dm["pliegues_a_favor"]} de los {n} pliegues. La condición 1 no se cumple. ')
    detalle = {k: f'{k} ({CORTO[tarea][k]}; {_p_ajustado(dm[k])})' for k in t['pruebas']}
    si, no = [detalle[k] for k in t['pruebas'] if k in t['rechazan']], [detalle[k] for k in t['pruebas'] if k not in t['rechazan']]
    if not si:
        texto += f'Ninguna de las {len(t["pruebas"])} pruebas rechaza la igualdad con {nb} a favor del original: {_lista(no)}. '
    elif not no:
        texto += f'Las {len(si)} pruebas rechazan la igualdad con {nb} a favor del original: {_lista(si)}. '
    else:
        texto += (f'De las {len(t["pruebas"])} pruebas, {"rechaza" if len(si) == 1 else "rechazan"} la igualdad con {nb} a favor del original '
                  f'{_lista(si)}; no {"rechaza" if len(no) == 1 else "rechazan"} {_lista(no)}. ')
    texto += f'La condición 2 pide tres rechazos: {"se cumple" if t["c2"] else "no se cumple"}. '
    efecto = f'δ de Cliff = {num(dm["cliff"], 2)} y *d* de Cohen = {num(dm["cohen"], 2)}'
    if t['c3']:
        texto += f'El tamaño del efecto entre los valores por pliegue ({efecto}) supera los umbrales de 0,147 y 0,2: la condición 3 se cumple. '
    elif dm['cliff'] < 0 or dm['cohen'] < 0:
        texto += f'El tamaño del efecto ({efecto}) va en contra del original: la condición 3 no se cumple. '
    else:
        texto += f'El tamaño del efecto ({efecto}) no llega a los umbrales de 0,147 y 0,2: la condición 3 no se cumple. '
    c4 = t['c4']
    if not c4:
        texto += 'La condición 4 está pendiente de los conjuntos adicionales. '
    elif not t['c4_completa']:
        texto += f'La condición 4 es provisional: con {c4["conjuntos"]} de {c4.get("esperados", 4)} conjuntos adicionales, el original es {c4["resumen"]}. '
    else:
        texto += (f'En los {c4["conjuntos"]} conjuntos adicionales el original es {c4["resumen"]}: la condición 4 '
                  f'{"se cumple" if c4["cumple"] else "no se cumple"}. ')
    texto += ('**Veredicto: superior.**' if t['superior'] else '**Veredicto: pendiente de la condición 4.**' if t['pendiente'] else '**Veredicto: no superior.**')
    # en prueba y análisis de la causa
    distancia_o, distancia_b = abs(t['pr_o'] - t['limite']), abs(t['pr_b'] - t['limite'])
    texto += (f'\n\nEn prueba, el original obtiene {num(t["pr_o"], d)} (puesto {t["puesto_prueba"]} de 10) y {nb}, {num(t["pr_b"], d)}; el límite teórico es '
              f'{num(t["limite"], d)}, así que el benchmark queda a {num(distancia_b, d + 1)} del límite y el original, a {num(distancia_o, d + 1)}. ')
    if not t['superior'] and not t['pendiente']:
        if t['c1'] and not t['c2']:
            texto += ('**Causa.** La diferencia es sistemática, pero tan pequeña que, con la corrección por comparaciones múltiples, las pruebas '
                      'no alcanzan a detectarla todas: el benchmark ya estaba casi en el límite del conjunto y no dejaba margen para una '
                      'diferencia mayor.')
        elif tarea == 'clasificacion' and not t['c1']:
            usa = _uso_del_puntaje(h)
            if usa['sin']:
                dif_sin = usa['original_sin'] - usa['benchmark_sin']
                dif_con = usa['original_con'] - usa['benchmark_con'] if usa['con'] else float('nan')
                texto += (f'**Análisis de la causa.** El espacio de búsqueda incluía la variante que aprende solo de la etiqueta '
                          f'(`usar_puntaje = False`), y la búsqueda interna la eligió en {usa["sin"]} de los {n} pliegues externos. En esos pliegues '
                          f'el original obtuvo un AUC medio de {num(usa["original_sin"], d)} y {nb}, {num(usa["benchmark_sin"], d)} '
                          f'(diferencia: {num(dif_sin, d)})')
                if usa['con']:
                    texto += (f'; en los {usa["con"]} en que usó el puntaje, {num(usa["original_con"], d)} frente a {num(usa["benchmark_con"], d)} '
                              f'(diferencia: {num(dif_con, d)}; por delante en {usa["gana_con"]} de {usa["con"]}). ')
                    if dif_con > dif_sin:
                        texto += 'La desventaja en la media viene sobre todo de los pliegues en que el modelo no usó el puntaje. '
                else:
                    texto += '. '
                texto += ('El modelo final, reajustado con toda la partición de entrenamiento, sí usa el puntaje. ' if usa['final_usa_puntaje'] else
                          'El modelo final tampoco usa el puntaje. ')
                texto += (f'En cualquier caso, estas diferencias son menores que la desviación entre pliegues del propio benchmark '
                          f'({num(r["desviaciones"][t["mejor"]], d)}): con estos datos, los primeros modelos de la tabla no se distinguen entre sí en '
                          f'esta tarea. La ablación de la sección 5.9 mide por separado la variante sin puntaje.')
            else:
                texto += ('**Causa.** El modelo usó el puntaje en todos los pliegues y aun así no supera al mejor benchmark en la media externa: '
                          'en esta tarea los benchmarks de árboles ya ordenan a los clientes tan bien como permite el conjunto.')
        elif not t['c1']:
            texto += '**Causa.** El original no mejora la media externa del mejor benchmark.'
    return texto


def veredicto_texto() -> None:
    h = _hechos()
    partes = [_parrafo_veredicto('regresion', h), _parrafo_veredicto('clasificacion', h)]
    h3, h4 = h['h3'], h['h4']
    partes.append(
        f'**Hipótesis secundarias (no cuentan para R3).** H3: con cobertura nominal del 90 %, el conjunto de predicción del modelo original mide '
        f'{num(h3["largo_conjunto"], 4)} (la unión de {h3["tramos"]} tramos) y el intervalo conforme de {h["regresion"]["nb"]}, '
        f'{num(h3["largo_benchmark"], 4)}; la cobertura medida en prueba es {num(100 * h3["cobertura"], 1)} %, dentro del error de muestreo '
        f'(±{num(100 * h3["error"], 1)} puntos). H3 {"se cumple" if h3["cumple"] else "no se cumple"}. '
        f'H4: la pérdida logarítmica del original en prueba es {num(h4["original"], 4)} y la de {h["clasificacion"]["nb"]} con sus probabilidades '
        f'recalibradas, {num(h4["benchmark"], 4)} (mínimo teórico: {num(h4["limite"], 4)}). H4 {"se cumple" if h4["cumple"] else "no se cumple"}'
        + ('.' if h4['cumple'] else '; la sección 7.9 muestra de dónde sale la diferencia.'))
    estados = {t: ('superior' if h[t]['superior'] else 'pendiente' if h[t]['pendiente'] else 'no superior') for t in MAYOR}
    if all(e == 'superior' for e in estados.values()):
        cierre = '**En conjunto, R3 se cumple en las dos tareas.**'
    elif all(e == 'no superior' for e in estados.values()):
        cierre = ('**En conjunto, R3 no se cumple en ninguna de las dos tareas**: con el protocolo registrado, no se puede afirmar que el modelo '
                  'original supere a los benchmarks.')
    else:
        cierre = '**En conjunto:** ' + '; '.join(f'{NOMBRE_TAREA[t]}, {e}' for t, e in estados.items()) + '.'
    if any(h[t]['repeticiones'] < 3 for t in MAYOR):
        cierre += ' El resultado es provisional hasta completar las tres repeticiones que fija el protocolo.'
    partes.append(cierre)
    escribir('veredicto_texto.md', '\n\n'.join(partes) + '\n')


def significancia_practica() -> None:
    h = _hechos()
    reg, cla, te = h['regresion'], h['clasificacion'], h['techo']
    sr, sc = reg['sub'], cla['sub']
    sd = float(te['prueba']['desviacion_del_puntaje'])
    rel = 100 * abs(reg['ventaja']) / reg['cv_b']
    sentido = 'menor' if reg['ventaja'] > 0 else 'mayor'
    texto = (f'**Regresión.** El RMSE medio del original en los pliegues externos es {num(abs(reg["ventaja"]), 6)} {sentido} que el de {reg["nb"]}: '
             f'el {num(rel, 2)} % del error del benchmark y el {num(100 * abs(reg["ventaja"]) / sd, 2)} % de la desviación típica del puntaje '
             f'({num(sd, 4)}). Ninguna decisión sobre un cliente cambia por una diferencia así. Lo que el número deja ver es otra cosa: casi todo '
             f'el error es irreducible. El predictor óptimo obtiene en prueba {num(reg["limite"], 5)}; por encima de ese valor, '
             f'{reg["nb"]} deja {num(abs(reg["pr_b"] - reg["limite"]), 6)} y el original, {num(abs(reg["pr_o"] - reg["limite"]), 6)}.')
    p4 = reg['dm'].get('P4', {})
    if 'ic' in p4:
        texto += (f' La diferencia de RMSE en prueba (original menos benchmark) es {num(p4["estadistico"], 6)}, con un intervalo de confianza del '
                  f'95 % de [{num(p4["ic"][0], 6)}; {num(p4["ic"][1], 6)}] (bootstrap pareado BCa, sin ajustar por comparaciones múltiples)'
                  + (', que incluye el cero.' if p4['ic'][0] <= 0 <= p4['ic'][1] else ', que no incluye el cero.'))
    if 'mae_mediana' in sr.columns and not np.isnan(sr.loc[plan.ORIGINAL, 'mae_mediana']):
        mae_o, mae_b = float(sr.loc[plan.ORIGINAL, 'mae_mediana']), float(sr.loc[reg['mejor'], 'mae'])
        texto += (f' La diferencia que sí se nota está en lo que el modelo sabe del error: como conoce los niveles, puede predecir la mediana en '
                  f'lugar de la media, y el error absoluto medio baja de {num(mae_b, 5)} a {num(mae_o, 5)} (un {num(100 * (mae_b - mae_o) / mae_b, 1)} % '
                  f'menos)')
        homogenea = _referencias_homogeneas()
        if homogenea:
            mae_h = homogenea['mae']['benchmark_con_mediana']
            texto += (f'. Parte de esa diferencia es de receta: si al pronóstico de {reg["nb"]} se le suma la mediana de sus residuos fuera de '
                      f'pliegue, su error absoluto baja a {num(mae_h, 5)} y la ventaja del original queda en un '
                      f'{num(100 * (mae_h - mae_o) / mae_h, 1)} % (comparación exploratoria)')
        texto += (f'. Con la misma cobertura nominal del 90 %, su conjunto de predicción mide {num(h["h3"]["largo_conjunto"], 4)} frente a '
                  f'{num(h["h3"]["largo_benchmark"], 4)} del intervalo del benchmark.' + _frase_conjunto_homogeneo())
    n, n_pos = h['n_prueba'], int(round(te['prueba']['prevalencia'] * h['n_prueba']))
    campana = int(round(0.25 * n))                                      # como src.pruebas.marcar_campana
    cap_o, cap_b = float(sc.loc[plan.ORIGINAL, 'captura_25']), float(sc.loc[cla['mejor'], 'captura_25'])
    dif = int(round((cap_o - cap_b) * n_pos))
    p3 = cla['dm'].get('P3', {})
    texto += (f'\n\n**Clasificación.** En una campaña que contacta al 25 % de los clientes de prueba con más riesgo ({entero(campana)} de {entero(n)}), '
              f'el original encuentra el {num(100 * cap_o, 1)} % de los {entero(n_pos)} clientes de riesgo alto y {cla["nb"]}, el {num(100 * cap_b, 1)} %: '
              + (f'{abs(dif)} clientes {"más" if dif > 0 else "menos"}. ' if dif else 'los mismos. '))
    emp = _empates()
    if emp:
        texto += _frase_empates(emp)
    if 'diferencia' in p3:
        texto += (f'La diferencia de AUC en prueba es {num(p3["diferencia"], 4)}, con un intervalo de confianza del 95 % de '
                  f'[{num(p3["ic"][0], 4)}; {num(p3["ic"][1], 4)}] (DeLong, sin ajustar por comparaciones múltiples)' + (', que incluye el cero' if p3['ic'][0] <= 0 <= p3['ic'][1] else ', que no incluye el cero'))
        try:
            bca = _ic_diferencia_auc(cla['mejor'])
            texto += f'; con bootstrap pareado BCa ({entero(bca["n_boot"])} réplicas), [{num(bca["ic_bca"][0], 4)}; {num(bca["ic_bca"][1], 4)}]. '
        except Exception:                                                # noqa: BLE001
            texto += '. '
    ece_recalibrado = float(sc.loc[cla['mejor'], 'ece_isotonica'])
    texto += (f'En la práctica las dos campañas encuentran casi el mismo número de clientes de riesgo alto. El original entrega probabilidades '
              f'que se pueden usar tal como salen: su error de calibración esperado es {num(h["h4"]["ece_original"], 3)}. El de {cla["nb"]} es '
              f'{num(h["h4"]["ece_benchmark_crudo"], 3)} sin recalibrar, porque se ajustó con ponderación de clases, y {num(ece_recalibrado, 3)} '
              f'tras recalibrarlo: la calibración del original no es mejor que la de un benchmark recalibrado; se ahorra ese paso.')
    texto += ('\n\n**En resumen**, en la métrica principal de las dos tareas la mejora posible era de milésimas y eso es lo que se observa. El aporte '
              'práctico del modelo original no es un punto más de precisión: es que describe la estructura del error (cuántos niveles hay, dónde '
              'están y con qué probabilidad), que es lo que los benchmarks tratan como ruido.')
    escribir('significancia_practica.md', texto + '\n')


def adicionales_texto() -> None:
    c4 = json.loads((R / 'condicion4.json').read_text(encoding='utf-8'))
    partes = ['**Lectura.** El protocolo declaró de antemano la expectativa: ventaja en los dos conjuntos con una variable omitida y empate en '
              'los dos de control.']
    for tarea in ('regresion', 'clasificacion'):
        c = c4.get(tarea)
        if not c:
            continue
        frases = []
        for x in c['detalle'].values():
            pct = f'{num(100 * abs(x["ventaja_relativa"]), 2)} %'
            estado = (f'mejor que {N[x["mejor_benchmark"]]} por un {pct}' if x['es_mejor'] else f'peor que {N[x["mejor_benchmark"]]} por un {pct}')
            estado += f' ({"p < 0,001" if x["p"] < 0.001 else "p = " + pvalor(x["p"])})'
            frases.append(f'en {x["nombre"]}{" †" if x["con_variable_omitida"] else ""}, {estado}')
        clases = [f'{x["nombre"]}, $K = {x["clases"]}$' for x in c['detalle'].values() if isinstance(x['clases'], int)]
        omit = [x for x in c['detalle'].values() if x['con_variable_omitida']]
        ctrl = [x for x in c['detalle'].values() if not x['con_variable_omitida']]
        cumple_omit = [x['nombre'] for x in omit if x['es_mejor']]
        empata_ctrl = [x['nombre'] for x in ctrl if x['p'] >= 0.05]
        lectura = ''
        if omit:
            lectura += (f' Con variable omitida, la ventaja esperada aparece en {len(cumple_omit)} de {len(omit)}'
                        + (f' ({_lista(cumple_omit)})' if cumple_omit and len(cumple_omit) < len(omit) else '') + '.')
        if ctrl:
            lectura += (f' En los controles, la diferencia con el mejor benchmark no es significativa en {len(empata_ctrl)} de {len(ctrl)}'
                        + (f' ({_lista(empata_ctrl)})' if empata_ctrl and len(empata_ctrl) < len(ctrl) else '') + '.')
        sin = [x for x in c['detalle'].values() if str(x.get('n_componentes_configurado')) == '1']
        if tarea == 'regresion' and sin:
            lectura += (f' En {_lista(x["nombre"] for x in sin)} la búsqueda de hiperparámetros eligió la variante sin mezcla para el modelo '
                        f'final (y en {_lista(str(x["pliegues_sin_mezcla"]) + " de " + str(x["pliegues"]) for x in sin)} pliegues externos, en ese '
                        'orden): ahí el original es un boosting de mínimos cuadrados con árboles propios, y lo que gane o pierda no se debe a la '
                        'clase latente.')
            if any(x['con_variable_omitida'] for x in sin):
                lectura += (' Que ocurra también en los conjuntos con una variable omitida dice que omitir una variable no basta: el modelo '
                            'necesita que su efecto sea un desplazamiento constante e independiente de los demás predictores. Una explicación '
                            'probable, que aquí no se contrastó: en Seguros médicos el efecto de fumar depende del índice de masa corporal, y en '
                            'Eficiencia energética el área de vidrio omitida se deduce en parte de otra variable del conjunto.')
        cuerpo = '; '.join(frases)
        partes.append(f'**{NOMBRE_TAREA[tarea].capitalize()}, el original frente al mejor benchmark de cada conjunto:** {cuerpo}.{lectura} '
                      + (f'Condición 4: {"se cumple" if c["cumple"] else "no se cumple"} ({c["resumen"]}).' if c.get('completo')
                         else f'Condición 4, provisional con {c["conjuntos"]} de {c.get("esperados", 4)} conjuntos: {c["resumen"]}.'))
    partes.append('En los conjuntos pequeños los pliegues externos tienen pocas filas y la desviación entre pliegues es grande: con cinco '
                  'pliegues, la prueba *t* corregida solo detecta diferencias amplias, y un «no significativo» no es evidencia de igualdad.')
    escribir('adicionales_texto.md', '\n\n'.join(partes) + '\n')


def errores_texto() -> None:
    h = _hechos()
    p = _problema()
    c = p['conjuntos']
    pred = pd.read_parquet(R / 'predicciones_prueba.parquet')
    y_bin, y_cont = p['y_test'].to_numpy(), c.y_test_cont.to_numpy()
    productos = p['df'].loc[c.X_test.index, 'active_products'].to_numpy()        # solo para analizar el error
    partes = []
    # 1. de qué está hecho el error de regresión
    err = y_cont - pred[f'regresion__{plan.ORIGINAL}'].to_numpy()
    indice_real = y_cont + h['techo']['delta'] * (productos - 1)
    if 'regresion__bdl__indice' in pred:
        dif = pred['regresion__bdl__indice'].to_numpy() - indice_real
        error_indice = float(np.std(dif))
        parte = 100 * (1 - error_indice ** 2 / np.mean(err ** 2))
        partes.append(
            f'**El error de regresión es casi todo escalón.** Con `active_products` (que el modelo nunca ve) se puede reconstruir el índice real de '
            f'cada cliente de prueba y compararlo con el que estima el modelo: la desviación de esa diferencia es {num(error_indice, 5)}, frente a un '
            f'RMSE total de {num(float(np.sqrt(np.mean(err ** 2))), 5)}. {"Más del 99,9" if parte >= 99.95 else "El " + num(parte, 1)} % del error '
            f'cuadrático es el escalón, que ningún modelo puede predecir sin la variable excluida.')
        por_nivel = pd.DataFrame({'k': productos, 'e2': err ** 2}).groupby('k')['e2'].agg(['mean', 'sum', 'size'])
        peor = por_nivel['mean'].idxmax()
        frase = (f'**Los errores grandes son los clientes con muchos productos.** Predecir la media obliga a equivocarse más con los niveles poco '
                 f'frecuentes: los clientes con {int(peor)} productos son el {num(100 * por_nivel.loc[peor, "size"] / len(err), 1)} % de la prueba y '
                 f'aportan el {num(100 * por_nivel.loc[peor, "sum"] / por_nivel["sum"].sum(), 1)} % del error cuadrático (RMSE de '
                 f'{num(float(np.sqrt(por_nivel.loc[peor, "mean"])), 4)} en ese grupo).')
        try:
            tuberia = _final('regresion')
            Z = tuberia[:-1].transform(c.X_test)
            grupo = productos == peor
            cubre = {q: float(tuberia.named_steps['modelo'].conjunto_cubre(Z, y_cont, q)[grupo].mean()) for q in (0.90, 0.95, 0.99)}
            frase += (f' La cobertura de los conjuntos de predicción es del total de clientes, no de cada grupo: como ese es el nivel menos '
                      f'probable, es el que queda fuera. Al 90 % el conjunto cubre al {num(100 * cubre[0.90], 1)} % de esos clientes; al 95 %, al '
                      f'{num(100 * cubre[0.95], 1)} %, y al 99 %, al {num(100 * cubre[0.99], 1)} %. Quien necesite cubrir a los clientes con muchos '
                      f'productos tiene que pedir una cobertura alta.')
        except FileNotFoundError:
            pass
        partes.append(frase)
    # 2. la cola de tx_count
    cola = (c.X_test['tx_count'] > 500).to_numpy()
    if cola.sum() > 20 and 0 < y_bin[cola].sum() < cola.sum():
        from sklearn.metrics import roc_auc_score
        mc = h['clasificacion']['mejor']
        a_o = roc_auc_score(y_bin[cola], pred[f'clasificacion__{plan.ORIGINAL}'].to_numpy()[cola])
        a_b = roc_auc_score(y_bin[cola], pred[f'clasificacion__{mc}'].to_numpy()[cola])
        r_o = float(np.sqrt(np.mean(err[cola] ** 2)))
        partes.append(
            f'**La cola de `tx_count`.** En los {entero(cola.sum())} clientes de prueba con más de 500 transacciones ({num(100 * cola.mean(), 1)} %) '
            f'el índice depende además de esa variable. Ahí el RMSE del original sube a {num(r_o, 5)} y su AUC es {num(a_o, 4)}, frente a '
            f'{num(a_b, 4)} de {N[mc]}' + ('. ' if a_o >= a_b else ': es el subgrupo donde el original pierde con más claridad. ')
            + 'Hay pocos clientes para aprender ese efecto, y el modelo lo confunde en parte con un cambio de nivel.')
    # 3. la pérdida logarítmica: errores con probabilidad casi cero o casi uno
    pb = pred[f'clasificacion__{plan.ORIGINAL}'].to_numpy()
    r = np.clip(pb, 1e-6, 1 - 1e-6)
    ll = -(y_bin * np.log(r) + (1 - y_bin) * np.log(1 - r))
    seguros = ((pb < 0.001) & (y_bin == 1)) | ((pb > 0.999) & (y_bin == 0))
    h4 = h['h4']
    if seguros.sum():
        mc = h['clasificacion']['mejor']
        frase = (
            f'**H4 no se cumple, y la diferencia está en {int(seguros.sum())} clientes.** La probabilidad del original tiene mesetas: vale casi 0 '
            f'cuando ningún nivel del escalón deja al cliente por encima del umbral, y salta al entrar el primero. Si el índice estimado queda una '
            f'milésima por debajo del real justo en ese borde, el modelo dice «casi imposible» y se equivoca. En prueba pasó con '
            f'{int(seguros.sum())} de {entero(len(pb))} clientes, que aportan {num(float(ll[seguros].sum() / len(ll)), 4)} de los '
            f'{num(float(ll.mean()), 4)} de pérdida logarítmica del original. ')
        clave_b = f'clasificacion__{mc}__isotonica'
        if clave_b in pred:
            rb = np.clip(pred[clave_b].to_numpy(), 1e-6, 1 - 1e-6)
            ll_b = -(y_bin * np.log(rb) + (1 - y_bin) * np.log(1 - rb))
            sin_o, sin_b = float(ll[~seguros].mean()), float(ll_b[~seguros].mean())
            frase += (f'Sin esos clientes, y quitándoselos a los dos modelos, la pérdida del original es {num(sin_o, 4)} y la de {N[mc]} '
                      f'recalibrado, {num(sin_b, 4)}: '
                      + ('un empate. ' if abs(sin_o - sin_b) < 0.0005 else 'el original queda por delante. ' if sin_o < sin_b else
                         'el benchmark sigue por delante. '))
        frase += ('Es el precio de una probabilidad tan nítida: la dispersión robusta con la que se calcula la excedencia hace los bordes muy '
                  'abruptos. Una dispersión mayor suavizaría los bordes; no se cambió porque el protocolo ya estaba registrado. Este desglose se '
                  'hizo después de ver los resultados.')
        partes.append(frase)
    elif not h4['cumple']:
        partes.append('**H4.** El original no tiene errores con probabilidad casi cero o casi uno; su pérdida logarítmica es mayor que la del '
                      'benchmark recalibrado por diferencias repartidas entre las mesetas.')
    # 4. subgrupos
    ruta = R / 'errores_subgrupos.csv'
    if ruta.exists():
        g = pd.read_csv(ruta)
        g = g[g.subgrupo != 'Todos']
        con_auc = g.dropna(subset=['auc_original'])
        partes.append(f'**Por subgrupos**, el original tiene menor RMSE que el mejor benchmark en {int((g.rmse_original < g.rmse_benchmark).sum())} de '
                      f'{len(g)} y mayor AUC en {int((con_auc.auc_original > con_auc.auc_benchmark).sum())} de {len(con_auc)}. Las diferencias dentro '
                      'de cada subgrupo son del mismo orden que en el total y cambian de signo de uno a otro: no hay un grupo de clientes al que '
                      'el modelo sirva claramente peor, salvo la cola de `tx_count`.')
    escribir('errores_texto.md', '\n\n'.join(partes) + '\n')


NOMBRE_OPTIMO = {'regresion': {'P1': 'P1. Wilcoxon (pliegues externos)', 'P2': 'P2. *t* corregida (pliegues externos)',
                               'P3': 'P3. Permutación pareada (prueba)', 'P4': 'P4. Bootstrap pareado BCa (prueba)'},
                 'clasificacion': {'P1': 'P1. Wilcoxon (pliegues externos)', 'P2': 'P2. *t* corregida (pliegues externos)',
                                   'P3': 'P3. DeLong (prueba)', 'P4': 'P4. McNemar, campaña del 25 % (prueba)'}}


def _frase_del_optimo(h: dict) -> str:
    """Lo que dice el análisis del predictor óptimo (``resultados/optimo.json``) sobre si R3 era alcanzable; '' si no existe."""
    ruta = R / 'optimo.json'
    if not ruta.exists():
        return ''
    o = json.loads(ruta.read_text(encoding='utf-8'))
    if 'rechazos_del_optimo' not in o['regresion']:                       # archivo de una versión anterior del guion
        return ''
    frases = []
    for tarea in ('clasificacion', 'regresion'):
        r = o[tarea]
        nb, k, n = N[r['mejor_benchmark']], r['rechazos_del_optimo'], len(r['optimo'])
        cuales = [c for c in r['optimo'] if r['optimo'][c]['rechaza']]
        if k == 0:
            frases.append(f'En {NOMBRE_TAREA[tarea]}, el predictor óptimo (el que conoce el índice real de cada cliente, que ningún modelo puede '
                          f'aprender sin la variable excluida) no rechaza la igualdad con {nb} en ninguna de las {n} pruebas.')
        elif k < 3:
            frases.append(f'En {NOMBRE_TAREA[tarea]}, el predictor óptimo rechaza la igualdad con {nb} en {k} de las {n} pruebas '
                          f'({_lista(cuales)}), una menos de las tres que pide la guía.' if k == 2 else
                          f'En {NOMBRE_TAREA[tarea]}, el predictor óptimo rechaza la igualdad con {nb} en {k} de las {n} pruebas ({_lista(cuales)}).')
        else:
            frases.append(f'En {NOMBRE_TAREA[tarea]}, el predictor óptimo sí reúne {k} rechazos frente a {nb} ({_lista(cuales)}): con una '
                          f'estimación casi perfecta del índice la condición era alcanzable, y el original se queda en '
                          f'{r["rechazos_del_original"]}.')
    if all(o[t]['rechazos_del_optimo'] < 3 for t in ('regresion', 'clasificacion')):
        frases.append('Con este protocolo y estas muestras, ni el mejor predictor posible habría reunido los tres rechazos: es muy improbable '
                      'que lo consiguiera un modelo que no ve la variable excluida.')
    if any(o[t]['pliegues'] < 15 for t in ('regresion', 'clasificacion')):
        frases.append(f'Las pruebas P1 y P2 usan los {o["regresion"]["pliegues"]} pliegues externos disponibles; cambiarán al completar las '
                      'tres repeticiones.')
    return ' '.join(frases)


def optimo() -> None:
    """¿Podía algún modelo cumplir R3? Las cuatro pruebas del protocolo para el predictor óptimo, junto a las del modelo original."""
    o = json.loads((R / 'optimo.json').read_text(encoding='utf-8'))
    filas = []
    for tarea in ('regresion', 'clasificacion'):
        r = o[tarea]
        for k in r['optimo']:
            a, b = r['optimo'][k], r['original'].get(k, {})
            filas.append([TAREA[tarea].split(' (')[0], NOMBRE_OPTIMO[tarea][k], N[r['mejor_benchmark']], pvalor(a['p']), pvalor(a['p_holm']),
                          'Sí' if a['rechaza'] else 'No', pvalor(b.get('p_holm')), 'Sí' if b.get('rechaza') else 'No'])
    reg, cla = o['regresion'], o['clasificacion']
    nota = (f'\nEl predictor óptimo obtiene en prueba un RMSE de {num(reg["metrica_del_optimo"], 5)} y un AUC de '
            f'{num(cla["metrica_del_optimo"], 4)} (son los valores de referencia de la sección 6.4), y en los {reg["pliegues"]} pliegues externos, '
            f'{num(reg["media_del_optimo_en_pliegues"], 5)} y {num(cla["media_del_optimo_en_pliegues"], 4)} de media (el original: '
            f'{num(reg["media_del_original_en_pliegues"], 5)} y {num(cla["media_del_original_en_pliegues"], 4)}). Los p ajustados son de Holm sobre las '
            f'{reg["comparaciones"]} comparaciones de cada prueba, como en el protocolo; un rechazo solo cuenta si la diferencia favorece al '
            f'modelo. La prueba global de Friedman, que el protocolo exige antes de P1, no se repite para el óptimo.\n\n'
            + _frase_del_optimo(_hechos()) + '\n')
    escribir('tabla_optimo.md', tabla_md(['Tarea', 'Prueba', 'Mejor benchmark', 'p crudo del óptimo', 'p ajustado del óptimo',
                                          '¿Rechaza el óptimo?', 'p ajustado del original', '¿Rechaza el original?'], filas, 'lllrrlrl') + nota)


def _referencias_homogeneas() -> dict | None:
    """Las referencias homogéneas (``resultados/referencias_homogeneas.json``), o ``None`` si todavía no existen."""
    ruta = R / 'referencias_homogeneas.json'
    return json.loads(ruta.read_text(encoding='utf-8')) if ruta.exists() else None


def _frase_conjunto_homogeneo() -> str:
    """Cuánto más corto es el conjunto del original frente a un conjunto del mismo tipo construido con el benchmark."""
    r = _referencias_homogeneas()
    if not r:
        return ''
    x = r['conjuntos']['por_cobertura']['0.90']
    return (f' Frente a un conjunto del mismo tipo construido con los residuos de {N[r["benchmark"]]} ({num(x["benchmark_conjunto"]["largo"], 4)}), '
            f'unas {num(x["benchmark_conjunto"]["largo"] / x["original_conjunto"]["largo"], 0)} veces (comparación exploratoria).')


def _frase_adicionales(h: dict) -> str:
    """El resultado en los conjuntos adicionales, para los resúmenes: sin ventaja, y por qué."""
    c4 = h['c4']
    if not c4:
        return ''
    tareas = [t for t in ('regresion', 'clasificacion') if c4.get(t)]
    if not tareas:
        return ''
    n = max(c4[t]['conjuntos'] for t in tareas)
    completo = all(c4[t].get('completo') for t in tareas)
    if all(c4[t]['mejor_en'] == 0 for t in tareas):
        cuerpo = 'el original no supera al mejor benchmark en ninguno, ni en regresión ni en clasificación'
    else:
        cuerpo = 'el original supera al mejor benchmark ' + ' y '.join(f'en {c4[t]["mejor_en"]} de {c4[t]["conjuntos"]} en {NOMBRE_TAREA[t]}' for t in tareas)
    peores = [f'en {NOMBRE_TAREA[t]} es peor de forma significativa en {_lista(c4[t]["peor_significativo_en"])}' for t in tareas
              if c4[t]['peor_significativo_en']]
    texto = (f'En los {n} conjuntos de datos adicionales' + ('' if completo else ' evaluados hasta ahora') + ', ' + cuerpo
             + ('; ' + ' y '.join(peores) if peores else '') + ' (sección 7.1).')
    reg = (c4.get('regresion') or {}).get('detalle', {})
    sin = [x for x in reg.values() if str(x.get('n_componentes_configurado')) == '1']
    if sin:
        texto += (f' En {len(sin)} de ellos la búsqueda de hiperparámetros eligió para la regresión la variante sin mezcla'
                  + (', incluidos los construidos con una variable omitida' if any(x['con_variable_omitida'] for x in sin) else '')
                  + ': la ventaja que el protocolo esperaba ahí no apareció.')
    return texto


def _empates() -> dict | None:
    """El análisis de los empates de la campaña (``resultados/empates_campana.json``), o ``None`` si todavía no existe."""
    ruta = R / 'empates_campana.json'
    return json.loads(ruta.read_text(encoding='utf-8')) if ruta.exists() else None


def _frase_empates(e: dict) -> str:
    """Una frase: de qué depende la captura del original en la campaña del 25 %."""
    o = e['modelos'][plan.ORIGINAL]
    if o['empatados'] <= 1:
        return ''
    return (f'Esa cifra depende del desempate: {entero(o["empatados"])} clientes comparten la probabilidad del corte y solo entran '
            f'{entero(o["cupos"])}; con un desempate al azar, la captura esperada del original es {num(100 * o["captura_esperada"], 1)} % ± '
            f'{num(100 * o["captura_desviacion"], 1)} puntos (Tabla 6.11). ')


def empates() -> None:
    """Empates en el corte de la campaña del 25 %: cuánto dependen de ellos la captura, la precisión media y la prueba P4."""
    e = json.loads((R / 'empates_campana.json').read_text(encoding='utf-8'))
    p4 = e['p4']
    mejor = p4['mejor_benchmark']
    orden = [plan.ORIGINAL, mejor] + [m for m, v in e['modelos'].items() if v['empatados'] > 1 and m not in (plan.ORIGINAL, mejor)]
    filas_c, filas_p = [], []
    for nombre, v in [(f'**{N[m]}**' if m == plan.ORIGINAL else N[m], e['modelos'][m]) for m in orden] + [('Predictor óptimo', e['optimo'])]:
        filas_c.append([nombre, entero(v['valores_distintos']), entero(v['empatados']), entero(v['cupos']), f'{num(100 * v["captura_protocolo"], 1)} %',
                        f'{num(100 * v["captura_esperada"], 1)} % ± {num(100 * v["captura_desviacion"], 1)}',
                        f'{num(100 * v["captura_minima"], 1)} % a {num(100 * v["captura_maxima"], 1)} %'])
        filas_p.append([nombre, num(v['precision_media'], 4), f'{num(v["precision_media_azar"], 4)} ± {num(v["precision_media_azar_desviacion"], 4)}'])
    o, b, op = e['modelos'][plan.ORIGINAL], e['modelos'][mejor], e['optimo']
    nb = N[mejor]
    texto = ('**Captura de la campaña del 25 %.**\n\n'
             + tabla_md(['Modelo', 'Valores distintos de la probabilidad', 'Empatados en el corte', 'De ellos entran', 'Captura (protocolo)',
                         'Captura esperada (al azar)', 'Captura posible'], filas_c, 'lrrrrrr')
             + '\n**Precisión media.**\n\n'
             + tabla_md(['Modelo', 'Convención de scikit-learn', 'Empates rotos al azar'], filas_p, 'lrr')
             + f'\nLa campaña contacta a los {entero(e["campana"])} clientes de mayor probabilidad de los {entero(e["n"])} de prueba. Empatados en el '
               'corte: los que tienen la misma probabilidad que el último contactado; de ellos solo entra una parte, y cuál, lo decide el desempate. '
               'Protocolo: desempate por posición en la tabla, el del guion de las pruebas. Al azar: valor esperado ± desviación típica, en puntos '
               f'porcentuales (distribución hipergeométrica); en la precisión media, media ± desviación de {e["replicas_precision_media"]} desempates.\n')
    lectura = (f'\n**Lectura.** La probabilidad del original toma {entero(o["valores_distintos"])} valores distintos y la del predictor óptimo, '
               f'{entero(op["valores_distintos"])}: es una función escalonada, y dentro de una meseta todos los clientes tienen la misma probabilidad '
               f'(Proposición 5). El corte de la campaña cae dentro de una meseta: {entero(o["empatados"])} clientes comparten la probabilidad del '
               f'último contactado y solo entran {entero(o["cupos"])}. La captura del original con el desempate del protocolo, '
               f'{num(100 * o["captura_protocolo"], 1)} %, es uno de los valores posibles: con un desempate al azar se espera '
               f'{num(100 * o["captura_esperada"], 1)} % ± {num(100 * o["captura_desviacion"], 1)} puntos')
    if abs(o['captura_esperada'] - op['captura_esperada']) < o['captura_desviacion']:
        lectura += f', lo mismo que con el predictor óptimo ({num(100 * op["captura_esperada"], 1)} %)'
    dif = abs(o['captura_esperada'] - b['captura_esperada'])
    lectura += (f'. La diferencia con {nb} ({num(100 * b["captura_protocolo"], 1)} %) es '
                + ('menor' if dif < o['captura_desviacion'] else 'mayor') + ' que esa variación.')
    estable = p4['rechaza_a_favor'] < 0.05 and p4['rechaza_en_contra'] < 0.05
    lectura += (f'\n\n**Qué significa para P4.** Repetida con {entero(e["replicas"])} desempates al azar de los dos modelos, la prueba de McNemar '
                f'del original frente a {nb} da un p crudo con mediana {pvalor(p4["p_crudo"]["mediana"])} (entre {pvalor(p4["p_crudo"]["q025"])} y '
                f'{pvalor(p4["p_crudo"]["q975"])} en el 95 % de los desempates). Tras el ajuste de Holm sobre las {p4["comparaciones"]} '
                f'comparaciones, rechaza la igualdad a favor del original en el {num(100 * p4["rechaza_a_favor"], 2)} % de los desempates y en '
                f'contra en el {num(100 * p4["rechaza_en_contra"], 2)} %. '
                + (f'El resultado registrado (no rechaza; p ajustado = {pvalor(p4["protocolo"]["p_holm"])}) no depende del desempate: es el de casi '
                   'todos los sorteos. Pero deja ver qué mide la prueba en este conjunto: dentro de la meseta del corte, el original y el predictor '
                   'óptimo eligen al azar.' if estable else
                   'El resultado de P4 sí depende del desempate, y hay que leerlo con esa reserva.'))
    if o['precision_media_azar'] - o['precision_media'] > 3 * o['precision_media_azar_desviacion']:
        lectura += (f'\n\n**Precisión media.** scikit-learn trata a los clientes empatados como un solo punto de la curva de precisión y '
                    f'exhaustividad, con la precisión del grupo entero. Con mesetas grandes eso da menos que cualquier orden concreto dentro de ellas: '
                    f'{num(o["precision_media"], 4)} para el original y {num(op["precision_media"], 4)} para el predictor óptimo, por debajo de '
                    f'{nb} ({num(b["precision_media"], 4)}). Con los empates rotos al azar son {num(o["precision_media_azar"], 4)} y '
                    f'{num(op["precision_media_azar"], 4)}: la desventaja del original en esa columna de la Tabla 7.11 viene de la convención, '
                    'no del orden que da el modelo.')
    ruta = plan.RAIZ.parent / 'Dashboard' / 'datos' / 'original' / 'lideres_clasificacion.csv'
    if ruta.exists():
        dash = pd.read_csv(ruta).set_index('modelo')
        if plan.ORIGINAL in dash.index and 'recall_op' in dash.columns:
            captura = float(dash.loc[plan.ORIGINAL, 'recall_op'])
            z = abs(captura - o['captura_esperada']) / o['captura_desviacion'] if o['captura_desviacion'] else 0.0
            lectura += (f'\n\nEl dashboard rompe los empates al azar con su propia semilla, como el Entregable 3, y muestra una captura de '
                        f'{num(100 * captura, 1)} % para el original: es otro sorteo de la misma distribución, a {num(z, 1)} desviaciones del '
                        'valor esperado.')
    escribir('tabla_empates.md', texto + lectura + '\n')


def hallazgos() -> None:
    h = _hechos()
    reg, cla, te = h['regresion'], h['clasificacion'], h['techo']
    puntos = []
    puntos.append(
        f'**Los benchmarks no fallan por falta de ajuste: tratan un escalón como ruido.** El error fuera de pliegue del mejor benchmark de '
        f'regresión se concentra en cinco niveles separados 0,05 (Figura 5.1), que son el efecto de la variable excluida por fuga '
        f'de información. Con esa estructura se puede calcular qué obtendría el mejor predictor posible, el que conociera el índice real '
        f'de cada cliente: en prueba, RMSE {num(reg["limite"], 5)} y AUC {num(cla["limite"], 4)}. Es el límite teórico de este informe, y los '
        f'mejores benchmarks ya estaban a {num(abs(reg["pr_b"] - reg["limite"]), 5)} y {num(abs(cla["pr_b"] - cla["limite"]), 4)} de él.')
    try:
        b = _final('regresion').named_steps['modelo']
        reales = np.sort(np.array(list(te['pesos'].values()), dtype=float))
        if b.n_componentes_ == len(reales):
            puntos.append(
                f'**El modelo original recupera esa estructura sin ver la variable que la causa.** El criterio BIC elige $K = {b.n_componentes_}$ '
                f'clases; sus pesos ({", ".join(num(x, 3) for x in np.sort(b.pi_))}) difieren de la distribución real de `active_products` en menos de '
                f'{num(float(np.abs(np.sort(b.pi_) - reales).max()), 3)}; la separación media entre desplazamientos es '
                f'{num(float(np.diff(b.centros_).mean()), 4)} (la real es {num(te["delta"], 2)}), y una vez descontado el escalón el error del índice '
                f'en validación es {num(b.sigma_validacion_, 5)}, unas {entero(te["prueba"]["rmse_minimo"] / b.sigma_validacion_)} veces menor que el RMSE.')
        else:
            puntos.append(f'**El modelo original encuentra $K = {b.n_componentes_}$ clases latentes** con el criterio BIC.')
    except FileNotFoundError:
        pass
    for t, nombre in ((reg, 'regresión'), (cla, 'clasificación')):
        d, n = t['d'], t['r']['pliegues']
        if t['c1']:
            frase = (f'**En {nombre}, el original tiene la mejor media externa de los diez modelos**: {t["metrica"]} {num(t["cv_o"], d)} frente a '
                     f'{num(t["cv_b"], d)} de {t["nb"]}, por delante en {t["dm"]["pliegues_a_favor"]} de {n} pliegues. ')
        else:
            frase = (f'**En {nombre}, el original no tiene la mejor media externa**: {t["metrica"]} {num(t["cv_o"], d)}, puesto {t["puesto_cv"]} de 10; '
                     f'{t["nb"]} obtiene {num(t["cv_b"], d)}. ')
        frase += (f'En prueba obtiene {num(t["pr_o"], d)} (puesto {t["puesto_prueba"]} de 10), a {num(abs(t["pr_o"] - t["limite"]), d + 1)} del límite '
                  f'teórico. ')
        frase += (f'{_cuantas(len(t["rechazan"]), len(t["pruebas"])).capitalize()} pruebas registradas '
                  f'{"rechaza" if len(t["rechazan"]) <= 1 else "rechazan"} la igualdad con el mejor benchmark a favor del original tras la corrección de Holm')
        frase += ('.' if t['superior'] else ', y la guía pide tres: **no se demuestra superioridad**.' if not t['pendiente'] else '.')
        puntos.append(frase)
    limite = _frase_del_optimo(h)
    if limite:
        puntos.append('**Qué habría obtenido el mejor predictor posible.** ' + limite)
    h3, h4 = h['h3'], h['h4']
    puntos.append(
        f'**La incertidumbre es donde el modelo cambia el resultado.** Con cobertura del 90 %, su conjunto de predicción mide '
        f'{num(h3["largo_conjunto"], 4)} y el intervalo conforme del mejor benchmark, {num(h3["largo_benchmark"], 4)}: unas '
        f'{num(h3["largo_benchmark"] / h3["largo_conjunto"], 0)} veces más corto, con una cobertura medida de {num(100 * h3["cobertura"], 1)} % '
        f'(H3 {"se cumple" if h3["cumple"] else "no se cumple"}). En lugar de un intervalo ancho, el modelo dice en cuál de los niveles puede estar el puntaje.'
        + _frase_conjunto_homogeneo())
    puntos.append(
        f'**Las probabilidades salen calibradas, pero con bordes abruptos.** El error de calibración esperado del original es '
        f'{num(h4["ece_original"], 3)} sin recalibrar ({h["clasificacion"]["nb"]}: {num(h4["ece_benchmark_crudo"], 3)}). Aun así, su pérdida '
        f'logarítmica ({num(h4["original"], 4)}) {"es menor que" if h4["cumple"] else "no mejora"} la del mejor benchmark recalibrado '
        f'({num(h4["benchmark"], 4)}): H4 {"se cumple" if h4["cumple"] else "no se cumple"}'
        + ('.' if h4['cumple'] else ', por unos pocos clientes en el borde de una meseta (sección 7.9).'))
    ruta = R / 'ablacion.csv'
    if ruta.exists():
        a = pd.read_csv(ruta)
        frases = []
        for _, f in a[a.variante != 'bdl'].iterrows():
            d = DEC[f['tarea']]
            p = f.get('p_t_corregida_holm')
            frases.append(f'{QUITA.get(f["variante"], f["nombre"])}, {num(f["aporte_cv"], d + 1)} de {METRICA[f["tarea"]]}'
                          + (f' (p = {pvalor(p)})' if p is not None and not pd.isna(p) else ''))
        sig = a[(a['p_prueba_holm'] < 0.05) & (a['aporte_prueba'] > 0)]
        puntos.append('**Qué aporta cada componente con todas las filas** (ablación, sección 5.9; lo que empeora la media externa con cada cambio, '
                      'con su p ajustado): ' + '; '.join(frases) + '. '
                      + ('En prueba ningún aporte es significativo.' if sig.empty else
                         f'En prueba {"es significativo" if len(sig) == 1 else "son significativos"} {_lista(QUITA.get(v, v) for v in sig["variante"])}.'))
    ruta, ruta_mu = R / 'robustez' / 'muestra_sin_mezcla.csv', R / 'robustez' / 'muestra.csv'
    if ruta.exists() and ruta_mu.exists():
        k1 = pd.read_csv(ruta)
        k1 = k1[k1.tarea == 'regresion'].set_index('n')['media']
        pr = pd.read_csv(ruta_mu)
        pr = pr[pr.tarea == 'regresion'].pivot(index='n', columns='modelo', values='media')
        poco, todo, rival = pr.index.min(), pr.index.max(), reg['mejor']
        if rival in pr.columns and abs(k1[poco] - pr.loc[poco, rival]) < abs(k1[poco] - pr.loc[poco, plan.ORIGINAL]):
            puntos.append(
                f'**La clase latente da eficiencia muestral, no precisión cuando sobran datos.** Con el {num(100 * poco / todo, 0)} % de las filas '
                f'({entero(poco)}), el modelo completo obtiene un RMSE de {num(pr.loc[poco, plan.ORIGINAL], 5)}; sin la mezcla, {num(k1[poco], 5)}, y '
                f'{reg["nb"]}, {num(pr.loc[poco, rival], 5)}. Con todas las filas los tres quedan en {num(pr.loc[todo, plan.ORIGINAL], 5)}, '
                f'{num(k1[todo], 5)} y {num(pr.loc[todo, rival], 5)}'
                + (f': con la mezcla, el modelo consigue con {entero(poco)} filas lo que {reg["nb"]} consigue con {entero(todo)}'
                   if pr.loc[poco, plan.ORIGINAL] <= pr.loc[todo, rival] else '')
                + ' (sección 7.5; análisis exploratorio, añadido después de ver los resultados).')
    fuera = _frase_adicionales(h)
    if fuera:
        puntos.append('**Fuera de COFINFAD no hay ventaja.** ' + fuera)
    ruta = R / 'semillas.csv'
    if ruta.exists():
        s = pd.read_csv(ruta)
        sr = s[s.tarea == 'regresion'].groupby('modelo')
        if plan.ORIGINAL in sr.groups and reg['mejor'] in sr.groups:
            o, b = sr.get_group(plan.ORIGINAL), sr.get_group(reg['mejor'])
            puntos.append(
                f'**El reajuste del modelo final casi no depende de la semilla, y el modelo cuesta más.** Al reajustar con diez semillas (misma '
                f'partición y mismos hiperparámetros; solo cambia la semilla del modelo), el RMSE en prueba del original '
                f'es {num(o["metrica"].mean(), 6)} ± {num(o["metrica"].std(ddof=1), 6)} y el de {reg["nb"]}, {num(b["metrica"].mean(), 6)} ± '
                f'{num(b["metrica"].std(ddof=1), 6)}. Ajustar el original tarda {num(o["ajuste_s"].mean(), 1)} s y {reg["nb"]}, '
                f'{num(b["ajuste_s"].mean(), 1)} s: está escrito en `numpy`, sin compilar.')
    try:
        a = json.loads((R / 'auditoria_fuga.json').read_text(encoding='utf-8'))['fecha_de_corte']
        puntos.append(f'**La corrección de la recencia no cambia ningún dato.** Calculada solo con los clientes de entrenamiento, la fecha de corte '
                      f'es la misma ({a["correccion_solo_entrenamiento"]}) y cambian {a["celdas_de_recencia_que_cambian"]} celdas: los resultados de '
                      f'las entregas anteriores siguen siendo válidos (capítulo 2).')
    except (FileNotFoundError, KeyError):
        pass
    escribir('hallazgos.md', '\n'.join(f'{i}. {x}' for i, x in enumerate(puntos, 1)) + '\n')


def resumen_ejecutivo() -> None:
    h = _hechos()
    reg, cla = h['regresion'], h['clasificacion']
    h3 = h['h3']
    total = h['n_train'] + h['n_prueba']
    parrafos = [
        f'El proyecto predice el riesgo de fuga de los clientes de una fintech con el conjunto COFINFAD ({entero(total)} clientes, sintético) en '
        f'dos tareas: la regresión del puntaje de propensión a la fuga y la clasificación de «riesgo alto». Esta entrega corrige las dos '
        f'observaciones del docente. El modelo original ya no es una librería, sino un algoritmo propio, programado desde cero. Y la recencia se '
        f'calcula solo con los clientes de entrenamiento; la auditoría muestra que esa corrección no cambia ningún dato.',
        'El modelo original, **BDL (Boosting con Desplazamiento Latente)**, parte de un hallazgo: el error de los mejores benchmarks no es ruido, '
        'sino cinco escalones que deja en el puntaje una variable excluida por fuga de información. BDL modela ese escalón como una clase latente y '
        'lo aprende con un algoritmo EM generalizado cuyo paso M es un paso de boosting. Tiene cinco propiedades demostradas y una búsqueda de '
        'literatura reproducible no encontró ningún trabajo que aplique este mecanismo a la fuga de clientes.']
    frases = []
    for t, nombre in ((reg, 'regresión'), (cla, 'clasificación')):
        d, n = t['d'], t['r']['pliegues']
        if t['c1']:
            frases.append(f'En {nombre}, el original tiene la mejor media de los diez modelos en los {n} pliegues externos ({t["metrica"]} '
                          f'{num(t["cv_o"], d)} frente a {num(t["cv_b"], d)} de {t["nb"]}) y obtiene {num(t["pr_o"], d)} en prueba, a '
                          f'{num(abs(t["pr_o"] - t["limite"]), d + 1)} del límite teórico del conjunto.')
        else:
            frases.append(f'En {nombre}, el original queda en el puesto {t["puesto_cv"]} de 10 en la media de los {n} pliegues externos '
                          f'({t["metrica"]} {num(t["cv_o"], d)} frente a {num(t["cv_b"], d)} de {t["nb"]}) y en el puesto {t["puesto_prueba"]} en '
                          f'prueba ({num(t["pr_o"], d)}, a {num(abs(t["pr_o"] - t["limite"]), d + 1)} del límite teórico).')
    frases.append(f'Los benchmarks ya estaban a milésimas de ese límite, así que la mejora posible en la métrica principal era mínima. Donde el '
                  f'modelo cambia el resultado es en la incertidumbre: con cobertura del 90 %, sus conjuntos de predicción son unas '
                  f'{num(h3["largo_benchmark"] / h3["largo_conjunto"], 0)} veces más cortos que los intervalos del mejor benchmark.'
                  + _frase_conjunto_homogeneo())
    fuera = _frase_adicionales(h)
    if fuera:
        frases.append(fuera)
    parrafos.append('**Resultados.** ' + ' '.join(frases))
    estados = {t: ('superior' if h[t]['superior'] else 'pendiente' if h[t]['pendiente'] else 'no superior') for t in ('regresion', 'clasificacion')}
    if all(e == 'no superior' for e in estados.values()):
        veredicto_r3 = ('Con el protocolo registrado antes de los experimentos, **no se demuestra que el modelo original supere a los benchmarks** '
                        f'(R3): en regresión {"rechaza" if len(reg["rechazan"]) <= 1 else "rechazan"} la igualdad '
                        f'{_cuantas(len(reg["rechazan"]), len(reg["pruebas"]))} pruebas y en clasificación, '
                        f'{_cuantas(len(cla["rechazan"]), len(cla["pruebas"]))}, y la guía pide tres.')
    elif all(e == 'superior' for e in estados.values()):
        veredicto_r3 = 'Con el protocolo registrado antes de los experimentos, **el modelo original supera a los benchmarks en las dos tareas** (R3).'
    else:
        veredicto_r3 = ('Con el protocolo registrado antes de los experimentos, el veredicto sobre R3 es: '
                        + '; '.join(f'{NOMBRE_TAREA[t]}, **{e}**' for t, e in estados.items()) + '.')
    if any(h[t]['repeticiones'] < 3 for t in MAYOR):
        veredicto_r3 += ' Este resultado es provisional: faltan repeticiones de la validación anidada.'
    limite = _frase_del_optimo(h)
    if limite and not all(e == 'superior' for e in estados.values()):
        veredicto_r3 += ' ' + limite
    parrafos.append('**Veredicto.** La novedad (R1) y la implementación propia (R2) se sustentan en los capítulos 4 y 5. ' + veredicto_r3
                    + ' Los datos son sintéticos: el escalón que el modelo aprovecha existe porque así se generó el conjunto, y ningún resultado '
                      'de este trabajo es evidencia sobre clientes reales.')
    escribir('resumen_ejecutivo.md', '\n\n'.join(parrafos) + '\n')


# ------------------------------------------------------------------ capítulo 3: espacios de búsqueda e hiperparámetros finales
def _valor(v) -> str:
    """Un valor de hiperparámetro, con coma decimal y potencias de diez para los muy pequeños o muy grandes."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return {True: 'sí', False: 'no', None: 'ninguno'}.get(v, f'«{v}»') if not isinstance(v, str) else f'«{v}»'
    if isinstance(v, (int, np.integer)):
        return entero(v)
    v = float(v)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        mantisa, exponente = f'{v:.1e}'.split('e')
        potencia = f'10{str(int(exponente)).translate(SUPERINDICES)}'
        return potencia if mantisa in ('1.0', '1') else f'{mantisa.rstrip("0").rstrip(".").replace(".", ",")} × {potencia}'
    return f'{v:.4g}'.replace('.', ',')


def _rango(definicion) -> str:
    tipo = definicion[0]
    if tipo == 'cat':
        return '{' + ', '.join(_valor(x) for x in definicion[1]) + '}'
    escala = {'int': 'entero', 'int_log': 'entero, escala logarítmica', 'log': 'escala logarítmica', 'float': 'continuo'}[tipo]
    return f'{_valor(definicion[1])} a {_valor(definicion[2])} ({escala})'


def espacios() -> None:
    """Espacio de búsqueda de cada modelo (``src/modelos.py``) e hiperparámetros finales de la semilla 42."""
    from . import modelos
    catalogos = {t: modelos.catalogo(t, incluir_nuevo=True, incluir_original=True) for t in ('clasificacion', 'regresion')}
    textos = {t: {m: '; '.join(f'`{p.split("__", 1)[-1]}`: {_rango(d)}' for p, d in e.espacio.items()) for m, e in c.items()}
              for t, c in catalogos.items()}
    filas = []
    for modelo in dict.fromkeys(list(catalogos['clasificacion']) + list(catalogos['regresion'])):
        c, r = textos['clasificacion'].get(modelo), textos['regresion'].get(modelo)
        nombre = f'**{N[modelo]}**' if modelo == plan.ORIGINAL else N[modelo]
        if c == r:
            filas.append([nombre, 'Las dos', c])
        else:
            filas += [[nombre, etiqueta, x] for etiqueta, x in (('Clasificación', c), ('Regresión', r)) if x]
    escribir('tabla_espacios.md', tabla_md(['Modelo', 'Tarea', 'Hiperparámetros y rango de búsqueda'], filas, 'lll'))
    tabla = plan.registro().cargar()
    filas = []
    for tarea in ('clasificacion', 'regresion'):
        for modelo in plan.modelos(tarea):
            f = _fila(tabla, tarea, modelo)
            if f is None:
                continue
            finales = _json(f['hiperparametros'])
            filas.append([TAREA[tarea].split(' (')[0], f'**{N[modelo]}**' if modelo == plan.ORIGINAL else N[modelo],
                          '; '.join(f'`{p.split("__", 1)[-1]}` = {_valor(v)}' for p, v in finales.items()) or '—'])
    notas = []
    for tarea in ('clasificacion', 'regresion'):
        for modelo in plan.modelos(tarea):
            spec = catalogos[tarea][modelo]
            if plan.combinacion(tarea, modelo)[1] != 'grid':
                continue
            fuera = [q for q in spec.espacio if q not in spec.espacio_grid]
            if fuera:
                base = spec.construir(SEED).get_params()
                notas.append(f'{N[modelo]} ({NOMBRE_TAREA[tarea]}), ' + ', '.join(
                    f'`{q.split("__", 1)[-1]}` = {_valor(base[q.split("__", 1)[-1]])}' for q in fuera))
    nota = ('\nLa tabla lista los hiperparámetros que fijó el optimizador elegido para cada modelo. Cuando ese optimizador es la '
            'rejilla, los que la rejilla no recorre quedan en el valor de partida del modelo: ' + '; '.join(notas) + '.\n') if notas else ''
    escribir('tabla_hiperparametros.md', tabla_md(['Tarea', 'Modelo', 'Hiperparámetros finales (semilla 42)'], filas, 'lll') + nota)


def _fuera_del_espacio(spec, hiper: dict) -> list[str]:
    """Hiperparámetros recomendados que la búsqueda del proyecto no puede alcanzar, con el motivo."""
    salida = []
    for clave, v in hiper.items():
        nombre = clave.split('__', 1)[-1]
        d = spec.espacio.get(clave)
        if d is None:
            fijo = (spec.fidelidad or {}).get('final', {}).get(clave)
            salida.append(f'`{nombre}` = {_valor(v)} (no se busca' + (f'; el modelo final usa {_valor(fijo)})' if fijo is not None else ')'))
        elif d[0] == 'cat':
            if v not in d[1]:
                salida.append(f'`{nombre}` = {_valor(v)} (el espacio es {_rango(d)})')
        elif isinstance(v, str) or not (d[1] <= float(v) <= d[2]):
            salida.append(f'`{nombre}` = {_valor(v)} (el espacio va de {_valor(d[1])} a {_valor(d[2])})')
    return salida


def recomendados() -> None:
    """EBM y LightGBM con los hiperparámetros por defecto de su librería frente a los ajustados (capítulo 3)."""
    from . import modelos
    r = pd.read_csv(R / 'recomendados.csv', dtype={'pliegue': str})
    prueba = pd.read_csv(R / 'metricas_prueba.csv')
    filas, lecturas, fuera, incompletos = [], [], [], []
    for tarea in ('clasificacion', 'regresion'):
        d, signo = DEC[tarea], (1 if MAYOR[tarea] else -1)
        M = pd.read_csv(R / f'pliegues_{tarea}.csv', index_col=0)
        M = M.loc[[i for i in M.index if str(i).startswith(f'{SEED}-')]]
        M = M.loc[sorted(M.index, key=lambda i: int(str(i).split('-')[1]))]
        catalogo = modelos.catalogo(tarea, incluir_nuevo=True, incluir_original=True)
        for modelo in ('ebm', 'lightgbm'):
            x = r[(r.tarea == tarea) & (r.modelo == modelo)]
            cv, te = x[x.pliegue != 'prueba'].sort_values('pliegue'), x[x.pliegue == 'prueba']
            if len(cv) < len(M) or te.empty:
                incompletos.append(f'{N[modelo]} ({NOMBRE_TAREA[tarea]})')
                continue
            rec, aj = cv['metrica'].to_numpy(), M[modelo].to_numpy()
            aj_te = float(prueba[(prueba.tarea == tarea) & (prueba.modelo == modelo)]['auc' if tarea == 'clasificacion' else 'rmse'].iloc[0])
            rec_te = float(te['metrica'].iloc[0])
            hiper = json.loads(te['hiperparametros'].iloc[0])
            gana = int((signo * (aj - rec) > 0).sum())
            filas.append([TAREA[tarea].split(' (')[0], N[modelo], _mas_menos(rec, d), _mas_menos(aj, d), f'{gana} de {len(aj)}',
                          num(rec_te, d), num(aj_te, d), num(float(te['ajuste_s'].iloc[0]), 1)])
            mejora_cv, mejora_te = signo * (aj.mean() - rec.mean()), signo * (aj_te - rec_te)
            lecturas.append((tarea, modelo, mejora_cv, mejora_te, gana, len(aj)))
            no = _fuera_del_espacio(catalogo[modelo], hiper)
            fuera.append(f'{N[modelo]} en {NOMBRE_TAREA[tarea]}: ' + '; '.join(f'`{k.split("__", 1)[-1]}` = {_valor(v)}' for k, v in hiper.items())
                         + ('. Fuera del alcance de la búsqueda: ' + '; '.join(no) if no else '. Todos están dentro del espacio de búsqueda'))
    if not filas:
        raise FileNotFoundError(2, 'faltan medidas', str(R / 'recomendados.csv'))
    peor_cv = [x for x in lecturas if x[2] < 0]
    peor_te = [x for x in lecturas if x[3] < 0]
    caso = lambda x: f'{N[x[1]]} en {NOMBRE_TAREA[x[0]]}'                # noqa: E731
    lectura = (f'**Lectura.** En los pliegues externos la versión ajustada mejora a la recomendada en {len(lecturas) - len(peor_cv)} de los '
               f'{len(lecturas)} casos' + (f' (no en {_lista(caso(x) for x in peor_cv)})' if peor_cv else '') + '; en prueba, en '
               f'{len(lecturas) - len(peor_te)} de {len(lecturas)}' + (f' (no en {_lista(caso(x) for x in peor_te)})' if peor_te else '') + '. ')
    lectura += ('El ajuste no dejó a ningún benchmark reciente por debajo de su punto de partida.' if not peor_cv and not peor_te else
                'Donde la versión recomendada queda por delante, la diferencia se informa tal como salió: la combinación de cada benchmark se '
                'eligió con el protocolo, antes de esta medida, y no se cambia.')
    escribir('tabla_recomendados.md',
             tabla_md(['Tarea', 'Modelo', 'Recomendados: externo', 'Ajustados: externo', 'Pliegues en que gana el ajustado', 'Recomendados: prueba',
                       'Ajustados: prueba', 'Ajuste con los recomendados (s)'], filas, 'llrrrrrr')
             + '\nMétrica: AUC en clasificación y RMSE en regresión. Externo: media ± desviación en los cinco pliegues externos de la semilla 42, '
               'los mismos para las dos versiones. Recomendados: los valores por defecto de la librería (interpret y LightGBM), con el mismo '
               'preprocesamiento y el mismo tratamiento del desbalance que la combinación elegida.\n\n**Valores recomendados.** '
             + '. '.join(fuera) + '.\n\n' + lectura
             + (f' Falta por medir: {_lista(incompletos)}.' if incompletos else '') + '\n')


# ------------------------------------------------------------------ capítulos 6 y 10: registro, entorno y tabla maestra
ETIQUETA = 'protocolo-modelo-original-v1'


def registro_protocolo() -> None:
    """Comprueba con Git que el protocolo se etiquetó antes de la primera corrida definitiva y que no cambió después."""
    import re
    import subprocess
    from datetime import datetime, timedelta
    raiz = plan.RAIZ.parent

    def git(*args) -> tuple[int, str]:
        try:
            r = subprocess.run(['git', *args], cwd=str(raiz), capture_output=True, text=True, encoding='utf-8')
        except OSError as e:
            raise FileNotFoundError(2, 'falta Git', 'git') from e
        return r.returncode, r.stdout.strip()

    codigo, fecha = git('for-each-ref', '--format=%(taggerdate:iso8601)', f'refs/tags/{ETIQUETA}')
    if codigo != 0 or not fecha:
        raise FileNotFoundError(2, 'falta la etiqueta', ETIQUETA)
    _, confirmacion = git('rev-list', '-n', '1', ETIQUETA)
    etiquetado = datetime.strptime(fecha[:19], '%Y-%m-%d %H:%M:%S')
    primera = (R / 'progreso.log').read_text(encoding='utf-8').splitlines()[0]
    fin = datetime.strptime(primera[:16], '%Y-%m-%d %H:%M')
    segundos = int(re.search(r'(\d+) s\)', primera).group(1))
    inicio = fin - timedelta(seconds=segundos)                           # el registro guarda la hora de fin sin segundos:
    inicio_tarde = inicio + timedelta(seconds=59)                        # la corrida empezó entre inicio e inicio + 59 s
    m = re.search(r'\[(\d+)\] (\w+) · (\w+)', primera)
    que = f'{N.get(m.group(3), m.group(3))}, {NOMBRE_TAREA.get(m.group(2), m.group(2))}, semilla {m.group(1)}' if m else 'la primera del registro'
    sin_cambios = {a: git('diff', '--quiet', ETIQUETA, '--', a)[0] == 0
                   for a in ('ModeloOriginal/src/original_model.py', 'ModeloOriginal/experiments/protocolo.md')}
    anterior = 'Sí' if etiquetado <= inicio else 'Sí, por menos de un minuto' if etiquetado <= inicio_tarde else 'No'
    filas = [
        ['Etiqueta de Git', f'`{ETIQUETA}`, sobre la confirmación `{confirmacion[:7]}`'],
        ['Fecha de la etiqueta (la guarda Git dentro de ella)', fecha],
        ['Primera corrida definitiva', f'{que}: terminó a las {fin:%H:%M} del {fin:%Y-%m-%d} tras {entero(segundos)} s, es decir, empezó '
                                       f'entre las {inicio:%H:%M:%S} y las {inicio_tarde:%H:%M:%S} (`resultados/progreso.log`)'],
        ['¿La etiqueta es anterior a la primera corrida?', anterior],
        ['`src/original_model.py` desde la etiqueta', 'Sin cambios' if sin_cambios['ModeloOriginal/src/original_model.py'] else 'Cambió'],
        ['`experiments/protocolo.md` desde la etiqueta', 'Sin cambios' if sin_cambios['ModeloOriginal/experiments/protocolo.md'] else 'Cambió'],
    ]
    escribir('tabla_registro.md', tabla_md(['Comprobación', 'Resultado'], filas, 'll'))


def entorno() -> None:
    """Versiones del software y descripción del equipo en que se generó el informe."""
    import os
    import platform
    from importlib import metadata
    paquetes = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'xgboost', 'lightgbm', 'interpret-core', 'optuna', 'deap',
                'joblib', 'matplotlib', 'jupyter-book']
    procesador, memoria = platform.processor() or platform.machine(), None
    if platform.system() == 'Windows':                                   # nombre comercial del procesador y memoria instalada
        import ctypes
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as clave:
                procesador = winreg.QueryValueEx(clave, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
        kb = ctypes.c_ulonglong(0)
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kb)):
            memoria = kb.value / 2 ** 20
    filas = [['Python', platform.python_version()], ['Sistema operativo', f'{platform.system()} {platform.release()}'],
             ['Procesador', f'{procesador}; {os.cpu_count()} hilos'],
             ['Memoria', f'{num(memoria, 0)} GB' if memoria else 'no disponible'],
             ['Aceleración', 'Ninguna: todos los modelos se ajustan en CPU']]
    for p in paquetes:
        try:
            filas.append([f'`{p}`', metadata.version(p)])
        except metadata.PackageNotFoundError:
            filas.append([f'`{p}`', 'no instalado'])
    escribir('tabla_entorno.md', tabla_md(['Componente', 'Versión'], filas, 'll'))


def maestra() -> None:
    """Resumen de la tabla maestra: corridas, evaluaciones y tiempo de cómputo, por origen y semilla."""
    from .variantes import NOMBRES as VARIANTES
    tabla = plan.registro().cargar()
    tabla = tabla[tabla.estado == 'ok']
    tabla = tabla.assign(evaluaciones=tabla['n_evaluaciones'].fillna(0) + tabla['n_evaluaciones_final'].fillna(0),
                         origen=tabla['origen'].fillna('Proyecto final'))
    filas, horas = [], 0.0
    for (origen, semilla), g in tabla.groupby(['origen', 'semilla']):
        horas += g['tiempo_total_s'].sum() / 3600
        filas.append(['COFINFAD', origen, int(semilla), len(g),
                      ', '.join(sorted(N.get(m, VARIANTES.get(m, m)) for m in g.modelo.unique())) if g.modelo.nunique() <= 3 else f'{g.modelo.nunique()} modelos',
                      entero(g['evaluaciones'].sum()), num(g['tiempo_total_s'].sum() / 3600, 1)])
    for nombre in conjuntos.CARGADORES:
        ruta = R / 'adicionales' / nombre / 'experimentos.parquet'
        if ruta.exists():
            g = plan.registro(nombre).cargar()
            g = g[g.estado == 'ok']
            horas += g['tiempo_total_s'].sum() / 3600
            ficha = R / 'adicionales' / nombre / 'ficha.json'
            titulo = json.loads(ficha.read_text(encoding='utf-8'))['nombre'] if ficha.exists() else nombre
            filas.append([titulo, 'Proyecto final', SEED, len(g), f'{g.modelo.nunique()} modelos',
                          entero((g['n_evaluaciones'].fillna(0) + g['n_evaluaciones_final'].fillna(0)).sum()),
                          num(g['tiempo_total_s'].sum() / 3600, 1)])
    e3 = pd.read_parquet(E3 / 'experimentos.parquet')
    nota = (f'\nCada corrida es una validación anidada completa de 5 pliegues externos × 3 internos más el reajuste final, y la dispersión que '
            f'se reporta es la desviación típica entre los pliegues externos. Las corridas de esta tabla suman {num(horas, 1)} horas de cómputo. '
            f'Las {len(e3)} corridas del Entregable 3 (las 140 combinaciones de la guía y las de la EBM) están en '
            f'`Entrega3/resultados/experimentos.parquet` y sumaron {num(e3["tiempo_total_s"].sum() / 3600, 0)} horas; las filas con origen '
            f'«Entregable 3» son la mejor combinación de cada benchmark, tomada de ahí. Las corridas del proyecto final posteriores a la primera repetición de COFINFAD se ejecutaron en dos colas '
            f'simultáneas y con otras tareas en el mismo equipo: sus tiempos incluyen esa competencia por los núcleos y sirven para comparar '
            f'órdenes de magnitud, no como medida exacta. El tiempo de ajuste de cada modelo final, medido aparte, está en la Tabla 7.6.\n')
    escribir('tabla_maestra.md', tabla_md(['Conjunto', 'Origen', 'Semilla', 'Corridas', 'Modelos', 'Evaluaciones', 'Tiempo (h)'], filas, 'llrrlrr') + nota)


INICIO_README, FIN_README = '<!-- resultados-proyecto-final:inicio -->', '<!-- resultados-proyecto-final:fin -->'


def portada() -> None:
    """Resultados principales para la portada del libro y, entre dos marcas, para el README del repositorio."""
    h = _hechos()
    reg, cla, h3 = h['regresion'], h['clasificacion'], h['h3']
    puntos = ['**Modelo original: BDL (Boosting con Desplazamiento Latente)**, programado desde cero con `numpy` y `scipy`. Modela como una clase '
              'latente el escalón que deja en el puntaje la variable excluida por fuga de información y lo aprende con un EM generalizado cuyo '
              'paso M es un paso de boosting.']
    for t, nombre in ((reg, 'Regresión'), (cla, 'Clasificación')):
        d, n = t['d'], t['r']['pliegues']
        puntos.append(
            f'**{nombre}** ({t["metrica"]}): en los {n} pliegues externos de la validación anidada, BDL {num(t["cv_o"], d)} y el mejor benchmark '
            f'({t["nb"]}) {num(t["cv_b"], d)}; en prueba, {num(t["pr_o"], d)} frente a {num(t["pr_b"], d)}, con un límite teórico de '
            f'{num(t["limite"], d)}. {_cuantas(len(t["rechazan"]), len(t["pruebas"])).capitalize()} pruebas de hipótesis registradas '
            f'{"rechaza" if len(t["rechazan"]) <= 1 else "rechazan"} la igualdad a favor de BDL (p ajustado por Holm < 0,05): '
            + ('**superior**.' if t['superior'] else '**pendiente de los conjuntos adicionales**.' if t['pendiente'] else '**no superior** según la guía, que pide tres.'))
    puntos.append(f'**Incertidumbre:** con cobertura del 90 %, los conjuntos de predicción de BDL miden {num(h3["largo_conjunto"], 4)} y los intervalos '
                  f'conformes del mejor benchmark, {num(h3["largo_benchmark"], 4)}; la cobertura medida en prueba es {num(100 * h3["cobertura"], 1)} %.')
    fuera = _frase_adicionales(h)
    if fuera:
        puntos.append('**Conjuntos adicionales:** ' + fuera)
    puntos.append('**Recencia:** la fecha de corte se calcula solo con los clientes de entrenamiento; la auditoría muestra que no cambia ningún dato.')
    if any(h[t]['repeticiones'] < 3 for t in MAYOR):
        puntos.append('*Resultados provisionales: faltan repeticiones de la validación anidada.*')
    texto = '\n'.join(f'- {x}' for x in puntos) + '\n'
    escribir('resultados_principales.md', texto)
    readme = plan.RAIZ.parent / 'README.md'
    actual = readme.read_text(encoding='utf-8')
    if INICIO_README in actual and FIN_README in actual:
        antes, resto = actual.split(INICIO_README, 1)
        despues = resto.split(FIN_README, 1)[1]
        readme.write_text(antes + INICIO_README + '\n' + texto + FIN_README + despues, encoding='utf-8', newline='\n')
        _aviso('   README.md (bloque de resultados del proyecto final)')


PIEZAS = [literatura, datos_eda, fuga, figura_escalones, figuras_eda, benchmarks, resumen, pruebas, figura_cd, techo, convergencia, ablacion,
          importancia, adicionales, veredicto, sensibilidad, ruido, muestra, ciudades, semillas, calibracion, incertidumbre, errores,
          veredicto_texto, significancia_practica, adicionales_texto, errores_texto, hallazgos, resumen_ejecutivo, portada,
          registro_protocolo, entorno, maestra, espacios, optimo, empates, orden_modelos, error_indice, recomendados]
# cada inclusión del libro que todavía no se puede generar queda con un aviso, para que el libro siempre compile
ESPERADOS = ['tabla_matriz', 'tabla_cadenas', 'tabla_citacion', 'tabla_fuga', 'tabla_benchmarks', 'tabla_resumen_regresion', 'tabla_resumen_clasificacion',
             'tabla_pruebas_regresion', 'tabla_pruebas_clasificacion', 'tabla_techo', 'tabla_bic', 'tabla_ablacion', 'tabla_importancia',
             'tabla_conjuntos', 'tabla_adicionales', 'tabla_consolidada', 'tabla_veredicto', 'tabla_sensibilidad', 'tabla_semillas', 'tabla_ruido',
             'tabla_muestra', 'tabla_columnas', 'tabla_ciudades', 'tabla_calibracion', 'tabla_incertidumbre', 'tabla_errores', 'veredicto_texto',
             'significancia_practica', 'adicionales_texto', 'errores_texto', 'hallazgos', 'resumen_ejecutivo', 'resultados_principales',
             'tabla_registro', 'tabla_entorno', 'tabla_maestra', 'tabla_espacios', 'tabla_hiperparametros', 'tabla_optimo',
             'tabla_empates', 'orden_modelos', 'error_indice', 'tabla_inferencia', 'tabla_etl', 'tabla_cohortes', 'tabla_asociacion',
             'tabla_faltantes', 'tabla_declaracion', 'tabla_recomendados']
AVISO = '*Resultado en ejecución: esta parte se completa cuando termine el experimento correspondiente.*\n'



def principal(solo=()) -> None:
    """Regenera todas las piezas (o solo las nombradas) y deja un aviso en las que todavía no se pueden generar."""
    solo = set(solo)
    for f in PIEZAS:
        if not solo or f.__name__ in solo:
            pieza(f)
    for nombre in ESPERADOS:
        ruta = GEN / f'{nombre}.md'
        if not ruta.exists():
            ruta.write_text(AVISO, encoding='utf-8', newline='\n')
            _aviso(f'   [aviso] {nombre}.md queda como pendiente')


# ------------------------------------------------------------------ interfaz para los cuadernos del libro
# qué función genera cada tabla o texto (archivo de _generado) y cada figura (archivo de figuras)
PRODUCE = {
    'tabla_matriz': literatura, 'tabla_cadenas': literatura, 'tabla_citacion': literatura, 'tabla_fuga': fuga, 'tabla_benchmarks': benchmarks,
    'tabla_resumen_regresion': resumen, 'tabla_resumen_clasificacion': resumen, 'tabla_pruebas_regresion': pruebas,
    'tabla_pruebas_clasificacion': pruebas, 'tabla_techo': techo, 'tabla_bic': convergencia, 'tabla_ablacion': ablacion,
    'tabla_importancia': importancia, 'tabla_conjuntos': adicionales, 'tabla_adicionales': adicionales,
    'tabla_consolidada': adicionales, 'tabla_veredicto': veredicto, 'tabla_sensibilidad': sensibilidad, 'tabla_ruido': ruido,
    'tabla_muestra': muestra, 'tabla_columnas': muestra, 'tabla_ciudades': ciudades, 'tabla_semillas': semillas,
    'tabla_calibracion': calibracion, 'tabla_incertidumbre': incertidumbre, 'tabla_errores': errores,
    'veredicto_texto': veredicto_texto, 'significancia_practica': significancia_practica, 'adicionales_texto': adicionales_texto,
    'errores_texto': errores_texto, 'hallazgos': hallazgos, 'resumen_ejecutivo': resumen_ejecutivo, 'resultados_principales': portada,
    'tabla_registro': registro_protocolo, 'tabla_entorno': entorno, 'tabla_maestra': maestra,
    'tabla_espacios': espacios, 'tabla_hiperparametros': espacios, 'tabla_optimo': optimo, 'tabla_empates': empates,
    'orden_modelos': orden_modelos, 'error_indice': error_indice, 'tabla_inferencia': muestra,
    'tabla_etl': datos_eda, 'tabla_cohortes': datos_eda, 'tabla_asociacion': datos_eda, 'tabla_faltantes': datos_eda, 'tabla_declaracion': datos_eda,
    'tabla_recomendados': recomendados,
}
DIBUJA = {
    'residuos_escalones': figura_escalones, 'escalera_productos': figuras_eda, 'objetivo': figuras_eda, 'cd_regresion': figura_cd,
    'cd_clasificacion': figura_cd, 'convergencia': convergencia, 'formas_bdl': importancia, 'sensibilidad': sensibilidad,
    'aprendizaje': muestra, 'semillas': semillas, 'calibracion': calibracion, 'distribucion_predictiva': incertidumbre,
}
_HECHAS: set = set()


def _generar(funcion) -> None:
    """Ejecuta una función generadora una sola vez por sesión, sin avisos.

    Si falta un resultado, la celda muestra la pieza ya generada que está versionada (o el aviso de pendiente). Con la
    variable de entorno ``INFORME_ESTRICTO=1`` la falta es un error: sirve para comprobar, en una copia limpia del
    repositorio, que todo se regenera con lo que está versionado.
    """
    global VERBOSO
    if funcion in _HECHAS:
        return
    antes, VERBOSO = VERBOSO, False
    try:
        funcion()
    except FileNotFoundError:
        if os.environ.get('INFORME_ESTRICTO') == '1':
            raise
    finally:
        VERBOSO = antes
    _HECHAS.add(funcion)


def tabla(*nombres: str):
    """Regenera, a partir de los resultados, las tablas pedidas y las muestra en la celda del cuaderno.

    Cada nombre es un archivo de ``ProyectoFinal/_generado`` (sin extensión). El texto se devuelve como Markdown para
    que el libro lo componga igual que el resto del capítulo.
    """
    from IPython.display import Markdown, display
    for nombre in nombres:
        _generar(PRODUCE[nombre])
        ruta = GEN / f'{nombre}.md'
        display(Markdown(ruta.read_text(encoding='utf-8') if ruta.exists() else AVISO))


texto = tabla                                                             # los párrafos generados se muestran igual que las tablas


def figura(nombre: str):
    """Vuelve a dibujar, a partir de los resultados, la figura pedida (``ProyectoFinal/figuras``) y la devuelve como imagen."""
    from IPython.display import Image, Markdown
    _generar(DIBUJA[nombre])
    ruta = FIG / f'{nombre}.png'
    return Image(filename=str(ruta)) if ruta.exists() else Markdown(AVISO)
