"""Genera las tablas (LaTeX) y figuras del artículo a partir de los datos y de los resultados.

Ninguna cifra de las tablas se escribe a mano: salen del conjunto de datos (EDA), de la tabla maestra,
de las predicciones de prueba, de la comparación con la Entrega 2 y de los experimentos de cómputo.
Las figuras del libro se copian de ``presentacion/figuras`` (exportadas por ``exportar_figuras.py``).
Todas las tablas están pensadas para caber derechas en el ancho de texto de la plantilla sn-jnl
(unos 360 pt): pocas columnas, encabezados cortos y letra pequeña.

Uso, desde la carpeta del proyecto::

    .venv\\Scripts\\python.exe articulo/generar_material.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import analisis, datos, graficos, modelos  # noqa: E402
from src.analisis import NOMBRES_MODELO  # noqa: E402

AQUI = RAIZ / 'articulo'
FIG, TAB = AQUI / 'figuras', AQUI / 'tablas'
RES = RAIZ / 'resultados'
FIGURAS_LIBRO = {  # archivo exportado del libro -> nombre en el artículo
    '01_datos_1.png': 'objetivo_escalera.png', '01_datos_2.png': 'senal_bivariado.png', '01_datos_3.png': 'correlacion.png',
    '06_evaluacion_1.png': 'roc.png', '08_modelo_nuevo_3.png': 'formas_ebm.png', '09_estadistica_1.png': 'cd_modelos.png',
}
NOMBRE_BALANCEO = {'ninguno': '--', 'smote': 'SMOTE', 'adasyn': 'ADASYN', 'class_weight': 'pesos'}
NOMBRE_OPT = {'grid': 'Grid', 'random': 'Random', 'bayesiana': 'Bayesiana', 'genetica': 'Genética'}
NOMBRE_CMP = {'logistica_e2': 'Logística (E2)', 'lasso_e2': 'Lasso (E2)', 'xgboost': 'XGBoost', 'ebm': 'EBM'}


# ----------------------------------------------------------------------------------------------- formato
def tex(s: str) -> str:
    """Escapa lo que LaTeX interpretaría dentro de una tabla."""
    return (str(s).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
            .replace('#', '\\#').replace('×', '$\\times$').replace('→', '$\\rightarrow$').replace('≥', '$\\geq$')
            .replace('≤', '$\\leq$').replace('−', '$-$').replace('±', '$\\pm$').replace('·', '$\\cdot$')
            .replace('ρ', '$\\rho$').replace('²', '$^2$').replace('³', '$^3$'))


def num(x, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    return f'{x:,.{d}f}'.replace(',', '@').replace('.', ',').replace('@', '.')


def num_compacto(x) -> str:
    """Cifras grandes en millones (M) y medianas sin decimales, para que la tabla descriptiva quepa."""
    if abs(x) >= 1e6:
        return num(x / 1e6, 1) + '\\,M'
    if abs(x) >= 1e3:
        return num(x, 0)
    return num(x, 3)


def pm(media, desv, d: int = 4) -> str:
    return f'{num(media, d)} $\\pm$ {num(desv, d)}'


def p_tex(p: float) -> str:
    if p <= 0:
        return '$<0{,}001$'
    if p < 1e-3:
        return f'$<10^{{{int(np.floor(np.log10(p))) + 1}}}$'
    return num(p, 3)


def tabla_latex(nombre: str, columnas: list[str], filas: list[list[str]], alineacion: str, caption: str, label: str,
                tamano: str = '\\scriptsize') -> None:
    cuerpo = ' \\\\\n'.join(' & '.join(f) for f in filas) + ' \\\\'
    cabecera = ' & '.join(f'\\textbf{{{c}}}' for c in columnas)
    texto = (f'\\begin{{table}}[htbp]\n\\caption{{{caption}}}\\label{{{label}}}\n{tamano}\\setlength{{\\tabcolsep}}{{2pt}}\n'
             f'\\begin{{tabular}}{{{alineacion}}}\n\\toprule\n{cabecera} \\\\\n\\midrule\n'
             f'{cuerpo}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n')
    (TAB / f'{nombre}.tex').write_text(texto, encoding='utf-8')
    print(f'tabla {nombre}: {len(filas)} filas')


P = '>{\\raggedright\\arraybackslash}p{%s}'   # columna de texto alineada a la izquierda


# ------------------------------------------------------------------------------------------------ EDA
def material_eda(p: dict) -> dict:
    c = p['conjuntos']
    X, y_cont, y_bin = c.X_train, c.y_train_cont, p['y_train']
    numericas = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) and X[col].dtype != bool and X[col].nunique() > 2]
    categoricas = [col for col in X.columns if col not in numericas]
    d = X[numericas].astype(float)
    # Estadística descriptiva (entrenamiento)
    filas = []
    for col in numericas + ['churn_probability']:
        s = d[col] if col in d else y_cont.astype(float)
        filas.append([tex(col), num_compacto(s.mean()), num_compacto(s.median()), num_compacto(s.std()), num_compacto(s.min()),
                      num_compacto(s.max()), str(int(s.nunique()))])
    tabla_latex('descriptiva', ['Variable', 'Media', 'Mediana', 'Desv.', 'Mín.', 'Máx.', 'Dist.'], filas, 'lrrrrrr',
                'Estadística descriptiva de los predictores numéricos y del objetivo en el entrenamiento (38.978 clientes; M = millones). '
                'Ningún predictor tiene valores faltantes tras la depuración.', 'tab:descriptiva')
    # Categóricas y booleanas
    filas = []
    for col in categoricas:
        s = X[col].astype(str)
        moda = s.value_counts(normalize=True)
        filas.append([tex(col), str(int(s.nunique())), tex(moda.index[0]), num(100 * moda.iloc[0], 1) + '\\,\\%'])
    tabla_latex('categoricas', ['Variable', 'Niveles', 'Nivel más frecuente', 'Frecuencia'], filas, 'lrlr',
                'Variables categóricas y booleanas del conjunto de modelado (entrenamiento): cardinalidad y nivel más frecuente.',
                'tab:categoricas')
    # Atípicos por la regla de Tukey (1,5 IQR)
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    iqr = q3 - q1
    fuera = ((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).mean().sort_values(ascending=False)
    filas = [[tex(col), num(100 * v, 1) + '\\,\\%'] for col, v in fuera.head(8).items() if v > 0]
    tabla_latex('atipicos', ['Variable', 'Fuera de 1,5 IQR'], filas, 'lr',
                'Variables con mayor proporción de valores atípicos según la regla de Tukey (1,5 veces el rango intercuartílico), '
                'en el entrenamiento.', 'tab:atipicos')
    graficos.estilo()
    # Figura: diagramas de caja estandarizados
    z = (d - d.mean()) / d.std(ddof=0)
    orden = list(fuera.index)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.boxplot([z[c_].dropna().to_numpy() for c_ in orden], tick_labels=orden, flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.4},
               medianprops={'color': '#d9822b'})
    ax.set_ylabel('valor estandarizado (z)')
    ax.set_title('Predictores numéricos estandarizados (entrenamiento): valores atípicos según la regla de Tukey')
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'eda_boxplots.png', dpi=160)
    plt.close(fig)
    # Figura: distribuciones univariadas de las variables con más señal y balance de clases
    rho = d.corrwith(y_cont.astype(float), method='spearman').abs().sort_values(ascending=False)
    top = list(rho.index[:5])
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, col in zip(axes.ravel(), top):
        s = d[col]
        if s.nunique() <= 12:
            vc = s.value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.to_numpy(), color='#3d78bf')
        else:
            ax.hist(s, bins=40, color='#7fb0e0', edgecolor='white')
        ax.set_title(f'{col} ($\\rho$ = {rho[col] * np.sign(d[col].corr(y_cont, method="spearman")):+.3f})', fontsize=9)
    ax = axes.ravel()[-1]
    etiquetas = ['riesgo alto\n(percentil 75)', 'resto', 'riesgo alto\n(mediana)', 'resto']
    p75 = float(y_bin.mean())
    med = float((y_cont > y_cont.median()).mean())
    ax.bar(etiquetas, [p75, 1 - p75, med, 1 - med], color=['#d9822b', '#a8cbec', '#d9822b', '#a8cbec'])
    for i, v in enumerate([p75, 1 - p75, med, 1 - med]):
        ax.text(i, v + 0.01, f'{100 * v:.1f} %', ha='center', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title('Balance de clases según la etiqueta', fontsize=9)
    fig.suptitle('Distribución de los cinco predictores más asociados al objetivo (entrenamiento) y balance de la etiqueta', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / 'eda_univariado.png', dpi=160)
    plt.close(fig)
    return {'n_train': int(len(X)), 'n_test': int(len(c.X_test)), 'p_numericas': len(numericas), 'p_categoricas': len(categoricas),
            'prop_riesgo_alto_train': p75, 'prop_riesgo_alto_test': float(p['y_test'].mean()), 'umbral': float(p['etiqueta'].umbral),
            'top_rho': {k: float(v) for k, v in rho.head(6).items()}, 'atipicos_max': {k: float(v) for k, v in fuera.head(8).items()}}


# ------------------------------------------------------------------------------------------- resultados
def material_resultados(tabla: pd.DataFrame, p: dict) -> dict:
    X_te = p['conjuntos'].X_test.sort_index()
    y_te, y_te_c = p['y_test'].loc[X_te.index], p['conjuntos'].y_test_cont.loc[X_te.index]
    pred = analisis.predecir_prueba(tabla, X_te, RES / 'predicciones_prueba.parquet', informar=lambda s: None)
    metricas = analisis.metricas_prueba(tabla, pred, y_te, y_te_c)
    metricas.to_csv(RES / 'metricas_prueba.csv', index=False)   # se actualiza con los 16 modelos (antes faltaba la EBM)
    tiempos_ruta = RES / 'computo' / 'tiempos_finales.csv'
    tiempos = pd.read_csv(tiempos_ruta) if tiempos_ruta.exists() else None
    clave = ['tarea', 'modelo', 'balanceo', 'optimizador']
    cifras = {}
    tiempos_por_modelo: dict[str, dict] = {}
    for tarea in ('clasificacion', 'regresion'):
        mejores = analisis.mejores_por_modelo(tabla, tarea)
        m = mejores.merge(metricas, on=clave, how='left', suffixes=('', '_prueba'))
        if tiempos is not None:
            m = m.merge(tiempos[clave + ['ajuste_s', 'inferencia_s', 'tamano_modelo_mb']], on=clave, how='left')
        filas = []
        for _, f in m.iterrows():
            nombre = NOMBRES_MODELO[f['modelo']]
            if f['modelo'] == 'ebm':
                nombre = '\\textbf{EBM (propuesto)}'
            fila = [nombre] + ([NOMBRE_BALANCEO[f['balanceo']]] if tarea == 'clasificacion' else []) + [NOMBRE_OPT[f['optimizador']]]
            if tarea == 'clasificacion':
                fila += [pm(f['metrica'], f['desv_ext']), num(f['auc'])]
            else:
                fila += [pm(f['metrica'], f['desv_ext']), num(f['rmse']), num(f['r2'], 3)]
            filas.append(fila)
            if tiempos is not None:
                t = tiempos_por_modelo.setdefault(f['modelo'], {'nombre': nombre})
                t[tarea] = (f['ajuste_s'], f['inferencia_s'], f['tiempo_total_s'] / 60)
        if tarea == 'clasificacion':
            cols = ['Modelo', 'Balanceo', 'Optimizador', 'AUC externa', 'AUC prueba']
            cap = ('Clasificación (etiqueta del percentil 75): mejor combinación de balanceo y optimizador de cada modelo según el bucle '
                   'externo, AUC externa (media $\\pm$ desviación de los 5 pliegues) y AUC en la prueba (9.745 clientes). \\emph{Pesos} es la '
                   'ponderación de clases; los tiempos están en la Tabla~\\ref{tab:tiempos}.')
        else:
            cols = ['Modelo', 'Optimizador', 'RMSE externo', 'RMSE prueba', '$R^2$ prueba']
            cap = ('Regresión de \\texttt{churn\\_probability}: mejor optimizador de cada modelo según el bucle externo, RMSE externo '
                   '(media $\\pm$ desviación de los 5 pliegues) y RMSE y $R^2$ en la prueba.')
        n_texto = 3 if tarea == 'clasificacion' else 2
        tabla_latex(tarea, cols, filas, 'l' * n_texto + 'r' * (len(cols) - n_texto), cap, f'tab:{tarea}')
        cifras[tarea] = m[['modelo', 'metrica', 'desv_ext'] + (['ajuste_s', 'inferencia_s'] if tiempos is not None else [])].to_dict('records')
    if tiempos is not None:
        filas = []
        for tarea in ('clasificacion', 'regresion'):
            for i, (_, f) in enumerate(analisis.mejores_por_modelo(tabla, tarea).iterrows()):
                t = tiempos_por_modelo[f['modelo']]
                a, inf, corrida = t[tarea]
                filas.append([graficos.NOMBRE_TAREA[tarea] if i == 0 else '', t['nombre'], num(a, 2), num(inf, 3), num(corrida, 1)])
        tabla_latex('tiempos', ['Tarea', 'Modelo', 'Ajuste (s)', 'Inferencia (s)', 'Corrida (min)'], filas, 'llrrr',
                    'Tiempo de cómputo de la mejor combinación de cada modelo: ajuste del modelo final sobre los 38.978 clientes de '
                    'entrenamiento y puntuación de los 9.745 de prueba (mediana de 3 y 5 repeticiones con el pipeline completo), y '
                    'duración de la corrida anidada completa (búsqueda $5\\times3$ más modelo final).', 'tab:tiempos')
    # Comparación con la Entrega 2 (etiqueta de la mediana): métricas y contrastes en dos tablas compactas
    cmp_dir = RES / 'comparacion_entrega2'
    busq = json.loads((cmp_dir / 'busquedas.json').read_text(encoding='utf-8'))
    c_clf = pd.read_csv(cmp_dir / 'contrastes_clasificacion.csv')
    c_reg = pd.read_csv(cmp_dir / 'contrastes_regresion.csv')
    techo = json.loads((cmp_dir / 'techo.json').read_text(encoding='utf-8'))
    nombres = {'logistica_e2': 'Logística L1 (Entrega 2)', 'lasso_e2': 'Lasso (Entrega 2)', 'xgboost': 'XGBoost', 'ebm': '\\textbf{EBM (propuesto)}'}
    filas = []
    for clave_m, nombre in nombres.items():
        clf = busq['modelos'].get(f'clasificacion/{clave_m}', {}).get('prueba', {})
        reg = busq['modelos'].get(f'regresion/{clave_m}', {}).get('prueba', {})
        filas.append([nombre, num(clf.get('auc', np.nan)), num(clf.get('recall', np.nan)), num(reg.get('rmse', np.nan)), num(reg.get('r2', np.nan))])
    filas.append(['Techo (oráculo)', num(techo['auc_mediana']), '--', '--', num(techo['r2'])])
    tabla_latex('entrega2', ['Modelo', 'AUC', 'Recall (0,5)', 'RMSE', '$R^2$'], filas, 'lrrrr',
                'Comparación con los modelos de la Entrega 2 en su misma partición, etiqueta (mediana estimada con el entrenamiento) y '
                'prueba. Cada modelo toca la prueba una sola vez; el techo es el mejor pronóstico posible sin \\texttt{active\\_products}.',
                'tab:entrega2')
    filas = []
    for _, f in c_clf.iterrows():
        filas.append([f"{NOMBRE_CMP[f['referencia']]} $\\rightarrow$ {NOMBRE_CMP[f['candidato']]}", 'AUC',
                      f"{num(f['diferencia'])} [{num(f['ic_bca_inf'])}; {num(f['ic_bca_sup'])}]", f"$z$ = {num(f['z'], 2)}",
                      p_tex(f['p_delong_ajustado'])])
    for _, f in c_reg.iterrows():
        filas.append([f"{NOMBRE_CMP[f['referencia']]} $\\rightarrow$ {NOMBRE_CMP[f['candidato']]}", '$R^2$',
                      f"{num(f['mejora_r2'])} [{num(f['mejora_r2_ic_inf'])}; {num(f['mejora_r2_ic_sup'])}]", f"DM = {num(f['dm_hln'], 2)}",
                      p_tex(f['p_dm_ajustado'])])
    tabla_latex('contrastes', ['Comparación', 'Métrica', '$\\Delta$ [IC BCa 95\\,\\%]', 'Estadístico', '$p$ (Holm)'], filas, 'llrlr',
                'Contrastes sobre la prueba con la etiqueta de la mediana: diferencia de métrica del candidato con intervalo bootstrap BCa '
                '(2.000 réplicas), estadístico $z$ de DeLong para el AUC y DM de Diebold-Mariano con la corrección de '
                'Harvey-Leybourne-Newbold para la pérdida cuadrática, con $p$-valores ajustados por Holm.', 'tab:contrastes')
    cifras['entrega2'] = {'contrastes_clf': c_clf.to_dict('records'), 'contrastes_reg': c_reg.to_dict('records'), 'techo': techo}
    return cifras


# ------------------------------------------------------------------------------------------ optimizadores
def material_optimizadores(tabla: pd.DataFrame) -> None:
    res = analisis.resumen_optimizadores(tabla)
    filas = []
    for tarea in ('clasificacion', 'regresion'):
        _, por_busqueda = analisis.curvas_anytime(tabla, tarea)
        area = por_busqueda.groupby('optimizador')['area'].mean()
        ev95 = por_busqueda.groupby('optimizador')['evals_al_95'].mean()
        for opt in ('grid', 'random', 'bayesiana', 'genetica'):
            r = res.loc[(tarea, opt)]
            filas.append([graficos.NOMBRE_TAREA[tarea] if opt == 'grid' else '', NOMBRE_OPT[opt], num(r['rango_medio'], 2), str(int(r['victorias'])),
                          num(r['s_por_evaluacion'], 2), num(area[opt], 3), num(ev95[opt], 1)])
    tabla_latex('optimizadores', ['Tarea', 'Método', 'Rango medio', 'Victorias', 's / eval.', 'Área arrep.', 'Evals. 95\\,\\%'], filas, 'llrrrrr',
                'Los cuatro métodos de optimización con el mismo presupuesto por modelo (27 bloques modelo $\\times$ balanceo en '
                'clasificación y 7 en regresión): rango medio por la métrica externa (1 = mejor), bloques ganados, segundos por '
                'evaluación, área media bajo la curva de arrepentimiento normalizado (menor es mejor) y evaluaciones hasta cerrar el '
                '95\\,\\% de la brecha.', 'tab:optimizadores')


# ----------------------------------------------------------------------------------------------- cómputo
def material_computo() -> None:
    carpeta = RES / 'computo'
    leer = lambda n: pd.read_csv(carpeta / f'{n}.csv')
    filas = []
    xg = leer('xgboost')
    m = xg[xg['experimento'] == 'método y dispositivo'].set_index('variante')
    ex, hi, gp = m[m.index.str.startswith('exact')].iloc[0], m.loc['hist · cpu · todos los hilos'], m[m.index.str.contains('cuda')].iloc[0]
    filas.append(['XGBoost', 'exact (CPU)', 'hist (CPU, 6 hilos)', num(ex['entrenamiento_s'], 2), num(hi['entrenamiento_s'], 2), num(ex['entrenamiento_s'] / hi['entrenamiento_s'], 1), num(hi['auc'] - ex['auc'], 4)])
    filas.append(['XGBoost', 'hist (CPU)', 'hist (GPU RTX 3050)', num(hi['entrenamiento_s'], 2), num(gp['entrenamiento_s'], 2), num(hi['entrenamiento_s'] / gp['entrenamiento_s'], 2), num(gp['auc'] - hi['auc'], 4)])
    es = xg[xg['experimento'] == 'early stopping'].reset_index(drop=True)
    filas.append(['XGBoost', '500 árboles fijos', f"parada temprana ({es.loc[1, 'arboles']:.0f} árboles)", num(es.loc[0, 'entrenamiento_s'], 2), num(es.loc[1, 'entrenamiento_s'], 2), num(es.loc[0, 'entrenamiento_s'] / es.loc[1, 'entrenamiento_s'], 1), num(es.loc[1, 'auc'] - es.loc[0, 'auc'], 4)])
    knn = leer('knn')
    k = knn[knn['dimension'] == '78 variables'].set_index('metodo')
    k4 = knn[knn['dimension'] == '4 variables con señal'].set_index('metodo')
    filas.append(['k-NN (consulta)', 'fuerza bruta, 78 var.', 'KD-Tree, 78 var.', num(k.loc['scikit-learn brute', 'consulta_s'], 2), num(k.loc['scikit-learn kd_tree', 'consulta_s'], 2), num(k.loc['scikit-learn brute', 'consulta_s'] / k.loc['scikit-learn kd_tree', 'consulta_s'], 2), num(0, 4)])
    filas.append(['k-NN (consulta)', 'fuerza bruta, 78 var.', 'FAISS IndexFlat, 78 var.', num(k.loc['scikit-learn brute', 'consulta_s'], 2), num(k.loc['FAISS IndexFlat (exacto)', 'consulta_s'], 2), num(k.loc['scikit-learn brute', 'consulta_s'] / k.loc['FAISS IndexFlat (exacto)', 'consulta_s'], 2), num(k.loc['FAISS IndexFlat (exacto)', 'auc'] - k.loc['scikit-learn brute', 'auc'], 4)])
    filas.append(['k-NN (consulta)', 'fuerza bruta, 78 var.', 'KD-Tree, 4 var. con señal', num(k.loc['scikit-learn brute', 'consulta_s'], 2), num(k4.loc['scikit-learn kd_tree', 'consulta_s'], 2), num(k.loc['scikit-learn brute', 'consulta_s'] / k4.loc['scikit-learn kd_tree', 'consulta_s'], 1), num(k4.loc['scikit-learn kd_tree', 'auc'] - k.loc['scikit-learn brute', 'auc'], 4)])
    lin = leer('lineales')
    r = lin[lin['modelo'] == 'Ridge'].set_index('solucionador')
    filas.append(['Ridge', 'SAGA', 'Cholesky', num(r.loc['saga', 'entrenamiento_s'], 3), num(r.loc['cholesky', 'entrenamiento_s'], 3), num(r.loc['saga', 'entrenamiento_s'] / r.loc['cholesky', 'entrenamiento_s'], 0), num(r.loc['cholesky', 'valor'] - r.loc['saga', 'valor'], 4)])
    lg = lin[lin['modelo'] == 'Logística L1'].set_index('solucionador')
    filas.append(['Logística L1', 'SAGA', 'liblinear', num(lg.loc['saga', 'entrenamiento_s'], 2), num(lg.loc['liblinear', 'entrenamiento_s'], 2), num(lg.loc['saga', 'entrenamiento_s'] / lg.loc['liblinear', 'entrenamiento_s'], 1), num(lg.loc['liblinear', 'valor'] - lg.loc['saga', 'valor'], 4)])
    svm = leer('svm')
    rbf = svm[svm['modelo'] == 'SVC núcleo RBF']
    a, b = np.polyfit(np.log(rbf['n']), np.log(rbf['entrenamiento_s']), 1)
    n_total = svm['n'].max()
    extrap = float(np.exp(b) * n_total ** a)
    lineal = svm[(svm['modelo'] == 'LinearSVC (primal, L2)') & (svm['n'] == n_total)].iloc[0]
    filas.append(['SVM', 'SVC RBF (extrapolado)', 'LinearSVC primal L2', num(extrap, 1), num(lineal['entrenamiento_s'], 2), num(extrap / lineal['entrenamiento_s'], 0), '--'])
    par = leer('paralelismo')
    rf = par[par['experimento'].str.startswith('Random Forest')].set_index('n_jobs')['tiempo_s']
    filas.append(['Random Forest', '1 hilo', '6 hilos', num(rf.loc[1], 1), num(rf.loc[6], 1), num(rf.loc[1] / rf.loc[6], 1), num(0, 4)])
    fid = leer('fidelidad')
    for mod, g in fid.groupby('modelo'):
        filas.append([NOMBRES_MODELO[mod], 'búsqueda, fidelidad alta', 'búsqueda, fidelidad baja', num(g['tiempo_alta_s'].sum(), 1), num(g['tiempo_baja_s'].sum(), 1), num(g['tiempo_alta_s'].sum() / g['tiempo_baja_s'].sum(), 1), num(g.loc[g['auc_baja'].idxmax(), 'auc_alta'] - g['auc_alta'].max(), 4)])
    tabla_latex('computo', ['Modelo', 'Estándar', 'Optimizada', '$t$ est. (s)', '$t$ opt. (s)', 'Acel.', '$\\Delta$ métrica'], filas,
                'l' + P % '2.0cm' + P % '2.2cm' + 'rrrr',
                'Versión estándar frente a versión optimizada, con los hiperparámetros finales en el equipo de referencia (6 núcleos, '
                'RTX 3050): tiempo de entrenamiento (en k-NN, de consulta sobre la prueba) y cambio en la métrica de prueba (AUC; RMSE en Ridge).',
                'tab:computo')


# -------------------------------------------------------------------------------------------- espacios
def material_espacios(tabla: pd.DataFrame) -> None:
    def describir(espacio):
        sub = lambda t: str(t).replace('_', chr(92) + '_')
        partes = []
        for k, (tipo, *args) in espacio.items():
            nombre = sub(k.split('__', 1)[1])
            if tipo == 'cat':
                partes.append(nombre + r' $\in$ \{' + ', '.join(sub(v) for v in args[0]) + r'\}')
            else:
                escala = {'log': 'log', 'int_log': 'entero log', 'int': 'entero', 'float': 'lineal'}.get(tipo, tipo)
                partes.append(rf"{nombre} $\in$ [{args[0]:g}, {args[1]:g}] ({escala})")
        return '; '.join(partes)
    for tarea in ('clasificacion', 'regresion'):
        mejores = analisis.mejores_por_modelo(tabla, tarea).set_index('modelo')
        filas = []
        for nombre, spec in modelos.catalogo(tarea, incluir_nuevo=True).items():
            hp = mejores.loc[nombre, 'hiperparametros'] if nombre in mejores.index else {}
            hp = json.loads(hp) if isinstance(hp, str) else dict(hp)
            finales = ', '.join(f"{k.split('__', 1)[1]}={v:.4g}" if isinstance(v, float) else f"{k.split('__', 1)[1]}={v}" for k, v in hp.items())
            filas.append([NOMBRES_MODELO[nombre], str(spec.n_configuraciones_grid()), describir(spec.espacio), tex(finales)])
        tabla_latex(f'espacios_{tarea}', ['Modelo', 'Rejilla', 'Espacio de búsqueda', 'Hiperparámetros finales'], filas,
                    'lr' + P % '4.7cm' + P % '3.3cm',
                    f'Espacios de búsqueda en {graficos.NOMBRE_TAREA[tarea].lower()}: puntos de la rejilla de Grid Search (igual al presupuesto '
                    'de los otros tres métodos), espacio tipado y hiperparámetros del modelo final de la mejor combinación.',
                    f'tab:espacios_{tarea}', tamano='\\tiny')


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    for viejo in TAB.glob('*.tex'):
        viejo.unlink()
    for origen, destino in FIGURAS_LIBRO.items():
        shutil.copy2(RAIZ / 'presentacion' / 'figuras' / origen, FIG / destino)
    print(f'{len(FIGURAS_LIBRO)} figuras del libro copiadas')
    tabla = analisis.cargar_tabla()
    p = datos.preparar_todo()
    cifras = {'eda': material_eda(p)}
    cifras.update(material_resultados(tabla, p))
    material_optimizadores(tabla[tabla['modelo'] != 'ebm'])   # como en el libro: solo las 140 corridas de la guía
    material_computo()
    material_espacios(tabla)
    (AQUI / 'cifras.json').write_text(json.dumps(cifras, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    print('cifras.json escrito')


if __name__ == '__main__':
    main()
