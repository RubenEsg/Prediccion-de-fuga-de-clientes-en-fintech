"""Gráficas del libro con un estilo común.

Cada función dibuja sobre un ``Axes`` (o crea la figura si no se le da uno) y lo devuelve, de
modo que el cuaderno decide la composición. Colores fijos por optimizador y por balanceo para
que se reconozcan de una gráfica a otra.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

AZUL = '#1f3a5f'
PALETA = ['#1f3a5f', '#3d78bf', '#7fb0e0', '#a8cbec', '#d9822b', '#8c8c8c']
COLOR_OPTIMIZADOR = {'grid': '#8c8c8c', 'random': '#7fb0e0', 'bayesiana': '#1f3a5f', 'genetica': '#d9822b'}
COLOR_BALANCEO = {'ninguno': '#1f3a5f', 'smote': '#3d78bf', 'adasyn': '#7fb0e0', 'class_weight': '#d9822b'}
NOMBRE_OPTIMIZADOR = {'grid': 'Grid', 'random': 'Random', 'bayesiana': 'Bayesiana (TPE)', 'genetica': 'Genética'}
NOMBRE_CORTO_OPTIMIZADOR = {'grid': 'Grid', 'random': 'Random', 'bayesiana': 'bayesiana', 'genetica': 'genética'}
NOMBRE_BALANCEO = {'ninguno': 'Sin balanceo', 'smote': 'SMOTE', 'adasyn': 'ADASYN', 'class_weight': 'class_weight'}
NOMBRE_TAREA = {'clasificacion': 'Clasificación', 'regresion': 'Regresión'}


def estilo() -> None:
    """Tema común: fondo con rejilla suave, títulos en negrita y la paleta del proyecto."""
    sns.set_theme(style='whitegrid')
    sns.set_palette(PALETA)
    plt.rcParams.update({'axes.titleweight': 'bold', 'figure.dpi': 110, 'savefig.dpi': 150,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 11, 'axes.labelsize': 10})


def _ejes(ax, tamano=(7, 4.2)):
    if ax is None:
        _, ax = plt.subplots(figsize=tamano)
    return ax


# ------------------------------------------------------------------ corridas (§2, §3)
def mapa_corridas(tabla: pd.DataFrame, tarea: str, ax=None, orden_modelos: list[str] | None = None):
    """Mapa de calor de la métrica externa por modelo y combinación balanceo × optimizador.

    Parameters
    ----------
    tabla : DataFrame
        Salida de ``analisis.cargar_tabla``.
    tarea : {'clasificacion', 'regresion'}
    ax : Axes, optional
    orden_modelos : list of str, optional
        Orden de las filas; por defecto de mejor a peor.
    """
    from .analisis import NOMBRES_MODELO, OPTIMIZADORES
    t = tabla[tabla['tarea'] == tarea].copy()
    mayor_es_mejor = tarea == 'clasificacion'
    columnas = ['balanceo', 'optimizador'] if tarea == 'clasificacion' else ['optimizador']
    piv = t.pivot_table(index='modelo', columns=columnas, values='metrica', aggfunc='first', dropna=False)
    if tarea == 'clasificacion':
        piv = piv.reindex(columns=pd.MultiIndex.from_product([list(COLOR_BALANCEO), OPTIMIZADORES]))
        etiquetas = [f'{NOMBRE_BALANCEO[b]}\n{NOMBRE_OPTIMIZADOR[o]}' for b, o in piv.columns]
    else:
        piv = piv.reindex(columns=OPTIMIZADORES)
        etiquetas = [NOMBRE_OPTIMIZADOR[o] for o in piv.columns]
    orden = orden_modelos or piv.max(axis=1).sort_values(ascending=not mayor_es_mejor).index.tolist()
    piv = piv.loc[orden]
    ax = _ejes(ax, (14, 0.5 * len(piv) + 2) if tarea == 'clasificacion' else (7, 0.5 * len(piv) + 1.5))
    sns.heatmap(piv, annot=True, fmt='.4f', cmap='Blues' if mayor_es_mejor else 'Blues_r', cbar=False,
                linewidths=0.5, linecolor='white', annot_kws={'size': 8}, ax=ax)
    ax.set_xticklabels(etiquetas, rotation=0, fontsize=7.5)
    ax.set_yticklabels([NOMBRES_MODELO.get(m, m) for m in piv.index], rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'{"AUC" if mayor_es_mejor else "RMSE"} media del bucle externo (5 pliegues)')
    return ax


def curvas_anytime(curvas: pd.DataFrame, ax=None, titulo: str | None = None):
    """Arrepentimiento normalizado medio (con banda intercuartílica) frente al presupuesto usado."""
    ax = _ejes(ax)
    for o, g in curvas.groupby('optimizador'):
        ax.plot(g['fraccion'] * 100, g['media'], color=COLOR_OPTIMIZADOR[o], lw=2, label=NOMBRE_OPTIMIZADOR[o])
        ax.fill_between(g['fraccion'] * 100, g['q25'], g['q75'], color=COLOR_OPTIMIZADOR[o], alpha=0.12)
    ax.set_xlabel('Presupuesto consumido (%)')
    ax.set_ylabel('Arrepentimiento normalizado')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    if titulo:
        ax.set_title(titulo)
    return ax


def diversidad_genetica(div: pd.DataFrame, ax=None):
    """Diversidad media de la población por generación, una línea por tarea."""
    ax = _ejes(ax)
    for (tarea, color) in (('clasificacion', AZUL), ('regresion', '#d9822b')):
        g = div[div['tarea'] == tarea].groupby('generacion')['diversidad']
        media, q25, q75 = g.mean(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(media.index, media.values, marker='o', color=color, label=NOMBRE_TAREA[tarea])
        ax.fill_between(media.index, q25.values, q75.values, color=color, alpha=0.12)
    ax.set_xlabel('Generación (0 = población inicial)')
    ax.set_ylabel('Desviación típica media de los genes')
    ax.set_title('Diversidad de la población en el algoritmo genético')
    ax.legend(frameon=False)
    return ax


def diagrama_cd(resultado_friedman: dict, ax=None, titulo: str | None = None, etiquetas: dict | None = None):
    """Diagrama de diferencia crítica (Demšar, 2006) a partir de :func:`estadistica.friedman`."""
    import scikit_posthocs as sp
    ax = _ejes(ax, (8, 2.6))
    rangos = resultado_friedman['rangos_medios']
    p = resultado_friedman['p_nemenyi']
    if etiquetas:
        rangos = rangos.rename(index=etiquetas)
        p = p.rename(index=etiquetas, columns=etiquetas)
    sp.critical_difference_diagram(rangos, p, ax=ax, label_fmt_left='{label} ({rank:.2f})',
                                   label_fmt_right='({rank:.2f}) {label}')
    ax.set_title(titulo or f'Diferencia crítica de Nemenyi = {resultado_friedman["dc"]:.2f} rangos', pad=12)
    return ax


# ------------------------------------------------------------------ evaluación (§5)
def curvas_roc(y, puntajes: dict[str, np.ndarray], ax=None, titulo: str = 'Curvas ROC en prueba'):
    """Curva ROC de cada modelo con su AUC en la leyenda."""
    from sklearn.metrics import roc_auc_score, roc_curve
    ax = _ejes(ax, (6, 5.5))
    colores = sns.color_palette('tab10', len(puntajes))
    for (nombre, s), c in zip(puntajes.items(), colores):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=c, lw=1.6, label=f'{nombre} (AUC {roc_auc_score(y, s):.4f})')
    ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=0.8)
    ax.set_xlabel('Tasa de falsos positivos')
    ax.set_ylabel('Tasa de verdaderos positivos (recall)')
    ax.set_title(titulo)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    return ax


def matriz_confusion(y, clase, ax=None, titulo: str = ''):
    """Matriz de confusión con conteos y porcentaje por fila real."""
    from sklearn.metrics import confusion_matrix
    ax = _ejes(ax, (3.4, 3))
    m = confusion_matrix(y, clase, labels=[0, 1])
    filas = m / m.sum(axis=1, keepdims=True)
    anot = np.array([[f'{m[i, j]:,}\n({filas[i, j]:.0%})'.replace(',', '.') for j in range(2)] for i in range(2)])
    sns.heatmap(filas, annot=anot, fmt='', cmap='Blues', vmin=0, vmax=1, cbar=False, ax=ax,
                xticklabels=['Bajo', 'Alto'], yticklabels=['Bajo', 'Alto'], annot_kws={'size': 9})
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    ax.set_title(titulo, fontsize=9)
    return ax


def diagrama_confiabilidad(y, probas: dict[str, np.ndarray], n_bins: int = 10, ax=None,
                           titulo: str = 'Diagrama de confiabilidad'):
    """Frecuencia observada frente a probabilidad media predicha por cuantiles de probabilidad."""
    from sklearn.calibration import calibration_curve
    ax = _ejes(ax, (6, 5.5))
    colores = sns.color_palette('tab10', len(probas))
    for (nombre, p), c in zip(probas.items(), colores):
        obs, pred = calibration_curve(y, p, n_bins=n_bins, strategy='quantile')
        ax.plot(pred, obs, marker='o', ms=3.5, color=c, lw=1.4, label=nombre)
    ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=0.8, label='Calibración perfecta')
    ax.set_xlabel('Probabilidad media predicha')
    ax.set_ylabel('Frecuencia observada de riesgo alto')
    ax.set_title(titulo)
    ax.legend(frameon=False, fontsize=8)
    return ax


def panel_residuos(y, pred, titulo: str = '', retardos: int = 40):
    """Cuatro paneles de diagnóstico: residuo frente a ajuste, histograma, Q-Q normal y ACF."""
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf
    r = np.asarray(y) - np.asarray(pred)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    ax[0].scatter(pred, r, s=3, alpha=0.25, color=AZUL)
    ax[0].axhline(0, color='grey', lw=0.8)
    ax[0].set_xlabel('Pronóstico')
    ax[0].set_ylabel('Residuo')
    ax[0].set_title('Residuo frente a pronóstico')
    ax[1].hist(r, bins=60, density=True, color='#7fb0e0', edgecolor='white')
    x = np.linspace(r.min(), r.max(), 200)
    ax[1].plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=AZUL, lw=1.5, label='Normal')
    ax[1].set_title('Histograma de residuos')
    ax[1].legend(frameon=False)
    stats.probplot(r, dist='norm', plot=ax[2])
    ax[2].get_lines()[0].set(markersize=2, alpha=0.4, color=AZUL)
    ax[2].set_title('Q-Q normal')
    plot_acf(r, lags=retardos, ax=ax[3], color=AZUL, vlines_kwargs={'colors': AZUL})
    ax[3].set_title('ACF (orden original de las filas)')
    if titulo:
        fig.suptitle(titulo, fontweight='bold', y=1.03)
    fig.tight_layout()
    return fig
