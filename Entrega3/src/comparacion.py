"""Comparación con la Entrega 2 sobre la misma partición y la misma prueba.

El profesor pidió (29-sep-2026) un modelo nuevo que supere las métricas de la Entrega 2,
demostrado sobre la misma partición 80/20 con semilla 42 y la misma etiqueta —la mediana de
``churn_probability`` estimada solo con entrenamiento— y confirmado con DeLong y
Diebold-Mariano. Este módulo reúne:

- los dos modelos de la Entrega 2 tal como se entregaron (:func:`pipeline_entrega2`) y las
  cifras publicadas que deben reproducir (:data:`PUBLICADO_ENTREGA2`);
- las tablas de intervalos por modelo (:func:`intervalos_clasificacion`,
  :func:`intervalos_regresion`) y de contrastes por pares (:func:`contrastar_clasificacion`,
  :func:`contrastar_regresion`), que usan el guion ``comparar_entrega2.py`` y el libro.

En cada par ``(referencia, candidato)`` las diferencias están orientadas para que un valor
positivo favorezca al candidato (ver :mod:`estadistica`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline

from . import estadistica as est
from .config import SEED
from .preprocesamiento import construir_preprocesador

PUBLICADO_ENTREGA2: dict[str, dict[str, float]] = {
    'clasificacion': {'auc': 0.6832, 'recall': 0.6515},
    'regresion': {'r2': 0.1511, 'rmse': 0.0617},
}
"""Métricas de prueba de la Entrega 2 (cuaderno corregido y ejecutado del 12-sep-2026), con los
cuatro decimales con que se publicaron."""


def pipeline_entrega2(tarea: str, X: pd.DataFrame, seed: int = SEED) -> Pipeline:
    """Modelo de la Entrega 2 con los hiperparámetros que eligió su ``GridSearchCV``.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    X : DataFrame
        Predictores (solo se usan tipos y nombres de columna).
    seed : int

    Returns
    -------
    Pipeline
        Sin ajustar: escalado + indicadoras y, en clasificación, regresión logística L1 con
        C = 0,1 (``penalty='l1'`` en la Entrega 2, que scikit-learn ≥ 1.8 escribe
        ``l1_ratio=1``); en regresión, Lasso con alpha = 1e-4.

    Raises
    ------
    ValueError
        Si la tarea es desconocida.
    """
    if tarea == 'clasificacion':
        modelo = LogisticRegression(l1_ratio=1.0, C=0.1, solver='liblinear', max_iter=2000,
                                    random_state=seed)
    elif tarea == 'regresion':
        modelo = Lasso(alpha=1e-4, max_iter=20000, random_state=seed)
    else:
        raise ValueError(f'tarea desconocida: {tarea!r}')
    return Pipeline([('preprocesado', construir_preprocesador(X)), ('modelo', modelo)])


def reproduce_entrega2(tarea: str, metricas: dict[str, float]) -> dict[str, dict]:
    """Compara las métricas obtenidas con las publicadas, a cuatro decimales.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    metricas : dict
        Salida de ``evaluacion.evaluar`` sobre la prueba.

    Returns
    -------
    dict
        ``metrica -> {'publicado', 'obtenido', 'coincide'}``.
    """
    return {k: {'publicado': v, 'obtenido': round(float(metricas[k]), 4),
                'coincide': round(float(metricas[k]), 4) == v}
            for k, v in PUBLICADO_ENTREGA2[tarea].items()}


# ---------------------------------------------------------------------- clasificación
def intervalos_clasificacion(y, puntajes: dict[str, np.ndarray], n_boot: int = est.N_BOOT,
                             nivel: float = est.NIVEL, seed: int = SEED) -> pd.DataFrame:
    """AUC de prueba de cada modelo con su intervalo de DeLong y su intervalo bootstrap BCa.

    Parameters
    ----------
    y : array-like
        Etiqueta 0/1 de prueba.
    puntajes : dict
        ``nombre -> puntaje`` sobre las mismas filas.
    n_boot, nivel, seed

    Returns
    -------
    DataFrame
        Una fila por modelo: ``auc``, ``ee_delong``, ``ic_delong_inf/sup``, ``ic_bca_inf/sup``.
    """
    filas = []
    for nombre, s in puntajes.items():
        d = est.delong(y, s, nivel=nivel)
        b = est.bootstrap_auc(y, s, n_boot=n_boot, nivel=nivel, seed=seed)
        filas.append({'modelo': nombre, 'auc': d['auc'], 'ee_delong': d['ee'],
                      'ic_delong_inf': d['ic_inf'], 'ic_delong_sup': d['ic_sup'],
                      'ic_bca_inf': b['ic_bca'][0], 'ic_bca_sup': b['ic_bca'][1]})
    return pd.DataFrame(filas).set_index('modelo')


def contrastar_clasificacion(y, puntajes: dict[str, np.ndarray], pares: list[tuple[str, str]],
                             auc_pliegues: dict[str, list[float]] | None = None,
                             n_boot: int = est.N_BOOT, nivel: float = est.NIVEL, seed: int = SEED,
                             correccion: str = 'holm') -> pd.DataFrame:
    """DeLong, bootstrap BCa de la diferencia y delta de Cliff para cada par de modelos.

    Parameters
    ----------
    y : array-like
        Etiqueta 0/1 de prueba.
    puntajes : dict
        ``nombre -> puntaje`` sobre las mismas filas.
    pares : list of (referencia, candidato)
    auc_pliegues : dict, optional
        ``nombre -> AUC en cada pliegue`` de la validación cruzada sobre entrenamiento (los
        mismos pliegues para todos). Si se da, se añade el delta de Cliff entre pliegues.
    n_boot, nivel, seed
    correccion : str
        Método de ajuste de los p-valores de DeLong por comparaciones múltiples.

    Returns
    -------
    DataFrame
        Una fila por par: AUC de cada modelo, ``diferencia`` (candidato − referencia), su error
        de DeLong, ``z``, ``p_delong`` (bilateral), ``p_delong_unilateral`` (H1: el candidato es
        mejor), ``p_delong_ajustado``, los intervalos de DeLong y BCa de la diferencia,
        ``prop_bootstrap_no_mejora`` y, si hay pliegues, ``delta_cliff`` y su magnitud.
    """
    filas = []
    for ref, cand in pares:
        d = est.delong(y, puntajes[ref], puntajes[cand], nivel=nivel)
        b = est.bootstrap_auc(y, puntajes[ref], puntajes[cand], n_boot=n_boot, nivel=nivel, seed=seed)
        fila = {'referencia': ref, 'candidato': cand, 'auc_referencia': d['auc_a'],
                'auc_candidato': d['auc_b'], 'diferencia': d['diferencia'], 'ee_delong': d['ee'],
                'z': d['z'], 'p_delong': d['p_bilateral'], 'p_delong_unilateral': d['p_unilateral'],
                'ic_delong_inf': d['ic_inf'], 'ic_delong_sup': d['ic_sup'],
                'correlacion_auc': d['correlacion'], 'ic_bca_inf': b['ic_bca'][0],
                'ic_bca_sup': b['ic_bca'][1], 'prop_bootstrap_no_mejora': b['prop_no_mejora']}
        if auc_pliegues:
            c = est.delta_cliff(auc_pliegues[cand], auc_pliegues[ref])
            fila.update({'delta_cliff': c['delta'], 'magnitud_cliff': c['magnitud']})
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.insert(tabla.columns.get_loc('p_delong_unilateral') + 1, 'p_delong_ajustado',
                 est.ajustar_pvalores(tabla['p_delong'], correccion))
    return tabla


# -------------------------------------------------------------------------- regresión
def intervalos_regresion(y, pronosticos: dict[str, np.ndarray], n_boot: int = est.N_BOOT,
                         nivel: float = est.NIVEL, seed: int = SEED) -> pd.DataFrame:
    """RMSE, MAE y R² de prueba de cada modelo con sus intervalos bootstrap BCa.

    Parameters
    ----------
    y : array-like
        Objetivo continuo de prueba.
    pronosticos : dict
        ``nombre -> pronóstico`` sobre las mismas filas.
    n_boot, nivel, seed

    Returns
    -------
    DataFrame
        Una fila por modelo: ``rmse``, ``mae``, ``r2`` y ``<métrica>_ic_inf/sup``.
    """
    filas = []
    for nombre, p in pronosticos.items():
        b = est.bootstrap_regresion(y, p, n_boot=n_boot, nivel=nivel, seed=seed)
        fila = {'modelo': nombre}
        for k in ('rmse', 'mae', 'r2'):
            fila.update({k: b[k]['estimacion'], f'{k}_ic_inf': b[k]['ic_bca'][0],
                         f'{k}_ic_sup': b[k]['ic_bca'][1]})
        filas.append(fila)
    return pd.DataFrame(filas).set_index('modelo')


def contrastar_regresion(y, pronosticos: dict[str, np.ndarray], pares: list[tuple[str, str]],
                         rmse_pliegues: dict[str, list[float]] | None = None,
                         n_boot: int = est.N_BOOT, nivel: float = est.NIVEL, seed: int = SEED,
                         correccion: str = 'holm') -> pd.DataFrame:
    """Diebold-Mariano (HLN), bootstrap estacionario y tamaños del efecto para cada par.

    Parameters
    ----------
    y : array-like
        Objetivo continuo de prueba, **en el orden original de las filas** (el bootstrap
        estacionario y la prueba de Ljung-Box dependen del orden).
    pronosticos : dict
        ``nombre -> pronóstico`` sobre las mismas filas.
    pares : list of (referencia, candidato)
    rmse_pliegues : dict, optional
        ``nombre -> RMSE en cada pliegue`` de la validación cruzada sobre entrenamiento. Si se
        da, se añaden la d de Cohen y el delta de Cliff entre pliegues.
    n_boot, nivel, seed
    correccion : str
        Método de ajuste de los p-valores de Diebold-Mariano.

    Returns
    -------
    DataFrame
        Una fila por par: mejora de RMSE, MAE y R² con su intervalo BCa; ``dm_hln`` y sus
        p-valores (pérdida cuadrática, bilateral, unilateral y ajustado); ``p_dm_absoluta``
        (la misma prueba con pérdida absoluta); ``bloque`` y ``p_bootstrap_estacionario``;
        ``p_ljung_box`` del diferencial (10 retardos); ``d_cohen_z`` del diferencial por fila y,
        con pliegues, ``d_cohen_pliegues`` y ``delta_cliff_pliegues``.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    y = np.asarray(y, dtype=float)
    filas = []
    for ref, cand in pares:
        pa, pb = np.asarray(pronosticos[ref], float), np.asarray(pronosticos[cand], float)
        ea, eb = (y - pa) ** 2, (y - pb) ** 2
        dm = est.diebold_mariano(ea, eb, h=1, hln=True)
        dm_abs = est.diebold_mariano(np.abs(y - pa), np.abs(y - pb), h=1, hln=True)
        be = est.bootstrap_estacionario(ea - eb, n_boot=n_boot, nivel=nivel, seed=seed)
        br = est.bootstrap_regresion(y, pa, pb, n_boot=n_boot, nivel=nivel, seed=seed)
        dz = est.d_cohen(ea - eb)
        fila = {'referencia': ref, 'candidato': cand}
        for k in ('rmse', 'mae', 'r2'):
            fila.update({f'mejora_{k}': br[k]['estimacion'], f'mejora_{k}_ic_inf': br[k]['ic_bca'][0],
                         f'mejora_{k}_ic_sup': br[k]['ic_bca'][1]})
        fila.update({'dm_hln': dm['estadistico'], 'p_dm': dm['p_bilateral'],
                     'p_dm_unilateral': dm['p_unilateral'], 'p_dm_absoluta': dm_abs['p_bilateral'],
                     'bloque': be['bloque'], 'p_bootstrap_estacionario': be['p_bilateral'],
                     'p_ljung_box': float(acorr_ljungbox(ea - eb, lags=[10])['lb_pvalue'].iloc[0]),
                     'd_cohen_z': dz['d'], 'magnitud_d_z': dz['magnitud']})
        if rmse_pliegues:
            dc = est.d_cohen(rmse_pliegues[ref], rmse_pliegues[cand])
            cl = est.delta_cliff(rmse_pliegues[ref], rmse_pliegues[cand])
            fila.update({'d_cohen_pliegues': dc['d'], 'magnitud_d_pliegues': dc['magnitud'],
                         'delta_cliff_pliegues': cl['delta']})
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.insert(tabla.columns.get_loc('p_dm_unilateral') + 1, 'p_dm_ajustado',
                 est.ajustar_pvalores(tabla['p_dm'], correccion))
    return tabla
