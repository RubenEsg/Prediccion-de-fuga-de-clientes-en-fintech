"""Métricas de clasificación y regresión sobre un pipeline ajustado."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)

METRICA_PRINCIPAL: dict[str, str] = {'clasificacion': 'roc_auc',
                                     'regresion': 'neg_root_mean_squared_error'}
"""Criterio de selección de hiperparámetros por tarea (nombre de *scorer* de scikit-learn).

Clasificación: AUC, independiente del umbral y de la prevalencia, adecuado para ordenar
clientes por riesgo. Regresión: RMSE, la métrica de §5.2.1 que penaliza más los errores
grandes. La justificación extendida y la métrica secundaria de negocio se discuten en la
sección de evaluación del libro."""

NOMBRE_METRICA: dict[str, str] = {'clasificacion': 'auc', 'regresion': 'rmse'}
"""Clave de la métrica principal dentro del diccionario que devuelve :func:`evaluar`."""


def puntajes(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Predicción de clase y puntaje continuo de un clasificador.

    Parameters
    ----------
    pipeline : Pipeline
        Clasificador ajustado.
    X : DataFrame
        Predictores.

    Returns
    -------
    pred : ndarray
        Clase predicha.
    puntaje : ndarray or None
        Probabilidad de la clase positiva si el modelo la ofrece; si no, la función de
        decisión (suficiente para el AUC pero no para el Brier); ``None`` si no hay ninguna.
    es_probabilidad : bool
        ``True`` cuando ``puntaje`` es una probabilidad.
    """
    pred = pipeline.predict(X)
    if hasattr(pipeline, 'predict_proba'):
        return pred, pipeline.predict_proba(X)[:, 1], True
    if hasattr(pipeline, 'decision_function'):
        return pred, pipeline.decision_function(X), False
    return pred, None, False


def metricas_clasificacion(y: np.ndarray, pred: np.ndarray, puntaje: np.ndarray | None,
                           es_probabilidad: bool) -> dict[str, float]:
    """Métricas de clasificación binaria.

    Parameters
    ----------
    y : ndarray
        Etiqueta real (0/1).
    pred : ndarray
        Clase predicha.
    puntaje : ndarray or None
        Probabilidad o función de decisión (ver :func:`puntajes`).
    es_probabilidad : bool

    Returns
    -------
    dict
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``auc`` (NaN sin puntaje) y
        ``brier`` (NaN si el puntaje no es una probabilidad).
    """
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'auc': float(roc_auc_score(y, puntaje)) if puntaje is not None else float('nan'),
        'brier': float(brier_score_loss(y, puntaje)) if (puntaje is not None and es_probabilidad) else float('nan'),
    }


def ece(y, p, n_bins: int = 10, estrategia: str = 'uniforme') -> float:
    """Error de calibración esperado (ECE).

    Parameters
    ----------
    y : array-like
        Etiqueta 0/1.
    p : array-like
        Probabilidad predicha de la clase positiva.
    n_bins : int
    estrategia : {'uniforme', 'cuantil'}
        Intervalos de igual ancho en [0, 1] (la definición habitual, Naeini y otros, 2015) o de
        igual frecuencia.

    Returns
    -------
    float
        ``Σ_b (n_b / n) · |frecuencia observada_b − probabilidad media_b|``: la distancia media,
        ponderada por el número de casos, entre lo que el modelo anuncia y lo que ocurre.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if estrategia == 'uniforme':
        bordes = np.linspace(0.0, 1.0, n_bins + 1)
    elif estrategia == 'cuantil':
        bordes = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError(f'estrategia desconocida: {estrategia!r}')
    idx = np.clip(np.searchsorted(bordes, p, side='right') - 1, 0, len(bordes) - 2)
    total = 0.0
    for b in range(len(bordes) - 1):
        en_bin = idx == b
        if en_bin.any():
            total += en_bin.mean() * abs(y[en_bin].mean() - p[en_bin].mean())
    return float(total)


def diagnostico_residuos(y, pred, n_bds: int = 3000, retardos: tuple[int, ...] = (10, 20)) -> pd.DataFrame:
    """Pruebas sobre los residuos de una regresión (guía §5.2).

    Parameters
    ----------
    y, pred : array-like
        Valor real y pronóstico, **en el orden en que se quiere evaluar la dependencia** (aquí,
        el orden original del archivo de clientes).
    n_bds : int
        Residuos consecutivos que usa la prueba BDS: su coste en memoria es cuadrático en n.
    retardos : tuple of int
        Retardos de la prueba de Ljung-Box.

    Returns
    -------
    DataFrame
        Una fila por prueba con ``estadistico``, ``p`` y la hipótesis nula: White
        (homocedasticidad, con el pronóstico y su cuadrado como regresores), BDS para dimensiones
        2 y 3 (independencia e idéntica distribución), Ljung-Box (ausencia de autocorrelación),
        Jarque-Bera y D'Agostino-Pearson (normalidad), y la asimetría y la curtosis en exceso.
    """
    import statsmodels.api as sm
    from scipy import stats as st
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
    from statsmodels.tsa.stattools import bds

    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    r = y - pred
    filas = []
    lm, p_lm, _, _ = het_white(r, sm.add_constant(pred))
    filas.append(('White', 'varianza constante de los residuos', lm, p_lm))
    z, p_z = bds(r[:n_bds], max_dim=3)
    for dim, zi, pi in zip((2, 3), np.atleast_1d(z), np.atleast_1d(p_z)):
        filas.append((f'BDS, dimensión {dim} ({min(n_bds, len(r))} residuos)', 'residuos i.i.d.', zi, pi))
    lb = acorr_ljungbox(r, lags=list(retardos))
    for k in retardos:
        filas.append((f'Ljung-Box, {k} retardos', 'sin autocorrelación', lb.loc[k, 'lb_stat'], lb.loc[k, 'lb_pvalue']))
    jb = st.jarque_bera(r)
    filas.append(('Jarque-Bera', 'normalidad', jb.statistic, jb.pvalue))
    k2 = st.normaltest(r)
    filas.append(("D'Agostino-Pearson", 'normalidad', k2.statistic, k2.pvalue))
    filas.append(('asimetría', '0 si es simétrica', float(st.skew(r)), np.nan))
    filas.append(('curtosis en exceso', '0 si es normal', float(st.kurtosis(r)), np.nan))
    return pd.DataFrame(filas, columns=['prueba', 'hipótesis nula', 'estadistico', 'p']).set_index('prueba')


def metricas_regresion(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Métricas de regresión.

    Parameters
    ----------
    y, pred : ndarray
        Valor real y predicho.

    Returns
    -------
    dict
        ``rmse``, ``mae`` y ``r2``.
    """
    return {
        'rmse': float(np.sqrt(mean_squared_error(y, pred))),
        'mae': float(mean_absolute_error(y, pred)),
        'r2': float(r2_score(y, pred)),
    }


def evaluar(pipeline, X: pd.DataFrame, y, tarea: str) -> dict[str, float]:
    """Evalúa un pipeline ajustado sobre ``(X, y)`` con las métricas de su tarea.

    Parameters
    ----------
    pipeline : Pipeline
        Ajustado.
    X : DataFrame
    y : array-like
        Etiqueta binaria (clasificación) o valor continuo (regresión).
    tarea : {'clasificacion', 'regresion'}

    Returns
    -------
    dict
        Ver :func:`metricas_clasificacion` y :func:`metricas_regresion`.

    Raises
    ------
    ValueError
        Si la tarea es desconocida.
    """
    y = np.asarray(y)
    if tarea == 'clasificacion':
        pred, puntaje, es_prob = puntajes(pipeline, X)
        return metricas_clasificacion(y, pred, puntaje, es_prob)
    if tarea == 'regresion':
        return metricas_regresion(y, pipeline.predict(X))
    raise ValueError(f'tarea desconocida: {tarea!r}')


def principal(metricas: dict[str, float], tarea: str) -> float:
    """Métrica principal en el sentido *mayor es mejor*.

    Parameters
    ----------
    metricas : dict
        Salida de :func:`evaluar`.
    tarea : {'clasificacion', 'regresion'}

    Returns
    -------
    float
        ``auc`` en clasificación; ``-rmse`` en regresión, para que el signo sea uniforme.
    """
    valor = metricas[NOMBRE_METRICA[tarea]]
    return -valor if tarea == 'regresion' else valor
