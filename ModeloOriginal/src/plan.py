"""Plan de los experimentos de la corrección: qué combinación corre cada modelo, dónde se registra y cómo se reutiliza
el Entregable 3.

El protocolo (``experiments/protocolo.md``) fija que todos los modelos pasan por la misma validación anidada, con las
mismas particiones y la misma semilla. Los siete modelos de clase y la EBM ya la pasaron en el Entregable 3 con la
semilla 42: esas corridas se importan tal cual (la auditoría de fuga demuestra que los datos son idénticos) y solo se
ejecutan las que faltan.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import datos
from .config import CARPETA_RESULTADOS, RAIZ, SEED
from .registro import Registro

TABLA_E3 = RAIZ.parent / 'Entrega3' / 'resultados' / 'experimentos.parquet'
"""Tabla maestra del Entregable 3 (160 corridas), tal como está publicada en el repositorio."""

PREDICCIONES_E3 = RAIZ.parent / 'Entrega3' / 'resultados' / 'predicciones_prueba.parquet'
"""Predicciones en prueba de las corridas del Entregable 3."""

BENCHMARKS_E3 = {'clasificacion': ['knn', 'naive_bayes', 'logistica', 'arbol', 'random_forest', 'xgboost', 'svm', 'ebm'],
                 'regresion': ['knn', 'ridge', 'lasso', 'arbol', 'random_forest', 'xgboost', 'svr', 'ebm']}
"""Los siete modelos de clase de cada tarea más la EBM (el modelo nuevo del Entregable 3, ahora benchmark reciente)."""

NUEVOS = {'lightgbm': {'clasificacion': ('class_weight', 'bayesiana'), 'regresion': ('ninguno', 'bayesiana')},
          'bdl': {'clasificacion': ('ninguno', 'bayesiana'), 'regresion': ('ninguno', 'bayesiana')}}
"""Combinación (balanceo, optimizador) de los dos modelos que no estaban en el Entregable 3, fijada de antemano en el
protocolo: optimización bayesiana, la que recomienda la guía; LightGBM pondera las clases, como XGBoost y la EBM; el
modelo original no remuestrea ni pondera, porque aprende del puntaje continuo."""

ORIGINAL = 'bdl'
NOMBRES = {'knn': 'k-NN', 'naive_bayes': 'Naive Bayes', 'logistica': 'Regresión logística', 'arbol': 'Árbol de decisión',
           'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'svm': 'SVM lineal', 'ebm': 'EBM', 'lightgbm': 'LightGBM',
           'bdl': 'BDL (original)', 'ridge': 'Ridge', 'lasso': 'Lasso', 'svr': 'SVR lineal'}


def modelos(tarea: str) -> list[str]:
    """Los diez modelos de la tarea: ocho del Entregable 3, LightGBM y el original."""
    return BENCHMARKS_E3[tarea] + ['lightgbm', ORIGINAL]


@lru_cache(maxsize=1)
def tabla_e3() -> pd.DataFrame:
    """Tabla maestra del Entregable 3 con ``media_ext`` numérica."""
    t = pd.read_parquet(TABLA_E3)
    t['media_ext'] = pd.to_numeric(t['media_ext'], errors='coerce')
    return t


@lru_cache(maxsize=None)
def combinacion(tarea: str, modelo: str) -> tuple[str, str]:
    """Combinación (balanceo, optimizador) con la que corre un modelo.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    modelo : str

    Returns
    -------
    (balanceo, optimizador)
        Para los modelos del Entregable 3, la de mejor métrica media del bucle externo (la regla de selección del
        dashboard, ``analisis.mejores_por_modelo``); para LightGBM y el original, la de ``NUEVOS``.
    """
    if modelo in NUEVOS:
        return NUEVOS[modelo][tarea]
    t = tabla_e3()
    t = t[(t['tarea'] == tarea) & (t['modelo'] == modelo) & (t['estado'] == 'ok')]
    fila = t.loc[t['media_ext'].idxmax()]
    return str(fila['balanceo']), str(fila['optimizador'])


def registro(conjunto: str = 'cofinfad') -> Registro:
    """Tabla maestra de la corrección para un conjunto de datos."""
    carpeta = CARPETA_RESULTADOS if conjunto == 'cofinfad' else CARPETA_RESULTADOS / 'adicionales' / conjunto
    return Registro(carpeta / 'experimentos.parquet')


def importar_entregable3(reg: Registro) -> int:
    """Copia a la tabla de la corrección la mejor combinación de cada modelo del Entregable 3 (semilla 42).

    Parameters
    ----------
    reg : Registro

    Returns
    -------
    int
        Filas añadidas (0 si ya estaban).

    Notes
    -----
    Cada fila se copia sin tocar sus métricas y se le añade ``origen = 'Entregable 3'``. Es válido reutilizarlas porque
    la corrección de la recencia no cambia ni una celda de los predictores (``experiments/01_auditoria_fuga.py``).
    """
    t = tabla_e3()
    anadidas = 0
    for tarea, lista in BENCHMARKS_E3.items():
        for modelo in lista:
            balanceo, optimizador = combinacion(tarea, modelo)
            if reg.existe(tarea=tarea, modelo=modelo, balanceo=balanceo, optimizador=optimizador, semilla=SEED):
                continue
            fila = t[(t['tarea'] == tarea) & (t['modelo'] == modelo) & (t['balanceo'] == balanceo)
                     & (t['optimizador'] == optimizador) & (t['semilla'] == SEED)].iloc[0].to_dict()
            for k, v in list(fila.items()):                         # las columnas JSON vuelven a ser objetos
                if isinstance(v, str) and v[:1] in '[{':
                    try:
                        fila[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(v, float) and np.isnan(v):
                    fila[k] = None
                elif isinstance(v, np.generic):
                    fila[k] = v.item()
            fila['origen'] = 'Entregable 3'
            fila.pop('registrado_en', None)
            reg.anadir(fila)
            anadidas += 1
    return anadidas


@lru_cache(maxsize=1)
def problema() -> dict:
    """Datos del problema principal ya preparados (con la recencia corregida)."""
    return datos.preparar_todo()


def objetivo(tarea: str, p: dict | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Predictores y objetivo de entrenamiento y de prueba de una tarea.

    Returns
    -------
    X_train, y_train, X_test, y_test
        ``y`` es la etiqueta binaria en clasificación y el puntaje continuo en regresión.
    """
    p = p or problema()
    c = p['conjuntos']
    if tarea == 'clasificacion':
        return c.X_train, p['y_train'], c.X_test, p['y_test']
    return c.X_train, c.y_train_cont, c.X_test, c.y_test_cont


def params_ajuste(tarea: str, modelo: str, puntaje_train) -> dict | None:
    """Parámetros de ajuste por fila: solo el modelo original, y solo en clasificación, recibe el puntaje continuo."""
    if modelo == ORIGINAL and tarea == 'clasificacion':
        return {'modelo__puntaje': np.asarray(puntaje_train, dtype=float)}
    return None


def puntajes_por_pliegue(fila) -> np.ndarray:
    """Métrica principal de cada pliegue externo de una fila de la tabla, en sentido natural (AUC o RMSE)."""
    v = fila['puntajes_ext']
    v = json.loads(v) if isinstance(v, str) else v
    return np.abs(np.asarray(v, dtype=float))


def metricas_por_pliegue(fila, metrica: str) -> np.ndarray:
    """Una métrica secundaria de cada pliegue externo de una fila de la tabla."""
    v = fila['metricas_ext']
    v = json.loads(v) if isinstance(v, str) else v
    return np.asarray(v[metrica], dtype=float)
