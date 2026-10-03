"""Semilla global, constantes del diseño experimental y localización de los datos.

Toda la aleatoriedad del proyecto se gobierna desde ``SEED``: se propaga a ``random``,
``numpy``, a cada estimador y muestreador de scikit-learn / imbalanced-learn / XGBoost
(``random_state=SEED``), al muestreador TPE de Optuna y al algoritmo genético de DEAP.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- semilla
SEED: int = 42
"""Única semilla del proyecto. Cualquier otra semilla (p. ej. en análisis de sensibilidad)
se deriva de ella o se declara explícitamente donde se use."""

# ------------------------------------------------------------------ diseño experimental
K_EXT: int = 5
"""Pliegues del bucle externo de la validación anidada (estimación de generalización)."""

K_INT: int = 3
"""Pliegues del bucle interno (selección de hiperparámetros dentro de cada pliegue externo)."""

TEST_SIZE: float = 0.20
"""Fracción reservada como conjunto de prueba, intacto hasta la evaluación final."""

PRESUPUESTO_MINIMO: int = 30
"""Piso del presupuesto de evaluaciones por búsqueda. La regla completa vive en
``optimizacion.presupuesto_por_defecto``: cada método evalúa tantas configuraciones como la
rejilla de Grid del mismo modelo (todas las rejillas tienen entre 30 y 48 puntos), de modo
que los cuatro optimizadores consumen lo mismo dentro de cada modelo (guía §3.4)."""

OBJETIVO: str = 'churn_probability'
"""Variable objetivo continua. La etiqueta binaria se deriva de ella solo con entrenamiento."""

ETIQUETA_METODO: str = 'percentil'
ETIQUETA_Q: float = 0.75
"""Definición por defecto de *riesgo alto*: el cuarto superior de ``churn_probability``,
con el umbral estimado solo con entrenamiento. Con la mediana las clases quedan al 50/50
por construcción y la etapa de balanceo no tiene sentido (SMOTE añade 8 filas sobre 38.978
y ADASYN no puede generar ninguna y lanza ``ValueError``)."""

# --------------------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = RAIZ / 'resultados'
CANDIDATAS_DATOS = [
    RAIZ / 'datos',
    RAIZ.parent / 'datos',
    RAIZ.parent / 'entrega 2' / 'datos',
    RAIZ / 'data',
]


def fijar_semillas(seed: int = SEED) -> np.random.Generator:
    """Fija ``random`` y ``numpy`` y devuelve un generador independiente.

    Parameters
    ----------
    seed : int, default SEED
        Semilla a fijar.

    Returns
    -------
    numpy.random.Generator
        Generador independiente sembrado con ``seed``, para muestreos propios
        (bootstrap, submuestras) sin tocar el estado global.

    Notes
    -----
    Los estimadores de scikit-learn, imbalanced-learn y XGBoost no leen el estado
    global de numpy: hay que pasarles ``random_state=seed`` explícitamente. Este
    módulo fija el estado global para el código propio y para DEAP, que sí usa
    ``random``.

    ``PYTHONHASHSEED`` se escribe en ``os.environ`` solo para que la hereden los
    subprocesos (trabajadores de joblib/loky con ``n_jobs=-1``); **no** cambia el
    hash del intérprete ya arrancado. Ningún módulo del proyecto depende del orden de
    ``set``/``dict`` por hash, así que la reproducibilidad no se apoya en esa variable.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return np.random.default_rng(seed)


def localizar_datos(candidatas: list[Path] | None = None) -> Path:
    """Devuelve la primera carpeta candidata que contiene ``customer_data.csv``.

    Parameters
    ----------
    candidatas : list of Path, optional
        Carpetas a probar en orden. Por defecto ``CANDIDATAS_DATOS``.

    Returns
    -------
    Path
        Carpeta con los dos CSV del conjunto COFINFAD.

    Raises
    ------
    FileNotFoundError
        Si ninguna candidata contiene los datos, con instrucciones de descarga.
    """
    for carpeta in candidatas or CANDIDATAS_DATOS:
        if (carpeta / 'customer_data.csv').exists() and (carpeta / 'transactions_data.csv').exists():
            return carpeta
    # En el repositorio publicado los CSV van comprimidos (transactions_data.csv supera el límite de GitHub):
    # se descomprimen en su carpeta la primera vez que se necesitan.
    import zipfile
    for carpeta in candidatas or CANDIDATAS_DATOS:
        zips = [carpeta / f'{nombre}.zip' for nombre in ('customer_data', 'transactions_data')]
        if all(z.exists() for z in zips):
            for z in zips:
                with zipfile.ZipFile(z) as archivo:
                    archivo.extractall(carpeta)
            return carpeta
    raise FileNotFoundError(
        'No se encontraron customer_data.csv y transactions_data.csv. Descargue el conjunto '
        'COFINFAD desde https://data.mendeley.com/datasets/mhb4zn3258/1 (DOI 10.17632/mhb4zn3258.1) '
        f'y ubique ambos archivos en una de estas carpetas: {[str(c) for c in CANDIDATAS_DATOS]}')
