"""Técnicas de balanceo y ensamblado del pipeline completo.

El pipeline siempre es ``preprocesado → [muestreo] → modelo`` construido con
``imblearn.pipeline.Pipeline``, de modo que SMOTE/ADASYN solo remuestrean las filas de
entrenamiento de cada pliegue y nunca las de validación o prueba. En regresión la etapa
de balanceo se omite (guía §2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline

from .config import SEED
from .modelos import EspecificacionModelo
from .preprocesamiento import construir_preprocesador

CATALOGO_BALANCEO: list[str] = ['ninguno', 'smote', 'adasyn', 'class_weight']
"""Las cuatro técnicas de la Etapa 2 (guía §2 y §5.1.2), en el orden de la guía."""

PREVALENCIA_MAXIMA_SOBREMUESTREO: float = 0.40
"""Si la clase minoritaria supera esta fracción, SMOTE/ADASYN no tienen nada que
generar: SMOTE añade un puñado de filas y ADASYN lanza ``ValueError`` (comprobado con
imbalanced-learn 0.14 y clases al 49,99 %). La combinación se registra como *no aplica*."""


def aplica(spec: EspecificacionModelo, balanceo: str,
           y: pd.Series | np.ndarray | None = None) -> tuple[bool, str]:
    """Indica si la combinación modelo × balanceo tiene sentido.

    Parameters
    ----------
    spec : EspecificacionModelo
    balanceo : str
        Uno de ``CATALOGO_BALANCEO``.
    y : Series or ndarray, optional
        Etiqueta de entrenamiento. Si se da y el balanceo es por sobremuestreo, se
        comprueba que las clases estén realmente desbalanceadas.

    Returns
    -------
    aplica : bool
    motivo : str
        Vacío si aplica; si no, la razón que se registra en la tabla maestra.

    Raises
    ------
    ValueError
        Si ``balanceo`` no está en el catálogo.
    """
    if balanceo not in CATALOGO_BALANCEO:
        raise ValueError(f'balanceo desconocido: {balanceo!r}')
    if spec.tarea == 'regresion' and balanceo != 'ninguno':
        return False, 'el balanceo de clases no aplica en regresión'
    if balanceo == 'class_weight' and spec.peso_clases is None:
        return False, f'{spec.nombre} no admite ponderación de clases'
    if balanceo in ('smote', 'adasyn') and y is not None:
        prev = float(np.mean(np.asarray(y)))
        minoritaria = min(prev, 1 - prev)
        if minoritaria > PREVALENCIA_MAXIMA_SOBREMUESTREO:
            return False, (f'clases casi equilibradas (clase minoritaria {minoritaria:.3f}): '
                           f'el sobremuestreo no genera muestras')
    return True, ''


def construir_muestreador(balanceo: str, seed: int = SEED):
    """Muestreador de imbalanced-learn para el balanceo pedido.

    Parameters
    ----------
    balanceo : str
    seed : int

    Returns
    -------
    SMOTE, ADASYN or None
        ``None`` para ``'ninguno'`` y ``'class_weight'``.
    """
    if balanceo == 'smote':
        return SMOTE(random_state=seed)
    if balanceo == 'adasyn':
        return ADASYN(random_state=seed)
    return None


def valor_peso(spec: EspecificacionModelo, y: pd.Series | np.ndarray | None) -> tuple[str, object]:
    """Valor concreto del parámetro de ponderación de clases.

    Parameters
    ----------
    spec : EspecificacionModelo
        Debe tener ``peso_clases``.
    y : Series or ndarray, optional
        Etiqueta del **conjunto de entrenamiento del pliegue** en curso; necesaria cuando
        el valor es ``'auto'`` (XGBoost), que se resuelve como ``negativos / positivos``.

    Returns
    -------
    parametro : str
    valor : object

    Raises
    ------
    ValueError
        Si el valor es ``'auto'`` y no se dio ``y``.
    """
    param, valor = spec.peso_clases
    if valor == 'auto':
        if y is None:
            raise ValueError(f'{spec.nombre} necesita y para calcular {param}')
        y = np.asarray(y)
        pos = max(int((y == 1).sum()), 1)
        valor = float((y == 0).sum()) / pos
    return param, valor


def construir_pipeline(spec: EspecificacionModelo, balanceo: str, X: pd.DataFrame,
                       seed: int = SEED, y: pd.Series | None = None) -> Pipeline:
    """Ensambla ``preprocesado → [muestreo] → modelo`` para una combinación.

    Parameters
    ----------
    spec : EspecificacionModelo
    balanceo : str
        Uno de ``CATALOGO_BALANCEO``.
    X : DataFrame
        Predictores (solo se usan tipos y nombres de columna).
    seed : int
    y : Series, optional
        Etiqueta de entrenamiento del pliegue; necesaria para ``scale_pos_weight`` y para
        comprobar que el sobremuestreo tiene sentido.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Sin ajustar. Los hiperparámetros del modelo se referencian como ``modelo__<param>``.

    Raises
    ------
    ValueError
        Si la combinación no aplica (ver :func:`aplica`).
    """
    ok, motivo = aplica(spec, balanceo, y)
    if not ok:
        raise ValueError(f'{spec.nombre} × {balanceo}: {motivo}')
    pasos = [('preprocesado', construir_preprocesador(X))]
    muestreador = construir_muestreador(balanceo, seed)
    if muestreador is not None:
        pasos.append(('muestreo', muestreador))
    estimador = spec.construir(seed)
    if balanceo == 'class_weight':
        param, valor = valor_peso(spec, y)
        estimador.set_params(**{param: valor})
    pasos.append(('modelo', estimador))
    return Pipeline(pasos)
