"""Preprocesamiento encapsulado en un ``ColumnTransformer``.

El transformador vive **dentro** de cada pipeline, de modo que en cada pliegue de
cualquier validación cruzada (externa o interna) el escalado y la codificación se
ajustan únicamente con las filas de entrenamiento de ese pliegue.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def columnas_por_tipo(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa las columnas numéricas/booleanas de las categóricas.

    Parameters
    ----------
    X : DataFrame
        Predictores. Las categóricas deben ser de tipo ``object`` o ``category``.

    Returns
    -------
    numericas, categoricas : list of str
    """
    numericas = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categoricas = [c for c in X.columns if c not in numericas]
    return numericas, categoricas


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """Escala las numéricas y codifica las categóricas con indicadoras.

    Parameters
    ----------
    X : DataFrame
        Predictores; solo se usan los nombres y tipos de columna, no sus valores.

    Returns
    -------
    ColumnTransformer
        ``StandardScaler`` sobre numéricas/booleanas y ``OneHotEncoder(drop='first',
        handle_unknown='ignore')`` sobre categóricas. ``drop='first'`` evita la
        colinealidad perfecta entre indicadoras; ``handle_unknown='ignore'`` protege
        frente a niveles que aparezcan solo en prueba.

    Notes
    -----
    El objeto devuelto **no está ajustado**: se ajusta dentro del pipeline con las filas
    de entrenamiento de cada pliegue.
    """
    numericas, categoricas = columnas_por_tipo(X)
    return ColumnTransformer([
        ('num', StandardScaler(), numericas),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categoricas),
    ])


def nombres_variables(preprocesador: ColumnTransformer) -> list[str]:
    """Nombres de las columnas tras transformar.

    Parameters
    ----------
    preprocesador : ColumnTransformer
        **Ajustado** (los nombres de las indicadoras dependen de los niveles vistos).

    Returns
    -------
    list of str
        Sin el prefijo ``num__`` / ``cat__`` que añade scikit-learn.
    """
    return [n.split('__', 1)[-1] for n in preprocesador.get_feature_names_out()]
