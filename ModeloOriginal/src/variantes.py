"""Variantes del modelo original para el estudio de ablación (protocolo, sección 9) y utilidades de reajuste.

Cada variante retira un componente y deja lo demás igual:

- ``bdl_sin_mezcla``: ``n_componentes = 1``. Sin clases latentes, el EM generalizado es un boosting de mínimos cuadrados.
- ``bdl_sin_puntaje`` (clasificación): ``usar_puntaje = False``. El modelo aprende de la etiqueta, como los benchmarks.
- ``bdl_conforme`` (clasificación): misma regla de aprendizaje, pero la probabilidad sale de la distribución empírica
  de los residuos de calibración en lugar de la mezcla.

Las variantes no tocan ``original_model.py``: se construyen aquí a partir de sus clases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import balanceo as bal
from .modelos import EspecificacionModelo, catalogo, especificacion_bdl
from .original_model import ClasificadorBDL, RegresorBDL

VARIANTES = {'bdl_sin_mezcla': ('clasificacion', 'regresion'), 'bdl_sin_puntaje': ('clasificacion',),
             'bdl_conforme': ('clasificacion',)}
NOMBRES = {'bdl_sin_mezcla': 'BDL sin mezcla (K = 1)', 'bdl_sin_puntaje': 'BDL sin puntaje (aprende de la etiqueta)',
           'bdl_conforme': 'BDL con salida conforme empírica'}


class ClasificadorBDLConforme(ClasificadorBDL):
    """``ClasificadorBDL`` cuya probabilidad es la fracción de residuos de calibración que superan el umbral."""

    def predict_proba(self, X) -> np.ndarray:
        if self.modo_ != 'excedencia':
            return super().predict_proba(X)
        p = self.regresor_.prob_excedencia(X, self.umbral_, metodo='conforme')
        return np.c_[1.0 - p, p]


def especificacion_variante(nombre: str, tarea: str) -> EspecificacionModelo:
    """Especificación de una variante: el mismo espacio de búsqueda del original sin el hiperparámetro que se fija.

    Parameters
    ----------
    nombre : {'bdl_sin_mezcla', 'bdl_sin_puntaje', 'bdl_conforme'}
    tarea : {'clasificacion', 'regresion'}

    Returns
    -------
    EspecificacionModelo

    Raises
    ------
    ValueError
        Si la variante no existe o no aplica a la tarea.
    """
    if nombre not in VARIANTES or tarea not in VARIANTES[nombre]:
        raise ValueError(f'la variante {nombre!r} no aplica a {tarea!r}')
    base = especificacion_bdl(tarea)
    espacio = dict(base.espacio)
    clase = ClasificadorBDL if tarea == 'clasificacion' else RegresorBDL
    if nombre == 'bdl_sin_mezcla':
        espacio.pop('modelo__n_componentes')
        construir = lambda s: clase(n_componentes=1, random_state=s)                    # noqa: E731
    elif nombre == 'bdl_sin_puntaje':
        espacio.pop('modelo__usar_puntaje')
        espacio.pop('modelo__n_componentes')                                             # sin puntaje no hay mezcla que elegir
        construir = lambda s: ClasificadorBDL(usar_puntaje=False, random_state=s)       # noqa: E731
    else:
        espacio.pop('modelo__usar_puntaje')
        construir = lambda s: ClasificadorBDLConforme(random_state=s)                   # noqa: E731
    return EspecificacionModelo(nombre, tarea, construir, base.espacio_grid, espacio, peso_clases=None,
                                complejidad=base.complejidad, justificacion=NOMBRES[nombre])


def reajustar(tarea: str, modelo: str, balanceo: str, hiperparametros: dict, X: pd.DataFrame, y, semilla: int,
              puntaje=None):
    """Ajusta un modelo del catálogo con hiperparámetros dados (sin búsqueda) y devuelve el pipeline.

    Parameters
    ----------
    tarea, modelo, balanceo : str
    hiperparametros : dict
        ``{'modelo__param': valor}``, tal como quedan en la tabla maestra.
    X : DataFrame
        Predictores crudos.
    y : array-like
        Objetivo de la tarea.
    semilla : int
        Semilla del modelo.
    puntaje : array-like, optional
        Puntaje continuo; solo lo usa el modelo original en clasificación.

    Returns
    -------
    Pipeline ajustado.
    """
    spec = catalogo(tarea, incluir_nuevo=True, incluir_original=True)[modelo]
    y = pd.Series(np.asarray(y))
    tuberia = bal.construir_pipeline(spec, balanceo, X, semilla, y if tarea == 'clasificacion' else None)
    tuberia.set_params(**hiperparametros)
    extra = {'modelo__puntaje': np.asarray(puntaje, dtype=float)} if (modelo == 'bdl' and tarea == 'clasificacion' and puntaje is not None) else {}
    return tuberia.fit(X, y, **extra)
