"""Catálogo de modelos base y sus espacios de búsqueda.

Cada modelo se describe una sola vez mediante :class:`EspecificacionModelo`, y de esa
descripción se derivan los cuatro optimizadores: la rejilla explícita alimenta a Grid
Search y el espacio tipado (``espacio``) alimenta a Random Search, Optuna y al algoritmo
genético, que lo interpretan cada uno a su manera (ver :mod:`optimizacion`).

Formato del espacio tipado
--------------------------
``{'modelo__param': (tipo, ...)}`` con ``tipo`` en:

- ``('log', bajo, alto)``: flotante muestreado en escala logarítmica (C, alpha, learning rate).
- ``('float', bajo, alto)``: flotante uniforme.
- ``('int', bajo, alto)``: entero uniforme (inclusive).
- ``('int_log', bajo, alto)``: entero muestreado en escala logarítmica (vecinos, estimadores).
- ``('cat', [opciones])``: categórico.

Los rangos están anclados en el tamaño del problema (n ≈ 39.000 filas de entrenamiento,
p = 78 columnas tras codificar) y su justificación va en ``justificacion``.

Modelo nuevo
------------
Además de los 7 + 7 modelos de la guía, :func:`especificacion_ebm` describe el **modelo nuevo**
del Entregable 3, la *Explainable Boosting Machine* (GA²M), que no forma parte de las notas del
curso. :func:`catalogo` solo la incluye si se pide con ``incluir_nuevo=True``, de modo que el
producto combinatorio de la guía sigue siendo de 140 corridas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

TAREAS = ('clasificacion', 'regresion')


@dataclass
class EspecificacionModelo:
    """Descripción completa de un modelo base.

    Attributes
    ----------
    nombre : str
        Identificador corto (``'knn'``, ``'xgboost'``…).
    tarea : {'clasificacion', 'regresion'}
    construir : callable
        ``seed -> estimador`` sin ajustar, con la semilla ya propagada.
    espacio_grid : dict
        Rejilla explícita para Grid Search (``{'modelo__param': [valores]}``).
    espacio : dict
        Espacio tipado para Random / Optuna / Genético (ver módulo).
    peso_clases : tuple or None
        ``(parametro, valor)`` que implementa el balanceo por ponderación en este
        estimador (``('class_weight', 'balanced')``, ``('priors', [0.5, 0.5])``,
        ``('scale_pos_weight', 'auto')``). ``None`` si el estimador no lo admite.
    complejidad : dict
        Órdenes asintóticos de entrenamiento e inferencia (n filas, p columnas,
        T árboles, d profundidad, k iteraciones, s vectores soporte).
    justificacion : str
        Por qué esos hiperparámetros y esos rangos.
    fidelidad : dict or None
        Multi-fidelidad (guía §3.3): ``{'busqueda': {...}, 'final': {...}}``. Durante la búsqueda
        de hiperparámetros el pipeline usa los parámetros de ``'busqueda'`` (baja fidelidad, más
        barata) y el modelo que se evalúa en el bucle externo y se guarda se reajusta con los de
        ``'final'``. ``None`` si el modelo se busca a fidelidad completa.
    """
    nombre: str
    tarea: str
    construir: Callable[[int], BaseEstimator]
    espacio_grid: dict[str, list]
    espacio: dict[str, tuple]
    peso_clases: tuple[str, object] | None = None
    complejidad: dict[str, str] = field(default_factory=dict)
    justificacion: str = ''
    fidelidad: dict[str, dict] | None = None

    def n_configuraciones_grid(self) -> int:
        """Número de combinaciones de la rejilla explícita.

        Returns
        -------
        int
            Producto del número de valores de cada hiperparámetro de ``espacio_grid``.
        """
        n = 1
        for v in self.espacio_grid.values():
            n *= len(v)
        return n


# ----------------------------------------------------------------------------- KNN
_KNN_GRID = {'modelo__n_neighbors': [10, 20, 40, 80, 150, 250, 400, 600],
             'modelo__weights': ['uniform', 'distance'],
             'modelo__p': [1, 2]}
_KNN_ESPACIO = {'modelo__n_neighbors': ('int_log', 5, 600),
                'modelo__weights': ('cat', ['uniform', 'distance']),
                'modelo__p': ('cat', [1, 2])}
_KNN_JUST = ('n_neighbors en escala logarítmica hasta 600: en la Entrega 2 el AUC seguía creciendo en '
             'k = 200, así que la rejilla debe cubrir más allá; weights=distance suaviza el voto y p '
             'elige entre distancia Manhattan y euclídea. Con p = 78 el árbol KD degenera a fuerza bruta.')
_KNN_JUST_REG = ('Mismo espacio que el KNN de clasificación, reutilizado por analogía: la Entrega 2 no '
                 'entrenó KNeighborsRegressor, pero X, n ≈ 39.000 y p = 78 son idénticos. En regresión '
                 'weights=distance pondera la media de los vecinos por el inverso de la distancia y p '
                 'elige entre Manhattan y euclídea; n_neighbors en escala logarítmica hasta 600 para '
                 'cubrir el régimen en que el error deja de bajar.')
_KNN_COMPLEJIDAD = {'entrenamiento': 'O(1) fuerza bruta; O(n·p·log n) KD/Ball Tree',
                    'inferencia': 'O(n·p) por consulta en fuerza bruta; O(p·log n) en árbol con p pequeño'}


# ---------------------------------------------------------------------- catálogos
def catalogo_clasificacion() -> dict[str, EspecificacionModelo]:
    """Los siete modelos de clasificación exigidos por la guía (§2, Etapa 1 y §7.2).

    Returns
    -------
    dict
        ``nombre -> EspecificacionModelo``, en el orden de la guía.

    Notes
    -----
    La semilla no se fija aquí: cada ``construir(seed)`` la recibe de
    ``balanceo.construir_pipeline`` en el momento de ensamblar el pipeline.
    """
    return {
        'knn': EspecificacionModelo(
            'knn', 'clasificacion', lambda s: KNeighborsClassifier(),
            _KNN_GRID, _KNN_ESPACIO, peso_clases=None,
            complejidad=_KNN_COMPLEJIDAD, justificacion=_KNN_JUST),
        'naive_bayes': EspecificacionModelo(
            'naive_bayes', 'clasificacion', lambda s: GaussianNB(),
            {'modelo__var_smoothing': list(np.logspace(-11, -3, 30))},
            {'modelo__var_smoothing': ('log', 1e-11, 1e-3)},
            peso_clases=('priors', [0.5, 0.5]),
            complejidad={'entrenamiento': 'O(n·p)', 'inferencia': 'O(p·clases) por consulta'},
            justificacion=('GaussianNB solo tiene var_smoothing, que añade una fracción de la varianza '
                           'máxima a todas las varianzas para estabilizar; se explora en escala log. '
                           'No admite class_weight: priors=[0.5, 0.5] es su equivalente exacto '
                           '(reponderar las clases a prioris iguales).')),
        'logistica': EspecificacionModelo(
            'logistica', 'clasificacion',
            lambda s: LogisticRegression(solver='liblinear', max_iter=2000, random_state=s),
            {'modelo__C': list(np.logspace(-3, 2, 15)), 'modelo__l1_ratio': [1.0, 0.0]},
            {'modelo__C': ('log', 1e-3, 1e2), 'modelo__l1_ratio': ('cat', [1.0, 0.0])},
            peso_clases=('class_weight', 'balanced'),
            complejidad={'entrenamiento': 'O(k·n·p) (liblinear, k iteraciones)', 'inferencia': 'O(p)'},
            justificacion=('C es el inverso de la penalización (escala log 1e-3..1e2, como en la Entrega '
                           '2, donde el óptimo fue interior). l1_ratio=1 es la penalización L1 (Lasso '
                           'logístico) y l1_ratio=0 la L2 (Ridge logístico); scikit-learn ≥ 1.8 deprecó '
                           'el parámetro penalty en favor de l1_ratio. liblinear soporta ambas en binario.')),
        'arbol': EspecificacionModelo(
            'arbol', 'clasificacion', lambda s: DecisionTreeClassifier(random_state=s),
            {'modelo__max_depth': [4, 8, 12, 20], 'modelo__min_samples_leaf': [5, 20, 50, 100],
             'modelo__ccp_alpha': [0.0, 1e-4, 1e-3]},
            {'modelo__max_depth': ('int', 3, 30), 'modelo__min_samples_leaf': ('int_log', 1, 300),
             'modelo__ccp_alpha': ('log', 1e-6, 1e-2), 'modelo__criterion': ('cat', ['gini', 'entropy'])},
            peso_clases=('class_weight', 'balanced'),
            complejidad={'entrenamiento': 'O(n·p·log n)', 'inferencia': 'O(d) por consulta'},
            justificacion=('max_depth y min_samples_leaf controlan el sobreajuste por dos vías (profundidad '
                           'y tamaño mínimo de hoja, log hasta 300 ≈ 0,8 % de n); ccp_alpha poda por '
                           'coste-complejidad en escala log.')),
        'random_forest': EspecificacionModelo(
            'random_forest', 'clasificacion',
            lambda s: RandomForestClassifier(random_state=s, n_jobs=-1),
            {'modelo__max_depth': [8, 16, None],
             'modelo__max_features': ['sqrt', 0.3, 0.5], 'modelo__min_samples_leaf': [1, 5, 20, 50]},
            {'modelo__max_depth': ('int', 4, 40),
             'modelo__max_features': ('cat', ['sqrt', 'log2', 0.3, 0.5]),
             'modelo__min_samples_leaf': ('int_log', 1, 100)},
            peso_clases=('class_weight', 'balanced'),
            complejidad={'entrenamiento': 'O(T·n·p·log n)', 'inferencia': 'O(T·d) por consulta'},
            justificacion=('n_estimators no se optimiza: el error de un bosque decrece de forma monótona '
                           'con el número de árboles y se estabiliza, así que no tiene un óptimo interior que '
                           'buscar. Se aplica multi-fidelidad (guía §3.3, «presupuestos parciales: menos '
                           'árboles»): durante la búsqueda cada configuración se evalúa con 100 árboles y el '
                           'modelo que se mide en el bucle externo y se guarda se reajusta con 300 (factor de '
                           'reducción 3). Supuesto: el orden relativo de las configuraciones de max_features, '
                           'max_depth y min_samples_leaf se conserva con 100 árboles; se contrasta en la '
                           'sección de cómputo. El presupuesto se gasta en max_features (decorrelación entre '
                           'árboles), max_depth y min_samples_leaf (regularización de cada árbol).'),
            fidelidad={'busqueda': {'modelo__n_estimators': 100}, 'final': {'modelo__n_estimators': 300}}),
        'xgboost': EspecificacionModelo(
            'xgboost', 'clasificacion',
            lambda s: XGBClassifier(tree_method='hist', random_state=s, n_jobs=-1, verbosity=0,
                                    eval_metric='logloss'),
            {'modelo__n_estimators': [200, 500], 'modelo__max_depth': [3, 6],
             'modelo__learning_rate': [0.03, 0.1], 'modelo__subsample': [0.7, 1.0],
             'modelo__colsample_bytree': [0.7, 1.0]},
            {'modelo__n_estimators': ('int_log', 100, 1000), 'modelo__max_depth': ('int', 2, 10),
             'modelo__learning_rate': ('log', 1e-3, 0.3), 'modelo__subsample': ('float', 0.5, 1.0),
             'modelo__colsample_bytree': ('float', 0.5, 1.0), 'modelo__reg_lambda': ('log', 1e-2, 10),
             'modelo__min_child_weight': ('int_log', 1, 50)},
            peso_clases=('scale_pos_weight', 'auto'),
            complejidad={'entrenamiento': 'O(T·d·(n·p + bins·p)) con hist; O(T·d·n·p·log n) exacto',
                         'inferencia': 'O(T·d) por consulta'},
            justificacion=('learning_rate en log y n_estimators en log porque interactúan (más árboles '
                           'compensan menor tasa); subsample/colsample introducen aleatoriedad '
                           'regularizadora; reg_lambda y min_child_weight regularizan las hojas. '
                           'tree_method=hist por coste (guía §4.2). scale_pos_weight = neg/pos es el '
                           'equivalente de class_weight.')),
        'svm': EspecificacionModelo(
            'svm', 'clasificacion',
            lambda s: LinearSVC(penalty='l2', dual=False, max_iter=5000, random_state=s),
            {'modelo__C': list(np.logspace(-4, 2, 30))},
            {'modelo__C': ('log', 1e-4, 1e2)},
            peso_clases=('class_weight', 'balanced'),
            complejidad={'entrenamiento': 'O(k·n·p) lineal; O(n²·p)–O(n³) con núcleo',
                         'inferencia': 'O(p) lineal; O(s·p) con núcleo'},
            justificacion=('Con n ≈ 39.000 el SVM con núcleo es cuadrático-cúbico en n (guía §4.2), de '
                           'modo que el modelo base es LinearSVC (dual=False, lineal en n); SVC(rbf) se '
                           'compara sobre una submuestra en la sección de cómputo. Solo penalización L2: '
                           'medido el 29-sep sobre un pliegue interno real (20.788 filas), el solucionador '
                           'primal de L1 no converge en 1.000 iteraciones con C ≥ 1 (≈ 16 s por ajuste con '
                           'max_iter=5000, sin llegar), mientras que L2 converge en 5 iteraciones y ≈ 1 s. La '
                           'selección de variables por L1 ya la cubre la regresión logística. Único '
                           'hiperparámetro: C en escala log entre 1e-4 y 1e2.')),
    }


def catalogo_regresion() -> dict[str, EspecificacionModelo]:
    """Los siete modelos de regresión exigidos por la guía (§2, Etapa 1 y §7.2).

    Returns
    -------
    dict
        ``nombre -> EspecificacionModelo``, en el orden de la guía.
    """
    return {
        'knn': EspecificacionModelo(
            'knn', 'regresion', lambda s: KNeighborsRegressor(),
            _KNN_GRID, _KNN_ESPACIO, complejidad=_KNN_COMPLEJIDAD, justificacion=_KNN_JUST_REG),
        'ridge': EspecificacionModelo(
            'ridge', 'regresion', lambda s: Ridge(),
            {'modelo__alpha': list(np.logspace(-3, 5, 30))},
            {'modelo__alpha': ('log', 1e-3, 1e5)},
            complejidad={'entrenamiento': 'O(n·p² + p³) cholesky; O(k·n·p) SAGA', 'inferencia': 'O(p)'},
            justificacion=('alpha en log entre 1e-3 y 1e5: en la Entrega 2 el óptimo fue 1000 y el '
                           'R² cae a partir de 1e4, así que el rango cubre ambos lados del óptimo.')),
        'lasso': EspecificacionModelo(
            'lasso', 'regresion', lambda s: Lasso(max_iter=20000, random_state=s),
            {'modelo__alpha': list(np.logspace(-6, -1, 30))},
            {'modelo__alpha': ('log', 1e-6, 1e-1)},
            complejidad={'entrenamiento': 'O(k·n·p) descenso por coordenadas', 'inferencia': 'O(p)'},
            justificacion=('alpha en log hasta 1e-1, valor que ya anula todos los coeficientes (R² ≈ 0 '
                           'en la Entrega 2); el óptimo anterior (1e-4) queda interior.')),
        'arbol': EspecificacionModelo(
            'arbol', 'regresion', lambda s: DecisionTreeRegressor(random_state=s),
            {'modelo__max_depth': [4, 8, 12, 20], 'modelo__min_samples_leaf': [5, 20, 50, 100],
             'modelo__ccp_alpha': [0.0, 1e-6, 1e-5]},
            {'modelo__max_depth': ('int', 3, 30), 'modelo__min_samples_leaf': ('int_log', 1, 300),
             'modelo__ccp_alpha': ('log', 1e-8, 1e-4)},
            complejidad={'entrenamiento': 'O(n·p·log n)', 'inferencia': 'O(d) por consulta'},
            justificacion=('Igual que el árbol de clasificación; ccp_alpha en una escala menor porque '
                           'la impureza (MSE) del objetivo, acotado en [0,11; 0,50], es del orden de 1e-3.')),
        'random_forest': EspecificacionModelo(
            'random_forest', 'regresion',
            lambda s: RandomForestRegressor(random_state=s, n_jobs=-1),
            {'modelo__max_depth': [8, 16, None],
             'modelo__max_features': ['sqrt', 0.3, 0.5], 'modelo__min_samples_leaf': [1, 5, 20, 50]},
            {'modelo__max_depth': ('int', 4, 40),
             'modelo__max_features': ('cat', ['sqrt', 'log2', 0.3, 0.5]),
             'modelo__min_samples_leaf': ('int_log', 1, 100)},
            complejidad={'entrenamiento': 'O(T·n·p·log n)', 'inferencia': 'O(T·d) por consulta'},
            justificacion='Igual que el bosque de clasificación, incluida la multi-fidelidad (100 → 300 árboles).',
            fidelidad={'busqueda': {'modelo__n_estimators': 100}, 'final': {'modelo__n_estimators': 300}}),
        'xgboost': EspecificacionModelo(
            'xgboost', 'regresion',
            lambda s: XGBRegressor(tree_method='hist', random_state=s, n_jobs=-1, verbosity=0),
            {'modelo__n_estimators': [200, 500], 'modelo__max_depth': [3, 6],
             'modelo__learning_rate': [0.03, 0.1], 'modelo__subsample': [0.7, 1.0],
             'modelo__colsample_bytree': [0.7, 1.0]},
            {'modelo__n_estimators': ('int_log', 100, 1000), 'modelo__max_depth': ('int', 2, 10),
             'modelo__learning_rate': ('log', 1e-3, 0.3), 'modelo__subsample': ('float', 0.5, 1.0),
             'modelo__colsample_bytree': ('float', 0.5, 1.0), 'modelo__reg_lambda': ('log', 1e-2, 10),
             'modelo__min_child_weight': ('int_log', 1, 50)},
            complejidad={'entrenamiento': 'O(T·d·(n·p + bins·p)) con hist; O(T·d·n·p·log n) exacto',
                         'inferencia': 'O(T·d) por consulta'},
            justificacion='Igual que XGBoost de clasificación, con objetivo reg:squarederror.'),
        'svr': EspecificacionModelo(
            'svr', 'regresion',
            lambda s: LinearSVR(loss='squared_epsilon_insensitive', dual=False, max_iter=10000, random_state=s),
            {'modelo__C': list(np.logspace(-3, 2, 10)), 'modelo__epsilon': [0.001, 0.01, 0.03]},
            {'modelo__C': ('log', 1e-3, 1e2), 'modelo__epsilon': ('log', 1e-4, 5e-2)},
            complejidad={'entrenamiento': 'O(k·n·p) lineal; O(n²·p)–O(n³) con núcleo',
                         'inferencia': 'O(p) lineal; O(s·p) con núcleo'},
            justificacion=('LinearSVR por la misma razón de coste que LinearSVC. Pérdida ε-insensible '
                           'cuadrática con el solucionador primal (dual=False): medido el 29-sep sobre un '
                           'pliegue interno real, la pérdida ε-insensible lineal solo admite el solucionador '
                           'dual, que no converge en 1.000 iteraciones para ningún C (45-85 s por ajuste con '
                           'max_iter=10000, sin llegar), mientras que la cuadrática converge en 4-5 '
                           'iteraciones y < 1 s. epsilon (ancho del tubo insensible) en log entre 1e-4 y '
                           '5e-2 porque la desviación típica del objetivo es ≈ 0,067: un tubo mayor que eso '
                           'ignoraría casi toda la señal.')),
    }


class EBMConPesos(ClassifierMixin, BaseEstimator):
    """EBM de clasificación con ``class_weight``, que la librería no ofrece como parámetro.

    ``interpret`` admite ``sample_weight`` en ``fit`` pero no ``class_weight``. Esta envoltura
    convierte ``class_weight='balanced'`` en pesos por fila, ``n / (2·n_clase)``, que es lo que
    hace scikit-learn por dentro en los modelos que sí lo admiten. Expone los hiperparámetros
    que se buscan y la fidelidad (``outer_bags``); el resto queda en los valores por defecto de
    la EBM. El modelo ajustado queda en ``ebm_`` (funciones de forma, ``explain_global``).

    Parameters
    ----------
    learning_rate, max_leaves, min_samples_leaf, interactions, max_bins, outer_bags
        Los de ``ExplainableBoostingClassifier``.
    class_weight : {'balanced'} or dict or None
    n_jobs : int
        Procesos para las bolsas externas. En la búsqueda se usan 2 por ajuste porque los
        pliegues internos ya corren en paralelo (3 × 2 = 6 núcleos).
    random_state : int or None
    """

    def __init__(self, learning_rate: float = 0.015, max_leaves: int = 2, min_samples_leaf: int = 4,
                 interactions: int = 10, max_bins: int = 1024, outer_bags: int = 8,
                 class_weight=None, n_jobs: int = 2, random_state: int | None = None) -> None:
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.interactions = interactions
        self.max_bins = max_bins
        self.outer_bags = outer_bags
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """Ajusta la EBM con los pesos por fila que implican ``class_weight``."""
        pesos = None if self.class_weight is None else compute_sample_weight(self.class_weight, y)
        self.ebm_ = ExplainableBoostingClassifier(
            learning_rate=self.learning_rate, max_leaves=self.max_leaves,
            min_samples_leaf=self.min_samples_leaf, interactions=self.interactions,
            max_bins=self.max_bins, outer_bags=self.outer_bags, n_jobs=self.n_jobs,
            random_state=self.random_state).fit(X, y, sample_weight=pesos)
        self.classes_ = self.ebm_.classes_
        return self

    def predict_proba(self, X):
        """Probabilidades de clase de la EBM ajustada."""
        return self.ebm_.predict_proba(X)

    def predict(self, X):
        """Clase predicha por la EBM ajustada."""
        return self.ebm_.predict(X)

    def decision_function(self, X):
        """Puntaje de decisión (logit) de la EBM ajustada."""
        return self.ebm_.decision_function(X)


_EBM_GRID = {'modelo__learning_rate': [0.008, 0.02, 0.05], 'modelo__min_samples_leaf': [4, 20, 100],
             'modelo__interactions': [0, 10], 'modelo__max_leaves': [2, 3]}
_EBM_ESPACIO = {'modelo__learning_rate': ('log', 0.008, 0.1), 'modelo__max_leaves': ('int', 2, 4),
                'modelo__min_samples_leaf': ('int_log', 2, 200), 'modelo__interactions': ('int', 0, 20),
                'modelo__max_bins': ('int_log', 64, 1024)}
_EBM_FIDELIDAD = {'busqueda': {'modelo__outer_bags': 2}, 'final': {'modelo__outer_bags': 8}}
_EBM_COMPLEJIDAD = {'entrenamiento': 'O(B·R·p·(n + bins)) efectos principales + O(p²·bins²) detección '
                                     'de pares (FAST) + O(B·R·K·(n + bins²)) pares',
                    'inferencia': 'O(p + K) por consulta: una búsqueda de bin por término'}
_EBM_JUST = (
    'Modelo nuevo, no explicado en clase: Explainable Boosting Machine (Lou, Caruana, Gehrke y Hooker, '
    '2013; paquete interpret de Nori y otros, 2019). Es un modelo aditivo generalizado con interacciones '
    'por pares (GA²M): g(E[y]) = β<sub>0</sub> + Σ<sub>j</sub> <i>f</i><sub>j</sub>(<i>x</i><sub>j</sub>) + Σ<sub>ij</sub> <i>f</i><sub>ij</sub>(<i>x</i><sub>i</sub>, <i>x</i><sub>j</sub>). Cada función de forma <i>f</i><sub>j</sub> se aprende '
    'por boosting cíclico de árboles de pocas hojas sobre una sola variable a la vez, con una tasa de '
    'aprendizaje baja para que el orden de las variables no importe, y se promedia sobre outer_bags '
    'submuestras. Es tan legible como una regresión logística (cada variable aporta una curva que se '
    'puede graficar) y capta efectos no lineales como el boosting, que es justo lo que faltó en la '
    'Entrega 2. learning_rate en log entre 0,008 y 0,1: incluye los valores por defecto de la librería '
    '(0,015 en clasificación y 0,04 en regresión), y por debajo de 0,008 cada ajuste tarda el doble '
    '(26 s frente a 12 s medidos sobre un pliegue interno de 20.788 filas). max_leaves entre 2 y 4: la '
    'EBM aprende en pasos pequeños. min_samples_leaf en log entre 2 y 200 regulariza las funciones de '
    'forma. interactions entre 0 (GAM puro, sin pares) y 20: el valor por defecto, 3·p ≈ 234 pares, '
    'tardó 112 s por ajuste y en el benchmark sobre entrenamiento las interacciones no aportaron. '
    'max_bins en log entre 64 y 1024 fija la resolución de las curvas. Multi-fidelidad como en el '
    'Random Forest: la búsqueda usa 2 bolsas externas y el modelo final 8 (las bolsas promedian, no '
    'cambian la forma de la función, así que el orden de las configuraciones se conserva). '
    'class_weight se implementa con sample_weight (EBMConPesos).')


def especificacion_ebm(tarea: str) -> EspecificacionModelo:
    """Especificación del modelo nuevo, la EBM, para la tarea pedida.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}

    Returns
    -------
    EspecificacionModelo
        Con el mismo espacio en las dos tareas y multi-fidelidad 2 → 8 bolsas externas.

    Raises
    ------
    ValueError
        Si la tarea no es una de ``TAREAS``.
    """
    if tarea == 'clasificacion':
        return EspecificacionModelo(
            'ebm', 'clasificacion', lambda s: EBMConPesos(random_state=s),
            _EBM_GRID, _EBM_ESPACIO, peso_clases=('class_weight', 'balanced'),
            complejidad=_EBM_COMPLEJIDAD, justificacion=_EBM_JUST, fidelidad=_EBM_FIDELIDAD)
    if tarea == 'regresion':
        return EspecificacionModelo(
            'ebm', 'regresion',
            lambda s: ExplainableBoostingRegressor(interactions=10, outer_bags=8, n_jobs=2, random_state=s),
            _EBM_GRID, _EBM_ESPACIO, complejidad=_EBM_COMPLEJIDAD, justificacion=_EBM_JUST,
            fidelidad=_EBM_FIDELIDAD)
    raise ValueError(f'tarea desconocida: {tarea!r}; use una de {TAREAS}')


def catalogo(tarea: str, incluir_nuevo: bool = False) -> dict[str, EspecificacionModelo]:
    """Catálogo de la tarea pedida.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    incluir_nuevo : bool
        Añadir el modelo nuevo (``'ebm'``) a los siete de la guía.

    Returns
    -------
    dict
        ``nombre -> EspecificacionModelo``.

    Raises
    ------
    ValueError
        Si la tarea no es una de ``TAREAS``.
    """
    if tarea == 'clasificacion':
        cat = catalogo_clasificacion()
    elif tarea == 'regresion':
        cat = catalogo_regresion()
    else:
        raise ValueError(f'tarea desconocida: {tarea!r}; use una de {TAREAS}')
    if incluir_nuevo:
        cat['ebm'] = especificacion_ebm(tarea)
    return cat
