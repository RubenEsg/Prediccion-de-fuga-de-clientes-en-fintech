"""Optimización de hiperparámetros con validación cruzada anidada (guía §3).

Los cuatro métodos comparten interfaz: reciben un pipeline sin ajustar, un espacio y
los datos del **pliegue de entrenamiento externo**, ejecutan su búsqueda con una
validación cruzada interna propia y devuelven un :class:`ResultadoBusqueda` con el
pipeline reajustado sobre todo ese pliegue. :func:`validacion_anidada` los envuelve en el
bucle externo —el único que produce la métrica reportada— y, al terminar, repite la
búsqueda una vez sobre todo el conjunto de entrenamiento para obtener el **modelo final**
y sus hiperparámetros finales (los que exige la tabla maestra, §7.3).

Presupuesto
-----------
Para comparar métodos de forma justa cada optimizador evalúa, por defecto, tantas
configuraciones como tiene la rejilla de Grid Search del mismo modelo
(:func:`presupuesto_por_defecto`; todas las rejillas tienen entre 30 y 48 puntos). Grid es
siempre exhaustivo; Random, Bayesiana y Genética consumen exactamente ese número de
configuraciones distintas, y ``n_evaluaciones`` registra las que realmente se evaluaron.
"""
from __future__ import annotations

import math
import random
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, KFold, ParameterGrid, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.utils import check_random_state

from . import evaluacion
from .config import K_EXT, K_INT, PRESUPUESTO_MINIMO, SEED

OPTIMIZADORES: list[str] = ['grid', 'random', 'bayesiana', 'genetica']
"""Los cuatro métodos de la Etapa 3 (guía §2 y §3.3), en el orden de la guía."""


# ---------------------------------------------------------------------- resultados
@dataclass
class ResultadoBusqueda:
    """Salida de una búsqueda sobre un conjunto de entrenamiento.

    Attributes
    ----------
    optimizador : str
    mejores_params : dict
    mejor_interno : float
        Mejor puntaje medio de la CV interna (*mayor es mejor*). Es el valor que **no**
        debe reportarse como desempeño: sirve para medir el sesgo de selección.
    n_evaluaciones : int
        Configuraciones distintas realmente evaluadas con la CV interna.
    tiempo_s : float
        Wall-clock de la búsqueda completa, incluido el reajuste final.
    traza : list of float
        Mejor puntaje encontrado hasta la evaluación *t* (curva *anytime*), en orden de
        evaluación. Las evaluaciones fallidas cuentan como ``-inf`` y no rompen la curva.
    estimador : Pipeline
        Reajustado con ``mejores_params`` sobre todo el conjunto recibido.
    detalle : dict
        Información específica del método (surrogate, operadores, diversidad, presupuesto
        efectivo, evaluaciones fallidas…).
    """
    optimizador: str
    mejores_params: dict
    mejor_interno: float
    n_evaluaciones: int
    tiempo_s: float
    traza: list[float]
    estimador: object
    detalle: dict = field(default_factory=dict)

    @property
    def tiempo_por_evaluacion_s(self) -> float:
        """Segundos de búsqueda por evaluación de hiperparámetros."""
        return self.tiempo_s / max(self.n_evaluaciones, 1)


@dataclass
class ResultadoAnidado:
    """Salida de la validación anidada completa para una combinación.

    Attributes
    ----------
    puntajes_ext : list of float
        Métrica principal en cada pliegue externo (*mayor es mejor*; RMSE con signo negativo).
    metricas_ext : dict
        ``nombre_metrica -> [valor por pliegue externo]`` con todas las métricas de la tarea.
    params_por_pliegue : list of dict
        Hiperparámetros ganadores en cada pliegue externo (diagnóstico de estabilidad).
    interno_por_pliegue : list of float
        ``mejor_interno`` de cada pliegue, para cuantificar el optimismo de selección.
    trazas : list of list
        Curva *anytime* de cada pliegue.
    n_evaluaciones : int
        Suma sobre los pliegues externos (sin contar el ajuste final).
    tiempo_total_s : float
        Suma de los tiempos de búsqueda de los pliegues externos.
    detalles : list of dict
        ``detalle`` de la búsqueda en cada pliegue externo.
    params_finales : dict or None
        Hiperparámetros del **modelo final**, elegidos con la misma búsqueda sobre todo el
        conjunto de entrenamiento. ``None`` si no se pidió el ajuste final.
    estimador_final : Pipeline or None
        Modelo final reajustado sobre todo el entrenamiento; es el que se evalúa en prueba.
    interno_final : float or None
        Mejor puntaje interno de la búsqueda final.
    traza_final, detalle_final, n_evaluaciones_final, tiempo_final_s
        Análogos para la búsqueda final.
    """
    puntajes_ext: list[float]
    metricas_ext: dict[str, list[float]]
    params_por_pliegue: list[dict]
    interno_por_pliegue: list[float]
    trazas: list[list[float]]
    n_evaluaciones: int
    tiempo_total_s: float
    detalles: list[dict]
    params_finales: dict | None = None
    estimador_final: object = None
    interno_final: float | None = None
    traza_final: list[float] = field(default_factory=list)
    detalle_final: dict = field(default_factory=dict)
    n_evaluaciones_final: int = 0
    tiempo_final_s: float = 0.0

    @property
    def media_ext(self) -> float:
        """Media de la métrica en los pliegues externos."""
        return float(np.mean(self.puntajes_ext))

    @property
    def desv_ext(self) -> float:
        """Desviación típica muestral de la métrica externa (0 con un solo pliegue)."""
        return float(np.std(self.puntajes_ext, ddof=1)) if len(self.puntajes_ext) > 1 else 0.0

    @property
    def optimismo_seleccion(self) -> float:
        """Media (interno − externo): cuánto sobreestima el bucle interno."""
        return float(np.mean(self.interno_por_pliegue) - self.media_ext)


# -------------------------------------------------------------- utilidades de espacio
class _EnteroLog:
    """Distribución de enteros log-uniforme en ``[bajo, alto]`` para ``RandomizedSearchCV``.

    Parameters
    ----------
    bajo, alto : int
        Extremos inclusivos.
    """

    def __init__(self, bajo: int, alto: int) -> None:
        self.bajo, self.alto = int(bajo), int(alto)

    def rvs(self, size=None, random_state=None):
        """Muestrea enteros con densidad proporcional a ``1/k``.

        Parameters
        ----------
        size : int, optional
            Número de muestras; ``None`` devuelve un solo ``int``.
        random_state : None, int or RandomState
            Se resuelve con ``sklearn.utils.check_random_state``: ``None`` usa el estado
            global de numpy (reproducible tras ``config.fijar_semillas``).

        Returns
        -------
        int or ndarray
        """
        rs = check_random_state(random_state)
        u = rs.uniform(math.log(self.bajo), math.log(self.alto + 1), size=size)
        v = np.floor(np.exp(u)).astype(int)
        if size is None:
            return int(min(max(int(v), self.bajo), self.alto))
        return np.clip(v, self.bajo, self.alto)


def distribuciones_random(espacio: dict[str, tuple]) -> dict:
    """Convierte el espacio tipado en distribuciones para ``RandomizedSearchCV``.

    Parameters
    ----------
    espacio : dict
        Espacio tipado (ver :mod:`modelos`).

    Returns
    -------
    dict
        ``nombre -> distribución scipy / _EnteroLog / lista``.

    Raises
    ------
    ValueError
        Si aparece un tipo desconocido.
    """
    dist = {}
    for nombre, (tipo, *args) in espacio.items():
        if tipo == 'log':
            dist[nombre] = loguniform(args[0], args[1])
        elif tipo == 'float':
            dist[nombre] = uniform(args[0], args[1] - args[0])
        elif tipo == 'int':
            dist[nombre] = randint(args[0], args[1] + 1)
        elif tipo == 'int_log':
            dist[nombre] = _EnteroLog(args[0], args[1])
        elif tipo == 'cat':
            dist[nombre] = list(args[0])
        else:
            raise ValueError(f'tipo de espacio desconocido: {tipo!r}')
    return dist


def sugerir_optuna(trial, espacio: dict[str, tuple]) -> dict:
    """Muestrea una configuración del espacio tipado con un ``trial`` de Optuna.

    Parameters
    ----------
    trial : optuna.Trial
    espacio : dict

    Returns
    -------
    dict
        ``nombre -> valor``, listo para ``pipeline.set_params``.
    """
    params = {}
    for nombre, (tipo, *args) in espacio.items():
        if tipo == 'log':
            params[nombre] = trial.suggest_float(nombre, args[0], args[1], log=True)
        elif tipo == 'float':
            params[nombre] = trial.suggest_float(nombre, args[0], args[1])
        elif tipo == 'int':
            params[nombre] = trial.suggest_int(nombre, args[0], args[1])
        elif tipo == 'int_log':
            params[nombre] = trial.suggest_int(nombre, args[0], args[1], log=True)
        elif tipo == 'cat':
            params[nombre] = trial.suggest_categorical(nombre, list(args[0]))
        else:
            raise ValueError(f'tipo de espacio desconocido: {tipo!r}')
    return params


def decodificar_genoma(espacio: dict[str, tuple], genoma: list[float]) -> dict:
    """Traduce un genoma de valores en [0, 1] a una configuración del espacio tipado.

    Parameters
    ----------
    espacio : dict
    genoma : list of float
        Un gen por hiperparámetro, en el mismo orden que ``espacio``. Se recorta a [0, 1].

    Returns
    -------
    dict
        ``nombre -> valor`` dentro de los rangos declarados (los extremos exactos 0 y 1
        decodifican a los límites del rango).
    """
    params = {}
    for g, (nombre, (tipo, *args)) in zip(genoma, espacio.items()):
        g = min(max(float(g), 0.0), 1.0)
        if tipo == 'log':
            lo, hi = math.log10(args[0]), math.log10(args[1])
            params[nombre] = float(min(max(10 ** (lo + g * (hi - lo)), args[0]), args[1]))
        elif tipo == 'float':
            params[nombre] = float(args[0] + g * (args[1] - args[0]))
        elif tipo == 'int':
            params[nombre] = int(round(args[0] + g * (args[1] - args[0])))
        elif tipo == 'int_log':
            lo, hi = math.log(args[0]), math.log(args[1])
            params[nombre] = int(min(max(round(math.exp(lo + g * (hi - lo))), args[0]), args[1]))
        elif tipo == 'cat':
            opciones = list(args[0])
            params[nombre] = opciones[min(int(g * len(opciones)), len(opciones) - 1)]
        else:
            raise ValueError(f'tipo de espacio desconocido: {tipo!r}')
    return params


def construir_cv(tarea: str, k: int, seed: int = SEED):
    """Validación cruzada barajada de ``k`` pliegues, estratificada solo en clasificación.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    k : int
    seed : int

    Returns
    -------
    StratifiedKFold or KFold
    """
    if tarea == 'clasificacion':
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return KFold(n_splits=k, shuffle=True, random_state=seed)


def presupuesto_por_defecto(espacio_grid: dict) -> int:
    """Presupuesto de evaluaciones cuando no se fija uno explícito.

    Parameters
    ----------
    espacio_grid : dict
        Rejilla explícita del modelo.

    Returns
    -------
    int
        ``max(tamaño de la rejilla, PRESUPUESTO_MINIMO)``: así Random, Bayesiana y
        Genética consumen lo mismo que Grid dentro de cada modelo (guía §3.4).
    """
    return max(len(ParameterGrid(espacio_grid)), PRESUPUESTO_MINIMO)


def _traza_acumulada(valores) -> list[float]:
    """Mejor valor hasta cada posición, ignorando ``NaN`` (evaluaciones fallidas)."""
    v = np.asarray(valores, dtype=float)
    return [float(x) for x in np.fmax.accumulate(v)]


def _finito(s: float) -> float:
    """``-inf`` en lugar de ``NaN`` para que una evaluación fallida no contamine nada."""
    return float(s) if np.isfinite(s) else -np.inf


# ------------------------------------------------------------------------ métodos
def buscar_grid(pipeline, espacio_grid: dict, X, y, cv_int, scoring: str, seed: int = SEED,
                n_jobs: int = -1, **_) -> ResultadoBusqueda:
    """Grid Search exhaustivo sobre la rejilla explícita del modelo.

    Parameters
    ----------
    pipeline : Pipeline
        Sin ajustar; los parámetros se referencian como ``modelo__<param>``.
    espacio_grid : dict
        ``{'modelo__param': [valores]}``.
    X, y
        Conjunto de entrenamiento del pliegue externo (o todo el entrenamiento en el
        ajuste final).
    cv_int
        Validación cruzada interna.
    scoring : str
        *Scorer* de scikit-learn (mayor es mejor).
    seed : int
        No se usa (la rejilla es determinista); se acepta por uniformidad de interfaz.
    n_jobs : int
    **_
        Argumentos ignorados (``presupuesto``), para despachar con la misma firma.

    Returns
    -------
    ResultadoBusqueda
    """
    t0 = time.perf_counter()
    gs = GridSearchCV(pipeline, espacio_grid, cv=cv_int, scoring=scoring, n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    puntajes = gs.cv_results_['mean_test_score']
    n = len(puntajes)
    return ResultadoBusqueda('grid', dict(gs.best_params_), float(gs.best_score_), n,
                             time.perf_counter() - t0, _traza_acumulada(puntajes), gs.best_estimator_,
                             {'presupuesto': n, 'n_fallidas': int(np.isnan(puntajes).sum())})


def buscar_random(pipeline, espacio: dict, X, y, cv_int, scoring: str, presupuesto: int,
                  seed: int = SEED, n_jobs: int = -1, **_) -> ResultadoBusqueda:
    """Random Search con distribuciones continuas derivadas del espacio tipado.

    Parameters
    ----------
    pipeline : Pipeline
    espacio : dict
        Espacio tipado; ver :func:`distribuciones_random`.
    X, y
    cv_int
    scoring : str
    presupuesto : int
        Configuraciones a muestrear (``n_iter``).
    seed : int
    n_jobs : int

    Returns
    -------
    ResultadoBusqueda
        ``n_evaluaciones`` es el número de candidatos realmente evaluados (puede ser
        menor que ``presupuesto`` si el espacio es discreto y pequeño).
    """
    t0 = time.perf_counter()
    rs = RandomizedSearchCV(pipeline, distribuciones_random(espacio), n_iter=presupuesto, cv=cv_int,
                            scoring=scoring, n_jobs=n_jobs, random_state=seed, refit=True)
    rs.fit(X, y)
    puntajes = rs.cv_results_['mean_test_score']
    n = len(puntajes)
    return ResultadoBusqueda('random', dict(rs.best_params_), float(rs.best_score_), n,
                             time.perf_counter() - t0, _traza_acumulada(puntajes), rs.best_estimator_,
                             {'presupuesto': presupuesto, 'muestreo': 'loguniform / uniform / randint / int_log',
                              'n_fallidas': int(np.isnan(puntajes).sum())})


def buscar_bayesiana(pipeline, espacio: dict, X, y, cv_int, scoring: str, presupuesto: int,
                     seed: int = SEED, n_jobs: int = -1, **_) -> ResultadoBusqueda:
    """Optimización bayesiana con Optuna (TPE).

    Parameters
    ----------
    pipeline : Pipeline
    espacio : dict
        Espacio tipado; ver :func:`sugerir_optuna`.
    X, y
    cv_int
    scoring : str
    presupuesto : int
        Número de *trials*.
    seed : int
        Semilla del ``TPESampler``.
    n_jobs : int

    Returns
    -------
    ResultadoBusqueda

    Notes
    -----
    Surrogate: *Tree-structured Parzen Estimator*. En lugar de modelar ``f(x)`` (proceso
    gaussiano), TPE modela dos densidades, ``l(x)`` de las configuraciones buenas y
    ``g(x)`` de las malas, y propone el ``x`` que maximiza ``l(x)/g(x)``, lo que equivale a
    maximizar la **Mejora Esperada** (EI). Escala linealmente con el número de trials y
    admite espacios mixtos (continuos, enteros, categóricos) sin adaptación. Los
    ``n_startup_trials`` iniciales son aleatorios (exploración pura); después, el
    cuantil ``gamma`` que separa buenas de malas (0,10 por defecto) gobierna el balance
    exploración-explotación.

    El pipeline completo (preprocesado + muestreo + modelo) se construye en **cada
    trial** y se evalúa con la CV interna, de modo que el preprocesamiento se ajusta solo
    con las filas de entrenamiento de cada pliegue: Optuna no lo garantiza por sí mismo.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()
    n_inicio = max(1, min(10, presupuesto // 3))
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_inicio)
    estudio = optuna.create_study(direction='maximize', sampler=sampler)

    def objetivo(trial):
        """Función objetivo de Optuna: métrica interna media de la configuración sugerida."""
        params = sugerir_optuna(trial, espacio)
        p = clone(pipeline).set_params(**params)
        return _finito(cross_val_score(p, X, y, cv=cv_int, scoring=scoring, n_jobs=n_jobs).mean())

    estudio.optimize(objetivo, n_trials=presupuesto)
    valores = [t.value if t.value is not None else -np.inf for t in estudio.trials]
    mejor = clone(pipeline).set_params(**estudio.best_params)
    mejor.fit(X, y)
    return ResultadoBusqueda('bayesiana', dict(estudio.best_params), float(estudio.best_value), len(valores),
                             time.perf_counter() - t0, _traza_acumulada(valores), mejor,
                             {'presupuesto': presupuesto, 'surrogate': 'TPE',
                              'adquisicion': 'EI (implícita: argmax l(x)/g(x))',
                              'n_startup_trials': n_inicio, 'gamma': 0.10,
                              'historial': [float(v) for v in valores],
                              'n_fallidas': int(sum(1 for v in valores if not np.isfinite(v)))})


def _diversidad(poblacion) -> float:
    """Desviación típica media de los genes (en [0, 1]) a través de la población.

    Parameters
    ----------
    poblacion : list of list

    Returns
    -------
    float
        0 cuando todos los individuos son idénticos.
    """
    if len(poblacion) < 2:
        return 0.0
    return float(np.mean(np.std(np.asarray(poblacion, dtype=float), axis=0)))


def buscar_genetica(pipeline, espacio: dict, X, y, cv_int, scoring: str, presupuesto: int,
                    seed: int = SEED, n_jobs: int = -1, **_) -> ResultadoBusqueda:
    """Algoritmo genético con DEAP, gobernado por el presupuesto de evaluaciones.

    Parameters
    ----------
    pipeline : Pipeline
    espacio : dict
        Espacio tipado; ver :func:`decodificar_genoma`.
    X, y
    cv_int
    scoring : str
    presupuesto : int
        Configuraciones distintas a evaluar. El bucle evolutivo continúa hasta
        consumirlas exactamente (o hasta que cinco generaciones seguidas no aporten
        genomas nuevos).
    seed : int
        Siembra el ``random`` global que usan los operadores de DEAP.
    n_jobs : int

    Returns
    -------
    ResultadoBusqueda
        ``detalle`` incluye los operadores, la población, las generaciones realizadas y
        las series por generación de diversidad, mejor y media del *fitness*.

    Notes
    -----
    - **Representación**: un individuo es un vector de genes en [0, 1], uno por
      hiperparámetro, que :func:`decodificar_genoma` traduce al espacio tipado (escala
      logarítmica para C, alpha, learning rate…).
    - **Selección**: torneo de tamaño 3. La presión selectiva se controla con el tamaño
      del torneo, es invariante a la escala del *fitness* y evita la dominancia prematura
      de la ruleta cuando las diferencias de AUC son de milésimas.
    - **Cruce**: uniforme por gen (``cxUniform``, ``indpb = 0,5``) con probabilidad 0,6.
    - **Mutación**: gaussiana por gen (``sigma = 0,15`` en la escala [0, 1], ``indpb =
      1/n_genes``) con probabilidad 0,3; los genes se recortan a [0, 1] tras mutar. Un
      hijo idéntico a un genoma ya evaluado se vuelve a mutar (hasta tres veces) para que
      el presupuesto se gaste en configuraciones nuevas.
    - **Elitismo**: el mejor individuo histórico (``HallOfFame(1)``) se reinserta en cada
      generación.
    - **Población**: ``max(4, min(presupuesto // 4, 12))`` individuos, de modo que con
      30-48 evaluaciones haya al menos 3-5 generaciones y la evolución de la diversidad
      sea observable. Las evaluaciones se memorizan por genoma: un individuo repetido no
      consume presupuesto.
    - **Diversidad**: en cada generación se registra la desviación típica media de los
      genes; su colapso antes de que converja el *fitness* señala convergencia prematura.
    - Cada evaluación construye el pipeline completo y lo valida con la CV interna; una
      evaluación fallida vale ``-inf`` y consume presupuesto.
    """
    from deap import base, creator, tools

    random.seed(seed)
    t0 = time.perf_counter()
    genes = list(espacio.keys())
    n_genes = len(genes)
    pob_n = min(max(4, min(presupuesto // 4, 12)), presupuesto)
    CXPB, MUTPB = 0.6, 0.3

    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individuo'):
        creator.create('Individuo', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('individuo', tools.initIterate, creator.Individuo,
                     lambda: [random.random() for _ in range(n_genes)])
    toolbox.register('poblacion', tools.initRepeat, list, toolbox.individuo)
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutGaussian, mu=0.0, sigma=0.15, indpb=1.0 / n_genes)
    toolbox.register('select', tools.selTournament, tournsize=3)

    cache: dict[tuple, float] = {}
    traza: list[float] = []
    n_fallidas = 0

    def clave(ind) -> tuple:
        """Genoma redondeado a 6 decimales, que sirve de clave en la caché de evaluaciones."""
        return tuple(round(float(g), 6) for g in ind)

    def evaluar(ind):
        """Aptitud DEAP del individuo: métrica interna media, con caché por genoma y recuento de fallos."""
        nonlocal n_fallidas
        k = clave(ind)
        if k not in cache:
            p = clone(pipeline).set_params(**decodificar_genoma(espacio, ind))
            s = _finito(cross_val_score(p, X, y, cv=cv_int, scoring=scoring, n_jobs=n_jobs).mean())
            if not np.isfinite(s):
                n_fallidas += 1
            cache[k] = s
            traza.append(max(traza[-1], s) if traza else s)
        return (cache[k],)

    def mutar(ind) -> None:
        """Mutación gaussiana con los genes recortados al intervalo [0, 1]."""
        toolbox.mutate(ind)
        ind[:] = [min(max(g, 0.0), 1.0) for g in ind]
        del ind.fitness.values

    poblacion = toolbox.poblacion(n=pob_n)
    for ind in poblacion:
        ind.fitness.values = evaluar(ind)
    hof = tools.HallOfFame(1)
    hof.update(poblacion)
    diversidad = [_diversidad(poblacion)]
    mejor_gen = [hof[0].fitness.values[0]]
    media_gen = [float(np.mean([i.fitness.values[0] for i in poblacion]))]

    generaciones, sin_novedad = 0, 0
    while len(cache) < presupuesto and sin_novedad < 5:
        generaciones += 1
        antes = len(cache)
        descendencia = [toolbox.clone(i) for i in toolbox.select(poblacion, max(len(poblacion) - 1, 1))]
        for h1, h2 in zip(descendencia[::2], descendencia[1::2]):
            if random.random() < CXPB:
                toolbox.mate(h1, h2)
                del h1.fitness.values, h2.fitness.values
        for m in descendencia:
            if random.random() < MUTPB:
                mutar(m)
            intentos = 0
            while not m.fitness.valid and clave(m) in cache and intentos < 3:
                mutar(m)
                intentos += 1
        nueva = []
        for ind in descendencia:
            if ind.fitness.valid:
                nueva.append(ind)                       # clon sin cambios: conserva su fitness
            elif clave(ind) in cache or len(cache) < presupuesto:
                ind.fitness.values = evaluar(ind)
                nueva.append(ind)
            # si el presupuesto se agotó y el genoma es nuevo, el hijo se descarta
        poblacion = nueva + [toolbox.clone(hof[0])]
        hof.update(poblacion)
        diversidad.append(_diversidad(poblacion))
        mejor_gen.append(hof[0].fitness.values[0])
        media_gen.append(float(np.mean([i.fitness.values[0] for i in poblacion])))
        sin_novedad = sin_novedad + 1 if len(cache) == antes else 0

    mejores = decodificar_genoma(espacio, hof[0])
    mejor = clone(pipeline).set_params(**mejores)
    mejor.fit(X, y)
    return ResultadoBusqueda('genetica', mejores, float(hof[0].fitness.values[0]), len(cache),
                             time.perf_counter() - t0, traza, mejor,
                             {'presupuesto': presupuesto, 'seleccion': 'torneo (tournsize=3)',
                              'cruce': f'uniforme indpb=0.5, CXPB={CXPB}',
                              'mutacion': f'gaussiana sigma=0.15 indpb=1/{n_genes}, MUTPB={MUTPB}',
                              'elitismo': 'HallOfFame(1) reinsertado cada generación',
                              'poblacion': pob_n, 'generaciones': generaciones,
                              'diversidad_por_generacion': diversidad,
                              'mejor_por_generacion': [float(v) for v in mejor_gen],
                              'media_por_generacion': media_gen, 'n_fallidas': n_fallidas})


BUSCADORES = {'grid': buscar_grid, 'random': buscar_random,
              'bayesiana': buscar_bayesiana, 'genetica': buscar_genetica}


def _reajustar_alta_fidelidad(res: ResultadoBusqueda, pipeline, X, y, fijos_busqueda: dict | None,
                              fijos_final: dict) -> ResultadoBusqueda:
    """Reajusta el ganador de una búsqueda de baja fidelidad con los parámetros finales.

    Parameters
    ----------
    res : ResultadoBusqueda
        Resultado de la búsqueda hecha con ``fijos_busqueda``.
    pipeline : Pipeline
        El pipeline de baja fidelidad con el que se buscó (sin ajustar).
    X, y
        Los mismos datos de la búsqueda.
    fijos_busqueda, fijos_final : dict
        Parámetros de baja y de alta fidelidad.

    Returns
    -------
    ResultadoBusqueda
        El mismo objeto con ``estimador`` reajustado a alta fidelidad, ``mejores_params`` con los
        valores finales, el tiempo del reajuste sumado y la fidelidad anotada en ``detalle``. La
        traza, el número de evaluaciones y ``mejor_interno`` siguen siendo los de la búsqueda.
    """
    t0 = time.perf_counter()
    params = {**res.mejores_params, **fijos_final}
    final = clone(pipeline).set_params(**params)
    final.fit(X, y)
    res.estimador = final
    res.mejores_params = params
    res.tiempo_s += time.perf_counter() - t0
    res.detalle = {**res.detalle, 'fidelidad': {'busqueda': fijos_busqueda or {}, 'final': fijos_final}}
    return res


def buscar(optimizador: str, pipeline, espacio_grid: dict, espacio: dict, X, y, cv_int,
           scoring: str, presupuesto: int | None = None, seed: int = SEED,
           n_jobs: int = -1, fijos_busqueda: dict | None = None,
           fijos_final: dict | None = None) -> ResultadoBusqueda:
    """Despacha al método pedido con un presupuesto comparable.

    Parameters
    ----------
    optimizador : {'grid', 'random', 'bayesiana', 'genetica'}
    pipeline : Pipeline
        Sin ajustar.
    espacio_grid : dict
        Rejilla explícita (la usa Grid).
    espacio : dict
        Espacio tipado (lo usan los otros tres).
    X, y
        Conjunto sobre el que se busca (pliegue de entrenamiento externo, o todo el
        entrenamiento en el ajuste final).
    cv_int
        Validación cruzada interna.
    scoring : str
    presupuesto : int, optional
        Configuraciones a evaluar. Por defecto :func:`presupuesto_por_defecto`. Grid es
        exhaustivo y avisa si su rejilla supera el presupuesto pedido.
    seed : int
    n_jobs : int
    fijos_busqueda : dict, optional
        Parámetros fijos durante la búsqueda (baja fidelidad), p. ej. ``{'modelo__n_estimators': 100}``.
    fijos_final : dict, optional
        Parámetros con los que se reajusta el ganador (alta fidelidad). Ver
        :func:`_reajustar_alta_fidelidad`.

    Returns
    -------
    ResultadoBusqueda

    Raises
    ------
    ValueError
        Si ``optimizador`` no está en ``OPTIMIZADORES``.
    """
    if optimizador not in BUSCADORES:
        raise ValueError(f'optimizador desconocido: {optimizador!r}; use uno de {OPTIMIZADORES}')
    if presupuesto is None:
        presupuesto = presupuesto_por_defecto(espacio_grid)
    if fijos_busqueda:
        pipeline = clone(pipeline).set_params(**fijos_busqueda)
    if optimizador == 'grid':
        n_rejilla = len(ParameterGrid(espacio_grid))
        if n_rejilla > presupuesto:
            warnings.warn(f'la rejilla tiene {n_rejilla} configuraciones, más que el presupuesto {presupuesto}; '
                          f'Grid es exhaustivo y evaluará las {n_rejilla}')
        res = buscar_grid(pipeline, espacio_grid, X, y, cv_int, scoring, seed=seed, n_jobs=n_jobs)
    else:
        res = BUSCADORES[optimizador](pipeline, espacio, X, y, cv_int, scoring, presupuesto=presupuesto,
                                      seed=seed, n_jobs=n_jobs)
    if fijos_final:
        res = _reajustar_alta_fidelidad(res, pipeline, X, y, fijos_busqueda, fijos_final)
    return res


# ---------------------------------------------------------------- validación anidada
def _alinear(X: pd.DataFrame, y) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve ``X`` e ``y`` con índice posicional y la misma correspondencia de filas.

    Parameters
    ----------
    X : DataFrame
    y : Series or array-like
        Si es una ``Series`` con el mismo conjunto de etiquetas de índice que ``X`` pero en
        otro orden, se realinea por etiqueta. Un arreglo se interpreta por posición.

    Returns
    -------
    X, y

    Raises
    ------
    ValueError
        Si ``y`` es una ``Series`` cuyo índice no coincide con el de ``X`` ni puede
        realinearse sin ambigüedad, o si las longitudes difieren.
    """
    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        if not (X.index.is_unique and y.index.is_unique and len(y) == len(X) and X.index.isin(y.index).all()):
            raise ValueError('y debe llevar el mismo índice que X (o pasarse como arreglo posicional)')
        y = y.loc[X.index]
    if len(y) != len(X):
        raise ValueError(f'X tiene {len(X)} filas e y {len(y)}')
    return X.reset_index(drop=True), pd.Series(np.asarray(y)).reset_index(drop=True)


def validacion_anidada(constructor, espacio_grid: dict, espacio: dict, optimizador: str,
                       X: pd.DataFrame, y, tarea: str, k_ext: int = K_EXT, k_int: int = K_INT,
                       presupuesto: int | None = None, seed: int = SEED, n_jobs: int = -1,
                       scoring: str | None = None, ajuste_final: bool = True,
                       fijos_busqueda: dict | None = None,
                       fijos_final: dict | None = None) -> ResultadoAnidado:
    """Validación cruzada anidada: búsqueda completa dentro de cada pliegue externo.

    Parameters
    ----------
    constructor : callable
        ``y_pliegue -> Pipeline`` sin ajustar. Recibe la etiqueta del pliegue de
        entrenamiento externo para poder fijar parámetros que dependen de ella
        (``scale_pos_weight``).
    espacio_grid, espacio : dict
        Rejilla explícita y espacio tipado del modelo.
    optimizador : str
    X : DataFrame
        Predictores del conjunto de **entrenamiento** completo (el de prueba no entra aquí).
    y : Series or array-like
        Objetivo alineado con ``X`` (ver :func:`_alinear`).
    tarea : {'clasificacion', 'regresion'}
    k_ext, k_int : int
        Pliegues externos e internos.
    presupuesto : int, optional
        Ver :func:`buscar`.
    seed : int
    n_jobs : int
    scoring : str, optional
        *Scorer* de selección; por defecto el de ``evaluacion.METRICA_PRINCIPAL``.
    ajuste_final : bool
        Si ``True``, tras el bucle externo repite la búsqueda sobre todo ``(X, y)`` para
        obtener el modelo final y sus hiperparámetros finales.
    fijos_busqueda, fijos_final : dict, optional
        Multi-fidelidad; ver :func:`buscar`. El modelo evaluado en cada pliegue externo es el de
        alta fidelidad.

    Returns
    -------
    ResultadoAnidado

    Notes
    -----
    La métrica de cada pliegue externo se calcula con el pipeline **reajustado sobre todo
    el pliegue de entrenamiento externo** con los hiperparámetros ganadores del bucle
    interno, y evaluado en el pliegue de prueba externo, que no intervino en la búsqueda.
    El umbral de la etiqueta es parte de la definición del problema: se estimó una sola
    vez con todo el entrenamiento y dentro de la validación se trata como constante.
    """
    scoring = scoring or evaluacion.METRICA_PRINCIPAL[tarea]
    cv_ext = construir_cv(tarea, k_ext, seed)
    cv_int = construir_cv(tarea, k_int, seed)
    X, y = _alinear(X, y)
    puntajes, internos, params, trazas, detalles = [], [], [], [], []
    metricas: dict[str, list[float]] = {}
    n_eval, t_total = 0, 0.0
    for i_tr, i_te in cv_ext.split(X, y):
        X_tr, X_te, y_tr, y_te = X.iloc[i_tr], X.iloc[i_te], y.iloc[i_tr], y.iloc[i_te]
        res = buscar(optimizador, constructor(y_tr), espacio_grid, espacio, X_tr, y_tr, cv_int,
                     scoring, presupuesto=presupuesto, seed=seed, n_jobs=n_jobs,
                     fijos_busqueda=fijos_busqueda, fijos_final=fijos_final)
        m = evaluacion.evaluar(res.estimador, X_te, y_te, tarea)
        for k, v in m.items():
            metricas.setdefault(k, []).append(v)
        puntajes.append(evaluacion.principal(m, tarea))
        internos.append(res.mejor_interno)
        params.append(res.mejores_params)
        trazas.append(res.traza)
        detalles.append(res.detalle)
        n_eval += res.n_evaluaciones
        t_total += res.tiempo_s
    salida = ResultadoAnidado(puntajes, metricas, params, internos, trazas, n_eval, t_total, detalles)
    if ajuste_final:
        fin = buscar(optimizador, constructor(y), espacio_grid, espacio, X, y, cv_int, scoring,
                     presupuesto=presupuesto, seed=seed, n_jobs=n_jobs,
                     fijos_busqueda=fijos_busqueda, fijos_final=fijos_final)
        salida.params_finales = fin.mejores_params
        salida.estimador_final = fin.estimador
        salida.interno_final = fin.mejor_interno
        salida.traza_final = fin.traza
        salida.detalle_final = fin.detalle
        salida.n_evaluaciones_final = fin.n_evaluaciones
        salida.tiempo_final_s = fin.tiempo_s
    return salida
