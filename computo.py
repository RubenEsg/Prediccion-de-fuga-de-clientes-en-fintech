"""Experimentos de optimización computacional (guía §4).

Mide, con los hiperparámetros finales de la mejor combinación de cada modelo en la tabla maestra:

- ``escalamiento``: tiempo de entrenamiento e inferencia frente a n, para contrastar la
  complejidad teórica con la pendiente log-log medida, y memoria pico al entrenar con todo.
- ``knn``: fuerza bruta, KD-Tree, Ball Tree y FAISS (exacto e IVF aproximado), en 78 dimensiones y
  con las 4 variables con señal.
- ``lineales``: Ridge con cholesky, svd, lsqr, sparse_cg, sag y saga; Lasso por descenso por
  coordenadas frente a SGD con penalización L1; logística L1 con liblinear frente a SAGA.
- ``naive_bayes``: ``fit`` frente a ``partial_fit`` por lotes (memoria y equivalencia).
- ``xgboost``: ``exact``, ``approx`` y ``hist`` en CPU, ``hist`` en GPU, hilos y *early stopping*.
- ``svm``: SVC con núcleo RBF sobre submuestras frente a LinearSVC y SGD, con extrapolación a n.
- ``paralelismo``: Random Forest con 1 a 6 hilos y búsqueda en rejilla con los *backends* loky,
  threading y multiprocessing de joblib.
- ``entre_modelos``: cuatro búsquedas en rejilla de modelos distintos, paralelizando dentro de cada
  búsqueda, entre búsquedas o de las dos formas a la vez.
- ``perfil``: cProfile de una validación cruzada del pipeline completo.
- ``fidelidad``: comprobación de la multi-fidelidad (Random Forest 100 frente a 300 árboles; EBM 2
  frente a 8 bolsas): ¿se conserva el orden de las configuraciones?

Los tiempos son de reloj (``time.perf_counter``), mediana de varias repeticiones, en el equipo de
referencia (6 núcleos lógicos, RTX 3050). Para que sean válidos el equipo no debe estar corriendo
otros experimentos: con ``--esperar`` el guion espera a que terminen las corridas del modelo nuevo.

Uso::

    .venv\\Scripts\\python.exe -u computo.py                  (todos los experimentos pendientes)
    .venv\\Scripts\\python.exe -u computo.py knn xgboost      (solo esos)
    .venv\\Scripts\\python.exe -u computo.py --esperar        (espera a que el PC quede libre)

Salidas en ``resultados/computo/``: un CSV por experimento, ``perfil.txt`` y ``progreso.log``.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pickle
import pstats
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / 'resultados' / 'computo'
PRUEBA = False
EXPERIMENTOS = ['escalamiento', 'knn', 'knn_euclidea', 'lineales', 'naive_bayes', 'xgboost', 'svm', 'paralelismo',
                'entre_modelos', 'perfil', 'fidelidad']
VARIABLES_CON_SENAL = ['app_logins_frequency', 'age', 'satisfaction_score', 'base_satisfaction']
"""Las cuatro variables que concentran la señal (benchmark del 29-sep, exploracion/GUIA_MODELO_NUEVO.md)."""

log_archivo = None


def log(msg: str) -> None:
    linea = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}'
    print(linea, flush=True)
    if log_archivo is not None:
        log_archivo.write(linea + '\n')
        log_archivo.flush()


def cronometrar(f, repeticiones: int = 3) -> tuple[float, object]:
    """Mediana del tiempo de reloj de ``f()`` y el resultado de la última llamada."""
    tiempos, r = [], None
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        r = f()
        tiempos.append(time.perf_counter() - t0)
    return float(np.median(tiempos)), r


def memoria_pico_mb(f) -> float:
    """Memoria residente máxima (MB) del proceso mientras se ejecuta ``f()``, por encima de la base."""
    from memory_profiler import memory_usage
    base = memory_usage(-1, interval=0.05, timeout=0.2)
    pico = memory_usage((f, (), {}), interval=0.05, max_usage=True)
    pico = max(pico) if isinstance(pico, (list, tuple)) else pico
    return float(pico - min(base))


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def puntaje(modelo, X):
    return modelo.predict_proba(X)[:, 1] if hasattr(modelo, 'predict_proba') else modelo.decision_function(X)


# ------------------------------------------------------------------------- contexto
def preparar() -> dict:
    """Datos preprocesados (ajuste solo con entrenamiento) e hiperparámetros finales por modelo."""
    from src import analisis, datos, modelos
    from src.config import SEED
    from src.preprocesamiento import construir_preprocesador, nombres_variables

    p = datos.preparar_todo()
    c = p['conjuntos']
    if PRUEBA:                                   # ensayo: submuestras pequeñas para validar el código
        c.X_train, c.y_train_cont = c.X_train.iloc[:3000], c.y_train_cont.iloc[:3000]
        c.X_test, c.y_test_cont = c.X_test.iloc[:1500], c.y_test_cont.iloc[:1500]
        p['y_train'], p['y_test'] = p['y_train'].iloc[:3000], p['y_test'].iloc[:1500]
    pre = construir_preprocesador(c.X_train).fit(c.X_train)
    nombres = nombres_variables(pre)
    tabla = analisis.cargar_tabla(RAIZ / 'resultados' / 'experimentos.parquet')
    hp = {}
    for tarea in ('clasificacion', 'regresion'):
        for _, f in analisis.mejores_por_modelo(tabla, tarea).iterrows():
            hp[(tarea, f['modelo'])] = {k.split('__', 1)[1]: v for k, v in f['hiperparametros'].items()}
    return {'Xtr': pre.transform(c.X_train), 'Xte': pre.transform(c.X_test),
            'ytr': p['y_train'].to_numpy(), 'yte': p['y_test'].to_numpy(),
            'ytr_c': c.y_train_cont.to_numpy(), 'yte_c': c.y_test_cont.to_numpy(),
            'X_crudo': c.X_train, 'y_crudo': p['y_train'], 'nombres': nombres,
            'senal': [nombres.index(v) for v in VARIABLES_CON_SENAL], 'hp': hp,
            'modelos': modelos, 'seed': SEED}


def estimador(ctx, tarea: str, nombre: str, **cambios):
    """Estimador del catálogo con los hiperparámetros finales de su mejor combinación."""
    spec = ctx['modelos'].catalogo(tarea, incluir_nuevo=True)[nombre]
    est = spec.construir(ctx['seed'])
    est.set_params(**{**ctx['hp'].get((tarea, nombre), {}), **cambios})
    return est


# ---------------------------------------------------------------------- experimentos
def exp_escalamiento(ctx) -> pd.DataFrame:
    """Tiempo frente a n para los 7 clasificadores y los 3 regresores lineales."""
    from sklearn.base import clone
    tamanos = [500, 1000, 2000, len(ctx['ytr'])] if PRUEBA else [2500, 5000, 10000, 20000, len(ctx['ytr'])]
    rng = np.random.default_rng(ctx['seed'])
    orden = rng.permutation(len(ctx['ytr']))
    casos = [('clasificacion', m) for m in ('naive_bayes', 'logistica', 'svm', 'arbol', 'knn', 'xgboost',
                                             'random_forest')]
    casos += [('regresion', m) for m in ('ridge', 'lasso', 'svr')]
    filas = []
    for tarea, nombre in casos:
        base = estimador(ctx, tarea, nombre)
        y = ctx['ytr'] if tarea == 'clasificacion' else ctx['ytr_c']
        for n in tamanos:
            if nombre == 'knn' and n <= base.get_params()['n_neighbors']:
                continue                            # menos filas que vecinos: no hay consulta posible
            idx = orden[:n]
            reps = 1 if (nombre == 'random_forest' and n > 10000) else 3
            t_fit, m = cronometrar(lambda: clone(base).fit(ctx['Xtr'][idx], y[idx]), reps)
            t_pred, s = cronometrar(lambda: puntaje(m, ctx['Xte']) if tarea == 'clasificacion'
                                    else m.predict(ctx['Xte']), reps)
            fila = {'tarea': tarea, 'modelo': nombre, 'n': n, 'entrenamiento_s': t_fit, 'inferencia_s': t_pred}
            if tarea == 'clasificacion':
                fila['auc'] = auc(ctx['yte'], s)
            if n == tamanos[-1]:
                fila['memoria_mb'] = memoria_pico_mb(lambda: clone(base).fit(ctx['Xtr'], y))
                fila['tamano_modelo_mb'] = len(pickle.dumps(m)) / 2 ** 20
            filas.append(fila)
            log(f'  escalamiento {nombre} n={n}: {t_fit:.3f} s + {t_pred:.3f} s')
    return pd.DataFrame(filas)


def exp_knn(ctx) -> pd.DataFrame:
    """Búsqueda de vecinos: árboles de scikit-learn frente a FAISS, en 78 y en 4 dimensiones."""
    import faiss
    from sklearn.neighbors import KNeighborsClassifier
    hp = ctx['hp'][('clasificacion', 'knn')]
    k, pesos, p = int(hp.get('n_neighbors', 150)), hp.get('weights', 'uniform'), int(hp.get('p', 2))
    filas = []
    for etiqueta, cols in (('78 variables', slice(None)), ('4 variables con señal', ctx['senal'])):
        Xtr = np.ascontiguousarray(ctx['Xtr'][:, cols], dtype=np.float32)
        Xte = np.ascontiguousarray(ctx['Xte'][:, cols], dtype=np.float32)
        for algoritmo in ('brute', 'kd_tree', 'ball_tree'):
            m = KNeighborsClassifier(n_neighbors=k, weights=pesos, p=p, algorithm=algoritmo, n_jobs=-1)
            t_fit, m = cronometrar(lambda: m.fit(Xtr, ctx['ytr']), 3)
            t_pred, s = cronometrar(lambda: m.predict_proba(Xte)[:, 1], 1 if algoritmo != 'brute' else 2)
            filas.append({'dimension': etiqueta, 'metodo': f'scikit-learn {algoritmo}', 'exacto': True,
                          'construccion_s': t_fit, 'consulta_s': t_pred, 'auc': auc(ctx['yte'], s),
                          'indice_mb': len(pickle.dumps(m)) / 2 ** 20})
            log(f'  knn {etiqueta} {algoritmo}: {t_fit:.2f} s + {t_pred:.2f} s')
        metrica = faiss.METRIC_L2 if p == 2 else faiss.METRIC_L1
        d = Xtr.shape[1]
        indices = {'FAISS IndexFlat (exacto)': lambda: faiss.IndexFlat(d, metrica)}
        nlist = 16 if PRUEBA else 128
        if metrica == faiss.METRIC_L2:
            indices['FAISS IVF (aproximado, nprobe=8)'] = lambda: faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
        indices['FAISS HNSW (aproximado, M=32)'] = lambda: faiss.IndexHNSWFlat(d, 32, metrica)
        for nombre, crear in indices.items():
            def construir():
                ix = crear()
                if not ix.is_trained:
                    ix.train(Xtr)
                ix.add(Xtr)
                if hasattr(ix, 'nprobe'):
                    ix.nprobe = 8
                if hasattr(ix, 'hnsw'):
                    ix.hnsw.efSearch = 2 * k               # debe ser ≥ k para devolver k vecinos
                return ix
            t_fit, ix = cronometrar(construir, 3)
            t_pred, (D, I) = cronometrar(lambda: ix.search(Xte, k), 2)
            vecinos = ctx['ytr'][I]
            if pesos == 'distance':
                w = 1.0 / np.maximum(np.sqrt(np.maximum(D, 0)) if p == 2 else D, 1e-12)
                s = (vecinos * w).sum(axis=1) / w.sum(axis=1)
            else:
                s = vecinos.mean(axis=1)
            filas.append({'dimension': etiqueta, 'metodo': nombre, 'exacto': 'exacto' in nombre,
                          'construccion_s': t_fit, 'consulta_s': t_pred, 'auc': auc(ctx['yte'], s),
                          'indice_mb': faiss.serialize_index(ix).nbytes / 2 ** 20})
            log(f'  knn {etiqueta} {nombre}: {t_fit:.2f} s + {t_pred:.2f} s')
    return pd.DataFrame(filas).assign(k=k, pesos=pesos, p=p)


def exp_knn_euclidea(ctx) -> pd.DataFrame:
    """FAISS donde rinde: distancia euclídea y pocos vecinos, con la exhaustividad del índice aproximado.

    Para k = 10 y k = 100 vecinos en las 78 dimensiones: fuerza bruta de scikit-learn, índice exacto de
    FAISS y HNSW. ``exhaustividad`` es la fracción de los k vecinos exactos que recupera cada método.
    """
    import faiss
    from sklearn.neighbors import NearestNeighbors
    Xtr = np.ascontiguousarray(ctx['Xtr'], dtype=np.float32)
    Xte = np.ascontiguousarray(ctx['Xte'], dtype=np.float32)
    d = Xtr.shape[1]
    filas = []
    for k in (10, 100):
        nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=-1).fit(Xtr)
        t_sk, (_, exactos) = cronometrar(lambda: nn.kneighbors(Xte), 2)
        filas.append({'k': k, 'metodo': 'scikit-learn brute', 'construccion_s': 0.0, 'consulta_s': t_sk,
                      'exhaustividad': 1.0})
        for nombre, crear in (('FAISS IndexFlatL2 (exacto)', lambda: faiss.IndexFlatL2(d)),
                              ('FAISS HNSW (aproximado, M=32)', lambda: faiss.IndexHNSWFlat(d, 32))):
            def construir():
                ix = crear()
                ix.add(Xtr)
                if hasattr(ix, 'hnsw'):
                    ix.hnsw.efSearch = max(2 * k, 64)
                return ix
            t_c, ix = cronometrar(construir, 2)
            t_q, (_, vecinos) = cronometrar(lambda: ix.search(Xte, k), 2)
            exhaustividad = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(vecinos, exactos)]))
            filas.append({'k': k, 'metodo': nombre, 'construccion_s': t_c, 'consulta_s': t_q,
                          'exhaustividad': exhaustividad})
            log(f'  knn euclídea k={k} {nombre}: {t_c:.2f} s + {t_q:.2f} s · exhaustividad {exhaustividad:.3f}')
    return pd.DataFrame(filas)


def exp_lineales(ctx) -> pd.DataFrame:
    """Solucionadores de Ridge, Lasso y logística L1: tiempo, iteraciones y métrica."""
    from sklearn.linear_model import Lasso, LogisticRegression, Ridge, SGDRegressor
    from sklearn.metrics import root_mean_squared_error
    filas = []
    a_ridge = ctx['hp'][('regresion', 'ridge')]['alpha']
    for solver in ('cholesky', 'svd', 'lsqr', 'sparse_cg', 'sag', 'saga'):
        m = Ridge(alpha=a_ridge, solver=solver, random_state=ctx['seed'], max_iter=10000, tol=1e-6)
        t_fit, m = cronometrar(lambda: m.fit(ctx['Xtr'], ctx['ytr_c']), 3)
        filas.append({'modelo': 'Ridge', 'solucionador': solver, 'entrenamiento_s': t_fit,
                      'iteraciones': None if m.n_iter_ is None else int(np.max(m.n_iter_)),
                      'metrica': 'rmse', 'valor': root_mean_squared_error(ctx['yte_c'], m.predict(ctx['Xte']))})
        log(f'  Ridge {solver}: {t_fit:.3f} s')
    a_lasso = ctx['hp'][('regresion', 'lasso')]['alpha']
    for nombre, m in (('descenso por coordenadas', Lasso(alpha=a_lasso, max_iter=20000, random_state=ctx['seed'])),
                      ('SGD con penalización L1', SGDRegressor(penalty='l1', alpha=a_lasso, max_iter=2000, tol=1e-6,
                                                               random_state=ctx['seed']))):
        t_fit, m = cronometrar(lambda: m.fit(ctx['Xtr'], ctx['ytr_c']), 3)
        filas.append({'modelo': 'Lasso', 'solucionador': nombre, 'entrenamiento_s': t_fit,
                      'iteraciones': int(np.max(m.n_iter_)), 'metrica': 'rmse',
                      'valor': root_mean_squared_error(ctx['yte_c'], m.predict(ctx['Xte'])),
                      'coeficientes_no_nulos': int(np.sum(np.abs(m.coef_) > 1e-10))})
        log(f'  Lasso {nombre}: {t_fit:.3f} s')
    hp = ctx['hp'][('clasificacion', 'logistica')]
    for solver in ('liblinear', 'saga'):
        m = LogisticRegression(C=hp['C'], l1_ratio=1.0, solver=solver, max_iter=5000, tol=1e-4,
                               random_state=ctx['seed'])
        t_fit, m = cronometrar(lambda: m.fit(ctx['Xtr'], ctx['ytr']), 3)
        filas.append({'modelo': 'Logística L1', 'solucionador': solver, 'entrenamiento_s': t_fit,
                      'iteraciones': int(np.max(m.n_iter_)), 'metrica': 'auc',
                      'valor': auc(ctx['yte'], m.predict_proba(ctx['Xte'])[:, 1]),
                      'coeficientes_no_nulos': int(np.sum(m.coef_ != 0))})
        log(f'  logística {solver}: {t_fit:.3f} s')
    return pd.DataFrame(filas)


def exp_naive_bayes(ctx) -> pd.DataFrame:
    """GaussianNB: ajuste completo frente a ``partial_fit`` en lotes, con la misma solución."""
    from sklearn.naive_bayes import GaussianNB
    vs = ctx['hp'][('clasificacion', 'naive_bayes')].get('var_smoothing', 1e-9)
    filas, ref = [], None
    for lote in (None, 20000, 5000, 1000):
        def ajustar():
            m = GaussianNB(var_smoothing=vs)
            if lote is None:
                return m.fit(ctx['Xtr'], ctx['ytr'])
            for i in range(0, len(ctx['ytr']), lote):
                m.partial_fit(ctx['Xtr'][i:i + lote], ctx['ytr'][i:i + lote], classes=[0, 1])
            return m
        t_fit, m = cronometrar(ajustar, 5)
        s = m.predict_proba(ctx['Xte'])[:, 1]
        ref = s if ref is None else ref
        filas.append({'modo': 'fit completo' if lote is None else f'partial_fit, lotes de {lote}',
                      'entrenamiento_s': t_fit, 'memoria_mb': memoria_pico_mb(ajustar),
                      'auc': auc(ctx['yte'], s), 'max_dif_probabilidad': float(np.max(np.abs(s - ref)))})
    return pd.DataFrame(filas)


def exp_xgboost(ctx) -> pd.DataFrame:
    """Métodos de construcción de árboles, CPU frente a GPU, hilos y *early stopping*."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    base = {k: v for k, v in ctx['hp'][('clasificacion', 'xgboost')].items()}
    base.update({'random_state': ctx['seed'], 'verbosity': 0, 'eval_metric': 'logloss'})
    filas = []
    variantes = [('exact', 'cpu', -1), ('approx', 'cpu', -1), ('hist', 'cpu', -1), ('hist', 'cpu', 1),
                 ('hist', 'cuda', -1)]
    for metodo, disp, hilos in variantes:
        m = XGBClassifier(**{**base, 'tree_method': metodo, 'device': disp, 'n_jobs': hilos})
        t_fit, m = cronometrar(lambda: m.fit(ctx['Xtr'], ctx['ytr']), 3)
        t_pred, s = cronometrar(lambda: m.predict_proba(ctx['Xte'])[:, 1], 3)
        filas.append({'experimento': 'método y dispositivo', 'variante': f'{metodo} · {disp} · '
                      f'{"todos los hilos" if hilos == -1 else "1 hilo"}', 'arboles': m.n_estimators,
                      'entrenamiento_s': t_fit, 'inferencia_s': t_pred, 'auc': auc(ctx['yte'], s)})
        log(f'  xgboost {metodo} {disp} hilos={hilos}: {t_fit:.2f} s')
    Xa, Xv, ya, yv = train_test_split(ctx['Xtr'], ctx['ytr'], test_size=0.1, stratify=ctx['ytr'],
                                      random_state=ctx['seed'])
    for nombre, kw in (('500 árboles fijos, tasa 0,05', {'n_estimators': 500}),
                       ('hasta 5.000 árboles con early stopping (50 rondas)',
                        {'n_estimators': 5000, 'early_stopping_rounds': 50})):
        m = XGBClassifier(**{**base, 'learning_rate': 0.05, 'tree_method': 'hist', 'n_jobs': -1, **kw})
        t_fit, m = cronometrar(lambda: m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False), 2)
        usados = (m.best_iteration + 1) if 'early_stopping_rounds' in kw else kw['n_estimators']
        filas.append({'experimento': 'early stopping', 'variante': nombre, 'arboles': usados,
                      'entrenamiento_s': t_fit, 'inferencia_s': None,
                      'auc': auc(ctx['yte'], m.predict_proba(ctx['Xte'])[:, 1])})
        log(f'  xgboost {nombre}: {t_fit:.2f} s, {usados} árboles')
    return pd.DataFrame(filas)


def exp_svm(ctx) -> pd.DataFrame:
    """SVC con núcleo RBF frente a LinearSVC y SGD, sobre submuestras crecientes."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC, LinearSVC
    C = ctx['hp'][('clasificacion', 'svm')]['C']
    rng = np.random.default_rng(ctx['seed'])
    orden = rng.permutation(len(ctx['ytr']))
    filas = []
    for n in ((500, 1000, len(ctx['ytr'])) if PRUEBA else (1000, 2000, 4000, 8000, len(ctx['ytr']))):
        idx = orden[:n]
        modelos = {'LinearSVC (primal, L2)': lambda: LinearSVC(C=C, dual=False, max_iter=5000),
                   'SGD con pérdida hinge': lambda: SGDClassifier(loss='hinge', alpha=1 / (C * n), max_iter=50,
                                                                 tol=1e-4, random_state=ctx['seed'])}
        if n <= 8000:
            modelos['SVC núcleo RBF'] = lambda: SVC(kernel='rbf', C=1.0, gamma='scale')
        for nombre, crear in modelos.items():
            t_fit, m = cronometrar(lambda: crear().fit(ctx['Xtr'][idx], ctx['ytr'][idx]), 1 if 'RBF' in nombre else 3)
            t_pred, s = cronometrar(lambda: m.decision_function(ctx['Xte']), 1)
            filas.append({'modelo': nombre, 'n': n, 'entrenamiento_s': t_fit, 'inferencia_s': t_pred,
                          'auc': auc(ctx['yte'], s),
                          'vectores_soporte': int(m.n_support_.sum()) if hasattr(m, 'n_support_') else None})
            log(f'  svm {nombre} n={n}: {t_fit:.2f} s + {t_pred:.2f} s')
    return pd.DataFrame(filas)


def exp_paralelismo(ctx) -> pd.DataFrame:
    """Aceleración con hilos (Random Forest) y *backends* de joblib (búsqueda en rejilla)."""
    from joblib import parallel_config
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    filas = []
    for hilos in (1, 2, 3, 4, 6):
        m = estimador(ctx, 'clasificacion', 'random_forest', n_jobs=hilos)
        t_fit, _ = cronometrar(lambda: m.fit(ctx['Xtr'], ctx['ytr']), 2)
        filas.append({'experimento': 'Random Forest (300 árboles)', 'backend': 'hilos nativos',
                      'n_jobs': hilos, 'tiempo_s': t_fit})
        log(f'  RF n_jobs={hilos}: {t_fit:.2f} s')
    rejilla = {'C': list(np.logspace(-3, 2, 12))}
    cv = StratifiedKFold(5, shuffle=True, random_state=ctx['seed'])
    for backend, hilos in (('loky', 1), ('loky', 2), ('loky', 4), ('loky', 6), ('threading', 6),
                           ('multiprocessing', 6)):
        def buscar():
            with parallel_config(backend=backend, n_jobs=hilos):
                gs = GridSearchCV(LogisticRegression(l1_ratio=1.0, solver='liblinear', max_iter=2000), rejilla,
                                  cv=cv, scoring='roc_auc', n_jobs=hilos)
                return gs.fit(ctx['Xtr'], ctx['ytr'])
        t, _ = cronometrar(buscar, 2)
        filas.append({'experimento': 'GridSearchCV logística L1 (12 × 5 ajustes)', 'backend': backend,
                      'n_jobs': hilos, 'tiempo_s': t})
        log(f'  grid {backend} n_jobs={hilos}: {t:.2f} s')
    return pd.DataFrame(filas)


def exp_entre_modelos(ctx) -> pd.DataFrame:
    """Paralelizar entre modelos frente a paralelizar dentro de cada modelo (guía §4.3).

    Cuatro búsquedas en rejilla (logística, Naive Bayes, árbol y SVM lineal; 8 configuraciones × 3
    pliegues cada una) con los 6 núcleos repartidos de cuatro maneras.
    """
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    cv = StratifiedKFold(3, shuffle=True, random_state=ctx['seed'])
    Xtr, ytr = ctx['Xtr'], ctx['ytr']
    busquedas = [(LogisticRegression(solver='liblinear', l1_ratio=1.0, max_iter=2000), {'C': list(np.logspace(-3, 2, 8))}),
                 (GaussianNB(), {'var_smoothing': list(np.logspace(-11, -3, 8))}),
                 (DecisionTreeClassifier(random_state=ctx['seed']), {'max_depth': [4, 6, 8, 12, 16, 20, 25, 30]}),
                 (LinearSVC(dual=False, max_iter=5000), {'C': list(np.logspace(-4, 2, 8))})]

    def buscar(est, rejilla, n_jobs):
        return GridSearchCV(est, rejilla, cv=cv, scoring='roc_auc', n_jobs=n_jobs).fit(Xtr, ytr).best_score_

    esquemas = {
        'todo en serie (1 núcleo)': lambda: [buscar(e, r, 1) for e, r in busquedas],
        'dentro de cada modelo (n_jobs=6), modelos en serie': lambda: [buscar(e, r, 6) for e, r in busquedas],
        'entre modelos (4 a la vez, n_jobs=1 dentro)': lambda: Parallel(n_jobs=4)(delayed(buscar)(e, r, 1) for e, r in busquedas),
        'mixto (2 modelos a la vez × n_jobs=3 dentro)': lambda: Parallel(n_jobs=2)(delayed(buscar)(e, r, 3) for e, r in busquedas),
    }
    filas = []
    for nombre, f in esquemas.items():
        t, _ = cronometrar(f, 1 if nombre.startswith('todo') else 2)
        filas.append({'esquema': nombre, 'tiempo_s': t})
        log(f'  entre modelos · {nombre}: {t:.2f} s')
    return pd.DataFrame(filas)


def exp_perfil(ctx) -> pd.DataFrame:
    """cProfile de una validación cruzada de 3 pliegues del pipeline completo con XGBoost."""
    from sklearn.model_selection import cross_val_score
    from src import balanceo as bal
    spec = ctx['modelos'].catalogo('clasificacion')['xgboost']
    pipe = bal.construir_pipeline(spec, 'smote', ctx['X_crudo'], ctx['seed'], ctx['y_crudo'])
    pipe.set_params(**{f'modelo__{k}': v for k, v in ctx['hp'][('clasificacion', 'xgboost')].items()})
    perfil = cProfile.Profile()
    perfil.enable()
    cross_val_score(pipe, ctx['X_crudo'], ctx['y_crudo'], cv=3, scoring='roc_auc', n_jobs=1)
    perfil.disable()
    texto = io.StringIO()
    st = pstats.Stats(perfil, stream=texto).sort_stats('cumulative')
    st.print_stats(25)
    (SALIDA / 'perfil.txt').write_text(texto.getvalue(), encoding='utf-8')
    total = st.total_tt
    filas = []
    for (archivo, linea, funcion), (cc, nc, tt, ct, _) in st.stats.items():
        filas.append({'funcion': funcion, 'archivo': Path(archivo).name, 'llamadas': nc, 'propio_s': tt,
                      'acumulado_s': ct})
    df = pd.DataFrame(filas).sort_values('acumulado_s', ascending=False)
    df['fraccion_del_total'] = df['acumulado_s'] / max(df['acumulado_s'].max(), total)
    return df.head(60)


def exp_fidelidad(ctx) -> pd.DataFrame:
    """¿Conserva la baja fidelidad el orden de las configuraciones? Random Forest y EBM."""
    from sklearn.base import clone
    from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
    from src.optimizacion import distribuciones_random
    cv = StratifiedKFold(3, shuffle=True, random_state=ctx['seed'])
    filas = []
    for nombre, n_cfg, baja, alta in (('random_forest', 2 if PRUEBA else 12, {'n_estimators': 100}, {'n_estimators': 300}),
                                      ('ebm', 2 if PRUEBA else 8, {'outer_bags': 2}, {'outer_bags': 8})):
        spec = ctx['modelos'].catalogo('clasificacion', incluir_nuevo=True)[nombre]
        espacio = {k.split('__', 1)[1]: v for k, v in spec.espacio.items()}
        for i, cfg in enumerate(ParameterSampler(distribuciones_random(espacio), n_iter=n_cfg,
                                                 random_state=ctx['seed'])):
            base = spec.construir(ctx['seed']).set_params(**cfg)
            fila = {'modelo': nombre, 'configuracion': i, **{f'hp_{k}': v for k, v in cfg.items()}}
            for etiqueta, fijos in (('baja', baja), ('alta', alta)):
                t0 = time.perf_counter()
                s = cross_val_score(clone(base).set_params(**fijos), ctx['Xtr'], ctx['ytr'], cv=cv,
                                    scoring='roc_auc', n_jobs=1 if nombre == 'random_forest' else 3)
                fila[f'auc_{etiqueta}'] = float(s.mean())
                fila[f'tiempo_{etiqueta}_s'] = time.perf_counter() - t0
            filas.append(fila)
            log(f'  fidelidad {nombre} cfg {i}: baja {fila["auc_baja"]:.4f} · alta {fila["auc_alta"]:.4f}')
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- espera
def esperar_pc_libre() -> None:
    """Espera a que terminen la comparación y las corridas del modelo nuevo (revisa cada minuto)."""
    import psutil
    cerrojo = RAIZ / 'resultados' / 'experimentos.parquet.en_uso'   # el de src.registro.ruta_cerrojo
    libre_desde = None
    avisado = False
    while True:
        ocupado = cerrojo.exists()
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                cmd = ' '.join(proc.info['cmdline'] or [])
            except psutil.Error:
                continue
            if 'python' in (proc.info['name'] or '').lower() and ('correr_140' in cmd or 'comparar_entrega2' in cmd):
                ocupado = True
                break
        if ocupado:
            libre_desde = None
            if not avisado:
                log('en espera: siguen en marcha la comparación o las corridas del modelo nuevo')
                avisado = True
        else:
            libre_desde = libre_desde or time.time()
            if time.time() - libre_desde >= 180:      # libre durante 3 minutos seguidos
                log('el PC quedó libre; empiezan los experimentos')
                return
        time.sleep(60)


def main() -> None:
    global log_archivo
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('experimentos', nargs='*', help=f'subconjunto de {EXPERIMENTOS}')
    ap.add_argument('--esperar', action='store_true')
    ap.add_argument('--rehacer', action='store_true')
    ap.add_argument('--prueba', action='store_true', help='ensayo rápido sobre una submuestra (resultados_prueba/)')
    args = ap.parse_args()
    desconocidos = set(args.experimentos) - set(EXPERIMENTOS)
    if desconocidos:
        ap.error(f'experimentos desconocidos: {sorted(desconocidos)}')
    global SALIDA, PRUEBA
    PRUEBA = args.prueba
    if PRUEBA:
        SALIDA = RAIZ / 'resultados_prueba' / 'computo'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    warnings.filterwarnings('ignore')
    SALIDA.mkdir(parents=True, exist_ok=True)
    log_archivo = open(SALIDA / 'progreso.log', 'a', encoding='utf-8')
    sys.path.insert(0, str(RAIZ))
    if sys.platform == 'win32':           # sin suspensión también mientras espera
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if args.esperar:
        esperar_pc_libre()
    from correr_140 import congelar_codigo
    destino = RAIZ / '_codigo_computo'
    huella = congelar_codigo(destino, refrescar=True)
    sys.path.insert(0, str(destino))
    pedidos = args.experimentos or EXPERIMENTOS
    log('=' * 78)
    log(f'inicio · código congelado en {destino.name} (huella {huella}) · experimentos: {pedidos}')
    t0 = time.perf_counter()
    ctx = preparar()
    log(f'datos listos: {ctx["Xtr"].shape} entrenamiento, {ctx["Xte"].shape} prueba')
    for nombre in pedidos:
        ruta = SALIDA / f'{nombre}.csv'
        if ruta.exists() and not args.rehacer:
            log(f'{nombre}: ya hecho, se salta')
            continue
        t1 = time.perf_counter()
        try:
            df = globals()[f'exp_{nombre}'](ctx)
        except Exception as e:                                   # noqa: BLE001
            import traceback
            log(f'{nombre}: ERROR {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}')
            continue
        df.to_csv(ruta, index=False)
        log(f'{nombre}: hecho en {(time.perf_counter() - t1) / 60:.1f} min')
    (SALIDA / 'equipo.json').write_text(json.dumps(equipo(), ensure_ascii=False, indent=1), encoding='utf-8')
    log(f'fin en {(time.perf_counter() - t0) / 60:.1f} min')


def equipo() -> dict:
    """Descripción del equipo de medición para el informe."""
    import platform
    import psutil
    info = {'procesador': platform.processor(), 'nucleos_logicos': psutil.cpu_count(),
            'nucleos_fisicos': psutil.cpu_count(logical=False),
            'memoria_gb': round(psutil.virtual_memory().total / 2 ** 30, 1), 'sistema': platform.platform(),
            'python': platform.python_version()}
    try:
        import subprocess
        info['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                            # noqa: BLE001
        info['gpu'] = None
    return info


if __name__ == '__main__':
    main()
