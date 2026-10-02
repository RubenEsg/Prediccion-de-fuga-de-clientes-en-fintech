"""Benchmark rápido de candidatos a *modelo nuevo* para el Entregable 3.

El profesor pide un modelo distinto de los de la Entrega 2 y no explicado en clase que supere
sus métricas. Antes de elegirlo, este guion mide a todos los candidatos razonables bajo un
mismo protocolo, junto con los modelos de referencia ya vistos en clase.

Protocolo
---------
* Misma partición 80/20 con semilla 42 que la Entrega 2; **la prueba no se toca**.
* 5 pliegues (estratificados en clasificación) sobre el conjunto de entrenamiento.
* Preprocesamiento (escalado + indicadoras) ajustado dentro de cada pliegue, vía
  ``src.preprocesamiento.construir_preprocesador``.
* Clasificación con dos etiquetas: **mediana** (la de la Entrega 2, AUC 0,6832 en prueba con
  regresión logística L1) y **percentil 75** (la etiqueta por defecto del Entregable 3).
* Regresión sobre ``churn_probability`` (Entrega 2: Lasso, R² 0,1511 y RMSE 0,0617 en prueba).
* Hiperparámetros razonables fijos, sin búsqueda: el objetivo es descartar y priorizar, no
  reportar. Los finalistas pasan después por la validación anidada de ``src.optimizacion``.
* Fila especial ``reg→rank``: regresor ajustado sobre el objetivo continuo cuyo pronóstico se
  usa como puntaje de clasificación. Explota que la etiqueta es una función determinista de
  ``churn_probability``.

Uso
---
    .venv/Scripts/python.exe exploracion/benchmark_modelo_nuevo.py
    .venv/Scripts/python.exe exploracion/benchmark_modelo_nuevo.py --folds 3 --bloques clf_mediana,reg

Salidas: ``exploracion/benchmark_modelo_nuevo.json`` y ``.md`` (se reescriben tras cada modelo).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import datos  # noqa: E402
from src.config import SEED, fijar_semillas  # noqa: E402
from src.preprocesamiento import construir_preprocesador  # noqa: E402

N_JOBS = 6
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def _opcional(modulo: str):
    try:
        return __import__(modulo)
    except ImportError:
        return None


lgb = _opcional('lightgbm')
cb = _opcional('catboost')
try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    EBM = True
except ImportError:                                                    # pragma: no cover
    EBM = False

REFERENCIA_ENTREGA_2 = {
    'clf_mediana': 'AUC 0,6832 en prueba (logística L1, C = 0,1)',
    'clf_p75': 'sin referencia directa: la Entrega 2 usó la mediana',
    'reg': 'R² 0,1511 y RMSE 0,0617 en prueba (Lasso, alpha = 1e-4)',
}


# --------------------------------------------------------------------------- catálogos
# Cada entrada: nombre -> (constructor, visto_en_clase, objetivo_de_ajuste)
# objetivo_de_ajuste: 'bin' ajusta con la etiqueta binaria; 'cont' ajusta con churn_probability
# y usa predict() como puntaje (fila reg→rank).
def clasificadores() -> dict:
    m = {
        'logistica L1 C=0,1 [Entrega 2]': (lambda: LogisticRegression(
            penalty='l1', C=0.1, solver='liblinear', max_iter=2000, random_state=SEED), True, 'bin'),
        'knn k=200 [Entrega 2]': (lambda: KNeighborsClassifier(n_neighbors=200, n_jobs=N_JOBS), True, 'bin'),
        'random forest [clase]': (lambda: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=N_JOBS, random_state=SEED), True, 'bin'),
        'xgboost [clase]': (lambda: XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', n_jobs=N_JOBS, random_state=SEED, verbosity=0), True, 'bin'),
        'hist gradient boosting': (lambda: HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=SEED), False, 'bin'),
        'extra trees': (lambda: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=N_JOBS, random_state=SEED), False, 'bin'),
        'mlp (128,64)': (lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=300,
            random_state=SEED), False, 'bin'),
        'stacking logit+hgb+knn+nb -> logit': (lambda: StackingClassifier(
            estimators=[
                ('logit', LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=2000,
                                             random_state=SEED)),
                ('hgb', HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=SEED)),
                ('knn', KNeighborsClassifier(n_neighbors=200, n_jobs=N_JOBS)),
                ('nb', GaussianNB())],
            final_estimator=LogisticRegression(max_iter=2000), cv=3, stack_method='predict_proba',
            n_jobs=1), False, 'bin'),
        'reg→rank: hgb regresor': (lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=SEED), False, 'cont'),
        'reg→rank: lasso 1e-4': (lambda: Lasso(alpha=1e-4, max_iter=20000), False, 'cont'),
    }
    if lgb is not None:
        m['lightgbm'] = (lambda: lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=N_JOBS, random_state=SEED, verbose=-1), False, 'bin')
    if cb is not None:
        m['catboost'] = (lambda: cb.CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3, thread_count=N_JOBS,
            random_seed=SEED, verbose=0, allow_writing_files=False), False, 'bin')
        m['reg→rank: catboost regresor'] = (lambda: cb.CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3, thread_count=N_JOBS,
            random_seed=SEED, verbose=0, allow_writing_files=False), False, 'cont')
    if EBM:
        m['ebm (GA2M)'] = (lambda: ExplainableBoostingClassifier(
            outer_bags=4, interactions=10, max_rounds=5000, random_state=SEED, n_jobs=N_JOBS), False, 'bin')
    return m


def regresores() -> dict:
    m = {
        'lasso 1e-4 [Entrega 2]': (lambda: Lasso(alpha=1e-4, max_iter=20000), True, 'cont'),
        'ridge [clase]': (lambda: Ridge(alpha=1.0), True, 'cont'),
        'knn k=200 [clase]': (lambda: KNeighborsRegressor(n_neighbors=200, n_jobs=N_JOBS), True, 'cont'),
        'random forest [clase]': (lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=N_JOBS, random_state=SEED), True, 'cont'),
        'xgboost [clase]': (lambda: XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', n_jobs=N_JOBS, random_state=SEED, verbosity=0), True, 'cont'),
        'hist gradient boosting': (lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=SEED), False, 'cont'),
        'extra trees': (lambda: ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=N_JOBS, random_state=SEED), False, 'cont'),
        'mlp (128,64)': (lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=300,
            random_state=SEED), False, 'cont'),
        'stacking lasso+hgb+knn -> ridge': (lambda: StackingRegressor(
            estimators=[
                ('lasso', Lasso(alpha=1e-4, max_iter=20000)),
                ('hgb', HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, random_state=SEED)),
                ('knn', KNeighborsRegressor(n_neighbors=200, n_jobs=N_JOBS))],
            final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=1), False, 'cont'),
    }
    if lgb is not None:
        m['lightgbm'] = (lambda: lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=N_JOBS, random_state=SEED, verbose=-1), False, 'cont')
    if cb is not None:
        m['catboost'] = (lambda: cb.CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3, thread_count=N_JOBS,
            random_seed=SEED, verbose=0, allow_writing_files=False), False, 'cont')
    if EBM:
        m['ebm (GA2M)'] = (lambda: ExplainableBoostingRegressor(
            outer_bags=4, interactions=10, max_rounds=5000, random_state=SEED, n_jobs=N_JOBS), False, 'cont')
    return m


# --------------------------------------------------------------------------- evaluación
def evaluar(nombre, ctor, visto, objetivo, X, y_bin, y_cont, tarea, pliegues) -> dict:
    """Ajusta y puntúa un modelo en cada pliegue; devuelve una fila de resultados."""
    pre = construir_preprocesador(X)
    y_fit = y_cont if objetivo == 'cont' else y_bin
    puntajes, secundaria, tiempos = [], [], []
    for tr, va in pliegues:
        pipe = Pipeline([('preprocesado', clone(pre)), ('modelo', ctor())])
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr], y_fit.iloc[tr])
        if tarea == 'clasificacion':
            if objetivo == 'cont':
                s = pipe.predict(X.iloc[va])
            else:
                s = pipe.predict_proba(X.iloc[va])[:, 1]
            puntajes.append(float(roc_auc_score(y_bin.iloc[va], s)))
        else:
            pred = pipe.predict(X.iloc[va])
            puntajes.append(float(r2_score(y_cont.iloc[va], pred)))
            secundaria.append(float(root_mean_squared_error(y_cont.iloc[va], pred)))
        tiempos.append(time.perf_counter() - t0)
    fila = {'modelo': nombre, 'visto_en_clase': visto,
            'metrica': 'AUC' if tarea == 'clasificacion' else 'R2',
            'media': float(np.mean(puntajes)), 'desv': float(np.std(puntajes, ddof=1)),
            'por_pliegue': puntajes, 'tiempo_medio_s': float(np.mean(tiempos))}
    if secundaria:
        fila['rmse_media'] = float(np.mean(secundaria))
    return fila


def tabla_markdown(resultados: dict, folds: int) -> str:
    lineas = [f'# Benchmark de candidatos a modelo nuevo ({folds} pliegues sobre entrenamiento)', '',
              'Semilla 42, misma partición 80/20 que la Entrega 2, preprocesamiento dentro de cada pliegue. '
              'Sin búsqueda de hiperparámetros: sirve para descartar y priorizar, no para reportar.', '']
    titulos = {'clf_mediana': 'Clasificación, etiqueta = mediana (comparable con la Entrega 2)',
               'clf_p75': 'Clasificación, etiqueta = percentil 75 (etiqueta del Entregable 3)',
               'reg': 'Regresión sobre churn_probability'}
    for bloque, filas in resultados.items():
        if not filas:
            continue
        lineas += [f'## {titulos[bloque]}', '', f'Referencia Entrega 2: {REFERENCIA_ENTREGA_2[bloque]}.', '']
        extra = ' | RMSE medio' if bloque == 'reg' else ''
        lineas += [f'| Modelo | Visto en clase | {filas[0]["metrica"]} media ± desv{extra} | s por ajuste |',
                   '|---|---|---|---|' if not extra else '|---|---|---|---|---|']
        for f in sorted(filas, key=lambda r: -r['media']):
            celda = f'{f["media"]:.4f} ± {f["desv"]:.4f}'
            if extra:
                celda += f' | {f["rmse_media"]:.4f}'
            lineas.append(f'| {f["modelo"]} | {"sí" if f["visto_en_clase"] else "**no**"} | {celda} | '
                          f'{f["tiempo_medio_s"]:.1f} |')
        lineas.append('')
    return '\n'.join(lineas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--bloques', default='clf_mediana,reg,clf_p75',
                    help='subconjunto de clf_mediana, reg, clf_p75 separado por comas')
    ap.add_argument('--salida', default=str(RAIZ / 'exploracion'))
    args = ap.parse_args()
    bloques = [b.strip() for b in args.bloques.split(',') if b.strip()]
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    fijar_semillas(SEED)
    t0 = time.perf_counter()
    p = datos.preparar_todo(metodo_etiqueta='mediana')
    c = p['conjuntos']
    X = c.X_train.reset_index(drop=True)
    y_cont = c.y_train_cont.reset_index(drop=True)
    y_mediana = p['y_train'].reset_index(drop=True)
    y_p75 = datos.construir_etiqueta(c.y_train_cont, metodo='percentil', q=0.75).binarizar(c.y_train_cont)
    y_p75 = y_p75.reset_index(drop=True)
    print(f'datos listos en {time.perf_counter() - t0:.0f} s: X_train {X.shape}, '
          f'positivos mediana {y_mediana.mean():.3f}, positivos p75 {y_p75.mean():.3f}; '
          f'lightgbm={lgb is not None} catboost={cb is not None} ebm={EBM}', flush=True)

    resultados = {b: [] for b in bloques}
    ruta_json, ruta_md = salida / 'benchmark_modelo_nuevo.json', salida / 'benchmark_modelo_nuevo.md'

    def guardar():
        ruta_json.write_text(json.dumps({'folds': args.folds, 'seed': SEED, 'n_train': int(len(X)),
                                         'resultados': resultados}, ensure_ascii=False, indent=1),
                             encoding='utf-8')
        ruta_md.write_text(tabla_markdown(resultados, args.folds), encoding='utf-8')

    for bloque in bloques:
        if bloque == 'reg':
            tarea, y_bin, catalogo = 'regresion', None, regresores()
            pliegues = list(KFold(args.folds, shuffle=True, random_state=SEED).split(X))
        else:
            tarea, y_bin, catalogo = 'clasificacion', (y_mediana if bloque == 'clf_mediana' else y_p75), clasificadores()
            pliegues = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(X, y_bin))
        print(f'\n=== {bloque}: {len(catalogo)} modelos ===', flush=True)
        for nombre, (ctor, visto, objetivo) in catalogo.items():
            t1 = time.perf_counter()
            try:
                fila = evaluar(nombre, ctor, visto, objetivo, X, y_bin, y_cont, tarea, pliegues)
            except Exception as e:                                        # noqa: BLE001
                print(f'  {nombre}: ERROR {type(e).__name__}: {e}', flush=True)
                continue
            resultados[bloque].append(fila)
            guardar()
            extra = f', RMSE {fila["rmse_media"]:.4f}' if 'rmse_media' in fila else ''
            print(f'  {nombre:40s} {fila["metrica"]} {fila["media"]:.4f} ± {fila["desv"]:.4f}{extra}  '
                  f'[{time.perf_counter() - t1:.0f} s]', flush=True)
    print(f'\nterminado en {(time.perf_counter() - t0) / 60:.1f} min; tabla en {ruta_md}', flush=True)


if __name__ == '__main__':
    main()
