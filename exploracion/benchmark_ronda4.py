"""Ronda 4: modelos vistos en clase con un cambio que los hace competitivos.

Las rondas 2 y 3 mostraron que casi toda la señal está en unas pocas variables y que el resto
es ruido. Un modelo de clase puede acercarse al techo si se le quita ese ruido o se le da la
forma funcional adecuada. Sobre los mismos 5 pliegues de entrenamiento se prueban:

* **k-NN, SVM (Nyström), Naive Bayes y Random Forest con selección de variables** dentro del
  pliegue (los métodos de distancia sufren la maldición de la dimensión con 78 columnas).
* **Regresión logística y Ridge con splines** sobre las variables seleccionadas, con y sin
  interacciones por pares entre las más importantes (un GAM construido con piezas de clase).
* **Logit Leaf Model** (De Caigny, Coussement y De Bock, 2018): híbrido propio de la
  literatura de churn; un árbol de decisión segmenta a los clientes y en cada hoja se ajusta
  una regresión logística. En regresión, árbol + Ridge por hoja.

Uso:  .venv/Scripts/python.exe exploracion/benchmark_ronda4.py
Salidas: exploracion/benchmark_ronda4.json y .md
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import benchmark_ronda2 as r2m  # noqa: E402
from benchmark_ronda3 import top_k  # noqa: E402
from src.config import SEED, fijar_semillas  # noqa: E402

warnings.filterwarnings('ignore')
N_JOBS = r2m.N_JOBS


def evaluar(ctx, k: int, ctor_clf=None, ctor_reg=None, transformar=None) -> dict:
    """Selección de k variables dentro del pliegue (k = 0 usa todas) y modelo directo.

    ``transformar(Xtr, Xva)`` permite construir bases (splines, Nyström) ajustadas solo con
    las filas de entrenamiento del pliegue.
    """
    filas = []
    for f in ctx.P:
        y_tr = ctx.y.values[f['tr']]
        idx = top_k(f['Xtr'], y_tr, k) if k else np.arange(f['Xtr'].shape[1])
        Xtr, Xva = f['Xtr'][:, idx], f['Xva'][:, idx]
        if transformar is not None:
            Xtr, Xva = transformar(Xtr, Xva)
        fila = {}
        if ctor_clf is not None:
            for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
                m = ctor_clf().fit(Xtr, y.values[f['tr']])
                fila[nombre] = float(roc_auc_score(y.values[f['va']], r2m.puntaje(m, Xva)))
        if ctor_reg is not None:
            fila['r2'] = float(r2_score(ctx.y.values[f['va']], ctor_reg().fit(Xtr, y_tr).predict(Xva)))
        filas.append(fila)
    return r2m.acumular(filas)


def splines(n_knots: int = 6, interacciones: int = 0):
    """Base de splines cúbicos por variable; opcionalmente productos por pares entre las
    ``interacciones`` primeras columnas (las más asociadas al objetivo)."""
    def t(Xtr, Xva):
        continuas = [j for j in range(Xtr.shape[1]) if len(np.unique(Xtr[:, j])) > 2]
        resto = [j for j in range(Xtr.shape[1]) if j not in continuas]
        sp = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False).fit(Xtr[:, continuas])
        partes_tr, partes_va = [sp.transform(Xtr[:, continuas]), Xtr[:, resto]], [sp.transform(Xva[:, continuas]), Xva[:, resto]]
        if interacciones:
            pares = list(combinations(range(min(interacciones, Xtr.shape[1])), 2))
            partes_tr.append(np.column_stack([Xtr[:, a] * Xtr[:, b] for a, b in pares]))
            partes_va.append(np.column_stack([Xva[:, a] * Xva[:, b] for a, b in pares]))
        return np.column_stack(partes_tr), np.column_stack(partes_va)
    return t


def nystroem(n: int = 600):
    def t(Xtr, Xva):
        ny = Nystroem(kernel='rbf', gamma=1.0 / Xtr.shape[1], n_components=n, random_state=SEED).fit(Xtr)
        return ny.transform(Xtr), ny.transform(Xva)
    return t


def logit_leaf_model(ctx, k: int, hojas: int) -> dict:
    """Árbol de decisión que segmenta + regresión logística (o Ridge) dentro de cada hoja."""
    filas = []
    for f in ctx.P:
        y_tr = ctx.y.values[f['tr']]
        idx = top_k(f['Xtr'], y_tr, k) if k else np.arange(f['Xtr'].shape[1])
        Xtr, Xva = f['Xtr'][:, idx], f['Xva'][:, idx]
        fila = {}
        for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
            yb = y.values[f['tr']]
            arbol = DecisionTreeClassifier(max_leaf_nodes=hojas, min_samples_leaf=1500, random_state=SEED).fit(Xtr, yb)
            h_tr, h_va = arbol.apply(Xtr), arbol.apply(Xva)
            s = np.zeros(len(Xva))
            for h in np.unique(h_tr):
                a, b = h_tr == h, h_va == h
                if not b.any():
                    continue
                if len(np.unique(yb[a])) < 2:
                    s[b] = yb[a].mean()
                else:
                    s[b] = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr[a], yb[a]).predict_proba(Xva[b])[:, 1]
            fila[nombre] = float(roc_auc_score(y.values[f['va']], s))
        arbol = DecisionTreeRegressor(max_leaf_nodes=hojas, min_samples_leaf=1500, random_state=SEED).fit(Xtr, y_tr)
        h_tr, h_va = arbol.apply(Xtr), arbol.apply(Xva)
        pred = np.zeros(len(Xva))
        for h in np.unique(h_tr):
            a, b = h_tr == h, h_va == h
            if b.any():
                pred[b] = Ridge(alpha=1.0).fit(Xtr[a], y_tr[a]).predict(Xva[b])
        fila['r2'] = float(r2_score(ctx.y.values[f['va']], pred))
        filas.append(fila)
    return r2m.acumular(filas)


def tabla(res: dict) -> str:
    ref = res['hgb directo, todas las variables [referencia]']
    lin = ['# Ronda 4: modelos de clase con un cambio (5 pliegues sobre entrenamiento)', '',
           'Δ = diferencia media pareada frente a HGB directo con todas las variables; p = t pareada.', '',
           '| Experimento | Modelo de clase de partida | AUC mediana | Δ (p) | AUC percentil 75 | R² | Δ (p) |',
           '|---|---|---|---|---|---|---|']

    def c(v):
        return f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}' if v else '—'

    def d(v, r):
        if not v or v is r:
            return '—'
        return f'{np.mean(np.array(v) - np.array(r)):+.4f} ({stats.ttest_rel(v, r).pvalue:.3f})'
    for nombre, r in res.items():
        lin.append(f'| {nombre} | {r.get("base", "")} | {c(r.get("auc_med"))} | {d(r.get("auc_med"), ref["auc_med"])} | '
                   f'{c(r.get("auc_p75"))} | {c(r.get("r2"))} | {d(r.get("r2"), ref["r2"])} |')
    return '\n'.join(lin) + '\n'


def main() -> None:
    fijar_semillas(SEED)
    t0 = time.perf_counter()
    ctx = r2m.Contexto(5)
    print(f'datos listos en {time.perf_counter() - t0:.0f} s', flush=True)
    res: dict = {}

    def registrar(nombre, base, r):
        r['base'] = base
        res[nombre] = r
        (AQUI / 'benchmark_ronda4.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                    encoding='utf-8')
        (AQUI / 'benchmark_ronda4.md').write_text(tabla(res), encoding='utf-8')
        partes = [f'{k} {np.mean(r[k]):.4f}' for k in ('auc_med', 'auc_p75', 'r2') if r.get(k)]
        print(f'  RESULTADO {nombre:58s} ' + ' | '.join(partes), flush=True)

    def logit():
        return LogisticRegression(C=0.5, max_iter=3000)

    def knn_c():
        return KNeighborsClassifier(n_neighbors=200, n_jobs=N_JOBS)

    def knn_r():
        return KNeighborsRegressor(n_neighbors=200, n_jobs=N_JOBS)

    def rf_c():
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=N_JOBS, random_state=SEED)

    def rf_r():
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=N_JOBS, random_state=SEED)

    def svc():
        return LinearSVC(C=0.5, dual='auto', max_iter=5000)

    registrar('hgb directo, todas las variables [referencia]', 'referencia', r2m.exp_referencia(ctx, 'hgb'))
    registrar('logistica / ridge lineal, todas [Entrega 2]', 'logística, Ridge',
              evaluar(ctx, 0, lambda: LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=2000),
                      lambda: Ridge(alpha=1.0)))
    registrar('knn k=200, todas las variables [Entrega 2]', 'k-NN', evaluar(ctx, 0, knn_c, knn_r))
    for k in (4, 6, 8):
        registrar(f'knn k=200, {k} variables', 'k-NN', evaluar(ctx, k, knn_c, knn_r))
    registrar('naive bayes, 6 variables', 'Naive Bayes', evaluar(ctx, 6, GaussianNB, None))
    registrar('random forest, 6 variables', 'Random Forest', evaluar(ctx, 6, rf_c, rf_r))
    registrar('svm rbf (nystroem), 6 variables', 'SVM', evaluar(ctx, 6, svc, lambda: Ridge(alpha=1.0), nystroem(600)))
    for k in (6, 12):
        registrar(f'logistica / ridge + splines, {k} variables', 'logística, Ridge',
                  evaluar(ctx, k, logit, lambda: Ridge(alpha=5.0), splines(6)))
    registrar('logistica / ridge + splines + interacciones, 6 variables', 'logística, Ridge',
              evaluar(ctx, 6, logit, lambda: Ridge(alpha=5.0), splines(6, interacciones=6)))
    registrar('logistica / ridge + splines (10 nudos) + interacciones, 8 variables', 'logística, Ridge',
              evaluar(ctx, 8, logit, lambda: Ridge(alpha=5.0), splines(10, interacciones=8)))
    for k, hojas in ((0, 4), (6, 4), (6, 8)):
        registrar(f'logit leaf model, {k or "todas las"} variables, {hojas} hojas', 'árbol + logística',
                  logit_leaf_model(ctx, k, hojas))
    print(f'terminado ronda 4 en {(time.perf_counter() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
