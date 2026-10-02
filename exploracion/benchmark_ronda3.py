"""Ronda 3: exprimir el margen que queda hasta el techo.

La ronda 2 mostró que (a) un oráculo con ``active_products`` explica el 99,9 % del objetivo,
(b) el techo sin esa variable está en AUC ≈ 0,723 (mediana) y R² ≈ 0,230, y (c) quedarse con
pocas variables mejora algo. Aquí se combinan las dos palancas sobre los mismos pliegues:

* **Clasificador de mezcla (LUPI)**: con el oráculo f(k, x) casi determinista, la probabilidad
  exacta de superar el umbral t sin conocer k es  P(y > t | x) = Σ_k p_k · 1[f(k, x) > t].
  Se evalúa en versión discreta y suavizada con la función de distribución normal.
* **Objetivo depurado (LUPI) + selección de variables**, con HGB y con EBM.
* **Modelos directos con pocas variables** (HGB, XGBoost, EBM), sin información privilegiada.

Uso:  .venv/Scripts/python.exe exploracion/benchmark_ronda3.py
Salidas: exploracion/benchmark_ronda3.json y .md
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import benchmark_ronda2 as r2m  # noqa: E402
from src.config import SEED, fijar_semillas  # noqa: E402

warnings.filterwarnings('ignore')
try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    EBM = True
except ImportError:                                                     # pragma: no cover
    EBM = False


def umbral(y: pd.Series, etiqueta: pd.Series) -> float:
    """Umbral implícito de una etiqueta binaria ya construida (punto medio de la frontera)."""
    return float((y[etiqueta == 0].max() + y[etiqueta == 1].min()) / 2)


def top_k(X: np.ndarray, objetivo: np.ndarray, k: int) -> np.ndarray:
    rho = np.array([abs(stats.spearmanr(X[:, j], objetivo).statistic) for j in range(X.shape[1])])
    return np.argsort(-np.nan_to_num(rho))[:k]


def residuo_depurado(y_tr: np.ndarray, ap_tr: np.ndarray) -> np.ndarray:
    return y_tr - pd.Series(y_tr).groupby(ap_tr).mean().loc[ap_tr].values


# --------------------------------------------------------------------------- experimentos
def exp_mezcla(ctx, t_med: float, t_p75: float) -> dict:
    """Clasificador de mezcla a partir del oráculo; R² con el pronóstico marginalizado."""
    disc, suave = [], []
    for f in ctx.P:
        y_tr, ap_tr = ctx.y.values[f['tr']], ctx.ap.values[f['tr']]
        Xo = np.column_stack([f['Xtr'], ap_tr])
        # sigma del oráculo: error fuera de muestra en una partición interna 80/20
        a, b = train_test_split(np.arange(len(y_tr)), test_size=0.2, random_state=SEED)
        sigma = float(np.std(y_tr[b] - r2m.hgb_reg().fit(Xo[a], y_tr[a]).predict(Xo[b])))
        m = r2m.hgb_reg().fit(Xo, y_tr)
        niveles, cuentas = np.unique(ap_tr, return_counts=True)
        pesos = cuentas / cuentas.sum()
        F = np.column_stack([m.predict(np.column_stack([f['Xva'], np.full(len(f['va']), k)])) for k in niveles])
        fila_d = {'r2': float(r2_score(ctx.y.values[f['va']], F @ pesos))}
        fila_s = dict(fila_d)
        for nombre, y_bin, t in (('auc_med', ctx.y_med, t_med), ('auc_p75', ctx.y_p75, t_p75)):
            fila_d[nombre] = float(roc_auc_score(y_bin.values[f['va']], (F > t) @ pesos))
            fila_s[nombre] = float(roc_auc_score(y_bin.values[f['va']], norm.cdf((F - t) / sigma) @ pesos))
        fila_s['sigma'] = sigma
        disc.append(fila_d); suave.append(fila_s)
    return {'discreta': r2m.acumular(disc), 'suavizada': r2m.acumular(suave)}


def exp_lupi_seleccion(ctx, k: int, ctor) -> dict:
    """Objetivo depurado con active_products + las k variables más asociadas al residuo."""
    filas = []
    for f in ctx.P:
        y_tr = ctx.y.values[f['tr']]
        res = residuo_depurado(y_tr, ctx.ap.values[f['tr']])
        idx = top_k(f['Xtr'], res, k) if k else np.arange(f['Xtr'].shape[1])
        m = ctor().fit(f['Xtr'][:, idx], res)
        fila = ctx.metricas_de_pronostico(f['va'], m.predict(f['Xva'][:, idx]) + y_tr.mean())
        filas.append(fila)
    return r2m.acumular(filas)


def exp_directo_seleccion(ctx, k: int, ctor_clf, ctor_reg) -> dict:
    """Sin información privilegiada: k variables por |Spearman| con el objetivo, modelo directo."""
    filas = []
    for f in ctx.P:
        y_tr = ctx.y.values[f['tr']]
        idx = top_k(f['Xtr'], y_tr, k)
        fila = {}
        for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
            m = ctor_clf().fit(f['Xtr'][:, idx], y.values[f['tr']])
            fila[nombre] = float(roc_auc_score(y.values[f['va']], r2m.puntaje(m, f['Xva'][:, idx])))
        fila['r2'] = float(r2_score(ctx.y.values[f['va']], ctor_reg().fit(f['Xtr'][:, idx], y_tr).predict(f['Xva'][:, idx])))
        fila['variables'] = [f['nombres'][j] for j in idx]
        filas.append(fila)
    sal = r2m.acumular([{a: b for a, b in fila.items() if a != 'variables'} for fila in filas])
    sal['variables_por_pliegue'] = [fila['variables'] for fila in filas]
    return sal


def ebm_clf():
    return ExplainableBoostingClassifier(outer_bags=8, interactions=10, random_state=SEED, n_jobs=r2m.N_JOBS)


def ebm_reg():
    return ExplainableBoostingRegressor(outer_bags=8, interactions=10, random_state=SEED, n_jobs=r2m.N_JOBS)


def tabla(res: dict) -> str:
    ref = res['hgb directo, todas las variables [referencia]']
    lin = ['# Ronda 3: selección de variables e información privilegiada (5 pliegues sobre entrenamiento)', '',
           'Δ = diferencia media pareada frente a HGB directo con todas las variables; p = t pareada.', '',
           '| Experimento | Usa active_products al entrenar | AUC mediana | Δ (p) | AUC percentil 75 | Δ (p) | R² | Δ (p) |',
           '|---|---|---|---|---|---|---|---|']

    def c(v):
        return f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}'

    def d(v, r):
        if v is r:
            return '—'
        return f'{np.mean(np.array(v) - np.array(r)):+.4f} ({stats.ttest_rel(v, r).pvalue:.3f})'
    for nombre, r in res.items():
        lin.append(f'| {nombre} | {"sí" if r.get("lupi") else "no"} | {c(r["auc_med"])} | {d(r["auc_med"], ref["auc_med"])} | '
                   f'{c(r["auc_p75"])} | {d(r["auc_p75"], ref["auc_p75"])} | {c(r["r2"])} | {d(r["r2"], ref["r2"])} |')
    return '\n'.join(lin) + '\n'


def main() -> None:
    fijar_semillas(SEED)
    t0 = time.perf_counter()
    ctx = r2m.Contexto(5)
    t_med, t_p75 = umbral(ctx.y, ctx.y_med), umbral(ctx.y, ctx.y_p75)
    print(f'datos listos en {time.perf_counter() - t0:.0f} s; umbral mediana {t_med:.4f}, p75 {t_p75:.4f}', flush=True)
    res: dict = {}

    def registrar(nombre, r, lupi):
        r['lupi'] = lupi
        res[nombre] = r
        (AQUI / 'benchmark_ronda3.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                    encoding='utf-8')
        (AQUI / 'benchmark_ronda3.md').write_text(tabla(res), encoding='utf-8')
        print(f'  RESULTADO {nombre:62s} auc_med {np.mean(r["auc_med"]):.4f} | auc_p75 {np.mean(r["auc_p75"]):.4f} '
              f'| r2 {np.mean(r["r2"]):.4f}', flush=True)

    registrar('hgb directo, todas las variables [referencia]', r2m.exp_referencia(ctx, 'hgb'), False)
    for k in (4, 5, 6, 8, 12):
        registrar(f'hgb directo, {k} variables', exp_directo_seleccion(ctx, k, r2m.hgb_clf, r2m.hgb_reg), False)
    registrar('xgboost directo, 6 variables', exp_directo_seleccion(ctx, 6, r2m.xgb_clf, r2m.xgb_reg), False)
    if EBM:
        registrar('ebm directo, 6 variables', exp_directo_seleccion(ctx, 6, ebm_clf, ebm_reg), False)
    m = exp_mezcla(ctx, t_med, t_p75)
    registrar('mezcla del oraculo, discreta', m['discreta'], True)
    registrar('mezcla del oraculo, suavizada', m['suavizada'], True)
    registrar('objetivo depurado + hgb, todas las variables', exp_lupi_seleccion(ctx, 0, r2m.hgb_reg), True)
    for k in (4, 6, 8):
        registrar(f'objetivo depurado + hgb, {k} variables', exp_lupi_seleccion(ctx, k, r2m.hgb_reg), True)
    if EBM:
        registrar('objetivo depurado + ebm, 6 variables', exp_lupi_seleccion(ctx, 6, ebm_reg), True)
    print(f'terminado ronda 3 en {(time.perf_counter() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
