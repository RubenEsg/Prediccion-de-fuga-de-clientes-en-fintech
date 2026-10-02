"""Ronda 2 del benchmark: ¿se puede subir el puntaje? Techo de información y variantes.

La ronda 1 mostró que todos los modelos modernos se agrupan en AUC ≈ 0,722 (etiqueta mediana) y
R² ≈ 0,224. Esta ronda responde tres preguntas sobre los mismos 5 pliegues de entrenamiento
(estratificados por la etiqueta de la mediana, semilla 42; la prueba no se toca):

1. **¿Dónde está el techo?** ``churn_probability`` depende sobre todo de ``active_products``
   (excluida por ser el ingrediente con el que se construyó la etiqueta), que además es
   independiente del resto de variables. Se ajusta un modelo *oráculo* con esa variable y se
   marginaliza sobre su distribución: eso estima el mejor predictor posible **sin** ella.
2. **¿Ayuda la información privilegiada (LUPI)?** Usar ``active_products`` solo en entrenamiento
   para depurar el objetivo (``y - media por nivel``) y predecir sin ella.
3. **¿Aporta cambiar algo en un modelo de clase?** XGBoost afinado con Optuna, XGBoost DART,
   selección de variables, logística y Ridge con *splines* (GAM), híbrido GBDT + logística,
   SVM con aproximación de Nyström y mezcla (*blending*) de boosting.

Uso:  .venv/Scripts/python.exe exploracion/benchmark_ronda2.py [--trials 25]
Salidas: exploracion/benchmark_ronda2.json y .md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier, XGBRegressor

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import datos  # noqa: E402
from src.config import SEED, fijar_semillas  # noqa: E402
from src.preprocesamiento import columnas_por_tipo, construir_preprocesador, nombres_variables  # noqa: E402

N_JOBS = 6
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import lightgbm as lgb
    import catboost as cb
except ImportError:                                                     # pragma: no cover
    lgb = cb = None


# --------------------------------------------------------------------------- modelos base
def xgb_clf(**kw):
    base = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                tree_method='hist', n_jobs=N_JOBS, random_state=SEED, verbosity=0)
    return XGBClassifier(**{**base, **kw})


def xgb_reg(**kw):
    base = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                tree_method='hist', n_jobs=N_JOBS, random_state=SEED, verbosity=0)
    return XGBRegressor(**{**base, **kw})


def hgb_clf():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=SEED)


def hgb_reg():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=SEED)


def puntaje(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, 'predict_proba') else m.decision_function(X)


# --------------------------------------------------------------------------- utilidades
class Contexto:
    """Datos, etiquetas, pliegues y matrices preprocesadas por pliegue."""

    def __init__(self, folds: int):
        p = datos.preparar_todo(metodo_etiqueta='mediana')
        c = p['conjuntos']
        ap = p['df'].loc[c.X_train.index, 'active_products']
        assert np.allclose(p['df'].loc[c.X_train.index, 'churn_probability'].values, c.y_train_cont.values)
        self.X = c.X_train.reset_index(drop=True)
        self.y = c.y_train_cont.reset_index(drop=True).astype(float)
        self.ap = ap.reset_index(drop=True).astype(int)
        self.y_med = p['y_train'].reset_index(drop=True).astype(int)
        e75 = datos.construir_etiqueta(c.y_train_cont, metodo='percentil', q=0.75)
        self.y_p75 = e75.binarizar(c.y_train_cont).reset_index(drop=True).astype(int)
        self.pliegues = list(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(self.X, self.y_med))
        pre = construir_preprocesador(self.X)
        self.P = []
        for tr, va in self.pliegues:
            pr = clone(pre).fit(self.X.iloc[tr])
            self.P.append({'tr': tr, 'va': va, 'Xtr': pr.transform(self.X.iloc[tr]),
                           'Xva': pr.transform(self.X.iloc[va]), 'nombres': nombres_variables(pr)})

    def metricas_de_pronostico(self, va, pred) -> dict:
        """R² del pronóstico continuo y AUC de ese pronóstico como puntaje para las dos etiquetas."""
        return {'r2': float(r2_score(self.y.values[va], pred)),
                'auc_med': float(roc_auc_score(self.y_med.values[va], pred)),
                'auc_p75': float(roc_auc_score(self.y_p75.values[va], pred))}


def acumular(filas: list[dict]) -> dict:
    claves = sorted({k for f in filas for k in f})
    return {k: [f[k] for f in filas if k in f] for k in claves}


def clasificador_directo(ctx: Contexto, ctor) -> dict:
    filas = []
    for f in ctx.P:
        fila = {}
        for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
            m = ctor().fit(f['Xtr'], y.values[f['tr']])
            fila[nombre] = float(roc_auc_score(y.values[f['va']], puntaje(m, f['Xva'])))
        filas.append(fila)
    return acumular(filas)


def regresor_directo(ctx: Contexto, ctor, solo_r2: bool = True) -> dict:
    filas = []
    for f in ctx.P:
        m = ctor().fit(f['Xtr'], ctx.y.values[f['tr']])
        met = ctx.metricas_de_pronostico(f['va'], m.predict(f['Xva']))
        filas.append({'r2': met['r2']} if solo_r2 else met)
    return acumular(filas)


# --------------------------------------------------------------------------- experimentos
def exp_referencia(ctx, nombre):
    clf, reg = (xgb_clf, xgb_reg) if nombre == 'xgboost' else (hgb_clf, hgb_reg)
    return {**clasificador_directo(ctx, clf), **regresor_directo(ctx, reg)}


def exp_oraculo(ctx):
    """Diagnóstico: modelo CON active_products. No es candidato; mide cuánto explica esa variable."""
    filas = []
    for f in ctx.P:
        Xtr = np.column_stack([f['Xtr'], ctx.ap.values[f['tr']]])
        Xva = np.column_stack([f['Xva'], ctx.ap.values[f['va']]])
        m = hgb_reg().fit(Xtr, ctx.y.values[f['tr']])
        filas.append(ctx.metricas_de_pronostico(f['va'], m.predict(Xva)))
    return acumular(filas)


def exp_techo_marginalizado(ctx):
    """Oráculo entrenado con active_products y marginalizado sobre su distribución al predecir.

    Al predecir NO usa active_products de la fila: promedia el pronóstico sobre los 5 niveles con
    las frecuencias del pliegue de entrenamiento. Estima el mejor predictor alcanzable sin la
    variable (y es, a la vez, un modelo LUPI: información privilegiada solo en entrenamiento).
    """
    filas = []
    for f in ctx.P:
        ap_tr = ctx.ap.values[f['tr']]
        m = hgb_reg().fit(np.column_stack([f['Xtr'], ap_tr]), ctx.y.values[f['tr']])
        niveles, cuentas = np.unique(ap_tr, return_counts=True)
        pesos = cuentas / cuentas.sum()
        pred = sum(w * m.predict(np.column_stack([f['Xva'], np.full(len(f['va']), k)]))
                   for k, w in zip(niveles, pesos))
        filas.append(ctx.metricas_de_pronostico(f['va'], pred))
    return acumular(filas)


def exp_lupi_objetivo_depurado(ctx, ctor):
    """LUPI: se resta al objetivo la media de su nivel de active_products (estimada en el pliegue de
    entrenamiento) y se ajusta el modelo SIN esa variable sobre el residuo."""
    filas = []
    for f in ctx.P:
        y_tr, ap_tr = ctx.y.values[f['tr']], ctx.ap.values[f['tr']]
        escalera = pd.Series(y_tr).groupby(ap_tr).mean()
        residuo = y_tr - escalera.loc[ap_tr].values
        m = ctor().fit(f['Xtr'], residuo)
        filas.append(ctx.metricas_de_pronostico(f['va'], m.predict(f['Xva']) + y_tr.mean()))
    return acumular(filas)


def exp_xgb_optuna(ctx, trials: int):
    """XGBoost afinado con Optuna (TPE) dentro de cada pliegue externo: CV interna de 3."""
    def espacio(t):
        return dict(n_estimators=t.suggest_int('n_estimators', 200, 900, step=100),
                    learning_rate=t.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    max_depth=t.suggest_int('max_depth', 2, 6),
                    min_child_weight=t.suggest_float('min_child_weight', 1, 60, log=True),
                    subsample=t.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
                    reg_lambda=t.suggest_float('reg_lambda', 0.1, 30, log=True),
                    gamma=t.suggest_float('gamma', 0.0, 5.0))
    filas, mejores = [], []
    for f in ctx.P:
        fila = {}
        for clave, y, ctor, scoring, cv in (
                ('auc_med', ctx.y_med, xgb_clf, 'roc_auc', StratifiedKFold(3, shuffle=True, random_state=SEED)),
                ('r2', ctx.y, xgb_reg, 'r2', KFold(3, shuffle=True, random_state=SEED))):
            y_tr = y.values[f['tr']]
            estudio = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
            estudio.optimize(lambda t: cross_val_score(ctor(**espacio(t)), f['Xtr'], y_tr, cv=cv,
                                                       scoring=scoring).mean(), n_trials=trials)
            m = ctor(**estudio.best_params).fit(f['Xtr'], y_tr)
            if clave == 'auc_med':
                fila[clave] = float(roc_auc_score(y.values[f['va']], puntaje(m, f['Xva'])))
            else:
                fila[clave] = float(r2_score(y.values[f['va']], m.predict(f['Xva'])))
            mejores.append({clave: estudio.best_params, 'interno': estudio.best_value})
        filas.append(fila)
    return {**acumular(filas), 'mejores_params': mejores}


def exp_seleccion(ctx, k: int):
    """HGB con las k variables de mayor |Spearman| con el objetivo, elegidas dentro del pliegue."""
    filas = []
    for f in ctx.P:
        y_tr = ctx.y.values[f['tr']]
        rho = np.array([abs(stats.spearmanr(f['Xtr'][:, j], y_tr).statistic) for j in range(f['Xtr'].shape[1])])
        idx = np.argsort(-np.nan_to_num(rho))[:k]
        fila = {}
        for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
            m = hgb_clf().fit(f['Xtr'][:, idx], y.values[f['tr']])
            fila[nombre] = float(roc_auc_score(y.values[f['va']], puntaje(m, f['Xva'][:, idx])))
        m = hgb_reg().fit(f['Xtr'][:, idx], y_tr)
        fila['r2'] = float(r2_score(ctx.y.values[f['va']], m.predict(f['Xva'][:, idx])))
        fila['variables'] = [f['nombres'][j] for j in idx]
        filas.append(fila)
    sal = acumular([{k_: v for k_, v in fila.items() if k_ != 'variables'} for fila in filas])
    sal['variables_pliegue_0'] = filas[0]['variables']
    return sal


def exp_splines(ctx):
    """Modelo de clase + cambio: logística (y Ridge) sobre bases de splines cúbicos = GAM."""
    numericas, categoricas = columnas_por_tipo(ctx.X)
    continuas = [c for c in numericas if ctx.X[c].nunique() > 2]
    binarias = [c for c in numericas if c not in continuas]
    pre = ColumnTransformer([
        ('spl', make_pipeline(StandardScaler(), SplineTransformer(n_knots=6, degree=3, include_bias=False)), continuas),
        ('bin', 'passthrough', binarias),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categoricas)])
    filas = []
    for tr, va in ctx.pliegues:
        pr = clone(pre).fit(ctx.X.iloc[tr])
        Xtr, Xva = pr.transform(ctx.X.iloc[tr]).astype(float), pr.transform(ctx.X.iloc[va]).astype(float)
        fila = {}
        for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
            m = LogisticRegression(C=0.5, max_iter=3000).fit(Xtr, y.values[tr])
            fila[nombre] = float(roc_auc_score(y.values[va], m.predict_proba(Xva)[:, 1]))
        m = Ridge(alpha=5.0).fit(Xtr, ctx.y.values[tr])
        fila['r2'] = float(r2_score(ctx.y.values[va], m.predict(Xva)))
        filas.append(fila)
    return acumular(filas)


def exp_gbdt_lr(ctx):
    """Híbrido GBDT + logística (He et al., 2014): las hojas de XGBoost, codificadas como
    indicadoras, son las variables de una regresión logística ajustada en otra mitad."""
    filas = []
    for f in ctx.P:
        fila = {}
        for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
            y_tr = y.values[f['tr']]
            Xa, Xb, ya, yb = train_test_split(f['Xtr'], y_tr, test_size=0.5, stratify=y_tr, random_state=SEED)
            arb = xgb_clf(n_estimators=150, max_depth=3, learning_rate=0.1).fit(Xa, ya)
            oh = OneHotEncoder(handle_unknown='ignore').fit(arb.apply(Xa))
            lr = LogisticRegression(C=0.05, max_iter=3000).fit(oh.transform(arb.apply(Xb)), yb)
            s = lr.predict_proba(oh.transform(arb.apply(f['Xva'])))[:, 1]
            fila[nombre] = float(roc_auc_score(y.values[f['va']], s))
        filas.append(fila)
    return acumular(filas)


def exp_dart(ctx):
    return clasificador_directo(ctx, lambda: xgb_clf(booster='dart', rate_drop=0.1, skip_drop=0.5))


def exp_nystroem(ctx):
    """SVM de clase + cambio: núcleo RBF aproximado con Nyström (600 componentes) y modelo lineal."""
    filas = []
    for f in ctx.P:
        ny = Nystroem(kernel='rbf', gamma=1.0 / f['Xtr'].shape[1], n_components=600, random_state=SEED)
        Ztr, Zva = ny.fit_transform(f['Xtr']), None
        Zva = ny.transform(f['Xva'])
        fila = {}
        for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
            m = LinearSVC(C=0.5, dual='auto', max_iter=5000).fit(Ztr, y.values[f['tr']])
            fila[nombre] = float(roc_auc_score(y.values[f['va']], m.decision_function(Zva)))
        m = Ridge(alpha=1.0).fit(Ztr, ctx.y.values[f['tr']])
        fila['r2'] = float(r2_score(ctx.y.values[f['va']], m.predict(Zva)))
        filas.append(fila)
    return acumular(filas)


def exp_blending(ctx):
    """Mezcla: promedio de rangos (clasificación) o de pronósticos (regresión) de cuatro boosting."""
    def clfs():
        return [xgb_clf(), hgb_clf(),
                lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=SEED,
                                   verbose=-1),
                cb.CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, thread_count=N_JOBS,
                                      random_seed=SEED, verbose=0, allow_writing_files=False)]

    def regs():
        return [xgb_reg(), hgb_reg(),
                lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=SEED,
                                  verbose=-1),
                cb.CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, thread_count=N_JOBS,
                                     random_seed=SEED, verbose=0, allow_writing_files=False)]
    filas = []
    for f in ctx.P:
        fila = {}
        for nombre, y in (('auc_med', ctx.y_med), ('auc_p75', ctx.y_p75)):
            rangos = [rankdata(puntaje(m.fit(f['Xtr'], y.values[f['tr']]), f['Xva'])) for m in clfs()]
            fila[nombre] = float(roc_auc_score(y.values[f['va']], np.mean(rangos, axis=0)))
        pred = np.mean([m.fit(f['Xtr'], ctx.y.values[f['tr']]).predict(f['Xva']) for m in regs()], axis=0)
        fila['r2'] = float(r2_score(ctx.y.values[f['va']], pred))
        filas.append(fila)
    return acumular(filas)


# --------------------------------------------------------------------------- salida
def tabla_markdown(res: dict, folds: int) -> str:
    ref = res.get('xgboost directo [clase, referencia]', {})
    lin = [f'# Ronda 2: techo de información y variantes ({folds} pliegues sobre entrenamiento)', '',
           'Mismos pliegues para todas las filas (estratificados por la etiqueta de la mediana, semilla 42). '
           'La columna Δ es la diferencia media pareada frente a XGBoost directo y p su prueba t pareada.', '',
           '| Experimento | Tipo | AUC mediana | Δ (p) | AUC percentil 75 | R² | Δ (p) | min |',
           '|---|---|---|---|---|---|---|---|']

    def celda(v):
        return f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}' if v else '—'

    def delta(v, r):
        if not v or not r or v is r:
            return '—'
        d = np.array(v) - np.array(r)
        p = stats.ttest_rel(v, r).pvalue
        return f'{d.mean():+.4f} ({p:.3f})'
    for nombre, r in res.items():
        lin.append(f'| {nombre} | {r.get("tipo", "")} | {celda(r.get("auc_med"))} | '
                   f'{delta(r.get("auc_med"), ref.get("auc_med"))} | {celda(r.get("auc_p75"))} | '
                   f'{celda(r.get("r2"))} | {delta(r.get("r2"), ref.get("r2"))} | {r.get("minutos", 0):.1f} |')
    return '\n'.join(lin) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--trials', type=int, default=25)
    ap.add_argument('--salida', default=str(RAIZ / 'exploracion'))
    args = ap.parse_args()
    salida = Path(args.salida)
    fijar_semillas(SEED)
    t0 = time.perf_counter()
    ctx = Contexto(args.folds)
    print(f'datos listos en {time.perf_counter() - t0:.0f} s; X {ctx.X.shape}; '
          f'niveles de active_products {sorted(ctx.ap.unique())}', flush=True)

    experimentos = [
        ('xgboost directo [clase, referencia]', 'referencia', lambda: exp_referencia(ctx, 'xgboost')),
        ('hgb directo [referencia]', 'referencia', lambda: exp_referencia(ctx, 'hgb')),
        ('ORACULO con active_products (diagnostico, no candidato)', 'diagnóstico', lambda: exp_oraculo(ctx)),
        ('TECHO: oraculo marginalizado sobre active_products (LUPI)', 'techo / LUPI',
         lambda: exp_techo_marginalizado(ctx)),
        ('LUPI objetivo depurado + hgb', 'LUPI', lambda: exp_lupi_objetivo_depurado(ctx, hgb_reg)),
        ('LUPI objetivo depurado + xgboost', 'LUPI', lambda: exp_lupi_objetivo_depurado(ctx, xgb_reg)),
        ('seleccion de variables k=6 + hgb', 'clase + cambio', lambda: exp_seleccion(ctx, 6)),
        ('seleccion de variables k=12 + hgb', 'clase + cambio', lambda: exp_seleccion(ctx, 12)),
        ('seleccion de variables k=24 + hgb', 'clase + cambio', lambda: exp_seleccion(ctx, 24)),
        ('logistica / ridge con splines (GAM)', 'clase + cambio', lambda: exp_splines(ctx)),
        ('hibrido GBDT + logistica', 'clase + cambio', lambda: exp_gbdt_lr(ctx)),
        ('xgboost DART', 'clase + cambio', lambda: exp_dart(ctx)),
        ('SVM RBF via Nystroem', 'clase + cambio', lambda: exp_nystroem(ctx)),
        ('blending xgb+hgb+lgbm+catboost', 'mezcla', lambda: exp_blending(ctx)),
        (f'xgboost afinado con Optuna ({args.trials} trials)', 'clase + cambio',
         lambda: exp_xgb_optuna(ctx, args.trials)),
    ]
    res: dict = {}
    for nombre, tipo, fn in experimentos:
        t1 = time.perf_counter()
        try:
            r = fn()
        except Exception as e:                                           # noqa: BLE001
            print(f'  {nombre}: ERROR {type(e).__name__}: {e}', flush=True)
            continue
        r.update({'tipo': tipo, 'minutos': (time.perf_counter() - t1) / 60})
        res[nombre] = r
        (salida / 'benchmark_ronda2.json').write_text(
            json.dumps({'folds': args.folds, 'seed': SEED, 'resultados': res}, ensure_ascii=False, indent=1,
                       default=str), encoding='utf-8')
        (salida / 'benchmark_ronda2.md').write_text(tabla_markdown(res, args.folds), encoding='utf-8')
        partes = [f'{k} {np.mean(r[k]):.4f}' for k in ('auc_med', 'auc_p75', 'r2') if r.get(k)]
        print(f'  RESULTADO {nombre:58s} ' + ' | '.join(partes) + f'  [{r["minutos"]:.1f} min]', flush=True)
    print(f'\nterminado ronda 2 en {(time.perf_counter() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
