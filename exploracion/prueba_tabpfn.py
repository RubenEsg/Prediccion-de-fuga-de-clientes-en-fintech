"""Prueba de viabilidad de TabPFN como *modelo nuevo* (Entregable 3).

TabPFN es un transformador preentrenado sobre millones de conjuntos sintéticos que clasifica o
regresa *en contexto* (sin descenso de gradiente sobre nuestros datos): no aparece en las notas
del curso. Este guion mide su AUC y su R² sobre los mismos pliegues que
``benchmark_modelo_nuevo.py`` y, sobre todo, cuánto tarda en esta máquina (solo CPU), porque
ese es el factor que decide si cabe en el pipeline combinatorio de la guía.

Uso
---
    .venv/Scripts/python.exe exploracion/prueba_tabpfn.py --pliegues 1 --submuestra 10000 --n-estimators 2
    .venv/Scripts/python.exe exploracion/prueba_tabpfn.py --pliegues 5 --submuestra 0 --n-estimators auto

``--submuestra 0`` usa todo el pliegue de entrenamiento (~31.000 filas). La regresión se evalúa
en los mismos pliegues; sus pronósticos se usan además como puntaje de clasificación para las dos
etiquetas (mediana y percentil 75), lo que da tres AUC por un solo ajuste.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import datos  # noqa: E402
from src.config import SEED, fijar_semillas  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pliegues', type=int, default=1, help='cuántos de los 5 pliegues correr')
    ap.add_argument('--submuestra', type=int, default=10000, help='filas del pliegue de entrenamiento; 0 = todas')
    ap.add_argument('--n-estimators', default='2', help="entero o 'auto'")
    ap.add_argument('--device', default='auto')
    ap.add_argument('--sin-regresion', action='store_true')
    ap.add_argument('--salida', default=str(RAIZ / 'exploracion' / 'prueba_tabpfn.json'))
    args = ap.parse_args()
    n_est = args.n_estimators if args.n_estimators == 'auto' else int(args.n_estimators)

    from tabpfn import TabPFNClassifier, TabPFNRegressor  # importación tardía: torch tarda en cargar

    fijar_semillas(SEED)
    p = datos.preparar_todo(metodo_etiqueta='mediana')
    c = p['conjuntos']
    X = c.X_train.reset_index(drop=True)
    y_cont = c.y_train_cont.reset_index(drop=True)
    y_med = p['y_train'].reset_index(drop=True)
    y_p75 = datos.construir_etiqueta(c.y_train_cont, metodo='percentil', q=0.75).binarizar(c.y_train_cont)
    y_p75 = y_p75.reset_index(drop=True)
    pliegues = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y_med))[:args.pliegues]
    print(f'X_train {X.shape}; submuestra={args.submuestra or "todas"}; n_estimators={n_est}; '
          f'device={args.device}', flush=True)

    rng = np.random.default_rng(SEED)
    filas = []
    for k, (tr, va) in enumerate(pliegues):
        if args.submuestra and args.submuestra < len(tr):
            tr = rng.choice(tr, size=args.submuestra, replace=False)
        fila = {'pliegue': k, 'n_train': int(len(tr)), 'n_val': int(len(va))}

        clf = TabPFNClassifier(n_estimators=n_est, device=args.device, random_state=SEED,
                               ignore_pretraining_limits=True)
        t0 = time.perf_counter(); clf.fit(X.iloc[tr], y_med.iloc[tr]); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); s = clf.predict_proba(X.iloc[va])[:, 1]; t_pred = time.perf_counter() - t0
        fila.update({'auc_mediana_clf': float(roc_auc_score(y_med.iloc[va], s)),
                     't_fit_clf_s': t_fit, 't_predict_clf_s': t_pred})
        print(f'pliegue {k}: clasificador AUC(mediana) {fila["auc_mediana_clf"]:.4f}  '
              f'[fit {t_fit:.0f} s, predict {t_pred:.0f} s]', flush=True)

        if not args.sin_regresion:
            reg = TabPFNRegressor(n_estimators=n_est, device=args.device, random_state=SEED,
                                  ignore_pretraining_limits=True)
            t0 = time.perf_counter(); reg.fit(X.iloc[tr], y_cont.iloc[tr]); t_fit = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = reg.predict(X.iloc[va]); t_pred = time.perf_counter() - t0
            fila.update({'r2_reg': float(r2_score(y_cont.iloc[va], pred)),
                         'rmse_reg': float(root_mean_squared_error(y_cont.iloc[va], pred)),
                         'auc_mediana_reg2rank': float(roc_auc_score(y_med.iloc[va], pred)),
                         'auc_p75_reg2rank': float(roc_auc_score(y_p75.iloc[va], pred)),
                         't_fit_reg_s': t_fit, 't_predict_reg_s': t_pred})
            print(f'pliegue {k}: regresor R2 {fila["r2_reg"]:.4f}, RMSE {fila["rmse_reg"]:.4f}; '
                  f'reg→rank AUC mediana {fila["auc_mediana_reg2rank"]:.4f}, '
                  f'p75 {fila["auc_p75_reg2rank"]:.4f}  [fit {t_fit:.0f} s, predict {t_pred:.0f} s]', flush=True)
        filas.append(fila)
        Path(args.salida).write_text(json.dumps({'config': vars(args), 'pliegues': filas}, indent=1,
                                                ensure_ascii=False), encoding='utf-8')
    print('terminado TabPFN', flush=True)


if __name__ == '__main__':
    main()
