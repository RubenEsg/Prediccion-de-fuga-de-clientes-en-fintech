"""Los dos métodos recientes con los hiperparámetros que recomiendan sus autores, frente a los ajustados.

La guía (§6.2) pide que los benchmarks recientes partan de sus hiperparámetros recomendados. En el proyecto la búsqueda
de cada benchmark recorre un espacio que contiene casi todos esos valores, pero no empieza en ellos. Este script mide
ese punto de partida: ajusta la EBM y LightGBM con los valores por defecto de su librería, que son los que recomiendan
sus autores, en los mismos cinco pliegues externos de la semilla 42 y en la partición de prueba, con el mismo
preprocesamiento y el mismo tratamiento del desbalance que la combinación elegida para cada uno. El informe los compara
con la versión ajustada (``resultados/pliegues_<tarea>.csv`` y ``resultados/metricas_prueba.csv``).

Es un análisis añadido después de ver los resultados (libro, sección 6.2): no cambia qué benchmark entra en las pruebas
de hipótesis ni ninguna cifra del protocolo.

Salida: ``resultados/recomendados.csv``. Uso: ``python experiments/03d_recomendados.py``. Se puede interrumpir y
relanzar: salta lo ya medido.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')      # dos hilos: no competir con otras tareas del equipo

import numpy as np                                                         # noqa: E402
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import balanceo as bal                                            # noqa: E402
from src import plan
from src.config import CARPETA_RESULTADOS, K_EXT, SEED
from src.modelos import catalogo
from src.optimizacion import construir_cv

RECIENTES = ('lightgbm', 'ebm')
EBM_BUSCADOS = ('learning_rate', 'max_leaves', 'min_samples_leaf', 'interactions', 'max_bins', 'outer_bags')
LGBM_BUSCADOS = ('n_estimators', 'num_leaves', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda', 'min_child_samples')


def recomendados(tarea: str, modelo: str) -> dict:
    """Valores por defecto de la librería para los hiperparámetros que la búsqueda del proyecto mueve (o fija)."""
    if modelo == 'ebm':
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
        defecto = (ExplainableBoostingClassifier if tarea == 'clasificacion' else ExplainableBoostingRegressor)().get_params()
        return {f'modelo__{k}': defecto[k] for k in EBM_BUSCADOS}
    from lightgbm import LGBMClassifier, LGBMRegressor
    defecto = (LGBMClassifier if tarea == 'clasificacion' else LGBMRegressor)().get_params()
    return {f'modelo__{k}': defecto[k] for k in LGBM_BUSCADOS}


def metrica(tarea: str, tuberia, X, y) -> float:
    if tarea == 'clasificacion':
        return float(roc_auc_score(y, tuberia.predict_proba(X)[:, 1]))
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - tuberia.predict(X)) ** 2)))


def ajustar(tarea: str, modelo: str, balanceo: str, hiper: dict, X, y):
    spec = catalogo(tarea, incluir_nuevo=True, incluir_original=True)[modelo]
    y = pd.Series(np.asarray(y))
    tuberia = bal.construir_pipeline(spec, balanceo, X, SEED, y if tarea == 'clasificacion' else None)
    tuberia.set_params(**hiper, modelo__n_jobs=2)
    return tuberia.fit(X, y)


def main() -> None:
    ruta = CARPETA_RESULTADOS / 'recomendados.csv'
    filas = pd.read_csv(ruta).to_dict('records') if ruta.exists() else []
    hechas = {(f['tarea'], f['modelo'], str(f['pliegue'])) for f in filas}
    p = plan.problema()
    for tarea in ('regresion', 'clasificacion'):
        X, y, X_te, y_te = plan.objetivo(tarea, p)
        y = pd.Series(np.asarray(y), index=X.index)
        pliegues = list(construir_cv(tarea, K_EXT, SEED).split(X, y))
        for modelo in RECIENTES:
            balanceo, _ = plan.combinacion(tarea, modelo)
            hiper = recomendados(tarea, modelo)
            texto = json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in hiper.items()}, ensure_ascii=False)
            partes = [(str(i), X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]) for i, (tr, va) in enumerate(pliegues)]
            partes.append(('prueba', X, y, X_te, y_te))
            for nombre, X_a, y_a, X_v, y_v in partes:
                if (tarea, modelo, nombre) in hechas:
                    continue
                t0 = time.perf_counter()
                tuberia = ajustar(tarea, modelo, balanceo, hiper, X_a, y_a)
                segundos = time.perf_counter() - t0
                filas.append({'tarea': tarea, 'modelo': modelo, 'balanceo': balanceo, 'hiperparametros': texto, 'pliegue': nombre,
                              'metrica': metrica(tarea, tuberia, X_v, y_v), 'ajuste_s': segundos})
                pd.DataFrame(filas).to_csv(ruta, index=False)
                print(f'{tarea} · {modelo} · {nombre}: {filas[-1]["metrica"]:.5f} ({segundos:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
