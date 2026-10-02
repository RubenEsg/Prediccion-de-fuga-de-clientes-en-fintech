"""Mide el coste real de un ajuste de cada modelo pendiente y estima la duración de las corridas.

Se ajusta cada modelo sobre un pliegue interno de tamaño real (≈ 20.800 filas), en serie y con
un solo hilo por ajuste (Random Forest con sus hilos, como en el pipeline), para varias
configuraciones muestreadas del espacio de búsqueda. Con esos tiempos se estima cada corrida
anidada según cómo paraleliza cada optimizador:

- Grid y Random: scikit-learn reparte configuraciones × pliegues entre los 6 núcleos.
- Bayesiana y Genética: evalúan una configuración cada vez; solo los 3 pliegues internos van en
  paralelo, así que usan la mitad de los núcleos salvo en Random Forest (hilos propios).

Uso: ``.venv/Scripts/python.exe exploracion/medir_costes_140.py``. No toca la tabla maestra.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings('ignore')

from src import balanceo as bal  # noqa: E402
from src import datos, modelos  # noqa: E402
from src.config import K_EXT, K_INT, SEED  # noqa: E402
from src.optimizacion import distribuciones_random, presupuesto_por_defecto  # noqa: E402

N_INTERNO = 20788          # filas de entrenamiento de un pliegue interno (38.978 × 4/5 × 2/3)
N_VALIDACION = 10394
NUCLEOS = 6
MUESTRAS = 3               # configuraciones medidas por modelo
FACTOR_TAMANO = 1.15       # la búsqueda final y los reajustes usan pliegues algo mayores
FACTOR_SOBREMUESTREO = 1.5 # SMOTE/ADASYN agrandan el pliegue ~1,5 veces con la etiqueta p75

PENDIENTES = {
    'regresion': ['svr', 'xgboost', 'knn', 'random_forest'],
    'clasificacion': ['naive_bayes', 'logistica', 'arbol', 'svm', 'xgboost', 'knn', 'random_forest'],
}


def medir(spec, X, y, Xv, yv, configs, fijos):
    """Segundos (media) de ajuste + predicción en serie para las configuraciones dadas."""
    tiempos = []
    for cfg in configs:
        pipe = bal.construir_pipeline(spec, 'ninguno', X, SEED, y)
        pipe.set_params(**{**cfg, **(fijos or {})})
        est = pipe.named_steps['modelo']
        if 'n_jobs' in est.get_params() and spec.nombre != 'random_forest':
            est.set_params(n_jobs=1)            # en los pliegues cada ajuste corre con un hilo
        t0 = time.perf_counter()
        pipe.fit(X, y)
        (pipe.predict_proba if hasattr(pipe, 'predict_proba') else pipe.predict)(Xv)
        tiempos.append(time.perf_counter() - t0)
    return float(np.mean(tiempos)), [round(t, 1) for t in tiempos]


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    p = datos.preparar_todo()
    c = p['conjuntos']
    X, Xv = c.X_train.iloc[:N_INTERNO], c.X_train.iloc[N_INTERNO:N_INTERNO + N_VALIDACION]
    objetivos = {'clasificacion': (p['y_train'].iloc[:N_INTERNO], p['y_train'].iloc[N_INTERNO:N_INTERNO + N_VALIDACION]),
                 'regresion': (c.y_train_cont.iloc[:N_INTERNO], c.y_train_cont.iloc[N_INTERNO:N_INTERNO + N_VALIDACION])}
    rng = np.random.RandomState(SEED)
    filas, total_h = [], 0.0
    for tarea, nombres in PENDIENTES.items():
        y, yv = objetivos[tarea]
        cat = modelos.catalogo(tarea)
        for nombre in nombres:
            spec = cat[nombre]
            P = presupuesto_por_defecto(spec.espacio_grid)
            fijos = (spec.fidelidad or {}).get('busqueda')
            rejilla = list(ParameterGrid(spec.espacio_grid))
            cfg_grid = [rejilla[i] for i in rng.choice(len(rejilla), size=min(MUESTRAS, len(rejilla)), replace=False)]
            cfg_esp = list(ParameterSampler(distribuciones_random(spec.espacio), n_iter=MUESTRAS, random_state=SEED))
            t_grid, det_g = medir(spec, X, y, Xv, yv, cfg_grid, fijos)
            t_esp, det_e = medir(spec, X, y, Xv, yv, cfg_esp, fijos)
            hilos = spec.nombre == 'random_forest'     # RF usa todos los núcleos por sí mismo
            ajustes = (K_EXT + 1) * P * K_INT         # ajustes de la búsqueda (5 externas + final)
            # grid/random: paralelismo completo; bayes/genético: 3 pliegues a la vez (o hilos en RF)
            par_seq = NUCLEOS if hilos else K_INT
            seg = {'grid': ajustes * t_grid / NUCLEOS,
                   'random': ajustes * t_esp / NUCLEOS,
                   'bayesiana': ajustes * t_esp / par_seq,
                   'genetica': ajustes * t_esp / par_seq}
            reajuste = (K_EXT + 1) * t_esp * 1.6 * (3 if fijos else 1) / (NUCLEOS if hilos else 1)
            n_bal = 1 if tarea == 'regresion' else (3 if nombre == 'knn' else 4)
            mult = 1.0 if tarea == 'regresion' else ((1 + 2 * FACTOR_SOBREMUESTREO) / 3 if nombre == 'knn'
                                                      else (2 + 2 * FACTOR_SOBREMUESTREO) / 4)
            horas = sum((s + reajuste) * FACTOR_TAMANO for s in seg.values()) * n_bal * mult / 3600
            total_h += horas
            filas.append({'tarea': tarea, 'modelo': nombre, 'P': P, 's_ajuste_rejilla': round(t_grid, 2),
                          's_ajuste_espacio': round(t_esp, 2), 'detalle_rejilla': det_g, 'detalle_espacio': det_e,
                          'corridas': n_bal * 4, 'horas': round(horas, 2)})
            print(f'{tarea:13s} {nombre:13s} P={P:2d}  s/ajuste rejilla {t_grid:6.2f} {det_g}  espacio {t_esp:6.2f} '
                  f'{det_e}  -> {n_bal * 4:2d} corridas ≈ {horas:5.2f} h', flush=True)
    print(f'\nTOTAL pendiente ≈ {total_h:.1f} h')
    (RAIZ / 'exploracion' / 'medir_costes_140.json').write_text(
        json.dumps({'total_horas': round(total_h, 1), 'filas': filas}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
