"""Estabilidad por semilla y cómputo: cada modelo final se reajusta con diez semillas y se evalúa en prueba.

Los hiperparámetros son los finales de la semilla 42 (tabla maestra); solo cambia la semilla del modelo, de 42 a 51.
Los modelos deterministas (k-NN, Ridge, Naive Bayes…) dan el mismo valor las diez veces, y así se reporta. De paso se
mide, en el mismo equipo y para los diez modelos, el tiempo de ajuste, el de inferencia sobre la prueba y el tamaño
del modelo serializado.

Salida: ``resultados/semillas.csv``. Uso: ``python experiments/03b_semillas.py``. Se puede interrumpir y relanzar.
"""
import io
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import evaluacion, plan                                           # noqa: E402
from src.config import CARPETA_RESULTADOS, N_SEMILLAS, SEED                # noqa: E402
from src.variantes import reajustar                                        # noqa: E402


def main() -> None:
    p = plan.problema()
    c = p['conjuntos']
    tabla = plan.registro().cargar()
    ruta = CARPETA_RESULTADOS / 'semillas.csv'
    hechas = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=['tarea', 'modelo', 'semilla'])
    filas = hechas.to_dict('records')
    listas = {(f['tarea'], f['modelo'], int(f['semilla'])) for f in filas}
    for tarea in ('regresion', 'clasificacion'):
        X, y, X_te, y_te = plan.objetivo(tarea, p)
        for modelo in plan.modelos(tarea):
            balanceo, optimizador = plan.combinacion(tarea, modelo)
            f = tabla[(tabla.tarea == tarea) & (tabla.modelo == modelo) & (tabla.balanceo == balanceo)
                      & (tabla.optimizador == optimizador) & (tabla.semilla == SEED)]
            if f.empty:
                print(f'{tarea} · {modelo}: falta la corrida de la semilla {SEED}', flush=True)
                continue
            hiper = f.iloc[0]['hiperparametros']
            hiper = json.loads(hiper) if isinstance(hiper, str) else dict(hiper)
            for semilla in range(SEED, SEED + N_SEMILLAS):
                if (tarea, modelo, semilla) in listas:
                    continue
                t0 = time.perf_counter()
                tuberia = reajustar(tarea, modelo, balanceo, hiper, X, y, semilla, puntaje=c.y_train_cont)
                t_ajuste = time.perf_counter() - t0
                t0 = time.perf_counter()
                m = evaluacion.evaluar(tuberia, X_te, y_te, tarea)
                t_inferencia = time.perf_counter() - t0
                memoria = io.BytesIO()
                joblib.dump(tuberia, memoria, compress=3)
                filas.append({'tarea': tarea, 'modelo': modelo, 'semilla': semilla, 'metrica': m[evaluacion.NOMBRE_METRICA[tarea]],
                              'ajuste_s': t_ajuste, 'inferencia_s': t_inferencia, 'tamano_kb': memoria.getbuffer().nbytes / 1024,
                              **{k: v for k, v in m.items() if k in ('mae', 'r2', 'ap', 'brier', 'logloss')}})
                pd.DataFrame(filas).to_csv(ruta, index=False)
            t = pd.DataFrame(filas).query('tarea == @tarea and modelo == @modelo')
            print(f'{tarea} · {modelo}: {t["metrica"].mean():.6f} ± {t["metrica"].std(ddof=1):.6f} '
                  f'(ajuste {t["ajuste_s"].mean():.1f} s, inferencia {t["inferencia_s"].mean():.2f} s, {t["tamano_kb"].mean():.0f} KB)', flush=True)


if __name__ == '__main__':
    main()
