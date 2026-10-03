"""Tiempo de ajuste e inferencia del modelo final de cada modelo, en las mismas condiciones.

Para el artículo del Entregable 3. Toma la mejor combinación (balanceo × optimizador según el bucle
externo) de cada uno de los 8 + 8 modelos, incluida la EBM, carga su pipeline final guardado, lo
reajusta desde cero sobre todo el entrenamiento (38.978 clientes) y mide la mediana de 3 ajustes y de 5
inferencias sobre la prueba (9.745 clientes), además del tamaño serializado y la métrica de prueba del
reajuste. Debe correr con el equipo sin otra carga. Escribe ``resultados/computo/tiempos_finales.csv``.

Uso::

    .venv\\Scripts\\python.exe exploracion/medir_tiempos_finales.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import analisis, datos  # noqa: E402

SALIDA = RAIZ / 'resultados' / 'computo' / 'tiempos_finales.csv'
REPETICIONES_AJUSTE, REPETICIONES_INFERENCIA = 3, 5


def puntuar(modelo, tarea: str, X: pd.DataFrame) -> np.ndarray:
    if tarea == 'regresion':
        return np.asarray(modelo.predict(X), dtype=float)
    if hasattr(modelo, 'predict_proba'):
        return np.asarray(modelo.predict_proba(X))[:, 1]
    return np.asarray(modelo.decision_function(X), dtype=float)


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = datos.preparar_todo()
    c = p['conjuntos']
    objetivos = {'clasificacion': (p['y_train'], p['y_test']), 'regresion': (c.y_train_cont, c.y_test_cont)}
    tabla = analisis.cargar_tabla()
    filas = []
    for tarea in ('clasificacion', 'regresion'):
        y_tr, y_te = objetivos[tarea]
        for _, f in analisis.mejores_por_modelo(tabla, tarea).iterrows():
            guardado = joblib.load(f['ruta_modelo'])
            ajustes = []
            for _ in range(REPETICIONES_AJUSTE):
                modelo = clone(guardado)
                t0 = time.perf_counter()
                modelo.fit(c.X_train, y_tr)
                ajustes.append(time.perf_counter() - t0)
            inferencias = []
            for _ in range(REPETICIONES_INFERENCIA):
                t0 = time.perf_counter()
                puntajes = puntuar(modelo, tarea, c.X_test)
                inferencias.append(time.perf_counter() - t0)
            if tarea == 'clasificacion':
                metrica = roc_auc_score(y_te, puntajes)
            else:
                metrica = float(np.sqrt(mean_squared_error(y_te, puntajes)))
            buf = io.BytesIO()
            joblib.dump(modelo, buf)
            fila = {'tarea': tarea, 'modelo': f['modelo'], 'balanceo': f['balanceo'], 'optimizador': f['optimizador'],
                    'ajuste_s': float(np.median(ajustes)), 'ajuste_desv_s': float(np.std(ajustes)),
                    'inferencia_s': float(np.median(inferencias)), 'tamano_modelo_mb': buf.getbuffer().nbytes / 2 ** 20,
                    'metrica_prueba': float(metrica), 'metrica_prueba_registrada': float('nan'),
                    'n_train': len(c.X_train), 'n_test': len(c.X_test)}
            filas.append(fila)
            print(f"{tarea:13s} {f['modelo']:14s} ajuste {fila['ajuste_s']:8.2f} s  inferencia {fila['inferencia_s']:7.3f} s  "
                  f"métrica de prueba {metrica:.4f}  ({fila['tamano_modelo_mb']:.1f} MB)", flush=True)
    salida = pd.DataFrame(filas)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(SALIDA, index=False)
    print(f'{SALIDA}: {len(salida)} filas')


if __name__ == '__main__':
    main()
