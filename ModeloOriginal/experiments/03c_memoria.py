"""Memoria que usa cada modelo final al ajustarse y predecir, y tiempo de inferencia del modelo original según las filas.

La guía pide reportar la memoria por modelo, además del tiempo. Cada modelo final se reajusta con sus hiperparámetros
finales **en un proceso aparte**, para que la medida no arrastre lo que dejó el modelo anterior: el proceso carga los
datos, anota la memoria que ocupa y, mientras ajusta el modelo y predice la partición de prueba, un hilo la vuelve a
leer cada cinco milisegundos. La diferencia entre el máximo observado y la memoria inicial es la atribuible al modelo.
Se mide la memoria residente del proceso, que incluye lo que reservan las librerías compiladas, no solo los objetos de
Python; un pico más breve que el intervalo de muestreo puede no verse.

De paso se mide cómo crece el tiempo de inferencia del modelo original y del mejor benchmark de regresión con el número
de filas (la partición de prueba repetida hasta completar cada tamaño; mediana de cinco medidas).

Todas las medidas se hacen con dos hilos de cálculo (``OMP_NUM_THREADS = 2``), para que no dependan de lo que esté
ejecutando el equipo: con todos los núcleos y otra tarea en marcha, los modelos de boosting se frenan entre sí.

Salidas: ``resultados/memoria.csv`` e ``resultados/inferencia_filas.csv``. Uso: ``python experiments/03c_memoria.py``.
Se puede interrumpir y relanzar: salta los modelos ya medidos.
"""
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')      # dos hilos de cálculo: la medida no debe depender de la carga del equipo

import numpy as np                                                         # noqa: E402
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import plan                                                       # noqa: E402
from src.config import CARPETA_RESULTADOS, SEED                            # noqa: E402
from src.variantes import reajustar                                        # noqa: E402

FILAS_INFERENCIA = (1000, 10000, 100000)


def configuracion(tabla: pd.DataFrame, tarea: str, modelo: str) -> tuple[str, dict]:
    balanceo, optimizador = plan.combinacion(tarea, modelo)
    f = tabla[(tabla.tarea == tarea) & (tabla.modelo == modelo) & (tabla.balanceo == balanceo) & (tabla.optimizador == optimizador)
              & (tabla.semilla == SEED)].iloc[0]
    hiper = f['hiperparametros']
    return balanceo, json.loads(hiper) if isinstance(hiper, str) else dict(hiper)


class _Muestreo:
    """Toma la memoria del proceso cada pocos milisegundos mientras dura un bloque y se queda con el máximo."""

    def __init__(self, proceso, pausa: float = 0.005) -> None:
        import threading
        self.proceso, self.pausa, self.maximo, self._fin = proceso, pausa, 0, threading.Event()
        self._hilo = threading.Thread(target=self._medir, daemon=True)

    def _medir(self) -> None:
        while not self._fin.is_set():
            self.maximo = max(self.maximo, self.proceso.memory_info().rss)
            self._fin.wait(self.pausa)

    def __enter__(self):
        self.maximo = self.proceso.memory_info().rss
        self._hilo.start()
        return self

    def __exit__(self, *_):
        self._fin.set()
        self._hilo.join()
        self.maximo = max(self.maximo, self.proceso.memory_info().rss)


def medir_uno(tarea: str, modelo: str) -> None:
    """Proceso hijo: ajusta un modelo, predice la prueba e imprime la memoria antes y los máximos observados, en MB."""
    import gc

    import psutil

    p = plan.problema()
    c = p['conjuntos']
    X, y, X_te, _ = plan.objetivo(tarea, p)
    balanceo, hiper = configuracion(plan.registro().cargar(), tarea, modelo)
    proceso = psutil.Process()
    gc.collect()
    antes = proceso.memory_info().rss
    with _Muestreo(proceso) as ajuste:
        tuberia = reajustar(tarea, modelo, balanceo, hiper, X, y, SEED, puntaje=c.y_train_cont)
    with _Muestreo(proceso) as inferencia:
        (tuberia.predict_proba if tarea == 'clasificacion' and hasattr(tuberia, 'predict_proba') else tuberia.predict)(X_te)
    print('MEMORIA ' + json.dumps({'antes_mb': antes / 2 ** 20, 'pico_ajuste_mb': max(ajuste.maximo, antes) / 2 ** 20,
                                   'pico_mb': max(ajuste.maximo, inferencia.maximo, antes) / 2 ** 20}), flush=True)


def main() -> None:
    ruta = CARPETA_RESULTADOS / 'memoria.csv'
    filas = pd.read_csv(ruta).to_dict('records') if ruta.exists() else []
    hechas = {(f['tarea'], f['modelo']) for f in filas}
    for tarea in ('regresion', 'clasificacion'):
        for modelo in plan.modelos(tarea):
            if (tarea, modelo) in hechas:
                continue
            r = subprocess.run([sys.executable, '-B', __file__, '--uno', tarea, modelo], capture_output=True, text=True, cwd=str(plan.RAIZ))
            linea = next((x for x in r.stdout.splitlines() if x.startswith('MEMORIA ')), None)
            if linea is None:
                print(f'{tarea} · {modelo}: sin medida ({r.stderr.strip()[-200:]})', flush=True)
                continue
            m = json.loads(linea[len('MEMORIA '):])
            filas.append({'tarea': tarea, 'modelo': modelo, 'memoria_base_mb': m['antes_mb'],
                          'memoria_ajuste_mb': m['pico_ajuste_mb'] - m['antes_mb'], 'memoria_mb': m['pico_mb'] - m['antes_mb']})
            pd.DataFrame(filas).to_csv(ruta, index=False)
            print(f'{tarea} · {modelo}: {filas[-1]["memoria_mb"]:.0f} MB sobre {m["antes_mb"]:.0f} MB', flush=True)

    # tiempo de inferencia según el número de filas: el original y el mejor benchmark de regresión
    p = plan.problema()
    c = p['conjuntos']
    X, y, X_te, _ = plan.objetivo('regresion', p)
    tabla = plan.registro().cargar()
    veredicto = json.loads((CARPETA_RESULTADOS / 'veredicto.json').read_text(encoding='utf-8'))
    medidas = []
    for modelo in (plan.ORIGINAL, veredicto['regresion']['mejor_benchmark']):
        balanceo, hiper = configuracion(tabla, 'regresion', modelo)
        tuberia = reajustar('regresion', modelo, balanceo, hiper, X, y, SEED, puntaje=c.y_train_cont)
        for n in FILAS_INFERENCIA:
            lote = X_te.iloc[np.arange(n) % len(X_te)]
            tiempos = []
            for _ in range(5):
                t0 = time.perf_counter()
                tuberia.predict(lote)
                tiempos.append(time.perf_counter() - t0)
            medidas.append({'modelo': modelo, 'filas': n, 'inferencia_s': float(np.median(tiempos)),
                            'microsegundos_por_fila': float(1e6 * np.median(tiempos) / n)})
            pd.DataFrame(medidas).to_csv(CARPETA_RESULTADOS / 'inferencia_filas.csv', index=False)
            print(medidas[-1], flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--uno':
        medir_uno(sys.argv[2], sys.argv[3])
    else:
        main()
