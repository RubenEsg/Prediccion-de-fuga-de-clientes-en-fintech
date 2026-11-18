"""Validación anidada del modelo original y de los benchmarks en el problema principal.

Todos los modelos pasan por el mismo procedimiento (``src/optimizacion.validacion_anidada``): 5 pliegues externos y 3
internos, la búsqueda completa de hiperparámetros dentro de cada pliegue externo y la misma semilla global. La semilla
42 es la del Entregable 3: sus corridas se importan y solo se ejecutan los modelos que faltaban (LightGBM y el
original). Las semillas 43 y 44 son las repeticiones 2 y 3: se ejecutan los diez modelos.

Uso (desde la carpeta ``ModeloOriginal``)::

    python experiments/02_validacion_anidada.py --semillas 42
    python experiments/02_validacion_anidada.py --semillas 43 44
    python experiments/02_validacion_anidada.py --semillas 42 --modelos bdl --tareas regresion

Se puede interrumpir y volver a lanzar: las corridas ya registradas se saltan.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import plan                                                       # noqa: E402
from src.config import CARPETA_RESULTADOS, PRESUPUESTO_ORIGINAL, SEED, fijar_semillas  # noqa: E402
from src.experimento import correr_experimento                             # noqa: E402
from src.modelos import TAREAS                                             # noqa: E402


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--semillas', type=int, nargs='+', default=[SEED])
    a.add_argument('--tareas', nargs='+', default=list(TAREAS), choices=TAREAS)
    a.add_argument('--modelos', nargs='+', default=None, help='por defecto, los diez de cada tarea')
    a.add_argument('--n-jobs', type=int, default=-1)
    args = a.parse_args()

    reg = plan.registro()
    print(f'filas importadas del Entregable 3: {plan.importar_entregable3(reg)}', flush=True)
    p = plan.problema()
    etiqueta = {'metodo': p['etiqueta'].metodo, 'umbral': p['etiqueta'].umbral, 'q': p['etiqueta'].q}
    bitacora = CARPETA_RESULTADOS / 'progreso.log'
    for semilla in args.semillas:
        for tarea in args.tareas:
            X, y, _, _ = plan.objetivo(tarea, p)
            for modelo in args.modelos or plan.modelos(tarea):
                if modelo not in plan.modelos(tarea):
                    continue
                balanceo, optimizador = plan.combinacion(tarea, modelo)
                if reg.estado(tarea=tarea, modelo=modelo, balanceo=balanceo, optimizador=optimizador, semilla=semilla) == 'ok':
                    print(f'[{semilla}] {tarea} · {modelo}: ya registrada', flush=True)
                    continue
                fijar_semillas(semilla)
                t0 = time.perf_counter()
                fila = correr_experimento(
                    tarea, modelo, balanceo, optimizador, X, y, seed=semilla, registro=reg, sobrescribir=True,
                    n_jobs=args.n_jobs, etiqueta=etiqueta,
                    presupuesto=PRESUPUESTO_ORIGINAL if modelo == plan.ORIGINAL else None,
                    params_ajuste=plan.params_ajuste(tarea, modelo, p['conjuntos'].y_train_cont),
                    carpeta_modelos=CARPETA_RESULTADOS / 'modelos',
                    carpeta_predicciones=CARPETA_RESULTADOS / 'fuera_de_pliegue')
                linea = (f'{time.strftime("%Y-%m-%d %H:%M")} [{semilla}] {tarea} · {modelo} · {balanceo} · {optimizador}: '
                         f'{fila["metrica_principal"]} = {abs(fila["media_ext"]):.6f} ± {fila["desv_ext"]:.6f} '
                         f'({fila["n_evaluaciones"]} evaluaciones, {time.perf_counter() - t0:.0f} s)')
                print(linea, flush=True)
                with bitacora.open('a', encoding='utf-8') as f:
                    f.write(linea + '\n')


if __name__ == '__main__':
    main()
