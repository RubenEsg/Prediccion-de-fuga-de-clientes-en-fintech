"""Evaluación en los cuatro conjuntos de datos adicionales del protocolo (sección 8).

En cada conjunto se repite el procedimiento del problema principal: 20 % de prueba (semilla 42), validación anidada de
5 × 3 con la semilla 42 para los diez modelos de cada tarea y, para la clasificación, la etiqueta «cuarto superior del
objetivo» con el umbral estimado solo con entrenamiento. Cada benchmark corre la combinación (balanceo, optimizador)
fijada en el protocolo. El modelo final de cada corrida se evalúa una vez en la prueba del conjunto.

Uso (desde ``ModeloOriginal``)::

    python experiments/05_conjuntos_adicionales.py                      # los cuatro conjuntos, las dos tareas
    python experiments/05_conjuntos_adicionales.py --conjuntos seguros  # uno solo

Salidas en ``resultados/adicionales/<conjunto>/``: ``experimentos.parquet`` (tabla maestra del conjunto),
``predicciones_prueba.parquet``, ``metricas_prueba.csv`` y ``ficha.json``. Se puede interrumpir y relanzar.
"""
import argparse
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

from src import conjuntos, datos, evaluacion, plan                         # noqa: E402
from src.config import CARPETA_RESULTADOS, PRESUPUESTO_ORIGINAL, SEED, fijar_semillas  # noqa: E402
from src.experimento import correr_experimento                             # noqa: E402
from src.modelos import TAREAS                                             # noqa: E402
from src.registro import resolver                                          # noqa: E402


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--conjuntos', nargs='+', default=list(conjuntos.CARGADORES), choices=list(conjuntos.CARGADORES))
    a.add_argument('--tareas', nargs='+', default=list(TAREAS), choices=TAREAS)
    a.add_argument('--modelos', nargs='+', default=None)
    a.add_argument('--n-jobs', type=int, default=-1)
    args = a.parse_args()

    for nombre in args.conjuntos:
        X, s, ficha = conjuntos.cargar(nombre)
        particion = datos.partir(X, seed=SEED)
        X_tr, X_te = X.loc[particion.idx_train], X.loc[particion.idx_test]
        s_tr, s_te = s.loc[particion.idx_train], s.loc[particion.idx_test]
        etiqueta = datos.construir_etiqueta(s_tr)
        objetivos = {'regresion': (s_tr, s_te), 'clasificacion': (etiqueta.binarizar(s_tr), etiqueta.binarizar(s_te))}
        carpeta = CARPETA_RESULTADOS / 'adicionales' / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        ficha.update({'n_train': int(len(X_tr)), 'n_test': int(len(X_te)), 'umbral': etiqueta.umbral,
                      'prevalencia_train': float(objetivos['clasificacion'][0].mean()),
                      'prevalencia_test': float(objetivos['clasificacion'][1].mean())})
        (carpeta / 'ficha.json').write_text(json.dumps(ficha, ensure_ascii=False, indent=2), encoding='utf-8')
        reg = plan.registro(nombre)
        ruta_pred = carpeta / 'predicciones_prueba.parquet'
        pred = pd.read_parquet(ruta_pred) if ruta_pred.exists() else pd.DataFrame(index=X_te.index)
        print(f'== {ficha["nombre"]}: {len(X_tr)} filas de entrenamiento, {len(X_te)} de prueba, {X.shape[1]} predictores', flush=True)

        for tarea in args.tareas:
            y_tr, y_te = objetivos[tarea]
            for modelo in args.modelos or plan.modelos(tarea):
                if modelo not in plan.modelos(tarea):
                    continue
                balanceo, optimizador = plan.combinacion(tarea, modelo)
                clave = f'{tarea}__{modelo}'
                if reg.estado(tarea=tarea, modelo=modelo, balanceo=balanceo, optimizador=optimizador, semilla=SEED) == 'ok' and clave in pred:
                    print(f'   {clave}: ya registrada', flush=True)
                    continue
                fijar_semillas(SEED)
                t0 = time.perf_counter()
                try:
                    fila = correr_experimento(
                        tarea, modelo, balanceo, optimizador, X_tr, y_tr, seed=SEED, registro=reg, sobrescribir=True,
                        n_jobs=args.n_jobs, etiqueta={'metodo': etiqueta.metodo, 'umbral': etiqueta.umbral, 'q': etiqueta.q},
                        presupuesto=PRESUPUESTO_ORIGINAL if modelo == plan.ORIGINAL else None,
                        params_ajuste=plan.params_ajuste(tarea, modelo, s_tr),
                        carpeta_modelos=carpeta / 'modelos', carpeta_predicciones=carpeta / 'fuera_de_pliegue')
                except Exception as e:                                   # noqa: BLE001
                    print(f'   {clave}: ERROR {type(e).__name__}: {e}'[:300], flush=True)
                    continue
                if fila['estado'] != 'ok':
                    print(f'   {clave}: no aplica ({fila.get("motivo")})', flush=True)
                    continue
                final = joblib.load(resolver(fila['ruta_modelo']))
                if tarea == 'clasificacion':
                    clase, puntaje, _ = evaluacion.puntajes(final, X_te)
                    pred[clave], pred[clave + '__clase'] = puntaje, clase
                else:
                    pred[clave] = final.predict(X_te)
                pred.to_parquet(ruta_pred)
                m = evaluacion.evaluar(final, X_te, y_te, tarea)
                print(f'   {clave} · {balanceo} · {optimizador}: externa {abs(fila["media_ext"]):.5f} ± {fila["desv_ext"]:.5f} | '
                      f'prueba {m[evaluacion.NOMBRE_METRICA[tarea]]:.5f} ({time.perf_counter() - t0:.0f} s)', flush=True)

        # métricas de prueba de todo lo que haya
        filas = []
        for tarea in TAREAS:
            y_te = np.asarray(objetivos[tarea][1])
            for modelo in plan.modelos(tarea):
                clave = f'{tarea}__{modelo}'
                if clave not in pred:
                    continue
                if tarea == 'clasificacion':
                    m = evaluacion.metricas_clasificacion(y_te, pred[clave + '__clase'].to_numpy(), pred[clave].to_numpy(), modelo != 'svm')
                else:
                    m = evaluacion.metricas_regresion(y_te, pred[clave].to_numpy())
                filas.append({'conjunto': nombre, 'tarea': tarea, 'modelo': modelo, **m})
        pd.DataFrame(filas).to_csv(carpeta / 'metricas_prueba.csv', index=False)


if __name__ == '__main__':
    main()
