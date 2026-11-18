"""Estudio de ablación del modelo original (protocolo, sección 9).

Cada variante retira un componente (``src/variantes.py``) y pasa por el mismo procedimiento que el modelo completo:
validación anidada de 5 × 3 con la semilla 42, optimización bayesiana con 30 evaluaciones y evaluación del modelo
final en prueba. Después se compara cada variante con el modelo completo con la prueba *t* corregida de Nadeau-Bengio
sobre los cinco pliegues (P2) y, en prueba, con DeLong o con la permutación pareada (P3).

Salidas en ``resultados/``: las corridas quedan en la tabla maestra (modelo = nombre de la variante),
``ablacion_prueba.parquet`` (predicciones) y ``ablacion.csv`` (la tabla del estudio).
Uso: ``python experiments/07_ablacion.py`` (después de 02 y 03).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import balanceo as bal                                            # noqa: E402
from src import estadistica, evaluacion, plan, pruebas                     # noqa: E402
from src.config import CARPETA_RESULTADOS, PRESUPUESTO_ORIGINAL, SEED, fijar_semillas  # noqa: E402
from src.optimizacion import validacion_anidada                            # noqa: E402
from src.variantes import NOMBRES, VARIANTES, especificacion_variante      # noqa: E402


def main() -> None:
    p = plan.problema()
    c = p['conjuntos']
    reg = plan.registro()
    ruta_pred = CARPETA_RESULTADOS / 'ablacion_prueba.parquet'
    pred = pd.read_parquet(ruta_pred) if ruta_pred.exists() else pd.DataFrame(index=c.X_test.index)
    for variante, tareas in VARIANTES.items():
        for tarea in tareas:
            clave = f'{tarea}__{variante}'
            if reg.estado(tarea=tarea, modelo=variante, balanceo='ninguno', optimizador='bayesiana', semilla=SEED) == 'ok' and clave in pred:
                print(f'{clave}: ya registrada', flush=True)
                continue
            X, y, X_te, y_te = plan.objetivo(tarea, p)
            spec = especificacion_variante(variante, tarea)
            fijar_semillas(SEED)
            t0 = time.perf_counter()
            res = validacion_anidada(
                lambda y_pliegue, spec=spec, X=X: bal.construir_pipeline(spec, 'ninguno', X, SEED, y_pliegue),
                spec.espacio_grid, spec.espacio, 'bayesiana', X, y, tarea, presupuesto=PRESUPUESTO_ORIGINAL, seed=SEED,
                params_ajuste={'modelo__puntaje': c.y_train_cont.to_numpy()} if tarea == 'clasificacion' else None)
            reg.anadir({'tarea': tarea, 'modelo': variante, 'balanceo': 'ninguno', 'optimizador': 'bayesiana', 'semilla': SEED,
                        'k_ext': 5, 'k_int': 3, 'metrica_principal': evaluacion.NOMBRE_METRICA[tarea], 'aplica': True, 'estado': 'ok',
                        'motivo': '', 'presupuesto': PRESUPUESTO_ORIGINAL, 'n_train': int(len(X)), 'media_ext': res.media_ext,
                        'desv_ext': res.desv_ext, 'puntajes_ext': res.puntajes_ext, 'metricas_ext': res.metricas_ext,
                        'hiperparametros': res.params_finales, 'params_por_pliegue': res.params_por_pliegue,
                        'n_evaluaciones': res.n_evaluaciones, 'n_evaluaciones_final': res.n_evaluaciones_final,
                        'tiempo_total_s': time.perf_counter() - t0, 'origen': 'ablación'}, sobrescribir=True)
            pred[clave] = (evaluacion.puntajes(res.estimador_final, X_te)[1] if tarea == 'clasificacion'
                           else res.estimador_final.predict(X_te))
            pred.to_parquet(ruta_pred)
            print(f'{clave}: externa {abs(res.media_ext):.6f} ± {res.desv_ext:.6f} ({time.perf_counter() - t0:.0f} s)', flush=True)

    # comparación de cada variante con el modelo completo
    tabla = reg.cargar()
    completo = pd.read_parquet(CARPETA_RESULTADOS / 'predicciones_prueba.parquet')
    y_bin, y_cont = p['y_test'].to_numpy(), c.y_test_cont.to_numpy()
    filas = []
    for tarea in ('regresion', 'clasificacion'):
        signo = 1.0 if tarea == 'clasificacion' else -1.0
        f0 = tabla[(tabla.tarea == tarea) & (tabla.modelo == plan.ORIGINAL) & (tabla.semilla == SEED)].iloc[0]
        base = plan.puntajes_por_pliegue(f0)
        a = completo[f'{tarea}__{plan.ORIGINAL}'].to_numpy()
        prueba_completo = (estadistica.delong(y_bin, a)['auc'] if tarea == 'clasificacion' else float(np.sqrt(np.mean((y_cont - a) ** 2))))
        filas.append({'tarea': tarea, 'variante': 'bdl', 'nombre': 'BDL completo', 'cv_media': base.mean(), 'cv_desv': base.std(ddof=1),
                      'prueba': prueba_completo})
        for variante, tareas in VARIANTES.items():
            if tarea not in tareas:
                continue
            f = tabla[(tabla.tarea == tarea) & (tabla.modelo == variante) & (tabla.semilla == SEED)]
            if f.empty or f'{tarea}__{variante}' not in pred:
                continue
            v = plan.puntajes_por_pliegue(f.iloc[0])
            b = pred[f'{tarea}__{variante}'].to_numpy()
            t = pruebas.t_corregida(base, v)
            if tarea == 'clasificacion':
                d = estadistica.delong(y_bin, b, a)
                p3, valor = d['p_bilateral'], d['auc_a']
            else:
                p3 = pruebas.permutacion_pareada((y_cont - a) ** 2, (y_cont - b) ** 2)['p']
                valor = float(np.sqrt(np.mean((y_cont - b) ** 2)))
            filas.append({'tarea': tarea, 'variante': variante, 'nombre': NOMBRES[variante], 'cv_media': v.mean(), 'cv_desv': v.std(ddof=1),
                          'prueba': valor, 'aporte_cv': float(signo * (base - v).mean()), 'pliegues_a_favor_del_completo': int((signo * (base - v) > 0).sum()),
                          'p_t_corregida': t['p'], 'aporte_prueba': float(signo * (prueba_completo - valor)), 'p_prueba': p3,
                          'cohen_pareada': float(signo * estadistica.d_cohen(base - v)['d'])})
    for tarea in ('regresion', 'clasificacion'):                    # Holm dentro de cada tarea y prueba
        idx = [i for i, f in enumerate(filas) if f['tarea'] == tarea and 'p_t_corregida' in f]
        for col in ('p_t_corregida', 'p_prueba'):
            if idx:
                for i, pa in zip(idx, estadistica.ajustar_pvalores([filas[i][col] for i in idx], 'holm')):
                    filas[i][col + '_holm'] = float(pa)
    t = pd.DataFrame(filas)
    t.to_csv(CARPETA_RESULTADOS / 'ablacion.csv', index=False)
    pd.set_option('display.width', 250, 'display.float_format', lambda v: f'{v:.6f}')
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
