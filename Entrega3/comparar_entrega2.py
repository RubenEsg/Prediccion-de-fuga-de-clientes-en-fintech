"""Comparación con la Entrega 2 sobre la misma prueba (exigencia del profesor del 29-sep-2026).

Reentrena los dos modelos de la Entrega 2 —logística L1 con C = 0,1 y Lasso con alpha = 1e-4,
los hiperparámetros que eligió su ``GridSearchCV``— sobre el mismo entrenamiento (partición
80/20 con semilla 42; etiqueta = mediana de ``churn_probability`` estimada solo con
entrenamiento), comprueba que reproducen las cifras publicadas y los compara en la misma prueba
con:

- la **EBM** (GA²M), el modelo nuevo, y
- **XGBoost**, el mejor modelo visto en clase en las 140 corridas,

afinados ambos con el mismo protocolo: Optuna (TPE) con el presupuesto de las 140 corridas,
validación cruzada de 5 pliegues sobre el entrenamiento (los mismos 5 pliegues de la Entrega 2)
y reajuste final sobre todo el entrenamiento. La prueba se usa una sola vez por modelo, al final.

Uso, desde la carpeta del proyecto::

    .venv\\Scripts\\python.exe -u comparar_entrega2.py            (unos 40 min)
    .venv\\Scripts\\python.exe -u comparar_entrega2.py --prueba   (ensayo corto sobre 3.000 filas)

Salidas en ``resultados/comparacion_entrega2/`` (``resultados_prueba/...`` con ``--prueba``):

- ``predicciones.parquet``: etiqueta, objetivo y puntaje, clase o pronóstico de prueba de cada
  modelo, en el orden original de las filas del archivo de clientes;
- ``modelos/``: los seis pipelines ajustados;
- ``busquedas.json``: hiperparámetros, puntaje interno, métrica por pliegue, traza *anytime*,
  tiempos y evaluaciones de cada modelo, métricas de prueba y verificación de la Entrega 2;
- ``intervalos_*.csv`` y ``contrastes_*.csv``: las tablas de ``src.comparacion``;
- ``progreso.log``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

RAIZ = Path(__file__).resolve().parent

CANDIDATOS: dict[str, str] = {'xgboost': 'mejor modelo de clase en las 140 corridas',
                              'ebm': 'modelo nuevo, no visto en clase'}
REFERENCIA: dict[str, str] = {'clasificacion': 'logistica_e2', 'regresion': 'lasso_e2'}


def _json(o):
    """Serializa tipos de numpy y tuplas para ``json.dump``."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _bajar_prioridad_y_evitar_suspension() -> None:
    """Prioridad por debajo de lo normal (la heredan los procesos hijos) y sin suspensión."""
    if sys.platform == 'win32':
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
        k.SetThreadExecutionState(0x80000000 | 0x00000001)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--prueba', action='store_true', help='ensayo corto sobre 3.000 filas de entrenamiento')
    ap.add_argument('--rehacer', action='store_true', help='repetir aunque ya existan los resultados')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    warnings.filterwarnings('ignore')

    salida = RAIZ / ('resultados_prueba' if args.prueba else 'resultados') / 'comparacion_entrega2'
    if (salida / 'contrastes_regresion.csv').exists() and not (args.rehacer or args.prueba):
        print(f'la comparación ya está hecha en {salida}; use --rehacer para repetirla', flush=True)
        return
    _bajar_prioridad_y_evitar_suspension()

    # Código congelado, como en las corridas: los procesos hijos importan src/ desde la copia, así
    # que editar src/ mientras esto corre no mezcla versiones.
    sys.path.insert(0, str(RAIZ))
    from correr_140 import congelar_codigo
    destino = RAIZ / '_codigo_comparacion'
    huella = congelar_codigo(destino, refrescar=True)
    sys.path.insert(0, str(destino))
    global bal, cmp, datos, estadistica, evaluacion, modelos, optimizacion, SEED, fijar_semillas
    from src import balanceo as bal
    from src import comparacion as cmp
    from src import datos, estadistica, evaluacion, modelos, optimizacion
    from src.config import SEED, fijar_semillas
    if Path(modelos.__file__).resolve().parent.parent != destino.resolve():
        raise RuntimeError(f'src se importó desde {modelos.__file__}, no desde la copia {destino}')
    (salida / 'modelos').mkdir(parents=True, exist_ok=True)
    archivo_log = open(salida / 'progreso.log', 'a', encoding='utf-8')

    def log(msg: str) -> None:
        linea = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}'
        print(linea, flush=True)
        archivo_log.write(linea + '\n')
        archivo_log.flush()

    k_cv, presupuesto, n_boot = (2, 3, 200) if args.prueba else (5, None, estadistica.N_BOOT)
    t_inicio = time.perf_counter()
    log('=' * 78)
    log(f'inicio {"(PRUEBA) " if args.prueba else ""}· código congelado en {destino.name} (huella {huella}) '
        f'· {k_cv} pliegues · presupuesto {presupuesto or "el de las 140 corridas"} · {n_boot} réplicas bootstrap')

    fijar_semillas(SEED)
    p = datos.preparar_todo(metodo_etiqueta='mediana')
    c, e = p['conjuntos'], p['etiqueta']
    X_tr = c.X_train
    X_te = c.X_test.sort_index()                 # orden original del archivo de clientes
    objetivos = {'clasificacion': (p['y_train'], p['y_test'].loc[X_te.index]),
                 'regresion': (c.y_train_cont, c.y_test_cont.loc[X_te.index])}
    if args.prueba:
        X_tr = X_tr.iloc[:3000]
        objetivos = {t: (ytr.loc[X_tr.index], yte) for t, (ytr, yte) in objetivos.items()}
    log(f'datos: entrenamiento {len(X_tr):,} × {X_tr.shape[1]} · prueba {len(X_te):,} · etiqueta '
        f'{e.metodo} con umbral {e.umbral:.6f} estimado en entrenamiento · riesgo alto en prueba '
        f'{objetivos["clasificacion"][1].mean() * 100:.2f} %')

    pred = pd.DataFrame({'y_bin': objetivos['clasificacion'][1], 'y_cont': objetivos['regresion'][1]},
                        index=X_te.index)
    resumen = {'fecha': f'{datetime.now():%Y-%m-%d %H:%M}', 'prueba': args.prueba, 'semilla': SEED, 'huella_codigo': huella,
               'k_cv': k_cv, 'optimizador': 'bayesiana (Optuna TPE)', 'n_train': int(len(X_tr)),
               'n_test': int(len(X_te)), 'etiqueta': {'metodo': e.metodo, 'umbral': e.umbral},
               'modelos': {}}
    pliegues_metrica: dict[str, dict[str, list[float]]] = {'clasificacion': {}, 'regresion': {}}

    for tarea in ('clasificacion', 'regresion'):
        y_tr, y_te = objetivos[tarea]
        cv = optimizacion.construir_cv(tarea, k_cv, SEED)
        scoring = evaluacion.METRICA_PRINCIPAL[tarea]
        signo = 1.0 if tarea == 'clasificacion' else -1.0      # RMSE positivo en los pliegues
        pref = 'clf' if tarea == 'clasificacion' else 'reg'
        for nombre in [REFERENCIA[tarea], *CANDIDATOS]:
            t0 = time.perf_counter()
            if nombre == REFERENCIA[tarea]:
                final = cmp.pipeline_entrega2(tarea, X_tr).fit(X_tr, y_tr)
                info = {'papel': 'Entrega 2', 'hiperparametros':
                        {k: v for k, v in final.named_steps['modelo'].get_params().items()
                         if k in ('C', 'l1_ratio', 'alpha', 'solver')}}
            else:
                spec = modelos.catalogo(tarea, incluir_nuevo=True)[nombre]
                fid = spec.fidelidad or {}
                res = optimizacion.buscar('bayesiana', bal.construir_pipeline(spec, 'ninguno', X_tr, SEED, y_tr),
                                          spec.espacio_grid, spec.espacio, X_tr, y_tr, cv, scoring,
                                          presupuesto=presupuesto, seed=SEED, n_jobs=-1,
                                          fijos_busqueda=fid.get('busqueda'), fijos_final=fid.get('final'))
                final = res.estimador
                info = {'papel': CANDIDATOS[nombre], 'hiperparametros': res.mejores_params,
                        'mejor_interno': res.mejor_interno, 'n_evaluaciones': res.n_evaluaciones,
                        'tiempo_busqueda_s': res.tiempo_s, 'traza': res.traza,
                        'fidelidad': spec.fidelidad}
            t_ajuste = time.perf_counter() - t0
            puntos = signo * cross_val_score(clone(final), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
            pliegues_metrica[tarea][nombre] = [float(v) for v in puntos]
            metricas = evaluacion.evaluar(final, X_te, y_te, tarea)
            if tarea == 'clasificacion':
                pred[f'{pref}_{nombre}'] = final.predict_proba(X_te)[:, 1]
                pred[f'{pref}_{nombre}_clase'] = final.predict(X_te)
            else:
                pred[f'{pref}_{nombre}'] = final.predict(X_te)
            info.update({'tiempo_total_s': t_ajuste, 'pliegues': pliegues_metrica[tarea][nombre],
                         'cv_media': float(np.mean(puntos)), 'cv_desv': float(np.std(puntos, ddof=1)),
                         'prueba': metricas})
            if nombre == REFERENCIA[tarea] and not args.prueba:
                info['reproduccion'] = cmp.reproduce_entrega2(tarea, metricas)
                ok = all(v['coincide'] for v in info['reproduccion'].values())
                log(f'{tarea}: la Entrega 2 {"SE REPRODUCE" if ok else "NO SE REPRODUCE"}: {info["reproduccion"]}')
            joblib.dump(final, salida / 'modelos' / f'{tarea}__{nombre}.joblib', compress=3)
            resumen['modelos'][f'{tarea}/{nombre}'] = info
            principal = metricas[evaluacion.NOMBRE_METRICA[tarea]]
            extra = f' · R² {metricas["r2"]:.4f}' if tarea == 'regresion' else f' · recall {metricas["recall"]:.4f}'
            log(f'{tarea} · {nombre}: CV {np.mean(puntos):.4f} ± {np.std(puntos, ddof=1):.4f} · prueba '
                f'{evaluacion.NOMBRE_METRICA[tarea]} {principal:.4f}{extra} · {t_ajuste / 60:.1f} min')
            pred.to_parquet(salida / 'predicciones.parquet')
            (salida / 'busquedas.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=_json),
                                                   encoding='utf-8')

    log('contrastes sobre la prueba…')
    for tarea, pref in (('clasificacion', 'clf'), ('regresion', 'reg')):
        ref = REFERENCIA[tarea]
        salidas = {n: pred[f'{pref}_{n}'].to_numpy() for n in [ref, 'xgboost', 'ebm']}
        pares = [(ref, 'ebm'), (ref, 'xgboost'), ('xgboost', 'ebm')]
        y_te = pred['y_bin'] if tarea == 'clasificacion' else pred['y_cont']
        if tarea == 'clasificacion':
            intervalos = cmp.intervalos_clasificacion(y_te, salidas, n_boot=n_boot)
            contrastes = cmp.contrastar_clasificacion(y_te, salidas, pares, pliegues_metrica[tarea], n_boot=n_boot)
        else:
            intervalos = cmp.intervalos_regresion(y_te, salidas, n_boot=n_boot)
            contrastes = cmp.contrastar_regresion(y_te, salidas, pares, pliegues_metrica[tarea], n_boot=n_boot)
        intervalos.to_csv(salida / f'intervalos_{tarea}.csv')
        contrastes.to_csv(salida / f'contrastes_{tarea}.csv', index=False)
        for _, f in contrastes.iterrows():
            if tarea == 'clasificacion':
                log(f'  {f["candidato"]} frente a {f["referencia"]}: ΔAUC {f["diferencia"]:+.4f} '
                    f'[BCa {f["ic_bca_inf"]:+.4f}, {f["ic_bca_sup"]:+.4f}] · DeLong p = {f["p_delong"]:.2e} '
                    f'(Holm {f["p_delong_ajustado"]:.2e}) · Cliff {f["delta_cliff"]:+.2f}')
            else:
                log(f'  {f["candidato"]} frente a {f["referencia"]}: ΔR² {f["mejora_r2"]:+.4f} · ΔRMSE '
                    f'{f["mejora_rmse"]:+.5f} [BCa {f["mejora_rmse_ic_inf"]:+.5f}, {f["mejora_rmse_ic_sup"]:+.5f}] '
                    f'· DM-HLN {f["dm_hln"]:.2f}, p = {f["p_dm"]:.2e} (Holm {f["p_dm_ajustado"]:.2e}) · '
                    f'bootstrap estacionario p = {f["p_bootstrap_estacionario"]:.3f}')
    log(f'fin en {(time.perf_counter() - t_inicio) / 60:.1f} min · resultados en {salida}')
    archivo_log.close()


if __name__ == '__main__':
    main()
