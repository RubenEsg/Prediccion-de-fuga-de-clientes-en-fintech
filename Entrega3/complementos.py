"""Complementos de la evaluación que requieren reajustar modelos.

- ``fuera_de_pliegue`` (§5.1 y §5.2): predicciones de validación cruzada de 5 pliegues sobre el
  entrenamiento y predicciones dentro de la muestra, para la mejor combinación de cada modelo.
  Alimentan la gráfica de entrenamiento, validación y prueba y la recalibración.
- ``calibracion`` (§5.1): recalibración de Platt e isotónica ajustada **solo con las predicciones
  fuera de pliegue del entrenamiento** y aplicada a la prueba; Brier, ECE y log-loss antes y
  después.
- ``semillas`` (§5.4): reentrena los modelos finales de Random Forest, XGBoost, SVM y la EBM con
  varias semillas (la del modelo y la del muestreador; la partición no cambia) y mide la
  dispersión de la métrica de prueba.
- ``semillas_optimizadores`` (§5.4): repite la búsqueda de Random, Bayesiana y Genética sobre el
  árbol de decisión con 5 semillas del optimizador y los mismos pliegues internos.

Uso::

    .venv\\Scripts\\python.exe -u complementos.py                 (todo lo pendiente)
    .venv\\Scripts\\python.exe -u complementos.py semillas        (solo eso)
    .venv\\Scripts\\python.exe -u complementos.py --esperar       (espera a que el PC quede libre)

Salidas en ``resultados/complementos/`` y ``progreso.log``.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / 'resultados' / 'complementos'
PRUEBA = False
EXPERIMENTOS = ['fuera_de_pliegue', 'calibracion', 'semillas', 'semillas_optimizadores']
SEMILLAS = list(range(10))
MODELOS_SEMILLAS = {'clasificacion': ['random_forest', 'xgboost', 'svm', 'ebm'],
                    'regresion': ['random_forest', 'xgboost', 'ebm']}

log_archivo = None


def log(msg: str) -> None:
    linea = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}'
    print(linea, flush=True)
    if log_archivo is not None:
        log_archivo.write(linea + '\n')
        log_archivo.flush()


def preparar() -> dict:
    """Datos con la etiqueta del percentil 75 y la mejor combinación de cada modelo."""
    from src import analisis, datos
    p = datos.preparar_todo()
    c = p['conjuntos']
    X, y, y_c = c.X_train, p['y_train'], c.y_train_cont
    X_te = c.X_test.sort_index()
    y_te, y_te_c = p['y_test'].loc[X_te.index], c.y_test_cont.loc[X_te.index]
    if PRUEBA:
        X, y, y_c = X.iloc[:3000], y.iloc[:3000], y_c.iloc[:3000]
        X_te, y_te, y_te_c = X_te.iloc[:1500], y_te.iloc[:1500], y_te_c.iloc[:1500]
    tabla = analisis.cargar_tabla(RAIZ / 'resultados' / 'experimentos.parquet')
    return {'X': X, 'y': y, 'y_c': y_c, 'X_te': X_te, 'y_te': y_te, 'y_te_c': y_te_c, 'tabla': tabla,
            'mejores': {t: analisis.mejores_por_modelo(tabla, t) for t in ('clasificacion', 'regresion')}}


def _salida(modelo, X, tarea: str) -> tuple[np.ndarray, str]:
    """Puntaje (clasificación) o pronóstico (regresión) y el método usado."""
    if tarea == 'regresion':
        return modelo.predict(X), 'predict'
    if hasattr(modelo, 'predict_proba'):
        return modelo.predict_proba(X)[:, 1], 'predict_proba'
    return modelo.decision_function(X), 'decision_function'


# ---------------------------------------------------------------------- experimentos
def exp_fuera_de_pliegue(ctx) -> pd.DataFrame:
    """Predicciones fuera de pliegue (5 pliegues) y dentro de la muestra sobre el entrenamiento."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict
    from src import evaluacion
    from src.optimizacion import construir_cv
    pred = pd.DataFrame({'y_bin': ctx['y'], 'y_cont': ctx['y_c']}, index=ctx['X'].index)
    filas = []
    for tarea in ('clasificacion', 'regresion'):
        y = ctx['y'] if tarea == 'clasificacion' else ctx['y_c']
        for _, f in ctx['mejores'][tarea].iterrows():
            t0 = time.perf_counter()
            final = joblib.load(f['ruta_modelo'])
            dentro, metodo = _salida(final, ctx['X'], tarea)
            oof = cross_val_predict(clone(final), ctx['X'], y, cv=construir_cv(tarea, 5), method=metodo, n_jobs=-1)
            oof = oof[:, 1] if oof.ndim == 2 else oof
            pred[f'{f["clave"]}__entrenamiento'] = dentro
            pred[f'{f["clave"]}__fuera_de_pliegue'] = oof
            fila = {'tarea': tarea, 'modelo': f['modelo'], 'clave': f['clave']}
            if tarea == 'clasificacion':
                from sklearn.metrics import roc_auc_score
                fila.update({'auc_entrenamiento': roc_auc_score(y, dentro), 'auc_fuera_de_pliegue': roc_auc_score(y, oof)})
            else:
                me, mo = evaluacion.metricas_regresion(y, dentro), evaluacion.metricas_regresion(y, oof)
                fila.update({'rmse_entrenamiento': me['rmse'], 'rmse_fuera_de_pliegue': mo['rmse'],
                             'r2_entrenamiento': me['r2'], 'r2_fuera_de_pliegue': mo['r2']})
            filas.append(fila)
            pred.to_parquet(SALIDA / 'fuera_de_pliegue.parquet')
            log(f'  fuera de pliegue {f["clave"]}: {time.perf_counter() - t0:.0f} s')
    return pd.DataFrame(filas)


def exp_calibracion(ctx) -> pd.DataFrame:
    """Platt e isotónica ajustadas con las predicciones fuera de pliegue, evaluadas en prueba."""
    from scipy.special import logit
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from src.evaluacion import ece
    oof = pd.read_parquet(SALIDA / 'fuera_de_pliegue.parquet')
    y, y_te = ctx['y'].to_numpy(), ctx['y_te'].to_numpy()
    probas = pd.DataFrame({'y_bin': y_te}, index=ctx['X_te'].index)
    filas = []
    for _, f in ctx['mejores']['clasificacion'].iterrows():
        final = joblib.load(f['ruta_modelo'])
        s_te, metodo = _salida(final, ctx['X_te'], 'clasificacion')
        s_oof = oof[f'{f["clave"]}__fuera_de_pliegue'].to_numpy()
        es_prob = metodo == 'predict_proba'

        def transformar(s):
            return logit(np.clip(s, 1e-6, 1 - 1e-6)) if es_prob else s

        platt = LogisticRegression(C=1e6, max_iter=1000).fit(transformar(s_oof).reshape(-1, 1), y)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(s_oof, y)
        versiones = {'Platt': platt.predict_proba(transformar(s_te).reshape(-1, 1))[:, 1],
                     'isotónica': iso.predict(s_te)}
        if es_prob:
            versiones = {'sin recalibrar': s_te, **versiones}
        for nombre, p in versiones.items():
            probas[f'{f["clave"]}__{nombre}'] = p
            filas.append({'modelo': f['modelo'], 'balanceo': f['balanceo'], 'optimizador': f['optimizador'],
                          'version': nombre, 'brier': brier_score_loss(y_te, p), 'ece': ece(y_te, p),
                          'ece_cuantil': ece(y_te, p, estrategia='cuantil'),
                          'log_loss': log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6)), 'auc': roc_auc_score(y_te, p),
                          'prob_media': float(np.mean(p)), 'prevalencia': float(np.mean(y_te))})
        log(f'  calibración {f["clave"]}')
    probas.to_parquet(SALIDA / 'calibracion_prueba.parquet')
    return pd.DataFrame(filas)


def exp_semillas(ctx) -> pd.DataFrame:
    """Métrica de prueba del modelo final reentrenado con distintas semillas."""
    from sklearn.base import clone
    from src import evaluacion
    filas = []
    for tarea, nombres in MODELOS_SEMILLAS.items():
        y, y_te = (ctx['y'], ctx['y_te']) if tarea == 'clasificacion' else (ctx['y_c'], ctx['y_te_c'])
        mej = ctx['mejores'][tarea].set_index('modelo')
        for nombre in nombres:
            if nombre not in mej.index:
                continue
            f = mej.loc[nombre]
            final = joblib.load(f['ruta_modelo'])
            semillas = SEMILLAS[:3] if PRUEBA else (SEMILLAS[:5] if nombre in ('random_forest', 'ebm') else SEMILLAS)
            for s in semillas:
                t0 = time.perf_counter()
                m = clone(final).set_params(modelo__random_state=s)
                if 'muestreo' in m.named_steps:
                    m.set_params(muestreo__random_state=s)
                m.fit(ctx['X'], y)
                met = evaluacion.evaluar(m, ctx['X_te'], y_te, tarea)
                filas.append({'tarea': tarea, 'modelo': nombre, 'balanceo': f['balanceo'],
                              'optimizador': f['optimizador'], 'semilla': s, **met,
                              'tiempo_s': time.perf_counter() - t0})
            log(f'  semillas {tarea} {nombre}: {len(semillas)} ajustes')
    return pd.DataFrame(filas)


def exp_semillas_optimizadores(ctx) -> pd.DataFrame:
    """Búsqueda del árbol de decisión con 5 semillas de cada optimizador estocástico."""
    from src import balanceo as bal
    from src import evaluacion, modelos
    from src.optimizacion import buscar, construir_cv
    spec = modelos.catalogo('clasificacion')['arbol']
    cv = construir_cv('clasificacion', 3)                  # mismos pliegues internos en todas las semillas
    filas = []
    for opt in ('random', 'bayesiana', 'genetica'):
        for s in (range(2) if PRUEBA else range(5)):
            pipe = bal.construir_pipeline(spec, 'ninguno', ctx['X'], 42, ctx['y'])
            res = buscar(opt, pipe, spec.espacio_grid, spec.espacio, ctx['X'], ctx['y'], cv, 'roc_auc',
                         presupuesto=3 if PRUEBA else None, seed=s)
            met = evaluacion.evaluar(res.estimador, ctx['X_te'], ctx['y_te'], 'clasificacion')
            filas.append({'optimizador': opt, 'semilla': s, 'mejor_interno': res.mejor_interno,
                          'auc_prueba': met['auc'], 'n_evaluaciones': res.n_evaluaciones,
                          'tiempo_s': res.tiempo_s, **{k.split('__', 1)[1]: v for k, v in res.mejores_params.items()}})
            log(f'  semillas del optimizador {opt} {s}: interno {res.mejor_interno:.4f} · prueba {met["auc"]:.4f}')
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- espera
def esperar_pc_libre() -> None:
    """Espera a que no haya corridas, comparación ni experimentos de cómputo en marcha."""
    import psutil
    cerrojo = RAIZ / 'resultados' / 'experimentos.parquet.en_uso'
    libre_desde, avisado = None, False
    while True:
        ocupado = cerrojo.exists()
        for proc in psutil.process_iter(['name', 'cmdline', 'pid']):
            try:
                cmd = ' '.join(proc.info['cmdline'] or [])
            except psutil.Error:
                continue
            if proc.info['pid'] != psutil.Process().pid and 'python' in (proc.info['name'] or '').lower() and \
                    any(x in cmd for x in ('correr_140', 'comparar_entrega2', 'computo.py')):
                ocupado = True
                break
        if ocupado:
            libre_desde = None
            if not avisado:
                log('en espera: hay corridas o experimentos de cómputo en marcha')
                avisado = True
        else:
            libre_desde = libre_desde or time.time()
            if time.time() - libre_desde >= 120:
                log('el PC quedó libre; empiezan los complementos')
                return
        time.sleep(60)


def main() -> None:
    global log_archivo, SALIDA, PRUEBA
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('experimentos', nargs='*', help=f'subconjunto de {EXPERIMENTOS}')
    ap.add_argument('--esperar', action='store_true')
    ap.add_argument('--rehacer', action='store_true')
    ap.add_argument('--prueba', action='store_true')
    args = ap.parse_args()
    desconocidos = set(args.experimentos) - set(EXPERIMENTOS)
    if desconocidos:
        ap.error(f'experimentos desconocidos: {sorted(desconocidos)}')
    PRUEBA = args.prueba
    if PRUEBA:
        SALIDA = RAIZ / 'resultados_prueba' / 'complementos'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    warnings.filterwarnings('ignore')
    SALIDA.mkdir(parents=True, exist_ok=True)
    log_archivo = open(SALIDA / 'progreso.log', 'a', encoding='utf-8')
    sys.path.insert(0, str(RAIZ))
    if sys.platform == 'win32':           # sin suspensión también mientras espera
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if args.esperar:
        esperar_pc_libre()
    from correr_140 import congelar_codigo
    destino = RAIZ / '_codigo_complementos'
    huella = congelar_codigo(destino, refrescar=True)
    sys.path.insert(0, str(destino))
    if sys.platform == 'win32':
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
        k.SetThreadExecutionState(0x80000000 | 0x00000001)
    pedidos = args.experimentos or EXPERIMENTOS
    log('=' * 78)
    log(f'inicio · código congelado en {destino.name} (huella {huella}) · {pedidos}')
    t0 = time.perf_counter()
    ctx = preparar()
    for nombre in pedidos:
        ruta = SALIDA / f'{nombre}.csv'
        if ruta.exists() and not args.rehacer:
            log(f'{nombre}: ya hecho, se salta')
            continue
        t1 = time.perf_counter()
        try:
            globals()[f'exp_{nombre}'](ctx).to_csv(ruta, index=False)
        except Exception as e:                                   # noqa: BLE001
            import traceback
            log(f'{nombre}: ERROR {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}')
            continue
        log(f'{nombre}: hecho en {(time.perf_counter() - t1) / 60:.1f} min')
    log(f'fin en {(time.perf_counter() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
