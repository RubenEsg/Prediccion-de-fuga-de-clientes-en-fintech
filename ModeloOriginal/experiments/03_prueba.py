"""Evaluación en el conjunto de prueba de los diez modelos finales de cada tarea (semilla 42).

Los modelos del Entregable 3 ya tienen sus predicciones de prueba publicadas (``Entrega3/resultados``): se leen de ahí.
LightGBM y el modelo original se cargan de ``resultados/modelos`` y predicen aquí. Además de las métricas del protocolo
(sección 3) se calculan:

- las probabilidades calibradas de cada clasificador (isotónica ajustada con sus predicciones fuera de pliegue de
  entrenamiento, como en el Entregable 3), para la hipótesis H4;
- los intervalos conformes de cada regresor (``pronóstico ± q``, con ``q`` el cuantil de sus errores absolutos fuera de
  pliegue) y los intervalos y conjuntos del modelo original, con su cobertura y su largo en prueba, para la hipótesis H3.

Salidas en ``resultados/``: ``predicciones_prueba.parquet``, ``metricas_prueba.csv`` e ``incertidumbre_prueba.csv``.
Uso: ``python experiments/03_prueba.py``.
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import evaluacion, plan, pruebas                                  # noqa: E402
from src.config import CARPETA_RESULTADOS, SEED                            # noqa: E402
from src.registro import resolver                                          # noqa: E402

COBERTURAS = (0.80, 0.90, 0.95)
E3 = plan.RAIZ.parent / 'Entrega3' / 'resultados'


def fila_de(tabla: pd.DataFrame, tarea: str, modelo: str) -> pd.Series:
    balanceo, optimizador = plan.combinacion(tarea, modelo)
    f = tabla[(tabla.tarea == tarea) & (tabla.modelo == modelo) & (tabla.balanceo == balanceo)
              & (tabla.optimizador == optimizador) & (tabla.semilla == SEED)]
    if f.empty:
        raise SystemExit(f'falta la corrida {tarea} · {modelo} (semilla {SEED}): ejecute 02_validacion_anidada.py')
    return f.iloc[0]


def fuera_de_pliegue(tabla: pd.DataFrame, tarea: str, modelo: str, oof_e3: pd.DataFrame, indice_train) -> np.ndarray:
    """Predicción fuera de pliegue de un modelo sobre entrenamiento, en el orden de las filas de entrenamiento."""
    balanceo, optimizador = plan.combinacion(tarea, modelo)
    if modelo in plan.BENCHMARKS_E3[tarea]:
        return oof_e3.loc[indice_train, f'{tarea}__{modelo}__{balanceo}__{optimizador}__fuera_de_pliegue'].to_numpy()
    return np.load(resolver(fila_de(tabla, tarea, modelo)['ruta_predicciones']))


def ece(y, p, n_bins: int = 10) -> float:
    """Error de calibración esperado con tramos de igual número de clientes."""
    orden = np.argsort(p)
    total = 0.0
    for tramo in np.array_split(orden, n_bins):
        total += len(tramo) / len(p) * abs(p[tramo].mean() - y[tramo].mean())
    return float(total)


def main() -> None:
    p = plan.problema()
    c = p['conjuntos']
    tabla = plan.registro().cargar()
    pred_e3 = pd.read_parquet(plan.PREDICCIONES_E3)
    oof_e3 = pd.read_parquet(E3 / 'complementos' / 'fuera_de_pliegue.parquet')
    assert set(pred_e3.index) == set(c.X_test.index) and pred_e3.index.is_unique, \
        'las predicciones del Entregable 3 no corresponden a estas filas de prueba'
    pred_e3 = pred_e3.loc[c.X_test.index]                      # el Entregable 3 las guardó ordenadas por índice
    y_bin, y_cont = p['y_test'].to_numpy(), c.y_test_cont.to_numpy()
    y_bin_tr, y_cont_tr = p['y_train'].to_numpy(), c.y_train_cont.to_numpy()
    pred = pd.DataFrame(index=c.X_test.index)
    metricas, incertidumbre, finales = [], [], {}

    for tarea in ('clasificacion', 'regresion'):
        for modelo in plan.modelos(tarea):
            balanceo, optimizador = plan.combinacion(tarea, modelo)
            clave = f'{tarea}__{modelo}'
            f = fila_de(tabla, tarea, modelo)
            t_inferencia = np.nan
            if modelo in plan.BENCHMARKS_E3[tarea]:
                clave_e3 = f'{tarea}__{modelo}__{balanceo}__{optimizador}'
                pred[clave] = pred_e3[clave_e3]
                if tarea == 'clasificacion':
                    pred[clave + '__clase'] = pred_e3[clave_e3 + '__clase']
            else:
                final = finales[clave] = joblib.load(resolver(f['ruta_modelo']))
                t0 = time.perf_counter()
                if tarea == 'clasificacion':
                    clase, puntaje, _ = evaluacion.puntajes(final, c.X_test)
                    pred[clave], pred[clave + '__clase'] = puntaje, clase
                else:
                    pred[clave] = final.predict(c.X_test)
                t_inferencia = time.perf_counter() - t0
            puntaje = pred[clave].to_numpy()
            base = {'tarea': tarea, 'modelo': modelo, 'balanceo': balanceo, 'optimizador': optimizador,
                    'metrica_cv': abs(float(f['media_ext'])), 'desv_cv': float(f['desv_ext']), 'inferencia_s': t_inferencia}
            if tarea == 'clasificacion':
                oof = fuera_de_pliegue(tabla, tarea, modelo, oof_e3, c.X_train.index)
                calibrada = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(oof, y_bin_tr).predict(puntaje)
                pred[clave + '__isotonica'] = calibrada
                es_prob = modelo != 'svm'
                campana = pruebas.marcar_campana(puntaje, 0.25)
                recortar = lambda v: np.clip(v, 1e-6, 1 - 1e-6)        # noqa: E731
                metricas.append({**base, 'auc': roc_auc_score(y_bin, puntaje), 'ap': average_precision_score(y_bin, puntaje),
                                 'captura_25': float(y_bin[campana == 1].sum() / y_bin.sum()),
                                 'logloss': log_loss(y_bin, recortar(puntaje)) if es_prob else np.nan,
                                 'brier': brier_score_loss(y_bin, puntaje) if es_prob else np.nan,
                                 'ece': ece(y_bin, puntaje) if es_prob else np.nan,
                                 'logloss_isotonica': log_loss(y_bin, recortar(calibrada)),
                                 'brier_isotonica': brier_score_loss(y_bin, calibrada), 'ece_isotonica': ece(y_bin, calibrada)})
            else:
                metricas.append({**base, **evaluacion.metricas_regresion(y_cont, puntaje)})
                oof = fuera_de_pliegue(tabla, tarea, modelo, oof_e3, c.X_train.index)
                errores = np.sort(np.abs(y_cont_tr - oof))
                for cobertura in COBERTURAS:
                    q = errores[min(int(np.ceil((len(errores) + 1) * cobertura)) - 1, len(errores) - 1)]
                    incertidumbre.append({'modelo': modelo, 'metodo': 'intervalo conforme (pronóstico ± q)', 'cobertura_nominal': cobertura,
                                          'cobertura': float(np.mean(np.abs(y_cont - puntaje) <= q)), 'largo': float(2 * q)})

    # lo que solo tiene el modelo original: índice, mediana, intervalo central y conjunto de predicción
    tuberia = finales.get(f'regresion__{plan.ORIGINAL}')
    if tuberia is not None:
        bdl = tuberia.named_steps['modelo']
        Z = tuberia[:-1].transform(c.X_test)
        pred['regresion__bdl__indice'] = bdl.indice(Z)
        pred['regresion__bdl__mediana'] = bdl.cuantil(Z, 0.5)
        for cobertura in COBERTURAS:
            bajo, alto = bdl.intervalo(Z, cobertura)
            incertidumbre.append({'modelo': plan.ORIGINAL, 'metodo': 'intervalo central del modelo', 'cobertura_nominal': cobertura,
                                  'cobertura': float(np.mean((y_cont >= bajo) & (y_cont <= alto))), 'largo': float(np.mean(alto - bajo))})
            tramos = bdl.conjunto_residual(cobertura)
            incertidumbre.append({'modelo': plan.ORIGINAL, 'metodo': 'conjunto de predicción del modelo', 'cobertura_nominal': cobertura,
                                  'cobertura': float(bdl.conjunto_cubre(Z, y_cont, cobertura).mean()),
                                  'largo': float((tramos[:, 1] - tramos[:, 0]).sum()), 'tramos': len(tramos)})
        m = next(f for f in metricas if f['tarea'] == 'regresion' and f['modelo'] == plan.ORIGINAL)
        m['mae_mediana'] = float(np.mean(np.abs(y_cont - pred['regresion__bdl__mediana'])))
        m.update({'K': int(bdl.n_componentes_), 'arboles': int(bdl.n_arboles_)})

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(CARPETA_RESULTADOS / 'predicciones_prueba.parquet')
    tm = pd.DataFrame(metricas)
    tm.to_csv(CARPETA_RESULTADOS / 'metricas_prueba.csv', index=False)
    pd.DataFrame(incertidumbre).to_csv(CARPETA_RESULTADOS / 'incertidumbre_prueba.csv', index=False)
    pd.set_option('display.width', 250, 'display.float_format', lambda v: f'{v:.5f}')
    print(tm[tm.tarea == 'clasificacion'].drop(columns=['tarea', 'rmse', 'mae', 'r2', 'mae_mediana', 'K', 'arboles'], errors='ignore')
          .sort_values('auc', ascending=False).to_string(index=False))
    print(tm[tm.tarea == 'regresion'][['modelo', 'optimizador', 'metrica_cv', 'desv_cv', 'rmse', 'mae', 'r2'] + [k for k in ('mae_mediana', 'K', 'arboles') if k in tm]]
          .sort_values('rmse').to_string(index=False))
    print(pd.DataFrame(incertidumbre).query('cobertura_nominal == 0.9').to_string(index=False))


if __name__ == '__main__':
    main()
