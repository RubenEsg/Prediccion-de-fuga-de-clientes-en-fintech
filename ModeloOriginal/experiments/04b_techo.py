"""Techo teórico del problema: lo mejor que puede hacer cualquier modelo que no reciba ``active_products``.

El puntaje de COFINFAD es un índice que depende de los predictores menos un escalón por cada producto activo:
``s = g(x) − δ·(productos − 1)``. Como ``active_products`` se excluye y no se puede predecir con las demás variables,
para un cliente con índice ``g(x)`` lo único que se puede saber es la distribución de su escalón. De ahí salen, en
forma cerrada, las mejores métricas alcanzables:

- regresión: predecir ``g(x) − δ·E[productos − 1]`` (la media) da el RMSE mínimo; predecir el nivel mediano, el MAE mínimo;
- clasificación: la probabilidad de riesgo alto es ``Σ_k π_k·1[g(x) − δ·(k − 1) > umbral]``, una función escalonada del
  índice; ordenar por ella da el AUC máximo y usarla como probabilidad, la pérdida logarítmica mínima.

``active_products`` se usa aquí **solo para este cálculo**, que no interviene en el ajuste de ningún modelo. ``δ`` y los
pesos ``π_k`` se estiman en entrenamiento.

Salida: ``resultados/techo.json``. Uso: ``python experiments/04b_techo.py``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import plan                                                       # noqa: E402
from src.config import CARPETA_RESULTADOS                                  # noqa: E402

TOLERANCIA = 1e-9
"""Un nivel que cae exactamente en el umbral no lo supera (la etiqueta es ``s > umbral``); la tolerancia evita que el
redondeo de la suma lo decida."""


def main() -> None:
    p = plan.problema()
    c, df, part, umbral = p['conjuntos'], p['df'], p['particion'], p['etiqueta'].umbral
    ap_tr = df.loc[part.idx_train, 'active_products'].to_numpy()
    ap_te = df.loc[part.idx_test, 'active_products'].to_numpy()
    s_tr, s_te = c.y_train_cont.to_numpy(), c.y_test_cont.to_numpy()

    # δ: pendiente del puntaje sobre el número de productos dentro de cada celda de los predictores que definen el índice
    celda = df.loc[part.idx_train, ['age', 'app_logins_frequency', 'satisfaction_score']].astype(str).agg('-'.join, axis=1).to_numpy()
    t = pd.DataFrame({'celda': celda, 's': s_tr, 'ap': ap_tr})
    t = t[df.loc[part.idx_train, 'tx_count'].to_numpy() <= 300]
    ds, da = t['s'] - t.groupby('celda')['s'].transform('mean'), t['ap'] - t.groupby('celda')['ap'].transform('mean')
    delta = float(-(ds * da).sum() / (da ** 2).sum())
    pi = pd.Series(ap_tr).value_counts(normalize=True).sort_index()
    niveles = pi.index.to_numpy()
    media_escalon = float(delta * ((niveles - 1) * pi.to_numpy()).sum())
    acumulado = np.cumsum(pi.to_numpy())
    nivel_mediano = int(niveles[np.searchsorted(acumulado, 0.5)])

    salida = {'delta': delta, 'pesos': {int(k): float(v) for k, v in pi.items()}, 'umbral': umbral, 'nivel_mediano': nivel_mediano,
              'desviacion_del_escalon': float(delta * np.sqrt((((niveles - 1) - (niveles - 1) @ pi.to_numpy()) ** 2) @ pi.to_numpy()))}
    for nombre, s, ap in (('entrenamiento', s_tr, ap_tr), ('prueba', s_te, ap_te)):
        g = s + delta * (ap - 1)                                   # índice del cliente: su puntaje sin escalón
        y = (s > umbral).astype(int)
        prob = np.zeros(len(g))
        for k, peso in zip(niveles, pi.to_numpy()):
            prob += peso * (g - delta * (k - 1) > umbral + TOLERANCIA)
        recortada = np.clip(prob, 1e-6, 1 - 1e-6)
        mesetas = pd.Series(np.round(prob, 4)).value_counts(normalize=True).sort_index()
        salida[nombre] = {
            'n': int(len(s)), 'prevalencia': float(y.mean()),
            'auc_maximo': float(roc_auc_score(y, prob)), 'logloss_minima': float(log_loss(y, recortada)),
            'brier_minimo': float(brier_score_loss(y, prob)),
            'rmse_minimo': float(np.sqrt(np.mean((s - (g - media_escalon)) ** 2))),
            'mae_minimo': float(np.mean(np.abs(s - (g - delta * (nivel_mediano - 1))))),
            'r2_maximo': float(1 - np.mean((s - (g - media_escalon)) ** 2) / np.var(s)),
            'desviacion_del_indice': float(np.std(g)), 'desviacion_del_puntaje': float(np.std(s)),
            'mesetas': {str(k): float(v) for k, v in mesetas.items()},
            'frecuencia_por_meseta': {str(k): float(v) for k, v in pd.Series(y).groupby(np.round(prob, 4)).mean().items()}}
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    (CARPETA_RESULTADOS / 'techo.json').write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(salida, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
