# Benchmark de candidatos a modelo nuevo (5 pliegues sobre entrenamiento)

Semilla 42, misma partición 80/20 que la Entrega 2, preprocesamiento dentro de cada pliegue. Sin búsqueda de hiperparámetros: sirve para descartar y priorizar, no para reportar.

## Clasificación, etiqueta = mediana (comparable con la Entrega 2)

Referencia Entrega 2: AUC 0,6832 en prueba (logística L1, C = 0,1).

| Modelo | Visto en clase | AUC media ± desv | s por ajuste |
|---|---|---|---|
| stacking logit+hgb+knn+nb -> logit | **no** | 0.7231 ± 0.0043 | 5.5 |
| hist gradient boosting | **no** | 0.7225 ± 0.0032 | 1.1 |
| lightgbm | **no** | 0.7222 ± 0.0027 | 1.4 |
| xgboost [clase] | sí | 0.7220 ± 0.0033 | 0.9 |
| catboost | **no** | 0.7218 ± 0.0023 | 5.3 |
| ebm (GA2M) | **no** | 0.7214 ± 0.0029 | 10.0 |
| reg→rank: hgb regresor | **no** | 0.7205 ± 0.0027 | 0.8 |
| reg→rank: catboost regresor | **no** | 0.7196 ± 0.0029 | 3.6 |
| random forest [clase] | sí | 0.7173 ± 0.0034 | 2.8 |
| extra trees | **no** | 0.7124 ± 0.0031 | 2.9 |
| mlp (128,64) | **no** | 0.7073 ± 0.0032 | 3.2 |
| reg→rank: lasso 1e-4 | **no** | 0.6829 ± 0.0044 | 0.4 |
| logistica L1 C=0,1 [Entrega 2] | sí | 0.6824 ± 0.0043 | 0.7 |
| knn k=200 [Entrega 2] | sí | 0.6645 ± 0.0040 | 0.7 |

## Regresión sobre churn_probability

Referencia Entrega 2: R² 0,1511 y RMSE 0,0617 en prueba (Lasso, alpha = 1e-4).

| Modelo | Visto en clase | R2 media ± desv | RMSE medio | s por ajuste |
|---|---|---|---|---|
| ebm (GA2M) | **no** | 0.2246 ± 0.0090 | 0.0591 | 9.5 |
| hist gradient boosting | **no** | 0.2240 ± 0.0075 | 0.0591 | 0.8 |
| stacking lasso+hgb+knn -> ridge | **no** | 0.2238 ± 0.0084 | 0.0591 | 4.9 |
| xgboost [clase] | sí | 0.2237 ± 0.0087 | 0.0591 | 0.9 |
| catboost | **no** | 0.2235 ± 0.0095 | 0.0591 | 3.6 |
| lightgbm | **no** | 0.2204 ± 0.0083 | 0.0592 | 1.3 |
| random forest [clase] | sí | 0.2049 ± 0.0074 | 0.0598 | 16.7 |
| extra trees | **no** | 0.1956 ± 0.0117 | 0.0602 | 16.4 |
| lasso 1e-4 [Entrega 2] | sí | 0.1519 ± 0.0101 | 0.0618 | 0.5 |
| ridge [clase] | sí | 0.1509 ± 0.0103 | 0.0618 | 0.2 |
| knn k=200 [clase] | sí | 0.0970 ± 0.0041 | 0.0638 | 0.7 |
| mlp (128,64) | **no** | 0.0571 ± 0.1011 | 0.0651 | 6.2 |

## Clasificación, etiqueta = percentil 75 (etiqueta del Entregable 3)

Referencia Entrega 2: sin referencia directa: la Entrega 2 usó la mediana.

| Modelo | Visto en clase | AUC media ± desv | s por ajuste |
|---|---|---|---|
| hist gradient boosting | **no** | 0.8339 ± 0.0038 | 1.0 |
| stacking logit+hgb+knn+nb -> logit | **no** | 0.8332 ± 0.0036 | 7.4 |
| lightgbm | **no** | 0.8316 ± 0.0022 | 1.8 |
| catboost | **no** | 0.8316 ± 0.0022 | 5.2 |
| xgboost [clase] | sí | 0.8311 ± 0.0015 | 0.9 |
| ebm (GA2M) | **no** | 0.8291 ± 0.0034 | 9.5 |
| reg→rank: hgb regresor | **no** | 0.8285 ± 0.0038 | 0.8 |
| reg→rank: catboost regresor | **no** | 0.8275 ± 0.0030 | 3.5 |
| random forest [clase] | sí | 0.8262 ± 0.0030 | 2.5 |
| extra trees | **no** | 0.8176 ± 0.0030 | 2.6 |
| mlp (128,64) | **no** | 0.7780 ± 0.0160 | 2.6 |
| logistica L1 C=0,1 [Entrega 2] | sí | 0.7641 ± 0.0046 | 1.4 |
| reg→rank: lasso 1e-4 | **no** | 0.7636 ± 0.0047 | 0.4 |
| knn k=200 [Entrega 2] | sí | 0.7509 ± 0.0050 | 0.7 |
