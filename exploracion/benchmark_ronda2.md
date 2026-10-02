# Ronda 2: techo de información y variantes (5 pliegues sobre entrenamiento)

Mismos pliegues para todas las filas (estratificados por la etiqueta de la mediana, semilla 42). La columna Δ es la diferencia media pareada frente a XGBoost directo y p su prueba t pareada.

| Experimento | Tipo | AUC mediana | Δ (p) | AUC percentil 75 | R² | Δ (p) | min |
|---|---|---|---|---|---|---|---|
| xgboost directo [clase, referencia] | referencia | 0.7220 ± 0.0033 | — | 0.8307 ± 0.0030 | 0.2224 ± 0.0043 | — | 0.2 |
| hgb directo [referencia] | referencia | 0.7225 ± 0.0032 | +0.0005 (0.408) | 0.8328 ± 0.0037 | 0.2234 ± 0.0035 | +0.0011 (0.124) | 0.2 |
| ORACULO con active_products (diagnostico, no candidato) | diagnóstico | 0.9999 ± 0.0001 | +0.2778 (0.000) | 0.9999 ± 0.0001 | 0.9994 ± 0.0001 | +0.7771 (0.000) | 0.2 |
| TECHO: oraculo marginalizado sobre active_products (LUPI) | techo / LUPI | 0.7232 ± 0.0040 | +0.0012 (0.191) | 0.8302 ± 0.0023 | 0.2303 ± 0.0051 | +0.0079 (0.000) | 0.2 |
| LUPI objetivo depurado + hgb | LUPI | 0.7232 ± 0.0040 | +0.0011 (0.205) | 0.8303 ± 0.0022 | 0.2303 ± 0.0051 | +0.0080 (0.000) | 0.1 |
| LUPI objetivo depurado + xgboost | LUPI | 0.7231 ± 0.0039 | +0.0011 (0.185) | 0.8302 ± 0.0024 | 0.2303 ± 0.0050 | +0.0080 (0.000) | 0.1 |
| seleccion de variables k=6 + hgb | clase + cambio | 0.7225 ± 0.0043 | +0.0004 (0.499) | 0.8360 ± 0.0039 | 0.2254 ± 0.0046 | +0.0030 (0.019) | 0.2 |
| seleccion de variables k=12 + hgb | clase + cambio | 0.7233 ± 0.0045 | +0.0012 (0.226) | 0.8353 ± 0.0046 | 0.2247 ± 0.0043 | +0.0023 (0.036) | 0.2 |
| seleccion de variables k=24 + hgb | clase + cambio | 0.7226 ± 0.0038 | +0.0005 (0.593) | 0.8332 ± 0.0061 | 0.2240 ± 0.0039 | +0.0016 (0.082) | 0.1 |
| logistica / ridge con splines (GAM) | clase + cambio | 0.7198 ± 0.0039 | -0.0022 (0.087) | 0.8272 ± 0.0022 | 0.2267 ± 0.0050 | +0.0044 (0.001) | 0.2 |
| hibrido GBDT + logistica | clase + cambio | 0.7180 ± 0.0029 | -0.0041 (0.010) | 0.8304 ± 0.0056 | — | — | 0.2 |
| xgboost DART | clase + cambio | 0.7225 ± 0.0033 | +0.0004 (0.081) | 0.8315 ± 0.0027 | — | — | 0.7 |
| SVM RBF via Nystroem | clase + cambio | 0.6916 ± 0.0050 | -0.0305 (0.000) | 0.7914 ± 0.0032 | 0.1727 ± 0.0036 | -0.0496 (0.000) | 1.1 |
| blending xgb+hgb+lgbm+catboost | mezcla | 0.7232 ± 0.0029 | +0.0012 (0.035) | 0.8317 ± 0.0038 | 0.2248 ± 0.0042 | +0.0024 (0.000) | 2.9 |
| xgboost afinado con Optuna (25 trials) | clase + cambio | 0.7236 ± 0.0030 | +0.0016 (0.058) | — | 0.2247 ± 0.0043 | +0.0024 (0.001) | 12.8 |
