# Ronda 4: modelos de clase con un cambio (5 pliegues sobre entrenamiento)

Δ = diferencia media pareada frente a HGB directo con todas las variables; p = t pareada.

| Experimento | Modelo de clase de partida | AUC mediana | Δ (p) | AUC percentil 75 | R² | Δ (p) |
|---|---|---|---|---|---|---|
| hgb directo, todas las variables [referencia] | referencia | 0.7225 ± 0.0032 | — | 0.8328 ± 0.0037 | 0.2234 ± 0.0035 | — |
| logistica / ridge lineal, todas [Entrega 2] | logística, Ridge | 0.6824 ± 0.0043 | -0.0401 (0.000) | 0.7641 ± 0.0044 | 0.1511 ± 0.0065 | -0.0724 (0.000) |
| knn k=200, todas las variables [Entrega 2] | k-NN | 0.6645 ± 0.0040 | -0.0580 (0.000) | 0.7525 ± 0.0065 | 0.0971 ± 0.0027 | -0.1264 (0.000) |
| knn k=200, 4 variables | k-NN | 0.7211 ± 0.0034 | -0.0014 (0.088) | 0.8322 ± 0.0053 | 0.2232 ± 0.0042 | -0.0003 (0.559) |
| knn k=200, 6 variables | k-NN | 0.7201 ± 0.0027 | -0.0024 (0.036) | 0.8274 ± 0.0038 | 0.2203 ± 0.0042 | -0.0032 (0.003) |
| knn k=200, 8 variables | k-NN | 0.7180 ± 0.0030 | -0.0045 (0.009) | 0.8258 ± 0.0044 | 0.2167 ± 0.0036 | -0.0068 (0.000) |
| naive bayes, 6 variables | Naive Bayes | 0.6763 ± 0.0050 | -0.0462 (0.000) | 0.7477 ± 0.0093 | — | — |
| random forest, 6 variables | Random Forest | 0.7239 ± 0.0034 | +0.0013 (0.122) | 0.8331 ± 0.0034 | 0.2091 ± 0.0074 | -0.0143 (0.001) |
| svm rbf (nystroem), 6 variables | SVM | 0.7219 ± 0.0039 | -0.0006 (0.381) | 0.8326 ± 0.0036 | 0.2254 ± 0.0050 | +0.0020 (0.122) |
| logistica / ridge + splines, 6 variables | logística, Ridge | 0.7212 ± 0.0043 | -0.0013 (0.305) | 0.8288 ± 0.0021 | 0.2285 ± 0.0048 | +0.0050 (0.002) |
| logistica / ridge + splines, 12 variables | logística, Ridge | 0.7212 ± 0.0041 | -0.0014 (0.273) | 0.8289 ± 0.0022 | 0.2284 ± 0.0051 | +0.0049 (0.004) |
| logistica / ridge + splines + interacciones, 6 variables | logística, Ridge | 0.7204 ± 0.0043 | -0.0021 (0.163) | 0.8275 ± 0.0028 | 0.2284 ± 0.0050 | +0.0049 (0.004) |
| logistica / ridge + splines (10 nudos) + interacciones, 8 variables | logística, Ridge | 0.7207 ± 0.0044 | -0.0018 (0.213) | 0.8272 ± 0.0027 | 0.2281 ± 0.0050 | +0.0047 (0.004) |
| logit leaf model, todas las variables, 4 hojas | árbol + logística | 0.7170 ± 0.0032 | -0.0055 (0.020) | 0.8238 ± 0.0023 | 0.2221 ± 0.0050 | -0.0014 (0.421) |
| logit leaf model, 6 variables, 4 hojas | árbol + logística | 0.7214 ± 0.0039 | -0.0011 (0.416) | 0.8261 ± 0.0024 | 0.2278 ± 0.0053 | +0.0043 (0.025) |
| logit leaf model, 6 variables, 8 hojas | árbol + logística | 0.7213 ± 0.0037 | -0.0012 (0.410) | 0.8292 ± 0.0029 | 0.2275 ± 0.0051 | +0.0041 (0.028) |
