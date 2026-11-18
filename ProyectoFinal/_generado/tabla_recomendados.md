| Tarea | Modelo | Recomendados: externo | Ajustados: externo | Pliegues en que gana el ajustado | Recomendados: prueba | Ajustados: prueba | Ajuste con los recomendados (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clasificación | EBM | 0,8277 ± 0,0023 | 0,8292 ± 0,0037 | 3 de 5 | 0,8255 | 0,8289 | 380,7 |
| Clasificación | LightGBM | 0,8303 ± 0,0033 | 0,8310 ± 0,0038 | 3 de 5 | 0,8270 | 0,8296 | 0,9 |
| Regresión | EBM | 0,05913 ± 0,00044 | 0,05903 ± 0,00043 | 5 de 5 | 0,05929 | 0,05925 | 505,6 |
| Regresión | LightGBM | 0,05927 ± 0,00039 | 0,05899 ± 0,00040 | 5 de 5 | 0,05948 | 0,05918 | 0,9 |

Métrica: AUC en clasificación y RMSE en regresión. Externo: media ± desviación en los cinco pliegues externos de la semilla 42, los mismos para las dos versiones. Recomendados: los valores por defecto de la librería (interpret y LightGBM), con el mismo preprocesamiento y el mismo tratamiento del desbalance que la combinación elegida.

**Valores recomendados.** EBM en clasificación: `learning_rate` = 0,015; `max_leaves` = 2; `min_samples_leaf` = 4; `interactions` = «3x»; `max_bins` = 1.024; `outer_bags` = 14. Fuera del alcance de la búsqueda: `interactions` = «3x» (el espacio va de 0 a 20); `outer_bags` = 14 (no se busca; el modelo final usa 8). LightGBM en clasificación: `n_estimators` = 100; `num_leaves` = 31; `learning_rate` = 0,1; `subsample` = 1; `colsample_bytree` = 1; `reg_lambda` = 0; `min_child_samples` = 20. Fuera del alcance de la búsqueda: `reg_lambda` = 0 (el espacio va de 0,01 a 10). EBM en regresión: `learning_rate` = 0,04; `max_leaves` = 2; `min_samples_leaf` = 4; `interactions` = «5x»; `max_bins` = 1.024; `outer_bags` = 14. Fuera del alcance de la búsqueda: `interactions` = «5x» (el espacio va de 0 a 20); `outer_bags` = 14 (no se busca; el modelo final usa 8). LightGBM en regresión: `n_estimators` = 100; `num_leaves` = 31; `learning_rate` = 0,1; `subsample` = 1; `colsample_bytree` = 1; `reg_lambda` = 0; `min_child_samples` = 20. Fuera del alcance de la búsqueda: `reg_lambda` = 0 (el espacio va de 0,01 a 10).

**Lectura.** En los pliegues externos la versión ajustada mejora a la recomendada en 4 de los 4 casos; en prueba, en 4 de 4. El ajuste no dejó a ningún benchmark reciente por debajo de su punto de partida.
