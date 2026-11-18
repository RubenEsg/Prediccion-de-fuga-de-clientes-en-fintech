| Tarea | Modelo | Hiperparámetros finales (semilla 42) |
| :--- | :--- | :--- |
| Clasificación | k-NN | `n_neighbors` = 600; `weights` = «distance»; `p` = 1 |
| Clasificación | Naive Bayes | `var_smoothing` = 0,001 |
| Clasificación | Regresión logística | `C` = 0,05631; `l1_ratio` = 1 |
| Clasificación | Árbol de decisión | `ccp_alpha` = 3,9 × 10⁻⁴; `criterion` = «gini»; `max_depth` = 22; `min_samples_leaf` = 7 |
| Clasificación | Random Forest | `max_depth` = 32; `max_features` = 0,5; `min_samples_leaf` = 11; `n_estimators` = 300 |
| Clasificación | XGBoost | `n_estimators` = 145; `max_depth` = 6; `learning_rate` = 0,003304; `subsample` = 0,8081; `colsample_bytree` = 0,9683; `reg_lambda` = 0,8792; `min_child_weight` = 11 |
| Clasificación | SVM lineal | `C` = 0,001083 |
| Clasificación | EBM | `interactions` = 0; `learning_rate` = 0,008; `max_leaves` = 2; `min_samples_leaf` = 100; `outer_bags` = 8 |
| Clasificación | LightGBM | `n_estimators` = 244; `num_leaves` = 16; `learning_rate` = 0,02042; `subsample` = 0,7871; `colsample_bytree` = 0,9424; `reg_lambda` = 0,03961; `min_child_samples` = 150 |
| Clasificación | **BDL (original)** | `tasa` = 0,09834; `profundidad` = 1; `min_hoja` = 11; `lambda_l2` = 0,02269; `max_bins` = 73; `n_componentes` = «auto»; `usar_puntaje` = sí |
| Regresión | k-NN | `n_neighbors` = 80; `p` = 1; `weights` = «distance» |
| Regresión | Ridge | `alpha` = 613,9 |
| Regresión | Lasso | `alpha` = 1,1 × 10⁻⁴ |
| Regresión | Árbol de decisión | `max_depth` = 13; `min_samples_leaf` = 101; `ccp_alpha` = 1,8 × 10⁻⁶ |
| Regresión | Random Forest | `max_depth` = 8; `max_features` = 0,5; `min_samples_leaf` = 50; `n_estimators` = 300 |
| Regresión | XGBoost | `n_estimators` = 512; `max_depth` = 2; `learning_rate` = 0,02361; `subsample` = 0,8025; `colsample_bytree` = 0,7268; `reg_lambda` = 0,2158; `min_child_weight` = 31 |
| Regresión | SVR lineal | `C` = 46,43; `epsilon` = 3,8 × 10⁻⁴ |
| Regresión | EBM | `interactions` = 0; `learning_rate` = 0,008; `max_leaves` = 3; `min_samples_leaf` = 20; `outer_bags` = 8 |
| Regresión | LightGBM | `n_estimators` = 260; `num_leaves` = 4; `learning_rate` = 0,05085; `subsample` = 0,7569; `colsample_bytree` = 0,5365; `reg_lambda` = 1,686; `min_child_samples` = 104 |
| Regresión | **BDL (original)** | `tasa` = 0,5816; `profundidad` = 1; `min_hoja` = 88; `lambda_l2` = 0,01508; `max_bins` = 885; `n_componentes` = «auto» |

La tabla lista los hiperparámetros que fijó el optimizador elegido para cada modelo. Cuando ese optimizador es la rejilla, los que la rejilla no recorre quedan en el valor de partida del modelo: EBM (clasificación), `max_bins` = 1.024; EBM (regresión), `max_bins` = 1.024.
