| Modelo | Tarea | Hiperparámetros y rango de búsqueda |
| :--- | :--- | :--- |
| k-NN | Las dos | `n_neighbors`: 5 a 600 (entero, escala logarítmica); `weights`: {«uniform», «distance»}; `p`: {1, 2} |
| Naive Bayes | Clasificación | `var_smoothing`: 10⁻¹¹ a 0,001 (escala logarítmica) |
| Regresión logística | Clasificación | `C`: 0,001 a 100 (escala logarítmica); `l1_ratio`: {1, 0} |
| Árbol de decisión | Clasificación | `max_depth`: 3 a 30 (entero); `min_samples_leaf`: 1 a 300 (entero, escala logarítmica); `ccp_alpha`: 10⁻⁶ a 0,01 (escala logarítmica); `criterion`: {«gini», «entropy»} |
| Árbol de decisión | Regresión | `max_depth`: 3 a 30 (entero); `min_samples_leaf`: 1 a 300 (entero, escala logarítmica); `ccp_alpha`: 10⁻⁸ a 10⁻⁴ (escala logarítmica) |
| Random Forest | Las dos | `max_depth`: 4 a 40 (entero); `max_features`: {«sqrt», «log2», 0,3, 0,5}; `min_samples_leaf`: 1 a 100 (entero, escala logarítmica) |
| XGBoost | Las dos | `n_estimators`: 100 a 1.000 (entero, escala logarítmica); `max_depth`: 2 a 10 (entero); `learning_rate`: 0,001 a 0,3 (escala logarítmica); `subsample`: 0,5 a 1 (continuo); `colsample_bytree`: 0,5 a 1 (continuo); `reg_lambda`: 0,01 a 10 (escala logarítmica); `min_child_weight`: 1 a 50 (entero, escala logarítmica) |
| SVM lineal | Clasificación | `C`: 10⁻⁴ a 100 (escala logarítmica) |
| EBM | Las dos | `learning_rate`: 0,008 a 0,1 (escala logarítmica); `max_leaves`: 2 a 4 (entero); `min_samples_leaf`: 2 a 200 (entero, escala logarítmica); `interactions`: 0 a 20 (entero); `max_bins`: 64 a 1.024 (entero, escala logarítmica) |
| LightGBM | Las dos | `n_estimators`: 100 a 1.000 (entero, escala logarítmica); `num_leaves`: 4 a 127 (entero, escala logarítmica); `learning_rate`: 0,001 a 0,3 (escala logarítmica); `subsample`: 0,5 a 1 (continuo); `colsample_bytree`: 0,5 a 1 (continuo); `reg_lambda`: 0,01 a 10 (escala logarítmica); `min_child_samples`: 5 a 200 (entero, escala logarítmica) |
| **BDL (original)** | Clasificación | `tasa`: 0,05 a 1 (escala logarítmica); `profundidad`: 1 a 6 (entero); `min_hoja`: 5 a 200 (entero, escala logarítmica); `lambda_l2`: 0,01 a 10 (escala logarítmica); `max_bins`: 32 a 1.024 (entero, escala logarítmica); `n_componentes`: {«auto», 1}; `usar_puntaje`: {sí, no} |
| **BDL (original)** | Regresión | `tasa`: 0,05 a 1 (escala logarítmica); `profundidad`: 1 a 6 (entero); `min_hoja`: 5 a 200 (entero, escala logarítmica); `lambda_l2`: 0,01 a 10 (escala logarítmica); `max_bins`: 32 a 1.024 (entero, escala logarítmica); `n_componentes`: {«auto», 1} |
| Ridge | Regresión | `alpha`: 0,001 a 10⁵ (escala logarítmica) |
| Lasso | Regresión | `alpha`: 10⁻⁶ a 0,1 (escala logarítmica) |
| SVR lineal | Regresión | `C`: 0,001 a 100 (escala logarítmica); `epsilon`: 10⁻⁴ a 0,05 (escala logarítmica) |
