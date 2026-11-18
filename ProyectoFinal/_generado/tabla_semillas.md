**Estabilidad: métrica en prueba con diez semillas.**

| Tarea | Modelo | Media | Desviación | IC 95 % de la media | Mínimo | Máximo |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Regresión | BDL (original) | 0,059116 | 0,000002 | [0,059115; 0,059118] | 0,059114 | 0,059119 |
| Regresión | XGBoost | 0,059171 | 0,000005 | [0,059168; 0,059175] | 0,059162 | 0,059178 |
| Regresión | LightGBM | 0,059183 | 0,000012 | [0,059174; 0,059192] | 0,059157 | 0,059203 |
| Regresión | EBM | 0,059259 | 0,000007 | [0,059254; 0,059263] | 0,059247 | 0,059269 |
| Regresión | Random Forest | 0,059306 | 0,000010 | [0,059299; 0,059314] | 0,059288 | 0,059321 |
| Regresión | Árbol de decisión | 0,059436 | 0,000000 | [0,059436; 0,059436] | 0,059436 | 0,059436 |
| Regresión | Lasso | 0,061661 | 0,000000 | [0,061661; 0,061661] | 0,061661 | 0,061661 |
| Regresión | Ridge | 0,061668 | 0,000000 | [0,061668; 0,061668] | 0,061668 | 0,061668 |
| Regresión | SVR lineal | 0,061703 | 0,000000 | [0,061703; 0,061703] | 0,061703 | 0,061703 |
| Regresión | k-NN | 0,063206 | 0,000000 | [0,063206; 0,063206] | 0,063206 | 0,063206 |
| Clasificación | BDL (original) | 0,83097 | 0,00014 | [0,83086; 0,83107] | 0,83072 | 0,83122 |
| Clasificación | LightGBM | 0,82919 | 0,00066 | [0,82872; 0,82966] | 0,82773 | 0,83015 |
| Clasificación | EBM | 0,82881 | 0,00009 | [0,82874; 0,82887] | 0,82867 | 0,82894 |
| Clasificación | XGBoost | 0,82811 | 0,00023 | [0,82794; 0,82827] | 0,82765 | 0,82853 |
| Clasificación | Random Forest | 0,82805 | 0,00136 | [0,82708; 0,82902] | 0,82601 | 0,82981 |
| Clasificación | Árbol de decisión | 0,82429 | 0,00000 | [0,82429; 0,82429] | 0,82429 | 0,82429 |
| Clasificación | k-NN | 0,78736 | 0,00000 | [0,78736; 0,78736] | 0,78736 | 0,78736 |
| Clasificación | Regresión logística | 0,77086 | 0,00000 | [0,77086; 0,77086] | 0,77086 | 0,77086 |
| Clasificación | SVM lineal | 0,77058 | 0,00000 | [0,77058; 0,77058] | 0,77058 | 0,77058 |
| Clasificación | Naive Bayes | 0,66625 | 0,00000 | [0,66625; 0,66625] | 0,66625 | 0,66625 |

**Cómputo de cada modelo final.**

| Tarea | Modelo | Ajuste (s) | Inferencia (s) | Memoria (MB) | Tamaño (KB) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Regresión | BDL (original) | 5,8 | 0,07 | 153 | 149 |
| Regresión | XGBoost | 1,7 | 0,06 | 154 | 58 |
| Regresión | LightGBM | 0,6 | 0,06 | 66 | 46 |
| Regresión | EBM | 25,5 | 0,05 | 81 | 1.032 |
| Regresión | Random Forest | 11,2 | 0,13 | 66 | 2.347 |
| Regresión | Árbol de decisión | 0,8 | 0,05 | 55 | 6 |
| Regresión | Lasso | 0,4 | 0,05 | 65 | 4 |
| Regresión | Ridge | 0,4 | 0,05 | 67 | 4 |
| Regresión | SVR lineal | 0,6 | 0,05 | 72 | 4 |
| Regresión | k-NN | 0,3 | 4,89 | 64 | 6.295 |
| Clasificación | BDL (original) | 11,2 | 0,15 | 158 | 90 |
| Clasificación | LightGBM | 1,2 | 0,17 | 64 | 187 |
| Clasificación | EBM | 23,4 | 0,14 | 91 | 422 |
| Clasificación | XGBoost | 1,1 | 0,13 | 162 | 193 |
| Clasificación | Random Forest | 16,2 | 0,30 | 111 | 12.992 |
| Clasificación | Árbol de decisión | 1,1 | 0,11 | 65 | 6 |
| Clasificación | k-NN | 0,2 | 9,68 | 227 | 6.208 |
| Clasificación | Regresión logística | 1,2 | 0,11 | 101 | 4 |
| Clasificación | SVM lineal | 0,8 | 0,11 | 72 | 4 |
| Clasificación | Naive Bayes | 0,3 | 0,12 | 78 | 6 |

Diez semillas (42 a 51) con los hiperparámetros finales; solo cambia la semilla del modelo, no la partición ni la búsqueda. IC 95 % de la media: con la *t* de Student sobre las diez semillas. Los modelos deterministas dan el mismo valor las diez veces. Ajuste e inferencia: media de las diez semillas; la inferencia es sobre los 9.745 clientes de prueba. Memoria: aumento máximo del conjunto de trabajo del proceso al ajustar el modelo y predecir la prueba, medido en un proceso aparte (`experiments/03c_memoria.py`). Tamaño: el modelo serializado y comprimido.
