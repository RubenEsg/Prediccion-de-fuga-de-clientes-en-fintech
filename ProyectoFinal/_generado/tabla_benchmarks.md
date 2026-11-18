**Clasificación (AUC, mayor es mejor), 15 pliegues externos.**

| Modelo | Combinación | Por qué esa combinación | Presupuesto | Evaluaciones | Tiempo (min) | AUC externo |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| XGBoost | ponderación de clases, genética | La mejor de 16; margen sobre la segunda: 0,00091 | 32 | 192 | 8,9 | 0,8333 ± 0,0041 |
| Árbol de decisión | ponderación de clases, aleatoria (Random) | La mejor de 16; margen sobre la segunda: 0,00027 | 48 | 288 | 1,8 | 0,8322 ± 0,0033 |
| Random Forest | sin balanceo, genética | La mejor de 16; margen sobre la segunda: 0,00035 | 36 | 216 | 20,9 | 0,8320 ± 0,0030 |
| LightGBM * | ponderación de clases, bayesiana (TPE) | La única, fijada en el protocolo | 32 | 192 | 13,2 | 0,8313 ± 0,0036 |
| **BDL (original)** | sin balanceo, bayesiana (TPE) | La única, fijada en el protocolo | 30 | 180 | 42,3 | 0,8313 ± 0,0038 |
| EBM * | ponderación de clases, rejilla (Grid) | La mejor de 16; margen sobre la segunda: 0,00012 | 36 | 216 | 18,7 | 0,8293 ± 0,0045 |
| k-NN | sin balanceo, genética | La mejor de 12; margen sobre la segunda: 0,00045 | 32 | 192 | 34,7 | 0,7841 ± 0,0069 |
| Regresión logística | ponderación de clases, genética | La mejor de 16; margen sobre la segunda: 0,00004 | 30 | 180 | 3,1 | 0,7644 ± 0,0060 |
| SVM lineal | ponderación de clases, rejilla (Grid) | La mejor de 16; margen sobre la segunda: 0,00002 | 30 | 180 | 2,1 | 0,7643 ± 0,0059 |
| Naive Bayes | sin balanceo, rejilla (Grid) | La mejor de 16; margen sobre la segunda: 0,00000 | 30 | 180 | 0,7 | 0,6715 ± 0,0117 |

**Regresión (RMSE, menor es mejor), 15 pliegues externos.**

| Modelo | Combinación | Por qué esa combinación | Presupuesto | Evaluaciones | Tiempo (min) | RMSE externo |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **BDL (original)** | bayesiana (TPE) | La única, fijada en el protocolo | 30 | 180 | 27,6 | 0,05887 ± 0,00047 |
| XGBoost | bayesiana (TPE) | La mejor de 4; margen sobre la segunda: 0,000012 | 32 | 192 | 10,4 | 0,05897 ± 0,00047 |
| LightGBM * | bayesiana (TPE) | La única, fijada en el protocolo | 32 | 192 | 6,8 | 0,05899 ± 0,00048 |
| EBM * | rejilla (Grid) | La mejor de 4; margen sobre la segunda: 0,000012 | 36 | 216 | 26,6 | 0,05903 ± 0,00047 |
| Random Forest | rejilla (Grid) | La mejor de 4; margen sobre la segunda: 0,000020 | 36 | 216 | 16,1 | 0,05913 ± 0,00049 |
| Árbol de decisión | genética | La mejor de 4; margen sobre la segunda: 0,000014 | 48 | 288 | 3,7 | 0,05928 ± 0,00048 |
| Lasso | genética | La mejor de 4; margen sobre la segunda: 0,000004 | 30 | 180 | 6,7 | 0,06181 ± 0,00053 |
| Ridge | genética | La mejor de 4; margen sobre la segunda: 0,000001 | 30 | 180 | 1,5 | 0,06183 ± 0,00053 |
| SVR lineal | bayesiana (TPE) | La mejor de 4; margen sobre la segunda: 0,000002 | 30 | 180 | 2,7 | 0,06185 ± 0,00052 |
| k-NN | rejilla (Grid) | La mejor de 4; margen sobre la segunda: 0,000007 | 32 | 192 | 24,9 | 0,06334 ± 0,00052 |

\* Método reciente. En negrita, el modelo original, que no es un benchmark: se incluye para comparar el presupuesto. Presupuesto: evaluaciones por búsqueda. Evaluaciones y tiempo: los de la repetición de la semilla 42 (cinco búsquedas externas y la final); los de los ocho modelos del Entregable 3 se midieron en ese entregable y los de LightGBM y el original, en este: son ejecuciones distintas, comparables en orden de magnitud y no al segundo. Margen: diferencia de la métrica externa entre la combinación elegida y la siguiente del mismo modelo.

**Mejor combinación global y finalistas.** En clasificación, la mejor combinación global entre los benchmarks es XGBoost con ponderación de clases y búsqueda genética: AUC externo de 0,8333 ± 0,0041. La prueba de Friedman rechaza que los diez modelos rindan igual (p < 0,001 con la corrección de Iman-Davenport) y son finalistas de Nemenyi los 6 modelos a menos de la diferencia crítica (3,50 rangos) del mejor rango medio: XGBoost, Árbol de decisión, Random Forest, LightGBM, BDL (original) y EBM. XGBoost está entre ellos, con rango medio 2,4. En regresión, la mejor combinación global entre los benchmarks es XGBoost con búsqueda bayesiana (TPE): RMSE externo de 0,05897 ± 0,00047. La prueba de Friedman rechaza que los diez modelos rindan igual (p < 0,001 con la corrección de Iman-Davenport) y son finalistas de Nemenyi los 4 modelos a menos de la diferencia crítica (3,50 rangos) del mejor rango medio: BDL (original), XGBoost, LightGBM y EBM. XGBoost está entre ellos, con rango medio 2,3. El *Model Confidence Set* al 90 % sobre los errores cuadráticos de prueba: entre los nueve benchmarks conserva a EBM, LightGBM y XGBoost, de modo que XGBoost pertenece al conjunto; con el modelo original incluido conserva a BDL (original) (XGBoost: p = 0,098). El *Model Confidence Set* no es una de las cuatro pruebas del protocolo: se reporta porque la guía lo pide para la selección de combinaciones, y no cuenta para el veredicto.
