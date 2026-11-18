| Métrica en prueba | Mejor benchmark | Benchmark | Original | Límite teórico | Parte de la distancia al límite que cierra el original |
| :--- | :--- | ---: | ---: | ---: | ---: |
| RMSE | XGBoost | 0,05917 | 0,05912 | 0,05911 | 86 % |
| MAE (el original predice la mediana) | XGBoost | 0,04706 | 0,04510 | 0,04508 | 99 % |
| AUC | XGBoost | 0,8283 | 0,8311 | 0,8313 | 92 % |
| Pérdida logarítmica (benchmark recalibrado) | XGBoost | 0,3675 | 0,3688 | 0,3652 | −55 % |

Límite teórico: el valor que obtiene en la muestra de prueba el predictor óptimo, el que conoce el índice real de cada cliente (δ = 0,050 y los pesos de los cinco niveles estimados en entrenamiento: 0,402, 0,301, 0,149, 0,099, 0,050). En esperanza ningún modelo lo mejora; en una muestra concreta un modelo puede quedar del otro lado por azar (le ocurre a la variante «BDL sin mezcla (K = 1)», con un RMSE de 0,059099 frente a 0,059105). La última columna describe esta muestra (un porcentaje negativo indica que el original queda más lejos del límite que el benchmark): las diferencias en prueba entre el original y el benchmark no son significativas tras el ajuste de Holm (tablas 6.7 y 6.8), así que no mide un aporte demostrado.
