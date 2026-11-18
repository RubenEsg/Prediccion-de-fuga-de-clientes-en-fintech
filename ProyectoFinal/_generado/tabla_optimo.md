| Tarea | Prueba | Mejor benchmark | p crudo del óptimo | p ajustado del óptimo | ¿Rechaza el óptimo? | p ajustado del original | ¿Rechaza el original? |
| :--- | :--- | :--- | ---: | ---: | :--- | ---: | :--- |
| Regresión | P1. Wilcoxon (pliegues externos) | XGBoost | < 0,001 | < 0,001 | Sí | < 0,001 | Sí |
| Regresión | P2. *t* corregida (pliegues externos) | XGBoost | < 0,001 | 0,002 | Sí | 0,001 | Sí |
| Regresión | P3. Permutación pareada (prueba) | XGBoost | 0,023 | 0,046 | Sí | 0,092 | No |
| Regresión | P4. Bootstrap pareado BCa (prueba) | XGBoost | 0,015 | 0,029 | Sí | 0,076 | No |
| Clasificación | P1. Wilcoxon (pliegues externos) | XGBoost | 0,073 | 0,073 | No | 0,128 | No |
| Clasificación | P2. *t* corregida (pliegues externos) | XGBoost | 0,407 | 0,407 | No | 1,000 | No |
| Clasificación | P3. DeLong (prueba) | XGBoost | 0,217 | 0,652 | No | 0,765 | No |
| Clasificación | P4. McNemar, campaña del 25 % (prueba) | XGBoost | 0,737 | 1,000 | No | 1,000 | No |

El predictor óptimo obtiene en prueba un RMSE de 0,05911 y un AUC de 0,8313 (son los valores de referencia de la sección 6.4), y en los 15 pliegues externos, 0,05887 y 0,8348 de media (el original: 0,05887 y 0,8313). Los p ajustados son de Holm sobre las 9 comparaciones de cada prueba, como en el protocolo; un rechazo solo cuenta si la diferencia favorece al modelo. La prueba global de Friedman, que el protocolo exige antes de P1, no se repite para el óptimo.

En clasificación, el predictor óptimo (el que conoce el índice real de cada cliente, que ningún modelo puede aprender sin la variable excluida) no rechaza la igualdad con XGBoost en ninguna de las 4 pruebas. En regresión, el predictor óptimo sí reúne 4 rechazos frente a XGBoost (P1, P2, P3 y P4): con una estimación casi perfecta del índice la condición era alcanzable, y el original se queda en 2.
