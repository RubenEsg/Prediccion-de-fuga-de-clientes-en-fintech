| Filas | Inferencia BDL (original) (s) | Inferencia XGBoost (s) | BDL (original): microsegundos por fila | XGBoost: microsegundos por fila |
| ---: | ---: | ---: | ---: | ---: |
| 1.000 | 0,014 | 0,012 | 14,1 | 12,0 |
| 10.000 | 0,064 | 0,062 | 6,4 | 6,2 |
| 100.000 | 0,623 | 0,585 | 6,2 | 5,9 |

Mediana de cinco medidas; la tubería incluye el preprocesamiento. Al multiplicar las filas por 100, el tiempo de inferencia del original se multiplica por 44,2: crece de forma lineal o menos, como predice la sección 5.7.
