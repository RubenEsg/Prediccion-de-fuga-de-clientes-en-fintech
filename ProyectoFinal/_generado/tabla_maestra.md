| Conjunto | Origen | Semilla | Corridas | Modelos | Evaluaciones | Tiempo (h) |
| :--- | :--- | ---: | ---: | :--- | ---: | ---: |
| COFINFAD | Entregable 3 | 42 | 16 | 11 modelos | 3.288 | 3,1 |
| COFINFAD | Proyecto final | 42 | 4 | BDL (original), LightGBM | 744 | 1,5 |
| COFINFAD | Proyecto final | 43 | 20 | 13 modelos | 4.032 | 4,4 |
| COFINFAD | Proyecto final | 44 | 20 | 13 modelos | 4.032 | 4,2 |
| COFINFAD | ablación | 42 | 4 | BDL con salida conforme empírica, BDL sin mezcla (K = 1), BDL sin puntaje (aprende de la etiqueta) | 720 | 2,6 |
| Seguros médicos | Proyecto final | 42 | 20 | 13 modelos | 4.032 | 0,3 |
| Eficiencia energética | Proyecto final | 42 | 20 | 13 modelos | 4.032 | 0,3 |
| Abulón | Proyecto final | 42 | 20 | 13 modelos | 4.032 | 0,9 |
| Vivienda de California | Proyecto final | 42 | 20 | 13 modelos | 4.032 | 6,0 |

Cada corrida es una validación anidada completa de 5 pliegues externos × 3 internos más el reajuste final, y la dispersión que se reporta es la desviación típica entre los pliegues externos. Las corridas de esta tabla suman 23,3 horas de cómputo. Las 160 corridas del Entregable 3 (las 140 combinaciones de la guía y las de la EBM) están en `Entrega3/resultados/experimentos.parquet` y sumaron 46 horas; las filas con origen «Entregable 3» son la mejor combinación de cada benchmark, tomada de ahí. Las corridas del proyecto final posteriores a la primera repetición de COFINFAD se ejecutaron en dos colas simultáneas y con otras tareas en el mismo equipo: sus tiempos incluyen esa competencia por los núcleos y sirven para comparar órdenes de magnitud, no como medida exacta. El tiempo de ajuste de cada modelo final, medido aparte, está en la Tabla 7.6.
