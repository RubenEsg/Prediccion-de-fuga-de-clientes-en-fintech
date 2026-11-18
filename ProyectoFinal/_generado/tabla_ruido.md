**Regresión (RMSE, menor es mejor).** El original es el mejor de los tres en 8 de 11 escenarios.

| Contaminación | Nivel | BDL (original) | XGBoost | LightGBM | Mejor |
| :--- | ---: | ---: | ---: | ---: | :--- |
| sin contaminar | 0,000 | 0,05888 | 0,05896 | 0,05899 | BDL (original) |
| ruido gaussiano en el objetivo | 0,005 | 0,05889 | 0,05896 | 0,05898 | BDL (original) |
| ruido gaussiano en el objetivo | 0,010 | 0,05890 | 0,05896 | 0,05899 | BDL (original) |
| ruido gaussiano en el objetivo | 0,020 | 0,05898 | 0,05897 | 0,05899 | XGBoost |
| ruido gaussiano en el objetivo | 0,050 | 0,05910 | 0,05901 | 0,05904 | XGBoost |
| atípicos en el objetivo | 0,010 | 0,05888 | 0,05897 | 0,05899 | BDL (original) |
| atípicos en el objetivo | 0,050 | 0,05893 | 0,05906 | 0,05908 | BDL (original) |
| atípicos en el objetivo | 0,100 | 0,05899 | 0,05923 | 0,05925 | BDL (original) |
| ruido en los predictores numéricos | 0,100 | 0,05897 | 0,05907 | 0,05907 | BDL (original) |
| ruido en los predictores numéricos | 0,250 | 0,05933 | 0,05940 | 0,05942 | BDL (original) |
| ruido en los predictores numéricos | 0,500 | 0,06157 | 0,06145 | 0,06147 | XGBoost |

**Clasificación (AUC, mayor es mejor).** El original es el mejor de los tres en 3 de 11 escenarios.

| Contaminación | Nivel | BDL (original) | XGBoost | LightGBM | Mejor |
| :--- | ---: | ---: | ---: | ---: | :--- |
| sin contaminar | 0,000 | 0,8339 | 0,8356 | 0,8325 | XGBoost |
| ruido gaussiano en el objetivo | 0,005 | 0,8302 | 0,8308 | 0,8305 | XGBoost |
| ruido gaussiano en el objetivo | 0,010 | 0,8301 | 0,8284 | 0,8297 | BDL (original) |
| ruido gaussiano en el objetivo | 0,020 | 0,8304 | 0,8260 | 0,8287 | BDL (original) |
| ruido gaussiano en el objetivo | 0,050 | 0,8297 | 0,8225 | 0,8262 | BDL (original) |
| atípicos en el objetivo | 0,010 | 0,8323 | 0,8354 | 0,8330 | XGBoost |
| atípicos en el objetivo | 0,050 | 0,8305 | 0,8348 | 0,8323 | XGBoost |
| atípicos en el objetivo | 0,100 | 0,8311 | 0,8331 | 0,8317 | XGBoost |
| ruido en los predictores numéricos | 0,100 | 0,8257 | 0,8310 | 0,8286 | XGBoost |
| ruido en los predictores numéricos | 0,250 | 0,8197 | 0,8180 | 0,8208 | LightGBM |
| ruido en los predictores numéricos | 0,500 | 0,7768 | 0,7802 | 0,7821 | LightGBM |

Nivel: desviación del ruido añadido al puntaje de entrenamiento; fracción de puntajes de entrenamiento reemplazados por un valor al azar; o desviación del ruido en los predictores numéricos, en desviaciones típicas de cada uno. La evaluación siempre es contra el puntaje y la etiqueta sin contaminar.

**Lectura.** En regresión, el original, con ruido gaussiano en el objetivo, es el mejor con 0,005 y 0,010 y no con 0,020 y 0,050 (en el nivel más alto, 0,05910 frente a 0,05901 de XGBoost); con atípicos en el objetivo, es el mejor de los tres en los 3 niveles (en el nivel más alto, 0,05899 frente a 0,05923 de XGBoost); con ruido en los predictores numéricos, es el mejor con 0,100 y 0,250 y no con 0,500 (en el nivel más alto, 0,06157 frente a 0,06145 de XGBoost). En clasificación, el original, con ruido gaussiano en el objetivo, es el mejor con 0,010, 0,020 y 0,050 y no con 0,005 (en el nivel más alto, 0,8297 frente a 0,8262 de LightGBM); con atípicos en el objetivo, no es el mejor en ninguno de los 3 niveles (en el nivel más alto, 0,8311 frente a 0,8331 de XGBoost); con ruido en los predictores numéricos, no es el mejor en ninguno de los 3 niveles (en el nivel más alto, 0,7768 frente a 0,7821 de LightGBM). En 19 de los 22 escenarios de las dos tareas la diferencia entre el original y el mejor de los otros dos es menor que la desviación del original entre pliegues; la supera a favor del original en 0 y en contra en 3 (atípicos en el objetivo de 0,050 en clasificación, ruido en los predictores numéricos de 0,100 en clasificación y ruido en los predictores numéricos de 0,500 en clasificación). El estudio muestra que una contaminación moderada no rompe al modelo, no que sea más robusto que los benchmarks. En regresión la ventaja se pierde cuando el ruido en el puntaje llega a 0,020: los niveles del escalón están separados 0,05, y la Proposición 4 solo promete ganancia mientras esa separación sea grande frente al ruido.
