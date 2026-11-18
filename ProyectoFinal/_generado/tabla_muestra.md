**Métrica: RMSE en regresión y AUC en clasificación.**

| Tarea | Filas | BDL (original) | XGBoost | LightGBM | BDL sin mezcla (K = 1) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Regresión | 3.118 | 0,05896 ± 0,00039 | 0,05972 ± 0,00032 | 0,05981 ± 0,00035 | 0,05971 ± 0,00035 |
| Regresión | 7.796 | 0,05891 ± 0,00040 | 0,05917 ± 0,00037 | 0,05920 ± 0,00035 | 0,05924 ± 0,00035 |
| Regresión | 15.591 | 0,05889 ± 0,00041 | 0,05901 ± 0,00037 | 0,05905 ± 0,00035 | 0,05908 ± 0,00035 |
| Regresión | 31.182 | 0,05888 ± 0,00041 | 0,05896 ± 0,00040 | 0,05899 ± 0,00040 | 0,05902 ± 0,00040 |
| Clasificación | 3.118 | 0,8318 ± 0,0025 | 0,8226 ± 0,0055 | 0,8241 ± 0,0039 | 0,8232 ± 0,0034 |
| Clasificación | 7.796 | 0,8338 ± 0,0025 | 0,8279 ± 0,0026 | 0,8282 ± 0,0019 | 0,8269 ± 0,0047 |
| Clasificación | 15.591 | 0,8336 ± 0,0030 | 0,8309 ± 0,0037 | 0,8306 ± 0,0032 | 0,8294 ± 0,0032 |
| Clasificación | 31.182 | 0,8339 ± 0,0030 | 0,8356 ± 0,0039 | 0,8325 ± 0,0018 | 0,8303 ± 0,0035 |

**Tiempo de ajuste, en segundos.**

| Tarea | Filas | BDL (original) | XGBoost | LightGBM |
| :--- | ---: | ---: | ---: | ---: |
| Regresión | 3.118 | 0,9 | 0,4 | 0,1 |
| Regresión | 7.796 | 1,9 | 0,4 | 0,2 |
| Regresión | 15.591 | 2,3 | 1,0 | 0,4 |
| Regresión | 31.182 | 4,9 | 1,1 | 0,5 |
| Clasificación | 3.118 | 1,5 | 0,4 | 0,5 |
| Clasificación | 7.796 | 2,6 | 0,4 | 0,3 |
| Clasificación | 15.591 | 4,6 | 0,5 | 0,5 |
| Clasificación | 31.182 | 8,5 | 0,8 | 1,2 |

Media ± desviación en los cinco pliegues externos de la semilla 42, con los hiperparámetros finales; las filas se submuestrean dentro de cada pliegue de entrenamiento y la evaluación usa el pliegue de prueba completo. Ningún modelo se reajusta para cada tamaño: parte de la diferencia con pocas filas puede deberse a que los hiperparámetros se eligieron con todas.

**Regresión.** Con 3.118 filas (el 10 %), el original obtiene 0,05896 y XGBoost, 0,05972; con las 31.182, 0,05888 y 0,05896. El original alcanza con 3.118 filas el valor que XGBoost obtiene con todas: la ventaja del modelo es de eficiencia muestral y se reduce al crecer la muestra, cuando todos se acercan al límite.

**Clasificación.** Con 3.118 filas (el 10 %), el original obtiene 0,8318 y XGBoost, 0,8226; con las 31.182, 0,8339 y 0,8356. Con todas las filas, XGBoost queda por delante: la ventaja del original aparece con muestras pequeñas y desaparece al crecer la muestra.

**Qué parte se debe a la mezcla.** La misma curva para el modelo original con $K = 1$ (línea discontinua): en regresión, con 3.118 filas el modelo completo obtiene 0,05896; sin la mezcla, 0,05971, y XGBoost, 0,05972 (con las 31.182, la variante sin mezcla llega a 0,05902, que el modelo completo ya supera con 3.118); en clasificación, con 3.118 filas el modelo completo obtiene 0,8318; sin la mezcla, 0,8232, y LightGBM, 0,8241 (con las 31.182, la variante sin mezcla llega a 0,8303, que el modelo completo ya supera con 3.118). Sin la clase latente, el modelo se comporta como los benchmarks: la ventaja con pocas filas la da la mezcla, que es lo que predice la Proposición 4. La variante usa aquí los hiperparámetros del modelo completo; en la ablación de la sección 5.9, donde tiene su propia búsqueda, la diferencia con todas las filas es de 0,000107 de RMSE y 0,00211 de AUC. Este análisis se añadió después de ver los resultados y es exploratorio (sección 6.2).

**Costo.** Al multiplicar las filas por 10, el tiempo de ajuste del original se multiplica por 5,5 (el de XGBoost, por 3,1): crece como mucho de forma lineal, como predice la sección 5.7. Con todas las filas el original tarda 4,5 veces lo que XGBoost.
