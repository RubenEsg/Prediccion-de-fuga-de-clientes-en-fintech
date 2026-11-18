**Regresión (RMSE, menor es mejor).**

| Conjunto | Mejor benchmark | Benchmark (externo) | Original (externo) | Puesto del original | Ventaja relativa | p (*t* corregida) | Resultado | Prueba: benchmark / original | $K$ del modelo final |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Seguros médicos † | EBM | 11.332 ± 279 | 11.435 ± 334 | 7 de 10 | −0,90 % | 0,414 | Peor, no significativo | 11.459 / 11.630 | 1 (sin mezcla) |
| Eficiencia energética † | EBM | 2,263 ± 0,116 | 2,313 ± 0,068 | 2 de 10 | −2,18 % | 0,474 | Peor, no significativo | 2,317 / 2,301 | 1 (sin mezcla) |
| Abulón | Random Forest | 2,129 ± 0,093 | 2,168 ± 0,094 | 5 de 10 | −1,83 % | 0,286 | Peor, no significativo | 2,187 / 2,217 | 1 (sin mezcla) |
| Vivienda de California | LightGBM | 0,44258 ± 0,01121 | 0,46449 ± 0,00910 | 3 de 10 | −4,95 % | 0,002 | Peor, significativo | 0,43536 / 0,45210 | 1 (sin mezcla) |

**Clasificación (AUC, mayor es mejor).**

| Conjunto | Mejor benchmark | Benchmark (externo) | Original (externo) | Puesto del original | Ventaja relativa | p (*t* corregida) | Resultado | Prueba: benchmark / original | $K$ del modelo final |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Seguros médicos † | EBM | 0,5722 ± 0,0323 | 0,5152 ± 0,0254 | 10 de 10 | −9,96 % | 0,037 | Peor, significativo | 0,5998 / 0,5659 | 1 (sin mezcla) |
| Eficiencia energética † | XGBoost | 0,9771 ± 0,0062 | 0,9733 ± 0,0099 | 4 de 10 | −0,38 % | 0,607 | Peor, no significativo | 0,9601 / 0,9629 | sin puntaje |
| Abulón | LightGBM | 0,8591 ± 0,0071 | 0,8485 ± 0,0116 | 7 de 10 | −1,24 % | 0,093 | Peor, no significativo | 0,8622 / 0,8606 | sin puntaje |
| Vivienda de California | LightGBM | 0,9680 ± 0,0038 | 0,9639 ± 0,0035 | 3 de 10 | −0,42 % | 0,017 | Peor, significativo | 0,9712 / 0,9664 | sin puntaje |

† Conjunto con una variable omitida. Externo: media ± desviación de los cinco pliegues externos. Ventaja relativa: positiva si el original es mejor. La prueba *t* corregida compara al original con el mejor benchmark del conjunto. $K$: clases latentes del modelo final. «Sin mezcla»: la búsqueda de hiperparámetros eligió `n_componentes = 1` y el modelo es un boosting de mínimos cuadrados con árboles propios. «BIC»: la búsqueda eligió la mezcla y el criterio BIC fijó el número de clases. «Sin puntaje»: en clasificación, la búsqueda eligió aprender de la etiqueta.
