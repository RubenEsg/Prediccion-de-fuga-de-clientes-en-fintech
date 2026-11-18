| Cobertura nominal | Método | Cobertura en prueba | Largo | Tramos |
| :--- | :--- | ---: | ---: | ---: |
| 80 % | XGBoost: intervalo conforme | 80,5 % | 0,1161 | — |
| 80 % | XGBoost: conjunto por densidad de sus residuos (exploratorio) | 82,2 % | 0,0301 | 4 |
| 80 % | BDL: intervalo conforme con la misma receta | 84,4 % | 0,1101 | — |
| 80 % | BDL: intervalo central | 80,5 % | 0,1500 | — |
| 80 % | BDL: conjunto de predicción | 79,8 % | 0,0040 | 3 |
| 90 % | XGBoost: intervalo conforme | 90,0 % | 0,1905 | — |
| 90 % | XGBoost: conjunto por densidad de sus residuos (exploratorio) | 91,6 % | 0,0436 | 5 |
| 90 % | BDL: intervalo conforme con la misma receta | 84,7 % | 0,1903 | — |
| 90 % | BDL: intervalo central | 89,9 % | 0,1502 | — |
| 90 % | BDL: conjunto de predicción | 89,9 % | 0,0062 | 4 |
| 95 % | XGBoost: intervalo conforme | 95,1 % | 0,2375 | — |
| 95 % | XGBoost: conjunto por densidad de sus residuos (exploratorio) | 95,9 % | 0,0550 | 5 |
| 95 % | BDL: intervalo conforme con la misma receta | 95,0 % | 0,2311 | — |
| 95 % | BDL: intervalo central | 95,4 % | 0,2001 | — |
| 95 % | BDL: conjunto de predicción | 95,0 % | 0,0099 | 5 |

Al 90 %, el conjunto de predicción del modelo original mide 0,0062 y el intervalo del mejor benchmark, 0,1905: **31 veces más corto** con la misma cobertura nominal. Con 9.745 clientes de prueba, el error de muestreo de una cobertura del 90 % es de ±0,6 puntos.

**La receta «pronóstico ± q» no sirve para el modelo original.** Su cobertura medida fue 84,4 % para el 80 % nominal y 84,7 % para el 90 % nominal. La causa no está en el modelo sino en la receta: cuando el error tiene escalones, el cuantil *q* cae justo sobre uno de ellos, y que ese nivel entero quede dentro o fuera del intervalo depende de diferencias de diezmilésimas entre el modelo final y los modelos de los pliegues con los que se calculó *q*. En los benchmarks no se nota porque su propio error difumina los escalones. Los conjuntos de predicción del modelo no tienen ese problema: se construyen con la densidad del error, no con una distancia al pronóstico.

**Con la misma receta para los dos.** El conjunto del original es un conjunto por densidad y el intervalo del benchmark es simétrico, así que parte de la diferencia es de receta. Una mezcla ajustada a los residuos fuera de pliegue de XGBoost también encuentra 5 niveles, y el conjunto por densidad que resulta mide 0,0436 al 90 % (cobertura en prueba: 91,6 %). Frente a esa referencia, el conjunto del original es 7 veces más corto: esa es la parte atribuible a que estima mejor el índice. Es una comparación exploratoria, añadida después de ver los resultados (`experiments/04e_referencias_homogeneas.py`).
