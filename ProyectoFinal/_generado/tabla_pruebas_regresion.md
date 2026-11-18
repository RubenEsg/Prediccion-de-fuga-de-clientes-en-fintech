**Regresión (RMSE, menor es mejor). Modelo original frente al mejor benchmark, XGBoost.** Ventaja media en los 15 pliegues: 0,000099 (a favor del original en 15). Los p ajustados son de Holm sobre las 9 comparaciones de cada prueba.

| Prueba | Hipótesis nula | Supuestos verificados | Estadístico | p crudo | p ajustado | Tamaño del efecto | Conclusión |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| Friedman con Iman-Davenport (global) | Los diez modelos rinden igual | 15 bloques completos, 10 modelos | F = 375,0 | < 0,001 | — | — | Rechaza |
| P1. Wilcoxon de rangos con signo | La mediana de las diferencias por pliegue es cero | Pares por pliegue; asimetría de las diferencias 0,46 | 0 | < 0,001 | < 0,001 | δ = 0,20; *d* = 0,21 | Rechaza, a favor del original |
| P2. *t* corregida de Nadeau-Bengio | La diferencia media por pliegue es cero | Normalidad (Shapiro-Wilk, p = 0,757); varianza corregida por 1/J + 1/4 | −4,6191 | < 0,001 | 0,001 | δ = 0,20; *d* = 0,21 | Rechaza, a favor del original |
| P3. Permutación pareada (prueba) | El error cuadrático medio es igual | Filas independientes; sin supuesto de distribución | −6,88 × 10⁻⁶ | 0,046 | 0,092 | — | No rechaza |
| P4. Bootstrap pareado BCa (prueba) | La diferencia de RMSE es cero | Filas independientes; intervalo BCa | −5,81 × 10⁻⁵ | 0,038 | 0,076 | — | No rechaza |

Estadísticos: en P1, la menor de las dos sumas de rangos; en P2, *t* de la diferencia original menos benchmark; en P3, la diferencia de error cuadrático medio, original menos benchmark (negativo: el original se equivoca menos); en P4, la diferencia de RMSE, original menos benchmark. Tamaño del efecto: δ de Cliff y *d* de Cohen entre los valores por pliegue de los dos modelos, con signo positivo a favor del original; la *d* pareada de las diferencias es 2,60. La diferencia crítica de Nemenyi es 3,50 rangos. Los pliegues de una validación cruzada comparten filas de entrenamiento, de modo que el p de Wilcoxon es optimista; la *t* corregida lo compensa.

**El original frente a cada benchmark** (ventaja positiva: a favor del original; en cada prueba, p crudo / p ajustado por Holm). Son 36 comparaciones en esta tarea (9 benchmarks por 4 pruebas) y 72 en las dos, además de la prueba global de cada una; el ajuste de Holm se hace dentro de cada prueba, sobre sus 9 comparaciones, como fija el protocolo.

| Benchmark | Ventaja media | Pliegues a favor | P1 | P2 | P3 | P4 | δ de Cliff | *d* de Cohen |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0,000099 | 15/15 | < 0,001 / < 0,001 | < 0,001 / 0,001 | 0,046 / 0,092 | 0,038 / 0,076 | 0,20 | 0,21 |
| LightGBM | 0,000119 | 15/15 | < 0,001 / < 0,001 | 0,002 / 0,003 | 0,054 / 0,092 | 0,051 / 0,076 | 0,21 | 0,25 |
| EBM | 0,000153 | 15/15 | < 0,001 / < 0,001 | 0,002 / 0,003 | 0,005 / 0,015 | 0,002 / 0,006 | 0,28 | 0,33 |
| Random Forest | 0,000260 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / 0,002 | < 0,001 / < 0,001 | 0,43 | 0,54 |
| Árbol de decisión | 0,000407 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 0,52 | 0,86 |
| Lasso | 0,002939 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 1,00 | 5,88 |
| Ridge | 0,002954 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 1,00 | 5,93 |
| SVR lineal | 0,002973 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 1,00 | 6,03 |
| k-NN | 0,004470 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 1,00 | 9,02 |
