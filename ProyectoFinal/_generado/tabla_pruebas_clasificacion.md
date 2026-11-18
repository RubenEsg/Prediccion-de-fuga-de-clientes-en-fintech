**Clasificación (AUC, mayor es mejor). Modelo original frente al mejor benchmark, XGBoost.** Ventaja media en los 15 pliegues: −0,00200 (a favor del original en 4). Los p ajustados son de Holm sobre las 9 comparaciones de cada prueba.

| Prueba | Hipótesis nula | Supuestos verificados | Estadístico | p crudo | p ajustado | Tamaño del efecto | Conclusión |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| Friedman con Iman-Davenport (global) | Los diez modelos rinden igual | 15 bloques completos, 10 modelos | F = 65,7 | < 0,001 | — | — | Rechaza |
| P1. Wilcoxon de rangos con signo | La mediana de las diferencias por pliegue es cero | Pares por pliegue; asimetría de las diferencias 0,68 | 21 | 0,026 | 0,128 | δ = −0,31; *d* = −0,51 | No rechaza |
| P2. *t* corregida de Nadeau-Bengio | La diferencia media por pliegue es cero | Normalidad (Shapiro-Wilk, p = 0,361); varianza corregida por 1/J + 1/4 | −1,1199 | 0,282 | 1,000 | δ = −0,31; *d* = −0,51 | No rechaza |
| P3. DeLong (prueba) | Los dos AUC son iguales | Filas independientes: 2.399 positivos y 7.346 negativos | 1,1385 | 0,255 | 0,765 | — | No rechaza |
| P4. McNemar, campaña del 25 % (prueba) | Los dos aciertan con la misma frecuencia | 1.960 pares discordantes | 20 | 0,668 | 1,000 | — | No rechaza |

Estadísticos: en P1, la menor de las dos sumas de rangos; en P2, *t* de la diferencia original menos benchmark; en P3, *z* de DeLong (positivo: el AUC del original es mayor); en P4, los aciertos que solo tiene el original menos los que solo tiene el benchmark. Tamaño del efecto: δ de Cliff y *d* de Cohen entre los valores por pliegue de los dos modelos, con signo positivo a favor del original; la *d* pareada de las diferencias es −0,63. La diferencia crítica de Nemenyi es 3,50 rangos. Los pliegues de una validación cruzada comparten filas de entrenamiento, de modo que el p de Wilcoxon es optimista; la *t* corregida lo compensa.

**El original frente a cada benchmark** (ventaja positiva: a favor del original; en cada prueba, p crudo / p ajustado por Holm). Son 36 comparaciones en esta tarea (9 benchmarks por 4 pruebas) y 72 en las dos, además de la prueba global de cada una; el ajuste de Holm se hace dentro de cada prueba, sobre sus 9 comparaciones, como fija el protocolo.

| Benchmark | Ventaja media | Pliegues a favor | P1 | P2 | P3 | P4 | δ de Cliff | *d* de Cohen |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | −0,00200 | 4/15 | 0,026 / 0,128 | 0,282 / 1,000 | 0,255 / 0,765 | 0,668 / 1,000 | −0,31 | −0,51 |
| Árbol de decisión | −0,00088 | 7/15 | 0,421 / 1,000 | 0,696 / 1,000 | 0,008 / 0,042 | 0,026 / 0,204 | −0,09 | −0,25 |
| Random Forest | −0,00075 | 6/15 | 0,489 / 1,000 | 0,644 / 1,000 | 0,040 / 0,160 | 0,026 / 0,204 | −0,13 | −0,22 |
| LightGBM | −0,00007 | 7/15 | 0,847 / 1,000 | 0,964 / 1,000 | 0,531 / 0,765 | 0,668 / 1,000 | −0,01 | −0,02 |
| EBM | 0,00193 | 9/15 | 0,169 / 0,675 | 0,466 / 1,000 | 0,375 / 0,765 | 0,804 / 1,000 | 0,28 | 0,47 |
| k-NN | 0,04714 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 0,522 / 1,000 | 1,00 | 8,45 |
| Regresión logística | 0,06685 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 0,705 / 1,000 | 1,00 | 13,37 |
| SVM lineal | 0,06702 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 0,705 / 1,000 | 1,00 | 13,45 |
| Naive Bayes | 0,15976 | 15/15 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | < 0,001 / < 0,001 | 1,00 | 18,32 |
