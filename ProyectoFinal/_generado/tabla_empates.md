**Captura de la campaña del 25 %.**

| Modelo | Valores distintos de la probabilidad | Empatados en el corte | De ellos entran | Captura (protocolo) | Captura esperada (al azar) | Captura posible |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BDL (original)** | 219 | 4.354 | 1.510 | 51,4 % | 51,3 % ± 0,6 | 26,0 % a 89,0 % |
| XGBoost | 6.721 | 1 | 1 | 51,0 % | 51,0 % ± 0,0 | 51,0 % a 51,0 % |
| Árbol de decisión | 20 | 1.203 | 332 | 49,5 % | 49,8 % ± 0,3 | 44,3 % a 58,1 % |
| Predictor óptimo | 4 | 4.448 | 1.534 | 51,4 % | 51,3 % ± 0,6 | 25,7 % a 89,6 % |

**Precisión media.**

| Modelo | Convención de scikit-learn | Empates rotos al azar |
| :--- | ---: | ---: |
| **BDL (original)** | 0,5103 | 0,5507 ± 0,0046 |
| XGBoost | 0,5480 | 0,5479 ± 0,0000 |
| Árbol de decisión | 0,5304 | 0,5403 ± 0,0022 |
| Predictor óptimo | 0,5092 | 0,5515 ± 0,0049 |

La campaña contacta a los 2.436 clientes de mayor probabilidad de los 9.745 de prueba. Empatados en el corte: los que tienen la misma probabilidad que el último contactado; de ellos solo entra una parte, y cuál, lo decide el desempate. Protocolo: desempate por posición en la tabla, el del guion de las pruebas. Al azar: valor esperado ± desviación típica, en puntos porcentuales (distribución hipergeométrica); en la precisión media, media ± desviación de 200 desempates.

**Lectura.** La probabilidad del original toma 219 valores distintos y la del predictor óptimo, 4: es una función escalonada, y dentro de una meseta todos los clientes tienen la misma probabilidad (Proposición 5). El corte de la campaña cae dentro de una meseta: 4.354 clientes comparten la probabilidad del último contactado y solo entran 1.510. La captura del original con el desempate del protocolo, 51,4 %, es uno de los valores posibles: con un desempate al azar se espera 51,3 % ± 0,6 puntos, lo mismo que con el predictor óptimo (51,3 %). La diferencia con XGBoost (51,0 %) es menor que esa variación.

**Qué significa para P4.** Repetida con 2.000 desempates al azar de los dos modelos, la prueba de McNemar del original frente a XGBoost da un p crudo con mediana 0,638 (entre 0,099 y 0,982 en el 95 % de los desempates). Tras el ajuste de Holm sobre las 9 comparaciones, rechaza la igualdad a favor del original en el 0,15 % de los desempates y en contra en el 0,00 %. El resultado registrado (no rechaza; p ajustado = 1,000) no depende del desempate: es el de casi todos los sorteos. Pero deja ver qué mide la prueba en este conjunto: dentro de la meseta del corte, el original y el predictor óptimo eligen al azar.

**Precisión media.** scikit-learn trata a los clientes empatados como un solo punto de la curva de precisión y exhaustividad, con la precisión del grupo entero. Con mesetas grandes eso da menos que cualquier orden concreto dentro de ellas: 0,5103 para el original y 0,5092 para el predictor óptimo, por debajo de XGBoost (0,5480). Con los empates rotos al azar son 0,5507 y 0,5515: la desventaja del original en esa columna de la Tabla 7.11 viene de la convención, no del orden que da el modelo.

El dashboard rompe los empates al azar con su propia semilla, como el Entregable 3, y muestra una captura de 51,9 % para el original: es otro sorteo de la misma distribución, a 0,9 desviaciones del valor esperado.
