# Ficha del modelo: BDL, el modelo original de fuga de clientes

Ficha del modelo (*model card*, Mitchell et al., 2019) del modelo original, BDL (Boosting con Desplazamiento Latente), del proyecto «Predicción de fuga de clientes en una fintech colombiana: dashboard predictivo, modelo original (Boosting con Desplazamiento Latente) y evaluación rigurosa». Versión 2.0.0, 18 de noviembre de 2026. La genera `generar_fichas.py` desde `datos/` y `artifacts/`: ninguna cifra se escribe a mano. La misma ficha está en la vista «Ficha del modelo» del dashboard (/modelos/ficha).

## Detalles del modelo

Qué es, quién lo hizo, con qué versión y con qué licencia; los artefactos y su huella.

### El modelo

- **Nombre:** BDL de fuga de clientes: regresión del puntaje de propensión y clasificación del riesgo alto
- **Tipo:** Boosting con Desplazamiento Latente, el modelo original del proyecto: un boosting de árboles propios con una mezcla latente de desplazamientos, ajustado con un algoritmo EM generalizado (ClasificadorBDL y RegresorBDL, en src/original_model.py)
- **Dependencias:** numpy 2.5.3, scipy 1.18.1 y scikit-learn 1.9.1 (la tubería y el preprocesamiento); el modelo no importa ninguna librería de modelos
- **Versión:** 2.0.0 (artefactos del 18 de noviembre de 2026)
- **Fecha de entrenamiento:** clasificación, 17 de noviembre de 2026; regresión, 17 de noviembre de 2026
- **Autores:** Valerie Martínez Pulido, Abrahan Basto Martínez y Rubén Esguerra Fernández
- **Licencia:** MIT para el código y los modelos; los datos son CC BY 4.0 y los derivados conservan su atribución
- **Repositorio:** https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech
- **Cita:** Valerie Martínez Pulido, Abrahan Basto Martínez y Rubén Esguerra Fernández (2026). Predicción de fuga de clientes en una fintech colombiana: dashboard predictivo, modelo original (Boosting con Desplazamiento Latente) y evaluación rigurosa. https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech

Fuentes: CITATION.cff · artifacts/indice.json · datos/original/experimentos.csv · artifacts/clasificacion/bdl/metadata.json · artifacts/regresion/bdl/metadata.json

### Entradas, salidas e hiperparámetros

- **Entradas:** 48 variables (32 numéricas, 7 booleanas y 9 categóricas), que el preprocesamiento convierte en 78 columnas. En el entrenamiento de la clasificación recibe, además, el puntaje continuo de las filas de entrenamiento; nunca el de validación ni el de prueba
- **Salidas:** regresión: el puntaje churn_probability previsto (la media de la mezcla), con su conjunto de predicción (una unión de tramos, uno por nivel del puntaje) y su intervalo conforme; clasificación: la probabilidad de que el puntaje supere el umbral de riesgo alto (probabilidad de excedencia de la mezcla)
- **Hiperparámetros de clasificación:** tasa de aprendizaje: 0,09834; profundidad de los árboles: 1; mínimo de filas por hoja: 11; regularización L2: 0,02269; tramos por variable: 73; clases latentes: automático (BIC); usa el puntaje continuo: sí; búsqueda: TPE, sin balanceo
- **Hiperparámetros de regresión:** tasa de aprendizaje: 0,5816; profundidad de los árboles: 1; mínimo de filas por hoja: 88; regularización L2: 0,01508; tramos por variable: 885; clases latentes: automático (BIC); búsqueda: TPE, sin balanceo
- **Lo que aprendió en clasificación:** 5 clases latentes (elegidas por BIC), 2.100 árboles (100 de calentamiento) y 400 vueltas del algoritmo EM
- **Lo que aprendió en regresión:** 5 clases latentes (elegidas por BIC), 547 árboles (22 de calentamiento) y 105 vueltas del algoritmo EM
- **Semilla:** 42

Fuentes: artifacts/clasificacion/bdl/metadata.json · artifacts/regresion/bdl/metadata.json · datos/original/modelo_clasificacion.json · datos/original/modelo_regresion.json · datos/eda/etl.json

### Artefactos y su huella

| Archivo | Tamaño | SHA-256 (inicio) | Qué es |
| --- | ---: | --- | --- |
| artifacts/clasificacion/bdl/modelo.joblib | 0,09 MB | 0c4dfcea9996 | la tubería de scikit-learn con el modelo de clasificación (preprocesamiento y modelo) |
| artifacts/regresion/bdl/modelo.joblib | 0,14 MB | 56a1214c2268 | la tubería de scikit-learn con el modelo de regresión (preprocesamiento y modelo) |
| src/original_model.py | — | 6e03a92afa23 | el código del modelo, copia exacta del de la corrección: las tuberías guardadas nombran este módulo |
| Corrección: clasificacion__bdl__ninguno__bayesiana__42.joblib | — | 707d95ea4ce6 | el modelo final guardado por la corrección |
| Corrección: regresion__bdl__ninguno__bayesiana__42.joblib | — | 242223b9593a | el modelo final guardado por la corrección |

*La ida y vuelta de los artefactos dio predicciones idénticas a las guardadas en 9.745 clientes de prueba. Una prueba comprueba que el código del modelo es idéntico al de la corrección. La huella completa está en artifacts/indice.json.*

Fuentes: artifacts/indice.json · datos/original/estado.json

## Para qué sirve y para qué no

El uso previsto, quién debería usarlo y los usos que no se recomiendan.

### Uso previsto

- Ordenar a los clientes de COFINFAD por su riesgo para priorizar una campaña de retención con capacidad limitada: contactando al 25 % de mayor puntaje (2.436 de 9.745) alcanza al 51,9 % de los clientes de riesgo alto, con una precisión del 51,1 %.
- Decir, además del pronóstico, en cuál de los niveles del puntaje puede estar un cliente: su conjunto de predicción al 90 % es una unión de 4 tramos que cubre el 89,9 % de la prueba.
- Explicar qué variables mueven el índice del puntaje: sus modelos finales usan árboles de profundidad uno, así que el índice es la suma exacta de lo que aporta cada variable.
- Docencia e investigación: un modelo original con su formulación, sus demostraciones y su código, comparado con los benchmarks con un protocolo registrado antes de los experimentos.

*El puntaje de BDL toma 219 valores distintos en la prueba: 4.354 clientes comparten el del corte de la campaña y solo entran 1.510. Cuáles entran lo decide el desempate, así que la captura que se muestra es una de las posibles: con un desempate al azar se espera 51,3 % ± 0,6 puntos, y puede ir de 26,0 % a 89,0 %. La precisión depende del mismo desempate: se espera 50,5 % ± 0,6 puntos. El libro del proyecto usa el desempate del protocolo de la corrección (por posición) y da 51,4 %.*

Fuentes: datos/original/lideres_clasificacion.csv · datos/conformal.json · datos/metadatos.json · datos/original/empates_campana.json

### Usuarios previstos

- El profesor y los estudiantes del curso de Machine Learning.
- Analistas que quieren estudiar un modelo con una variable latente y compararlo con los ensambles de árboles.
- Equipos de retención que quieren ver el método, no aplicarlo sin antes validarlo con sus propios datos.

Fuentes: Libro del proyecto final (ProyectoFinal): capítulos 5, 6, 7 y 9

### Usos no recomendados

- Decidir sobre clientes reales: los datos son sintéticos y el escalón que el modelo aprovecha existe porque así se generó el conjunto.
- Decisiones de crédito, precio, cobranza o exclusión de un servicio.
- Leer su probabilidad como probabilidad de fuga observada: es la de que el puntaje del conjunto supere su umbral.
- Usarlo donde la etiqueta no sea el umbral de una variable continua observada en el entrenamiento: sin ella el modelo aprende de la etiqueta y pierde su ventaja.
- Puntuar valores fuera del rango de entrenamiento: la edad, por ejemplo, solo toma 6 valores (29, 30, 44, 45, 59, 60).
- Tratar «riesgo alto» como fuga observada, o usar productos activos como entrada: es la variable con que se generó el puntaje y el modelo recuperaría la regla en vez de predecir.
- Llevarlo a otra entidad, otro país u otro periodo sin reentrenarlo y validarlo.

Fuentes: datos/variables.csv · Libro del proyecto final (ProyectoFinal): capítulos 5, 6, 7 y 9

## Factores, métricas y datos

Con qué se evaluó, en qué datos y en qué punto de operación. Al lado va el mejor benchmark de cada tarea en la validación anidada.

### Factores relevantes

- Cohorte de edad: el conjunto tiene 3 cohortes y el riesgo alto va del 8,8 % (44–45 años) al 45,7 % (59–60). La edad reúne el 46,1 % de la importancia del modelo en clasificación.
- Productos activos: la variable excluida con la que se generó el puntaje. El modelo no la ve; la trata como una clase latente y recupera sus niveles.
- El género, la ubicación y el ingreso también son entradas, pero el modelo casi no los usa: ninguno llega al 0,1 % de la importancia.

Fuentes: datos/resultados/errores_segmentos.csv · datos/shap/indice.json

### Datos de entrenamiento y de evaluación

- **Conjunto:** COFINFAD (Mendeley Data, CC BY 4.0), sintético: ver DATA_STATEMENT.md
- **Partición:** aleatoria por índices, 80/20, con semilla 42, sin estratificar, antes de derivar variables y de calcular cualquier estadístico del objetivo
- **Entrenamiento:** 38.978 clientes; riesgo alto: 24,6 %. El modelo reparte esas filas en ajuste, validación y calibración
- **Evaluación:** 9.745 clientes de prueba; riesgo alto: 24,6 % (2.399 clientes). La prueba ya se había usado para evaluar los benchmarks del Entregable 3; no intervino en el diseño del modelo
- **Etiqueta:** riesgo alto = churn_probability por encima de su percentil 75 en el entrenamiento (0,3927)
- **Selección:** validación cruzada anidada de 5 × 3 pliegues, con 30 configuraciones por búsqueda y una sola combinación, fijada en el protocolo
- **Punto de operación:** contactar al 25 % de mayor puntaje: 2.436 clientes

Fuentes: datos/metadatos.json · datos/original/costo.csv · ModeloOriginal/experiments/protocolo.md (protocolo pre-registrado)

### Métricas en la prueba y en la validación anidada

| Métrica | BDL | XGBoost | Qué mide |
| --- | --- | --- | --- |
| AUC de prueba [IC 95 %] | 0,8311 [0,8247; 0,8377] | 0,8283 [0,8203; 0,8359] | qué tan bien ordena a los clientes por riesgo |
| AUC externo (media ± desviación de 15 pliegues) | 0,8313 ± 0,0038 | 0,8333 ± 0,0041 | la validación anidada: con ella se decide el veredicto |
| AUC-PR | 0,510 | 0,548 | orden de los positivos, sensible a la prevalencia |
| Captura con el 25 % de capacidad | 51,9 % | 51,0 % | parte de los clientes de riesgo alto que alcanza la campaña |
| Precisión con el 25 % de capacidad | 51,1 % | 50,2 % | parte de los contactados que son de riesgo alto |
| Error de calibración (ECE): sin recalibrar → isotónica | 0,004 → 0,004 | 0,220 → 0,007 | distancia entre la probabilidad dicha y la observada |
| RMSE de prueba [IC 95 %] | 0,05912 [0,05834; 0,05996] | 0,05917 [0,05841; 0,06002] | error del puntaje previsto |
| RMSE externo (media ± desviación de 15 pliegues) | 0,05887 ± 0,00047 | 0,05897 ± 0,00047 | la validación anidada |
| MAE y R² de prueba | 0,0470 y 0,220 | 0,0471 y 0,218 | error absoluto medio y varianza explicada |
| Incertidumbre al 90 %: cobertura · largo | 89,9 % · 0,0062 (conjunto de 4 tramos) | 90,0 % · 0,1905 (intervalo conforme) | parte de los puntajes reales dentro del conjunto o del intervalo, y cuánto mide |

*Con la regla de superioridad de la guía, el veredicto del modelo original es: en clasificación, no superior (no cumple ninguna de las 4 condiciones evaluadas); en regresión, no superior (cumple 2 de las 4 condiciones evaluadas). El límite teórico del conjunto en la prueba es un AUC de 0,8313 y un RMSE de 0,05911: los mejores benchmarks ya estaban a milésimas de él.*

*Filas que dependen de los empates del puntaje. La captura y la precisión con el 25 % de capacidad son las de uno de los desempates posibles (4.354 clientes comparten el puntaje del corte y entran 1.510): con un desempate al azar se espera una captura del 51,3 % ± 0,6 puntos. El libro del proyecto usa el desempate del protocolo de la corrección (por posición) y da 51,4 %. El AUC-PR de BDL (0,510) cuenta a los clientes con el mismo puntaje como un solo punto de la curva; con los empates rotos al azar es 0,551 ± 0,005.*

*El ECE (error de calibración esperado) se calcula aquí con diez tramos de igual ancho, como en el Entregable 3; la Tabla 7.11 del libro del proyecto usa tramos de igual número de clientes, y por eso sus cifras no coinciden exactamente con estas.*

Fuentes: datos/resultados/clasificacion_prueba.csv · datos/resultados/regresion_prueba.csv · datos/original/lideres_clasificacion.csv · datos/original/lideres_regresion.csv · datos/calibracion_modelos.json · datos/conformal.json · datos/original/veredicto.json · datos/original/condicion4.json · datos/original/techo.json · datos/original/empates_campana.json

## Análisis por segmento

La prueba dividida por edad, por la cola de transacciones y por segmento de cliente: el modelo original frente al mejor benchmark de cada tarea.

### Dónde acierta y dónde falla

| Subgrupo de prueba | Clientes | RMSE de BDL | RMSE de XGBoost | AUC de BDL | AUC de XGBoost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Todos | 9.745 | 0,05912 | 0,05917 | 0,8311 | 0,8283 |
| Edad 29 | 1.643 | 0,05928 | 0,05942 | 0,8252 | 0,8269 |
| Edad 30 | 1.581 | 0,06012 | 0,06010 | 0,8497 | 0,8537 |
| Edad 44 | 1.598 | 0,05843 | 0,05853 | 0,9183 | 0,9153 |
| Edad 45 | 1.576 | 0,05830 | 0,05830 | 0,9384 | 0,9448 |
| Edad 59 | 1.631 | 0,05858 | 0,05869 | 0,6171 | 0,6172 |
| Edad 60 | 1.716 | 0,05989 | 0,05993 | 0,6675 | 0,6595 |
| tx_count > 500 (cola) | 134 | 0,06306 | 0,06344 | 0,8117 | 0,8531 |
| tx_count ≤ 500 | 9.611 | 0,05906 | 0,05911 | 0,8309 | 0,8275 |
| Segmento inactive | 1.954 | 0,05874 | 0,05875 | 0,8275 | 0,8333 |
| Segmento occasional | 3.852 | 0,05870 | 0,05877 | 0,8342 | 0,8268 |
| Segmento power | 1.026 | 0,05885 | 0,05894 | 0,8272 | 0,8195 |
| Segmento regular | 2.913 | 0,06000 | 0,06006 | 0,8309 | 0,8308 |

*La cohorte de 59–60 años es la de más riesgo alto (45,7 %) y la de 44–45, la de menos (8,8 %). Dentro de una sola edad, donde esa variable ya no ordena a los clientes, el AUC de BDL va de 0,6171 (edad 59) a 0,9384 (edad 45), frente a 0,8311 en el total. Las diferencias entre los dos modelos cambian de signo de un subgrupo a otro y no llevan prueba de hipótesis; la mayor en AUC está en «tx_count > 500 (cola)» (134 clientes), donde BDL obtiene 0,8117 y XGBoost, 0,8531.*

Fuentes: datos/original/errores_subgrupos.csv · datos/resultados/errores_segmentos.csv

## Ética, limitaciones y recomendaciones

Lo que hay que saber antes de usarlo.

### Consideraciones éticas

- Los datos son públicos y sintéticos: no hay personas reales identificables.
- La edad, el género y la ubicación son entradas. La edad domina el modelo (46,1 % de la importancia): una campaña real que lo siguiera trataría distinto a las personas según su edad, y eso habría que justificarlo y revisarlo.
- No se mide fuga real: usarlo para negar o encarecer un servicio no tendría respaldo en estos datos.

Fuentes: datos/shap/indice.json · Libro del proyecto final (ProyectoFinal): capítulos 5, 6, 7 y 9

### Limitaciones y riesgos

- Los datos son sintéticos: el escalón que el modelo aprovecha existe porque el generador resta una cantidad fija por producto activo; en datos reales un efecto así sería menos limpio.
- El margen de mejora era mínimo: los mejores benchmarks ya estaban a milésimas del límite teórico. Lo que el modelo añade es la estructura del puntaje y una incertidumbre mejor descrita, no precisión puntual.
- Supone un desplazamiento aditivo, independiente de los predictores y con la misma dispersión en todas las clases. Cuando no se cumple, el criterio BIC deja una sola clase y el modelo es un boosting de mínimos cuadrados.
- Su probabilidad salta de una meseta a otra en un intervalo estrecho del índice: un cliente mal situado en ese borde recibe una probabilidad casi segura y equivocada.
- El intervalo simétrico «pronóstico ± q» no le sirve: al 90 % cubre el 84,7 % de la prueba, porque su error cae en niveles separados. Su conjunto de predicción, con la misma cobertura nominal, cubre el 89,9 %; esa garantía es marginal: vale sobre el total de clientes, no dentro de cada grupo.
- Los conjuntos conformes de clases cubren de más (96,9 % al 90 %): el 45 % de los clientes recibe las dos clases.
- Su probabilidad es escalonada (219 valores distintos en la prueba): cuando el corte de una campaña cae en una meseta, a quién se contacta dentro de ella lo decide el desempate, no el modelo.
- Ajustarlo cuesta más que ajustar los benchmarks de boosting: en regresión, 5,8 s frente a 1,7 s de XGBoost (media de 10 semillas); está escrito en numpy, sin compilar.

Fuentes: datos/conformal.json · datos/calibracion_modelos.json · datos/original/semillas.csv · Libro del proyecto final (ProyectoFinal): capítulos 5, 6, 7 y 9 · datos/original/empates_campana.json

### Recomendaciones

- Ordenar a los clientes con la probabilidad del modelo y expresar la incertidumbre del pronóstico con su conjunto de predicción, no con el intervalo simétrico.
- Validar en el tiempo con fuga observada (entrenar con un periodo y probar con el siguiente) antes de cualquier uso real.
- Revisar los resultados por cohorte de edad y reentrenar si cambia la distribución de los clientes.
- Mantener una revisión humana de cada decisión que afecte a un cliente.

Fuentes: Libro del proyecto final (ProyectoFinal): capítulos 5, 6, 7 y 9
