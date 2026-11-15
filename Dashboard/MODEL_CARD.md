# Ficha del modelo: EBM de fuga de clientes

Ficha del modelo (*model card*, Mitchell et al., 2019) de la Explainable Boosting Machine del proyecto «Fuga de clientes · Laboratorio del modelo: dashboard predictivo del riesgo de fuga en una fintech». Versión 1.0.0, 15 de noviembre de 2026. La genera `generar_fichas.py` desde `datos/` y `artifacts/`: ninguna cifra se escribe a mano. La misma ficha está en la vista «Ficha del modelo» del dashboard (/modelos/ficha).

## Detalles del modelo

Qué es, quién lo hizo, con qué versión y con qué licencia; los artefactos y su huella.

### El modelo

- **Nombre:** EBM de fuga de clientes: clasificación del riesgo alto y regresión del puntaje de propensión
- **Tipo:** Explainable Boosting Machine (modelo aditivo generalizado aprendido por boosting), ExplainableBoostingClassifier y ExplainableBoostingRegressor de interpret-core 0.7.8
- **Versión:** 1.0.0 (artefactos de 15 de noviembre de 2026)
- **Fecha de entrenamiento:** clasificación, 1 de octubre de 2026; regresión, 1 de octubre de 2026
- **Autores:** Valerie Martínez Pulido, Abrahan Basto Martínez y Rubén Esguerra Fernández
- **Licencia:** MIT para el código y los modelos; los datos son CC BY 4.0 y los derivados conservan su atribución
- **Repositorio:** https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech
- **Cita:** Valerie Martínez Pulido, Abrahan Basto Martínez y Rubén Esguerra Fernández (2026). Fuga de clientes · Laboratorio del modelo: dashboard predictivo del riesgo de fuga en una fintech. https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech

Fuentes: CITATION.cff · artifacts/indice.json · datos/entrega3/corridas.csv · artifacts/clasificacion/ebm/metadata.json · artifacts/regresion/ebm/metadata.json

### Entradas, salidas e hiperparámetros

- **Entradas:** 48 variables (32 numéricas, 7 booleanas y 9 categóricas), que el preprocesamiento convierte en 78 columnas
- **Salidas:** clasificación: puntaje de riesgo alto (sin recalibrar, para ordenar) y probabilidad recalibrada (isotónica); regresión: el puntaje churn_probability previsto, con su intervalo conforme
- **Hiperparámetros de clasificación:** η = 0,008, L = 2 hojas, m = 100 por hoja, B = 8 bolsas, K = 0 pares; balanceo: ponderación; búsqueda: Grid
- **Hiperparámetros de regresión:** η = 0,008, L = 3 hojas, m = 20 por hoja, B = 8 bolsas, K = 0 pares; balanceo: sin balanceo; búsqueda: Grid
- **Semilla:** 42

Fuentes: artifacts/clasificacion/ebm/metadata.json · artifacts/regresion/ebm/metadata.json · datos/metadatos.json · datos/eda/etl.json

### Artefactos y su huella

| Archivo | Tamaño | SHA-256 (inicio) | Qué es |
| --- | ---: | --- | --- |
| artifacts/clasificacion/ebm/modelo.joblib | 0,43 MB | 3e5d7ab579f1 | la tubería de scikit-learn con la EBM de clasificación (preprocesamiento y modelo) |
| artifacts/regresion/ebm/modelo.joblib | 1,06 MB | a6739bda5aad | la tubería de scikit-learn con la EBM de regresión (preprocesamiento y modelo) |
| datos/modelos/ebm_clasificacion.json | 1,08 MB | 21a19c4abcd6 | las tablas del modelo de clasificación que evalúa el motor en numpy de la app |
| datos/modelos/ebm_regresion.json | 1,11 MB | cd778d5f9a12 | las tablas del modelo de regresión que evalúa el motor en numpy de la app |
| Entregable 3: clasificacion__ebm__class_weight__grid__42.joblib | — | 76f9bf4ff275 | el modelo final guardado por el proyecto |
| Entregable 3: regresion__ebm__ninguno__grid__42.joblib | — | 5eef46ea78c8 | el modelo final guardado por el proyecto |

*La ida y vuelta de los artefactos dio predicciones idénticas a las guardadas en 9.745 clientes de prueba, y el motor en numpy difiere del modelo guardado a lo sumo 6,1·10⁻¹⁶ en 48.723 clientes. La huella completa está en artifacts/indice.json y datos/origen.json.*

Fuentes: artifacts/indice.json · datos/origen.json · datos/metadatos.json

## Para qué sirve y para qué no

El uso previsto, quién debería usarlo y los usos que no se recomiendan.

### Uso previsto

- Ordenar a los clientes de COFINFAD por su riesgo para priorizar una campaña de retención con capacidad limitada: contactando al 25 % de mayor puntaje (2.436 de 9.745) alcanza al 51,2 % de los clientes de riesgo alto, con una precisión del 50,4 %.
- Explicar qué variables mueven el puntaje y cómo: cada predicción es la suma exacta de lo que aporta cada variable.
- Docencia y comparación de métodos: un modelo legible frente a los ensambles de árboles, con el mismo protocolo.

Fuentes: datos/resultados/clasificacion_prueba.csv · datos/metadatos.json

### Usuarios previstos

- El profesor y los estudiantes del curso de Machine Learning.
- Analistas que quieren estudiar un modelo aditivo legible y compararlo con modelos de caja negra.
- Equipos de retención que quieren ver el método, no aplicarlo sin antes validarlo con sus propios datos.

Fuentes: Informe final (metodologia.tex, resultados.tex, conclusiones.tex)

### Usos no recomendados

- Decidir sobre clientes reales: los datos son sintéticos y el resultado describe las reglas con que se generaron.
- Decisiones de crédito, precio, cobranza o exclusión de un servicio.
- Leer el puntaje sin recalibrar como una probabilidad de fuga: con los pesos de clase su media es 0,416 y la prevalencia, 0,246.
- Leer las curvas como efectos causales («subir la satisfacción reduce la fuga»): describen lo que el modelo asigna.
- Puntuar valores fuera del rango de entrenamiento: la edad, por ejemplo, solo toma 6 valores (29, 30, 44, 45, 59, 60).
- Tratar «riesgo alto» como fuga observada, o usar productos activos como entrada: es la variable con que se generó el puntaje y el modelo recuperaría la regla en vez de predecir.
- Llevarlo a otra entidad, otro país u otro periodo sin reentrenarlo y validarlo.

Fuentes: datos/entrega3/complementos/calibracion.csv · datos/variables.csv · Informe final (metodologia.tex, resultados.tex, conclusiones.tex)

## Factores, métricas y datos

Con qué se evaluó, en qué datos y en qué punto de operación. XGBoost va al lado porque la EBM empata con él.

### Factores relevantes

- Cohorte de edad: el conjunto tiene 3 cohortes y el riesgo alto va del 8,8 % (44–45 años) al 45,7 % (59–60). La edad reúne el 37,8 % de la importancia del modelo.
- Productos activos: la variable excluida con la que se generó el puntaje; los errores grandes de la regresión siguen su escalera.
- Variables personales que entran al modelo: género (0,2 % de la importancia), ubicación (1,8 %) e ingreso (2,4 %).
- En la evaluación se separó la prueba por cohorte de edad y por productos activos (tabla de abajo).

Fuentes: datos/resultados/errores_segmentos.csv · datos/modelos/ebm_clasificacion.json

### Datos de entrenamiento y de evaluación

- **Conjunto:** COFINFAD (Mendeley Data, CC BY 4.0), sintético: ver DATA_STATEMENT.md
- **Partición:** aleatoria por índices, 80/20, con semilla 42, sin estratificar, antes de calcular cualquier estadístico del objetivo
- **Entrenamiento:** 38.978 clientes; riesgo alto: 24,6 %
- **Evaluación:** 9.745 clientes de prueba, tocados una sola vez; riesgo alto: 24,6 % (2.399 clientes)
- **Etiqueta:** riesgo alto = churn_probability por encima de su percentil 75 en el entrenamiento (0,3927)
- **Selección:** validación cruzada anidada de 5 × 3 pliegues, con 36 configuraciones por búsqueda
- **Punto de operación:** contactar al 25 % de mayor puntaje: 2.436 clientes; el último contactado tiene un puntaje de 0,6526

Fuentes: datos/metadatos.json · datos/entrega3/corridas.csv

### Métricas en la prueba

| Métrica | EBM | XGBoost | Qué mide |
| --- | ---: | ---: | --- |
| AUC de prueba [IC 95 %] | 0,8289 [0,8206; 0,8366] | 0,8283 [0,8203; 0,8359] | qué tan bien ordena a los clientes por riesgo |
| AUC externa (media ± desviación de 5 pliegues) | 0,8292 ± 0,0037 | 0,8332 ± 0,0049 | la validación anidada, antes de tocar la prueba |
| AUC-PR | 0,551 | 0,548 | orden de los positivos, sensible a la prevalencia |
| Captura con el 25 % de capacidad | 51,2 % | 51,0 % | parte de los clientes de riesgo alto que alcanza la campaña |
| Precisión con el 25 % de capacidad | 50,4 % | 50,2 % | parte de los contactados que son de riesgo alto |
| MCC con el 25 % de capacidad | 0,346 | 0,343 | correlación entre lo contactado y lo real |
| Error de calibración (ECE): sin recalibrar → isotónica | 0,170 → 0,006 | 0,007 (isotónica) | distancia entre la probabilidad dicha y la observada |
| RMSE de prueba [IC 95 %] | 0,05925 [0,05849; 0,06009] | 0,05917 [0,05841; 0,06002] | error del puntaje previsto |
| MAE y R² de prueba | 0,0472 y 0,216 | 0,0471 y 0,218 | error absoluto medio y varianza explicada |
| Cobertura del intervalo conforme al 90 % | 89,7 % | 90,0 % | parte de los puntajes reales dentro del intervalo |

*Frente a XGBoost, Random Forest y el árbol de decisión ninguna prueba encuentra diferencia tras la corrección de Holm (probabilidad de equivalencia práctica con XGBoost: 0,94); supera con significación a k-NN, la regresión logística, el SVM lineal y Naive Bayes. En el informe, la conclusión es que la EBM alcanza a los modelos de árboles, no que los supere.*

Fuentes: datos/resultados/clasificacion_prueba.csv · datos/resultados/regresion_prueba.csv · datos/calibracion_modelos.json · datos/conformal.json · datos/resultados/pruebas_clasificacion.csv

## Análisis por segmento

La prueba dividida por cohorte de edad y por productos activos, con el 25 % de capacidad.

### Dónde acierta y dónde falla

| Segmento | Clientes | Riesgo alto | AUC de la EBM | AUC de XGBoost | Riesgo alto que la EBM deja fuera | … y XGBoost | RMSE de la EBM | Sesgo de la EBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Edad: 29–30 | 3.224 | 18,3 % | 0,830 | 0,840 | 60 % | 38 % | 0,0598 | +0,0010 |
| Edad: 44–45 | 3.174 | 8,8 % | 0,922 | 0,930 | 91 % | 34 % | 0,0586 | 0,0000 |
| Edad: 59–60 | 3.347 | 45,7 % | 0,645 | 0,636 | 37 % | 56 % | 0,0594 | −0,0004 |
| Productos activos: 1 | 3.918 | 54,1 % | 0,997 | 0,999 | 55 % | 55 % | 0,0548 | −0,0546 |
| Productos activos: 2 | 2.939 | 9,3 % | 0,996 | 0,995 | 0 % | 0 % | 0,0067 | −0,0048 |
| Productos activos: 3 | 1.390 | 0,4 % | 0,999 | 0,988 | 0 % | 0 % | 0,0456 | +0,0453 |
| Productos activos: 4 | 1.016 | 0,0 % | — | — | — | — | 0,0954 | +0,0953 |
| Productos activos: 5 | 482 | 0,0 % | — | — | — | — | 0,1453 | +0,1452 |

*La cohorte de 59–60 años es la más difícil (AUC 0,645) y la de 44–45, la más fácil (0,922). Con la misma captura global, la EBM concentra la campaña en la cohorte mayor: deja fuera al 37 % de sus clientes de riesgo alto y al 91 % de los de 44–45; XGBoost reparte más. Por productos activos el sesgo de la regresión forma una escalera: es la parte del puntaje que depende de la variable excluida.*

Fuentes: datos/resultados/errores_segmentos.csv

## Ética, limitaciones y recomendaciones

Lo que hay que saber antes de usarlo.

### Consideraciones éticas

- Los datos son públicos y sintéticos: no hay personas reales identificables.
- La edad, el género y la ubicación son entradas. La edad domina el modelo (37,8 % de la importancia): una campaña real que lo siguiera trataría distinto a las personas según su edad, y eso habría que justificarlo y revisarlo.
- El modelo es legible: cada decisión se descompone exactamente en lo que aporta cada variable, lo que facilita auditarlo.
- No se mide fuga real: usarlo para negar o encarecer un servicio no tendría respaldo en estos datos.

Fuentes: datos/modelos/ebm_clasificacion.json · Informe final (metodologia.tex, resultados.tex, conclusiones.tex)

### Limitaciones y riesgos

- Un solo conjunto de datos, con rasgos sintéticos (tres cohortes de edad, una escalera de productos activos, transacciones sin señal): las conclusiones describen esas reglas.
- El objetivo es el puntaje de otro modelo, no la fuga observada; sin productos activos hay un techo de información: con la etiqueta de la mediana, la EBM cubre el 98 % de lo que se puede alcanzar sin esa variable.
- Las probabilidades sin recalibrar están sesgadas (ECE 0,170); hay que recalibrarlas.
- Los conjuntos conformes de clases cubren de más (96,6 % al 90 %): el 45 % de los clientes recibe las dos clases.
- Con cinco pliegues las pruebas pareadas tienen poca potencia; los tiempos son de un solo equipo.

Fuentes: datos/entrega3/comparacion_entrega2/techo.json · datos/resultados/entrega2.csv · datos/calibracion_modelos.json · Informe final (metodologia.tex, resultados.tex, conclusiones.tex)

### Recomendaciones

- Ordenar con el puntaje sin recalibrar e informar con la probabilidad recalibrada.
- Elegir la capacidad con el costo de cada contacto en mente: con el 25 % casi todos los modelos alcanzan a los mismos clientes, pero con el 50 % la EBM captura el 91,2 % y la regresión logística el 76,7 %.
- Validar en el tiempo con fuga observada (entrenar con un periodo y probar con el siguiente) antes de cualquier uso real.
- Revisar los resultados por cohorte de edad y reentrenar si cambia la distribución de los clientes.
- Mantener una revisión humana de cada decisión que afecte a un cliente.

Fuentes: datos/resultados/clasificacion_prueba.csv · Informe final (metodologia.tex, resultados.tex, conclusiones.tex)
