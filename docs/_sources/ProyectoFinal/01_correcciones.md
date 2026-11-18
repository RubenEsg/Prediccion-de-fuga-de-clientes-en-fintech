# 1. Trazabilidad de las correcciones

Este capítulo recoge cada observación del docente, qué se hizo, dónde está en el libro y cuál es la evidencia. Incluye
las dos observaciones del entregable anterior y las tres de la primera versión del proyecto final (calificada con 3,0
por el tope que fija la guía cuando el modelo original proviene de un `import`).

## 1.1. Tabla de trazabilidad

La {ref}`Tabla 1.1 <tab-trazabilidad>` tiene una fila por observación.

(tab-trazabilidad)=
**Tabla 1.1.** Trazabilidad de las correcciones: observación del docente, acción realizada, ubicación en el libro y
evidencia.

| N.º | Observación del docente | Acción realizada | Ubicación en el libro | Evidencia |
|---|---|---|---|---|
| 1 | «La EBM es un modelo ya publicado y se usa a través de la librería `interpret`, sin programar el núcleo del algoritmo. Para la guía eso es un *import* y fija el tope en 3.0.» | La EBM deja de presentarse como modelo original y pasa a ser un benchmark reciente. El modelo original es ahora **BDL** (Boosting con Desplazamiento Latente), programado desde cero con `numpy` y `scipy`: discretización, árboles, EM de la mezcla y salida conforme. Se añade LightGBM como segundo benchmark reciente | Capítulos 4 y 5 | [`ModeloOriginal/src/original_model.py`](https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech/blob/main/ModeloOriginal/src/original_model.py); la prueba `test_el_modelo_original_no_importa_ningun_modelo` revisa las importaciones del módulo; sección 5.8 |
| 2 | «Para subir la nota necesitan una regla de aprendizaje o combinación propia, programada por ustedes.» | BDL tiene una regla de aprendizaje y una regla de salida programadas por el equipo. La de aprendizaje es un EM generalizado cuyo paso M sobre el índice es un paso de boosting contra el objetivo sin escalón: una variante, para un índice compartido con desplazamientos latentes, de la familia de EM con boosting (sección 4.4). La de salida es la probabilidad de excedencia de la mezcla y los conjuntos de predicción con la densidad de la mezcla. Cada una tiene su derivación, su demostración y su prueba unitaria | Secciones 5.3 a 5.5 | Algoritmo 1; Proposiciones 1 a 5; `ModeloOriginal/tests/test_original_model.py` |
| 3 | «La recencia se calcula con la fecha máxima de todo el detalle, que incluye el periodo de prueba.» | La partición se hace antes de derivar variables y la fecha de corte de la recencia se calcula solo con los clientes de entrenamiento (`fecha_de_corte` en `ModeloOriginal/src/datos.py`). La auditoría comprueba que la fecha es la misma (29 de diciembre de 2023), que no cambia ninguna celda y que, aunque cambiara, la recencia estandarizada no dependería de ella | Sección 2.5 | {ref}`Tabla 2.6 <tab-fuga>`; `ModeloOriginal/resultados/auditoria_fuga.json`; `tests/test_marco.py` |
| 4 | Entregable anterior: no se nombraba Successive Halving ni su factor de reducción. | Se nombra y se explica por qué la búsqueda con dos fidelidades del proyecto no es Successive Halving: todas las configuraciones se evalúan con el presupuesto bajo y solo la ganadora se reajusta con el alto (factor 3 en Random Forest, de 100 a 300 árboles; factor 4 en la EBM, de 2 a 8 bolsas); no hay rondas que descarten configuraciones | Sección 3.3 | Dashboard › Contexto › «Dos fidelidades, no Successive Halving», con la comparación lado a lado |
| 5 | Entregable anterior: el carácter sintético de los datos limita la utilidad práctica. | Se reconoce como la primera limitación del proyecto y se evalúa el modelo original en cuatro conjuntos de datos públicos, además de COFINFAD: dos de mediciones reales (Abulón y Vivienda de California) y dos simulados (Seguros médicos y Eficiencia energética) | Capítulos 7 y 9 | Dashboard › Contexto › aviso de datos sintéticos; `Dashboard/DATA_STATEMENT.md`; {ref}`Tabla 7.3 <tab-adicionales>` |

## 1.2. Resultados que cambian con las correcciones

La guía pide recalcular todo lo que dependa de un resultado modificado. Se revisó observación por observación:

- **Recencia (3).** No modifica ningún resultado. La auditoría muestra que las tablas de predictores de entrenamiento y
  de prueba son idénticas antes y después de la corrección (misma huella SHA-256), porque la fecha de corte calculada
  solo con entrenamiento es la misma. Por eso las 160 corridas del Entregable 3 siguen siendo válidas y la repetición
  de la semilla 42 se reutiliza tal cual; las dos repeticiones nuevas ya usan el flujo corregido.
- **Modelo original (1 y 2).** Añade resultados: los de BDL y LightGBM, las dos repeticiones nuevas de la validación
  anidada para los diez modelos, las pruebas de hipótesis con el modelo original incluido, los conjuntos adicionales y
  los estudios de robustez (capítulos 6 y 7). La tabla de líderes, el diagrama de diferencia crítica y la comparación
  estadística del dashboard se regeneraron con el modelo original incluido.
- **Orden de los modelos.** La sección 6.5 compara el orden de los modelos antes y después de la corrección.

## 1.3. Lo que no se pudo corregir

El conjunto de datos sigue siendo sintético: no existe una versión real de COFINFAD. La evaluación en conjuntos
adicionales, dos de ellos con mediciones reales, mide el comportamiento del modelo fuera de COFINFAD, pero no convierte
sus resultados sobre COFINFAD en evidencia sobre clientes reales (capítulo 9).
