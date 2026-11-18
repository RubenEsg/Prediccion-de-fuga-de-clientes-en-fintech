# Protocolo experimental pre-registrado del modelo original

**Proyecto:** Predicción de fuga de clientes en fintech (COFINFAD).
**Modelo original:** Boosting con Desplazamiento Latente (BDL), `src/original_model.py`.
**Fecha de registro:** 17 de noviembre de 2026. **Etiqueta de Git:** `protocolo-modelo-original-v1`.

Este documento se congela **antes** de ejecutar los experimentos definitivos (guía, §6.1). Todo lo que aquí no figure
como confirmatorio es exploratorio y no sustenta por sí solo la condición R3. Cualquier desviación posterior se
declarará en el informe con su motivo.

## 1. Qué se sabía al registrar el protocolo

El protocolo se registra como parte de la corrección posterior a la retroalimentación del docente: el modelo que se
había presentado como original (la EBM) proviene de una librería y pasa a ser un benchmark.

Antes de este registro se hizo lo siguiente, y nada más:

- **Diseño del algoritmo con pilotos sobre la partición de entrenamiento.** Se evaluaron cinco iteraciones de diseño,
  todas con los cinco pliegues externos de la semilla 42 y sin búsqueda de hiperparámetros:
  1. ordenar a los clientes con un regresor del puntaje (descartada: AUC menor que el clasificador directo);
  2. separar el puntaje en índice y escalón con componentes de XGBoost y escalón duro (descartada: sin ganancia);
  3. primera versión de BDL: EM generalizado con paso de boosting y salida conforme empírica;
  4. cambios de ingeniería: varios árboles por vuelta de EM, calentamiento con parada temprana, inicialización de la
     mezcla sobre el histograma y regla de parsimonia para el número de clases;
  5. salida de la clasificación con la mezcla y una dispersión robusta en lugar de la distribución empírica (adoptada).
- **El conjunto de prueba (9.745 clientes) no intervino en el diseño del modelo original.** Se había usado en el
  Entregable 3 para evaluar los benchmarks, y se usará una sola vez para el modelo original, con este protocolo.
- **Las repeticiones 2 y 3 de la validación anidada no se han ejecutado para ningún modelo.**
- **Los conjuntos de datos adicionales solo se han descargado** para comprobar que están disponibles (forma y nombres
  de columnas). No se ha ajustado ningún modelo en ellos.
- La variable `active_products`, que el ETL excluye por fuga, se usó en los pilotos **solo para analizar** por qué el
  puntaje tiene escalones y para calcular el techo teórico. El modelo no la recibe en ningún momento.

Los pilotos mostraron que los mejores benchmarks están muy cerca del límite que permite el conjunto (el techo de
Bayes): la mejora posible es pequeña en las dos tareas. Las hipótesis se plantean sabiendo eso.

## 2. Hipótesis

Confirmatorias (problema principal, COFINFAD):

- **H1 (regresión).** El RMSE del modelo original es menor que el del mejor benchmark.
- **H2 (clasificación).** El AUC del modelo original es mayor que el del mejor benchmark.

«Mejor benchmark» es, en cada tarea, el de mejor métrica media del bucle externo entre los nueve (los siete de clase,
la EBM y LightGBM), sobre las quince particiones externas de la sección 4.

Confirmatorias secundarias (calidad probabilística; no cuentan para R3):

- **H3.** Con cobertura nominal del 90 %, el conjunto de predicción del modelo original es más corto que el intervalo
  conforme del mejor benchmark de regresión, y su cobertura empírica en prueba no es menor que la nominal más allá del
  error de muestreo.
- **H4.** La pérdida logarítmica del modelo original en prueba es menor que la del mejor benchmark de clasificación
  con sus probabilidades calibradas (isotónica ajustada con predicciones fuera de pliegue de entrenamiento).

Exploratorio: todo lo demás (MAE con la mediana predictiva, calibración por tramos, análisis por subgrupos,
sensibilidad, ruido, tamaño de muestra, cambio de distribución).

## 3. Métricas

- **Principal:** AUC-ROC en clasificación y RMSE en regresión. Son las del Entregable 2 y del Entregable 3 (`src/evaluacion.py`):
  no se cambian.
- **Secundarias:** en clasificación, precisión media (área bajo la curva de precisión-recall), pérdida logarítmica,
  Brier, error de calibración esperado y captura de la campaña del 25 %; en regresión, MAE, R², cobertura y largo de
  los intervalos o conjuntos de predicción al 80 %, 90 % y 95 %.

## 4. Particiones, repeticiones y semillas

- **Prueba:** el 20 % reservado en el Entregable 3 (semilla 42), sin cambios.
- **Validación anidada repetida, 3 × 5:** tres repeticiones de cinco pliegues externos, con tres pliegues internos
  para la búsqueda de hiperparámetros. Semillas globales 42, 43 y 44; cada semilla fija a la vez la partición externa,
  la interna, la semilla de los modelos y la del optimizador, igual para los diez modelos. La repetición de la semilla
  42 es la del Entregable 3 y sus corridas se reutilizan (la auditoría de fuga, `01_auditoria_fuga.py`, muestra que los
  datos son idénticos); para las semillas 43 y 44 se ejecutan los diez modelos.
- **Estabilidad por semilla:** cada modelo final se reajusta con diez semillas (42 a 51) y se evalúa en prueba.

## 5. Modelos y presupuesto de ajuste

Diez modelos por tarea, con el mismo preprocesamiento (`src/preprocesamiento.py`), las mismas variables de entrada y la
misma métrica de selección:

| Modelo | Combinación | Presupuesto (evaluaciones por búsqueda) |
|---|---|---|
| Siete modelos de clase y EBM | La mejor combinación (balanceo × optimizador) de cada uno en el Entregable 3, que es la del dashboard | El del Entregable 3: el tamaño de su rejilla, entre 30 y 48 |
| LightGBM (segundo método reciente, además de la EBM) | Ponderación de clases en clasificación; optimización bayesiana | 32 |
| BDL (original) | Sin balanceo; optimización bayesiana (Optuna, TPE) | 30, el mínimo que recibió un benchmark |

El espacio de búsqueda de BDL es el de `src/modelos.py`: `tasa` (log, 0,05 a 1), `profundidad` (1 a 6), `min_hoja`
(log, 5 a 200), `lambda_l2` (log, 0,01 a 10), `max_bins` (log, 32 a 1024), `n_componentes` (`auto` o 1) y, en
clasificación, `usar_puntaje` (verdadero o falso). Se reporta el presupuesto y el tiempo reales de cada modelo.

## 6. Pruebas de hipótesis

Los datos son de corte transversal (una fila por cliente), así que se usan las pruebas de la columna «datos
independientes» de la guía (Cuadro 3). Nivel de significancia α = 0,05, bilateral. En cada prueba el modelo original se
compara con cada uno de los nueve benchmarks y los p-valores se ajustan por Holm dentro de la prueba; se reportan el
p-valor crudo, el ajustado y el número total de comparaciones.

| | Regresión (H1) | Clasificación (H2) |
|---|---|---|
| **P1, global** | Friedman con corrección de Iman-Davenport sobre los 15 pliegues externos × 10 modelos; después, Wilcoxon de rangos con signo del original contra cada benchmark, con Holm | Igual, con el diagrama de diferencia crítica |
| **P2** | Prueba *t* corregida de Nadeau-Bengio sobre las 15 diferencias de RMSE por pliegue (razón prueba/entrenamiento 1/4) | Igual, sobre las diferencias de AUC |
| **P3** | Prueba de permutación pareada (cambio de signo, 10.000 permutaciones) sobre la diferencia de errores cuadráticos en prueba | DeLong para la diferencia de AUC en prueba |
| **P4** | Bootstrap pareado BCa (2.000 réplicas) de la diferencia de RMSE en prueba | McNemar en prueba, con cada modelo marcando como positivo su 25 % de mayor puntaje (la campaña) |

Tamaño del efecto: δ de Cliff y *d* de Cohen entre los 15 valores por pliegue de los dos modelos, y *d* pareada de las
diferencias. Se acompaña cada resultado significativo con la magnitud en unidades del problema.

## 7. Veredicto sobre R3

Se aplica la definición operativa de la guía (§6.4), por separado en cada tarea. El modelo original supera a los
benchmarks si se cumplen a la vez:

1. su métrica media en las 15 particiones externas es mejor que la del mejor benchmark;
2. al menos tres de las pruebas P1 a P4 rechazan la igualdad con el mejor benchmark con p ajustado < 0,05 (en P1 deben
   rechazar la prueba global y la comparación por pares);
3. el tamaño del efecto no es despreciable: |δ de Cliff| ≥ 0,147 o |*d* de Cohen| ≥ 0,2, calculados entre los valores
   por pliegue (no la *d* pareada);
4. en los conjuntos adicionales, es mejor en la métrica media en al menos dos de los cuatro y no es significativamente
   peor que el mejor benchmark en ninguno (prueba *t* corregida de Nadeau-Bengio sobre los cinco pliegues externos).

Si alguna condición falla, el veredicto es «no superior» y se analiza la causa.

## 8. Conjuntos de datos adicionales

Cuatro conjuntos públicos con objetivo continuo, de naturaleza distinta a COFINFAD (que es sintético), elegidos con el
criterio siguiente y **sin conocer el desempeño de ningún modelo en ellos**:

- dos **con una variable omitida**: una variable categórica que se sabe que desplaza el objetivo se retira para todos
  los modelos, igual que el ETL retira `active_products` en el problema principal. Es la situación para la que se
  diseñó el modelo;
- dos **sin modificar**, como control: no hay motivo para esperar un desplazamiento latente y el modelo debería
  reducirse a un boosting de mínimos cuadrados.

| Conjunto | Fuente | n | Predictores | Objetivo | Modificación |
|---|---|---|---|---|---|
| Seguros médicos | *Medical Cost Personal Datasets* (Lantz, *Machine Learning with R*); `insurance.csv` | 1.338 | 5 | Costo facturado | Se omite `smoker` |
| Eficiencia energética | UCI ML Repository, n.º 242 (Tsanas y Xifara, 2012) | 768 | 7 | Carga de calefacción (Y1) | Se omite el área de vidrio (X7); se retira Y2 |
| Abulón | UCI ML Repository, n.º 1 (Nash y otros, 1994) | 4.177 | 8 | Anillos | Ninguna |
| Vivienda de California | StatLib (Pace y Barry, 1997), vía `sklearn.datasets` | 20.640 | 8 | Valor mediano de la vivienda | Ninguna |

Varían en tamaño (768 a 20.640), dominio (salud, edificios, biología, vivienda), ruido (una simulación casi sin ruido
frente a datos de censo) y tipo de objetivo (continuo, entero, censurado por arriba).

En cada conjunto se repite el procedimiento del problema principal: 20 % de prueba (semilla 42), validación anidada de
5 × 3 con la semilla 42 para los diez modelos y, para la clasificación, la etiqueta «cuarto superior del objetivo» con
el umbral estimado solo con entrenamiento. Cada benchmark usa la combinación de la sección 5. Los resultados se
reportan por conjunto, sin promediar métricas de escalas distintas, y se consolidan con la prueba de Friedman con
corrección de Iman-Davenport sobre los rangos de los diez modelos en los cinco conjuntos (los cuatro adicionales y
COFINFAD) y Wilcoxon-Holm del original contra cada benchmark.

Expectativa declarada de antemano: ventaja en los dos conjuntos con variable omitida y empate en los dos de control.

## 9. Ablación y estudios de robustez

- **Ablación** (semilla 42, mismo procedimiento anidado y pruebas P2 y P3): BDL sin mezcla (`n_componentes = 1`), BDL
  sin puntaje en clasificación (`usar_puntaje = False`) y BDL con la salida conforme empírica.
- **Robustez** (descriptivos, sobre entrenamiento con validación cruzada salvo indicación): sensibilidad a cada
  hiperparámetro propio; ruido y valores atípicos en los predictores y en el objetivo; curva de aprendizaje con el
  10 %, 25 %, 50 % y 100 % de las filas y tiempo de cómputo al variar *n* y *p*; cambio de distribución dejando fuera
  una ciudad cada vez; calibración e incertidumbre en prueba; y análisis de errores por subgrupos.

## 10. Equidad y cómputo

Mismo equipo para todo (6 núcleos); se registran el tiempo de búsqueda, de ajuste y de inferencia de cada modelo. No se
da más presupuesto al modelo original que a ningún benchmark. Se reportan todas las corridas, no la mejor de varias.
