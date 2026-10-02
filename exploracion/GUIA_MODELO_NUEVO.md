# Guía para elegir el modelo nuevo del Entregable 3

Última revisión: 29 de septiembre de 2026. Todas las cifras son medias de 5 pliegues sobre el
conjunto de **entrenamiento** (semilla 42, misma partición 80/20 que la Entrega 2, preprocesamiento
y selección de variables ajustados dentro de cada pliegue). **La prueba no se ha tocado.** Los
guiones y las tablas completas están en esta carpeta (`benchmark_modelo_nuevo`, `benchmark_ronda2`,
`benchmark_ronda3` y `benchmark_ronda4`, cada uno con su `.py`, `.json` y `.md`).

## 1. Lo primero: qué permiten los datos

| Hallazgo | Evidencia |
|---|---|
| El objetivo es casi determinista | Un modelo *oráculo* que conoce `active_products` explica el 99,94 % de `churn_probability` (R² 0,9994; AUC 0,9999). Según la ficha del conjunto en Mendeley, `churn_probability` es ella misma la salida de un modelo predictivo con ventana de 30 días. |
| `active_products` es independiente del resto | Correlación 0,007 con la suma de los cinco productos. Ninguna otra variable la reconstruye. |
| Sin ella, la señal está en 4 variables | `app_logins_frequency` (ρ = −0,28), `age` (+0,21), `satisfaction_score` (−0,19) y `base_satisfaction` (−0,16). Las transacciones no aportan nada (|ρ| < 0,02). |
| **Hay un techo** | Marginalizando el oráculo sobre `active_products` se obtiene el mejor pronóstico posible sin ella: **AUC ≈ 0,723 (etiqueta mediana) y R² ≈ 0,230**. Ningún algoritmo puede superarlo con estas variables. |
| La Entrega 2 quedó lejos por dos motivos | La relación es **no lineal** (por eso fallan la logística y el Lasso lineales) y hay **74 columnas de ruido** (por eso falla k-NN). No era un problema de falta de datos ni de optimización. |

Consecuencia: no existe un modelo con un puntaje "mucho mayor". La meta realista es pasar de
0,68 a 0,72 de AUC y de 0,15 a 0,23 de R², y eso lo logran varias opciones. La elección se hace
por lo que cada una permite contar y defender.

## 2. Las tres rutas, con sus puntajes

Referencias: Entrega 2 = logística L1 (AUC 0,6824) y Ridge/Lasso (R² 0,1511); XGBoost de clase =
AUC 0,7220 y R² 0,2224.

### Ruta A. Un modelo de clase con un cambio

| Modelo | Cambio | AUC mediana | AUC percentil 75 | R² |
|---|---|---|---|---|
| k-NN (k = 200) | Selección de 4 variables dentro del pipeline | 0,7211 (antes 0,6645) | 0,8322 | 0,2232 (antes 0,0971) |
| SVM de núcleo RBF | Aproximación de Nyström + selección de 6 variables | 0,7219 (antes 0,6916) | 0,8326 | 0,2254 |
| Regresión logística y Ridge | Splines cúbicos por variable (modelo aditivo, GAM) + selección de 6 | 0,7212 (antes 0,6824) | 0,8288 | **0,2285** (antes 0,1511) |
| Random Forest | Selección de 6 variables | **0,7239** | 0,8331 | 0,2091 |
| Árbol + logística por hoja (*Logit Leaf Model*) | Híbrido propio de la literatura de churn, con 6 variables | 0,7214 | 0,8292 | 0,2278 |
| XGBoost | Variante DART (*dropout* de árboles) | 0,7225 | 0,8315 | — |
| XGBoost | Afinado con Optuna (25 ensayos, anidado) | 0,7236 (sin afinar 0,7220) | — | 0,2247 (sin afinar 0,2224) |
| Naive Bayes | Selección de 6 variables | 0,6763 | 0,7477 | — |

Las interacciones entre variables no aportan (splines con interacciones: 0,7204): el efecto es
aditivo. En k-NN cada variable de más resta (4 → 6 → 8 variables: 0,7211 → 0,7201 → 0,7180).

### Ruta B. Un modelo nuevo de la literatura

| Modelo | En las notas del curso | AUC mediana | AUC percentil 75 | R² | Segundos por ajuste |
|---|---|---|---|---|---|
| EBM, Explainable Boosting Machine (GA²M) | no aparece | 0,7214 | 0,8291 | 0,2246 (0,2280 con 6 variables) | 10 |
| CatBoost | solo mencionado | 0,7218 | 0,8316 | 0,2235 | 5 |
| LightGBM | solo mencionado | 0,7222 | 0,8316 | 0,2204 | 1,4 |
| HistGradientBoosting + selección de 5 variables | solo mencionado | 0,7234 | **0,8358** | 0,2252 | 1 |
| Mezcla de cuatro boosting (promedio de rangos) | — | 0,7232 | 0,8317 | 0,2248 | 9 |
| Stacking (logística + HGB + k-NN + NB) | mencionado | 0,7231 | 0,8332 | 0,2238 | 5,5 |
| MLP (128, 64) | solo mencionado | 0,7073 | 0,7780 | 0,0571 | 3 |

Lo que se usa hoy para este tema, según la revisión hecha el 29 de septiembre de 2026:

- En churn sobre datos tabulares el estándar sigue siendo el *gradient boosting* (XGBoost,
  LightGBM, CatBoost). Las tendencias de la literatura reciente son los ensambles y el stacking,
  los modelos híbridos (como el *Logit Leaf Model*), la evaluación orientada al beneficio, las
  probabilidades calibradas y las redes para secuencias de transacciones.
- En tabulares en general, el banco de pruebas TabArena (2025) pone arriba a RealMLP y TabM
  (redes), seguidos de LightGBM y CatBoost; los modelos fundacionales TabPFN-2.5, TabICL y TabFM
  de Google (junio de 2026) compiten sin ajuste de hiperparámetros.
- En este proyecto no se probaron: TabPFN exige aceptar una licencia con cuenta de Hugging Face
  (ver `prueba_tabpfn.py`), y RealMLP, TabM y TabICL necesitan PyTorch. Dado el techo, su
  ganancia máxima posible sería de una milésima de AUC.

### Ruta C. Un modelo creado por el equipo: información privilegiada (LUPI)

Idea: `active_products` no puede ser predictor, pero se conoce para las filas de entrenamiento. Se
usa **solo al entrenar** para quitarle al objetivo la parte que depende de ella (se resta la media
de cada nivel) y el modelo aprende el residuo con las demás variables. Al predecir no se usa.
Es el paradigma *learning using privileged information* de Vapnik y Vashist (2009), unificado con
la destilación por López-Paz y otros (2016).

| Variante | AUC mediana | AUC percentil 75 | R² |
|---|---|---|---|
| Objetivo depurado + HGB | 0,7232 | 0,8303 | **0,2303** |
| Oráculo marginalizado (mezcla suavizada) | **0,7239** | 0,8304 | **0,2303** |

Es la única ruta que alcanza el techo en regresión (+0,008 de R² sobre XGBoost, p < 0,001). En
clasificación con la mediana empata con los mejores; con el percentil 75 queda por debajo de los
modelos directos con pocas variables.

**Riesgo que hay que valorar:** el equipo declaró `active_products` como fuga. Aquí nunca entra al
predecir y solo se usan filas de entrenamiento, pero un evaluador estricto puede objetarlo. Si se
presenta, debe ir con su versión sin información privilegiada al lado y con esta explicación.

## 3. Recomendación

Presentar un paquete de tres piezas, que cubre las dos lecturas del requisito del profesor
(modelo nuevo no explicado en clase, o modelo de clase con algo diferente):

1. **Modelo nuevo oficial: EBM (GA²M).** No aparece en las notas, es otra familia de modelos y
   es interpretable por construcción: sus funciones de forma muestran la no linealidad de las
   cuatro variables, que es justo la razón por la que falló la Entrega 2. Referencia: Lou,
   Caruana, Gehrke y Hooker (2013), paquete `interpret`.
2. **Modelos de clase con un cambio: selección de variables dentro del pipeline y splines.**
   Cuesta muy poco (son pasos de scikit-learn: selección + `SplineTransformer`) y convierte a
   k-NN, SVM y la logística de la Entrega 2 en modelos competitivos. Además mejora las corridas de
   los 7 modelos de la guía.
3. **Sección "techo de información".** El oráculo y su marginalización demuestran que los modelos
   están a una milésima del máximo alcanzable. El modelo con información privilegiada va aquí como
   aportación propia y opcional, con su advertencia.

Si el equipo prefiere **un solo modelo** y lo más barato de integrar: logística y Ridge con
splines y selección de variables. Empata con la EBM, entrena en décimas de segundo, admite
`class_weight` y entra en los cuatro optimizadores sin dependencias nuevas.

## 4. Qué se puede afirmar y qué no

| Afirmación | ¿Se sostiene? |
|---|---|
| El modelo nuevo supera a la Entrega 2 | Sí: +0,04 de AUC y +0,07 a +0,08 de R², en los 5 pliegues (p < 0,001). Falta confirmarlo en prueba con DeLong y Diebold-Mariano. |
| El modelo nuevo supera a XGBoost | No. Las diferencias son de una milésima y no significativas. |
| Se puede llegar a 0,80 de AUC o más | Solo con la etiqueta del percentil 75, que no es comparable con el 0,6832 de la Entrega 2 (la misma logística da 0,7641 con esa etiqueta). |
| Incluir `active_products` como predictor | No: da AUC 0,9999 porque reproduce la fórmula de la etiqueta. |
| Restaurar `nps_score` u `occupation` | No aporta (+0,0004 de AUC, no significativo). |

## 5. Siguientes pasos

1. Elegir el modelo nuevo oficial y la estrategia de etiquetas (mediana para comparar con la
   Entrega 2; percentil 75 para balanceo y el resto del pipeline).
2. Añadir a `src/` el paso de selección de variables, la opción de splines y la especificación de
   la EBM con su espacio de búsqueda.
3. Sección "Comparación con la Entrega 2" sobre la prueba: logística L1 y Lasso de la Entrega 2,
   XGBoost y el modelo nuevo, con DeLong, intervalo bootstrap, Diebold-Mariano y tamaño del efecto.
4. Lanzar las corridas del pipeline combinatorio con el modelo nuevo incluido.

## Referencias

- De Caigny, Coussement y De Bock (2018). A new hybrid classification algorithm for customer churn
  prediction based on logistic regression and decision trees. *European Journal of Operational
  Research*, 269(2), 760-772. <https://ideas.repec.org/a/eee/ejores/v269y2018i2p760-772.html>
- Vapnik y Vashist (2009). A new learning paradigm: learning using privileged information.
  *Neural Networks*. Resumen en <https://users.sussex.ac.uk/~nq28/lupi/>
- López-Paz, Bottou, Schölkopf y Vapnik (2016). Unifying distillation and privileged information.
  ICLR. <https://arxiv.org/abs/1511.03643>
- Lou, Caruana, Gehrke y Hooker (2013). Accurate intelligible models with pairwise interactions.
  KDD. <https://www.microsoft.com/en-us/research/publication/accurate-intelligible-models-pairwise-interactions/>
- Documentación de la EBM: <https://interpret.ml/docs/ebm>
- TabArena: a living benchmark for machine learning on tabular data (2025).
  <https://arxiv.org/pdf/2506.16791>
- TabPFN-2.5 (2025). <https://arxiv.org/pdf/2511.08667>
- COFINFAD en Mendeley Data. <https://data.mendeley.com/datasets/mhb4zn3258/1>
- Notas del curso. <https://lihkir.github.io/MachineLearning/intro.html>
