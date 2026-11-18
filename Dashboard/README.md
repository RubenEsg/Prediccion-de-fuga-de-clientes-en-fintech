# Fuga de clientes · Laboratorio del modelo

Dashboard predictivo en Plotly Dash del proyecto final de Machine Learning sobre la **fuga de clientes de una *fintech***
(conjunto público COFINFAD): explica el problema, explora los datos de entrenamiento y pone a predecir, en vivo, a los
modelos finales de cada tarea (el modelo original del proyecto y ocho benchmarks), con su incertidumbre y su explicación
local. Es el Bloque E de la guía «Proyecto Final: Tercer Entregable» y la sustentación se hace sobre él.

**En línea: <https://fuga-de-clientes-dash.onrender.com>** (Render, plan gratuito: si lleva un rato sin visitas, la primera carga tarda cerca de un
minuto).

Proyecto final de Machine Learning · Pregrado en Ciencia de Datos, Universidad del Norte · Valerie Martínez Pulido,
Abrahan Basto Martínez y Rubén Esguerra Fernández.

El **modelo original** del proyecto es **BDL (Boosting con Desplazamiento Latente)**, programado desde cero en
`src/original_model.py` con numpy y scipy. El puntaje del conjunto es un índice de los predictores menos 0,05 por cada
producto activo; esa variable se excluye por fuga de información, así que el puntaje queda en cinco niveles que los
benchmarks tratan como ruido. BDL modela ese escalón como una clase latente (`s = F(x) + c_z + e`): un EM generalizado
cuyo paso M sobre el índice es un paso de boosting contra el objetivo sin escalón. Clasifica con la probabilidad de
excedencia de la mezcla y da conjuntos de predicción conformes, que son una unión de tramos (uno por nivel).

Se compara con **nueve benchmarks**: los siete modelos de clase y dos recientes, LightGBM y la *Explainable Boosting
Machine* (EBM). La EBM fue el «modelo nuevo» del Entregable 3; como viene de la librería `interpret`, no es el modelo
original: aquí es un benchmark interpretable y conserva sus vistas.

**El veredicto no está escrito en ningún archivo de la app.** La regla de superioridad de la guía (§6.4, cuatro
condiciones por tarea) se aplica a `datos/original/veredicto.json` y `condicion4.json`, que son copias de los resultados
de la corrección, cada vez que se arma una vista (`datos_app.estado_veredicto`). Si una condición no se cumple, la app
dice «no superior» y explica por qué; si faltan resultados, dice que el veredicto es provisional o que está pendiente.

## Secciones

| Sección | Ruta | Qué tiene |
|---|---|---|
| **Contexto** | `/` | El problema y la cifra del modelo original con su veredicto, por qué importa, la pregunta y los objetivos, las correcciones (el modelo original es propio y la recencia se calcula solo con el entrenamiento), la formulación, los datos (fuente, licencia, diccionario de variables), el pipeline de la metodología, las limitaciones, la ética, las conclusiones y los autores |
| **EDA** | `/eda` | Análisis exploratorio del entrenamiento con filtros globales: objetivo y balance, distribuciones, explorador bivariado, correlaciones, faltantes y atípicos, y la variable que generó el puntaje |
| **Modelos** | `/modelos/<vista>` | Trece vistas en tres grupos |

Vistas de Modelos:

| Grupo | Vista | Qué tiene |
|---|---|---|
| Los diez modelos | Tabla de líderes | La combinación de cada modelo (balanceo, optimizador, métrica media ± desviación del bucle externo, prueba con su intervalo, tiempo de inferencia), ordenable y filtrable, con el modelo original y LightGBM |
| | Predicción en vivo | Formulario generado del esquema de variables, selector de modelo (BDL por omisión), predicción con probabilidad recalibrada o intervalo conforme y su cobertura, y explicación local. Con BDL, además, su conjunto de predicción (la unión de tramos) y cómo el índice se convierte en probabilidad |
| | Diagnóstico | Matriz de confusión, ROC, precisión-recall y calibración; o reales frente a predichos, residuos, ACF e histograma, de los diez modelos |
| | Interpretabilidad | Resumen SHAP global del modelo elegido, comparado con el modelo original |
| | Comparación estadística | El veredicto del original condición por condición, el diagrama de diferencia crítica con los diez modelos (Friedman y Nemenyi), el original frente a cada benchmark (P1 a P4, p ajustados por Holm y tamaños del efecto) y, aparte, el análisis del Entregable 3 de la EBM frente a sus siete referencias |
| El modelo original: BDL | Modelo original | La subsección que pide la guía: descripción y limitación que atiende, formulación, pseudocódigo, proposiciones, diferencias con la literatura, convergencia, lo que aprendió el índice, techo teórico, pruebas de hipótesis (con la sensibilidad de la prueba P4 a los empates de la campaña), robustez y veredicto explícito por tarea, con el análisis de la causa (las pruebas del predictor óptimo) |
| | Ficha del modelo | La *model card* de BDL (también en `MODEL_CARD.md`) |
| La EBM, benchmark interpretable | Algoritmo de la EBM | Formulación, pseudocódigo, sustento teórico, complejidad, pruebas y robustez de la EBM, como en el Entregable 3 |
| | Cómo decide | Las curvas de forma y la importancia de la EBM, leídas del modelo en vivo |
| | Un cliente | La explicación exacta de un cliente con la EBM y el simulador |
| | Campaña | A quién contactar si la capacidad no alcanza para todos (la EBM frente a cualquiera de los otros modelos) |
| | Robustez | Ablación, optimizadores, sensibilidad, costo y curva de aprendizaje de la EBM |
| | Errores | Dónde se equivoca la EBM y por qué |

Cada vista responde una pregunta con cifras leídas de los datos; el botón «…» de cada tarjeta abre la tabla con los datos
del gráfico y dice de qué archivos salen. Al pie de cada una, «Siguiente» lleva por el recorrido de la sustentación.
Las direcciones de la versión anterior (`/cliente`, `/comparacion`…) redirigen a su vista, y `/original` lleva a la del
modelo original. Las vistas de la EBM llevan arriba un aviso: es un benchmark, no el modelo original.

Mientras la corrección siga produciendo resultados, cada bloque cuyo archivo todavía no existe se muestra como
pendiente (qué falta y qué archivo lo traerá) y no rompe la vista; `datos/original/estado.json` dice qué había la última
vez que se armaron los datos.

## Cómo instalar y correr

Hace falta Python 3.13 (probado con 3.13.3 en Windows 11).

**En Windows:** doble clic en `iniciar_dash.bat`. La primera vez crea `.venv` en esta carpeta e instala
`requirements.txt` (unos minutos, con internet); después abre el navegador en <http://127.0.0.1:8060>. Para detener la
app, cierre la ventana negra.

**En cualquier sistema**, desde esta carpeta:

```
python -m venv .venv
.venv\Scripts\activate              (Windows)
source .venv/bin/activate           (macOS o Linux)
pip install -r requirements.txt
python app.py
```

Abra **http://127.0.0.1:8060**. Para otro puerto, defina la variable `PORT`. Si ya existe `.venv`, `python app.py`
funciona también con otro Python (el global o el botón de ejecutar de VS Code): la app se vuelve a lanzar sola con el
Python de `.venv`. No necesita internet ni el resto del proyecto: todo lo que usa está en `datos/` y `artifacts/`.

## Despliegue en Render

En el repositorio del proyecto (<https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech>) el dash está
en la carpeta `Dashboard/`, y `render.yaml` (Blueprint) y el workflow de las pruebas están en la raíz del repositorio,
porque Render y GitHub Actions los buscan allí: `render.yaml` dice `rootDir: Dashboard` y el workflow,
`working-directory: Dashboard`. Aquí están `requirements.txt` con versiones exactas (incluido `gunicorn`) y
`.python-version`. Pasos:

1. En Render: **New → Blueprint**, elegir el repositorio `Prediccion-de-fuga-de-clientes-en-fintech` (rama `main`).
   Render lee `render.yaml`: construye en `Dashboard/` con `pip install -r requirements.txt` y arranca con
   `gunicorn app:server --workers 1 --threads 4 --timeout 120`. Solo vuelve a desplegar cuando cambia `Dashboard/`.
2. Revisar los registros de la construcción y abrir la URL pública; comprobar las tres secciones y las trece vistas.

El plan gratuito tiene 512 MB y 0,1 CPU, y se suspende tras un rato sin visitas: el primer acceso tarda en responder
(abrir la URL unos minutos antes de la sustentación). La app nunca entrena ni optimiza: solo carga artefactos. Al arrancar,
antes de atender visitas, carga las tuberías de los modelos y arma cada vista una vez (`FUGA_PRECALENTAR=1`; unos 25 s en
Render con las 14 del Entregable 3), así la primera visita no espera. Lo hace en el hilo principal y no en uno aparte: en Render, importar el lector de parquet
en dos hilos a la vez dejó rotas todas las lecturas de parquet.

El `Dockerfile` reproduce ese servidor (Linux, Python 3.13.3, `requirements.txt`, el mismo `gunicorn`). Con los límites del
plan gratuito (`docker run --memory=512m --cpus=0.1 -p 8060:8060 fuga-dash`), el 15-nov-2026: el precalentamiento tardó
40 s; con los 14 modelos cargados la memoria quedó en 379 MiB y el pico, recorriendo las tres secciones y las doce vistas,
fue de 428 MiB, sin reinicios ni errores de consola; cada vista tardó entre 4 y 11 s en cargar y la predicción con k-NN
unos 20 s (su explicación evalúa cientos de consultas sobre los 38.978 clientes). Para la sustentación, la predicción con
la EBM o XGBoost responde en pocos segundos; con el plan Starter (0,5 CPU) todo es unas cinco veces más rápido.

Esas medidas son anteriores al modelo original y no se han repetido con él en el contenedor. BDL no añade dependencias
(solo numpy, scipy y scikit-learn, que ya estaban): sus dos tuberías pesan 0,24 MB en disco y, con las librerías ya
cargadas, ocupan menos de 1 MB en memoria y se cargan en menos de 0,1 s (medido en Windows con `tracemalloc`); puntuar con
ellas los 9.745 clientes de prueba en las dos tareas tarda cerca de un segundo. La vista «Modelo original» es la más larga
del dash y la que más tarda en armarse.

Para caber en 512 MB: XGBoost sin CUDA, sin la librería `shap` en el servidor y el Random Forest de clasificación en una
representación compacta (`bosque.py`: sus mismos 300 árboles en arreglos de numpy, 16 MB en memoria en lugar de 85, con
probabilidades idénticas al artefacto en los 9.745 clientes de prueba). Si aun así Render reiniciara por memoria, la
variable `FUGA_CACHE_MODELOS` limita cuántos modelos guarda.

## Cómo funciona

**Los modelos.** `artifacts/<tarea>/<modelo>/` guarda, para cada modelo en vivo de cada tarea (los 8 del Entregable 3 y
el modelo original, `bdl`), la tubería final (preprocesado + modelo) de su combinación, reentrenada con todo el
entrenamiento, y su `metadata.json`: versiones de las librerías, semilla, hiperparámetros, esquema de entrada (nombres,
tipos y rangos válidos), métricas del bucle externo y de la prueba, tiempos y el resultado de la **prueba de ida y
vuelta** (el artefacto cargado de disco da exactamente las mismas predicciones que el modelo en memoria sobre los 9.745
clientes de prueba). `artifacts/indice.json` lista los 18 con su huella SHA-256. `modelos_vivos.py` los carga y los
evalúa (el Random Forest de clasificación, desde su forma compacta `bosque.npz`, comprobada idéntica). LightGBM, el
segundo benchmark reciente, no se sirve en vivo, para no añadir una dependencia al servidor: está en la tabla de
líderes, en el diagnóstico y en la comparación estadística, con las predicciones que guardó la corrección.

**El modelo original.** `src/original_model.py` es una copia exacta del módulo de la corrección
(`ModeloOriginal/src/original_model.py`; una prueba compara los dos archivos enteros y otra comprueba su huella): las
tuberías guardadas nombran ese módulo. En regresión, la app muestra el pronóstico (la media de la mezcla), el intervalo conforme
común a todos los modelos y el **conjunto de predicción del modelo**, una unión de tramos alrededor de cada nivel del
puntaje (`RegresorBDL.conjunto_residual`), con su cobertura nominal y la medida en la prueba. En clasificación, la
probabilidad de excedencia y cómo sale del índice del cliente: el aporte de cada clase latente según su nivel quede por
encima o por debajo del umbral.

**La EBM.** Una EBM sin interacciones es una suma de tablas: para cada variable, unos cortes y un puntaje por tramo.
`motor.py` evalúa las tablas exportadas del modelo final (78 tablas en `datos/modelos/ebm_*.json`) con numpy, da lo mismo
que el modelo guardado (diferencias del orden de 10⁻¹⁶) y descompone cada puntaje en el aporte exacto de cada variable.

**La incertidumbre.** En clasificación, una probabilidad recalibrada (regresión isotónica ajustada con las predicciones
fuera de pliegue del entrenamiento) y un conjunto conforme de clases; en regresión, un intervalo conforme a partir de los
residuos fuera de pliegue. La cobertura de cada uno se mide en la prueba y se muestra junto a la nominal. Con BDL ese
intervalo simétrico no cumple su cobertura nominal (su error cae en niveles separados), y la app lo dice: lo que la
cumple es su conjunto de predicción, que además es mucho más corto.

**Las explicaciones.** `explicacion.py` calcula valores SHAP por variable original. En vivo son exactos en los modelos
lineales, Naive Bayes, XGBoost, la EBM y BDL (con `RegresorBDL.contribuciones`; en clasificación reparten el índice
F(x), no la probabilidad), y se estiman por permutaciones antitéticas (con su error de Monte Carlo) en el árbol, Random
Forest y k-NN: la librería `shap` no cabe en el servidor gratuito. El resumen global, calculado fuera de
línea, es exacto también en el árbol y Random Forest (TreeSHAP). Cada explicación cumple base + suma de aportes = salida
del modelo, y se verificaron contra la librería `shap`.

## De dónde salen los datos

Se arman con tres guiones a partir de la carpeta de trabajo del proyecto (el Entregable 3, con los modelos guardados, la
sección de resultados del informe y la corrección, `ModeloOriginal/`), que solo se leen:

```
"..\entrega 3\.venv\Scripts\python.exe" -B preparar_datos.py         (datos/: clientes, resultados, EDA, la EBM en JSON)
"..\entrega 3\.venv\Scripts\python.exe" -B preparar_artefactos.py    (artifacts/, calibración, intervalos, SHAP, Nemenyi)
"..\entrega 3\.venv\Scripts\python.exe" -B preparar_original.py      (el modelo original y LightGBM, encima de lo anterior)
```

`preparar_original.py` no borra ni rehace nada de los otros dos: copia `src/original_model.py`, deja los dos artefactos
de BDL con su ida y vuelta, añade BDL y LightGBM a las tablas que ya existían (puntajes y pronósticos de prueba,
calibración, intervalos conformes, resumen SHAP, `cd.json` con los diez modelos) y arma `datos/original/` con lo que
lee la vista del modelo original: copias exactas de los resultados de la corrección (`veredicto.json`,
`pruebas_hipotesis.csv`, `pliegues_<tarea>.csv`, `techo.json`, `optimo.json`, `empates_campana.json`, `auditoria_fuga.json`,
`condicion4.json`, los estudios de robustez…) y extractos (las filas de la tabla de líderes, el modelo final por dentro, los conjuntos
adicionales). Se
puede volver a correr cada vez que la corrección tenga resultados nuevos (tarda menos de un minuto): un insumo que todavía
no existe se anota en `datos/original/estado.json` y su bloque se muestra como pendiente. La app lee los datos al
arrancar y los guarda en memoria: después de correrlo hay que reiniciarla (en Render, volver a desplegar). Al final vuelve a generar
`MODEL_CARD.md` y `DATA_STATEMENT.md`, que citan el veredicto y otras cifras de esos archivos, para que no queden atrás
de los datos. Busca la corrección en `../entrega 3/repo_entregable3/ModeloOriginal`, en `../ModeloOriginal` o en la
variable `MODELO_ORIGINAL`.

`datos/origen.json` y `artifacts/indice.json` anotan, para cada archivo, de dónde salió, cómo y su huella. Ninguna cifra
de la app está escrita a mano: todas se leen de `datos/` o de `artifacts/`, o las calculan los modelos.

## Cómo se comprueba

| Comando | Qué comprueba |
|---|---|
| `python -m pytest -q` | Artefactos completos, ida y vuelta de los 18 modelos, motor de la EBM igual a su artefacto, propiedades de las explicaciones (eficiencia, coeficiente por desvío, reproducibilidad), cobertura de los conjuntos e intervalos conformes, validación de entradas, rutas y tres de las comprobaciones de abajo (`probar_app.py`, `verificar_cifras.py` y la parte estática de `verificar_estilo.py`). Del modelo original (`tests/test_original.py`): que su código es idéntico al de la corrección y no importa ningún modelo, la ida y vuelta de sus artefactos, que su probabilidad es la excedencia de la mezcla, que su conjunto de predicción es una unión de tramos con la cobertura que muestra la app, que las tablas tienen los diez modelos, que el veredicto de las vistas es el que da la regla aplicada a `veredicto.json` y `condicion4.json`, y que la sección de novedad no presenta como nueva la regla de aprendizaje y su tarjeta de la búsqueda dice lo que dice `literatura.json`. Necesita `pip install -r requirements-dev.txt`. Corre en GitHub Actions en cada push que cambie `Dashboard/` (`.github/workflows/dashboard.yml`, en la raíz del repositorio) |
| `python probar_app.py [vista]` | Que cada sección y vista se arma y que todos sus callbacks responden, sin navegador |
| `python verificar_cifras.py` | Que las cifras principales de cada vista son las de los archivos de `datos/` |
| `python verificar_motor.py` | Que el motor en numpy da lo mismo que el modelo guardado y reproduce las cifras del informe |
| `python verificar_estilo.py [--vivo PUERTO]` | Las reglas del sistema de diseño, en el código y en el navegador (Playwright y Microsoft Edge) |
| `python capturar.py --puerto PUERTO` | Una captura de cada vista a 1366 y a 400 px |

## Sistema de diseño

Página gris claro, tarjetas casi blancas, navegación fija arriba con la sección activa en píldora negra, títulos grandes
en peso normal y barras rayadas con la serie en foco sólida. Letra Inter, incluida en `assets/fuentes/` (licencia SIL
OFL 1.1). El color solo codifica significado (cinco roles): azul para el modelo en foco (el modelo original, BDL, en las
vistas de los diez modelos y en la suya; la EBM en las vistas de su grupo y en la mitad del Entregable 3 de la
comparación), rosa para los benchmarks con que compite en la parte alta de la tabla (la EBM, LightGBM y los modelos de
árboles), gris para los lineales, k-NN y Naive Bayes, casi negro para lo observado y gris claro para el contexto.
`DISENO.md` lo explica entero y `CONTRATO.md` es la guía para escribir o cambiar una vista.

## Archivos

```
app.py                  entrada (python app.py; en Render, gunicorn app:server): secciones, vistas, redirecciones
paginas/                las tres secciones: contexto.py, eda.py, modelos.py (y la página de dirección desconocida)
secciones/              las vistas de Modelos (original.py es la del modelo original) y las piezas que reutilizan Contexto y EDA
src/original_model.py   el modelo original (BDL), copia exacta del de la corrección: lo nombran sus tuberías guardadas
modelos_vivos.py        los 18 modelos en vivo desde artifacts/: predicción, incertidumbre, explicación y lo propio de BDL
bosque.py               el Random Forest de clasificación en arreglos de numpy (idéntico, cabe en 512 MB)
explicacion.py          valores SHAP por variable original, con el método de cada familia de modelos
motor.py                la EBM en numpy: puntuar, explicar, simular, curvas, campaña
datos_app.py, formato.py, ui.py, tokens.py, assets/      lectura de datos, números en es-CO y sistema de diseño
artifacts/              las 18 tuberías finales con su metadata.json (16 del Entregable 3 y las dos de BDL)
datos/                  lo que la app lee; origen.json dice de dónde salió cada archivo
datos/original/         los resultados de la corrección: veredicto, pruebas, pliegues, techo, robustez, el modelo por dentro
tests/                  pruebas (pytest); test_original.py, las del modelo original
preparar_datos.py, preparar_artefactos.py               arman datos/ y artifacts/ desde el proyecto
preparar_original.py    añade encima el modelo original y LightGBM desde ModeloOriginal/ (se vuelve a correr sin rehacer lo demás)
generar_fichas.py       escribe MODEL_CARD.md y DATA_STATEMENT.md con las cifras de datos/ y artifacts/
verificar_*.py, probar_app.py, capturar.py              las comprobaciones
MODEL_CARD.md, DATA_STATEMENT.md                        ficha del modelo y declaración de los datos
Dockerfile, requirements.txt, requirements-dev.txt, .python-version, iniciar_dash.bat
                        (render.yaml y .github/workflows/dashboard.yml están en la raíz del repositorio)
LICENSE, CITATION.cff   licencia del código (MIT) y forma de citar
```

## Datos y licencias

Conjunto de datos: Muñoz Guerrero, L. E., Ceballos, Y. F. y Trejos Rojas, L. D. (2025). *COFINFAD: Colombian fintech
financial analytics dataset*. Mendeley Data, V1. doi:10.17632/mhb4zn3258.1 (licencia CC BY 4.0). Los datos son
sintéticos: las conclusiones describen las reglas con que se generaron, no clientes reales. Código: licencia MIT
(`LICENSE`). Fuente Inter: SIL Open Font License 1.1 (`assets/fuentes/OFL.txt`).
