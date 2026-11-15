# Fuga de clientes · Laboratorio del modelo

Dashboard predictivo en Plotly Dash del proyecto final de Machine Learning sobre la **fuga de clientes de una *fintech***
(conjunto público COFINFAD): explica el problema, explora los datos de entrenamiento y pone a predecir, en vivo, a los
ocho modelos finales de cada tarea, con su incertidumbre y su explicación local. Es el Bloque E de la guía «Proyecto
Final: Tercer Entregable» y la sustentación se hace sobre él.

Proyecto final de Machine Learning · Pregrado en Ciencia de Datos, Universidad del Norte · Valerie Martínez Pulido,
Abrahan Basto Martínez y Rubén Esguerra Fernández.

La conclusión es la del informe y la app no la sube de tono: el modelo propuesto, una *Explainable Boosting Machine*
(EBM), **alcanza** a los mejores modelos de árboles (empata con XGBoost, Random Forest y el árbol de decisión) y supera a
los lineales, a k-NN y a Naive Bayes; **no los supera**. Con la definición de superioridad de la guía, el veredicto es de
no superioridad, y la app lo dice.

## Secciones

| Sección | Ruta | Qué tiene |
|---|---|---|
| **Contexto** | `/` | El problema y por qué importa, la pregunta y los objetivos, la formulación, los datos (fuente, licencia, diccionario de variables), el pipeline de la metodología, las limitaciones, la ética, las conclusiones y los autores |
| **EDA** | `/eda` | Análisis exploratorio del entrenamiento con filtros globales: objetivo y balance, distribuciones, explorador bivariado, correlaciones, faltantes y atípicos, y la variable que generó el puntaje |
| **Modelos** | `/modelos/<vista>` | Doce vistas en dos grupos |

Vistas de Modelos:

| Grupo | Vista | Qué tiene |
|---|---|---|
| Los ocho modelos | Tabla de líderes | La mejor combinación de cada modelo (balanceo, optimizador, métrica media ± desviación del bucle externo, tiempo de inferencia), ordenable y filtrable |
| | Predicción en vivo | Formulario generado del esquema de variables, selector de modelo, predicción con probabilidad recalibrada o intervalo conforme y su cobertura, y explicación SHAP local |
| | Diagnóstico | Matriz de confusión, ROC, precisión-recall y calibración; o reales frente a predichos, residuos, ACF e histograma |
| | Interpretabilidad | Resumen SHAP global del modelo elegido, comparado con la EBM |
| | Comparación estadística | Diagrama de diferencia crítica (Friedman y Nemenyi) o tabla del MCS, y la EBM frente a cada referencia |
| El modelo propuesto | Algoritmo y veredicto | Formulación, pseudocódigo, sustento teórico, complejidad, diferencias con la literatura, pruebas con p ajustados, robustez y veredicto |
| | Cómo decide | Las curvas de forma y la importancia de la EBM, leídas del modelo en vivo |
| | Un cliente | La explicación exacta de un cliente y el simulador |
| | Campaña | A quién contactar si la capacidad no alcanza para todos |
| | Robustez | Ablación, optimizadores, sensibilidad, costo y curva de aprendizaje |
| | Errores | Dónde se equivoca y por qué |
| | Ficha del modelo | La *model card* (también en `MODEL_CARD.md`) |

Cada vista responde una pregunta con cifras leídas de los datos; el botón «…» de cada tarjeta abre la tabla con los datos
del gráfico y dice de qué archivos salen. Al pie de cada una, «Siguiente» lleva por el recorrido de la sustentación.
Las direcciones de la versión anterior (`/cliente`, `/comparacion`…) redirigen a su vista.

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
2. Revisar los registros de la construcción y abrir la URL pública; comprobar las tres secciones y las doce vistas.

El plan gratuito tiene 512 MB y 0,1 CPU, y se suspende tras un rato sin visitas: el primer acceso tarda en responder
(abrir la URL unos minutos antes de la sustentación). La app nunca entrena ni optimiza: solo carga artefactos. Al arrancar,
antes de atender visitas, carga los 14 modelos y arma cada vista una vez (`FUGA_PRECALENTAR=1`; unos 25 s en Render),
así la primera visita no espera. Lo hace en el hilo principal y no en uno aparte: en Render, importar el lector de parquet
en dos hilos a la vez dejó rotas todas las lecturas de parquet.

El `Dockerfile` reproduce ese servidor (Linux, Python 3.13.3, `requirements.txt`, el mismo `gunicorn`). Con los límites del
plan gratuito (`docker run --memory=512m --cpus=0.1 -p 8060:8060 fuga-dash`), el 15-nov-2026: el precalentamiento tardó
40 s; con los 14 modelos cargados la memoria quedó en 379 MiB y el pico, recorriendo las tres secciones y las doce vistas,
fue de 428 MiB, sin reinicios ni errores de consola; cada vista tardó entre 4 y 11 s en cargar y la predicción con k-NN
unos 20 s (su explicación evalúa cientos de consultas sobre los 38.978 clientes). Para la sustentación, la predicción con
la EBM o XGBoost responde en pocos segundos; con el plan Starter (0,5 CPU) todo es unas cinco veces más rápido.

Para caber en 512 MB: XGBoost sin CUDA, sin la librería `shap` en el servidor y el Random Forest de clasificación en una
representación compacta (`bosque.py`: sus mismos 300 árboles en arreglos de numpy, 16 MB en memoria en lugar de 85, con
probabilidades idénticas al artefacto en los 9.745 clientes de prueba). Si aun así Render reiniciara por memoria, la
variable `FUGA_CACHE_MODELOS` limita cuántos modelos guarda.

## Cómo funciona

**Los modelos.** `artifacts/<tarea>/<modelo>/` guarda, para cada uno de los 8 modelos de cada tarea, la tubería final
(preprocesado + modelo) de su mejor combinación, reentrenada con todo el entrenamiento, y su `metadata.json`: versiones de
las librerías, semilla, hiperparámetros, esquema de entrada (nombres, tipos y rangos válidos), métricas del bucle externo
y de la prueba, tiempos y el resultado de la **prueba de ida y vuelta** (el artefacto cargado de disco da exactamente las
mismas predicciones que el modelo en memoria sobre los 9.745 clientes de prueba). `artifacts/indice.json` lista los 16
con su huella SHA-256. `modelos_vivos.py` los carga y los evalúa (el Random Forest de clasificación, desde su forma
compacta `bosque.npz`, comprobada idéntica).

**La EBM.** Una EBM sin interacciones es una suma de tablas: para cada variable, unos cortes y un puntaje por tramo.
`motor.py` evalúa las tablas exportadas del modelo final (78 tablas en `datos/modelos/ebm_*.json`) con numpy, da lo mismo
que el modelo guardado (diferencias del orden de 10⁻¹⁶) y descompone cada puntaje en el aporte exacto de cada variable.

**La incertidumbre.** En clasificación, una probabilidad recalibrada (regresión isotónica ajustada con las predicciones
fuera de pliegue del entrenamiento) y un conjunto conforme de clases; en regresión, un intervalo conforme a partir de los
residuos fuera de pliegue. La cobertura de cada uno se mide en la prueba y se muestra junto a la nominal.

**Las explicaciones.** `explicacion.py` calcula valores SHAP por variable original. En vivo son exactos en los modelos
lineales, Naive Bayes, XGBoost y la EBM, y se estiman por permutaciones antitéticas (con su error de Monte Carlo) en el
árbol, Random Forest y k-NN: la librería `shap` no cabe en el servidor gratuito. El resumen global, calculado fuera de
línea, es exacto también en el árbol y Random Forest (TreeSHAP). Cada explicación cumple base + suma de aportes = salida
del modelo, y se verificaron contra la librería `shap`.

## De dónde salen los datos

Se arman con dos guiones a partir de la carpeta de trabajo del proyecto (el Entregable 3, con los modelos guardados, y la
sección de resultados del informe), que solo se leen:

```
"..\entrega 3\.venv\Scripts\python.exe" -B preparar_datos.py         (datos/: clientes, resultados, EDA, la EBM en JSON)
"..\entrega 3\.venv\Scripts\python.exe" -B preparar_artefactos.py    (artifacts/, calibración, intervalos, SHAP, Nemenyi)
```

`datos/origen.json` y `artifacts/indice.json` anotan, para cada archivo, de dónde salió, cómo y su huella. Ninguna cifra
de la app está escrita a mano: todas se leen de `datos/` o de `artifacts/`, o las calculan los modelos.

## Cómo se comprueba

| Comando | Qué comprueba |
|---|---|
| `python -m pytest -q` | Artefactos completos, ida y vuelta de los 16 modelos, motor de la EBM igual a su artefacto, propiedades de las explicaciones (eficiencia, coeficiente por desvío, reproducibilidad), cobertura de los conjuntos e intervalos conformes, validación de entradas, rutas y las dos comprobaciones de abajo. Necesita `pip install -r requirements-dev.txt`. Corre en GitHub Actions en cada push que cambie `Dashboard/` (`.github/workflows/dashboard.yml`, en la raíz del repositorio) |
| `python probar_app.py [vista]` | Que cada sección y vista se arma y que todos sus callbacks responden, sin navegador |
| `python verificar_cifras.py` | Que las cifras principales de cada vista son las de los archivos de `datos/` |
| `python verificar_motor.py` | Que el motor en numpy da lo mismo que el modelo guardado y reproduce las cifras del informe |
| `python verificar_estilo.py [--vivo PUERTO]` | Las reglas del sistema de diseño, en el código y en el navegador (Playwright y Microsoft Edge) |
| `python capturar.py --puerto PUERTO` | Una captura de cada vista a 1366 y a 400 px |

## Sistema de diseño

Página gris claro, tarjetas casi blancas, navegación fija arriba con la sección activa en píldora negra, títulos grandes
en peso normal y barras rayadas con la serie en foco sólida. Letra Inter, incluida en `assets/fuentes/` (licencia SIL
OFL 1.1). El color solo codifica significado y es el mismo en todas las vistas (cinco roles): azul para la EBM, rosa para
los modelos de árboles con los que empata, gris para los que supera, casi negro para lo observado y gris claro para el
contexto. `DISENO.md` lo explica entero y `CONTRATO.md` es la guía para escribir o cambiar una vista.

## Archivos

```
app.py                  entrada (python app.py; en Render, gunicorn app:server): secciones, vistas, redirecciones
paginas/                las tres secciones: contexto.py, eda.py, modelos.py (y la página de dirección desconocida)
secciones/              las vistas de Modelos y las piezas que reutilizan Contexto y EDA
modelos_vivos.py        los 16 modelos en vivo desde artifacts/: predicción, incertidumbre, explicación
bosque.py               el Random Forest de clasificación en arreglos de numpy (idéntico, cabe en 512 MB)
explicacion.py          valores SHAP por variable original, con el método de cada familia de modelos
motor.py                la EBM en numpy: puntuar, explicar, simular, curvas, campaña
datos_app.py, formato.py, ui.py, tokens.py, assets/      lectura de datos, números en es-CO y sistema de diseño
artifacts/              las 16 tuberías finales con su metadata.json
datos/                  lo que la app lee; origen.json dice de dónde salió cada archivo
tests/                  pruebas (pytest)
preparar_datos.py, preparar_artefactos.py               arman datos/ y artifacts/ desde el proyecto
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
