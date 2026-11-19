# Predicción de fuga de clientes en una fintech colombiana — Proyecto de Machine Learning

**Universidad del Norte — Pregrado en Ciencia de Datos — Machine Learning**

Martínez Pulido, Valerie · Basto Martínez, Abrahan · Esguerra Fernández, Rubén

Las tres entregas del proyecto sobre COFINFAD, un conjunto público de 48.723 clientes de una *fintech* colombiana: el
análisis exploratorio (Entrega 1), los modelos lineales de referencia (Entrega 2) y el entregable final (Entrega 3): un
pipeline combinatorio de 140 corridas con cuatro métodos de optimización de hiperparámetros y validación cruzada anidada,
optimización computacional, evaluación, interpretabilidad, comparación estadística y un modelo nuevo, la *Explainable
Boosting Machine*, que supera a los modelos de la Entrega 2. El informe final (`InformeFinal/`) reúne la metodología y
los resultados de las tres entregas, escritos con las guías del curso, con su introducción y sus conclusiones. El
dashboard predictivo (`Dashboard/`) pone a predecir en vivo a los modelos finales y se despliega en Render.

El **proyecto final** añade un modelo original programado desde cero, **BDL (Boosting con Desplazamiento Latente)**
(`ModeloOriginal/`), su comparación con nueve benchmarks bajo un protocolo registrado antes de los experimentos, y el
informe técnico en forma de Jupyter Book (`ProyectoFinal/`, capítulos 0 a 10 de la guía). La EBM, que viene de una
librería, pasa a ser un benchmark.

## Sitio

El Jupyter Book con el informe del proyecto final, las tres entregas y todo el código:
**https://rubenesg.github.io/Prediccion-de-fuga-de-clientes-en-fintech/**

Sin conexión, se abre igual desde [`docs/index.html`](docs/index.html).

El dashboard predictivo, desplegado en Render: **https://fuga-de-clientes-dash.onrender.com** (plan gratuito: si lleva un rato sin visitas, la primera
carga tarda cerca de un minuto).

## Entregables del proyecto final

Cada entregable de la guía (§10) y dónde está en este repositorio:

| N.º | Entregable | Dónde está |
|---|---|---|
| 1 | URL pública del dashboard | <https://fuga-de-clientes-dash.onrender.com> |
| 2 | Repositorio con README, versiones exactas, semilla, licencia y forma de citar | Este README; `Dashboard/requirements.txt`, `Entrega3/requirements.txt` y `ModeloOriginal/requirements-dev.txt`; semilla global 42 (capítulo 10 del libro); [`LICENSE`](LICENSE); [`CITATION.cff`](CITATION.cff) |
| 3 | Jupyter Book: fuentes y versión publicada | [`ProyectoFinal/`](ProyectoFinal/) (capítulos 0 a 10 como cuadernos ejecutados, con sus fuentes), `_config.yml`, `_toc.yml` y `ProyectoFinal/references.bib`; publicado en GitHub Pages desde `docs/`; en PDF, `ProyectoFinal/ProyectoFinal_informe.pdf` |
| 4 | Código del modelo original, con *docstrings* y pruebas en integración continua | [`ModeloOriginal/src/original_model.py`](ModeloOriginal/src/original_model.py) (el dashboard usa una copia idéntica en `Dashboard/src/`), `ModeloOriginal/tests/` y `.github/workflows/modelo_original.yml` |
| 5 | Protocolo experimental pre-registrado | [`ModeloOriginal/experiments/protocolo.md`](ModeloOriginal/experiments/protocolo.md), etiqueta de Git `protocolo-modelo-original-v1` |
| 6 | Tabla maestra de experimentos | `ModeloOriginal/resultados/experimentos.parquet` (modelo original, LightGBM y la mejor combinación de cada benchmark, tres repeticiones), `ModeloOriginal/resultados/adicionales/<conjunto>/experimentos.parquet` (conjuntos adicionales) y `Entrega3/resultados/experimentos.parquet` (las 140 combinaciones de la guía y las de la EBM) |
| 7 | Artefactos de los modelos con `metadata.json` | `Dashboard/artifacts/` |
| 8 | Ficha del modelo y declaración de los datos | [`Dashboard/MODEL_CARD.md`](Dashboard/MODEL_CARD.md) y [`Dashboard/DATA_STATEMENT.md`](Dashboard/DATA_STATEMENT.md) |
| 9 | Archivo del repositorio con identificador persistente | Zenodo, DOI [10.5281/zenodo.24356311](https://doi.org/10.5281/zenodo.24356311): la *release* `v2.0.0`, la versión entregada (capítulo 10 del libro y [`CITATION.cff`](CITATION.cff)) |
| 10 | Declaración de autoría y uso de herramientas | Capítulo 9 del libro, sección 9.6 |

El repositorio reúne las tres entregas y el proyecto final, así que la estructura que propone la guía (§7.2) está
repartida por carpetas: `app.py`, `pages/` (aquí `paginas/`), `assets/`, `artifacts/`, `data/` (aquí `datos/`),
`MODEL_CARD.md`, `Dockerfile` y `requirements.txt` están en `Dashboard/`; `src/original_model.py`, `experiments/` y
`tests/`, en `ModeloOriginal/`; las fuentes del libro, en `ProyectoFinal/` y en la raíz; y `render.yaml`, `LICENSE`,
`CITATION.cff` y `.github/workflows/`, en la raíz.

## Contenido

| Carpeta | Contenido |
|---|---|
| `ProyectoFinal/` | El informe técnico del proyecto final (más abajo): los capítulos 0 a 10 del Jupyter Book, sus figuras, sus tablas generadas y la bibliografía |
| `ModeloOriginal/` | El modelo original, BDL (más abajo): el algoritmo, sus pruebas unitarias, el protocolo pre-registrado, los guiones de los experimentos y sus resultados |
| `Entrega1/` | `eda_bancario.ipynb`, el cuaderno ejecutado del análisis exploratorio, y su artículo `Entrega1_EDA_articulo.pdf` |
| `Entrega2/` | `entrega2_modelo.ipynb`, el cuaderno ejecutado de los modelos base, y su artículo `Entrega2_articulo.pdf` |
| `Entrega3/` | El entregable final (tabla siguiente) |
| `InformeFinal/` | El informe final del proyecto (más abajo): las secciones de metodología y de resultados escritas con las guías del curso y el informe que las une |
| `Dashboard/` | El dashboard predictivo en Plotly Dash (más abajo): Contexto, EDA y Modelos, con los modelos finales en vivo, incluido el modelo original |
| `render.yaml`, `.github/workflows/` | El despliegue del dashboard en Render (Blueprint) y sus pruebas en GitHub Actions |
| `datos/` | COFINFAD comprimido (`customer_data.zip` y `transactions_data.zip`) y cómo usarlo |
| `codigo/` | Páginas del Jupyter Book que muestran el código fuente completo |
| `docs/` | El Jupyter Book construido, que sirve GitHub Pages |
| `index.md`, `_config.yml`, `_toc.yml`, `_static/` | Portada y configuración del Jupyter Book |

### Proyecto final: modelo original y evaluación

| Archivo o carpeta | Contenido |
|---|---|
| [`ModeloOriginal/src/original_model.py`](ModeloOriginal/src/original_model.py) | **El modelo original**: `RegresorBDL`, `ClasificadorBDL` y el alias `ModeloOriginal`. Árboles por histogramas, EM de la mezcla, probabilidad de excedencia y conjuntos de predicción conformes, con `numpy` y `scipy`; no importa ningún modelo |
| `ModeloOriginal/tests/` | Pruebas unitarias (`python -m pytest -q`): solución conocida, semilla, entradas inválidas, `clone`, `Pipeline` y las propiedades demostradas. Se ejecutan en cada *push* (`.github/workflows/modelo_original.yml`) |
| [`ModeloOriginal/experiments/protocolo.md`](ModeloOriginal/experiments/protocolo.md) | Protocolo experimental registrado antes de los experimentos definitivos (etiqueta `protocolo-modelo-original-v1`) |
| `ModeloOriginal/experiments/` | Guiones numerados que producen cada resultado: búsqueda de literatura, auditoría de fuga, validación anidada, prueba, pruebas de hipótesis, conjuntos adicionales, robustez, ablación y tablas del libro |
| `ModeloOriginal/resultados/` | Tabla maestra `experimentos.parquet`, predicciones y métricas de prueba, pruebas de hipótesis, veredicto, techo teórico, ablación, robustez y conjuntos adicionales |
| `ProyectoFinal/` | Los capítulos del informe como cuadernos ejecutados (`00_introduccion.ipynb` a `10_reproducibilidad.ipynb`), sus fuentes en `fuentes/`, `construir.py` (los genera y ejecuta), `exportar_pdf.py`, `references.bib`, `figuras/` y `_generado/` (tablas y textos que salen de los resultados) |

[`ModeloOriginal/README.md`](ModeloOriginal/README.md) explica la idea del modelo y cómo usarlo.

### Entrega 3

| Archivo o carpeta | Contenido |
|---|---|
| `Entregable3_articulo.pdf` | El artículo en la plantilla Springer (sn-jnl); fuente LaTeX, tablas y figuras generadas en `articulo/` |
| `Entregable3.ipynb` | **Un solo notebook, ejecutado**, con las once secciones de esta entrega |
| `Entregable3_notebook.pdf` | El notebook de esta entrega impreso en PDF, con un marcador por sección |
| `notebook/` | Las once secciones como cuadernos ejecutados (tabla siguiente); `fuentes/` y `construir.py` los generan y ejecutan; `exportar_pdf.py` las imprime a PDF |
| `src/` | Paquete con la lógica, separada por módulos: datos, preprocesamiento, modelos, balanceo, optimización, experimento, registro, evaluación, estadística, comparación, análisis y gráficas |
| `tests/` | Pruebas automáticas (`python -m pytest tests`) |
| `resultados/` | Tabla maestra `experimentos.parquet` (una fila por corrida: hiperparámetros finales, métricas del bucle externo, tiempos y semilla), predicciones de prueba, comparación con la Entrega 2, experimentos de cómputo y complementos |
| `correr_140.py`, `comparar_entrega2.py`, `computo.py`, `complementos.py`, `medir_tiempos_finales.py` | Guiones que producen los resultados |
| `requirements.txt` | Versiones exactas del entorno (Python 3.13.3) |

| Sección del notebook | Contenido |
|---|---|
| [`Entrega3/notebook/00_introduccion.ipynb`](Entrega3/notebook/00_introduccion.ipynb) | Resumen, organización del libro y cómo reproducir |
| [`Entrega3/notebook/01_datos.ipynb`](Entrega3/notebook/01_datos.ipynb) | Data: ETL, partición antes de mirar el objetivo, etiqueta, EDA y correcciones de la Entrega 1 |
| [`Entrega3/notebook/02_metodologia.ipynb`](Entrega3/notebook/02_metodologia.ipynb) | Pipeline, modelos y espacios de búsqueda, balanceo, validación anidada y los cuatro optimizadores |
| [`Entrega3/notebook/03_corridas.ipynb`](Entrega3/notebook/03_corridas.ipynb) | Las 140 corridas: tabla maestra, mejor combinación por modelo y efecto del balanceo |
| [`Entrega3/notebook/04_optimizadores.ipynb`](Entrega3/notebook/04_optimizadores.ipynb) | Comparación de Grid, Random, Bayesiana y Genética: métrica externa, tiempo y curvas anytime |
| [`Entrega3/notebook/05_computo.ipynb`](Entrega3/notebook/05_computo.ipynb) | Optimización computacional: complejidad, variantes eficientes, paralelismo y tabla estándar frente a optimizado |
| [`Entrega3/notebook/06_evaluacion.ipynb`](Entrega3/notebook/06_evaluacion.ipynb) | Evaluación en prueba, calibración, residuos y robustez |
| [`Entrega3/notebook/07_interpretabilidad.ipynb`](Entrega3/notebook/07_interpretabilidad.ipynb) | SHAP, LIME y las funciones de forma de la EBM |
| [`Entrega3/notebook/08_modelo_nuevo.ipynb`](Entrega3/notebook/08_modelo_nuevo.ipynb) | El modelo nuevo (EBM) frente a la Entrega 2, con DeLong y Diebold-Mariano |
| [`Entrega3/notebook/09_estadistica.ipynb`](Entrega3/notebook/09_estadistica.ipynb) | Friedman, Nemenyi, DeLong, MCS, SPA, Giacomini-White y Diebold-Mariano |
| [`Entrega3/notebook/10_conclusiones.ipynb`](Entrega3/notebook/10_conclusiones.ipynb) | Conclusiones |

### Informe final

Es anterior al proyecto final: reúne las tres entregas y presenta la EBM como el modelo nuevo. El informe del proyecto
final, con el modelo original, es el Jupyter Book de `ProyectoFinal/`.

| Archivo o carpeta | Contenido |
|---|---|
| [`InformeFinal/informe/main.pdf`](InformeFinal/informe/main.pdf) | **El informe final** (31 páginas): resumen, introducción, metodología, resultados, conclusiones y referencias; `main.tex`, `armar.py` y `compilar.py` lo arman a partir de las dos secciones y lo compilan |
| [`InformeFinal/metodologia/main.pdf`](InformeFinal/metodologia/main.pdf) | La sección de metodología según la guía del curso (ocho subsecciones, 20 referencias); fuente LaTeX, `figuras.py` y sus tres figuras |
| [`InformeFinal/resultados/main.pdf`](InformeFinal/resultados/main.pdf) | La sección de resultados según la guía del curso (ocho subsecciones); `generar.py` reproduce cada prueba estadística, tabla y figura desde `Entrega3/resultados/`, con toda cifra en `datos/*.csv` y las nueve figuras en `figures/` |

Cada carpeta tiene su `README.md` con los archivos y cómo regenerarlos.

### Dashboard

[`Dashboard/`](Dashboard/) es la app en Plotly Dash sobre la que se hace la sustentación, en línea en <https://fuga-de-clientes-dash.onrender.com>. Tiene tres secciones:
**Contexto** (problema, pregunta, formulación, datos, pipeline, limitaciones y ética), **EDA** (con filtros globales,
solo sobre el entrenamiento) y **Modelos**. Modelos tiene trece vistas en tres grupos: los diez modelos (tabla de
líderes, predicción en vivo con incertidumbre y explicación local, diagnóstico, interpretabilidad y comparación
estadística); el modelo original, BDL (su subsección, con la descripción, el pseudocódigo, las proposiciones, las
diferencias con la literatura, la convergencia, las pruebas de hipótesis, la robustez y el veredicto, y la ficha del
modelo); y la EBM como benchmark interpretable (seis vistas). Carga las 18 tuberías finales de `Dashboard/artifacts/`
(cada una con su `metadata.json` y su prueba de ida y vuelta) y nunca entrena.

```bash
cd Dashboard
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # en Linux o macOS: .venv/bin/pip
.venv/Scripts/python app.py                        # http://127.0.0.1:8060
```

En Windows basta con doble clic en `Dashboard/iniciar_dash.bat`. Se despliega en Render con `render.yaml` (New →
Blueprint, este repositorio). [`Dashboard/README.md`](Dashboard/README.md) explica el despliegue, cómo funciona y cómo se
comprueba. [`MODEL_CARD.md`](Dashboard/MODEL_CARD.md) es la ficha del modelo y
[`DATA_STATEMENT.md`](Dashboard/DATA_STATEMENT.md) la declaración de los datos.

## Resultados principales

### Proyecto final

<!-- resultados-proyecto-final:inicio -->
- **Modelo original: BDL (Boosting con Desplazamiento Latente)**, programado desde cero con `numpy` y `scipy`. Modela como una clase latente el escalón que deja en el puntaje la variable excluida por fuga de información y lo aprende con un EM generalizado cuyo paso M es un paso de boosting.
- **Regresión** (RMSE): en los 15 pliegues externos de la validación anidada, BDL 0,05887 y el mejor benchmark (XGBoost) 0,05897; en prueba, 0,05912 frente a 0,05917, con un límite teórico de 0,05911. 2 de las 4 pruebas de hipótesis registradas rechazan la igualdad a favor de BDL (p ajustado por Holm < 0,05): **no superior** según la guía, que pide tres.
- **Clasificación** (AUC): en los 15 pliegues externos de la validación anidada, BDL 0,8313 y el mejor benchmark (XGBoost) 0,8333; en prueba, 0,8311 frente a 0,8283, con un límite teórico de 0,8313. Ninguna de las 4 pruebas de hipótesis registradas rechaza la igualdad a favor de BDL (p ajustado por Holm < 0,05): **no superior** según la guía, que pide tres.
- **Incertidumbre:** con cobertura del 90 %, los conjuntos de predicción de BDL miden 0,0062 y los intervalos conformes del mejor benchmark, 0,1905; la cobertura medida en prueba es 89,9 %.
- **Conjuntos adicionales:** En los 4 conjuntos de datos adicionales, el original no supera al mejor benchmark en ninguno, ni en regresión ni en clasificación; en regresión es peor de forma significativa en Vivienda de California y en clasificación es peor de forma significativa en Seguros médicos y Vivienda de California (sección 7.1). En 4 de ellos la búsqueda de hiperparámetros eligió para la regresión la variante sin mezcla, incluidos los construidos con una variable omitida: la ventaja que el protocolo esperaba ahí no apareció.
- **Recencia:** la fecha de corte se calcula solo con los clientes de entrenamiento; la auditoría muestra que no cambia ningún dato.
<!-- resultados-proyecto-final:fin -->

### Entregas anteriores

- **Pipeline de la guía**: 140 corridas (136 entrenadas y 4 que no aplican), con validación cruzada anidada 5 × 3. Mejor modelo de clasificación: XGBoost (AUC externa 0,8332); de regresión: XGBoost (RMSE externo 0,0590).
- **Modelo nuevo (EBM) frente a la Entrega 2**, en la misma prueba y con la misma etiqueta: AUC 0,6832 → 0,7187 (DeLong p < 10⁻¹⁸) y R² 0,1511 → 0,2159 (Diebold-Mariano con HLN p < 10⁻³⁹). Empata con XGBoost (p = 0,196 en AUC) y queda cerca del techo de información (0,7223 de AUC y 0,2197 de R²).

Todas las cifras del libro y de este README se generan a partir de los resultados; ninguna está escrita a mano. El
bloque del proyecto final lo escribe `ModeloOriginal/experiments/08_tablas_libro.py`.

## Datos

COFINFAD — *Colombian Fintech Financial Analytics Dataset*, [Mendeley Data](https://data.mendeley.com/datasets/mhb4zn3258/1)
(DOI: 10.17632/mhb4zn3258.1), licencia CC BY 4.0. Está en `datos/` comprimido, porque `transactions_data.csv` pesa 114 MB y
GitHub no admite archivos de más de 100 MB; el código lo descomprime solo la primera vez que lo necesita.

## Compilar el Jupyter Book

El Jupyter Book se construye con `execute_notebooks: "off"`: usa las salidas ya guardadas en los cuadernos, así que no vuelve a
entrenar nada.

```bash
pip install -r Entrega3/requirements.txt
jupyter-book build .
rm -rf docs && cp -r _build/html docs && touch docs/.nojekyll
```

## Reproducir el proyecto final

Con el entorno de la Entrega 3 más LightGBM (`pip install -r ModeloOriginal/requirements-dev.txt`), desde `ModeloOriginal/`:

```bash
python -m pytest -q                                   # pruebas del modelo, del marco y de las pruebas de hipótesis
python experiments/00_busqueda_literatura.py          # matriz de novedad (consulta OpenAlex, arXiv y Crossref)
python experiments/01_auditoria_fuga.py               # auditoría de fuga: la recencia corregida no cambia ningún dato
python experiments/correr_todo.py                     # validación anidada, prueba, pruebas de hipótesis, semillas y conjuntos adicionales
python experiments/correr_todo.py --cola b            # ablación y robustez (se puede lanzar en otra terminal)
python experiments/09_rutas_relativas.py              # deja relativas al repositorio las rutas de las tablas maestras
python experiments/08_tablas_libro.py                 # tablas, figuras y textos del libro, y el bloque de resultados de este README
cd .. && python ProyectoFinal/construir.py            # ejecuta los capítulos del proyecto final y guarda sus salidas
```

Cada paso salta lo que ya está registrado, así que se puede interrumpir y relanzar. El capítulo 10 del libro detalla
qué produce cada guion. De los modelos serializados solo se versionan los del modelo original (menos de 1 MB), que son
los que leen las celdas del libro; los de los benchmarks se regeneran con los guiones.

## Entorno y reproducción de los experimentos de la Entrega 3

Python 3.13.3 · numpy 2.5.3 · pandas 3.0.6 · scikit-learn 1.9.1 · imbalanced-learn 0.14.2 · xgboost 3.4.1 · interpret-core 0.7.8 · optuna 5.0.0 · deap 1.4.4 · shap 0.52.0 · jupyter-book 1.0.4.post1 · semilla única 42, propagada a numpy,
scikit-learn, XGBoost, Optuna y DEAP. Tiempos medidos en un equipo de 6 núcleos.

```bash
cd Entrega3
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -u correr_140.py                         # 140 corridas (unas 27 h; reanudable)
.venv/Scripts/python -u comparar_entrega2.py                  # comparación con la Entrega 2
.venv/Scripts/python -u correr_140.py --plan modelo_nuevo     # 20 corridas de la EBM (unas 19 h)
.venv/Scripts/python -u computo.py                            # experimentos de cómputo (equipo sin otra carga)
.venv/Scripts/python -u complementos.py                       # calibración, semillas, fuera de pliegue
.venv/Scripts/python medir_tiempos_finales.py                 # tiempos de ajuste e inferencia de los modelos finales
.venv/Scripts/python notebook/construir.py                    # ejecuta las secciones del notebook
.venv/Scripts/python articulo/generar_material.py             # tablas y figuras del artículo desde los resultados
.venv/Scripts/python articulo/compilar.py                     # el artículo en PDF (Tectonic o TeX Live)
```

Los modelos serializados (unos 490 MB) no se incluyen: se regeneran con los hiperparámetros finales que guarda la
tabla maestra.
