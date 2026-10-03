# Predicción de fuga de clientes en una fintech colombiana — Proyecto de Machine Learning

**Universidad del Norte — Pregrado en Ciencia de Datos — Machine Learning**

Martínez Pulido, Valerie · Basto Martínez, Abrahan · Esguerra Fernández, Rubén

Las tres entregas del proyecto sobre COFINFAD, un conjunto público de 48.723 clientes de una *fintech* colombiana: el
análisis exploratorio (Entrega 1), los modelos lineales de referencia (Entrega 2) y el entregable final (Entrega 3): un
pipeline combinatorio de 140 corridas con cuatro métodos de optimización de hiperparámetros y validación cruzada anidada,
optimización computacional, evaluación, interpretabilidad, comparación estadística y un modelo nuevo, la *Explainable
Boosting Machine*, que supera a los modelos de la Entrega 2.

## Sitio

El libro con las tres entregas y todo el código se publica como Jupyter Book:
**https://rubenesg.github.io/Prediccion-de-fuga-de-clientes-en-fintech/**

Sin conexión, se abre igual desde [`docs/index.html`](docs/index.html).

## Contenido

| Carpeta | Contenido |
|---|---|
| `Entrega1/` | `eda_bancario.ipynb`, el cuaderno ejecutado del análisis exploratorio, y su artículo `Entrega1_EDA_articulo.pdf` |
| `Entrega2/` | `entrega2_modelo.ipynb`, el cuaderno ejecutado de los modelos base, y su artículo `Entrega2_articulo.pdf` |
| `Entrega3/` | El entregable final (tabla siguiente) |
| `datos/` | COFINFAD comprimido (`customer_data.zip` y `transactions_data.zip`) y cómo usarlo |
| `codigo/` | Páginas del libro que muestran el código fuente completo |
| `docs/` | El libro construido, que sirve GitHub Pages |
| `index.md`, `_config.yml`, `_toc.yml`, `_static/` | Portada y configuración del Jupyter Book |

### Entrega 3

| Archivo o carpeta | Contenido |
|---|---|
| `Entregable3_articulo.pdf` | El artículo en la plantilla Springer (sn-jnl); fuente LaTeX, tablas y figuras generadas en `articulo/` |
| `Entregable3.ipynb` | **Un solo cuaderno, ejecutado**, con las once secciones del libro de esta entrega |
| `Entregable3_libro.pdf` | El libro de esta entrega en PDF, con un marcador por sección |
| `presentacion/` | Diapositivas de la sustentación de 10 minutos (PowerPoint y PDF) |
| `libro/` | Las once secciones como cuadernos ejecutados (tabla siguiente); `fuentes/` y `construir.py` los generan y ejecutan; `exportar_pdf.py` imprime el libro a PDF |
| `src/` | Paquete con la lógica, separada por módulos: datos, preprocesamiento, modelos, balanceo, optimización, experimento, registro, evaluación, estadística, comparación, análisis y gráficas |
| `tests/` | Pruebas automáticas (`python -m pytest tests`) |
| `resultados/` | Tabla maestra `experimentos.parquet` (una fila por corrida: hiperparámetros finales, métricas del bucle externo, tiempos y semilla), predicciones de prueba, comparación con la Entrega 2, experimentos de cómputo y complementos |
| `correr_140.py`, `comparar_entrega2.py`, `computo.py`, `complementos.py`, `medir_tiempos_finales.py` | Guiones que producen los resultados |
| `requirements.txt` | Versiones exactas del entorno (Python 3.13.3) |

| Sección del libro | Contenido |
|---|---|
| [`Entrega3/libro/00_introduccion.ipynb`](Entrega3/libro/00_introduccion.ipynb) | Resumen, organización del libro y cómo reproducir |
| [`Entrega3/libro/01_datos.ipynb`](Entrega3/libro/01_datos.ipynb) | Data: ETL, partición antes de mirar el objetivo, etiqueta, EDA y correcciones de la Entrega 1 |
| [`Entrega3/libro/02_metodologia.ipynb`](Entrega3/libro/02_metodologia.ipynb) | Pipeline, modelos y espacios de búsqueda, balanceo, validación anidada y los cuatro optimizadores |
| [`Entrega3/libro/03_corridas.ipynb`](Entrega3/libro/03_corridas.ipynb) | Las 140 corridas: tabla maestra, mejor combinación por modelo y efecto del balanceo |
| [`Entrega3/libro/04_optimizadores.ipynb`](Entrega3/libro/04_optimizadores.ipynb) | Comparación de Grid, Random, Bayesiana y Genética: métrica externa, tiempo y curvas anytime |
| [`Entrega3/libro/05_computo.ipynb`](Entrega3/libro/05_computo.ipynb) | Optimización computacional: complejidad, variantes eficientes, paralelismo y tabla estándar frente a optimizado |
| [`Entrega3/libro/06_evaluacion.ipynb`](Entrega3/libro/06_evaluacion.ipynb) | Evaluación en prueba, calibración, residuos y robustez |
| [`Entrega3/libro/07_interpretabilidad.ipynb`](Entrega3/libro/07_interpretabilidad.ipynb) | SHAP, LIME y las funciones de forma de la EBM |
| [`Entrega3/libro/08_modelo_nuevo.ipynb`](Entrega3/libro/08_modelo_nuevo.ipynb) | El modelo nuevo (EBM) frente a la Entrega 2, con DeLong y Diebold-Mariano |
| [`Entrega3/libro/09_estadistica.ipynb`](Entrega3/libro/09_estadistica.ipynb) | Friedman, Nemenyi, DeLong, MCS, SPA, Giacomini-White y Diebold-Mariano |
| [`Entrega3/libro/10_conclusiones.ipynb`](Entrega3/libro/10_conclusiones.ipynb) | Conclusiones |

## Resultados principales

- **Pipeline de la guía**: 140 corridas (136 entrenadas y 4 que no aplican), con validación cruzada anidada 5 × 3. Mejor modelo de clasificación: XGBoost (AUC externa 0,8332); de regresión: XGBoost (RMSE externo 0,0590).
- **Modelo nuevo (EBM) frente a la Entrega 2**, en la misma prueba y con la misma etiqueta: AUC 0,6832 → 0,7187 (DeLong p < 10⁻¹⁸) y R² 0,1511 → 0,2159 (Diebold-Mariano con HLN p < 10⁻³⁹). Empata con XGBoost (p = 0,196 en AUC) y queda cerca del techo de información (0,7223 de AUC y 0,2197 de R²).

Todas las cifras del libro y de este README se generan a partir de los resultados; ninguna está escrita a mano.

## Datos

COFINFAD — *Colombian Fintech Financial Analytics Dataset*, [Mendeley Data](https://data.mendeley.com/datasets/mhb4zn3258/1)
(DOI: 10.17632/mhb4zn3258.1), licencia CC BY 4.0. Está en `datos/` comprimido, porque `transactions_data.csv` pesa 114 MB y
GitHub no admite archivos de más de 100 MB; el código lo descomprime solo la primera vez que lo necesita.

## Compilar el libro

El libro se construye con `execute_notebooks: "off"`: usa las salidas ya guardadas en los cuadernos, así que no vuelve a
entrenar nada.

```bash
pip install -r Entrega3/requirements.txt
jupyter-book build .
rm -rf docs && cp -r _build/html docs && touch docs/.nojekyll
```

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
.venv/Scripts/python libro/construir.py                       # ejecuta las páginas del libro
.venv/Scripts/python articulo/generar_material.py             # tablas y figuras del artículo desde los resultados
.venv/Scripts/python articulo/compilar.py                     # el artículo en PDF (Tectonic o TeX Live)
```

Los modelos serializados (unos 490 MB) no se incluyen: se regeneran con los hiperparámetros finales que guarda la
tabla maestra.
