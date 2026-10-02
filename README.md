# Predicción de fuga de clientes en una *fintech* colombiana — Entregable 3

**Universidad del Norte — Pregrado en Ciencia de Datos — Machine Learning**

Martínez Pulido, Valerie · Basto Martínez, Abrahan · Esguerra Fernández, Rubén

Pipeline combinatorio de modelos de clasificación y regresión con balanceo, cuatro métodos de
optimización de hiperparámetros y validación cruzada anidada; optimización computacional, evaluación,
interpretabilidad y comparación estadística; y un modelo nuevo, la *Explainable Boosting Machine*,
comparado con la Entrega 2. Continúa el [análisis exploratorio](https://rubenesg.github.io/Machine_learning/) de la Entrega 1 y los
[modelos base](https://rubenesg.github.io/-MACHINE-LEARNING-Entregable-2/) de la Entrega 2.

## Sitio

El libro se publica como Jupyter Book, con todo el código, sus salidas y la interpretación:
**https://rubenesg.github.io/predicci-n-de-fuga-de-clientes-en-fintech/**

Sin conexión, se abre igual desde [`docs/index.html`](docs/index.html).

## Contenido

| Archivo o carpeta | Descripción |
|---|---|
| `libro/` | Los 11 cuadernos del libro, **ejecutados de principio a fin**, uno por sección (tabla siguiente); `fuentes/` y `construir.py` los generan y ejecutan |
| `src/` | Paquete con la lógica, separada por módulos: datos, preprocesamiento, modelos, balanceo, optimización, experimento, registro, evaluación, estadística, comparación, análisis y gráficas |
| `tests/` | Pruebas automáticas (`python -m pytest tests`) |
| `resultados/experimentos.parquet` | Tabla maestra: una fila por corrida con modelo, balanceo, optimizador, hiperparámetros finales, métricas del bucle externo, tiempos y semilla |
| `resultados/` | Predicciones de prueba, comparación con la Entrega 2, experimentos de cómputo y complementos (calibración, semillas, fuera de pliegue) |
| `exploracion/` | Benchmark de los candidatos a modelo nuevo |
| `correr_140.py`, `comparar_entrega2.py`, `computo.py`, `complementos.py` | Guiones que producen los resultados; `correr_*.bat` los lanzan en Windows con prioridad baja y se pueden reanudar |
| `presentacion/` | Diapositivas de la sustentación de 10 minutos (PowerPoint y PDF) y su guion |
| `_config.yml`, `_toc.yml`, `_static/` | Configuración del Jupyter Book |
| `requirements.txt` | Versiones exactas del entorno (Python 3.13.3) |
| `docs/` | Sitio compilado que sirve GitHub Pages |

### Las secciones del libro

| Cuaderno | Contenido |
|---|---|
| [`libro/00_introduccion.ipynb`](libro/00_introduccion.ipynb) | Resumen, organización del libro y cómo reproducir |
| [`libro/01_datos.ipynb`](libro/01_datos.ipynb) | Data: ETL, partición antes de mirar el objetivo, etiqueta, EDA y correcciones de la Entrega 1 |
| [`libro/02_metodologia.ipynb`](libro/02_metodologia.ipynb) | Pipeline, modelos y espacios de búsqueda, balanceo, validación anidada y los cuatro optimizadores |
| [`libro/03_corridas.ipynb`](libro/03_corridas.ipynb) | Las 140 corridas: tabla maestra, mejor combinación por modelo y efecto del balanceo |
| [`libro/04_optimizadores.ipynb`](libro/04_optimizadores.ipynb) | Comparación de Grid, Random, Bayesiana y Genética: métrica externa, tiempo y curvas anytime |
| [`libro/05_computo.ipynb`](libro/05_computo.ipynb) | Optimización computacional: complejidad, variantes eficientes, paralelismo y tabla estándar frente a optimizado |
| [`libro/06_evaluacion.ipynb`](libro/06_evaluacion.ipynb) | Evaluación en prueba, calibración, residuos y robustez |
| [`libro/07_interpretabilidad.ipynb`](libro/07_interpretabilidad.ipynb) | SHAP, LIME y las funciones de forma de la EBM |
| [`libro/08_modelo_nuevo.ipynb`](libro/08_modelo_nuevo.ipynb) | El modelo nuevo (EBM) frente a la Entrega 2, con DeLong y Diebold-Mariano |
| [`libro/09_estadistica.ipynb`](libro/09_estadistica.ipynb) | Friedman, Nemenyi, DeLong, MCS, SPA, Giacomini-White y Diebold-Mariano |
| [`libro/10_conclusiones.ipynb`](libro/10_conclusiones.ipynb) | Conclusiones |

## Resultados principales

- **Pipeline de la guía**: 140 corridas (136 entrenadas y 4 que no aplican), con validación cruzada anidada 5 × 3. Mejor modelo de clasificación: XGBoost (AUC externa 0,8332); de regresión: XGBoost (RMSE externo 0,0590).
- **Modelo nuevo (EBM) frente a la Entrega 2**, en la misma prueba y con la misma etiqueta: AUC 0,6832 → 0,7187 (DeLong p < 10⁻¹⁸) y R² 0,1511 → 0,2159 (Diebold-Mariano con HLN p < 10⁻³⁹). Empata con XGBoost (p = 0,196 en AUC) y queda cerca del techo de información (0,7223 de AUC y 0,2197 de R²).

Todas las cifras del libro y de este README se generan a partir de los resultados; ninguna está escrita a mano.

## Compilar el libro

El libro se construye con `execute_notebooks: "off"`, es decir, **usa las salidas ya guardadas en los
cuadernos**: se compila sin los datos y sin volver a entrenar nada.

```bash
pip install -r requirements.txt
jupyter-book build .
rm -rf docs && cp -r _build/html docs && touch docs/.nojekyll
```

Para volver a ejecutar los cuadernos (hacen falta `datos/` y `resultados/`): `python libro/construir.py`,
o `python libro/construir.py 08` para una sola página.

## Datos

El conjunto **no está en este repositorio** porque es público y pesa 136 MB
(`transactions_data.csv` supera el límite de 100 MB por archivo de GitHub).

Se trata de COFINFAD — *Colombian Fintech Financial Analytics Dataset*, disponible en
[Mendeley Data](https://data.mendeley.com/datasets/mhb4zn3258/1) (DOI: 10.17632/mhb4zn3258.1), bajo
licencia CC BY 4.0. Para reproducir los experimentos, descargue `customer_data.csv` y
`transactions_data.csv` y ubíquelos en una carpeta `datos/` en la raíz del repositorio:

```
.
├── correr_140.py
├── libro/
└── datos/
    ├── customer_data.csv
    └── transactions_data.csv
```

## Entorno y reproducción de los experimentos

Python 3.13.3 · numpy 2.5.3 · pandas 3.0.6 · scikit-learn 1.9.1 · imbalanced-learn 0.14.2 · xgboost 3.4.1 · interpret-core 0.7.8 · optuna 5.0.0 · deap 1.4.4 · shap 0.52.0 · jupyter-book 1.0.4.post1 · semilla única 42, propagada a numpy,
scikit-learn, XGBoost, Optuna y DEAP. Tiempos medidos en un equipo de 6 núcleos.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -u correr_140.py                         # 140 corridas (unas 27 h; reanudable)
.venv/Scripts/python -u comparar_entrega2.py                  # comparación con la Entrega 2
.venv/Scripts/python -u correr_140.py --plan modelo_nuevo     # 20 corridas de la EBM (unas 19 h)
.venv/Scripts/python -u computo.py                            # experimentos de cómputo (equipo sin otra carga)
.venv/Scripts/python -u complementos.py                       # calibración, semillas, fuera de pliegue
.venv/Scripts/python libro/construir.py                       # ejecuta los cuadernos del libro
.venv/Scripts/jupyter-book build .                            # genera el HTML
```

Los modelos serializados (unos 490 MB) no se incluyen: se regeneran con los hiperparámetros finales
que guarda la tabla maestra.
