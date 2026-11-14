# Sección de resultados del proyecto (Entregables 1 a 3)

Sección «Resultados» del proyecto de Machine Learning (predicción de fuga de clientes con COFINFAD), escrita
según la guía del curso *Cómo escribir la sección de resultados* (ocho subsecciones con presupuesto de palabras,
pruebas de significancia con corrección por comparaciones múltiples, tamaños de efecto, ablación, sensibilidad,
costo computacional, interpretabilidad y análisis de errores, figuras en PDF vectorial con texto en inglés).
Acompaña a la sección de metodología de `../Metodologia Entregable 3/` y usa su misma notación.

Describe lo entregado en el Entregable 3 (la EBM como modelo propuesto frente a los modelos del curso) a partir
de sus resultados guardados; esta carpeta está fuera de los repositorios del proyecto y **no modifica nada de
`entrega 3/`**: solo lee `resultados/`, `datos/` y `src/`.

| Archivo | Qué es |
|---|---|
| `main.pdf` | El documento compilado (portada, la sección con sus ocho subsecciones y las referencias). |
| `main.tex` | Preámbulo (el mismo de la sección de metodología) e inclusión de la sección. |
| `resultados.tex` | La sección: texto y la inclusión de tablas y figuras. Es lo que se copia al informe completo. |
| `generar.py` | **Único guion** que reproduce cada tabla, cada prueba y cada figura desde los resultados del Entregable 3. |
| `tablas/` | Tablas en LaTeX (booktabs) que genera `generar.py`. |
| `datos/` | CSV con toda cifra que aparece en el texto (pruebas, métricas, ablación, sensibilidad, costo, importancias). |
| `figures/` | Las nueve figuras (PDF vectorial, inglés, 12 pt, ancho máximo 6,3 in). |
| `referencias.bib` | 20 entradas (la sección cita 12). |
| `compilar.py` | Compila con Tectonic y cuenta las palabras por subsección frente al presupuesto de la guía. |

## Cómo regenerar

Desde esta carpeta, con el Python del entorno `entrega 3/.venv` (tiene pandas, scikit-learn, interpret, arch,
statsmodels y matplotlib):

```powershell
$env:PYTHONIOENCODING = "utf-8"
& "..\entrega 3\.venv\Scripts\python.exe" generar.py     # pruebas, tablas, datos y figuras (unos minutos)
& "..\entrega 3\.venv\Scripts\python.exe" compilar.py    # main.pdf + conteo de palabras
```

En el repositorio publicado esta carpeta es `InformeFinal/resultados/`: la carpeta del Entregable 3 es
`../../Entrega3` (los guiones la buscan allí o en la variable de entorno `ENTREGA3`) y el intérprete es el del
entorno creado en ella (`..\..\Entrega3\.venv\Scripts\python.exe`). Los pipelines finales de
`resultados/modelos/` (490 MB) no están en el repositorio: sin ellos `generar.py` completo no puede correr, pero
`--tablas` y `--figuras` sí.

`generar.py --figuras` solo vuelve a dibujar las figuras y `generar.py --tablas` solo reescribe las tablas, en los
dos casos a partir de `datos/*.csv`, sin repetir los cálculos.

## Qué calcula `generar.py` y de dónde

- **Métricas y pruebas** desde `entrega 3/resultados/experimentos.parquet` (los cinco puntajes externos de cada
  corrida, con los mismos pliegues para todos los modelos) y `predicciones_prueba.parquet` (puntajes y clases de
  cada modelo final sobre los 9.745 clientes de prueba): *t* corregida de Nadeau–Bengio, DeLong, McNemar exacta
  en el punto de operación, Diebold–Mariano con HLN, *t* correlacionada bayesiana con ROPE, intervalos
  *bootstrap* BCa, *Model Confidence Set*, *d* de Cohen y delta de Cliff, con corrección de Holm. Las funciones
  de DeLong, *bootstrap*, Diebold–Mariano y MCS son las de `entrega 3/src/estadistica.py`; las demás están en
  el guion.
- **Ablación** desde las 20 corridas de la EBM (balanceo y optimizador), `computo/fidelidad.csv` (fidelidad),
  `complementos/semillas.csv` (semillas) y un reajuste de la EBM sin interacciones en la partición y etiqueta de
  la comparación con la Entrega 2 (`comparacion_entrega2/`).
- **Sensibilidad** desde las semillas, las configuraciones ganadoras de las 16 corridas de clasificación de la
  EBM, `computo/escalamiento.csv`, una curva de aprendizaje de la EBM que el guion ajusta (cinco tamaños) y
  curvas de un hiperparámetro a la vez (validación cruzada de 3 pliegues con B = 2), también ajustadas aquí.
- **Comparación con la Entrega 2** desde `comparacion_entrega2/` (intervalos y contrastes ya calculados) y
  **optimizadores** desde la tabla maestra sin las corridas de la EBM (27 y 7 bloques, con Friedman) más las
  columnas de eficiencia publicadas en el artículo del Entregable 3.
- **Costo** desde `computo/tiempos_finales.csv`, `computo/escalamiento.csv` y los pipelines guardados en
  `resultados/modelos/` (número de parámetros).
- **Interpretabilidad y errores** desde el pipeline final de la EBM (importancia de términos, funciones de forma,
  importancia por permutación sobre la prueba) y las predicciones de prueba cruzadas con la edad y
  `active_products` de `datos/customer_data.csv`.
