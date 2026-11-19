# Informe del proyecto final: datos, metodología y resultados (modelo original BDL)

Informe del proyecto final de Machine Learning (predicción de la fuga de clientes con COFINFAD) con el modelo original
BDL (Boosting con Desplazamiento Latente). Responde a la calificación de la entrega anterior de esta sección (2,4/5),
cuyo comentario fue: «entregaron solo la sección de resultados. No hay EDA, benchmarks, modelo, pipeline, fórmulas ni
tabla de hiperparámetros. Además, la EBM es un modelo ya existente…». Por eso el documento trae ahora:

| Parte | Dónde | Qué contiene |
|---|---|---|
| Introducción | `introduccion.tex` | Problema, cambio de la EBM a BDL y contribuciones. |
| Datos y EDA | `metodologia.tex`, §2.2 | Fuente, ETL (tabla de columnas eliminadas), hallazgos del análisis exploratorio y figura del objetivo. |
| Pipeline | `metodologia.tex`, §2.3 | Figura del pipeline con la validación anidada, partición primero y corrección de la recencia (fórmula). |
| Benchmarks | `metodologia.tex`, §2.4 | Los modelos del curso, la regresión bayesiana, la EBM y LightGBM, con optimizador y presupuesto. |
| Tabla de hiperparámetros | `tablas/hiperparametros.tex` | Espacio de búsqueda e hiperparámetros finales de los 11 modelos. |
| Modelo y fórmulas | `metodologia.tex`, §2.5 | Formulación de BDL, verosimilitud, pasos E y M, paso de boosting, algoritmo, regla de salida y sustento teórico. |
| Protocolo | `metodologia.tex`, §2.6 | Validación anidada, protocolo registrado, pruebas de la guía, conjuntos adicionales y ablación. |
| Resultados | `resultados.tex` | Las ocho subsecciones de la guía de resultados, en 2.293 palabras (máximo 3.000). |
| Conclusiones | `conclusiones.tex` | Hallazgos, limitaciones y trabajo futuro. |

| Archivo | Qué es |
|---|---|
| `main.pdf` | El informe compilado (21 páginas). |
| `main.tex` | Preámbulo e inclusión de las cuatro partes. |
| `generar.py` | **Único guion** que reproduce cada tabla, cada prueba y cada figura (la del objetivo se copia de la sección de metodología anterior, `../metodologia/figures` en el repositorio, donde la genera `figuras.py`). |
| `Informe_Proyecto_Final_BDL.pdf` | Copia de `main.pdf` con un nombre descriptivo, para compartir. |
| `tablas/`, `datos/`, `figures/` | Tablas en LaTeX (booktabs), toda cifra en CSV y JSON, y las figuras (PDF vectorial, inglés, fuente sin serifa). |
| `referencias.bib`, `armar_bib.py` | Las 29 entradas citadas, tomadas de la bibliografía del libro y de la versión anterior. |
| `compilar.py` | Compila con Tectonic y cuenta las palabras de cada parte (las de resultados, por subsección frente a la guía). |

## Cómo regenerar

Desde esta carpeta, con el Python del entorno `entrega 3/.venv`:

```powershell
& "..\entrega 3\.venv\Scripts\python.exe" -B generar.py        # todo (cerca de un minuto y medio)
& "..\entrega 3\.venv\Scripts\python.exe" -B generar.py --tablas
& "..\entrega 3\.venv\Scripts\python.exe" -B generar.py --figuras
python armar_bib.py
python compilar.py
```

`generar.py` busca `ModeloOriginal` en `../entrega 3/repo_entregable3/ModeloOriginal`, en `../../ModeloOriginal` o en
la variable de entorno `MODELO_ORIGINAL`, y solo lee sus resultados.

## Qué se añadió después del protocolo registrado

- **Regresión bayesiana (BayesianRidge).** La guía la incluye entre los modelos obligatorios y el protocolo no la
  tenía. Usa los mismos pliegues externos (comprobados: Ridge, reajustado con sus hiperparámetros guardados, reproduce
  el RMSE de cada pliegue con una diferencia de 10⁻¹⁷) y el mismo preprocesamiento; sus hiperparámetros se estiman por
  máxima evidencia dentro de cada ajuste.
- **Diebold-Mariano-HLN y la prueba bayesiana con ROPE.** Son las pruebas que la guía exige para regresión tabular; el
  protocolo registró Wilcoxon, la t corregida, la permutación pareada y el bootstrap BCa. El veredicto sobre R3 sigue
  siendo el del protocolo.
- **Importancia por permutación** en los clientes de prueba.

El SVR con kernel RBF que también pide la guía no se incluyó: el exacto no escala a 38.978 filas y el proyecto usó SVR
lineal. Queda declarado como limitación en el texto.
