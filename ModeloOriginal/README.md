# Modelo original: Boosting con Desplazamiento Latente (BDL)

Corrección del proyecto final a partir de la retroalimentación del docente:

1. **El modelo original debe tener una regla de aprendizaje propia, programada por el equipo.** La EBM del Entregable 3
   viene de una librería (`interpret`), así que pasa a ser un benchmark reciente. El modelo original es ahora
   **BDL**, programado desde cero en [`src/original_model.py`](src/original_model.py) con `numpy` y `scipy`.
2. **La recencia se calculaba con la fecha máxima de todo el detalle de transacciones**, que incluye a los clientes de
   prueba. Ahora la fecha de corte sale solo de los clientes de entrenamiento ([`src/datos.py`](src/datos.py)).
   [`experiments/01_auditoria_fuga.py`](experiments/01_auditoria_fuga.py) comprueba que la corrección no cambia ningún
   dato (la fecha es la misma, 29 de diciembre de 2023) y por qué tampoco podría cambiar un resultado.

## La idea del modelo

El puntaje de fuga de COFINFAD es un índice que depende de los predictores **más un escalón** que depende de una
variable que el modelo no recibe (los productos activos, excluidos por fuga de información). Un regresor corriente
trata ese escalón como ruido. BDL lo modela como una clase latente:

    s = F(x) + c_z + e,    z ∈ {1, …, K},  P(z = k) = π_k,  e ~ N(0, σ²)

y aprende `F` (una suma de árboles pequeños), los desplazamientos `c_k` y sus pesos `π_k` con un algoritmo EM
generalizado en el que el paso M sobre `F` es un paso de boosting contra el objetivo **sin escalón**. De la mezcla salen
la probabilidad de superar el umbral de riesgo alto (la clasificación) y conjuntos de predicción con cobertura
garantizada.

## Contenido

| Ruta | Qué es |
|---|---|
| `src/original_model.py` | El modelo: `RegresorBDL`, `ClasificadorBDL` y el alias `ModeloOriginal` |
| `src/` (resto) | Marco de experimentos del Entregable 3 con la recencia corregida (`datos.py`), LightGBM y el modelo original en el catálogo (`modelos.py`), el plan de la comparación (`plan.py`), las pruebas de hipótesis (`pruebas.py`, `estadistica.py`), las variantes de la ablación (`variantes.py`), los conjuntos adicionales (`conjuntos.py`) y el generador de tablas, figuras y textos del libro (`informe.py`) |
| `tests/` | Pruebas unitarias (`pytest`): solución conocida, semilla, entradas inválidas, `clone`, `Pipeline` y las propiedades teóricas del modelo; el marco, las pruebas de hipótesis y los formatos del informe |
| `experiments/protocolo.md` | Protocolo experimental pre-registrado (etiqueta `protocolo-modelo-original-v1`) |
| `experiments/` | Guiones numerados que producen cada resultado (tabla siguiente) y `correr_todo.py`, que los ejecuta en orden |
| `resultados/` | Tabla maestra `experimentos.parquet`, auditoría de fuga, predicciones y métricas de prueba, pruebas de hipótesis, veredicto, techo teórico, estabilidad por semilla, ablación, robustez, búsqueda de literatura y conjuntos adicionales |

| Guion | Qué produce |
|---|---|
| `00_busqueda_literatura.py` | Matriz de novedad: consulta OpenAlex, arXiv y Crossref y guarda lo que devuelven |
| `01_auditoria_fuga.py` | Auditoría de fuga: compara el flujo del Entregable 3 con el corregido |
| `02_validacion_anidada.py` | Validación anidada de 5 × 3 de los diez modelos, por semilla |
| `03_prueba.py`, `03b_semillas.py`, `03c_memoria.py`, `03d_recomendados.py` | Evaluación en prueba, calibración e incertidumbre; estabilidad con diez semillas y costo de cómputo; memoria de cada modelo e inferencia según las filas; EBM y LightGBM con los hiperparámetros recomendados por su librería |
| `04_pruebas_hipotesis.py`, `04b_techo.py`, `04c_optimo.py`, `04d_empates_campana.py`, `04e_referencias_homogeneas.py` | Las cuatro pruebas del protocolo con el ajuste de Holm y el veredicto; el techo teórico del conjunto; las mismas pruebas para el predictor óptimo; la sensibilidad de la campaña del 25 % a los empates; las referencias homogéneas del benchmark (MAE con la mediana y conjunto de predicción por densidad) |
| `05_conjuntos_adicionales.py` | El mismo procedimiento en cuatro conjuntos públicos |
| `06_robustez.py`, `07_ablacion.py` | Sensibilidad, ruido, tamaño de muestra (con y sin la mezcla) y cambio de distribución; ablación |
| `08_tablas_libro.py`, `09_rutas_relativas.py` | Tablas, figuras y textos del libro, a partir de los resultados; rutas de las tablas maestras relativas al repositorio |

## Uso

Desde esta carpeta, con las dependencias de `requirements-dev.txt`:

    python -m pytest -q                                  # pruebas unitarias
    python experiments/01_auditoria_fuga.py              # auditoría de fuga
    python experiments/correr_todo.py                    # todos los experimentos, en orden; salta lo ya hecho
    python experiments/correr_todo.py --cola b           # ablación y robustez, en otra terminal

```python
from src.original_model import ClasificadorBDL, RegresorBDL

regresor = RegresorBDL().fit(X_train, puntaje_train)            # índice + mezcla + distribución predictiva
regresor.predict(X_test)                                         # media
regresor.conjunto_cubre(X_test, puntaje_test, cobertura=0.9)     # ¿el conjunto de predicción contiene el valor?

clasificador = ClasificadorBDL().fit(X_train, etiqueta_train, puntaje=puntaje_train)
clasificador.predict_proba(X_test)                               # P(puntaje > umbral | x)
```
