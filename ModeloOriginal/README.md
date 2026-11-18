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
| `src/` (resto) | Marco de experimentos del Entregable 3 con la recencia corregida, LightGBM y el modelo original en el catálogo |
| `tests/` | Pruebas unitarias (`pytest`): solución conocida, semilla, entradas inválidas, `clone`, `Pipeline` y las propiedades teóricas |
| `experiments/protocolo.md` | Protocolo experimental pre-registrado (etiqueta `protocolo-modelo-original-v1`) |
| `experiments/` | Guiones que reproducen cada tabla |
| `resultados/` | Tabla maestra, auditoría de fuga y resultados de las pruebas |

## Uso

Desde esta carpeta, con el entorno del Entregable 3 (`../Entrega3/requirements.txt`):

    python -m pytest -q                                  # pruebas unitarias
    python experiments/01_auditoria_fuga.py              # auditoría de fuga
    python experiments/02_validacion_anidada.py          # validación anidada (semilla 42)

```python
from src.original_model import ClasificadorBDL, RegresorBDL

regresor = RegresorBDL().fit(X_train, puntaje_train)            # índice + mezcla + distribución predictiva
regresor.predict(X_test)                                         # media
regresor.conjunto_cubre(X_test, puntaje_test, cobertura=0.9)     # ¿el conjunto de predicción contiene el valor?

clasificador = ClasificadorBDL().fit(X_train, etiqueta_train, puntaje=puntaje_train)
clasificador.predict_proba(X_test)                               # P(puntaje > umbral | x)
```
