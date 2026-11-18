# Guía de construcción de las secciones y vistas

Documento interno para quien escribe o modifica una sección del dash. Lo que el usuario final necesita está en `README.md`;
el sistema visual, en `DISENO.md`.

## 1. Qué es la app y para quién

**Fuga de clientes · Laboratorio del modelo**: el dashboard predictivo del proyecto del curso de Machine Learning
(Universidad del Norte) sobre la fuga de clientes de una *fintech* (conjunto COFINFAD). Es el **Bloque E** de la guía
«Proyecto Final: Tercer Entregable — Dashboard Predictivo, Modelo Original y Evaluación Rigurosa»: una aplicación Dash
multipágina con tres secciones (**Contexto**, **EDA** y **Modelos**), desplegada en Render (Bloque F), que permite a una
persona no experta comprender el problema, explorar los datos y usar los modelos para predecir. La sustentación (15 a 20
minutos) se hace **sobre el dashboard desplegado**, así que el dash también es el guion de la exposición: problema y
formulación, EDA, fundamentos teóricos del modelo, implementación y diseño experimental, resultados, conclusiones.

El profesor penalizó en entregas anteriores las cifras escritas a mano y las fugas de información: aquí toda cifra sale
de un archivo y se dice de cuál. Sus comentarios al Entregable 3 (nota 4,9) piden dos cosas que el dash debe atender:
nombrar **Successive Halving con su factor de reducción** (el proyecto usó dos fidelidades en Random Forest y la EBM, no
SH: hay que decirlo con precisión) y reconocer que **el carácter sintético de los datos limita la utilidad práctica** de
las conclusiones.

La primera versión del proyecto final se calificó con 3,0 porque el «modelo original» era la EBM, que viene de la
librería `interpret` (para la guía, un *import*), y porque la recencia se calculaba con una fecha que incluía a los
clientes de prueba. La corrección está en `ModeloOriginal/` y el dash la trae con `preparar_original.py`:

- **El modelo original es BDL (Boosting con Desplazamiento Latente)**, programado desde cero (`src/original_model.py`,
  copia exacta del de la corrección). El puntaje es un índice de los predictores menos 0,05 por producto activo; esa
  variable se excluye por fuga de información y deja el puntaje en cinco niveles que los benchmarks tratan como ruido.
  BDL modela ese escalón como una clase latente (`s = F(x) + c_z + e`): un EM generalizado cuyo paso M sobre F es un paso
  de boosting contra el objetivo sin escalón. Clasifica con la probabilidad de excedencia de la mezcla y da conjuntos de
  predicción conformes.
- **La EBM no es el modelo original**: es un benchmark reciente e interpretable, junto con LightGBM. Sus vistas se
  conservan en su propio grupo y ninguna la presenta como modelo propuesto ni original.
- **El veredicto de cada tarea sale de `datos/original/veredicto.json` y de `condicion4.json`, nunca de un texto fijo.**
  La regla de superioridad de la guía (§6.4: mejor media que el mejor benchmark, tres de las cuatro pruebas rechazan a su
  favor tras Holm, efecto no despreciable y consistencia en los conjuntos adicionales) la aplica
  `datos_app.estado_veredicto(tarea)`; las vistas escriben el resultado con `D.frase_veredicto(tarea)`. Si una condición
  no se cumple se dice «no superior» y por qué; si faltan repeticiones o conjuntos, se dice que es provisional o que está
  pendiente. Ninguna vista afirma más que lo que dicen los archivos.

El tono es sobrio. Sobre la EBM sigue valiendo la conclusión del informe del Entregable 3, sin subirle el tono: **«la
alcanza, no la supera»** (es equivalente en precisión a XGBoost, Random Forest y el árbol de decisión, supera a los
lineales, a k-NN y a Naive Bayes, es más lenta de ajustar, igual de rápida al predecir y se puede leer).

## 2. Estructura y archivos

| Ruta | Página | Qué tiene |
|---|---|---|
| `/` | `paginas/contexto.py` | Problema y relevancia (con una cifra citada), la cifra del modelo original con su veredicto, pregunta y objetivos, las correcciones (el modelo original es propio; la recencia, de `auditoria_fuga.json`), formulación, datos (fuente, periodo, licencia, tamaño, diccionario, objetivo, tarea), metodología (diagrama del pipeline), limitaciones, ética y sesgos, conclusiones, autores, versión y fecha |
| `/eda` | `paginas/eda.py` | Filtros globales, objetivo y balance, univariadas, explorador bivariado (x, y, color), correlaciones, faltantes y atípicos, la trampa de `active_products`; una frase de interpretación por gráfico |
| `/modelos/<vista>` | `paginas/modelos.py` | Subnavegación y trece vistas de `secciones/` en tres grupos (tabla abajo); pone el aviso «la EBM es un benchmark» sobre las vistas de su grupo |

| Vista | Módulo | Grupo | Qué pide la guía |
|---|---|---|---|
| `lideres` | `secciones/lideres.py` | Los diez modelos | Tabla (`dash_table.DataTable`, ordenable y filtrable) con la combinación de cada modelo: balanceo, optimizador, métrica media ± desviación del bucle externo, prueba con su intervalo, tiempo de inferencia; mejor combinación global, finalistas (Friedman → Nemenyi con los diez) y dónde queda el original con su veredicto. BDL y LightGBM traen la única combinación del protocolo |
| `prediccion` | `secciones/prediccion.py` | Los diez modelos | Formulario generado del esquema (validación de rangos y tipos), selector de modelo (BDL por omisión); predicción, probabilidad o intervalo, incertidumbre con su cobertura nominal y medida, y explicación local. Con BDL: su conjunto de predicción (unión de tramos) y cómo el índice se convierte en probabilidad |
| `diagnostico` | `secciones/diagnostico.py` | Los diez modelos | Clasificación: matriz de confusión, ROC, precisión-recall, calibración. Regresión: real frente a predicho, residuos, ACF de residuos, histograma. También LightGBM, con sus puntajes guardados |
| `interpretabilidad` | `secciones/interpretabilidad.py` | Los diez modelos | Resumen SHAP global del modelo elegido (importancia media y dispersión), comparado con el modelo original |
| `comparacion` | `secciones/comparacion.py` | Los diez modelos | El veredicto del original condición por condición, el diagrama de diferencia crítica con los diez modelos en las dos tareas, el original frente a cada benchmark (P1 a P4, Holm, tamaños del efecto), desempeño y pliegues de los diez; aparte, el análisis del Entregable 3 de la EBM frente a sus siete referencias (con el MCS en regresión) |
| `original` | `secciones/original.py` | El modelo original: BDL | La subsección del modelo original (guía §7.3.3): descripción y limitación que atiende, formulación, pseudocódigo, proposiciones y complejidad, diferencias con la literatura, curva de convergencia, lo que aprendió el índice, techo teórico, tabla de pruebas con p ajustados y tamaño del efecto, la tarjeta «P4 y los empates de la campaña» (`original/empates_campana.json`), resumen de robustez y **veredicto explícito por tarea** con las cuatro condiciones y el análisis de la causa (las pruebas del predictor óptimo, `original/optimo.json`) |
| `ficha` | `secciones/ficha.py` | El modelo original: BDL | Ficha del modelo (*model card*) de BDL: uso previsto, datos, métricas, limitaciones, riesgos, usos no recomendados |
| `propuesto` | `secciones/propuesto.py` | La EBM, benchmark interpretable | El algoritmo de la EBM y su análisis del Entregable 3: formulación, pseudocódigo, sustento teórico y complejidad, curva de aprendizaje, pruebas y robustez frente a los modelos de clase |
| `decide` | `secciones/modelo.py` | La EBM, benchmark interpretable | Las curvas de forma, la importancia y las 78 tablas |
| `cliente` | `secciones/cliente.py` | La EBM, benchmark interpretable | Un cliente, su explicación exacta y el simulador |
| `campana` | `secciones/campana.py` | La EBM, benchmark interpretable | Campaña con capacidad limitada: la EBM frente a cualquier otro modelo |
| `robustez` | `secciones/robustez.py` | La EBM, benchmark interpretable | Ablación, optimizadores, sensibilidad, costo, curva de aprendizaje |
| `errores` | `secciones/errores.py` | La EBM, benchmark interpretable | Segmentos, escalera, casos y cohortes |

| Archivo | Qué es | Quién lo cambia |
|---|---|---|
| `app.py` | Cascarón: secciones (`SECCIONES`), vistas (`VISTAS`, `GRUPOS`, `DESTINOS`, `RECORRIDO`), redirecciones de las direcciones viejas, `/documento/<clave>`, `/ficha.md` | solo integración |
| `ui.py`, `tokens.py`, `assets/estilos.css` | Sistema de diseño: componentes, roles de color, plantilla de Plotly | solo integración |
| `formato.py` | Números en es-CO (`fmt`, `num`, `con_signo`, `pct`, `entero`, `intervalo`, `p_valor`, `prob`) | solo integración |
| `motor.py` | La EBM en vivo (numpy): clientes, puntuar, explicar, simular, formas, campaña | solo integración |
| `modelos_vivos.py`, `explicacion.py`, `bosque.py` | Los 18 modelos en vivo desde `artifacts/` (16 del Entregable 3 y los dos de BDL): predicción, incertidumbre, SHAP y lo propio del original (`V.excedencia`, `V.distribucion`); el Random Forest de clasificación en arreglos de numpy (idéntico, para caber en 512 MB) | solo integración |
| `src/original_model.py` | El modelo original (BDL): copia exacta de `ModeloOriginal/src/original_model.py`, que deja `preparar_original.py` | nadie a mano (una prueba lo compara) |
| `datos_app.py` | Lectura de `datos/`, nombres visibles, roles de color, los lectores de `datos/original/` y el veredicto del original (`estado_veredicto`, `frase_veredicto`) | solo integración |
| `paginas/<seccion>.py`, `secciones/<vista>.py` | Una sección o una vista | **su autor** |
| `datos/`, `artifacts/` | Lo que la app lee (lo arman `preparar_datos.py`, `preparar_artefactos.py` y, encima, `preparar_original.py`); `datos/origen.json` y `artifacts/indice.json` dicen de dónde salió cada archivo | nadie a mano |

Una vista **solo** escribe su archivo. Si le falta algo en un módulo compartido (un componente, una clase CSS, un lector de
datos), lo resuelve dentro de su archivo con una función privada (`_algo`) usando solo componentes de `ui`, clases que ya
existen y tokens, y lo anota en su informe final para que integración lo suba al módulo compartido.

## 3. El patrón de una vista

```python
"""Predicción en vivo · ¿Qué predice cada modelo para un cliente?"""
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html

import app
import datos_app as D
import formato as F
import modelos_vivos as V
import ui

VISTA = "prediccion"
PRE = "pr"          # prefijo de TODOS los id de la vista: f"{PRE}-modelo"


def layout(**_):    # recibe la query string como argumentos (p. ej. modelo="xgboost"); siempre **_ para lo que no use
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), "Subtítulo que responde con la cifra.", [controles…], pantalla="cliente"),
        ui.seccion("Título-mensaje de la sección (h2)", [ui.rejilla([tarjeta_a, tarjeta_b], "2-1")]),
    ])


@callback(Output(f"{PRE}-cascada", "figure"), Input(f"{PRE}-modelo", "value"))
def _dibujar(modelo):
    …
```

- Prefijos de `id` (únicos en toda la app): contexto `cx-`, eda `ed-`, lideres `ld-`, prediccion `pr-`, diagnostico `dg-`,
  interpretabilidad `ip-`, comparacion `co-`, original `og-`, propuesto `pp-`, decide `mo-`, cliente `cl-`, campana `ca-`,
  robustez `ro-`, errores `er-`, ficha `fi-`. (Los módulos `secciones/inicio.py` `in-` y `secciones/limites.py` `li-` son de la versión anterior:
  Contexto reutiliza sus funciones, pero ya no se montan como pantallas.)
- `ui.grafico(id, …)` reserva los sufijos `-titulo`, `-sub`, `-leyenda`, `-tabla`, `-nota`, `-lienzo` y `-figura`;
  `ui.deslizador(id, …)`, el sufijo `-valor`. Un callback puede escribir en ellos.
- Los callbacks deben funcionar con los valores con que nace la vista (se ejecutan al cargar) y aguantar `None`.
  Nada de estado global mutable: los callbacks corren en varios hilos. `motor`, `modelos_vivos` y `datos_app` traen caché.
- `import app` da `app.pregunta(vista)`, `app.ruta_vista(vista)`, `app.VISTAS`, `app.DESTINOS` y `app.documentos_disponibles()`.
- Un enlace a otra vista: `app.ruta_vista("cliente") + "?fila=…"`; a una sección: `/` o `/eda`.
- Antes de escribir, lea `ui.py` (cada función tiene su docstring), `DISENO.md` y, como ejemplo completo,
  `secciones/inicio.py` o `secciones/comparacion.py`.

## 4. Datos

**La EBM en vivo y los clientes de prueba** (`import motor as M`; documentado en los docstrings de `motor.py`):

| Llamada | Devuelve |
|---|---|
| `M.clientes()` | Los 9.745 clientes de prueba, índice `fila` (posición en `customer_data.csv`). Columnas: contexto (`customer_id`, `active_products`, `occupation`, `nps_score`, `clv_segment`, `customer_lifetime_value`), los 48 predictores, `cohorte`, `churn_probability`, `riesgo_alto`, `p_ebm`, `p_ebm_cal`, `percentil_ebm`, `contactar_ebm`, `resultado_ebm`, `pron_ebm`, `p_xgboost`, `contactar_xgboost`, `pron_xgboost`, `p_logistica`, `contactar_logistica`. **No modificar** (compartida): `.copy()` si hace falta |
| `M.meta()` | `metadatos.json`: `etiqueta.umbral`, `n_entrenamiento`, `n_prueba`, `prevalencia_*`, `punto_operacion`, `ejemplos`, `ebm.<tarea>`, `metricas_prueba`, `modelos_finales`, `versiones` |
| `M.variables()` | `variables.csv`, índice `variable`: `nombre` visible, `unidad`, `tipo`, `derivada_etl`, `papel`, `regla`, `descripcion`, rangos y percentiles, `valores`, `categorias`, importancia y efectos de la EBM |
| `M.puntuar(X)`, `M.explicar(X, tarea)`, `M.simular(fila, cambios)`, `M.forma`, `M.efecto_categorias`, `M.densidad`, `M.importancia`, `M.ebm(tarea)`, `M.percentil`, `M.puesto`, `M.resumen_capacidad`, `M.curva_captura`, `M.curva_roc`, `M.auc` | Ver los docstrings |

**Los 18 modelos en vivo** (`import modelos_vivos as V`): la app carga las tuberías de `artifacts/` cuando se piden (nunca
entrena). Son los 16 del Entregable 3 y los dos del modelo original (`bdl`); LightGBM no tiene artefacto y solo está en
las tablas. Todo está en los docstrings de `modelos_vivos.py` y `explicacion.py`.

| Llamada | Devuelve |
|---|---|
| `V.modelos(tarea)` | Los modelos en vivo de la tarea, en el orden de `D.ORDEN_MODELOS`: el original primero (`V.hay_original(tarea)` dice si ya está) y los ocho del Entregable 3 |
| `V.excedencia(X)` | BDL, clasificación: cómo sale la probabilidad del índice de cada fila (`indice`, `umbral`, `pesos`, `desplazamientos`, `niveles`, `aporte`, `probabilidad`); `None` si el clasificador está en modo directo |
| `V.distribucion(X, "0.90")` | BDL, regresión: `indice`, `media` (el pronóstico), `mediana`, `central` (intervalo central), `tramos` (el conjunto de predicción, una unión de tramos), `largo`, `niveles`, `pesos` y las coberturas medidas en la prueba |
| `V.metadata(tarea, modelo)` | Su `metadata.json`: `balanceo`, `optimizador`, `semilla`, `hiperparametros`, `versiones`, `entrada.esquema`, `metricas_bucle_externo` (`metrica`, `media`, `desviacion`), `metricas_prueba` (la fila de `clasificacion_prueba.csv` o `regresion_prueba.csv`), `computo` (`ajuste_s`, `inferencia_prueba_s` para 9.745 clientes, `carga_artefacto_s`), `ida_y_vuelta` |
| `V.esquema()` | Las 48 variables: `nombre`, `nombre_visible`, `unidad`, `descripcion`, `tipo` (`entera`, `real`, `booleana`, `categorica`), `min`, `max`, `mediana` o `categorias`, `moda` |
| `V.a_tabla(fila_dict)`, `V.validar(fila_dict)` | La fila como DataFrame con los tipos del entrenamiento; la lista de problemas (vacía si es válida) |
| `V.predecir(tarea, modelo, X)` | El puntaje: probabilidad sin recalibrar (el SVM lineal da su margen) o `churn_probability` prevista |
| `V.probabilidad(modelo, s)` | Probabilidad recalibrada (isotónica sobre predicciones fuera de pliegue). Datos en `datos/calibracion_modelos.json` (ECE y Brier en prueba) |
| `V.conjunto(modelo, p_cal, "0.90")` | Conjunto conforme de clases, con `cobertura_prueba` medida |
| `V.intervalo(modelo, yhat, "0.90")` | Intervalo conforme de regresión, con `cobertura_prueba` (`datos/conformal.json`, niveles 0.80, 0.90, 0.95) |
| `V.explicar(tarea, modelo, X)` | `{"aportes": DataFrame filas × 48, "base", "salida", "metodo", "escala", "exacto", "error"}`: base + suma de aportes = salida en la `escala` del modelo (logit, margen, probabilidad o pronóstico). En vivo, k-NN, el árbol y Random Forest se estiman por permutaciones (`exacto` False, con `error` de Monte Carlo; k-NN ≈ 0,4 s por fila, los árboles ≈ 0,1 s en este PC, unas diez veces más en Render); los lineales, Naive Bayes, XGBoost, la EBM y BDL son exactos (BDL, con `RegresorBDL.contribuciones`; en clasificación reparte el índice F(x), `explicacion.ESCALA_INDICE`). La librería shap no está en el servidor: no la importe una vista |
| `V.vecinos(tarea, "knn", X)` | Los k vecinos de la fila en el entrenamiento (`posiciones`, `distancias`, `etiquetas`, `filas`) |
| `V.shap_indice()`, `V.shap_muestra(tarea, modelo)` | Resumen SHAP global precalculado: método, escala, importancia media absoluta, verificación frente a la librería shap; valores de 500 clientes de prueba fijos (150 en k-NN) |
| `V.entrenamiento()` | `datos/eda/entrenamiento.parquet`: los 38.978 clientes de entrenamiento con los 48 predictores, `active_products`, `churn_probability`, `riesgo_alto`, `fila` y las indicadoras `falta__<columna>` de las columnas originales con faltantes |

**Tablas de resultados** (`import datos_app as D`): `D.tabla(rel)`, `D.json(rel)`, `D.clave_valor(rel)`, `D.fila(...)`,
`D.fuente(rel)` (texto para `fuentes=[…]`), `D.prueba(tarea)`, `D.pruebas(tarea)`, `D.puntajes()`, `D.pronosticos()`,
`D.ablacion()`, `D.errores_segmentos()`, `D.techo()`, `D.resumen()`, `D.veredicto(fila, tarea)`, `D.en_frase(clave)` y los
nombres visibles **exactos del informe y de la corrección**: `D.NOMBRE_MODELO`, `D.NOMBRE_CORTO`, `D.NOMBRE_BALANCEO`,
`D.NOMBRE_OPTIMIZADOR`, `D.ORDEN_MODELOS[tarea]` (diez modelos, el original primero), `D.ROL_MODELO` y `D.ROL_VISTA_EBM`.
`D.prueba(tarea)` y `D.pruebas(tarea)` son las tablas del Entregable 3 (ocho modelos; la EBM frente a siete referencias).

**La corrección** (`datos/original/`, que arma `preparar_original.py`). Cada lector devuelve `None` si su archivo todavía no
existe, y la vista muestra ese bloque como pendiente con `ui.aviso` (qué falta y qué archivo lo traerá):

| Llamada | Devuelve |
|---|---|
| `D.lideres(tarea)` | Los diez modelos con las columnas de `resultados/<tarea>_prueba.csv` (las ocho filas del informe y las de BDL y LightGBM, `datos/original/lideres_<tarea>.csv`), más `origen`, `externa_media`, `externa_desv` y `externa_pliegues` (de todos los pliegues externos disponibles, los mismos para los diez) |
| `D.veredicto_original(tarea)`, `D.condicion4(tarea)` | La entrada de la tarea en `veredicto.json` (medias, mejor benchmark, Friedman, P1 a P4 frente al mejor, rechazos, condiciones 1 a 3) y en `condicion4.json` |
| `D.estado_veredicto(tarea)`, `D.frase_veredicto(tarea)` | **El veredicto**: `palabra` («superior», «no superior» o «pendiente»), las condiciones, el estado de la cuarta, cuántas se cumplen y si es provisional; y la frase con que se escribe en cualquier vista |
| `D.pruebas_original(tarea)`, `D.pliegues(tarea)` | Las pruebas del original frente a cada benchmark (`pruebas_hipotesis.csv`) y la métrica de cada pliegue externo de los diez modelos |
| `D.modelo_original(tarea)` | El modelo final por dentro (`modelo_<tarea>.json`): K, pesos, desplazamientos, árboles, historial de convergencia, BIC, importancia, curvas, conjunto de predicción |
| `D.corrida_original(tarea, modelo)`, `D.costo_original(tarea, modelo)` | La corrida en la tabla maestra de la corrección (con los hiperparámetros de cada pliegue) y el costo de BDL o LightGBM |
| `D.empates_campana()`, `D.empate_campana(modelo)`, `D.modelos_con_empate()` | Los empates en el corte de la campaña del informe (`empates_campana.json`): cuántos clientes comparten el puntaje del último contactado, cuántos entran, la captura según el desempate y el AUC-PR según la convención; `empate_campana` devuelve `None` si en el corte de ese modelo no hay empate |
| `D.frase_empate_captura(modelo)`, `D.frase_empate_corta(modelo)`, `D.frase_empate_auc_pr(modelo)` | **Las notas de los empates**, calculadas de ese archivo: van junto a la captura o la precisión del punto de operación y junto al AUC-PR de un modelo cuyo corte cae en un empate (los de puntaje escalonado). Devuelven `None` si no hace falta la nota: no se escribe otra a mano |
| `D.nota_ece()` | **La nota del ECE**: la app lo calcula con diez tramos de igual ancho (`D.TRAMOS_ECE`, la definición del Entregable 3) y la Tabla 7.11 del libro, con tramos de igual número de clientes, así que sus cifras no coinciden. Va una vez en cada vista que muestre un ECE; no se cambia un cálculo para igualar al otro |
| `D.original_tabla(rel)`, `D.original_json(rel)`, `D.hay_original(rel)`, `D.estado_original()` | Cualquier archivo de `datos/original/` (techo, auditoría, robustez, semillas, conjuntos adicionales, la búsqueda de literatura…) y qué había la última vez que corrió `preparar_original.py` |

Archivos de `datos/` (`datos/origen.json` dice de dónde salió cada uno):

- `resultados/` (copia exacta de la sección de resultados del informe): `clasificacion_prueba.csv`, `regresion_prueba.csv`
  (una fila por modelo: métrica externa de la validación anidada `*_ext_media`/`*_ext_desv`, de prueba con intervalo BCa,
  punto de operación, MCS `p_mcs`/`en_mcs`), `pruebas_clasificacion.csv`, `pruebas_regresion.csv` (la EBM frente a cada
  referencia, con p crudos y de Holm `*_holm` y tamaños de efecto), `pliegues_auc.csv`, `pliegues_rmse.csv` (5 pliegues
  externos × 8 modelos), `curva_aprendizaje.csv`, `ablacion.csv`, `optimizadores.csv`, `semillas*.csv`, `sensibilidad_*.csv`,
  `costo.csv`, `fidelidad_*.csv`, `errores_segmentos.csv`, `importancia_*.csv`, `forma_ebm_*.csv`, `resumen.json`, …
- `cd.json`: por tarea, rangos medios por pliegue externo, Friedman, Iman-Davenport y la diferencia crítica de Nemenyi, con
  los diez modelos (lo rehace `preparar_original.py` desde `original/pliegues_<tarea>.csv`).
- `original/`: los resultados de la corrección. Copias exactas (`veredicto.json`, `pruebas_hipotesis.csv`,
  `pliegues_<tarea>.csv`, `techo.json`, `optimo.json`, `empates_campana.json`, `auditoria_fuga.json`, `condicion4.json`,
  `metricas_prueba.csv`,
  `incertidumbre_prueba.csv`, `ablacion.csv`, `semillas.csv`, `errores_subgrupos.csv`, `robustez/*.csv`) y extractos
  (`lideres_<tarea>.csv`, `costo.csv`, `experimentos.csv`, `adicionales.csv`, `literatura.json`, `modelo_<tarea>.json`,
  `estado.json`). Un archivo que la corrección todavía no produjo no está: su bloque se muestra como pendiente.
- `entrega3/`: `metricas_prueba.csv`, `corridas.csv` (160 corridas), `curvas_anytime.csv`, `complementos/`, `computo/`
  (`tiempos_finales.csv`, `escalamiento.csv`, `fidelidad.csv`, `equipo.json`), `comparacion_entrega2/`.
- `eda/`: `entrenamiento.parquet`, `faltantes.csv` (54 columnas originales), `por_variable.csv`, `objetivo_hist.csv`,
  `cohortes.csv`, `asociacion.csv`, `etl.json`.
- `puntajes_prueba.parquet` y `pronosticos_prueba.parquet`: una columna por modelo final (más `y`), índice `fila`; con
  `bdl` y `lightgbm` desde que corre `preparar_original.py` (también en `calibracion_modelos.json`, `conformal.json`, donde
  `bdl` trae además su intervalo central y su conjunto de predicción, y `shap/`).

Trampas de los datos:

- Signos: en `pruebas_*.csv` un delta positivo **favorece a la EBM**; en `ablacion.csv`, `delta_completo` positivo significa
  que **la variante pierde**. `corridas.csv` trae `media_ext` con el RMSE en negativo: use `metrica_externa`. En
  `datos/original/` una ventaja, un δ de Cliff o una d de Cohen positivos **favorecen al modelo original**, y un rechazo
  cuenta para la regla solo si va a su favor (`a_favor`).
- Con BDL el intervalo conforme simétrico («pronóstico ± q») no cumple su cobertura nominal (su error cae en niveles
  separados): la incertidumbre del original se dice con su conjunto de predicción, y la vista lo advierte.
- En las vistas del grupo de la EBM y en la mitad del Entregable 3 de la comparación el modelo en foco es la EBM: use
  `D.ROL_VISTA_EBM`, no `D.ROL_MODELO`.
- Fracciones: `recall_*`, `precision_*`, `prevalencia`, cuotas de importancia, coberturas y `mape_prueba` son fracciones
  (`F.pct` recibe una fracción).
- El puntaje del SVM lineal es una función de decisión, no una probabilidad. Con ponderación de clases los puntajes están
  inflados (media 0,416 con prevalencia 0,246): nunca se muestran como «probabilidad de fuga» ni se compara un umbral 0,5
  entre modelos. Para ordenar clientes, el puntaje; para informar una probabilidad, la recalibrada.
- Dos etiquetas que no se mezclan: **percentil 75** (la del trabajo; AUC 0,8289) y **mediana** (solo en la comparación con
  la entrega anterior).
- El EDA se hace **solo con el entrenamiento** (el profesor lo pidió en la Entrega 1): nunca con los clientes de prueba.
- La EBM final no tiene interacciones (K = 0). La edad solo toma seis valores (29, 30, 44, 45, 59, 60) en tres cohortes.
- `active_products` **nunca** es una entrada: es la variable con la que se generó el puntaje y se excluyó (la «trampa»).
- `credit_utilization_ratio` falta en los clientes sin tarjeta de crédito (estructural): el ETL pone 0 sin tarjeta.
- Los datos son **sintéticos** (COFINFAD se generó con reglas): las conclusiones describen esas reglas, no clientes reales.

## 5. Reglas que no se negocian

**Diseño** (las comprueba `verificar_estilo.py`; el detalle está en `DISENO.md`):

1. Ningún color, tamaño de letra ni estilo en línea en una vista: solo componentes de `ui`, roles (`ui.traza(rol, forma)`,
   `ui.ROL[rol]`, `ui.VAR[rol]`, `ui.fuente_svg`) y clases del CSS. Paleta limitada: los cinco roles.
2. El color codifica significado: `modelo` = el modelo en foco (azul): el original, BDL, en las vistas de los diez modelos
   y en la suya (`D.ROL_MODELO[clave]`), y la EBM en las vistas de su grupo y en la mitad del Entregable 3 de la comparación
   (`D.ROL_VISTA_EBM[clave]`); `rival` = los benchmarks de la parte alta de la tabla: la EBM, LightGBM, XGBoost, Random
   Forest y el árbol de decisión (rosa); `base` = lineales, k-NN, Naive Bayes y la Entrega 2 (gris); `real` = lo observado
   (casi negro); `contexto` = lo demás. La leyenda de cada gráfico dice quién es el azul. Verde, ámbar y rojo son solo
   estados, con icono y palabra.
3. Formato es-CO en todo: `ui.num`, `ui.con_signo`, `F.pct`, `F.intervalo`, `F.p_valor`; en Plotly, texto ya formateado con
   `ui.hover(…)` y ejes con `ui.marcas(…)` o `ui.formato_eje(fig, …)`.
4. **Ninguna cifra escrita a mano**: toda cifra visible sale de `datos/`, de `artifacts/` o del motor, también en los
   subtítulos. Cada tarjeta y cada gráfico llevan `fuentes=[D.fuente(…)]` (o la ruta del artefacto).
5. Un solo eje y por gráfico, la unidad con `ui.unidad_eje`, leyenda HTML cuando hay dos series o más, y la tabla gemela en
   el «…» (`tabla=(encabezados, filas)`): es el texto alternativo de cada gráfico.
6. Título-mensaje: el título de un gráfico o de una sección dice la conclusión con su cifra.
7. Encabezados sin saltos: `h1` → `h2` → `h3`. Hasta 8 tamaños de letra por vista.
8. Debe verse bien a 1366 px y en tableta (768 px), y no salirse a 400 px.
9. Lo que falla se dice con `ui.aviso(…)` (también una entrada inválida del usuario: la app nunca se cae por una entrada).
10. Solo modo claro. Nada de `go.Waterfall`, `showlegend=True`, `shape="spline"`, `pattern` escrito en la vista ni títulos
    de eje. Fórmulas: `dcc.Markdown(..., mathjax=True)`.

**Lenguaje** (la referencia es el libro del proyecto final para el modelo original y el informe del Entregable 3 para la
EBM):

- El modelo original se llama «BDL» o «el modelo original». La EBM es «la EBM» o «el benchmark interpretable»: nunca «el
  modelo propuesto» ni «el modelo original».
- La novedad de BDL se dice como en la sección 4.5 del libro, con sus cuatro bloques (qué no es nuevo, qué es propio de BDL,
  qué se reutiliza y qué no se afirma). La regla de aprendizaje, un EM cuyo paso M es un paso de boosting, **no es nueva**:
  pertenece a una familia ya publicada. Lo propio de BDL es el modelo al que se aplica (un índice compartido con
  desplazamientos latentes constantes), su regla de salida y su implementación. Ninguna vista dice más que eso.
- El veredicto del original se escribe con `D.frase_veredicto(tarea)`: «superior», «no superior» o «pendiente de la cuarta
  condición», con cuántas condiciones cumple y, si faltan repeticiones, «resultado provisional». Nunca un texto fijo, y
  nunca «gana» ni «el mejor modelo»: se dice en qué condición falla y con qué valores.
- EBM frente a XGBoost, Random Forest y árbol: «empata», «equivalente», «ninguna prueba encuentra diferencia». Nunca
  «gana», «supera» ni «el mejor modelo». El 0,8289 nunca va solo.
- «Supera con significación tras la corrección de Holm» solo a k-NN, regresión logística, SVM lineal y Naive Bayes
  (clasificación) y a Lasso, Ridge, SVR lineal y k-NN (regresión).
- «Riesgo alto» es el cuarto superior de un puntaje de propensión del conjunto de datos, no fuga observada. Sin lenguaje
  causal. «Puntaje del modelo» (sin recalibrar) y «probabilidad recalibrada».
- Nombres de modelos, balanceos y optimizadores: los de `datos_app`. Castellano llano y sobrio, sin exclamaciones.

## 6. Cómo se prueba

Puerto: el de omisión es 8060; para probar use otro (8061 en adelante; nunca 8000, 8020 ni 8050, que son de otras apps).

```
.venv/Scripts/python.exe -B probar_app.py prediccion                     (arma la vista y ejecuta sus callbacks; sin navegador)
.venv/Scripts/python.exe -B verificar_estilo.py --pantallas prediccion   (reglas de diseño en el código)
FUGA_SOLO=prediccion PORT=8061 .venv/Scripts/python.exe -B app.py        (la app solo con esa vista; en segundo plano)
python -B capturar.py --puerto 8061 --rutas /modelos/prediccion --salida capturas
python -B verificar_estilo.py --vivo 8061 --pantallas prediccion
```

Para probar como en Render (Linux, 512 MB y 0,1 CPU): `docker build -t fuga-dash .` y
`docker run --rm -p 8071:8060 --memory=512m --cpus=0.1 fuga-dash`; `docker stats` da la memoria. Ahí todo es unas diez
veces más lento que en este PC: una vista que tarda 1 s aquí tarda 10 s allá.

En Git Bash, anteponga `MSYS_NO_PATHCONV=1` a los comandos con una ruta de la app como argumento. La app no recarga los
cambios sola: tras editar, detenga su servidor y vuelva a lanzarlo. Hay que **mirar las capturas** (1366, 768 y 400 px).
Una vista está lista cuando `probar_app.py` y los dos modos de `verificar_estilo.py` pasan, las capturas se ven bien, cada
cifra coincide con su archivo y cada interacción hace lo que promete. Al terminar, detenga el servidor que lanzó.

## 7. Referencias

- El libro del proyecto final (`ProyectoFinal/` en el repositorio): los capítulos de literatura, del modelo original, de
  validación y de evaluación ampliada son la fuente del relato de BDL; `ModeloOriginal/experiments/protocolo.md`, el
  protocolo pre-registrado, y `ModeloOriginal/resultados/`, lo que copia `preparar_original.py`.
- El informe final del Entregable 3 (`../Informe Final/` en la carpeta de trabajo; `InformeFinal/informe/` en el
  repositorio): `introduccion.tex`, `metodologia.tex`, `resultados.tex`, `conclusiones.tex`, `tablas/`, `referencias.bib`.
- `datos/origen.json` y `artifacts/indice.json`: de dónde salió cada archivo.
- `DISENO.md`; `ui.py`, `motor.py`, `modelos_vivos.py`, `explicacion.py` y `datos_app.py` (docstrings).
