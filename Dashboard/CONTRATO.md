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

La conclusión del informe es deliberadamente sobria y la app no puede subirle el tono: **«la alcanza, no la supera»**. La
EBM es equivalente en precisión a los mejores modelos de árboles (XGBoost, Random Forest, árbol de decisión), supera a los
lineales, a k-NN y a Naive Bayes, es más lenta de ajustar, igual de rápida al predecir y se puede leer. Con la regla de
superioridad de la guía (§6.4: mejor media que todos, tres pruebas significativas tras Holm, efecto no despreciable,
consistencia) **la EBM no es superior**: el veredicto explícito es de no superioridad, y se dice.

## 2. Estructura y archivos

| Ruta | Página | Qué tiene |
|---|---|---|
| `/` | `paginas/contexto.py` | Problema y relevancia (con una cifra citada), pregunta y objetivos, formulación, datos (fuente, periodo, licencia, tamaño, diccionario, objetivo, tarea), metodología (diagrama del pipeline), limitaciones, ética y sesgos, conclusiones, autores, versión y fecha |
| `/eda` | `paginas/eda.py` | Filtros globales, objetivo y balance, univariadas, explorador bivariado (x, y, color), correlaciones, faltantes y atípicos, la trampa de `active_products`; una frase de interpretación por gráfico |
| `/modelos/<vista>` | `paginas/modelos.py` | Subnavegación y doce vistas de `secciones/` (tabla abajo) |

| Vista | Módulo | Grupo | Qué pide la guía |
|---|---|---|---|
| `lideres` | `secciones/lideres.py` | Los ocho modelos | Tabla (`dash_table.DataTable`, ordenable y filtrable) con la mejor combinación de cada modelo: balanceo, optimizador, métrica media ± desviación del bucle externo, tiempo de inferencia; mejor combinación global y su pertenencia a los finalistas (Friedman → Nemenyi, MCS) |
| `prediccion` | `secciones/prediccion.py` | Los ocho modelos | Formulario generado del esquema (validación de rangos y tipos), selector de modelo; predicción, probabilidad o intervalo, incertidumbre con su cobertura nominal y explicación local SHAP |
| `diagnostico` | `secciones/diagnostico.py` | Los ocho modelos | Clasificación: matriz de confusión, ROC, precisión-recall, calibración. Regresión: real frente a predicho, residuos, ACF de residuos, histograma |
| `interpretabilidad` | `secciones/interpretabilidad.py` | Los ocho modelos | Resumen SHAP global del modelo elegido (importancia media y dispersión) |
| `comparacion` | `secciones/comparacion.py` | Los ocho modelos | Diagrama de diferencia crítica (clasificación) o tabla del MCS (regresión) con la EBM incluida, y la EBM frente a cada referencia |
| `propuesto` | `secciones/propuesto.py` | La EBM | Descripción, formulación, pseudocódigo, sustento teórico y complejidad, diferencias con la literatura, curva de aprendizaje, tabla de pruebas con p ajustados y tamaño del efecto, resumen de robustez y **veredicto explícito** |
| `decide` | `secciones/modelo.py` | La EBM | Las curvas de forma, la importancia y las 78 tablas |
| `cliente` | `secciones/cliente.py` | La EBM | Un cliente, su explicación exacta y el simulador |
| `campana` | `secciones/campana.py` | La EBM | Campaña con capacidad limitada |
| `robustez` | `secciones/robustez.py` | La EBM | Ablación, optimizadores, sensibilidad, costo, curva de aprendizaje |
| `errores` | `secciones/errores.py` | La EBM | Segmentos, escalera, casos y cohortes |
| `ficha` | `secciones/ficha.py` | La EBM | Ficha del modelo (*model card*): uso previsto, datos, métricas, limitaciones, riesgos, usos no recomendados |

| Archivo | Qué es | Quién lo cambia |
|---|---|---|
| `app.py` | Cascarón: secciones (`SECCIONES`), vistas (`VISTAS`, `GRUPOS`, `DESTINOS`, `RECORRIDO`), redirecciones de las direcciones viejas, `/documento/<clave>`, `/ficha.md` | solo integración |
| `ui.py`, `tokens.py`, `assets/estilos.css` | Sistema de diseño: componentes, roles de color, plantilla de Plotly | solo integración |
| `formato.py` | Números en es-CO (`fmt`, `num`, `con_signo`, `pct`, `entero`, `intervalo`, `p_valor`, `prob`) | solo integración |
| `motor.py` | La EBM en vivo (numpy): clientes, puntuar, explicar, simular, formas, campaña | solo integración |
| `modelos_vivos.py`, `explicacion.py`, `bosque.py` | Los 16 modelos en vivo desde `artifacts/`: predicción, incertidumbre, SHAP; el Random Forest de clasificación en arreglos de numpy (idéntico, para caber en 512 MB) | solo integración |
| `datos_app.py` | Lectura de `datos/` y nombres visibles del informe | solo integración |
| `paginas/<seccion>.py`, `secciones/<vista>.py` | Una sección o una vista | **su autor** |
| `datos/`, `artifacts/` | Lo que la app lee (lo arman `preparar_datos.py` y `preparar_artefactos.py`); `datos/origen.json` y `artifacts/indice.json` dicen de dónde salió cada archivo | nadie a mano |

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
  interpretabilidad `ip-`, comparacion `co-`, propuesto `pp-`, decide `mo-`, cliente `cl-`, campana `ca-`, robustez `ro-`,
  errores `er-`, ficha `fi-`. (Los módulos `secciones/inicio.py` `in-` y `secciones/limites.py` `li-` son de la versión anterior:
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

**Los 16 modelos en vivo** (`import modelos_vivos as V`): la app carga las tuberías de `artifacts/` cuando se piden (nunca
entrena). Todo está en los docstrings de `modelos_vivos.py` y `explicacion.py`.

| Llamada | Devuelve |
|---|---|
| `V.modelos(tarea)` | Los ocho modelos de la tarea en el orden del informe |
| `V.metadata(tarea, modelo)` | Su `metadata.json`: `balanceo`, `optimizador`, `semilla`, `hiperparametros`, `versiones`, `entrada.esquema`, `metricas_bucle_externo` (`metrica`, `media`, `desviacion`), `metricas_prueba` (la fila de `clasificacion_prueba.csv` o `regresion_prueba.csv`), `computo` (`ajuste_s`, `inferencia_prueba_s` para 9.745 clientes, `carga_artefacto_s`), `ida_y_vuelta` |
| `V.esquema()` | Las 48 variables: `nombre`, `nombre_visible`, `unidad`, `descripcion`, `tipo` (`entera`, `real`, `booleana`, `categorica`), `min`, `max`, `mediana` o `categorias`, `moda` |
| `V.a_tabla(fila_dict)`, `V.validar(fila_dict)` | La fila como DataFrame con los tipos del entrenamiento; la lista de problemas (vacía si es válida) |
| `V.predecir(tarea, modelo, X)` | El puntaje: probabilidad sin recalibrar (el SVM lineal da su margen) o `churn_probability` prevista |
| `V.probabilidad(modelo, s)` | Probabilidad recalibrada (isotónica sobre predicciones fuera de pliegue). Datos en `datos/calibracion_modelos.json` (ECE y Brier en prueba) |
| `V.conjunto(modelo, p_cal, "0.90")` | Conjunto conforme de clases, con `cobertura_prueba` medida |
| `V.intervalo(modelo, yhat, "0.90")` | Intervalo conforme de regresión, con `cobertura_prueba` (`datos/conformal.json`, niveles 0.80, 0.90, 0.95) |
| `V.explicar(tarea, modelo, X)` | `{"aportes": DataFrame filas × 48, "base", "salida", "metodo", "escala", "exacto", "error"}`: base + suma de aportes = salida en la `escala` del modelo (logit, margen, probabilidad o pronóstico). En vivo, k-NN, el árbol y Random Forest se estiman por permutaciones (`exacto` False, con `error` de Monte Carlo; k-NN ≈ 0,4 s por fila, los árboles ≈ 0,1 s en este PC, unas diez veces más en Render); los lineales, Naive Bayes, XGBoost y la EBM son exactos. La librería shap no está en el servidor: no la importe una vista |
| `V.vecinos(tarea, "knn", X)` | Los k vecinos de la fila en el entrenamiento (`posiciones`, `distancias`, `etiquetas`, `filas`) |
| `V.shap_indice()`, `V.shap_muestra(tarea, modelo)` | Resumen SHAP global precalculado: método, escala, importancia media absoluta, verificación frente a la librería shap; valores de 500 clientes de prueba fijos (150 en k-NN) |
| `V.entrenamiento()` | `datos/eda/entrenamiento.parquet`: los 38.978 clientes de entrenamiento con los 48 predictores, `active_products`, `churn_probability`, `riesgo_alto`, `fila` y las indicadoras `falta__<columna>` de las columnas originales con faltantes |

**Tablas de resultados** (`import datos_app as D`): `D.tabla(rel)`, `D.json(rel)`, `D.clave_valor(rel)`, `D.fila(...)`,
`D.fuente(rel)` (texto para `fuentes=[…]`), `D.prueba(tarea)`, `D.pruebas(tarea)`, `D.puntajes()`, `D.pronosticos()`,
`D.ablacion()`, `D.errores_segmentos()`, `D.techo()`, `D.resumen()`, `D.veredicto(fila, tarea)`, `D.en_frase(clave)` y los
nombres visibles **exactos del informe**: `D.NOMBRE_MODELO`, `D.NOMBRE_CORTO`, `D.NOMBRE_BALANCEO`, `D.NOMBRE_OPTIMIZADOR`,
`D.ORDEN_MODELOS[tarea]`, `D.ROL_MODELO`.

Archivos de `datos/` (`datos/origen.json` dice de dónde salió cada uno):

- `resultados/` (copia exacta de la sección de resultados del informe): `clasificacion_prueba.csv`, `regresion_prueba.csv`
  (una fila por modelo: métrica externa de la validación anidada `*_ext_media`/`*_ext_desv`, de prueba con intervalo BCa,
  punto de operación, MCS `p_mcs`/`en_mcs`), `pruebas_clasificacion.csv`, `pruebas_regresion.csv` (la EBM frente a cada
  referencia, con p crudos y de Holm `*_holm` y tamaños de efecto), `pliegues_auc.csv`, `pliegues_rmse.csv` (5 pliegues
  externos × 8 modelos), `curva_aprendizaje.csv`, `ablacion.csv`, `optimizadores.csv`, `semillas*.csv`, `sensibilidad_*.csv`,
  `costo.csv`, `fidelidad_*.csv`, `errores_segmentos.csv`, `importancia_*.csv`, `forma_ebm_*.csv`, `resumen.json`, …
- `cd.json`: por tarea, rangos medios por pliegue externo, Friedman, Iman-Davenport y la diferencia crítica de Nemenyi.
- `entrega3/`: `metricas_prueba.csv`, `corridas.csv` (160 corridas), `curvas_anytime.csv`, `complementos/`, `computo/`
  (`tiempos_finales.csv`, `escalamiento.csv`, `fidelidad.csv`, `equipo.json`), `comparacion_entrega2/`.
- `eda/`: `entrenamiento.parquet`, `faltantes.csv` (54 columnas originales), `por_variable.csv`, `objetivo_hist.csv`,
  `cohortes.csv`, `asociacion.csv`, `etl.json`.
- `puntajes_prueba.parquet` y `pronosticos_prueba.parquet`: una columna por modelo final (más `y`), índice `fila`.

Trampas de los datos:

- Signos: en `pruebas_*.csv` un delta positivo **favorece a la EBM**; en `ablacion.csv`, `delta_completo` positivo significa
  que **la variante pierde**. `corridas.csv` trae `media_ext` con el RMSE en negativo: use `metrica_externa`.
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
2. El color codifica significado y es el mismo en todas las vistas: `modelo` = la EBM (azul); `rival` = XGBoost, Random
   Forest y árbol de decisión (rosa); `base` = lineales, k-NN, Naive Bayes y la Entrega 2 (gris); `real` = lo observado (casi
   negro); `contexto` = lo demás. `D.ROL_MODELO[clave]`. Verde, ámbar y rojo son solo estados, con icono y palabra.
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

**Lenguaje** (la referencia es el informe final: sus resultados y sus conclusiones):

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

- El informe final (`../Informe Final/` en la carpeta de trabajo; `InformeFinal/informe/` en el repositorio):
  `introduccion.tex`, `metodologia.tex`, `resultados.tex`, `conclusiones.tex`, `tablas/`, `referencias.bib`.
- `datos/origen.json` y `artifacts/indice.json`: de dónde salió cada archivo.
- `DISENO.md`; `ui.py`, `motor.py`, `modelos_vivos.py`, `explicacion.py` y `datos_app.py` (docstrings).
