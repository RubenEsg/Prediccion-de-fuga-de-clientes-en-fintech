"""Predicción en vivo · ¿Qué predice cada modelo para un cliente?

La vista de la guía (§7.3.3) para los modelos en vivo de cada tarea: el modelo original (BDL) y los ocho del Entregable 3
(LightGBM no se sirve en vivo: solo está en las tablas). El formulario se genera del esquema de entrada de los
artefactos (modelos_vivos.esquema(): las variables con su tipo, su rango de entrenamiento o sus categorías), agrupado por
temas y con las cuatro variables que concentran la señal siempre a la vista; se llena con un cliente de prueba (los casos
del informe, uno al azar o una fila), con la mediana del entrenamiento (la moda en las categóricas y en las de sí o no) o a
mano. Para el modelo elegido se muestran el puntaje, la probabilidad recalibrada y el conjunto conforme (clasificación) o
el pronóstico y el intervalo conforme (regresión), con su cobertura nominal y la medida en prueba; su explicación local
SHAP en una cascada (y, en k-NN, sus vecinos); y las predicciones de todos los modelos en vivo para el mismo cliente.

Con el modelo original se muestra además lo que solo él tiene (el bloque «pr-original»): en regresión, el conjunto de
predicción del modelo, que es una unión de tramos alrededor de los niveles del escalón, con su cobertura nominal y la
medida en prueba, junto al intervalo central y al intervalo simétrico que usan los demás; en clasificación, cómo sale la
probabilidad del índice F(x): cada clase latente aporta su peso si su nivel queda por encima del umbral. Su explicación
local es exacta (RegresorBDL.contribuciones): en regresión descompone el pronóstico y en clasificación, el índice.

Cada entrada se valida con modelos_vivos.validar antes de predecir: un valor vacío, que no es un número o que se sale del
rango del entrenamiento da un aviso con la lista, nunca un error. Los campos numéricos son de texto (se escriben en es-CO o
con punto decimal) y muestran el valor redondeado, pero un campo que nadie tocó se evalúa con el valor exacto del cliente
cargado (la tienda «pr-base»): así la predicción de un cliente de prueba es exactamente la guardada por el proyecto.

Callbacks: _opciones_modelo (los modelos de la tarea), _cargar (llena el formulario: caso, fila, al azar, mediana o
recalcular las variables atadas), _predecir (validación, la predicción del modelo elegido, lo propio del modelo original,
la de todos los modelos y el estado de
las variables atadas) y _explicar (la cascada SHAP y los vecinos de k-NN; va aparte, con su indicador de cálculo, porque
k-NN tarda). Uno más, en el navegador, hace que la dirección siga a la tarea, al modelo, al nivel y al cliente
(?tarea=…&modelo=…&nivel=…&fila=…), que es como enlaza aquí la Tabla de líderes. Predicciones y explicaciones se guardan
en caché por cliente (funciones puras): cambiar de modelo o de nivel no vuelve a cargar las tuberías.
"""
import json
import logging
import math
import re
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

import app
import datos_app as D
import explicacion as E
import formato as F
import modelos_vivos as V
import motor as M
import ui

VISTA = "prediccion"
PRE = "pr"
PANTALLA = "cliente"                 # el icono de la cabecera (el de las vistas de un cliente)
REGISTRO = logging.getLogger(__name__)

TAREA_OMISION = "clasificacion"
O = D.ORIGINAL                       # el modelo original (BDL): con él nace la vista, si ya tiene artefacto en la tarea
MODELO_OMISION = O
MODELO_RESPALDO = "ebm"              # si el original todavía no está en una tarea
NIVEL_OMISION = "0.90"
CASO_OMISION = "riesgo bajo con mayor puntaje"
# Los tres casos del informe (claves de M.meta()["ejemplos"]), con los nombres cortos de la vista «Un cliente»
CASOS = {"riesgo bajo con mayor puntaje": "Falsa alarma", "riesgo alto con menor puntaje": "Se escapa",
         "mayor error de regresión": "Mayor error"}
# Si el modelo pedido no existe en la otra tarea, su análogo (el SVM lineal y el SVR lineal; la logística y Lasso)
ANALOGO = {"logistica": "lasso", "svm": "svr", "naive_bayes": "ebm", "lasso": "logistica", "ridge": "logistica", "svr": "svm"}
CARDINAL = {0: "ninguno", 1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve",
            10: "diez"}
NUMERICOS = ("entera", "real")

N_SENAL = 4                    # las variables que concentran la señal (informe, introducción): siempre a la vista
N_PASOS = 10                   # variables con paso propio en la cascada; las demás van juntas
LISTA_CORTA = 6                # una lista de categorías más larga lleva buscador
DEC_PUNTAJE = 4                # decimales de un puntaje o un pronóstico (los del informe)
DEC_LOGIT = 2                  # decimales de un aporte en logit o en margen
DEC_PROBABILIDAD = 3           # decimales de un aporte en probabilidad o en churn_probability
DEC_TABLA = 4                  # decimales de los aportes en la tabla gemela
DEC_CAMPO = 4                  # decimales con que se escribe un valor pequeño en un campo
DEC_ECE = 3                    # decimales del error de calibración
DEC_DISTANCIA = 2              # decimales de una distancia entre clientes
VALOR_GRANDE = 1000            # desde aquí un valor real se escribe sin decimales
VALOR_MEDIO = 10               # desde aquí, con dos
DEC_MEDIO = 2
LARGO_TEXTO = 40               # caracteres de una entrada inválida que se citan en el aviso
N_VECINOS_TABLA = 5            # vecinos de k-NN que se listan
GROSOR_BARRA = 0.55            # grosor de una barra de la cascada, en filas
AIRE_FILA = 0.6                # aire arriba de la primera fila y debajo de la última, en filas
ROTULO_EJE = 28                # caracteres de un rótulo del eje de la cascada (a 400 px no caben más)
AIRE_EJE = 0.08                # aire a cada lado del recorrido de la cascada (fracción del recorrido)
AIRE_MODELOS = 0.08            # lo mismo en el gráfico de todos los modelos
AIRE_NIVELES = 0.15            # aire a cada lado de los niveles en el gráfico del modelo original (fracción del recorrido)
ANCHO_NIVEL = 0.012            # grosor de la barra de un nivel, como fracción del recorrido del eje
ALTO_TRAMO = 0.5               # alto de la banda de un tramo del conjunto de predicción, como fracción del eje
PASO_PESO = 10                 # marcas del eje del peso de cada nivel, en puntos porcentuales
AIRE_MARCAS = 0.06             # aire bajo el cero de ese eje, como fracción de su tope
DEC_NIVEL = 4                  # decimales de un nivel, de un tramo o del índice
TOLERANCIA_COBERTURA = 0.02    # una cobertura medida que se aparta más que esto de la nominal se señala
PASO_PORCENTAJE = 50           # una marca cada tantos puntos porcentuales en el eje de la probabilidad (a 400 px no caben más)
MARCAS_EJE = 4                 # marcas que se buscan en el eje de la cascada
TOLERANCIA = 1e-9              # dos valores de una variable atada son el mismo si difieren menos que esto
CERCA_DEL_BORDE = 0.001        # una probabilidad más cerca que esto de 0 o de 1 se escribe con dos decimales
DEMORA_CARGA_MS = 300          # el indicador de cálculo solo aparece si la explicación tarda más que esto
EN_CACHE_PUNTAJES = 1024       # puntajes (modelo × cliente) que se recuerdan
EN_CACHE_EXPLICACIONES = 48    # explicaciones que se recuerdan

PARTIDA = "Punto de partida"
SALIDA = "Salida del modelo"
SIMBOLO = {"modelo": "circle", "rival": "circle", "base": "square"}       # el gris y el rosa nunca con la misma forma
FORMA_LEYENDA = {"modelo": "punto", "rival": "punto", "base": "barra"}
BASE_TAREA = {"clasificacion": "Lineales, k-NN y Naive Bayes", "regresion": "Lineales y k-NN"}
# Cómo se calcula la explicación exacta de cada familia (explicacion.py) y qué es su punto de partida
METODO_EXACTO = {
    "bdl": "exacta, con el método del propio modelo (RegresorBDL.contribuciones): el aporte de cada variable es su curva menos la "
           "media de esa curva en las filas de ajuste",
    "ebm": "exacta: la EBM es una suma de funciones de forma y el aporte de cada variable es su término",
    "lineal": "exacta: en un modelo lineal, el aporte de una variable es su coeficiente por su distancia a la media del fondo",
    "bayes": "exacta: el logit de Naive Bayes es una suma de un término por variable",
    "xgboost": "exacta: TreeSHAP de la propia librería XGBoost",
    "arbol": "exacta: TreeSHAP sobre los árboles del modelo",
}

COLUMNAS = tuple(V.columnas())
FUENTES_PREDICCION = ["artifacts/<tarea>/<modelo>/modelo.joblib", "artifacts/<tarea>/<modelo>/metadata.json",
                      D.fuente("calibracion_modelos.json"), D.fuente("conformal.json"), D.fuente("puntajes_prueba.parquet"),
                      D.fuente("pronosticos_prueba.parquet"), D.fuente("clientes_prueba.parquet"), D.fuente("metadatos.json")]
FUENTES_MODELOS = ["artifacts/<tarea>/<modelo>/modelo.joblib", D.fuente("calibracion_modelos.json"), D.fuente("conformal.json"),
                   D.fuente("puntajes_prueba.parquet"), D.fuente("metadatos.json")]
FUENTES_CASCADA = ["artifacts/<tarea>/<modelo>/modelo.joblib", D.fuente("fondo.parquet"), D.fuente("modelos/ebm_clasificacion.json"),
                   D.fuente("modelos/ebm_regresion.json"), "explicacion.py", "src/original_model.py"]
FUENTES_ORIGINAL = [f"artifacts/<tarea>/{O}/modelo.joblib", "src/original_model.py", D.fuente("conformal.json"),
                    D.fuente("calibracion_modelos.json"), D.fuente("metadatos.json")]
FUENTES_VECINOS = ["artifacts/<tarea>/knn/modelo.joblib", D.fuente("eda/entrenamiento.parquet")]
FUENTES_LECTURA = [D.fuente("calibracion_modelos.json"), D.fuente("conformal.json"), D.fuente("puntajes_prueba.parquet"),
                   D.fuente("metadatos.json"), D.fuente("variables.csv"), "artifacts/indice.json"]


def _i(nombre):
    """id de un componente de la vista."""
    return f"{PRE}-{nombre}"


def _id_campo(variable):
    return f"{PRE}-v-{variable}"


# --- El esquema y cómo se escribe cada valor ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _esq():
    """{variable: su entrada de modelos_vivos.esquema()}."""
    return {v["nombre"]: v for v in V.esquema()}


def _nombre(variable):
    e = _esq().get(variable, {})
    return e.get("nombre_visible") or M.nombre(variable)


def _rotulo(e):
    return f"{e['nombre_visible']} ({e['unidad']})" if e.get("unidad") else e["nombre_visible"]


def _python(x):
    """Un escalar de numpy como el tipo de Python que guarda una tienda de Dash."""
    return x.item() if isinstance(x, np.generic) else x


def _decimales(x):
    a = abs(float(x))
    return 0 if a >= VALOR_GRANDE else DEC_MEDIO if a >= VALOR_MEDIO else DEC_CAMPO


def _sin_ceros(s):
    """«0,5000» → «0,5»; «12,00» → «12»."""
    return s.rstrip("0").rstrip(",") if "," in s else s


def _numero(variable, x, redondeo=None):
    """Un valor numérico como se escribe en el campo: entero con punto de miles o decimal en es-CO sin ceros de más.
    `redondeo` «arriba» o «abajo» lleva el límite de un rango hacia adentro, para que el rango escrito no prometa de más."""
    if _esq()[variable]["tipo"] == "entera":
        return F.entero(x)
    d = _decimales(x)
    if redondeo:
        f = 10 ** d
        x = (math.ceil if redondeo == "arriba" else math.floor)(float(x) * f) / f
    return _sin_ceros(ui.num(x, d))


def _rango(variable):
    e = _esq()[variable]
    return _numero(variable, e["min"], "arriba"), _numero(variable, e["max"], "abajo")


def _valor_texto(variable, x):
    """El valor de una variable como se lee en una frase: sí o no, la categoría o el número con su unidad."""
    e = _esq()[variable]
    if e["tipo"] == "booleana":
        return "sí" if bool(x) else "no"
    if e["tipo"] == "categorica":
        return str(x)
    texto = _numero(variable, x)
    return f"{texto} {e['unidad']}" if e.get("unidad") else texto


def _descripcion(e):
    """La ayuda de un campo (title): la descripción del esquema y su rango o sus categorías."""
    texto = e["descripcion"].rstrip(".")
    if e["tipo"] in NUMERICOS:
        lo, hi = _rango(e["nombre"])
        return f"{texto}. En el entrenamiento va de {lo} a {hi}."
    if e["tipo"] == "categorica":
        return f"{texto}. Categorías: {', '.join(e['categorias'])}."
    return f"{texto}."


_MILES = re.compile(r"-?\d\d?\d?(?:\.\d\d\d)+")        # «1.234.567»: grupos de tres cifras separados por puntos


def _leer(texto, tipo):
    """Lo que se escribió en un campo numérico: número (int en una entera si no tiene decimales), None si está vacío o el
    texto tal cual si no es un número. Acepta es-CO («1.234,5») y el punto decimal («0.5»); «1.234» es mil doscientos."""
    if texto is None:
        return None
    if isinstance(texto, (int, float)) and not isinstance(texto, bool):
        x = float(texto)
    else:
        s = str(texto).strip().replace(F.FIJO, "").replace(" ", "").replace(F.MENOS, "-")
        if not s:
            return None
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
        elif _MILES.fullmatch(s) and not s.lstrip("-").startswith("0"):
            s = s.replace(".", "")
        try:
            x = float(s)
        except ValueError:
            return str(texto).strip()
    if not math.isfinite(x):
        # modelos_vivos.validar convierte con float(): «inf» pasaría y en una entera int(inf) lanza una excepción
        return f"«{str(texto).strip()}»"
    return int(x) if tipo == "entera" and x.is_integer() else x


def _es_numero(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --- El cliente cargado (la tienda pr-base) --------------------------------------------------------------------------------
def _textos(valores):
    return {v: _numero(v, x) for v, x in valores.items() if _esq()[v]["tipo"] in NUMERICOS}


def _base(valores, origen):
    """La tienda del cliente cargado: sus valores exactos (los que valen mientras nadie toque el campo), los textos con que
    se escribieron, los valores originales (para saber qué cambió) y de dónde salió."""
    valores = {v: _python(valores[v]) for v in COLUMNAS}
    return {"origen": origen, "valores": valores, "textos": _textos(valores), "original": dict(valores)}


def _base_prueba(fila):
    c = M.clientes().loc[int(fila)]
    return _base({v: c[v] for v in COLUMNAS}, {"tipo": "prueba", "fila": int(fila)})


def _base_mediana():
    """La mediana del entrenamiento en las numéricas (redondeada en las enteras) y la moda en las demás (esquema)."""
    valores = {}
    for v in COLUMNAS:
        e = _esq()[v]
        if e["tipo"] == "entera":
            valores[v] = int(math.floor(float(e["mediana"]) + 0.5))
        elif e["tipo"] == "real":
            valores[v] = float(e["mediana"])
        elif e["tipo"] == "booleana":
            valores[v] = bool(e["moda"])
        else:
            valores[v] = str(e["moda"])
    return _base(valores, {"tipo": "mediana"})


def _valor_campo(base, variable):
    """Lo que muestra el campo de una variable para el cliente cargado."""
    return base["textos"][variable] if _esq()[variable]["tipo"] in NUMERICOS else base["valores"][variable]


def _entrada(campos, base):
    """(fila, cambiadas): la fila de entrada que forman los campos y las variables que difieren del cliente cargado. Un campo
    numérico que nadie tocó vale lo exacto del cliente (el texto es solo su redondeo)."""
    base = base or {}
    exactos, textos, original = base.get("valores") or {}, base.get("textos") or {}, base.get("original") or {}
    fila, cambiadas = {}, []
    for v, x in zip(COLUMNAS, campos):
        tipo = _esq()[v]["tipo"]
        if tipo in NUMERICOS:
            fila[v] = exactos[v] if (v in exactos and x == textos.get(v)) else _leer(x, tipo)
        else:
            fila[v] = x
        if v not in original or fila[v] != original[v]:
            cambiadas.append(v)
    return fila, cambiadas


def _clave(fila):
    """La fila como tupla (variable, valor): la llave de las cachés de puntajes y explicaciones."""
    return tuple((v, fila[v]) for v in COLUMNAS)


def _real(base, cambiadas):
    """La fila del cliente de prueba si se ve tal cual está en los datos (sin cambios); si no, None."""
    origen = (base or {}).get("origen") or {}
    return origen.get("fila") if origen.get("tipo") == "prueba" and not cambiadas else None


# --- Validación -----------------------------------------------------------------------------------------------------------
def _problemas(fila):
    """Los problemas de la fila con sus números en es-CO. Decide modelos_vivos.validar (vacía = válida); aquí solo se
    redactan, porque sus mensajes escriben los rangos con «:g» (3.18215e+06, 0.937546)."""
    crudos = V.validar(fila)
    if not crudos:
        return []
    propios = []
    for v in COLUMNAS:
        e, x = _esq()[v], fila.get(v)
        nombre = _nombre(v)
        if x is None or (isinstance(x, str) and not x.strip()):
            propios.append(f"{nombre}: falta el valor.")
        elif e["tipo"] == "categorica":
            if str(x) not in e["categorias"]:
                propios.append(f"{nombre}: «{str(x)[:LARGO_TEXTO]}» no es una de sus categorías.")
        elif e["tipo"] == "booleana":
            if x not in (True, False):
                propios.append(f"{nombre}: debe ser sí o no.")
        elif isinstance(x, str):
            propios.append(f"{nombre}: «{x.strip('«»')[:LARGO_TEXTO]}» no es un número.")
        else:
            xf = float(x)
            if e["tipo"] == "entera" and not xf.is_integer():
                propios.append(f"{nombre}: debe ser un número entero (se escribió {_sin_ceros(ui.num(xf, DEC_CAMPO))}).")
            elif xf < e["min"] or xf > e["max"]:
                lo, hi = _rango(v)
                escrito = F.entero(xf) if e["tipo"] == "entera" else _sin_ceros(ui.num(xf, max(_decimales(xf), DEC_MEDIO)))
                propios.append(f"{nombre}: {escrito} está fuera del rango del entrenamiento, de {lo} a {hi}.")
    return propios or list(crudos)


def _aviso_problemas(problemas):
    n = len(problemas)
    texto = problemas + ["Los rangos son los de los clientes de entrenamiento: fuera de ellos un modelo extrapola, y la vista no lo "
                         "deja predecir. Nada se calcula hasta que el formulario sea válido."]
    titulo = "Un valor impide predecir" if n == 1 else f"{F.entero(n)} valores impiden predecir"
    return ui.aviso(texto, "atencion", titulo, rol="status")


def _aviso_error(titulo, e):
    detalle = f"{type(e).__name__}: {e}"[:300]
    return ui.aviso([f"La vista sigue funcionando: pruebe con otros valores o recargue la página. Detalle técnico: {detalle}"],
                    "critico", titulo)


# --- Las variables atadas -------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _calculadas():
    """Las variables que motor.coherente recalcula (papel «calculada» de variables.csv), por importancia en la EBM."""
    v = M.variables()
    return tuple(x for x in v[v["papel"] == "calculada"].sort_values("puesto_clf").index if x in COLUMNAS)


ENTRADAS_ATADAS = ("tx_count", "base_satisfaction", "credit_card", "credit_utilization_ratio") + tuple(M.PRODUCTOS)


def _malas_para_atar(fila):
    return [v for v in ENTRADAS_ATADAS
            if not (fila.get(v) in (True, False) if _esq()[v]["tipo"] == "booleana" else _es_numero(fila.get(v)))]


def _texto_atadas():
    return (f"{F.enumerar([_nombre(v) for v in _calculadas()])} se calculan a partir de otras variables, con identidades que se cumplen "
            f"en todos los clientes de los datos.")


def _estado_atadas(fila):
    """Si las variables atadas cuadran con las demás (motor.coherente): un chip; si no, un aviso con lo que no cuadra."""
    malas = _malas_para_atar(fila)
    if malas:
        return ui.nota(f"{_texto_atadas()} Se revisan cuando {F.enumerar([_nombre(v) for v in malas])} tengan un valor válido.")
    esperado = M.coherente(fila)
    distintas = []
    for v in _calculadas():
        x, y = fila.get(v), _python(esperado[v])
        if not _es_numero(x) or abs(float(x) - float(y)) > TOLERANCIA:
            escrito = _valor_texto(v, x) if _es_numero(x) else "un valor no válido"
            distintas.append(f"{_nombre(v)}: {escrito}; según las demás variables, {_valor_texto(v, y)}.")
    if not distintas:
        return html.Div(ui.chip("Las variables atadas cuadran", "bien"), className="chips", title=_texto_atadas())
    return ui.aviso(distintas + ["Los modelos puntúan igual, pero en los datos no hay clientes así. «Recalcular las atadas» las pone de "
                                 "acuerdo con las demás."], "atencion", "Hay variables atadas que no cuadran", rol="status")


def _atar(campos, base):
    """(base nueva, aviso): el cliente del formulario con sus variables atadas recalculadas por motor.coherente."""
    fila, _cambiadas = _entrada(campos, base)
    malas = _malas_para_atar(fila)
    if malas:
        return None, ui.aviso(f"Para recalcularlas, corrija antes {F.enumerar([_nombre(v) for v in malas])}.", "atencion",
                              "No se pueden recalcular las variables atadas", rol="status")
    esperado = M.coherente(fila)
    nueva = json.loads(json.dumps(base))
    for v in _calculadas():
        x = _python(esperado[v])
        x = int(x) if _esq()[v]["tipo"] == "entera" else float(x)
        nueva["valores"][v] = x
        nueva["textos"][v] = _numero(v, x)
    return nueva, None


# --- La dirección y los clientes de prueba -------------------------------------------------------------------------------
def _primero(x):
    return (x[0] if x else None) if isinstance(x, (list, tuple)) else x


def _modelo_de(tarea, modelo):
    """El modelo pedido si es de la tarea; si no, su análogo en ella o el de omisión."""
    modelos = V.modelos(tarea)
    if modelo in modelos:
        return modelo
    if ANALOGO.get(modelo) in modelos:
        return ANALOGO[modelo]
    return next((m for m in (MODELO_OMISION, MODELO_RESPALDO) if m in modelos), modelos[0])


def _nivel(nivel):
    """Un nivel de V.NIVELES a partir de «0.9», «0.90» o «90»; None si no es ninguno."""
    try:
        x = float(str(nivel).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    x = x / 100 if x > 1 else x
    return next((n for n in V.NIVELES if abs(float(n) - x) < TOLERANCIA), None)


def _fila_valida(x):
    try:
        n = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n) or not n.is_integer():
        return None
    return int(n) if int(n) in M.clientes().index else None


@lru_cache(maxsize=1)
def _filas():
    return tuple(sorted(int(f) for f in M.clientes().index))


def _casos():
    """{fila: clave del caso del informe}, con el caso de omisión primero."""
    ejemplos = M.meta()["ejemplos"]
    claves = [CASO_OMISION] + [k for k in ejemplos if k != CASO_OMISION]
    return {int(ejemplos[k]): k for k in claves if k in ejemplos}


def _omision():
    return int(M.meta()["ejemplos"][CASO_OMISION])


def _resolver(tarea, modelo, fila, nivel):
    """(tarea, modelo, nivel, fila, avisos) a partir de la dirección: lo que no existe se cambia por lo de omisión y se dice."""
    tarea, modelo, fila, nivel = (_primero(x) for x in (tarea, modelo, fila, nivel))
    avisos = []
    if tarea not in V.TAREAS:
        if tarea:
            avisos.append(f"«{str(tarea)[:LARGO_TEXTO]}» no es una tarea: se muestra la {D.NOMBRE_TAREA[TAREA_OMISION].lower()}.")
        tarea = TAREA_OMISION
    elegido = _modelo_de(tarea, modelo)
    if modelo and elegido != modelo:
        pedido = D.NOMBRE_MODELO.get(modelo, f"«{str(modelo)[:LARGO_TEXTO]}»")
        avisos.append(f"{pedido} no es uno de los modelos de {D.NOMBRE_TAREA[tarea].lower()}: se muestra {D.en_frase(elegido)}.")
    n = _nivel(nivel) if nivel not in (None, "") else NIVEL_OMISION
    if n is None:
        avisos.append(f"«{str(nivel)[:LARGO_TEXTO]}» no es una de las coberturas nominales: se muestra la de "
                      f"{F.pct(float(NIVEL_OMISION), 0)}.")
        n = NIVEL_OMISION
    elegida = _fila_valida(fila) if fila not in (None, "") else _omision()
    if elegida is None:
        avisos.append(f"«{str(fila).strip()[:LARGO_TEXTO]}» no es la fila de un cliente de prueba (son {F.entero(len(M.clientes()))}): "
                      f"se muestra el caso con que abre la vista ({CASOS[CASO_OMISION].lower()}, fila {_omision()}).")
        elegida = _omision()
    return tarea, elegido, n, elegida, avisos


def _aviso_direccion(avisos):
    if not avisos:
        return None
    return html.Div(ui.aviso(avisos, "atencion", "La dirección pedía algo que no hay", rol="status"), className="aviso-global")


def _texto_origen(base, cambiadas):
    origen = (base or {}).get("origen") or {}
    if origen.get("tipo") == "prueba":
        fila = origen["fila"]
        texto = f"Cliente de prueba de la fila {fila}"
        if fila in _casos():
            texto += f" ({CASOS.get(_casos()[fila], _casos()[fila]).lower()})"
    elif origen.get("tipo") == "mediana":
        texto = "Mediana del entrenamiento (la moda en las categóricas y en las de sí o no)"
    else:
        texto = "Cliente escrito a mano"
    if cambiadas:
        nombres = [_nombre(v) for v in cambiadas]
        lista = F.enumerar(nombres) if len(nombres) <= 3 else f"{', '.join(nombres[:3])} y {F.entero(len(nombres) - 3)} más"
        texto += f", con {F.entero(len(cambiadas))} {'valor cambiado' if len(cambiadas) == 1 else 'valores cambiados'}: {lista}"
    return texto + "."


def _enlace_cliente(base):
    origen = (base or {}).get("origen") or {}
    ruta = app.ruta_vista("cliente")
    return f"{ruta}?fila={origen['fila']}" if origen.get("tipo") == "prueba" else ruta


# --- Puntajes y explicaciones (con caché por cliente; no modificar lo que devuelven) --------------------------------------
@lru_cache(maxsize=EN_CACHE_PUNTAJES)
def _puntaje(tarea, modelo, clave):
    """El puntaje del modelo para el cliente (V.predecir con su artefacto)."""
    return float(V.predecir(tarea, modelo, V.a_tabla(dict(clave)))[0])


def _puntajes(tarea, clave):
    """({modelo: puntaje}, {modelo: error}) de los modelos en vivo de la tarea; uno que falle no tumba a los demás."""
    puntajes, errores = {}, {}
    for m in V.modelos(tarea):
        try:
            puntajes[m] = _puntaje(tarea, m, clave)
        except Exception as e:  # noqa: BLE001  (un artefacto que no carga se informa, no detiene la vista)
            REGISTRO.exception("no se pudo puntuar %s/%s", tarea, m)
            errores[m] = f"{type(e).__name__}: {e}"
    return puntajes, errores


@lru_cache(maxsize=EN_CACHE_EXPLICACIONES)
def _explicacion(tarea, modelo, clave):
    """V.explicar para el cliente, como diccionario de números de Python."""
    e = V.explicar(tarea, modelo, V.a_tabla(dict(clave)))
    aportes = e["aportes"].iloc[0]
    error = e.get("error")
    return {"aportes": {v: float(aportes[v]) for v in COLUMNAS}, "base": float(np.asarray(e["base"])[0]),
            "salida": float(np.asarray(e["salida"])[0]), "metodo": e["metodo"], "escala": e["escala"], "exacto": bool(e["exacto"]),
            "error": None if error is None else {v: float(error.iloc[0][v]) for v in COLUMNAS},
            "eficiencia": float(e.get("eficiencia", float("nan")))}


@lru_cache(maxsize=EN_CACHE_EXPLICACIONES)
def _vecinos(tarea, clave):
    """V.vecinos de k-NN para el cliente: distancias, etiquetas y las variables de los vecinos que se listan."""
    v = V.vecinos(tarea, "knn", V.a_tabla(dict(clave)))
    filas = v["filas"]
    columnas = [c for c in ("fila", "churn_probability", "riesgo_alto") + tuple(M.importancia("clasificacion").index[:N_SENAL])
                if c in filas.columns]
    return {"distancias": tuple(float(x) for x in v["distancias"]), "etiquetas": tuple(float(x) for x in v["etiquetas"]),
            "k": int(v["k"]), "ponderacion": str(v["ponderacion"]), "metrica": str(v["metrica"]),
            "filas": tuple(tuple(_python(x) for x in f) for f in filas[columnas].itertuples(index=False)), "columnas": tuple(columnas)}


@lru_cache(maxsize=EN_CACHE_EXPLICACIONES)
def _distribucion(clave, nivel):
    """V.distribucion para el cliente: lo que el modelo original predice en regresión además de la media."""
    return V.distribucion(V.a_tabla(dict(clave)), nivel)


@lru_cache(maxsize=EN_CACHE_EXPLICACIONES)
def _excedencia(clave):
    """V.excedencia para el cliente, con números de Python: cómo sale la probabilidad del modelo original de su índice."""
    e = V.excedencia(V.a_tabla(dict(clave)))
    if e is None:
        return None
    return {"indice": float(e["indice"][0]), "umbral": e["umbral"], "pesos": [float(x) for x in e["pesos"]],
            "niveles": [float(x) for x in e["niveles"][0]], "aporte": [float(x) for x in e["aporte"][0]],
            "probabilidad": float(e["probabilidad"][0]), "sigma": e["sigma"], "piso": e["piso"], "techo": e["techo"]}


@lru_cache(maxsize=None)
def _ordenados(modelo):
    return np.sort(D.puntajes()[modelo].to_numpy(dtype=float))


def _puesto(modelo, s):
    """Lugar del puntaje entre los clientes de prueba según ese modelo (1 = el de mayor puntaje)."""
    o = _ordenados(modelo)
    return int(len(o) - np.searchsorted(o, s, side="right") + 1)


@lru_cache(maxsize=None)
def _corte(modelo):
    """Puntaje del último cliente que ese modelo contacta con la capacidad del informe (motor.punto_operacion)."""
    s = D.puntajes()[modelo].to_numpy(dtype=float)
    return float(s[M.punto_operacion(s) == 1].min())


@lru_cache(maxsize=None)
def _empate_del_corte(modelo):
    """(clientes de prueba con el puntaje del corte de la campaña, cuántos de ellos entran), con el orden de ese modelo. La
    igualdad es exacta, como en el orden de la campaña y en el análisis de la corrección (empates_campana.json)."""
    s = D.puntajes()[modelo].to_numpy(dtype=float)
    iguales = s == _corte(modelo)
    return int(iguales.sum()), int((iguales & (M.punto_operacion(s) == 1)).sum())


def _en_campana(modelo, s):
    """«entra», «queda fuera» o, si el puntaje es exactamente el del corte y varios clientes de prueba lo comparten, cuántos
    de ellos entran: en un modelo de puntaje escalonado eso lo decide el desempate, no el modelo. La igualdad es exacta, como
    en el orden de la campaña (el puntaje de BDL calculado en vivo es idéntico al guardado). En un modelo sin empate, el
    último contactado entra aunque su puntaje recalculado difiera en el último decimal (la tolerancia de explicacion.py)."""
    corte = _corte(modelo)
    empatados, entran = _empate_del_corte(modelo)
    if s == corte and empatados > 1:
        return (f"empata en el corte: de los {F.entero(empatados)} clientes de prueba con ese puntaje entran {F.entero(entran)}, "
                "según el desempate")
    if empatados <= 1 and abs(s - corte) <= E.TOLERANCIA * max(1.0, abs(corte)):
        return "entra"
    return "entra" if s > corte else "queda fuera"


def _probabilidad(modelo, s):
    return float(np.asarray(V.probabilidad(modelo, s)))


def _pct(p):
    """Una probabilidad recalibrada en porcentaje: el 0 y el 1 de la isotónica se escriben tal cual y lo que está muy cerca
    de ellos lleva un decimal más, para que «0,0 %» no esconda un 0,03 %."""
    if p in (0, 1):
        return F.pct(p, 0)
    return F.pct(p, DEC_MEDIO) if min(p, 1 - p) < CERCA_DEL_BORDE else F.pct(p)


def _ruta_artefacto(tarea, modelo):
    """De dónde sale la predicción: el artefacto (la EBM se evalúa con sus tablas exportadas, idénticas al artefacto; el
    modelo original, con su código de src/original_model.py)."""
    ruta = f"artifacts/{tarea}/{modelo}/modelo.joblib"
    if modelo == O:
        return f"{ruta}, con el código de src/original_model.py"
    return f"{ruta}, evaluado con sus tablas de {D.fuente(f'modelos/ebm_{tarea}.json')}" if modelo == "ebm" else ruta


# --- La predicción del modelo elegido -------------------------------------------------------------------------------------
def _ficha_guardada(tarea, modelo, fila_real, s):
    """Para un cliente de prueba sin cambios: lo guardado por el proyecto y lo observado en los datos."""
    if fila_real is None:
        return []
    tabla = D.puntajes() if tarea == "clasificacion" else D.pronosticos()
    guardado = float(tabla.loc[fila_real, modelo])
    # XGBoost calcula en float32: el guardado y el de ahora difieren en el octavo decimal (la tolerancia de explicacion.py)
    coincide = ("coincide" if abs(guardado - s) <= E.TOLERANCIA * max(1.0, abs(s))
                else f"difiere en {ui.num(abs(guardado - s), DEC_TABLA)}")
    archivo = "puntajes_prueba.parquet" if tarea == "clasificacion" else "pronosticos_prueba.parquet"
    c = M.clientes().loc[fila_real]
    pares = [(f"Guardado por el proyecto ({archivo})", f"{ui.num(guardado, DEC_PUNTAJE)}, {coincide}")]
    if tarea == "clasificacion":
        pares.append(("Riesgo alto en los datos", f"{'sí' if bool(c['riesgo_alto']) else 'no'} (puntaje real "
                                                  f"{ui.num(c['churn_probability'], DEC_PUNTAJE)})"))
    return pares


def _prediccion_clasificacion(modelo, s, nivel, fila_real, clave=None):
    p = _probabilidad(modelo, s)
    conj = V.conjunto(modelo, p, nivel)
    cal = V.calibracion()["modelos"][modelo]
    c_nivel = cal["conformal"][nivel]
    n = len(D.puntajes())
    es_margen = "decision_function" in str(V.metadata("clasificacion", modelo).get("salida", ""))
    margen = "su margen, con decision_function" if es_margen else "predict_proba, sin recalibrar"
    if modelo == O:
        margen = "predict_proba: la probabilidad de excedencia de la mezcla, sin recalibrar"
    clases = conj["clases"]
    if len(clases) == 2:
        lectura = "Con esa cobertura el modelo no descarta ninguna de las dos clases."
    elif len(clases) == 1:
        lectura = f"Con esa cobertura basta una clase: {clases[0]}."
    else:
        lectura = "Con esa cobertura el conjunto queda vacío."
    notas = [f"{lectura} En la prueba, el {F.pct(c_nivel['fraccion_ambos'])} de los clientes recibió las dos clases."]
    if p in (0, 1):
        notas.append(f"La recalibración va por escalones y este es el {'más bajo' if p == 0 else 'más alto'}: entre las predicciones "
                     "fuera de pliegue del entrenamiento con un puntaje así, ningún cliente fue de riesgo "
                     f"{'alto' if p == 0 else 'bajo'}. No es una certeza.")
    if es_margen:
        notas.append("El puntaje del SVM lineal es su margen (la distancia al hiperplano): solo la recalibración lo vuelve una "
                     "probabilidad.")
    if modelo == O:
        notas.append("El puntaje de BDL ya es una probabilidad que sale del modelo (la de superar el umbral, según la mezcla de "
                     "niveles): el bloque de abajo muestra cómo se arma. Se recalibra igual que los demás para compararlos.")
    conjunto = {2: "Las dos clases", 1: f"Solo {clases[0]}" if clases else ""}.get(len(clases), "Ninguna clase")
    cifras = ui.cifras([
        ui.cifra("Probabilidad recalibrada de riesgo alto", _pct(p), clase="cifra--llana", id=_i("probabilidad")),
        ui.cifra("Puntaje del modelo", ui.num(s, DEC_PUNTAJE),
                 "margen" if es_margen else ("probabilidad de excedencia" if modelo == O else "sin recalibrar"),
                 mini=True, clase="cifra--llana", id=_i("puntaje")),
        ui.cifra(f"Conjunto conforme al {F.pct(float(nivel), 0)}", conjunto, mini=True, clase="cifra--llana", id=_i("conjunto"),
                 nota=f"Cobertura nominal {F.pct(float(nivel), 0)}; medida en los {F.entero(n)} clientes de prueba, "
                      f"{F.pct(conj['cobertura_prueba'])}."),
    ], clase="cifras--3")
    puesto = _puesto(modelo, s)
    fraccion = M.meta()["punto_operacion"]["fraccion"]
    pares = [("Puesto en el orden de este modelo", f"{F.entero(puesto)} de {F.entero(n)}"),
             (f"Campaña con capacidad para el {F.pct(fraccion, 0)}", _en_campana(modelo, s)),
             ("Error de calibración en prueba (ECE)",
              (f"{ui.num(cal['ece_prueba_sin_recalibrar'], DEC_ECE)} sin recalibrar; " if cal.get("ece_prueba_sin_recalibrar") is not None
               else "") + f"{ui.num(cal['ece_prueba_recalibrada'], DEC_ECE)} recalibrada")]
    pares += _ficha_guardada("clasificacion", modelo, fila_real, s)
    titulo = f"{D.NOMBRE_MODELO[modelo]}: probabilidad recalibrada de {_pct(p)}"
    cuerpo = [cifras, ui.rejilla([ui.nota(" ".join(notas)), ui.ficha(pares)], "1-1")]
    detalles = [ui.ficha([("Puntaje", f"{_ruta_artefacto('clasificacion', modelo)} ({margen})"),
                          ("Probabilidad recalibrada", f"isotónica de {D.fuente('calibracion_modelos.json')} ({F.entero(cal['nudos'])} nudos)"),
                          ("Conjunto conforme", f"umbral {ui.num(c_nivel['umbral'], DEC_PUNTAJE)} de {D.fuente('calibracion_modelos.json')}"),
                          ("Mejor combinación", _combinacion("clasificacion", modelo))])]
    return titulo, cuerpo, detalles


def _a_tramo(x):
    """La distancia de un puntaje al tramo más cercano, con los decimales de un nivel o, si con esos se leería cero, con los
    que hagan falta: un puntaje real puede quedar fuera de un tramo por menos de una diezmilésima, y con el redondeo
    parecería que está dentro."""
    return ui.num(x, max(DEC_NIVEL, math.ceil(-math.log10(x))) if x > 0 else DEC_NIVEL)


def _prediccion_regresion_original(yhat, nivel, fila_real, clave):
    """La predicción del modelo original en regresión: el pronóstico (la media), el conjunto de predicción del modelo (una
    unión de tramos, con su cobertura nominal y la medida en prueba) y, al lado, el intervalo simétrico de los demás."""
    d = _distribucion(clave, nivel)
    iv = V.intervalo(O, yhat, nivel)
    umbral = M.meta()["etiqueta"]["umbral"]
    n = F.entero(len(D.pronosticos()))
    pct_nivel = F.pct(float(nivel), 0)
    tramos = d["tramos"]
    sobre = sum(a > umbral for a, _b in tramos)
    if sobre == len(tramos):
        lectura = "Todos los tramos del conjunto quedan por encima del umbral de riesgo alto."
    elif sobre == 0:
        lectura = "Todos los tramos del conjunto quedan por debajo del umbral de riesgo alto."
    else:
        lectura = (f"{_tantos(sobre, 'tramo queda', 'tramos quedan').capitalize()} por encima del umbral de riesgo alto y "
                   f"{_cuantos(len(tramos) - sobre, 'queda', 'quedan')} por debajo: con esa cobertura no se puede decir de qué lado queda.")
    medida = (f"medida en los {n} clientes de prueba, {F.pct(d['cobertura_conjunto'])}" if d["cobertura_conjunto"] is not None
              else "sin medida en prueba todavía")
    cifras = ui.cifras([
        ui.cifra("Pronóstico de churn_probability", ui.num(yhat, DEC_PUNTAJE), clase="cifra--llana", id=_i("pronostico"),
                 nota=f"Es la media: el índice del cliente, {ui.num(d['indice'], DEC_NIVEL)}, más el desplazamiento medio."),
        ui.cifra(f"Conjunto de predicción del modelo al {pct_nivel}", _tantos(len(tramos), "tramo", "tramos").capitalize(),
                 f"que suman {ui.num(d['largo'], DEC_NIVEL)}", mini=True, clase="cifra--llana", id=_i("conjunto-original"),
                 nota=f"Cobertura nominal {pct_nivel}; {medida}."),
        ui.cifra(f"Intervalo conforme al {pct_nivel}", F.intervalo(iv["desde"], iv["hasta"], DEC_PUNTAJE), mini=True,
                 clase="cifra--llana", id=_i("intervalo"),
                 nota=f"El pronóstico ± {ui.num(iv['semiancho'], DEC_PUNTAJE)}, como en los demás modelos. Cobertura medida: "
                      f"{F.pct(iv['cobertura_prueba'])}."),
    ], clase="cifras--3")
    pares = [("Tramos", "; ".join(F.intervalo(a, b, DEC_NIVEL) for a, b in tramos)),
             ("Mediana predictiva", ui.num(d["mediana"], DEC_PUNTAJE)),
             ("Intervalo central del modelo", F.intervalo(*d["central"], DEC_PUNTAJE)
              + (f"; cobertura medida {F.pct(d['cobertura_central'])}" if d["cobertura_central"] is not None else "")),
             (f"Umbral de riesgo alto (percentil {ui.num(100 * M.meta()['etiqueta']['q'])} del entrenamiento)", ui.num(umbral, DEC_PUNTAJE))]
    pares += _ficha_guardada("regresion", O, fila_real, yhat)
    if fila_real is not None:
        y = float(M.clientes().loc[fila_real, "churn_probability"])
        if any(a <= y <= b for a, b in tramos):
            lugar = "dentro del conjunto de predicción"
        else:
            lugar = f"fuera del conjunto de predicción, a {_a_tramo(min(max(a - y, y - b) for a, b in tramos))} del tramo más cercano"
        pares.append(("Puntaje real del cliente", f"{ui.num(y, DEC_PUNTAJE)}, {lugar}"))
    notas = [f"{lectura} El conjunto de predicción no es un intervalo: son los tramos alrededor de los niveles en que el modelo sitúa "
             "al cliente. La cobertura nominal es la fracción de clientes cuyo puntaje real debería caer dentro; la medida es la que se "
             "observó en la prueba."]
    if abs(iv["cobertura_prueba"] - float(nivel)) > TOLERANCIA_COBERTURA:
        notas.append(f"Con BDL el intervalo simétrico no cumple su cobertura nominal ({F.pct(iv['cobertura_prueba'])} frente a "
                     f"{pct_nivel}): su error cae en niveles separados y el cuantil queda en el hueco entre dos de ellos. La "
                     "incertidumbre de este modelo se lee en su conjunto de predicción.")
    titulo = (f"{D.NOMBRE_MODELO[O]}: pronóstico de {ui.num(yhat, DEC_PUNTAJE)} y un conjunto de predicción de "
              f"{_tantos(len(tramos), 'tramo', 'tramos')} que suman {ui.num(d['largo'], DEC_NIVEL)}")
    detalles = [ui.ficha([("Pronóstico", f"{_ruta_artefacto('regresion', O)} (predict)"),
                          ("Conjunto de predicción", "RegresorBDL.conjunto_residual: la densidad de la mezcla como medida de conformidad, "
                                                     f"calibrada con las filas de calibración del modelo; cobertura en prueba de {D.fuente('conformal.json')}"),
                          ("Intervalo central", "RegresorBDL.intervalo: cuantiles de los residuos de calibración"),
                          ("Intervalo conforme", f"pronóstico ± el semiancho de {D.fuente('conformal.json')}"),
                          ("Mejor combinación", _combinacion("regresion", O))])]
    return titulo, [cifras, ui.rejilla([html.Div([ui.nota(t) for t in notas], className="rejilla"), ui.ficha(pares)], "1-1")], detalles


def _prediccion_regresion(modelo, yhat, nivel, fila_real, clave=None):
    if modelo == O and clave is not None:
        return _prediccion_regresion_original(yhat, nivel, fila_real, clave)
    iv = V.intervalo(modelo, yhat, nivel)
    umbral = M.meta()["etiqueta"]["umbral"]
    n = len(D.pronosticos())
    if iv["desde"] > umbral:
        lectura = "El intervalo entero queda por encima del umbral de riesgo alto."
    elif iv["hasta"] < umbral:
        lectura = "El intervalo entero queda por debajo del umbral de riesgo alto."
    else:
        lectura = "El intervalo contiene el umbral de riesgo alto: con esa cobertura no se puede decir de qué lado queda."
    cifras = ui.cifras([
        ui.cifra("Pronóstico de churn_probability", ui.num(yhat, DEC_PUNTAJE), clase="cifra--llana", id=_i("pronostico")),
        ui.cifra(f"Intervalo conforme al {F.pct(float(nivel), 0)}", F.intervalo(iv["desde"], iv["hasta"], DEC_PUNTAJE), mini=True,
                 clase="cifra--llana", id=_i("intervalo"), nota=f"El pronóstico ± {ui.num(iv['semiancho'], DEC_PUNTAJE)}."),
        ui.cifra("Cobertura medida en prueba", F.pct(iv["cobertura_prueba"]), mini=True, clase="cifra--llana", id=_i("cobertura"),
                 nota=f"Nominal: {F.pct(float(nivel), 0)}, en los {F.entero(n)} clientes de prueba."),
    ], clase="cifras--3")
    pares = [("Cobertura nominal", F.pct(float(nivel), 0)),
             ("Semiancho del intervalo", f"± {ui.num(iv['semiancho'], DEC_PUNTAJE)}"),
             (f"Umbral de riesgo alto (percentil {ui.num(100 * M.meta()['etiqueta']['q'])} del entrenamiento)", ui.num(umbral, DEC_PUNTAJE))]
    pares += _ficha_guardada("regresion", modelo, fila_real, yhat)
    if fila_real is not None:
        y = float(M.clientes().loc[fila_real, "churn_probability"])
        dentro = iv["desde"] <= y <= iv["hasta"]
        pares.append(("Puntaje real del cliente", f"{ui.num(y, DEC_PUNTAJE)}, {'dentro' if dentro else 'fuera'} del intervalo"))
    titulo = f"{D.NOMBRE_MODELO[modelo]}: pronóstico de {ui.num(yhat, DEC_PUNTAJE)}"
    detalles = [ui.ficha([("Pronóstico", f"{_ruta_artefacto('regresion', modelo)} (predict)"),
                          ("Intervalo conforme", f"pronóstico ± el semiancho de {D.fuente('conformal.json')}"),
                          ("Mejor combinación", _combinacion("regresion", modelo))])]
    nota = ui.nota(f"{lectura} La cobertura nominal es la fracción de clientes cuyo puntaje real debería caer dentro de su intervalo; "
                   "la medida es la que se observó en la prueba.")
    return titulo, [cifras, ui.rejilla([nota, ui.ficha(pares)], "1-1")], detalles


def _combinacion(tarea, modelo):
    m = V.metadata(tarea, modelo)
    externa = m.get("metricas_bucle_externo") or {}
    texto = f"{D.NOMBRE_BALANCEO.get(m.get('balanceo'), m.get('balanceo'))}, {D.NOMBRE_OPTIMIZADOR.get(m.get('optimizador'), m.get('optimizador'))}"
    if externa:
        d = DEC_PUNTAJE if tarea == "clasificacion" else DEC_PUNTAJE + 1
        texto += f"; {externa['metrica']} externa {ui.num(externa['media'], d)} ± {ui.num(externa['desviacion'], d)}"
    return texto


# --- Lo que solo tiene el modelo original: los niveles del escalón --------------------------------------------------------
def _tantos(n, singular, plural):
    """«un tramo», «cuatro tramos» (el número delante de un nombre masculino)."""
    return f"un {singular}" if n == 1 else f"{CARDINAL.get(n, F.entero(n))} {plural}"


def _figura_niveles(niveles, pesos, dentro, umbral, puntos=(), franja=None):
    """Barras de los niveles en que el modelo original sitúa al cliente (x = el nivel del puntaje, y = el peso de su clase
    latente): sólidas las de `dentro` y rayadas las demás; la línea vertical es el umbral de riesgo alto. `puntos` son
    marcas sobre el eje, (x, rol, texto), y `franja`, un intervalo (desde, hasta) que se dibuja detrás."""
    xs = list(niveles) + [umbral] + [p[0] for p in puntos] + (list(franja) if franja else [])
    lo, hi = min(xs), max(xs)
    aire = (hi - lo) * AIRE_NIVELES or AIRE_NIVELES
    lo, hi = lo - aire, hi + aire
    ancho = (hi - lo) * ANCHO_NIVEL
    fig = go.Figure()
    for forma, cuales in (("rayada", [k for k in range(len(niveles)) if not dentro[k]]), ("barra", [k for k in range(len(niveles)) if dentro[k]])):
        if cuales:
            textos = [f"Nivel {ui.num(niveles[k], DEC_NIVEL)}: peso <b>{F.pct(pesos[k])}</b>" for k in cuales]
            fig.add_trace(go.Bar(x=[niveles[k] for k in cuales], y=[100 * pesos[k] for k in cuales],
                                 **ui.traza("modelo", forma, ancho=ancho), **ui.hover_texto(textos)))
    for x, rol, texto in puntos:
        fig.add_trace(go.Scatter(x=[x], y=[0], cliponaxis=False, **ui.traza(rol, "punto"), **ui.hover_texto([texto])))
    if franja:
        ui.franja(fig, *franja)
    ui.referencia(fig, x=umbral)                 # sin rótulo: en el teléfono choca con la unidad del eje; lo nombra la leyenda
    tope = math.ceil(100 * max(pesos) / PASO_PESO) * PASO_PESO
    valores, textos_eje = ui.marcas(0, tope, PASO_PESO)
    ui.unidad_eje(fig, "peso de cada nivel")
    fig.update_layout(hovermode="closest", bargap=0)
    # algo de aire bajo el cero para que las marcas del pronóstico y del puntaje real se vean enteras sobre el eje
    fig.update_yaxes(range=[-tope * AIRE_MARCAS if puntos else 0, tope], tickvals=valores, ticktext=[f"{t}{F.FIJO}%" for t in textos_eje])
    fig.update_xaxes(range=[lo, hi])
    ui.formato_eje(fig, "x", DEC_MEDIO)
    return fig


def _bloque_original(tarea, clave, nivel, fila_real, yhat):
    """El gráfico de lo que solo tiene el modelo original para el cliente del formulario: en clasificación, cómo sale la
    probabilidad del índice; en regresión, el conjunto de predicción frente al intervalo simétrico."""
    umbral_etiqueta = M.meta()["etiqueta"]["umbral"]
    real = None if fila_real is None else float(M.clientes().loc[fila_real, "churn_probability"])
    puntos = [] if real is None else [(real, "real", f"Puntaje real del cliente: <b>{ui.num(real, DEC_PUNTAJE)}</b>")]
    if tarea == "clasificacion":
        e = _excedencia(clave)
        if e is None:
            return ui.aviso("El clasificador final de BDL aprendió de la etiqueta (modo directo): no hay niveles ni mezcla que mostrar; "
                            "su probabilidad es la logística de la suma de sus árboles.", None, "Modo directo")
        niveles, pesos, k = e["niveles"], e["pesos"], len(e["pesos"])
        sobre = [x > e["umbral"] for x in niveles]
        n_sobre = sum(sobre)
        fig = _figura_niveles(niveles, pesos, sobre, e["umbral"], puntos)
        superan = "ninguno supera" if n_sobre == 0 else ("uno supera" if n_sobre == 1 else f"{CARDINAL.get(n_sobre, n_sobre)} superan")
        titulo = (f"El índice del cliente vale {ui.num(e['indice'], DEC_NIVEL)}: de sus {CARDINAL.get(k, k)} niveles, {superan} el umbral, "
                  f"y la probabilidad de riesgo alto es {_pct(e['probabilidad'])}")
        sub = (f"BDL sitúa el puntaje del cliente en uno de {CARDINAL.get(k, k)} niveles: su índice más el desplazamiento de cada clase "
               f"latente. Cada nivel aporta su peso si queda por encima del umbral ({ui.num(e['umbral'], DEC_NIVEL)}): la probabilidad es "
               "la suma de los pesos de las barras sólidas.")
        filas = [[F.entero(i + 1), ui.num(niveles[j], DEC_NIVEL), F.pct(pesos[j]), "sí" if sobre[j] else "no", _pct(e["aporte"][j])]
                 for i, j in enumerate(sorted(range(k), key=lambda j: -niveles[j]))]
        tabla = (["Clase latente", "Nivel: índice más desplazamiento", "Peso", "Supera el umbral", "Aporta a la probabilidad"], filas)
        nota = (f"La transición en el umbral se suaviza con la dispersión robusta del índice ({ui.num(e['sigma'], 5)}); la probabilidad "
                f"nunca baja de {_pct(e['piso'])} ni pasa de {_pct(e['techo'])}, lo que permite comprobar la muestra de calibración. La "
                "cascada de abajo explica el índice: qué variables lo suben o lo bajan.")
        leyenda = [("Nivel sobre el umbral: aporta su peso", "modelo", "barra"), ("Nivel bajo el umbral", "modelo", "rayada"),
                   ("Umbral de riesgo alto", "referencia", "linea")]
        etiqueta = f"{titulo}. " + "; ".join(f"nivel {f[1]}, peso {f[2]}, {'sobre' if f[3] == 'sí' else 'bajo'} el umbral" for f in filas) + "."
    else:
        d = _distribucion(clave, nivel)
        iv = V.intervalo(O, yhat, nivel)
        niveles, pesos, k, tramos = d["niveles"], d["pesos"], len(d["pesos"]), d["tramos"]
        en_conjunto = [any(a <= x <= b for a, b in tramos) for x in niveles]
        pct_nivel = F.pct(float(nivel), 0)
        puntos = [(yhat, "modelo", f"Pronóstico (la media): <b>{ui.num(yhat, DEC_PUNTAJE)}</b>")] + puntos
        fig = _figura_niveles(niveles, pesos, en_conjunto, umbral_etiqueta, puntos, (iv["desde"], iv["hasta"]))
        titulo = (f"BDL sitúa al cliente en {CARDINAL.get(k, k)} niveles: al {pct_nivel}, su conjunto de predicción son "
                  f"{_tantos(len(tramos), 'tramo', 'tramos')} que suman {ui.num(d['largo'], DEC_NIVEL)}, frente a un intervalo simétrico "
                  f"de {ui.num(iv['ancho'] if 'ancho' in iv else 2 * iv['semiancho'], DEC_NIVEL)}")
        medida = "" if d["cobertura_conjunto"] is None else (
            f" En los {F.entero(len(D.pronosticos()))} clientes de prueba el conjunto cubrió el {F.pct(d['cobertura_conjunto'])} y el "
            f"intervalo simétrico, el {F.pct(iv['cobertura_prueba'])}.")
        sub = (f"Cada barra es un nivel posible del puntaje (el índice del cliente más el desplazamiento de una clase latente) con su "
               f"peso. Las sólidas forman el conjunto de predicción al {pct_nivel}; la franja es el intervalo simétrico que usan los "
               f"demás modelos (pronóstico ± {ui.num(iv['semiancho'], DEC_PUNTAJE)}).{medida}")
        filas = []
        for i, j in enumerate(sorted(range(k), key=lambda j: -niveles[j])):
            tramo = next((F.intervalo(a, b, DEC_NIVEL) for a, b in tramos if a <= niveles[j] <= b), F.RAYA)
            filas.append([F.entero(i + 1), ui.num(niveles[j], DEC_NIVEL), F.pct(pesos[j]), "sí" if en_conjunto[j] else "no", tramo])
        tabla = (["Clase latente", "Nivel: índice más desplazamiento", "Peso", f"En el conjunto al {pct_nivel}", "Tramo"], filas)
        ancho = max(b - a for a, b in tramos)
        nota = ("El pronóstico es la media de los niveles y suele caer entre dos de ellos, donde el modelo no espera al cliente: por eso "
                "un intervalo alrededor de la media es ancho y el conjunto de predicción, corto. El tramo más ancho mide "
                f"{ui.num(ancho, DEC_NIVEL)}"
                + (f": {ui.num(ancho / d['sigma'], 1)} veces la dispersión dentro de cada nivel ({ui.num(d['sigma'], 5)})." if d["sigma"] > 0 else "."))
        leyenda = [("Nivel dentro del conjunto de predicción", "modelo", "barra"), ("Nivel fuera del conjunto", "modelo", "rayada"),
                   ("Pronóstico (la media)", "modelo", "punto"), ("Umbral de riesgo alto", "referencia", "linea"),
                   ("Intervalo simétrico (pronóstico ± q)", "franja", "barra")]
        etiqueta = f"{titulo}. " + "; ".join(f"nivel {f[1]}, peso {f[2]}" for f in filas) + "."
    if real is not None:
        leyenda.append(("Puntaje real del cliente", "real", "punto"))
    return ui.grafico(fig, titulo, sub, etiqueta, tabla=tabla, leyenda=leyenda, nota=nota, alto="bajo", id=_i("niveles"),
                      fuentes=FUENTES_ORIGINAL)


# --- Todos los modelos en vivo ante el mismo cliente ----------------------------------------------------------------------
def _marca(rol):
    estilo = ui.traza(rol, "punto")
    estilo["marker"] = dict(estilo["marker"], symbol=SIMBOLO[rol])
    return estilo


def _bigotes(valor, inf, sup):
    """Barras de error asimétricas de un intervalo [inf; sup] (en tinta: el color lo lleva el punto)."""
    return {"type": "data", "symmetric": False, "array": [s - x for s, x in zip(sup, valor)],
            "arrayminus": [x - i for x, i in zip(valor, inf)], "color": ui.ROL["tinta_2"], "thickness": ui.ANCHO_LINEA}


def _lienzo_modelos(fig, nombres, lo, hi):
    aire = (hi - lo) * AIRE_MODELOS
    fig.update_layout(hovermode="closest", margin_r=ui.MARGEN_ETIQUETAS + ui.px("--e-4"))
    fig.update_xaxes(range=[lo - aire, hi + aire], showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=nombres[::-1], tickfont=ui.fuente_svg("tinta_2", "--t-s"))


def _valores_al_borde(fig, nombres, textos, foco=None):
    for nombre, texto in zip(nombres, textos):
        fig.add_annotation(x=1, xref="paper", y=nombre, xanchor="left", xshift=ui.px("--e-2"), text=texto, showarrow=False,
                           font=ui.fuente_svg("tinta" if nombre == foco else "tinta_2"))


def _cuantos(n, singular, plural):
    return f"{CARDINAL.get(n, F.entero(n))} {singular if n == 1 else plural}"


def _leyenda_modelos(tarea):
    """La leyenda del gráfico de todos los modelos: el original (si está en la tarea), los rivales y la base."""
    original = [(D.NOMBRE_MODELO[O], "modelo", FORMA_LEYENDA["modelo"])] if V.hay_original(tarea) else []
    return original + [("EBM y modelos de árboles", "rival", FORMA_LEYENDA["rival"]), (BASE_TAREA[tarea], "base", FORMA_LEYENDA["base"])]


def _modelos(tarea, modelo, puntajes, errores, nivel):
    """Figura, título, subtítulo, leyenda, tabla, nota y etiqueta del gráfico de todos los modelos en vivo de la tarea."""
    orden = [m for m in V.modelos(tarea) if m in puntajes]
    nombres = [D.NOMBRE_CORTO[m] + (" · elegido" if m == modelo else "") for m in orden]
    nombre_de = dict(zip(orden, nombres))
    pct_nivel = F.pct(float(nivel), 0)
    fig = go.Figure()
    if tarea == "clasificacion":
        p = {m: _probabilidad(m, puntajes[m]) for m in orden}
        conj = {m: V.conjunto(m, p[m], nivel) for m in orden}
        for rol in ("base", "rival", "modelo"):
            ms = [m for m in orden if D.ROL_MODELO[m] == rol]
            if ms:
                textos = [f"{D.NOMBRE_MODELO[m]}: <b>{_pct(p[m])}</b> recalibrada · puntaje {ui.num(puntajes[m], DEC_PUNTAJE)} · "
                          f"conjunto al {pct_nivel}: {', '.join(conj[m]['clases'])}" for m in ms]
                fig.add_trace(go.Scatter(x=[100 * p[m] for m in ms], y=[nombre_de[m] for m in ms], **_marca(rol), **ui.hover_texto(textos)))
        prevalencia = M.meta()["prevalencia_entrenamiento"]
        ui.referencia(fig, x=100 * prevalencia)
        ui.unidad_eje(fig, "Probabilidad recalibrada de riesgo alto")
        _lienzo_modelos(fig, nombres, 0, 100)
        valores, textos_eje = ui.marcas(0, 100, PASO_PORCENTAJE)
        fig.update_xaxes(tickvals=valores, ticktext=[f"{t}{F.FIJO}%" for t in textos_eje])
        _valores_al_borde(fig, nombres, [_pct(p[m]) for m in orden], foco=nombre_de.get(modelo))
        bajo, alto = min(orden, key=lambda m: p[m]), max(orden, key=lambda m: p[m])
        tipos = {"alto": [m for m in orden if conj[m]["clases"] == ["riesgo alto"]],
                 "ambos": [m for m in orden if len(conj[m]["clases"]) == 2],
                 "bajo": [m for m in orden if conj[m]["clases"] == ["riesgo bajo"]]}
        partes = [f"{_cuantos(len(tipos['alto']), 'lo deja', 'lo dejan')} solo en riesgo alto" if tipos["alto"] else None,
                  f"{_cuantos(len(tipos['ambos']), 'da', 'dan')} las dos clases" if tipos["ambos"] else None,
                  f"{_cuantos(len(tipos['bajo']), 'lo deja', 'lo dejan')} solo en riesgo bajo" if tipos["bajo"] else None]
        titulo = (f"La probabilidad recalibrada va de {_pct(p[bajo])} ({D.en_frase(bajo)}) a {_pct(p[alto])} ({D.en_frase(alto)}); "
                  f"en su conjunto al {pct_nivel}, {F.enumerar([x for x in partes if x])}")
        sub = (f"Para el mismo cliente, la probabilidad recalibrada de cada modelo: los puntajes crudos no se comparan entre modelos. "
               f"La línea vertical es la prevalencia de riesgo alto en el entrenamiento ({F.pct(prevalencia)}).")
        encabezados = ["Modelo", "Puntaje del modelo", "Probabilidad recalibrada", f"Conjunto al {pct_nivel}", "Cobertura medida",
                       "Puesto en prueba"]
        filas = [[D.NOMBRE_MODELO[m], ui.num(puntajes[m], DEC_PUNTAJE), _pct(p[m]), ", ".join(conj[m]["clases"]),
                  F.pct(conj[m]["cobertura_prueba"]), F.entero(_puesto(m, puntajes[m]))] for m in orden]
        num = [False, True, True, False, True, True]
        ocultar = (1, 4, 5)
        etiqueta_valores = [f"{D.NOMBRE_MODELO[m]}: {_pct(p[m])}" for m in orden]
    else:
        iv = {m: V.intervalo(m, puntajes[m], nivel) for m in orden}
        umbral = M.meta()["etiqueta"]["umbral"]
        for rol in ("base", "rival", "modelo"):
            ms = [m for m in orden if D.ROL_MODELO[m] == rol]
            if ms:
                textos = [f"{D.NOMBRE_MODELO[m]}: <b>{ui.num(puntajes[m], DEC_PUNTAJE)}</b> · intervalo al {pct_nivel} "
                          f"{F.intervalo(iv[m]['desde'], iv[m]['hasta'], DEC_PUNTAJE)}" for m in ms]
                fig.add_trace(go.Scatter(x=[puntajes[m] for m in ms], y=[nombre_de[m] for m in ms], **_marca(rol),
                                         error_x=_bigotes([puntajes[m] for m in ms], [iv[m]["desde"] for m in ms], [iv[m]["hasta"] for m in ms]),
                                         **ui.hover_texto(textos)))
        ui.referencia(fig, x=umbral)
        ui.unidad_eje(fig, "churn_probability pronosticada")
        lo = min([iv[m]["desde"] for m in orden] + [umbral])
        hi = max([iv[m]["hasta"] for m in orden] + [umbral])
        _lienzo_modelos(fig, nombres, lo, hi)
        ui.formato_eje(fig, "x", DEC_MEDIO)
        _valores_al_borde(fig, nombres, [ui.num(puntajes[m], DEC_PROBABILIDAD) for m in orden], foco=nombre_de.get(modelo))
        bajo, alto = min(orden, key=lambda m: puntajes[m]), max(orden, key=lambda m: puntajes[m])
        sobre = [m for m in orden if iv[m]["desde"] > umbral]
        debajo = [m for m in orden if iv[m]["hasta"] < umbral]
        contienen = [m for m in orden if m not in sobre and m not in debajo]
        partes = [f"{_cuantos(len(sobre), 'intervalo queda entero', 'intervalos quedan enteros')} sobre el umbral de riesgo alto" if sobre else None,
                  f"{_cuantos(len(contienen), 'lo contiene', 'lo contienen')}" if contienen else None,
                  f"{_cuantos(len(debajo), 'queda', 'quedan')} por debajo" if debajo else None]
        titulo = (f"Los pronósticos van de {ui.num(puntajes[bajo], DEC_PROBABILIDAD)} ({D.en_frase(bajo)}) a "
                  f"{ui.num(puntajes[alto], DEC_PROBABILIDAD)} ({D.en_frase(alto)}); al {pct_nivel}, {F.enumerar([x for x in partes if x])}")
        sub = (f"Pronóstico de cada modelo con su intervalo conforme al {pct_nivel}. La línea vertical es el umbral de riesgo alto "
               f"({ui.num(umbral, DEC_PUNTAJE)}, el percentil {ui.num(100 * M.meta()['etiqueta']['q'])} del entrenamiento).")
        encabezados = ["Modelo", "Pronóstico", f"Intervalo al {pct_nivel}", "Cobertura medida", "Respecto del umbral"]
        filas = [[D.NOMBRE_MODELO[m], ui.num(puntajes[m], DEC_PUNTAJE), F.intervalo(iv[m]["desde"], iv[m]["hasta"], DEC_PUNTAJE),
                  F.pct(iv[m]["cobertura_prueba"]), "por encima" if m in sobre else "por debajo" if m in debajo else "lo contiene"]
                 for m in orden]
        num = [False, True, True, True, False]
        ocultar = (3,)
        etiqueta_valores = [f"{D.NOMBRE_MODELO[m]}: {ui.num(puntajes[m], DEC_PUNTAJE)}" for m in orden]
    titulo = titulo[0].upper() + titulo[1:]
    tabla = ui.tabla(encabezados, filas, compacta=True, muestras=[D.ROL_MODELO[m] for m in orden], num=num, ocultar_movil=ocultar)
    nota = (f"Los {CARDINAL.get(len(orden), F.entero(len(orden)))} se calculan en el momento con sus artefactos; la explicación local "
            "de arriba es solo la del modelo elegido. LightGBM, el otro benchmark reciente, no se sirve en vivo: está en la tabla de "
            "líderes y en la comparación estadística.")
    if errores:
        nota += " No se pudo calcular: " + "; ".join(f"{D.NOMBRE_MODELO[m]} ({t[:80]})" for m, t in errores.items()) + "."
    etiqueta = f"{titulo}. " + "; ".join(etiqueta_valores) + "."
    return fig, titulo, sub, ui.leyenda(_leyenda_modelos(tarea)), tabla, nota, etiqueta


# --- La explicación local (cascada) ---------------------------------------------------------------------------------------
ESCALA_REGRESION = "churn_probability predicha"
ESCALA_LOGIT = "logit del puntaje del modelo"
ESCALA_MARGEN = "margen del SVM lineal (no es una probabilidad)"
ESCALA_PUNTAJE = "puntaje del modelo (probabilidad sin recalibrar)"
ESCALA_INDICE = "índice del puntaje, sin el escalón latente"
# Lo que dice la unidad del eje de la cascada (corta: a 400 px va junto a los rótulos y una más larga se corta en el borde)
UNIDAD_ESCALA = {ESCALA_REGRESION: "Puntaje predicho", ESCALA_LOGIT: "Logit del puntaje", ESCALA_MARGEN: "Margen del SVM",
                 ESCALA_INDICE: "Índice del puntaje",
                 ESCALA_PUNTAJE: "Puntaje sin recalibrar"}


def _escala(tarea, escala):
    """(nombre de la escala para el texto, es logit, decimales). El puntaje crudo nunca se llama probabilidad."""
    if tarea == "regresion":
        return ESCALA_REGRESION, False, DEC_PROBABILIDAD
    if escala == E.ESCALA_INDICE:                 # el modelo original: lo que suman sus árboles es el índice, no una probabilidad
        return ESCALA_INDICE, False, DEC_PROBABILIDAD
    if escala.startswith("logit"):
        return ESCALA_LOGIT, True, DEC_LOGIT
    if escala.startswith("margen"):
        return ESCALA_MARGEN, False, DEC_LOGIT
    return ESCALA_PUNTAJE, False, DEC_PROBABILIDAD


def _rotulo_eje(variable, x):
    """«Edad = 60 años» si cabe en ROTULO_EJE caracteres; si no, solo el nombre (recortado si hace falta). El valor completo
    va en la etiqueta flotante y en la tabla gemela."""
    completo, nombre = f"{_nombre(variable)} = {_valor_texto(variable, x)}", _nombre(variable)
    if len(completo) <= ROTULO_EJE:
        return completo
    return nombre if len(nombre) <= ROTULO_EJE else nombre[:ROTULO_EJE - 1].rstrip() + "…"


def _texto_metodo(exp):
    """Cómo se calculó la explicación, según los campos «exacto» y «error» del resultado."""
    if exp["exacto"]:
        return METODO_EXACTO.get(exp["metodo"], "exacta")
    muestreo = E.MUESTREO.get(exp["metodo"])
    texto = "estimada por permutaciones antitéticas"
    if muestreo:
        texto += f" ({F.entero(muestreo[0])} permutaciones contra {F.entero(muestreo[1])} clientes de entrenamiento del fondo)"
    if exp["error"]:
        texto += f"; el mayor error de Monte Carlo de un aporte es ± {ui.num(max(exp['error'].values()), DEC_TABLA)}"
    return texto


def _texto_partida(exp):
    if exp["metodo"] == "ebm":
        return "el intercepto de la EBM"
    if exp["metodo"] == "bdl":
        return "la salida media del modelo en sus filas de ajuste"
    if exp["metodo"] == "xgboost":
        return "el valor esperado del modelo según sus árboles"
    if exp["exacto"]:
        return f"la salida media del modelo en los {F.entero(len(V.fondo()))} clientes de entrenamiento del fondo"
    muestreo = E.MUESTREO.get(exp["metodo"])
    return (f"la salida media del modelo en los {F.entero(muestreo[1])} clientes del fondo" if muestreo
            else "la salida media del modelo en el fondo")


def _cascada(tarea, modelo, fila, exp, s):
    """Figura, título, subtítulo, leyenda, tabla, nota y etiqueta de la cascada: del punto de partida a la salida, con un
    paso por cada una de las N_PASOS variables de mayor aporte absoluto y otro con las demás."""
    a, err = exp["aportes"], exp["error"]
    orden = sorted(a, key=lambda v: -abs(a[v]))
    propias, otras = orden[:N_PASOS], orden[N_PASOS:]
    pasos = [a[v] for v in propias] + ([sum(a[v] for v in otras)] if otras else [])
    inicio, final = exp["base"], exp["salida"]
    llegadas = [float(x) for x in inicio + np.cumsum(pasos)]
    salidas = [inicio] + llegadas[:-1]
    escala, es_logit, d = _escala(tarea, exp["escala"])
    # Un rótulo por fila en el eje (cabe también a 400 px); el valor de la variable va si el rótulo no se alarga de más
    rotulos = ([PARTIDA] + [f"{_nombre(v)} = {_valor_texto(v, fila[v])}" for v in propias]
               + ([f"Las otras {F.entero(len(otras))} variables"] if otras else []) + [SALIDA])
    en_eje = [PARTIDA] + [_rotulo_eje(v, fila[v]) for v in propias] + rotulos[len(propias) + 1:]
    valores = [ui.num(inicio, d)] + [ui.con_signo(p, d) for p in pasos] + [ui.num(final, d)]
    n = len(rotulos)
    rol = D.ROL_MODELO.get(modelo, "contexto")

    def mas_menos(v):
        return f" ± {ui.num(err[v], d + 1)}" if err else ""
    textos = ([f"{r}: <b>{x}</b>{mas_menos(v)} · va en {ui.num(ll, d)}" for r, x, v, ll in zip(rotulos[1:], valores[1:], propias, llegadas)]
              + ([f"{rotulos[-2]}: <b>{valores[-2]}</b> · va en {ui.num(llegadas[-1], d)}"] if otras else []))
    fig = go.Figure()
    fig.add_trace(go.Bar(y=list(range(1, n - 1)), x=pasos, base=salidas, orientation="h", name="Aporte de cada variable",
                         **ui.traza(rol, "rayada", ancho=GROSOR_BARRA), **ui.hover_texto(textos)))
    fig.add_trace(go.Scatter(x=[inicio, final], y=[0, n - 1], name="Punto de partida y salida", **ui.traza(rol, "punto"),
                             **ui.hover_texto([f"{PARTIDA}: <b>{valores[0]}</b>", f"{SALIDA}: <b>{valores[-1]}</b>"])))
    posiciones = [inicio, final] + llegadas
    lo, hi = min(posiciones), max(posiciones)
    recorrido = max(hi - lo, abs(hi) * AIRE_EJE, TOLERANCIA)
    lo, hi = lo - recorrido * AIRE_EJE, hi + recorrido * AIRE_EJE
    paso = ui.paso_redondo(hi - lo, MARCAS_EJE)
    marcas, textos_eje = ui.marcas(lo, hi, paso)
    dentro = [(m, t) for m, t in zip(marcas, textos_eje) if lo <= m <= hi]
    marcas, textos_eje = [m for m, _t in dentro], [t for _m, t in dentro]
    if es_logit:
        textos_eje = [f"{t}<br>{ui.num(M.sigmoide(m), DEC_MEDIO)}" for m, t in zip(marcas, textos_eje)]
    ui.unidad_eje(fig, UNIDAD_ESCALA[escala])
    if lo < 0 < hi:
        ui.linea_cero(fig, "x")
    for i, valor in enumerate(valores):                      # el valor de cada paso, en el margen derecho
        fig.add_annotation(x=1, xref="paper", y=i, yref="y", xanchor="left", xshift=ui.px("--e-2"), text=valor, showarrow=False,
                           font=ui.fuente_svg("tinta" if i in (0, n - 1) else "tinta_2"))
    fig.update_layout(barmode="overlay", hovermode="closest", margin_r=ui.MARGEN_ETIQUETAS)
    fig.update_xaxes(range=[lo, hi], tickvals=marcas, ticktext=textos_eje, showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_yaxes(range=[n - 1 + AIRE_FILA, -AIRE_FILA], tickvals=list(range(n)), ticktext=en_eje, showgrid=False,
                     tickfont=ui.fuente_svg("tinta_2"), ticklabelstandoff=ui.px("--e-2"))

    principales = [f"{_nombre(v)} ({ui.con_signo(a[v], d)})" for v in propias[:2]]
    titulo = (f"De {ui.num(inicio, d)} a {ui.num(final, d)} en la escala del modelo; lo que más aporta: {F.enumerar(principales)}")
    if escala == ESCALA_INDICE:
        salida_txt = (f"La salida es el índice del cliente, {ui.num(final, DEC_PUNTAJE)}: la probabilidad, {ui.num(s, DEC_PUNTAJE)}, sale "
                      "de comparar con el umbral cada nivel (el índice más un desplazamiento), como muestra el gráfico de arriba.")
    else:
        salida_txt = (f"La salida, {ui.num(final, d)}, es el logit de su puntaje, {ui.num(s, DEC_PUNTAJE)}; bajo cada marca del eje, el "
                      "puntaje de ese logit." if es_logit
                      else f"La salida es el {'pronóstico' if tarea == 'regresion' else 'puntaje'}, {ui.num(final, DEC_PUNTAJE)}.")
    sub = (f"Explicación local de {D.en_frase(modelo)}, {_texto_metodo(exp)}. Punto de partida: {_texto_partida(exp)}; cada variable suma "
           f"o resta su aporte, en {escala}. {salida_txt}")
    leyenda = ui.leyenda([("Punto de partida y salida", rol, "punto"), ("Aporte de cada variable", rol, "rayada")])
    encabezados = ["Variable", "Valor", "Aporte"] + (["Error de Monte Carlo"] if err else [])
    filas = [[PARTIDA, F.RAYA, ui.num(inicio, DEC_TABLA)] + ([F.RAYA] if err else [])]
    filas += [[_nombre(v), _valor_texto(v, fila[v]), ui.con_signo(a[v], DEC_TABLA)] + ([f"± {ui.num(err[v], DEC_TABLA)}"] if err else [])
              for v in orden]
    tabla = ui.tabla(encabezados, filas, total=[SALIDA, "", ui.num(final, DEC_TABLA)] + ([""] if err else []), compacta=True,
                     num=[False, False, True] + ([True] if err else []), ocultar_movil=(3,) if err else ())
    nota = (f"Las {F.entero(len(propias))} variables de mayor aporte absoluto tienen paso propio; las otras van juntas. El punto de partida "
            f"más los aportes de las {F.entero(len(a))} variables da la salida. Los aportes dicen cuánto mueve cada variable la salida de "
            "este modelo para este cliente: no son efectos causales ni se comparan en magnitud entre modelos con otra escala.")
    etiqueta = f"{titulo}. " + "; ".join(f"{r}: {x}" for r, x in zip(rotulos, valores)) + "."
    return fig, titulo, sub, leyenda, tabla, nota, etiqueta


def _tarjeta_vecinos(tarea, clave, s):
    """k-NN: la predicción es el promedio ponderado de las etiquetas de sus k vecinos más parecidos del entrenamiento."""
    v = _vecinos(tarea, clave)
    d, y = np.asarray(v["distancias"]), np.asarray(v["etiquetas"])
    if v["ponderacion"] == "distance":
        w = (d == 0).astype(float) if (d == 0).any() else 1 / d
    else:
        w = np.ones_like(d)
    promedio = float((w * y).sum() / w.sum())
    k = v["k"]
    clas = tarea == "clasificacion"
    if clas:
        titulo = (f"k-NN promedia a sus {F.entero(k)} vecinos más parecidos del entrenamiento: el {F.pct(promedio)} de su peso es de "
                  "riesgo alto")
    else:
        titulo = (f"k-NN promedia a sus {F.entero(k)} vecinos más parecidos del entrenamiento: su churn_probability media ponderada es "
                  f"{ui.num(promedio, DEC_PUNTAJE)}")
    pares = [("Vecinos que promedia (k)", F.entero(k)),
             ("Peso de cada vecino", "el inverso de su distancia" if v["ponderacion"] == "distance" else "el mismo para todos"),
             ("Distancia", f"{v['metrica']}, sobre las variables escaladas y codificadas"),
             ("Distancia al más cercano y al más lejano", f"{ui.num(d.min(), DEC_DISTANCIA)} y {ui.num(d.max(), DEC_DISTANCIA)}")]
    if clas:
        pares.append(("Vecinos de riesgo alto, sin ponderar", f"{F.entero(int(y.sum()))} de {F.entero(k)}"))
    pares += [("Promedio ponderado de sus etiquetas", ui.num(promedio, DEC_PUNTAJE)),
              ("Puntaje del modelo" if clas else "Pronóstico del modelo", ui.num(s, DEC_PUNTAJE))]
    cols = list(v["columnas"])
    otras = [c for c in cols if c not in ("fila", "churn_probability", "riesgo_alto")]
    encabezados = ["Vecino", "Distancia", "Riesgo alto" if clas else "churn_probability"] + [_nombre(c) for c in otras]
    filas = []
    for i, f in enumerate(v["filas"][:N_VECINOS_TABLA]):
        r = dict(zip(cols, f))
        etiqueta = ("sí" if y[i] else "no") if clas else ui.num(y[i], DEC_PUNTAJE)
        filas.append([f"Fila {r['fila']}" if "fila" in r else F.entero(i + 1), ui.num(d[i], DEC_DISTANCIA), etiqueta]
                     + [_valor_texto(c, r[c]) for c in otras])
    tabla = ui.tabla(encabezados, filas, compacta=True, num=[False, True, not clas] + [False] * len(otras),
                     ocultar_movil=tuple(range(3, len(encabezados))))
    mitad = (len(pares) + 1) // 2
    return ui.tarjeta([
        ui.rejilla([ui.ficha(pares[:mitad]), ui.ficha(pares[mitad:])], "1-1"),
        html.Div([html.P(f"Los {F.entero(min(N_VECINOS_TABLA, k))} vecinos más cercanos", className="campo__rotulo"), tabla],
                 className="campo-bloque"),
        ui.nota("La predicción de k-NN es el promedio ponderado de las etiquetas de los k vecinos más parecidos del entrenamiento (en "
                "clasificación, la parte de su peso que es de riesgo alto). No hay una fórmula de aportes por variable: por eso su "
                "explicación se estima."),
    ], titulo=titulo, fuentes=FUENTES_VECINOS, id=_i("vecinos-tarjeta"))


# --- Pantalla -------------------------------------------------------------------------------------------------------------
def _opciones(tarea):
    return [{"label": D.NOMBRE_MODELO[m], "value": m} for m in V.modelos(tarea)]


def _controles(tarea, modelo, nivel):
    return [ui.control("Tarea:", ui.segmentado(_i("tarea"), [{"label": D.NOMBRE_TAREA[t], "value": t} for t in V.TAREAS], tarea,
                                               persistir=False)),
            ui.control("Modelo:", ui.selector(_i("modelo"), _opciones(tarea), modelo, persistir=False)),
            ui.control("Cobertura nominal:", ui.segmentado(_i("nivel"), [{"label": F.pct(float(n), 0), "value": n} for n in V.NIVELES],
                                                           nivel, persistir=False))]


def _subtitulo():
    original = (" El modelo original, BDL, muestra además su conjunto de predicción y cómo llega a su probabilidad."
                if any(V.hay_original(t) for t in V.TAREAS) else "")
    return (f"Los {F.entero(len(V.indice()['artefactos']))} modelos finales, cargados de artifacts/, predicen en el momento para el "
            f"cliente del formulario, con su incertidumbre medida en los {F.entero(M.meta()['n_prueba'])} clientes de prueba y su "
            f"explicación local.{original}")


def _campo(variable, base):
    """El control de una variable según su tipo en el esquema: texto numérico con su rango, sí o no, o lista de categorías."""
    e = _esq()[variable]
    cid = _id_campo(variable)
    ayuda = _descripcion(e)
    if e["tipo"] in NUMERICOS:
        lo, hi = _rango(variable)
        return html.Div([html.Label(_rotulo(e), htmlFor=cid, className="campo__rotulo"),
                         dcc.Input(id=cid, type="text", inputMode="numeric" if e["tipo"] == "entera" else "decimal",
                                   value=_valor_campo(base, variable), debounce=True, autoComplete="off", className="campo"),
                         html.P(f"De {lo} a {hi}", className="campo__ayuda")], className="campo-bloque", title=ayuda)
    if e["tipo"] == "booleana":
        return html.Div(ui.control(_rotulo(e), ui.segmentado(cid, [{"label": "Sí", "value": True}, {"label": "No", "value": False}],
                                                             bool(base["valores"][variable]), pequeno=True, persistir=False)),
                        title=ayuda)
    return html.Div(ui.selector(cid, [{"label": c, "value": c} for c in e["categorias"]], base["valores"][variable], rotulo=_rotulo(e),
                                buscar=len(e["categorias"]) > LISTA_CORTA, persistir=False), title=ayuda)


TEMAS = [
    ("Perfil", ("age", "gender", "location", "income_bracket", "education_level", "marital_status", "household_size",
                "acquisition_channel", "customer_segment", "customer_tenure")),
    ("Productos", ("savings_account", "credit_card", "personal_loan", "investment_account", "insurance_product", "credit_utilization_ratio")),
    ("Uso de la aplicación", ("app_logins_frequency", "feature_usage_diversity", "bill_payment_user", "auto_savings_enabled",
                              "app_store_rating")),
    ("Satisfacción y comentarios", ("base_satisfaction", "satisfaction_score", "tx_satisfaction", "product_satisfaction",
                                    "feedback_sentiment")),
    ("Soporte", ("support_tickets_count", "resolved_tickets_ratio")),
    ("Transacciones (resumen)", ("tx_count", "avg_tx_value", "total_tx_volume", "transaction_frequency", "preferred_transaction_type",
                                 "weekend_transaction_ratio", "international_transactions", "failed_transactions")),
]
DETALLE = "Detalle de transacciones (derivadas en el ETL)"


@lru_cache(maxsize=1)
def _bloques():
    """(variables a la vista, [(tema, variables)]): las que concentran la señal y las demás por tema, sin repetir ninguna."""
    clave = tuple(v for v in M.importancia("clasificacion").index[:N_SENAL] if v in COLUMNAS)
    usadas, bloques = set(clave), []
    for tema, vs in TEMAS:
        propias = [v for v in vs if v in COLUMNAS and v not in usadas]
        usadas |= set(propias)
        if propias:
            bloques.append((tema, tuple(propias)))
    detalle = [v for v in COLUMNAS if v not in usadas and _esq()[v].get("derivada_etl")]
    usadas |= set(detalle)
    if detalle:
        bloques.append((DETALLE, tuple(detalle)))
    resto = [v for v in COLUMNAS if v not in usadas]
    if resto:
        bloques.append(("Otras variables", tuple(resto)))
    return clave, bloques


def _panel(base):
    """El formulario: cómo cargar un cliente, el estado de las variables atadas, las que concentran la señal y los temas."""
    fila = base["origen"].get("fila")
    casos = [{"label": f"{CASOS.get(k, k.capitalize())} · fila {f}", "value": f} for f, k in _casos().items()]
    clave, bloques = _bloques()
    return html.Div([
        html.Div([html.H2("El cliente", className="panel__titulo")], className="panel__cab"),
        html.P(_texto_origen(base, []), id=_i("origen"), className="nota"),
        ui.selector(_i("caso"), casos, fila if fila in _casos() else None, rotulo="Caso del informe", persistir=False,
                    placeholder="Elija un caso"),
        ui.selector(_i("ir"), list(_filas()), fila, rotulo="Cliente de prueba por su fila", buscar=True, persistir=False,
                    placeholder="Escriba la fila"),
        html.Div([ui.boton("Otro al azar", id=_i("azar"), icono="canje", pequeno=True),
                  ui.boton("Mediana del entrenamiento", id=_i("mediana"), pequeno=True),
                  ui.boton("Abrir en «Un cliente»", href=_enlace_cliente(base), tipo="fantasma", pequeno=True, id=_i("a-cliente"),
                           titulo="La explicación exacta de la EBM y su simulador, para este cliente de prueba")], className="botonera"),
        html.Div([html.H3(f"Las {CARDINAL.get(len(clave), F.entero(len(clave)))} variables que concentran la señal", className="mas__titulo")]
                 + [_campo(v, base) for v in clave], className="panel__grupo"),
        html.Div([html.Div(id=_i("atadas")),
                  html.Div(ui.boton("Recalcular las atadas", id=_i("atar"), pequeno=True, icono="canje", titulo=_texto_atadas()),
                           className="botonera")], className="campo-bloque"),
        *[ui.desplegable(f"{tema} ({F.entero(len(vs))})", html.Div([_campo(v, base) for v in vs], className="panel__grupo"), llano=True)
          for tema, vs in bloques],
    ], className="panel-parametros")


def _tarjeta_prediccion():
    """La tarjeta del modelo elegido: título y subtítulo, el «…» con de dónde sale cada cifra y el cuerpo (callback)."""
    return html.Div([
        html.Div([html.Div([html.H3(id=_i("prediccion-titulo"), className="tarjeta__titulo"),
                            html.P(id=_i("prediccion-sub"), className="tarjeta__sub")], className="tarjeta__textos"),
                  ui.mas(html.Div(id=_i("prediccion-detalles")), FUENTES_PREDICCION)], className="tarjeta__cab"),
        html.Div(id=_i("prediccion-cuerpo"), className="panel__grupo"),
    ], className="tarjeta", id=_i("prediccion"))


def _cargando(hijos):
    """Mientras se calcula la explicación, un chip en lugar de la gráfica (solo si tarda: k-NN y los estimados)."""
    return dcc.Loading(hijos, delay_show=DEMORA_CARGA_MS, parent_className="resultados-carga",
                       custom_spinner=ui.chip("Calculando la explicación local…", icono="reloj"))


def _sin_prefijo(texto, prefijo):
    return texto[len(prefijo):] if texto.lower().startswith(prefijo) else texto


# Los modelos de cada familia de explicacion.py (la EBM se explica con motor.py)
FAMILIAS = {"bdl": (O,), "ebm": ("ebm",), "lineal": ("logistica", "svm", "lasso", "ridge", "svr"), "bayes": ("naive_bayes",),
            "xgboost": ("xgboost",), "arbol": ("arbol", "random_forest"), "vecinos": ("knn",)}


def _texto_familias():
    """Qué explicaciones son exactas y cuáles se estiman en vivo: se estiman las familias con muestreo en explicacion.MUESTREO."""
    def nombres(familias):
        return F.enumerar([D.en_frase(m) for f in familias for m in FAMILIAS[f]])
    estimadas = [f for f in FAMILIAS if f in E.MUESTREO]
    exactas = [f for f in FAMILIAS if f not in E.MUESTREO]
    texto = f"Es exacta en {nombres(exactas)}."
    if estimadas:
        texto += f" En {nombres(estimadas)} se estima por permutaciones en el momento y lleva su error de Monte Carlo."
    return texto


def _que_mide():
    cal = V.calibracion()["modelos"]
    ece_sin = [c["ece_prueba_sin_recalibrar"] for c in cal.values() if c.get("ece_prueba_sin_recalibrar") is not None]
    ece_rec = [c["ece_prueba_recalibrada"] for c in cal.values() if c.get("ece_prueba_recalibrada") is not None]
    con_ece = [m for m, c in cal.items() if c.get("ece_prueba_sin_recalibrar") is not None]
    ebm = cal["ebm"]["conformal"][NIVEL_OMISION]
    p = D.puntajes()
    props = [v for v in COLUMNAS if v.startswith("prop_")]
    return ui.desplegable("Qué mide cada cifra", [
        ui.parrafo("Puntaje del modelo: lo que devuelve la tubería guardada en artifacts/, la probabilidad sin recalibrar de riesgo alto "
                   "(el SVM lineal da su margen, una distancia al hiperplano). Sirve para ordenar clientes con un mismo modelo; no se "
                   f"compara entre modelos ni se lee como probabilidad: con ponderación de clases los puntajes están inflados (el puntaje "
                   f"medio de la EBM en la prueba es {ui.num(p['ebm'].mean(), DEC_ECE)} frente a una prevalencia de "
                   f"{ui.num(p['y'].mean(), DEC_ECE)})."),
        ui.parrafo(f"Probabilidad recalibrada: {V.calibracion()['metodo'].split(';')[0]}. En la prueba, el error de calibración (ECE) "
                   f"va de {ui.num(min(ece_sin), DEC_ECE)} a {ui.num(max(ece_sin), DEC_ECE)} sin recalibrar"
                   + (f" (sin {F.enumerar([D.en_frase(m) for m in cal if m not in con_ece])}, cuyo margen no es una probabilidad)"
                      if len(con_ece) < len(cal) else "")
                   + f" y de {ui.num(min(ece_rec), DEC_ECE)} a {ui.num(max(ece_rec), DEC_ECE)} recalibrada. {D.nota_ece()}"),
        ui.parrafo("Conjunto conforme (clasificación): las clases que el modelo no puede descartar con la cobertura elegida. Una clase entra "
                   "si su probabilidad recalibrada es al menos 1 menos un umbral calibrado con las predicciones fuera de pliegue del "
                   "entrenamiento. La cobertura nominal es la fracción de clientes cuyo conjunto debería contener su clase real; la medida "
                   "es la que se observó en la prueba. Como la probabilidad recalibrada va por escalones, la medida puede quedar por encima "
                   f"de la nominal: en la EBM, al {F.pct(float(NIVEL_OMISION), 0)} se midió el {F.pct(ebm['cobertura_prueba'])}."),
        ui.parrafo(f"Intervalo conforme (regresión), {_sin_prefijo(V.conformal()['metodo'], 'intervalo conforme ')}. La cobertura medida es la "
                   "fracción de clientes de prueba cuyo puntaje real cayó dentro de su intervalo."),
        *([ui.parrafo("Modelo original (BDL): en regresión, además del intervalo conforme común, da su conjunto de predicción, una unión "
                      "de tramos alrededor de los niveles en que sitúa al cliente, calibrado con filas que no entraron en sus árboles; en "
                      "clasificación, su puntaje es la probabilidad de superar el umbral según la mezcla de niveles. Las dos salidas se "
                      "muestran en el gráfico que aparece debajo de la predicción cuando el modelo elegido es BDL.")]
          if any(V.hay_original(t) for t in V.TAREAS) else []),
        ui.parrafo("Explicación local (valores SHAP): el punto de partida más el aporte de cada variable da exactamente la salida del "
                   "modelo, en la escala en que el modelo suma (logit, margen, probabilidad sin recalibrar o pronóstico). "
                   f"{_texto_familias()} Dice qué mueve la salida para este cliente, no qué pasaría si el cliente cambiara."),
        ui.parrafo(f"Variables atadas: {_texto_atadas()} También van juntas en los datos el detalle y el resumen de las transacciones y "
                   f"las {CARDINAL.get(len(props), F.entero(len(props)))} proporciones por tipo de transacción, que suman 1: se pueden "
                   "escribir libres, pero un cliente así no existe en los datos."),
    ])


def _seccion_lectura():
    n = F.entero(M.meta()["n_prueba"])
    destinos = [d for d in app.DESTINOS if d[0] in (app.ruta_vista("cliente"), app.ruta_vista("interpretabilidad"))]
    return ui.seccion(f"La incertidumbre se midió en los {n} clientes de prueba, que ningún modelo vio al ajustarse", [
        ui.aviso("«Riesgo alto» es el cuarto superior del puntaje de propensión del conjunto de datos, que es sintético (COFINFAD se "
                 "generó con reglas): no es una fuga observada. El modelo asigna más o menos riesgo según lo que aprendió de esos datos; no "
                 "dice qué pasaría si el cliente cambiara una variable.", titulo="Una predicción no es una fuga observada"),
        _que_mide(),
        ui.destinos([ui.destino(ruta, pantalla, nombre, pregunta) for ruta, pantalla, nombre, pregunta in destinos]),
    ], subtitulo="De dónde sale cada cifra de la predicción, qué significa y qué no.", fuentes=FUENTES_LECTURA)


def layout(tarea=None, modelo=None, fila=None, nivel=None, **_):
    tarea, modelo, nivel, fila, avisos = _resolver(tarea, modelo, fila, nivel)
    base = _base_prueba(fila)
    cascada = ui.grafico(_i("cascada"), alto="alto", fuentes=FUENTES_CASCADA, resumen_tabla="Ver los datos (el aporte de cada variable)")
    modelos = ui.grafico(_i("modelos"), fuentes=FUENTES_MODELOS, resumen_tabla="Ver los datos (todos los modelos en vivo)")
    resultados = html.Div([
        html.Div(id=_i("validacion")),
        _tarjeta_prediccion(),
        html.Div(id=_i("original")),
        _cargando(cascada),
        html.Div(id=_i("vecinos")),
        modelos,
    ], className="resultados")
    laboratorio = html.Div([
        html.Div(_aviso_direccion(avisos), id=_i("aviso")),
        html.Div([_panel(base), resultados], className="laboratorio"),
        dcc.Store(id=_i("base"), data=base),
        dcc.Store(id=_i("direccion")),
    ])
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), _subtitulo(), _controles(tarea, modelo, nivel), respuesta=laboratorio, pantalla=PANTALLA),
        _seccion_lectura(),
    ])


# --- Callbacks ------------------------------------------------------------------------------------------------------------
CAMPOS_ENTRADA = [Input(_id_campo(v), "value") for v in COLUMNAS]
CAMPOS_ESTADO = [State(_id_campo(v), "value") for v in COLUMNAS]
CAMPOS_SALIDA = [Output(_id_campo(v), "value") for v in COLUMNAS]


def _salidas_grafico(gid):
    return [Output(gid, "figure"), Output(f"{gid}-titulo", "children"), Output(f"{gid}-sub", "children"),
            Output(f"{gid}-leyenda", "children"), Output(f"{gid}-tabla", "children"), Output(f"{gid}-nota", "children"),
            Output(f"{gid}-lienzo", "aria-label")]


def _grafico_vacio(texto):
    return [ui.figura_vacia(texto), texto, None, None, None, None, texto]


@callback(Output(_i("modelo"), "options"), Output(_i("modelo"), "value"), Input(_i("tarea"), "value"), State(_i("modelo"), "value"),
          prevent_initial_call=True)
def _opciones_modelo(tarea, modelo):
    """Los modelos en vivo de la tarea; si el elegido no es de ella, su análogo."""
    tarea = tarea if tarea in V.TAREAS else TAREA_OMISION
    return _opciones(tarea), _modelo_de(tarea, modelo)


@callback(*CAMPOS_SALIDA, Output(_i("base"), "data"), Output(_i("caso"), "value"), Output(_i("ir"), "value"), Output(_i("aviso"), "children"),
          Input(_i("caso"), "value"), Input(_i("ir"), "value"), Input(_i("azar"), "n_clicks"), Input(_i("mediana"), "n_clicks"),
          Input(_i("atar"), "n_clicks"), *CAMPOS_ESTADO, State(_i("base"), "data"), prevent_initial_call=True)
def _cargar(caso, ir, _clic_azar, _clic_mediana, _clic_atar, *resto):
    """Llena el formulario con un caso del informe, una fila, un cliente al azar o la mediana del entrenamiento, o recalcula
    las variables atadas del cliente que está escrito."""
    *campos, base = resto
    origen = ctx.triggered_id
    sin_cambios = [no_update] * (len(COLUMNAS) + 3)
    try:
        if origen == _i("caso"):
            fila = _fila_valida(caso)
            if fila is None:
                raise PreventUpdate
            nueva = _base_prueba(fila)
        elif origen == _i("ir"):
            if ir is None or str(ir).strip() == "":
                raise PreventUpdate
            fila = _fila_valida(ir)
            if fila is None:
                return sin_cambios + [_aviso_direccion([f"«{str(ir).strip()[:LARGO_TEXTO]}» no es la fila de un cliente de prueba."])]
            nueva = _base_prueba(fila)
        elif origen == _i("azar"):
            actual = ((base or {}).get("origen") or {}).get("fila")
            candidatos = [f for f in _filas() if f != actual]
            nueva = _base_prueba(int(np.random.default_rng().choice(candidatos)))
        elif origen == _i("mediana"):
            nueva = _base_mediana()
        elif origen == _i("atar"):
            nueva, aviso = _atar(campos, base)
            if nueva is None:
                return sin_cambios + [html.Div(aviso, className="aviso-global")]
            atadas = set(_calculadas())
            return ([_valor_campo(nueva, v) if v in atadas else x for v, x in zip(COLUMNAS, campos)]
                    + [nueva, no_update, no_update, None])
        else:
            raise PreventUpdate
    except PreventUpdate:
        raise
    except Exception as e:  # noqa: BLE001  (una entrada nunca detiene la vista)
        REGISTRO.exception("no se pudo cargar el cliente")
        return sin_cambios + [html.Div(_aviso_error("No se pudo cargar el cliente", e), className="aviso-global")]
    fila = nueva["origen"].get("fila")
    return [_valor_campo(nueva, v) for v in COLUMNAS] + [nueva, fila if fila in _casos() else None, fila, None]


@callback(Output(_i("validacion"), "children"), Output(_i("prediccion-titulo"), "children"), Output(_i("prediccion-sub"), "children"),
          Output(_i("prediccion-cuerpo"), "children"), Output(_i("prediccion-detalles"), "children"), *_salidas_grafico(_i("modelos")),
          Output(_i("atadas"), "children"), Output(_i("origen"), "children"), Output(_i("a-cliente"), "href"),
          Output(_i("original"), "children"),
          Input(_i("tarea"), "value"), Input(_i("modelo"), "value"), Input(_i("nivel"), "value"), *CAMPOS_ENTRADA, Input(_i("base"), "data"))
def _predecir(tarea, modelo, nivel, *resto):
    """Valida el formulario y, si es válido, da la predicción del modelo elegido, lo propio del modelo original (si es el
    elegido) y la de todos los modelos en vivo de la tarea."""
    *campos, base = resto
    tarea = tarea if tarea in V.TAREAS else TAREA_OMISION
    modelo = _modelo_de(tarea, modelo)
    nivel = _nivel(nivel) or NIVEL_OMISION
    try:
        fila, cambiadas = _entrada(campos, base)
        origen, enlace = _texto_origen(base, cambiadas), _enlace_cliente(base)
        atadas = _estado_atadas(fila)
        problemas = _problemas(fila)
        if problemas:
            vacio = "Sin predicción: hay valores que corregir"
            return ([_aviso_problemas(problemas), vacio, origen, ui.nota("Corrija los valores del aviso para ver la predicción."), None]
                    + _grafico_vacio(vacio) + [atadas, origen, enlace, None])
        clave = _clave(fila)
        puntajes, errores = _puntajes(tarea, clave)
        if modelo not in puntajes:
            raise RuntimeError(f"{D.NOMBRE_MODELO[modelo]} no se pudo evaluar: {errores.get(modelo, '')}")
        hacer = _prediccion_clasificacion if tarea == "clasificacion" else _prediccion_regresion
        real = _real(base, cambiadas)
        titulo, cuerpo, detalles = hacer(modelo, puntajes[modelo], nivel, real, clave)
        propio = _bloque_original(tarea, clave, nivel, real, puntajes[modelo]) if modelo == O else None
        return ([None, titulo, origen, cuerpo, detalles] + list(_modelos(tarea, modelo, puntajes, errores, nivel))
                + [atadas, origen, enlace, propio])
    except Exception as e:  # noqa: BLE001  (una entrada nunca detiene la vista)
        REGISTRO.exception("no se pudo predecir")
        vacio = "Sin predicción: no se pudo calcular"
        return ([_aviso_error("No se pudo calcular la predicción", e), vacio, None, None, None] + _grafico_vacio(vacio)
                + [no_update, no_update, no_update, None])


@callback(*_salidas_grafico(_i("cascada")), Output(_i("vecinos"), "children"),
          Input(_i("tarea"), "value"), Input(_i("modelo"), "value"), *CAMPOS_ENTRADA, Input(_i("base"), "data"))
def _explicar(tarea, modelo, *resto):
    """La explicación local SHAP del modelo elegido para el cliente del formulario (y sus vecinos, si es k-NN)."""
    *campos, base = resto
    tarea = tarea if tarea in V.TAREAS else TAREA_OMISION
    modelo = _modelo_de(tarea, modelo)
    try:
        fila, _cambiadas = _entrada(campos, base)
        if _problemas(fila):
            return _grafico_vacio("Sin explicación: hay valores que corregir") + [None]
        clave = _clave(fila)
        s = _puntaje(tarea, modelo, clave)
        salidas = list(_cascada(tarea, modelo, fila, _explicacion(tarea, modelo, clave), s))
        return salidas + [_tarjeta_vecinos(tarea, clave, s) if modelo == "knn" else None]
    except Exception as e:  # noqa: BLE001
        REGISTRO.exception("no se pudo explicar")
        vacio = "Sin explicación: no se pudo calcular"
        return [ui.figura_vacia(vacio), vacio, _aviso_error("No se pudo calcular la explicación", e), None, None, None, vacio, None]


# En el navegador: la dirección sigue a la tarea, al modelo, al nivel y al cliente de prueba (?tarea=…&modelo=…&nivel=…&fila=…)
# con history.replaceState, que no dispara el enrutado de Dash: al recargar o al volver se abre lo mismo.
clientside_callback(
    r"""function (tarea, modelo, nivel, base) {
        var sin_cambio = window.dash_clientside.no_update;
        var ruta = window.location.pathname;
        if (ruta.replace(/\/+$/, '') !== __RUTA__) { return sin_cambio; }
        var q = new URLSearchParams();
        if (tarea) { q.set('tarea', tarea); }
        if (modelo) { q.set('modelo', modelo); }
        if (nivel) { q.set('nivel', nivel); }
        var origen = base && base.origen;
        if (origen && origen.fila !== undefined && origen.fila !== null) { q.set('fila', origen.fila); }
        var destino = ruta + '?' + q.toString();
        if (ruta + window.location.search !== destino) { window.history.replaceState(window.history.state, '', destino); }
        return sin_cambio;
    }""".replace("__RUTA__", json.dumps(app.ruta_vista(VISTA))),
    Output(_i("direccion"), "data"), Input(_i("tarea"), "value"), Input(_i("modelo"), "value"), Input(_i("nivel"), "value"),
    Input(_i("base"), "data"))
