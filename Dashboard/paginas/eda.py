"""EDA · ¿Qué muestran los datos de entrenamiento?

La sección EDA del dash (/eda, prefijo de id «ed-»). Todo se calcula en el servidor con los clientes de entrenamiento
(modelos_vivos.entrenamiento(), que lee datos/eda/entrenamiento.parquet una vez y lo guarda en caché): los clientes de
prueba no entran en ningún gráfico, tabla ni frase. El profesor penalizó en la Entrega 1 un EDA hecho sin partición, y la
página lo dice. De la prueba solo se leen sus índices (para comprobar que no se cruzan con el entrenamiento) y dos
cifras guardadas (metadatos.json: su tamaño y su proporción de riesgo alto); la tarjeta de la trampa cita dos resultados
de modelos del proyecto (techo.json) y lo rotula.

Arriba van los filtros globales (cohorte de edad, segmento de uso, canal de llegada y el rango de una variable numérica,
con un contador de clientes). Cada gráfico escucha los cinco filtros, se recalcula con los clientes que los cumplen y
lleva en su pie una frase de lectura calculada con esos mismos clientes. Orden: fuente, estructura y estadística
descriptiva → el objetivo y el balance de clases → la trampa de los productos activos → distribuciones univariadas →
explorador bivariado (x, y y color) → correlaciones de Spearman → faltantes y atípicos (Tukey) → transformaciones del ETL
y control de la fuga de información.

Color (DISENO.md): lo observado va en «real» (sólido: riesgo alto o la barra en foco; rayado: lo demás); todo el
entrenamiento, cuando hay filtros, en «contexto»; los grupos del color del explorador, en una rampa de grises hecha con los
tokens (contexto → base → tinta-2 → real), solo para variables con orden; la matriz de correlación, en un degradado de un
solo tono (de --superficie a --serie-real) sobre |ρ|, con el signo escrito en la celda.

Lo que esta página resuelve por su cuenta y podría subir a ui.py: _deslizador_rango (ui.deslizador con dcc.RangeSlider),
_campo (un control con su rótulo encima), _eje_fraccion (eje de porcentajes) y _marcas (marcas con millones en «M»).
"""
import math
import unicodedata
from functools import lru_cache

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

import app
import datos_app as D
import formato as F
import modelos_vivos as V
import motor as M
import ui

dash.register_page(__name__, path="/eda", name="EDA", title=f"EDA · {app.MARCA}")

PRE = "ed"
PANTALLA = "datos"                    # icono de la cabecera (ui.PANTALLA_ICONO)
PREGUNTA = next(p for ruta, _i, _n, p in app.SECCIONES if ruta == "/eda")
OBJ, AP, ETQ = "churn_probability", "active_products", "riesgo_alto"
TODAS = "todas"
TUKEY = 1.5                           # cercas de Tukey: 1,5 rangos intercuartílicos
MIN_CLIENTES = 30                     # con menos clientes filtrados no se dibuja nada
N_MUESTRA = 1500                      # puntos del explorador: una muestra fija con la semilla del proyecto
N_TRAMOS = 30                         # barras del histograma de una variable continua
N_TRAMOS_LINEA = 10                   # tramos de x para la línea de medianas del explorador
MAX_DISCRETOS = 30                    # hasta tantos valores distintos, una barra por valor
MAX_CAJAS = 12                        # hasta tantos valores distintos, una x numérica se dibuja en cajas
MAX_HORIZONTAL = 6                    # una categórica con más niveles va en barras horizontales
MAX_ROTULOS = 10                      # rótulos que caben bajo las barras a 400 px
N_MATRIZ = 10                         # variables de la matriz corta (el puntaje, los productos y ocho predictores)
N_ATIPICOS = 12                       # variables del gráfico de atípicos (todas en la tabla)
N_PARES = 20                          # pares de la tabla gemela de la matriz
P_COLA = 0.99                         # una variable con cola larga se dibuja hasta su percentil 99
COLA = 0.5                            # cola larga: el máximo pasa al percentil 99 en más de media amplitud p1–p99
DUPLICADO = 0.99                      # |ρ| desde el que dos predictores repiten la misma información
ASIMETRIA = 0.5                       # |asimetría| desde la que una distribución se dice asimétrica
OSCURO = 0.5                          # |ρ| desde el que la celda es oscura y su texto va en blanco
MILLON = 1_000_000
TRES = 3                              # en la escalera, «tres o más productos» (sin riesgo alto)
FIJO = F.FIJO
PUNTO = ui.MARCADOR / 2               # puntos de la muestra: la mitad del marcador de la app y sin anillo
OPACIDAD = 0.45
AIRE = 0.08                           # aire sobre la barra más alta, como fracción del tope
RHO_ABS = "|" + chr(0x2060) + "ρ" + chr(0x2060) + "|"   # |ρ| que no se parte al final de una línea
FUERZA = ((0.1, "prácticamente nula"), (0.3, "débil"), (0.5, "moderada"), (math.inf, "fuerte"))
TERCIOS = ("tercio inferior", "tercio medio", "tercio superior")
# Rampa de grises (de claro a oscuro) para los grupos del color del explorador, según cuántos son
RAMPA = {1: ["real"], 2: ["contexto", "real"], 3: ["contexto", "base", "real"], 4: ["contexto", "base", "tinta_2", "real"]}
INICIAL = {"uni": "app_logins_frequency", "x": "base_satisfaction", "y": OBJ, "color": "age", "caja": "tx_count", "matriz": "corta"}
RANGO_VARIABLES = ("app_logins_frequency", "customer_tenure")
CORTO = {"app_logins_frequency": "Inicios de sesión", "customer_tenure": "Antigüedad"}
NOMBRE_EXTRA = {OBJ: "Puntaje de propensión", AP: "Productos activos"}
BOOLEANO = {True: "Sí", False: "No"}
PALABRA = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
           11: "once", 12: "doce"}
# Categóricas con un orden natural (para la rampa de grises y para el eje); las demás se ordenan por frecuencia
ORDINALES = {"customer_segment": ["inactive", "occasional", "regular", "power"],
             "income_bracket": ["Low", "Medium", "High", "Very High"],
             "education_level": ["High School", "Bachelor", "Master", "PhD"],
             "feedback_sentiment": ["Negative", "Neutral", "Positive"]}
FECHAS = ("first_transaction_date", "last_transaction_date", "last_survey_date", "first_tx", "last_tx")
# Razón de cada columna eliminada, según la tabla del ETL de la metodología del informe (tab:etl, sin sus cifras)
RAZON = {
    "customer_id": "Identificador del cliente: sin valor predictivo.",
    AP: "Fuga de información: el puntaje se construyó a partir de ella.",
    "customer_lifetime_value": "Es el segundo objetivo del proyecto.",
    "clv_segment": "Es el segmento del segundo objetivo del proyecto.",
    "average_transaction_value": "Resumen de transacciones que duplica a otra columna.",
    "total_transaction_volume": "Resumen de transacciones que duplica a otra columna.",
    "avg_daily_transactions": "Resumen de transacciones que duplica a otra columna.",
    "monthly_transaction_count": "Resumen de transacciones que duplica a otra columna.",
    "first_transaction_date": "Fecha que duplica a otra fecha.",
    "last_transaction_date": "Fecha que duplica a otra fecha.",
    "nps_score": "Colinealidad con la satisfacción.",
    "occupation": "Muchos niveles y sin señal.",
    "complaint_topics": "Texto con muchos valores ausentes, sin mecanismo ni señal.",
    "feature_requests": "Texto con muchos valores ausentes, sin mecanismo ni señal.",
    "last_survey_date": "Fecha sin señal.",
    "first_tx": "Fecha sin señal.",
    "last_tx": "Fecha sin señal.",
}
AUDITORIA = {"montos_no_positivos": "montos no positivos", "fechas_nulas": "fechas nulas", "huerfanas": "transacciones sin cliente",
             "clientes_sin_tx": "clientes sin transacciones"}

F_ENT, F_META, F_ETL = D.fuente("eda/entrenamiento.parquet"), D.fuente("metadatos.json"), D.fuente("eda/etl.json")
F_FALT, F_ASOC, F_VARS = D.fuente("eda/faltantes.csv"), D.fuente("eda/asociacion.csv"), D.fuente("variables.csv")
F_HIST, F_TECHO = D.fuente("eda/objetivo_hist.csv"), D.fuente("entrega3/comparacion_entrega2/techo.json")
F_CLIENTES = D.fuente("clientes_prueba.parquet")
# Las partes de ui.grafico que escribe un callback, en este orden
PARTES = (("", "figure"), ("-titulo", "children"), ("-sub", "children"), ("-leyenda", "children"), ("-tabla", "children"),
          ("-nota", "children"), ("-lienzo", "aria-label"))
FILTROS = [Input(f"{PRE}-{k}", "value") for k in ("cohorte", "segmento", "canal", "rango-var", "rango")]


# --- Lecturas ---------------------------------------------------------------------------------------------------------
def _T():
    """El entrenamiento (compartido y en caché: no se modifica)."""
    return V.entrenamiento()


def _etl():
    return D.json("eda/etl.json")


def _tau():
    return float(M.meta()["etiqueta"]["umbral"])


def _percentil():
    """«percentil 75», con el número leído de la etiqueta del trabajo."""
    return f"percentil{FIJO}{F.num(100 * M.meta()['etiqueta']['q'])}"


def _pc(q):
    """«percentil 25», «percentil 99»: el número sale de la fracción q."""
    return f"percentil{FIJO}{F.num(100 * q)}"


P25, P75 = 0.25, 0.75


def _numericas():
    return list(_etl()["tipos"]["numericas"])


@lru_cache(maxsize=None)
def _tipo(v):
    """«numerica», «categorica» o «booleana» (el puntaje y los productos activos son numéricos)."""
    t = _etl()["tipos"]
    return "categorica" if v in t["categoricas"] else "booleana" if v in t["booleanas"] else "numerica"


@lru_cache(maxsize=1)
def _predictores():
    """Los predictores en el orden de su importancia en la EBM (la edad primero)."""
    tabla = M.variables()
    return tuple(sorted(M.predictores(), key=lambda v: (int(tabla.at[v, "puesto_clf"]) if v in tabla.index else len(tabla), v)))


@lru_cache(maxsize=None)
def _distintos(v):
    return int(_T()[v].nunique())


@lru_cache(maxsize=None)
def _niveles(v):
    """Los valores de una variable discreta en su orden: no/sí, el orden natural de una ordinal, la frecuencia en todo el
    entrenamiento de una nominal, o de menor a mayor si es numérica."""
    x = _T()[v]
    if _tipo(v) == "booleana":
        return (False, True)
    if _tipo(v) == "numerica":
        return tuple(sorted(x.unique().tolist()))
    frecuencia = x.value_counts()
    orden = ORDINALES.get(v)
    if orden and set(orden) == set(frecuencia.index):
        return tuple(orden)
    return tuple(frecuencia.index)


@lru_cache(maxsize=1)
def _orden():
    """Orden aleatorio fijo de las filas del entrenamiento (semilla del proyecto): la muestra del explorador son los
    primeros clientes de este orden que cumplen los filtros."""
    return np.random.default_rng(M.SEMILLA).permutation(len(_T()))


@lru_cache(maxsize=1)
def _faltan():
    """Las columnas originales con faltantes que trae el entrenamiento como indicadoras falta__<columna>."""
    return tuple(c[len("falta__"):] for c in _T().columns if c.startswith("falta__"))


# --- Nombres y números ------------------------------------------------------------------------------------------------
def _sin_tildes(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch)).lower()


def _nombre(v):
    return NOMBRE_EXTRA.get(v) or M.nombre(v)


def _minuscula(texto):
    return texto[:1].lower() + texto[1:]


def _en_frase(v):
    return _minuscula(_nombre(v))


def _unidad(v):
    tabla = M.variables()
    u = tabla.at[v, "unidad"] if v in tabla.index else None
    return f" {u}" if isinstance(u, str) and u else ""


@lru_cache(maxsize=None)
def _dec(v):
    """Decimales con que se escribe un valor de la variable (según sus valores habituales, percentiles 5 y 95)."""
    if v == OBJ:
        return 3
    if v == AP or v not in M.variables().index:
        return 0
    f = M.variables().loc[v]
    if f["tipo"] in ("entera", "binaria"):
        return 0
    tope = max(abs(float(f["p5"])), abs(float(f["p95"])))
    return 0 if tope >= 100 else 1 if tope >= 20 else 2 if tope >= 2 else 3


def _dec_media(v):
    """Decimales de una media o una desviación: los de la variable, y uno más si es entera y pequeña."""
    d = _dec(v)
    if v == AP:
        return 2
    return 1 if d == 0 and v in M.variables().index and abs(float(M.variables().at[v, "p95"])) < MILLON else d


@lru_cache(maxsize=None)
def _en_millones(v):
    """La variable se escribe en millones («1,8 M»): sus valores habituales (percentil 95) pasan del millón."""
    tabla = M.variables()
    return v in tabla.index and abs(float(tabla.at[v, "p95"])) >= MILLON


def _val(v, x, d=None):
    """Un valor de la variable en es-CO; las variables de montos, en millones con «M» (todos sus valores, para que una fila
    de la tabla no mezcle «318.215» con «1,3 M»)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return F.RAYA
    if not math.isfinite(x):
        return F.RAYA
    if _en_millones(v) or abs(x) >= MILLON:
        return "0" if x == 0 else f"{ui.num(x / MILLON, 1)}{FIJO}M"
    return ui.num(x, _dec(v) if d is None else d)


def _num_justo(x, d_max):
    """El número con los decimales que necesita, hasta d_max: 4 → «4», 3,5 → «3,5»."""
    for d in range(d_max + 1):
        if round(float(x), d) == float(x):
            return ui.num(x, d)
    return ui.num(x, d_max)


def _etq(v, valor):
    """El rótulo de un valor de una variable discreta: «Sí», la categoría o el número justo."""
    if _tipo(v) == "booleana":
        return BOOLEANO[bool(valor)]
    if _tipo(v) == "categorica":
        return str(valor)
    return _num_justo(float(valor), 3)


def _fuerza(r):
    a = abs(r)
    return next(palabra for tope, palabra in FUERZA if a < tope)


def _forma(asim):
    if asim <= -ASIMETRIA:
        return "es asimétrica a la izquierda (cola hacia los valores bajos)"
    if asim >= ASIMETRIA:
        return "es asimétrica a la derecha (cola hacia los valores altos)"
    return "es casi simétrica"


def _paso_fino(amplitud, marcas=5):
    """Paso redondo (1, 2, 2,5 o 5 por una potencia de diez) para repartir unas pocas marcas en un eje de cualquier tamaño."""
    bruto = (abs(amplitud) or 1) / marcas
    magnitud = 10 ** math.floor(math.log10(bruto))
    for k in (1, 2, 2.5, 5, 10):
        if k * magnitud >= bruto - 1e-12:
            return k * magnitud
    return 10 * magnitud


def _marcas(lo, hi, marcas=5):
    """(tickvals, ticktext) en es-CO de un eje de lo a hi; los millones, con «M»."""
    escala, sufijo = (MILLON, f"{FIJO}M") if max(abs(lo), abs(hi)) >= MILLON else (1, "")
    a, b = lo / escala, hi / escala
    paso = _paso_fino(b - a, marcas)
    vals, textos = ui.marcas(math.ceil(a / paso - 1e-9) * paso, b, paso)
    return [x * escala for x in vals], [t + sufijo for t in textos]


def _eje_fraccion(fig, tope, eje="y", marcas=4):
    """Eje de fracciones de 0 al tope, con las marcas en porcentaje («10 %»)."""
    tope = max(float(tope), 1e-6)
    paso = ui.paso_redondo(tope, marcas)
    vals = ui.marcas(0, tope, paso)[0]
    d = 0 if abs(paso * 100 - round(paso * 100)) < 1e-9 else 1
    props = dict(tickvals=vals, ticktext=["0" if x == 0 else F.pct(x, d) for x in vals], range=[0, tope * (1 + AIRE)])
    (fig.update_yaxes if eje == "y" else fig.update_xaxes)(**props)
    return fig


def _campo(rotulo, componente):
    """Un control con su rótulo encima, como ui.selector(rotulo=…): para alinear un segmentado con un desplegable."""
    return html.Div([html.Label(rotulo, className="campo__rotulo"), html.Div(componente)], className="campo-bloque")


def _cuartiles(y):
    """(q1, mediana, q3, bigote inferior, bigote superior, cerca inferior, cerca superior) de Tukey."""
    q1, med, q3 = np.quantile(y, [0.25, 0.5, 0.75])
    ric = q3 - q1
    lo, hi = q1 - TUKEY * ric, q3 + TUKEY * ric
    dentro = y[(y >= lo) & (y <= hi)]
    return float(q1), float(med), float(q3), float(dentro.min()), float(dentro.max()), float(lo), float(hi)


def _rho(a, b):
    """ρ de Spearman de dos columnas, con rangos promedio (Series.corr con «spearman» importa scipy: un segundo en frío)."""
    ra = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    if len(ra) < 2 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _eta2(y, grupos):
    """η²: la parte de la varianza de y que explican los grupos."""
    y = np.asarray(y, dtype=float)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return float("nan")
    s = pd.Series(y).groupby(np.asarray(grupos)).agg(["mean", "count"])
    return float((s["count"] * (s["mean"] - y.mean()) ** 2).sum() / total)


# --- Filtros ----------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cohortes():
    """((rótulo, edad menor, edad mayor), …): las edades consecutivas del entrenamiento forman una cohorte."""
    grupos = []
    for a in sorted(int(x) for x in _T()["age"].unique()):
        if grupos and a - grupos[-1][-1] == 1:
            grupos[-1].append(a)
        else:
            grupos.append([a])
    return tuple((f"{g[0]}–{g[-1]}" if len(g) > 1 else f"{g[0]}", g[0], g[-1]) for g in grupos)


@lru_cache(maxsize=None)
def _limites(v):
    """(mínimo, máximo, paso) del deslizador de rango de una variable."""
    x = _T()[v]
    if pd.api.types.is_integer_dtype(x):
        return float(x.min()), float(x.max()), 1
    paso = 10.0 ** (math.floor(math.log10(float(x.max() - x.min()))) - 1)
    lo = math.floor(float(x.min()) / paso + 1e-9) * paso
    hi = math.ceil(float(x.max()) / paso - 1e-9) * paso
    return round(lo, 6), round(hi, 6), paso


def _dec_paso(paso):
    return 0 if float(paso).is_integer() else max(0, -math.floor(math.log10(paso)))


def _marcas_rango(v):
    """Marcas del deslizador: el mínimo, unos valores redondos y el máximo (los que caben sin tocarse)."""
    lo, hi, paso = _limites(v)
    p = _paso_fino(hi - lo, 4)
    redondos = [round(k * p, 6) for k in range(math.ceil(lo / p), math.floor(hi / p) + 1)]
    vals = [lo] + [x for x in redondos if lo + 0.1 * (hi - lo) < x < hi - 0.1 * (hi - lo)] + [hi]
    d = _dec_paso(paso)
    return {(int(x) if float(x).is_integer() else x): _num_justo(x, d) for x in vals}


def _texto_rango(v, a, b):
    d = _dec_paso(_limites(v)[2])
    return f"de {_num_justo(a, d)} a {_num_justo(b, d)}{_unidad(v)}"


def _filtro(cohorte, segmento, canal, var, rango):
    """Los cinco filtros validados, como una tupla (lo que no se reconoce cuenta como «sin filtro»)."""
    cohorte = cohorte if cohorte in [c for c, _a, _b in _cohortes()] else TODAS
    segmento = segmento if segmento in _niveles("customer_segment") else TODAS
    canal = canal if canal in _niveles("acquisition_channel") else TODAS
    var = var if var in RANGO_VARIABLES else RANGO_VARIABLES[0]
    lo, hi, _paso = _limites(var)
    try:
        a, b = sorted(float(x) for x in rango)
    except (TypeError, ValueError):
        a, b = lo, hi
    a, b = min(max(a, lo), hi), min(max(b, lo), hi)
    return (cohorte, segmento, canal, var, round(a, 6), round(b, 6))


def _rango_activo(f):
    lo, hi, _paso = _limites(f[3])
    return f[4] > lo + 1e-9 or f[5] < hi - 1e-9


def _activo(f):
    return any(x != TODAS for x in f[:3]) or _rango_activo(f)


@lru_cache(maxsize=64)
def _mascara(f):
    """Qué filas del entrenamiento cumplen los filtros (arreglo de solo lectura)."""
    cohorte, segmento, canal, var, a, b = f
    T = _T()
    m = np.ones(len(T), dtype=bool)
    if cohorte != TODAS:
        _c, e0, e1 = next(c for c in _cohortes() if c[0] == cohorte)
        m &= T["age"].between(e0, e1).to_numpy()
    if segmento != TODAS:
        m &= (T["customer_segment"] == segmento).to_numpy()
    if canal != TODAS:
        m &= (T["acquisition_channel"] == canal).to_numpy()
    if _rango_activo(f):
        x = T[var].to_numpy(dtype=float)
        m &= (x >= a - 1e-9) & (x <= b + 1e-9)
    m.setflags(write=False)
    return m


@lru_cache(maxsize=4)
def _sub(f):
    """Los clientes de entrenamiento que cumplen los filtros (compartida: no se modifica)."""
    return _T()[_mascara(f)]


def _texto_filtros(f):
    cohorte, segmento, canal, var, a, b = f
    partes = []
    if cohorte != TODAS:
        partes.append(f"{cohorte} años")
    if segmento != TODAS:
        partes.append(f"segmento {segmento}")
    if canal != TODAS:
        partes.append(f"canal {canal}")
    if _rango_activo(f):
        partes.append(f"{_en_frase(var)} {_texto_rango(var, a, b)}")
    return " · ".join(partes)


def _quienes(f, n):
    """«los 38.978 clientes de entrenamiento» o «los 6.512 clientes de entrenamiento que cumplen los filtros (…)»."""
    if not _activo(f):
        return f"los {F.entero(n)} clientes de entrenamiento"
    return f"los {F.entero(n)} clientes de entrenamiento que cumplen los filtros ({_texto_filtros(f)})"


def _contador(f):
    T = _T()
    m = _mascara(f)
    n, N = int(m.sum()), len(T)
    k = int(T[ETQ].to_numpy()[m].sum())
    variacion = f"Riesgo alto: {F.pct(k / n)} ({F.entero(k)} clientes)" if n else "Ningún cliente cumple los filtros"
    nota = (_texto_filtros(f) + ".") if _activo(f) else "Sin filtros: todo el entrenamiento."
    hijos = [ui.cifra("Clientes que cumplen los filtros", F.entero(n), f"de {F.entero(N)}", variacion=variacion, nota=nota, mini=True)]
    if n < MIN_CLIENTES:
        hijos.append(ui.aviso(f"Con estos filtros quedan {F.entero(n)} clientes: hacen falta al menos {F.entero(MIN_CLIENTES)} para "
                              "describirlos. Amplíe los filtros o quítelos.", "atencion", "Muy pocos clientes", rol="status"))
    return hijos


# --- Piezas comunes de los gráficos -------------------------------------------------------------------------------------
def _salidas(gid):
    return [Output(f"{gid}{sufijo}", propiedad) for sufijo, propiedad in PARTES]


def _partes(g):
    return [g["figura"], g["titulo"], g["sub"], g.get("leyenda"), g["tabla"], g["nota"], g["etiqueta"]]


def _vacio(f, n, que):
    texto = (f"Con estos filtros quedan {F.entero(n)} clientes: hacen falta al menos {F.entero(MIN_CLIENTES)} para {que}. "
             "Amplíe los filtros.")
    return {"figura": ui.figura_vacia("Muy pocos clientes con estos filtros"), "titulo": "Sin clientes suficientes con estos filtros",
            "sub": f"Quedan {_quienes(f, n)}.", "leyenda": None, "tabla": ui.nota(texto), "nota": texto, "etiqueta": texto}


def _fallo(e):
    texto = f"No se pudo calcular este gráfico con estos filtros ({type(e).__name__}). Cambie los filtros o recargue la página."
    return {"figura": ui.figura_vacia("No se pudo calcular"), "titulo": "No se pudo calcular este gráfico",
            "sub": None, "leyenda": None, "tabla": ui.aviso(texto, "atencion"), "nota": texto, "etiqueta": texto}


def _seguro(funcion, *args):
    """El gráfico, o un aviso si algo falla (la página nunca se cae por una combinación de filtros)."""
    try:
        return funcion(*args)
    except Exception as e:  # noqa: BLE001
        return _fallo(e)


def _lectura(texto):
    return f"Lectura: {texto}"


def _sobre_columnas(fig, lineas):
    """Texto sobre cada columna, arriba del lienzo, por posición (las categorías son números: «1», «2»…)."""
    for i, texto in enumerate(lineas):
        fig.add_annotation(x=i, y=1, yref="paper", yanchor="bottom", yshift=ui.px("--e-1"), text=texto, showarrow=False,
                           align="center", font=ui.fuente_svg("tinta_2", "--t-xs"))
    fig.update_layout(margin_t=ui.px("--e-8"))
    return fig


def _marcador(rol, tam=PUNTO, opacidad=OPACIDAD):
    return dict(size=tam, color=ui.ROL[rol], opacity=opacidad, line=dict(width=0))


# --- 1. Fuente, estructura y estadística descriptiva -------------------------------------------------------------------
def _fichas_datos():
    etl, meta = _etl(), M.meta()
    aud = etl["auditoria_transacciones"]
    limpias, duplicadas = etl["n_transacciones_limpias"], aud["duplicadas"]
    sin = [AUDITORIA[k] for k in AUDITORIA if aud.get(k) == 0]
    hallazgo = f" La auditoría no encontró {', '.join(sin[:-1])} ni {sin[-1]}." if len(sin) > 1 else ""
    t = etl["tipos"]
    return [
        ui.cifra("Clientes en COFINFAD", F.entero(etl["n_clientes"]), mini=True,
                 nota=f"Un registro por cliente con {etl['n_columnas']} columnas: perfil, tenencia de productos, uso digital, "
                      "satisfacción, resúmenes de transacciones y los objetivos."),
        ui.cifra("Transacciones limpias", F.entero(limpias), mini=True,
                 nota=f"El detalle de cada cliente, tras descartar {F.entero(duplicadas)} duplicadas de las "
                      f"{F.entero(limpias + duplicadas)} del archivo.{hallazgo}"),
        ui.cifra("Predictores", F.entero(len(etl["predictores"])), mini=True,
                 nota=f"{len(t['numericas'])} numéricos, {len(t['categoricas'])} categóricos y {len(t['booleanas'])} booleanos "
                      f"({etl['n_columnas_modelo']} columnas para el modelo), tras eliminar {len(etl['eliminadas'])} columnas y "
                      f"derivar {len(etl['derivadas'])} del detalle de transacciones."),
        ui.cifra("Clientes de entrenamiento", F.entero(meta["n_entrenamiento"]), F.pct(1 - meta["fraccion_prueba"], 0), mini=True,
                 nota="Con ellos se calcula todo lo de esta página, también con los filtros."),
        ui.cifra("Clientes de prueba", F.entero(meta["n_prueba"]), F.pct(meta["fraccion_prueba"], 0), mini=True,
                 nota=f"Apartados al azar con la semilla {meta['semilla']} antes de calcular cualquier estadístico del objetivo. "
                      "No aparecen en esta página."),
        ui.cifra("Riesgo alto en el entrenamiento", F.pct(meta["prevalencia_entrenamiento"]), mini=True,
                 nota=f"Clientes con {OBJ} por encima de τ = {ui.num(_tau(), 4)}, el {_percentil()} del puntaje en el "
                      "entrenamiento."),
    ]


def _descriptiva(f):
    """Nota, tabla de las numéricas y tabla de las categóricas y booleanas, con los clientes filtrados."""
    sub = _sub(f)
    n = len(sub)
    if n < MIN_CLIENTES:
        texto = _vacio(f, n, "calcular la estadística descriptiva")["nota"]
        return texto, ui.nota(texto), ui.nota(texto)
    filas = []
    for v in [OBJ, AP] + _numericas():
        x = sub[v].astype(float)
        dm = _dec_media(v)
        filas.append([_nombre(v), _val(v, x.mean(), dm), _val(v, x.std(), dm), _val(v, x.min()), _val(v, x.quantile(0.25)),
                      _val(v, x.median()), _val(v, x.quantile(0.75)), _val(v, x.max()),
                      ui.con_signo(x.skew(), 2) if x.nunique() > 2 else F.RAYA])
    numericas = ui.tabla(["Variable", "Media", "Desviación", "Mínimo", "P25", "Mediana", "P75", "Máximo", "Asimetría"], filas,
                         compacta=True, alta=True, ocultar_movil=(2, 3, 4, 6, 7))
    filas = []
    t = _etl()["tipos"]
    for v in t["categoricas"] + t["booleanas"]:
        frec = sub[v].value_counts()
        filas.append([_nombre(v), F.entero(len(frec)), _etq(v, frec.index[0]), F.pct(frec.iloc[0] / n),
                      _etq(v, frec.index[-1]), F.pct(frec.iloc[-1] / n)])
    categoricas = ui.tabla(["Variable", "Niveles", "Más frecuente", "Su parte", "Menos frecuente", "Su parte"], filas, compacta=True,
                           num=[False, True, False, True, False, True], ocultar_movil=(4, 5))
    nota = (f"Calculada en {_quienes(f, n)}. P25 y P75 son el {_pc(P25)} y el {_pc(P75)}; la asimetría es la de Fisher (0 en una "
            "distribución simétrica). Los millones van con «M». Las categorías se muestran como vienen en COFINFAD.")
    return nota, numericas, categoricas


def _seccion_datos():
    etl, meta = _etl(), M.meta()
    return ui.seccion(
        f"COFINFAD: {F.entero(etl['n_clientes'])} clientes y {F.entero(etl['n_transacciones_limpias'])} transacciones; este EDA "
        f"usa solo los {F.entero(meta['n_entrenamiento'])} de entrenamiento",
        [ui.cifras(_fichas_datos(), clase="cifras--3", id=f"{PRE}-fichas"),
         ui.rejilla([ui.aviso(f"En la Entrega 1 el análisis exploratorio se hizo con todos los clientes, sin partir antes los datos, y el profesor "
                  f"lo señaló. Aquí la partición va primero: los {F.entero(meta['n_prueba'])} clientes de prueba se apartaron al azar "
                  f"(semilla {meta['semilla']}) antes de calcular cualquier estadístico del objetivo y no entran en ningún gráfico, "
                  f"tabla ni frase de esta página. Los filtros de arriba también trabajan solo sobre los "
                  f"{F.entero(meta['n_entrenamiento'])} de entrenamiento.", titulo="Todo, solo con el entrenamiento"),
         ui.aviso("COFINFAD es un conjunto de datos sintético: se generó con reglas. Lo que muestra esta página describe esas reglas, "
                  "no a clientes reales, y son asociaciones: ninguna es una causa.", "atencion", "Datos sintéticos")], "1-1"),
         ui.tarjeta([html.P(id=f"{PRE}-desc-nota", className="nota"), html.Div(id=f"{PRE}-desc-num")],
                    titulo=f"Estadística descriptiva de los {len(_numericas())} predictores numéricos, el puntaje y los productos activos",
                    fuentes=[F_ENT, F_VARS], id=f"{PRE}-desc"),
         ui.tarjeta(html.Div(id=f"{PRE}-desc-cat"),
                    titulo=f"Los {len(etl['tipos']['categoricas'])} predictores categóricos y los {len(etl['tipos']['booleanas'])} booleanos",
                    fuentes=[F_ENT], id=f"{PRE}-desc-cat-tarjeta")],
        subtitulo="COFINFAD (Colombian Fintech Financial Analytics Dataset) es un conjunto público de Mendeley Data con dos tablas: "
                  "un registro por cliente y el detalle de sus transacciones. Las tablas de estadística descriptiva cambian con los "
                  "filtros.",
        fuentes=[F_ETL, F_META, F_ENT], id=f"{PRE}-sec-datos")


# --- 2. El objetivo y el balance de clases ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _bordes_objetivo():
    """Los tramos del histograma del puntaje: los de eda/objetivo_hist.csv (todo el entrenamiento)."""
    h = D.tabla("eda/objetivo_hist.csv")
    return tuple(np.r_[h["desde"].to_numpy(dtype=float), float(h["hasta"].iloc[-1])])


def _histograma_objetivo(f):
    sub, T = _sub(f), _T()
    n, activo = len(sub), _activo(f)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "dibujar la distribución del puntaje")
    tau = _tau()
    b = np.asarray(_bordes_objetivo())
    x = sub[OBJ].to_numpy(dtype=float)
    cuenta = np.histogram(x, bins=b)[0]
    frac = cuenta / n
    frac_todo = np.histogram(T[OBJ].to_numpy(dtype=float), bins=b)[0] / len(T)
    centro, ancho = (b[:-1] + b[1:]) / 2, float(b[1] - b[0])
    lado = ["contiene" if a < tau < c else ("encima" if a >= tau else "debajo") for a, c in zip(b[:-1], b[1:])]
    encima = centro > tau
    tramos = [f"{ui.num(a, 3)} a {ui.num(c, 3)}" for a, c in zip(b[:-1], b[1:])]
    textos = [f"Puntaje de {t}: <b>{F.pct(p)}</b> de los clientes ({F.entero(k)})" + (" · el umbral cae en este tramo" if s == "contiene" else "")
              for t, p, k, s in zip(tramos, frac, cuenta, lado)]
    fig = go.Figure()
    for alto, forma in ((True, "barra"), (False, "rayada")):
        m = encima == alto
        fig.add_trace(go.Bar(x=centro[m], y=frac[m], width=ancho * 0.88, **ui.traza("real", forma),
                             **ui.hover_texto([t for t, k in zip(textos, m) if k])))
    if activo:
        fig.add_trace(go.Scatter(x=centro, y=frac_todo, **ui.traza("contexto", "escalon"),
                                 **ui.hover_texto([f"Todo el entrenamiento, {t}: <b>{F.pct(p)}</b>" for t, p in zip(tramos, frac_todo)])))
    ui.referencia(fig, x=tau, texto=f"τ = {ui.num(tau, 4)}")
    ui.unidad_eje(fig, "% de los clientes")
    _eje_fraccion(fig, max(frac.max(), frac_todo.max() if activo else 0))
    paso = ui.paso_redondo(float(b[-1] - b[0]), 8)
    vals, marcas = ui.marcas(math.ceil(float(b[0]) / paso) * paso, float(b[-1]), paso)
    fig.update_xaxes(tickvals=vals, ticktext=marcas, range=[float(b[0]) - ancho, float(b[-1]) + ancho])
    fig.update_layout(barmode="overlay", hovermode="closest")

    p, p_todo = float(sub[ETQ].mean()), float(T[ETQ].mean())
    q1, med, q3, _bi, _bs, cerca_i, cerca_s = _cuartiles(x)
    bajo, alto = float((x < cerca_i).mean()), float((x > cerca_s).mean())
    asim = float(pd.Series(x).skew())
    if activo:
        titulo = (f"Con los filtros, el {F.pct(p)} de los clientes queda por encima de τ = {ui.num(tau, 4)}; en todo el "
                  f"entrenamiento, el {F.pct(p_todo)}")
    else:
        titulo = (f"El {F.pct(p)} de los clientes de entrenamiento queda por encima de τ = {ui.num(tau, 4)}: el cuarto superior "
                  "del puntaje, por construcción")
    sub_ = (f"Histograma del puntaje de propensión ({OBJ}) en {_quienes(f, n)}, como fracción de ellos. Las barras sólidas quedan "
            f"por encima de τ, el {_percentil()} del puntaje en todo el entrenamiento: un umbral fijo, que no cambia con los filtros."
            + (" La línea gris claro es todo el entrenamiento." if activo else ""))
    nota = _lectura(f"el puntaje va de {ui.num(x.min(), 3)} a {ui.num(x.max(), 3)}, con mediana {ui.num(med, 3)}; la distribución "
                    f"{_forma(asim)}, con asimetría {ui.con_signo(asim, 2)}. Por la regla de Tukey ({ui.num(TUKEY, 1)} × RIC), el "
                    f"{F.pct(bajo)} queda bajo la cerca inferior ({ui.num(cerca_i, 3)}) y el {F.pct(alto)} sobre la superior "
                    f"({ui.num(cerca_s, 3)}): son puntajes válidos y no se recortan.")
    leyenda = [("Por encima de τ: riesgo alto", "real", "barra"), ("Por debajo de τ", "real", "rayada")]
    if activo:
        leyenda.append(("Todo el entrenamiento", "contexto", "escalon"))
    encabezados = ["Puntaje", "Clientes", "% de los clientes"] + (["% en todo el entrenamiento"] if activo else []) + ["Frente a τ"]
    filas = [[t, F.entero(k), F.pct(pf)] + ([F.pct(pt)] if activo else []) + [{"encima": "por encima", "debajo": "por debajo",
                                                                               "contiene": "contiene a τ"}[s]]
             for t, k, pf, pt, s in zip(tramos, cuenta, frac, frac_todo, lado)]
    tabla = ui.tabla(encabezados, filas, compacta=True, alta=True, num=[False, True, True] + ([True] if activo else []) + [False])
    etiqueta = (f"{titulo}. Mediana del puntaje {ui.num(med, 3)}; asimetría {ui.con_signo(asim, 2)}; "
                f"{F.pct(bajo)} bajo la cerca inferior de Tukey.")
    return {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": ui.leyenda(leyenda), "tabla": tabla, "nota": nota, "etiqueta": etiqueta}


def _balance(f):
    sub, T = _sub(f), _T()
    n, activo = len(sub), _activo(f)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "medir el balance de clases")
    k = int(sub[ETQ].sum())
    p, p_todo = k / n, float(T[ETQ].mean())
    fr = [p, 1 - p]
    cats = [f"Riesgo alto · {F.pct(p)}", f"Resto · {F.pct(1 - p)}"]
    cuentas = [k, n - k]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=cats, x=fr, orientation="h", **ui.barras_foco("real", cats, foco=cats[0], ancho=ui.ancho_barra(2, 240)),
                         **ui.hover_texto([f"{c.split(' · ')[0]}: <b>{F.pct(x)}</b> ({F.entero(m)} clientes)" for c, x, m in zip(cats, fr, cuentas)])))
    if activo:
        fig.add_trace(go.Scatter(y=cats, x=[p_todo, 1 - p_todo], **ui.traza("contexto", "punto"),
                                 **ui.hover_texto([f"Todo el entrenamiento: <b>{F.pct(x)}</b>" for x in (p_todo, 1 - p_todo)])))
    vals, marcas = ui.marcas(0, 1, 0.25)
    fig.update_xaxes(tickvals=vals, ticktext=["0" if x == 0 else F.pct(x, 0) for x in vals], range=[0, 1 + AIRE / 2], showgrid=True,
                     gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=cats[::-1], tickfont=ui.fuente_svg("tinta_2", "--t-s"))
    fig.update_layout(hovermode="closest")
    if k == 0 or k == n:
        titulo = "Con estos filtros todos los clientes quedan del mismo lado de τ"
        nota = _lectura("no hay dos clases que comparar con estos filtros.")
    else:
        titulo = f"Desbalance de {ui.num((1 - p) / p, 1)} a 1: {F.entero(k)} clientes de riesgo alto frente a {F.entero(n - k)}"
        balanceos = F.enumerar(D.NOMBRE_BALANCEO[b] for b in D.ORDEN_BALANCEOS if b != "ninguno")
        if activo:
            nota = _lectura(f"con los filtros, uno de cada {ui.num(1 / p, 1)} clientes es de riesgo alto, frente a uno de cada "
                            f"{ui.num(1 / p_todo, 1)} en todo el entrenamiento. Es una asociación del subconjunto, no un efecto de los "
                            "filtros.")
        else:
            nota = _lectura(f"uno de cada {ui.num(1 / p, 1)} clientes es de riesgo alto: el problema está desbalanceado por "
                            f"construcción, porque τ es el {_percentil()}. Por eso el proyecto compara no balancear con {balanceos}, "
                            "siempre ajustados dentro de cada pliegue.")
    sub_ = (f"Clientes de riesgo alto (puntaje por encima de τ) y el resto, en {_quienes(f, n)}."
            + (" Los puntos gris claro son todo el entrenamiento." if activo else ""))
    leyenda = [("Riesgo alto", "real", "barra"), ("Resto", "real", "rayada")] + ([("Todo el entrenamiento", "contexto", "punto")] if activo else [])
    filas = [[c.split(" · ")[0], F.entero(m), F.pct(x)] + ([F.pct(t)] if activo else []) for c, m, x, t in zip(cats, cuentas, fr, (p_todo, 1 - p_todo))]
    tabla = ui.tabla(["Clase", "Clientes", "Parte"] + (["Parte en todo el entrenamiento"] if activo else []), filas, compacta=True)
    etiqueta = f"{titulo}. Riesgo alto: {F.pct(p)}; resto: {F.pct(1 - p)}."
    return {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": ui.leyenda(leyenda), "tabla": tabla, "nota": nota, "etiqueta": etiqueta}


def _seccion_objetivo():
    meta = M.meta()
    return ui.seccion(
        f"El objetivo: riesgo alto es el cuarto superior del puntaje ({F.pct(meta['prevalencia_entrenamiento'])} del entrenamiento), "
        "no una fuga observada",
        [ui.rejilla([ui.grafico(f"{PRE}-hist", fuentes=[F_ENT, F_HIST, F_META]),
                     ui.grafico(f"{PRE}-balance", alto="bajo", fuentes=[F_ENT, F_META])], "2-1")],
        subtitulo=f"COFINFAD trae para cada cliente {OBJ}, un puntaje de propensión a la fuga. La etiqueta del trabajo marca como "
                  f"riesgo alto a quien supera τ, el {_percentil()} del puntaje, calculado solo con el entrenamiento.",
        fuentes=[F_ENT, F_HIST, F_META], id=f"{PRE}-sec-objetivo")


# --- 3. La trampa de los productos activos ------------------------------------------------------------------------------
def _escalera(f):
    sub = _sub(f)
    n = len(sub)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "dibujar la escalera")
    tau = _tau()
    g = sub.groupby(AP).agg(n=(OBJ, "size"), media=(OBJ, "mean"), alto=(ETQ, "mean")).sort_index()
    cats = [_num_justo(c, 0) for c in g.index.astype(float)]
    saltos = g["media"].diff()
    medias = [ui.num(m, 4) for m in g["media"]]
    pasos = ["" if math.isnan(s) else ui.con_signo(s, 4) for s in saltos]
    textos = [f"{c} {'producto activo' if c == '1' else 'productos activos'}: puntaje medio <b>{m}</b> · {F.entero(k)} clientes"
              for c, m, k in zip(cats, medias, g["n"])]
    fig = go.Figure(go.Bar(x=cats, y=g["media"], **ui.traza("real", "rayada", ancho=ui.ancho_barra(len(cats), 800)), **ui.hover_texto(textos)))
    _sobre_columnas(fig, [f"{m}<br>{p or FIJO}" for m, p in zip(medias, pasos)])
    ui.referencia(fig, y=tau, texto=f"τ = {ui.num(tau, 4)}")
    tope = max(float(g["media"].max()), tau)
    vals, marcas = ui.marcas(0, tope, ui.paso_redondo(tope, 4))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(type="category", range=[-0.5, len(cats) - 0.5])
    fig.update_yaxes(range=[0, tope * 1.12], tickvals=vals, ticktext=marcas)

    r = float(np.corrcoef(sub[AP].to_numpy(dtype=float), sub[OBJ].to_numpy(dtype=float))[0, 1]) if sub[AP].nunique() > 1 else float("nan")
    validos = saltos.dropna()
    if len(validos):
        salto = float(validos.mean())
        parejos = (validos.max() - validos.min()) <= 0.1 * abs(salto)
        cuantos = PALABRA.get(len(validos), F.entero(len(validos)))
        titulo = (f"Cada producto activo adicional baja el puntaje medio cerca de {ui.num(abs(salto), 3)}: {cuantos} "
                  + ("pasos casi idénticos" if parejos else "pasos de distinto tamaño"))
    else:
        titulo = "Con estos filtros solo queda un número de productos activos"
    sub_ = (f"Media del puntaje según el número de productos activos, en {_quienes(f, n)}; sobre cada barra, la media y el salto "
            "frente al peldaño anterior. "
            + (f"Esa sola variable explica el {F.pct(r * r)} de la varianza del puntaje (r = {ui.num(r, 3)})." if math.isfinite(r) else ""))
    uno = g["alto"].iloc[0]
    tres = sub.loc[sub[AP] >= TRES, ETQ].mean() if (sub[AP] >= TRES).any() else float("nan")
    nota = _lectura("un paso constante es la firma de la regla con que se generó el puntaje, no de una asociación empírica: un modelo "
                    "que viera esta variable recuperaría la fórmula en vez de predecir. Por eso se excluye de los predictores (fuga de "
                    f"información). Con {cats[0]} producto, el {F.pct(uno)} es de riesgo alto"
                    + (f"; con {PALABRA[TRES]} o más, el {F.pct(tres)}." if math.isfinite(tres) else "."))
    filas = [[c, F.entero(k), m, p or F.RAYA, F.pct(a)] for c, k, m, p, a in zip(cats, g["n"], medias, pasos, g["alto"])]
    tabla = ui.tabla(["Productos", "Clientes", "Puntaje medio", "Salto", "Riesgo alto"], filas, compacta=True, ocultar_movil=(1, 4))
    etiqueta = f"{titulo}. Puntaje medio según los productos activos: " + "; ".join(f"{c}: {m}" for c, m in zip(cats, medias)) + "."
    return {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": None, "tabla": tabla, "nota": nota, "etiqueta": etiqueta}


def _lectura_trampa():
    techo = D.techo()
    return ui.tarjeta_insight(
        ui.num(techo["r2_oraculo"], 4),
        "es el R² de un modelo que sí ve los productos activos: reconstruye el puntaje casi entero.",
        f"Sin esa variable, el techo de información queda en un R² de {ui.num(techo['r2'], 4)}. Por eso se excluye: es la receta del "
        "objetivo, no una señal. Estas dos cifras son resultados de modelos medidos al final del proyecto, no de este EDA; la "
        "decisión de excluirla se tomó antes, con la escalera del entrenamiento.",
        chip_="La trampa", id=f"{PRE}-trampa")


def _seccion_trampa():
    r = float(D.fila(D.tabla("eda/asociacion.csv"), variable=AP)["pearson"])
    return ui.seccion(
        f"La trampa: el puntaje se generó con el número de productos activos (r = {ui.num(r, 3)})",
        [ui.rejilla([ui.grafico(f"{PRE}-escalera", fuentes=[F_ENT, F_ASOC, F_META]), _lectura_trampa()], "2-1")],
        subtitulo="La media del puntaje baja en una escalera de pasos casi iguales. La variable se muestra aquí como explicación: "
                  "ningún modelo la ve.",
        fuentes=[F_ENT, F_ASOC, F_TECHO], id=f"{PRE}-sec-trampa")


# --- 4. Distribuciones univariadas --------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _bordes(v):
    """Tramos fijos del histograma de una variable continua (con todo el entrenamiento, para que los filtros se comparen) y
    si tiene cola larga: entonces los tramos llegan al percentil 99 y lo de más arriba va junto en una barra aparte."""
    x = _T()[v].to_numpy(dtype=float)
    lo, hi = float(x.min()), float(x.max())
    p1, p99 = (float(q) for q in np.quantile(x, [1 - P_COLA, P_COLA]))
    cola = hi > p99 + COLA * (p99 - p1)
    return tuple(np.linspace(lo, p99 if cola else hi, N_TRAMOS + 1)), cola


def _discreta(v):
    return _tipo(v) != "numerica" or _distintos(v) <= MAX_DISCRETOS


def _univariada(f, v):
    sub, T = _sub(f), _T()
    n, activo, N = len(sub), _activo(f), len(T)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "dibujar la distribución"), None, "grafico__lienzo"
    nombre, unidad, tipo = _nombre(v), _unidad(v), _tipo(v)
    fig = go.Figure()
    alto = "normal"
    leyenda = []
    if _discreta(v):
        niveles = list(_niveles(v))
        cuenta = sub[v].value_counts().reindex(niveles, fill_value=0).to_numpy()
        frac, frac_todo = cuenta / n, T[v].value_counts().reindex(niveles, fill_value=0).to_numpy() / N
        etiquetas = [_etq(v, x) for x in niveles]
        foco = etiquetas[int(np.argmax(frac))]
        textos = [f"{nombre}, {e}{unidad if tipo == 'numerica' else ''}: <b>{F.pct(p)}</b> de los clientes ({F.entero(k)})"
                  for e, p, k in zip(etiquetas, frac, cuenta)]
        textos_todo = [f"Todo el entrenamiento, {e}: <b>{F.pct(p)}</b>" for e, p in zip(etiquetas, frac_todo)]
        tope = max(frac.max(), frac_todo.max() if activo else 0)
        if tipo == "categorica" and len(niveles) > MAX_HORIZONTAL:
            alto = "alto" if len(niveles) > 2 * MAX_HORIZONTAL else "normal"
            ancho = ui.ancho_barra(len(niveles), 420 if alto == "alto" else 320)
            fig.add_trace(go.Bar(y=etiquetas, x=frac, orientation="h", **ui.barras_foco("real", etiquetas, foco=foco, ancho=ancho),
                                 **ui.hover_texto(textos)))
            if activo:
                fig.add_trace(go.Scatter(y=etiquetas, x=frac_todo, **ui.traza("contexto", "punto"), **ui.hover_texto(textos_todo)))
            fig.update_yaxes(type="category", categoryorder="array", categoryarray=etiquetas[::-1], showgrid=False,
                             tickfont=ui.fuente_svg("tinta_2", "--t-xs" if alto == "alto" else "--t-s"))
            _eje_fraccion(fig, tope, eje="x")
            fig.update_xaxes(showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
        else:
            fig.add_trace(go.Bar(x=etiquetas, y=frac, **ui.barras_foco("real", etiquetas, foco=foco, ancho=ui.ancho_barra(len(etiquetas))),
                                 **ui.hover_texto(textos)))
            if activo:
                fig.add_trace(go.Scatter(x=etiquetas, y=frac_todo, **ui.traza("contexto", "punto"), **ui.hover_texto(textos_todo)))
            ui.unidad_eje(fig, "% de los clientes")
            salto = math.ceil(len(etiquetas) / MAX_ROTULOS)
            fig.update_xaxes(type="category", range=[-0.5, len(etiquetas) - 0.5], tickvals=etiquetas[::salto], ticktext=etiquetas[::salto])
            _eje_fraccion(fig, tope)
        leyenda = [("El valor más frecuente", "real", "barra"), ("Los demás", "real", "rayada")]
        if activo:
            leyenda.append(("Todo el entrenamiento", "contexto", "punto"))
        filas = [[e, F.entero(k), F.pct(p)] + ([F.pct(t)] if activo else []) for e, k, p, t in zip(etiquetas, cuenta, frac, frac_todo)]
        que = "Valor" if tipo == "numerica" else "Categoría"
        cola = False
    else:
        bordes, cola = _bordes(v)
        b = np.asarray(bordes)
        w = float(b[1] - b[0])
        x, xt = sub[v].to_numpy(dtype=float), T[v].to_numpy(dtype=float)
        cuenta = np.histogram(x[x <= b[-1]], bins=b)[0]
        frac, frac_todo = cuenta / n, np.histogram(xt[xt <= b[-1]], bins=b)[0] / N
        centro = (b[:-1] + b[1:]) / 2
        tramos = [f"de {_val(v, a)} a {_val(v, c)}{unidad}" for a, c in zip(b[:-1], b[1:])]
        textos = [f"{nombre} {t}: <b>{F.pct(p)}</b> de los clientes ({F.entero(k)})" for t, p, k in zip(tramos, frac, cuenta)]
        fig.add_trace(go.Bar(x=centro, y=frac, width=w * 0.88, **ui.traza("real", "barra"), **ui.hover_texto(textos)))
        filas = [[t, F.entero(k), F.pct(p)] + ([F.pct(q)] if activo else []) for t, k, p, q in zip(tramos, cuenta, frac, frac_todo)]
        tope = max(frac.max(), frac_todo.max() if activo else 0)
        leyenda = [(f"Clientes por tramo de {_en_frase(v)}", "real", "barra")]
        if cola:
            extra, extra_todo = int((x > b[-1]).sum()), float((xt > b[-1]).mean())
            texto_extra = f"más de {_val(v, b[-1])}{unidad} ({_pc(P_COLA)} de todo el entrenamiento), hasta {_val(v, x.max())}"
            fig.add_trace(go.Bar(x=[b[-1] + 1.5 * w], y=[extra / n], width=w * 0.88, **ui.traza("real", "rayada"),
                                 **ui.hover_texto([f"{nombre}, {texto_extra}: <b>{F.pct(extra / n)}</b> ({F.entero(extra)} clientes)"])))
            filas.append([texto_extra[:1].upper() + texto_extra[1:], F.entero(extra), F.pct(extra / n)] + ([F.pct(extra_todo)] if activo else []))
            leyenda.append((f"Por encima del {_pc(P_COLA)}, juntos", "real", "rayada"))
            tope = max(tope, extra / n)
        if activo:
            fig.add_trace(go.Scatter(x=centro, y=frac_todo, **ui.traza("contexto", "escalon"),
                                     **ui.hover_texto([f"Todo el entrenamiento, {t}: <b>{F.pct(p)}</b>" for t, p in zip(tramos, frac_todo)])))
            leyenda.append(("Todo el entrenamiento", "contexto", "escalon"))
        ui.unidad_eje(fig, "% de los clientes")
        _eje_fraccion(fig, tope)
        vals, marcas = _marcas(float(b[0]), float(b[-1]))
        fig.update_xaxes(tickvals=vals, ticktext=marcas, range=[float(b[0]) - w, float(b[-1]) + (2.5 if cola else 1) * w])
        fig.update_layout(barmode="overlay")
        que = "Tramo"
    fig.update_layout(hovermode="closest")
    tabla = ui.tabla([que, "Clientes", "% de los clientes"] + (["% en todo el entrenamiento"] if activo else []), filas, compacta=True,
                     alta=True, num=[False, True, True] + ([True] if activo else []))

    # --- la lectura y la ficha
    origen = ("derivada del detalle de transacciones en el ETL" if v in _etl()["derivadas"] else "del archivo de clientes")
    descripcion = str(M.variables().at[v, "descripcion"]).rstrip(".") if v in M.variables().index else nombre
    falta = (f"{F.pct(sub[f'falta__{v}'].mean())} en el archivo original" if v in _faltan() else "ninguno")
    if tipo == "numerica":
        xs = sub[v].astype(float)
        q1, med, q3, _bi, _bs, cerca_i, cerca_s = _cuartiles(xs.to_numpy())
        atip = float(((xs < cerca_i) | (xs > cerca_s)).mean())
        asim = float(xs.skew()) if xs.nunique() > 2 else 0.0
        rho = _rho(xs, sub[OBJ])
        titulo = f"{nombre}: mediana {_val(v, med)}{unidad}; la mitad central va de {_val(v, q1)} a {_val(v, q3)}"
        frases = [f"la distribución {_forma(asim)}; la media ({_val(v, xs.mean(), _dec_media(v))}) "
                  + ("supera a" if xs.mean() > med else "queda por debajo de" if xs.mean() < med else "coincide con")
                  + f" la mediana. El {F.pct(atip)} queda fuera de las cercas de Tukey"
                  + (" (el rango intercuartílico es 0: todo valor distinto del más común cuenta como atípico)" if q3 == q1 else "") + "."]
        if v in _faltan():
            frases.append(f"El {F.pct(float((xs == 0).mean()))} vale 0: son los clientes sin el dato en el archivo original, a los que "
                          "el ETL imputa 0 (sección de faltantes).")
        if cola:
            frases.append(f"La barra rayada junta la cola por encima del {_pc(P_COLA)}.")
        pares = [("Clientes", F.entero(n)), ("Media", _val(v, xs.mean(), _dec_media(v))), ("Desviación estándar", _val(v, xs.std(), _dec_media(v))),
                 ("Mínimo", _val(v, xs.min())), (_pc(P25).capitalize(), _val(v, q1)), ("Mediana", _val(v, med)),
                 (_pc(P75).capitalize(), _val(v, q3)),
                 ("Máximo", _val(v, xs.max())), ("Asimetría", ui.con_signo(asim, 2)),
                 (f"Atípicos (Tukey, {ui.num(TUKEY, 1)} × RIC)", F.pct(atip)),
                 ("ρ de Spearman con el puntaje", ui.con_signo(rho, 3) if math.isfinite(rho) else F.RAYA)]
    else:
        frec = sub[v].value_counts()
        mayor, menor = frec.index[0], frec.index[-1]
        eta = _eta2(sub[OBJ], sub[v].astype(str))
        titulo = f"{nombre}: «{_etq(v, mayor)}» reúne el {F.pct(frec.iloc[0] / n)} de los clientes"
        frases = [f"la categoría más frecuente es «{_etq(v, mayor)}» ({F.pct(frec.iloc[0] / n)}) y la menos frecuente, «{_etq(v, menor)}» "
                  f"({F.pct(frec.iloc[-1] / n)})" + (", con " + F.entero(len(frec)) + " niveles." if len(frec) > 2 else ".")
                  + (" Las categorías se muestran como vienen en COFINFAD." if tipo == "categorica" else "")]
        pares = [("Clientes", F.entero(n)), ("Niveles", F.entero(len(frec))),
                 ("Más frecuente", f"{_etq(v, mayor)} ({F.pct(frec.iloc[0] / n)})"),
                 ("Menos frecuente", f"{_etq(v, menor)} ({F.pct(frec.iloc[-1] / n)})"),
                 ("η² con el puntaje", ui.num(eta, 4) if math.isfinite(eta) else F.RAYA)]
    pares += [("Faltantes", falta), ("Origen", origen)]
    sub_ = (f"{descripcion}. Distribución en {_quienes(f, n)}, como fracción de ellos."
            + (" La línea o los puntos gris claro son todo el entrenamiento." if activo else ""))
    nota = _lectura(" ".join(frases))
    etiqueta = f"{titulo}. " + " ".join(frases)
    clase = "grafico__lienzo" if alto == "normal" else f"grafico__lienzo grafico__lienzo--{alto}"
    g = {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": ui.leyenda(leyenda), "tabla": tabla, "nota": nota, "etiqueta": etiqueta}
    return g, ui.ficha(pares), clase


def _opciones(variables, rotulos=None):
    """Opciones de un desplegable: el nombre visible; la búsqueda mira el nombre con y sin tildes y la clave."""
    rotulos = rotulos or {}
    return [{"label": rotulos.get(v, _nombre(v)), "value": v, "search": f"{_nombre(v)} {_sin_tildes(_nombre(v))} {v}"} for v in variables]


def _seccion_univariadas():
    edades = [c for c in _cohortes()]
    n_edades = _distintos("age")
    selector = ui.selector(f"{PRE}-uni", _opciones(_predictores()), INICIAL["uni"], "Variable", buscar=True)
    return ui.seccion(
        f"Cada variable por separado: la edad solo toma {PALABRA.get(n_edades, F.entero(n_edades))} valores, en "
        f"{PALABRA.get(len(edades), F.entero(len(edades)))} cohortes",
        [ui.rejilla([ui.grafico(f"{PRE}-uni-graf", fuentes=[F_ENT, F_VARS, F_FALT]),
                     ui.tarjeta(html.Div(id=f"{PRE}-uni-ficha"), titulo="La variable elegida, en cifras", fuentes=[F_ENT, F_VARS],
                                id=f"{PRE}-uni-resumen")], "2-1")],
        subtitulo=f"Elija uno de los {len(_predictores())} predictores (van primero los que más pesan en la EBM). Las numéricas con "
                  f"pocos valores y las categóricas van en barras, una por valor; las continuas, en un histograma de "
                  f"{N_TRAMOS} tramos fijos. Las edades son {F.enumerar(_num_justo(a, 0) for a in _niveles('age'))} años: un rasgo "
                  "del conjunto sintético, que se generó por cohortes.",
        controles=[selector], fuentes=[F_ENT, F_VARS], id=f"{PRE}-sec-univariadas")


# --- 5. Explorador bivariado ----------------------------------------------------------------------------------------------
def _opciones_color():
    t = _etl()["tipos"]
    base = [{"label": "Sin color", "value": "ninguno"}, {"label": "Riesgo alto (la etiqueta)", "value": ETQ},
            {"label": "Edad (las cohortes)", "value": "age"}]
    ordinales = [v for v in t["categoricas"] if v in ORDINALES]
    numericas = [v for v in _predictores() if _tipo(v) == "numerica" and v != "age"]
    return base + _opciones(ordinales + t["booleanas"] + numericas + [AP])


def _valores_color():
    return [o["value"] for o in _opciones_color()]


def _grupos_color(sub, color):
    """(rótulos, código de grupo de cada fila de sub, roles) del color del explorador, o None sin color. Los grupos van de
    menos a más (la rampa de grises va de claro a oscuro): no/sí, el orden de una ordinal, las cohortes o los tercios."""
    if color == "ninguno" or color not in _valores_color():
        return None
    if color == ETQ:
        codigos, rotulos = sub[ETQ].astype(int).to_numpy(), ["Resto", "Riesgo alto"]
    elif color == "age":
        codigos, rotulos = np.full(len(sub), -1), []
        for i, (c, e0, e1) in enumerate(_cohortes()):
            codigos[sub["age"].between(e0, e1).to_numpy()] = i
            rotulos.append(f"{c} años")
    elif _tipo(color) == "booleana":
        codigos, rotulos = sub[color].astype(int).to_numpy(), [f"{_nombre(color)}: no", f"{_nombre(color)}: sí"]
    elif _tipo(color) == "categorica":
        niveles = list(_niveles(color))
        codigos, rotulos = pd.Categorical(sub[color], categories=niveles).codes.astype(int), [str(x) for x in niveles]
    else:
        x = sub[color].astype(float)
        unicos = np.sort(x.unique())
        if len(unicos) <= len(TERCIOS):
            codigos = np.searchsorted(unicos, x.to_numpy())
            rotulos = [f"{_etq(color, u)}{_unidad(color)}" for u in unicos]
        else:
            codigos = pd.qcut(x, len(TERCIOS), labels=False, duplicates="drop").to_numpy().astype(int)
            k = int(codigos.max()) + 1
            nombres = TERCIOS if k == len(TERCIOS) else ("mitad inferior", "mitad superior")[:k]
            rotulos = [f"{_nombre(color)}, {nombres[i]} ({_val(color, x[codigos == i].min())} a {_val(color, x[codigos == i].max())})"
                       for i in range(k)]
    presentes = [i for i in range(len(rotulos)) if (codigos == i).any()]
    if not presentes or len(presentes) > max(RAMPA):
        return None
    nuevo = np.full(len(codigos), -1)
    for j, i in enumerate(presentes):
        nuevo[codigos == i] = j
    return [rotulos[i] for i in presentes], nuevo, RAMPA[len(presentes)]


def _cajas(v):
    """La x se dibuja en cajas: categórica, booleana o numérica con pocos valores."""
    return _tipo(v) != "numerica" or _distintos(v) <= MAX_CAJAS


def _rango_eje(v, valores):
    """Recorrido del eje de una variable: completo, o hasta su percentil 99 si tiene cola larga (y cuántos quedan fuera)."""
    lo, hi = float(np.min(valores)), float(np.max(valores))
    if _tipo(v) == "numerica" and v not in (OBJ, AP) and _distintos(v) > MAX_DISCRETOS and _bordes(v)[1]:
        hi = float(np.quantile(valores, P_COLA))
    margen = (hi - lo) * 0.03 or 1
    return lo - margen, hi + margen, int((np.asarray(valores) > hi).sum())


def _bivariado(f, vx, vy, color):
    sub = _sub(f)
    n = len(sub)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "relacionar dos variables")
    nx, ny = _nombre(vx), _nombre(vy)
    grupos = _grupos_color(sub, color)
    rotulos, codigos, roles = grupos if grupos else ([nx], np.zeros(n, dtype=int), ["real"])
    ys = sub[vy].to_numpy(dtype=float)
    fig = go.Figure()
    notas, filas = [], []
    if _cajas(vx):
        niveles = [x for x in _niveles(vx) if (sub[vx] == x).any()]
        etiquetas = [_etq(vx, x) for x in niveles]
        cajas = []
        for x in niveles:
            yv = ys[(sub[vx] == x).to_numpy()]
            cajas.append(_cuartiles(yv) + (len(yv),))
        q1, med, q3, bi, bs = ([c[i] for c in cajas] for i in range(5))
        fig.add_trace(go.Box(x=etiquetas, q1=q1, median=med, q3=q3, lowerfence=bi, upperfence=bs, boxpoints=False, hoverinfo="skip",
                             line=dict(color=ui.ROL["real"], width=ui.ANCHO_LINEA / 2 + 0.5), fillcolor=ui.alfa(ui.ROL["real"], ui.LAVADO),
                             width=0.5))
        textos = [f"{nx}: {e} · mediana de {_en_frase(vy)} <b>{_val(vy, c[1])}</b> · caja de {_val(vy, c[0])} a {_val(vy, c[2])} · "
                  f"bigotes de {_val(vy, c[3])} a {_val(vy, c[4])} · {F.entero(c[7])} clientes" for e, c in zip(etiquetas, cajas)]
        fig.add_trace(go.Scatter(x=etiquetas, y=med, mode="markers", marker=dict(size=2 * ui.MARCADOR, opacity=0, color=ui.ROL["real"]),
                                 **ui.hover_texto(textos)))
        if grupos:
            for j, (rot, rol) in enumerate(zip(rotulos, roles)):
                xs_, meds = [], []
                for e, x in zip(etiquetas, niveles):
                    m = (sub[vx] == x).to_numpy() & (codigos == j)
                    if m.any():
                        xs_.append(e)
                        meds.append(float(np.median(ys[m])))
                fig.add_trace(go.Scatter(x=xs_, y=meds, **ui.traza(rol, "punto"),
                                         **ui.hover_texto([f"{rot}, {nx} {e}: mediana de {_en_frase(vy)} <b>{_val(vy, m_)}</b>" for e, m_ in zip(xs_, meds)])))
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=etiquetas)
        lo, hi = min(bi), max(bs)
        margen = (hi - lo) * 0.05 or 1
        vals, marcas = _marcas(lo - margen, hi + margen, 4)
        fig.update_yaxes(range=[lo - margen, hi + margen], tickvals=vals, ticktext=marcas)
        eta = _eta2(ys, sub[vx].astype(str).to_numpy())
        i_min, i_max = int(np.argmin(med)), int(np.argmax(med))
        titulo = (f"{_nombre(vy)} según {_en_frase(vx)}: la mediana va de {_val(vy, med[i_min])} ({etiquetas[i_min]}) a "
                  f"{_val(vy, med[i_max])} ({etiquetas[i_max]})")
        lectura = (f"{nx} explica el {F.pct(eta)} de la varianza de {_en_frase(vy)} (η² = {ui.num(eta, 4)}): una asociación "
                   f"{_fuerza(math.sqrt(eta)) if math.isfinite(eta) else 'que no se puede medir'}.")
        sub_ = (f"Una caja por valor de {_en_frase(vx)} en {_quienes(f, n)}: la caja va del {_pc(P25)} al {_pc(P75)}, la raya es la "
                f"mediana y los bigotes llegan al último valor dentro de las cercas de Tukey.")
        filas = [[e, F.entero(c[7]), _val(vy, c[0]), _val(vy, c[1]), _val(vy, c[2])] for e, c in zip(etiquetas, cajas)]
        tabla = ui.tabla([nx, "Clientes", "P25", "Mediana", "P75"], filas, compacta=True, alta=True)
        leyenda = []
        if grupos:
            sub_ += " Los puntos son la mediana de cada grupo del color."
    else:
        xs = sub[vx].to_numpy(dtype=float)
        mascara = _mascara(f)
        posiciones = np.flatnonzero(mascara)
        orden = _orden()
        muestra = np.searchsorted(posiciones, orden[mascara[orden]][:N_MUESTRA])
        for j, (rot, rol) in enumerate(zip(rotulos, roles)):
            sel = muestra[codigos[muestra] == j]
            textos = [f"{nx}: {_val(vx, a)} · {ny}: <b>{_val(vy, b)}</b>" + (f" · {rot}" if grupos else "") for a, b in zip(xs[sel], ys[sel])]
            fig.add_trace(go.Scatter(x=xs[sel], y=ys[sel], mode="markers", marker=_marcador(rol), **ui.hover_texto(textos)))
        lineas = []
        for j, (rot, rol) in enumerate(zip(rotulos, roles)):
            m = codigos == j
            if m.sum() < N_TRAMOS_LINEA:
                continue
            bordes = np.unique(np.quantile(xs[m], np.linspace(0, 1, N_TRAMOS_LINEA + 1)))
            if len(bordes) < 3:
                continue
            tramo = np.clip(np.searchsorted(bordes, xs[m], side="right") - 1, 0, len(bordes) - 2)
            t = pd.DataFrame({"x": xs[m], "y": ys[m], "t": tramo}).groupby("t").median()
            lineas.append((rot, t))
            fig.add_trace(go.Scatter(x=t["x"], y=t["y"], **ui.traza(rol, "linea", marcadores=True),
                                     **ui.hover_texto([f"{rot if grupos else 'Todos'}: mediana de {_en_frase(vy)} <b>{_val(vy, b)}</b> "
                                                       f"con {_en_frase(vx)} cerca de {_val(vx, a)}" for a, b in zip(t["x"], t["y"])])))
        x0, x1, fuera_x = _rango_eje(vx, xs[muestra])
        y0, y1, fuera_y = _rango_eje(vy, ys[muestra])
        vals, marcas = _marcas(x0, x1)
        fig.update_xaxes(range=[x0, x1], tickvals=vals, ticktext=marcas)
        vals, marcas = _marcas(y0, y1, 4)
        fig.update_yaxes(range=[y0, y1], tickvals=vals, ticktext=marcas)
        if fuera_x or fuera_y:
            notas.append(f"Los ejes de las variables con cola larga llegan a su {_pc(P_COLA)}: quedan fuera {F.entero(fuera_x + fuera_y)} "
                         "puntos de la muestra.")
        rho = _rho(xs, ys)
        signo = "positiva: a mayor" if rho > 0 else "negativa: a mayor"
        mas = "mayor" if rho > 0 else "menor"
        if math.isfinite(rho):
            titulo = f"{nx} y {_en_frase(vy)}: ρ de Spearman = {ui.con_signo(rho, 3)}, una asociación {_fuerza(rho)}"
            lectura = (f"la asociación es {_fuerza(rho)} y {signo} {_en_frase(vx)}, {mas} {_en_frase(vy)} en promedio"
                       + ".")
        else:
            titulo, lectura = f"{nx} y {_en_frase(vy)}: sin variación suficiente para medir la asociación", "con estos filtros una de las dos variables no varía."
        k = len(muestra)
        sub_ = (f"Muestra fija de {F.entero(k)} de {_quienes(f, n)} (semilla {M.SEMILLA}); la línea de cada grupo es la mediana de "
                f"{_en_frase(vy)} por tramo de {_en_frase(vx)}, calculada con todos los clientes filtrados.")
        for rot, t in lineas:
            for _i, fila in t.iterrows():
                filas.append([rot, _val(vx, fila["x"]), _val(vy, fila["y"])])
        tabla = ui.tabla(["Grupo", f"{nx} (mediana del tramo)", f"Mediana de {_en_frase(vy)}"], filas, compacta=True, alta=True,
                         num=[False, True, True])
        leyenda = [("Cada punto, un cliente de la muestra", "real", "punto")] if not grupos else []
    if grupos:
        leyenda = [(rot, rol, "punto") for rot, rol in zip(rotulos, roles)]
        meds = [float(np.median(ys[codigos == j])) for j in range(len(rotulos))]
        j_min, j_max = int(np.argmin(meds)), int(np.argmax(meds))
        por_grupo = (f" Por {_en_frase(color) if color not in (ETQ, 'age') else ('la etiqueta' if color == ETQ else 'cohorte de edad')}, "
                     f"la mediana de {_en_frase(vy)} va de {_val(vy, meds[j_min])} ({rotulos[j_min]}) a {_val(vy, meds[j_max])} "
                     f"({rotulos[j_max]}).")
        if color == ETQ and vy == OBJ:
            por_grupo += " Es así por construcción: la etiqueta es el puntaje por encima de τ."
        lectura += por_grupo
        notas.append("Los grises van de claro a oscuro según el orden del grupo; solo distinguen grupos.")
    fig.update_layout(hovermode="closest")
    if vx == vy:
        notas.append("El eje x y el eje y son la misma variable.")
    nota = _lectura(lectura) + (" " + " ".join(notas) if notas else "") + " Asociación, no causa: los datos son sintéticos."
    etiqueta = f"{titulo}. {lectura}"
    return {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": ui.leyenda(leyenda, ordenar=False), "tabla": tabla, "nota": nota,
            "etiqueta": etiqueta}


@lru_cache(maxsize=1)
def _maximo_rho():
    """El mayor |ρ| de Spearman de un predictor con el puntaje, en todo el entrenamiento (eda/asociacion.csv)."""
    a = D.tabla("eda/asociacion.csv")
    a = a[(a["medida"] == "spearman") & (a["variable"] != AP)]
    fila = a.loc[a["valor"].abs().idxmax()]
    return float(fila["valor"]), str(fila["variable"])


def _seccion_bivariado():
    rho, v = _maximo_rho()
    x_ops = _opciones(list(_predictores()) + [AP], {AP: "Productos activos (excluida)"})
    y_ops = _opciones([OBJ] + [p for p in _predictores() if _tipo(p) == "numerica"] + [AP], {AP: "Productos activos (excluida)"})
    controles = ui.rejilla([ui.selector(f"{PRE}-x", x_ops, INICIAL["x"], "Eje x", buscar=True),
                            ui.selector(f"{PRE}-y", y_ops, INICIAL["y"], "Eje y", buscar=True),
                            ui.selector(f"{PRE}-color", _opciones_color(), INICIAL["color"], "Color", buscar=True)], "3",
                           id=f"{PRE}-controles-bivariado")
    return ui.seccion(
        f"Relaciones entre variables: sin los productos activos, ningún predictor pasa de {RHO_ABS} = {ui.num(abs(rho), 3)} con el puntaje",
        [controles, ui.grafico(f"{PRE}-bi", alto="alto", fuentes=[F_ENT, F_VARS, F_ASOC])],
        subtitulo=f"El mayor es {_en_frase(v)} ({ui.con_signo(rho, 3)}). Elija la variable del eje x, la del eje y y la del color. Con "
                  f"una x continua se dibuja una muestra fija de clientes y la mediana por tramos; con una x categórica o de pocos "
                  f"valores, cajas. El color agrupa en a lo más {PALABRA[max(RAMPA)]} niveles con orden (la etiqueta, las cohortes, una "
                  "ordinal o los tercios de una numérica): las categóricas sin orden van en el eje x.",
        fuentes=[F_ENT, F_ASOC], id=f"{PRE}-sec-bivariado")


# --- 6. Correlaciones de Spearman -----------------------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _spearman(f):
    """Matriz de Spearman de las numéricas, el puntaje y los productos activos (sin las que no varían) y la η² del puntaje
    con cada categórica y booleana, con los clientes filtrados."""
    sub = _sub(f)
    cols = [c for c in _numericas() + [OBJ, AP] if sub[c].nunique() > 1]
    R = sub[cols].corr(method="spearman")
    t = _etl()["tipos"]
    eta = pd.Series({v: _eta2(sub[OBJ], sub[v].astype(str).to_numpy()) for v in t["categoricas"] + t["booleanas"]})
    return R, eta


def _pares(R, variables):
    """Los pares de variables distintas, de mayor a menor |ρ|: [(a, b, ρ)]."""
    out = [(a, b, float(R.at[a, b])) for i, a in enumerate(variables) for b in variables[i + 1:]]
    return sorted(out, key=lambda p: -abs(p[2]))


def _correlacion(f, cuantas):
    sub = _sub(f)
    n = len(sub)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "calcular las correlaciones")
    R, eta = _spearman(f)
    if OBJ not in R.columns:
        return _vacio(f, n, "calcular las correlaciones con un puntaje que varíe")
    predictores = [c for c in R.columns if c not in (OBJ, AP)]
    con_obj = R[OBJ].reindex(predictores)
    orden = con_obj.abs().sort_values(ascending=False).index.tolist()
    extra = [c for c in (OBJ, AP) if c in R.columns]
    vs = extra + (orden if cuantas == "todas" else orden[:N_MATRIZ - len(extra)])
    Rm = R.loc[vs, vs]
    k = len(vs)
    nombres = [f"{i + 1}. {_nombre(v)}" for i, v in enumerate(vs)]
    hover = [[f"{_nombre(a)} y {_en_frase(b)}: ρ = <b>{ui.con_signo(Rm.at[a, b], 3)}</b>" for b in vs] for a in vs]
    z = Rm.abs().to_numpy()
    textos = [[ui.num(Rm.at[a, b], 2) for b in vs] for a in vs]
    fig = go.Figure()
    # Dos trazas que se reparten las celdas: las claras con el texto en tinta y las oscuras con el texto en blanco (el color
    # de la letra de un mapa de calor es uno por traza; una anotación por celda tarda segundos)
    for oscuras, rol in ((False, "tinta"), (True, "tinta_inversa")):
        parte = np.where((z >= OSCURO) == oscuras, z, np.nan)
        texto = [[t if ((z[i, j] >= OSCURO) == oscuras and cuantas != "todas") else "" for j, t in enumerate(fila)] for i, fila in enumerate(textos)]
        fig.add_trace(go.Heatmap(z=parte, x=list(range(k)), y=list(range(k)), zmin=0, zmax=1, showscale=False, hoverongaps=False,
                                 colorscale=[[0, ui.ROL["superficie"]], [1, ui.ROL["real"]]], xgap=2, ygap=2, customdata=hover,
                                 hovertemplate="%{customdata}<extra></extra>", text=texto, texttemplate="%{text}",
                                 textfont=ui.fuente_svg(rol, "--t-xs")))
    # Los dos ejes llevan el número de cada variable y los nombres van encima, en la leyenda: con los nombres en el eje, a
    # 400 px no quedaba sitio para las celdas
    numeros = [str(i + 1) for i in range(k)]
    salto = 1 if k <= 2 * N_MATRIZ else 2          # con todas las numéricas, un número de cada dos (no caben a 420 px de alto)
    marcas = dict(tickvals=list(range(0, k, salto)), ticktext=numeros[::salto], showgrid=False)
    fig.update_xaxes(**marcas, showline=False, range=[-0.5, k - 0.5], side="top")
    fig.update_yaxes(**marcas, range=[k - 0.5, -0.5])
    fig.update_layout(hovermode="closest")
    leyenda = html.P("Variables: " + " · ".join(nombres) + ".", className="nota")

    top = orden[0]
    pares = _pares(R, predictores)
    duplicados = [p for p in pares if abs(p[2]) >= DUPLICADO]
    ap = f"los productos activos llegan a ρ = {ui.con_signo(R.at[OBJ, AP], 3)} y, " if AP in R.columns else ""
    titulo = f"Con el puntaje, {ap}entre los predictores, a lo más {ui.con_signo(con_obj[top], 3)} ({_en_frase(top)})"
    titulo = titulo[:1].upper() + titulo[1:]
    frases = []
    if duplicados:
        a, b, r = duplicados[0]
        frases.append(f"{PALABRA.get(len(duplicados), F.entero(len(duplicados)))} pares de predictores tienen {RHO_ABS} ≥ "
                      f"{ui.num(DUPLICADO, 2)}: repiten la misma información (por ejemplo, {_en_frase(a)} y {_en_frase(b)}, ρ = "
                      f"{ui.con_signo(r, 3)}) y cada modelo se la reparte con su regularización.")
    elif pares:
        a, b, r = pares[0]
        frases.append(f"El par de predictores más asociado es {_en_frase(a)} y {_en_frase(b)} (ρ = {ui.con_signo(r, 3)}).")
    eta = eta.dropna()
    if len(eta):
        v = eta.idxmax()
        frases.append(f"Entre las categóricas y booleanas, la mayor η² con el puntaje es la de {_en_frase(v)} ({ui.num(eta[v], 4)}): "
                      f"{'casi nada' if eta[v] < FUERZA[0][0] ** 2 else 'una asociación ' + _fuerza(math.sqrt(eta[v]))}.")
    nota = _lectura(" ".join(frases) + " Son asociaciones, no causas.")
    que = (f"las {PALABRA.get(k, F.entero(k))} variables más asociadas con el puntaje (él incluido)" if cuantas != "todas"
           else f"las {F.entero(k)} variables numéricas")
    sub_ = (f"ρ de Spearman entre {que}, en {_quienes(f, n)}. Más oscuro, asociación más fuerte, positiva o negativa; el signo va "
            "en la celda" + (" (con todas, al pasar el puntero)" if cuantas == "todas" else "") + ". Cada número es una variable de la "
            "lista.")
    filas = [[_nombre(a), _nombre(b), ui.con_signo(r, 3)] for a, b, r in _pares(R, list(R.columns))[:N_PARES]]
    tabla = [ui.tabla(["Variable", "Variable", "ρ"], filas, compacta=True, num=[False, False, True]),
             ui.tabla(["Predictor", "ρ con el puntaje"], [[_nombre(v), ui.con_signo(con_obj[v], 3)] for v in orden], compacta=True, alta=True)]
    etiqueta = f"{titulo}. " + " ".join(frases)
    return {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": leyenda, "tabla": tabla, "nota": nota, "etiqueta": etiqueta}


@lru_cache(maxsize=1)
def _duplicados_todo():
    f = _filtro(None, None, None, None, None)
    R, _eta = _spearman(f)
    pred = [c for c in R.columns if c not in (OBJ, AP)]
    return len([p for p in _pares(R, pred) if abs(p[2]) >= DUPLICADO])


def _seccion_correlacion():
    k = _duplicados_todo()
    control = _campo("Variables", ui.segmentado(f"{PRE}-matriz", [
        {"label": f"Las {PALABRA[N_MATRIZ]} más asociadas", "value": "corta"},
        {"label": "Todas las numéricas", "value": "todas"}], INICIAL["matriz"], pequeno=True))
    return ui.seccion(
        f"Correlaciones: {PALABRA.get(k, F.entero(k))} pares de predictores con {RHO_ABS} ≥ {ui.num(DUPLICADO, 2)} repiten la misma información",
        [ui.grafico(f"{PRE}-corr", alto="alto", fuentes=[F_ENT, F_ASOC])],
        subtitulo=f"Spearman mide asociaciones monótonas, aunque no sean lineales, y no se altera con las colas largas. Los "
                  f"{len(_numericas())} predictores numéricos, el puntaje y los productos activos; las categóricas y booleanas se "
                  "resumen con su η² con el puntaje.",
        controles=[control], fuentes=[F_ENT, F_ASOC], id=f"{PRE}-sec-correlacion")


# --- 7. Faltantes y atípicos ----------------------------------------------------------------------------------------------
def _faltantes(f):
    sub = _sub(f)
    n, activo = len(sub), _activo(f)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "contar faltantes")
    tabla_f = D.tabla("eda/faltantes.csv")
    cols = list(_faltan())
    frac = {c: float(sub[f"falta__{c}"].mean()) for c in cols}
    cols = sorted(cols, key=lambda c: -frac[c])
    en_modelo = dict(zip(tabla_f["columna"], tabla_f["en_modelo"].astype(bool)))
    todo = dict(zip(tabla_f["columna"], tabla_f["fraccion"].astype(float)))
    rotulos = list(cols)
    fig = go.Figure()
    for conserva, forma in ((True, "barra"), (False, "rayada")):
        sel = [i for i, c in enumerate(cols) if bool(en_modelo.get(c)) == conserva]
        if sel:
            fig.add_trace(go.Bar(y=[rotulos[i] for i in sel], x=[frac[cols[i]] for i in sel], orientation="h",
                                 **ui.traza("real", forma, ancho=ui.ancho_barra(len(cols), 240)),
                                 **ui.hover_texto([f"{cols[i]}: <b>{F.pct(frac[cols[i]])}</b> de los clientes sin el dato" for i in sel])))
    if activo:
        fig.add_trace(go.Scatter(y=rotulos, x=[todo[c] for c in cols], **ui.traza("contexto", "punto"),
                                 **ui.hover_texto([f"Todo el entrenamiento: <b>{F.pct(todo[c])}</b>" for c in cols])))
    fig.update_yaxes(categoryorder="array", categoryarray=rotulos[::-1], showgrid=False, tickfont=ui.fuente_svg("tinta_2", "--t-s"))
    _eje_fraccion(fig, max(max(frac.values()), max(todo[c] for c in cols) if activo else 0), eje="x", marcas=2)
    fig.update_xaxes(showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_layout(hovermode="closest", barmode="overlay")

    sin = int((tabla_f["faltantes"] == 0).sum())
    total = len(tabla_f)
    cupo = "credit_utilization_ratio"
    frases = []
    if cupo in cols:
        sin_tarjeta = ~sub["credit_card"].astype(bool)
        a = float(sub.loc[sin_tarjeta, f"falta__{cupo}"].mean()) if sin_tarjeta.any() else float("nan")
        b = float(sub.loc[~sin_tarjeta, f"falta__{cupo}"].mean()) if (~sin_tarjeta).any() else float("nan")
        frases.append(f"{cupo} ({_en_frase(cupo)}) falta en el {F.pct(a)} de los clientes sin tarjeta de crédito y en el "
                      f"{F.pct(b)} de los que la tienen: no es un dato perdido sino un «no aplica», y el ETL lo imputa con 0.")
    texto_libre = [c for c in cols if not en_modelo.get(c)]
    for c in texto_libre[:1]:
        m = sub[f"falta__{c}"].astype(bool)
        if m.any() and (~m).any():
            frases.append(f"En {c}, texto libre, el riesgo alto es del {F.pct(sub.loc[m, ETQ].mean())} entre quienes no lo tienen y del "
                          f"{F.pct(sub.loc[~m, ETQ].mean())} entre quienes sí: sin señal, como {F.enumerar(texto_libre)}, que se eliminaron.")
    titulo = f"Solo {PALABRA.get(total - sin, F.entero(total - sin))} de las {F.entero(total)} columnas originales tienen faltantes"
    if cupo in cols:
        exacto = a == 1 and b == 0
        titulo += (f"; {cupo}, la única que se conserva, falta en el {F.pct(frac[cupo])}: "
                   + ("justo los clientes sin tarjeta" if exacto else "sobre todo en los clientes sin tarjeta"))
    sub_ = (f"Fracción de {_quienes(f, n)} sin el dato en el archivo original, para las columnas que tienen alguno; las otras "
            f"{F.entero(sin)} están completas." + (" Los puntos gris claro son todo el entrenamiento." if activo else ""))
    nota = _lectura(" ".join(frases))
    leyenda = [("Se conserva (imputada)", "real", "barra"), ("Se eliminó en el ETL", "real", "rayada")] + (
        [("Todo el entrenamiento", "contexto", "punto")] if activo else [])
    filas = [[c, F.pct(frac[c]), F.pct(todo[c]), "sí" if en_modelo.get(c) else "no"] for c in cols]
    tabla = [ui.tabla(["Columna", "Faltan (filtrados)", "Faltan (todo el entrenamiento)", "En el modelo"], filas, compacta=True,
                      num=[False, True, True, False]),
             ui.nota(f"Las otras {F.entero(sin)} columnas originales no tienen faltantes (eda/faltantes.csv).")]
    etiqueta = f"{titulo}. " + " ".join(frases)
    return {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": ui.leyenda(leyenda), "tabla": tabla, "nota": nota, "etiqueta": etiqueta}


def _tukey(sub):
    """Por cada predictor numérico: fracción y número de clientes fuera de las cercas de Tukey, las cercas y si su RIC es 0."""
    cols = _numericas()
    X = sub[cols].to_numpy(dtype=float)
    q1, q3 = np.quantile(X, [0.25, 0.75], axis=0)
    ric = q3 - q1
    lo, hi = q1 - TUKEY * ric, q3 + TUKEY * ric
    fuera = (X < lo) | (X > hi)
    return pd.DataFrame({"frac": fuera.mean(axis=0), "n": fuera.sum(axis=0), "lo": lo, "hi": hi, "ric_cero": ric == 0}, index=cols)


def _atipicos(f):
    sub = _sub(f)
    n = len(sub)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "contar atípicos")
    t = _tukey(sub).sort_values("frac", ascending=False)
    con = t[t["frac"] > 0]
    if con.empty:
        return {"figura": ui.figura_vacia("Ningún atípico con estos filtros"), "titulo": "Ningún predictor numérico tiene atípicos con estos filtros",
                "sub": f"Regla de Tukey en {_quienes(f, n)}.", "leyenda": None, "tabla": ui.nota("Sin atípicos."),
                "nota": _lectura("sin atípicos que tratar."), "etiqueta": "Sin atípicos."}
    dib = con.iloc[:N_ATIPICOS]
    cats = [_nombre(v) for v in dib.index]
    fig = go.Figure(go.Bar(y=cats, x=dib["frac"], orientation="h", **ui.barras_foco("real", cats, foco=cats[0], ancho=ui.ancho_barra(len(cats), 420)),
                           **ui.hover_texto([f"{c}: <b>{F.pct(p)}</b> fuera de las cercas ({F.entero(k)} clientes)" for c, p, k in zip(cats, dib["frac"], dib["n"])])))
    fig.update_yaxes(categoryorder="array", categoryarray=cats[::-1], showgrid=False, tickfont=ui.fuente_svg("tinta_2", "--t-xs"))
    _eje_fraccion(fig, float(dib["frac"].max()), eje="x", marcas=3)
    fig.update_xaxes(showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_layout(hovermode="closest")
    total = len(t)
    v0 = dib.index[0]
    titulo = (f"{PALABRA.get(len(con), F.entero(len(con))).capitalize()} de los {F.entero(total)} predictores numéricos tienen atípicos "
              f"por Tukey; {_en_frase(v0)} llega al {F.pct(dib['frac'].iloc[0])}")
    ceros = [v for v in con.index if t.at[v, "ric_cero"]]
    # Los clientes atípicos en alguna variable derivada de las transacciones: cuántos son y si concentran el riesgo alto
    derivadas = [v for v in _etl()["derivadas"] if v in t.index]
    X = sub[derivadas].to_numpy(dtype=float)
    atipico = ((X < t.loc[derivadas, "lo"].to_numpy()) | (X > t.loc[derivadas, "hi"].to_numpy())).any(axis=1)
    frases = []
    if atipico.any() and (~atipico).any():
        etq = sub[ETQ].to_numpy()
        frases.append(f"el {F.pct(atipico.mean())} de los clientes es atípico en al menos una de las {F.entero(len(derivadas))} variables "
                      f"derivadas de las transacciones, y entre ellos el riesgo alto es del {F.pct(etq[atipico].mean())}, frente al "
                      f"{F.pct(etq[~atipico].mean())} del resto.")
    frases.append(("No" if frases else "no") + " se recortan: en las variables de conteo y de montos son clientes muy activos, no errores, y "
                  "los modelos de árboles no cambian con transformaciones monótonas.")
    if ceros:
        frases.append(f"Con rango intercuartílico 0 ({F.enumerar(_en_frase(v) for v in ceros)}), todo valor distinto del más común cuenta "
                      "como atípico.")
    sub_ = (f"Fracción de {_quienes(f, n)} fuera de las cercas de Tukey ({_pc(P25)} − {ui.num(TUKEY, 1)} × RIC y {_pc(P75)} + "
            f"{ui.num(TUKEY, 1)} × RIC) en cada predictor numérico; las {PALABRA.get(len(dib), F.entero(len(dib)))} con más atípicos "
            "(todas, en la tabla).")
    filas = [[_nombre(v), F.pct(r["frac"]), F.entero(r["n"]), _val(v, r["lo"]), _val(v, r["hi"])] for v, r in t.iterrows()]
    tabla = ui.tabla(["Predictor", "Atípicos", "Clientes", "Cerca inferior", "Cerca superior"], filas, compacta=True, alta=True,
                     ocultar_movil=(3, 4))
    etiqueta = f"{titulo}. " + "; ".join(f"{c}: {F.pct(p)}" for c, p in zip(cats, dib["frac"])) + "."
    return {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": None, "tabla": tabla, "nota": _lectura(" ".join(frases)), "etiqueta": etiqueta}


def _caja(f, v):
    sub = _sub(f)
    n = len(sub)
    if n < MIN_CLIENTES:
        return _vacio(f, n, "dibujar las cajas")
    nombre = _nombre(v)
    x = sub[v].to_numpy(dtype=float)
    _q1, _med, _q3, _bi, _bs, cerca_i, cerca_s = _cuartiles(x)
    fuera = (x < cerca_i) | (x > cerca_s)
    grupos = [("Riesgo alto", sub[ETQ].to_numpy() == 1), ("Resto", sub[ETQ].to_numpy() == 0)]
    grupos = [(g, m) for g, m in grupos if m.any()]
    cajas = [(g, _cuartiles(x[m]), int(m.sum()), int((fuera & m).sum())) for g, m in grupos]
    etiquetas = [g for g, *_ in cajas]
    fig = go.Figure(go.Box(x=etiquetas, q1=[c[0] for _g, c, *_ in cajas], median=[c[1] for _g, c, *_ in cajas],
                           q3=[c[2] for _g, c, *_ in cajas], lowerfence=[c[3] for _g, c, *_ in cajas],
                           upperfence=[c[4] for _g, c, *_ in cajas], boxpoints=False, hoverinfo="skip", width=0.4,
                           line=dict(color=ui.ROL["real"], width=ui.ANCHO_LINEA / 2 + 0.5), fillcolor=ui.alfa(ui.ROL["real"], ui.LAVADO)))
    textos = [f"{g}: mediana <b>{_val(v, c[1])}</b>{_unidad(v)} · caja de {_val(v, c[0])} a {_val(v, c[2])} · bigotes de "
              f"{_val(v, c[3])} a {_val(v, c[4])} · {F.entero(k)} clientes, {F.entero(a)} atípicos" for g, c, k, a in cajas]
    fig.add_trace(go.Scatter(x=etiquetas, y=[c[1] for _g, c, *_ in cajas], mode="markers",
                             marker=dict(size=2 * ui.MARCADOR, opacity=0, color=ui.ROL["real"]), **ui.hover_texto(textos)))
    lo, hi = min(c[3] for _g, c, *_ in cajas), max(c[4] for _g, c, *_ in cajas)
    margen = (hi - lo) * 0.06 or 1
    vals, marcas = _marcas(lo - margen, hi + margen, 4)
    fig.update_yaxes(range=[lo - margen, hi + margen], tickvals=vals, ticktext=marcas)
    fig.update_xaxes(type="category")
    fig.update_layout(hovermode="closest")
    if _unidad(v):
        ui.unidad_eje(fig, _unidad(v).strip())
    titulo = (f"{nombre}: el {F.pct(fuera.mean())} queda fuera de las cercas de Tukey ({_val(v, cerca_i)} y {_val(v, cerca_s)}"
              f"{_unidad(v)})")
    if len(cajas) == 2:
        (g1, c1, _k1, a1), (g2, c2, _k2, a2) = cajas
        comp = ("mayor en riesgo alto" if c1[1] > c2[1] else "menor en riesgo alto" if c1[1] < c2[1]
                else "la misma en los dos grupos")
        lectura = (f"la mediana es {_val(v, c1[1])} en riesgo alto y {_val(v, c2[1])} en el resto ({comp}). Los "
                   f"{F.entero(int(fuera.sum()))} atípicos no se dibujan (llegan hasta {_val(v, x.max())}{_unidad(v)}) y no se recortan.")
    else:
        lectura = f"con estos filtros solo hay clientes de una clase. Los {F.entero(int(fuera.sum()))} atípicos no se dibujan."
    sub_ = (f"Diagramas de caja de {_en_frase(v)} en {_quienes(f, n)}, separados por la etiqueta: la caja va del {_pc(P25)} al {_pc(P75)}, "
            "la raya es la mediana y los bigotes llegan al último valor dentro de las cercas de cada grupo. Las cercas del título son "
            "las de todos los clientes filtrados.")
    filas = [[g, F.entero(k), _val(v, c[0]), _val(v, c[1]), _val(v, c[2]), _val(v, c[3]), _val(v, c[4]), F.entero(a)] for g, c, k, a in cajas]
    tabla = ui.tabla(["Grupo", "Clientes", "P25", "Mediana", "P75", "Bigote inferior", "Bigote superior", "Atípicos"], filas, compacta=True,
                     ocultar_movil=(5, 6))
    return {"figura": fig, "titulo": titulo, "sub": sub_, "leyenda": None, "tabla": tabla, "nota": _lectura(lectura),
            "etiqueta": f"{titulo}. {lectura}"}


def _seccion_faltantes_atipicos():
    tabla_f = D.tabla("eda/faltantes.csv")
    con = int((tabla_f["faltantes"] > 0).sum())
    selector = ui.selector(f"{PRE}-caja", _opciones([v for v in _predictores() if _tipo(v) == "numerica"]), INICIAL["caja"], "Variable de las cajas",
                           buscar=True)
    return ui.seccion(
        f"Faltantes y atípicos: {PALABRA.get(con, F.entero(con))} de las {F.entero(len(tabla_f))} columnas tienen faltantes y ningún "
        "atípico se recorta",
        [ui.rejilla([ui.grafico(f"{PRE}-faltantes", alto="bajo", fuentes=[F_FALT, F_ENT]),
                     ui.grafico(f"{PRE}-atipicos", alto="alto", fuentes=[F_ENT])], "1-1"),
         ui.fila_controles([selector]),
         ui.grafico(f"{PRE}-cajas", fuentes=[F_ENT])],
        subtitulo="Los faltantes se cuentan en las columnas originales, antes del ETL (con los filtros, con las indicadoras falta__ del "
                  f"entrenamiento). Los atípicos, con la regla de Tukey: fuera de {ui.num(TUKEY, 1)} rangos intercuartílicos de la caja "
                  "central.",
        fuentes=[F_FALT, F_ENT], id=f"{PRE}-sec-faltantes")


# --- 8. Transformaciones y control de la fuga de información ----------------------------------------------------------------
def _estado(bien, texto_bien="comprobado aquí"):
    return ui.chip(texto_bien, "bien") if bien else ui.chip("no se cumple", "critico")


def _auditoria():
    meta, etl, T = M.meta(), _etl(), _T()
    prueba = M.clientes().index                                   # solo los índices de la prueba, para comprobar el cruce
    solape = len(set(T["fila"]) & set(prueba))
    tau_aqui = float(T[OBJ].quantile(float(meta["etiqueta"]["q"])))
    tau = _tau()
    eliminadas = set(etl["eliminadas"])
    cruce = eliminadas & set(M.predictores())
    r_ap = float(D.fila(D.tabla("eda/asociacion.csv"), variable=AP)["pearson"])
    cupo = "credit_utilization_ratio"
    falta = T[f"falta__{cupo}"].astype(bool) if cupo in _faltan() else pd.Series(False, index=T.index)
    regla = bool((~T.loc[falta, "credit_card"].astype(bool)).all() and (T.loc[falta, cupo] == 0).all()
                 and T.loc[~falta, "credit_card"].astype(bool).all())
    declarado = ui.chip("según la metodología")
    filas = [
        ["Partición antes de todo estadístico del objetivo",
         f"{meta['particion']}: {F.entero(meta['n_entrenamiento'])} de entrenamiento y {F.entero(meta['n_prueba'])} de prueba, que "
         f"suman los {F.entero(etl['n_clientes'])} clientes. Al azar y no por posición, porque el archivo está ordenado por bloques "
         "de cohortes.",
         _estado(meta["n_entrenamiento"] + meta["n_prueba"] == etl["n_clientes"])],
        ["Ningún cliente en los dos lados",
         f"Ninguna de las {F.entero(len(T))} filas de entrenamiento está entre las {F.entero(len(prueba))} de prueba (se cruzan solo "
         f"los índices; hay {F.entero(solape)} en común).",
         _estado(solape == 0)],
        ["τ, solo con el entrenamiento",
         f"El {_percentil()} de {OBJ}, recalculado aquí con el entrenamiento, es {ui.num(tau_aqui, 4)}; el guardado en metadatos.json, "
         f"{ui.num(tau, 4)}. La prueba conserva el balance: metadatos.json guarda en ella un {F.pct(meta['prevalencia_prueba'])} de "
         f"riesgo alto, frente al {F.pct(meta['prevalencia_entrenamiento'])} del entrenamiento.",
         _estado(abs(tau_aqui - tau) < 1e-12)],
        ["Variables que codifican la respuesta, fuera",
         f"{AP} (r = {ui.num(r_ap, 3)} con el puntaje: la escalera), customer_lifetime_value y clv_segment (el segundo objetivo). "
         f"Ninguna de las {F.entero(len(eliminadas))} columnas eliminadas está entre los {F.entero(len(M.predictores()))} predictores.",
         _estado(not cruce and AP not in M.predictores())],
        ["Variables derivadas sin el objetivo",
         f"Las {F.entero(len(etl['derivadas']))} variables derivadas resumen el detalle de transacciones de cada cliente; ninguna usa "
         f"{OBJ}.", declarado],
        ["Imputación con una regla fija",
         f"{cupo} vale 0 en quien no tiene tarjeta: es un «no aplica», no un estadístico del conjunto (ni la media ni la mediana), así "
         "que no lleva información de otros clientes.", _estado(regla)],
        ["Transformaciones dentro de cada pliegue",
         "Las indicadoras, la estandarización y el sobremuestreo se ajustan con las filas de entrenamiento de cada pliegue de la "
         "validación cruzada anidada; la prueba no se toca hasta la evaluación final.", declarado],
        ["Esta página no usa la prueba",
         f"Todos sus gráficos, tablas y frases salen de {F_ENT}; de la prueba solo se leen sus índices (esta tabla) y dos cifras de "
         "metadatos.json, y la tarjeta de la trampa cita dos resultados de modelos.", _estado(True, "por construcción")],
    ]
    return ui.tabla(["Control", "Cómo se comprueba", "Estado"], filas, compacta=True, num=[False, False, False])


def _transformaciones():
    etl, meta = _etl(), M.meta()
    t = etl["tipos"]
    cupo = "credit_utilization_ratio"
    falta = float(D.fila(D.tabla("eda/faltantes.csv"), columna=cupo)["fraccion"])
    balanceos = F.enumerar(D.NOMBRE_BALANCEO[b] for b in D.ORDEN_BALANCEOS if b != "ninguno")
    filas = [
        ["Variables derivadas", f"{F.entero(len(etl['derivadas']))} por cliente a partir del detalle de transacciones: número, monto total, "
                                "medio y mediano, recencia, días y meses activos, volatilidad y proporción de cada tipo.", "Una vez, sin el objetivo"],
        ["Columnas eliminadas", f"{F.entero(len(etl['eliminadas']))}: el identificador, la que generó el puntaje, el segundo objetivo, "
                                "resúmenes duplicados, fechas y texto sin señal.", "Revalidadas con el entrenamiento"],
        ["Imputación", f"{cupo} = 0 sin tarjeta de crédito (falta en el {F.pct(falta)} del entrenamiento): un «no aplica».", "Regla fija"],
        ["Codificación", f"Indicadoras para las {F.entero(len(t['categoricas']))} categóricas, descartando un nivel: "
                         f"{F.entero(len(etl['predictores']))} predictores pasan a {F.entero(etl['n_columnas_modelo'])} columnas.",
         "Dentro de cada pliegue"],
        ["Estandarización", "Las numéricas y las booleanas.", "Dentro de cada pliegue"],
        ["Balanceo", f"No balancear, {balanceos}, según la combinación de cada modelo.", "Dentro de cada pliegue, solo con sus filas"],
        ["Atípicos", "No se recortan: son clientes muy activos, no errores, y los árboles son invariantes a transformaciones monótonas.",
         "No se transforman"],
        ["Etiqueta", f"Riesgo alto si {OBJ} > τ = {ui.num(_tau(), 4)} (el {_percentil()} del entrenamiento).", "Con el entrenamiento"],
    ]
    return ui.tabla(["Paso", "Qué se hace", "Dónde se ajusta"], filas, compacta=True, num=[False, False, False])


def _seccion_transformaciones():
    etl = _etl()
    tabla = M.variables()
    eliminadas = ui.tabla(["Columna eliminada", "Razón, según el informe"], [[c, RAZON.get(c, F.RAYA)] for c in etl["eliminadas"]],
                          compacta=True, alta=True, num=[False, False])
    derivadas = ui.tabla(["Variable derivada", "Nombre en la app", "Qué es"],
                         [[c, M.nombre(c), str(tabla.at[c, "descripcion"]) if c in tabla.index else F.RAYA] for c in etl["derivadas"]],
                         compacta=True, alta=True, num=[False, False, False], ocultar_movil=(1,))
    return ui.seccion(
        f"Transformaciones y control de la fuga de información: {F.entero(len(etl['eliminadas']))} columnas fuera, "
        f"{F.entero(len(etl['derivadas']))} derivadas y la partición antes que nada",
        [ui.tarjeta(_auditoria(), titulo="Auditoría de la fuga de información", fuentes=[F_META, F_ETL, F_ENT, F_ASOC, F_CLIENTES],
                    id=f"{PRE}-auditoria"),
         ui.tarjeta(_transformaciones(), titulo="Cada transformación y dónde se ajusta", fuentes=[F_ETL, F_FALT, F_META],
                    id=f"{PRE}-transformaciones"),
         ui.rejilla([ui.tarjeta(eliminadas, titulo=f"Las {F.entero(len(etl['eliminadas']))} columnas eliminadas y por qué",
                                fuentes=[F_ETL], id=f"{PRE}-eliminadas"),
                     ui.tarjeta(derivadas, titulo=f"Las {F.entero(len(etl['derivadas']))} variables derivadas de las transacciones",
                                fuentes=[F_ETL, F_VARS], id=f"{PRE}-derivadas")], "1-1")],
        subtitulo="Lo que se hizo con los datos antes de modelar, por qué, y cómo se evitó que la respuesta o la prueba se colaran en "
                  "el entrenamiento. Lo marcado «comprobado aquí» se calcula al abrir la página.",
        fuentes=[F_ETL, F_META, F_ENT], id=f"{PRE}-sec-fuga")


# --- Filtros globales ---------------------------------------------------------------------------------------------------
def _deslizador_rango(id, rotulo, minimo, maximo, paso, valor, marcas, texto):
    """ui.deslizador con dcc.RangeSlider: rótulo y valor en una línea («de 4 a 100», en «<id>-valor») → pista con dos
    extremos → marcas. El rótulo también tiene id («<id>-rotulo»): cambia con la variable."""
    return html.Div([
        html.Div([html.Label(rotulo, htmlFor=id, className="deslizador__rotulo", id=f"{id}-rotulo"),
                  html.Span(texto, id=f"{id}-valor", className="deslizador__valor")], className="deslizador__cab"),
        dcc.RangeSlider(id=id, min=minimo, max=maximo, step=paso, value=valor, marks=marcas, updatemode="mouseup", allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": False}, allow_direct_input=False, className="deslizador"),
    ], className="deslizador-bloque")


def _tarjeta_filtros():
    var = RANGO_VARIABLES[0]
    lo, hi, paso = _limites(var)
    etl = _etl()
    fechas = [c for c in FECHAS if c in etl["eliminadas"]]
    cohortes = [{"label": "Todas las edades", "value": TODAS}] + [{"label": f"{c} años", "value": c} for c, _a, _b in _cohortes()]
    segmentos = [{"label": "Todos los segmentos", "value": TODAS}] + [{"label": s, "value": s} for s in _niveles("customer_segment")]
    canales = [{"label": "Todos los canales", "value": TODAS}] + [{"label": c, "value": c} for c in _niveles("acquisition_channel")]
    rango = html.Div([
        _campo("Rango de", ui.segmentado(f"{PRE}-rango-var", [{"label": CORTO[v], "value": v} for v in RANGO_VARIABLES], var, pequeno=True,
                                         persistir=False)),
        _deslizador_rango(f"{PRE}-rango", _nombre(var), lo, hi, paso, [lo, hi], _marcas_rango(var), _texto_rango(var, lo, hi)),
    ], className="panel__grupo")
    return ui.tarjeta([
        html.Div([html.P("Filtros globales", className="panel__titulo"),
                  ui.boton("Quitar los filtros", id=f"{PRE}-limpiar", icono="cerrar", pequeno=True)], className="panel__cab"),
        ui.rejilla([ui.selector(f"{PRE}-cohorte", cohortes, TODAS, "Cohorte de edad", persistir=False),
                    ui.selector(f"{PRE}-segmento", segmentos, TODAS, f"{_nombre('customer_segment')}", persistir=False),
                    ui.selector(f"{PRE}-canal", canales, TODAS, f"{_nombre('acquisition_channel')}", persistir=False)], "3"),
        ui.rejilla([rango, html.Div(_contador(_filtro(None, None, None, None, None)), id=f"{PRE}-contador")], "1-1"),
        ui.nota(f"Cambian todos los gráficos, tablas y frases de la página. Solo los {F.entero(len(_T()))} clientes de entrenamiento. No hay selector de fechas: las "
                f"{PALABRA.get(len(fechas), F.entero(len(fechas)))} columnas de fecha ({F.enumerar(fechas)}) se eliminaron en el ETL y "
                "ningún predictor es una fecha. Los segmentos y canales van como vienen en COFINFAD."),
    ], id=f"{PRE}-filtros")


def _subtitulo():
    meta = M.meta()
    rho, v = _maximo_rho()
    return (f"Los {F.entero(meta['n_entrenamiento'])} clientes de entrenamiento ({F.pct(1 - meta['fraccion_prueba'], 0)} de COFINFAD), "
            f"con {len(M.predictores())} predictores. El {F.pct(meta['prevalencia_entrenamiento'])} es de riesgo alto; sin los "
            f"productos activos, que generaron el puntaje, ningún predictor pasa de {RHO_ABS} = {ui.num(abs(rho), 3)} con él "
            f"({_en_frase(v)}).")


def layout(**_):
    return ui.pantalla([
        ui.encabezado(PREGUNTA, _subtitulo(), pantalla=PANTALLA, respuesta=[_tarjeta_filtros()]),
        _seccion_datos(),
        _seccion_objetivo(),
        _seccion_trampa(),
        _seccion_univariadas(),
        _seccion_bivariado(),
        _seccion_correlacion(),
        _seccion_faltantes_atipicos(),
        _seccion_transformaciones(),
    ])


# --- Callbacks ------------------------------------------------------------------------------------------------------------
@callback(Output(f"{PRE}-rango", "min"), Output(f"{PRE}-rango", "max"), Output(f"{PRE}-rango", "step"), Output(f"{PRE}-rango", "marks"),
          Output(f"{PRE}-rango", "value"), Output(f"{PRE}-rango-rotulo", "children"),
          Input(f"{PRE}-rango-var", "value"), Input(f"{PRE}-limpiar", "n_clicks"))
def _poner_rango(var, _n):
    """El deslizador de rango con los límites de la variable elegida, entero (también al quitar los filtros)."""
    var = var if var in RANGO_VARIABLES else RANGO_VARIABLES[0]
    lo, hi, paso = _limites(var)
    return lo, hi, paso, _marcas_rango(var), [lo, hi], _nombre(var)


@callback(Output(f"{PRE}-rango-valor", "children"), Input(f"{PRE}-rango", "drag_value"), Input(f"{PRE}-rango", "value"),
          State(f"{PRE}-rango-var", "value"))
def _texto_del_rango(arrastre, valor, var):
    var = var if var in RANGO_VARIABLES else RANGO_VARIABLES[0]
    lo, hi, _paso = _limites(var)
    disparo = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    actual = arrastre if (disparo.endswith("drag_value") and arrastre) else valor
    try:
        a, b = sorted(float(x) for x in actual)
    except (TypeError, ValueError):
        a, b = lo, hi
    return _texto_rango(var, min(max(a, lo), hi), min(max(b, lo), hi))


@callback(Output(f"{PRE}-cohorte", "value"), Output(f"{PRE}-segmento", "value"), Output(f"{PRE}-canal", "value"),
          Input(f"{PRE}-limpiar", "n_clicks"), prevent_initial_call=True)
def _quitar_filtros(n):
    if not n:
        raise PreventUpdate
    return TODAS, TODAS, TODAS


@callback(Output(f"{PRE}-contador", "children"), *FILTROS)
def _contar(*filtros):
    return _contador(_filtro(*filtros))


@callback(Output(f"{PRE}-desc-nota", "children"), Output(f"{PRE}-desc-num", "children"), Output(f"{PRE}-desc-cat", "children"), *FILTROS)
def _dibujar_descriptiva(*filtros):
    try:
        return _descriptiva(_filtro(*filtros))
    except Exception as e:  # noqa: BLE001
        aviso = _fallo(e)["tabla"]
        return _fallo(e)["nota"], aviso, aviso


@callback(*_salidas(f"{PRE}-hist"), *_salidas(f"{PRE}-balance"), *FILTROS)
def _dibujar_objetivo(*filtros):
    f = _filtro(*filtros)
    return _partes(_seguro(_histograma_objetivo, f)) + _partes(_seguro(_balance, f))


@callback(*_salidas(f"{PRE}-escalera"), *FILTROS)
def _dibujar_escalera(*filtros):
    return _partes(_seguro(_escalera, _filtro(*filtros)))


@callback(*_salidas(f"{PRE}-uni-graf"), Output(f"{PRE}-uni-ficha", "children"), Output(f"{PRE}-uni-graf-lienzo", "className"),
          *FILTROS, Input(f"{PRE}-uni", "value"))
def _dibujar_univariada(*entradas):
    *filtros, v = entradas
    v = v if v in M.predictores() else INICIAL["uni"]
    try:
        g, ficha, clase = _univariada(_filtro(*filtros), v)
    except Exception as e:  # noqa: BLE001
        g, ficha, clase = _fallo(e), None, "grafico__lienzo"
    return _partes(g) + [ficha, clase]


@callback(*_salidas(f"{PRE}-bi"), *FILTROS, Input(f"{PRE}-x", "value"), Input(f"{PRE}-y", "value"), Input(f"{PRE}-color", "value"))
def _dibujar_bivariado(*entradas):
    *filtros, vx, vy, color = entradas
    vx = vx if vx in list(M.predictores()) + [AP] else INICIAL["x"]
    vy = vy if vy in [OBJ, AP] + [p for p in M.predictores() if _tipo(p) == "numerica"] else INICIAL["y"]
    color = color if color in _valores_color() else INICIAL["color"]
    return _partes(_seguro(_bivariado, _filtro(*filtros), vx, vy, color))


@callback(*_salidas(f"{PRE}-corr"), *FILTROS, Input(f"{PRE}-matriz", "value"))
def _dibujar_correlacion(*entradas):
    *filtros, cuantas = entradas
    return _partes(_seguro(_correlacion, _filtro(*filtros), cuantas if cuantas in ("corta", "todas") else INICIAL["matriz"]))


@callback(*_salidas(f"{PRE}-faltantes"), *_salidas(f"{PRE}-atipicos"), *FILTROS)
def _dibujar_faltantes_atipicos(*filtros):
    f = _filtro(*filtros)
    return _partes(_seguro(_faltantes, f)) + _partes(_seguro(_atipicos, f))


@callback(*_salidas(f"{PRE}-cajas"), *FILTROS, Input(f"{PRE}-caja", "value"))
def _dibujar_cajas(*entradas):
    *filtros, v = entradas
    v = v if v in _numericas() else INICIAL["caja"]
    return _partes(_seguro(_caja, _filtro(*filtros), v))
