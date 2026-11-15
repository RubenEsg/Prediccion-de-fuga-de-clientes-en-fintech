"""Interpretabilidad · ¿Qué variables usa cada modelo?

La interpretabilidad global del modelo elegido con el resumen SHAP precalculado (V.shap_indice() y V.shap_muestra(): la app
no calcula SHAP al cargar esta vista ni necesita la librería shap):

  · la importancia de cada variable (la media del valor absoluto de su aporte en los clientes de la muestra fija), las
    principales y el resto agrupado;
  · el resumen tipo enjambre de las principales: un punto por cliente, en la posición de su aporte y con un degradado de
    un solo tono (del gris claro al casi negro: los tokens «contexto» y «real») según el valor de la variable;
  · qué es un valor SHAP, en qué escala está cada modelo (logit, margen, probabilidad o pronóstico: entre modelos solo se
    comparan orden y signo), cómo se calculó (exacto o estimado) y cómo se verificó (indice.json, «verificacion»);
  · la comparación con la EBM (con XGBoost, el mejor modelo de referencia en la prueba, cuando el elegido es la EBM): el
    puesto de cada variable en los dos, cuántas de las primeras comparten y la correlación de rangos de las importancias.

La dirección puede pedir la tarea y el modelo: /modelos/interpretabilidad?tarea=regresion&modelo=lasso.
"""
from functools import lru_cache
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

import app
import datos_app as D
import formato as F
import modelos_vivos as V
import motor as M
import ui

VISTA = "interpretabilidad"
PRE = "ip"
PANTALLA = "errores"                  # el icono de la cabecera (la lupa, como la vista en app.VISTAS)
TAREAS = ("clasificacion", "regresion")
INICIAL = "ebm"

TOP_BARRAS = 10                        # variables con barra propia; las demás van agrupadas
TOP_ENJAMBRE = 8                       # variables del resumen tipo enjambre
TOP_COMPARTIDAS = 5                    # «cuántas de las cinco primeras comparten»
FILAS_COMPARACION = 8                  # filas de la tabla de la comparación
DIRECCIONES = 3                        # variables cuya dirección se describe bajo el enjambre
CORTES_ENJAMBRE = 60                   # tramos del eje en que se apilan los puntos de una fila
ALTO_FILA = 0.5                        # alto de los puntos de una fila del enjambre, en unidades de fila (encima va su nombre)
MARGEN_FILA = 0.1                      # aire bajo la última fila
PERCENTILES_EJE = (0.5, 99.5)          # con aportes extremos, el eje del enjambre va de este percentil a este otro
RECORTE = 3                            # se recorta si el recorrido total pasa de 3 veces el de esos percentiles
PERCENTILES_COLOR = (5, 95)            # el degradado va del percentil 5 al 95 del valor (lo de fuera, al extremo)
MONOTONA = 0.8                         # |ρ| desde el que el aporte «sube» o «baja» con el valor
TAM_PUNTO = ui.MARCADOR * 0.75         # px
OPACIDAD = 0.85
MEDIO = 0.5                            # tono de las variables categóricas (sin orden): el medio del degradado
HOLGURA = 1.06                         # aire a los lados del eje del enjambre

FUENTE_INDICE = D.fuente("shap/indice.json")
FUENTE_CLIENTES = D.fuente("clientes_prueba.parquet")
FUENTE_TERMINOS = D.fuente("resultados/importancia_terminos_ebm.csv")

# Qué cantidad reparte cada modelo, en llano (las claves son las «escala» de datos/shap/indice.json)
ESCALA = {
    "logit del riesgo alto": "el logit del riesgo alto (el logaritmo de la razón entre riesgo alto y riesgo bajo): no es una "
                             "probabilidad, pero sube y baja con ella",
    "margen del SVM (distancia al hiperplano)": "el margen del SVM lineal, su distancia con signo al hiperplano que separa las "
                                                "dos clases: no es una probabilidad",
    "probabilidad de riesgo alto": "la probabilidad de riesgo alto que da el modelo, sin recalibrar",
    "churn_probability predicha": "el pronóstico del puntaje de propensión (churn_probability), en sus mismas unidades",
}
EJE = {"logit del riesgo alto": "aporte al logit del riesgo alto", "margen del SVM (distancia al hiperplano)": "aporte al margen del SVM",
       "probabilidad de riesgo alto": "aporte a la probabilidad de riesgo alto", "churn_probability predicha": "aporte al pronóstico"}
# Cómo se calculó, por método (explicacion.py y preparar_artefactos.py)
METODO = {
    "lineal": "Exacto. En un modelo lineal el aporte de una variable es su coeficiente por la distancia entre su valor y el "
              "promedio del fondo (Lundberg y Lee, 2017).",
    "bayes": "Exacto. El logit de Naive Bayes es una suma de un término por columna: el aporte de una variable es su término "
             "menos el promedio de ese término en el fondo.",
    "xgboost": "Exacto. TreeSHAP de la propia librería XGBoost, que recorre los árboles del modelo.",
    "arbol": "Exacto. TreeSHAP de la librería shap, calculado fuera de línea al preparar los datos de la app (la explicación "
             "local en vivo de la vista Predicción estima este modelo por permutaciones; este resumen es el exacto).",
    "vecinos": "Estimado. k-NN no tiene una fórmula cerrada: el aporte se estima con permutaciones antitéticas de las variables "
               "(Strumbelj y Kononenko, 2014) contra clientes de entrenamiento del fondo, y se informa su error de Monte Carlo. "
               "Por su costo se explica a menos clientes.",
    "ebm": "Exacto. La EBM es una suma de una función por variable: el aporte de cada variable es el valor de su término, sin "
           "explicador auxiliar.",
}
FRENTE_A = {"frente_a_shap_TreeExplainer": "Frente a TreeExplainer de la librería shap",
            "frente_a_shap_LinearExplainer": "Frente a LinearExplainer de la librería shap",
            "frente_a_shap_PermutationExplainer": "Frente a PermutationExplainer de la librería shap"}
SUPERINDICE = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
# Las partes de ui.grafico que escribe el callback, en este orden
PARTES_GRAFICO = (("", "figure"), ("-titulo", "children"), ("-sub", "children"), ("-leyenda", "children"), ("-tabla", "children"),
                  ("-nota", "children"), ("-lienzo", "aria-label"))


# --- Textos ---------------------------------------------------------------------------------------------------------------
def _mayuscula(texto):
    """Primera letra en mayúscula, salvo en un nombre que empieza en minúscula (k-NN)."""
    return texto if texto.startswith("k-") else texto[:1].upper() + texto[1:]


def _tras(preposicion, frase):
    if frase.startswith("el "):
        return {"a": "al", "de": "del"}[preposicion] + frase[2:]
    return f"{preposicion} {frase}"


def _cientifico(x):
    """Una diferencia minúscula en notación científica es-CO: 4,6 × 10⁻⁷; el cero, «0»."""
    x = float(x)
    if x == 0:
        return "0"
    mantisa, exponente = f"{x:.1e}".split("e")
    return f"{ui.num(float(mantisa), 1)}{F.FIJO}×{F.FIJO}10{str(int(exponente)).translate(SUPERINDICE)}"


def _decimales(valor):
    """Decimales para que un valor del orden de `valor` muestre dos cifras significativas (0,32 · 0,012)."""
    v = abs(float(valor))
    return 2 if v == 0 else max(2, 1 - int(np.floor(np.log10(v))))


def _nombre(variable):
    return M.nombre(variable)


def _tabla(encabezados, filas, alta=False, num=None):
    """La tabla gemela de un gráfico como UN solo componente: el HTML de ui.tabla (mismas clases «tabla-marco», «tabla»,
    «tabla--compacta» y «num» de estilos.css) dentro de un dcc.Markdown. Una tabla de ui.tabla es un componente de Dash
    por celda, y rehacer componentes es lo caro en el navegador (1 a 3 ms cada uno): con las tablas gemelas de esta vista
    hechas así (la de importancia tiene una fila por variable), cambiar de modelo tardaba más de un segundo. Las celdas son textos ya formateados de los datos
    (se escapan); la primera columna es el rótulo de la fila."""
    n = len(encabezados)
    es_num = list(num) if num is not None else [j > 0 for j in range(n)]

    def clase(j):
        return ' class="num"' if es_num[j] else ""
    cab = "".join(f'<th scope="col"{clase(j)}>{escape(str(h))}</th>' for j, h in enumerate(encabezados))
    cuerpo = "".join("<tr>" + f'<th scope="row"{clase(0)}>{escape(str(f[0]))}</th>'
                     + "".join(f"<td{clase(j)}>{escape(str(v))}</td>" for j, v in enumerate(f[1:], 1)) + "</tr>" for f in filas)
    marco = "tabla-marco tabla-marco--alta" if alta else "tabla-marco"
    return dcc.Markdown(f'<div class="{marco}"><table class="tabla tabla--compacta"><thead><tr>{cab}</tr></thead>'
                        f"<tbody>{cuerpo}</tbody></table></div>", dangerously_allow_html=True)


# --- Tarea y modelo (de la dirección, de los controles) ----------------------------------------------------------------
def _tarea(t):
    return t if t in TAREAS else "clasificacion"


def _modelo(tarea, m):
    ms = V.modelos(tarea)
    return m if m in ms else (INICIAL if INICIAL in ms else ms[0])


def _de_la_direccion(tarea, modelo):
    """(tarea, modelo, pedida) a partir de ?tarea= y ?modelo= (un modelo de una sola tarea la elige: Lasso → regresión)."""
    if tarea not in TAREAS and modelo:
        en = [t for t in TAREAS if modelo in V.modelos(t)]
        if len(en) == 1:
            tarea = en[0]
    pedida = tarea in TAREAS or bool(modelo)
    t = _tarea(tarea)
    return t, _modelo(t, modelo), pedida


def _opciones(tarea):
    """Los ocho modelos de la tarea, con su nombre como texto (ver secciones/diagnostico.py::_opciones)."""
    return [{"label": D.NOMBRE_MODELO[m], "value": m} for m in V.modelos(tarea)]


def _rotulo_modelo(modelo):
    """El rótulo del selector con la muestra del color del rol del modelo elegido (azul, rosa o gris): el mismo color con
    que se dibuja en los gráficos."""
    return [ui.muestra(D.ROL_MODELO.get(modelo, "contexto")), html.Span("Modelo:")]


def _selector_modelo(tarea, modelo, persistir):
    """El selector nace con los modelos de las dos tareas (ver secciones/diagnostico.py: si el valor recordado no estuviera
    entre las opciones al recargar, el desplegable lo borraría) y con alto para los ocho."""
    propias = _opciones(tarea)
    valores = {o["value"] for o in propias}
    otras = [o for t in TAREAS if t != tarea for o in _opciones(t) if o["value"] not in valores]
    selector = ui.selector(f"{PRE}-modelo", propias + otras, modelo, persistir=persistir)
    selector.maxHeight = len(propias) * ui.px("--e-7")
    return selector


def _referente(tarea, modelo):
    """Con quién se compara: la EBM; si el elegido es la EBM, el mejor modelo de referencia en la prueba (el primero de la
    tabla del informe que no es la EBM)."""
    if modelo != "ebm":
        return "ebm"
    return next(m for m in D.prueba(tarea)["modelo"] if m != "ebm")


# --- Datos (con caché; no modificar lo que devuelven) -------------------------------------------------------------------
@lru_cache(maxsize=None)
def _resumen(tarea, modelo):
    info = V.shap_indice()["modelos"][f"{tarea}__{modelo}"]
    cols = V.columnas()
    imp = pd.Series(info["importancia_media_abs"], dtype=float).reindex(cols).fillna(0.0)
    imp = imp.iloc[np.argsort(-imp.to_numpy(), kind="mergesort")]
    S = V.shap_muestra(tarea, modelo)
    return {"info": info, "imp": imp, "cuota": imp / imp.sum(), "S": S, "n": len(S), "cols": cols,
            "base": float(S["__base"].mean()), "salida": float(S["__salida"].mean())}


@lru_cache(maxsize=1)
def _tipos():
    return {v["nombre"]: v["tipo"] for v in V.esquema()}


def _tono(valores, tipo):
    """El tono de cada cliente (0 = valor bajo, 1 = alto) del percentil 5 al 95 de la variable; las categóricas, el medio."""
    if tipo == "categorica":
        return np.full(len(valores), MEDIO)
    v = pd.to_numeric(pd.Series(valores), errors="coerce").astype(float).to_numpy()
    lo, hi = np.nanpercentile(v, PERCENTILES_COLOR)
    if hi <= lo:
        lo, hi = np.nanmin(v), np.nanmax(v)
    if hi <= lo:
        return np.full(len(v), MEDIO)
    return np.clip((v - lo) / (hi - lo), 0, 1)


def _apilar(x):
    """Desplazamiento vertical de cada punto de una fila del enjambre: los puntos que caen en el mismo tramo del eje se
    apilan hacia arriba y hacia abajo en orden, para que el ancho de la fila muestre cuántos clientes hay en cada aporte."""
    x = np.asarray(x, dtype=float)
    rango = float(x.max() - x.min())
    if rango == 0:
        return np.zeros(len(x))
    tramo = np.round(CORTES_ENJAMBRE * (x - x.min()) / rango).astype(int)
    capa = np.zeros(len(x))
    for t in np.unique(tramo):
        idx = np.where(tramo == t)[0]
        idx = idx[np.argsort(x[idx], kind="mergesort")]
        k = np.arange(len(idx))
        capa[idx] = np.ceil(k / 2) * np.where(k % 2 == 0, 1, -1)
    tope = max(float(np.abs(capa).max()), 1.0)
    return capa / tope * ALTO_FILA / 2


def _valor_texto(valor, tipo):
    if tipo == "booleana":
        return "sí" if bool(valor) else "no"
    if tipo == "categorica":
        return str(valor)
    if tipo == "entera":
        return ui.num(valor, 0)
    return ui.num(valor, 2)


def _rho(a, b):
    """Correlación de rangos de Spearman (empates con el rango medio)."""
    a, b = pd.Series(np.asarray(a, dtype=float)), pd.Series(np.asarray(b, dtype=float))
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


# --- Textos de la cabecera -------------------------------------------------------------------------------------------------
def _respuesta(tarea, modelo):
    r = _resumen(tarea, modelo)
    top = list(r["imp"].index[:DIRECCIONES])
    nombres = F.enumerar(_nombre(v).lower() for v in top)
    ref = _referente(tarea, modelo)
    rr = _resumen(tarea, ref)
    iguales = _prefijo_comun(_usadas(r), _usadas(rr))
    texto = (f"En {D.en_frase(modelo)}, {nombres} reúnen el {F.pct(r['cuota'].iloc[:DIRECCIONES].sum())} de la importancia SHAP "
             f"media de {F.entero(r['n'])} clientes de prueba.")
    if iguales:
        texto += (f" {_mayuscula(D.en_frase(ref))} coincide en {'la primera' if iguales == 1 else f'las {iguales} primeras'}, en el "
                  "mismo orden.")
    else:
        texto += f" {_mayuscula(D.en_frase(ref))} empieza por otra variable: {_nombre(rr['imp'].index[0]).lower()}."
    return texto


def _usadas(r, k=None):
    """Las variables con aporte distinto de cero, de mayor a menor importancia (las k primeras): un árbol puede no usar
    la mayoría, y su orden entre los ceros no significa nada."""
    v = [x for x in r["imp"].index if r["imp"][x] > 0]
    return v if k is None else v[:k]


def _prefijo_comun(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


# --- 1. La importancia ----------------------------------------------------------------------------------------------------
def _importancia(tarea, modelo):
    """La importancia como lista de barras rayadas (ui.barra_rayas: rótulo y cifra encima de la barra, que cabe en
    cualquier ancho): las TOP_BARRAS primeras variables y el resto agrupado; cada barra se mide contra la mayor."""
    r = _resumen(tarea, modelo)
    imp, cuota, info = r["imp"], r["cuota"], r["info"]
    rol, frase = D.ROL_MODELO[modelo], D.en_frase(modelo)
    escala = info["escala"]
    d = _decimales(imp.iloc[0])
    propias = _usadas(r, TOP_BARRAS)
    resto = len(imp) - len(propias)
    cats = [_nombre(v) for v in propias] + [f"Las otras {resto} variables"]
    vals = [float(cuota[v]) for v in propias] + [float(cuota.drop(propias).sum())]
    top = list(imp.index[:DIRECCIONES])
    titulo = (f"{_mayuscula(F.enumerar(_nombre(v).lower() if i else _nombre(v) for i, v in enumerate(top)))} reúnen el "
              f"{F.pct(cuota.iloc[:DIRECCIONES].sum())} de la importancia {_tras('de', frase)}")
    sub = (f"Importancia de cada variable: la media del valor absoluto de su aporte en {F.entero(r['n'])} clientes de prueba fijos, "
           f"como parte de la suma de las {len(imp)} variables. La última barra agrupa a las otras {resto}.")
    ceros = int((imp == 0).sum())
    nota = (f"Cada barra se mide contra la de {_nombre(imp.index[0]).lower()}, la mayor; la cifra es la parte de la variable en el "
            "total." + (f" Las otras {resto} variables suman el {F.pct(vals[-1])}." if vals[-1] > 0 else ""))
    if ceros:
        nota += (f" {ceros} de las {len(imp)} variables tienen aporte cero en todos esos clientes: {frase} "
                 f"{'no la usa' if ceros == 1 else 'no las usa'}.")
    mayor = max(vals)
    barras = html.Div([ui.barra_rayas(c, F.pct(v), v / mayor, rol) for c, v in zip(cats, vals)], className="rayas-lista")
    filas = [[str(i + 1), _nombre(v), ui.num(imp[v], d + 1), F.pct(cuota[v], 2)] for i, v in enumerate(imp.index)]
    tabla = _tabla(["Puesto", "Variable", f"Media del aporte absoluto ({escala})", "Parte"], filas, alta=True,
                     num=[True, False, True, True])
    return {"titulo": titulo, "sub": sub, "cuerpo": [barras, ui.nota(nota)], "tabla": tabla}


def _lectura(tarea, modelo):
    """Qué es un valor SHAP y en qué escala está el modelo elegido."""
    r = _resumen(tarea, modelo)
    escala = r["info"]["escala"]
    d = _decimales(r["imp"].iloc[0])
    sube = "hacia el riesgo alto" if tarea == "clasificacion" else "hacia un pronóstico mayor"
    return [
        ui.parrafo("Un valor SHAP reparte la salida del modelo para un cliente entre sus variables: la base (la salida media "
                   f"sobre un fondo de clientes de entrenamiento) más la suma de los aportes da exactamente la salida. Un aporte "
                   f"positivo empuja {sube}; uno negativo, en sentido contrario."),
        ui.parrafo(f"En {D.en_frase(modelo)} lo que se reparte es {ESCALA.get(escala, escala)}. En los {F.entero(r['n'])} clientes la "
                   f"base media es {ui.num(r['base'], d)} y la salida media, {ui.num(r['salida'], d)}."),
        ui.nota("Cada modelo reparte su propia escala (logit, margen, probabilidad o pronóstico), así que los aportes de dos modelos "
                "se comparan en orden y en signo, no en magnitud."),
    ]


# --- 2. El enjambre ---------------------------------------------------------------------------------------------------------
def _enjambre(tarea, modelo):
    r = _resumen(tarea, modelo)
    S, imp, info = r["S"], r["imp"], r["info"]
    tipos = _tipos()
    clientes = M.clientes().reindex(S.index)
    variables = _usadas(r, TOP_ENJAMBRE)
    k = len(variables)
    d = _decimales(float(S[variables].abs().to_numpy().max()))
    xs, tonos, textos, filas_tabla, direcciones = [], [], [], [], []
    for i, v in enumerate(variables):
        aporte = S[v].to_numpy(dtype=float)
        valores = clientes[v].to_numpy()
        xs.append(aporte)
        tonos.append(_tono(valores, tipos.get(v)))
        nombre = _nombre(v)
        textos.append((nombre, [_valor_texto(val, tipos.get(v)) for val in valores], [ui.con_signo(a, d) for a in aporte]))
        rho = _rho(pd.to_numeric(pd.Series(valores), errors="coerce"), aporte) if tipos.get(v) != "categorica" else float("nan")
        filas_tabla.append([nombre, ui.con_signo(aporte.min(), d), ui.con_signo(float(np.median(aporte)), d), ui.con_signo(aporte.max(), d),
                            ui.con_signo(rho, 2) if np.isfinite(rho) else "categórica"])
        if i < DIRECCIONES:
            if tipos.get(v) == "categorica":
                direcciones.append(f"{nombre}: es categórica, su aporte depende de la categoría")
            elif not np.isfinite(rho):
                direcciones.append(f"{nombre}: no varía en la muestra")
            elif rho >= MONOTONA:
                direcciones.append(f"{nombre}: a valor más alto, aporte más alto (ρ {ui.con_signo(rho, 2)})")
            elif rho <= -MONOTONA:
                direcciones.append(f"{nombre}: a valor más alto, aporte más bajo (ρ {ui.con_signo(rho, 2)})")
            else:
                direcciones.append(f"{nombre}: el aporte no sube ni baja de forma pareja con el valor (ρ {ui.con_signo(rho, 2)})")
    x = np.concatenate(xs)
    # Un aporte extremo (Naive Bayes llega a cientos en el logit) aplastaría a todos los demás contra el cero: entonces el eje
    # va del percentil 0,5 al 99,5 y los extremos se dibujan en el borde (su valor exacto sigue al pasar el puntero y en la tabla)
    lo_p, hi_p = (min(float(v), 0.0) if j == 0 else max(float(v), 0.0) for j, v in enumerate(np.percentile(x, PERCENTILES_EJE)))
    recorta = (x.max() - x.min()) > RECORTE * (hi_p - lo_p) > 0
    afuera = int(np.sum((x < lo_p) | (x > hi_p))) if recorta else 0
    vistos = [np.clip(xv, lo_p, hi_p) if recorta else xv for xv in xs]
    ys = [k - 1 - i + _apilar(xv) for i, xv in enumerate(vistos)]
    fig = go.Figure()
    for xv, yv, tv, (nombre, val_txt, aporte_txt) in zip(vistos, ys, tonos, textos):  # una traza por variable: su nombre va en la plantilla
        fig.add_trace(go.Scattergl(x=xv, y=yv, mode="markers",
                                 marker={"size": TAM_PUNTO, "color": tv, "colorscale": [[0, ui.ROL["contexto"]], [1, ui.ROL["real"]]],
                                         "cmin": 0, "cmax": 1, "opacity": OPACIDAD, "line": {"width": 0}},
                                 customdata=[[a, b] for a, b in zip(val_txt, aporte_txt)],
                                 hovertemplate=nombre + " = %{customdata[0]} · aporte <b>%{customdata[1]}</b><extra></extra>"))
    ui.linea_cero(fig, "x")
    xv_todos = np.concatenate(vistos)
    tope = float(np.abs(xv_todos).max()) * HOLGURA
    lo, hi = float(xv_todos.min()), float(xv_todos.max())
    lo, hi = lo - (hi - lo) * (HOLGURA - 1), hi + (hi - lo) * (HOLGURA - 1)
    vals, ticks = ui.marcas(lo, hi, ui.paso_redondo(tope, 3), signo=True)
    fig.update_xaxes(range=[lo, hi], tickvals=vals, ticktext=ticks, showgrid=True, gridcolor=ui.ROL["rejilla"])
    # El nombre de cada variable va encima de su fila, a la izquierda, y no en el eje: así el enjambre usa todo el ancho
    # también en el teléfono
    for i, v in enumerate(variables):
        fig.add_annotation(x=0, xref="paper", y=k - 1 - i + ALTO_FILA / 2, text=_nombre(v), xanchor="left", yanchor="bottom",
                           showarrow=False, font=ui.fuente_svg("tinta_2", "--t-xs"), bgcolor=ui.ROL["superficie"])
    fig.update_yaxes(range=[-ALTO_FILA / 2 - MARGEN_FILA, k - 1 + 1 - ALTO_FILA / 2], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(hovermode="closest")
    ui.unidad_eje(fig, EJE.get(info["escala"], "aporte"))
    primera = S[variables[0]]
    frase = D.en_frase(modelo)
    titulo = (f"{_nombre(variables[0])} es la variable que más mueve la salida {_tras('de', frase)}: su aporte va de "
              f"{ui.con_signo(primera.min(), d)} a {ui.con_signo(primera.max(), d)} según el cliente")
    sub = (f"Las {k} variables más importantes. Cada punto es uno de los {F.entero(r['n'])} clientes de prueba de la muestra; su "
           f"posición, el aporte de la variable a la salida ({info['escala']}); donde se apilan, hay muchos clientes con ese aporte. "
           "El tono va del gris claro (valor bajo de la variable en la muestra) al casi negro (valor alto); las categóricas, que no "
           "tienen orden, van en el gris medio.")
    nota = ("; ".join(direcciones) + ". ρ es la correlación de rangos entre el valor de la variable y su aporte. Esto "
            "describe al modelo, no al cliente: un aporte no dice qué pasaría si el cliente cambiara esa variable.")
    if afuera:
        nota = (f"{afuera} {'aporte extremo queda' if afuera == 1 else 'aportes extremos quedan'} fuera del eje (de "
                f"{ui.con_signo(x.min(), d)} a {ui.con_signo(x.max(), d)}) y se {'dibuja' if afuera == 1 else 'dibujan'} en su borde; "
                "el valor exacto aparece al pasar el puntero y en la tabla. " + nota)
    tabla = [_tabla(["Variable", "Aporte mínimo", "Mediana", "Aporte máximo", "ρ valor–aporte"], filas_tabla),
             ui.nota(f"Aportes en la escala {_tras('de', frase)}: {info['escala']}.")]
    leyenda = ui.leyenda([("Valor bajo de la variable", "contexto", "punto"), ("Valor alto de la variable", "real", "punto")], ordenar=False)
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: de {f[1]} a {f[3]}" for f in filas_tabla) + "."
    return {"figura": fig, "titulo": titulo, "sub": sub, "leyenda": leyenda, "tabla": tabla, "nota": nota, "etiqueta": etiqueta}


# --- 3. Frente a la EBM (o, si el elegido es la EBM, frente al mejor modelo de referencia) ---------------------------------
def _comparacion(tarea, modelo):
    ref = _referente(tarea, modelo)
    a, b = _resumen(tarea, modelo), _resumen(tarea, ref)
    orden_a, orden_b = _usadas(a), _usadas(b)
    puesto_b = {v: i + 1 for i, v in enumerate(orden_b)}
    frase, frase_ref = D.en_frase(modelo), D.en_frase(ref)
    iguales = _prefijo_comun(orden_a, orden_b)
    primeras = orden_a[:TOP_COMPARTIDAS]
    comparten = len(set(primeras) & set(orden_b[:TOP_COMPARTIDAS]))
    rho = _rho(a["imp"].reindex(a["cols"]), b["imp"].reindex(a["cols"]))
    if iguales:
        titulo = (f"{_mayuscula(frase)} y {frase_ref} coinciden en {'la primera variable' if iguales == 1 else f'las {iguales} primeras variables'}"
                  f", en el mismo orden")
    else:
        titulo = (f"{_mayuscula(frase)} y {frase_ref} no empiezan por la misma variable: {_nombre(orden_a[0]).lower()} frente a "
                  f"{_nombre(orden_b[0]).lower()}")
    filas = [[_nombre(v), str(i + 1), F.pct(a["cuota"][v], 1), str(puesto_b[v]) if v in puesto_b else "no la usa",
              F.pct(b["cuota"][v], 1)] for i, v in enumerate(orden_a[:FILAS_COMPARACION])]
    tabla = ui.tabla(["Variable", f"Puesto en {D.NOMBRE_CORTO[modelo]}", "Parte", f"Puesto en {D.NOMBRE_CORTO[ref]}", "Parte"], filas,
                     compacta=True, num=[False, True, True, True, True], ocultar_movil=(2, 4))
    cifras = ui.cifras([
        ui.cifra(f"De las {len(primeras)} primeras {_tras('de', frase)}", f"{comparten} de {len(primeras)}", mini=True,
                 nota=f"También están entre las {TOP_COMPARTIDAS} primeras {_tras('de', frase_ref)}."),
        ui.cifra(f"Correlación de rangos en las {len(a['cols'])} variables", ui.num(rho, 2), mini=True,
                 nota="Spearman entre las dos importancias: 1 sería el mismo orden."),
    ])
    nota = ("Se comparan puestos y partes de la importancia de cada modelo, no aportes: cada uno reparte su propia escala. La "
            "correlación usa todas las variables, así que la cola de variables con aportes casi nulos (cuyo orden es casi azar) "
            "la baja aunque las primeras coincidan.")
    cuerpo = [cifras, tabla, ui.nota(nota)]
    fuentes = [FUENTE_INDICE, D.fuente(f"shap/{tarea}__{modelo}.parquet"), D.fuente(f"shap/{tarea}__{ref}.parquet")]
    if tarea == "clasificacion" and "ebm" in (modelo, ref):
        t = D.clave_valor("resultados/importancia_terminos_ebm.csv")
        primera, cuota_ebm = next(iter(t)), _resumen(tarea, "ebm")["cuota"]
        cuerpo.append(ui.nota(f"El informe ordena las variables de la EBM con la importancia de sus términos en el entrenamiento "
                              f"({_nombre(primera).lower()}: {F.pct(t[primera])}); aquí la parte sale de los aportes de "
                              f"{F.entero(_resumen(tarea, 'ebm')['n'])} clientes de prueba ({F.pct(cuota_ebm[primera])}): son dos medidas "
                              "parecidas, no la misma."))
        fuentes.append(FUENTE_TERMINOS)
    titulo_seccion = ("Frente a la EBM" if modelo != "ebm" else
                      f"Frente a {D.NOMBRE_MODELO[ref]}, el mejor modelo de referencia en la prueba")
    return titulo_seccion, titulo, cuerpo, fuentes


def _metodo(tarea, modelo):
    """Cómo se calculó y cómo se verificó el resumen del modelo elegido."""
    r = _resumen(tarea, modelo)
    info = r["info"]
    verif = info["verificacion"]
    pares = [("Método", "exacto" if info["exacto"] else "estimado por muestreo"),
             ("Clientes de prueba explicados", F.entero(info["filas"])),
             ("Fondo", str(V.shap_indice()["fondo"])),
             ("Base + suma de los aportes frente a la salida", f"diferencia máxima {_cientifico(verif['eficiencia_max_dif'])}")]
    for clave, rotulo in FRENTE_A.items():
        if clave in verif:
            pares.append((rotulo, "idénticos" if verif[clave] == 0 else f"diferencia máxima {_cientifico(verif[clave])}"))
    if "error_monte_carlo_medio" in verif:
        pares.append(("Error de Monte Carlo medio de cada aporte",
                      f"{ui.num(verif['error_monte_carlo_medio'], _decimales(verif['error_monte_carlo_medio']))} ({info['escala']})"))
    cuerpo = [ui.parrafo(METODO.get(info["metodo"], info["metodo"])), ui.ficha(pares)]
    if not any(k in verif for k in FRENTE_A):
        cuerpo.append(ui.nota("La comprobación es la de eficiencia: la base más la suma de los aportes reproduce la salida del modelo "
                              "en cada cliente." + (" Se calculó con la propia librería shap." if info["metodo"] == "arbol" else "")))
    else:
        cuerpo.append(ui.nota("Además de la eficiencia, los aportes se compararon con la librería shap en una parte de la muestra."))
    return cuerpo


def _enlaces(tarea, modelo):
    return [ui.boton("Ver cómo decide la EBM", href=app.ruta_vista("decide"), icono="modelo"),
            ui.boton("Ver la explicación de un cliente", href=app.ruta_vista("prediccion") + f"?tarea={tarea}&modelo={modelo}", icono="equipo"),
            ui.boton("Ver el diagnóstico", href=app.ruta_vista("diagnostico") + f"?tarea={tarea}&modelo={modelo}", icono="grafico")]


# --- La pantalla ----------------------------------------------------------------------------------------------------------
def _salidas(gid):
    return [Output(f"{gid}{sufijo}", propiedad) for sufijo, propiedad in PARTES_GRAFICO]


def _partes(g):
    return [g["figura"], g["titulo"], g["sub"], g["leyenda"], g["tabla"], g["nota"], g["etiqueta"]]


def layout(tarea=None, modelo=None, **_):
    """`/modelos/interpretabilidad?tarea=regresion&modelo=lasso` abre con esa tarea y ese modelo; entonces los controles no
    reponen el valor recordado en la sesión."""
    t, m, pedida = _de_la_direccion(tarea, modelo)
    control_tarea = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", [{"label": D.NOMBRE_TAREA[x], "value": x} for x in TAREAS], t,
                                                       persistir=not pedida))
    control_modelo = ui.control(html.Span(_rotulo_modelo(m), id=f"{PRE}-rotulo-modelo", className="tabla__clave"),
                                _selector_modelo(t, m, not pedida))
    fuentes = [FUENTE_INDICE, D.fuente("shap/<tarea>__<modelo>.parquet")]
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), html.Span(id=f"{PRE}-respuesta"), [control_tarea, control_modelo],
                      [html.Div(_enlaces(t, m), className="botonera", id=f"{PRE}-enlaces")], pantalla=PANTALLA),
        ui.seccion("Qué variables pesan más", [
            ui.rejilla([ui.tarjeta(html.Div(id=f"{PRE}-imp", className="resultados"), titulo=html.Span(id=f"{PRE}-imp-titulo"),
                                   subtitulo=html.Span(id=f"{PRE}-imp-sub"), detalles=html.Div(id=f"{PRE}-imp-tabla"), fuentes=fuentes),
                        ui.tarjeta(html.Div(id=f"{PRE}-lectura", className="resultados"), titulo="Qué es un valor SHAP",
                                   fuentes=[FUENTE_INDICE])], "2-1"),
        ], subtitulo="El resumen SHAP global del modelo elegido, precalculado con clientes de prueba fijos (los mismos para todos los "
                     "modelos; en k-NN, una parte de ellos)."),
        ui.seccion("Cliente por cliente", [
            ui.grafico(f"{PRE}-enj", alto="alto", fuentes=fuentes + [FUENTE_CLIENTES]),
        ], subtitulo="El mismo resumen sin promediar: además de cuánto pesa cada variable, en qué sentido empuja según su valor."),
        ui.seccion(html.Span(id=f"{PRE}-s3-titulo"), [
            ui.rejilla([ui.tarjeta(html.Div(id=f"{PRE}-comp", className="resultados"), titulo=html.Span(id=f"{PRE}-comp-titulo"),
                                   fuentes=[FUENTE_INDICE, FUENTE_TERMINOS]),
                        ui.tarjeta(html.Div(id=f"{PRE}-metodo", className="resultados"), titulo="Cómo se calculó y cómo se verificó",
                                   fuentes=[FUENTE_INDICE])], "2-1"),
            ui.aviso("Un valor SHAP describe cómo el modelo reparte su salida entre las variables de un cliente. No dice qué pasaría "
                     "si el cliente cambiara una variable ni por qué un cliente se va: los datos son sintéticos y el objetivo es un "
                     "puntaje de propensión, no la fuga observada.", None, "Describe al modelo, no al cliente"),
        ], subtitulo="La guía pide comparar la interpretabilidad del modelo propuesto con la de los demás: aquí, el orden de las "
                     "variables según cada uno. La EBM, además, se lee directamente en sus curvas de forma (vista «Cómo decide»)."),
    ])


@callback(Output(f"{PRE}-modelo", "options"), Output(f"{PRE}-modelo", "value"), Input(f"{PRE}-tarea", "value"),
          State(f"{PRE}-modelo", "value"))
def _por_tarea(tarea, modelo):
    """Deja en el selector los ocho modelos de la tarea; si el elegido no es de ella, pasa al propuesto."""
    t = _tarea(tarea)
    valido = _modelo(t, modelo)
    return _opciones(t), (no_update if valido == modelo else valido)


@callback(Output(f"{PRE}-respuesta", "children"), Output(f"{PRE}-imp-titulo", "children"), Output(f"{PRE}-imp-sub", "children"),
          Output(f"{PRE}-imp", "children"), Output(f"{PRE}-imp-tabla", "children"), Output(f"{PRE}-lectura", "children"),
          *_salidas(f"{PRE}-enj"), Output(f"{PRE}-s3-titulo", "children"), Output(f"{PRE}-comp-titulo", "children"),
          Output(f"{PRE}-comp", "children"), Output(f"{PRE}-metodo", "children"), Output(f"{PRE}-enlaces", "children"),
          Output(f"{PRE}-rotulo-modelo", "children"),
          Input(f"{PRE}-tarea", "value"), Input(f"{PRE}-modelo", "value"))
def _dibujar(tarea, modelo):
    """Todo lo que depende del modelo elegido. Si el modelo no es de la tarea (acaba de cambiar), espera al selector."""
    t = _tarea(tarea)
    n_salidas = 1 + 4 + len(PARTES_GRAFICO) + 7
    if modelo is not None and modelo not in V.modelos(t):
        return (no_update,) * n_salidas
    m = _modelo(t, modelo)
    try:
        titulo_seccion, titulo_comp, comp, _fuentes = _comparacion(t, m)
        imp = _importancia(t, m)
        return (_respuesta(t, m), imp["titulo"], imp["sub"], imp["cuerpo"], imp["tabla"], _lectura(t, m), *_partes(_enjambre(t, m)), titulo_seccion,
                titulo_comp, comp, _metodo(t, m), _enlaces(t, m), _rotulo_modelo(m))
    except Exception as e:  # noqa: BLE001  (la pantalla no se cae: lo dice en un aviso)
        falla = ui.aviso(f"No se pudo leer el resumen SHAP de {D.NOMBRE_MODELO.get(m, m)} ({type(e).__name__}: {e}).", "critico",
                         "Falló la lectura")
        vacio = ui.figura_vacia("Sin datos")
        grafico = [vacio, "Sin datos", "", None, None, None, "Sin datos"]
        return ("", "Sin datos", "", falla, None, None, *grafico, "Frente a la EBM", "Sin datos", None, None, _enlaces(t, m), _rotulo_modelo(m))
