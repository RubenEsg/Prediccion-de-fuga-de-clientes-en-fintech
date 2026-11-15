"""Piezas de interfaz del laboratorio con el estilo de la referencia que eligió el usuario (DISENO.md): la navegación arriba,
títulos enormes, tarjetas casi blancas con el botón redondo «…», cifras grandes, chips en píldora y barras rayadas.

Todo color sale de tokens.py (que lee assets/estilos.css): ROL para Plotly y var(--…) para los estilos en línea. Aquí no se
escribe ningún color ni tamaño de letra. Contenido:
  · números y fechas en es-CO (con el menos tipográfico «−»; fmt y pct vienen de formato.py, y pct recibe una fracción);
  · la plantilla «fuga» de Plotly y las ayudas de los gráficos (traza con las formas rayada, foco y escalon_rayado,
    barras_foco, banda_foco, etiquetas_columnas, hover, paso_redondo, marcas, formato_eje, unidad_eje, linea_cero,
    referencia, banda, franja, fuente_svg);
  · los componentes del catálogo (DISENO.md §4): encabezado, arriba, rejilla, seccion, tarjeta (con «…»), bloque, mas,
    cifra, heroe, saldo, antes_despues, chip, chip_cambio, chip_valor, barra_rayas, puntos, tarjeta_insight, aviso, matriz,
    ficha, grafico, leyenda, tabla, desplegable, fuentes, destino, boton, boton_redondo, segmentado, control, casilla,
    deslizador, campo_numero, selector, pie.

Roles de serie (DISENO.md §2), en el orden de la leyenda: «modelo» (azul: la EBM, lo que se propone), «rival» (rosa:
XGBoost y los otros modelos de árboles), «base» (gris: modelos lineales, k-NN, Naive Bayes y la Entrega 2), «real» (casi
negro: lo observado) y «contexto» (gris claro).
"""
import math
from decimal import Decimal

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html

import tokens as T
import formato as _F
from formato import fmt, pct  # noqa: F401  (se reexportan para las páginas)
from tokens import LAVADO, ROL, TOKENS, VAR, alfa, px  # noqa: F401  (se reexportan)

TOK = TOKENS                                  # el nombre que usa DISENO.md
FUENTE = T.FUENTE_TEXTO

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
MESES_CORTOS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

# Especificaciones fijas de las marcas (dataviz: marks-and-anatomy.md; DISENO.md §5)
ANCHO_LINEA = 2          # px: toda línea
ANCHO_REAL = 2.5         # px: lo observado
MARCADOR = 8             # px: marcadores, siempre con anillo
ANILLO = 2               # px: anillo de --superficie alrededor de un marcador
PUNTA_BARRA = 6          # px: punta redondeada de las barras (base cuadrada)
BARRA_MAX = 24           # px: grosor máximo de una barra (se fija con width en unidades de categoría)
RAYA = 9                 # px: paso del rayado diagonal de las barras (marker.pattern)
RAYA_SOLIDEZ = 0.34      # parte del paso que ocupa la raya
RAYA_LLENADO = 7         # px: paso del rayado vertical bajo un escalón (fillpattern)
MARGEN_ETIQUETAS = 56    # px: margen derecho cuando hay valores en la punta de barras horizontales

GRAF_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


# --- Números y fechas -------------------------------------------------------------------------------------------------
def _finito(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def num(x, d=0):
    """Número en es-CO con el menos tipográfico: 1.234,5 · −0,32 (los niveles van sin «+»). Ojo: sin `d` redondea a
    entero; una métrica entre 0 y 1 pide sus decimales (ui.num(0.8289, 4) → «0,8289»)."""
    return _F.num(x, d)


def con_signo(x, d=0):
    """Número con signo y el menos tipográfico: +1.234 / −0,32 (diferencias, pasos de una cascada y saldos). Lo que
    redondea a cero va sin signo, igual que en formato.con_signo."""
    return _F.con_signo(x, d)


def fecha_larga(f, con_dia=True):
    f = pd.Timestamp(f)
    s = f"{f.day} de {MESES[f.month - 1]} de {f.year}"
    return f"{DIAS[f.dayofweek]} {s}" if con_dia else s


def fecha_corta(f):
    f = pd.Timestamp(f)
    return f"{f.day} {MESES_CORTOS[f.month - 1]}"


# --- Plantilla de Plotly «fuga» (DISENO.md §5) -------------------------------------------------------------------------
TRANSPARENTE = alfa(ROL["superficie"], 0)


def fuente_svg(rol="tinta_2", paso="--t-xs"):
    """Fuente del texto de un gráfico (ejes, anotaciones, valor en la punta): familia de la app, un paso de la escala y un
    rol de texto («tinta», «tinta_2», «tinta_3»). Nunca el color de la serie."""
    return dict(family=FUENTE, size=px(paso), color=ROL[rol])


# Estilo de la referencia: rótulos grises (oscurecidos lo justo para 4,5:1), rejilla muy clara, sin títulos de eje, la
# etiqueta flotante en píldora blanca con borde fino y texto oscuro. El eje y trae enteros (",.0f"): un eje de valores
# decimales se ajusta con ui.formato_eje o con ui.marcas.
_EJE_X = dict(showgrid=False, zeroline=False, showline=True, linecolor=ROL["rejilla"], linewidth=1, ticks="", tickangle=0,
              tickfont=fuente_svg("tinta_3"), automargin=True, fixedrange=True)
_EJE_Y = dict(showgrid=True, gridcolor=ROL["rejilla"], gridwidth=1, griddash="solid", zeroline=False, showline=False, ticks="",
              tickfont=fuente_svg("tinta_3"), tickformat=",.0f", automargin=True, fixedrange=True)

pio.templates["fuga"] = go.layout.Template(
    layout=go.Layout(
        font=fuente_svg("tinta_2"),
        paper_bgcolor=TRANSPARENTE, plot_bgcolor=TRANSPARENTE,   # la tarjeta pone --superficie
        separators=",.",                                          # coma decimal y punto de miles
        colorway=[ROL[r] for r in T.SERIES],                      # solo por si una traza no declara su rol
        margin=dict(l=px("--e-1"), r=px("--e-4"), t=px("--e-3"), b=px("--e-1")),
        showlegend=False,                                         # la leyenda es HTML (ui.leyenda)
        hovermode="x unified",
        hoverlabel=dict(bgcolor=ROL["blanco"], bordercolor=ROL["linea"], font=fuente_svg("tinta", "--t-s"), align="left", namelength=-1),
        xaxis=_EJE_X, yaxis=_EJE_Y,
        barcornerradius=PUNTA_BARRA, bargap=0.22, bargroupgap=0.08,
        annotationdefaults=dict(font=fuente_svg("tinta_2"), showarrow=False, arrowcolor=ROL["tinta_3"], arrowwidth=1),
        shapedefaults=dict(line=dict(color=ROL["tinta_3"], width=1)),
        dragmode=False,
    ),
    data=dict(
        bar=[go.Bar(marker=dict(line=dict(width=0)), textposition="outside", cliponaxis=False, textfont=fuente_svg("tinta"))],
        scatter=[go.Scatter(line=dict(width=ANCHO_LINEA, shape="linear"),
                            marker=dict(size=MARCADOR, line=dict(width=ANILLO, color=ROL["superficie"])))],
    ),
)
pio.templates.default = "fuga"


# --- Ayudas de los gráficos -------------------------------------------------------------------------------------------
# Forma por defecto de cada rol de serie cuando es una línea (DISENO.md §5). El gris y el rosa nunca llevan la misma forma:
# por eso la base va a trazos.
_FORMA = {"modelo": "linea", "rival": "linea", "base": "trazos", "real": "gruesa", "contexto": "linea"}
FORMAS = ("barra", "foco", "rayada", "linea", "gruesa", "trazos", "punteada", "escalon", "escalon_rayado", "area", "punto")
# Tinte de cada familia: el fondo del rayado y de la matriz de puntos. Cubre los cinco roles de serie; un rol que no esté
# aquí no se puede rayar (ui.tinte falla en vez de pintar un gris que nadie pidió).
TINTE = {"modelo": "modelo_tinte", "rival": "rival_tinte", "base": "base_tinte", "real": "base_tinte", "contexto": "base_tinte"}
for _r in T.SERIES:
    if _r not in TINTE or _r not in _FORMA:
        raise KeyError(f"al rol de serie {_r!r} le falta su tinte o su forma por defecto en ui.py")


def _rol(rol):
    """Comprueba que `rol` es un rol de tokens.ROL y lo devuelve."""
    if rol not in ROL:
        raise KeyError(f"rol desconocido: {rol!r} (roles: {', '.join(ROL)})")
    return rol


def _rol_tinte(rol):
    """Nombre del rol de tinte de la familia de `rol`; falla si el rol no tiene tinte."""
    if rol not in TINTE:
        raise KeyError(f"rol sin tinte: {rol!r} (con tinte: {', '.join(TINTE)})")
    return TINTE[rol]


def tinte(rol):
    """Color del tinte de la familia de un rol (fondo del rayado): ROL["modelo_tinte"] para «modelo», etc."""
    return ROL[_rol_tinte(rol)]


def _rayado(rol, forma="/"):
    """marker.pattern del rayado de un rol: rayas del color del rol sobre su tinte. forma "" = barra sólida (la de foco)."""
    return {"shape": forma, "fgcolor": _color(rol), "bgcolor": tinte(rol), "size": RAYA, "solidity": RAYA_SOLIDEZ, "fillmode": "replace"}


def _color(rol):
    """Color de un rol."""
    if rol not in ROL:
        raise KeyError(f"rol sin color: {rol!r} (roles: {', '.join(ROL)})")
    return ROL[rol]


def _marcador(c):
    return dict(size=MARCADOR, color=c, line=dict(width=ANILLO, color=ROL["superficie"]))


def traza(rol, forma=None, *, escalon=False, marcadores=None, ancho=None):
    """Estilo de una traza según su rol y su forma, para mezclar en go.Bar / go.Scatter con **:
    go.Bar(x=…, y=…, **ui.traza("modelo", "barra", ancho=0.34)) · go.Scatter(x=…, y=…, **ui.traza("real")).
    Roles de serie: modelo, rival, base, real, contexto. Formas: barra o foco (sólida: la serie en foco), rayada (rayas
    diagonales del color sobre su tinte: las demás de la familia, los rivales y la base), linea, gruesa (2,5 px), trazos,
    punteada, escalon, escalon_rayado (escalón «hv» con relleno de rayas verticales claras, como «Retention» de la
    referencia), area, punto. Sin forma, la del rol: modelo, rival y contexto en línea, base a trazos, real gruesa.
    Los marcadores (8 px con anillo) solo si se piden."""
    c = _color(rol)
    forma = forma or _FORMA.get(rol, "linea")
    if forma not in FORMAS:
        raise ValueError(f"forma desconocida: {forma!r}")
    if forma in ("barra", "foco", "rayada"):
        d = {"marker": {"color": c, "line": {"width": 0}}}
        if forma == "rayada":
            d["marker"]["pattern"] = _rayado(rol)
        if ancho is not None:
            d["width"] = ancho
        return d
    if forma == "escalon_rayado":
        return {"mode": "lines", "line": {"color": c, "width": ANCHO_REAL, "shape": "hv"}, "fill": "tozeroy",
                "fillcolor": alfa(c, LAVADO / 2),
                "fillpattern": {"shape": "|", "fgcolor": alfa(c, 0.3), "size": RAYA_LLENADO, "solidity": 0.22, "fillmode": "overlay"}}
    if forma == "punto":
        return {"mode": "markers", "marker": _marcador(c)}
    linea = {"color": c, "width": ANCHO_REAL if forma == "gruesa" else ANCHO_LINEA,
             "shape": "hvh" if (escalon or forma == "escalon") else "linear"}
    if forma == "trazos":
        linea["dash"] = "dash"
    elif forma == "punteada":
        linea["dash"] = "dot"
    d = {"mode": "lines+markers" if marcadores else "lines", "line": linea}
    if marcadores:
        d["marker"] = _marcador(c)
    if forma == "area":
        d.update(fill="tozeroy", fillcolor=alfa(c, LAVADO))
    return d


def hover(nombre, valores, unidad="", d=0, signo=False, categorias=None):
    """customdata ya formateado (es-CO, «−») y su hovertemplate: «Nombre: <b>valor</b> unidad». El valor manda (dataviz).
    Con `categorias` (barras horizontales), «categoría: <b>valor</b> unidad». Uso: go.Bar(…, **ui.hover("AUC", aucs, d=4))."""
    f = con_signo if signo else num
    textos = [f(v, d) for v in valores]
    u = f" {unidad}" if unidad else ""
    if categorias is not None:
        cd = [[str(c), t] for c, t in zip(categorias, textos)]
        plantilla = ("" if not nombre else f"{nombre} · ") + "%{customdata[0]}: <b>%{customdata[1]}</b>" + u
    else:
        cd = [[t] for t in textos]
        plantilla = f"{nombre}: <b>%{{customdata[0]}}</b>{u}"
    return {"customdata": cd, "hovertemplate": plantilla + "<extra></extra>"}


def hover_texto(textos):
    """Texto al pasar el puntero ya armado por la página (una cadena por punto, con sus cifras ya formateadas)."""
    return {"customdata": [[t] for t in textos], "hovertemplate": "%{customdata[0]}<extra></extra>"}


def paso_redondo(tope, maximo_marcas=3):
    """Paso «redondo» para que un eje llegue a `tope` con pocas marcas a cada lado del cero. También con topes menores
    que 1 (métricas y probabilidades): paso_redondo(0.87) → 0,5 · paso_redondo(0.06) → 0,02 · paso_redondo(37) → 20."""
    tope = abs(tope) or 1
    if tope < 1:
        magnitud = 10 ** math.floor(math.log10(tope))
        for k in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10):
            p = round(k * magnitud, 12)
            if p < 1 and math.ceil(tope / p - 1e-9) <= maximo_marcas:
                return p
    for p in (1, 2, 2.5, 4, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000):
        if math.ceil(tope / p) <= maximo_marcas:
            return p
    return 10 ** math.ceil(math.log10(tope))


def ancho_barra(n_categorias, largo_eje_px=None):
    """Ancho de barra (en unidades de categoría, el `width` de go.Bar) para que no pase de 24 px ni en el lienzo más ancho:
    el contenido llega como mucho a --ancho-contenido menos el relleno del gráfico. Con 17 categorías da ≈ 0,33. Para
    barras horizontales, `largo_eje_px` es el alto del lienzo (240 en «bajo»)."""
    largo = largo_eje_px or (px("--ancho-contenido") - 2 * px("--e-5"))
    return round(min(0.8, BARRA_MAX / (largo / max(n_categorias, 1))), 3)


def _decimales(paso):
    """Decimales que hacen falta para escribir un paso: 0,25 → 2 · 0,5 → 1 · 5 → 0."""
    return max(0, -Decimal(repr(float(paso))).normalize().as_tuple().exponent)


def marcas(lo, hi, paso, d=None, signo=False):
    """(tickvals, ticktext) de lo a hi cada `paso`, en es-CO; con signo=True los positivos llevan «+» (ejes de diferencias).
    Sin `d`, los decimales salen del paso (0,25 → 2; 0,5 → 1; 5 → 0). El cero va siempre como «0».
    ui.marcas(0, 1, 0.25) → ([0.0, 0.25, 0.5, 0.75, 1.0], ['0', '0,25', '0,50', '0,75', '1,00'])."""
    if d is None:
        d = _decimales(paso)
    k = math.floor(lo / paso + 1e-9)
    vals = []
    while k * paso <= hi + 1e-9:
        vals.append(round(k * paso, 10))
        k += 1

    def texto(t):
        return "0" if t == 0 else (con_signo(t, d) if signo else num(t, d))
    return vals, [texto(t) for t in vals]


def formato_eje(fig, eje="y", d=2, sufijo=""):
    """Formato de las marcas de un eje de valores decimales: `d` decimales con coma (la plantilla trae enteros en el eje y)
    y un sufijo opcional (« %»). Devuelve la figura. ui.formato_eje(fig, "y", 2) · ui.formato_eje(fig, "x", 0, " %")."""
    if eje not in ("x", "y"):
        raise ValueError(f"eje desconocido: {eje!r} (x o y)")
    props = dict(tickformat=f",.{int(d)}f", ticksuffix=sufijo)
    (fig.update_yaxes if eje == "y" else fig.update_xaxes)(**props)
    return fig


def unidad_eje(fig, texto):
    """La unidad del eje de valores en horizontal, arriba a la izquierda (no hay títulos de eje girados)."""
    fig.add_annotation(text=texto, xref="paper", yref="paper", x=0, y=1, xanchor="left", yanchor="bottom", yshift=px("--e-1"),
                       showarrow=False, font=fuente_svg("tinta_3"))
    fig.update_layout(margin_t=px("--e-6"))
    return fig


def linea_cero(fig, eje="y"):
    """Línea de cero en --tinta-3, 1 px: solo donde importa el signo."""
    linea = dict(color=ROL["cero"], width=1)
    (fig.add_hline(y=0, line=linea) if eje == "y" else fig.add_vline(x=0, line=linea))
    return fig


def referencia(fig, x=None, y=None, texto=None):
    """Línea de referencia sólida de 1 px en --tinta-3 con su rótulo en --tinta-2 (p. ej. el umbral de decisión). Nunca a
    trazos."""
    linea = dict(color=ROL["referencia"], width=1)
    if x is not None:
        fig.add_vline(x=x, line=linea)
        if texto:
            fig.add_annotation(x=x, y=1, yref="paper", text=texto, yanchor="bottom", showarrow=False, font=fuente_svg("tinta_2"))
    if y is not None:
        fig.add_hline(y=y, line=linea)
        if texto:
            fig.add_annotation(x=1, xref="paper", y=y, text=texto, xanchor="right", yanchor="bottom", showarrow=False, font=fuente_svg("tinta_2"))
    return fig


def banda(fig, x, bajo, alto, rol="modelo"):
    """Banda de un intervalo (p. ej. un intervalo de confianza) en el lavado del color del rol, detrás de las líneas: dos
    trazas sin línea, la de abajo rellena hasta la de arriba. Sin texto al pasar el puntero (la cifra va en la línea)."""
    c = _color(rol)
    sin_linea = {"mode": "lines", "line": {"width": 0, "color": c}, "hoverinfo": "skip"}
    fig.add_trace(go.Scatter(x=x, y=list(alto), name="Límite superior", **sin_linea))
    fig.add_trace(go.Scatter(x=x, y=list(bajo), name="Límite inferior", fill="tonexty", fillcolor=alfa(c, LAVADO), **sin_linea))
    return fig


def franja(fig, x0, x1):
    """Franja de rango o de selección (--franja: el azul al 9 %), detrás de las marcas."""
    fig.add_vrect(x0=x0, x1=x1, fillcolor=ROL["franja"], line_width=0, layer="below")
    return fig


def barras_foco(rol, categorias, foco=None, ancho=None):
    """Barras de una serie con una columna en foco (sólida) y las demás rayadas, como «Payments» de la referencia:
    go.Bar(x=cats, y=vals, **ui.barras_foco("modelo", cats, foco="EBM")). Sin foco, todas rayadas."""
    formas = ["" if (foco is not None and c == foco) else "/" for c in categorias]
    d = {"marker": {"color": _color(rol), "line": {"width": 0}, "pattern": dict(_rayado(rol), shape=formas)}}
    if ancho is not None:
        d["width"] = ancho
    return d


def banda_foco(fig, x, categorias=None, medio_ancho=0.5):
    """Banda vertical azul clara detrás de la columna en foco (la de «Successful Payments»). `x` es la categoría (con
    `categorias`, la lista del eje) o la posición numérica."""
    i = categorias.index(x) if categorias is not None else x
    fig.add_vrect(x0=i - medio_ancho, x1=i + medio_ancho, fillcolor=ROL["franja"], line_width=0, layer="below")
    return fig


def etiquetas_columnas(fig, categorias, textos, foco=None, rotulos=None):
    """Rótulo y valor encima de cada columna, arriba del lienzo (gris; el de foco en tinta), como en «Payments».
    `textos` ya formateados (ui.num); `rotulos`, una línea opcional encima del valor."""
    for i, (c, t) in enumerate(zip(categorias, textos)):
        rol = "tinta" if c == foco else "tinta_3"
        texto = t if not rotulos else f"{rotulos[i]}<br>{t}"
        fig.add_annotation(x=c, y=1, yref="paper", yanchor="bottom", yshift=px("--e-1"), text=texto, showarrow=False,
                           align="center", font=fuente_svg(rol, "--t-s"))
    fig.update_layout(margin_t=px("--e-8") if rotulos else px("--e-6"))
    return fig


def figura_vacia(texto="Sin datos"):
    fig = go.Figure()
    fig.add_annotation(text=texto, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=fuente_svg("tinta_3", "--t-s"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# --- Piezas básicas ---------------------------------------------------------------------------------------------------
def _lista(x):
    if x is None:
        return []
    return [h for h in x if h is not None] if isinstance(x, (list, tuple)) else [x]


def _kw(**pares):
    """Solo las propiedades con valor (Dash no siempre acepta None)."""
    return {k: v for k, v in pares.items() if v is not None}


def _clases(*cs):
    return " ".join(c for c in cs if c)


def _persistencia(persistir):
    """Los controles recuerdan su valor durante la sesión del navegador, salvo que se pida lo contrario."""
    return {"persistence": True, "persistence_type": "session"} if persistir else {}


def icono(nombre, clase=""):
    """Icono de la familia de la app (assets/iconos), del color del texto."""
    return html.Span(className=_clases("glifo", f"i-{nombre}", clase), **{"aria-hidden": "true"})


_glifo = icono   # para las funciones con un parámetro llamado «icono»

# Pantalla → icono de su cuadro (ui.destino y la cabecera)
PANTALLA_ICONO = {"inicio": "inicio", "datos": "metodo", "modelo": "modelo", "cliente": "equipo", "campana": "demanda",
                  "comparacion": "balanza", "robustez": "grafico", "errores": "lupa", "limites": "info",
                  "contexto": "inicio", "eda": "metodo", "lideres": "tabla", "prediccion": "resultado", "diagnostico": "grafico",
                  "interpretabilidad": "lupa", "propuesto": "modelo", "decide": "modelo", "ficha": "info"}


def cuadro(pantalla, icono=None, clase=""):
    """Cuadro redondo con el icono de una pantalla (36 px). `pantalla` es su id (inicio, datos, modelo, cliente, campana,
    comparacion, robustez, errores, limites); con `icono` se elige otro glifo."""
    return html.Span(_glifo(icono or PANTALLA_ICONO.get(pantalla, "inicio")), className=_clases("cuadro-color", clase),
                     **{"aria-hidden": "true"})


# --- Estructura de la pantalla ----------------------------------------------------------------------------------------
def pantalla(hijos, clase=""):
    """El contenedor de una pantalla: cabecera (o el grupo de arriba) y sus secciones, con 48 px entre ellas."""
    return html.Div(_lista(hijos), className=_clases("pantalla", clase))


def arriba(cabecera_, controles=None, respuesta=None):
    """Grupo de arriba: en una fila, la cabecera (título enorme + subtítulo) a la izquierda y los controles de la pantalla a
    la derecha en un grupo de píldoras (debajo si no caben) → la respuesta (24 px). Los 48 px empiezan después."""
    fila = [cabecera_] + ([fila_controles(controles)] if controles is not None else [])
    return html.Div([html.Div(fila, className="encabezado")] + _lista(respuesta), className="pantalla__arriba")


def encabezado(titulo, subtitulo=None, controles=None, respuesta=None, pantalla=None):
    """Encabezado de pantalla de la referencia («Overview» + controles a la derecha): h1 enorme en peso normal → subtítulo
    gris que responde con la cifra; `controles` (lista de segmentados, selectores…) van en la píldora de la derecha;
    `respuesta` (fichas, avisos) debajo. ui.encabezado("¿Es mejor que los otros modelos?", "Empata con…", [sel_metrica])."""
    return arriba(cabecera(pantalla or "inicio", titulo, subtitulo), controles, respuesta)


def rejilla(hijos, columnas="2-1", clase="", id=None):
    """Rejilla de tarjetas del tablero: «2-1» (ancha + angosta, como Payments + Gross Volume), «1-2», «1-1» o «3». Por
    debajo de 900 px, una columna."""
    return html.Div(_lista(hijos), className=_clases("rejilla", f"rejilla--{columnas}", clase), **_kw(id=id))


def cabecera(pantalla, pregunta, subtitulo=None):
    """Cabecera de pantalla: la pregunta (h1) → el subtítulo que la responde con la cifra. Una por pantalla y sin
    controles. (El cuadro con el icono de la pantalla viaja en el DOM, pero el CSS no lo muestra.)"""
    texto = [cuadro(pantalla, None, "cuadro-color--cabecera"), html.H1(pregunta, className="cabecera__pregunta")]
    if subtitulo is not None:
        texto.append(html.P(subtitulo, className="cabecera__subtitulo"))
    return html.Header(html.Div(texto, className="cabecera__texto"), className="cabecera")


def fila_controles(hijos):
    """Fila de controles sin tarjeta, alineada a la izquierda, encima de lo que cambia (en el teléfono, uno por línea)."""
    return html.Div(_lista(hijos), className="controles")


def seccion(titulo, hijos, subtitulo=None, fuentes=None, controles=None, id=None, intro=None):
    """Sección: h2 con el mensaje → subtítulo debajo → intro → fila de controles (lo que cambia esa parte) → cuerpo (24 px)
    → un pie de fuentes. `fuentes` es una lista de rutas o un componente."""
    cab = [html.H2(titulo, className="seccion__titulo")]
    if subtitulo is not None:
        cab.append(html.P(subtitulo, className="seccion__sub"))
    partes = [html.Div(cab, className="seccion__cab")]
    if intro is not None:
        partes.append(html.P(intro, className="seccion__intro"))
    if controles is not None:
        partes.append(fila_controles(controles))
    partes += _lista(hijos)
    if fuentes:
        partes.append(globals()["fuentes"](fuentes) if isinstance(fuentes, (list, tuple, str)) else fuentes)
    return html.Section(partes, className="seccion", **_kw(id=id))


def mas(detalles=None, fuentes_=None, titulo="Ver los datos", id=None, clase=""):
    """Botón redondo «…» de la esquina de una tarjeta: abre (details/summary, sin callbacks) un panel flotante con
    `detalles` (la tabla gemela, una nota) y el pie de `fuentes_`. El texto «Ver los datos» queda para lectores de pantalla.
    ui.grafico ya lo trae (con su tabla y sus fuentes); ui.tarjeta lo pone si recibe detalles o fuentes."""
    cuerpo = [html.P(titulo, className="mas__titulo")] + _lista(detalles)
    if fuentes_:
        cuerpo.append(globals()["fuentes"](fuentes_) if isinstance(fuentes_, (list, tuple, str)) else fuentes_)
    return html.Details([html.Summary([_glifo("puntos"), html.Span(titulo, className="oculto")], title=f"{titulo} y las fuentes"),
                         html.Div(cuerpo, className="mas__panel")], className=_clases("mas", clase), **_kw(id=id))


def tarjeta(hijos=None, titulo=None, subtitulo=None, clase="", id=None, detalles=None, fuentes=None, cuerpo=None):
    """Tarjeta de la referencia (casi blanca, radio 26, borde fino, sombra difusa): cabecera con el título (24 px, peso
    normal) y, si hay `detalles` o `fuentes`, el botón redondo «…» a la derecha que los abre → el cuerpo. Lo anidado va en
    blanco con borde fino. Uso: ui.tarjeta(cuerpo, titulo="Por modelo", detalles=ui.tabla(…), fuentes=["datos/x.csv"]).
    (También ui.bloque(titulo, cuerpo, detalles=…); `cuerpo` es otro nombre de `hijos`.) Un gráfico va en ui.grafico."""
    partes = []
    hay_mas = detalles is not None or bool(fuentes)
    if titulo is not None or hay_mas:
        t = [html.H3(titulo, className="tarjeta__titulo")] if titulo is not None else []
        if subtitulo is not None:
            t.append(html.P(subtitulo, className="tarjeta__sub"))
        cab = [html.Div(t, className="tarjeta__textos")] + ([mas(detalles, fuentes)] if hay_mas else [])
        partes.append(html.Div(cab, className="tarjeta__cab"))
    partes += _lista(cuerpo if hijos is None else hijos)
    return html.Div(partes, className=_clases("tarjeta", clase), **_kw(id=id))


def bloque(titulo, cuerpo, detalles=None, subtitulo=None, fuentes=None, clase="", id=None):
    """La tarjeta con el orden de la referencia: título → cuerpo, y el «…» con `detalles` y `fuentes`."""
    return tarjeta(cuerpo, titulo=titulo, subtitulo=subtitulo, clase=clase, id=id, detalles=detalles, fuentes=fuentes)


def parrafo(texto, clase=""):
    return html.P(texto, className=_clases("parrafo", clase))


def nota(texto, clase=""):
    """Nota de una línea (--t-s, --tinta-3)."""
    return html.P(texto, className=_clases("nota", clase))


def pie(hijos):
    """Pie de pantalla: de dónde salen los datos y quién hizo el trabajo."""
    return html.Footer(_lista(hijos), className="pie")


# --- Cifras -----------------------------------------------------------------------------------------------------------
def icono_cambio(v):
    """Flecha abajo, flecha arriba o igual según el signo (la dirección nunca va solo en el color)."""
    return "flecha-abajo" if v < 0 else "flecha-arriba" if v > 0 else "igual"


def cifras(fichas, clase="", id=None):
    """Fila de fichas (auto-fit, mínimo 230 px por ficha). Con clase «cifras--3», tres columnas fijas."""
    return html.Div(_lista(fichas), className=_clases("cifras", clase), **_kw(id=id))


def cifra(etiqueta, valor, unidad=None, variacion=None, tono=None, nota=None, *, flecha=None, chip=None, tenue=False, mini=False,
          clase="", id=None):
    """Ficha de cifra: etiqueta → valor con unidad → variación de una línea (flecha + texto; color de estado solo si el
    sentido es inequívoco: tono «bien» o «mal») → nota. `valor` llega ya formateado (ui.num). `flecha` es el número cuyo
    signo elige la flecha. `chip` va junto a la etiqueta («Modelo propuesto»). `tenue=True` apaga el valor (--tinta-tenue):
    la cifra de comparación. `mini=True`, el valor a 32 px. Sin punto de color ni ruta de archivo."""
    cab = [html.Span(etiqueta, className="cifra__etiqueta")]
    if chip is not None:
        cab.append(chip)
    valor_ = [html.Span(valor, className="cifra__numero")] + ([html.Span(unidad, className="cifra__unidad")] if unidad else [])
    hijos = [html.Div(cab, className="cifra__cab"), html.P(valor_, className="cifra__valor")]
    if variacion is not None:
        tono = "mal" if tono == "critico" else tono
        ic = icono_cambio(flecha) if flecha is not None else ({"bien": "bien", "mal": "alerta"}.get(tono) if tono else None)
        hijos.append(html.P(([_glifo(ic, "glifo--s")] if ic else []) + [html.Span(variacion)],
                            className=_clases("cifra__variacion", tono and f"cifra__variacion--{tono}")))
    if nota is not None:
        hijos.append(html.P(nota, className="cifra__nota"))
    return html.Div(hijos, className=_clases("cifra", tenue and "cifra--tenue", mini and "cifra--mini", clase), **_kw(id=id))


def saldo(titulo, valor, definicion=None, unidad="", d=0, palabras=("a favor", "en contra"), extremo=None):
    """Una cifra con signo que acota a la protagonista: título → valor con su signo (número; se formatea con `d`
    decimales) y su unidad → chip → definición de una línea. El chip dice el sentido con icono y palabra: `palabras[0]`
    si es positiva, `palabras[1]` si es negativa y «sin diferencia» si es cero. Con `extremo` («límite inferior», «límite
    superior») la cifra es el borde de un rango y no un resultado, así que el chip va neutro y dice solo qué borde es:
    un piso en rojo se leería como una pérdida segura."""
    if extremo:
        estado = chip(extremo)
    else:
        estado = chip(palabras[0], "bien") if valor > 0 else chip(palabras[1], "critico") if valor < 0 else chip("sin diferencia")
    hijos = [html.P(titulo, className="saldo__titulo"),
             html.P([html.Span(con_signo(valor, d))] + ([html.Span(unidad, className="saldo__unidad")] if unidad else []),
                    className="saldo__valor"),
             estado]
    if definicion:
        hijos.append(html.P(definicion, className="saldo__texto"))
    return html.Div(hijos, className="saldo")


def heroe(etiqueta, valor, unidad, texto, saldos, lado=None, pie=None, id=None):
    """Cifra protagonista (como mucho una por pantalla): etiqueta → valor a --t-hero (el signo del mismo tamaño) con su
    unidad → una línea → las dos cifras que la acotan (ui.saldo) → la columna de la segunda respuesta (a la derecha del
    valor en escritorio, debajo en el teléfono) → el pie. Ese es también el orden del DOM: la cifra grande nunca va sin
    lo que la acota (verificar_estilo.py exige exactamente dos .saldo dentro de la protagonista)."""
    principal = [html.P(etiqueta, className="heroe__etiqueta"),
                 html.P([html.Span(valor, className="heroe__numero"), html.Span(unidad, className="heroe__unidad")], className="heroe__valor")]
    for t in _lista(texto):
        principal.append(html.P(t, className="heroe__texto") if isinstance(t, str) else t)
    partes = [html.Div(principal, className="heroe__principal"), html.Div(_lista(saldos), className="saldos")]
    if lado is not None:
        partes.append(html.Div(_lista(lado), className="heroe__lado"))
    if pie is not None:
        partes.append(html.Div(_lista(pie), className="heroe__pie"))
    return html.Div(partes, className="heroe cifra--hero", **_kw(id=id))


def antes_despues(etiqueta, hoy, plan, cambio=None, nota=None, plan_rotulo="Después", hoy_rotulo="Antes", clase=""):
    """Ficha antes → después: etiqueta (qué, unidad, periodo) → el valor de antes (`hoy`, en --tinta-2, con el rótulo
    `hoy_rotulo`) → el de después (`plan`, con `plan_rotulo`) → chip de cambio → nota. Los valores llegan ya formateados.
    Para niveles comparados: ui.antes_despues("AUC en prueba", "0,7500", "0,8289", hoy_rotulo="Entrega 2", plan_rotulo="EBM")."""
    filas = html.Dl([
        html.Div([html.Dt(hoy_rotulo), html.Dd(hoy, className="cambio__hoy")], className="cambio__fila"),
        html.Div([html.Dt(plan_rotulo), html.Dd(plan, className="cambio__plan")], className="cambio__fila"),
    ], className="cambio__pares")
    hijos = [html.P(etiqueta, className="cambio__etiqueta"), filas]
    if cambio is not None:
        hijos.append(html.Div(cambio, className="cambio__chip"))
    if nota is not None:
        hijos.append(html.P(nota, className="cambio__nota"))
    return html.Div(hijos, className=_clases("cambio", clase))


def cambios(fichas, clase=""):
    """Rejilla de fichas antes → después (auto-fit, minmax(min(100%, 280px), 1fr))."""
    return html.Div(_lista(fichas), className=_clases("cambios", clase))


# --- Chips y avisos ---------------------------------------------------------------------------------------------------
_ICONO_ESTADO = {"bien": "bien", "atencion": "alerta", "critico": "alerta"}


def chip(texto, tipo=None, icono=None):
    """Chip en píldora blanca: neutro (sin tipo o «neutro») o de estado («bien», «atencion», «critico»: siempre con icono
    y palabra) o «acento» («Modelo propuesto»). Con `icono`, cualquier glifo en un chip neutro. Nunca para marcar «mejor»
    (eso lo hace ui.tabla)."""
    tipo = None if tipo == "neutro" else tipo
    if icono is None and isinstance(texto, str):
        icono = _ICONO_ESTADO.get(tipo)
    partes = ([_glifo(icono, "glifo--s")] if icono else []) + list(_lista(texto))
    return html.Span(partes, className=_clases("chip", tipo and f"chip--{tipo}"))


def triangulo(valor, tono=None):
    """▲ / ▼ / = según el signo de `valor` (un triángulo dibujado, sin texto): verde o rojo solo con `tono` «bien» o
    «critico» (cuando el sentido es inequívoco); si no, gris."""
    direccion = "arriba" if valor > 0 else "abajo" if valor < 0 else "igual"
    tono = {"mal": "critico"}.get(tono, tono)
    return html.Span(className=_clases("triangulo", f"triangulo--{direccion}", tono in ("bien", "critico") and f"triangulo--{tono}"),
                     **{"aria-hidden": "true"})


def chip_cambio(valor, texto, tono=None):
    """Chip de una diferencia (el «▲ 15%» de la referencia): triángulo según el signo de `valor` + el texto con su signo.
    `tono` («bien» | «critico») solo cuando el sentido es inequívoco; una diferencia que no es ni buena ni mala va sin tono."""
    tono = None if tono in (None, "neutro") else tono
    return html.Span([triangulo(valor, tono), html.Span(texto)], className="chip")


def chip_valor(rotulo, valor, id=None):
    """Chip de la referencia «Peak: Wed»: rótulo en gris y valor en tinta. Uso: ui.chip_valor("Umbral:", ui.num(u, 2))."""
    return html.Span([html.Span(rotulo, className="chip__rotulo"), html.Span(valor)], className="chip", **_kw(id=id))


def barra_rayas(etiqueta, valor, fraccion, rol="modelo", id=None):
    """Barra de progreso rayada de la referencia (rótulo y cifra encima de la barra): rótulo gris a la izquierda, valor a
    la derecha (ya formateado) y la barra con rayas del color del rol hasta `fraccion` (de 0 a 1; lo que se salga se
    acota). Para partes de un total, una probabilidad o la importancia de una variable; nunca para una cifra con signo."""
    p = max(0.0, min(1.0, float(fraccion or 0)))
    return html.Div([html.Div([html.Span(etiqueta, className="rayas__etiqueta"), html.Span(valor, className="rayas__valor")], className="rayas__cab"),
                     html.Div(html.Div(className="rayas__relleno", style={"--p": f"{p * 100:.1f}%", "--c": VAR[_rol(rol)]}),
                              className="rayas__pista", role="img", **{"aria-label": f"{etiqueta}: {valor}"})],
                    className="rayas", **_kw(id=id))


def puntos(cuentas, foco=None, rol="modelo", etiqueta=None, id=None):
    """Matriz de puntos de la referencia («Transactions»): una columna por categoría con `cuentas[i]` puntos (enteros
    pequeños, p. ej. ya escalados de 0 a 5); la columna `foco` (índice) va sólida y las demás en el tinte. `etiqueta`
    (aria-label) dice qué representa con sus cifras."""
    cols = [html.Span([html.Span(className="puntos__p") for _ in range(int(n))],
                      className=_clases("puntos__col", i == foco and "puntos__col--foco")) for i, n in enumerate(cuentas)]
    estilo = {"--c": VAR[_rol(rol)], "--c-tinte": VAR[_rol_tinte(rol)]}
    return html.Div(cols, className="puntos", style=estilo, role="img", **{"aria-label": etiqueta or ""}, **_kw(id=id))


def tarjeta_insight(valor, titulo, texto=None, chip_="Lectura", id=None):
    """Tarjeta «Insights» de la referencia: degradado naranja → azul con grano, chip arriba, número grande blanco, una
    frase blanca y un texto. Una por pantalla como mucho; el número sale de datos/ (ya formateado)."""
    cuerpo = [html.P(valor, className="insight__valor"), html.P(titulo, className="insight__titulo")]
    if texto is not None:
        cuerpo.append(html.P(texto, className="insight__texto"))
    return html.Div([html.Span([_glifo("info", "glifo--s"), chip_], className="insight__chip"), html.Div(cuerpo, className="insight__cuerpo")],
                    className="insight", **_kw(id=id))


_AVISO_ICONO = {None: "info", "canje": "canje", "atencion": "alerta", "critico": "alerta", "bien": "bien"}


def aviso(texto, tipo=None, titulo=None, rol=None, id=None):
    """Aviso: icono de 22 px + título y texto a --t-m, sobre --superficie (o blanco anidado); el color solo en el icono y
    el título. Tipos: None (nota), «canje», «atencion», «critico» (el único con fondo teñido y role=alert), «bien».
    `texto` es una cadena, una lista de párrafos (cadenas) o un componente. `rol="status"` para los que llegan por callback."""
    if tipo == "nota":
        tipo = None
    cuerpo = [html.P(titulo, className="aviso__titulo")] if titulo is not None else []
    if isinstance(texto, str):
        cuerpo.append(html.P(texto))
    elif isinstance(texto, (list, tuple)) and all(isinstance(t, str) for t in texto):
        cuerpo += [html.P(t) for t in texto]
    else:
        cuerpo.append(html.Div(_lista(texto)))
    return html.Div([_glifo(_AVISO_ICONO.get(tipo, "info"), "aviso__icono"), html.Div(cuerpo, className="aviso__cuerpo")],
                    className=_clases("aviso", tipo and f"aviso--{tipo}"), role=rol or ("alert" if tipo == "critico" else "note"),
                    **_kw(id=id))


# --- Matriz de confusión y ficha --------------------------------------------------------------------------------------
# Rótulos por defecto de ui.matriz: filas = lo que pasó; columnas = lo que se hizo; nombre de cada celda en el orden
# (vp, fp, fn, vn)
MATRIZ_ROTULOS = {"filas": ("Riesgo alto real", "Riesgo bajo real"), "columnas": ("Contactado", "No contactado"),
                  "celdas": ("acierto", "falsa alarma", "se escapa", "bien descartado")}


def matriz(vp, fp, fn, vn, rotulos=None, id=None):
    """Matriz de confusión de dos por dos: las filas son lo que pasó y las columnas, lo que se hizo. Los cuatro números
    llegan ya formateados (ui.num): `vp` (fila 1, columna 1), `fn` (fila 1, columna 2), `fp` (fila 2, columna 1) y `vn`
    (fila 2, columna 2). Cada celda lleva su nombre debajo y los dos aciertos van sobre el tinte del azul. `rotulos` cambia
    cualquiera de las tres partes de ui.MATRIZ_ROTULOS: {"filas": (…, …), "columnas": (…, …), "celdas": (vp, fp, fn, vn)}.
    ui.matriz(ui.num(150), ui.num(60), ui.num(90), ui.num(700))."""
    sobran = set(rotulos or {}) - set(MATRIZ_ROTULOS)
    if sobran:
        raise KeyError(f"rótulos desconocidos: {sorted(sobran)} (se admiten: {', '.join(MATRIZ_ROTULOS)})")
    r = dict(MATRIZ_ROTULOS, **(rotulos or {}))
    (fila_si, fila_no), (col_si, col_no) = r["filas"], r["columnas"]
    n_vp, n_fp, n_fn, n_vn = r["celdas"]

    def celda(valor, nombre, acierto):
        return html.Div([html.Span(valor, className="matriz__numero"), html.Span(nombre, className="matriz__nombre")],
                        className=_clases("matriz__celda", acierto and "matriz__celda--acierto"))

    hijos = [html.Span(className="matriz__rotulo"),
             html.Span(col_si, className="matriz__rotulo matriz__rotulo--col"), html.Span(col_no, className="matriz__rotulo matriz__rotulo--col"),
             html.Span(fila_si, className="matriz__rotulo"), celda(vp, n_vp, True), celda(fn, n_fn, False),
             html.Span(fila_no, className="matriz__rotulo"), celda(fp, n_fp, False), celda(vn, n_vn, True)]
    lectura = [(fila_si, col_si, vp, n_vp), (fila_si, col_no, fn, n_fn), (fila_no, col_si, fp, n_fp), (fila_no, col_no, vn, n_vn)]
    if all(isinstance(v, (str, int, float)) for _f, _c, v, _n in lectura):
        accesible = {"role": "img", "aria-label": "Matriz de confusión. " + " ".join(f"{f}, {c}: {v} ({n})." for f, c, v, n in lectura)}
    else:
        accesible = {"role": "group", "aria-label": "Matriz de confusión"}
    return html.Div(hijos, className="matriz", **accesible, **_kw(id=id))


def ficha(pares, id=None):
    """Ficha de pares rótulo–valor (una lista de definiciones con una línea fina entre filas): los datos de un cliente.
    `pares` es una lista de (rótulo, valor) o un diccionario; los valores llegan ya formateados.
    ui.ficha([("Antigüedad", "14 meses"), ("Productos activos", "2")])."""
    pares = list(pares.items()) if isinstance(pares, dict) else list(pares)
    return html.Dl([html.Div([html.Dt(k), html.Dd(v)], className="ficha__fila") for k, v in pares], className="ficha", **_kw(id=id))


# --- Leyenda, tabla y gráfico -----------------------------------------------------------------------------------------
# Orden fijo de la leyenda: modelo (la EBM), rival, base, real y contexto; lo que no sea una serie va después
_ORDEN_LEYENDA = {r: i for i, r in enumerate(T.SERIES)}


def leyenda(items, ordenar=True, id=None):
    """Leyenda HTML: [(texto, rol, forma)], una muestra por cada par rol-forma que se ve en el gráfico (también cuando una
    sola traza usa dos colores). Formas: barra, rayada, linea, gruesa, trazos, punteada, escalon, area, punto. La muestra
    lleva --c: var(--serie-…), nunca un color escrito. Se ordena siempre igual (modelo, rival, base, real, contexto) salvo
    con ordenar=False. Un gráfico de un solo rol no lleva leyenda: lo nombra el título."""
    filas = []
    for i, (texto, rol, forma) in enumerate(items):
        rol = _rol(rol)
        forma = {"puntos": "punteada", "foco": "barra", "escalon_rayado": "escalon"}.get(forma, forma)
        filas.append((_ORDEN_LEYENDA.get(rol, 99) if ordenar else 0, i, texto, rol, forma))
    filas.sort(key=lambda f: (f[0], f[1]))
    return html.Ul([html.Li([html.Span(className=f"leyenda__muestra leyenda__muestra--{forma}", style={"--c": VAR[rol]}), html.Span(texto)],
                            className="leyenda__item") for _o, _i, texto, rol, forma in filas],
                   className="leyenda", **_kw(id=id))


def muestra(rol):
    """Muestra cuadrada de 10 px del color de un rol (va siempre al lado de su nombre)."""
    return html.Span(className="tabla__muestra", style={"--c": VAR[_rol(rol)]}, **{"aria-hidden": "true"})


def clave(texto, rol):
    """Nombre con su muestra de color (los modelos en las tablas)."""
    return html.Span([muestra(rol), html.Span(texto)], className="tabla__clave")


def tabla(encabezados, filas, total=None, mejor=None, compacta=False, muestras=None, ocultar_movil=(), num=None, alta=False, id=None,
          mejor_texto="mejor"):
    """Tabla en su marco blanco (radio 26; 16 anidada; desplazamiento propio). La primera columna es el rótulo de la fila;
    las demás son números (.num, a la derecha y tabulares) salvo que `num` diga otra cosa. Las celdas llegan ya formateadas.
    `total`: fila de total en 600. `mejor`: celdas (fila, columna) con el mejor valor; van en negrita y en tinta, con la
    palabra «mejor» debajo (`mejor_texto=None` deja solo la negrita): nunca en el color de una serie, porque el mejor puede
    ser un rival. `muestras`: rol de cada fila (pone su cuadrado de color). `ocultar_movil`: columnas que se ocultan en
    ≤ 600 px. `alta`: alto máximo con desplazamiento propio."""
    n = len(encabezados)
    es_num = list(num) if num is not None else [j > 0 for j in range(n)]
    mejor = set(mejor or ())
    ocultar = set(ocultar_movil or ())

    def cls(j, *extra):
        return _clases("num" if es_num[j] else "", "solo-ancho" if j in ocultar else "", *extra) or None

    def celda(i, j, v):
        if (i, j) not in mejor:
            return html.Td(v, **_kw(className=cls(j)))
        marca = [html.Br(), html.Span(mejor_texto, className="tabla__mejor-txt")] if mejor_texto else []
        return html.Td([html.Span(v)] + marca, className=cls(j, "tabla__mejor"))

    cab = html.Thead(html.Tr([html.Th(h, scope="col", **_kw(className=cls(j))) for j, h in enumerate(encabezados)]))
    cuerpo = []
    for i, f in enumerate(filas):
        rotulo = f[0]
        if muestras and muestras[i]:
            rotulo = clave(f[0], muestras[i])
        cuerpo.append(html.Tr([html.Th(rotulo, scope="row", **_kw(className=cls(0)))] + [celda(i, j, v) for j, v in enumerate(f[1:], 1)]))
    if total is not None:
        cuerpo.append(html.Tr([html.Th(total[0], scope="row")] + [html.Td(v, **_kw(className=cls(j))) for j, v in enumerate(total[1:], 1)],
                              className="tabla__fila-total"))
    t = html.Table([cab, html.Tbody(cuerpo)], className=_clases("tabla", compacta and "tabla--compacta"))
    return html.Div(t, className=_clases("tabla-marco", alta and "tabla-marco--alta"), **_kw(id=id))


def desplegable(resumen, cuerpo, llano=False, pequeno=False, abierto=False, clase="", id=None):
    """Desplegable con chevron que gira: «Cómo se calcula», «Qué mide cada cifra». La variante pequeña es «Ver los datos»
    (icono de tabla, --t-s). Nunca para esconder un canje o un límite: esos van en un aviso a la vista."""
    titulo = [_glifo("chevron-derecha", "desplegable__chevron glifo--s")] + ([_glifo("tabla", "glifo--s")] if pequeno else []) + [html.Span(resumen)]
    return html.Details([html.Summary(titulo, className="desplegable__resumen"), html.Div(_lista(cuerpo), className="desplegable__cuerpo")],
                        open=abierto, className=_clases("desplegable", llano and "desplegable--llano", pequeno and "desplegable--pequeno", clase),
                        **_kw(id=id))


def grafico(id_o_fig, titulo=None, subtitulo=None, etiqueta=None, tabla=None, leyenda=None, nota=None, alto="normal", id=None,
            resumen_tabla="Ver los datos", fuentes=None):
    """Contenedor de gráfico (figure.grafico): título-mensaje (h3) → subtítulo (qué es cada color, unidad, periodo) → leyenda
    HTML → lienzo (role=img con aria-label: el mensaje y las cifras clave) → nota. El botón «…» abre la tabla gemela («Ver
    los datos») y las `fuentes`; se dibuja siempre, así que la tabla va siempre. Sin controles dentro: los de una sola
    figura van en la fila de su sección. `alto`: «normal» (320), «bajo» (240), «alto» (420).
    `id_o_fig` es el id del dcc.Graph (la figura la pone un callback) o una figura fija. Con un id, los callbacks pueden
    escribir en «<id>-titulo», «<id>-sub», «<id>-leyenda», «<id>-tabla», «<id>-nota» (children) y en «<id>-lienzo»
    (aria-label). `tabla` es un componente o (encabezados, filas); `leyenda`, un componente o [(texto, rol, forma)]."""
    gid = id_o_fig if isinstance(id_o_fig, str) else id
    sub = (lambda s: f"{gid}-{s}" if gid else None)
    props = dict(config=GRAF_CONFIG, responsive=True, className="grafico__plotly", style={"height": "100%"})
    graf = dcc.Graph(id=gid, **props) if isinstance(id_o_fig, str) else dcc.Graph(figure=id_o_fig, **props, **_kw(id=gid))
    lienzo = html.Div(graf, className=_clases("grafico__lienzo", alto in ("bajo", "alto") and f"grafico__lienzo--{alto}"),
                      role="img", **{"aria-label": etiqueta or ""}, **_kw(id=sub("lienzo")))
    if isinstance(leyenda, (list, tuple)):
        leyenda = globals()["leyenda"](leyenda)
    if isinstance(tabla, tuple) and len(tabla) == 2:
        tabla = globals()["tabla"](tabla[0], tabla[1], compacta=True)
    textos = html.Div([html.H3(titulo, className="grafico__titulo", **_kw(id=sub("titulo"))),
                       html.P(subtitulo, className="grafico__sub", **_kw(id=sub("sub")))], className="grafico__textos")
    boton_mas = mas(html.Div(tabla, **_kw(id=sub("tabla"))), fuentes, resumen_tabla, clase="grafico__tabla")
    partes = [html.Div([textos, boton_mas], className="grafico__cab")]
    if leyenda is not None or gid:
        partes.append(html.Div(leyenda, **_kw(id=sub("leyenda"))))
    partes.append(lienzo)
    if nota is not None or gid:
        partes.append(html.P(nota, className="grafico__pie", **_kw(id=sub("nota"))))
    return html.Figure(partes, className="grafico", **_kw(id=sub("figura")))


# --- Pie de fuentes, destinos, botones y controles --------------------------------------------------------------------
def fuentes(rutas, rotulo="Fuentes:"):
    """Pie de fuentes de una sección: en escritorio, una línea (máx. 2) con icono, «Fuentes:» y las rutas relativas, con el
    texto completo en title; en el teléfono, «Fuentes (n)» plegado."""
    rutas = [r for r in _lista(rutas) if r]
    texto = " · ".join(rutas)
    escritorio = html.P([_glifo("fuente", "glifo--s"), html.Span(rotulo, className="fuentes__rotulo"), html.Span(texto)],
                        className="fuentes fuentes--escritorio", title=texto)
    movil = html.Details([html.Summary([_glifo("fuente", "glifo--s"), f"Fuentes ({len(rutas)})"]),
                          html.Ul([html.Li(r) for r in rutas], className="fuentes__lista")], className="fuentes fuentes--movil")
    return html.Div([escritorio, movil], className="fuentes-pie")


def destino(href, pantalla, titulo, texto, externo=False, icono=None):
    """Enlace a otra pantalla o a un documento: cuadro con el icono de la pantalla + h3 → texto → «Abrir ›». Con
    `externo=True` abre en otra pestaña."""
    hijos = [html.Div([cuadro(pantalla, icono), html.H3(titulo, className="destino__titulo")], className="destino__cab"),
             html.P(texto, className="destino__texto"),
             html.Span(["Abrir", _glifo("externo" if externo else "chevron-derecha", "glifo--s")], className="destino__ir")]
    if externo:
        return html.A(hijos, href=href, target="_blank", rel="noopener", className="destino")
    return dcc.Link(hijos, href=href, className="destino")


def destinos(hijos):
    return html.Div(_lista(hijos), className="destinos")


def boton(texto, href=None, tipo="secundario", icono=None, id=None, pequeno=False, externo=False, titulo=None):
    """Botón en píldora (40 px; 32 el pequeño): «primario» (uno por bloque), «secundario» o «fantasma». Con href es un enlace."""
    partes = ([_glifo(icono, "glifo--m")] if icono else []) + [texto]
    cls = _clases("boton", f"boton--{tipo}", pequeno and "boton--pequeno")
    if href:
        extra = {"target": "_blank", "rel": "noopener"} if externo else {}
        return html.A(partes, href=href, className=cls, **extra, **_kw(id=id, title=titulo))
    return html.Button(partes, className=cls, type="button", **_kw(id=id, title=titulo))


def boton_redondo(icono, href=None, titulo="", id=None, externo=False):
    """Botón redondo de borde fino (44 px) con un glifo, como los de la barra de la referencia. Con href es un enlace;
    `titulo` es su nombre (title y aria-label)."""
    hijos = [_glifo(icono)]
    kw = {"aria-label": titulo, "title": titulo}
    if href:
        extra = {"target": "_blank", "rel": "noopener"} if externo else {}
        return html.A(hijos, href=href, className="boton-redondo", **kw, **extra, **_kw(id=id))
    return html.Button(hijos, type="button", className="boton-redondo", **kw, **_kw(id=id))


def segmentado(id, opciones, valor, pequeno=False, clase="", persistir=True):
    """Segmentado (radios de verdad) para 2 a 6 opciones cortas y excluyentes; la elegida, en blanco con sombra suave. Sin
    puntos de color. `opciones`: cadenas o {"label", "value"}. Se escucha su propiedad `value`."""
    ops = []
    for o in opciones:
        o = dict(o) if isinstance(o, dict) else {"label": o, "value": o}
        if isinstance(o["label"], str):
            o["label"] = html.Span(o["label"], className="segmentado__etiqueta")
        ops.append(o)
    return dcc.RadioItems(id=id, options=ops, value=valor, inline=True, className=_clases("segmentado", pequeno and "segmentado--pequeno", clase),
                          labelClassName="segmentado__opcion", inputClassName="segmentado__input", **_persistencia(persistir))


def control(rotulo, componente, ayuda=None, clase=""):
    """Control con su rótulo delante, en línea («Modelo:», «Métrica:»); la ayuda, si hay, debajo."""
    hijos = [html.Label(rotulo, className="control-rotulado__rotulo"), componente]
    if ayuda is not None:
        hijos.append(html.P(ayuda, className="control-rotulado__ayuda"))
    return html.Div(hijos, className=_clases("control-rotulado", clase))


def casilla(id, opciones, valor=None, lista=False, persistir=True):
    """Casillas nativas de 16 px en el acento (dcc.Checklist). `lista=True` las pone una debajo de otra."""
    return dcc.Checklist(id=id, options=opciones, value=list(valor or []), className=_clases("casilla", lista and "casilla--lista"),
                         **_persistencia(persistir))


def deslizador(id, rotulo, minimo, maximo, paso, valor, marcas=None, ayuda=None, persistir=True, valor_inicial_texto=None,
               updatemode="mouseup"):
    """Deslizador de un panel de parámetros: rótulo y valor en una línea → pista → marcas a --t-xs → ayuda. El valor se
    muestra en el Span «<id>-valor»: nace con `valor_inicial_texto` (ya formateado, p. ej. ui.num(valor, 2)) y lo mantiene
    al día un callback (Input(id, "drag_value") + State(id, "value") → Output(f"{id}-valor", "children")). `marcas` es
    {valor: texto en es-CO}. `updatemode`: «mouseup» (recalcula al soltar) o «drag» (mientras se arrastra)."""
    hijos = [html.Div([html.Label(rotulo, htmlFor=id, className="deslizador__rotulo"),
                       html.Span(valor_inicial_texto, id=f"{id}-valor", className="deslizador__valor")], className="deslizador__cab"),
             dcc.Slider(id=id, min=minimo, max=maximo, step=paso, value=valor, marks=marcas, updatemode=updatemode,
                        tooltip={"placement": "bottom", "always_visible": False}, allow_direct_input=False, className="deslizador",
                        **_persistencia(persistir))]
    if ayuda is not None:
        hijos.append(html.P(ayuda, className="deslizador__ayuda"))
    return html.Div(hijos, className="deslizador-bloque")


def campo_numero(id, valor, minimo, maximo, paso, rotulo=None, ayuda=None, persistir=True):
    """Campo numérico en píldora (dcc.Input de tipo número, con sus botones de paso): rótulo encima → campo → ayuda. Avisa
    del cambio al salir del campo o con Intro (debounce), no con cada tecla. Se escucha su propiedad `value` (None si
    el campo está vacío o fuera de rango). ui.campo_numero("cl-edad", 35, 18, 90, 1, "Edad", "En años cumplidos.")."""
    hijos = [html.Label(rotulo, htmlFor=id, className="campo__rotulo")] if rotulo is not None else []
    hijos.append(dcc.Input(id=id, type="number", value=valor, min=minimo, max=maximo, step=paso, debounce=True, className="campo",
                           **_persistencia(persistir)))
    if ayuda is not None:
        hijos.append(html.P(ayuda, className="campo__ayuda"))
    return html.Div(hijos, className="campo-bloque")


# Textos del desplegable de Dash, en español (propiedad `labels` de dcc.Dropdown en Dash 4)
_TEXTOS_SELECTOR = {"select_all": "Seleccionar todo", "deselect_all": "Quitar todo", "selected_count": "{num_selected} elegidos",
                    "search": "Buscar", "clear_search": "Borrar la búsqueda", "clear_selection": "Quitar la selección",
                    "no_options_found": "Sin resultados"}


def selector(id, opciones, valor, rotulo=None, buscar=False, persistir=True, placeholder=None):
    """Desplegable de una sola elección (dcc.Dropdown, siempre con un valor elegido), para listas largas donde no cabe un
    segmentado: un cliente, una variable. `opciones`: cadenas o {"label", "value"}. `buscar=True` añade el campo de
    búsqueda. Con `rotulo`, el rótulo va encima (como ui.campo_numero); sin él devuelve solo el desplegable, que se puede
    poner en línea con ui.control("Cliente:", ui.selector(…)). Se escucha su propiedad `value`.
    ui.selector("cl-cliente", [{"label": "Cliente 104", "value": 104}], 104, "Cliente", buscar=True)."""
    d = dcc.Dropdown(id=id, options=opciones, value=valor, clearable=False, searchable=buscar, className="desplegable-campo",
                     labels=_TEXTOS_SELECTOR, **_kw(placeholder=placeholder), **_persistencia(persistir))
    if rotulo is None:
        return d
    return html.Div([html.Label(rotulo, htmlFor=id, className="campo__rotulo"), d], className="campo-bloque")
