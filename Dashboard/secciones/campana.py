"""Campaña · ¿A quién contactar si no alcanza para todos?

Segunda pantalla en vivo. Una campaña con capacidad limitada contacta a los clientes a los que el modelo da mayor puntaje:
el deslizador fija la capacidad y todo se recalcula en el momento con el puntaje guardado de los modelos finales
(D.puntajes(): los ocho del Entregable 3 y, si la corrección ya los dejó, el modelo original y LightGBM) y la regla del
informe (M.resumen_capacidad, con su desempate): las cifras de la EBM, la curva de captura con el punto elegido, la matriz
de confusión, todos los modelos con esa capacidad y quién queda fuera por cohorte de edad. La vista es del grupo de la EBM:
ella es la serie en foco y cualquier otro modelo, también el original, se puede elegir como comparación. Es la
pantalla de «significancia estadística no es relevancia práctica»: con la capacidad del informe la EBM y la regresión
logística alcanzan casi a los mismos clientes; la diferencia aparece con más capacidad.

El deslizador y lo que mueve se ven juntos: la tarjeta de control (deslizador, modelo de comparación y cifras) queda pegada
bajo la barra mientras se recorren la curva y la matriz, y la curva usa el lienzo bajo cuando la ventana es baja.

Para que el deslizador se sienta instantáneo:
  · lo que no depende de la capacidad se guarda por modelo y cada campaña se calcula una vez por modelo y capacidad (lru_cache);
  · mientras se arrastra (`drag_value`) solo se pone al día la parte interactiva (cifras, curva y matriz); las tablas y el
    gráfico por cohorte, que quedan más abajo, se rehacen al soltar (`value`);
  · al navegador solo viajan textos y lo que se mueve en la curva (dash.Patch). Rehacer componentes es lo caro en el
    navegador (medido: 1 a 2 ms por componente), así que las cifras son hojas con id (`_hoja`) dentro de las piezas de ui,
    y la figura completa solo se envía cuando cambia el modelo de comparación.
"""
from functools import lru_cache

import dash
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, callback, clientside_callback, ctx, dcc, html

import app
import datos_app as D
import formato as F
import motor as M
import ui


PRE = "ca"
PANTALLA = "campana"
PREGUNTA = app.pregunta(PANTALLA)

FOCO = "ebm"                                                # el modelo en foco de la vista (el benchmark interpretable)
# los clasificadores con puntaje guardado en la prueba, en el orden de las tablas
MODELOS = [m for m in D.ORDEN_MODELOS["clasificacion"] if m in D.puntajes().columns]
REFERENCIAS = [m for m in MODELOS if m != FOCO]             # los modelos con los que se puede comparar la EBM
RIVAL_INICIAL = "logistica"                                 # la referencia con la que el informe discute el punto de operación
RIVAL_COHORTES = "xgboost"                                  # el modelo con el que el informe compara los errores por cohorte
N_COHORTES = 3                                              # la edad solo toma tres cohortes en el conjunto de datos
NUMERO = {1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez"}

# La capacidad de la campaña se maneja en puntos porcentuales de los clientes de prueba (lo que mueve el deslizador)
CAP_MIN, CAP_MAX, CAP_PASO = 5, 100, 1
CAP_INFORME = round(100 * M.CAPACIDAD_INFORME)              # el punto de operación del informe
CAP_AMPLIA = 50                                             # la otra capacidad con la que el informe muestra la diferencia
MARCAS = (CAP_MIN, CAP_INFORME, CAP_AMPLIA, 75, CAP_MAX)    # marcas del deslizador
PASO_EJE = 25                                               # marcas de los dos ejes de la curva
PASO_TABLA = 5                                              # la tabla gemela de la curva va de cinco en cinco puntos
CAP_GIRO = 70                                               # desde aquí el rótulo de la capacidad va a la izquierda de su línea
HOLGURA_EJE = 4                                             # puntos por encima del 100 % para que la curva no toque el borde
UMBRAL_CLASE = 0.5                                          # umbral de probabilidad con el que cada modelo predice su clase
DEC_CORTE = 4                                               # decimales del corte de la campaña (los mismos que en «Cliente»)
ALTO_BAJO = 240                                             # px del lienzo «bajo»: con él ui.ancho_barra no pasa de 24 px
HOLGURA_BARRAS = 1.1                                        # el eje de las barras deja sitio al valor de la punta
ALTO_LISTA = 320                                            # px de la lista del desplegable: caben los siete modelos sin desplazarse
ALTO_VENTANA_AMPLIA = 840                                   # px de alto de ventana desde los que la curva cabe con el lienzo normal
CLASE_LIENZO = "grafico__lienzo"                            # clases del lienzo de un gráfico (estilos.css): normal y bajo
CLASE_LIENZO_BAJO = f"{CLASE_LIENZO} {CLASE_LIENZO}--bajo"

# Posición de lo que se mueve dentro de la curva (lo fija _figura_curva; lo usa _parche_curva)
I_MARCA_RIVAL, I_MARCA_EBM = 3, 4                           # trazas de los dos puntos
I_ROTULO = 1                                                # anotación «25 % de capacidad» (la 0 es la unidad del eje)

# Hojas: los textos que cambian los callbacks, por su clave (el id es PRE-clave), en el orden en que se devuelven
HOJAS_CIFRAS = ("f-vp", "f-vp-de", "f-precision", "f-precision-de", "f-elevacion", "f-rival")
HOJAS_MATRIZ = ("matriz-titulo", "m-vp", "m-fp", "m-fn", "m-vn", "m-corte")
CAMPOS_MODELO = ("vp", "captura", "precision", "corte")     # columnas de la tabla de modelos que dependen de la capacidad
HOJAS_MODELOS = tuple(f"t-{m}-{c}" for m in MODELOS for c in CAMPOS_MODELO)
HOJAS_COHORTES = tuple(f"c-{i}-{quien}" for i in range(N_COHORTES + 1) for quien in ("ebm", "rival"))   # la última fila es el total

FUENTE_PUNTAJES = D.fuente("puntajes_prueba.parquet")
FUENTE_CLIENTES = D.fuente("clientes_prueba.parquet")
FUENTE_PRUEBA = f"{D.fuente('resultados/clasificacion_prueba.csv')} · {D.fuente_original('lideres_clasificacion.csv')}"

RAYA_EDAD = "–"                                        # la raya de las cohortes («59–60»)
UNION = "⁠"                                            # impide el salto de línea después de esa raya


# --- Textos y piezas -----------------------------------------------------------------------------------------------------
def _pct(puntos, d=0):
    """Puntos porcentuales como texto: 25 → «25 %»."""
    return F.pct(puntos / 100, d)


def _edad(nivel):
    """Una cohorte dentro de una frase: «59–60 años», sin partirse entre líneas."""
    return f"{str(nivel).replace(RAYA_EDAD, RAYA_EDAD + UNION)}{F.FIJO}años"


def _capacidad(valor):
    """La capacidad del deslizador como entero entre CAP_MIN y CAP_MAX; sin valor o con uno ilegible, la del informe."""
    try:
        v = int(round(float(valor)))
    except (TypeError, ValueError, OverflowError):
        return CAP_INFORME
    return min(max(v, CAP_MIN), CAP_MAX)


def _referencia(valor):
    return valor if valor in REFERENCIAS else RIVAL_INICIAL


def _hoja(clave, texto):
    """Un texto que un callback cambia sin rehacer lo que lo rodea: un Span con id (PRE-clave) que se pasa a las piezas de
    ui en el lugar de la cifra."""
    return html.Span(texto, id=f"{PRE}-{clave}")


def _hojas(claves, textos):
    return {clave: _hoja(clave, texto) for clave, texto in zip(claves, textos)}


def _selector(id, opciones, valor, rotulo, ayuda):
    """ui.selector con toda la lista a la vista (por omisión la lista del desplegable de Dash mide 200 px y dejaría dos de
    los siete modelos bajo su barra de desplazamiento) y con una línea de ayuda debajo, como la de ui.campo_numero."""
    bloque = ui.selector(id, opciones, valor, rotulo=rotulo)
    bloque.children[-1].maxHeight = ALTO_LISTA
    bloque.children.append(html.P(ayuda, className="campo__ayuda"))
    return bloque


# --- Datos: la prueba, la campaña de cada modelo y su curva (todo con caché y de solo lectura) -----------------------
@lru_cache(maxsize=1)
def _prueba():
    """Los clientes de prueba como los ve la campaña: la etiqueta (1 = riesgo alto) y la cohorte de edad de cada uno,
    alineadas por «fila» con D.puntajes(), y las cohortes con su número de clientes y de clientes de riesgo alto."""
    puntajes = D.puntajes()
    y = puntajes["y"].to_numpy(dtype=int)
    cohorte = M.clientes()["cohorte"].reindex(puntajes.index).to_numpy(dtype=object)
    cohortes = tuple({"nivel": n, "clientes": int((cohorte == n).sum()), "positivos": int(((cohorte == n) & (y == 1)).sum())}
                     for n in sorted(set(cohorte)))
    if len(cohortes) != N_COHORTES:
        raise ValueError(f"se esperaban {N_COHORTES} cohortes de edad y hay {len(cohortes)}")
    return y, cohorte, cohortes


@lru_cache(maxsize=None)
def _resumen(modelo, cap):
    """La campaña que contacta al `cap` % de clientes a los que `modelo` da mayor puntaje (sin recalibrar):
    M.resumen_capacidad, la regla del informe con su desempate. El resultado es compartido: no modificarlo."""
    return M.resumen_capacidad(_prueba()[0], D.puntajes()[modelo], cap / 100)


@lru_cache(maxsize=None)
def _fuera(modelo, cap):
    """Clientes de riesgo alto que esa campaña no alcanza en cada cohorte de edad (en el orden de _prueba)."""
    y, cohorte, cohortes = _prueba()
    sin_contactar = (M.punto_operacion(D.puntajes()[modelo], cap / 100) == 0) & (y == 1)
    return tuple(int((sin_contactar & (cohorte == c["nivel"])).sum()) for c in cohortes)


@lru_cache(maxsize=None)
def _curva(modelo):
    """Curva de captura de un modelo de punto en punto porcentual (M.curva_captura): la fila i es la capacidad del i %.
    Compartida: no modificarla."""
    return M.curva_captura(_prueba()[0], D.puntajes()[modelo])


@lru_cache(maxsize=None)
def _textos_curva(modelo):
    """Texto al pasar el puntero por cada punto de la curva de un modelo."""
    c = _curva(modelo)
    return tuple(f"{D.NOMBRE_MODELO[modelo]}: <b>{F.pct(f)}</b> · {F.entero(v)} clientes" for f, v in zip(c["captura"], c["vp"]))


@lru_cache(maxsize=1)
def _auc_prueba():
    t = D.lideres("clasificacion")
    return dict(zip(t["modelo"], t["auc_prueba"]))


def _texto_capacidad(cap):
    return f"{_pct(cap)} · {F.entero(_resumen(FOCO, cap)['contactados'])} clientes"


# --- Tarjeta de control: deslizador, modelo de comparación y cifras de la EBM ----------------------------------------
def _textos_cifras(cap, rival):
    """Los textos de las cifras de la EBM con esa capacidad y lo que alcanza el modelo de comparación, en el orden de
    HOJAS_CIFRAS."""
    e, r = _resumen(FOCO, cap), _resumen(rival, cap)
    dif = e["vp"] - r["vp"]
    if dif:
        frente = f"Alcanza a {F.entero(r['vp'])}: {F.entero(abs(dif))} {'menos' if dif > 0 else 'más'} que la EBM."
    else:
        frente = f"Alcanza a {F.entero(r['vp'])}, igual que la EBM."
    return (F.entero(e["vp"]), f"de {F.entero(e['positivos'])} · {F.pct(e['captura'])}", F.pct(e["precision"]),
            f"de {F.entero(e['contactados'])} contactados", ui.num(e["elevacion"], 2), frente)


def _control(cap, rival):
    """La tarjeta que queda pegada bajo la barra: el deslizador y, debajo, el modelo de comparación (con lo que alcanza) y las
    tres cifras de la EBM, sin marco. Es baja a propósito: debajo tienen que caber la curva y la matriz."""
    h = _hojas(HOJAS_CIFRAS, _textos_cifras(cap, rival))
    marcas = {v: _pct(v) for v in MARCAS}
    marcas[CAP_INFORME] = f"{_pct(CAP_INFORME)} (informe)"
    return ui.tarjeta([html.Div([
        # El valor se confirma al soltar (`value`); mientras se arrastra, la parte interactiva sigue a `drag_value`
        ui.deslizador(f"{PRE}-capacidad", "Capacidad de la campaña", CAP_MIN, CAP_MAX, CAP_PASO, cap, marcas=marcas,
                      valor_inicial_texto=_texto_capacidad(cap)),
        ui.cifras([
            _selector(f"{PRE}-rival", [{"label": D.NOMBRE_MODELO[m], "value": m} for m in REFERENCIAS], rival, "Comparar la EBM con",
                      h["f-rival"]),
            ui.cifra("Riesgo alto que alcanza la EBM", h["f-vp"], h["f-vp-de"], clase="cifra--llana"),
            ui.cifra("Precisión de la EBM", h["f-precision"], h["f-precision-de"], clase="cifra--llana"),
            ui.cifra("Elevación de la EBM", h["f-elevacion"], "veces el azar", clase="cifra--llana"),
        ]),
    ], className="panel__grupo")], clase="pegada")


# --- Matriz de confusión de la EBM -----------------------------------------------------------------------------------
def _textos_matriz(cap):
    """Título de la tarjeta de la matriz de confusión de la EBM y sus cifras, en el orden de HOJAS_MATRIZ."""
    e = _resumen(FOCO, cap)
    if e["fn"]:
        titulo = (f"La EBM alcanza a {F.entero(e['vp'])} clientes de riesgo alto y se le "
                  f"{'escapa' if e['fn'] == 1 else 'escapan'} {F.entero(e['fn'])}")
    else:
        titulo = f"La EBM alcanza a los {F.entero(e['vp'])} clientes de riesgo alto: no se le escapa ninguno"
    return titulo, F.entero(e["vp"]), F.entero(e["fp"]), F.entero(e["fn"]), F.entero(e["vn"]), ui.num(e["corte"], DEC_CORTE)


def _matriz(cap):
    h = _hojas(HOJAS_MATRIZ, _textos_matriz(cap))
    return ui.tarjeta([html.Div([ui.matriz(h["m-vp"], h["m-fp"], h["m-fn"], h["m-vn"]), ui.ficha([("Corte de la campaña", h["m-corte"])])],
                                className="panel__grupo")],
                      titulo=h["matriz-titulo"], fuentes=[FUENTE_PUNTAJES],
                      subtitulo="Matriz de confusión de la EBM con la capacidad elegida. El corte de la campaña es el puntaje de la EBM, "
                                "sin recalibrar, del último cliente contactado.")


# --- Curva de captura ------------------------------------------------------------------------------------------------
def _porcentajes(fracciones):
    return [round(100 * float(f), 3) for f in fracciones]


def _marca_curva(cap, rival):
    """Lo que se mueve en la curva con la capacidad: los dos puntos, la línea de referencia y su rótulo."""
    a_la_izquierda = cap > CAP_GIRO
    return {"x": cap, "y_ebm": round(100 * _resumen(FOCO, cap)["captura"], 3), "y_rival": round(100 * _resumen(rival, cap)["captura"], 3),
            "text": f"{_pct(cap)} de capacidad", "xanchor": "right" if a_la_izquierda else "left",
            "xshift": -ui.px("--e-2") if a_la_izquierda else ui.px("--e-2")}


def _figura_curva(cap, rival):
    """La curva de captura de la EBM (gruesa) y del modelo de comparación (a trazos, encima, para que se vean las dos donde
    coinciden), la diagonal del azar y la capacidad elegida con el punto de cada modelo."""
    ce, cr = _curva(FOCO), _curva(rival)
    rol = D.ROL_VISTA_EBM[rival]
    x = [round(100 * c) for c in ce["capacidad"]]
    m = _marca_curva(cap, rival)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=_porcentajes(ce["captura"]), name=D.NOMBRE_MODELO[FOCO], **ui.traza("modelo", "gruesa"),
                             **ui.hover_texto(_textos_curva(FOCO))))
    fig.add_trace(go.Scatter(x=x, y=_porcentajes(cr["captura"]), name=D.NOMBRE_MODELO[rival], **ui.traza(rol, "trazos"),
                             **ui.hover_texto(_textos_curva(rival))))
    fig.add_trace(go.Scatter(x=x, y=x, name="Contactar al azar", **ui.traza("contexto"),
                             **ui.hover_texto([f"Contactar al azar: <b>{_pct(v, 1)}</b>" for v in x])))
    fig.add_trace(go.Scatter(x=[m["x"]], y=[m["y_rival"]], name=D.NOMBRE_MODELO[rival], hoverinfo="skip", cliponaxis=False,
                             **ui.traza(rol, "punto")))
    fig.add_trace(go.Scatter(x=[m["x"]], y=[m["y_ebm"]], name=D.NOMBRE_MODELO[FOCO], hoverinfo="skip", cliponaxis=False,
                             **ui.traza("modelo", "punto")))
    ui.unidad_eje(fig, "clientes de riesgo alto alcanzados")
    ui.referencia(fig, x=m["x"])
    fig.update_shapes(layer="below")
    fig.add_annotation(x=m["x"], y=0, yref="paper", yanchor="bottom", xanchor=m["xanchor"], xshift=m["xshift"], text=m["text"])
    marcas = ui.marcas(0, CAP_MAX, PASO_EJE)[0]
    fig.update_xaxes(range=[0, CAP_MAX], tickvals=marcas, ticktext=[_pct(v) for v in marcas])
    fig.update_yaxes(range=[0, CAP_MAX + HOLGURA_EJE], tickvals=marcas[1:], ticktext=[_pct(v) for v in marcas[1:]])   # el cero ya está en el otro eje
    ui.formato_eje(fig, "x", 0, " %")                       # el encabezado de la etiqueta flotante: «60 %»
    fig.update_layout(margin_r=ui.px("--e-5"))              # sitio para la última marca del eje, que si no cabe no se dibuja
    return fig


def _parche_curva(cap, rival):
    """Solo lo que se mueve en la curva ya dibujada para ese modelo de comparación."""
    m = _marca_curva(cap, rival)
    p = Patch()
    for i, clave in ((I_MARCA_RIVAL, "y_rival"), (I_MARCA_EBM, "y_ebm")):
        p["data"][i]["x"] = [m["x"]]
        p["data"][i]["y"] = [m[clave]]
    p["layout"]["shapes"][0]["x0"] = m["x"]
    p["layout"]["shapes"][0]["x1"] = m["x"]
    for clave in ("x", "text", "xanchor", "xshift"):
        p["layout"]["annotations"][I_ROTULO][clave] = m[clave]
    return p


def _titulo_curva(cap, rival):
    e, r = _resumen(FOCO, cap), _resumen(rival, cap)
    return f"Con el {_pct(cap)} la EBM alcanza al {F.pct(e['captura'])} del riesgo alto; {D.en_frase(rival)}, al {F.pct(r['captura'])}"


def _etiqueta_curva(cap, rival):
    ce, cr = _curva(FOCO), _curva(rival)
    otras = "; ".join(f"con el {_pct(c)}, {F.pct(ce['captura'].iat[c])} y {F.pct(cr['captura'].iat[c])}"
                      for c in range(PASO_EJE, CAP_MAX, PASO_EJE))
    return f"{_titulo_curva(cap, rival)}. Captura de la EBM y de {D.NOMBRE_MODELO[rival]} con otras capacidades: {otras}."


def _leyenda_curva(rival):
    return ui.leyenda([(D.NOMBRE_MODELO[FOCO], "modelo", "gruesa"), (D.NOMBRE_MODELO[rival], D.ROL_VISTA_EBM[rival], "trazos"),
                       ("Contactar al azar", "contexto", "linea")])


def _tabla_curva(rival):
    """Tabla gemela de la curva: clientes de riesgo alto alcanzados (y captura) por los dos modelos, de cinco en cinco puntos."""
    ce, cr = _curva(FOCO), _curva(rival)
    filas = [[_pct(c), F.entero(ce["contactados"].iat[c]), f"{F.entero(ce['vp'].iat[c])} ({F.pct(ce['captura'].iat[c])})",
              f"{F.entero(cr['vp'].iat[c])} ({F.pct(cr['captura'].iat[c])})"] for c in range(PASO_TABLA, CAP_MAX + 1, PASO_TABLA)]
    return ui.tabla(["Capacidad", "Contactados", D.NOMBRE_MODELO[FOCO], D.NOMBRE_MODELO[rival]], filas, compacta=True)


# --- Todos los modelos con la misma capacidad ------------------------------------------------------------------------
def _grupos(res):
    """Los modelos que van juntos y los que se descuelgan con esa capacidad, de más a menos clientes alcanzados. El corte
    está en el mayor salto entre dos modelos consecutivos, y solo cuenta si ese salto es mayor que toda la distancia entre
    los que quedan arriba y arriba queda al menos la mitad de los modelos; si no, nadie se descuelga."""
    orden = sorted(MODELOS, key=lambda m: -res[m]["vp"])
    vp = [res[m]["vp"] for m in orden]
    saltos = [a - b for a, b in zip(vp, vp[1:])]
    i = max(range(len(saltos)), key=lambda j: saltos[j])
    if saltos[i] > vp[0] - vp[i] and i + 1 >= len(orden) - (i + 1):
        return orden[:i + 1], orden[i + 1:]
    return orden, []


def _alcanzan(res, modelos):
    """«entre 1.184 y 1.228» o, si todos alcanzan lo mismo, «2.399»."""
    menor, mayor = min(res[m]["vp"] for m in modelos), max(res[m]["vp"] for m in modelos)
    return F.entero(mayor) if menor == mayor else f"entre {F.entero(menor)} y {F.entero(mayor)}"


def _textos_modelos(cap):
    """Título de la tarjeta (los modelos que van juntos y, aparte, los que se descuelgan) y las celdas que dependen de la
    capacidad, en el orden de HOJAS_MODELOS."""
    res = {m: _resumen(m, cap) for m in MODELOS}
    juntos, aparte = _grupos(res)
    quienes = f"{NUMERO[len(juntos)]} de los {NUMERO[len(MODELOS)]} modelos" if aparte else f"los {NUMERO[len(MODELOS)]} modelos"
    rango = _alcanzan(res, juntos)
    titulo = f"Con el {_pct(cap)} de capacidad, {quienes} alcanzan {rango if rango.startswith('entre') else 'a ' + rango} clientes de riesgo alto"
    if len(aparte) == 1:
        titulo += f"; {D.en_frase(aparte[0])}, {F.entero(res[aparte[0]]['vp'])}"
    elif aparte:
        titulo += f"; los otros {NUMERO[len(aparte)]}, {_alcanzan(res, aparte)}"
    celdas = [t for m in MODELOS for t in (F.entero(res[m]["vp"]), F.pct(res[m]["captura"]), F.pct(res[m]["precision"]),
                                           ui.num(res[m]["corte"], DEC_CORTE))]
    return titulo, celdas


def _nota_empates():
    """Los modelos cuyo corte, con la capacidad del informe, cae en un empate (datos/original/empates_campana.json, de la
    corrección): su fila de la tabla es uno de los desempates posibles. None si no hay ninguno o si el análisis no está."""
    con = [(m, D.empate_campana(m)) for m in D.modelos_con_empate() if m in MODELOS]
    if not con:
        return None
    partes = [f"en {D.en_frase(m)} ({F.entero(x['empatados'])} clientes con el puntaje del corte, de los que entran {F.entero(x['cupos'])}; "
              f"captura esperada con un desempate al azar: {F.pct(x['captura_esperada'])} ± {ui.num(100 * x['captura_desviacion'], 1)} puntos)"
              for m, x in con]
    return ui.nota(f"Con el {_pct(CAP_INFORME)} de capacidad, el corte cae en un empate {F.enumerar(partes)}: a quién se contacta "
                   "dentro de ese grupo lo decide el desempate, y la fila de la tabla es uno de los posibles. Con otra capacidad el "
                   "corte puede caer en otro empate.")


def _modelos(cap):
    """La tarjeta con la tabla de todos los modelos: el AUC de prueba es fijo; lo demás son hojas que cambian con la capacidad."""
    titulo, celdas = _textos_modelos(cap)
    auc = _auc_prueba()
    empates = _nota_empates()
    filas = [[D.NOMBRE_MODELO[m], ui.num(auc[m], 4)] + [_hoja(f"t-{m}-{c}", t) for c, t in zip(CAMPOS_MODELO, celdas[i * len(CAMPOS_MODELO):])]
             for i, m in enumerate(MODELOS)]
    tabla = ui.tabla(["Modelo", "AUC de prueba", "Alcanzados", "Captura", "Precisión", "Corte de la campaña"], filas, compacta=True,
                     muestras=[D.ROL_VISTA_EBM[m] for m in MODELOS], ocultar_movil=(1, 4, 5))
    return ui.tarjeta([
        tabla,
        ui.nota(f"El corte de la campaña es el puntaje, sin recalibrar, del último cliente que contacta cada modelo; no se compara "
                f"entre modelos. El del {D.NOMBRE_MODELO['svm']} es una función de decisión, no una probabilidad."),
        *([empates] if empates is not None else []),
    ], titulo=_hoja("modelos-titulo", titulo),
        fuentes=[FUENTE_PUNTAJES, FUENTE_PRUEBA] + ([D.fuente_original("empates_campana.json")] if empates is not None else []),
        subtitulo="Todos contactan al mismo número de clientes, cada uno a los que él da mayor puntaje. El AUC de prueba no depende "
                  "de la capacidad.")


def _lectura():
    """La lectura del informe sobre el punto de operación: en AUC la EBM supera a la regresión logística, pero con la
    capacidad del informe alcanzan casi a los mismos clientes y McNemar no las separa."""
    e, r = _resumen(FOCO, CAP_INFORME), _resumen(RIVAL_INICIAL, CAP_INFORME)
    e2, r2 = _resumen(FOCO, CAP_AMPLIA), _resumen(RIVAL_INICIAL, CAP_AMPLIA)
    f = D.fila(D.pruebas("clasificacion"), modelo=RIVAL_INICIAL)
    dif = e["vp"] - r["vp"]
    return ui.tarjeta_insight(
        f"{F.entero(dif)} {'cliente' if dif == 1 else 'clientes'}", "Significancia estadística no es relevancia práctica",
        f"Es lo que separa a la EBM de {D.en_frase(RIVAL_INICIAL)} con el {_pct(CAP_INFORME)} de capacidad ({F.entero(e['vp'])} y "
        f"{F.entero(r['vp'])} de {F.entero(e['positivos'])}), aunque en AUC de prueba la EBM la supera con significación tras la "
        f"corrección de Holm (diferencia de {ui.con_signo(f['delta_auc'], 4)}; DeLong, {F.p_valor(f['p_delong_holm'])} ajustado). "
        f"En ese punto la prueba de McNemar no las separa ({F.p_valor(f['p_mcnemar_holm'])} ajustado). Con el {_pct(CAP_AMPLIA)} "
        f"la diferencia es de {F.entero(e2['vp'] - r2['vp'])} clientes.", chip_="Lectura del informe")


# --- Quién queda fuera, por cohorte de edad --------------------------------------------------------------------------
def _figura_cohortes(cap, rival):
    """Barras horizontales por cohorte: la EBM sólida y el modelo de comparación rayado. El eje no cambia con la capacidad
    (llega hasta la cohorte con más clientes de riesgo alto), así que al mover el deslizador solo se acortan las barras."""
    cohortes = _prueba()[2]
    categorias = [f"{c['nivel']} años<br>{F.entero(c['positivos'])} de riesgo alto" for c in cohortes]
    ancho = ui.ancho_barra(len(cohortes), ALTO_BAJO)
    hueco = ancho * 0.1
    fig = go.Figure()
    for modelo, rol, forma, offset in ((FOCO, "modelo", "barra", hueco), (rival, D.ROL_VISTA_EBM[rival], "rayada", -ancho - hueco)):
        fuera = _fuera(modelo, cap)
        textos = [f"{D.NOMBRE_MODELO[modelo]}: <b>{F.entero(v)}</b> de {F.entero(c['positivos'])} ({F.pct(v / c['positivos'])})"
                  for v, c in zip(fuera, cohortes)]
        fig.add_trace(go.Bar(y=categorias, x=list(fuera), orientation="h", name=D.NOMBRE_MODELO[modelo], offset=offset,
                             text=[F.entero(v) for v in fuera], **ui.traza(rol, forma, ancho=ancho), **ui.hover_texto(textos)))
    tope = max(c["positivos"] for c in cohortes)
    vals, textos = ui.marcas(0, tope, ui.paso_redondo(tope, 4))
    ui.unidad_eje(fig, "clientes que quedan fuera")
    fig.update_layout(barmode="overlay", hovermode="y unified")
    fig.update_xaxes(range=[0, tope * HOLGURA_BARRAS], tickvals=vals, ticktext=textos, showgrid=True, gridcolor=ui.ROL["rejilla"],
                     showline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=categorias[::-1], tickfont=ui.fuente_svg("tinta_2"),
                     ticklabelstandoff=ui.px("--e-2"))
    return fig


def _textos_cohortes(cap, rival):
    """Título, subtítulo y etiqueta del gráfico por cohorte, y las celdas de su tabla gemela en el orden de HOJAS_COHORTES
    (cada cohorte y, al final, el total: clientes de riesgo alto que quedan fuera con la EBM y con el otro modelo)."""
    cohortes = _prueba()[2]
    fe, fr = _fuera(FOCO, cap), _fuera(rival, cap)
    if sum(fe) or sum(fr):
        i = max(range(len(cohortes)), key=lambda j: abs(fe[j] - fr[j]))        # la cohorte en la que más difieren
        titulo = (f"Con el {_pct(cap)}, la EBM deja fuera a {F.entero(fe[i])} clientes de riesgo alto de {_edad(cohortes[i]['nivel'])}; "
                  f"{D.en_frase(rival)}, a {F.entero(fr[i])}")
    else:
        titulo = f"Con el {_pct(cap)} de capacidad no queda fuera ningún cliente de riesgo alto"
    sub = (f"Clientes de riesgo alto que la campaña no alcanza, por cohorte de edad. En total, {F.entero(sum(fe))} con la EBM y "
           f"{F.entero(sum(fr))} con {D.en_frase(rival)}.")
    etiqueta = (f"{titulo}. " + "; ".join(f"{_edad(c['nivel'])}: {F.entero(a)} con la EBM y {F.entero(b)} con {D.en_frase(rival)}"
                                         for c, a, b in zip(cohortes, fe, fr)) + ".")
    positivos = [c["positivos"] for c in cohortes] + [sum(c["positivos"] for c in cohortes)]
    celdas = [f"{F.entero(v)} ({F.pct(v / n)})" for a, b, n in zip(fe + (sum(fe),), fr + (sum(fr),), positivos) for v in (a, b)]
    return titulo, sub, etiqueta, celdas


def _tabla_cohortes(cap, rival):
    cohortes = _prueba()[2]
    h = _hojas(HOJAS_COHORTES, _textos_cohortes(cap, rival)[3])
    filas = [[_edad(c["nivel"]), F.entero(c["positivos"]), h[f"c-{i}-ebm"], h[f"c-{i}-rival"]] for i, c in enumerate(cohortes)]
    total = ["Total", F.entero(sum(c["positivos"] for c in cohortes)), h[f"c-{N_COHORTES}-ebm"], h[f"c-{N_COHORTES}-rival"]]
    return ui.tabla(["Cohorte de edad", "De riesgo alto", D.NOMBRE_MODELO[FOCO], _hoja("c-rival", D.NOMBRE_MODELO[rival])], filas,
                    total=total, compacta=True)


def _leyenda_cohortes(rival):
    return ui.leyenda([(D.NOMBRE_MODELO[FOCO], "modelo", "barra"), (D.NOMBRE_MODELO[rival], D.ROL_VISTA_EBM[rival], "rayada")])


def _lectura_cohortes():
    """Lo que dice el informe (EBM frente a XGBoost con su capacidad), con las cifras de errores_segmentos.csv."""
    t = D.errores_segmentos()
    t = t[t["segmento"] == "cohorte de edad"]
    mayor, menor = t.loc[t["prevalencia"].idxmax()], t.loc[t["prevalencia"].idxmin()]
    rival = D.NOMBRE_MODELO[RIVAL_COHORTES]
    col = f"fn_n_{RIVAL_COHORTES}"
    return ui.tarjeta([
        ui.parrafo(f"En el punto de operación del informe ({_pct(CAP_INFORME)}), la EBM y {rival} dejan fuera casi al mismo número de "
                   f"clientes de riesgo alto ({F.entero(t['fn_n_ebm'].sum())} y {F.entero(t[col].sum())}), pero no a los mismos."),
        ui.parrafo(f"De la cohorte de {_edad(mayor['nivel'])}, donde el {F.pct(mayor['prevalencia'])} de los clientes de prueba es de "
                   f"riesgo alto, la EBM deja fuera a {F.entero(mayor['fn_n_ebm'])} y {rival} a {F.entero(mayor[col])}; de la de "
                   f"{_edad(menor['nivel'])}, a {F.entero(menor['fn_n_ebm'])} y a {F.entero(menor[col])}. La edad domina al modelo aditivo."),
        ui.nota(f"Para verlo en el gráfico, elija {rival} en «Comparar la EBM con» y deje la capacidad en el {_pct(CAP_INFORME)}."),
        ui.boton("Ver los errores por segmento", href="/modelos/errores", icono="lupa"),
    ], titulo="La misma captura, con clientes distintos", fuentes=[D.fuente("resultados/errores_segmentos.csv")])


# --- Por qué no se usa el umbral de probabilidad ---------------------------------------------------------------------
def _umbral():
    """Cuántos clientes marca cada modelo con su umbral de clase (vp_05 + fp_05 de la tabla de prueba) y por qué eso no
    sirve para comparar: los puntajes no están calibrados ni son comparables entre modelos."""
    t = D.lideres("clasificacion")
    t = t[t["modelo"].isin(MODELOS)]
    marcados = (t["vp_05"] + t["fp_05"]).astype(int).tolist()
    modelos = t["modelo"].tolist()
    n = M.meta()["n_prueba"]
    umbral = ui.num(UMBRAL_CLASE, 1)

    def cuantos(k):
        return F.entero(k) if k else "ninguno"
    i_min, i_max = marcados.index(min(marcados)), marcados.index(max(marcados))
    titulo = (f"Con el umbral {umbral} cada modelo marca a un número distinto de clientes: de {cuantos(marcados[i_min])} "
              f"({D.NOMBRE_MODELO[modelos[i_min]]}) a {cuantos(marcados[i_max])} ({D.NOMBRE_MODELO[modelos[i_max]]})")
    filas = [[D.NOMBRE_MODELO[m], D.NOMBRE_BALANCEO[b], cuantos(k), F.pct(k / n)] for m, b, k in zip(modelos, t["balanceo"], marcados)]
    tabla = ui.tabla(["Modelo", "Balanceo", "Marcados", "Parte de la prueba"], filas, compacta=True,
                     muestras=[D.ROL_VISTA_EBM[m] for m in modelos], num=[False, False, True, True], ocultar_movil=(1,))
    cal = D.fila(D.tabla("entrega3/complementos/calibracion.csv"), modelo=FOCO, version="sin recalibrar")
    marcar = ui.tarjeta([tabla], titulo=titulo, fuentes=[FUENTE_PRUEBA],
                        subtitulo=f"Clientes de prueba (de {F.entero(n)}) que cada modelo final clasifica como de riesgo alto. El "
                                  f"{D.NOMBRE_MODELO['svm']} no da una probabilidad: se cuenta la clase que predice.")
    razon = ui.tarjeta([
        ui.parrafo(f"Un umbral fijo solo sirve para comparar si el puntaje es una probabilidad calibrada, y aquí no lo es: con "
                   f"ponderación de clases el puntaje de la EBM promedia {ui.num(cal['prob_media'], 3)} cuando la proporción de clientes "
                   f"de riesgo alto es {ui.num(cal['prevalencia'], 3)}."),
        ui.parrafo(f"Con la misma capacidad, en cambio, todos los modelos contactan al mismo número de clientes y solo cambia a "
                   f"quiénes. Por eso el informe compara con la capacidad del {_pct(CAP_INFORME)} y esta pantalla ordena siempre por "
                   f"el puntaje sin recalibrar."),
    ], titulo="Los puntajes no son comparables entre modelos", fuentes=[D.fuente("entrega3/complementos/calibracion.csv")])
    return marcar, razon


# --- Pantalla ----------------------------------------------------------------------------------------------------------
def _subtitulo():
    e, r = _resumen(FOCO, CAP_INFORME), _resumen(RIVAL_INICIAL, CAP_INFORME)
    e2, r2 = _resumen(FOCO, CAP_AMPLIA), _resumen(RIVAL_INICIAL, CAP_AMPLIA)
    return (f"Con capacidad para el {_pct(CAP_INFORME)} de los clientes ({F.entero(e['contactados'])}), la EBM alcanza a "
            f"{F.entero(e['vp'])} de los {F.entero(e['positivos'])} de riesgo alto ({F.pct(e['captura'])}), casi lo mismo que "
            f"{D.en_frase(RIVAL_INICIAL)} ({F.entero(r['vp'])}). La diferencia aparece con más capacidad: con el {_pct(CAP_AMPLIA)}, "
            f"{F.pct(e2['captura'])} frente a {F.pct(r2['captura'])}.")


def layout(**_):
    cap, rival = CAP_INFORME, RIVAL_INICIAL                 # la pantalla nace con la capacidad y la comparación del informe
    n = _resumen(FOCO, cap)["n"]
    # La curva nace con el lienzo bajo (cabe bajo la tarjeta pegada en cualquier ventana); _alto_curva lo agranda si hay sitio
    curva = ui.grafico(_figura_curva(cap, rival), _titulo_curva(cap, rival), f"Captura acumulada en los {F.entero(n)} clientes de prueba.",
                       _etiqueta_curva(cap, rival), tabla=_tabla_curva(rival), leyenda=_leyenda_curva(rival), alto="bajo",
                       id=f"{PRE}-curva", fuentes=[FUENTE_PUNTAJES])
    titulo_cohortes, sub_cohortes, etiqueta_cohortes, _celdas = _textos_cohortes(cap, rival)
    cohortes = ui.grafico(_figura_cohortes(cap, rival), titulo_cohortes, sub_cohortes, etiqueta_cohortes,
                          tabla=_tabla_cohortes(cap, rival), leyenda=_leyenda_cohortes(rival), alto="bajo", id=f"{PRE}-cohortes",
                          fuentes=[FUENTE_PUNTAJES, FUENTE_CLIENTES])
    marcar, razon = _umbral()
    return ui.pantalla([
        ui.encabezado(PREGUNTA, _subtitulo(), pantalla=PANTALLA),
        ui.seccion("La campaña en vivo", [
            _control(cap, rival),
            ui.rejilla([curva, _matriz(cap)], "1-1"),
            dcc.Store(id=f"{PRE}-dibujado", data=rival),
        ], subtitulo="La campaña contacta primero a los clientes a los que el modelo da mayor puntaje, hasta agotar su capacidad. "
                     "«Riesgo alto» es el cuarto superior del puntaje de propensión del conjunto de datos, no fuga observada; "
                     "«alcanzar» es contactar a un cliente de riesgo alto.",
            fuentes=[FUENTE_PUNTAJES]),
        ui.seccion(f"Los {NUMERO[len(MODELOS)]} modelos con la misma capacidad", [
            ui.rejilla([_modelos(cap), _lectura()], "2-1"),
        ], fuentes=[FUENTE_PUNTAJES, FUENTE_PRUEBA, D.fuente("resultados/pruebas_clasificacion.csv")]),
        ui.seccion("Quién queda fuera", [
            ui.rejilla([cohortes, _lectura_cohortes()], "2-1"),
        ], subtitulo="Dos modelos con la misma captura pueden dejar fuera a clientes distintos. La edad solo toma tres cohortes en "
                     "este conjunto de datos."),
        ui.seccion("Por qué se compara con la misma capacidad y no con un umbral", [
            ui.rejilla([marcar, razon], "2-1"),
            ui.aviso("El proyecto no tiene el costo de cada contacto ni el valor de retener a un cliente, así que aquí no se calcula "
                     "ninguna ganancia en dinero. El informe deja como trabajo futuro incorporar el costo de cada acción de retención "
                     "para fijar el umbral y la capacidad.", None, "Esta pantalla cuenta clientes, no dinero"),
        ]),
        ui.pie([ui.nota(f"Todo se calcula en el momento con el puntaje sin recalibrar que cada modelo final dio a los "
                        f"{F.entero(n)} clientes de prueba. Los empates de puntaje se rompen al azar con la semilla del proyecto, "
                        f"como en el informe.")]),
    ])


@callback(
    Output(f"{PRE}-capacidad-valor", "children"),
    *[Output(f"{PRE}-{clave}", "children") for clave in HOJAS_CIFRAS + HOJAS_MATRIZ],
    Output(f"{PRE}-curva", "figure"), Output(f"{PRE}-curva-titulo", "children"), Output(f"{PRE}-curva-lienzo", "aria-label"),
    Output(f"{PRE}-curva-leyenda", "children"), Output(f"{PRE}-curva-tabla", "children"), Output(f"{PRE}-dibujado", "data"),
    Input(f"{PRE}-capacidad", "drag_value"), Input(f"{PRE}-capacidad", "value"), Input(f"{PRE}-rival", "value"),
    State(f"{PRE}-dibujado", "data"))
def _en_vivo(arrastre, capacidad, rival, dibujado):
    """La parte interactiva: se pone al día mientras se arrastra el deslizador. Si la curva del navegador ya es la de ese
    modelo de comparación (`dibujado`), solo se parchan sus partes móviles y no se reenvían su leyenda ni su tabla."""
    solo_arrastre = [p["prop_id"] for p in ctx.triggered] == [f"{PRE}-capacidad.drag_value"]
    cap = _capacidad(arrastre if solo_arrastre and arrastre is not None else capacidad)
    rival = _referencia(rival)
    if dibujado == rival:
        curva, del_modelo = _parche_curva(cap, rival), (dash.no_update, dash.no_update)
    else:
        curva, del_modelo = _figura_curva(cap, rival), (_leyenda_curva(rival), _tabla_curva(rival))
    return (_texto_capacidad(cap), *_textos_cifras(cap, rival), *_textos_matriz(cap), curva, _titulo_curva(cap, rival),
            _etiqueta_curva(cap, rival), *del_modelo, rival)


@callback(
    Output(f"{PRE}-modelos-titulo", "children"),
    *[Output(f"{PRE}-{clave}", "children") for clave in HOJAS_MODELOS],
    Output(f"{PRE}-cohortes", "figure"), Output(f"{PRE}-cohortes-titulo", "children"), Output(f"{PRE}-cohortes-sub", "children"),
    Output(f"{PRE}-cohortes-lienzo", "aria-label"), Output(f"{PRE}-cohortes-leyenda", "children"), Output(f"{PRE}-c-rival", "children"),
    *[Output(f"{PRE}-{clave}", "children") for clave in HOJAS_COHORTES],
    Input(f"{PRE}-capacidad", "value"), Input(f"{PRE}-rival", "value"))
def _al_soltar(capacidad, rival):
    """Lo que queda bajo la parte interactiva (todos los modelos y las cohortes): se rehace al soltar el deslizador, al
    moverlo con el teclado o al cambiar el modelo de comparación."""
    cap, rival = _capacidad(capacidad), _referencia(rival)
    titulo_modelos, celdas_modelos = _textos_modelos(cap)
    titulo, sub, etiqueta, celdas = _textos_cohortes(cap, rival)
    return (titulo_modelos, *celdas_modelos, _figura_cohortes(cap, rival), titulo, sub, etiqueta, _leyenda_cohortes(rival),
            D.NOMBRE_MODELO[rival], *celdas)


# El lienzo de la curva según el alto de la ventana (en el navegador): normal si la curva y la matriz caben bajo la tarjeta
# pegada, bajo si no. Se revisa al cargar, al cambiar el modelo y cada vez que se confirma una capacidad.
clientside_callback(
    f"function (_modelo, _capacidad) {{ return window.innerHeight >= {ALTO_VENTANA_AMPLIA} ? '{CLASE_LIENZO}' : '{CLASE_LIENZO_BAJO}'; }}",
    Output(f"{PRE}-curva-lienzo", "className"), Input(f"{PRE}-dibujado", "data"), Input(f"{PRE}-capacidad", "value"))
