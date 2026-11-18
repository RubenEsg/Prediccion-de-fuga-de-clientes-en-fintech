"""Comparación estadística · ¿Supera el modelo original a los benchmarks?

Dos mitades, y cada una dice cuál es:

  · El modelo original (BDL) frente a los nueve benchmarks, con el protocolo pre-registrado de la corrección: el veredicto
    de superioridad con las cuatro condiciones de la guía (de veredicto.json y condicion4.json, nunca de un texto fijo); los
    diez modelos a la vez, con el diagrama de diferencia crítica de las dos tareas (rangos medios de los pliegues externos,
    Friedman, Iman-Davenport y la diferencia crítica de Nemenyi, de datos/cd.json); el original frente al mejor benchmark y
    frente a cada uno, con las pruebas P1 a P4, sus p ajustados por Holm y los tamaños del efecto (pruebas_hipotesis.csv);
    el desempeño de los diez en la prueba con su intervalo y sus pliegues de la validación anidada. En estos gráficos la
    serie en azul es BDL. Las tarjetas del veredicto y de las pruebas son las de secciones/original.py.
  · El análisis del Entregable 3, sin cambios: la EBM, que hoy es un benchmark, frente a sus siete referencias de clase (la
    diferencia con cada una sobre la banda de equivalencia práctica, la ficha de una referencia con todas sus pruebas y, en
    regresión, el Model Confidence Set del informe) y, aparte, con la etiqueta de la mediana, la comparación con la entrega
    anterior y el techo de información. En estos gráficos la serie en azul es la EBM (ROL = D.ROL_VISTA_EBM).

El veredicto de cada referencia de la EBM es el de datos_app.veredicto (p ajustados por Holm y ALFA): ninguno se escribe a
mano. Toda cifra sale de datos/ y cada tarjeta dice de qué archivo. Si la corrección todavía no dejó sus archivos, la
primera mitad muestra cada bloque como pendiente y los gráficos de todos los modelos caen en los ocho del Entregable 3.
"""
from functools import lru_cache

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

import app
import datos_app as D
import formato as F
import motor as M
import ui
from secciones import original as OG


VISTA = "comparacion"
PRE = "co"
PANTALLA = "comparacion"
O = D.ORIGINAL
ROL = D.ROL_VISTA_EBM             # el rol de color en la mitad del Entregable 3: la EBM en foco (azul) frente a sus referencias
RIVAL = "xgboost"                 # la referencia con que nace la ficha: el modelo de árboles más cercano a la EBM
NIVEL_MCS = 0.90                  # nivel del Model Confidence Set del informe
UMBRAL_EQUIVALENCIA = 0.95        # «el umbral habitual del 95 %» con que el informe lee la probabilidad de equivalencia
AZAR_AUC = 0.5                    # AUC de ordenar al azar: el piso sobre el que se mide el techo de información
PISO_PROBABILIDAD = 1e-4          # una probabilidad posterior más cerca de 0 o de 1 que esto se escribe como cota
MARGEN_EJE = 0.04                 # aire a cada lado del eje de valores, como fracción de su recorrido
MARCAS_EJE = 4                    # marcas del eje de la diferencia a la derecha del cero
PASO_ROC = 0.25                   # paso de las marcas de los dos ejes de la curva ROC
MARGEN_ROC = 0.02                 # aire alrededor de la curva ROC, para que la línea no se corte en el borde
AZAR_ROTULO = 0.72                # punto de la diagonal donde va el rótulo «azar»
PASO_TABLA_ROC = 10               # la tabla gemela de la curva ROC lleva uno de cada diez puntos de la curva
DECIMALES_REPARTO = 3             # las tres barras del reparto bayesiano llevan los mismos decimales, para que sumen 1
DECIMALES_UMBRAL = 4              # el umbral y la mediana del puntaje, como en Datos y en Cliente

FIJADA = "fijada antes"                 # protocolo de cada prueba, en una palabra (el aviso de la sección lo dice entero)
EXPLORATORIA = "exploratoria"
CARDINAL = {1: "un", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
            15: "quince"}
ORDINAL_F = {1: "primera", 2: "segunda", 3: "tercera", 4: "cuarta", 5: "quinta", 6: "sexta", 7: "séptima", 8: "octava", 9: "novena",
             10: "décima"}
ORDINAL_M = {1: "primer", 2: "segundo", 3: "tercer", 4: "cuarto", 5: "quinto", 6: "sexto", 7: "séptimo", 8: "octavo"}
ORDINAL_MASC = {1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto", 5: "quinto", 6: "sexto", 7: "séptimo", 8: "octavo", 9: "noveno",
                10: "décimo"}
TIPO_VEREDICTO = {"supera": "bien", "empata": None, "discrepan": "atencion", "pierde": "critico"}
# Gris y rosa nunca con la misma forma: los modelos de árboles van en círculo y la base en cuadrado (en la leyenda, «barra»)
SIMBOLO = {"modelo": "circle", "rival": "circle", "base": "square"}
FORMA_LEYENDA = {"modelo": "punto", "rival": "punto", "base": "barra"}

# Lo que cambia entre las dos tareas: columnas de datos/resultados, decimales del informe y nombres de las pruebas
TAREA = {
    "clasificacion": {
        "metrica": "AUC", "d": 4, "d_eje": 2, "prueba": "auc_prueba", "inf": "auc_bca_inf", "sup": "auc_bca_sup",
        "externa": "auc_ext_media", "externa_desv": "auc_ext_desv", "externa_nombre": "AUC externo", "delta": "delta_auc",
        "fijada": "DeLong", "simbolo": "z", "est_fijada": "z_delong", "p_fijada_cruda": "p_delong", "p_fijada": "p_delong_holm",
        "rope": "auc", "d_rope": 2, "mayor_es_mejor": True, "pliegues": "resultados/pliegues_auc.csv",
        "resta": "AUC de la EBM menos AUC de la referencia", "base": "Lineales, k-NN y Naive Bayes", "e2": "logistica_e2",
        "que": "Clasificación del riesgo alto"},
    "regresion": {
        "metrica": "RMSE", "d": 5, "d_eje": 3, "prueba": "rmse_prueba", "inf": "rmse_bca_inf", "sup": "rmse_bca_sup",
        "externa": "rmse_ext_media", "externa_desv": "rmse_ext_desv", "externa_nombre": "RMSE externo", "delta": "delta_rmse",
        "fijada": "Diebold-Mariano", "simbolo": "DM", "est_fijada": "dm_hln", "p_fijada_cruda": "p_dm", "p_fijada": "p_dm_holm",
        "rope": "rmse", "d_rope": 4, "mayor_es_mejor": False, "pliegues": "resultados/pliegues_rmse.csv",
        "resta": "RMSE de la referencia menos RMSE de la EBM", "base": "Lineales y k-NN", "e2": "lasso_e2",
        "que": "Regresión del puntaje"},
}
FUENTES_PRUEBA = [D.fuente(f"resultados/{t}_prueba.csv") for t in TAREA] + [D.fuente("resultados/resumen.json")]
FUENTES_PRUEBAS = [D.fuente(f"resultados/pruebas_{t}.csv") for t in TAREA] + FUENTES_PRUEBA
FUENTES_PLIEGUES = ([D.fuente_original("pliegues_<tarea>.csv")] + [D.fuente(TAREA[t]["pliegues"]) for t in TAREA]
                    + FUENTES_PRUEBA[:-1] + [D.fuente_original("lideres_<tarea>.csv")])
FUENTES_PAR = [D.fuente("puntajes_prueba.parquet"), D.fuente(TAREA["regresion"]["pliegues"])] + FUENTES_PRUEBAS
FUENTES_E2 = [D.fuente("resultados/entrega2.csv"), D.fuente("entrega3/comparacion_entrega2/techo.json")]
BUSQUEDAS_E2 = "entrega3/comparacion_entrega2/busquedas.json"      # cómo se afinaron la EBM y XGBoost de esa comparación
CD_JSON = "cd.json"               # Friedman, Iman-Davenport y Nemenyi sobre los pliegues externos, por tarea
FUENTES_CD = [D.fuente(CD_JSON), D.fuente_original("pliegues_<tarea>.csv")]
FUENTES_MCS = [D.fuente("resultados/regresion_prueba.csv"), D.fuente("resultados/resumen.json"),
               D.fuente("resultados/pruebas_regresion.csv")]
FUENTES_DIEZ = FUENTES_PRUEBA[:2] + [D.fuente_original("lideres_<tarea>.csv"), D.fuente_original("pliegues_<tarea>.csv"),
                                     FUENTES_PRUEBA[-1]]
FILA_DC = "Diferencia crítica"    # la fila de arriba del diagrama: la barra de la diferencia crítica de Nemenyi
TOLERANCIA_RANGO = 1e-9           # los rangos medios recalculados de los pliegues deben dar los de cd.json
# Las partes de ui.grafico que escribe un callback, en este orden
PARTES_GRAFICO = (("", "figure"), ("-titulo", "children"), ("-sub", "children"), ("-leyenda", "children"), ("-tabla", "children"),
                  ("-nota", "children"), ("-lienzo", "aria-label"))


# --- Textos y números -------------------------------------------------------------------------------------------------
def _p(p):
    """Un valor p para una celda cuyo encabezado ya dice que es un p: «0,245» o «< 0,001»."""
    return F.p_valor(p).replace(F.FIJO, " ").replace("p = ", "").replace("p ", "").replace(" ", F.FIJO)


def _prob(p, prosa=False):
    """Probabilidad posterior con los decimales justos para no redondearla a 0 ni a 1 (0,94 · 0,997 · 0,9998), como en el
    texto del informe; más cerca de 0 o de 1 que PISO_PROBABILIDAD, una cota («< 0,0001»; en una frase, «menor que 0,0001»)."""
    menor, mayor = ("menor que ", "mayor que ") if prosa else (f"<{F.FIJO}", f">{F.FIJO}")
    if p < PISO_PROBABILIDAD:
        return menor + ui.num(PISO_PROBABILIDAD, 4)
    if p > 1 - PISO_PROBABILIDAD:
        return mayor + ui.num(1 - PISO_PROBABILIDAD, 4)
    return F.prob(p)


def _mayuscula(texto):
    return texto[:1].upper() + texto[1:]


def _cardinal(n):
    return CARDINAL.get(n, F.entero(n))


def _tras(preposicion, frase):
    """«a» o «de» delante de un nombre en frase, con su contracción: «al árbol de decisión», «del SVM lineal», «de la EBM»."""
    if frase.startswith("el "):
        return {"a": "al", "de": "del"}[preposicion] + frase[2:]
    return f"{preposicion} {frase}"


def _lista(modelos):
    """Varios modelos dentro de una frase (D.en_frase): «XGBoost, Random Forest y el árbol de decisión»."""
    return F.enumerar(D.en_frase(m) for m in modelos)


def _rotulo_ficha(ref):
    """El texto del botón que lleva a la ficha de la EBM frente a la referencia elegida."""
    return f"Ver la EBM frente {_tras('a', D.en_frase(ref))}"


def _referencia(tarea, ref):
    """La referencia pedida si es una de las siete de esa tarea; si no, la que trae la pantalla al nacer."""
    refs = list(D.pruebas(tarea)["modelo"])
    return ref if ref in refs else (RIVAL if RIVAL in refs else refs[0])


def _opciones(tarea):
    return [{"label": D.NOMBRE_MODELO[m], "value": m} for m in D.pruebas(tarea)["modelo"]]


def _selector_referencia(tarea):
    """ui.selector de la referencia. Nace con las referencias de las dos tareas (el callback de la tarea deja enseguida las
    siete que tocan): el selector recuerda su valor en la sesión y, al recargar, Dash lo repone antes de que corra ningún
    callback; si ese valor no estuviera entre las opciones (Lasso con la lista de clasificación), el desplegable lo
    borraría y la ficha quedaría sin referencia. Además se le da alto para la lista entera: con el alto por omisión hay
    que desplazarse para ver las dos últimas referencias (Naive Bayes entre ellas)."""
    todas = _opciones(tarea) + [o for t in TAREA if t != tarea for o in _opciones(t) if o not in _opciones(tarea)]
    selector = ui.selector(f"{PRE}-ref", todas, _referencia(tarea, RIVAL))
    selector.maxHeight = len(_opciones(tarea)) * ui.px("--e-7")
    return selector


def _n_pliegues(tarea):
    return len(D.tabla(TAREA[tarea]["pliegues"]))


def _foco(modelos):
    """El modelo en foco de los gráficos con todos los modelos: el original si ya está en la tarea; si no, la EBM."""
    return O if O in set(modelos) else "ebm"


def _rol_diez(modelos):
    """El rol de color de cada modelo en esos gráficos: el de D.ROL_MODELO (el original en azul y la EBM entre los rivales)
    o, si el original todavía no está, el de las vistas de la EBM."""
    return D.ROL_MODELO if O in set(modelos) else ROL


def _queda(modelo, puesto):
    """«queda cuarto» (BDL, un modelo) o «queda cuarta» (la EBM), con el puesto en letras."""
    ordinal = (ORDINAL_F if D.en_frase(modelo).startswith("la ") else ORDINAL_MASC).get(puesto, f"en el puesto {F.entero(puesto)}")
    return f"queda {ordinal}"


def _mcs():
    """Model Confidence Set de la regresión: los modelos que quedan dentro, el p de la EBM y si es la última en salir."""
    t = D.prueba("regresion")
    dentro = [m for m, esta in zip(t["modelo"], t["en_mcs"]) if esta]
    fuera = t[~t["en_mcs"].astype(bool)].sort_values("p_mcs", ascending=False)
    return dentro, float(D.fila(t, modelo="ebm")["p_mcs"]), (not fuera.empty and fuera.iloc[0]["modelo"] == "ebm")


def _texto_mcs():
    dentro, p_ebm, ultima = _mcs()
    estado = "la EBM queda dentro" if "ebm" in dentro else ("la EBM es la última en salir" if ultima else "la EBM queda fuera")
    return (f"El Model Confidence Set al {F.pct(NIVEL_MCS, 0)} deja {'solo ' if len(dentro) == 1 else ''}{_tras('a', _lista(dentro))}; "
            f"{estado} ({F.p_valor(p_ebm)}).")


def _respuesta_original(tarea):
    """La respuesta a la pregunta de la pantalla en la tarea elegida, con lo que digan veredicto.json y condicion4.json."""
    c = TAREA[tarea]
    que = D.NOMBRE_TAREA[tarea].lower()
    r, e = D.veredicto_original(tarea), D.estado_veredicto(tarea)
    if r is None:
        return f"Todavía no se puede responder en {que}: las pruebas de hipótesis del modelo original no han llegado."
    mejor = r["mejor_benchmark"]
    lado = "mejor" if r["condicion_1_media_mejor"] else "peor"
    inicio = {"superior": "Sí", "no superior": "No", "pendiente": "Todavía no se sabe"}[e["palabra"]]
    return (f"{inicio}. En {que}, el veredicto de BDL con la regla de superioridad de la guía es: {D.frase_veredicto(tarea)}. Su "
            f"{c['externa_nombre']} es {ui.num(r['medias'][O], c['d'])} frente a {ui.num(r['medias'][mejor], c['d'])} "
            f"{_tras('de', D.en_frase(mejor))}, el mejor benchmark ({lado} por {ui.num(abs(r['ventaja_sobre_el_mejor']), c['d'] + 1)}).")


def _respuesta_ebm(tarea):
    """Lo que dice el análisis del Entregable 3 de la EBM frente a sus siete referencias, con las cifras de la tarea."""
    c = TAREA[tarea]
    t, pr, v = D.prueba(tarea), D.pruebas(tarea), D.veredictos(tarea)
    empates = [m for m in v.index if v[m] == "empata"]
    superadas = [m for m in v.index if v[m] == "supera"]
    resto = [m for m in v.index if v[m] not in ("empata", "supera")]
    sujeto = "La EBM, benchmark del Entregable 3,"
    frases = []
    if tarea == "clasificacion":
        if empates:
            arboles = {m for m in v.index if ROL[m] == "rival"}
            quienes = "los modelos de árboles" if set(empates) == arboles else "todos los demás"
            mcnemar = all(D.fila(pr, modelo=m)["p_mcnemar_holm"] >= D.ALFA for m in empates)
            pruebas = "ninguna prueba encuentra diferencia" if mcnemar else f"ni la t corregida ni {c['fijada']} encuentran diferencia"
            frases.append(f"{sujeto} no es mejor que {quienes}: frente {_tras('a', _lista(empates))} {pruebas} tras la corrección de Holm "
                          f"(probabilidad de equivalencia práctica: {F.enumerar(_prob(D.fila(pr, modelo=m)['p_rope']) for m in empates)}).")
    else:
        e, puesto, primero = D.fila(t, modelo="ebm"), list(t["modelo"]).index("ebm") + 1, t.iloc[0]
        entre = f"entre los {CARDINAL[len(t)]} modelos de esa entrega"
        if puesto == 1:
            lugar = f"{sujeto} tiene el menor {c['metrica']} de prueba {entre} ({ui.num(e[c['prueba']], c['d'])})"
        else:
            lugar = (f"{sujeto} tiene el {ORDINAL_M[puesto]} menor {c['metrica']} de prueba {entre} ({ui.num(e[c['prueba']], c['d'])}), a "
                     f"{ui.num(abs(e[c['prueba']] - primero[c['prueba']]), c['d'])} {_tras('de', D.en_frase(primero['modelo']))}")
        frases.append(lugar + (f"; con los p ajustados, ni la t corregida ni {c['fijada']} la separan {_tras('de', _lista(empates))}."
                               if empates else "."))
    if superadas:
        frases.append(("Sí supera" if frases else f"{sujeto} supera") + f" con significación, tras la corrección de Holm, "
                      f"{_tras('a', _lista(superadas))}.")
    if resto:
        frases.append(f"Frente {_tras('a', _lista(resto))} el veredicto es otro: véalo en su ficha.")
    return " ".join(frases)


def _respuesta(tarea):
    """El subtítulo de la pantalla: primero el modelo original (el veredicto de los archivos) y después la EBM."""
    return f"{_respuesta_original(tarea)} {_respuesta_ebm(tarea)}"


# --- Piezas de los gráficos con una fila por modelo ---------------------------------------------------------------------
def _marca(rol):
    """Estilo del punto de un modelo: el color de su rol y la forma de su grupo (SIMBOLO)."""
    estilo = ui.traza(rol, "punto")
    estilo["marker"] = dict(estilo["marker"], symbol=SIMBOLO[rol])
    return estilo


def _linea(rol):
    """Estilo de la línea con marcadores de un modelo: la forma de línea de su rol y el marcador de su grupo."""
    estilo = ui.traza(rol, marcadores=True)
    estilo["marker"] = dict(estilo["marker"], symbol=SIMBOLO[rol])
    return estilo


def _bigotes(valor, inf, sup):
    """Barras de error asimétricas de un intervalo [inf; sup] alrededor de un valor (en tinta: el color lo lleva el punto)."""
    return {"type": "data", "symmetric": False, "array": [s - x for s, x in zip(sup, valor)],
            "arrayminus": [x - i for x, i in zip(valor, inf)], "color": ui.ROL["tinta_2"], "thickness": ui.ANCHO_LINEA}


def _lienzo_modelos(fig, nombres, lo, hi):
    """Lo común de los gráficos con una fila por modelo: el orden del informe de arriba abajo, la rejilla vertical, el
    recorrido del eje con algo de aire y sitio a la derecha para el valor de cada fila."""
    aire = (hi - lo) * MARGEN_EJE
    fig.update_layout(hovermode="closest", margin_r=ui.MARGEN_ETIQUETAS + ui.px("--e-4"))
    fig.update_xaxes(range=[lo - aire, hi + aire], showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=nombres[::-1], tickfont=ui.fuente_svg("tinta_2", "--t-s"))
    return fig


def _valores_al_borde(fig, nombres, textos, foco=None):
    """El valor de cada fila, ya formateado, en el margen derecho (el de la fila en foco, en tinta)."""
    for nombre, texto in zip(nombres, textos):
        fig.add_annotation(x=1, xref="paper", y=nombre, xanchor="left", xshift=ui.px("--e-2"), text=texto, showarrow=False,
                           font=ui.fuente_svg("tinta" if nombre == foco else "tinta_2"))
    return fig


def _leyenda_modelos(tarea, con_ebm=True, extra=None):
    """La leyenda de los gráficos del Entregable 3: la EBM en foco, los modelos de árboles y la base."""
    items = ([(D.NOMBRE_MODELO["ebm"], "modelo", FORMA_LEYENDA["modelo"])] if con_ebm else []) + [
        ("Modelos de árboles", "rival", FORMA_LEYENDA["rival"]), (TAREA[tarea]["base"], "base", FORMA_LEYENDA["base"])]
    return ui.leyenda(items + ([extra] if extra else []))


def _leyenda_diez(tarea, modelos, extra=None):
    """La leyenda de los gráficos con todos los modelos: el modelo en foco (el original), los rivales y la base."""
    foco = _foco(modelos)
    rivales = "EBM, LightGBM y modelos de árboles" if foco == O else "Modelos de árboles"
    items = [(D.NOMBRE_MODELO[foco], "modelo", FORMA_LEYENDA["modelo"]), (rivales, "rival", FORMA_LEYENDA["rival"]),
             (TAREA[tarea]["base"], "base", FORMA_LEYENDA["base"])]
    return ui.leyenda(items + list(extra or []))


def _salidas(gid):
    return [Output(f"{gid}{sufijo}", propiedad) for sufijo, propiedad in PARTES_GRAFICO]


def _partes(g):
    return [g["figura"], g["titulo"], g["sub"], g["leyenda"], g["tabla"], g["nota"], g["etiqueta"]]


# --- 0. El veredicto del modelo original y todos los modelos a la vez: Friedman → Nemenyi -----------------------------------
def _veredicto():
    """El veredicto del modelo original en las dos tareas: las cuatro condiciones de la regla, una por una, y por qué. Son
    las mismas piezas de la vista «Modelo original»: todo sale de veredicto.json y de condicion4.json."""
    if all(D.veredicto_original(t) is None for t in TAREA):
        return [OG._pendiente("El veredicto del modelo original", "veredicto.json", "lo calcula 04_pruebas_hipotesis.py"),
                OG._tarjeta_condiciones()]
    por_que = [" ".join(x for x in (OG._por_que(t), OG._causa(t)) if x) for t in TAREA]
    ir = html.Div(ui.boton("Ver el modelo original", href=app.ruta_vista("original"), icono="modelo"),
                  className="botonera")
    return [OG._tarjeta_condiciones(), ui.aviso(por_que, None, "Por qué el veredicto es ese"), ir]


def _frente(tarea):
    """El modelo original frente al mejor benchmark, prueba por prueba, y frente a cada uno de los demás."""
    if D.veredicto_original(tarea) is None or D.pruebas_original(tarea) is None:
        return [OG._pendiente("Las pruebas de hipótesis del modelo original", "pruebas_hipotesis.csv", "las calcula 04_pruebas_hipotesis.py")]
    piezas = [OG._detalle_mejor(tarea), OG._tabla_benchmarks(tarea)]
    frase = OG._frase_p4_empates() if tarea == "clasificacion" else None     # P4 marca la campaña, y el corte de BDL cae en un empate
    if frase:
        piezas.append(ui.nota([f"{frase} El detalle está en la tarjeta «P4 y los empates de la campaña» de ",
                               dcc.Link("la vista del modelo original", href=app.ruta_vista("original") + "?tarea=clasificacion"), "."]))
    return piezas


def _nemenyi(tarea):
    """Friedman → Nemenyi sobre los pliegues externos (datos/cd.json, con los diez modelos cuando la corrección ya los dejó):
    rangos medios de menor a mayor (en cada pliegue, 1 es la mejor métrica), la diferencia crítica, los modelos a menos de
    ella del mejor rango medio, los que Nemenyi separa del modelo en foco y los rangos de cada pliegue, recalculados de las
    métricas de cada pliegue para comprobar el archivo."""
    n = D.json(CD_JSON)[tarea]
    rangos = pd.Series(n["rango_medio"], dtype=float).sort_values(kind="stable")
    mejor, dc = float(rangos.min()), float(n["diferencia_critica"])
    P = pd.DataFrame(n["metricas_pliegue"])
    R = P.rank(axis=1, ascending=not TAREA[tarea]["mayor_es_mejor"], method="average")
    coincide = bool((R.mean().reindex(rangos.index) - rangos).abs().max() < TOLERANCIA_RANGO)
    foco = _foco(rangos.index)
    return {"datos": n, "rangos": rangos, "mejor": mejor, "dc": dc, "por_pliegue": R, "coincide": coincide, "foco": foco,
            "dentro": [m for m in rangos.index if rangos[m] - mejor < dc],
            "separa": [m for m in rangos.index if m != foco and abs(rangos[m] - rangos[foco]) >= dc]}


def _aviso_rangos(n):
    if n["coincide"]:
        return None
    return ui.aviso(f"Los rangos medios de {D.fuente(CD_JSON)} no coinciden con los que se recalculan de sus métricas por pliegue: "
                    "revise el archivo antes de leer el diagrama.", "critico", "El diagrama no cuadra con sus datos")


def _diagrama_cd(tarea):
    """El diagrama de diferencia crítica de una tarea. Una fila por modelo con su rango medio en los pliegues externos; la
    franja va del mejor rango medio a ese rango más la diferencia crítica de Nemenyi, que se dibuja arriba como una barra."""
    c = TAREA[tarea]
    n = _nemenyi(tarea)
    datos, rangos, mejor, dc, foco = n["datos"], n["rangos"], n["mejor"], n["dc"], n["foco"]
    rol_de = _rol_diez(rangos.index)
    k, pliegues = int(datos["modelos"]), int(datos["pliegues"])
    orden = list(rangos.index)
    nombres = [D.NOMBRE_CORTO[m] for m in orden]
    filas_y = [FILA_DC] + nombres
    fig = go.Figure()
    texto_dc = (f"Diferencia crítica de Nemenyi: <b>{ui.num(dc, 2)}</b> rangos (α = {ui.num(datos['alfa'], 2)}), desde el mejor rango "
                f"medio, {ui.num(mejor, 1)}")
    fig.add_trace(go.Scatter(x=[mejor, mejor + dc], y=[FILA_DC, FILA_DC], mode="lines",
                             line=dict(color=ui.ROL["referencia"], width=ui.ANCHO_REAL), **ui.hover_texto([texto_dc, texto_dc])))
    for rol in ("base", "rival", "modelo"):
        grupo = [m for m in orden if rol_de[m] == rol]
        if not grupo:
            continue
        textos = [f"{D.NOMBRE_MODELO[m]}: rango medio <b>{ui.num(rangos[m], 1)}</b> · a {ui.num(rangos[m] - mejor, 1)} del mejor · "
                  + ("a menos de la diferencia crítica" if m in n["dentro"] else "más allá de la diferencia crítica") for m in grupo]
        fig.add_trace(go.Scatter(x=[rangos[m] for m in grupo], y=[D.NOMBRE_CORTO[m] for m in grupo], **_marca(rol),
                                 **ui.hover_texto(textos)))
    ui.franja(fig, mejor, mejor + dc)
    extremo, opuesto = (f"el mayor {c['metrica']}", "el menor") if c["mayor_es_mejor"] else (f"el menor {c['metrica']}", "el mayor")
    ui.unidad_eje(fig, f"rango medio (1 = {extremo})")       # corta: tiene que caber a 400 px; el subtítulo lo dice entero
    _lienzo_modelos(fig, filas_y, 1, k)
    valores, textos_eje = ui.marcas(1, k, 1)
    fig.update_xaxes(tickvals=valores, ticktext=textos_eje)
    _valores_al_borde(fig, filas_y, [ui.num(dc, 2)] + [ui.num(rangos[m], 1) for m in orden], foco=D.NOMBRE_CORTO[foco])

    separa, al_foco = n["separa"], _tras("a", D.en_frase(foco))
    titulo = (f"La diferencia crítica de Nemenyi es de {ui.num(dc, 2)} rangos: {_cardinal(len(n['dentro']))} de los {_cardinal(k)} "
              "modelos quedan a menos de ella del mejor"
              + (f" y Nemenyi no separa {al_foco} de ningún modelo" if not separa else
                 f"; frente {al_foco}, solo {'queda separado' if len(separa) == 1 else 'quedan separados'} {_lista(separa)}"))
    sub = (f"Rango medio de cada modelo en los {_cardinal(pliegues)} pliegues externos de la validación anidada: en cada pliegue, "
           f"1 es {extremo} y {k}, {opuesto}. La franja azul va del mejor rango medio ({ui.num(mejor, 1)}) a ese rango más la "
           f"diferencia crítica (α = {ui.num(datos['alfa'], 2)}); arriba, la barra de esa diferencia.")
    nota = ("Friedman y Nemenyi comparan todos los modelos a la vez, con los mismos pliegues. Con pocos pliegues Nemenyi tiene poca "
            f"potencia ({OG._cita_en('demsar2006')}): se presenta como descripción, y la regla de superioridad se decide con las pruebas "
            "por pares de la sección siguiente. A la derecha, el rango medio de cada modelo.")
    R = n["por_pliegue"]
    bloques = [str(b) for b in datos.get("bloques") or range(1, len(R) + 1)]
    filas = [[D.NOMBRE_MODELO[m], ui.num(rangos[m], 1), ui.num(rangos[m] - mejor, 1), "sí" if m in n["dentro"] else "no"]
             + [ui.num(R[m].iloc[i], 1) for i in range(len(R))] for m in orden]
    tabla = [ui.tabla(["Modelo", "Rango medio", "Distancia al mejor", "A menos de la diferencia crítica"]
                      + [f"Pliegue {b}" for b in bloques], filas, compacta=True, muestras=[rol_de[m] for m in orden],
                      num=[False, True, True, False] + [True] * len(R)),
             ui.nota(f"Rangos de cada pliegue recalculados de las métricas de {D.fuente(CD_JSON)} (los empates reparten el rango). Los "
                     "pliegues se nombran «semilla-pliegue».")]
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: rango medio {f[1]}" for f in filas) + f". Diferencia crítica: {ui.num(dc, 2)}."
    leyenda = _leyenda_diez(tarea, orden, extra=[("Diferencia crítica de Nemenyi", "referencia", "gruesa"),
                                                 ("A menos de la diferencia crítica del mejor rango medio", "franja", "barra")])
    return ui.grafico(fig, titulo=titulo, subtitulo=sub, etiqueta=etiqueta, tabla=tabla, leyenda=leyenda, nota=nota, alto="alto",
                      fuentes=FUENTES_CD)


def _tarjeta_friedman(tarea):
    """Las tres cifras de la prueba global (Friedman, Iman-Davenport, diferencia crítica) y su lectura honesta."""
    n = _nemenyi(tarea)
    datos, rangos, foco = n["datos"], n["rangos"], n["foco"]
    k, pliegues = int(datos["modelos"]), int(datos["pliegues"])
    gl = [int(g) for g in datos["gl"]]
    fuera = [m for m in rangos.index if m not in n["dentro"]]
    cifras = ui.cifras([
        ui.cifra(f"Friedman, χ² con {gl[0]} grados de libertad", ui.num(datos["chi2_friedman"], 2),
                 variacion=F.p_valor(datos["p_friedman"]), mini=True),
        ui.cifra(f"Iman-Davenport, F({gl[0]}; {gl[1]})", ui.num(datos["f_iman_davenport"], 2),
                 variacion=F.p_valor(datos["p_iman_davenport"]), mini=True),
        ui.cifra("Diferencia crítica de Nemenyi", ui.num(n["dc"], 2), "rangos", mini=True,
                 variacion=f"q = {ui.num(datos['q_alfa'], 3)} con α = {ui.num(datos['alfa'], 2)}",
                 nota=f"Con {_cardinal(k)} modelos y {_cardinal(pliegues)} pliegues; crece con los modelos y baja con la raíz de los "
                      "pliegues."),
    ])
    rechaza = datos["p_friedman"] < D.ALFA
    titulo = (f"Friedman {'rechaza' if rechaza else 'no rechaza'} que los {_cardinal(k)} modelos rindan igual "
              f"({F.p_valor(datos['p_friedman'])}); con {_cardinal(pliegues)} pliegues, Nemenyi tiene poca potencia")
    primera = (f"Solo {_lista(fuera)} {'quedan' if len(fuera) > 1 else 'queda'} a más de {ui.num(n['dc'], 2)} rangos del mejor rango "
               "medio." if fuera else f"Ningún modelo queda a más de {ui.num(n['dc'], 2)} rangos del mejor rango medio.")
    al_foco = _tras("a", D.en_frase(foco))
    if n["separa"]:
        segunda = f"Nemenyi separa {al_foco} {_tras('de', _lista(n['separa']))}."
    else:
        lejos = max((m for m in rangos.index if m != foco), key=lambda m: abs(rangos[m] - rangos[foco]))
        segunda = (f"Nemenyi no separa {al_foco} de ningún modelo: su mayor distancia, con {D.en_frase(lejos)}, es de "
                   f"{ui.num(abs(rangos[lejos] - rangos[foco]), 1)} rangos.")
    puesto = int((rangos < rangos[foco]).sum()) + 1
    tercera = (f"El rango medio {_tras('de', D.en_frase(foco))} es {ui.num(rangos[foco], 1)}: {_queda(foco, puesto)} de los "
               f"{_cardinal(k)}. Es una descripción: la regla de superioridad se decide con las pruebas por pares frente al mejor "
               "benchmark, con los p ajustados por Holm.")
    cuerpo = [cifras, ui.parrafo(f"{primera} {segunda} {tercera}")]
    aviso = _aviso_rangos(n)
    if aviso is not None:
        cuerpo.insert(0, aviso)
    return ui.tarjeta(cuerpo, titulo=titulo, fuentes=FUENTES_CD)


def _diez(tarea):
    """El bloque de todos los modelos a la vez de la tarea, con el modelo original incluido, y el subtítulo de su sección."""
    k = int(D.json(CD_JSON)[tarea]["modelos"])
    return ([_diagrama_cd(tarea), _tarjeta_friedman(tarea)],        # el diagrama a todo el ancho y debajo sus cifras, en una fila
            f"Sobre los mismos pliegues externos: la prueba de Friedman y el post hoc de Nemenyi con los {_cardinal(k)} modelos"
            + (", el modelo original entre ellos." if O in D.json(CD_JSON)[tarea]["rango_medio"] else "."))


def _tabla_mcs():
    """Regresión: los ocho modelos del Entregable 3 con su p del Model Confidence Set y si quedan en el conjunto al nivel del
    informe, ordenados de mayor a menor p; comprueba que la columna en_mcs y resumen.json dicen lo mismo que los p."""
    c = TAREA["regresion"]
    t = D.prueba("regresion").sort_values(["p_mcs", c["prueba"]], ascending=[False, True], kind="stable")
    resumen = D.resumen()
    umbral = 1 - NIVEL_MCS
    dentro = [m for m, esta in zip(t["modelo"], t["en_mcs"]) if bool(esta)]
    no_cuadra = [m for m, p, esta in zip(t["modelo"], t["p_mcs"], t["en_mcs"]) if bool(esta) != (p > umbral) and bool(esta) != (p >= umbral)]
    if set(dentro) != set(resumen.get("mcs_incluidos", dentro)):
        no_cuadra.append("resumen.json")
    filas = [[D.NOMBRE_MODELO[f["modelo"]], ui.num(f[c["prueba"]], c["d"]), _p(f["p_mcs"]),
              ui.chip("dentro", "bien") if bool(f["en_mcs"]) else ui.chip("fuera")] for _, f in t.iterrows()]
    nivel = F.pct(NIVEL_MCS, 0)
    tabla = ui.tabla(["Modelo", "RMSE de prueba", "p del MCS", f"En el conjunto al {nivel}"], filas,
                     muestras=[ROL[m] for m in t["modelo"]], num=[False, True, True, False])
    sub = (f"Los {CARDINAL[len(t)]} regresores del Entregable 3 sobre las {F.entero(resumen['n_prueba'])} pérdidas cuadráticas de la "
           f"prueba, con bootstrap estacionario de bloques de {CARDINAL.get(int(resumen['mcs_bloque']), F.entero(resumen['mcs_bloque']))}. "
           f"Quedan en el conjunto los modelos con p mayor que {ui.num(umbral, 2)}; van de mayor a menor p.")
    cuerpo = [tabla, ui.nota(f"El conjunto se fijó antes de evaluar la prueba. Un modelo con {F.p_valor(0)} sale del conjunto con mucha "
                             "evidencia en contra.")]
    if no_cuadra:
        cuerpo.insert(0, ui.aviso(f"La columna en_mcs, los p del MCS y resumen.json no dicen lo mismo ({', '.join(no_cuadra)}).",
                                  "critico", "La tabla no cuadra con sus datos"))
    return ui.tarjeta(cuerpo, titulo=_texto_mcs().rstrip("."), subtitulo=sub, fuentes=FUENTES_MCS)


def _tarjeta_mcs_lectura():
    """Regresión: el p de la EBM en el Model Confidence Set del Entregable 3 y cómo se lee junto a lo demás."""
    dentro, p_ebm, ultima = _mcs()
    t = D.prueba("regresion")
    cifras = ui.cifras([
        ui.cifra(f"p de la EBM en el MCS al {F.pct(NIVEL_MCS, 0)}", _p(p_ebm), mini=True,
                 variacion=("La EBM queda dentro." if "ebm" in dentro else "La EBM es la última en salir." if ultima else "La EBM queda fuera.")),
        ui.cifra("Modelos en el conjunto", F.entero(len(dentro)), f"de {F.entero(len(t))}", mini=True,
                 variacion=_mayuscula(_lista(dentro)) + "."),
    ])
    lectura = (f"El conjunto reúne a los modelos que no se distinguen del mejor en la prueba. Se calculó en el Entregable 3 con sus "
               f"{CARDINAL[len(t)]} modelos y es más exigente que las pruebas por pares. No se recalculó con el modelo original ni con "
               "LightGBM: para ellos, la comparación formal es la del protocolo de la corrección, en las secciones de arriba.")
    return ui.tarjeta([cifras, ui.parrafo(lectura)],
                      titulo="El Model Confidence Set es del Entregable 3: no incluye al modelo original ni a LightGBM", fuentes=FUENTES_MCS)


def _bloque_mcs(tarea):
    """El Model Confidence Set del Entregable 3, que solo existe en regresión."""
    return [ui.rejilla([_tabla_mcs(), _tarjeta_mcs_lectura()], "1-1")] if tarea == "regresion" else []


# --- 1. El desempeño de cada modelo en la prueba --------------------------------------------------------------------------
def _desempeno(tarea):
    """La métrica de prueba de cada modelo con su intervalo BCa, de mejor a peor; la franja es el intervalo del modelo en
    foco (el original, si ya está en la tarea)."""
    c = TAREA[tarea]
    t = D.lideres(tarea).sort_values(c["prueba"], ascending=not c["mayor_es_mejor"], kind="stable").reset_index(drop=True)
    orden = list(t["modelo"])
    foco, rol_de = _foco(orden), _rol_diez(orden)
    e = D.fila(t, modelo=foco)
    nombres = [D.NOMBRE_CORTO[m] for m in orden]
    fig = go.Figure()
    for rol in ("base", "rival", "modelo"):
        s = t[t["modelo"].map(rol_de) == rol]
        if s.empty:
            continue
        textos = [f"{D.NOMBRE_MODELO[f['modelo']]}: <b>{ui.num(f[c['prueba']], c['d'])}</b> {F.intervalo(f[c['inf']], f[c['sup']], c['d'])}"
                  f" · {c['externa_nombre']}: {ui.num(f['externa_media'], c['d'])}" for _, f in s.iterrows()]
        fig.add_trace(go.Scatter(x=s[c["prueba"]], y=[D.NOMBRE_CORTO[m] for m in s["modelo"]], **_marca(rol),
                                 error_x=_bigotes(s[c["prueba"]], s[c["inf"]], s[c["sup"]]), **ui.hover_texto(textos)))
    ui.franja(fig, e[c["inf"]], e[c["sup"]])
    ui.unidad_eje(fig, f"{c['metrica']} de prueba")
    ui.formato_eje(fig, "x", c["d_eje"])
    lo, hi = t[c["inf"]].min(), t[c["sup"]].max()
    _lienzo_modelos(fig, nombres, lo, hi)
    _valores_al_borde(fig, nombres, [ui.num(x, c["d"]) for x in t[c["prueba"]]], foco=D.NOMBRE_CORTO[foco])

    del_foco = _tras("de", D.en_frase(foco))
    solapan = [f["modelo"] for _, f in t.iterrows() if f[c["inf"]] <= e[c["sup"]] and f[c["sup"]] >= e[c["inf"]]]
    otro = D.fila(t, modelo=next(m for m in orden if m != foco))
    cuantos = (f"Los intervalos de los {_cardinal(len(solapan))} primeros modelos se solapan" if solapan == orden[:len(solapan)]
               else f"Los intervalos de {len(solapan) - 1} modelos se solapan con el {del_foco}")
    titulo = (f"{cuantos}: {D.en_frase(foco)}, {ui.num(e[c['prueba']], c['d'])}; {D.en_frase(otro['modelo'])}, "
              f"{ui.num(otro[c['prueba']], c['d'])}")
    sub = (f"{c['metrica']} en los {F.entero(D.resumen()['n_prueba'])} clientes de prueba, con su intervalo bootstrap BCa del 95 % "
           f"({F.entero(D.resumen()['n_boot'])} réplicas). La franja azul es el intervalo {del_foco}; a la derecha, el valor de cada "
           "modelo.")
    nota = (f"El eje no parte de cero: va de {ui.num(lo, c['d_eje'])} a {ui.num(hi, c['d_eje'])} para que se distingan los intervalos. "
            "Que dos intervalos se solapen no es una prueba: la comparación formal está en las pruebas de hipótesis de arriba.")
    externa = [f"{ui.num(f['externa_media'], c['d'])} ± {ui.num(f['externa_desv'], c['d'])}" for _, f in t.iterrows()]
    if tarea == "clasificacion":
        encabezados = ["Modelo", "Combinación", "AUC externo", "AUC de prueba [IC 95 %]", "AUC-PR"]
        filas = [[D.NOMBRE_MODELO[f["modelo"]], f"{D.NOMBRE_BALANCEO[f['balanceo']]}, {D.NOMBRE_OPTIMIZADOR[f['optimizador']]}", x,
                  f"{ui.num(f[c['prueba']], c['d'])} {F.intervalo(f[c['inf']], f[c['sup']], c['d'])}", ui.num(f["auc_pr"], 3)]
                 for (_, f), x in zip(t.iterrows(), externa)]
    else:
        encabezados = ["Modelo", "Optimizador", "RMSE externo", "RMSE de prueba [IC 95 %]", "MAE", "R² [IC 95 %]"]
        filas = [[D.NOMBRE_MODELO[f["modelo"]], D.NOMBRE_OPTIMIZADOR[f["optimizador"]], x,
                  f"{ui.num(f[c['prueba']], c['d'])} {F.intervalo(f[c['inf']], f[c['sup']], c['d'])}", ui.num(f["mae_prueba"], 4),
                  f"{ui.num(f['r2_prueba'], 3)} {F.intervalo(f['r2_bca_inf'], f['r2_bca_sup'], 3)}"]
                 for (_, f), x in zip(t.iterrows(), externa)]
    tabla = [ui.tabla(encabezados, filas, compacta=True, muestras=[rol_de[m] for m in orden],
                      num=[False, False] + [True] * (len(encabezados) - 2)),
             ui.nota("La métrica externa es la media ± la desviación de los pliegues externos de la validación anidada, los mismos para "
                     "todos los modelos.")]
    if tarea == "clasificacion":                 # el AUC-PR de un puntaje escalonado depende de cómo se cuenten los empates
        tabla += [ui.nota(x) for x in (D.frase_empate_auc_pr(m) for m in orden) if x]
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: {f[3]}" for f in filas) + "."
    return {"figura": fig, "titulo": titulo, "sub": sub, "tabla": tabla, "nota": nota, "etiqueta": etiqueta,
            "leyenda": _leyenda_diez(tarea, orden, extra=[(f"Intervalo {del_foco}", "franja", "barra")])}


def _acota(tarea):
    """Título y fichas de la tarjeta que acota el resultado de la prueba: el puesto del modelo en foco en la validación
    anidada y, para el modelo original, su ventaja sobre el mejor benchmark (veredicto.json)."""
    c = TAREA[tarea]
    t = D.lideres(tarea).sort_values("externa_media", ascending=not c["mayor_es_mejor"], kind="stable").reset_index(drop=True)
    foco = _foco(t["modelo"])
    e, primero = D.fila(t, modelo=foco), t.iloc[0]
    puesto = int(t.index[t["modelo"] == foco][0]) + 1
    lugar = f"Puesto {F.entero(puesto)} de {F.entero(len(t))} modelos"
    if puesto > 1:
        lugar += (f", a {ui.num(abs(primero['externa_media'] - e['externa_media']), c['d'])} {_tras('de', D.en_frase(primero['modelo']))} "
                  f"({ui.num(primero['externa_media'], c['d'])})")
    externa = ui.cifra(f"{c['externa_nombre']} {_tras('de', D.en_frase(foco))}", ui.num(e["externa_media"], c["d"]),
                       f"± {ui.num(e['externa_desv'], c['d'])}", variacion=lugar + ".", mini=True,
                       nota=f"Media y desviación de los {_cardinal(int(e['externa_pliegues']))} pliegues externos, los mismos para todos "
                            "los modelos, sin mirar la prueba.")
    titulo = f"En la validación anidada {D.en_frase(foco)} {_queda(foco, puesto)}"
    r = D.veredicto_original(tarea) if foco == O else None
    if r is not None:
        dm = r["frente_al_mejor"]
        segunda = ui.cifra(f"Ventaja de {c['externa_nombre']} sobre {D.en_frase(r['mejor_benchmark'])}, el mejor benchmark",
                           ui.con_signo(r["ventaja_sobre_el_mejor"], c["d"] + 1),
                           variacion=f"A favor de BDL en {F.entero(dm['pliegues_a_favor'])} de {F.entero(r['pliegues'])} pliegues.",
                           nota=f"Positiva: a favor de BDL. Veredicto con la regla de la guía: {D.frase_veredicto(tarea)}.", mini=True)
        titulo += f"; con la regla de la guía, {D.frase_veredicto(tarea, detalle=False)}"
    else:
        segunda = ui.cifra(f"{c['metrica']} de prueba {_tras('de', D.en_frase(foco))}", ui.num(e[c["prueba"]], c["d"]),
                           variacion=f"Intervalo del 95 %: {F.intervalo(e[c['inf']], e[c['sup']], c['d'])}.", mini=True)
    return titulo, [externa, segunda]


# --- 2. La EBM frente a cada referencia -----------------------------------------------------------------------------------
def _diferencias(tarea):
    """La diferencia de la EBM con cada referencia, con su intervalo BCa, sobre la banda de equivalencia práctica."""
    c = TAREA[tarea]
    t, v = D.pruebas(tarea), D.veredictos(tarea)
    rope = D.resumen()["rope"][c["rope"]]
    orden = list(t["modelo"])
    nombres = [D.NOMBRE_CORTO[m] for m in orden]
    fig = go.Figure()
    for rol in ("base", "rival"):
        s = t[t["modelo"].map(ROL) == rol]
        if s.empty:
            continue
        textos = [f"{D.NOMBRE_MODELO[f['modelo']]}: <b>{ui.con_signo(f[c['delta']], c['d'])}</b> "
                  f"{F.intervalo(f['delta_bca_inf'], f['delta_bca_sup'], c['d'], signo=True)} · {D.VEREDICTO_TEXTO[v[f['modelo']]]}"
                  for _, f in s.iterrows()]
        fig.add_trace(go.Scatter(x=s[c["delta"]], y=[D.NOMBRE_CORTO[m] for m in s["modelo"]], **_marca(rol),
                                 error_x=_bigotes(s[c["delta"]], s["delta_bca_inf"], s["delta_bca_sup"]), **ui.hover_texto(textos)))
    ui.franja(fig, -rope, rope)
    ui.linea_cero(fig, "x")
    ui.unidad_eje(fig, f"diferencia de {c['metrica']} a favor de la EBM")
    lo, hi = min(t["delta_bca_inf"].min(), -rope), max(t["delta_bca_sup"].max(), rope)
    _lienzo_modelos(fig, nombres, lo, hi)
    valores, textos_eje = ui.marcas(lo, hi, ui.paso_redondo(hi, MARCAS_EJE), signo=True)
    fig.update_xaxes(tickvals=valores, ticktext=textos_eje)
    _valores_al_borde(fig, nombres, [ui.con_signo(x, c["d"]) for x in t[c["delta"]]])

    filas_t = [f for _, f in t.iterrows()]
    dentro = [f["modelo"] for f in filas_t if abs(f[c["delta"]]) <= rope]
    fuera = [f for f in filas_t if f["modelo"] not in dentro]
    asoman = [f for f in filas_t if f["modelo"] in dentro and (f["delta_bca_inf"] < -rope or f["delta_bca_sup"] > rope)]
    lejos = all(f["delta_bca_inf"] > rope or f["delta_bca_sup"] < -rope for f in fuera)
    titulo = f"{len(dentro)} de las {len(t)} diferencias caen dentro de la banda de equivalencia"
    if fuera:
        titulo += f"; las otras {len(fuera)} quedan fuera" + (" con todo su intervalo" if lejos else "")
    banda = f"±{ui.num(rope, c['d_rope'])} de {c['metrica']}"
    sub = (f"{c['resta']} en la prueba, con su intervalo bootstrap BCa del 95 %: un valor positivo favorece a la EBM. La franja azul es "
           f"la banda de equivalencia práctica ({banda}): una diferencia dentro de ella no cambia una decisión.")
    nota = "A la derecha, la diferencia de cada referencia. La banda se fijó antes de ejecutar la prueba bayesiana que la usa."
    if asoman:
        cuales = (f"El intervalo {_tras('de', _lista(f['modelo'] for f in asoman))} asoma" if len(asoman) == 1
                  else f"Los intervalos {_tras('de', _lista(f['modelo'] for f in asoman))} asoman")
        hasta = max((x for f in asoman for x in (f["delta_bca_inf"], f["delta_bca_sup"])), key=abs)
        nota = f"{cuales} fuera de la banda (llega a {ui.con_signo(hasta, c['d'])}), aunque su valor puntual queda dentro. " + nota

    mcs = D.prueba(tarea).set_index("modelo")["p_mcs"] if tarea == "regresion" else None
    filas = []
    for f in filas_t:          # lo que decide el veredicto va primero: la tabla se desplaza hacia la derecha en su panel
        fila = [D.NOMBRE_MODELO[f["modelo"]], D.VEREDICTO_TEXTO[v[f["modelo"]]],
                f"{ui.con_signo(f[c['delta']], c['d'])} {F.intervalo(f['delta_bca_inf'], f['delta_bca_sup'], c['d'], signo=True)}",
                _prob(f["p_rope"]), f"{ui.num(f['t_corr'], 2)}; {_p(f['p_t_holm'])}",
                f"{ui.num(f[c['est_fijada']], 2)}; {_p(f[c['p_fijada']])}"]
        if tarea == "clasificacion":
            fila.append(f"{F.entero(f['disc_base_bien'])} / {F.entero(f['disc_ebm_bien'])}; {_p(f['p_mcnemar_holm'])}")
        else:
            fila.append(_p(mcs[f["modelo"]]))
        filas.append(fila)
    tercera = "McNemar: b / c; p" if tarea == "clasificacion" else "p del MCS"
    encabezados = ["Referencia", "Veredicto", f"Δ{c['metrica']} [IC 95 %]", "Equivalencia práctica", "t corregida: t; p",
                   f"{c['fijada']}: {c['simbolo']}; p", tercera]
    aclaracion = (f"Los p de la t corregida y de {c['fijada']} son bilaterales y están ajustados por Holm sobre las "
                  f"{CARDINAL[len(t)]} comparaciones. La equivalencia práctica es una probabilidad posterior, no un valor p.")
    if tarea == "clasificacion":
        aclaracion = (f"Los p son bilaterales y están ajustados por Holm sobre las {CARDINAL[len(t)]} comparaciones de cada prueba. En "
                      "McNemar, b son los clientes que solo la referencia clasifica bien y c, los que solo clasifica bien la EBM. La "
                      "equivalencia práctica es una probabilidad posterior, no un valor p.")
    tabla = [ui.tabla(encabezados, filas, compacta=True, muestras=[ROL[m] for m in orden],
                      num=[False, False, True, True, True, True, True]), ui.nota(aclaracion)]
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: {f[2]}, {f[1]}" for f in filas) + "."
    return {"figura": fig, "titulo": titulo, "sub": sub, "tabla": tabla, "nota": nota, "etiqueta": etiqueta,
            "leyenda": _leyenda_modelos(tarea, con_ebm=False, extra=("Banda de equivalencia práctica", "franja", "barra"))}


def _resumen_veredictos(tarea):
    """El subtítulo de la sección de diferencias: cuántas referencias supera la EBM y con cuántas empata."""
    c = TAREA[tarea]
    v = D.veredictos(tarea)
    n_sup, n_emp = int((v == "supera").sum()), int((v == "empata").sum())
    texto = (f"La EBM supera con significación, tras la corrección de Holm, a {n_sup} de las {len(v)} referencias; frente a "
             f"{'las otras ' if n_sup + n_emp == len(v) else ''}{n_emp}, ni la t corregida ni {c['fijada']} encuentran diferencia.")
    if n_sup + n_emp < len(v):
        texto += f" En {len(v) - n_sup - n_emp} el veredicto es otro."
    return texto


# --- 3. Ficha de una referencia -------------------------------------------------------------------------------------------
def _ficha(tarea, ref):
    """Título y cuerpo de la ficha de una referencia: veredicto calculado, pruebas, reparto bayesiano y tamaños de efecto."""
    c = TAREA[tarea]
    pr = D.pruebas(tarea)
    f = D.fila(pr, modelo=ref)
    v = D.veredicto(f, tarea)
    frase = D.en_frase(ref)               # el modelo dentro de una frase: «el árbol de decisión», «la regresión logística»
    rope = D.resumen()["rope"][c["rope"]]
    banda = f"±{ui.num(rope, c['d_rope'])} de {c['metrica']}"
    pliegues = CARDINAL[_n_pliegues(tarea)]
    capacidad = F.pct(M.CAPACIDAD_INFORME, 0)
    umbral = F.pct(UMBRAL_EQUIVALENCIA, 0)
    p_t, p_f = f["p_t_holm"], f[c["p_fijada"]]
    equivalencia, llega = _prob(f["p_rope"]), f["p_rope"] >= UMBRAL_EQUIVALENCIA
    # (nombre en la tabla, nombre en una frase, protocolo, estadístico, p sin corregir, p ajustado por Holm)
    pruebas = [("t corregida", "la t corregida", EXPLORATORIA, f"t = {ui.num(f['t_corr'], 2)}", f["p_t"], p_t),
               (c["fijada"], c["fijada"], FIJADA, f"{c['simbolo']} = {ui.num(f[c['est_fijada']], 2)}", f[c["p_fijada_cruda"]], p_f)]
    if tarea == "clasificacion":
        b, solo_ebm = F.entero(f["disc_base_bien"]), F.entero(f["disc_ebm_bien"])
        pruebas.append(("McNemar", "McNemar", EXPLORATORIA, f"b / c = {b} / {solo_ebm}", f["p_mcnemar"], f["p_mcnemar_holm"]))
    solo_crudas = [(corto, p) for _l, corto, _pr, _e, p, ph in pruebas if p < D.ALFA <= ph]

    # --- el veredicto, en palabras
    par = f"({F.p_valor(p_t)} y {F.p_valor(p_f)}, ajustados)"
    incluye_cero = f["delta_bca_inf"] <= 0 <= f["delta_bca_sup"]
    if v == "empata":
        crudas = [(corto, p) for corto, p in solo_crudas if corto != "McNemar"]
        titulo = (f"Sin evidencia de diferencia con {frase}: equivalencia práctica {equivalencia}" if incluye_cero and not crudas
                  else f"La diferencia con {frase} no sobrevive a la corrección de Holm: equivalencia práctica {equivalencia}")
        frases = [f"Ni la t corregida ni {c['fijada']} rechazan la igualdad tras la corrección de Holm {par}."]
        if crudas:
            partes = ([] if incluye_cero else ["el intervalo de la diferencia no incluye el cero"]) + [
                f"{corto} rechazaría sin corregir ({F.p_valor(p)})" for corto, p in crudas]
            frases.append(_mayuscula(F.enumerar(partes)) + ", pero la señal no sobrevive a la corrección de Holm: no se cuenta como "
                          "diferencia.")
        else:
            frases.append("El intervalo de la diferencia incluye el cero." if incluye_cero else
                          "El intervalo de la diferencia no incluye el cero, pero ninguna de las dos pruebas lo confirma.")
    elif v == "discrepan":
        titulo = f"Frente {_tras('a', frase)} las pruebas discrepan"
        frases = [f"Tras la corrección de Holm solo una de las dos pruebas rechaza la igualdad {par}: no se cuenta como diferencia "
                  "establecida."]
    else:
        favor = D.en_frase("ebm") if v == "supera" else frase
        titulo = (f"La EBM supera {_tras('a', frase)} con significación tras la corrección de Holm" if v == "supera"
                  else f"{_mayuscula(frase)} supera a la EBM con significación tras la corrección de Holm")
        frases = [f"La t corregida y {c['fijada']} rechazan la igualdad a favor {_tras('de', favor)} tras la corrección de Holm {par}.",
                  "El intervalo de la diferencia no incluye el cero." if not incluye_cero else "El intervalo de la diferencia incluye el cero."]
    if v in ("empata", "discrepan"):
        frases.append(f"La probabilidad posterior de que la diferencia quede dentro de la banda de {banda} es {equivalencia}: "
                      + (f"pasa del {umbral} habitual." if llega else f"no llega al {umbral} habitual.")
                      + " Es una probabilidad de equivalencia, no un valor p.")
    else:
        frases.append(f"La probabilidad posterior de que {favor} sea mejor, más allá de la banda de {banda}, es "
                      f"{_prob(f['p_mejor'] if v == 'supera' else f['p_peor'], prosa=True)}.")
    chips = [ui.chip(D.VEREDICTO_TEXTO[v], TIPO_VEREDICTO[v]),
             ui.chip_valor(f"Δ{c['metrica']}:", f"{ui.con_signo(f[c['delta']], c['d'])} "
                                                f"{F.intervalo(f['delta_bca_inf'], f['delta_bca_sup'], c['d'], signo=True)}")]
    if v in ("empata", "discrepan"):
        chips.append(ui.chip(f"equivalencia {equivalencia}: pasa del {umbral}", "bien") if llega
                     else ui.chip(f"equivalencia {equivalencia}: no llega al {umbral}", "atencion"))
    if tarea == "clasificacion":
        separa = f["p_mcnemar_holm"] < D.ALFA
        ajustado = f"{F.p_valor(f['p_mcnemar_holm'])} ajustado"
        if v == "supera" and not separa:
            chips.append(ui.chip("McNemar no los separa", "atencion"))
            frases.append(f"McNemar es la excepción: con el {capacidad} de capacidad no los separa ({ajustado}), porque solo "
                          f"{frase} clasifica bien a {b} clientes y solo la EBM a {solo_ebm}.")
        elif v == "supera":
            chips.append(ui.chip("McNemar también los separa", "bien"))
            frases.append(f"McNemar también los separa con el {capacidad} de capacidad ({ajustado}): solo la EBM clasifica bien a "
                          f"{solo_ebm} clientes y solo {frase} a {b}.")
        else:
            frases.append(f"McNemar, con el {capacidad} de capacidad, {'sí' if separa else 'tampoco'} los separa ({ajustado}).")
    chips += [ui.chip(f"{_mayuscula(corto)}: rechaza sin corregir", "atencion") for corto, _p_cruda in solo_crudas]

    # --- las pruebas, una por fila
    filas = [[[html.Span(largo), html.Br(), html.Span(protocolo, className="tinta-3")], estadistico, _p(p), _p(ph),
              "rechaza" if ph < D.ALFA else "no rechaza"] for largo, _c, protocolo, estadistico, p, ph in pruebas]
    tabla = ui.tabla(["Prueba", "Estadístico", "p sin corregir", "p ajustado", "Decisión"], filas, compacta=True,
                     num=[False, True, True, True, True], ocultar_movil=(1, 2))
    aclaracion = (f"La t corregida de Nadeau-Bengio compara los {pliegues} pliegues externos y {c['fijada']}, los clientes de prueba"
                  + (f"; McNemar exacta, las clases con el {capacidad} de capacidad (b: clientes que solo {frase} clasifica bien; "
                     "c: solo la EBM)" if tarea == "clasificacion" else "")
                  + f". Los p son bilaterales; el ajustado lleva la corrección de Holm sobre las {CARDINAL[len(pr)]} comparaciones de "
                    f"cada prueba y la decisión usa α = {ui.num(D.ALFA, 2)}.")
    cuerpo = [html.Div(chips, className="chips"), ui.parrafo(" ".join(frases)), tabla, ui.nota(aclaracion)]

    # --- el reparto bayesiano: la lectura más probable da el título
    lecturas = [(f["p_peor"], "La EBM es peor"), (f["p_rope"], f"Equivalentes (dentro de {banda})"), (f["p_mejor"], "La EBM es mejor")]
    mayor = max(range(len(lecturas)), key=lambda k: lecturas[k][0])
    titulo_bayes = (("La EBM es peor", "Equivalentes", "La EBM es mejor")[mayor]
                    + f" con probabilidad {_prob(lecturas[mayor][0], prosa=True)}, según la prueba bayesiana")
    bayes = [html.Div([ui.barra_rayas(rotulo, ui.num(p, DECIMALES_REPARTO), p) for p, rotulo in lecturas], className="rayas-lista"),
             ui.nota(f"Prueba t correlacionada bayesiana sobre los {pliegues} pliegues: posterior y exploratoria, con la banda fijada antes de "
                     "ejecutarla. Son probabilidades posteriores de cada lectura de la diferencia, no valores p.")]

    # --- los tamaños de efecto: el delta de Cliff entre pliegues da el título
    cliff = f["delta_cliff"]
    if abs(cliff) == 1:
        orden_pliegues = (f"la EBM queda siempre por delante {_tras('de', frase)}" if cliff > 0
                          else f"{frase} queda siempre por delante de la EBM")
    elif cliff == 0:
        orden_pliegues = f"ni la EBM ni {frase} quedan por delante"
    else:
        orden_pliegues = (f"la EBM tiende a quedar por delante {_tras('de', frase)}" if cliff > 0
                          else f"{frase} tiende a quedar por delante de la EBM")
    titulo_efectos = f"En los pliegues {orden_pliegues} (δ de Cliff {ui.con_signo(cliff, 2)})"
    pares = [("d_z de Cohen entre pliegues", ui.num(f["d_z_pliegues"], 2))]
    if tarea == "clasificacion":
        pares += [("δ de Cliff entre pliegues", ui.con_signo(cliff, 2)),
                  ("Clientes que solo clasifica bien la EBM", solo_ebm),
                  (f"Clientes que solo clasifica bien {frase}", b),
                  ("Tasa de error de la EBM", F.pct(f["err_ebm_op"], 1)),
                  (f"Tasa de error {_tras('de', frase)}", F.pct(f["err_base_op"], 1))]
        lectura = (f"d_z y δ son positivos cuando la EBM tiene más AUC que la referencia; un δ de {ui.con_signo(1, 2)} dice que queda por delante "
                   f"en todas las comparaciones entre pliegues. Los clientes y las tasas de error son los del punto de operación del "
                   f"{capacidad}, donde se aplica McNemar.")
    else:
        p_mcs = D.fila(D.prueba(tarea), modelo=ref)["p_mcs"]
        pares += [("d_z de Cohen por cliente", ui.num(f["d_z_clientes"], 3)),
                  ("δ de Cliff entre pliegues", ui.con_signo(cliff, 2)),
                  (f"Diferencia de R² ({D.en_frase('ebm')} menos {frase})", ui.con_signo(f["delta_r2"], 4)),
                  (f"p {_tras('de', frase)} en el Model Confidence Set", _p(p_mcs))]
        lectura = (f"d_z y δ son positivos cuando la EBM tiene menos error que la referencia; un δ de {ui.con_signo(1, 2)} dice que "
                   "queda por delante en todas las comparaciones entre pliegues. El d_z por cliente compara las pérdidas cuadráticas "
                   "de la prueba, cliente a cliente.")
    efectos = [ui.ficha(pares), ui.nota(lectura)]
    return {"titulo": titulo, "cuerpo": cuerpo, "titulo_bayes": titulo_bayes, "bayes": bayes, "titulo_efectos": titulo_efectos,
            "efectos": efectos}


@lru_cache(maxsize=None)
def _curva(modelo):
    """Curva ROC de un modelo final en la prueba, en tuplas (se comparte entre hilos sin riesgo de que alguien la cambie)."""
    P = D.puntajes()
    curva = M.curva_roc(P["y"], P[modelo])
    return tuple(float(x) for x in curva["fpr"]), tuple(float(x) for x in curva["tpr"])


def _roc(ref):
    """Clasificación: la curva ROC de la EBM y de la referencia en la prueba, con la diagonal del azar."""
    c = TAREA["clasificacion"]
    t = D.prueba("clasificacion")
    v = D.veredicto(D.fila(D.pruebas("clasificacion"), modelo=ref), "clasificacion")
    nombre, rol, frase = D.NOMBRE_MODELO[ref], ROL[ref], D.en_frase(ref)
    auc_ebm, auc_ref = ui.num(D.fila(t, modelo="ebm")[c["prueba"]], c["d"]), ui.num(D.fila(t, modelo=ref)[c["prueba"]], c["d"])
    fpr, tpr_ebm = _curva("ebm")
    tpr_ref = _curva(ref)[1]
    dif = [a - b for a, b in zip(tpr_ebm, tpr_ref)]
    i = max(range(len(dif)), key=lambda k: abs(dif[k]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], hoverinfo="skip", **ui.traza("contexto")))
    fig.add_trace(go.Scatter(x=fpr, y=tpr_ref, **ui.traza(rol), **ui.hover(D.NOMBRE_CORTO[ref], tpr_ref, d=3)))
    fig.add_trace(go.Scatter(x=fpr, y=tpr_ebm, **ui.traza("modelo"), **ui.hover(D.NOMBRE_CORTO["ebm"], tpr_ebm, d=3)))
    fig.add_annotation(x=AZAR_ROTULO, y=AZAR_ROTULO, text="azar", xanchor="left", yanchor="top", showarrow=False)
    ui.unidad_eje(fig, "fracción de clientes de riesgo alto alcanzados")
    valores, textos_eje = ui.marcas(0, 1, PASO_ROC)
    recorrido = [-MARGEN_ROC, 1 + MARGEN_ROC]
    fig.update_xaxes(range=recorrido, tickvals=valores, ticktext=textos_eje, hoverformat=",.3f")
    fig.update_yaxes(range=recorrido, tickvals=valores, ticktext=textos_eje)

    puntos = ui.num(abs(dif[i]) * 100, 1)
    if v == "supera":
        titulo = f"La curva de la EBM queda hasta {ui.num(max(dif) * 100, 1)} puntos porcentuales por encima de la {_tras('de', frase)}"
    elif v == "pierde":
        titulo = f"La curva {_tras('de', frase)} queda hasta {ui.num(-min(dif) * 100, 1)} puntos porcentuales por encima de la de la EBM"
    else:
        titulo = f"Las curvas de la EBM y {_tras('de', frase)} no se separan más de {puntos} puntos porcentuales"
    sub = (f"Curva ROC en los {F.entero(D.resumen()['n_prueba'])} clientes de prueba. En horizontal, la fracción de clientes de riesgo "
           f"bajo que el modelo marca por error; en vertical, la de clientes de riesgo alto que alcanza. AUC: la EBM, {auc_ebm}; "
           f"{frase}, {auc_ref}.")
    nota = (f"La separación se mide en puntos porcentuales de clientes de riesgo alto alcanzados; la mayor está en "
            f"{ui.num(fpr[i], 2)} del eje horizontal. La diagonal es el azar (AUC {ui.num(AZAR_AUC, 1)}).")
    if ref == "svm":
        nota += " El puntaje del SVM lineal es una función de decisión, no una probabilidad: la curva solo usa el orden."
    filas = [[ui.num(fpr[k], 2), ui.num(tpr_ebm[k], 3), ui.num(tpr_ref[k], 3), ui.con_signo(dif[k], 3)]
             for k in range(0, len(fpr), PASO_TABLA_ROC)]
    tabla = ui.tabla(["Riesgo bajo marcado por error", D.NOMBRE_CORTO["ebm"], D.NOMBRE_CORTO[ref], "Diferencia"], filas, compacta=True,
                     alta=True)
    etiqueta = f"{titulo}. AUC de prueba: la EBM, {auc_ebm}; {frase}, {auc_ref}."
    leyenda = ui.leyenda([(D.NOMBRE_MODELO["ebm"], "modelo", "linea"), (nombre, rol, "trazos" if rol == "base" else "linea"),
                          ("Azar", "contexto", "linea")])
    return {"figura": fig, "titulo": titulo, "sub": sub, "tabla": tabla, "nota": nota, "etiqueta": etiqueta, "leyenda": leyenda}


def _pliegues_par(ref):
    """Regresión: el RMSE de la EBM y de la referencia en cada pliegue externo (lo que compara la t corregida)."""
    c = TAREA["regresion"]
    P = D.tabla(c["pliegues"])
    t = D.prueba("regresion")
    nombre, rol, frase = D.NOMBRE_MODELO[ref], ROL[ref], D.en_frase(ref)
    ebm, otro = list(P["ebm"]), list(P[ref])
    n = len(P)
    rotulos = [f"Pliegue {k + 1}" for k in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rotulos, y=otro, **_linea(rol), **ui.hover(D.NOMBRE_CORTO[ref], otro, d=c["d"])))
    fig.add_trace(go.Scatter(x=rotulos, y=ebm, **_linea("modelo"), **ui.hover(D.NOMBRE_CORTO["ebm"], ebm, d=c["d"])))
    ui.unidad_eje(fig, f"{c['externa_nombre']} de cada pliegue")
    ui.formato_eje(fig, "y", c["d"] - 1)

    menos = sum(a < b for a, b in zip(ebm, otro))
    if menos == n:
        titulo = f"La EBM tiene menos {c['metrica']} que {frase} en los {n} pliegues"
    elif menos == 0:
        titulo = f"{_mayuscula(frase)} tiene menos {c['metrica']} que la EBM en los {n} pliegues"
    else:
        titulo = f"La EBM tiene menos {c['metrica']} que {frase} en {menos} de los {n} pliegues"
    medias = (ui.num(D.fila(t, modelo="ebm")[c["externa"]], c["d"]), ui.num(D.fila(t, modelo=ref)[c["externa"]], c["d"]))
    sub = (f"{c['metrica']} de cada pliegue externo de la validación anidada, con los mismos pliegues para los dos modelos: es lo que "
           f"compara la t corregida. Media: la EBM, {medias[0]}; {frase}, {medias[1]}.")
    nota = (f"El eje no parte de cero. Con {CARDINAL[n]} pliegues la t corregida tiene {n - 1} grados de libertad y poca potencia: "
            "por eso una ventaja en todos los pliegues puede no bastar tras la corrección de Holm.")
    filas = [[rotulos[k], ui.num(ebm[k], c["d"]), ui.num(otro[k], c["d"]), ui.con_signo(otro[k] - ebm[k], c["d"])] for k in range(n)]
    tabla = ui.tabla(["Pliegue", D.NOMBRE_CORTO["ebm"], D.NOMBRE_CORTO[ref], "Diferencia a favor de la EBM"], filas, compacta=True)
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: la EBM, {f[1]}; {frase}, {f[2]}" for f in filas) + "."
    leyenda = ui.leyenda([(D.NOMBRE_MODELO["ebm"], "modelo", "linea"), (nombre, rol, "trazos" if rol == "base" else "linea")])
    return {"figura": fig, "titulo": titulo, "sub": sub, "tabla": tabla, "nota": nota, "etiqueta": etiqueta, "leyenda": leyenda}


def _avisos(tarea):
    """Lo que hay que saber para leer las pruebas: qué se fijó antes y la prueba que se aparta del resto."""
    c = TAREA[tarea]
    pr, v = D.pruebas(tarea), D.veredictos(tarea)
    rope = D.resumen()["rope"][c["rope"]]
    comunes = (f"el {c['metrica']} como métrica principal, α = {ui.num(D.ALFA, 2)}, la corrección de Holm sobre las {CARDINAL[len(pr)]} "
               "comparaciones")
    banda = f"con su banda de ±{ui.num(rope, c['d_rope'])} de {c['metrica']} fijada antes de ejecutarla"
    if tarea == "clasificacion":
        protocolo = [f"Fijados antes de evaluar la prueba: {comunes} y la prueba de {c['fijada']}.",
                     f"Posteriores y exploratorias: la t corregida, McNemar y la bayesiana ({banda})."]
        capacidad = F.pct(M.CAPACIDAD_INFORME, 0)
        separa = [m for m, p in zip(pr["modelo"], pr["p_mcnemar_holm"]) if p < D.ALFA]
        resto = [m for m in v.index if v[m] == "supera" and m not in separa]
        texto = (f"Con el {capacidad} de capacidad, McNemar solo separa {_tras('a', _lista(separa))}" if separa
                 else f"Con el {capacidad} de capacidad, McNemar no separa a ninguna referencia")
        if resto:
            alcance = D.prueba(tarea).set_index("modelo")["vp_op"]
            ps = [ui.num(D.fila(pr, modelo=m)["p_mcnemar_holm"], 3) for m in resto]
            ajustado = f"el p ajustado es {ps[0]}" if len(set(ps)) == 1 else f"el p ajustado no baja de {min(ps)}"
            texto += (f"; frente {_tras('a', _lista(resto))}, que la EBM supera en las otras pruebas tras la corrección de Holm, {ajustado}. "
                      f"A esa capacidad la EBM alcanza a {F.entero(alcance['ebm'])} clientes de riesgo alto y esos modelos alcanzan "
                      f"entre {F.entero(min(alcance[m] for m in resto))} y {F.entero(max(alcance[m] for m in resto))}: "
                      "la ventaja de AUC aparece en otras capacidades.")
        else:
            texto += "."
        excepcion = ui.aviso(texto, "atencion", "McNemar es la excepción")
    else:
        protocolo = [f"Fijados antes de evaluar la prueba: {comunes}, {c['fijada']} y el Model Confidence Set.",
                     f"Posteriores y exploratorias: la t corregida y la bayesiana ({banda})."]
        dentro = _mcs()[0]
        texto = _texto_mcs()
        if dentro and "ebm" not in dentro:
            f = D.fila(pr, modelo=dentro[0])
            texto += (f" Detecta una ventaja {_tras('de', D.en_frase(dentro[0]))} de {ui.num(abs(f[c['delta']]), c['d'])} de {c['metrica']}"
                      + (f" que ni la t corregida ni {c['fijada']} confirman al {F.pct(D.ALFA, 0)} tras la corrección de Holm."
                         if v[dentro[0]] == "empata" else "."))
        excepcion = ui.aviso(texto, "atencion", "El Model Confidence Set es más exigente")
    return ui.rejilla([ui.aviso(protocolo, None, "Qué se fijó antes y qué es exploratorio"), excepcion], "1-1")


# --- 4. Los pliegues de la validación anidada ---------------------------------------------------------------------------
def _matriz_pliegues(tarea):
    """La métrica de cada pliegue externo (filas) de cada modelo (columnas) y el rótulo de cada pliegue: la de la corrección
    (con todos los modelos y todas las repeticiones, «semilla-pliegue») o, si todavía no está, la del Entregable 3."""
    P = D.pliegues(tarea)
    if P is not None and set(D.lideres(tarea)["modelo"]) <= set(P.columns):
        return P, [str(i) for i in P.index], D.fuente_original(f"pliegues_{tarea}.csv")
    P = D.tabla(TAREA[tarea]["pliegues"])
    return P, [str(k + 1) for k in range(len(P))], D.fuente(TAREA[tarea]["pliegues"])


def _pliegues(tarea):
    """Una fila por modelo con la métrica de cada pliegue externo; la franja es el recorrido del modelo en foco."""
    c = TAREA[tarea]
    P, rotulos, _fuente = _matriz_pliegues(tarea)
    t = D.lideres(tarea)
    t = t[t["modelo"].isin(P.columns)].sort_values(c["prueba"], ascending=not c["mayor_es_mejor"], kind="stable")
    orden = list(t["modelo"])
    foco, rol_de = _foco(orden), _rol_diez(orden)
    n = len(P)
    media, desv = P[orden].mean(), P[orden].std(ddof=1)
    nombres = [D.NOMBRE_CORTO[m] for m in orden]
    fig = go.Figure()
    for rol in ("base", "rival", "modelo"):
        grupo = [m for m in orden if rol_de[m] == rol]
        if not grupo:
            continue
        textos = [f"{D.NOMBRE_MODELO[m]} · pliegue {rotulos[k]}: <b>{ui.num(P[m].iloc[k], c['d'])}</b>" for m in grupo for k in range(n)]
        fig.add_trace(go.Scatter(x=[P[m].iloc[k] for m in grupo for k in range(n)],
                                 y=[D.NOMBRE_CORTO[m] for m in grupo for _k in range(n)], **_marca(rol), **ui.hover_texto(textos)))
    ui.franja(fig, P[foco].min(), P[foco].max())
    ui.unidad_eje(fig, f"{c['externa_nombre']} de cada pliegue")
    ui.formato_eje(fig, "x", c["d_eje"])
    lo, hi = float(P[orden].min().min()), float(P[orden].max().max())
    _lienzo_modelos(fig, nombres, lo, hi)
    _valores_al_borde(fig, nombres, [ui.num(media[m], c["d"]) for m in orden], foco=D.NOMBRE_CORTO[foco])

    por_media = media.sort_values(ascending=not c["mayor_es_mejor"], kind="stable")
    puesto, primero = list(por_media.index).index(foco) + 1, por_media.index[0]
    solapan = [m for m in orden if P[m].min() <= P[foco].max() and P[m].max() >= P[foco].min()]
    titulo = f"En los pliegues {D.en_frase(foco)} {_queda(foco, puesto)}"
    if puesto > 1:
        titulo += f", a {ui.num(abs(media[primero] - media[foco]), c['d'])} {_tras('de', D.en_frase(primero))}"
    titulo += (f"; los recorridos de los {_cardinal(len(solapan))} primeros modelos se solapan" if solapan == orden[:len(solapan)]
               else f"; su recorrido se solapa con el de {len(solapan) - 1} modelos")
    sub = (f"{c['metrica']} de cada uno de los {_cardinal(n)} pliegues externos de la validación anidada (un punto por pliegue), con la "
           f"combinación de cada modelo. La franja azul es el recorrido {_tras('de', D.en_frase(foco))}; a la derecha, la media de cada "
           "modelo.")
    nota = ("El eje no parte de cero. Los modelos van en el orden de la prueba, no en el de los pliegues: la métrica externa se midió "
            "sin mirar la prueba. Son los pliegues que usan las pruebas P1 y P2 del modelo original.")
    filas = [[D.NOMBRE_MODELO[m]] + [ui.num(P[m].iloc[k], c["d"]) for k in range(n)] + [ui.num(media[m], c["d"]), ui.num(desv[m], c["d"])]
             for m in orden]
    tabla = ui.tabla(["Modelo"] + [f"Pliegue {r}" for r in rotulos] + ["Media", "Desviación"], filas, compacta=True,
                     muestras=[rol_de[m] for m in orden])
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: media {f[-2]}" for f in filas) + "."
    return {"figura": fig, "titulo": titulo, "sub": sub, "tabla": tabla, "nota": nota, "etiqueta": etiqueta,
            "leyenda": _leyenda_diez(tarea, orden, extra=[(f"Recorrido {_tras('de', D.en_frase(foco))}", "franja", "barra")])}


# --- 5. Frente a la entrega anterior (etiqueta de la mediana) -------------------------------------------------------------
def _entrega_anterior():
    """Las dos tarjetas antes → después con la partición y la etiqueta de la Entrega 2, el techo de información y el aviso
    de que estas cifras no se comparan con las del resto de la pantalla."""
    e2 = D.tabla("resultados/entrega2.csv")
    techo = D.techo()
    tarjetas = []
    for tarea, c in TAREA.items():
        antes, despues = D.fila(e2, tarea=tarea, modelo=c["e2"]), D.fila(e2, tarea=tarea, modelo="ebm")
        rival = D.fila(e2, tarea=tarea, modelo=RIVAL)
        cambio = despues["metrica"] - antes["metrica"]
        mejora = despues["delta"] > 0
        intervalo = F.intervalo(despues["delta_inf"], despues["delta_sup"], c["d"])
        if tarea == "clasificacion":
            cubre = (despues["metrica"] - AZAR_AUC) / (techo["auc_mediana"] - AZAR_AUC)
            que = "Clasificación, con la etiqueta y la partición de la entrega anterior."
            medida = (f"El techo, {ui.num(techo['auc_mediana'], c['d'])} de AUC con esta etiqueta, es lo mejor que puede lograr un modelo "
                      f"sin la variable excluida (active_products). La parte cubierta se mide sobre el azar: AUC menos "
                      f"{ui.num(AZAR_AUC, 1)}, dividido entre techo menos {ui.num(AZAR_AUC, 1)}.")
        else:
            cubre = despues["secundaria"] / techo["r2"]
            que = "Regresión del puntaje, con la partición de la entrega anterior."
            medida = (f"El techo es {ui.num(techo['r2'], 4)} de R² y la EBM llega a {ui.num(despues['secundaria'], 4)}: la parte "
                      "cubierta es su cociente.")
        verbo = "sube" if cambio > 0 else "baja"
        titulo = f"El {c['metrica']} {verbo} de {ui.num(antes['metrica'], c['d'])} a {ui.num(despues['metrica'], c['d'])}"
        ficha = ui.antes_despues(
            f"{c['metrica']} de prueba", ui.num(antes["metrica"], c["d"]), ui.num(despues["metrica"], c["d"]),
            cambio=ui.chip_cambio(cambio, ui.con_signo(cambio, c["d"]), "bien" if mejora else "critico"),
            nota=(f"{'Mejora' if mejora else 'Cambio'}: {ui.num(abs(despues['delta']), c['d'])}, IC 95 % {intervalo}; "
                  f"{F.p_valor(despues['p_holm'])} ajustado por Holm ({c['fijada']}). {_mayuscula(D.en_frase(RIVAL))}, en la misma "
                  f"partición: {ui.num(rival['metrica'], c['d'])}."),
            hoy_rotulo=D.NOMBRE_MODELO[c["e2"]], plan_rotulo=D.NOMBRE_MODELO["ebm"])
        barra = ui.barra_rayas("Parte del techo de información que cubre la EBM", F.pct(cubre, 0), cubre)
        modelos = [c["e2"], RIVAL, "ebm"]
        filas = []
        for m in modelos:
            f = D.fila(e2, tarea=tarea, modelo=m)
            fila = [D.NOMBRE_MODELO[m], f"{ui.num(f['metrica'], c['d'])} {F.intervalo(f['ic_inf'], f['ic_sup'], c['d'])}"]
            if tarea == "regresion":
                fila.append(ui.num(f["secundaria"], 4))
            if m == c["e2"]:
                fila += [F.RAYA, F.RAYA, F.RAYA]
            else:
                fila += [f"{ui.con_signo(f['delta'], c['d'])} {F.intervalo(f['delta_inf'], f['delta_sup'], c['d'])}",
                         ui.num(f["estadistico"], 2), _p(f["p_holm"])]
            filas.append(fila)
        encabezados = (["Modelo", f"{c['metrica']} [IC 95 %]"] + (["R²"] if tarea == "regresion" else [])
                       + ["Mejora frente a la E2 [IC 95 %]", f"{c['fijada']}: {c['simbolo']}", "p ajustado"])
        tabla = ui.tabla(encabezados, filas, compacta=True, muestras=[ROL[m] for m in modelos])
        tarjetas.append(ui.tarjeta([ficha, barra, ui.nota(medida)], titulo=titulo, subtitulo=que, detalles=tabla, fuentes=FUENTES_E2))

    meta = M.meta()["etiqueta"]
    ebm = D.fila(D.prueba("clasificacion"), modelo="ebm")
    mediana = D.fila(e2, tarea="clasificacion", modelo="ebm")
    c, r = TAREA["clasificacion"], TAREA["regresion"]
    # La EBM y XGBoost de esta comparación no son los modelos finales: se afinaron aparte (metodología del informe)
    busquedas = D.json(BUSQUEDAS_E2)
    optimizador = D.NOMBRE_OPTIMIZADOR.get(str(busquedas["optimizador"]).split(" ")[0], str(busquedas["optimizador"]))
    aqui, arriba = D.fila(e2, tarea="regresion", modelo="ebm")["metrica"], D.fila(D.prueba("regresion"), modelo="ebm")[r["prueba"]]
    aviso = ui.aviso(
        [f"Aquí «riesgo alto» es la mitad superior del puntaje (por encima de la mediana, "
         f"{ui.num(meta['mediana_entrenamiento'], DECIMALES_UMBRAL)}), la etiqueta de la Entrega 2. En el resto de la pantalla es el "
         f"cuarto superior (percentil {ui.num(meta['q'] * 100)}, {ui.num(meta['umbral'], DECIMALES_UMBRAL)}).",
         f"El AUC de {ui.num(mediana['metrica'], c['d'])} de esta sección y el de {ui.num(ebm[c['prueba']], c['d'])} de la EBM en la prueba "
         "con la etiqueta del cuarto superior miden "
         "problemas distintos: no se restan ni se comparan. El techo de información solo se calculó para esta etiqueta.",
         f"{_mayuscula(D.en_frase('ebm'))} y {D.en_frase(RIVAL)} de esta comparación se afinaron aparte ({optimizador}, validación cruzada "
         f"de {int(busquedas['k_cv'])} pliegues); por eso difieren en la última cifra de los modelos finales: el {r['metrica']} de la EBM "
         f"es {ui.num(aqui, r['d'])} aquí y {ui.num(arriba, r['d'])} en su modelo final."],
        "atencion", "Otra etiqueta: estas cifras no se comparan con las de arriba")
    return [ui.rejilla(tarjetas, "1-1"), aviso]


# --- La pantalla ----------------------------------------------------------------------------------------------------------
def _aviso_entregable3():
    """Lo que separa las dos mitades de la pantalla: de aquí hacia abajo, el análisis del Entregable 3 y la EBM en azul."""
    return ui.aviso([
        "De aquí hacia abajo está el análisis del Entregable 3, sin cambios: la EBM, que fue el modelo nuevo de esa entrega y hoy es "
        "un benchmark reciente (viene de una librería), frente a las siete referencias de clase, con las pruebas de entonces (t "
        "corregida, DeLong o Diebold-Mariano, McNemar, la prueba bayesiana y el Model Confidence Set).",
        "En estos gráficos la serie en azul es la EBM. El modelo original y LightGBM no entran en estas pruebas: las suyas son las "
        "del protocolo de la corrección, en las secciones de arriba."], None, "Lo que sigue es el análisis del Entregable 3")


def layout(tarea=None, **_):
    """`/modelos/comparacion?tarea=regresion` abre con esa tarea (un valor desconocido cae en clasificación). Cuando la
    dirección pide una tarea, el control no repone el valor recordado en la sesión: manda la dirección."""
    pedida = tarea in TAREA
    inicial = tarea if pedida else "clasificacion"
    control_tarea = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", [{"label": D.NOMBRE_TAREA[t], "value": t} for t in TAREA],
                                                       inicial, persistir=not pedida))
    referencia = ui.control("Referencia:", _selector_referencia(inicial))
    # La ficha de la EBM queda varias pantallas más abajo: un enlace interno la deja a la vista (el callback de la referencia
    # pone en el botón el nombre de la que esté elegida)
    a_la_ficha = html.Div(ui.boton(html.Span(_rotulo_ficha(_referencia(inicial, RIVAL)), id=f"{PRE}-ir-ficha"), href=f"#{PRE}-fichas",
                                   icono="flecha-abajo"), className="botonera")
    diez, diez_sub = _diez(inicial)
    k = _cardinal(int(D.json(CD_JSON)[inicial]["modelos"]))
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), html.Span(_respuesta(inicial), id=f"{PRE}-respuesta"), [control_tarea], [a_la_ficha],
                      pantalla=PANTALLA),
        ui.seccion("El veredicto del modelo original, condición por condición", _veredicto(),
                   subtitulo="La regla de superioridad de la guía (§6.4), en las dos tareas, con lo que dicen los archivos de la "
                             "corrección. No cambia con el control de tarea.",
                   fuentes=[D.fuente_original("veredicto.json"), D.fuente_original("condicion4.json")]),
        ui.seccion(f"Los {k} modelos a la vez: Friedman y Nemenyi", [html.Div(diez, id=f"{PRE}-todos", className="rejilla")],
                   subtitulo=html.Span(diez_sub, id=f"{PRE}-s0-sub")),
        ui.seccion("El modelo original frente a cada benchmark", [html.Div(_frente(inicial), id=f"{PRE}-frente", className="rejilla")],
                   subtitulo="Las cuatro pruebas del protocolo pre-registrado (P1 a P4), con p crudos, p ajustados por Holm y tamaños "
                             "del efecto: primero frente al mejor benchmark, que es el que decide la regla, y después frente a todos.",
                   fuentes=[D.fuente_original("veredicto.json"), D.fuente_original("pruebas_hipotesis.csv")]),
        ui.seccion("El desempeño de cada modelo en la prueba", [
            ui.rejilla([
                ui.grafico(f"{PRE}-des", fuentes=FUENTES_DIEZ, alto="alto"),
                ui.tarjeta([html.Div(id=f"{PRE}-acota", className="resultados")], titulo=html.Span(id=f"{PRE}-acota-titulo"),
                           fuentes=FUENTES_DIEZ + [D.fuente_original("veredicto.json")]),
            ], "2-1"),
        ], subtitulo=html.Span(id=f"{PRE}-s1-sub")),
        ui.seccion("Los pliegues de la validación anidada", [
            ui.grafico(f"{PRE}-pli", fuentes=FUENTES_PLIEGUES),
        ], subtitulo="Lo que obtuvo cada modelo dentro del entrenamiento, antes de tocar la prueba, con los mismos pliegues para todos."),
        ui.seccion("El análisis del Entregable 3: la EBM frente a cada referencia", [
            _aviso_entregable3(),
            ui.grafico(f"{PRE}-dif", fuentes=FUENTES_PRUEBAS),
            html.Div(_bloque_mcs(inicial), id=f"{PRE}-mcs", className="rejilla"),
        ], subtitulo=html.Span(id=f"{PRE}-s2-sub")),
        ui.seccion("La EBM frente a una referencia, prueba por prueba", [
            ui.rejilla([
                ui.tarjeta([html.Div(id=f"{PRE}-ficha", className="resultados")], titulo=html.Span(id=f"{PRE}-ficha-titulo"),
                           fuentes=FUENTES_PRUEBAS),
                ui.grafico(f"{PRE}-par", fuentes=FUENTES_PAR, alto="alto"),
                ui.tarjeta([html.Div(id=f"{PRE}-bayes", className="resultados")], titulo=html.Span(id=f"{PRE}-bayes-titulo"),
                           fuentes=FUENTES_PRUEBAS),
                ui.tarjeta([html.Div(id=f"{PRE}-efectos", className="resultados")], titulo=html.Span(id=f"{PRE}-efectos-titulo"),
                           fuentes=FUENTES_PRUEBAS),
            ], "1-1"),
            html.Div(id=f"{PRE}-avisos"),
        ], subtitulo=f"Elija una referencia. El veredicto se calcula con los p ajustados por Holm y α = {ui.num(D.ALFA, 2)}; la "
                     "probabilidad de equivalencia práctica se informa aparte, porque no es un valor p.",
            controles=[referencia], id=f"{PRE}-fichas"),
        ui.seccion("La EBM frente a la entrega anterior, con la etiqueta de la mediana", _entrega_anterior(),
                   subtitulo="Sección aparte: usa la partición y la etiqueta de la Entrega 2 y no cambia con el control de tarea.",
                   fuentes=FUENTES_E2 + [D.fuente(BUSQUEDAS_E2), D.fuente("metadatos.json")] + FUENTES_PRUEBA[:2]),
    ])


@callback(Output(f"{PRE}-respuesta", "children"), Output(f"{PRE}-s1-sub", "children"), *_salidas(f"{PRE}-des"),
          Output(f"{PRE}-acota-titulo", "children"), Output(f"{PRE}-acota", "children"),
          Output(f"{PRE}-s2-sub", "children"), *_salidas(f"{PRE}-dif"),
          Output(f"{PRE}-ref", "options"), Output(f"{PRE}-ref", "value"), Output(f"{PRE}-avisos", "children"),
          *_salidas(f"{PRE}-pli"),
          Input(f"{PRE}-tarea", "value"), State(f"{PRE}-ref", "value"))
def _por_tarea(tarea, ref):
    """Todo lo que depende solo de la tarea; además deja en el selector las siete referencias de esa tarea."""
    tarea = tarea if tarea in TAREA else "clasificacion"
    c = TAREA[tarea]
    etiqueta = M.meta()["etiqueta"]
    alcance = (f" Riesgo alto es el cuarto superior del puntaje (percentil {ui.num(etiqueta['q'] * 100)})." if tarea == "clasificacion"
               else " El objetivo es el puntaje de propensión del conjunto de datos.")
    s1 = (f"{c['que']}: el ajuste final de cada modelo, con la combinación que eligió o fijó la validación anidada, medido una sola "
          f"vez en los {F.entero(D.resumen()['n_prueba'])} clientes de prueba.{alcance}")
    titulo_acota, acota = _acota(tarea)
    valida = _referencia(tarea, ref)
    return (_respuesta(tarea), s1, *_partes(_desempeno(tarea)), titulo_acota, acota, _resumen_veredictos(tarea),
            *_partes(_diferencias(tarea)), _opciones(tarea), (no_update if valida == ref else valida), _avisos(tarea),
            *_partes(_pliegues(tarea)))


@callback(Output(f"{PRE}-todos", "children"), Output(f"{PRE}-s0-sub", "children"), Output(f"{PRE}-frente", "children"),
          Output(f"{PRE}-mcs", "children"), Input(f"{PRE}-tarea", "value"))
def _por_tarea_todos(tarea):
    """Lo que no es un ui.grafico con partes: todos los modelos a la vez (el diagrama de diferencia crítica y Friedman), el
    modelo original frente a cada benchmark y, en regresión, el Model Confidence Set del Entregable 3."""
    tarea = tarea if tarea in TAREA else "clasificacion"
    return (*_diez(tarea), _frente(tarea), _bloque_mcs(tarea))


@callback(Output(f"{PRE}-ficha-titulo", "children"), Output(f"{PRE}-ficha", "children"), *_salidas(f"{PRE}-par"),
          Output(f"{PRE}-bayes-titulo", "children"), Output(f"{PRE}-bayes", "children"),
          Output(f"{PRE}-efectos-titulo", "children"), Output(f"{PRE}-efectos", "children"), Output(f"{PRE}-ir-ficha", "children"),
          Input(f"{PRE}-ref", "value"), Input(f"{PRE}-tarea", "value"))
def _por_referencia(ref, tarea):
    """La ficha de la referencia elegida, su gráfico (la curva ROC en clasificación, los pliegues en regresión), el reparto
    bayesiano y los tamaños de efecto."""
    tarea = tarea if tarea in TAREA else "clasificacion"
    ref = _referencia(tarea, ref)
    f = _ficha(tarea, ref)
    return (f["titulo"], f["cuerpo"], *_partes(_roc(ref) if tarea == "clasificacion" else _pliegues_par(ref)),
            f["titulo_bayes"], f["bayes"], f["titulo_efectos"], f["efectos"], _rotulo_ficha(ref))
