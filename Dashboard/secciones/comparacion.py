"""Comparación · ¿Es mejor que los otros modelos?

Primero, los ocho modelos a la vez con la EBM incluida: en clasificación, el diagrama de diferencia crítica (rangos medios
de los pliegues externos, Friedman, Iman-Davenport y la diferencia crítica de Nemenyi, de datos/cd.json); en regresión, la
tabla del Model Confidence Set del informe. Después, la EBM frente a las siete referencias de cada tarea: el desempeño en
la prueba con su intervalo, la diferencia con cada referencia sobre la banda de equivalencia práctica, la ficha de una
referencia con todas sus pruebas y los pliegues de la validación anidada. Aparte, con la etiqueta de la mediana, la
comparación con la entrega anterior y el techo de información.

El veredicto de cada referencia es el de datos_app.veredicto (p ajustados por Holm y ALFA), el mismo de Inicio: ninguno se
escribe a mano. Toda cifra sale de datos/ y cada tarjeta dice de qué archivo.
"""
from functools import lru_cache

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, no_update

import datos_app as D
import formato as F
import motor as M
import ui


PRE = "co"
PANTALLA = "comparacion"
PREGUNTA = "¿Es mejor que los otros modelos?"
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
CARDINAL = {1: "un", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho"}
ORDINAL_F = {1: "primera", 2: "segunda", 3: "tercera", 4: "cuarta", 5: "quinta", 6: "sexta", 7: "séptima", 8: "octava"}
ORDINAL_M = {1: "primer", 2: "segundo", 3: "tercer", 4: "cuarto", 5: "quinto", 6: "sexto", 7: "séptimo", 8: "octavo"}
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
FUENTES_PLIEGUES = [D.fuente(TAREA[t]["pliegues"]) for t in TAREA] + FUENTES_PRUEBA[:-1]
FUENTES_PAR = [D.fuente("puntajes_prueba.parquet"), D.fuente(TAREA["regresion"]["pliegues"])] + FUENTES_PRUEBAS
FUENTES_E2 = [D.fuente("resultados/entrega2.csv"), D.fuente("entrega3/comparacion_entrega2/techo.json")]
BUSQUEDAS_E2 = "entrega3/comparacion_entrega2/busquedas.json"      # cómo se afinaron la EBM y XGBoost de esa comparación
CD_JSON = "cd.json"               # Friedman, Iman-Davenport y Nemenyi sobre los pliegues externos, por tarea
FUENTES_CD = [D.fuente(CD_JSON), D.fuente(TAREA["clasificacion"]["pliegues"]), D.fuente("resultados/pruebas_clasificacion.csv")]
FUENTES_MCS = [D.fuente("resultados/regresion_prueba.csv"), D.fuente("resultados/resumen.json"), D.fuente(CD_JSON),
               D.fuente("resultados/pruebas_regresion.csv")]
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


def _tras(preposicion, frase):
    """«a» o «de» delante de un nombre en frase, con su contracción: «al árbol de decisión», «del SVM lineal», «de la EBM»."""
    if frase.startswith("el "):
        return {"a": "al", "de": "del"}[preposicion] + frase[2:]
    return f"{preposicion} {frase}"


def _lista(modelos):
    """Varios modelos dentro de una frase (D.en_frase): «XGBoost, Random Forest y el árbol de decisión»."""
    return F.enumerar(D.en_frase(m) for m in modelos)


def _rotulo_ficha(ref):
    """El texto del botón que lleva a la ficha de la referencia elegida."""
    return f"Ver la ficha {_tras('de', D.en_frase(ref))}"


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


def _externa(tarea):
    """Puesto de la EBM en la validación anidada (1 = la mejor métrica externa) y la fila del modelo que queda primero."""
    c = TAREA[tarea]
    t = D.prueba(tarea).sort_values(c["externa"], ascending=not c["mayor_es_mejor"]).reset_index(drop=True)
    return int(t.index[t["modelo"] == "ebm"][0]) + 1, t.iloc[0]


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


def _respuesta(tarea):
    """El subtítulo de la pantalla: la respuesta a la pregunta, con las cifras de la tarea elegida."""
    c = TAREA[tarea]
    t, pr, v = D.prueba(tarea), D.pruebas(tarea), D.veredictos(tarea)
    empates = [m for m in v.index if v[m] == "empata"]
    superadas = [m for m in v.index if v[m] == "supera"]
    resto = [m for m in v.index if v[m] not in ("empata", "supera")]
    frases = []
    if tarea == "clasificacion":
        if empates:
            arboles = {m for m in v.index if D.ROL_MODELO[m] == "rival"}
            quienes = "los modelos de árboles" if set(empates) == arboles else "todos los demás"
            mcnemar = all(D.fila(pr, modelo=m)["p_mcnemar_holm"] >= D.ALFA for m in empates)
            pruebas = "ninguna prueba encuentra diferencia" if mcnemar else f"ni la t corregida ni {c['fijada']} encuentran diferencia"
            frases.append(f"No es mejor que {quienes}: frente {_tras('a', _lista(empates))} {pruebas} tras la corrección de Holm (probabilidad "
                          f"de equivalencia práctica: {F.enumerar(_prob(D.fila(pr, modelo=m)['p_rope']) for m in empates)}).")
    else:
        e, puesto, primero = D.fila(t, modelo="ebm"), list(t["modelo"]).index("ebm") + 1, t.iloc[0]
        if puesto == 1:
            lugar = f"En regresión tiene el menor {c['metrica']} de prueba ({ui.num(e[c['prueba']], c['d'])})"
        else:
            lugar = (f"No: en regresión tiene el {ORDINAL_M[puesto]} menor {c['metrica']} de prueba ({ui.num(e[c['prueba']], c['d'])}), a "
                     f"{ui.num(abs(e[c['prueba']] - primero[c['prueba']]), c['d'])} {_tras('de', D.en_frase(primero['modelo']))}")
        frases.append(lugar + (f"; con los p ajustados, ni la t corregida ni {c['fijada']} la separan {_tras('de', _lista(empates))}."
                               if empates else "."))
    if superadas:
        frases.append(f"Sí supera con significación, tras la corrección de Holm, {_tras('a', _lista(superadas))}.")
    if resto:
        frases.append(f"Frente {_tras('a', _lista(resto))} el veredicto es otro: véalo en su ficha.")
    if tarea == "regresion":
        frases.append(_texto_mcs())
    return " ".join(frases)


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
    items = ([(D.NOMBRE_MODELO["ebm"], "modelo", FORMA_LEYENDA["modelo"])] if con_ebm else []) + [
        ("Modelos de árboles", "rival", FORMA_LEYENDA["rival"]), (TAREA[tarea]["base"], "base", FORMA_LEYENDA["base"])]
    return ui.leyenda(items + ([extra] if extra else []))


def _salidas(gid):
    return [Output(f"{gid}{sufijo}", propiedad) for sufijo, propiedad in PARTES_GRAFICO]


def _partes(g):
    return [g["figura"], g["titulo"], g["sub"], g["leyenda"], g["tabla"], g["nota"], g["etiqueta"]]


# --- 0. Los ocho modelos a la vez: Friedman → Nemenyi (clasificación) y el Model Confidence Set (regresión) ---------------
def _nemenyi(tarea):
    """Friedman → Nemenyi sobre los pliegues externos (datos/cd.json): rangos medios de menor a mayor (en cada pliegue, 1 es
    la mejor métrica), la diferencia crítica, los modelos a menos de ella del mejor rango medio, los que Nemenyi separa de la
    EBM y los rangos de cada pliegue, recalculados de las métricas de cada pliegue para comprobar el archivo."""
    n = D.json(CD_JSON)[tarea]
    rangos = pd.Series(n["rango_medio"], dtype=float).sort_values(kind="stable")
    mejor, dc = float(rangos.min()), float(n["diferencia_critica"])
    P = pd.DataFrame(n["metricas_pliegue"])
    R = P.rank(axis=1, ascending=not TAREA[tarea]["mayor_es_mejor"], method="average")
    coincide = bool((R.mean().reindex(rangos.index) - rangos).abs().max() < TOLERANCIA_RANGO)
    return {"datos": n, "rangos": rangos, "mejor": mejor, "dc": dc, "por_pliegue": R, "coincide": coincide,
            "dentro": [m for m in rangos.index if rangos[m] - mejor < dc],
            "separa_ebm": [m for m in rangos.index if m != "ebm" and abs(rangos[m] - rangos["ebm"]) >= dc]}


def _aviso_rangos(n):
    if n["coincide"]:
        return None
    return ui.aviso(f"Los rangos medios de {D.fuente(CD_JSON)} no coinciden con los que se recalculan de sus métricas por pliegue: "
                    "revise el archivo antes de leer el diagrama.", "critico", "El diagrama no cuadra con sus datos")


def _diagrama_cd():
    """Clasificación: el diagrama de diferencia crítica. Una fila por modelo con su rango medio en los pliegues externos; la
    franja va del mejor rango medio a ese rango más la diferencia crítica de Nemenyi, que se dibuja arriba como una barra."""
    n = _nemenyi("clasificacion")
    datos, rangos, mejor, dc = n["datos"], n["rangos"], n["mejor"], n["dc"]
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
        grupo = [m for m in orden if D.ROL_MODELO[m] == rol]
        if not grupo:
            continue
        textos = [f"{D.NOMBRE_MODELO[m]}: rango medio <b>{ui.num(rangos[m], 1)}</b> · a {ui.num(rangos[m] - mejor, 1)} del mejor · "
                  + ("a menos de la diferencia crítica" if m in n["dentro"] else "más allá de la diferencia crítica") for m in grupo]
        fig.add_trace(go.Scatter(x=[rangos[m] for m in grupo], y=[D.NOMBRE_CORTO[m] for m in grupo], **_marca(rol),
                                 **ui.hover_texto(textos)))
    ui.franja(fig, mejor, mejor + dc)
    ui.unidad_eje(fig, "rango medio (1 = el mayor AUC)")    # corta: tiene que caber a 400 px; el subtítulo lo dice entero
    _lienzo_modelos(fig, filas_y, 1, k)
    valores, textos_eje = ui.marcas(1, k, 1)
    fig.update_xaxes(tickvals=valores, ticktext=textos_eje)
    _valores_al_borde(fig, filas_y, [ui.num(dc, 2)] + [ui.num(rangos[m], 1) for m in orden], foco=D.NOMBRE_CORTO["ebm"])

    separa = n["separa_ebm"]
    titulo = (f"La diferencia crítica de Nemenyi es de {ui.num(dc, 2)} rangos: {CARDINAL[len(n['dentro'])]} de los {CARDINAL[k]} "
              "modelos quedan a menos de ella del mejor y "
              + ("no separa a la EBM de ningún modelo" if not separa else f"solo separa a la EBM {_tras('de', _lista(separa))}"))
    sub = (f"Rango medio de cada modelo en los {CARDINAL[pliegues]} pliegues externos de la validación anidada: en cada pliegue, "
           f"1 es el mayor AUC y {k} el menor. La franja azul va del mejor rango medio ({ui.num(mejor, 1)}) a ese rango más la "
           f"diferencia crítica (α = {ui.num(datos['alfa'], 2)}); arriba, la barra de esa diferencia.")
    nota = ("Complemento de la app: el informe compara la EBM con cada referencia por pares; este diagrama se calculó con los mismos "
            "AUC de los pliegues. A la derecha, el rango medio de cada modelo.")
    R = n["por_pliegue"]
    filas = [[D.NOMBRE_MODELO[m], ui.num(rangos[m], 1), ui.num(rangos[m] - mejor, 1), "sí" if m in n["dentro"] else "no"]
             + [ui.num(R[m].iloc[i], 1) for i in range(len(R))] for m in orden]
    tabla = [ui.tabla(["Modelo", "Rango medio", "Distancia al mejor", "A menos de la diferencia crítica"]
                      + [f"Pliegue {i + 1}" for i in range(len(R))], filas, compacta=True, muestras=[D.ROL_MODELO[m] for m in orden],
                      num=[False, True, True, False] + [True] * len(R)),
             ui.nota(f"Rangos de cada pliegue recalculados de los AUC de {D.fuente(CD_JSON)} (los empates reparten el rango).")]
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: rango medio {f[1]}" for f in filas) + f". Diferencia crítica: {ui.num(dc, 2)}."
    leyenda = ui.leyenda([(D.NOMBRE_MODELO["ebm"], "modelo", FORMA_LEYENDA["modelo"]),
                          ("Modelos de árboles", "rival", FORMA_LEYENDA["rival"]),
                          (TAREA["clasificacion"]["base"], "base", FORMA_LEYENDA["base"]),
                          ("Diferencia crítica de Nemenyi", "referencia", "gruesa"),
                          ("A menos de la diferencia crítica del mejor rango medio", "franja", "barra")])
    return ui.grafico(fig, titulo=titulo, subtitulo=sub, etiqueta=etiqueta, tabla=tabla, leyenda=leyenda, nota=nota, alto="alto",
                      fuentes=FUENTES_CD)


def _lectura_nemenyi(tarea, n):
    """Lo que dice y lo que no dice Nemenyi frente a las pruebas por pares del informe (con los p ajustados por Holm)."""
    v = D.veredictos(tarea)
    rangos = n["rangos"]
    superadas = [m for m in v.index if v[m] == "supera"]
    empates = [m for m in v.index if v[m] == "empata"]
    confirma = [m for m in superadas if m in n["separa_ebm"]]
    no_confirma = [m for m in superadas if m not in n["separa_ebm"]]
    contradice = [m for m in empates if m in n["separa_ebm"]]
    fijada = TAREA[tarea]["fijada"]
    lejos = max((m for m in rangos.index if m != "ebm"), key=lambda m: abs(rangos[m] - rangos["ebm"]))
    frases = []
    if not n["separa_ebm"]:
        frases.append(f"Nemenyi no separa a la EBM de ningún modelo: su mayor distancia, con {D.en_frase(lejos)}, es de "
                      f"{ui.num(abs(rangos[lejos] - rangos['ebm']), 1)} rangos.")
    if empates and not contradice:
        frases.append(f"{'Tampoco contradice' if not n['separa_ebm'] else 'No contradice'} el informe: como las pruebas por pares, no "
                      f"separa a la EBM {_tras('de', _lista(empates))}.")
    if confirma:
        frases.append(f"Confirma la ventaja que las pruebas por pares encuentran frente {_tras('a', _lista(confirma))}.")
    if no_confirma:
        frases.append(f"Pero no tiene potencia para confirmar las ventajas que la t corregida y {fijada} sí encuentran, con la corrección de "
                      f"Holm, frente {_tras('a', _lista(no_confirma))}: con {CARDINAL[int(n['datos']['pliegues'])]} pliegues la "
                      f"diferencia crítica es de {ui.num(n['dc'], 2)} rangos en una escala de 1 a {int(n['datos']['modelos'])}.")
    if contradice:
        frases.append(f"Separa a la EBM {_tras('de', _lista(contradice))}, con {'el' if len(contradice) == 1 else 'los'} que las pruebas por "
                      "pares no encuentran diferencia: véalo en la ficha de cada referencia.")
    else:
        frases.append("La conclusión del informe no cambia.")
    return " ".join(frases)


def _tarjeta_friedman():
    """Clasificación: las tres cifras de la prueba (Friedman, Iman-Davenport, diferencia crítica) y su lectura honesta."""
    n = _nemenyi("clasificacion")
    datos = n["datos"]
    k, pliegues = int(datos["modelos"]), int(datos["pliegues"])
    gl = [int(g) for g in datos["gl"]]
    fuera = [m for m in n["rangos"].index if m not in n["dentro"]]
    cifras = ui.cifras([
        ui.cifra(f"Friedman, χ² con {gl[0]} grados de libertad", ui.num(datos["chi2_friedman"], 2),
                 variacion=F.p_valor(datos["p_friedman"]), mini=True),
        ui.cifra(f"Iman-Davenport, F({gl[0]}; {gl[1]})", ui.num(datos["f_iman_davenport"], 2),
                 variacion=F.p_valor(datos["p_iman_davenport"]), mini=True),
        ui.cifra("Diferencia crítica de Nemenyi", ui.num(n["dc"], 2), "rangos", mini=True,
                 variacion=f"q = {ui.num(datos['q_alfa'], 3)} con α = {ui.num(datos['alfa'], 2)}",
                 nota=f"Con {CARDINAL[k]} modelos y {CARDINAL[pliegues]} pliegues; crece con los modelos y baja con la raíz de los "
                      "pliegues."),
    ])
    titulo = (f"Friedman rechaza que los {CARDINAL[k]} modelos rindan igual ({F.p_valor(datos['p_friedman'])}); con {CARDINAL[pliegues]} "
              "pliegues, Nemenyi tiene poca potencia")
    primera = (f"Solo {_lista(fuera)} quedan a más de {ui.num(n['dc'], 2)} rangos del mejor rango medio." if fuera
               else f"Ningún modelo queda a más de {ui.num(n['dc'], 2)} rangos del mejor rango medio.")
    cuerpo = [cifras, ui.parrafo(primera + " " + _lectura_nemenyi("clasificacion", n))]
    aviso = _aviso_rangos(n)
    if aviso is not None:
        cuerpo.insert(0, aviso)
    return ui.tarjeta(cuerpo, titulo=titulo, fuentes=FUENTES_CD)


def _tabla_mcs():
    """Regresión: los ocho modelos con su p del Model Confidence Set y si quedan en el conjunto al nivel del informe,
    ordenados de mayor a menor p; comprueba que la columna en_mcs y resumen.json dicen lo mismo que los p."""
    c = TAREA["regresion"]
    t = D.prueba("regresion").sort_values(["p_mcs", c["prueba"]], ascending=[False, True], kind="stable")
    resumen = D.resumen()
    umbral = 1 - NIVEL_MCS
    dentro = [m for m, esta in zip(t["modelo"], t["en_mcs"]) if bool(esta)]
    no_cuadra = [m for m, p, esta in zip(t["modelo"], t["p_mcs"], t["en_mcs"]) if bool(esta) != (p > umbral) and bool(esta) != (p >= umbral)]
    if set(dentro) != set(resumen.get("mcs_incluidos", dentro)):
        no_cuadra.append("resumen.json")
    # (el intervalo del RMSE de cada modelo está en la sección siguiente: aquí cabe la tabla entera en media columna)
    filas = [[D.NOMBRE_MODELO[f["modelo"]], ui.num(f[c["prueba"]], c["d"]), _p(f["p_mcs"]),
              ui.chip("dentro", "bien") if bool(f["en_mcs"]) else ui.chip("fuera")] for _, f in t.iterrows()]
    nivel = F.pct(NIVEL_MCS, 0)
    tabla = ui.tabla(["Modelo", "RMSE de prueba", "p del MCS", f"En el conjunto al {nivel}"], filas,
                     muestras=[D.ROL_MODELO[m] for m in t["modelo"]], num=[False, True, True, False])
    sub = (f"Los {CARDINAL[len(t)]} regresores sobre las {F.entero(resumen['n_prueba'])} pérdidas cuadráticas de la prueba, con bootstrap "
           f"estacionario de bloques de {CARDINAL.get(int(resumen['mcs_bloque']), F.entero(resumen['mcs_bloque']))}. Quedan en el "
           f"conjunto los modelos con p mayor que {ui.num(umbral, 2)}; van de mayor a menor p.")
    cuerpo = [tabla, ui.nota(f"El conjunto se fijó antes de evaluar la prueba. Un modelo con {F.p_valor(0)} sale del conjunto con mucha "
                             "evidencia en contra.")]
    if no_cuadra:
        cuerpo.insert(0, ui.aviso(f"La columna en_mcs, los p del MCS y resumen.json no dicen lo mismo ({', '.join(no_cuadra)}).",
                                  "critico", "La tabla no cuadra con sus datos"))
    return ui.tarjeta(cuerpo, titulo=_texto_mcs().rstrip("."), subtitulo=sub, fuentes=FUENTES_MCS)


def _tarjeta_mcs_lectura():
    """Regresión: el p de la EBM, cuántos quedan en el conjunto y, aparte, Friedman → Nemenyi sobre los pliegues."""
    dentro, p_ebm, ultima = _mcs()
    n = _nemenyi("regresion")
    datos = n["datos"]
    rangos = n["rangos"]
    nivel = F.pct(NIVEL_MCS, 0)
    primero = rangos.index[0]
    cifras = ui.cifras([
        ui.cifra(f"p de la EBM en el MCS al {nivel}", _p(p_ebm), mini=True,
                 variacion=("La EBM queda dentro." if "ebm" in dentro else "La EBM es la última en salir." if ultima else "La EBM queda fuera.")),
        ui.cifra("Diferencia crítica de Nemenyi en los pliegues", ui.num(n["dc"], 2), "rangos", mini=True,
                 variacion=f"Friedman: χ² = {ui.num(datos['chi2_friedman'], 2)}; {F.p_valor(datos['p_friedman'])}"),
    ])
    titulo = (f"En los pliegues, Nemenyi deja a {CARDINAL[len(n['dentro'])]} modelos a menos de la diferencia crítica del mejor"
              + (", la EBM entre ellos" if "ebm" in n["dentro"] else ""))
    mcs = "El MCS se calcula sobre la prueba y es más exigente que las pruebas por pares."
    nemenyi = (f"En los pliegues, Friedman rechaza que los {CARDINAL[int(datos['modelos'])]} modelos rindan igual, pero con "
               f"{CARDINAL[int(datos['pliegues'])]} pliegues la diferencia crítica es de {ui.num(n['dc'], 2)} rangos: quedan a menos de ella "
               f"del mejor rango medio (el {_tras('de', D.en_frase(primero))}, {ui.num(rangos[primero], 1)}) {_lista(n['dentro'])}.")
    cuerpo = [cifras, ui.parrafo(f"{mcs} {nemenyi} {_lectura_nemenyi('regresion', n)}")]
    aviso = _aviso_rangos(n)
    if aviso is not None:
        cuerpo.insert(0, aviso)
    return ui.tarjeta(cuerpo, titulo=titulo, fuentes=FUENTES_MCS)


def _todos(tarea):
    """El bloque de los ocho modelos a la vez de la tarea, con la EBM incluida, y el subtítulo de su sección."""
    if tarea == "clasificacion":
        # el diagrama a todo el ancho (una fila por modelo) y debajo sus cifras, en una fila
        return ([_diagrama_cd(), _tarjeta_friedman()],
                "Sobre los mismos pliegues externos: la prueba de Friedman y el post hoc de Nemenyi, con la EBM incluida. Es un "
                "complemento: las conclusiones del informe salen de las pruebas por pares de las secciones siguientes.")
    return ([ui.rejilla([_tabla_mcs(), _tarjeta_mcs_lectura()], "1-1")],
            "Sobre la prueba: el Model Confidence Set del informe, con la EBM incluida; aparte, Friedman y Nemenyi sobre los "
            "pliegues externos.")


# --- 1. El desempeño de cada modelo en la prueba --------------------------------------------------------------------------
def _desempeno(tarea):
    """La métrica de prueba de cada modelo con su intervalo BCa; la franja es el intervalo de la EBM."""
    c = TAREA[tarea]
    t = D.prueba(tarea)
    e = D.fila(t, modelo="ebm")
    orden = list(t["modelo"])
    nombres = [D.NOMBRE_CORTO[m] for m in orden]
    fig = go.Figure()
    for rol in ("base", "rival", "modelo"):
        s = t[t["modelo"].map(D.ROL_MODELO) == rol]
        if s.empty:
            continue
        textos = [f"{D.NOMBRE_MODELO[f['modelo']]}: <b>{ui.num(f[c['prueba']], c['d'])}</b> {F.intervalo(f[c['inf']], f[c['sup']], c['d'])}"
                  f" · {c['externa_nombre']}: {ui.num(f[c['externa']], c['d'])}" for _, f in s.iterrows()]
        fig.add_trace(go.Scatter(x=s[c["prueba"]], y=[D.NOMBRE_CORTO[m] for m in s["modelo"]], **_marca(rol),
                                 error_x=_bigotes(s[c["prueba"]], s[c["inf"]], s[c["sup"]]), **ui.hover_texto(textos)))
    ui.franja(fig, e[c["inf"]], e[c["sup"]])
    ui.unidad_eje(fig, f"{c['metrica']} de prueba")
    ui.formato_eje(fig, "x", c["d_eje"])
    lo, hi = t[c["inf"]].min(), t[c["sup"]].max()
    _lienzo_modelos(fig, nombres, lo, hi)
    _valores_al_borde(fig, nombres, [ui.num(x, c["d"]) for x in t[c["prueba"]]], foco=D.NOMBRE_CORTO["ebm"])

    solapan = [f["modelo"] for _, f in t.iterrows() if f[c["inf"]] <= e[c["sup"]] and f[c["sup"]] >= e[c["inf"]]]
    otro = D.fila(t, modelo=next(m for m in orden if m != "ebm"))
    cuantos = (f"Los intervalos de los {CARDINAL[len(solapan)]} primeros modelos se solapan" if solapan == orden[:len(solapan)]
               else f"Los intervalos de {len(solapan) - 1} referencias se solapan con el de la EBM")
    titulo = (f"{cuantos}: la EBM, {ui.num(e[c['prueba']], c['d'])}; {D.en_frase(otro['modelo'])}, "
              f"{ui.num(otro[c['prueba']], c['d'])}")
    sub = (f"{c['metrica']} en los {F.entero(D.resumen()['n_prueba'])} clientes de prueba, con su intervalo bootstrap BCa del 95 % "
           f"({F.entero(D.resumen()['n_boot'])} réplicas). La franja azul es el intervalo de la EBM; a la derecha, el valor de cada modelo.")
    nota = (f"El eje no parte de cero: va de {ui.num(lo, c['d_eje'])} a {ui.num(hi, c['d_eje'])} para que se distingan los intervalos. "
            "Que dos intervalos se solapen no es una prueba: la comparación formal está en la sección siguiente.")
    if tarea == "clasificacion":
        encabezados = ["Modelo", "Mejor combinación", "AUC externo", "AUC de prueba [IC 95 %]", "AUC-PR"]
        filas = [[D.NOMBRE_MODELO[f["modelo"]], f"{D.NOMBRE_BALANCEO[f['balanceo']]}, {D.NOMBRE_OPTIMIZADOR[f['optimizador']]}",
                  f"{ui.num(f[c['externa']], c['d'])} ± {ui.num(f[c['externa_desv']], c['d'])}",
                  f"{ui.num(f[c['prueba']], c['d'])} {F.intervalo(f[c['inf']], f[c['sup']], c['d'])}", ui.num(f["auc_pr"], 3)]
                 for _, f in t.iterrows()]
    else:
        encabezados = ["Modelo", "Optimizador", "RMSE externo", "RMSE de prueba [IC 95 %]", "MAE", "R² [IC 95 %]", "p del MCS"]
        filas = [[D.NOMBRE_MODELO[f["modelo"]], D.NOMBRE_OPTIMIZADOR[f["optimizador"]],
                  f"{ui.num(f[c['externa']], c['d'])} ± {ui.num(f[c['externa_desv']], c['d'])}",
                  f"{ui.num(f[c['prueba']], c['d'])} {F.intervalo(f[c['inf']], f[c['sup']], c['d'])}", ui.num(f["mae_prueba"], 4),
                  f"{ui.num(f['r2_prueba'], 3)} {F.intervalo(f['r2_bca_inf'], f['r2_bca_sup'], 3)}", _p(f["p_mcs"])]
                 for _, f in t.iterrows()]
    tabla = [ui.tabla(encabezados, filas, compacta=True, muestras=[D.ROL_MODELO[m] for m in orden],
                      num=[False, False] + [True] * (len(encabezados) - 2)),
             ui.nota("La métrica externa es la media ± la desviación de los pliegues externos de la validación anidada.")]
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: {f[3]}" for f in filas) + "."
    return {"figura": fig, "titulo": titulo, "sub": sub, "tabla": tabla, "nota": nota, "etiqueta": etiqueta,
            "leyenda": _leyenda_modelos(tarea, extra=("Intervalo de la EBM", "franja", "barra"))}


def _acota(tarea):
    """Título y fichas de la tarjeta que acota el resultado de la prueba: la validación anidada y, según la tarea, la
    diferencia con el modelo más cercano o el Model Confidence Set."""
    c = TAREA[tarea]
    t = D.prueba(tarea)
    e = D.fila(t, modelo="ebm")
    puesto, primero = _externa(tarea)
    lugar = f"{ORDINAL_F[puesto].capitalize()} de {len(t)} modelos"
    if puesto > 1:
        lugar += (f", a {ui.num(abs(primero[c['externa']] - e[c['externa']]), c['d'])} {_tras('de', D.en_frase(primero['modelo']))} "
                  f"({ui.num(primero[c['externa']], c['d'])})")
    externa = ui.cifra(f"{c['externa_nombre']} de la EBM", ui.num(e[c["externa"]], c["d"]), f"± {ui.num(e[c['externa_desv']], c['d'])}",
                       variacion=lugar + ".", mini=True,
                       nota=f"Media y desviación de los {CARDINAL[_n_pliegues(tarea)]} pliegues externos. Con esta cifra se eligió la "
                            "combinación de cada modelo, sin mirar la prueba.")
    titulo = f"En la validación anidada la EBM queda {ORDINAL_F[puesto]}"
    if tarea == "clasificacion":
        orden = list(t["modelo"])
        otro = next(m for m in orden if m != "ebm")
        f = D.fila(D.pruebas(tarea), modelo=otro)
        incluye = f["delta_bca_inf"] <= 0 <= f["delta_bca_sup"]
        lectura = "El intervalo incluye el cero" if incluye else "El intervalo no incluye el cero"
        if orden[0] == "ebm" and incluye:
            lectura += f": tener el mayor {c['metrica']} de prueba es un valor puntual, no una ventaja establecida"
        segunda = ui.cifra(f"Diferencia de {c['metrica']} de prueba con {D.en_frase(otro)}", ui.con_signo(f[c["delta"]], c["d"]),
                           variacion=f"Intervalo del 95 %: {F.intervalo(f['delta_bca_inf'], f['delta_bca_sup'], c['d'], signo=True)}.",
                           nota=lectura + ".", mini=True)
    else:
        dentro, p_ebm, ultima = _mcs()
        # el Model Confidence Set se calcula sobre la prueba, no sobre los pliegues: el título lo separa de la validación
        titulo += f"; en la prueba, {'dentro' if 'ebm' in dentro else 'fuera'} del Model Confidence Set al {F.pct(NIVEL_MCS, 0)}"
        estado = "La EBM queda dentro." if "ebm" in dentro else ("La EBM es la última en salir." if ultima else "La EBM queda fuera.")
        segunda = ui.cifra(f"p de la EBM en el Model Confidence Set al {F.pct(NIVEL_MCS, 0)}", _p(p_ebm),
                           variacion=f"Quedan dentro los modelos con p mayor que {ui.num(1 - NIVEL_MCS, 2)}: {_lista(dentro)}.",
                           nota=f"{estado} Se calcula sobre la prueba, se fijó antes de evaluarla y es más exigente que las pruebas por pares.",
                           mini=True)
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
        s = t[t["modelo"].map(D.ROL_MODELO) == rol]
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
    tabla = [ui.tabla(encabezados, filas, compacta=True, muestras=[D.ROL_MODELO[m] for m in orden],
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
    nombre, rol, frase = D.NOMBRE_MODELO[ref], D.ROL_MODELO[ref], D.en_frase(ref)
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
    nombre, rol, frase = D.NOMBRE_MODELO[ref], D.ROL_MODELO[ref], D.en_frase(ref)
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
def _pliegues(tarea):
    """Una fila por modelo con la métrica de cada pliegue externo; la franja es el recorrido de la EBM."""
    c = TAREA[tarea]
    P = D.tabla(c["pliegues"])
    t = D.prueba(tarea).set_index("modelo")
    orden = list(t.index)
    n = len(P)
    nombres = [D.NOMBRE_CORTO[m] for m in orden]
    fig = go.Figure()
    for rol in ("base", "rival", "modelo"):
        grupo = [m for m in orden if D.ROL_MODELO[m] == rol]
        if not grupo:
            continue
        textos = [f"{D.NOMBRE_MODELO[m]} · pliegue {k + 1}: <b>{ui.num(P[m].iloc[k], c['d'])}</b>" for m in grupo for k in range(n)]
        fig.add_trace(go.Scatter(x=[P[m].iloc[k] for m in grupo for k in range(n)],
                                 y=[D.NOMBRE_CORTO[m] for m in grupo for _k in range(n)], **_marca(rol), **ui.hover_texto(textos)))
    ui.franja(fig, P["ebm"].min(), P["ebm"].max())
    ui.unidad_eje(fig, f"{c['externa_nombre']} de cada pliegue")
    ui.formato_eje(fig, "x", c["d_eje"])
    lo, hi = float(P[orden].min().min()), float(P[orden].max().max())
    _lienzo_modelos(fig, nombres, lo, hi)
    _valores_al_borde(fig, nombres, [ui.num(t.loc[m, c["externa"]], c["d"]) for m in orden], foco=D.NOMBRE_CORTO["ebm"])

    puesto, primero = _externa(tarea)
    solapan = [m for m in orden if P[m].min() <= P["ebm"].max() and P[m].max() >= P["ebm"].min()]
    titulo = f"En los pliegues la EBM queda {ORDINAL_F[puesto]}"
    if puesto > 1:
        titulo += f", a {ui.num(abs(primero[c['externa']] - t.loc['ebm', c['externa']]), c['d'])} {_tras('de', D.en_frase(primero['modelo']))}"
    titulo += (f"; los recorridos de los {CARDINAL[len(solapan)]} primeros modelos se solapan" if solapan == orden[:len(solapan)]
               else f"; su recorrido se solapa con el de {len(solapan) - 1} referencias")
    sub = (f"{c['metrica']} de cada uno de los {CARDINAL[n]} pliegues externos de la validación anidada (un punto por pliegue), con la "
           "mejor combinación de cada modelo. La franja azul es el recorrido de la EBM; a la derecha, la media de cada modelo.")
    nota = ("El eje no parte de cero. Los modelos van en el orden de la prueba, no en el de los pliegues: la métrica externa eligió "
            "la combinación de cada modelo sin mirar la prueba.")
    filas = [[D.NOMBRE_MODELO[m]] + [ui.num(P[m].iloc[k], c["d"]) for k in range(n)]
             + [ui.num(t.loc[m, c["externa"]], c["d"]), ui.num(t.loc[m, c["externa_desv"]], c["d"])] for m in orden]
    tabla = ui.tabla(["Modelo"] + [f"Pliegue {k + 1}" for k in range(n)] + ["Media", "Desviación"], filas, compacta=True,
                     muestras=[D.ROL_MODELO[m] for m in orden])
    etiqueta = f"{titulo}. " + "; ".join(f"{f[0]}: media {f[-2]}" for f in filas) + "."
    return {"figura": fig, "titulo": titulo, "sub": sub, "tabla": tabla, "nota": nota, "etiqueta": etiqueta,
            "leyenda": _leyenda_modelos(tarea, extra=("Recorrido de la EBM", "franja", "barra"))}


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
        tabla = ui.tabla(encabezados, filas, compacta=True, muestras=[D.ROL_MODELO[m] for m in modelos])
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
         f"El AUC de {ui.num(mediana['metrica'], c['d'])} de esta sección y el de {ui.num(ebm[c['prueba']], c['d'])} de arriba miden "
         "problemas distintos: no se restan ni se comparan. El techo de información solo se calculó para esta etiqueta.",
         f"{_mayuscula(D.en_frase('ebm'))} y {D.en_frase(RIVAL)} de esta comparación se afinaron aparte ({optimizador}, validación cruzada "
         f"de {int(busquedas['k_cv'])} pliegues); por eso difieren en la última cifra de los modelos finales: el {r['metrica']} de la EBM "
         f"es {ui.num(aqui, r['d'])} aquí y {ui.num(arriba, r['d'])} arriba."],
        "atencion", "Otra etiqueta: estas cifras no se comparan con las de arriba")
    return [ui.rejilla(tarjetas, "1-1"), aviso]


# --- La pantalla ----------------------------------------------------------------------------------------------------------
def layout(tarea=None, **_):
    """`/comparacion?tarea=regresion` abre con esa tarea (un valor desconocido cae en clasificación). Cuando la dirección
    pide una tarea, el control no repone el valor recordado en la sesión: manda la dirección."""
    pedida = tarea in TAREA
    inicial = tarea if pedida else "clasificacion"
    control_tarea = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", [{"label": D.NOMBRE_TAREA[t], "value": t} for t in TAREA],
                                                       inicial, persistir=not pedida))
    referencia = ui.control("Referencia:", _selector_referencia(inicial))
    pliegues = CARDINAL[_n_pliegues(inicial)]
    # La ficha queda casi tres pantallas más abajo: un enlace interno la deja a la vista (el callback de la referencia pone
    # en el botón el nombre de la que esté elegida)
    a_la_ficha = html.Div(ui.boton(html.Span(_rotulo_ficha(_referencia(inicial, RIVAL)), id=f"{PRE}-ir-ficha"), href=f"#{PRE}-fichas",
                                   icono="flecha-abajo"), className="botonera")
    todos, todos_sub = _todos(inicial)
    return ui.pantalla([
        ui.encabezado(PREGUNTA, html.Span(_respuesta(inicial), id=f"{PRE}-respuesta"), [control_tarea], [a_la_ficha],
                      pantalla=PANTALLA),
        ui.seccion(f"Los {CARDINAL[len(D.prueba(inicial))]} modelos a la vez", [html.Div(todos, id=f"{PRE}-todos", className="resultados")],
                   subtitulo=html.Span(todos_sub, id=f"{PRE}-s0-sub")),
        ui.seccion("El desempeño de cada modelo en la prueba", [
            ui.rejilla([
                ui.grafico(f"{PRE}-des", fuentes=FUENTES_PRUEBA, alto="alto"),
                ui.tarjeta([html.Div(id=f"{PRE}-acota", className="resultados")], titulo=html.Span(id=f"{PRE}-acota-titulo"),
                           fuentes=FUENTES_PRUEBAS + FUENTES_PLIEGUES[:2]),
            ], "2-1"),
        ], subtitulo=html.Span(id=f"{PRE}-s1-sub")),
        ui.seccion("La EBM frente a cada referencia", [
            ui.grafico(f"{PRE}-dif", fuentes=FUENTES_PRUEBAS),
        ], subtitulo=html.Span(id=f"{PRE}-s2-sub")),
        ui.seccion("Las pruebas, referencia por referencia", [
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
        ui.seccion(f"Los {pliegues} pliegues de la validación anidada", [
            ui.grafico(f"{PRE}-pli", fuentes=FUENTES_PLIEGUES),
        ], subtitulo="Lo que obtuvo cada modelo dentro del entrenamiento, antes de tocar la prueba, con los mismos pliegues para todos."),
        ui.seccion("Frente a la entrega anterior, con la etiqueta de la mediana", _entrega_anterior(),
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
    s1 = (f"{c['que']}: la mejor combinación de cada modelo según la validación anidada, medida una sola vez en los "
          f"{F.entero(D.resumen()['n_prueba'])} clientes de prueba.{alcance}")
    titulo_acota, acota = _acota(tarea)
    valida = _referencia(tarea, ref)
    return (_respuesta(tarea), s1, *_partes(_desempeno(tarea)), titulo_acota, acota, _resumen_veredictos(tarea),
            *_partes(_diferencias(tarea)), _opciones(tarea), (no_update if valida == ref else valida), _avisos(tarea),
            *_partes(_pliegues(tarea)))


@callback(Output(f"{PRE}-todos", "children"), Output(f"{PRE}-s0-sub", "children"), Input(f"{PRE}-tarea", "value"))
def _por_tarea_todos(tarea):
    """Los ocho modelos a la vez: el diagrama de diferencia crítica en clasificación y la tabla del MCS en regresión."""
    return _todos(tarea if tarea in TAREA else "clasificacion")


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
