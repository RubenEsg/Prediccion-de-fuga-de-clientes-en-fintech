"""Errores · ¿Dónde se equivoca la EBM y por qué?

Los errores de la EBM, el benchmark interpretable. Cuatro partes: los errores por segmento (cohorte de edad o número de productos activos) de la EBM y de XGBoost, el
histograma del error de la regresión que se separa en un pico por número de productos, cuatro clientes de ejemplo con
enlace a su explicación y, por cohorte, a quién contacta cada campaña y las cuatro cuentas de la matriz de confusión.

`active_products` aparece aquí solo para explicar el error: se excluyó a propósito de los predictores porque es la receta
del objetivo. Las cifras por segmento salen de datos/resultados/errores_segmentos.csv; el histograma, los cruces y las
cuentas por cohorte se calculan en vivo con los clientes de prueba (motor.clientes()), una sola vez (caché del módulo).
"""
import math
from functools import lru_cache

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, html

import datos_app as D
import formato as F
import motor as M
import ui


PRE = "er"
PANTALLA = "errores"
PREGUNTA = "¿Dónde se equivoca la EBM y por qué?"       # la de app.VISTAS

RIVAL = "xgboost"                      # el modelo de árboles con el que el informe compara los errores de la EBM
MODELOS = ("ebm", RIVAL)
EBM, XGB = D.NOMBRE_CORTO["ebm"], D.NOMBRE_MODELO[RIVAL]
AZAR = 0.5                             # AUC de ordenar al azar
PASO_EJE = 0.25                        # marcas de los ejes que van de 0 a 1
D_AUC, D_PUNTAJE, D_RMSE = 3, 3, 4     # decimales de las tablas de errores y de ejemplos del informe
D_TRAMO = 4                            # decimales de los bordes de un tramo del histograma
D_CASO = 4                             # decimales del puntaje real, el pronóstico y el umbral de un cliente: con tres, el
                                       # puntaje real del que se escapa y el umbral se escriben igual
CASI_NULO = 0.01                       # una centésima del puntaje: por debajo, un sesgo es «casi nulo» y una distancia al
                                       # umbral, «apenas»
ALTO_LIENZO = 320                      # px del lienzo «normal» de ui.grafico, para que ui.ancho_barra no pase de 24 px
ANCHO_MAXIMO = 0.36                    # tope del ancho de una barra (en unidades de categoría): caben dos por nivel
HUECO = 0.08                           # separación entre las dos barras de un nivel, como parte de su ancho
MARCAS_SESGO = 2                       # marcas a cada lado del cero en el eje del sesgo (a 400 px no caben más)
AIRE = 0.3                             # aire a la izquierda del eje del sesgo para el valor de las barras negativas
ANCHO_TRAMO = 0.0025                   # ancho de cada tramo del histograma del error, en unidades del puntaje
PASO_ERROR = 0.05                      # marcas del eje del error
TECHO = 1.18                           # el eje de cuentas llega un poco más arriba que el pico más alto (rótulos)
PALABRA = {1: "un", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis"}
TODOS = "todos"
RAYA_COHORTE = "–"                     # la raya de «59–60» en los datos
SIN_CORTE = "\u2060"                   # unión de palabras: tras la raya no se parte el renglón

SEGMENTO = {
    "cohorte": {"csv": "cohorte de edad", "rotulo": "Cohorte de edad", "cada": "cada cohorte de edad", "columna": "Cohorte"},
    "productos": {"csv": "productos activos", "rotulo": "Productos activos", "cada": "cada número de productos activos",
                  "columna": "Productos activos"},
}
MEDIDA = {
    "escapa": {"rotulo": "Riesgo alto que se escapa", "col": "fn", "unidad": "% del riesgo alto que se escapa"},
    "auc": {"rotulo": "AUC dentro del segmento", "col": "auc", "unidad": "AUC"},
    "sesgo": {"rotulo": "Sesgo de la regresión", "col": "sesgo", "unidad": "pronóstico − real"},
}
# Los tres clientes de la tabla de ejemplos del informe: (clave de resultados/ejemplos.csv, título, cómo se eligió)
CASOS = [
    ("riesgo alto con menor puntaje", "Se escapa", f"El cliente de riesgo alto con el menor puntaje de la {EBM}."),
    ("riesgo bajo con mayor puntaje", "Falsa alarma", f"El cliente de riesgo bajo con el mayor puntaje de la {EBM}."),
    ("mayor error de regresión", "Mayor error", "El cliente con el mayor error absoluto de la regresión."),
]
MAYOR_ERROR = CASOS[-1][0]
ACIERTO = ("acierto", "Bien predicho", "De los aciertos de la campaña, el de menor error de la regresión.")
SENAL = ["age", "app_logins_frequency"]                  # las dos variables que más pesan en la EBM
RESULTADO = dict(zip(("VP", "FP", "FN", "VN"), ui.MATRIZ_ROTULOS["celdas"]))

F_SEGMENTOS = D.fuente("resultados/errores_segmentos.csv")
F_CLIENTES = D.fuente("clientes_prueba.parquet")
F_PRUEBA = D.fuente("resultados/clasificacion_prueba.csv")
F_EJEMPLOS = D.fuente("resultados/ejemplos.csv")
F_META = D.fuente("metadatos.json")
F_EDA = D.fuente("eda/por_variable.csv")
F_EBM = D.fuente("modelos/ebm_clasificacion.json")


# --- Palabras -------------------------------------------------------------------------------------------------------------
def _cuantos(k):
    """«un», «dos», … «cinco» (el número con cifras si no tiene palabra)."""
    k = int(k)
    return PALABRA[k] if k in PALABRA else F.entero(k)


def _productos(k, palabra=True, activos=True):
    """«un producto activo», «cinco productos activos» (sin «activos» si activos=False); con palabra=False, el rótulo
    corto («1 producto», «5 productos»)."""
    k = int(k)
    plural = "" if k == 1 else "s"
    if not palabra:
        return f"{k} producto{plural}"
    return f"{_cuantos(k)} producto{plural}" + (f" activo{plural}" if activos else "")


def _anos(cohorte):
    """«59–60 años», sin que el renglón se pueda partir después de la raya."""
    return f"{str(cohorte).replace(RAYA_COHORTE, RAYA_COHORTE + SIN_CORTE)} años"


def _nivel(segmento, nivel):
    """Rótulo corto de un nivel: «59–60 años» o «2 productos»."""
    return _anos(nivel) if segmento == "cohorte" else _productos(nivel, palabra=False)


def _de_nivel(segmento, nivel):
    """El nivel dentro de una frase: «de 59–60 años» o «con un producto activo»."""
    return f"de {_anos(nivel)}" if segmento == "cohorte" else f"con {_productos(nivel)}"


def _capacidad():
    return F.pct(M.CAPACIDAD_INFORME, 0)


# --- Datos (una vez; solo lectura) ----------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _segmentos():
    """errores_segmentos.csv con los clientes de riesgo alto de cada nivel. Donde no hay ninguno no hay fracción que se
    escape (el informe la deja en blanco), igual que no hay AUC. No modificar el resultado: es la caché."""
    t = D.errores_segmentos()
    t["positivos"] = (t["n"] * t["prevalencia"]).round().astype(int)
    for m in MODELOS:
        t[f"fn_{m}"] = t[f"fn_{m}"].where(t["positivos"] > 0)
    return t


def _niveles(segmento):
    t = _segmentos()
    return t[t["segmento"] == SEGMENTO[segmento]["csv"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def _error():
    """El error de la regresión de la EBM (pronóstico − puntaje real) de cada cliente de prueba, su histograma (todos y
    por número de productos activos) y el resumen por nivel. Se calcula una vez."""
    C = M.clientes()
    e = (C["pron_ebm"] - C["churn_probability"]).astype(float)
    k = C["active_products"].astype(int)
    lo = math.floor(e.min() / ANCHO_TRAMO) * ANCHO_TRAMO
    n_tramos = int(math.ceil((e.max() - lo) / ANCHO_TRAMO))
    bordes = lo + ANCHO_TRAMO * np.arange(n_tramos + 1)
    niveles = sorted(int(v) for v in k.unique())
    cuentas = {v: np.histogram(e[k == v], bordes)[0] for v in niveles}
    resumen = pd.DataFrame({"n": e.groupby(k).size(), "sesgo": e.groupby(k).mean(), "minimo": e.groupby(k).min(),
                            "maximo": e.groupby(k).max(), "rmse": e.pow(2).groupby(k).mean().pow(0.5)}).loc[niveles]
    resumen["pico"] = [int(cuentas[v].max()) for v in niveles]
    return {"error": e, "bordes": bordes, "todos": np.histogram(e, bordes)[0], "cuentas": cuentas, "resumen": resumen,
            "niveles": niveles}


@lru_cache(maxsize=1)
def _cruce():
    """Clientes de riesgo alto según en cuál de las dos campañas entran (la de la EBM y la de XGBoost, con la capacidad
    del informe)."""
    C = M.clientes()
    alto = C["riesgo_alto"].to_numpy() == 1
    e, x = C["contactar_ebm"].to_numpy() == 1, C[f"contactar_{RIVAL}"].to_numpy() == 1
    return {"alto": int(alto.sum()), "ambos": int((alto & e & x).sum()), "solo_ebm": int((alto & e & ~x).sum()),
            "solo_rival": int((alto & ~e & x).sum()), "ninguno": int((alto & ~e & ~x).sum())}


@lru_cache(maxsize=1)
def _por_cohorte():
    """Las cuatro cuentas de la campaña de la EBM en cada cohorte (VP, FP, FN, VN) y los contactos de cada modelo."""
    C = M.clientes()
    t = C.groupby("cohorte")["resultado_ebm"].value_counts().unstack(fill_value=0).reindex(columns=list(RESULTADO), fill_value=0)
    t["n"] = C.groupby("cohorte").size()
    for m in MODELOS:
        t[f"contactos_{m}"] = C.groupby("cohorte")[f"contactar_{m}"].sum()
    return t.sort_index()


def _fila_acierto():
    """El cliente bien predicho en las dos tareas: de los aciertos de la campaña (riesgo alto y contactado), el de menor
    error absoluto de la regresión."""
    C = M.clientes()
    return int(_error()["error"][C["resultado_ebm"] == "VP"].abs().idxmin())


# --- 1. Por segmento ------------------------------------------------------------------------------------------------------
def _valor(medida, v):
    """El valor de una medida ya formateado («—» si no existe)."""
    if medida == "escapa":
        return F.pct(v, 0)
    return ui.num(v, D_AUC) if medida == "auc" else ui.con_signo(v, D_PUNTAJE)


def _eje(medida):
    """Rango y marcas del eje de valores. Son los mismos para los dos segmentos: así un sesgo de milésimas se ve como lo
    que es al lado de la escalera."""
    if medida != "sesgo":
        vals = [float(v) for v in np.arange(0, 1 + PASO_EJE / 2, PASO_EJE)]
        return [0, 1], vals, ([F.pct(v, 0) for v in vals] if medida == "escapa" else ui.marcas(0, 1, PASO_EJE)[1])
    t = _segmentos()
    v = pd.concat([t[f"sesgo_{m}"] for m in MODELOS])
    lo, hi = float(v.min()), float(v.max())
    paso = ui.paso_redondo(max(abs(lo), abs(hi)), MARCAS_SESGO)
    vals, textos = ui.marcas(math.floor(lo / paso) * paso, hi, paso, signo=True)
    return [lo - AIRE * (hi - lo), hi], vals, textos


def _al_pasar(segmento, medida, m, f):
    """Texto de una barra al pasar el puntero, con sus cifras ya formateadas."""
    quien = f"{D.NOMBRE_CORTO[m]} · {_nivel(segmento, f['nivel'])}"
    if medida == "escapa":
        return (f"{quien}: se le escapan <b>{F.entero(f[f'fn_n_{m}'])}</b> de {F.entero(f['positivos'])} clientes de riesgo alto "
                f"({F.pct(f[f'fn_{m}'])})")
    if medida == "auc":
        return (f"{quien}: AUC <b>{ui.num(f[f'auc_{m}'], D_AUC)}</b> con {F.entero(f['n'])} clientes "
                f"({F.entero(f['positivos'])} de riesgo alto)")
    return (f"{quien}: sesgo <b>{ui.con_signo(f[f'sesgo_{m}'], D_PUNTAJE)}</b> y RMSE {ui.num(f[f'rmse_{m}'], D_RMSE)} "
            f"en {F.entero(f['n'])} clientes")


def _figura_segmento(t, segmento, medida):
    """Barras horizontales, dos por nivel: la EBM sólida (azul) y XGBoost rayado (rosa), con su valor en la punta. Donde
    la medida no existe no hay barra."""
    col = MEDIDA[medida]["col"]
    filas = [f for _, f in t.iterrows()]
    debajo = [f"{F.entero(f['n'])} clientes" if medida == "sesgo" else f"{F.entero(f['positivos'])} de riesgo alto" for f in filas]
    cats = [f"{_nivel(segmento, f['nivel'])}<br>{d}" for f, d in zip(filas, debajo)]
    ancho = min(ui.ancho_barra(len(cats), ALTO_LIENZO), ANCHO_MAXIMO)
    fig = go.Figure()
    for m, forma, corrimiento in (("ebm", "barra", -ancho * (1 + HUECO)), (RIVAL, "rayada", ancho * HUECO)):
        v = [None if pd.isna(f[f"{col}_{m}"]) else float(f[f"{col}_{m}"]) for f in filas]
        fig.add_trace(go.Bar(y=cats, x=v, orientation="h", offset=corrimiento, name=D.NOMBRE_MODELO[m],
                             text=["" if x is None else _valor(medida, x) for x in v],
                             **ui.traza(D.ROL_VISTA_EBM[m], forma, ancho=ancho),
                             **ui.hover_texto([_al_pasar(segmento, medida, m, f) for f in filas])))
    rango, vals, textos = _eje(medida)
    fig.update_layout(barmode="overlay", hovermode="closest", margin_r=ui.MARGEN_ETIQUETAS)
    fig.update_xaxes(range=rango, tickvals=vals, ticktext=textos, showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, categoryorder="array", categoryarray=cats,
                     tickfont=ui.fuente_svg("tinta_2"), ticklabelstandoff=ui.px("--e-2"))
    if medida == "auc":
        ui.referencia(fig, x=AZAR, texto="azar")
    elif medida == "sesgo":
        ui.linea_cero(fig, "x")
    ui.unidad_eje(fig, MEDIDA[medida]["unidad"])
    return fig


def _titulo_segmento(t, segmento, medida):
    """El mensaje del gráfico, con sus cifras leídas de la tabla."""
    if medida == "escapa":
        total = int(t["fn_n_ebm"].sum())
        e, x = t.loc[t["fn_ebm"].idxmax()], t.loc[t[f"fn_{RIVAL}"].idxmax()]
        if int(e["fn_n_ebm"]) == total and segmento == "productos":
            return (f"Los {F.entero(total)} clientes de riesgo alto que se le escapan a la {EBM} tienen todos "
                    f"{_productos(e['nivel'])}")
        if e["nivel"] == x["nivel"]:
            return (f"Donde más se les escapa a los dos es en el riesgo alto {_de_nivel(segmento, e['nivel'])}: el "
                    f"{F.pct(e['fn_ebm'], 0)} a la {EBM} y el {F.pct(x[f'fn_{RIVAL}'], 0)} a {XGB}")
        return (f"A la {EBM} se le escapa el {F.pct(e['fn_ebm'], 0)} del riesgo alto {_de_nivel(segmento, e['nivel'])}; a {XGB}, "
                f"el {F.pct(x[f'fn_{RIVAL}'], 0)} del {_de_nivel(segmento, x['nivel'])}")
    if medida == "auc":
        if segmento == "cohorte":
            f = t.loc[t["auc_ebm"].idxmin()]
            return (f"La cohorte de {_anos(f['nivel'])} es la más difícil: AUC de {ui.num(f['auc_ebm'], D_AUC)} en la {EBM} y de "
                    f"{ui.num(f[f'auc_{RIVAL}'], D_AUC)} en {XGB}")
        v = pd.concat([t[f"auc_{m}"] for m in MODELOS]).dropna()
        return (f"Dentro de cada número de productos el orden es casi perfecto: AUC entre {ui.num(v.min(), D_AUC)} y "
                f"{ui.num(v.max(), D_AUC)}")
    if segmento == "cohorte":
        v = pd.concat([t[f"sesgo_{m}"] for m in MODELOS], ignore_index=True)
        return (f"Por cohorte de edad la regresión casi no tiene sesgo: {ui.con_signo(v[v.abs().idxmax()], D_PUNTAJE)} "
                f"como mucho")
    a, b = t.iloc[0], t.iloc[-1]
    return (f"El sesgo forma una escalera: de {ui.con_signo(a['sesgo_ebm'], D_PUNTAJE)} con {_productos(a['nivel'])} a "
            f"{ui.con_signo(b['sesgo_ebm'], D_PUNTAJE)} con {_cuantos(b['nivel'])}")


def _sub_segmento(segmento, medida):
    cada = SEGMENTO[segmento]["cada"]
    if medida == "escapa":
        return (f"Parte de los clientes de riesgo alto de {cada} que queda fuera de la campaña que contacta al {_capacidad()} "
                f"de mayor puntaje. Debajo de cada rótulo, cuántos clientes de riesgo alto tiene.")
    if medida == "auc":
        return (f"AUC calculada solo con los clientes de {cada}: qué tan bien los ordena el modelo entre ellos. La línea "
                f"marca el azar ({ui.num(AZAR, 1)}).")
    return (f"Media del pronóstico de la regresión menos el puntaje real en {cada}. Positivo: el modelo pronostica de más; "
            f"negativo, de menos.")


def _nota_segmento(t, segmento, medida):
    """Lo que hay que saber para leer el gráfico y no cabe en el título."""
    if segmento == "cohorte":
        if medida != "escapa":
            return None
        c = _por_cohorte()
        mayor = c["contactos_ebm"].idxmax()
        return (f"La {EBM} dedica a la cohorte de {_anos(mayor)} {F.entero(c.at[mayor, 'contactos_ebm'])} de sus "
                f"{F.entero(c['contactos_ebm'].sum())} contactos; {XGB} reparte más y le dedica "
                f"{F.entero(c.at[mayor, f'contactos_{RIVAL}'])}.")
    if medida == "sesgo":
        return (f"{XGB} repite la misma escalera: sin el número de productos activos, que se excluyó a propósito, ningún "
                f"modelo recupera esa parte del puntaje.")
    sin = F.enumerar(_cuantos(f["nivel"]) for _, f in t.iterrows() if f["positivos"] == 0)
    if not sin:
        return None
    que = "no hay nada que se pueda escapar" if medida == "escapa" else "el AUC no existe"
    return f"Con {sin} productos activos no hay clientes de riesgo alto en la prueba: {que} y por eso no llevan barra."


def _tabla_segmento(t, segmento):
    """La tabla gemela: todas las columnas de errores_segmentos.csv para el segmento elegido."""
    filas = []
    for _, f in t.iterrows():
        fila = [_nivel(segmento, f["nivel"]), F.entero(f["n"]), f"{F.entero(f['positivos'])} ({F.pct(f['prevalencia'])})"]
        fila += [ui.num(f[f"auc_{m}"], D_AUC) for m in MODELOS]
        fila += [f"{F.entero(f[f'fn_n_{m}'])} ({F.pct(f[f'fn_{m}'])})" if f["positivos"] else F.RAYA for m in MODELOS]
        fila += [ui.num(f[f"rmse_{m}"], D_RMSE) for m in MODELOS]
        fila += [ui.con_signo(f[f"sesgo_{m}"], D_PUNTAJE) for m in MODELOS]
        filas.append(fila)
    cab = [SEGMENTO[segmento]["columna"], "Clientes", "De riesgo alto"]
    for que in ("AUC", "Se escapan", "RMSE", "Sesgo"):
        cab += [f"{que} · {EBM}", f"{que} · {XGB}"]
    return ui.tabla(cab, filas, compacta=True, ocultar_movil=(7, 8))


@callback(Output(f"{PRE}-seg", "figure"), Output(f"{PRE}-seg-titulo", "children"), Output(f"{PRE}-seg-sub", "children"),
          Output(f"{PRE}-seg-tabla", "children"), Output(f"{PRE}-seg-nota", "children"), Output(f"{PRE}-seg-lienzo", "aria-label"),
          Input(f"{PRE}-segmento", "value"), Input(f"{PRE}-medida", "value"))
def _por_segmento(segmento, medida):
    segmento = segmento if segmento in SEGMENTO else "cohorte"
    medida = medida if medida in MEDIDA else "escapa"
    t = _niveles(segmento)
    titulo = _titulo_segmento(t, segmento, medida)
    col = MEDIDA[medida]["col"]
    etiqueta = f"{titulo}. " + "; ".join(
        f"{_nivel(segmento, f['nivel'])}: {EBM} {_valor(medida, f[f'{col}_ebm'])}, {XGB} {_valor(medida, f[f'{col}_{RIVAL}'])}"
        for _, f in t.iterrows()) + "."
    return (_figura_segmento(t, segmento, medida), titulo, _sub_segmento(segmento, medida), _tabla_segmento(t, segmento),
            _nota_segmento(t, segmento, medida), etiqueta)


def _captura():
    """Tarjeta que acompaña al gráfico: la captura global de los dos modelos es casi la misma y los clientes, no."""
    t = D.prueba("clasificacion")
    e, x = D.fila(t, modelo="ebm"), D.fila(t, modelo=RIVAL)
    c = _cruce()
    barras = html.Div([ui.barra_rayas(D.NOMBRE_MODELO["ebm"], F.pct(e["recall_op"]), e["recall_op"], D.ROL_VISTA_EBM["ebm"]),
                       ui.barra_rayas(XGB, F.pct(x["recall_op"]), x["recall_op"], D.ROL_VISTA_EBM[RIVAL])], className="rayas-lista")
    cruce = ui.ficha([("Los alcanzan los dos modelos", F.entero(c["ambos"])), (f"Solo la {EBM}", F.entero(c["solo_ebm"])),
                      (f"Solo {XGB}", F.entero(c["solo_rival"])), ("Se les escapan a los dos", F.entero(c["ninguno"]))])
    return ui.tarjeta([
        barras, cruce,
        ui.nota(f"De los {F.entero(c['alto'])} clientes de riesgo alto de la prueba, {F.entero(c['solo_ebm'] + c['solo_rival'])} "
                f"entran en la campaña de un modelo y quedan fuera de la del otro."),
    ], titulo=f"La captura global es casi la misma: {F.pct(e['recall_op'])} y {F.pct(x['recall_op'])}",
        subtitulo=f"Clientes de riesgo alto que alcanza cada modelo al contactar al {_capacidad()} de mayor puntaje.",
        fuentes=[F_PRUEBA, F_CLIENTES], id=f"{PRE}-captura")


# --- 2. La escalera del error ---------------------------------------------------------------------------------------------
def _figura_escalera(nivel):
    """Histograma del error de la regresión de todos los clientes de prueba; con un nivel elegido, sus clientes van en azul
    sobre el total en gris claro. Encima de cada pico, su número de productos activos."""
    h = _error()
    bordes, r = h["bordes"], h["resumen"]
    centros = (bordes[:-1] + bordes[1:]) / 2
    tramos = [f"Error de {ui.con_signo(a, D_TRAMO)} a {ui.con_signo(b, D_TRAMO)}" for a, b in zip(bordes[:-1], bordes[1:])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centros, y=h["todos"], name="Todos los clientes de prueba",
                         **ui.traza("modelo" if nivel is None else "contexto", "barra", ancho=ANCHO_TRAMO),
                         **ui.hover_texto([f"{t}: <b>{F.entero(c)}</b> clientes" for t, c in zip(tramos, h["todos"])])))
    if nivel is not None:
        fig.add_trace(go.Bar(x=centros, y=h["cuentas"][nivel], name=f"Con {_productos(nivel)}",
                             **ui.traza("modelo", "barra", ancho=ANCHO_TRAMO),
                             **ui.hover_texto([f"{t}: <b>{F.entero(c)}</b> de los {F.entero(n)} clientes del tramo tienen {_productos(nivel)}"
                                               for t, c, n in zip(tramos, h["cuentas"][nivel], h["todos"])])))
    for k in h["niveles"]:
        fig.add_annotation(x=float(r.at[k, "sesgo"]), y=int(r.at[k, "pico"]), text=str(k), showarrow=False, yanchor="bottom",
                           yshift=ui.px("--e-1"), bgcolor=ui.ROL["superficie"],
                           font=ui.fuente_svg("tinta" if nivel in (None, k) else "tinta_3", "--t-s"))
    vals, textos = ui.marcas(math.ceil(bordes[0] / PASO_ERROR) * PASO_ERROR, bordes[-1], PASO_ERROR, signo=True)
    fig.update_layout(barmode="overlay", bargap=0, barcornerradius=0, hovermode="closest")
    fig.update_xaxes(range=[float(bordes[0]), float(bordes[-1])], tickvals=vals, ticktext=textos)
    fig.update_yaxes(range=[0, float(h["todos"].max()) * TECHO])
    ui.linea_cero(fig, "x")
    ui.unidad_eje(fig, "clientes por tramo de error")
    return fig


def _tabla_escalera():
    """La tabla gemela: el sesgo de cada nivel calculado en vivo al lado del que trae el archivo del informe."""
    h = _error()
    del_informe = _niveles("productos").set_index("nivel")["sesgo_ebm"]
    filas = [[str(k), F.entero(f["n"]), ui.con_signo(f["sesgo"], D_PUNTAJE), ui.con_signo(del_informe.get(str(k)), D_PUNTAJE),
              ui.num(f["rmse"], D_RMSE), ui.con_signo(f["minimo"], D_PUNTAJE), ui.con_signo(f["maximo"], D_PUNTAJE)]
             for k, f in h["resumen"].iterrows()]
    e = h["error"]
    total = ["Todos", F.entero(len(e)), ui.con_signo(e.mean(), D_PUNTAJE), F.RAYA, ui.num(math.sqrt((e ** 2).mean()), D_RMSE),
             ui.con_signo(e.min(), D_PUNTAJE), ui.con_signo(e.max(), D_PUNTAJE)]
    return ui.tabla(["Productos activos", "Clientes", "Sesgo (en vivo)", "Sesgo (informe)", "RMSE", "Error mínimo", "Error máximo"],
                    filas, total=total, compacta=True, ocultar_movil=(5, 6))


@callback(Output(f"{PRE}-esc", "figure"), Output(f"{PRE}-esc-titulo", "children"), Output(f"{PRE}-esc-leyenda", "children"),
          Output(f"{PRE}-esc-lienzo", "aria-label"), Input(f"{PRE}-nivel", "value"))
def _escalera(nivel):
    h = _error()
    r = h["resumen"]
    try:
        nivel = int(nivel)
    except (TypeError, ValueError):
        nivel = None
    nivel = nivel if nivel in h["niveles"] else None
    resumen = "; ".join(f"{_productos(k, palabra=False)}: sesgo medio {ui.con_signo(f['sesgo'], D_PUNTAJE)} en {F.entero(f['n'])} clientes"
                        for k, f in r.iterrows())
    if nivel is None:
        titulo = f"El error tiene {_cuantos(len(r))} picos: uno por cada número de productos activos"
        return _figura_escalera(None), titulo, None, f"{titulo}. {resumen}."
    f = r.loc[nivel]
    cifra = ui.con_signo(f["sesgo"], D_PUNTAJE)
    if abs(f["sesgo"]) < CASI_NULO:
        titulo = f"Con {_productos(nivel)}, la {EBM} pronostica casi sin sesgo: {cifra} de media en {F.entero(f['n'])} clientes"
    else:
        sentido = "de más" if f["sesgo"] > 0 else "de menos"
        titulo = f"Con {_productos(nivel)}, la {EBM} pronostica {sentido}: sesgo medio de {cifra} en {F.entero(f['n'])} clientes"
    leyenda = ui.leyenda([(f"Clientes con {_productos(nivel)}", "modelo", "barra"),
                          (f"Los {F.entero(len(h['error']))} clientes de prueba", "contexto", "barra")])
    return _figura_escalera(nivel), titulo, leyenda, f"{titulo}. {resumen}."


def _porque():
    """Tarjeta que explica la escalera: el puntaje se generó con la variable que se excluyó a propósito."""
    eda = D.tabla("eda/por_variable.csv")
    eda = eda[eda["variable"] == "active_products"].sort_values("x")
    primero, ultimo = eda.iloc[0], eda.iloc[-1]
    paso = (primero["media_objetivo"] - ultimo["media_objetivo"]) / (len(eda) - 1)
    t = _niveles("productos")
    auc = pd.concat([t[f"auc_{m}"] for m in MODELOS]).dropna()
    centro = t.loc[t["sesgo_ebm"].abs().idxmin(), "nivel"]
    return ui.tarjeta([
        ui.parrafo(f"En el entrenamiento, la media del puntaje baja de {ui.num(primero['media_objetivo'], D_PUNTAJE)} con "
                   f"{_productos(primero['x'])} a {ui.num(ultimo['media_objetivo'], D_PUNTAJE)} con {_cuantos(ultimo['x'])}: unos "
                   f"{ui.num(paso, D_PUNTAJE)} por cada producto adicional, un peldaño casi constante."),
        ui.parrafo(f"Por eso el número de productos activos se excluyó a propósito de los predictores: es la receta del "
                   f"objetivo, no una señal que aprender. Sin verlo, la {EBM} pronostica a todos cerca del nivel de "
                   f"{_cuantos(centro)} productos, y el error es la parte del puntaje que depende de esa variable."),
        ui.cifra("AUC dentro de cada número de productos", f"{ui.num(auc.min(), D_AUC)} a {ui.num(auc.max(), D_AUC)}",
                 nota=f"En los niveles con clientes de riesgo alto, en la {EBM} y en {XGB}: entre clientes con los mismos "
                      f"productos el orden es casi perfecto.", mini=True),
        html.Div(ui.boton("Ver la variable excluida en Datos", href="/eda", icono="metodo"), className="botonera"),
    ], titulo="El puntaje se generó con la variable excluida", fuentes=[F_EDA, F_SEGMENTOS], id=f"{PRE}-porque")


# --- 3. Cuatro clientes ---------------------------------------------------------------------------------------------------
def _sus_productos(k):
    """«su único producto activo» o «sus cuatro productos activos»."""
    k = int(k)
    return "su único producto activo" if k == 1 else f"sus {_cuantos(k)} productos activos"


def _lectura(clave, fila, cl):
    """Una frase que dice por qué el caso sale así, armada con sus datos (el puntaje real, el pronóstico y el umbral, con
    los mismos decimales que en la ficha)."""
    k = int(cl["active_products"])
    real, pronostico = ui.num(cl["churn_probability"], D_CASO), ui.num(cl["pron_ebm"], D_CASO)
    if clave == MAYOR_ERROR:
        return (f"Con {_productos(k)}, que el modelo no ve, su puntaje real es {real}; el pronóstico de la {EBM} es "
                f"{pronostico}: el mayor error de los {F.entero(M.meta()['n_prueba'])} clientes de prueba.")
    if clave == ACIERTO[0]:
        t = _niveles("productos")
        centro = int(t.loc[t["sesgo_ebm"].abs().idxmin(), "nivel"])
        donde = ", el nivel alrededor del cual el modelo pronostica a todos" if k == centro else ""
        coincide = "coincide" if pronostico == real else "casi coincide"
        return f"Con {_productos(k)}{donde}, el pronóstico de la {EBM}, {pronostico}, {coincide} con su puntaje real, {real}."
    _, con = M.explicar(M.clientes().loc[[fila]], "clasificacion")
    aporte = con.iloc[0][SENAL]
    sentido = "alto" if (aporte > 0).all() else "bajo" if (aporte < 0).all() else None
    umbral = M.meta()["etiqueta"]["umbral"]
    arriba = cl["churn_probability"] > umbral
    apenas = "apenas " if abs(cl["churn_probability"] - umbral) < CASI_NULO else ""
    queda = (f"con {_sus_productos(k)} su puntaje real, {real}, queda {apenas}{'por encima' if arriba else 'por debajo'} del "
             f"umbral, {ui.num(umbral, D_CASO)}.")
    if sentido is None:
        return f"Para la {EBM}, su edad y su uso de la app no apuntan en el mismo sentido; {queda}"
    nexo = "pero" if (sentido == "alto") != arriba else "y"
    return f"Para la {EBM}, su edad y su uso de la app apuntan a riesgo {sentido}, {nexo} {queda}"


def _caso(clave, titulo, eleccion, fila):
    """Tarjeta de un cliente de prueba: sus variables de señal, lo que el modelo no ve, su puntaje real frente al umbral,
    lo que hizo el clasificador y lo que pronosticó la regresión."""
    cl = M.clientes().loc[fila]
    error = float(_error()["error"].loc[fila])
    pares = [
        (M.nombre("age"), f"{F.entero(cl['age'])} {M.variables().at['age', 'unidad']}"),
        (M.nombre("app_logins_frequency"), F.entero(cl["app_logins_frequency"])),
        (M.nombre("satisfaction_score"), F.entero(cl["satisfaction_score"])),
        ("Productos activos (el modelo no los ve)", F.entero(cl["active_products"])),
        ("Puntaje real", ui.num(cl["churn_probability"], D_CASO)),
        ("Umbral de riesgo alto", ui.num(M.meta()["etiqueta"]["umbral"], D_CASO)),
        ("Riesgo alto real", "sí" if cl["riesgo_alto"] == 1 else "no"),
        (f"Puntaje de la {EBM}", ui.num(cl["p_ebm"], D_PUNTAJE)),
        (f"Puntaje de {XGB}", ui.num(cl[f"p_{RIVAL}"], D_PUNTAJE)),
        ("En la campaña", RESULTADO[cl["resultado_ebm"]]),
        (f"Pronóstico de la {EBM}", ui.num(cl["pron_ebm"], D_CASO)),
        ("Error de la regresión", ui.con_signo(error, D_CASO)),
    ]
    return ui.tarjeta([
        ui.ficha(pares),
        ui.nota(_lectura(clave, fila, cl)),
        html.Div(ui.boton("Ver su explicación", href=f"/modelos/cliente?fila={fila}", icono="lupa"), className="botonera"),
    ], titulo=titulo, subtitulo=eleccion, id=f"{PRE}-caso-{fila}")


def _casos():
    filas = M.meta()["ejemplos"]
    casos = [(clave, titulo, eleccion, int(filas[clave])) for clave, titulo, eleccion in CASOS if clave in filas]
    casos.append((*ACIERTO, _fila_acierto()))
    return [_caso(*c) for c in casos]


# --- 4. La clasificación por cohorte --------------------------------------------------------------------------------------
def _cohortes():
    """Una tarjeta por cohorte: qué parte de los contactos de cada campaña le toca y las cuatro cuentas de la matriz de
    confusión de la campaña de la EBM (las cifras de cuatro dígitos no caben en ui.matriz a un tercio del ancho)."""
    c = _por_cohorte()
    tarjetas = []
    for i, (cohorte, f) in enumerate(c.iterrows()):
        alto = int(f["VP"] + f["FN"])
        barras = []
        for m in MODELOS:
            parte = f[f"contactos_{m}"] / c[f"contactos_{m}"].sum()
            barras.append(ui.barra_rayas(f"Contactos de {D.en_frase(m)}",
                                         f"{F.entero(f[f'contactos_{m}'])} ({F.pct(parte)})", parte, D.ROL_VISTA_EBM[m]))
        cuentas = [("Aciertos", f["VP"]), ("Falsas alarmas", f["FP"]), ("Se escapan", f["FN"]), ("Bien descartados", f["VN"])]
        tarjetas.append(ui.tarjeta([
            html.Div(barras, className="rayas-lista"),
            ui.ficha([(rotulo, F.entero(v)) for rotulo, v in cuentas] + [("Parte de su riesgo alto que se escapa", F.pct(f["FN"] / alto))]),
        ], titulo=_anos(cohorte), subtitulo=f"{F.entero(f['n'])} clientes, {F.entero(alto)} de riesgo alto ({F.pct(alto / f['n'])})",
            id=f"{PRE}-cohorte-{i}"))
    return tarjetas


def _textos():
    """Los mensajes de las secciones y el subtítulo de la pantalla, con sus cifras leídas de los datos."""
    coh, pro = _niveles("cohorte"), _niveles("productos")
    dificil = coh.loc[coh["auc_ebm"].idxmin()]
    a, b = pro.iloc[0], pro.iloc[-1]
    escalera = (f"de {ui.con_signo(a['sesgo_ebm'], D_PUNTAJE)} con {_productos(a['nivel'], activos=False)} a "
                f"{ui.con_signo(b['sesgo_ebm'], D_PUNTAJE)} con {_cuantos(b['nivel'])}")
    c = _por_cohorte()
    peor, mayor = c["FN"].idxmax(), c["contactos_ebm"].idxmax()
    peso = M.importancia("clasificacion")
    edad = (f"La edad es la variable que más pesa en la {EBM} ({F.pct(peso.iloc[0])} de la importancia) y su campaña"
            if peso.index[0] == "age" else f"La campaña de la {EBM}")
    return {
        "subtitulo": (f"Se equivoca donde manda la variable que se excluyó. La cohorte de {_anos(dificil['nivel'])} es la más difícil "
                      f"(AUC {ui.num(dificil['auc_ebm'], D_AUC)} dentro de la cohorte) y el error de la regresión forma una escalera "
                      f"según los productos activos ({escalera}): es la parte del puntaje que ningún modelo puede recuperar sin "
                      f"esa variable."),
        "segmentos": f"La {EBM} y {XGB} alcanzan casi la misma captura con clientes distintos",
        "segmentos_sub": (f"«Se escapa» es un cliente de riesgo alto que queda fuera de la campaña que contacta al {_capacidad()} de "
                          f"mayor puntaje, la regla del informe. Elija el segmento y qué mirar."),
        "escalera": f"El error de la regresión es una escalera: {escalera}",
        "escalera_sub": ("El número de productos activos no es un predictor: se excluyó a propósito y aquí solo sirve para "
                         "separar el error. Elija un número de productos y vea qué pico le corresponde."),
        "casos": "Cuatro clientes: tres errores representativos y uno bien predicho",
        "casos_sub": (f"Los tres primeros son los de la tabla de ejemplos del informe. El «puntaje de la {EBM}» es el del "
                      f"clasificador (de cero a uno, sin recalibrar: sirve para ordenar); el «pronóstico» es el de la regresión, "
                      f"en la escala del puntaje real."),
        "cohortes": (f"De los {F.entero(c['FN'].sum())} clientes de riesgo alto que se le escapan a la {EBM}, "
                     f"{F.entero(c.at[peor, 'FN'])} son de {_anos(peor)}"),
        "cohortes_sub": (f"{edad} va sobre todo a la cohorte de {_anos(mayor)}: "
                         f"{F.pct(c.at[mayor, 'contactos_ebm'] / c['contactos_ebm'].sum())} de sus contactos, frente al "
                         f"{F.pct(c.at[mayor, f'contactos_{RIVAL}'] / c[f'contactos_{RIVAL}'].sum())} de {XGB}. Las cuentas son las de "
                         f"la campaña de la {EBM}: un acierto es un cliente de riesgo alto contactado; una falsa alarma, un "
                         f"contactado que no lo era."),
    }


# --- La pantalla ----------------------------------------------------------------------------------------------------------
def _pantalla():
    textos = _textos()
    h = _error()
    segmento = ui.control("Segmento:", ui.segmentado(f"{PRE}-segmento", [{"label": s["rotulo"], "value": k} for k, s in SEGMENTO.items()],
                                                    "cohorte"))
    medida = ui.control("Qué mirar:", ui.segmentado(f"{PRE}-medida", [{"label": m["rotulo"], "value": k} for k, m in MEDIDA.items()],
                                                   "escapa"))
    nivel = ui.control("Productos activos:", ui.segmentado(
        f"{PRE}-nivel", [{"label": "Todos", "value": TODOS}] + [{"label": str(k), "value": str(k)} for k in h["niveles"]], TODOS))
    return ui.pantalla([
        ui.encabezado(PREGUNTA, textos["subtitulo"], pantalla=PANTALLA),
        ui.seccion(textos["segmentos"], [
            ui.rejilla([
                ui.grafico(f"{PRE}-seg", fuentes=[F_SEGMENTOS, F_CLIENTES],
                           leyenda=[(D.NOMBRE_MODELO["ebm"], D.ROL_VISTA_EBM["ebm"], "barra"), (XGB, D.ROL_VISTA_EBM[RIVAL], "rayada")]),
                _captura(),
            ], "2-1"),
        ], subtitulo=textos["segmentos_sub"], controles=[segmento, medida]),
        ui.seccion(textos["escalera"], [
            ui.rejilla([
                ui.grafico(f"{PRE}-esc", tabla=_tabla_escalera(), fuentes=[F_CLIENTES, F_SEGMENTOS], alto="alto",
                           subtitulo=(f"Error de la regresión de la {EBM} (pronóstico menos puntaje real) de los "
                                      f"{F.entero(len(h['error']))} clientes de prueba. A la derecha del cero el modelo pronostica "
                                      f"de más. El número sobre cada pico es el de productos activos de sus clientes.")),
                _porque(),
            ], "2-1"),
        ], subtitulo=textos["escalera_sub"], controles=[nivel]),
        ui.seccion(textos["casos"], [ui.cifras(_casos())], subtitulo=textos["casos_sub"],
                   fuentes=[F_EJEMPLOS, F_META, F_CLIENTES, F_EBM]),
        ui.seccion(textos["cohortes"], [ui.rejilla(_cohortes(), "3")], subtitulo=textos["cohortes_sub"],
                   fuentes=[F_CLIENTES, F_EBM]),
        ui.pie([ui.nota("«Riesgo alto» es el cuarto superior del puntaje de propensión del conjunto de datos, no fuga observada. "
                        "Los errores se miden en los clientes de prueba, que ningún modelo vio al entrenarse.")]),
    ])


def layout(**_):
    try:
        return _pantalla()
    except OSError as e:           # falta un archivo de datos/: se dice, no se rompe la pantalla
        return ui.pantalla([ui.encabezado(PREGUNTA, pantalla=PANTALLA),
                            ui.aviso(f"No se pudo leer {getattr(e, 'filename', None) or 'un archivo de datos'}. Genere la carpeta "
                                     f"de datos y reinicie la app.", "critico", "Faltan los datos de esta pantalla")])
