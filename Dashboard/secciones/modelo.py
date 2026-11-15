"""Modelo · ¿Qué modelo es y cómo decide?

La EBM por dentro, leída del modelo en vivo: qué variables pesan (cuota de importancia), la curva que el modelo aprendió
para cada una de las 48 (su función de forma, con la desviación entre bolsas y los clientes de entrenamiento que la
respaldan) y las cifras del modelo completo (términos, parámetros, intercepto, hiperparámetros y costo).

Toda cifra sale del motor (datos/modelos/ebm_*.json), de datos/metadatos.json, de datos/variables.csv, de
datos/eda/por_variable.csv (recuentos del entrenamiento) o de datos/resultados/. Las constantes de abajo son de
presentación (cuántas barras, cuándo un eje es logarítmico), no cifras del proyecto.
"""
import math
import unicodedata

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, clientside_callback, html, no_update

import datos_app as D
import formato as F
import motor as M
import ui


PRE = "mo"
PREGUNTA = "¿Qué modelo es y cómo decide?"
VARIABLE_INICIAL = "age"

N_FOCO = 4                # las variables que el informe nombra: van sólidas
PALABRA = {3: "tres", 4: "cuatro", 5: "cinco"}
N_PRIMERAS = 2            # las variables que nombra el título de la sección de importancia
MITAD = 0.5               # … si entre las dos pasan de esta parte de la importancia
N_BARRAS = 8              # variables con barra propia en el gráfico de importancia (tantas como filas tiene la tabla del informe)
LARGO_ROTULO = 18         # un rótulo de eje más largo se parte en dos líneas
POCOS_VALORES = 30        # hasta aquí una variable numérica se dibuja solo en los valores que existen
RAZON_LOG = 50            # máximo / mínimo desde el que una variable continua positiva va en eje logarítmico
MAX_VERTICALES = 5        # hasta aquí las categorías van en columnas; con más, en barras horizontales
MAX_FILAS = 40            # una curva con más tramos se resume en la tabla gemela
GRUPOS_TABLA = 20         # … en este número de grupos de tramos consecutivos
MAX_MARCAS = 14           # hasta aquí se rotula cada valor del eje
SEPARACION = 0.06         # dos valores más juntos que esta fracción del eje comparten marca («29 y 30»)
FRACCION_BARRA = 0.7      # ancho de una barra respecto de la distancia al valor vecino
HOLGURA_EJE = 0.08        # aire a cada lado del eje de una variable de pocos valores, como fracción de su recorrido
MARCAS_POR_LADO = 4       # marcas del eje del efecto a cada lado del cero
MAX_ESTIRAR = 2           # cuánto puede estirar la recta el eje del efecto de una variable de pocos valores
AIRE = 0.08               # aire del eje del efecto, como fracción de su recorrido
CENTRAL = 0.90            # parte de los clientes entre los percentiles 5 y 95
PAREJO = 0.9              # si el valor con menos clientes tiene al menos esta parte del que más tiene, el título da el intervalo
MAX_EMPATE = 3            # hasta aquí se nombran los valores empatados en el título; con más, se cuentan
ESPERA_SALTO_MS = 250     # tras el clic en una barra de importancia, lo que se espera antes de llevar la curva a la vista
COLAS = (("p1", "p99", 0.01), ("p5", "p95", 0.05))      # percentiles de variables.csv y la parte que queda más allá
MILLON = 1_000_000
DECIMALES_MILLON = 3      # decimales de una cifra en millones por debajo de cien millones
CIFRAS = 4                # cifras significativas de un valor en eje logarítmico
MAX_DECIMALES = 5         # tope de decimales al escribir el borde de un tramo
TRAMO_FINO = 0.004        # un tramo más angosto que esta fracción del eje se dibuja con un solo punto
SIN_LIMITE = -1           # hoverdistance de Plotly: la etiqueta flotante es la del punto más cercano, esté a la distancia que esté
BASE = 10
SUPERINDICE = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
ALTO_LIENZO = {"normal": 320, "bajo": 240, "alto": 420}      # alto de cada lienzo de ui.grafico (estilos.css)
RIVAL, BOSQUE = "xgboost", "random_forest"
EBM, LOGISTICA = D.NOMBRE_CORTO["ebm"], D.NOMBRE_MODELO["logistica"]              # en rótulos, leyendas y tablas
LA_EBM, LA_LOGISTICA = D.en_frase("ebm"), D.en_frase("logistica")                 # dentro de una frase

TAREA = {
    "clasificacion": {"rotulo": "Clasificación (riesgo alto)", "nombre": "clasificación", "unidad": "contribución al logit del riesgo alto",
                      "en": "en el logit", "mas": "más riesgo", "menos": "menos"},
    "regresion": {"rotulo": "Regresión (puntaje)", "nombre": "regresión", "unidad": "contribución al pronóstico",
                  "en": "en el pronóstico", "mas": "un pronóstico mayor", "menos": "uno menor"},
}
HIPER = {"interactions": "Interacciones por pares (K)", "learning_rate": "Tasa de aprendizaje (η)", "max_leaves": "Hojas por árbol (L)",
         "min_samples_leaf": "Mínimo de clientes por hoja (m)", "outer_bags": "Bolsas externas (B)"}

FUENTES_EBM = [D.fuente(f"modelos/ebm_{t}.json") for t in TAREA]
FUENTES_IMPORTANCIA = FUENTES_EBM + [D.fuente("resultados/importancia_permutacion_ebm.csv"), D.fuente("variables.csv")]
FUENTES_CURVA = FUENTES_EBM + [D.fuente("modelos/logistica_clasificacion.json"), D.fuente("variables.csv"), D.fuente("eda/por_variable.csv")]
FUENTES_DENTRO = [D.fuente("metadatos.json"), D.fuente("resultados/costo.csv"), D.fuente("entrega3/corridas.csv")] + FUENTES_EBM


# --- Ayudas de texto y de formato ---------------------------------------------------------------------------------------
def _tarea(tarea):
    """La tarea elegida o, si el control llega vacío o con otra cosa, la clasificación."""
    return tarea if isinstance(tarea, str) and tarea in TAREA else "clasificacion"


def _variable(variable):
    """La variable elegida o, si el control llega vacío o con otra cosa, la inicial."""
    return variable if isinstance(variable, str) and variable in M.variables().index else VARIABLE_INICIAL


def _partir(texto, largo=LARGO_ROTULO):
    """Rótulo largo de un eje en dos líneas, cortado en el espacio más cercano a la mitad."""
    cortes = [i for i, ch in enumerate(texto) if ch == " "]
    if len(texto) <= largo or not cortes:
        return texto
    i = min(cortes, key=lambda c: abs(c - len(texto) // 2))
    return texto[:i] + "<br>" + texto[i + 1:]


def _decimales(tope):
    """Decimales para escribir un efecto con dos cifras significativas en su valor más grande (entre 2 y 5)."""
    if not tope > 0:
        return 2
    return int(min(5, max(2, math.ceil(-math.log10(tope)) + 1)))


def _corto(x, d=2):
    """Número con hasta `d` decimales y sin ceros de relleno: 29 · 0,2 · 0,33."""
    s = ui.num(x, d)
    return s.rstrip("0").rstrip(",") if "," in s else s


def _valor(x, d=2, significativas=False, millones=False):
    """Valor de una variable en es-CO. Con `millones` (las variables que llegan al millón), en millones: «1,759 M». Con
    `significativas` (eje logarítmico), con CIFRAS cifras significativas y sin ceros de relleno."""
    x = float(x)
    if millones:
        return f"{ui.num(x / MILLON, DECIMALES_MILLON if abs(x) < 100 * MILLON else 0)} M"
    if significativas and x:
        return _corto(x, max(0, CIFRAS - 1 - math.floor(math.log10(abs(x)))))
    return ui.num(x, d)


def _clientes(n):
    return f"{F.entero(n)} cliente" + ("" if int(n) == 1 else "s")


def _sin_tildes(texto):
    """El texto sin tildes ni diéresis, para que la búsqueda del selector encuentre «satisfaccion»."""
    return "".join(ch for ch in unicodedata.normalize("NFD", texto) if unicodedata.category(ch) != "Mn")


def _y(siguiente):
    """La conjunción delante de una palabra: «e» si empieza por el sonido i («e inicios»), «y» si no."""
    s = _sin_tildes(siguiente).lower()
    suena_i = s.startswith(("i", "hi")) and not s.startswith(("ia", "ie", "io", "hia", "hie", "hio"))
    return " e " if suena_i else " y "


def _titulo_clientes(nombres, n, femenino=False):
    """Título del gráfico de clientes de una variable de pocos valores o de categorías, siempre en clientes: el intervalo
    si todos tienen un número parecido; si no, quién tiene más y quién menos, con los empates."""
    cosa, cosas, el, los, cada = (("categoría", "categorías", "La", "Las", "cada una") if femenino
                                  else ("valor", "valores", "El", "Los", "cada uno"))
    mas, menos = max(n), min(n)
    if mas == menos:
        return f"{los} {len(n)} {cosas} tienen {_clientes(mas)} de entrenamiento {cada}"
    if menos >= PAREJO * mas:
        return f"{los} {len(n)} {cosas} tienen entre {F.entero(menos)} y {F.entero(mas)} clientes de entrenamiento {cada}"

    def quienes(cuantos):
        return [g for g, x in zip(nombres, n) if x == cuantos]

    arriba, abajo = quienes(mas), quienes(menos)
    if len(arriba) == 1:
        primero = f"{el} {cosa} con más clientes de entrenamiento es {arriba[0]} ({_clientes(mas)})"
    elif len(arriba) <= MAX_EMPATE:
        primero = f"{los} {cosas} con más clientes de entrenamiento son {F.enumerar(arriba)} ({_clientes(mas)} {cada})"
    else:
        primero = f"{los} {len(arriba)} {cosas} con más clientes de entrenamiento tienen {_clientes(mas)} {cada}"
    if len(abajo) == 1:
        segundo = f"{el.lower()} que menos tiene, {abajo[0]} ({_clientes(menos)})"
    elif len(abajo) <= MAX_EMPATE:
        segundo = f"{los.lower()} que menos tienen, {F.enumerar(abajo)} ({_clientes(menos)} {cada})"
    else:
        segundo = f"{los.lower()} {len(abajo)} que menos tienen, {_clientes(menos)} {cada}"
    return f"{primero}; {segundo}"


def _banda(tarea, que):
    """Qué es la banda (o la línea fina) de una curva: la desviación entre las bolsas del modelo."""
    b = _bolsas(tarea)
    return f" {que} es la desviación entre las {b} bolsas del modelo ({b} ajustes sobre submuestras, que se promedian), no un intervalo de confianza."


def _cientifico(x):
    """Un número muy pequeño en notación científica y es-CO: 4,4 × 10⁻¹⁶."""
    if not x:
        return "0"
    e = math.floor(math.log10(abs(x)))
    return f"{ui.num(x / BASE ** e, 1)} × {BASE}{str(e).translate(SUPERINDICE)}"


def _bolsas(tarea):
    return int(M.meta()["ebm"][tarea]["hiperparametros"]["modelo__outer_bags"])


def _clase(variable):
    """Cómo se dibuja una variable: «categorica», «binaria», «valores» (numérica de pocos valores) o «continua»."""
    v = M.variables().loc[variable]
    if M.es_categorica(variable):
        return "categorica"
    if v["tipo"] == "binaria":
        return "binaria"
    return "valores" if int(v["n_distintos"]) <= POCOS_VALORES else "continua"


def _conteos(variable):
    """Clientes de entrenamiento por valor, categoría o veintil de una variable (datos/eda/por_variable.csv)."""
    t = D.tabla("eda/por_variable.csv")
    return t[t["variable"] == variable].sort_values("orden").reset_index(drop=True)


def _lineal(variable):
    """Lo que la regresión logística asigna a una variable. Numérica: (función del valor, coeficiente). Categórica:
    ({categoría: coeficiente}, None); la categoría de referencia no está en el diccionario."""
    ts = [t for t in M.logistica().terminos if t["variable"] == variable]
    if ts and ts[0]["clase"] == "numerica":
        t = ts[0]
        return (lambda x: t["coef"] * (np.asarray(x, dtype=float) - t["media"]) / t["escala"]), float(t["coef"])
    return {t["categoria"]: float(t["coef"]) for t in ts}, None


def _eje_efecto(fig, lo, hi, eje="y"):
    """Eje del efecto: marcas con signo en es-CO (pocas a cada lado del cero) y un rango que cubre los datos con aire."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    tope = max(abs(lo), abs(hi)) or 1.0
    paso = ui.paso_redondo(tope, MARCAS_POR_LADO)
    vals, textos = ui.marcas(math.ceil(lo / paso - 1e-9) * paso, math.floor(hi / paso + 1e-9) * paso, paso, signo=True)
    aire = AIRE * ((hi - lo) or tope)
    (fig.update_yaxes if eje == "y" else fig.update_xaxes)(tickvals=vals, ticktext=textos, range=[lo - aire, hi + aire])


def _escalones(desde, hasta, log=False, fino=0.0):
    """Puntos de una curva en escalones: (x de cada punto, índice de su tramo). Cada tramo se dibuja con sus dos bordes
    (salvo los más angostos que `fino`, como fracción del eje), para que la etiqueta flotante del punto más cercano al
    puntero sea la del tramo que tiene debajo. La línea va con forma «hv»."""
    desde, hasta = np.asarray(desde, dtype=float), np.asarray(hasta, dtype=float)
    ancho = np.log10(hasta / desde) if log else hasta - desde
    doble = ancho > fino * ancho.sum()
    doble[-1] = True                                    # el último tramo siempre se cierra
    ix = np.repeat(np.arange(len(desde)), np.where(doble, 2, 1))
    xs = desde[ix].copy()
    segundo = np.r_[False, ix[1:] == ix[:-1]]           # el segundo punto de un tramo va en su borde derecho
    xs[segundo] = hasta[ix[segundo]]
    return xs, ix


def _hover_largo(textos):
    """Como ui.hover_texto (una cadena ya formateada por punto), para trazas de más de mil puntos: el texto va en una
    lista llana, que Plotly valida mucho más rápido que el customdata anidado."""
    return {"text": textos, "hovertemplate": "%{text}<extra></extra>"}


def _alinear(fig):
    """El mismo margen izquierdo en la curva y en el gráfico de clientes, para que sus ejes x coincidan."""
    fig.update_yaxes(automargin=False)
    fig.update_xaxes(ticklabelstandoff=ui.px("--e-1"))      # en el teléfono, la primera marca del eje x no toca el «0» del eje y
    fig.update_layout(margin_l=ui.px("--e-7"))


def _eje_clientes(fig, eje="y"):
    """Eje del número de clientes de entrenamiento: enteros con punto de miles, desde cero."""
    ui.formato_eje(fig, eje, 0)
    (fig.update_yaxes if eje == "y" else fig.update_xaxes)(rangemode="tozero")
    ui.unidad_eje(fig, "clientes de entrenamiento")


# --- Sección 1: qué pesa ------------------------------------------------------------------------------------------------
def _permutacion():
    """Importancia por permutación de la EBM de clasificación (caída de AUC en la prueba al barajar cada variable)."""
    t = D.tabla("resultados/importancia_permutacion_ebm.csv")
    return t.set_index(t.columns[0])


def _importancia(tarea):
    """Figura, título, subtítulo, tabla con las 48, nota y etiqueta del gráfico de importancia."""
    c = TAREA[tarea]
    imp = M.importancia(tarea)
    perm = _permutacion() if tarea == "clasificacion" else None
    foco, medio, otras = imp.iloc[:N_FOCO], imp.iloc[N_FOCO:N_BARRAS], imp.iloc[N_BARRAS:]
    rotulo_otras = _partir(f"Las otras {len(otras)} variables")
    ancho = ui.ancho_barra(N_BARRAS + 1, ALTO_LIENZO["alto"])
    puesto = {v: i + 1 for i, v in enumerate(imp.index)}

    def texto(v):
        t = f"{M.nombre(v)}: <b>{F.pct(imp[v])}</b> de la importancia · puesto {puesto[v]} de {len(imp)}"
        if perm is not None:
            t += f"<br>Caída de AUC al barajarla: {ui.num(perm.at[v, 'media'], 4)}"
        return t

    fig = go.Figure()
    for serie, forma in ((foco, "barra"), (medio, "rayada")):
        fig.add_trace(go.Bar(y=[_partir(M.nombre(v)) for v in serie.index], x=serie.to_numpy(), orientation="h",
                             text=[F.pct(x) for x in serie], customdata=[[texto(v), v] for v in serie.index],
                             hovertemplate="%{customdata[0]}<extra></extra>", **ui.traza("modelo", forma, ancho=ancho)))
    fig.add_trace(go.Bar(y=[rotulo_otras], x=[otras.sum()], orientation="h", text=[F.pct(otras.sum())],
                         customdata=[[f"Las otras {len(otras)} variables juntas: <b>{F.pct(otras.sum())}</b> de la importancia", None]],
                         hovertemplate="%{customdata[0]}<extra></extra>", **ui.traza("contexto", "barra", ancho=ancho)))
    rotulos = [_partir(M.nombre(v)) for v in imp.index[:N_BARRAS]] + [rotulo_otras]
    fig.update_layout(barmode="overlay", hovermode="closest",
                      margin=dict(l=ui.px("--e-1"), r=ui.MARGEN_ETIQUETAS, t=ui.px("--e-1"), b=ui.px("--e-1")))
    fig.update_xaxes(visible=False, range=[0, float(imp.max())])
    fig.update_yaxes(autorange="reversed", showgrid=False, categoryorder="array", categoryarray=rotulos,
                     tickfont=ui.fuente_svg("tinta_2"), ticklabelstandoff=ui.px("--e-2"))

    titulo = f"Las {PALABRA[N_FOCO]} primeras: " + F.enumerar(f"{M.nombre(v).lower()} ({F.pct(imp[v])})" for v in foco.index)
    sub = f"Cuota de importancia de cada variable en {LA_EBM} de {c['nombre']}."
    if perm is not None:
        orden = perm["media"].sort_values(ascending=False)
        if set(orden.index[:N_FOCO]) == set(foco.index):
            sub += (f" La importancia por permutación en la prueba coincide: barajar una de las {PALABRA[N_FOCO]} baja el AUC entre "
                    f"{ui.num(orden.iloc[0], 3)} y {ui.num(orden.iloc[N_FOCO - 1], 4)}; la mayor caída entre las demás es "
                    f"{ui.num(orden.iloc[N_FOCO], 4)}.")
    elif list(M.importancia("clasificacion").index[:N_FOCO]) == list(foco.index):
        sub += f" Son las mismas {PALABRA[N_FOCO]} de la clasificación, en el mismo orden."

    filas = []
    for v in imp.index:
        fila = [M.nombre(v), F.entero(puesto[v]), F.pct(imp[v], 2)]
        if perm is not None:
            fila.append(f"{ui.num(perm.at[v, 'media'], 4)} ± {ui.num(perm.at[v, 'desv'], 4)}")
        filas.append(fila)
    encabezados = ["Variable", "Puesto", "Cuota de importancia"] + (["Caída de AUC al barajarla"] if perm is not None else [])
    detalle = ("La cuota de una variable categórica es la suma de la de sus indicadoras."
               + (" La caída de AUC es la media ± la desviación de las repeticiones, en los clientes de prueba; las negativas son ruido."
                  if perm is not None else ""))
    tabla = html.Div([ui.tabla(encabezados, filas, compacta=True, alta=True), ui.nota(detalle)], className="resultados")
    nota = ("La cuota es la media del valor absoluto de lo que aporta la variable en el entrenamiento, sobre el total; la de una "
            "variable categórica suma la de sus indicadoras. Un clic en una barra lleva a su curva, en la sección siguiente.")
    etiqueta = (f"{titulo}. " + "; ".join(f"{M.nombre(v)}: {F.pct(imp[v])}" for v in imp.index[N_FOCO:N_BARRAS])
                + f"; las otras {len(otras)} variables: {F.pct(otras.sum())}.")
    return fig, titulo, sub, tabla, nota, etiqueta


def _pesa(tarea):
    """Título de la sección de importancia: lo que pesan las dos primeras variables frente a todas las demás juntas."""
    imp = M.importancia(tarea)
    primeras = [M.nombre(v) for v in imp.index[:N_PRIMERAS]]
    if imp.iloc[:N_PRIMERAS].sum() > MITAD:
        segunda = primeras[-1][0].lower() + primeras[-1][1:]
        return f"{primeras[0]}{_y(segunda)}{segunda} pesan más que las otras {len(imp) - N_PRIMERAS} variables juntas"
    return f"Las otras {len(imp) - N_FOCO} variables se reparten el {F.pct(imp.iloc[N_FOCO:].sum(), 0)} de la importancia"


def _lectura(tarea, id=None):
    """Tarjeta de lectura: por qué basta un modelo aditivo."""
    foco = M.importancia(tarea).iloc[:N_FOCO]
    nombres = F.enumerar(M.nombre(v).lower() for v in foco.index).capitalize()
    return ui.tarjeta_insight(F.pct(foco.sum(), 0), f"de la importancia está en {PALABRA[N_FOCO]} variables",
                              f"{nombres}. La señal está en pocas variables, con efectos no lineales y casi independientes: "
                              "por eso basta sumar una curva por variable.", chip_="Por qué una EBM", id=id)


# --- Sección 2: la curva de una variable ----------------------------------------------------------------------------------
def _marcas_valores(xs):
    """Marcas en los valores que existen; dos valores pegados comparten marca («29 y 30»). None si son demasiados o si se
    encadenan más de dos (entonces las pone Plotly)."""
    if len(xs) > MAX_MARCAS:
        return None
    rango = (xs[-1] - xs[0]) or 1.0
    grupos = [[xs[0]]]
    for x in xs[1:]:
        if x - grupos[-1][-1] < SEPARACION * rango:
            grupos[-1].append(x)
        else:
            grupos.append([x])
    if any(len(g) > 2 for g in grupos):
        return None
    return [sum(g) / len(g) for g in grupos], [" y ".join(_corto(x) for x in g) for g in grupos]


def _marcas_log(lo, hi, millones=False):
    """Marcas de un eje logarítmico en es-CO: potencias de diez y, si el eje abarca pocas, también sus dobles y quíntuplos."""
    pasos = (1,) if math.log10(hi / lo) >= 3 else (1, 2, 5)
    vals = [m * float(BASE) ** k for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1) for m in pasos]
    vals = [x for x in vals if lo <= x <= hi]

    def texto(x):
        if millones:
            return f"{_corto(x / MILLON, 1)} M"
        return F.entero(x) if x >= 1 else ui.num(x, -math.floor(math.log10(x)))
    return vals, [texto(x) for x in vals]


def _extremos(c, d, alto, bajo, donde_alto, donde_bajo):
    """Frase con el efecto más alto y el más bajo de una variable y dónde ocurren."""
    return (f"Lo más alto que suma el modelo es {ui.con_signo(alto, d)}, {donde_alto}; lo más bajo, {ui.con_signo(bajo, d)}, "
            f"{donde_bajo}. Por encima de cero asigna {c['mas']}; por debajo, {c['menos']}.")


def _apoyo(v, desde, hasta):
    """Si un tramo queda en una cola del entrenamiento, la frase que dice cuántos clientes lo respaldan; si no, ''."""
    for bajo, alto, cola in COLAS:
        if desde >= float(v[alto]):
            return f" (por encima del percentil {alto[1:]} del entrenamiento: a lo sumo el {F.pct(cola, 0)} de los clientes)"
        if hasta <= float(v[bajo]):
            return f" (por debajo del percentil {bajo[1:]} del entrenamiento: a lo sumo el {F.pct(cola, 0)} de los clientes)"
    return ""


def _textos_curva(variable, tarea, lo, hi):
    """Título, subtítulo y decimales de la curva de una variable (lo que no depende de cómo se dibuja)."""
    c, v = TAREA[tarea], M.variables().loc[variable]
    imp = M.importancia(tarea)
    d = _decimales(max(abs(lo), abs(hi)))
    titulo = f"{v['nombre']}: de {ui.con_signo(lo, d)} a {ui.con_signo(hi, d)} {c['en']}"
    sub = (f"{v['descripcion']}. Cuota de importancia en {c['nombre']}: {F.pct(imp[variable], 2 if imp[variable] < 0.01 else 1)}, "
           f"puesto {list(imp.index).index(variable) + 1} de {len(imp)}.")
    return titulo, sub, d


def _nota_recta(coef):
    if coef == 0:
        return f" {LA_LOGISTICA.capitalize()} dejó en cero el coeficiente de esta variable: su recta es plana."
    return (f" La recta a trazos es lo que asigna {LA_LOGISTICA}: su coeficiente ({ui.con_signo(coef, max(3, _decimales(abs(coef))))}) "
            "por el valor estandarizado de la variable.")


def _error(desv):
    """Barra de error de una desviación entre bolsas (en tinta, no en el color de la serie)."""
    return {"type": "data", "array": desv, "color": ui.ROL["tinta_2"], "thickness": ui.ANCHO_LINEA}


def _discreta(variable, tarea, recta):
    """Variable binaria o categórica: barras del efecto de cada categoría y, debajo, los clientes de cada una."""
    c = TAREA[tarea]
    n_ent = int(M.meta()["n_entrenamiento"])
    conteo = _conteos(variable).set_index("grupo")["n"]
    funcion, coef = _lineal(variable)
    if _clase(variable) == "binaria":
        nombres, grupos = ["No", "Sí"], ["False", "True"]
        efecto = M.ebm(tarea).efecto_en(variable, [0, 1])
        f = M.forma(variable, tarea)
        desv = (f["sup"] - f["efecto"]).to_numpy()
        lineal = [float(y) for y in funcion([0, 1])] if recta else None
        referencia = None
    else:
        e = M.efecto_categorias(variable, tarea).sort_values(ascending=False)
        nombres, efecto, desv = list(e.index), e.to_numpy(), None
        grupos = nombres
        lineal = [funcion.get(g, 0.0) for g in nombres] if recta else None
        referencia = next(g for g in M.ebm(tarea).categoricas[variable] if g not in funcion)
    n = [int(conteo[g]) for g in grupos]
    horizontal = len(nombres) > MAX_VERTICALES
    rotulos = [g.split(",")[0] for g in nombres] if horizontal else [_partir(g, LARGO_ROTULO // 2) for g in nombres]
    alto = "alto" if horizontal else "normal"
    ancho = ui.ancho_barra(len(nombres), ALTO_LIENZO[alto] if horizontal else None)
    lo, hi = float(np.min(efecto)), float(np.max(efecto))
    titulo, sub, d = _textos_curva(variable, tarea, lo, hi)
    eje_v, eje_c = ("x", "y") if horizontal else ("y", "x")          # eje del valor y eje de las categorías

    def barras(valores, rol, textos, error=None):
        extra = {f"error_{eje_v}": _error(error)} if error is not None else {}
        return go.Bar(orientation="h" if horizontal else "v", **{eje_c: rotulos, eje_v: valores}, **extra,
                      **ui.traza(rol, "barra", ancho=ancho), **ui.hover_texto(textos))

    def ejes(fig):
        fig.update_layout(hovermode=f"{eje_c} unified", hoverdistance=SIN_LIMITE)
        if horizontal:
            fig.update_xaxes(showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
            fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=rotulos[::-1], tickfont=ui.fuente_svg("tinta_2"),
                             ticklabelstandoff=ui.px("--e-2"), dtick=1)
        else:
            fig.update_xaxes(categoryorder="array", categoryarray=rotulos, tickfont=ui.fuente_svg("tinta_2"))
            _alinear(fig)

    textos = [f"{EBM}: <b>{ui.con_signo(x, d)}</b>" + ("" if desv is None else f" (±{ui.num(desv[i], d)})") for i, x in enumerate(efecto)]
    fig = go.Figure()
    fig.add_trace(barras(efecto, "modelo", textos, desv))
    leyenda = None
    if lineal is not None:
        estilo = ui.traza("base", "punto") if referencia else ui.traza("base", "trazos", marcadores=True)
        fig.add_trace(go.Scatter(**{eje_c: rotulos, eje_v: lineal}, **estilo,
                                 **ui.hover_texto([f"{LOGISTICA}: <b>{ui.con_signo(x, d)}</b>" for x in lineal])))
        leyenda = [(f"{EBM}: lo que suma en cada categoría", "modelo", "barra"),
                   (LOGISTICA, "base", "punto" if referencia else "trazos")]
        lo, hi = min(lo, *lineal), max(hi, *lineal)
    if desv is not None:
        lo, hi = min(lo, float(np.min(efecto - desv))), max(hi, float(np.max(efecto + desv)))
    ui.linea_cero(fig, eje_v)
    ui.unidad_eje(fig, c["unidad"])
    _eje_efecto(fig, lo, hi, eje_v)
    ejes(fig)

    i_alto, i_bajo = int(np.argmax(efecto)), int(np.argmin(efecto))
    nota = _extremos(c, d, efecto[i_alto], efecto[i_bajo], f"en «{nombres[i_alto]}» ({_clientes(n[i_alto])} de entrenamiento)",
                     f"en «{nombres[i_bajo]}» ({_clientes(n[i_bajo])})")
    if desv is not None:
        nota += _banda(tarea, "La línea fina sobre cada barra")
    if lineal is not None and referencia:
        nota += (f" Los puntos son los coeficientes de {LA_LOGISTICA} para cada categoría; la de referencia («{referencia}») "
                 "vale cero.")
    elif lineal is not None:
        nota += _nota_recta(coef)
    if referencia:
        nota += " Las categorías se muestran como vienen en el conjunto de datos."
    encabezados = ["Categoría", f"Efecto en {LA_EBM}"] + (["Desviación entre bolsas"] if desv is not None else []) + ["Clientes de entrenamiento"]
    filas = []
    for i, g in enumerate(nombres):
        fila = [g, ui.con_signo(efecto[i], d)] + ([f"±{ui.num(desv[i], d)}"] if desv is not None else []) + [F.entero(n[i])]
        filas.append(fila + ([ui.con_signo(lineal[i], d)] if lineal is not None else []))
    if lineal is not None:
        encabezados.append(LOGISTICA)
    etiqueta = f"{titulo}. " + "; ".join(f"{g}: {ui.con_signo(x, d)}" for g, x in zip(nombres, efecto)) + "."
    curva = ui.grafico(fig, titulo, sub, etiqueta, (encabezados, filas), leyenda, nota, alto=alto, id=f"{PRE}-forma", fuentes=FUENTES_CURVA)

    # Dónde están los clientes de entrenamiento
    parte = [x / n_ent for x in n]
    fig2 = go.Figure()
    fig2.add_trace(barras(n, "contexto", [f"<b>{_clientes(x)}</b> de entrenamiento ({F.pct(p)})" for x, p in zip(n, parte)]))
    _eje_clientes(fig2, eje_v)
    ejes(fig2)
    titulo2 = _titulo_clientes([f"«{g}»" for g in nombres], n, femenino=True)
    sub2 = f"Cuántos de los {F.entero(n_ent)} clientes de entrenamiento hay en cada categoría, en el mismo orden de arriba."
    etiqueta2 = f"{titulo2}. " + "; ".join(f"{g}: {F.entero(x)}" for g, x in zip(nombres, n)) + "."
    tabla2 = (["Categoría", "Clientes de entrenamiento", "Parte"], [[g, F.entero(x), F.pct(p)] for g, x, p in zip(nombres, n, parte)])
    clientes = ui.grafico(fig2, titulo2, sub2, etiqueta2, tabla2, alto="alto" if horizontal else "bajo", id=f"{PRE}-dens",
                          fuentes=[D.fuente("eda/por_variable.csv")])
    return [curva, clientes]


def _valores(variable, tarea, recta):
    """Variable numérica de pocos valores: el efecto solo en los valores que existen (entre ellos la curva es extrapolación)."""
    c, v = TAREA[tarea], M.variables().loc[variable]
    n_ent = int(M.meta()["n_entrenamiento"])
    conteo = _conteos(variable)
    xs, n = [float(x) for x in conteo["x"]], [int(x) for x in conteo["n"]]
    efecto = M.ebm(tarea).efecto_en(variable, xs)
    f = M.forma(variable, tarea)
    tramo = np.clip(np.searchsorted(f["desde"].to_numpy(), xs, side="right") - 1, 0, len(f) - 1)
    desv = (f["sup"] - f["efecto"]).to_numpy()[tramo]
    unidad = f" {v['unidad']}" if isinstance(v["unidad"], str) else ""
    rango = xs[-1] - xs[0]
    x0, x1 = xs[0] - HOLGURA_EJE * rango, xs[-1] + HOLGURA_EJE * rango
    ancho = min(FRACCION_BARRA * float(np.min(np.diff(xs))), ui.ancho_barra(1) * (x1 - x0))
    lo, hi = float(np.min(efecto - desv)), float(np.max(efecto + desv))
    titulo, sub, d = _textos_curva(variable, tarea, float(np.min(efecto)), float(np.max(efecto)))
    funcion, coef = _lineal(variable)
    lineal = [float(y) for y in funcion(xs)] if recta else None
    marcas = _marcas_valores(xs)

    def ejes(fig):
        fig.update_xaxes(range=[x0, x1], hoverformat=",.2~f")
        fig.update_layout(hoverdistance=SIN_LIMITE)
        if marcas:
            fig.update_xaxes(tickvals=marcas[0], ticktext=marcas[1])
        _alinear(fig)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=efecto, error_y=_error(desv), **ui.traza("modelo", "barra", ancho=ancho),
                         **ui.hover_texto([f"{EBM}: <b>{ui.con_signo(e, d)}</b> (±{ui.num(s, d)})" for e, s in zip(efecto, desv)])))
    leyenda = None
    if lineal is not None:
        fig.add_trace(go.Scatter(x=xs, y=lineal, **ui.traza("base", "trazos"),
                                 **ui.hover_texto([f"{LOGISTICA}: <b>{ui.con_signo(y, d)}</b>" for y in lineal])))
        leyenda = [(f"{EBM}: lo que suma en cada valor", "modelo", "barra"), (f"{LOGISTICA}: una recta", "base", "trazos")]
        # El eje cubre la recta entera si con eso no se estira demasiado; si no, solo entre los percentiles 1 y 99 (la recta
        # se sale del lienzo en los valores extremos, que es lo que hace una recta)
        cabe = max(hi, *lineal) - min(lo, *lineal) <= MAX_ESTIRAR * (hi - lo)
        cubre = lineal if cabe else funcion([float(v[COLAS[0][0]]), float(v[COLAS[0][1]])])
        lo, hi = min(lo, *cubre), max(hi, *cubre)
    ui.linea_cero(fig)
    ui.unidad_eje(fig, c["unidad"])
    _eje_efecto(fig, lo, hi)
    ejes(fig)

    i_alto, i_bajo = int(np.argmax(efecto)), int(np.argmin(efecto))
    nota = _extremos(c, d, efecto[i_alto], efecto[i_bajo],
                     f"cuando vale {_corto(xs[i_alto])}{unidad} ({_clientes(n[i_alto])} de entrenamiento)",
                     f"cuando vale {_corto(xs[i_bajo])}{unidad} ({_clientes(n[i_bajo])})")
    nota += f" Solo se dibujan los {len(xs)} valores que existen en los datos: entre ellos no hay clientes."
    nota += _banda(tarea, "La línea fina sobre cada barra")
    if lineal is not None:
        nota += _nota_recta(coef)
    encabezados = ["Valor", f"Efecto en {LA_EBM}", "Desviación entre bolsas", "Clientes de entrenamiento"]
    filas = [[_corto(x), ui.con_signo(e, d), f"±{ui.num(s, d)}", F.entero(k)] + ([ui.con_signo(lineal[i], d)] if lineal is not None else [])
             for i, (x, e, s, k) in enumerate(zip(xs, efecto, desv, n))]
    if lineal is not None:
        encabezados.append(LOGISTICA)
    etiqueta = f"{titulo}. " + "; ".join(f"{_corto(x)}: {ui.con_signo(e, d)}" for x, e in zip(xs, efecto)) + "."
    curva = ui.grafico(fig, titulo, sub, etiqueta, (encabezados, filas), leyenda, nota, id=f"{PRE}-forma", fuentes=FUENTES_CURVA)

    # Dónde están los clientes de entrenamiento
    parte = [k / n_ent for k in n]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=xs, y=n, **ui.traza("contexto", "barra", ancho=ancho),
                          **ui.hover_texto([f"<b>{_clientes(k)}</b> de entrenamiento ({F.pct(p)})" for k, p in zip(n, parte)])))
    _eje_clientes(fig2)
    ejes(fig2)
    titulo2 = _titulo_clientes([_corto(x) for x in xs], n)
    sub2 = f"Cuántos de los {F.entero(n_ent)} clientes de entrenamiento tienen cada valor, en el mismo eje de la curva."
    etiqueta2 = f"{titulo2}. " + "; ".join(f"{_corto(x)}: {F.entero(k)}" for x, k in zip(xs, n)) + "."
    tabla2 = (["Valor", "Clientes de entrenamiento", "Parte"], [[_corto(x), F.entero(k), F.pct(p, 2)] for x, k, p in zip(xs, n, parte)])
    clientes = ui.grafico(fig2, titulo2, sub2, etiqueta2, tabla2, alto="bajo", id=f"{PRE}-dens", fuentes=[D.fuente("eda/por_variable.csv")])
    return [curva, clientes]


def _continua(variable, tarea, recta):
    """Variable continua: la función de forma en escalones con la desviación entre bolsas y, debajo, el histograma del
    entrenamiento. Las de cola muy larga van en eje logarítmico."""
    c, v = TAREA[tarea], M.variables().loc[variable]
    n_ent = int(M.meta()["n_entrenamiento"])
    f = M.forma(variable, tarea)
    x0, x1 = float(f["desde"].iloc[0]), float(f["hasta"].iloc[-1])
    log = x0 > 0 and x1 / x0 >= RAZON_LOG
    rango = x1 - x0
    dx = 0 if rango >= 1000 else 1 if rango >= 20 else 2 if rango >= 2 else 3
    # Decimales que distinguen los dos bordes de un tramo (una curva de mil tramos los tiene muy juntos)
    d_tramo = min(MAX_DECIMALES, max(dx, _decimales(float((f["hasta"] - f["desde"]).median())) - 1))
    millones = x1 >= MILLON

    entera = v["tipo"] == "entera"          # una variable entera no lleva ceros de relleno («176», «117,5»)

    def val(x):
        return _corto(x, dx) if entera and not (log or millones) else _valor(x, dx, significativas=log, millones=millones)

    def borde(x):
        return val(x) if (log or millones) else (_corto if entera else ui.num)(x, d_tramo)

    xs, ix = _escalones(f["desde"], f["hasta"], log, TRAMO_FINO)

    def escalon(columna):
        return f[columna].to_numpy()[ix]

    def ejes(fig):
        if log:
            vals, marcas = _marcas_log(x0, x1, millones)
            fig.update_xaxes(type="log", range=[math.log10(x0), math.log10(x1)], tickvals=vals, ticktext=marcas, hoverformat=f",.{CIFRAS}r")
        else:
            fig.update_xaxes(range=[x0, x1], hoverformat=f",.{dx}f")
        fig.update_layout(hoverdistance=SIN_LIMITE)
        _alinear(fig)

    lo, hi = float(f["efecto"].min()), float(f["efecto"].max())
    titulo, sub, d = _textos_curva(variable, tarea, lo, hi)
    desv = (f["sup"] - f["efecto"]).to_numpy()
    cortes = [borde(x) for x in f["desde"]] + [borde(x1)]          # cada borde se escribe una vez: cierra un tramo y abre el siguiente
    textos = np.array([f"De {cortes[i]} a {cortes[i + 1]}: <b>{ui.con_signo(e, d)}</b> (±{ui.num(s, d)})"
                       for i, (e, s) in enumerate(zip(f["efecto"], desv))])
    fig = go.Figure()
    ui.banda(fig, xs, escalon("inf").tolist(), escalon("sup").tolist(), "modelo")
    fig.add_trace(go.Scatter(x=xs, y=escalon("efecto"), **ui.traza("modelo", "linea"), **_hover_largo(textos[ix])))
    fig.update_traces(line_shape="hv")
    leyenda = [(f"{EBM}: lo que suma en cada tramo", "modelo", "linea"), (f"Desviación entre las {_bolsas(tarea)} bolsas", "modelo", "area")]
    y_lo, y_hi = float(f["inf"].min()), float(f["sup"].max())
    funcion, coef = _lineal(variable)
    if recta:
        linea = funcion(xs)
        fig.add_trace(go.Scatter(x=xs, y=linea, **ui.traza("base", "trazos"),
                                 **_hover_largo(np.array([f"{LOGISTICA}: <b>{ui.con_signo(y, d)}</b>" for y in linea]))))
        leyenda.append((f"{LOGISTICA}: una recta", "base", "trazos"))
        centro = funcion([float(v[COLAS[0][0]]), float(v[COLAS[0][1]])])          # la recta, entre los percentiles 1 y 99
        y_lo, y_hi = min(y_lo, *centro), max(y_hi, *centro)
    ui.linea_cero(fig)
    ui.unidad_eje(fig, c["unidad"])
    _eje_efecto(fig, y_lo, y_hi)
    ejes(fig)

    def donde(i):
        a, b = float(f["desde"].iloc[i]), float(f["hasta"].iloc[i])
        return f"entre {borde(a)} y {borde(b)}" + _apoyo(v, a, b)

    i_alto, i_bajo = int(f["efecto"].idxmax()), int(f["efecto"].idxmin())
    nota = _extremos(c, d, hi, lo, donde(i_alto), donde(i_bajo))
    nota += f" La curva tiene {F.entero(len(f))} tramos." + _banda(tarea, "La banda")
    if log:
        nota += " El eje va en escala logarítmica" + ("; M son millones." if millones else ".")
    if recta:
        nota += _nota_recta(coef) + (" En este eje la recta se ve curva." if log and coef else "")
    if len(f) <= MAX_FILAS:
        encabezados = ["Desde", "Hasta", f"Efecto en {LA_EBM}", "Desviación entre bolsas"]
        filas = [[borde(a), borde(b), ui.con_signo(e, d), f"±{ui.num(s, d)}"] for a, b, e, s in zip(f["desde"], f["hasta"], f["efecto"], desv)]
        resumen = None
    else:
        encabezados = ["Desde", "Hasta", "Efecto medio", "Mínimo", "Máximo"]
        grupos = np.array_split(np.arange(len(f)), GRUPOS_TABLA)
        filas = [[val(f["desde"].iloc[g[0]]), val(f["hasta"].iloc[g[-1]]), ui.con_signo(f["efecto"].iloc[g].mean(), d),
                  ui.con_signo(f["efecto"].iloc[g].min(), d), ui.con_signo(f["efecto"].iloc[g].max(), d)] for g in grupos]
        resumen = (f"La curva tiene {F.entero(len(f))} tramos: la tabla los resume en {len(grupos)} grupos de tramos consecutivos. "
                   f"Los valores exactos están en {D.fuente(f'modelos/ebm_{tarea}.json')}.")
    tabla = html.Div([ui.tabla(encabezados, filas, compacta=True, alta=True)] + ([ui.nota(resumen)] if resumen else []), className="resultados")
    etiqueta = f"{titulo}. El efecto más alto está {donde(i_alto)} y el más bajo, {donde(i_bajo)}."
    curva = ui.grafico(fig, titulo, sub, etiqueta, tabla, leyenda, nota, id=f"{PRE}-forma", fuentes=FUENTES_CURVA)

    # Dónde están los clientes de entrenamiento
    titulo2 = f"El {F.pct(CENTRAL, 0)} de los clientes de entrenamiento está entre {val(v[COLAS[1][0]])} y {val(v[COLAS[1][1]])}"
    fig2 = go.Figure()
    if log:
        # Eje logarítmico: los veintiles del entrenamiento (grupos de tamaño parecido), con la altura como densidad
        g = _conteos(variable)
        bordes = ([float(g["desde"].iloc[0])] + [math.sqrt(a * b) for a, b in zip(g["hasta"].iloc[:-1], g["desde"].iloc[1:])]
                  + [float(g["hasta"].iloc[-1])])
        parte = (g["n"] / g["n"].sum()).to_numpy()
        altura = parte / np.diff(np.log10(bordes))
        textos2 = [f"De {val(a)} a {val(b)}: <b>{_clientes(k)}</b> de entrenamiento ({F.pct(p)})"
                   for a, b, k, p in zip(g["desde"], g["hasta"], g["n"], parte)]
        xh, ih = _escalones(bordes[:-1], bordes[1:], log=True)
        fig2.add_trace(go.Scatter(x=xh, y=altura[ih], **ui.traza("contexto", "escalon_rayado"), **ui.hover_texto([textos2[i] for i in ih])))
        fig2.update_yaxes(showticklabels=False, rangemode="tozero")
        ui.unidad_eje(fig2, "densidad de clientes de entrenamiento")
        sub2 = (f"Los {F.entero(n_ent)} clientes de entrenamiento ordenados en {len(g)} grupos de tamaño parecido: cada escalón es un "
                "grupo y, cuanto más alto, más juntos están sus clientes en el eje de la curva.")
        tabla2 = (["Desde", "Hasta", "Clientes de entrenamiento", "Parte"],
                  [[val(a), val(b), F.entero(k), F.pct(p)] for a, b, k, p in zip(g["desde"], g["hasta"], g["n"], parte)])
        fuentes2 = [D.fuente("eda/por_variable.csv"), D.fuente("variables.csv")]
    else:
        # Histograma de los clientes de entrenamiento que guarda el modelo (M.densidad: sin pesos de clase)
        h = M.densidad(variable)
        parte = (h["peso"] / h["peso"].sum()).to_numpy()
        textos2 = [f"De {val(a)} a {val(b)}: <b>{_clientes(round(k))}</b> de entrenamiento ({F.pct(p)})"
                   for a, b, k, p in zip(h["desde"], h["hasta"], h["peso"], parte)]
        xh, ih = _escalones(h["desde"], h["hasta"])
        fig2.add_trace(go.Scatter(x=xh, y=h["peso"].to_numpy()[ih], **ui.traza("contexto", "escalon_rayado"),
                                  **ui.hover_texto([textos2[i] for i in ih])))
        _eje_clientes(fig2)
        sub2 = (f"Histograma de los {F.entero(n_ent)} clientes de entrenamiento en {len(h)} tramos de igual ancho, en el mismo eje de "
                "la curva.")
        tabla2 = (["Desde", "Hasta", "Clientes de entrenamiento", "Parte"],
                  [[val(a), val(b), F.entero(round(k)), F.pct(p, 2)] for a, b, k, p in zip(h["desde"], h["hasta"], h["peso"], parte)])
        fuentes2 = [D.fuente("modelos/ebm_regresion.json"), D.fuente("variables.csv")]
    ejes(fig2)
    nota2 = (f"Mínimo {val(v['min'])}, mediana {val(v['mediana'])} y máximo {val(v['max'])} en el entrenamiento. Donde hay pocos "
             "clientes, la curva de arriba se apoya en pocos datos.")
    clientes = ui.grafico(fig2, titulo2, sub2, f"{titulo2}. {nota2}", tabla2, nota=nota2, alto="bajo", id=f"{PRE}-dens", fuentes=fuentes2)
    return [curva, clientes]


def _detalle(variable, tarea, recta):
    """Los dos gráficos de una variable: su curva y los clientes de entrenamiento que la respaldan."""
    recta = bool(recta) and tarea == "clasificacion"
    clase = _clase(variable)
    if clase in ("categorica", "binaria"):
        return _discreta(variable, tarea, recta)
    return (_valores if clase == "valores" else _continua)(variable, tarea, recta)


def _opciones_variable():
    """Las 48 variables, de mayor a menor importancia en la clasificación. La búsqueda encuentra cada una por su nombre,
    con tildes o sin ellas, y por su clave en los datos."""
    return [{"label": M.nombre(v), "value": v, "search": f"{M.nombre(v)} {_sin_tildes(M.nombre(v))} {v}"}
            for v in M.importancia("clasificacion").index]


def _opciones_recta(tarea):
    if tarea == "clasificacion":
        return [{"label": f"Comparar con una recta ({LOGISTICA.lower()})", "value": "recta"}]
    return [{"label": "Comparar con una recta (solo en clasificación)", "value": "recta", "disabled": True}]


# --- Sección 3: el modelo por dentro --------------------------------------------------------------------------------------
def _dentro(tarea):
    """Fichas del modelo de una tarea, cómo se lee una predicción, cómo se aprende y la combinación elegida."""
    c = TAREA[tarea]
    meta, modelo = M.meta(), M.ebm(tarea)
    ficha = meta["ebm"][tarea]
    n_num = sum(t["clase"] == "numerica" for t in modelo.terminos)
    n_ind = len(modelo.terminos) - n_num
    costo = D.tabla("resultados/costo.csv")
    ce, cx, cb = (D.fila(costo, tarea=tarea, modelo=m) for m in ("ebm", RIVAL, BOSQUE))
    if tarea == "clasificacion":
        nota_b0 = (f"El punto de partida, antes de sumar ninguna variable: equivale a un puntaje de "
                   f"{ui.num(float(M.sigmoide(modelo.intercepto)), 3)}.")
    else:
        nota_b0 = (f"El pronóstico de partida, antes de sumar ninguna variable. La media del puntaje real en la prueba es "
                   f"{ui.num(M.clientes()['churn_probability'].mean(), 4)}.")
    fichas = ui.cifras([
        ui.cifra("Términos", F.entero(ficha["terminos"]), "tablas", mini=True,
                 nota=f"{F.entero(n_num)} de variables numéricas y binarias y {F.entero(n_ind)} indicadoras de las "
                      f"{len(modelo.categoricas)} variables categóricas. Ninguna interacción (K = 0)."),
        ui.cifra("Parámetros", F.entero(ficha["parametros"]), "números", mini=True,
                 nota=f"Un puntaje por tramo de cada tabla. {D.en_frase(BOSQUE)} guarda {F.entero(cb['parametros'])} nodos."),
        ui.cifra("Intercepto", ui.num(ficha["intercepto"], 4), c["en"], mini=True, nota=nota_b0),
        ui.cifra("Ajuste del modelo final", ui.num(ce["ajuste_s"], 1), "s", mini=True,
                 variacion=f"{ui.num(ce['ajuste_s'] / cx['ajuste_s'], 0)} veces lo que tarda {D.en_frase(RIVAL)}",
                 nota=f"Predice a los {F.entero(meta['n_prueba'])} clientes de prueba en {ui.num(ce['inferencia_s'], 3)} s, ocupa "
                      f"{ui.num(ce['tamano_mb'], 1)} MB y su ajuste necesita {ui.num(ce['memoria_mb'], 0)} MB de memoria."),
    ])

    n_var, n_tablas = len(M.predictores()), len(modelo.terminos)
    if tarea == "clasificacion":
        suma = ("El logit del riesgo alto es el intercepto más la suma de esos aportes, y la función logística lo convierte en el "
                "puntaje del modelo.")
    else:
        suma = "El pronóstico es el intercepto más la suma de esos aportes."
    ver = meta["verificacion"][f"ebm_{tarea}"]
    lectura = ui.tarjeta([
        ui.parrafo(f"Para un cliente, el modelo busca en cada una de las {n_tablas} tablas el tramo donde cae el valor de su variable y "
                   f"toma un puntaje. Agrupados por variable (las indicadoras de una categórica se suman), son {n_var} aportes. {suma}"),
        ui.parrafo("No hay interacciones (K = 0): cada variable aporta por separado, sin depender de las demás. Las curvas de la "
                   "sección anterior no son una explicación aproximada del modelo: son el modelo."),
        ui.aviso(f"Esta app evalúa esas mismas tablas con numpy, sin el modelo original. verificar_motor.py comparó los dos en "
                 f"{F.entero(ver['filas_comparadas'])} clientes: la mayor diferencia es "
                 f"{_cientifico(ver['max_dif_motor_frente_al_modelo_guardado'])}.", "bien", "La app reproduce el modelo guardado"),
        html.Div(ui.boton("Ver la suma de un cliente", href="/modelos/cliente", icono="equipo"), className="botonera"),
    ], titulo="Cómo se lee una predicción", fuentes=[D.fuente("metadatos.json"), D.fuente(f"modelos/ebm_{tarea}.json")])

    # Cómo se aprende: el boosting cíclico y las bolsas, con los hiperparámetros de la combinación elegida (las letras son las
    # de la ficha de al lado y las del informe: L hojas, m clientes por hoja, tasa η, B bolsas)
    exportado = meta["modelos_exportados"][f"ebm_{tarea}"]
    hp = {clave.split("__")[-1]: x for clave, x in ficha["hiperparametros"].items()}
    bolsas = F.entero(hp["outer_bags"])
    parrafos = [
        ui.parrafo(["Las curvas se aprenden por ", html.Em("boosting"),
                    f" cíclico: en cada ronda el modelo recorre las variables y, para cada una, ajusta un árbol pequeño, de "
                    f"L = {F.entero(hp['max_leaves'])} hojas y al menos m = {F.entero(hp['min_samples_leaf'])} clientes por hoja, al error "
                    f"que todavía queda. Lo suma a la curva de esa variable multiplicado por una tasa de aprendizaje pequeña "
                    f"(η = {ui.num(hp['learning_rate'], 3)})."]),
        ui.parrafo(f"El ajuste se repite en B = {bolsas} submuestras del entrenamiento, las bolsas externas, y las curvas se promedian. "
                   f"La desviación entre las {bolsas} bolsas es la banda de cada curva (la línea fina sobre cada barra)."),
    ]
    if exportado["balanceo"] == "class_weight":
        parrafos.append(ui.parrafo(f"Como la combinación elegida usa {D.NOMBRE_BALANCEO[exportado['balanceo']]} de clases, cada cliente "
                                   "entra con un peso según el tamaño de su clase. La librería no lo trae: lo añadió el proyecto."))
    aprendizaje = ui.tarjeta(parrafos, titulo="Cómo se aprende", fuentes=[D.fuente("metadatos.json")])

    corridas = D.tabla("entrega3/corridas.csv")
    n_corridas = int(((corridas["modelo"] == "ebm") & (corridas["tarea"] == tarea)).sum())
    pares = [("Balanceo", D.NOMBRE_BALANCEO[exportado["balanceo"]])] if tarea == "clasificacion" else []      # en regresión no hay clases
    pares.append(("Optimizador", D.NOMBRE_OPTIMIZADOR[exportado["optimizador"]]))
    for clave, x in hp.items():
        pares.append((HIPER.get(clave, clave), F.entero(x) if float(x) == int(x) else ui.num(x, 3)))
    combinacion = ui.tarjeta([
        ui.ficha(pares),
        ui.nota(f"La mejor de las {n_corridas} corridas de {LA_EBM} en {c['nombre']}, elegida en la validación cruzada anidada sin mirar "
                "la prueba."),
    ], titulo="La combinación elegida", fuentes=[D.fuente("metadatos.json"), D.fuente("entrega3/corridas.csv")])
    return [fichas, ui.rejilla([lectura, aprendizaje, combinacion], "3")]


# --- Pantalla -------------------------------------------------------------------------------------------------------------
def _subtitulo():
    ficha = M.meta()["ebm"]["clasificacion"]
    cuota = {t: M.importancia(t).iloc[:N_FOCO].sum() for t in TAREA}
    return ["Una ", html.Em("Explainable Boosting Machine"),
            f": el puntaje sale de sumar una curva por variable. {PALABRA[N_FOCO].capitalize()} variables reúnen el "
            f"{F.pct(cuota['clasificacion'], 0)} de la importancia en clasificación y el {F.pct(cuota['regresion'], 0)} en regresión, y "
            f"el modelo completo son {F.entero(ficha['terminos'])} tablas con {F.entero(ficha['parametros'])} números que se pueden leer."]


def layout(variable=None, **_):
    ficha = M.meta()["ebm"]["clasificacion"]
    tarea = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", [{"label": c["rotulo"], "value": t} for t, c in TAREA.items()], "clasificacion"))
    selector = ui.control("Variable:", ui.selector(f"{PRE}-variable", _opciones_variable(), _variable(variable), buscar=True))
    recta = ui.casilla(f"{PRE}-recta", _opciones_recta("clasificacion"), [])
    return ui.pantalla([
        ui.encabezado(PREGUNTA, _subtitulo(), [tarea], pantalla="modelo"),
        ui.seccion(html.Span(_pesa("clasificacion"), id=f"{PRE}-pesa"), [
            ui.rejilla([
                ui.grafico(f"{PRE}-imp", alto="alto", fuentes=FUENTES_IMPORTANCIA,
                           leyenda=[(f"Las {PALABRA[N_FOCO]} primeras", "modelo", "barra"), ("Las siguientes", "modelo", "rayada"),
                                    ("El resto, juntas", "contexto", "barra")]),
                _lectura("clasificacion", id=f"{PRE}-lectura"),
            ], "2-1"),
        ], subtitulo="Qué pesa en la decisión: la parte de la importancia del modelo que corresponde a cada variable.",
            fuentes=FUENTES_IMPORTANCIA),
        ui.seccion("Cada variable aporta una curva, y la curva se puede leer", [
            html.Div(id=f"{PRE}-detalle", className="resultados"),
        ], subtitulo="Lo que el modelo suma según el valor de la variable, leído del modelo en vivo, y los clientes de entrenamiento "
                     "que respaldan cada tramo. Describe lo que el modelo asigna, no el efecto de cambiar la variable. El logit es la "
                     f"escala en que suma el modelo de clasificación: 0 equivale a un puntaje de {ui.num(float(M.sigmoide(0)), 1)} y, "
                     "cuanto más alto, más riesgo.",
            controles=[selector, recta]),
        ui.seccion(f"El modelo completo son {F.entero(ficha['terminos'])} tablas con {F.entero(ficha['parametros'])} números", [
            html.Div(id=f"{PRE}-dentro", className="resultados"),
        ], subtitulo="Una tabla por término: los cortes de la variable y un puntaje por tramo.", fuentes=FUENTES_DENTRO),
    ])


@callback(Output(f"{PRE}-imp", "figure"), Output(f"{PRE}-imp-titulo", "children"), Output(f"{PRE}-imp-sub", "children"),
          Output(f"{PRE}-imp-tabla", "children"), Output(f"{PRE}-imp-nota", "children"), Output(f"{PRE}-imp-lienzo", "aria-label"),
          Output(f"{PRE}-lectura", "children"), Output(f"{PRE}-dentro", "children"), Output(f"{PRE}-recta", "options"),
          Output(f"{PRE}-pesa", "children"), Input(f"{PRE}-tarea", "value"))
def _por_tarea(tarea):
    tarea = _tarea(tarea)
    fig, titulo, sub, tabla, nota, etiqueta = _importancia(tarea)
    return fig, titulo, sub, tabla, nota, etiqueta, _lectura(tarea).children, _dentro(tarea), _opciones_recta(tarea), _pesa(tarea)


@callback(Output(f"{PRE}-variable", "value"), Output(f"{PRE}-imp", "clickData"), Input(f"{PRE}-imp", "clickData"))
def _elegir(clic):
    """Un clic en una barra de importancia elige esa variable. El clic se borra para que repetirlo vuelva a contar."""
    try:
        variable = clic["points"][0]["customdata"][1]
    except (TypeError, KeyError, IndexError):
        return no_update, no_update
    return (variable if isinstance(variable, str) and variable in M.variables().index else no_update), None


# Tras ese clic, la curva (que está más abajo) se lleva a la vista: sin esto parece que el clic no hizo nada. Corre en el
# navegador; la salida es un atributo que nadie lee.
clientside_callback(
    """function (clic) {
        var barra = clic && clic.points && clic.points[0];
        if (barra && barra.customdata && barra.customdata[1]) {
            var suave = !(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
            setTimeout(function () {
                var curva = document.getElementById('%s');
                if (curva) { curva.scrollIntoView({behavior: suave ? 'smooth' : 'auto', block: 'start'}); }
            }, %d);
        }
        return window.dash_clientside.no_update;
    }""" % (f"{PRE}-forma-figura", ESPERA_SALTO_MS),
    Output(f"{PRE}-detalle", "data-salto"), Input(f"{PRE}-imp", "clickData"))


@callback(Output(f"{PRE}-detalle", "children"), Input(f"{PRE}-variable", "value"), Input(f"{PRE}-tarea", "value"),
          Input(f"{PRE}-recta", "value"))
def _por_variable(variable, tarea, recta):
    return _detalle(_variable(variable), _tarea(tarea), recta)
