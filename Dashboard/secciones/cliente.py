"""Un cliente · ¿Por qué la EBM marca a este cliente?

La pantalla de la demostración en vivo con la EBM, el benchmark interpretable (la predicción en vivo de los diez
modelos, con el original, está en secciones/prediccion.py). Se elige un cliente de la prueba (los tres casos del informe, uno al azar o una
fila; también con /cliente?fila=16840) y la EBM lo puntúa y lo explica en el momento: la cascada va del intercepto al logit
del cliente. El panel «¿Y si…?» cambia las variables que tiene sentido mover; M.simular recalcula las variables atadas,
M.puntuar da los puntajes y M.explicar, los aportes. Debajo, lo que pasó con el cliente real (con la variable que el modelo
no ve) y lo que le asignan los otros modelos. Toda cifra sale de datos/ o del motor; un cliente simulado es hipotético.

Callbacks: _elegir fija el cliente y _reponer pone sus valores en los controles; _calcular, ligero, responde a cada
movimiento (también mientras se arrastra un deslizador: escucha su drag_value) y pone las fichas y la cascada; _detallar, al
soltar, rehace la tabla de los 48 aportes y la tarjeta de los otros modelos; _cliente_real cambia solo con el cliente. Uno
más, en el navegador, hace que la dirección siga al cliente (?fila=…) y que el desplegable «Fila» se abra listo para escribir.
"""
import json
import math
from functools import lru_cache

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

import datos_app as D
import formato as F
import motor as M
import ui


PRE = "cl"
PANTALLA = "cliente"
PREGUNTA = "¿Por qué la EBM marca a este cliente?"      # la de app.VISTAS

# Los tres casos del informe (claves de M.meta()["ejemplos"]) con su nombre corto; la pantalla nace con el primero
CASO_OMISION = "riesgo bajo con mayor puntaje"
CASOS = {"riesgo bajo con mayor puntaje": "Falsa alarma", "riesgo alto con menor puntaje": "Se escapa",
         "mayor error de regresión": "Mayor error"}
RESULTADOS = dict(zip(("VP", "FP", "FN", "VN"), ui.MATRIZ_ROTULOS["celdas"]))    # acierto, falsa alarma, se escapa, bien descartado
TODOS = "todos"                # valor de «sin filtro» en los dos filtros del cliente al azar
MODELOS = ("ebm", "xgboost", "logistica")        # los modelos con puntaje por cliente en clientes_prueba.parquet
EN_VIVO = {"ebm": "p", "logistica": "logistica"}   # los que el motor vuelve a puntuar para un cliente simulado
GUARDADO = {"p": "p_ebm", "cal": "p_ebm_cal", "pron": "pron_ebm", "logistica": "p_logistica"}   # lo mismo, guardado para el cliente real

# Constantes de definición de la pantalla
N_SENAL = 4                    # las variables con señal del informe: las que tienen paso propio en la cascada
N_CALCULADAS = 3               # las variables atadas que el panel muestra siempre (las del puntaje de satisfacción)
PASO_BASE = 0.05               # paso del deslizador de la satisfacción de base
DEC_BASE = 2                   # decimales de ese paso
MARCAS_LOGINS = 5              # marcas del deslizador de inicios de sesión
DEC_PUNTAJE = 4                # decimales de un puntaje (los del informe)
DEC_LOGIT = 2                  # decimales de un logit en la cascada
DEC_TABLA = 4                  # decimales de un aporte en la tabla gemela
DEC_EJE = 2                    # decimales del puntaje que acompaña a cada marca del eje
DEC_VALOR = 2                  # decimales del valor de una variable continua
VALOR_GRANDE = 100             # desde aquí una variable continua se escribe sin decimales
PASO_EJE = 2                   # una marca del eje cada dos unidades de logit
AIRE_EJE = 1.05                # el eje llega un poco más allá del mayor logit de la prueba
GROSOR_BARRA = 0.3             # grosor de una barra de la cascada, en filas
AIRE_ARRIBA = 0.7              # espacio para el rótulo de la primera fila, en filas
AIRE_ABAJO = 0.12              # espacio bajo la barra de la última fila, en filas
ROTULO_MAX = 46                # caracteres de un rótulo de la cascada antes de recortarlo
SIMULADOS_EN_MEMORIA = 512     # clientes simulados que se recuerdan (al soltar un deslizador se pide otra vez el mismo)
TOLERANCIA_ESCALON = 1e-9      # dos probabilidades recalibradas son del mismo escalón si difieren menos que esto

PARTIDA = "Punto de partida (intercepto)"
SUMA = "Suma: logit del cliente"
ESCALA_CLASIFICADOR = "Escala del clasificador"                   # los dos bloques de la tarjeta de cifras
ESCALA_DATOS = "Escala del puntaje del conjunto de datos"

FUENTES_MODELO = [D.fuente("clientes_prueba.parquet"), D.fuente("modelos/ebm_clasificacion.json"), D.fuente("variables.csv"),
                  D.fuente("metadatos.json")]
FUENTES_FICHAS = FUENTES_MODELO + [D.fuente("modelos/ebm_regresion.json"), D.fuente("calibracion_isotonica_ebm.json")]
FUENTES_ABAJO = [D.fuente("clientes_prueba.parquet"), D.fuente("eda/por_variable.csv"), D.fuente("modelos/logistica_clasificacion.json"),
                 D.fuente("entrega3/complementos/calibracion.csv"), D.fuente("metadatos.json")]


def _i(nombre):
    """id de un componente de la pantalla."""
    return f"{PRE}-{nombre}"


# --- Lo que no cambia entre clientes (con caché; solo lectura) ----------------------------------------------------------
@lru_cache(maxsize=None)
def _valores(variable):
    """Los valores distintos de una variable discreta (columna «valores» de variables.csv), como enteros."""
    return tuple(int(float(x)) for x in str(M.variables().at[variable, "valores"]).split("|"))


@lru_cache(maxsize=1)
def _rango_base():
    """Extremos del deslizador de la satisfacción de base: múltiplos del paso entre el p1 y el p99 del entrenamiento."""
    v = M.variables().loc["base_satisfaction"]
    return (round(math.ceil(float(v["p1"]) / PASO_BASE) * PASO_BASE, DEC_BASE),
            round(math.floor(float(v["p99"]) / PASO_BASE) * PASO_BASE, DEC_BASE))


@lru_cache(maxsize=None)
def _ordenado(columna):
    return np.sort(M.clientes()[columna].to_numpy(dtype=float))


def _puesto(modelo, p):
    """Lugar de un puntaje entre los clientes de prueba según ese modelo (1 = el de mayor puntaje), como M.puesto."""
    if modelo == "ebm":
        return M.puesto(p)
    s = _ordenado(f"p_{modelo}")
    return int(len(s) - np.searchsorted(s, p, side="right") + 1)


@lru_cache(maxsize=None)
def _corte(modelo):
    """Puntaje del último cliente que ese modelo contacta con la capacidad del informe."""
    if modelo == "ebm":
        return float(M.meta()["punto_operacion"]["puntaje_minimo_contactado_ebm"])
    C = M.clientes()
    return float(C.loc[C[f"contactar_{modelo}"] == 1, f"p_{modelo}"].min())


@lru_cache(maxsize=1)
def _tope_logit():
    """El mayor logit (en valor absoluto) que la EBM le da a un cliente de prueba: fija el eje de la cascada."""
    p = _ordenado("p_ebm")
    return float(np.abs(np.log(p / (1 - p))).max())


@lru_cache(maxsize=1)
def _filas():
    """Las filas de los clientes de prueba, en orden."""
    return tuple(sorted(int(f) for f in M.clientes().index))


@lru_cache(maxsize=1)
def _cohortes():
    return tuple(sorted(M.clientes()["cohorte"].unique()))


@lru_cache(maxsize=1)
def _n_recalibradas():
    """Cuántos valores distintos toma la probabilidad recalibrada en la prueba."""
    return int(M.clientes()["p_ebm_cal"].nunique())


@lru_cache(maxsize=1)
def _recalibradas():
    """La probabilidad recalibrada y la etiqueta de riesgo alto de los clientes de prueba, como arreglos (no modificar)."""
    C = M.clientes()
    return C["p_ebm_cal"].to_numpy(dtype=float), C["riesgo_alto"].to_numpy(dtype=int)


@lru_cache(maxsize=1)
def _info():
    """{variable: (tipo, unidad)} de variables.csv, para escribir un valor sin consultar la tabla cada vez."""
    v = M.variables()
    return {k: (t, u if isinstance(u, str) else "") for k, t, u in zip(v.index, v["tipo"], v["unidad"])}


@lru_cache(maxsize=1)
def _calculadas():
    """Las variables que el motor recalcula tras un cambio, de mayor a menor importancia en el modelo."""
    v = M.variables()
    return tuple(v[v["papel"] == "calculada"].sort_values("puesto_clf").index)


@lru_cache(maxsize=1)
def _con_paso():
    """Las variables con paso propio en la cascada (las de mayor importancia en el modelo, en ese orden) y las demás. Son
    las mismas para todos los clientes: al cambiar de cliente o al simular, las barras no saltan de sitio."""
    propias = tuple(M.importancia("clasificacion").index[:N_SENAL])
    return propias, tuple(v for v in M.predictores() if v not in propias)


def _omision():
    return int(M.meta()["ejemplos"][CASO_OMISION])


def _casos():
    """{fila: clave del caso del informe}, con el caso de omisión primero."""
    ejemplos = M.meta()["ejemplos"]
    claves = [CASO_OMISION] + [k for k in ejemplos if k != CASO_OMISION]
    return {int(ejemplos[k]): k for k in claves if k in ejemplos}


# --- El cliente elegido ---------------------------------------------------------------------------------------------------
def _fila_valida(x):
    """La fila pedida, como entero, si es un cliente de prueba; si no, None."""
    try:
        n = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n) or n != int(n):
        return None
    return int(n) if int(n) in M.clientes().index else None


def _fila_o_la_de_omision(x):
    fila = _fila_valida(x)
    return _omision() if fila is None else fila


def _aviso(titulo, texto):
    return html.Div(ui.aviso(texto, "atencion", titulo, rol="status"), className="aviso-global")


def _resolver(pedida):
    """(fila, aviso): la fila pedida si es un cliente de prueba; si no, la de omisión y el aviso que lo dice."""
    if pedida is None or str(pedida).strip() == "":
        return _omision(), None
    fila = _fila_valida(pedida)
    if fila is not None:
        return fila, None
    return _omision(), _aviso(
        "Esa fila no es un cliente de prueba",
        f"«{str(pedida).strip()[:ROTULO_MAX]}» no está entre los {F.entero(len(M.clientes()))} clientes de prueba, los únicos que el "
        f"modelo no vio al ajustarse. Se muestra el caso con que abre la pantalla: {CASO_OMISION} (fila {_omision()}).")


def _al_azar(resultado, cohorte, actual):
    """(fila, aviso): un cliente de prueba al azar que cumpla los dos filtros y no sea el que ya se ve."""
    C = M.clientes()
    cumple = np.ones(len(C), dtype=bool)
    if resultado in RESULTADOS:
        cumple &= (C["resultado_ebm"] == resultado).to_numpy()
    if cohorte in _cohortes():
        cumple &= (C["cohorte"] == cohorte).to_numpy()
    candidatos = C.index[cumple]
    if len(candidatos) > 1:
        candidatos = candidatos[candidatos != actual]
    if not len(candidatos):
        return _fila_o_la_de_omision(actual), _aviso("No hay clientes con esos filtros",
                                                     "Ningún cliente de prueba tiene ese resultado en esa cohorte de edad.")
    return int(np.random.default_rng().choice(candidatos)), None


# --- El simulador ---------------------------------------------------------------------------------------------------------
def _controles(fila):
    """Los valores de los controles del panel que corresponden al cliente real de esa fila."""
    c = M.clientes().loc[fila]
    logins, v = _valores("app_logins_frequency"), int(c["app_logins_frequency"])
    lo, hi = _rango_base()
    base = round(round(float(c["base_satisfaction"]) / PASO_BASE) * PASO_BASE, DEC_BASE)
    return {"edad": int(c["age"]), "logins": min(range(len(logins)), key=lambda i: abs(logins[i] - v)),
            "base": min(max(base, lo), hi), "productos": [p for p in M.PRODUCTOS if bool(c[p])]}


def _cambios(fila, edad, logins, base, productos):
    """((variable, valor nuevo), …) con lo que el panel tiene distinto del cliente real. Un control que nadie movió no
    cuenta: así el cliente real se puntúa con sus valores exactos (la satisfacción de base no cae en el paso del deslizador)."""
    c, propios, cambios = M.clientes().loc[fila], _controles(fila), {}
    try:
        if edad is not None and int(edad) != propios["edad"] and int(edad) in _valores("age"):
            cambios["age"] = int(edad)
        if logins is not None:
            i, valores = int(round(float(logins))), _valores("app_logins_frequency")
            if i != propios["logins"] and 0 <= i < len(valores):
                cambios["app_logins_frequency"] = valores[i]
        if base is not None and math.isfinite(float(base)) and abs(float(base) - propios["base"]) > PASO_BASE / 2:
            cambios["base_satisfaction"] = round(float(base), DEC_BASE)
    except (TypeError, ValueError):
        pass
    if isinstance(productos, (list, tuple)):
        for p in M.PRODUCTOS:
            if (p in productos) != bool(c[p]):
                cambios[p] = p in productos
    return tuple(cambios.items())


def _evaluar(X):
    """El motor sobre un cliente (tabla de una fila con los 48 predictores): puntajes, aportes y logit."""
    p = M.puntuar(X).iloc[0]
    intercepto, aportes = M.explicar(X, "clasificacion")
    a = aportes.iloc[0]
    return {"p": float(p["p_ebm"]), "cal": float(p["p_ebm_cal"]), "pron": float(p["pron_ebm"]), "logistica": float(p["p_logistica"]),
            "intercepto": float(intercepto), "aportes": a.to_dict(), "logit": float(intercepto + a.sum()), "x": X.iloc[0].to_dict()}


@lru_cache(maxsize=256)
def _real(fila):
    """El cliente real, con sus valores tal como están en los datos (sin recalcular nada): los aportes los da el motor y
    los puntajes son los guardados, que difieren de los del motor en el último bit y son los que ordenan a los clientes
    de prueba (con el del motor el puesto podía correrse un lugar). No modificar el resultado."""
    c = M.clientes().loc[fila]
    real = _evaluar(M.simular(c, None, atar=False))
    real.update({clave: float(c[columna]) for clave, columna in GUARDADO.items()})
    real["puntaje_real"] = float(c["churn_probability"])
    return real


@lru_cache(maxsize=SIMULADOS_EN_MEMORIA)
def _simulado(fila, cambios):
    """El cliente con los cambios del panel y sus variables atadas recalculadas. No modificar el resultado."""
    return _evaluar(M.simular(M.clientes().loc[fila], dict(cambios)))


def _estado(fila, cambios):
    """(real, simulado, hay cambios): sin cambios, el simulado es el real."""
    real = _real(fila)
    return (real, _simulado(fila, cambios), True) if cambios else (real, real, False)


def _valor(variable, v):
    """El valor de una variable como se lee: sí o no, la categoría, el entero o el decimal, con su unidad."""
    tipo, unidad = _info()[variable]
    if tipo == "binaria":
        return "sí" if bool(v) else "no"
    if tipo == "categórica":
        return str(v)
    texto = F.entero(v) if tipo == "entera" else ui.num(v, DEC_VALOR if abs(float(v)) < VALOR_GRANDE else 0)
    return f"{texto} {unidad}" if unidad else texto


def _con_real(simulado, real, hay):
    """Texto de un valor del cliente simulado; si difiere del real, lo dice al lado."""
    return f"{simulado} (real: {real})" if hay and simulado != real else simulado


def _texto_cambios(real, sim, cambios):
    return "; ".join(f"{M.nombre(v)}: {_valor(v, real['x'][v])} → {_valor(v, sim['x'][v])}" for v, _nuevo in cambios)


# --- Resultados -----------------------------------------------------------------------------------------------------------
def _escalon(recalibrada):
    """(clientes de prueba, cuántos de riesgo alto) con esa misma probabilidad recalibrada: su escalón de la isotónica.
    Un cliente simulado puede caer entre dos escalones, y entonces no la comparte ninguno."""
    valores, altos = _recalibradas()
    mismos = np.abs(valores - recalibrada) < TOLERANCIA_ESCALON
    return int(mismos.sum()), int(altos[mismos].sum())


def _texto_escalon(recalibrada):
    n, altos = _escalon(recalibrada)
    if not n:
        return "Entre dos escalones de la isotónica: ningún cliente de prueba tiene esa probabilidad."
    return (f"Escalón de la isotónica: {F.entero(n)} {'cliente' if n == 1 else 'clientes'} de prueba, {F.entero(altos)} de riesgo alto.")


def _rotulo_bloque(texto):
    """Rótulo de un bloque de cifras dentro de una tarjeta: un h3 con el aspecto de «.mas__titulo» (pequeño y en seminegrita)."""
    return html.H3(texto, className="mas__titulo")


def _fichas(real, sim, hay):
    """Las cifras del cliente en dos bloques, uno por escala. En la del clasificador: puntaje, puesto, corte de la campaña y
    probabilidad recalibrada con su escalón. En la del puntaje del conjunto de datos (el rótulo va dentro): el pronóstico de
    la EBM de regresión, el umbral de riesgo alto y, para el cliente real, su puntaje real."""
    meta, n = M.meta(), len(M.clientes())
    corte, umbral = _corte("ebm"), meta["etiqueta"]["umbral"]
    p, p0 = sim["p"], real["p"]
    entra = p >= corte
    if hay:
        campana = ui.chip("Entraría en la campaña", "acento") if entra else ui.chip("Quedaría fuera")
    else:
        campana = ui.chip("Entra en la campaña", "acento") if entra else ui.chip("Queda fuera")
    puntaje = ui.cifra("Puntaje del modelo", ui.num(p, DEC_PUNTAJE), "sin recalibrar",
                       variacion=f"{ui.con_signo(p - p0, DEC_PUNTAJE)} (real: {ui.num(p0, DEC_PUNTAJE)})" if hay else None,
                       flecha=(p - p0) if hay else None, chip=ui.chip("Simulado", "acento") if hay else ui.chip("Real"),
                       mini=True, clase="cifra--llana", id=_i("puntaje"))
    puesto, puesto0 = F.entero(M.puesto(p)), F.entero(M.puesto(p0))
    lugar = ui.cifra("Puesto", puesto, _con_real(f"de {F.entero(n)}", puesto0, hay and puesto != puesto0),
                     chip=campana, mini=True, clase="cifra--llana", id=_i("puesto"))

    def con_el_real(sim_txt, real_txt):          # el valor del simulado y, debajo y en gris, el del cliente real
        return [sim_txt, html.Br(), html.Span(f"real: {real_txt}", className="tinta-3")] if hay and sim_txt != real_txt else sim_txt
    clasificador = [("Corte de la campaña", ui.num(corte, DEC_PUNTAJE)),
                    ("Probabilidad recalibrada", con_el_real(F.pct(sim["cal"]), F.pct(real["cal"])))]
    datos = [("Pronóstico de la EBM de regresión", con_el_real(ui.num(sim["pron"], DEC_PUNTAJE), ui.num(real["pron"], DEC_PUNTAJE))),
             ("Umbral de riesgo alto", ui.num(umbral, DEC_PUNTAJE))]
    if not hay:
        datos.append(("Puntaje real del cliente", ui.num(real["puntaje_real"], DEC_PUNTAJE)))
    return [puntaje, lugar, ui.ficha(clasificador), ui.nota(_texto_escalon(sim["cal"])),
            _rotulo_bloque(ESCALA_DATOS), ui.ficha(datos)]


@lru_cache(maxsize=1)
def _molde():
    """Lo que no cambia de la figura de la cascada (plantilla, línea de cero, corte de la campaña, unidad y ejes), ya como
    diccionario: armarlo cuesta más que el resto de la figura y se hace una sola vez. Nadie lo modifica; cada figura copia
    por encima lo que cambia."""
    corte = _corte("ebm")
    logit_corte = math.log(corte / (1 - corte))
    fig = go.Figure()
    ui.linea_cero(fig, "x")
    ui.referencia(fig, x=logit_corte)
    fig.add_annotation(x=logit_corte, y=1, yref="paper", text="Corte de la campaña", xanchor="left", yanchor="bottom",
                       yshift=ui.px("--e-1"), showarrow=False, font=ui.fuente_svg("tinta_2"))
    ui.unidad_eje(fig, "Logit")
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_yaxes(visible=False)
    return fig.to_plotly_json()["layout"]


def _cascada(fila, real, sim, cambios):
    """Figura, título, subtítulo y etiqueta de la cascada: del intercepto al logit del cliente, con un paso por cada una
    de las variables de _con_paso y otro con la suma de las demás. La figura es un diccionario: dos go.Bar con base sobre
    el molde."""
    hay = bool(cambios)
    propias, otras = _con_paso()
    a, x, x0 = sim["aportes"], sim["x"], real["x"]
    pasos = [a[v] for v in propias] + [sum(a[v] for v in otras)]
    inicio, final = sim["intercepto"], sim["logit"]
    llegadas = [inicio + sum(pasos[:i + 1]) for i in range(len(pasos))]
    salidas = [inicio] + llegadas[:-1]
    rotulos = ([PARTIDA] + [f"{M.nombre(v)} = {_con_real(_valor(v, x[v]), _valor(v, x0[v]), hay)}" for v in propias]
               + [f"Las otras {len(otras)} variables", SUMA])
    valores = [ui.num(inicio, DEC_LOGIT)] + [ui.con_signo(p, DEC_LOGIT) for p in pasos] + [ui.num(final, DEC_LOGIT)]
    n = len(rotulos)

    anotaciones = []                               # cada fila: rótulo y valor en una línea y, pegada debajo, su barra
    for i, (rotulo, valor) in enumerate(zip(rotulos, valores)):
        corto = rotulo if len(rotulo) <= ROTULO_MAX else rotulo[:ROTULO_MAX - 1].rstrip() + "…"
        comun = {"y": i, "yref": "y", "xref": "paper", "yanchor": "bottom", "showarrow": False, "bgcolor": ui.ROL["superficie"]}
        anotaciones.append(dict(comun, x=0, xanchor="left", text=corto, font=ui.fuente_svg("tinta" if i in (0, n - 1) else "tinta_2", "--t-s")))
        anotaciones.append(dict(comun, x=1, xanchor="right", text=valor, font=ui.fuente_svg("tinta", "--t-s")))
    # los aportes, rayados, parten de donde llegó el anterior; los dos extremos, sólidos, parten del cero
    aportes = go.Bar(
        y=list(range(1, n - 1)), x=pasos, base=salidas, orientation="h", name="Aporte de cada variable", offset=0,
        **ui.traza("modelo", "rayada", ancho=GROSOR_BARRA),
        **ui.hover_texto([f"{r}: <b>{v}</b> al logit · va en {ui.num(ll, DEC_LOGIT)} (puntaje {ui.num(M.sigmoide(ll), DEC_PUNTAJE)})"
                          for r, v, ll in zip(rotulos[1:-1], valores[1:-1], llegadas)]))
    extremos = go.Bar(
        y=[0, n - 1], x=[inicio, final], base=[0, 0], orientation="h", name="Punto de partida y suma", offset=0,
        **ui.traza("modelo", "barra", ancho=GROSOR_BARRA),
        **ui.hover_texto([f"{PARTIDA}: <b>{valores[0]}</b> (puntaje {ui.num(M.sigmoide(inicio), DEC_PUNTAJE)})",
                          f"{SUMA}: <b>{valores[-1]}</b> (puntaje {ui.num(sim['p'], DEC_PUNTAJE)})"]))
    tope = max([_tope_logit()] + [abs(v) for v in [inicio, final] + llegadas]) * AIRE_EJE
    marcas = [k * PASO_EJE for k in range(-int(tope // PASO_EJE), int(tope // PASO_EJE) + 1)]
    molde = _molde()
    fig = {"data": [aportes, extremos],
           "layout": dict(molde, annotations=list(molde["annotations"]) + anotaciones,
                          xaxis=dict(molde["xaxis"], range=[-tope, tope], tickvals=marcas,
                                     ticktext=[f"{ui.num(m)}<br>{ui.num(M.sigmoide(m), DEC_EJE)}" for m in marcas]),
                          yaxis=dict(molde["yaxis"], range=[n - 1 + GROSOR_BARRA + AIRE_ABAJO, -AIRE_ARRIBA]))}

    corte = _corte("ebm")
    donde = "sobre" if sim["p"] >= corte else "bajo"
    titulo = (f"{'Con los cambios suman' if hay else 'Los aportes suman'} un logit de {ui.num(final, DEC_LOGIT)}: puntaje "
              f"{ui.num(sim['p'], DEC_PUNTAJE)}, {donde} el corte de la campaña")
    if hay:
        sub = f"Fila {fila}, cliente simulado: {_texto_cambios(real, sim, cambios)}."
    else:
        sub = (f"Fila {fila}, cliente real. Desde el intercepto, cada variable suma o resta su aporte; bajo cada marca del eje, el "
               "puntaje de ese logit.")
    etiqueta = f"{titulo}. " + "; ".join(f"{r}: {v}" for r, v in zip(rotulos, valores)) + "."
    return fig, titulo, sub, etiqueta


def _nota_cascada():
    propias, _otras = _con_paso()
    return (f"Las {len(propias)} variables con paso propio concentran el {F.pct(M.importancia('clasificacion').iloc[:len(propias)].sum(), 0)} "
            "de la importancia del modelo.")


def _tabla_aportes(real, sim, hay):
    """Tabla gemela de la cascada: el aporte de cada una de las variables, de mayor a menor en valor absoluto."""
    a, x, a0, x0 = sim["aportes"], sim["x"], real["aportes"], real["x"]
    filas = [[PARTIDA, F.RAYA, ui.con_signo(sim["intercepto"], DEC_TABLA)] + ([F.RAYA, ui.con_signo(real["intercepto"], DEC_TABLA)] if hay else [])]
    for v in sorted(a, key=lambda k: -abs(a[k])):
        f = [M.nombre(v), _valor(v, x[v]), ui.con_signo(a[v], DEC_TABLA)]
        filas.append(f + ([_valor(v, x0[v]), ui.con_signo(a0[v], DEC_TABLA)] if hay else []))
    total = ([SUMA, f"puntaje {ui.num(sim['p'], DEC_PUNTAJE)}", ui.num(sim["logit"], DEC_TABLA)]
             + ([f"puntaje {ui.num(real['p'], DEC_PUNTAJE)}", ui.num(real["logit"], DEC_TABLA)] if hay else []))
    return ui.tabla(["Variable", "Valor", "Aporte al logit"] + (["Valor real", "Aporte real"] if hay else []), filas, total=total,
                    compacta=True, num=[False, False, True] + ([False, True] if hay else []), ocultar_movil=(3,) if hay else ())


def _lo_que_paso(fila):
    """Tarjeta del cliente real: su puntaje real, su resultado en la campaña y la variable que el modelo no ve."""
    C, meta = M.clientes(), M.meta()
    c = C.loc[fila]
    alto, contactado = bool(c["riesgo_alto"]), bool(c["contactar_ebm"])
    productos = int(c["active_products"])
    por_productos = D.tabla("eda/por_variable.csv")
    por_productos = por_productos[(por_productos["variable"] == "active_products") & (por_productos["x"] == productos)]
    pares = [
        ("Puntaje real, el del conjunto de datos", ui.num(c["churn_probability"], DEC_PUNTAJE)),
        (f"Riesgo alto: puntaje real por encima de {ui.num(meta['etiqueta']['umbral'], DEC_PUNTAJE)}", "sí" if alto else "no"),
        (f"Resultado con capacidad para el {F.pct(meta['punto_operacion']['fraccion'], 0)}",
         f"{RESULTADOS[c['resultado_ebm']]} ({c['resultado_ebm']})"),
        ("Productos activos (el modelo no ve esta variable)", F.entero(productos)),
    ]
    if len(por_productos) == 1:
        pares.append((f"Clientes de entrenamiento con {F.entero(productos)} {'producto activo' if productos == 1 else 'productos activos'} "
                      "que son de riesgo alto",
                      F.pct(por_productos["tasa_riesgo_alto"].iloc[0])))
    pares += [("Cohorte de edad", f"{c['cohorte']} años"), ("Identificador del cliente", str(int(c["customer_id"]))),
              ("Fila en el archivo de clientes", str(int(fila)))]
    if int(fila) in _casos():
        pares.append(("Caso del informe", _casos()[int(fila)]))
    titulo = (f"Puntaje real de {ui.num(c['churn_probability'], DEC_PUNTAJE)}: {'es' if alto else 'no es'} de riesgo alto y "
              f"{'entra en' if contactado else 'queda fuera de'} la campaña ({RESULTADOS[c['resultado_ebm']]})")
    return ui.tarjeta([ui.ficha(pares),
                       ui.nota("El número de productos activos generó el puntaje del conjunto de datos y por eso no es un predictor: "
                               "aquí solo ayuda a leer el resultado. Esta tarjeta muestra siempre al cliente real.")],
                      titulo=titulo, id=_i("paso"),
                      fuentes=[D.fuente("clientes_prueba.parquet"), D.fuente("eda/por_variable.csv"), D.fuente("metadatos.json")])


def _otros_modelos(fila, sim, hay):
    """Tarjeta con el puesto que le da cada modelo: el guardado del cliente real y, para un simulado, los que se recalculan."""
    C = M.clientes()
    c = C.loc[fila]
    filas, puestos = [], {}
    for m in MODELOS:
        puestos[m] = _puesto(m, float(c[f"p_{m}"]))
        puesto, campana = F.entero(puestos[m]), "entra" if bool(c[f"contactar_{m}"]) else "queda fuera"
        if hay and m in EN_VIVO:                       # tras la flecha, el cliente simulado
            p = sim[EN_VIVO[m]]
            puesto = f"{puesto} → {F.entero(_puesto(m, p))}"
            campana = f"{campana} → {'entraría' if p >= _corte(m) else 'quedaría fuera'}"
        filas.append([D.NOMBRE_MODELO[m], ui.num(c[f"p_{m}"], DEC_PUNTAJE), puesto, campana])
    tabla = ui.tabla(["Modelo", "Puntaje", "Puesto", "Campaña"], filas, compacta=True, muestras=[D.ROL_VISTA_EBM[m] for m in MODELOS],
                     num=[False, True, True, False], ocultar_movil=(1,))
    titulo = (f"{D.en_frase('xgboost')} lo pone en el puesto {F.entero(puestos['xgboost'])}; {D.en_frase('logistica')}, en el "
              f"{F.entero(puestos['logistica'])}; {D.en_frase('ebm')}, en el {F.entero(puestos['ebm'])}")
    titulo = titulo[0].upper() + titulo[1:]
    simulado = ("Tras la flecha va el cliente simulado: su puesto entre los mismos clientes y si entraría en la campaña. El puntaje es "
                "siempre el del cliente real. ") if hay else ""
    return ui.tarjeta([tabla,
                       ui.nota(f"Puesto del cliente real entre los {F.entero(len(C))} clientes de prueba según cada modelo; el primero es "
                               "el de mayor puntaje. Se comparan los puestos y no los puntajes: cada modelo tiene su escala."),
                       ui.nota(f"{simulado}El puntaje de {D.en_frase('xgboost')} está guardado para los clientes de prueba y no se "
                               f"recalcula para un cliente simulado; {D.en_frase('ebm')} y {D.en_frase('logistica')} se calculan en el "
                               "momento.")],
                      titulo=titulo, id=_i("otros"),
                      fuentes=[D.fuente("clientes_prueba.parquet"), D.fuente("modelos/logistica_clasificacion.json")])


def _vivo(fila, cambios):
    """Lo que cambia con cada movimiento del panel, en el orden de las salidas de _calcular."""
    real, sim, hay = _estado(fila, cambios)
    fig, titulo, sub, etiqueta = _cascada(fila, real, sim, cambios)
    atadas = [(v, _valor(v, sim["x"][v]), _valor(v, real["x"][v])) for v in _calculadas()]
    calculadas = ui.ficha([(M.nombre(v), _con_real(s, r, hay)) for i, (v, s, r) in enumerate(atadas) if i < N_CALCULADAS or s != r])
    tiene, tenia = [sum(bool(e["x"][p]) for p in M.PRODUCTOS) for e in (sim, real)]
    productos = f"Productos que tiene: {_con_real(f'{tiene} de {len(M.PRODUCTOS)}', f'{tenia} de {len(M.PRODUCTOS)}', hay)}"
    return (_fichas(real, sim, hay), fig, titulo, sub, etiqueta, calculadas,
            _valor("app_logins_frequency", sim["x"]["app_logins_frequency"]), _valor("base_satisfaction", sim["x"]["base_satisfaction"]),
            productos, not hay)


def _detalle(fila, cambios):
    """Lo que solo se rehace al soltar un control: la tabla de los aportes y la tarjeta de los otros modelos."""
    real, sim, hay = _estado(fila, cambios)
    return _tabla_aportes(real, sim, hay), _otros_modelos(fila, sim, hay)


# --- Pantalla -------------------------------------------------------------------------------------------------------------
def _subtitulo():
    """Una sola línea a 1366 px (cada línea de la cabecera baja la parte interactiva), con la cifra del corte."""
    return f"Porque sus variables suman un puntaje sobre o bajo el corte de la campaña ({ui.num(_corte('ebm'), DEC_PUNTAJE)})."


def _selector_cliente(fila):
    """Los controles de la cabecera: los tres casos del informe, otro cliente al azar (con dos filtros) y la fila. La fila
    se elige en un desplegable con búsqueda que solo ofrece los clientes de prueba."""
    casos = _casos()
    caso = ui.control("Casos del informe:", ui.segmentado(_i("caso"), [{"label": CASOS.get(k, k.capitalize()), "value": f} for f, k in casos.items()],
                                                         fila if fila in casos else None, persistir=False))
    resultados = [{"label": "Cualquier resultado", "value": TODOS}] + [{"label": f"{nombre.capitalize()} ({clave})", "value": clave}
                                                                       for clave, nombre in RESULTADOS.items()]
    cohortes = [{"label": "Cualquier edad", "value": TODOS}] + [{"label": f"{c} años", "value": c} for c in _cohortes()]
    azar = html.Div([ui.selector(_i("f-resultado"), resultados, TODOS, persistir=False),
                     ui.selector(_i("f-cohorte"), cohortes, TODOS, persistir=False),
                     ui.boton("Otro al azar", id=_i("azar"), tipo="secundario", icono="canje", pequeno=True)], className="controles")
    ir = ui.control("Fila:", ui.selector(_i("ir"), list(_filas()), fila, buscar=True, persistir=False))
    return [caso, azar, ir]


def _panel(fila, calculadas, texto_logins, texto_base, texto_productos):
    """Panel «¿Y si…?»: solo las variables que tiene sentido mover, con los valores del cliente. Los cinco productos son
    variables secundarias y van plegados. Los deslizadores avisan al soltar (value) y mientras se arrastran (drag_value)."""
    propios, v = _controles(fila), M.variables()
    logins = _valores("app_logins_frequency")
    lo, hi = _rango_base()
    marcas_logins = {int(i): F.entero(logins[int(i)]) for i in np.linspace(0, len(logins) - 1, MARCAS_LOGINS).round()}
    marcas_base = {k: ui.num(k) for k in range(math.ceil(lo), math.floor(hi) + 1)}
    return html.Div([
        html.Div([html.H2("¿Y si…?", className="panel__titulo"),
                  ui.boton("Volver al cliente real", id=_i("volver"), tipo="fantasma", pequeno=True)], className="panel__cab"),
        ui.control(f"{M.nombre('age')} ({v.at['age', 'unidad']})",
                   ui.segmentado(_i("edad"), [{"label": str(e), "value": e} for e in _valores("age")], propios["edad"], pequeno=True,
                                 persistir=False)),
        ui.deslizador(_i("logins"), M.nombre("app_logins_frequency"), 0, len(logins) - 1, 1, propios["logins"], marcas=marcas_logins,
                      persistir=False, valor_inicial_texto=texto_logins),
        ui.deslizador(_i("base"), M.nombre("base_satisfaction"), lo, hi, PASO_BASE, propios["base"], marcas=marcas_base,
                      persistir=False, valor_inicial_texto=texto_base),
        ui.desplegable(html.Span(texto_productos, id=_i("productos-resumen")),
                       ui.casilla(_i("productos"), [{"label": M.nombre(p), "value": p} for p in M.PRODUCTOS], propios["productos"],
                                  lista=True, persistir=False), llano=True),
        html.Div([html.P("Variables que se recalculan", className="deslizador__rotulo"),
                  html.Div(calculadas, id=_i("calculadas"))], className="deslizador-bloque"),
    ], className="panel-parametros")


def _que_mide():
    """«Qué mide cada cifra», con las cifras de definición leídas de los datos."""
    meta, n = M.meta(), len(M.clientes())
    cal = D.fila(D.tabla("entrega3/complementos/calibracion.csv"), modelo="ebm", version="sin recalibrar")
    alta = float(_recalibradas()[0].max())                 # el escalón más alto de la isotónica entre los clientes de prueba
    n_alta, altos_alta = _escalon(alta)
    return ui.desplegable("Qué mide cada cifra", [
        ui.parrafo(f"Puntaje del modelo: la probabilidad sin recalibrar que la EBM de clasificación asigna a la clase de riesgo alto. "
                   f"Sirve para ordenar a los clientes, no para leerla como probabilidad: el modelo se ajustó con ponderación de clases y "
                   f"su puntaje medio en la prueba es {ui.num(cal['prob_media'], 3)} frente a una prevalencia de {ui.num(cal['prevalencia'], 3)}."),
        ui.parrafo(f"Puesto en el orden de riesgo: lugar del puntaje entre los {F.entero(n)} clientes de prueba; el primero es el de mayor "
                   f"puntaje. Con capacidad para el {F.pct(meta['punto_operacion']['fraccion'], 0)}, la campaña contacta a los "
                   f"{F.entero(meta['punto_operacion']['contactados'])} primeros: desde un puntaje de {ui.num(_corte('ebm'), DEC_PUNTAJE)}. "
                   f"Un cliente simulado se ubica entre esos mismos clientes."),
        ui.parrafo(f"Probabilidad recalibrada: el puntaje pasado por la recalibración isotónica del proyecto, que va por escalones: a "
                   f"todos los clientes de un tramo de puntaje les da la fracción de riesgo alto que tuvo ese tramo al ajustarla. Por eso "
                   f"solo toma {_n_recalibradas()} valores distintos en la prueba, los clientes parecidos la comparten y no sirve para "
                   f"ordenarlos. Los escalones de los extremos tienen pocos clientes: el más alto, de {F.pct(alta, 0)}, lo comparten "
                   f"{F.entero(n_alta)} clientes de prueba y {F.entero(altos_alta)} son de riesgo alto. No es una certeza."),
        ui.parrafo(f"Pronóstico de la EBM de regresión: el valor que ese modelo pronostica para el puntaje del conjunto de datos. Va en "
                   f"otra escala que el puntaje del clasificador: el riesgo alto es el cuarto superior del puntaje real, por encima de "
                   f"{ui.num(meta['etiqueta']['umbral'], DEC_PUNTAJE)}."),
        ui.parrafo(f"Aporte al logit: lo que cada variable suma o resta en la escala en que el modelo suma. El punto de partida más los "
                   f"aportes de las {len(M.predictores())} variables es exactamente el logit del cliente, y su puntaje es la función "
                   f"logística de esa suma. No hay aproximación."),
        ui.parrafo("Variables que se recalculan: el puntaje de satisfacción, la satisfacción con los productos y con las transacciones y "
                   "el uso del cupo de la tarjeta están atados a otras variables por identidades que se cumplen en todos los clientes. "
                   "Al mover un control, el simulador las recalcula para no puntuar a un cliente imposible."),
    ])


def _tarjeta_cifras(fichas):
    """La tarjeta de las cifras: su cabecera es el rótulo del primer bloque (con el «…» de las fuentes) y no un título de
    tarjeta, para que los dos bloques se rotulen igual y la tarjeta no crezca."""
    return html.Div([
        html.Div([html.Div(_rotulo_bloque(ESCALA_CLASIFICADOR), className="tarjeta__textos"), ui.mas(None, FUENTES_FICHAS)],
                 className="tarjeta__cab"),
        html.Div(fichas, id=_i("fichas"), className="panel__grupo"),
    ], className="tarjeta")


def layout(fila=None, **_):
    if isinstance(fila, (list, tuple)):          # ?fila=16840&fila=6383: vale la primera
        fila = fila[0] if fila else None
    elegida, aviso = _resolver(fila)
    fichas, fig, titulo, sub, etiqueta, calculadas, texto_logins, texto_base, texto_productos, _sin_cambios = _vivo(elegida, ())
    tabla, otros = _detalle(elegida, ())
    resultados = html.Div(ui.rejilla([
        _tarjeta_cifras(fichas),
        ui.grafico(fig, titulo, sub, etiqueta, tabla, nota=_nota_cascada(), id=_i("cascada"), fuentes=FUENTES_MODELO,
                   leyenda=[("Punto de partida y suma", "modelo", "foco"), ("Aporte de cada variable", "modelo", "rayada")],
                   resumen_tabla="Ver los datos (los aportes de todas las variables)"),
    ], "1-2"), className="resultados")
    laboratorio = html.Div([
        html.Div(aviso, id=_i("aviso")),
        html.Div([_panel(elegida, calculadas, texto_logins, texto_base, texto_productos), resultados], className="laboratorio"),
        dcc.Store(id=_i("fila"), data=elegida),
        dcc.Store(id=_i("direccion")),               # salida muda del callback del navegador que sigue al cliente
    ])
    return ui.pantalla([
        ui.encabezado(PREGUNTA, _subtitulo(), _selector_cliente(elegida), respuesta=laboratorio, pantalla=PANTALLA),
        ui.seccion("Lo que pasó con el cliente real y lo que el modelo no ve", [
            ui.aviso("El panel «¿Y si…?» muestra el puntaje que el modelo asigna a un cliente con esos valores. No dice qué pasaría con "
                     "el cliente real si cambiara: el modelo describe lo que aprendió de los datos, no el efecto de intervenir.",
                     titulo="Un cliente simulado es hipotético"),
            ui.rejilla([_lo_que_paso(elegida), otros], "1-1"),
            _que_mide(),
        ], subtitulo="El cliente real tiene un resultado; el simulado, no. Los otros modelos se comparan por el puesto que le dan.",
            fuentes=FUENTES_ABAJO),
    ])


# --- Callbacks ------------------------------------------------------------------------------------------------------------
@callback(Output(_i("fila"), "data"), Output(_i("caso"), "value"), Output(_i("ir"), "value"), Output(_i("aviso"), "children"),
          Input(_i("caso"), "value"), Input(_i("azar"), "n_clicks"), Input(_i("ir"), "value"),
          State(_i("f-resultado"), "value"), State(_i("f-cohorte"), "value"), State(_i("fila"), "data"), prevent_initial_call=True)
def _elegir(caso, _clics, ir, resultado, cohorte, actual):
    """El cliente elegido, venga de un caso del informe, del botón «Otro al azar» o del desplegable «Fila»."""
    origen = ctx.triggered_id
    if origen == _i("caso") and _fila_valida(caso) is not None:
        fila, aviso = _fila_valida(caso), None
    elif origen == _i("azar"):
        fila, aviso = _al_azar(resultado, cohorte, actual)
    elif origen == _i("ir") and ir is not None and str(ir).strip():
        fila, aviso = _resolver(ir)
    else:
        raise PreventUpdate
    return fila, (fila if fila in _casos() else None), fila, aviso


@callback(Output(_i("edad"), "value"), Output(_i("logins"), "value"), Output(_i("base"), "value"), Output(_i("productos"), "value"),
          Input(_i("fila"), "data"), Input(_i("volver"), "n_clicks"))
def _reponer(fila, _clics):
    """Al cambiar de cliente, o con «Volver al cliente real», los controles toman los valores del cliente."""
    propios = _controles(_fila_o_la_de_omision(fila))
    return propios["edad"], propios["logins"], propios["base"], propios["productos"]


@callback(Output(_i("fichas"), "children"), Output(_i("cascada"), "figure"), Output(_i("cascada-titulo"), "children"),
          Output(_i("cascada-sub"), "children"), Output(_i("cascada-lienzo"), "aria-label"), Output(_i("calculadas"), "children"),
          Output(_i("logins-valor"), "children"), Output(_i("base-valor"), "children"), Output(_i("productos-resumen"), "children"),
          Output(_i("volver"), "disabled"),
          Input(_i("fila"), "data"), Input(_i("edad"), "value"), Input(_i("logins"), "value"), Input(_i("logins"), "drag_value"),
          Input(_i("base"), "value"), Input(_i("base"), "drag_value"), Input(_i("productos"), "value"))
def _calcular(fila, edad, logins, logins_arrastre, base, base_arrastre, productos):
    """Con cada movimiento: M.simular → M.puntuar → M.explicar, las fichas y la cascada. Mientras se arrastra un deslizador
    vale su drag_value; en cualquier otro caso, su value (el drag_value queda viejo cuando el valor lo repone un callback)."""
    movidos = ctx.triggered_prop_ids

    def vigente(nombre, valor, arrastre):
        solo_arrastre = f"{_i(nombre)}.drag_value" in movidos and f"{_i(nombre)}.value" not in movidos
        return arrastre if solo_arrastre and arrastre is not None else valor
    fila = _fila_o_la_de_omision(fila)
    return _vivo(fila, _cambios(fila, edad, vigente("logins", logins, logins_arrastre), vigente("base", base, base_arrastre), productos))


@callback(Output(_i("cascada-tabla"), "children"), Output(_i("otros"), "children"),
          Input(_i("fila"), "data"), Input(_i("edad"), "value"), Input(_i("logins"), "value"), Input(_i("base"), "value"),
          Input(_i("productos"), "value"))
def _detallar(fila, edad, logins, base, productos):
    """Al soltar un control: la tabla con los 48 aportes y lo que dicen los otros modelos."""
    fila = _fila_o_la_de_omision(fila)
    tabla, otros = _detalle(fila, _cambios(fila, edad, logins, base, productos))
    return tabla, otros.children


@callback(Output(_i("paso"), "children"), Input(_i("fila"), "data"))
def _cliente_real(fila):
    """Lo que pasó con el cliente real: solo cambia con el cliente."""
    return _lo_que_paso(_fila_o_la_de_omision(fila)).children


# En el navegador, con cada cliente elegido:
#  · la dirección pasa a /cliente?fila=<fila> con history.replaceState, que no dispara el enrutado de Dash: al recargar, o al
#    volver con «atrás» desde otra pantalla, se abre el mismo cliente;
#  · el desplegable «Fila» queda vigilado (una sola vez): al abrirse, Dash deja el foco en la opción elegida y aquí se pasa
#    al buscador, para poder escribir el número sin más.
clientside_callback(
    r"""function (fila) {
        var sin_cambio = window.dash_clientside.no_update;
        if (fila === null || fila === undefined) { return sin_cambio; }
        var ruta = window.location.pathname;
        if (ruta.replace(/\/+$/, '') === __RUTA__) {
            var destino = ruta + '?fila=' + encodeURIComponent(fila);
            if (ruta + window.location.search !== destino) { window.history.replaceState(window.history.state, '', destino); }
        }
        var boton = document.getElementById(__DESPLEGABLE__);
        if (boton && !boton.dataset.buscador) {
            boton.dataset.buscador = 'si';
            new MutationObserver(function () {
                if (boton.getAttribute('aria-expanded') !== 'true') { return; }
                window.requestAnimationFrame(function () { window.requestAnimationFrame(function () {
                    var lista = document.getElementById(boton.getAttribute('aria-controls')) || document;
                    var buscador = lista.querySelector('.dash-dropdown-search');
                    if (buscador) { buscador.focus(); }
                }); });
            }).observe(boton, {attributes: true, attributeFilter: ['aria-expanded']});
        }
        return sin_cambio;
    }""".replace("__RUTA__", json.dumps("/modelos/cliente")).replace("__DESPLEGABLE__", json.dumps(_i("ir"))),
    Output(_i("direccion"), "data"), Input(_i("fila"), "data"))
