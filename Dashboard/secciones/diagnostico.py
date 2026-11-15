"""Diagnóstico · ¿Cómo acierta y cómo se equivoca cada modelo?

El diagnóstico del modelo elegido en la prueba (guía §7.3.3), con los mismos puntajes y pronósticos que reproducen los
artefactos (D.puntajes() y D.pronosticos(), una columna por modelo final):

  · Clasificación: la matriz de confusión en el punto de operación del informe (contactar al 25 % de mayor puntaje, con la
    regla y el desempate de M.punto_operacion), la curva ROC con su AUC, la curva de precisión-recall con su área y el
    diagrama de calibración (el puntaje sin recalibrar cuando está entre 0 y 1 y la probabilidad recalibrada, con el ECE
    de los mismos tramos que datos/calibracion_modelos.json y el Brier).
  · Regresión: el puntaje observado frente al pronóstico (una muestra fija de clientes y la media de todos por grupos),
    el residuo frente al pronóstico, la autocorrelación del residuo en el orden del archivo original (que agrupa a los
    clientes por cohorte de edad: no es una serie de tiempo) con Ljung-Box, y el histograma del residuo.

Las cifras que el informe publica (AUC con su intervalo, AUC-PR, MCC, RMSE, MAE, R², ECE, Brier) se leen de
datos/resultados/*_prueba.csv y de datos/calibracion_modelos.json; las curvas se calculan aquí con los puntajes y la vista
comprueba en cada dibujo que lo calculado coincide con lo publicado (si no, lo dice en un aviso). La dirección puede pedir
la tarea y el modelo: /modelos/diagnostico?tarea=regresion&modelo=lasso (así enlaza la Tabla de líderes).
"""
from functools import lru_cache
from html import escape

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
from scipy import stats

import app
import datos_app as D
import formato as F
import modelos_vivos as V
import motor as M
import ui

VISTA = "diagnostico"
PRE = "dg"
PANTALLA = "robustez"                 # el icono de la cabecera (el de «gráfico», como la vista en app.VISTAS)
TAREAS = ("clasificacion", "regresion")
INICIAL = "ebm"                        # el modelo con que nace la vista: el propuesto

CAPACIDAD = M.CAPACIDAD_INFORME        # punto de operación del informe
TRAMOS_CALIBRACION = 10                # tramos de igual ancho del ECE (preparar_artefactos.py::ece); se comprueba al dibujar
PUNTOS_CURVA = 201                     # puntos de las curvas ROC y de captura (para dibujar; las áreas son las exactas)
PASO_TABLA_CURVA = 20                  # la tabla gemela de la curva ROC lleva uno de cada veinte puntos
PASO_TABLA_PR = 0.05                   # la de precisión-recall, de cinco en cinco puntos de capacidad
CAPACIDAD_CORTA = 0.10                 # la otra capacidad que discute el informe (recall_10, precision_10)
MUESTRA_NUBE = 2000                    # clientes que se dibujan como puntos (al azar, con la semilla del proyecto)
GRUPOS = 20                            # grupos de igual tamaño ordenados por el pronóstico (como la figura del informe)
RETARDOS_ACF = 20                      # retardos de la autocorrelación del residuo
RETARDOS_LB = 10                       # Ljung-Box con 10 retardos, como el diagnóstico del Entregable 3 (src/evaluacion.py)
NIVEL = 0.95                           # nivel de la banda de la autocorrelación
TRAMOS_HISTOGRAMA = 40                 # barras del histograma del residuo
PAREJA = 1.25                          # la dispersión del residuo es «pareja» si la mayor de los grupos no pasa de 1,25 veces la menor
SIGUE = 0.25                           # por grupos el pronóstico «sigue» a lo observado si no se aparta más de 1/4 de la desviación del puntaje
FRANJA = 0.4                           # los niveles de active_products se ven como franjas si el residuo dentro de cada uno se
                                       # dispersa menos de 0,4 veces la menor distancia entre las medias de dos niveles
SESGO_PEQUENO = 0.25                   # un sesgo de grupo es «pequeño» si no pasa de 1/4 de la desviación del residuo
TOLERANCIA = 1e-6                      # diferencia admitida entre lo calculado aquí y lo publicado

TAM_NUBE = ui.MARCADOR * 0.5           # px: los puntos de una nube de clientes, sin anillo
OPACIDAD_NUBE = 0.28
PASO_EJE_01 = 0.25                     # marcas de los ejes que van de 0 a 1
MARGEN_01 = 0.02                       # aire alrededor de los ejes de 0 a 1
HOLGURA = 1.25                         # aire sobre el mayor valor de un eje simétrico

FUENTE_PUNTAJES = D.fuente("puntajes_prueba.parquet")
FUENTE_PRONOSTICOS = D.fuente("pronosticos_prueba.parquet")
FUENTE_CLF = D.fuente("resultados/clasificacion_prueba.csv")
FUENTE_REG = D.fuente("resultados/regresion_prueba.csv")
FUENTE_CAL = D.fuente("calibracion_modelos.json")
FUENTE_CLIENTES = D.fuente("clientes_prueba.parquet")
FUENTE_PRUEBAS = {t: D.fuente(f"resultados/pruebas_{t}.csv") for t in TAREAS}
PRUEBA_FIJADA = {"clasificacion": "DeLong", "regresion": "Diebold-Mariano"}


# --- Textos ---------------------------------------------------------------------------------------------------------------
def _mayuscula(texto):
    """Primera letra en mayúscula, salvo en un nombre que empieza en minúscula (k-NN)."""
    return texto if texto.startswith("k-") else texto[:1].upper() + texto[1:]


def _tras(preposicion, frase):
    """«a» o «de» delante de un nombre en frase, con su contracción: «al árbol de decisión», «del SVM lineal»."""
    if frase.startswith("el "):
        return {"a": "al", "de": "del"}[preposicion] + frase[2:]
    return f"{preposicion} {frase}"


def _lista(modelos):
    return F.enumerar(D.en_frase(m) for m in modelos)


def _pct_cap():
    return F.pct(CAPACIDAD, 0)


# --- Tarea y modelo (de la dirección, de los controles) ----------------------------------------------------------------
def _tarea(t):
    return t if t in TAREAS else "clasificacion"


def _modelo(tarea, m):
    """El modelo pedido si es uno de los ocho de la tarea; si no, el propuesto."""
    ms = V.modelos(tarea)
    return m if m in ms else (INICIAL if INICIAL in ms else ms[0])


def _de_la_direccion(tarea, modelo):
    """(tarea, modelo, pedida) a partir de ?tarea= y ?modelo=. Un modelo que solo existe en una tarea la elige (Lasso →
    regresión); lo desconocido cae en clasificación y en la EBM."""
    if tarea not in TAREAS and modelo:
        en = [t for t in TAREAS if modelo in V.modelos(t)]
        if len(en) == 1:
            tarea = en[0]
    pedida = tarea in TAREAS or bool(modelo)
    t = _tarea(tarea)
    return t, _modelo(t, modelo), pedida


def _opciones(tarea):
    """Los ocho modelos de la tarea, con su nombre como texto. (Con un componente de rótulo, ui.clave, el desplegable de
    Dash 4.5 deja en blanco las opciones que reaparecen tras cambiar las opciones por callback: el color del modelo va en
    la muestra del rótulo «Modelo:», _rotulo_modelo.)"""
    return [{"label": D.NOMBRE_MODELO[m], "value": m} for m in V.modelos(tarea)]


def _rotulo_modelo(modelo):
    """El rótulo del selector con la muestra del color del rol del modelo elegido (azul, rosa o gris): el mismo color con
    que se dibuja en los gráficos."""
    return [ui.muestra(D.ROL_MODELO.get(modelo, "contexto")), html.Span("Modelo:")]


def _selector_modelo(tarea, modelo, persistir):
    """El selector nace con los modelos de las dos tareas (el de la tarea deja enseguida los ocho que tocan): recuerda su
    valor en la sesión y, si ese valor no estuviera entre las opciones al recargar, el desplegable lo borraría."""
    propias = _opciones(tarea)
    valores = {o["value"] for o in propias}
    otras = [o for t in TAREAS if t != tarea for o in _opciones(t) if o["value"] not in valores]
    selector = ui.selector(f"{PRE}-modelo", propias + otras, modelo, persistir=persistir)
    selector.maxHeight = len(propias) * ui.px("--e-7")
    return selector


def _tabla(encabezados, filas, alta=False, num=None):
    """La tabla gemela de un gráfico como UN solo componente: el HTML de ui.tabla (mismas clases «tabla-marco», «tabla»,
    «tabla--compacta» y «num» de estilos.css) dentro de un dcc.Markdown. Una tabla de ui.tabla es un componente de Dash
    por celda, y rehacer componentes es lo caro en el navegador (1 a 3 ms cada uno): con las tablas gemelas de esta vista
    hechas así, cambiar de modelo tardaba más de un segundo y medio. Las celdas son textos ya formateados de los datos
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


# --- Piezas de gráfico ----------------------------------------------------------------------------------------------------
def _forma_linea(rol):
    """La forma de línea de un rol: la base va a trazos (el gris y el rosa nunca con la misma forma)."""
    return "trazos" if rol == "base" else "linea"


def _nube(rol):
    """Una nube de clientes: puntos pequeños del color del rol, sin anillo y con transparencia (se ve la densidad)."""
    return {"mode": "markers", "marker": {"size": TAM_NUBE, "color": ui.ROL[rol], "opacity": OPACIDAD_NUBE, "line": {"width": 0}}}


def _hover_pares(plantilla, *columnas):
    """Texto flotante de muchos puntos (una nube): la plantilla fija va una sola vez y cada punto solo lleva sus valores ya
    formateados en customdata ({0}, {1}… de la plantilla son sus columnas). Pesa mucho menos que ui.hover_texto, que repite
    la frase entera en cada punto."""
    campos = plantilla.format(*[f"%{{customdata[{i}]}}" for i in range(len(columnas))])
    return {"customdata": [list(f) for f in zip(*columnas)], "hovertemplate": campos + "<extra></extra>"}


def _ejes_01(fig):
    """Los dos ejes de 0 a 1 con marcas en es-CO y algo de aire para que la línea no se corte en el borde."""
    valores, textos = ui.marcas(0, 1, PASO_EJE_01)
    rango = [-MARGEN_01, 1 + MARGEN_01]
    fig.update_xaxes(range=rango, tickvals=valores, ticktext=textos, showgrid=True, gridcolor=ui.ROL["rejilla"])
    fig.update_yaxes(range=rango, tickvals=valores, ticktext=textos)
    fig.update_layout(hovermode="closest")
    return fig


def _eje(fig, eje, lo, hi, maximo_marcas=4, signo=False, d=None):
    """Un eje de valores con marcas redondas en es-CO de lo a hi."""
    paso = ui.paso_redondo(max(abs(lo), abs(hi)) if signo else hi - lo, maximo_marcas)
    valores, textos = ui.marcas(lo, hi, paso, d=d, signo=signo)
    (fig.update_xaxes if eje == "x" else fig.update_yaxes)(range=[lo, hi], tickvals=valores, ticktext=textos)
    return fig


def _grafico(fig, titulo, sub, tabla, leyenda, nota, fuentes, gid, alto="normal"):
    etiqueta = f"{titulo}. {nota}"
    return ui.grafico(fig, titulo, sub, etiqueta, tabla=tabla, leyenda=leyenda, nota=nota, alto=alto, id=gid, fuentes=fuentes)


def _avisos_coincidencia(diferencias):
    """Un aviso si alguna cifra calculada aquí no coincide con la publicada; nada si todo coincide."""
    malas = [que for que, a, b in diferencias if a is not None and b is not None and abs(float(a) - float(b)) > TOLERANCIA]
    if not malas:
        return None
    return ui.aviso(f"Lo calculado en la pantalla no coincide con lo publicado en: {F.enumerar(malas)}. Las cifras de las fichas "
                    "son las publicadas.", "atencion", "Revisar los datos")


# --- Clasificación: los números (con caché; no modificar lo que devuelven) ----------------------------------------------
def _tramos(y, p):
    """Diagrama de calibración en TRAMOS_CALIBRACION tramos de igual ancho entre 0 y 1: por tramo, clientes, valor medio
    y fracción observada de riesgo alto; y el ECE (diferencia media ponderada por clientes)."""
    b = np.clip((p * TRAMOS_CALIBRACION).astype(int), 0, TRAMOS_CALIBRACION - 1)
    filas = []
    for i in range(TRAMOS_CALIBRACION):
        m = b == i
        if m.any():
            filas.append({"tramo": i, "n": int(m.sum()), "dice": float(p[m].mean()), "observado": float(y[m].mean())})
    ece = sum(abs(f["observado"] - f["dice"]) * f["n"] for f in filas) / len(p)
    return tuple(filas), float(ece)


@lru_cache(maxsize=None)
def _clasificacion_datos(modelo):
    P = D.puntajes()
    y = P["y"].to_numpy(dtype=int)
    s = P[modelo].to_numpy(dtype=float)
    fila = D.fila(D.prueba("clasificacion"), modelo=modelo)
    cal = V.calibracion()["modelos"][modelo]
    en_01 = bool(s.min() >= 0 and s.max() <= 1)
    p_cal = V.probabilidad(modelo, s)
    crudo = _tramos(y, s) if en_01 else (None, None)
    recal = _tramos(y, p_cal)
    return {
        "y": y, "s": s, "fila": fila, "cal": cal, "en_01": en_01, "n": len(y), "positivos": int(y.sum()),
        "prevalencia": float(y.mean()), "media": float(s.mean()),
        "op": M.resumen_capacidad(y, s, CAPACIDAD), "roc": M.curva_roc(y, s, PUNTOS_CURVA),
        "captura": M.curva_captura(y, s, PUNTOS_CURVA), "auc": M.auc(y, s),
        "tramos_crudo": crudo[0], "ece_crudo": crudo[1], "tramos_cal": recal[0], "ece_cal": recal[1],
        "brier_crudo": float(np.mean((s - y) ** 2)) if en_01 else None, "brier_cal": float(np.mean((p_cal - y) ** 2)),
    }


def _relacion(tarea, modelo):
    """Qué dicen las pruebas del informe sobre el modelo y la EBM (D.veredicto, p ajustados por Holm)."""
    pr = D.pruebas(tarea)
    fijada = PRUEBA_FIJADA[tarea]
    if modelo == "ebm":
        v = D.veredictos(tarea)
        empates, superadas = [m for m in v.index if v[m] == "empata"], [m for m in v.index if v[m] == "supera"]
        partes = []
        if empates:
            partes.append(f"ni la t corregida ni {fijada} la separan {_tras('de', _lista(empates))} tras la corrección de Holm")
        if superadas:
            partes.append(f"supera con significación {_tras('a', _lista(superadas))}")
        return _mayuscula("; ".join(partes)) + "." if partes else ""
    v = D.veredicto(D.fila(pr, modelo=modelo), tarea)
    frase = D.en_frase(modelo)
    return {"supera": f"La EBM supera {_tras('a', frase)} con significación tras la corrección de Holm.",
            "empata": f"Ni la t corregida ni {fijada} encuentran diferencia con la EBM tras la corrección de Holm.",
            "discrepan": "Frente a la EBM, las pruebas discrepan tras la corrección de Holm.",
            "pierde": f"{_mayuscula(frase)} supera a la EBM con significación tras la corrección de Holm."}[v]


def _respuesta(tarea, modelo):
    """El subtítulo de la pantalla: la cifra del modelo elegido y lo que dicen las pruebas frente a la EBM."""
    frase = _mayuscula(D.en_frase(modelo))
    if tarea == "clasificacion":
        d = _clasificacion_datos(modelo)
        f, op = d["fila"], d["op"]
        texto = (f"{frase}: AUC de prueba {ui.num(f['auc_prueba'], 4)} {F.intervalo(f['auc_bca_inf'], f['auc_bca_sup'], 4)}; "
                 f"contactando al {_pct_cap()} de mayor puntaje alcanza a {F.entero(op['vp'])} de los {F.entero(op['positivos'])} "
                 f"clientes de riesgo alto ({F.pct(op['captura'])}).")
    else:
        f = D.fila(D.prueba("regresion"), modelo=modelo)
        texto = (f"{frase}: RMSE de prueba {ui.num(f['rmse_prueba'], 5)} {F.intervalo(f['rmse_bca_inf'], f['rmse_bca_sup'], 5)} y "
                 f"R² {ui.num(f['r2_prueba'], 3)}: explica el {F.pct(f['r2_prueba'])} de la variación del puntaje.")
    return f"{texto} {_relacion(tarea, modelo)}".strip()


# --- Clasificación: las piezas --------------------------------------------------------------------------------------------
def _cifras_clasificacion(modelo):
    d = _clasificacion_datos(modelo)
    f, cal = d["fila"], d["cal"]
    ece0 = cal["ece_prueba_sin_recalibrar"]
    sin = f"Sin recalibrar: {ui.num(ece0, 3)}." if ece0 is not None else "Sin recalibrar no aplica: el puntaje es un margen."
    brier0 = f"Sin recalibrar: {ui.num(d['brier_crudo'], 3)}." if d["brier_crudo"] is not None else "Sin recalibrar no aplica."
    return ui.cifras([
        ui.cifra("AUC de prueba", ui.num(f["auc_prueba"], 4), variacion=f"IC 95 %: {F.intervalo(f['auc_bca_inf'], f['auc_bca_sup'], 4)}",
                 nota="Bootstrap BCa en los clientes de prueba.", mini=True),
        ui.cifra("AUC-PR (precisión media)", ui.num(f["auc_pr"], 3), variacion=f"Azar: {ui.num(d['prevalencia'], 3)}",
                 nota="El azar da la proporción de riesgo alto.", mini=True),
        ui.cifra("Error de calibración (ECE)", ui.num(cal["ece_prueba_recalibrada"], 3), variacion=sin,
                 nota="Con la probabilidad recalibrada.", mini=True),
        ui.cifra("Brier de la probabilidad recalibrada", ui.num(cal["brier_prueba_recalibrada"], 3), variacion=brier0,
                 nota="Error cuadrático medio de la probabilidad.", mini=True),
    ])


def _matriz(modelo):
    d = _clasificacion_datos(modelo)
    op, f = d["op"], d["fila"]
    frase = D.en_frase(modelo)
    titulo = (f"Contactando al {_pct_cap()} de mayor puntaje, {frase} alcanza a {F.entero(op['vp'])} de los "
              f"{F.entero(op['positivos'])} clientes de riesgo alto y se le escapan {F.entero(op['fn'])}")
    corte = "margen" if not d["en_01"] else "puntaje sin recalibrar"
    pares = [("Captura (recall)", F.pct(op["captura"])), ("Precisión", F.pct(op["precision"])),
             ("Elevación sobre el azar", f"{ui.num(op['elevacion'], 2)} veces"), ("MCC", ui.num(f["mcc_op"], 3)),
             ("Exactitud balanceada", ui.num(f["exactitud_bal_op"], 3)), (f"Corte ({corte} del último contactado)", ui.num(op["corte"], 4))]
    tabla = ui.tabla(["Celda", "Clientes"], [["Aciertos (contactados de riesgo alto)", F.entero(op["vp"])],
                                             ["Falsas alarmas (contactados de riesgo bajo)", F.entero(op["fp"])],
                                             ["Se escapan (riesgo alto sin contactar)", F.entero(op["fn"])],
                                             ["Bien descartados (riesgo bajo sin contactar)", F.entero(op["vn"])]],
                     compacta=True, total=["Clientes de prueba", F.entero(op["n"])])
    nota = (f"Las filas son lo que pasó (la etiqueta de la prueba) y las columnas lo que hizo la campaña, que contacta a "
            f"{F.entero(op['contactados'])} clientes; los empates de puntaje se rompen al azar con la semilla del proyecto, como "
            "en el informe. «Riesgo alto» es el cuarto superior del puntaje de propensión del conjunto de datos, no fuga observada.")
    return ui.tarjeta([ui.matriz(F.entero(op["vp"]), F.entero(op["fp"]), F.entero(op["fn"]), F.entero(op["vn"]), id=f"{PRE}-matriz"),
                       ui.ficha(pares), ui.nota(nota)],
                      titulo=titulo, subtitulo="Matriz de confusión en el punto de operación del informe.", detalles=tabla,
                      fuentes=[FUENTE_PUNTAJES, FUENTE_CLF])


def _roc(modelo):
    d = _clasificacion_datos(modelo)
    f, op = d["fila"], d["op"]
    rol, nombre, frase = D.ROL_MODELO[modelo], D.NOMBRE_MODELO[modelo], D.en_frase(modelo)
    fpr, tpr = d["roc"]["fpr"].to_numpy(), d["roc"]["tpr"].to_numpy()
    fpr_op, tpr_op = op["fp"] / (op["fp"] + op["vn"]), op["captura"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], hoverinfo="skip", **ui.traza("contexto")))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, **ui.traza(rol, _forma_linea(rol)), **ui.hover_texto(
        [f"Riesgo bajo marcado por error: {ui.num(a, 3)} · riesgo alto alcanzado: <b>{ui.num(b, 3)}</b>" for a, b in zip(fpr, tpr)])))
    fig.add_trace(go.Scatter(x=[fpr_op], y=[tpr_op], cliponaxis=False, **ui.traza(rol, "punto"), **ui.hover_texto(
        [f"Contactar al {_pct_cap()}: riesgo bajo marcado {ui.num(fpr_op, 3)} · alcanzado <b>{ui.num(tpr_op, 3)}</b>"])))
    ui.unidad_eje(fig, "fracción del riesgo alto alcanzada")
    _ejes_01(fig)
    titulo = (f"Entre un cliente de riesgo alto y uno de riesgo bajo tomados al azar, {frase} da mayor puntaje al primero en el "
              f"{F.pct(f['auc_prueba'])} de los pares")
    sub = (f"Curva ROC en los {F.entero(d['n'])} clientes de prueba: en horizontal, la fracción de clientes de riesgo bajo marcados "
           f"por error; en vertical, la de riesgo alto alcanzados. AUC {ui.num(f['auc_prueba'], 4)}, intervalo BCa del 95 % "
           f"{F.intervalo(f['auc_bca_inf'], f['auc_bca_sup'], 4)}.")
    nota = (f"El punto es la campaña del {_pct_cap()}: alcanza al {F.pct(tpr_op)} del riesgo alto y marca por error al "
            f"{F.pct(fpr_op)} del riesgo bajo. El AUC es esa probabilidad de ordenar bien un par (los empates cuentan como medio "
            "acierto); la diagonal es el azar.")
    if not d["en_01"]:
        nota += " El puntaje del SVM lineal es un margen, no una probabilidad: la curva solo usa el orden."
    filas = [[ui.num(fpr[k], 2), ui.num(tpr[k], 3)] for k in range(0, len(fpr), PASO_TABLA_CURVA)]
    tabla = _tabla(["Riesgo bajo marcado por error", "Riesgo alto alcanzado"], filas)
    leyenda = [(nombre, rol, _forma_linea(rol)), (f"Contactar al {_pct_cap()}", rol, "punto"), ("Azar", "contexto", "linea")]
    return _grafico(fig, titulo, sub, tabla, leyenda, nota, [FUENTE_PUNTAJES, FUENTE_CLF], f"{PRE}-roc")


def _precision_recall(modelo):
    d = _clasificacion_datos(modelo)
    f, op = d["fila"], d["op"]
    rol, nombre, frase = D.ROL_MODELO[modelo], D.NOMBRE_MODELO[modelo], D.en_frase(modelo)
    c = d["captura"]
    c = c[c["contactados"] > 0]
    x, y = c["captura"].to_numpy(), c["precision"].to_numpy()
    cap = c["capacidad"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, **ui.traza(rol, _forma_linea(rol)), **ui.hover_texto(
        [f"Contactando al {F.pct(k)}: captura {F.pct(a)} · precisión <b>{F.pct(b)}</b>" for k, a, b in zip(cap, x, y)])))
    fig.add_trace(go.Scatter(x=[op["captura"]], y=[op["precision"]], cliponaxis=False, **ui.traza(rol, "punto"), **ui.hover_texto(
        [f"Contactar al {_pct_cap()}: captura {F.pct(op['captura'])} · precisión <b>{F.pct(op['precision'])}</b>"])))
    ui.referencia(fig, y=d["prevalencia"])
    fig.add_annotation(x=0, xref="paper", y=d["prevalencia"], text=f"azar: {ui.num(d['prevalencia'], 3)}", xanchor="left",
                       yanchor="bottom", showarrow=False, font=ui.fuente_svg("tinta_2"))
    ui.unidad_eje(fig, "precisión entre los contactados")
    _ejes_01(fig)
    titulo = (f"Contactando al {_pct_cap()}, el {F.pct(op['precision'])} de los contactados es de riesgo alto: "
              f"{ui.num(op['elevacion'], 2)} veces el {F.pct(d['prevalencia'])} que daría el azar")
    sub = (f"Curva de precisión-recall en la prueba: en horizontal, la captura (la fracción del riesgo alto alcanzada); en vertical, "
           f"la precisión. Área bajo la curva (precisión media): {ui.num(f['auc_pr'], 3)}; el azar daría "
           f"{ui.num(d['prevalencia'], 3)}.")
    k = int(np.argmin(np.abs(cap - CAPACIDAD_CORTA)))
    nota = (f"Cada punto de la curva es una capacidad de campaña: contactar a menos clientes sube la precisión y baja la captura "
            f"(con el {F.pct(cap[k], 0)}, precisión {F.pct(y[k])} y captura {F.pct(x[k])}). La línea horizontal es la "
            "precisión de contactar al azar.")
    filas = [[F.pct(cap[j], 0), F.pct(x[j]), F.pct(y[j])] for j in range(len(cap)) if float(round(cap[j] / PASO_TABLA_PR, 6)).is_integer()]
    tabla = _tabla(["Capacidad", "Captura", "Precisión"], filas, alta=True)
    leyenda = [(nombre, rol, _forma_linea(rol)), (f"Contactar al {_pct_cap()}", rol, "punto"), ("Azar", "contexto", "linea")]
    return _grafico(fig, titulo, sub, tabla, leyenda, nota, [FUENTE_PUNTAJES, FUENTE_CLF], f"{PRE}-pr")


def _calibracion(modelo):
    d = _clasificacion_datos(modelo)
    cal, f = d["cal"], d["fila"]
    rol, frase = D.ROL_MODELO[modelo], D.en_frase(modelo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], hoverinfo="skip", **ui.traza("contexto")))
    series = [("Probabilidad recalibrada", d["tramos_cal"], _forma_linea(rol))]
    if d["en_01"]:
        series.insert(0, ("Puntaje sin recalibrar", d["tramos_crudo"], "punteada"))
    for rotulo, tramos, forma in series:
        textos = [f"{rotulo} · tramo {ui.num(t['tramo'] / TRAMOS_CALIBRACION, 1)}–{ui.num((t['tramo'] + 1) / TRAMOS_CALIBRACION, 1)}: "
                  f"dice {ui.num(t['dice'], 3)} · observado <b>{ui.num(t['observado'], 3)}</b> · {F.entero(t['n'])} clientes"
                  for t in tramos]
        fig.add_trace(go.Scatter(x=[t["dice"] for t in tramos], y=[t["observado"] for t in tramos], cliponaxis=False,
                                 **ui.traza(rol, forma, marcadores=True), **ui.hover_texto(textos)))
    ui.unidad_eje(fig, "fracción observada de riesgo alto")
    _ejes_01(fig)
    ece1 = ui.num(cal["ece_prueba_recalibrada"], 3)
    if d["en_01"]:
        titulo = (f"Sin recalibrar, el puntaje se aparta {ui.num(cal['ece_prueba_sin_recalibrar'], 3)} en promedio de lo observado; "
                  f"recalibrado, {ece1}")
        balanceo = D.NOMBRE_BALANCEO[f["balanceo"]]
        nota = (f"Con {balanceo}, el puntaje medio es {ui.num(d['media'], 3)} y la proporción de riesgo alto, "
                f"{ui.num(d['prevalencia'], 3)}. Para ordenar clientes sirve el puntaje; para informar una probabilidad, la "
                "recalibrada.")
    else:
        titulo = f"El margen del SVM lineal no es una probabilidad; recalibrado, se aparta {ece1} en promedio de lo observado"
        nota = (f"El SVM lineal da un margen (su distancia al hiperplano), que va de {ui.num(d['s'].min(), 2)} a "
                f"{ui.num(d['s'].max(), 2)}: no se puede leer como probabilidad y solo se dibuja la recalibrada. Para ordenar "
                "clientes sirve el margen.")
    sub = (f"Diagrama de calibración en {TRAMOS_CALIBRACION} tramos de igual ancho: en horizontal, lo que dice el modelo en "
           "promedio en cada tramo; en vertical, la fracción de esos clientes que es de riesgo alto. Sobre la diagonal, el modelo "
           "dice lo que pasa. ECE: diferencia media, ponderada por clientes. La recalibración es isotónica, ajustada con las "
           "predicciones fuera de pliegue del entrenamiento.")
    crudo = {t["tramo"]: t for t in (d["tramos_crudo"] or ())}
    filas = []
    for t in d["tramos_cal"]:
        c = crudo.get(t["tramo"])
        filas.append([f"{ui.num(t['tramo'] / TRAMOS_CALIBRACION, 1)}–{ui.num((t['tramo'] + 1) / TRAMOS_CALIBRACION, 1)}",
                      F.entero(t["n"]), ui.num(t["dice"], 3), ui.num(t["observado"], 3),
                      f"{ui.num(c['dice'], 3)} / {ui.num(c['observado'], 3)}" if c else F.RAYA])
    tabla = [_tabla(["Tramo de la probabilidad recalibrada", "Clientes", "Dice", "Observado", "Sin recalibrar en el mismo tramo: dice / observado"],
                     filas),
             ui.nota(f"Brier de la recalibrada: {ui.num(cal['brier_prueba_recalibrada'], 3)}"
                     + (f"; sin recalibrar: {ui.num(d['brier_crudo'], 3)}." if d["brier_crudo"] is not None else "."))]
    leyenda = ([("Puntaje sin recalibrar", rol, "punteada")] if d["en_01"] else []) + [
        ("Probabilidad recalibrada", rol, _forma_linea(rol)), ("Calibración perfecta", "contexto", "linea")]
    return _grafico(fig, titulo, sub, tabla, leyenda, nota, [FUENTE_PUNTAJES, FUENTE_CAL], f"{PRE}-cal")


def _clasificacion(modelo):
    d = _clasificacion_datos(modelo)
    f, cal, op = d["fila"], d["cal"], d["op"]
    revisar = _avisos_coincidencia([
        ("el AUC", d["auc"], f["auc_prueba"]), ("la matriz de confusión", sum(abs(op[k] - f[f"{k}_op"]) for k in ("vp", "fp", "fn", "vn")), 0),
        ("el ECE recalibrado", d["ece_cal"], cal["ece_prueba_recalibrada"]), ("el Brier", d["brier_cal"], cal["brier_prueba_recalibrada"]),
        ("el ECE sin recalibrar", d["ece_crudo"], cal["ece_prueba_sin_recalibrar"])])
    partes = [ui.seccion("Las cifras del modelo en la prueba", [_cifras_clasificacion(modelo)] + ([revisar] if revisar else []),
                         subtitulo=f"{D.NOMBRE_MODELO[modelo]}, con su mejor combinación ({D.NOMBRE_BALANCEO[f['balanceo']]}, "
                                   f"{D.NOMBRE_OPTIMIZADOR[f['optimizador']]}), medido una sola vez en los {F.entero(d['n'])} clientes "
                                   "de prueba.",
                         fuentes=[FUENTE_CLF, FUENTE_CAL])]
    if not d["en_01"]:
        partes.append(ui.aviso(f"El SVM lineal no da una probabilidad sino un margen (su distancia al hiperplano). La matriz, la curva "
                               "ROC y la de precisión-recall solo usan el orden de los clientes, así que valen igual; la calibración "
                               "se muestra solo para la probabilidad recalibrada.", "atencion", "Un margen, no una probabilidad"))
    partes.append(ui.seccion("Cómo acierta y cómo se equivoca", [
        ui.rejilla([_matriz(modelo), _roc(modelo)], "1-1"),
        ui.rejilla([_precision_recall(modelo), _calibracion(modelo)], "1-1"),
    ], subtitulo="«Riesgo alto» es el cuarto superior del puntaje de propensión del conjunto de datos, no fuga observada. Las "
                 "curvas usan el puntaje sin recalibrar, que es el que ordena a los clientes."))
    return partes


# --- Regresión: los números -------------------------------------------------------------------------------------------
def _acf(r, k):
    x = r - r.mean()
    d = float(x @ x)
    return np.array([float(x[:-j] @ x[j:]) / d for j in range(1, k + 1)])


def _ljung_box(acf, n, h):
    q = float(n * (n + 2) * np.sum(acf[:h] ** 2 / (n - np.arange(1, h + 1))))
    return q, float(stats.chi2.sf(q, h))


@lru_cache(maxsize=None)
def _orden_cohortes():
    """Las cohortes de edad de los clientes de prueba en el orden del archivo original (índice «fila»), y cuántos bloques
    contiguos forman: el archivo agrupa a los clientes por cohorte."""
    Q = D.pronosticos()
    cohorte = M.clientes()["cohorte"].reindex(Q.index).to_numpy(dtype=object)
    cambios = int(np.sum(cohorte[1:] != cohorte[:-1]))
    orden = []
    for c in cohorte:
        if c not in orden:
            orden.append(c)
    return cohorte, tuple(orden), cambios + 1


@lru_cache(maxsize=None)
def _regresion_datos(modelo):
    Q = D.pronosticos()
    y = Q["y"].to_numpy(dtype=float)
    p = Q[modelo].to_numpy(dtype=float)
    r = y - p
    n = len(y)
    orden = np.argsort(p, kind="mergesort")
    grupos = [orden[g] for g in np.array_split(np.arange(n), GRUPOS)]
    g = [{"pron": float(p[i].mean()), "obs": float(y[i].mean()), "res": float(r[i].mean()), "desv": float(r[i].std())} for i in grupos]
    muestra = np.sort(np.random.default_rng(M.SEMILLA).choice(n, size=min(MUESTRA_NUBE, n), replace=False))
    acf = _acf(r, RETARDOS_ACF)
    q, p_lb = _ljung_box(acf, n, RETARDOS_LB)
    cohorte, orden_c, bloques = _orden_cohortes()
    medias = {c: float(r[cohorte == c].mean()) for c in orden_c}
    sin_cohorte = r - np.array([medias[c] for c in cohorte])
    # active_products no es una entrada (la variable excluida): aquí solo sirve para leer el residuo
    productos = M.clientes()["active_products"].reindex(Q.index).to_numpy()
    niveles = sorted(set(productos.tolist()))
    por_nivel = {int(k): float(r[productos == k].mean()) for k in niveles}
    dentro = r - np.array([por_nivel[int(k)] for k in productos])
    return {"y": y, "p": p, "r": r, "n": n, "grupos": tuple(g), "muestra": muestra, "filas": Q.index.to_numpy(),
            "acf": acf, "banda": float(stats.norm.ppf(1 - (1 - NIVEL) / 2) / np.sqrt(n)), "q_lb": q, "p_lb": p_lb,
            "acf1_sin_cohorte": float(_acf(sin_cohorte, 1)[0]), "medias_cohorte": medias, "orden_cohortes": orden_c,
            "bloques": bloques, "rmse": float(np.sqrt(np.mean(r ** 2))), "mae": float(np.mean(np.abs(r))),
            "r2": float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)), "asimetria": float(stats.skew(r)),
            "por_productos": por_nivel, "desv_dentro_productos": float(dentro.std()),
            "franjas": bool(float(dentro.std()) <= FRANJA * float(np.min(np.abs(np.diff([por_nivel[int(k)] for k in niveles]))))),
            "parte_productos": float(1 - dentro.var() / r.var()), "fila": D.fila(D.prueba("regresion"), modelo=modelo)}


def _edad(nivel):
    return f"{nivel}{F.FIJO}años"


# --- Regresión: las piezas --------------------------------------------------------------------------------------------
def _cifras_regresion(modelo):
    f = _regresion_datos(modelo)["fila"]
    return ui.cifras([
        ui.cifra("RMSE de prueba", ui.num(f["rmse_prueba"], 5), variacion=f"IC 95 %: {F.intervalo(f['rmse_bca_inf'], f['rmse_bca_sup'], 5)}",
                 nota="Raíz del error cuadrático medio, en unidades del puntaje.", mini=True),
        ui.cifra("MAE", ui.num(f["mae_prueba"], 4), nota="Error absoluto medio: lo que se equivoca un cliente típico.", mini=True),
        ui.cifra("R²", ui.num(f["r2_prueba"], 3), variacion=f"IC 95 %: {F.intervalo(f['r2_bca_inf'], f['r2_bca_sup'], 3)}",
                 nota="Parte de la variación del puntaje que explica el pronóstico.", mini=True),
        ui.cifra("MAPE", F.pct(f["mape_prueba"]), nota="Error relativo medio.", mini=True),
    ])


def _observado(modelo):
    d = _regresion_datos(modelo)
    rol, nombre = D.ROL_MODELO[modelo], D.NOMBRE_MODELO[modelo]
    i = d["muestra"]
    x, y = d["p"][i], d["y"][i]
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    g = d["grupos"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], hoverinfo="skip", **ui.traza("contexto")))
    fig.add_trace(go.Scattergl(x=x, y=y, **_nube(rol), **_hover_pares("Pronóstico {0} · observado <b>{1}</b>",
                                                                      [ui.num(a, 3) for a in x], [ui.num(b, 3) for b in y])))
    fig.add_trace(go.Scatter(x=[t["pron"] for t in g], y=[t["obs"] for t in g], **ui.traza("real", "gruesa", marcadores=True),
                             **ui.hover_texto([f"Grupo {k + 1}: pronóstico medio {ui.num(t['pron'], 3)} · observado medio "
                                               f"<b>{ui.num(t['obs'], 3)}</b>" for k, t in enumerate(g)])))
    ui.unidad_eje(fig, "puntaje observado (churn_probability)")
    _eje(fig, "x", lo, hi, maximo_marcas=5, d=1)
    _eje(fig, "y", lo, hi, maximo_marcas=5, d=1)
    fig.update_xaxes(showgrid=True, gridcolor=ui.ROL["rejilla"])
    fig.update_layout(hovermode="closest")
    desvio = max(abs(t["obs"] - t["pron"]) for t in g)
    compresion = f"su desviación es {ui.num(d['p'].std(), 3)} y la del puntaje, {ui.num(d['y'].std(), 3)}"
    if desvio <= SIGUE * d["y"].std():
        titulo = f"Por grupos el pronóstico sigue a lo observado (se aparta como mucho {ui.num(desvio, 3)}), pero se comprime: {compresion}"
    else:
        titulo = f"Por grupos el pronóstico se aparta hasta {ui.num(desvio, 3)} de lo observado y se comprime: {compresion}"
    sub = (f"En horizontal, el pronóstico; en vertical, el puntaje observado. Los puntos son {F.entero(len(i))} de los "
           f"{F.entero(d['n'])} clientes de prueba, elegidos al azar con la semilla del proyecto; la línea oscura es el observado "
           f"medio de todos en {GRUPOS} grupos de igual tamaño ordenados por el pronóstico.")
    nota = (f"Sobre la diagonal, el pronóstico acierta. La nube es ancha en vertical: con un R² de {ui.num(d['r2'], 3)}, el pronóstico "
            f"deja sin explicar la mayor parte de la variación de cada cliente, aunque en promedio por grupo acierta."
            + (f" Las {len(d['por_productos'])} franjas son los niveles de active_products, la variable excluida (ver el histograma)."
               if d["franjas"] else ""))
    filas = [[str(k + 1), ui.num(t["pron"], 4), ui.num(t["obs"], 4), ui.con_signo(t["obs"] - t["pron"], 4)] for k, t in enumerate(g)]
    tabla = _tabla(["Grupo", "Pronóstico medio", "Observado medio", "Diferencia"], filas, alta=True)
    leyenda = [(f"Un cliente ({F.entero(len(i))} de la prueba)", rol, "punto"), ("Observado medio por grupo", "real", "gruesa"),
               ("Pronóstico perfecto", "contexto", "linea")]
    return _grafico(fig, titulo, sub, tabla, leyenda, nota, [FUENTE_PRONOSTICOS, FUENTE_REG], f"{PRE}-obs")


def _residuos(modelo):
    d = _regresion_datos(modelo)
    rol = D.ROL_MODELO[modelo]
    i = d["muestra"]
    x, r = d["p"][i], d["r"][i]
    g = d["grupos"]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=r, **_nube(rol), **_hover_pares("Pronóstico {0} · residuo <b>{1}</b>",
                                                                      [ui.num(a, 3) for a in x], [ui.con_signo(b, 3) for b in r])))
    fig.add_trace(go.Scatter(x=[t["pron"] for t in g], y=[t["res"] for t in g], **ui.traza("real", "gruesa", marcadores=True),
                             **ui.hover_texto([f"Grupo {k + 1}: residuo medio <b>{ui.con_signo(t['res'], 4)}</b> · desviación "
                                               f"{ui.num(t['desv'], 4)}" for k, t in enumerate(g)])))
    ui.linea_cero(fig)
    ui.unidad_eje(fig, "residuo: observado menos pronóstico")
    tope = float(np.abs(r).max()) * 1.02
    _eje(fig, "y", -tope, tope, signo=True)
    fig.update_xaxes(showgrid=True, gridcolor=ui.ROL["rejilla"])
    fig.update_layout(hovermode="closest")
    ui.formato_eje(fig, "x", 2)
    desv = [t["desv"] for t in g]
    medias = [t["res"] for t in g]
    pareja = max(desv) <= PAREJA * min(desv)
    titulo = ((f"La dispersión del residuo es pareja en todo el recorrido del pronóstico: su desviación va de {ui.num(min(desv), 3)} "
               f"a {ui.num(max(desv), 3)} en los {GRUPOS} grupos") if pareja else
              (f"La dispersión del residuo cambia con el pronóstico: su desviación va de {ui.num(min(desv), 3)} a "
               f"{ui.num(max(desv), 3)} en los {GRUPOS} grupos"))
    sub = (f"Residuo (puntaje observado menos pronóstico) frente al pronóstico, en la misma muestra de {F.entero(len(i))} clientes; la "
           f"línea oscura es el residuo medio de todos por grupo.")
    sesgo = max(medias, key=abs)
    parte = abs(sesgo) / float(d["r"].std())
    nota = (f"El residuo medio por grupo va de {ui.con_signo(min(medias), 4)} a {ui.con_signo(max(medias), 4)}; el mayor es el "
            f"{F.pct(parte, 0)} de la desviación del residuo: "
            + ("un sesgo pequeño frente a lo que se dispersa cada cliente." if parte <= SESGO_PEQUENO else
               "un sesgo visible, el modelo se queda corto o se pasa en algunos tramos del pronóstico.")
            + (" Las franjas horizontales de la nube son los niveles de active_products, la variable excluida (ver el histograma)."
               if d["franjas"] else ""))
    filas = [[str(k + 1), ui.num(t["pron"], 4), ui.con_signo(t["res"], 4), ui.num(t["desv"], 4)] for k, t in enumerate(g)]
    tabla = _tabla(["Grupo", "Pronóstico medio", "Residuo medio", "Desviación del residuo"], filas, alta=True)
    leyenda = [(f"Un cliente ({F.entero(len(i))} de la prueba)", rol, "punto"), ("Residuo medio por grupo", "real", "gruesa")]
    return _grafico(fig, titulo, sub, tabla, leyenda, nota, [FUENTE_PRONOSTICOS, FUENTE_CLIENTES], f"{PRE}-res")


def _autocorrelacion(modelo):
    d = _regresion_datos(modelo)
    rol, frase = D.ROL_MODELO[modelo], D.en_frase(modelo)
    acf, banda = d["acf"], d["banda"]
    retardos = list(range(1, RETARDOS_ACF + 1))
    fig = go.Figure()
    fig.add_hrect(y0=-banda, y1=banda, fillcolor=ui.ROL["franja"], line_width=0, layer="below")
    fig.add_trace(go.Bar(x=retardos, y=acf, **ui.traza(rol, "barra", ancho=ui.ancho_barra(RETARDOS_ACF)), **ui.hover_texto(
        [f"Retardo {k}: <b>{ui.con_signo(a, 3)}</b>" for k, a in zip(retardos, acf)])))
    ui.linea_cero(fig)
    ui.unidad_eje(fig, "autocorrelación del residuo")
    tope = max(float(np.abs(acf).max()), banda) * HOLGURA
    _eje(fig, "y", -tope, tope, signo=True)
    fig.update_xaxes(tickvals=[k for k in retardos if k % 5 == 0 or k == 1], ticktext=[str(k) for k in retardos if k % 5 == 0 or k == 1])
    fig.update_layout(hovermode="closest")
    fuera = int(np.sum(np.abs(acf) > banda))
    lb = f"Ljung-Box con {RETARDOS_LB} retardos, {F.p_valor(d['p_lb'])}"
    nivel = F.pct(NIVEL, 0)
    if d["p_lb"] >= D.ALFA:
        cuantos = ("Ningún retardo sale" if not fuera else f"Solo {fuera} de {RETARDOS_ACF} retardos "
                   + ("sale" if fuera == 1 else "salen"))
        titulo = f"{cuantos} de la banda del {nivel}: el residuo no depende del orden del archivo ({lb})"
    else:
        cuantos = f"Los {RETARDOS_ACF} retardos salen" if fuera == RETARDOS_ACF else f"{fuera} de los {RETARDOS_ACF} retardos salen"
        titulo = (f"{cuantos} de la banda del {nivel}: el residuo arrastra un sesgo que comparten los clientes vecinos en el archivo "
                  f"({lb})")
    cohortes = F.enumerar(_edad(c) for c in d["orden_cohortes"])
    agrupado = (f"El archivo original agrupa a los clientes en {d['bloques']} bloques contiguos de edad ({cohortes}): no es una serie "
                "de tiempo" if d["bloques"] == len(d["orden_cohortes"]) else
                f"Los clientes del archivo original forman {d['bloques']} tramos contiguos de edad: no es una serie de tiempo")
    sub = (f"Correlación del residuo de cada cliente de prueba con el de los que lo siguen en el archivo original "
           f"(customer_data.csv), de 1 a {RETARDOS_ACF} posiciones. {agrupado}. La banda es la del {nivel} si no hubiera "
           "autocorrelación.")
    medias = "; ".join(f"{_edad(c)}, {ui.con_signo(m, 4)}" for c, m in d["medias_cohorte"].items())
    if d["p_lb"] >= D.ALFA:
        nota = (f"Una autocorrelación alta diría que clientes vecinos en el archivo comparten un error, es decir, que {frase} deja sin "
                f"explicar algo de su cohorte. No pasa: el residuo medio por cohorte es casi cero ({medias}).")
    else:
        nota = (f"Como los vecinos en el archivo son de la misma cohorte, la autocorrelación dice que {frase} deja sin explicar algo de "
                f"la edad: el residuo medio por cohorte no es cero ({medias}). Al restar a cada residuo la media de su cohorte, la "
                f"autocorrelación del primer retardo baja de {ui.con_signo(acf[0], 3)} a {ui.con_signo(d['acf1_sin_cohorte'], 3)}.")
    filas = [[str(k), ui.con_signo(a, 4), "sí" if abs(a) > banda else "no"] for k, a in zip(retardos, acf)]
    tabla = [_tabla(["Retardo", "Autocorrelación", "Fuera de la banda"], filas, alta=True),
             ui.nota(f"Banda: ±{ui.num(banda, 4)}. Ljung-Box con {RETARDOS_LB} retardos: Q = {ui.num(d['q_lb'], 1)}, {F.p_valor(d['p_lb'])}.")]
    leyenda = [("Autocorrelación del residuo", rol, "barra"), (f"Banda del {nivel} sin autocorrelación", "franja", "barra")]
    return _grafico(fig, titulo, sub, tabla, leyenda, nota, [FUENTE_PRONOSTICOS, FUENTE_CLIENTES], f"{PRE}-acf")


def _histograma(modelo):
    d = _regresion_datos(modelo)
    rol = D.ROL_MODELO[modelo]
    r = d["r"]
    cuentas, bordes = np.histogram(r, bins=TRAMOS_HISTOGRAMA)
    centros, ancho = (bordes[:-1] + bordes[1:]) / 2, float(bordes[1] - bordes[0])
    mu, sd = float(r.mean()), float(r.std())
    xs = np.linspace(bordes[0], bordes[-1], PUNTOS_CURVA)
    normal = d["n"] * ancho * np.exp(-((xs - mu) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centros, y=cuentas, **ui.traza(rol, "barra", ancho=ancho * 0.85), **ui.hover_texto(
        [f"Residuo de {ui.con_signo(a, 3)} a {ui.con_signo(b, 3)}: <b>{F.entero(c)}</b> clientes"
         for a, b, c in zip(bordes[:-1], bordes[1:], cuentas)])))
    fig.add_trace(go.Scatter(x=xs, y=normal, hoverinfo="skip", **ui.traza("contexto", "linea")))
    ui.unidad_eje(fig, "clientes")
    vals, textos = ui.marcas(bordes[0], bordes[-1], ui.paso_redondo(bordes[-1] - bordes[0], 6), signo=True)
    fig.update_xaxes(tickvals=vals, ticktext=textos)
    _eje(fig, "y", 0, float(max(cuentas.max(), normal.max())) * 1.08)
    fig.update_layout(hovermode="closest", bargap=0)
    asim = d["asimetria"]
    cola = ("la cola larga es la de los clientes cuyo puntaje observado queda muy por debajo del pronóstico" if asim < 0 else
            "la cola larga es la de los clientes cuyo puntaje observado queda muy por encima del pronóstico")
    titulo = f"El residuo es asimétrico (asimetría {ui.num(asim, 2)}): {cola}"
    sub = (f"Distribución del residuo de los {F.entero(d['n'])} clientes de prueba en {TRAMOS_HISTOGRAMA} tramos. La curva gris es la "
           f"normal con la misma media ({ui.con_signo(mu, 4)}) y la misma desviación ({ui.num(sd, 4)}).")
    niveles = d["por_productos"]
    primero, ultimo = min(niveles), max(niveles)
    inicio = ("Los picos del histograma son los niveles de active_products" if d["franjas"] else
              "El residuo depende del número de productos activos (active_products)")
    nota = (f"{inicio}, la variable con la que se generó el puntaje y que se excluyó de las entradas: el residuo medio va de {ui.con_signo(niveles[primero], 3)} con {primero} "
            f"{'producto' if primero == 1 else 'productos'} a {ui.con_signo(niveles[ultimo], 3)} con {ultimo}, y dentro de cada nivel "
            f"su desviación es {ui.num(d['desv_dentro_productos'], 3)}. El número de productos da cuenta del "
            f"{F.pct(d['parte_productos'], 0)} de la varianza del residuo: es la parte del puntaje que ningún modelo puede recuperar "
            "sin esa variable.")
    filas = [[f"{ui.con_signo(a, 3)} a {ui.con_signo(b, 3)}", F.entero(c)] for a, b, c in zip(bordes[:-1], bordes[1:], cuentas)]
    tabla = _tabla(["Residuo", "Clientes"], filas, alta=True)
    leyenda = [("Residuo", rol, "barra"), ("Normal con la misma media y desviación", "contexto", "linea")]
    return _grafico(fig, titulo, sub, tabla, leyenda, nota, [FUENTE_PRONOSTICOS, FUENTE_CLIENTES], f"{PRE}-hist")


def _regresion(modelo):
    d = _regresion_datos(modelo)
    f = d["fila"]
    revisar = _avisos_coincidencia([("el RMSE", d["rmse"], f["rmse_prueba"]), ("el MAE", d["mae"], f["mae_prueba"]),
                                    ("el R²", d["r2"], f["r2_prueba"])])
    return [
        ui.seccion("Las cifras del modelo en la prueba", [_cifras_regresion(modelo)] + ([revisar] if revisar else []),
                   subtitulo=f"{D.NOMBRE_MODELO[modelo]}, con su mejor combinación ({D.NOMBRE_OPTIMIZADOR[f['optimizador']]}), medido "
                             f"una sola vez en los {F.entero(d['n'])} clientes de prueba. El objetivo es el puntaje de propensión del "
                             "conjunto de datos (churn_probability), no la fuga observada.",
                   fuentes=[FUENTE_REG]),
        ui.seccion("Cómo se equivoca", [
            ui.rejilla([_observado(modelo), _residuos(modelo)], "1-1"),
            ui.rejilla([_autocorrelacion(modelo), _histograma(modelo)], "1-1"),
        ], subtitulo="El residuo es el puntaje observado menos el pronóstico: positivo si el modelo se queda corto."),
    ]


# --- La pantalla ----------------------------------------------------------------------------------------------------------
def layout(tarea=None, modelo=None, **_):
    """`/modelos/diagnostico?tarea=regresion&modelo=lasso` abre con esa tarea y ese modelo. Cuando la dirección los pide, los
    controles no reponen el valor recordado en la sesión: manda la dirección."""
    t, m, pedida = _de_la_direccion(tarea, modelo)
    control_tarea = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", [{"label": D.NOMBRE_TAREA[x], "value": x} for x in TAREAS], t,
                                                       persistir=not pedida))
    control_modelo = ui.control(html.Span(_rotulo_modelo(m), id=f"{PRE}-rotulo-modelo", className="tabla__clave"),
                                _selector_modelo(t, m, not pedida))
    otras = html.Div([ui.boton("Ver las pruebas frente a la EBM", href=app.ruta_vista("comparacion") + f"?tarea={t}", icono="balanza"),
                      ui.boton("Ver qué variables usa", href=app.ruta_vista("interpretabilidad") + f"?tarea={t}&modelo={m}", icono="lupa")],
                     className="botonera", id=f"{PRE}-enlaces")
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), html.Span(id=f"{PRE}-respuesta"), [control_tarea, control_modelo], [otras],
                      pantalla=PANTALLA),
        html.Div(id=f"{PRE}-cuerpo", className="pantalla"),
    ])


@callback(Output(f"{PRE}-modelo", "options"), Output(f"{PRE}-modelo", "value"), Input(f"{PRE}-tarea", "value"),
          State(f"{PRE}-modelo", "value"))
def _por_tarea(tarea, modelo):
    """Deja en el selector los ocho modelos de la tarea; si el elegido no es de ella, pasa al propuesto."""
    t = _tarea(tarea)
    valido = _modelo(t, modelo)
    return _opciones(t), (no_update if valido == modelo else valido)


@callback(Output(f"{PRE}-respuesta", "children"), Output(f"{PRE}-cuerpo", "children"), Output(f"{PRE}-enlaces", "children"),
          Output(f"{PRE}-rotulo-modelo", "children"), Input(f"{PRE}-tarea", "value"), Input(f"{PRE}-modelo", "value"))
def _dibujar(tarea, modelo):
    """El diagnóstico del modelo elegido. Si el modelo no es de la tarea (acaba de cambiar la tarea), espera a que el
    selector pase al propuesto."""
    t = _tarea(tarea)
    if modelo is not None and modelo not in V.modelos(t):
        return no_update, no_update, no_update, no_update
    m = _modelo(t, modelo)
    enlaces = [ui.boton("Ver las pruebas frente a la EBM", href=app.ruta_vista("comparacion") + f"?tarea={t}", icono="balanza"),
               ui.boton("Ver qué variables usa", href=app.ruta_vista("interpretabilidad") + f"?tarea={t}&modelo={m}", icono="lupa")]
    try:
        cuerpo = _clasificacion(m) if t == "clasificacion" else _regresion(m)
        return _respuesta(t, m), cuerpo, enlaces, _rotulo_modelo(m)
    except Exception as e:  # noqa: BLE001  (la pantalla no se cae: lo dice en un aviso)
        return "", ui.aviso(f"No se pudo calcular el diagnóstico de {D.NOMBRE_MODELO.get(m, m)} ({type(e).__name__}: {e}).",
                            "critico", "Falló el cálculo"), enlaces, _rotulo_modelo(m)
