"""Robustez · ¿Depende la EBM de cómo se entrenó?

La robustez de la EBM, el benchmark interpretable (la del modelo original está en secciones/original.py). Siete secciones: las cinco hipótesis de la ablación con el veredicto del informe, cuánto cambia cada variante, los cuatro
optimizadores, la sensibilidad a los hiperparámetros, la calibración, el costo y la curva de aprendizaje. Toda cifra sale
de datos/ o del motor. Los veredictos de las hipótesis no se calculan: son los del informe final y se citan (VEREDICTO).
El nombre de la pestaña, el título del navegador y el orden salen de NAV, en app.py.
"""
import math
import re
from collections import Counter
from functools import lru_cache

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback

import datos_app as D
import formato as F
import motor as M
import ui


PRE = "ro"
PANTALLA = "robustez"
PREGUNTA = "¿Depende la EBM de cómo se entrenó?"        # la de app.VISTAS
RIVAL = "xgboost"                 # el modelo de árboles con el que el informe compara el costo de la EBM
UMBRAL_MEDIO = 0.5                # umbral de decisión con que la ablación mide la captura (recall)
BOLSAS_BAJA = 2                   # bolsas externas (B) de la búsqueda de baja fidelidad y de las curvas de sensibilidad
PLIEGUES_SENSIBILIDAD = 3         # pliegues de la validación cruzada de las curvas de sensibilidad
TRAMOS_CALIBRACION = 10           # tramos de igual ancho del diagrama de confiabilidad (los del ECE del informe)
MIN_CLIENTES_TRAMO = 20           # como la figura del informe: solo se dibujan los tramos con al menos estos clientes
N_REFERENCIA = 10_000             # tamaño desde el que el informe mide lo que aún gana la curva de aprendizaje
ARREPENTIMIENTO_MAX = 1.0         # el arrepentimiento normalizado va de 0 (lo mejor que encontró cualquiera) a 1 (sin mejorar)
ALTO_LIENZO = 420                 # alto del lienzo «alto» de ui.grafico, para el grosor de las barras horizontales
ESPACIO_DURO = F.FIJO             # entre una cifra y su unidad, para que no se separen al cambiar de línea
VERSIONES = ("sin recalibrar", "isotónica")          # nombres de las versiones en datos/entrega3/complementos/
NUMERO = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco"}
PUESTO_AREA = ["la menor", "la segunda menor", "la segunda mayor", "la mayor"]

# Veredictos de las cinco hipótesis de la ablación. No se calculan: son los que da el informe final y se citan tal cual
# (Informe Final/resultados.tex, «Resultados del estudio de ablación», puntos 1 a 5, y conclusiones.tex, segundo párrafo).
# Cada uno: palabra del chip, estado del chip y la frase del informe (sin las cifras que van entre paréntesis). «Refutada»
# va sin estado (chip neutro): que la semilla no pese es una buena noticia para la EBM, no una alarma.
CONFIRMADA, EN_PARTE, REFUTADA = "confirmada", "confirmada en parte", "refutada"
VEREDICTO = {
    "interacciones": (CONFIRMADA, "bien", "«Hipótesis confirmada: la aditividad basta.»"),
    "balanceo": (CONFIRMADA, "bien", "«Hipótesis confirmada: el balanceo elige el punto de operación.»"),
    "fidelidad": (CONFIRMADA, "bien", "«[…] la ganadora es la misma […]. Hipótesis confirmada.»"),
    "optimizador": (EN_PARTE, "atencion",
                    "«Hipótesis confirmada en su segunda parte (la eficiencia por evaluación distingue a los métodos) y solo en "
                    "parte en la primera: la métrica final no distingue a TPE, Grid y la genética, pero sí a Random Search en "
                    "clasificación.»"),
    "semilla": (REFUTADA, None,
                "«Hipótesis refutada para la EBM, más estable que Random Forest […] y XGBoost […], para los que sí se sostiene.»"),
}
CITA = "Informe final, sección «Resultados del estudio de ablación» y conclusiones."

# Lo que cambia entre las dos tareas en la ablación: la métrica, sus decimales y la escala de las diferencias (el informe
# da las de RMSE en milésimas)
TAREA = {
    "clasificacion": {"metrica": "AUC", "d": 4, "escala": 1, "d_delta": 4, "unidad": "de AUC", "rope": "auc", "prueba": "DeLong",
                      "eje": "pérdida de AUC", "externa": "AUC externo"},
    "regresion": {"metrica": "RMSE", "d": 5, "escala": 1000, "d_delta": 2, "unidad": "milésimas de RMSE", "rope": "rmse",
                  "prueba": "Diebold-Mariano", "eje": "RMSE añadido, en milésimas", "externa": "RMSE externo"},
}
OPCIONES_TAREA = [{"label": D.NOMBRE_TAREA[t], "value": t} for t in TAREA]

HIPER = {"learning_rate": ("Tasa de aprendizaje", "la tasa de aprendizaje"),
         "max_leaves": ("Hojas por árbol", "las hojas por árbol"),
         "min_samples_leaf": ("Mínimo por hoja", "el mínimo de clientes por hoja"),
         "interactions": ("Interacciones", "el número de interacciones")}

MEDIDA = {"ajuste_s": {"opcion": "Ajuste (s)", "eje": "segundos de ajuste", "unidad": "s"},
          "inferencia_s": {"opcion": "Inferencia (s)", "eje": "segundos de inferencia", "unidad": "s"},
          "tamano_mb": {"opcion": "Tamaño (MB)", "eje": "MB del modelo guardado", "unidad": "MB"}}

FUENTES_ABLACION = [D.fuente("resultados/ablacion.csv"), D.fuente("resultados/resumen.json")]
FUENTES_HIPOTESIS = [D.fuente("resultados/ablacion.csv"), D.fuente("resultados/ablacion_interacciones_e2.csv"),
                     D.fuente("resultados/fidelidad_resumen.csv"), D.fuente("resultados/semillas.csv")]
FUENTES_OPTIMIZADORES = [D.fuente("resultados/optimizadores.csv"), D.fuente("resultados/friedman_optimizadores.csv")]
FUENTES_CURVAS = [D.fuente("entrega3/curvas_anytime.csv"), D.fuente("resultados/optimizadores.csv"), D.fuente("entrega3/corridas.csv")]
FUENTES_SENSIBILIDAD = [D.fuente("resultados/sensibilidad_curvas.csv"), D.fuente("resultados/resumen.json"),
                        D.fuente("modelos/ebm_clasificacion.json")]
FUENTES_CALIBRACION = [D.fuente("entrega3/complementos/calibracion_prueba.parquet"), D.fuente("entrega3/complementos/calibracion.csv")]
FUENTES_COSTO = [D.fuente("resultados/costo.csv"), D.fuente("entrega3/computo/equipo.json")]
FUENTES_APRENDIZAJE = [D.fuente("resultados/curva_aprendizaje.csv"), D.fuente("resultados/clasificacion_prueba.csv")]


# --- Ayudas -------------------------------------------------------------------------------------------------------------
def _tarea(valor):
    return valor if valor in TAREA else "clasificacion"


def _frase(texto):
    """Primera letra en mayúscula, sin tocar el resto (las siglas se quedan como están)."""
    texto = str(texto)
    return texto[:1].upper() + texto[1:]


def _finito(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _p(p, d=3):
    """Valor p, o una raya si la prueba no se hizo para esa fila."""
    return F.p_valor(p, d) if _finito(p) else F.RAYA


def _decimales(v):
    """Decimales con que se escribe un número corto: 0,008 → 3 · 0,05 → 2 · 100 → 0."""
    texto = repr(float(v))
    return 0 if texto.endswith(".0") else len(texto.split(".")[1])


def _valor_hp(v):
    """Valor de un hiperparámetro con los decimales con que se escribió (0,008 · 0,05 · 100)."""
    return ui.num(v, _decimales(v))


def _hijos_tarjeta(*args, **kwargs):
    """Los hijos de una ui.tarjeta, para rellenar desde un callback una tarjeta que ya está en la pantalla (así su título
    puede ser un mensaje con la cifra del momento sin sacar la tarjeta de su rejilla)."""
    return ui.tarjeta(*args, **kwargs).children


def _repartir(valores, separacion, piso):
    """Alturas para rotular el final de varias líneas que terminan muy juntas: en el mismo orden que sus valores y
    separadas al menos `separacion` (en unidades del eje)."""
    alturas, previa = {}, piso - separacion
    for clave in sorted(valores, key=valores.get):
        previa = max(valores[clave], previa + separacion)
        alturas[clave] = previa
    return alturas


def _bigote(tabla, centro, inf, sup, escala=1):
    """Barra de error asimétrica (un intervalo) del color de la EBM."""
    return {"type": "data", "symmetric": False, "array": (tabla[sup] - tabla[centro]) * escala,
            "arrayminus": (tabla[centro] - tabla[inf]) * escala, "color": ui.ROL["modelo"], "thickness": ui.ANCHO_LINEA}


@lru_cache(maxsize=None)
def _ablacion():
    """resultados/ablacion.csv con el nombre corto de cada variante («SMOTE», «Random»). No modificar el resultado."""
    t = D.ablacion()
    t["nombre"] = [v.split(": ", 1)[1].split(" (")[0] if ": " in v else "" for v in t["variante"]]
    return t


def _completa(tarea):
    """El modelo completo de una tarea (la fila «completo» de la ablación)."""
    return D.fila(_ablacion(), tarea=tarea, componente="completo")


def _variantes(tarea, componente=None):
    """Las variantes de una tarea (en clasificación, también la de interacciones, que usa la etiqueta de la mediana)."""
    t = _ablacion()
    m = t["tarea"].str.startswith(tarea) & (t["componente"] != "completo")
    if componente is not None:
        m &= t["componente"] == componente
    return t[m].reset_index(drop=True)


def _pares_e2():
    """Número de pares de la EBM de la comparación con la entrega anterior, leído del nombre de la variante."""
    hallado = re.search(r"con (\d+) pares", D.fila(_ablacion(), componente="interacciones")["variante"])
    return int(hallado.group(1)) if hallado else None


def _bolsas():
    return int(M.meta()["ebm"]["clasificacion"]["hiperparametros"]["modelo__outer_bags"])


def _n_guia():
    """Corridas de la guía del curso: todas menos las de la EBM."""
    return int((D.tabla("entrega3/corridas.csv")["modelo"] != "ebm").sum())


def _presupuesto(tarea):
    """El presupuesto de las búsquedas de la guía en una tarea, leído de corridas.csv en los bloques con las cuatro búsquedas
    completas (los de la tabla de optimizadores): evaluaciones por búsqueda (mínimo y máximo según el modelo), pliegues
    externos, evaluaciones por corrida (una búsqueda por pliegue externo) y si los bloques cruzan modelo con balanceo."""
    c = D.tabla("entrega3/corridas.csv")
    c = c[(c["tarea"] == tarea) & (c["modelo"] != "ebm") & (c["estado"] == "ok")]
    c = c.groupby(["modelo", "balanceo"]).filter(lambda bloque: len(bloque) == len(D.ORDEN_OPTIMIZADORES))
    return {"minimo": int(c["presupuesto"].min()), "maximo": int(c["presupuesto"].max()), "pliegues": int(c["k_ext"].iloc[0]),
            "por_corrida": float(c["n_evaluaciones"].mean()), "bloques": int(c.groupby(["modelo", "balanceo"]).ngroups),
            "con_balanceo": c["balanceo"].nunique() > 1}


# --- 1. Las cinco hipótesis ---------------------------------------------------------------------------------------------
def _hipotesis():
    """Las cinco hipótesis de la ablación en el orden del informe: (clave, hipótesis, valor, unidad, variación, nota)."""
    cl = _variantes("clasificacion")
    completa = _completa("clasificacion")

    e2 = D.clave_valor("resultados/ablacion_interacciones_e2.csv")
    inter = D.fila(cl, componente="interacciones")
    h_inter = ("interacciones", _frase(inter["hipotesis"]), ui.num(e2["auc_ebm"] - e2["auc_gam"], 4), "de AUC",
               f"cuesta quitar las interacciones (DeLong, {F.p_valor(e2['p_delong'], 2)})",
               f"{ui.num(e2['auc_gam'], 4)} sin pares frente a {ui.num(e2['auc_ebm'], 4)} con {_pares_e2()} pares, con la "
               "etiqueta de la mediana (la de la entrega anterior).")

    bal = cl[cl["componente"] == "balanceo"].set_index("nombre")
    sin = bal.loc[D.NOMBRE_BALANCEO["ninguno"]]
    auc_con, auc_sin = ui.num(completa["metrica_prueba"], 4), ui.num(sin["metrica_prueba"], 4)
    iguales = f"{auc_con} en los dos casos" if auc_con == auc_sin else f"{auc_con} frente a {auc_sin}"
    veces = [ui.num(bal.loc[D.NOMBRE_BALANCEO[b], "corrida_min"] / completa["corrida_min"], 1) for b in ("smote", "adasyn")]
    h_bal = ("balanceo", _frase(sin["hipotesis"]), ui.num(-sin["delta_completo"], 4), "de AUC",
             f"cambia el AUC de prueba al quitar la ponderación ({iguales})",
             f"Mueve el punto de operación: la captura con umbral {ui.num(UMBRAL_MEDIO, 1)} (recall) pasa de "
             f"{ui.num(sin['recall_05'], 3)} a {ui.num(completa['recall_05'], 3)}. {D.NOMBRE_BALANCEO['smote']} y "
             f"{D.NOMBRE_BALANCEO['adasyn']} multiplican el tiempo de la corrida por {veces[0]} y {veces[1]}.")

    fi = D.clave_valor("resultados/fidelidad_resumen.csv")
    ganadora = "la misma ganadora" if str(fi["misma_ganadora"]) == "True" else "otra ganadora"
    h_fid = ("fidelidad", "La búsqueda de baja fidelidad conserva el orden", ui.num(fi["rho_spearman"], 2), "ρ de Spearman",
             f"{ganadora} y {F.pct(fi['ahorro_tiempo'], 0)} menos de tiempo de búsqueda",
             f"Correlación entre el AUC de {F.entero(fi['configuraciones'])} configuraciones aleatorias buscadas con "
             f"{BOLSAS_BAJA} bolsas y con las {_bolsas()} del modelo final.")

    opt = cl[cl["componente"] == "optimizador"].set_index("nombre")
    peor = opt.loc[opt["delta_completo"].idxmax()]
    resto = opt.drop(index=peor.name)
    del_modelo = D.NOMBRE_OPTIMIZADOR[D.fila(D.prueba("clasificacion"), modelo="ebm")["optimizador"]]
    h_opt = ("optimizador", _frase(peor["hipotesis"]), ui.num(peor["delta_completo"], 4), "de AUC",
             f"pierde {peor.name} frente a {del_modelo} en la prueba; no sobrevive a Holm ({F.p_valor(peor['p_prueba_holm'], 2)} ajustado)",
             f"{F.enumerar(resto.index)} quedan a {ui.num(resto['delta_completo'].max(), 4)} o menos. La eficiencia por "
             "evaluación sí distingue a los métodos.")

    se = D.tabla("resultados/semillas.csv")
    se = se[se["tarea"] == "clasificacion"].set_index("modelo")
    h_sem = ("semilla", "La semilla pesa tanto como las diferencias entre modelos",
             ui.num(se.loc["ebm", "auc_desv"], 5), "de AUC",
             f"desviación del AUC de prueba de la EBM entre {F.entero(se.loc['ebm', 'semillas'])} semillas",
             f"Refutada para la EBM, que es casi determinista; sí se sostiene para {D.en_frase('random_forest')} "
             f"({ui.num(se.loc['random_forest', 'auc_desv'], 4)}) y {D.en_frase('xgboost')} ({ui.num(se.loc['xgboost', 'auc_desv'], 4)}).")
    return [h_inter, h_bal, h_fid, h_opt, h_sem]


def _titulo_hipotesis():
    cuenta = Counter(palabra for palabra, _tipo, _cita in VEREDICTO.values())
    return (f"De las {NUMERO[len(VEREDICTO)]} hipótesis de la ablación, {NUMERO[cuenta[CONFIRMADA]]} se confirmaron, "
            f"{NUMERO[cuenta[EN_PARTE]]} solo en parte y {NUMERO[cuenta[REFUTADA]]} quedó refutada")


def _seccion_hipotesis():
    hip = _hipotesis()
    fichas = [ui.cifra(texto, valor, unidad, variacion=variacion, nota=nota, chip=ui.chip(*VEREDICTO[clave][:2]))
              for clave, texto, valor, unidad, variacion, nota in hip]
    citas = ui.tabla(["Hipótesis", "Veredicto", "Lo que dice el informe"],
                     [[texto, VEREDICTO[clave][0], VEREDICTO[clave][2]] for clave, texto, *_ in hip],
                     compacta=True, num=[False, False, False])
    return ui.seccion(_titulo_hipotesis(), [
        ui.cifras(fichas),
        ui.desplegable("Lo que dice el informe de cada hipótesis", [citas, ui.nota(CITA)]),
    ], subtitulo="La hipótesis, la cifra que la decide y el veredicto que le da el informe final.", fuentes=FUENTES_HIPOTESIS)


def _subtitulo():
    e2 = D.clave_valor("resultados/ablacion_interacciones_e2.csv")
    opt = _variantes("clasificacion", "optimizador")
    tope = math.ceil(opt["delta_completo"].max() * 1000) / 1000
    se = D.fila(D.tabla("resultados/semillas.csv"), tarea="clasificacion", modelo="ebm")
    return (f"Poco. Quitar las interacciones cuesta {ui.num(e2['auc_ebm'] - e2['auc_gam'], 4)} de AUC, el balanceo no cambia el "
            f"AUC y solo mueve el punto de operación, los {NUMERO[len(D.ORDEN_OPTIMIZADORES)]} optimizadores quedan a menos de "
            f"{ui.num(tope, 3)} y la semilla mueve el AUC {ui.num(se['auc_desv'], 5)}. Lo que sí cambia es el tiempo.")


# --- 2. Cuánto cambia cada variante ------------------------------------------------------------------------------------
def _rotulo(f):
    """Rótulo corto de una variante en el eje (el nombre completo va en la tabla y en la etiqueta flotante)."""
    if f["componente"] == "interacciones":
        return "Sin interacciones<br>(etiqueta: mediana)"
    if f["nombre"] == D.NOMBRE_BALANCEO["ninguno"]:
        return _frase(f["nombre"])
    return f"Con {f['nombre']}"


def _accion(f):
    if f["componente"] == "interacciones":
        return "quitar las interacciones"
    if f["componente"] == "balanceo":
        return "quitar el balanceo" if f["nombre"] == D.NOMBRE_BALANCEO["ninguno"] else f"balancear con {f['nombre']}"
    return f"buscar con {f['nombre']}"


def _variante_ablacion(tarea):
    """Figura, título, subtítulo, tabla, etiqueta y nota del gráfico de la ablación."""
    c = TAREA[tarea]
    k, dd = c["escala"], c["d_delta"]
    v, completa = _variantes(tarea), _completa(tarea)
    rope = D.resumen()["rope"][c["rope"]] * k
    rotulos = [_rotulo(f) for _, f in v.iterrows()]
    delta = [ui.con_signo(f["delta_completo"] * k, dd) for _, f in v.iterrows()]
    ic = [F.intervalo(f["delta_bca_inf"] * k, f["delta_bca_sup"] * k, dd, signo=True) for _, f in v.iterrows()]
    # El nombre completo de cada variante; la de interacciones dice su etiqueta (no se mezclan niveles de AUC sin rotular)
    nombres = [f["variante"] + (", con la etiqueta de la mediana" if f["componente"] == "interacciones" else "") for _, f in v.iterrows()]
    textos = [f"{_frase(n)}: <b>{d_}</b> {i} {c['unidad']}" for n, d_, i in zip(nombres, delta, ic)]

    fig = go.Figure(go.Scatter(x=v["delta_completo"] * k, y=rotulos, **ui.traza("modelo", "punto"),
                               error_x=_bigote(v, "delta_completo", "delta_bca_inf", "delta_bca_sup", k), **ui.hover_texto(textos)))
    tope = max(rope, float(v[["delta_bca_inf", "delta_bca_sup"]].abs().to_numpy().max()) * k)
    vals, marcas = ui.marcas(-tope, tope, ui.paso_redondo(tope, 1), signo=True)
    ui.franja(fig, -rope, rope)
    ui.linea_cero(fig, "x")
    ui.unidad_eje(fig, c["eje"])
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[-1.5 * tope, 1.5 * tope], tickvals=vals, ticktext=marcas, showgrid=True, gridcolor=ui.ROL["rejilla"],
                     showline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=rotulos[::-1], tickfont=ui.fuente_svg("tinta_2", "--t-s"))

    peor = v.loc[v["delta_completo"].idxmax()]
    dentro = bool(((v["delta_bca_inf"] * k > -rope) & (v["delta_bca_sup"] * k < rope)).all())
    banda = f"±{ui.num(rope, _decimales(rope))} {c['unidad']}"
    titulo = (f"Lo que más cuesta es {_accion(peor)}: {ui.num(peor['delta_completo'] * k, dd)} {c['unidad']}"
              + (", y ningún intervalo sale de la banda de equivalencia" if dentro else ""))
    cambia = F.enumerar(f"el {x}" if x != "interacciones" else f"las {x}" for x in v["componente"].unique())
    sub = (f"Diferencia de {c['metrica']} de prueba entre la EBM completa ({completa['variante'].split('(', 1)[1].rstrip(')')}) y "
           f"cada variante, que cambia un solo componente ({cambia.replace(' y ', ' o ')}), con su intervalo del 95{ESPACIO_DURO}%. A la derecha "
           f"del cero la variante pierde. La franja es la banda de equivalencia práctica ({banda}).")

    def fila(f, nombre, es_completa=False):
        externa = (f"{ui.num(f['metrica_ext_media'], c['d'])} ± {ui.num(f['metrica_ext_desv'], c['d'])}"
                   if _finito(f["metrica_ext_media"]) else F.RAYA)
        p_prueba = _p(f["p_prueba_holm"]) if _finito(f["p_prueba_holm"]) else (
            f"{_p(f['p_prueba'])} (sin ajustar)" if _finito(f["p_prueba"]) else F.RAYA)
        celdas = [nombre, externa, ui.num(f["metrica_prueba"], c["d"])]
        if tarea == "clasificacion":
            celdas.append(ui.num(f["recall_05"], 3))
        celdas += [ui.num(f["corrida_min"], 1),
                   F.RAYA if es_completa else ui.con_signo(f["delta_completo"] * k, dd),
                   F.RAYA if es_completa else F.intervalo(f["delta_bca_inf"] * k, f["delta_bca_sup"] * k, dd, signo=True),
                   _p(f["p_t_holm"]), p_prueba, ui.num(f["p_rope"], 3)]
        return celdas

    encabezados = ["Variante", c["externa"], f"{c['metrica']} de prueba"]
    if tarea == "clasificacion":
        encabezados.append(f"Captura con umbral {ui.num(UMBRAL_MEDIO, 1)} (recall)")
    encabezados += ["Corrida (min)", "Diferencia " + (c["unidad"] if k == 1 else f"en {c['unidad']}"), "Intervalo del 95 %",
                    "t corregida (p ajustado)", f"{c['prueba']} (p ajustado)", "Equivalencia práctica"]
    tabla = ui.tabla(encabezados, [fila(completa, completa["variante"], True)] + [fila(f, n) for (_, f), n in zip(v.iterrows(), nombres)],
                     compacta=True)
    etiqueta = f"{titulo}. " + "; ".join(f"{n}: {d_} {i}" for n, d_, i in zip(nombres, delta, ic)) + "."
    nota = ""
    if (v["componente"] == "interacciones").any():
        nota = (f"La variante sin interacciones se midió con la etiqueta de la mediana, la de la entrega anterior, frente a la EBM "
                f"con {_pares_e2()} pares: aquí se compara su diferencia, no su nivel.")
    return fig, titulo, sub, tabla, etiqueta, nota


def _titulo_ablacion():
    lejos = {tarea: _variantes(tarea)["delta_completo"].abs().max() * TAREA[tarea]["escala"] for tarea in TAREA}
    return ("Ninguna variante se aleja de la EBM completa más de "
            + " ni de ".join(f"{ui.num(lejos[tarea], c['d_delta'])} {c['unidad']}" for tarea, c in TAREA.items()))


# --- 3. Los cuatro optimizadores ---------------------------------------------------------------------------------------
def _menor_area():
    """El optimizador con menos área de arrepentimiento en cada tarea."""
    t = D.tabla("resultados/optimizadores.csv")
    return {tarea: g.loc[g["area_arrepentimiento"].idxmin(), "optimizador"] for tarea, g in t.groupby("tarea")}


def _titulo_optimizadores():
    ganadores = set(_menor_area().values())
    cuantos = NUMERO[len(D.ORDEN_OPTIMIZADORES)]
    if len(ganadores) == 1:
        return (f"Los {cuantos} optimizadores terminan cerca; {D.NOMBRE_OPTIMIZADOR[ganadores.pop()]} es el que menos "
                "arrepentimiento acumula en las dos tareas")
    return f"Los {cuantos} optimizadores terminan cerca; cambia lo que tardan en llegar"


def _curvas_arrepentimiento(tarea, foco):
    """Figura, título, subtítulo, leyenda, tabla y etiqueta de las curvas de arrepentimiento con un optimizador en foco."""
    cur = D.tabla("entrega3/curvas_anytime.csv")
    cur = cur[cur["tarea"] == tarea].sort_values("fraccion")
    series = {o: cur[cur["optimizador"] == o] for o in D.ORDEN_OPTIMIZADORES}
    nombre = D.NOMBRE_OPTIMIZADOR[foco]
    f = series[foco]
    x = (100 * f["fraccion"]).tolist()
    tope = ARREPENTIMIENTO_MAX

    fig = go.Figure()
    ui.banda(fig, x, f["q25"], f["q75"], rol="real")
    for o in D.ORDEN_OPTIMIZADORES:
        if o != foco:
            s = series[o]
            fig.add_trace(go.Scatter(x=(100 * s["fraccion"]).tolist(), y=s["media"], name=D.NOMBRE_OPTIMIZADOR[o],
                                     **ui.traza("contexto"), **ui.hover(D.NOMBRE_OPTIMIZADOR[o], s["media"], d=3)))
    fig.add_trace(go.Scatter(x=x, y=f["media"], name=nombre, **ui.traza("real"), **ui.hover(nombre, f["media"], d=3)))
    finales = {o: float(series[o]["media"].iloc[-1]) for o in D.ORDEN_OPTIMIZADORES}
    for o, altura in _repartir(finales, 0.075 * tope, 0.02 * tope).items():
        fig.add_annotation(x=1, xref="paper", xanchor="left", xshift=ui.px("--e-2"), y=altura, yanchor="middle",
                           text=D.NOMBRE_OPTIMIZADOR[o], showarrow=False, font=ui.fuente_svg("tinta" if o == foco else "tinta_3"))
    vals_y, marcas_y = ui.marcas(0, tope, tope / 4)
    vals_x, _ = ui.marcas(25, 100, 25)
    ui.unidad_eje(fig, "arrepentimiento normalizado")
    fig.update_layout(margin_r=ui.px("--e-8"))
    fig.update_xaxes(range=[0, 100.5], tickvals=vals_x, ticktext=[F.pct(v / 100, 0) for v in vals_x])
    fig.update_yaxes(range=[0, 1.04 * tope], tickvals=vals_y, ticktext=marcas_y)

    opt = D.tabla("resultados/optimizadores.csv")
    opt = opt[opt["tarea"] == tarea].set_index("optimizador")
    puesto = int(opt["area_arrepentimiento"].rank(method="min")[foco]) - 1
    titulo = (f"{nombre} tiene {PUESTO_AREA[puesto]} área de arrepentimiento de las {NUMERO[len(opt)]}: "
              f"{ui.num(opt.loc[foco, 'area_arrepentimiento'], 3)}")
    pre = _presupuesto(tarea)
    sub = (f"Arrepentimiento normalizado medio (1 = aún sin mejorar; 0 = ya halló lo mejor que encontró cualquiera) según la parte "
           f"del presupuesto de una búsqueda ya gastada, en las {F.entero(_n_guia())} corridas de la guía, sin la EBM. La banda va "
           f"del primer al tercer cuartil de {nombre}, que cierra el 95{ESPACIO_DURO}% de la brecha en "
           f"{ui.num(opt.loc[foco, 'evals_95'], 1)} de las {F.entero(pre['minimo'])} a {F.entero(pre['maximo'])} evaluaciones de una "
           "búsqueda.")
    leyenda = ui.leyenda([(f"{nombre}, en foco", "real", "gruesa"), (f"Entre los cuartiles de {nombre}", "real", "area"),
                          ("Los otros optimizadores, con su nombre al final", "contexto", "linea")])
    filas = [[F.pct(fr, 0)] + [ui.num(series[o]["media"].iloc[i], 3) for o in D.ORDEN_OPTIMIZADORES]
             + [ui.num(f["q25"].iloc[i], 3), ui.num(f["q75"].iloc[i], 3)] for i, fr in enumerate(f["fraccion"])]
    tabla = ui.tabla(["Presupuesto gastado"] + [D.NOMBRE_OPTIMIZADOR[o] for o in D.ORDEN_OPTIMIZADORES]
                     + [f"{nombre}: primer cuartil", f"{nombre}: tercer cuartil"], filas, compacta=True, alta=True)
    etiqueta = (f"{titulo}. Arrepentimiento al terminar el presupuesto: "
                + "; ".join(f"{D.NOMBRE_OPTIMIZADOR[o]}, {ui.num(finales[o], 3)}" for o in D.ORDEN_OPTIMIZADORES) + ".")
    return fig, titulo, sub, leyenda, tabla, etiqueta


def _ficha_optimizadores(tarea):
    """Los hijos de la tarjeta con la tabla de los optimizadores y lo que dice Friedman de la métrica final."""
    opt = D.tabla("resultados/optimizadores.csv")
    opt = opt[opt["tarea"] == tarea].set_index("optimizador").loc[D.ORDEN_OPTIMIZADORES]
    fr = D.fila(D.tabla("resultados/friedman_optimizadores.csv"), tarea=tarea)
    cuantos = NUMERO[len(opt)]
    pares = [(a, b, float(fr[f"nemenyi_{a}_{b}"])) for a in D.ORDEN_OPTIMIZADORES for b in D.ORDEN_OPTIMIZADORES
             if f"nemenyi_{a}_{b}" in fr.index and fr[f"nemenyi_{a}_{b}"] < D.ALFA]
    nota = None
    if fr["p"] < D.ALFA:
        comunes = set(D.ORDEN_OPTIMIZADORES)
        for a, b, _p_par in pares:
            comunes &= {a, b}
        quien = D.NOMBRE_OPTIMIZADOR[comunes.pop()] if len(comunes) == 1 else None
        titulo = (f"Por la métrica final solo se separa {quien} (Friedman, {F.p_valor(fr['p'])})" if quien
                  else f"Por la métrica final los {cuantos} no son equivalentes (Friedman, {F.p_valor(fr['p'])})")
        if pares:
            nota = ui.nota("Pares que separa Nemenyi: " + "; ".join(
                f"{D.NOMBRE_OPTIMIZADOR[a]} frente a {D.NOMBRE_OPTIMIZADOR[b]}, {F.p_valor(p_par)}" for a, b, p_par in pares) + ".")
    else:
        titulo = f"Por la métrica final Friedman no distingue a los {cuantos} ({F.p_valor(fr['p'])})"
    pre = _presupuesto(tarea)
    bloques = "bloques de modelo y balanceo" if pre["con_balanceo"] else "bloques de modelo"
    sub = (f"Rango por la métrica externa en los {F.entero(fr['bloques'])} {bloques} con las {cuantos} búsquedas completas de las "
           f"{F.entero(_n_guia())} corridas de la guía, sin la EBM. Todos con el mismo presupuesto por búsqueda (de "
           f"{F.entero(pre['minimo'])} a {F.entero(pre['maximo'])} evaluaciones según el modelo; unas {ui.num(pre['por_corrida'], 0)} "
           f"por corrida al sumar los {F.entero(pre['pliegues'])} pliegues externos). En el área y en las evaluaciones, menos es "
           "mejor.")
    medidas = [("Rango medio (1 = mejor)", "rango_medio", 2, min), ("Bloques ganados", "victorias", 0, max),
               ("Segundos por evaluación", "s_por_evaluacion", 2, min), ("Área de arrepentimiento", "area_arrepentimiento", 3, min),
               (f"Evaluaciones de una búsqueda hasta cerrar el 95{ESPACIO_DURO}% de la brecha", "evals_95", 1, min)]
    filas = [[rotulo] + [ui.num(opt.loc[o, col], d) for o in opt.index] for rotulo, col, d, _elegir in medidas]
    mejor = {(i, j) for i, (_rotulo, col, _d, elegir) in enumerate(medidas) for j, o in enumerate(opt.index, 1)
             if opt.loc[o, col] == elegir(opt[col])}
    tabla = ui.tabla(["Medida"] + [D.NOMBRE_OPTIMIZADOR[o] for o in opt.index], filas, mejor=mejor, compacta=True)
    detalle = ui.tabla(["Optimizador", "Bloques", "Métrica externa media", "Evaluaciones por corrida", "Minutos por corrida",
                        "Horas en total", "Optimismo de la selección"],
                       [[D.NOMBRE_OPTIMIZADOR[o], F.entero(opt.loc[o, "corridas"]), ui.num(opt.loc[o, "metrica_media"], 4),
                         ui.num(opt.loc[o, "evaluaciones"], 1), ui.num(opt.loc[o, "min_por_corrida"], 1),
                         ui.num(opt.loc[o, "horas_totales"], 1), ui.con_signo(opt.loc[o, "optimismo"], 5)] for o in opt.index],
                       compacta=True)
    return _hijos_tarjeta([tabla, nota], titulo=titulo, subtitulo=sub, detalles=detalle, fuentes=FUENTES_OPTIMIZADORES)


# --- 4. Sensibilidad a los hiperparámetros -----------------------------------------------------------------------------
def _curvas_sensibilidad():
    return D.tabla("resultados/sensibilidad_curvas.csv")


def _titulo_sensibilidad():
    t = _curvas_sensibilidad()
    return (f"La sensibilidad a los hiperparámetros es baja: el AUC se mueve entre {ui.num(t['auc_media'].min(), 4)} y "
            f"{ui.num(t['auc_media'].max(), 4)}")


def _sensibilidad(hp):
    """Figura, título, subtítulo, tabla y etiqueta del AUC al variar un hiperparámetro desde la configuración final."""
    t = _curvas_sensibilidad()
    s = t[t["hiperparametro"] == hp].sort_values("valor").reset_index(drop=True)
    final = M.meta()["ebm"]["clasificacion"]["hiperparametros"][f"modelo__{hp}"]
    i_final = int((s["valor"] - final).abs().idxmin())
    auc_final = float(s.loc[i_final, "auc_media"])
    rope = D.resumen()["rope"]["auc"]
    cats = [_valor_hp(v) for v in s["valor"]]
    aucs = [ui.num(a, 4) for a in s["auc_media"]]
    textos = [f"{HIPER[hp][0]} {c}: AUC <b>{a}</b> ± {ui.num(d, 4)}" for c, a, d in zip(cats, aucs, s["auc_desv"])]

    lugares = list(range(len(cats)))               # posiciones numéricas: «5» o «10» como categorías se leerían como números
    fig = go.Figure(go.Scatter(x=lugares, y=s["auc_media"], **ui.traza("modelo", "punto"),
                               error_y={"type": "data", "array": s["auc_desv"], "color": ui.ROL["modelo"], "thickness": ui.ANCHO_LINEA},
                               **ui.hover_texto(textos)))
    ui.banda_foco(fig, i_final)
    ui.referencia(fig, y=auc_final)
    ui.etiquetas_columnas(fig, lugares, aucs, foco=i_final, rotulos=["final" if i == i_final else "" for i in lugares])
    vals, marcas = ui.marcas(auc_final - rope, auc_final + rope, rope / 2)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[-0.5, len(cats) - 0.5], tickvals=lugares, ticktext=cats)
    fig.update_yaxes(range=[auc_final - rope, auc_final + rope], tickvals=vals, ticktext=marcas)

    lo, hi = float(s["auc_media"].min()), float(s["auc_media"].max())
    titulo = f"Cambiar {HIPER[hp][1]} mueve el AUC {ui.num(hi - lo, 4)}: de {ui.num(lo, 4)} a {ui.num(hi, 4)}"
    sub = (f"AUC medio ± desviación en una validación cruzada de {PLIEGUES_SENSIBILIDAD} pliegues sobre el entrenamiento, con "
           f"{BOLSAS_BAJA} bolsas, al variar solo este hiperparámetro desde la configuración final (columna resaltada). El eje "
           f"abarca la banda de equivalencia práctica: ±{ui.num(rope, 2)} de AUC alrededor de esa configuración.")
    tabla = ui.tabla([HIPER[hp][0], "AUC medio", "Desviación", "Diferencia con la configuración final"],
                     [[c + (" (final)" if i == i_final else ""), a, ui.num(d, 4), ui.con_signo(m - auc_final, 4)]
                      for i, (c, a, d, m) in enumerate(zip(cats, aucs, s["auc_desv"], s["auc_media"]))], compacta=True)
    etiqueta = f"{titulo}. " + "; ".join(f"{c}: {a}" for c, a in zip(cats, aucs)) + "."
    return fig, titulo, sub, tabla, etiqueta


def _fichas_sensibilidad():
    con = D.fila(D.tabla("resultados/concordancia_validacion_prueba.csv"), tarea="clasificacion")
    concordancia = ui.cifra("Correlación entre el AUC de la validación anidada y el de prueba", ui.num(con["correlacion"], 3),
                            f"en {F.entero(con['corridas'])} corridas",
                            nota="La validación anidada predice la prueba casi sin sesgo: la diferencia media es "
                                 f"{ui.con_signo(con['diferencia_media'], 4)}.")
    hp = D.tabla("resultados/sensibilidad_hiperparametros.csv")
    pliegues = _completa("clasificacion")["metrica_ext_desv"]
    corridas = ui.cifra(f"AUC externo de las {F.entero(len(hp))} corridas de clasificación de la EBM",
                        f"{ui.num(hp['auc_ext'].min(), 4)} a {ui.num(hp['auc_ext'].max(), 4)}",
                        nota=f"Terminan con configuraciones muy distintas (de {F.entero(hp['interactions'].min())} a "
                             f"{F.entero(hp['interactions'].max())} interacciones, tasa de aprendizaje de "
                             f"{ui.num(hp['learning_rate'].min(), 3)} a {ui.num(hp['learning_rate'].max(), 3)}) y su rango es "
                             f"menor que la desviación entre pliegues ({ui.num(pliegues, 4)}).")
    return [concordancia, corridas]


# --- 5. Calibración ----------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _confiabilidad():
    """Diagrama de confiabilidad de la EBM en la prueba, sin recalibrar y con la isotónica: por versión, los tramos de
    igual ancho (desde, hasta, n, media, observada), el ECE que resulta y cuántos valores distintos tiene."""
    p = D.compartida("entrega3/complementos/calibracion_prueba.parquet")
    f = D.fila(D.prueba("clasificacion"), modelo="ebm")
    clave = f"clasificacion__ebm__{f['balanceo']}__{f['optimizador']}"
    y = p["y_bin"].to_numpy()
    bordes = np.linspace(0, 1, TRAMOS_CALIBRACION + 1)
    salida = {}
    for version in VERSIONES:
        s = p[f"{clave}__{version}"].to_numpy()
        tramo = np.clip(np.searchsorted(bordes, s, side="right") - 1, 0, TRAMOS_CALIBRACION - 1)
        filas = [{"desde": bordes[b], "hasta": bordes[b + 1], "n": int((tramo == b).sum()), "media": float(s[tramo == b].mean()),
                  "observada": float(y[tramo == b].mean())} for b in range(TRAMOS_CALIBRACION) if (tramo == b).any()]
        t = pd.DataFrame(filas)
        salida[version] = {"tramos": t, "ece": float((t["n"] / len(y) * (t["media"] - t["observada"]).abs()).sum()),
                           "distintos": int(np.unique(s).size), "n": int(len(y))}
    return salida


def _calibracion_ebm():
    """Las filas de la EBM de calibracion.csv, por versión."""
    t = D.tabla("entrega3/complementos/calibracion.csv")
    return t[t["modelo"] == "ebm"].set_index("version")


def _titulo_calibracion():
    cal = _calibracion_ebm()
    return ("El puntaje sin recalibrar sobreestima; la recalibración isotónica baja el error de calibración de "
            f"{ui.num(cal.loc[VERSIONES[0], 'ece'], 3)} a {ui.num(cal.loc[VERSIONES[1], 'ece'], 3)}")


def _grafico_calibracion():
    con, cal = _confiabilidad(), _calibracion_ebm()
    nombres = {VERSIONES[0]: "Puntaje sin recalibrar", VERSIONES[1]: "Probabilidad recalibrada (isotónica)"}
    formas = {VERSIONES[0]: "linea", VERSIONES[1]: "trazos"}
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], **ui.traza("contexto"), hoverinfo="skip"))
    filas = []
    for version in VERSIONES:
        t = con[version]["tramos"]
        t = t[t["n"] >= MIN_CLIENTES_TRAMO]
        textos = [f"{nombres[version]} · tramo de {ui.num(f['desde'], 1)} a {ui.num(f['hasta'], 1)}<br>Media del tramo: "
                  f"<b>{ui.num(f['media'], 3)}</b> · riesgo alto observado: <b>{F.pct(f['observada'])}</b> · "
                  f"{F.entero(f['n'])} clientes" for _, f in t.iterrows()]
        fig.add_trace(go.Scatter(x=t["media"], y=t["observada"], **ui.traza("modelo", formas[version], marcadores=True),
                                 **ui.hover_texto(textos)))
        filas += [[nombres[version], f"{ui.num(f['desde'], 1)} a {ui.num(f['hasta'], 1)}", F.entero(f["n"]), ui.num(f["media"], 3),
                   F.pct(f["observada"])] for _, f in t.iterrows()]
    vals, marcas = ui.marcas(0, 1, 0.25)
    fig.add_annotation(x=1, y=1, text="calibración perfecta", xanchor="right", yanchor="bottom", showarrow=False,
                       font=ui.fuente_svg("tinta_3"))
    ui.unidad_eje(fig, "fracción observada de riesgo alto")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[-0.02, 1.02], tickvals=vals, ticktext=marcas, showgrid=True, gridcolor=ui.ROL["rejilla"])
    fig.update_yaxes(range=[-0.04, 1.1], tickvals=vals, ticktext=marcas)

    crudo = con[VERSIONES[0]]["tramos"]
    crudo = crudo[crudo["n"] >= MIN_CLIENTES_TRAMO].iloc[-1]
    sin = cal.loc[VERSIONES[0]]
    titulo = (f"Donde el puntaje sin recalibrar promedia {ui.num(crudo['media'], 2)}, solo el {F.pct(crudo['observada'], 0)} de los "
              "clientes es de riesgo alto")
    sub = (f"Diagrama de confiabilidad de la EBM en los {F.entero(con[VERSIONES[0]]['n'])} clientes de prueba, en "
           f"{TRAMOS_CALIBRACION} tramos de igual ancho (los que tienen al menos {MIN_CLIENTES_TRAMO} clientes). En horizontal, lo "
           "que dice el modelo en promedio; en vertical, lo que se observa. «Riesgo alto» es el cuarto superior del puntaje de "
           "propensión del conjunto de datos, no fuga observada.")
    nota = (f"Con ponderación de clases el puntaje medio es {ui.num(sin['prob_media'], 3)} y la prevalencia, "
            f"{ui.num(sin['prevalencia'], 3)}: sirve para ordenar a los clientes, no para leerlo como una probabilidad.")
    etiqueta = (f"{titulo}. Con la recalibración isotónica los tramos quedan sobre la diagonal: el error de calibración esperado "
                f"baja de {ui.num(sin['ece'], 3)} a {ui.num(cal.loc[VERSIONES[1], 'ece'], 3)}.")
    return ui.grafico(fig, titulo, sub, etiqueta,
                      tabla=(["Versión", "Tramo", "Clientes", "Media del tramo", "Riesgo alto observado"], filas),
                      leyenda=[(nombres[VERSIONES[0]], "modelo", "linea"), (nombres[VERSIONES[1]], "modelo", "trazos"),
                               ("Calibración perfecta", "contexto", "linea")],
                      nota=nota, fuentes=FUENTES_CALIBRACION, id=f"{PRE}-cal")


def _fichas_calibracion():
    con, cal = _confiabilidad(), _calibracion_ebm()
    sin, iso = cal.loc[VERSIONES[0]], cal.loc[VERSIONES[1]]
    cambio_ece, cambio_auc = iso["ece"] - sin["ece"], iso["auc"] - sin["auc"]
    ece = ui.antes_despues("Error de calibración esperado (ECE) en la prueba", ui.num(sin["ece"], 3), ui.num(iso["ece"], 3),
                           cambio=ui.chip_cambio(cambio_ece, ui.con_signo(cambio_ece, 3), "bien"),
                           nota="Distancia media entre lo que dice el modelo y lo que se observa, tramo a tramo. La isotónica "
                                "se ajusta con predicciones fuera de pliegue del entrenamiento.",
                           hoy_rotulo="Sin recalibrar", plan_rotulo="Isotónica")
    prueba = D.fila(D.prueba("clasificacion"), modelo="ebm")
    auc = ui.antes_despues("AUC de prueba de la EBM", ui.num(sin["auc"], 4), ui.num(iso["auc"], 4),
                           cambio=ui.chip_cambio(cambio_auc, ui.con_signo(cambio_auc, 4)),
                           nota=f"El cambio cabe en el intervalo del 95{ESPACIO_DURO}% del AUC sin recalibrar "
                                f"({F.intervalo(prueba['auc_bca_inf'], prueba['auc_bca_sup'], 4)}), pero la isotónica deja solo "
                                f"{F.entero(con[VERSIONES[1]]['distintos'])} valores distintos y crea empates: se usa para informar "
                                "una probabilidad, no para ordenar.",
                           hoy_rotulo="Sin recalibrar", plan_rotulo="Isotónica")
    return ui.cambios([ece, auc])


# --- 6. Cuánto cuesta --------------------------------------------------------------------------------------------------
def _medida(v, medida):
    """Una medida de costo con los decimales que pide su tamaño: 26,7 s · 0,28 s · 0,067 s · 44,0 MB · 0,010 MB."""
    if medida == "inferencia_s":
        return ui.num(v, 3)
    return ui.num(v, 1) if v >= 1 else ui.num(v, 2 if medida == "ajuste_s" else 3)


def _con_unidad(v, medida):
    """La medida con su unidad, unidas con un espacio duro para que no se separen al cambiar de línea."""
    return f"{_medida(v, medida)}{ESPACIO_DURO}{MEDIDA[medida]['unidad']}"


def _tabla_costo(tarea):
    t = D.tabla("resultados/costo.csv")
    return D.ordenar_modelos(t[t["tarea"] == tarea], tarea).set_index("modelo")


def _veces_ajuste(tarea):
    t = _tabla_costo(tarea)
    return t.loc["ebm", "ajuste_s"] / t.loc[RIVAL, "ajuste_s"]


def _titulo_costo():
    tablas = {tarea: _tabla_costo(tarea) for tarea in TAREA}
    siempre = all(t["ajuste_s"].idxmax() == "ebm" for t in tablas.values())
    lento = "es el modelo más lento de ajustar" if siempre else "es de los más lentos de ajustar"
    tiempos = " y ".join(f"{_con_unidad(t.loc['ebm', 'ajuste_s'], 'ajuste_s')} en {D.NOMBRE_TAREA[tarea].lower()}" for tarea, t in tablas.items())
    return f"Lo que sí cambia es el tiempo: la EBM {lento} ({tiempos})"


def _costo(medida, tarea):
    """Figura, título, subtítulo, tabla y etiqueta del costo de cada modelo."""
    t = _tabla_costo(tarea)
    meta = M.meta()
    orden = t[medida].sort_values(ascending=False).index.tolist()
    nombres = [D.NOMBRE_MODELO[m] for m in orden]
    unidad = MEDIDA[medida]["unidad"]
    ancho = ui.ancho_barra(len(orden), ALTO_LIENZO)
    fig = go.Figure()
    for rol, forma in (("modelo", "barra"), ("rival", "barra"), ("base", "rayada")):      # gris rayado y rosa sólido
        ms = [m for m in orden if D.ROL_VISTA_EBM[m] == rol]
        if not ms:
            continue
        valores = [t.loc[m, medida] for m in ms]
        textos = [_medida(v, medida) for v in valores]
        fig.add_trace(go.Bar(y=[D.NOMBRE_MODELO[m] for m in ms], x=valores, orientation="h", text=textos,
                             **ui.traza(rol, forma, ancho=ancho),
                             **ui.hover_texto([f"{D.NOMBRE_MODELO[m]}: <b>{x}</b> {unidad}" for m, x in zip(ms, textos)])))
    ui.unidad_eje(fig, MEDIDA[medida]["eje"])
    ui.formato_eje(fig, "x", 0)
    fig.update_layout(hovermode="closest", barmode="overlay", margin_r=ui.MARGEN_ETIQUETAS)
    fig.update_xaxes(range=[0, float(t[medida].max()) * 1.02], showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=nombres[::-1], tickfont=ui.fuente_svg("tinta_2", "--t-s"),
                     ticklabelstandoff=ui.px("--e-2"))

    e, x = t.loc["ebm"], t.loc[RIVAL]
    rival = D.en_frase(RIVAL)
    mayor = t.loc[t[medida].idxmax()]
    if medida == "ajuste_s":
        titulo = (f"La EBM tarda {_con_unidad(e['ajuste_s'], medida)} en ajustarse, {ui.num(_veces_ajuste(tarea), 0)} veces lo que "
                  f"{rival}, y predice igual de rápido ({_con_unidad(e['inferencia_s'], 'inferencia_s')})")
        que = f"Segundos que tarda un reajuste del modelo final con los {F.entero(meta['n_entrenamiento'])} clientes de entrenamiento"
    elif medida == "inferencia_s":
        titulo = (f"La EBM puntúa a los {F.entero(meta['n_prueba'])} clientes de prueba en {_con_unidad(e[medida], medida)}, como {rival} "
                  f"({_con_unidad(x[medida], medida)}); {D.en_frase(mayor.name)} tarda {_con_unidad(mayor[medida], medida)}")
        que = f"Segundos que tarda cada modelo final en puntuar a los {F.entero(meta['n_prueba'])} clientes de prueba"
    else:
        titulo = (f"La EBM ocupa {_con_unidad(e[medida], medida)}; el mayor, {D.en_frase(mayor.name)}, "
                  f"{_con_unidad(mayor[medida], medida)}")
        que = "Tamaño de cada modelo final guardado, en MB"
    sub = f"{que}, en {D.NOMBRE_TAREA[tarea].lower()}. Cada barra lleva su valor; el eje es lineal y empieza en cero."
    tabla = ui.tabla(["Modelo", "Ajuste (s)", "Inferencia (s)", "Corrida completa (min)", "Tamaño (MB)", "Valores guardados",
                      "Memoria pico del ajuste (MB)"],
                     [[D.NOMBRE_MODELO[m], _medida(t.loc[m, "ajuste_s"], "ajuste_s"), _medida(t.loc[m, "inferencia_s"], "inferencia_s"),
                       ui.num(t.loc[m, "corrida_min"], 1), _medida(t.loc[m, "tamano_mb"], "tamano_mb"), F.entero(t.loc[m, "parametros"]),
                       ui.num(t.loc[m, "memoria_mb"], 1)] for m in orden],
                     compacta=True, muestras=[D.ROL_VISTA_EBM[m] for m in orden])
    etiqueta = f"{titulo}. " + "; ".join(f"{D.NOMBRE_MODELO[m]}: {_medida(t.loc[m, medida], medida)} {unidad}" for m in orden) + "."
    return fig, titulo, sub, tabla, etiqueta


def _fichas_costo(tarea):
    """Las tres medidas de la EBM con lo que las acota, para leer la sección sin mover el control."""
    t = _tabla_costo(tarea)
    e, x = t.loc["ebm"], t.loc[RIVAL]
    rival = D.en_frase(RIVAL)
    lento, grande = t.loc[t["inferencia_s"].idxmax()], t.loc[t["tamano_mb"].idxmax()]
    segundo = t["ajuste_s"].drop(index="ebm").idxmax()
    return [
        ui.cifra("Ajuste de la EBM", _medida(e["ajuste_s"], "ajuste_s"), "s",
                 variacion=f"{ui.num(_veces_ajuste(tarea), 0)} veces lo que tarda {rival} ({_con_unidad(x['ajuste_s'], 'ajuste_s')})",
                 nota=f"El siguiente más lento es {D.en_frase(segundo)} ({_con_unidad(t.loc[segundo, 'ajuste_s'], 'ajuste_s')}).",
                 mini=True),
        ui.cifra("Inferencia de la EBM", _medida(e["inferencia_s"], "inferencia_s"), "s",
                 variacion=f"{rival}: {_con_unidad(x['inferencia_s'], 'inferencia_s')}, para los {F.entero(M.meta()['n_prueba'])} "
                           "clientes de prueba",
                 nota=f"El más lento es {D.en_frase(lento.name)} ({_con_unidad(lento['inferencia_s'], 'inferencia_s')}).", mini=True),
        ui.cifra("Tamaño de la EBM", _medida(e["tamano_mb"], "tamano_mb"), "MB",
                 variacion=f"guarda {F.entero(e['parametros'])} valores",
                 nota=f"El mayor es {D.en_frase(grande.name)}: {_con_unidad(grande['tamano_mb'], 'tamano_mb')} y "
                      f"{F.entero(grande['parametros'])} valores.", mini=True),
    ]


def _aviso_costo():
    veces = sorted(ui.num(_veces_ajuste(tarea), 0) for tarea in TAREA)
    equipo = D.json("entrega3/computo/equipo.json")
    return ui.aviso([f"La EBM no mejora la métrica de {D.en_frase(RIVAL)} y su ajuste cuesta entre {veces[0]} y {veces[-1]} veces "
                     "más. Pero el ajuste ocurre una vez y la consulta es igual de barata: lo que compra el costo extra es un "
                     "modelo que se puede leer sin un explicador.",
                     f"Los tiempos son de un solo equipo ({F.entero(equipo['nucleos_fisicos'])} núcleos y "
                     f"{ui.num(equipo['memoria_gb'], 1)} GB de memoria)."],
                    "canje", titulo="El canje: más tiempo de ajuste a cambio de un modelo legible")


# --- 7. Cuántos datos hacen falta --------------------------------------------------------------------------------------
def _curva_aprendizaje():
    return D.tabla("resultados/curva_aprendizaje.csv").pivot(index="modelo", columns="n", values="auc")


def _titulo_aprendizaje():
    t = _curva_aprendizaje()
    menor, todos = t.columns.min(), t.columns.max()
    if t.loc["ebm", menor] > t.loc["logistica", todos]:
        return (f"Con {F.entero(menor)} clientes la EBM ya supera a {D.en_frase('logistica')} entrenada con "
                f"los {F.entero(todos)}")
    return f"La EBM gana {ui.num(t.loc['ebm', todos] - t.loc['ebm', menor], 4)} de AUC entre {F.entero(menor)} y {F.entero(todos)} clientes"


def _aprendizaje(n):
    """Figura, título, tabla y etiqueta de la curva de aprendizaje con un tamaño de entrenamiento en foco."""
    t = _curva_aprendizaje()
    ns = list(t.columns)
    n = n if n in ns else ns[0]
    cats = [F.entero(v) for v in ns]
    orden = [m for m in D.ORDEN_MODELOS["clasificacion"] if m in t.index]
    fig = go.Figure()
    for m in orden[::-1]:                                              # la EBM, al final: queda encima
        fig.add_trace(go.Scatter(x=cats, y=t.loc[m], name=D.NOMBRE_MODELO[m], **ui.traza(D.ROL_VISTA_EBM[m], marcadores=(m == "ebm")),
                                 **ui.hover(D.NOMBRE_MODELO[m], t.loc[m], d=4)))
    ui.banda_foco(fig, F.entero(n), cats)
    paso = 0.05
    lo, hi = math.floor(t.to_numpy().min() / paso) * paso, math.ceil(t.to_numpy().max() / paso) * paso
    vals, marcas = ui.marcas(lo, hi, paso)
    ui.unidad_eje(fig, "AUC de prueba")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[lo - paso / 5, hi + paso / 5], tickvals=vals, ticktext=marcas)

    ebm, todos = t.loc["ebm"], ns[-1]
    if n == todos:
        desde = N_REFERENCIA if N_REFERENCIA in ns else ns[0]
        titulo = (f"Con los {F.entero(n)} clientes la EBM llega a {ui.num(ebm[n], 4)} de AUC; desde {F.entero(desde)} solo gana "
                  f"{ui.num(ebm[n] - ebm[desde], 4)}")
    elif ebm[n] > t.loc["logistica", todos]:
        titulo = (f"Con {F.entero(n)} clientes la EBM llega a {ui.num(ebm[n], 4)} de AUC: más que {D.en_frase('logistica')} con los "
                  f"{F.entero(todos)} ({ui.num(t.loc['logistica', todos], 4)})")
    else:
        titulo = f"Con {F.entero(n)} clientes la EBM llega a {ui.num(ebm[n], 4)} de AUC"
    tabla = ui.tabla(["Modelo"] + [f"{c} clientes" for c in cats], [[D.NOMBRE_MODELO[m]] + [ui.num(v, 4) for v in t.loc[m]] for m in orden],
                     compacta=True, muestras=[D.ROL_VISTA_EBM[m] for m in orden])
    etiqueta = f"{titulo}. Con {F.entero(n)} clientes: " + "; ".join(f"{D.NOMBRE_MODELO[m]}, {ui.num(t.loc[m, n], 4)}" for m in orden) + "."
    return fig, titulo, tabla, etiqueta


def _finales():
    """AUC de prueba de los modelos finales de clasificación (los del resto de la app), por modelo."""
    return D.prueba("clasificacion").set_index("modelo")["auc_prueba"]


def _nota_aprendizaje():
    """Por qué el último punto de cada curva no es la cifra del modelo final (Resultados Entregable 3/generar.py,
    calculos_pesados y sensibilidad; resultados.tex, «Sensibilidad y robustez»)."""
    t, final = _curva_aprendizaje(), _finales()
    todos = t.columns.max()

    def frente(m):
        return f"{ui.num(t.loc[m, todos], 4)} frente a {ui.num(final[m], 4)}"

    return ("Estas curvas son reajustes hechos para este análisis, no los modelos finales, y por eso su último punto no repite "
            "las cifras del resto de la app. La EBM se reajustó con la configuración del modelo final sobre subconjuntos al azar "
            f"del entrenamiento; el último son los {F.entero(todos)} clientes en otro orden y da {frente('ebm')} del modelo final. "
            "Las curvas de las referencias vienen del experimento de escalamiento del Entregable 3, con los hiperparámetros "
            f"finales pero fuera de la validación anidada: {D.en_frase('logistica')} da {frente('logistica')} de su modelo final y "
            f"{D.en_frase('random_forest')} termina {ui.num(t.loc['random_forest', todos] - final['random_forest'], 3)} por encima "
            "del suyo.")


def _fichas_aprendizaje(n):
    """La EBM con el tamaño elegido frente a los demás, lo que aún gana con más datos y el modelo que más depende de ellos."""
    t = _curva_aprendizaje()
    ns = list(t.columns)
    n = n if n in ns else ns[0]
    menor, todos = ns[0], ns[-1]
    desde = N_REFERENCIA if N_REFERENCIA in ns else menor
    ebm = t.loc["ebm"]
    gana = (t[todos] - t[menor]).drop(index="ebm")
    sensible = gana.idxmax()
    demas = t[n].drop(index="ebm")
    nota = f"{D.NOMBRE_MODELO['logistica']}: {ui.num(t.loc['logistica', n], 4)}; {D.NOMBRE_MODELO['arbol']}: {ui.num(t.loc['arbol', n], 4)}."
    if n == todos:
        nota += f" Es un reajuste hecho para esta curva: el modelo final de la EBM da {ui.num(_finales()['ebm'], 4)}."
    return [
        ui.cifra(f"AUC de prueba de la EBM con {F.entero(n)} clientes", ui.num(ebm[n], 4),
                 variacion=f"{_frase(D.en_frase(demas.idxmax()))}, el mayor de los demás con ese tamaño: {ui.num(demas.max(), 4)}",
                 nota=nota, mini=True),
        ui.cifra(f"Lo que gana la EBM de {F.entero(desde)} a {F.entero(todos)} clientes", ui.con_signo(ebm[todos] - ebm[desde], 4), "de AUC",
                 nota=f"De {F.entero(menor)} a {F.entero(todos)} clientes sube de {ui.num(ebm[menor], 4)} a {ui.num(ebm[todos], 4)}.",
                 mini=True),
        ui.cifra("El modelo que más depende del tamaño", ui.con_signo(gana[sensible], 4), "de AUC",
                 nota=f"{_frase(D.en_frase(sensible))}: de {ui.num(t.loc[sensible, menor], 4)} con {F.entero(menor)} clientes a "
                      f"{ui.num(t.loc[sensible, todos], 4)} con los {F.entero(todos)}.", mini=True),
    ]


# --- Pantalla ----------------------------------------------------------------------------------------------------------
def layout(**_):
    tarea_abl = ui.control("Tarea:", ui.segmentado(f"{PRE}-abl-tarea", OPCIONES_TAREA, "clasificacion"))
    tarea_opt = ui.control("Tarea:", ui.segmentado(f"{PRE}-opt-tarea", OPCIONES_TAREA, "clasificacion"))
    foco_opt = ui.control("En foco:", ui.segmentado(f"{PRE}-opt-foco", [{"label": D.NOMBRE_OPTIMIZADOR[o], "value": o}
                                                                      for o in D.ORDEN_OPTIMIZADORES], _menor_area()["clasificacion"]))
    hiper = ui.control("Hiperparámetro:", ui.segmentado(f"{PRE}-sen-hp", [{"label": HIPER[h][0], "value": h} for h in HIPER],
                                                        "interactions"))
    medida = ui.control("Medida:", ui.segmentado(f"{PRE}-costo-medida", [{"label": MEDIDA[m]["opcion"], "value": m} for m in MEDIDA],
                                                 "ajuste_s"))
    tarea_costo = ui.control("Tarea:", ui.segmentado(f"{PRE}-costo-tarea", OPCIONES_TAREA, "clasificacion"))
    tamanos = list(_curva_aprendizaje().columns)
    tamano = ui.control("Clientes de entrenamiento:", ui.segmentado(f"{PRE}-apr-n", [{"label": F.entero(n), "value": int(n)}
                                                                                    for n in tamanos], int(tamanos[0])))
    # Las tarjetas y fichas que rellena un callback nacen ya con el contenido de los valores iniciales (no aparecen vacías)
    ficha_opt = ui.tarjeta([], titulo="", id=f"{PRE}-opt-ficha")
    ficha_opt.children = _ficha_optimizadores("clasificacion")
    return ui.pantalla([
        ui.encabezado(PREGUNTA, _subtitulo(), pantalla=PANTALLA),
        _seccion_hipotesis(),
        ui.seccion(_titulo_ablacion(), [ui.grafico(f"{PRE}-abl", fuentes=FUENTES_ABLACION)], controles=[tarea_abl],
                   subtitulo="Cada variante cambia un solo componente de la EBM completa y conserva el resto."),
        ui.seccion(_titulo_optimizadores(), [
            ui.rejilla([ui.grafico(f"{PRE}-opt", fuentes=FUENTES_CURVAS), ficha_opt], "1-1"),
        ], controles=[tarea_opt, foco_opt],
            subtitulo="Los optimizadores no tienen color propio: el que se elige va en foco y los demás, en gris con su nombre."),
        ui.seccion(_titulo_sensibilidad(), [
            ui.rejilla([ui.grafico(f"{PRE}-sen", fuentes=FUENTES_SENSIBILIDAD), ui.cifras(_fichas_sensibilidad())], "2-1"),
            ui.aviso("El informe declara exploratorios los estudios de sensibilidad: se añadieron después de ver la prueba.",
                     titulo="Análisis exploratorio"),
        ], controles=[hiper],
            fuentes=[D.fuente("resultados/concordancia_validacion_prueba.csv"), D.fuente("resultados/sensibilidad_hiperparametros.csv"),
                     D.fuente("resultados/ablacion.csv")]),
        ui.seccion(_titulo_calibracion(), [ui.rejilla([_grafico_calibracion(), _fichas_calibracion()], "2-1"), ui.nota(D.nota_ece())],
                   fuentes=FUENTES_CALIBRACION),
        ui.seccion(_titulo_costo(), [
            ui.rejilla([ui.grafico(f"{PRE}-costo", fuentes=FUENTES_COSTO, alto="alto",
                                   leyenda=[(D.NOMBRE_MODELO["ebm"], "modelo", "barra"), ("Modelos de árboles", "rival", "barra"),
                                            ("Lineales, k-NN y Naive Bayes", "base", "rayada")]),
                        ui.cifras(_fichas_costo("clasificacion"), id=f"{PRE}-costo-fichas")], "2-1"),
            _aviso_costo(),
        ], controles=[medida, tarea_costo], fuentes=FUENTES_COSTO),
        ui.seccion(_titulo_aprendizaje(), [
            ui.rejilla([ui.grafico(f"{PRE}-apr", fuentes=FUENTES_APRENDIZAJE, nota=_nota_aprendizaje(),
                                   subtitulo="AUC de prueba según el número de clientes con que se entrena cada modelo, con sus "
                                             "hiperparámetros finales. La columna resaltada es el tamaño elegido.",
                                   leyenda=[(D.NOMBRE_MODELO["ebm"], "modelo", "linea"), ("Modelos de árboles", "rival", "linea"),
                                            ("Lineales, k-NN y Naive Bayes", "base", "trazos")]),
                        ui.cifras(_fichas_aprendizaje(tamanos[0]), id=f"{PRE}-apr-fichas")], "2-1"),
        ], controles=[tamano], fuentes=FUENTES_APRENDIZAJE),
    ])


# --- Callbacks ---------------------------------------------------------------------------------------------------------
@callback(Output(f"{PRE}-abl", "figure"), Output(f"{PRE}-abl-titulo", "children"), Output(f"{PRE}-abl-sub", "children"),
          Output(f"{PRE}-abl-tabla", "children"), Output(f"{PRE}-abl-lienzo", "aria-label"), Output(f"{PRE}-abl-nota", "children"),
          Input(f"{PRE}-abl-tarea", "value"))
def _dibujar_ablacion(tarea):
    return _variante_ablacion(_tarea(tarea))


@callback(Output(f"{PRE}-opt", "figure"), Output(f"{PRE}-opt-titulo", "children"), Output(f"{PRE}-opt-sub", "children"),
          Output(f"{PRE}-opt-leyenda", "children"), Output(f"{PRE}-opt-tabla", "children"), Output(f"{PRE}-opt-lienzo", "aria-label"),
          Output(f"{PRE}-opt-ficha", "children"), Input(f"{PRE}-opt-tarea", "value"), Input(f"{PRE}-opt-foco", "value"))
def _dibujar_optimizadores(tarea, foco):
    tarea = _tarea(tarea)
    foco = foco if foco in D.ORDEN_OPTIMIZADORES else _menor_area()[tarea]
    return (*_curvas_arrepentimiento(tarea, foco), _ficha_optimizadores(tarea))


@callback(Output(f"{PRE}-sen", "figure"), Output(f"{PRE}-sen-titulo", "children"), Output(f"{PRE}-sen-sub", "children"),
          Output(f"{PRE}-sen-tabla", "children"), Output(f"{PRE}-sen-lienzo", "aria-label"), Input(f"{PRE}-sen-hp", "value"))
def _dibujar_sensibilidad(hp):
    return _sensibilidad(hp if hp in HIPER else "interactions")


@callback(Output(f"{PRE}-costo", "figure"), Output(f"{PRE}-costo-titulo", "children"), Output(f"{PRE}-costo-sub", "children"),
          Output(f"{PRE}-costo-tabla", "children"), Output(f"{PRE}-costo-lienzo", "aria-label"), Output(f"{PRE}-costo-fichas", "children"),
          Input(f"{PRE}-costo-medida", "value"), Input(f"{PRE}-costo-tarea", "value"))
def _dibujar_costo(medida, tarea):
    tarea = _tarea(tarea)
    return (*_costo(medida if medida in MEDIDA else "ajuste_s", tarea), _fichas_costo(tarea))


@callback(Output(f"{PRE}-apr", "figure"), Output(f"{PRE}-apr-titulo", "children"), Output(f"{PRE}-apr-tabla", "children"),
          Output(f"{PRE}-apr-lienzo", "aria-label"), Output(f"{PRE}-apr-fichas", "children"), Input(f"{PRE}-apr-n", "value"))
def _dibujar_aprendizaje(n):
    return (*_aprendizaje(n), _fichas_aprendizaje(n))
