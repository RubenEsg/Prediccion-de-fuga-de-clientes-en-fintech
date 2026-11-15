"""Algoritmo y veredicto · ¿Qué es la EBM, por qué y supera a los demás?

La subsección del modelo propuesto que pide la guía del Tercer Entregable (§7.3.3), sobre el dash: la limitación que
atiende, la formulación matemática (modelo aditivo, funciones escalonadas, identificabilidad, pérdidas, pesos de clase,
boosting por variable, bolsas, regularización y supuestos), el pseudocódigo, una propiedad con su demostración, la
complejidad contrastada con los tiempos medidos, qué es nuevo y qué se reutiliza con las diferencias frente a los métodos
afines, las curvas de aprendizaje y de convergencia, la tabla de pruebas con p crudos y ajustados por Holm y tamaños de
efecto, el resumen de robustez y el veredicto explícito con la definición operativa de superioridad de la guía (§6.4).

La fuente de las afirmaciones es el informe final (metodologia.tex, resultados.tex y conclusiones.tex). Toda cifra sale de
datos/, de artifacts/ o de los módulos (motor, modelos_vivos, datos_app) y cada tarjeta dice de qué archivo. Las
constantes de abajo son los umbrales de la regla de superioridad de la guía, constantes matemáticas de la demostración o
de presentación; ninguna es un resultado del proyecto. El veredicto de cada referencia es el de datos_app.veredicto.
"""
import json
import math
import re
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import app
import datos_app as D
import formato as F
import motor as M
import modelos_vivos as V
import ui


VISTA = "propuesto"
PRE = "pp"
PANTALLA = "modelo"
RIVAL = "xgboost"                 # el modelo de árboles con el que el informe compara a la EBM
TAREAS = ("clasificacion", "regresion")

# La regla de superioridad de la guía (§6.4): cuatro condiciones y sus umbrales
CONDICIONES = 4
PRUEBAS_MINIMAS = 3               # pruebas distintas que deben rechazar la igualdad frente al mejor modelo de referencia
ALFA_MCS = 0.10                   # umbral del Model Confidence Set en esa condición
UMBRAL_CLIFF = 0.147              # |δ de Cliff| desde el que el efecto deja de ser despreciable
CONJUNTOS_MINIMOS = 3             # conjuntos de datos adicionales en los que debe repetirse
CONJUNTOS_EVALUADOS = 0           # el proyecto usó un único conjunto de datos (informe, «Discusión»: «Hay un solo conjunto de datos»)

NIVEL_MCS = 0.90                  # nivel del Model Confidence Set del informe
UMBRAL_EQUIVALENCIA = 0.95        # el umbral habitual con que el informe lee la probabilidad de equivalencia práctica
C_LOGIT = 0.25                    # máximo de p(1 − p): cota de la segunda derivada de la log-loss en la escala del logit
C_CUADRATICA = 2.0                # segunda derivada de (s − u)², la pérdida de la regresión
N_DESDE = 10_000                  # desde aquí se mide la pendiente del tiempo de ajuste (el informe mide desde ahí lo que aún gana la curva)
MICRO = 1e6                       # segundos → microsegundos
N_PRINCIPALES = 4                 # las variables con señal que nombra el informe
UMBRAL_MEDIO = 0.5                # umbral con que la ablación mide la captura (recall) del balanceo
PLIEGUES_SENSIBILIDAD = 3         # pliegues de la validación cruzada de las curvas de sensibilidad (informe, «Sensibilidad y robustez»)
PISO_PROBABILIDAD = 1e-4          # una probabilidad posterior más cerca de 0 o de 1 que esto se escribe como cota
NIVEL_CONFORME = "0.90"           # el nivel de los conjuntos conformes con que nace la predicción en vivo
PASO_AUC = 0.05                   # paso de las marcas del eje de la curva de aprendizaje
PASO_TRAZA = 0.0025               # paso de las marcas del eje de la curva de convergencia
CARDINAL = {0: "ninguna", 1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho"}
ORDINAL_F = {1: "primera", 2: "segunda", 3: "tercera", 4: "cuarta", 5: "quinta", 6: "sexta", 7: "séptima", 8: "octava"}
SUPERINDICE = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
RAYA = F.RAYA

# Referencias del informe (Informe Final/referencias.bib) que cita esta vista: autores y año
REF = {"lou2012": ("Lou, Caruana y Gehrke", 2012), "lou2013": ("Lou, Caruana, Gehrke y Hooker", 2013),
       "nori2019": ("Nori, Jenkins, Koch y Caruana", 2019), "breiman1984": ("Breiman, Friedman, Olshen y Stone", 1984),
       "breiman2001": ("Breiman", 2001), "chen2016": ("Chen y Guestrin", 2016), "cortes1995": ("Cortes y Vapnik", 1995),
       "pedregosa2011": ("Pedregosa et al.", 2011), "lundberg2017": ("Lundberg y Lee", 2017), "nadeau2003": ("Nadeau y Bengio", 2003),
       "delong1988": ("DeLong, DeLong y Clarke-Pearson", 1988), "mcnemar1947": ("McNemar", 1947),
       "benavoli2017": ("Benavoli, Corani, Demšar y Zaffalon", 2017), "diebold1995": ("Diebold y Mariano", 1995),
       "harvey1997": ("Harvey, Leybourne y Newbold", 1997), "hansen2011": ("Hansen, Lunde y Nason", 2011),
       "holm1979": ("Holm", 1979)}

# Lo que cambia entre las dos tareas: columnas de datos/resultados, decimales del informe y nombres de las pruebas
TAREA = {
    "clasificacion": {"metrica": "AUC", "d": 4, "externa": "auc_ext_media", "externa_desv": "auc_ext_desv", "prueba": "auc_prueba",
                      "mayor": True, "delta": "delta_auc", "fijada": "DeLong", "simbolo": "z", "est": "z_delong", "p": "p_delong",
                      "p_holm": "p_delong_holm", "rope": "auc", "d_rope": 2, "externa_nombre": "AUC externa",
                      "pliegues": "resultados/pliegues_auc.csv"},
    "regresion": {"metrica": "RMSE", "d": 5, "externa": "rmse_ext_media", "externa_desv": "rmse_ext_desv", "prueba": "rmse_prueba",
                  "mayor": False, "delta": "delta_rmse", "fijada": "Diebold-Mariano", "simbolo": "DM", "est": "dm_hln", "p": "p_dm",
                  "p_holm": "p_dm_holm", "rope": "rmse", "d_rope": 4, "externa_nombre": "RMSE externo",
                  "pliegues": "resultados/pliegues_rmse.csv"},
}
OPCIONES_TAREA = [{"label": D.NOMBRE_TAREA[t], "value": t} for t in TAREA]

INFORME_MODELO = "Informe final: metodologia.tex, «Modelo propuesto»"
INFORME_ENTRENAMIENTO = "Informe final: metodologia.tex, «Entrenamiento y optimización»"
INFORME_PROTOCOLO = "Informe final: metodologia.tex, «Protocolo de evaluación»; resultados.tex, «Resumen del montaje experimental»"
FUENTES_EBM = [D.fuente(f"modelos/ebm_{t}.json") for t in TAREAS] + [D.fuente("metadatos.json")]
FUENTES_META = [f"artifacts/{t}/ebm/metadata.json" for t in TAREAS]
FUENTES_PRUEBA = [D.fuente(f"resultados/{t}_prueba.csv") for t in TAREAS]
FUENTES_PRUEBAS = [D.fuente(f"resultados/pruebas_{t}.csv") for t in TAREAS] + [D.fuente("resultados/resumen.json")]
FUENTES_COSTO = [D.fuente("entrega3/computo/tiempos_finales.csv"), D.fuente("resultados/costo.csv")]
FUENTES_APRENDIZAJE = [D.fuente("resultados/curva_aprendizaje.csv"), D.fuente("resultados/clasificacion_prueba.csv")]
BUSQUEDAS_E2 = "entrega3/comparacion_entrega2/busquedas.json"
FUENTES_ROBUSTEZ = [D.fuente("resultados/semillas.csv"), D.fuente("resultados/sensibilidad_curvas.csv"), D.fuente("resultados/ablacion.csv"),
                    D.fuente("resultados/ablacion_interacciones_e2.csv"), D.fuente("resultados/fidelidad_resumen.csv"),
                    D.fuente("calibracion_modelos.json"), D.fuente("conformal.json")]


# --- Textos y números ----------------------------------------------------------------------------------------------------
def _finito(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _mayuscula(texto):
    return texto[:1].upper() + texto[1:]


def _minuscula(texto):
    """Primera letra en minúscula para usar un nombre dentro de una frase (no toca las siglas: «AUC» se queda)."""
    return texto[:1].lower() + texto[1:] if texto[1:2].islower() else texto


def _cita(clave):
    """Cita en el texto: «Lou, Caruana y Gehrke (2012)»."""
    autores, anio = REF[clave]
    return f"{autores} ({anio})"


def _cita_en(clave):
    """Cita dentro de un paréntesis: «Lou, Caruana y Gehrke, 2012»."""
    autores, anio = REF[clave]
    return f"{autores}, {anio}"


def _cita_corta(clave):
    """Cita corta para una tabla: «Lou et al. (2012)» con tres autores o más; si no, la cita completa."""
    autores, anio = REF[clave]
    cuantos = autores.count(",") + 1 + (" y " in autores)
    primero = autores.split(",")[0].split(" y ")[0]
    return f"{primero} et al. ({anio})" if cuantos > 2 and "et al." not in autores else f"{autores} ({anio})"


def _igual(simbolo, valor):
    """«B = 8» con espacios que no parten la línea."""
    return f"{simbolo}{F.FIJO}={F.FIJO}{valor}"


def _decimales(v):
    """Decimales con que se escribe un hiperparámetro: 0,008 → 3 · 100 → 0."""
    texto = repr(float(v))
    return 0 if texto.endswith(".0") else len(texto.split(".")[1])


def _valor(v):
    return ui.num(v, _decimales(v))


def _p(p):
    """Un valor p en una celda cuyo encabezado ya dice que es un p: «0,245» o «< 0,001»."""
    if not _finito(p):
        return RAYA
    return F.p_valor(p).replace(F.FIJO, " ").replace("p = ", "").replace("p ", "").replace(" ", F.FIJO)


def _prob(p):
    """Probabilidad posterior con los decimales justos (0,94 · 0,997 · 0,9998); más cerca de 0 o de 1 que PISO_PROBABILIDAD,
    una cota (« < 0,0001 »), como en «Comparación»."""
    if not _finito(p):
        return RAYA
    if p < PISO_PROBABILIDAD:
        return f"<{F.FIJO}{ui.num(PISO_PROBABILIDAD, 4)}"
    if p > 1 - PISO_PROBABILIDAD:
        return f">{F.FIJO}{ui.num(1 - PISO_PROBABILIDAD, 4)}"
    return F.prob(p)


def _es(rotulo, texto):
    """«P(equivalente) = 0,94» o, si el valor es una cota, «P(equivalente) < 0,0001»."""
    return f"{rotulo} {texto}" if str(texto)[:1] in "<>" else f"{rotulo} = {texto}"


def _cientifico(x, d=1):
    """Una cantidad muy pequeña en notación científica: 3,2·10⁻¹⁴."""
    if not _finito(x) or x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    return f"{ui.num(x / 10 ** e, d)}·10{str(e).translate(SUPERINDICE)}"


def _tex(x, d=0):
    """Un número en es-CO dentro de una fórmula de MathJax: 38{.}978 · 0{,}008 · -0{,}0132."""
    return F.num(x, d).replace(F.MENOS, "-").replace(".", "{.}").replace(",", "{,}")


def _sust(texto, **valores):
    """Pone cada valor en su «marca» de una plantilla (así las fórmulas se escriben sin duplicar las llaves de LaTeX)."""
    for clave, valor in valores.items():
        texto = texto.replace(f"«{clave}»", str(valor))
    return texto


def _pegar(texto):
    """Mete dentro de cada fórmula en línea la puntuación que la sigue y el paréntesis que la abre (como texto de la
    fórmula): MathJax dibuja la fórmula como un bloque en línea y, sin esto, el navegador puede dejar «:» o «)» solos al
    comienzo de la línea siguiente."""
    texto = re.sub(r"(?<!\$)\$([,.;:)\]]+)", lambda m: r"\text{" + m.group(1) + "}$", texto)
    return re.sub(r"([(\[])\$(?!\$)", lambda m: r"$\text{" + m.group(1) + "}", texto)


def _md(texto, **valores):
    """Un párrafo con fórmulas en línea ($…$), en la medida de lectura del texto."""
    return dcc.Markdown(_pegar(_sust(texto, **valores)), mathjax=True, className="parrafo")


def _ec(tex, **valores):
    """Una ecuación en bloque, en su marco blanco con desplazamiento propio (no se sale de la tarjeta en el teléfono)."""
    return html.Div(dcc.Markdown("$$\n" + _sust(tex, **valores) + "\n$$", mathjax=True), className="tabla-marco")


def _lista(modelos):
    return F.enumerar(D.en_frase(m) for m in modelos)


def _tras(preposicion, frase):
    """«a» o «de» delante de un nombre en frase, con su contracción: «al árbol de decisión», «del SVM lineal»."""
    if frase.startswith("el "):
        return {"a": "al", "de": "del"}[preposicion] + frase[2:]
    return f"{preposicion} {frase}"


# --- Lecturas de datos con caché -------------------------------------------------------------------------------------------
def _hp(tarea):
    """Hiperparámetros del modelo final de la EBM (artifacts/<tarea>/ebm/metadata.json, iguales a los de metadatos.json)."""
    return {k.replace("modelo__", ""): v for k, v in V.metadata(tarea, "ebm")["hiperparametros"].items()}


def _clave(tarea, modelo="ebm"):
    md = V.metadata(tarea, modelo)
    return f"{tarea}__{modelo}__{md['balanceo']}__{md['optimizador']}"


def _corrida(tarea):
    """La fila de entrega3/corridas.csv de la mejor combinación de la EBM en esa tarea."""
    return D.fila(D.tabla("entrega3/corridas.csv"), clave=_clave(tarea))


def _fidelidad():
    """Bolsas en la búsqueda y en el modelo final (la multi-fidelidad de dos niveles), leídas de la corrida."""
    f = json.loads(_corrida("clasificacion")["fidelidad"])
    return int(f["busqueda"]["modelo__outer_bags"]), int(f["final"]["modelo__outer_bags"])


def _bins():
    """max_bins de la mejor combinación de clasificación (resultados/sensibilidad_hiperparametros.csv)."""
    md = V.metadata("clasificacion", "ebm")
    return int(D.fila(D.tabla("resultados/sensibilidad_hiperparametros.csv"), balanceo=md["balanceo"],
                      optimizador=md["optimizador"])["max_bins"])


def _tramos(tarea):
    """Tramos de cada término del modelo exportado (sin el de faltantes ni el de desconocidos)."""
    return [len(t["cortes"]) + 1 for t in M.ebm(tarea).terminos]


def _costo(tarea):
    t = D.tabla("resultados/costo.csv")
    return t[t["tarea"] == tarea].set_index("modelo")


def _tiempos(tarea):
    t = D.tabla("entrega3/computo/tiempos_finales.csv")
    return t[t["tarea"] == tarea].set_index("modelo")


@lru_cache(maxsize=1)
def _entrenamiento():
    """Lo que se comprueba en vivo con los clientes de entrenamiento y el motor: los pesos de clase, el centrado de cada
    función de forma (media ponderada en el entrenamiento), la media del puntaje y el rango de p(1 − p) del modelo final.
    Es de solo lectura: se comparte entre hilos."""
    E = V.entrenamiento()
    y = E["riesgo_alto"].to_numpy().astype(int)
    n, pos = len(y), int(y.sum())
    w = np.where(y == 1, n / (2 * pos), n / (2 * (n - pos)))
    X = E[M.predictores()]
    centrado = {}
    for tarea in TAREAS:
        C = M.ebm(tarea).contribuciones(X)
        pesos = w if V.metadata(tarea, "ebm")["balanceo"] == "class_weight" else np.ones(n)
        centrado[tarea] = float(np.abs((C * pesos[:, None]).sum(axis=0) / pesos.sum()).max())
    p = M.ebm("clasificacion").predecir(X)
    return {"n": n, "pos": pos, "neg": n - pos, "w_pos": n / (2 * pos), "w_neg": n / (2 * (n - pos)), "centrado": centrado,
            "media_s": float(E["churn_probability"].mean()), "min_q": float((p * (1 - p)).min()), "media_p": float(p.mean())}


# --- Las pruebas de una referencia ---------------------------------------------------------------------------------------
def _contrastes(tarea, f):
    """Las pruebas con valor p de la EBM frente a una referencia (una fila de pruebas_<tarea>.csv): nombre, estadístico,
    p crudo, p ajustado por Holm y si el estadístico favorece a la EBM (un valor positivo la favorece)."""
    c = TAREA[tarea]
    filas = [{"prueba": "t corregida", "estadistico": f"t = {ui.num(f['t_corr'], 2)}", "p": f["p_t"], "p_holm": f["p_t_holm"],
              "favorece": f["t_corr"] > 0},
             {"prueba": c["fijada"], "estadistico": f"{c['simbolo']} = {ui.num(f[c['est']], 2)}", "p": f[c["p"]],
              "p_holm": f[c["p_holm"]], "favorece": f[c["est"]] > 0}]
    if tarea == "clasificacion":
        filas.append({"prueba": "McNemar", "estadistico": f"b / c = {F.entero(f['disc_base_bien'])} / {F.entero(f['disc_ebm_bien'])}",
                      "p": f["p_mcnemar"], "p_holm": f["p_mcnemar_holm"], "favorece": f["disc_ebm_bien"] > f["disc_base_bien"]})
    return filas


def _rechazan(contrastes):
    return [x for x in contrastes if _finito(x["p_holm"]) and x["p_holm"] < D.ALFA and x["favorece"]]


def _decision(x, ref):
    if _finito(x["p_holm"]) and x["p_holm"] < D.ALFA:
        return "rechaza H₀ a favor de la EBM" if x["favorece"] else f"rechaza H₀ a favor de {D.NOMBRE_CORTO[ref]}"
    if _finito(x["p"]) and x["p"] < D.ALFA:
        return "rechaza sin corregir; no tras Holm"
    return "no rechaza H₀"


def _mcs():
    """p del Model Confidence Set de cada modelo de regresión y si queda dentro."""
    t = D.prueba("regresion").set_index("modelo")
    return t["p_mcs"], t["en_mcs"].astype(bool)


# --- 0. La respuesta de la pantalla --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _veredictos():
    """Por tarea, las referencias con las que la EBM empata y las que supera (datos_app.veredicto)."""
    salida = {}
    for tarea in TAREAS:
        v = D.veredictos(tarea)
        salida[tarea] = {"empata": [m for m in v.index if v[m] == "empata"], "supera": [m for m in v.index if v[m] == "supera"],
                         "otro": [m for m in v.index if v[m] not in ("empata", "supera")]}
    return salida


def _quien_supera(con_a=True):
    """«a los modelos lineales, a k-NN y a Naive Bayes» (o sin las «a») si las que supera son las de la base en las dos
    tareas; si no, la lista de cada tarea."""
    v = _veredictos()
    base = all(set(v[t]["supera"]) == {m for m in D.pruebas(t)["modelo"] if D.ROL_MODELO[m] == "base"} for t in TAREAS)
    if base:
        return "a los modelos lineales, a k-NN y a Naive Bayes" if con_a else "los modelos lineales, k-NN y Naive Bayes"
    listas = [_tras("a", _lista(v[t]["supera"])) if con_a else _lista(v[t]["supera"]) for t in TAREAS]
    return f"{listas[0]} en clasificación y {listas[1]} en regresión"


def _con_quien_empata():
    v = _veredictos()
    arboles = all(set(v[t]["empata"]) == {m for m in D.pruebas(t)["modelo"] if D.ROL_MODELO[m] == "rival"} for t in TAREAS)
    return _lista([m for m in D.ORDEN_MODELOS["clasificacion"] if D.ROL_MODELO.get(m) == "rival"]) if arboles else \
        _lista(v["clasificacion"]["empata"])


def _respuesta():
    clf = D.prueba("clasificacion").set_index("modelo")
    cumplidas = {t: sum(x["cumple"] for x in _condiciones(t)["condiciones"]) for t in TAREAS}
    if len(set(cumplidas.values())) == 1:
        k = next(iter(cumplidas.values()))
        reglas = (f"no cumple {CARDINAL[k]} de sus {CARDINAL[CONDICIONES]} condiciones" if k == 0
                  else f"cumple {CARDINAL[k]} de sus {CARDINAL[CONDICIONES]} condiciones en cada tarea")
    else:
        reglas = " y ".join(f"cumple {CARDINAL[k]} de {CARDINAL[CONDICIONES]} en {D.NOMBRE_TAREA[t].lower()}" for t, k in cumplidas.items())
    superior = all(k == CONDICIONES for k in cumplidas.values())
    return (f"{'Sí' if superior else 'No'}. Es un modelo aditivo de funciones de una variable aprendidas por boosting: empata con "
            f"{_con_quien_empata()} (AUC de prueba {ui.num(clf.loc['ebm', 'auc_prueba'], 4)} frente a "
            f"{ui.num(clf.loc[RIVAL, 'auc_prueba'], 4)} {_tras('de', D.en_frase(RIVAL))}; ninguna prueba encuentra diferencia tras Holm) y "
            f"supera con significación {_quien_supera()}. Con la regla de superioridad de la guía {reglas}: el veredicto es de "
            f"{'superioridad' if superior else 'no superioridad'}.")


# --- 1. Qué es y por qué -------------------------------------------------------------------------------------------------
def _seccion_motivacion():
    clf = D.prueba("clasificacion").set_index("modelo")
    costo = _costo("clasificacion")
    hx = V.metadata("clasificacion", RIVAL)["hiperparametros"]
    ebm = M.meta()["ebm"]["clasificacion"]
    imp = M.importancia("clasificacion")
    principales = F.enumerar(_minuscula(M.nombre(v)) for v in imp.index[:N_PRINCIPALES])

    def auc(m):
        return ui.num(clf.loc[m, "auc_prueba"], 4)

    fichas = [
        ui.cifra(f"{D.NOMBRE_MODELO['logistica']}: una pendiente por columna", auc("logistica"), "de AUC de prueba",
                 nota=f"{F.entero(costo.loc['logistica', 'parametros'])} coeficientes. Se lee sin ayuda, pero cada variable solo puede "
                      "subir o bajar el logit en línea recta.", mini=True),
        ui.cifra(f"{D.NOMBRE_MODELO[RIVAL]}: {F.entero(hx['modelo__n_estimators'])} árboles de profundidad "
                 f"{F.entero(hx['modelo__max_depth'])}", auc(RIVAL), "de AUC de prueba",
                 nota=f"{F.entero(costo.loc[RIVAL, 'parametros'])} nodos que se combinan entre sí: explicar una predicción pide un "
                      "explicador aparte, como SHAP.", mini=True),
        ui.cifra(f"EBM: {F.entero(ebm['terminos'])} funciones de una variable", auc("ebm"), "de AUC de prueba",
                 chip=ui.chip("Modelo propuesto", "acento"),
                 nota=f"{F.entero(costo.loc['ebm', 'parametros'])} valores en tablas. La predicción es la suma de lo que aporta cada "
                      "variable, y cada aporte se lee en su curva.", mini=True),
    ]
    titulo = (f"Ni rígida ni opaca: la EBM llega al AUC {_tras('de', D.en_frase(RIVAL))} ({auc('ebm')} frente a {auc(RIVAL)}) con "
              f"{F.entero(ebm['terminos'])} funciones de una variable")
    intro = (f"La limitación que atiende es la de los dos extremos: los modelos lineales se leen, pero obligan a que cada variable "
             "actúe en línea recta; los ensambles de árboles captan lo no lineal, pero su decisión no se lee sin un explicador "
             f"aparte. La EBM ({_cita_en('lou2012')}; {_cita_en('lou2013')}) suma una curva por variable aprendida con boosting: la "
             "flexibilidad del boosting con la legibilidad de un modelo aditivo. No se vio en el curso. Conviene aquí porque, según el "
             "análisis exploratorio, la señal está en pocas variables con efectos no lineales y casi independientes, que es la "
             f"hipótesis aditiva: en el modelo final, {principales} reúnen el {F.pct(imp.iloc[:N_PRINCIPALES].sum(), 0)} de la importancia.")
    return ui.seccion(titulo, [ui.cifras(fichas)], intro=intro,
                      subtitulo="Qué es, qué limitación de los métodos existentes atiende y por qué encaja en estos datos.",
                      fuentes=[D.fuente("resultados/clasificacion_prueba.csv"), D.fuente("resultados/costo.csv"),
                               f"artifacts/clasificacion/{RIVAL}/metadata.json", D.fuente("modelos/ebm_clasificacion.json"),
                               INFORME_MODELO])


# --- 2. Formulación ------------------------------------------------------------------------------------------------------
EC_MODELO = r"""\begin{aligned}
g\big(\mathbb{E}[\,y \mid \mathbf{x}\,]\big) &= \beta_0 + \sum_{j=1}^{p} f_j(x_j) \\
&\quad + \sum_{(j,k) \in \mathcal{I}} f_{jk}(x_j, x_k)
\end{aligned}"""
EC_ESCALON = r"f_j(x) = \sum_{t=1}^{T_j} \theta_{jt}\, \mathbf{1}\big[\, c_{j,t-1} < x \le c_{jt} \,\big]"
EC_CENTRADO = r"\sum_{i=1}^{n} w_i\, f_j(x_{ij}) = 0, \qquad j = 1, \dots, p"
EC_PERDIDAS = r"""\begin{aligned}
\mathcal{L}_C &= -\frac{1}{n}\sum_{i=1}^{n} w_i \big[\, z_i \ln \hat{p}_i \\
&\qquad\qquad + (1 - z_i) \ln (1 - \hat{p}_i) \,\big] \\
\mathcal{L}_R &= \frac{1}{n}\sum_{i=1}^{n} (s_i - \hat{s}_i)^2
\end{aligned}"""
EC_PESOS = r"""w_i = \frac{n}{2\, n_{z_i}} = \begin{cases} «wpos» & \text{si } z_i = 1 \\ «wneg» & \text{si } z_i = 0 \end{cases}"""
EC_PASO = r"""\begin{aligned}
f_j^{(r+1)}(x_j) &= f_j^{(r)}(x_j) + \eta\, h_j^{(r)}(x_j) \\
h_j^{(r)} &= \arg\min_{h} \sum_{i=1}^{n} w_i \big( \varrho_i^{(r)} - h(x_{ij}) \big)^2 \\
\varrho_i^{(r)} &= -\,\partial \ell\big(y_i, \hat{u}_i^{(r)}\big) \big/ \partial \hat{u}_i^{(r)}
\end{aligned}"""
EC_GRADIENTES = r"""\varrho_i = \begin{cases} z_i - \hat{p}_i & \text{(log-loss, en el logit)} \\ 2\,(s_i - \hat{s}_i) & \text{(error al cuadrado)} \end{cases}"""
EC_BOLSAS = r"f_j = \frac{1}{B} \sum_{b=1}^{B} f_j^{(b)}, \qquad B = «B»"


def _tarjeta_modelo():
    ebm = M.meta()["ebm"]["clasificacion"]
    tramos = _tramos("clasificacion")
    varios = [t for t in tramos if t > 2]
    dos = [t for t in tramos if t <= 2]
    return ui.tarjeta([
        _ec(EC_MODELO),
        _md(r"donde $y$ es la etiqueta $z$ o el puntaje $s$ según la tarea, $g$ es el enlace (logit en clasificación, identidad en "
            r"regresión), $f_j$ es la función de forma de la columna $j$ ($p = «p»$ columnas tras el preprocesamiento) e "
            r"$\mathcal{I}$ es un conjunto de $K$ pares elegido con el algoritmo FAST. En los dos modelos finales $K = «K»$: el "
            r"último término desaparece y el modelo es un GAM puro.", p=_tex(ebm["terminos"]), K=_tex(_hp("clasificacion")["interactions"])),
        _ec(EC_ESCALON),
        _md(r"A diferencia de un GAM con splines, cada $f_j$ es escalonada: $T_j \le \mathrm{bins}$ tramos con cortes $c_{jt}$ en los "
            r"cuantiles de $x_j$ y un valor $\theta_{jt}$ por tramo. En el modelo de clasificación, «n_varios» columnas usan de «min» a "
            r"«max» tramos y las «n_dos» indicadoras y booleanas, «t_dos»; con el tramo de faltantes y el de valores desconocidos, el "
            r"modelo guarda «param» valores.", n_varios=F.entero(len(varios)), min=F.entero(min(varios)), max=F.entero(max(varios)),
            n_dos=F.entero(len(dos)), t_dos=F.entero(max(dos)), param=F.entero(ebm["parametros"])),
    ], titulo="El modelo aditivo", subtitulo="La predicción, en la escala del enlace, es una suma de funciones de una variable.",
        fuentes=FUENTES_EBM + [INFORME_MODELO])


def _tarjeta_identificabilidad():
    e = _entrenamiento()
    b0 = {t: M.ebm(t).intercepto for t in TAREAS}
    return ui.tarjeta([
        _md(r"Sumar una constante a $f_j$ y restarla de $\beta_0$ no cambia ninguna predicción: sin una restricción, la "
            r"descomposición no es única. Se fija centrando cada función en el entrenamiento:"),
        _ec(EC_CENTRADO),
        _md(r"con $w_i$ los pesos de clase en clasificación y $w_i = 1$ en regresión. Así $\beta_0$ es el nivel medio y cada $f_j$ "
            r"dice solo cuánto se aparta de él. Comprobado en vivo con los «n» clientes de entrenamiento: la media ponderada de las "
            r"«terminos» funciones es a lo sumo «cen_c» en clasificación y «cen_r» en regresión. En regresión, "
            r"$\beta_0 = «b0r»$ es la media del puntaje en el entrenamiento («media»); en clasificación, como los pesos igualan "
            r"las dos clases, $\beta_0 = «b0c»$ queda cerca del logit de un medio (0).",
            n=F.entero(e["n"]), terminos=F.entero(M.meta()["ebm"]["clasificacion"]["terminos"]), cen_c=_cientifico(e["centrado"]["clasificacion"]),
            cen_r=_cientifico(e["centrado"]["regresion"]), b0r=_tex(b0["regresion"], 5), media=ui.num(e["media_s"], 5),
            b0c=_tex(b0["clasificacion"], 4)),
    ], titulo="Identificabilidad: cada función está centrada",
        subtitulo="La restricción que hace única la descomposición, comprobada con el modelo en vivo.",
        fuentes=FUENTES_EBM + [D.fuente("eda/entrenamiento.parquet")])


def _tarjeta_perdidas():
    e = _entrenamiento()
    md = V.metadata("clasificacion", "ebm")
    meta = M.meta()
    return ui.tarjeta([
        _ec(EC_PERDIDAS),
        _md(r"En clasificación, $\hat{p}_i$ es la probabilidad predicha y $z_i$ la etiqueta de riesgo alto: el cuarto superior del "
            r"puntaje, $z_i = \mathbf{1}[s_i > \tau]$ con $\tau = «tau»$ (percentil «q» del entrenamiento). En regresión, "
            r"$\hat{s}_i$ es el pronóstico del puntaje $s_i$. Los pesos de clase son",
            tau=_tex(meta["etiqueta"]["umbral"], 4), q=F.entero(meta["etiqueta"]["q"] * 100)),
        _ec(EC_PESOS, wpos=_tex(e["w_pos"], 2), wneg=_tex(e["w_neg"], 2)),
        _md(r"con $n_{z_i}$ los clientes de la clase de $i$ («pos» de riesgo alto y «neg» de los demás): las dos clases pesan lo mismo, "
            r"como en la ponderación balanceada de scikit-learn que reciben las referencias. La mejor combinación de clasificación "
            r"usa «balanceo» («optimizador»); la regresión no tiene balanceo y $w_i = 1$.",
            pos=F.entero(e["pos"]), neg=F.entero(e["neg"]), balanceo=D.NOMBRE_BALANCEO[md["balanceo"]],
            optimizador=D.NOMBRE_OPTIMIZADOR[md["optimizador"]]),
        ui.nota(f"Los pesos inflan el puntaje: en el entrenamiento su media es {ui.num(e['media_p'], 3)} con una prevalencia de "
                f"{ui.num(meta['prevalencia_entrenamiento'], 3)}. Sirve para ordenar; para leerlo como probabilidad se recalibra."),
    ], titulo="Las pérdidas que minimiza cada tarea",
        subtitulo="Entropía cruzada ponderada en clasificación y error cuadrático en regresión.",
        fuentes=[D.fuente("metadatos.json"), "artifacts/clasificacion/ebm/metadata.json", D.fuente("eda/entrenamiento.parquet"), INFORME_MODELO])


def _tarjeta_aprendizaje():
    h = _hp("clasificacion")
    return ui.tarjeta([
        _ec(EC_PASO),
        _md(r"Cada paso ajusta un árbol $h$ de a lo sumo $L$ hojas, con al menos $m$ clientes por hoja y sobre una sola columna, al "
            r"gradiente negativo $\varrho_i$ de la pérdida en la predicción acumulada $\hat{u}_i$ (en la escala del enlace: "
            r"$\hat{u}_i = \operatorname{logit} \hat{p}_i$ o $\hat{u}_i = \hat{s}_i$), y lo suma escalado por una tasa $\eta$ "
            r"pequeña. Para las dos pérdidas:"),
        _ec(EC_GRADIENTES),
        _md(r"El procedimiento se repite en $B$ bolsas: cada una deja fuera una parte del entrenamiento para su parada temprana, y "
            r"las funciones se promedian:"),
        _ec(EC_BOLSAS, B=_tex(h["outer_bags"])),
        _md(r"El promedio reduce la varianza de cada curva y da su banda (la desviación entre bolsas que muestra «Cómo decide»). "
            r"La tasa baja hace que el orden de las variables no importe y reparte la señal compartida entre variables correlacionadas."),
    ], titulo="Cómo aprende: boosting de una variable a la vez",
        subtitulo="El principio de aprendizaje: descenso por gradiente en el espacio de las funciones de forma.",
        fuentes=FUENTES_META + [INFORME_MODELO])


def _tarjeta_regularizacion():
    hc, hr = _hp("clasificacion"), _hp("regresion")
    tc, tr = _tramos("clasificacion"), _tramos("regresion")
    filas = [
        ["Tasa de aprendizaje", "η", "cuánto se mueve cada función en un paso; con η pequeño el orden de las variables no importa",
         _valor(hc["learning_rate"]), _valor(hr["learning_rate"])],
        ["Hojas por árbol", "L", "cuántos escalones nuevos puede añadir un paso", F.entero(hc["max_leaves"]), F.entero(hr["max_leaves"])],
        ["Mínimo de clientes por hoja", "m", "cuántos clientes respaldan cada escalón", F.entero(hc["min_samples_leaf"]),
         F.entero(hr["min_samples_leaf"])],
        ["Tramos por columna (el máximo que usa)", f"Tⱼ{F.FIJO}≤{F.FIJO}bins", "la resolución de cada curva", F.entero(max(tc)),
         F.entero(max(tr))],
        ["Bolsas externas", "B", "promedio de B ajustes sobre submuestras: menos varianza", F.entero(hc["outer_bags"]), F.entero(hr["outer_bags"])],
        ["Pares", "K", "capacidad para interacciones (0: aditivo puro)", F.entero(hc["interactions"]), F.entero(hr["interactions"])],
        ["Parada temprana", RAYA, "detiene cada bolsa cuando la pérdida en su validación deja de mejorar", "sí", "sí"],
    ]
    tabla = ui.tabla(["Hiperparámetro", "Símbolo", "Qué controla", D.NOMBRE_TAREA["clasificacion"], D.NOMBRE_TAREA["regresion"]], filas,
                     compacta=True, num=[False, False, False, True, True], ocultar_movil=(1,))
    return ui.tarjeta([
        tabla,
        ui.nota("No hay penalización L1 ni L2 explícita (informe, «Entrenamiento y optimización»): la capacidad la limitan la tasa "
                "baja, las pocas hojas, el mínimo por hoja, la parada temprana, el promedio de bolsas y la restricción a funciones de "
                "una variable sin pares."),
    ], titulo=f"Regularización: sin penalización explícita, la ponen {CARDINAL[len(filas)]} controles",
        subtitulo="Los hiperparámetros del modelo final de cada tarea, elegidos por la validación anidada.",
        fuentes=FUENTES_META + FUENTES_EBM[:2] + [INFORME_ENTRENAMIENTO])


def _tarjeta_supuestos():
    e2 = D.clave_valor("resultados/ablacion_interacciones_e2.csv")
    pares = int(D.json(BUSQUEDAS_E2)["modelos"]["clasificacion/ebm"]["hiperparametros"]["modelo__interactions"])
    abl = D.ablacion()
    completa = D.fila(abl, tarea="clasificacion", componente="completo")
    sin = abl[(abl["tarea"] == "clasificacion") & (abl["componente"] == "balanceo")
              & abl["variante"].str.contains(D.NOMBRE_BALANCEO["ninguno"], regex=False)].iloc[0]
    edad = M.variables().loc["age"]
    auc_con, auc_sin = ui.num(completa["metrica_prueba"], 4), ui.num(sin["metrica_prueba"], 4)
    iguales = f"{auc_con} en los dos casos" if auc_con == auc_sin else f"{auc_con} frente a {auc_sin}"
    filas = [
        ["Aditividad (sin interacciones en el modelo final)", "el efecto de una variable no depende de las demás",
         f"Se sostiene: la búsqueda eligió K = {F.entero(_hp('clasificacion')['interactions'])} en las dos tareas, y quitar los "
         f"{F.entero(pares)} pares de la EBM de la comparación con la entrega anterior cuesta {ui.num(e2['auc_ebm'] - e2['auc_gam'], 4)} "
         f"de AUC (DeLong, {F.p_valor(e2['p_delong'], 2)})."],
        ["Funciones escalonadas por cuantiles", "cada efecto es constante dentro de un tramo",
         f"Natural aquí: la edad solo toma {F.entero(edad['n_distintos'])} valores y la satisfacción actúa por escalones."],
        ["Enlace logit y pérdida ponderada", "los pesos cambian el nivel del puntaje, no el orden de los clientes",
         f"Se sostiene: sin ponderación el AUC de prueba es el mismo ({iguales}); cambia la captura con umbral "
         f"{ui.num(UMBRAL_MEDIO, 1)} (de {ui.num(sin['recall_05'], 3)} a {ui.num(completa['recall_05'], 3)})."],
        ["Entrenamiento y prueba de la misma distribución", "lo aprendido vale para clientes como los del entrenamiento",
         f"Partición aleatoria {ui.num(100 * (1 - M.meta()['fraccion_prueba']))}/{ui.num(100 * M.meta()['fraccion_prueba'])} con semilla "
         f"{M.meta()['semilla']}, no por posición: el archivo está ordenado por cohortes. Hay un solo conjunto de datos y es sintético."],
        ["Lectura asociativa, no causal", "una curva describe lo que el modelo asigna, no el efecto de cambiar la variable",
         "No se puede comprobar con estos datos: las curvas no se leen como palancas."],
    ]
    return ui.tarjeta([ui.tabla(["Supuesto", "Qué implica", "En estos datos"], filas, compacta=True, num=[False, False, False])],
                      titulo="Supuestos y si se sostienen aquí",
                      subtitulo="Lo que el modelo da por cierto y la evidencia de cada supuesto en COFINFAD.",
                      fuentes=[D.fuente("resultados/ablacion_interacciones_e2.csv"), D.fuente("resultados/ablacion.csv"),
                               D.fuente(BUSQUEDAS_E2), D.fuente("variables.csv"), D.fuente("metadatos.json")])


def _seccion_formulacion():
    ebm = M.meta()["ebm"]["clasificacion"]
    k = {t: _hp(t)["interactions"] for t in TAREAS}
    pares = "ningún par" if all(v == 0 for v in k.values()) else f"{F.entero(max(k.values()))} pares como mucho"
    titulo = f"La predicción es el intercepto más una función escalonada por columna: {F.entero(ebm['terminos'])} términos y {pares}"
    return ui.seccion(titulo, [
        ui.rejilla([_tarjeta_modelo(), _tarjeta_identificabilidad()], "1-1"),
        ui.rejilla([_tarjeta_perdidas(), _tarjeta_aprendizaje()], "1-1"),
        _tarjeta_regularizacion(),
        _tarjeta_supuestos(),
    ], subtitulo="La formulación del informe (sección «Modelo propuesto»), con los valores de los dos modelos finales.")


# --- 3. Pseudocódigo -----------------------------------------------------------------------------------------------------
def _li(texto, hijos=None, **valores):
    contenido = [dcc.Markdown(_pegar(_sust(texto, **valores)), mathjax=True)]
    if hijos:
        contenido.append(html.Ol([h for h in hijos]))
    return html.Li(contenido)


def _seccion_algoritmo():
    hc, hr = _hp("clasificacion"), _hp("regresion")
    baja, alta = _fidelidad()
    e = _entrenamiento()
    corrida = _corrida("clasificacion")
    fi = D.clave_valor("resultados/fidelidad_resumen.csv")
    version = V.metadata("clasificacion", "ebm")["versiones"]["interpret"]
    valores = dict(n=_tex(e["n"]), p=_tex(M.meta()["ebm"]["clasificacion"]["terminos"]),
                   eta_c=_tex(hc["learning_rate"], _decimales(hc["learning_rate"])), L_c=_tex(hc["max_leaves"]),
                   m_c=_tex(hc["min_samples_leaf"]), B_c=_tex(hc["outer_bags"]), K_c=_tex(hc["interactions"]),
                   eta_r=_tex(hr["learning_rate"], _decimales(hr["learning_rate"])), L_r=_tex(hr["max_leaves"]),
                   m_r=_tex(hr["min_samples_leaf"]), B_r=_tex(hr["outer_bags"]), K_r=_tex(hr["interactions"]),
                   baja=_tex(baja), alta=_tex(alta))
    entradas = [
        _md(r"**Entradas:** el entrenamiento $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ ($n = «n»$ clientes, $p = «p»$ columnas); los pesos "
            r"$w_i$ (1, o $n / (2\, n_{z_i})$ con ponderación); las bolsas $B$; la tasa $\eta$; las hojas $L$; el mínimo por hoja $m$; "
            r"los pares $K$; los tramos $\mathrm{bins}$.", **valores),
        _md(r"**Valores del modelo final:** en clasificación $\eta = «eta_c»$, $L = «L_c»$, $m = «m_c»$, $B = «B_c»$ y $K = «K_c»$; en "
            r"regresión $\eta = «eta_r»$, $L = «L_r»$, $m = «m_r»$, $B = «B_r»$ y $K = «K_r»$.", **valores),
        _md(r"**Salidas:** el intercepto $\beta_0$ y las funciones de forma $f_j$ (y $f_{jk}$ si $K > 0$).", **valores),
    ]
    pasos = html.Ol([
        _li(r"Discretizar cada $x_j$ en a lo sumo $\mathrm{bins}$ tramos por cuantiles (una sola vez)."),
        _li(r"Para $b = 1, \dots, B$:", [
            _li(r"tomar una submuestra del entrenamiento y reservar el resto como validación de la bolsa;"),
            _li(r"$f_j^{(b)} \gets 0$ para todo $j$ y $\hat{u}_i \gets \beta_0$ (el logit de la media ponderada, o la media);"),
            _li(r"mientras la pérdida ponderada en la validación de la bolsa mejore (parada temprana), para $j = 1, \dots, p$:", [
                _li(r"$\varrho_i \gets -\partial \ell(y_i, \hat{u}_i) / \partial \hat{u}_i$ para todo $i$;"),
                _li(r"ajustar un árbol $h$ de $L$ hojas y mínimo $m$ sobre $x_j$ al residuo $\varrho$, con pesos $w_i$;"),
                _li(r"$f_j^{(b)} \gets f_j^{(b)} + \eta\, h$ y $\hat{u}_i \gets \hat{u}_i + \eta\, h(x_{ij})$;"),
            ]),
            _li(r"si $K > 0$, elegir con FAST los $K$ pares que más reducen el residuo y repetir el ciclo sobre ellos con los "
                r"$f_j^{(b)}$ fijos (en los dos modelos finales $K = «K_c»$: no se ejecuta).", **valores),
        ]),
        _li(r"$f_j \gets \frac{1}{B}\sum_b f_j^{(b)}$ y $f_{jk} \gets \frac{1}{B}\sum_b f_{jk}^{(b)}$, centradas; "
            r"$\beta_0 \gets$ la media de las predicciones."),
        _li(r"**Búsqueda:** evaluar cada configuración con $B = «baja»$ en la validación interna. **Modelo final:** reajustar la "
            r"ganadora con $B = «alta»$.", **valores),
    ])
    parada = _md(r"**Criterio de parada:** cada bolsa se detiene cuando la pérdida ponderada en su validación deja de mejorar; la "
                 r"búsqueda, al agotar su presupuesto: los «presupuesto» puntos de la rejilla en cada uno de los «k» pliegues externos "
                 r"(«evals» evaluaciones por corrida). **Costo:** $O(B \cdot M \cdot p \cdot (n + \mathrm{bins}))$ el ajuste y "
                 r"$O(p + K)$ cada predicción (sección de complejidad).",
                 presupuesto=F.entero(corrida["presupuesto"]), k=F.entero(corrida["k_ext"]), evals=F.entero(corrida["n_evaluaciones"]))
    algoritmo = ui.tarjeta([
        html.Div(entradas, className="resultados"),
        ui.desplegable("Pasos del algoritmo", [pasos], abierto=True, llano=True),
        parada,
    ], titulo="Algoritmo 1. Entrenamiento de la EBM con bolsas externas, pesos de clase y multi-fidelidad",
        subtitulo="El del informe, con los valores de los modelos finales.",
        fuentes=FUENTES_META + [D.fuente("entrega3/corridas.csv"), INFORME_MODELO])

    fidelidad = ui.aviso([
        f"Es una multi-fidelidad de dos niveles, no Successive Halving: todas las configuraciones se evalúan con B = {F.entero(baja)} "
        f"y solo la ganadora se reajusta con B = {F.entero(alta)}. Successive Halving haría rondas: en cada una se quedaría con una "
        "fracción 1/η_SH de las configuraciones (η_SH es su factor de reducción, que no tiene que ver con la tasa η) y subiría la "
        "fidelidad de las que siguen. Aquí no hubo "
        "rondas de descarte.",
        f"El supuesto que la sostiene se comprobó: en {F.entero(fi['configuraciones'])} configuraciones al azar, la correlación de "
        f"Spearman entre el AUC con B = {F.entero(baja)} y con B = {F.entero(alta)} es {ui.num(fi['rho_spearman'], 2)}, la ganadora es "
        f"{'la misma' if str(fi['misma_ganadora']) == 'True' else 'otra'} y la fidelidad baja ahorra el "
        f"{F.pct(fi['ahorro_tiempo'], 0)} del tiempo de búsqueda.",
    ], None, "La búsqueda multi-fidelidad, con precisión")
    libreria = ui.aviso([
        f"El esquema de arriba es el del informe y el de {_cita('lou2012')}: un ciclo por las columnas en orden. La librería con que se "
        f"entrenó el modelo (interpret-core {version}), con sus valores por defecto, que el proyecto no cambió, lo modifica en tres "
        "puntos que no cambian la forma del modelo ni la pérdida que se reduce:",
        "1) empieza con unas rondas de suavizado con cortes al azar; 2) después alterna ciclos que solo estiman cuánto reduciría la "
        "pérdida cada columna con pasos codiciosos sobre la de mayor ganancia; 3) su parada temprana mira una ventana de pasos sin "
        "mejora y devuelve el mejor modelo de la bolsa.",
        "Se verificó en los parámetros de los modelos guardados (artifacts/<tarea>/ebm/modelo.joblib) y en la documentación de la "
        "librería; el informe no lo describe. La propiedad de la sección siguiente vale para cualquier orden de las columnas.",
    ], "atencion", "Lo que la librería hace distinto de este esquema")
    return ui.seccion(f"El algoritmo: boosting de una columna a la vez, en {F.entero(hc['outer_bags'])} bolsas y con parada temprana", [
        algoritmo,
        ui.rejilla([fidelidad, libreria], "1-1"),
    ], subtitulo="Pseudocódigo formal: entradas, salidas, pasos y criterio de parada.",
        fuentes=[D.fuente("resultados/fidelidad_resumen.csv"), D.fuente("entrega3/corridas.csv")] + FUENTES_META)


# --- 4. Sustento teórico -------------------------------------------------------------------------------------------------
EC_PROPIEDAD = r"""\begin{aligned}
L(u + \eta h) &\le L(u) - \eta\Big(1 - \frac{\eta\, c}{2}\Big) \sum_{i} w_i\, h(x_{ij})^2 \\
&\le L(u)
\end{aligned}"""
EC_TAYLOR = r"\ell(u_i + \eta h_i) \le \ell(u_i) + \eta\, h_i\, \ell'(u_i) + \frac{c}{2}\, \eta^2 h_i^2"
EC_SUMA = r"""\begin{aligned}
L(u + \eta h) &\le L(u) - \eta \sum_i w_i\, \varrho_i h_i \\
&\quad + \frac{c}{2}\, \eta^2 \sum_i w_i\, h_i^2
\end{aligned}"""
EC_HOJA = r"""\begin{aligned}
h_A &= \frac{\sum_{i \in A} w_i \varrho_i}{\sum_{i \in A} w_i} \\
\Rightarrow\ \sum_{i \in A} w_i \varrho_i\, h_A &= h_A^2 \sum_{i \in A} w_i
\end{aligned}"""


def _seccion_teoria():
    hc, hr = _hp("clasificacion"), _hp("regresion")
    e = _entrenamiento()
    eta = hc["learning_rate"]
    cota_c, cota_r = 2 / C_LOGIT, 2 / C_CUADRATICA
    newton = cota_c * e["min_q"]
    con = D.fila(D.tabla("resultados/concordancia_validacion_prueba.csv"), tarea="clasificacion")
    ebm = D.fila(D.prueba("clasificacion"), modelo="ebm")
    propiedad = ui.tarjeta([
        _md(r"**Propiedad (descenso por paso).** Sea $L(u) = \sum_i w_i\, \ell(y_i, u_i)$, con $\ell$ convexa y dos veces derivable "
            r"en $u$ y $\ell'' \le c$. Sea $h$ un árbol sobre una sola columna cuyas hojas valen la media ponderada del gradiente "
            r"negativo $\varrho_i$ de sus clientes (el ajuste por mínimos cuadrados de la ecuación de aprendizaje). Entonces, para "
            r"$0 < \eta \le 2/c$,"),
        _ec(EC_PROPIEDAD),
        _md(r"con igualdad solo si $h = 0$. Como la pérdida no es negativa, la sucesión de pérdidas de entrenamiento es no creciente y "
            r"acotada: converge (a un valor que no tiene por qué ser el mínimo entre todas las sumas de funciones escalonadas)."),
        _md(r"**Supuestos:** pérdida convexa con segunda derivada acotada y hojas iguales a la media ponderada del gradiente. No "
            r"depende del orden de las columnas (vale igual para un ciclo que para un paso codicioso) ni de cómo se eligen los cortes."),
    ], titulo="La propiedad", subtitulo="Lo que se puede afirmar con precisión sobre el principio de aprendizaje.",
        fuentes=[INFORME_MODELO])
    demostracion = ui.tarjeta([
        _md(r"Por el teorema de Taylor con la cota $\ell'' \le c$, para cada cliente:"),
        _ec(EC_TAYLOR),
        _md(r"Se suma con los pesos y se usa $\ell'(u_i) = -\varrho_i$:"),
        _ec(EC_SUMA),
        _md(r"En cada hoja $A$ el árbol vale la media ponderada del gradiente, así que"),
        _ec(EC_HOJA),
        _md(r"y, sumando sobre las hojas, $\sum_i w_i \varrho_i h_i = \sum_i w_i h_i^2$. Al sustituir queda la desigualdad de la "
            r"propiedad. $\blacksquare$"),
        ui.nota(f"Es la desigualdad del descenso de la optimización convexa aplicada a un paso del algoritmo; el informe no trae esta "
                f"demostración. Ajustar un árbol al gradiente negativo es el boosting de gradiente con que {_cita('lou2012')} aprenden "
                "las funciones de forma."),
    ], titulo="Demostración", subtitulo="Tres líneas: Taylor, la suma ponderada y el valor de cada hoja.", fuentes=[INFORME_MODELO])
    filas = [[D.NOMBRE_TAREA["clasificacion"], "log-loss en el logit: ℓ'' = p(1 − p)", ui.num(C_LOGIT, 2), ui.num(cota_c, 0),
              _valor(eta), f"{F.entero(cota_c / eta)} veces"],
             [D.NOMBRE_TAREA["regresion"], "(s − u)²: ℓ'' constante", ui.num(C_CUADRATICA, 0), ui.num(cota_r, 0),
              _valor(hr["learning_rate"]), f"{F.entero(cota_r / hr['learning_rate'])} veces"]]
    constantes = ui.tarjeta([
        ui.tabla(["Tarea", "Pérdida", "c", "Cota 2/c", "η del modelo final", "Margen"], filas, compacta=True,
                 num=[False, False, True, True, True, True]),
        _md(r"Si la librería calcula el valor de cada hoja con la segunda derivada (un paso de Newton, $h_A = G_A / H_A$, como sugiere "
            r"su parámetro de hessiana mínima), el mismo argumento da descenso si $\eta \le (2/c)\, \bar{q}_A$, con $\bar{q}_A$ la "
            r"media de $p(1 - p)$ en la hoja. En el modelo final, el menor $p(1 - p)$ de los «n» clientes de entrenamiento es «q»: "
            r"la condición pide $\eta \le «newton»$ y se cumple con $\eta = «eta»$. En regresión la segunda derivada es constante y "
            r"los dos pasos coinciden. No se verificó en el núcleo compilado de la librería cuál de los dos usa, y el rango de "
            r"$p(1-p)$ es el del modelo final, no el de cada paso: es evidencia de plausibilidad, no una prueba.",
            n=F.entero(e["n"]), q=ui.num(e["min_q"], 4), newton=_tex(newton, 3), eta=_tex(eta, _decimales(eta))),
    ], titulo="Las constantes en estos datos", subtitulo="La condición de la propiedad se cumple con mucho margen en las dos tareas.",
        fuentes=FUENTES_META + [D.fuente("eda/entrenamiento.parquet"), D.fuente("modelos/ebm_clasificacion.json")])
    limites = ui.tarjeta([
        ui.parrafo(f"La propiedad es sobre la pérdida de entrenamiento. Que el modelo generalice lo controlan la parada temprana (cada "
                   f"bolsa se detiene cuando la pérdida en su validación deja de bajar), el promedio de {F.entero(hc['outer_bags'])} bolsas "
                   "y la restricción a funciones de una variable."),
        ui.parrafo(f"La evidencia de que generaliza es empírica: en {F.entero(con['corridas'])} corridas de clasificación, la "
                   f"correlación entre el AUC de la validación anidada y el de prueba es {ui.num(con['correlacion'], 3)} (diferencia "
                   f"media {ui.con_signo(con['diferencia_media'], 4)}), y el AUC de prueba de la EBM, {ui.num(ebm['auc_prueba'], 4)}, "
                   f"queda dentro de una desviación de su AUC externa ({ui.num(ebm['auc_ext_media'], 4)} ± {ui.num(ebm['auc_ext_desv'], 4)})."),
    ], titulo="Lo que no garantiza", subtitulo="Bajar la pérdida de entrenamiento no es generalizar.",
        fuentes=[D.fuente("resultados/concordancia_validacion_prueba.csv"), D.fuente("resultados/clasificacion_prueba.csv")])
    titulo = (f"Por qué el entrenamiento converge: cada paso baja la pérdida si η ≤ 2/c, y aquí η = {_valor(eta)} frente a cotas de "
              f"{ui.num(cota_c, 0)} y {ui.num(cota_r, 0)}")
    return ui.seccion(titulo, [ui.rejilla([propiedad, demostracion], "1-1"), ui.rejilla([constantes, limites], "2-1")],
                      subtitulo="Sustento teórico: una propiedad del paso de boosting, con sus supuestos y su plausibilidad en estos datos.")


# --- 5. Complejidad ------------------------------------------------------------------------------------------------------
EC_COSTO = r"""\begin{aligned}
T_{\text{ajuste}} &= O\big(B \cdot M \cdot p \cdot (n + \mathrm{bins})\big) \\
T_{\text{inferencia}} &= O(p + K) \ \text{por cliente} \\
|\theta| &= O(p \cdot \mathrm{bins} + K \cdot \mathrm{bins}^2)
\end{aligned}"""
MODELOS_TIEMPO = [("ebm", "linea"), (RIVAL, "linea"), ("random_forest", "punteada"), ("logistica", "trazos")]


def _curva_tiempo():
    t = D.tabla("resultados/curva_aprendizaje.csv").pivot(index="modelo", columns="n", values="entrenamiento_s")
    return t


def _pendiente(serie):
    """Pendiente de log(tiempo) frente a log(n) desde N_DESDE: 1 es crecimiento lineal."""
    s = serie[serie.index >= N_DESDE]
    return float(np.polyfit(np.log(s.index.to_numpy(dtype=float)), np.log(s.to_numpy(dtype=float)), 1)[0])


def _grafico_tiempo():
    t = _curva_tiempo()
    ns = [int(n) for n in t.columns]
    fig = go.Figure()
    for m, forma in MODELOS_TIEMPO[::-1]:                   # la EBM, al final: queda encima
        fig.add_trace(go.Scatter(x=ns, y=t.loc[m], name=D.NOMBRE_MODELO[m], **ui.traza(D.ROL_MODELO[m], forma, marcadores=(m == "ebm")),
                                 **ui.hover_texto([f"{D.NOMBRE_MODELO[m]} con {F.entero(n)} clientes: <b>{ui.num(v, 2 if v < 10 else 1)}</b> s"
                                                   for n, v in zip(ns, t.loc[m])])))
    lo, hi = float(t.loc[[m for m, _f in MODELOS_TIEMPO]].to_numpy().min()), float(t.loc[[m for m, _f in MODELOS_TIEMPO]].to_numpy().max())
    marcas = [10.0 ** k for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]
    fig.update_xaxes(type="log", tickvals=ns, ticktext=[F.entero(n) for n in ns], range=[math.log10(ns[0]) - 0.05, math.log10(ns[-1]) + 0.05])
    fig.update_yaxes(type="log", tickvals=marcas, ticktext=[ui.num(v, max(0, -int(math.log10(v)))) for v in marcas],
                     range=[math.log10(marcas[0]), math.log10(marcas[-1])])
    ui.unidad_eje(fig, "segundos de ajuste (escala logarítmica)")
    fig.update_layout(hovermode="closest")
    pend = {m: _pendiente(t.loc[m]) for m, _f in MODELOS_TIEMPO}
    ebm = t.loc["ebm"]
    titulo = (f"De {F.entero(N_DESDE)} a {F.entero(ns[-1])} clientes, el ajuste de la EBM crece casi en proporción a n: pendiente "
              f"{ui.num(pend['ebm'], 2)} en escala logarítmica")
    sub = (f"Segundos de ajuste según los clientes de entrenamiento, con los hiperparámetros finales, en escala logarítmica en los dos "
           f"ejes: una pendiente de 1 es crecimiento lineal. Por debajo de {F.entero(N_DESDE)} clientes la EBM no baja de "
           f"{ui.num(ebm.min(), 1)} s: domina lo que no depende de n.")
    nota = (f"Pendientes desde {F.entero(N_DESDE)} clientes: " + "; ".join(f"{D.NOMBRE_MODELO[m]}, {ui.num(pend[m], 2)}" for m, _f in MODELOS_TIEMPO)
            + f". Son los reajustes de la curva de aprendizaje; el ajuste del modelo final, medido aparte como mediana de tres, tarda "
              f"{ui.num(_tiempos('clasificacion').loc['ebm', 'ajuste_s'], 1)} s. La cota supone M fijo; M lo decide la parada temprana "
              "de cada bolsa y no quedó registrado en datos/.")
    filas = [[F.entero(n)] + [ui.num(t.loc[m, n], 3) for m, _f in MODELOS_TIEMPO] for n in t.columns]
    tabla = (["Clientes"] + [D.NOMBRE_MODELO[m] for m, _f in MODELOS_TIEMPO], filas)
    leyenda = [(D.NOMBRE_MODELO["ebm"], "modelo", "linea"), (D.NOMBRE_MODELO[RIVAL], "rival", "linea"),
               (D.NOMBRE_MODELO["random_forest"], "rival", "punteada"), (D.NOMBRE_MODELO["logistica"], "base", "trazos")]
    etiqueta = f"{titulo}. " + "; ".join(f"{D.NOMBRE_MODELO[m]}: de {ui.num(t.loc[m].iloc[0], 2)} a {ui.num(t.loc[m].iloc[-1], 2)} s"
                                         for m, _f in MODELOS_TIEMPO) + "."
    return ui.grafico(fig, titulo, sub, etiqueta, tabla=tabla, leyenda=leyenda, nota=nota, id=f"{PRE}-tiempo",
                      fuentes=[D.fuente("resultados/curva_aprendizaje.csv"), D.fuente("entrega3/computo/tiempos_finales.csv")])


def _fichas_costo():
    tc, tr = _tiempos("clasificacion"), _tiempos("regresion")
    costo = _costo("clasificacion")
    n_te = int(tc.loc["ebm", "n_test"])
    por = {m: tc.loc[m, "inferencia_s"] / n_te * MICRO for m in tc.index}
    tramos = _tramos("clasificacion")
    cota = sum(1 for x in tramos if x > 2) * _bins() + sum(2 for x in tramos if x <= 2)
    return [
        ui.cifra("Ajuste del modelo final de clasificación", ui.num(tc.loc["ebm", "ajuste_s"], 1), "s",
                 variacion=f"{ui.num(tc.loc['ebm', 'ajuste_s'] / tc.loc[RIVAL, 'ajuste_s'], 0)} veces lo que tarda {D.en_frase(RIVAL)}",
                 nota=f"En regresión, {ui.num(tr.loc['ebm', 'ajuste_s'], 1)} s. Mediana de tres ajustes con los {F.entero(tc.loc['ebm', 'n_train'])} "
                      "clientes de entrenamiento.", mini=True),
        ui.cifra("Predicción de un cliente", ui.num(por["ebm"], 1), "µs",
                 variacion=f"{D.NOMBRE_MODELO[RIVAL]}: {ui.num(por[RIVAL], 1)} µs; k-NN: {ui.num(por['knn'], 0)} µs",
                 nota=f"{ui.num(tc.loc['ebm', 'inferencia_s'], 3)} s para los {F.entero(n_te)} clientes de prueba: "
                      f"{F.entero(M.meta()['ebm']['clasificacion']['terminos'])} búsquedas en tablas por cliente.", mini=True),
        ui.cifra("Valores guardados", F.entero(costo.loc["ebm", "parametros"]), f"de {F.entero(cota)} posibles",
                 variacion=f"{ui.num(costo.loc['ebm', 'tamano_mb'], 1)} MB; {D.NOMBRE_MODELO['random_forest']}: "
                           f"{ui.num(costo.loc['random_forest', 'tamano_mb'], 0)} MB",
                 nota=f"La cota con bins = {F.entero(_bins())} y K = 0: las columnas numéricas por bins y las de dos valores por 2.", mini=True),
    ]


def _tarjeta_inferencia():
    tc = _tiempos("clasificacion")
    n_te = int(tc.loc["ebm", "n_test"])
    h = {m: V.metadata("clasificacion", m)["hiperparametros"] for m in ("xgboost", "random_forest", "arbol")}
    p = M.meta()["ebm"]["clasificacion"]["terminos"]
    orden = {"ebm": f"O(p + K): {F.entero(p)} búsquedas en tablas",
             "xgboost": f"O(T · d): T = {F.entero(h['xgboost']['modelo__n_estimators'])} árboles, d = {F.entero(h['xgboost']['modelo__max_depth'])}",
             "random_forest": f"O(T · d): T = {F.entero(h['random_forest']['modelo__n_estimators'])}, d ≤ {F.entero(h['random_forest']['modelo__max_depth'])}",
             "arbol": f"O(d): d ≤ {F.entero(h['arbol']['modelo__max_depth'])}",
             "logistica": f"O(p): {F.entero(p)} columnas",
             "knn": f"O(n · p): compara con los {F.entero(tc.loc['knn', 'n_train'])} clientes de entrenamiento"}
    modelos = [m for m in D.ORDEN_MODELOS["clasificacion"] if m in orden]
    filas = [[D.NOMBRE_MODELO[m], orden[m], ui.num(tc.loc[m, "inferencia_s"], 3), ui.num(tc.loc[m, "inferencia_s"] / n_te * MICRO, 1)]
             for m in modelos]
    return ui.tarjeta([
        _ec(EC_COSTO),
        _md(r"$M$ son las rondas hasta la parada temprana y $|\theta|$, los valores que guarda el modelo; aquí $p = «p»$ columnas, "
            r"$n = «n»$ clientes, $B = «B»$ bolsas y $K = «K»$. La inferencia no depende de $n$: es una búsqueda por columna.",
            p=_tex(p), n=_tex(tc.loc["ebm", "n_train"]),
            B=_tex(_hp("clasificacion")["outer_bags"]), K=_tex(_hp("clasificacion")["interactions"])),
        ui.tabla(["Modelo", "Orden de una predicción", f"Los {F.entero(n_te)} de prueba (s)", "Por cliente (µs)"], filas, compacta=True,
                 muestras=[D.ROL_MODELO[m] for m in modelos], num=[False, False, True, True]),
    ], titulo="Las cotas y lo medido en inferencia",
        subtitulo="Tiempo de inferencia de cada modelo final de clasificación, mediana de cinco mediciones.",
        fuentes=FUENTES_COSTO + [f"artifacts/clasificacion/{m}/metadata.json" for m in ("xgboost", "random_forest", "arbol")] + [INFORME_MODELO])


def _seccion_complejidad():
    tc = _tiempos("clasificacion")
    por = tc.loc["ebm", "inferencia_s"] / tc.loc["ebm", "n_test"] * MICRO
    titulo = (f"Ajustar cuesta O(B·M·p·(n{F.FIJO}+{F.FIJO}bins)) y predecir, O(p{F.FIJO}+{F.FIJO}K) por cliente: "
              f"{ui.num(tc.loc['ebm', 'ajuste_s'], 1)}{F.FIJO}s de "
              f"ajuste y {ui.num(por, 1)} µs por cliente")
    return ui.seccion(titulo, [
        ui.rejilla([_grafico_tiempo(), ui.cifras(_fichas_costo())], "2-1"),
        _tarjeta_inferencia(),
    ], subtitulo="La complejidad del entrenamiento y de la inferencia, contrastada con los tiempos medidos en un solo equipo.",
        fuentes=FUENTES_COSTO + [D.fuente("resultados/sensibilidad_hiperparametros.csv")])


# --- 6. Qué es nuevo y diferencias con la literatura ---------------------------------------------------------------------
def _seccion_novedad():
    version = V.metadata("clasificacion", "ebm")["versiones"]["interpret"]
    baja, alta = _fidelidad()
    verif = M.meta()["verificacion"]
    dif = max(verif[f"ebm_{t}"]["max_dif_motor_frente_al_modelo_guardado"] for t in TAREAS)
    filas_verif = verif["ebm_clasificacion"]["filas_comparadas"]
    reutiliza = ui.tarjeta([html.Ul([
        html.Li(f"La EBM tal como la publican {_cita('lou2012')} y {_cita('lou2013')}, y su implementación interpret-core "
                f"({_cita_en('nori2019')}), versión {version}: el entrenamiento es el de la librería."),
        html.Li("El conjunto de datos y las decisiones de depuración de las entregas anteriores (columnas eliminadas y variables "
                "transaccionales derivadas), revalidadas con la partición hecha antes de cualquier estadístico del objetivo."),
        html.Li("El pipeline combinatorio y el protocolo estadístico de la guía del curso."),
    ])], titulo="Se reutiliza", fuentes=["Informe final: metodologia.tex, «Qué es nuevo y qué se reutiliza»"])
    propio = ui.tarjeta([html.Ul([
        html.Li("La elección del modelo para este problema (no se vio en el curso) y su justificación con el análisis exploratorio."),
        html.Li("La ponderación de clases por observación, que la librería no trae como opción: un envoltorio le pasa los pesos "
                "w_i = n / (2 n_z) de cada cliente."),
        html.Li(f"La búsqueda multi-fidelidad de dos niveles: {_igual('B', F.entero(baja))} en la búsqueda y "
                f"{_igual('B', F.entero(alta))} en el modelo final."),
        html.Li("La evaluación con las mismas particiones, el mismo preprocesamiento, el mismo presupuesto y las mismas pruebas que los "
                "modelos de referencia, para que la comparación sea justa."),
        html.Li(f"Para esta app, un motor en numpy que lee las tablas exportadas del modelo y lo evalúa sin la librería: diferencia "
                f"máxima con el modelo guardado de {_cientifico(dif)} en {F.entero(filas_verif)} clientes."),
    ])], titulo="Lo propio de este trabajo", fuentes=[D.fuente("metadatos.json"), D.fuente("entrega3/corridas.csv"),
                                                     "Informe final: metodologia.tex, «Adaptaciones»"])
    return ui.seccion("Qué es nuevo: la elección, la adaptación y la evaluación; no el algoritmo", [
        ui.rejilla([reutiliza, propio], "1-1"),
        ui.aviso("No se afirma una novedad algorítmica ni una implementación propia del algoritmo: la EBM es la de Lou et al. y se "
                 "entrena con interpret-core.", None, "Lo que no se afirma"),
        _tarjeta_literatura(),
    ], subtitulo="El párrafo «Qué es nuevo y qué se reutiliza» del informe, y la EBM frente a los métodos afines de cada familia.")


def _metodo(nombre, familia):
    """Celda del método: el nombre y, debajo y en gris, su familia."""
    return [html.Span(nombre), html.Br(), html.Span(familia, className="tinta-3")]


def _tarjeta_literatura():
    clf = D.prueba("clasificacion").set_index("modelo")
    e2 = D.clave_valor("resultados/ablacion_interacciones_e2.csv")
    pares = int(D.json(BUSQUEDAS_E2)["modelos"]["clasificacion/ebm"]["hiperparametros"]["modelo__interactions"])

    def auc(m):
        return ui.num(clf.loc[m, "auc_prueba"], 4)

    filas = [
        [_metodo(D.NOMBRE_MODELO["logistica"], "lineal"), "β₀ + Σ βⱼ xⱼ en el logit", "máxima verosimilitud con penalización L1 o L2",
         "no", "sí: un coeficiente por columna", auc("logistica"), _cita_corta("pedregosa2011")],
        [_metodo(D.NOMBRE_MODELO["svm"], "lineal"), "margen w · x + b", "pérdida de bisagra con penalización", "no",
         "sí: un peso por columna", auc("svm"), _cita_corta("cortes1995")],
        [_metodo("GAM con splines", "aditiva"), "β₀ + Σ fⱼ(xⱼ), fⱼ suaves", "mínimos cuadrados penalizados", "no",
         "sí: una curva suave por variable", RAYA, _cita_corta("lou2012")],
        [_metodo("EBM sin pares (la de este trabajo)", "aditiva"), "β₀ + Σ fⱼ(xⱼ), fⱼ escalonadas", "boosting por columna y bolsas",
         "no", "sí: una curva escalonada por variable", auc("ebm"), _cita_corta("lou2012")],
        [_metodo("GA²M (EBM con pares)", "aditiva con pares"), "+ Σ fⱼₖ(xⱼ, xₖ)", "boosting y FAST para elegir los pares", "K pares",
         "sí: curvas y superficies de dos variables", f"{ui.num(e2['auc_ebm'], 4)}*",
         f"{_cita_corta('lou2013')}; {_cita_corta('nori2019')}"],
        [_metodo(D.NOMBRE_MODELO["arbol"], "árboles"), "una partición del espacio", "división recursiva codiciosa (CART)",
         "todas, implícitas", "rama por rama, si es poco profundo", auc("arbol"), _cita_corta("breiman1984")],
        [_metodo(D.NOMBRE_MODELO["random_forest"], "ensamble de árboles"), "promedio de T árboles", "bagging con columnas al azar",
         "todas", "no: pide un explicador", auc("random_forest"), _cita_corta("breiman2001")],
        [_metodo(D.NOMBRE_MODELO[RIVAL], "ensamble de árboles"), "suma de T árboles", "boosting de gradiente regularizado",
         "hasta la profundidad", "no: pide un explicador", auc(RIVAL), _cita_corta("chen2016")],
        [_metodo(f"{D.NOMBRE_MODELO[RIVAL]} con SHAP", "ensamble y explicador"), "la de XGBoost, repartida entre las variables",
         "explicación posterior al ajuste", "las del ensamble", "no: SHAP es el explicador, no la estructura del modelo", RAYA,
         _cita_corta("lundberg2017")],
    ]
    tabla = ui.tabla(["Método", "Forma de la predicción", "Cómo aprende", "Interacciones", "Se lee sin explicador", "AUC de prueba aquí",
                      "Referencia"], filas, compacta=True, num=[False] * 5 + [True, False], ocultar_movil=(2, 3))
    nota = (f"Los AUC son los de la prueba de este trabajo (etiqueta del percentil {F.entero(M.meta()['etiqueta']['q'] * 100)}). *El "
            f"de GA²M viene de la ablación, con la etiqueta de la entrega anterior: {ui.num(e2['auc_ebm'], 4)} con {F.entero(pares)} "
            f"pares y {ui.num(e2['auc_gam'], 4)} sin ellos; no se compara con los demás. El GAM con splines no se evaluó aquí; "
            f"{_cita('lou2012')} lo comparan con las funciones aprendidas con árboles. Todas las referencias son las del informe final.")
    return ui.tarjeta([tabla, ui.nota(nota)],
                      titulo=f"La EBM frente a {CARDINAL[len(filas) - 1]} métodos afines",
                      subtitulo="Qué cambia en la forma de la predicción, en cómo se aprende y en si se puede leer sin un explicador.",
                      fuentes=[D.fuente("resultados/clasificacion_prueba.csv"), D.fuente("resultados/ablacion_interacciones_e2.csv"),
                               D.fuente(BUSQUEDAS_E2), "Informe final: referencias.bib"])


# --- 7. Curva de aprendizaje y de convergencia ---------------------------------------------------------------------------
def _curva_aprendizaje():
    return D.tabla("resultados/curva_aprendizaje.csv").pivot(index="modelo", columns="n", values="auc")


def _grafico_aprendizaje():
    t = _curva_aprendizaje()
    ns = list(t.columns)
    cats = [F.entero(v) for v in ns]
    orden = [m for m in D.ORDEN_MODELOS["clasificacion"] if m in t.index]
    fig = go.Figure()
    for m in orden[::-1]:
        fig.add_trace(go.Scatter(x=cats, y=t.loc[m], name=D.NOMBRE_MODELO[m], **ui.traza(D.ROL_MODELO[m], marcadores=(m == "ebm")),
                                 **ui.hover(D.NOMBRE_MODELO[m], t.loc[m], d=4)))
    lo = math.floor(t.to_numpy().min() / PASO_AUC) * PASO_AUC
    hi = math.ceil(t.to_numpy().max() / PASO_AUC) * PASO_AUC
    vals, marcas = ui.marcas(lo, hi, PASO_AUC)
    ui.unidad_eje(fig, "AUC de prueba")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[lo - PASO_AUC / 5, hi + PASO_AUC / 5], tickvals=vals, ticktext=marcas)
    ebm, menor, todos = t.loc["ebm"], ns[0], ns[-1]
    desde = N_DESDE if N_DESDE in ns else menor
    supera = ebm[menor] > t.loc["logistica", todos]
    titulo = (f"Con {F.entero(menor)} clientes la EBM ya tiene más AUC que {D.en_frase('logistica')} con los {F.entero(todos)}; de "
              f"{F.entero(desde)} a {F.entero(todos)} solo gana {ui.num(ebm[todos] - ebm[desde], 4)}" if supera else
              f"La EBM gana {ui.num(ebm[todos] - ebm[menor], 4)} de AUC entre {F.entero(menor)} y {F.entero(todos)} clientes")
    sub = "AUC de prueba según los clientes con que se entrena cada modelo, con sus hiperparámetros finales."
    final = D.fila(D.prueba("clasificacion"), modelo="ebm")["auc_prueba"]
    nota = (f"Reajustes hechos para este análisis (la EBM, sobre subconjuntos al azar del entrenamiento; las referencias, del "
            f"experimento de escalamiento del Entregable 3): el último punto de la EBM da {ui.num(ebm[todos], 4)} y su modelo final, "
            f"{ui.num(final, 4)}.")
    tabla = (["Modelo"] + [f"{c} clientes" for c in cats], [[D.NOMBRE_MODELO[m]] + [ui.num(v, 4) for v in t.loc[m]] for m in orden])
    etiqueta = f"{titulo}. Con {F.entero(todos)} clientes: " + "; ".join(f"{D.NOMBRE_MODELO[m]}, {ui.num(t.loc[m, todos], 4)}" for m in orden) + "."
    return ui.grafico(fig, titulo, sub, etiqueta, tabla=tabla, nota=nota, id=f"{PRE}-apr", fuentes=FUENTES_APRENDIZAJE,
                      leyenda=[(D.NOMBRE_MODELO["ebm"], "modelo", "linea"), ("Modelos de árboles", "rival", "linea"),
                               ("Lineales, k-NN y Naive Bayes", "base", "trazos")])


def _grafico_convergencia():
    b = D.json(BUSQUEDAS_E2)
    trazas = {m: np.asarray(b["modelos"][f"clasificacion/{m}"]["traza"], dtype=float) for m in ("ebm", RIVAL)}
    fig = go.Figure()
    for m in (RIVAL, "ebm"):
        y = trazas[m]
        x = list(range(1, len(y) + 1))
        estilo = ui.traza(D.ROL_MODELO[m], "linea")
        estilo["line"] = dict(estilo["line"], shape="hv")
        fig.add_trace(go.Scatter(x=x, y=y, name=D.NOMBRE_MODELO[m], **estilo,
                                 **ui.hover_texto([f"{D.NOMBRE_MODELO[m]} tras {k} evaluaciones: <b>{ui.num(v, 4)}</b>" for k, v in zip(x, y)])))
    todos = np.concatenate(list(trazas.values()))
    lo = math.floor(todos.min() / PASO_TRAZA) * PASO_TRAZA
    hi = math.ceil(todos.max() / PASO_TRAZA) * PASO_TRAZA
    vals, marcas = ui.marcas(lo, hi, PASO_TRAZA)
    largo = max(len(y) for y in trazas.values())
    vx, mx = ui.marcas(0, largo, ui.paso_redondo(largo, 4))
    ui.unidad_eje(fig, "mejor AUC interno hasta esa evaluación")
    fig.update_xaxes(range=[0.5, largo + 0.5], tickvals=vx, ticktext=mx)
    fig.update_yaxes(range=[lo, hi], tickvals=vals, ticktext=marcas)
    fig.update_layout(hovermode="closest")
    y = trazas["ebm"]
    ultima = max(i for i in range(len(y)) if i == 0 or y[i] > y[i - 1]) + 1
    titulo = (f"La búsqueda de la EBM deja de mejorar en la evaluación {F.entero(ultima)} de {F.entero(len(y))}: sube "
              f"{ui.num(y[-1] - y[0], 4)} de AUC interno desde la primera")
    md = b["modelos"]["clasificacion/ebm"]
    sub = (f"Mejor AUC de la validación cruzada de {F.entero(b['k_cv'])} pliegues acumulado tras cada evaluación de la búsqueda TPE, en "
           "la comparación con la entrega anterior: etiqueta de la mediana, no la del resto de la app.")
    nota = (f"Es la única búsqueda de la EBM que guardó su traza ({F.entero(md['n_evaluaciones'])} evaluaciones con B = "
            f"{F.entero(md['fidelidad']['busqueda']['modelo__outer_bags'])}). El proyecto no guardó la pérdida por paso de boosting: la "
            "librería la usa para la parada temprana de cada bolsa.")
    filas = [[F.entero(k + 1)] + [ui.num(trazas[m][k], 4) if k < len(trazas[m]) else RAYA for m in ("ebm", RIVAL)] for k in range(largo)]
    tabla = (["Evaluación", D.NOMBRE_MODELO["ebm"], D.NOMBRE_MODELO[RIVAL]], filas)
    etiqueta = (f"{titulo}. La EBM pasa de {ui.num(y[0], 4)} a {ui.num(y[-1], 4)}; {D.en_frase(RIVAL)}, de "
                f"{ui.num(trazas[RIVAL][0], 4)} a {ui.num(trazas[RIVAL][-1], 4)}.")
    return ui.grafico(fig, titulo, sub, etiqueta, tabla=tabla, nota=nota, id=f"{PRE}-conv", fuentes=[D.fuente(BUSQUEDAS_E2)],
                      leyenda=[(D.NOMBRE_MODELO["ebm"], "modelo", "escalon"), (D.NOMBRE_MODELO[RIVAL], "rival", "escalon")])


def _seccion_curvas():
    return ui.seccion("Cuántos datos hacen falta y cuándo deja de mejorar la búsqueda", [
        ui.rejilla([_grafico_aprendizaje(), _grafico_convergencia()], "1-1"),
    ], subtitulo="La curva de aprendizaje en la prueba y la curva de convergencia de la búsqueda de hiperparámetros.",
        fuentes=FUENTES_APRENDIZAJE + [D.fuente(BUSQUEDAS_E2)])


# --- 8. Pruebas de hipótesis ---------------------------------------------------------------------------------------------
def _descripcion_pruebas(tarea):
    """Las pruebas de una tarea: hipótesis nula, supuestos, sobre qué datos y si se fijaron antes de ver la prueba."""
    c = TAREA[tarea]
    k = len(D.tabla(c["pliegues"]))
    n_te = D.resumen()["n_prueba"]
    rope = D.resumen()["rope"][c["rope"]]
    banda = f"±{ui.num(rope, c['d_rope'])} de {c['metrica']}"
    t_corr = [f"t corregida ({_cita_en('nadeau2003')})", f"la diferencia media de {c['metrica']} entre los dos modelos en los pliegues "
              "externos es cero", "diferencias por pliegue aproximadamente normales; la varianza se corrige por el solapamiento de "
              "los entrenamientos (factor 1/k + n_prueba / n_entrenamiento)",
              f"los {CARDINAL[k]} pliegues externos ({F.entero(k - 1)} grados de libertad)", "exploratoria"]
    bayes = [f"t correlacionada bayesiana ({_cita_en('benavoli2017')})", "no tiene: da la probabilidad posterior de que la diferencia "
             f"quede por debajo, dentro o por encima de la banda de equivalencia ({banda})", "diferencias por pliegue con "
             "distribución t y correlación entre pliegues; previa no informativa", f"los {CARDINAL[k]} pliegues externos",
             "exploratoria, con la banda fijada antes"]
    if tarea == "clasificacion":
        filas = [t_corr,
                 [f"DeLong ({_cita_en('delong1988')})", "los dos modelos tienen el mismo AUC", "los mismos clientes puntuados por los dos "
                  "modelos, independientes entre sí; el AUC (un estadístico U) es asintóticamente normal",
                  f"los {F.entero(n_te)} clientes de prueba", "fijada antes de evaluar la prueba"],
                 [f"McNemar exacta ({_cita_en('mcnemar1947')})", "entre los clientes en que los modelos discrepan, cada uno acierta con "
                  "la misma probabilidad", f"pares discordantes independientes; la clase la fija el punto de operación "
                  f"({F.pct(M.CAPACIDAD_INFORME, 0)} de capacidad)", f"las clases de los {F.entero(n_te)} clientes de prueba", "exploratoria"],
                 bayes]
    else:
        filas = [t_corr,
                 [f"Diebold-Mariano ({_cita_en('diebold1995')}) con la corrección de {_cita('harvey1997')}", "las dos pérdidas cuadráticas "
                  "esperadas son iguales", "la diferencia de pérdidas es estacionaria en covarianza; varianza de largo plazo con "
                  "corrección para muestras finitas; filas en el orden original del archivo", f"los {F.entero(n_te)} clientes de prueba",
                  "fijada antes de evaluar la prueba"],
                 bayes,
                 [f"Model Confidence Set al {F.pct(NIVEL_MCS, 0)} ({_cita_en('hansen2011')})", "en cada paso, todos los modelos del conjunto "
                  "tienen la misma pérdida esperada; se elimina el peor hasta no rechazar", f"bootstrap estacionario con bloques de "
                  f"{F.entero(D.resumen()['mcs_bloque'])} sobre las pérdidas en el orden del archivo",
                  f"los {F.entero(n_te)} clientes y los {CARDINAL[len(D.prueba(tarea))]} modelos a la vez", "fijada antes de evaluar la prueba"]]
    return ui.tabla(["Prueba", "Hipótesis nula", "Supuestos", "Sobre qué datos", "Protocolo"], filas, compacta=True, num=[False] * 5,
                    ocultar_movil=(3,))


def _resultados_pruebas(tarea):
    """Una fila por referencia y prueba: estadístico, p crudo, p ajustado por Holm, tamaño del efecto y conclusión."""
    c = TAREA[tarea]
    pr = D.pruebas(tarea)
    rope = D.resumen()["rope"][c["rope"]]
    p_mcs, en_mcs = _mcs() if tarea == "regresion" else (None, None)
    filas, muestras = [], []
    for _, f in pr.iterrows():
        ref = f["modelo"]
        nombre = D.NOMBRE_MODELO[ref]
        delta = f"Δ{c['metrica']} = {ui.con_signo(f[c['delta']], c['d'])} {F.intervalo(f['delta_bca_inf'], f['delta_bca_sup'], c['d'], signo=True)}"
        efectos = {"t corregida": f"d_z = {ui.num(f['d_z_pliegues'], 2)}; δ de Cliff = {ui.con_signo(f['delta_cliff'], 2)}"}
        if tarea == "clasificacion":
            efectos[c["fijada"]] = delta
            efectos["McNemar"] = f"error {F.pct(f['err_ebm_op'], 1)} (EBM) y {F.pct(f['err_base_op'], 1)}"
        else:
            efectos[c["fijada"]] = f"{delta}; d_z por cliente = {ui.num(f['d_z_clientes'], 3)}"
        for x in _contrastes(tarea, f):
            filas.append([nombre, x["prueba"].replace(" ", F.FIJO), x["estadistico"], _p(x["p"]), _p(x["p_holm"]), efectos[x["prueba"]],
                          _decision(x, ref)])
            muestras.append(D.ROL_MODELO[ref])
        if f["p_rope"] >= UMBRAL_EQUIVALENCIA:
            lectura = f"equivalencia práctica: pasa del {F.pct(UMBRAL_EQUIVALENCIA, 0)}"
        elif f["p_mejor"] >= UMBRAL_EQUIVALENCIA:
            lectura = "la EBM es mejor con probabilidad posterior alta"
        elif f["p_peor"] >= UMBRAL_EQUIVALENCIA:
            lectura = f"{D.NOMBRE_CORTO[ref]} es mejor con probabilidad posterior alta"
        else:
            lectura = f"sin conclusión al {F.pct(UMBRAL_EQUIVALENCIA, 0)}"
        filas.append([nombre, "bayesiana", _es("P(equivalente)", _prob(f['p_rope'])), RAYA, RAYA,
                      f"{_es('P(EBM mejor)', _prob(f['p_mejor']))}; {_es('P(peor)', _prob(f['p_peor']))}", lectura])
        muestras.append(D.ROL_MODELO[ref])
        if tarea == "regresion":
            p_ref, p_ebm = p_mcs[ref], p_mcs["ebm"]
            if en_mcs[ref]:
                lectura = f"{D.NOMBRE_CORTO[ref]} queda en el conjunto" + ("; la EBM, no" if not en_mcs["ebm"] else "")
            else:
                lectura = f"{D.NOMBRE_CORTO[ref]} sale del conjunto " + ("antes que la EBM" if p_ref < p_ebm else "después de la EBM")
            filas.append([nombre, f"MCS al {F.pct(NIVEL_MCS, 0)}", _es("p_MCS", _p(p_ref)), _p(p_ref), RAYA,
                          _es("la EBM: p_MCS", _p(p_ebm)), lectura])
            muestras.append(D.ROL_MODELO[ref])
    return ui.tabla(["Referencia", "Prueba", "Estadístico", "p sin corregir", "p ajustado (Holm)", "Tamaño del efecto", "Conclusión"],
                    filas, compacta=True, muestras=muestras, num=[False, False, True, True, True, False, False])


def _n_contrastes():
    return {t: len(D.pruebas(t)) * len(_contrastes(t, D.pruebas(t).iloc[0])) for t in TAREAS}


def _sub_pruebas():
    n = _n_contrastes()
    m = len(D.pruebas("clasificacion"))
    pruebas = {t: len(_contrastes(t, D.pruebas(t).iloc[0])) for t in TAREAS}
    return (f"La EBM frente a cada una de las {CARDINAL[m]} referencias de cada tarea: {F.entero(n['clasificacion'])} contrastes con valor "
            f"p en clasificación ({CARDINAL[m]} referencias por {CARDINAL[pruebas['clasificacion']]} pruebas) y {F.entero(n['regresion'])} "
            f"en regresión (por {CARDINAL[pruebas['regresion']]}), {F.entero(sum(n.values()))} en total, más la bayesiana y el Model "
            f"Confidence Set. Holm ({REF['holm1979'][1]}) ajusta dentro de cada prueba sobre las {CARDINAL[m]} comparaciones, con "
            f"α = {ui.num(D.ALFA, 2)}. Un estadístico positivo favorece a la EBM.")


def _seccion_pruebas():
    v = _veredictos()
    n_sup = {t: len(v[t]["supera"]) for t in TAREAS}
    m = len(D.pruebas("clasificacion"))
    rivales = {t: [x for x in D.pruebas(t)["modelo"] if D.ROL_MODELO[x] == "rival"] for t in TAREAS}
    ninguna = all(set(rivales[t]) <= set(v[t]["empata"]) for t in TAREAS)
    sup = (f"a {CARDINAL[n_sup['clasificacion']]} de las {CARDINAL[m]} referencias en cada tarea" if len(set(n_sup.values())) == 1
           else " y ".join(f"a {CARDINAL[k]} en {D.NOMBRE_TAREA[t].lower()}" for t, k in n_sup.items()))
    titulo = (f"Tras la corrección de Holm, la EBM supera {sup}"
              + (f" y ninguna prueba al {F.pct(D.ALFA, 0)} la separa de los modelos de árboles" if ninguna else ""))
    control = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", OPCIONES_TAREA, "clasificacion"))
    protocolo = ui.aviso(["Fijados antes de evaluar la prueba (Entregable 3): la métrica principal, α, la corrección de Holm, DeLong, "
                          "Diebold-Mariano y el Model Confidence Set.",
                          "Posteriores y exploratorias: la t corregida, McNemar y la bayesiana (con sus bandas fijadas antes de "
                          "ejecutarla)."], None, "Qué se fijó antes y qué es exploratorio")
    return ui.seccion(titulo, [
        ui.tarjeta(html.Div(_descripcion_pruebas("clasificacion"), id=f"{PRE}-desc"), titulo="Las pruebas: hipótesis nula y supuestos",
                   subtitulo="Qué contrasta cada una, qué supone y sobre qué datos se calcula.",
                   fuentes=[D.fuente("resultados/resumen.json"), D.fuente("resultados/pliegues_auc.csv"), INFORME_PROTOCOLO]),
        ui.tarjeta(html.Div(_resultados_pruebas("clasificacion"), id=f"{PRE}-res"), titulo="Resultados, referencia por referencia",
                   subtitulo="Estadístico, p sin corregir, p ajustado por Holm, tamaño del efecto y conclusión de cada contraste.",
                   fuentes=FUENTES_PRUEBAS + FUENTES_PRUEBA),
        protocolo,
    ], subtitulo=_sub_pruebas(), controles=[control], fuentes=FUENTES_PRUEBAS)


# --- 9. Robustez ---------------------------------------------------------------------------------------------------------
def _seccion_robustez():
    se = D.tabla("resultados/semillas.csv")
    se = se[se["tarea"] == "clasificacion"].set_index("modelo")
    sen = D.tabla("resultados/sensibilidad_curvas.csv")
    rango = sen.groupby("hiperparametro")["auc_media"].agg(lambda s: s.max() - s.min())
    influyente = rango.idxmax()
    nombres_hp = {"learning_rate": "la tasa de aprendizaje", "max_leaves": "las hojas por árbol", "min_samples_leaf": "el mínimo por hoja",
                  "interactions": "el número de interacciones"}
    e2 = D.clave_valor("resultados/ablacion_interacciones_e2.csv")
    fi = D.clave_valor("resultados/fidelidad_resumen.csv")
    baja, alta = _fidelidad()
    abl = D.ablacion()
    cl = abl[abl["tarea"].str.startswith("clasificacion") & (abl["componente"] != "completo")]
    lejos = cl.loc[cl["delta_completo"].abs().idxmax()]
    cal = V.calibracion()["modelos"]["ebm"]
    conf = V.conformal()["modelos"]["ebm"]
    niveles = sorted(conf, key=float)
    conj = cal["conformal"]
    bajo = min(conj, key=float)
    fichas = [
        ui.cifra(f"Desviación del AUC de prueba entre {F.entero(se.loc['ebm', 'semillas'])} semillas", ui.num(se.loc["ebm", "auc_desv"], 5),
                 variacion=f"{D.NOMBRE_MODELO['random_forest']}: {ui.num(se.loc['random_forest', 'auc_desv'], 4)}; "
                           f"{D.NOMBRE_MODELO[RIVAL]}: {ui.num(se.loc[RIVAL, 'auc_desv'], 4)}",
                 nota="La EBM es casi determinista: la semilla pesa menos que las diferencias entre modelos.", mini=True),
        ui.cifra("AUC al variar un hiperparámetro a la vez", f"{ui.num(sen['auc_media'].min(), 4)} a {ui.num(sen['auc_media'].max(), 4)}",
                 nota=f"Validación cruzada de {CARDINAL[PLIEGUES_SENSIBILIDAD]} pliegues con B = {F.entero(baja)}; el más influyente es "
                      f"{nombres_hp.get(influyente, influyente)}.", mini=True),
        ui.cifra("Ablación: lo que más se aleja del modelo completo", ui.num(abs(lejos["delta_completo"]), 4), "de AUC",
                 nota=f"Quitar los pares cuesta {ui.num(e2['auc_ebm'] - e2['auc_gam'], 4)} (DeLong, {F.p_valor(e2['p_delong'], 2)}); el "
                      "balanceo no cambia el AUC, solo el punto de operación.", mini=True),
        ui.cifra(f"Fidelidad baja (B = {F.entero(baja)}) frente a alta (B = {F.entero(alta)})", ui.num(fi["rho_spearman"], 2), "ρ de Spearman",
                 nota=f"{'La misma ganadora' if str(fi['misma_ganadora']) == 'True' else 'Otra ganadora'} y "
                      f"{F.pct(fi['ahorro_tiempo'], 0)} menos de tiempo de búsqueda.", mini=True),
        ui.cifra("Error de calibración (ECE) en la prueba", ui.num(cal["ece_prueba_recalibrada"], 3), "recalibrada",
                 variacion=f"sin recalibrar: {ui.num(cal['ece_prueba_sin_recalibrar'], 3)}",
                 nota="Recalibración isotónica con predicciones fuera de pliegue del entrenamiento.", mini=True),
        ui.cifra(f"Cobertura de los intervalos conformes al {F.pct(float(NIVEL_CONFORME), 0)} (regresión)",
                 F.pct(conf[NIVEL_CONFORME]["cobertura_prueba"], 1), "en la prueba",
                 variacion="; ".join(f"al {F.pct(float(nv), 0)}: {F.pct(conf[nv]['cobertura_prueba'], 1)}" for nv in niveles
                                     if nv != NIVEL_CONFORME),
                 nota="Los conjuntos de clases cubren "
                      f"{F.pct(conj[bajo]['cobertura_prueba'], 1)} al {F.pct(float(bajo), 0)} y "
                      f"{F.pct(conj[NIVEL_CONFORME]['cobertura_prueba'], 1)} al {F.pct(float(NIVEL_CONFORME), 0)}: de más, porque el "
                      f"{F.pct(conj[NIVEL_CONFORME]['fraccion_ambos'], 0)} de los clientes recibe las dos clases.", mini=True),
    ]
    destinos = ui.destinos([ui.destino(app.ruta_vista(v), v, n, app.pregunta(v)) for v, n in (("robustez", "Robustez"), ("errores", "Errores"))])
    titulo = (f"La EBM es estable: la semilla mueve su AUC {ui.num(se.loc['ebm', 'auc_desv'], 5)} y los hiperparámetros lo dejan entre "
              f"{ui.num(sen['auc_media'].min(), 4)} y {ui.num(sen['auc_media'].max(), 4)}")
    return ui.seccion(titulo, [ui.cifras(fichas, "cifras--3"), destinos],
                      subtitulo="Resumen de los estudios de robustez: semillas, sensibilidad, ablación, fidelidad, calibración y cobertura. "
                                "El detalle está en «Robustez» y en «Errores».",
                      fuentes=FUENTES_ROBUSTEZ)


# --- 10. Veredicto -------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _condiciones(tarea):
    """Las cuatro condiciones de superioridad de la guía (§6.4) en una tarea, con sus valores: el mejor modelo de referencia es
    el de mejor métrica externa entre los otros siete."""
    c = TAREA[tarea]
    t = D.prueba(tarea)
    orden = t.sort_values(c["externa"], ascending=not c["mayor"]).reset_index(drop=True)
    puesto = int(orden.index[orden["modelo"] == "ebm"][0]) + 1
    e = D.fila(t, modelo="ebm")
    mejor = orden[orden["modelo"] != "ebm"].iloc[0]
    ref = mejor["modelo"]
    f = D.fila(D.pruebas(tarea), modelo=ref)
    contrastes = _contrastes(tarea, f)
    a_favor = _rechazan(contrastes)
    detalle = "; ".join(f"{x['prueba']}, {_p(x['p_holm'])}" for x in contrastes)
    texto2 = f"{F.entero(len(a_favor))} de {F.entero(len(contrastes) + (tarea == 'regresion'))}: p ajustados {detalle}"
    if tarea == "regresion":
        p_mcs, en_mcs = _mcs()
        if not en_mcs["ebm"] and en_mcs[ref]:
            texto2 += (f". El MCS sí separa al {F.pct(ALFA_MCS, 0)}, pero en contra de la EBM ({_p(p_mcs['ebm'])}): "
                       f"{D.NOMBRE_CORTO[ref]} queda en el conjunto y la EBM no")
        elif en_mcs["ebm"] and not en_mcs[ref]:
            a_favor.append({"prueba": "MCS"})
            texto2 += f". El MCS deja a la EBM y saca a {D.NOMBRE_CORTO[ref]} ({_p(p_mcs[ref])})"
    cliff = f["delta_cliff"]
    if abs(cliff) >= UMBRAL_CLIFF and cliff < 0:
        texto3 = f"δ = {ui.con_signo(cliff, 2)}: no despreciable, pero a favor de {D.NOMBRE_CORTO[ref]}"
    elif abs(cliff) >= UMBRAL_CLIFF:
        texto3 = f"δ = {ui.con_signo(cliff, 2)}: no despreciable y a favor de la EBM"
    else:
        texto3 = f"δ = {ui.con_signo(cliff, 2)}: despreciable"
    externa = f"{ORDINAL_F[puesto]} de {len(t)}: {ui.num(e[c['externa']], c['d'])} frente a {ui.num(mejor[c['externa']], c['d'])} de {D.NOMBRE_CORTO[ref]}"
    condiciones = [
        {"cumple": puesto == 1, "texto": _mayuscula(externa)},
        {"cumple": len(a_favor) >= PRUEBAS_MINIMAS, "texto": texto2},
        {"cumple": cliff >= UMBRAL_CLIFF, "texto": texto3},
        {"cumple": CONJUNTOS_EVALUADOS >= CONJUNTOS_MINIMOS,
         "texto": f"{F.entero(CONJUNTOS_EVALUADOS)} de {F.entero(CONJUNTOS_MINIMOS)}: el proyecto usó un único conjunto de datos (COFINFAD)"},
    ]
    return {"referencia": ref, "condiciones": condiciones, "cliff": cliff, "a_favor": len(a_favor)}


def _estado(cumple):
    return ui.chip("se cumple", "bien") if cumple else ui.chip("no se cumple", icono="cerrar")


def _tarjeta_condiciones():
    cond = {t: _condiciones(t) for t in TAREAS}
    m = len(D.pruebas("clasificacion"))
    nombres = [f"La métrica media del bucle externo es mejor que la de la mejor combinación de cada uno de los otros {CARDINAL[m]} modelos",
               f"Al menos {CARDINAL[PRUEBAS_MINIMAS]} pruebas distintas rechazan la igualdad frente al mejor modelo de referencia, con p "
               f"ajustado < {ui.num(D.ALFA, 2)} (< {ui.num(ALFA_MCS, 2)} en el MCS)",
               f"El tamaño del efecto frente a él no es despreciable: |δ de Cliff| ≥ {ui.num(UMBRAL_CLIFF, 3)}, a favor de la EBM",
               f"La ventaja se repite en al menos {CARDINAL[CONJUNTOS_MINIMOS]} conjuntos de datos adicionales"]
    filas = []
    for i, nombre in enumerate(nombres):
        fila = [f"{i + 1}. {nombre}"]
        for t in TAREAS:
            x = cond[t]["condiciones"][i]
            fila.append([_estado(x["cumple"]), html.Br(), html.Span(x["texto"], className="tinta-2")])
        filas.append(fila)
    refs = {cond[t]["referencia"] for t in TAREAS}
    mejor = (D.en_frase(refs.pop()) + " en las dos tareas") if len(refs) == 1 else " y ".join(
        f"{D.en_frase(cond[t]['referencia'])} en {D.NOMBRE_TAREA[t].lower()}" for t in TAREAS)
    return ui.tarjeta([ui.tabla(["Condición", D.NOMBRE_TAREA["clasificacion"], D.NOMBRE_TAREA["regresion"]], filas, compacta=True,
                                num=[False, False, False])],
                      titulo="Las cuatro condiciones de la guía, una por una",
                      subtitulo=f"Definición operativa de superioridad (§6.4). El mejor modelo de referencia, por la métrica externa, es "
                                f"{mejor}.",
                      fuentes=FUENTES_PRUEBA + FUENTES_PRUEBAS)


def _tarjeta_frente_a():
    filas, muestras = [], []
    for tarea in TAREAS:
        c = TAREA[tarea]
        t = D.prueba(tarea).set_index("modelo")
        p_mcs, en_mcs = _mcs() if tarea == "regresion" else (None, None)
        for _, f in D.pruebas(tarea).iterrows():
            ref = f["modelo"]
            v = D.veredicto(f, tarea)
            contrastes = _contrastes(tarea, f)
            a_favor = _rechazan(contrastes)
            mejor_ext = (t.loc["ebm", c["externa"]] > t.loc[ref, c["externa"]]) == c["mayor"]
            if v == "empata":
                lectura = "equivalente: " + _es("P(equivalencia práctica)", _prob(f['p_rope']))
            elif v == "supera":
                faltan = [x["prueba"] for x in contrastes if x not in a_favor]
                lectura = "supera en todas las pruebas por pares" if not faltan else (
                    f"supera; {F.enumerar(faltan)} no los separa" + (f" con el {F.pct(M.CAPACIDAD_INFORME, 0)} de capacidad"
                                                                     if "McNemar" in faltan else ""))
            else:
                lectura = D.VEREDICTO_TEXTO[v]
            if tarea == "regresion" and not en_mcs[ref]:
                lectura += f"; el MCS la excluye ({_p(p_mcs[ref])})"
            filas.append([f"{D.NOMBRE_MODELO[ref]} ({D.NOMBRE_TAREA[tarea].lower()})", D.VEREDICTO_TEXTO[v],
                          "la EBM" if mejor_ext else D.NOMBRE_CORTO[ref], f"{F.entero(len(a_favor))} de {F.entero(len(contrastes))}",
                          ui.con_signo(f["delta_cliff"], 2), lectura])
            muestras.append(D.ROL_MODELO[ref])
    return ui.tarjeta([ui.tabla(["Referencia", "Veredicto", "Mejor externa", "Pruebas a favor", "δ de Cliff", "Lectura"], filas, compacta=True, muestras=muestras,
                                num=[False, False, False, True, True, False]),
                       ui.nota(f"El veredicto es el del resto de la app: «supera» cuando la t corregida y {TAREA['clasificacion']['fijada']} "
                               f"(o {TAREA['regresion']['fijada']}) rechazan a favor de la EBM tras Holm; «sin diferencia» cuando ninguna "
                               "de las dos rechaza. «Mejor externa»: el modelo con mejor métrica media en el bucle externo; «Pruebas a "
                               "favor»: las pruebas por pares con p ajustado por Holm menor que α que rechazan a favor de la EBM.")],
                      titulo="Frente a cada referencia",
                      subtitulo="El mismo criterio aplicado a cada uno de los otros modelos: dónde hay equivalencia y dónde ventaja.",
                      fuentes=FUENTES_PRUEBAS + FUENTES_PRUEBA)


def _por_que(palabra):
    """Por qué el veredicto es el que es, condición por condición y en las dos tareas (lo que falla, con sus valores)."""
    fallan = []
    cond = {t: _condiciones(t) for t in TAREAS}
    if not any(cond[t]["condiciones"][0]["cumple"] for t in TAREAS):
        fallan.append("no tiene la mejor métrica externa en ninguna tarea")
    if not any(cond[t]["condiciones"][1]["cumple"] for t in TAREAS):
        refs = F.enumerar(sorted({D.en_frase(cond[t]["referencia"]) for t in TAREAS}))
        if all(cond[t]["a_favor"] == 0 for t in TAREAS):
            fallan.append(f"ninguna prueba la separa a su favor {_tras('de', refs)}")
        else:
            fallan.append(f"menos de {CARDINAL[PRUEBAS_MINIMAS]} pruebas la separan a su favor {_tras('de', refs)}")
    if not any(cond[t]["condiciones"][2]["cumple"] for t in TAREAS):
        cliffs = "; ".join(f"{ui.con_signo(cond[t]['cliff'], 2)} en {D.NOMBRE_TAREA[t].lower()}" for t in TAREAS)
        fallan.append(f"el δ de Cliff frente al mejor modelo de referencia no la favorece ({cliffs})")
    if CONJUNTOS_EVALUADOS < CONJUNTOS_MINIMOS:
        fallan.append("no se evaluó en otros conjuntos de datos")
    if not fallan:
        return f"Con la definición de la guía, el veredicto es de {palabra}."
    return f"Con la definición de la guía, la EBM {'no es superior' if palabra != 'superioridad' else 'es superior'}: " + F.enumerar(fallan) + "."


def _seccion_veredicto():
    cumplidas = {t: sum(x["cumple"] for x in _condiciones(t)["condiciones"]) for t in TAREAS}
    superior = all(k == CONDICIONES for k in cumplidas.values())
    if len(set(cumplidas.values())) == 1:
        valor = f"{F.entero(next(iter(cumplidas.values())))} de {F.entero(CONDICIONES)}"
        cuanto = "en clasificación y en regresión"
    else:
        valor = " y ".join(F.entero(k) for k in cumplidas.values()) + f" de {F.entero(CONDICIONES)}"
        cuanto = "en clasificación y en regresión, respectivamente"
    palabra = "superioridad" if superior else "no superioridad"
    insight = ui.tarjeta_insight(valor, f"condiciones de superioridad de la guía que cumple la EBM, {cuanto}",
                                 f"Veredicto: {palabra}. Es equivalente a {_con_quien_empata()} y supera con significación, tras Holm, "
                                 f"{_quien_supera()}.", chip_="Veredicto")
    lo_que_dice = ui.aviso([_por_que(palabra),
                            f"Lo que sí sostiene la evidencia es la equivalencia con {_con_quien_empata()} y la ventaja sobre "
                            f"{_quien_supera(con_a=False)}, con un modelo que se lee sin explicador: la alcanza, no la supera."],
                           None, "Lo que dice y lo que no dice el veredicto")
    return ui.seccion(f"Veredicto: {palabra}. La EBM cumple {valor} condiciones de la guía {cuanto}", [
        ui.rejilla([insight, _tarjeta_condiciones()], "1-2"),
        _tarjeta_frente_a(),
        lo_que_dice,
        ui.destinos([ui.destino(app.ruta_vista("comparacion"), "comparacion", "Comparación estadística", app.pregunta("comparacion"))]),
    ], subtitulo="El veredicto explícito de superioridad o no superioridad, condición por condición y con los valores reales.",
        fuentes=FUENTES_PRUEBA + FUENTES_PRUEBAS)


# --- La pantalla ----------------------------------------------------------------------------------------------------------
def layout(**_):
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), _respuesta(), pantalla=PANTALLA),
        _seccion_motivacion(),
        _seccion_formulacion(),
        _seccion_algoritmo(),
        _seccion_teoria(),
        _seccion_complejidad(),
        _seccion_novedad(),
        _seccion_curvas(),
        _seccion_pruebas(),
        _seccion_robustez(),
        _seccion_veredicto(),
    ])


@callback(Output(f"{PRE}-desc", "children"), Output(f"{PRE}-res", "children"), Input(f"{PRE}-tarea", "value"))
def _por_tarea(tarea):
    tarea = tarea if tarea in TAREA else "clasificacion"
    return _descripcion_pruebas(tarea), _resultados_pruebas(tarea)
