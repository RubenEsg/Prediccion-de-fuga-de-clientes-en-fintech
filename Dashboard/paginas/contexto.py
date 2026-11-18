"""Contexto · ¿Qué problema resolvemos y con qué datos?

La primera sección del dash y el primer criterio de la sustentación (problema y formulación matemática). En una sola
página, en el orden de la guía (§7.3.1): el problema y la cifra que lo responde, por qué importa, la pregunta y los
objetivos, la formulación, los datos con su diccionario, la metodología (el pipeline, con un enlace a la vista que muestra
cada paso), la búsqueda con dos fidelidades (y por qué no es Successive Halving), las limitaciones, la ética y los sesgos,
las conclusiones, los autores y el recorrido de la sustentación.

Toda cifra sale de datos/, de artifacts/, del motor o de CITATION.cff, y cada tarjeta dice de qué archivo; la única
excepción es la del costo de retener frente a conseguir un cliente, citada de Gallo (2014). El relato es el del proyecto
final corregido: el modelo original es BDL (Boosting con Desplazamiento Latente), programado desde cero, y la EBM, que
viene de una librería, es un benchmark reciente. El veredicto de superioridad del original no se escribe aquí: sale de
datos/original/veredicto.json y condicion4.json (D.estado_veredicto y D.frase_veredicto) y la página dice lo que digan,
sea cual sea. La corrección de la recencia se lee de datos/original/auditoria_fuga.json. Las conclusiones sobre la EBM
siguen el informe del Entregable 3 con su tono: alcanza a los modelos de árboles, no los supera. Reutiliza el héroe del
Entregable 3 de secciones/inicio.py (solo si el original todavía no está), los enlaces de secciones/limites.py y las
frases del veredicto de secciones/original.py.
"""
import json
import math
import re
import unicodedata
from functools import lru_cache

import dash
import pandas as pd
from dash import Input, Output, callback, dash_table, dcc, html

import app
import datos_app as D
import formato as F
import modelos_vivos as V
import motor as M
import ui
from secciones import inicio, limites
from secciones import original as OG

dash.register_page(__name__, path="/", name="Contexto", title=f"Contexto · {app.MARCA}")

PRE = "cx"
O = D.ORIGINAL                    # el modelo original (BDL)
TAREAS = tuple(inicio.TAREA)      # clasificación y regresión, en el orden del control de tarea
RIVAL = "xgboost"                 # el modelo de árboles con que se compara siempre a la EBM
LINEAL = "logistica"              # la referencia lineal de la campaña
NIVEL_MCS = inicio.NIVEL_MCS      # nivel del Model Confidence Set del informe
FACTOR_SH = 3                     # factor de reducción habitual de Successive Halving y Hyperband (Li et al.)
AZAR_AUC = 0.5                    # AUC de ordenar al azar
CAPACIDAD_ALTA = 0.50             # la otra capacidad de campaña que citan las conclusiones (recall_50)
VARIABLES_CON_SENAL = 4           # las cuatro variables con señal del análisis exploratorio
NIVEL_CONFORME = "0.90"           # nivel del intervalo conforme que se cita (clave de datos/conformal.json)
LARGO_LISTA = 40                  # caracteres: una lista de categorías más larga no se escribe en el rango del diccionario
LARGO_HUELLA = 16                 # caracteres de la huella sha256 del código del modelo original que se muestran
UNION = "⁠"                  # une sin espacio: «29–30» no se parte en la raya en el teléfono
CITACION = app.AQUI / "CITATION.cff"
LICENCIA_MIT = "https://opensource.org/license/mit"
LICENCIA_CC_BY = "https://creativecommons.org/licenses/by/4.0/deed.es"
CARDINAL = {2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
            11: "once", 12: "doce"}
ORDINAL_M = {1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto", 5: "quinto", 6: "sexto", 7: "séptimo", 8: "octavo", 9: "noveno",
             10: "décimo"}
# Cómo se dice el veredicto del modelo original en un título (la palabra sale de D.estado_veredicto)
TITULO_VEREDICTO = {"superior": "el modelo original supera a los benchmarks",
                    "no superior": "no se demuestra que el modelo original supere a los benchmarks",
                    "pendiente": "el veredicto del modelo original queda pendiente de la cuarta condición",
                    None: "el veredicto del modelo original está pendiente de las pruebas de hipótesis"}
PROTOCOLO = "ModeloOriginal/experiments/protocolo.md (protocolo pre-registrado)"

# Referencias que cita la página: las del informe (referencias.bib), y las de Successive Halving y la cifra del costo de
# retener, que no están en él (`motivo` dice por qué se citan; `url`, dónde leerla).
# El año va como número para escribirlo con la cita.
REFERENCIAS = {
    "kaufman": {"corto": "Kaufman et al.", "autores": "Kaufman, S., Rosset, S., Perlich, C. y Stitelman, O.", "anio": 2012,
                "titulo": "Leakage in data mining: Formulation, detection, and avoidance",
                "fuente": "ACM Transactions on Knowledge Discovery from Data", "informe": True},
    "lou2012": {"corto": "Lou, Caruana y Gehrke", "autores": "Lou, Y., Caruana, R. y Gehrke, J.", "anio": 2012,
                "titulo": "Intelligible models for classification and regression",
                "fuente": "Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
                "informe": True},
    "lou2013": {"corto": "Lou et al.", "autores": "Lou, Y., Caruana, R., Gehrke, J. y Hooker, G.", "anio": 2013,
                "titulo": "Accurate intelligible models with pairwise interactions",
                "fuente": "Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
                "informe": True},
    "nori": {"corto": "Nori et al.", "autores": "Nori, H., Jenkins, S., Koch, P. y Caruana, R.", "anio": 2019,
             "titulo": "InterpretML: A unified framework for machine learning interpretability", "fuente": "arXiv:1909.09223",
             "informe": True},
    "lundberg": {"corto": "Lundberg y Lee", "autores": "Lundberg, S. M. y Lee, S.-I.", "anio": 2017,
                 "titulo": "A unified approach to interpreting model predictions",
                 "fuente": "Advances in Neural Information Processing Systems", "informe": True},
    "breiman": {"corto": "Breiman", "autores": "Breiman, L.", "anio": 2001, "titulo": "Random forests", "fuente": "Machine Learning",
                "informe": True},
    "chen": {"corto": "Chen y Guestrin", "autores": "Chen, T. y Guestrin, C.", "anio": 2016,
             "titulo": "XGBoost: A scalable tree boosting system",
             "fuente": "Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
             "informe": True},
    "cawley": {"corto": "Cawley y Talbot", "autores": "Cawley, G. C. y Talbot, N. L. C.", "anio": 2010,
               "titulo": "On over-fitting in model selection and subsequent selection bias in performance evaluation",
               "fuente": "Journal of Machine Learning Research", "informe": True},
    "jamieson": {"corto": "Jamieson y Talwalkar", "autores": "Jamieson, K. y Talwalkar, A.", "anio": 2016,
                 "titulo": "Non-stochastic best arm identification and hyperparameter optimization",
                 "fuente": "Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)",
                 "informe": False, "motivo": "para precisar la búsqueda con dos fidelidades"},
    "li": {"corto": "Li et al.", "autores": "Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. y Talwalkar, A.", "anio": 2018,
           "titulo": "Hyperband: A novel bandit-based approach to hyperparameter optimization",
           "fuente": "Journal of Machine Learning Research", "informe": False,
           "motivo": "para precisar la búsqueda con dos fidelidades"},
    "dempster": {"corto": "Dempster, Laird y Rubin", "autores": "Dempster, A. P., Laird, N. M. y Rubin, D. B.", "anio": 1977,
                 "titulo": "Maximum likelihood from incomplete data via the EM algorithm",
                 "fuente": "Journal of the Royal Statistical Society: Series B", "informe": False, "libro": True,
                 "motivo": "por el algoritmo EM con que aprende el modelo original"},
    "friedman": {"corto": "Friedman", "autores": "Friedman, J. H.", "anio": 2001,
                 "titulo": "Greedy function approximation: A gradient boosting machine", "fuente": "The Annals of Statistics",
                 "informe": False, "libro": True, "motivo": "por el boosting de árboles del índice del modelo original"},
    "desarbo": {"corto": "DeSarbo y Cron", "autores": "DeSarbo, W. S. y Cron, W. L.", "anio": 1988,
                "titulo": "A maximum likelihood methodology for clusterwise linear regression", "fuente": "Journal of Classification",
                "informe": False, "libro": True,
                "motivo": "por las mezclas de regresiones por EM, una de las familias de métodos afines al modelo original"},
    "hou": {"corto": "Hou, Li y Gao", "autores": "Hou, Y., Li, J. y Gao, G.", "anio": 2025,
            "titulo": "Insurance loss modeling with gradient tree-boosted mixture models",
            "fuente": "Insurance: Mathematics and Economics", "informe": False, "libro": True,
            "motivo": "como uno de los antecedentes algorítmicos más cercanos al modelo original: el algoritmo Expectation-Boosting, "
                      "un EM cuyo paso M es un paso de boosting"},
    "vovk": {"corto": "Vovk, Gammerman y Shafer", "autores": "Vovk, V., Gammerman, A. y Shafer, G.", "anio": 2005,
             "titulo": "Algorithmic learning in a random world", "fuente": "Springer", "informe": False, "libro": True,
             "motivo": "por la predicción conforme de los conjuntos de predicción"},
    "ke": {"corto": "Ke et al.", "autores": "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. y Liu, T.-Y.", "anio": 2017,
           "titulo": "LightGBM: A highly efficient gradient boosting decision tree",
           "fuente": "Advances in Neural Information Processing Systems", "informe": False, "libro": True,
           "motivo": "por LightGBM, el segundo benchmark reciente"},
    "demsar": {"corto": "Demšar", "autores": "Demšar, J.", "anio": 2006,
               "titulo": "Statistical comparisons of classifiers over multiple data sets", "fuente": "Journal of Machine Learning Research",
               "informe": False, "libro": True, "motivo": "por las pruebas de Friedman y Nemenyi con todos los modelos"},
    "nadeau": {"corto": "Nadeau y Bengio", "autores": "Nadeau, C. y Bengio, Y.", "anio": 2003,
               "titulo": "Inference for the generalization error", "fuente": "Machine Learning", "informe": False, "libro": True,
               "motivo": "por la t corregida de las pruebas del modelo original"},
    "holm": {"corto": "Holm", "autores": "Holm, S.", "anio": 1979, "titulo": "A simple sequentially rejective multiple test procedure",
             "fuente": "Scandinavian Journal of Statistics", "informe": False, "libro": True,
             "motivo": "por la corrección de los valores p por comparaciones múltiples"},
    "gallo": {"corto": "Gallo", "autores": "Gallo, A.", "anio": 2014, "titulo": "The value of keeping the right customers",
              "fuente": "Harvard Business Review, 29 de octubre", "informe": False,
              "motivo": "para la cifra del costo de retener frente a conseguir un cliente",
              "url": "https://hbr.org/2014/10/the-value-of-keeping-the-right-customers"},
}

# Las columnas que el ETL eliminó: nombre visible, tipo y motivo (la tabla «Columnas eliminadas» de la metodología)
ELIMINADAS = {
    "customer_id": ("Identificador del cliente", "identificador", "identificador"),
    "active_products": ("Productos activos", "numérica entera", "fuga"),
    "customer_lifetime_value": ("Valor del cliente (CLV)", "numérica real", "segundo_objetivo"),
    "clv_segment": ("Segmento de valor del cliente", "categórica", "segundo_objetivo"),
    "average_transaction_value": ("Valor medio por transacción (otro resumen)", "numérica", "duplica"),
    "total_transaction_volume": ("Volumen total transado (otro resumen)", "numérica", "duplica"),
    "avg_daily_transactions": ("Transacciones diarias promedio", "numérica", "duplica"),
    "monthly_transaction_count": ("Transacciones mensuales", "numérica", "duplica"),
    "first_transaction_date": ("Fecha de la primera transacción", "fecha", "fecha_duplica"),
    "last_transaction_date": ("Fecha de la última transacción", "fecha", "fecha_duplica"),
    "last_survey_date": ("Fecha de la última encuesta", "fecha", "fecha_sin_senal"),
    "first_tx": ("Primera transacción (resumen)", "fecha", "fecha_sin_senal"),
    "last_tx": ("Última transacción (resumen)", "fecha", "fecha_sin_senal"),
    "nps_score": ("Puntaje NPS", "numérica entera", "colineal"),
    "occupation": ("Ocupación", "categórica", "ocupacion"),
    "complaint_topics": ("Temas de las quejas", "texto", "texto"),
    "feature_requests": ("Funciones solicitadas", "texto", "texto"),
}
MOTIVO_CORTO = {"identificador": "identificador", "fuga": "fuga de información", "segundo_objetivo": "segundo objetivo",
                "duplica": "duplica a otra columna", "fecha_duplica": "fecha duplicada", "fecha_sin_senal": "fecha sin señal",
                "colineal": "colineal", "ocupacion": "sin señal", "texto": "texto con faltantes"}
TIPO = {"entera": "numérica entera", "real": "numérica real", "booleana": "booleana", "categorica": "categórica"}
MEDIDA = {"spearman": "ρ de Spearman", "eta2": "η²"}
PARAMETRO = {"modelo__outer_bags": ("B", "bolsas externas"), "modelo__n_estimators": ("T", "árboles")}
SEGMENTOS = {"cohorte": ("Cohorte de edad", "cohorte"), "genero": ("Género", "gender"), "ciudad": ("Ciudad", "location"),
             "ingresos": ("Tramo de ingresos", "income_bracket")}
ORDEN_INGRESOS = ["Low", "Medium", "High", "Very High"]
GRUPOS_DICC = {"todas": "Todas", "predictores": "Predictores", "fuera": "Fuera del modelo"}

FUENTES_PRUEBAS = inicio.FUENTES_PRUEBAS
F_META, F_ETL, F_FALTANTES = D.fuente("metadatos.json"), D.fuente("eda/etl.json"), D.fuente("eda/faltantes.csv")
F_CORRIDAS, F_CLF = D.fuente("entrega3/corridas.csv"), D.fuente("resultados/clasificacion_prueba.csv")


# --- Lecturas y formato -------------------------------------------------------------------------------------------------
def _cardinal(n):
    return CARDINAL.get(int(n), F.entero(n))


def _mayuscula(texto):
    return texto[:1].upper() + texto[1:]


def _tex(texto):
    """Un número ya formateado (es-CO) dentro de una fórmula de LaTeX: la coma decimal entre llaves y el menos de LaTeX."""
    return str(texto).replace(",", "{,}").replace(F.MENOS, "-")


def _formula(*lineas):
    """Fórmulas en modo display (MathJax), una por línea (los $$ van en su propia línea: si no, se leen como fórmula en línea)."""
    return dcc.Markdown("\n\n".join(f"$$\n{linea}\n$$" for linea in lineas), mathjax=True)


def _texto_math(texto):
    """Un párrafo con fórmulas en línea ($…$)."""
    return dcc.Markdown(texto, mathjax=True)


def _cita(clave):
    r = REFERENCIAS[clave]
    return f"{r['corto']}, {r['anio']}"


def _fecha(iso):
    return ui.fecha_larga(pd.Timestamp(iso), con_dia=False) if iso else "sin fecha"


@lru_cache(maxsize=1)
def _citacion():
    """Versión, fecha, licencia, repositorio y autores de CITATION.cff, y la referencia del conjunto de datos que trae."""
    texto = CITACION.read_text(encoding="utf-8") if CITACION.is_file() else ""
    propio, _, datos = texto.partition("\nreferences:")

    def campo(bloque, clave):
        m = re.search(rf'^\s*-?\s*{re.escape(clave)}:\s*"?([^"\n]+?)"?\s*$', bloque, re.M)
        return m.group(1).strip() if m else None

    def autores(bloque):
        return re.findall(r'family-names:\s*"([^"]+)"\s*\n\s*given-names:\s*"([^"]+)"', bloque)

    return {"version": campo(propio, "version"), "fecha": campo(propio, "date-released"), "licencia": campo(propio, "license"),
            "repositorio": campo(propio, "repository-code") or limites.REPOSITORIO, "autores": autores(propio),
            "datos": {"titulo": campo(datos, "title"), "autores": autores(datos), "anio": campo(datos, "year"),
                      "editor": campo(datos, "name"), "version": campo(datos, "version"), "doi": campo(datos, "doi"),
                      "licencia": (campo(datos, "license") or "").replace("-", " ")}}


def _autores_prosa(pares):
    return F.enumerar(f"{nombre} {apellido}" for apellido, nombre in pares)


def _autores_cita(pares):
    """«Muñoz Guerrero, L. E., Ceballos, Y. F. y Trejos Rojas, L. D.»"""
    def iniciales(nombre):
        return " ".join(f"{p[0]}." for p in nombre.split())
    return F.enumerar(f"{apellido}, {iniciales(nombre)}" for apellido, nombre in pares)


def _cita_datos():
    d = _citacion()["datos"]
    apellidos = [a for a, _n in d["autores"]]
    corto = F.enumerar(apellidos) if len(apellidos) <= 3 else f"{apellidos[0]} et al."
    return f"{corto}, {d['anio']}"


@lru_cache(maxsize=1)
def _origen():
    o = D.json("origen.json")
    return {"datos": o.get("generado"), "artefactos": o.get("artefactos") or V.indice().get("generado")}


@lru_cache(maxsize=1)
def _hechos():
    """Las cifras de los datos que usa la página, leídas una vez."""
    meta, etl, corridas = M.meta(), D.json("eda/etl.json"), D.tabla("entrega3/corridas.csv")
    E = V.entrenamiento()
    s, tau = E["churn_probability"], meta["etiqueta"]["umbral"]
    asoc = D.tabla("eda/asociacion.csv").set_index("variable")
    pv = D.tabla("eda/por_variable.csv")
    escalera = pv[pv["variable"] == "active_products"].sort_values("orden")["media_objetivo"].diff().dropna().tolist()
    tipos = etl["tipos"]
    clf = corridas[corridas["tarea"] == "clasificacion"]
    fecha_detalle = re.search(r"\((\d+-\d+-\d+)\)", str(M.variables().at["recencia_dias", "descripcion"]))     # «(2023-12-29)»
    return {
        "meta": meta, "etl": etl, "tau": tau, "n_empates": int((s == tau).sum()), "s_min": float(s.min()), "s_max": float(s.max()),
        "prev_train": meta["prevalencia_entrenamiento"], "prev_test": meta["prevalencia_prueba"],
        "edades": sorted(int(x) for x in E["age"].unique()), "cohortes": int(M.clientes()["cohorte"].nunique()),
        "escalera": escalera, "r_ap": float(asoc.at["active_products", "pearson"]),
        "rho_derivadas": float(asoc.loc[etl["derivadas"], "valor"].abs().max()),
        "n_num": len(tipos["numericas"]), "n_bool": len(tipos["booleanas"]), "n_cat": len(tipos["categoricas"]),
        "n_pred": len(etl["predictores"]), "p": etl["n_columnas_modelo"], "n_elim": len(etl["eliminadas"]), "n_deriv": len(etl["derivadas"]),
        "tx_total": etl["n_transacciones_limpias"] + etl["auditoria_transacciones"]["duplicadas"],
        "duplicadas": etl["auditoria_transacciones"]["duplicadas"],
        "falta_cupo": float(D.tabla("eda/faltantes.csv").set_index("columna").at["credit_utilization_ratio", "fraccion"]),
        "fecha_detalle": fecha_detalle.group(1) if fecha_detalle else None,
        "k_ext": int(corridas["k_ext"].iloc[0]), "k_int": int(corridas["k_int"].iloc[0]),
        "corridas": len(corridas), "corridas_clf": len(clf), "corridas_reg": len(corridas) - len(clf),
        "no_aplican": int((corridas["estado"] != "ok").sum()), "corridas_guia": int((corridas["modelo"] != "ebm").sum()),
        "corridas_ebm": int((corridas["modelo"] == "ebm").sum()), "horas": corridas["tiempo_total_s"].sum() / 3600,
        "n_modelos": int(corridas.groupby("tarea")["modelo"].nunique().max()), "n_balanceos": int(clf["balanceo"].nunique()),
        "n_optimizadores": int(corridas["optimizador"].nunique()),
        "presupuesto_min": int(corridas["presupuesto"].min()), "presupuesto_max": int(corridas["presupuesto"].max()),
        "fraccion_prueba": meta["fraccion_prueba"],
    }


def _particion():
    f = _hechos()["fraccion_prueba"]
    return f"{F.entero(round(100 * (1 - f)))}/{F.entero(round(100 * f))}"


# --- 1. El héroe ------------------------------------------------------------------------------------------------------
def _veredictos():
    """El veredicto del modelo original en las dos tareas, en una frase, con lo que digan veredicto.json y condicion4.json."""
    estados = {t: D.estado_veredicto(t) for t in TAREAS}
    if all(e is None for e in estados.values()):
        return "pendiente: las pruebas de hipótesis todavía no han llegado"
    palabras = {t: D.frase_veredicto(t, detalle=False) for t in TAREAS}
    if len(set(palabras.values())) == 1:
        texto = f"{palabras[TAREAS[0]]} en las dos tareas"
    else:
        texto = " y ".join(f"{p} en {D.NOMBRE_TAREA[t].lower()}" for t, p in palabras.items())
    if any(e is not None and e["provisional"] for e in estados.values()):
        texto += " (resultado provisional: faltan repeticiones de la validación anidada)"
    return texto


def _titulo_respuesta():
    """El título de la sección de la cifra principal: lo que diga el veredicto de los archivos."""
    palabras = {t: (D.estado_veredicto(t) or {}).get("palabra") for t in TAREAS}
    if len(set(palabras.values())) == 1:
        return f"La respuesta en una cifra: {TITULO_VEREDICTO[palabras[TAREAS[0]]]}, en las dos tareas"
    return "La respuesta en una cifra: " + "; ".join(f"en {D.NOMBRE_TAREA[t].lower()}, {TITULO_VEREDICTO[p]}" for t, p in palabras.items())


def _benchmarks():
    """Cuántos benchmarks acompañan al modelo original en la tabla de líderes."""
    t = D.lideres(TAREAS[0])
    return len(t) - int(O in set(t["modelo"]))


def _subtitulo():
    h = _hechos()
    inicio_ = (f"Una fintech quiere ordenar a sus clientes por riesgo de fuga para concentrar en ellos una campaña de retención que no "
               f"alcanza para todos. Con los {F.entero(h['etl']['n_clientes'])} clientes del conjunto público COFINFAD, que es sintético, ")
    if not any(D.hay_modelo_original(t) for t in TAREAS):
        return (inicio_ + f"el proyecto compara {_cardinal(h['n_modelos'])} modelos, entre ellos uno que se puede leer, la EBM: empata con "
                "los mejores modelos de árboles y supera a los lineales, a k-NN y a Naive Bayes. El modelo original, BDL, todavía no "
                "tiene resultados en esta app.")
    return (inicio_ + f"el proyecto propone un modelo original programado desde cero, BDL ({D.NOMBRE_LARGO_ORIGINAL}), y lo compara con "
            f"{_cardinal(_benchmarks())} benchmarks. Con la regla de superioridad de la guía, su veredicto es: {_veredictos()}.")


def _heroe_original(tarea):
    """La cifra principal: el modelo original en la prueba, con lo que la acota (su ventaja sobre el mejor benchmark en la
    validación anidada, que es la que decide, y lo que faltaba para el límite teórico del conjunto) y el veredicto de los
    archivos. Si la corrección todavía no dejó al original en la tarea, el héroe del Entregable 3 (la EBM)."""
    c = inicio.TAREA[tarea]
    t = D.lideres(tarea)
    r, e_v = D.veredicto_original(tarea), D.estado_veredicto(tarea)
    if O not in set(t["modelo"]) or r is None:
        return inicio._heroe(tarea)
    d, mejor, dm = c["d"], r["mejor_benchmark"], r["frente_al_mejor"]
    e, b = D.fila(t, modelo=O), D.fila(t, modelo=mejor)
    puesto_prueba = list(t.sort_values(c["prueba"], ascending=not c["mayor_es_mejor"], kind="stable")["modelo"]).index(O) + 1
    puesto_cv = sorted(r["medias"], key=r["medias"].get, reverse=c["mayor_es_mejor"]).index(O) + 1
    rechazos = r["rechazos_frente_al_mejor"]
    n_rech = sum(bool(v) for v in rechazos.values())
    lectura = (f"En la prueba queda {ORDINAL_M.get(puesto_prueba, F.entero(puesto_prueba))} de los {_cardinal(len(t))} modelos "
               f"({D.NOMBRE_MODELO[mejor]}, el mejor benchmark de la validación anidada, obtiene {ui.num(b[c['prueba']], d)}). En la "
               f"validación anidada, que es la que decide, queda {ORDINAL_M.get(puesto_cv, F.entero(puesto_cv))}: "
               f"{ui.num(r['medias'][O], d)} frente a {ui.num(r['medias'][mejor], d)} de {D.NOMBRE_MODELO[mejor]}. Con la regla de "
               f"superioridad de la guía, su veredicto es: {D.frase_veredicto(tarea)}.")
    ventaja = ui.saldo(f"Ventaja sobre {D.en_frase(mejor)} en la validación anidada", r["ventaja_sobre_el_mejor"],
                       f"A favor de BDL en {F.entero(dm['pliegues_a_favor'])} de {F.entero(r['pliegues'])} pliegues externos. "
                       f"{_mayuscula(OG._rechazan(n_rech, len(rechazos)))} la igualdad a su favor tras la corrección de Holm; la regla "
                       f"pide {OG.CARDINAL[OG.PRUEBAS_MINIMAS]}.",
                       unidad=f"de {c['metrica']} externo", d=d + 1, palabras=("a favor de BDL", "en contra de BDL"))
    te = D.original_json("techo.json")
    if te is not None:
        limite = float(te["prueba"]["auc_maximo" if tarea == "clasificacion" else "rmse_minimo"])
        falta, falta_b = abs(limite - e[c["prueba"]]), abs(limite - b[c["prueba"]])
        segundo = ui.saldo("Distancia al límite teórico del conjunto, en la prueba", falta,
                           f"El límite es {ui.num(limite, d)}: lo mejor que puede lograr un modelo sin la variable excluida. "
                           f"{_mayuscula(D.en_frase(mejor))} queda a {ui.num(falta_b, d + 1)}: el margen de mejora era mínimo.",
                           unidad=f"de {c['metrica']}", d=d + 1, extremo="lo que faltaba")
    else:
        diferencia = (e[c["prueba"]] - b[c["prueba"]]) * (1 if c["mayor_es_mejor"] else -1)
        segundo = ui.saldo(f"Diferencia con {D.en_frase(mejor)} en la prueba", diferencia,
                           f"{D.NOMBRE_MODELO[mejor]} obtiene {ui.num(b[c['prueba']], d)} "
                           f"{F.intervalo(b[c['inf']], b[c['sup']], d)}: los dos intervalos se solapan.",
                           unidad=f"de {c['metrica']}", d=d + 1, palabras=("a favor de BDL", "en contra de BDL"))
    lado = [ui.cifra("Condiciones de la regla que cumple", f"{F.entero(e_v['cumplidas'])} de {F.entero(e_v['evaluadas'])}", mini=True,
                     nota="De las que ya se pueden evaluar. La cuarta, los conjuntos de datos adicionales, está "
                          + {"cumple": "cumplida", "no cumple": "incumplida", "provisional": "en curso", "pendiente": "pendiente"}[e_v["c4"]]
                          + "."),
            ui.cifra("Pruebas que rechazan a su favor", f"{F.entero(n_rech)} de {F.entero(len(rechazos))}", mini=True,
                     nota="Frente al mejor benchmark, con los p ajustados por Holm."
                          + (" Resultado provisional: faltan repeticiones de la validación anidada." if e_v["provisional"] else ""))]
    return ui.heroe(f"{c['metrica']} de prueba de BDL, el modelo original, en los {F.entero(M.meta()['n_prueba'])} clientes que no vio",
                    ui.num(e[c["prueba"]], d), f"intervalo del 95 %: {F.intervalo(e[c['inf']], e[c['sup']], d)}", lectura,
                    [ventaja, segundo], lado=lado,
                    pie=ui.boton(f"Ver las {OG.CARDINAL[OG.CONDICIONES]} condiciones",
                                 href=app.ruta_vista("comparacion") + ("?tarea=regresion" if tarea == "regresion" else ""), icono="balanza"))


# --- 2. Por qué importa -------------------------------------------------------------------------------------------------
def _importa():
    meta, h = M.meta(), _hechos()
    t = D.lideres("clasificacion")
    foco = O if O in set(t["modelo"]) else "ebm"
    e, lin = D.fila(t, modelo=foco), D.fila(t, modelo=LINEAL)
    aditivos = [m.get("aditivo") for m in (D.modelo_original(x) for x in TAREAS) if m is not None]
    indice = (" El índice del modelo original, BDL, también lo es: sus modelos finales usan árboles de profundidad uno, una curva "
              "por variable." if aditivos and all(aditivos) else "")
    op = meta["punto_operacion"]
    d = _citacion()["datos"]
    empate = D.frase_empate_corta(foco, clientes=True)          # el corte de la campaña cae en un empate del modelo en foco
    campana = ui.tarjeta([
        ui.cifra(f"Caben en una campaña del {F.pct(op['fraccion'], 0)}", F.entero(op["contactados"]),
                 f"de {F.entero(meta['n_prueba'])} clientes", mini=True,
                 nota=f"Entre los {F.entero(meta['n_prueba'])} clientes de prueba hay {F.entero(meta['positivos_prueba'])} de riesgo alto "
                      f"({F.pct(meta['prevalencia_prueba'])})."),
        ui.parrafo(f"Contactando a los de mayor puntaje, {D.en_frase(foco)} alcanza a {F.entero(e['vp_op'])} de ellos y "
                   f"{D.en_frase(LINEAL)} a {F.entero(lin['vp_op'])}: el orden del modelo decide quién recibe la llamada."),
        *([ui.nota(f"La cifra de {D.en_frase(foco)} es la de {empate}.")] if empate else []),
    ], titulo="La campaña no alcanza para todos", fuentes=[F_META, F_CLF, D.fuente_original("lideres_clasificacion.csv")]
        + ([D.fuente_original("empates_campana.json")] if empate else []))
    huella = ui.tarjeta([
        ui.parrafo("En una fintech casi toda la relación con el cliente deja huella digital: uso de la aplicación, tenencia de productos, "
                   "satisfacción declarada y actividad transaccional."),
        ui.parrafo([f"COFINFAD describe a {F.entero(h['etl']['n_clientes'])} clientes de una fintech colombiana con "
                    f"{F.entero(h['etl']['n_columnas'])} columnas y {F.entero(h['tx_total'])} transacciones de "
                    f"{(h['fecha_detalle'] or '')[:4]} ({_cita_datos()}, ",
                    html.A(f"doi:{d['doi']}", href=f"https://doi.org/{d['doi']}", target="_blank", rel="noopener"), ")."]),
    ], titulo="Una fintech deja huella digital", fuentes=[F_ETL, "CITATION.cff"])
    explicar = ui.tarjeta([
        ui.parrafo(f"Los ensambles de árboles como Random Forest ({_cita('breiman')}) y XGBoost ({_cita('chen')}) ordenan bien, pero "
                   f"necesitan un explicador como SHAP ({_cita('lundberg')}) para decir por qué marcan a un cliente."),
        ui.parrafo(f"La EBM ({_cita('lou2012')}; {_cita('nori')}) es una suma de curvas que se pueden graficar y auditar.{indice} "
                   "En esos modelos la explicación es el propio modelo, y en un contexto financiero la explicación forma parte del "
                   "producto."),
    ], titulo="En finanzas hay que poder explicar la decisión")
    g = REFERENCIAS["gallo"]
    costo = ui.tarjeta([ui.rejilla([
        ui.cifra("Conseguir un cliente nuevo cuesta", f"5{UNION}–{UNION}25", "veces más que retener uno",
                 nota=f"Según el estudio y el sector ({_cita('gallo')}, Harvard Business Review)."),
        html.Div([
            ui.parrafo(["Es la cifra con la que suele justificarse la retención: ",
                        html.A(f"{g['corto']} ({g['anio']})", href=g["url"], target="_blank", rel="noopener"),
                        " la recoge en Harvard Business Review y advierte que el múltiplo depende del estudio y del sector. Por "
                        "eso conviene ordenar a los clientes por riesgo y llegar a tiempo a los que se van a ir."]),
            ui.parrafo("Es una cifra general, no medida en esta fintech: COFINFAD es sintético y no trae lo que cuesta captar ni retener "
                       "a un cliente, así que el proyecto no traduce la campaña a dinero. Incorporar el costo de cada acción de "
                       "retención queda como trabajo futuro."),
        ], className="apilado"),
    ], "1-2")], titulo="Retener cuesta menos que conseguir")
    return [costo, ui.rejilla([campana, huella, explicar], "3")]


# --- 3. Pregunta y objetivos ----------------------------------------------------------------------------------------------
def _pregunta():
    pregunta = ui.tarjeta([
        ui.parrafo(html.Strong("¿Supera a los benchmarks un modelo que trata como una clase latente el escalón que la variable "
                               "excluida deja en el puntaje, en lugar de tratarlo como ruido?")),
        ui.parrafo([html.Strong("Hipótesis. "), "En regresión, el RMSE del modelo original es menor que el del mejor benchmark; en "
                    "clasificación, su AUC es mayor. Se fijaron en un protocolo, antes de ejecutar los experimentos, junto con la regla "
                    "de superioridad de la guía."]),
        ui.parrafo([html.Strong("Respuesta. "), f"Con esa regla, el veredicto del modelo original es: {_veredictos()}. La app dice lo "
                    "que dicen los archivos de resultados, sea cual sea (", html.A("Conclusiones", href=f"#{PRE}-conclusiones"), ")."]),
    ], titulo="La pregunta de investigación",
        fuentes=[D.fuente_original("veredicto.json"), D.fuente_original("condicion4.json"), PROTOCOLO])

    def objetivo(titulo, texto, enlaces):
        partes = [html.Strong(f"{titulo}. "), texto, " "]
        for i, (rotulo, href) in enumerate(enlaces):
            enlace = html.A(rotulo, href=href) if href.startswith("#") else dcc.Link(rotulo, href=href)
            partes += ([" · "] if i else []) + [enlace]
        return ui.parrafo(partes)

    objetivos = ui.tarjeta([
        objetivo("Corregir las observaciones del docente", "La EBM viene de una librería y deja de presentarse como modelo original; "
                 "la recencia se calcula solo con los clientes de entrenamiento.", [("Las correcciones", f"#{PRE}-correcciones")]),
        objetivo("Diseñar, programar y sustentar un modelo original", f"BDL ({D.NOMBRE_LARGO_ORIGINAL}): su formulación, su "
                 "pseudocódigo, sus proposiciones demostradas y su código, escrito desde cero.",
                 [("Modelo original", app.ruta_vista("original")), ("Ficha del modelo", app.ruta_vista("ficha"))]),
        objetivo("Compararlo con los benchmarks de forma justa", f"Frente a {_cardinal(_benchmarks())} benchmarks, con el protocolo "
                 "registrado: validación anidada repetida, cuatro pruebas de hipótesis por tarea, corrección de Holm y tamaños del "
                 "efecto. El resultado se reporta sea cual sea.",
                 [("Tabla de líderes", app.ruta_vista("lideres")), ("Comparación estadística", app.ruta_vista("comparacion"))]),
        objetivo("Medir qué sostiene el resultado", "El techo teórico del problema, la ablación, la estabilidad por semilla, la "
                 "sensibilidad, el ruido, el tamaño de muestra, otros conjuntos de datos y la incertidumbre.",
                 [("Modelo original", app.ruta_vista("original"))]),
        objetivo("Llevarlo a un producto en vivo", "Los modelos finales predicen en el momento, con su incertidumbre y su explicación; "
                 "el modelo original muestra, además, su conjunto de predicción.", [("Predicción en vivo", app.ruta_vista("prediccion"))]),
    ], titulo="Los objetivos")
    return ui.rejilla([pregunta, objetivos], "1-2")


# --- Las correcciones tras la evaluación del proyecto final -----------------------------------------------------------------
def _correcciones():
    """Las dos correcciones del proyecto final: el modelo original ya no es una librería y la recencia se calcula solo con el
    entrenamiento (datos/original/auditoria_fuga.json, que copia la auditoría de la corrección)."""
    estado = D.estado_original()
    codigo = (estado.get("codigo") or {})
    pares = [("Modelo original", f"BDL ({D.NOMBRE_LARGO_ORIGINAL})"), ("Código", "src/original_model.py, con numpy y scipy"),
             ("Benchmarks recientes", "EBM (librería interpret) y LightGBM")]
    if codigo.get("sha256"):
        pares.append(("Huella del código", f"sha256 {codigo['sha256'][:LARGO_HUELLA]}… (la misma que en ModeloOriginal/)"))
    propio = ui.tarjeta([
        ui.parrafo("La primera versión del proyecto final presentaba como modelo original a la EBM, que se usa a través de una "
                   "librería: para la guía eso es un import. El modelo original es ahora BDL, programado desde cero: discretización, "
                   "árboles, el algoritmo EM de la mezcla y la salida conforme. La EBM queda como benchmark reciente e "
                   "interpretable, y se añade LightGBM como segundo benchmark reciente."),
        ui.ficha(pares),
        html.Div(ui.boton("Ver el modelo original", href=app.ruta_vista("original"), icono="modelo"), className="botonera"),
    ], titulo="El modelo original ya no viene de una librería", fuentes=[D.fuente_original("estado.json"), "src/original_model.py"])

    a = D.original_json("auditoria_fuga.json")
    if a is None:
        recencia = OG._pendiente("La auditoría de la corrección de la recencia", "auditoria_fuga.json", "la escribe 01_auditoria_fuga.py")
    else:
        f, p, pl = a["fecha_de_corte"], a["particion"], a.get("pliegues", {})
        igual = bool(f["coinciden"]) and bool(f["predictores_identicos"])
        titulo = ("La fecha de corte de la recencia es la misma con todo el detalle y solo con el entrenamiento: no cambia ningún "
                  "predictor" if igual else
                  f"La corrección de la recencia cambia {F.entero(f['celdas_de_recencia_que_cambian'])} celdas de los predictores")
        ficha = [("Fecha de corte con todo el detalle (Entregable 3)", _fecha(f["entregable_3_todo_el_detalle"])),
                 ("Fecha de corte solo con el entrenamiento (corrección)", _fecha(f["correccion_solo_entrenamiento"])),
                 ("Celdas de la recencia que cambian", F.entero(f["celdas_de_recencia_que_cambian"])),
                 ("Predictores de entrenamiento y de prueba", "idénticos antes y después (misma huella)" if f["predictores_identicos"]
                  else "distintos"),
                 ("Clientes que están en las dos particiones", F.entero(p["clientes_en_ambas"]))]
        if pl:
            ficha.append(("Pliegues con otra fecha de corte", f"{F.entero(pl['pliegues_con_otra_fecha_de_corte'])} de "
                                                              f"{F.entero(pl['pliegues_revisados'])} revisados"))
        consecuencia = ("Por eso las corridas del Entregable 3 siguen siendo válidas y se reutilizan tal cual."
                        if igual else "Los resultados que dependen de la recencia se recalcularon con el flujo corregido.")
        recencia = ui.tarjeta([
            ui.parrafo("La recencia se calculaba con la fecha máxima de todo el detalle de transacciones, que incluye a los clientes "
                       "de prueba. Ahora la partición se hace antes de derivar variables y la fecha de corte sale solo de los "
                       f"clientes de entrenamiento. La auditoría compara las dos versiones. {consecuencia}"),
            ui.ficha(ficha),
        ], titulo=titulo, fuentes=[D.fuente_original("auditoria_fuga.json")])
    return [ui.rejilla([propio, recencia], "1-1")]


# --- 4. La formulación ----------------------------------------------------------------------------------------------------
def _formulacion():
    h, meta = _hechos(), M.meta()
    tau, q = h["tau"], meta["etiqueta"]["q"]
    n, n_te = F.entero(meta["n_entrenamiento"]), F.entero(meta["n_prueba"])
    clasificacion = ui.tarjeta([
        _formula(r"z_i = \mathbf{1}\left[\, s_i > \tau \,\right]"),
        _texto_math(f"con $\\tau = {_tex(ui.num(tau, 4))}$, el percentil {ui.num(q * 100)} del puntaje en los {n} clientes de "
                    f"entrenamiento. Riesgo alto es un puntaje por encima del umbral, no fuga observada: {F.pct(h['prev_train'])} de "
                    f"positivos en el entrenamiento y {F.pct(h['prev_test'])} en la prueba. La desigualdad es estricta: "
                    f"{F.entero(h['n_empates'])} clientes de entrenamiento tienen exactamente el puntaje del umbral y quedan en riesgo bajo."),
        _texto_math(r"Se busca $f_C : \mathcal{X} \to [0, 1]$ que minimice la entropía cruzada ponderada"),
        _formula(r"\mathcal{L}_C = -\frac{1}{n} \sum_{i=1}^{n} w_i \, \ell_i",
                 r"\ell_i = z_i \ln \hat{p}_i + (1 - z_i) \ln (1 - \hat{p}_i)"),
        _texto_math(r"con $w_i = n / (2\, n_{z_i})$ si la corrida pondera las clases y $w_i = 1$ si no. La métrica principal es"),
        _formula(r"\mathrm{AUC} = P\left( \hat{p}_i > \hat{p}_j \mid z_i = 1,\ z_j = 0 \right)"),
        ui.nota(f"La probabilidad de que el modelo ponga a un cliente de riesgo alto por encima de uno de riesgo bajo: no depende del "
                f"umbral ni de la prevalencia y mide lo que necesita la campaña, ordenar. {ui.num(AZAR_AUC, 1)} es el azar."),
    ], titulo="Clasificación binaria: el cuarto superior del puntaje", subtitulo="Tarea 1. Etiqueta, pérdida y métrica principal.",
        fuentes=[F_META, D.fuente("eda/entrenamiento.parquet")])
    regresion = ui.tarjeta([
        _texto_math(f"El objetivo es el puntaje mismo, $s_i \\in [{_tex(ui.num(h['s_min'], 3))};\\ {_tex(ui.num(h['s_max'], 3))}]$ en "
                    "el entrenamiento: nunca vale 0 ni 1, porque no es un evento observado sino un puntaje de propensión. Se busca "
                    r"$f_R : \mathcal{X} \to \mathbb{R}$ que minimice el error cuadrático medio"),
        _formula(r"\mathcal{L}_R = \frac{1}{n} \sum_{i=1}^{n} \left( s_i - \hat{s}_i \right)^2",
                 r"\mathrm{RMSE} = \sqrt{ \tfrac{1}{n_{\mathrm{te}}} \textstyle\sum_{i} \left( s_i - \hat{s}_i \right)^2 }"),
        ui.nota("El RMSE va en las unidades del puntaje (menor es mejor); se informan también el MAE y el R²."),
    ], titulo="Regresión: el puntaje de propensión", subtitulo="Tarea 2. Objetivo, pérdida y métrica principal.",
        fuentes=[D.fuente("eda/entrenamiento.parquet")])
    predictores = ui.tarjeta([
        _formula(r"\mathbf{x}_i \in \mathcal{X} \subset \mathbb{R}^{p}, \qquad p = " + _tex(F.entero(h["p"]))),
        ui.parrafo(f"{F.entero(h['n_pred'])} predictores: {F.entero(h['n_num'] + h['n_bool'])} numéricos y booleanos "
                   f"({F.entero(h['n_num'])} numéricos y {F.entero(h['n_bool'])} booleanos) y {F.entero(h['n_cat'])} categóricos. Las "
                   "categóricas pasan a indicadoras (sin un nivel de referencia) y las numéricas y booleanas se estandarizan, dentro de "
                   f"cada pliegue: quedan {F.entero(h['p'])} columnas."),
        ui.nota("active_products, la variable con que se generó el puntaje, no es un predictor (sección Limitaciones)."),
    ], titulo=f"Los {F.entero(h['n_pred'])} predictores", subtitulo="Las dos tareas usan los mismos.", fuentes=[F_ETL])
    validacion = ui.tarjeta([
        _formula(r"n = " + _tex(n) + r", \qquad n_{\mathrm{te}} = " + _tex(n_te)),
        ui.parrafo(f"Partición {_particion()} por índices con semilla {meta['semilla']}, antes de calcular cualquier estadístico del "
                   "objetivo: el umbral sale solo del entrenamiento y la prueba se toca una vez, con el modelo final."),
        ui.parrafo(f"Dentro del entrenamiento, validación cruzada anidada {h['k_ext']} × {h['k_int']} ({_cita('cawley')}): los "
                   f"{_cardinal(h['k_ext'])} pliegues externos dan la métrica que se informa y los {_cardinal(h['k_int'])} internos eligen "
                   f"los hiperparámetros con uno de {_cardinal(h['n_optimizadores'])} optimizadores, todos con el mismo presupuesto (de "
                   f"{h['presupuesto_min']} a {h['presupuesto_max']} evaluaciones por búsqueda)."),
    ], titulo="Partición y validación", subtitulo="Cómo se elige y cómo se mide sin mirar la prueba.", fuentes=[F_META, F_CORRIDAS])
    return ui.rejilla([clasificacion, regresion, predictores, validacion], "1-1")


# --- 5. Los datos ---------------------------------------------------------------------------------------------------------
def _sinteticos():
    h = _hechos()
    edades = F.enumerar(str(e) for e in h["edades"])
    escalones = ", ".join(ui.num(x, 4) for x in h["escalera"])
    return ui.aviso(
        f"COFINFAD tiene la firma de un conjunto generado por reglas: la edad solo toma {_cardinal(len(h['edades']))} valores ({edades}) "
        f"en {_cardinal(h['cohortes'])} cohortes, la media del puntaje baja en escalones casi idénticos con cada producto activo "
        f"({escalones}) y las {_cardinal(h['n_deriv'])} variables transaccionales no tienen señal (|ρ| ≤ {ui.num(h['rho_derivadas'], 3)}). "
        "Lo que se aprende aquí son esas reglas, no el comportamiento de clientes reales.",
        "atencion", "Los datos son sintéticos")


def _datos():
    h, meta = _hechos(), M.meta()
    d, cita = _citacion()["datos"], _citacion()
    etl = h["etl"]
    aud = etl["auditoria_transacciones"]
    anio = (h["fecha_detalle"] or "")[:4]
    cifras = ui.cifras([
        ui.cifra("Clientes", F.entero(etl["n_clientes"]), nota="Una fila por cliente.", mini=True),
        ui.cifra("Columnas de la tabla de clientes", F.entero(etl["n_columnas"]), nota="Perfil, productos, uso digital, satisfacción, "
                 "resúmenes transaccionales y los objetivos.", mini=True),
        ui.cifra("Transacciones del detalle", F.entero(h["tx_total"]), nota=f"{F.entero(h['duplicadas'])} duplicadas, que se eliminaron.",
                 mini=True),
        ui.cifra("Riesgo alto en el entrenamiento", F.pct(h["prev_train"]),
                 nota=f"El cuarto superior del puntaje (τ{F.FIJO}={F.FIJO}{ui.num(h['tau'], 4)}).", mini=True),
    ])
    fuente = ui.tarjeta([ui.ficha([
        ("Conjunto", d["titulo"] or "COFINFAD"),
        ("Autores", f"{_autores_cita(d['autores'])} ({d['anio']})"),
        ("Publicado en", f"{d['editor']}, versión {d['version']}"),
        ("DOI", html.A(d["doi"], href=f"https://doi.org/{d['doi']}", target="_blank", rel="noopener")),
        ("Licencia", html.A(d["licencia"], href=LICENCIA_CC_BY, target="_blank", rel="noopener")),
        ("Periodo", f"{anio}; la última transacción del detalle es del {_fecha(h['fecha_detalle'])}"),
        ("Unidad de observación", "el cliente"),
    ])], titulo="Fuente, periodo y licencia", fuentes=["CITATION.cff", D.fuente("variables.csv")])
    objetivo = ui.tarjeta([ui.ficha([
        ("Variable objetivo", "churn_probability, un puntaje de propensión a la fuga"),
        ("Recorrido en el entrenamiento", F.intervalo(h["s_min"], h["s_max"], 3)),
        ("Clasificación", f"riesgo alto si el puntaje supera τ{F.FIJO}={F.FIJO}{ui.num(h['tau'], 4)}"),
        ("Regresión", "el puntaje mismo"),
        ("Métricas principales", "AUC (clasificación) y RMSE (regresión)"),
        ("Tarea", "aprendizaje supervisado: clasificación binaria y regresión"),
        ("Entrenamiento y prueba", f"{F.entero(meta['n_entrenamiento'])} y {F.entero(meta['n_prueba'])} clientes"),
    ])], titulo="Objetivo y tipo de tarea", fuentes=[F_META, D.fuente("eda/entrenamiento.parquet")])
    limpieza = ui.tarjeta([ui.ficha([
        ("Transacciones duplicadas", f"{F.entero(aud['duplicadas'])}, eliminadas"),
        ("Montos no positivos, fechas nulas y huérfanas", f"{F.entero(aud['montos_no_positivos'])}, {F.entero(aud['fechas_nulas'])} y "
                                                          f"{F.entero(aud['huerfanas'])}"),
        ("Variables derivadas del detalle", f"{F.entero(h['n_deriv'])}, ninguna usa el objetivo"),
        ("Columnas eliminadas", f"{F.entero(h['n_elim'])}, con su motivo en el diccionario"),
        ("Faltante que se conserva", f"uso del cupo de la tarjeta ({F.pct(h['falta_cupo'])} del entrenamiento): falta cuando no hay "
                                     "tarjeta, así que se imputa con 0"),
        ("Atípicos", "no se recortan: son clientes muy activos, no errores"),
    ])], titulo="Lo que hizo el ETL", fuentes=[F_ETL, F_FALTANTES])
    return [_sinteticos(), cifras, ui.rejilla([fuente, objetivo, limpieza], "3")]


# --- El diccionario de variables -------------------------------------------------------------------------------------------
def _decimales(maximo):
    return 0 if abs(maximo) >= 1000 else 2 if abs(maximo) >= 10 else 3


def _rango(v):
    if v["tipo"] == "booleana":
        return "sí o no"
    if v["tipo"] == "categorica":
        cats = list(v["categorias"])
        lista = " · ".join(cats)
        return f"{len(cats)} categorías" + (f": {lista}" if len(lista) <= LARGO_LISTA else "")
    d = 0 if v["tipo"] == "entera" else _decimales(v["max"])
    return f"{ui.num(v['min'], d)} a {ui.num(v['max'], d)}"


def _razon(motivo, columna, falta):
    h = _hechos()
    duplicadas = sum(1 for *_x, m in ELIMINADAS.values() if m == "duplica")
    textos = {
        "identificador": "Identificador del cliente, sin valor predictivo.",
        "fuga": (f"Fuga de información: el puntaje se construyó a partir de ella (r = {ui.num(h['r_ap'], 3)}; explica el "
                 f"{F.pct(h['r_ap'] ** 2)} de su varianza en el entrenamiento)."),
        "segundo_objetivo": "Es el segundo objetivo del proyecto (el valor del cliente y su segmento).",
        "duplica": f"Uno de los {_cardinal(duplicadas)} resúmenes transaccionales que duplican a otras columnas.",
        "fecha_duplica": "Fecha que duplica a otra fecha.",
        "fecha_sin_senal": "Fecha sin señal con el objetivo.",
        "colineal": "Colineal con la satisfacción.",
        "ocupacion": "Muchos niveles y ninguna señal con el objetivo.",
        "texto": (f"Texto con el {F.pct(falta)} de valores ausentes en el entrenamiento, sin mecanismo ni señal." if falta is not None
                  else "Texto sin mecanismo ni señal."),
    }
    return textos.get(motivo, "Eliminada en el ETL.")


@lru_cache(maxsize=1)
def _filas_diccionario():
    """Una fila por columna: los 48 predictores, el objetivo y las columnas eliminadas (lo que el usuario filtra y ordena)."""
    falt = D.tabla("eda/faltantes.csv").set_index("columna")["fraccion"]
    asoc = D.tabla("eda/asociacion.csv").set_index("variable")
    E = V.entrenamiento()
    etl = D.json("eda/etl.json")

    def faltantes(col):
        if col in falt.index:
            return round(100 * float(falt[col]), 1)
        return round(100 * float(E[col].isna().mean()), 1) if col in E.columns else None

    def asociacion(col):
        if col not in asoc.index:
            return "", None
        return MEDIDA.get(asoc.at[col, "medida"], asoc.at[col, "medida"]), round(float(asoc.at[col, "valor"]), 4)

    filas = []
    for v in V.esquema():
        medida, valor = asociacion(v["nombre"])
        filas.append({"variable": v["nombre"], "nombre": v.get("nombre_visible") or v["nombre"], "uso": "Predictor",
                      "tipo": TIPO.get(v["tipo"], v["tipo"]), "unidad": v.get("unidad") or "—", "descripcion": v.get("descripcion") or "",
                      "rango": _rango(v), "faltantes_num": faltantes(v["nombre"]), "medida": medida, "asociacion_num": valor,
                      "derivada": "sí" if v.get("derivada_etl") else "no"})
    s = E["churn_probability"]
    etiqueta = M.meta()["etiqueta"]
    filas.append({"variable": "churn_probability", "nombre": "Puntaje de propensión a la fuga", "uso": "Objetivo",
                  "tipo": "numérica real", "unidad": "—",
                  "descripcion": (f"El objetivo de la regresión. La clasificación marca riesgo alto cuando supera el umbral τ = "
                                  f"{ui.num(etiqueta['umbral'], 4)}, el percentil {ui.num(etiqueta['q'] * 100)} del entrenamiento."),
                  "rango": f"{ui.num(s.min(), 3)} a {ui.num(s.max(), 3)}", "faltantes_num": faltantes("churn_probability"), "medida": "",
                  "asociacion_num": None, "derivada": "no"})
    for col in etl["eliminadas"]:
        nombre, tipo, motivo = ELIMINADAS.get(col, (col, "—", ""))
        medida, valor = asociacion(col)
        rango = (f"{F.entero(E[col].min())} a {F.entero(E[col].max())}" if col in E.columns and pd.api.types.is_numeric_dtype(E[col])
                 else "—")
        f = faltantes(col)
        filas.append({"variable": col, "nombre": nombre, "uso": f"Eliminada: {MOTIVO_CORTO.get(motivo, 'en el ETL')}", "tipo": tipo,
                      "unidad": "—", "descripcion": _razon(motivo, col, None if f is None else f / 100), "rango": rango,
                      "faltantes_num": f, "medida": medida, "asociacion_num": valor, "derivada": "no"})
    return tuple(filas)


def _filas(grupo):
    filas = [dict(f) for f in _filas_diccionario()]
    if grupo == "predictores":
        return [f for f in filas if f["uso"] == "Predictor"]
    if grupo == "fuera":
        return [f for f in filas if f["uso"] != "Predictor"]
    return filas


def _numero(decimales, sufijo=""):
    """Formato es-CO de una columna numérica de la tabla (coma decimal, punto de miles; el negativo sale con «−»)."""
    formato = dash_table.Format.Format(precision=decimales, scheme=dash_table.Format.Scheme.fixed,
                                       decimal_delimiter=",", group_delimiter=".")
    return formato.symbol(dash_table.Format.Symbol.yes).symbol_suffix(sufijo) if sufijo else formato


# Las columnas cortas primero (la tabla se desplaza en horizontal); la descripción, con el motivo de cada eliminación, al final
COLUMNAS_DICC = [
    {"name": "Variable", "id": "variable"},
    {"name": "Nombre", "id": "nombre"},
    {"name": "Uso", "id": "uso"},
    {"name": "Tipo", "id": "tipo"},
    {"name": "Unidad", "id": "unidad"},
    {"name": "Rango en el entrenamiento", "id": "rango"},
    {"name": "Faltantes", "id": "faltantes_num", "type": "numeric", "format": _numero(1, F.FIJO + "%")},
    {"name": "Asociación con el puntaje", "id": "asociacion_num", "type": "numeric", "format": _numero(4)},
    {"name": "Medida", "id": "medida"},
    {"name": "Derivada del ETL", "id": "derivada"},
    {"name": "Descripción y motivo", "id": "descripcion"},
]


def _diccionario():
    filas = _filas("todas")
    tabla = dash_table.DataTable(id=f"{PRE}-dicc", columns=COLUMNAS_DICC, data=filas, filter_action="native", sort_action="native",
                                 filter_options={"case": "insensitive", "placeholder_text": "Filtrar…"}, page_action="none")
    n_pred = sum(f["uso"] == "Predictor" for f in filas)
    return ui.tarjeta([
        html.P(f"Se muestran las {len(filas)} columnas.", id=f"{PRE}-dicc-cuenta", className="nota"),
        html.Div(tabla, className="tabla-datos tabla-marco tabla-marco--alta"),
        ui.nota("Escriba en la fila de filtros para buscar (por ejemplo, «fuga» en Uso o «fecha» en Tipo) y ordene con las flechas del "
                "encabezado; la tabla se desplaza hacia la derecha hasta la descripción y el motivo de cada eliminación. Los rangos, los "
                "faltantes y la asociación (ρ de Spearman o η² con el puntaje) se calculan solo con el entrenamiento; los faltantes son "
                "los de la tabla original, antes de imputar."),
    ], titulo=f"{F.entero(n_pred)} predictores, el objetivo y {F.entero(len(filas) - n_pred - 1)} columnas eliminadas",
        fuentes=["artifacts/clasificacion/ebm/metadata.json", D.fuente("variables.csv"), F_ETL, F_FALTANTES,
                 D.fuente("eda/asociacion.csv"), D.fuente("eda/entrenamiento.parquet")])


# --- 6. La metodología ------------------------------------------------------------------------------------------------------
ICONO_VISTA = {v: icono for v, _n, _p, _g, _m, icono in app.VISTAS}


def _paso(i, href, titulo, texto, pantalla="datos", vista=None):
    return ui.destino(href, pantalla, f"{i}. {titulo}", texto, icono=ICONO_VISTA.get(vista) if vista else None)


def _texto_incertidumbre(cal, conf):
    """El paso de calibración e incertidumbre del pipeline: con el modelo original, su conjunto de predicción frente al
    intervalo conforme del mejor benchmark; si todavía no está, lo del Entregable 3 (la EBM)."""
    nivel = F.pct(float(NIVEL_CONFORME), 0)
    c = V.conformal()["modelos"]
    propio = (c.get(O, {}).get(NIVEL_CONFORME) or {}).get("conjunto")
    r = D.veredicto_original("regresion")
    mejor = r["mejor_benchmark"] if r is not None and r["mejor_benchmark"] in c else RIVAL
    if propio is None:
        return (f"La isotónica baja el ECE de la EBM de {ui.num(cal['ece_prueba_sin_recalibrar'], 3)} a "
                f"{ui.num(cal['ece_prueba_recalibrada'], 3)}; el intervalo conforme del {nivel} cubre el "
                f"{F.pct(conf['cobertura_prueba'])} de la prueba en regresión.")
    return (f"En regresión, el conjunto de predicción de BDL al {nivel} cubre el {F.pct(propio['cobertura_prueba'])} de la prueba con "
            f"un largo de {ui.num(propio['largo'], 4)}; el intervalo conforme {_tras_de(mejor)} mide "
            f"{ui.num(c[mejor][NIVEL_CONFORME]['ancho'], 4)}. La isotónica baja el ECE de la EBM de "
            f"{ui.num(cal['ece_prueba_sin_recalibrar'], 3)} a {ui.num(cal['ece_prueba_recalibrada'], 3)}.")


def _tras_de(modelo):
    """«de XGBoost», «del árbol de decisión», «de la EBM»."""
    frase = D.en_frase(modelo)
    return "del" + frase[2:] if frase.startswith("el ") else f"de {frase}"


def _pipeline():
    h, meta = _hechos(), M.meta()
    refs = len(D.pruebas("clasificacion"))
    cal = V.calibracion()["modelos"]["ebm"]
    conf = V.conformal()["modelos"]["ebm"][NIVEL_CONFORME]
    pasos = [
        ("/eda", "ETL y limpieza", f"Auditoría del detalle ({F.entero(h['duplicadas'])} duplicadas fuera), {_cardinal(h['n_deriv'])} variables "
         f"derivadas, {F.entero(h['n_elim'])} columnas eliminadas y {F.entero(h['n_pred'])} predictores.", "datos", None),
        ("/eda", "Partición y EDA solo con el entrenamiento", f"{_particion()} con semilla {meta['semilla']}: {F.entero(meta['n_entrenamiento'])} "
         f"clientes para explorar y entrenar; los {F.entero(meta['n_prueba'])} de prueba no se miran hasta el final.", "datos", None),
        (app.ruta_vista("lideres"), f"{_mayuscula(_cardinal(h['n_modelos']))} modelos × balanceos × optimizadores, más BDL y LightGBM",
         f"Clasificación: {h['n_modelos']} × {h['n_balanceos']} × {h['n_optimizadores']} = {F.entero(h['corridas_clf'])} corridas "
         f"({_cardinal(h['no_aplican'])} no aplican: k-NN con ponderación). Regresión: {h['n_modelos']} × {h['n_optimizadores']} = "
         f"{F.entero(h['corridas_reg'])}. En total {F.entero(h['corridas'])}: {F.entero(h['corridas_guia'])} de la guía y "
         f"{F.entero(h['corridas_ebm'])} de la EBM. El modelo original y LightGBM compiten con una sola combinación, fijada en el "
         "protocolo.", "comparacion", "lideres"),
        (app.ruta_vista("comparacion"), f"Validación anidada {h['k_ext']} × {h['k_int']}",
         f"Los pliegues externos miden y los internos eligen los hiperparámetros; {ui.num(h['horas'], 1)} horas de cómputo en un solo "
         "equipo.", "comparacion", "comparacion"),
        (app.ruta_vista("lideres"), "La mejor combinación de cada modelo", "Se elige por la métrica externa, sin mirar la prueba, y se "
         "reajusta con todo el entrenamiento: es el modelo final, que toca la prueba una sola vez.", "comparacion", "lideres"),
        (app.ruta_vista("original"), "El modelo original: BDL", "Un boosting de árboles propios que modela como una clase latente el "
         "escalón del puntaje y lo aprende con un algoritmo EM generalizado; con su pseudocódigo, sus proposiciones y su convergencia.",
         "modelo", "original"),
        (app.ruta_vista("comparacion"), "Comparación estadística", f"El modelo original frente a los {_cardinal(_benchmarks())} "
         "benchmarks, con el protocolo registrado: Friedman y Nemenyi, Wilcoxon, t corregida, DeLong o permutación pareada, McNemar o "
         f"bootstrap pareado, Holm y tamaños del efecto. Aparte, la EBM frente a sus {_cardinal(refs)} referencias del Entregable 3.",
         "comparacion", "comparacion"),
        (app.ruta_vista("original"), "Robustez y veredicto", "Ablación, semillas, sensibilidad, ruido, tamaño de muestra, otros "
         "conjuntos de datos e incertidumbre del modelo original, y su veredicto con las cuatro condiciones de la guía. La robustez "
         "de la EBM está en su vista.", "robustez", "robustez"),
        (app.ruta_vista("diagnostico"), "Calibración e incertidumbre", _texto_incertidumbre(cal, conf), "comparacion", "diagnostico"),
        (app.ruta_vista("prediccion"), "El dashboard", f"Los {F.entero(len(V.indice()['artefactos']))} modelos finales guardados (los de las "
         "dos tareas, menos LightGBM, que solo aparece en las tablas), en vivo y sin reentrenar: predicción, incertidumbre y explicación "
         "de cada cliente.", "cliente", "prediccion"),
    ]
    return [ui.destinos([_paso(i, *p) for i, p in enumerate(pasos, 1)])]


# --- 7. Dos fidelidades, no Successive Halving ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _fidelidades():
    """Por modelo con búsqueda de dos fidelidades: el hiperparámetro, su valor en la búsqueda y en el modelo final, y cuántas
    configuraciones evalúa cada búsqueda (corridas.csv, columnas fidelidad y presupuesto)."""
    c = D.tabla("entrega3/corridas.csv")
    out = {}
    for modelo in c.loc[c["fidelidad"].notna(), "modelo"].unique():
        f = json.loads(c.loc[(c["modelo"] == modelo) & c["fidelidad"].notna(), "fidelidad"].iloc[0])
        clave = next(iter(f["busqueda"]))
        out[str(modelo)] = {"clave": clave, "baja": f["busqueda"][clave], "alta": f["final"][clave],
                            "n": int(c.loc[c["modelo"] == modelo, "presupuesto"].iloc[0])}
    return out


def _sh(n, r):
    """Las rondas de Successive Halving con n configuraciones y factor r: (configuraciones, presupuesto en b₀, pasan)."""
    rondas, presupuesto = [], 1
    while n > 1:
        pasan = max(1, n // r)
        rondas.append((n, presupuesto, pasan))
        n, presupuesto = pasan, presupuesto * r
    return rondas


def _fidelidad():
    fid = _fidelidades()
    res = D.clave_valor("resultados/fidelidad_resumen.csv")
    tabla_fid = D.tabla("entrega3/computo/fidelidad.csv")
    k_ext = _hechos()["k_ext"]
    ebm, rf = fid.get("ebm"), fid.get("random_forest")
    if not ebm or not rf:
        return [ui.aviso("corridas.csv no trae la columna de fidelidad de la EBM o de Random Forest.", "atencion", "Faltan datos")]

    def etiqueta(f, valor):
        return f"{PARAMETRO[f['clave']][0]}{F.FIJO}={F.FIJO}{F.entero(valor)}"

    hecho = ui.tarjeta([
        ui.ficha([
            ("EBM, durante la búsqueda", f"{etiqueta(ebm, ebm['baja'])} {PARAMETRO[ebm['clave']][1]}"),
            ("EBM, modelo final", f"{etiqueta(ebm, ebm['alta'])} {PARAMETRO[ebm['clave']][1]}"),
            ("Random Forest, durante la búsqueda", f"{etiqueta(rf, rf['baja'])} {PARAMETRO[rf['clave']][1]}"),
            ("Random Forest, modelo final", f"{etiqueta(rf, rf['alta'])} {PARAMETRO[rf['clave']][1]}"),
            ("Configuraciones por búsqueda", f"{F.entero(ebm['n'])}, en cada pliegue externo ({_cardinal(k_ext)}) y en la búsqueda final"),
        ]),
        ui.parrafo("Cada configuración se evalúa en la validación interna con el modelo barato; la ganadora se reajusta con el modelo "
                   "completo y solo ella se mide en el bucle externo. La elección entera se hace con la fidelidad baja: la ganadora nunca "
                   "se compara con las demás en la alta."),
        ui.parrafo([f"Por eso el esquema vale si la fidelidad baja conserva el orden, y el informe lo comprobó: en "
                    f"{_cardinal(int(res['configuraciones']))} configuraciones aleatorias de la EBM, la ρ de Spearman entre el AUC con "
                    f"{etiqueta(ebm, ebm['baja'])} y con {etiqueta(ebm, ebm['alta'])} es {ui.num(res['rho_spearman'], 2)}, la ganadora es "
                    f"{'la misma' if str(res['misma_ganadora']) == 'True' else 'otra'} y se ahorra el {F.pct(res['ahorro_tiempo'], 0)} del "
                    f"tiempo; en Random Forest, con {_cardinal(int((tabla_fid['modelo'] == 'random_forest').sum()))} configuraciones, "
                    f"ρ{F.FIJO}={F.FIJO}{ui.num(res['rho_rf'], 2)}. ", dcc.Link("Ver en Robustez", href=app.ruta_vista("robustez")), "."]),
    ], titulo="Lo que hizo el proyecto: buscar barato y reajustar a la ganadora", subtitulo="Lo que el informe llama «multi-fidelidad».",
        fuentes=[F_CORRIDAS, D.fuente("resultados/fidelidad_resumen.csv"), D.fuente("entrega3/computo/fidelidad.csv")])

    teoria = ui.tarjeta([
        _texto_math(f"Successive Halving ({_cita('jamieson')}) empieza con $n$ configuraciones y un presupuesto pequeño $b_0$ (bolsas, "
                    "árboles, iteraciones). En cada ronda evalúa a las que siguen con el presupuesto de la ronda, conserva la fracción "
                    "$1/r$ de las mejores y multiplica por $r$ el presupuesto de las que pasan:"),
        _formula(r"n_{k+1} = \left\lfloor n_k / r \right\rfloor, \qquad b_{k+1} = r \, b_k"),
        _texto_math(f"hasta que queda una, elegida con el presupuesto mayor. $r$ es el factor de reducción, habitualmente "
                    f"{FACTOR_SH}; Hyperband ({_cita('li')}) repite Successive Halving con varias parejas $(n, b_0)$ para no depender de "
                    "esa elección. Aquí se escribe $r$ porque en el informe $\\eta$ es la tasa de aprendizaje de la EBM."),
    ], titulo="Successive Halving: rondas que descartan y suben el presupuesto", subtitulo="La técnica que el esquema no es.")

    rondas = _sh(ebm["n"], FACTOR_SH)
    filas = [["Successive Halving", F.entero(k), F.entero(n), "b₀" if b == 1 else f"{F.entero(b)} b₀", F.entero(pasan), "—"]
             for k, (n, b, pasan) in enumerate(rondas, 1)]
    for nombre, f in (("EBM", ebm), ("Random Forest", rf)):
        filas.append([f"Este proyecto: {nombre}", "única", F.entero(f["n"]), etiqueta(f, f["baja"]), "1",
                      f"{etiqueta(f, f['alta'])} (× {F.entero(f['alta'] / f['baja'])})"])
    comparacion = ui.tarjeta([
        ui.tabla(["Esquema", "Ronda", "Configuraciones", "Presupuesto de cada una", "Pasan", "Reajuste de la ganadora"], filas,
                 compacta=True, num=[False, True, True, False, True, False]),
        ui.parrafo(f"Con esa notación, el esquema del proyecto es una sola ronda que conserva una configuración de {F.entero(ebm['n'])} "
                   f"(la fracción 1/{F.entero(ebm['n'])}, no 1/r) y multiplica el presupuesto de la ganadora por "
                   f"{F.entero(ebm['alta'] / ebm['baja'])} en la EBM y por {F.entero(rf['alta'] / rf['baja'])} en Random Forest. No es "
                   "Successive Halving: no hay rondas intermedias y ninguna configuración compite en alta fidelidad. Es una búsqueda "
                   "con una aproximación barata del modelo, validada por la conservación del orden."),
    ], titulo=f"Lado a lado: Successive Halving con las mismas {F.entero(ebm['n'])} configuraciones y r{F.FIJO}={F.FIJO}{FACTOR_SH}, y lo que hizo el "
              "proyecto", fuentes=[F_CORRIDAS])
    return [ui.rejilla([hecho, teoria], "1-1"), comparacion]


# --- 8. Limitaciones, ética y sesgos ------------------------------------------------------------------------------------------
def _no_permite():
    """Lo que el trabajo no permite afirmar (las limitaciones del informe, como en secciones/limites.py)."""
    clf, reg = D.pruebas("clasificacion"), D.prueba("regresion")
    h = _hechos()
    rope_x = D.fila(clf, modelo=RIVAL)["p_rope"]
    mcs_ebm = D.fila(reg, modelo="ebm")["p_mcs"]
    nb = D.fila(clf, modelo="naive_bayes")
    separa = [D.NOMBRE_MODELO[m] for m in clf.loc[clf["p_mcnemar_holm"] < D.ALFA, "modelo"]]
    xgb = D.NOMBRE_MODELO[RIVAL]
    puntos = [
        ("Que se repita en otra fintech", "COFINFAD es sintético. El escalón que el modelo original aprovecha existe porque el "
         "generador del conjunto resta una cantidad fija por producto activo; en datos reales un efecto así sería menos limpio. Nada "
         "garantiza que el empate entre modelos o el orden de las variables se repitan con clientes reales."),
        ("Que el modelo original supere a los benchmarks", f"Con la regla de superioridad de la guía, su veredicto es: {_veredictos()}."
         + (" Los mejores benchmarks ya estaban a milésimas del límite teórico del conjunto: ningún modelo puede ofrecer una mejora "
            "grande en la métrica principal." if D.original_json("techo.json") is not None else "")),
        ("Que prediga fuga real", "«Riesgo alto» es el cuarto superior de un puntaje de propensión que trae el conjunto; el AUC dice qué "
         "tan bien el modelo ordena según ese puntaje, no cuántos clientes se van."),
        (f"Que la EBM sea mejor que {xgb}", f"En clasificación la probabilidad de equivalencia práctica con {xgb} es {F.prob(rope_x)}; en "
         f"regresión el Model Confidence Set al {F.pct(NIVEL_MCS, 0)} deja solo a {xgb} (la EBM sale con {F.p_valor(mcs_ebm)})."),
        ("Que las pruebas vean diferencias pequeñas", f"Con {_cardinal(h['k_ext'])} pliegues la t corregida tiene {h['k_ext'] - 1} grados "
         f"de libertad; McNemar, con el {F.pct(M.CAPACIDAD_INFORME, 0)} de capacidad, solo separa a {F.enumerar(separa)} "
         f"({F.p_valor(nb['p_mcnemar_holm'])} ajustado)."),
        ("Lo que no se probó", "El SVM y el SVR son solo lineales, la regresión logística cubre L1 y L2 como un solo modelo, la regresión "
         "no incluye un modelo bayesiano y los tiempos son de un solo equipo."),
        ("Que no quede una fuga residual", "Las variables derivadas no usan el objetivo y la que lo genera se excluyó, pero en un conjunto "
         "sintético puede quedar información del objetivo en otra variable."),
        ("Que el simulador muestre efectos", "Cambiar una variable en la vista Un cliente muestra lo que el modelo asigna a otro cliente "
         "hipotético, no lo que pasaría si se interviene sobre una persona: nada aquí es causal."),
        ("Lo decidido después de ver la prueba", "La t corregida, McNemar, la comparación bayesiana y los estudios de sensibilidad se "
         "añadieron después y el informe los declara exploratorios."),
    ]
    return ui.destinos([ui.tarjeta([ui.parrafo(texto)], titulo=titulo) for titulo, texto in puntos])


def _trampa():
    h = _hechos()
    techo = D.techo()
    return ui.tarjeta([
        ui.parrafo(f"churn_probability no es fuga observada: es un puntaje generado en buena parte con active_products, el número de "
                   f"productos activos (r = {ui.num(h['r_ap'], 3)}; explica el {F.pct(h['r_ap'] ** 2)} de la varianza del puntaje en el "
                   f"entrenamiento). Un modelo que la viera recuperaría la fórmula: R² de {ui.num(techo['r2_oraculo'], 4)}, frente al techo "
                   f"de {ui.num(techo['r2'], 4)} sin ella."),
        ui.parrafo(["Por eso se excluye por fuga de información (", f"{_cita('kaufman')}", ") y lo que el modelo aprende es a reproducir "
                    "el resto del puntaje. ", dcc.Link("Ver la trampa en el EDA", href="/eda"), "."]),
    ], titulo="La etiqueta es un puntaje, y la variable que lo genera está fuera",
        fuentes=[D.fuente("eda/asociacion.csv"), D.fuente("entrega3/comparacion_entrega2/techo.json")])


def _usos():
    usos = [
        ("Decidir sobre un cliente real concreto", "contactarlo, cobrarle o cerrarle un producto: el modelo aprendió las reglas de un "
         "conjunto sintético."),
        ("Crédito, precios, cupos o cualquier decisión que afecte derechos", "no se diseñó ni se validó para eso."),
        ("Tratar distinto a alguien por su edad, su género o su ciudad", "son entradas del modelo; con datos reales podrían reproducir "
         "una discriminación."),
        ("Leer el puntaje como la probabilidad de que el cliente se vaya", "es un puntaje del conjunto; para informar una probabilidad "
         "está la recalibrada, y aun así es la del cuarto superior del puntaje."),
        ("Sacar conclusiones causales del simulador", "muestra asociaciones del modelo, no efectos de una intervención."),
        ("Usarlo en otra fintech sin más", "habría que reentrenar y validar en el tiempo con fuga observada."),
    ]
    return ui.tarjeta([ui.parrafo([html.Strong(f"{t}: "), texto]) for t, texto in usos]
                      + [ui.parrafo(dcc.Link("Ver la ficha del modelo", href=app.ruta_vista("ficha")))],
                      titulo="Usos inadecuados")


# --- Los sesgos por grupo -------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _por_grupo(columna):
    """AUC, prevalencia y campaña del 25 % de la EBM dentro de cada grupo de los clientes de prueba (tuplas: sin estado mutable)."""
    C = M.clientes()
    filas = []
    for grupo, d in C.groupby(columna, observed=True, sort=False):
        y = d["riesgo_alto"].astype(int).to_numpy()
        contactado = d["contactar_ebm"].astype(bool).to_numpy()
        pos = int(y.sum())
        auc = M.auc(y, d["p_ebm"]) if 0 < pos < len(d) else float("nan")
        filas.append((str(grupo), len(d), pos / len(d), auc, float(contactado.mean()),
                      float((contactado & (y == 1)).sum() / pos) if pos else float("nan")))
    if columna == "income_bracket":
        filas.sort(key=lambda f: ORDEN_INGRESOS.index(f[0]) if f[0] in ORDEN_INGRESOS else len(ORDEN_INGRESOS))
    elif columna == "cohorte":
        filas.sort(key=lambda f: f[0])
    else:
        filas.sort(key=lambda f: -f[1])
    return tuple(filas)


def _total():
    C = M.clientes()
    y, c = C["riesgo_alto"].astype(int).to_numpy(), C["contactar_ebm"].astype(bool).to_numpy()
    return (len(C), y.mean(), M.auc(y, C["p_ebm"]), c.mean(), (c & (y == 1)).sum() / y.sum())


def _sesgo(segmento):
    segmento = segmento if segmento in SEGMENTOS else "cohorte"
    nombre, columna = SEGMENTOS[segmento]
    filas = _por_grupo(columna)
    aucs = [f[3] for f in filas if not math.isnan(f[3])]
    capturas = [f[5] for f in filas if not math.isnan(f[5])]
    capacidad = F.pct(M.CAPACIDAD_INFORME, 0)
    titulo = (f"Por {nombre.lower()}, el AUC va de {ui.num(min(aucs), 3)} a {ui.num(max(aucs), 3)} y la campaña del {capacidad} alcanza "
              f"del {F.pct(min(capturas), 0)} al {F.pct(max(capturas), 0)} de los clientes de riesgo alto de cada grupo")
    n, prev, auc, cont, cap = _total()
    tabla = ui.tabla(["Grupo", "Clientes", "Riesgo alto", "AUC de la EBM", "Contactados", "Captura"],
                     [[g.replace("–", UNION + "–" + UNION), F.entero(m), F.pct(p), ui.num(a, 3), F.pct(c), F.pct(k)]
                      for g, m, p, a, c, k in filas],
                     total=["Todos", F.entero(n), F.pct(prev), ui.num(auc, 3), F.pct(cont), F.pct(cap)], compacta=True,
                     alta=len(filas) > 8)
    if segmento == "cohorte":
        nota = ("Son las cifras de la tabla de errores del informe: la edad pesa más que ninguna otra variable y la campaña se concentra "
                "en la cohorte mayor, donde el riesgo alto es más frecuente.")
    else:
        nota = (f"Cálculo de esta app sobre los clientes de prueba: el informe no desglosa por {nombre.lower()}. En los grupos pequeños el "
                "AUC cambia mucho con pocos clientes: lea la columna Clientes antes de comparar.")
    return titulo, [tabla, ui.nota(nota)]


def _edad_en_el_original():
    """La misma advertencia para el modelo original, con su importancia SHAP: la tabla de la izquierda se calculó con la EBM."""
    info = V.shap_indice()["modelos"].get(f"clasificacion__{O}")
    if info is None:
        return []
    imp = pd.Series(info["importancia_media_abs"], dtype=float)
    primera = imp.idxmax()
    return [ui.parrafo(f"La tabla se calculó con la EBM, el benchmark que se lee sin explicador. En el modelo original, BDL, "
                       + ("la edad también es la variable de más peso" if primera == "age" else
                          f"la variable de más peso es {M.nombre(primera).lower()}")
                       + f" ({F.pct(imp['age'] / imp.sum())} de su importancia SHAP): la advertencia vale igual para él.")]


def _sesgos_lado():
    imp = M.importancia()
    cohorte = _por_grupo("cohorte")
    mayor = max(cohorte, key=lambda f: f[2])
    menor = min(cohorte, key=lambda f: f[2])
    contactos = sum(f[1] * f[4] for f in cohorte)
    cuanto = "la variable que más pesa" if imp.index[0] == "age" else "una de las variables que más pesan"
    return ui.tarjeta([
        ui.parrafo(f"La edad es {cuanto} en la EBM ({F.pct(imp['age'])} de su importancia). La cohorte de {mayor[0]} años tiene el {F.pct(mayor[2], 0)} de riesgo alto y recibe el "
                   f"{F.pct(mayor[1] * mayor[4] / contactos, 0)} de los contactos; la de {menor[0]}, con el {F.pct(menor[2], 0)}, queda casi "
                   f"fuera: solo se contacta al {F.pct(menor[4])} de sus clientes."),
        ui.parrafo(f"El género y la ciudad también son entradas del modelo, aunque pesan poco ({F.pct(imp['gender'])} y "
                   f"{F.pct(imp['location'])} de la importancia). En datos sintéticos esto describe las reglas del generador; con datos "
                   "reales habría que auditar si el trato distinto por grupo se justifica."),
        *_edad_en_el_original(),
    ], titulo="Qué grupos trata distinto", fuentes=[D.fuente("modelos/ebm_clasificacion.json"), D.fuente("clientes_prueba.parquet"),
                                                   D.fuente("resultados/errores_segmentos.csv")])


# --- 9. Conclusiones ----------------------------------------------------------------------------------------------------------
def _conclusiones_original():
    """Lo que sostienen los resultados del modelo original, con lo que dicen los archivos de la corrección: el veredicto por
    tarea con su causa, la estructura que recupera, la incertidumbre y el margen que dejaba el límite teórico. Cada tarjeta
    falta si su archivo todavía no ha llegado."""
    if all(D.veredicto_original(t) is None for t in TAREAS):
        return [OG._pendiente("Las conclusiones sobre el modelo original", "veredicto.json", "lo calcula 04_pruebas_hipotesis.py")]
    estados = {t: D.estado_veredicto(t) for t in TAREAS}
    hay = [t for t in TAREAS if estados[t] is not None]
    parrafos = []
    for t in hay:
        parrafos.append(ui.parrafo(" ".join(x for x in (OG._por_que(t), OG._en_prueba(t)) if x)))
        causa = OG._causa(t)
        if causa:
            parrafos.append(ui.parrafo(causa))
    veredicto = ui.tarjeta(parrafos, titulo=_mayuscula(_titulo_respuesta().split(": ", 1)[-1]),
                           fuentes=[D.fuente_original("veredicto.json"), D.fuente_original("condicion4.json"),
                                    D.fuente_original("techo.json"), D.fuente_original("experimentos.csv")])
    provisional = any(estados[t]["provisional"] for t in hay)
    valor = " y ".join(f"{F.entero(estados[t]['cumplidas'])} de {F.entero(estados[t]['evaluadas'])}" for t in hay)
    lectura = ui.tarjeta_insight(valor, "condiciones de la regla de superioridad que cumple el modelo original, de las evaluadas, en "
                                 + " y en ".join(D.NOMBRE_TAREA[t].lower() for t in hay),
                                 f"Veredicto{' provisional' if provisional else ''}: {_veredictos().split(' (')[0]}.", chip_="Veredicto")
    tarjetas = []
    m, te = D.modelo_original("regresion"), D.original_json("techo.json")
    if m is not None and m.get("K") and te is not None:
        reales, aprendidos = sorted(te["pesos"].values()), sorted(m["pesos"])
        iguales = len(reales) == len(aprendidos)
        pesos = (f"sus pesos difieren de la distribución real de active_products en menos de "
                 f"{ui.num(max(abs(a - b) for a, b in zip(aprendidos, reales)), 3)}" if iguales
                 else f"la variable excluida tiene {_cardinal(len(reales))} niveles")
        tarjetas.append(ui.tarjeta([ui.parrafo(
            f"El modelo no ve active_products, la variable excluida por fuga de información. Aun así, el criterio BIC elige "
            f"{_cardinal(int(m['K']))} clases latentes; {pesos}, y la separación media entre sus desplazamientos es "
            f"{ui.num(m['separacion_media'], 4)} (la del generador es {ui.num(te['delta'], 2)}). Los benchmarks tratan ese escalón como "
            "ruido.")], titulo="Recupera la estructura del puntaje sin ver la variable que la causa",
            fuentes=[D.fuente_original("modelo_regresion.json"), D.fuente_original("techo.json")]))
    c = V.conformal()["modelos"]
    propio = (c.get(O, {}).get(NIVEL_CONFORME) or {}).get("conjunto")
    r = D.veredicto_original("regresion")
    if propio is not None and r is not None and r["mejor_benchmark"] in c:
        mejor = r["mejor_benchmark"]
        ancho = c[mejor][NIVEL_CONFORME]["ancho"]
        tarjetas.append(ui.tarjeta([ui.parrafo(
            f"Con cobertura nominal del {F.pct(float(NIVEL_CONFORME), 0)}, el conjunto de predicción del modelo original mide "
            f"{ui.num(propio['largo'], 4)} (la unión de {F.entero(propio['tramos'])} tramos) y cubre el "
            f"{F.pct(propio['cobertura_prueba'])} de la prueba; el intervalo conforme {_tras_de(mejor)} mide {ui.num(ancho, 4)}, unas "
            f"{F.entero(ancho / propio['largo'])} veces más. En lugar de un intervalo ancho, el modelo dice en cuál de los niveles "
            "puede estar el puntaje.")], titulo="La incertidumbre es donde el modelo cambia el resultado",
            fuentes=[D.fuente("conformal.json")]))
    if te is not None:
        reg, clf = D.lideres("regresion").set_index("modelo"), D.lideres("clasificacion").set_index("modelo")
        rr, rc = D.veredicto_original("regresion"), D.veredicto_original("clasificacion")
        if rr is not None and rc is not None and rr["mejor_benchmark"] in reg.index and rc["mejor_benchmark"] in clf.index:
            falta_r = abs(reg.loc[rr["mejor_benchmark"], "rmse_prueba"] - te["prueba"]["rmse_minimo"])
            falta_c = abs(te["prueba"]["auc_maximo"] - clf.loc[rc["mejor_benchmark"], "auc_prueba"])
            tarjetas.append(ui.tarjeta([ui.parrafo(
                f"Con la estructura del escalón se puede calcular el límite del problema: en la prueba, un RMSE de "
                f"{ui.num(te['prueba']['rmse_minimo'], 5)} y un AUC de {ui.num(te['prueba']['auc_maximo'], 4)}. Los mejores benchmarks "
                f"ya estaban a {ui.num(falta_r, 5)} y a {ui.num(falta_c, 4)} de ese límite: los modelos flexibles empatan porque están "
                "cerca del techo, y lo que el modelo original puede añadir es estructura e incertidumbre mejor descrita, no precisión "
                "puntual.")], titulo="El margen de mejora era mínimo",
                fuentes=[D.fuente_original("techo.json"), D.fuente_original("lideres_regresion.csv"), F_CLF]))
    a = D.original_json("auditoria_fuga.json")
    if a is not None and a["fecha_de_corte"]["predictores_identicos"]:
        tarjetas.append(ui.tarjeta([ui.parrafo(
            f"Calculada solo con los clientes de entrenamiento, la fecha de corte de la recencia es la misma "
            f"({_fecha(a['fecha_de_corte']['correccion_solo_entrenamiento'])}) y cambian "
            f"{F.entero(a['fecha_de_corte']['celdas_de_recencia_que_cambian'])} celdas: los resultados de las entregas anteriores "
            "siguen siendo válidos.")], titulo="La corrección de la recencia no cambia ningún dato",
            fuentes=[D.fuente_original("auditoria_fuga.json")]))
    filas = [ui.rejilla(tarjetas[i:i + 2], "1-1") if len(tarjetas[i:i + 2]) > 1 else tarjetas[i] for i in range(0, len(tarjetas), 2)]
    return [ui.rejilla([veredicto, lectura], "2-1")] + filas


def _conclusiones():
    t, r = D.prueba("clasificacion"), D.prueba("regresion")
    pc, pr = D.pruebas("clasificacion"), D.pruebas("regresion")
    vc, vr = D.veredictos("clasificacion"), D.veredictos("regresion")
    e, x, lin = D.fila(t, modelo="ebm"), D.fila(t, modelo=RIVAL), D.fila(t, modelo=LINEAL)
    er, xr = D.fila(r, modelo="ebm"), D.fila(r, modelo=RIVAL)
    empates_c = [m for m in vc.index if vc[m] == "empata"]
    rope_c = [D.fila(pc, modelo=m)["p_rope"] for m in empates_c]
    mcnemar = all(D.fila(pc, modelo=m)["p_mcnemar_holm"] >= D.ALFA for m in empates_c)
    pruebas = "ninguna prueba encuentra diferencia" if mcnemar else "ni la t corregida ni DeLong encuentran diferencia"
    sup_c, sup_r = int((vc == "supera").sum()), int((vr == "supera").sum())
    supera = (f"{sup_c} de las {len(vc)} referencias de cada tarea" if sup_c == sup_r and len(vc) == len(vr)
              else f"{sup_c} de {len(vc)} en clasificación y {sup_r} de {len(vr)} en regresión")
    dentro = [m for m, esta in zip(r["modelo"], r["en_mcs"]) if esta]
    alcanza = ui.tarjeta([ui.parrafo(
        f"La EBM tiene el mayor AUC de prueba de los {_cardinal(len(t))} clasificadores del Entregable 3 ({ui.num(e['auc_prueba'], 4)}, frente a "
        f"{ui.num(x['auc_prueba'], 4)} de {D.NOMBRE_MODELO[RIVAL]}) y el segundo RMSE de regresión ({ui.num(er['rmse_prueba'], 5)}, a "
        f"{ui.num(er['rmse_prueba'] - xr['rmse_prueba'], 5)} de {D.NOMBRE_MODELO[RIVAL]}). Supera con significación tras la corrección de "
        f"Holm a {supera}: los lineales, k-NN y Naive Bayes. Frente a {_lista_frase(empates_c)} {pruebas} (equivalencia práctica de "
        f"{F.prob(min(rope_c))} a {F.prob(max(rope_c))}; {F.prob(D.fila(pr, modelo=RIVAL)['p_rope'])} con {D.NOMBRE_MODELO[RIVAL]} en "
        f"regresión). En regresión, el Model Confidence Set al {F.pct(NIVEL_MCS, 0)} deja dentro solo a "
        f"{F.enumerar(D.NOMBRE_MODELO[m] for m in dentro)}: detecta una ventaja de "
        f"{ui.num(er['rmse_prueba'] - xr['rmse_prueba'], 5)} de RMSE que las pruebas por pares no confirman.")],
        titulo="La EBM alcanza a los modelos de árboles, no los supera", fuentes=FUENTES_PRUEBAS)

    e2, techo = D.tabla("resultados/entrega2.csv"), D.techo()
    a_c, d_c = D.fila(e2, tarea="clasificacion", modelo="logistica_e2"), D.fila(e2, tarea="clasificacion", modelo="ebm")
    a_r, d_r = D.fila(e2, tarea="regresion", modelo="lasso_e2"), D.fila(e2, tarea="regresion", modelo="ebm")
    cubre = (d_c["metrica"] - AZAR_AUC) / (techo["auc_mediana"] - AZAR_AUC)
    datos = ui.tarjeta([ui.parrafo(
        f"En la partición y la etiqueta de la entrega anterior, la EBM sube el AUC de {ui.num(a_c['metrica'], 4)} a "
        f"{ui.num(d_c['metrica'], 4)} y baja el RMSE de {ui.num(a_r['metrica'], 5)} a {ui.num(d_r['metrica'], 5)} "
        f"({F.p_valor(max(d_c['p_holm'], d_r['p_holm']))}), y cubre el {F.pct(cubre, 0)} del techo de información que deja excluir "
        "active_products: el empate entre los modelos flexibles es una propiedad de los datos, no una debilidad del modelo.")],
        titulo="El empate es una propiedad de los datos", fuentes=[D.fuente("resultados/entrega2.csv"),
                                                                    D.fuente("entrega3/comparacion_entrega2/techo.json")])

    cal = D.tabla("entrega3/complementos/calibracion.csv")
    iso = cal[cal["version"] == "isotónica"]["ece"]
    sin = D.fila(cal, modelo="ebm", version="sin recalibrar")["ece"]
    p_lin = D.fila(pc, modelo=LINEAL)["p_delong_holm"]
    meta = M.meta()
    practica = ui.tarjeta([ui.parrafo(
        f"Con la capacidad del {F.pct(M.CAPACIDAD_INFORME, 0)}, la EBM alcanza a {F.entero(e['vp_op'])} de los "
        f"{F.entero(meta['positivos_prueba'])} clientes de riesgo alto y {D.en_frase(LINEAL)} a {F.entero(lin['vp_op'])}, aunque el AUC "
        f"las separe ({F.p_valor(p_lin)}). Con el {F.pct(CAPACIDAD_ALTA, 0)} la EBM captura el {F.pct(e['recall_50'], 0)} y la logística el "
        f"{F.pct(lin['recall_50'], 0)}: el modelo se elige con la capacidad de la campaña en mente y con probabilidades recalibradas (la "
        f"isotónica baja el ECE de {ui.num(sin, 3)} a entre {ui.num(iso.min(), 3)} y {ui.num(iso.max(), 3)}).")],
        titulo="La significancia no es relevancia práctica", fuentes=[F_CLF, D.fuente("resultados/pruebas_clasificacion.csv"),
                                                                       D.fuente("entrega3/complementos/calibracion.csv")])

    costo = D.tabla("resultados/costo.csv")
    veces = [D.fila(costo, tarea=k, modelo="ebm")["ajuste_s"] / D.fila(costo, tarea=k, modelo=RIVAL)["ajuste_s"]
             for k in ("clasificacion", "regresion")]
    ce = D.fila(costo, tarea="clasificacion", modelo="ebm")
    imp = M.importancia()
    legible = ui.tarjeta([ui.parrafo(
        f"La EBM es la más lenta de ajustar ({ui.num(ce['ajuste_s'], 1)} s, de {ui.num(min(veces), 0)} a {ui.num(max(veces), 0)} veces "
        f"{D.NOMBRE_MODELO[RIVAL]}), pero predice a los {F.entero(meta['n_prueba'])} clientes de prueba en {ui.num(ce['inferencia_s'], 3)} s "
        f"y ocupa {ui.num(ce['tamano_mb'], 1)} MB. Lo que compra el ajuste es la legibilidad: {_cardinal(VARIABLES_CON_SENAL)} variables "
        f"reúnen el {F.pct(imp.iloc[:VARIABLES_CON_SENAL].sum(), 0)} de la importancia y sus curvas muestran la no linealidad sin "
        "explicador auxiliar.")], titulo="La legibilidad cuesta en el ajuste, no en el uso",
        fuentes=[D.fuente("resultados/costo.csv"), D.fuente("modelos/ebm_clasificacion.json")])

    sem = D.fila(D.tabla("resultados/semillas.csv"), tarea="clasificacion", modelo="ebm")
    conc = D.fila(D.tabla("resultados/concordancia_validacion_prueba.csv"), tarea="clasificacion")
    seg = D.errores_segmentos()
    coh = seg[seg["segmento"] == "cohorte de edad"].sort_values("auc_ebm")
    prod = seg[seg["segmento"] == "productos activos"].sort_values("nivel")
    dificil = coh.iloc[0]
    robusta = ui.tarjeta([
        ui.parrafo(f"La EBM es casi determinista (desviación de {ui.num(sem['auc_desv'], 5)} de AUC entre {_cardinal(int(sem['semillas']))} "
                   f"semillas) y la validación anidada predijo la prueba casi sin sesgo (correlación de {ui.num(conc['correlacion'], 3)} en "
                   f"{F.entero(conc['corridas'])} corridas): las conclusiones no dependen de una configuración afortunada."),
        ui.parrafo(f"Donde falla: la cohorte de {dificil['nivel']} años, con el {F.pct(dificil['prevalencia'], 0)} de riesgo alto, es la "
                   f"más difícil (AUC {ui.num(dificil['auc_ebm'], 3)}), y el sesgo de la regresión sube en escalera con los productos "
                   f"activos ({ui.con_signo(prod.iloc[0]['sesgo_ebm'], 3)} con uno, {ui.con_signo(prod.iloc[-1]['sesgo_ebm'], 3)} con "
                   f"{_cardinal(int(prod.iloc[-1]['nivel']))}): es la parte del objetivo que depende de la variable excluida."),
    ], titulo="Es estable, y se sabe dónde falla", fuentes=[D.fuente("resultados/semillas.csv"),
                                                            D.fuente("resultados/concordancia_validacion_prueba.csv"),
                                                            D.fuente("resultados/errores_segmentos.csv")])
    return (_conclusiones_original()
            + [html.H3("La EBM, el benchmark interpretable: lo que sostiene el informe del Entregable 3"),
               alcanza, ui.rejilla([datos, practica, legible, robusta], "1-1"), limites._futuro()])


def _lista_frase(modelos):
    """Varios modelos dentro de una frase, con su artículo: «XGBoost, Random Forest y el árbol de decisión»."""
    return F.enumerar(D.en_frase(m) for m in modelos)


# --- 10. Autores, versión y documentos ---------------------------------------------------------------------------------------
def _orden_alfabetico(texto):
    """Clave para ordenar apellidos sin que cuenten las mayúsculas ni los signos: Dempster, Demšar, DeSarbo."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii").casefold()


def _autores():
    c, o = _citacion(), _origen()
    original = V.indice().get("original") or {}
    bdl = f"; los de BDL, el {_fecha(original['generado'])}" if original.get("generado") else ""
    equipo = ui.tarjeta([ui.ficha([
        ("Autores", _autores_prosa(c["autores"])),
        ("Curso", "Machine Learning · Pregrado en Ciencia de Datos"),
        ("Universidad", "Universidad del Norte, Barranquilla"),
        ("Versión del dashboard", f"{c['version']}, del {_fecha(c['fecha'])}"),
        ("Datos de la app", f"generados el {_fecha(o['datos'])}"),
        ("Modelos guardados (artifacts/)", f"generados el {_fecha(o['artefactos'])}{bdl}"),
        ("Licencia del código", html.A(c["licencia"] or "MIT", href=LICENCIA_MIT, target="_blank", rel="noopener")),
        ("Licencia de los datos", html.A(c["datos"]["licencia"], href=LICENCIA_CC_BY, target="_blank", rel="noopener")),
        ("Repositorio", html.A(c["repositorio"].replace("https://", ""), href=c["repositorio"], target="_blank", rel="noopener")),
    ]), ui.nota("Cómo citarlo: CITATION.cff, junto a la app; el texto de la licencia, en LICENSE.")],
        titulo="Autores y versión", fuentes=["CITATION.cff", D.fuente("origen.json"), "artifacts/indice.json"])
    d = c["datos"]
    refs = [ui.nota([f"{_autores_cita(d['autores'])} ({d['anio']}). {d['titulo']}. {d['editor']}, versión {d['version']}. ",
                     html.A(f"doi:{d['doi']}", href=f"https://doi.org/{d['doi']}", target="_blank", rel="noopener"), "."])]
    for r in sorted(REFERENCIAS.values(), key=lambda r: _orden_alfabetico(r["autores"])):
        texto = [f"{r['autores']} ({r['anio']}). {r['titulo']}. {r['fuente']}."]
        if r.get("url"):
            texto += [" ", html.A(r["url"].replace("https://", ""), href=r["url"], target="_blank", rel="noopener"), "."]
        if r.get("libro"):
            texto.append(f" Del libro del proyecto final: se cita {r['motivo']}.")
        elif not r["informe"]:
            texto.append(f" No está en el informe: se cita {r['motivo']}.")
        refs.append(ui.nota(texto))
    referencias = ui.tarjeta(refs, titulo="Referencias de esta página y del modelo original",
                             subtitulo="Las del informe del Entregable 3, las principales del modelo original (el libro del proyecto "
                                       "final trae la lista completa), las dos de Successive Halving y Hyperband y la de la cifra "
                                       "del costo de retener.")
    return [equipo, referencias, ui.destinos(limites._destinos())]


# --- 11. Cómo recorrer el dashboard en la sustentación -----------------------------------------------------------------------
def _recorrido():
    nombre = {ruta: n for ruta, _p, n, _q in app.DESTINOS}

    def vistas(*vs):
        enlaces = []
        for i, v in enumerate(vs):
            ruta = v if v.startswith("/") else app.ruta_vista(v)
            enlaces += ([" · "] if i else []) + [dcc.Link(nombre.get(ruta, ruta), href=ruta)]
        return html.Span(enlaces)

    def aqui(*anclas):
        partes = []
        for i, (texto, ancla) in enumerate(anclas):
            partes += ([" · "] if i else []) + [html.A(texto, href=f"#{PRE}-{ancla}")]
        return partes

    filas = [
        ["Problema y formulación matemática", html.Span(["Contexto › "] + aqui(("Por qué importa", "importa"), ("Pregunta", "pregunta"),
                                                                              ("Formulación", "formulacion"), ("Datos", "datos"),
                                                                              ("Diccionario", "diccionario")))],
        ["Análisis exploratorio", vistas("/eda")],
        ["Fundamentos teóricos del modelo original", html.Span(["Modelos › ", vistas("original")])],
        ["Implementación y diseño experimental", html.Span(["Contexto › "] + aqui(("Correcciones", "correcciones"),
                                                                                  ("Metodología", "metodologia"),
                                                                                  ("Dos fidelidades", "fidelidad"))
                                                           + [" · Modelos › ", vistas("lideres", "original")])],
        ["Resultados", html.Span(["Modelos › ", vistas("lideres", "comparacion", "prediccion", "diagnostico", "interpretabilidad")])],
        ["Conclusiones y limitaciones", html.Span(["Contexto › "] + aqui(("Conclusiones", "conclusiones"), ("Limitaciones", "limites"),
                                                                         ("Sesgos", "sesgos"))
                                                  + [" · Modelos › ", vistas("ficha")])],
        ["La EBM, benchmark interpretable (Entregable 3)", html.Span(["Modelos › ", vistas("propuesto", "decide", "errores")])],
    ]
    return ui.tarjeta([ui.tabla(["Criterio de la rúbrica", "Dónde se muestra"], filas, num=[False, False]),
                       ui.nota("El pie de cada vista lleva a la siguiente del recorrido; la barra de arriba vuelve a las tres secciones.")],
                      titulo="Un criterio, una parada")


# --- La página ---------------------------------------------------------------------------------------------------------------
def layout(**_):
    h = _hechos()
    c = _citacion()
    tarea = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", [{"label": D.NOMBRE_TAREA[t], "value": t} for t in inicio.TAREA],
                                               "clasificacion"))
    grupo = ui.control("Mostrar:", ui.segmentado(f"{PRE}-dicc-grupo", [{"label": v, "value": k} for k, v in GRUPOS_DICC.items()], "todas"))
    segmento = ui.control("Grupo:", ui.segmentado(f"{PRE}-segmento", [{"label": v[0], "value": k} for k, v in SEGMENTOS.items()], "cohorte"))
    correccion = D.estado_original().get("generado")          # cuándo se trajeron por última vez los resultados de la corrección
    respuesta = html.Div([
        html.Div([ui.chip("Datos sintéticos", "atencion"), ui.chip_valor("Versión", c["version"] or "—"),
                  ui.chip_valor("Datos del", _fecha(_origen()["datos"]))]
                 + ([ui.chip_valor("BDL, resultados del", _fecha(correccion))] if correccion else []), className="chips"),
        html.Div(ui.boton("Cómo recorrer el dashboard", href=f"#{PRE}-recorrido", icono="flecha-abajo"), className="botonera"),
    ])
    fid = _fidelidades()
    sesgo_titulo, sesgo_cuerpo = _sesgo("cohorte")
    return ui.pantalla([
        ui.encabezado(app.SECCIONES[0][3], _subtitulo(), respuesta=[respuesta], pantalla="inicio"),
        ui.seccion(_titulo_respuesta(), [html.Div(id=f"{PRE}-heroe")],
                   subtitulo="La cifra principal va con lo que la acota: la ventaja del modelo original sobre el mejor benchmark en la "
                             "validación anidada y lo que faltaba para el límite teórico. Cambie la tarea para ver la regresión.",
                   controles=[tarea],
                   fuentes=FUENTES_PRUEBAS[:2] + [D.fuente_original("lideres_<tarea>.csv"), D.fuente_original("veredicto.json"),
                                                  D.fuente_original("condicion4.json"), D.fuente_original("techo.json")],
                   id=f"{PRE}-respuesta"),
        ui.seccion("Por qué importa: retener cuesta menos que conseguir, la campaña no alcanza para todos y la decisión debe poder "
                   "explicarse", _importa(), id=f"{PRE}-importa"),
        ui.seccion("La pregunta: ¿supera a los benchmarks un modelo que trata el escalón del puntaje como una clase latente?",
                   [_pregunta()], id=f"{PRE}-pregunta"),
        ui.seccion("Las correcciones: el modelo original es propio y la recencia se calcula solo con el entrenamiento", _correcciones(),
                   subtitulo="Las dos observaciones a la primera versión del proyecto final y lo que se hizo con cada una.",
                   id=f"{PRE}-correcciones"),
        ui.seccion(f"La formulación: una clasificación y una regresión sobre los mismos {F.entero(h['n_pred'])} predictores",
                   [_formulacion()], subtitulo="Aprendizaje supervisado. Cada cliente es un vector de predictores y un puntaje de "
                                               "propensión; las dos tareas comparten partición, preprocesamiento y validación.",
                   id=f"{PRE}-formulacion"),
        ui.seccion(f"Los datos: COFINFAD, {F.entero(h['etl']['n_clientes'])} clientes sintéticos de una fintech colombiana", _datos(),
                   subtitulo=f"Dos tablas de {(h['fecha_detalle'] or '')[:4]}: una fila por cliente y el detalle de las transacciones. "
                             "El análisis exploratorio usa solo el entrenamiento.", id=f"{PRE}-datos"),
        ui.seccion("El diccionario de variables: qué entra al modelo y qué se eliminó, con su motivo", [_diccionario()],
                   controles=[grupo], id=f"{PRE}-diccionario"),
        ui.seccion(f"La metodología: de los datos al dashboard, con {F.entero(h['corridas'])} corridas de validación anidada", _pipeline(),
                   subtitulo="Cada paso abre la sección o la vista del dash que lo muestra.", id=f"{PRE}-metodologia"),
        ui.seccion("La búsqueda de hiperparámetros usa dos fidelidades; no es Successive Halving", _fidelidad(),
                   subtitulo=(f"La EBM se busca con {PARAMETRO[fid['ebm']['clave']][0]} = {F.entero(fid['ebm']['baja'])} y se reajusta "
                              f"con {F.entero(fid['ebm']['alta'])}; Random Forest, con {F.entero(fid['random_forest']['baja'])} y "
                              f"{F.entero(fid['random_forest']['alta'])} árboles. Successive Halving es otra cosa."
                              if "ebm" in fid and "random_forest" in fid else None), id=f"{PRE}-fidelidad"),
        ui.seccion("Limitaciones: el carácter sintético de los datos limita la utilidad práctica de las conclusiones", [
            ui.aviso("Como señaló la evaluación del Entregable 3, las conclusiones describen las reglas con que se generó COFINFAD, no el "
                     "comportamiento de los clientes de una fintech real: que los modelos empaten aquí, o que el modelo original recupere "
                     "el escalón del puntaje, no garantiza que ocurra con otros datos, y ningún resultado de esta app sirve para decidir "
                     "sobre personas reales sin validarlo antes con fuga observada.", "atencion",
                     "Datos sintéticos, utilidad práctica limitada"),
            ui.rejilla([_trampa(), _usos()], "1-1"),
            html.H3("Lo que este trabajo no permite afirmar"),
            _no_permite(),
        ], subtitulo="Lo que declara el informe, con su consecuencia, y los usos que no corresponden.", id=f"{PRE}-limites",
            fuentes=[D.fuente("resultados/pruebas_clasificacion.csv"), D.fuente("resultados/regresion_prueba.csv"), F_CORRIDAS,
                     D.fuente_original("veredicto.json"), D.fuente_original("condicion4.json")]),
        ui.seccion("Sesgos posibles: la EBM acierta y contacta distinto según la cohorte de edad", [
            ui.rejilla([
                ui.tarjeta([html.Div(sesgo_cuerpo, id=f"{PRE}-sesgo-cuerpo")], titulo=html.Span(sesgo_titulo, id=f"{PRE}-sesgo-titulo"),
                           subtitulo=f"En los {F.entero(M.meta()['n_prueba'])} clientes de prueba: AUC dentro del grupo y qué parte de "
                                     f"sus clientes de riesgo alto alcanza la campaña del {F.pct(M.CAPACIDAD_INFORME, 0)}.",
                           fuentes=[D.fuente("clientes_prueba.parquet"), D.fuente("resultados/errores_segmentos.csv")]),
                _sesgos_lado(),
            ], "2-1"),
        ], controles=[segmento], id=f"{PRE}-sesgos"),
        ui.seccion("Conclusiones: lo que sostienen los resultados", _conclusiones(),
                   subtitulo="Primero, el modelo original, con lo que dicen los archivos de la corrección; después, la EBM, con las "
                             "conclusiones del informe del Entregable 3. Cada cifra sale de los resultados guardados.",
                   id=f"{PRE}-conclusiones"),
        ui.seccion("Quién lo hizo y dónde está el resto", _autores(),
                   subtitulo="El informe y sus dos secciones, el repositorio, el libro y los datos se abren en otra pestaña.",
                   id=f"{PRE}-autores"),
        ui.seccion("Cómo recorrer el dashboard en la sustentación", [_recorrido()],
                   subtitulo="El recorrido sigue los criterios de la rúbrica, de arriba abajo.", id=f"{PRE}-recorrido"),
        ui.pie([ui.nota(f"Datos: «{c['datos']['titulo']}» ({_cita_datos()}; {c['datos']['editor']}, licencia {c['datos']['licencia']}). "
                        "«Riesgo alto» es el cuarto superior del puntaje de propensión del conjunto de datos, no fuga observada."),
                ui.nota(D.nota_ece()),
                ui.nota(f"Proyecto final de Machine Learning · Pregrado en Ciencia de Datos, Universidad del Norte · "
                        f"{_autores_prosa(c['autores'])}.")]),
    ])


# --- Callbacks ---------------------------------------------------------------------------------------------------------------
@callback(Output(f"{PRE}-heroe", "children"), Input(f"{PRE}-tarea", "value"))
def _heroe(tarea):
    return _heroe_original(tarea if tarea in inicio.TAREA else "clasificacion")


@callback(Output(f"{PRE}-dicc", "data"), Output(f"{PRE}-dicc-cuenta", "children"), Input(f"{PRE}-dicc-grupo", "value"))
def _dicc(grupo):
    grupo = grupo if grupo in GRUPOS_DICC else "todas"
    filas, total = _filas(grupo), len(_filas_diccionario())
    cuenta = (f"Se muestran las {total} columnas." if len(filas) == total
              else f"Se muestran {len(filas)} de las {total} columnas ({GRUPOS_DICC[grupo].lower()}).")
    return filas, cuenta


@callback(Output(f"{PRE}-sesgo-titulo", "children"), Output(f"{PRE}-sesgo-cuerpo", "children"), Input(f"{PRE}-segmento", "value"))
def _sesgos(segmento):
    return _sesgo(segmento)
