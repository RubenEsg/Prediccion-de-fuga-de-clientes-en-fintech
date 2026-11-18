"""Lectura de la carpeta datos/ de la app: una función con caché por tipo de archivo y los nombres visibles del informe.

    import datos_app as D
    D.tabla("resultados/clasificacion_prueba.csv")     -> copia de la tabla (se puede modificar sin dañar la caché)
    D.json("entrega3/comparacion_entrega2/techo.json") -> dict
    D.clave_valor("resultados/fidelidad_resumen.csv")  -> dict de los csv que son pares clave–valor
    D.fila(D.tabla(...), modelo="ebm")                 -> la fila (Series) que cumple las condiciones
    D.fuente("resultados/costo.csv")                   -> "datos/resultados/costo.csv", para fuentes=[…] de las tarjetas
    D.NOMBRE_MODELO["bdl"]                             -> "BDL (original)"    (los nombres de la corrección y del informe)

Dos familias de tablas, que no se mezclan sin decirlo:
  · datos/resultados/ y datos/entrega3/: la sección de resultados del informe del Entregable 3 (ocho modelos por tarea, con
    la EBM comparada con las otras siete). D.prueba(tarea) y D.pruebas(tarea) leen eso, tal cual.
  · datos/original/: la corrección (el modelo original, BDL, y LightGBM), que arma preparar_original.py con lo que haya
    en ModeloOriginal/resultados. Cada lector devuelve None si su archivo todavía no existe, para que la vista muestre el
    bloque como pendiente: D.original_tabla, D.original_json, D.estado_original, D.veredicto_original, D.condicion4,
    D.pruebas_original, D.pliegues, D.modelo_original, D.corrida_original, D.costo_original. D.lideres(tarea) junta las
    dos familias: los ocho del informe y las filas de BDL y de LightGBM, con las mismas columnas. El veredicto de
    superioridad del modelo original lo dan D.estado_veredicto(tarea) y D.frase_veredicto(tarea), que aplican la regla
    del protocolo a veredicto.json y a condicion4.json: ninguna vista lo escribe a mano.

Los archivos y sus columnas están descritos en datos/origen.json (de dónde salió cada uno) y en CONTRATO.md. El modelo en
vivo y los clientes de prueba se leen con motor.py, no aquí.
"""
from __future__ import annotations

import json as _json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

import formato as _F

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"

NECESARIOS = ["metadatos.json", "clientes_prueba.parquet", "puntajes_prueba.parquet", "pronosticos_prueba.parquet",
              "calibracion_isotonica_ebm.json", "variables.csv", "modelos/ebm_clasificacion.json", "modelos/ebm_regresion.json",
              "modelos/logistica_clasificacion.json", "resultados/clasificacion_prueba.csv", "resultados/regresion_prueba.csv",
              "eda/por_variable.csv", "entrega3/corridas.csv"]

# --- Nombres visibles: los de la corrección (ModeloOriginal/src/plan.py, NOMBRES) y los del informe (Resultados Entregable
# 3/generar.py, diccionarios NOMBRES, BALANCEO y OPTIM). La EBM fue el «modelo nuevo» del Entregable 3; viene de la librería
# interpret, así que ahora es un benchmark reciente y se llama «EBM» a secas. El modelo original del proyecto es BDL.
ORIGINAL = "bdl"
"""Clave del modelo original (Boosting con Desplazamiento Latente, src/original_model.py)."""
NUEVOS = ("bdl", "lightgbm")
"""Los dos modelos que añade la corrección al Entregable 3: el original y el segundo benchmark reciente."""
SOLO_TABLAS = ("lightgbm",)
"""Modelos que aparecen en las tablas pero no se sirven en vivo (no hay artefacto: no se añaden dependencias al servidor)."""

NOMBRE_MODELO = {"bdl": "BDL (original)", "ebm": "EBM", "lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "Random Forest",
                 "arbol": "Árbol de decisión", "knn": "k-NN", "logistica": "Regresión logística", "svm": "SVM lineal",
                 "naive_bayes": "Naive Bayes", "ridge": "Ridge", "lasso": "Lasso", "svr": "SVR lineal",
                 "logistica_e2": "Logística L1 (E2)", "lasso_e2": "Lasso (E2)"}
NOMBRE_CORTO = dict(NOMBRE_MODELO, bdl="BDL", arbol="Árbol", logistica="Logística")
NOMBRE_LARGO_ORIGINAL = "Boosting con Desplazamiento Latente"
NOMBRE_BALANCEO = {"ninguno": "sin balanceo", "class_weight": "ponderación", "smote": "SMOTE", "adasyn": "ADASYN"}
NOMBRE_OPTIMIZADOR = {"grid": "Grid", "random": "Random", "bayesiana": "TPE", "genetica": "Genética"}
NOMBRE_TAREA = {"clasificacion": "Clasificación", "regresion": "Regresión"}

# Orden de las listas y de las tablas: el modelo original primero; después los ocho del informe en el orden de sus tablas
# (clasificación por AUC de prueba, de mayor a menor, y regresión por RMSE de prueba), con LightGBM al final de los modelos
# de boosting y de árboles, antes de los lineales. D.prueba(tarea) solo trae los ocho del informe, así que conserva su orden.
ORDEN_MODELOS = {"clasificacion": ["bdl", "ebm", "xgboost", "random_forest", "arbol", "lightgbm", "knn", "logistica", "svm", "naive_bayes"],
                 "regresion": ["bdl", "xgboost", "ebm", "random_forest", "arbol", "lightgbm", "lasso", "ridge", "svr", "knn"]}
ORDEN_OPTIMIZADORES = ["grid", "random", "bayesiana", "genetica"]
ORDEN_BALANCEOS = ["ninguno", "class_weight", "smote", "adasyn"]

# Rol de color de cada modelo (DISENO.md): el azul es el modelo original (BDL); los benchmarks con que compite en la parte
# alta de la tabla (EBM, LightGBM, XGBoost, Random Forest y el árbol de decisión) son los rivales, en rosa; los lineales,
# k-NN, Naive Bayes y la Entrega 2 son la base, en gris.
ROL_MODELO = {"bdl": "modelo", "ebm": "rival", "lightgbm": "rival", "xgboost": "rival", "random_forest": "rival", "arbol": "rival",
              "knn": "base", "logistica": "base", "svm": "base", "naive_bayes": "base", "ridge": "base", "lasso": "base", "svr": "base",
              "logistica_e2": "base", "lasso_e2": "base"}
# En las seis vistas del grupo «La EBM, benchmark interpretable» (y en la mitad del Entregable 3 de la comparación) la serie
# en foco es la EBM y va en azul; el modelo original, cuando aparece en ellas, es un rival más. Las demás, igual.
ROL_VISTA_EBM = dict(ROL_MODELO, ebm="modelo", bdl="rival")


def faltan_datos() -> list[str]:
    """Archivos imprescindibles que no están en datos/ (lista vacía si está todo)."""
    return [n for n in NECESARIOS if not (DATOS / n).is_file()]


def fuente(rel: str) -> str:
    """Ruta que se muestra en el pie de fuentes de una tarjeta."""
    return f"datos/{rel}"


@lru_cache(maxsize=None)
def _tabla(rel: str) -> pd.DataFrame:
    ruta = DATOS / rel
    if ruta.suffix == ".parquet":
        return pd.read_parquet(ruta)
    return pd.read_csv(ruta)


def tabla(rel: str) -> pd.DataFrame:
    """Un csv o parquet de datos/ (ruta relativa). Devuelve una copia: la caché no se toca."""
    return _tabla(rel).copy()


def compartida(rel: str) -> pd.DataFrame:
    """Lo mismo sin copiar, para las tablas grandes (puntajes y pronósticos por cliente). NO modificar el resultado."""
    return _tabla(rel)


@lru_cache(maxsize=None)
def json(rel: str):
    return _json.loads((DATOS / rel).read_text(encoding="utf-8"))


def clave_valor(rel: str) -> dict:
    """Los csv de pares clave–valor (primera columna sin nombre y una columna «0»): {clave: valor}, con los números como
    float cuando se puede."""
    t = _tabla(rel)
    out = {}
    for k, v in zip(t.iloc[:, 0], t.iloc[:, 1]):
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            out[str(k)] = v
    return out


def fila(t: pd.DataFrame, **condiciones) -> pd.Series:
    """La única fila de t que cumple columna=valor para todas las condiciones (error si no hay exactamente una)."""
    m = pd.Series(True, index=t.index)
    for col, val in condiciones.items():
        m &= t[col] == val
    if int(m.sum()) != 1:
        raise KeyError(f"se esperaba una fila con {condiciones} y hay {int(m.sum())}")
    return t[m].iloc[0]


def texto_llano(s) -> str:
    """Texto de los datos sin marcas de imprenta: «$K=0$» → «K = 0», «29--30» → «29–30»."""
    s = str(s)
    s = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\s*=\s*", " = ", m.group(1)), s)
    return s.replace("--", "–")


def nombre_modelo(clave: str, corto: bool = False) -> str:
    return (NOMBRE_CORTO if corto else NOMBRE_MODELO).get(clave, clave)


ARTICULO = {"ebm": "la", "logistica": "la", "arbol": "el", "svm": "el", "svr": "el"}
NOMBRE_COMUN = ("logistica", "arbol")


def en_frase(clave: str) -> str:
    """El modelo dentro de una frase: «BDL», «la EBM», «la regresión logística», «el árbol de decisión», «el SVM lineal»,
    «XGBoost», «LightGBM», «Random Forest», «k-NN», «Naive Bayes». Para rótulos, tablas y leyendas se usa NOMBRE_MODELO."""
    n = NOMBRE_CORTO[clave] if clave in ("ebm", ORIGINAL) else NOMBRE_MODELO.get(clave, clave)
    if clave in NOMBRE_COMUN:
        n = n[0].lower() + n[1:]
    return f"{ARTICULO[clave]} {n}" if clave in ARTICULO else n


def rol_modelo(clave: str) -> str:
    return ROL_MODELO.get(clave, "contexto")


def ordenar_modelos(t: pd.DataFrame, tarea: str, columna: str = "modelo") -> pd.DataFrame:
    """Las filas de t en el orden de las tablas del informe para esa tarea (los modelos que no estén en la lista, al final)."""
    orden = {m: i for i, m in enumerate(ORDEN_MODELOS[tarea])}
    return (t.assign(_o=t[columna].map(orden).fillna(len(orden))).sort_values("_o", kind="stable").drop(columns="_o")
            .reset_index(drop=True))


# --- Atajos de las tablas más usadas ---------------------------------------------------------------------------------------
def prueba(tarea: str) -> pd.DataFrame:
    """Desempeño de la mejor combinación de cada uno de los ocho modelos del Entregable 3: resultados/clasificacion_prueba.csv
    o regresion_prueba.csv, en el orden del informe. Para los diez modelos (con BDL y LightGBM), lideres(tarea)."""
    return ordenar_modelos(tabla(f"resultados/{tarea}_prueba.csv"), tarea)


def pruebas(tarea: str) -> pd.DataFrame:
    """Pruebas de significancia de la EBM frente a cada una de las otras siete del Entregable 3 (resultados/pruebas_<tarea>.csv).
    Un delta positivo favorece a la EBM (más AUC o menos RMSE). Las del modelo original están en pruebas_original(tarea)."""
    return ordenar_modelos(tabla(f"resultados/pruebas_{tarea}.csv"), tarea)


def puntajes() -> pd.DataFrame:
    """Puntaje de los clasificadores finales para cada cliente de prueba (índice «fila»; columna «y» = riesgo alto): los ocho
    del Entregable 3 y, si preparar_original.py ya corrió, bdl y lightgbm. Compartida: no modificar."""
    return compartida("puntajes_prueba.parquet")


def pronosticos() -> pd.DataFrame:
    """Pronóstico de los regresores finales para cada cliente de prueba (índice «fila»; columna «y» = churn_probability):
    los ocho del Entregable 3 y, si ya están, bdl y lightgbm. Compartida: no modificar."""
    return compartida("pronosticos_prueba.parquet")


# --- La corrección: el modelo original (BDL) y LightGBM (datos/original/, que arma preparar_original.py) -------------------
def fuente_original(rel: str) -> str:
    """Ruta que se muestra en el pie de fuentes de una tarjeta para un archivo de datos/original/."""
    return f"datos/original/{rel}"


def hay_original(rel: str) -> bool:
    """Si ese archivo de datos/original/ ya existe (un resultado de la corrección puede no haber llegado todavía)."""
    return (DATOS / "original" / rel).is_file()


def original_tabla(rel: str):
    """Un csv de datos/original/ (copia; la caché no se toca) o None si todavía no existe."""
    return tabla(f"original/{rel}") if hay_original(rel) else None


def original_json(rel: str):
    """Un json de datos/original/ o None si todavía no existe. NO modificar el resultado."""
    return json(f"original/{rel}") if hay_original(rel) else None


def estado_original() -> dict:
    """datos/original/estado.json: qué insumo de la corrección había y cuál faltaba la última vez que corrió
    preparar_original.py ({} si nunca corrió). Claves: generado, en_vivo, repeticiones, bloques y pendientes."""
    return original_json("estado.json") or {}


def veredicto_original(tarea: str):
    """La entrada de la tarea en datos/original/veredicto.json (04_pruebas_hipotesis.py): medias y desviaciones por modelo,
    mejor benchmark, ventaja, Friedman, las cuatro pruebas frente al mejor benchmark, rechazos y las condiciones 1 a 3 de
    la regla de superioridad de la guía. None si todavía no existe."""
    v = original_json("veredicto.json")
    return None if v is None else v.get(tarea)


def condicion4(tarea: str):
    """La condición 4 de la regla (conjuntos de datos adicionales) para la tarea, de datos/original/condicion4.json:
    conjuntos (los que ya tienen resultado), esperados, completo, mejor_en, peor_significativo_en, cumple, detalle y
    resumen. None si todavía no existe. Mientras «completo» sea falso, «cumple» es provisional (estado_condicion4)."""
    v = original_json("condicion4.json")
    return None if v is None else v.get(tarea)


REPETICIONES_PROTOCOLO = 3
"""Repeticiones de la validación anidada con que el protocolo pre-registrado fija el veredicto (semillas 42, 43 y 44)."""
VEREDICTO_ORIGINAL = {"superior": "superior", "no superior": "no superior", "pendiente": "pendiente de la cuarta condición"}
"""Cómo se escribe cada veredicto del modelo original dentro de una frase."""


def estado_condicion4(tarea: str) -> str:
    """La cuarta condición de la regla en una tarea, según condicion4.json: «pendiente» (todavía no hay ningún conjunto
    adicional), «provisional» (hay parte de los cuatro, o falta algún modelo en alguno), «cumple» o «no cumple»."""
    c4 = condicion4(tarea) or {}
    if not c4:
        return "pendiente"
    if not c4.get("completo"):
        return "provisional"
    return "cumple" if c4.get("cumple") else "no cumple"


def estado_veredicto(tarea: str):
    """El veredicto de superioridad del modelo original en una tarea, con la regla de la guía (§6.4) como quedó en el
    protocolo pre-registrado (sección 7) y como la aplica el libro del proyecto; None si veredicto.json todavía no existe.
    Nada se decide aquí a mano: todo sale de veredicto.json y de condicion4.json. Devuelve
      palabra       «superior», «no superior» o «pendiente» (cumple las tres primeras y la cuarta no está completa)
      cond          las condiciones 1 a 3, en orden (True o False)
      c4            «cumple», «no cumple», «provisional» (solo hay parte de los conjuntos adicionales) o «pendiente»
      cumplidas, evaluadas   cuántas se cumplen de las que ya se pueden evaluar (la cuarta cuenta solo si está completa)
      repeticiones  cuántas repeticiones de la validación anidada entraron en las pruebas
      provisional   si son menos de las tres que fija el protocolo
    Con una de las tres primeras en falso el veredicto ya es «no superior», llegue o no la cuarta."""
    r = veredicto_original(tarea)
    if r is None:
        return None
    cond = [bool(r.get("condicion_1_media_mejor")), bool(r.get("condicion_2_tres_pruebas")), bool(r.get("condicion_3_efecto"))]
    cuatro = estado_condicion4(tarea)
    if all(cond) and cuatro == "cumple":
        palabra = "superior"
    elif all(cond) and cuatro in ("pendiente", "provisional"):
        palabra = "pendiente"
    else:
        palabra = "no superior"
    evaluadas = cond + ([cuatro == "cumple"] if cuatro in ("cumple", "no cumple") else [])
    hechas = len(r.get("repeticiones") or [])
    return {"palabra": palabra, "cond": cond, "c4": cuatro, "cumplidas": int(sum(evaluadas)), "evaluadas": len(evaluadas),
            "repeticiones": hechas, "provisional": hechas < REPETICIONES_PROTOCOLO}


def frase_veredicto(tarea: str, detalle: bool = True) -> str:
    """El veredicto del modelo original en una tarea, para decirlo igual en todas las vistas: «no superior (cumple 2 de las 3
    condiciones evaluadas; resultado provisional, con 1 de 3 repeticiones)», «superior (…)», «pendiente de la cuarta
    condición (…)» o, si veredicto.json no existe, «pendiente de las pruebas de hipótesis». Con detalle=False, solo la
    palabra."""
    e = estado_veredicto(tarea)
    if e is None:
        return "pendiente de las pruebas de hipótesis"
    texto = VEREDICTO_ORIGINAL[e["palabra"]]
    if not detalle:
        return texto
    partes = [f"no cumple ninguna de las {_F.entero(e['evaluadas'])} condiciones evaluadas" if e["cumplidas"] == 0
              else f"cumple {_F.entero(e['cumplidas'])} de las {_F.entero(e['evaluadas'])} condiciones evaluadas"]
    if e["provisional"]:
        partes.append(f"resultado provisional, con {_F.entero(e['repeticiones'])} de {_F.entero(REPETICIONES_PROTOCOLO)} repeticiones")
    return f"{texto} ({'; '.join(partes)})"


# --- Los empates en el corte de la campaña ----------------------------------------------------------------------------------
# Un modelo con puntaje escalonado da el mismo puntaje a muchos clientes. Si el corte de la campaña cae en uno de esos grupos,
# a quién se contacta dentro de él lo decide el desempate, no el modelo: la captura, la precisión del punto de operación y la
# prueba P4 dependen de esa regla, y el AUC-PR, de cómo se cuenten los empatados. La corrección lo mide en
# ModeloOriginal/experiments/04d_empates_campana.py; aquí se lee su copia y se dice con las mismas frases en todas las vistas.
DESVIACIONES_EMPATE = 3
"""El AUC-PR de un modelo se anota si romper sus empates al azar lo mueve más que estas desviaciones (criterio del libro)."""
_CAMPOS_EMPATE = ("valores_distintos", "empatados", "cupos", "captura_esperada", "captura_desviacion", "captura_minima",
                  "captura_maxima")
_CAMPOS_PRECISION_MEDIA = ("precision_media", "precision_media_azar", "precision_media_azar_desviacion")


def _es_numero(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _de(frase: str) -> str:
    """«de» delante de un modelo en frase, con su contracción: «de BDL», «del árbol de decisión»."""
    return "del" + frase[2:] if frase.startswith("el ") else f"de {frase}"


def empates_campana():
    """El análisis de la corrección sobre los empates en el corte de la campaña del informe (datos/original/
    empates_campana.json, copia exacta): «n», «campana» y «positivos» de la prueba; por modelo («modelos») y para el predictor
    óptimo («optimo»), cuántos valores distintos toma el puntaje, cuántos clientes comparten el del corte («empatados»),
    cuántos de ellos entran («cupos»), la captura con el desempate del protocolo, la esperada y su desviación con un desempate
    al azar, la mínima y la máxima posibles, y la precisión media con la convención de scikit-learn y con los empates rotos al
    azar; y «p4», la prueba P4 repetida con desempates al azar. None si todavía no existe. NO modificar el resultado."""
    e = original_json("empates_campana.json")
    return e if isinstance(e, dict) and isinstance(e.get("modelos"), dict) else None


def empate_campana(modelo: str):
    """Lo que le pasa a un modelo en el corte de esa campaña cuando el corte cae en un empate (más de un cliente con el
    puntaje del último contactado): su bloque de empates_campana.json más «n», «campana» y «positivos». None si el archivo no
    está, si no trae al modelo o si en su corte no hay empate."""
    e = empates_campana()
    m = None if e is None else e["modelos"].get(modelo)
    if not isinstance(m, dict) or not all(_es_numero(m.get(k)) for k in _CAMPOS_EMPATE) or m["empatados"] <= 1:
        return None
    return {**m, "n": e.get("n"), "campana": e.get("campana"), "positivos": e.get("positivos")}


def modelos_con_empate() -> list:
    """Los modelos cuyo corte de la campaña cae en un empate, en el orden del catálogo."""
    e = empates_campana()
    if e is None:
        return []
    orden = ORDEN_MODELOS["clasificacion"]
    return sorted((m for m in e["modelos"] if empate_campana(m) is not None), key=lambda m: orden.index(m) if m in orden else len(orden))


def frase_empate_captura(modelo: str, precision: bool = False):
    """Una nota, calculada de empates_campana.json, para poner junto a la captura de un modelo en la campaña del informe:
    cuántos clientes empatan en el corte y cuántos entran, que la cifra mostrada es uno de los desempates posibles, y el
    valor esperado, su desviación y su rango con un desempate al azar. Con precision=True añade lo mismo para la precisión
    del punto de operación (la captura por los positivos entre los contactados). None si en el corte no hay empate."""
    m = empate_campana(modelo)
    if m is None:
        return None
    texto = (f"El puntaje {_de(en_frase(modelo))} toma {_F.entero(m['valores_distintos'])} valores distintos en la prueba: "
             f"{_F.entero(m['empatados'])} clientes comparten el del corte de la campaña y solo entran {_F.entero(m['cupos'])}. Cuáles "
             "entran lo decide el desempate, así que la captura que se muestra es una de las posibles: con un desempate al azar se "
             f"espera {_F.pct(m['captura_esperada'])} ± {_F.num(100 * m['captura_desviacion'], 1)} puntos, y puede ir de "
             f"{_F.pct(m['captura_minima'])} a {_F.pct(m['captura_maxima'])}")
    if precision and _es_numero(m.get("positivos")) and _es_numero(m.get("campana")) and m["campana"]:
        k = m["positivos"] / m["campana"]
        texto += (f". La precisión depende del mismo desempate: se espera {_F.pct(k * m['captura_esperada'])} ± "
                  f"{_F.num(100 * k * m['captura_desviacion'], 1)} puntos")
    return texto + "."


def frase_empate_corta(modelo: str, clientes: bool = False):
    """Lo mismo en una línea, para un subtítulo o una tabla: «uno de los desempates posibles (tantos clientes comparten el
    puntaje del corte y entran tantos): con un desempate al azar se espera tal captura ± tantos puntos». Con clientes=True, en
    clientes alcanzados («se esperan tantos ± tantos clientes»). Sin mayúscula ni punto final. None si en el corte no hay
    empate."""
    m = empate_campana(modelo)
    if m is None:
        return None
    if clientes and _es_numero(m.get("positivos")):
        esperado = (f"se esperan {_F.entero(m['captura_esperada'] * m['positivos'])} ± "
                    f"{_F.entero(m['captura_desviacion'] * m['positivos'])} clientes alcanzados")
    else:
        esperado = f"se espera una captura del {_F.pct(m['captura_esperada'])} ± {_F.num(100 * m['captura_desviacion'], 1)} puntos"
    return (f"uno de los desempates posibles ({_F.entero(m['empatados'])} clientes comparten el puntaje del corte y entran "
            f"{_F.entero(m['cupos'])}): con un desempate al azar {esperado}")


def frase_empate_auc_pr(modelo: str, d: int = 3, corta: bool = False):
    """Una nota para poner junto al AUC-PR (la precisión media) de un modelo: average_precision_score cuenta a los clientes
    con el mismo puntaje como un solo punto de la curva; la frase da el valor con los empates rotos al azar y su desviación.
    None si el archivo no está o si la diferencia entre las dos convenciones no pasa de DESVIACIONES_EMPATE desviaciones."""
    e = empates_campana()
    m = None if e is None else e["modelos"].get(modelo)
    if not isinstance(m, dict) or not all(_es_numero(m.get(k)) for k in _CAMPOS_PRECISION_MEDIA):
        return None
    if m["precision_media_azar"] - m["precision_media"] <= DESVIACIONES_EMPATE * m["precision_media_azar_desviacion"]:
        return None
    azar = f"{_F.num(m['precision_media_azar'], d)} ± {_F.num(m['precision_media_azar_desviacion'], d)}"
    if corta:
        return f"Cuenta a los empatados como un solo punto; con los empates rotos al azar, {azar}."
    return (f"El AUC-PR {_de(en_frase(modelo))} ({_F.num(m['precision_media'], d)}) cuenta a los clientes con el mismo puntaje como un "
            f"solo punto de la curva; con los empates rotos al azar es {azar}.")


# --- El error de calibración esperado (ECE) ---------------------------------------------------------------------------------
# La app calcula el ECE con tramos de igual ancho entre 0 y 1 (la función ece de preparar_artefactos.py y de
# preparar_original.py), que es la definición del Entregable 3. El libro del proyecto final lo calcula, en su Tabla 7.11, con
# tramos de igual número de clientes: las cifras de uno y de otro no coinciden y ninguna es un error, así que no se cambia un
# cálculo para igualar al otro. Se dice con la misma frase en todas las vistas que muestran un ECE.
TRAMOS_ECE = 10
"""Tramos de igual ancho con que se calcula el ECE que muestra la app (el mismo número que usa el libro, con otro reparto)."""
TABLA_ECE_LIBRO = "Tabla 7.11"
"""La tabla del libro del proyecto final con el ECE de los diez clasificadores, en tramos de igual número de clientes."""


def nota_ece() -> str:
    """Una nota para poner, una vez por vista, donde se muestra un ECE: que la app lo calcula con TRAMOS_ECE tramos de igual
    ancho y que la tabla del libro usa tramos de igual número de clientes, por lo que sus cifras no coinciden exactamente."""
    tramos = {10: "diez"}.get(TRAMOS_ECE, _F.entero(TRAMOS_ECE))
    return (f"El ECE (error de calibración esperado) se calcula aquí con {tramos} tramos de igual ancho, como en el Entregable 3; la "
            f"{TABLA_ECE_LIBRO} del libro del proyecto usa tramos de igual número de clientes, y por eso sus cifras no coinciden "
            "exactamente con estas.")


def pruebas_original(tarea: str):
    """Las pruebas de hipótesis del modelo original frente a cada benchmark en una tarea (datos/original/pruebas_hipotesis.csv):
    una fila por benchmark y prueba (P1 a P4), con el estadístico, el p crudo, el p ajustado por Holm y los tamaños del
    efecto. Una ventaja positiva favorece al original. None si todavía no existe."""
    t = original_tabla("pruebas_hipotesis.csv")
    return None if t is None else t[t["tarea"] == tarea].reset_index(drop=True)


def pliegues(tarea: str):
    """La métrica principal de cada pliegue externo de los diez modelos (datos/original/pliegues_<tarea>.csv): una fila por
    pliegue, con el índice «semilla-pliegue», y una columna por modelo. None si todavía no existe."""
    rel = f"original/pliegues_{tarea}.csv"
    if not (DATOS / rel).is_file():
        return None
    t = tabla(rel)
    return t.set_index(t.columns[0]).rename_axis("pliegue")


def modelo_original(tarea: str):
    """datos/original/modelo_<tarea>.json: el modelo final de BDL por dentro (K, pesos, desplazamientos, árboles, historial
    de convergencia, tabla del BIC, conjunto de predicción, importancia y curvas). None si todavía no existe."""
    return original_json(f"modelo_{tarea}.json")


def corrida_original(tarea: str, modelo: str, semilla: int = 42):
    """La corrida de un modelo en la tabla maestra de la corrección (datos/original/experimentos.csv), con sus
    hiperparámetros, sus métricas por pliegue y los hiperparámetros que eligió la búsqueda en cada pliegue externo ya como
    objetos. None si no está."""
    t = original_tabla("experimentos.csv")
    if t is None:
        return None
    f = t[(t["tarea"] == tarea) & (t["modelo"] == modelo) & (t["semilla"] == semilla)]
    if f.empty:
        return None
    f = f.iloc[0].copy()
    for col in ("puntajes_ext", "hiperparametros", "params_por_pliegue"):
        if col in f.index:
            f[col] = _json.loads(f[col]) if isinstance(f[col], str) else None
    return f


def costo_original(tarea: str, modelo: str):
    """La fila de datos/original/costo.csv de BDL o de LightGBM en una tarea (ajuste, inferencia, corrida, tamaño,
    parámetros, presupuesto y de dónde sale cada cifra). None si no está."""
    t = original_tabla("costo.csv")
    if t is None:
        return None
    f = t[(t["tarea"] == tarea) & (t["modelo"] == modelo)]
    return None if f.empty else f.iloc[0]


def lideres(tarea: str) -> pd.DataFrame:
    """La mejor combinación de cada modelo con las columnas de resultados/<tarea>_prueba.csv: las ocho filas del informe y,
    si ya existen, las de BDL y LightGBM (datos/original/lideres_<tarea>.csv, con las mismas definiciones), en el orden de
    ORDEN_MODELOS. Añade «origen» («Entregable 3» o «corrección») y, con la métrica por pliegue de la corrección
    (datos/original/pliegues_<tarea>.csv), «externa_media», «externa_desv» y «externa_pliegues»: la media y la desviación
    de todos los pliegues externos disponibles, los mismos para los diez modelos. Sin ese archivo, son las de la
    repetición de la semilla 42 (las columnas *_ext_media y *_ext_desv)."""
    metrica = "auc" if tarea == "clasificacion" else "rmse"
    t = tabla(f"resultados/{tarea}_prueba.csv").assign(origen="Entregable 3")
    nuevas = original_tabla(f"lideres_{tarea}.csv")
    if nuevas is not None and len(nuevas):
        t = pd.concat([t, nuevas.assign(origen="corrección")], ignore_index=True)
    t["externa_media"], t["externa_desv"] = t[f"{metrica}_ext_media"], t[f"{metrica}_ext_desv"]
    t["externa_pliegues"] = len(_tabla(f"resultados/pliegues_{metrica}.csv"))
    p = pliegues(tarea)
    if p is not None and set(t["modelo"]) <= set(p.columns):
        t["externa_media"] = t["modelo"].map(p.mean())
        t["externa_desv"] = t["modelo"].map(p.std(ddof=1))
        t["externa_pliegues"] = len(p)
    return ordenar_modelos(t, tarea)


def hay_modelo_original(tarea: str) -> bool:
    """Si la corrección ya dejó la fila del modelo original en la tabla de líderes de esa tarea."""
    t = original_tabla(f"lideres_{tarea}.csv")
    return t is not None and ORIGINAL in set(t["modelo"])


def ablacion() -> pd.DataFrame:
    """resultados/ablacion.csv con la columna «variante» sin marcas de imprenta. delta_completo positivo = la variante
    pierde frente a la EBM completa."""
    t = tabla("resultados/ablacion.csv")
    t["variante"] = t["variante"].map(texto_llano)
    return t


def errores_segmentos() -> pd.DataFrame:
    t = tabla("resultados/errores_segmentos.csv")
    t["nivel"] = t["nivel"].map(texto_llano)
    return t


def techo() -> dict:
    """Techo de información (comparación con la Entrega 2, etiqueta de la mediana): auc_mediana, r2 y r2_oraculo."""
    return json("entrega3/comparacion_entrega2/techo.json")


def resumen() -> dict:
    """resultados/resumen.json: umbral, prevalencia, ROPE, réplicas bootstrap, memoria de la EBM y miembros del MCS."""
    return json("resultados/resumen.json")


# --- Veredicto de la EBM frente a una referencia (el mismo criterio en todas las pantallas) -------------------------------
ALFA = 0.05
"""Nivel de significancia del informe, fijado antes de evaluar la prueba."""

VEREDICTO_TEXTO = {"supera": "supera tras Holm", "empata": "sin diferencia", "discrepan": "las pruebas discrepan",
                   "pierde": "pierde tras Holm"}


def veredicto(f, tarea: str) -> str:
    """Qué dicen las pruebas de una fila de pruebas(tarea), con los p ajustados por Holm y ALFA: «supera» (la t corregida y
    DeLong o Diebold-Mariano rechazan a favor de la EBM), «pierde» (rechazan en contra), «empata» (ninguna rechaza) o
    «discrepan» (una rechaza y la otra no). McNemar no entra: en el punto de operación del informe casi nunca separa."""
    delta = f["delta_auc"] if tarea == "clasificacion" else f["delta_rmse"]
    rechaza = [f["p_t_holm"] < ALFA, (f["p_delong_holm"] if tarea == "clasificacion" else f["p_dm_holm"]) < ALFA]
    if all(rechaza):
        return "supera" if delta > 0 else "pierde"
    return "discrepan" if any(rechaza) else "empata"


def veredictos(tarea: str) -> pd.Series:
    """El veredicto de cada referencia (índice: clave del modelo), en el orden del informe."""
    t = pruebas(tarea)
    return pd.Series([veredicto(f, tarea) for _, f in t.iterrows()], index=t["modelo"].tolist())
