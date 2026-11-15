"""Lectura de la carpeta datos/ de la app: una función con caché por tipo de archivo y los nombres visibles del informe.

    import datos_app as D
    D.tabla("resultados/clasificacion_prueba.csv")     -> copia de la tabla (se puede modificar sin dañar la caché)
    D.json("entrega3/comparacion_entrega2/techo.json") -> dict
    D.clave_valor("resultados/fidelidad_resumen.csv")  -> dict de los csv que son pares clave–valor
    D.fila(D.tabla(...), modelo="ebm")                 -> la fila (Series) que cumple las condiciones
    D.fuente("resultados/costo.csv")                   -> "datos/resultados/costo.csv", para fuentes=[…] de las tarjetas
    D.NOMBRE_MODELO["ebm"]                             -> "EBM (propuesto)"   (los nombres exactos del informe)

Los archivos y sus columnas están descritos en datos/origen.json (de dónde salió cada uno) y en CONTRATO.md. El modelo en
vivo y los clientes de prueba se leen con motor.py, no aquí.
"""
from __future__ import annotations

import json as _json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"

NECESARIOS = ["metadatos.json", "clientes_prueba.parquet", "puntajes_prueba.parquet", "pronosticos_prueba.parquet",
              "calibracion_isotonica_ebm.json", "variables.csv", "modelos/ebm_clasificacion.json", "modelos/ebm_regresion.json",
              "modelos/logistica_clasificacion.json", "resultados/clasificacion_prueba.csv", "resultados/regresion_prueba.csv",
              "eda/por_variable.csv", "entrega3/corridas.csv"]

# --- Nombres visibles: los del informe (Resultados Entregable 3/generar.py, diccionarios NOMBRES, BALANCEO y OPTIM) --------
NOMBRE_MODELO = {"ebm": "EBM (propuesto)", "xgboost": "XGBoost", "random_forest": "Random Forest", "arbol": "Árbol de decisión",
                 "knn": "k-NN", "logistica": "Regresión logística", "svm": "SVM lineal", "naive_bayes": "Naive Bayes",
                 "ridge": "Ridge", "lasso": "Lasso", "svr": "SVR lineal",
                 "logistica_e2": "Logística L1 (E2)", "lasso_e2": "Lasso (E2)"}
NOMBRE_CORTO = dict(NOMBRE_MODELO, ebm="EBM", arbol="Árbol", logistica="Logística")
NOMBRE_BALANCEO = {"ninguno": "sin balanceo", "class_weight": "ponderación", "smote": "SMOTE", "adasyn": "ADASYN"}
NOMBRE_OPTIMIZADOR = {"grid": "Grid", "random": "Random", "bayesiana": "TPE", "genetica": "Genética"}
NOMBRE_TAREA = {"clasificacion": "Clasificación", "regresion": "Regresión"}

# Orden de las tablas del informe: clasificación por AUC de prueba (de mayor a menor) y regresión por RMSE de prueba
ORDEN_MODELOS = {"clasificacion": ["ebm", "xgboost", "random_forest", "arbol", "knn", "logistica", "svm", "naive_bayes"],
                 "regresion": ["xgboost", "ebm", "random_forest", "arbol", "lasso", "ridge", "svr", "knn"]}
ORDEN_OPTIMIZADORES = ["grid", "random", "bayesiana", "genetica"]
ORDEN_BALANCEOS = ["ninguno", "class_weight", "smote", "adasyn"]

# Rol de color de cada modelo (DISENO.md): la EBM es lo que se propone; los modelos de árboles son los rivales con los que
# empata; los lineales, k-NN, Naive Bayes y la Entrega 2 son la referencia que se supera.
ROL_MODELO = {"ebm": "modelo", "xgboost": "rival", "random_forest": "rival", "arbol": "rival", "knn": "base", "logistica": "base",
              "svm": "base", "naive_bayes": "base", "ridge": "base", "lasso": "base", "svr": "base", "logistica_e2": "base",
              "lasso_e2": "base"}


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
    """El modelo dentro de una frase: «la EBM», «la regresión logística», «el árbol de decisión», «el SVM lineal»,
    «XGBoost», «Random Forest», «k-NN», «Naive Bayes». Para rótulos, tablas y leyendas se usa NOMBRE_MODELO."""
    n = NOMBRE_CORTO[clave] if clave == "ebm" else NOMBRE_MODELO.get(clave, clave)
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
    """Desempeño de la mejor combinación de cada modelo: resultados/clasificacion_prueba.csv o regresion_prueba.csv, en el
    orden del informe."""
    return ordenar_modelos(tabla(f"resultados/{tarea}_prueba.csv"), tarea)


def pruebas(tarea: str) -> pd.DataFrame:
    """Pruebas de significancia de la EBM frente a cada referencia (resultados/pruebas_<tarea>.csv). Un delta positivo
    favorece a la EBM (más AUC o menos RMSE)."""
    return ordenar_modelos(tabla(f"resultados/pruebas_{tarea}.csv"), tarea)


def puntajes() -> pd.DataFrame:
    """Puntaje de los ocho clasificadores finales para cada cliente de prueba (índice «fila»; columna «y» = riesgo alto).
    Compartida: no modificar."""
    return compartida("puntajes_prueba.parquet")


def pronosticos() -> pd.DataFrame:
    """Pronóstico de los ocho regresores finales para cada cliente de prueba (índice «fila»; columna «y» = churn_probability).
    Compartida: no modificar."""
    return compartida("pronosticos_prueba.parquet")


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
