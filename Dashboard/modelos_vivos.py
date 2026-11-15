"""Los 16 modelos finales en vivo: catálogo, carga de artifacts/, predicción, incertidumbre y explicación local.

La app nunca entrena ni optimiza: carga las tuberías ya entrenadas de artifacts/<tarea>/<modelo>/modelo.joblib (las arma
preparar_artefactos.py con la prueba de ida y vuelta) y las evalúa. Las carga cuando se piden y las guarda (lru_cache):
las 14 juntas ocupan unos 260 MB en Windows (menos en Linux, con XGBoost sin CUDA). Si el servidor se queda sin memoria,
la variable FUGA_CACHE_MODELOS limita cuántas guarda (con 3, cada cliente nuevo recarga las que falten). La EBM se evalúa con motor.py (sus tablas en numpy,
idénticas al artefacto), así que la app no necesita interpret.

    predecir(tarea, modelo, X)       el puntaje del modelo: probabilidad (o margen del SVM lineal) o churn_probability
    probabilidad(modelo, s)          probabilidad recalibrada (isotónica sobre predicciones fuera de pliegue)
    conjunto(modelo, p, nivel)       conjunto conforme de clases al nivel dado, con su cobertura medida en prueba
    intervalo(modelo, yhat, nivel)   intervalo conforme de regresión, con su cobertura medida en prueba
    explicar(tarea, modelo, X)       valores SHAP por variable (explicacion.py; la EBM, exacta con motor.py)
    vecinos(tarea, modelo, X)        k-NN: los vecinos de la fila en el entrenamiento
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import explicacion
import formato as F
import motor

AQUI = Path(__file__).resolve().parent
ARTEFACTOS = AQUI / "artifacts"
DATOS = AQUI / "datos"
TAREAS = ("clasificacion", "regresion")
NIVELES = ("0.80", "0.90", "0.95")
CACHE_MODELOS = int(os.environ.get("FUGA_CACHE_MODELOS", "16"))


def _json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def indice() -> dict:
    """artifacts/indice.json: los 16 artefactos, sus huellas, versiones y el resultado de la ida y vuelta."""
    return _json(ARTEFACTOS / "indice.json")


def modelos(tarea: str) -> list[str]:
    """Los ocho modelos de una tarea, en el orden de las tablas del informe."""
    import datos_app as D
    hay = {a["modelo"] for a in indice()["artefactos"] if a["tarea"] == tarea}
    return [m for m in D.ORDEN_MODELOS[tarea] if m in hay]


@lru_cache(maxsize=None)
def metadata(tarea: str, modelo: str) -> dict:
    """El metadata.json de un artefacto: versiones, semilla, hiperparámetros, esquema, métricas, tiempos, ida y vuelta."""
    return _json(ARTEFACTOS / tarea / modelo / "metadata.json")


@lru_cache(maxsize=1)
def esquema() -> list[dict]:
    """Las 48 variables de entrada: nombre, nombre visible, tipo, rango válido o categorías (el mismo en los 16)."""
    return metadata("clasificacion", "ebm")["entrada"]["esquema"]


def columnas() -> list[str]:
    return [v["nombre"] for v in esquema()]


@lru_cache(maxsize=CACHE_MODELOS)
def tuberia(tarea: str, modelo: str):
    """La tubería de scikit-learn de un artefacto (no para la EBM, que se evalúa con motor.py)."""
    if modelo == "ebm":
        raise ValueError("la EBM se evalúa con motor.py")
    carpeta = ARTEFACTOS / tarea / modelo
    if (carpeta / "bosque.npz").exists():           # Random Forest de clasificación: sus mismos árboles en numpy (bosque.py)
        import bosque
        return bosque.cargar(carpeta)
    import joblib
    return joblib.load(carpeta / "modelo.joblib")


def a_tabla(fila: dict) -> pd.DataFrame:
    """Una fila de entrada (dict variable -> valor) como DataFrame con los tipos del entrenamiento."""
    X = pd.DataFrame([{v: fila.get(v) for v in columnas()}])
    for v in esquema():
        n = v["nombre"]
        if v["tipo"] == "booleana":
            X[n] = X[n].astype(bool)
        elif v["tipo"] == "categorica":
            X[n] = X[n].astype("string")
        elif v["tipo"] == "entera":
            X[n] = X[n].astype("int64")
        else:
            X[n] = X[n].astype(float)
    return X


def validar(fila: dict) -> list[str]:
    """Los problemas de una fila de entrada (vacía si es válida): faltantes, tipos y valores fuera del rango de
    entrenamiento o categorías desconocidas."""
    problemas = []
    for v in esquema():
        n, nombre, x = v["nombre"], v.get("nombre_visible") or v["nombre"], fila.get(v["nombre"])
        if x is None or (isinstance(x, float) and np.isnan(x)) or x == "":
            problemas.append(f"{nombre}: falta el valor")
            continue
        if v["tipo"] == "categorica":
            if str(x) not in v["categorias"]:
                problemas.append(f"{nombre}: «{x}» no es una de sus {len(v['categorias'])} categorías")
        elif v["tipo"] == "booleana":
            if x not in (True, False, 0, 1):
                problemas.append(f"{nombre}: debe ser sí o no")
        else:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                problemas.append(f"{nombre}: debe ser un número")
                continue
            if not math.isfinite(xf):
                problemas.append(f"{nombre}: debe ser un número finito")
                continue
            if v["tipo"] == "entera" and xf != int(xf):
                problemas.append(f"{nombre}: debe ser un número entero")
            if xf < v["min"] or xf > v["max"]:
                problemas.append(f"{nombre}: {_cifra(xf)} está fuera del rango del entrenamiento "
                                 f"({_cifra(v['min'])} a {_cifra(v['max'])})")
    return problemas


def _cifra(x: float) -> str:
    """Un valor de un rango en es-CO: entero con puntos de miles, o con hasta cuatro decimales."""
    return F.entero(x) if float(x) == int(x) else F.num(x, 4).rstrip("0").rstrip(",")


def predecir(tarea: str, modelo: str, X: pd.DataFrame) -> np.ndarray:
    """El puntaje guardado por el proyecto: probabilidad sin recalibrar (el SVM lineal da su margen) o churn_probability."""
    X = X[columnas()]
    if modelo == "ebm":
        return motor.ebm(tarea).predecir(X)
    pipe = tuberia(tarea, modelo)
    if tarea == "regresion":
        return np.asarray(pipe.predict(X), dtype=float)
    if hasattr(pipe.named_steps["modelo"], "predict_proba"):
        return np.asarray(pipe.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(pipe.decision_function(X), dtype=float)


@lru_cache(maxsize=1)
def calibracion() -> dict:
    return _json(DATOS / "calibracion_modelos.json")


@lru_cache(maxsize=1)
def conformal() -> dict:
    return _json(DATOS / "conformal.json")


def probabilidad(modelo: str, s) -> np.ndarray:
    """Probabilidad recalibrada de riesgo alto (isotónica ajustada con las predicciones fuera de pliegue)."""
    c = calibracion()["modelos"][modelo]
    return np.interp(np.asarray(s, dtype=float), np.asarray(c["x"], dtype=float), np.asarray(c["y"], dtype=float))


def conjunto(modelo: str, p_cal: float, nivel: str = "0.90") -> dict:
    """Conjunto conforme de clases para una probabilidad recalibrada: {"clases": [...], "nivel", "cobertura_prueba"}."""
    c = calibracion()["modelos"][modelo]["conformal"][nivel]
    clases = [k for k, prob in (("riesgo alto", p_cal), ("riesgo bajo", 1 - p_cal)) if 1 - prob <= c["umbral"]]
    return {"clases": clases, "nivel": float(nivel), "cobertura_prueba": c["cobertura_prueba"], "umbral": c["umbral"]}


def intervalo(modelo: str, yhat: float, nivel: str = "0.90") -> dict:
    """Intervalo conforme de regresión: {"desde", "hasta", "nivel", "cobertura_prueba", "semiancho"}."""
    c = conformal()["modelos"][modelo][nivel]
    return {"desde": yhat - c["semiancho"], "hasta": yhat + c["semiancho"], "nivel": float(nivel),
            "cobertura_prueba": c["cobertura_prueba"], "semiancho": c["semiancho"]}


@lru_cache(maxsize=1)
def fondo() -> pd.DataFrame:
    """Los 100 clientes de entrenamiento que sirven de fondo a las explicaciones."""
    return pd.read_parquet(DATOS / "fondo.parquet")


def explicar(tarea: str, modelo: str, X: pd.DataFrame) -> dict:
    """Valores SHAP por variable para las filas de X (ver explicacion.py). La EBM, con su descomposición exacta."""
    X = X[columnas()]
    if modelo == "ebm":
        m = motor.ebm(tarea)
        aportes = m.por_variable(X)[columnas()]
        y = m.puntaje(X)
        base = np.full(len(X), m.intercepto)
        return {"aportes": aportes, "base": base, "salida": y, "metodo": "ebm", "exacto": True, "error": None,
                "escala": "logit del riesgo alto" if tarea == "clasificacion" else "churn_probability predicha",
                "eficiencia": float(np.max(np.abs(base + aportes.sum(axis=1).to_numpy() - y)))}
    return explicacion.explicar(tuberia(tarea, modelo), X, fondo(), tarea)


@lru_cache(maxsize=1)
def entrenamiento() -> pd.DataFrame:
    """El entrenamiento en el orden en que se ajustaron los modelos (datos/eda/entrenamiento.parquet)."""
    return pd.read_parquet(DATOS / "eda" / "entrenamiento.parquet")


def vecinos(tarea: str, modelo: str, X: pd.DataFrame) -> dict:
    """k-NN: los vecinos de la primera fila de X en el entrenamiento, con sus filas (para mostrar en qué se parecen)."""
    v = explicacion.vecinos(tuberia(tarea, modelo), X[columnas()].iloc[[0]])
    v["filas"] = entrenamiento().iloc[v["posiciones"]]
    return v


@lru_cache(maxsize=1)
def shap_indice() -> dict:
    """datos/shap/indice.json: método, escala, importancia media y verificación del resumen SHAP de cada modelo."""
    return _json(DATOS / "shap" / "indice.json")


@lru_cache(maxsize=16)
def shap_muestra(tarea: str, modelo: str) -> pd.DataFrame:
    """Valores SHAP de la muestra fija de clientes de prueba (índice «fila»; columnas: variables, __base, __salida)."""
    return pd.read_parquet(DATOS / "shap" / f"{tarea}__{modelo}.parquet").set_index("fila")
