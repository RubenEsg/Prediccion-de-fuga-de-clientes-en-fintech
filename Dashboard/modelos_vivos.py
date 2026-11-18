"""Los modelos finales en vivo: catálogo, carga de artifacts/, predicción, incertidumbre y explicación local.

Son los 16 del Entregable 3 (ocho por tarea) y el modelo original del proyecto, BDL (Boosting con Desplazamiento Latente),
en las tareas en que la corrección ya lo dejó (artifacts/indice.json lo dice). LightGBM, el segundo benchmark reciente,
solo aparece en las tablas: no tiene artefacto, para no añadir una dependencia al servidor.

La app nunca entrena ni optimiza: carga las tuberías ya entrenadas de artifacts/<tarea>/<modelo>/modelo.joblib (las arman
preparar_artefactos.py y preparar_original.py con la prueba de ida y vuelta) y las evalúa. Las carga cuando se piden y las
guarda (lru_cache): las 14 del Entregable 3 ocupan juntas unos 260 MB en Windows (menos en Linux, con XGBoost sin CUDA) y
las dos de BDL, menos de 1 MB. Si el servidor se queda sin memoria, la variable FUGA_CACHE_MODELOS limita cuántas guarda
(con 3, cada cliente nuevo recarga las que falten). La EBM se evalúa con motor.py (sus tablas en numpy, idénticas al
artefacto), así que la app no necesita interpret. BDL solo necesita numpy, scipy y scikit-learn: su código está en
src/original_model.py, copia exacta del de la corrección, y las tuberías serializadas nombran ese módulo.

    predecir(tarea, modelo, X)       el puntaje del modelo: probabilidad (o margen del SVM lineal) o churn_probability
    probabilidad(modelo, s)          probabilidad recalibrada (isotónica sobre predicciones fuera de pliegue)
    conjunto(modelo, p, nivel)       conjunto conforme de clases al nivel dado, con su cobertura medida en prueba
    intervalo(modelo, yhat, nivel)   intervalo conforme de regresión, con su cobertura medida en prueba
    explicar(tarea, modelo, X)       valores SHAP por variable (explicacion.py; la EBM, exacta con motor.py; BDL, con
                                     RegresorBDL.contribuciones)
    vecinos(tarea, modelo, X)        k-NN: los vecinos de la fila en el entrenamiento
    excedencia(X)                    BDL, clasificación: cómo se convierte el índice F(x) en la probabilidad de riesgo alto
    distribucion(X, nivel)           BDL, regresión: el índice, la media, la mediana, el intervalo central y el conjunto de
                                     predicción del modelo (una unión de tramos), con su cobertura medida en prueba
"""
from __future__ import annotations

import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import explicacion
import formato as F
import motor

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:                 # src.original_model: lo nombran las tuberías serializadas del modelo original
    sys.path.insert(0, str(AQUI))
ARTEFACTOS = AQUI / "artifacts"
DATOS = AQUI / "datos"
TAREAS = ("clasificacion", "regresion")
NIVELES = ("0.80", "0.90", "0.95")
ORIGINAL = "bdl"
CACHE_MODELOS = int(os.environ.get("FUGA_CACHE_MODELOS", "16"))


def _json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def indice() -> dict:
    """artifacts/indice.json: los artefactos (los 16 del Entregable 3 y los del modelo original), sus huellas, versiones y
    el resultado de la ida y vuelta."""
    return _json(ARTEFACTOS / "indice.json")


def modelos(tarea: str) -> list[str]:
    """Los modelos en vivo de una tarea (los que tienen artefacto), en el orden de datos_app.ORDEN_MODELOS: el modelo
    original, si la corrección ya lo dejó en esa tarea, y los ocho del Entregable 3."""
    import datos_app as D
    hay = {a["modelo"] for a in indice()["artefactos"] if a["tarea"] == tarea}
    return [m for m in D.ORDEN_MODELOS[tarea] if m in hay]


def hay_original(tarea: str) -> bool:
    """Si el modelo original tiene artefacto en esa tarea (la corrección puede no haber terminado todavía una de las dos)."""
    return ORIGINAL in modelos(tarea)


@lru_cache(maxsize=None)
def metadata(tarea: str, modelo: str) -> dict:
    """El metadata.json de un artefacto: versiones, semilla, hiperparámetros, esquema, métricas, tiempos, ida y vuelta."""
    return _json(ARTEFACTOS / tarea / modelo / "metadata.json")


@lru_cache(maxsize=1)
def esquema() -> list[dict]:
    """Las 48 variables de entrada: nombre, nombre visible, tipo, rango válido o categorías (el mismo en todos)."""
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


# --- Lo que solo tiene el modelo original (BDL) ----------------------------------------------------------------------------
def _original(tarea: str):
    """(preprocesado, estimador, RegresorBDL) del modelo original en una tarea; el regresor es None si el clasificador está
    en modo directo (aprendió de la etiqueta, sin mezcla)."""
    pipe = tuberia(tarea, ORIGINAL)
    est = pipe.named_steps["modelo"]
    return pipe.named_steps["preprocesado"], est, explicacion.regresor_bdl(est)


def excedencia(X: pd.DataFrame):
    """Clasificación con el modelo original: cómo sale la probabilidad de riesgo alto del índice de cada fila de X.

    P(s > τ | x) = Σ_k π_k · Φ((F(x) + c_k − τ) / σ_r): cada clase latente aporta su peso si su nivel, F(x) + c_k, queda
    por encima del umbral τ. Devuelve {"indice" (n,), "umbral", "pesos" (K,), "desplazamientos" (K,), "niveles" (n, K),
    "aporte" (n, K), "probabilidad" (n,), "sigma" (la dispersión robusta de validación), "piso" y "techo" (los límites
    que impone el tamaño de la muestra de calibración)}; None si el clasificador está en modo directo."""
    pre, est, reg = _original("clasificacion")
    if reg is None:
        return None
    from scipy.special import ndtr
    z = pre.transform(X[columnas()])
    f = np.asarray(reg.indice(z), dtype=float)
    niveles = f[:, None] + reg.centros_[None, :]
    aporte = ndtr((niveles - est.umbral_) / reg.sigma_robusta_) * reg.pi_[None, :]
    m = len(reg.residuos_calibracion_)
    return {"indice": f, "umbral": float(est.umbral_), "pesos": np.asarray(reg.pi_, dtype=float),
            "desplazamientos": np.asarray(reg.centros_, dtype=float), "niveles": niveles, "aporte": aporte,
            "probabilidad": np.asarray(est.predict_proba(z)[:, 1], dtype=float), "sigma": float(reg.sigma_robusta_),
            "piso": 0.5 / (m + 1.0), "techo": 1.0 - 0.5 / (m + 1.0)}


@lru_cache(maxsize=None)
def _tramos_residuales(nivel: str) -> tuple:
    """El conjunto de predicción del modelo original en la escala del residuo (no depende del cliente), como tuplas."""
    reg = _original("regresion")[2]
    return tuple((float(a), float(b)) for a, b in reg.conjunto_residual(float(nivel)))


def distribucion(X: pd.DataFrame, nivel: str = "0.90"):
    """Regresión con el modelo original: lo que predice para la primera fila de X además de la media.

    Devuelve {"indice" (F(x)), "media" (el pronóstico, F(x) + Σ π_k c_k), "mediana", "central" ((desde, hasta): el
    intervalo central conforme del modelo), "tramos" (lista de (desde, hasta): el conjunto de predicción, F(x) más los
    tramos alrededor de cada desplazamiento), "largo" (la suma de los largos de los tramos), "niveles" (F(x) + c_k),
    "pesos" (π_k), "sigma" (dispersión dentro de clase en validación), "nivel", "cobertura_conjunto",
    "cobertura_central" y "largo_central_medio" (las tres medidas en la prueba, de datos/conformal.json)}."""
    pre, _est, reg = _original("regresion")
    z = pre.transform(X[columnas()].iloc[[0]])
    f = float(reg.indice(z)[0])
    bajo, alto = reg.intervalo(z, float(nivel))
    tramos = [(f + a, f + b) for a, b in _tramos_residuales(nivel)]
    medido = conformal()["modelos"].get(ORIGINAL, {}).get(nivel, {})
    return {"indice": f, "media": float(reg.predict(z)[0]), "mediana": float(reg.cuantil(z, 0.5)[0]),
            "central": (float(bajo[0]), float(alto[0])), "tramos": tramos, "largo": float(sum(b - a for a, b in tramos)),
            "niveles": [f + float(c) for c in reg.centros_], "pesos": [float(p) for p in reg.pi_],
            "sigma": float(reg.sigma_validacion_), "nivel": float(nivel),
            "cobertura_conjunto": (medido.get("conjunto") or {}).get("cobertura_prueba"),
            "cobertura_central": (medido.get("central") or {}).get("cobertura_prueba"),
            "largo_central_medio": (medido.get("central") or {}).get("largo_medio")}


@lru_cache(maxsize=1)
def fondo() -> pd.DataFrame:
    """Los 100 clientes de entrenamiento que sirven de fondo a las explicaciones."""
    return pd.read_parquet(DATOS / "fondo.parquet")


def explicar(tarea: str, modelo: str, X: pd.DataFrame) -> dict:
    """Valores SHAP por variable para las filas de X (ver explicacion.py). La EBM, con su descomposición exacta; el modelo
    original, con la suya (en clasificación descompone el índice F(x): excedencia(X) lo lleva a probabilidad)."""
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
