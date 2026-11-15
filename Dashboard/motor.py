"""Motor del modelo en vivo: puntúa, explica y simula clientes con las tablas de la propia EBM, en numpy puro.

Una Explainable Boosting Machine sin interacciones es una suma de tablas: para cada término, unos cortes y un puntaje por
tramo. preparar_datos.py exporta esas tablas del modelo final guardado por el proyecto (datos/modelos/ebm_*.json) y este
módulo las evalúa. No necesita interpret ni scikit-learn; verificar_motor.py comprueba que da lo mismo que el modelo
original (diferencia de redondeo de máquina).

    import motor as M
    M.clientes()                          -> los 9.745 clientes de prueba (índice «fila»)
    M.puntuar(X)                          -> puntaje de la EBM, su versión recalibrada, pronóstico y puntaje de la logística
    M.explicar(X)                         -> (intercepto, contribuciones n × 48): intercepto + suma = logit del puntaje
    M.simular(fila, {"age": 60})          -> el cliente con el cambio y sus variables atadas recalculadas (1 fila)
    M.forma("age")                        -> la curva que el modelo aprendió para la edad, en años
    M.resumen_capacidad(y, s, 0.25)       -> a quién alcanza una campaña que contacta al 25 % de mayor puntaje

Todas las lecturas van con caché y son de solo lectura (los callbacks pueden correr en varios hilos).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"

PRODUCTOS = ["savings_account", "credit_card", "personal_loan", "investment_account", "insurance_product"]
"""Los cinco productos cuya tenencia define product_satisfaction (y, por ahí, satisfaction_score)."""

TOPE_TX_SATISFACCION = 200
"""tx_satisfaction = min(tx_count / 200, 1): identidad exacta en el 100 % de los clientes."""

SEMILLA = 42
CAPACIDAD_INFORME = 0.25
"""Punto de operación del informe: se contacta al 25 % de clientes de mayor puntaje."""


# ============================================================================ los modelos como tablas
class MotorEBM:
    """EBM de efectos principales exportada a un diccionario (preparar_datos.py::exportar_ebm)."""

    def __init__(self, d: dict) -> None:
        self.d = d
        self.intercepto = float(d["intercepto"])
        self.enlace = d["enlace"]
        self.columnas = list(d["columnas"])
        self.terminos = d["terminos"]
        self.nombres = [t["termino"] for t in self.terminos]
        self.variable_de = np.array([t["variable"] for t in self.terminos])
        self.categoricas = d["categoricas"]
        self._cortes = [np.asarray(t["cortes"], dtype=float) for t in self.terminos]
        self._punt = [np.asarray(t["puntajes"], dtype=float) for t in self.terminos]

    @property
    def n_parametros(self) -> int:
        return int(sum(len(p) for p in self._punt))

    # ---------------------------------------------------------------- puntuación
    @staticmethod
    def _z(X: pd.DataFrame, t: dict) -> np.ndarray:
        """La columna que ve el modelo: la variable estandarizada o la indicadora de una categoría."""
        if t["clase"] == "numerica":
            return (X[t["variable"]].to_numpy(dtype=float) - t["media"]) / t["escala"]
        return (X[t["variable"]].to_numpy(dtype=object) == t["categoria"]).astype(float)

    def contribuciones(self, X: pd.DataFrame) -> np.ndarray:
        """Matriz n × términos con la contribución de cada término (logit en clasificación, objetivo en regresión)."""
        out = np.empty((len(X), len(self.terminos)), dtype=float)
        for j, t in enumerate(self.terminos):
            z = self._z(X, t)
            b = np.searchsorted(self._cortes[j], z, side="right") + 1      # el tramo 0 es «faltante»
            out[:, j] = self._punt[j][np.where(np.isnan(z), 0, b)]
        return out

    def por_variable(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contribuciones agregadas a las variables originales (las indicadoras de una categórica se suman)."""
        c = self.contribuciones(X)
        return pd.DataFrame({v: c[:, self.variable_de == v].sum(axis=1) for v in self.columnas}, index=X.index)

    def puntaje(self, X: pd.DataFrame) -> np.ndarray:
        """Intercepto + suma de contribuciones: logit (clasificación) o pronóstico (regresión)."""
        return self.intercepto + self.contribuciones(X).sum(axis=1)

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad sin recalibrar (clasificación) o pronóstico (regresión)."""
        s = self.puntaje(X)
        return 1.0 / (1.0 + np.exp(-s)) if self.enlace == "logit" else s

    # ---------------------------------------------------------------- explicación global
    def _termino_numerico(self, variable: str) -> tuple[int, dict]:
        for j, t in enumerate(self.terminos):
            if t["variable"] == variable and t["clase"] == "numerica":
                return j, t
        raise KeyError(f"{variable!r} no es una variable numérica del modelo")

    def forma(self, variable: str) -> pd.DataFrame:
        """Función de forma de una variable numérica en sus unidades: un tramo por fila (desde, hasta, efecto, inf, sup).
        inf y sup son el efecto ± la desviación entre las bolsas del modelo."""
        _, t = self._termino_numerico(variable)
        bordes = np.r_[t["limites"][0], t["cortes"], t["limites"][1]] * t["escala"] + t["media"]
        s = np.asarray(t["puntajes"])[1:-1]
        e = np.asarray(t["desviaciones"])[1:-1]
        return pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:], "efecto": s, "inf": s - e, "sup": s + e})

    def densidad(self, variable: str) -> pd.DataFrame:
        """Histograma de entrenamiento de una variable numérica en sus unidades (desde, hasta, peso)."""
        _, t = self._termino_numerico(variable)
        b = np.asarray(t["hist_bordes"]) * t["escala"] + t["media"]
        w = np.asarray(t["hist_pesos"])[1:-1]
        return pd.DataFrame({"desde": b[:-1], "hasta": b[1:], "peso": w})

    def efecto_categorias(self, variable: str) -> pd.Series:
        """Efecto total de cada categoría de una variable categórica (suma de sus indicadoras)."""
        cats = self.categoricas[variable]
        ts = [t for t in self.terminos if t["variable"] == variable]
        base = sum(t["puntajes"][1] for t in ts)                    # todas las indicadoras en 0: la categoría de referencia
        ef = {cats[0]: base}
        for t in ts:
            ef[t["categoria"]] = base - t["puntajes"][1] + t["puntajes"][2]
        return pd.Series(ef, name=variable)

    def efecto_en(self, variable: str, valores) -> np.ndarray:
        """Efecto de una variable (numérica o categórica) en los valores dados, en sus unidades."""
        if variable in self.categoricas:
            return self.efecto_categorias(variable).reindex(list(valores)).to_numpy()
        j, t = self._termino_numerico(variable)
        z = (np.asarray(valores, dtype=float) - t["media"]) / t["escala"]
        return self._punt[j][np.searchsorted(self._cortes[j], z, side="right") + 1]

    def importancia(self) -> pd.Series:
        """Cuota de importancia de cada variable original: la importancia de sus términos (media del valor absoluto de la
        contribución en entrenamiento) sobre el total. Suma 1; va de mayor a menor."""
        imp = pd.Series([t["importancia"] for t in self.terminos]).groupby(self.variable_de).sum()
        return (imp / imp.sum()).sort_values(ascending=False)


class MotorLogistica:
    """Regresión logística exportada a un diccionario (preparar_datos.py::exportar_logistica)."""

    def __init__(self, d: dict) -> None:
        self.d = d
        self.intercepto = float(d["intercepto"])
        self.columnas = list(d["columnas"])
        self.terminos = d["terminos"]
        self.variable_de = np.array([t["variable"] for t in self.terminos])

    def contribuciones(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(X), len(self.terminos)), dtype=float)
        for j, t in enumerate(self.terminos):
            out[:, j] = MotorEBM._z(X, t) * t["coef"]
        return out

    def puntaje(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercepto + self.contribuciones(X).sum(axis=1)

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.puntaje(X)))


# ============================================================================ lecturas con caché
def _json(nombre: str):
    return json.loads((DATOS / nombre).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def meta() -> dict:
    """metadatos.json: partición, etiqueta, punto de operación, métricas reproducidas y de dónde salió cada modelo."""
    return _json("metadatos.json")


@lru_cache(maxsize=1)
def clientes() -> pd.DataFrame:
    """Los clientes de prueba (índice «fila» = posición en customer_data.csv): contexto, los 48 predictores, el objetivo,
    la etiqueta y los puntajes guardados de la EBM, XGBoost y la regresión logística. No modificar: es compartida."""
    return pd.read_parquet(DATOS / "clientes_prueba.parquet")


@lru_cache(maxsize=1)
def variables() -> pd.DataFrame:
    """variables.csv (índice «variable»): nombre visible, tipo, papel en el simulador, rangos de entrenamiento e importancia."""
    return pd.read_csv(DATOS / "variables.csv", keep_default_na=False, na_values=[""]).set_index("variable")


@lru_cache(maxsize=None)
def ebm(tarea: str = "clasificacion") -> MotorEBM:
    """La EBM final de una tarea («clasificacion» o «regresion»)."""
    if tarea not in ("clasificacion", "regresion"):
        raise ValueError(f"tarea desconocida: {tarea!r}")
    return MotorEBM(_json(f"modelos/ebm_{tarea}.json"))


@lru_cache(maxsize=1)
def logistica() -> MotorLogistica:
    """La regresión logística final (la referencia lineal), para comparar en el simulador."""
    return MotorLogistica(_json("modelos/logistica_clasificacion.json"))


@lru_cache(maxsize=1)
def _calibrador() -> tuple[np.ndarray, np.ndarray]:
    c = _json("calibracion_isotonica_ebm.json")
    return np.asarray(c["x"], dtype=float), np.asarray(c["y"], dtype=float)


def predictores() -> list[str]:
    """Los 48 predictores, en el orden del modelo."""
    return list(meta()["predictores"])


def nombre(variable: str) -> str:
    """Nombre visible de una variable (el de variables.csv; si no está, la clave)."""
    v = variables()
    return str(v.at[variable, "nombre"]) if variable in v.index else variable


# ============================================================================ puntuar y explicar
def calibrar(p) -> np.ndarray:
    """Probabilidad recalibrada (isotónica ajustada con las predicciones fuera de pliegue del entrenamiento). Informa la
    probabilidad; para ordenar clientes se usa el puntaje sin recalibrar (la recalibrada tiene pocos valores distintos)."""
    x, y = _calibrador()
    return np.interp(np.asarray(p, dtype=float), x, y)


def puntuar(X: pd.DataFrame) -> pd.DataFrame:
    """Puntajes en vivo para las filas de X (con los 48 predictores por nombre): p_ebm (probabilidad sin recalibrar de
    riesgo alto), p_ebm_cal (recalibrada), pron_ebm (churn_probability prevista) y p_logistica."""
    X = X[predictores()]
    p = ebm("clasificacion").predecir(X)
    return pd.DataFrame({"p_ebm": p, "p_ebm_cal": calibrar(p), "pron_ebm": ebm("regresion").predecir(X),
                         "p_logistica": logistica().predecir(X)}, index=X.index)


def explicar(X: pd.DataFrame, tarea: str = "clasificacion") -> tuple[float, pd.DataFrame]:
    """Descomposición aditiva exacta: (intercepto, contribuciones n × 48). En clasificación, intercepto + suma de la fila
    es el logit del puntaje; en regresión, el pronóstico."""
    m = ebm(tarea)
    return m.intercepto, m.por_variable(X[predictores()])


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ============================================================================ simulador «¿y si…?»
def coherente(fila: dict) -> dict:
    """Recalcula las variables atadas tras un cambio (identidades exactas en el 100 % de los datos): la satisfacción con
    los productos y con las transacciones, el puntaje de satisfacción y el uso del cupo de quien no tiene tarjeta."""
    f = dict(fila)
    f["product_satisfaction"] = sum(bool(f[c]) for c in PRODUCTOS) / len(PRODUCTOS)
    f["tx_satisfaction"] = min(float(f["tx_count"]) / TOPE_TX_SATISFACCION, 1.0)
    f["satisfaction_score"] = int(np.round(0.5 * float(f["base_satisfaction"]) + 0.25 * f["tx_satisfaction"]
                                           + 0.25 * f["product_satisfaction"]))
    if not bool(f["credit_card"]):
        f["credit_utilization_ratio"] = 0.0
    elif float(f["credit_utilization_ratio"]) <= 0:
        f["credit_utilization_ratio"] = float(meta()["simulador"]["uso_cupo_mediano_con_tarjeta"])
    return f


def simular(fila_base, cambios: dict | None = None, atar: bool = True) -> pd.DataFrame:
    """Cliente hipotético: fila_base (una fila de clientes() o un diccionario) con «cambios» aplicados y, si atar=True, sus
    variables atadas recalculadas. Devuelve una tabla de 1 fila con los 48 predictores."""
    f = {k: fila_base[k] for k in predictores()}
    f.update(cambios or {})
    if atar:
        f = coherente(f)
    return pd.DataFrame([f])[predictores()]


# ============================================================================ explicación global
def forma(variable: str, tarea: str = "clasificacion") -> pd.DataFrame:
    return ebm(tarea).forma(variable)


def densidad(variable: str) -> pd.DataFrame:
    """Histograma de los clientes de entrenamiento de una variable numérica (desde, hasta, peso = número de clientes).
    Sale de la EBM de regresión, que se ajustó sin pesos; el de la EBM de clasificación cuenta a cada cliente con su peso
    de clase (la ponderación del balanceo) y no sirve para decir cuántos clientes hay."""
    return ebm("regresion").densidad(variable)


def efecto_categorias(variable: str, tarea: str = "clasificacion") -> pd.Series:
    return ebm(tarea).efecto_categorias(variable)


def es_categorica(variable: str) -> bool:
    return variable in ebm("clasificacion").categoricas


def importancia(tarea: str = "clasificacion") -> pd.Series:
    return ebm(tarea).importancia()


# ============================================================================ el orden de riesgo y la campaña
@lru_cache(maxsize=1)
def _puntajes_ordenados() -> np.ndarray:
    return np.sort(clientes()["p_ebm"].to_numpy(dtype=float))


def percentil(p_ebm: float) -> float:
    """Fracción de clientes de prueba con puntaje de la EBM menor o igual que p_ebm."""
    s = _puntajes_ordenados()
    return float(np.searchsorted(s, p_ebm, side="right") / len(s))


def puesto(p_ebm: float) -> int:
    """Puesto en el orden de riesgo entre los clientes de prueba: 1 es el de mayor puntaje."""
    s = _puntajes_ordenados()
    return int(len(s) - np.searchsorted(s, p_ebm, side="right") + 1)


def punto_operacion(s, fraccion: float = CAPACIDAD_INFORME, seed: int = SEMILLA) -> np.ndarray:
    """1 para la fracción de clientes de mayor puntaje (empates rotos al azar con la semilla del proyecto). Es la regla del
    informe (Resultados Entregable 3/generar.py::punto_operacion), copiada tal cual para reproducir sus cifras."""
    s = np.asarray(s, dtype=float)
    n_contactar = int(round(fraccion * len(s)))
    desempate = np.random.default_rng(seed).random(len(s))
    clase = np.zeros(len(s), dtype=int)
    clase[np.lexsort((desempate, -s))[:n_contactar]] = 1
    return clase


def resumen_capacidad(y, s, fraccion: float = CAPACIDAD_INFORME) -> dict:
    """Qué logra una campaña que contacta a la «fraccion» de clientes de mayor puntaje s, frente a la etiqueta y (1 = riesgo
    alto): matriz de confusión, captura (recall), precisión, elevación sobre el azar y el puntaje del último contactado."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    c = punto_operacion(s, fraccion)
    vp, fp = int(((c == 1) & (y == 1)).sum()), int(((c == 1) & (y == 0)).sum())
    fn, vn = int(((c == 0) & (y == 1)).sum()), int(((c == 0) & (y == 0)).sum())
    n, pos, cont = len(y), int(y.sum()), vp + fp
    captura = vp / pos if pos else float("nan")
    precision = vp / cont if cont else float("nan")
    return {"fraccion": float(fraccion), "n": n, "positivos": pos, "contactados": cont, "vp": vp, "fp": fp, "fn": fn, "vn": vn,
            "captura": captura, "precision": precision, "prevalencia": pos / n,
            "elevacion": (precision / (pos / n)) if cont and pos else float("nan"),
            "corte": float(s[c == 1].min()) if cont else float("nan")}


def curva_captura(y, s, puntos: int = 101) -> pd.DataFrame:
    """Captura acumulada: para cada capacidad (fracción contactada, de 0 a 1), la fracción de clientes de riesgo alto que
    quedan dentro, con el mismo desempate del punto de operación. Columnas: capacidad, contactados, vp, captura, precision."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    orden = np.lexsort((np.random.default_rng(SEMILLA).random(len(s)), -s))
    acum = np.cumsum(y[orden])
    caps = np.linspace(0, 1, puntos)
    k = np.round(caps * len(s)).astype(int)
    vp = np.where(k > 0, acum[np.maximum(k, 1) - 1], 0)
    precision = np.where(k > 0, vp / np.maximum(k, 1), np.nan)
    return pd.DataFrame({"capacidad": caps, "contactados": k, "vp": vp, "captura": vp / max(int(y.sum()), 1),
                         "precision": precision})


def curva_roc(y, s, puntos: int = 201) -> pd.DataFrame:
    """Curva ROC diezmada a «puntos» (fpr, tpr), suficiente para dibujar; el AUC exacto sale de auc()."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    orden = np.argsort(-s, kind="mergesort")
    ys, ss = y[orden], s[orden]
    distinto = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    acum = np.cumsum(ys)[distinto]
    tp = np.r_[0, acum]
    fp = np.r_[0, (distinto + 1) - acum]
    tpr, fpr = tp / max(int(ys.sum()), 1), fp / max(int((1 - ys).sum()), 1)
    rejilla = np.linspace(0, 1, puntos)
    return pd.DataFrame({"fpr": rejilla, "tpr": np.interp(rejilla, fpr, tpr)})


def auc(y, s) -> float:
    """Área bajo la curva ROC por rangos (con empates a medias), igual que roc_auc_score."""
    y = np.asarray(y).astype(int)
    r = pd.Series(np.asarray(s, dtype=float)).rank(method="average").to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
