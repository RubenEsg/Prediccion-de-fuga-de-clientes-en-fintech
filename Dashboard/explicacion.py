"""Explicación local (valores SHAP) de los modelos finales, por variable original.

Cada modelo final es una tubería de scikit-learn: un preprocesado (escalado de las 39 numéricas y codificación one-hot de las
9 categóricas: 78 columnas) y el estimador. Las explicaciones se dan por las 48 variables que recibe la tubería: el aporte
de una variable categórica es la suma de los aportes de sus columnas one-hot. Todas cumplen la propiedad de eficiencia de
los valores de Shapley: base + suma de los aportes = la salida del modelo en la escala que se descompone (se comprueba en
cada llamada). El método depende de la familia del modelo:

    lineal   regresión logística, SVM lineal, Lasso, Ridge y SVR lineal. Exacto: phi_j = sum_c w_c (z_c - media_fondo z_c),
             el valor SHAP intervencional de un modelo lineal con variables independientes (Lundberg y Lee, 2017).
    bayes    Naive Bayes gaussiano. Exacto: su log-odds es aditivo por columna, g_c(z) = log N(z; theta_1c, s_1c) -
             log N(z; theta_0c, s_0c), así que phi_j = sum_c [g_c(z_c) - media_fondo g_c].
    xgboost  XGBoost. TreeSHAP exacto (dependiente del camino) que trae la propia librería (pred_contribs=True).
    arbol    árbol de decisión y Random Forest, sobre la probabilidad de la hoja (clasificación) o la predicción
             (regresión). En vivo se estiman por permutaciones, como k-NN: la librería shap (con numba y llvmlite) no cabe
             en los 512 MB del servidor gratuito y su TreeExplainer copia los árboles en cada llamada. El resumen global
             que arma preparar_artefactos.py sí es TreeSHAP exacto (arbol_exacto=True, con shap instalado).
    vecinos  k-NN. No hay fórmula cerrada: se estima con permutaciones antitéticas (Strumbelj y Kononenko, 2014) contra
             un fondo de clientes de entrenamiento; la eficiencia se cumple igual (cada permutación reparte exactamente
             f(x) - media del fondo) y se informa el error de Monte Carlo de cada aporte.
    ebm      la EBM se explica con su descomposición aditiva exacta (motor.py), no con este módulo.
    bdl      el modelo original (Boosting con Desplazamiento Latente, src/original_model.py). Exacto, con el método del
             propio modelo (RegresorBDL.contribuciones): con árboles de un corte el índice F(x) es aditivo y el aporte de
             una columna es su curva menos la media de esa curva en las filas de ajuste; con árboles más profundos, cada
             árbol se reparte enumerando los subconjuntos de sus variables. En regresión se descompone el pronóstico
             (la media, F(x) + Σ π_k c_k); en clasificación, el índice F(x), que es lo único que depende del cliente: la
             probabilidad sale después de la mezcla (modelos_vivos.excedencia muestra esa conversión).

La escala que se descompone es la natural de cada modelo (ESCALA): el logit en la logística, XGBoost y Naive Bayes, el
margen en el SVM lineal, la probabilidad en el árbol, Random Forest y k-NN de clasificación, el índice del puntaje en el
modelo original de clasificación y la churn_probability predicha en regresión. Por eso los aportes de dos modelos distintos
no se comparan en magnitud, solo en orden y signo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LINEALES = {"LogisticRegression", "LinearSVC", "Lasso", "Ridge", "LinearSVR"}
ARBOLES = {"DecisionTreeClassifier", "DecisionTreeRegressor", "RandomForestClassifier", "RandomForestRegressor",
           "BosqueCompacto"}
XGB = {"XGBClassifier", "XGBRegressor"}
VECINOS = {"KNeighborsClassifier", "KNeighborsRegressor"}
BAYES = {"GaussianNB"}
BDL = {"RegresorBDL", "ClasificadorBDL"}             # el modelo original (src/original_model.py)
ESCALA_INDICE = "índice F(x) del puntaje, sin el escalón latente"

# Muestreo en vivo (permutaciones antitéticas, filas del fondo): k-NN es caro de evaluar; el árbol y Random Forest, no
MUESTREO = {"vecinos": (4, 3), "arbol": (6, 8)}
PERMUTACIONES, FONDO_VECINOS = MUESTREO["vecinos"]
TOLERANCIA = 1e-6            # eficiencia: base + suma de aportes frente a la salida


def metodo(est) -> str:
    """El método de explicación que corresponde a un estimador (ver el docstring del módulo)."""
    n = type(est).__name__
    if n in LINEALES:
        return "lineal"
    if n in BAYES:
        return "bayes"
    if n in XGB:
        return "xgboost"
    if n in ARBOLES:
        return "arbol"
    if n in VECINOS:
        return "vecinos"
    if "ExplainableBoosting" in n or n == "EBMConPesos":
        return "ebm"
    if n in BDL:
        return "bdl"
    raise ValueError(f"no hay método de explicación para {n}")


def escala(est, tarea: str) -> str:
    """Qué cantidad se descompone, en palabras (para el eje de la gráfica)."""
    if tarea == "regresion":
        return "churn_probability predicha"
    n = type(est).__name__
    if n == "ClasificadorBDL":
        return ESCALA_INDICE if getattr(est, "modo_", None) == "excedencia" else "logit del riesgo alto"
    if n == "LinearSVC":
        return "margen del SVM (distancia al hiperplano)"
    if n in {"LogisticRegression", "GaussianNB"} or n in XGB:
        return "logit del riesgo alto"
    if "ExplainableBoosting" in n:
        return "logit del riesgo alto"
    return "probabilidad de riesgo alto"


def salida(pipe, X: pd.DataFrame, tarea: str) -> np.ndarray:
    """La salida del modelo en la escala que se descompone (ESCALA), para cada fila de X."""
    pre, est = pipe.named_steps["preprocesado"], pipe.named_steps["modelo"]
    z = _denso(pre.transform(X))
    return _salida_z(est, z, tarea)


def _salida_z(est, z: np.ndarray, tarea: str) -> np.ndarray:
    n = type(est).__name__
    if tarea == "regresion":
        return np.asarray(est.predict(z), dtype=float)
    if n == "LinearSVC":
        return np.asarray(est.decision_function(z), dtype=float)
    if n == "LogisticRegression":
        return np.asarray(est.decision_function(z), dtype=float)
    if n == "GaussianNB":
        jll = est.predict_joint_log_proba(z) if hasattr(est, "predict_joint_log_proba") else est._joint_log_likelihood(z)
        return np.asarray(jll[:, 1] - jll[:, 0], dtype=float)
    if n in XGB:
        return np.asarray(est.get_booster().predict(_dmatriz(est, z), output_margin=True), dtype=float)
    if n == "ClasificadorBDL":          # el índice F(x) (modo excedencia) o el logit (modo directo): lo que suman sus árboles
        return np.asarray(regresor_bdl(est).indice(z) if est.modo_ == "excedencia" else est.decision_function(z), dtype=float)
    return np.asarray(est.predict_proba(z)[:, 1], dtype=float)


def _denso(z) -> np.ndarray:
    return np.asarray(z.toarray() if hasattr(z, "toarray") else z, dtype=float)


def _dmatriz(est, z):
    import xgboost
    return xgboost.DMatrix(z, feature_names=est.get_booster().feature_names)


def columnas_por_variable(pre) -> list[str]:
    """Para cada columna de salida del preprocesado, la variable original de la que sale."""
    out = []
    for nombre, tr, cols in pre.transformers_:
        if nombre == "remainder" or (isinstance(tr, str) and tr == "drop"):
            continue
        cols = [cols] if isinstance(cols, str) else list(cols)
        final = tr.steps[-1][1] if hasattr(tr, "steps") else tr
        if hasattr(final, "categories_"):
            caidas = getattr(final, "drop_idx_", None)
            for i, (v, cats) in enumerate(zip(cols, final.categories_)):
                n = len(cats) - (1 if caidas is not None and caidas[i] is not None else 0)
                out += [v] * n
        else:
            out += cols
    if len(out) != len(pre.get_feature_names_out()):
        raise ValueError("no se pudo asignar cada columna del preprocesado a su variable")
    return out


def _agrupar(phi_cols: np.ndarray, columnas: list[str], variables: list[str]) -> np.ndarray:
    """Suma los aportes de las columnas de cada variable: (n, columnas) -> (n, variables)."""
    a = np.zeros((len(columnas), len(variables)))
    pos = {v: j for j, v in enumerate(variables)}
    for i, v in enumerate(columnas):
        a[i, pos[v]] = 1.0
    return phi_cols @ a


def explicar(pipe, X: pd.DataFrame, fondo: pd.DataFrame, tarea: str, semilla: int = 42,
             permutaciones: int | None = None, n_fondo: int | None = None, arbol_exacto: bool = False) -> dict:
    """Valores SHAP de cada fila de X por variable original.

    Devuelve {"aportes": DataFrame (filas x variables), "base": array, "salida": array, "metodo": str, "escala": str,
    "error": DataFrame o None (error de Monte Carlo de los estimados), "exacto": bool}. Comprueba la eficiencia.
    `permutaciones` y `n_fondo` cambian el muestreo (por omisión, MUESTREO); `arbol_exacto` usa TreeSHAP de la librería
    shap en el árbol y Random Forest.
    """
    pre, est = pipe.named_steps["preprocesado"], pipe.named_steps["modelo"]
    variables = list(X.columns)
    cols = columnas_por_variable(pre)
    m = metodo(est)
    if m == "bdl":
        return explicar_bdl(pipe, X, tarea)
    z = _denso(pre.transform(X))
    error = None
    if m == "lineal":
        zf = _denso(pre.transform(fondo))
        w = np.asarray(est.coef_, dtype=float).ravel()
        phi = (z - zf.mean(axis=0)) * w
        base = np.full(len(X), float(np.mean(_salida_z(est, zf, tarea))))
    elif m == "bayes":
        zf = _denso(pre.transform(fondo))
        g, g_f = _bayes_terminos(est, z), _bayes_terminos(est, zf)
        phi = g - g_f.mean(axis=0)
        base = np.full(len(X), float(np.mean(_salida_z(est, zf, tarea))))
    elif m == "xgboost":
        contrib = est.get_booster().predict(_dmatriz(est, z), pred_contribs=True)
        phi, base = contrib[:, :-1], contrib[:, -1]
    elif m == "arbol" and arbol_exacto:
        import shap
        expl = shap.TreeExplainer(est)
        sv = expl.shap_values(z, check_additivity=False)
        ev = np.atleast_1d(expl.expected_value)
        if tarea == "clasificacion":
            sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
            ev = ev[1] if len(ev) > 1 else ev[0]
        else:
            ev = ev[0]
        phi, base = np.asarray(sv, dtype=float), np.full(len(X), float(ev))
    elif m in ("vecinos", "arbol"):
        por_omision = MUESTREO[m]
        aportes, base, error = _permutaciones(pipe, X, fondo, tarea, semilla, permutaciones or por_omision[0],
                                              n_fondo or por_omision[1])
        out = _resultado(aportes, base, salida(pipe, X, tarea), m, escala(est, tarea), error, exacto=False)
        return out
    else:
        raise ValueError("la EBM se explica con motor.py")
    aportes = pd.DataFrame(_agrupar(phi, cols, variables), columns=variables, index=X.index)
    return _resultado(aportes, base, _salida_z(est, z, tarea), m, escala(est, tarea), None, exacto=True)


def regresor_bdl(est):
    """El RegresorBDL que hay detrás de un estimador del modelo original: él mismo, o el que guarda un ClasificadorBDL en
    modo excedencia (None en modo directo, que no tiene mezcla)."""
    if type(est).__name__ == "RegresorBDL":
        return est
    return est.regresor_ if getattr(est, "modo_", None) == "excedencia" else None


def explicar_bdl(pipe, X: pd.DataFrame, tarea: str) -> dict:
    """Aportes exactos del modelo original por variable original, con el mismo formato que explicar().

    Regresión: base + suma de aportes = pronóstico (la media, F(x) + Σ π_k c_k). Clasificación en modo excedencia: base +
    suma de aportes = el índice F(x); la probabilidad es Σ π_k Φ((F(x) + c_k − τ)/σ) y no se reparte entre variables
    porque la mezcla no depende del cliente. Clasificación en modo directo (sin puntaje): se descompone el logit, con
    los valores de Shapley exactos de cada árbol."""
    pre, est = pipe.named_steps["preprocesado"], pipe.named_steps["modelo"]
    variables = list(X.columns)
    cols = columnas_por_variable(pre)
    z = _denso(pre.transform(X))
    reg = regresor_bdl(est)
    if reg is not None:
        phi, base = reg.contribuciones(z)
        if tarea == "clasificacion":
            base, y = float(base) - float(reg.desplazamiento_medio_), np.asarray(reg.indice(z), dtype=float)
        else:
            y = np.asarray(reg.predict(z), dtype=float)
    else:                                              # modo directo: los árboles suman el logit
        import importlib
        modulo = importlib.import_module(type(est).__module__)
        C = modulo.codificar(z, est.cortes_)
        phi = np.zeros(C.shape, dtype=float)
        base = float(est.intercepto_)
        for arbol in zip(est.arboles_variable_, est.arboles_corte_, est.arboles_valor_, est.arboles_cobertura_):
            base += modulo._aportes_arbol(*arbol, C, phi)
        y = np.asarray(est.decision_function(z), dtype=float)
    aportes = pd.DataFrame(_agrupar(np.asarray(phi, dtype=float), cols, variables), columns=variables, index=X.index)
    return _resultado(aportes, np.full(len(X), float(base)), y, "bdl", escala(est, tarea), None, exacto=True)


def _resultado(aportes, base, y, m, esc, error, exacto) -> dict:
    base = np.asarray(base, dtype=float)
    falta = float(np.max(np.abs(base + aportes.sum(axis=1).to_numpy() - y))) if len(y) else 0.0
    if falta > TOLERANCIA * max(1.0, float(np.max(np.abs(y)))):
        raise AssertionError(f"eficiencia: base + aportes difiere de la salida en {falta:.3g} ({m})")
    return {"aportes": aportes, "base": base, "salida": np.asarray(y, dtype=float), "metodo": m, "escala": esc,
            "error": error, "exacto": exacto, "eficiencia": falta}


def _bayes_terminos(est, z: np.ndarray) -> np.ndarray:
    """g_c(z) = log N(z; theta_1c, var_1c) - log N(z; theta_0c, var_0c), por columna (Naive Bayes gaussiano)."""
    th, var = np.asarray(est.theta_, dtype=float), np.asarray(est.var_, dtype=float)

    def logn(k):
        return -0.5 * np.log(2 * np.pi * var[k]) - 0.5 * (z - th[k]) ** 2 / var[k]
    return logn(1) - logn(0)


def _permutaciones(pipe, X, fondo, tarea, semilla, permutaciones, n_fondo):
    """Shapley por muestreo de permutaciones antitéticas contra un fondo (para k-NN). Una sola llamada al modelo por fila."""
    rng = np.random.default_rng(semilla)
    variables = list(X.columns)
    p = len(variables)
    fondo = fondo.sample(n=min(n_fondo, len(fondo)), random_state=semilla) if len(fondo) > n_fondo else fondo
    k = len(fondo)
    pares = max(1, permutaciones // 2)
    ordenes = []
    for _ in range(pares):
        o = rng.permutation(p)
        ordenes += [o, o[::-1]]
    aportes = np.zeros((len(X), p))
    error = np.zeros((len(X), p))
    base = np.zeros(len(X))
    fondo_v = fondo[variables].reset_index(drop=True)
    n_ord = len(ordenes)
    # rango[t, j] = posición de la variable j en la permutación t; en el paso s ya se reemplazaron las de rango < s
    rango = np.empty((n_ord, p), dtype=int)
    for t, o in enumerate(ordenes):
        rango[t, o] = np.arange(p)
    paso = np.arange(p + 1)
    reemplazada = (rango[:, None, :] < paso[None, :, None])               # (n_ord, p + 1, p)
    for r in range(len(X)):
        fila = X.iloc[r]
        columnas = {}
        for j, v in enumerate(variables):
            de_x = np.repeat(reemplazada[:, :, j][:, :, None], k, axis=2).ravel()   # (n_ord * (p + 1) * k,)
            base_v = np.tile(fondo_v[v].to_numpy(), n_ord * (p + 1))
            col = np.where(de_x, fila[v], base_v)
            columnas[v] = pd.Series(col).astype(X[v].dtype)                # tipos de las categóricas y booleanas
        todo = pd.DataFrame(columnas)
        y = salida(pipe, todo, tarea).reshape(n_ord, p + 1, k).mean(axis=2)
        marg = np.zeros((n_ord, p))
        for t, o in enumerate(ordenes):
            marg[t, o] = np.diff(y[t])
        aportes[r] = marg.mean(axis=0)
        pares_m = marg.reshape(pares, 2, p).mean(axis=1)    # las dos de cada par antitético cuentan como una muestra
        error[r] = pares_m.std(axis=0, ddof=1) / np.sqrt(pares) if pares > 1 else np.nan
        base[r] = y[0, 0]
    return (pd.DataFrame(aportes, columns=variables, index=X.index), base,
            pd.DataFrame(error, columns=variables, index=X.index))


def vecinos(pipe, X: pd.DataFrame) -> dict:
    """Los vecinos de una fila en el entrenamiento (posiciones en el orden de entrenamiento), sus distancias y su
    etiqueta: la predicción de k-NN es el promedio de esas etiquetas, ponderado por 1/distancia si weights='distance'."""
    pre, est = pipe.named_steps["preprocesado"], pipe.named_steps["modelo"]
    z = _denso(pre.transform(X))
    dist, idx = est.kneighbors(z, n_neighbors=est.n_neighbors)
    y = np.asarray(est._y)
    etiquetas = est.classes_[y[idx[0]]] if hasattr(est, "classes_") else y[idx[0]]
    return {"posiciones": idx[0], "distancias": dist[0], "etiquetas": np.asarray(etiquetas, dtype=float),
            "k": int(est.n_neighbors), "ponderacion": est.weights, "metrica": f"Minkowski p = {est.p}"}
