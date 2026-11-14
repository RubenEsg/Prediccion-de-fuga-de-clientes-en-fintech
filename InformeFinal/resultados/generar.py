"""Sección de resultados del proyecto (Entregables 1 a 3): un único guion reproduce cada tabla, cada prueba
estadística y cada figura a partir de los resultados guardados del Entregable 3 (carpeta ../entrega 3, que
solo se lee), siguiendo la guía del curso «Cómo escribir la sección de resultados».

    python generar.py              calcula todo (pruebas, tablas, datos intermedios y figuras; unos minutos)
    python generar.py --figuras    solo vuelve a dibujar las figuras a partir de datos/*.csv

Insumos (solo lectura): resultados/experimentos.parquet (160 corridas con sus 5 puntajes externos),
resultados/predicciones_prueba.parquet (puntajes y clases de cada corrida sobre los 9.745 clientes de prueba),
resultados/complementos (calibración, semillas), resultados/computo (fidelidad, escalamiento, tiempos),
resultados/comparacion_entrega2 (contrastes con la Entrega 2 y sus modelos) y resultados/modelos (pipelines
finales). Lo único que se vuelve a ajustar aquí, y se marca como exploratorio en el texto, es la curva de
aprendizaje de la EBM, la EBM sin interacciones de la comparación con la Entrega 2 y la importancia por
permutación; todo lo demás se deriva de lo que ya se entregó.

Salidas: tablas/*.tex (booktabs), datos/*.csv (toda cifra del texto) y figures/*.pdf (vectoriales, en inglés,
12 pt, ancho máximo 6,3 in).
"""
from __future__ import annotations

import os
import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent


def carpeta_entrega3() -> Path:
    """Carpeta del Entregable 3: ../entrega 3 (carpeta de trabajo), ../../Entrega3 (repositorio publicado) o la
    variable de entorno ENTREGA3; debe contener resultados/experimentos.parquet."""
    candidatas = (os.environ.get("ENTREGA3"), AQUI.parent / "entrega 3", AQUI.parent.parent / "Entrega3")
    for candidata in candidatas:
        if candidata and (Path(candidata) / "resultados" / "experimentos.parquet").exists():
            return Path(candidata)
    sys.exit("No se encontró la carpeta del Entregable 3 (../entrega 3 o ../../Entrega3); fije la variable ENTREGA3")


E3 = carpeta_entrega3()
RES = E3 / "resultados"
sys.path.insert(0, str(E3))                      # paquete src del Entregable 3 (solo lectura)
TABLAS, FIGURAS, DATOS = AQUI / "tablas", AQUI / "figures", AQUI / "datos"
SEMILLA = 42
ALFA = 0.05
ROPE = {"auc": 0.01, "rmse": 0.0005}             # regiones de equivalencia práctica, fijadas antes de calcular
N_BOOT = 2000
K_EXT = 5
RAZON = 0.25                                     # n_prueba / n_entrenamiento de cada pliegue externo (1/4)
NOMBRES = {"ebm": "EBM (propuesto)", "xgboost": "XGBoost", "random_forest": "Random Forest",
           "arbol": "Árbol de decisión", "knn": "k-NN", "logistica": "Regresión logística",
           "svm": "SVM lineal", "naive_bayes": "Naive Bayes", "ridge": "Ridge", "lasso": "Lasso",
           "svr": "SVR lineal"}
CORTOS_EN = {"xgboost": "XGBoost", "random_forest": "Random Forest", "arbol": "Decision tree", "knn": "k-NN",
             "logistica": "Logistic", "svm": "Linear SVM", "naive_bayes": "Naive Bayes", "ridge": "Ridge",
             "lasso": "Lasso", "svr": "Linear SVR"}
NOMBRES_EN = {"ebm": "EBM (proposed)", "xgboost": "XGBoost", "random_forest": "Random Forest",
              "arbol": "Decision tree", "knn": "k-NN", "logistica": "Logistic regression",
              "svm": "Linear SVM", "naive_bayes": "Naive Bayes", "ridge": "Ridge", "lasso": "Lasso",
              "svr": "Linear SVR"}
BALANCEO = {"ninguno": "sin balanceo", "class_weight": "ponderación", "smote": "SMOTE", "adasyn": "ADASYN"}
OPTIM = {"grid": "Grid", "random": "Random", "bayesiana": "TPE", "genetica": "Genética"}
SENAL = ["app_logins_frequency", "age", "satisfaction_score", "base_satisfaction"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ utilidades estadísticas
def t_corregida(a, b, razon: float = RAZON) -> dict:
    """t pareada con varianza corregida de Nadeau y Bengio (2003) sobre J puntajes por pliegue.

    ``d = a − b``; ``t = media(d) / sqrt((1/J + razon) · var(d))`` con J − 1 grados de libertad.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    j = d.size
    var = d.var(ddof=1)
    if var == 0:
        return {"t": 0.0, "p": 1.0, "media": float(d.mean()), "desv": 0.0, "j": j}
    t = d.mean() / np.sqrt((1.0 / j + razon) * var)
    return {"t": float(t), "p": float(2 * stats.t.sf(abs(t), df=j - 1)), "media": float(d.mean()),
            "desv": float(d.std(ddof=1)), "j": j}


def bayes_rope(a, b, rope: float, razon: float = RAZON) -> dict:
    """t correlacionada bayesiana (Corani y Benavoli, 2015; Benavoli y otros, 2017) con región de equivalencia.

    La posterior de la diferencia media es una t de Student con J − 1 grados de libertad, centro ``media(d)`` y
    escala ``sqrt((1/J + razon) · var(d))``; devuelve P(d < −rope), P(|d| ≤ rope) y P(d > rope).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    j = d.size
    var = d.var(ddof=1)
    if var == 0:
        m = float(d.mean())
        return {"p_peor": float(m < -rope), "p_rope": float(abs(m) <= rope), "p_mejor": float(m > rope)}
    escala = np.sqrt((1.0 / j + razon) * var)
    dist = stats.t(df=j - 1, loc=d.mean(), scale=escala)
    p_peor = float(dist.cdf(-rope))
    p_mejor = float(dist.sf(rope))
    return {"p_peor": p_peor, "p_rope": float(1 - p_peor - p_mejor), "p_mejor": p_mejor}


def mcnemar_exacto(y, clase_a, clase_b) -> dict:
    """McNemar exacta (binomial) sobre los casos discordantes de dos clasificadores en las mismas filas."""
    from statsmodels.stats.contingency_tables import mcnemar
    y = np.asarray(y)
    ok_a, ok_b = np.asarray(clase_a) == y, np.asarray(clase_b) == y
    n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    tabla = [[int((ok_a & ok_b).sum()), n01], [n10, int((~ok_a & ~ok_b).sum())]]
    r = mcnemar(tabla, exact=True)
    return {"p": float(r.pvalue), "a_bien_b_mal": n01, "a_mal_b_bien": n10,
            "error_a": float(1 - ok_a.mean()), "error_b": float(1 - ok_b.mean())}


def punto_operacion(s, fraccion: float = 0.25, seed: int = SEMILLA) -> np.ndarray:
    """Clase 1 para la fracción de clientes de mayor puntaje (empates rotos al azar), como en el Entregable 3."""
    s = np.asarray(s, dtype=float)
    n_contactar = int(round(fraccion * len(s)))
    desempate = np.random.default_rng(seed).random(len(s))
    clase = np.zeros(len(s), dtype=int)
    clase[np.lexsort((desempate, -s))[:n_contactar]] = 1
    return clase


def mape(y, pred) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred) / np.abs(y)))


# ============================================================================ formato LaTeX
def num(x, d: int = 4, signo: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = str(Decimal(repr(float(abs(x)))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace(".", "{,}")
    if round(abs(x), d) == 0:
        return f"${s}$"
    if x < 0:
        return f"$-{s}$"
    return f"$+{s}$" if signo else f"${s}$"


def ent(x) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def pval(p) -> str:
    if p is None or np.isnan(p):
        return "--"
    if p == 0:
        return "$<0{,}001$"
    if p < 0.001:
        mant, exp = f"{p:.1e}".split("e")
        return f"${mant.replace('.', '{,}')} \\times 10^{{{int(exp)}}}$"
    return "$" + str(Decimal(repr(float(p))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)).replace(".", "{,}") + "$"


def intervalo(inf, sup, d: int = 4) -> str:
    return f"[{num(inf, d)}; {num(sup, d)}]"


def mas_menos(m, s, d: int = 4) -> str:
    return f"{num(m, d)} $\\pm$ {num(s, d)}"


def marcar(valores: dict, mayor_es_mejor: bool = True, d: int = 4) -> dict:
    """Mejor valor en negrita y segundo subrayado; los empates al redondeo mostrado comparten la marca."""
    limpio = {k: round(float(v), d) for k, v in valores.items() if v is not None and not np.isnan(v)}
    distintos = sorted(set(limpio.values()), reverse=mayor_es_mejor)
    marcas = {}
    for k, v in limpio.items():
        if v == distintos[0]:
            marcas[k] = "negrita"
        elif len(distintos) > 1 and v == distintos[1]:
            marcas[k] = "subrayado"
    return marcas


def decorar(texto: str, marca: str | None) -> str:
    if marca == "negrita":
        return f"{{\\boldmath\\textbf{{{texto}}}}}"
    if marca == "subrayado":
        return f"\\underline{{{texto}}}"
    return texto


def escribir_tabla(nombre: str, caption: str, label: str, columnas: str, cabecera: list[str],
                   filas: list[list[str]], notas: str | None = None, tamano: str = "\\scriptsize",
                   colsep: str = "3pt", lineas_medias: list[int] | None = None) -> None:
    TABLAS.mkdir(exist_ok=True)
    out = ["\\begin{table}[tbp]", "  \\centering" + tamano + f"\\setlength{{\\tabcolsep}}{{{colsep}}}",
           f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", f"  \\begin{{tabular}}{{{columnas}}}",
           "    \\toprule", "    " + " & ".join(cabecera) + " \\\\", "    \\midrule"]
    for i, f in enumerate(filas):
        if lineas_medias and i in lineas_medias:
            out.append("    \\midrule")
        out.append("    " + " & ".join(f) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}"]
    if notas:
        out.append(f"  \\par\\smallskip\\begin{{minipage}}{{0.97\\linewidth}}\\scriptsize {notas}\\end{{minipage}}")
    out.append("\\end{table}\n")
    (TABLAS / nombre).write_text("\n".join(out), encoding="utf-8")
    log(f"tabla {nombre}")


# ============================================================================ carga de insumos
def cargar_basicos() -> dict:
    from src import analisis
    tabla = analisis.cargar_tabla()
    for col in ("puntajes_ext", "metricas_ext", "hiperparametros"):
        tabla[col] = tabla[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
    pred = pd.read_parquet(RES / "predicciones_prueba.parquet")
    df = pd.read_csv(E3 / "datos" / "customer_data.csv")
    from sklearn.model_selection import train_test_split
    idx_tr, idx_te = train_test_split(df.index, test_size=0.20, random_state=SEMILLA)
    assert np.array_equal(np.sort(idx_te), pred.index.values), "la prueba no coincide con las predicciones"
    s_tr = df.loc[idx_tr, "churn_probability"]
    tau = float(s_tr.quantile(0.75))
    y_cont = df.loc[pred.index, "churn_probability"]
    y_bin = (y_cont > tau).astype(int)
    mejores = {t: analisis.mejores_por_modelo(tabla, t) for t in ("clasificacion", "regresion")}
    return {"tabla": tabla, "pred": pred, "df": df, "idx_tr": idx_tr, "y_bin": y_bin, "y_cont": y_cont,
            "tau": tau, "mejores": mejores, "analisis": analisis}


# ============================================================================ clasificación
def clasificacion(b: dict) -> dict:
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)
    from src import estadistica as est
    y = b["y_bin"].to_numpy()
    pred = b["pred"]
    mej = b["mejores"]["clasificacion"].set_index("modelo")
    cal = pd.read_csv(RES / "complementos" / "calibracion.csv")
    modelos = list(mej.index)
    filas, pliegues, puntajes, clases_op = [], {}, {}, {}
    for m in modelos:
        r = mej.loc[m]
        s = pred[r["clave"]].to_numpy()
        puntajes[m] = s
        pliegues[m] = np.asarray(r["puntajes_ext"], dtype=float)
        c_op = punto_operacion(s)
        clases_op[m] = c_op
        c_05 = pred[r["clave"] + "__clase"].to_numpy().astype(int)      # clases con umbral 0,5 guardadas por el Entregable 3
        boot = est.bootstrap_auc(y, s, n_boot=N_BOOT)
        fila_cal = cal[(cal["modelo"] == m) & (cal["balanceo"] == r["balanceo"]) & (cal["optimizador"] == r["optimizador"])
                       & (cal["version"] == "isotónica")]
        ece_iso, auc_iso = fila_cal["ece"], fila_cal["auc"]
        met = r["metricas_ext"]
        filas.append({
            "modelo": m, "balanceo": r["balanceo"], "optimizador": r["optimizador"],
            "auc_ext_media": float(np.mean(pliegues[m])), "auc_ext_desv": float(np.std(pliegues[m], ddof=1)),
            "f1_ext_media": float(np.mean(met["f1"])), "f1_ext_desv": float(np.std(met["f1"], ddof=1)),
            "auc_prueba": float(roc_auc_score(y, s)), "auc_bca_inf": boot["ic_bca"][0], "auc_bca_sup": boot["ic_bca"][1],
            "auc_pr": float(average_precision_score(y, s)),
            "recall_op": float(recall_score(y, c_op)), "precision_op": float(precision_score(y, c_op)),
            "mcc_op": float(matthews_corrcoef(y, c_op)), "exactitud_bal_op": float(balanced_accuracy_score(y, c_op)),
            "f1_op": float(f1_score(y, c_op)),
            "vp_op": int(((y == 1) & (c_op == 1)).sum()), "fp_op": int(((y == 0) & (c_op == 1)).sum()),
            "fn_op": int(((y == 1) & (c_op == 0)).sum()), "vn_op": int(((y == 0) & (c_op == 0)).sum()),
            "vp_05": int(((y == 1) & (c_05 == 1)).sum()), "fp_05": int(((y == 0) & (c_05 == 1)).sum()),
            "fn_05": int(((y == 1) & (c_05 == 0)).sum()), "vn_05": int(((y == 0) & (c_05 == 0)).sum()),
            "recall_10": float(recall_score(y, punto_operacion(s, 0.10))), "recall_50": float(recall_score(y, punto_operacion(s, 0.50))),
            "precision_10": float(precision_score(y, punto_operacion(s, 0.10))),
            "ece_isotonica": float(ece_iso.iloc[0]) if len(ece_iso) else np.nan,
            "auc_isotonica": float(auc_iso.iloc[0]) if len(auc_iso) else np.nan,
            "corrida_min": float(r["tiempo_total_s"]) / 60,
        })
    tabla = pd.DataFrame(filas).set_index("modelo")
    log("clasificación: métricas de prueba listas")

    # ---- pruebas EBM frente a cada modelo de referencia
    bases = [m for m in modelos if m != "ebm"]
    pruebas = []
    for m in bases:
        tc = t_corregida(pliegues["ebm"], pliegues[m])
        dl = est.delong(y, puntajes[m], puntajes["ebm"])
        bo = est.bootstrap_auc(y, puntajes[m], puntajes["ebm"], n_boot=N_BOOT)
        mc = mcnemar_exacto(y, clases_op[m], clases_op["ebm"])
        by = bayes_rope(pliegues["ebm"], pliegues[m], ROPE["auc"])
        cl = est.delta_cliff(pliegues["ebm"], pliegues[m])
        d = pliegues["ebm"] - pliegues[m]
        pruebas.append({"modelo": m, "delta_auc": dl["diferencia"], "delta_bca_inf": bo["ic_bca"][0],
                        "delta_bca_sup": bo["ic_bca"][1], "t_corr": tc["t"], "p_t": tc["p"],
                        "z_delong": dl["z"], "p_delong": dl["p_bilateral"], "corr_auc": dl["correlacion"],
                        "p_mcnemar": mc["p"], "err_base_op": mc["error_a"], "err_ebm_op": mc["error_b"],
                        "disc_base_bien": mc["a_bien_b_mal"], "disc_ebm_bien": mc["a_mal_b_bien"],
                        "p_peor": by["p_peor"], "p_rope": by["p_rope"], "p_mejor": by["p_mejor"],
                        "d_z_pliegues": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan,
                        "delta_cliff": cl["delta"]})
    pr = pd.DataFrame(pruebas).set_index("modelo")
    for col in ("p_t", "p_delong", "p_mcnemar"):
        pr[col + "_holm"] = est.ajustar_pvalores(pr[col].to_numpy(), "holm")
    log("clasificación: pruebas listas")
    return {"tabla": tabla, "pruebas": pr, "pliegues": pliegues, "puntajes": puntajes, "clases_op": clases_op, "y": y}


# ============================================================================ regresión
def regresion(b: dict) -> dict:
    from src import estadistica as est
    y = b["y_cont"].to_numpy()
    pred = b["pred"]
    mej = b["mejores"]["regresion"].set_index("modelo")
    modelos = list(mej.index)
    filas, pliegues, pron = [], {}, {}
    for m in modelos:
        r = mej.loc[m]
        p = pred[r["clave"]].to_numpy(dtype=float)
        pron[m] = p
        pliegues[m] = -np.asarray(r["puntajes_ext"], dtype=float)         # se guardan como −RMSE
        met = r["metricas_ext"]
        bo = est.bootstrap_regresion(y, p, n_boot=N_BOOT)
        filas.append({"modelo": m, "optimizador": r["optimizador"],
                      "rmse_ext_media": float(np.mean(pliegues[m])), "rmse_ext_desv": float(np.std(pliegues[m], ddof=1)),
                      "r2_ext_media": float(np.mean(met["r2"])), "r2_ext_desv": float(np.std(met["r2"], ddof=1)),
                      "rmse_prueba": bo["rmse"]["estimacion"], "rmse_bca_inf": bo["rmse"]["ic_bca"][0],
                      "rmse_bca_sup": bo["rmse"]["ic_bca"][1], "mae_prueba": bo["mae"]["estimacion"],
                      "mape_prueba": mape(y, p), "r2_prueba": bo["r2"]["estimacion"],
                      "r2_bca_inf": bo["r2"]["ic_bca"][0], "r2_bca_sup": bo["r2"]["ic_bca"][1],
                      "corrida_min": float(r["tiempo_total_s"]) / 60})
    tabla = pd.DataFrame(filas).set_index("modelo")
    perdidas = pd.DataFrame({m: (y - pron[m]) ** 2 for m in modelos})
    mcs = est.conjunto_confianza_modelos(perdidas, alfa=0.10, n_boot=N_BOOT)
    tabla["p_mcs"] = mcs["p_mcs"].reindex(tabla.index)
    tabla["en_mcs"] = tabla["p_mcs"] > 0.10
    log(f"regresión: MCS listo (bloque {mcs['bloque']}, incluidos {mcs['incluidos']})")

    bases = [m for m in modelos if m != "ebm"]
    pruebas = []
    for m in bases:
        tc = t_corregida(pliegues[m], pliegues["ebm"])                      # positivo favorece a la EBM
        dm = est.diebold_mariano(perdidas[m].to_numpy(), perdidas["ebm"].to_numpy())
        bo = est.bootstrap_regresion(y, pron[m], pron["ebm"], n_boot=N_BOOT)
        by = bayes_rope(pliegues[m], pliegues["ebm"], ROPE["rmse"])
        cl = est.delta_cliff(pliegues[m], pliegues["ebm"])                  # positivo: la referencia pierde más
        d = pliegues[m] - pliegues["ebm"]
        pruebas.append({"modelo": m, "delta_cliff": cl["delta"], "delta_rmse": bo["rmse"]["estimacion"], "delta_bca_inf": bo["rmse"]["ic_bca"][0],
                        "delta_bca_sup": bo["rmse"]["ic_bca"][1], "delta_r2": bo["r2"]["estimacion"],
                        "t_corr": tc["t"], "p_t": tc["p"], "dm_hln": dm["estadistico"], "p_dm": dm["p_bilateral"],
                        "p_peor": by["p_peor"], "p_rope": by["p_rope"], "p_mejor": by["p_mejor"],
                        "d_z_pliegues": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan,
                        "d_z_clientes": est.d_cohen(perdidas[m].to_numpy() - perdidas["ebm"].to_numpy())["d"]})
    pr = pd.DataFrame(pruebas).set_index("modelo")
    for col in ("p_t", "p_dm"):
        pr[col + "_holm"] = est.ajustar_pvalores(pr[col].to_numpy(), "holm")
    log("regresión: pruebas listas")
    return {"tabla": tabla, "pruebas": pr, "pliegues": pliegues, "pron": pron, "y": y, "mcs": mcs}


# ============================================================================ ablación
def ablacion(b: dict, clf: dict, reg: dict, pesado: dict) -> pd.DataFrame:
    from src import estadistica as est
    tabla = b["tabla"]
    pred = b["pred"]
    y = clf["y"]
    filas = []

    mp = pd.read_csv(RES / "metricas_prueba.csv")

    def recall_05(fila_tabla) -> float:
        f = mp[(mp["tarea"] == fila_tabla["tarea"]) & (mp["modelo"] == "ebm") & (mp["balanceo"] == fila_tabla["balanceo"])
               & (mp["optimizador"] == fila_tabla["optimizador"])]
        return float(f["recall"].iloc[0]) if len(f) else np.nan

    def fila_clf(nombre, componente, hipotesis, fila_tabla, completo):
        s = pred[fila_tabla["clave"]].to_numpy()
        pl = np.asarray(fila_tabla["puntajes_ext"], dtype=float)
        tc = t_corregida(completo["pliegues"], pl)
        dl = est.delong(y, s, completo["puntaje"])
        bo = est.bootstrap_auc(y, s, completo["puntaje"], n_boot=N_BOOT)
        by = bayes_rope(completo["pliegues"], pl, ROPE["auc"])
        return {"tarea": "clasificacion", "componente": componente, "variante": nombre, "hipotesis": hipotesis,
                "metrica_ext_media": float(pl.mean()), "metrica_ext_desv": float(pl.std(ddof=1)),
                "metrica_prueba": dl["auc_a"], "delta_completo": dl["diferencia"], "delta_bca_inf": bo["ic_bca"][0],
                "delta_bca_sup": bo["ic_bca"][1], "t_corr": tc["t"], "p_t": tc["p"], "p_prueba": dl["p_bilateral"],
                "estadistico_prueba": dl["z"], "p_peor": by["p_peor"], "p_rope": by["p_rope"], "p_mejor": by["p_mejor"],
                "recall_05": recall_05(fila_tabla), "corrida_min": float(fila_tabla["tiempo_total_s"]) / 60}

    ok = tabla[(tabla["estado"] == "ok") & (tabla["modelo"] == "ebm")]
    c = ok[(ok["tarea"] == "clasificacion") & (ok["balanceo"] == "class_weight") & (ok["optimizador"] == "grid")].iloc[0]
    completo = {"pliegues": np.asarray(c["puntajes_ext"], dtype=float), "puntaje": pred[c["clave"]].to_numpy()}
    filas.append({"tarea": "clasificacion", "componente": "completo", "variante": "EBM completa (ponderación, Grid, $K=0$)",
                  "hipotesis": "", "metrica_ext_media": float(completo["pliegues"].mean()),
                  "metrica_ext_desv": float(completo["pliegues"].std(ddof=1)),
                  "metrica_prueba": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, completo["puntaje"])),
                  "recall_05": recall_05(c), "corrida_min": float(c["tiempo_total_s"]) / 60})
    for bal in ("ninguno", "smote", "adasyn"):
        f = ok[(ok["tarea"] == "clasificacion") & (ok["balanceo"] == bal) & (ok["optimizador"] == "grid")].iloc[0]
        filas.append(fila_clf(f"balanceo: {BALANCEO[bal]} (Grid)", "balanceo",
                              "el balanceo no mejora el AUC", f, completo))
    for opt in ("random", "bayesiana", "genetica"):
        f = ok[(ok["tarea"] == "clasificacion") & (ok["balanceo"] == "class_weight") & (ok["optimizador"] == opt)].iloc[0]
        filas.append(fila_clf(f"optimizador: {OPTIM[opt]} (ponderación)", "optimizador",
                              "la métrica final no distingue a los optimizadores", f, completo))
    # interacciones: K = 0 frente a la EBM con pares, en la partición y etiqueta de la Entrega 2 (reajuste)
    g = pesado["gam_e2"]
    filas.append({"tarea": "clasificacion (etiqueta de la mediana)", "componente": "interacciones",
                  "variante": "sin interacciones ($K=0$) frente a la EBM con 5 pares (TPE)",
                  "hipotesis": "la hipótesis aditiva basta", "metrica_ext_media": np.nan, "metrica_ext_desv": np.nan,
                  "metrica_prueba": g["auc_gam"], "delta_completo": g["auc_ebm"] - g["auc_gam"],
                  "delta_bca_inf": g["ic_bca"][0], "delta_bca_sup": g["ic_bca"][1], "t_corr": np.nan, "p_t": np.nan,
                  "p_prueba": g["p_delong"], "estadistico_prueba": g["z"], "p_peor": np.nan, "p_rope": np.nan,
                  "p_mejor": np.nan, "corrida_min": np.nan})
    # regresión: optimizadores
    cr = ok[(ok["tarea"] == "regresion") & (ok["optimizador"] == "grid")].iloc[0]
    plr = -np.asarray(cr["puntajes_ext"], dtype=float)
    pr_c = pred[cr["clave"]].to_numpy(dtype=float)
    yc = reg["y"]
    filas.append({"tarea": "regresion", "componente": "completo", "variante": "EBM completa (Grid, $K=0$)", "hipotesis": "",
                  "metrica_ext_media": float(plr.mean()), "metrica_ext_desv": float(plr.std(ddof=1)),
                  "metrica_prueba": float(np.sqrt(np.mean((yc - pr_c) ** 2))), "corrida_min": float(cr["tiempo_total_s"]) / 60})
    for opt in ("random", "bayesiana", "genetica"):
        f = ok[(ok["tarea"] == "regresion") & (ok["optimizador"] == opt)].iloc[0]
        pl = -np.asarray(f["puntajes_ext"], dtype=float)
        p = pred[f["clave"]].to_numpy(dtype=float)
        tc = t_corregida(pl, plr)                                            # positivo: la variante pierde más
        dm = est.diebold_mariano((yc - p) ** 2, (yc - pr_c) ** 2)
        bo = est.bootstrap_regresion(yc, p, pr_c, n_boot=N_BOOT)
        by = bayes_rope(pl, plr, ROPE["rmse"])
        filas.append({"tarea": "regresion", "componente": "optimizador", "variante": f"optimizador: {OPTIM[opt]}",
                      "hipotesis": "la métrica final no distingue a los optimizadores",
                      "metrica_ext_media": float(pl.mean()), "metrica_ext_desv": float(pl.std(ddof=1)),
                      "metrica_prueba": float(np.sqrt(np.mean((yc - p) ** 2))),
                      "delta_completo": bo["rmse"]["estimacion"], "delta_bca_inf": bo["rmse"]["ic_bca"][0],
                      "delta_bca_sup": bo["rmse"]["ic_bca"][1], "t_corr": tc["t"], "p_t": tc["p"],
                      "p_prueba": dm["p_bilateral"], "estadistico_prueba": dm["estadistico"],
                      "p_peor": by["p_peor"], "p_rope": by["p_rope"], "p_mejor": by["p_mejor"],
                      "corrida_min": float(f["tiempo_total_s"]) / 60})
    ab = pd.DataFrame(filas)
    for tarea in ("clasificacion", "regresion"):
        sel = (ab["tarea"] == tarea) & (ab["componente"] != "completo")
        ab.loc[sel, "p_t_holm"] = est.ajustar_pvalores(ab.loc[sel, "p_t"].to_numpy(), "holm")
        ab.loc[sel, "p_prueba_holm"] = est.ajustar_pvalores(ab.loc[sel, "p_prueba"].to_numpy(), "holm")
    log("ablación lista")
    return ab


# ============================================================================ cálculos que ajustan modelos
def calculos_pesados(b: dict) -> dict:
    """Curva de aprendizaje de la EBM, EBM sin interacciones (Entrega 2), importancia por permutación,
    parámetros y memoria de cada modelo final. Lee los pipelines guardados; no escribe en entrega 3."""
    import joblib
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from src import datos
    from src import estadistica as est
    from src.preprocesamiento import nombres_variables

    log("cargando los datos con el ETL del Entregable 3 (solo lectura)...")
    p = datos.preparar_todo()
    c = p["conjuntos"]
    pred = b["pred"]
    X_te = c.X_test.loc[pred.index]
    y_te = b["y_bin"].loc[pred.index].to_numpy()
    y_tr = p["y_train"].loc[c.X_train.index].to_numpy()
    assert abs(p["y_test"].loc[pred.index].mean() - y_te.mean()) < 1e-12
    salida = {}

    # ---- EBM final de clasificación: parámetros, importancias, funciones de forma, permutación, curva de aprendizaje
    mej_c = b["mejores"]["clasificacion"].set_index("modelo")
    ruta = mej_c.loc["ebm", "ruta_modelo"]
    pipe = joblib.load(RES / "modelos" / Path(ruta).name)
    pre, modelo = pipe.named_steps["preprocesado"], pipe.named_steps["modelo"]
    ebm = getattr(modelo, "ebm_", modelo)
    nv = nombres_variables(pre)
    imp = pd.Series(ebm.term_importances(), index=[" × ".join(nv[j] for j in f) for f in ebm.term_features_])
    salida["importancia_terminos"] = (imp / imp.sum()).sort_values(ascending=False)
    formas = {}
    for v in SENAL:
        formas[v] = b["analisis"].forma_ebm(pipe, v)
    salida["formas"] = formas
    log("importancia por permutación de la EBM sobre la prueba (48 predictores × 5 repeticiones)...")
    pi = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=SEMILLA, n_jobs=1)
    salida["permutacion"] = pd.DataFrame({"media": pi.importances_mean, "desv": pi.importances_std},
                                         index=X_te.columns).sort_values("media", ascending=False)

    log("curva de aprendizaje de la EBM (5 tamaños, modelo final con B = 8)...")
    Xtr, Xte = pre.transform(c.X_train), pre.transform(X_te)
    orden = np.random.default_rng(SEMILLA).permutation(len(y_tr))
    curva = []
    for n in (2500, 5000, 10000, 20000, len(y_tr)):
        idx = orden[:n]
        t0 = time.perf_counter()
        m = clone(modelo).fit(Xtr[idx], y_tr[idx])
        t_fit = time.perf_counter() - t0
        curva.append({"modelo": "ebm", "n": n, "auc": float(roc_auc_score(y_te, m.predict_proba(Xte)[:, 1])),
                      "entrenamiento_s": t_fit})
        log(f"   n = {n}: AUC {curva[-1]['auc']:.4f} ({t_fit:.1f} s)")
    salida["curva_ebm"] = pd.DataFrame(curva)
    try:
        from memory_profiler import memory_usage
        incrementos = []
        for _ in range(3):                                                  # mediana de tres ajustes: la medida varía
            base = memory_usage(-1, interval=0.05, timeout=0.2)
            pico = memory_usage((lambda: clone(modelo).fit(Xtr, y_tr), (), {}), interval=0.05, max_usage=True)
            pico = max(pico) if isinstance(pico, (list, tuple)) else pico
            incrementos.append(float(pico - min(base)))
        salida["memoria_ebm_mb"] = float(np.median(incrementos))
        log(f"memoria pico de la EBM: {incrementos} MB -> mediana {salida['memoria_ebm_mb']:.1f}")
    except Exception as e:                                                  # noqa: BLE001
        log(f"memoria de la EBM no medida: {e}")
        salida["memoria_ebm_mb"] = np.nan

    # ---- sensibilidad a cada hiperparámetro por separado (validación cruzada de 3 pliegues, fidelidad B = 2)
    log("sensibilidad a los hiperparámetros: un hiperparámetro a la vez (3 pliegues, B = 2)...")
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(3, shuffle=True, random_state=SEMILLA)
    curvas_hp = []
    for hp, valores in (("learning_rate", [0.008, 0.02, 0.05, 0.1]), ("max_leaves", [2, 3, 4]),
                        ("min_samples_leaf", [4, 20, 100]), ("interactions", [0, 5, 10])):
        for v in valores:
            sc = cross_val_score(clone(modelo).set_params(**{hp: v, "outer_bags": 2}), Xtr, y_tr, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
            curvas_hp.append({"hiperparametro": hp, "valor": v, "auc_media": float(sc.mean()), "auc_desv": float(sc.std(ddof=1))})
            log(f"   {hp} = {v}: AUC {sc.mean():.4f} ± {sc.std(ddof=1):.4f}")
    salida["curvas_hiperparametros"] = pd.DataFrame(curvas_hp)

    # ---- número de parámetros de cada modelo final
    def parametros(pipeline, nombre: str) -> int:
        m = pipeline.named_steps["modelo"]
        m = getattr(m, "ebm_", m)
        if nombre == "ebm":
            return int(sum(np.asarray(s).size for s in m.term_scores_))
        if nombre == "xgboost":
            return int(len(m.get_booster().trees_to_dataframe()))
        if nombre == "random_forest":
            return int(sum(t.tree_.node_count for t in m.estimators_))
        if nombre == "arbol":
            return int(m.tree_.node_count)
        if nombre == "naive_bayes":
            return int(m.theta_.size + m.var_.size + m.class_prior_.size)
        if nombre == "knn":
            return int(np.prod(m._fit_X.shape))
        return int(np.ravel(m.coef_).size + np.ravel(m.intercept_).size)

    params = {}
    for tarea in ("clasificacion", "regresion"):
        for m, r in b["mejores"][tarea].set_index("modelo").iterrows():
            pl = pipe if (tarea == "clasificacion" and m == "ebm") else joblib.load(RES / "modelos" / Path(r["ruta_modelo"]).name)
            params[(tarea, m)] = parametros(pl, m)
    salida["parametros"] = params
    log("parámetros contados")

    # ---- memoria pico del ajuste de los modelos finales de regresión (el Entregable 3 solo la midió en clasificación)
    memoria_reg = {}
    try:
        from memory_profiler import memory_usage
        y_tr_c = c.y_train_cont.loc[c.X_train.index].to_numpy()
        for m, r in b["mejores"]["regresion"].set_index("modelo").iterrows():
            pl = joblib.load(RES / "modelos" / Path(r["ruta_modelo"]).name)
            Xr = pl.named_steps["preprocesado"].transform(c.X_train)
            est_r = pl.named_steps["modelo"]
            inc = []
            for _ in range(3 if m == "ebm" else 1):
                base = memory_usage(-1, interval=0.05, timeout=0.2)
                pico = memory_usage((lambda: clone(est_r).fit(Xr, y_tr_c), (), {}), interval=0.05, max_usage=True)
                pico = max(pico) if isinstance(pico, (list, tuple)) else pico
                inc.append(float(pico - min(base)))
            memoria_reg[m] = float(np.median(inc))
            log(f"   memoria regresión {m}: {memoria_reg[m]:.1f} MB")
    except Exception as e:                                                  # noqa: BLE001
        log(f"memoria de regresión no medida: {e}")
    salida["memoria_regresion_mb"] = memoria_reg

    # ---- EBM sin interacciones en la comparación con la Entrega 2 (etiqueta de la mediana)
    log("EBM sin interacciones (K = 0) en la partición y etiqueta de la Entrega 2...")
    carpeta = RES / "comparacion_entrega2"
    pred_e2 = pd.read_parquet(carpeta / "predicciones.parquet")
    mediana = float(c.y_train_cont.median())
    y_tr_med = (c.y_train_cont > mediana).astype(int).to_numpy()
    y_te_med = pred_e2["y_bin"].to_numpy()
    assert abs(((c.y_test_cont.loc[pred_e2.index] > mediana).astype(int) - y_te_med).abs().max()) == 0
    ebm_e2 = joblib.load(carpeta / "modelos" / "clasificacion__ebm.joblib")
    gam = clone(ebm_e2).set_params(modelo__interactions=0).fit(c.X_train, y_tr_med)
    s_gam = gam.predict_proba(c.X_test.loc[pred_e2.index])[:, 1]
    dl = est.delong(y_te_med, s_gam, pred_e2["clf_ebm"].to_numpy())
    bo = est.bootstrap_auc(y_te_med, s_gam, pred_e2["clf_ebm"].to_numpy(), n_boot=N_BOOT)
    dl2 = est.delong(y_te_med, pred_e2["clf_logistica_e2"].to_numpy(), s_gam)
    salida["gam_e2"] = {"auc_gam": dl["auc_a"], "auc_ebm": dl["auc_b"], "z": dl["z"], "p_delong": dl["p_bilateral"],
                        "ic_bca": bo["ic_bca"], "mejora_sobre_e2": dl2["diferencia"], "p_sobre_e2": dl2["p_bilateral"]}
    log(f"   GAM {dl['auc_a']:.4f} frente a EBM {dl['auc_b']:.4f}, DeLong p = {dl['p_bilateral']:.3f}")

    # ---- análisis de errores por segmento (prueba)
    df = b["df"]
    seg = pd.DataFrame({"edad": df.loc[pred.index, "age"].to_numpy(), "productos": df.loc[pred.index, "active_products"].to_numpy()},
                       index=pred.index)
    cohorte = {29: "29--30", 30: "29--30", 44: "44--45", 45: "44--45", 59: "59--60", 60: "59--60"}
    seg["cohorte"] = seg["edad"].map(cohorte)
    salida["segmentos"] = seg
    return salida


def analisis_errores(b: dict, clf: dict, reg: dict, pesado: dict) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    seg = pesado["segmentos"]
    y, yc = clf["y"], reg["y"]
    filas = []
    for nombre, grupos in (("cohorte de edad", seg["cohorte"]), ("productos activos", seg["productos"].astype(str))):
        for g in sorted(grupos.unique()):
            i = (grupos == g).to_numpy()
            fila = {"segmento": nombre, "nivel": g, "n": int(i.sum()), "prevalencia": float(y[i].mean())}
            for m in ("ebm", "xgboost"):
                s = clf["puntajes"][m][i]
                fila[f"auc_{m}"] = float(roc_auc_score(y[i], s)) if 0 < y[i].mean() < 1 else np.nan
                c_op = clf["clases_op"][m][i]
                fila[f"fn_n_{m}"] = int(((y[i] == 1) & (c_op == 0)).sum())
                fila[f"fn_{m}"] = float(fila[f"fn_n_{m}"] / max(1, (y[i] == 1).sum()))
                fila[f"rmse_{m}"] = float(np.sqrt(np.mean((yc[i] - reg["pron"][m][i]) ** 2)))
                fila[f"sesgo_{m}"] = float(np.mean(reg["pron"][m][i] - yc[i]))
            filas.append(fila)
    return pd.DataFrame(filas)


def ejemplos(b: dict, clf: dict, reg: dict) -> pd.DataFrame:
    """Tres clientes de prueba representativos de los errores: el de riesgo alto con menor puntaje de la EBM, el de
    riesgo bajo con mayor puntaje y el de mayor error absoluto de regresión."""
    df = b["df"].loc[b["pred"].index]
    y, yc = clf["y"], reg["y"]
    s_e, s_x = clf["puntajes"]["ebm"], clf["puntajes"]["xgboost"]
    c_e = clf["clases_op"]["ebm"]
    p_e, p_x = reg["pron"]["ebm"], reg["pron"]["xgboost"]
    fn = np.flatnonzero((y == 1) & (c_e == 0))
    fp = np.flatnonzero((y == 0) & (c_e == 1))
    casos = (("riesgo alto con menor puntaje", int(fn[np.argmin(s_e[fn])])),
             ("riesgo bajo con mayor puntaje", int(fp[np.argmax(s_e[fp])])),
             ("mayor error de regresión", int(np.argmax(np.abs(yc - p_e)))))
    filas = []
    for caso, i in casos:
        r = df.iloc[i]
        filas.append({"caso": caso, "edad": int(r["age"]), "logins": float(r["app_logins_frequency"]),
                      "satisfaccion": float(r["satisfaction_score"]), "productos": int(r["active_products"]),
                      "riesgo_alto": int(y[i]), "puntaje_real": float(yc[i]), "prob_ebm": float(s_e[i]), "prob_xgb": float(s_x[i]),
                      "pron_ebm": float(p_e[i]), "pron_xgb": float(p_x[i])})
    return pd.DataFrame(filas)


# ============================================================================ sensibilidad y costo
def sensibilidad(b: dict, pesado: dict) -> dict:
    sem = pd.read_csv(RES / "complementos" / "semillas.csv")
    semo = pd.read_csv(RES / "complementos" / "semillas_optimizadores.csv")
    esc = pd.read_csv(RES / "computo" / "escalamiento.csv")
    fid = pd.read_csv(RES / "computo" / "fidelidad.csv")
    tabla = b["tabla"]
    res_sem = sem.groupby(["tarea", "modelo"]).agg(semillas=("semilla", "size"), auc_media=("auc", "mean"),
                                                   auc_desv=("auc", "std"), rmse_media=("rmse", "mean"),
                                                   rmse_desv=("rmse", "std")).reset_index()
    res_semo = semo.groupby("optimizador").agg(semillas=("semilla", "size"), auc_media=("auc_prueba", "mean"),
                                               auc_desv=("auc_prueba", "std")).reset_index()
    curva = pd.concat([esc[esc["tarea"] == "clasificacion"][["modelo", "n", "auc", "entrenamiento_s"]],
                       pesado["curva_ebm"]], ignore_index=True)
    ebm_c = tabla[(tabla["modelo"] == "ebm") & (tabla["tarea"] == "clasificacion") & (tabla["estado"] == "ok")]
    hip = pd.DataFrame([{"balanceo": r["balanceo"], "optimizador": r["optimizador"], "auc_ext": r["media_ext"],
                         **{k.replace("modelo__", ""): v for k, v in r["hiperparametros"].items()}} for _, r in ebm_c.iterrows()])
    hip["max_bins"] = hip["max_bins"].fillna(1024)
    fid_ebm = fid[fid["modelo"] == "ebm"]
    rho = float(stats.spearmanr(fid_ebm["auc_baja"], fid_ebm["auc_alta"]).statistic)
    fid_res = {"configuraciones": int(len(fid_ebm)), "rho_spearman": rho,
               "misma_ganadora": bool(fid_ebm["auc_baja"].idxmax() == fid_ebm["auc_alta"].idxmax()),
               "ahorro_tiempo": float(1 - fid_ebm["tiempo_baja_s"].sum() / fid_ebm["tiempo_alta_s"].sum()),
               "rho_rf": float(stats.spearmanr(fid[fid.modelo == "random_forest"]["auc_baja"],
                                               fid[fid.modelo == "random_forest"]["auc_alta"]).statistic)}
    mp = pd.read_csv(RES / "metricas_prueba.csv")
    conc = []
    for tarea, col in (("clasificacion", "auc"), ("regresion", "rmse")):
        sub = mp[mp["tarea"] == tarea].dropna(subset=[col])
        conc.append({"tarea": tarea, "corridas": int(len(sub)), "correlacion": float(np.corrcoef(sub["metrica_cv"], sub[col])[0, 1]),
                     "diferencia_media": float((sub[col] - sub["metrica_cv"]).mean())})
    formas = []
    for v in SENAL:
        f = pesado["formas"][v]
        formas.append({"variable": v, "efecto_min": float(f["efecto"].min()), "efecto_max": float(f["efecto"].max()),
                       "tramos": int(len(f))})
    return {"semillas": res_sem, "semillas_optimizadores": res_semo, "curva": curva, "hiperparametros": hip,
            "fidelidad": fid_res, "fidelidad_puntos": fid_ebm, "concordancia": pd.DataFrame(conc),
            "formas_resumen": pd.DataFrame(formas), "curvas_hiperparametros": pesado["curvas_hiperparametros"]}


def costo(b: dict, pesado: dict, clf: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    tf = pd.read_csv(RES / "computo" / "tiempos_finales.csv")
    esc = pd.read_csv(RES / "computo" / "escalamiento.csv")
    mem = esc.dropna(subset=["memoria_mb"]).set_index(["tarea", "modelo"])["memoria_mb"]
    filas = []
    for tarea, tab in (("clasificacion", clf), ("regresion", reg)):
        for m, r in b["mejores"][tarea].set_index("modelo").iterrows():
            t = tf[(tf["tarea"] == tarea) & (tf["modelo"] == m)].iloc[0]
            filas.append({"tarea": tarea, "modelo": m, "ajuste_s": t["ajuste_s"], "inferencia_s": t["inferencia_s"],
                          "corrida_min": float(r["tiempo_total_s"]) / 60, "tamano_mb": t["tamano_modelo_mb"],
                          "parametros": pesado["parametros"][(tarea, m)],
                          "memoria_mb": pesado["memoria_ebm_mb"] if (m == "ebm" and tarea == "clasificacion")
                          else (pesado["memoria_regresion_mb"].get(m, np.nan) if tarea == "regresion" else mem.get((tarea, m), np.nan))})
    return pd.DataFrame(filas)


def optimizadores(b: dict) -> pd.DataFrame:
    """Rango medio y victorias de cada optimizador (de la tabla maestra) y las columnas de eficiencia por
    evaluación que publicó el Entregable 3 (tablas/optimizadores.tex)."""
    from src import estadistica as est
    sin_ebm = b["tabla"][b["tabla"]["modelo"] != "ebm"]                   # los 27 y 7 bloques de la guía
    res = b["analisis"].resumen_optimizadores(sin_ebm).reset_index()
    fried = []
    for tarea in ("clasificacion", "regresion"):
        f = est.friedman(b["analisis"].matriz_optimizadores(sin_ebm, tarea))
        fila = {"tarea": tarea, "bloques": int(f["n_bloques"]), "chi2": float(f["chi2"]), "p": float(f["p"]),
                "p_iman_davenport": float(f["p_iman_davenport"]), "dc": float(f["dc"])}
        fila.update({f"rango_{k}": float(v) for k, v in f["rangos_medios"].items()})
        pn = f["p_nemenyi"]
        fila.update({f"nemenyi_{a}_{c}": float(pn.loc[a, c]) for a in pn.index for c in pn.columns if a < c})
        fried.append(fila)
    DATOS.mkdir(exist_ok=True)
    pd.DataFrame(fried).to_csv(DATOS / "friedman_optimizadores.csv", index=False)
    tex = (E3 / "articulo" / "tablas" / "optimizadores.tex").read_text(encoding="utf-8")
    filas, tarea = [], None
    for linea in tex.splitlines():
        if "&" not in linea or "textbf" in linea:
            continue
        celdas = [c.strip().replace("\\\\", "") for c in linea.split("&")]
        if len(celdas) != 7:
            continue
        if celdas[0]:
            tarea = {"Clasificación": "clasificacion", "Regresión": "regresion"}[celdas[0]]
        metodo = {"Grid": "grid", "Random": "random", "Bayesiana": "bayesiana", "Genética": "genetica"}[celdas[1]]
        area, evals = (float(celdas[5].replace(",", ".")), float(celdas[6].replace(",", ".")))
        filas.append({"tarea": tarea, "optimizador": metodo, "area_arrepentimiento": area, "evals_95": evals})
    pub = pd.DataFrame(filas)
    return res.merge(pub, on=["tarea", "optimizador"], how="left")


def entrega2() -> pd.DataFrame:
    """Comparación con los modelos de la entrega anterior (partición y etiqueta de la mediana), tal como la calculó
    el Entregable 3 (resultados/comparacion_entrega2)."""
    carpeta = RES / "comparacion_entrega2"
    bus = json.loads((carpeta / "busquedas.json").read_text(encoding="utf-8"))["modelos"]
    ic = pd.read_csv(carpeta / "intervalos_clasificacion.csv").set_index("modelo")
    cc = pd.read_csv(carpeta / "contrastes_clasificacion.csv")
    ir = pd.read_csv(carpeta / "intervalos_regresion.csv").set_index("modelo")
    cr = pd.read_csv(carpeta / "contrastes_regresion.csv")
    filas = []
    for m in ("logistica_e2", "xgboost", "ebm"):
        k = cc[(cc["referencia"] == "logistica_e2") & (cc["candidato"] == m)]
        filas.append({"tarea": "clasificacion", "modelo": m, "metrica": ic.loc[m, "auc"], "ic_inf": ic.loc[m, "ic_bca_inf"],
                      "ic_sup": ic.loc[m, "ic_bca_sup"], "secundaria": bus[f"clasificacion/{m}"]["prueba"]["recall"],
                      "delta": k["diferencia"].iloc[0] if len(k) else np.nan,
                      "delta_inf": k["ic_bca_inf"].iloc[0] if len(k) else np.nan, "delta_sup": k["ic_bca_sup"].iloc[0] if len(k) else np.nan,
                      "estadistico": k["z"].iloc[0] if len(k) else np.nan, "p_holm": k["p_delong_ajustado"].iloc[0] if len(k) else np.nan,
                      "delta_cliff": k["delta_cliff"].iloc[0] if len(k) else np.nan})
    for m in ("lasso_e2", "xgboost", "ebm"):
        k = cr[(cr["referencia"] == "lasso_e2") & (cr["candidato"] == m)]
        filas.append({"tarea": "regresion", "modelo": m, "metrica": ir.loc[m, "rmse"], "ic_inf": ir.loc[m, "rmse_ic_inf"],
                      "ic_sup": ir.loc[m, "rmse_ic_sup"], "secundaria": ir.loc[m, "r2"],
                      "delta": k["mejora_rmse"].iloc[0] if len(k) else np.nan,
                      "delta_inf": k["mejora_rmse_ic_inf"].iloc[0] if len(k) else np.nan, "delta_sup": k["mejora_rmse_ic_sup"].iloc[0] if len(k) else np.nan,
                      "estadistico": k["dm_hln"].iloc[0] if len(k) else np.nan, "p_holm": k["p_dm_ajustado"].iloc[0] if len(k) else np.nan,
                      "delta_cliff": k["delta_cliff_pliegues"].iloc[0] if len(k) else np.nan})
    return pd.DataFrame(filas)


# ============================================================================ escritura de tablas
def guardar_datos(b, clf, reg, ab, err, sens, cost, opt, pesado, e2, ej) -> None:
    DATOS.mkdir(exist_ok=True)
    e2.to_csv(DATOS / "entrega2.csv", index=False)
    ej.to_csv(DATOS / "ejemplos.csv", index=False)
    sens["curvas_hiperparametros"].to_csv(DATOS / "sensibilidad_curvas.csv", index=False)
    clf["tabla"].to_csv(DATOS / "clasificacion_prueba.csv"); clf["pruebas"].to_csv(DATOS / "pruebas_clasificacion.csv")
    reg["tabla"].to_csv(DATOS / "regresion_prueba.csv"); reg["pruebas"].to_csv(DATOS / "pruebas_regresion.csv")
    ab.to_csv(DATOS / "ablacion.csv", index=False); err.to_csv(DATOS / "errores_segmentos.csv", index=False)
    sens["semillas"].to_csv(DATOS / "semillas.csv", index=False)
    sens["semillas_optimizadores"].to_csv(DATOS / "semillas_optimizadores.csv", index=False)
    sens["curva"].to_csv(DATOS / "curva_aprendizaje.csv", index=False)
    sens["hiperparametros"].to_csv(DATOS / "sensibilidad_hiperparametros.csv", index=False)
    sens["fidelidad_puntos"].to_csv(DATOS / "fidelidad_ebm.csv", index=False)
    sens["concordancia"].to_csv(DATOS / "concordancia_validacion_prueba.csv", index=False)
    sens["formas_resumen"].to_csv(DATOS / "formas_resumen.csv", index=False)
    pd.Series(sens["fidelidad"]).to_csv(DATOS / "fidelidad_resumen.csv")
    cost.to_csv(DATOS / "costo.csv", index=False); opt.to_csv(DATOS / "optimizadores.csv", index=False)
    pesado["importancia_terminos"].to_csv(DATOS / "importancia_terminos_ebm.csv")
    pesado["permutacion"].to_csv(DATOS / "importancia_permutacion_ebm.csv")
    pd.Series(pesado["gam_e2"]).to_csv(DATOS / "ablacion_interacciones_e2.csv")
    pd.DataFrame({m: v for m, v in clf["pliegues"].items()}).to_csv(DATOS / "pliegues_auc.csv", index=False)
    pd.DataFrame({m: v for m, v in reg["pliegues"].items()}).to_csv(DATOS / "pliegues_rmse.csv", index=False)
    for v, f in pesado["formas"].items():
        f.to_csv(DATOS / f"forma_ebm_{v}.csv", index=False)
    pd.DataFrame({"y": clf["y"], **{m: s for m, s in clf["puntajes"].items()}}).to_csv(DATOS / "puntajes_prueba.csv", index=False)
    pd.DataFrame({"y": reg["y"], **{m: s for m, s in reg["pron"].items()}}).to_csv(DATOS / "pronosticos_prueba.csv", index=False)


def cargar_desde_csv() -> tuple:
    """Reconstruye desde datos/*.csv lo que necesita :func:`tablas` (modo --tablas)."""
    def lee(nombre, idx=None):
        return pd.read_csv(DATOS / nombre, index_col=idx)
    clf = {"tabla": lee("clasificacion_prueba.csv", 0), "pruebas": lee("pruebas_clasificacion.csv", 0)}
    reg = {"tabla": lee("regresion_prueba.csv", 0), "pruebas": lee("pruebas_regresion.csv", 0)}
    fr = lee("fidelidad_resumen.csv", 0).iloc[:, 0]
    fid = {"configuraciones": int(float(fr["configuraciones"])), "rho_spearman": float(fr["rho_spearman"]),
           "misma_ganadora": str(fr["misma_ganadora"]) == "True", "ahorro_tiempo": float(fr["ahorro_tiempo"]),
           "rho_rf": float(fr["rho_rf"])}
    sens = {"semillas": lee("semillas.csv"), "semillas_optimizadores": lee("semillas_optimizadores.csv"), "fidelidad": fid}
    pesado = {"importancia_terminos": lee("importancia_terminos_ebm.csv", 0).iloc[:, 0],
              "permutacion": lee("importancia_permutacion_ebm.csv", 0)}
    return (clf, reg, lee("ablacion.csv"), lee("errores_segmentos.csv"), sens, lee("costo.csv"),
            lee("optimizadores.csv"), pesado, lee("entrega2.csv"), lee("ejemplos.csv"))


def tablas(clf, reg, ab, err, sens, cost, opt, pesado, e2, ej) -> None:
    TABLAS.mkdir(exist_ok=True)
    # ---- tabla principal de clasificación
    t, pr = clf["tabla"], clf["pruebas"]
    t = t.sort_values("auc_prueba", ascending=False)
    decimales = {"auc_ext_media": 4, "auc_prueba": 4, "auc_pr": 3, "f1_op": 3, "mcc_op": 3, "exactitud_bal_op": 3, "ece_isotonica": 3}
    marcas = {col: marcar(t[col].to_dict(), mayor_es_mejor=(col != "ece_isotonica"), d=d) for col, d in decimales.items()}
    filas = []
    for m, r in t.iterrows():
        simb = ""
        if m != "ebm":
            simb = ("\\dagger" if pr.loc[m, "p_delong_holm"] < ALFA else "") + ("\\ddagger" if pr.loc[m, "p_t_holm"] < ALFA else "")
            simb = f"$^{{{simb}}}$" if simb else ""
        nombre = ("\\textbf{EBM (propuesto)}" if m == "ebm" else NOMBRES[m]) + simb
        filas.append([nombre,
                      decorar(mas_menos(r["auc_ext_media"], r["auc_ext_desv"]), marcas["auc_ext_media"].get(m)),
                      decorar(f"{num(r['auc_prueba'])} {intervalo(r['auc_bca_inf'], r['auc_bca_sup'])}", marcas["auc_prueba"].get(m)),
                      decorar(num(r["auc_pr"], 3), marcas["auc_pr"].get(m)),
                      decorar(num(r["f1_op"], 3), marcas["f1_op"].get(m)),
                      decorar(num(r["mcc_op"], 3), marcas["mcc_op"].get(m)),
                      decorar(num(r["exactitud_bal_op"], 3), marcas["exactitud_bal_op"].get(m)),
                      decorar(num(r["ece_isotonica"], 3), marcas["ece_isotonica"].get(m))])
    escribir_tabla("clasificacion.tex",
                   "Clasificación (riesgo alto = percentil 75): mejor combinación de cada modelo según el bucle externo. "
                   "AUC externa (media $\\pm$ desviación de los 5 pliegues), AUC de prueba con intervalo \\emph{bootstrap} "
                   "BCa al 95~\\%, AUC-PR y, en el punto de operación que contacta al 25~\\% de mayor riesgo, F1, MCC y exactitud "
                   "balanceada; ECE tras la recalibración isotónica. Las métricas secundarias son valores puntuales de la prueba; "
                   "el \\emph{recall} y la precisión de ese punto están en la tabla~\\ref{tab:confusion}. Negrita: mejor; subrayado: "
                   "segundo (los empates comparten la marca). $^{\\dagger}$: difiere de la EBM en DeLong tras Holm; $^{\\ddagger}$: "
                   "en la $t$ corregida tras Holm.",
                   "tab:clasificacion", "@{}l c c c c c c c@{}",
                   ["Modelo", "AUC externa", "AUC prueba [IC 95~\\%]", "AUC-PR", "F1$_{25}$", "MCC$_{25}$", "Ex. bal.$_{25}$", "ECE"],
                   filas)

    # ---- matrices de confusión del modelo propuesto y del mejor modelo de referencia
    filas = []
    for umbral, suf in (("25~\\% de mayor riesgo", "op"), ("umbral 0,5", "05")):
        for m in ("ebm", "xgboost"):
            r = t.loc[m]
            vp, fp, fn, vn = (int(r[f"{k}_{suf}"]) for k in ("vp", "fp", "fn", "vn"))
            filas.append([umbral, "\\textbf{EBM (propuesto)}" if m == "ebm" else NOMBRES[m], ent(vp), ent(fp), ent(fn), ent(vn),
                          num(vp / max(1, vp + fp), 3), num(vp / max(1, vp + fn), 3)])
    escribir_tabla("confusion.tex",
                   "Matrices de confusión en la prueba (9.745 clientes, 2.399 de riesgo alto) de la EBM y de XGBoost con la "
                   "regla de decisión de la campaña (se contacta al 25~\\% de mayor puntaje, 2.436 clientes) y con el umbral 0,5 "
                   "de probabilidad: verdaderos positivos (VP), falsos positivos (FP), falsos negativos (FN), verdaderos negativos "
                   "(VN), precisión y \\emph{recall}.",
                   "tab:confusion", "@{}l l r r r r c c@{}",
                   ["Regla", "Modelo", "VP", "FP", "FN", "VN", "Precisión", "Recall"], filas, tamano="\\small", lineas_medias=[2])

    # ---- pruebas de significancia, clasificación
    filas, efectos = [], []
    for m in t.index:
        if m == "ebm":
            continue
        r = pr.loc[m]
        filas.append([NOMBRES[m],
                      f"{num(r['t_corr'], 2)}; {pval(r['p_t_holm'])}",
                      f"{num(r['z_delong'], 2)}; {pval(r['p_delong_holm'])}",
                      f"{ent(r['disc_base_bien'])}/{ent(r['disc_ebm_bien'])}; {pval(r['p_mcnemar_holm'])}"])
        efectos.append([NOMBRES[m],
                        f"{num(r['delta_auc'], 4, True)} {intervalo(r['delta_bca_inf'], r['delta_bca_sup'])}",
                        f"{num(r['p_peor'], 2)} / {num(r['p_rope'], 2)} / {num(r['p_mejor'], 2)}",
                        num(r['d_z_pliegues'], 2), num(r['delta_cliff'], 2, True)])
    escribir_tabla("pruebas_clasificacion.tex",
                   "Clasificación: pruebas de la EBM frente a cada modelo de referencia. $t$ corregida de Nadeau--Bengio sobre los "
                   "AUC de los 5 pliegues externos (positiva cuando la EBM supera a la referencia); $z$ de DeLong sobre la prueba; "
                   "McNemar exacta sobre las clases del punto de operación ($b$: clientes que solo la referencia clasifica bien; "
                   "$c$: solo la EBM). Los $p$ son bilaterales y están ajustados por Holm sobre las siete comparaciones de cada "
                   "prueba; los tamaños de efecto están en la tabla~\\ref{tab:efectos_clasificacion}.",
                   "tab:pruebas_clasificacion", "@{}l c c c@{}",
                   ["Referencia", "$t$ corr.; $p$", "DeLong $z$; $p$", "McNemar $b$/$c$; $p$"], filas, tamano="\\small")
    escribir_tabla("efectos_clasificacion.tex",
                   "Clasificación: tamaños de efecto de la EBM frente a cada referencia. $\\Delta$AUC $=$ AUC$_{\\mathrm{EBM}} - $"
                   "AUC$_{\\mathrm{ref}}$ en la prueba con intervalo \\emph{bootstrap} BCa al 95~\\%; probabilidad posterior de la "
                   "$t$ correlacionada bayesiana de que la EBM sea peor ($-$) / equivalente (ROPE $\\pm 0{,}01$) / mejor ($+$); "
                   "$d_z$ de Cohen y delta de Cliff (no pareada) entre pliegues, positivos cuando la EBM supera a la referencia.",
                   "tab:efectos_clasificacion", "@{}l c c c c@{}",
                   ["Referencia", "$\\Delta$AUC [IC 95~\\%]", "Bayes $-$ / ROPE / $+$", "$d_z$", "$\\delta$"], efectos,
                   tamano="\\small")

    # ---- tabla principal de regresión
    t, pr = reg["tabla"].sort_values("rmse_prueba"), reg["pruebas"]
    decimales = {"rmse_ext_media": 5, "rmse_prueba": 5, "mae_prueba": 4, "mape_prueba": 4, "r2_prueba": 3}
    marcas = {col: marcar(t[col].to_dict(), mayor_es_mejor=(col in ("r2_prueba",)), d=d) for col, d in decimales.items()}
    filas = []
    for m, r in t.iterrows():
        simb = ""
        if m != "ebm":
            simb = ("\\dagger" if pr.loc[m, "p_dm_holm"] < ALFA else "") + ("\\ddagger" if pr.loc[m, "p_t_holm"] < ALFA else "")
            simb = f"$^{{{simb}}}$" if simb else ""
        nombre = ("\\textbf{EBM (propuesto)}" if m == "ebm" else NOMBRES[m]) + simb
        filas.append([nombre,
                      decorar(mas_menos(r["rmse_ext_media"], r["rmse_ext_desv"], 5), marcas["rmse_ext_media"].get(m)),
                      decorar(f"{num(r['rmse_prueba'], 5)} {intervalo(r['rmse_bca_inf'], r['rmse_bca_sup'], 5)}", marcas["rmse_prueba"].get(m)),
                      decorar(num(r["mae_prueba"]), marcas["mae_prueba"].get(m)),
                      decorar(num(r["mape_prueba"] * 100, 2) + "~\\%", marcas["mape_prueba"].get(m)),
                      decorar(f"{num(r['r2_prueba'], 3)} {intervalo(r['r2_bca_inf'], r['r2_bca_sup'], 3)}", marcas["r2_prueba"].get(m)),
                      pval(r["p_mcs"])])
    escribir_tabla("regresion.tex",
                   "Regresión de \\cp: mejor optimizador de cada modelo según el bucle externo. RMSE externo (media $\\pm$ "
                   "desviación de los 5 pliegues), RMSE de prueba con intervalo BCa al 95~\\%, MAE, MAPE y $R^2$ con intervalo BCa, "
                   "y $p$-valor del \\emph{Model Confidence Set} al 90~\\% (pertenecen al conjunto los modelos con $p > 0{,}10$). "
                   "Negrita: mejor; subrayado: segundo. "
                   "$^{\\dagger}$: difiere de la EBM en Diebold-Mariano tras Holm; $^{\\ddagger}$: en la $t$ corregida tras Holm.",
                   "tab:regresion", "@{}l c c c c c c@{}",
                   ["Modelo", "RMSE externo", "RMSE prueba [IC 95~\\%]", "MAE", "MAPE", "$R^2$ [IC 95~\\%]", "$p$ MCS"], filas,
                   colsep="1.5pt")

    filas, efectos = [], []
    for m in t.index:
        if m == "ebm":
            continue
        r = pr.loc[m]
        filas.append([NOMBRES[m], f"{num(r['t_corr'], 2)}; {pval(r['p_t_holm'])}", f"{num(r['dm_hln'], 2)}; {pval(r['p_dm_holm'])}",
                      pval(reg["tabla"].loc[m, "p_mcs"])])
        efectos.append([NOMBRES[m],
                        f"{num(r['delta_rmse'], 5, True)} {intervalo(r['delta_bca_inf'], r['delta_bca_sup'], 5)}",
                        f"{num(r['p_peor'], 2)} / {num(r['p_rope'], 2)} / {num(r['p_mejor'], 2)}",
                        num(r['d_z_pliegues'], 2), num(r['d_z_clientes'], 3), num(r['delta_cliff'], 2, True)])
    escribir_tabla("pruebas_regresion.tex",
                   "Regresión: pruebas de la EBM frente a cada modelo de referencia. $t$ corregida sobre el RMSE de los 5 pliegues "
                   "(positiva cuando la EBM pierde menos); Diebold-Mariano con la corrección HLN sobre las pérdidas cuadráticas de la "
                   "prueba; $p$ del \\emph{Model Confidence Set} al 90~\\% de la referencia. Los $p$ de las dos primeras son "
                   "bilaterales y están ajustados por Holm sobre las siete comparaciones; los tamaños de efecto están en la "
                   "tabla~\\ref{tab:efectos_regresion}.",
                   "tab:pruebas_regresion", "@{}l c c c@{}",
                   ["Referencia", "$t$ corr.; $p$", "DM-HLN; $p$", "$p$ MCS"], filas, tamano="\\small")
    escribir_tabla("efectos_regresion.tex",
                   "Regresión: tamaños de efecto de la EBM frente a cada referencia. $\\Delta$RMSE $=$ RMSE$_{\\mathrm{ref}} - $"
                   "RMSE$_{\\mathrm{EBM}}$ en la prueba (positivo favorece a la EBM) con intervalo BCa al 95~\\%; probabilidad "
                   "posterior bayesiana de que la EBM sea peor ($-$) / equivalente (ROPE $\\pm 0{,}0005$ de RMSE) / mejor ($+$); "
                   "$d_z$ de Cohen entre pliegues y por cliente y delta de Cliff (no pareada) entre pliegues, positivos cuando la "
                   "EBM pierde menos.",
                   "tab:efectos_regresion", "@{}l c c c c c@{}",
                   ["Referencia", "$\\Delta$RMSE [IC 95~\\%]", "Bayes $-$ / ROPE / $+$", "$d_z$ pliegues", "$d_z$ clientes",
                    "$\\delta$"], efectos, tamano="\\small")

    # ---- ablación
    filas, medias = [], []
    for i, r in ab.iterrows():
        completo = r["componente"] == "completo"
        if completo and i > 0:
            medias.append(len(filas))
        ext = "--" if np.isnan(r["metrica_ext_media"]) else mas_menos(r["metrica_ext_media"], r["metrica_ext_desv"])
        rec = "--" if ("recall_05" not in r or np.isnan(r["recall_05"])) else num(r["recall_05"], 3)
        if completo:
            filas.append([r["variante"], ext, num(r["metrica_prueba"]), rec, "--", "--", "--", "--"])
            continue
        if r["tarea"] == "regresion":
            delta = f"{num(r['delta_completo'] * 1000, 2, True)} {intervalo(r['delta_bca_inf'] * 1000, r['delta_bca_sup'] * 1000, 2)}"
        else:
            delta = f"{num(r['delta_completo'], 4, True)} {intervalo(r['delta_bca_inf'], r['delta_bca_sup'], 4)}"
        pt = "--" if np.isnan(r["p_t"]) else pval(r["p_t_holm"])
        pp = pval(r["p_prueba"]) + "$^{*}$" if np.isnan(r["p_prueba_holm"]) else pval(r["p_prueba_holm"])
        bayes = "--" if np.isnan(r["p_rope"]) else num(r["p_rope"], 3)
        filas.append([r["variante"], ext, num(r["metrica_prueba"]), rec, delta, pt, pp, bayes])
    escribir_tabla("ablacion.tex",
                   "Ablación: cada variante cambia un componente de la EBM completa y conserva el resto. Métrica externa "
                   "(AUC o RMSE; media $\\pm$ desviación de los 5 pliegues) y de prueba, \\emph{recall} de prueba con umbral 0,5 "
                   "(clasificación); $\\Delta$ entre la variante y el modelo completo en la prueba, orientada para que un valor "
                   "positivo indique que la variante pierde (AUC$_{\\mathrm{comp}} -$ AUC$_{\\mathrm{var}}$; en regresión "
                   "RMSE$_{\\mathrm{var}} - $RMSE$_{\\mathrm{comp}}$ en milésimas), con intervalo BCa; $p$ de la $t$ corregida sobre "
                   "los pliegues y de DeLong o Diebold-Mariano sobre la prueba, ajustados por Holm dentro de cada tarea "
                   "($^{*}$: comparación única, sin ajustar); probabilidad bayesiana de equivalencia (ROPE) entre la variante y el "
                   "modelo completo. La variante sin interacciones se evalúa en la partición y la etiqueta de la comparación con la "
                   "entrega anterior, donde la búsqueda sí eligió pares.",
                   "tab:ablacion", "@{}>{\\raggedright\\arraybackslash}p{0.26\\linewidth} c c c c c c c@{}",
                   ["Variante", "Métrica externa", "Prueba", "Recall$_{0{,}5}$", "$\\Delta$ [IC 95~\\%]",
                    "$p$ $t$ corr.", "$p$ prueba", "$P$(ROPE)"], filas, lineas_medias=medias, colsep="1.5pt")

    # ---- sensibilidad: semillas
    filas = []
    for _, r in sens["semillas"].iterrows():
        if r["tarea"] == "clasificacion":
            filas.append(["Clasificación", NOMBRES[r["modelo"]], ent(r["semillas"]), "AUC", num(r["auc_media"]), num(r["auc_desv"], 5)])
        else:
            filas.append(["Regresión", NOMBRES[r["modelo"]], ent(r["semillas"]), "RMSE", num(r["rmse_media"]), num(r["rmse_desv"], 6)])
    for _, r in sens["semillas_optimizadores"].iterrows():
        filas.append(["Clasificación", f"Árbol, búsqueda con {OPTIM[r['optimizador']]}", ent(r["semillas"]), "AUC",
                      num(r["auc_media"]), num(r["auc_desv"], 5)])
    f = sens["fidelidad"]
    escribir_tabla("semillas.tex",
                   "Sensibilidad a la semilla: modelos finales reentrenados con distintas semillas (los datos no cambian) y "
                   "búsqueda del árbol de decisión repetida con cinco semillas de cada optimizador estocástico; media y desviación "
                   "de la métrica de prueba.",
                   "tab:semillas", "@{}l l r l c c@{}", ["Tarea", "Modelo", "Semillas", "Métrica", "Media", "Desviación"], filas,
                   notas=f"Multi-fidelidad de la EBM: {f['configuraciones']} configuraciones aleatorias evaluadas con $B=2$ y $B=8$; "
                         f"$\\rho$ de Spearman entre los dos AUC {num(f['rho_spearman'], 3)}, misma configuración ganadora "
                         f"({'sí' if f['misma_ganadora'] else 'no'}), ahorro de tiempo de búsqueda {num(f['ahorro_tiempo'] * 100, 0)}~\\%.")

    # ---- costo
    filas, medias = [], []
    for tarea in ("clasificacion", "regresion"):
        sub = cost[cost["tarea"] == tarea].sort_values("ajuste_s")
        if tarea == "regresion":
            medias.append(len(filas))
        for _, r in sub.iterrows():
            nombre = "\\textbf{EBM (propuesto)}" if r["modelo"] == "ebm" else NOMBRES[r["modelo"]]
            filas.append(["Clas." if tarea == "clasificacion" else "Regr.", nombre, num(r["ajuste_s"], 2), num(r["inferencia_s"], 3),
                          num(r["corrida_min"], 1), num(r["tamano_mb"], 2) if r["tamano_mb"] >= 0.01 else "$<0{,}01$",
                          ent(r["parametros"]), "--" if np.isnan(r["memoria_mb"]) else ("$<1$" if r["memoria_mb"] < 1 else num(r["memoria_mb"], 0))])
    escribir_tabla("costo.tex",
                   "Costo computacional de la mejor combinación de cada modelo en el equipo de referencia (AMD de 6 núcleos, "
                   "15,9~GB): ajuste del modelo final sobre los 38.978 clientes y puntuación de los 9.745 de prueba (mediana de 3 y 5 "
                   "repeticiones con el pipeline completo), duración de la corrida anidada completa, tamaño serializado, número de "
                   "parámetros (valores almacenados: nodos de los árboles, puntajes de las funciones de forma, coeficientes; k-NN "
                   "guarda la matriz de entrenamiento) y memoria pico del ajuste por encima de la base: en clasificación la midió el "
                   "Entregable 3 (la de la EBM se midió aquí, mediana de tres ajustes) y en regresión se midió aquí para todos.",
                   "tab:costo", "@{}l l r r r r r r@{}",
                   ["Tarea", "Modelo", "Ajuste (s)", "Inferencia (s)", "Corrida (min)", "Tamaño (MB)", "Parámetros", "Memoria (MB)"],
                   filas, lineas_medias=medias)

    # ---- interpretabilidad: importancias
    it, pm = pesado["importancia_terminos"], pesado["permutacion"]
    filas = []
    for i in range(8):
        a, pa = it.index[i], it.iloc[i]
        bname, pb = pm.index[i], pm.iloc[i]
        filas.append([f"\\texttt{{{a.replace('_', '\\_')}}}", num(pa * 100, 1) + "~\\%",
                      f"\\texttt{{{bname.replace('_', '\\_')}}}", f"{num(pb['media'], 4)} $\\pm$ {num(pb['desv'], 4)}"])
    escribir_tabla("importancia.tex",
                   "Interpretabilidad de la EBM de clasificación: cuota de la importancia de cada término (media del valor absoluto "
                   "de su función de forma sobre el entrenamiento; los 78 términos suman 100~\\%) e importancia por permutación "
                   "sobre la prueba (caída media $\\pm$ desviación del AUC al permutar cada predictor original, 5 repeticiones).",
                   "tab:importancia", "@{}l r l c@{}",
                   ["Término (EBM)", "Cuota", "Predictor permutado", "Caída del AUC"], filas, tamano="\\small")

    # ---- errores por segmento
    filas, medias = [], []
    for i, r in err.iterrows():
        if i > 0 and r["segmento"] != err.iloc[i - 1]["segmento"]:
            medias.append(len(filas))
        filas.append([f"{r['segmento']} {str(r['nivel']).replace(chr(8211), '--')}", ent(r["n"]), num(r["prevalencia"] * 100, 0) + "~\\%",
                      num(r["auc_ebm"], 3), num(r["auc_xgboost"], 3),
                      ("--" if r["prevalencia"] == 0 else num(r["fn_ebm"] * 100, 0) + "~\\%"),
                      ("--" if r["prevalencia"] == 0 else num(r["fn_xgboost"] * 100, 0) + "~\\%"),
                      num(r["rmse_ebm"]), num(r["rmse_xgboost"]), num(r["sesgo_ebm"], 3, True)])
    escribir_tabla("errores.tex",
                   "Análisis de errores en la prueba por cohorte de edad y por número de productos activos (variable excluida "
                   "del modelo): tamaño, prevalencia de riesgo alto, AUC de la EBM y de XGBoost dentro del segmento, fracción de "
                   "clientes de riesgo alto que cada modelo deja fuera del 25~\\% contactado (FN, falsos negativos), RMSE de regresión "
                   "de los dos modelos y sesgo medio del pronóstico de la EBM (pronóstico $-$ real).",
                   "tab:errores", "@{}l r r c c c c c c c@{}",
                   ["Segmento", "$n$", "Prev.", "AUC EBM", "AUC XGB", "FN EBM", "FN XGB", "RMSE EBM", "RMSE XGB", "Sesgo EBM"],
                   filas, lineas_medias=medias, colsep="2pt")

    # ---- optimizadores
    filas, medias = [], []
    for tarea in ("clasificacion", "regresion"):
        sub = opt[opt["tarea"] == tarea]
        if tarea == "regresion":
            medias.append(len(filas))
        for _, r in sub.iterrows():
            filas.append(["Clas." if tarea == "clasificacion" else "Regr.", OPTIM[r["optimizador"]], num(r["rango_medio"], 2),
                          ent(r["victorias"]), num(r["s_por_evaluacion"], 2), num(r["area_arrepentimiento"], 3), num(r["evals_95"], 1)])
    escribir_tabla("optimizadores.tex",
                   "Los cuatro optimizadores con el mismo presupuesto, en los bloques modelo $\\times$ balanceo con las cuatro "
                   "búsquedas completas de las 140 corridas de la guía (27 en clasificación, 7 en regresión; las de la EBM se "
                   "excluyen): rango medio por la métrica externa (1 = mejor), bloques ganados, segundos por evaluación, área media "
                   "bajo la curva de arrepentimiento normalizado (menor es mejor) y evaluaciones hasta cerrar el 95~\\% de la "
                   "brecha, estas dos últimas tal como las publicó el Entregable 3.",
                   "tab:optimizadores", "@{}l l r r r r r@{}",
                   ["Tarea", "Método", "Rango", "Victorias", "s/eval.", "Área arrep.", "Evals.\\ 95~\\%"], filas,
                   lineas_medias=medias, tamano="\\small")

    # ---- comparación con la entrega anterior
    NOMBRES_E2 = {"logistica_e2": "Logística L1 (E2)", "lasso_e2": "Lasso (E2)",
                  "xgboost": "XGBoost", "ebm": "\\textbf{EBM (propuesto)}"}
    filas, medias = [], []
    for i, r in e2.iterrows():
        if i > 0 and r["tarea"] != e2.iloc[i - 1]["tarea"]:
            medias.append(len(filas))
        d = 4 if r["tarea"] == "clasificacion" else 5
        sec = num(r["secundaria"], 4) if r["tarea"] == "clasificacion" else num(r["secundaria"], 4)
        if np.isnan(r["delta"]):
            filas.append([NOMBRES_E2[r["modelo"]], f"{num(r['metrica'], d)} {intervalo(r['ic_inf'], r['ic_sup'], d)}", sec, "--", "--", "--"])
        else:
            filas.append([NOMBRES_E2[r["modelo"]], f"{num(r['metrica'], d)} {intervalo(r['ic_inf'], r['ic_sup'], d)}", sec,
                          f"{num(r['delta'], d, True)} {intervalo(r['delta_inf'], r['delta_sup'], d)}",
                          f"{num(r['estadistico'], 2)}; {pval(r['p_holm'])}", num(r["delta_cliff"], 2, True)])
    escribir_tabla("entrega2.tex",
                   "Comparación con los modelos de la entrega anterior (E2) en su misma partición, etiqueta (mediana estimada con "
                   "el entrenamiento) y prueba, tal como la calculó el Entregable 3. Clasificación: AUC con intervalo BCa, "
                   "\\emph{recall} con umbral 0,5, mejora de AUC frente a la logística L1 con intervalo BCa, $z$ de DeLong y $p$ "
                   "ajustado por Holm, delta de Cliff sobre los 5 pliegues. Regresión: RMSE con intervalo BCa, $R^2$, reducción del "
                   "RMSE frente al Lasso con intervalo BCa, Diebold-Mariano (HLN) y $p$ ajustado, delta de Cliff entre pliegues.",
                   "tab:entrega2", "@{}l c c c c c@{}",
                   ["Modelo", "AUC o RMSE [IC 95~\\%]", "Recall$_{0{,}5}$ o $R^2$", "Mejora vs.\\ E2 [IC 95~\\%]",
                    "$z$ o DM; $p$", "$\\delta$"], filas, lineas_medias=medias, colsep="2pt")

    # ---- tres clientes representativos (transpuesta: una columna por caso)
    e = ej.set_index("caso")
    casos = list(e.index)
    cortos = {"riesgo alto con menor puntaje": "FN más grave", "riesgo bajo con mayor puntaje": "FP más grave",
              "mayor error de regresión": "Mayor error de regresión"}
    filas = [["Edad (años)"] + [ent(e.loc[c, "edad"]) for c in casos],
             ["Inicios de sesión"] + [num(e.loc[c, "logins"], 0) for c in casos],
             ["Satisfacción"] + [num(e.loc[c, "satisfaccion"], 0) for c in casos],
             ["\\ap"] + [ent(e.loc[c, "productos"]) for c in casos],
             ["Riesgo alto real"] + [("sí" if e.loc[c, "riesgo_alto"] == 1 else "no") for c in casos],
             ["Puntaje real"] + [num(e.loc[c, "puntaje_real"], 3) for c in casos],
             ["Prob. EBM / XGBoost"] + [f"{num(e.loc[c, 'prob_ebm'], 3)} / {num(e.loc[c, 'prob_xgb'], 3)}" for c in casos],
             ["Pron. EBM / XGBoost"] + [f"{num(e.loc[c, 'pron_ebm'], 3)} / {num(e.loc[c, 'pron_xgb'], 3)}" for c in casos]]
    escribir_tabla("ejemplos.tex",
                   "Tres clientes de prueba representativos de los errores: el de riesgo alto con menor puntaje de la EBM (FN, el "
                   "falso negativo más grave), el de riesgo bajo con mayor puntaje (FP, el falso positivo más grave) y el de mayor "
                   "error absoluto de regresión, con sus variables de señal, el nivel de \\ap\\ (que el modelo no ve), su "
                   "puntaje real y las predicciones de la EBM y de XGBoost.",
                   "tab:ejemplos", "@{}l c c c@{}", ["Variable"] + [cortos[c] for c in casos], filas)


# ============================================================================ figuras
def figuras() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pypdf import PdfReader
    from sklearn.metrics import precision_recall_curve, roc_curve

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 12, "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.3,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    NAVY, AMBER, GREEN, RED, GREY, LILAC = "#1F3A5F", "#E0A526", "#2E8B57", "#B03A2E", "#6E6E6E", "#7B5EA7"
    COLOR = {"ebm": AMBER, "xgboost": NAVY, "random_forest": GREEN, "arbol": LILAC, "knn": GREY, "logistica": RED,
             "svm": "#2C7FB8", "naive_bayes": "#8C564B", "lasso": RED, "ridge": "#2C7FB8", "svr": "#8C564B"}

    fallos = []

    def guardar(fig, nombre, recortar=False):
        FIGURAS.mkdir(exist_ok=True)
        ruta = FIGURAS / nombre
        if not recortar:
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            w, h = fig.get_size_inches()
            tb = fig.get_tightbbox(r)                    # en pulgadas: todo lo que se dibuja (rótulos incluidos)
            if tb.x0 < -0.02 or tb.y0 < -0.02 or tb.x1 > w + 0.02 or tb.y1 > h + 0.02:
                fallos.append(f"{nombre}: el dibujo sobresale del lienzo de {w:.2f} x {h:.2f} in: "
                              f"x=[{tb.x0:.2f}, {tb.x1:.2f}] y=[{tb.y0:.2f}, {tb.y1:.2f}]")
        opciones = {"bbox_inches": "tight", "pad_inches": 0.04} if recortar else {}
        fig.savefig(ruta, **opciones)
        plt.close(fig)
        caja = PdfReader(str(ruta)).pages[0].mediabox
        ancho, alto = float(caja.width) / 72, float(caja.height) / 72
        if ancho > 6.3 + 1e-3:
            fallos.append(f"{nombre}: {ancho:.2f} in de ancho (> 6.3 in)")
        log(f"figura {nombre} ({ancho:.2f} x {alto:.2f} in)")

    clf = pd.read_csv(DATOS / "clasificacion_prueba.csv", index_col=0)
    reg = pd.read_csv(DATOS / "regresion_prueba.csv", index_col=0)
    prc = pd.read_csv(DATOS / "pruebas_clasificacion.csv", index_col=0)
    prr = pd.read_csv(DATOS / "pruebas_regresion.csv", index_col=0)
    pl_auc = pd.read_csv(DATOS / "pliegues_auc.csv")
    pl_rmse = pd.read_csv(DATOS / "pliegues_rmse.csv")
    punt = pd.read_csv(DATOS / "puntajes_prueba.csv")
    pron = pd.read_csv(DATOS / "pronosticos_prueba.csv")
    ab = pd.read_csv(DATOS / "ablacion.csv")
    curva = pd.read_csv(DATOS / "curva_aprendizaje.csv")
    hip = pd.read_csv(DATOS / "sensibilidad_hiperparametros.csv")
    fidp = pd.read_csv(DATOS / "fidelidad_ebm.csv")
    calp = pd.read_parquet(RES / "complementos" / "calibracion_prueba.parquet")

    # ---- 1. intervalos de las diferencias frente a cada referencia
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), gridspec_kw={"width_ratios": [1, 1]}, layout="constrained")
    prr_m = prr.copy()
    for c in ("delta_rmse", "delta_bca_inf", "delta_bca_sup"):
        prr_m[c] = prr_m[c] * 1000
    for ax, pr, col, lo, hi, titulo, rope in (
            (axes[0], prc, "delta_auc", "delta_bca_inf", "delta_bca_sup", "ΔAUC", ROPE["auc"]),
            (axes[1], prr_m, "delta_rmse", "delta_bca_inf", "delta_bca_sup", "ΔRMSE × 10³", ROPE["rmse"] * 1000)):
        orden = pr.sort_values(col).index
        yv = np.arange(len(orden))
        ax.axvspan(-rope, rope, color=GREY, alpha=0.12, lw=0)
        ax.axvline(0, color=GREY, lw=1.5, ls="--")
        ax.errorbar(pr.loc[orden, col], yv, xerr=[pr.loc[orden, col] - pr.loc[orden, lo], pr.loc[orden, hi] - pr.loc[orden, col]],
                    fmt="o", color=NAVY, ecolor=NAVY, capsize=4, ms=6, lw=2)
        ax.set_yticks(yv)
        ax.set_yticklabels([CORTOS_EN[m] for m in orden])
        ax.set_xlabel(titulo, fontsize=12)
        ax.grid(axis="y", alpha=0)
    axes[0].set_xlim(-0.02, None)
    axes[1].set_xlim(-1.0, None)
    guardar(fig, "fig_intervals.pdf")

    # ---- 2. métrica primaria por pliegue y por modelo
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), layout="constrained")
    for ax, pl, titulo in ((axes[0], pl_auc, "Outer-fold AUC"), (axes[1], pl_rmse, "Outer-fold RMSE")):
        orden = list(pl.mean().sort_values(ascending=(titulo.endswith("RMSE"))).index)
        ax.boxplot([pl[m].to_numpy() for m in orden], positions=range(len(orden)), widths=0.55, showfliers=False,
                   medianprops={"color": "black", "lw": 2}, boxprops={"lw": 2, "color": GREY}, whiskerprops={"lw": 2, "color": GREY},
                   capprops={"lw": 2, "color": GREY})
        for i, m in enumerate(orden):
            v = pl[m].to_numpy()
            ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=22, color=COLOR[m], zorder=3)
        ax.set_xticks(range(len(orden)))
        ax.set_xticklabels([CORTOS_EN.get(m, "EBM*") for m in orden], rotation=60, ha="right")
        ax.set_ylabel(titulo)
    guardar(fig, "fig_folds.pdf")

    # ---- 3. ROC y PR
    y = punt["y"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), layout="constrained")
    for m in ("ebm", "xgboost", "random_forest", "logistica", "naive_bayes"):
        s = punt[m].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        pre, rec, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, color=COLOR[m], label=NOMBRES_EN[m], lw=2 if m == "ebm" else 1.6, alpha=0.95)
        axes[1].plot(rec, pre, color=COLOR[m], lw=2 if m == "ebm" else 1.6, alpha=0.95)
    axes[0].plot([0, 1], [0, 1], color=GREY, ls=":", lw=1.5)
    axes[1].axhline(y.mean(), color=GREY, ls=":", lw=1.5)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[0].legend(frameon=False, loc="lower right")
    guardar(fig, "fig_roc_pr.pdf")

    # ---- 4. calibración antes y después de la recalibración isotónica
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), sharey=True, layout="constrained")
    claves = {"ebm": "clasificacion__ebm__class_weight__grid", "xgboost": "clasificacion__xgboost__class_weight__genetica",
              "random_forest": "clasificacion__random_forest__ninguno__genetica"}
    yb = calp["y_bin"].to_numpy()
    bordes = np.linspace(0, 1, 11)
    for ax, version, titulo in ((axes[0], "sin recalibrar", "Model probabilities"), (axes[1], "isotónica", "After isotonic recalibration")):
        for m, k in claves.items():
            p = calp[f"{k}__{version}"].to_numpy()
            idx = np.clip(np.searchsorted(bordes, p, side="right") - 1, 0, 9)
            xs, ys = [], []
            for bb in range(10):
                en = idx == bb
                if en.sum() >= 20:
                    xs.append(p[en].mean()); ys.append(yb[en].mean())
            ax.plot(xs, ys, marker="o", ms=5, color=COLOR[m], label=NOMBRES_EN[m])
        ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1.5)
        ax.set_xlabel("Predicted probability"); ax.set_title(titulo, fontsize=12)
    axes[0].set_ylabel("Observed high-risk fraction")
    axes[0].legend(frameon=False, loc="upper left")
    guardar(fig, "fig_calibration.pdf")

    # ---- 5. regresión: pronóstico frente a observado y residuos
    yc = pron["y"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), layout="constrained")
    axes[0].scatter(pron["ebm"].to_numpy(), yc, s=3, color=AMBER, alpha=0.12, lw=0, rasterized=True, zorder=1)
    for m in ("ebm", "xgboost", "lasso"):
        p = pron[m].to_numpy()
        orden = np.argsort(p)
        grupos = np.array_split(orden, 20)
        axes[0].plot([p[g].mean() for g in grupos], [yc[g].mean() for g in grupos], marker="o", ms=4, color=COLOR[m],
                     label=NOMBRES_EN[m])
    lim = [np.quantile(yc, 0.02), np.quantile(yc, 0.98)]
    axes[0].plot(lim, lim, color=GREY, ls=":", lw=1.5)
    axes[0].set_xlabel("Predicted score"); axes[0].set_ylabel("Observed score")
    axes[0].set_xlim(0.24, 0.46); axes[0].set_ylim(0.1, 0.52)
    axes[0].legend(frameon=False, loc="upper left")
    res = yc - pron["ebm"].to_numpy()
    axes[1].hist(res, bins=60, color=AMBER, alpha=0.9, edgecolor="none")
    axes[1].axvline(0, color=GREY, ls="--", lw=1.5)
    axes[1].set_xlabel("EBM residual"); axes[1].set_ylabel("Test clients")
    guardar(fig, "fig_regression.pdf")

    # ---- 6. ablación
    sub = ab[ab["tarea"] == "clasificacion"]
    etiquetas = {"EBM completa (ponderación, Grid, $K=0$)": "complete\n(weights, Grid)",
                 "balanceo: sin balanceo (Grid)": "no balancing", "balanceo: SMOTE (Grid)": "SMOTE",
                 "balanceo: ADASYN (Grid)": "ADASYN", "optimizador: Random (ponderación)": "Random search",
                 "optimizador: TPE (ponderación)": "TPE", "optimizador: Genética (ponderación)": "genetic"}
    fig, ax = plt.subplots(figsize=(6.3, 3.2), layout="constrained")
    xs = np.arange(len(sub))
    colores = [AMBER if r["componente"] == "completo" else (NAVY if r["componente"] == "balanceo" else GREEN) for _, r in sub.iterrows()]
    ax.bar(xs, sub["metrica_ext_media"], yerr=sub["metrica_ext_desv"], color=colores, capsize=4, width=0.65,
           error_kw={"lw": 2}, label="outer-fold AUC (mean ± sd)")
    ax.scatter(xs, sub["metrica_prueba"], marker="D", color="black", s=30, zorder=4, label="test AUC")
    ax.set_ylim(0.80, 0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([etiquetas[v] for v in sub["variante"]], rotation=40, ha="right")
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    guardar(fig, "fig_ablation.pdf")

    # ---- 7. sensibilidad: curva de aprendizaje e hiperparámetros
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), layout="constrained")
    for m in ("ebm", "xgboost", "random_forest", "logistica"):
        g = curva[curva["modelo"] == m].sort_values("n")
        axes[0].plot(g["n"], g["auc"], marker="o", ms=5, color=COLOR[m], label=NOMBRES_EN[m])
    axes[0].set_xscale("log")
    axes[0].set_xticks([2500, 5000, 10000, 20000, 40000])
    axes[0].set_xticklabels(["2.5k", "5k", "10k", "20k", "40k"])
    axes[0].minorticks_off()
    axes[0].set_xlabel("Training clients"); axes[0].set_ylabel("Test AUC")
    axes[0].legend(frameon=False, loc="center right")
    k = hip["interactions"].to_numpy()
    for valor, color, etiqueta in ((0, NAVY, "winners, K = 0"), (1, GREEN, "winners, K > 0")):
        sel = (k == 0) if valor == 0 else (k > 0)
        axes[1].scatter(hip["learning_rate"][sel], hip["auc_ext"][sel], color=color, s=40, zorder=3, label=etiqueta)
    axes[1].scatter(fidp["hp_learning_rate"], fidp["auc_alta"], marker="^", color=GREY, s=40, zorder=3,
                    label="random (3-fold CV)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Learning rate η"); axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.8250, 0.8300)
    axes[1].legend(frameon=False, loc="lower left")
    guardar(fig, "fig_sensitivity.pdf")

    # ---- 7b. sensibilidad a cada hiperparámetro por separado
    sc = pd.read_csv(DATOS / "sensibilidad_curvas.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.0), layout="constrained")
    titulos = {"learning_rate": ("Learning rate η", True), "max_leaves": ("Leaves per tree L", False),
               "min_samples_leaf": ("Min. clients per leaf m", True), "interactions": ("Pairwise interactions K", False)}
    finales = {"learning_rate": 0.008, "max_leaves": 2, "min_samples_leaf": 100, "interactions": 0}
    for ax, (hp, (etiqueta, log_x)) in zip(axes.ravel(), titulos.items()):
        g = sc[sc["hiperparametro"] == hp].sort_values("valor")
        ax.errorbar(g["valor"], g["auc_media"], yerr=g["auc_desv"], marker="o", ms=6, color=AMBER, capsize=4, lw=2)
        ax.axvline(finales[hp], color=GREY, ls=":", lw=1.5)
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(list(g["valor"]))
        ax.set_xticklabels([f"{v:g}" for v in g["valor"]])
        ax.minorticks_off()
        ax.set_xlabel(etiqueta)
    axes[0, 0].set_ylabel("3-fold CV AUC"); axes[1, 0].set_ylabel("3-fold CV AUC")
    guardar(fig, "fig_hyper.pdf")

    # ---- 8. funciones de forma de la EBM
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2), layout="constrained")
    nombres_en = {"app_logins_frequency": "App logins frequency", "age": "Age (years)",
                  "satisfaction_score": "Satisfaction score", "base_satisfaction": "Base satisfaction"}
    for ax, v in zip(axes.ravel(), SENAL):
        f = pd.read_csv(DATOS / f"forma_ebm_{v}.csv")
        x = np.r_[f["desde"].iloc[0], f["hasta"].to_numpy()]
        ef = np.r_[f["efecto"].to_numpy(), f["efecto"].iloc[-1]]
        ax.step(x, ef, where="post", color=AMBER, lw=2)
        ax.fill_between(x, np.r_[f["inf"].to_numpy(), f["inf"].iloc[-1]], np.r_[f["sup"].to_numpy(), f["sup"].iloc[-1]],
                        step="post", color=AMBER, alpha=0.2, lw=0)
        ax.axhline(0, color=GREY, lw=1.5, ls=":")
        ax.set_xlabel(nombres_en[v], fontsize=12)
    axes[0, 0].set_ylabel("Logit contribution"); axes[1, 0].set_ylabel("Logit contribution")
    guardar(fig, "fig_shapes.pdf")
    if fallos:
        raise SystemExit("\n".join(fallos))


# ============================================================================ principal
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figuras", action="store_true", help="solo redibujar las figuras desde datos/*.csv")
    parser.add_argument("--tablas", action="store_true", help="solo reescribir las tablas desde datos/*.csv")
    args = parser.parse_args()
    if args.tablas:
        tablas(*cargar_desde_csv())
        return
    if not args.figuras:
        b = cargar_basicos()
        log(f"insumos cargados: {len(b['tabla'])} corridas, prueba de {len(b['y_bin'])} clientes, tau = {b['tau']:.4f}")
        clf = clasificacion(b)
        reg = regresion(b)
        pesado = calculos_pesados(b)
        ab = ablacion(b, clf, reg, pesado)
        err = analisis_errores(b, clf, reg, pesado)
        sens = sensibilidad(b, pesado)
        cost = costo(b, pesado, clf["tabla"], reg["tabla"])
        opt = optimizadores(b)
        e2 = entrega2()
        ej = ejemplos(b, clf, reg)
        guardar_datos(b, clf, reg, ab, err, sens, cost, opt, pesado, e2, ej)
        tablas(clf, reg, ab, err, sens, cost, opt, pesado, e2, ej)
        resumen = {"tau": b["tau"], "n_prueba": int(len(b["y_bin"])), "prevalencia_prueba": float(b["y_bin"].mean()),
                   "mcs_incluidos": reg["mcs"]["incluidos"], "mcs_bloque": reg["mcs"]["bloque"],
                   "memoria_ebm_mb": pesado["memoria_ebm_mb"], "rope": ROPE, "n_boot": N_BOOT}
        (DATOS / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    figuras()
    log("listo")


if __name__ == "__main__":
    main()
