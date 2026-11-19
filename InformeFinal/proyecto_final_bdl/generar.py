"""Sección de resultados del proyecto final (modelo original BDL frente a los modelos del curso), escrita según la guía
«Cómo escribir la sección de resultados». Este único guion reproduce cada tabla, cada prueba estadística y cada figura
a partir de los resultados versionados del proyecto final (ModeloOriginal/resultados), que solo se leen.

    python generar.py             calcula todo (pruebas, tablas, datos intermedios y figuras; unos minutos)
    python generar.py --tablas    solo vuelve a escribir las tablas a partir de datos/*.csv
    python generar.py --figuras   solo vuelve a dibujar las figuras a partir de datos/*.csv

Insumos (solo lectura): experimentos.parquet (las 15 particiones externas de cada modelo: tres repeticiones de cinco
pliegues), predicciones_prueba.parquet (los 9.745 clientes de prueba), ablacion*.csv/parquet, semillas.csv,
memoria.csv, errores_subgrupos.csv, robustez/*.csv, adicionales/<conjunto>/experimentos.parquet y modelos/*.joblib.
Lo único que se ajusta aquí, y el texto lo declara como añadido después del protocolo, es la regresión bayesiana
(BayesianRidge), que la guía pide y el protocolo registrado no incluía, y la importancia por permutación.

Salidas: tablas/*.tex (booktabs), datos/*.csv y datos/*.json (toda cifra del texto) y figures/*.pdf (vectoriales, en
inglés, fuente sin serifa, ancho máximo 6,3 in).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np                                                                    # noqa: E402
import pandas as pd                                                                   # noqa: E402
from scipy import stats                                                               # noqa: E402

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent


def carpeta_modelo() -> Path:
    """ModeloOriginal del repositorio: ../entrega 3/repo_entregable3/ModeloOriginal (carpeta de trabajo),
    ../../ModeloOriginal (dentro del repositorio publicado) o la variable de entorno MODELO_ORIGINAL."""
    candidatas = (os.environ.get("MODELO_ORIGINAL"),
                  AQUI.parent / "entrega 3" / "repo_entregable3" / "ModeloOriginal",
                  AQUI.parent.parent / "ModeloOriginal")
    for c in candidatas:
        if c and (Path(c) / "resultados" / "experimentos.parquet").exists():
            return Path(c)
    sys.exit("No se encontró ModeloOriginal/resultados; fije la variable MODELO_ORIGINAL")


MO = carpeta_modelo()
RES = MO / "resultados"
sys.path.insert(0, str(MO))
TABLAS, FIGURAS, DATOS = AQUI / "tablas", AQUI / "figures", AQUI / "datos"

ALFA = 0.05
ROPE = 0.0005            # región de equivalencia práctica en RMSE (≈ 0,85 % del RMSE), la misma de la versión anterior
N_BOOT = 2000
RAZON = 0.25             # n_prueba / n_entrenamiento de cada pliegue externo (1/4)
SEMILLAS = (42, 43, 44)
K_EXT = 5
MODELOS = ["bdl", "xgboost", "lightgbm", "ebm", "random_forest", "arbol", "ridge", "lasso", "bayesiana", "svr", "knn"]
BASES = [m for m in MODELOS if m != "bdl"]
ADICIONALES = ["seguros", "energia", "abulon", "california"]
NOMBRES = {"bdl": "BDL (propuesto)", "xgboost": "XGBoost", "lightgbm": "LightGBM", "ebm": "EBM",
           "random_forest": "Random Forest", "arbol": "Árbol de decisión", "ridge": "Ridge", "lasso": "Lasso",
           "bayesiana": "Regresión bayesiana", "svr": "SVR lineal", "knn": "k-NN"}
NOMBRES_EN = {"bdl": "BDL (proposed)", "xgboost": "XGBoost", "lightgbm": "LightGBM", "ebm": "EBM",
              "random_forest": "Random forest", "arbol": "Decision tree", "ridge": "Ridge", "lasso": "Lasso",
              "bayesiana": "Bayesian ridge", "svr": "Linear SVR", "knn": "k-NN"}
CONJ = {"cofinfad": "COFINFAD", "seguros": "Seguros médicos", "energia": "Eficiencia energética", "abulon": "Abulón",
        "california": "Vivienda de California"}
CONJ_EN = {"cofinfad": "COFINFAD", "seguros": "Insurance", "energia": "Energy", "abulon": "Abalone",
           "california": "California"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ utilidades estadísticas
def t_corregida(a, b, razon: float = RAZON) -> dict:
    """t pareada con varianza corregida de Nadeau y Bengio (2003): d = a − b; t = media(d)/sqrt((1/J + razon)·var(d)),
    J − 1 grados de libertad; con el intervalo del 95 % de la diferencia media con la misma varianza corregida."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    j = d.size
    ee = np.sqrt((1.0 / j + razon) * d.var(ddof=1))
    t = d.mean() / ee
    q = stats.t.ppf(0.975, df=j - 1)
    return {"t": float(t), "p": float(2 * stats.t.sf(abs(t), df=j - 1)), "media": float(d.mean()),
            "ic_inf": float(d.mean() - q * ee), "ic_sup": float(d.mean() + q * ee), "j": j}


def bayes_rope(a, b, rope: float = ROPE, razon: float = RAZON) -> dict:
    """t correlacionada bayesiana (Corani y Benavoli, 2015; Benavoli y otros, 2017): posterior t de Student con J − 1
    grados de libertad, centro media(d) y escala sqrt((1/J + razon)·var(d)), con d = a − b. Devuelve P(d > rope)
    (la referencia a pierde por más que la región), P(|d| ≤ rope) y P(d < −rope)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    j = d.size
    dist = stats.t(df=j - 1, loc=d.mean(), scale=np.sqrt((1.0 / j + razon) * d.var(ddof=1)))
    p_peor, p_mejor = float(dist.cdf(-rope)), float(dist.sf(rope))
    return {"p_mejor": p_mejor, "p_rope": float(1 - p_peor - p_mejor), "p_peor": p_peor}


def a12(menor_es_mejor_x, y) -> float:
    """A12 de Vargha y Delaney: probabilidad de que un valor de x sea mejor (menor) que uno de y, más la mitad de los
    empates, sobre todos los pares."""
    x, y = np.asarray(menor_es_mejor_x, dtype=float)[:, None], np.asarray(y, dtype=float)[None, :]
    return float(np.mean(x < y) + 0.5 * np.mean(x == y))


def metricas(y, p) -> dict:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    e = y - p
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), "mape": float(np.mean(np.abs(e) / np.abs(y)))}


# ============================================================================ formato LaTeX
def num(x, d: int = 4, signo: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = str(Decimal(repr(float(abs(x)))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    entero, _, dec = s.partition(".")
    entero = f"{int(entero):,}".replace(",", ".")
    s = entero + ("{,}" + dec if dec else "")
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
    if p < 0.001:
        mant, exp = f"{p:.1e}".split("e")
        return f"${mant.replace('.', '{,}')} \\times 10^{{{int(exp)}}}$"
    return "$" + str(Decimal(repr(float(p))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)).replace(".", "{,}") + "$"


def mas_menos(m, s, d: int = 4) -> str:
    return f"{num(m, d)} $\\pm$ {num(s, d)}"


def marcar(valores: dict, mayor_es_mejor: bool, d: int = 4) -> dict:
    """Mejor valor en negrita y segundo subrayado (los empates al redondeo mostrado comparten la marca)."""
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


def escribir_tabla(nombre: str, caption: str, label: str, columnas: str, cabecera: list[str], filas: list[list[str]],
                   notas: str | None = None, tamano: str = "\\scriptsize", colsep: str = "3pt",
                   lineas_medias: list[int] | None = None) -> None:
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


def guardar_json(nombre: str, obj) -> None:
    (DATOS / nombre).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


# ============================================================================ insumos
def leer(v):
    return json.loads(v) if isinstance(v, str) else v


def cargar() -> dict:
    """Métricas por pliegue (15 particiones externas) y pronósticos de prueba de los diez modelos del protocolo."""
    from src import plan
    t = pd.read_parquet(RES / "experimentos.parquet")
    reg = t[(t.tarea == "regresion") & (t.estado == "ok")]
    F = {}
    for m in [x for x in MODELOS if x != "bayesiana"]:
        g = reg[reg.modelo == m].sort_values("semilla")
        assert list(g.semilla) == list(SEMILLAS), f"{m}: faltan repeticiones"
        met = [leer(x) for x in g.metricas_ext]
        F[m] = {k: np.concatenate([np.asarray(mm[k], dtype=float) for mm in met]) for k in ("rmse", "mae", "r2")}
    control = pd.read_csv(RES / "pliegues_regresion.csv", index_col=0)
    for m in F:
        assert np.allclose(F[m]["rmse"], control[m].to_numpy()), f"{m}: los pliegues no coinciden con pliegues_regresion.csv"
    X_tr, y_tr, X_te, y_te = plan.objetivo("regresion")
    pred = pd.read_parquet(RES / "predicciones_prueba.parquet")
    assert np.array_equal(pred.index.to_numpy(), X_te.index.to_numpy()), "la prueba no coincide con las predicciones"
    y = y_te.to_numpy(dtype=float)
    P = {m: pred[f"regresion__{m}"].to_numpy(dtype=float) for m in F}
    mp = pd.read_csv(RES / "metricas_prueba.csv").query("tarea == 'regresion'").set_index("modelo")
    for m in P:
        assert abs(metricas(y, P[m])["rmse"] - mp.loc[m, "rmse"]) < 1e-10, f"{m}: RMSE de prueba distinto"
    log(f"insumos: {len(F)} modelos, {len(y)} clientes de prueba, {len(X_tr)} de entrenamiento")
    return {"reg": reg, "F": F, "P": P, "y": y, "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te}


def tuberia_bayesiana(X):
    from sklearn.linear_model import BayesianRidge
    from sklearn.pipeline import Pipeline
    from src.preprocesamiento import construir_preprocesador
    return Pipeline([("preprocesado", construir_preprocesador(X)), ("modelo", BayesianRidge())])


def bayesiana(b: dict) -> dict:
    """BayesianRidge con los mismos pliegues externos, el mismo preprocesamiento y la misma prueba. Sus hiperparámetros
    (las precisiones del ruido y de los pesos) se estiman dentro de cada ajuste por máxima evidencia, sin búsqueda.
    Antes se comprueba que los pliegues reconstruidos son los del protocolo: Ridge, reajustado en cada pliegue con
    sus hiperparámetros guardados, debe reproducir su RMSE guardado."""
    from src.optimizacion import construir_cv
    from src.variantes import reajustar
    X, y = b["X_tr"], b["y_tr"]
    ridge = b["reg"][b["reg"].modelo == "ridge"].set_index("semilla")
    por_pliegue = {k: [] for k in ("rmse", "mae", "r2", "mape")}
    control, tiempos = [], []
    for s in SEMILLAS:
        params, guardado = leer(ridge.loc[s, "params_por_pliegue"]), leer(ridge.loc[s, "metricas_ext"])["rmse"]
        for j, (tr, va) in enumerate(construir_cv("regresion", K_EXT, s).split(X, y)):
            yv = y.iloc[va].to_numpy(dtype=float)
            r = reajustar("regresion", "ridge", "ninguno", params[j], X.iloc[tr], y.iloc[tr], s)
            control.append(abs(metricas(yv, r.predict(X.iloc[va]))["rmse"] - guardado[j]))
            t0 = time.perf_counter()
            p = tuberia_bayesiana(X).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
            tiempos.append(time.perf_counter() - t0)
            for k, v in metricas(yv, p).items():
                por_pliegue[k].append(v)
    assert max(control) < 1e-9, f"los pliegues reconstruidos no son los del protocolo (diferencia {max(control):.2e})"
    log(f"regresión bayesiana: pliegues comprobados con Ridge (diferencia máxima {max(control):.1e})")
    ajustes, inferencias = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        final = tuberia_bayesiana(X).fit(X, y)
        ajustes.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        p_te = final.predict(b["X_te"])
        inferencias.append(time.perf_counter() - t0)
    import io
    import joblib
    buf = io.BytesIO()
    mod = final.named_steps["modelo"]
    joblib.dump(mod, buf)                       # solo el estimador, como el tamaño de semillas.csv para los demás
    return {"F": {k: np.asarray(v) for k, v in por_pliegue.items()}, "P": p_te, "control": float(max(control)),
            "ajuste_s": float(np.median(ajustes)), "inferencia_s": float(np.median(inferencias)),
            "tamano_kb": buf.tell() / 1024, "parametros": int(mod.coef_.size + 1),
            "alpha": float(mod.alpha_), "lambda": float(mod.lambda_)}


def bayesiana_adicionales() -> dict:
    """La regresión bayesiana en los cuatro conjuntos adicionales (cinco pliegues externos de la semilla 42), con el
    mismo control de los pliegues."""
    from src import conjuntos, datos
    from src.optimizacion import construir_cv
    from src.variantes import reajustar
    out = {}
    for ds in ADICIONALES:
        X, s_, _ = conjuntos.cargar(ds)
        part = datos.partir(X, seed=SEMILLAS[0])
        Xtr, ytr = X.loc[part.idx_train], pd.Series(s_.loc[part.idx_train].to_numpy(dtype=float), index=part.idx_train)
        a = pd.read_parquet(RES / "adicionales" / ds / "experimentos.parquet")
        rr = a[(a.tarea == "regresion") & (a.modelo == "ridge")].iloc[0]
        params, guardado = leer(rr.params_por_pliegue), leer(rr.metricas_ext)["rmse"]
        rm, ctrl = [], []
        for j, (tr, va) in enumerate(construir_cv("regresion", K_EXT, SEMILLAS[0]).split(Xtr, ytr)):
            yv = ytr.iloc[va].to_numpy(dtype=float)
            r = reajustar("regresion", "ridge", "ninguno", params[j], Xtr.iloc[tr], ytr.iloc[tr], SEMILLAS[0])
            ctrl.append(abs(metricas(yv, r.predict(Xtr.iloc[va]))["rmse"] / guardado[j] - 1))
            rm.append(metricas(yv, tuberia_bayesiana(Xtr).fit(Xtr.iloc[tr], ytr.iloc[tr]).predict(Xtr.iloc[va]))["rmse"])
        out[ds] = {"rmse": rm, "rmse_medio": float(np.mean(rm)), "control": float(max(ctrl)), "ok": bool(max(ctrl) < 1e-7)}
        log(f"regresión bayesiana en {ds}: {np.mean(rm):.5g} (control {max(ctrl):.1e})")
    return out


# ============================================================================ desempeño y pruebas
def desempeno(b: dict, F: dict, P: dict) -> pd.DataFrame:
    from src import estadistica as est
    y, filas = b["y"], []
    for m in MODELOS:
        bo = est.bootstrap_regresion(y, P[m], n_boot=N_BOOT)
        mt = metricas(y, P[m])
        filas.append({"modelo": m, **{f"{k}_media": float(np.mean(F[m][k])) for k in ("rmse", "mae", "r2")},
                      **{f"{k}_desv": float(np.std(F[m][k], ddof=1)) for k in ("rmse", "mae", "r2")},
                      "rmse_prueba": mt["rmse"], "rmse_inf": bo["rmse"]["ic_bca"][0], "rmse_sup": bo["rmse"]["ic_bca"][1],
                      "mae_prueba": mt["mae"], "mape_prueba": mt["mape"], "r2_prueba": mt["r2"],
                      "pliegues_bdl_gana": int(np.sum(F["bdl"]["rmse"] < F[m]["rmse"])) if m != "bdl" else np.nan})
    return pd.DataFrame(filas).set_index("modelo")


def pruebas(b: dict, F: dict, P: dict) -> tuple[pd.DataFrame, dict]:
    """Combinación mínima de la guía para regresión tabular (t corregida + Diebold-Mariano-HLN + bayesiana con ROPE),
    con Holm sobre los diez contrastes de cada prueba, tamaños de efecto, Wilcoxon como verificación y el MCS."""
    from src import estadistica as est
    y = b["y"]
    perd = {m: (y - P[m]) ** 2 for m in MODELOS}
    filas = []
    for m in BASES:
        a, c = F[m]["rmse"], F["bdl"]["rmse"]
        tc, by = t_corregida(a, c), bayes_rope(a, c)
        dm = est.diebold_mariano(perd[m], perd["bdl"])                  # d_media > 0: BDL pierde menos
        bo = est.bootstrap_regresion(y, P[m], P["bdl"], n_boot=N_BOOT)  # rmse: RMSE_m − RMSE_BDL
        filas.append({"modelo": m, "delta_pliegues": tc["media"], "delta_inf": tc["ic_inf"], "delta_sup": tc["ic_sup"],
                      "t": tc["t"], "p_t": tc["p"], "dm": dm["estadistico"], "p_dm": dm["p_bilateral"],
                      "delta_prueba": bo["rmse"]["estimacion"], "delta_prueba_inf": bo["rmse"]["ic_bca"][0],
                      "delta_prueba_sup": bo["rmse"]["ic_bca"][1], **by,
                      "d_cohen": est.d_cohen(a, c)["d"], "a12": a12(c, a), "pliegues_bdl_gana": int(np.sum(c < a)),
                      "p_wilcoxon": float(stats.wilcoxon(a, c).pvalue)})
    pr = pd.DataFrame(filas).set_index("modelo")
    for col in ("p_t", "p_dm", "p_wilcoxon"):
        pr[col + "_holm"] = est.ajustar_pvalores(pr[col].to_numpy(), "holm")
    perdidas = pd.DataFrame({m: perd[m] for m in MODELOS})
    mcs = est.conjunto_confianza_modelos(perdidas, alfa=0.10, n_boot=1000)
    pr["p_mcs"] = pd.Series(mcs["p_mcs"]).reindex(pr.index)
    extra = {"mcs_incluidos": list(mcs["incluidos"]), "mcs_bloque": mcs.get("bloque"),
             "p_mcs_bdl": float(pd.Series(mcs["p_mcs"]).get("bdl", np.nan))}
    # Las cuatro pruebas registradas en el protocolo (para el veredicto confirmatorio, que no cambia)
    reg = pd.read_csv(RES / "pruebas_hipotesis.csv").query("tarea == 'regresion' and benchmark == 'xgboost'")
    extra["protocolo_xgboost"] = {r.prueba: {"nombre": r.nombre, "p": r.p_crudo, "p_holm": r.p_holm, "rechaza": bool(r.rechaza)}
                                  for r in reg.itertuples()}
    log(f"pruebas listas; MCS al 90 %: {extra['mcs_incluidos']}")
    return pr, extra


def conjuntos_adicionales(F: dict, bay_ad: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """RMSE medio de cada modelo en los cinco conjuntos, la condición 4 por conjunto (t corregida frente al mejor
    benchmark de cada uno) y Friedman con Iman-Davenport y Nemenyi con los conjuntos como bloques."""
    from src import estadistica as est
    medias = {"cofinfad": {m: float(np.mean(F[m]["rmse"])) for m in MODELOS}}
    cond = []
    for ds in ADICIONALES:
        a = pd.read_parquet(RES / "adicionales" / ds / "experimentos.parquet")
        r = a[(a.tarea == "regresion") & (a.estado == "ok")].set_index("modelo")
        rm = {m: np.asarray(leer(r.loc[m, "metricas_ext"])["rmse"], dtype=float) for m in r.index}
        medias[ds] = {m: float(v.mean()) for m, v in rm.items()}
        if bay_ad.get(ds, {}).get("ok"):
            medias[ds]["bayesiana"] = bay_ad[ds]["rmse_medio"]
        bench = {m: v for m, v in medias[ds].items() if m not in ("bdl", "bayesiana")}
        mejor = min(bench, key=bench.get)
        tc = t_corregida(rm[mejor], rm["bdl"])
        orden = sorted(medias[ds], key=medias[ds].get)
        cond.append({"conjunto": ds, "mejor": mejor, "rmse_mejor": bench[mejor], "rmse_bdl": medias[ds]["bdl"],
                     "ventaja_relativa": (bench[mejor] - medias[ds]["bdl"]) / bench[mejor], "p_t": tc["p"],
                     "puesto": orden.index("bdl") + 1, "modelos": len(orden)})
    mat = pd.DataFrame(medias).T
    mat = mat[[m for m in MODELOS if m in mat.columns and mat[m].notna().all()]]
    fr = est.friedman(mat, mayor_es_mejor=False)
    rangos = fr["rangos_medios"]
    info = {"modelos": list(mat.columns), "n_conjuntos": int(mat.shape[0]), "chi2": fr["chi2"], "p_friedman": fr["p"],
            "F": fr["F"], "p_iman_davenport": fr["p_iman_davenport"], "dc": fr["dc"],
            "rangos": {m: float(v) for m, v in rangos.items()},
            "p_nemenyi_bdl": {m: float(fr["p_nemenyi"].loc["bdl", m]) for m in mat.columns if m != "bdl"}}
    log(f"Friedman con {info['n_conjuntos']} conjuntos y {len(info['modelos'])} modelos: p = {fr['p']:.3f}, DC = {fr['dc']:.2f}")
    return mat, pd.DataFrame(cond).set_index("conjunto"), info


def ablacion(b: dict, F: dict, P: dict) -> tuple[pd.DataFrame, dict]:
    """BDL completo frente a BDL sin mezcla (K = 1) con la misma prueba de la comparación principal, más el barrido de K
    (análisis de sensibilidad) y la ganancia de la mezcla según el tamaño de entrenamiento."""
    from src import estadistica as est
    t = pd.read_parquet(RES / "experimentos.parquet")
    k1 = t[(t.tarea == "regresion") & (t.modelo == "bdl_sin_mezcla")].iloc[0]
    s = int(k1.semilla)
    a = np.asarray(leer(k1.metricas_ext)["rmse"], dtype=float)
    idx = SEMILLAS.index(s)
    c = F["bdl"]["rmse"][idx * K_EXT:(idx + 1) * K_EXT]
    tc, by = t_corregida(a, c), bayes_rope(a, c)
    y = b["y"]
    p_k1 = pd.read_parquet(RES / "ablacion_prueba.parquet")["regresion__bdl_sin_mezcla"].to_numpy(dtype=float)
    dm = est.diebold_mariano((y - p_k1) ** 2, (y - P["bdl"]) ** 2)
    bo = est.bootstrap_regresion(y, p_k1, P["bdl"], n_boot=N_BOOT)
    ab_csv = pd.read_csv(RES / "ablacion.csv").query("tarea == 'regresion' and variante == 'bdl_sin_mezcla'").iloc[0]
    fila = {"semilla": s, "rmse_completo": float(c.mean()), "desv_completo": float(c.std(ddof=1)),
            "rmse_k1": float(a.mean()), "desv_k1": float(a.std(ddof=1)), "aporte": tc["media"], "ic_inf": tc["ic_inf"],
            "ic_sup": tc["ic_sup"], "p_t": tc["p"], **by, "pliegues_a_favor": int(np.sum(c < a)),
            "prueba_completo": metricas(y, P["bdl"])["rmse"], "prueba_k1": metricas(y, p_k1)["rmse"],
            "aporte_prueba": bo["rmse"]["estimacion"], "aporte_prueba_inf": bo["rmse"]["ic_bca"][0],
            "aporte_prueba_sup": bo["rmse"]["ic_bca"][1], "dm": dm["estadistico"], "p_dm": dm["p_bilateral"],
            "p_permutacion_protocolo": float(ab_csv.p_prueba)}
    assert abs(fila["p_t"] - float(ab_csv.p_t_corregida)) < 1e-9, "la t corregida de la ablación no reproduce la del libro"
    sens = pd.read_csv(RES / "robustez" / "sensibilidad.csv").query("tarea == 'regresion'")
    barrido = sens[sens.parametro == "n_componentes"][["valor", "media", "desv", "es_el_final"]]
    mu = pd.read_csv(RES / "robustez" / "muestra.csv").query("tarea == 'regresion'")
    ms = pd.read_csv(RES / "robustez" / "muestra_sin_mezcla.csv").query("tarea == 'regresion'")
    tam = mu[mu.modelo == "bdl"][["fraccion", "n", "media", "desv"]].merge(
        ms[["fraccion", "media", "desv"]].rename(columns={"media": "media_k1", "desv": "desv_k1"}), on="fraccion")
    tam["ganancia"] = tam["media_k1"] - tam["media"]
    return pd.DataFrame([fila]), {"barrido_k": barrido, "tamano": tam}


def robustez() -> dict:
    sem = pd.read_csv(RES / "semillas.csv").query("tarea == 'regresion'")
    semillas = sem.groupby("modelo")["metrica"].agg(["mean", "std", "min", "max", "count"])
    sens = pd.read_csv(RES / "robustez" / "sensibilidad.csv").query("tarea == 'regresion'")
    ruido = pd.read_csv(RES / "robustez" / "ruido.csv").query("tarea == 'regresion'")
    ref = ruido[(ruido.escenario == "sin contaminar")].set_index("modelo")["media"]
    piv = ruido.pivot_table(index=["escenario", "nivel"], columns="modelo", values="media")
    piv["mejor"] = piv[["bdl", "xgboost", "lightgbm"]].idxmin(axis=1)
    mu = pd.read_csv(RES / "robustez" / "muestra.csv").query("tarea == 'regresion'")
    return {"semillas": semillas, "sensibilidad": sens, "ref_xgboost": float(ref["xgboost"]),
            "ref_lightgbm": float(ref["lightgbm"]), "ruido": piv.reset_index(), "muestra": mu}


def contar_parametros(tuberia) -> int | None:
    """Números que el modelo final guarda para predecir: coeficientes (lineales), nodos (árboles y conjuntos de
    árboles), celdas de las funciones de forma (EBM y BDL, más los cortes y la mezcla de BDL) y la matriz de
    entrenamiento (k-NN, que la guarda entera)."""
    mod = tuberia.named_steps["modelo"]
    nombre = type(mod).__name__
    try:
        if nombre in ("Ridge", "Lasso", "LinearSVR", "SVR", "BayesianRidge", "ElasticNet"):
            return int(np.size(mod.coef_) + np.size(mod.intercept_))
        if nombre == "DecisionTreeRegressor":
            return int(mod.tree_.node_count)
        if nombre == "RandomForestRegressor":
            return int(sum(e.tree_.node_count for e in mod.estimators_))
        if nombre == "XGBRegressor":
            return int(len(mod.get_booster().trees_to_dataframe()))
        if nombre == "LGBMRegressor":
            return int(len(mod.booster_.trees_to_dataframe()))
        if nombre == "ExplainableBoostingRegressor":
            return int(sum(np.size(s) for s in mod.term_scores_) + 1)
        if nombre == "KNeighborsRegressor":
            return int(np.size(mod._fit_X) + np.size(mod._y))
        if nombre == "RegresorBDL":
            return int(sum(np.size(f) for f in mod.formas_) + sum(np.size(c) for c in mod.cortes_) + 1
                       + np.size(mod.pi_) + np.size(mod.centros_) + 1)
    except Exception as e:  # noqa: BLE001
        log(f"no se pudieron contar los parámetros de {nombre}: {e}")
    return None


def costo(bay: dict) -> pd.DataFrame:
    import joblib
    sem = pd.read_csv(RES / "semillas.csv").query("tarea == 'regresion'").groupby("modelo")[
        ["ajuste_s", "inferencia_s", "tamano_kb"]].mean()
    mem = pd.read_csv(RES / "memoria.csv").query("tarea == 'regresion'").set_index("modelo")["memoria_mb"]
    filas = []
    for m in MODELOS:
        if m == "bayesiana":
            filas.append({"modelo": m, "ajuste_s": bay["ajuste_s"], "inferencia_s": bay["inferencia_s"],
                          "memoria_mb": np.nan, "tamano_kb": bay["tamano_kb"], "parametros": bay["parametros"]})
            continue
        ruta = next(iter(sorted((RES / "modelos").glob(f"regresion__{m}__*__43.joblib"))), None)
        parametros = contar_parametros(joblib.load(ruta)) if ruta else None
        filas.append({"modelo": m, "ajuste_s": sem.loc[m, "ajuste_s"], "inferencia_s": sem.loc[m, "inferencia_s"],
                      "memoria_mb": mem.get(m, np.nan), "tamano_kb": sem.loc[m, "tamano_kb"], "parametros": parametros})
    return pd.DataFrame(filas).set_index("modelo")


def interpretabilidad(b: dict) -> pd.DataFrame:
    """Importancia por permutación en los clientes de prueba (aumento del RMSE al barajar cada variable original, cinco
    repeticiones) de BDL y del mejor benchmark, con sus modelos finales de la semilla 43 (los dos están guardados)."""
    import joblib
    from sklearn.inspection import permutation_importance
    filas = []
    for m in ("bdl", "xgboost"):
        ruta = next(iter(sorted((RES / "modelos").glob(f"regresion__{m}__*__43.joblib"))))
        r = permutation_importance(joblib.load(ruta), b["X_te"], b["y"], scoring="neg_root_mean_squared_error",
                                   n_repeats=5, random_state=42, n_jobs=1)
        for v, mu, sd in zip(b["X_te"].columns, r.importances_mean, r.importances_std):
            filas.append({"modelo": m, "variable": v, "importancia": float(mu), "desv": float(sd)})
        log(f"importancia por permutación de {m} lista")
    return pd.DataFrame(filas)


def errores(b: dict, P: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = pd.read_csv(RES / "errores_subgrupos.csv")[["subgrupo", "clientes", "rmse_original", "rmse_benchmark"]]
    y = b["y"]
    dec = pd.qcut(y, 5, labels=False)
    filas = []
    for q in range(5):
        i = dec == q
        filas.append({"quintil": q + 1, "desde": float(y[i].min()), "hasta": float(y[i].max()), "clientes": int(i.sum()),
                      "rmse_bdl": metricas(y[i], P["bdl"][i])["rmse"], "rmse_xgboost": metricas(y[i], P["xgboost"][i])["rmse"],
                      "sesgo_bdl": float(np.mean(P["bdl"][i] - y[i])), "sesgo_xgboost": float(np.mean(P["xgboost"][i] - y[i]))})
    return sub, pd.DataFrame(filas)


# ============================================================================ cálculo completo
def calcular() -> None:
    DATOS.mkdir(exist_ok=True)
    b = cargar()
    F, P = dict(b["F"]), dict(b["P"])
    bay = bayesiana(b)
    F["bayesiana"], P["bayesiana"] = {k: bay["F"][k] for k in ("rmse", "mae", "r2")}, bay["P"]
    bay_ad = bayesiana_adicionales()
    pd.DataFrame({m: F[m]["rmse"] for m in MODELOS}).to_csv(DATOS / "pliegues_rmse.csv", index=False)
    pd.DataFrame({"y": b["y"], **{m: P[m] for m in MODELOS}}).to_csv(DATOS / "pronosticos_prueba.csv", index=False)
    des = desempeno(b, F, P)
    des.to_csv(DATOS / "desempeno.csv")
    pr, extra = pruebas(b, F, P)
    pr.to_csv(DATOS / "pruebas.csv")
    mat, cond, fr = conjuntos_adicionales(F, bay_ad)
    mat.to_csv(DATOS / "conjuntos_rmse.csv")
    cond.to_csv(DATOS / "condicion4.csv")
    ab, ab_extra = ablacion(b, F, P)
    ab.to_csv(DATOS / "ablacion.csv", index=False)
    ab_extra["barrido_k"].to_csv(DATOS / "ablacion_k.csv", index=False)
    ab_extra["tamano"].to_csv(DATOS / "ablacion_tamano.csv", index=False)
    rb = robustez()
    rb["semillas"].to_csv(DATOS / "semillas.csv")
    rb["sensibilidad"].to_csv(DATOS / "sensibilidad.csv", index=False)
    rb["ruido"].to_csv(DATOS / "ruido.csv", index=False)
    rb["muestra"].to_csv(DATOS / "muestra.csv", index=False)
    co = costo(bay)
    co.to_csv(DATOS / "costo.csv")
    imp = interpretabilidad(b)
    imp.to_csv(DATOS / "importancia.csv", index=False)
    sub, qu = errores(b, P)
    sub.to_csv(DATOS / "errores_subgrupos.csv", index=False)
    qu.to_csv(DATOS / "errores_quintiles.csv", index=False)
    import joblib
    bdl = joblib.load(next(iter(sorted((RES / "modelos").glob("regresion__bdl__*__42.joblib"))))).named_steps["modelo"]
    resid = b["y"] - P["bdl"]
    guardar_json("resumen.json", {
        "n_entrenamiento": len(b["X_tr"]), "n_prueba": len(b["y"]), "rope": ROPE, "n_boot": N_BOOT,
        "bayesiana": {k: bay[k] for k in ("control", "ajuste_s", "inferencia_s", "tamano_kb", "parametros", "alpha", "lambda")},
        "bayesiana_adicionales": bay_ad, "friedman": fr, **extra,
        "bdl_final_42": {"K": int(bdl.n_componentes_), "pi": bdl.pi_.tolist(), "centros": bdl.centros_.tolist(),
                         "sigma": float(bdl.sigma_), "arboles": int(bdl.n_arboles_), "variables": int(bdl.n_features_in_)},
        "residuos_bdl": {"media": float(resid.mean()), "desv": float(resid.std()), "asimetria": float(stats.skew(resid)),
                         "curtosis": float(stats.kurtosis(resid))},
        "techo": json.loads((RES / "techo.json").read_text(encoding="utf-8"))["pesos"],
        "optimo": {k: v for k, v in json.loads((RES / "optimo.json").read_text(encoding="utf-8"))["regresion"].items()
                   if k != "optimo"},
        "veredicto_clasificacion": json.loads((RES / "veredicto.json").read_text(encoding="utf-8")).get("clasificacion", {}),
    })
    log("datos guardados")


# ============================================================================ tablas
def tablas() -> None:
    des = pd.read_csv(DATOS / "desempeno.csv", index_col=0)
    pr = pd.read_csv(DATOS / "pruebas.csv", index_col=0)
    res = json.loads((DATOS / "resumen.json").read_text(encoding="utf-8"))

    # ---- 1. desempeño global
    orden = list(des.sort_values("rmse_media").index)
    cols = {"rmse_media": False, "mae_media": False, "r2_media": True, "rmse_prueba": False, "mae_prueba": False,
            "mape_prueba": False, "r2_prueba": True}
    marcas = {c: marcar(des[c].to_dict(), mayor, 4 if c.startswith("r2") else 5) for c, mayor in cols.items()}
    marcas["mape_prueba"] = marcar((des["mape_prueba"] * 100).to_dict(), False, 2)
    filas, filas_p = [], []
    for m in orden:
        r = des.loc[m]
        t_sig = m != "bdl" and pr.loc[m, "p_t_holm"] < ALFA
        dm_sig = m != "bdl" and pr.loc[m, "p_dm_holm"] < ALFA
        filas.append([NOMBRES[m] + ("$^\\dagger$" if t_sig else ""),
                      decorar(mas_menos(r.rmse_media, r.rmse_desv, 5), marcas["rmse_media"].get(m)),
                      decorar(mas_menos(r.mae_media, r.mae_desv, 5), marcas["mae_media"].get(m)),
                      decorar(mas_menos(r.r2_media, r.r2_desv, 4), marcas["r2_media"].get(m)),
                      "--" if m == "bdl" else f"{int(r.pliegues_bdl_gana)}/15"])
        filas_p.append([NOMBRES[m] + ("$^\\ddagger$" if dm_sig else ""),
                        decorar(num(r.rmse_prueba, 5), marcas["rmse_prueba"].get(m)),
                        f"[{num(r.rmse_inf, 5)}; {num(r.rmse_sup, 5)}]",
                        decorar(num(r.mae_prueba, 5), marcas["mae_prueba"].get(m)),
                        decorar(num(r.mape_prueba * 100, 2), marcas["mape_prueba"].get(m)),
                        decorar(num(r.r2_prueba, 4), marcas["r2_prueba"].get(m))])
    escribir_tabla(
        "desempeno.tex", "Desempeño en validación cruzada: media $\\pm$ desviación en las 15 particiones externas "
        "(tres repeticiones de cinco pliegues), en orden de RMSE.", "tab:desempeno", "lcccc",
        ["Modelo", "RMSE", "MAE", "$R^2$", "BDL gana en"], filas,
        notas="En negrita, el mejor de cada columna; subrayado, el segundo. $^\\dagger$: la $t$ corregida de Nadeau-Bengio "
              "rechaza la igualdad con BDL ($p$ de Holm $<0{,}05$ sobre diez comparaciones). La última columna cuenta las "
              "particiones en que BDL tiene menor RMSE. La regresión bayesiana se añadió después del protocolo, con los "
              "mismos pliegues: Ridge, reajustado en ellos con sus hiperparámetros guardados, reproduce su RMSE de cada "
              "pliegue con una diferencia de $10^{-17}$; sus hiperparámetros se estiman por máxima evidencia.",
        colsep="4pt")
    escribir_tabla(
        "prueba.tex", "Desempeño en los 9.745 clientes de prueba, con el intervalo bootstrap BCa del 95~\\% del RMSE "
        f"({N_BOOT} réplicas).", "tab:prueba", "lccccc",
        ["Modelo", "RMSE", "IC 95~\\% del RMSE", "MAE", "MAPE (\\%)", "$R^2$"], filas_p,
        notas="En negrita, el mejor de cada columna; subrayado, el segundo. $^\\ddagger$: Diebold-Mariano-HLN sobre el error "
              "cuadrático de cada cliente rechaza la igualdad con BDL ($p$ de Holm $<0{,}05$). El MAPE solo se calcula en "
              "prueba, porque el protocolo no guardó los pronósticos de cada pliegue.", colsep="6pt")

    # ---- 2. pruebas de significancia
    filas = []
    for m in pr.sort_values("delta_pliegues").index:
        r = pr.loc[m]
        filas.append([NOMBRES[m],
                      f"{num(r.delta_pliegues * 1e3, 3, True)} [{num(r.delta_inf * 1e3, 3)}; {num(r.delta_sup * 1e3, 3)}]",
                      num(r.t, 2), pval(r.p_t_holm), num(r.dm, 2), pval(r.p_dm_holm),
                      f"{num(r.p_mejor, 2)} / {num(r.p_rope, 2)} / {num(r.p_peor, 2)}",
                      num(r.d_cohen, 2), num(r.a12, 2)])
    escribir_tabla(
        "pruebas.tex", "Pruebas de significancia de BDL frente a cada referencia (combinación mínima de la guía para regresión "
        "tabular). $\\Delta$: RMSE de la referencia menos RMSE de BDL, $\\times 10^{3}$ (positivo: BDL se equivoca menos).",
        "tab:pruebas", "lcccccccc",
        ["Referencia", "$\\Delta$ pliegues [IC 95~\\%]", "$t$ corr.", "$p_{\\mathrm{Holm}}$", "DM-HLN", "$p_{\\mathrm{Holm}}$",
         "P(mejor / ROPE / peor)", "$d$", "$A_{12}$"], filas,
        notas="$t$ corregida de Nadeau-Bengio sobre las 15 particiones (14 grados de libertad, intervalo con la misma varianza "
              "corregida); Diebold-Mariano con la corrección de Harvey-Leybourne-Newbold sobre el error cuadrático de los "
              "9.745 clientes de prueba; $p$ ajustados con Holm sobre las diez referencias en cada prueba ($\\alpha=0{,}05$). "
              f"Prueba bayesiana: $t$ correlacionada con ROPE de $\\pm{num(ROPE, 4)[1:-1]}$ en RMSE; probabilidades posteriores de "
              "que BDL sea mejor, equivalente o peor. $d$ de Cohen y $A_{12}$ de Vargha-Delaney entre los valores por pliegue "
              "(positivo y mayor que $0{,}5$ favorecen a BDL). BDL tiene menor RMSE que cada referencia en las 15 particiones.",
        colsep="1.8pt")

    # ---- 3. conjuntos adicionales
    cond = pd.read_csv(DATOS / "condicion4.csv", index_col=0)
    fr = res["friedman"]
    filas = []
    for ds, r in cond.iterrows():
        d = 0 if ds == "california" else (1 if ds == "energia" else 0)
        filas.append([CONJ[ds], NOMBRES[r.mejor], num(r.rmse_mejor, 4 if r.rmse_mejor < 100 else 0),
                      num(r.rmse_bdl, 4 if r.rmse_bdl < 100 else 0), num(r.ventaja_relativa * 100, 2, True) + "~\\%",
                      pval(r.p_t), f"{int(r.puesto)} de {int(r.modelos)}"])
    rangos = fr["rangos"]
    escribir_tabla(
        "conjuntos.tex", "BDL en los cuatro conjuntos adicionales: RMSE medio en los cinco pliegues externos (semilla 42) del "
        "mejor benchmark de cada conjunto y de BDL, ventaja relativa (positiva si BDL es mejor), $p$ de la $t$ corregida y "
        "puesto entre los modelos.", "tab:conjuntos", "llccccc",
        ["Conjunto", "Mejor benchmark", "RMSE benchmark", "RMSE BDL", "Ventaja", "$p$ ($t$ corr.)", "Puesto"], filas,
        notas=f"Friedman con los {fr['n_conjuntos']} conjuntos como bloques y {len(fr['modelos'])} modelos: "
              f"$\\chi^2={num(fr['chi2'], 2)[1:-1]}$, $p={pval(fr['p_friedman'])[1:-1]}$; Iman-Davenport $p={pval(fr['p_iman_davenport'])[1:-1]}$. "
              f"Rango medio de BDL: {num(rangos['bdl'], 1)}; diferencia crítica de Nemenyi: {num(fr['dc'], 2)} "
              "(Figura~\\ref{fig:cd}).")

    # ---- 4. ablación
    ab = pd.read_csv(DATOS / "ablacion.csv").iloc[0]
    kk = pd.read_csv(DATOS / "ablacion_k.csv")
    filas = [["BDL completo", mas_menos(ab.rmse_completo, ab.desv_completo, 5), "--", "--", "--",
              num(ab.prueba_completo, 5), "--", "--"],
             ["Sin mezcla ($K=1$)", mas_menos(ab.rmse_k1, ab.desv_k1, 5),
              f"{num(ab.aporte * 1e3, 3, True)} [{num(ab.ic_inf * 1e3, 3)}; {num(ab.ic_sup * 1e3, 3)}]",
              pval(ab.p_t), num(ab.p_rope, 4), num(ab.prueba_k1, 5), num(ab.aporte_prueba * 1e3, 3, True), pval(ab.p_dm)]]
    ref_bic = float(kk[kk.valor.astype(str) == "auto"].media.iloc[0])      # BDL con K por BIC, mismos ajustes sin búsqueda
    for _, r in kk.iterrows():
        etiqueta = "$K$ por BIC (final)" if str(r.valor) == "auto" else f"$K={r.valor}$ fijo"
        filas.append([etiqueta, mas_menos(r.media, r.desv, 5),
                      "--" if str(r.valor) == "auto" else num((r.media - ref_bic) * 1e3, 3, True), "--", "--", "--", "--", "--"])
    escribir_tabla(
        "ablacion.tex", "Ablación de la mezcla latente en regresión: RMSE en los cinco pliegues externos de la semilla 42 y en "
        "prueba. Aporte: RMSE de la variante menos RMSE de BDL completo, $\\times 10^{3}$ (positivo: la mezcla ayuda).",
        "tab:ablacion", "lccccccc",
        ["Variante", "RMSE (pliegues)", "Aporte [IC 95~\\%]", "$p$ ($t$ corr.)", "P(ROPE)", "RMSE prueba", "Aporte prueba",
         "$p$ (DM)"], filas, lineas_medias=[2],
        notas="Las dos primeras filas son la ablación registrada, con las mismas pruebas de la comparación principal "
              "(hiperparámetros elegidos por la búsqueda en cada pliegue). Las demás vienen del análisis de sensibilidad: "
              "la configuración final con $K$ fijo o elegido por BIC, en los mismos pliegues y sin nueva búsqueda; son "
              "exploratorias y su aporte se mide frente a la última fila (positivo: la variante es peor).", colsep="1.8pt")

    # ---- 5. robustez
    sem = pd.read_csv(DATOS / "semillas.csv", index_col=0)
    sens = pd.read_csv(DATOS / "sensibilidad.csv")
    ruido = pd.read_csv(DATOS / "ruido.csv")
    mu = pd.read_csv(DATOS / "muestra.csv")
    ref = float(ruido[ruido.escenario == "sin contaminar"]["xgboost"].iloc[0])
    filas = []
    for m in ("bdl", "xgboost", "lightgbm", "ebm", "random_forest"):
        r = sem.loc[m]
        filas.append([NOMBRES[m], num(r["mean"], 6), num(r["std"] * 1e6, 1), f"{int(r['count'])}"])
    escribir_tabla("semillas.tex", "Variación entre semillas del modelo final: RMSE en prueba al reajustar con diez semillas.",
                   "tab:semillas", "lccc", ["Modelo", "RMSE medio", "Desviación ($\\times 10^{-6}$)", "Semillas"], filas)
    guardar_json("robustez_resumen.json", {
        "sens_configuraciones": int(len(sens)), "sens_peor": float(sens.media.max()), "sens_mejor": float(sens.media.min()),
        "sens_mejores_que_xgboost": int((sens.media < ref).sum()), "ref_xgboost": ref,
        "ruido_escenarios": int(len(ruido)), "ruido_bdl_mejor": int((ruido.mejor == "bdl").sum()),
        "muestra": mu.pivot_table(index="n", columns="modelo", values="media").round(6).to_dict()})

    # ---- 6. costo
    co = pd.read_csv(DATOS / "costo.csv", index_col=0)
    filas = []
    for m in co.sort_values("ajuste_s").index:
        r = co.loc[m]
        filas.append([NOMBRES[m], num(r.ajuste_s, 2), num(r.inferencia_s, 3), num(r.memoria_mb, 0) if not np.isnan(r.memoria_mb) else "--",
                      num(r.tamano_kb, 0), ent(r.parametros) if not np.isnan(r.parametros) else "--"])
    escribir_tabla(
        "costo.tex", "Costo computacional del modelo final de cada referencia: ajuste con todo el entrenamiento (38.978 "
        "clientes) e inferencia sobre los 9.745 de prueba (medias de diez semillas), memoria pico del ajuste, tamaño "
        "serializado y número de parámetros.", "tab:costo", "lccccc",
        ["Modelo", "Ajuste (s)", "Inferencia (s)", "Memoria (MB)", "Tamaño (KB)", "Parámetros"], filas,
        notas="Equipo: AMD Ryzen 5 3500X (6 núcleos), 16 GB de RAM, Windows 11, sin GPU; Python 3.13.3. Parámetros: "
              "coeficientes en los modelos lineales; nodos en los árboles y conjuntos de árboles; celdas de las funciones de "
              "forma en la EBM; en BDL, las celdas de sus 78 funciones de forma (los 2.020 tocones se acumulan en ellas), "
              "los cortes y la mezcla; en k-NN, la matriz de entrenamiento que guarda. La memoria de la regresión bayesiana "
              "no se midió con el mismo procedimiento.", colsep="4pt")

    # ---- 7. errores por subgrupo y por quintil del objetivo
    sub = pd.read_csv(DATOS / "errores_subgrupos.csv")
    qu = pd.read_csv(DATOS / "errores_quintiles.csv")
    filas = []
    for _, r in sub.iterrows():
        filas.append([r.subgrupo.replace("tx_count", "\\texttt{tx\\_count}").replace("≤", "$\\leq$").replace(">", "$>$"),
                      ent(r.clientes), num(r.rmse_original, 5), num(r.rmse_benchmark, 5),
                      num((r.rmse_benchmark - r.rmse_original) * 1e3, 3, True)])
    for _, r in qu.iterrows():
        filas.append([f"Quintil {int(r.quintil)} del objetivo [{num(r.desde, 3)}; {num(r.hasta, 3)}]", ent(r.clientes),
                      num(r.rmse_bdl, 5), num(r.rmse_xgboost, 5), num((r.rmse_xgboost - r.rmse_bdl) * 1e3, 3, True)])
    escribir_tabla(
        "errores.tex", "Error en prueba por subgrupo de clientes y por quintil del objetivo: BDL frente a XGBoost. "
        "Diferencia: RMSE de XGBoost menos RMSE de BDL, $\\times 10^{3}$ (positiva: BDL se equivoca menos).",
        "tab:errores", "lcccc", ["Subgrupo", "Clientes", "RMSE BDL", "RMSE XGBoost", "Diferencia"], filas,
        lineas_medias=[1, 7, 9, 13], colsep="5pt")


# ============================================================================ tabla de hiperparámetros (metodología)
ESPACIOS = {
    "bdl": "$\\nu \\in [0{,}05;\\,1]$ (log); $d \\in [1, 6]$; mín. por hoja $\\in [5, 200]$ (log); $\\lambda \\in [0{,}01;\\,10]$ "
           "(log); bins $M \\in [32, 1024]$ (log); $K \\in \\{$BIC$, 1\\}$",
    "xgboost": "árboles $\\in [100, 1000]$ (log); profundidad $\\in [2, 10]$; tasa $\\in [10^{-3};\\,0{,}3]$ (log); "
               "submuestreo de filas y de columnas $\\in [0{,}5;\\,1]$; $\\lambda \\in [0{,}01;\\,10]$ (log); peso mín. por "
               "hoja $\\in [1, 50]$ (log)",
    "lightgbm": "árboles $\\in [100, 1000]$ (log); hojas $\\in [4, 127]$ (log); tasa $\\in [10^{-3};\\,0{,}3]$ (log); "
                "submuestreo de filas y de columnas $\\in [0{,}5;\\,1]$; $\\lambda \\in [0{,}01;\\,10]$ (log); mín. por hoja "
                "$\\in [5, 200]$ (log)",
    "ebm": "tasa $\\in [0{,}008;\\,0{,}1]$ (log); hojas $\\in \\{2, 3, 4\\}$; mín. por hoja $\\in [2, 200]$ (log); "
           "interacciones $\\in [0, 20]$; bins $\\in [64, 1024]$ (log)",
    "random_forest": "profundidad $\\in [4, 40]$; variables por corte $\\in \\{\\sqrt{p}, \\log_2 p, 0{,}3, 0{,}5\\}$; mín. por "
                     "hoja $\\in [1, 100]$ (log); 100 árboles en la búsqueda y 300 en el final",
    "arbol": "profundidad $\\in [3, 30]$; mín. por hoja $\\in [1, 300]$ (log); poda $\\alpha_{\\mathrm{ccp}} \\in "
             "[10^{-8}, 10^{-4}]$ (log)",
    "ridge": "$\\alpha \\in [10^{-3}, 10^{5}]$ (log)",
    "lasso": "$\\alpha \\in [10^{-6}, 10^{-1}]$ (log)",
    "bayesiana": "sin búsqueda: las precisiones del ruido y de los pesos se estiman por máxima evidencia en cada ajuste",
    "svr": "$C \\in [10^{-3}, 10^{2}]$ (log); $\\varepsilon \\in [10^{-4};\\,0{,}05]$ (log)",
    "knn": "$k \\in [5, 600]$ (entero, log); pesos uniformes o por distancia; métrica $L_q$, $q \\in \\{1, 2\\}$",
}
PARAMETROS = {   # nombre en la tabla maestra -> (rótulo, decimales; None = entero o texto)
    "tasa": ("$\\nu$", 4), "profundidad": ("$d$", None), "min_hoja": ("mín. hoja", None), "lambda_l2": ("$\\lambda$", 4),
    "max_bins": ("$M$", None), "n_componentes": ("$K$", None), "n_estimators": ("árboles", None),
    "max_depth": ("prof.", None), "learning_rate": ("tasa", 4), "subsample": ("filas", 2),
    "colsample_bytree": ("columnas", 2), "reg_lambda": ("$\\lambda$", 3), "min_child_weight": ("peso mín.", None),
    "num_leaves": ("hojas", None), "min_child_samples": ("mín. hoja", None), "interactions": ("interacciones", None),
    "max_leaves": ("hojas", None), "min_samples_leaf": ("mín. hoja", None), "outer_bags": ("bolsas", None),
    "max_features": ("variables", 2), "ccp_alpha": ("$\\alpha_{\\mathrm{ccp}}$", None), "alpha": ("$\\alpha$", 1),
    "C": ("$C$", 2), "epsilon": ("$\\varepsilon$", None), "n_neighbors": ("$k$", None), "p": ("$q$", None),
    "weights": ("pesos", None), "max_bins_ebm": ("bins", None),
}
OPTIMIZADORES = {"grid": "rejilla", "random": "aleatoria", "bayesiana": "TPE", "genetica": "genética"}


def _valor(v, decimales):
    if isinstance(v, str):
        return {"auto": "BIC", "distance": "distancia", "uniform": "uniforme"}.get(v, v)
    if isinstance(v, bool):
        return "sí" if v else "no"
    v = float(v)
    if decimales is None and float(v).is_integer():
        return ent(v)
    if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4):
        mant, exp = f"{v:.1e}".split("e")
        return f"${mant.replace('.', '{,}')} \\times 10^{{{int(exp)}}}$"
    return num(v, decimales if decimales is not None else 4)[1:-1].join(("$", "$"))


def tabla_hiperparametros() -> None:
    t = pd.read_parquet(RES / "experimentos.parquet")
    reg = t[(t.tarea == "regresion") & (t.estado == "ok") & (t.semilla == SEMILLAS[0])].set_index("modelo")
    res = json.loads((DATOS / "resumen.json").read_text(encoding="utf-8"))
    filas = []
    for m in MODELOS:
        if m == "bayesiana":
            b = res["bayesiana"]
            filas.append([NOMBRES[m], "--", ESPACIOS[m], f"$\\hat\\alpha = {num(b['alpha'], 1)[1:-1]}$; "
                          f"$\\hat\\lambda = {ent(b['lambda'])}$ (precisiones estimadas)"])
            continue
        r = reg.loc[m]
        h = leer(r.hiperparametros)
        partes = []
        for k, v in h.items():
            nombre = k.split("__", 1)[-1]
            rot, dec = PARAMETROS.get(nombre, (nombre.replace("_", "\\_"), 4))
            partes.append(f"{rot} = {_valor(v, dec)}")
        filas.append([NOMBRES[m], f"{OPTIMIZADORES.get(r.optimizador, r.optimizador)} ({int(r.presupuesto)})",
                      ESPACIOS[m], "; ".join(partes)])
    escribir_tabla(
        "hiperparametros.tex", "Espacios de búsqueda e hiperparámetros finales de cada modelo de regresión: optimizador "
        "elegido con su presupuesto (evaluaciones por búsqueda) y valores que fijó la búsqueda sobre todo el entrenamiento "
        "(repetición de la semilla 42).", "tab:hiperparametros",
        ">{\\raggedright\\arraybackslash}p{0.13\\linewidth} >{\\raggedright\\arraybackslash}p{0.09\\linewidth} "
        ">{\\raggedright\\arraybackslash}p{0.38\\linewidth} >{\\raggedright\\arraybackslash}p{0.32\\linewidth}",
        ["Modelo", "Búsqueda", "Espacio de búsqueda", "Hiperparámetros finales"], filas,
        notas="(log): búsqueda en escala logarítmica. El presupuesto es el número de puntos de la rejilla del modelo y "
              "se aplica igual con los cuatro optimizadores (rejilla, aleatoria, TPE y genética); el optimizador de cada "
              "benchmark es el mejor de los cuatro en el Entregable 3, y el de BDL y LightGBM quedó fijado en el "
              "protocolo. En BDL, $\\nu$ es la tasa de aprendizaje, $d$ la profundidad de los árboles, $\\lambda$ la "
              "penalización L2 de las hojas, $M$ el número máximo de intervalos por variable y $K$ el número de clases "
              "latentes (BIC: elegido por el criterio BIC; con $K=1$ no hay mezcla).", colsep="3pt")


def figura_pipeline_y_objetivo() -> None:
    """Figura del pipeline del proyecto final (nueva) y copia de la figura del objetivo de la metodología anterior, que
    genera ../Metodologia Entregable 3/figuras.py a partir de COFINFAD (en inglés y con el mismo formato)."""
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    FIGURAS.mkdir(exist_ok=True)
    candidatas = (AQUI.parent / "Metodologia Entregable 3" / "figures" / "fig_target.pdf",   # carpeta de trabajo
                  AQUI.parent / "metodologia" / "figures" / "fig_target.pdf")                 # InformeFinal/ del repositorio
    origen = next((c for c in candidatas if c.exists()), candidatas[0])
    if origen.exists():
        shutil.copy2(origen, FIGURAS / "fig_target.pdf")
        log("figura fig_target.pdf (copiada de la metodología anterior)")
    elif not (FIGURAS / "fig_target.pdf").exists():
        sys.exit("Falta figures/fig_target.pdf: genérela con ../Metodologia Entregable 3/figuras.py")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10.5,
                         "pdf.fonttype": 42})
    NAVY, AMBER, GREEN = "#1F3A5F", "#E0A526", "#2E8B57"
    CLARO, MEDIO, FUERTE = "#EAF0F7", "#D5E0EE", "#EDC24A"
    fig = plt.figure(figsize=(6.3, 7.6 * 0.87))        # misma escala vertical; se omite el margen inferior vacío
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.13, 1)
    ax.axis("off")
    pasos = [
        ("COFINFAD: 48,723 clients × 54 columns; 3,159,157 transactions (2023)", CLARO),
        ("80/20 split by index, seed 42, before anything else: 38,978 train / 9,745 test;\n"
         "label threshold τ and recency cut-off estimated on the training clients only", CLARO),
        ("ETL: 12 transaction features per client; 17 columns removed (active_products: leakage);\n"
         "48 predictors → 78 columns (imputation, scaling and encoding inside each pipeline)", CLARO),
        ("11 regression models: BDL (proposed); k-NN, Ridge, Lasso, Bayesian ridge, tree,\n"
         "random forest, XGBoost, linear SVR (course); EBM and LightGBM (state of the art)", MEDIO),
        ("each model: nested CV, 3 repetitions (seeds 42, 43, 44) × 5 outer × 3 inner folds,\n"
         "30–48 evaluations per search → final refit → test set scored once", FUERTE),
        ("pre-registered protocol (Git tag): corrected t, Diebold–Mariano–HLN, Bayesian ROPE,\n"
         "Holm; 4 additional datasets; Friedman with datasets as blocks", CLARO),
    ]
    y0, hueco, alto = 0.985, 0.022, {1: 0.052, 2: 0.078}
    bordes = []
    for texto, color in pasos:
        h = alto[texto.count("\n") + 1]
        ax.add_patch(FancyBboxPatch((0.02, y0 - h), 0.96, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=color, ec=NAVY, lw=1.6))
        ax.text(0.5, y0 - h / 2, texto, ha="center", va="center", linespacing=1.3, fontsize=9.6)
        bordes.append(y0 - h)
        y0 = y0 - h - hueco
    for i in range(len(pasos) - 1):
        ax.add_patch(FancyArrowPatch((0.5, bordes[i]), (0.5, bordes[i] - hueco), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.6, color=NAVY, shrinkA=0, shrinkB=0))
    base = y0 - 0.01
    ax.text(0.02, base, "Nested cross-validation inside the training set (one repetition):", ha="left", va="top")
    fila1 = base - 0.07
    for k in range(5):
        ax.add_patch(FancyBboxPatch((0.02 + k * 0.96 / 5 + 0.004, fila1), 0.96 / 5 - 0.008, 0.035,
                                    boxstyle="square,pad=0", fc=AMBER if k == 1 else NAVY, ec="none"))
    ax.text(0.02, fila1 - 0.012, "outer loop: 5 folds, the same for every model; fold 2 shown as validation;\n"
                                 "repeated with seeds 42, 43 and 44 → 15 outer folds per model", ha="left", va="top")
    fila2 = fila1 - 0.12
    for j in range(3):
        ax.add_patch(FancyBboxPatch((0.02 + j * 0.96 * 0.8 / 3 + 0.004, fila2), 0.96 * 0.8 / 3 - 0.008, 0.035,
                                    boxstyle="square,pad=0", fc=GREEN if j == 1 else MEDIO, ec=NAVY, lw=1.6))
    ax.text(0.02, fila2 - 0.012, "inner loop: the four outer training folds, pooled, are split into 3 folds that\n"
                                 "choose the hyperparameters (grid, random, TPE or genetic search; same budget)",
            ha="left", va="top")
    fig.savefig(FIGURAS / "fig_pipeline.pdf")
    plt.close(fig)
    log("figura fig_pipeline.pdf")


# ============================================================================ figuras
def figuras() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pypdf import PdfReader

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
        "font.size": 10, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "lines.linewidth": 1.8, "axes.grid": True, "grid.alpha": 0.3, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    NAVY, AMBER, GREEN, RED, GREY = "#1F3A5F", "#E0A526", "#2E8B57", "#B03A2E", "#6E6E6E"
    fallos = []

    def guardar(fig, nombre):
        FIGURAS.mkdir(exist_ok=True)
        ruta = FIGURAS / nombre
        fig.savefig(ruta)
        plt.close(fig)
        caja = PdfReader(str(ruta)).pages[0].mediabox
        ancho = float(caja.width) / 72
        if ancho > 6.3 + 1e-3:
            fallos.append(f"{nombre}: {ancho:.2f} in")
        log(f"figura {nombre} ({ancho:.2f} x {float(caja.height) / 72:.2f} in)")

    pl = pd.read_csv(DATOS / "pliegues_rmse.csv")
    pr = pd.read_csv(DATOS / "pruebas.csv", index_col=0)
    res = json.loads((DATOS / "resumen.json").read_text(encoding="utf-8"))

    # ---- 1. RMSE por pliegue y modelo
    orden = list(pl.mean().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(6.3, 3.6), layout="constrained")
    caja = ax.boxplot([pl[m] * 1e3 for m in orden], vert=False, widths=0.6, patch_artist=True,
                      medianprops={"color": "black"}, flierprops={"markersize": 3})
    for patch, m in zip(caja["boxes"], orden):
        patch.set_facecolor(AMBER if m == "bdl" else "#C9D3E0")
    for i, m in enumerate(orden, start=1):
        ax.scatter(pl[m] * 1e3, np.full(len(pl), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(pl)),
                   s=6, color=NAVY, alpha=0.6, zorder=3)
    ax.set_yticks(range(1, len(orden) + 1), [NOMBRES_EN[m] for m in orden])
    ax.set_xlabel("Outer-fold RMSE (× 10⁻³), 15 folds (3 repetitions × 5 folds)")
    guardar(fig, "fig_folds.pdf")

    # ---- 2. intervalos de las diferencias frente a cada referencia
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), layout="constrained")
    grupos = (["xgboost", "lightgbm", "ebm", "random_forest", "arbol"], ["ridge", "lasso", "bayesiana", "svr", "knn"])
    for ax, grupo, titulo in zip(axes, grupos, ("Tree-based and state of the art", "Linear and k-NN")):
        grupo = sorted(grupo, key=lambda m: pr.loc[m, "delta_pliegues"])
        yy = np.arange(len(grupo))
        for k, (col, lo, hi, color, desp, rot) in enumerate((
                ("delta_pliegues", "delta_inf", "delta_sup", NAVY, -0.12, "Folds (corrected t)"),
                ("delta_prueba", "delta_prueba_inf", "delta_prueba_sup", AMBER, 0.12, "Test set (BCa bootstrap)"))):
            v = pr.loc[grupo, col].to_numpy() * 1e3
            err = np.vstack([v - pr.loc[grupo, lo].to_numpy() * 1e3, pr.loc[grupo, hi].to_numpy() * 1e3 - v])
            ax.errorbar(v, yy + desp, xerr=err, fmt="o" if k == 0 else "s", color=color, ms=4, capsize=2, label=rot)
        ax.axvspan(-ROPE * 1e3, ROPE * 1e3, color=GREY, alpha=0.15, lw=0, label="ROPE (±0.5 × 10⁻³)")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(yy, [NOMBRES_EN[m] for m in grupo])
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("ΔRMSE (× 10⁻³)")
    axes[1].set_xlim(-0.6, 5.0)
    manejadores, rotulos = axes[0].get_legend_handles_labels()
    fig.legend(manejadores, rotulos, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    guardar(fig, "fig_intervals.pdf")

    # ---- 3. diagrama de diferencia crítica (conjuntos como bloques)
    fr = res["friedman"]
    rangos = pd.Series(fr["rangos"]).sort_values()
    k = len(rangos)
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.set_xlim(-4.4, k + 4.4)
    ax.set_ylim(0, 1.1)
    ax.axis("off")
    ax.plot([1, k], [0.82, 0.82], color="black", lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0.82, 0.85], color="black", lw=1)
        ax.text(r, 0.88, str(r), ha="center", va="bottom", fontsize=9)
    mitad = (k + 1) // 2
    for i, (m, r) in enumerate(rangos.items()):
        izquierda = i < mitad
        yl = 0.70 - (i if izquierda else (k - 1 - i)) * 0.105
        xl = 0.4 if izquierda else k + 0.6
        color = AMBER if m == "bdl" else "black"
        ax.plot([r, r, xl], [0.82, yl, yl], color=color, lw=1.2 if m == "bdl" else 0.8)
        ax.text(xl + (-0.05 if izquierda else 0.05), yl, f"{'BDL' if m == 'bdl' else NOMBRES_EN[m]} ({r:.1f})",
                ha="right" if izquierda else "left",
                va="center", fontsize=8.5, color=color, fontweight="bold" if m == "bdl" else "normal")
    dc = fr["dc"]
    ax.plot([1, 1 + dc], [0.97, 0.97], color=RED, lw=2)
    ax.text(1 + dc / 2, 0.985, f"CD = {dc:.2f} (Nemenyi, α = 0.05)", ha="center", va="bottom", fontsize=8.5, color=RED)
    ax.text(k / 2 + 0.5, 0.02, f"Mean rank over {fr['n_conjuntos']} datasets (1 = lowest RMSE); "
            f"Friedman p = {fr['p_friedman']:.3f}", ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    guardar(fig, "fig_cd.pdf")

    # ---- 4. pronosticado frente a observado y residuos
    pron = pd.read_csv(DATOS / "pronosticos_prueba.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), layout="constrained")
    hb = axes[0].hexbin(pron["bdl"], pron["y"], gridsize=45, bins="log", cmap="Blues", mincnt=1)
    lim = [min(pron["y"].min(), pron["bdl"].min()), max(pron["y"].max(), pron["bdl"].max())]
    axes[0].plot(lim, lim, color=RED, lw=1, ls="--")
    axes[0].set_xlabel("BDL prediction")
    axes[0].set_ylabel("Observed churn_probability")
    fig.colorbar(hb, ax=axes[0], label="Customers (log)")
    for m, color in (("bdl", AMBER), ("xgboost", NAVY)):
        axes[1].hist(pron["y"] - pron[m], bins=80, histtype="step", color=color, lw=1.4, label=NOMBRES_EN[m], density=True)
    axes[1].set_xlabel("Residual (obs. − pred.)")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False)
    guardar(fig, "fig_regression.pdf")

    # ---- 5. ablación
    kk = pd.read_csv(DATOS / "ablacion_k.csv")
    tam = pd.read_csv(DATOS / "ablacion_tamano.csv")
    ruido = pd.read_csv(DATOS / "ruido.csv")
    ref = float(ruido[ruido.escenario == "sin contaminar"]["xgboost"].iloc[0])
    ab = pd.read_csv(DATOS / "ablacion.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), layout="constrained")
    valores = np.array([ab.aporte, ab.aporte_prueba]) * 1e3
    errores_ab = np.array([[ab.aporte - ab.ic_inf, ab.aporte_prueba - ab.aporte_prueba_inf],
                           [ab.ic_sup - ab.aporte, ab.aporte_prueba_sup - ab.aporte_prueba]]) * 1e3
    axes[0].bar([0, 1], valores, yerr=errores_ab, color=[NAVY, AMBER], capsize=4, edgecolor="black", lw=0.5, width=0.55)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks([0, 1], ["Outer folds\n(corrected t CI)", "Test set\n(BCa bootstrap CI)"])
    axes[0].set_ylabel("RMSE(K=1) − RMSE(BDL)  (× 10⁻³)")
    axes[0].set_title("Removing the latent mixture", fontsize=10)
    axes[1].errorbar(tam.n, tam.media * 1e3, yerr=tam.desv * 1e3 / np.sqrt(5), color=AMBER, marker="o", ms=4, capsize=2,
                     label="BDL (K by BIC)")
    axes[1].errorbar(tam.n, tam.media_k1 * 1e3, yerr=tam.desv_k1 * 1e3 / np.sqrt(5), color=GREY, marker="s", ms=4,
                     capsize=2, label="BDL without mixture (K=1)")
    axes[1].set_xscale("log")
    axes[1].set_xticks(tam.n, [f"{n / 1000:.1f}k" for n in tam.n])
    axes[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[1].set_xlabel("Training rows")
    axes[1].set_ylabel("Fold RMSE (× 10⁻³)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Mixture gain vs. sample size", fontsize=10)
    guardar(fig, "fig_ablation.pdf")

    # ---- 6. sensibilidad: curva de aprendizaje e hiperparámetros
    mu = pd.read_csv(DATOS / "muestra.csv")
    sens = pd.read_csv(DATOS / "sensibilidad.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), layout="constrained")
    for m, color, mk in (("bdl", AMBER, "o"), ("xgboost", NAVY, "s"), ("lightgbm", GREEN, "^")):
        d = mu[mu.modelo == m].sort_values("n")
        axes[0].errorbar(d.n, d.media * 1e3, yerr=d.desv * 1e3 / np.sqrt(5), color=color, marker=mk, ms=4, capsize=2,
                         label=NOMBRES_EN[m])
    axes[0].set_xscale("log")
    ns = sorted(mu.n.unique())
    axes[0].set_xticks(ns, [f"{n / 1000:.1f}k" for n in ns])
    axes[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_xlabel("Training rows")
    axes[0].set_ylabel("Fold RMSE (× 10⁻³)")
    axes[0].set_title("Learning curve", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    params = list(dict.fromkeys(sens.parametro))
    EN = {"tasa": "learning rate", "profundidad": "depth", "min_hoja": "min. leaf size", "lambda_l2": "L2 penalty",
          "max_bins": "max. bins", "n_componentes": "latent classes K", "rondas_calentamiento": "warm-up rounds"}
    for i, p in enumerate(params):
        d = sens[sens.parametro == p]
        axes[1].scatter(d.media * 1e3, np.full(len(d), i), color=NAVY, s=14, zorder=3)
        fin = d[d.es_el_final]
        axes[1].scatter(fin.media * 1e3, np.full(len(fin), i), color=AMBER, marker="D", s=30, zorder=4)
    axes[1].axvline(ref * 1e3, color=RED, ls="--", lw=1, label="XGBoost (tuned)")
    axes[1].set_yticks(range(len(params)), [EN.get(p, p) for p in params])
    axes[1].set_xlabel("Fold RMSE (× 10⁻³)")
    axes[1].set_title("One-at-a-time hyperparameters", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    guardar(fig, "fig_sensitivity.pdf")

    # ---- 7. importancia por permutación
    imp = pd.read_csv(DATOS / "importancia.csv")
    top = (imp.groupby("variable")["importancia"].max().sort_values(ascending=False).head(8).index)[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 2.9), layout="constrained")
    yy = np.arange(len(top))
    for desp, m, color in ((-0.18, "bdl", AMBER), (0.18, "xgboost", NAVY)):
        d = imp[imp.modelo == m].set_index("variable").loc[top]
        ax.barh(yy + desp, d.importancia * 1e3, height=0.36, xerr=d.desv * 1e3, color=color, label=NOMBRES_EN[m],
                error_kw={"lw": 0.8, "capsize": 1.5})
    ax.set_yticks(yy, [t.replace("_", "\\_") if False else t for t in top])
    ax.set_xlabel("Permutation importance: RMSE increase on the test set (× 10⁻³)")
    ax.legend(frameon=False, loc="lower right")
    guardar(fig, "fig_importance.pdf")

    if fallos:
        sys.exit("Figuras fuera de formato:\n" + "\n".join(fallos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tablas", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    a = ap.parse_args()
    if not (a.tablas or a.figuras):
        calcular()
    if a.tablas or not a.figuras:
        tablas()
        tabla_hiperparametros()
    if a.figuras or not a.tablas:
        figuras()
        figura_pipeline_y_objetivo()


if __name__ == "__main__":
    main()
