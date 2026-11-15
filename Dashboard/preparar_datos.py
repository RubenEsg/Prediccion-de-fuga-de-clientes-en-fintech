"""Arma la carpeta datos/ de la app a partir del proyecto entregado, que solo se lee.

    "../entrega 3/.venv/Scripts/python.exe" -B preparar_datos.py

Hay que correrlo con el Python del Entregable 3: necesita su paquete src, imbalanced-learn, interpret-core y scikit-learn
para abrir los modelos guardados, y los modelos de resultados/modelos/ (490 MB), que no están en el repositorio publicado.
Por eso datos/ se entrega ya armada y la app solo necesita dash, plotly, pandas, numpy y pyarrow.

Qué deja en datos/ (unos 9 MB):
  modelos/        las dos EBM finales y la regresión logística exportadas a JSON (lo que evalúa motor.py) y las dos EBM como
                  tubería de scikit-learn sin clases propias (para verificar_motor.py)
  clientes_prueba.parquet, puntajes_prueba.parquet, pronosticos_prueba.parquet, calibracion_isotonica_ebm.json
  variables.csv   las 48 variables: nombre visible, papel en el simulador, rangos de entrenamiento, importancia
  eda/            resúmenes del análisis exploratorio, calculados solo con el entrenamiento
  resultados/     las tablas de la sección de resultados del informe (copia exacta)
  entrega3/       tablas pequeñas del Entregable 3, la tabla de corridas sin rutas locales y las curvas de los optimizadores
  metadatos.json, origen.json (de dónde salió cada archivo, con su huella)

Las carpetas de origen se buscan al lado (../entrega 3 y ../Resultados Entregable 3), en la disposición del repositorio
(../../Entrega3 y ../resultados) o en las variables de entorno ENTREGA3 y RESULTADOS_E3.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
DESTINO = AQUI / "datos"
TAREAS = ("clasificacion", "regresion")
CAPACIDAD = 0.25
SEMILLA = 42


def _carpeta(variable: str, candidatas: list[Path], prueba: str, nombre: str) -> Path:
    for c in [os.environ.get(variable), *candidatas]:
        if c and (Path(c) / prueba).exists():
            return Path(c).resolve()
    sys.exit(f"No se encontró {nombre} (se buscó {prueba} en {[str(c) for c in candidatas]}); fije la variable {variable}")


E3 = _carpeta("ENTREGA3", [AQUI.parent / "entrega 3", AQUI.parent.parent / "Entrega3"], "resultados/experimentos.parquet",
              "la carpeta del Entregable 3")
RES = _carpeta("RESULTADOS_E3", [AQUI.parent / "Resultados Entregable 3", AQUI.parent / "resultados"], "datos/puntajes_prueba.csv",
               "la carpeta de la sección de resultados")
if not (E3 / "resultados" / "modelos").is_dir():
    sys.exit("Faltan los modelos guardados (resultados/modelos/ del Entregable 3): no están en el repositorio publicado. "
             "La carpeta datos/ ya viene armada; este guion solo hace falta para regenerarla desde la carpeta de trabajo.")

sys.path.insert(0, str(E3))            # paquete src: el ETL y la clase EBMConPesos de los modelos guardados
sys.path.insert(0, str(AQUI))          # motor.py (para comprobar la exportación)

import joblib  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline as PipelineSk  # noqa: E402

import motor  # noqa: E402
from src import analisis, datos  # noqa: E402

T0 = time.perf_counter()
ORIGEN: dict[str, dict] = {}           # archivo de datos/ -> de dónde salió y cómo


def anotar(rel: str, de: list[str] | str, como: str) -> None:
    ORIGEN[rel.replace("\\", "/")] = {"de": [de] if isinstance(de, str) else list(de), "como": como}


def _a_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def escribir_json(rel: str, objeto, de, como: str, sangria: int | None = None) -> None:
    ruta = DESTINO / rel
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=sangria, default=_a_json), encoding="utf-8")
    anotar(rel, de, como)


def escribir_csv(rel: str, tabla: pd.DataFrame, de, como: str, indice: bool = False) -> None:
    ruta = DESTINO / rel
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=indice, encoding="utf-8", lineterminator="\n")
    anotar(rel, de, como)


def copiar(origen: Path, rel: str, raiz: Path, nombre_raiz: str) -> None:
    ruta = DESTINO / rel
    ruta.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origen, ruta)
    anotar(rel, f"{nombre_raiz}: {origen.relative_to(raiz).as_posix()}", "copia exacta")


# ============================================================================ exportación de los modelos a tablas
def _partes(pipe):
    pre = pipe.named_steps["preprocesado"]
    est = pipe.named_steps["modelo"]
    est = getattr(est, "ebm_", est)                       # EBMConPesos guarda la EBM de interpret en .ebm_
    (nombre_num, esc, num_cols), (nombre_cat, ohe, cat_cols) = pre.transformers_[0], pre.transformers_[1]
    assert (nombre_num, nombre_cat) == ("num", "cat")
    salida = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    terminos, j = [], 0
    for k, v in enumerate(num_cols):
        assert salida[j] == v
        terminos.append({"termino": salida[j], "variable": v, "clase": "numerica", "indice": j,
                         "media": float(esc.mean_[k]), "escala": float(esc.scale_[k])})
        j += 1
    for k, v in enumerate(cat_cols):
        cats = list(ohe.categories_[k])
        assert ohe.drop_idx_[k] == 0
        for c in cats[1:]:
            assert salida[j] == f"{v}_{c}", (salida[j], v, c)
            terminos.append({"termino": salida[j], "variable": v, "clase": "indicadora", "indice": j, "categoria": str(c)})
            j += 1
    assert j == len(salida)
    categoricas = {v: [str(c) for c in ohe.categories_[k]] for k, v in enumerate(cat_cols)}
    return pre, est, terminos, list(num_cols), categoricas


def exportar_ebm(pipe) -> dict:
    """Todo lo que hace falta para puntuar y explicar con la EBM sin bibliotecas de modelos (lo lee motor.MotorEBM)."""
    pre, ebm, terminos, num_cols, categoricas = _partes(pipe)
    assert len(terminos) == len(ebm.term_scores_)
    assert all(len(f) == 1 for f in ebm.term_features_), "hay interacciones: el motor solo cubre efectos principales"
    assert [f[0] for f in ebm.term_features_] == list(range(len(ebm.term_features_)))
    assert all(t == "continuous" for t in ebm.feature_types_in_)
    importancias = ebm.term_importances()
    for t in terminos:
        i = t["indice"]
        t["cortes"] = [float(x) for x in np.asarray(ebm.bins_[i][0], dtype=float)]          # en la escala que ve la EBM
        t["puntajes"] = [float(x) for x in np.asarray(ebm.term_scores_[i], dtype=float)]     # [faltante, tramos…, desconocido]
        t["desviaciones"] = [float(x) for x in np.asarray(ebm.standard_deviations_[i], dtype=float)]
        t["limites"] = [float(x) for x in np.asarray(ebm.feature_bounds_[i], dtype=float)]
        bordes = ebm.histogram_edges_[i]
        t["hist_bordes"] = [float(x) for x in np.asarray(bordes, dtype=float)] if bordes is not None else []
        t["hist_pesos"] = [float(x) for x in np.asarray(ebm.histogram_weights_[i], dtype=float)]
        t["importancia"] = float(importancias[i])
        assert len(t["puntajes"]) == len(t["cortes"]) + 3
    return {"tipo": type(ebm).__name__, "enlace": str(ebm.link_), "intercepto": float(np.asarray(ebm.intercept_).ravel()[0]),
            "columnas": list(pre.feature_names_in_), "numericas": num_cols, "categoricas": categoricas, "terminos": terminos}


def exportar_logistica(pipe) -> dict:
    pre, est, terminos, num_cols, categoricas = _partes(pipe)
    coef = np.asarray(est.coef_, dtype=float).ravel()
    assert len(coef) == len(terminos)
    for t in terminos:
        t["coef"] = float(coef[t["indice"]])
    return {"tipo": type(est).__name__, "enlace": "logit", "intercepto": float(np.asarray(est.intercept_).ravel()[0]),
            "columnas": list(pre.feature_names_in_), "numericas": num_cols, "categoricas": categoricas, "terminos": terminos}


def punto_operacion(s, fraccion: float = CAPACIDAD, seed: int = SEMILLA) -> np.ndarray:
    return motor.punto_operacion(s, fraccion, seed)       # la regla del informe (generar.py::punto_operacion)


# ============================================================================ 1. ETL del Entregable 3
print(f"Entregable 3: {E3.name} | resultados: {RES.name} | destino: datos/")
if DESTINO.exists():
    shutil.rmtree(DESTINO)
(DESTINO / "modelos").mkdir(parents=True)

p = datos.preparar_todo()
c = p["conjuntos"]
clientes = p["clientes"]
X_tr = c.X_train
X_te = c.X_test.sort_index()                              # el orden de todo lo guardado: posición en customer_data.csv
y_tr = c.y_train_cont.loc[X_tr.index]
yb_tr = p["y_train"].loc[X_tr.index].astype(int)
y_te = c.y_test_cont.loc[X_te.index]
yb_te = p["y_test"].loc[X_te.index].astype(int)
tau = float(p["etiqueta"].umbral)
PREDICTORES = list(X_te.columns)
print(f"ETL: {time.perf_counter() - T0:.1f} s | entrenamiento {X_tr.shape} | prueba {X_te.shape} | umbral {tau:.6f}")

# ============================================================================ 2. modelos finales (la regla del informe)
tabla = analisis.cargar_tabla()
mejores = {t: analisis.mejores_por_modelo(tabla, t).set_index("modelo") for t in TAREAS}
pred = pd.read_parquet(E3 / "resultados" / "predicciones_prueba.parquet")
assert pred.index.equals(X_te.index), "predicciones_prueba.parquet no está en el orden del índice de prueba"
X_todo = pd.concat([X_tr, X_te])

finales, puntajes, verificacion = [], {}, {}
for tarea in TAREAS:
    for modelo, f in mejores[tarea].iterrows():
        finales.append({"tarea": tarea, "modelo": modelo, "balanceo": f["balanceo"], "optimizador": f["optimizador"],
                        "clave": f["clave"], "metrica_externa": abs(float(f["metrica"])), "desv_externa": float(f["desv_ext"]),
                        "hiperparametros": f["hiperparametros"], "archivo_origen": Path(f["ruta_modelo"]).name})

EXPORTAR = [("clasificacion", "ebm"), ("regresion", "ebm"), ("clasificacion", "logistica")]
ficha = {}
for tarea, modelo in EXPORTAR:
    f = mejores[tarea].loc[modelo]
    origen = E3 / "resultados" / "modelos" / Path(f["ruta_modelo"]).name       # ruta_modelo guarda una ruta absoluta
    pipe = joblib.load(origen)
    est = getattr(pipe.named_steps["modelo"], "ebm_", pipe.named_steps["modelo"])
    nuevo = PipelineSk([("preprocesado", pipe.named_steps["preprocesado"]), ("modelo", est)])
    nombre = f"{modelo}_{tarea}"
    de = f"Entregable 3: resultados/modelos/{origen.name}"
    s = nuevo.predict_proba(X_te)[:, 1] if tarea == "clasificacion" else nuevo.predict(X_te)
    dif_guardado = float(np.abs(s - pred[f["clave"]].to_numpy(dtype=float)).max())
    assert dif_guardado < 1e-12, (nombre, dif_guardado)
    puntajes[nombre] = s
    if modelo == "ebm":
        joblib.dump(nuevo, DESTINO / "modelos" / f"{nombre}.joblib", compress=3)
        anotar(f"modelos/{nombre}.joblib", de, "la misma tubería con la EBM de interpret sin la clase propia que la envuelve")
        exportado = exportar_ebm(nuevo)
        escribir_json(f"modelos/{nombre}.json", exportado, de, "tablas de la EBM (cortes y puntajes de cada término) y su preprocesado")
        m = motor.MotorEBM(exportado)
    else:
        exportado = exportar_logistica(nuevo)
        escribir_json(f"modelos/{nombre}.json", exportado, de, "coeficientes de la regresión logística y su preprocesado")
        m = motor.MotorLogistica(exportado)
    todo = nuevo.predict_proba(X_todo)[:, 1] if tarea == "clasificacion" else nuevo.predict(X_todo)
    dif_motor = float(np.abs(m.predecir(X_todo) - todo).max())
    assert dif_motor < 1e-9, (nombre, dif_motor)
    if modelo == "ebm":
        crudo = est.decision_function(nuevo.named_steps["preprocesado"].transform(X_todo)) if tarea == "clasificacion" else todo
        dif_suma = float(np.abs(m.puntaje(X_todo) - crudo).max())
        assert dif_suma < 1e-9, (nombre, dif_suma)
    verificacion[nombre] = {"filas_comparadas": int(len(X_todo)), "max_dif_motor_frente_al_modelo_guardado": dif_motor,
                            "max_dif_con_predicciones_prueba": dif_guardado}
    ficha[nombre] = {"origen": origen.name, "clave": f["clave"], "balanceo": f["balanceo"], "optimizador": f["optimizador"],
                     "hiperparametros": f["hiperparametros"], "sha256_origen": hashlib.sha256(origen.read_bytes()).hexdigest()}
    print(f"modelo {nombre}: motor frente al modelo guardado, dif. máx. {dif_motor:.1e} en {len(X_todo):,} filas; "
          f"frente a predicciones_prueba, {dif_guardado:.1e}")

ebm_clf, ebm_reg = motor.MotorEBM(json.loads((DESTINO / "modelos" / "ebm_clasificacion.json").read_text(encoding="utf-8"))), \
    motor.MotorEBM(json.loads((DESTINO / "modelos" / "ebm_regresion.json").read_text(encoding="utf-8")))
for tarea, modelo in (("clasificacion", "xgboost"), ("regresion", "xgboost")):
    puntajes[f"{modelo}_{tarea}"] = pred[mejores[tarea].loc[modelo, "clave"]].to_numpy(dtype=float)

# ============================================================================ 3. recalibración isotónica de la EBM
oof = pd.read_parquet(E3 / "resultados" / "complementos" / "fuera_de_pliegue.parquet")
clave_ebm = ficha["ebm_clasificacion"]["clave"]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
    oof[f"{clave_ebm}__fuera_de_pliegue"].to_numpy(), oof["y_bin"].to_numpy())
s_ebm = puntajes["ebm_clasificacion"]
p_cal = np.interp(s_ebm, iso.X_thresholds_, iso.y_thresholds_)
cal = pd.read_parquet(E3 / "resultados" / "complementos" / "calibracion_prueba.parquet")
dif_iso = float(np.abs(p_cal - cal[f"{clave_ebm}__isotónica"].to_numpy()).max())
assert dif_iso < 1e-12, dif_iso
verificacion["calibracion_isotonica"] = {"nudos": int(len(iso.X_thresholds_)), "max_dif_con_calibracion_prueba": dif_iso}
escribir_json("calibracion_isotonica_ebm.json", {
    "que_es": "recalibración isotónica de la probabilidad de la EBM de clasificación, ajustada con las predicciones fuera de "
              "pliegue del entrenamiento (complementos.py del Entregable 3)",
    "uso": "p_recalibrada = numpy.interp(p_ebm, x, y)", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
    "Entregable 3: resultados/complementos/fuera_de_pliegue.parquet", "IsotonicRegression sobre el puntaje fuera de pliegue")
print(f"isotónica: {len(iso.X_thresholds_)} nudos, dif. máx. con calibracion_prueba.parquet {dif_iso:.1e}")

# ============================================================================ 4. clientes de prueba
CONTEXTO = ["customer_id", "active_products", "occupation", "nps_score", "clv_segment", "customer_lifetime_value"]
COHORTES = {29: "29–30", 30: "29–30", 44: "44–45", 45: "44–45", 59: "59–60", 60: "59–60"}
T = pd.concat([clientes.loc[X_te.index, CONTEXTO], X_te], axis=1)
T["cohorte"] = T["age"].map(COHORTES)
assert T["cohorte"].notna().all()
T["churn_probability"] = y_te
T["riesgo_alto"] = yb_te.astype("int8")
T["p_ebm"] = s_ebm
T["p_ebm_cal"] = p_cal
T["percentil_ebm"] = pd.Series(s_ebm).rank(pct=True).to_numpy()
T["contactar_ebm"] = punto_operacion(s_ebm).astype("int8")
T["resultado_ebm"] = np.select(
    [(T["contactar_ebm"] == 1) & (T["riesgo_alto"] == 1), (T["contactar_ebm"] == 1) & (T["riesgo_alto"] == 0),
     (T["contactar_ebm"] == 0) & (T["riesgo_alto"] == 1)], ["VP", "FP", "FN"], default="VN")
T["pron_ebm"] = puntajes["ebm_regresion"]
T["p_xgboost"] = puntajes["xgboost_clasificacion"]
T["contactar_xgboost"] = punto_operacion(T["p_xgboost"].to_numpy()).astype("int8")
T["pron_xgboost"] = puntajes["xgboost_regresion"]
T["p_logistica"] = puntajes["logistica_clasificacion"]
T["contactar_logistica"] = punto_operacion(T["p_logistica"].to_numpy()).astype("int8")
T.index.name = "fila"
T.to_parquet(DESTINO / "clientes_prueba.parquet")
anotar("clientes_prueba.parquet", ["Entregable 3: datos/customer_data.csv y transactions_data.csv (ETL de src/datos.py)",
                                   "Entregable 3: resultados/predicciones_prueba.parquet"],
       "los 9.745 clientes de prueba con su contexto, los 48 predictores, el objetivo, la etiqueta y los puntajes guardados")

for nombre, col_y, de_y in (("puntajes_prueba", yb_te, "y"), ("pronosticos_prueba", y_te, "y")):
    t = pd.read_csv(RES / "datos" / f"{nombre}.csv")
    assert len(t) == len(X_te)
    assert np.allclose(t[de_y].to_numpy(dtype=float), col_y.to_numpy(dtype=float)), nombre
    t.index = X_te.index
    t.index.name = "fila"
    t.to_parquet(DESTINO / f"{nombre}.parquet")
    anotar(f"{nombre}.parquet", f"sección de resultados: datos/{nombre}.csv", "las mismas columnas con el índice del cliente (fila)")
assert np.abs(pd.read_parquet(DESTINO / "puntajes_prueba.parquet")["ebm"].to_numpy() - s_ebm).max() < 1e-6

# los tres clientes de la tabla de ejemplos del informe
ej = pd.read_csv(RES / "datos" / "ejemplos.csv")
filas_ej = {
    "riesgo alto con menor puntaje": int(T.loc[T["riesgo_alto"] == 1, "p_ebm"].idxmin()),
    "riesgo bajo con mayor puntaje": int(T.loc[T["riesgo_alto"] == 0, "p_ebm"].idxmax()),
    "mayor error de regresión": int((T["churn_probability"] - T["pron_ebm"]).abs().idxmax()),
}
for _, r in ej.iterrows():
    fila = filas_ej[r["caso"]]
    assert abs(T.at[fila, "p_ebm"] - r["prob_ebm"]) < 1e-9 and abs(T.at[fila, "churn_probability"] - r["puntaje_real"]) < 1e-9, r["caso"]
print(f"clientes_prueba.parquet: {T.shape} | ejemplos del informe: {filas_ej}")

# ============================================================================ 5. variables
base = pd.read_csv(AQUI / "variables_base.csv")
assert list(base["variable"]) and set(base["variable"]) == set(PREDICTORES)
imp_clf, imp_reg = ebm_clf.importancia(), ebm_reg.importancia()
filas = []
for _, b in base.set_index("variable").loc[PREDICTORES].reset_index().iterrows():
    v = b["variable"]
    x = X_tr[v]
    f = dict(b)
    es_cat = v in ebm_clf.categoricas
    f["n_distintos"] = int(x.nunique())
    if es_cat:
        ef_c, ef_r = ebm_clf.efecto_categorias(v), ebm_reg.efecto_categorias(v)
        f["categorias"] = "|".join(ebm_clf.categoricas[v])
        f["moda"] = str(x.mode().iloc[0])
    else:
        xf = x.astype(float)
        q = xf.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        f.update({"min": float(xf.min()), "p1": float(q[0.01]), "p5": float(q[0.05]), "p25": float(q[0.25]),
                  "mediana": float(q[0.5]), "p75": float(q[0.75]), "p95": float(q[0.95]), "p99": float(q[0.99]),
                  "max": float(xf.max()), "media": float(xf.mean())})
        if f["n_distintos"] <= 30:
            f["valores"] = "|".join(f"{u:g}" for u in sorted(xf.unique()))
        ef_c, ef_r = ebm_clf.forma(v)["efecto"], ebm_reg.forma(v)["efecto"]
    f.update({"importancia_clf": float(imp_clf[v]), "puesto_clf": int(list(imp_clf.index).index(v) + 1),
              "importancia_reg": float(imp_reg[v]), "puesto_reg": int(list(imp_reg.index).index(v) + 1),
              "efecto_logit_min": float(np.min(ef_c)), "efecto_logit_max": float(np.max(ef_c)),
              "efecto_reg_min": float(np.min(ef_r)), "efecto_reg_max": float(np.max(ef_r))})
    filas.append(f)
variables = pd.DataFrame(filas)
escribir_csv("variables.csv", variables, ["variables_base.csv (nombres y papeles, curado a mano)",
                                         "Entregable 3: ETL de src/datos.py (rangos del entrenamiento)", "datos/modelos/ebm_*.json (importancia y efectos)"],
             "una fila por predictor")
top4 = float(imp_clf.iloc[:4].sum())
print(f"variables.csv: {variables.shape} | cuatro primeras: {list(imp_clf.index[:4])} = {top4:.4f} de la importancia")

# ============================================================================ 6. análisis exploratorio (solo entrenamiento)
tr = X_tr.copy()
tr["active_products"] = clientes.loc[X_tr.index, "active_products"]
tr["_y"] = y_tr
tr["_alto"] = yb_tr
MAX_VALORES = 30


def resumen_variable(v: str) -> pd.DataFrame:
    x = tr[v]
    if v in ebm_clf.categoricas or x.dtype == bool or x.dtype == object or str(x.dtype) in ("str", "string"):
        g = tr.groupby(x.astype(str), observed=True)
        out = g.agg(n=("_y", "size"), media_objetivo=("_y", "mean"), tasa_riesgo_alto=("_alto", "mean")).reset_index(names="grupo")
        out["x"] = np.nan
        out["desde"] = np.nan
        out["hasta"] = np.nan
        out["clase"] = "categorias"
    else:
        xf = x.astype(float)
        if xf.nunique() <= MAX_VALORES:
            llave, clase = xf, "valores"
        else:
            llave, clase = pd.qcut(xf, 20, duplicates="drop"), "veintiles"
        g = tr.assign(_x=xf).groupby(llave, observed=True)
        out = g.agg(n=("_y", "size"), media_objetivo=("_y", "mean"), tasa_riesgo_alto=("_alto", "mean"), x=("_x", "median"),
                    desde=("_x", "min"), hasta=("_x", "max")).reset_index(drop=True)
        out["grupo"] = [f"{a:g}" if a == b else f"{a:g} a {b:g}" for a, b in zip(out["desde"], out["hasta"])]
        out["clase"] = clase
    out.insert(0, "variable", v)
    out.insert(1, "orden", range(len(out)))
    return out[["variable", "orden", "clase", "grupo", "x", "desde", "hasta", "n", "media_objetivo", "tasa_riesgo_alto"]]


por_variable = pd.concat([resumen_variable(v) for v in ["active_products", *PREDICTORES]], ignore_index=True)
escribir_csv("eda/por_variable.csv", por_variable, "Entregable 3: ETL de src/datos.py, solo el entrenamiento",
             "media de churn_probability y fracción de riesgo alto por valor, veintil o categoría de cada variable")

conteos, bordes = np.histogram(y_tr.to_numpy(), bins=60)
escribir_csv("eda/objetivo_hist.csv", pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:], "n": conteos}),
             "Entregable 3: ETL de src/datos.py, solo el entrenamiento", "histograma de churn_probability en 60 tramos")

cohortes = []
for conjunto, X_, y_, yb_ in (("entrenamiento", X_tr, y_tr, yb_tr), ("prueba", X_te, y_te, yb_te)):
    d = pd.DataFrame({"cohorte": X_["age"].map(COHORTES), "y": y_, "alto": yb_})
    g = d.groupby("cohorte").agg(n=("y", "size"), media_objetivo=("y", "mean"), tasa_riesgo_alto=("alto", "mean")).reset_index()
    g.insert(0, "conjunto", conjunto)
    cohortes.append(g)
escribir_csv("eda/cohortes.csv", pd.concat(cohortes, ignore_index=True), "Entregable 3: ETL de src/datos.py",
             "clientes, media del objetivo y fracción de riesgo alto por cohorte de edad")

asoc = []
for v in ["active_products", *PREDICTORES]:
    x = tr[v]
    if v in ebm_clf.categoricas:
        asoc.append({"variable": v, "medida": "eta2", "valor": float(datos._eta2(x, tr["_y"])), "pearson": np.nan})
    else:
        xf = x.astype(float)
        asoc.append({"variable": v, "medida": "spearman", "valor": float(xf.corr(tr["_y"], method="spearman")),
                     "pearson": float(xf.corr(tr["_y"]))})
escribir_csv("eda/asociacion.csv", pd.DataFrame(asoc), "Entregable 3: ETL de src/datos.py, solo el entrenamiento",
             "asociación de cada variable con churn_probability: ρ de Spearman y r de Pearson (numéricas) o η² (categóricas)")

derivadas = [col for col in p["df"].columns if col not in clientes.columns]
tipos = {"numericas": [v for v in ebm_clf.d["numericas"] if v not in datos.BOOLEANAS], "booleanas": list(datos.BOOLEANAS),
         "categoricas": list(ebm_clf.categoricas)}
etl = {"n_clientes": int(len(clientes)), "n_columnas": int(clientes.shape[1]), "n_transacciones_limpias": int(len(p["tx"])),
       "auditoria_transacciones": {k: v for k, v in p["auditoria"].items()},
       "derivadas": derivadas, "eliminadas": list(datos.COLUMNAS_A_ELIMINAR), "predictores": PREDICTORES, "tipos": tipos,
       "n_columnas_modelo": int(len(ebm_clf.terminos)), "n_entrenamiento": int(len(X_tr)), "n_prueba": int(len(X_te))}
escribir_json("eda/etl.json", etl, "Entregable 3: ETL de src/datos.py", "recuentos y listas de columnas de cada paso del ETL", 1)
print(f"eda/: por_variable {por_variable.shape}, derivadas {len(derivadas)}, eliminadas {len(etl['eliminadas'])}, "
      f"tipos {[len(v) for v in tipos.values()]}")

# ============================================================================ 7. tablas de resultados (copias exactas)
for f in sorted((RES / "datos").iterdir()):
    if f.is_file() and f.name not in ("puntajes_prueba.csv", "pronosticos_prueba.csv"):
        copiar(f, f"resultados/{f.name}", RES, "sección de resultados")
COPIAS_E3 = ["metricas_prueba.csv", "complementos/calibracion.csv", "complementos/fuera_de_pliegue.csv", "complementos/semillas.csv",
             "complementos/semillas_optimizadores.csv", "complementos/calibracion_prueba.parquet", "computo/tiempos_finales.csv",
             "computo/escalamiento.csv", "computo/fidelidad.csv", "computo/equipo.json", "comparacion_entrega2/busquedas.json",
             "comparacion_entrega2/intervalos_clasificacion.csv", "comparacion_entrega2/intervalos_regresion.csv",
             "comparacion_entrega2/contrastes_clasificacion.csv", "comparacion_entrega2/contrastes_regresion.csv",
             "comparacion_entrega2/techo.json"]
for rel in COPIAS_E3:
    copiar(E3 / "resultados" / rel, f"entrega3/{rel}", E3, "Entregable 3")

# ============================================================================ 8. corridas y curvas de los optimizadores
exp = pd.read_parquet(E3 / "resultados" / "experimentos.parquet")
PESADAS = ["trazas", "detalles", "detalle", "params_por_pliegue", "metricas_ext", "ruta_modelo"]
cor = exp.drop(columns=[col for col in PESADAS if col in exp.columns]).copy()
cor.insert(0, "clave", cor["tarea"] + "__" + cor["modelo"] + "__" + cor["balanceo"] + "__" + cor["optimizador"])
cor["metrica_externa"] = cor["media_ext"].abs()          # RMSE positivo en regresión (media_ext lo guarda negativo)
mp = pd.read_csv(E3 / "resultados" / "metricas_prueba.csv")
llaves = ["tarea", "modelo", "balanceo", "optimizador"]
mp = mp.rename(columns={col: f"{col}_prueba" for col in mp.columns if col not in llaves})
cor = cor.merge(mp, on=llaves, how="left")
escribir_csv("entrega3/corridas.csv", cor, ["Entregable 3: resultados/experimentos.parquet", "Entregable 3: resultados/metricas_prueba.csv"],
             "las 160 corridas sin las columnas pesadas ni la ruta local del modelo, con sus métricas de prueba")

curvas = []
sin_ebm = tabla[tabla["modelo"] != "ebm"]                 # las 140 corridas de la guía, como el informe
for tarea in TAREAS:
    cv, _ = analisis.curvas_anytime(sin_ebm, tarea)
    cv.insert(0, "tarea", tarea)
    curvas.append(cv)
escribir_csv("entrega3/curvas_anytime.csv", pd.concat(curvas, ignore_index=True), "Entregable 3: resultados/experimentos.parquet (columna trazas)",
             "arrepentimiento normalizado de cada optimizador según la fracción del presupuesto (src/analisis.py::curvas_anytime), sin la EBM")
print(f"entrega3/: corridas {cor.shape}, curvas {sum(len(x) for x in curvas)} filas")

# ============================================================================ 9. metadatos y procedencia
orden = np.sort(s_ebm)[::-1]
n_op = int(round(CAPACIDAD * len(s_ebm)))
uso = X_tr.loc[X_tr["credit_card"].astype(bool), "credit_utilization_ratio"].astype(float)
meta = {
    "generado": datetime.now().isoformat(timespec="seconds"),
    "semilla": SEMILLA, "fraccion_prueba": 0.20,
    "particion": "train_test_split(df.index, test_size=0.20, random_state=42), sin estratificar",
    "objetivo": "churn_probability",
    "etiqueta": {"metodo": "percentil", "q": 0.75, "umbral": tau, "mediana_entrenamiento": float(y_tr.median())},
    "n_entrenamiento": int(len(X_tr)), "n_prueba": int(len(X_te)), "positivos_prueba": int(yb_te.sum()),
    "prevalencia_prueba": float(yb_te.mean()), "prevalencia_entrenamiento": float(yb_tr.mean()),
    "predictores": PREDICTORES, "columnas_contexto": CONTEXTO,
    "punto_operacion": {"fraccion": CAPACIDAD, "contactados": n_op, "puntaje_minimo_contactado_ebm": float(orden[n_op - 1]),
                        "puntaje_siguiente_ebm": float(orden[n_op])},
    "simulador": {"uso_cupo_mediano_con_tarjeta": float(uso.median()), "productos": motor.PRODUCTOS,
                  "tope_tx_satisfaccion": motor.TOPE_TX_SATISFACCION},
    "ejemplos": filas_ej,
    "ebm": {t: {"terminos": len(m.terminos), "parametros": m.n_parametros, "intercepto": m.intercepto,
                "hiperparametros": ficha[f"ebm_{t}"]["hiperparametros"]} for t, m in (("clasificacion", ebm_clf), ("regresion", ebm_reg))},
    "metricas_prueba": {
        "ebm_clasificacion": {"auc": float(roc_auc_score(yb_te, s_ebm))},
        "xgboost_clasificacion": {"auc": float(roc_auc_score(yb_te, T["p_xgboost"]))},
        "logistica_clasificacion": {"auc": float(roc_auc_score(yb_te, T["p_logistica"]))},
        "ebm_regresion": {"rmse": float(np.sqrt(mean_squared_error(y_te, T["pron_ebm"]))), "r2": float(r2_score(y_te, T["pron_ebm"]))},
        "xgboost_regresion": {"rmse": float(np.sqrt(mean_squared_error(y_te, T["pron_xgboost"]))), "r2": float(r2_score(y_te, T["pron_xgboost"]))},
    },
    "modelos_exportados": ficha, "modelos_finales": finales, "verificacion": verificacion,
    "versiones": {m: __import__(m).__version__ for m in ("sklearn", "interpret", "numpy", "pandas", "joblib")},
}
escribir_json("metadatos.json", meta, "preparar_datos.py", "partición, etiqueta, punto de operación, métricas reproducidas y modelos", 1)

total = 0
registro = []
for f in sorted(DESTINO.rglob("*")):
    if f.is_file():
        rel = f.relative_to(DESTINO).as_posix()
        total += f.stat().st_size
        registro.append({"archivo": rel, "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                         **ORIGEN.get(rel, {"de": [], "como": ""})})
sin_origen = [r["archivo"] for r in registro if not r["de"]]
assert not sin_origen, f"archivos sin procedencia anotada: {sin_origen}"
(DESTINO / "origen.json").write_text(json.dumps({"generado": meta["generado"], "archivos": registro}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
print(f"datos/: {len(registro) + 1} archivos, {total / 1e6:.2f} MB, en {time.perf_counter() - T0:.1f} s")
print("punto de operación:", meta["punto_operacion"])
print("métricas reproducidas:", json.dumps(meta["metricas_prueba"]))
