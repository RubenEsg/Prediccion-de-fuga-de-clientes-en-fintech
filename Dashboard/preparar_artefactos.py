"""Arma artifacts/ (los 16 modelos finales con su metadata.json) y los datos que el Tercer Entregable pide además de datos/.

    "../entrega 3/.venv/Scripts/python.exe" -B preparar_artefactos.py

Se corre después de preparar_datos.py (que borra y rehace datos/) y con el Python del Entregable 3: necesita su paquete src,
los modelos de resultados/modelos/ (que no están en el repositorio publicado), scikit-learn, xgboost, interpret y shap.

Qué deja:
  artifacts/<tarea>/<modelo>/modelo.joblib    la tubería final (preprocesado + modelo) de la mejor combinación de cada uno
                                              de los 8 modelos por tarea, reentrenada con todo el entrenamiento
  artifacts/<tarea>/<modelo>/metadata.json    versiones, semilla, hiperparámetros, esquema de entrada, métricas del bucle
                                              externo y de la prueba, tiempos y la prueba de ida y vuelta
  artifacts/indice.json                       los 16 artefactos, su huella y el resultado de la ida y vuelta
  datos/eda/entrenamiento.parquet             el entrenamiento (38.978 clientes, en el orden en que se ajustaron los
                                              modelos): predictores, active_products, objetivo, etiqueta y faltantes
  datos/eda/faltantes.csv                     valores faltantes de las 54 columnas originales en el entrenamiento
  datos/fondo.parquet                         100 clientes de entrenamiento: el fondo de las explicaciones SHAP
  datos/calibracion_modelos.json              recalibración isotónica de los 8 clasificadores y su conjunto conforme
  datos/conformal.json                        intervalos conformes de los 8 modelos de regresión, con su cobertura en prueba
  datos/shap/<tarea>__<modelo>.parquet        valores SHAP de una muestra fija de la prueba (resumen global)
  datos/shap/indice.json                      método, escala y verificación de cada explicación
  datos/cd.json                               Friedman, Iman-Davenport y la diferencia crítica de Nemenyi por tarea
"""
from __future__ import annotations

import hashlib
import json
import math
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
ARTEFACTOS = AQUI / "artifacts"
TAREAS = ("clasificacion", "regresion")
SEMILLA = 42
N_SHAP = 500            # filas de prueba para el resumen SHAP global (las mismas para todos los modelos)
N_SHAP_VECINOS = 150    # k-NN: sus valores SHAP se estiman por muestreo y cuestan más
NIVELES = (0.80, 0.90, 0.95)
Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}  # Demšar (2006), alfa 0,05


def _carpeta(variable: str, candidatas: list[Path], prueba: str, nombre: str) -> Path:
    for c in [os.environ.get(variable), *candidatas]:
        if c and (Path(c) / prueba).exists():
            return Path(c).resolve()
    sys.exit(f"No se encontró {nombre} (se buscó {prueba}); fije la variable {variable}")


E3 = _carpeta("ENTREGA3", [AQUI.parent / "entrega 3", AQUI.parent.parent / "Entrega3"], "resultados/experimentos.parquet",
              "la carpeta del Entregable 3")
RES = _carpeta("RESULTADOS_E3", [AQUI.parent / "Resultados Entregable 3", AQUI.parent / "resultados"], "datos/puntajes_prueba.csv",
               "la carpeta de la sección de resultados")
if not (DESTINO / "metadatos.json").exists():
    sys.exit("Primero corra preparar_datos.py: este guion completa la carpeta datos/ que ese arma.")
if not (E3 / "resultados" / "modelos").is_dir():
    sys.exit("Faltan los modelos guardados (resultados/modelos/ del Entregable 3).")

sys.path.insert(0, str(E3))
sys.path.insert(0, str(AQUI))

import joblib  # noqa: E402
import scipy  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline as PipelineSk  # noqa: E402
from src import analisis, datos  # noqa: E402

import explicacion  # noqa: E402
import motor  # noqa: E402

T0 = time.perf_counter()
ORIGEN: dict[str, dict] = {}


def anotar(rel: str, de, como: str) -> None:
    ORIGEN[rel] = {"de": [de] if isinstance(de, str) else list(de), "como": como}


def _a_json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def escribir_json(ruta: Path, objeto) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=1, default=_a_json), encoding="utf-8")


def sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


# ============================================================================ 1. ETL y modelos finales del Entregable 3
p = datos.preparar_todo()
c = p["conjuntos"]
clientes = p["clientes"]
X_tr = c.X_train                                          # el orden en que se ajustaron los modelos (k-NN guarda filas)
X_te = c.X_test.sort_index()
y_tr, yb_tr = c.y_train_cont.loc[X_tr.index], p["y_train"].loc[X_tr.index].astype(int)
y_te, yb_te = c.y_test_cont.loc[X_te.index], p["y_test"].loc[X_te.index].astype(int)
PRED = list(X_te.columns)
tabla = analisis.cargar_tabla()
mejores = {t: analisis.mejores_por_modelo(tabla, t).set_index("modelo") for t in TAREAS}
pred = pd.read_parquet(E3 / "resultados" / "predicciones_prueba.parquet")
oof = pd.read_parquet(E3 / "resultados" / "complementos" / "fuera_de_pliegue.parquet")
tiempos = pd.read_csv(E3 / "resultados" / "computo" / "tiempos_finales.csv")
prueba_tabla = {"clasificacion": pd.read_csv(RES / "datos" / "clasificacion_prueba.csv").set_index("modelo"),
                "regresion": pd.read_csv(RES / "datos" / "regresion_prueba.csv").set_index("modelo")}
variables = pd.read_csv(DESTINO / "variables.csv", keep_default_na=False, na_values=[""]).set_index("variable")
print(f"ETL listo en {time.perf_counter() - T0:.0f} s: entrenamiento {X_tr.shape}, prueba {X_te.shape}")

VERSIONES = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
             "interpret": __import__("interpret").__version__, "numpy": np.__version__, "pandas": pd.__version__,
             "scipy": scipy.__version__, "joblib": joblib.__version__, "shap": shap.__version__}


def esquema() -> list[dict]:
    """Nombre, tipo y rango válido de cada una de las 48 variables de entrada (el formulario de la app sale de aquí)."""
    out = []
    for v in PRED:
        x = X_tr[v]
        fila = variables.loc[v] if v in variables.index else None
        d = {"nombre": v, "nombre_visible": None if fila is None else fila["nombre"],
             "unidad": None if fila is None or pd.isna(fila["unidad"]) else fila["unidad"],
             "descripcion": None if fila is None or pd.isna(fila["descripcion"]) else fila["descripcion"],
             "derivada_etl": bool(fila["derivada_etl"]) if fila is not None else False}
        if x.dtype == bool:
            d.update(tipo="booleana", valores=[False, True], moda=bool(x.mode().iloc[0]))
        elif pd.api.types.is_string_dtype(x) or x.dtype == object:
            d.update(tipo="categorica", categorias=sorted(map(str, x.unique())), moda=str(x.mode().iloc[0]))
        elif pd.api.types.is_integer_dtype(x):
            d.update(tipo="entera", min=int(x.min()), max=int(x.max()), mediana=float(x.median()))
        else:
            d.update(tipo="real", min=float(x.min()), max=float(x.max()), mediana=float(x.median()))
        out.append(d)
    return out


ESQUEMA = esquema()


def salida_modelo(pipe, X: pd.DataFrame, tarea: str) -> np.ndarray:
    """El puntaje que guarda el proyecto: probabilidad (o margen del SVM lineal) en clasificación, pronóstico en regresión."""
    if tarea == "regresion":
        return np.asarray(pipe.predict(X), dtype=float)
    est = pipe.named_steps["modelo"]
    if hasattr(est, "predict_proba"):
        return np.asarray(pipe.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(pipe.decision_function(X), dtype=float)


# ============================================================================ 2. artifacts/: serialización e ida y vuelta
if ARTEFACTOS.exists():
    shutil.rmtree(ARTEFACTOS)
indice, TUBERIAS = [], {}
for tarea in TAREAS:
    for modelo, f in mejores[tarea].iterrows():
        origen = E3 / "resultados" / "modelos" / Path(f["ruta_modelo"]).name
        original = joblib.load(origen)
        assert [n for n, _ in original.steps] == ["preprocesado", "modelo"], original.steps   # sin muestreador
        est = original.named_steps["modelo"]
        # El proyecto guardó una imblearn.pipeline.Pipeline; sin muestreador es la misma tubería de scikit-learn, así que se
        # guarda como sklearn.pipeline.Pipeline (la app no necesita imbalanced-learn). La EBM con pesos se guarda como la
        # EBM de interpret que envuelve (la app no necesita el paquete src del proyecto).
        pipe = PipelineSk([("preprocesado", original.named_steps["preprocesado"]),
                           ("modelo", est.ebm_ if hasattr(est, "ebm_") else est)])
        if "RandomForest" in type(est).__name__:          # sumar los árboles en un solo hilo: resultado determinista
            pipe.named_steps["modelo"].set_params(n_jobs=1)
        en_memoria = salida_modelo(pipe, X_te, tarea)
        dif_original = float(np.max(np.abs(en_memoria - salida_modelo(original, X_te, tarea))))
        assert dif_original == 0.0, (tarea, modelo, dif_original)
        guardada = pred[f["clave"]].to_numpy(dtype=float)
        carpeta = ARTEFACTOS / tarea / modelo
        carpeta.mkdir(parents=True)
        ruta = carpeta / "modelo.joblib"
        joblib.dump(pipe, ruta, compress=3)
        t1 = time.perf_counter()
        recargado = joblib.load(ruta)
        carga_s = time.perf_counter() - t1
        t1 = time.perf_counter()
        ida_vuelta = salida_modelo(recargado, X_te, tarea)
        inferencia_s = time.perf_counter() - t1
        dif_rt = float(np.max(np.abs(ida_vuelta - en_memoria)))
        dif_guardada = float(np.max(np.abs(en_memoria - guardada)))
        assert np.array_equal(ida_vuelta, en_memoria), (tarea, modelo, dif_rt)
        assert dif_guardada < 1e-9, (tarea, modelo, dif_guardada)
        motor_dif = None
        if modelo == "ebm":                              # la app evalúa la EBM con motor.py: debe dar lo mismo
            m = motor.MotorEBM(json.loads((DESTINO / "modelos" / f"ebm_{tarea}.json").read_text(encoding="utf-8")))
            motor_dif = float(np.max(np.abs(m.predecir(X_te) - en_memoria)))
            assert motor_dif < 1e-9, motor_dif
        if modelo == "knn":                              # las posiciones de los vecinos son filas del entrenamiento
            z = recargado.named_steps["preprocesado"].transform(X_tr)
            assert np.allclose(np.asarray(z, dtype=float), recargado.named_steps["modelo"]._fit_X), "orden del entrenamiento"
        hp = f["hiperparametros"]
        try:
            hp = json.loads(hp) if isinstance(hp, str) else hp
        except json.JSONDecodeError:
            pass
        t = tiempos[(tiempos.tarea == tarea) & (tiempos.modelo == modelo)].iloc[0]
        pt = prueba_tabla[tarea].loc[modelo]
        metrica = "AUC" if tarea == "clasificacion" else "RMSE"
        meta = {
            "tarea": tarea, "modelo": modelo, "estimador": type(pipe.named_steps["modelo"]).__module__ + "."
            + type(pipe.named_steps["modelo"]).__name__,
            "balanceo": f["balanceo"], "optimizador": f["optimizador"], "semilla": SEMILLA,
            "hiperparametros": hp, "versiones": VERSIONES,
            "notas": (["n_jobs fijado en 1 al guardar: con varios hilos la suma de los árboles cambia de orden y la "
                       "probabilidad difiere en 1e-16 entre dos ejecuciones"] if modelo == "random_forest" else []),
            "entrada": {"columnas": PRED, "esquema": ESQUEMA,
                        "preprocesado": "StandardScaler en las 39 numéricas y booleanas; OneHotEncoder en las 9 categóricas"},
            "salida": ("probabilidad de riesgo alto (predict_proba)" if hasattr(pipe.named_steps["modelo"], "predict_proba")
                       else "margen del SVM lineal (decision_function); la app lo recalibra") if tarea == "clasificacion"
            else "churn_probability prevista",
            "metricas_bucle_externo": {"metrica": metrica, "media": abs(float(f["metrica"])), "desviacion": float(f["desv_ext"]),
                                       "pliegues_externos": 5, "pliegues_internos": 3},
            "metricas_prueba": {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else v))
                                for k, v in pt.items()},
            "computo": {"ajuste_s": float(t["ajuste_s"]), "ajuste_desv_s": float(t["ajuste_desv_s"]),
                        "inferencia_prueba_s": float(t["inferencia_s"]), "n_prueba": int(t["n_test"]),
                        "carga_artefacto_s": carga_s, "inferencia_medida_aqui_s": inferencia_s},
            "tuberia": "sklearn.pipeline.Pipeline con dos pasos (preprocesado y modelo); la del proyecto era una "
                       "imblearn.pipeline.Pipeline sin muestreador, con los mismos pasos (diferencia de predicciones: 0)",
            "ida_y_vuelta": {"filas": int(len(X_te)), "identicas": True, "max_dif_memoria": dif_rt,
                             "max_dif_predicciones_guardadas": dif_guardada, "max_dif_motor_numpy": motor_dif},
            "origen": {"archivo": origen.name, "sha256_origen": sha(origen)},
            "generado": datetime.now().isoformat(timespec="seconds"),
        }
        escribir_json(carpeta / "metadata.json", meta)
        indice.append({"tarea": tarea, "modelo": modelo, "ruta": f"{tarea}/{modelo}/modelo.joblib",
                       "bytes": ruta.stat().st_size, "sha256": sha(ruta), "estimador": meta["estimador"],
                       "balanceo": f["balanceo"], "optimizador": f["optimizador"],
                       "metrica_externa": meta["metricas_bucle_externo"], "ida_y_vuelta": "idénticas"})
        TUBERIAS[(tarea, modelo)] = recargado
        print(f"  {tarea:13} {modelo:14} {ruta.stat().st_size / 1e6:6.2f} MB  ida y vuelta idéntica en {len(X_te):,} filas"
              f"{'' if motor_dif is None else f'  motor numpy dif {motor_dif:.1e}'}")
# Random Forest de clasificación: sus árboles en numpy para el servidor de 512 MB (bosque.py), con igualdad exacta
import bosque  # noqa: E402
print("bosque compacto:", bosque.exportar(ARTEFACTOS / "clasificacion" / "random_forest", X_te))
escribir_json(ARTEFACTOS / "indice.json", {"generado": datetime.now().isoformat(timespec="seconds"), "versiones": VERSIONES,
                                           "semilla": SEMILLA, "artefactos": indice})
print(f"artifacts/: {len(indice)} modelos, {sum(a['bytes'] for a in indice) / 1e6:.1f} MB")

# ============================================================================ 3. entrenamiento para el EDA y fondo SHAP
CRUDAS_CON_FALTANTES = [col for col in clientes.columns if clientes[col].isna().any()]
tr = X_tr.copy()
tr["active_products"] = clientes.loc[X_tr.index, "active_products"].to_numpy()
tr["churn_probability"] = y_tr.to_numpy()
tr["riesgo_alto"] = yb_tr.to_numpy()
for col in CRUDAS_CON_FALTANTES:
    tr[f"falta__{col}"] = clientes.loc[X_tr.index, col].isna().to_numpy()
tr.insert(0, "fila", X_tr.index.to_numpy())
tr.reset_index(drop=True).to_parquet(DESTINO / "eda" / "entrenamiento.parquet", index=False)
anotar("eda/entrenamiento.parquet", "Entregable 3: ETL de src/datos.py, partición 80/20 con semilla 42",
       "el entrenamiento en el orden de ajuste: 48 predictores, active_products, churn_probability, riesgo_alto e "
       "indicadoras de faltantes de las columnas originales que los tienen")
falt = pd.DataFrame({"columna": clientes.columns,
                     "faltantes": clientes.loc[X_tr.index].isna().sum().to_numpy(),
                     "fraccion": clientes.loc[X_tr.index].isna().mean().to_numpy(),
                     "en_modelo": [col in PRED for col in clientes.columns]})
falt.to_csv(DESTINO / "eda" / "faltantes.csv", index=False, lineterminator="\n")
anotar("eda/faltantes.csv", "Entregable 3: customer_data.csv, filas de entrenamiento", "faltantes de las 54 columnas originales")
fondo = X_tr.sample(n=100, random_state=SEMILLA)
fondo.to_parquet(DESTINO / "fondo.parquet")
anotar("fondo.parquet", "Entregable 3: entrenamiento", "100 clientes de entrenamiento (semilla 42): fondo de las explicaciones")
print(f"eda/entrenamiento.parquet {tr.shape}; faltantes en {CRUDAS_CON_FALTANTES}")

# ============================================================================ 4. probabilidad recalibrada y conjuntos conformes
def ece(y, p, tramos: int = 10) -> float:
    b = np.clip((p * tramos).astype(int), 0, tramos - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(tramos) if (b == i).any()))


calib = {}
for modelo, f in mejores["clasificacion"].iterrows():
    s_oof = oof[f"{f['clave']}__fuera_de_pliegue"].to_numpy(dtype=float)
    y_oof = oof["y_bin"].to_numpy(dtype=int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s_oof, y_oof)
    s_te = pred[f["clave"]].to_numpy(dtype=float)
    p_te = iso.predict(s_te)
    p_oof = iso.predict(s_oof)
    entrada = {}
    for nivel in NIVELES:                       # conjunto conforme LAC (Sadinle et al., 2019) con la probabilidad recalibrada
        a = 1 - nivel
        no_conf = 1 - np.where(y_oof == 1, p_oof, 1 - p_oof)
        n = len(no_conf)
        q = float(np.quantile(no_conf, min(1.0, math.ceil((n + 1) * (1 - a)) / n), method="higher"))
        incluye1, incluye0 = (1 - p_te) <= q, p_te <= q
        cubre = np.where(yb_te.to_numpy() == 1, incluye1, incluye0)
        entrada[f"{nivel:.2f}"] = {"umbral": q, "cobertura_prueba": float(cubre.mean()),
                                   "tamano_medio": float((incluye1.astype(int) + incluye0.astype(int)).mean()),
                                   "fraccion_ambos": float((incluye1 & incluye0).mean())}
    cruda_en_01 = bool(s_te.min() >= 0 and s_te.max() <= 1)
    calib[modelo] = {"x": iso.X_thresholds_, "y": iso.y_thresholds_, "nudos": int(len(iso.X_thresholds_)),
                     "ece_prueba_sin_recalibrar": ece(yb_te.to_numpy(), s_te) if cruda_en_01 else None,
                     "ece_prueba_recalibrada": ece(yb_te.to_numpy(), p_te),
                     "brier_prueba_recalibrada": float(np.mean((p_te - yb_te.to_numpy()) ** 2)),
                     "auc_prueba": float(roc_auc_score(yb_te, s_te)), "conformal": entrada}
escribir_json(DESTINO / "calibracion_modelos.json", {
    "metodo": "regresión isotónica ajustada con las predicciones fuera de pliegue del entrenamiento (5 pliegues); "
              "conjunto conforme LAC con la probabilidad recalibrada, calibrado con las mismas predicciones fuera de pliegue",
    "modelos": calib})
anotar("calibracion_modelos.json", "Entregable 3: resultados/complementos/fuera_de_pliegue.parquet y predicciones_prueba.parquet",
       "recalibración isotónica y conjuntos conformes de los 8 clasificadores, con su cobertura en prueba")

conf = {}
for modelo, f in mejores["regresion"].iterrows():
    r = np.abs(oof["y_cont"].to_numpy(dtype=float) - oof[f"{f['clave']}__fuera_de_pliegue"].to_numpy(dtype=float))
    n = len(r)
    yhat = pred[f["clave"]].to_numpy(dtype=float)
    entrada = {}
    for nivel in NIVELES:
        q = float(np.quantile(r, min(1.0, math.ceil((n + 1) * nivel) / n), method="higher"))
        entrada[f"{nivel:.2f}"] = {"semiancho": q, "cobertura_prueba": float((np.abs(y_te.to_numpy() - yhat) <= q).mean()),
                                   "ancho": 2 * q}
    conf[modelo] = entrada
escribir_json(DESTINO / "conformal.json", {
    "metodo": "intervalo conforme por validación cruzada: pronóstico del modelo final ± el cuantil ceil((n+1)(1-α))/n de los "
              "residuos absolutos fuera de pliegue del entrenamiento (n = 38.978)",
    "modelos": conf})
anotar("conformal.json", "Entregable 3: resultados/complementos/fuera_de_pliegue.parquet y predicciones_prueba.parquet",
       "intervalos conformes al 80, 90 y 95 % de los 8 modelos de regresión, con su cobertura en prueba")
print("calibración y conformes:", {m: round(v["conformal"]["0.90"]["cobertura_prueba"], 3) for m, v in calib.items()},
      {m: round(v["0.90"]["cobertura_prueba"], 3) for m, v in conf.items()})

# ============================================================================ 5. resumen SHAP global
(DESTINO / "shap").mkdir(exist_ok=True)
muestra = X_te.sample(n=N_SHAP, random_state=SEMILLA).sort_index()
indice_shap = {}
for (tarea, modelo), pipe in TUBERIAS.items():
    t1 = time.perf_counter()
    if modelo == "ebm":
        m = motor.ebm(tarea)
        aportes = m.por_variable(muestra)[PRED]
        base = np.full(len(muestra), m.intercepto)
        y_out = m.puntaje(muestra)
        r = {"aportes": aportes, "base": base, "salida": y_out, "metodo": "ebm", "exacto": True, "error": None,
             "escala": "logit del riesgo alto" if tarea == "clasificacion" else "churn_probability predicha",
             "eficiencia": float(np.max(np.abs(base + aportes.sum(axis=1).to_numpy() - y_out)))}
    else:
        filas = muestra.iloc[:N_SHAP_VECINOS] if modelo == "knn" else muestra
        r = explicacion.explicar(pipe, filas, fondo, tarea, permutaciones=4, n_fondo=16, arbol_exacto=True)
    tabla_shap = r["aportes"].copy()
    tabla_shap["__base"] = r["base"]
    tabla_shap["__salida"] = r["salida"]
    tabla_shap.index.name = "fila"
    tabla_shap.reset_index().to_parquet(DESTINO / "shap" / f"{tarea}__{modelo}.parquet", index=False)
    anotar(f"shap/{tarea}__{modelo}.parquet", f"artifacts/{tarea}/{modelo}/modelo.joblib y datos/fondo.parquet",
           f"valores SHAP ({r['metodo']}) de {len(tabla_shap)} clientes de prueba fijos (semilla 42)")
    verif = {"eficiencia_max_dif": r["eficiencia"]}
    est = pipe.named_steps["modelo"]
    pre = pipe.named_steps["preprocesado"]
    z = np.asarray(pre.transform(muestra.iloc[:20]), dtype=float)
    zf = np.asarray(pre.transform(fondo), dtype=float)
    if r["metodo"] == "lineal":                   # contra shap.LinearExplainer (intervencional) por columna
        ref = shap.LinearExplainer(est, shap.maskers.Independent(zf, max_samples=len(zf))).shap_values(z)
        mio = (z - zf.mean(axis=0)) * np.asarray(est.coef_, dtype=float).ravel()
        verif["frente_a_shap_LinearExplainer"] = float(np.max(np.abs(np.asarray(ref) - mio)))
    elif r["metodo"] == "xgboost":
        ref = shap.TreeExplainer(est).shap_values(z)
        mio = est.get_booster().predict(explicacion._dmatriz(est, z), pred_contribs=True)[:, :-1]
        verif["frente_a_shap_TreeExplainer"] = float(np.max(np.abs(np.asarray(ref) - mio)))
    elif r["metodo"] == "bayes":                   # aditivo: una permutación de shap ya es exacta
        f_ = lambda zz: explicacion._salida_z(est, zz, tarea)  # noqa: E731
        ex = shap.PermutationExplainer(f_, shap.maskers.Independent(zf, max_samples=len(zf)))
        ref = ex(z[:5], max_evals=2 * z.shape[1] + 1).values
        mio = explicacion._bayes_terminos(est, z[:5]) - explicacion._bayes_terminos(est, zf).mean(axis=0)
        verif["frente_a_shap_PermutationExplainer"] = float(np.max(np.abs(ref - mio)))
    if r["error"] is not None:
        verif["error_monte_carlo_medio"] = float(np.nanmean(r["error"].to_numpy()))
    importancia = r["aportes"].abs().mean().sort_values(ascending=False)
    indice_shap[f"{tarea}__{modelo}"] = {"metodo": r["metodo"], "escala": r["escala"], "exacto": r["exacto"],
                                        "filas": int(len(tabla_shap)), "verificacion": verif,
                                        "importancia_media_abs": importancia.to_dict(), "segundos": time.perf_counter() - t1}
    print(f"  SHAP {tarea:13} {modelo:14} {r['metodo']:8} {len(tabla_shap):4} filas  {time.perf_counter() - t1:6.1f} s  {verif}")
escribir_json(DESTINO / "shap" / "indice.json", {"filas_muestra": N_SHAP, "filas_knn": N_SHAP_VECINOS, "semilla": SEMILLA,
                                                 "fondo": "datos/fondo.parquet (100 clientes de entrenamiento)",
                                                 "modelos": indice_shap})
anotar("shap/indice.json", "preparar_artefactos.py", "método, escala, importancia y verificación de cada resumen SHAP")

# ============================================================================ 6. Friedman, Iman-Davenport y Nemenyi
cd = {}
for tarea, archivo, mayor_mejor in (("clasificacion", "pliegues_auc.csv", True), ("regresion", "pliegues_rmse.csv", False)):
    m = pd.read_csv(RES / "datos" / archivo)
    n, k = m.shape
    rangos = (-m if mayor_mejor else m).rank(axis=1, method="average")
    r = rangos.mean()
    chi2 = 12 * n / (k * (k + 1)) * (float((r ** 2).sum()) - k * (k + 1) ** 2 / 4)
    f_id = (n - 1) * chi2 / (n * (k - 1) - chi2)
    cd_val = Q_NEMENYI[k] * math.sqrt(k * (k + 1) / (6 * n))
    cd[tarea] = {"pliegues": n, "modelos": k, "metrica": "AUC" if mayor_mejor else "RMSE",
                 "rango_medio": r.sort_values().to_dict(), "chi2_friedman": chi2,
                 "p_friedman": float(stats.chi2.sf(chi2, k - 1)), "f_iman_davenport": f_id,
                 "p_iman_davenport": float(stats.f.sf(f_id, k - 1, (k - 1) * (n - 1))),
                 "gl": [k - 1, (k - 1) * (n - 1)], "q_alfa": Q_NEMENYI[k], "alfa": 0.05, "diferencia_critica": cd_val,
                 "metricas_pliegue": m.to_dict(orient="list")}
escribir_json(DESTINO / "cd.json", cd)
anotar("cd.json", "Resultados Entregable 3: datos/pliegues_auc.csv y pliegues_rmse.csv",
       "rangos medios por pliegue externo, Friedman, Iman-Davenport y diferencia crítica de Nemenyi (alfa 0,05)")
print("diferencia crítica:", {t: round(v["diferencia_critica"], 3) for t, v in cd.items()},
      {t: round(v["p_iman_davenport"], 4) for t, v in cd.items()})

# ============================================================================ 7. procedencia (se añade a datos/origen.json)
o = json.loads((DESTINO / "origen.json").read_text(encoding="utf-8"))
previos = {a["archivo"]: a for a in o["archivos"] if a["archivo"] not in ORIGEN}
for rel, d in ORIGEN.items():
    f = DESTINO / rel
    previos[rel] = {"archivo": rel, "bytes": f.stat().st_size, "sha256": sha(f), **d}
o["archivos"] = sorted(previos.values(), key=lambda a: a["archivo"])
o["artefactos"] = datetime.now().isoformat(timespec="seconds")
(DESTINO / "origen.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"listo en {time.perf_counter() - T0:.0f} s")
