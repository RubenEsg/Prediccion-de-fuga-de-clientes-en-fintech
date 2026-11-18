"""Lleva al dash el modelo original (BDL, Boosting con Desplazamiento Latente) y el segundo benchmark reciente (LightGBM).

    "../entrega 3/.venv/Scripts/python.exe" -B preparar_original.py

Se corre después de preparar_datos.py y de preparar_artefactos.py, y se vuelve a correr cada vez que la corrección
(ModeloOriginal/resultados) tiene resultados nuevos: no borra ni rehace nada de lo que dejaron esos dos guiones; añade
encima y reemplaza solo lo suyo. Si un insumo todavía no existe, lo anota en datos/original/estado.json y sigue: la vista
«Modelo original» muestra ese bloque como pendiente. Hay que correrlo con el Python del Entregable 3 (las tuberías
guardadas son imblearn.pipeline.Pipeline) y con la carpeta ModeloOriginal a la vista, que solo se lee: se busca en
../entrega 3/repo_entregable3/ModeloOriginal (carpeta de trabajo), en ../ModeloOriginal (repositorio publicado) o en la
variable MODELO_ORIGINAL.

Qué deja:
  src/original_model.py                      copia exacta de ModeloOriginal/src/original_model.py (las tuberías serializadas
                                             nombran el módulo src.original_model) y src/__init__.py
  artifacts/<tarea>/bdl/modelo.joblib        la tubería final del modelo original como sklearn.pipeline.Pipeline
  artifacts/<tarea>/bdl/metadata.json        el mismo esquema que los otros artefactos, con la prueba de ida y vuelta
  artifacts/indice.json                      se le añaden (o se le reemplazan) las entradas de bdl
  datos/puntajes_prueba.parquet              se le añaden las columnas bdl y lightgbm (lo mismo en pronosticos_prueba.parquet)
  datos/calibracion_modelos.json             se le añaden bdl y lightgbm (isotónica y conjunto conforme de clases)
  datos/conformal.json                       se le añaden bdl y lightgbm; bdl lleva además el intervalo central y el
                                             conjunto de predicción del propio modelo, con su cobertura y su largo en prueba
  datos/shap/<tarea>__bdl.parquet            aportes exactos del modelo original en la misma muestra fija de la prueba
  datos/shap/indice.json                     se le añaden las entradas de bdl
  datos/cd.json                              Friedman, Iman-Davenport y Nemenyi con los diez modelos
  datos/original/                            lo que lee la vista «Modelo original» (copias exactas y extractos):
      estado.json                            qué insumo había y cuál faltaba en esta corrida
      veredicto.json, pruebas_hipotesis.csv, pliegues_<tarea>.csv, techo.json, optimo.json, empates_campana.json,
      auditoria_fuga.json, metricas_prueba.csv, incertidumbre_prueba.csv, ablacion.csv, semillas.csv, condicion4.json,
      errores_subgrupos.csv, robustez/<estudio>.csv   (copias exactas)
      experimentos.csv                       la tabla maestra de la corrección, sin trazas ni rutas locales (con los
                                             hiperparámetros que eligió la búsqueda en cada pliegue externo)
      lideres_<tarea>.csv                    las filas de bdl y lightgbm con las columnas de resultados/<tarea>_prueba.csv
      costo.csv                              ajuste, inferencia y tamaño de bdl y lightgbm, y de dónde sale cada cifra
      adicionales.csv                        el original frente al mejor benchmark en cada conjunto de datos adicional
      literatura.json                        la matriz de novedad, la fecha y las cadenas de cada consulta y la búsqueda
                                             por citación
      modelo_<tarea>.json                    el modelo final: K, pesos, desplazamientos, árboles, historial de convergencia,
                                             tabla del BIC, importancia, curvas de las variables principales y conjunto
  datos/origen.json                          de dónde salió cada archivo que este guion escribe o cambia
  MODEL_CARD.md, DATA_STATEMENT.md           al final se regeneran con generar_fichas.py, porque citan el veredicto y otras
                                             cifras de datos/original/ y no deben quedar atrás de los datos

Nada de esto entrena ni busca hiperparámetros: carga tuberías ya ajustadas, las evalúa y copia tablas. Lo único que se
calcula aquí son remuestreos de las predicciones de prueba ya hechas (los intervalos BCa de las dos filas nuevas de la
tabla de líderes) y los estadísticos de rangos de cd.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"
DESTINO = DATOS / "original"
ARTEFACTOS = AQUI / "artifacts"
TAREAS = ("clasificacion", "regresion")
ORIGINAL, NUEVO = "bdl", "lightgbm"
NUEVOS = (ORIGINAL, NUEVO)
SEMILLA = 42
N_BOOT = 2000                      # réplicas de los intervalos BCa, las mismas del informe (resultados/resumen.json)
NIVELES = (0.80, 0.90, 0.95)
ALFA = 0.05
N_FORMAS = 4                       # curvas de las variables numéricas principales que se guardan
BARRAS_RESIDUO = 200               # barras del histograma de los residuos de calibración
METRICA = {"clasificacion": "AUC", "regresion": "RMSE"}
MAYOR_ES_MEJOR = {"clasificacion": True, "regresion": False}
ROBUSTEZ = ("sensibilidad", "ruido", "muestra", "columnas", "ciudades")
COPIAS = ("veredicto.json", "pruebas_hipotesis.csv", "pliegues_clasificacion.csv", "pliegues_regresion.csv", "techo.json",
          "auditoria_fuga.json", "metricas_prueba.csv", "incertidumbre_prueba.csv", "ablacion.csv", "semillas.csv",
          "condicion4.json", "errores_subgrupos.csv", "optimo.json", "empates_campana.json") \
    + tuple(f"robustez/{e}.csv" for e in ROBUSTEZ)
MARCA_ORIGEN = " · Corrección: "   # separa, en datos/origen.json, lo que hizo preparar_artefactos.py de lo que añade este guion
EN_ESTADO = ("nota", "de", "modificado", "sha256", "conjuntos", "completos", "semillas_original")   # lo que estado.json guarda de cada bloque


def _carpeta_modelo_original() -> Path:
    candidatas = [os.environ.get("MODELO_ORIGINAL"), AQUI.parent / "entrega 3" / "repo_entregable3" / "ModeloOriginal",
                  AQUI.parent / "ModeloOriginal"]
    for c in candidatas:
        if c and (Path(c) / "src" / "original_model.py").exists():
            return Path(c).resolve()
    sys.exit("No se encontró la carpeta ModeloOriginal (se buscó src/original_model.py); fije la variable MODELO_ORIGINAL")


MO = _carpeta_modelo_original()
R = MO / "resultados"
if not (DATOS / "metadatos.json").exists() or not (ARTEFACTOS / "indice.json").exists():
    sys.exit("Primero corra preparar_datos.py y preparar_artefactos.py: este guion añade encima de lo que ellos arman.")

sys.path.insert(0, str(AQUI))      # motor.py y explicacion.py del dash
sys.path.insert(0, str(MO))        # el paquete src de la corrección: el modelo original y sus pruebas estadísticas

import joblib  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline as PipelineSk  # noqa: E402
from src import estadistica, pruebas  # noqa: E402

import explicacion  # noqa: E402
import motor  # noqa: E402

T0 = time.perf_counter()
AHORA = datetime.now().isoformat(timespec="seconds")
ORIGEN: dict[str, dict] = {}
ESTADO: dict[str, dict] = {}
ESCRITOS: set[str] = set()          # archivos de datos/original/ escritos en esta corrida (los demás se borran al final)


class Falta(Exception):
    """Un insumo de la corrección que todavía no existe: el bloque queda pendiente y el guion sigue."""


# ============================================================================ utilidades
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


def _limpio(o):
    """El objeto sin NaN ni infinitos (JSON no los admite), con los tipos de numpy como tipos de Python."""
    if isinstance(o, dict):
        return {str(k): _limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpio(v) for v in o]
    if isinstance(o, np.ndarray):
        return _limpio(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def escribir_json(ruta: Path, objeto) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(_limpio(objeto), ensure_ascii=False, indent=1, default=_a_json), encoding="utf-8")


def sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def leer(ruta: Path, lector, intentos: int = 4, espera: float = 1.5):
    """Lee un archivo de la corrección; si otra sesión lo está escribiendo en ese momento, espera y vuelve a intentar."""
    if not ruta.exists():
        raise Falta(f"falta {ruta.relative_to(MO).as_posix()}")
    ultimo = None
    for _ in range(intentos):
        try:
            return lector(ruta)
        except Exception as e:  # noqa: BLE001  (un archivo a medio escribir)
            ultimo = e
            time.sleep(espera)
    raise Falta(f"{ruta.relative_to(MO).as_posix()} no se pudo leer ({type(ultimo).__name__}: {str(ultimo)[:120]})")


def leer_json(ruta: Path):
    return leer(ruta, lambda r: json.loads(r.read_text(encoding="utf-8")))


def anotar(rel: str, de, como: str) -> None:
    ORIGEN[rel] = {"de": [de] if isinstance(de, str) else list(de), "como": como}


def anotar_anadido(rel: str, de: str, como: str) -> None:
    """Un archivo que ya existía y al que este guion le añade algo: conserva de dónde salía y añade la corrección."""
    ORIGEN[rel] = {"anade": True, "de": de, "como": como}


def en_original(rel: str) -> Path:
    ruta = DESTINO / rel
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ESCRITOS.add(rel)
    return ruta


def bloque(clave: str, funcion, *args):
    """Ejecuta un bloque; si le falta un insumo lo anota como pendiente y devuelve None."""
    try:
        r = funcion(*args)
        ESTADO[clave] = {"listo": True, **({k: v for k, v in r.items() if k in EN_ESTADO} if isinstance(r, dict) else {})}
        print(f"  listo      {clave}" + (f": {r['nota']}" if isinstance(r, dict) and r.get("nota") else ""))
        return r if r is not None else True
    except Falta as e:
        ESTADO[clave] = {"listo": False, "motivo": str(e)}
        print(f"  PENDIENTE  {clave}: {e}")
        return None


def _json_o_valor(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def tabla_maestra(ruta: Path) -> pd.DataFrame:
    """La tabla maestra de una carpeta de resultados, con las columnas JSON ya como objetos."""
    t = leer(ruta, pd.read_parquet).copy()
    for col in ("puntajes_ext", "metricas_ext", "hiperparametros"):
        if col in t:
            t[col] = t[col].map(_json_o_valor)
    t["origen"] = t["origen"].astype(object).where(t["origen"].notna(), "Corrección") if "origen" in t else "Corrección"
    return t[t["estado"] == "ok"].reset_index(drop=True)


def corrida(t: pd.DataFrame, tarea: str, modelo: str, semilla: int = SEMILLA):
    """La fila de la tabla maestra de un modelo en una tarea y una semilla (None si todavía no está)."""
    f = t[(t["tarea"] == tarea) & (t["modelo"] == modelo) & (t["semilla"] == semilla)]
    return None if f.empty else f.iloc[0]


def pliegues_de(f) -> np.ndarray:
    """La métrica principal de cada pliegue externo, en su sentido natural (el RMSE se guarda con el signo cambiado)."""
    return np.abs(np.asarray(f["puntajes_ext"], dtype=float))


def clave_de(f) -> str:
    return f"{f['tarea']}__{f['modelo']}__{f['balanceo']}__{f['optimizador']}__{int(f['semilla'])}"


def salida_modelo(pipe, X: pd.DataFrame, tarea: str) -> np.ndarray:
    """El puntaje que guarda el proyecto: probabilidad en clasificación, pronóstico en regresión."""
    if tarea == "regresion":
        return np.asarray(pipe.predict(X), dtype=float)
    return np.asarray(pipe.predict_proba(X)[:, 1], dtype=float)


def ece(y, p, tramos: int = 10) -> float:
    """Error de calibración esperado con tramos de igual ancho, como preparar_artefactos.py."""
    b = np.clip((p * tramos).astype(int), 0, tramos - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(tramos) if (b == i).any()))


# ============================================================================ 0. insumos del dash
print(f"Modelo original: {MO}")
META_BASE = json.loads((ARTEFACTOS / "clasificacion" / "ebm" / "metadata.json").read_text(encoding="utf-8"))
ESQUEMA = META_BASE["entrada"]["esquema"]
COLUMNAS = [v["nombre"] for v in ESQUEMA]
clientes = pd.read_parquet(DATOS / "clientes_prueba.parquet")
X_te = clientes[COLUMNAS]
entrenamiento = pd.read_parquet(DATOS / "eda" / "entrenamiento.parquet")
PUNTAJES = pd.read_parquet(DATOS / "puntajes_prueba.parquet")
PRONOSTICOS = pd.read_parquet(DATOS / "pronosticos_prueba.parquet")
assert PUNTAJES.index.equals(X_te.index) and PRONOSTICOS.index.equals(X_te.index), "las tablas de prueba no tienen las mismas filas"
Y_TE = {"clasificacion": PUNTAJES["y"].to_numpy(dtype=int), "regresion": PRONOSTICOS["y"].to_numpy(dtype=float)}
Y_TR = {"clasificacion": entrenamiento["riesgo_alto"].to_numpy(dtype=int), "regresion": entrenamiento["churn_probability"].to_numpy(dtype=float)}
TABLA_SALIDAS = {"clasificacion": PUNTAJES, "regresion": PRONOSTICOS}
ARCHIVO_SALIDAS = {"clasificacion": "puntajes_prueba.parquet", "regresion": "pronosticos_prueba.parquet"}
VERSIONES = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
             "scipy": scipy.__version__, "joblib": joblib.__version__}

# ============================================================================ 1. el código del modelo original
(AQUI / "src").mkdir(exist_ok=True)
shutil.copyfile(MO / "src" / "original_model.py", AQUI / "src" / "original_model.py")
HUELLA_MODELO = sha(AQUI / "src" / "original_model.py")
assert HUELLA_MODELO == sha(MO / "src" / "original_model.py")
if not (AQUI / "src" / "__init__.py").exists():
    (AQUI / "src" / "__init__.py").write_text(
        '"""El modelo original del proyecto (Boosting con Desplazamiento Latente). original_model.py es una copia exacta de\n'
        'ModeloOriginal/src/original_model.py: la deja preparar_original.py y la comprueba tests/test_original.py."""\n',
        encoding="utf-8", newline="\n")
print(f"src/original_model.py copiado (sha256 {HUELLA_MODELO[:16]}…)")

# ============================================================================ 2. tabla maestra y predicciones de prueba
MAESTRA = bloque("experimentos", tabla_maestra, R / "experimentos.parquet")
if MAESTRA is None:
    MAESTRA = pd.DataFrame(columns=["tarea", "modelo", "semilla"])
PRED = bloque("predicciones_prueba", lambda: leer(R / "predicciones_prueba.parquet", pd.read_parquet))
if PRED is not None and not set(X_te.index) <= set(PRED.index):
    sys.exit("predicciones_prueba.parquet no tiene las filas de la prueba del dash")


def guardada(tarea: str, modelo: str, sufijo: str = ""):
    """La predicción de prueba que guardó la corrección (03_prueba.py), en el orden de las filas del dash, o None."""
    col = f"{tarea}__{modelo}{sufijo}"
    if PRED is None or col not in PRED:
        return None
    return PRED.loc[X_te.index, col].to_numpy()


def fuera_de_pliegue(tarea: str, modelo: str) -> np.ndarray:
    """Predicción fuera de pliegue de un modelo nuevo sobre el entrenamiento, en el orden de datos/eda/entrenamiento.parquet."""
    f = corrida(MAESTRA, tarea, modelo)
    if f is None:
        raise Falta(f"la tabla maestra no tiene todavía la corrida {tarea} · {modelo} (semilla {SEMILLA})")
    v = np.asarray(leer(R / "fuera_de_pliegue" / f"{clave_de(f)}.npy", np.load), dtype=float)
    if len(v) != len(entrenamiento):
        raise Falta(f"fuera_de_pliegue/{clave_de(f)}.npy no tiene las filas del entrenamiento")
    return v


# ============================================================================ 3. artifacts/<tarea>/bdl
TUBERIAS: dict[str, object] = {}
SALIDAS: dict[tuple[str, str], np.ndarray] = {}        # (tarea, modelo) -> puntaje o pronóstico en la prueba
CLASES: dict[str, np.ndarray] = {}                     # modelo -> clase con umbral 0,5 en la prueba (clasificación)
ARTEFACTO: dict[str, dict] = {}


def artefacto(tarea: str) -> dict:
    f = corrida(MAESTRA, tarea, ORIGINAL)
    if f is None:
        raise Falta(f"la tabla maestra no tiene todavía la corrida {tarea} · {ORIGINAL} (semilla {SEMILLA})")
    origen = R / "modelos" / f"{clave_de(f)}.joblib"
    original = leer(origen, joblib.load)
    assert [n for n, _ in original.steps] == ["preprocesado", "modelo"], original.steps          # sin muestreador
    assert list(original.named_steps["preprocesado"].feature_names_in_) == COLUMNAS, "la tubería espera otras columnas"
    # La corrección guarda una imblearn.pipeline.Pipeline sin muestreador: es la misma tubería de scikit-learn, así que se
    # guarda como sklearn.pipeline.Pipeline (la app no necesita imbalanced-learn)
    pipe = PipelineSk([("preprocesado", original.named_steps["preprocesado"]), ("modelo", original.named_steps["modelo"])])
    en_memoria = salida_modelo(pipe, X_te, tarea)
    dif_original = float(np.max(np.abs(en_memoria - salida_modelo(original, X_te, tarea))))
    assert dif_original == 0.0, (tarea, dif_original)
    carpeta = ARTEFACTOS / tarea / ORIGINAL
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "modelo.joblib"
    joblib.dump(pipe, ruta, compress=3)
    t1 = time.perf_counter()
    recargado = joblib.load(ruta)
    carga_s = time.perf_counter() - t1
    t1 = time.perf_counter()
    ida_vuelta = salida_modelo(recargado, X_te, tarea)
    inferencia_s = time.perf_counter() - t1
    assert np.array_equal(ida_vuelta, en_memoria), (tarea, float(np.max(np.abs(ida_vuelta - en_memoria))))
    previa = guardada(tarea, ORIGINAL)
    dif_guardada = None if previa is None else float(np.max(np.abs(en_memoria - previa.astype(float))))
    assert dif_guardada is None or dif_guardada < 1e-9, (tarea, dif_guardada)
    TUBERIAS[tarea] = recargado
    SALIDAS[tarea, ORIGINAL] = en_memoria
    if tarea == "clasificacion":
        CLASES[ORIGINAL] = np.asarray(recargado.predict(X_te)).astype(int)
    est = recargado.named_steps["modelo"]
    ARTEFACTO[tarea] = {"fila": f, "origen": origen, "ruta": ruta, "carga_s": carga_s, "inferencia_s": inferencia_s,
                        "dif_guardada": dif_guardada, "estimador": f"{type(est).__module__}.{type(est).__name__}"}
    return {"nota": f"{ruta.stat().st_size / 1e3:.0f} KB, ida y vuelta idéntica en {len(X_te):,} filas"
                    + ("" if dif_guardada is not None else "; sin predicciones_prueba.parquet con que compararla todavía")}


for _t in TAREAS:
    bloque(f"artefacto_{_t}", artefacto, _t)


# ============================================================================ 4. salidas de prueba de LightGBM (no va en vivo)
def salidas_lightgbm(tarea: str) -> dict:
    s = guardada(tarea, NUEVO)
    de = "predicciones_prueba.parquet"
    if s is None:               # 03_prueba.py aún no corrió: la tubería guardada da exactamente lo mismo
        f = corrida(MAESTRA, tarea, NUEVO)
        if f is None:
            raise Falta(f"la tabla maestra no tiene todavía la corrida {tarea} · {NUEVO} (semilla {SEMILLA})")
        try:
            pipe = leer(R / "modelos" / f"{clave_de(f)}.joblib", joblib.load)
        except ModuleNotFoundError as e:
            raise Falta(f"no se pudo abrir la tubería de LightGBM ({e})") from e
        s = salida_modelo(pipe, X_te, tarea)
        if tarea == "clasificacion":
            CLASES[NUEVO] = np.asarray(pipe.predict(X_te)).astype(int)
        de = f"modelos/{clave_de(f)}.joblib"
    elif tarea == "clasificacion":
        clase = guardada(tarea, NUEVO, "__clase")
        if clase is not None:
            CLASES[NUEVO] = np.asarray(clase).astype(int)
    SALIDAS[tarea, NUEVO] = np.asarray(s, dtype=float)
    return {"nota": f"de {de}", "de": de}


for _t in TAREAS:
    bloque(f"salidas_lightgbm_{_t}", salidas_lightgbm, _t)

# ============================================================================ 5. puntajes y pronósticos de prueba
for _t in TAREAS:
    tabla = TABLA_SALIDAS[_t]
    antes = [m for m in NUEVOS if m in tabla]
    for m in NUEVOS:
        if (_t, m) in SALIDAS:
            tabla[m] = SALIDAS[_t, m]
        elif m in tabla:                     # quedó de una corrida anterior y ya no hay con qué respaldarla
            del tabla[m]
    presentes = [m for m in NUEVOS if m in tabla]
    if presentes or antes:
        tabla.to_parquet(DATOS / ARCHIVO_SALIDAS[_t])
    if presentes:
        anotar_anadido(ARCHIVO_SALIDAS[_t], "ModeloOriginal/resultados/modelos y predicciones_prueba.parquet",
                       f"columnas {', '.join(presentes)}: el artefacto del modelo original evaluado en la prueba y la predicción "
                       "guardada de LightGBM")
print("salidas de prueba:", {t: [m for m in NUEVOS if m in TABLA_SALIDAS[t]] for t in TAREAS})


# ============================================================================ 6. filas nuevas de la tabla de líderes
def fila_lider(tarea: str, modelo: str) -> dict:
    f = corrida(MAESTRA, tarea, modelo)
    if f is None or (tarea, modelo) not in SALIDAS:
        raise Falta(f"faltan la corrida o las predicciones de prueba de {tarea} · {modelo}")
    y, s, pl = Y_TE[tarea], SALIDAS[tarea, modelo], pliegues_de(f)
    met = f["metricas_ext"] if isinstance(f["metricas_ext"], dict) else {}
    if tarea == "clasificacion":
        boot = estadistica.bootstrap_auc(y, s, n_boot=N_BOOT)
        c_op = motor.punto_operacion(s)
        cal = CALIBRACION.get(modelo, {})
        fila = {"modelo": modelo, "balanceo": f["balanceo"], "optimizador": f["optimizador"],
                "auc_ext_media": float(pl.mean()), "auc_ext_desv": float(pl.std(ddof=1)),
                "f1_ext_media": float(np.mean(met["f1"])) if "f1" in met else np.nan,
                "f1_ext_desv": float(np.std(met["f1"], ddof=1)) if "f1" in met else np.nan,
                "auc_prueba": float(roc_auc_score(y, s)), "auc_bca_inf": boot["ic_bca"][0], "auc_bca_sup": boot["ic_bca"][1],
                "auc_pr": float(average_precision_score(y, s)),
                "recall_op": float(recall_score(y, c_op)), "precision_op": float(precision_score(y, c_op)),
                "mcc_op": float(matthews_corrcoef(y, c_op)), "exactitud_bal_op": float(balanced_accuracy_score(y, c_op)),
                "f1_op": float(f1_score(y, c_op)),
                "vp_op": int(((y == 1) & (c_op == 1)).sum()), "fp_op": int(((y == 0) & (c_op == 1)).sum()),
                "fn_op": int(((y == 1) & (c_op == 0)).sum()), "vn_op": int(((y == 0) & (c_op == 0)).sum())}
        c_05 = CLASES.get(modelo)
        for k, a, b in (("vp_05", 1, 1), ("fp_05", 0, 1), ("fn_05", 1, 0), ("vn_05", 0, 0)):
            fila[k] = np.nan if c_05 is None else int(((y == a) & (c_05 == b)).sum())
        fila.update({"recall_10": float(recall_score(y, motor.punto_operacion(s, 0.10))),
                     "recall_50": float(recall_score(y, motor.punto_operacion(s, 0.50))),
                     "precision_10": float(precision_score(y, motor.punto_operacion(s, 0.10))),
                     "ece_isotonica": cal.get("ece_prueba_recalibrada", np.nan), "auc_isotonica": cal.get("auc_prueba_recalibrada", np.nan)})
    else:
        bo = estadistica.bootstrap_regresion(y, s, n_boot=N_BOOT)
        fila = {"modelo": modelo, "optimizador": f["optimizador"],
                "rmse_ext_media": float(pl.mean()), "rmse_ext_desv": float(pl.std(ddof=1)),
                "r2_ext_media": float(np.mean(met["r2"])) if "r2" in met else np.nan,
                "r2_ext_desv": float(np.std(met["r2"], ddof=1)) if "r2" in met else np.nan,
                "rmse_prueba": bo["rmse"]["estimacion"], "rmse_bca_inf": bo["rmse"]["ic_bca"][0], "rmse_bca_sup": bo["rmse"]["ic_bca"][1],
                "mae_prueba": bo["mae"]["estimacion"], "mape_prueba": float(np.mean(np.abs(y - s) / np.abs(y))),
                "r2_prueba": bo["r2"]["estimacion"], "r2_bca_inf": bo["r2"]["ic_bca"][0], "r2_bca_sup": bo["r2"]["ic_bca"][1]}
    fila["corrida_min"] = float(f["tiempo_total_s"]) / 60
    if tarea == "regresion":                 # el Model Confidence Set del informe solo tiene los ocho modelos del Entregable 3
        fila.update({"p_mcs": np.nan, "en_mcs": np.nan})
    return fila


# ============================================================================ 7. probabilidad recalibrada y conjuntos conformes
CALIBRACION: dict[str, dict] = {}


def calibrar(modelo: str) -> dict:
    """Lo mismo que preparar_artefactos.py hace con los ocho clasificadores del Entregable 3."""
    if ("clasificacion", modelo) not in SALIDAS:
        raise Falta(f"faltan las predicciones de prueba de clasificacion · {modelo}")
    s_oof, y_oof, y = fuera_de_pliegue("clasificacion", modelo), Y_TR["clasificacion"], Y_TE["clasificacion"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s_oof, y_oof)
    s_te = SALIDAS["clasificacion", modelo]
    p_te, p_oof = iso.predict(s_te), iso.predict(s_oof)
    entrada = {}
    for nivel in NIVELES:                    # conjunto conforme LAC (Sadinle et al., 2019) con la probabilidad recalibrada
        a = 1 - nivel
        no_conf = 1 - np.where(y_oof == 1, p_oof, 1 - p_oof)
        n = len(no_conf)
        q = float(np.quantile(no_conf, min(1.0, math.ceil((n + 1) * (1 - a)) / n), method="higher"))
        incluye1, incluye0 = (1 - p_te) <= q, p_te <= q
        cubre = np.where(y == 1, incluye1, incluye0)
        entrada[f"{nivel:.2f}"] = {"umbral": q, "cobertura_prueba": float(cubre.mean()),
                                   "tamano_medio": float((incluye1.astype(int) + incluye0.astype(int)).mean()),
                                   "fraccion_ambos": float((incluye1 & incluye0).mean())}
    cruda_en_01 = bool(s_te.min() >= 0 and s_te.max() <= 1)
    CALIBRACION[modelo] = {"x": iso.X_thresholds_, "y": iso.y_thresholds_, "nudos": int(len(iso.X_thresholds_)),
                           "ece_prueba_sin_recalibrar": ece(y, s_te) if cruda_en_01 else None,
                           "ece_prueba_recalibrada": ece(y, p_te), "brier_prueba_recalibrada": float(np.mean((p_te - y) ** 2)),
                           "auc_prueba": float(roc_auc_score(y, s_te)), "auc_prueba_recalibrada": float(roc_auc_score(y, p_te)),
                           "conformal": entrada}
    return {"nota": f"ECE recalibrada {CALIBRACION[modelo]['ece_prueba_recalibrada']:.4f}; cobertura al 90 % "
                    f"{entrada['0.90']['cobertura_prueba']:.3f}"}


for m in NUEVOS:
    bloque(f"calibracion_{m}", calibrar, m)
cal_json = json.loads((DATOS / "calibracion_modelos.json").read_text(encoding="utf-8"))
for m in NUEVOS:
    cal_json["modelos"].pop(m, None)
    if m in CALIBRACION:
        cal_json["modelos"][m] = CALIBRACION[m]
escribir_json(DATOS / "calibracion_modelos.json", cal_json)
if CALIBRACION:
    anotar_anadido("calibracion_modelos.json", "ModeloOriginal/resultados/fuera_de_pliegue/*.npy y las predicciones de prueba",
                   f"recalibración isotónica y conjunto conforme de {', '.join(CALIBRACION)}, con el mismo procedimiento")


def conformal_regresion(modelo: str) -> dict:
    if ("regresion", modelo) not in SALIDAS:
        raise Falta(f"faltan las predicciones de prueba de regresion · {modelo}")
    r = np.abs(Y_TR["regresion"] - fuera_de_pliegue("regresion", modelo))
    n, y, yhat = len(r), Y_TE["regresion"], SALIDAS["regresion", modelo]
    entrada = {}
    for nivel in NIVELES:
        q = float(np.quantile(r, min(1.0, math.ceil((n + 1) * nivel) / n), method="higher"))
        entrada[f"{nivel:.2f}"] = {"semiancho": q, "cobertura_prueba": float((np.abs(y - yhat) <= q).mean()), "ancho": 2 * q}
    if modelo == ORIGINAL:                   # lo que solo tiene el modelo original: su intervalo central y su conjunto
        pipe = TUBERIAS["regresion"]
        bdl = pipe.named_steps["modelo"]
        Z = pipe.named_steps["preprocesado"].transform(X_te)
        for nivel in NIVELES:
            bajo, alto = bdl.intervalo(Z, nivel)
            tramos = bdl.conjunto_residual(nivel)
            entrada[f"{nivel:.2f}"]["central"] = {"cobertura_prueba": float(np.mean((y >= bajo) & (y <= alto))),
                                                  "largo_medio": float(np.mean(alto - bajo))}
            entrada[f"{nivel:.2f}"]["conjunto"] = {"cobertura_prueba": float(bdl.conjunto_cubre(Z, y, nivel).mean()),
                                                   "largo": float((tramos[:, 1] - tramos[:, 0]).sum()), "tramos": int(len(tramos))}
    CONFORMAL[modelo] = entrada
    return {"nota": f"cobertura al 90 % {entrada['0.90']['cobertura_prueba']:.3f}"
                    + (f"; conjunto del modelo {entrada['0.90']['conjunto']['cobertura_prueba']:.3f} con largo "
                       f"{entrada['0.90']['conjunto']['largo']:.4f} en {entrada['0.90']['conjunto']['tramos']} tramos"
                       if modelo == ORIGINAL else "")}


CONFORMAL: dict[str, dict] = {}
for m in NUEVOS:
    bloque(f"conformal_{m}", conformal_regresion, m)
conf_json = json.loads((DATOS / "conformal.json").read_text(encoding="utf-8"))
for m in NUEVOS:
    conf_json["modelos"].pop(m, None)
    if m in CONFORMAL:
        conf_json["modelos"][m] = CONFORMAL[m]
escribir_json(DATOS / "conformal.json", conf_json)
if CONFORMAL:
    anotar_anadido("conformal.json", "ModeloOriginal/resultados/fuera_de_pliegue/*.npy y las predicciones de prueba",
                   f"intervalos conformes de {', '.join(CONFORMAL)}; bdl lleva además el intervalo central y el conjunto de "
                   "predicción del propio modelo (RegresorBDL.intervalo y conjunto_residual), con su cobertura y su largo en prueba")

# ============================================================================ 8. las dos filas nuevas de la tabla de líderes
for _t in TAREAS:
    filas = [r for r in (bloque(f"lider_{_t}_{m}", fila_lider, _t, m) for m in NUEVOS) if isinstance(r, dict)]
    for r in filas:
        r.pop("nota", None)
    if filas:
        pd.DataFrame(filas).to_csv(en_original(f"lideres_{_t}.csv"), index=False, lineterminator="\n")
        anotar(f"original/lideres_{_t}.csv", ["ModeloOriginal/resultados/experimentos.parquet", f"datos/{ARCHIVO_SALIDAS[_t]}"],
               f"las filas de {', '.join(r['modelo'] for r in filas)} con las columnas y las definiciones de "
               f"resultados/{_t}_prueba.csv (Resultados Entregable 3/generar.py): métrica externa de la validación anidada, "
               f"métrica de prueba con intervalo BCa ({N_BOOT} réplicas) y, en clasificación, el punto de operación del 25 %")


# ============================================================================ 9. resumen SHAP global del modelo original
def shap_original(tarea: str) -> dict:
    if tarea not in TUBERIAS:
        raise Falta(f"falta el artefacto de {tarea} · {ORIGINAL}")
    t1 = time.perf_counter()
    referencia = pd.read_parquet(DATOS / "shap" / f"{tarea}__xgboost.parquet")            # la misma muestra fija que los demás
    muestra = X_te.loc[referencia["fila"].to_numpy()]
    r = explicacion.explicar_bdl(TUBERIAS[tarea], muestra, tarea)
    t = r["aportes"].copy()
    t["__base"] = r["base"]
    t["__salida"] = r["salida"]
    t.index.name = "fila"
    t.reset_index().to_parquet(DATOS / "shap" / f"{tarea}__{ORIGINAL}.parquet", index=False)
    anotar(f"shap/{tarea}__{ORIGINAL}.parquet", f"artifacts/{tarea}/{ORIGINAL}/modelo.joblib",
           f"aportes exactos del modelo original (RegresorBDL.contribuciones) de {len(t)} clientes de prueba fijos, los mismos de "
           "los demás modelos")
    importancia = r["aportes"].abs().mean().sort_values(ascending=False)
    SHAP[tarea] = {"metodo": r["metodo"], "escala": r["escala"], "exacto": r["exacto"], "filas": int(len(t)),
                   "verificacion": {"eficiencia_max_dif": r["eficiencia"]}, "importancia_media_abs": importancia.to_dict(),
                   "segundos": time.perf_counter() - t1}
    return {"nota": f"{len(t)} filas, eficiencia {r['eficiencia']:.1e}; primeras: {', '.join(importancia.index[:3])}"}


SHAP: dict[str, dict] = {}
for _t in TAREAS:
    bloque(f"shap_{_t}", shap_original, _t)
indice_shap = json.loads((DATOS / "shap" / "indice.json").read_text(encoding="utf-8"))
for _t in TAREAS:
    indice_shap["modelos"].pop(f"{_t}__{ORIGINAL}", None)
    if _t in SHAP:
        indice_shap["modelos"][f"{_t}__{ORIGINAL}"] = SHAP[_t]
    elif (DATOS / "shap" / f"{_t}__{ORIGINAL}.parquet").exists():
        (DATOS / "shap" / f"{_t}__{ORIGINAL}.parquet").unlink()
escribir_json(DATOS / "shap" / "indice.json", indice_shap)
if SHAP:
    anotar_anadido("shap/indice.json", "preparar_original.py", f"las entradas de {', '.join(f'{t}__{ORIGINAL}' for t in SHAP)}")


# ============================================================================ 10. copias exactas de la corrección
def copiar(rel: str) -> dict:
    origen = R / rel
    if not origen.exists():
        raise Falta(f"falta {rel}")
    destino = en_original(rel)
    for _ in range(4):                      # si otra sesión lo está escribiendo, la copia puede quedar a medias
        shutil.copyfile(origen, destino)
        try:
            if rel.endswith(".json"):
                json.loads(destino.read_text(encoding="utf-8"))
            else:
                pd.read_csv(destino)
            break
        except Exception:  # noqa: BLE001
            time.sleep(1.5)
    else:
        destino.unlink(missing_ok=True)
        ESCRITOS.discard(rel)
        raise Falta(f"{rel} no se pudo leer completo (se está escribiendo)")
    anotar(f"original/{rel}", f"ModeloOriginal/resultados/{rel}", "copia exacta")
    return {"modificado": datetime.fromtimestamp(origen.stat().st_mtime).isoformat(timespec="seconds"), "sha256": sha(destino)}


for rel in COPIAS:
    bloque(Path(rel).stem if "/" not in rel else rel.replace("/", "_").replace(".csv", ""), copiar, rel)


# ============================================================================ 11. matriz de pliegues y cd.json con los diez
def matriz_pliegues(tarea: str):
    """La métrica de cada pliegue externo de los diez modelos: la de 04_pruebas_hipotesis.py si ya existe; si no, la misma
    matriz armada con la tabla maestra (repetición de la semilla 42), siempre que estén los diez modelos."""
    ruta = DESTINO / f"pliegues_{tarea}.csv"
    if ruta.exists():
        return pd.read_csv(ruta, index_col=0), f"datos/original/pliegues_{tarea}.csv"
    modelos = [m for m in MAESTRA.loc[(MAESTRA["tarea"] == tarea) & (MAESTRA["semilla"] == SEMILLA), "modelo"]
               if m in NOMBRES_DIEZ]
    if ORIGINAL not in modelos or len(set(modelos)) < 10:
        raise Falta(f"falta pliegues_{tarea}.csv y la tabla maestra todavía no tiene los diez modelos de {tarea} "
                    f"(hay {len(set(modelos))})")
    M = pd.DataFrame({m: pliegues_de(corrida(MAESTRA, tarea, m)) for m in modelos})
    M.index = [f"{SEMILLA}-{i + 1}" for i in range(len(M))]
    M.to_csv(en_original(f"pliegues_{tarea}.csv"), lineterminator="\n")
    anotar(f"original/pliegues_{tarea}.csv", "ModeloOriginal/resultados/experimentos.parquet",
           "la matriz de la métrica por pliegue externo de los diez modelos (semilla 42), armada con la tabla maestra porque "
           "04_pruebas_hipotesis.py todavía no la había escrito; es la misma matriz")
    return M, "la tabla maestra (04_pruebas_hipotesis.py aún no había escrito la matriz)"


NOMBRES_DIEZ = {"knn", "naive_bayes", "logistica", "arbol", "random_forest", "xgboost", "svm", "ebm", "ridge", "lasso", "svr",
                "lightgbm", "bdl"}


def diferencia_critica(tarea: str) -> dict:
    M, de = matriz_pliegues(tarea)
    M = M.dropna()
    n, k = M.shape
    mayor = MAYOR_ES_MEJOR[tarea]
    rangos = (-M if mayor else M).rank(axis=1, method="average").mean()
    chi2 = 12 * n / (k * (k + 1)) * (float((rangos ** 2).sum()) - k * (k + 1) ** 2 / 4)
    f_id = (n - 1) * chi2 / (n * (k - 1) - chi2)
    q = float(stats.studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2))      # rango estudentizado entre raíz de 2 (Demšar, 2006)
    CD[tarea] = {"pliegues": int(n), "modelos": int(k), "metrica": METRICA[tarea],
                 "repeticiones": sorted({int(str(i).split("-")[0]) for i in M.index}),
                 "rango_medio": rangos.sort_values().to_dict(), "chi2_friedman": chi2,
                 "p_friedman": float(stats.chi2.sf(chi2, k - 1)), "f_iman_davenport": f_id,
                 "p_iman_davenport": float(stats.f.sf(f_id, k - 1, (k - 1) * (n - 1))),
                 "gl": [k - 1, (k - 1) * (n - 1)], "q_alfa": q, "alfa": ALFA,
                 "diferencia_critica": q * math.sqrt(k * (k + 1) / (6 * n)),
                 "metricas_pliegue": M.to_dict(orient="list"), "bloques": [str(i) for i in M.index], "de": de}
    nota = f"{k} modelos × {n} pliegues, DC = {CD[tarea]['diferencia_critica']:.3f}"
    ruta_v = DESTINO / "veredicto.json"
    if ruta_v.exists():                      # la misma prueba que 04_pruebas_hipotesis.py: tiene que dar lo mismo
        fr = json.loads(ruta_v.read_text(encoding="utf-8")).get(tarea, {}).get("friedman")
        if fr:
            dif = max(abs(fr["chi2"] - chi2), abs(fr["diferencia_critica_nemenyi"] - CD[tarea]["diferencia_critica"]),
                      max(abs(fr["rangos_medios"][m] - v) for m, v in rangos.items()))
            assert dif < 1e-6, f"cd.json no coincide con veredicto.json en {tarea}: diferencia {dif:.2e}"
            nota += "; coincide con veredicto.json"
    return {"nota": nota, "de": de}


CD: dict[str, dict] = {}
for _t in TAREAS:
    bloque(f"cd_{_t}", diferencia_critica, _t)
if CD:
    cd_json = json.loads((DATOS / "cd.json").read_text(encoding="utf-8"))
    cd_json.update(CD)
    escribir_json(DATOS / "cd.json", cd_json)
    anotar("cd.json", ["datos/original/pliegues_<tarea>.csv (ModeloOriginal/resultados)"]
           + ([] if len(CD) == len(TAREAS) else ["Resultados Entregable 3: datos/pliegues_auc.csv y pliegues_rmse.csv"]),
           f"rangos medios por pliegue externo, Friedman, Iman-Davenport y diferencia crítica de Nemenyi (alfa 0,05) con los diez "
           f"modelos en {' y '.join(CD)}" + ("" if len(CD) == len(TAREAS) else "; la otra tarea sigue con los ocho del Entregable 3"))


# ============================================================================ 12. tabla maestra sin trazas ni rutas locales
def experimentos() -> dict:
    if MAESTRA.empty:
        raise Falta("falta experimentos.parquet")
    t = tabla_maestra(R / "experimentos.parquet")
    cols = ["tarea", "modelo", "balanceo", "optimizador", "semilla", "k_ext", "k_int", "presupuesto", "n_evaluaciones",
            "n_evaluaciones_final", "media_ext", "desv_ext", "puntajes_ext", "hiperparametros", "params_por_pliegue", "tiempo_total_s",
            "tiempo_busqueda_s", "tiempo_ajuste_final_s", "origen", "registrado_en"]
    t = t[[c for c in cols if c in t]].copy()
    t.insert(t.columns.get_loc("media_ext"), "metrica_externa", t["media_ext"].abs())
    for col in ("puntajes_ext", "hiperparametros", "params_por_pliegue"):      # los hiperparámetros que eligió cada pliegue externo
        if col in t:
            t[col] = t[col].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                                else json.dumps(_limpio(_json_o_valor(v)), ensure_ascii=False))
    t.to_csv(en_original("experimentos.csv"), index=False, lineterminator="\n")
    anotar("original/experimentos.csv", "ModeloOriginal/resultados/experimentos.parquet",
           "las corridas «ok» de la tabla maestra de la corrección, sin las trazas de la búsqueda ni las rutas locales")
    semillas = {tarea: sorted(int(s) for s in t.loc[(t["tarea"] == tarea) & (t["modelo"] == ORIGINAL), "semilla"].unique())
                for tarea in TAREAS}
    return {"nota": f"{len(t)} corridas; semillas del original: {semillas}", "semillas_original": semillas}


bloque("experimentos_csv", experimentos)


# ============================================================================ 13. costo de los dos modelos nuevos
def costo() -> dict:
    ruta_s = DESTINO / "semillas.csv"
    sem = pd.read_csv(ruta_s) if ruta_s.exists() else None
    ruta_m = DESTINO / "metricas_prueba.csv"
    mp = pd.read_csv(ruta_m) if ruta_m.exists() else None
    filas = []
    for tarea in TAREAS:
        for m in NUEVOS:
            f = corrida(MAESTRA, tarea, m)
            if f is None:
                continue
            fila = {"tarea": tarea, "modelo": m, "ajuste_s": np.nan, "inferencia_s": np.nan, "corrida_min": float(f["tiempo_total_s"]) / 60,
                    "tamano_mb": np.nan, "parametros": np.nan, "presupuesto": int(f["presupuesto"]),
                    "evaluaciones": int(f["n_evaluaciones"]) + (0 if pd.isna(f["n_evaluaciones_final"]) else int(f["n_evaluaciones_final"])),
                    "de": []}
            s = None if sem is None else sem[(sem["tarea"] == tarea) & (sem["modelo"] == m)]
            if s is not None and len(s):
                fila.update({"ajuste_s": float(s["ajuste_s"].mean()), "inferencia_s": float(s["inferencia_s"].mean()),
                             "tamano_mb": float(s["tamano_kb"].mean()) / 1024})
                fila["de"].append(f"semillas.csv (media de {len(s)} semillas)")
            else:
                if mp is not None and "inferencia_s" in mp:
                    x = mp[(mp["tarea"] == tarea) & (mp["modelo"] == m)]["inferencia_s"]
                    if len(x) and pd.notna(x.iloc[0]):
                        fila["inferencia_s"] = float(x.iloc[0])
                        fila["de"].append("metricas_prueba.csv (inferencia)")
                if m == ORIGINAL and tarea in ARTEFACTO:
                    if pd.isna(fila["inferencia_s"]):
                        fila["inferencia_s"] = ARTEFACTO[tarea]["inferencia_s"]
                        fila["de"].append("medida al armar el artefacto (inferencia)")
                    fila["tamano_mb"] = ARTEFACTO[tarea]["ruta"].stat().st_size / 1024 ** 2
                    fila["de"].append("artifacts/ (tamaño)")
                else:
                    ruta = R / "modelos" / f"{clave_de(f)}.joblib"
                    if ruta.exists():
                        fila["tamano_mb"] = ruta.stat().st_size / 1024 ** 2
                        fila["de"].append("resultados/modelos (tamaño)")
            if m == ORIGINAL and tarea in TUBERIAS:
                fila["parametros"] = parametros_bdl(TUBERIAS[tarea].named_steps["modelo"])
            fila["de"] = "; ".join(fila["de"])
            filas.append(fila)
    if not filas:
        raise Falta("la tabla maestra no tiene todavía corridas de bdl ni de lightgbm")
    pd.DataFrame(filas).to_csv(en_original("costo.csv"), index=False, lineterminator="\n")
    anotar("original/costo.csv", ["ModeloOriginal/resultados/experimentos.parquet", "semillas.csv", "metricas_prueba.csv", "artifacts/"],
           "tiempo de ajuste, de inferencia sobre la prueba y de la corrida anidada, tamaño serializado y parámetros de bdl y "
           "lightgbm; la columna «de» dice de dónde sale cada cifra (el ajuste solo se mide en 03b_semillas.py)")
    return {"nota": f"{len(filas)} filas" + ("" if sem is not None else "; sin semillas.csv el tiempo de ajuste queda pendiente")}


def parametros_bdl(est) -> int:
    """Números que guarda el modelo original para predecir: los valores de sus curvas (o de las hojas de sus árboles), los
    cortes, el intercepto y, con mezcla, los pesos, los desplazamientos y la dispersión."""
    reg = explicacion.regresor_bdl(est)
    base = reg if reg is not None else est
    cortes = int(sum(len(c) for c in base.cortes_))
    if getattr(base, "formas_", None) is not None:
        valores = int(sum(len(f) for f in base.formas_))
    else:
        valores = int(np.asarray(base.arboles_valor_).size)
    mezcla = 2 * int(reg.n_componentes_) + 1 if reg is not None else 0
    return valores + cortes + 1 + mezcla


bloque("costo", costo)


# ============================================================================ 14. el modelo final por dentro
def modelo_final(tarea: str) -> dict:
    if tarea not in TUBERIAS:
        raise Falta(f"falta el artefacto de {tarea} · {ORIGINAL}")
    pipe = TUBERIAS[tarea]
    pre, est = pipe.named_steps["preprocesado"], pipe.named_steps["modelo"]
    reg = explicacion.regresor_bdl(est)
    f = ARTEFACTO[tarea]["fila"]
    salida = {"tarea": tarea, "clase": type(est).__name__, "modo": getattr(est, "modo_", None),
              "hiperparametros": {k: v for k, v in est.get_params().items()},
              "hiperparametros_buscados": f["hiperparametros"], "parametros": parametros_bdl(est),
              "columnas": int(len(pre.get_feature_names_out())), "variables": len(COLUMNAS)}
    if reg is None:                          # clasificación en modo directo: boosting logístico, sin mezcla
        h = est.historial_
        salida.update({"n_arboles": int(est.n_arboles_), "intercepto": float(est.intercepto_),
                       "historial": {"perdida": h["perdida"], "perdida_validacion": h["perdida_validacion"]},
                       "verificaciones": {"perdida_no_sube": bool(np.all(np.diff(h["perdida"]) <= 1e-12)),
                                          "mayor_subida": float(np.max(np.diff(h["perdida"]))) if len(h["perdida"]) > 1 else 0.0}})
        escribir_json(en_original(f"modelo_{tarea}.json"), salida)
        return {"nota": "modo directo (sin mezcla)"}
    h = reg.historial_
    vero = np.asarray(h["verosimilitud"], dtype=float)
    pasos = np.diff(vero)
    nombres = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    de_variable = explicacion.columnas_por_variable(pre)
    esq = {v["nombre"]: v for v in ESQUEMA}
    salida.update({
        "umbral": float(est.umbral_) if getattr(est, "modo_", None) == "excedencia" else None,
        "filas": {"ajuste": int(reg.n_ajuste_), "validacion": int(reg.n_validacion_), "calibracion": int(reg.n_calibracion_)},
        "intercepto": float(reg.intercepto_), "n_calentamiento": int(reg.n_calentamiento_), "n_arboles": int(reg.n_arboles_),
        "arboles_por_vuelta": int(reg.arboles_por_vuelta), "vueltas_em": int(len(vero)),
        "vueltas_usadas": int(round((reg.n_arboles_ - reg.n_calentamiento_) / reg.arboles_por_vuelta)),
        "K": int(reg.n_componentes_), "pesos": reg.pi_, "desplazamientos": reg.centros_,
        "separacion_media": float(np.diff(reg.centros_).mean()) if reg.n_componentes_ > 1 else None,
        "sigma": float(reg.sigma_), "sigma_validacion": float(reg.sigma_validacion_), "sigma_robusta": float(reg.sigma_robusta_),
        "desplazamiento_medio": float(reg.desplazamiento_medio_), "aditivo": reg.formas_ is not None,
        "historial": {"verosimilitud": vero, "verosimilitud_por_fila": vero / reg.n_ajuste_,
                      "error_validacion": h["error_validacion"], "error_calentamiento": h["error_calentamiento"]},
        "bic": [{"K": int(b["K"]), "verosimilitud": float(b["verosimilitud"]), "bic": float(b["bic"]), "utiles": int(b["utiles"])}
                for b in h["bic"]],
        "verificaciones": {"verosimilitud_no_decrece": bool(len(pasos) == 0 or pasos.min() >= -1e-9 * max(1.0, abs(vero).max())),
                           "menor_paso": float(pasos.min()) if len(pasos) else None,
                           "vueltas": int(len(vero))},
    })
    # el conjunto de predicción en la escala del residuo (no depende del cliente) y el histograma de los residuos de calibración
    salida["conjunto"] = {}
    for nivel in NIVELES:
        tramos = reg.conjunto_residual(nivel)
        salida["conjunto"][f"{nivel:.2f}"] = {"tramos": tramos, "largo": float((tramos[:, 1] - tramos[:, 0]).sum()),
                                              "umbral_densidad": float(reg.umbral_densidad(nivel))}
    cuenta, bordes = np.histogram(reg.residuos_calibracion_, bins=BARRAS_RESIDUO)
    salida["residuos_calibracion"] = {"n": int(len(reg.residuos_calibracion_)), "bordes": bordes, "cuenta": cuenta,
                                      "minimo": float(reg.residuos_calibracion_[0]), "maximo": float(reg.residuos_calibracion_[-1])}
    # importancia por variable original (la del resumen SHAP: media del aporte absoluto en la muestra fija de la prueba)
    imp = SHAP.get(tarea, {}).get("importancia_media_abs", {})
    total = sum(imp.values()) or 1.0
    salida["importancia"] = [{"variable": v, "nombre": esq[v].get("nombre_visible") or v, "importancia": float(x), "parte": float(x / total)}
                             for v, x in imp.items()]
    # las curvas de las variables numéricas principales, con los cortes en las unidades originales
    salida["formas"] = []
    if reg.formas_ is not None and "num" in pre.named_transformers_:
        escalador, numericas = pre.named_transformers_["num"], list(pre.transformers_[0][2])
        for v in [x for x in imp if x in numericas and esq[x]["tipo"] in ("entera", "real")][:N_FORMAS]:
            j = nombres.index(v)
            assert de_variable[j] == v
            media, desv = float(escalador.mean_[numericas.index(v)]), float(escalador.scale_[numericas.index(v)])
            peso = reg.conteo_bins_[j] / reg.conteo_bins_[j].sum()
            forma = reg.formas_[j] - float(reg.formas_[j] @ peso)
            salida["formas"].append({"variable": v, "nombre": esq[v].get("nombre_visible") or v, "unidad": esq[v].get("unidad"),
                                     "minimo": esq[v]["min"], "maximo": esq[v]["max"], "cortes": reg.cortes_[j] * desv + media,
                                     "valores": forma, "filas": reg.conteo_bins_[j]})
    escribir_json(en_original(f"modelo_{tarea}.json"), salida)
    anotar(f"original/modelo_{tarea}.json", f"artifacts/{tarea}/{ORIGINAL}/modelo.joblib",
           "el modelo final por dentro: hiperparámetros, reparto de filas, K, pesos, desplazamientos, dispersiones, árboles, "
           "historial de convergencia (historial_), tabla del BIC, conjunto de predicción en la escala del residuo, histograma de "
           "los residuos de calibración, importancia por variable y curvas (formas_) de las variables numéricas principales")
    return {"nota": f"K = {reg.n_componentes_}, {reg.n_arboles_} árboles ({reg.n_calentamiento_} de calentamiento), "
                    f"{len(vero)} vueltas de EM; verosimilitud no decrece: {salida['verificaciones']['verosimilitud_no_decrece']}"}


for _t in TAREAS:
    bloque(f"modelo_{_t}", modelo_final, _t)


# ============================================================================ 15. conjuntos de datos adicionales
def adicionales() -> dict:
    carpeta = R / "adicionales"
    if not carpeta.is_dir():
        raise Falta("falta adicionales/")
    filas = []
    for c in sorted(p for p in carpeta.iterdir() if p.is_dir()):
        if not (c / "experimentos.parquet").exists() or not (c / "ficha.json").exists():
            continue
        try:
            ficha, t = leer_json(c / "ficha.json"), tabla_maestra(c / "experimentos.parquet")
        except Falta:
            continue
        ruta_mp = c / "metricas_prueba.csv"
        mp = pd.read_csv(ruta_mp) if ruta_mp.exists() else pd.DataFrame(columns=["tarea", "modelo"])
        for tarea in TAREAS:
            mayor = MAYOR_ES_MEJOR[tarea]
            pl = {str(f["modelo"]): pliegues_de(f) for _, f in t[(t["tarea"] == tarea) & (t["semilla"] == SEMILLA)].iterrows()
                  if f["modelo"] in NOMBRES_DIEZ}
            base = {"conjunto": c.name, "nombre": ficha.get("nombre", c.name), "n": ficha.get("n"), "p": ficha.get("p"),
                    "omitida": ficha.get("omitida"), "objetivo": ficha.get("objetivo"), "n_train": ficha.get("n_train"),
                    "n_test": ficha.get("n_test"), "prevalencia_train": ficha.get("prevalencia_train"),
                    "sha256": ficha.get("sha256"), "tarea": tarea, "modelos": len(pl), "completo": len(pl) == 10}
            if ORIGINAL not in pl or len(pl) < 3:
                filas.append(base)
                continue
            medias = {m: float(v.mean()) for m, v in pl.items()}
            otros = {m: v for m, v in medias.items() if m != ORIGINAL}
            mejor = (max if mayor else min)(otros, key=otros.get)
            signo = 1.0 if mayor else -1.0
            ventaja = signo * (medias[ORIGINAL] - medias[mejor])
            tc = pruebas.t_corregida(pl[ORIGINAL], pl[mejor])
            col = "auc" if tarea == "clasificacion" else "rmse"
            prueba = mp[mp["tarea"] == tarea].set_index("modelo")[col] if col in mp else pd.Series(dtype=float)
            es_mejor = bool(ventaja > 0)
            base.update({"mejor_benchmark": mejor, "benchmark_media": medias[mejor], "benchmark_desv": float(pl[mejor].std(ddof=1)),
                         "original_media": medias[ORIGINAL], "original_desv": float(pl[ORIGINAL].std(ddof=1)),
                         "puesto": int(pd.Series(medias).rank(ascending=not mayor)[ORIGINAL]), "ventaja": float(ventaja),
                         "ventaja_relativa": float(ventaja / abs(medias[mejor])), "t": tc["t"], "p_t_corregida": tc["p"],
                         "es_mejor": es_mejor, "peor_significativo": bool((not es_mejor) and tc["p"] < ALFA),
                         "prueba_benchmark": float(prueba[mejor]) if mejor in prueba.index else np.nan,
                         "prueba_original": float(prueba[ORIGINAL]) if ORIGINAL in prueba.index else np.nan})
            filas.append(base)
    if not filas:
        raise Falta("adicionales/ no tiene todavía ningún conjunto con resultados")
    t = pd.DataFrame(filas)
    t.to_csv(en_original("adicionales.csv"), index=False, lineterminator="\n")
    anotar("original/adicionales.csv", "ModeloOriginal/resultados/adicionales/<conjunto>/experimentos.parquet, ficha.json y metricas_prueba.csv",
           "el modelo original frente al mejor benchmark de cada conjunto adicional y tarea: media ± desviación de los cinco "
           "pliegues externos, puesto, ventaja, prueba t corregida de Nadeau-Bengio y métrica de prueba (el cálculo de "
           "experiments/08_tablas_libro.py); «completo» dice si ya están los diez modelos")
    completos = sorted(set(t.loc[t["completo"], "conjunto"]))
    return {"nota": f"{t['conjunto'].nunique()} conjuntos con resultados; con los diez modelos: {completos or 'ninguno todavía'}",
            "conjuntos": sorted(t["conjunto"].unique()), "completos": completos}


bloque("adicionales", adicionales)


# ============================================================================ 16. matriz de novedad
def literatura() -> dict:
    """La matriz de novedad y la búsqueda por citación, con el mismo cálculo que las Tablas 4.1 y 4.2 del libro
    (ModeloOriginal/src/informe.py, literatura): la consulta vigente (busqueda.json), la lectura de títulos y resúmenes
    anotada a mano (literatura_relevancia.csv) y las copias con fecha de las consultas anteriores (busqueda_<fecha>.json),
    que dicen cuándo se hizo cada consulta, con qué cadenas, qué se añadió después y cuántos resultados cambiaron."""
    carpeta = R / "literatura"
    d = leer_json(carpeta / "busqueda.json")
    rel = leer(MO / "experiments" / "literatura_relevancia.csv", pd.read_csv).set_index(["base", "cadena"])
    bases = {"openalex": "OpenAlex", "arxiv": "arXiv", "crossref": "Crossref"}
    tope = d["revisados_por_cadena"]
    anteriores = {f: leer_json(carpeta / f"busqueda_{f}.json") for f in sorted(d.get("consultas_anteriores", []))
                  if (carpeta / f"busqueda_{f}.json").exists()}

    def desde(seccion: str, clave: str) -> str:
        """La fecha de la primera consulta que ya traía ese antecedente o ese trabajo."""
        return next((f for f, a in anteriores.items() if clave in a.get(seccion, {})), d["fecha_de_consulta"])

    matriz, comparables, entradas_nuevas = [], 0, 0
    for clave, c in d["cadenas"].items():
        previas = [a["cadenas"][clave] for a in anteriores.values() if clave in a["cadenas"]]
        for base, nombre in bases.items():
            r = c[base]
            if (base, clave) not in rel.index:
                raise Falta(f"experiments/literatura_relevancia.csv no tiene la lectura de {base} · {clave}")
            matriz.append({"base": nombre, "cadena": clave, "tema": c["tema"], "consulta": r.get("cadena"), "resultados": r.get("total"),
                           "revisados": min(len(r.get("primeros", [])), tope), "relevantes": int(rel.loc[(base, clave), "relevantes"]),
                           "observacion": rel.loc[(base, clave), "observacion"]})
            for previa in previas:                   # lo que entró entre los primeros desde una consulta anterior
                if previa.get(base, {}).get("total") is None:      # esa consulta falló: no hay con qué comparar
                    continue
                antes = {(x.get("titulo") or "").strip().lower() for x in previa[base].get("primeros", [])}
                entradas_nuevas += sum((x.get("titulo") or "").strip().lower() not in antes for x in r.get("primeros", []))
                comparables += 1
    consultas = [{"fecha": f, "cadenas": list(a["cadenas"])} for f, a in anteriores.items()]
    consultas.append({"fecha": d["fecha_de_consulta"], "cadenas": list(d["cadenas"])})
    adelante = [{"antecedente": n, "citado_por": s["citado_por"], "sobre_fuga": s["citan_y_tratan_fuga"], "desde": desde("citacion", n)}
                for n, s in d["citacion"].items() if "citado_por" in s]
    atras = [{"trabajo": n, "referencias": s["referencias"], "sobre_mezclas": s["sobre_mezclas_o_clases_latentes"],
              "sobre_fuga": s["sobre_fuga"], "mezcla_y_fuga": s["que_combinan_mezcla_y_fuga"], "desde": desde("citacion_hacia_atras", n)}
             for n, s in d["citacion_hacia_atras"].items() if "referencias" in s]
    repos = [{"consulta": k, "total": v["total"]} for k, v in d.get("repositorios", {}).items() if v.get("total") is not None]
    escribir_json(en_original("literatura.json"), {
        "fecha_de_consulta": d["fecha_de_consulta"], "consultas": consultas,
        "comparacion": {"celdas": comparables, "entradas_nuevas": entradas_nuevas} if anteriores else None,
        "revisados_por_cadena": tope, "bases": list(bases.values()), "cadenas": len(d["cadenas"]), "matriz": matriz,
        "citacion_adelante": adelante, "citacion_atras": atras, "repositorios": repos})
    anotar("original/literatura.json", ["ModeloOriginal/resultados/literatura/busqueda.json"]
           + [f"ModeloOriginal/resultados/literatura/busqueda_{f}.json" for f in anteriores]
           + ["ModeloOriginal/experiments/literatura_relevancia.csv"],
           "la matriz de novedad (base × cadena: resultados, revisados, relevantes y observación), la fecha y las cadenas de cada "
           "consulta, lo que cambió entre fechas, la búsqueda por citación en los dos sentidos (con la consulta en que entró cada "
           "trabajo) y la consulta de repositorios, con el mismo cálculo que ModeloOriginal/src/informe.py (Tablas 4.1 y 4.2 del libro)")
    return {"nota": f"{len(matriz)} celdas de la matriz ({len(d['cadenas'])} cadenas), consulta del {d['fecha_de_consulta']}"
                    + (f"; consultas anteriores: {', '.join(anteriores)}" if anteriores else "")}


bloque("literatura", literatura)


# ============================================================================ 17. metadata.json e indice.json de bdl
def metadata(tarea: str) -> dict:
    if tarea not in ARTEFACTO:
        raise Falta(f"falta el artefacto de {tarea} · {ORIGINAL}")
    a = ARTEFACTO[tarea]
    f = a["fila"]
    pl = pliegues_de(f)
    ruta_l = DESTINO / f"lideres_{tarea}.csv"
    prueba = {}
    if ruta_l.exists():
        fila = pd.read_csv(ruta_l).set_index("modelo")
        if ORIGINAL in fila.index:
            prueba = {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else v))
                      for k, v in fila.loc[ORIGINAL].items()}
    ruta_c = DESTINO / "costo.csv"
    c = None
    if ruta_c.exists():
        c = pd.read_csv(ruta_c)
        c = c[(c["tarea"] == tarea) & (c["modelo"] == ORIGINAL)]
        c = c.iloc[0] if len(c) else None
    est = TUBERIAS[tarea].named_steps["modelo"]
    meta = {
        "tarea": tarea, "modelo": ORIGINAL, "estimador": a["estimador"], "balanceo": f["balanceo"], "optimizador": f["optimizador"],
        "semilla": SEMILLA, "hiperparametros": f["hiperparametros"], "versiones": VERSIONES,
        "notas": ["Modelo original del proyecto (Boosting con Desplazamiento Latente), programado desde cero: "
                  "src/original_model.py es una copia exacta de ModeloOriginal/src/original_model.py "
                  f"(sha256 {HUELLA_MODELO}).",
                  "Para cargarlo y predecir solo hacen falta numpy, scipy y scikit-learn."]
        + (["En clasificación recibió, además de la etiqueta, el puntaje continuo de las filas de entrenamiento "
            "(fit(X, y, puntaje=s)): es el mecanismo que se evalúa."] if tarea == "clasificacion" else []),
        "entrada": {"columnas": COLUMNAS, "esquema": ESQUEMA, "preprocesado": META_BASE["entrada"]["preprocesado"]},
        "salida": ("probabilidad de riesgo alto (predict_proba): la probabilidad de excedencia de la mezcla, P(s > τ | x)"
                   if getattr(est, "modo_", None) == "excedencia" else "probabilidad de riesgo alto (predict_proba)")
        if tarea == "clasificacion" else "churn_probability prevista (predict): la media, F(x) + Σ π_k c_k",
        "metricas_bucle_externo": {"metrica": METRICA[tarea], "media": float(pl.mean()), "desviacion": float(pl.std(ddof=1)),
                                   "pliegues_externos": int(f["k_ext"]), "pliegues_internos": int(f["k_int"])},
        "metricas_prueba": prueba,
        "computo": {"ajuste_s": None if c is None or pd.isna(c["ajuste_s"]) else float(c["ajuste_s"]), "ajuste_desv_s": None,
                    "inferencia_prueba_s": float(c["inferencia_s"]) if c is not None and pd.notna(c["inferencia_s"]) else a["inferencia_s"],
                    "n_prueba": int(len(X_te)), "carga_artefacto_s": a["carga_s"], "inferencia_medida_aqui_s": a["inferencia_s"],
                    "busqueda_s": float(f["tiempo_busqueda_s"]), "corrida_s": float(f["tiempo_total_s"]),
                    "presupuesto": int(f["presupuesto"]), "de": None if c is None or pd.isna(c["de"]) else c["de"]},
        "tuberia": "sklearn.pipeline.Pipeline con dos pasos (preprocesado y modelo); la de la corrección era una "
                   "imblearn.pipeline.Pipeline sin muestreador, con los mismos pasos (diferencia de predicciones: 0)",
        "ida_y_vuelta": {"filas": int(len(X_te)), "identicas": True, "max_dif_memoria": 0.0,
                         "max_dif_predicciones_guardadas": a["dif_guardada"], "max_dif_motor_numpy": None},
        "origen": {"archivo": a["origen"].name, "sha256_origen": sha(a["origen"]), "codigo": "src/original_model.py",
                   "sha256_codigo": HUELLA_MODELO},
        "generado": AHORA,
    }
    escribir_json(a["ruta"].parent / "metadata.json", meta)
    INDICE.append({"tarea": tarea, "modelo": ORIGINAL, "ruta": f"{tarea}/{ORIGINAL}/modelo.joblib", "bytes": a["ruta"].stat().st_size,
                   "sha256": sha(a["ruta"]), "estimador": a["estimador"], "balanceo": f["balanceo"], "optimizador": f["optimizador"],
                   "metrica_externa": meta["metricas_bucle_externo"], "ida_y_vuelta": "idénticas"})
    return {"nota": f"{METRICA[tarea]} externo {pl.mean():.5f} ± {pl.std(ddof=1):.5f}"}


INDICE: list[dict] = []
for _t in TAREAS:
    if bloque(f"metadata_{_t}", metadata, _t) is None and (ARTEFACTOS / _t / ORIGINAL).is_dir():
        shutil.rmtree(ARTEFACTOS / _t / ORIGINAL)              # quedó de una corrida anterior y ya no se puede respaldar
indice = json.loads((ARTEFACTOS / "indice.json").read_text(encoding="utf-8"))
indice["artefactos"] = [a for a in indice["artefactos"] if a["modelo"] != ORIGINAL] + INDICE
indice["original"] = {"generado": AHORA, "versiones": VERSIONES, "codigo": "src/original_model.py", "sha256_codigo": HUELLA_MODELO}
escribir_json(ARTEFACTOS / "indice.json", indice)
print(f"artifacts/indice.json: {len(indice['artefactos'])} artefactos ({len(INDICE)} del modelo original)")

# ============================================================================ 18. estado, limpieza y procedencia
for viejo in sorted(p for p in DESTINO.rglob("*") if p.is_file()):
    rel = viejo.relative_to(DESTINO).as_posix()
    if rel not in ESCRITOS and rel != "estado.json":
        viejo.unlink()
        print(f"  se retira datos/original/{rel} (ya no tiene respaldo en la corrección)")
ruta_v = DESTINO / "veredicto.json"
repeticiones = {}
if ruta_v.exists():
    repeticiones = {t: v.get("repeticiones") for t, v in json.loads(ruta_v.read_text(encoding="utf-8")).items()}
escribir_json(DESTINO / "estado.json", {
    "generado": AHORA, "de": "ModeloOriginal/resultados", "codigo": {"archivo": "src/original_model.py", "sha256": HUELLA_MODELO},
    "en_vivo": sorted(ARTEFACTO), "repeticiones": repeticiones, "bloques": ESTADO,
    "pendientes": {k: v["motivo"] for k, v in ESTADO.items() if not v.get("listo")}})
anotar("original/estado.json", "preparar_original.py", "qué insumo de la corrección había y cuál faltaba en la última corrida")

o = json.loads((DATOS / "origen.json").read_text(encoding="utf-8"))
previos = {a["archivo"]: a for a in o["archivos"]}
for rel in [k for k in previos if k.startswith("original/") or (k.startswith("shap/") and k.endswith(f"__{ORIGINAL}.parquet"))]:
    del previos[rel]
for rel, d in ORIGEN.items():
    ruta = DATOS / rel
    if not ruta.exists():
        continue
    if d.get("anade") and rel in previos:    # se conserva de dónde salía y se añade lo de la corrección
        antes = previos[rel]
        de = [x for x in antes["de"] if not x.startswith("Corrección: ")] + [f"Corrección: {d['de']}"]
        como = antes["como"].split(MARCA_ORIGEN)[0] + MARCA_ORIGEN + d["como"]
    else:
        de, como = ([d["de"]] if isinstance(d["de"], str) else d["de"]), d["como"]
    previos[rel] = {"archivo": rel, "bytes": ruta.stat().st_size, "sha256": sha(ruta), "de": de, "como": como}
o["archivos"] = sorted(previos.values(), key=lambda a: a["archivo"])
o["original"] = AHORA
(DATOS / "origen.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")

# ============================================================================ 19. la ficha del modelo y la declaración de los datos
# MODEL_CARD.md y DATA_STATEMENT.md citan cifras de datos/original/ (el veredicto, las métricas, la auditoría de la recencia):
# se vuelven a generar aquí para que no queden atrás de los datos. Va en otro proceso porque generar_fichas.py importa los
# módulos de la app y aquí «src» es el paquete de la corrección.
fichas = subprocess.run([sys.executable, "-B", str(AQUI / "generar_fichas.py")], cwd=AQUI, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
if fichas.returncode == 0:
    print("MODEL_CARD.md y DATA_STATEMENT.md, regenerados con generar_fichas.py")
else:
    print("AVISO: generar_fichas.py no terminó bien; córralo aparte (.venv/Scripts/python.exe -B generar_fichas.py):\n"
          + (fichas.stderr or fichas.stdout)[-600:])

pendientes = {k: v["motivo"] for k, v in ESTADO.items() if not v.get("listo")}
print(f"\nlisto en {time.perf_counter() - T0:.0f} s: {len(ESTADO) - len(pendientes)} bloques listos, {len(pendientes)} pendientes")
for k, v in pendientes.items():
    print(f"  pendiente: {k}: {v}")
