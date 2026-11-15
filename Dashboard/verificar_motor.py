"""Comprueba que el motor de la app (motor.py, numpy puro) da lo mismo que los modelos guardados y que el informe.

    python verificar_motor.py

Termina con código 1 si algo falla. La comparación con las tuberías de scikit-learn + interpret (datos/modelos/*.joblib)
solo se hace si esas bibliotecas están instaladas (pip install -r requirements-dev.txt); lo demás no las necesita.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.dont_write_bytecode = True

import datos_app as D  # noqa: E402
import motor as M  # noqa: E402

TOL = 1e-9
bien, mal, avisos = [], [], []


def comprobar(nombre: str, condicion: bool, detalle: str = "") -> None:
    (bien if condicion else mal).append(nombre)
    print(("  bien  " if condicion else "  FALLA ") + nombre + (f"  [{detalle}]" if detalle else ""))


def cerca(a, b, tol=TOL) -> bool:
    return bool(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) <= tol)


C = M.clientes()
X = C[M.predictores()]
y = C["riesgo_alto"].to_numpy()
meta = M.meta()
print(f"Clientes de prueba: {len(C):,} · predictores: {len(M.predictores())} · umbral de riesgo alto: {meta['etiqueta']['umbral']:.6f}")

print("\n1. El motor reproduce los puntajes guardados por el proyecto")
P = M.puntuar(X)
for col, guardada in (("p_ebm", "p_ebm"), ("p_ebm_cal", "p_ebm_cal"), ("pron_ebm", "pron_ebm"), ("p_logistica", "p_logistica")):
    dif = float(np.abs(P[col] - C[guardada]).max())
    comprobar(f"{col} en vivo = {guardada} guardado", dif <= TOL, f"dif. máx. {dif:.1e}")
comprobar("el puntaje guardado de la EBM es el de puntajes_prueba.parquet", cerca(C["p_ebm"], D.puntajes()["ebm"], 1e-6))
comprobar("la etiqueta es la de puntajes_prueba.parquet", bool((D.puntajes()["y"].to_numpy() == y).all()))
cal = D.compartida("entrega3/complementos/calibracion_prueba.parquet")
col_iso = [c for c in cal.columns if "__ebm__" in c and c.endswith("isotónica")]
comprobar("la probabilidad recalibrada es la isotónica del proyecto", len(col_iso) == 1 and cerca(C["p_ebm_cal"], cal[col_iso[0]]))

print("\n2. La explicación es exacta: intercepto + contribuciones = puntaje")
for tarea, col in (("clasificacion", "p_ebm"), ("regresion", "pron_ebm")):
    b0, contrib = M.explicar(X, tarea)
    suma = b0 + contrib.sum(axis=1).to_numpy()
    objetivo = np.log(C[col] / (1 - C[col])).to_numpy() if tarea == "clasificacion" else C[col].to_numpy()
    comprobar(f"{tarea}: la suma de las 48 contribuciones más el intercepto da el {'logit del puntaje' if tarea == 'clasificacion' else 'pronóstico'}",
              cerca(suma, objetivo, 1e-8), f"dif. máx. {float(np.abs(suma - objetivo).max()):.1e}")
    m = M.ebm(tarea)
    comprobar(f"{tarea}: {len(m.terminos)} términos y {m.n_parametros:,} parámetros, como dice metadatos.json",
              len(m.terminos) == meta["ebm"][tarea]["terminos"] and m.n_parametros == meta["ebm"][tarea]["parametros"])
imp = M.importancia()
comprobar("la importancia por variable suma 1", abs(float(imp.sum()) - 1) < 1e-12)

print("\n3. Las cifras del informe salen de estos puntajes")
clf, reg = D.prueba("clasificacion"), D.prueba("regresion")
f_ebm = D.fila(clf, modelo="ebm")
comprobar("AUC de prueba de la EBM", abs(M.auc(y, C["p_ebm"]) - f_ebm["auc_prueba"]) < 1e-9, f"{M.auc(y, C['p_ebm']):.6f}")
rmse = float(np.sqrt(np.mean((C["churn_probability"] - C["pron_ebm"]) ** 2)))
comprobar("RMSE de prueba de la EBM", abs(rmse - D.fila(reg, modelo="ebm")["rmse_prueba"]) < 1e-9, f"{rmse:.6f}")
S = D.puntajes()
for modelo in clf["modelo"]:
    f = D.fila(clf, modelo=modelo)
    r = M.resumen_capacidad(y, S[modelo], M.CAPACIDAD_INFORME)
    iguales = (r["vp"], r["fp"], r["fn"], r["vn"]) == tuple(int(f[k]) for k in ("vp_op", "fp_op", "fn_op", "vn_op"))
    comprobar(f"matriz de confusión al 25 % de capacidad: {modelo}", iguales, f"VP {r['vp']}, FP {r['fp']}, FN {r['fn']}, VN {r['vn']}")
    r10, r50 = M.resumen_capacidad(y, S[modelo], 0.10), M.resumen_capacidad(y, S[modelo], 0.50)
    comprobar(f"captura al 10 % y al 50 % de capacidad: {modelo}",
              abs(r10["captura"] - f["recall_10"]) < 1e-12 and abs(r50["captura"] - f["recall_50"]) < 1e-12,
              f"{r10['captura']:.4f} y {r50['captura']:.4f}")
    comprobar(f"AUC de prueba: {modelo}", abs(M.auc(y, S[modelo]) - f["auc_prueba"]) < 1e-9)
curva = M.curva_captura(y, C["p_ebm"])
comprobar("la curva de captura pasa por el punto de operación del informe",
          abs(float(curva.loc[curva["capacidad"].sub(0.25).abs().idxmin(), "captura"]) - f_ebm["recall_op"]) < 1e-12)
comprobar("el corte del 25 % es el de metadatos.json",
          abs(M.resumen_capacidad(y, C["p_ebm"])["corte"] - meta["punto_operacion"]["puntaje_minimo_contactado_ebm"]) < 1e-12)

print("\n4. El simulador respeta las identidades de los datos")
atadas = ["product_satisfaction", "tx_satisfaction", "satisfaction_score", "credit_utilization_ratio"]
registros = X.to_dict("records")
distintas = 0
for f in registros:
    g = M.coherente(f)
    distintas += any(abs(float(g[k]) - float(f[k])) > 1e-9 for k in atadas)
comprobar("recalcular las variables atadas de un cliente real no lo cambia (los 9.745)", distintas == 0, f"{distintas} distintos")
fila = C.loc[meta["ejemplos"]["riesgo bajo con mayor puntaje"]]
igual = M.puntuar(M.simular(fila, {}))
comprobar("simular sin cambios devuelve el puntaje del cliente", abs(float(igual["p_ebm"].iloc[0]) - float(fila["p_ebm"])) < 1e-12)
comprobar("el puesto y el percentil son coherentes", M.puesto(float(C["p_ebm"].max())) == 1 and abs(M.percentil(float(C["p_ebm"].max())) - 1) < 1e-12)

print("\n5. El motor frente a las tuberías de scikit-learn e interpret")
try:
    import joblib
    import interpret  # noqa: F401
    import sklearn  # noqa: F401
    hay = True
except ImportError as e:
    hay = False
    avisos.append(f"no se comparó con datos/modelos/*.joblib: falta {e.name} (pip install -r requirements-dev.txt)")
    print("  aviso  " + avisos[-1])
if hay:
    for tarea, col in (("clasificacion", "p_ebm"), ("regresion", "pron_ebm")):
        pipe = joblib.load(M.DATOS / "modelos" / f"ebm_{tarea}.joblib")
        s = pipe.predict_proba(X)[:, 1] if tarea == "clasificacion" else pipe.predict(X)
        comprobar(f"{tarea}: motor en numpy = modelo de interpret en los {len(X):,} clientes", cerca(P[col], s, 1e-12),
                  f"dif. máx. {float(np.abs(P[col] - s).max()):.1e}")
        ebm = pipe.named_steps["modelo"]
        terminos = ebm.eval_terms(pipe.named_steps["preprocesado"].transform(X))
        comprobar(f"{tarea}: contribuciones por término = eval_terms de interpret", cerca(M.ebm(tarea).contribuciones(X), terminos, 1e-12))

print(f"\n{'PASA' if not mal else 'FALLA'}: {len(bien)} comprobaciones bien, {len(mal)} fallas, {len(avisos)} avisos")
for m in mal:
    print("  falla:", m)
sys.exit(1 if mal else 0)
