"""Comprueba que las cifras principales que muestra cada pantalla son las de los datos (y, por tanto, las del informe).

    python verificar_cifras.py

Sin navegador: arma cada pantalla, ejecuta sus callbacks con los valores con que nace (y, donde hay un control de tarea,
también con «regresión») y junta todo el texto que saldría en pantalla. Aparte, lee cada cifra directamente de su archivo
de datos/ con pandas, la escribe con el formato del informe y la busca en ese texto. Termina con código 1 si falta alguna.

No sustituye a mirar la pantalla: comprueba que la cifra está, no dónde ni cómo se lee.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.dont_write_bytecode = True
os.environ.pop("FUGA_SOLO", None)

import dash  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):          # en Windows, con la salida redirigida, cp1252 no tiene «−» ni «≤»
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app as A  # noqa: E402
import formato as F  # noqa: E402

DATOS = AQUI / "datos"
ESPACIOS = {chr(160): " ", chr(8239): " ", chr(8201): " ", chr(8288): "", chr(8203): ""}     # espacios duros y uniones invisibles


def llano(texto: str) -> str:
    for a, b in ESPACIOS.items():
        texto = texto.replace(a, b)
    return texto


# ============================================================================ el texto de cada pantalla
def cadenas(nodo, salida: list) -> None:
    """Todas las cadenas de un árbol de componentes o de la respuesta JSON de un callback."""
    if isinstance(nodo, str):
        salida.append(nodo)
    elif isinstance(nodo, dict):
        for v in nodo.values():
            cadenas(v, salida)
    elif isinstance(nodo, (list, tuple)):
        for v in nodo:
            cadenas(v, salida)
    elif hasattr(nodo, "to_plotly_json"):
        cadenas(nodo.to_plotly_json().get("props", {}), salida)


def ids(nodo, salida: dict) -> None:
    if isinstance(nodo, (list, tuple)):
        for n in nodo:
            ids(n, salida)
    elif hasattr(nodo, "to_plotly_json"):
        props = nodo.to_plotly_json().get("props", {})
        if isinstance(props.get("id"), str):
            salida[props["id"]] = props
        for v in props.values():
            if hasattr(v, "to_plotly_json") or isinstance(v, (list, tuple)):
                ids(v, salida)


def partes(salida: str):
    def una(p):
        cid, prop = p.rsplit(".", 1)
        return {"id": cid, "property": prop}
    return [una(p) for p in salida.strip(".").split("...")] if salida.startswith("..") else una(salida)


cliente = A.server.test_client()
deps = [d for d in cliente.get("/_dash-dependencies").get_json() if not d.get("clientside_function") and "_pages_" not in d["output"]]


# Dónde vive ahora lo que mostraba cada pantalla de la versión de nueve: Inicio y Límites en Contexto, Datos en EDA y el
# resto en su vista de Modelos (Modelo pasó a llamarse «Cómo decide»)
DONDE = {"inicio": "/", "limites": "/", "datos": "/eda", "modelo": "/modelos/decide"}


def texto_de(pantalla: str) -> str:
    ruta = DONDE.get(pantalla, pantalla if pantalla.startswith("/") else A.ruta_vista(pantalla))
    if ruta.startswith("/modelos/"):
        pagina = next(p for p in dash.page_registry.values() if p.get("path_template"))
        arbol = pagina["layout"](vista=ruta.rsplit("/", 1)[1])
    else:
        pagina = next(p for p in dash.page_registry.values() if p["path"] == ruta)
        arbol = pagina["layout"]() if callable(pagina["layout"]) else pagina["layout"]
    componentes: dict = {}
    ids(arbol, componentes)
    trozos: list = []
    cadenas(arbol, trozos)
    for dep in deps:
        entradas = dep.get("inputs", []) + dep.get("state", [])
        if not entradas or any(e["id"] not in componentes for e in entradas):
            continue
        variantes = [{}]
        if any(e["id"].endswith("-tarea") for e in dep.get("inputs", [])):
            variantes.append({e["id"]: "regresion" for e in dep["inputs"] if e["id"].endswith("-tarea")})
        for cambio in variantes:
            def valor(e):
                v = cambio.get(e["id"]) if e["property"] == "value" and e["id"] in cambio else componentes[e["id"]].get(e["property"])
                return {"id": e["id"], "property": e["property"], "value": v}
            r = cliente.post("/_dash-update-component", json={
                "output": dep["output"], "outputs": partes(dep["output"]), "inputs": [valor(e) for e in dep.get("inputs", [])],
                "changedPropIds": [], "state": [valor(e) for e in dep.get("state", [])]})
            if r.status_code == 200:
                cadenas(json.loads(r.get_data(as_text=True)), trozos)
            elif r.status_code != 204:
                raise SystemExit(f"{pantalla}: el callback {dep['output'][:60]} respondió {r.status_code}")
    return llano(" \n ".join(trozos))


# ============================================================================ las cifras, leídas de datos/
def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(DATOS / rel)


def fila(t: pd.DataFrame, **cond) -> pd.Series:
    m = np.ones(len(t), dtype=bool)
    for k, v in cond.items():
        m &= (t[k] == v).to_numpy()
    assert m.sum() == 1, (cond, int(m.sum()))
    return t[m].iloc[0]


meta = json.loads((DATOS / "metadatos.json").read_text(encoding="utf-8"))
etl = json.loads((DATOS / "eda" / "etl.json").read_text(encoding="utf-8"))
techo = json.loads((DATOS / "entrega3" / "comparacion_entrega2" / "techo.json").read_text(encoding="utf-8"))
clf, reg = csv("resultados/clasificacion_prueba.csv"), csv("resultados/regresion_prueba.csv")
pr_c, pr_r = csv("resultados/pruebas_clasificacion.csv"), csv("resultados/pruebas_regresion.csv")
e2, costo, abl = csv("resultados/entrega2.csv"), csv("resultados/costo.csv"), csv("resultados/ablacion.csv")
seg, semillas, corridas = csv("resultados/errores_segmentos.csv"), csv("resultados/semillas.csv"), csv("entrega3/corridas.csv")
cal, asoc = csv("entrega3/complementos/calibracion.csv"), csv("eda/asociacion.csv")
conc = csv("resultados/concordancia_validacion_prueba.csv")
fid = dict(zip(*[csv("resultados/fidelidad_resumen.csv").iloc[:, i] for i in (0, 1)]))
inter = dict(zip(*[csv("resultados/ablacion_interacciones_e2.csv").iloc[:, i] for i in (0, 1)]))
clientes = pd.read_parquet(DATOS / "clientes_prueba.parquet")
imp = pd.Series({t["variable"]: 0.0 for t in json.loads((DATOS / "modelos" / "ebm_clasificacion.json").read_text(encoding="utf-8"))["terminos"]})
for t in json.loads((DATOS / "modelos" / "ebm_clasificacion.json").read_text(encoding="utf-8"))["terminos"]:
    imp[t["variable"]] += t["importancia"]
imp = (imp / imp.sum()).sort_values(ascending=False)

ebm_c, xgb_c, log_c = (fila(clf, modelo=m) for m in ("ebm", "xgboost", "logistica"))
ebm_r, xgb_r = fila(reg, modelo="ebm"), fila(reg, modelo="xgboost")
e2_antes, e2_ebm = fila(e2, tarea="clasificacion", modelo="logistica_e2"), fila(e2, tarea="clasificacion", modelo="ebm")
e2r_antes, e2r_ebm = fila(e2, tarea="regresion", modelo="lasso_e2"), fila(e2, tarea="regresion", modelo="ebm")
costo_ebm = fila(costo, tarea="clasificacion", modelo="ebm")
caso = clientes.loc[meta["ejemplos"]["riesgo bajo con mayor puntaje"]]
cohortes = seg[seg["segmento"] == "cohorte de edad"]
productos = seg[seg["segmento"] == "productos activos"]
cal_ebm = cal[cal["modelo"] == "ebm"].set_index("version")
sin_balanceo = abl[(abl["tarea"] == "clasificacion") & abl["variante"].str.contains("sin balanceo")].iloc[0]
completa = abl[(abl["tarea"] == "clasificacion") & (abl["componente"] == "completo")].iloc[0]
random_ = abl[(abl["tarea"] == "clasificacion") & abl["variante"].str.contains("Random")].iloc[0]

ESPERADO = {
    "inicio": [
        ("AUC de prueba de la EBM", F.num(ebm_c["auc_prueba"], 4)), ("AUC de prueba de XGBoost", F.num(xgb_c["auc_prueba"], 4)),
        ("intervalo del AUC de la EBM", F.intervalo(ebm_c["auc_bca_inf"], ebm_c["auc_bca_sup"], 4)),
        ("diferencia con XGBoost", F.con_signo(fila(pr_c, modelo="xgboost")["delta_auc"], 4)),
        ("equivalencia práctica con XGBoost", F.prob(fila(pr_c, modelo="xgboost")["p_rope"])),
        ("AUC de la entrega anterior", F.num(e2_antes["metrica"], 4)), ("AUC de la EBM con la etiqueta de la mediana", F.num(e2_ebm["metrica"], 4)),
        ("parte del techo de información", F.pct((e2_ebm["metrica"] - 0.5) / (techo["auc_mediana"] - 0.5), 0)),
        ("importancia de las cuatro primeras variables", F.pct(imp.iloc[:4].sum(), 0)), ("ajuste de la EBM, en s", F.num(costo_ebm["ajuste_s"], 1)),
        ("corridas", F.entero(len(corridas))), ("horas de cómputo", F.num(corridas["tiempo_total_s"].sum() / 3600, 1)),
        ("RMSE de prueba de la EBM (regresión)", F.num(ebm_r["rmse_prueba"], 5)), ("RMSE de XGBoost (regresión)", F.num(xgb_r["rmse_prueba"], 5)),
    ],
    "datos": [
        ("clientes", F.entero(etl["n_clientes"])), ("transacciones limpias", F.entero(etl["n_transacciones_limpias"])),
        ("clientes de entrenamiento", F.entero(meta["n_entrenamiento"])), ("clientes de prueba", F.entero(meta["n_prueba"])),
        ("umbral de riesgo alto", F.num(meta["etiqueta"]["umbral"], 4)), ("riesgo alto en la prueba", F.pct(meta["prevalencia_prueba"], 1)),
        ("correlación de los productos activos", "r = " + F.num(fila(asoc, variable="active_products")["pearson"], 3)),
        ("R² del modelo que ve los productos activos", F.num(techo["r2_oraculo"], 4)), ("techo de R²", F.num(techo["r2"], 4)),
    ],
    "modelo": [(f"cuota de importancia de {v}", F.pct(imp[v], 1)) for v in imp.index[:4]] + [
        ("cuatro primeras variables", F.pct(imp.iloc[:4].sum(), 0)), ("parámetros", F.entero(meta["ebm"]["clasificacion"]["parametros"])),
        ("intercepto", F.num(meta["ebm"]["clasificacion"]["intercepto"], 4)), ("ajuste, en s", F.num(costo_ebm["ajuste_s"], 1)),
    ],
    "cliente": [
        ("puntaje del caso «riesgo bajo con mayor puntaje»", F.num(caso["p_ebm"], 4)), ("su puntaje real", F.num(caso["churn_probability"], 4)),
        ("su pronóstico de regresión", F.num(caso["pron_ebm"], 4)), ("corte de la campaña", F.num(meta["punto_operacion"]["puntaje_minimo_contactado_ebm"], 4)),
        ("clientes de prueba", F.entero(meta["n_prueba"])),
    ],
    "campana": [
        ("contactados con el 25 %", F.entero(meta["punto_operacion"]["contactados"])), ("aciertos de la EBM", F.entero(ebm_c["vp_op"])),
        ("falsas alarmas", F.entero(ebm_c["fp_op"])), ("se escapan", F.entero(ebm_c["fn_op"])), ("bien descartados", F.entero(ebm_c["vn_op"])),
        ("captura de la EBM", F.pct(ebm_c["recall_op"], 1)), ("precisión de la EBM", F.pct(ebm_c["precision_op"], 1)),
        ("aciertos de la regresión logística", F.entero(log_c["vp_op"])), ("captura de la EBM con el 50 %", F.pct(ebm_c["recall_50"], 1)),
        ("captura de la logística con el 50 %", F.pct(log_c["recall_50"], 1)), ("clientes de riesgo alto", F.entero(meta["positivos_prueba"])),
    ],
    "comparacion": [
        ("AUC externa de la EBM", F.num(ebm_c["auc_ext_media"], 4)), ("AUC de prueba de la EBM", F.num(ebm_c["auc_prueba"], 4)),
        ("equivalencia con XGBoost", F.prob(fila(pr_c, modelo="xgboost")["p_rope"])),
        ("equivalencia con Random Forest", F.prob(fila(pr_c, modelo="random_forest")["p_rope"])),
        ("equivalencia con el árbol", F.prob(fila(pr_c, modelo="arbol")["p_rope"])),
        ("AUC de la entrega anterior", F.num(e2_antes["metrica"], 4)), ("AUC de la EBM (mediana)", F.num(e2_ebm["metrica"], 4)),
        ("RMSE de la entrega anterior", F.num(e2r_antes["metrica"], 5)), ("RMSE de la EBM (mediana)", F.num(e2r_ebm["metrica"], 5)),
        ("RMSE de prueba de la EBM (regresión)", F.num(ebm_r["rmse_prueba"], 5)), ("p del MCS de la EBM", F.num(ebm_r["p_mcs"], 3)),
    ],
    "robustez": [
        ("costo de quitar las interacciones", F.num(float(inter["auc_ebm"]) - float(inter["auc_gam"]), 4)), ("p de DeLong", F.num(float(inter["p_delong"]), 2)),
        ("recall con umbral 0,5 sin balanceo", F.num(sin_balanceo["recall_05"], 3)), ("recall con ponderación", F.num(completa["recall_05"], 3)),
        ("ρ de la búsqueda de baja fidelidad", F.num(float(fid["rho_spearman"]), 2)), ("ahorro de tiempo", F.pct(float(fid["ahorro_tiempo"]), 0)),
        ("lo que pierde Random", F.num(random_["delta_completo"], 4)), ("desviación entre semillas", F.num(fila(semillas, tarea="clasificacion", modelo="ebm")["auc_desv"], 5)),
        ("ECE sin recalibrar", F.num(cal_ebm.loc["sin recalibrar", "ece"], 3)), ("ECE con la isotónica", F.num(cal_ebm.loc["isotónica", "ece"], 3)),
        ("ajuste de la EBM, en s", F.num(costo_ebm["ajuste_s"], 1)), ("concordancia validación–prueba", F.num(fila(conc, tarea="clasificacion")["correlacion"], 3)),
    ],
    "errores": [
        ("AUC en la cohorte de 59–60", F.num(cohortes.iloc[2]["auc_ebm"], 3)), ("sesgo con un producto", F.con_signo(productos.iloc[0]["sesgo_ebm"], 3)),
        ("sesgo con cinco productos", F.con_signo(productos.iloc[4]["sesgo_ebm"], 3)),
    ] + [(f"se escapan a la EBM, cohorte {c['nivel'].replace('--', '–')}", F.entero(c["fn_n_ebm"])) for _, c in cohortes.iterrows()]
      + [(f"se escapan a XGBoost, cohorte {c['nivel'].replace('--', '–')}", F.entero(c["fn_n_xgboost"])) for _, c in cohortes.iterrows()],
    "limites": [
        ("equivalencia con XGBoost", F.prob(fila(pr_c, modelo="xgboost")["p_rope"])), ("p del MCS de la EBM", F.num(ebm_r["p_mcs"], 3)),
        ("corridas", F.entero(len(corridas))), ("semilla", str(meta["semilla"])), ("clientes de entrenamiento", F.entero(meta["n_entrenamiento"])),
    ],
}

# ============================================================================ la comparación
fallas, total = [], 0
for pantalla, lista in ESPERADO.items():
    texto = texto_de(pantalla)
    print(f"\n== {pantalla} ({len(texto):,} caracteres de texto) ==")
    for que, cifra in lista:
        total += 1
        esta = llano(cifra) in texto
        print(f"  {'bien ' if esta else 'FALTA'} {cifra:<22} {que}")
        if not esta:
            fallas.append(f"{pantalla}: {que} ({cifra})")

print(f"\n{'PASA' if not fallas else 'FALLA'}: {total} cifras buscadas, {len(fallas)} sin encontrar")
for f in fallas:
    print("  falta:", f)
sys.exit(1 if fallas else 0)
