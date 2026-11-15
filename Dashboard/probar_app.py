"""Prueba de humo sin navegador: arma cada pantalla y ejecuta todos sus callbacks con los valores con que nace.

    python probar_app.py                  las tres secciones y las doce vistas de Modelos
    python probar_app.py cliente eda      solo esas (contexto, eda o el id de una vista de Modelos)

Para cada pantalla comprueba que su layout se arma, que no repite ningún id, que cada callback del servidor encuentra en la
pantalla todas sus entradas, estados y salidas (un id mal escrito no da error en Dash con páginas) y que responde 200 con
los valores iniciales. Informa el tiempo y el tamaño de cada respuesta. Termina con código 1 si algo falla.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.dont_write_bytecode = True

SOLO = [a for a in sys.argv[1:] if not a.startswith("-")]
if SOLO:                                # solo esos módulos se importan (FUGA_SOLO en app.py)
    os.environ["FUGA_SOLO"] = ",".join(SOLO)

import dash  # noqa: E402

import app as A  # noqa: E402

LENTO_MS = 1500          # un callback que tarda más se señala (no falla)
PESADO_KB = 1500         # una respuesta más grande se señala (no falla)
fallas, avisos = [], []


def recorrer(nodo, ids, repetidos):
    """Recoge {id: props} de un árbol de componentes de Dash (los ids de diccionario se guardan como JSON ordenado)."""
    if isinstance(nodo, (list, tuple)):
        for n in nodo:
            recorrer(n, ids, repetidos)
        return
    if not hasattr(nodo, "to_plotly_json"):
        return
    props = nodo.to_plotly_json().get("props", {})
    cid = props.get("id")
    if cid is not None:
        clave = cid if isinstance(cid, str) else json.dumps(cid, sort_keys=True, separators=(",", ":"))
        if clave in ids:
            repetidos.append(clave)
        ids[clave] = props
    for nombre, valor in props.items():
        if nombre != "id" and (hasattr(valor, "to_plotly_json") or isinstance(valor, (list, tuple))):
            recorrer(valor, ids, repetidos)


def partes(salida: str):
    """«..a.children...b.figure..» -> [{id, property}, …]; «a.children» -> {id, property}."""
    def una(p):
        cid, prop = p.rsplit(".", 1)
        return {"id": cid, "property": prop}
    if salida.startswith(".."):
        return [una(p) for p in salida.strip(".").split("...")]
    return una(salida)


def ids_de(dep):
    salidas = partes(dep["output"])
    salidas = salidas if isinstance(salidas, list) else [salidas]
    return ([i["id"] for i in dep.get("inputs", [])], [s["id"] for s in dep.get("state", [])],
            [s["id"] for s in salidas])


cliente = A.server.test_client()
deps = cliente.get("/_dash-dependencies").get_json()
del_servidor = [d for d in deps if not d.get("clientside_function") and "_pages_" not in d["output"]]
# Cada sección es una entrada; Modelos (path_template /modelos/<vista>) da una entrada por vista
paginas = []
for pagina in sorted(dash.page_registry.values(), key=lambda p: p["path"]):
    if pagina.get("path_template"):
        for vista, *_resto in A.VISTAS:
            paginas.append({"nombre": vista, "path": A.ruta_vista(vista),
                            "layout": (lambda v=vista, f=pagina["layout"]: f(vista=v))})
    else:
        paginas.append({"nombre": pagina["module"].split(".")[-1], "path": pagina["path"], "layout": pagina["layout"]})
if SOLO:
    paginas = [p for p in paginas if p["nombre"] in SOLO]
    if not paginas:
        sys.exit(f"Ninguna sección o vista se llama {SOLO}")
print(f"{len(paginas)} secciones y vistas, {len(del_servidor)} callbacks del servidor ({len(deps) - len(del_servidor)} del cliente o del enrutado)")

usados = set()
for pagina in paginas:
    nombre = pagina["nombre"]
    t0 = time.perf_counter()
    try:
        arbol = pagina["layout"]() if callable(pagina["layout"]) else pagina["layout"]
    except Exception as e:  # noqa: BLE001
        fallas.append(f"{nombre}: el layout no se arma: {type(e).__name__}: {e}")
        print(f"\n== {nombre} ({pagina['path']}): FALLA al armar el layout: {type(e).__name__}: {e}")
        continue
    ids, repetidos = {}, []
    recorrer(arbol, ids, repetidos)
    recorrer(A.app.layout, ids, [])            # url, contenido y lo demás del cascarón
    print(f"\n== {nombre} ({pagina['path']}): layout en {(time.perf_counter() - t0) * 1000:.0f} ms, {len(ids)} componentes con id")
    if repetidos:
        fallas.append(f"{nombre}: ids repetidos: {sorted(set(repetidos))}")
        print("   FALLA ids repetidos:", sorted(set(repetidos)))
    propios = [k for k in ids if not k.startswith(("url", "nav-", "contenido", "siguiente", "_pages"))]
    n = 0
    for dep in del_servidor:
        entradas, estados, salidas = ids_de(dep)
        if any(i.startswith("{") for i in entradas + estados + salidas):
            if any(i in ids for i in entradas + estados + salidas):
                avisos.append(f"{nombre}: callback con ids de patrón no probado: {dep['output'][:80]}")
            continue
        toca = [i for i in entradas + estados + salidas if i in propios]
        if not toca:
            continue
        usados.add(dep["output"])
        faltan = sorted({i for i in entradas + estados + salidas if i not in ids})
        if faltan:
            fallas.append(f"{nombre}: el callback {dep['output'][:70]} nombra ids que la pantalla no tiene: {faltan}")
            print(f"   FALLA {dep['output'][:70]}: ids que no están en la pantalla: {faltan}")
            continue
        def valor(e):
            return {"id": e["id"], "property": e["property"], "value": ids[e["id"]].get(e["property"])}
        cuerpo = {"output": dep["output"], "outputs": partes(dep["output"]), "inputs": [valor(e) for e in dep.get("inputs", [])],
                  "changedPropIds": [], "state": [valor(e) for e in dep.get("state", [])]}
        t1 = time.perf_counter()
        r = cliente.post("/_dash-update-component", json=cuerpo)
        ms = (time.perf_counter() - t1) * 1000
        kb = len(r.get_data()) / 1024
        n += 1
        marca = "bien "
        if r.status_code not in (200, 204):
            marca = "FALLA"
            texto = r.get_data(as_text=True)
            fallas.append(f"{nombre}: {dep['output'][:70]} respondió {r.status_code}: {texto[-300:].strip()}")
        elif ms > LENTO_MS or kb > PESADO_KB:
            marca = "aviso"
            avisos.append(f"{nombre}: {dep['output'][:70]} tardó {ms:.0f} ms y pesa {kb:.0f} KB")
        print(f"   {marca} {r.status_code} {ms:6.0f} ms {kb:7.1f} KB  {dep['output'][:90]}")
    print(f"   {n} callbacks ejecutados")

sin_pantalla = [d["output"] for d in del_servidor if d["output"] not in usados and not d["output"].startswith("..{") and "{" not in d["output"]
                and not d["output"].startswith("siguiente.")]                      # el pie común lo pone app.py
if sin_pantalla and not SOLO:
    for s in sin_pantalla:
        avisos.append(f"callback que no toca ningún componente de ninguna pantalla: {s[:90]}")

print(f"\n{'PASA' if not fallas else 'FALLA'}: {len(fallas)} fallas, {len(avisos)} avisos")
for f in fallas:
    print("  falla:", f)
for a in avisos:
    print("  aviso:", a)
sys.exit(1 if fallas else 0)
