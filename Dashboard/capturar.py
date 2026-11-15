"""Capturas de las pantallas de la app en marcha, con Edge sin interfaz (necesita Playwright: pip install playwright).

    python capturar.py --puerto 8061                          todas las pantallas a 1366 y 400 px
    python capturar.py --puerto 8061 --rutas /cliente /datos --anchos 1600 --salida capturas

Además de guardar un PNG de página completa por pantalla y ancho, informa los errores de consola, las peticiones fallidas,
si falta el h1 y si algo se sale del ancho de la ventana. Termina con código 1 si hay errores de consola o desborde.
La app se lanza aparte (PORT=8061 python app.py); nunca en los puertos 8000, 8020 ni 8050.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROHIBIDOS = {8000, 8020, 8050}


def rutas_de_nav() -> list[str]:
    """Las rutas de Contexto, EDA y cada vista de Modelos, leídas de SECCIONES y VISTAS en app.py (sin importar Dash)."""
    arbol = ast.parse((AQUI / "app.py").read_text(encoding="utf-8"))
    listas = {}
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 and isinstance(nodo.targets[0], ast.Name) \
                and nodo.targets[0].id in ("SECCIONES", "VISTAS"):
            listas[nodo.targets[0].id] = ast.literal_eval(nodo.value)
    rutas = [fila[0] for fila in listas.get("SECCIONES", [("/",)]) if fila[0] != "/modelos"]
    return rutas + [f"/modelos/{fila[0]}" for fila in listas.get("VISTAS", [])]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):                 # con la salida redirigida, Windows usa cp1252 y no tiene «→»
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--puerto", type=int, required=True)
    p.add_argument("--rutas", nargs="*", default=None)
    p.add_argument("--anchos", nargs="*", type=int, default=[1366, 400])
    p.add_argument("--alto", type=int, default=900)
    p.add_argument("--salida", default=str(AQUI / "capturas"))
    p.add_argument("--espera", type=int, default=1200, help="ms de espera tras cargar, para que terminen los callbacks")
    p.add_argument("--prefijo", default="")
    a = p.parse_args()
    if a.puerto in PROHIBIDOS:
        print(f"El puerto {a.puerto} es de otra app de este equipo: use otro.")
        return 2
    from playwright.sync_api import sync_playwright

    rutas = a.rutas or rutas_de_nav()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    problemas = 0
    with sync_playwright() as pw:
        navegador = pw.chromium.launch(channel="msedge", headless=True)
        for ancho in a.anchos:
            contexto = navegador.new_context(viewport={"width": ancho, "height": a.alto}, device_scale_factor=1, locale="es-CO")
            for ruta in rutas:
                pagina = contexto.new_page()
                consola, fallidas = [], []
                pagina.on("console", lambda m: consola.append(m.text) if m.type == "error" else None)
                pagina.on("pageerror", lambda e: consola.append(f"error de página: {e}"))
                pagina.on("response", lambda r: fallidas.append(f"{r.status} {r.url}") if r.status >= 400 else None)
                pagina.goto(f"http://127.0.0.1:{a.puerto}{ruta}", wait_until="networkidle", timeout=60000)
                pagina.wait_for_timeout(a.espera)
                datos = pagina.evaluate("""() => {
                    const de = document.documentElement;
                    const fuera = [];
                    for (const el of document.querySelectorAll('#contenido *')) {
                        const r = el.getBoundingClientRect();
                        if (r.width > 0 && r.right > de.clientWidth + 1 && getComputedStyle(el).position !== 'fixed') {
                            let oculto = false;
                            for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
                                const o = getComputedStyle(p).overflowX;
                                if (o === 'auto' || o === 'scroll' || o === 'hidden' || o === 'clip') { oculto = true; break; }
                            }
                            if (!oculto) fuera.push((el.className && el.className.baseVal === undefined ? el.className : el.tagName) + ' → ' + Math.round(r.right));
                        }
                    }
                    return {h1: (document.querySelector('#contenido h1') || {}).innerText || null, ancho_documento: de.scrollWidth,
                            ancho_ventana: de.clientWidth, alto: de.scrollHeight, fuera: fuera.slice(0, 8),
                            graficos: document.querySelectorAll('.js-plotly-plot').length,
                            vacios: [...document.querySelectorAll('.js-plotly-plot')].filter(g => !g.querySelector('.plot, .cartesianlayer, .pielayer')).length};
                }""")
                nombre = (a.prefijo + (ruta.strip("/").replace("/", "_").replace("?", "_").replace("=", "-").replace("&", "_") or "inicio"))
                archivo = salida / f"{nombre}_{ancho}.png"
                pagina.screenshot(path=str(archivo), full_page=True)
                desborde = datos["ancho_documento"] > datos["ancho_ventana"] + 1
                estado = "bien "
                if consola or desborde or not datos["h1"]:
                    estado = "FALLA"
                    problemas += 1
                print(f"{estado} {ruta:<14} {ancho:>5} px · alto {datos['alto']:>5} · gráficos {datos['graficos']}"
                      f"{' (sin dibujar: ' + str(datos['vacios']) + ')' if datos['vacios'] else ''} · h1: {datos['h1']!r} → {archivo.name}")
                if desborde:
                    print(f"      se sale del ancho: documento {datos['ancho_documento']} px, ventana {datos['ancho_ventana']} px; {datos['fuera']}")
                for linea in consola[:6]:
                    print("      consola:", linea[:300])
                for linea in fallidas[:6]:
                    print("      petición fallida:", linea[:200])
                pagina.close()
            contexto.close()
        navegador.close()
    print(f"\n{'PASA' if not problemas else 'FALLA'}: {problemas} pantallas con problemas; capturas en {salida}")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
