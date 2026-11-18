"""Capturas del dashboard para el capítulo 8 del libro (``ProyectoFinal/figuras/dash_*.png``).

Toma tres pantallas del dashboard en marcha, con Microsoft Edge sin interfaz (necesita Playwright:
``pip install playwright``). El dashboard se lanza aparte, en local o desplegado::

    cd Dashboard && PORT=8061 python app.py                      # en otra terminal
    python ProyectoFinal/capturar_dash.py --url http://127.0.0.1:8061
    python ProyectoFinal/capturar_dash.py --url https://fuga-de-clientes-dash.onrender.com

Cada captura es la parte alta de una vista, al ancho de un portátil, y se guarda con el nombre que usa el capítulo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FIGURAS = Path(__file__).resolve().parent / 'figuras'
ANCHO = 1280
PANTALLAS = [
    # (archivo, ruta, alto de la captura en px, texto de un encabezado desde el que empezar o None para el principio)
    ('dash_original.png', '/modelos/original?tarea=regresion', 1500, None),
    ('dash_prediccion.png', '/modelos/prediccion?tarea=regresion&modelo=bdl', 1750, None),
    ('dash_lideres.png', '/modelos/lideres?tarea=regresion', 1500, None),
]


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--url', required=True, help='dirección del dashboard en marcha, sin barra final')
    p.add_argument('--espera', type=int, default=2500, help='ms de espera tras cargar, para que terminen los callbacks')
    a = p.parse_args()
    from playwright.sync_api import sync_playwright

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fallas = 0
    with sync_playwright() as pw:
        navegador = pw.chromium.launch(channel='msedge', headless=True)
        for archivo, ruta, alto, desde in PANTALLAS:
            pagina = navegador.new_page(viewport={'width': ANCHO, 'height': alto}, device_scale_factor=1, locale='es-CO')
            errores = []
            pagina.on('console', lambda m, e=errores: e.append(m.text[:160]) if m.type == 'error' else None)
            pagina.goto(a.url.rstrip('/') + ruta, wait_until='domcontentloaded', timeout=180000)
            try:
                pagina.wait_for_load_state('networkidle', timeout=120000)
            except Exception:                                            # noqa: BLE001 - una vista con sondeo no llega a estar ociosa
                pass
            pagina.wait_for_timeout(a.espera)
            if desde:
                pagina.locator('h1, h2, h3').filter(has_text=desde).first.evaluate('e => e.scrollIntoView({block: "start"})')
                pagina.wait_for_timeout(600)
            titulo = pagina.locator('h1').first.inner_text() if pagina.locator('h1').count() else ''
            pagina.screenshot(path=str(FIGURAS / archivo))
            print(f'{archivo}: {ruta} · «{titulo[:70]}» · {len(errores)} errores de consola')
            fallas += bool(errores) or not titulo
            pagina.close()
        navegador.close()
    return 1 if fallas else 0


if __name__ == '__main__':
    sys.exit(main())
