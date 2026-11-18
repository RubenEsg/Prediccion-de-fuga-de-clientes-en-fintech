"""Exporta el informe del proyecto final a un solo PDF (``ProyectoFinal/ProyectoFinal_informe.pdf``).

Imprime cada capítulo del libro ya construido con Microsoft Edge (o Chrome) en modo *headless*, con un estilo de
impresión propio (sin barras laterales, tablas ajustadas al ancho de la hoja), y une los PDF con ``pypdf`` añadiendo un
marcador por capítulo. No necesita LaTeX.

Uso, desde la raíz del repositorio y con el libro ya construido (``jupyter-book build .``)::

    python ProyectoFinal/exportar_pdf.py                  # usa _build/html
    python ProyectoFinal/exportar_pdf.py --html docs      # o el libro publicado
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml
from pypdf import PdfReader, PdfWriter

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / 'ProyectoFinal' / 'ProyectoFinal_informe.pdf'
NAVEGADORES = [
    Path(os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')) / 'Microsoft' / 'Edge' / 'Application' / 'msedge.exe',
    Path(os.environ.get('PROGRAMFILES', r'C:\Program Files')) / 'Microsoft' / 'Edge' / 'Application' / 'msedge.exe',
    Path(os.environ.get('PROGRAMFILES', r'C:\Program Files')) / 'Google' / 'Chrome' / 'Application' / 'chrome.exe',
]
# Las reglas se aplican también en pantalla, para que el ajuste de las tablas (que mide la página antes de imprimir)
# vea la disposición del papel: A4 con márgenes de 13 mm, sin barras laterales.
ESTILO_IMPRESION = """
<style>
@page { size: A4; margin: 14mm 13mm; }
html, body { width: 184mm; max-width: 184mm; }
.bd-sidebar-primary, .bd-sidebar-secondary, .bd-header, .bd-header-article, .bd-footer, .bd-footer-content,
.prev-next-area, .skip-link, .headerlink { display: none !important; }
.bd-main .bd-content { margin-left: 0; }
.bd-main .bd-content .container, .bd-article-container { max-width: none !important; min-width: 0; }
.pst-scrollable-table-container { overflow: visible; }
.pst-scrollable-table-container table { font-size: 0.78em; }
img { max-width: 100%; height: auto; }
div.highlight, div.literal-block-wrapper, div[class*=highlight-] { break-inside: auto; overflow: visible; }
div.highlight pre { white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; overflow: visible; font-size: 0.72em; }
.bd-main .bd-content table { break-inside: auto; }
.bd-main .bd-content tr { break-inside: avoid; }
.admonition { break-inside: auto; }
h1, h2, h3 { break-after: avoid; }
figure { break-inside: avoid; }
</style>
<script>
// Las tablas con muchas columnas que no caben ni envolviendo el texto reducen su letra lo justo para entrar.
window.addEventListener('load', () => {
  const ajustar = () => {
    for (const t of document.querySelectorAll('.bd-article table')) {
      const disponible = t.closest('.bd-article').getBoundingClientRect().width;
      const razon = t.getBoundingClientRect().width / disponible;
      if (razon > 1.005) {
        const actual = parseFloat(getComputedStyle(t).fontSize);
        t.style.fontSize = Math.max(5.5, actual / razon * 0.985).toFixed(2) + 'px';
      }
    }
  };
  setTimeout(() => { ajustar(); setTimeout(ajustar, 300); setTimeout(ajustar, 600); }, 400);
});
</script>
"""


def navegador() -> Path:
    for ruta in NAVEGADORES:
        if ruta.exists():
            return ruta
    raise SystemExit('No se encontró Microsoft Edge ni Google Chrome para imprimir las páginas.')


def capitulos(html: Path) -> list[Path]:
    """Los capítulos del proyecto final, en el orden de ``_toc.yml``."""
    toc = yaml.safe_load((RAIZ / '_toc.yml').read_text(encoding='utf-8'))
    nombres = [c['file'] for parte in toc.get('parts', []) for c in parte.get('chapters', []) if c['file'].startswith('ProyectoFinal/')]
    return [html / (n + '.html') for n in nombres]


def titulo_de(html: Path) -> str:
    m = re.search(r'<title>(.*?)(?: &#8212;.*)?</title>', html.read_text(encoding='utf-8'), re.S)
    return re.sub(r'\s+', ' ', m.group(1)).strip() if m else html.stem


def imprimir(exe: Path, html: Path, pdf: Path, perfil: Path) -> None:
    """Imprime una copia de la página con el estilo de impresión inyectado, junto a la original para que sus rutas
    relativas sigan funcionando. ``perfil`` es un directorio de datos propio: sin él, si el navegador ya está abierto,
    la orden se delega a esa instancia y el proceso vuelve antes de que el PDF exista."""
    copia = html.with_name(f'__imprimir__{html.name}')
    copia.write_text(html.read_text(encoding='utf-8').replace('</head>', ESTILO_IMPRESION + '</head>', 1), encoding='utf-8')
    try:
        subprocess.run([str(exe), '--headless=new', '--disable-gpu', '--no-first-run', '--no-pdf-header-footer',
                        f'--user-data-dir={perfil}', '--virtual-time-budget=25000', f'--print-to-pdf={pdf}',
                        copia.resolve().as_uri()],
                       check=True, timeout=240, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):                      # por si el navegador vuelve antes de terminar de escribir
            if pdf.exists() and pdf.stat().st_size > 0:
                break
            time.sleep(0.5)
    finally:
        copia.unlink(missing_ok=True)
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise SystemExit(f'No se generó {pdf}')


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--html', default=str(RAIZ / '_build' / 'html'), help='carpeta del libro construido')
    a.add_argument('--salida', default=str(SALIDA))
    args = a.parse_args()
    html = Path(args.html)
    html = html if html.is_absolute() else RAIZ / html
    paginas = capitulos(html)
    if not paginas or not paginas[0].exists():
        raise SystemExit(f'Falta el libro construido en {html}: ejecuta antes  jupyter-book build .')
    exe = navegador()
    config = yaml.safe_load((RAIZ / '_config.yml').read_text(encoding='utf-8'))
    escritor = PdfWriter()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        perfil = Path(tmp) / 'perfil'
        for pagina in paginas:
            pdf = Path(tmp) / (pagina.stem + '.pdf')
            imprimir(exe, pagina, pdf, perfil)
            lector = PdfReader(pdf)
            inicio = len(escritor.pages)
            for hoja in lector.pages:
                escritor.add_page(hoja)
            escritor.add_outline_item(titulo_de(pagina), inicio)
            print(f'{pagina.name}: {len(lector.pages)} páginas')
    escritor.add_metadata({'/Title': 'Proyecto final: dashboard predictivo, modelo original y evaluación rigurosa',
                           '/Author': config['author'], '/Subject': 'Proyecto final · Machine Learning · Universidad del Norte'})
    salida = Path(args.salida)
    escritor.write(salida)
    print(f'{salida.name}: {len(escritor.pages)} páginas, {salida.stat().st_size / 2 ** 20:.1f} MB')


if __name__ == '__main__':
    main()
