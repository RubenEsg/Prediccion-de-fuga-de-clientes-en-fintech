"""Exporta el libro construido a un solo PDF (``Entregable3_libro.pdf`` en la raíz del proyecto).

Imprime cada página HTML de ``_build/html/libro/`` con Microsoft Edge (o Chrome) en modo *headless*,
que respeta el CSS de impresión del tema (sin barras laterales, con el índice de cada página), y une
los PDF con ``pypdf`` añadiendo un marcador por página. No necesita LaTeX ni descargar un navegador.

Uso, desde la carpeta del proyecto y con el libro ya construido (``jupyter-book build .``)::

    .venv\\Scripts\\python.exe libro/exportar_pdf.py
"""
from __future__ import annotations

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
HTML = RAIZ / '_build' / 'html' / 'libro'
SALIDA = RAIZ / 'Entregable3_libro.pdf'
NAVEGADORES = [
    Path(os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')) / 'Microsoft' / 'Edge' / 'Application' / 'msedge.exe',
    Path(os.environ.get('PROGRAMFILES', r'C:\Program Files')) / 'Microsoft' / 'Edge' / 'Application' / 'msedge.exe',
    Path(os.environ.get('PROGRAMFILES', r'C:\Program Files')) / 'Google' / 'Chrome' / 'Application' / 'chrome.exe',
]
# Formato de página y ajustes de impresión. Las reglas se aplican también en pantalla (la copia que se
# imprime no se ve de otro modo), para que el ajuste de las tablas, que mide la página en pantalla antes
# de imprimir, vea exactamente la disposición del papel: A4 con márgenes de 13 mm, sin barras laterales.
ESTILO_IMPRESION = """
<style>
@page { size: A4; margin: 14mm 13mm; }
html, body { width: 184mm; max-width: 184mm; }
.bd-sidebar-primary, .bd-sidebar-secondary, .bd-header, .bd-header-article, .bd-footer, .bd-footer-content, .skip-link { display: none; }
.bd-main .bd-content { margin-left: 0; }
.bd-main .bd-content .container, .bd-article-container { max-width: none; min-width: 0; }
div.cell_output table { font-size: 0.68em; }
div.cell_output pre { font-size: 0.8em; }
div.cell_output .output { overflow: visible; }
img { max-width: 100%; height: auto; }
/* El tema prohíbe partir div.highlight entre páginas pero no su contenedor: la caja gris se dibujaba en
   una página y el código en la siguiente. Se permite partir el código y se envuelven las líneas largas en
   lugar de recortarlas con una barra de desplazamiento. overflow-wrap: anywhere (y no break-word) reduce
   además el ancho mínimo del bloque: si no, un token largo sin espacios (una ruta) ensancha el documento
   y el navegador reduce la escala de toda la sección para que quepa. */
div.cell_input, div.highlight, div.literal-block-wrapper, div[class*=highlight-] { break-inside: auto; overflow: visible; }
div.highlight pre, div.cell_output pre { white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; overflow: visible; }
div.cell_output pre, div.cell_output .output { max-width: 100%; min-width: 0; }
/* Las celdas envuelven el texto en los espacios; los encabezados (nombres de columna con guiones bajos)
   se parten donde haga falta, pero las cifras nunca. */
div.cell_output td, div.cell_output th { white-space: normal; word-break: normal; overflow-wrap: normal;
                                         padding: 0.12rem 0.28rem; line-height: 1.25; }
div.cell_output th { overflow-wrap: anywhere; }
/* Las tablas largas se parten por filas en vez de dejar media página en blanco. */
.bd-main .bd-content table { break-inside: auto; }
.bd-main .bd-content tr { break-inside: avoid; }
</style>
<script>
// Las tablas de pandas con muchas columnas no caben en el ancho de la hoja ni envolviendo el texto, y el
// navegador reduciría la escala de toda la sección para que quepan. En su lugar se reduce la letra solo
// de esas tablas, lo justo para que entren (con un mínimo de 5,5 px).
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


def paginas_en_orden() -> list[Path]:
    """Las páginas del libro en el orden de ``_toc.yml``."""
    toc = yaml.safe_load((RAIZ / '_toc.yml').read_text(encoding='utf-8'))
    nombres = [toc['root']]
    for parte in toc.get('parts', []):
        nombres += [c['file'] for c in parte.get('chapters', [])]
    return [HTML / (Path(n).name + '.html') for n in nombres]


def titulo_de(html: Path) -> str:
    m = re.search(r'<title>(.*?)(?: &#8212;.*)?</title>', html.read_text(encoding='utf-8'), re.S)
    return re.sub(r'\s+', ' ', m.group(1)).strip() if m else html.stem


def imprimir(exe: Path, html: Path, pdf: Path, perfil: Path) -> None:
    """Imprime una copia de la página con el estilo de impresión inyectado (junto a la original, para
    que sus rutas relativas a ``../_static`` y ``../_images`` sigan funcionando).

    ``perfil`` es un directorio de datos propio: sin él, si Edge ya está abierto, la orden se delega a
    esa instancia y el proceso vuelve antes de que el PDF exista.
    """
    copia = html.with_name(f'__imprimir__{html.name}')
    texto = html.read_text(encoding='utf-8').replace('</head>', ESTILO_IMPRESION + '</head>', 1)
    copia.write_text(texto, encoding='utf-8')
    try:
        subprocess.run([str(exe), '--headless=new', '--disable-gpu', '--no-first-run', '--no-pdf-header-footer',
                        f'--user-data-dir={perfil}', '--virtual-time-budget=20000', f'--print-to-pdf={pdf}',
                        copia.resolve().as_uri()],
                       check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    if not (HTML / '00_introduccion.html').exists():
        raise SystemExit('Falta el libro construido: ejecuta antes  .venv/Scripts/jupyter-book build .')
    exe = navegador()
    config = yaml.safe_load((RAIZ / '_config.yml').read_text(encoding='utf-8'))
    escritor = PdfWriter()
    # El navegador deja procesos en segundo plano que sueltan la caché del perfil un poco después.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        perfil = Path(tmp) / 'perfil'
        for html in paginas_en_orden():
            pdf = Path(tmp) / (html.stem + '.pdf')
            imprimir(exe, html, pdf, perfil)
            lector = PdfReader(pdf)
            inicio = len(escritor.pages)
            for pagina in lector.pages:
                escritor.add_page(pagina)
            escritor.add_outline_item(titulo_de(html), inicio)
            print(f'{html.name}: {len(lector.pages)} páginas')
    escritor.add_metadata({'/Title': config['title'], '/Author': config['author'], '/Subject': 'Entregable 3 · Machine Learning · Universidad del Norte'})
    escritor.write(SALIDA)
    print(f'{SALIDA.name}: {len(escritor.pages)} páginas, {SALIDA.stat().st_size / 2 ** 20:.1f} MB')


if __name__ == '__main__':
    main()
