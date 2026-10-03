"""Convierte las fuentes de las páginas del libro en cuadernos y los ejecuta.

Cada página se escribe en ``libro/fuentes/NN_nombre.txt`` como una sucesión de celdas separadas
por líneas ``#%% md`` (Markdown) o ``#%% code`` (Python); tras ``code`` pueden ir etiquetas de
Jupyter Book entre corchetes, p. ej. ``#%% code [hide-input]``. El guion genera
``libro/NN_nombre.ipynb``, lo ejecuta de principio a fin con el kernel de este entorno y guarda las
salidas en el propio cuaderno: el libro se construye después sin volver a ejecutar
(``execute_notebooks: off``) y GitHub muestra los cuadernos ya ejecutados.

Uso, desde la carpeta del proyecto::

    .venv\\Scripts\\python.exe libro/construir.py              (todas las páginas)
    .venv\\Scripts\\python.exe libro/construir.py 03 06        (solo las que empiezan por 03 o 06)
    .venv\\Scripts\\python.exe libro/construir.py --sin-ejecutar 03
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

LIBRO = Path(__file__).resolve().parent
FUENTES = LIBRO / 'fuentes'
SEPARADOR = re.compile(r'^#%% (md|code)(?: \[(.*)\])?\s*$', re.M)


def leer_fuente(ruta: Path) -> nbformat.NotebookNode:
    """Construye el cuaderno a partir de una fuente con celdas ``#%% md`` / ``#%% code``."""
    texto = ruta.read_text(encoding='utf-8')
    partes = SEPARADOR.split(texto)
    nb = nbformat.v4.new_notebook()
    for tipo, etiquetas, cuerpo in zip(partes[1::3], partes[2::3], partes[3::3]):
        cuerpo = cuerpo.strip('\n')
        if not cuerpo.strip():
            continue
        if tipo == 'md':
            celda = nbformat.v4.new_markdown_cell(cuerpo)
        else:
            celda = nbformat.v4.new_code_cell(cuerpo)
            if etiquetas:
                celda.metadata['tags'] = [e.strip() for e in etiquetas.split(',')]
        nb.cells.append(celda)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('paginas', nargs='*', help='prefijos de las páginas a construir (por defecto todas)')
    ap.add_argument('--sin-ejecutar', action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.platform == 'win32':          # prioridad baja (la hereda el kernel): no quita CPU a las corridas
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
    fuentes = sorted(FUENTES.glob('*.txt'))
    if args.paginas:
        fuentes = [f for f in fuentes if any(f.name.startswith(p) for p in args.paginas)]
    for fuente in fuentes:
        destino = LIBRO / (fuente.stem + '.ipynb')
        nb = leer_fuente(fuente)
        t0 = time.perf_counter()
        if not args.sin_ejecutar:
            NotebookClient(nb, timeout=7200, kernel_name='python3',
                           resources={'metadata': {'path': str(LIBRO)}}).execute()
        nbformat.write(nb, destino)
        print(f'{destino.name}: {len(nb.cells)} celdas · {time.perf_counter() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
