"""Compila el artículo (plantilla Springer sn-jnl) y deja ``Entregable3_articulo.pdf`` en la raíz del proyecto.

Usa Tectonic si está disponible (``tectonic`` en el PATH, la variable de entorno ``TECTONIC`` o
``articulo/herramientas/tectonic.exe``); si no, pdflatex + bibtex de una instalación de TeX Live o MiKTeX.
Tectonic descarga los paquetes que necesita la primera vez (hace falta conexión).

Uso, desde la carpeta del proyecto::

    .venv\\Scripts\\python.exe articulo/compilar.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
FUENTE = 'articulo'
SALIDA = RAIZ / 'Entregable3_articulo.pdf'


def tectonic() -> str | None:
    candidatos = [os.environ.get('TECTONIC'), shutil.which('tectonic'), str(AQUI / 'herramientas' / 'tectonic.exe')]
    return next((c for c in candidatos if c and Path(c).exists()), None)


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    for f in ('articulo.tex', 'sn-jnl.cls', 'sn-mathphys-num.bst', 'sn-bibliography.bib', 'tablas/clasificacion.tex'):
        if not (AQUI / f).exists():
            raise SystemExit(f'Falta {f}; si es una tabla, ejecuta antes  python articulo/generar_material.py')
    exe = tectonic()
    if exe:
        subprocess.run([exe, '--keep-logs', f'{FUENTE}.tex'], cwd=AQUI, check=True)
    elif shutil.which('pdflatex') and shutil.which('bibtex'):
        for orden in ([ 'pdflatex', '-interaction=nonstopmode', FUENTE], ['bibtex', FUENTE],
                      ['pdflatex', '-interaction=nonstopmode', FUENTE], ['pdflatex', '-interaction=nonstopmode', FUENTE]):
            subprocess.run(orden, cwd=AQUI, check=True, stdout=subprocess.DEVNULL)
    else:
        raise SystemExit('No hay compilador de LaTeX: instala Tectonic (https://tectonic-typesetting.github.io) o TeX Live/MiKTeX.')
    shutil.copy2(AQUI / f'{FUENTE}.pdf', SALIDA)
    print(f'{SALIDA.name}: {SALIDA.stat().st_size / 2 ** 20:.1f} MB')


if __name__ == '__main__':
    main()
