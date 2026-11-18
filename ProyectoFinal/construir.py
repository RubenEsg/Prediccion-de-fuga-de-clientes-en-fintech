"""Convierte las fuentes de los capítulos del informe en cuadernos y los ejecuta.

Cada capítulo se escribe en ``ProyectoFinal/fuentes/NN_nombre.txt`` como una sucesión de celdas:

``#%% md``
    Celda de texto (Markdown de MyST: fórmulas, citas, referencias cruzadas).
``#%% code`` o ``#%% code [etiqueta, ...]``
    Celda de Python. Las etiquetas son las de Jupyter Book (``hide-input``, ``remove-cell``...).
``#%% figura <nombre> [<ancho>]``
    Celda de Python cuya salida es una figura con leyenda. El cuerpo tiene dos partes separadas por una línea ``---``:
    primero la leyenda (Markdown) y después el código. ``<nombre>`` es la etiqueta para citarla con ``{ref}``.

El guion genera ``ProyectoFinal/NN_nombre.ipynb``, lo ejecuta de principio a fin con el kernel de este entorno y guarda
las salidas en el propio cuaderno. El libro se construye después sin volver a ejecutar (``execute_notebooks: off``) y
GitHub muestra los cuadernos ya ejecutados. Las celdas no entrenan nada: leen los resultados versionados en
``ModeloOriginal/resultados`` y vuelven a generar cada tabla y cada figura con ``ModeloOriginal/src/informe.py``.

Uso, desde la raíz del repositorio::

    python ProyectoFinal/construir.py              (todos los capítulos)
    python ProyectoFinal/construir.py 06 07        (solo los que empiezan por 06 o 07)
    python ProyectoFinal/construir.py --sin-ejecutar 05
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

LIBRO = Path(__file__).resolve().parent
FUENTES = LIBRO / 'fuentes'
SEPARADOR = re.compile(r'^#%% (md|code|figura)(?: ([^\n]*))?$', re.M)


def ajustar_comentarios(codigo: str, ancho: int = 84) -> str:
    """Recompone cada bloque de comentarios al ancho de la celda en el libro, para que ninguna línea quede cortada."""
    lineas, bloque = [], []

    def vaciar() -> None:
        if bloque:
            lineas.extend(textwrap.wrap(' '.join(bloque), width=ancho, initial_indent='# ', subsequent_indent='# '))
            bloque.clear()

    for linea in codigo.split('\n'):
        if linea.startswith('# '):
            bloque.append(linea[2:].strip())
        else:
            vaciar()
            lineas.append(linea)
    vaciar()
    return '\n'.join(lineas)


def leer_fuente(ruta: Path) -> nbformat.NotebookNode:
    """Construye el cuaderno a partir de una fuente con celdas ``#%% md``, ``#%% code`` y ``#%% figura``."""
    partes = SEPARADOR.split(ruta.read_text(encoding='utf-8'))
    nb = nbformat.v4.new_notebook()
    for tipo, opciones, cuerpo in zip(partes[1::3], partes[2::3], partes[3::3]):
        cuerpo, opciones = cuerpo.strip('\n'), (opciones or '').strip()
        if not cuerpo.strip():
            continue
        if tipo == 'md':
            celda = nbformat.v4.new_markdown_cell(cuerpo)
        elif tipo == 'code':
            celda = nbformat.v4.new_code_cell(ajustar_comentarios(cuerpo))
            if opciones:
                celda.metadata['tags'] = [e.strip() for e in opciones.strip('[]').split(',') if e.strip()]
        else:
            nombre, _, ancho = opciones.partition(' ')
            leyenda, separador, codigo = cuerpo.partition('\n---\n')
            if not separador:
                raise SystemExit(f'{ruta.name}: a la figura {nombre} le falta la línea --- entre la leyenda y el código')
            celda = nbformat.v4.new_code_cell(ajustar_comentarios(codigo.strip('\n')))
            celda.metadata['mystnb'] = {'figure': {'name': nombre, 'caption': ' '.join(leyenda.split())},
                                        'image': {'width': ancho.strip() or '90%', 'align': 'center'}}
        nb.cells.append(celda)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    # las salidas en Markdown (tablas y párrafos generados) se componen con MyST, igual que el texto del capítulo
    nb.metadata['mystnb'] = {'render_markdown_format': 'myst'}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('capitulos', nargs='*', help='prefijos de los capítulos a construir (por defecto todos)')
    ap.add_argument('--sin-ejecutar', action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.platform == 'win32':          # prioridad baja (la hereda el kernel): no quita CPU a los experimentos
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
    fuentes = sorted(FUENTES.glob('*.txt'))
    if args.capitulos:
        fuentes = [f for f in fuentes if any(f.name.startswith(p) for p in args.capitulos)]
    for fuente in fuentes:
        destino = LIBRO / (fuente.stem + '.ipynb')
        nb = leer_fuente(fuente)
        t0 = time.perf_counter()
        if not args.sin_ejecutar:
            NotebookClient(nb, timeout=3600, kernel_name='python3', resources={'metadata': {'path': str(LIBRO)}}).execute()
        nbformat.write(nb, destino)
        print(f'{destino.name}: {len(nb.cells)} celdas · {time.perf_counter() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
