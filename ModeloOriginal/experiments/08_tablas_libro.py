"""Genera las tablas, figuras y textos del informe (``ProyectoFinal/_generado`` y ``ProyectoFinal/figuras``) a partir
de los resultados, para que ninguna cifra del texto se escriba a mano. La lógica está en ``src/informe.py``, que es el
mismo módulo que usan las celdas de los cuadernos del libro.

Uso: ``python experiments/08_tablas_libro.py`` (todas las piezas, después de los guiones 00 a 07) o
``python experiments/08_tablas_libro.py pruebas veredicto`` (solo las nombradas).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import informe                                                    # noqa: E402

if __name__ == '__main__':
    informe.principal(sys.argv[1:])
