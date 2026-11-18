"""Prepara las tablas maestras para publicarlas: convierte en relativas al repositorio las rutas que guardó cada corrida.

Cada corrida registra dónde quedó su modelo final y sus predicciones fuera de pliegue con la ruta absoluta del equipo
donde se ejecutó. Esas rutas no deben publicarse ni sirven en otro equipo. Este guion las reescribe relativas a la raíz
del repositorio en la tabla de COFINFAD y en la de cada conjunto adicional. Se puede ejecutar varias veces: lo que ya es
relativo no cambia. Los guiones que leen una ruta usan ``src.registro.resolver``, que acepta las dos formas.

    python experiments/09_rutas_relativas.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import conjuntos, plan                                            # noqa: E402
from src.registro import COLUMNAS_RUTA, resolver                           # noqa: E402


def main() -> None:
    for conjunto in ['cofinfad'] + list(conjuntos.CARGADORES):
        reg = plan.registro(conjunto)
        if not reg.ruta.exists():
            print(f'{conjunto}: sin tabla')
            continue
        cambiadas = reg.rutas_relativas()
        tabla = reg.cargar(decodificar=False)
        rutas = [r for c in COLUMNAS_RUTA if c in tabla.columns for r in tabla[c] if r]
        absolutas = sum(Path(r).is_absolute() for r in rutas)
        presentes = sum(resolver(r).is_file() for r in rutas)
        print(f'{conjunto}: {cambiadas} rutas convertidas; {len(rutas)} en total, {absolutas} absolutas, '
              f'{presentes} con su archivo en este equipo')


if __name__ == '__main__':
    main()
