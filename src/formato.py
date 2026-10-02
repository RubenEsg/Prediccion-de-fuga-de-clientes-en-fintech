"""Números en la convención del castellano para el texto del libro.

Toda cifra que aparece en la prosa se genera con estas funciones a partir de variables (regla
de la Entrega 2: ninguna cifra escrita a mano). Las tablas de pandas conservan el punto decimal.
"""
from __future__ import annotations

import math


def dec(x: float, d: int = 4, signo: bool = False) -> str:
    """Decimal con coma y signo menos tipográfico: ``dec(0.68321) == '0,6832'``, ``dec(-0.5, 1) == '−0,5'``;
    con ``signo``, también el más: ``'+0,0400'``."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    s = f'{x:+,.{d}f}' if signo else f'{x:,.{d}f}'
    return s.replace(',', '@').replace('.', ',').replace('@', '.').replace('-', '−')


def ent(x: float) -> str:
    """Entero con punto de miles: ``ent(48723) == '48.723'``."""
    return f'{int(round(x)):,}'.replace(',', '.')


def pct(x: float, d: int = 1, signo: bool = False) -> str:
    """Fracción como porcentaje: ``pct(0.2573) == '25,7 %'``."""
    return f'{dec(100 * x, d, signo)} %'


_SUPERINDICES = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def p_valor(p: float, d: int = 3, grafico: bool = False) -> str:
    """p-valor con su signo de relación: ``'p = 0,004'``, ``'p < 0,001'`` o ``'p < 10⁻⁷'``.

    Con ``grafico=True`` el exponente se escribe en *mathtext* (``$10^{-7}$``), porque la fuente
    de las figuras no tiene los superíndices Unicode.
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 'p no disponible'
    if p >= 10 ** -d:
        return f'p = {dec(p, d)}'
    if p <= 0:                     # un p-valor bootstrap nulo solo dice que es menor que la resolución
        return f'p < {dec(10 ** -d, d)}'
    exponente = int(math.floor(math.log10(p))) + 1
    if exponente >= -d:
        return f'p < {dec(10 ** -d, d)}'
    if grafico:
        return f'p < $10^{{{exponente}}}$'
    return f'p < 10{str(exponente).translate(_SUPERINDICES)}'


def enumerar(elementos) -> str:
    """Lista en prosa: ``enumerar(['a', 'b', 'c']) == 'a, b y c'``."""
    elementos = [str(e) for e in elementos]
    if len(elementos) <= 1:
        return ''.join(elementos)
    return ', '.join(elementos[:-1]) + ' y ' + elementos[-1]


def intervalo(inf: float, sup: float, d: int = 4, signo: bool = False) -> str:
    """Intervalo ``[a; b]`` con punto y coma, para no confundir la coma decimal."""
    return f'[{dec(inf, d, signo)}; {dec(sup, d, signo)}]'
