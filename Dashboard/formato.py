"""Números en es-CO para toda la app (coma decimal, punto de miles, menos tipográfico «−»).

Redondea mitad hacia arriba con Decimal, igual que el informe (Resultados Entregable 3/generar.py::num), para que una cifra
de la app y la misma cifra del PDF tengan siempre los mismos dígitos. Solo biblioteca estándar.

    fmt(0.82892, 4)            -> '0,8289'      (con «-» corto si es negativo: lo usan ui.num y ui.con_signo)
    num(-0.0057, 4)            -> '−0,0057'
    con_signo(0.0006, 4)       -> '+0,0006'
    pct(0.5119)                -> '51,2 %'      (recibe una FRACCIÓN, como el proyecto entregado)
    prob(0.99689)              -> '0,997'       (los decimales justos para no redondear a 1)
    entero(48723)              -> '48.723'
    intervalo(0.8206, 0.8366)  -> '[0,8206; 0,8366]'
    p_valor(0.0004)            -> 'p < 0,001'
"""
import math
from decimal import ROUND_HALF_UP, Decimal

MENOS = "−"
RAYA = "—"
FIJO = chr(160)       # espacio que no parte la línea: la cifra y su «%» no se separan


def _finito(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _absoluto(x, d):
    """Texto del valor absoluto redondeado a d decimales (mitad hacia arriba), con punto de miles y coma decimal."""
    q = Decimal(repr(float(abs(x)))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    entero, _, frac = f"{q:f}".partition(".")
    return f"{int(entero):,}".replace(",", ".") + ("," + frac if frac else "")


def _es_cero(texto):
    return not any(c in "123456789" for c in texto)


def fmt(x, d=0):
    """Número en es-CO: 1.234,5. Lo que no es un número finito sale como «—». El negativo lleva «-» corto."""
    if not _finito(x):
        return RAYA
    s = _absoluto(x, d)
    return "-" + s if (float(x) < 0 and not _es_cero(s)) else s


def num(x, d=0):
    """Como fmt, con el menos tipográfico: −0,32."""
    s = fmt(x, d)
    return MENOS + s[1:] if s.startswith("-") else s


def con_signo(x, d=0):
    """Con signo explícito: +0,0006 / −0,0057 (el cero va sin signo)."""
    if not _finito(x):
        return RAYA
    s = _absoluto(x, d)
    if _es_cero(s):
        return s
    return ("+" if float(x) > 0 else MENOS) + s


def pct(x, d=1, signo=False):
    """Fracción como porcentaje: pct(0.2573) -> '25,7 %' (con un espacio que no parte la línea entre la cifra y el signo)."""
    if not _finito(x):
        return RAYA
    return (con_signo(100 * float(x), d) if signo else num(100 * float(x), d)) + FIJO + "%"


def prob(p):
    """Probabilidad con los decimales justos para no redondearla a 0 ni a 1: 0,94 · 0,997 · 0,9998 (como el informe)."""
    if not _finito(p):
        return RAYA
    for d in (2, 3, 4):
        texto = num(p, d)
        if texto not in (num(0, d), num(1, d)):
            return texto
    return num(p, 4)


def entero(x):
    """Entero con punto de miles: 48.723."""
    return num(x, 0)


def intervalo(inf, sup, d=4, signo=False):
    """Intervalo [a; b] con punto y coma, para no confundir la coma decimal (con un espacio que no parte la línea)."""
    f = con_signo if signo else num
    return f"[{f(inf, d)};{FIJO}{f(sup, d)}]"


def p_valor(p, d=3):
    """Valor p con su signo de relación: 'p = 0,004' o 'p < 0,001' (con espacios que no parten la línea)."""
    if not _finito(p):
        return "p no disponible"
    piso = 10 ** -d
    return f"p{FIJO}<{FIJO}{num(piso, d)}" if float(p) < piso else f"p{FIJO}={FIJO}{num(p, d)}"


def enumerar(elementos):
    """Lista en prosa: enumerar(['a', 'b', 'c']) -> 'a, b y c'."""
    elementos = [str(e) for e in elementos]
    if len(elementos) <= 1:
        return "".join(elementos)
    return ", ".join(elementos[:-1]) + " y " + elementos[-1]
