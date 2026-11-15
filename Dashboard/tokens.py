"""Tokens del sistema de diseño, leídos del bloque :root de assets/estilos.css (la única fuente; DISENO.md §2 y §3).

Python nunca repite un color: TOKENS es {nombre: valor tal como está en el CSS}, ROL es {rol: color} y cada rol apunta a un
token. Solo usa la biblioteca estándar, así que también lo importa verificar_estilo.py desde cualquier Python.

    TOKENS["--serie-modelo"]  -> el valor escrito en estilos.css
    color("--franja")         -> el color resuelto (sigue los var() y calcula los color-mix con transparent)
    ROL["contexto"]           -> el color del rol «contexto» (--serie-contexto)
    VAR["contexto"]           -> "var(--serie-contexto)", para los estilos en línea (--c, --c-tinte)
    alfa(ROL["modelo"], LAVADO) -> rgba con el alfa pedido (lavado de área, franja)
    px("--t-xs")              -> 12 (los tamaños de letra de Plotly salen de aquí)
"""
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CSS = AQUI / "assets" / "estilos.css"


# --- Lectura del CSS --------------------------------------------------------------------------------------------------
def quitar_comentarios(texto):
    """Quita los comentarios /* … */ y conserva los saltos de línea (para poder dar el número de línea)."""
    return re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), texto, flags=re.S)


def estatutos(texto):
    """Estatutos de primer nivel de un bloque CSS: lista de (preludio, cuerpo, línea del preludio, inicio del cuerpo),
    con el cuerpo sin las llaves exteriores. La línea y el inicio son relativos a `texto`."""
    out, i, n, inicio = [], 0, len(texto), 0
    while i < n:
        c = texto[i]
        if c == "{":
            crudo = texto[inicio:i]
            linea = texto.count("\n", 0, inicio + len(crudo) - len(crudo.lstrip())) + 1
            prof, j = 1, i + 1
            while j < n and prof:
                if texto[j] == "{":
                    prof += 1
                elif texto[j] == "}":
                    prof -= 1
                j += 1
            out.append((crudo.strip(), texto[i + 1:j - 1], linea, i + 1))
            i = inicio = j
            continue
        if c == ";":
            inicio = i + 1          # @import, @charset u otra regla sin cuerpo
        i += 1
    return out


def declaraciones(cuerpo):
    """Declaraciones de un cuerpo CSS: lista de (propiedad, valor), separando por «;» fuera de paréntesis y comillas."""
    out, prof, actual, comilla = [], 0, [], None
    for c in cuerpo:
        if comilla:
            actual.append(c)
            if c == comilla:
                comilla = None
            continue
        if c in "\"'":
            comilla = c
        elif c == "(":
            prof += 1
        elif c == ")":
            prof -= 1
        elif c == ";" and prof == 0:
            out.append("".join(actual))
            actual = []
            continue
        actual.append(c)
    out.append("".join(actual))
    pares = []
    for d in out:
        if ":" in d:
            k, v = d.split(":", 1)
            k, v = k.strip(), " ".join(v.split())
            if k and v:
                pares.append((k, v))
    return pares


def bloques_root(texto):
    """Los tres bloques de tokens: «raiz» (:root fuera de @media), «telefono» (:root dentro de @media (max-width: 600px)) y
    «movimiento» (:root dentro de @media (prefers-reduced-motion: reduce)). Otros selectores no cuentan."""
    texto = quitar_comentarios(texto)
    res = {"raiz": {}, "telefono": {}, "movimiento": {}}
    for pre, cuerpo, *_ in estatutos(texto):
        if pre == ":root":
            res["raiz"].update({k: v for k, v in declaraciones(cuerpo) if k.startswith("--")})
        elif pre.startswith("@media"):
            clave = "telefono" if re.search(r"max-width:\s*600px", pre) else "movimiento" if "prefers-reduced-motion" in pre else None
            if not clave:
                continue
            for pre2, cuerpo2, *_ in estatutos(cuerpo):
                if pre2 == ":root":
                    res[clave].update({k: v for k, v in declaraciones(cuerpo2) if k.startswith("--")})
    return res


def leer(ruta=CSS):
    return bloques_root(Path(ruta).read_text(encoding="utf-8"))


_BLOQUES = leer()
TOKENS = dict(_BLOQUES["raiz"])
TOKENS_TELEFONO = dict(_BLOQUES["telefono"])
TOKENS_MOVIMIENTO_REDUCIDO = dict(_BLOQUES["movimiento"])


# --- Resolver valores -------------------------------------------------------------------------------------------------
_VAR = re.compile(r"^var\(\s*(--[\w-]+)\s*(?:,\s*(.+))?\)$")
_MEZCLA = re.compile(r"^color-mix\(\s*in\s+srgb\s*,\s*(.+?)\s+(\d+(?:\.\d+)?)%\s*,\s*transparent\s*\)$")


def resolver(valor_o_nombre, tokens=None, _visto=()):
    """Valor final de un token o de un valor CSS: sigue los var() (con su respaldo) hasta llegar a un valor literal."""
    tokens = TOKENS if tokens is None else tokens
    v = tokens.get(valor_o_nombre, valor_o_nombre) if valor_o_nombre.startswith("--") else valor_o_nombre
    m = _VAR.match(v.strip())
    while m:
        nombre, respaldo = m.group(1), m.group(2)
        if nombre in _visto:
            raise ValueError(f"var() circular en {nombre}")
        _visto = _visto + (nombre,)
        if nombre in tokens:
            v = tokens[nombre]
        elif respaldo is not None:
            v = respaldo
        else:
            raise KeyError(f"token sin definir: {nombre}")
        m = _VAR.match(v.strip())
    return v.strip()


def rgba(valor):
    """(r, g, b, a) de un color CSS: #rgb, #rrggbb, #rrggbbaa, rgb(), rgba(), transparent o color-mix(in srgb, X p%, transparent)
    (con var() dentro). None si el valor no es un color."""
    v = resolver(valor).lower()
    if v == "transparent":
        return (0, 0, 0, 0.0)
    m = _MEZCLA.match(v)
    if m:
        base = rgba(m.group(1))
        if base is None:
            return None
        return (base[0], base[1], base[2], round(base[3] * float(m.group(2)) / 100, 4))
    if re.fullmatch(r"#[0-9a-f]{3}", v):
        return tuple(int(c * 2, 16) for c in v[1:]) + (1.0,)
    if re.fullmatch(r"#[0-9a-f]{6}([0-9a-f]{2})?", v):
        a = int(v[7:9], 16) / 255 if len(v) == 9 else 1.0
        return (int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16), round(a, 4))
    m = re.fullmatch(r"rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)(?:[\s,/]+([\d.]+%?))?\s*\)", v)
    if m:
        a = m.group(4)
        a = 1.0 if a is None else (float(a[:-1]) / 100 if a.endswith("%") else float(a))
        return (round(float(m.group(1))), round(float(m.group(2))), round(float(m.group(3))), a)
    return None


def a_texto(r, g, b, a=1.0):
    """#rrggbb si es opaco; rgba(r, g, b, a) si no."""
    if a >= 1:
        return f"#{r:02x}{g:02x}{b:02x}"
    return f"rgba({r}, {g}, {b}, {round(a, 4):g})"


def color(nombre):
    """Color resuelto de un token («--serie-modelo» → su hex; «--franja» → su rgba)."""
    c = rgba(nombre)
    if c is None:
        raise ValueError(f"{nombre} no es un color: {resolver(nombre)}")
    return a_texto(*c)


def alfa(c, a):
    """El color c (hex, rgba o nombre de token) con opacidad a (0 a 1), como rgba(…) para Plotly."""
    valor = rgba(str(c))
    if valor is None:
        raise ValueError(f"{c} no es un color")
    r, g, b, a0 = valor
    return f"rgba({r}, {g}, {b}, {round(a0 * a, 4):g})"


def px(nombre, tokens=None):
    """Número de píxeles de un token de tamaño («12px» → 12). Falla con clamp() u otras expresiones."""
    v = resolver(nombre, tokens)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)px", v)
    if not m:
        raise ValueError(f"{nombre} no es un tamaño en px: {v}")
    x = float(m.group(1))
    return int(x) if x.is_integer() else x


def porcentaje(nombre):
    """Fracción de un token en % («10%» → 0.1)."""
    v = resolver(nombre)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%", v)
    if not m:
        raise ValueError(f"{nombre} no es un porcentaje: {v}")
    return float(m.group(1)) / 100


def colores_de_la_paleta():
    """{token: (r, g, b, a)} de todos los tokens que son un color (para comprobar los colores calculados en el navegador)."""
    out = {}
    for k in TOKENS:
        try:
            c = rgba(k)
        except (KeyError, ValueError):
            c = None
        if c is not None:
            out[k] = c
    return out


# --- Roles (DISENO.md §2) ---------------------------------------------------------------------------------------------
# Cada rol apunta a un token: el color nunca se escribe aquí. Si un rol apunta a un token que no existe, la importación
# falla (un renombrado a medias no pasa inadvertido).
ROL_TOKEN = {
    # Series de los gráficos, en el orden fijo de la leyenda. Cada entidad tiene el mismo color en las nueve pantallas:
    "modelo": "--serie-modelo",             # azul: la EBM, lo que se propone (sólida = la serie en foco; rayada = las demás de su familia)
    "rival": "--serie-rival",               # rosa: XGBoost y los otros modelos de árboles con los que la EBM empata
    "base": "--serie-base",                 # gris: modelos lineales, k-NN, Naive Bayes y la Entrega 2; nunca con la misma forma que el rosa
    "real": "--serie-real",                 # casi negro: lo observado; línea de 2,5 px o puntos
    "contexto": "--serie-contexto",         # gris claro: el contexto (1,6:1 sobre la tarjeta: siempre con rótulo o tabla)
    # Tintes: el fondo del rayado y de las bandas de cada familia (nunca una serie por sí solos)
    "modelo_tinte": "--serie-modelo-tinte",
    "rival_tinte": "--serie-rival-tinte",
    "base_tinte": "--serie-base-tinte",
    # Mobiliario de los gráficos
    "referencia": "--tinta-3",              # línea de referencia sólida de 1 px (un umbral, el azar)
    "cero": "--tinta-3",                    # línea de cero, solo donde importa el signo
    "rejilla": "--rejilla",
    "eje": "--eje",
    "franja": "--franja",                   # el azul al 9 %: banda de foco o de rango
    # Estado: solo estados, siempre con icono y palabra; nunca una serie
    "bien": "--bien",
    "atencion": "--atencion",
    "atencion_texto": "--atencion-texto",
    "critico": "--critico",
    # Interfaz
    "acento": "--acento",
    "acento_fuerte": "--acento-fuerte",
    "acento_suave": "--acento-suave",
    "foco": "--foco",
    "tinta": "--tinta",
    "tinta_2": "--tinta-2",
    "tinta_3": "--tinta-3",
    "tinta_tenue": "--tinta-tenue",         # solo texto de 24 px o más y marcas inactivas (3,3:1)
    "tinta_inversa": "--tinta-inversa",
    "blanco": "--blanco",                   # píldoras, chips, etiqueta flotante de Plotly, pistas
    "superficie": "--superficie",
    "superficie_2": "--superficie-2",
    "fondo": "--fondo",
    "linea": "--linea",
    "linea_fuerte": "--linea-fuerte",
}

# Los cinco roles de serie, en el orden de la leyenda
SERIES = ("modelo", "rival", "base", "real", "contexto")

ROL = {rol: color(tok) for rol, tok in ROL_TOKEN.items()}
VAR = {rol: f"var({tok})" for rol, tok in ROL_TOKEN.items()}

LAVADO = porcentaje("--lavado")           # 0.1: el lavado de área es el color de la serie al 10 %
FUENTE_TEXTO = TOKENS["--fuente-texto"]
FUENTE_TITULOS = TOKENS["--fuente-titulos"]
FUENTE_MONO = TOKENS["--fuente-mono"]
