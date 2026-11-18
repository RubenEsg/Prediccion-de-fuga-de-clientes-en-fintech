"""Fuga de clientes · Laboratorio del modelo (Dash).

El modelo original del proyecto, BDL (Boosting con Desplazamiento Latente, programado desde cero en src/original_model.py),
y sus nueve benchmarks para predecir el riesgo de fuga de los clientes de una fintech, funcionando en vivo: cada modelo
final puntúa y explica a un cliente en el momento, con los artefactos de artifacts/ y los datos de prueba de datos/.

Uso:
    python -m venv .venv
    .venv\\Scripts\\activate            (Windows)   ·   source .venv/bin/activate   (macOS / Linux)
    pip install -r requirements.txt
    python app.py                      ->  http://127.0.0.1:8060      (otro puerto: variable PORT)

Si falta la carpeta datos/, se genera con `python preparar_datos.py` y `python preparar_artefactos.py` (necesitan la carpeta
del Entregable 3); lo del modelo original lo añade `python preparar_original.py` (necesita la carpeta ModeloOriginal).
El sistema de diseño (colores por rol, escala, componentes) está en DISENO.md; lo comprueba verificar_estilo.py.

Tres secciones, como pide la guía del Tercer Entregable: Contexto (/), EDA (/eda) y Modelos (/modelos/<vista>), cada una
una página de paginas/ (use_pages). Modelos tiene trece vistas en tres grupos: los diez modelos, el modelo original (la
subsección que pide la guía, §7.3.3, y su ficha) y la EBM, que fue el «modelo nuevo» del Entregable 3 y ahora es un
benchmark interpretable. Cada vista es un módulo de secciones/ con su layout() y sus callbacks. Las direcciones de la
versión de nueve pantallas (/cliente, /comparacion…) redirigen a su vista. De lo que hay en el disco solo se sirven tres
PDF del proyecto, si están junto a esta carpeta (/documento/<clave>), y la ficha del modelo (/ficha.md).
"""
import importlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True               # no dejar __pycache__ dentro de la carpeta de la app

if __name__ == "__main__":                    # lanzada con un Python que no tiene dash (el global, o el botón de VS Code):
    try:                                       # se vuelve a lanzar sola con el Python del .venv de esta carpeta, si existe
        import dash  # noqa: F401
    except ModuleNotFoundError:
        _carpeta = "Scripts" if os.name == "nt" else "bin"
        _venv = Path(__file__).resolve().parent / ".venv" / _carpeta / ("python.exe" if os.name == "nt" else "python")
        if _venv.exists() and Path(sys.executable).resolve() != _venv.resolve():
            import subprocess
            try:
                sys.exit(subprocess.call([str(_venv), "-B", str(Path(__file__).resolve()), *sys.argv[1:]]))
            except KeyboardInterrupt:
                sys.exit(0)
        sys.exit("Falta el paquete dash. Abra iniciar_dash.bat con doble clic (crea el entorno .venv la primera vez) "
                 "o instale las dependencias con: pip install -r requirements.txt")

# Este archivo se ejecuta como guion (python app.py): se registra también con su nombre de módulo para que una página
# pueda hacer `import app` sin ejecutarlo por segunda vez.
sys.modules.setdefault("app", sys.modules[__name__])

import dash  # noqa: E402
import flask  # noqa: E402
from dash import Dash, Input, Output, dcc, html  # noqa: E402

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

import datos_app as D  # noqa: E402
import ui  # noqa: E402

# El lector de parquet de pandas (pyarrow.parquet) se importa aquí, en el hilo principal y antes de atender visitas. Si su
# primera importación ocurre en dos hilos a la vez, Python 3.13 puede dejar el submódulo a medio registrar y desde entonces
# toda lectura de parquet falla («cannot access submodule 'parquet' of module 'pyarrow'»): pasó en Render el 15-nov-2026.
from pandas.io.parquet import get_engine as _motor_parquet  # noqa: E402

_motor_parquet("auto")

MARCA = "Fuga de clientes"
TITULO = "Fuga de clientes · Laboratorio del modelo"
DESCRIPCION = ("Un modelo original programado desde cero, Boosting con Desplazamiento Latente (BDL), frente a nueve benchmarks "
               "para predecir el riesgo de fuga de clientes de una fintech, funcionando en vivo.")

# Las tres secciones de la barra: (ruta, id, nombre, pregunta). El nombre va en la barra y en el título del navegador; la
# pregunta, en el title del enlace.
SECCIONES = [
    ("/", "contexto", "Contexto", "¿Qué problema resolvemos y con qué datos?"),
    ("/eda", "eda", "EDA", "¿Qué muestran los datos de entrenamiento?"),
    ("/modelos", "modelos", "Modelos", "¿Qué modelos hay, cómo predicen y cuál conviene?"),
]
NAV = SECCIONES

# Las vistas de Modelos, en tres grupos: (id, nombre, pregunta, grupo, módulo de secciones/, icono). El orden es el del
# recorrido de la sustentación: los diez modelos, el modelo original y, al final, la EBM como benchmark interpretable.
GRUPOS = [("todos", "Los diez modelos"), ("original", "El modelo original: BDL"), ("ebm", "La EBM, benchmark interpretable")]
VISTAS = [
    ("lideres", "Tabla de líderes", "¿Cuál es la mejor combinación de cada modelo?", "todos", "lideres", "balanza"),
    ("prediccion", "Predicción en vivo", "¿Qué predice cada modelo para un cliente?", "todos", "prediccion", "equipo"),
    ("diagnostico", "Diagnóstico", "¿Cómo acierta y cómo se equivoca cada modelo?", "todos", "diagnostico", "grafico"),
    ("interpretabilidad", "Interpretabilidad", "¿Qué variables usa cada modelo?", "todos", "interpretabilidad", "lupa"),
    ("comparacion", "Comparación estadística", "¿Supera el modelo original a los benchmarks?", "todos", "comparacion", "balanza"),
    ("original", "Modelo original", "¿Qué es BDL, cómo aprende y supera o no a los benchmarks?", "original", "original", "modelo"),
    ("ficha", "Ficha del modelo", "¿Para qué sirve el modelo original y para qué no?", "original", "ficha", "info"),
    ("propuesto", "Algoritmo de la EBM", "¿Qué es la EBM y cómo queda frente a los modelos de clase?", "ebm", "propuesto", "modelo"),
    ("decide", "Cómo decide", "¿Cómo decide la EBM?", "ebm", "modelo", "modelo"),
    ("cliente", "Un cliente", "¿Por qué la EBM marca a este cliente?", "ebm", "cliente", "equipo"),
    ("campana", "Campaña", "¿A quién contactar si no alcanza para todos?", "ebm", "campana", "demanda"),
    ("robustez", "Robustez", "¿Depende la EBM de cómo se entrenó?", "ebm", "robustez", "grafico"),
    ("errores", "Errores", "¿Dónde se equivoca la EBM y por qué?", "ebm", "errores", "lupa"),
]
VISTA_INICIAL = "lideres"


def ruta_vista(vista: str) -> str:
    return f"/modelos/{vista}"


def pregunta(vista: str) -> str:
    """La pregunta (h1) de una vista de Modelos."""
    return next(p for v, _n, p, *_ in VISTAS if v == vista)


# Cada destino del dash como (ruta, icono de pantalla, nombre, pregunta), para ui.destino
DESTINOS = ([("/", "inicio", "Contexto", SECCIONES[0][3]), ("/eda", "datos", "EDA", SECCIONES[1][3])]
            + [(ruta_vista(v), {"decide": "modelo", "propuesto": "modelo", "original": "modelo"}.get(v, v), n, p)
               for v, n, p, *_ in VISTAS])
# El recorrido de la sustentación: el pie «Siguiente» de cada vista lleva a la que sigue
RECORRIDO = [d[0] for d in DESTINOS]
# Las direcciones de la versión de nueve pantallas y un atajo a la subsección del modelo original
REDIRECCIONES = {"/inicio": "/", "/datos": "/eda", "/limites": "/", "/modelos": ruta_vista(VISTA_INICIAL),
                 "/modelo": ruta_vista("decide"), "/cliente": ruta_vista("cliente"), "/campana": ruta_vista("campana"),
                 "/comparacion": ruta_vista("comparacion"), "/robustez": ruta_vista("robustez"),
                 "/errores": ruta_vista("errores"), "/original": ruta_vista("original")}

# Los únicos archivos del disco que se sirven: tres PDF del proyecto, si existen. Cada clave se busca, relativa a la
# carpeta de la app, en la carpeta de trabajo del equipo y en el repositorio publicado (la app en Dashboard/ y los PDF en
# InformeFinal/). Si no está (Render solo copia Dashboard/), la sección Contexto enlaza el PDF en GitHub.
DOCUMENTOS = {
    "informe": ("../Informe Final/main.pdf", "../InformeFinal/informe/main.pdf"),
    "metodologia": ("../Metodologia Entregable 3/main.pdf", "../InformeFinal/metodologia/main.pdf"),
    "resultados": ("../Resultados Entregable 3/main.pdf", "../InformeFinal/resultados/main.pdf"),
}


def _ruta_documento(clave):
    """El PDF de una clave de DOCUMENTOS, o None si la clave no está en la lista o el archivo no existe."""
    for relativa in DOCUMENTOS.get(clave, ()):
        ruta = (AQUI / relativa).resolve()
        if ruta.is_file():
            return ruta
    return None


def documentos_disponibles():
    """{clave: bool}: cuáles de los tres PDF se pueden enlazar con /documento/<clave> (para la pantalla Límites)."""
    return {clave: _ruta_documento(clave) is not None for clave in DOCUMENTOS}


# FUGA_SOLO=lideres,eda → solo esas secciones o vistas de Modelos (para desarrollar una sin depender de las demás; las
# otras dan «página no encontrada»). Vacía, todo: Dash importa él mismo paginas/ (contexto, eda, modelos y la 404).
SOLO = [p.strip() for p in os.environ.get("FUGA_SOLO", "").split(",") if p.strip()]
_desconocidas = [s for s in SOLO if s not in ["contexto", "eda"] + [v for v, *_ in VISTAS]]
if _desconocidas:
    raise SystemExit(f"FUGA_SOLO nombra lo que no existe: {', '.join(_desconocidas)} (hay contexto, eda y las vistas "
                     f"{', '.join(v for v, *_ in VISTAS)})")
app = Dash(__name__, use_pages=True, pages_folder="" if SOLO else str(AQUI / "paginas"), assets_folder=str(AQUI / "assets"),
           title=TITULO, update_title=None, suppress_callback_exceptions=True,
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, viewport-fit=cover"},
                      {"name": "color-scheme", "content": "light"},
                      {"name": "theme-color", "content": ui.ROL["fondo"]}])
server = app.server

# La página se declara en español (lectores de pantalla, partición de palabras, corrector)
assert "<html>" in app.index_string, "la plantilla de Dash cambió: revisar el atributo lang"
app.index_string = app.index_string.replace("<html>", '<html lang="es">', 1)

if SOLO:
    _paginas = [s for s in ("contexto", "eda") if s in SOLO] + (["modelos"] if any(s not in ("contexto", "eda") for s in SOLO)
                                                                   else []) + ["not_found_404"]
    for _id in _paginas:
        # Lo mismo que hace Dash con cada archivo de la carpeta: importar el módulo (que se registra solo) y tomar su layout
        _modulo = importlib.import_module(f"paginas.{_id}")
        _registro = dash.page_registry[_modulo.__name__]
        if _registro.get("supplied_layout") is None:
            _registro["layout"] = _modulo.layout

# El nombre y el título del navegador de cada página salen de NAV, aunque la página registre otros. La descripción
# (la etiqueta meta que Dash escribe por página) es la pregunta de la pantalla seguida de la descripción de la app.
for _pagina in dash.page_registry.values():
    for _ruta, _id, _nombre, _pregunta in NAV:
        if _pagina.get("path") == _ruta and _id != "modelos":          # Modelos pone el título de cada vista
            _pagina["name"] = _nombre
            _pagina["title"] = f"{_nombre} · {MARCA}"
            _pagina["description"] = f"{_pregunta} {DESCRIPCION}"


CLASE_ENLACE = "barra__enlace"
ESTADO = "Modelo en vivo"


def barra_superior():
    """Navegación arriba, fija: logo + «Fuga de clientes» a la izquierda, las tres secciones como pestañas de texto (la
    activa en píldora negra) y el chip «Modelo en vivo» a la derecha."""
    enlaces = []
    for i, (ruta, _pantalla, nombre, pregunta) in enumerate(NAV):
        destino = ruta_vista(VISTA_INICIAL) if ruta == "/modelos" else ruta
        enlaces.append(html.Li(dcc.Link(nombre, href=destino, id=f"nav-{i}", title=pregunta, className=CLASE_ENLACE)))
    return html.Header(html.Div([
        dcc.Link([html.Span(ui.icono("equipo"), className="barra__logo"), html.Span(MARCA, className="barra__nombre")],
                 href="/", className="barra__marca", title="Contexto · Laboratorio del modelo"),
        html.Nav(html.Ul(enlaces, className="barra__lista"), className="barra__nav", **{"aria-label": "Secciones"}),
        html.Div(html.Span([ui.icono("bien", "glifo--s"), html.Span(ESTADO, className="barra__estado-texto")],
                           className="barra__estado",
                           title="El modelo original (BDL) y sus benchmarks puntúan y explican cada cliente en el momento · "
                                 "Machine Learning · Universidad del Norte",
                           **{"aria-label": ESTADO}),
                 className="barra__derecha"),
    ], className="barra__dentro"), className="barra")


def aviso_datos():
    """Aviso global si falta algún archivo imprescindible de datos/ (la app arranca igual y lo dice, en vez de fallar).
    Se evalúa al arrancar: después de generar los datos hay que reiniciar la app."""
    faltan = D.faltan_datos()
    if not faltan:
        return None
    return html.Div(ui.aviso([html.Span("Faltan archivos en datos/: " + ", ".join(faltan) + ". "),
                              html.Span("Genérelos con "), html.Code("python preparar_datos.py"),
                              html.Span(" (necesita la carpeta del Entregable 3) y, lo del modelo original, con "),
                              html.Code("python preparar_original.py"), html.Span(".")],
                             "critico", "La app no encuentra sus datos"), className="aviso-global")


def esqueleto():
    """El layout raíz: la barra, el aviso de los datos (si hace falta) y el contenedor de la pantalla (#contenido)."""
    return html.Div([
        dcc.Location(id="url"),
        barra_superior(),
        html.Main([aviso_datos(), dash.page_container, html.Div(id="siguiente", className="pantalla siguiente")],
                  className="contenido", id="contenido"),
    ], className="app")


app.layout = esqueleto()


def siguiente(ruta):
    """Al pie de cada sección o vista, el enlace a la que sigue en el recorrido (en la última, la vuelta al Contexto)."""
    actual = (ruta or "/").rstrip("/") or "/"
    actual = REDIRECCIONES.get(actual, actual)
    if actual not in RECORRIDO:
        return None
    i = RECORRIDO.index(actual)
    destino, pantalla, nombre, pregunta_ = DESTINOS[(i + 1) % len(DESTINOS)]
    rotulo = f"Siguiente: {nombre}" if i + 1 < len(DESTINOS) else f"Volver a {nombre}"
    return ui.destinos([ui.destino(destino, pantalla, rotulo, pregunta_)])


@app.callback(Output("siguiente", "children"), Input("url", "pathname"))
def _siguiente(ruta):
    return siguiente(ruta)

# La pantalla activa lleva aria-current="page" (dcc.Link no lo admite como propiedad: se pone en el navegador) y la clase
# --activo de respaldo (la píldora negra). Donde las pestañas se desplazan, la activa se centra a la vista; se repite
# cuando termina de cargar la fuente, porque con ella cambia el ancho de las pestañas.
app.clientside_callback(
    r"""function (ruta) {
        var rutas = %s, clase = %s;
        var actual = (ruta || '/').replace(/\/+$/, '') || '/';
        var activa = function (r) { return r === actual || (r !== '/' && actual.indexOf(r + '/') === 0); };
        return rutas.map(function (r, i) {
            var a = document.getElementById('nav-' + i);
            if (a) {
                if (activa(r)) {
                    a.setAttribute('aria-current', 'page');
                    var lista = a.closest('ul');
                    if (lista) {
                        var centrar = function () {
                            if (a.getAttribute('aria-current') !== 'page') { return; }
                            lista.scrollLeft = a.parentNode.offsetLeft - lista.offsetLeft - (lista.clientWidth - a.offsetWidth) / 2;
                        };
                        centrar();
                        if (document.fonts && document.fonts.ready) { document.fonts.ready.then(centrar); }
                    }
                } else { a.removeAttribute('aria-current'); }
            }
            return activa(r) ? clase + ' ' + clase + '--activo' : clase;
        });
    }""" % (json.dumps([r for r, *_ in NAV]), json.dumps(CLASE_ENLACE)),
    [Output(f"nav-{i}", "className") for i in range(len(NAV))], Input("url", "pathname"))


def _redirigir(destino):
    def vista():
        consulta = flask.request.query_string.decode()
        return flask.redirect(destino + (("&" if "?" in destino else "?") + consulta if consulta else ""), code=302)
    return vista


for _vieja, _nueva in REDIRECCIONES.items():
    server.add_url_rule(_vieja, endpoint=f"redirigir{_vieja.replace('/', '_')}", view_func=_redirigir(_nueva))


@server.route("/ficha.md")
def servir_ficha():
    """La ficha del modelo original (MODEL_CARD.md) tal cual, para descargarla desde la vista Ficha del modelo."""
    ruta = AQUI / "MODEL_CARD.md"
    if not ruta.is_file():
        flask.abort(404)
    return flask.send_file(ruta, mimetype="text/markdown", download_name="MODEL_CARD.md")


@server.route("/documento/<clave>")
def servir_documento(clave):
    """Uno de los tres PDF de DOCUMENTOS. 404 si la clave no está en la lista o el archivo no existe."""
    ruta = _ruta_documento(clave)
    if ruta is None:
        flask.abort(404)
    return flask.send_file(ruta, mimetype="application/pdf", download_name=f"{clave}.pdf")


def precalentar():
    """Llena las cachés antes de la primera visita: carga las tuberías de artifacts/ (las 14 del Entregable 3 y las del
    modelo original) y predice una vez con cada una, y
    arma una vez cada sección y vista. Con FUGA_PRECALENTAR=1 (lo pone render.yaml) corre al importar la app, en el hilo
    principal: gunicorn no atiende visitas hasta que termina (unos 25 s en Render). No corre en un hilo aparte a propósito:
    así ninguna librería se importa ni ningún modelo se carga en dos hilos a la vez; en el servidor gratuito, sin esto, la
    primera predicción tardaría varios segundos."""
    import time
    t0 = time.perf_counter()
    fallas = 0
    try:
        import modelos_vivos as V
        import motor as M
        X = M.clientes()[V.columnas()].iloc[[0]]
        for tarea in V.TAREAS:
            for modelo in V.modelos(tarea):
                V.predecir(tarea, modelo, X)
    except Exception as e:  # noqa: BLE001 - el precalentamiento nunca debe tumbar la app
        fallas += 1
        print(f"precalentar: los modelos fallaron: {type(e).__name__}: {e}", flush=True)
    for pagina in list(dash.page_registry.values()):
        variantes = ([{"vista": v} for v, *_ in VISTAS] if pagina.get("path_template") else [{}])
        for kw in variantes:
            try:
                pagina["layout"](**kw) if callable(pagina["layout"]) else None
            except Exception as e:  # noqa: BLE001
                fallas += 1
                print(f"precalentar: {pagina['module']} {kw} falló: {type(e).__name__}: {e}", flush=True)
    print(f"precalentar: listo en {time.perf_counter() - t0:.1f} s, {fallas} fallas", flush=True)


if os.environ.get("FUGA_PRECALENTAR") == "1":
    precalentar()


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", "8060"))
    print(f"{TITULO}: abra http://127.0.0.1:{puerto} en el navegador (Ctrl+C para detener).")
    app.run(host="127.0.0.1", port=puerto, debug=False)
