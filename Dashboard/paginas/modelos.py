"""Sección Modelos (/modelos/<vista>): una sola página con las vistas de app.VISTAS en tres grupos (app.GRUPOS).

«Los diez modelos» reúne lo que la guía pide para todos (tabla de líderes, predicción en vivo con selector de modelo,
diagnóstico, interpretabilidad global y comparación estadística); «El modelo original: BDL» reúne la subsección del
modelo original (descripción, pseudocódigo, diferencias con la literatura, convergencia, pruebas, robustez y veredicto) y
su ficha; «La EBM, benchmark interpretable» conserva las vistas de la EBM del Entregable 3 (su algoritmo, cómo decide, un
cliente, la campaña, su robustez y sus errores), con un aviso arriba que dice que es un benchmark y no el modelo original.
La subnavegación va arriba de cada vista; cada vista es un módulo de secciones/ y la página lo arma con los parámetros de
la dirección (?tarea=, ?fila=, ?modelo=…). Todos los módulos se importan al cargar este archivo, para que sus callbacks
queden registrados antes de que arranque el servidor.
"""
import importlib

import dash
from dash import dcc, html

import app
import ui

MODULOS = {v: importlib.import_module(f"secciones.{modulo}") for v, _n, _p, _g, modulo, _i in app.VISTAS
           if not app.SOLO or v in app.SOLO}
NOMBRE = {v: n for v, n, *_ in app.VISTAS}
GRUPO = {v: g for v, _n, _p, g, *_ in app.VISTAS}


def _titulo(vista=None, **_):
    return f"{NOMBRE.get(vista, 'Modelos')} · Modelos · {app.MARCA}"


def _descripcion(vista=None, **_):
    pregunta = next((p for v, _n, p, *_ in app.VISTAS if v == vista), app.SECCIONES[2][3])
    return f"{pregunta} {app.DESCRIPCION}"


dash.register_page(__name__, path=app.ruta_vista(app.VISTA_INICIAL), path_template="/modelos/<vista>", name="Modelos",
                   title=_titulo, description=_descripcion)


def subnav(vista):
    """Las vistas de Modelos en sus grupos de píldoras; la actual en negro (y «vista actual» para lectores de pantalla)."""
    bloques = []
    for grupo, rotulo in app.GRUPOS:
        enlaces = []
        for v, n, p, g, _m, _i in app.VISTAS:
            if g != grupo:
                continue
            activa = v == vista
            hijos = [n] + ([html.Span(" (vista actual)", className="oculto")] if activa else [])
            enlaces.append(html.Li(dcc.Link(hijos, href=app.ruta_vista(v), title=p,
                                            className="subnav__enlace" + (" subnav__enlace--activo" if activa else ""))))
        bloques.append(html.Div([html.Span(rotulo, className="subnav__grupo"), html.Ul(enlaces, className="subnav__lista")],
                                className="subnav__bloque"))
    return html.Nav(bloques, className="subnav", **{"aria-label": "Vistas de Modelos"})


def aviso_benchmark():
    """El aviso que encabeza las vistas del grupo de la EBM: es un benchmark, y el modelo original es otro."""
    return ui.aviso(html.Div([
        html.P("La EBM viene de la librería interpret: fue el «modelo nuevo» del Entregable 3 y, tras la corrección del proyecto, "
               "queda como benchmark reciente e interpretable. Estas vistas conservan su análisis de entonces, frente a los siete "
               "modelos de clase. El modelo original del proyecto es BDL, programado desde cero."),
        html.Div(ui.boton("Ver el modelo original", href=app.ruta_vista("original"), icono="modelo", pequeno=True), className="botonera"),
    ]), None, "La EBM es un benchmark, no el modelo original")


def layout(vista=None, **consulta):
    if vista not in MODULOS:
        return html.Div([subnav(None), ui.pantalla([
            ui.encabezado("Esa vista no existe", "Elija una de las vistas de Modelos de arriba.", pantalla="inicio")])],
            className="modelos")
    pantalla = MODULOS[vista].layout(**consulta)
    if GRUPO.get(vista) == "ebm" and isinstance(getattr(pantalla, "children", None), list):
        pantalla.children.insert(1, aviso_benchmark())          # bajo la cabecera de la vista, con el espacio de una sección
    return html.Div([subnav(vista), pantalla], className="modelos")
