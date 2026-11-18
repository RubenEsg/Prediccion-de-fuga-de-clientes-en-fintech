"""Ficha del modelo · ¿Para qué sirve el modelo original y para qué no?

La ficha del modelo (model card, Mitchell et al., 2019) del modelo original, BDL: detalles (versión, fecha, tipo, autores,
licencia, artefactos y su huella), uso previsto, usuarios previstos, usos no recomendados, factores, métricas, datos de
evaluación y de entrenamiento, análisis por segmento, consideraciones éticas, limitaciones y riesgos y recomendaciones.
Mientras la corrección no haya dejado al original en la app, generar_fichas escribe la de la EBM y esta vista la muestra.

El contenido sale de generar_fichas.contenido_ficha(), la misma función que escribe MODEL_CARD.md (la que se descarga en
/ficha.md): la página y el archivo dicen lo mismo, con las cifras leídas de datos/, artifacts/ y CITATION.cff. Esta vista
solo lo dibuja con los componentes de ui.
"""
from dash import html

import app
import datos_app as D
import formato as F
import generar_fichas as G
import motor as M
import ui


VISTA = "ficha"
PRE = "fi"
PANTALLA = "modelo"
RIVAL = "xgboost"


def _bloque(b):
    """Un bloque de la ficha como componente de ui."""
    tipo = b[0]
    if tipo == "parrafo":
        return ui.parrafo(b[1])
    if tipo == "nota":
        return ui.nota(b[1])
    if tipo == "lista":
        return html.Ul([html.Li(t) for t in b[1]])
    if tipo == "pares":
        return ui.ficha(b[1])
    if tipo == "tabla":
        num = b[3] if len(b) > 3 else None
        ocultar = b[4] if len(b) > 4 else ()
        return ui.tabla(b[1], b[2], compacta=True, num=num, ocultar_movil=ocultar)
    return ui.aviso(f"Bloque desconocido en la ficha: {tipo}", "atencion")


def _tarjeta(t):
    return ui.tarjeta([_bloque(b) for b in t["bloques"]], titulo=t["titulo"], subtitulo=t.get("subtitulo"), fuentes=t.get("fuentes"))


def _grupo(g):
    """Un grupo de la ficha: las tarjetas anchas (tablas) a lo ancho y las demás en su rejilla."""
    anchas = [t for t in g["tarjetas"] if t.get("ancho")]
    resto = [t for t in g["tarjetas"] if not t.get("ancho")]
    hijos = []
    if resto:
        hijos.append(ui.rejilla([_tarjeta(t) for t in resto], g.get("columnas", "1-1")) if len(resto) > 1 else _tarjeta(resto[0]))
    hijos += [_tarjeta(t) for t in anchas]
    return ui.seccion(g["titulo"], hijos, subtitulo=g.get("subtitulo"))


def _respuesta():
    clf = D.lideres("clasificacion").set_index("modelo")
    meta = M.meta()
    limite = (f"No para decidir sobre personas: los datos son sintéticos y «riesgo alto» es el cuarto superior (percentil "
              f"{F.entero(meta['etiqueta']['q'] * 100)}) de un puntaje, no fuga observada.")
    if not G.hay_original():
        e = clf.loc["ebm"]
        return (f"Para ordenar a los clientes de COFINFAD por su riesgo y priorizar una campaña de capacidad limitada: contactando al "
                f"{F.pct(M.CAPACIDAD_INFORME, 0)} de mayor puntaje, la EBM alcanza al {F.pct(e['recall_op'])} de los clientes de riesgo "
                f"alto (AUC {ui.num(e['auc_prueba'], 4)}, como {D.en_frase(RIVAL)} con {ui.num(clf.loc[RIVAL, 'auc_prueba'], 4)}). "
                f"{limite} El modelo original, BDL, todavía no está en la app: esta es la ficha de la EBM.")
    e = clf.loc[D.ORIGINAL]
    corta = D.frase_empate_corta(D.ORIGINAL)                    # el corte de la campaña cae en un empate: lo dice junto a la cifra
    empate = f" Esa captura es la de {corta}." if corta else ""
    veredicto = "; ".join(f"en {D.NOMBRE_TAREA[t].lower()}, {D.frase_veredicto(t, detalle=False)}" for t in G.TAREAS)
    return (f"Para ordenar a los clientes de COFINFAD por su riesgo y priorizar una campaña de capacidad limitada, y para decir en qué "
            f"nivel del puntaje puede estar cada uno: contactando al {F.pct(M.CAPACIDAD_INFORME, 0)} de mayor puntaje, BDL alcanza al "
            f"{F.pct(e['recall_op'])} de los clientes de riesgo alto (AUC de prueba {ui.num(e['auc_prueba'], 4)}).{empate} {limite} Frente a "
            f"los benchmarks, con la regla de superioridad de la guía, su veredicto es: {veredicto}.")


def _cabecera():
    """El botón que descarga la ficha en Markdown y los datos de la versión, debajo del título."""
    cff = G.cita_cff()
    boton = ui.boton("Descargar la ficha (Markdown)", href="/ficha.md", tipo="primario", icono="descargar", id=f"{PRE}-descargar",
                     titulo="Descarga MODEL_CARD.md, la misma ficha en Markdown")
    boton.download = "MODEL_CARD.md"
    # La versión y su fecha son las del dashboard (CITATION.cff); los artefactos del modelo original tienen fecha propia
    chips = [ui.chip_valor("Versión del dashboard:", cff["version"]),
             ui.chip_valor("Fecha de esa versión:", ui.fecha_larga(cff["fecha"], con_dia=False))]
    original = G.indice().get("original") if G.hay_original() else None
    if original and original.get("generado"):
        chips.append(ui.chip_valor("Artefactos de BDL:", ui.fecha_larga(original["generado"], con_dia=False)))
    chips.append(ui.chip_valor("Licencia:", cff["licencia"]))
    return [html.Div([boton] + chips, className="botonera")]


def layout(**_):
    try:
        grupos = G.contenido_ficha()
    except Exception as error:  # noqa: BLE001  (la ficha no puede tumbar la página: lo dice en un aviso)
        return ui.pantalla([ui.encabezado(app.pregunta(VISTA), "No se pudo armar la ficha del modelo.", pantalla=PANTALLA),
                            ui.aviso(f"{type(error).__name__}: {error}", "critico", "La ficha no se pudo leer de los datos")])
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), _respuesta(), respuesta=_cabecera(), pantalla=PANTALLA),
        *[_grupo(g) for g in grupos],
        ui.aviso(["La ficha la genera generar_fichas.py con las cifras de datos/ y artifacts/; el archivo descargable (MODEL_CARD.md) y "
                  "esta página salen de la misma función. La declaración de los datos está en DATA_STATEMENT.md, junto a la app.",
                  G.REFERENCIAS_FORMATO], None, "De dónde sale esta ficha"),
    ])
