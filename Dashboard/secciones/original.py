"""Modelo original · ¿Qué es BDL, cómo aprende y supera o no a los benchmarks?

La subsección del modelo original que pide la guía del Tercer Entregable (§7.3.3), sobre el dash: la descripción y la
limitación que atiende, la formulación, el pseudocódigo (Algoritmo 1), el sustento teórico (cinco proposiciones, con la
prueba unitaria que verifica cada una y lo que se observa en el modelo final), la complejidad, la declaración de novedad
(qué no es nuevo, qué es propio de BDL, qué se reutiliza y qué no se afirma) con los métodos afines y la búsqueda de
literatura, la curva de convergencia, lo que aprendió el índice, el techo teórico del problema, la
tabla de pruebas de hipótesis con p crudos y ajustados por Holm y tamaños del efecto, el resumen de los estudios de
robustez y el veredicto explícito de superioridad o no superioridad en cada tarea, con las cuatro condiciones de la guía
(§6.4).

El modelo original es BDL (Boosting con Desplazamiento Latente), programado desde cero en src/original_model.py. El relato
es el del libro del proyecto (ProyectoFinal, capítulos 4 a 7) y las cifras salen de datos/original/, que arma
preparar_original.py con lo que haya en ModeloOriginal/resultados: cada bloque cuyo archivo todavía no existe se muestra
como pendiente y no rompe la vista (y un estudio de robustez cuyo archivo no se pueda leer lo dice en un aviso). El
veredicto se lee de veredicto.json (condiciones 1 a 3) y de condicion4.json (condición 4) con D.estado_veredicto, que
aplica la regla como el libro: «superior» solo con las cuatro condiciones, «pendiente» si cumple las tres primeras y la
cuarta no está completa, «no superior» en cualquier otro caso, y «provisional» mientras falten repeticiones de la
validación anidada. Nunca se escribe a mano, y si es «no superior» la vista dice qué condición falla, con sus valores, y
el análisis de la causa. Las constantes de abajo son los umbrales de la regla de la guía y del protocolo pre-registrado,
o de presentación; ninguna es un resultado del proyecto.

Comparación estadística y Contexto reutilizan de aquí las tarjetas y las frases del veredicto y de las pruebas
(_tarjeta_condiciones, _por_que, _causa, _en_prueba, _detalle_mejor, _tabla_benchmarks y _pendiente).
"""
import math
import re

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import app
import datos_app as D
import formato as F
import modelos_vivos as V
import motor as M
import ui
from src import original_model as BDL

VISTA = "original"
PRE = "og"
PANTALLA = "modelo"
O = D.ORIGINAL
TAREAS = ("clasificacion", "regresion")

# La regla de superioridad de la guía (§6.4), como quedó en el protocolo pre-registrado (sección 7)
CONDICIONES = 4
PRUEBAS = ("P1", "P2", "P3", "P4")
PRUEBAS_MINIMAS = 3               # pruebas de P1 a P4 que deben rechazar la igualdad con el mejor benchmark
PRUEBAS_EN_PRUEBA = ("P3", "P4")  # las dos que se hacen con el conjunto de prueba (P1 y P2 usan los pliegues externos)
UMBRAL_CLIFF = 0.147              # |δ de Cliff| desde el que el efecto no es despreciable
UMBRAL_COHEN = 0.2                # |d de Cohen| desde el que el efecto no es despreciable
CONJUNTOS_ADICIONALES = 4         # conjuntos de datos adicionales del protocolo
CONJUNTOS_MINIMOS = 2             # en cuántos debe ser mejor en la métrica media
PERMUTACIONES = 10_000            # permutaciones de la prueba P3 de regresión (protocolo, sección 6)
SEMILLAS_ESTABILIDAD = 10         # semillas con que se reajusta cada modelo final (protocolo, sección 4)
NIVEL = "0.90"                    # la cobertura nominal con que se resume la incertidumbre
N_PRINCIPALES = 3                 # variables que nombra la lectura del índice
PARTE_MINIMA = 0.01               # se dibuja la curva de una variable si reúne al menos esta parte del aporte
PASO_RESIDUO = 0.05               # paso de las marcas del eje del residuo
AIRE_ROTULO = 1.18                # aire sobre la barra más alta, para el rótulo con el peso de cada nivel
DECADAS = 1.0                     # el eje logarítmico del error lleva una marca por década
MUESTRA_TABLA = 25                # la tabla gemela de una curva larga lleva uno de cada tantos puntos
PASO_CALENTAMIENTO = 10           # y la del calentamiento, uno de cada diez árboles
TOLERANCIA_COBERTURA = 0.02       # una cobertura medida que se aparta más que esto de la nominal se señala
ERRORES_ESTANDAR = 2              # errores estándar de muestreo con que se compara una cobertura medida con la nominal
PARTE_ESTABLE = 0.05              # P4 no depende del desempate si rechaza en menos de esta parte de los sorteos (criterio del libro)
NIVEL_SORTEOS = 0.95              # la parte central de los sorteos que resumen los cuantiles de empates_campana.json
COTA_LOGISTICA = 4                # p(1 − p) ≤ 1/4: la cota de la segunda derivada de la pérdida logística (Proposición 2)
REPETICIONES = (42, 43, 44)       # semillas de las tres repeticiones de la validación anidada (protocolo, sección 4)
TRES = (O, "xgboost", "lightgbm")  # los modelos de los estudios de robustez: el original y los dos benchmarks de boosting
CONTAMINACIONES = ("ruido gaussiano en el objetivo", "atípicos en el objetivo", "ruido en los predictores numéricos")   # 06_robustez.py
D_RMSE_SUBGRUPO = 6               # decimales del RMSE por subgrupo: con cinco, dos modelos distintos se leen iguales
MINIMO_DECIMAL = 0.001            # un estadístico menor que esto se escribe en notación científica, como en el libro
BASE_CIENTIFICA = 10              # la base de esa notación
SUPERINDICE = {ord(str(i)): s for i, s in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")} | {ord("-"): "⁻"}
RAYA = F.RAYA
CARDINAL = {0: "ninguna", 1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve",
            10: "diez", 11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince"}

# Lo que cambia entre las dos tareas
TAREA = {
    "clasificacion": {"metrica": "AUC", "d": 4, "mayor": True, "prueba": "auc_prueba", "inf": "auc_bca_inf", "sup": "auc_bca_sup",
                      "hipotesis": "H2", "que": "clasificación"},
    "regresion": {"metrica": "RMSE", "d": 5, "mayor": False, "prueba": "rmse_prueba", "inf": "rmse_bca_inf", "sup": "rmse_bca_sup",
                  "hipotesis": "H1", "que": "regresión"},
}
OPCIONES_TAREA = [{"label": D.NOMBRE_TAREA[t], "value": t} for t in TAREAS]

# Referencias del libro del proyecto (ProyectoFinal/references.bib) que cita esta vista: autores y año, con la clave del libro
REF = {"dempster1977": ("Dempster, Laird y Rubin", 1977), "friedman2001": ("Friedman", 2001), "mclachlan2000": ("McLachlan y Peel", 2000),
       "vovk2005": ("Vovk, Gammerman y Shafer", 2005), "lei2018": ("Lei et al.", 2018), "schwarz1978": ("Schwarz", 1978),
       "desarbo1988": ("DeSarbo y Cron", 1988), "grun2008": ("Grün y Leisch", 2008), "heckman1984": ("Heckman y Singer", 1984),
       "aitkin1999": ("Aitkin", 1999), "jacobs1991": ("Jacobs et al.", 1991), "rugamer2023": ("Rügamer et al.", 2023),
       "xue2024": ("Xue, Yao y Xiang", 2024), "jobst2024": ("Jobst", 2024), "mayr2012": ("Mayr et al.", 2012),
       "duan2020": ("Duan et al.", 2020), "ali2026": ("Ali, Haggag y Fouad", 2026), "hepp2020": ("Hepp et al.", 2020),
       "lou2013": ("Lou et al.", 2013), "nori2019": ("Nori et al.", 2019), "guo2024": ("Guo y vanden Broucke", 2024),
       "sauzet2016": ("Sauzet y Kleine", 2016), "decaigny2018": ("De Caigny, Coussement y De Bock", 2018),
       "adegboye2026": ("Adegboye et al.", 2026), "cohen1983": ("Cohen", 1983), "royston2006": ("Royston, Altman y Sauerbrei", 2006),
       "bohning1988": ("Böhning y Lindsay", 1988), "nadeau2003": ("Nadeau y Bengio", 2003), "demsar2006": ("Demšar", 2006),
       "holm1979": ("Holm", 1979),
       # las que cita, además, el capítulo 4 del libro en sus secciones 4.4 y 4.5 (la Tabla 4.3 y la declaración de novedad)
       "hou2025": ("Hou, Li y Gao", 2025), "zhou2022": ("Zhou, Qian y Yang", 2022), "adam2022": ("Adam, Mayr y Kneib", 2022),
       "meek2002": ("Meek, Thiesson y Heckerman", 2002), "zheng2026": ("Zheng, Li y Peng", 2026), "marz2019": ("März", 2019),
       "hajjem2011": ("Hajjem, Bellavance y Larocque", 2011), "hajjem2014": ("Hajjem, Bellavance y Larocque", 2014),
       "sela2012": ("Sela y Simonoff", 2012), "sigrist2022": ("Sigrist", 2022), "pellagatti2021": ("Pellagatti et al.", 2021),
       "jamal2006": ("Jamal y Bucklin", 2006), "bachmann2021": ("Bachmann, Meierer y Näf", 2021), "dong2026": ("Dong et al.", 2026),
       "buginga2024": ("Buginga y de Souza e Silva", 2024), "su2025": ("Su y Qu", 2025),
       "wattanawongwan2023": ("Wattanawongwan et al.", 2023), "lemmens2006": ("Lemmens y Croux", 2006),
       "vakeel2022": ("Vakeel et al.", 2022)}

LIBRO = "Libro del proyecto (ProyectoFinal)"
LIBRO_MODELO = f"{LIBRO}: capítulo 5, «El modelo original»"
LIBRO_LITERATURA = f"{LIBRO}: capítulo 4, «Revisión de literatura y novedad»"
LIBRO_BUSQUEDA = f"{LIBRO}: secciones 4.1 a 4.3, protocolo de búsqueda, matriz de novedad y búsqueda por citación (Tablas 4.1 y 4.2)"
LIBRO_AFINES = f"{LIBRO}: sección 4.4, «Trabajos más cercanos y diferencias» (Tabla 4.3)"
LIBRO_NOVEDAD = f"{LIBRO}: sección 4.5, «Declaración de novedad»"
GUION_BUSQUEDA = "ModeloOriginal/experiments/00_busqueda_literatura.py (la búsqueda, que se puede repetir)"
LIBRO_VALIDACION = f"{LIBRO}: capítulo 6, «Protocolo y validación estadística»"
LIBRO_EMPATES = f"{LIBRO}: sección 6.9, «¿Podía algún modelo cumplir R3?» (Tabla 6.11)"
PROTOCOLO = "ModeloOriginal/experiments/protocolo.md (protocolo pre-registrado)"
CODIGO = "src/original_model.py"
PRUEBAS_UNITARIAS = "ModeloOriginal/tests/test_original_model.py"


def _fo(rel):
    return D.fuente_original(rel)


# --- Textos y números ----------------------------------------------------------------------------------------------------
def _finito(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _mayuscula(texto):
    return texto[:1].upper() + texto[1:]


def _cita(clave):
    """Cita en el texto: «Dempster, Laird y Rubin (1977)»."""
    autores, anio = REF[clave]
    return f"{autores} ({anio})"


def _cita_en(clave):
    """Cita dentro de un paréntesis: «Dempster, Laird y Rubin, 1977»."""
    autores, anio = REF[clave]
    return f"{autores}, {anio}"


def _p(p):
    """Un valor p en una celda cuyo encabezado ya dice que es un p: «0,245» o «< 0,001»; «—» si no hay."""
    if not _finito(p):
        return RAYA
    return F.p_valor(p).replace(F.FIJO, " ").replace("p = ", "").replace("p ", "").replace(" ", F.FIJO)


def _tex(x, d=0):
    """Un número en es-CO dentro de una fórmula de MathJax: 38{.}978 · 0{,}008 · -0{,}0132."""
    return F.num(x, d).replace(F.MENOS, "-").replace(".", "{.}").replace(",", "{,}")


def _decimales(v, tope=4):
    """Decimales con que se escribe un hiperparámetro: 0,098 → 3 · 73 → 0 (los reales, con `tope` cifras significativas)."""
    v = float(v)
    if v == int(v):
        return 0
    return max(0, tope - 1 - math.floor(math.log10(abs(v))))


def _valor(v):
    """Un hiperparámetro en es-CO: sí o no, texto, entero o real con cuatro cifras significativas."""
    if isinstance(v, bool):
        return "sí" if v else "no"
    if v is None:
        return "ninguno"
    if isinstance(v, str):
        return "automático (BIC)" if v == "auto" else v
    texto = ui.num(v, _decimales(v))
    return texto.rstrip("0").rstrip(",") if "," in texto else texto


def _sust(texto, **valores):
    """Pone cada valor en su «marca» de una plantilla (así las fórmulas se escriben sin duplicar las llaves de LaTeX)."""
    for clave, valor in valores.items():
        texto = texto.replace(f"«{clave}»", str(valor))
    return texto


def _pegar(texto):
    """Mete dentro de cada fórmula en línea la puntuación que la sigue y el paréntesis que la abre: sin esto, el navegador
    puede dejar «:» o «)» solos al comienzo de la línea siguiente."""
    texto = re.sub(r"(?<!\$)\$([,.;:)\]]+)", lambda m: r"\text{" + m.group(1) + "}$", texto)
    return re.sub(r"([(\[])\$(?!\$)", lambda m: r"$\text{" + m.group(1) + "}", texto)


def _md(texto, **valores):
    """Un párrafo con fórmulas en línea ($…$), en la medida de lectura del texto."""
    return dcc.Markdown(_pegar(_sust(texto, **valores)), mathjax=True, className="parrafo")


def _ec(tex, **valores):
    """Una ecuación en bloque, en su marco blanco con desplazamiento propio (no se sale de la tarjeta en el teléfono)."""
    return html.Div(dcc.Markdown("$$\n" + _sust(tex, **valores) + "\n$$", mathjax=True), className="tabla-marco")


def _li(texto, hijos=None, **valores):
    contenido = [dcc.Markdown(_pegar(_sust(texto, **valores)), mathjax=True)]
    if hijos:
        contenido.append(html.Ol(list(hijos)))
    return html.Li(contenido)


def _nombre(m):
    return D.NOMBRE_MODELO.get(m, m)


def _tras(preposicion, frase):
    """«a» o «de» delante de un nombre en frase, con su contracción: «al árbol de decisión», «del SVM lineal»."""
    if frase.startswith("el "):
        return {"a": "al", "de": "del"}[preposicion] + frase[2:]
    return f"{preposicion} {frase}"


# --- Lo que ya llegó de la corrección y lo que falta ---------------------------------------------------------------------
def _pendiente(que, archivo=None, cuando=None):
    """Aviso de un bloque cuyo resultado todavía no existe: qué falta, en qué archivo llegará y qué lo produce."""
    texto = [f"{que} todavía no {'están' if que.lower().startswith(('las ', 'los ')) else 'está'} entre los resultados de la corrección."]
    if archivo:
        texto.append(f"Aparecerá aquí cuando exista {_fo(archivo)}" + (f" ({cuando})." if cuando else ".")
                     + " La app lo muestra al volver a correr preparar_original.py y reiniciarla.")
    return ui.aviso(texto, "atencion", "Pendiente")


def _modelo(tarea):
    """El modelo final de BDL de una tarea por dentro (datos/original/modelo_<tarea>.json) o None."""
    return D.modelo_original(tarea)


def _con_mezcla(tarea):
    m = _modelo(tarea)
    return m if m is not None and m.get("K") is not None else None


def _tarea_inicial():
    """La tarea con que nace la vista: la regresión (el modelo aprende del puntaje continuo) si su modelo ya está."""
    return next((t for t in ("regresion", "clasificacion") if _modelo(t) is not None), "regresion")


def _techo():
    return D.original_json("techo.json")


def _lider(tarea, modelo=O):
    t = D.lideres(tarea)
    f = t[t["modelo"] == modelo]
    return None if f.empty else f.iloc[0]


def _estado_veredicto(tarea):
    """El veredicto de una tarea como lo fija el protocolo (D.estado_veredicto): la palabra, las condiciones 1 a 3, el estado
    de la cuarta, cuántas se cumplen de las evaluadas y si el resultado es provisional. None si veredicto.json no existe."""
    return D.estado_veredicto(tarea)


def _faltan_pruebas(tarea):
    """Las pruebas del protocolo que todavía no están en veredicto.json (P3 y P4 se calculan sobre la prueba)."""
    r = D.veredicto_original(tarea)
    return [] if r is None else [k for k in PRUEBAS if k not in r["frente_al_mejor"]]


def _repeticiones_hechas():
    """Cuántas repeticiones de la validación anidada entraron en el veredicto (la menor de las dos tareas), o None."""
    hechas = [len(D.veredicto_original(t).get("repeticiones") or []) for t in TAREAS if D.veredicto_original(t) is not None]
    return min(hechas) if hechas else None


# --- 0. La respuesta de la pantalla --------------------------------------------------------------------------------------
def _frase_veredicto(tarea):
    c = TAREA[tarea]
    e = _estado_veredicto(tarea)
    r = D.veredicto_original(tarea)
    mejor = r["mejor_benchmark"]
    medias = r["medias"]
    lado = "mejor" if r["condicion_1_media_mejor"] else "peor"
    cumple = (f"no cumple ninguna de las {CARDINAL[e['evaluadas']]} condiciones evaluadas" if e["cumplidas"] == 0
              else f"cumple {CARDINAL[e['cumplidas']]} de las {CARDINAL[e['evaluadas']]} condiciones evaluadas")
    base = (f"en {c['que']}, {c['metrica']} externo de {ui.num(medias[O], c['d'])} frente a {ui.num(medias[mejor], c['d'])} "
            f"{_tras('de', D.en_frase(mejor))}, el mejor benchmark ({lado} por {ui.num(abs(r['ventaja_sobre_el_mejor']), c['d'] + 1)}), "
            f"y {cumple}")
    if e["palabra"] == "pendiente":
        return base + ": falta completar la cuarta para dar el veredicto"
    return base + f": {e['palabra']}"


def _respuesta():
    que = (f"BDL ({D.NOMBRE_LARGO_ORIGINAL}) es un boosting de árboles propios que modela como una clase latente el escalón del "
           "puntaje que los benchmarks tratan como ruido.")
    if all(D.veredicto_original(t) is None for t in TAREAS):
        return que + " El veredicto de superioridad todavía no se puede dar: las pruebas de hipótesis no han llegado."
    frases = [_frase_veredicto(t) for t in TAREAS if D.veredicto_original(t) is not None]
    hechas = _repeticiones_hechas()
    provisional = ""
    if hechas is not None and hechas < len(REPETICIONES):
        provisional = (f" Es un resultado provisional: el protocolo fija el veredicto con {CARDINAL[len(REPETICIONES)]} repeticiones de "
                       f"la validación anidada y hay {CARDINAL[hechas]}.")
    return f"{que} Con la regla de superioridad de la guía: " + "; ".join(frases) + "." + provisional


def _aviso_pendientes():
    """Lo que la corrección todavía no ha producido, dicho arriba una sola vez."""
    faltan = []
    sin_prueba = sorted({k for t in TAREAS for k in _faltan_pruebas(t)})
    if all(D.veredicto_original(t) is None for t in TAREAS):
        faltan.append("las pruebas de hipótesis y las condiciones 1 a 3 del veredicto")
    elif sin_prueba:
        faltan.append(f"las pruebas {F.enumerar(sin_prueba)}, que se calculan sobre las predicciones en la prueba")
    hechas = _repeticiones_hechas()
    if hechas is not None and hechas < len(REPETICIONES):
        semillas = [str(s) for s in REPETICIONES[hechas:]]
        una = len(semillas) == 1
        faltan.append(f"{'la repetición' if una else 'las repeticiones'} de la validación anidada con "
                      f"{'la semilla' if una else 'las semillas'} {F.enumerar(semillas)} (el protocolo fija "
                      f"{CARDINAL[len(REPETICIONES)]} repeticiones)")
    c4 = [D.condicion4(t) or {} for t in TAREAS]
    if not any(c4):
        faltan.append("la cuarta condición del veredicto (conjuntos de datos adicionales)")
    elif not all(x.get("completo") for x in c4):
        faltan.append("parte de los conjuntos de datos adicionales (la cuarta condición del veredicto es provisional)")
    for rel, que in (("ablacion.csv", "la ablación"), ("semillas.csv", "la estabilidad por semilla"),
                     ("robustez/sensibilidad.csv", "la sensibilidad a los hiperparámetros"), ("robustez/ruido.csv", "el ruido"),
                     ("robustez/muestra.csv", "el tamaño de muestra"), ("robustez/ciudades.csv", "el cambio de distribución"),
                     ("incertidumbre_prueba.csv", "la tabla de incertidumbre en prueba"),
                     ("optimo.json", "el análisis del predictor óptimo"),
                     ("empates_campana.json", "el análisis de los empates de la campaña")):
        if not D.hay_original(rel):
            faltan.append(que)
    if not faltan:
        return None
    generado = D.estado_original().get("generado")
    cuando = f" Última actualización de los datos: {ui.fecha_larga(generado, con_dia=False)}." if generado else ""
    return ui.aviso([f"Todavía no han llegado: {F.enumerar(faltan)}.",
                     "Cada bloque pendiente lo dice en su lugar. El veredicto de abajo se calcula con lo que hay y cambia solo cuando "
                     f"cambian los archivos de la corrección.{cuando}"], "atencion", "La corrección sigue corriendo: hay resultados pendientes")


# --- 1. Qué es y qué limitación atiende ----------------------------------------------------------------------------------
def _grafico_niveles(tarea):
    """Histograma de los residuos de calibración del modelo final (puntaje menos índice): los niveles del escalón."""
    m = _con_mezcla(tarea)
    if m is None:
        return _pendiente(f"El modelo final de {TAREA[tarea]['que']}", f"modelo_{tarea}.json", "sale del artefacto de BDL")
    r = m["residuos_calibracion"]
    bordes, cuenta = np.asarray(r["bordes"], dtype=float), np.asarray(r["cuenta"], dtype=float)
    centros = (bordes[:-1] + bordes[1:]) / 2
    con = cuenta > 0
    c, pi, k = np.asarray(m["desplazamientos"], dtype=float), np.asarray(m["pesos"], dtype=float), int(m["K"])
    fig = go.Figure()
    textos = [f"Residuo cerca de {ui.con_signo(x, 3)}: <b>{F.entero(n)}</b> clientes de calibración" for x, n in zip(centros[con], cuenta[con])]
    fig.add_trace(go.Bar(x=centros[con], y=cuenta[con], **ui.traza("modelo", "barra", ancho=float(bordes[1] - bordes[0])),
                         **ui.hover_texto(textos)))
    medio = float(np.diff(np.sort(c)).min()) / 2 if k > 1 else float(bordes[-1] - bordes[0])
    for ck, pk in zip(c, pi):                   # sobre cada nivel, el peso de su clase latente
        cerca = np.abs(centros - ck) <= medio
        alto = float(cuenta[cerca].max()) if cerca.any() else 0.0
        fig.add_annotation(x=float(ck), y=alto, yanchor="bottom", yshift=ui.px("--e-1"), text=F.pct(pk, 0), showarrow=False,
                           font=ui.fuente_svg("tinta_2"))
    lo = math.floor(min(bordes[0], c.min()) / PASO_RESIDUO) * PASO_RESIDUO
    hi = math.ceil(max(bordes[-1], c.max()) / PASO_RESIDUO) * PASO_RESIDUO
    vals, marcas = ui.marcas(lo, hi, PASO_RESIDUO, signo=True)
    ui.unidad_eje(fig, "clientes de calibración")
    fig.update_layout(hovermode="closest", bargap=0)
    fig.update_xaxes(range=[lo, hi], tickvals=vals, ticktext=marcas)
    fig.update_yaxes(range=[0, float(cuenta.max()) * AIRE_ROTULO])
    sep = m["separacion_media"]
    te = _techo()
    titulo = (f"El puntaje menos el índice cae en {CARDINAL[k]} niveles separados por {ui.num(sep, 4)}" if sep is not None
              else "El puntaje menos el índice forma una sola campana: BDL no encuentra escalón")
    if te is not None and sep is not None:
        titulo += f": el escalón de {ui.num(te['delta'], 2)} por producto activo, que el modelo no recibe"
    sub = (f"Residuo de los {F.entero(r['n'])} clientes de calibración del modelo final de {TAREA[tarea]['que']} (no entraron en los "
           "árboles ni en la parada temprana). Sobre cada nivel, el peso que BDL le dio a esa clase latente.")
    # los niveles, del mayor desplazamiento al menor; al lado, la frecuencia real del número de productos activos
    reales = [] if te is None else [te["pesos"][x] for x in sorted(te["pesos"], key=int)]
    filas = []
    for i, j in enumerate(np.argsort(-c)):
        fila = [F.entero(i + 1), ui.con_signo(c[j], 4), F.pct(pi[j], 1)]
        if len(reales) == k:
            fila += [F.entero(i + 1), F.pct(reales[i], 1), ui.con_signo(100 * (pi[j] - reales[i]), 1)]
        filas.append(fila)
    encabezados = ["Clase latente", "Desplazamiento", "Peso aprendido"]
    if len(reales) == k:
        encabezados += ["Productos activos", "Frecuencia real", "Diferencia (puntos)"]
    nota = ("BDL no recibe el número de productos activos (se excluye por fuga de información): los niveles y sus pesos los aprende "
            "del puntaje. " + ("La tabla del botón «…» los compara con la frecuencia real de uno a "
                               f"{CARDINAL[k]} productos en el entrenamiento, que solo se usó para calcular el techo teórico."
                               if len(reales) == k else ""))
    etiqueta = f"{titulo}. " + "; ".join(f"nivel {f[0]}: desplazamiento {f[1]}, peso {f[2]}" for f in filas) + "."
    return ui.grafico(fig, titulo, sub, etiqueta, tabla=(encabezados, filas), nota=nota, id=f"{PRE}-niv-{tarea}",
                      fuentes=[_fo(f"modelo_{tarea}.json")] + ([_fo("techo.json")] if te is not None else []))


def _cifras_descripcion(tarea):
    te, m = _techo(), _con_mezcla(tarea)
    fichas = []
    if te is not None:
        niveles = len(te["pesos"])
        fichas.append(ui.cifra("Escalón del puntaje por producto activo", ui.num(te["delta"], 2), "de puntaje",
                               nota=f"Estimado con el entrenamiento. Hay {CARDINAL[niveles]} niveles; el más frecuente reúne al "
                                    f"{F.pct(max(te['pesos'].values()), 0)} de los clientes.", mini=True))
        fichas.append(ui.cifra("Desviación del escalón", ui.num(te["desviacion_del_escalon"], 4), "de puntaje",
                               variacion=f"Desviación del puntaje en la prueba: {ui.num(te['prueba']['desviacion_del_puntaje'], 4)}",
                               nota="Es la parte del puntaje que ningún modelo puede predecir sin la variable excluida.", mini=True))
    if m is not None:
        veces = None if te is None else te["desviacion_del_escalon"] / m["sigma_validacion"]
        fichas.append(ui.cifra("Dispersión dentro de cada nivel, según BDL", ui.num(m["sigma_validacion"], 5), "de puntaje",
                               variacion=None if veces is None else f"{F.entero(veces)} veces menor que la del escalón",
                               nota=f"Error dentro de clase en las {F.entero(m['filas']['validacion'])} filas de validación del modelo "
                                    f"final de {TAREA[tarea]['que']}.", mini=True))
    return ui.cifras(fichas, "cifras--3") if fichas else None


def _descripcion_tarea(tarea):
    return [x for x in (_cifras_descripcion(tarea), _grafico_niveles(tarea)) if x is not None]


def _seccion_descripcion(tarea):
    te = _techo()
    escalon = f"de {ui.num(te['delta'], 2)} " if te is not None else ""
    niveles = CARDINAL[len(te["pesos"])] if te is not None else "varios"
    titulo = f"BDL modela como una clase latente el escalón {escalon}del puntaje que los benchmarks tratan como ruido"
    intro = (f"El puntaje de fuga de COFINFAD es un índice que depende de los predictores menos un escalón {escalon}por cada producto "
             "activo del cliente. Esa variable (active_products) se excluye por fuga de información, así que para unos mismos "
             f"predictores el puntaje no se dispersa como una campana alrededor de su media: cae en uno de {niveles} niveles. Los "
             "regresores de referencia minimizan el error cuadrático, que trata ese escalón como ruido; los clasificadores solo ven la "
             f"etiqueta «riesgo alto» y desechan el puntaje continuo que la define ({_cita_en('cohen1983')}; {_cita_en('royston2006')}).")
    que = ui.tarjeta([
        ui.parrafo(f"BDL supone que el puntaje es un índice F(x), una suma de árboles pequeños sobre las variables discretizadas, más "
                   "el desplazamiento de una clase latente que no se observa, más un error pequeño. Aprende las dos cosas a la vez con un "
                   "algoritmo EM generalizado: en cada vuelta estima a qué clase pertenece cada cliente, le quita al puntaje el "
                   "desplazamiento esperado de su clase y da unos pasos de boosting contra ese objetivo sin escalón."),
        ui.parrafo("De la mezcla salen las dos salidas: la probabilidad de superar el umbral de riesgo alto (la clasificación) y, para "
                   "la regresión, conjuntos de predicción que son uniones de tramos cortos, uno por nivel, con cobertura garantizada."),
        ui.parrafo(f"Todo está programado desde cero en {CODIGO} con numpy y scipy: la discretización, el árbol de regresión, los "
                   "histogramas por bin, la mezcla, el EM y la salida conforme. De scikit-learn solo se usan las clases base, para que "
                   "el modelo entre en una tubería y en la validación cruzada. No se importa ningún modelo."),
    ], titulo="Qué es", subtitulo=f"{D.NOMBRE_LARGO_ORIGINAL}: un índice aprendido por boosting y un escalón aprendido por EM.",
        fuentes=[CODIGO, LIBRO_MODELO])
    ebm = ui.aviso([f"La EBM, que el Entregable 3 presentó como modelo nuevo, viene de la librería interpret: para la guía es un "
                    "import, no un modelo original. Aquí es un benchmark reciente, junto con LightGBM, y sus vistas siguen en el "
                    "grupo «La EBM, benchmark interpretable».",
                    "El modelo original es BDL: tiene una regla de aprendizaje y una regla de salida programadas por el equipo, cada una "
                    "con su derivación, su demostración y su prueba unitaria. La de aprendizaje es una variante, para un índice "
                    "compartido con desplazamientos latentes, de una familia ya publicada: el EM con paso de boosting."],
                   None, "Qué cambió respecto al Entregable 3")
    return ui.seccion(titulo, [ui.rejilla([que, ebm], "2-1"), html.Div(_descripcion_tarea(tarea), id=f"{PRE}-desc", className="rejilla")],
                      intro=intro, subtitulo="Descripción del modelo y de la limitación de los métodos existentes que atiende.",
                      fuentes=[_fo("techo.json"), CODIGO, LIBRO_MODELO])


# --- 2. Formulación ------------------------------------------------------------------------------------------------------
EC_MODELO = r"""\begin{aligned}
s &= F(x) + c_{z} + e \\
z &\in \{1, \dots, K\}, \qquad P(z = k) = \pi_k \\
e &\sim \mathcal{N}(0, \sigma^2)
\end{aligned}"""
EC_INDICE = r"F(x) = \beta_0 + \nu \sum_{t=1}^{T} h_t(x)"
EC_VERO = r"\ell(F, \theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k\, \varphi_\sigma\big(s_i - F(x_i) - c_k\big)"
EC_E = r"\gamma_{ik} = \frac{\pi_k\, \varphi_\sigma(r_i - c_k)}{\sum_{l} \pi_l\, \varphi_\sigma(r_i - c_l)}, \qquad r_i = s_i - F(x_i)"
EC_SIN_ESCALON = r"\tilde{s}_i = s_i - \sum_{k} \gamma_{ik}\, c_k"
EC_GANANCIA = r"\frac{G_L^2}{N_L + \lambda} + \frac{G_R^2}{N_R + \lambda} - \frac{G^2}{N + \lambda}"
EC_M = r"""\begin{aligned}
\pi_k &= \frac{1}{n}\sum_i \gamma_{ik} \\
c_k &= \frac{\sum_i \gamma_{ik}\, r_i}{\sum_i \gamma_{ik}} \\
\sigma^2 &= \frac{1}{n}\sum_i\sum_k \gamma_{ik}\,(r_i - c_k)^2
\end{aligned}"""
EC_MEDIA = r"\hat{s}(x) = F(x) + \sum_k \pi_k\, c_k"
EC_CONJUNTO = r"""\begin{aligned}
C_\alpha(x) &= \{\, s : f(s - F(x)) \ge t_\alpha \,\} \\
f(r) &= \sum_k \pi_k\, \varphi_{\sigma}(r - c_k)
\end{aligned}"""
EC_EXCEDENCIA = r"P(s > \tau \mid x) = \sum_{k=1}^{K} \pi_k\, \Phi\!\left(\frac{F(x) + c_k - \tau}{\sigma_r}\right)"


def _seccion_formulacion():
    modelo = ui.tarjeta([
        _ec(EC_MODELO),
        _md(r"$s$ es el puntaje y $x$ los predictores. $F$ es el **índice**: la parte del puntaje que explican los predictores. $z$ es "
            r"la **clase latente** (no se observa), independiente de $x$ y de $e$; $c_k$ es el **desplazamiento** de la clase $k$, "
            r"$\pi_k$ su peso y $\sigma$ la dispersión dentro de cada clase. Con $K = 1$ el modelo es el boosting de mínimos "
            r"cuadrados de «friedman».", friedman=_cita("friedman2001")),
        _ec(EC_INDICE),
        _md(r"El índice es una suma de $T$ árboles de regresión $h_t$ de profundidad $d$ sobre las variables discretizadas, cada uno "
            r"multiplicado por la tasa $\nu$. Con $d = 1$ es un modelo aditivo, $F(x) = \beta_0 + \sum_j f_j(x_j)$: una curva por "
            r"variable, que se puede graficar."),
        _ec(EC_VERO),
        _md(r"Los parámetros son $F$ y $\theta = (\pi, c, \sigma)$; $\varphi_\sigma$ es la densidad normal de media cero y desviación "
            r"$\sigma$. La suma dentro del logaritmo impide maximizar la verosimilitud directamente: el algoritmo EM («dempster») "
            r"la sustituye por una cota que sí se puede maximizar.", dempster=_cita_en("dempster1977")),
    ], titulo="El modelo", subtitulo="Un índice, un desplazamiento latente y un error pequeño.", fuentes=[LIBRO_MODELO, CODIGO])
    aprendizaje = ui.tarjeta([
        _md(r"**Paso E.** Con el índice y la mezcla actuales, la probabilidad de que el cliente $i$ pertenezca a la clase $k$:"),
        _ec(EC_E),
        _md(r"**Pasos de boosting.** Como $\sum_k \gamma_{ik} = 1$, maximizar la verosimilitud esperada en $F$ es minimizar "
            r"$\sum_i (\tilde{s}_i - F(x_i))^2$, con el **objetivo sin escalón**:"),
        _ec(EC_SIN_ESCALON),
        _md(r"No se resuelve del todo: se dan $B$ pasos de boosting. En cada uno se ajusta un árbol $h$ al residuo "
            r"$\tilde{s}_i - F(x_i)$ y se actualiza $F \leftarrow F + \nu h$. En cada nodo el árbol elige la variable y el corte "
            r"que maximizan la ganancia ($G$ y $N$ son la suma de residuos y el número de filas; $L$ y $R$, los dos hijos), y el "
            r"valor de una hoja es $G / (N + \lambda)$:"),
        _ec(EC_GANANCIA),
        _md(r"**Paso M.** Con el índice ya actualizado, las fórmulas cerradas de la mezcla:"),
        _ec(EC_M),
    ], titulo="La regla de aprendizaje", subtitulo="EM generalizado: el paso M sobre el índice es un paso de boosting.",
        fuentes=[LIBRO_MODELO, CODIGO])
    salida = ui.tarjeta([
        _md(r"**Regresión.** La media predicha suma al índice el desplazamiento medio:"),
        _ec(EC_MEDIA),
        _md(r"Una parte de las filas no entra en los árboles: con sus residuos $r_j = s_j - F(x_j)$ se arma la distribución predictiva "
            r"de un cliente nuevo, $F(x) + r_j$. El **conjunto de predicción** de cobertura $1 - \alpha$ usa la densidad de la mezcla "
            r"como medida de conformidad; $t_\alpha$ es el $\lfloor \alpha (m + 1) \rfloor$-ésimo menor valor de $f(r_j)$ entre los "
            r"$m$ residuos de calibración. Con escalones es una unión de tramos cortos, uno alrededor de cada nivel:"),
        _ec(EC_CONJUNTO),
        _md(r"**Clasificación.** La etiqueta es $y = \mathbf{1}[s > \tau]$ y la probabilidad de la clase positiva sale del modelo: "
            r"cada clase latente aporta su peso si su nivel, $F(x) + c_k$, queda por encima del umbral. $\sigma_r$ es la dispersión "
            r"robusta del error dentro de clase, medida en validación; $\tau$ no es un hiperparámetro: se deduce de los datos."),
        _ec(EC_EXCEDENCIA),
    ], titulo="La regla de salida", subtitulo="De la mezcla salen la probabilidad de excedencia y los conjuntos de predicción.",
        fuentes=[LIBRO_MODELO, CODIGO])
    supuestos = ui.aviso([
        "S1: el desplazamiento es aditivo y no depende de los predictores. S2: la clase latente es independiente de los predictores. "
        "S3: el error dentro de cada clase tiene la misma dispersión.",
        "Cuando no se cumplen, el modelo no deja de funcionar: el criterio BIC elige una sola clase y BDL se reduce a un boosting de "
        "mínimos cuadrados. Es lo que se espera en los dos conjuntos adicionales de control.",
    ], None, "Supuestos")
    return ui.seccion("La formulación: un índice aprendido por boosting más un desplazamiento latente", [
        ui.rejilla([modelo, aprendizaje], "1-1"),
        ui.rejilla([salida, supuestos], "2-1"),
    ], subtitulo="El modelo estadístico, la regla de aprendizaje y la regla de salida, como en el capítulo del modelo original del "
                 "libro del proyecto.")


# --- 3. Pseudocódigo -----------------------------------------------------------------------------------------------------
def _valores_algoritmo(tarea):
    """Los valores con que quedó el modelo final de una tarea: los que buscó la optimización y lo que resultó."""
    m = _modelo(tarea)
    if m is None:
        return _pendiente(f"El modelo final de {TAREA[tarea]['que']}", f"modelo_{tarea}.json", "sale del artefacto de BDL")
    h = m["hiperparametros"]
    pares = [("Tasa de aprendizaje, ν", _valor(h["tasa"])), ("Profundidad de cada árbol, d", _valor(h["profundidad"])),
             ("Bins por variable, M", _valor(h["max_bins"])), ("Mínimo de filas por hoja", _valor(h["min_hoja"])),
             ("Regularización de las hojas, λ", _valor(h["lambda_l2"])), ("Clases latentes", _valor(h["n_componentes"]))]
    if "usar_puntaje" in h:
        pares.append(("Aprende del puntaje continuo", _valor(bool(h["usar_puntaje"]))))
    pares += [("Vueltas máximas de EM, V", _valor(h["n_vueltas"])), ("Árboles por vuelta, B", _valor(h["arboles_por_vuelta"])),
              ("Árboles de calentamiento, T₀ (máximo)", _valor(h["rondas_calentamiento"])),
              ("Clases máximas", _valor(h["max_componentes"])), ("Paciencia, P", _valor(h["paciencia"]))]
    resultado = []
    if m.get("K") is not None:
        f = m["filas"]
        resultado = [("Filas de ajuste, validación y calibración", f"{F.entero(f['ajuste'])}, {F.entero(f['validacion'])} y {F.entero(f['calibracion'])}"),
                     ("Árboles de calentamiento usados", F.entero(m["n_calentamiento"])),
                     ("Vueltas de EM hasta la mejor en validación", F.entero(m["vueltas_usadas"])),
                     ("Árboles del modelo final", F.entero(m["n_arboles"])), ("Clases latentes elegidas, K", F.entero(m["K"])),
                     ("Números que guarda el modelo", F.entero(m["parametros"]))]
    else:
        resultado = [("Modo", "directo: aprendió de la etiqueta, sin mezcla"), ("Árboles del modelo final", F.entero(m["n_arboles"]))]
    buscados = len(m.get("hiperparametros_buscados") or {})
    return ui.rejilla([
        ui.tarjeta([ui.ficha(pares), ui.nota(f"Los {CARDINAL.get(buscados, F.entero(buscados))} primeros los eligió la optimización "
                                             "bayesiana dentro de la validación anidada; los demás quedaron en su valor por defecto.")],
                   titulo=f"Hiperparámetros del modelo final de {TAREA[tarea]['que']}", fuentes=[_fo(f"modelo_{tarea}.json")]),
        ui.tarjeta([ui.ficha(resultado)], titulo="Lo que resultó del ajuste", fuentes=[_fo(f"modelo_{tarea}.json")]),
    ], "1-1")


def _seccion_algoritmo(tarea):
    valores = dict(pc=_tex(BDL.PACIENCIA_CALENTAMIENTO), sep=_tex(BDL.SEPARACION_MINIMA), peso=_tex(100 * BDL.PESO_MINIMO),
                   barras=_tex(BDL.BINS_HISTOGRAMA))
    entradas = [
        _md(r"**Entradas:** los datos $(x_i, s_i)_{i=1}^{n}$; la tasa $\nu \in (0, 1]$; la profundidad $d$; los bins $M$; el mínimo de "
            r"filas por hoja; $\lambda \ge 0$; las vueltas máximas $V$; los árboles por vuelta $B$; los árboles de calentamiento "
            r"$T_0$; $K_{\max}$; la paciencia $P$."),
        _md(r"**Salidas:** el índice $F$, la mezcla $(\pi, c, \sigma)$ y los residuos de calibración."),
    ]
    pasos = html.Ol([
        _li(r"**Reparto.** Dividir las filas al azar en ajuste, validación y calibración."),
        _li(r"**Discretización.** Con las filas de ajuste, cortar cada variable en a lo sumo $M$ bins: un bin por valor si hay pocos "
            r"valores distintos; cuantiles si hay muchos."),
        _li(r"**Calentamiento ($K = 1$).** $F \gets \bar{s}$. Repetir hasta $T_0$ veces: ajustar un árbol a $s_i - F(x_i)$ y sumar "
            r"$\nu h$ a $F$. Cortar antes si el error de validación no baja en «pc» árboles, y volver al mejor.", **valores),
        _li(r"**Mezcla inicial.** Con el residuo $r_i$ de las filas de ajuste, para $K = 1, \dots, K_{\max}$: K-medias exacto en una "
            r"dimensión sobre el histograma de $r$ («barras» barras) y EM de la mezcla. Elegir el $K$ de menor BIC y conservar solo "
            r"los desplazamientos separados por al menos «sep» $\sigma$ y con al menos el «peso» % de las filas.", **valores),
        _li(r"**EM generalizado.** Para $v = 1, \dots, V$:", [
            _li(r"*Paso E:* calcular $\gamma_{ik}$ y el objetivo sin escalón $\tilde{s}_i$;"),
            _li(r"*Boosting:* $B$ veces, ajustar un árbol a $\tilde{s}_i - F(x_i)$ y sumar $\nu h$ a $F$;"),
            _li(r"*Paso M:* actualizar $\pi$, $c$ y $\sigma$;"),
            _li(r"*Parada:* medir en validación el error dentro de clase, $\big(\tfrac{1}{m} \sum_i \min_k (r_i - c_k)^2\big)^{1/2}$; "
                r"si no mejora en $P$ vueltas, detenerse."),
        ]),
        _li(r"**Selección.** Quedarse con los árboles y la mezcla de la mejor vuelta en validación."),
        _li(r"**Calibración.** Guardar los residuos $r_j = s_j - F(x_j)$ de las filas de calibración."),
    ])
    parada = _md(r"**Criterio de parada:** $P$ vueltas sin mejora en validación, o $V$ vueltas. **Sin puntaje:** si el clasificador no "
                 r"recibe el puntaje continuo, el mismo motor de árboles minimiza la pérdida logística con pasos de hoja que no "
                 r"pueden subirla (modo directo); es la variante «sin puntaje» de la ablación.")
    algoritmo = ui.tarjeta([html.Div(entradas, className="rejilla"), ui.desplegable("Pasos del algoritmo", [pasos], abierto=True, llano=True),
                            parada],
                           titulo="Algoritmo 1. Ajuste de BDL", subtitulo="El del libro del proyecto; las constantes son las del código.",
                           fuentes=[LIBRO_MODELO, CODIGO])
    diseno = ui.tabla(["Decisión de diseño", "Justificación"], [
        ["Un solo índice y desplazamientos constantes, no una regresión por clase",
         "Es lo que dice la evidencia: el salto entre niveles es constante. Con una regresión por clase habría tantas veces más "
         "parámetros como clases y cada una vería menos datos."],
        ["Calentamiento con una sola clase y parada temprana",
         "Los escalones solo se distinguen cuando el residuo es menor que su separación; cortar a tiempo evita que el sobreajuste "
         "los deforme."],
        ["K-medias exacto sobre el histograma para iniciar la mezcla",
         "Es determinista, su costo no depende del número de filas y no deja dos centros en el mismo escalón."],
        ["Número de clases por BIC con una regla de parsimonia",
         "El BIC premia partir un escalón en dos componentes superpuestos o dedicar uno a un puñado de atípicos; ninguno es una clase "
         "latente."],
        ["Varios árboles por vuelta de EM", "Las responsabilidades cambian despacio; recalcularlas tras cada árbol cuesta más que el árbol."],
        ["Dispersión robusta en la probabilidad de excedencia",
         "El índice se equivoca mucho en muy pocos clientes; con la raíz del error cuadrático medio esos pocos ensanchan la transición "
         "de todos."],
        ["Reparto interno en ajuste, validación y calibración",
         "Separa los tres usos de los datos: ajustar los árboles, decidir cuándo parar y calibrar la salida conforme."],
    ], compacta=True, num=[False, False])
    return ui.seccion("El algoritmo: calentamiento, mezcla inicial y un EM cuyo paso M es un paso de boosting", [
        algoritmo,
        html.Div(_valores_algoritmo(tarea), id=f"{PRE}-alg"),
        ui.tarjeta([diseno], titulo="Decisiones de diseño y su justificación", fuentes=[LIBRO_MODELO]),
    ], subtitulo="Pseudocódigo formal: entradas, salidas, pasos y criterio de parada, con los valores del modelo final.")


# --- 4. Sustento teórico y complejidad -----------------------------------------------------------------------------------
EC_FISHER = r"\frac{1}{\sigma^2 + \operatorname{Var}(c_z)} \;\le\; I_f \;\le\; \frac{1}{\sigma^2}"
EC_LOGISTICA = r"L(F) = \sum_i \big[\log(1 + e^{F_i}) - y_i F_i\big]"
EC_EXCEDENCIA_BAYES = r"P(s > \tau \mid x) = \sum_k \pi_k\, \Phi\!\left(\frac{F(x) + c_k - \tau}{\sigma}\right)"
EC_COSTO = r"""\begin{aligned}
T_{\text{ajuste}} &= O\big(T\,(n\,p\,d + 2^d\, p\, M) + V\, n\, K\big) \\
T_{\text{inferencia}} &= O(p \log M) \ \text{por cliente, con } d = 1
\end{aligned}"""


def _observado_p1():
    """Proposición 1 en los modelos finales: la verosimilitud guardada en cada vuelta de EM."""
    partes = []
    for t in TAREAS:
        m = _con_mezcla(t)
        if m is None:
            continue
        v = m["verificaciones"]
        estado = "nunca baja" if v["verosimilitud_no_decrece"] else "baja en alguna vuelta"
        partes.append(f"en el modelo final de {TAREA[t]['que']} {estado} en {F.entero(v['vueltas'])} vueltas de EM")
    return ("En lo guardado: " + F.enumerar(partes) + ".") if partes else None


def _observado_p3():
    """Proposición 3 en la prueba: cobertura medida del conjunto de predicción frente a la nominal."""
    c = V.conformal()["modelos"].get(O)
    if not c or "conjunto" not in next(iter(c.values())):
        return None
    niveles, n = sorted(c, key=float), int(D.resumen()["n_prueba"])
    partes = [f"{F.pct(c[k]['conjunto']['cobertura_prueba'], 1)} con nominal de {F.pct(float(k), 0)}" for k in niveles]
    # La garantía es en promedio sobre muestras: una cobertura medida se aparta de la nominal por azar (error estándar binomial)
    margen = {k: ERRORES_ESTANDAR * math.sqrt(float(k) * (1 - float(k)) / n) for k in niveles}
    fuera = [F.pct(float(k), 0) for k in niveles if abs(c[k]["conjunto"]["cobertura_prueba"] - float(k)) > margen[k]]
    lectura = "las coberturas medidas quedan dentro de ese margen" if not fuera else f"la del {F.enumerar(fuera)} queda fuera de ese margen"
    return (f"En los {F.entero(n)} clientes de prueba, el conjunto de predicción cubre " + F.enumerar(partes) + ". La garantía es en "
            f"promedio sobre muestras: con ese tamaño, {CARDINAL[ERRORES_ESTANDAR]} errores estándar de muestreo son de "
            f"{ui.num(100 * min(margen.values()), 1)} a {ui.num(100 * max(margen.values()), 1)} puntos porcentuales según el nivel, y "
            f"{lectura}.")


def _observado_p4():
    te, m = _techo(), _con_mezcla("regresion") or _con_mezcla("clasificacion")
    if te is None or m is None:
        return None
    razon = (te["desviacion_del_escalon"] / m["sigma_validacion"]) ** 2
    return (f"En COFINFAD la desviación del escalón es {ui.num(te['desviacion_del_escalon'], 4)} y la dispersión dentro de clase que "
            f"mide BDL, {ui.num(m['sigma_validacion'], 5)}: la varianza que un regresor de mínimos cuadrados trata como ruido es unas "
            f"{F.entero(razon)} veces la que queda dentro de cada nivel. Por eso BDL afina el índice aunque el RMSE apenas se mueva: "
            "el RMSE está dominado por el escalón, que ningún modelo puede predecir.")


def _tabla_mesetas():
    """Proposición 5: las mesetas de la probabilidad teórica y la frecuencia de riesgo alto observada en cada una (prueba)."""
    te = _techo()
    if te is None:
        return None
    p = te["prueba"]
    filas = [[ui.num(float(k), 4), F.pct(p["mesetas"][k], 1), ui.num(p["frecuencia_por_meseta"][k], 4)]
             for k in sorted(p["mesetas"], key=float)]
    return ui.tabla(["Probabilidad de la meseta", "Clientes de prueba", "Frecuencia de riesgo alto"], filas, compacta=True)


def _proposicion(numero, nombre, enunciado, idea, prueba, observado=None, extra=None, fuentes=None, alcance=None):
    """La tarjeta de una proposición: enunciado, idea de la demostración, lo que el libro precisa después (`alcance`), lo
    que se observa en los modelos finales y las pruebas unitarias. `prueba` es un nombre o varios; cada uno puede ir solo o
    como (nombre, qué comprueba)."""
    cuerpo = [enunciado, ui.desplegable("Idea de la demostración", [idea], llano=True)]
    if alcance is not None:
        cuerpo.append(alcance)
    if observado:
        cuerpo.append(ui.parrafo(observado))
    if extra is not None:
        cuerpo.append(extra)
    unitarias = [prueba] if isinstance(prueba, str) else list(prueba)    # en html.Code: un nombre largo parte donde haga falta
    partes = ["Prueba unitaria: " if len(unitarias) == 1 else "Pruebas unitarias: "]
    for i, unitaria in enumerate(unitarias):
        funcion, que = (unitaria, None) if isinstance(unitaria, str) else unitaria
        partes += ([" y "] if i else []) + [html.Code(funcion)] + ([f" ({que})"] if que else [])
    cuerpo.append(ui.nota(partes + ["."]))
    return ui.tarjeta(cuerpo, titulo=f"Proposición {numero}. {nombre}", fuentes=[LIBRO_MODELO, PRUEBAS_UNITARIAS] + (fuentes or []))


def _seccion_teoria():
    clas = _modelo("clasificacion")
    directo = clas is not None and clas.get("modo") == "directo"
    p2_obs = ("El clasificador final quedó en este modo." if directo else
              "El clasificador final no usa este modo (aprende del puntaje); lo usa la variante «sin puntaje» de la ablación.")
    te = _techo()
    p5_obs = None
    if te is not None:
        p5_obs = (f"La consecuencia práctica es un techo: dentro de una meseta todos los clientes tienen la misma probabilidad y ningún "
                  f"clasificador puede ordenarlos mejor que al azar. En la prueba el AUC máximo es {ui.num(te['prueba']['auc_maximo'], 4)}.")
    props = [
        _proposicion(1, "La verosimilitud no decrece",
                     _md(r"Sea $(F, \theta)$ el estado al empezar una vuelta del Algoritmo 1 y $(F', \theta')$ el estado al terminarla, "
                         r"con $\nu \in (0, 2)$ y $\lambda \ge 0$. Entonces $\ell(F', \theta') \ge \ell(F, \theta)$. Si además $\sigma$ "
                         r"tiene un piso positivo, la sucesión de verosimilitudes está acotada y converge."),
                     _md(r"Por la descomposición de EM, la mejora de la verosimilitud es al menos la mejora de la verosimilitud esperada "
                         r"$Q$ calculada con las $\gamma_{ik}$ del paso E; basta ver que ningún paso baja $Q$. Un paso de boosting con "
                         r"valor de hoja $G/(N + \lambda)$ cambia la suma de cuadrados de cada hoja en $-\nu (2 - \nu)$ por una "
                         r"cantidad que no es negativa. El paso M es el máximo de $Q$ en $\pi$, en $c$ y en $\sigma$. Una sucesión "
                         r"monótona y acotada converge. $\blacksquare$"),
                     ("test_verosimilitud_no_decrece", "test_verosimilitud_no_decrece_con_arboles_profundos"), _observado_p1(),
                     fuentes=[_fo(f"modelo_{t}.json") for t in TAREAS if _modelo(t) is not None]),
        _proposicion(2, "En modo directo, la pérdida logística no sube",
                     html.Div([_md(r"Sea $L$ la pérdida logística y $p_i = 1/(1 + e^{-F_i})$:"), _ec(EC_LOGISTICA),
                               _md(r"Si cada hoja $R_l$ da el paso $v_l = G_l / (n_l/«c» + \lambda)$, con "
                                   r"$G_l = \sum_{i \in R_l} (y_i - p_i)$, y $\nu \in (0, 2)$, entonces $L(F + \nu h) \le L(F)$.",
                                   c=_tex(COTA_LOGISTICA))], className="rejilla"),
                     _md(r"La segunda derivada de la pérdida logística es $p(1 - p) \le 1/«c»$, de modo que la pérdida queda por debajo "
                         r"de una cota cuadrática («bohning»). Con el paso de la hoja, el cambio de esa cota en cada hoja es "
                         r"$-\nu (1 - \nu/2)$ por una cantidad que no es negativa, así que la cota no sube. $\blacksquare$",
                         bohning=_cita_en("bohning1988"), c=_tex(COTA_LOGISTICA)),
                     "test_perdida_logistica_no_sube", p2_obs),
        _proposicion(3, "Cobertura de los conjuntos de predicción",
                     _md(r"Si las filas de calibración y el cliente nuevo son intercambiables, el conjunto de predicción cumple "
                         r"$P(s \in C_\alpha(x)) \ge 1 - \alpha$, sin suponer que el modelo sea correcto."),
                     _md(r"$F$ y $f$ se calculan con las filas de ajuste y de validación, así que son fijas respecto de las de "
                         r"calibración. Los valores $f(r_j)$ de las $m$ filas de calibración y el del cliente nuevo son intercambiables "
                         r"y el rango del nuevo entre los $m + 1$ es uniforme: queda fuera del conjunto con probabilidad a lo sumo "
                         r"$\alpha$ («vovk»; «lei»). $\blacksquare$", vovk=_cita_en("vovk2005"), lei=_cita_en("lei2018")),
                     "test_cobertura_conforme", _observado_p3(), fuentes=[D.fuente("conformal.json")]),
        _proposicion(4, "El error cuadrático desperdicia información cuando hay escalones",
                     html.Div([_md(r"Sea $r = c_z + e$ el error del modelo, con densidad $f$, e $I_f$ su información de Fisher para un "
                                   r"parámetro de localización. Entonces"), _ec(EC_FISHER),
                               _md(r"con igualdad a la izquierda solo si $K = 1$, e $I_f \to 1/\sigma^2$ cuando la separación entre "
                                   r"desplazamientos crece frente a $\sigma$.")], className="rejilla"),
                     _md(r"A la izquierda, la media muestral es un estimador insesgado de la localización con varianza "
                         r"$(\sigma^2 + \operatorname{Var}(c_z))/n$, que por Cramér-Rao es al menos $1/(n I_f)$. A la derecha, si además "
                         r"se observara $z$ la información sería $1/\sigma^2$, y observar menos no puede dar más. Con las clases "
                         r"separadas, la información que falta tiende a cero. $\blacksquare$"),
                     (("test_informacion_de_fisher_entre_las_cotas", "integra numéricamente la información de Fisher de mezclas con "
                       "distinta separación y comprueba las dos cotas"),
                      ("test_la_mezcla_mejora_al_boosting_de_minimos_cuadrados", "la consecuencia en datos sintéticos: con la mezcla el "
                       "índice se recupera con menos error que con el boosting de mínimos cuadrados")),
                     _observado_p4(), fuentes=[_fo("techo.json")]),
        _proposicion(5, "La probabilidad de excedencia es la regla de Bayes del modelo",
                     html.Div([_md(r"Si el modelo se cumple, la probabilidad de superar el umbral $\tau$ es"), _ec(EC_EXCEDENCIA_BAYES),
                               _md(r"Cuando $\sigma \to 0$ es una función escalonada de $F(x)$ que solo toma los valores "
                                   r"$\sum_k \pi_k$ sobre las clases con $F(x) + c_k > \tau$: las **mesetas**.")], className="rejilla"),
                     _md(r"Condicionando en la clase, la probabilidad de superar el umbral es $\Phi((F(x) + c_k - \tau)/\sigma)$, y se "
                         r"promedia con $\pi_k$ porque $z$ es independiente de $x$. $\blacksquare$"),
                     (("test_excedencia_coincide_con_la_frecuencia", "compara la probabilidad con la frecuencia observada en datos "
                       "nuevos"),),
                     p5_obs, _tabla_mesetas(), fuentes=[_fo("techo.json")],
                     alcance=_md(r"La regla implementada es esa misma expresión con los parámetros estimados y una sola sustitución: en "
                                 r"lugar de la $\sigma$ del paso M usa la dispersión robusta $\sigma_r$ medida en validación. Es, por "
                                 r"tanto, un estimador de la regla de Bayes y no la regla exacta: coincide con ella cuando el modelo se "
                                 r"cumple y $\sigma_r$ estima $\sigma$. La razón de preferir $\sigma_r$ está en la tabla de decisiones "
                                 r"de diseño.")),
    ]
    costo = D.costo_original("regresion", O) if D.costo_original("regresion", O) is not None else D.costo_original("clasificacion", O)
    medido = []
    if costo is not None and _finito(costo["inferencia_s"]):
        por = costo["inferencia_s"] / D.resumen()["n_prueba"] * 1e6
        medido.append(f"puntuar los {F.entero(D.resumen()['n_prueba'])} clientes de prueba tarda {ui.num(costo['inferencia_s'], 3)} s "
                      f"({ui.num(por, 1)} µs por cliente)")
    if costo is not None and _finito(costo["ajuste_s"]):
        medido.append(f"ajustar el modelo final, {ui.num(costo['ajuste_s'], 1)} s")
    if costo is not None and _finito(costo["tamano_mb"]):
        medido.append(f"el modelo serializado ocupa {ui.num(1024 * costo['tamano_mb'], 0)} KB")
    complejidad = ui.tarjeta([
        _ec(EC_COSTO),
        _md(r"Con $n$ filas, $p$ columnas, $M$ bins por columna, $T$ árboles de profundidad $d$, $V$ vueltas de EM y $K$ clases. El "
            r"histograma de la raíz de un árbol es un producto de la matriz indicadora (un uno por fila y columna) por el vector de "
            r"residuos; en cada nivel siguiente solo se calcula el del hijo con menos filas y el otro sale por resta. Con $d = 1$ "
            r"la inferencia no recorre árboles: busca el bin de cada variable y consulta su curva. La memoria es $O(n\,p)$."),
        ui.parrafo(("Medido en este equipo: " + F.enumerar(medido) + ".") if medido else
                   "Los tiempos medidos llegan con la estabilidad por semilla y la curva de aprendizaje."),
    ], titulo="Complejidad", subtitulo="Lineal en las filas y en las columnas; la inferencia consulta una tabla por variable.",
        fuentes=[LIBRO_MODELO, _fo("costo.csv")])
    return ui.seccion("Sustento teórico: cinco proposiciones, cada una con su demostración y su prueba unitaria", [
        ui.rejilla(props[:2], "1-1"), ui.rejilla(props[2:4], "1-1"), ui.rejilla([props[4], complejidad], "1-1"),
    ], subtitulo="Lo que se puede afirmar con precisión del algoritmo, la idea de cada demostración (completa en el libro del "
                 "proyecto) y lo que se observa en los modelos finales.",
        fuentes=[LIBRO_MODELO, PRUEBAS_UNITARIAS])


# --- 5. Declaración de novedad y diferencias con la literatura -----------------------------------------------------------
def _metodo(familia, trabajos):
    """Celda del método, como en la Tabla 4.3 del libro: la familia y, debajo y en gris, los trabajos que la representan."""
    return [html.Span(familia), html.Br(), html.Span(trabajos, className="tinta-3")]


def _tarjeta_afines():
    """La Tabla 4.3 del libro: las familias de métodos afines, de la más parecida a la menos."""
    hajjem = f"{_cita_en('hajjem2011')} y {REF['hajjem2014'][1]}"
    filas = [
        [_metodo("EM con un paso de boosting para modelos con variable latente",
                 f"EMTboost ({_cita_en('zhou2022')}); modelos de Markov con estados latentes ({_cita_en('adam2022')}); algoritmo "
                 f"Expectation-Boosting ({_cita_en('hou2025')})"),
         "Sustituyen el paso M del EM por pasos de boosting para estimar las funciones de cada componente de un modelo con exceso de "
         "ceros, de un modelo con estados latentes o de una mezcla",
         [html.Strong("La regla de aprendizaje"), ": un EM generalizado cuyo paso M es un paso de boosting"],
         "El modelo al que se aplica: un solo índice compartido por todas las clases, con desplazamientos y pesos constantes, de modo "
         "que el paso M es un único boosting contra el objetivo menos el desplazamiento esperado; y la regla de salida (excedencia y "
         "conjuntos conformes)",
         "Estiman una función por componente: cuando las clases comparten el efecto de los predictores, eso multiplica los parámetros "
         "y reparte los datos. Ninguno deriva la probabilidad de superar un umbral ni se ha aplicado a fuga"],
        [_metodo("Boosting de una densidad de mezcla sin EM",
                 f"{_cita('jobst2024')}; XGBoostLSS con densidades de mezcla ({_cita_en('marz2019')})"),
         "Boosting por gradiente de la verosimilitud de una mezcla cuyos parámetros (localización, escala y pesos de cada componente) "
         "dependen de los predictores",
         "Boosting sobre la verosimilitud de una mezcla",
         "Un índice y desplazamientos latentes constantes; EM en lugar de gradiente directo; árboles propios",
         "Un boosting por parámetro y por componente; no hay un efecto común de los predictores que las clases compartan"],
        [_metodo("Árboles y boosting con efectos aleatorios",
                 f"MERT y MERF ({hajjem}); RE-EM ({_cita_en('sela2012')}); GPBoost ({_cita_en('sigrist2022')}); aplicado a la deserción "
                 f"estudiantil por {_cita('pellagatti2021')}"),
         ["Alternan entre estimar el efecto aleatorio de cada grupo ", html.Strong("observado"),
          " y ajustar árboles a la respuesta sin ese efecto"],
         "Quitar un efecto de grupo de la respuesta antes de ajustar los árboles",
         "El grupo no se observa: es una clase latente con sus pesos, y lo que se resta es el desplazamiento esperado según las "
         "responsabilidades del paso E",
         "Necesitan saber a qué grupo pertenece cada fila; aquí la variable que define el grupo está excluida"],
        [_metodo("EM con aprendizaje automático en el paso M", _cita("xue2024")),
         "Mezcla semiparamétrica: cada componente tiene su función de regresión, ajustada con un método de aprendizaje automático",
         "EM con un aprendiz flexible en el paso M",
         "Un índice común y salida de excedencia y conforme",
         "Estima una función por clase; con desplazamientos constantes eso multiplica los parámetros sin necesidad"],
        [_metodo("Mezclas de regresiones por EM", f"{_cita('desarbo1988')}; FlexMix ({_cita_en('grun2008')})"),
         "Una regresión lineal (o un GLM) por clase latente",
         "El EM y la idea de clase latente",
         "Un solo índice no paramétrico y desplazamientos constantes, no una regresión por clase",
         "Cada clase estima su propia regresión con una fracción de los datos; aquí las clases comparten el mismo efecto de los predictores"],
        [_metodo("Heterogeneidad no observada con puntos de masa", f"{_cita('heckman1984')}; {_cita('aitkin1999')}"),
         "Intercepto aleatorio con distribución discreta en modelos lineales generalizados",
         "Es el mismo modelo estadístico del error",
         "El predictor es un boosting de árboles; la mezcla se usa para predecir (excedencia y conjuntos), no solo para corregir "
         "coeficientes",
         "Predictor lineal: no capta efectos no lineales; pensado para datos agrupados o de duración"],
        [_metodo("Clases latentes y abandono latente en fuga",
                 f"{_cita('jamal2006')}; {_cita('bachmann2021')}; {_cita('dong2026')}; agrupamiento de curvas de supervivencia de "
                 f"{_cita('buginga2024')}"),
         "Modelan la fuga con clases latentes o efectos aleatorios, cada clase con su modelo de riesgo",
         "La idea de que hay heterogeneidad no observada entre clientes",
         "Predictor no paramétrico por boosting; la clase latente solo desplaza el puntaje y no tiene parámetros propios",
         "Modelos paramétricos por clase, pensados para duraciones y para explicar; no usan boosting ni el puntaje continuo"],
        [_metodo("Mezcla de expertos",
                 f"{_cita('jacobs1991')}; con regresión distribucional, {_cita('rugamer2023')}; anclada en un boosting, {_cita('su2025')}"),
         "Una red de compuerta reparte cada caso entre expertos",
         "La mezcla",
         "No hay compuerta: la clase no depende de los predictores; árboles en lugar de redes",
         "Si la clase no depende de los predictores la compuerta solo añade varianza; además, las redes están excluidas por la guía"],
        [_metodo("Boosting distribucional",
                 f"gamboostLSS ({_cita_en('mayr2012')}); NGBoost ({_cita_en('duan2020')}); aplicado a fuga por {_cita('ali2026')}"),
         "Boosting de todos los parámetros de una distribución paramétrica",
         "La idea de predecir una distribución",
         "Ruido de mezcla en lugar de una familia unimodal",
         "Con una familia unimodal el escalón sigue siendo ruido"],
        [_metodo("Regresión distribucional con clases latentes", f"{_cita('hepp2020')}; {_cita('wattanawongwan2023')}"),
         "Dos o más clases, cada una con su GAMLSS, por EM",
         "Clase latente y modelo aditivo",
         "Varios desplazamientos que comparten un índice; boosting de árboles; predicción",
         "Una regresión completa por clase; objetivo distinto (límites de referencia, exposición crediticia)"],
        [_metodo("Modelos aditivos por boosting", f"GA²M y EBM ({_cita_en('lou2013')}; {_cita_en('nori2019')})"),
         "Boosting cíclico de un modelo aditivo con pérdida cuadrática o logística",
         "La forma aditiva del índice, con árboles de profundidad uno",
         "La regla de aprendizaje (EM) y la regla de salida",
         "Pérdida cuadrática o logística: trata el escalón como ruido y solo ve la etiqueta"],
        [_metodo("Predicción conforme en fuga", _cita("guo2024")),
         "Usa intervalos conformes para mejorar un clasificador de fuga",
         "La predicción conforme",
         "La medida de conformidad es la densidad de la mezcla: conjuntos que son uniones de tramos",
         "No modela el puntaje continuo"],
        [_metodo("Método distribucional para respuestas dicotomizadas", _cita("sauzet2016")),
         "Estima proporciones sobre un umbral a partir de la media y la desviación de la respuesta continua",
         "Usar la respuesta continua en lugar de la etiqueta",
         "Predicción por cliente con una mezcla aprendida, no una comparación de grupos bajo normalidad",
         "Supone una distribución paramétrica sin clases latentes"],
        [_metodo("Boosting e híbridos para fuga",
                 f"{_cita('lemmens2006')}; {_cita('decaigny2018')}; {_cita('vakeel2022')}; {_cita('adegboye2026')}"),
         "Boosting o combinaciones de árboles y regresión logística sobre la etiqueta; cuando hay una mezcla, agrupa clientes fuera del "
         "modelo predictivo",
         RAYA,
         "Aprende del puntaje y de su estructura latente",
         "Solo usan la etiqueta"],
    ]
    tabla = ui.tabla(["Método (familia)", "Qué hace", "Qué conserva BDL", "Qué es distinto en BDL", "Por qué no resuelve la limitación"],
                     filas, compacta=True, num=[False] * 5, ocultar_movil=(1, 2))
    nota = (f"La relación entre el EM y el boosting se señaló pronto ({_cita_en('meek2002')}): añadir componentes a una mezcla con un "
            "EM estructural se parece a un boosting. Los algoritmos de la primera fila van en la otra dirección y son el antecedente "
            "directo de BDL. El boosting de árboles también se ha usado para estimar modelos de cura de mezcla, que separan a los "
            f"individuos en dos clases latentes ({_cita_en('zheng2026')}).")
    return ui.tarjeta([tabla, ui.nota(nota)],
                      titulo=f"BDL frente a {CARDINAL.get(len(filas), F.entero(len(filas)))} familias de métodos afines, de la más "
                             "parecida a la menos: la primera, el EM con paso de boosting, es su antecedente directo",
                      subtitulo="Qué hace cada una, qué conserva BDL, qué es distinto en BDL y por qué el método no resuelve la "
                                "limitación identificada.",
                      fuentes=[LIBRO_AFINES])


def _consultas(lit):
    """Las consultas de la búsqueda de literatura, de la primera a la vigente: [{fecha, cadenas}]. Un literatura.json sin ese
    campo (el de una búsqueda hecha una sola vez) da la suya."""
    return lit.get("consultas") or [{"fecha": lit["fecha_de_consulta"],
                                     "cadenas": list(dict.fromkeys(f["cadena"] for f in lit["matriz"]))}]


def _rango(cadenas):
    """«C1 a C9»: de la primera a la última de una lista de cadenas de búsqueda (una sola, tal cual)."""
    return cadenas[0] if len(cadenas) == 1 else f"{cadenas[0]} a {cadenas[-1]}"


def _fecha_consulta(consulta):
    return ui.fecha_larga(consulta["fecha"], con_dia=False)


def _nota_consultas(lit):
    """Cuándo se hizo cada consulta, con qué cadenas y qué cambió entre fechas: lo que dice la cabecera de la Tabla 4.1 del
    libro, con las cifras de literatura.json."""
    consultas = _consultas(lit)
    primera, ultima = consultas[0], consultas[-1]
    bases = F.enumerar(lit["bases"])
    if len(consultas) == 1:
        return f"Consulta del {_fecha_consulta(ultima)} en {bases}."
    nuevas = [c for c in ultima["cadenas"] if c not in primera["cadenas"]]
    mas_citas = any(f.get("desde") not in (None, primera["fecha"]) for f in lit["citacion_adelante"] + lit["citacion_atras"])
    anadido = ([f"añadió las cadenas {_rango(nuevas)}"] if nuevas else []) + (["amplió la búsqueda por citación"] if mas_citas else [])
    texto = (f"La búsqueda se hizo {CARDINAL.get(len(consultas), F.entero(len(consultas)))} veces en {bases}. La primera consulta, del "
             f"{_fecha_consulta(primera)}, usó las cadenas {_rango(primera['cadenas'])}. La del {_fecha_consulta(ultima)} es la "
             "repetición que pide la guía antes de la entrega" + (f": {F.enumerar(anadido)}." if anadido else "."))
    comparacion = lit.get("comparacion")
    if comparacion:
        entre = "Entre las dos fechas" if len(consultas) == 2 else "De una consulta a las anteriores"
        texto += (f" {entre}, en las {F.entero(comparacion['celdas'])} combinaciones de base y cadena que se pueden comparar entraron "
                  f"{F.entero(comparacion['entradas_nuevas'])} resultados que no estaban entre los primeros "
                  f"{F.entero(lit['revisados_por_cadena'])}; se leyeron y no cambian ninguna observación.")
    return texto


def _notas_citacion(lit):
    """Lo que dice la búsqueda por citación, como la nota de la Tabla 4.2 del libro: cómo se cuenta hacia adelante, qué pasa
    con los antecedentes que entraron después de la primera consulta y qué hay entre las referencias de los más cercanos."""
    consultas = _consultas(lit)
    adelante = lit["citacion_adelante"]
    notas = ["Hacia adelante, la cuenta es automática (OpenAlex) y cada resultado se leyó: no todos tratan de fuga de clientes, porque "
             "la palabra «churn» también nombra la rotación de personal."]
    nuevos = [f for f in adelante if len(consultas) > 1 and f.get("desde") not in (None, consultas[0]["fecha"])]
    if nuevos:
        citan, mencionan = sum(f["citado_por"] for f in nuevos), sum(f["sobre_fuga"] for f in nuevos)
        cuales = (f"Los {_cardinal_m(len(nuevos))} antecedentes de las últimas filas" if adelante[-len(nuevos):] == nuevos
                  else f"{_mayuscula(_cardinal_m(len(nuevos)))} de los antecedentes")
        cuando = "en la segunda consulta" if len(consultas) == 2 else "después de la primera consulta"
        fuga = "ninguno menciona la fuga" if mencionan == 0 else f"{F.entero(mencionan)} mencionan la fuga"
        notas.append(f"{cuales} entraron en la búsqueda por citación {cuando}: los citan {F.entero(citan)} trabajos y {fuga}.")
    ambas = sum(f["mezcla_y_fuga"] for f in lit["citacion_atras"])
    notas.append(f"Hacia atrás, entre las referencias de los trabajos más cercanos, las que combinan mezclas y fuga son {F.entero(ambas)}.")
    return notas


def _tarjeta_busqueda():
    lit = D.original_json("literatura.json")
    if lit is None:
        return _pendiente("La matriz de novedad", "literatura.json", "sale de la búsqueda reproducible de literatura")
    matriz, bases, consultas = lit["matriz"], lit["bases"], _consultas(lit)
    cadenas = {}
    for f in matriz:
        cadenas.setdefault(f["cadena"], {"tema": f["tema"], "filas": []})["filas"].append(f)
    revisados = sum(f["revisados"] for f in matriz)

    def resultados(f):
        return F.entero(f["resultados"]) if f is not None and f["resultados"] is not None else RAYA

    filas = []
    for clave, c in cadenas.items():
        por_base = {f["base"]: f for f in c["filas"]}
        filas.append([f"{clave}. {c['tema']}"] + [resultados(por_base.get(b)) for b in bases]
                     + [F.entero(sum(f["revisados"] for f in c["filas"])), F.entero(sum(f["relevantes"] for f in c["filas"]))])
    tabla = ui.tabla(["Cadena de búsqueda"] + [f"Resultados en {b}" for b in bases] + ["Revisados", "Cercanos"], filas,
                     compacta=True, ocultar_movil=tuple(range(1, 1 + len(bases))))
    detalle = ui.tabla(["Base", "Cadena", "Resultados", "Revisados", "Cercanos", "Observación"],
                       [[f["base"], f["cadena"], resultados(f), F.entero(f["revisados"]), F.entero(f["relevantes"]), f["observacion"]]
                        for f in matriz], compacta=True, num=[False, False, True, True, True, False], alta=True, ocultar_movil=(2, 3))
    adelante = ui.tabla(["Antecedente", "Citado por", "De ellos, mencionan «churn»"],
                        [[f["antecedente"], F.entero(f["citado_por"]), F.entero(f["sobre_fuga"])] for f in lit["citacion_adelante"]],
                        compacta=True)
    atras = ui.tabla(["Trabajo cercano", "Referencias", "Sobre mezclas o clases latentes", "Sobre fuga", "Mezcla y fuga a la vez"],
                     [[f["trabajo"], F.entero(f["referencias"]), F.entero(f["sobre_mezclas"]), F.entero(f["sobre_fuga"]),
                       F.entero(f["mezcla_y_fuga"])] for f in lit["citacion_atras"]], compacta=True, ocultar_movil=(1,))
    tope = F.entero(lit["revisados_por_cadena"])
    veces = f", hecha {CARDINAL.get(len(consultas), F.entero(len(consultas)))} veces" if len(consultas) > 1 else ""
    titulo = (f"La búsqueda{veces}: {CARDINAL.get(len(bases), F.entero(len(bases)))} bases, "
              f"{CARDINAL.get(lit['cadenas'], F.entero(lit['cadenas']))} cadenas y {F.entero(revisados)} resultados revisados; ninguno "
              "aplica a la fuga de clientes un boosting con una mezcla latente ni otro algoritmo de la familia de EM con boosting")
    fechas = F.enumerar(f"del {_fecha_consulta(c)}" for c in consultas)
    lectura = (f"De cada base y cadena se leen el título y el resumen de los primeros {tope} resultados por relevancia (en Crossref "
               "solo el título, porque no da resumen). Los totales de Crossref no se comparan con los demás: su búsqueda cuenta "
               "cualquier registro con alguno de los términos. Cercanos (la columna «Relev.» de la Tabla 4.1 del libro): de los "
               "revisados, los trabajos que comparten con la propuesta la familia del modelo o el problema con un componente latente o "
               "probabilístico; son los que nombra la observación de cada base y cadena, y un mismo trabajo puede aparecer en más de "
               "una. Un trabajo cercano no es una coincidencia: ninguno aplica el modelo propuesto a la fuga de clientes.")
    cuerpo = [tabla, ui.nota(_nota_consultas(lit)), ui.nota(lectura)]
    if len(consultas) > 1 and any(c not in consultas[0]["cadenas"] for c in consultas[-1]["cadenas"]):
        cuerpo.append(ui.aviso(
            "La primera consulta no devolvió la familia de algoritmos más cercana, la del EM con paso de boosting, porque ninguna de "
            "sus cadenas la nombraba; apareció al ampliarlas. Es una advertencia sobre el alcance de cualquier búsqueda por palabras "
            f"clave, incluida esta: se leyeron los primeros {tope} resultados de cada base y cadena, no toda la literatura.",
            None, "El alcance de la búsqueda"))
    cuerpo += [ui.desplegable("Qué se encontró en cada base y cadena", [detalle], pequeno=True),
               ui.desplegable("Búsqueda por citación, hacia adelante y hacia atrás",
                              [adelante, atras] + [ui.nota(t) for t in _notas_citacion(lit)], pequeno=True)]
    return ui.tarjeta(cuerpo, titulo=titulo,
                      subtitulo=f"La novedad no se declara: se documenta. Consulta{'s' if len(consultas) > 1 else ''} {fechas}; la "
                                "búsqueda se puede repetir con el guion.",
                      fuentes=[_fo("literatura.json"), LIBRO_BUSQUEDA, GUION_BUSQUEDA])


def _declaracion():
    """La declaración de novedad del libro (sección 4.5), con el alcance de la búsqueda que diga literatura.json."""
    lit = D.original_json("literatura.json")
    alcance = "según la búsqueda descrita"
    if lit is not None:
        bases, cadenas = len(lit["bases"]), lit["cadenas"]
        alcance += (f" ({CARDINAL.get(bases, F.entero(bases))} bases de datos, {CARDINAL.get(cadenas, F.entero(cadenas))} cadenas, búsqueda "
                    "por citación en los dos sentidos" + (" y revisión de repositorios" if lit.get("repositorios") else "")
                    + ", en las fechas que registra la matriz)")
    return (f"Hasta donde sabemos, y {alcance}, no se encontraron estudios que apliquen a la predicción de fuga de clientes un modelo "
            "de boosting con una mezcla latente de desplazamientos ajustado por EM generalizado, ni otro algoritmo de la familia de EM "
            "con paso de boosting, ni que deriven de un modelo así la probabilidad de superar un umbral de riesgo. La novedad que se "
            "afirma es precisa y acotada.")


def _seccion_novedad():
    """La declaración de novedad con los cuatro bloques del libro (sección 4.5): qué no es nuevo, qué es propio de BDL, qué se
    reutiliza y qué no se afirma; después, los métodos afines (Tabla 4.3) y la búsqueda (Tablas 4.1 y 4.2)."""
    no_nuevo = ui.tarjeta([
        ui.parrafo("Sustituir el paso M del EM por un paso de boosting ya se había hecho: en mezclas con exceso de ceros "
                   f"({_cita_en('zhou2022')}), en modelos con estados latentes ({_cita_en('adam2022')}) y en mezclas para pérdidas de "
                   f"seguros, con el nombre de Expectation-Boosting ({_cita_en('hou2025')})."),
        ui.parrafo("BDL es un miembro de esa familia, no su origen."),
    ], titulo="Qué no es nuevo", fuentes=[LIBRO_NOVEDAD])
    propio = ui.tarjeta([html.Ul([
        html.Li("El modelo: un índice único aprendido por boosting de árboles, con desplazamientos latentes constantes y pesos que no "
                "dependen de los predictores. Es la contraparte con clase latente de los árboles con efectos aleatorios, y un caso "
                "mucho más restringido que una mezcla de regresiones."),
        html.Li("La forma que toma el paso M con ese modelo: un solo paso de boosting contra el objetivo al que se le resta el "
                "desplazamiento esperado (el objetivo sin escalón), en lugar de un boosting por componente."),
        html.Li("La regla de salida: la probabilidad de excedencia de la mezcla como clasificador de una respuesta continua "
                "dicotomizada."),
        html.Li("La densidad de la mezcla como medida de conformidad de los conjuntos de predicción."),
        html.Li("La implementación, programada desde cero, y su aplicación a la fuga de clientes."),
    ])], titulo="Qué es propio de BDL", fuentes=[LIBRO_NOVEDAD, CODIGO])
    reutiliza = ui.tarjeta([html.Ul([
        html.Li(f"El algoritmo EM ({_cita_en('dempster1977')})."),
        html.Li(f"El boosting de gradiente ({_cita_en('friedman2001')})."),
        html.Li(f"Las mezclas finitas ({_cita_en('mclachlan2000')})."),
        html.Li(f"La predicción conforme por partición ({_cita_en('vovk2005')}; {_cita_en('lei2018')})."),
        html.Li(f"El criterio BIC ({_cita_en('schwarz1978')})."),
    ]), ui.parrafo("Ninguna de esas piezas es nuestra.")], titulo="Qué se reutiliza", fuentes=[LIBRO_NOVEDAD])
    no_afirma = ui.tarjeta([html.Ul([
        html.Li("No se afirma que el modelo estadístico del error sea nuevo: es el de los interceptos aleatorios con puntos de masa."),
        html.Li("Tampoco se afirma que lo sea la alternancia entre EM y boosting."),
        html.Li("Ni que BDL sea mejor fuera de los problemas con un desplazamiento latente: sin él se reduce a un boosting de mínimos "
                "cuadrados."),
    ])], titulo="Qué no se afirma", fuentes=[LIBRO_NOVEDAD])
    cercanos = ui.aviso([
        f"Los antecedentes algorítmicos más cercanos son el algoritmo Expectation-Boosting de {_cita('hou2025')}, la regresión de "
        f"mezcla con boosting de {_cita('jobst2024')} y el EM con aprendizaje automático de {_cita('xue2024')}; la tabla de métodos "
        "afines precisa las diferencias.",
        "El mecanismo está formulado de manera general: sirve para una regresión en la que una variable discreta que no está entre los "
        "predictores desplaza la respuesta, y para clasificar una respuesta continua dicotomizada. Tiene condiciones: el "
        "desplazamiento debe ser aditivo, no depender de los predictores y ser grande frente al error del índice.",
    ], None, "Los antecedentes más cercanos y las condiciones del mecanismo")
    return ui.seccion("Qué es propio de BDL: el modelo al que se aplica, su regla de salida y su implementación; la regla de "
                      "aprendizaje pertenece a una familia ya publicada", [
        ui.rejilla([no_nuevo, propio], "1-2"), ui.rejilla([reutiliza, no_afirma], "1-1"), cercanos, _tarjeta_afines(),
        _tarjeta_busqueda(),
    ], subtitulo="Diferencias respecto a la literatura: la declaración de novedad, los métodos afines y la matriz de la búsqueda.",
        intro=_declaracion())


# --- 6. Convergencia -----------------------------------------------------------------------------------------------------
def _eje_log(fig, valores):
    """Eje y logarítmico con una marca por década, escrita en es-CO."""
    lo, hi = math.floor(math.log10(min(valores)) / DECADAS) * DECADAS, math.ceil(math.log10(max(valores)) / DECADAS) * DECADAS
    exps = np.arange(lo, hi + DECADAS / 2, DECADAS)
    fig.update_yaxes(type="log", range=[lo, hi], tickvals=[10 ** e for e in exps],
                     ticktext=[ui.num(10 ** e, max(0, int(-e))) for e in exps])


def _grafico_verosimilitud(tarea, m):
    y = np.asarray(m["historial"]["verosimilitud_por_fila"], dtype=float)
    x = list(range(1, len(y) + 1))
    fig = go.Figure(go.Scatter(x=x, y=y, **ui.traza("modelo", "linea"),
                               **ui.hover_texto([f"Vuelta {F.entero(k)}: <b>{ui.num(v, 3)}</b> por cliente" for k, v in zip(x, y)])))
    paso = ui.paso_redondo(y.max() - y.min(), 4)
    lo, hi = math.floor(y.min() / paso) * paso, math.ceil(y.max() / paso) * paso
    vals, marcas = ui.marcas(lo, hi, paso)
    vx, mx = ui.marcas(0, len(y), ui.paso_redondo(len(y), 4))
    ui.unidad_eje(fig, "log-verosimilitud por cliente de ajuste")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[0, len(y) + 1], tickvals=vx, ticktext=mx)
    fig.update_yaxes(range=[lo, hi], tickvals=vals, ticktext=marcas)
    sube = m["verificaciones"]["verosimilitud_no_decrece"]
    titulo = (f"La verosimilitud sube en cada una de las {F.entero(len(y))} vueltas de EM: de {ui.num(y[0], 2)} a {ui.num(y[-1], 2)} por "
              "cliente" if sube else
              f"La verosimilitud pasa de {ui.num(y[0], 2)} a {ui.num(y[-1], 2)} por cliente en {F.entero(len(y))} vueltas, con algún retroceso")
    sub = (f"Log-verosimilitud observada al empezar cada vuelta, dividida entre las {F.entero(m['filas']['ajuste'])} filas de ajuste del "
           f"modelo final de {TAREA[tarea]['que']}. Es lo que garantiza la Proposición 1.")
    nota = (f"El modelo final se queda con la mejor vuelta en validación (la {F.entero(m['vueltas_usadas'])}): las demás se descartan."
            if m["vueltas_usadas"] < len(y) else "El ajuste llegó al máximo de vueltas sin agotar la paciencia.")
    filas = [[F.entero(k), ui.num(y[k - 1], 4)] for k in x if k == 1 or k % MUESTRA_TABLA == 0 or k == len(y)]
    etiqueta = f"{titulo}."
    return ui.grafico(fig, titulo, sub, etiqueta, tabla=(["Vuelta de EM", "Log-verosimilitud por cliente"], filas), nota=nota,
                      id=f"{PRE}-vero-{tarea}", fuentes=[_fo(f"modelo_{tarea}.json")])


def _grafico_error(tarea, m):
    h = m["historial"]
    cal, em = np.asarray(h["error_calentamiento"], dtype=float), np.asarray(h["error_validacion"], dtype=float)
    x_cal = list(range(1, len(cal) + 1))
    x_em = [m["n_calentamiento"] + (k + 1) * m["arboles_por_vuelta"] for k in range(len(em))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_cal, y=cal, **ui.traza("contexto", "gruesa"),
                             **ui.hover_texto([f"Calentamiento, {F.entero(k)} árboles: RMSE <b>{ui.num(v, 5)}</b>" for k, v in zip(x_cal, cal)])))
    fig.add_trace(go.Scatter(x=x_em, y=em, **ui.traza("modelo", "linea"),
                             **ui.hover_texto([f"EM, {F.entero(k)} árboles: error dentro de clase <b>{ui.num(v, 5)}</b>" for k, v in zip(x_em, em)])))
    ui.referencia(fig, x=m["n_arboles"])        # sin rótulo: a 390 px pisa la unidad del eje; lo nombra la leyenda
    _eje_log(fig, list(cal) + list(em))
    tope = max(x_cal[-1] if x_cal else 0, x_em[-1] if x_em else 0)
    vx, mx = ui.marcas(0, tope, ui.paso_redondo(tope, 4))
    ui.unidad_eje(fig, "error en validación (escala logarítmica)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[0, tope * 1.02], tickvals=vx, ticktext=mx)
    final = float(m["sigma_validacion"])
    antes = float(cal[min(m["n_calentamiento"], len(cal)) - 1]) if len(cal) and m["n_calentamiento"] else float("nan")
    veces = antes / final if _finito(antes) and final > 0 else None
    titulo = (f"Al separar el escalón, el error de validación baja de {ui.num(antes, 4)} a {ui.num(final, 5)}"
              + (f": {F.entero(veces)} veces menos" if veces else ""))
    sub = (f"Error en las {F.entero(m['filas']['validacion'])} filas de validación según los árboles sumados al índice. En gris, el "
           f"calentamiento (RMSE del boosting de mínimos cuadrados, con el escalón dentro); en azul, el EM (raíz del error cuadrático "
           "medio respecto del desplazamiento más cercano).")
    nota = (f"El calentamiento usó {F.entero(m['n_calentamiento'])} de sus {F.entero(m['hiperparametros']['rondas_calentamiento'])} "
            f"árboles; el modelo final tiene {F.entero(m['n_arboles'])}. Las dos curvas miden cosas distintas: la primera incluye el "
            "escalón, que no se puede predecir; la segunda, solo el error del índice.")
    filas = ([[f"Calentamiento, {F.entero(k)} árboles", ui.num(v, 5)] for k, v in zip(x_cal, cal) if k == 1 or k % PASO_CALENTAMIENTO == 0 or k == len(cal)]
             + [[f"EM, {F.entero(k)} árboles", ui.num(v, 5)] for i, (k, v) in enumerate(zip(x_em, em))
                if i == 0 or (i + 1) % MUESTRA_TABLA == 0 or i == len(em) - 1])
    return ui.grafico(fig, titulo, sub, f"{titulo}.", tabla=(["Etapa", "Error en validación"], filas), nota=nota,
                      id=f"{PRE}-err-{tarea}", fuentes=[_fo(f"modelo_{tarea}.json")],
                      leyenda=[("EM: error dentro de clase", "modelo", "linea"), ("Calentamiento: RMSE", "contexto", "gruesa"),
                               ("Árboles del modelo final", "referencia", "linea")])


def _tarjeta_bic(tarea, m):
    bic = m["bic"]
    if not bic:
        return ui.tarjeta([ui.parrafo(f"El número de clases se fijó de antemano en {F.entero(m['K'])}: no hubo selección por BIC.")],
                          titulo="Número de clases latentes", fuentes=[_fo(f"modelo_{tarea}.json")])
    mejor = min(range(len(bic)), key=lambda i: bic[i]["bic"])
    filas = [[F.entero(b["K"]), F.entero(b["bic"]), F.entero(b["utiles"])] for b in bic]
    tabla = ui.tabla(["Clases candidatas, K", "BIC", "Clases que quedan tras la regla de parsimonia"], filas, mejor=[(mejor, 1)],
                     mejor_texto="menor", compacta=True)
    pesos = ", ".join(F.pct(p, 1) for p in sorted(m["pesos"], reverse=True))
    return ui.tarjeta([
        tabla,
        ui.nota(f"BIC de la mezcla ajustada al residuo del calentamiento. Del K de menor BIC solo se conservan los escalones separados "
                f"por al menos {ui.num(BDL.SEPARACION_MINIMA, 0)} desviaciones y con al menos el {F.pct(BDL.PESO_MINIMO, 0)} de las "
                f"filas. Modelo final: K = {F.entero(m['K'])}; pesos {pesos}; dispersión dentro de clase en validación "
                f"{ui.num(m['sigma_validacion'], 5)} (robusta: {ui.num(m['sigma_robusta'], 5)})."),
    ], titulo=f"El BIC y la regla de parsimonia dejan {CARDINAL.get(m['K'], F.entero(m['K']))} clases latentes"
              + (": con más candidatas no queda ninguna más" if max(b["utiles"] for b in bic) == m["K"] else ""),
        fuentes=[_fo(f"modelo_{tarea}.json"), CODIGO])


def _convergencia_tarea(tarea):
    m = _modelo(tarea)
    if m is None:
        return [_pendiente(f"El modelo final de {TAREA[tarea]['que']}", f"modelo_{tarea}.json", "sale del artefacto de BDL")]
    if m.get("K") is None:
        return [ui.aviso(f"El clasificador final aprendió de la etiqueta (modo directo): no hay mezcla ni vueltas de EM. Su pérdida "
                         f"logística de ajuste {'no sube' if m['verificaciones']['perdida_no_sube'] else 'sube'} en ninguna de sus "
                         f"{F.entero(len(m['historial']['perdida']))} rondas (Proposición 2).", None, "Modo directo")]
    return [ui.rejilla([_grafico_verosimilitud(tarea, m), _grafico_error(tarea, m)], "1-1"), _tarjeta_bic(tarea, m)]


def _seccion_convergencia(tarea):
    return ui.seccion("La curva de convergencia: la verosimilitud nunca baja y el error cae al separar el escalón",
                      [html.Div(_convergencia_tarea(tarea), id=f"{PRE}-conv", className="rejilla")],
                      subtitulo="El historial que guarda el modelo final (historial_): la verosimilitud en cada vuelta de EM, el error "
                                "de validación y la elección del número de clases.")


# --- 7. Lo que aprendió el índice ----------------------------------------------------------------------------------------
def _grafico_forma(tarea, f, k):
    cortes, valores = np.asarray(f["cortes"], dtype=float), np.asarray(f["valores"], dtype=float)
    bordes = np.concatenate(([f["minimo"]], cortes, [f["maximo"]]))
    x = np.repeat(bordes, 2)[1:-1]
    y = np.repeat(valores, 2)
    d = 0 if f["maximo"] - f["minimo"] >= 10 else 2
    unidad = f" {f['unidad']}" if f.get("unidad") else ""
    textos = [f"{f['nombre']} = {ui.num(a, d)}{unidad}: <b>{ui.con_signo(v, 4)}</b> al puntaje" for a, v in zip(x, y)]
    fig = go.Figure(go.Scatter(x=x, y=y, **ui.traza("modelo", "linea"), **ui.hover_texto(textos)))
    ui.linea_cero(fig, "y")
    paso = ui.paso_redondo(max(abs(valores.min()), abs(valores.max())), 3)
    lo, hi = math.floor(valores.min() / paso) * paso, math.ceil(valores.max() / paso) * paso
    vals, marcas = ui.marcas(lo, hi, paso, signo=True)
    ui.unidad_eje(fig, "lo que suma o resta al puntaje")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(range=[lo - paso / 5, hi + paso / 5], tickvals=vals, ticktext=marcas)
    ui.formato_eje(fig, "x", d)
    titulo = (f"{f['nombre']}: mueve el puntaje entre {ui.con_signo(valores.min(), 3)} y {ui.con_signo(valores.max(), 3)}")
    sub = f"Curva del índice para esta variable ({F.entero(len(valores))} tramos), centrada en su media en las filas de ajuste."
    paso_t = max(1, len(valores) // MUESTRA_TABLA)
    filas = [[f"de {ui.num(bordes[i], d)} a {ui.num(bordes[i + 1], d)}{unidad}", ui.con_signo(valores[i], 4), F.entero(f["filas"][i])]
             for i in range(0, len(valores), paso_t)]
    return ui.grafico(fig, titulo, sub, f"{titulo}.", tabla=(["Tramo", "Aporte al puntaje", "Filas de ajuste"], filas), alto="bajo",
                      id=f"{PRE}-forma-{tarea}-{k}", fuentes=[_fo(f"modelo_{tarea}.json")])


def _mismas_que_el_benchmark(tarea, variables, referencia="xgboost"):
    """«: las mismas tres que en XGBoost» si las variables principales de BDL son también las del benchmark de referencia en
    el resumen SHAP global (datos/shap/indice.json); si no, nada."""
    otro = V.shap_indice()["modelos"].get(f"{tarea}__{referencia}")
    if not otro or set(list(otro["importancia_media_abs"])[:len(variables)]) != set(variables):
        return ""
    return f": las mismas {CARDINAL.get(len(variables), F.entero(len(variables)))} que en {D.NOMBRE_MODELO[referencia]}"


def _indice_tarea(tarea):
    m = _con_mezcla(tarea)
    if m is None:
        return [_pendiente(f"El modelo final de {TAREA[tarea]['que']}", f"modelo_{tarea}.json", "sale del artefacto de BDL")]
    imp = m["importancia"]
    principales = imp[:N_PRINCIPALES]
    barras = html.Div([ui.barra_rayas(v["nombre"], F.pct(v["parte"], 1), v["parte"]) for v in imp[:N_PRINCIPALES + 1]], className="rayas-lista")
    parte = sum(v["parte"] for v in principales)
    lectura = ui.tarjeta([
        barras,
        ui.nota(f"Parte del aporte medio absoluto al {'índice' if tarea == 'clasificacion' else 'pronóstico'} en la muestra fija de "
                f"{F.entero(V.shap_indice()['filas_muestra'])} clientes de prueba (los mismos de «Interpretabilidad»). Los aportes son "
                "exactos: el punto de partida más los aportes de las variables da la salida del modelo."),
        html.Div(ui.boton("Ver la interpretabilidad de los modelos", href=app.ruta_vista("interpretabilidad") + f"?tarea={tarea}&modelo={O}",
                          icono="lupa", pequeno=True), className="botonera"),
    ], titulo=(f"{_mayuscula(F.enumerar(v['nombre'].lower() for v in principales))} reúnen el {F.pct(parte, 1)} del aporte"
               + _mismas_que_el_benchmark(tarea, [v["variable"] for v in principales])),
        fuentes=[D.fuente(f"shap/{tarea}__{O}.parquet"), D.fuente("shap/indice.json"), _fo(f"modelo_{tarea}.json")])
    if not m["formas"]:
        return [lectura, ui.aviso("El modelo final usa árboles de más de un corte: el índice no es aditivo y no hay una curva por "
                                  "variable. Los aportes se reparten árbol por árbol con valores de Shapley exactos.", None,
                                  "Sin curvas por variable")]
    pesa = {v["variable"] for v in imp if v["parte"] >= PARTE_MINIMA}
    formas = [_grafico_forma(tarea, f, k) for k, f in enumerate(m["formas"]) if f["variable"] in pesa]
    if not formas:
        return [lectura]
    return [lectura, ui.rejilla(formas, "3" if len(formas) == 3 else "1-1")]


def _seccion_indice(tarea):
    return ui.seccion("Lo que aprendió: con árboles de un corte, el índice se lee variable por variable",
                      [html.Div(_indice_tarea(tarea), id=f"{PRE}-indice", className="rejilla")],
                      subtitulo="Importancia y curvas (formas_) de las variables principales del modelo final. Como en la EBM, cada "
                                "aporte se lee en su curva; a diferencia de ella, el escalón queda aparte, en la mezcla.")


# --- 8. El techo teórico -------------------------------------------------------------------------------------------------
def _fila_techo(tarea, nombre, col, clave, d, menor_es_mejor):
    """Una fila de la tabla del techo: el mejor benchmark, el original, el límite y la parte de la distancia que cierra."""
    te, t = _techo(), D.lideres(tarea).set_index("modelo")
    r = D.veredicto_original(tarea)
    if O not in t.index or col not in t.columns or not _finito(t.loc[O, col]):
        return None
    limite, orig = te["prueba"][clave], float(t.loc[O, col])
    otros = t.drop(index=O)
    mejor = r["mejor_benchmark"] if r is not None else (otros[col].idxmin() if menor_es_mejor else otros[col].idxmax())
    bench = float(t.loc[mejor, col])
    falta = (bench - limite) if menor_es_mejor else (limite - bench)
    cierra = ((bench - orig) if menor_es_mejor else (orig - bench)) / falta if abs(falta) > 1e-12 else float("nan")
    return [nombre, _nombre(mejor), ui.num(bench, d), ui.num(orig, d), ui.num(limite, d), ui.num(abs(falta), d),
            F.pct(cierra, 0) if _finito(cierra) else RAYA]


def _seccion_techo():
    te = _techo()
    if te is None:
        return ui.seccion("El techo teórico del problema", [_pendiente("El techo teórico", "techo.json", "lo calcula 04b_techo.py")])
    filas = [f for f in (_fila_techo("clasificacion", "AUC (mayor es mejor)", "auc_prueba", "auc_maximo", 4, False),
                         _fila_techo("regresion", "RMSE (menor es mejor)", "rmse_prueba", "rmse_minimo", 5, True),
                         _fila_techo("regresion", "MAE (menor es mejor)", "mae_prueba", "mae_minimo", 5, True)) if f is not None]
    p = te["prueba"]
    pesos = ", ".join(F.pct(te["pesos"][k], 1) for k in sorted(te["pesos"], key=int))
    cuerpo = [ui.cifras([
        ui.cifra("AUC máximo en la prueba", ui.num(p["auc_maximo"], 4), nota="Ordenando por la probabilidad escalonada de la Proposición 5.",
                 mini=True),
        ui.cifra("RMSE mínimo en la prueba", ui.num(p["rmse_minimo"], 5), nota=f"Prediciendo el índice menos el escalón medio; R² máximo "
                                                                               f"de {ui.num(p['r2_maximo'], 4)}.", mini=True),
        ui.cifra("Pérdida logarítmica mínima", ui.num(p["logloss_minima"], 4), nota=f"Con Brier mínimo de {ui.num(p['brier_minimo'], 4)}.",
                 mini=True),
    ], "cifras--3")]
    if filas:
        cuerpo.append(ui.tarjeta([
            ui.tabla(["Métrica en prueba", "Mejor benchmark", "Benchmark", "BDL", "Límite teórico", "Lo que le faltaba al benchmark",
                      "Parte que cierra BDL"], filas, compacta=True, num=[False, False, True, True, True, True, True], ocultar_movil=(5,)),
            ui.nota("El mejor benchmark es el de mejor métrica media en los pliegues externos (el de la regla de superioridad). La "
                    "última columna es la parte de la distancia entre ese benchmark y el límite que recorre BDL; es negativa si BDL "
                    "queda más lejos. El MAE de BDL es el de su media; el modelo también puede predecir la mediana."),
        ], titulo="El mejor benchmark ya estaba a milésimas del límite: cualquier mejora posible es pequeña",
            fuentes=[_fo("techo.json"), _fo("lideres_clasificacion.csv"), _fo("lideres_regresion.csv"), D.fuente("resultados/clasificacion_prueba.csv"),
                     D.fuente("resultados/regresion_prueba.csv")]))
    cuerpo.append(ui.aviso([
        f"El límite se calcula con el escalón ({ui.num(te['delta'], 2)} por producto activo) y las frecuencias de los "
        f"{CARDINAL[len(te['pesos'])]} niveles estimadas en entrenamiento ({pesos}). La variable active_products se usa solo para este "
        "cálculo: no interviene en el ajuste de ningún modelo.",
        "Eso acota de antemano el tamaño de cualquier mejora, y es la razón por la que el veredicto depende de pruebas pareadas, que "
        "son las que pueden detectar diferencias pequeñas y sistemáticas."], None, "Cómo se calcula y qué implica"))
    return ui.seccion(f"El techo del problema: sin la variable excluida, en la prueba el AUC no puede pasar de {ui.num(p['auc_maximo'], 4)} ni el RMSE "
                      f"bajar de {ui.num(p['rmse_minimo'], 5)}", cuerpo,
                      subtitulo="Lo mejor que puede hacer cualquier modelo que no reciba active_products, en forma cerrada.",
                      fuentes=[_fo("techo.json"), LIBRO_VALIDACION])


# --- 9. Pruebas de hipótesis ---------------------------------------------------------------------------------------------
def _nombres_pruebas(tarea):
    capacidad = F.pct(M.CAPACIDAD_INFORME, 0)
    return {"P1": "Wilcoxon de rangos con signo (tras Friedman)", "P2": f"t corregida ({_cita_en('nadeau2003')})",
            "P3": "DeLong" if tarea == "clasificacion" else "Permutación pareada",
            "P4": f"McNemar con la campaña del {capacidad}" if tarea == "clasificacion" else "Bootstrap pareado BCa"}


def _descripcion_pruebas(tarea):
    """Las cuatro pruebas del protocolo: hipótesis nula, supuestos y sobre qué datos."""
    c = TAREA[tarea]
    r = D.veredicto_original(tarea)
    p = D.pliegues(tarea)
    pliegues = r["pliegues"] if r is not None else (len(p) if p is not None else None)
    en_pliegues = f"los {F.entero(pliegues)} pliegues externos" if pliegues else "los pliegues externos"
    n_te = F.entero(D.resumen()["n_prueba"])
    nombres = _nombres_pruebas(tarea)
    filas = [
        [f"P1. {nombres['P1']}", f"la mediana de las diferencias de {c['metrica']} por pliegue es cero",
         "diferencias pareadas en los mismos pliegues; los pliegues comparten filas de entrenamiento, así que su p es optimista",
         en_pliegues],
        [f"P2. {nombres['P2']}", f"la diferencia media de {c['metrica']} por pliegue es cero",
         "diferencias aproximadamente normales; la dependencia entre pliegues se corrige inflando la varianza", en_pliegues],
    ]
    if tarea == "clasificacion":
        filas += [[f"P3. {nombres['P3']}", "los dos modelos tienen el mismo AUC",
                   "un cliente por fila, filas independientes, puntuadas por los dos modelos", f"los {n_te} clientes de prueba"],
                  [f"P4. {nombres['P4']}", "entre los clientes en que discrepan, cada modelo acierta con la misma frecuencia",
                   "solo cuentan los pares discordantes; cada modelo marca su cuarto de mayor puntaje", f"los {n_te} clientes de prueba"]]
    else:
        filas += [[f"P3. {nombres['P3']}", "el error cuadrático medio es igual",
                   f"filas independientes; no supone distribución ({F.entero(PERMUTACIONES)} cambios de signo)", f"los {n_te} clientes de prueba"],
                  [f"P4. {nombres['P4']}", "la diferencia de RMSE es cero",
                   f"filas independientes remuestreadas en pareja ({F.entero(D.resumen()['n_boot'])} réplicas)", f"los {n_te} clientes de prueba"]]
    return ui.tabla(["Prueba", "Hipótesis nula", "Supuestos", "Sobre qué datos"], filas, compacta=True, num=[False] * 4, ocultar_movil=(2,))


def _global_pruebas(tarea):
    r = D.veredicto_original(tarea)
    fr = r["friedman"]
    k = len(r["medias"])
    rechaza = fr["p_iman_davenport"] < D.ALFA
    return ui.cifras([
        ui.cifra(f"Friedman con Iman-Davenport, {F.entero(k)} modelos", f"F = {ui.num(fr['F_iman_davenport'], 1)}",
                 variacion=F.p_valor(fr["p_iman_davenport"]),
                 nota=f"{'Se rechaza' if rechaza else 'No se rechaza'} que los {F.entero(k)} modelos rindan igual en los "
                      f"{F.entero(r['pliegues'])} pliegues.", mini=True),
        ui.cifra("Rango medio de BDL", ui.num(fr["rangos_medios"][O], 1), f"de {F.entero(k)}",
                 variacion=f"{_nombre(r['mejor_benchmark'])}: {ui.num(fr['rangos_medios'][r['mejor_benchmark']], 1)}",
                 nota="En cada pliegue, 1 es la mejor métrica.", mini=True),
        ui.cifra("Diferencia crítica de Nemenyi", ui.num(fr["diferencia_critica_nemenyi"], 2), "rangos",
                 variacion=f"BDL frente al mejor benchmark: {F.p_valor(fr['p_nemenyi_original_vs_mejor'])}",
                 nota=f"Con {F.entero(k)} modelos tiene poca potencia ({_cita_en('demsar2006')}): se presenta como descripción.", mini=True),
    ], "cifras--3")


def _estadistico(v):
    """El estadístico de una prueba, como en el libro: entero si lo es y en notación científica si es muy pequeño."""
    if not _finito(v):
        return RAYA
    v = float(v)
    if v == int(v):
        return F.entero(v)
    if abs(v) < MINIMO_DECIMAL:
        exponente = math.floor(math.log10(abs(v)))
        return f"{ui.num(v / BASE_CIENTIFICA ** exponente, 2)} × {F.entero(BASE_CIENTIFICA)}{str(exponente).translate(SUPERINDICE)}"
    return ui.num(v, 4 if abs(v) < 100 else 1)


def _conclusion_prueba(k, x, r):
    """(si la diferencia va a favor del original, conclusión) de una prueba frente al mejor benchmark, como en el libro. Un
    rechazo cuenta para la segunda condición solo si va a favor del original y, en P1, si además rechaza la prueba global
    (protocolo, sección 7): eso es lo que guarda «rechazos_frente_al_mejor»."""
    favor = bool(x.get("a_favor", r["frente_al_mejor"]["ventaja_media"] > 0))
    p = x.get("p_holm")
    significativa = _finito(p) and p < D.ALFA and (k != "P1" or r["friedman"]["p_iman_davenport"] < D.ALFA)
    if r["rechazos_frente_al_mejor"].get(k):
        return favor, "se rechaza, a favor de BDL"
    if significativa and not favor:
        return favor, "se rechaza, a favor del benchmark"
    return favor, "no se rechaza"


def _rechazan(n, total):
    """«una de las cuatro pruebas disponibles rechaza», «ninguna de las cuatro … rechaza», «tres de las cuatro … rechazan»."""
    return f"{CARDINAL[n]} de las {CARDINAL[total]} pruebas disponibles {'rechazan' if n > 1 else 'rechaza'}"


def _detalle_mejor(tarea):
    """El original frente al mejor benchmark, prueba por prueba: estadístico, p crudo, p ajustado, sentido y conclusión."""
    c = TAREA[tarea]
    r = D.veredicto_original(tarea)
    dm, rechazos = r["frente_al_mejor"], r["rechazos_frente_al_mejor"]
    nombres = _nombres_pruebas(tarea)
    mejor = D.NOMBRE_CORTO.get(r["mejor_benchmark"], r["mejor_benchmark"])
    filas = []
    for k in PRUEBAS:
        if k not in dm:
            filas.append([f"{k}. {nombres[k]}", RAYA, RAYA, RAYA, RAYA, "pendiente: se calcula sobre la prueba"])
            continue
        x = dm[k]
        favor, conclusion = _conclusion_prueba(k, x, r)
        filas.append([f"{k}. {nombres[k]}", _estadistico(x["estadistico"]), _p(x["p"]), _p(x.get("p_holm")),
                      D.NOMBRE_CORTO[O] if favor else mejor, conclusion])
    minimo = 2 / 2 ** r["pliegues"]
    notas = [f"Ventaja media de BDL sobre {D.en_frase(r['mejor_benchmark'])} en los {F.entero(r['pliegues'])} pliegues: "
             f"{ui.con_signo(r['ventaja_sobre_el_mejor'], c['d'] + 1)} de {c['metrica']} (a favor de BDL en "
             f"{F.entero(dm['pliegues_a_favor'])}). Tamaño del efecto, con signo positivo a favor de BDL: δ de Cliff "
             f"{ui.con_signo(dm['cliff'], 2)}; d de Cohen {ui.con_signo(dm['cohen'], 2)}; d pareada {ui.con_signo(dm['cohen_pareada'], 2)}. "
             f"Los p ajustados son de Holm sobre las {F.entero(r['comparaciones_por_prueba'])} comparaciones de cada prueba; "
             f"α = {ui.num(D.ALFA, 2)}. Un rechazo cuenta para la regla solo si la diferencia va a favor de BDL.",
             "Estadísticos: en P1, la menor de las dos sumas de rangos; en P2, la t de la diferencia de BDL menos el benchmark; en P3, "
             + ("la z de DeLong (positiva: el AUC de BDL es mayor)" if tarea == "clasificacion"
                else "la diferencia de error cuadrático medio, BDL menos el benchmark (negativa: BDL se equivoca menos)")
             + "; en P4, "
             + ("los aciertos que solo tiene BDL menos los que solo tiene el benchmark." if tarea == "clasificacion"
                else "la diferencia de RMSE, BDL menos el benchmark.")]
    if minimo > D.ALFA:
        notas.append(f"Con {F.entero(r['pliegues'])} pliegues la prueba de Wilcoxon no puede dar un p menor que {ui.num(minimo, 4)}: P1 no "
                     f"puede rechazar hasta que estén las {CARDINAL[len(REPETICIONES)]} repeticiones de la validación anidada.")
    n_rech = sum(bool(v) for v in rechazos.values())
    return ui.tarjeta([ui.tabla(["Prueba", "Estadístico", "p crudo", "p ajustado (Holm)", "A favor de", "Conclusión"], filas, compacta=True,
                                num=[False, True, True, True, False, False], ocultar_movil=(1,))] + [ui.nota(n) for n in notas],
                      titulo=f"Frente al mejor benchmark, {D.en_frase(r['mejor_benchmark'])}: {_rechazan(n_rech, len(rechazos))} la "
                             "igualdad a favor de BDL",
                      subtitulo=f"Hipótesis {c['hipotesis']} del protocolo: el {c['metrica']} del modelo original es "
                                f"{'mayor' if c['mayor'] else 'menor'} que el del mejor benchmark.",
                      fuentes=[_fo("veredicto.json")])


def _lectura_benchmark(s):
    """Cuántas de las pruebas de BDL frente a un benchmark rechazan la igualdad y a favor de quién, en pocas palabras."""
    rechaza = s["rechaza"].astype(bool)
    favor = s["a_favor_del_original"].astype(bool) if "a_favor_del_original" in s else rechaza
    a_favor, en_contra, total = int((rechaza & favor).sum()), int((rechaza & ~favor).sum()), F.entero(len(s))
    if a_favor + en_contra == 0:
        return f"ninguna de {total}".replace(" ", F.FIJO)
    partes = ([f"{F.entero(a_favor)} de {total} · BDL"] if a_favor else []) \
        + ([f"{F.entero(en_contra)} de {total} · benchmark"] if en_contra else [])
    return "; ".join(p.replace(" ", F.FIJO) for p in partes)      # sin partir: «1 de 4 · BDL» quedaba con «BDL» en otra línea


def _tabla_benchmarks(tarea):
    """El original frente a cada uno de los nueve benchmarks: ventaja, pliegues a favor, p ajustados y tamaños del efecto."""
    c = TAREA[tarea]
    t = D.pruebas_original(tarea)
    r = D.veredicto_original(tarea)
    orden = sorted(t["benchmark"].unique(), key=lambda m: r["medias"][m], reverse=c["mayor"])
    filas, muestras = [], []
    for b in orden:
        s = t[t["benchmark"] == b].set_index("prueba")
        uno = s.iloc[0]
        filas.append([_nombre(b) + (" (el mejor)" if bool(uno["es_el_mejor_benchmark"]) else ""),
                      ui.con_signo(uno["ventaja_media_cv"], c["d"] + 1), f"{F.entero(uno['pliegues_a_favor'])} de {F.entero(r['pliegues'])}"]
                     + [_p(s.loc[k, "p_holm"]) if k in s.index else RAYA for k in PRUEBAS]
                     + [ui.con_signo(uno["cliff"], 2), ui.con_signo(uno["cohen"], 2), _lectura_benchmark(s)])
        muestras.append(D.ROL_MODELO.get(b, "contexto"))
    tabla = ui.tabla(["Benchmark", f"Ventaja media de {c['metrica']}", "Pliegues a favor"] + [f"{k}: p ajustado" for k in PRUEBAS]
                     + ["δ de Cliff", "d de Cohen", "Rechazan"], filas, compacta=True, muestras=muestras,
                     num=[False, True, True, True, True, True, True, True, True, False], ocultar_movil=(2, 7, 8))
    faltan = _faltan_pruebas(tarea)
    nota = (f"Una ventaja positiva favorece a BDL ({'más AUC' if c['mayor'] else 'menos RMSE'}). Los benchmarks van del mejor al peor por "
            "su métrica media externa. δ y d comparan los valores por pliegue de los dos modelos, con signo positivo a favor de BDL. La "
            "última columna dice cuántas de las pruebas rechazan la igualdad con p ajustado menor que α y a favor de quién.")
    if faltan:
        nota += f" {F.enumerar(faltan)} se calculan sobre la prueba y todavía no han llegado."
    return ui.tarjeta([tabla, ui.nota(nota)], titulo=f"BDL frente a cada uno de los {CARDINAL[len(orden)]} benchmarks",
                      subtitulo="p ajustados por Holm dentro de cada prueba y tamaños del efecto.",
                      fuentes=[_fo("pruebas_hipotesis.csv"), _fo("veredicto.json")])


# --- 9 bis. P4 y los empates en el corte de la campaña -------------------------------------------------------------------
def _pct_fino(x):
    """Una parte pequeña de los sorteos como porcentaje: el cero, sin decimales; lo demás, con dos."""
    return F.pct(x, 0) if x == 0 else F.pct(x, 2)


def _p4_empates():
    """La prueba P4 de clasificación repetida con desempates al azar (el bloque «p4» de empates_campana.json). None si el
    análisis no está, si le falta algo o si se hizo frente a un benchmark que ya no es el mejor del veredicto (habría que
    repetirlo: los empates de cada modelo siguen valiendo, la prueba frente a otro benchmark no)."""
    e = D.empates_campana()
    p4 = (e or {}).get("p4")
    if not isinstance(p4, dict):
        return None
    if not all(_finito(p4.get(k)) for k in ("rechaza_a_favor", "rechaza_en_contra")):
        return None
    if not all(_finito((p4.get(q) or {}).get("mediana")) for q in ("p_crudo", "p_holm")):
        return None
    r = D.veredicto_original("clasificacion")
    if r is not None and p4.get("mejor_benchmark") != r["mejor_benchmark"]:
        return None
    return p4


def _p4_estable(p4):
    """El resultado registrado de P4 no depende del desempate si casi ningún sorteo rechaza, ni a favor ni en contra."""
    return p4["rechaza_a_favor"] < PARTE_ESTABLE and p4["rechaza_en_contra"] < PARTE_ESTABLE


def _frase_p4_empates():
    """Una frase: qué pasa con la prueba P4 de clasificación al repetirla con desempates al azar. None si no está."""
    e, p4 = D.empates_campana(), _p4_empates()
    if p4 is None:
        return None
    sorteos = f"{F.entero(e['replicas'])} desempates" if _finito(e.get("replicas")) else "desempates"
    return (f"Repetida con {sorteos} al azar de los dos modelos, la prueba P4 de BDL frente {_tras('a', D.en_frase(p4['mejor_benchmark']))} "
            f"da un p crudo con mediana {_p(p4['p_crudo']['mediana'])} y, tras el ajuste de Holm, rechaza la igualdad a favor de BDL en el "
            f"{_pct_fino(p4['rechaza_a_favor'])} de los desempates y en contra en el {_pct_fino(p4['rechaza_en_contra'])}: "
            + ("el resultado registrado no depende del desempate." if _p4_estable(p4)
               else "el resultado de P4 sí depende del desempate y hay que leerlo con esa reserva."))


def _fila_empates(nombre, v):
    """Una fila de la tabla de empates: valores distintos del puntaje, empatados en el corte, cuántos entran, la captura
    según el desempate y el AUC-PR según la convención. Lo que el archivo no traiga sale con una raya."""
    def uno(clave, formato):
        return formato(v[clave]) if _finito(v.get(clave)) else RAYA

    def dos(a, b, texto):
        return texto(v[a], v[b]) if _finito(v.get(a)) and _finito(v.get(b)) else RAYA

    return [nombre, uno("valores_distintos", F.entero), uno("empatados", F.entero), uno("cupos", F.entero), uno("captura_protocolo", F.pct),
            dos("captura_esperada", "captura_desviacion", lambda m, s: f"{F.pct(m)} ± {ui.num(100 * s, 1)}"),
            dos("captura_minima", "captura_maxima", lambda a, b: f"{F.pct(a)} a {F.pct(b)}"),
            uno("precision_media", lambda x: ui.num(x, 4)),
            dos("precision_media_azar", "precision_media_azar_desviacion", lambda m, s: f"{ui.num(m, 4)} ± {ui.num(s, 4)}")]


def _lectura_empates(e, mejor):
    """Lo que dice la tabla, como en el libro: la probabilidad de BDL es escalonada, su captura es uno de los desempates
    posibles y con uno al azar se espera lo mismo que con el predictor óptimo; y lo que hace la convención con el AUC-PR."""
    o, op = D.empate_campana(O), e.get("optimo") if isinstance(e.get("optimo"), dict) else None
    if o is None:
        return []
    texto = f"La probabilidad de BDL toma {F.entero(o['valores_distintos'])} valores distintos en la prueba"
    if op is not None and _finito(op.get("valores_distintos")):
        texto += f" y la del predictor óptimo, {F.entero(op['valores_distintos'])}"
    texto += (": es una función escalonada, como dice la Proposición 5, y el corte de la campaña cae dentro de una meseta. Con un desempate al "
              f"azar se espera una captura del {F.pct(o['captura_esperada'])} ± {ui.num(100 * o['captura_desviacion'], 1)} puntos")
    if op is not None and _finito(op.get("captura_esperada")) and abs(o["captura_esperada"] - op["captura_esperada"]) < o["captura_desviacion"]:
        texto += f", lo mismo que con el predictor óptimo ({F.pct(op['captura_esperada'])})"
    b = (e.get("modelos") or {}).get(mejor)
    if isinstance(b, dict) and _finito(b.get("captura_esperada")) and _finito(b.get("captura_protocolo")):
        menor = abs(o["captura_esperada"] - b["captura_esperada"]) < o["captura_desviacion"]
        texto += (f". La diferencia con {D.en_frase(mejor)} ({F.pct(b['captura_protocolo'])}) es {'menor' if menor else 'mayor'} que esa "
                  "variación")
    lider = _lider("clasificacion")
    if lider is not None and _finito(lider.get("recall_op")):
        texto += (f". La app, como el Entregable 3, rompe los empates al azar con la semilla del proyecto: la captura de BDL que "
                  f"muestra, {F.pct(lider['recall_op'])}, es otro de esos desempates")
    notas = [texto + "."]
    pr = D.frase_empate_auc_pr(O, d=4)
    if pr:
        if op is not None and all(_finito(op.get(k)) for k in ("precision_media", "precision_media_azar")):
            pr += (f" Al predictor óptimo le pasa lo mismo ({ui.num(op['precision_media'], 4)} y {ui.num(op['precision_media_azar'], 4)}): "
                   "esa desventaja viene de la convención, no del orden de los clientes.")
        notas.append(pr)
    return notas


def _tarjeta_empates():
    """«P4 y los empates de la campaña»: la tabla de empates (el original, el mejor benchmark, los demás modelos con empate
    y el predictor óptimo) y la prueba P4 repetida con desempates al azar, como la Tabla 6.11 del libro. Todo sale de
    empates_campana.json; si el archivo no está, el bloque queda pendiente."""
    e = D.empates_campana()
    if e is None:
        return _pendiente("El análisis de los empates de la campaña", "empates_campana.json", "lo calcula 04d_empates_campana.py")
    r, p4 = D.veredicto_original("clasificacion"), _p4_empates()
    mejor = r["mejor_benchmark"] if r is not None else (e.get("p4") or {}).get("mejor_benchmark")
    modelos = e["modelos"]
    orden = [m for m in (O, mejor) if m in modelos]
    orden += [m for m in D.modelos_con_empate() if m not in orden]
    filas = [_fila_empates(_nombre(m), modelos[m]) for m in orden]
    if isinstance(e.get("optimo"), dict):
        filas.append(_fila_empates("Predictor óptimo", e["optimo"]))
    cuerpo = [ui.tabla(["Modelo", "Valores distintos", "Empatados", "Entran", "Captura (protocolo)", "Captura al azar", "Captura posible",
                        "AUC-PR", "AUC-PR al azar"], filas, compacta=True, ocultar_movil=(1, 4, 6)),
              ui.nota("Empatados: los clientes con la misma probabilidad que el último contactado; de ellos solo entra una parte, y cuál, "
                      "lo decide el desempate. Protocolo: el desempate por posición en la tabla, el del guion de las pruebas y el del "
                      "libro. Al azar: valor esperado ± desviación en puntos porcentuales (distribución hipergeométrica) y, en el AUC-PR, "
                      "media ± desviación de "
                      + (f"{F.entero(e['replicas_precision_media'])} desempates." if _finito(e.get("replicas_precision_media"))
                         else "los desempates."))]
    cuerpo += [ui.nota(n) for n in _lectura_empates(e, mejor)]
    o = D.empate_campana(O)
    if o is None:
        titulo = "P4 y los empates de la campaña: en el corte de BDL no hay empate"
    else:
        titulo = (f"P4 y los empates de la campaña: {F.entero(o['empatados'])} clientes comparten la probabilidad del corte de BDL y solo "
                  f"entran {F.entero(o['cupos'])}")
    if p4 is not None:
        titulo += "; el resultado de P4 no depende del desempate" if _p4_estable(p4) else "; el resultado de P4 depende del desempate"
        central = F.pct(NIVEL_SORTEOS, 0)
        pares = []
        if _finito(e.get("replicas")):
            pares.append(("Desempates al azar de los dos modelos", F.entero(e["replicas"])))
        for clave, rotulo in (("p_crudo", "p crudo"), ("p_holm", "p ajustado por Holm")):
            q = p4[clave]
            rango = f" (de {_p(q['q025'])} a {_p(q['q975'])})" if _finito(q.get("q025")) and _finito(q.get("q975")) else ""
            pares.append((f"{rotulo}: mediana y {central} central de los desempates", f"{_p(q['mediana'])}{rango}"))
        if _finito(p4.get("a_favor_del_original")):
            pares.append(("La diferencia va a favor de BDL", f"en el {F.pct(p4['a_favor_del_original'], 0)} de los desempates"))
        pares += [("Rechaza la igualdad a favor de BDL", f"en el {_pct_fino(p4['rechaza_a_favor'])}"),
                  ("Rechaza la igualdad en contra de BDL", f"en el {_pct_fino(p4['rechaza_en_contra'])}")]
        registrado = p4.get("protocolo") or {}
        if _finito(registrado.get("p")):
            pares.append(("Resultado registrado (desempate del protocolo)",
                          f"{F.p_valor(registrado['p'])}" + (f"; ajustado, {_p(registrado['p_holm'])}" if _finito(registrado.get("p_holm")) else "")))
        cuerpo += [html.H3(f"La prueba P4 frente {_tras('a', D.en_frase(p4['mejor_benchmark']))}, repetida con desempates al azar",
                           className="mas__titulo"),
                   ui.ficha(pares),
                   ui.nota("El resultado registrado no depende del desempate: es el de casi todos los sorteos." if _p4_estable(p4)
                           else "El resultado de P4 sí depende del desempate y hay que leerlo con esa reserva.")]
    elif isinstance(e.get("p4"), dict):
        cuerpo.append(ui.aviso("La prueba P4 con desempates al azar de este archivo no corresponde al mejor benchmark del veredicto o "
                               "está incompleta: hay que repetir 04d_empates_campana.py. Los empates de cada modelo no cambian.", "atencion",
                               "La sensibilidad de P4 está pendiente de repetirse"))
    cuerpo.append(ui.nota("Es un análisis de sensibilidad posterior a los resultados y fuera del protocolo: no cambia el veredicto; dice "
                          "cuánto depende de una decisión que el protocolo no fijó."))
    subtitulo = "Clasificación: la prueba P4 (McNemar) compara a quién acierta cada modelo en la campaña."
    if _finito(e.get("campana")) and _finito(e.get("n")) and e["n"]:
        subtitulo = (f"Clasificación: la campaña del {F.pct(e['campana'] / e['n'], 0)} contacta a los {F.entero(e['campana'])} clientes de "
                     f"mayor probabilidad de los {F.entero(e['n'])} de prueba, y la prueba P4 (McNemar) compara a quién acierta cada modelo "
                     "en ella.")
    return ui.tarjeta(cuerpo, titulo=titulo, subtitulo=subtitulo, id=f"{PRE}-empates",
                      fuentes=[_fo("empates_campana.json"), _fo("veredicto.json"), LIBRO_EMPATES])


def _pruebas_tarea(tarea):
    if D.veredicto_original(tarea) is None or D.pruebas_original(tarea) is None:
        return [ui.tarjeta(_descripcion_pruebas(tarea), titulo="Las pruebas del protocolo: hipótesis nula y supuestos", fuentes=[PROTOCOLO]),
                _pendiente("Las pruebas de hipótesis", "pruebas_hipotesis.csv", "las calcula 04_pruebas_hipotesis.py")]
    return [_global_pruebas(tarea),
            ui.tarjeta(_descripcion_pruebas(tarea), titulo="Las cuatro pruebas del protocolo: hipótesis nula y supuestos",
                       subtitulo="Datos de corte transversal: las pruebas de la columna «datos independientes» de la guía.",
                       fuentes=[PROTOCOLO, LIBRO_VALIDACION]),
            _detalle_mejor(tarea), _tabla_benchmarks(tarea)]


def _seccion_pruebas(tarea):
    protocolo = ui.aviso([
        f"El protocolo se congeló antes de ejecutar los experimentos definitivos: hipótesis, métricas, particiones, "
        f"{CARDINAL[len(REPETICIONES)]} repeticiones de la validación anidada (semillas {F.enumerar(str(s) for s in REPETICIONES)}), "
        "las cuatro pruebas, la corrección de Holm, el presupuesto de ajuste y los conjuntos adicionales.",
        "Equidad: los diez modelos usan los mismos pliegues, el mismo preprocesamiento y las mismas variables; ningún benchmark recibió "
        "menos evaluaciones de hiperparámetros que el original. En clasificación, el original recibe además el puntaje continuo de "
        "las filas de entrenamiento: no es una ventaja oculta sino el mecanismo que se evalúa, y la ablación «sin puntaje» mide "
        "cuánto aporta."], None, "Protocolo pre-registrado y equidad de la comparación")
    return ui.seccion("Las pruebas de hipótesis: el original frente a cada benchmark, con p ajustados y tamaño del efecto",
                      [html.Div(_pruebas_tarea(tarea), id=f"{PRE}-pruebas", className="rejilla"), _tarjeta_empates(), protocolo],
                      subtitulo="Prueba global y, después, el modelo original frente a cada uno de los nueve benchmarks con las cuatro "
                                "pruebas del protocolo.",
                      fuentes=[_fo("pruebas_hipotesis.csv"), _fo("veredicto.json")]
                      + ([_fo("empates_campana.json")] if D.hay_original("empates_campana.json") else []) + [PROTOCOLO])


# --- 10. Robustez --------------------------------------------------------------------------------------------------------
def _dec(tarea):
    return TAREA[tarea]["d"]


def _cardinal_m(n):
    """El cardinal delante de un nombre masculino: «uno de los nueve estudios», «ninguno» (CARDINAL va en femenino)."""
    return {0: "ninguno", 1: "uno"}.get(n, CARDINAL.get(n, F.entero(n)))


def _cuantas_de(k, n):
    """«las cuatro», «una de las cuatro», «ninguna de las cuatro» (de un total de n, en femenino)."""
    total = CARDINAL.get(n, F.entero(n))
    return f"las {total}" if k == n else f"{CARDINAL.get(k, F.entero(k))} de las {total}"


def _tarjeta_ablacion():
    """La tabla de la ablación, como la del libro: cada variante frente al modelo completo en los pliegues externos y en la
    prueba, con los p ajustados. El título cuenta en cuántas queda por delante el completo y en cuántas la diferencia es
    significativa."""
    t = D.original_tabla("ablacion.csv")
    if t is None:
        return None
    tablas = []
    for tarea in TAREAS:
        c, d = TAREA[tarea], _dec(tarea)
        filas = []
        for _, f in t[t["tarea"] == tarea].iterrows():
            base = [str(f["nombre"]), f"{ui.num(f['cv_media'], d)} ± {ui.num(f['cv_desv'], d)}", ui.num(f["prueba"], d)]
            if f["variante"] == O:
                filas.append(base + [RAYA] * 5)
            else:
                favor = f.get("pliegues_a_favor_del_completo")
                filas.append(base + [ui.con_signo(f["aporte_cv"], d + 1), F.entero(favor) if _finito(favor) else RAYA,
                                     _p(f.get("p_t_corregida_holm")), ui.con_signo(f["aporte_prueba"], d + 1), _p(f.get("p_prueba_holm"))])
        if filas:                 # una tabla por tarea y encabezados cortos: los encabezados no parten línea y ocho largos no caben
            tablas += [html.H3(f"{D.NOMBRE_TAREA[tarea]}: {c['metrica']}", className="mas__titulo"),
                       ui.tabla(["Variante", "Externo", "En prueba", "Aporte externo", "Pliegues a favor", "p externo", "Aporte en prueba",
                                 "p en prueba"], filas, compacta=True, ocultar_movil=(2, 4))]
    variantes = t[t["variante"] != O]
    n = len(variantes)
    delante = int((variantes["aporte_cv"] > 0).sum())
    titulo = f"Ablación: en los pliegues externos el modelo completo queda por delante de {_cuantas_de(delante, n)} variantes"
    if "p_t_corregida_holm" in variantes:
        claras = int(((variantes["aporte_cv"] > 0) & (variantes["p_t_corregida_holm"] < D.ALFA)).sum())
        titulo += f" ({CARDINAL.get(claras, F.entero(claras))} con diferencia significativa)"
    titulo += f"; en la prueba, de {_cuantas_de(int((variantes['aporte_prueba'] > 0).sum()), n)}"
    return ui.tarjeta(tablas + [
        ui.nota("Cada variante retira o cambia un componente y pasa por el mismo procedimiento que el modelo completo, con los pliegues "
                f"externos de la semilla {REPETICIONES[0]}. Externo: media ± desviación de la métrica en esos pliegues. Aporte: lo que "
                "mejora el modelo completo respecto de la variante (positivo: el componente ayuda); «pliegues a favor» son los pliegues "
                "en que gana el completo. Los p van ajustados por Holm: el externo es el de la t corregida; el de prueba, el de DeLong en "
                "clasificación y el de la permutación pareada en regresión."),
    ], titulo=titulo, fuentes=[_fo("ablacion.csv")])


def _tarjeta_semillas():
    t = D.original_tabla("semillas.csv")
    if t is None:
        return None
    filas, frases = [], []
    for tarea in TAREAS:
        c = TAREA[tarea]
        s = t[t["tarea"] == tarea]
        if s.empty:
            continue
        g = s.groupby("modelo")["metrica"].agg(["mean", "std", "count"])
        primeros = list(g.sort_values("mean", ascending=not c["mayor"]).index[:N_PRINCIPALES + 1])
        if O in g.index and O not in primeros:
            primeros.append(O)
        for m in primeros:
            filas.append([f"{_nombre(m)} ({c['que']})", ui.num(g.loc[m, "mean"], c["d"] + 1), ui.num(g.loc[m, "std"], c["d"] + 1),
                          F.entero(g.loc[m, "count"])])
        if O in g.index:
            frases.append(f"de {ui.num(g.loc[O, 'std'], c['d'] + 1)} de {c['metrica']} en {c['que']} ({F.entero(g.loc[O, 'count'])} semillas)")
    completos = int(t.groupby(["tarea", "modelo"])["semilla"].count().eq(SEMILLAS_ESTABILIDAD).sum())
    total = int(t.groupby(["tarea", "modelo"]).ngroups)
    titulo = (("Semillas: al reajustar, la desviación de BDL en prueba es " + " y ".join(frases)) if frases else
              "Semillas: el estudio va por los benchmarks y todavía no llega a BDL")
    return ui.tarjeta([
        ui.tabla(["Modelo", "Media en prueba", "Desviación", "Semillas"], filas, compacta=True),
        ui.nota(f"Cada modelo final se reajusta con {F.entero(SEMILLAS_ESTABILIDAD)} semillas y los hiperparámetros finales, y se evalúa "
                f"en la prueba. La tabla muestra, de cada tarea, los {CARDINAL[N_PRINCIPALES + 1]} modelos de mejor media (y BDL, si no "
                f"está entre ellos); el archivo trae {F.entero(total)} modelos y {F.entero(completos)} ya tienen todas sus semillas"
                + ("." if completos == total else ": el estudio sigue corriendo.")
                + " Una diferencia entre modelos menor que la desviación por semilla no es atribuible al modelo."),
    ], titulo=titulo, fuentes=[_fo("semillas.csv")])


def _tarjeta_sensibilidad():
    t = D.original_tabla("robustez/sensibilidad.csv")
    if t is None:
        return None
    filas, frases = [], []
    for tarea in TAREAS:
        c = TAREA[tarea]
        s = t[t["tarea"] == tarea]
        p, r = D.pliegues(tarea), D.veredicto_original(tarea)
        if s.empty:
            continue
        ref = None
        if p is not None and r is not None:
            ref = float(p.loc[[i for i in p.index if str(i).startswith("42-")], r["mejor_benchmark"]].mean())
        gana = None if ref is None else int(((s["media"] > ref) if c["mayor"] else (s["media"] < ref)).sum())
        peor, mejor = (s["media"].min(), s["media"].max()) if c["mayor"] else (s["media"].max(), s["media"].min())
        filas.append([D.NOMBRE_TAREA[tarea], F.entero(len(s)), RAYA if gana is None else F.entero(gana), ui.num(peor, c["d"]),
                      ui.num(mejor, c["d"]), RAYA if ref is None else ui.num(ref, c["d"])])
        if gana is not None:
            frases.append(f"{F.entero(gana)} de {F.entero(len(s))}{'' if frases else ' configuraciones'} en {c['que']}")
    titulo = ("Sensibilidad: BDL queda por delante del mejor benchmark en " + " y en ".join(frases)) if frases \
        else "Sensibilidad a los hiperparámetros propios"
    return ui.tarjeta([
        ui.tabla(["Tarea", "Configuraciones", "Mejores que el mejor benchmark", "Peor valor", "Mejor valor", "Mejor benchmark"], filas,
                 compacta=True),
        ui.nota(f"Se varía un hiperparámetro cada vez alrededor de la configuración final, con los cinco pliegues externos de la "
                f"semilla {REPETICIONES[0]} y sin nueva búsqueda; en clasificación se fija la variante que usa el puntaje. El mejor "
                "benchmark se mide en los mismos pliegues."),
    ], titulo=titulo, fuentes=[_fo("robustez/sensibilidad.csv"), _fo("veredicto.json")])


def _tarjeta_ruido():
    t = D.original_tabla("robustez/ruido.csv")
    if t is None:
        return None
    filas, frases = [], []
    for tarea in TAREAS:
        c = TAREA[tarea]
        piv = t[t["tarea"] == tarea].pivot_table(index=["escenario", "nivel"], columns="modelo", values="media", sort=False)
        if piv.empty or O not in piv:
            continue
        piv = piv.reindex(columns=list(TRES))
        completo = piv.notna().all(axis=1)       # el guion escribe fila a fila: un escenario a medias se muestra, pero no se cuenta
        mejor = piv.idxmax(axis=1) if c["mayor"] else piv.idxmin(axis=1)
        if completo.any():
            frases.append(f"{F.entero(int((mejor[completo] == O).sum()))} de {F.entero(int(completo.sum()))}"
                          f"{'' if frases else ' escenarios'} en {c['que']}")
        for (esc, niv), f in piv.iterrows():
            filas.append([f"{_mayuscula(str(esc))} ({c['que']})", ui.num(niv, 3)] + [ui.num(f[m], c["d"]) if _finito(f[m]) else RAYA for m in TRES]
                         + [_nombre(mejor[(esc, niv)]) if completo[(esc, niv)] else "en curso"])
    titulo = ("Ruido: BDL es el mejor de los tres en " + " y en ".join(frases)) if frases else "Ruido y valores atípicos"
    presentes = {str(e) for e in t["escenario"]}
    faltan = [e for e in CONTAMINACIONES if e not in presentes]
    notas = ["Se contamina solo el entrenamiento; la evaluación es siempre contra el puntaje y la etiqueta sin contaminar, y se compara "
             "con los dos mejores benchmarks de boosting. Nivel: la desviación del ruido que se suma al puntaje, la fracción de puntajes "
             "que se cambia por valores atípicos o, en los predictores, la desviación del ruido en desviaciones estándar de cada variable."]
    if faltan:
        notas.append(f"El estudio sigue corriendo y escribe cada escenario al terminarlo: todavía faltan los de {F.enumerar(faltan)}.")
    return ui.tarjeta([
        ui.tabla(["Contaminación", "Nivel", _nombre(O), _nombre("xgboost"), _nombre("lightgbm"), "Mejor"], filas, compacta=True,
                 num=[False, True, True, True, True, False], alta=True),
    ] + [ui.nota(n) for n in notas], titulo=titulo, fuentes=[_fo("robustez/ruido.csv")])


def _tarjeta_muestra():
    t = D.original_tabla("robustez/muestra.csv")
    if t is None:
        return None
    filas = []
    for tarea in TAREAS:
        c = TAREA[tarea]
        for n, s in t[t["tarea"] == tarea].groupby("n"):
            s = s.set_index("modelo")
            filas.append([f"{F.entero(n)} filas ({c['que']})"] + [ui.num(s.loc[m, "media"], c["d"]) if m in s.index else RAYA
                                                                 for m in (O, "xgboost", "lightgbm")]
                         + [ui.num(s.loc[O, "ajuste_s"], 1) if O in s.index else RAYA])
    cuerpo = [ui.tabla(["Filas de entrenamiento", _nombre(O), _nombre("xgboost"), _nombre("lightgbm"), "Ajuste de BDL (s)"], filas,
                       compacta=True)]
    col = D.original_tabla("robustez/columnas.csv")
    if col is not None and O in set(col["modelo"]):
        s = col[col["modelo"] == O].sort_values("columnas")
        cuerpo.append(ui.nota("Tiempo de ajuste de BDL al variar las columnas originales, con las mismas filas: "
                              + "; ".join(f"{F.entero(f['columnas'])} columnas, {ui.num(f['ajuste_s'], 1)} s" for _, f in s.iterrows()) + "."))
    cuerpo.append(ui.nota("Curva de aprendizaje con una parte de las filas de entrenamiento de cada pliegue externo y los "
                          "hiperparámetros finales. La complejidad predice un costo lineal en las filas y en las columnas."))
    return ui.tarjeta(cuerpo, titulo="Tamaño de muestra: la métrica y el tiempo de ajuste según las filas",
                      fuentes=[_fo("robustez/muestra.csv"), _fo("robustez/columnas.csv")])


def _tarjeta_ciudades():
    t = D.original_tabla("robustez/ciudades.csv")
    if t is None:
        return None
    filas, frases = [], []
    for tarea in TAREAS:
        c = TAREA[tarea]
        piv = t[t["tarea"] == tarea].pivot(index="ciudad", columns="modelo", values="metrica")
        if piv.empty or O not in piv:
            continue
        piv = piv.reindex(columns=list(TRES)).dropna()      # el guion escribe fila a fila: una ciudad a medias no se cuenta
        if piv.empty:
            continue
        otros = piv.drop(columns=O)
        gana = int(((piv[O] > otros.max(axis=1)) if c["mayor"] else (piv[O] < otros.min(axis=1))).sum())
        frases.append(f"{F.entero(gana)} de {F.entero(len(piv))}{'' if frases else ' ciudades'} en {c['que']}")
        filas.append([f"Media de las ciudades ({c['que']})"] + [ui.num(piv[m].mean(), c["d"]) for m in TRES])
    titulo = ("Cambio de distribución: BDL es el mejor de los tres en " + " y en ".join(frases)) if frases \
        else "Cambio de distribución"
    return ui.tarjeta([
        ui.tabla(["Ciudad que se deja fuera", _nombre(O), _nombre("xgboost"), _nombre("lightgbm")], filas, compacta=True),
        ui.nota("Se deja fuera una ciudad cada vez: los modelos se ajustan con los clientes de las demás y se evalúan en la que quedó "
                "fuera, que no intervino en ninguna decisión. La tabla ciudad por ciudad está en el botón «…» de esta tarjeta."),
    ], titulo=titulo, detalles=_detalle_ciudades(t), fuentes=[_fo("robustez/ciudades.csv")])


def _detalle_ciudades(t):
    """La métrica de los tres modelos en cada ciudad que se deja fuera, con sus clientes; en negrita, el mejor de cada fila."""
    filas, mejores = [], set()
    for tarea in TAREAS:
        c = TAREA[tarea]
        s = t[t["tarea"] == tarea]
        piv = s.pivot(index="ciudad", columns="modelo", values="metrica")
        clientes = s.groupby("ciudad")["n"].first() if "n" in s else None
        for ciudad, f in piv.iterrows():
            presentes = {m: float(f[m]) for m in TRES if m in f and _finito(f[m])}
            if len(presentes) == len(TRES):
                gana = (max if c["mayor"] else min)(presentes, key=presentes.get)
                mejores.add((len(filas), 2 + TRES.index(gana)))
            filas.append([f"{ciudad} ({c['que']})", RAYA if clientes is None else F.entero(clientes[ciudad])]
                         + [ui.num(presentes[m], c["d"]) if m in presentes else RAYA for m in TRES])
    return ui.tabla(["Ciudad que se deja fuera", "Clientes"] + [_nombre(m) for m in TRES], filas, mejor=mejores, mejor_texto=None,
                    compacta=True, alta=True)


def _tarjeta_adicionales():
    t = D.original_tabla("adicionales.csv")
    c4 = D.original_json("condicion4.json")
    if t is None:
        return None
    filas = []
    for _, f in t.iterrows():
        tarea = f["tarea"]
        nombre = f"{f['nombre']}{' †' if isinstance(f['omitida'], str) and f['omitida'] else ''} ({TAREA[tarea]['que']})"
        diez = F.entero(len(D.ORDEN_MODELOS[tarea]))
        if not _finito(f.get("original_media")):
            filas.append([nombre, RAYA, RAYA, RAYA, RAYA, RAYA, f"en curso: {F.entero(f['modelos'])} de {diez} modelos"])
            continue
        d = 4 if tarea == "clasificacion" else (5 if f["benchmark_media"] < 1 else (3 if f["benchmark_media"] < 100 else 0))
        estado = "mejor" if bool(f["es_mejor"]) else ("peor, significativo" if bool(f["peor_significativo"]) else "peor, no significativo")
        if not bool(f["completo"]):
            estado += f" (provisional: {F.entero(f['modelos'])} de {diez} modelos)"
        filas.append([nombre, _nombre(f["mejor_benchmark"]), f"{ui.num(f['benchmark_media'], d)} ± {ui.num(f['benchmark_desv'], d)}",
                      f"{ui.num(f['original_media'], d)} ± {ui.num(f['original_desv'], d)}", F.pct(f["ventaja_relativa"], 2, signo=True),
                      _p(f["p_t_corregida"]), estado])
    completos = int(t.groupby("conjunto")["completo"].all().sum())
    cuerpo = [ui.tabla(["Conjunto (tarea)", "Mejor benchmark", "Benchmark (externo)", "BDL (externo)", "Ventaja relativa", "p (t corregida)",
                        "Resultado"], filas, compacta=True, num=[False, False, True, True, True, True, False], ocultar_movil=(2, 5)),
              ui.nota("† Conjunto con una variable omitida para todos los modelos, la situación para la que se diseñó BDL; los otros son "
                      "de control. Externo: media ± desviación de los cinco pliegues externos. La t corregida compara a BDL con el "
                      "mejor benchmark del conjunto.")]
    if c4 is None:
        cuerpo.append(_pendiente("La cuarta condición del veredicto", "condicion4.json",
                                 "se consolida cuando terminan los cuatro conjuntos adicionales"))
    else:
        estado = {"cumple": "se cumple", "no cumple": "no se cumple", "provisional": "provisional", "pendiente": "pendiente"}
        cuerpo.append(ui.nota(f"Cuarta condición del veredicto ({_fo('condicion4.json')}): " + "; ".join(
            f"en {TAREA[t_]['que']}, {estado[D.estado_condicion4(t_)]} ({_texto_cuarta(t_)})" for t_ in TAREAS if t_ in c4) + "."))
        con_resultado = t[t["original_media"].map(_finito)].groupby("tarea")["conjunto"].nunique()
        if any(int(con_resultado.get(t_, 0)) > int(c4[t_].get("conjuntos", 0)) for t_ in TAREAS if t_ in c4):
            cuerpo.append(ui.nota("La tabla de arriba ya trae resultados que condicion4.json todavía no ha consolidado: la condición se "
                                  "actualiza cuando la corrección vuelve a escribir ese archivo."))
    cuantos = "ninguno" if completos == 0 else ("uno" if completos == 1 else CARDINAL.get(completos, F.entero(completos)))
    titulo = (f"Conjuntos adicionales: {cuantos} de {CARDINAL[CONJUNTOS_ADICIONALES]} con los diez modelos evaluados")
    return ui.tarjeta(cuerpo, titulo=titulo, subtitulo="Cuatro conjuntos públicos elegidos en el protocolo, antes de ajustar ningún "
                                                      "modelo en ellos: dos con una variable omitida y dos de control.",
                      fuentes=[_fo("adicionales.csv"), _fo("condicion4.json"), PROTOCOLO])


def _tarjeta_subgrupos():
    """Errores por subgrupo de la prueba: el modelo final de BDL frente al mejor benchmark de cada tarea."""
    t = D.original_tabla("errores_subgrupos.csv")
    if t is None or t.empty:
        return None
    br, bc = t["benchmark_regresion"].iloc[0], t["benchmark_clasificacion"].iloc[0]
    filas, gana_r, gana_c, con_auc, partes = [], 0, 0, 0, 0
    for _, f in t.iterrows():
        mejor_r = bool(f["rmse_original"] < f["rmse_benchmark"])
        hay_auc = _finito(f["auc_original"]) and _finito(f["auc_benchmark"])
        mejor_c = bool(hay_auc and f["auc_original"] > f["auc_benchmark"])
        if str(f["subgrupo"]).strip().lower() != "todos":
            partes, gana_r, gana_c, con_auc = partes + 1, gana_r + mejor_r, gana_c + mejor_c, con_auc + hay_auc
        filas.append([f["subgrupo"], F.entero(f["clientes"]), ui.num(f["rmse_original"], D_RMSE_SUBGRUPO), ui.num(f["rmse_benchmark"], D_RMSE_SUBGRUPO),
                      D.NOMBRE_CORTO[O] if mejor_r else D.NOMBRE_CORTO.get(br, br),
                      ui.num(f["auc_original"], 4) if hay_auc else RAYA, ui.num(f["auc_benchmark"], 4) if hay_auc else RAYA,
                      RAYA if not hay_auc else (D.NOMBRE_CORTO[O] if mejor_c else D.NOMBRE_CORTO.get(bc, bc))])
    titulo = (f"Errores por subgrupo: BDL tiene menos RMSE que {D.en_frase(br)} en {F.entero(gana_r)} de {F.entero(partes)} subgrupos "
              f"de la prueba y más AUC que {D.en_frase(bc)} en {F.entero(gana_c)} de {F.entero(con_auc)}")
    return ui.tarjeta([
        ui.tabla(["Subgrupo de prueba", "Clientes", "RMSE de BDL", f"RMSE de {_nombre(br)}", "Menor RMSE", "AUC de BDL",
                  f"AUC de {_nombre(bc)}", "Mayor AUC"], filas, compacta=True, num=[False, True, True, True, False, True, True, False],
                 ocultar_movil=(1,)),
        ui.nota("Los clientes de prueba por edad (los valores que toma en el conjunto), por la cola de tx_count y por segmento. En cada "
                "subgrupo, el modelo final de cada tarea frente al mejor benchmark de esa tarea. Son diferencias pequeñas y sin prueba "
                "de hipótesis: describen dónde se equivoca cada modelo y no cuentan para el veredicto."),
    ], titulo=titulo, fuentes=[_fo("errores_subgrupos.csv")])


def _tarjeta_incertidumbre():
    c = V.conformal()["modelos"]
    if O not in c or "conjunto" not in c[O].get(NIVEL, {}):
        return None
    r = D.veredicto_original("regresion")
    mejor = r["mejor_benchmark"] if r is not None and r["mejor_benchmark"] in c else "xgboost"
    filas = []
    for nivel in sorted(c[O], key=float):
        b, o = c[mejor][nivel], c[O][nivel]
        filas.append([F.pct(float(nivel), 0), f"{F.pct(b['cobertura_prueba'], 1)} · {ui.num(b['ancho'], 4)}",
                      f"{F.pct(o['cobertura_prueba'], 1)} · {ui.num(o['ancho'], 4)}",
                      f"{F.pct(o['central']['cobertura_prueba'], 1)} · {ui.num(o['central']['largo_medio'], 4)}",
                      f"{F.pct(o['conjunto']['cobertura_prueba'], 1)} · {ui.num(o['conjunto']['largo'], 4)} en "
                      f"{F.entero(o['conjunto']['tramos'])} tramos"])
    o, b = c[O][NIVEL], c[mejor][NIVEL]
    veces = b["ancho"] / o["conjunto"]["largo"]
    cuerpo = [
        ui.tabla(["Cobertura nominal", f"{_nombre(mejor)}: pronóstico ± q", "BDL: pronóstico ± q", "BDL: intervalo central",
                  "BDL: conjunto de predicción"], filas, compacta=True, ocultar_movil=(2, 3)),
        ui.nota(f"Cada celda: cobertura medida en los {F.entero(D.resumen()['n_prueba'])} clientes de prueba · largo. «Pronóstico ± q» "
                "es el intervalo conforme con el cuantil de los errores absolutos fuera de pliegue, el que usan todos los modelos en "
                "«Predicción en vivo». El conjunto de predicción es la unión de tramos alrededor de cada nivel; su largo es la suma de "
                "los tramos."),
    ]
    desvio = abs(o["cobertura_prueba"] - float(NIVEL))
    if desvio > TOLERANCIA_COBERTURA:
        cuerpo.append(ui.aviso(f"Con BDL el intervalo simétrico «pronóstico ± q» no cumple su cobertura nominal: al {F.pct(float(NIVEL), 0)} "
                               f"mide {F.pct(o['cobertura_prueba'], 1)}. El error de BDL cae en niveles separados y el cuantil queda en "
                               "el hueco entre dos de ellos, así que un cambio mínimo deja dentro o fuera a un nivel entero. El "
                               f"conjunto de predicción del modelo, que sigue esos niveles, mide {F.pct(o['conjunto']['cobertura_prueba'], 1)}: "
                               + ("queda en la nominal." if abs(o["conjunto"]["cobertura_prueba"] - float(NIVEL)) <= TOLERANCIA_COBERTURA
                                  else "tampoco llega a la nominal."), "atencion",
                               "El intervalo simétrico no es el adecuado para un error con escalones"))
    inc = D.original_tabla("incertidumbre_prueba.csv")
    if inc is None:
        cuerpo.append(ui.nota(f"La tabla de incertidumbre de la corrección ({_fo('incertidumbre_prueba.csv')}) todavía no ha llegado: las "
                              "cifras de arriba se calcularon con el artefacto de BDL al armar los datos y deben coincidir con ella."))
    titulo = (f"Incertidumbre: al {F.pct(float(NIVEL), 0)}, el conjunto de predicción de BDL cubre el "
              f"{F.pct(o['conjunto']['cobertura_prueba'], 1)} con un largo {F.entero(veces)} veces menor que el intervalo "
              f"{_tras('de', D.en_frase(mejor))}")
    return ui.tarjeta(cuerpo, titulo=titulo, subtitulo="Hipótesis H3 del protocolo (secundaria): conjuntos más cortos con la cobertura "
                                                      "nominal.",
                      fuentes=[D.fuente("conformal.json"), _fo("incertidumbre_prueba.csv")])


# Los estudios de robustez: (nombre, archivo de datos/original/ que lo trae, quién lo produce, tarjeta, a todo el ancho)
ESTUDIOS = [
    ("La cuantificación de la incertidumbre en regresión", "modelo_regresion.json", "el artefacto de BDL de regresión", _tarjeta_incertidumbre, True),
    ("La ablación", "ablacion.csv", "07_ablacion.py", _tarjeta_ablacion, True),
    ("La estabilidad por semilla", "semillas.csv", "03b_semillas.py", _tarjeta_semillas, False),
    ("La sensibilidad a los hiperparámetros propios", "robustez/sensibilidad.csv", "06_robustez.py", _tarjeta_sensibilidad, False),
    ("El cambio de distribución (una ciudad fuera cada vez)", "robustez/ciudades.csv", "06_robustez.py", _tarjeta_ciudades, False),
    ("El ruido y los valores atípicos", "robustez/ruido.csv", "06_robustez.py", _tarjeta_ruido, True),
    ("El tamaño de muestra y la escalabilidad", "robustez/muestra.csv", "06_robustez.py", _tarjeta_muestra, True),
    ("El análisis de errores por subgrupo", "errores_subgrupos.csv", "src/informe.py (la pieza «errores» del libro)", _tarjeta_subgrupos, True),
    ("Los conjuntos de datos adicionales", "adicionales.csv", "05_conjuntos_adicionales.py", _tarjeta_adicionales, True),
]


def _seccion_robustez():
    """Una tarjeta por estudio que ya llegó y, para los que faltan, un solo aviso que dice cuáles son y de dónde saldrán."""
    cuerpo, medias, faltan = [], [], []

    def vaciar():
        """Pone las tarjetas de media anchura que hay en espera, de dos en dos (una sola ocupa todo el ancho)."""
        for i in range(0, len(medias), 2):
            par = medias[i:i + 2]
            cuerpo.append(ui.rejilla(par, "1-1") if len(par) == 2 else par[0])
        medias.clear()

    for nombre, archivo, quien, tarjeta, ancha in ESTUDIOS:
        try:
            t = tarjeta()
        except Exception as error:  # noqa: BLE001  (un archivo con otro esquema no tumba la vista: lo dice en un aviso)
            t, ancha = ui.aviso(f"{nombre}: {_fo(archivo)} existe, pero no se pudo leer ({type(error).__name__}: {error}). Vuelva a "
                                "correr preparar_original.py; si persiste, el archivo de la corrección cambió de formato.",
                                "critico", "Este estudio no se pudo leer"), True
        if t is None:
            faltan.append([nombre, _fo(archivo), quien])
        elif ancha:
            vaciar()
            cuerpo.append(t)
        else:
            medias.append(t)
    vaciar()
    if faltan:
        n = len(faltan)
        cuerpo.append(ui.aviso([
            html.P(f"{_mayuscula(_cardinal_m(n))} de los {CARDINAL.get(len(ESTUDIOS), F.entero(len(ESTUDIOS)))} estudios "
                   f"todavía no {'ha' if n == 1 else 'han'} llegado. Cada uno aparecerá aquí cuando exista su archivo y se vuelva a "
                   "correr preparar_original.py; la app los lee al reiniciarse."),
            ui.tabla(["Estudio pendiente", "Archivo que lo trae", "Lo produce"], faltan, compacta=True, num=[False, False, False]),
        ], "atencion", "Estudios de robustez pendientes"))
    hechos = len(ESTUDIOS) - len(faltan)
    return ui.seccion(f"Robustez: {_cardinal_m(hechos)} de los {CARDINAL.get(len(ESTUDIOS), F.entero(len(ESTUDIOS)))} "
                      "estudios del protocolo ya tienen resultados" if faltan else
                      "Robustez: qué pasa al quitar piezas y al cambiar la semilla, los hiperparámetros, el ruido, la muestra y los datos",
                      cuerpo,
                      subtitulo="Resumen de los estudios de robustez del protocolo. Salvo la evaluación en prueba, todo se hace dentro del entrenamiento, "
                                f"con los pliegues externos de la semilla {REPETICIONES[0]} y los hiperparámetros finales de cada modelo.",
                      fuentes=[PROTOCOLO, f"{LIBRO}: capítulo 7, «Evaluación ampliada y robustez»"])


# --- 11. Veredicto -------------------------------------------------------------------------------------------------------
CHIP_ESTADO = {"cumple": ("se cumple", "bien", None), "no cumple": ("no se cumple", None, "cerrar"),
               "provisional": ("provisional", "atencion", None), "pendiente": ("pendiente", "atencion", None)}
VEREDICTO_TABLA = {"superior": "Superior", "no superior": "No superior", "pendiente": "Pendiente de la cuarta condición"}


def _chip_estado(estado):
    """El chip de una condición: se cumple, no se cumple, provisional (con parte de los resultados) o pendiente."""
    texto, tipo, icono = CHIP_ESTADO[estado]
    return ui.chip(texto, tipo, icono=icono)


def _texto_cuarta(tarea):
    """La cuarta condición en una tarea, con lo que diga condicion4.json: pendiente, provisional (con cuántos conjuntos) o
    su resumen."""
    c4 = D.condicion4(tarea)
    if not c4:
        return "los conjuntos adicionales todavía no han dado resultados"
    resumen = str(c4.get("resumen", "")).strip()
    if not c4.get("completo"):
        return (f"con {F.entero(c4.get('conjuntos', 0))} de {F.entero(c4.get('esperados', CONJUNTOS_ADICIONALES))} conjuntos: "
                f"{resumen}")
    return resumen


def _textos_condiciones(tarea):
    """El texto de cada una de las cuatro condiciones en una tarea, con sus valores."""
    c = TAREA[tarea]
    r = D.veredicto_original(tarea)
    cuatro = _mayuscula(_texto_cuarta(tarea))
    if r is None:
        return ["Faltan las pruebas de hipótesis"] * (CONDICIONES - 1) + [cuatro]
    mejor, dm, rechazos = r["mejor_benchmark"], r["frente_al_mejor"], r["rechazos_frente_al_mejor"]
    uno = (f"{c['metrica']} externo de {ui.num(r['medias'][O], c['d'])} frente a {ui.num(r['medias'][mejor], c['d'])} "
           f"{_tras('de', D.en_frase(mejor))}: ventaja de {ui.con_signo(r['ventaja_sobre_el_mejor'], c['d'] + 1)}")
    detalle = "; ".join(f"{k}, {_p(dm[k].get('p_holm'))}" for k in PRUEBAS if k in dm)
    dos = f"{_mayuscula(_rechazan(sum(bool(v) for v in rechazos.values()), len(rechazos)))} a favor de BDL (p ajustados: {detalle})"
    faltan = _faltan_pruebas(tarea)
    if faltan:
        dos += f". Faltan {F.enumerar(faltan)}, que se calculan sobre la prueba"
    if not r["condicion_1_media_mejor"]:
        dos += ". La condición exige, además, que la media sea mejor"
    efecto = f"δ de Cliff {ui.con_signo(dm['cliff'], 2)} y d de Cohen {ui.con_signo(dm['cohen'], 2)}"
    if r["condicion_3_efecto"]:
        tres = f"{efecto}: a favor de BDL y sobre los umbrales"
    elif dm["cliff"] < 0 or dm["cohen"] < 0:
        tres = f"{efecto}: el efecto va en contra de BDL"
    else:
        tres = f"{efecto}: no llega a los umbrales"
    return [_mayuscula(uno), dos, tres, cuatro]


def _tarjeta_condiciones():
    nombres = ["Su métrica media en los pliegues externos es mejor que la del mejor benchmark",
               f"Al menos {CARDINAL[PRUEBAS_MINIMAS]} de las pruebas P1 a P4 rechazan la igualdad con el mejor benchmark a favor del "
               f"original, con p ajustado menor que {ui.num(D.ALFA, 2)} (en P1 deben rechazar la prueba global y la comparación por pares)",
               f"El tamaño del efecto a favor del original no es despreciable: δ de Cliff ≥ {ui.num(UMBRAL_CLIFF, 3)} o d de Cohen ≥ "
               f"{ui.num(UMBRAL_COHEN, 1)}, entre los valores por pliegue",
               f"En los conjuntos adicionales es mejor en la métrica media en al menos {CARDINAL[CONJUNTOS_MINIMOS]} de los "
               f"{CARDINAL[CONJUNTOS_ADICIONALES]} y no es significativamente peor en ninguno"]
    estados = {t: _estado_veredicto(t) for t in TAREAS}
    textos = {t: _textos_condiciones(t) for t in TAREAS}
    filas = []
    for i, nombre in enumerate(nombres):
        fila = [f"{i + 1}. {nombre}"]
        for t in TAREAS:
            if i == CONDICIONES - 1:
                estado = D.estado_condicion4(t)
            elif estados[t] is None:
                estado = "pendiente"
            else:
                estado = "cumple" if estados[t]["cond"][i] else "no cumple"
            fila.append([_chip_estado(estado), html.Br(), html.Span(textos[t][i], className="tinta-2")])
        filas.append(fila)
    total = ["Veredicto"] + ["Pendiente de las pruebas" if estados[t] is None else VEREDICTO_TABLA[estados[t]["palabra"]] for t in TAREAS]
    refs = {D.veredicto_original(t)["mejor_benchmark"] for t in TAREAS if D.veredicto_original(t) is not None}
    if len(refs) == 1:
        mejor = f"El mejor benchmark, por la métrica media externa, es {D.en_frase(next(iter(refs)))} en las dos tareas."
    elif refs:
        mejor = "El mejor benchmark, por la métrica media externa, es " + " y ".join(
            f"{D.en_frase(D.veredicto_original(t)['mejor_benchmark'])} en {TAREA[t]['que']}" for t in TAREAS
            if D.veredicto_original(t) is not None) + "."
    else:
        mejor = "El mejor benchmark se sabrá con las pruebas de hipótesis."
    notas = ["En las condiciones 1 y 3 el signo es el de la ventaja de BDL (positivo: a su favor). En la 2 cuentan las pruebas que "
             "rechazan a favor de BDL con p ajustado menor que α. La cuarta es provisional mientras no terminen los "
             f"{CARDINAL[CONJUNTOS_ADICIONALES]} conjuntos adicionales con los diez modelos."]
    provisional = [f"{TAREA[t]['que']}, {F.entero(estados[t]['repeticiones'])} de {F.entero(len(REPETICIONES))} repeticiones "
                   f"({F.entero(D.veredicto_original(t)['pliegues'])} pliegues)" for t in TAREAS
                   if estados[t] is not None and estados[t]["provisional"]]
    if provisional:
        notas.append(f"Resultado provisional: {'; '.join(provisional)}. El protocolo fija el veredicto con las "
                     f"{CARDINAL[len(REPETICIONES)]} repeticiones.")
    return ui.tarjeta([ui.tabla(["Condición", D.NOMBRE_TAREA["clasificacion"], D.NOMBRE_TAREA["regresion"]], filas, total=total,
                                compacta=True, num=[False, False, False])] + [ui.nota(n) for n in notas],
                      titulo="Las cuatro condiciones de la guía, una por una",
                      subtitulo=f"Definición operativa de superioridad (§6.4), por separado en cada tarea. {mejor}",
                      fuentes=[_fo("veredicto.json"), _fo("condicion4.json"), PROTOCOLO])


def _por_que(tarea):
    """Por qué el veredicto de una tarea es el que es: las condiciones que fallan, con sus valores."""
    c = TAREA[tarea]
    e, r = _estado_veredicto(tarea), D.veredicto_original(tarea)
    if e is None:
        return f"En {c['que']} el veredicto está pendiente: faltan las pruebas de hipótesis."
    dm, mejor = r["frente_al_mejor"], r["mejor_benchmark"]
    cond = e["cond"]
    if e["palabra"] == "superior":
        return (f"En {c['que']} BDL es superior con la regla de la guía: cumple las {CARDINAL[CONDICIONES]} condiciones frente "
                f"{_tras('a', D.en_frase(mejor))}.")
    if e["palabra"] == "pendiente":
        cuarta = ("los conjuntos adicionales todavía no han dado resultados" if e["c4"] == "pendiente"
                  else f"la cuarta es provisional ({_texto_cuarta(tarea)})")
        return (f"En {c['que']} BDL cumple las tres primeras condiciones frente {_tras('a', D.en_frase(mejor))} y el veredicto queda "
                f"pendiente: {cuarta}.")
    causas = []
    if not cond[0]:
        causas.append(f"su {c['metrica']} medio externo no es mejor que el {_tras('de', D.en_frase(mejor))} "
                      f"({ui.con_signo(r['ventaja_sobre_el_mejor'], c['d'] + 1)}; a favor de BDL en {F.entero(dm['pliegues_a_favor'])} de "
                      f"{F.entero(r['pliegues'])} pliegues)")
    if not cond[1]:
        rechazos = r["rechazos_frente_al_mejor"]
        causa = f"{_rechazan(sum(bool(v) for v in rechazos.values()), len(rechazos))} la igualdad a su favor"
        if cond[0]:
            causa += f" y hacen falta {CARDINAL[PRUEBAS_MINIMAS]}"
        faltan = _faltan_pruebas(tarea)
        if faltan:
            causa += f" ({F.enumerar(faltan)} todavía no se han calculado)"
        causas.append(causa)
    if not cond[2]:
        causas.append("el tamaño del efecto va en su contra" if dm["cliff"] < 0 or dm["cohen"] < 0
                      else "el tamaño del efecto frente a él es despreciable")
    if e["c4"] == "no cumple":
        causas.append(f"no se sostiene en los conjuntos adicionales ({_texto_cuarta(tarea)})")
    return f"En {c['que']} BDL no es superior con la regla de la guía: " + F.enumerar(causas) + "."


def _en_prueba(tarea):
    """BDL y el mejor benchmark en la prueba: su puesto entre los diez y su distancia al límite teórico. None si falta algo."""
    c = TAREA[tarea]
    r, te = D.veredicto_original(tarea), _techo()
    valores = D.lideres(tarea).set_index("modelo")[c["prueba"]].astype(float)
    if r is None or te is None or O not in valores.index or r["mejor_benchmark"] not in valores.index:
        return None
    mejor = r["mejor_benchmark"]
    orden = valores.sort_values(ascending=not c["mayor"]).index.tolist()
    limite = float(te["prueba"]["auc_maximo" if tarea == "clasificacion" else "rmse_minimo"])
    return (f"En la prueba, BDL obtiene {ui.num(valores[O], c['d'])} (puesto {F.entero(orden.index(O) + 1)} de {F.entero(len(orden))}) "
            f"y {D.en_frase(mejor)}, {ui.num(valores[mejor], c['d'])}; el límite teórico es {ui.num(limite, c['d'])}, así que el benchmark "
            f"queda a {ui.num(abs(valores[mejor] - limite), c['d'] + 1)} del límite y BDL, a {ui.num(abs(valores[O] - limite), c['d'] + 1)}.")


def _uso_del_puntaje():
    """En cuántos pliegues externos la búsqueda interna del clasificador eligió la variante que no usa el puntaje continuo
    (usar_puntaje = False) y con qué AUC, frente al mejor benchmark: el análisis de la causa del libro (capítulo 6). None si
    la tabla maestra no trae los hiperparámetros de cada pliegue."""
    r, p = D.veredicto_original("clasificacion"), D.pliegues("clasificacion")
    if r is None or p is None:
        return None
    con, sin = [], []
    for semilla in r.get("repeticiones") or []:
        f = D.corrida_original("clasificacion", O, semilla)
        por_pliegue = None if f is None else f.get("params_por_pliegue")
        if not por_pliegue:
            return None
        for i, elegidos in enumerate(por_pliegue):
            (con if elegidos.get("modelo__usar_puntaje", True) else sin).append(f"{semilla}-{i + 1}")
    mejor = r["mejor_benchmark"]
    if not set(con + sin) <= set(p.index) or mejor not in p.columns or O not in p.columns:
        return None
    final = D.corrida_original("clasificacion", O, REPETICIONES[0])

    def media(filas, modelo):
        return float(p.loc[filas, modelo].mean()) if filas else float("nan")

    return {"con": len(con), "sin": len(sin), "original_con": media(con, O), "original_sin": media(sin, O),
            "benchmark_con": media(con, mejor), "benchmark_sin": media(sin, mejor),
            "gana_con": int((p.loc[con, O] > p.loc[con, mejor]).sum()) if con else 0,
            "final_usa_puntaje": None if final is None else bool((final["hiperparametros"] or {}).get("modelo__usar_puntaje", True))}


def _parte_que_cierra(tarea):
    """La parte de la distancia entre el mejor benchmark y el límite teórico que recorre BDL en la prueba, en la métrica
    principal de la tarea (negativa si BDL queda más lejos del límite que el benchmark). None si falta algo."""
    c = TAREA[tarea]
    te, r = _techo(), D.veredicto_original(tarea)
    t = D.lideres(tarea).set_index("modelo")
    if te is None or r is None or O not in t.index or r["mejor_benchmark"] not in t.index:
        return None
    limite = float(te["prueba"]["auc_maximo" if tarea == "clasificacion" else "rmse_minimo"])
    original, benchmark = float(t.loc[O, c["prueba"]]), float(t.loc[r["mejor_benchmark"], c["prueba"]])
    falta = (limite - benchmark) if c["mayor"] else (benchmark - limite)
    if not _finito(original) or not _finito(falta) or abs(falta) <= 1e-12:
        return None
    return ((original - benchmark) if c["mayor"] else (benchmark - original)) / falta


def _optimo(tarea):
    """El análisis del predictor óptimo en una tarea (datos/original/optimo.json): las pruebas P3 y P4 del predictor del
    techo teórico frente al mejor benchmark. None si el archivo no existe, si no trae las dos pruebas o si se calculó frente
    a un benchmark que ya no es el mejor del veredicto (habría que repetirlo)."""
    o = (D.original_json("optimo.json") or {}).get(tarea)
    r = D.veredicto_original(tarea)
    pruebas = (o or {}).get("optimo") or {}
    if not all(_finito((pruebas.get(k) or {}).get("p_holm")) and "rechaza" in pruebas[k] for k in PRUEBAS_EN_PRUEBA):
        return None
    if r is not None and o.get("mejor_benchmark") != r["mejor_benchmark"]:
        return None
    return o


def _rechaza_por_poco(o):
    """El óptimo rechaza por poco si alguno de sus p ajustados pasa de la mitad de α (el criterio del libro)."""
    return max(o["optimo"][k]["p_holm"] for k in PRUEBAS_EN_PRUEBA) > D.ALFA / 2


def _frase_optimo(tarea):
    """Lo que dice el análisis del predictor óptimo sobre si la segunda condición era alcanzable en una tarea, como en el
    libro (sección 6.9). None si el análisis no está."""
    o = _optimo(tarea)
    if o is None:
        return None
    c = TAREA[tarea]
    nb = D.en_frase(o["mejor_benchmark"])
    ps = " y ".join(_p(o["optimo"][k]["p_holm"]) for k in PRUEBAS_EN_PRUEBA)
    sin_rechazo = sum(not o["optimo"][k]["rechaza"] for k in PRUEBAS_EN_PRUEBA)
    if sin_rechazo == len(PRUEBAS_EN_PRUEBA):
        return (f"En {c['que']}, ni el predictor óptimo (el del techo teórico, que ningún modelo puede alcanzar sin la variable "
                f"excluida) rechaza la igualdad con {nb} en las dos pruebas que se hacen con el conjunto de prueba (p ajustados {ps}): "
                f"ningún modelo podía reunir los {CARDINAL[PRUEBAS_MINIMAS]} rechazos que pide la guía.")
    if sin_rechazo:
        return (f"En {c['que']}, el predictor óptimo solo rechaza la igualdad con {nb} en una de las dos pruebas que se hacen con el "
                f"conjunto de prueba (p ajustados {ps}): un modelo tendría que rechazar en todas las demás.")
    cierra = _parte_que_cierra(tarea)
    cola = "" if cierra is None else f", y BDL cierra el {F.pct(cierra, 0)} de la distancia entre {nb} y ese óptimo"
    return (f"En {c['que']}, el predictor óptimo sí rechaza la igualdad con {nb}{', por poco' if _rechaza_por_poco(o) else ''} "
            f"(p ajustados {ps}): hacía falta estimar el índice casi sin error{cola}.")


def _tarjeta_optimo():
    """¿Podía algún modelo cumplir la segunda condición? Las pruebas P3 y P4 del predictor óptimo frente al mejor benchmark,
    junto a las de BDL, como la Tabla 6.10 del libro. None si el análisis no está en ninguna tarea."""
    hay = [t for t in TAREAS if _optimo(t) is not None]
    if not hay:
        return None
    filas, lecturas, metricas = [], [], []
    for t in hay:
        c, o, nombres, r = TAREA[t], _optimo(t), _nombres_pruebas(t), D.veredicto_original(t)
        for k in PRUEBAS_EN_PRUEBA:
            a = o["optimo"][k]
            b = ((r or {}).get("frente_al_mejor") or {}).get(k) or (o.get("original") or {}).get(k) or {}
            rechaza = (r["rechazos_frente_al_mejor"].get(k) if r is not None else b.get("rechaza"))
            filas.append([f"{k}, {c['que']}", nombres[k], _nombre(o["mejor_benchmark"]), _p(a.get("p")), _p(a.get("p_holm")),
                          "sí" if a.get("rechaza") else "no", _p(b.get("p_holm")), RAYA if not b else ("sí" if rechaza else "no")])
        sin_rechazo = sum(not o["optimo"][k]["rechaza"] for k in PRUEBAS_EN_PRUEBA)
        if sin_rechazo == len(PRUEBAS_EN_PRUEBA):
            lecturas.append(f"en {c['que']}, ni el predictor óptimo rechaza la igualdad con el mejor benchmark")
        elif sin_rechazo:
            lecturas.append(f"en {c['que']}, el óptimo solo rechaza en una de las dos pruebas")
        else:
            lecturas.append(f"en {c['que']}, el óptimo sí rechaza{', por poco' if _rechaza_por_poco(o) else ''}")
        if _finito(o.get("metrica_del_optimo")):
            metricas.append(f"un {c['metrica']} de {ui.num(o['metrica_del_optimo'], c['d'])}")
    comparaciones = sorted({int(_optimo(t)["comparaciones"]) for t in hay if _finito(_optimo(t).get("comparaciones"))})
    notas = ["El predictor óptimo es el del techo teórico: conoce el índice real de cada cliente, que se reconstruye con active_products "
             "(la variable excluida), y aplica la regla de Bayes. Ningún modelo puede alcanzarlo sin esa variable, así que lo que él no "
             "consiga no lo puede conseguir ninguno."
             + (f" En la prueba obtiene {' y '.join(metricas)}." if metricas else "")]
    notas.append(("Los p ajustados son de Holm" + (f" sobre las {F.entero(comparaciones[0])} comparaciones de cada prueba" if len(comparaciones) == 1 else "")
                  + f", como en el protocolo, y se rechaza con p ajustado menor que {ui.num(D.ALFA, 2)}. Es un análisis de la causa, "
                  "posterior a los resultados y fuera del protocolo: no cambia el veredicto, lo explica."))
    return ui.tarjeta([
        ui.tabla(["Prueba y tarea", "Qué prueba es", "Mejor benchmark", "p del óptimo", "p ajustado del óptimo",
                  "¿Rechaza el óptimo?", "p ajustado de BDL", "¿Rechaza BDL?"], filas, compacta=True,
                 num=[False, False, False, True, True, False, True, False], ocultar_movil=(1, 2, 3)),
    ] + [ui.nota(n) for n in notas],
        titulo="¿Podía algún modelo cumplir la segunda condición? " + _mayuscula("; ".join(lecturas)),
        subtitulo=f"Las dos pruebas que se hacen con el conjunto de prueba ({' y '.join(PRUEBAS_EN_PRUEBA)}), repetidas para el mejor "
                  "modelo posible.",
        fuentes=[_fo("optimo.json"), _fo("veredicto.json"), _fo("techo.json"), LIBRO_VALIDACION])


def _causa(tarea):
    """El análisis de la causa de un «no superior», como en el libro (capítulo 6): lo que muestran los pliegues y, si está
    el análisis del predictor óptimo, si la segunda condición era alcanzable. None si no hay con qué decirlo."""
    e = _estado_veredicto(tarea)
    if e is None or e["palabra"] != "no superior":
        return None
    return " ".join(x for x in (_causa_pliegues(tarea), _frase_optimo(tarea)) if x) or None


def _causa_pliegues(tarea):
    """La parte del análisis de la causa que sale de los pliegues externos y de la búsqueda interna. None si no hay con qué
    decirlo."""
    c = TAREA[tarea]
    e, r = _estado_veredicto(tarea), D.veredicto_original(tarea)
    if e is None or e["palabra"] != "no superior":
        return None
    mejor, dm, d = r["mejor_benchmark"], r["frente_al_mejor"], c["d"]
    if e["cond"][0] and not e["cond"][1]:
        limite = (": el mejor benchmark ya estaba casi en el límite del conjunto y no dejaba margen para una diferencia mayor."
                  if _techo() is not None else ".")              # lo del límite solo se dice si techo.json lo respalda
        return (f"Causa: la diferencia es sistemática (BDL va por delante en {F.entero(dm['pliegues_a_favor'])} de los "
                f"{F.entero(r['pliegues'])} pliegues), pero tan pequeña que, con la corrección por comparaciones múltiples, no todas las "
                f"pruebas alcanzan a detectarla{limite}")
    if e["cond"][0] or tarea != "clasificacion":
        return None
    u = _uso_del_puntaje()
    if u is None:
        return None
    if not u["sin"]:
        return ("Causa: el modelo usó el puntaje en todos los pliegues y aun así no supera al mejor benchmark en la media externa: en "
                "esta tarea los benchmarks de árboles ya ordenan a los clientes tan bien como permite el conjunto.")
    nb = D.en_frase(mejor)
    texto = ("Análisis de la causa: el espacio de búsqueda incluía la variante que aprende solo de la etiqueta (usar_puntaje = False), y "
             f"la búsqueda interna la eligió en {F.entero(u['sin'])} de los {F.entero(u['con'] + u['sin'])} pliegues externos. En esos "
             f"pliegues BDL obtuvo un AUC medio de {ui.num(u['original_sin'], d)} y {nb}, {ui.num(u['benchmark_sin'], d)} (diferencia: "
             f"{ui.con_signo(u['original_sin'] - u['benchmark_sin'], d)})")
    if u["con"]:
        dif_con, dif_sin = u["original_con"] - u["benchmark_con"], u["original_sin"] - u["benchmark_sin"]
        texto += (f"; en los {F.entero(u['con'])} en que usó el puntaje, {ui.num(u['original_con'], d)} frente a "
                  f"{ui.num(u['benchmark_con'], d)} (diferencia: {ui.con_signo(dif_con, d)}; por delante en {F.entero(u['gana_con'])} de "
                  f"{F.entero(u['con'])}).")
        if dif_con > dif_sin:
            texto += " La desventaja en la media viene sobre todo de los pliegues en que el modelo no usó el puntaje."
    else:
        texto += "."
    if u["final_usa_puntaje"] is not None:
        texto += (" El modelo final, reajustado con toda la partición de entrenamiento, sí usa el puntaje." if u["final_usa_puntaje"]
                  else " El modelo final tampoco usa el puntaje.")
    return (texto + f" En cualquier caso, estas diferencias son menores que la desviación entre pliegues del propio benchmark "
            f"({ui.num(r['desviaciones'][mejor], d)}): con estos datos, los primeros modelos de la tabla no se distinguen entre sí en "
            "esta tarea. La ablación mide por separado la variante sin puntaje.")


def _seccion_veredicto():
    estados = {t: _estado_veredicto(t) for t in TAREAS}
    hay = [t for t in TAREAS if estados[t] is not None]
    if not hay:
        return ui.seccion("Veredicto: pendiente de las pruebas de hipótesis", [
            _pendiente("El veredicto", "veredicto.json", "lo calcula 04_pruebas_hipotesis.py"), _tarjeta_condiciones()],
            subtitulo="El veredicto explícito de superioridad o no superioridad, por tarea, con las cuatro condiciones de la guía.")
    partes = [f"{D.VEREDICTO_ORIGINAL[estados[t]['palabra']]} en {TAREA[t]['que']}" for t in hay]
    provisional = any(estados[t]["provisional"] for t in hay)
    valor = " y ".join(f"{F.entero(estados[t]['cumplidas'])} de {F.entero(estados[t]['evaluadas'])}" for t in hay)
    te = _techo()
    techo = ""
    if te is not None:
        techo = (f" Los mejores benchmarks ya estaban a milésimas del techo teórico (AUC máximo de {ui.num(te['prueba']['auc_maximo'], 4)} "
                 f"y RMSE mínimo de {ui.num(te['prueba']['rmse_minimo'], 5)} en la prueba): cualquier mejora es pequeña.")
    insight = ui.tarjeta_insight(valor, "condiciones de la regla de superioridad que cumple BDL, de las evaluadas, en "
                                 + " y en ".join(TAREA[t]["que"] for t in hay),
                                 f"Veredicto{' provisional' if provisional else ''}: " + "; ".join(partes) + ".", chip_="Veredicto")
    parrafos = []
    for t in TAREAS:
        parrafos.append(" ".join(x for x in (_por_que(t), _en_prueba(t) if estados[t] is not None else None) if x))
        causa = _causa(t)
        if causa:
            parrafos.append(causa)
    faltan = (["las repeticiones restantes de la validación anidada"] if provisional else []) \
        + (["las pruebas sobre el conjunto de prueba"] if any(_faltan_pruebas(t) for t in hay) else []) \
        + (["los conjuntos adicionales que no han terminado"] if any(estados[t]["c4"] in ("pendiente", "provisional") for t in hay) else [])
    cierre = []
    if faltan:
        cierre.append(f"Este veredicto {'es provisional: ' if provisional else ''}se calculó con los resultados disponibles y puede "
                      f"cambiar cuando lleguen {F.enumerar(faltan)}. La app muestra siempre lo que dicen los archivos.")
    lectura = ui.aviso(parrafos + [
        "Lo que sí sostiene la evidencia se lee condición por condición: dónde la media es mejor, qué pruebas rechazan y con qué tamaño "
        f"del efecto. No se afirma más que eso.{techo}"] + cierre, None, "Lo que dice y lo que no dice el veredicto")
    optimo = _tarjeta_optimo()
    if optimo is None:
        optimo = _pendiente("El análisis del predictor óptimo", "optimo.json", "lo calcula 04c_optimo.py")
    return ui.seccion("Veredicto con la regla de la guía: " + "; ".join(partes) + (" (resultado provisional)" if provisional else ""), [
        ui.rejilla([insight, _tarjeta_condiciones()], "1-2"),
        lectura,
        optimo,
        ui.destinos([ui.destino(app.ruta_vista(v), v, n, app.pregunta(v)) for v, n in (("comparacion", "Comparación estadística"),
                                                                                       ("lideres", "Tabla de líderes"))]),
    ], subtitulo="El veredicto explícito de superioridad o no superioridad, por tarea, con las cuatro condiciones de la guía (§6.4) y "
                 "los valores reales.",
        fuentes=[_fo("veredicto.json"), _fo("condicion4.json"), _fo("experimentos.csv")]
        + ([_fo("optimo.json")] if D.hay_original("optimo.json") else []) + [PROTOCOLO])


# --- La pantalla ----------------------------------------------------------------------------------------------------------
def layout(tarea=None, **_):
    """`/modelos/original?tarea=clasificacion` abre con el modelo final de esa tarea en las secciones que dependen de ella
    (los niveles, los valores del algoritmo, la convergencia, el índice y las pruebas)."""
    pedida = tarea in TAREA
    inicial = tarea if pedida else _tarea_inicial()
    control = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", OPCIONES_TAREA, inicial, persistir=not pedida))
    aviso = _aviso_pendientes()
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), _respuesta(), [control], [aviso] if aviso is not None else None, pantalla=PANTALLA),
        _seccion_descripcion(inicial),
        _seccion_formulacion(),
        _seccion_algoritmo(inicial),
        _seccion_teoria(),
        _seccion_novedad(),
        _seccion_convergencia(inicial),
        _seccion_indice(inicial),
        _seccion_techo(),
        _seccion_pruebas(inicial),
        _seccion_robustez(),
        _seccion_veredicto(),
    ])


@callback(Output(f"{PRE}-desc", "children"), Output(f"{PRE}-alg", "children"), Output(f"{PRE}-conv", "children"),
          Output(f"{PRE}-indice", "children"), Output(f"{PRE}-pruebas", "children"), Input(f"{PRE}-tarea", "value"))
def _por_tarea(tarea):
    """Lo que depende del modelo final de la tarea elegida."""
    tarea = tarea if tarea in TAREA else _tarea_inicial()
    return (_descripcion_tarea(tarea), _valores_algoritmo(tarea), _convergencia_tarea(tarea), _indice_tarea(tarea),
            _pruebas_tarea(tarea))
