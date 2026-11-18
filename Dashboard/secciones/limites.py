"""Límites · ¿Qué no se puede concluir y dónde está el resto?

Lo que el informe declara como limitación, qué pruebas se fijaron antes de ver la prueba y cuáles después, cómo se
reproduce el trabajo y dónde están el informe, el código y los datos. Los textos parafrasean el último apartado de la
sección de resultados y el último párrafo de las conclusiones del informe final; las cifras salen de datos/.
"""
import math

import dash
from dash import html

import app
import datos_app as D
import formato as F
import motor as M
import ui


PRE = "li"
RIVAL = "xgboost"
NIVEL_HABITUAL = 0.95        # la probabilidad de equivalencia que suele exigirse
NIVEL_MCS = 0.90             # nivel del Model Confidence Set del informe
REPOSITORIO = "https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech"
SITIO = "https://rubenesg.github.io/Prediccion-de-fuga-de-clientes-en-fintech/"
CONJUNTO = "https://data.mendeley.com/datasets/mhb4zn3258/1"
# clave de /documento/<clave>, título, ruta en el repositorio y una línea
DOCUMENTOS = [
    ("informe", "Informe final", "InformeFinal/informe/main.pdf", "Introducción, metodología, resultados y conclusiones, en PDF."),
    ("metodologia", "Sección de metodología", "InformeFinal/metodologia/main.pdf", "Datos, ETL, modelos, validación anidada y protocolo de evaluación."),
    ("resultados", "Sección de resultados", "InformeFinal/resultados/main.pdf", "Todas las tablas, las pruebas de significancia y la ablación."),
]
SUPERINDICES = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


ANIO = {"datos": 2025, "ebm": 2012, "pares": 2013, "interpret": 2019}      # años de las referencias, como en el informe


def _orden_de_magnitud(x):
    """«10⁻¹⁶» para una diferencia de redondeo de máquina."""
    return "10" + str(int(math.floor(math.log10(x)))).translate(SUPERINDICES) if x and x > 0 else "0"


def _limites():
    clf, reg = D.pruebas("clasificacion"), D.prueba("regresion")
    corridas = D.tabla("entrega3/corridas.csv")
    pliegues = int(corridas["k_ext"].iloc[0])
    rope_x = D.fila(clf, modelo=RIVAL)["p_rope"]
    mcs_ebm = D.fila(reg, modelo="ebm")["p_mcs"]
    nb = D.fila(clf, modelo="naive_bayes")
    separa = [D.NOMBRE_MODELO[m] for m in clf.loc[clf["p_mcnemar_holm"] < D.ALFA, "modelo"]]
    xgb = D.NOMBRE_MODELO[RIVAL]
    puntos = [
        ("Un solo conjunto de datos, con rasgos sintéticos",
         "La edad solo toma seis valores, en tres cohortes, y el puntaje baja en escalera con el número de productos activos. "
         "Nada garantiza que el resultado se repita con los clientes de otra fintech."),
        ("El objetivo no es fuga observada",
         "«Riesgo alto» es el cuarto superior de un puntaje de propensión que trae el conjunto de datos. Lo que el modelo "
         "aprendió es a reproducir ese puntaje: el AUC dice qué tan bien ordena a los clientes según él, no cuántos se van."),
        (f"Equivalente a {xgb}, no mejor",
         f"En clasificación la probabilidad de equivalencia práctica con {xgb} es {ui.num(rope_x, 2)}, por debajo del "
         f"{F.pct(NIVEL_HABITUAL, 0)} habitual. En regresión el Model Confidence Set al {F.pct(NIVEL_MCS, 0)} deja solo a {xgb} "
         f"(la EBM sale con {F.p_valor(mcs_ebm)})."),
        ("Pruebas con poca potencia",
         f"Con {pliegues} pliegues la t corregida tiene {pliegues - 1} grados de libertad. McNemar, en el punto de operación del "
         f"{F.pct(M.CAPACIDAD_INFORME, 0)}, no ve diferencias que el AUC sí ve: solo separa a {F.enumerar(separa)} "
         f"({F.p_valor(nb['p_mcnemar_holm'])} ajustado)."),
        ("Lo que no se probó",
         "El SVM y el SVR son solo lineales, la regresión logística cubre las penalizaciones L1 y L2 como un solo modelo, la "
         "regresión no incluye un modelo bayesiano y los tiempos son de un solo equipo."),
        ("No puede descartarse una fuga residual",
         "Las variables derivadas no usan el objetivo y la variable que lo genera se excluyó, pero en un conjunto sintético "
         "puede quedar información del objetivo en otra variable."),
        ("El simulador no es un experimento",
         "Al cambiar una variable en la pantalla Cliente se ve lo que el modelo asigna a otro cliente hipotético, no lo que "
         "pasaría si se interviene sobre un cliente real."),
    ]
    return [ui.tarjeta([ui.parrafo(texto)], titulo=titulo) for titulo, texto in puntos]


def _protocolo():
    rope = D.resumen()["rope"]
    antes = ui.ficha([
        ("Métrica principal", "AUC en clasificación y RMSE en regresión"),
        ("Nivel de significancia", f"α = {ui.num(D.ALFA, 2)}"),
        ("Comparaciones múltiples", "corrección de Holm sobre las siete referencias de cada prueba"),
        ("Pruebas", "DeLong, Diebold-Mariano y Model Confidence Set"),
    ])
    despues = ui.ficha([
        ("Pruebas añadidas", "t corregida de Nadeau-Bengio, McNemar y comparación bayesiana"),
        ("Banda de equivalencia", f"±{ui.num(rope['auc'], 2)} de AUC y ±{ui.num(rope['rmse'], 4)} de RMSE, fijada antes de ejecutar la bayesiana"),
        ("Análisis", "los estudios de sensibilidad (hiperparámetros, curva de aprendizaje y concordancia entre validación y prueba)"),
    ])
    return [
        ui.tarjeta([antes], titulo="Fijado antes de evaluar la prueba",
                   subtitulo="Lo que estas pruebas dicen es confirmatorio."),
        ui.tarjeta([despues], titulo="Añadido después de ver la prueba",
                   subtitulo="El informe lo declara posterior y exploratorio."),
    ]


def _reproducir():
    meta = M.meta()
    corridas = D.tabla("entrega3/corridas.csv")
    equipo = D.json("entrega3/computo/equipo.json")
    no_aplica = int((corridas["estado"] != "ok").sum())
    v = meta["versiones"]
    ver = meta["verificacion"]["ebm_clasificacion"]
    hechos = ui.ficha([
        ("Semilla", str(meta["semilla"])),
        ("Partición", f"{F.entero(meta['n_entrenamiento'])} clientes de entrenamiento y {F.entero(meta['n_prueba'])} de prueba"),
        ("Experimentos", f"{F.entero(len(corridas))} corridas ({no_aplica} no aplican) en {ui.num(corridas['tiempo_total_s'].sum() / 3600, 1)} horas"),
        ("Equipo", f"{equipo['nucleos_fisicos']} núcleos, {ui.num(equipo['memoria_gb'], 1)} GB de memoria, Python {equipo['python']}"),
        ("Versiones", f"scikit-learn {v['sklearn']}, interpret-core {v['interpret']}, numpy {v['numpy']} y pandas {v['pandas']}"),
        ("Motor de la app", f"igual al modelo guardado en {F.entero(ver['filas_comparadas'])} clientes"),
        ("Diferencia máxima", f"del orden de {_orden_de_magnitud(ver['max_dif_motor_frente_al_modelo_guardado'])}"),
    ])
    comandos = ui.ficha([
        ("El motor frente al modelo guardado y al informe", html.Code("python verificar_motor.py")),
        ("Las pantallas y sus callbacks, sin navegador", html.Code("python probar_app.py")),
        ("Las reglas de diseño", html.Code("python verificar_estilo.py")),
        ("Rehacer los datos de la app desde el proyecto", html.Code("python preparar_datos.py")),
    ])
    fuentes = [D.fuente("metadatos.json"), D.fuente("entrega3/corridas.csv"), D.fuente("entrega3/computo/equipo.json"), D.fuente("origen.json")]
    return [ui.tarjeta([hechos], titulo="Con qué se hizo", fuentes=fuentes),
            ui.tarjeta([comandos, ui.nota("Cada archivo de datos/ tiene anotado su origen y su huella en datos/origen.json.")],
                       titulo="Cómo se comprueba")]


def _futuro():
    pasos = [
        "Validar en el tiempo con fuga observada: entrenar con un periodo y probar con el siguiente.",
        "Incorporar el costo de cada acción de retención para fijar la capacidad y el umbral de la campaña.",
        "En el modelo original, hacer que los pesos de las clases latentes dependan de los predictores (una compuerta aprendida por "
        "boosting) y permitir una dispersión distinta por clase.",
        "Suavizar los bordes de las mesetas de su probabilidad eligiendo la dispersión con una regla de puntuación propia, sin perder "
        "el orden de los clientes.",
        "Dar a sus conjuntos de predicción una cobertura condicionada al nivel del escalón, no solo sobre el total de clientes.",
        "Compilar el bucle de histogramas del modelo original para reducir su tiempo de ajuste.",
        "Tratar las transacciones como series, con modelos de secuencias.",
    ]
    return ui.tarjeta([ui.parrafo(p) for p in pasos],
                      titulo="Trabajo futuro: lo que proponen el libro del proyecto final y el informe del Entregable 3")


def _destinos():
    hay = app.documentos_disponibles()
    docs = []
    for clave, titulo, en_repo, texto in DOCUMENTOS:
        href = f"/documento/{clave}" if hay.get(clave) else f"{REPOSITORIO}/blob/main/{en_repo}"
        docs.append(ui.destino(href, "limites", titulo, texto, externo=True, icono="descargar"))
    fuera = [
        ui.destino(REPOSITORIO, "limites", "Repositorio", "El código de las tres entregas, los cuadernos, los resultados y esta app.",
                   externo=True, icono="externo"),
        ui.destino(SITIO, "limites", "Libro en línea", "Las tres entregas del proyecto, página por página.", externo=True, icono="externo"),
        ui.destino(CONJUNTO, "datos", "Conjunto de datos COFINFAD", "Mendeley Data, versión 1, licencia CC BY 4.0.", externo=True),
    ]
    return docs + fuera


def layout(**_):
    clf = D.pruebas("clasificacion")
    documentos = _destinos()
    subtitulo = (f"No se puede concluir que la EBM sea mejor que {D.NOMBRE_MODELO[RIVAL]} (la diferencia de AUC es "
                 f"{ui.con_signo(D.fila(clf, modelo=RIVAL)['delta_auc'], 4)}, dentro de la banda de equivalencia) ni que esto prediga "
                 f"fuga real: hay un único conjunto de datos y el objetivo es el puntaje de otro modelo.")
    return ui.pantalla([
        ui.encabezado("¿Qué no se puede concluir y dónde está el resto?", subtitulo, pantalla="limites"),
        ui.seccion("Lo que este trabajo no permite afirmar", [ui.destinos(_limites())],
                   subtitulo="Las limitaciones que declara el informe, cada una con su consecuencia.",
                   fuentes=[D.fuente("resultados/pruebas_clasificacion.csv"), D.fuente("resultados/regresion_prueba.csv"),
                            D.fuente("entrega3/corridas.csv")]),
        ui.seccion("Qué se decidió antes de ver la prueba y qué después", [ui.rejilla(_protocolo(), "1-1")],
                   fuentes=[D.fuente("resultados/resumen.json")]),
        ui.seccion("Cómo se reproduce", [ui.rejilla(_reproducir(), "1-1")]),
        ui.seccion("Lo que sigue", [_futuro()]),
        ui.seccion("Dónde está el resto", [ui.destinos(documentos)],
                   subtitulo="El informe y sus dos secciones, el repositorio, el libro y los datos se abren en otra pestaña."),
        ui.pie([
            ui.nota("Predicción de la fuga de clientes de una fintech con una Explainable Boosting Machine. Valerie Martínez Pulido, "
                    "Abrahan Basto Martínez y Rubén Esguerra Fernández. Pregrado en Ciencia de Datos, Universidad del Norte, Barranquilla."),
            ui.nota(f"Datos: Muñoz Guerrero, L. E., Ceballos, Y. F. y Trejos Rojas, L. D. ({ANIO['datos']}). COFINFAD: Colombian Fintech "
                    f"Financial Analytics Dataset. Mendeley Data. El modelo: Lou, Caruana y Gehrke ({ANIO['ebm']}); Lou, Caruana, Gehrke y "
                    f"Hooker ({ANIO['pares']}); Nori, Jenkins, Koch y Caruana ({ANIO['interpret']}, InterpretML)."),
        ]),
    ])
