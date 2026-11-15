"""Ficha del modelo (MODEL_CARD.md) y declaración de los datos (DATA_STATEMENT.md), generadas desde datos/ y artifacts/.

    .venv/Scripts/python.exe -B generar_fichas.py        escribe los dos archivos en esta carpeta

Ninguna cifra se escribe a mano: todas salen de datos/ (por datos_app, motor y modelos_vivos), de artifacts/indice.json y de
CITATION.cff. La vista «Ficha del modelo» (secciones/ficha.py) arma la misma ficha con contenido_ficha(), así que la página y
el Markdown que se descarga en /ficha.md dicen lo mismo; si cambian los datos, se vuelve a correr este guion.

El formato sigue a Mitchell et al. (2019), «Model Cards for Model Reporting» (FAT* '19), para la ficha, y la idea de una
declaración de los datos de Bender y Friedman (2018) y Gebru et al. (2021, «Datasheets for Datasets»). Las afirmaciones son
las del informe final (Informe Final/metodologia.tex, resultados.tex y conclusiones.tex).

Cada contenido es una lista de grupos (un h2 en la página y un «##» en el Markdown); cada grupo tiene tarjetas (h3, «###»)
con bloques: ("parrafo", texto), ("lista", [textos]), ("pares", [(rótulo, valor)]), ("tabla", encabezados, filas, num, columnas que se ocultan en el teléfono) y
("nota", texto), y las fuentes de sus cifras.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

import datos_app as D  # noqa: E402
import formato as F  # noqa: E402
import modelos_vivos as V  # noqa: E402
import motor as M  # noqa: E402

TAREAS = ("clasificacion", "regresion")
RIVAL = "xgboost"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
NIVEL_CONFORME = "0.90"           # el nivel de los intervalos y conjuntos conformes con que nace la predicción en vivo
UMBRAL_MEDIO = 0.5                # el AUC del azar (la parte del techo se mide sobre él)
CAPACIDAD_MITAD = 0.5             # la capacidad de las columnas recall_50 de resultados/clasificacion_prueba.csv
PERIODO_DATOS = 2023              # el año que describe COFINFAD (informe final, metodologia.tex, «Fuente y tamaño»)
SUPERINDICE = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
HUELLA = 12                       # caracteres de la huella SHA-256 que se muestran (la completa está en los .json)
REFERENCIAS_FORMATO = ("Referencias del formato: Mitchell et al. (2019), «Model Cards for Model Reporting», FAT* '19, para la ficha; Bender y Friedman (2018) y Gebru et al. (2021, «Datasheets for Datasets») para la declaración de los datos.")
FICHA = AQUI / "MODEL_CARD.md"
DECLARACION = AQUI / "DATA_STATEMENT.md"
INFORME = "Informe final (metodologia.tex, resultados.tex, conclusiones.tex)"


# --- Lecturas --------------------------------------------------------------------------------------------------------------
def cita_cff() -> dict:
    """Lo que dice CITATION.cff: título, versión, fecha, licencia, repositorio, autores y el conjunto de datos citado."""
    texto = (AQUI / "CITATION.cff").read_text(encoding="utf-8")

    def campo(nombre, desde=0):
        m = re.compile(rf'^\s*{re.escape(nombre)}:\s*"?([^"\n]*)"?\s*$', re.M).search(texto, desde)
        return m.group(1).strip() if m else ""

    corte = texto.find("references:")
    propio = texto[:corte] if corte >= 0 else texto
    refs = texto[corte:] if corte >= 0 else ""
    autores = [f"{dado} {familia}" for familia, dado in
               re.findall(r'family-names:\s*"([^"]+)"\s*\n\s*given-names:\s*"([^"]+)"', propio)]
    autores_datos = [f"{familia}, {dado}" for familia, dado in
                     re.findall(r'family-names:\s*"([^"]+)"\s*\n\s*given-names:\s*"([^"]+)"', refs)]
    return {"titulo": campo("title"), "version": campo("version"), "fecha": campo("date-released"), "licencia": campo("license"),
            "repositorio": campo("repository-code"), "autores": autores,
            "datos": {"titulo": re.search(r'title:\s*"([^"]+)"', refs).group(1) if refs else "", "autores": autores_datos,
                      "anio": re.search(r"year:\s*(\d+)", refs).group(1) if refs else "",
                      "editorial": re.search(r'name:\s*"([^"]+)"', refs).group(1) if refs else "",
                      "version": re.search(r'version:\s*"([^"]+)"', refs).group(1) if refs else "",
                      "doi": re.search(r'doi:\s*"([^"]+)"', refs).group(1) if refs else "",
                      "licencia": re.search(r"license:\s*(\S+)", refs).group(1) if refs else ""}}


def indice() -> dict:
    return json.loads((AQUI / "artifacts" / "indice.json").read_text(encoding="utf-8"))


def _origen(nombre: str) -> dict:
    return next(a for a in D.json("origen.json")["archivos"] if a["archivo"] == nombre)


def _fecha(texto) -> str:
    """«1 de octubre de 2026» a partir de una fecha ISO."""
    a, m, d = (int(x) for x in str(texto)[:10].split("-"))
    return f"{d} de {MESES[m - 1]} de {a}"


def _corrida(tarea: str):
    md = V.metadata(tarea, "ebm")
    return D.fila(D.tabla("entrega3/corridas.csv"), clave=f"{tarea}__ebm__{md['balanceo']}__{md['optimizador']}")


def _hp(tarea: str) -> dict:
    return {k.replace("modelo__", ""): v for k, v in V.metadata(tarea, "ebm")["hiperparametros"].items()}


def _decimales(v) -> int:
    texto = repr(float(v))
    return 0 if texto.endswith(".0") else len(texto.split(".")[1])


def _hiperparametros(tarea: str) -> str:
    h = _hp(tarea)
    md = V.metadata(tarea, "ebm")
    return (f"η = {F.num(h['learning_rate'], _decimales(h['learning_rate']))}, L = {F.entero(h['max_leaves'])} hojas, "
            f"m = {F.entero(h['min_samples_leaf'])} por hoja, B = {F.entero(h['outer_bags'])} bolsas, K = {F.entero(h['interactions'])} "
            f"pares; balanceo: {D.NOMBRE_BALANCEO[md['balanceo']]}; búsqueda: {D.NOMBRE_OPTIMIZADOR[md['optimizador']]}")


def _cientifico(x, d=1) -> str:
    """Una cantidad muy pequeña en notación científica: 6,1·10⁻¹⁶."""
    if not x:
        return "0"
    e = math.floor(math.log10(abs(x)))
    return f"{F.num(x / 10 ** e, d)}·10{str(e).translate(SUPERINDICE)}"


def _particion(meta: dict) -> str:
    """La partición en palabras, con sus cifras de metadatos.json."""
    f = meta["fraccion_prueba"]
    estratos = "sin estratificar" if "sin estratificar" in meta["particion"] else "estratificada"
    return f"aleatoria por índices, {F.num(100 * (1 - f))}/{F.num(100 * f)}, con semilla {meta['semilla']}, {estratos}"


def _megas(b) -> str:
    return f"{F.num(b / 1e6, 2)} MB"


# --- La ficha del modelo ------------------------------------------------------------------------------------------------------
def contenido_ficha() -> list[dict]:
    meta = M.meta()
    cff = cita_cff()
    clf, reg = D.prueba("clasificacion").set_index("modelo"), D.prueba("regresion").set_index("modelo")
    e, x = clf.loc["ebm"], clf.loc[RIVAL]
    er, xr = reg.loc["ebm"], reg.loc[RIVAL]
    etl = D.json("eda/etl.json")
    tipos = etl["tipos"]
    version = V.metadata("clasificacion", "ebm")["versiones"]["interpret"]
    estimadores = " y ".join(V.metadata(t, "ebm")["estimador"].split(".")[-1] for t in TAREAS)
    capacidad = F.pct(M.CAPACIDAD_INFORME, 0)
    op = meta["punto_operacion"]
    cal = V.calibracion()["modelos"]["ebm"]
    conf = V.conformal()["modelos"]["ebm"][NIVEL_CONFORME]
    conj = cal["conformal"][NIVEL_CONFORME]
    imp = M.importancia("clasificacion")
    edad = M.variables().loc["age"]
    seg = D.errores_segmentos()
    coh = seg[seg["segmento"] == "cohorte de edad"].set_index("nivel")
    mayor, menor = coh["prevalencia"].idxmax(), coh["prevalencia"].idxmin()
    techo = D.techo()
    e2 = D.fila(D.tabla("resultados/entrega2.csv"), tarea="clasificacion", modelo="ebm")
    cubre = (e2["metrica"] - UMBRAL_MEDIO) / (techo["auc_mediana"] - UMBRAL_MEDIO)
    cal_csv = D.tabla("entrega3/complementos/calibracion.csv")
    sin_recal = cal_csv[(cal_csv["modelo"] == "ebm") & (cal_csv["version"] == "sin recalibrar")].iloc[0]
    pruebas = D.pruebas("clasificacion")
    v = D.veredictos("clasificacion")
    empata = F.enumerar(D.en_frase(m) for m in v.index if v[m] == "empata")
    supera = F.enumerar(D.en_frase(m) for m in v.index if v[m] == "supera")
    rope = D.fila(pruebas, modelo=RIVAL)["p_rope"]
    idx = indice()
    artefactos = [a for a in idx["artefactos"] if a["modelo"] == "ebm"]
    entrena = {t: _fecha(_corrida(t)["registrado_en"]) for t in TAREAS}
    ida = {t: V.metadata(t, "ebm")["ida_y_vuelta"] for t in TAREAS}
    verif = meta["verificacion"]
    num_cat = len(tipos["categoricas"])

    detalles = [
        ("Nombre", "EBM de fuga de clientes: clasificación del riesgo alto y regresión del puntaje de propensión"),
        ("Tipo", f"Explainable Boosting Machine (modelo aditivo generalizado aprendido por boosting), {estimadores} de interpret-core "
                 f"{version}"),
        ("Versión", f"{cff['version']} (artefactos de {_fecha(idx['generado'])})"),
        ("Fecha de entrenamiento", f"clasificación, {entrena['clasificacion']}; regresión, {entrena['regresion']}"),
        ("Autores", F.enumerar(cff["autores"])),
        ("Licencia", f"{cff['licencia']} para el código y los modelos; los datos son CC BY 4.0 y los derivados conservan su atribución"),
        ("Repositorio", cff["repositorio"]),
        ("Cita", f"{F.enumerar(cff['autores'])} ({cff['fecha'][:4]}). {cff['titulo']}. {cff['repositorio']}"),
    ]
    especificacion = [
        ("Entradas", f"{F.entero(len(etl['predictores']))} variables ({F.entero(len(tipos['numericas']))} numéricas, "
                     f"{F.entero(len(tipos['booleanas']))} booleanas y {F.entero(num_cat)} categóricas), que el preprocesamiento "
                     f"convierte en {F.entero(etl['n_columnas_modelo'])} columnas"),
        ("Salidas", "clasificación: puntaje de riesgo alto (sin recalibrar, para ordenar) y probabilidad recalibrada (isotónica); "
                    "regresión: el puntaje churn_probability previsto, con su intervalo conforme"),
        ("Hiperparámetros de clasificación", _hiperparametros("clasificacion")),
        ("Hiperparámetros de regresión", _hiperparametros("regresion")),
        ("Semilla", F.entero(meta["semilla"])),
    ]
    filas_art = []
    for a in artefactos:
        filas_art.append([f"artifacts/{a['ruta']}", _megas(a["bytes"]), a["sha256"][:HUELLA],
                          f"la tubería de scikit-learn con la EBM de {D.NOMBRE_TAREA[a['tarea']].lower()} (preprocesamiento y modelo)"])
    for t in TAREAS:
        o = _origen(f"modelos/ebm_{t}.json")
        filas_art.append([f"datos/{o['archivo']}", _megas(o["bytes"]), o["sha256"][:HUELLA],
                          f"las tablas del modelo de {D.NOMBRE_TAREA[t].lower()} que evalúa el motor en numpy de la app"])
    for t in TAREAS:
        o = V.metadata(t, "ebm")["origen"]
        filas_art.append([f"Entregable 3: {o['archivo']}", RAYA_TXT, o["sha256_origen"][:HUELLA], "el modelo final guardado por el proyecto"])
    max_motor = max(verif[f"ebm_{t}"]["max_dif_motor_frente_al_modelo_guardado"] for t in TAREAS)

    metricas = [
        ["AUC de prueba [IC 95 %]", f"{F.num(e['auc_prueba'], 4)} {F.intervalo(e['auc_bca_inf'], e['auc_bca_sup'])}",
         f"{F.num(x['auc_prueba'], 4)} {F.intervalo(x['auc_bca_inf'], x['auc_bca_sup'])}", "qué tan bien ordena a los clientes por riesgo"],
        ["AUC externa (media ± desviación de 5 pliegues)", f"{F.num(e['auc_ext_media'], 4)} ± {F.num(e['auc_ext_desv'], 4)}",
         f"{F.num(x['auc_ext_media'], 4)} ± {F.num(x['auc_ext_desv'], 4)}", "la validación anidada, antes de tocar la prueba"],
        ["AUC-PR", F.num(e["auc_pr"], 3), F.num(x["auc_pr"], 3), "orden de los positivos, sensible a la prevalencia"],
        [f"Captura con el {capacidad} de capacidad", F.pct(e["recall_op"]), F.pct(x["recall_op"]),
         "parte de los clientes de riesgo alto que alcanza la campaña"],
        [f"Precisión con el {capacidad} de capacidad", F.pct(e["precision_op"]), F.pct(x["precision_op"]),
         "parte de los contactados que son de riesgo alto"],
        [f"MCC con el {capacidad} de capacidad", F.num(e["mcc_op"], 3), F.num(x["mcc_op"], 3), "correlación entre lo contactado y lo real"],
        ["Error de calibración (ECE): sin recalibrar → isotónica", f"{F.num(cal['ece_prueba_sin_recalibrar'], 3)} → "
                                                                    f"{F.num(cal['ece_prueba_recalibrada'], 3)}",
         F.num(x["ece_isotonica"], 3) + " (isotónica)", "distancia entre la probabilidad dicha y la observada"],
        ["RMSE de prueba [IC 95 %]", f"{F.num(er['rmse_prueba'], 5)} {F.intervalo(er['rmse_bca_inf'], er['rmse_bca_sup'], 5)}",
         f"{F.num(xr['rmse_prueba'], 5)} {F.intervalo(xr['rmse_bca_inf'], xr['rmse_bca_sup'], 5)}", "error del puntaje previsto"],
        ["MAE y R² de prueba", f"{F.num(er['mae_prueba'], 4)} y {F.num(er['r2_prueba'], 3)}", f"{F.num(xr['mae_prueba'], 4)} y "
                                                                                                f"{F.num(xr['r2_prueba'], 3)}",
         "error absoluto medio y varianza explicada"],
        [f"Cobertura del intervalo conforme al {F.pct(float(NIVEL_CONFORME), 0)}", F.pct(conf["cobertura_prueba"]),
         F.pct(V.conformal()["modelos"][RIVAL][NIVEL_CONFORME]["cobertura_prueba"]), "parte de los puntajes reales dentro del intervalo"],
    ]
    segmentos = []
    for _, f in seg.iterrows():
        rotulo = {"cohorte de edad": "Edad", "productos activos": "Productos activos"}.get(f["segmento"], f["segmento"])
        segmentos.append([f"{rotulo}: {f['nivel']}", F.entero(f["n"]), F.pct(f["prevalencia"]),
                          F.num(f["auc_ebm"], 3) if f["auc_ebm"] == f["auc_ebm"] else RAYA_TXT,
                          F.num(f["auc_xgboost"], 3) if f["auc_xgboost"] == f["auc_xgboost"] else RAYA_TXT,
                          F.pct(f["fn_ebm"], 0) if f["prevalencia"] > 0 else RAYA_TXT,
                          F.pct(f["fn_xgboost"], 0) if f["prevalencia"] > 0 else RAYA_TXT, F.num(f["rmse_ebm"], 4),
                          F.con_signo(f["sesgo_ebm"], 4)])

    return [
        {"titulo": "Detalles del modelo", "columnas": "1-1",
         "subtitulo": "Qué es, quién lo hizo, con qué versión y con qué licencia; los artefactos y su huella.",
         "tarjetas": [
             {"titulo": "El modelo", "bloques": [("pares", detalles)],
              "fuentes": ["CITATION.cff", "artifacts/indice.json", D.fuente("entrega3/corridas.csv")]
                         + [f"artifacts/{t}/ebm/metadata.json" for t in TAREAS]},
             {"titulo": "Entradas, salidas e hiperparámetros", "bloques": [("pares", especificacion)],
              "fuentes": [f"artifacts/{t}/ebm/metadata.json" for t in TAREAS] + [D.fuente("metadatos.json"), D.fuente("eda/etl.json")]},
             {"titulo": "Artefactos y su huella", "ancho": True, "bloques": [
                 ("tabla", ["Archivo", "Tamaño", "SHA-256 (inicio)", "Qué es"], filas_art, [False, True, False, False], (3,)),
                 ("nota", f"La ida y vuelta de los artefactos dio predicciones idénticas a las guardadas en "
                          f"{F.entero(ida['clasificacion']['filas'])} clientes de prueba, y el motor en numpy difiere del modelo guardado "
                          f"a lo sumo {_cientifico(max_motor)} en {F.entero(verif['ebm_clasificacion']['filas_comparadas'])} "
                          "clientes. La huella completa está en artifacts/indice.json y datos/origen.json.")],
              "fuentes": ["artifacts/indice.json", D.fuente("origen.json"), D.fuente("metadatos.json")]},
         ]},
        {"titulo": "Para qué sirve y para qué no", "columnas": "3",
         "subtitulo": "El uso previsto, quién debería usarlo y los usos que no se recomiendan.",
         "tarjetas": [
             {"titulo": "Uso previsto", "bloques": [("lista", [
                 f"Ordenar a los clientes de COFINFAD por su riesgo para priorizar una campaña de retención con capacidad limitada: "
                 f"contactando al {capacidad} de mayor puntaje ({F.entero(op['contactados'])} de {F.entero(meta['n_prueba'])}) alcanza "
                 f"al {F.pct(e['recall_op'])} de los clientes de riesgo alto, con una precisión del {F.pct(e['precision_op'])}.",
                 "Explicar qué variables mueven el puntaje y cómo: cada predicción es la suma exacta de lo que aporta cada variable.",
                 "Docencia y comparación de métodos: un modelo legible frente a los ensambles de árboles, con el mismo protocolo.",
             ])], "fuentes": [D.fuente("resultados/clasificacion_prueba.csv"), D.fuente("metadatos.json")]},
             {"titulo": "Usuarios previstos", "bloques": [("lista", [
                 "El profesor y los estudiantes del curso de Machine Learning.",
                 "Analistas que quieren estudiar un modelo aditivo legible y compararlo con modelos de caja negra.",
                 "Equipos de retención que quieren ver el método, no aplicarlo sin antes validarlo con sus propios datos.",
             ])], "fuentes": [INFORME]},
             {"titulo": "Usos no recomendados", "bloques": [("lista", [
                 "Decidir sobre clientes reales: los datos son sintéticos y el resultado describe las reglas con que se generaron.",
                 "Decisiones de crédito, precio, cobranza o exclusión de un servicio.",
                 f"Leer el puntaje sin recalibrar como una probabilidad de fuga: con los pesos de clase su media es "
                 f"{F.num(sin_recal['prob_media'], 3)} y la prevalencia, {F.num(sin_recal['prevalencia'], 3)}.",
                 "Leer las curvas como efectos causales («subir la satisfacción reduce la fuga»): describen lo que el modelo asigna.",
                 f"Puntuar valores fuera del rango de entrenamiento: la edad, por ejemplo, solo toma {F.entero(edad['n_distintos'])} valores "
                 f"({str(edad['valores']).replace('|', ', ')}).",
                 "Tratar «riesgo alto» como fuga observada, o usar productos activos como entrada: es la variable con que se generó el "
                 "puntaje y el modelo recuperaría la regla en vez de predecir.",
                 "Llevarlo a otra entidad, otro país u otro periodo sin reentrenarlo y validarlo.",
             ])], "fuentes": [D.fuente("entrega3/complementos/calibracion.csv"), D.fuente("variables.csv"), INFORME]},
         ]},
        {"titulo": "Factores, métricas y datos", "columnas": "1-1",
         "subtitulo": f"Con qué se evaluó, en qué datos y en qué punto de operación. XGBoost va al lado porque la EBM empata con él.",
         "tarjetas": [
             {"titulo": "Factores relevantes", "bloques": [("lista", [
                 f"Cohorte de edad: el conjunto tiene {F.entero(len(coh))} cohortes y el riesgo alto va del {F.pct(coh.loc[menor, 'prevalencia'])} "
                 f"({menor} años) al {F.pct(coh.loc[mayor, 'prevalencia'])} ({mayor}). La edad reúne el {F.pct(imp['age'])} de la "
                 "importancia del modelo.",
                 "Productos activos: la variable excluida con la que se generó el puntaje; los errores grandes de la regresión siguen su "
                 "escalera.",
                 f"Variables personales que entran al modelo: género ({F.pct(imp['gender'])} de la importancia), ubicación "
                 f"({F.pct(imp['location'])}) e ingreso ({F.pct(imp['income_bracket'])}).",
                 "En la evaluación se separó la prueba por cohorte de edad y por productos activos (tabla de abajo).",
             ])], "fuentes": [D.fuente("resultados/errores_segmentos.csv"), D.fuente("modelos/ebm_clasificacion.json")]},
             {"titulo": "Datos de entrenamiento y de evaluación", "bloques": [("pares", [
                 ("Conjunto", "COFINFAD (Mendeley Data, CC BY 4.0), sintético: ver DATA_STATEMENT.md"),
                 ("Partición", f"{_particion(meta)}, antes de calcular cualquier estadístico del objetivo"),
                 ("Entrenamiento", f"{F.entero(meta['n_entrenamiento'])} clientes; riesgo alto: {F.pct(meta['prevalencia_entrenamiento'])}"),
                 ("Evaluación", f"{F.entero(meta['n_prueba'])} clientes de prueba, tocados una sola vez; riesgo alto: "
                                f"{F.pct(meta['prevalencia_prueba'])} ({F.entero(meta['positivos_prueba'])} clientes)"),
                 ("Etiqueta", f"riesgo alto = churn_probability por encima de su percentil {F.entero(meta['etiqueta']['q'] * 100)} en el "
                              f"entrenamiento ({F.num(meta['etiqueta']['umbral'], 4)})"),
                 ("Selección", f"validación cruzada anidada de {F.entero(_corrida('clasificacion')['k_ext'])} × "
                               f"{F.entero(_corrida('clasificacion')['k_int'])} pliegues, con "
                               f"{F.entero(_corrida('clasificacion')['presupuesto'])} configuraciones por búsqueda"),
                 ("Punto de operación", f"contactar al {capacidad} de mayor puntaje: {F.entero(op['contactados'])} clientes; el último "
                                        f"contactado tiene un puntaje de {F.num(op['puntaje_minimo_contactado_ebm'], 4)}"),
             ])], "fuentes": [D.fuente("metadatos.json"), D.fuente("entrega3/corridas.csv")]},
             {"titulo": "Métricas en la prueba", "ancho": True, "bloques": [
                 ("tabla", ["Métrica", "EBM", D.NOMBRE_MODELO[RIVAL], "Qué mide"], metricas, [False, True, True, False], (3,)),
                 ("nota", f"Frente a {empata} ninguna prueba encuentra diferencia tras la corrección de Holm (probabilidad de equivalencia "
                          f"práctica con {D.NOMBRE_MODELO[RIVAL]}: {F.prob(rope)}); supera con significación a {supera}. En el informe, "
                          f"la conclusión es que la EBM alcanza a los modelos de árboles, no que los supere.")],
              "fuentes": [D.fuente("resultados/clasificacion_prueba.csv"), D.fuente("resultados/regresion_prueba.csv"),
                          D.fuente("calibracion_modelos.json"), D.fuente("conformal.json"), D.fuente("resultados/pruebas_clasificacion.csv")]},
         ]},
        {"titulo": "Análisis por segmento", "columnas": "1-1",
         "subtitulo": f"La prueba dividida por cohorte de edad y por productos activos, con el {capacidad} de capacidad.",
         "tarjetas": [
             {"titulo": "Dónde acierta y dónde falla", "ancho": True, "bloques": [
                 ("tabla", ["Segmento", "Clientes", "Riesgo alto", "AUC de la EBM", f"AUC de {D.NOMBRE_MODELO[RIVAL]}",
                            "Riesgo alto que la EBM deja fuera", f"… y {D.NOMBRE_MODELO[RIVAL]}", "RMSE de la EBM", "Sesgo de la EBM"],
                  segmentos, [False] + [True] * 8, (4, 6)),
                 ("nota", f"La cohorte de {mayor} años es la más difícil (AUC {F.num(coh.loc[mayor, 'auc_ebm'], 3)}) y la de {menor}, la más "
                          f"fácil ({F.num(coh.loc[menor, 'auc_ebm'], 3)}). Con la misma captura global, la EBM concentra la campaña en la "
                          f"cohorte mayor: deja fuera al {F.pct(coh.loc[mayor, 'fn_ebm'], 0)} de sus clientes de riesgo alto y al "
                          f"{F.pct(coh.loc[menor, 'fn_ebm'], 0)} de los de {menor}; {D.NOMBRE_MODELO[RIVAL]} reparte más. Por productos "
                          "activos el sesgo de la regresión forma una escalera: es la parte del puntaje que depende de la variable excluida.")],
              "fuentes": [D.fuente("resultados/errores_segmentos.csv")]},
         ]},
        {"titulo": "Ética, limitaciones y recomendaciones", "columnas": "3",
         "subtitulo": "Lo que hay que saber antes de usarlo.",
         "tarjetas": [
             {"titulo": "Consideraciones éticas", "bloques": [("lista", [
                 "Los datos son públicos y sintéticos: no hay personas reales identificables.",
                 f"La edad, el género y la ubicación son entradas. La edad domina el modelo ({F.pct(imp['age'])} de la importancia): una "
                 "campaña real que lo siguiera trataría distinto a las personas según su edad, y eso habría que justificarlo y revisarlo.",
                 "El modelo es legible: cada decisión se descompone exactamente en lo que aporta cada variable, lo que facilita auditarlo.",
                 "No se mide fuga real: usarlo para negar o encarecer un servicio no tendría respaldo en estos datos.",
             ])], "fuentes": [D.fuente("modelos/ebm_clasificacion.json"), INFORME]},
             {"titulo": "Limitaciones y riesgos", "bloques": [("lista", [
                 "Un solo conjunto de datos, con rasgos sintéticos (tres cohortes de edad, una escalera de productos activos, "
                 "transacciones sin señal): las conclusiones describen esas reglas.",
                 f"El objetivo es el puntaje de otro modelo, no la fuga observada; sin productos activos hay un techo de información: con "
                 f"la etiqueta de la mediana, la EBM cubre el {F.pct(cubre, 0)} de lo que se puede alcanzar sin esa variable.",
                 f"Las probabilidades sin recalibrar están sesgadas (ECE {F.num(cal['ece_prueba_sin_recalibrar'], 3)}); hay que recalibrarlas.",
                 f"Los conjuntos conformes de clases cubren de más ({F.pct(conj['cobertura_prueba'])} al {F.pct(float(NIVEL_CONFORME), 0)}): "
                 f"el {F.pct(conj['fraccion_ambos'], 0)} de los clientes recibe las dos clases.",
                 "Con cinco pliegues las pruebas pareadas tienen poca potencia; los tiempos son de un solo equipo.",
             ])], "fuentes": [D.fuente("entrega3/comparacion_entrega2/techo.json"), D.fuente("resultados/entrega2.csv"),
                              D.fuente("calibracion_modelos.json"), INFORME]},
             {"titulo": "Recomendaciones", "bloques": [("lista", [
                 "Ordenar con el puntaje sin recalibrar e informar con la probabilidad recalibrada.",
                 f"Elegir la capacidad con el costo de cada contacto en mente: con el {F.pct(M.CAPACIDAD_INFORME, 0)} casi todos los modelos "
                 f"alcanzan a los mismos clientes, pero con el {F.pct(CAPACIDAD_MITAD, 0)} la EBM captura el {F.pct(e['recall_50'])} y la regresión logística el "
                 f"{F.pct(clf.loc['logistica', 'recall_50'])}.",
                 "Validar en el tiempo con fuga observada (entrenar con un periodo y probar con el siguiente) antes de cualquier uso real.",
                 "Revisar los resultados por cohorte de edad y reentrenar si cambia la distribución de los clientes.",
                 "Mantener una revisión humana de cada decisión que afecte a un cliente.",
             ])], "fuentes": [D.fuente("resultados/clasificacion_prueba.csv"), INFORME]},
         ]},
    ]


RAYA_TXT = F.RAYA


# --- La declaración de los datos ---------------------------------------------------------------------------------------------
def contenido_datos() -> list[dict]:
    meta = M.meta()
    cff = cita_cff()
    dat = cff["datos"]
    etl = D.json("eda/etl.json")
    tipos = etl["tipos"]
    aud = etl["auditoria_transacciones"]
    coh = D.tabla("eda/cohortes.csv")
    coh = coh[coh["conjunto"] == "entrenamiento"].set_index("cohorte")
    aso = D.tabla("eda/asociacion.csv").set_index("variable")
    falt = D.tabla("eda/faltantes.csv").set_index("columna")
    entr = V.entrenamiento()["churn_probability"]
    derivadas = etl["derivadas"]
    r_prod = aso.loc["active_products", "pearson"]
    max_trans = aso.loc[derivadas, "valor"].abs().max()
    cat = aso[aso["medida"] == "eta2"]
    edad = M.variables().loc["age"]
    total_tx = int(etl["n_transacciones_limpias"]) + int(aud["duplicadas"])
    sin_tarjeta = falt.loc["credit_utilization_ratio", "fraccion"]
    texto_falt = falt[(falt["fraccion"] > 0) & ~falt["en_modelo"].astype(bool)]
    return [
        {"titulo": "Resumen", "columnas": "1-1", "subtitulo": "Qué conjunto es, de dónde viene y con qué licencia.",
         "tarjetas": [
             {"titulo": "El conjunto", "bloques": [("pares", [
                 ("Nombre", dat["titulo"]),
                 ("Autores", "; ".join(dat["autores"])),
                 ("Publicación", f"{dat['editorial']}, versión {dat['version']}, {dat['anio']}"),
                 ("DOI", f"{dat['doi']} (https://doi.org/{dat['doi']})"),
                 ("Licencia", f"{dat['licencia'].replace('-', ' ')}: se puede usar y redistribuir citando a los autores"),
                 ("Periodo que describe", f"{PERIODO_DATOS}, según la descripción del conjunto que recoge el informe final"),
                 ("Tamaño", f"{F.entero(etl['n_clientes'])} clientes con {F.entero(etl['n_columnas'])} columnas y "
                            f"{F.entero(total_tx)} transacciones en una tabla aparte"),
             ])], "fuentes": ["CITATION.cff", D.fuente("eda/etl.json"), INFORME]},
         ]},
        {"titulo": "Cómo se generaron los datos", "columnas": "1-1",
         "subtitulo": "Lo que se sabe de su origen sale del análisis exploratorio del entrenamiento.",
         "tarjetas": [
             {"titulo": "Datos sintéticos, generados con reglas", "bloques": [
                 ("parrafo", "Los datos son sintéticos: describen a los clientes de una fintech colombiana, pero llevan las huellas de una "
                             "generación por reglas, no de una muestra de clientes reales. Las conclusiones que se saquen de ellos describen "
                             "esas reglas."),
                 ("lista", [
                     f"La edad solo toma {F.entero(edad['n_distintos'])} valores ({str(edad['valores']).replace('|', ', ')}) en "
                     f"{F.entero(len(coh))} cohortes de tamaño casi igual, y el archivo está ordenado por bloques que las siguen.",
                     f"El puntaje churn_probability baja en una escalera de pasos casi idénticos con el número de productos activos "
                     f"(r de Pearson {F.num(r_prod, 3)}: explica el {F.pct(r_prod ** 2)} de su varianza): la firma de una fórmula.",
                     f"Las {F.entero(len(derivadas))} variables transaccionales derivadas no tienen señal (|ρ| ≤ {F.num(max_trans, 3)}).",
                     f"El uso del cupo falta exactamente cuando el cliente no tiene tarjeta de crédito ({F.pct(sin_tarjeta)} de los clientes de "
                     "entrenamiento): "
                     "es un «no aplica», no un dato perdido.",
                 ]),
             ], "fuentes": [D.fuente("eda/asociacion.csv"), D.fuente("eda/cohortes.csv"), D.fuente("eda/faltantes.csv"), D.fuente("variables.csv")]},
             {"titulo": "Estructura", "bloques": [("pares", [
                 ("Unidad", "un cliente por fila; las transacciones, en una tabla aparte que se resume por cliente"),
                 ("Objetivo", f"churn_probability, un puntaje de propensión entre {F.num(entr.min(), 3)} y {F.num(entr.max(), 3)} en el "
                              "entrenamiento: nunca vale 0 ni 1, no es un evento observado"),
                 ("Etiqueta del proyecto", f"riesgo alto = puntaje por encima de su percentil {F.entero(meta['etiqueta']['q'] * 100)} del "
                                           f"entrenamiento ({F.num(meta['etiqueta']['umbral'], 4)})"),
                 ("Predictores", f"{F.entero(len(etl['predictores']))}: {F.entero(len(tipos['numericas']))} numéricos, "
                                 f"{F.entero(len(tipos['booleanas']))} booleanos y {F.entero(len(tipos['categoricas']))} categóricos "
                                 f"({F.entero(etl['n_columnas_modelo'])} columnas tras el preprocesamiento)"),
                 ("Excluidas", f"{F.entero(len(etl['eliminadas']))} columnas: {', '.join(etl['eliminadas'])}"),
             ])], "fuentes": [D.fuente("eda/etl.json"), D.fuente("eda/entrenamiento.parquet"), D.fuente("metadatos.json")]},
         ]},
        {"titulo": "Sesgos conocidos", "columnas": "1-1", "subtitulo": "Lo que hay que tener en cuenta al usarlos.",
         "tarjetas": [
             {"titulo": "Sesgos y particularidades", "bloques": [("lista", [
                 "Cohortes de edad con tasas de riesgo alto muy distintas en el entrenamiento: " + "; ".join(
                     f"{c} años, {F.pct(coh.loc[c, 'tasa_riesgo_alto'])}" for c in coh.index) + ".",
                 "El archivo está ordenado por cohortes: una partición por posición daría conjuntos distintos; por eso se parte al azar.",
                 "El puntaje se construyó a partir de los productos activos: usarlos como entrada es una fuga de información.",
                 f"Las categóricas casi no se asocian con el puntaje (η² ≤ {F.num(cat['valor'].max(), 3)}), tampoco el género ni la ubicación.",
                 "Columnas de texto con " + " y ".join(f"{F.pct(texto_falt.loc[c, 'fraccion'], 0)} de faltantes ({c})" for c in texto_falt.index)
                 + ", sin mecanismo ni señal: se excluyen.",
             ])], "fuentes": [D.fuente("eda/cohortes.csv"), D.fuente("eda/asociacion.csv"), D.fuente("eda/faltantes.csv")]},
             {"titulo": "Preprocesamiento y partición", "bloques": [("lista", [
                 f"Auditoría de las transacciones: {F.entero(aud['duplicadas'])} duplicadas eliminadas; montos no positivos, fechas nulas y "
                 f"transacciones huérfanas: {F.entero(aud['montos_no_positivos'] + aud['fechas_nulas'] + aud['huerfanas'])}.",
                 f"{F.entero(len(derivadas))} variables de comportamiento derivadas de las transacciones ({', '.join(derivadas)}), sin usar el "
                 "objetivo.",
                 f"{F.entero(len(etl['eliminadas']))} columnas eliminadas (identificador, la variable que generó el puntaje, el segundo "
                 "objetivo, duplicados, fechas sin señal, una colineal y textos con faltantes).",
                 "El uso del cupo se imputa con 0 cuando no hay tarjeta; las categóricas se codifican con indicadoras y las numéricas se "
                 "estandarizan dentro de cada pliegue de cada validación cruzada.",
                 f"Partición {_particion(meta)}: {F.entero(meta['n_entrenamiento'])} clientes de entrenamiento y "
                 f"{F.entero(meta['n_prueba'])} de prueba. El umbral de la etiqueta y todo el análisis exploratorio usan solo el "
                 "entrenamiento.",
             ])], "fuentes": [D.fuente("eda/etl.json"), D.fuente("metadatos.json")]},
         ]},
        {"titulo": "Usos adecuados e inadecuados", "columnas": "1-1", "subtitulo": "Para qué sirven estos datos y para qué no.",
         "tarjetas": [
             {"titulo": "Adecuados", "bloques": [("lista", [
                 "Enseñar y comparar métodos de aprendizaje supervisado con un protocolo reproducible.",
                 "Estudiar fugas de información, desbalance, calibración y modelos legibles.",
                 "Reproducir los resultados de este proyecto.",
             ])], "fuentes": [INFORME]},
             {"titulo": "Inadecuados", "bloques": [("lista", [
                 "Sacar conclusiones sobre clientes reales de una fintech colombiana.",
                 "Estimar tasas reales de abandono: el objetivo es un puntaje, no la fuga observada.",
                 "Tomar decisiones sobre personas o leer las asociaciones como causas.",
                 "Predecir el puntaje con los productos activos: el modelo recuperaría la regla con que se generó.",
             ])], "fuentes": [INFORME]},
         ]},
    ]


# --- Markdown ---------------------------------------------------------------------------------------------------------------
def _celda(texto) -> str:
    return str(texto).replace("|", "\\|").replace("\n", " ")


def _bloque_md(b) -> list[str]:
    tipo = b[0]
    if tipo == "parrafo":
        return [b[1], ""]
    if tipo == "nota":
        return [f"*{b[1]}*", ""]
    if tipo == "lista":
        return [f"- {t}" for t in b[1]] + [""]
    if tipo == "pares":
        return [f"- **{k}:** {v}" for k, v in b[1]] + [""]
    if tipo == "tabla":
        enc, filas = b[1], b[2]
        num = b[3] if len(b) > 3 else [False] * len(enc)
        alinea = ["---:" if n else "---" for n in num]
        lineas = ["| " + " | ".join(_celda(h) for h in enc) + " |", "| " + " | ".join(alinea) + " |"]
        lineas += ["| " + " | ".join(_celda(c) for c in f) + " |" for f in filas]
        return lineas + [""]
    raise ValueError(f"bloque desconocido: {tipo!r}")


def a_markdown(titulo: str, entrada: list[str], grupos: list[dict]) -> str:
    lineas = [f"# {titulo}", ""] + entrada + [""]
    for g in grupos:
        lineas += [f"## {g['titulo']}", ""]
        if g.get("subtitulo"):
            lineas += [g["subtitulo"], ""]
        for t in g["tarjetas"]:
            lineas += [f"### {t['titulo']}", ""]
            for b in t["bloques"]:
                lineas += _bloque_md(b)
            if t.get("fuentes"):
                lineas += [f"Fuentes: {' · '.join(t['fuentes'])}", ""]
    return "\n".join(lineas).rstrip() + "\n"


def entrada_ficha() -> list[str]:
    cff = cita_cff()
    return [f"Ficha del modelo (*model card*, Mitchell et al., 2019) de la Explainable Boosting Machine del proyecto «{cff['titulo']}». "
            f"Versión {cff['version']}, {_fecha(cff['fecha'])}. La genera `generar_fichas.py` desde `datos/` y `artifacts/`: ninguna cifra "
            "se escribe a mano. La misma ficha está en la vista «Ficha del modelo» del dashboard (/modelos/ficha)."]


def entrada_datos() -> list[str]:
    cff = cita_cff()
    return [f"Declaración de los datos (en la línea de Bender y Friedman, 2018, y Gebru et al., 2021) del conjunto que usa el proyecto "
            f"«{cff['titulo']}». La genera `generar_fichas.py` desde `datos/`: ninguna cifra se escribe a mano."]


def main() -> int:
    FICHA.write_text(a_markdown("Ficha del modelo: EBM de fuga de clientes", entrada_ficha(), contenido_ficha()), encoding="utf-8",
                     newline="\n")
    DECLARACION.write_text(a_markdown("Declaración de los datos: COFINFAD", entrada_datos(), contenido_datos()), encoding="utf-8",
                           newline="\n")
    for ruta in (FICHA, DECLARACION):
        print(f"{ruta.name}: {len(ruta.read_text(encoding='utf-8').splitlines())} líneas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
