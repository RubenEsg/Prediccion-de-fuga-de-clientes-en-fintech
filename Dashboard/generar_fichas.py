"""Ficha del modelo (MODEL_CARD.md) y declaración de los datos (DATA_STATEMENT.md), generadas desde datos/ y artifacts/.

    .venv/Scripts/python.exe -B generar_fichas.py        escribe los dos archivos en esta carpeta

La ficha es la del modelo original del proyecto, BDL (Boosting con Desplazamiento Latente): contenido_ficha_original().
Mientras la corrección no lo haya dejado en la app (artefactos y tabla de líderes), se escribe la de la EBM del Entregable 3
(contenido_ficha_ebm()), que hoy es un benchmark. Ninguna cifra se escribe a mano: todas salen de datos/ (por datos_app,
motor y modelos_vivos), de artifacts/indice.json y de CITATION.cff; el veredicto frente a los benchmarks, de
D.frase_veredicto (veredicto.json y condicion4.json). La vista «Ficha del modelo» (secciones/ficha.py) arma la misma ficha
con contenido_ficha(), así que la página y el Markdown que se descarga en /ficha.md dicen lo mismo; si cambian los datos
(por ejemplo, tras volver a correr preparar_original.py), se vuelve a correr este guion.

El formato sigue a Mitchell et al. (2019), «Model Cards for Model Reporting» (FAT* '19), para la ficha, y la idea de una
declaración de los datos de Bender y Friedman (2018) y Gebru et al. (2021, «Datasheets for Datasets»). Las afirmaciones
sobre el modelo original son las del libro del proyecto final (ProyectoFinal, capítulos 5 a 9); las de la EBM y las de los
datos, las del informe del Entregable 3 (Informe Final/metodologia.tex, resultados.tex y conclusiones.tex).

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
O = D.ORIGINAL
LIBRO = "Libro del proyecto final (ProyectoFinal): capítulos 5, 6, 7 y 9"
PROTOCOLO = "ModeloOriginal/experiments/protocolo.md (protocolo pre-registrado)"
IMPORTANCIA_MINIMA = 0.001         # por debajo de esta parte de la importancia, una variable «casi no se usa»
NOMBRE_HP = {"tasa": "tasa de aprendizaje", "profundidad": "profundidad de los árboles", "min_hoja": "mínimo de filas por hoja",
             "lambda_l2": "regularización L2", "max_bins": "tramos por variable", "n_componentes": "clases latentes",
             "usar_puntaje": "usa el puntaje continuo"}


def hay_original() -> bool:
    """Si la corrección ya dejó el modelo original en las dos tareas (artefacto y fila en la tabla de líderes)."""
    return all(V.hay_original(t) and D.hay_modelo_original(t) for t in TAREAS)


def contenido_ficha() -> list[dict]:
    """La ficha del modelo original, BDL; mientras la corrección no lo haya dejado en la app, la de la EBM del Entregable 3."""
    return contenido_ficha_original() if hay_original() else contenido_ficha_ebm()


def titulo_ficha() -> str:
    return ("Ficha del modelo: BDL, el modelo original de fuga de clientes" if hay_original()
            else "Ficha del modelo: EBM de fuga de clientes")


def _valor_hp(v) -> str:
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, str):
        return "automático (BIC)" if v == "auto" else v
    if isinstance(v, int) or float(v) == int(v):
        return F.entero(v)
    d = max(0, 3 - math.floor(math.log10(abs(v))))
    return F.num(v, d).rstrip("0").rstrip(",")


def _hiperparametros_original(tarea: str) -> str:
    md = V.metadata(tarea, O)
    pares = [f"{NOMBRE_HP.get(k.replace('modelo__', ''), k.replace('modelo__', ''))}: {_valor_hp(v)}" for k, v in md["hiperparametros"].items()]
    return "; ".join(pares) + f"; búsqueda: {D.NOMBRE_OPTIMIZADOR[md['optimizador']]}, {D.NOMBRE_BALANCEO[md['balanceo']]}"


def _aprendido(tarea: str) -> str:
    """Lo que quedó en el modelo final de una tarea (datos/original/modelo_<tarea>.json), en una línea."""
    m = D.modelo_original(tarea)
    if m is None or not m.get("K"):
        return "todavía no disponible"
    return (f"{F.entero(m['K'])} clases latentes (elegidas por BIC), {F.entero(m['n_arboles'])} árboles ({F.entero(m['n_calentamiento'])} de "
            f"calentamiento) y {F.entero(m['vueltas_usadas'])} vueltas del algoritmo EM")


def _lectura_subgrupos(sub) -> str:
    """Lo que dice la tabla de errores por subgrupo sobre el AUC: cuánto se aparta del total dentro de una sola edad y en qué
    subgrupo difieren más los dos modelos. Todo sale de la tabla; '' si no trae el AUC."""
    con_auc = sub[sub["auc_original"].notna() & sub["auc_benchmark"].notna()]
    if con_auc.empty:
        return ""
    partes = []
    edades, todos = con_auc[con_auc["subgrupo"].str.startswith("Edad ")], con_auc[con_auc["subgrupo"] == "Todos"]
    if len(edades) and len(todos):
        baja, alta = edades.loc[edades["auc_original"].idxmin()], edades.loc[edades["auc_original"].idxmax()]
        partes.append(f"Dentro de una sola edad, donde esa variable ya no ordena a los clientes, el AUC de BDL va de "
                      f"{F.num(baja['auc_original'], 4)} ({baja['subgrupo'].lower()}) a {F.num(alta['auc_original'], 4)} "
                      f"({alta['subgrupo'].lower()}), frente a {F.num(todos['auc_original'].iloc[0], 4)} en el total.")
    diferencia = con_auc["auc_original"] - con_auc["auc_benchmark"]
    mayor = con_auc.loc[diferencia.abs().idxmax()]
    signo = ("cambian de signo de un subgrupo a otro y " if (diferencia > 0).any() and (diferencia < 0).any() else "")
    partes.append(f"Las diferencias entre los dos modelos {signo}no llevan prueba de hipótesis; la mayor en AUC está en «{mayor['subgrupo']}» "
                  f"({F.entero(mayor['clientes'])} clientes), donde BDL obtiene {F.num(mayor['auc_original'], 4)} y "
                  f"{D.NOMBRE_MODELO[mayor['benchmark_clasificacion']]}, {F.num(mayor['auc_benchmark'], 4)}.")
    return " " + " ".join(partes)


def contenido_ficha_original() -> list[dict]:
    meta = M.meta()
    cff = cita_cff()
    clf, reg = D.lideres("clasificacion").set_index("modelo"), D.lideres("regresion").set_index("modelo")
    vc, vr = D.veredicto_original("clasificacion"), D.veredicto_original("regresion")
    bc = vc["mejor_benchmark"] if vc is not None and vc["mejor_benchmark"] in clf.index else RIVAL
    br = vr["mejor_benchmark"] if vr is not None and vr["mejor_benchmark"] in reg.index else RIVAL
    e, x, er, xr = clf.loc[O], clf.loc[bc], reg.loc[O], reg.loc[br]
    etl = D.json("eda/etl.json")
    tipos = etl["tipos"]
    md = {t: V.metadata(t, O) for t in TAREAS}
    versiones = md["regresion"]["versiones"]
    estimadores = " y ".join(md[t]["estimador"].split(".")[-1] for t in TAREAS)
    capacidad = F.pct(M.CAPACIDAD_INFORME, 0)
    nivel = F.pct(float(NIVEL_CONFORME), 0)
    op = meta["punto_operacion"]
    cal, cal_b = V.calibracion()["modelos"][O], V.calibracion()["modelos"].get(bc)
    conf, conf_b = V.conformal()["modelos"][O][NIVEL_CONFORME], V.conformal()["modelos"][br][NIVEL_CONFORME]
    conj = cal["conformal"][NIVEL_CONFORME]
    shap = V.shap_indice()["modelos"][f"clasificacion__{O}"]["importancia_media_abs"]
    total = sum(shap.values())
    imp = {k: v / total for k, v in shap.items()}
    edad = M.variables().loc["age"]
    seg = D.errores_segmentos()
    coh = seg[seg["segmento"] == "cohorte de edad"].set_index("nivel")
    mayor, menor = coh["prevalencia"].idxmax(), coh["prevalencia"].idxmin()
    idx = indice()
    artefactos = [a for a in idx["artefactos"] if a["modelo"] == O]
    corridas = {t: D.corrida_original(t, O) for t in TAREAS}
    entrena = {t: (_fecha(corridas[t]["registrado_en"]) if corridas[t] is not None else RAYA_TXT) for t in TAREAS}
    ida = {t: md[t]["ida_y_vuelta"] for t in TAREAS}
    costo = {t: D.costo_original(t, O) for t in TAREAS}
    semillas = D.original_tabla("semillas.csv")
    personales = ("gender", "location", "income_bracket")
    if max(imp.get(v, 0.0) for v in personales) < IMPORTANCIA_MINIMA:
        texto_personales = ("El género, la ubicación y el ingreso también son entradas, pero el modelo casi no los usa: ninguno llega "
                            f"al {F.pct(IMPORTANCIA_MINIMA, 1)} de la importancia.")
    else:
        texto_personales = (f"Variables personales que entran al modelo: género ({F.pct(imp['gender'])} de la importancia), ubicación "
                            f"({F.pct(imp['location'])}) e ingreso ({F.pct(imp['income_bracket'])}).")
    aditivos = [bool((D.modelo_original(t) or {}).get("aditivo")) for t in TAREAS]
    te = D.original_json("techo.json")
    codigo = md["regresion"]["origen"]
    mismo = bc == br
    rotulo_b = D.NOMBRE_MODELO[bc] if mismo else "Mejor benchmark"

    def del_b(valor, modelo):
        """El valor del mejor benchmark de una tarea; si no es el mismo en las dos, con su nombre."""
        return valor if mismo else f"{valor} ({D.NOMBRE_MODELO[modelo]})"

    detalles = [
        ("Nombre", "BDL de fuga de clientes: regresión del puntaje de propensión y clasificación del riesgo alto"),
        ("Tipo", f"{D.NOMBRE_LARGO_ORIGINAL}, el modelo original del proyecto: un boosting de árboles propios con una mezcla latente de "
                 f"desplazamientos, ajustado con un algoritmo EM generalizado ({estimadores}, en {codigo['codigo']})"),
        ("Dependencias", f"numpy {versiones['numpy']}, scipy {versiones['scipy']} y scikit-learn {versiones['scikit-learn']} (la tubería y "
                         "el preprocesamiento); el modelo no importa ninguna librería de modelos"),
        ("Versión", f"{cff['version']} (artefactos del {_fecha(idx.get('original', idx)['generado'])})"),
        ("Fecha de entrenamiento", f"clasificación, {entrena['clasificacion']}; regresión, {entrena['regresion']}"),
        ("Autores", F.enumerar(cff["autores"])),
        ("Licencia", f"{cff['licencia']} para el código y los modelos; los datos son CC BY 4.0 y los derivados conservan su atribución"),
        ("Repositorio", cff["repositorio"]),
        ("Cita", f"{F.enumerar(cff['autores'])} ({cff['fecha'][:4]}). {cff['titulo']}. {cff['repositorio']}"),
    ]
    especificacion = [
        ("Entradas", f"{F.entero(len(etl['predictores']))} variables ({F.entero(len(tipos['numericas']))} numéricas, "
                     f"{F.entero(len(tipos['booleanas']))} booleanas y {F.entero(len(tipos['categoricas']))} categóricas), que el "
                     f"preprocesamiento convierte en {F.entero(etl['n_columnas_modelo'])} columnas. En el entrenamiento de la clasificación "
                     "recibe, además, el puntaje continuo de las filas de entrenamiento; nunca el de validación ni el de prueba"),
        ("Salidas", "regresión: el puntaje churn_probability previsto (la media de la mezcla), con su conjunto de predicción (una unión "
                    "de tramos, uno por nivel del puntaje) y su intervalo conforme; clasificación: la probabilidad de que el puntaje "
                    "supere el umbral de riesgo alto (probabilidad de excedencia de la mezcla)"),
        ("Hiperparámetros de clasificación", _hiperparametros_original("clasificacion")),
        ("Hiperparámetros de regresión", _hiperparametros_original("regresion")),
        ("Lo que aprendió en clasificación", _aprendido("clasificacion")),
        ("Lo que aprendió en regresión", _aprendido("regresion")),
        ("Semilla", F.entero(md["regresion"]["semilla"])),
    ]
    filas_art = [[f"artifacts/{a['ruta']}", _megas(a["bytes"]), a["sha256"][:HUELLA],
                  f"la tubería de scikit-learn con el modelo de {D.NOMBRE_TAREA[a['tarea']].lower()} (preprocesamiento y modelo)"]
                 for a in artefactos]
    filas_art.append([codigo["codigo"], RAYA_TXT, codigo["sha256_codigo"][:HUELLA],
                      "el código del modelo, copia exacta del de la corrección: las tuberías guardadas nombran este módulo"])
    for t in TAREAS:
        o = md[t]["origen"]
        filas_art.append([f"Corrección: {o['archivo']}", RAYA_TXT, o["sha256_origen"][:HUELLA], "el modelo final guardado por la corrección"])

    n_pliegues = int(e["externa_pliegues"])
    metricas = [
        ["AUC de prueba [IC 95 %]", f"{F.num(e['auc_prueba'], 4)} {F.intervalo(e['auc_bca_inf'], e['auc_bca_sup'])}",
         del_b(f"{F.num(x['auc_prueba'], 4)} {F.intervalo(x['auc_bca_inf'], x['auc_bca_sup'])}", bc),
         "qué tan bien ordena a los clientes por riesgo"],
        [f"AUC externo (media ± desviación de {F.entero(n_pliegues)} pliegues)", f"{F.num(e['externa_media'], 4)} ± {F.num(e['externa_desv'], 4)}",
         del_b(f"{F.num(x['externa_media'], 4)} ± {F.num(x['externa_desv'], 4)}", bc), "la validación anidada: con ella se decide el veredicto"],
        ["AUC-PR", F.num(e["auc_pr"], 3), del_b(F.num(x["auc_pr"], 3), bc), "orden de los positivos, sensible a la prevalencia"],
        [f"Captura con el {capacidad} de capacidad", F.pct(e["recall_op"]), del_b(F.pct(x["recall_op"]), bc),
         "parte de los clientes de riesgo alto que alcanza la campaña"],
        [f"Precisión con el {capacidad} de capacidad", F.pct(e["precision_op"]), del_b(F.pct(x["precision_op"]), bc),
         "parte de los contactados que son de riesgo alto"],
        ["Error de calibración (ECE): sin recalibrar → isotónica",
         f"{F.num(cal['ece_prueba_sin_recalibrar'], 3)} → {F.num(cal['ece_prueba_recalibrada'], 3)}",
         del_b(f"{F.num(cal_b['ece_prueba_sin_recalibrar'], 3)} → {F.num(cal_b['ece_prueba_recalibrada'], 3)}", bc) if cal_b else RAYA_TXT,
         "distancia entre la probabilidad dicha y la observada"],
        ["RMSE de prueba [IC 95 %]", f"{F.num(er['rmse_prueba'], 5)} {F.intervalo(er['rmse_bca_inf'], er['rmse_bca_sup'], 5)}",
         del_b(f"{F.num(xr['rmse_prueba'], 5)} {F.intervalo(xr['rmse_bca_inf'], xr['rmse_bca_sup'], 5)}", br), "error del puntaje previsto"],
        [f"RMSE externo (media ± desviación de {F.entero(int(er['externa_pliegues']))} pliegues)",
         f"{F.num(er['externa_media'], 5)} ± {F.num(er['externa_desv'], 5)}",
         del_b(f"{F.num(xr['externa_media'], 5)} ± {F.num(xr['externa_desv'], 5)}", br), "la validación anidada"],
        ["MAE y R² de prueba", f"{F.num(er['mae_prueba'], 4)} y {F.num(er['r2_prueba'], 3)}",
         del_b(f"{F.num(xr['mae_prueba'], 4)} y {F.num(xr['r2_prueba'], 3)}", br), "error absoluto medio y varianza explicada"],
        [f"Incertidumbre al {nivel}: cobertura · largo",
         f"{F.pct(conf['conjunto']['cobertura_prueba'])} · {F.num(conf['conjunto']['largo'], 4)} (conjunto de {F.entero(conf['conjunto']['tramos'])} tramos)",
         del_b(f"{F.pct(conf_b['cobertura_prueba'])} · {F.num(conf_b['ancho'], 4)} (intervalo conforme)", br),
         "parte de los puntajes reales dentro del conjunto o del intervalo, y cuánto mide"],
    ]
    veredicto = "; ".join(f"en {D.NOMBRE_TAREA[t].lower()}, {D.frase_veredicto(t)}" for t in TAREAS)
    limite = ""
    if te is not None:
        limite = (f" El límite teórico del conjunto en la prueba es un AUC de {F.num(te['prueba']['auc_maximo'], 4)} y un RMSE de "
                  f"{F.num(te['prueba']['rmse_minimo'], 5)}: los mejores benchmarks ya estaban a milésimas de él.")

    sub = D.original_tabla("errores_subgrupos.csv")
    if sub is not None and len(sub):
        b_r, b_c = sub["benchmark_regresion"].iloc[0], sub["benchmark_clasificacion"].iloc[0]
        filas_sub = [[f["subgrupo"], F.entero(f["clientes"]), F.num(f["rmse_original"], 5), F.num(f["rmse_benchmark"], 5),
                      F.num(f["auc_original"], 4) if f["auc_original"] == f["auc_original"] else RAYA_TXT,
                      F.num(f["auc_benchmark"], 4) if f["auc_benchmark"] == f["auc_benchmark"] else RAYA_TXT] for _, f in sub.iterrows()]
        segmento = [("tabla", ["Subgrupo de prueba", "Clientes", "RMSE de BDL", f"RMSE de {D.NOMBRE_MODELO[b_r]}", "AUC de BDL",
                               f"AUC de {D.NOMBRE_MODELO[b_c]}"], filas_sub, [False] + [True] * 5, (3, 5)),
                    ("nota", f"La cohorte de {mayor} años es la de más riesgo alto ({F.pct(coh.loc[mayor, 'prevalencia'])}) y la de {menor}, la "
                             f"de menos ({F.pct(coh.loc[menor, 'prevalencia'])}).{_lectura_subgrupos(sub)}")]
        fuentes_seg = [D.fuente_original("errores_subgrupos.csv"), D.fuente("resultados/errores_segmentos.csv")]
    else:
        segmento = [("nota", f"El análisis de errores por subgrupo del modelo original todavía no está entre los resultados de la "
                             f"corrección ({D.fuente_original('errores_subgrupos.csv')}). La ficha se completa al volver a correr "
                             "preparar_original.py y generar_fichas.py.")]
        fuentes_seg = [D.fuente_original("errores_subgrupos.csv")]

    lento = ""
    if semillas is not None:                     # el mismo guion mide a todos los modelos, con las mismas semillas y en el mismo equipo
        s = semillas[semillas["tarea"] == "regresion"]
        ajuste = s.groupby("modelo")["ajuste_s"].mean()
        if O in ajuste.index and br in ajuste.index and ajuste[O] > ajuste[br]:
            lento = (f"Ajustarlo cuesta más que ajustar los benchmarks de boosting: en regresión, {F.num(ajuste[O], 1)} s frente a "
                     f"{F.num(ajuste[br], 1)} s de {D.NOMBRE_MODELO[br]} (media de {F.entero(s['semilla'].nunique())} semillas); está "
                     "escrito en numpy, sin compilar.")

    # El corte de la campaña cae en un empate del puntaje de BDL: la captura, la precisión del punto de operación y el AUC-PR
    # dependen de cómo se rompan o se cuenten los empates. Las notas salen de empates_campana.json (D.frase_empate_*).
    em = D.empate_campana(O)
    nota_uso, nota_metricas, limite_empates, fuente_empates = [], [], "", []
    partes_metricas = []
    if em is not None:
        protocolo = em.get("captura_protocolo")
        del_libro = (f" El libro del proyecto usa el desempate del protocolo de la corrección (por posición) y da {F.pct(protocolo)}."
                     if isinstance(protocolo, (int, float)) and protocolo == protocolo else "")
        nota_uso = [("nota", D.frase_empate_captura(O, precision=True) + del_libro)]
        partes_metricas.append(f"La captura y la precisión con el {capacidad} de capacidad son las de {D.frase_empate_corta(O)}.{del_libro}")
        limite_empates = (f"Su probabilidad es escalonada ({F.entero(em['valores_distintos'])} valores distintos en la prueba): cuando el "
                          "corte de una campaña cae en una meseta, a quién se contacta dentro de ella lo decide el desempate, no el "
                          "modelo.")
    if D.frase_empate_auc_pr(O):
        partes_metricas.append(D.frase_empate_auc_pr(O))
    if partes_metricas:
        nota_metricas = [("nota", "Filas que dependen de los empates del puntaje. " + " ".join(partes_metricas))]
        fuente_empates = [D.fuente_original("empates_campana.json")]
    return [
        {"titulo": "Detalles del modelo", "columnas": "1-1",
         "subtitulo": "Qué es, quién lo hizo, con qué versión y con qué licencia; los artefactos y su huella.",
         "tarjetas": [
             {"titulo": "El modelo", "bloques": [("pares", detalles)],
              "fuentes": ["CITATION.cff", "artifacts/indice.json", D.fuente_original("experimentos.csv")]
                         + [f"artifacts/{t}/{O}/metadata.json" for t in TAREAS]},
             {"titulo": "Entradas, salidas e hiperparámetros", "bloques": [("pares", especificacion)],
              "fuentes": [f"artifacts/{t}/{O}/metadata.json" for t in TAREAS]
                         + [D.fuente_original(f"modelo_{t}.json") for t in TAREAS] + [D.fuente("eda/etl.json")]},
             {"titulo": "Artefactos y su huella", "ancho": True, "bloques": [
                 ("tabla", ["Archivo", "Tamaño", "SHA-256 (inicio)", "Qué es"], filas_art, [False, True, False, False], (3,)),
                 ("nota", f"La ida y vuelta de los artefactos dio predicciones idénticas a las guardadas en "
                          f"{F.entero(ida['clasificacion']['filas'])} clientes de prueba. Una prueba comprueba que el código del modelo es "
                          "idéntico al de la corrección. La huella completa está en artifacts/indice.json.")],
              "fuentes": ["artifacts/indice.json", D.fuente_original("estado.json")]},
         ]},
        {"titulo": "Para qué sirve y para qué no", "columnas": "3",
         "subtitulo": "El uso previsto, quién debería usarlo y los usos que no se recomiendan.",
         "tarjetas": [
             {"titulo": "Uso previsto", "bloques": [("lista", [
                 f"Ordenar a los clientes de COFINFAD por su riesgo para priorizar una campaña de retención con capacidad limitada: "
                 f"contactando al {capacidad} de mayor puntaje ({F.entero(op['contactados'])} de {F.entero(meta['n_prueba'])}) alcanza "
                 f"al {F.pct(e['recall_op'])} de los clientes de riesgo alto, con una precisión del {F.pct(e['precision_op'])}.",
                 f"Decir, además del pronóstico, en cuál de los niveles del puntaje puede estar un cliente: su conjunto de predicción al "
                 f"{nivel} es una unión de {F.entero(conf['conjunto']['tramos'])} tramos que cubre el "
                 f"{F.pct(conf['conjunto']['cobertura_prueba'])} de la prueba.",
                 ("Explicar qué variables mueven el índice del puntaje: sus modelos finales usan árboles de profundidad uno, así que el "
                  "índice es la suma exacta de lo que aporta cada variable." if all(aditivos) else
                  "Explicar qué variables mueven el índice del puntaje, con aportes exactos calculados por el propio modelo."),
                 "Docencia e investigación: un modelo original con su formulación, sus demostraciones y su código, comparado con los "
                 "benchmarks con un protocolo registrado antes de los experimentos.",
             ])] + nota_uso,
              "fuentes": [D.fuente_original("lideres_clasificacion.csv"), D.fuente("conformal.json"), D.fuente("metadatos.json")]
                         + fuente_empates},
             {"titulo": "Usuarios previstos", "bloques": [("lista", [
                 "El profesor y los estudiantes del curso de Machine Learning.",
                 "Analistas que quieren estudiar un modelo con una variable latente y compararlo con los ensambles de árboles.",
                 "Equipos de retención que quieren ver el método, no aplicarlo sin antes validarlo con sus propios datos.",
             ])], "fuentes": [LIBRO]},
             {"titulo": "Usos no recomendados", "bloques": [("lista", [
                 "Decidir sobre clientes reales: los datos son sintéticos y el escalón que el modelo aprovecha existe porque así se "
                 "generó el conjunto.",
                 "Decisiones de crédito, precio, cobranza o exclusión de un servicio.",
                 "Leer su probabilidad como probabilidad de fuga observada: es la de que el puntaje del conjunto supere su umbral.",
                 "Usarlo donde la etiqueta no sea el umbral de una variable continua observada en el entrenamiento: sin ella el "
                 "modelo aprende de la etiqueta y pierde su ventaja.",
                 f"Puntuar valores fuera del rango de entrenamiento: la edad, por ejemplo, solo toma {F.entero(edad['n_distintos'])} valores "
                 f"({str(edad['valores']).replace('|', ', ')}).",
                 "Tratar «riesgo alto» como fuga observada, o usar productos activos como entrada: es la variable con que se generó el "
                 "puntaje y el modelo recuperaría la regla en vez de predecir.",
                 "Llevarlo a otra entidad, otro país u otro periodo sin reentrenarlo y validarlo.",
             ])], "fuentes": [D.fuente("variables.csv"), LIBRO]},
         ]},
        {"titulo": "Factores, métricas y datos", "columnas": "1-1",
         "subtitulo": "Con qué se evaluó, en qué datos y en qué punto de operación. Al lado va el mejor benchmark de cada tarea en la "
                      "validación anidada.",
         "tarjetas": [
             {"titulo": "Factores relevantes", "bloques": [("lista", [
                 f"Cohorte de edad: el conjunto tiene {F.entero(len(coh))} cohortes y el riesgo alto va del {F.pct(coh.loc[menor, 'prevalencia'])} "
                 f"({menor} años) al {F.pct(coh.loc[mayor, 'prevalencia'])} ({mayor}). La edad reúne el {F.pct(imp['age'])} de la "
                 "importancia del modelo en clasificación.",
                 "Productos activos: la variable excluida con la que se generó el puntaje. El modelo no la ve; la trata como una clase "
                 "latente y recupera sus niveles.",
                 texto_personales,
             ])], "fuentes": [D.fuente("resultados/errores_segmentos.csv"), D.fuente("shap/indice.json")]},
             {"titulo": "Datos de entrenamiento y de evaluación", "bloques": [("pares", [
                 ("Conjunto", "COFINFAD (Mendeley Data, CC BY 4.0), sintético: ver DATA_STATEMENT.md"),
                 ("Partición", f"{_particion(meta)}, antes de derivar variables y de calcular cualquier estadístico del objetivo"),
                 ("Entrenamiento", f"{F.entero(meta['n_entrenamiento'])} clientes; riesgo alto: {F.pct(meta['prevalencia_entrenamiento'])}. "
                                   "El modelo reparte esas filas en ajuste, validación y calibración"),
                 ("Evaluación", f"{F.entero(meta['n_prueba'])} clientes de prueba; riesgo alto: {F.pct(meta['prevalencia_prueba'])} "
                                f"({F.entero(meta['positivos_prueba'])} clientes). La prueba ya se había usado para evaluar los benchmarks "
                                "del Entregable 3; no intervino en el diseño del modelo"),
                 ("Etiqueta", f"riesgo alto = churn_probability por encima de su percentil {F.entero(meta['etiqueta']['q'] * 100)} en el "
                              f"entrenamiento ({F.num(meta['etiqueta']['umbral'], 4)})"),
                 ("Selección", f"validación cruzada anidada de {F.entero(md['regresion']['metricas_bucle_externo']['pliegues_externos'])} × "
                               f"{F.entero(md['regresion']['metricas_bucle_externo']['pliegues_internos'])} pliegues, con "
                               f"{F.entero(costo['regresion']['presupuesto']) if costo['regresion'] is not None else RAYA_TXT} configuraciones "
                               "por búsqueda y una sola combinación, fijada en el protocolo"),
                 ("Punto de operación", f"contactar al {capacidad} de mayor puntaje: {F.entero(op['contactados'])} clientes"),
             ])], "fuentes": [D.fuente("metadatos.json"), D.fuente_original("costo.csv"), PROTOCOLO]},
             {"titulo": "Métricas en la prueba y en la validación anidada", "ancho": True, "bloques": [
                 # las columnas de valores van como texto: alineadas a la derecha y sin salto, en el teléfono las cifras cortas
                 # (AUC-PR, captura, precisión) quedaban fuera del marco de la tabla
                 ("tabla", ["Métrica", "BDL", rotulo_b, "Qué mide"], metricas, [False, False, False, False], (3,)),
                 ("nota", f"Con la regla de superioridad de la guía, el veredicto del modelo original es: {veredicto}.{limite}")]
                + nota_metricas + [("nota", D.nota_ece())],
              "fuentes": [D.fuente("resultados/clasificacion_prueba.csv"), D.fuente("resultados/regresion_prueba.csv"),
                          D.fuente_original("lideres_clasificacion.csv"), D.fuente_original("lideres_regresion.csv"),
                          D.fuente("calibracion_modelos.json"), D.fuente("conformal.json"), D.fuente_original("veredicto.json"),
                          D.fuente_original("condicion4.json"), D.fuente_original("techo.json")] + fuente_empates},
         ]},
        {"titulo": "Análisis por segmento", "columnas": "1-1",
         "subtitulo": "La prueba dividida por edad, por la cola de transacciones y por segmento de cliente: el modelo original frente "
                      "al mejor benchmark de cada tarea.",
         "tarjetas": [{"titulo": "Dónde acierta y dónde falla", "ancho": True, "bloques": segmento, "fuentes": fuentes_seg}]},
        {"titulo": "Ética, limitaciones y recomendaciones", "columnas": "3",
         "subtitulo": "Lo que hay que saber antes de usarlo.",
         "tarjetas": [
             {"titulo": "Consideraciones éticas", "bloques": [("lista", [
                 "Los datos son públicos y sintéticos: no hay personas reales identificables.",
                 f"La edad, el género y la ubicación son entradas. La edad domina el modelo ({F.pct(imp['age'])} de la importancia): una "
                 "campaña real que lo siguiera trataría distinto a las personas según su edad, y eso habría que justificarlo y revisarlo.",
                 "No se mide fuga real: usarlo para negar o encarecer un servicio no tendría respaldo en estos datos.",
             ])], "fuentes": [D.fuente("shap/indice.json"), LIBRO]},
             {"titulo": "Limitaciones y riesgos", "bloques": [("lista", [x_ for x_ in [
                 "Los datos son sintéticos: el escalón que el modelo aprovecha existe porque el generador resta una cantidad fija por "
                 "producto activo; en datos reales un efecto así sería menos limpio.",
                 "El margen de mejora era mínimo: los mejores benchmarks ya estaban a milésimas del límite teórico. Lo que el modelo "
                 "añade es la estructura del puntaje y una incertidumbre mejor descrita, no precisión puntual.",
                 "Supone un desplazamiento aditivo, independiente de los predictores y con la misma dispersión en todas las clases. "
                 "Cuando no se cumple, el criterio BIC deja una sola clase y el modelo es un boosting de mínimos cuadrados.",
                 "Su probabilidad salta de una meseta a otra en un intervalo estrecho del índice: un cliente mal situado en ese borde "
                 "recibe una probabilidad casi segura y equivocada.",
                 f"El intervalo simétrico «pronóstico ± q» no le sirve: al {nivel} cubre el {F.pct(conf['cobertura_prueba'])} de la prueba, "
                 "porque su error cae en niveles separados. Su conjunto de predicción, con la misma cobertura nominal, cubre el "
                 f"{F.pct(conf['conjunto']['cobertura_prueba'])}; esa garantía es marginal: vale sobre el total de clientes, no dentro de "
                 "cada grupo.",
                 f"Los conjuntos conformes de clases cubren de más ({F.pct(conj['cobertura_prueba'])} al {nivel}): el "
                 f"{F.pct(conj['fraccion_ambos'], 0)} de los clientes recibe las dos clases.",
                 limite_empates,
                 lento,
             ] if x_])], "fuentes": [D.fuente("conformal.json"), D.fuente("calibracion_modelos.json"),
                                     D.fuente_original("semillas.csv"), LIBRO] + fuente_empates},
             {"titulo": "Recomendaciones", "bloques": [("lista", [
                 "Ordenar a los clientes con la probabilidad del modelo y expresar la incertidumbre del pronóstico con su conjunto de "
                 "predicción, no con el intervalo simétrico.",
                 "Validar en el tiempo con fuga observada (entrenar con un periodo y probar con el siguiente) antes de cualquier uso real.",
                 "Revisar los resultados por cohorte de edad y reentrenar si cambia la distribución de los clientes.",
                 "Mantener una revisión humana de cada decisión que afecte a un cliente.",
             ])], "fuentes": [LIBRO]},
         ]},
    ]


def contenido_ficha_ebm() -> list[dict]:
    """La ficha de la EBM, como en el Entregable 3 (respaldo mientras el modelo original no esté en la app)."""
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
                          f"la conclusión es que la EBM alcanza a los modelos de árboles, no que los supere."),
                 ("nota", D.nota_ece())],
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
    auditoria = D.original_json("auditoria_fuga.json")          # la corrección de la recencia, si la corrección ya la dejó
    recencia = []
    if auditoria is not None:
        corte = auditoria["fecha_de_corte"]
        recencia = [f"La recencia se mide hasta una fecha de corte que sale solo de los clientes de entrenamiento "
                    f"({_fecha(corte['correccion_solo_entrenamiento'])}); antes salía de todo el detalle de transacciones. La auditoría de "
                    f"la corrección comprueba que cambian {F.entero(corte['celdas_de_recencia_que_cambian'])} celdas y que los predictores de "
                    f"entrenamiento y de prueba son {'idénticos' if corte['predictores_identicos'] else 'distintos'} antes y después."]
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
             ] + recencia)], "fuentes": [D.fuente("eda/etl.json"), D.fuente("metadatos.json")]
                             + ([D.fuente_original("auditoria_fuga.json")] if recencia else [])},
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
    modelo = (f"del modelo original, BDL ({D.NOMBRE_LARGO_ORIGINAL}),"
              if hay_original() else "de la Explainable Boosting Machine")
    return [f"Ficha del modelo (*model card*, Mitchell et al., 2019) {modelo} del proyecto «{cff['titulo']}». "
            f"Versión {cff['version']}, {_fecha(cff['fecha'])}. La genera `generar_fichas.py` desde `datos/` y `artifacts/`: ninguna cifra "
            "se escribe a mano. La misma ficha está en la vista «Ficha del modelo» del dashboard (/modelos/ficha)."]


def entrada_datos() -> list[str]:
    cff = cita_cff()
    return [f"Declaración de los datos (en la línea de Bender y Friedman, 2018, y Gebru et al., 2021) del conjunto que usa el proyecto "
            f"«{cff['titulo']}». La genera `generar_fichas.py` desde `datos/`: ninguna cifra se escribe a mano."]


def main() -> int:
    FICHA.write_text(a_markdown(titulo_ficha(), entrada_ficha(), contenido_ficha()), encoding="utf-8", newline="\n")
    DECLARACION.write_text(a_markdown("Declaración de los datos: COFINFAD", entrada_datos(), contenido_datos()), encoding="utf-8",
                           newline="\n")
    for ruta in (FICHA, DECLARACION):
        print(f"{ruta.name}: {len(ruta.read_text(encoding='utf-8').splitlines())} líneas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
