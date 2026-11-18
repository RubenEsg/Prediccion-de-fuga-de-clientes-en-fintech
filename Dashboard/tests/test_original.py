"""Pruebas del modelo original (BDL) dentro del dash: su código, sus artefactos, lo propio de sus salidas y su veredicto.

    pytest -q tests/test_original.py

El código del modelo vive en src/original_model.py y debe ser idéntico al de la corrección (ModeloOriginal/src): la prueba
compara los dos archivos enteros (sin contar el tipo de salto de línea, que Git puede cambiar en una copia de trabajo de
Windows) cuando esa carpeta está a la vista (en ../ModeloOriginal, como en el repositorio publicado; en
../entrega 3/repo_entregable3/ModeloOriginal, como en la carpeta de trabajo; o en la variable MODELO_ORIGINAL) y, siempre,
contra la huella que anotó preparar_original.py. La ida y vuelta de sus dos artefactos, su explicación y su metadata se
prueban con los demás modelos en tests/test_modelos.py (V.modelos incluye a «bdl»); aquí va lo que solo tiene el original.
El veredicto de superioridad se vuelve a calcular desde veredicto.json y condicion4.json con la regla del protocolo y se
compara con el que muestra la app: ninguna vista lo escribe a mano.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import RAIZ

import datos_app as D
import formato as F
import modelos_vivos as V
import motor as M

O = V.ORIGINAL
TAREAS = ("clasificacion", "regresion")
CODIGO = RAIZ / "src" / "original_model.py"
ORIGINAL = RAIZ / "datos" / "original"
PERMITIDAS = {"__future__", "numpy", "scipy", "sklearn.base", "sklearn.utils.validation"}     # lo único que importa el modelo
NIVELES = ("0.80", "0.90", "0.95")
TOLERANCIA_COBERTURA = 0.02


def _huella(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _carpeta_de_la_correccion() -> Path | None:
    """La carpeta ModeloOriginal de la corrección, si está a la vista."""
    candidatas = [os.environ.get("MODELO_ORIGINAL"), RAIZ.parent / "ModeloOriginal",
                  RAIZ.parent / "entrega 3" / "repo_entregable3" / "ModeloOriginal"]
    for c in candidatas:
        if c and (Path(c) / "src" / "original_model.py").is_file():
            return Path(c)
    return None


@pytest.fixture(scope="module")
def prueba() -> pd.DataFrame:
    return M.clientes()[V.columnas()]


# --- El código ---------------------------------------------------------------------------------------------------------------
def test_el_codigo_es_identico_al_de_la_correccion():
    carpeta = _carpeta_de_la_correccion()
    if carpeta is None:
        pytest.skip("la carpeta ModeloOriginal no está a la vista (ni en ../ModeloOriginal ni en MODELO_ORIGINAL)")
    propio, suyo = (p.read_bytes().replace(b"\r\n", b"\n") for p in (CODIGO, carpeta / "src" / "original_model.py"))
    assert propio == suyo, "src/original_model.py difiere del de la corrección: vuelva a correr preparar_original.py"


def test_la_huella_del_codigo_es_la_que_anoto_preparar_original():
    """Los artefactos se guardaron con este mismo código: la huella de src/original_model.py es la de artifacts/indice.json,
    la de datos/original/estado.json y la del metadata.json de cada artefacto. (Se admite también la huella con los saltos
    de línea de Unix: Git puede dejar el archivo con CRLF en una copia de trabajo de Windows sin que cambie su contenido.)"""
    crudo = CODIGO.read_bytes()
    huellas = {hashlib.sha256(crudo).hexdigest(), hashlib.sha256(crudo.replace(b"\r\n", b"\n")).hexdigest()}
    assert V.indice()["original"]["sha256_codigo"] in huellas
    assert D.estado_original()["codigo"]["sha256"] in huellas
    for t in TAREAS:
        assert V.metadata(t, O)["origen"]["sha256_codigo"] in huellas


def test_el_modelo_original_no_importa_ningun_modelo():
    """El módulo solo importa numpy, scipy y las clases base y los validadores de scikit-learn: ningún modelo de librería."""
    arbol = ast.parse(CODIGO.read_text(encoding="utf-8"))
    importados = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            importados |= {a.name for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom):
            importados.add(nodo.module or "")
    fuera = {m for m in importados if not any(m == p or m.startswith(p + ".") for p in PERMITIDAS)}
    assert not fuera, f"el modelo original importa {sorted(fuera)}"


# --- Los artefactos ----------------------------------------------------------------------------------------------------------
def test_el_original_esta_en_vivo_en_las_dos_tareas():
    for t in TAREAS:
        assert V.hay_original(t) and V.modelos(t)[0] == O
        a = next(x for x in V.indice()["artefactos"] if x["tarea"] == t and x["modelo"] == O)
        assert _huella(V.ARTEFACTOS / a["ruta"]) == a["sha256"]
        assert a["ida_y_vuelta"] == "idénticas"


@pytest.mark.parametrize("tarea", TAREAS)
def test_la_tuberia_del_original_es_de_scikit_learn_y_usa_el_codigo_del_dash(tarea):
    """La tubería guardada es una sklearn.pipeline.Pipeline (sin imblearn) cuyo modelo es una clase de src/original_model.py."""
    from sklearn.pipeline import Pipeline
    from src import original_model
    pipe = V.tuberia(tarea, O)
    assert type(pipe) is Pipeline and list(pipe.named_steps) == ["preprocesado", "modelo"]
    esperada = original_model.ClasificadorBDL if tarea == "clasificacion" else original_model.RegresorBDL
    assert type(pipe.named_steps["modelo"]) is esperada
    assert Path(original_model.__file__).resolve() == CODIGO.resolve()


@pytest.mark.parametrize("tarea", TAREAS)
def test_ida_y_vuelta_del_original(tarea, prueba):
    """Cargado de disco, el artefacto del original da en los clientes de prueba exactamente lo que guardó la corrección
    (datos/puntajes_prueba.parquet y pronosticos_prueba.parquet) y lo mismo en dos cargas distintas."""
    import joblib
    guardadas = pd.read_parquet(V.DATOS / ("puntajes_prueba.parquet" if tarea == "clasificacion" else "pronosticos_prueba.parquet"))
    s = V.predecir(tarea, O, prueba)
    assert np.max(np.abs(s - guardadas.loc[prueba.index, O].to_numpy(dtype=float))) < 1e-9
    otra = joblib.load(V.ARTEFACTOS / tarea / O / "modelo.joblib")
    b = otra.predict(prueba) if tarea == "regresion" else otra.predict_proba(prueba)[:, 1]
    assert np.array_equal(s, np.asarray(b, dtype=float))


# --- Lo propio de sus salidas ------------------------------------------------------------------------------------------------
def test_la_probabilidad_es_la_excedencia_de_la_mezcla(prueba):
    """Clasificación: la probabilidad del modelo es la suma, sobre las clases latentes, del peso de cada una por la
    probabilidad de que su nivel supere el umbral, acotada por el piso y el techo de la muestra de calibración."""
    X = prueba.iloc[:500]
    e = V.excedencia(X)
    if e is None:
        pytest.skip("el clasificador final está en modo directo (sin mezcla)")
    assert e["niveles"].shape == (len(X), len(e["pesos"])) and abs(float(np.sum(e["pesos"])) - 1) < 1e-9
    np.testing.assert_allclose(np.clip(e["aporte"].sum(axis=1), e["piso"], e["techo"]), e["probabilidad"], atol=1e-12)
    np.testing.assert_array_equal(e["probabilidad"], V.predecir("clasificacion", O, X))


def test_el_conjunto_de_prediccion_es_una_union_de_tramos(prueba):
    """Regresión: el conjunto de predicción de un cliente es su índice más los tramos del conjunto residual: ordenados, sin
    solaparse, con el largo que anota datos/conformal.json, y el pronóstico es la media de la mezcla. (Un tramo puede
    reducirse a un punto: el nivel menos probable entra solo por su moda cuando el umbral de densidad lo roza.)"""
    X = prueba.iloc[[0]]
    for nivel in NIVELES:
        d = V.distribucion(X, nivel)
        medido = V.conformal()["modelos"][O][nivel]["conjunto"]
        tramos = d["tramos"]
        assert len(tramos) == medido["tramos"] and all(a <= b for a, b in tramos)
        assert all(tramos[i][1] < tramos[i + 1][0] for i in range(len(tramos) - 1))
        assert d["largo"] == pytest.approx(medido["largo"], abs=1e-12)
        assert d["media"] == pytest.approx(V.predecir("regresion", O, X)[0], abs=1e-12)
        assert d["media"] == pytest.approx(d["indice"] + float(np.dot(d["pesos"], np.subtract(d["niveles"], d["indice"]))), abs=1e-12)


def test_la_cobertura_del_conjunto_se_reproduce_y_cumple_lo_nominal(prueba):
    """La cobertura del conjunto de predicción que muestra la app es la que da el artefacto en los clientes de prueba, queda a
    menos de dos puntos de la nominal y el conjunto es más corto que el intervalo conforme simétrico."""
    pre, _est, reg = V._original("regresion")
    z = pre.transform(prueba)
    y = D.pronosticos()["y"].loc[prueba.index].to_numpy(dtype=float)
    for nivel in NIVELES:
        medido = V.conformal()["modelos"][O][nivel]
        cobertura = float(np.mean(reg.conjunto_cubre(z, y, float(nivel))))
        assert cobertura == pytest.approx(medido["conjunto"]["cobertura_prueba"], abs=1e-12)
        assert abs(cobertura - float(nivel)) < TOLERANCIA_COBERTURA
        assert medido["conjunto"]["largo"] < medido["ancho"]


@pytest.mark.parametrize("tarea", TAREAS)
def test_la_explicacion_del_original_es_exacta(tarea, prueba):
    """Sus aportes salen de su propio código, son exactos y suman, con la base, la salida que se explica: el pronóstico en
    regresión y el índice F(x) en clasificación."""
    X = prueba.iloc[:200]
    e = V.explicar(tarea, O, X)
    assert e["metodo"] == "bdl" and e["exacto"] is True
    np.testing.assert_allclose(e["base"] + e["aportes"].sum(axis=1).to_numpy(), e["salida"], atol=1e-9)
    if tarea == "regresion":
        np.testing.assert_allclose(e["salida"], V.predecir(tarea, O, X), atol=1e-9)
    elif V.excedencia(X) is not None:
        np.testing.assert_allclose(e["salida"], V.excedencia(X)["indice"], atol=1e-9)


# --- Las tablas con los diez modelos -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("tarea", TAREAS)
def test_la_tabla_de_lideres_tiene_los_diez_modelos(tarea):
    t = D.lideres(tarea)
    metrica = "auc" if tarea == "clasificacion" else "rmse"
    assert list(t["modelo"]) == D.ORDEN_MODELOS[tarea] and len(t) == 10
    assert set(t.loc[t["origen"] == "corrección", "modelo"]) == set(D.NUEVOS)
    for col in (f"{metrica}_prueba", f"{metrica}_bca_inf", f"{metrica}_bca_sup", "externa_media", "externa_desv"):
        assert t[col].notna().all(), col
    assert ((t[f"{metrica}_bca_inf"] <= t[f"{metrica}_prueba"]) & (t[f"{metrica}_prueba"] <= t[f"{metrica}_bca_sup"])).all()
    columnas = pd.read_parquet(V.DATOS / ("puntajes_prueba.parquet" if tarea == "clasificacion" else "pronosticos_prueba.parquet")).columns
    assert set(D.ORDEN_MODELOS[tarea]) <= set(columnas)


@pytest.mark.parametrize("tarea", TAREAS)
def test_los_rangos_de_cd_json_salen_de_los_pliegues_de_la_correccion(tarea):
    """datos/cd.json tiene los diez modelos y sus rangos medios son los de datos/original/pliegues_<tarea>.csv."""
    cd = json.loads((V.DATOS / "cd.json").read_text(encoding="utf-8"))[tarea]
    p = D.pliegues(tarea)
    assert cd["modelos"] == 10 and cd["pliegues"] == len(p) and set(cd["rango_medio"]) == set(p.columns)
    rangos = p.rank(axis=1, ascending=(tarea == "regresion"), method="average").mean()
    for m, r in cd["rango_medio"].items():
        assert r == pytest.approx(float(rangos[m]), abs=1e-9)


# --- El veredicto ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tarea", TAREAS)
def test_el_veredicto_sale_de_los_archivos(tarea):
    """El veredicto que muestran las vistas (D.estado_veredicto) es el que da la regla del protocolo aplicada, aquí aparte,
    a veredicto.json y condicion4.json: superior solo con las cuatro condiciones; pendiente si cumple las tres primeras y la
    cuarta no está completa; no superior en cualquier otro caso. Y las medias del veredicto son las de los pliegues."""
    v = json.loads((ORIGINAL / "veredicto.json").read_text(encoding="utf-8"))[tarea]
    ruta4 = ORIGINAL / "condicion4.json"
    c4 = json.loads(ruta4.read_text(encoding="utf-8")).get(tarea, {}) if ruta4.is_file() else {}
    tres = bool(v["condicion_1_media_mejor"] and v["condicion_2_tres_pruebas"] and v["condicion_3_efecto"])
    completa = bool(c4.get("completo"))
    esperado = "superior" if tres and completa and c4.get("cumple") else ("pendiente" if tres and not completa else "no superior")
    e = D.estado_veredicto(tarea)
    assert e["palabra"] == esperado
    assert e["cond"] == [bool(v["condicion_1_media_mejor"]), bool(v["condicion_2_tres_pruebas"]), bool(v["condicion_3_efecto"])]
    assert e["provisional"] == (len(v["repeticiones"]) < D.REPETICIONES_PROTOCOLO)
    assert D.frase_veredicto(tarea).startswith(D.VEREDICTO_ORIGINAL[esperado])
    p = D.pliegues(tarea)
    assert len(p) == v["pliegues"]
    for m, media in v["medias"].items():
        assert media == pytest.approx(float(p[m].mean()), abs=1e-9)
    signo = 1 if tarea == "clasificacion" else -1
    ventaja = signo * (v["medias"][O] - v["medias"][v["mejor_benchmark"]])
    assert v["ventaja_sobre_el_mejor"] == pytest.approx(ventaja, abs=1e-9)
    assert bool(v["condicion_1_media_mejor"]) == (ventaja > 0)


def _texto(nodo) -> str:
    """Todo el texto de un árbol de componentes de Dash."""
    if isinstance(nodo, str):
        return nodo
    if isinstance(nodo, (list, tuple)):
        return " ".join(_texto(n) for n in nodo)
    hijos = getattr(nodo, "children", None)
    return _texto(hijos) if hijos is not None else ""


def test_las_vistas_dicen_el_veredicto_de_los_archivos():
    """La vista del modelo original, la comparación, la tabla de líderes, Contexto y la ficha escriben el veredicto que da la
    regla (D.estado_veredicto y D.frase_veredicto): el texto que arman contiene esa palabra en cada tarea."""
    import app  # noqa: F401  (crea la app y registra las páginas)
    from paginas import contexto
    from secciones import comparacion, ficha, lideres, original
    for tarea in TAREAS:
        palabra = D.VEREDICTO_ORIGINAL[D.estado_veredicto(tarea)["palabra"]]
        assert palabra in comparacion._respuesta(tarea)
        assert palabra in lideres._respuesta(tarea)
        assert palabra in _texto(contexto._heroe_original(tarea))
        assert f"en {D.NOMBRE_TAREA[tarea].lower()}, {palabra}" in ficha._respuesta()
        if D.estado_veredicto(tarea)["palabra"] != "pendiente":
            assert original._frase_veredicto(tarea).endswith(f": {D.estado_veredicto(tarea)['palabra']}")
    palabras = {D.VEREDICTO_ORIGINAL[D.estado_veredicto(t)["palabra"]] for t in TAREAS}
    assert all(p in contexto._subtitulo() for p in palabras)


def test_la_novedad_no_afirma_mas_que_el_libro():
    """La sección de novedad sigue la declaración del libro (sección 4.5): tiene sus cuatro bloques y no presenta como nueva
    la regla de aprendizaje, un EM con paso de boosting que pertenece a una familia ya publicada. Y la tarjeta de la búsqueda
    dice lo que dice datos/original/literatura.json: cuántas cadenas, cuántas veces se hizo y en qué fechas."""
    import app  # noqa: F401  (crea la app y registra las páginas)
    import ui
    from secciones import original
    texto = _texto(original._seccion_novedad())
    for bloque in ("Qué no es nuevo", "Qué es propio de BDL", "Qué se reutiliza", "Qué no se afirma"):
        assert bloque in texto, bloque
    assert "Qué es nuevo" not in texto
    assert "ya se había hecho" in texto and "no su origen" in texto
    ruta = ORIGINAL / "literatura.json"
    if not ruta.is_file():
        assert "todavía no está" in _texto(original._tarjeta_busqueda())
        return
    lit = json.loads(ruta.read_text(encoding="utf-8"))
    busqueda = _texto(original._tarjeta_busqueda())
    assert f"{original.CARDINAL[lit['cadenas']]} cadenas" in busqueda
    assert lit["cadenas"] == len({f["cadena"] for f in lit["matriz"]}) and len(lit["matriz"]) == lit["cadenas"] * len(lit["bases"])
    consultas = original._consultas(lit)
    assert consultas[-1]["fecha"] == lit["fecha_de_consulta"]
    for consulta in consultas:
        assert ui.fecha_larga(consulta["fecha"], con_dia=False) in busqueda
    assert ("hecha dos veces" in busqueda) == (len(consultas) == 2) == ("La búsqueda se hizo dos veces" in busqueda)


def test_el_analisis_del_predictor_optimo_dice_lo_que_dice_su_archivo():
    """La tarjeta y la frase del predictor óptimo salen de datos/original/optimo.json: dicen que el óptimo rechaza donde lo
    dice su p ajustado, los p ajustados de BDL que muestra la tarjeta son los de veredicto.json y la frase entra en el
    análisis de la causa de un «no superior»."""
    ruta = ORIGINAL / "optimo.json"
    if not ruta.is_file():
        pytest.skip("la corrección todavía no ha escrito optimo.json")
    import app  # noqa: F401  (crea la app y registra las páginas)
    from secciones import original
    optimo = json.loads(ruta.read_text(encoding="utf-8"))
    veredicto = json.loads((ORIGINAL / "veredicto.json").read_text(encoding="utf-8"))
    tarjeta = original._tarjeta_optimo()
    for tarea in TAREAS:
        o = optimo[tarea]
        if o["mejor_benchmark"] != veredicto[tarea]["mejor_benchmark"]:
            assert original._optimo(tarea) is None           # un análisis hecho frente a otro benchmark no se muestra
            continue
        sin_rechazo = [k for k in original.PRUEBAS_EN_PRUEBA if not o["optimo"][k]["p_holm"] < D.ALFA]
        assert sin_rechazo == [k for k in original.PRUEBAS_EN_PRUEBA if not o["optimo"][k]["rechaza"]]
        frase = original._frase_optimo(tarea)
        assert ("ni el predictor óptimo" in frase) == (len(sin_rechazo) == len(original.PRUEBAS_EN_PRUEBA))
        assert ("sí rechaza" in frase) == (not sin_rechazo)
        for k in original.PRUEBAS_EN_PRUEBA:
            assert original._p(o["optimo"][k]["p_holm"]) in frase
            assert original._p(veredicto[tarea]["frente_al_mejor"][k]["p_holm"]) in _texto(tarjeta)
        if D.estado_veredicto(tarea)["palabra"] == "no superior":
            assert frase in original._causa(tarea)


def test_la_ablacion_cuenta_lo_que_dice_su_archivo():
    """El título de la tarjeta de la ablación cuenta, con datos/original/ablacion.csv, las variantes a las que el modelo
    completo supera en los pliegues externos, cuántas de esas diferencias son significativas y a cuántas supera en la prueba."""
    ruta = ORIGINAL / "ablacion.csv"
    if not ruta.is_file():
        pytest.skip("la corrección todavía no ha escrito ablacion.csv")
    import app  # noqa: F401
    from secciones import original
    t = pd.read_csv(ruta)
    v = t[t["variante"] != O]
    delante = int((v["aporte_cv"] > 0).sum())
    claras = int(((v["aporte_cv"] > 0) & (v["p_t_corregida_holm"] < D.ALFA)).sum())
    en_prueba = int((v["aporte_prueba"] > 0).sum())
    texto = _texto(original._tarjeta_ablacion())
    assert f"queda por delante de {original._cuantas_de(delante, len(v))} variantes" in texto
    assert f"({original.CARDINAL[claras]} con diferencia significativa)" in texto
    assert f"en la prueba, de {original._cuantas_de(en_prueba, len(v))}" in texto
    assert all(str(nombre) in texto for nombre in t["nombre"])


def test_el_ruido_no_cuenta_los_escenarios_a_medias(monkeypatch):
    """El guion de robustez escribe ruido.csv fila a fila: un escenario al que todavía le falta algún modelo se muestra
    «en curso», no entra en la cuenta del título y la nota dice qué contaminaciones faltan."""
    import app  # noqa: F401
    from secciones import original
    filas = [("sin contaminar", 0.0, m, x) for m, x in ((O, 0.0588), ("xgboost", 0.0590), ("lightgbm", 0.0591))]
    filas.append(("ruido gaussiano en el objetivo", 0.01, O, 0.0700))            # a medias: solo está el original
    tabla = pd.DataFrame([{"escenario": e, "nivel": n, "tarea": "regresion", "modelo": m, "media": x, "desv": 0.0} for e, n, m, x in filas])
    real = D.original_tabla
    monkeypatch.setattr(D, "original_tabla", lambda rel: tabla if rel == "robustez/ruido.csv" else real(rel))
    texto = _texto(original._tarjeta_ruido())
    assert "en 1 de 1 escenarios en regresión" in texto
    assert "en curso" in texto
    assert "todavía faltan los de atípicos en el objetivo y ruido en los predictores numéricos" in texto


# --- Los empates en el corte de la campaña (datos/original/empates_campana.json) --------------------------------------------
EMPATES = ORIGINAL / "empates_campana.json"


def _empates() -> dict:
    if not EMPATES.is_file():
        pytest.skip("la corrección todavía no ha escrito empates_campana.json")
    return json.loads(EMPATES.read_text(encoding="utf-8"))


def test_los_empates_de_la_campana_son_los_de_los_puntajes_del_dash():
    """El análisis de la corrección habla de la misma campaña y de los mismos puntajes que muestra el dash: el tamaño de la
    campaña y los positivos coinciden; los empates del original en el corte se recuentan de datos/puntajes_prueba.parquet y
    dan lo mismo; el AUC-PR de cada modelo es el publicado; y la captura que muestra el dash para cada modelo, con su propio
    desempate, cae dentro del rango posible."""
    e = _empates()
    P = D.puntajes()
    y = P["y"].to_numpy(dtype=int)
    assert e["n"] == len(P) and e["positivos"] == int(y.sum())
    assert e["campana"] == M.meta()["punto_operacion"]["contactados"] == int(round(M.CAPACIDAD_INFORME * len(P)))
    lideres = D.lideres("clasificacion").set_index("modelo")
    for modelo, v in e["modelos"].items():
        if modelo not in P.columns:
            continue
        captura = M.resumen_capacidad(y, P[modelo].to_numpy(dtype=float))["captura"]
        assert v["captura_minima"] - 1e-12 <= captura <= v["captura_maxima"] + 1e-12, modelo
        assert lideres.loc[modelo, "recall_op"] == pytest.approx(captura, abs=1e-12), modelo
        assert lideres.loc[modelo, "auc_pr"] == pytest.approx(v["precision_media"], abs=1e-6), modelo
    s, v = P[O].to_numpy(dtype=float), e["modelos"][O]
    corte = np.sort(s)[::-1][e["campana"] - 1]
    arriba, iguales = s > corte, s == corte
    assert corte == pytest.approx(v["corte"], abs=1e-12)
    assert (int(iguales.sum()), e["campana"] - int(arriba.sum()), int(y[iguales].sum()), int(y[arriba].sum()), len(np.unique(s))) == \
        (v["empatados"], v["cupos"], v["positivos_empatados"], v["positivos_seguros"], v["valores_distintos"])
    esperada = (v["positivos_seguros"] + v["cupos"] * v["positivos_empatados"] / v["empatados"]) / e["positivos"]
    assert esperada == pytest.approx(v["captura_esperada"], abs=1e-12)


def test_las_notas_de_los_empates_salen_del_archivo():
    """Las frases que acompañan a la captura, a la precisión y al AUC-PR se arman con empates_campana.json: llevan sus
    cifras, no salen para un modelo sin empate en el corte y la del AUC-PR solo sale si romper los empates lo mueve."""
    e = _empates()
    for modelo, v in e["modelos"].items():
        nota, corta = D.frase_empate_captura(modelo, precision=True), D.frase_empate_corta(modelo)
        if v["empatados"] <= 1:
            assert nota is None and corta is None and D.empate_campana(modelo) is None
        else:
            for texto in (nota, corta):
                assert F.entero(v["empatados"]) in texto and F.entero(v["cupos"]) in texto
                assert F.pct(v["captura_esperada"]) in texto and F.num(100 * v["captura_desviacion"], 1) in texto
            assert F.pct(v["captura_minima"]) in nota and F.pct(v["captura_maxima"]) in nota
            assert F.pct(v["captura_esperada"] * e["positivos"] / e["campana"]) in nota          # la precisión esperada
            assert F.entero(v["captura_esperada"] * e["positivos"]) in D.frase_empate_corta(modelo, clientes=True)
        area = D.frase_empate_auc_pr(modelo)
        mueve = v["precision_media_azar"] - v["precision_media"] > D.DESVIACIONES_EMPATE * v["precision_media_azar_desviacion"]
        assert (area is not None) == mueve, modelo
        if area:
            assert F.num(v["precision_media"], 3) in area and F.num(v["precision_media_azar"], 3) in area
    assert set(D.modelos_con_empate()) == {m for m, v in e["modelos"].items() if v["empatados"] > 1}


def test_la_tarjeta_de_p4_y_los_empates_dice_lo_que_dice_su_archivo(monkeypatch):
    """«P4 y los empates de la campaña» lleva la tabla de empates y la prueba P4 repetida con desempates al azar, con las
    cifras de empates_campana.json, y la comparación remite a ella; si el archivo se calculó frente a un benchmark que ya no
    es el mejor del veredicto, los empates se muestran y la parte de P4 queda por repetir."""
    e = _empates()
    import app  # noqa: F401
    from secciones import comparacion, original
    veredicto = json.loads((ORIGINAL / "veredicto.json").read_text(encoding="utf-8"))["clasificacion"]
    texto = _texto(original._tarjeta_empates())
    o, p4 = e["modelos"][O], e["p4"]
    for cifra in (F.entero(o["empatados"]), F.entero(o["cupos"]), F.pct(o["captura_protocolo"]), F.num(o["precision_media_azar"], 4)):
        assert cifra in texto
    assert "Predictor óptimo" in texto and D.NOMBRE_MODELO[veredicto["mejor_benchmark"]] in texto
    if p4["mejor_benchmark"] == veredicto["mejor_benchmark"]:
        frase = original._frase_p4_empates()
        for cifra in (original._pct_fino(p4["rechaza_a_favor"]), original._p(p4["p_crudo"]["mediana"])):
            assert cifra in frase and cifra in texto
        estable = p4["rechaza_a_favor"] < original.PARTE_ESTABLE and p4["rechaza_en_contra"] < original.PARTE_ESTABLE
        assert ("no depende del desempate" in frase) == estable == ("no depende del desempate" in texto)
        assert frase in _texto(comparacion._frente("clasificacion"))
    else:
        assert original._frase_p4_empates() is None and "hay que repetir" in texto
    otro = json.loads(EMPATES.read_text(encoding="utf-8"))
    otro["p4"]["mejor_benchmark"] = next(m for m in otro["modelos"] if m not in (O, veredicto["mejor_benchmark"]))
    real = D.original_json
    monkeypatch.setattr(D, "original_json", lambda rel: otro if rel == "empates_campana.json" else real(rel))
    texto = _texto(original._tarjeta_empates())
    assert original._frase_p4_empates() is None and "hay que repetir" in texto and F.entero(o["empatados"]) in texto
    assert original._frase_p4_empates() is None and "desempates al azar" not in _texto(comparacion._frente("clasificacion"))


def test_sin_el_analisis_de_empates_no_hay_notas_y_nada_se_rompe(monkeypatch):
    """Si empates_campana.json no ha llegado, las notas no salen, la tarjeta de la vista original queda pendiente y las
    vistas que las llevan se arman igual."""
    import app  # noqa: F401
    import generar_fichas as G
    from paginas import contexto
    from secciones import campana, comparacion, diagnostico, ficha, original
    json_real, hay_real = D.original_json, D.hay_original
    monkeypatch.setattr(D, "original_json", lambda rel: None if rel == "empates_campana.json" else json_real(rel))
    monkeypatch.setattr(D, "hay_original", lambda rel: False if rel == "empates_campana.json" else hay_real(rel))
    assert D.empates_campana() is None and D.modelos_con_empate() == []
    assert D.frase_empate_captura(O) is None and D.frase_empate_corta(O) is None and D.frase_empate_auc_pr(O) is None
    assert "todavía no está" in _texto(original._tarjeta_empates()) and original._frase_p4_empates() is None
    assert "el análisis de los empates de la campaña" in _texto(original._aviso_pendientes())
    assert "desempate" not in ficha._respuesta() and "desempate al azar" not in str(G.contenido_ficha_original())
    for armar in (lambda: original.layout(tarea="clasificacion"), lambda: diagnostico._clasificacion(O),
                  lambda: comparacion._frente("clasificacion"), lambda: campana._modelos(campana.CAP_INFORME), contexto._importa):
        assert "desempate al azar" not in _texto(armar())


def test_un_cliente_en_el_corte_de_la_campana_no_se_da_por_contactado():
    """En la predicción en vivo, un puntaje igual al del corte de la campaña que comparten varios clientes de prueba no dice
    «entra»: dice cuántos de los empatados entran. Por encima del corte entra y por debajo queda fuera."""
    import app  # noqa: F401
    from secciones import prediccion
    s = D.puntajes()[O].to_numpy(dtype=float)
    corte = prediccion._corte(O)
    empatados, entran = prediccion._empate_del_corte(O)
    assert empatados == int((s == corte).sum()) and entran == int(((s == corte) & (M.punto_operacion(s) == 1)).sum())
    en_corte = prediccion._en_campana(O, corte)
    if empatados > 1:
        assert en_corte.startswith("empata en el corte") and F.entero(empatados) in en_corte and F.entero(entran) in en_corte
        # la igualdad es exacta, como en el orden de la campaña: un puntaje un ápice distinto no empata
        assert prediccion._en_campana(O, float(np.nextafter(corte, 1.0))) == "entra"
        assert prediccion._en_campana(O, float(np.nextafter(corte, 0.0))) == "queda fuera"
    else:
        assert en_corte == "entra"
    if s.max() > corte:
        assert prediccion._en_campana(O, float(s.max())) == "entra"
    if s.min() < corte:
        assert prediccion._en_campana(O, float(s.min())) == "queda fuera"
