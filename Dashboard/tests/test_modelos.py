"""Pruebas de los modelos en vivo: artefactos, ida y vuelta, motor de la EBM, explicaciones e incertidumbre.

    pytest -q

Usan los clientes de prueba guardados en datos/ y los artefactos de artifacts/: los 16 del Entregable 3 y los dos del modelo
original (BDL), que entran en todas las pruebas parametrizadas porque V.modelos los incluye. Lo que solo tiene el original
(su código, su probabilidad de excedencia, su conjunto de predicción y su veredicto) se prueba en tests/test_original.py. La
comparación del motor de la EBM con su tubería de interpret se salta si interpret-core no está instalado
(requirements-dev.txt lo instala).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import explicacion
import modelos_vivos as V
import motor as M

TAREAS = ("clasificacion", "regresion")
CASOS = [(t, m) for t in TAREAS for m in V.modelos(t)]


@pytest.fixture(scope="module")
def prueba() -> pd.DataFrame:
    return M.clientes()[V.columnas()]


@pytest.fixture(scope="module")
def guardadas() -> dict:
    return {"clasificacion": pd.read_parquet(V.DATOS / "puntajes_prueba.parquet"),
            "regresion": pd.read_parquet(V.DATOS / "pronosticos_prueba.parquet")}


def test_estan_los_ocho_del_entregable_3_y_el_original():
    """En vivo están los ocho modelos del Entregable 3 de cada tarea y, primero en la lista, el modelo original."""
    for tarea in TAREAS:
        vivos = V.modelos(tarea)
        assert vivos[0] == V.ORIGINAL, "falta el modelo original: corra preparar_original.py"
        assert len(vivos) == 9 and len(set(vivos)) == 9 and "ebm" in vivos


@pytest.mark.parametrize("tarea,modelo", CASOS)
def test_metadata_completa(tarea, modelo):
    m = V.metadata(tarea, modelo)
    for clave in ("versiones", "semilla", "hiperparametros", "entrada", "metricas_bucle_externo", "metricas_prueba",
                  "computo", "ida_y_vuelta"):
        assert clave in m, clave
    assert m["semilla"] == 42
    assert len(m["entrada"]["esquema"]) == 48
    assert m["ida_y_vuelta"]["identicas"] is True


@pytest.mark.parametrize("tarea,modelo", CASOS)
def test_ida_y_vuelta_reproduce_las_predicciones_guardadas(tarea, modelo, prueba, guardadas):
    """El artefacto, cargado de disco, da sobre los 9.745 clientes de prueba lo mismo que guardó el proyecto. En el mismo
    entorno la igualdad es exacta (preparar_artefactos.py lo exige); entre entornos XGBoost, que calcula en float32, puede
    diferir en el último dígito de float32 (~3e-8), así que su tolerancia es 1e-6."""
    s = V.predecir(tarea, modelo, prueba)
    esperado = guardadas[tarea].loc[prueba.index, modelo].to_numpy(dtype=float)
    assert np.max(np.abs(s - esperado)) < (1e-6 if modelo == "xgboost" else 1e-9)


@pytest.mark.parametrize("tarea,modelo", CASOS)
def test_ida_y_vuelta_en_este_entorno_es_exacta(tarea, modelo, prueba):
    """Dos cargas del mismo artefacto dan exactamente las mismas predicciones."""
    if modelo == "ebm":
        pytest.skip("la EBM se evalúa con motor.py")
    import joblib
    otra = joblib.load(V.ARTEFACTOS / tarea / modelo / "modelo.joblib")
    X = prueba.iloc[:500]
    a = V.predecir(tarea, modelo, X)
    b = otra.predict(X) if tarea == "regresion" else (otra.predict_proba(X)[:, 1] if hasattr(otra.named_steps["modelo"], "predict_proba")
                                                      else otra.decision_function(X))
    assert np.array_equal(a, np.asarray(b, dtype=float))


@pytest.mark.parametrize("tarea", TAREAS)
def test_motor_ebm_igual_a_su_artefacto(tarea, prueba):
    pytest.importorskip("interpret")
    import joblib
    pipe = joblib.load(V.ARTEFACTOS / tarea / "ebm" / "modelo.joblib")
    ref = pipe.predict_proba(prueba)[:, 1] if tarea == "clasificacion" else pipe.predict(prueba)
    assert np.max(np.abs(M.ebm(tarea).predecir(prueba) - ref)) < 1e-9


@pytest.mark.parametrize("tarea,modelo", CASOS)
def test_explicacion_cumple_la_eficiencia(tarea, modelo, prueba):
    """Valores de Shapley: base + suma de aportes = salida del modelo en su escala, fila por fila."""
    X = prueba.iloc[:3]
    e = V.explicar(tarea, modelo, X)
    assert list(e["aportes"].columns) == V.columnas()
    np.testing.assert_allclose(e["base"] + e["aportes"].sum(axis=1).to_numpy(), e["salida"], atol=1e-6)


@pytest.mark.parametrize("tarea,modelo", [c for c in CASOS if c[1] in ("logistica", "svm", "lasso", "ridge", "svr")])
def test_lineal_es_coeficiente_por_desvio(tarea, modelo, prueba):
    """En un modelo lineal, el aporte de una variable numérica es w (z - media del fondo z)."""
    pipe = V.tuberia(tarea, modelo)
    X = prueba.iloc[:2]
    e = explicacion.explicar(pipe, X, V.fondo(), tarea)
    cols = explicacion.columnas_por_variable(pipe.named_steps["preprocesado"])
    j = cols.index("age")
    z = np.asarray(pipe.named_steps["preprocesado"].transform(X), dtype=float)[:, j]
    zf = np.asarray(pipe.named_steps["preprocesado"].transform(V.fondo()), dtype=float)[:, j].mean()
    w = np.asarray(pipe.named_steps["modelo"].coef_, dtype=float).ravel()[j]
    np.testing.assert_allclose(e["aportes"]["age"].to_numpy(), w * (z - zf), atol=1e-12)


def test_bosque_compacto_igual_al_artefacto(prueba):
    """El Random Forest de clasificación que usa la app (bosque.py) da exactamente lo mismo que su modelo.joblib."""
    import joblib
    carpeta = V.ARTEFACTOS / "clasificacion" / "random_forest"
    ref = joblib.load(carpeta / "modelo.joblib").predict_proba(prueba)
    assert type(V.tuberia("clasificacion", "random_forest").named_steps["modelo"]).__name__ == "BosqueCompacto"
    assert np.array_equal(V.tuberia("clasificacion", "random_forest").predict_proba(prueba), ref)


def test_variable_irrelevante_aporta_cero(prueba):
    """Simetría/nulidad: si la fila tiene el valor medio del fondo en una variable, un modelo lineal le da aporte 0."""
    pipe = V.tuberia("regresion", "ridge")
    fondo = V.fondo().copy()
    X = prueba.iloc[[0]].copy()
    fondo["age"] = 45
    X["age"] = 45
    e = explicacion.explicar(pipe, X, fondo, "regresion")
    assert abs(e["aportes"]["age"].iloc[0]) < 1e-12


def test_permutaciones_exactas_en_un_modelo_aditivo(prueba):
    """En un modelo aditivo todas las contribuciones marginales coinciden: el estimador por permutaciones (el de k-NN, el
    árbol y Random Forest) debe dar el valor cerrado del modelo lineal con el mismo fondo."""
    pipe = V.tuberia("clasificacion", "logistica")
    X = prueba.iloc[:2]
    fondo = V.fondo().sample(n=10, random_state=42)
    estimado = explicacion._permutaciones(pipe, X, fondo, "clasificacion", 42, 4, 10)[0]
    exacto = explicacion.explicar(pipe, X, fondo, "clasificacion")["aportes"]
    assert float((estimado - exacto).abs().max().max()) < 1e-10


def test_knn_reproducible_con_la_misma_semilla(prueba):
    X = prueba.iloc[[0]]
    a = V.explicar("regresion", "knn", X)["aportes"]
    b = V.explicar("regresion", "knn", X)["aportes"]
    pd.testing.assert_frame_equal(a, b)


def test_vecinos_explican_la_prediccion_de_knn(prueba):
    """La predicción de k-NN es el promedio ponderado por 1/distancia de las etiquetas de sus k vecinos."""
    X = prueba.iloc[[0]]
    v = V.vecinos("regresion", "knn", X)
    w = 1 / np.maximum(v["distancias"], 1e-12)
    assert len(v["etiquetas"]) == v["k"]
    np.testing.assert_allclose(np.sum(w * v["etiquetas"]) / w.sum(), V.predecir("regresion", "knn", X)[0], atol=1e-9)


@pytest.mark.parametrize("modelo", V.modelos("regresion"))
def test_intervalo_conforme_cubre_lo_nominal(modelo):
    """El intervalo conforme simétrico (pronóstico ± q) de cada benchmark cubre lo nominal en la prueba. El del modelo original
    no tiene por qué: su error cae en niveles separados y el cuantil queda en el hueco entre dos de ellos. Lo que debe cubrir
    lo nominal es su conjunto de predicción, la unión de tramos alrededor de cada nivel (la app lo dice así en la vista)."""
    c = V.conformal()["modelos"][modelo]
    for nivel, d in c.items():
        cobertura = d["conjunto"]["cobertura_prueba"] if modelo == V.ORIGINAL else d["cobertura_prueba"]
        assert abs(cobertura - float(nivel)) < 0.02, (modelo, nivel, cobertura)


@pytest.mark.parametrize("modelo", V.modelos("clasificacion"))
def test_conjunto_conforme_cubre_al_menos_lo_nominal(modelo):
    c = V.calibracion()["modelos"][modelo]["conformal"]
    for nivel, d in c.items():
        assert d["cobertura_prueba"] >= float(nivel) - 0.02, (modelo, nivel, d["cobertura_prueba"])


@pytest.mark.parametrize("modelo", V.modelos("clasificacion"))
def test_recalibracion_monotona_y_en_0_1(modelo):
    c = V.calibracion()["modelos"][modelo]
    y = np.asarray(c["y"], dtype=float)
    assert np.all(np.diff(y) >= 0) and y.min() >= 0 and y.max() <= 1


def test_validar_rechaza_entradas_invalidas(prueba):
    fila = prueba.iloc[0].to_dict()
    assert V.validar(fila) == []
    malos = V.validar({**fila, "age": 120, "gender": "Desconocido", "household_size": None, "tx_count": "muchos"})
    assert len(malos) == 4
    assert len(V.validar({**fila, "age": "inf", "tx_count": "nan"})) == 2      # no finitos: aviso, no excepción


def test_a_tabla_conserva_los_tipos(prueba):
    X = V.a_tabla(prueba.iloc[0].to_dict())
    assert X.shape == (1, 48)
    for v in V.esquema():
        if v["tipo"] == "booleana":
            assert X[v["nombre"]].dtype == bool


def test_diferencia_critica_de_nemenyi():
    cd = json.loads((V.DATOS / "cd.json").read_text(encoding="utf-8"))
    for t in TAREAS:
        n, k = cd[t]["pliegues"], cd[t]["modelos"]
        assert cd[t]["diferencia_critica"] == pytest.approx(cd[t]["q_alfa"] * np.sqrt(k * (k + 1) / (6 * n)))
        assert sum(cd[t]["rango_medio"].values()) == pytest.approx(k * (k + 1) / 2)
