"""Pruebas unitarias del modelo original (Boosting con Desplazamiento Latente).

Cubren lo que pide la guía (§5.4): que aprende en datos sintéticos con solución conocida, la reproducibilidad con la
misma semilla, el manejo de entradas inválidas, la compatibilidad con ``sklearn.base.clone`` y con ``Pipeline``, y la
verificación numérica de las propiedades teóricas (la verosimilitud no decrece, la pérdida logística no sube, la
cobertura de los intervalos y conjuntos conformes, las cotas de la información de Fisher de la mezcla, la calibración de
la probabilidad de excedencia y que los aportes de las variables suman la predicción).

Ejecución: ``python -m pytest -q`` desde la carpeta ``ModeloOriginal``.
"""
import itertools

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import original_model as om
from src.original_model import ClasificadorBDL, ModeloOriginal, RegresorBDL

RAPIDO = dict(n_vueltas=60, arboles_por_vuelta=5, tasa=0.5, rondas_calentamiento=40, paciencia=10)
"""Hiperparámetros pequeños para que las pruebas tarden segundos."""
CENTROS = np.array([-0.30, 0.0, 0.30])
PESOS = np.array([0.2, 0.5, 0.3])


def datos_con_escalones(n=4000, semilla=0, sigma=0.01):
    """Objetivo con solución conocida: índice aditivo en dos variables + escalón latente de tres niveles + ruido pequeño."""
    rng = np.random.default_rng(semilla)
    X = np.c_[rng.integers(0, 6, n), rng.uniform(-1, 1, n), rng.normal(size=(n, 3))].astype(float)
    indice = 0.08 * X[:, 0] + 0.25 * np.sin(3 * X[:, 1])
    clase = rng.choice(3, size=n, p=PESOS)
    return X, indice + CENTROS[clase] + rng.normal(0, sigma, n), indice, clase


@pytest.fixture(scope='module')
def escalones():
    return datos_con_escalones()


@pytest.fixture(scope='module')
def regresor(escalones):
    X, s, *_ = escalones
    return RegresorBDL(**RAPIDO, random_state=0).fit(X, s)


@pytest.fixture
def datos():
    """Clasificación sintética corriente (sin puntaje), como en el esqueleto de la guía."""
    return make_classification(n_samples=600, n_features=6, n_informative=4, random_state=0)


# ------------------------------------------------------------------ las cuatro pruebas del esqueleto de la guía
def test_aprende_mejor_que_azar(datos):
    X, y = datos
    modelo = ModeloOriginal(**RAPIDO, profundidad=2, random_state=0).fit(X, y)
    assert modelo.modo_ == 'directo'
    assert (modelo.predict(X) == y).mean() > 0.7


def test_reproducible_con_misma_semilla(datos, escalones):
    X, y = datos
    p1 = ModeloOriginal(**RAPIDO, random_state=1).fit(X, y).predict_proba(X)
    p2 = ModeloOriginal(**RAPIDO, random_state=1).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)
    Xe, s, *_ = escalones
    r1 = RegresorBDL(**RAPIDO, random_state=3).fit(Xe, s).predict(Xe)
    r2 = RegresorBDL(**RAPIDO, random_state=3).fit(Xe, s).predict(Xe)
    np.testing.assert_array_equal(r1, r2)


def test_clone_conserva_hiperparametros():
    for modelo in (ModeloOriginal(n_vueltas=10, tasa=0.3, usar_puntaje=False), RegresorBDL(n_vueltas=10, tasa=0.3, n_componentes=2)):
        copia = clone(modelo)
        assert copia.get_params() == modelo.get_params()
        assert not hasattr(copia, 'cortes_')


def test_entrada_invalida(datos, escalones):
    X, y = datos
    X_malo = X.copy()
    X_malo[0, 0] = np.nan
    with pytest.raises(ValueError):
        ModeloOriginal().fit(X_malo, y)
    Xe, s, *_ = escalones
    with pytest.raises(ValueError):
        RegresorBDL().fit(X_malo[:, :5], s[:len(X_malo)])
    with pytest.raises(ValueError):
        RegresorBDL(tasa=0.0).fit(Xe, s)
    with pytest.raises(ValueError):
        RegresorBDL(profundidad=0).fit(Xe, s)
    with pytest.raises(ValueError):
        RegresorBDL(n_componentes='muchos').fit(Xe, s)
    with pytest.raises(ValueError):
        ModeloOriginal().fit(X, np.zeros(len(y)))                 # una sola clase
    with pytest.raises(ValueError):
        RegresorBDL(**RAPIDO).fit(Xe, s).predict(Xe[:, :3])       # menos columnas que en el ajuste


# ------------------------------------------------------------------ aprende la solución conocida
def test_recupera_los_escalones(escalones, regresor):
    X, s, indice, _ = escalones
    assert regresor.n_componentes_ == 3
    np.testing.assert_allclose(np.diff(regresor.centros_), np.diff(CENTROS), atol=0.01)
    np.testing.assert_allclose(regresor.pi_, PESOS, atol=0.04)
    error = regresor.indice(X) - indice
    assert np.std(error) < 0.02                                    # el índice se recupera salvo una constante


def test_la_mezcla_mejora_al_boosting_de_minimos_cuadrados(escalones, regresor):
    X, s, indice, _ = escalones
    Xn, sn, indice_n, _ = datos_con_escalones(n=3000, semilla=1)
    sin_mezcla = RegresorBDL(**RAPIDO, n_componentes=1, random_state=0).fit(X, s)
    assert sin_mezcla.n_componentes_ == 1
    error_con = np.std(regresor.indice(Xn) - indice_n)
    error_sin = np.std(sin_mezcla.indice(Xn) - indice_n)
    assert error_con < 0.5 * error_sin                             # quitar el escalón afina el índice
    rmse_con = np.sqrt(np.mean((sn - regresor.predict(Xn)) ** 2))
    rmse_sin = np.sqrt(np.mean((sn - sin_mezcla.predict(Xn)) ** 2))
    assert rmse_con < rmse_sin


def test_ruido_unimodal_deja_un_solo_componente():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3000, 4))
    y = X[:, 0] - 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.3, 3000)
    modelo = RegresorBDL(**RAPIDO, random_state=0).fit(X, y)
    assert modelo.n_componentes_ == 1
    assert np.corrcoef(modelo.predict(X), y)[0, 1] > 0.9


# ------------------------------------------------------------------ propiedades teóricas, verificadas numéricamente
def test_verosimilitud_no_decrece(regresor):
    """Proposición 1 del informe: cada vuelta del EM generalizado aumenta o conserva la verosimilitud observada."""
    v = np.array(regresor.historial_['verosimilitud'])
    assert len(v) > 5
    assert np.all(np.diff(v) >= -1e-7 * np.abs(v[:-1]))
    assert v[-1] > v[0]


def test_verosimilitud_no_decrece_con_arboles_profundos(escalones):
    X, s, *_ = escalones
    for profundidad, tasa in ((2, 1.0), (3, 0.3)):
        v = np.array(RegresorBDL(**{**RAPIDO, 'tasa': tasa}, profundidad=profundidad, random_state=0).fit(X, s).historial_['verosimilitud'])
        assert np.all(np.diff(v) >= -1e-7 * np.abs(v[:-1]))


def test_perdida_logistica_no_sube(datos):
    """Proposición 2 del informe: el paso por cota cuadrática no puede aumentar la pérdida logística de ajuste."""
    X, y = datos
    for tasa in (0.1, 0.5, 1.0):
        modelo = ModeloOriginal(**{**RAPIDO, 'tasa': tasa, 'paciencia': 1000}, profundidad=2, random_state=0).fit(X, y)
        perdida = np.array(modelo.historial_['perdida'])
        assert len(perdida) > 50
        assert np.all(np.diff(perdida) <= 1e-12)


def test_cobertura_conforme(regresor):
    """Proposición 3 del informe: intervalos y conjuntos cubren al menos lo pedido en datos nuevos (salvo error de muestreo)."""
    Xn, sn, *_ = datos_con_escalones(n=20000, semilla=7)
    for cobertura in (0.8, 0.9):
        bajo, alto = regresor.intervalo(Xn, cobertura)
        en_intervalo = np.mean((sn >= bajo) & (sn <= alto))
        en_conjunto = regresor.conjunto_cubre(Xn, sn, cobertura).mean()
        tolerancia = 4 * np.sqrt(cobertura * (1 - cobertura) / regresor.n_calibracion_)
        assert en_intervalo > cobertura - tolerancia
        assert en_conjunto > cobertura - tolerancia
    assert regresor.largo_conjunto(0.9) < 0.5 * np.mean(alto - bajo)   # con escalones, el conjunto es mucho más corto


def test_conjunto_es_union_de_intervalos_alrededor_de_los_escalones(regresor):
    tramos = regresor.conjunto_residual(0.9)
    assert tramos.shape[1] == 2 and np.all(tramos[:, 1] >= tramos[:, 0])
    assert 2 <= len(tramos) <= 3                                   # tres escalones; el menos pesado puede quedar fuera
    assert np.all(np.diff(tramos.ravel()) >= 0)                    # intervalos ordenados y sin solaparse


def test_informacion_de_fisher_entre_las_cotas():
    """Proposición 4 del informe: 1/(σ² + Var(c)) ≤ I_f ≤ 1/σ², con igualdad si K = 1 e I_f → 1/σ² al separar las clases."""
    sigma = 1.0

    def informacion(pi, c):
        r = np.linspace(c.min() - 12 * sigma, c.max() + 12 * sigma, 400_001)
        f = om.densidad_mezcla(r, pi, c, sigma)
        g = np.gradient(f, r) ** 2 / np.maximum(f, 1e-300)
        return float(((g[1:] + g[:-1]) / 2 * np.diff(r)).sum())    # regla del trapecio

    assert informacion(np.array([1.0]), np.array([0.0])) == pytest.approx(1 / sigma ** 2, rel=1e-5)   # K = 1: las dos cotas coinciden
    valores, ganancias = [], []
    for separacion in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        c = separacion * np.array([-1.0, 0.0, 1.0])
        varianza_c = float(PESOS @ (c - PESOS @ c) ** 2)
        valor = informacion(PESOS, c)
        assert 1 / (sigma ** 2 + varianza_c) < valor < 1 / sigma ** 2    # desigualdades estrictas con K > 1
        valores.append(valor)
        ganancias.append(valor * (sigma ** 2 + varianza_c))             # cuánto mejora la cota respecto de los mínimos cuadrados
    assert np.all(np.diff(valores[2:]) > 0)                         # al separar las clases la información crece...
    assert valores[-1] == pytest.approx(1 / sigma ** 2, rel=1e-4)   # ...hasta 1/σ²: no se pierde nada por no observar la clase
    assert ganancias[0] < 1.05 and ganancias[-1] > 100              # clases pegadas: casi nada que ganar; separadas: mucho


def test_aportes_suman_la_prediccion(escalones):
    X, s, *_ = escalones
    for profundidad in (1, 2, 3):
        modelo = RegresorBDL(**{**RAPIDO, 'n_vueltas': 15}, profundidad=profundidad, random_state=0).fit(X, s)
        aportes, base = modelo.contribuciones(X[:200])
        np.testing.assert_allclose(base + aportes.sum(axis=1), modelo.predict(X[:200]), atol=1e-9)
        assert np.abs(aportes[:, 2:]).max() < 0.5 * np.abs(aportes[:, :2]).max()   # las variables de ruido aportan poco


def test_formas_e_importancia_con_profundidad_uno(regresor):
    assert regresor.formas_ is not None and len(regresor.formas_) == 5
    importancia = regresor.importancia()
    assert set(np.argsort(importancia)[-2:]) == {0, 1}             # las dos variables que generan el índice


# ------------------------------------------------------------------ clasificación por excedencia
def test_excedencia_supera_al_modo_directo(escalones):
    X, s, *_ = escalones
    umbral = np.quantile(s, 0.75)
    y = (s > umbral).astype(int)
    Xn, sn, *_ = datos_con_escalones(n=4000, semilla=2)
    yn = (sn > umbral).astype(int)
    con_puntaje = ClasificadorBDL(**RAPIDO, random_state=0).fit(X, y, puntaje=s)
    sin_puntaje = ClasificadorBDL(**RAPIDO, random_state=0).fit(X, y)
    ignorado = ClasificadorBDL(**RAPIDO, usar_puntaje=False, random_state=0).fit(X, y, puntaje=s)
    assert con_puntaje.modo_ == 'excedencia' and sin_puntaje.modo_ == 'directo' and ignorado.modo_ == 'directo'
    assert s[y == 0].max() < con_puntaje.umbral_ < s[y == 1].min()
    p = con_puntaje.predict_proba(Xn)
    assert p.shape == (len(Xn), 2) and np.allclose(p.sum(axis=1), 1.0) and p.min() > 0 and p.max() < 1
    assert roc_auc_score(yn, p[:, 1]) > roc_auc_score(yn, sin_puntaje.predict_proba(Xn)[:, 1])
    np.testing.assert_allclose(sin_puntaje.predict_proba(Xn), ignorado.predict_proba(Xn))
    assert (con_puntaje.predict(Xn) == yn).mean() > 0.75


def test_excedencia_coincide_con_la_frecuencia(escalones, regresor):
    """La probabilidad de excedencia está calibrada: en cada meseta coincide con la frecuencia observada en datos nuevos."""
    Xn, sn, *_ = datos_con_escalones(n=20000, semilla=3)
    umbral = 0.15
    for metodo in ('mezcla', 'conforme'):
        p = regresor.prob_excedencia(Xn, umbral, metodo=metodo)
        assert 0 < p.min() and p.max() < 1
        assert abs(p.mean() - np.mean(sn > umbral)) < 0.02
    p = regresor.prob_excedencia(Xn, umbral)
    for bajo, alto in ((0.1, 0.4), (0.4, 0.75)):
        grupo = (p > bajo) & (p <= alto)
        if grupo.sum() > 500:
            assert abs(p[grupo].mean() - np.mean(sn[grupo] > umbral)) < 0.05
    with pytest.raises(ValueError):
        regresor.prob_excedencia(Xn, umbral, metodo='otro')


def test_etiqueta_que_no_es_un_umbral(escalones):
    X, s, *_ = escalones
    y = (s > np.median(s)).astype(int)
    y[np.argmax(s)] = 0                                            # el puntaje más alto, marcado como negativo
    with pytest.raises(ValueError):
        ClasificadorBDL(**RAPIDO).fit(X, y, puntaje=s)
    with pytest.raises(ValueError):
        ClasificadorBDL(**RAPIDO).fit(X, (s > np.median(s)).astype(int), puntaje=s[:-1])


def test_entra_en_pipeline_y_validacion_cruzada(escalones):
    X, s, *_ = escalones
    X, s = X[:1500], s[:1500]
    y = (s > np.quantile(s, 0.75)).astype(int)
    tuberia = Pipeline([('pre', StandardScaler()), ('modelo', ClasificadorBDL(**{**RAPIDO, 'n_vueltas': 20}, random_state=0))])
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    con = cross_val_score(tuberia, X, y, cv=cv, scoring='roc_auc', params={'modelo__puntaje': s})
    sin = cross_val_score(tuberia, X, y, cv=cv, scoring='roc_auc')
    assert con.mean() > 0.7 and sin.mean() > 0.6
    regresion = Pipeline([('pre', StandardScaler()), ('modelo', RegresorBDL(**{**RAPIDO, 'n_vueltas': 20}, random_state=0))])
    assert cross_val_score(regresion, X, s, cv=3, scoring='neg_root_mean_squared_error').mean() < 0


# ------------------------------------------------------------------ piezas internas
def test_arbol_propio():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    r = np.where(X[:, 0] > 0.2, 1.0, -1.0) + np.where(X[:, 1] > -0.5, 0.5, 0.0) + rng.normal(0, 0.01, 500)
    cortes = om.calcular_cortes(X, 64)
    C = om.codificar(X, cortes)
    hist = om._Histogramas(C, np.array([len(c) + 1 for c in cortes]))
    for profundidad in (1, 2, 3):
        variable, corte, valor, cobertura, nodo = om.ajustar_arbol(hist, C, r, profundidad, 5, 0.0)
        prediccion = om.predecir_arbol(variable, corte, valor, C)
        np.testing.assert_allclose(prediccion, valor[nodo])        # predecir recorre el árbol igual que el ajuste
        assert cobertura[0] == 500 and variable[0] == 0             # el primer corte es por la variable más informativa
        assert np.sum((r - prediccion) ** 2) < np.sum((r - r.mean()) ** 2)
        hojas = np.unique(nodo)
        np.testing.assert_allclose(valor[hojas], [r[nodo == h].mean() for h in hojas])   # con λ = 0, la media de la hoja
    error = [np.sum((r - om.predecir_arbol(*om.ajustar_arbol(hist, C, r, d, 5, 0.0)[:3], C)) ** 2) for d in (1, 2, 3)]
    assert error[0] > error[1] > error[2]


def test_histograma_por_filas_coincide_con_el_total():
    rng = np.random.default_rng(1)
    C = rng.integers(0, 7, size=(300, 4)).astype(np.int32)
    hist = om._Histogramas(C, np.full(4, 7))
    v = rng.normal(size=300)
    filas = np.flatnonzero(rng.random(300) < 0.4)
    suma, conteo = hist.sumar_filas(filas, v[filas])
    mascara = np.zeros(300)
    mascara[filas] = 1.0
    np.testing.assert_allclose(suma, hist.sumar(v * mascara))
    np.testing.assert_allclose(conteo, hist.sumar(mascara))


def test_kmedias_1d_es_exacto():
    rng = np.random.default_rng(2)
    posicion = np.sort(rng.normal(size=12))
    peso = rng.integers(1, 5, 12).astype(float)
    for K in (2, 3):
        centros = om._kmedias_1d(posicion, peso, K)
        asignado = np.argmin(np.abs(posicion[:, None] - centros[None, :]), axis=1)
        costo = float(peso @ (posicion - centros[asignado]) ** 2)
        mejor = np.inf
        for cortes in itertools.combinations(range(1, 12), K - 1):   # fuerza bruta: todos los cortes contiguos
            limites = (0, *cortes, 12)
            total = 0.0
            for a, b in zip(limites[:-1], limites[1:]):
                media = peso[a:b] @ posicion[a:b] / peso[a:b].sum()
                total += peso[a:b] @ (posicion[a:b] - media) ** 2
            mejor = min(mejor, total)
        assert costo == pytest.approx(mejor, rel=1e-9)


def test_pasos_e_y_m():
    rng = np.random.default_rng(3)
    clase = rng.choice(2, 5000, p=[0.3, 0.7])
    r = np.array([-1.0, 1.0])[clase] + rng.normal(0, 0.1, 5000)
    gamma, log_fila = om.paso_e(r, np.array([0.5, 0.5]), np.array([-0.8, 0.8]), 0.2)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0)
    pi, c, sigma = om.paso_m(r, gamma, np.array([-0.8, 0.8]), 1e-9)
    np.testing.assert_allclose(pi, [0.3, 0.7], atol=0.02)
    np.testing.assert_allclose(c, [-1.0, 1.0], atol=0.02)
    assert sigma == pytest.approx(0.1, abs=0.01)
    assert om.paso_e(r, pi, c, sigma)[1].sum() > log_fila.sum()   # una vuelta de EM mejora la verosimilitud
    assert om._clases_utiles(np.array([0.5, 0.495, 0.005]), np.array([0.0, 1.0, 5.0]), 0.1) == 2      # la clase con 0,5 % no cuenta
    assert om._clases_utiles(np.array([0.5, 0.5]), np.array([0.0, 0.1]), 0.1) == 1                    # separadas por 1 σ: una sola
