"""Pruebas del marco de experimentos de la corrección: recencia sin fuga, puntaje por pliegue y catálogo.

No necesitan los datos de COFINFAD: usan tablas sintéticas pequeñas, para que corran en integración continua.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import datos, modelos, optimizacion, plan
from src.config import PRESUPUESTO_ORIGINAL


# ------------------------------------------------------------------ recencia: la fecha de corte sale de entrenamiento
@pytest.fixture
def transacciones():
    return pd.DataFrame({
        'customer_id': ['a', 'a', 'b', 'b', 'c', 'c'],
        'date': pd.to_datetime(['2023-01-10', '2023-03-01', '2023-02-01', '2023-06-15', '2023-05-01', '2023-12-20']),
        'amount': [10.0, 20.0, 5.0, 15.0, 7.0, 9.0],
        'type': ['Deposit', 'Payment', 'Deposit', 'Deposit', 'Payment', 'Transfer']})


def test_fecha_de_corte_usa_solo_entrenamiento(transacciones):
    assert datos.fecha_de_corte(transacciones, ['a', 'b']) == pd.Timestamp('2023-06-15')
    assert datos.fecha_de_corte(transacciones, ['a', 'b', 'c']) == pd.Timestamp('2023-12-20')
    with pytest.raises(ValueError):
        datos.fecha_de_corte(transacciones, ['z'])


def test_recencia_no_depende_de_los_clientes_de_prueba(transacciones):
    """Cambiar las transacciones de un cliente de prueba no mueve la recencia de los de entrenamiento."""
    corte = datos.fecha_de_corte(transacciones, ['a', 'b'])
    base = datos.derivar_variables_transaccionales(transacciones, corte)
    assert base.loc['a', 'recencia_dias'] == (pd.Timestamp('2023-06-15') - pd.Timestamp('2023-03-01')).days
    assert base.loc['b', 'recencia_dias'] == 0
    assert base.loc['c', 'recencia_dias'] == 0                     # operó después del corte: se deja en 0, no negativa
    otras = transacciones.copy()
    otras.loc[otras['customer_id'] == 'c', 'date'] += pd.Timedelta(days=200)
    cambiada = datos.derivar_variables_transaccionales(otras, datos.fecha_de_corte(otras, ['a', 'b']))
    pd.testing.assert_series_equal(base.loc[['a', 'b'], 'recencia_dias'], cambiada.loc[['a', 'b'], 'recencia_dias'])


def test_recencia_estandarizada_no_depende_del_corte(transacciones):
    """La razón por la que la corrección no altera resultados: al estandarizar, la fecha de corte se cancela."""
    ultima = transacciones.groupby('customer_id')['date'].max()
    z = []
    for corte in (pd.Timestamp('2023-12-20'), pd.Timestamp('2024-06-30')):
        recencia = (corte - ultima).dt.days.to_numpy(dtype=float).reshape(-1, 1)
        z.append(StandardScaler().fit_transform(recencia))
    np.testing.assert_allclose(z[0], z[1], atol=1e-12)


# ------------------------------------------------------------------ el puntaje llega recortado a cada pliegue
class _Espia(ClassifierMixin, BaseEstimator):
    """Clasificador de prueba: comprueba que el puntaje que recibe corresponde, fila a fila, a la etiqueta."""

    registro: list = []

    def __init__(self, umbral=0.0, c=1.0):
        self.umbral = umbral
        self.c = c

    def fit(self, X, y, puntaje=None):
        assert puntaje is not None and len(puntaje) == len(y)
        assert np.array_equal((np.asarray(puntaje) > self.umbral).astype(int), np.asarray(y))   # alineado con y
        _Espia.registro.append(len(y))
        self.classes_ = np.array([0, 1])
        self.media_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.clip(self.media_ + 0.01 * np.asarray(X)[:, 0], 0.01, 0.99)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def test_validacion_anidada_recorta_el_puntaje_por_pliegue():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list('abc'), index=rng.permutation(n) + 1000)
    puntaje = pd.Series(X['a'].to_numpy() + rng.normal(0, 0.5, n), index=X.index)
    y = (puntaje > 0).astype(int)
    _Espia.registro = []

    def constructor(_y):
        return Pipeline([('preprocesado', StandardScaler()), ('modelo', _Espia())])

    res = optimizacion.validacion_anidada(
        constructor, {'modelo__c': [1.0, 2.0]}, {'modelo__c': ('log', 0.1, 10)}, 'grid', X, y, 'clasificacion',
        k_ext=5, k_int=3, n_jobs=1, params_ajuste={'modelo__puntaje': puntaje.to_numpy()}, guardar_predicciones=True)
    assert len(res.puntajes_ext) == 5 and res.estimador_final is not None
    assert not np.isnan(res.predicciones_ext).any() and len(res.predicciones_ext) == n
    internos = [m for m in _Espia.registro if m < 0.8 * n * 0.9]      # ajustes de los pliegues internos
    assert internos and max(_Espia.registro) == n                     # el ajuste final usa todas las filas
    assert all(m < n for m in _Espia.registro[:-1])                   # ningún ajuste anterior vio el pliegue de evaluación
    with pytest.raises(ValueError):
        optimizacion.validacion_anidada(constructor, {'modelo__c': [1.0]}, {}, 'grid', X, y, 'clasificacion',
                                        n_jobs=1, params_ajuste={'modelo__puntaje': puntaje.to_numpy()[:-1]})


# ------------------------------------------------------------------ catálogo y plan
def test_catalogo_incluye_el_original_y_lightgbm():
    for tarea in modelos.TAREAS:
        cat = modelos.catalogo(tarea, incluir_nuevo=True, incluir_original=True)
        assert len(cat) == 10 and list(cat)[-2:] == ['lightgbm', 'bdl']
        assert plan.modelos(tarea) == list(cat)
        assert optimizacion.presupuesto_por_defecto(cat['bdl'].espacio_grid) == PRESUPUESTO_ORIGINAL == 30
        presupuestos = [optimizacion.presupuesto_por_defecto(s.espacio_grid) for n, s in cat.items() if n != 'bdl']
        assert min(presupuestos) >= PRESUPUESTO_ORIGINAL            # ningún benchmark recibe menos que el original
        estimador = cat['bdl'].construir(7)
        assert estimador.random_state == 7 and type(estimador).__module__.endswith('original_model')
    assert len(modelos.catalogo('clasificacion')) == 7               # el producto de 140 corridas no cambia


def test_el_modelo_original_no_importa_ningun_modelo():
    """R2: el módulo del modelo solo usa numpy, scipy y las clases base de scikit-learn."""
    import ast
    from pathlib import Path

    fuente = Path(modelos.__file__).with_name('original_model.py').read_text(encoding='utf-8')
    importados = set()
    for nodo in ast.walk(ast.parse(fuente)):
        if isinstance(nodo, ast.Import):
            importados |= {a.name for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom):
            importados.add(nodo.module)
    assert importados == {'__future__', 'numpy', 'scipy', 'scipy.special', 'sklearn.base', 'sklearn.utils.validation'}


# ------------------------------------------------------------------ tabla maestra con dos procesos
def test_registro_no_pierde_filas_de_otro_proceso(tmp_path):
    """Dos colas escriben en la misma tabla: cada escritura parte de lo que hay en disco, no de una copia vieja."""
    from src.registro import Registro

    ruta = tmp_path / 'experimentos.parquet'
    a, b = Registro(ruta), Registro(ruta)                # dos «procesos» que cargaron la tabla vacía
    fila = {'balanceo': 'ninguno', 'optimizador': 'bayesiana', 'semilla': 42, 'media_ext': 0.5, 'puntajes_ext': [0.4, 0.6]}
    a.anadir({'tarea': 'regresion', 'modelo': 'bdl', **fila})
    b.anadir({'tarea': 'regresion', 'modelo': 'xgboost', **fila})
    a.anadir({'tarea': 'clasificacion', 'modelo': 'bdl', **fila})
    assert len(Registro(ruta).cargar()) == 3
    assert set(a.cargar()['modelo']) == {'bdl', 'xgboost'} and len(b.cargar()) == 3
    assert not ruta.with_name(ruta.name + '.candado').exists()
    with pytest.raises(ValueError):
        b.anadir({'tarea': 'regresion', 'modelo': 'bdl', **fila})
    b.anadir({'tarea': 'regresion', 'modelo': 'bdl', **{**fila, 'media_ext': 0.7}}, sobrescribir=True)
    t = a.cargar()
    assert len(t) == 3 and float(t[(t.tarea == 'regresion') & (t.modelo == 'bdl')]['media_ext'].iloc[0]) == 0.7


# ------------------------------------------------------------------ rutas de la tabla maestra
def test_rutas_de_la_tabla_relativas_al_repositorio(tmp_path):
    """Las rutas se publican relativas a la raíz del repositorio y se resuelven igual desde las dos formas."""
    from pathlib import Path

    from src.registro import RAIZ_REPO, Registro, relativa, resolver

    corta = 'ModeloOriginal/resultados/modelos/m.joblib'
    dentro = RAIZ_REPO / 'ModeloOriginal' / 'resultados' / 'modelos' / 'm.joblib'
    assert relativa(str(dentro)) == corta and relativa(corta) == corta                    # y no cambia lo que ya es relativo
    assert relativa('') == '' and relativa(None) == '' and relativa(float('nan')) == ''
    fuera = Path(RAIZ_REPO.anchor) / 'otra' / 'carpeta' / 'resultados' / 'modelos' / 'm.joblib'
    assert relativa(str(fuera)) == 'Entrega3/resultados/modelos/m.joblib'                # los modelos del Entregable 3
    movido = Path(RAIZ_REPO.anchor) / 'otro' / 'clon' / 'ModeloOriginal' / 'resultados' / 'modelos' / 'm.joblib'
    assert relativa(str(movido)) == corta                                                # el repositorio en otra carpeta
    assert resolver(str(dentro)) == dentro and resolver(corta) == dentro
    ruta = tmp_path / 'experimentos.parquet'
    reg = Registro(ruta)
    reg.anadir({'tarea': 'regresion', 'modelo': 'bdl', 'balanceo': 'ninguno', 'optimizador': 'bayesiana', 'semilla': 42,
                'media_ext': 0.5, 'puntajes_ext': [0.4, 0.6], 'ruta_modelo': str(dentro), 'ruta_predicciones': ''})
    assert reg.rutas_relativas() == 1 and reg.rutas_relativas() == 0
    assert Registro(ruta).cargar()['ruta_modelo'].iloc[0] == corta
