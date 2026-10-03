"""Pruebas del modelo nuevo (EBM) y de la comparación con la Entrega 2, con datos sintéticos.

Se ejecutan con ``python -m pytest tests -q`` desde la raíz del proyecto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import balanceo as bal
from src import comparacion as cmp
from src import experimento, modelos, optimizacion
from src.config import SEED


@pytest.fixture(scope='module')
def sinteticos():
    rng = np.random.default_rng(SEED)
    n = 800
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n), 'c': rng.uniform(size=n),
                      'segmento': rng.choice(['x', 'y', 'z'], size=n)})
    z = np.sin(2 * X['a']) + 0.5 * X['b'] ** 2 + rng.normal(scale=0.7, size=n)
    y_cont = pd.Series(1 / (1 + np.exp(-z)), name='churn_probability')
    y_bin = (y_cont > y_cont.quantile(0.75)).astype(int)
    return X, y_bin, y_cont


def test_el_catalogo_de_la_guia_sigue_teniendo_siete_modelos():
    for tarea in ('clasificacion', 'regresion'):
        assert len(modelos.catalogo(tarea)) == 7 and 'ebm' not in modelos.catalogo(tarea)
        assert list(modelos.catalogo(tarea, incluir_nuevo=True))[-1] == 'ebm'
    assert len(experimento.combinaciones('clasificacion')) == 112
    assert len(experimento.combinaciones('clasificacion', ['ebm'])) == 16


def test_la_ebm_cumple_la_especificacion():
    for tarea in ('clasificacion', 'regresion'):
        spec = modelos.especificacion_ebm(tarea)
        assert spec.n_configuraciones_grid() >= 30
        assert set(spec.espacio_grid) <= set(spec.espacio)
        assert spec.fidelidad == {'busqueda': {'modelo__outer_bags': 2}, 'final': {'modelo__outer_bags': 8}}
        assert spec.justificacion and spec.complejidad
    assert bal.aplica(modelos.especificacion_ebm('clasificacion'), 'class_weight') == (True, '')


def test_class_weight_de_la_ebm_desplaza_las_probabilidades(sinteticos):
    X, y, _ = sinteticos
    Xn = X[['a', 'b', 'c']].to_numpy()
    sin = modelos.EBMConPesos(outer_bags=2, interactions=0, n_jobs=1, random_state=SEED).fit(Xn, y)
    con = clone(sin).set_params(class_weight='balanced').fit(Xn, y)
    assert con.predict_proba(Xn)[:, 1].mean() > sin.predict_proba(Xn)[:, 1].mean() + 0.1
    assert list(con.classes_) == [0, 1]


def test_validacion_anidada_de_la_ebm_de_extremo_a_extremo(sinteticos):
    X, y, y_cont = sinteticos
    for tarea, objetivo, balanceo in (('clasificacion', y, 'class_weight'), ('regresion', y_cont, 'ninguno')):
        spec = modelos.especificacion_ebm(tarea)
        res = optimizacion.validacion_anidada(
            lambda yp: bal.construir_pipeline(spec, balanceo, X, SEED, yp), spec.espacio_grid, spec.espacio,
            'bayesiana', X, objetivo, tarea, k_ext=2, k_int=2, presupuesto=2,
            fijos_busqueda=spec.fidelidad['busqueda'], fijos_final=spec.fidelidad['final'])
        assert np.isfinite(res.media_ext)
        assert res.params_finales['modelo__outer_bags'] == 8
        assert res.estimador_final.named_steps['modelo'].get_params()['outer_bags'] == 8


def test_modelos_de_la_entrega_2_con_sus_hiperparametros(sinteticos):
    X, y, y_cont = sinteticos
    clf = cmp.pipeline_entrega2('clasificacion', X).fit(X, y)
    reg = cmp.pipeline_entrega2('regresion', X).fit(X, y_cont)
    assert clf.named_steps['modelo'].C == 0.1 and clf.named_steps['modelo'].l1_ratio == 1.0
    assert reg.named_steps['modelo'].alpha == 1e-4
    assert cmp.reproduce_entrega2('regresion', {'r2': 0.15114, 'rmse': 0.06171}) == {
        'r2': {'publicado': 0.1511, 'obtenido': 0.1511, 'coincide': True},
        'rmse': {'publicado': 0.0617, 'obtenido': 0.0617, 'coincide': True}}


def test_tablas_de_contraste_orientadas_a_favor_del_candidato(sinteticos):
    _, y, y_cont = sinteticos
    rng = np.random.default_rng(1)
    bueno = y + rng.normal(scale=0.6, size=len(y))
    malo = y + rng.normal(scale=1.5, size=len(y))
    t = cmp.contrastar_clasificacion(y, {'ref': malo, 'cand': bueno}, [('ref', 'cand')],
                                     {'ref': [0.6, 0.62, 0.61], 'cand': [0.7, 0.71, 0.69]}, n_boot=200)
    f = t.iloc[0]
    assert f['diferencia'] > 0 and f['p_delong_unilateral'] < 0.01 and f['delta_cliff'] == 1.0
    r = cmp.contrastar_regresion(y_cont, {'ref': y_cont + rng.normal(scale=0.2, size=len(y)),
                                          'cand': y_cont + rng.normal(scale=0.1, size=len(y))},
                                 [('ref', 'cand')], {'ref': [0.2, 0.21], 'cand': [0.1, 0.11]}, n_boot=200)
    g = r.iloc[0]
    assert g['mejora_rmse'] > 0 and g['mejora_r2'] > 0 and g['p_dm_unilateral'] < 0.01
    assert g['d_cohen_pliegues'] > 0 and g['delta_cliff_pliegues'] == 1.0
