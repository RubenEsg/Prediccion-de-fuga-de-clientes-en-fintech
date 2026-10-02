"""Pruebas de ``src.estadistica`` con datos sintéticos.

Cada contraste se compara con su definición directa (el AUC por pares, el jackknife
reajustando, la t pareada de scipy) en lugar de con cifras copiadas de otra fuente.
Se ejecutan con ``python -m pytest tests -q`` desde la raíz del proyecto.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score

from src import estadistica as est

RNG = np.random.default_rng(7)


def _clasificacion(n: int = 400, prevalencia: float = 0.3, separacion: float = 1.0, empates: bool = True):
    """Etiqueta y dos puntajes correlacionados; con ``empates`` se redondean para forzarlos."""
    y = (RNG.random(n) < prevalencia).astype(int)
    comun = RNG.normal(size=n)
    a = separacion * y + comun + 0.8 * RNG.normal(size=n)
    b = 1.3 * separacion * y + comun + 0.8 * RNG.normal(size=n)
    if empates:
        a, b = np.round(a, 1), np.round(b, 1)
    return y, a, b


def _delong_por_definicion(y, puntajes):
    """Covarianza de DeLong recorriendo los m × n pares (DeLong y otros, 1988)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    v10, v01 = [], []
    for s in puntajes:
        psi = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
        v10.append(psi.mean(axis=1))
        v01.append(psi.mean(axis=0))
    return np.cov(np.vstack(v10)) / m + np.cov(np.vstack(v01)) / n


# ---------------------------------------------------------------------------- DeLong
def test_auc_y_colocaciones_coinciden_con_sklearn_con_empates():
    y, a, _ = _clasificacion()
    auc, v_pos, v_neg = est.colocaciones(y, a)
    assert auc == pytest.approx(roc_auc_score(y, a), abs=1e-12)
    assert v_pos.mean() == pytest.approx(v_neg.mean(), abs=1e-12)


def test_covarianza_de_delong_igual_a_la_definicion_por_pares():
    y, a, b = _clasificacion(n=300)
    S = _delong_por_definicion(y, [a, b])
    r = est.delong(y, a, b)
    assert r['ee'] ** 2 == pytest.approx(S[0, 0] + S[1, 1] - 2 * S[0, 1], rel=1e-10)
    assert est.delong(y, a)['ee'] ** 2 == pytest.approx(S[0, 0], rel=1e-10)
    assert r['diferencia'] == pytest.approx(roc_auc_score(y, b) - roc_auc_score(y, a), abs=1e-12)


def test_delong_es_antisimetrico_y_su_p_no_depende_del_orden():
    y, a, b = _clasificacion()
    ab, ba = est.delong(y, a, b), est.delong(y, b, a)
    assert ab['diferencia'] == pytest.approx(-ba['diferencia'])
    assert ab['p_bilateral'] == pytest.approx(ba['p_bilateral'])
    assert ab['p_unilateral'] == pytest.approx(1 - ba['p_unilateral'])


def test_delong_detecta_una_mejora_clara_y_no_una_inexistente():
    y, a, b = _clasificacion(n=3000, empates=False)
    assert est.delong(y, a, b)['p_unilateral'] < 0.01
    r = est.delong(y, a, 2 * a + 5)          # transformación monótona: mismo orden
    assert r['diferencia'] == 0 and r['p_bilateral'] == 1.0


def test_varianza_de_delong_parecida_a_la_bootstrap():
    y, a, b = _clasificacion(n=1500, empates=False)
    boot = est.bootstrap_auc(y, a, b, n_boot=800, seed=1)
    assert boot['ee_bootstrap'] == pytest.approx(est.delong(y, a, b)['ee'], rel=0.15)


# ------------------------------------------------------------------ jackknife y BCa
def test_jackknife_cerrado_igual_a_recalcular_sin_cada_fila():
    y, a, _ = _clasificacion(n=80)
    jk = est.jackknife_auc(y, a)
    for i in range(len(y)):
        fuera = np.arange(len(y)) != i
        assert jk[i] == pytest.approx(roc_auc_score(y[fuera], a[fuera]), abs=1e-12)


def test_jackknife_de_regresion_igual_a_recalcular():
    y = RNG.normal(size=50)
    pred = y + RNG.normal(scale=0.5, size=50)
    jk = est._jackknife_regresion(y, pred)
    for i in range(50):
        f = np.arange(50) != i
        e = y[f] - pred[f]
        assert jk['rmse'][i] == pytest.approx(np.sqrt(np.mean(e ** 2)))
        assert jk['mae'][i] == pytest.approx(np.mean(np.abs(e)))
        assert jk['r2'][i] == pytest.approx(r2_score(y[f], pred[f]))


def test_bca_sin_sesgo_ni_asimetria_es_el_percentil():
    replicas = np.linspace(-1, 1, 2001)
    lo, hi = est.intervalo_bca(0.0, replicas, np.array([1.0, -1.0, 1.0, -1.0]), nivel=0.9)
    assert (lo, hi) == pytest.approx(tuple(np.quantile(replicas, [0.05, 0.95])), abs=2e-3)


def test_intervalos_bootstrap_del_auc_contienen_la_estimacion():
    y, a, b = _clasificacion(n=800)
    r = est.bootstrap_auc(y, a, b, n_boot=500)
    assert r['ic_bca'][0] < r['estimacion'] < r['ic_bca'][1]
    assert r['estimacion'] == pytest.approx(roc_auc_score(y, b) - roc_auc_score(y, a))
    assert 0 <= r['prop_no_mejora'] <= 1 and len(r['replicas']) == 500


def test_bootstrap_de_regresion_orienta_la_mejora_a_favor_del_candidato():
    y = RNG.normal(size=600)
    malo, bueno = y + RNG.normal(scale=1.0, size=600), y + RNG.normal(scale=0.5, size=600)
    r = est.bootstrap_regresion(y, malo, bueno, n_boot=300)
    for k in ('rmse', 'mae', 'r2'):
        assert r[k]['estimacion'] > 0 and r[k]['ic_bca'][0] > 0
    assert r['r2']['estimacion'] == pytest.approx(r2_score(y, bueno) - r2_score(y, malo))


# ------------------------------------------------------------------ Diebold-Mariano
def test_dm_con_hln_y_h1_es_la_t_pareada():
    a, b = RNG.gamma(2.0, size=500), RNG.gamma(2.1, size=500)
    r = est.diebold_mariano(a, b, h=1, hln=True)
    t = stats.ttest_rel(a, b)
    assert r['estadistico'] == pytest.approx(t.statistic, rel=1e-10)
    assert r['p_bilateral'] == pytest.approx(t.pvalue, rel=1e-8)


def test_dm_con_horizonte_mayor_usa_autocovarianzas():
    d = np.convolve(RNG.normal(size=400), [1, 0.8], mode='valid')   # MA(1): γ1 > 0
    r1 = est.diebold_mariano(d, np.zeros_like(d), h=1, hln=False)
    r2 = est.diebold_mariano(d, np.zeros_like(d), h=2, hln=False)
    assert abs(r2['estadistico_dm']) < abs(r1['estadistico_dm'])


def test_bootstrap_estacionario_separa_diferencias_reales_de_nulas():
    real = est.bootstrap_estacionario(RNG.normal(0.3, 1, 800), n_boot=400)
    nula = est.bootstrap_estacionario(RNG.normal(0.0, 1, 800), n_boot=400, seed=3)
    assert real['p_bilateral'] < 0.01 and real['ic_percentil'][0] > 0
    assert nula['p_bilateral'] > 0.01 and real['bloque'] >= 1


# ------------------------------------------------------------ tamaños del efecto
def test_delta_de_cliff_por_definicion_y_relacion_con_el_auc():
    x, y = RNG.normal(size=30), RNG.normal(0.5, size=40)
    pares = np.sign(x[:, None] - y[None, :]).mean()
    assert est.delta_cliff(x, y)['delta'] == pytest.approx(pares)
    etiqueta, puntaje, _ = _clasificacion()
    d = est.delta_cliff(puntaje[etiqueta == 1], puntaje[etiqueta == 0])['delta']
    assert d == pytest.approx(2 * roc_auc_score(etiqueta, puntaje) - 1)
    assert est.delta_cliff([1, 2, 3], [4, 5, 6]) == {'delta': -1.0, 'magnitud': 'grande'}


def test_d_de_cohen_pareada_y_de_dos_muestras():
    d = RNG.normal(0.5, 2, 200)
    assert est.d_cohen(d)['d'] == pytest.approx(d.mean() / d.std(ddof=1))
    x, y = np.array([1.0, 2, 3, 4]), np.array([2.0, 3, 4, 5])
    assert est.d_cohen(x, y)['d'] == pytest.approx(-1 / np.std([1, 2, 3, 4], ddof=1))


def test_holm_ajusta_en_el_orden_original():
    p = est.ajustar_pvalores([0.01, 0.04, 0.03], 'holm')
    assert p == pytest.approx([0.03, 0.06, 0.06])


def test_friedman_coincide_con_scipy_y_ordena_los_rangos():
    import pandas as pd
    base = RNG.normal(size=(12, 1))
    m = pd.DataFrame(base + np.array([0.0, 0.3, 1.0]) + RNG.normal(scale=0.2, size=(12, 3)),
                     columns=['malo', 'medio', 'bueno'])
    r = est.friedman(m)
    assert r['chi2'] == pytest.approx(stats.friedmanchisquare(m['malo'], m['medio'], m['bueno']).statistic)
    assert list(r['rangos_medios'].index) == ['bueno', 'medio', 'malo']
    assert r['dc'] == pytest.approx(2.344 * np.sqrt(3 * 4 / (6 * 12)), rel=1e-3)
    assert r['p_nemenyi'].loc['bueno', 'malo'] < 0.01
    assert list(est.friedman(-m, mayor_es_mejor=False)['rangos_medios'].index) == ['bueno', 'medio', 'malo']


def test_mcs_deja_fuera_al_modelo_claramente_peor():
    import pandas as pd
    perd = pd.DataFrame({'a': RNG.gamma(2.0, size=1500), 'b': RNG.gamma(2.0, size=1500),
                         'malo': RNG.gamma(2.0, size=1500) + 1.0})
    r = est.conjunto_confianza_modelos(perd, alfa=0.10, n_boot=300)
    assert 'malo' in r['excluidos'] and set(r['incluidos']) <= {'a', 'b'} and r['incluidos']
    assert r['p_mcs']['malo'] < 0.10


def test_giacomini_white_detecta_ventaja_condicional():
    x = RNG.normal(size=3000)
    a = RNG.gamma(2.0, size=3000)
    b = a - 0.5 * x + RNG.normal(scale=0.5, size=3000)    # b gana cuando x > 0 y pierde cuando x < 0
    r = est.giacomini_white(a, b, instrumentos=x)
    assert r['p'] < 1e-6 and r['coeficientes'][1] > 0.4
    nulo = est.giacomini_white(a, a + RNG.normal(scale=0.5, size=3000), instrumentos=RNG.normal(size=3000))
    assert nulo['p'] > 0.001


def test_ece_cero_si_calibrado_y_igual_al_desplazamiento_si_no():
    from src.evaluacion import ece
    p = RNG.uniform(0.05, 0.85, 200000)
    y = (RNG.random(200000) < p).astype(int)
    assert ece(y, p) < 0.01 and ece(y, p, estrategia='cuantil') < 0.01
    assert ece(y, p + 0.1) == pytest.approx(0.1, abs=0.01)


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        est.delong([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        est.delong([0, 1, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        est.diebold_mariano([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
