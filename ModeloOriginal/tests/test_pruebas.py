"""Pruebas de las pruebas de hipótesis del protocolo: casos con respuesta conocida."""
import numpy as np
import pytest
from scipy import stats

from src import pruebas


def test_t_corregida_es_mas_conservadora_que_la_t_pareada():
    rng = np.random.default_rng(0)
    a = rng.normal(0.80, 0.01, 15)
    b = a - 0.004 + rng.normal(0, 0.003, 15)
    r = pruebas.t_corregida(a, b)
    corriente = stats.ttest_rel(a, b)
    assert r['gl'] == 14 and 0 < r['t'] < corriente.statistic and r['p'] > corriente.pvalue
    d = a - b
    assert r['t'] == pytest.approx(d.mean() / np.sqrt((1 / 15 + 0.25) * d.var(ddof=1)))
    assert pruebas.t_corregida(a, a)['p'] == 1.0


def test_wilcoxon_con_todas_las_diferencias_a_favor():
    a, b = np.arange(1, 16) / 100 + 0.5, np.zeros(15) + 0.5
    r = pruebas.wilcoxon_pareado(a, b)
    assert r['a_favor'] == 15 and r['p'] == pytest.approx(2 / 2 ** 15)      # el p-valor exacto mínimo con 15 pliegues
    assert pruebas.wilcoxon_pareado(a, a)['p'] == 1.0


def test_permutacion_detecta_una_diferencia_y_respeta_la_nula():
    rng = np.random.default_rng(1)
    base = rng.normal(1.0, 0.5, 3000) ** 2
    mejor = pruebas.permutacion_pareada(base - 0.05 + rng.normal(0, 0.2, 3000), base, n_perm=2000)
    assert mejor['diferencia'] < 0 and mejor['p'] < 0.01
    igual = pruebas.permutacion_pareada(base + rng.normal(0, 0.2, 3000), base + rng.normal(0, 0.2, 3000), n_perm=2000)
    assert igual['p'] > 0.05


def test_mcnemar_exacto():
    y = np.array([1] * 40 + [0] * 60)
    a, b = y.copy(), y.copy()
    a[:3] = 0                                   # a falla 3 que b acierta
    b[40:55] = 1                                # b falla 15 que a acierta
    r = pruebas.mcnemar(y, a, b)
    assert (r['solo_a'], r['solo_b']) == (15, 3)
    assert r['p'] == pytest.approx(stats.binomtest(15, 18, 0.5).pvalue)
    assert pruebas.mcnemar(y, a, a)['p'] == 1.0


def test_campana_marca_la_fraccion_pedida():
    puntaje = np.array([0.9, 0.1, 0.5, 0.5, 0.5, 0.2, 0.8, 0.3])
    marcado = pruebas.marcar_campana(puntaje, 0.25)
    assert marcado.sum() == 2 and marcado[0] == 1 and marcado[6] == 1
    assert pruebas.marcar_campana(np.zeros(100), 0.25).sum() == 25          # con empates, la misma cantidad


def test_bootstrap_bca_de_la_diferencia_de_rmse():
    rng = np.random.default_rng(2)
    y = rng.normal(size=4000)
    ruido = rng.normal(0, 1.0, 4000)
    a, b = y + 0.95 * ruido, y + 1.0 * ruido                                # a es un 5 % mejor, con errores pareados
    r = pruebas.bootstrap_rmse_pareado(y, a, b, n_boot=600)
    assert r['diferencia'] == pytest.approx(-0.05 * np.sqrt(np.mean(ruido ** 2)), rel=1e-9)
    assert r['ic_bajo'] < r['diferencia'] < r['ic_alto'] < 0 and r['p'] < 0.01
    c = y + rng.normal(0, 1.0, 4000)
    d = y + rng.normal(0, 1.0, 4000)
    nulo = pruebas.bootstrap_rmse_pareado(y, c, d, n_boot=600)
    assert nulo['ic_bajo'] < 0 < nulo['ic_alto'] and nulo['p'] > 0.05
