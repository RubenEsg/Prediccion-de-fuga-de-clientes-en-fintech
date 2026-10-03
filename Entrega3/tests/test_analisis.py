"""Pruebas de los módulos de análisis y presentación: formato, ECE, residuos y optimizadores.

Se ejecutan con ``python -m pytest tests -q`` desde la raíz del proyecto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import analisis, evaluacion
from src.formato import dec, ent, enumerar, intervalo, p_valor, pct


def test_formato_en_castellano():
    assert dec(0.68321) == '0,6832'
    assert dec(-0.5, 1) == '−0,5' and dec(0.04, 4, True) == '+0,0400'
    assert ent(48723) == '48.723' and pct(0.2573) == '25,7 %'
    assert p_valor(0.0042) == 'p = 0,004' and p_valor(0.0) == 'p < 0,001'
    assert p_valor(3e-19) == 'p < 10⁻¹⁸' and p_valor(3e-19, grafico=True) == 'p < $10^{-18}$'
    assert enumerar(['a', 'b', 'c']) == 'a, b y c' and enumerar(['a']) == 'a'
    assert intervalo(0.1, 0.2, 2) == '[0,10; 0,20]'


def test_diagnostico_residuos_no_rechaza_con_ruido_blanco():
    rng = np.random.default_rng(3)
    x = rng.normal(size=4000)
    y = x + rng.normal(scale=0.5, size=4000)
    d = evaluacion.diagnostico_residuos(y, x, n_bds=1500)
    assert d.loc['White', 'p'] > 0.01 and d.loc['Ljung-Box, 10 retardos', 'p'] > 0.01
    assert d.loc['Jarque-Bera', 'p'] > 0.01
    sesgado = evaluacion.diagnostico_residuos(x + rng.exponential(size=4000), x, n_bds=1500)
    assert sesgado.loc['Jarque-Bera', 'p'] < 0.001 and sesgado.loc['asimetría', 'estadistico'] > 1


def _tabla_sintetica():
    """Tabla maestra mínima: 2 bloques × 4 optimizadores, con trazas de 10 evaluaciones."""
    filas = []
    rng = np.random.default_rng(0)
    for modelo in ('m1', 'm2'):
        for opt, ventaja in zip(analisis.OPTIMIZADORES, (0.0, 0.001, 0.003, 0.002)):
            trazas = [list(np.maximum.accumulate(0.70 + ventaja + rng.normal(0, 0.01, 10))) for _ in range(5)]
            filas.append({'tarea': 'clasificacion', 'modelo': modelo, 'balanceo': 'ninguno', 'optimizador': opt,
                          'estado': 'ok', 'media_ext': 0.75 + ventaja, 'desv_ext': 0.003, 'metrica': 0.75 + ventaja,
                          'n_evaluaciones': 50, 'tiempo_por_evaluacion_s': 1.0 + ventaja * 100,
                          'tiempo_total_s': 60.0, 'optimismo_seleccion': -0.001, 'trazas': trazas,
                          'clave': f'clasificacion__{modelo}__ninguno__{opt}'})
    return pd.DataFrame(filas)


def test_resumen_y_curvas_anytime_de_optimizadores():
    t = _tabla_sintetica()
    r = analisis.resumen_optimizadores(t)
    assert list(r.index.get_level_values('optimizador')) == analisis.OPTIMIZADORES
    assert r.loc[('clasificacion', 'bayesiana'), 'rango_medio'] == 1.0
    curvas, por_busqueda = analisis.curvas_anytime(t, 'clasificacion', puntos=5)
    assert set(curvas['optimizador']) == set(analisis.OPTIMIZADORES)
    assert ((curvas['media'] >= 0) & (curvas['media'] <= 1)).all()
    assert len(por_busqueda) == 2 * 5 * 4
    m = analisis.matriz_optimizadores(t, 'clasificacion')
    assert m.shape == (2, 4)


def test_mejores_por_modelo_elige_sin_mirar_la_prueba():
    t = _tabla_sintetica()
    mej = analisis.mejores_por_modelo(t, 'clasificacion')
    assert set(mej['modelo']) == {'m1', 'm2'} and (mej['optimizador'] == 'bayesiana').all()
