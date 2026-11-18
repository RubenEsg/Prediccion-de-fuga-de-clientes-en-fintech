"""Pruebas de los formatos del informe (``src/informe.py``): las cifras del libro pasan todas por estas funciones."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip('matplotlib')
from src import informe as inf                                             # noqa: E402


def test_numeros_con_coma_decimal_y_signo_menos():
    assert inf.num(0.058871, 5) == '0,05887'
    assert inf.num(-0.00119, 5) == '−0,00119'
    assert inf.num(float('nan')) == '—' and inf.num(None) == '—'
    assert inf.entero(48723) == '48.723' and inf.entero(9745.0) == '9.745'


def test_p_valores():
    assert inf.pvalor(0.0004) == '< 0,001'
    assert inf.pvalor(0.0372) == '0,037'
    assert inf.pvalor(float('nan')) == '—'


def test_estadisticos_enteros_pequenos_y_corrientes():
    assert inf.estad(20.0) == '20'
    assert inf.estad(-4.3258) == '−4,3258'
    assert inf.estad(-6.8757e-06) == '−6,88 × 10⁻⁶'
    assert inf.estad(1234.56) == '1234,6'


def test_valores_y_rangos_de_hiperparametros():
    assert inf._valor(1e-11) == '10⁻¹¹' and inf._valor(1e5) == '10⁵'
    assert inf._valor(0.001) == '0,001' and inf._valor(1024) == '1.024'
    assert inf._valor(True) == 'sí' and inf._valor('auto') == '«auto»'
    assert inf._rango(('int_log', 5, 200)) == '5 a 200 (entero, escala logarítmica)'
    assert inf._rango(('cat', ['auto', 1])) == '{«auto», 1}'


def test_tabla_en_markdown_y_cuantas():
    tabla = inf.tabla_md(['Modelo', 'RMSE'], [['BDL', '0,05887']])
    assert tabla.splitlines() == ['| Modelo | RMSE |', '| :--- | ---: |', '| BDL | 0,05887 |']
    assert inf._cuantas(0, 4) == 'ninguna de las 4' and inf._cuantas(1, 4) == '1 de las 4' and inf._cuantas(4, 4) == 'las 4'
    assert inf._lista(['a', 'b', 'c']) == 'a, b y c' and inf._lista(['a']) == 'a'


def test_cada_pieza_del_libro_tiene_quien_la_genere():
    """Toda tabla o texto esperado por el libro está en el registro que usan las celdas de los cuadernos."""
    assert set(inf.ESPERADOS) == set(inf.PRODUCE)
    assert all(f in inf.PIEZAS for f in set(inf.PRODUCE.values()) | set(inf.DIBUJA.values()))
    assert np.all([callable(f) for f in inf.PIEZAS])
