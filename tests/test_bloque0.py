"""Pruebas del Bloque 0 sobre una submuestra de los datos reales.

Se ejecutan con ``python -m pytest tests -q`` desde la raíz del proyecto. Usan pocas
filas y presupuestos mínimos para terminar en un par de minutos; su objetivo es
comprobar la ausencia de fugas, la uniformidad de las interfaces y que las 14
especificaciones y los 4 optimizadores funcionan de extremo a extremo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import balanceo as bal
from src import datos, evaluacion, experimento, modelos, optimizacion
from src.config import SEED, fijar_semillas
from src.registro import Registro

N_SUB = 3000


@pytest.fixture(scope='session')
def preparado():
    fijar_semillas(SEED)
    clientes, tx = datos.cargar_tablas()
    clientes = clientes.sample(N_SUB, random_state=SEED).reset_index(drop=True)
    tx = tx[tx['customer_id'].isin(clientes['customer_id'])]
    df = datos.unir(clientes, datos.derivar_variables_transaccionales(datos.limpiar_transacciones(tx)))
    particion = datos.partir(df, test_size=0.2, seed=SEED)
    diagnosticos = datos.diagnosticos_objetivo(df, particion)
    conjuntos = datos.separar(datos.depurar(df), particion)
    etiqueta = datos.construir_etiqueta(conjuntos.y_train_cont, metodo='percentil', q=0.75)
    return {'df': df, 'particion': particion, 'diagnosticos': diagnosticos, 'conjuntos': conjuntos,
            'etiqueta': etiqueta, 'y_train': etiqueta.binarizar(conjuntos.y_train_cont),
            'y_test': etiqueta.binarizar(conjuntos.y_test_cont)}


def _pipe_logistica(X, balanceo='ninguno', y=None):
    spec = modelos.catalogo('clasificacion')['logistica']
    return spec, bal.construir_pipeline(spec, balanceo, X, SEED, y)


# ------------------------------------------------------------------- datos / fugas
def test_particion_no_se_solapa_y_es_reproducible(preparado):
    p = preparado['particion']
    assert len(p.idx_train.intersection(p.idx_test)) == 0
    p2 = datos.partir(preparado['df'], test_size=0.2, seed=SEED)
    assert p.idx_train.equals(p2.idx_train)


def test_umbral_se_estima_solo_con_entrenamiento(preparado):
    c, e, p = preparado['conjuntos'], preparado['etiqueta'], preparado['particion']
    assert e.n_train == len(p.idx_train)
    assert e.umbral == pytest.approx(float(c.y_train_cont.quantile(0.75)))
    assert 0.20 < preparado['y_train'].mean() < 0.30


@pytest.mark.parametrize('metodo,q', [('mediana', None), ('percentil', 0.8)])
def test_umbral_no_depende_de_prueba_caso_sintetico(metodo, q):
    # entrenamiento con valores bajos, prueba con valores altos: el cuantil del total difiere
    y = pd.Series(np.r_[np.linspace(0.1, 0.3, 200), np.linspace(0.6, 0.9, 100)])
    idx_train, idx_test = y.index[:200], y.index[200:]
    e = datos.construir_etiqueta(y.loc[idx_train], metodo=metodo, q=q)
    esperado = y.loc[idx_train].median() if metodo == 'mediana' else y.loc[idx_train].quantile(q)
    assert e.umbral == pytest.approx(float(esperado))
    assert e.umbral != pytest.approx(float(y.median() if metodo == 'mediana' else y.quantile(q)))
    assert e.binarizar(y.loc[idx_test]).all()               # toda la prueba queda por encima


def test_etiqueta_por_defecto_es_percentil_75(preparado):
    e = datos.construir_etiqueta(preparado['conjuntos'].y_train_cont)
    assert e.metodo == 'percentil' and e.q == 0.75
    assert 0.20 < e.binarizar(preparado['conjuntos'].y_train_cont).mean() < 0.30


def test_diagnosticos_usan_train_para_el_objetivo_y_total_para_estructura(preparado):
    d, p = preparado['diagnosticos'], preparado['particion']
    assert d['n_train'] == len(p.idx_train) and d['n_total'] == len(preparado['df'])
    assert d['r_active_products'] < -0.8            # la escalera sigue presente en la submuestra
    assert set(d['escalera'].columns) == {'count', 'mean', 'std', 'salto'}
    assert d['corr_tx'].abs().max() < 0.2
    assert 'pares_duplicados' in d['ambito']['total'] and 'r_active_products' in d['ambito']['entrenamiento']


def test_duplicados_permutados_se_reconocen_sobre_la_tabla_completa():
    # una columna permutada entre las 48.723 filas solo coincide como conjunto de valores
    # cuando se compara la tabla entera; sobre una submuestra la coincidencia se pierde
    clientes = pd.read_csv(datos.localizar_datos() / 'customer_data.csv')
    particion = datos.partir(clientes, test_size=0.2, seed=SEED)
    d = datos.diagnosticos_objetivo(clientes, particion)
    pares = d['pares_duplicados'].set_index('par')
    assert pares.loc['avg_tx_value / average_transaction_value', 'mismos_valores_pct'] == pytest.approx(100.0)
    assert pares.loc['avg_tx_value / average_transaction_value', 'coincide_por_fila_pct'] < 5
    assert pares.loc['transaction_frequency / avg_daily_transactions', 'coincide_por_fila_pct'] == pytest.approx(100.0)
    sub = clientes.sample(3000, random_state=SEED)
    d_sub = datos.diagnosticos_objetivo(sub, datos.partir(sub, test_size=0.2, seed=SEED))
    assert d_sub['pares_duplicados'].set_index('par').loc['avg_tx_value / average_transaction_value',
                                                         'mismos_valores_pct'] < 5
    assert d['r_active_products'] == pytest.approx(-0.876, abs=0.01)


def test_particion_con_validacion(preparado):
    p = datos.partir(preparado['df'], test_size=0.2, val_size=0.16, seed=SEED)
    assert p.idx_val is not None
    total = len(p.idx_train) + len(p.idx_val) + len(p.idx_test)
    assert total == len(preparado['df'])
    assert abs(len(p.idx_test) / total - 0.2) < 0.01
    assert abs(len(p.idx_val) / total - 0.16) < 0.01


# ------------------------------------------------------------- catálogo y pipelines
@pytest.mark.parametrize('tarea', modelos.TAREAS)
def test_catalogo_tiene_siete_modelos_con_rejillas_comparables(tarea):
    cat = modelos.catalogo(tarea)
    assert len(cat) == 7
    for spec in cat.values():
        assert spec.espacio_grid and spec.espacio and spec.justificacion
        assert 30 <= spec.n_configuraciones_grid() <= 48
        assert set(spec.complejidad) == {'entrenamiento', 'inferencia'}
    assert 'AUC' not in modelos.catalogo('regresion')['knn'].justificacion


@pytest.mark.parametrize('tarea', modelos.TAREAS)
@pytest.mark.parametrize('balanceo', bal.CATALOGO_BALANCEO)
def test_pipelines_se_construyen_y_ajustan(preparado, tarea, balanceo):
    c = preparado['conjuntos']
    X = c.X_train.iloc[:400]
    y = preparado['y_train'].iloc[:400] if tarea == 'clasificacion' else c.y_train_cont.iloc[:400]
    for nombre, spec in modelos.catalogo(tarea).items():
        ok, motivo = bal.aplica(spec, balanceo, y if tarea == 'clasificacion' else None)
        if not ok:
            assert motivo
            with pytest.raises(ValueError):
                bal.construir_pipeline(spec, balanceo, X, SEED, y)
            continue
        pipe = bal.construir_pipeline(spec, balanceo, X, SEED, y)
        assert pipe.steps[0][0] == 'preprocesado' and pipe.steps[-1][0] == 'modelo'
        if balanceo in ('smote', 'adasyn'):
            assert pipe.steps[1][0] == 'muestreo'
        for clave in list(spec.espacio_grid) + list(spec.espacio):
            assert clave in pipe.get_params(), f'{nombre}: {clave}'
        pipe.fit(X, y)
        m = evaluacion.evaluar(pipe, c.X_test.iloc[:200],
                               (preparado['y_test'] if tarea == 'clasificacion' else c.y_test_cont).iloc[:200], tarea)
        assert np.isfinite(m[evaluacion.NOMBRE_METRICA[tarea]])


def test_class_weight_xgboost_usa_proporcion_del_pliegue(preparado):
    spec = modelos.catalogo('clasificacion')['xgboost']
    y = preparado['y_train']
    pipe = bal.construir_pipeline(spec, 'class_weight', preparado['conjuntos'].X_train, SEED, y)
    esperado = (y == 0).sum() / (y == 1).sum()
    assert pipe.named_steps['modelo'].get_params()['scale_pos_weight'] == pytest.approx(esperado)


def test_sobremuestreo_no_aplica_con_clases_equilibradas(preparado):
    c = preparado['conjuntos']
    y_mediana = datos.construir_etiqueta(c.y_train_cont, metodo='mediana').binarizar(c.y_train_cont)
    spec = modelos.catalogo('clasificacion')['logistica']
    for b in ('smote', 'adasyn'):
        ok, motivo = bal.aplica(spec, b, y_mediana)
        assert not ok and 'equilibradas' in motivo
        assert bal.aplica(spec, b, preparado['y_train'])[0]        # con percentil 0,75 sí aplica
    fila = experimento.correr_experimento('clasificacion', 'logistica', 'adasyn', 'grid',
                                          c.X_train.iloc[:600], y_mediana.iloc[:600], n_jobs=1)
    assert fila['estado'] == 'no_aplica' and 'equilibradas' in fila['motivo']


# ---------------------------------------------------------- espacio y optimizadores
@pytest.mark.parametrize('tarea', modelos.TAREAS)
def test_decodificacion_del_espacio_respeta_rangos_incluidos_extremos(tarea):
    rng = np.random.default_rng(SEED)
    for spec in modelos.catalogo(tarea).values():
        n = len(spec.espacio)
        genomas = [list(rng.random(n)) for _ in range(30)] + [[0.0] * n, [1.0] * n]
        for g in genomas:
            p = optimizacion.decodificar_genoma(spec.espacio, g)
            for nombre, (tipo, *args) in spec.espacio.items():
                if tipo == 'cat':
                    assert p[nombre] in args[0]
                else:
                    assert args[0] <= p[nombre] <= args[1], (spec.nombre, nombre, p[nombre])
                if tipo in ('int', 'int_log'):
                    assert isinstance(p[nombre], int)
        assert set(optimizacion.distribuciones_random(spec.espacio)) == set(spec.espacio)


def test_entero_log_es_reproducible_con_estado_global():
    d = optimizacion._EnteroLog(5, 600)
    fijar_semillas(SEED)
    a = [d.rvs() for _ in range(5)]
    fijar_semillas(SEED)
    assert a == [d.rvs() for _ in range(5)]
    assert d.rvs(random_state=3) == d.rvs(random_state=3)
    v = d.rvs(size=200, random_state=SEED)
    assert v.min() >= 5 and v.max() <= 600


def test_traza_acumulada_ignora_nan():
    assert optimizacion._traza_acumulada([0.5, np.nan, 0.7, 0.6]) == [0.5, 0.5, 0.7, 0.7]
    assert optimizacion._traza_acumulada([np.nan, 0.6])[1] == 0.6


@pytest.mark.parametrize('optimizador', optimizacion.OPTIMIZADORES)
def test_cuatro_optimizadores_misma_interfaz(preparado, optimizador):
    c = preparado['conjuntos']
    X, y = c.X_train.iloc[:800], preparado['y_train'].iloc[:800]
    spec, pipe = _pipe_logistica(X)
    cv_int = optimizacion.construir_cv('clasificacion', 2, SEED)
    res = optimizacion.buscar(optimizador, pipe, spec.espacio_grid, spec.espacio, X, y, cv_int,
                              'roc_auc', presupuesto=8, seed=SEED, n_jobs=1)
    assert res.optimizador == optimizador
    if optimizador == 'grid':
        assert res.n_evaluaciones == spec.n_configuraciones_grid()      # exhaustivo
    else:
        assert res.n_evaluaciones == 8                                  # consume el presupuesto exacto
    assert len(res.traza) == res.n_evaluaciones and res.tiempo_s > 0
    assert all(b >= a for a, b in zip(res.traza, res.traza[1:]))       # anytime: no decrece
    assert set(res.mejores_params) == set(spec.espacio)
    assert res.detalle['presupuesto'] == (spec.n_configuraciones_grid() if optimizador == 'grid' else 8)
    assert np.isfinite(evaluacion.evaluar(res.estimador, c.X_test, preparado['y_test'], 'clasificacion')['auc'])
    if optimizador == 'genetica':
        d = res.detalle
        assert d['generaciones'] >= 1 and len(d['diversidad_por_generacion']) == d['generaciones'] + 1
    if optimizador == 'bayesiana':
        assert res.detalle['surrogate'] == 'TPE'


@pytest.mark.parametrize('presupuesto', [3, 5, 12])
def test_genetico_consume_exactamente_el_presupuesto(preparado, presupuesto):
    c = preparado['conjuntos']
    X, y = c.X_train.iloc[:500], preparado['y_train'].iloc[:500]
    spec, pipe = _pipe_logistica(X)
    cv_int = optimizacion.construir_cv('clasificacion', 2, SEED)
    res = optimizacion.buscar_genetica(pipe, spec.espacio, X, y, cv_int, 'roc_auc',
                                       presupuesto=presupuesto, seed=SEED, n_jobs=1)
    assert res.n_evaluaciones == presupuesto == len(res.traza)


def test_validacion_anidada_devuelve_k_ext_puntajes_y_modelo_final(preparado):
    c = preparado['conjuntos']
    X, y = c.X_train.iloc[:900], preparado['y_train'].iloc[:900]
    spec = modelos.catalogo('clasificacion')['logistica']
    res = optimizacion.validacion_anidada(
        lambda yf: bal.construir_pipeline(spec, 'ninguno', X, SEED, yf),
        spec.espacio_grid, spec.espacio, 'random', X, y, 'clasificacion',
        k_ext=3, k_int=2, presupuesto=4, seed=SEED, n_jobs=1)
    assert len(res.puntajes_ext) == 3 and len(res.params_por_pliegue) == 3 and len(res.detalles) == 3
    assert set(res.metricas_ext) == {'accuracy', 'precision', 'recall', 'f1', 'auc', 'brier'}
    assert res.n_evaluaciones == 12 and res.n_evaluaciones_final == 4
    assert 0.5 < res.media_ext < 1.0 and res.desv_ext >= 0
    assert set(res.params_finales) == set(spec.espacio)
    assert res.estimador_final is not None and len(res.traza_final) == 4
    assert np.isfinite(evaluacion.evaluar(res.estimador_final, c.X_test, preparado['y_test'], 'clasificacion')['auc'])


def test_validacion_anidada_realinea_y_permutada(preparado):
    c = preparado['conjuntos']
    X, y = c.X_train.iloc[:600], preparado['y_train'].iloc[:600]
    spec = modelos.catalogo('clasificacion')['naive_bayes']
    ctor = lambda yf: bal.construir_pipeline(spec, 'ninguno', X, SEED, yf)   # noqa: E731
    comun = dict(k_ext=2, k_int=2, seed=SEED, n_jobs=1, ajuste_final=False)
    a = optimizacion.validacion_anidada(ctor, spec.espacio_grid, spec.espacio, 'grid', X, y, 'clasificacion', **comun)
    b = optimizacion.validacion_anidada(ctor, spec.espacio_grid, spec.espacio, 'grid', X,
                                        y.sample(frac=1, random_state=1), 'clasificacion', **comun)
    assert a.puntajes_ext == pytest.approx(b.puntajes_ext)
    with pytest.raises(ValueError):
        optimizacion.validacion_anidada(ctor, spec.espacio_grid, spec.espacio, 'grid', X,
                                        y.iloc[:-1], 'clasificacion', **comun)


def test_regresion_anidada_rmse_negativo(preparado):
    c = preparado['conjuntos']
    X, y = c.X_train.iloc[:900], c.y_train_cont.iloc[:900]
    spec = modelos.catalogo('regresion')['ridge']
    res = optimizacion.validacion_anidada(
        lambda yf: bal.construir_pipeline(spec, 'ninguno', X, SEED, yf),
        spec.espacio_grid, spec.espacio, 'grid', X, y, 'regresion', k_ext=3, k_int=2, seed=SEED,
        n_jobs=1, ajuste_final=False)
    assert all(p < 0 for p in res.puntajes_ext)                 # -RMSE
    assert set(res.metricas_ext) == {'rmse', 'mae', 'r2'}
    assert res.params_finales is None


# -------------------------------------------------------------------- registro
def test_registro_ida_y_vuelta_con_tipos_numpy(tmp_path):
    r = Registro(tmp_path / 'exp.parquet')
    fila = {'tarea': 'clasificacion', 'modelo': 'knn', 'balanceo': 'ninguno', 'optimizador': 'grid',
            'semilla': SEED, 'aplica': True, 'media_ext': 0.7, 'desv_ext': 0.01,
            'hiperparametros': {'modelo__n_neighbors': np.int64(50), 'modelo__flag': np.bool_(True),
                                'modelo__x': np.float32(0.5)},
            'metricas_ext': {'auc': [np.float64(0.7), 0.71]}, 'tiempo_total_s': 1.0, 'n_evaluaciones': 4}
    r.anadir(fila)
    assert len(r) == 1 and r.existe(tarea='clasificacion', modelo='knn', balanceo='ninguno',
                                    optimizador='grid', semilla=SEED)
    with pytest.raises(ValueError):
        r.anadir(fila)
    r.anadir({**fila, 'media_ext': 0.75}, sobrescribir=True)
    assert not (tmp_path / 'exp.parquet.tmp').exists()          # escritura atómica: no queda el temporal
    df = Registro(tmp_path / 'exp.parquet').cargar()
    assert len(df) == 1 and df.loc[0, 'media_ext'] == 0.75 and df.loc[0, 'estado'] == 'ok'
    h = df.loc[0, 'hiperparametros']
    assert h['modelo__n_neighbors'] == 50 and isinstance(h['modelo__n_neighbors'], int)
    assert h['modelo__flag'] is True and isinstance(h['modelo__x'], float)
    r.anadir({**fila, 'modelo': 'svm', 'aplica': False, 'motivo': 'x'})
    assert Registro(tmp_path / 'exp.parquet').cargar().set_index('modelo').loc['svm', 'estado'] == 'no_aplica'


def test_cerrojo_impide_dos_escritores(tmp_path):
    import os
    import subprocess
    import sys
    from src.registro import dueno_del_cerrojo, ruta_cerrojo
    tabla = tmp_path / 'exp.parquet'
    fila = {'tarea': 'regresion', 'modelo': 'ridge', 'balanceo': 'ninguno', 'optimizador': 'grid',
            'semilla': SEED, 'aplica': True, 'media_ext': -0.06}
    otro = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        ruta_cerrojo(tabla).write_text(f'{otro.pid} prueba', encoding='utf-8')
        assert dueno_del_cerrojo(tabla) == otro.pid
        with pytest.raises(RuntimeError, match='en uso'):
            Registro(tabla).anadir(fila)
        assert Registro(tabla).cargar().empty                    # leer sigue permitido
        ruta_cerrojo(tabla).write_text(f'{os.getpid()} yo', encoding='utf-8')
        Registro(tabla).anadir(fila)                              # el dueño sí escribe
    finally:
        otro.kill()
        otro.wait()
    ruta_cerrojo(tabla).write_text(f'{otro.pid} huerfano', encoding='utf-8')
    assert dueno_del_cerrojo(tabla) is None                       # cerrojo de un proceso muerto: libre
    Registro(tabla).anadir({**fila, 'modelo': 'lasso'})
    assert len(Registro(tabla)) == 2


def test_combinaciones_suman_140():
    assert len(experimento.combinaciones('clasificacion')) == 112
    assert len(experimento.combinaciones('regresion')) == 28


def test_correr_experimento_extremo_a_extremo(preparado, tmp_path):
    c = preparado['conjuntos']
    X, y = c.X_train.iloc[:700], preparado['y_train'].iloc[:700]
    r = Registro(tmp_path / 'exp.parquet')
    fila = experimento.correr_experimento('clasificacion', 'logistica', 'class_weight', 'random', X, y,
                                          seed=SEED, k_ext=2, k_int=2, presupuesto=3, registro=r, n_jobs=1,
                                          etiqueta={'metodo': 'percentil', 'umbral': 0.4},
                                          carpeta_modelos=tmp_path / 'modelos')
    assert fila['estado'] == 'ok' and len(r) == 1
    assert fila['n_evaluaciones'] == 6 and fila['n_evaluaciones_final'] == 3
    assert fila['tiempo_por_evaluacion_s'] > 0 and fila['etiqueta_metodo'] == 'percentil'
    assert set(fila['hiperparametros']) == set(modelos.catalogo('clasificacion')['logistica'].espacio)
    assert len(fila['detalles']) == 2 and fila['presupuesto'] == 3
    modelo = experimento.cargar_modelo('clasificacion', 'logistica', 'class_weight', 'random', SEED,
                                       carpeta=tmp_path / 'modelos')
    assert np.isfinite(evaluacion.evaluar(modelo, c.X_test, preparado['y_test'], 'clasificacion')['auc'])
    # no aplica: KNN × class_weight queda registrado con motivo
    fila2 = experimento.correr_experimento('clasificacion', 'knn', 'class_weight', 'grid', X, y,
                                           registro=r, n_jobs=1)
    assert fila2['estado'] == 'no_aplica' and 'no admite' in fila2['motivo'] and len(r) == 2
    # reanudación: la existente se salta
    hechas = experimento.correr_todo(X, y, 'clasificacion', r, modelos=['logistica'],
                                     balanceos=['class_weight'], optimizadores=['random'],
                                     presupuesto=3, k_ext=2, k_int=2, n_jobs=1, informar=lambda s: None)
    assert hechas == []


def test_correr_todo_continua_tras_un_error(preparado, tmp_path, monkeypatch):
    c = preparado['conjuntos']
    X, y = c.X_train.iloc[:400], preparado['y_train'].iloc[:400]
    original = experimento.validacion_anidada

    def rota(constructor, espacio_grid, espacio, optimizador, X_, y_, tarea, **kw):
        if 'var_smoothing' in ' '.join(espacio):            # naive_bayes falla a propósito
            raise RuntimeError('fallo simulado')
        return original(constructor, espacio_grid, espacio, optimizador, X_, y_, tarea, **kw)

    monkeypatch.setattr(experimento, 'validacion_anidada', rota)
    r = Registro(tmp_path / 'exp.parquet')
    hechas = experimento.correr_todo(X, y, 'clasificacion', r, modelos=['naive_bayes', 'logistica'],
                                     balanceos=['ninguno'], optimizadores=['grid'], k_ext=2, k_int=2,
                                     n_jobs=1, guardar_modelo=False, informar=lambda s: None)
    estados = {f['modelo']: f['estado'] for f in hechas}
    assert estados == {'naive_bayes': 'error', 'logistica': 'ok'}
    df = r.cargar()
    assert 'fallo simulado' in df.set_index('modelo').loc['naive_bayes', 'motivo']
    # al reanudar sin el fallo, la corrida con error se repite y la que salió bien se salta
    monkeypatch.setattr(experimento, 'validacion_anidada', original)
    assert r.estado(tarea='clasificacion', modelo='naive_bayes', balanceo='ninguno', optimizador='grid',
                    semilla=SEED) == 'error'
    otra = experimento.correr_todo(X, y, 'clasificacion', r, modelos=['naive_bayes', 'logistica'],
                                   balanceos=['ninguno'], optimizadores=['grid'], k_ext=2, k_int=2,
                                   n_jobs=1, guardar_modelo=False, informar=lambda s: None)
    assert [f['modelo'] for f in otra] == ['naive_bayes'] and otra[0]['estado'] == 'ok'
    assert len(r) == 2 and r.estado(tarea='clasificacion', modelo='naive_bayes', balanceo='ninguno',
                                    optimizador='grid', semilla=SEED) == 'ok'
    monkeypatch.setattr(experimento, 'validacion_anidada', rota)
    with pytest.raises(RuntimeError):
        experimento.correr_todo(X, y, 'clasificacion', Registro(tmp_path / 'e2.parquet'), modelos=['naive_bayes'],
                                balanceos=['ninguno'], optimizadores=['grid'], k_ext=2, k_int=2, n_jobs=1,
                                continuar_ante_error=False, guardar_modelo=False, informar=lambda s: None)


# ------------------------------------------------------------------ multi-fidelidad (RF)
def test_multifidelidad_random_forest_busca_con_100_y_reajusta_con_300(preparado):
    c = preparado['conjuntos']
    X, y = c.X_train.iloc[:500], preparado['y_train'].iloc[:500]
    spec = modelos.catalogo('clasificacion')['random_forest']
    assert spec.fidelidad == {'busqueda': {'modelo__n_estimators': 100}, 'final': {'modelo__n_estimators': 300}}
    assert 'modelo__n_estimators' not in spec.espacio_grid and 'modelo__n_estimators' not in spec.espacio
    pipe = bal.construir_pipeline(spec, 'ninguno', X, SEED, y)
    cv_int = optimizacion.construir_cv('clasificacion', 2, SEED)
    visto = []
    original = optimizacion.cross_val_score

    def espia(est, *a, **kw):                  # registra con cuántos árboles se evalúa cada configuración
        visto.append(est.get_params()['modelo__n_estimators'])
        return original(est, *a, **kw)

    optimizacion.cross_val_score = espia
    try:
        res = optimizacion.buscar('genetica', pipe, spec.espacio_grid, spec.espacio, X, y, cv_int, 'roc_auc',
                                  presupuesto=3, seed=SEED, n_jobs=1,
                                  fijos_busqueda=spec.fidelidad['busqueda'], fijos_final=spec.fidelidad['final'])
    finally:
        optimizacion.cross_val_score = original
    assert visto and set(visto) == {100}                   # la búsqueda usa baja fidelidad
    assert res.n_evaluaciones == 3
    assert res.estimador.named_steps['modelo'].n_estimators == 300   # el ganador se reajusta a alta
    assert res.mejores_params['modelo__n_estimators'] == 300
    assert res.detalle['fidelidad'] == {'busqueda': {'modelo__n_estimators': 100},
                                        'final': {'modelo__n_estimators': 300}}


def test_correr_experimento_registra_la_fidelidad(preparado, tmp_path):
    c = preparado['conjuntos']
    r = Registro(tmp_path / 'exp.parquet')
    fila = experimento.correr_experimento('regresion', 'random_forest', 'ninguno', 'random',
                                          c.X_train.iloc[:400], c.y_train_cont.iloc[:400], k_ext=2, k_int=2,
                                          presupuesto=2, registro=r, n_jobs=1, guardar_modelo=False)
    assert fila['estado'] == 'ok'
    assert fila['hiperparametros']['modelo__n_estimators'] == 300
    assert all(pp['modelo__n_estimators'] == 300 for pp in fila['params_por_pliegue'])
    assert r.cargar().loc[0, 'fidelidad']['final'] == {'modelo__n_estimators': 300}
