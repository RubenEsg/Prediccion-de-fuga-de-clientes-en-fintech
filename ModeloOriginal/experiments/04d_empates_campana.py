"""Empates en el corte de la campaña del 25 %: cuánto dependen la captura y la prueba P4 de cómo se rompen.

La prueba P4 de clasificación (McNemar) marca como positivo el 25 % de clientes de mayor puntaje de cada modelo. El
protocolo no decía qué hacer cuando varios clientes comparten el puntaje del corte; el guion de las pruebas los ordena
por posición (``src.pruebas.marcar_campana``). En los benchmarks casi no hay empates. En el modelo original sí: su
probabilidad es una función escalonada (Proposición 5) y miles de clientes comparten el valor del corte, de modo que
quiénes entran en la campaña lo decide el desempate.

Este guion mide ese efecto, para los diez modelos y para el predictor óptimo (el del techo teórico):

- cuántos clientes empatan en el corte, cuántos de ellos entran y cuántos son de riesgo alto;
- la captura con el desempate del protocolo (por posición), su valor esperado y su desviación con un desempate al azar
  (hipergeométrica) y su rango posible. El dashboard, como el Entregable 3, rompe los empates al azar con su semilla y
  con las filas en otro orden: su cifra es un sorteo más de esa distribución;
- la prueba P4 del original frente a cada benchmark repetida con ``REPLICAS`` desempates al azar de los dos modelos, con
  el ajuste de Holm sobre las nueve comparaciones, como en el protocolo;
- la precisión media con la convención de scikit-learn (un punto por valor distinto del puntaje) y con los empates rotos
  al azar: con mesetas grandes la primera es menor que la de cualquier orden concreto.

Es un **análisis de sensibilidad posterior a los resultados** y fuera del protocolo: no cambia el veredicto, dice cuánto
depende de una decisión que el protocolo no fijó. ``active_products`` se usa solo para construir el predictor óptimo.

Salida: ``resultados/empates_campana.json``. Uso: ``python experiments/04d_empates_campana.py`` (después de 03, 04 y 04b).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import estadistica, plan, pruebas                                 # noqa: E402
from src.config import CARPETA_RESULTADOS, SEED                            # noqa: E402

FRACCION = 0.25
REPLICAS = 2000
REPLICAS_AP = 200


def al_azar(puntaje: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Marca los ``n`` clientes de mayor puntaje rompiendo los empates al azar."""
    marcado = np.zeros(len(puntaje), dtype=int)
    marcado[np.lexsort((rng.random(len(puntaje)), -puntaje))[:n]] = 1
    return marcado


def precision_media(y: np.ndarray, puntaje: np.ndarray, rng: np.random.Generator) -> dict:
    """Precisión media con la convención de scikit-learn y con los empates rotos al azar.

    ``average_precision_score`` trata cada grupo de clientes con el mismo puntaje como un solo punto de la curva, con la
    precisión del grupo entero: con mesetas grandes da un valor menor que el de cualquier orden concreto dentro de ellas.
    La alternativa es ordenar al azar a los empatados (``REPLICAS_AP`` veces) y promediar.
    """
    al_azar = []
    for _ in range(REPLICAS_AP):
        rango = np.empty(len(puntaje))
        rango[np.lexsort((rng.random(len(puntaje)), -puntaje))] = np.arange(len(puntaje), 0, -1)   # mismo orden, sin empates
        al_azar.append(average_precision_score(y, rango))
    return {'precision_media': float(average_precision_score(y, puntaje)), 'precision_media_azar': float(np.mean(al_azar)),
            'precision_media_azar_desviacion': float(np.std(al_azar, ddof=1))}


def empates(y: np.ndarray, puntaje: np.ndarray, n: int) -> dict:
    """Qué pasa en el corte de la campaña de un modelo y qué captura resulta con cada regla de desempate."""
    corte = np.sort(puntaje)[::-1][n - 1]
    arriba, empatados = puntaje > corte, puntaje == corte
    cupos, n_emp, pos_emp = n - int(arriba.sum()), int(empatados.sum()), int(y[empatados].sum())
    seguros, positivos = int(y[arriba].sum()), int(y.sum())
    fraccion = pos_emp / n_emp
    esperados = seguros + cupos * fraccion
    varianza = cupos * fraccion * (1 - fraccion) * (n_emp - cupos) / max(n_emp - 1, 1)   # hipergeométrica
    return {'valores_distintos': int(len(np.unique(puntaje))), 'corte': float(corte), 'empatados': n_emp, 'cupos': cupos,
            'positivos_empatados': pos_emp, 'positivos_seguros': seguros,
            'captura_protocolo': float(y[pruebas.marcar_campana(puntaje, FRACCION) == 1].sum() / positivos),
            'captura_esperada': float(esperados / positivos), 'captura_desviacion': float(np.sqrt(varianza) / positivos),
            'captura_minima': float((seguros + max(0, cupos - (n_emp - pos_emp))) / positivos),
            'captura_maxima': float((seguros + min(cupos, pos_emp)) / positivos)}


def main() -> None:
    p = plan.problema()
    c = p['conjuntos']
    y = p['y_test'].to_numpy()
    n = int(round(FRACCION * len(y)))
    pred = pd.read_parquet(CARPETA_RESULTADOS / 'predicciones_prueba.parquet')
    veredicto = json.loads((CARPETA_RESULTADOS / 'veredicto.json').read_text(encoding='utf-8'))['clasificacion']
    techo = json.loads((CARPETA_RESULTADOS / 'techo.json').read_text(encoding='utf-8'))
    modelos = plan.modelos('clasificacion')
    puntajes = {m: pred[f'clasificacion__{m}'].to_numpy(dtype=float) for m in modelos}
    # el predictor óptimo: P(puntaje > umbral | índice real), como en 04c_optimo.py
    pesos = np.array([techo['pesos'][str(k)] for k in range(1, 6)])
    niveles = -techo['delta'] * np.arange(5)
    productos = p['df'].loc[c.X_test.index, 'active_products'].to_numpy()
    indice = c.y_test_cont.to_numpy() + techo['delta'] * (productos - 1)
    optimo = ((indice[:, None] + niveles[None, :]) > techo['umbral']) @ pesos

    rng_ap = np.random.default_rng(SEED)
    salida = {'n': int(len(y)), 'campana': n, 'positivos': int(y.sum()), 'replicas': REPLICAS, 'replicas_precision_media': REPLICAS_AP,
              'semilla': SEED,
              'modelos': {m: {**empates(y, puntajes[m], n), **precision_media(y, puntajes[m], rng_ap)} for m in modelos},
              'optimo': {**empates(y, optimo, n), **precision_media(y, optimo, rng_ap)}}

    # P4 del original frente a cada benchmark, con los empates de los dos modelos rotos al azar en cada réplica
    benchmarks = [m for m in modelos if m != plan.ORIGINAL]
    mejor = veredicto['mejor_benchmark']
    rng = np.random.default_rng(SEED)
    crudos, ajustados, favor = [], [], []
    for _ in range(REPLICAS):
        original = al_azar(puntajes[plan.ORIGINAL], n, rng)
        ps, diferencias = [], []
        for b in benchmarks:
            mc = pruebas.mcnemar(y, original, al_azar(puntajes[b], n, rng))
            ps.append(mc['p'])
            diferencias.append(mc['solo_a'] - mc['solo_b'])
        holm = estadistica.ajustar_pvalores(np.array(ps), 'holm')
        i = benchmarks.index(mejor)
        crudos.append(ps[i])
        ajustados.append(float(holm[i]))
        favor.append(diferencias[i] > 0)
    crudos, ajustados, favor = np.array(crudos), np.array(ajustados), np.array(favor)
    registrado = veredicto['frente_al_mejor']['P4']
    cuantiles = lambda v: {'q025': float(np.quantile(v, 0.025)), 'mediana': float(np.median(v)), 'q975': float(np.quantile(v, 0.975)),   # noqa: E731
                           'minimo': float(v.min())}
    salida['p4'] = {'mejor_benchmark': mejor, 'comparaciones': len(benchmarks),
                    'protocolo': {'p': registrado['p'], 'p_holm': registrado.get('p_holm')},
                    'p_crudo': cuantiles(crudos), 'p_holm': cuantiles(ajustados),
                    'a_favor_del_original': float(favor.mean()),
                    'rechaza_a_favor': float(((ajustados < 0.05) & favor).mean()),
                    'rechaza_en_contra': float(((ajustados < 0.05) & ~favor).mean())}
    (CARPETA_RESULTADOS / 'empates_campana.json').write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding='utf-8')
    resumen = {m: {k: salida['modelos'][m][k] for k in ('valores_distintos', 'empatados', 'cupos', 'captura_protocolo',
                                                        'captura_esperada', 'captura_desviacion')} for m in (plan.ORIGINAL, mejor)}
    print(json.dumps({'modelos': resumen, 'optimo': salida['optimo'], 'p4': salida['p4']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
