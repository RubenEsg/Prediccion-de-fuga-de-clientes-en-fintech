"""¿Podía algún modelo cumplir la condición R3 en COFINFAD? Las cuatro pruebas del protocolo, para el predictor óptimo.

El predictor óptimo es el del techo teórico (``04b_techo.py``): conoce el índice real de cada cliente, que se
reconstruye con ``active_products``, y predice con la regla de Bayes. Ningún modelo puede aprenderlo, porque necesitaría
la variable excluida; es el mejor predictor posible **en esperanza** para quien no la recibe. En una muestra finita un
modelo puede quedar por delante de él por azar, así que lo que sigue no es una imposibilidad lógica: dice qué habría
obtenido, con este protocolo y estas muestras, el mejor predictor posible.

Las pruebas P1 y P2 usan los pliegues externos de la validación anidada (las mismas particiones y semillas que los diez
modelos, reconstruidas con ``construir_cv``); P3 y P4, el conjunto de prueba. En todas, el ajuste de Holm es sobre las
nueve comparaciones, como en el protocolo.

Es un **análisis de la causa**, posterior a los resultados y fuera del protocolo: no cambia el veredicto, lo explica.
``active_products`` se usa solo para este cálculo, como en el techo.

Salida: ``resultados/optimo.json``. Uso: ``python experiments/04c_optimo.py`` (después de 03, 04 y 04b).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import estadistica, plan, pruebas                                 # noqa: E402
from src.config import CARPETA_RESULTADOS, K_EXT                           # noqa: E402
from src.optimizacion import construir_cv                                  # noqa: E402

PRUEBAS = ('P1', 'P2', 'P3', 'P4')


def main() -> None:
    p = plan.problema()
    c = p['conjuntos']
    techo = json.loads((CARPETA_RESULTADOS / 'techo.json').read_text(encoding='utf-8'))
    veredicto = json.loads((CARPETA_RESULTADOS / 'veredicto.json').read_text(encoding='utf-8'))
    pred = pd.read_parquet(CARPETA_RESULTADOS / 'predicciones_prueba.parquet')
    delta, umbral = techo['delta'], techo['umbral']
    pesos = np.array([techo['pesos'][str(k)] for k in range(1, 6)])       # distribución de active_products en entrenamiento
    niveles = -delta * np.arange(5)                                       # desplazamiento con 1, 2, …, 5 productos

    def optimo_de(puntaje: np.ndarray, productos: np.ndarray) -> dict:
        indice = puntaje + delta * (productos - 1)                        # índice real: lo que ningún modelo observa
        return {'regresion': indice + pesos @ niveles,                    # media condicional del puntaje
                'clasificacion': ((indice[:, None] + niveles[None, :]) > umbral) @ pesos}   # P(puntaje > umbral | índice)

    y_cont, y_bin = c.y_test_cont.to_numpy(), p['y_test'].to_numpy()
    optimo = optimo_de(y_cont, p['df'].loc[c.X_test.index, 'active_products'].to_numpy())
    # el mismo predictor en entrenamiento, para evaluarlo en los pliegues externos de la validación anidada
    s_train = c.y_train_cont.to_numpy()
    optimo_train = optimo_de(s_train, p['df'].loc[c.X_train.index, 'active_products'].to_numpy())
    y_train = {'regresion': s_train, 'clasificacion': (s_train > p['etiqueta'].umbral).astype(int)}
    X_train = c.X_train.reset_index(drop=True)
    salida = {}

    for tarea in ('regresion', 'clasificacion'):
        mayor = tarea == 'clasificacion'
        signo = 1.0 if mayor else -1.0
        M = pd.read_csv(CARPETA_RESULTADOS / f'pliegues_{tarea}.csv', index_col=0)      # pliegues × modelos (04_pruebas_hipotesis.py)
        por_semilla = {}
        valores = []
        for fila in M.index:                                              # «semilla-pliegue», en el orden de la validación anidada
            semilla, pliegue = (int(x) for x in str(fila).split('-'))
            if semilla not in por_semilla:
                por_semilla[semilla] = list(construir_cv(tarea, K_EXT, semilla).split(X_train, y_train[tarea]))
            _, i_te = por_semilla[semilla][pliegue - 1]
            if mayor:
                valores.append(float(roc_auc_score(y_train[tarea][i_te], optimo_train[tarea][i_te])))
            else:
                valores.append(float(np.sqrt(np.mean((s_train[i_te] - optimo_train[tarea][i_te]) ** 2))))
        en_pliegues = np.array(valores)
        benchmarks = [m for m in plan.modelos(tarea) if m != plan.ORIGINAL]
        filas = []
        for b in benchmarks:
            otro_cv = M[b].to_numpy()
            w = pruebas.wilcoxon_pareado(en_pliegues, otro_cv)
            t = pruebas.t_corregida(en_pliegues, otro_cv)
            otro = pred[f'{tarea}__{b}'].to_numpy()
            fila = {'benchmark': b, 'P1': w['p'], 'P2': t['p'],
                    'P1_a_favor': bool(np.median(signo * (en_pliegues - otro_cv)) > 0), 'P2_a_favor': bool(signo * (en_pliegues - otro_cv).mean() > 0)}
            if mayor:                                                     # DeLong (P3) y McNemar con la campaña del 25 % (P4)
                dl = estadistica.delong(y_bin, otro, optimo[tarea])
                mc = pruebas.mcnemar(y_bin, pruebas.marcar_campana(optimo[tarea]), pruebas.marcar_campana(otro))
                fila.update({'diferencia': dl['diferencia'], 'P3': dl['p_bilateral'], 'P4': mc['p'],
                             'P3_a_favor': bool(dl['diferencia'] > 0), 'P4_a_favor': bool(mc['solo_a'] > mc['solo_b'])})
            else:                                                         # permutación pareada (P3) y bootstrap BCa de la diferencia de RMSE (P4)
                pe = pruebas.permutacion_pareada((y_cont - optimo[tarea]) ** 2, (y_cont - otro) ** 2)
                bo = pruebas.bootstrap_rmse_pareado(y_cont, optimo[tarea], otro)
                fila.update({'diferencia': bo['diferencia'], 'P3': pe['p'], 'P4': bo['p'],
                             'P3_a_favor': bool(pe['diferencia'] < 0), 'P4_a_favor': bool(bo['diferencia'] < 0)})
            filas.append(fila)
        en_prueba = float(roc_auc_score(y_bin, optimo[tarea])) if mayor else float(np.sqrt(np.mean((y_cont - optimo[tarea]) ** 2)))
        salida[tarea] = resumir(tarea, filas, veredicto, en_prueba, en_pliegues, M)

    (CARPETA_RESULTADOS / 'optimo.json').write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(salida, ensure_ascii=False, indent=1))


def resumir(tarea: str, filas: list[dict], veredicto: dict, metrica: float, en_pliegues: np.ndarray, M: pd.DataFrame) -> dict:
    """Ajusta por Holm dentro de cada prueba (nueve comparaciones, como en el protocolo) y se queda con el mejor benchmark."""
    t = pd.DataFrame(filas)
    for k in PRUEBAS:
        t[f'{k}_holm'] = estadistica.ajustar_pvalores(t[k].to_numpy(), 'holm')
    mejor = veredicto[tarea]['mejor_benchmark']
    f = t[t.benchmark == mejor].iloc[0]
    original = veredicto[tarea]['frente_al_mejor']
    del_optimo = {k: {'p': float(f[k]), 'p_holm': float(f[f'{k}_holm']), 'a_favor': bool(f[f'{k}_a_favor']),
                      'rechaza': bool(f[f'{k}_holm'] < 0.05 and f[f'{k}_a_favor'])} for k in PRUEBAS}
    return {'metrica_del_optimo': metrica, 'mejor_benchmark': mejor, 'diferencia_con_el_mejor': float(f['diferencia']),
            'comparaciones': len(t), 'pliegues': int(len(en_pliegues)),
            'media_del_optimo_en_pliegues': float(en_pliegues.mean()), 'desviacion_del_optimo_en_pliegues': float(en_pliegues.std(ddof=1)),
            'media_del_mejor_en_pliegues': float(M[mejor].mean()), 'media_del_original_en_pliegues': float(M[plan.ORIGINAL].mean()),
            'optimo': del_optimo,
            'original': {k: {'p': original[k]['p'], 'p_holm': original[k].get('p_holm'),
                             'rechaza': bool(veredicto[tarea]['rechazos_frente_al_mejor'].get(k, False))} for k in PRUEBAS if k in original},
            'rechazos_del_optimo': int(sum(x['rechaza'] for x in del_optimo.values())),
            'rechazos_del_original': int(sum(bool(x) for x in veredicto[tarea]['rechazos_frente_al_mejor'].values())),
            'pruebas_que_el_optimo_no_rechaza': [k for k in PRUEBAS if not del_optimo[k]['rechaza']]}


if __name__ == '__main__':
    main()
