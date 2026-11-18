"""Pruebas de hipótesis del protocolo (sección 6) y condiciones 1 a 3 del veredicto R3 (sección 7).

Para cada tarea arma la matriz de la métrica principal por pliegue externo (filas: los pliegues de todas las
repeticiones completas; columnas: los diez modelos) y compara el modelo original con cada benchmark:

- P1: Friedman con Iman-Davenport sobre la matriz (global) y Wilcoxon de rangos con signo por pares;
- P2: prueba *t* corregida de Nadeau-Bengio sobre las diferencias por pliegue;
- P3: en prueba, DeLong (clasificación) o permutación pareada de errores cuadráticos (regresión);
- P4: en prueba, McNemar con la campaña del 25 % (clasificación) o bootstrap pareado BCa del RMSE (regresión).

Los p-valores se ajustan por Holm dentro de cada prueba (nueve comparaciones). Tamaños del efecto: δ de Cliff y *d* de
Cohen entre los valores por pliegue, y *d* pareada.

Salidas en ``resultados/``: ``pliegues_<tarea>.csv`` (la matriz), ``pruebas_hipotesis.csv`` (una fila por comparación y
prueba) y ``veredicto.json``. Uso: ``python experiments/04_pruebas_hipotesis.py`` (después de 02 y 03).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import estadistica, plan, pruebas                                 # noqa: E402
from src.config import CARPETA_RESULTADOS, SEMILLAS_REPETICION             # noqa: E402

ALFA = 0.05
MAYOR_ES_MEJOR = {'clasificacion': True, 'regresion': False}
METRICA = {'clasificacion': 'AUC', 'regresion': 'RMSE'}


def matriz_de_pliegues(tabla: pd.DataFrame, tarea: str) -> tuple[pd.DataFrame, list[int]]:
    """Métrica principal por pliegue externo de los diez modelos, en las repeticiones que están completas."""
    bloques, completas = [], []
    for semilla in SEMILLAS_REPETICION:
        columnas = {}
        for modelo in plan.modelos(tarea):
            balanceo, optimizador = plan.combinacion(tarea, modelo)
            f = tabla[(tabla.tarea == tarea) & (tabla.modelo == modelo) & (tabla.balanceo == balanceo)
                      & (tabla.optimizador == optimizador) & (tabla.semilla == semilla) & (tabla.estado == 'ok')]
            if f.empty:
                break
            columnas[modelo] = plan.puntajes_por_pliegue(f.iloc[0])
        else:
            bloque = pd.DataFrame(columnas)
            bloque.index = [f'{semilla}-{i + 1}' for i in range(len(bloque))]
            bloques.append(bloque)
            completas.append(semilla)
    if not bloques:
        raise SystemExit(f'{tarea}: ninguna repetición está completa')
    return pd.concat(bloques), completas


def main() -> None:
    tabla = plan.registro().cargar()
    p = plan.problema()
    y_bin, y_cont = p['y_test'].to_numpy(), p['conjuntos'].y_test_cont.to_numpy()
    ruta_pred = CARPETA_RESULTADOS / 'predicciones_prueba.parquet'
    pred = pd.read_parquet(ruta_pred) if ruta_pred.exists() else None
    filas, veredicto = [], {}

    for tarea in ('regresion', 'clasificacion'):
        M, completas = matriz_de_pliegues(tabla, tarea)
        M.to_csv(CARPETA_RESULTADOS / f'pliegues_{tarea}.csv')
        mayor = MAYOR_ES_MEJOR[tarea]
        signo = 1.0 if mayor else -1.0                              # ventaja = signo · (original − benchmark)
        medias = M.mean()
        benchmarks = [m for m in M.columns if m != plan.ORIGINAL]
        mejor = (medias[benchmarks].idxmax() if mayor else medias[benchmarks].idxmin())
        fr = estadistica.friedman(M, mayor_es_mejor=mayor, alfa=ALFA)
        original = M[plan.ORIGINAL].to_numpy()

        crudos = {k: [] for k in ('P1', 'P2', 'P3', 'P4')}
        detalle = {}
        for b in benchmarks:
            otro = M[b].to_numpy()
            w = pruebas.wilcoxon_pareado(original, otro)
            t = pruebas.t_corregida(original, otro)
            d = {'ventaja_media': float(signo * (original - otro).mean()),
                 'pliegues_a_favor': int((signo * (original - otro) > 0).sum()), 'pliegues': int(len(M)),
                 'P1': {'estadistico': w['estadistico'], 'p': w['p'], 'a_favor': bool(np.median(signo * (original - otro)) > 0)},
                 'P2': {'estadistico': t['t'], 'p': t['p'], 'gl': t['gl'], 'a_favor': bool(signo * (original - otro).mean() > 0)},
                 'cliff': signo * estadistica.delta_cliff(original, otro)['delta'],
                 'cohen': signo * estadistica.d_cohen(original, otro)['d'],
                 'cohen_pareada': signo * estadistica.d_cohen(original - otro)['d']}
            if pred is not None:
                a, c = pred[f'{tarea}__{plan.ORIGINAL}'].to_numpy(), pred[f'{tarea}__{b}'].to_numpy()
                if tarea == 'clasificacion':
                    dl = estadistica.delong(y_bin, c, a)                # diferencia = AUC original − AUC benchmark
                    mc = pruebas.mcnemar(y_bin, pruebas.marcar_campana(a), pruebas.marcar_campana(c))
                    d['P3'] = {'estadistico': dl['z'], 'p': dl['p_bilateral'], 'diferencia': dl['diferencia'],
                               'ic': [dl['ic_inf'], dl['ic_sup']], 'a_favor': bool(dl['diferencia'] > 0)}
                    d['P4'] = {'estadistico': mc['solo_a'] - mc['solo_b'], 'p': mc['p'], 'solo_original': mc['solo_a'],
                               'solo_benchmark': mc['solo_b'], 'a_favor': bool(mc['solo_a'] > mc['solo_b'])}
                else:
                    pe = pruebas.permutacion_pareada((y_cont - a) ** 2, (y_cont - c) ** 2)
                    bo = pruebas.bootstrap_rmse_pareado(y_cont, a, c)
                    d['P3'] = {'estadistico': pe['diferencia'], 'p': pe['p'], 'a_favor': bool(pe['diferencia'] < 0)}
                    d['P4'] = {'estadistico': bo['diferencia'], 'p': bo['p'], 'ic': [bo['ic_bajo'], bo['ic_alto']],
                               'a_favor': bool(bo['diferencia'] < 0)}
            for k in crudos:
                crudos[k].append(d[k]['p'] if k in d else np.nan)
            detalle[b] = d
        for k, ps in crudos.items():                                   # Holm dentro de cada prueba
            ps = np.asarray(ps, dtype=float)
            if np.isnan(ps).all():
                continue
            ajustados = estadistica.ajustar_pvalores(ps, 'holm')
            for b, pa in zip(benchmarks, ajustados):
                detalle[b][k]['p_holm'] = float(pa)

        nombres = {'P1': 'Wilcoxon de rangos con signo (tras Friedman)', 'P2': 't corregida de Nadeau-Bengio',
                   'P3': 'DeLong en prueba' if tarea == 'clasificacion' else 'Permutación pareada en prueba',
                   'P4': 'McNemar en prueba (campaña 25 %)' if tarea == 'clasificacion' else 'Bootstrap pareado BCa en prueba'}
        for b in benchmarks:
            for k in ('P1', 'P2', 'P3', 'P4'):
                if k not in detalle[b]:
                    continue
                r = detalle[b][k]
                filas.append({'tarea': tarea, 'benchmark': b, 'es_el_mejor_benchmark': b == mejor, 'prueba': k, 'nombre': nombres[k],
                              'estadistico': r['estadistico'], 'p_crudo': r['p'], 'p_holm': r.get('p_holm'),
                              'ventaja_media_cv': detalle[b]['ventaja_media'], 'pliegues_a_favor': detalle[b]['pliegues_a_favor'],
                              'cliff': detalle[b]['cliff'], 'cohen': detalle[b]['cohen'], 'cohen_pareada': detalle[b]['cohen_pareada'],
                              'rechaza': bool(r.get('p_holm', 1.0) < ALFA), 'a_favor_del_original': bool(r['a_favor'])})

        # un rechazo solo cuenta para la condición 2 si la diferencia va a favor del original
        dm = detalle[mejor]
        global_rechaza = fr['p_iman_davenport'] < ALFA
        rechazos = {k: bool(dm[k].get('p_holm', 1.0) < ALFA and dm[k]['a_favor'] and (k != 'P1' or global_rechaza))
                    for k in ('P1', 'P2', 'P3', 'P4') if k in dm}
        c1 = bool(signo * (medias[plan.ORIGINAL] - medias[mejor]) > 0)
        c2 = bool(c1 and sum(rechazos.values()) >= 3)
        c3 = bool(dm['cliff'] >= 0.147 or dm['cohen'] >= 0.2)           # con signo: el efecto tiene que ir a favor del original
        veredicto[tarea] = {
            'metrica': METRICA[tarea], 'repeticiones': completas, 'pliegues': int(len(M)),
            'medias': {m: float(v) for m, v in medias.items()}, 'desviaciones': {m: float(v) for m, v in M.std(ddof=1).items()},
            'mejor_benchmark': mejor, 'ventaja_sobre_el_mejor': float(signo * (medias[plan.ORIGINAL] - medias[mejor])),
            'friedman': {'chi2': fr['chi2'], 'p': fr['p'], 'F_iman_davenport': fr['F'], 'p_iman_davenport': fr['p_iman_davenport'],
                         'rangos_medios': {m: float(v) for m, v in fr['rangos_medios'].items()}, 'diferencia_critica_nemenyi': fr['dc'],
                         'p_nemenyi_original_vs_mejor': float(fr['p_nemenyi'].loc[plan.ORIGINAL, mejor])},
            'frente_al_mejor': dm, 'rechazos_frente_al_mejor': rechazos,
            'condicion_1_media_mejor': c1, 'condicion_2_tres_pruebas': c2, 'condicion_3_efecto': c3,
            'comparaciones_por_prueba': len(benchmarks)}
        print(f'\n== {tarea} ({METRICA[tarea]}), {len(M)} pliegues de las repeticiones {completas}')
        print(pd.DataFrame({'media': medias, 'desv': M.std(ddof=1), 'rango': fr['rangos_medios']}).sort_values('rango').round(6).to_string())
        print(f'Friedman: chi2 = {fr["chi2"]:.2f}, p = {fr["p"]:.2e}; Iman-Davenport F = {fr["F"]:.2f}, p = {fr["p_iman_davenport"]:.2e}; DC de Nemenyi = {fr["dc"]:.2f}')
        print(f'mejor benchmark: {mejor}; ventaja del original = {veredicto[tarea]["ventaja_sobre_el_mejor"]:+.6f}; a favor en {dm["pliegues_a_favor"]}/{len(M)} pliegues')
        for k in ('P1', 'P2', 'P3', 'P4'):
            if k in dm:
                print(f'   {k} {nombres[k]}: estadístico {dm[k]["estadistico"]:.5g}, p = {dm[k]["p"]:.4g}, p Holm = {dm[k].get("p_holm", float("nan")):.4g}')
        print(f'   δ de Cliff = {dm["cliff"]:+.3f}, d de Cohen = {dm["cohen"]:+.3f}, d pareada = {dm["cohen_pareada"]:+.3f}')
        print(f'   condición 1 (media mejor): {c1} | condición 2 (≥ 3 pruebas): {c2} {rechazos} | condición 3 (efecto): {c3}')

    pd.DataFrame(filas).to_csv(CARPETA_RESULTADOS / 'pruebas_hipotesis.csv', index=False)
    (CARPETA_RESULTADOS / 'veredicto.json').write_text(json.dumps(veredicto, ensure_ascii=False, indent=2, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
