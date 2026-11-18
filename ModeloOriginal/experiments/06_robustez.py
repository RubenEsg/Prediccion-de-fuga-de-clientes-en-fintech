"""Estudios de robustez del modelo original (protocolo, sección 9; guía, Cuadro 4).

Todos se hacen **dentro de la partición de entrenamiento**, con los cinco pliegues externos de la semilla 42 y los
hiperparámetros finales de cada modelo (sin nueva búsqueda): el conjunto de prueba no interviene. El modelo original se
compara con los dos mejores benchmarks de boosting (XGBoost y LightGBM), que reciben exactamente las mismas filas.

- ``sensibilidad``: se varía un hiperparámetro propio cada vez alrededor de la configuración final.
- ``ruido``: ruido gaussiano y valores atípicos en el objetivo de entrenamiento, y ruido en los predictores numéricos.
- ``muestra``: curva de aprendizaje con el 10 %, 25 %, 50 % y 100 % de las filas, con el tiempo de ajuste; y tiempo del
  modelo original al variar el número de columnas.
- ``ciudades``: cambio de distribución; se deja fuera una ciudad cada vez y se evalúa en ella.
- ``mezcla``: la misma curva de aprendizaje para el modelo original sin la mezcla (``n_componentes = 1``), con las mismas
  filas. Se añadió después de ver los resultados, para separar qué parte de la ventaja con pocas filas se debe a la clase
  latente; es un análisis exploratorio y no interviene en el veredicto.

Salidas: ``resultados/robustez/<estudio>.csv``. Uso: ``python experiments/06_robustez.py [estudio …]``.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import plan                                                       # noqa: E402
from src.config import CARPETA_RESULTADOS, K_EXT, SEED                     # noqa: E402
from src.optimizacion import construir_cv                                  # noqa: E402
from src.variantes import reajustar                                        # noqa: E402

MODELOS = ['bdl', 'xgboost', 'lightgbm']
SALIDA = CARPETA_RESULTADOS / 'robustez'


class Contexto:
    """Datos de entrenamiento, pliegues y configuración final de cada modelo."""

    def __init__(self) -> None:
        p = plan.problema()
        c = p['conjuntos']
        self.X = c.X_train.reset_index(drop=True)
        self.s = c.y_train_cont.to_numpy()
        self.umbral = p['etiqueta'].umbral
        self.y = {'regresion': self.s, 'clasificacion': (self.s > self.umbral).astype(int)}
        self.pliegues = {t: list(construir_cv(t, K_EXT, SEED).split(self.X, self.y[t])) for t in self.y}
        tabla = plan.registro().cargar()
        self.config = {}
        for tarea in self.y:
            for modelo in MODELOS:
                balanceo, optimizador = plan.combinacion(tarea, modelo)
                f = tabla[(tabla.tarea == tarea) & (tabla.modelo == modelo) & (tabla.balanceo == balanceo)
                          & (tabla.optimizador == optimizador) & (tabla.semilla == SEED)].iloc[0]
                h = f['hiperparametros']
                self.config[tarea, modelo] = (balanceo, json.loads(h) if isinstance(h, str) else dict(h))

    def medir(self, tarea, modelo, i_tr, i_te, X=None, s_tr=None, hiper=None) -> tuple[float, float]:
        """Ajusta en ``i_tr`` (con el objetivo de entrenamiento que se indique) y devuelve (métrica en ``i_te``, segundos)."""
        X = self.X if X is None else X
        s_tr = self.s[i_tr] if s_tr is None else s_tr
        balanceo, final = self.config[tarea, modelo]
        y_tr = s_tr if tarea == 'regresion' else (s_tr > self.umbral).astype(int)
        t0 = time.perf_counter()
        tuberia = reajustar(tarea, modelo, balanceo, {**final, **(hiper or {})}, X.iloc[i_tr], y_tr, SEED, puntaje=s_tr)
        segundos = time.perf_counter() - t0
        if tarea == 'regresion':
            return float(np.sqrt(np.mean((self.s[i_te] - tuberia.predict(X.iloc[i_te])) ** 2))), segundos
        return float(roc_auc_score(self.y['clasificacion'][i_te], tuberia.predict_proba(X.iloc[i_te])[:, 1])), segundos


def guardar(nombre: str, filas: list[dict]) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SALIDA / f'{nombre}.csv', index=False)


def sensibilidad(ctx: Contexto) -> None:
    rejilla = {'modelo__tasa': [0.05, 0.1, 0.3, 0.6, 1.0], 'modelo__profundidad': [1, 2, 3, 4, 6],
               'modelo__min_hoja': [5, 20, 50, 200], 'modelo__lambda_l2': [0.01, 0.1, 1.0, 10.0],
               'modelo__max_bins': [32, 64, 255, 1024], 'modelo__n_componentes': [1, 2, 3, 4, 5, 6, 8, 'auto'],
               'modelo__rondas_calentamiento': [10, 30, 100, 300]}
    filas = []
    for tarea in ('regresion', 'clasificacion'):
        final = ctx.config[tarea, 'bdl'][1]
        for parametro, valores in rejilla.items():
            for valor in valores:
                extra = {parametro: valor}
                if tarea == 'clasificacion':
                    extra['modelo__usar_puntaje'] = True                 # la sensibilidad es del modelo con puntaje
                metricas = [ctx.medir(tarea, 'bdl', a, b, hiper=extra)[0] for a, b in ctx.pliegues[tarea]]
                filas.append({'tarea': tarea, 'parametro': parametro.replace('modelo__', ''), 'valor': str(valor),
                              'es_el_final': str(final.get(parametro, POR_DEFECTO.get(parametro))) == str(valor),
                              'media': float(np.mean(metricas)), 'desv': float(np.std(metricas, ddof=1))})
                guardar('sensibilidad', filas)
                print(filas[-1], flush=True)


POR_DEFECTO = {"modelo__rondas_calentamiento": 100}


def ruido(ctx: Contexto) -> None:
    rng = np.random.default_rng(SEED)
    numericas = [c for c in ctx.X.columns if pd.api.types.is_numeric_dtype(ctx.X[c]) and not pd.api.types.is_bool_dtype(ctx.X[c])]
    desv = ctx.X[numericas].std()
    escenarios = [('sin contaminar', 0.0, None)]
    escenarios += [('ruido gaussiano en el objetivo', v, 'objetivo') for v in (0.005, 0.01, 0.02, 0.05)]
    escenarios += [('atípicos en el objetivo', v, 'atipicos') for v in (0.01, 0.05, 0.10)]
    escenarios += [('ruido en los predictores numéricos', v, 'predictores') for v in (0.1, 0.25, 0.5)]
    filas = []
    for nombre, nivel, tipo in escenarios:
        X = ctx.X
        if tipo == 'predictores':
            X = ctx.X.copy()
            X[numericas] = X[numericas] + rng.normal(size=(len(X), len(numericas))) * (nivel * desv.to_numpy())
        for tarea in ('regresion', 'clasificacion'):
            for modelo in MODELOS:
                metricas = []
                for a, b in ctx.pliegues[tarea]:
                    s_tr = ctx.s[a].copy()
                    r = np.random.default_rng(SEED + len(a))
                    if tipo == 'objetivo':
                        s_tr = s_tr + r.normal(0, nivel, len(a))
                    elif tipo == 'atipicos':
                        cuales = r.random(len(a)) < nivel
                        s_tr[cuales] = r.uniform(ctx.s.min(), ctx.s.max(), int(cuales.sum()))
                    metricas.append(ctx.medir(tarea, modelo, a, b, X=X, s_tr=s_tr)[0])
                filas.append({'escenario': nombre, 'nivel': nivel, 'tarea': tarea, 'modelo': modelo,
                              'media': float(np.mean(metricas)), 'desv': float(np.std(metricas, ddof=1))})
                guardar('ruido', filas)
                print(filas[-1], flush=True)


def muestra(ctx: Contexto) -> None:
    filas = []
    for fraccion in (0.10, 0.25, 0.50, 1.00):
        for tarea in ('regresion', 'clasificacion'):
            for modelo in MODELOS:
                metricas, tiempos = [], []
                for a, b in ctx.pliegues[tarea]:
                    sub = np.random.default_rng(SEED).choice(a, int(round(fraccion * len(a))), replace=False)
                    m, t = ctx.medir(tarea, modelo, np.sort(sub), b)
                    metricas.append(m)
                    tiempos.append(t)
                filas.append({'fraccion': fraccion, 'n': int(round(fraccion * len(ctx.pliegues[tarea][0][0]))), 'tarea': tarea, 'modelo': modelo,
                              'media': float(np.mean(metricas)), 'desv': float(np.std(metricas, ddof=1)), 'ajuste_s': float(np.mean(tiempos))})
                guardar('muestra', filas)
                print(filas[-1], flush=True)
    filas = []
    a, b = ctx.pliegues['regresion'][0]
    columnas = list(ctx.X.columns)
    for p_cols in (6, 12, 24, len(columnas)):
        for modelo in ('bdl', 'xgboost'):
            elegidas = columnas[:p_cols]
            _, t = ctx.medir('regresion', modelo, a, b, X=ctx.X[elegidas])
            filas.append({'columnas': p_cols, 'modelo': modelo, 'n': len(a), 'ajuste_s': t})
            guardar('columnas', filas)
            print(filas[-1], flush=True)


def ciudades(ctx: Contexto) -> None:
    filas = []
    for ciudad in sorted(ctx.X['location'].unique()):
        fuera = np.flatnonzero((ctx.X['location'] == ciudad).to_numpy())
        dentro = np.flatnonzero((ctx.X['location'] != ciudad).to_numpy())
        for tarea in ('regresion', 'clasificacion'):
            for modelo in MODELOS:
                m, _ = ctx.medir(tarea, modelo, dentro, fuera)
                filas.append({'ciudad': ciudad, 'n': len(fuera), 'tarea': tarea, 'modelo': modelo, 'metrica': m})
                guardar('ciudades', filas)
                print(filas[-1], flush=True)


def mezcla(ctx: Contexto) -> None:
    filas = []
    for fraccion in (0.10, 0.25, 0.50, 1.00):
        for tarea in ('regresion', 'clasificacion'):
            metricas = []
            for a, b in ctx.pliegues[tarea]:
                sub = np.random.default_rng(SEED).choice(a, int(round(fraccion * len(a))), replace=False)   # las filas de ``muestra``
                metricas.append(ctx.medir(tarea, 'bdl', np.sort(sub), b, hiper={'modelo__n_componentes': 1})[0])
            filas.append({'fraccion': fraccion, 'n': int(round(fraccion * len(ctx.pliegues[tarea][0][0]))), 'tarea': tarea,
                          'modelo': 'bdl_sin_mezcla', 'media': float(np.mean(metricas)), 'desv': float(np.std(metricas, ddof=1))})
            guardar('muestra_sin_mezcla', filas)
            print(filas[-1], flush=True)


ESTUDIOS = {'ciudades': ciudades, 'muestra': muestra, 'ruido': ruido, 'sensibilidad': sensibilidad, 'mezcla': mezcla}

if __name__ == '__main__':
    ctx = Contexto()
    for nombre in sys.argv[1:] or list(ESTUDIOS):
        t0 = time.perf_counter()
        ESTUDIOS[nombre](ctx)
        print(f'== {nombre}: {time.perf_counter() - t0:.0f} s', flush=True)
