"""Referencias homogéneas para dos comparaciones del informe: el error absoluto y los conjuntos de predicción.

Dos de las ventajas que el informe atribuye al modelo original comparan cosas distintas:

- **Error absoluto.** El original predice la mediana de su distribución predictiva y el benchmark, la media. La mediana
  minimiza el error absoluto y la media no, así que parte de la diferencia es de receta. Aquí se le da al benchmark la
  corrección equivalente que no necesita ningún modelo nuevo: sumar a su pronóstico la mediana de sus residuos fuera de
  pliegue, estimada en entrenamiento.
- **Conjuntos de predicción.** El del original es un conjunto por densidad (una unión de tramos); el del benchmark, en
  la hipótesis H3 del protocolo, es un intervalo simétrico. Aquí se construye para el benchmark el mismo tipo de
  conjunto: una mezcla ajustada a la mitad de sus residuos fuera de pliegue (con las funciones del modelo original,
  ``elegir_componentes`` y ``ajustar_mezcla``) y el umbral de densidad conforme calibrado con la otra mitad.

Lo que queda de cada ventaja después de igualar la receta es lo atribuible al modelo. Es un **análisis exploratorio,
posterior a los resultados** y fuera del protocolo: H3 se evalúa tal como se registró.

Salida: ``resultados/referencias_homogeneas.json``. Uso: ``python experiments/04e_referencias_homogeneas.py``
(después de 03 y 04).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

from src import plan                                                       # noqa: E402
from src.config import CARPETA_RESULTADOS, SEED                            # noqa: E402
from src.original_model import ajustar_mezcla, densidad_mezcla, elegir_componentes   # noqa: E402
from src.registro import resolver                                          # noqa: E402

COBERTURAS = (0.80, 0.90, 0.95)
E3 = plan.RAIZ.parent / 'Entrega3' / 'resultados'


def fuera_de_pliegue(tarea: str, modelo: str, indice_train) -> np.ndarray:
    """Predicción fuera de pliegue del modelo sobre entrenamiento (cada fila, con un modelo que no la vio)."""
    balanceo, optimizador = plan.combinacion(tarea, modelo)
    if modelo in plan.BENCHMARKS_E3[tarea]:
        oof = pd.read_parquet(E3 / 'complementos' / 'fuera_de_pliegue.parquet')
        return oof.loc[indice_train, f'{tarea}__{modelo}__{balanceo}__{optimizador}__fuera_de_pliegue'].to_numpy()
    tabla = plan.registro().cargar()
    fila = tabla[(tabla.tarea == tarea) & (tabla.modelo == modelo) & (tabla.semilla == SEED)].iloc[0]
    return np.load(resolver(fila['ruta_predicciones']))


def largo_del_conjunto(pi, c, sigma: float, nivel: float, residuos: np.ndarray, puntos_por_sigma: int = 40) -> tuple[float, int]:
    """Largo total y número de tramos del conjunto {r : densidad(r) ≥ nivel}, como ``RegresorBDL.conjunto_residual``."""
    minimo, maximo = min(c.min() - 10 * sigma, residuos.min()), max(c.max() + 10 * sigma, residuos.max())
    if nivel <= 0.0:
        return float(maximo - minimo), 1
    rejilla = np.linspace(minimo, maximo, int(min(max((maximo - minimo) / sigma * puntos_por_sigma, 200), 400000)))
    dentro = densidad_mezcla(rejilla, pi, c, sigma) >= nivel
    cambio = np.diff(np.concatenate(([0], dentro.astype(np.int8), [0])))
    inicio, fin = rejilla[np.flatnonzero(cambio == 1)], rejilla[np.flatnonzero(cambio == -1) - 1]
    return float((fin - inicio).sum()), int(len(inicio))


def main() -> None:
    p = plan.problema()
    c = p['conjuntos']
    veredicto = json.loads((CARPETA_RESULTADOS / 'veredicto.json').read_text(encoding='utf-8'))
    mejor = veredicto['regresion']['mejor_benchmark']
    pred = pd.read_parquet(CARPETA_RESULTADOS / 'predicciones_prueba.parquet')
    y_train, y_test = c.y_train_cont.to_numpy(), c.y_test_cont.to_numpy()
    residuos = y_train - fuera_de_pliegue('regresion', mejor, c.X_train.index)
    en_prueba = pred[f'regresion__{mejor}'].to_numpy()
    metricas = pd.read_csv(CARPETA_RESULTADOS / 'metricas_prueba.csv')
    original = metricas[(metricas.tarea == 'regresion') & (metricas.modelo == plan.ORIGINAL)].iloc[0]

    # 1. error absoluto: al benchmark se le suma la mediana de sus residuos fuera de pliegue
    desplazamiento = float(np.median(residuos))
    mae = {'benchmark': mejor, 'desplazamiento': desplazamiento,
           'benchmark_media': float(np.mean(np.abs(y_test - en_prueba))),
           'benchmark_con_mediana': float(np.mean(np.abs(y_test - (en_prueba + desplazamiento)))),
           'original_media': float(original['mae']), 'original_mediana': float(original['mae_mediana'])}

    # 2. conjunto por densidad para el benchmark: mezcla con una mitad de los residuos, umbral con la otra
    orden = np.random.default_rng(SEED).permutation(len(residuos))
    ajuste, calibracion = residuos[orden[:len(orden) // 2]], residuos[orden[len(orden) // 2:]]
    sigma_min = 1e-6 * float(np.std(y_train))
    K, tabla_bic = elegir_componentes(ajuste, 8, sigma_min)
    pi, centros, sigma, _ = ajustar_mezcla(ajuste, K, sigma_min)
    densidad = np.sort(densidad_mezcla(calibracion, pi, centros, sigma))
    incertidumbre = pd.read_csv(CARPETA_RESULTADOS / 'incertidumbre_prueba.csv')
    conjuntos = {}
    for cobertura in COBERTURAS:
        k = int(np.floor((1.0 - cobertura) * (len(densidad) + 1)))
        nivel = 0.0 if k < 1 else float(densidad[k - 1])
        largo, tramos = largo_del_conjunto(pi, centros, sigma, nivel, calibracion)
        del_original = incertidumbre[(incertidumbre.modelo == plan.ORIGINAL) & (incertidumbre.cobertura_nominal == cobertura)
                                     & (incertidumbre.metodo == 'conjunto de predicción del modelo')].iloc[0]
        del_intervalo = incertidumbre[(incertidumbre.modelo == mejor) & (incertidumbre.cobertura_nominal == cobertura)
                                      & (incertidumbre.metodo == 'intervalo conforme (pronóstico ± q)')].iloc[0]
        conjuntos[f'{cobertura:.2f}'] = {
            'benchmark_conjunto': {'largo': largo, 'tramos': tramos,
                                   'cobertura': float((densidad_mezcla(y_test - en_prueba, pi, centros, sigma) >= nivel).mean())},
            'benchmark_intervalo': {'largo': float(del_intervalo['largo']), 'cobertura': float(del_intervalo['cobertura'])},
            'original_conjunto': {'largo': float(del_original['largo']), 'tramos': int(del_original['tramos']),
                                  'cobertura': float(del_original['cobertura'])}}
    salida = {'benchmark': mejor, 'mae': mae,
              'conjuntos': {'clases_de_la_mezcla_del_benchmark': int(K), 'dispersion': float(sigma), 'centros': [float(x) for x in centros],
                            'pesos': [float(x) for x in pi], 'filas_de_ajuste': int(len(ajuste)), 'filas_de_calibracion': int(len(calibracion)),
                            'por_cobertura': conjuntos}}
    (CARPETA_RESULTADOS / 'referencias_homogeneas.json').write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(salida, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
