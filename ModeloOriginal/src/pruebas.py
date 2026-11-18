"""Pruebas de hipótesis del protocolo (sección 6) que no estaban en ``estadistica.py``.

``estadistica.py`` (Entregable 3) ya trae Friedman con Iman-Davenport y Nemenyi, DeLong, el bootstrap BCa, la δ de
Cliff, la *d* de Cohen y el ajuste de Holm. Aquí se añaden las que pide la guía para datos de corte transversal:
Wilcoxon de rangos con signo, la *t* corregida de Nadeau y Bengio, la prueba de permutación pareada, McNemar y el
bootstrap pareado BCa de la diferencia de RMSE con su p-valor.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_pareado(a, b) -> dict:
    """Wilcoxon de rangos con signo sobre las diferencias pareadas ``a − b``.

    Parameters
    ----------
    a, b : array-like de forma (J,)
        Métrica por pliegue de dos modelos, en los mismos pliegues.

    Returns
    -------
    dict
        ``estadistico``, ``p`` (bilateral; exacto si no hay empates y J ≤ 50), ``n`` y ``a_favor`` (pliegues con
        ``a > b``).

    Notes
    -----
    Hipótesis nula: las diferencias son simétricas alrededor de cero. Supone diferencias independientes; los pliegues
    de una validación cruzada repetida comparten filas de entrenamiento, así que el p-valor es optimista y por eso el
    protocolo lo acompaña de la *t* corregida.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return {'estadistico': 0.0, 'p': 1.0, 'n': int(len(d)), 'a_favor': 0}
    r = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='auto')
    return {'estadistico': float(r.statistic), 'p': float(r.pvalue), 'n': int(len(d)), 'a_favor': int((d > 0).sum())}


def t_corregida(a, b, razon_prueba_entrenamiento: float = 0.25) -> dict:
    """Prueba *t* remuestreada con la corrección de varianza de Nadeau y Bengio (2003).

    Parameters
    ----------
    a, b : array-like de forma (J,)
        Métrica por pliegue de dos modelos en los mismos J pliegues (de una o varias repeticiones).
    razon_prueba_entrenamiento : float
        ``n_prueba / n_entrenamiento`` de cada pliegue: 1/4 con cinco pliegues.

    Returns
    -------
    dict
        ``t``, ``gl`` (J − 1), ``p`` (bilateral), ``media`` y ``desv`` de las diferencias ``a − b``.

    Notes
    -----
    Los J pliegues comparten filas de entrenamiento, así que sus diferencias no son independientes y la *t* corriente
    subestima la varianza. La corrección sustituye ``1/J`` por ``1/J + n_prueba/n_entrenamiento``:

        t = media(d) / sqrt((1/J + n_prueba/n_entrenamiento) · var(d)),   con J − 1 grados de libertad.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    J = len(d)
    media, var = float(d.mean()), float(d.var(ddof=1))
    if var <= 0:
        return {'t': float('inf') if media != 0 else 0.0, 'gl': J - 1, 'p': 0.0 if media != 0 else 1.0, 'media': media, 'desv': 0.0}
    t = media / np.sqrt((1.0 / J + razon_prueba_entrenamiento) * var)
    return {'t': float(t), 'gl': J - 1, 'p': float(2 * stats.t.sf(abs(t), J - 1)), 'media': media, 'desv': float(np.sqrt(var))}


def permutacion_pareada(perdida_a, perdida_b, n_perm: int = 10000, seed: int = 42) -> dict:
    """Prueba de permutación pareada (cambio de signo) sobre la diferencia media de pérdidas por fila.

    Parameters
    ----------
    perdida_a, perdida_b : array-like de forma (n,)
        Pérdida de cada fila de prueba con cada modelo (error cuadrático, absoluto…).
    n_perm : int
    seed : int

    Returns
    -------
    dict
        ``diferencia`` (media de ``perdida_a − perdida_b``; negativa si ``a`` es mejor), ``p`` (bilateral) y ``n_perm``.

    Notes
    -----
    Hipótesis nula: los dos modelos son intercambiables fila a fila, de modo que el signo de cada diferencia es
    arbitrario. Se cambia el signo de cada diferencia al azar y se cuenta cuántas veces la media permutada es al menos
    tan extrema como la observada; ``p = (cuenta + 1)/(n_perm + 1)``. No supone normalidad ni varianzas iguales.
    """
    d = np.asarray(perdida_a, dtype=float) - np.asarray(perdida_b, dtype=float)
    observado = abs(d.mean())
    rng = np.random.default_rng(seed)
    extremos = 0
    for inicio in range(0, n_perm, 500):                            # por bloques, para no gastar memoria
        bloque = min(500, n_perm - inicio)
        signos = rng.choice(np.array([-1.0, 1.0]), size=(bloque, len(d)))
        extremos += int((np.abs(signos @ d) / len(d) >= observado - 1e-15).sum())
    return {'diferencia': float(d.mean()), 'p': float((extremos + 1) / (n_perm + 1)), 'n_perm': int(n_perm)}


def mcnemar(y, pred_a, pred_b) -> dict:
    """McNemar exacto sobre los aciertos de dos clasificadores en las mismas filas.

    Parameters
    ----------
    y : array-like de forma (n,)
        Etiqueta real.
    pred_a, pred_b : array-like de forma (n,)
        Clase predicha por cada modelo.

    Returns
    -------
    dict
        ``solo_a`` (filas que solo acierta ``a``), ``solo_b``, ``p`` (binomial exacta, bilateral) y ``n``.

    Notes
    -----
    Hipótesis nula: los dos modelos se equivocan con la misma frecuencia. Solo informan las filas en las que uno
    acierta y el otro no; bajo la nula, ``solo_a`` sigue una binomial(``solo_a + solo_b``, 1/2).
    """
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    acierta_a, acierta_b = pred_a == y, pred_b == y
    solo_a, solo_b = int((acierta_a & ~acierta_b).sum()), int((~acierta_a & acierta_b).sum())
    p = 1.0 if solo_a + solo_b == 0 else float(stats.binomtest(solo_a, solo_a + solo_b, 0.5).pvalue)
    return {'solo_a': solo_a, 'solo_b': solo_b, 'p': p, 'n': int(len(y))}


def marcar_campana(puntaje, fraccion: float = 0.25) -> np.ndarray:
    """Marca como positivos los clientes de mayor puntaje hasta completar la fracción de la campaña.

    Los empates en el borde se resuelven por posición, igual para todos los modelos (orden estable).
    """
    puntaje = np.asarray(puntaje, dtype=float)
    n = int(round(fraccion * len(puntaje)))
    marcado = np.zeros(len(puntaje), dtype=int)
    marcado[np.argsort(-puntaje, kind='stable')[:n]] = 1
    return marcado


def bootstrap_rmse_pareado(y, pred_a, pred_b, n_boot: int = 2000, nivel: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap pareado BCa de la diferencia de RMSE entre dos modelos en las mismas filas.

    Parameters
    ----------
    y, pred_a, pred_b : array-like de forma (n,)
    n_boot : int
    nivel : float
        Nivel del intervalo de confianza.
    seed : int

    Returns
    -------
    dict
        ``diferencia`` (``RMSE_a − RMSE_b``; negativa si ``a`` es mejor), ``ic_bajo``, ``ic_alto`` (BCa) y ``p``
        (bilateral: el nivel al que el intervalo BCa toca el cero, obtenido de las réplicas corregidas).

    Notes
    -----
    Se remuestrean filas con reemplazo, las mismas para los dos modelos. El intervalo BCa (Efron, 1987) corrige el
    sesgo de la distribución bootstrap (``z0``, de la fracción de réplicas por debajo de la estimación) y su asimetría
    (``a``, de la influencia de cada fila, estimada con el jackknife, que aquí tiene forma cerrada porque el RMSE
    depende de la suma de errores cuadráticos). El p-valor es el doble de la cola corregida que queda del lado del cero.
    """
    y, pa, pb = (np.asarray(v, dtype=float) for v in (y, pred_a, pred_b))
    ea, eb = (y - pa) ** 2, (y - pb) ** 2
    n = len(y)
    estimacion = float(np.sqrt(ea.mean()) - np.sqrt(eb.mean()))
    rng = np.random.default_rng(seed)
    replicas = np.empty(n_boot)
    for i in range(n_boot):
        filas = rng.integers(0, n, n)
        replicas[i] = np.sqrt(ea[filas].mean()) - np.sqrt(eb[filas].mean())
    # jackknife en forma cerrada: quitar la fila i cambia cada suma de cuadrados en su propio error
    jack = np.sqrt((ea.sum() - ea) / (n - 1)) - np.sqrt((eb.sum() - eb) / (n - 1))
    centrado = jack.mean() - jack
    aceleracion = float((centrado ** 3).sum() / (6.0 * ((centrado ** 2).sum() ** 1.5 + 1e-300)))
    z0 = float(stats.norm.ppf(np.clip((replicas < estimacion).mean(), 1e-6, 1 - 1e-6)))

    def cuantil_bca(alfa: float) -> float:
        z = stats.norm.ppf(alfa)
        ajustado = stats.norm.cdf(z0 + (z0 + z) / (1 - aceleracion * (z0 + z)))
        return float(np.quantile(replicas, np.clip(ajustado, 0.0, 1.0)))

    alfa = (1 - nivel) / 2
    # p-valor: nivel α al que el extremo BCa del lado del cero toca el cero. Se despeja de la fórmula del cuantil:
    # si F es la fracción de réplicas a un lado del cero, z_F = z0 + (z0 + z_α)/(1 − a(z0 + z_α)) ⇒ z_α.
    fraccion = float(np.clip((replicas <= 0).mean() if estimacion > 0 else (replicas >= 0).mean(), 1e-6, 1 - 1e-6))
    w = stats.norm.ppf(fraccion if estimacion > 0 else 1 - fraccion) - z0
    z_alfa = w / (1 + aceleracion * w) - z0
    cola = stats.norm.cdf(z_alfa) if estimacion > 0 else stats.norm.sf(z_alfa)
    return {'diferencia': estimacion, 'ic_bajo': cuantil_bca(alfa), 'ic_alto': cuantil_bca(1 - alfa),
            'p': float(min(1.0, 2 * cola)), 'n_boot': int(n_boot)}
