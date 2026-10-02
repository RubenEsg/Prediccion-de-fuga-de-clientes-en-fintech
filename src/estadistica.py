"""Pruebas estadísticas para comparar dos modelos sobre las mismas filas (guía §6).

Clasificación
-------------
:func:`delong` compara AUC correlacionadas (los dos modelos puntúan las mismas filas) con el
contraste no paramétrico de DeLong, DeLong y Clarke-Pearson (1988), en la formulación rápida
de Sun y Xu (2014). :func:`bootstrap_auc` da intervalos bootstrap estratificados por clase,
percentil y BCa, para el AUC de un modelo o para la diferencia entre dos.

Regresión
---------
:func:`diebold_mariano` contrasta la igualdad de precisión de dos pronósticos con la
corrección de Harvey, Leybourne y Newbold (1997) para muestras finitas.
:func:`bootstrap_estacionario` repite el contraste sin suponer normalidad, con el bootstrap
estacionario de Politis y Romano (1994) y el bloque de Politis y White (2004).
:func:`bootstrap_regresion` da intervalos percentil y BCa para RMSE, MAE y R².

Tamaños del efecto
------------------
:func:`delta_cliff` (no paramétrico) y :func:`d_cohen` (pareada o de dos muestras).

Convención
----------
``a`` es siempre el modelo de referencia y ``b`` el candidato. Las diferencias se expresan
de forma que **un valor positivo favorece al candidato**: ``AUC_b − AUC_a`` en métricas de
mayor-es-mejor y ``pérdida_a − pérdida_b`` en pérdidas y errores.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .config import SEED

NIVEL: float = 0.95
"""Nivel de confianza por defecto de todos los intervalos."""

N_BOOT: int = 2000
"""Réplicas bootstrap por defecto. Con 2.000 el error de Monte Carlo de un percentil del 2,5 %
es de unas pocas milésimas de la desviación bootstrap, suficiente para BCa."""


# -------------------------------------------------------------------------- utilidades
def _binaria(y) -> np.ndarray:
    """Valida y devuelve una etiqueta 0/1 que contiene las dos clases."""
    y = np.asarray(y).ravel()
    valores = np.unique(y)
    if len(valores) != 2 or not np.isin(valores, [0, 1]).all():
        raise ValueError('y debe ser binaria (0/1) y contener las dos clases')
    return y.astype(int)


def _vector(x, n: int | None = None, nombre: str = 'x') -> np.ndarray:
    """Convierte a vector de flotantes y comprueba la longitud."""
    v = np.asarray(x, dtype=float).ravel()
    if n is not None and len(v) != n:
        raise ValueError(f'{nombre} tiene {len(v)} valores y se esperaban {n}')
    if not np.isfinite(v).all():
        raise ValueError(f'{nombre} contiene valores no finitos')
    return v


def _magnitud(valor: float, cortes: tuple[float, float, float]) -> str:
    """Etiqueta cualitativa de un tamaño del efecto según tres cortes crecientes."""
    v = abs(valor)
    if v < cortes[0]:
        return 'despreciable'
    if v < cortes[1]:
        return 'pequeño'
    if v < cortes[2]:
        return 'mediano'
    return 'grande'


# --------------------------------------------------------------------------- DeLong
def colocaciones(y, puntaje) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC y colocaciones (*placement values*) de DeLong de un puntaje.

    Parameters
    ----------
    y : array-like
        Etiqueta 0/1.
    puntaje : array-like
        Probabilidad o función de decisión; solo importa el orden.

    Returns
    -------
    auc : float
    v_pos : ndarray, forma (m,)
        Para cada positivo, fracción de negativos con puntaje menor (un empate cuenta 1/2).
    v_neg : ndarray, forma (n,)
        Para cada negativo, fracción de positivos con puntaje mayor (un empate cuenta 1/2).

    Notes
    -----
    Con rangos medios, ``rango_en_el_total − rango_entre_positivos`` es exactamente el número
    de negativos por debajo de cada positivo más la mitad de los empatados (Sun y Xu, 2014),
    lo que evita el producto m × n de la definición. ``AUC = media(v_pos) = media(v_neg)``.
    """
    y = _binaria(y)
    s = _vector(puntaje, len(y), 'puntaje')
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    r_total = stats.rankdata(np.concatenate([pos, neg]))
    v_pos = (r_total[:m] - stats.rankdata(pos)) / n
    v_neg = 1.0 - (r_total[m:] - stats.rankdata(neg)) / m
    return float(v_pos.mean()), v_pos, v_neg


def delong(y, puntaje_a, puntaje_b=None, nivel: float = NIVEL) -> dict:
    """Intervalo del AUC de un modelo o contraste de DeLong entre dos modelos.

    Parameters
    ----------
    y : array-like
        Etiqueta 0/1 del conjunto de evaluación.
    puntaje_a : array-like
        Puntaje del modelo de referencia.
    puntaje_b : array-like, optional
        Puntaje del candidato sobre las **mismas filas**. Si se omite, solo se estima el AUC
        de ``a`` con su error estándar de DeLong.
    nivel : float
        Nivel de confianza del intervalo.

    Returns
    -------
    dict
        Con un modelo: ``auc``, ``ee``, ``ic_inf``, ``ic_sup``. Con dos: ``auc_a``, ``auc_b``,
        ``diferencia`` (= ``auc_b − auc_a``), ``ee`` de la diferencia, ``z``, ``p_bilateral``,
        ``p_unilateral`` (H1: ``auc_b > auc_a``), ``ic_inf`` e ``ic_sup`` de la diferencia y
        ``correlacion`` entre las dos AUC estimadas. Siempre ``n_pos``, ``n_neg`` y ``nivel``.

    Notes
    -----
    La varianza de cada AUC y su covarianza salen de las varianzas muestrales de las
    colocaciones: ``S = S10/m + S01/n``. La covarianza es la que distingue este contraste de
    comparar dos intervalos por separado: dos modelos evaluados sobre las mismas filas tienen
    AUC positivamente correlacionadas, y eso estrecha el error de la diferencia.
    """
    y = _binaria(y)
    z_crit = float(stats.norm.ppf(0.5 + nivel / 2))
    auc_a, pa, na = colocaciones(y, puntaje_a)
    m, n = len(pa), len(na)
    base = {'n_pos': m, 'n_neg': n, 'nivel': nivel}
    if puntaje_b is None:
        ee = float(np.sqrt(np.var(pa, ddof=1) / m + np.var(na, ddof=1) / n))
        return {'auc': auc_a, 'ee': ee, 'ic_inf': max(0.0, auc_a - z_crit * ee),
                'ic_sup': min(1.0, auc_a + z_crit * ee), **base}
    auc_b, pb, nb = colocaciones(y, puntaje_b)
    S = np.cov(np.vstack([pa, pb])) / m + np.cov(np.vstack([na, nb])) / n
    dif = auc_b - auc_a
    ee = float(np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 0.0)))
    if ee > 0:
        z = dif / ee
        p2, p1 = float(2 * stats.norm.sf(abs(z))), float(stats.norm.sf(z))
    else:                                      # puntajes con el mismo orden: no hay diferencia
        z, p2, p1 = 0.0, 1.0, 0.5
    den = np.sqrt(S[0, 0] * S[1, 1])
    return {'auc_a': auc_a, 'auc_b': auc_b, 'diferencia': dif, 'ee': ee, 'z': float(z),
            'p_bilateral': p2, 'p_unilateral': p1, 'ic_inf': dif - z_crit * ee,
            'ic_sup': dif + z_crit * ee, 'correlacion': float(S[0, 1] / den) if den > 0 else 1.0,
            **base}


def jackknife_auc(y, puntaje) -> np.ndarray:
    """AUC al dejar fuera cada fila, en el orden original (forma cerrada, sin reajustes).

    Parameters
    ----------
    y, puntaje : array-like

    Returns
    -------
    ndarray, forma (len(y),)

    Notes
    -----
    El AUC es la media de una comparación por pares, así que quitar el positivo *i* resta su
    colocación: ``AUC_(i) = (m·AUC − v_pos_i)/(m − 1)``; quitar el negativo *j* da
    ``(n·AUC − v_neg_j)/(n − 1)``. Es lo que necesita la aceleración de BCa sin recalcular
    el AUC 10.000 veces.
    """
    y = _binaria(y)
    auc, v_pos, v_neg = colocaciones(y, puntaje)
    m, n = len(v_pos), len(v_neg)
    salida = np.empty(len(y))
    salida[y == 1] = (m * auc - v_pos) / (m - 1)
    salida[y == 0] = (n * auc - v_neg) / (n - 1)
    return salida


# ------------------------------------------------------------------------ bootstrap
def intervalo_bca(estimacion: float, replicas, jackknife, nivel: float = NIVEL) -> tuple[float, float]:
    """Intervalo bootstrap BCa (corregido por sesgo y acelerado).

    Parameters
    ----------
    estimacion : float
        Estadístico sobre la muestra original.
    replicas : array-like
        Estadístico en cada réplica bootstrap.
    jackknife : array-like
        Estadístico al dejar fuera cada observación (para la aceleración).
    nivel : float

    Returns
    -------
    (inferior, superior) : tuple of float

    Notes
    -----
    Efron y Tibshirani (1993, §14.3): ``z0 = Φ⁻¹(P*(θ* < θ̂))`` corrige el sesgo de mediana
    y ``a = Σu³ / (6 (Σu²)^{3/2})``, con ``u = media(θ_(·)) − θ_(i)``, la asimetría. Los
    percentiles ``Φ(z0 + (z0 + z_α)/(1 − a(z0 + z_α)))`` sustituyen a α y 1 − α. Con
    ``z0 = a = 0`` coincide con el intervalo percentil.
    """
    r = np.asarray(replicas, dtype=float)
    r = r[np.isfinite(r)]
    B = len(r)
    prop = (np.sum(r < estimacion) + 0.5 * np.sum(r == estimacion)) / B
    z0 = stats.norm.ppf(np.clip(prop, 1 / (2 * B), 1 - 1 / (2 * B)))
    u = np.mean(jackknife) - np.asarray(jackknife, dtype=float)
    den = 6.0 * np.sum(u ** 2) ** 1.5
    a = float(np.sum(u ** 3) / den) if den > 0 else 0.0
    alfa = (1 - nivel) / 2
    zs = stats.norm.ppf([alfa, 1 - alfa])
    cuantiles = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    return float(np.quantile(r, cuantiles[0])), float(np.quantile(r, cuantiles[1]))


def _auc_rangos(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC como estadístico U de Mann-Whitney normalizado (empates a 1/2)."""
    m, n = len(pos), len(neg)
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:m].sum() - m * (m + 1) / 2) / (m * n))


def bootstrap_auc(y, puntaje_a, puntaje_b=None, n_boot: int = N_BOOT, nivel: float = NIVEL,
                  seed: int = SEED) -> dict:
    """Bootstrap estratificado y pareado del AUC o de la diferencia de AUC.

    Parameters
    ----------
    y : array-like
        Etiqueta 0/1.
    puntaje_a : array-like
        Modelo de referencia.
    puntaje_b : array-like, optional
        Candidato; si se da, el estadístico es ``AUC_b − AUC_a`` sobre la misma réplica.
    n_boot : int
    nivel : float
    seed : int

    Returns
    -------
    dict
        ``estimacion``, ``ee_bootstrap``, ``ic_percentil``, ``ic_bca``, ``replicas`` (para
        graficar), ``n_boot``, ``nivel`` y, con dos modelos, ``prop_no_mejora``: fracción de
        réplicas en las que el candidato no supera a la referencia.

    Notes
    -----
    Se remuestrean positivos y negativos por separado (bootstrap estratificado, como
    ``pROC`` por defecto): cada réplica conserva la prevalencia de la prueba y el AUC
    siempre está definido. Las dos AUC de una réplica usan **las mismas filas**, de modo que
    la correlación entre modelos se conserva igual que en DeLong.
    """
    y = _binaria(y)
    sa = _vector(puntaje_a, len(y), 'puntaje_a')
    sb = None if puntaje_b is None else _vector(puntaje_b, len(y), 'puntaje_b')
    i_pos, i_neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    def estadistico(ip: np.ndarray, ineg: np.ndarray) -> float:
        """AUC de ``a`` en las filas dadas o, si hay ``b``, la diferencia AUC(b) − AUC(a)."""
        auc_a = _auc_rangos(sa[ip], sa[ineg])
        return auc_a if sb is None else _auc_rangos(sb[ip], sb[ineg]) - auc_a

    rng = np.random.default_rng(seed)
    estimacion = estadistico(i_pos, i_neg)
    replicas = np.array([estadistico(rng.choice(i_pos, len(i_pos)), rng.choice(i_neg, len(i_neg)))
                         for _ in range(n_boot)])
    jk = jackknife_auc(y, sa) if sb is None else jackknife_auc(y, sb) - jackknife_auc(y, sa)
    alfa = (1 - nivel) / 2
    salida = {'estimacion': estimacion, 'ee_bootstrap': float(replicas.std(ddof=1)),
              'ic_percentil': tuple(float(q) for q in np.quantile(replicas, [alfa, 1 - alfa])),
              'ic_bca': intervalo_bca(estimacion, replicas, jk, nivel),
              'replicas': replicas, 'n_boot': n_boot, 'nivel': nivel}
    if sb is not None:
        salida['prop_no_mejora'] = float(np.mean(replicas <= 0))
    return salida


def _metricas_regresion(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    e = y - pred
    sst = np.sum((y - y.mean()) ** 2)
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
            'r2': float(1 - np.sum(e ** 2) / sst)}


def _jackknife_regresion(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """RMSE, MAE y R² dejando fuera cada fila (forma cerrada)."""
    n = len(y)
    e2, ea = (y - pred) ** 2, np.abs(y - pred)
    sse, sae = e2.sum(), ea.sum()
    sst = np.sum((y - y.mean()) ** 2)
    sst_i = sst - n / (n - 1) * (y - y.mean()) ** 2
    return {'rmse': np.sqrt((sse - e2) / (n - 1)), 'mae': (sae - ea) / (n - 1),
            'r2': 1 - (sse - e2) / sst_i}


def bootstrap_regresion(y, pred_a, pred_b=None, n_boot: int = N_BOOT, nivel: float = NIVEL,
                        seed: int = SEED) -> dict[str, dict]:
    """Bootstrap pareado de RMSE, MAE y R² (o de sus diferencias entre dos modelos).

    Parameters
    ----------
    y : array-like
        Objetivo continuo.
    pred_a : array-like
        Pronóstico de referencia.
    pred_b : array-like, optional
        Pronóstico candidato sobre las mismas filas. Si se da, cada estadístico es la mejora
        del candidato: ``rmse_a − rmse_b``, ``mae_a − mae_b`` y ``r2_b − r2_a``.
    n_boot, nivel, seed

    Returns
    -------
    dict
        ``{'rmse' | 'mae' | 'r2': {'estimacion', 'ee_bootstrap', 'ic_percentil', 'ic_bca'}}``
        y, con dos modelos, ``prop_no_mejora`` en cada entrada.
    """
    y = _vector(y, nombre='y')
    pa = _vector(pred_a, len(y), 'pred_a')
    pb = None if pred_b is None else _vector(pred_b, len(y), 'pred_b')
    signo = {'rmse': -1.0, 'mae': -1.0, 'r2': 1.0}       # orientación «positivo favorece a b»

    def estadisticos(idx: np.ndarray) -> dict[str, float]:
        """Métricas de ``a`` en las filas ``idx`` o, si hay ``b``, las diferencias b − a orientadas para que lo positivo favorezca a ``b``."""
        ma = _metricas_regresion(y[idx], pa[idx])
        if pb is None:
            return ma
        mb = _metricas_regresion(y[idx], pb[idx])
        return {k: signo[k] * (mb[k] - ma[k]) for k in ma}

    n = len(y)
    rng = np.random.default_rng(seed)
    estim = estadisticos(np.arange(n))
    reps = {k: np.empty(n_boot) for k in estim}
    for b in range(n_boot):
        e = estadisticos(rng.integers(0, n, n))
        for k in e:
            reps[k][b] = e[k]
    jka = _jackknife_regresion(y, pa)
    if pb is None:
        jk = jka
    else:
        jkb = _jackknife_regresion(y, pb)
        jk = {k: signo[k] * (jkb[k] - jka[k]) for k in jka}
    alfa = (1 - nivel) / 2
    salida = {}
    for k in estim:
        salida[k] = {'estimacion': estim[k], 'ee_bootstrap': float(reps[k].std(ddof=1)),
                     'ic_percentil': tuple(float(q) for q in np.quantile(reps[k], [alfa, 1 - alfa])),
                     'ic_bca': intervalo_bca(estim[k], reps[k], jk[k], nivel)}
        if pb is not None:
            salida[k]['prop_no_mejora'] = float(np.mean(reps[k] <= 0))
    return salida


# ------------------------------------------------------------------ Diebold-Mariano
def diebold_mariano(perdida_a, perdida_b, h: int = 1, hln: bool = True) -> dict:
    """Contraste de Diebold-Mariano de igual precisión predictiva.

    Parameters
    ----------
    perdida_a, perdida_b : array-like
        Pérdida de cada observación (p. ej. error al cuadrado) del modelo de referencia y del
        candidato, sobre las mismas filas.
    h : int
        Horizonte del pronóstico: las autocovarianzas del diferencial hasta el retardo
        ``h − 1`` entran en la varianza de largo plazo. En datos de corte transversal, ``h = 1``.
    hln : bool
        Aplica la corrección de Harvey, Leybourne y Newbold (1997) y usa la t de Student con
        ``n − 1`` grados de libertad.

    Returns
    -------
    dict
        ``n``, ``h``, ``d_media`` (media de ``perdida_a − perdida_b``; positiva si el candidato
        pierde menos), ``estadistico_dm`` (sin corregir), ``estadistico`` (el que se contrasta),
        ``p_bilateral``, ``p_unilateral`` (H1: el candidato es más preciso), ``hln``, ``gl``.

    Notes
    -----
    ``DM = d̄ / sqrt(γ̂/n)`` con ``γ̂ = γ0 + 2 Σ_{k<h} γk``; la corrección multiplica por
    ``sqrt((n + 1 − 2h + h(h − 1)/n) / n)``. Con ``h = 1`` el estadístico corregido coincide
    exactamente con la t pareada sobre las pérdidas.
    """
    a = _vector(perdida_a, nombre='perdida_a')
    d = a - _vector(perdida_b, len(a), 'perdida_b')
    n = len(d)
    if not 1 <= h < n:
        raise ValueError('h debe estar entre 1 y n − 1')
    dc = d - d.mean()
    gammas = [float(np.dot(dc[k:], dc[:n - k]) / n) for k in range(h)]
    var_lp = gammas[0] + 2 * sum(gammas[1:])
    if var_lp <= 0:
        raise ValueError('varianza de largo plazo no positiva: los dos modelos pierden lo mismo')
    dm = float(d.mean() / np.sqrt(var_lp / n))
    if hln:
        estadistico = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dist, gl = stats.t(df=n - 1), n - 1
    else:
        estadistico, dist, gl = dm, stats.norm, None
    return {'n': n, 'h': h, 'd_media': float(d.mean()), 'estadistico_dm': dm,
            'estadistico': float(estadistico), 'p_bilateral': float(2 * dist.sf(abs(estadistico))),
            'p_unilateral': float(dist.sf(estadistico)), 'hln': hln, 'gl': gl}


def bootstrap_estacionario(diferencial, n_boot: int = N_BOOT, bloque: float | None = None,
                           nivel: float = NIVEL, seed: int = SEED) -> dict:
    """Contraste de media nula del diferencial de pérdidas con bootstrap estacionario.

    Parameters
    ----------
    diferencial : array-like
        ``perdida_a − perdida_b`` en el orden de las observaciones.
    n_boot : int
    bloque : float, optional
        Longitud media de bloque. Por defecto, la óptima de Politis y White (2004) con la
        corrección de Patton, Politis y White (2009), acotada por debajo en 1.
    nivel, seed

    Returns
    -------
    dict
        ``d_media``, ``bloque``, ``p_bilateral`` y ``p_unilateral`` (H1: media > 0), ambas con
        la distribución bootstrap centrada, e ``ic_percentil`` de la media.

    Notes
    -----
    Politis y Romano (1994): bloques de longitud geométrica con media ``bloque`` y arranque
    uniforme, que conservan la dependencia serial y dan una serie remuestreada estacionaria.
    Si el diferencial no tiene dependencia, el bloque óptimo sale ≈ 1 y el método se reduce al
    bootstrap i.i.d.: el resultado es el mismo contraste sin la hipótesis de normalidad del DM.
    """
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    d = _vector(diferencial, nombre='diferencial')
    if bloque is None:
        bloque = float(optimal_block_length(d)['stationary'].iloc[0])
    bloque = max(1.0, float(bloque))
    medias = StationaryBootstrap(bloque, d, seed=seed).apply(np.mean, n_boot).ravel()
    d_media = float(d.mean())
    centradas = medias - d_media
    alfa = (1 - nivel) / 2
    return {'d_media': d_media, 'bloque': bloque, 'n_boot': n_boot,
            'p_bilateral': float(np.mean(np.abs(centradas) >= abs(d_media))),
            'p_unilateral': float(np.mean(centradas >= d_media)),
            'ic_percentil': tuple(float(q) for q in np.quantile(medias, [alfa, 1 - alfa]))}


# -------------------------------------------------------------- tamaños del efecto
def delta_cliff(x, y) -> dict:
    """Delta de Cliff: ``P(X > Y) − P(X < Y)`` entre dos muestras.

    Parameters
    ----------
    x, y : array-like
        Por ejemplo, el AUC por pliegue del candidato (``x``) y el de la referencia (``y``):
        un delta positivo indica que el candidato tiende a puntuar más.

    Returns
    -------
    dict
        ``delta`` en [−1, 1] y ``magnitud`` con los cortes de Romano y otros (2006): 0,147,
        0,33 y 0,474.

    Notes
    -----
    Se obtiene del U de Mann-Whitney sin recorrer los pares: ``delta = 2U/(n_x·n_y) − 1``.
    Entre los puntajes de positivos y negativos de un clasificador vale ``2·AUC − 1``.
    """
    x, y = _vector(x, nombre='x'), _vector(y, nombre='y')
    u = stats.mannwhitneyu(x, y, alternative='two-sided').statistic
    delta = float(2 * u / (len(x) * len(y)) - 1)
    return {'delta': delta, 'magnitud': _magnitud(delta, (0.147, 0.33, 0.474))}


def d_cohen(x, y=None) -> dict:
    """d de Cohen pareada (``d_z``) o de dos muestras independientes.

    Parameters
    ----------
    x : array-like
        Diferencias pareadas si ``y`` es ``None`` (p. ej. ``pérdida_a − pérdida_b`` por fila);
        si no, la primera muestra.
    y : array-like, optional
        Segunda muestra.

    Returns
    -------
    dict
        ``d`` y ``magnitud`` con los cortes de Cohen (1988): 0,2, 0,5 y 0,8.

    Notes
    -----
    Pareada: ``d_z = media(x) / desv(x)``. Dos muestras: diferencia de medias entre la
    desviación combinada. La magnitud convencional se pensó para la segunda; en la pareada
    sobre miles de filas un ``d_z`` pequeño puede ser muy significativo, y por eso se reporta
    junto a la mejora absoluta de la métrica.
    """
    x = _vector(x, nombre='x')
    if y is None:
        d = float(x.mean() / x.std(ddof=1))
    else:
        y = _vector(y, nombre='y')
        nx, ny = len(x), len(y)
        sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
        d = float((x.mean() - y.mean()) / sp)
    return {'d': d, 'magnitud': _magnitud(d, (0.2, 0.5, 0.8))}


def ajustar_pvalores(pvalores, metodo: str = 'holm') -> np.ndarray:
    """Corrección por comparaciones múltiples.

    Parameters
    ----------
    pvalores : array-like
    metodo : {'holm', 'fdr_bh', 'bonferroni'}
        Holm-Bonferroni controla el error de familia; Benjamini-Hochberg (``'fdr_bh'``), la
        tasa de falsos descubrimientos.

    Returns
    -------
    ndarray
        p-valores ajustados, en el mismo orden.
    """
    from statsmodels.stats.multitest import multipletests
    return multipletests(np.asarray(pvalores, dtype=float), method=metodo)[1]


# ------------------------------------------------------------ varios modelos (§6)
def friedman(matriz, mayor_es_mejor: bool = True, alfa: float = 0.05) -> dict:
    """Friedman con la corrección de Iman-Davenport y comparaciones de Nemenyi.

    Parameters
    ----------
    matriz : DataFrame
        Filas = bloques (pliegues, combinaciones), columnas = tratamientos (modelos u
        optimizadores). Los bloques incompletos se descartan.
    mayor_es_mejor : bool
        Orientación de la métrica: el rango 1 es siempre el mejor tratamiento.
    alfa : float
        Nivel de la diferencia crítica de Nemenyi.

    Returns
    -------
    dict
        ``n_bloques``, ``k``, ``chi2`` y ``p`` (Friedman), ``F`` y ``p_iman_davenport``,
        ``rangos_medios`` (Series ordenada, 1 = mejor), ``dc`` (diferencia crítica de Nemenyi) y
        ``p_nemenyi`` (DataFrame k × k).

    Notes
    -----
    Demšar (2006): Friedman compara los rangos medios de k tratamientos sobre N bloques; la
    corrección de Iman y Davenport (1980), ``F = (N − 1)χ² / (N(k − 1) − χ²)`` con
    ``(k − 1, (k − 1)(N − 1))`` grados de libertad, es menos conservadora. Dos tratamientos
    difieren según Nemenyi si sus rangos medios se separan más que
    ``DC = q_α · sqrt(k(k + 1)/(6N))``, con ``q_α`` el rango estudentizado entre √2.
    """
    import pandas as pd
    import scikit_posthocs as sp

    m = pd.DataFrame(matriz).dropna()
    n_bloques, k = m.shape
    chi2, p = stats.friedmanchisquare(*[m[c].to_numpy() for c in m.columns])
    den = n_bloques * (k - 1) - chi2
    F = float((n_bloques - 1) * chi2 / den) if den > 0 else float('inf')
    rangos = m.rank(axis=1, ascending=not mayor_es_mejor).mean().sort_values()
    q = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    p_nem = sp.posthoc_nemenyi_friedman(m.to_numpy())
    p_nem.index = p_nem.columns = m.columns
    return {'n_bloques': int(n_bloques), 'k': int(k), 'chi2': float(chi2), 'p': float(p), 'F': F,
            'p_iman_davenport': float(stats.f.sf(F, k - 1, (k - 1) * (n_bloques - 1))),
            'rangos_medios': rangos, 'dc': float(q * np.sqrt(k * (k + 1) / (6 * n_bloques))),
            'p_nemenyi': p_nem}


def conjunto_confianza_modelos(perdidas, alfa: float = 0.10, n_boot: int = 1000,
                               bloque: int | None = None, seed: int = SEED) -> dict:
    """*Model Confidence Set* de Hansen, Lunde y Nason (2011) con bootstrap estacionario.

    Parameters
    ----------
    perdidas : DataFrame
        Una columna de pérdidas por modelo (p. ej. errores al cuadrado), mismas filas.
    alfa : float
        El conjunto contiene al mejor modelo con probabilidad ``1 − alfa``.
    n_boot : int
    bloque : int, optional
        Longitud media de bloque; por defecto la mayor de las óptimas de Politis y White de los
        diferenciales de cada modelo frente a la pérdida media, acotada por debajo en 1.
    seed : int

    Returns
    -------
    dict
        ``incluidos`` (modelos del conjunto), ``excluidos`` (en el orden en que se eliminaron),
        ``p_mcs`` (Series: p-valor MCS de cada modelo), ``bloque`` y ``alfa``.

    Notes
    -----
    Se eliminan modelos de uno en uno mientras se rechaza la igualdad de precisión del
    conjunto (estadístico de rango ``T_R``); el p-valor MCS de un modelo es el mayor p-valor
    de las pruebas que lo eliminaron, así que los modelos con p-valor mayor que ``alfa``
    forman el conjunto.
    """
    import pandas as pd
    from arch.bootstrap import MCS, optimal_block_length

    perdidas = pd.DataFrame(perdidas)
    if bloque is None:
        media = perdidas.mean(axis=1)
        bloque = max(float(optimal_block_length((perdidas[c] - media).to_numpy())['stationary'].iloc[0])
                     for c in perdidas.columns)
    bloque = max(1, int(round(bloque)))
    mcs = MCS(perdidas, size=alfa, reps=n_boot, block_size=bloque, method='R', bootstrap='stationary',
              seed=seed)
    mcs.compute()
    return {'incluidos': list(mcs.included), 'excluidos': list(mcs.excluded),
            'p_mcs': mcs.pvalues.iloc[:, 0].sort_values(ascending=False), 'bloque': bloque, 'alfa': alfa}


def giacomini_white(perdida_a, perdida_b, instrumentos=None) -> dict:
    """Contraste de capacidad predictiva condicional de Giacomini y White (2006), horizonte 1.

    Parameters
    ----------
    perdida_a, perdida_b : array-like
        Pérdidas del modelo de referencia y del candidato sobre las mismas filas.
    instrumentos : array-like, optional
        Variables (n × q) que pueden anticipar qué modelo será mejor; se añade la constante.
        Por defecto, el diferencial retardado ``d_{t−1}`` (la versión de series temporales).

    Returns
    -------
    dict
        ``estadistico`` (χ² con ``gl`` = número de instrumentos más la constante), ``p``,
        ``d_media`` y ``coeficientes`` de la regresión de ``d`` sobre los instrumentos, que
        dicen *cuándo* gana cada modelo.

    Notes
    -----
    Con ``Z_t = h_t·d_t``, el estadístico es ``n · Z̄' Ω̂⁻¹ Z̄`` con ``Ω̂ = Σ Z_t Z_t' / n``. Con
    solo la constante se reduce a un Diebold-Mariano sin corrección: los instrumentos añaden la
    pregunta de si la ventaja depende de algo observable, como el nivel de riesgo predicho.
    """
    d = _vector(perdida_a, nombre='perdida_a') - _vector(perdida_b, nombre='perdida_b')
    if instrumentos is None:
        H = np.column_stack([np.ones(len(d) - 1), d[:-1]])
        d = d[1:]
    else:
        inst = np.asarray(instrumentos, dtype=float).reshape(len(d), -1)
        H = np.column_stack([np.ones(len(d)), inst])
    n, q = H.shape
    Z = H * d[:, None]
    zbar = Z.mean(axis=0)
    omega = Z.T @ Z / n
    estadistico = float(n * zbar @ np.linalg.solve(omega, zbar))
    coef = np.linalg.lstsq(H, d, rcond=None)[0]
    return {'estadistico': estadistico, 'gl': int(q), 'p': float(stats.chi2.sf(estadistico, q)),
            'd_media': float(d.mean()), 'coeficientes': [float(c) for c in coef]}
