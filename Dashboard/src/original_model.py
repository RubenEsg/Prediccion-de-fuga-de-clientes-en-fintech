"""Boosting con Desplazamiento Latente (BDL): el modelo original del proyecto.

Problema que resuelve
---------------------
En COFINFAD el puntaje de fuga ``s`` es la suma de un índice que depende de los predictores y de un escalón que depende
de una variable que el modelo no recibe (el número de productos activos, excluido por fuga de información). Dado ``x``,
el puntaje no se dispersa como una campana alrededor de su media: cae en uno de varios niveles separados. BDL supone

    s = F(x) + c_z + e,        z ∈ {1, …, K},   P(z = k) = π_k,   e ~ N(0, σ²),

donde ``F`` es una suma de árboles pequeños sobre las variables discretizadas, ``z`` es una clase latente (no se
observa), ``c_k`` es el desplazamiento de la clase ``k`` y ``σ`` la dispersión dentro de cada clase. Con ``K = 1`` el
modelo es el boosting de mínimos cuadrados de siempre.

Qué tiene de propio
-------------------
1. **Regla de aprendizaje: EM generalizado con un paso de boosting.** Un regresor de mínimos cuadrados trata el
   desplazamiento ``c_z`` como ruido, y ese ruido es mucho mayor que ``σ``. BDL alterna tres pasos:

   - *Paso E*: con el índice actual, calcula la probabilidad de que cada cliente pertenezca a cada clase latente,
     ``γ_ik ∝ π_k · N(s_i − F(x_i); c_k, σ²)``.
   - *Pasos de boosting*: ajusta unos pocos árboles al residuo **sin escalón**, ``s_i − Σ_k γ_ik c_k − F(x_i)``, y suma
     cada uno a ``F`` multiplicado por la tasa de aprendizaje.
   - *Paso M*: actualiza ``π``, ``c`` y ``σ`` con sus fórmulas cerradas.

   Cada vuelta aumenta (o deja igual) la verosimilitud observada; la demostración está en el informe y la prueba
   ``test_verosimilitud_no_decrece`` la comprueba numéricamente.

2. **Regla de salida.** La probabilidad de superar un umbral ``τ`` (la clasificación) sale de la mezcla:
   ``P(s > τ | x) = Σ_k π_k · Φ((F(x) + c_k − τ)/σ)``, es decir, cada clase latente aporta su peso si su nivel queda por
   encima del umbral. Para la regresión, una parte de las filas no entra en los árboles: con sus residuos
   ``r_j = s_j − F(x_j)`` se arma la distribución predictiva de un cliente nuevo, ``F(x) + r_j``, y de ella salen los
   cuantiles, los intervalos y los conjuntos de predicción, con cobertura garantizada bajo intercambiabilidad
   (predicción conforme por partición, con la densidad de la mezcla como medida de conformidad).

Solo se usan ``numpy`` y ``scipy`` como herramientas de cálculo, y las clases base de scikit-learn para que el modelo
entre en un ``Pipeline``, en ``clone`` y en la validación cruzada. No se importa ningún modelo.

Clases
------
``RegresorBDL``
    Aprende ``F``, la mezcla y la distribución predictiva a partir del puntaje continuo.
``ClasificadorBDL``
    Clasifica «el puntaje supera el umbral». Si en ``fit`` recibe el puntaje continuo (``puntaje=``) usa un
    ``RegresorBDL`` y la probabilidad de excedencia; si no lo recibe, aprende directamente de la etiqueta con el mismo
    motor de árboles y pérdida logística.
``ModeloOriginal``
    Alias de ``ClasificadorBDL`` (el nombre que usa la guía del curso).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import ndtr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

LOG_2PI = float(np.log(2.0 * np.pi))
BINS_HISTOGRAMA = 512
"""Resolución del histograma de residuos con el que se inicializa la mezcla."""
PACIENCIA_CALENTAMIENTO = 10
"""Árboles sin mejora en validación con los que se corta el calentamiento."""
SEPARACION_MINIMA = 4.0
"""Distancia mínima entre dos desplazamientos, en desviaciones típicas, para contarlos como clases distintas."""
PESO_MINIMO = 0.01
"""Fracción mínima de filas de una clase latente."""


# ============================================================================ 1. discretización
def calcular_cortes(X: np.ndarray, max_bins: int) -> list[np.ndarray]:
    """Cortes que separan los bins de cada columna.

    Parameters
    ----------
    X : ndarray de forma (n, p)
        Filas con las que se aprenden los cortes (solo las de ajuste de árboles).
    max_bins : int
        Máximo de bins por columna.

    Returns
    -------
    list of ndarray
        Para cada columna, los cortes en orden creciente. Si la columna tiene a lo sumo ``max_bins`` valores distintos,
        hay un bin por valor (los cortes son los puntos medios); si tiene más, los cortes son cuantiles, de modo que
        cada bin recibe aproximadamente la misma cantidad de filas.
    """
    cortes = []
    for j in range(X.shape[1]):
        valores = np.unique(X[:, j])
        if len(valores) <= max_bins:
            c = (valores[:-1] + valores[1:]) / 2.0
        else:
            c = np.unique(np.quantile(X[:, j], np.linspace(0.0, 1.0, max_bins + 1)[1:-1]))
        cortes.append(c)
    return cortes


def codificar(X: np.ndarray, cortes: list[np.ndarray]) -> np.ndarray:
    """Reemplaza cada valor por el número de su bin.

    Parameters
    ----------
    X : ndarray de forma (n, p)
    cortes : list of ndarray
        Salida de :func:`calcular_cortes`.

    Returns
    -------
    ndarray de enteros de forma (n, p)
        ``C[i, j] = b`` significa que ``X[i, j]`` cae en el bin ``b`` de la columna ``j``: es mayor que ``b`` cortes.
        Un valor fuera del rango visto cae en el primer o en el último bin.
    """
    C = np.empty(X.shape, dtype=np.int32)
    for j, c in enumerate(cortes):
        C[:, j] = np.searchsorted(c, X[:, j], side='left')
    return C


class _Histogramas:
    """Suma, por bin de cada variable, un vector definido sobre las filas de ajuste.

    La matriz indicadora ``Z`` tiene una fila por cliente y una columna por bin (de todas las variables, una tras
    otra): ``Z[i, col] = 1`` si el cliente ``i`` cae en ese bin. Entonces ``Zᵀ v`` es la suma de ``v`` dentro de cada
    bin, que es todo lo que un árbol necesita para elegir su corte. ``Z`` se guarda en formato disperso: tiene un uno
    por cliente y variable.
    """

    def __init__(self, C: np.ndarray, n_bins: np.ndarray) -> None:
        n, p = C.shape
        self.inicio = np.concatenate(([0], np.cumsum(n_bins))).astype(np.int64)   # primer bin de cada variable
        self.variable_de = np.repeat(np.arange(p), n_bins)                       # variable a la que pertenece cada bin
        columnas = (C + self.inicio[:-1]).ravel()
        filas = np.repeat(np.arange(n), p)
        self.Z = sparse.csr_matrix((np.ones(n * p), (filas, columnas)), shape=(n, int(self.inicio[-1])))
        self.Zt = self.Z.T.tocsr()
        self.conteo_raiz = np.asarray(self.Zt.sum(axis=1)).ravel()               # filas por bin (no cambia)

    def sumar(self, v: np.ndarray) -> np.ndarray:
        """Suma de ``v`` dentro de cada bin, con todas las filas."""
        return self.Zt @ v

    def sumar_filas(self, filas: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Suma de ``v`` y número de filas por bin, usando solo las filas indicadas.

        Parameters
        ----------
        filas : ndarray de enteros
            Posiciones de las filas del nodo.
        v : ndarray
            Valores de esas filas, en el mismo orden.

        Returns
        -------
        suma, conteo : ndarray
            Cuesta en proporción al número de filas del nodo, no al total.
        """
        Z_nodo = self.Z[filas]
        return Z_nodo.T @ v, np.bincount(Z_nodo.indices, minlength=self.Z.shape[1]).astype(np.float64)


# ============================================================================ 2. árbol de regresión propio
def _mejor_corte(G: np.ndarray, N: np.ndarray, suma: float, n: float, hist: _Histogramas, min_hoja: int,
                 lam: float) -> tuple[int, int] | None:
    """Elige la variable y el bin que más reducen el error cuadrático de un nodo.

    Parameters
    ----------
    G, N : ndarray
        Suma de residuos y número de filas del nodo en cada bin (todas las variables, una tras otra).
    suma, n : float
        Suma de residuos y número de filas del nodo.
    hist : _Histogramas
    min_hoja : int
        Mínimo de filas a cada lado del corte.
    lam : float
        Regularización L2 del valor de las hojas.

    Returns
    -------
    (variable, bin) or None
        El corte «código ≤ bin va a la izquierda», o ``None`` si ningún corte válido mejora el nodo.

    Notes
    -----
    Con valor de hoja ``v = G/(N + λ)``, la reducción del error cuadrático regularizado al partir un nodo en izquierda
    (L) y derecha (R) es ``G_L²/(N_L + λ) + G_R²/(N_R + λ) − G²/(N + λ)``. Las sumas a la izquierda de cada corte se
    obtienen con una suma acumulada sobre los bins de la variable.
    """
    acum_g, acum_n = np.cumsum(G), np.cumsum(N)
    fin_anterior = hist.inicio[1:-1] - 1                       # último bin de la variable anterior
    base_g = np.concatenate(([0.0], acum_g[fin_anterior]))[hist.variable_de]
    base_n = np.concatenate(([0.0], acum_n[fin_anterior]))[hist.variable_de]
    g_izq, n_izq = acum_g - base_g, acum_n - base_n
    g_der, n_der = suma - g_izq, n - n_izq
    valido = (n_izq >= min_hoja) & (n_der >= min_hoja)
    if not valido.any():
        return None
    ganancia = np.full(len(G), -np.inf)
    ganancia[valido] = (g_izq[valido] ** 2 / (n_izq[valido] + lam) + g_der[valido] ** 2 / (n_der[valido] + lam)
                        - suma ** 2 / (n + lam))
    mejor = int(np.argmax(ganancia))
    if not ganancia[mejor] > 1e-14:
        return None
    variable = int(hist.variable_de[mejor])
    return variable, int(mejor - hist.inicio[variable])


def ajustar_arbol(hist: _Histogramas, C: np.ndarray, r: np.ndarray, profundidad: int, min_hoja: int,
                  lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Árbol de regresión de mínimos cuadrados, crecido por niveles sobre las variables discretizadas.

    Parameters
    ----------
    hist : _Histogramas
        Construido con las mismas filas que ``C`` y ``r``.
    C : ndarray de enteros de forma (n, p)
        Códigos de bin de las filas de ajuste.
    r : ndarray de forma (n,)
        Residuo que el árbol debe aproximar.
    profundidad : int
        Niveles de corte (1 = un solo corte, dos hojas).
    min_hoja : int
    lam : float

    Returns
    -------
    variable, corte : ndarray de enteros de forma (2^(profundidad+1) − 1,)
        Árbol binario completo guardado por posiciones: los hijos del nodo ``k`` son ``2k+1`` y ``2k+2``.
        ``variable[k] = −1`` marca una hoja; si no, el nodo manda a la izquierda las filas con código ≤ ``corte[k]``.
    valor : ndarray
        Valor de cada hoja, ``Σ r / (n_hoja + λ)``.
    cobertura : ndarray
        Filas de ajuste que pasan por cada nodo (se usa para repartir aportes entre variables).
    nodo : ndarray de enteros de forma (n,)
        Hoja en la que quedó cada fila de ajuste.

    Notes
    -----
    Cada nodo necesita su histograma (suma de residuos y número de filas por bin). El de la raíz usa todas las filas.
    Al partir un nodo solo se calcula el histograma del hijo con menos filas; el del otro hijo es el del padre menos
    ese. Así un nivel entero cuesta a lo sumo la mitad que la raíz.
    """
    n = len(r)
    n_nodos = 2 ** (profundidad + 1) - 1
    variable = np.full(n_nodos, -1, dtype=np.int32)
    corte = np.zeros(n_nodos, dtype=np.int32)
    nodo = np.zeros(n, dtype=np.int32)
    pendientes = {0: (None, hist.sumar(r), hist.conteo_raiz, float(r.sum()), float(n))}   # filas, G, N, suma, n
    for nivel in range(profundidad):
        siguientes = {}
        for k, (filas, G, N, suma_k, n_k) in pendientes.items():
            elegido = _mejor_corte(G, N, suma_k, n_k, hist, min_hoja, lam)
            if elegido is None:
                continue
            j, b = elegido
            variable[k], corte[k] = j, b
            if filas is None:
                a_izquierda = C[:, j] <= b
                izquierda, derecha = np.flatnonzero(a_izquierda), np.flatnonzero(~a_izquierda)
            else:
                a_izquierda = C[filas, j] <= b
                izquierda, derecha = filas[a_izquierda], filas[~a_izquierda]
            nodo[izquierda], nodo[derecha] = 2 * k + 1, 2 * k + 2
            if nivel + 1 == profundidad:
                continue                                        # los hijos ya son hojas: no necesitan histograma
            menor, mayor = (izquierda, derecha) if len(izquierda) <= len(derecha) else (derecha, izquierda)
            r_menor = r[menor]
            G_menor, N_menor = hist.sumar_filas(menor, r_menor)
            suma_menor = float(r_menor.sum())
            hijos = {id(menor): (menor, G_menor, N_menor, suma_menor, float(len(menor))),
                     id(mayor): (mayor, G - G_menor, N - N_menor, suma_k - suma_menor, n_k - len(menor))}
            siguientes[2 * k + 1], siguientes[2 * k + 2] = hijos[id(izquierda)], hijos[id(derecha)]
        pendientes = siguientes
    cobertura = np.bincount(nodo, minlength=n_nodos).astype(np.float64)
    suma = np.bincount(nodo, weights=r, minlength=n_nodos)
    valor = suma / np.maximum(cobertura + lam, 1e-12)
    for k in range(n_nodos - 1, 0, -1):                        # un nodo interno cubre lo que cubren sus hijos
        cobertura[(k - 1) // 2] += cobertura[k]
    return variable, corte, valor, cobertura, nodo


def predecir_arbol(variable: np.ndarray, corte: np.ndarray, valor: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Valor de la hoja en la que cae cada fila.

    Parameters
    ----------
    variable, corte, valor : ndarray
        Árbol devuelto por :func:`ajustar_arbol`.
    C : ndarray de enteros de forma (n, p)
        Códigos de bin de las filas.

    Returns
    -------
    ndarray de forma (n,)
    """
    nodo = np.zeros(C.shape[0], dtype=np.int64)
    while True:
        v = variable[nodo]
        internos = np.flatnonzero(v >= 0)
        if len(internos) == 0:
            return valor[nodo]
        k = nodo[internos]
        a_derecha = C[internos, v[internos]] > corte[k]
        nodo[internos] = 2 * k + 1 + a_derecha


# ============================================================================ 3. mezcla de desplazamientos
def paso_e(r: np.ndarray, pi: np.ndarray, c: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Paso E: probabilidad de cada clase latente dado el residuo, y verosimilitud de cada fila.

    Parameters
    ----------
    r : ndarray de forma (n,)
        Residuo ``s − F(x)``.
    pi, c : ndarray de forma (K,)
        Pesos y desplazamientos de la mezcla.
    sigma : float
        Desviación típica común dentro de cada clase.

    Returns
    -------
    gamma : ndarray de forma (n, K)
        ``γ_ik = π_k N(r_i; c_k, σ²) / Σ_l π_l N(r_i; c_l, σ²)``.
    log_fila : ndarray de forma (n,)
        ``log Σ_k π_k N(r_i; c_k, σ²)``; su suma es la verosimilitud observada.

    Notes
    -----
    Se trabaja con logaritmos y se resta el máximo de cada fila antes de exponenciar, para que una ``σ`` muy pequeña no
    produzca ceros.
    """
    z = (r[:, None] - c[None, :]) / sigma
    logp = -0.5 * z * z + np.log(np.maximum(pi, 1e-300))[None, :]
    maximo = logp.max(axis=1, keepdims=True)
    gamma = np.exp(logp - maximo)
    total = gamma.sum(axis=1, keepdims=True)
    gamma /= total
    return gamma, maximo[:, 0] + np.log(total[:, 0]) - np.log(sigma) - 0.5 * LOG_2PI


def paso_m(r: np.ndarray, gamma: np.ndarray, c_anterior: np.ndarray, sigma_min: float,
           peso: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Paso M: pesos, desplazamientos y dispersión que maximizan la verosimilitud esperada.

    Parameters
    ----------
    r : ndarray de forma (n,)
    gamma : ndarray de forma (n, K)
        Responsabilidades del paso E.
    c_anterior : ndarray de forma (K,)
        Desplazamientos vigentes; una clase que se queda sin filas conserva el suyo.
    sigma_min : float
        Piso de ``σ`` (evita que una clase degenere en un punto).
    peso : ndarray de forma (n,), optional
        Número de filas que representa cada residuo (cuando ``r`` son las barras de un histograma).

    Returns
    -------
    pi, c : ndarray de forma (K,)
        ``π_k = Σ_i γ_ik / n`` y ``c_k = Σ_i γ_ik r_i / Σ_i γ_ik``.
    sigma : float
        ``σ² = Σ_i Σ_k γ_ik (r_i − c_k)² / n``, no menor que ``sigma_min``.

    Notes
    -----
    Las tres sumas ``Σ_i γ_ik``, ``Σ_i γ_ik r_i`` y ``Σ_i γ_ik r_i²`` bastan: la varianza se arma con ellas,
    ``Σ_i γ_ik (r_i − c_k)² = Σ γ r² − 2 c_k Σ γ r + c_k² Σ γ``.
    """
    if peso is not None:
        gamma = gamma * peso[:, None]
    masa = gamma.sum(axis=0)
    total = float(masa.sum())
    suma_r, suma_r2 = gamma.T @ r, gamma.T @ (r * r)
    c = np.where(masa > 1e-9, suma_r / np.maximum(masa, 1e-9), c_anterior)
    varianza = float(np.sum(suma_r2 - 2.0 * c * suma_r + c * c * masa) / total)
    return masa / total, c, max(float(np.sqrt(max(varianza, 0.0))), sigma_min)


def _kmedias_1d(posicion: np.ndarray, peso: np.ndarray, K: int) -> np.ndarray:
    """K-medias exacto en una dimensión sobre un histograma, por programación dinámica.

    Parameters
    ----------
    posicion, peso : ndarray de forma (m,)
        Centro y número de filas de cada barra del histograma, en orden creciente de posición.
    K : int

    Returns
    -------
    ndarray de forma (K,)
        Centros de los ``K`` grupos contiguos que minimizan la suma de cuadrados dentro de los grupos.

    Notes
    -----
    En una dimensión los grupos óptimos son intervalos, así que basta decidir dónde cortar. ``costo[k][j]`` es la mejor
    suma de cuadrados al repartir las barras ``0..j`` en ``k+1`` grupos; se llena de izquierda a derecha probando el
    inicio ``i`` del último grupo. La suma de cuadrados de las barras ``i..j`` sale de tres sumas acumuladas.
    """
    m = len(posicion)
    w = np.concatenate(([0.0], np.cumsum(peso)))
    wx = np.concatenate(([0.0], np.cumsum(peso * posicion)))
    wxx = np.concatenate(([0.0], np.cumsum(peso * posicion ** 2)))

    def dentro(i: np.ndarray, j) -> np.ndarray:
        """Suma de cuadrados de las barras ``i..j`` alrededor de su media."""
        masa = w[j + 1] - w[i]
        return (wxx[j + 1] - wxx[i]) - (wx[j + 1] - wx[i]) ** 2 / np.maximum(masa, 1e-300)

    costo = np.full((K, m), np.inf)
    corte = np.zeros((K, m), dtype=np.int64)
    costo[0] = dentro(np.zeros(m, dtype=np.int64), np.arange(m))    # un solo grupo: las barras 0..j
    for k in range(1, K):
        for j in range(k, m):
            inicios = np.arange(k, j + 1)
            total = costo[k - 1, inicios - 1] + dentro(inicios, j)
            mejor = int(np.argmin(total))
            costo[k, j], corte[k, j] = total[mejor], inicios[mejor]
    centros, j = np.zeros(K), m - 1
    for k in range(K - 1, -1, -1):
        i = corte[k, j] if k > 0 else 0
        centros[k] = (wx[j + 1] - wx[i]) / max(w[j + 1] - w[i], 1e-300)
        j = i - 1
    return centros


def _histograma(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Histograma del residuo: centro y número de filas de cada barra no vacía, y ancho de barra."""
    cuenta, bordes = np.histogram(r, bins=BINS_HISTOGRAMA)
    con_filas = cuenta > 0
    return ((bordes[:-1] + bordes[1:]) / 2.0)[con_filas], cuenta[con_filas].astype(np.float64), float(bordes[1] - bordes[0])


def ajustar_mezcla(r: np.ndarray, K: int, sigma_min: float, n_iter: int = 300,
                   tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Mezcla de ``K`` normales con varianza común ajustada al histograma de un residuo, por EM.

    Parameters
    ----------
    r : ndarray de forma (n,)
    K : int
    sigma_min : float
    n_iter : int
        Máximo de iteraciones de EM.
    tol : float
        Se detiene cuando la verosimilitud por fila mejora menos que ``tol``.

    Returns
    -------
    pi, c : ndarray de forma (K,)
    sigma : float
    verosimilitud : float
        Verosimilitud del histograma con los parámetros devueltos.

    Notes
    -----
    Se ajusta sobre el histograma (512 barras con su número de filas) y no sobre las filas: cuesta lo mismo con mil
    filas que con un millón. Los centros iniciales salen de un K-medias exacto (:func:`_kmedias_1d`): es determinista y
    no deja dos centros en el mismo escalón, que es el fallo típico de una inicialización por cuantiles. ``σ`` no baja
    de media barra, porque el histograma no distingue nada más fino; el EM generalizado la afina después con las filas.
    """
    posicion, peso, ancho = _histograma(r)
    n = float(peso.sum())
    piso = max(sigma_min, ancho / 2.0)
    K = int(min(K, len(posicion)))
    c = _kmedias_1d(posicion, peso, K)
    cercano = np.argmin(np.abs(posicion[:, None] - c[None, :]), axis=1)
    pi = np.maximum(np.bincount(cercano, weights=peso, minlength=K) / n, 1e-12)
    sigma = max(float(np.sqrt(peso @ (posicion - c[cercano]) ** 2 / n)), piso)
    anterior = -np.inf
    for _ in range(n_iter):
        gamma, log_fila = paso_e(posicion, pi, c, sigma)
        vero = float(peso @ log_fila)
        if vero - anterior < tol * n:
            break
        anterior = vero
        pi, c, sigma = paso_m(posicion, gamma, c, piso, peso)
    return pi, c, sigma, float(peso @ paso_e(posicion, pi, c, sigma)[1])


def _clases_utiles(pi: np.ndarray, c: np.ndarray, sigma: float) -> int:
    """Cuenta las clases latentes que son escalones de verdad.

    Parameters
    ----------
    pi, c : ndarray de forma (K,)
    sigma : float

    Returns
    -------
    int
        Número de clases tras unir las que distan menos de ``SEPARACION_MINIMA`` desviaciones de la anterior y
        descartar las que reúnen menos de ``PESO_MINIMO`` de las filas (nunca menos de 1).

    Notes
    -----
    El BIC premia cualquier mejora del ajuste, y el error dentro de un escalón no es exactamente normal: a veces parte
    un escalón en dos componentes casi superpuestos, o dedica un componente a un puñado de filas atípicas. Ninguno de
    los dos es un desplazamiento latente, así que no se cuentan.
    """
    orden = np.argsort(c)
    grupos = [[c[orden[0]], pi[orden[0]]]]
    for k in orden[1:]:
        if c[k] - grupos[-1][0] < SEPARACION_MINIMA * sigma:
            grupos[-1][1] += pi[k]
        else:
            grupos.append([c[k], pi[k]])
    return max(1, sum(1 for _, peso in grupos if peso >= PESO_MINIMO))


def elegir_componentes(r: np.ndarray, max_componentes: int, sigma_min: float) -> tuple[int, list[dict]]:
    """Número de clases latentes: el que minimiza el BIC, depurado con la regla de parsimonia.

    Parameters
    ----------
    r : ndarray de forma (n,)
        Residuo tras el calentamiento.
    max_componentes : int
    sigma_min : float

    Returns
    -------
    K : int
    tabla : list of dict
        ``K``, ``verosimilitud``, ``bic`` y ``utiles`` (clases que quedan tras la depuración) de cada candidato.

    Notes
    -----
    ``BIC(K) = −2·verosimilitud + 2K·log n``: la mezcla tiene ``K − 1`` pesos libres, ``K`` desplazamientos y una
    dispersión. Del ``K`` de menor BIC solo se conservan los escalones separados y con filas suficientes
    (:func:`_clases_utiles`). Con ruido unimodal queda ``K = 1`` y el modelo se reduce al boosting de mínimos cuadrados.
    """
    tabla = []
    for K in range(1, max_componentes + 1):
        pi, c, sigma, vero = ajustar_mezcla(r, K, sigma_min)
        tabla.append({'K': K, 'verosimilitud': vero, 'bic': -2.0 * vero + 2.0 * K * np.log(len(r)),
                      'utiles': _clases_utiles(pi, c, sigma)})
    return min(tabla, key=lambda f: f['bic'])['utiles'], tabla


def densidad_mezcla(r: np.ndarray, pi: np.ndarray, c: np.ndarray, sigma: float) -> np.ndarray:
    """Densidad de la mezcla de desplazamientos en cada residuo."""
    z = (np.asarray(r, dtype=np.float64)[..., None] - c) / sigma
    return (pi * np.exp(-0.5 * z ** 2)).sum(axis=-1) / (sigma * np.sqrt(2.0 * np.pi))


def _errores_intraclase(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distancia con signo de cada residuo al desplazamiento más cercano: el error dentro de su clase."""
    cercano = np.argmin(np.abs(r[:, None] - c[None, :]), axis=1)
    return r - c[cercano]


def _error_intraclase(r: np.ndarray, c: np.ndarray) -> float:
    """Raíz del error cuadrático medio de cada residuo respecto del desplazamiento más cercano."""
    return float(np.sqrt(np.mean(_errores_intraclase(r, c) ** 2)))


def _dispersion_robusta(e: np.ndarray) -> float:
    """Desviación típica robusta: 1,4826 veces la mediana de las desviaciones absolutas respecto de la mediana.

    Notes
    -----
    El factor 1,4826 hace que coincida con la desviación típica cuando los errores son normales. A diferencia de la
    raíz del error cuadrático medio, un puñado de errores grandes no la mueve.
    """
    return float(1.4826 * np.median(np.abs(e - np.median(e))))


# ============================================================================ 4. utilidades comunes
def _repartir_filas(n: int, fraccion_validacion: float, fraccion_calibracion: float,
                    random_state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reparte las filas en ajuste de árboles, validación (parada temprana) y calibración (salida conforme)."""
    orden = np.random.default_rng(random_state).permutation(n)
    n_cal = int(round(fraccion_calibracion * n))
    n_val = int(round(fraccion_validacion * n))
    if n - n_cal - n_val < 10 or n_val < 2:
        raise ValueError(f'con {n} filas no alcanza para ajustar ({n - n_cal - n_val}), validar ({n_val}) y calibrar ({n_cal})')
    return orden[n_cal + n_val:], orden[n_cal:n_cal + n_val], orden[:n_cal]


class _Bosque:
    """Lista de árboles ya multiplicados por la tasa de aprendizaje."""

    def __init__(self) -> None:
        self.variable, self.corte, self.valor, self.cobertura = [], [], [], []

    def agregar(self, variable, corte, valor, cobertura) -> None:
        self.variable.append(variable)
        self.corte.append(corte)
        self.valor.append(valor)
        self.cobertura.append(cobertura)

    def recortar(self, n_arboles: int) -> None:
        self.variable, self.corte = self.variable[:n_arboles], self.corte[:n_arboles]
        self.valor, self.cobertura = self.valor[:n_arboles], self.cobertura[:n_arboles]

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (np.array(self.variable, dtype=np.int32), np.array(self.corte, dtype=np.int32),
                np.array(self.valor, dtype=np.float64), np.array(self.cobertura, dtype=np.float64))


def _validar_comunes(m) -> None:
    """Comprueba los hiperparámetros que comparten el regresor y el clasificador."""
    if not (isinstance(m.n_vueltas, (int, np.integer)) and m.n_vueltas >= 1):
        raise ValueError('n_vueltas debe ser un entero positivo')
    if not (isinstance(m.arboles_por_vuelta, (int, np.integer)) and m.arboles_por_vuelta >= 1):
        raise ValueError('arboles_por_vuelta debe ser un entero positivo')
    if not 0.0 < m.tasa <= 1.0:
        raise ValueError('tasa debe estar en (0, 1]')
    if not (isinstance(m.profundidad, (int, np.integer)) and 1 <= m.profundidad <= 6):
        raise ValueError('profundidad debe ser un entero entre 1 y 6')
    if not (isinstance(m.max_bins, (int, np.integer)) and 2 <= m.max_bins <= 1024):
        raise ValueError('max_bins debe ser un entero entre 2 y 1024')
    if m.min_hoja < 1 or m.lambda_l2 < 0:
        raise ValueError('min_hoja debe ser ≥ 1 y lambda_l2 ≥ 0')
    if not 0.0 < m.fraccion_validacion < 0.5:
        raise ValueError('fraccion_validacion debe estar en (0, 0.5)')


def _sumar_arboles(variables, cortes, valores, C: np.ndarray) -> np.ndarray:
    """Suma de las predicciones de todos los árboles sobre filas ya codificadas."""
    total = np.zeros(C.shape[0])
    for v, c, w in zip(variables, cortes, valores):
        total += predecir_arbol(v, c, w, C)
    return total


def _aportes_arbol(variable, corte, valor, cobertura, C: np.ndarray, aportes: np.ndarray) -> float:
    """Reparte la predicción de un árbol entre sus variables con valores de Shapley exactos.

    Parameters
    ----------
    variable, corte, valor, cobertura : ndarray
        Un árbol.
    C : ndarray de enteros de forma (n, p)
    aportes : ndarray de forma (n, p)
        Se le suma, en su lugar, el aporte de cada variable.

    Returns
    -------
    float
        Valor esperado del árbol (lo que predice sin conocer ninguna variable), que va a la base.

    Notes
    -----
    Sea ``U`` el conjunto de variables que usa el árbol (a lo sumo ``2^profundidad − 1``). Para un subconjunto ``S`` de
    ``U``, ``v(S)`` es la predicción esperada cuando solo se conocen las variables de ``S``: en un nodo que corta por una
    variable conocida se sigue la rama del cliente; en uno que corta por una desconocida se promedian las dos ramas con
    el peso de las filas de ajuste que tomó cada una. El aporte de la variable ``j`` es el promedio de
    ``v(S ∪ {j}) − v(S)`` con los pesos de Shapley, ``|S|!·(|U| − |S| − 1)!/|U|!``. Con árboles tan pequeños se
    enumeran todos los subconjuntos, así que el resultado es exacto: los aportes y la base suman la predicción.
    """
    usadas = sorted({int(v) for v in variable if v >= 0})
    n_nodos = len(variable)
    if not usadas:
        return float(valor[0])
    lugar = {v: i for i, v in enumerate(usadas)}
    m = len(usadas)

    def esperado(conocidas: int) -> np.ndarray:
        """``v(S)`` para cada fila; ``conocidas`` es la máscara de bits de ``S``."""
        nivel = [None] * n_nodos
        for k in range(n_nodos - 1, -1, -1):
            if variable[k] < 0:
                nivel[k] = valor[k]
                continue
            izq, der = nivel[2 * k + 1], nivel[2 * k + 2]
            if conocidas >> lugar[int(variable[k])] & 1:
                nivel[k] = np.where(C[:, variable[k]] <= corte[k], izq, der)
            else:
                nivel[k] = (cobertura[2 * k + 1] * izq + cobertura[2 * k + 2] * der) / max(cobertura[k], 1e-300)
        return np.broadcast_to(nivel[0], (C.shape[0],)).astype(np.float64)

    v = [esperado(s) for s in range(2 ** m)]
    factorial = [1.0]
    for i in range(1, m + 1):
        factorial.append(factorial[-1] * i)
    for j, var in enumerate(usadas):
        for s in range(2 ** m):
            if s >> j & 1:
                continue
            tam = bin(s).count('1')
            peso = factorial[tam] * factorial[m - tam - 1] / factorial[m]
            aportes[:, var] += peso * (v[s | (1 << j)] - v[s])
    return float(v[0][0])


# ============================================================================ 5. el regresor
class RegresorBDL(RegressorMixin, BaseEstimator):
    """Boosting con Desplazamiento Latente para un objetivo continuo.

    Parameters
    ----------
    n_vueltas : int, default 400
        Máximo de vueltas de EM. Cada vuelta hace un paso E, ``arboles_por_vuelta`` pasos de boosting y un paso M. La
        parada temprana suele usar menos.
    arboles_por_vuelta : int, default 5
        Árboles que se suman al índice en cada vuelta, todos contra el mismo objetivo sin escalón.
    tasa : float, default 0.3
        Tasa de aprendizaje: fracción de cada árbol que se suma al índice.
    profundidad : int, default 1
        Niveles de corte de cada árbol. Con 1 el índice es aditivo (una curva por variable).
    max_bins : int, default 255
        Máximo de bins por variable.
    min_hoja : int, default 20
        Mínimo de filas a cada lado de un corte.
    lambda_l2 : float, default 1.0
        Regularización L2 del valor de las hojas.
    rondas_calentamiento : int, default 100
        Máximo de árboles de boosting de mínimos cuadrados antes de buscar la mezcla: el residuo tiene que ser lo
        bastante pequeño para que los escalones se distingan. El calentamiento se corta antes si el error de
        validación deja de bajar, para que el sobreajuste no deforme los escalones.
    n_componentes : int or 'auto', default 'auto'
        Número de clases latentes ``K``. ``'auto'`` lo elige por BIC entre 1 y ``max_componentes``; con 1 el modelo es
        el boosting de mínimos cuadrados (la variante «sin mezcla» de la ablación).
    max_componentes : int, default 8
    fraccion_validacion : float, default 0.1
        Fracción de filas para la parada temprana y para medir la dispersión fuera de muestra.
    fraccion_calibracion : float, default 0.1
        Fracción de filas para la distribución predictiva conforme.
    paciencia : int, default 20
        Vueltas sin mejora en validación antes de detenerse.
    random_state : int or None, default 42
        Semilla del reparto de filas. Dado el reparto, el algoritmo es determinista.

    Attributes
    ----------
    intercepto_ : float
        Media del objetivo en las filas de ajuste; el índice arranca ahí.
    cortes_ : list of ndarray
        Cortes de cada variable.
    arboles_variable_, arboles_corte_, arboles_valor_, arboles_cobertura_ : ndarray
        Los árboles, uno por fila (el valor ya está multiplicado por la tasa).
    n_calentamiento_, n_arboles_ : int
        Árboles del calentamiento y total tras la parada temprana.
    n_componentes_ : int
        ``K`` elegido.
    pi_, centros_ : ndarray de forma (K,)
        Pesos y desplazamientos de la mezcla.
    sigma_ : float
        Dispersión dentro de cada clase, en las filas de ajuste.
    sigma_validacion_ : float
        Error dentro de cada clase en las filas de validación (fuera de muestra): raíz del error cuadrático medio.
    sigma_robusta_ : float
        Lo mismo con una medida robusta (mediana de las desviaciones absolutas): la dispersión del cliente típico, sin
        los pocos clientes en los que el índice se equivoca mucho. Es la que usa la probabilidad de excedencia.
    desplazamiento_medio_ : float
        ``Σ_k π_k c_k``: lo que se suma al índice para predecir la media.
    residuos_calibracion_ : ndarray
        Residuos ``s − F(x)`` de las filas de calibración, ordenados.
    historial_ : dict
        ``verosimilitud`` (al empezar cada vuelta de EM, en las filas de ajuste), ``error_validacion`` (al terminar
        cada vuelta), ``error_calentamiento`` (tras cada árbol del calentamiento) y ``bic`` (tabla de la elección de ``K``).
    formas_ : list of ndarray or None
        Con ``profundidad = 1``, la curva de cada variable (un valor por bin); si no, ``None``.
    """

    def __init__(self, n_vueltas: int = 400, arboles_por_vuelta: int = 5, tasa: float = 0.3, profundidad: int = 1,
                 max_bins: int = 255, min_hoja: int = 20, lambda_l2: float = 1.0, rondas_calentamiento: int = 100,
                 n_componentes='auto', max_componentes: int = 8, fraccion_validacion: float = 0.1,
                 fraccion_calibracion: float = 0.1, paciencia: int = 20, random_state: int | None = 42) -> None:
        self.n_vueltas = n_vueltas
        self.arboles_por_vuelta = arboles_por_vuelta
        self.tasa = tasa
        self.profundidad = profundidad
        self.max_bins = max_bins
        self.min_hoja = min_hoja
        self.lambda_l2 = lambda_l2
        self.rondas_calentamiento = rondas_calentamiento
        self.n_componentes = n_componentes
        self.max_componentes = max_componentes
        self.fraccion_validacion = fraccion_validacion
        self.fraccion_calibracion = fraccion_calibracion
        self.paciencia = paciencia
        self.random_state = random_state

    # ------------------------------------------------------------------ ajuste
    def fit(self, X, y):
        """Aprende el índice, la mezcla de desplazamientos y la distribución predictiva.

        Parameters
        ----------
        X : array-like de forma (n, p)
            Predictores numéricos, sin faltantes.
        y : array-like de forma (n,)
            Objetivo continuo.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Si hay faltantes o infinitos, si un hiperparámetro no es válido o si las filas no alcanzan para el reparto.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        _validar_comunes(self)
        if not 0.0 < self.fraccion_calibracion < 0.5:
            raise ValueError('fraccion_calibracion debe estar en (0, 0.5)')
        if self.n_componentes != 'auto' and not (isinstance(self.n_componentes, (int, np.integer)) and self.n_componentes >= 1):
            raise ValueError("n_componentes debe ser 'auto' o un entero positivo")
        self.n_features_in_ = X.shape[1]
        i_aj, i_val, i_cal = _repartir_filas(len(y), self.fraccion_validacion, self.fraccion_calibracion, self.random_state)
        if len(i_cal) < 2:
            raise ValueError('la calibración necesita al menos dos filas')

        # 1. discretización (los cortes se aprenden solo con las filas de ajuste)
        self.cortes_ = calcular_cortes(X[i_aj], int(self.max_bins))
        C = codificar(X, self.cortes_)
        C_aj, C_val = C[i_aj], C[i_val]
        hist = _Histogramas(C_aj, np.array([len(c) + 1 for c in self.cortes_]))
        y_aj, y_val = y[i_aj], y[i_val]
        escala = float(np.std(y_aj)) or 1.0
        sigma_min = 1e-6 * escala

        # 2. calentamiento: boosting de mínimos cuadrados (K = 1) hasta que el error de validación deja de bajar
        self.intercepto_ = float(y_aj.mean())
        F_aj, F_val = np.full(len(y_aj), self.intercepto_), np.full(len(y_val), self.intercepto_)
        bosque = _Bosque()
        error_calentamiento = []
        optimo = {'error': float(np.sqrt(np.mean((y_val - F_val) ** 2))), 'arboles': 0, 'F_aj': F_aj, 'F_val': F_val}
        for _ in range(int(self.rondas_calentamiento)):
            F_aj, F_val = self._ronda(hist, C_aj, C_val, y_aj - F_aj, F_aj, F_val, bosque)
            error_calentamiento.append(float(np.sqrt(np.mean((y_val - F_val) ** 2))))
            if error_calentamiento[-1] < optimo['error']:
                optimo = {'error': error_calentamiento[-1], 'arboles': len(bosque.valor), 'F_aj': F_aj, 'F_val': F_val}
            elif len(bosque.valor) - optimo['arboles'] >= PACIENCIA_CALENTAMIENTO:
                break
        bosque.recortar(optimo['arboles'])
        F_aj, F_val = optimo['F_aj'], optimo['F_val']
        self.n_calentamiento_ = len(bosque.valor)

        # 3. mezcla inicial sobre el residuo del calentamiento
        r = y_aj - F_aj
        if self.n_componentes == 'auto':
            K, tabla_bic = elegir_componentes(r, int(self.max_componentes), sigma_min)
        else:
            K, tabla_bic = int(self.n_componentes), []
        pi, c, sigma, _ = ajustar_mezcla(r, K, sigma_min)
        K = len(c)

        # 4. EM generalizado: paso E, pasos de boosting sobre el objetivo sin escalón, paso M
        verosimilitud, error_validacion = [], []
        mejor = {'error': _error_intraclase(y_val - F_val, c), 'arboles': len(bosque.valor), 'pi': pi, 'c': c, 'sigma': sigma,
                 'F_val': F_val}
        sin_mejora = 0
        for _ in range(int(self.n_vueltas)):
            gamma, log_fila = paso_e(y_aj - F_aj, pi, c, sigma)             # paso E
            verosimilitud.append(float(log_fila.sum()))
            sin_escalon = y_aj - gamma @ c                                  # objetivo al que se le quitó el escalón esperado
            for _ in range(int(self.arboles_por_vuelta)):                   # pasos de boosting
                F_aj, F_val = self._ronda(hist, C_aj, C_val, sin_escalon - F_aj, F_aj, F_val, bosque)
            pi, c, sigma = paso_m(y_aj - F_aj, gamma, c, sigma_min)         # paso M
            error = _error_intraclase(y_val - F_val, c)
            error_validacion.append(error)
            if error < mejor['error'] * (1.0 - 1e-6):
                mejor = {'error': error, 'arboles': len(bosque.valor), 'pi': pi, 'c': c, 'sigma': sigma, 'F_val': F_val}
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= int(self.paciencia):
                    break

        # 5. se queda con la mejor ronda en validación
        bosque.recortar(mejor['arboles'])
        (self.arboles_variable_, self.arboles_corte_, self.arboles_valor_, self.arboles_cobertura_) = bosque.matrices()
        self.n_arboles_ = mejor['arboles']
        self.n_componentes_ = K
        orden = np.argsort(mejor['c'])
        self.pi_, self.centros_, self.sigma_ = mejor['pi'][orden], mejor['c'][orden], float(mejor['sigma'])
        self.sigma_validacion_ = max(float(mejor['error']), sigma_min)
        self.sigma_robusta_ = max(_dispersion_robusta(_errores_intraclase(y_val - mejor['F_val'], mejor['c'])), sigma_min)
        self.desplazamiento_medio_ = float(self.pi_ @ self.centros_)
        self.historial_ = {'verosimilitud': verosimilitud, 'error_validacion': error_validacion,
                           'error_calentamiento': error_calentamiento, 'bic': tabla_bic}
        self._compilar_formas(hist)

        # 6. distribución predictiva conforme: residuos de filas que no entraron en los árboles ni en la parada
        self.residuos_calibracion_ = np.sort(y[i_cal] - self._indice_codigos(C[i_cal]))
        self.n_ajuste_, self.n_validacion_, self.n_calibracion_ = len(i_aj), len(i_val), len(i_cal)
        return self

    def _ronda(self, hist, C_aj, C_val, residuo, F_aj, F_val, bosque):
        """Ajusta un árbol al residuo, lo guarda multiplicado por la tasa y actualiza el índice."""
        variable, corte, valor, cobertura, nodo = ajustar_arbol(hist, C_aj, residuo, int(self.profundidad),
                                                                 int(self.min_hoja), float(self.lambda_l2))
        valor = self.tasa * valor
        bosque.agregar(variable, corte, valor, cobertura)
        return F_aj + valor[nodo], F_val + predecir_arbol(variable, corte, valor, C_val)

    def _compilar_formas(self, hist: _Histogramas) -> None:
        """Con árboles de un solo corte, suma los árboles de cada variable en una curva por bin."""
        self.conteo_bins_ = [hist.conteo_raiz[hist.inicio[j]:hist.inicio[j + 1]].copy() for j in range(len(self.cortes_))]
        if int(self.profundidad) != 1:
            self.formas_, self.constante_formas_ = None, 0.0
            return
        formas = [np.zeros(len(c) + 1) for c in self.cortes_]
        constante = 0.0
        for variable, corte, valor in zip(self.arboles_variable_, self.arboles_corte_, self.arboles_valor_):
            if variable[0] < 0:
                constante += valor[0]
            else:
                formas[variable[0]][:corte[0] + 1] += valor[1]
                formas[variable[0]][corte[0] + 1:] += valor[2]
        self.formas_, self.constante_formas_ = formas, float(constante)

    # ------------------------------------------------------------------ predicción
    def _indice_codigos(self, C: np.ndarray) -> np.ndarray:
        """Índice ``F(x)`` de filas ya codificadas."""
        if self.formas_ is not None:
            total = np.full(C.shape[0], self.intercepto_ + self.constante_formas_)
            for j, forma in enumerate(self.formas_):
                total += forma[C[:, j]]
            return total
        return self.intercepto_ + _sumar_arboles(self.arboles_variable_, self.arboles_corte_, self.arboles_valor_, C)

    def _codigos(self, X) -> np.ndarray:
        check_is_fitted(self, 'residuos_calibracion_')
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X tiene {X.shape[1]} columnas y el modelo se ajustó con {self.n_features_in_}')
        return codificar(X, self.cortes_)

    def indice(self, X) -> np.ndarray:
        """Índice ``F(x)``: la parte del puntaje que explican los predictores, sin el desplazamiento latente."""
        return self._indice_codigos(self._codigos(X))

    def predict(self, X) -> np.ndarray:
        """Media predicha del objetivo: ``F(x) + Σ_k π_k c_k``."""
        return self.indice(X) + self.desplazamiento_medio_

    def prob_excedencia(self, X, umbral: float, metodo: str = 'mezcla') -> np.ndarray:
        """Probabilidad de que el objetivo supere ``umbral``.

        Parameters
        ----------
        X : array-like de forma (n, p)
        umbral : float
        metodo : {'mezcla', 'conforme'}
            ``'mezcla'`` (por defecto) usa el modelo: ``Σ_k π_k · Φ((F(x) + c_k − umbral) / σ)``, con ``σ`` la
            dispersión robusta de validación. Cada clase latente aporta su peso si su nivel ``F(x) + c_k`` queda por
            encima del umbral, y la transición se suaviza con la incertidumbre del índice. ``'conforme'`` no supone
            nada sobre la forma del ruido: es la fracción de residuos de calibración ``r_j`` con ``F(x) + r_j > umbral``.

        Returns
        -------
        ndarray de forma (n,)
            Entre ``0,5/(m + 1)`` y ``1 − 0,5/(m + 1)``, con ``m`` el número de residuos de calibración: el modelo no
            afirma más certeza que la que esa muestra permite comprobar.

        Notes
        -----
        Con clases bien separadas, ``'mezcla'`` da el mismo valor a todos los clientes cuyo índice cae en el mismo
        tramo entre dos escalones (una meseta), que es lo que corresponde: dentro de una meseta lo único que decide si
        el cliente supera el umbral es su clase latente, y esa no depende de ``x``. ``'conforme'`` reparte entre todos
        los clientes los errores grandes de unos pocos y, con ello, ordena por ``F(x)`` dentro de cada meseta.
        """
        F = self.indice(X)
        m = len(self.residuos_calibracion_)
        if metodo == 'conforme':
            mayores = m - np.searchsorted(self.residuos_calibracion_, umbral - F, side='right')
            return (mayores + 0.5) / (m + 1.0)
        if metodo != 'mezcla':
            raise ValueError("metodo debe ser 'mezcla' o 'conforme'")
        p = ndtr((F[:, None] + self.centros_[None, :] - umbral) / self.sigma_robusta_) @ self.pi_
        return np.clip(p, 0.5 / (m + 1.0), 1.0 - 0.5 / (m + 1.0))

    def cuantil(self, X, q: float) -> np.ndarray:
        """Cuantil ``q`` de la distribución predictiva (``q = 0,5`` es la mediana)."""
        m = len(self.residuos_calibracion_)
        posicion = int(np.clip(np.ceil((m + 1) * q) - 1, 0, m - 1))
        return self.indice(X) + self.residuos_calibracion_[posicion]

    def intervalo(self, X, cobertura: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Intervalo central de predicción con la cobertura pedida (predicción conforme con residuos con signo)."""
        alfa = 1.0 - cobertura
        return self.cuantil(X, alfa / 2.0), self.cuantil(X, 1.0 - alfa / 2.0)

    def umbral_densidad(self, cobertura: float = 0.9) -> float:
        """Nivel de densidad que deja dentro la fracción ``cobertura`` de los residuos de calibración.

        Notes
        -----
        Predicción conforme con la densidad de la mezcla como medida de conformidad: un valor es tanto más «normal»
        cuanto más densa es la mezcla en su residuo. El nivel es el ``⌊(1 − cobertura)(m + 1)⌋``-ésimo menor valor de
        la densidad en los ``m`` residuos de calibración, lo que garantiza cobertura ≥ ``cobertura`` para un cliente
        nuevo intercambiable con los de calibración. La densidad usa la dispersión medida en validación.
        """
        check_is_fitted(self, 'residuos_calibracion_')
        densidad = np.sort(densidad_mezcla(self.residuos_calibracion_, self.pi_, self.centros_, self.sigma_validacion_))
        k = int(np.floor((1.0 - cobertura) * (len(densidad) + 1)))
        return 0.0 if k < 1 else float(densidad[k - 1])

    def conjunto_residual(self, cobertura: float = 0.9, puntos_por_sigma: int = 40) -> np.ndarray:
        """Conjunto de predicción en la escala del residuo: unión de intervalos alrededor de los desplazamientos.

        Parameters
        ----------
        cobertura : float
        puntos_por_sigma : int
            Resolución de la rejilla con la que se localizan los bordes.

        Returns
        -------
        ndarray de forma (n_intervalos, 2)
            Extremos de cada intervalo. El conjunto de predicción de un cliente es ``F(x)`` más estos intervalos.
        """
        nivel = self.umbral_densidad(cobertura)
        s = self.sigma_validacion_
        minimo = min(self.centros_.min() - 10 * s, self.residuos_calibracion_[0])
        maximo = max(self.centros_.max() + 10 * s, self.residuos_calibracion_[-1])
        if nivel <= 0.0:
            return np.array([[minimo, maximo]])
        n_puntos = int(min(max((maximo - minimo) / s * puntos_por_sigma, 200), 400000))
        rejilla = np.linspace(minimo, maximo, n_puntos)
        dentro = densidad_mezcla(rejilla, self.pi_, self.centros_, s) >= nivel
        cambio = np.diff(np.concatenate(([0], dentro.astype(np.int8), [0])))
        return np.c_[rejilla[np.flatnonzero(cambio == 1)], rejilla[np.flatnonzero(cambio == -1) - 1]]

    def largo_conjunto(self, cobertura: float = 0.9) -> float:
        """Largo total del conjunto de predicción (la suma de los largos de sus intervalos)."""
        tramos = self.conjunto_residual(cobertura)
        return float((tramos[:, 1] - tramos[:, 0]).sum())

    def conjunto_cubre(self, X, y, cobertura: float = 0.9) -> np.ndarray:
        """Indica, fila a fila, si el conjunto de predicción contiene el valor observado."""
        residuo = np.asarray(y, dtype=np.float64) - self.indice(X)
        return densidad_mezcla(residuo, self.pi_, self.centros_, self.sigma_validacion_) >= self.umbral_densidad(cobertura)

    # ------------------------------------------------------------------ interpretación
    def contribuciones(self, X) -> tuple[np.ndarray, float]:
        """Aporte de cada variable a la predicción de cada fila (valores de Shapley exactos).

        Returns
        -------
        aportes : ndarray de forma (n, p)
        base : float
            Predicción media. ``base + aportes.sum(axis=1)`` es igual a ``predict(X)``.

        Notes
        -----
        Con ``profundidad = 1`` el índice es aditivo y el aporte de una variable es su curva menos la media de esa
        curva en las filas de ajuste. Con árboles más profundos se reparte cada árbol con :func:`_aportes_arbol`.
        """
        C = self._codigos(X)
        aportes = np.zeros(C.shape, dtype=np.float64)
        base = self.intercepto_ + self.desplazamiento_medio_
        if self.formas_ is not None:
            base += self.constante_formas_
            for j, forma in enumerate(self.formas_):
                media = float(forma @ self.conteo_bins_[j] / self.conteo_bins_[j].sum())
                aportes[:, j] = forma[C[:, j]] - media
                base += media
            return aportes, base
        for arbol in zip(self.arboles_variable_, self.arboles_corte_, self.arboles_valor_, self.arboles_cobertura_):
            base += _aportes_arbol(*arbol, C, aportes)
        return aportes, base

    def importancia(self) -> np.ndarray:
        """Importancia global de cada variable: media del valor absoluto de su aporte en las filas de ajuste.

        Solo con ``profundidad = 1`` (se calcula con las curvas y el conteo de filas por bin, sin volver a los datos).
        """
        check_is_fitted(self, 'residuos_calibracion_')
        if self.formas_ is None:
            raise ValueError('importancia() requiere profundidad = 1; use contribuciones(X) con una muestra')
        salida = np.zeros(len(self.formas_))
        for j, forma in enumerate(self.formas_):
            peso = self.conteo_bins_[j] / self.conteo_bins_[j].sum()
            salida[j] = float(np.abs(forma - forma @ peso) @ peso)
        return salida


# ============================================================================ 6. el clasificador
class ClasificadorBDL(ClassifierMixin, BaseEstimator):
    """Clasificador de «el puntaje supera el umbral» con Boosting con Desplazamiento Latente.

    Parameters
    ----------
    n_vueltas, arboles_por_vuelta, tasa, profundidad, max_bins, min_hoja, lambda_l2, rondas_calentamiento,
    n_componentes, max_componentes, fraccion_validacion, fraccion_calibracion, paciencia, random_state
        Los de :class:`RegresorBDL`. En modo directo no hay mezcla: el máximo de árboles es
        ``rondas_calentamiento + n_vueltas · arboles_por_vuelta`` y la paciencia se cuenta en esas mismas unidades.
    usar_puntaje : bool, default True
        Si es ``False``, el modelo ignora el puntaje aunque ``fit`` lo reciba y aprende solo de la etiqueta (modo
        directo). Permite que la búsqueda de hiperparámetros decida si el puntaje ayuda, y es la variante «sin
        puntaje» de la ablación.

    Attributes
    ----------
    classes_ : ndarray de forma (2,)
        Las dos clases; la segunda es la positiva.
    modo_ : {'excedencia', 'directo'}
        ``'excedencia'`` si ``fit`` recibió el puntaje continuo; ``'directo'`` si solo recibió la etiqueta.
    umbral_ : float
        Solo en modo excedencia: umbral del puntaje que separa las dos clases.
    regresor_ : RegresorBDL
        Solo en modo excedencia.
    historial_ : dict
        En modo directo, ``perdida`` (pérdida logística media en las filas de ajuste tras cada ronda) y
        ``perdida_validacion``.

    Notes
    -----
    **Modo excedencia.** La etiqueta es ``y = 1[s > τ]``. Un clasificador corriente solo ve ``y`` y desecha cuánto le
    falta o le sobra a cada cliente para llegar al umbral. Aquí se ajusta un :class:`RegresorBDL` al puntaje y la
    probabilidad de la clase positiva es ``P(s > τ | x)``, leída de su distribución predictiva.

    **Modo directo.** Sin puntaje, el mismo motor de árboles minimiza la pérdida logística. Cada hoja da el paso
    ``Σ(y − p) / (n_hoja/4 + λ)``: como la segunda derivada de la pérdida logística nunca supera 1/4, ese paso minimiza
    una cota cuadrática de la pérdida y por eso la pérdida de ajuste no puede subir (``test_perdida_logistica_no_sube``).
    """

    def __init__(self, n_vueltas: int = 400, arboles_por_vuelta: int = 5, tasa: float = 0.3, profundidad: int = 1,
                 max_bins: int = 255, min_hoja: int = 20, lambda_l2: float = 1.0, rondas_calentamiento: int = 100,
                 n_componentes='auto', max_componentes: int = 8, fraccion_validacion: float = 0.1,
                 fraccion_calibracion: float = 0.1, paciencia: int = 20, usar_puntaje: bool = True,
                 random_state: int | None = 42) -> None:
        self.n_vueltas = n_vueltas
        self.arboles_por_vuelta = arboles_por_vuelta
        self.tasa = tasa
        self.profundidad = profundidad
        self.max_bins = max_bins
        self.min_hoja = min_hoja
        self.lambda_l2 = lambda_l2
        self.rondas_calentamiento = rondas_calentamiento
        self.n_componentes = n_componentes
        self.max_componentes = max_componentes
        self.fraccion_validacion = fraccion_validacion
        self.fraccion_calibracion = fraccion_calibracion
        self.paciencia = paciencia
        self.usar_puntaje = usar_puntaje
        self.random_state = random_state

    def fit(self, X, y, puntaje=None):
        """Ajusta el clasificador.

        Parameters
        ----------
        X : array-like de forma (n, p)
        y : array-like de forma (n,)
            Etiqueta binaria.
        puntaje : array-like de forma (n,), optional
            Puntaje continuo del que sale la etiqueta (``y`` es positiva exactamente cuando el puntaje supera un
            umbral). Si se da, el modelo aprende del puntaje; si no, de la etiqueta.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Si ``y`` no tiene exactamente dos clases, si hay faltantes, o si ``puntaje`` no separa las dos clases con
            un umbral.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        _validar_comunes(self)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f'se necesitan exactamente dos clases y hay {len(self.classes_)}')
        self.n_features_in_ = X.shape[1]
        positiva = y == self.classes_[1]
        if puntaje is None or not self.usar_puntaje:
            self.modo_ = 'directo'
            self._ajustar_directo(X, positiva.astype(np.float64))
            return self
        puntaje = np.asarray(puntaje, dtype=np.float64).ravel()
        if len(puntaje) != len(y) or not np.isfinite(puntaje).all():
            raise ValueError('puntaje debe tener un valor finito por fila')
        bajo, alto = puntaje[~positiva].max(), puntaje[positiva].min()
        if not bajo < alto:
            raise ValueError('la etiqueta no es un umbral del puntaje: hay negativos con puntaje mayor o igual que algún positivo')
        self.modo_ = 'excedencia'
        self.umbral_ = float((bajo + alto) / 2.0)
        parametros = {k: v for k, v in self.get_params().items() if k != 'usar_puntaje'}
        self.regresor_ = RegresorBDL(**parametros).fit(X, puntaje)
        return self

    def _ajustar_directo(self, X: np.ndarray, y: np.ndarray) -> None:
        """Boosting logístico con el motor de árboles propio y pasos que minimizan una cota de la pérdida."""
        i_aj, i_val, _ = _repartir_filas(len(y), self.fraccion_validacion, 0.0, self.random_state)
        self.cortes_ = calcular_cortes(X[i_aj], int(self.max_bins))
        C = codificar(X, self.cortes_)
        C_aj, C_val, y_aj, y_val = C[i_aj], C[i_val], y[i_aj], y[i_val]
        hist = _Histogramas(C_aj, np.array([len(c) + 1 for c in self.cortes_]))
        prevalencia = float(np.clip(y_aj.mean(), 1e-6, 1 - 1e-6))
        self.intercepto_ = float(np.log(prevalencia / (1.0 - prevalencia)))
        F_aj, F_val = np.full(len(y_aj), self.intercepto_), np.full(len(y_val), self.intercepto_)
        bosque = _Bosque()
        perdida, perdida_validacion = [], []
        mejor, sin_mejora = {'perdida': _perdida_logistica(y_val, F_val), 'arboles': 0}, 0
        max_arboles = int(self.rondas_calentamiento) + int(self.n_vueltas) * int(self.arboles_por_vuelta)
        for _ in range(max_arboles):
            residuo = 4.0 * (y_aj - _sigmoide(F_aj))             # 4·(y − p): con λ' = 4λ, la hoja vale Σ(y − p)/(n/4 + λ)
            variable, corte, valor, cobertura, nodo = ajustar_arbol(hist, C_aj, residuo, int(self.profundidad),
                                                                     int(self.min_hoja), 4.0 * float(self.lambda_l2))
            valor = self.tasa * valor
            bosque.agregar(variable, corte, valor, cobertura)
            F_aj = F_aj + valor[nodo]
            F_val = F_val + predecir_arbol(variable, corte, valor, C_val)
            perdida.append(_perdida_logistica(y_aj, F_aj))
            actual = _perdida_logistica(y_val, F_val)
            perdida_validacion.append(actual)
            if actual < mejor['perdida'] * (1.0 - 1e-7):
                mejor, sin_mejora = {'perdida': actual, 'arboles': len(bosque.valor)}, 0
            else:
                sin_mejora += 1
                if sin_mejora >= int(self.paciencia) * int(self.arboles_por_vuelta):
                    break
        bosque.recortar(mejor['arboles'])
        (self.arboles_variable_, self.arboles_corte_, self.arboles_valor_, self.arboles_cobertura_) = bosque.matrices()
        self.n_arboles_ = mejor['arboles']
        self.historial_ = {'perdida': perdida, 'perdida_validacion': perdida_validacion}

    def decision_function(self, X) -> np.ndarray:
        """Puntaje de decisión: ``F(x) − τ`` en modo excedencia; el logit en modo directo."""
        check_is_fitted(self, 'modo_')
        if self.modo_ == 'excedencia':
            return self.regresor_.indice(X) - self.umbral_
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X tiene {X.shape[1]} columnas y el modelo se ajustó con {self.n_features_in_}')
        C = codificar(X, self.cortes_)
        return self.intercepto_ + _sumar_arboles(self.arboles_variable_, self.arboles_corte_, self.arboles_valor_, C)

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidad de cada clase.

        Returns
        -------
        ndarray de forma (n, 2)
            La segunda columna es ``P(s > τ | x)`` en modo excedencia y ``sigmoide(F(x))`` en modo directo.
        """
        check_is_fitted(self, 'modo_')
        if self.modo_ == 'excedencia':
            p = self.regresor_.prob_excedencia(X, self.umbral_)
        else:
            p = _sigmoide(self.decision_function(X))
        return np.c_[1.0 - p, p]

    def predict(self, X) -> np.ndarray:
        """Clase más probable."""
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


def _sigmoide(f: np.ndarray) -> np.ndarray:
    """Función logística, estable para valores grandes en valor absoluto."""
    f = np.clip(f, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-f))


def _perdida_logistica(y: np.ndarray, f: np.ndarray) -> float:
    """Pérdida logística media: ``log(1 + e^f) − y·f``."""
    return float(np.mean(np.logaddexp(0.0, f) - y * f))


ModeloOriginal = ClasificadorBDL
"""Nombre con el que la guía del curso se refiere a la clase del modelo original."""
