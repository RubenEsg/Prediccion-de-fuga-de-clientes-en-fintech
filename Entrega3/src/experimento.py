"""Orquestación: una corrida ``(tarea, modelo, balanceo, optimizador)`` y el producto completo.

Con 7 modelos × 4 balanceos × 4 optimizadores en clasificación y 7 × 4 en regresión el
producto es de 140 corridas (guía §2). :func:`correr_todo` las recorre con
``itertools.product``, registra cada una en la tabla maestra a medida que termina
—también las que no aplican y las que fallan, con su motivo— y salta las que ya existen,
de modo que puede interrumpirse y reanudarse.

Cada corrida deja además su **modelo final** (reajustado sobre todo el entrenamiento con
los hiperparámetros finales) serializado en ``resultados/modelos/``, para que la
evaluación en prueba, la calibración y la interpretabilidad no tengan que repetir la
búsqueda.
"""
from __future__ import annotations

import itertools
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import joblib
import pandas as pd

from . import balanceo as bal
from . import evaluacion
from .config import CARPETA_RESULTADOS, K_EXT, K_INT, SEED
from .modelos import TAREAS, catalogo
from .optimizacion import OPTIMIZADORES, presupuesto_por_defecto, validacion_anidada
from .registro import Registro


def combinaciones(tarea: str, modelos: Iterable[str] | None = None,
                  balanceos: Iterable[str] | None = None,
                  optimizadores: Iterable[str] | None = None) -> list[tuple[str, str, str, str]]:
    """Producto cartesiano de la tarea.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    modelos, balanceos, optimizadores : iterable of str, optional
        Subconjuntos a recorrer; por defecto los catálogos completos. En regresión la
        etapa de balanceo se omite (solo ``'ninguno'``), como fija la guía §2.

    Returns
    -------
    list of tuple
        ``(tarea, modelo, balanceo, optimizador)``: 112 tuplas en clasificación y 28 en
        regresión con los catálogos completos.
    """
    modelos = list(modelos or catalogo(tarea).keys())
    optimizadores = list(optimizadores or OPTIMIZADORES)
    if tarea == 'regresion':
        balanceos = ['ninguno']
    else:
        balanceos = list(balanceos or bal.CATALOGO_BALANCEO)
    return [(tarea, m, b, o) for m, b, o in itertools.product(modelos, balanceos, optimizadores)]


def nombre_modelo(tarea: str, modelo: str, balanceo: str, optimizador: str, seed: int) -> str:
    """Nombre de archivo del modelo final de una combinación."""
    return f'{tarea}__{modelo}__{balanceo}__{optimizador}__{seed}.joblib'


def cargar_modelo(tarea: str, modelo: str, balanceo: str, optimizador: str, seed: int = SEED,
                  carpeta: Path | str | None = None):
    """Carga el modelo final serializado de una combinación.

    Parameters
    ----------
    tarea, modelo, balanceo, optimizador : str
    seed : int
    carpeta : Path or str, optional
        Por defecto ``resultados/modelos``.

    Returns
    -------
    Pipeline
        Ajustado sobre todo el entrenamiento.
    """
    carpeta = Path(carpeta) if carpeta is not None else CARPETA_RESULTADOS / 'modelos'
    return joblib.load(carpeta / nombre_modelo(tarea, modelo, balanceo, optimizador, seed))


def correr_experimento(tarea: str, modelo: str, balanceo: str, optimizador: str,
                       X: pd.DataFrame, y: pd.Series, seed: int = SEED, k_ext: int = K_EXT,
                       k_int: int = K_INT, presupuesto: int | None = None,
                       registro: Registro | None = None, sobrescribir: bool = False,
                       n_jobs: int = -1, etiqueta: dict | None = None,
                       guardar_modelo: bool = True, carpeta_modelos: Path | str | None = None) -> dict:
    """Ejecuta una combinación con validación anidada y la registra.

    Parameters
    ----------
    tarea, modelo, balanceo, optimizador : str
    X : DataFrame
        Predictores crudos del conjunto de entrenamiento.
    y : Series
        Objetivo de la tarea: etiqueta binaria en clasificación, valor continuo en regresión.
    seed : int
    k_ext, k_int : int
        Pliegues externos e internos.
    presupuesto : int, optional
        Ver :func:`optimizacion.buscar`.
    registro : Registro, optional
        Si se da, la fila se añade a la tabla maestra.
    sobrescribir : bool
        Reemplazar una fila existente con la misma clave.
    n_jobs : int
    etiqueta : dict, optional
        Metadatos de la etiqueta (``metodo``, ``umbral``, ``q``) que se copian a la fila.
    guardar_modelo : bool
        Serializar el modelo final con ``joblib`` en ``carpeta_modelos``.
    carpeta_modelos : Path or str, optional
        Por defecto ``resultados/modelos``.

    Returns
    -------
    dict
        La fila registrada. ``estado`` es ``'ok'``, o ``'no_aplica'`` cuando la
        combinación no tiene sentido (p. ej. KNN × class_weight, o sobremuestreo con
        clases ya equilibradas); en ese caso lleva ``motivo`` y no tiene métricas.

    Raises
    ------
    ValueError
        Si la tarea no es válida.
    """
    if tarea not in TAREAS:
        raise ValueError(f'tarea desconocida: {tarea!r}')
    spec = catalogo(tarea, incluir_nuevo=True)[modelo]
    fila = {'tarea': tarea, 'modelo': modelo, 'balanceo': balanceo, 'optimizador': optimizador,
            'semilla': seed, 'k_ext': k_ext, 'k_int': k_int,
            'metrica_principal': evaluacion.NOMBRE_METRICA[tarea],
            'etiqueta_metodo': (etiqueta or {}).get('metodo'), 'etiqueta_umbral': (etiqueta or {}).get('umbral'),
            'n_train': int(len(X)), 'n_columnas': int(X.shape[1]),
            'presupuesto': presupuesto if presupuesto is not None else presupuesto_por_defecto(spec.espacio_grid)}
    ok, motivo = bal.aplica(spec, balanceo, y if tarea == 'clasificacion' else None)
    if not ok:
        fila.update({'aplica': False, 'estado': 'no_aplica', 'motivo': motivo, 'media_ext': None,
                     'desv_ext': None, 'n_evaluaciones': 0, 'tiempo_total_s': 0.0,
                     'tiempo_por_evaluacion_s': 0.0})
        if registro is not None:
            registro.anadir(fila, sobrescribir=sobrescribir)
        return fila

    t0 = time.perf_counter()
    fidelidad = spec.fidelidad or {}
    res = validacion_anidada(lambda y_pliegue: bal.construir_pipeline(spec, balanceo, X, seed, y_pliegue),
                             spec.espacio_grid, spec.espacio, optimizador, X, y, tarea,
                             k_ext=k_ext, k_int=k_int, presupuesto=presupuesto, seed=seed, n_jobs=n_jobs,
                             fijos_busqueda=fidelidad.get('busqueda'), fijos_final=fidelidad.get('final'))
    tiempo = time.perf_counter() - t0
    ruta_modelo = ''
    if guardar_modelo and res.estimador_final is not None:
        carpeta = Path(carpeta_modelos) if carpeta_modelos is not None else CARPETA_RESULTADOS / 'modelos'
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_modelo = str(carpeta / nombre_modelo(tarea, modelo, balanceo, optimizador, seed))
        joblib.dump(res.estimador_final, ruta_modelo, compress=3)
    fila.update({
        'aplica': True, 'estado': 'ok', 'motivo': '',
        'media_ext': res.media_ext, 'desv_ext': res.desv_ext,
        'puntajes_ext': res.puntajes_ext, 'metricas_ext': res.metricas_ext,
        'mejor_interno_medio': float(sum(res.interno_por_pliegue) / len(res.interno_por_pliegue)),
        'optimismo_seleccion': res.optimismo_seleccion,
        'hiperparametros': res.params_finales, 'params_por_pliegue': res.params_por_pliegue,
        'mejor_interno_final': res.interno_final,
        'n_evaluaciones': res.n_evaluaciones, 'n_evaluaciones_final': res.n_evaluaciones_final,
        'tiempo_total_s': tiempo, 'tiempo_busqueda_s': res.tiempo_total_s,
        'tiempo_por_evaluacion_s': res.tiempo_total_s / max(res.n_evaluaciones, 1),
        'tiempo_ajuste_final_s': res.tiempo_final_s,
        'trazas': res.trazas, 'detalles': res.detalles, 'detalle': res.detalle_final,
        'fidelidad': spec.fidelidad, 'ruta_modelo': ruta_modelo,
    })
    if registro is not None:
        registro.anadir(fila, sobrescribir=sobrescribir)
    return fila


def correr_todo(X: pd.DataFrame, y: pd.Series, tarea: str, registro: Registro,
                modelos: Iterable[str] | None = None, balanceos: Iterable[str] | None = None,
                optimizadores: Iterable[str] | None = None, seed: int = SEED,
                k_ext: int = K_EXT, k_int: int = K_INT, presupuesto: int | None = None,
                saltar_existentes: bool = True, continuar_ante_error: bool = True,
                reintentar_errores: bool = True,
                n_jobs: int = -1, etiqueta: dict | None = None, guardar_modelo: bool = True,
                carpeta_modelos: Path | str | None = None,
                informar: Callable[[str], None] = print) -> list[dict]:
    """Recorre el producto cartesiano de una tarea registrando cada corrida al terminar.

    Parameters
    ----------
    X, y : DataFrame, Series
        Conjunto de entrenamiento y objetivo de la tarea.
    tarea : {'clasificacion', 'regresion'}
    registro : Registro
        Tabla maestra donde se anota cada corrida.
    modelos, balanceos, optimizadores : iterable of str, optional
        Subconjuntos a recorrer (ver :func:`combinaciones`).
    seed, k_ext, k_int, presupuesto, n_jobs
        Ver :func:`correr_experimento`.
    saltar_existentes : bool
        Si ``True``, las combinaciones ya presentes en ``registro`` (misma semilla) no se
        vuelven a ejecutar: permite reanudar.
    continuar_ante_error : bool
        Si ``True``, una excepción en una combinación se registra como fila con
        ``estado='error'`` (y el mensaje en ``motivo``) y el bucle continúa; si ``False``
        la excepción se propaga.
    reintentar_errores : bool
        Si ``True``, las combinaciones registradas con ``estado='error'`` no se saltan: se
        vuelven a ejecutar y su fila se reemplaza. Así un fallo pasajero (memoria, un apagón)
        se corrige al reanudar.
    etiqueta : dict, optional
        Metadatos de la etiqueta que se copian a cada fila.
    guardar_modelo : bool
    carpeta_modelos : Path or str, optional
    informar : callable
        Receptor de mensajes de progreso (``print`` por defecto).

    Returns
    -------
    list of dict
        Filas ejecutadas en esta llamada (no incluye las saltadas).
    """
    plan = combinaciones(tarea, modelos, balanceos, optimizadores)
    hechas = []
    for i, (t, m, b, o) in enumerate(plan, 1):
        etiqueta_log = f'[{i:3d}/{len(plan)}] {t} · {m} · {b} · {o}'
        previo = registro.estado(tarea=t, modelo=m, balanceo=b, optimizador=o, semilla=seed)
        if saltar_existentes and previo is not None and not (reintentar_errores and previo == 'error'):
            informar(f'{etiqueta_log}: ya registrada, se salta')
            continue
        t0 = time.perf_counter()
        try:
            fila = correr_experimento(t, m, b, o, X, y, seed=seed, k_ext=k_ext, k_int=k_int,
                                      presupuesto=presupuesto, registro=registro, n_jobs=n_jobs,
                                      etiqueta=etiqueta, guardar_modelo=guardar_modelo,
                                      carpeta_modelos=carpeta_modelos, sobrescribir=previo is not None)
        except Exception as ex:                                    # noqa: BLE001
            if not continuar_ante_error:
                raise
            fila = {'tarea': t, 'modelo': m, 'balanceo': b, 'optimizador': o, 'semilla': seed,
                    'aplica': False, 'estado': 'error',
                    'motivo': f'{type(ex).__name__}: {ex}'[:500],
                    'traza_error': traceback.format_exc()[-2000:],
                    'media_ext': None, 'desv_ext': None, 'n_evaluaciones': 0,
                    'tiempo_total_s': time.perf_counter() - t0}
            registro.anadir(fila, sobrescribir=True)
            informar(f'{etiqueta_log}: ERROR {fila["motivo"][:120]}')
            hechas.append(fila)
            continue
        if fila['estado'] == 'ok':
            informar(f'{etiqueta_log}: {fila["metrica_principal"]} = {abs(fila["media_ext"]):.4f} '
                     f'± {fila["desv_ext"]:.4f} ({fila["n_evaluaciones"]} evals + {fila["n_evaluaciones_final"]} '
                     f'finales, {time.perf_counter() - t0:.0f} s)')
        else:
            informar(f'{etiqueta_log}: no aplica ({fila["motivo"]})')
        hechas.append(fila)
    return hechas
