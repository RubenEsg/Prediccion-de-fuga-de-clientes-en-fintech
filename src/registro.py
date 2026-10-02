"""Tabla maestra de experimentos (guía §2 y §7.3).

Una fila por combinación ``(tarea, modelo, balanceo, optimizador, semilla)`` con los
hiperparámetros finales, las métricas del bucle externo, el número de evaluaciones, los
tiempos y las trazas de convergencia. Se escribe en disco tras **cada** corrida, de modo
que el producto de 140 combinaciones puede interrumpirse y reanudarse sin perder nada.

Cada fila lleva un ``estado``: ``'ok'`` (ejecutada), ``'no_aplica'`` (la combinación no
tiene sentido, con el motivo) o ``'error'`` (la corrida lanzó una excepción, con el
mensaje), de modo que el producto completo queda documentado aunque algo falle.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CARPETA_RESULTADOS

CLAVE: tuple[str, ...] = ('tarea', 'modelo', 'balanceo', 'optimizador', 'semilla')
"""Columnas que identifican una corrida de forma única."""

COLUMNAS_JSON: tuple[str, ...] = ('hiperparametros', 'metricas_ext', 'puntajes_ext',
                                  'params_por_pliegue', 'trazas', 'detalle', 'detalles', 'fidelidad')
"""Columnas con estructuras anidadas; se serializan a JSON al guardar."""

ESTADOS: tuple[str, ...] = ('ok', 'no_aplica', 'error')


def ruta_cerrojo(ruta: Path) -> Path:
    """Archivo que marca la tabla maestra como en uso por un proceso (``<tabla>.en_uso``)."""
    return ruta.with_name(ruta.name + '.en_uso')


def dueno_del_cerrojo(ruta: Path) -> int | None:
    """PID vivo que tiene tomada la tabla maestra, o ``None`` si está libre.

    Parameters
    ----------
    ruta : Path
        Ruta de la tabla maestra.

    Returns
    -------
    int or None
        El PID escrito en el cerrojo si ese proceso sigue vivo y no es el actual. Un cerrojo
        huérfano (el proceso murió, p. ej. por un apagón) cuenta como libre.
    """
    cerrojo = ruta_cerrojo(ruta)
    try:
        pid = int(cerrojo.read_text(encoding='utf-8').split()[0])
    except (OSError, ValueError, IndexError):
        return None
    if pid == os.getpid():
        return None
    try:
        import psutil
        return pid if psutil.pid_exists(pid) else None
    except ImportError:
        return pid


def _json_default(o):
    """Convierte escalares y arreglos de numpy a tipos nativos para ``json.dumps``.

    Parameters
    ----------
    o : object
        Objeto que ``json`` no sabe serializar.

    Returns
    -------
    object
        ``o.item()`` para escalares numpy, ``o.tolist()`` para arreglos, ``str(o)``
        para cualquier otro tipo (fechas, enumeraciones…).
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Registro:
    """Registro persistente de corridas.

    Parameters
    ----------
    ruta : Path or str, optional
        Archivo ``.parquet`` (o ``.csv``). Por defecto ``resultados/experimentos.parquet``.

    Notes
    -----
    El archivo completo se reescribe en cada :meth:`anadir`; con 140 filas es barato y
    garantiza que en disco siempre haya un estado consistente.
    """

    def __init__(self, ruta: Path | str | None = None) -> None:
        self.ruta = Path(ruta) if ruta is not None else CARPETA_RESULTADOS / 'experimentos.parquet'
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._df = self._leer()

    # ------------------------------------------------------------------ persistencia
    def _leer(self) -> pd.DataFrame:
        """Lee el archivo si existe.

        Returns
        -------
        DataFrame
            Vacío si el archivo aún no existe.
        """
        if not self.ruta.exists():
            return pd.DataFrame()
        if self.ruta.suffix == '.parquet':
            return pd.read_parquet(self.ruta)
        return pd.read_csv(self.ruta)

    def _escribir(self) -> None:
        """Reescribe el archivo completo de forma atómica.

        Escribe primero en ``<ruta>.tmp`` y lo renombra sobre ``ruta`` con ``os.replace``, que
        es atómico en el mismo volumen: si el equipo se apaga a mitad de la escritura queda la
        versión anterior intacta, nunca un archivo corrupto.
        """
        tmp = self.ruta.with_name(self.ruta.name + '.tmp')
        if self.ruta.suffix == '.parquet':
            self._df.to_parquet(tmp, index=False)
        else:
            self._df.to_csv(tmp, index=False)
        os.replace(tmp, self.ruta)

    # ------------------------------------------------------------------ operaciones
    @staticmethod
    def _serializar(fila: dict) -> dict:
        """Copia de ``fila`` con las columnas anidadas como JSON y la marca de tiempo.

        Parameters
        ----------
        fila : dict

        Returns
        -------
        dict
        """
        f = dict(fila)
        for c in COLUMNAS_JSON:
            if c in f and not isinstance(f[c], str):
                f[c] = json.dumps(f[c], ensure_ascii=False, default=_json_default)
        f.setdefault('registrado_en', datetime.now().isoformat(timespec='seconds'))
        return f

    def _mascara(self, clave: dict) -> pd.Series:
        m = pd.Series(True, index=self._df.index)
        for k in CLAVE:
            m &= self._df[k] == clave[k]
        return m

    def existe(self, **clave) -> bool:
        """Indica si ya hay una fila con esa clave.

        Parameters
        ----------
        **clave
            ``tarea``, ``modelo``, ``balanceo``, ``optimizador`` y ``semilla``.

        Returns
        -------
        bool
        """
        if self._df.empty:
            return False
        return bool(self._mascara(clave).any())

    def estado(self, **clave) -> str | None:
        """Estado registrado de una corrida.

        Parameters
        ----------
        **clave
            ``tarea``, ``modelo``, ``balanceo``, ``optimizador`` y ``semilla``.

        Returns
        -------
        str or None
            ``'ok'``, ``'no_aplica'`` o ``'error'``; ``None`` si la corrida no está registrada.
        """
        if self._df.empty:
            return None
        m = self._mascara(clave)
        if not m.any():
            return None
        if 'estado' not in self._df.columns:
            return 'ok'
        return str(self._df.loc[m, 'estado'].iloc[-1])

    def anadir(self, fila: dict, sobrescribir: bool = False) -> None:
        """Añade (o reemplaza) una corrida y escribe el archivo.

        Parameters
        ----------
        fila : dict
            Debe contener las columnas de ``CLAVE``. Si no trae ``estado`` se infiere de
            ``aplica`` (``'ok'`` / ``'no_aplica'``).
        sobrescribir : bool
            Si ``True`` reemplaza una fila existente con la misma clave; si ``False`` y ya
            existe, lanza ``ValueError``.

        Raises
        ------
        ValueError
            Si la clave ya existe y ``sobrescribir`` es ``False``.
        RuntimeError
            Si otro proceso vivo tiene tomada la tabla (ver :func:`dueno_del_cerrojo`). Leer
            con :meth:`cargar` sigue permitido.
        """
        dueno = dueno_del_cerrojo(self.ruta)
        if dueno is not None:
            raise RuntimeError(f'{self.ruta.name} está en uso por otro proceso (pid {dueno}); no se puede '
                               f'escribir en ella hasta que termine (cerrojo: {ruta_cerrojo(self.ruta).name})')
        fila = dict(fila)
        fila.setdefault('estado', 'ok' if fila.get('aplica', True) else 'no_aplica')
        clave = {k: fila[k] for k in CLAVE}
        if self.existe(**clave):
            if not sobrescribir:
                raise ValueError(f'la corrida {clave} ya está registrada')
            self._df = self._df[~self._mascara(clave)]
        nueva = pd.DataFrame([self._serializar(fila)])
        self._df = pd.concat([self._df, nueva], ignore_index=True) if not self._df.empty else nueva
        self._escribir()

    def cargar(self, decodificar: bool = True) -> pd.DataFrame:
        """Devuelve la tabla completa.

        Parameters
        ----------
        decodificar : bool
            Si ``True``, las columnas JSON vuelven a listas y diccionarios.

        Returns
        -------
        DataFrame
        """
        df = self._df.copy()
        if decodificar:
            for c in COLUMNAS_JSON:
                if c in df.columns:
                    df[c] = df[c].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
        return df

    def resumen(self) -> pd.DataFrame:
        """Conteo de corridas por tarea y optimizador con la media de la métrica principal.

        Returns
        -------
        DataFrame
            Columnas ``tarea``, ``optimizador``, ``corridas``, ``ok``, ``media_ext`` y
            ``tiempo_s``. Vacío si no hay corridas.
        """
        if self._df.empty:
            return pd.DataFrame()
        df = self._df.copy()
        df['ok'] = df['estado'] == 'ok'
        return (df.groupby(['tarea', 'optimizador'])
                .agg(corridas=('modelo', 'size'), ok=('ok', 'sum'),
                     media_ext=('media_ext', 'mean'), tiempo_s=('tiempo_total_s', 'sum'))
                .reset_index())

    def __len__(self) -> int:
        return len(self._df)
