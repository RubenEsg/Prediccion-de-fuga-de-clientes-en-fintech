"""Conjuntos de datos adicionales del protocolo (sección 8): descarga, lectura y la modificación declarada.

Los cuatro conjuntos son públicos y se descargan de su fuente la primera vez (quedan en ``datos_adicionales/``, que no
se versiona). Cada cargador devuelve los predictores, el objetivo continuo y una ficha con la fuente, la huella del
archivo descargado y la modificación aplicada. Las variables categóricas quedan como texto, para que el
preprocesamiento común las codifique igual que en el problema principal.
"""
from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from .config import RAIZ

CARPETA = RAIZ / 'datos_adicionales'
AGENTE = {'User-Agent': 'descarga-de-datos-publicos'}
URL_SEGUROS = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
URL_ENERGIA = 'https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip'
URL_ABULON = 'https://archive.ics.uci.edu/static/public/1/abalone.zip'
NOMBRES = {'seguros': 'Seguros médicos', 'energia': 'Eficiencia energética', 'abulon': 'Abulón',
           'california': 'Vivienda de California'}
CON_VARIABLE_OMITIDA = ('seguros', 'energia')
"""Conjuntos en los que se retira para todos los modelos una variable categórica que desplaza el objetivo."""


def _descargar(url: str, nombre: str) -> bytes:
    """Devuelve el archivo de ``url``; lo descarga solo si no está ya en la carpeta local."""
    CARPETA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA / nombre
    if not ruta.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=180) as respuesta:
            ruta.write_bytes(respuesta.read())
    return ruta.read_bytes()


def _huella(crudo: bytes) -> str:
    return hashlib.sha256(crudo).hexdigest()


def _leer_xlsx(crudo: bytes) -> pd.DataFrame:
    """Lee la primera hoja de un libro ``.xlsx`` con la biblioteca estándar (no hay ``openpyxl`` en el entorno).

    Un ``.xlsx`` es un zip con XML: ``xl/sharedStrings.xml`` guarda los textos y ``xl/worksheets/sheet1.xml`` las
    celdas, cada una con su referencia (``B3``) y su valor. La primera fila se toma como encabezado.
    """
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    z = zipfile.ZipFile(io.BytesIO(crudo))
    textos = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ElementTree.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns):
            textos.append(''.join(t.text or '' for t in si.iter('{%s}t' % ns['m'])))
    filas = []
    for fila in ElementTree.fromstring(z.read('xl/worksheets/sheet1.xml')).iter('{%s}row' % ns['m']):
        valores = {}
        for celda in fila.findall('m:c', ns):
            v = celda.find('m:v', ns)
            if v is None or v.text is None:
                continue
            letras = re.match(r'[A-Z]+', celda.get('r')).group()
            columna = 0
            for letra in letras:
                columna = columna * 26 + ord(letra) - 64
            valores[columna - 1] = textos[int(v.text)] if celda.get('t') == 's' else float(v.text)
        if valores:
            filas.append(valores)
    tabla = pd.DataFrame(filas).sort_index(axis=1)
    tabla.columns = [str(c).strip() for c in tabla.iloc[0]]
    return tabla.iloc[1:].reset_index(drop=True).apply(pd.to_numeric, errors='coerce').dropna(how='all').dropna(axis=1, how='all')


def cargar_seguros() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Costo médico facturado; se omite ``smoker`` (la variable que más desplaza el costo)."""
    crudo = _descargar(URL_SEGUROS, 'insurance.csv')
    t = pd.read_csv(io.BytesIO(crudo))
    X = t.drop(columns=['charges', 'smoker'])
    return X, t['charges'].astype(float).rename('objetivo'), {
        'fuente': URL_SEGUROS, 'sha256': _huella(crudo), 'omitida': 'smoker', 'objetivo': 'charges (dólares)'}


def cargar_energia() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Carga de calefacción de edificios simulados (ENB2012); se omite el área de vidrio (X7) y se retira Y2."""
    crudo = _descargar(URL_ENERGIA, 'energy_efficiency.zip')
    t = _leer_xlsx(zipfile.ZipFile(io.BytesIO(crudo)).read('ENB2012_data.xlsx'))
    X = t.drop(columns=['Y1', 'Y2', 'X7']).rename(columns={
        'X1': 'compacidad_relativa', 'X2': 'area_superficie', 'X3': 'area_muros', 'X4': 'area_techo',
        'X5': 'altura', 'X6': 'orientacion', 'X8': 'distribucion_vidrio'})
    return X, t['Y1'].astype(float).rename('objetivo'), {
        'fuente': URL_ENERGIA, 'sha256': _huella(crudo), 'omitida': 'X7 (área de vidrio)', 'objetivo': 'Y1, carga de calefacción (kWh/m²)'}


def cargar_abulon() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Anillos del abulón (su edad) a partir de medidas físicas; sin modificar."""
    crudo = _descargar(URL_ABULON, 'abalone.zip')
    columnas = ['sexo', 'largo', 'diametro', 'alto', 'peso_total', 'peso_carne', 'peso_visceras', 'peso_concha', 'anillos']
    t = pd.read_csv(zipfile.ZipFile(io.BytesIO(crudo)).open('abalone.data'), header=None, names=columnas)
    return t.drop(columns=['anillos']), t['anillos'].astype(float).rename('objetivo'), {
        'fuente': URL_ABULON, 'sha256': _huella(crudo), 'omitida': None, 'objetivo': 'anillos (entero)'}


def cargar_california() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Valor mediano de la vivienda por distrito censal de California (1990); sin modificar."""
    from sklearn.datasets import fetch_california_housing

    b = fetch_california_housing(data_home=str(CARPETA), as_frame=True)
    huella = hashlib.sha256(pd.util.hash_pandas_object(b.frame, index=False).values.tobytes()).hexdigest()
    return b.data.copy(), b.target.astype(float).rename('objetivo'), {
        'fuente': 'sklearn.datasets.fetch_california_housing (StatLib, Pace y Barry, 1997)', 'sha256': huella,
        'omitida': None, 'objetivo': 'valor mediano de la vivienda (cientos de miles de dólares, tope en 5)'}


CARGADORES = {'seguros': cargar_seguros, 'energia': cargar_energia, 'abulon': cargar_abulon, 'california': cargar_california}


def cargar(nombre: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Predictores, objetivo continuo y ficha de un conjunto adicional.

    Parameters
    ----------
    nombre : {'seguros', 'energia', 'abulon', 'california'}

    Returns
    -------
    X : DataFrame
        Con índice 0..n−1; las categóricas, como texto.
    y : Series
    ficha : dict
        ``nombre``, ``fuente``, ``sha256``, ``omitida``, ``objetivo``, ``n`` y ``p``.

    Raises
    ------
    ValueError
        Si el nombre no es uno de los cuatro, o si la tabla trae faltantes.
    """
    if nombre not in CARGADORES:
        raise ValueError(f'conjunto desconocido: {nombre!r}; use uno de {list(CARGADORES)}')
    X, y, ficha = CARGADORES[nombre]()
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    if X.isna().any().any() or y.isna().any() or not np.isfinite(y).all():
        raise ValueError(f'{nombre}: hay valores faltantes')
    ficha.update({'nombre': NOMBRES[nombre], 'n': int(len(X)), 'p': int(X.shape[1])})
    return X, y, ficha
