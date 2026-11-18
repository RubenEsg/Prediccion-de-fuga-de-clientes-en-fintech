"""Búsqueda sistemática de literatura para la matriz de novedad (guía §5.2), reproducible.

Consulta tres bases de datos abiertas con las cadenas de ``CADENAS`` y guarda, para cada base y cadena, el número de
resultados y los primeros 25 por relevancia (título, año, identificador y resumen cuando la base lo da). Después hace
la búsqueda por citación: hacia adelante (entre los trabajos que citan a cada antecedente de ``SEMILLAS``, cuántos tratan
de fuga de clientes) y hacia atrás (entre las referencias de los trabajos más cercanos, ``CERCANOS``, cuántas tratan de
mezclas y cuántas de fuga). Por último consulta GitHub para la revisión de repositorios de código afines.

- **OpenAlex** (índice abierto de la literatura científica; cubre revistas, congresos y prepublicaciones).
- **arXiv** (prepublicaciones de aprendizaje automático y estadística).
- **Crossref** (registro de DOI de editoriales: revistas y actas arbitradas). Su búsqueda no admite operadores: ordena
  por relevancia todo lo que contiene alguno de los términos, así que el total es grande y se revisan los 25 primeros.

Scopus, Web of Science y Google Scholar no ofrecen consulta programática abierta; se usan bases cuya consulta se puede
repetir con este guion. La relevancia de cada resultado se juzga a mano leyendo título y resumen; ese juicio va en
``experiments/literatura_relevancia.csv`` y en el informe, junto a esta matriz.

La guía pide repetir la búsqueda antes de la entrega. Cada ejecución guarda su copia con fecha
(``busqueda_<fecha>.json`` y ``matriz_<fecha>.csv``) y deja la más reciente en ``busqueda.json`` y ``matriz.csv``. La
primera consulta fue el 17 de noviembre de 2026, con las cadenas C1 a C9; en la segunda se añadieron las cadenas C10 a
C13 (EM con boosting para mezclas, árboles con efectos aleatorios por EM, modelos de abandono latente y sus aplicaciones
a fuga) y la búsqueda por citación desde los trabajos más cercanos que aparecieron con ellas.

Si una consulta falla (límite diario de la base, corte de red) queda anotada con su error; al volver a ejecutar el guion
el mismo día solo se repiten las que fallaron.

Salida: ``resultados/literatura/busqueda.json`` (todo) y ``resultados/literatura/matriz.csv`` (la matriz).
Uso: ``python experiments/00_busqueda_literatura.py``.
"""
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CARPETA_RESULTADOS  # noqa: E402

AGENTE = {'User-Agent': 'busqueda-bibliografica-academica'}
REVISADOS = 25

CADENAS = {
    'C1': {'tema': 'Mezcla o clase latente + boosting + fuga de clientes',
           'openalex': '"customer churn" AND ("latent class" OR "finite mixture" OR "mixture of regressions") AND boosting',
           'arxiv': 'all:"customer churn" AND (all:mixture OR all:"latent class")',
           'crossref': 'churn mixture boosting'},
    'C2': {'tema': 'Heterogeneidad no observada + boosting + fuga de clientes',
           'openalex': '"customer churn" AND ("unobserved heterogeneity" OR "latent variable") AND "gradient boosting"',
           'arxiv': 'all:churn AND all:"latent variable" AND all:boosting',
           'crossref': 'churn latent boosting'},
    'C3': {'tema': 'Mezcla de regresiones ajustada con boosting y EM (familia del algoritmo, cualquier aplicación)',
           'openalex': '("mixture of regressions" OR "finite mixture regression" OR "latent class regression") AND ("gradient boosting" OR "boosted trees") AND ("expectation-maximization" OR "EM algorithm")',
           'arxiv': '(all:"mixture of regressions" OR all:"mixture regression") AND all:boosting',
           'crossref': 'mixture regression boosting EM'},
    'C4': {'tema': 'Boosting con ruido multimodal o de mezcla',
           'openalex': '"gradient boosting" AND ("mixture noise" OR "multimodal noise" OR "mixture of Gaussians noise") AND regression',
           'arxiv': 'all:"gradient boosting" AND all:"mixture density"',
           'crossref': 'boosting mixture density regression'},
    'C5': {'tema': 'Predicción conforme + fuga de clientes',
           'openalex': '("customer churn" OR "customer attrition") AND ("conformal prediction" OR "prediction sets")',
           'arxiv': 'all:churn AND all:"conformal prediction"',
           'crossref': 'churn conformal prediction'},
    'C6': {'tema': 'Regresión distribucional o probabilística + fuga de clientes',
           'openalex': '("customer churn" OR "customer attrition") AND ("distributional regression" OR "probabilistic regression" OR NGBoost OR "GAMLSS")',
           'arxiv': 'all:churn AND (all:"distributional regression" OR all:NGBoost OR all:"probabilistic regression")',
           'crossref': 'churn distributional regression'},
    'C7': {'tema': 'Clasificar una respuesta continua dicotomizada usando la respuesta continua',
           'openalex': '("dichotomized" OR "dichotomised") AND "continuous outcome" AND ("exceedance probability" OR "distributional approach") AND classification',
           'arxiv': 'all:dichotomized AND all:"continuous outcome"',
           'crossref': 'dichotomized continuous outcome classification'},
    'C8': {'tema': 'Modelo aditivo (GAM, EBM) con clases latentes o mezcla',
           'openalex': '("generalized additive model" OR "explainable boosting machine") AND ("finite mixture" OR "latent class") AND boosting',
           'arxiv': 'all:"additive model" AND all:"finite mixture"',
           'crossref': 'additive model finite mixture boosting'},
    'C9': {'tema': 'En español: fuga o deserción de clientes + clase latente, mezcla o boosting',
           'openalex': '("fuga de clientes" OR "deserción de clientes" OR "abandono de clientes") AND ("clase latente" OR "mezcla finita" OR boosting)',
           'arxiv': 'all:"fuga de clientes" OR all:"abandono de clientes"',
           'crossref': 'fuga de clientes'},
    # añadidas en la segunda consulta
    'C10': {'tema': 'EM con un paso de boosting para modelos de mezcla (regla de aprendizaje, cualquier aplicación)',
            'openalex': '("finite mixture" OR "mixture model") AND ("gradient boosting" OR "boosted trees" OR "tree boosting") AND ("EM algorithm" OR "expectation-maximization" OR "expectation-boosting")',
            'arxiv': 'all:mixture AND all:boosting AND (all:"EM algorithm" OR all:"expectation-maximization" OR all:"expectation-boosting")',
            'crossref': 'expectation boosting mixture model gradient tree boosting EM'},
    'C11': {'tema': 'Árboles o boosting con efectos aleatorios estimados por EM (grupos observados)',
            'openalex': '("mixed effects" OR "random effects") AND ("random forest" OR "gradient boosting" OR "regression tree") AND ("EM algorithm" OR "expectation-maximization")',
            'arxiv': '(all:"mixed effects" OR all:"random effects") AND (all:boosting OR all:"random forest") AND (all:"EM algorithm" OR all:"expectation-maximization")',
            'crossref': 'mixed effects random forest boosting EM algorithm clustered data'},
    'C12': {'tema': 'Modelos de abandono latente o de mezcla aplicados a fuga de clientes (sin boosting)',
            'openalex': '("customer churn" OR "customer attrition" OR "latent attrition") AND ("finite mixture" OR "latent class" OR "mixture model")',
            'arxiv': 'all:churn AND (all:"finite mixture" OR all:"latent class" OR all:"mixture model" OR all:"latent attrition")',
            'crossref': 'latent attrition finite mixture latent class customer churn'},
    'C13': {'tema': 'Boosting con efectos aleatorios o con densidad de mezcla aplicado a fuga de clientes',
            'openalex': '("customer churn" OR "customer attrition") AND ("mixed effects" OR "random effects" OR "mixture density" OR XGBoostLSS OR GPBoost) AND boosting',
            'arxiv': 'all:churn AND all:boosting AND (all:"mixed effects" OR all:"random effects" OR all:"mixture density")',
            'crossref': 'customer churn mixed effects boosting mixture density'},
}

SEMILLAS = {
    'DeSarbo y Cron (1988), regresión por conglomerados con máxima verosimilitud': 'https://doi.org/10.1007/bf01897167',
    'Jacobs y otros (1991), mezcla de expertos': 'https://doi.org/10.1162/neco.1991.3.1.79',
    'Grün y Leisch (2008), FlexMix versión 2': 'https://doi.org/10.18637/jss.v028.i04',
    'Mayr y otros (2012), gamboostLSS': 'https://doi.org/10.1111/j.1467-9876.2011.01033.x',
    'Duan y otros (2020), NGBoost': 'https://doi.org/10.48550/arxiv.1910.03225',
    'Lou y otros (2013), GA2M': 'https://doi.org/10.1145/2487575.2487579',
    'Vovk y otros (2019), distribuciones predictivas conformes': 'https://doi.org/10.1007/s10994-018-5755-8',
    # añadidos en la segunda consulta: los antecedentes algorítmicos más cercanos
    'Hou, Li y Gao (2025), algoritmo Expectation-Boosting para mezclas': 'https://doi.org/10.1016/j.insmatheco.2024.12.007',
    'Zhou, Qian y Yang (2022), EMTboost': 'https://doi.org/10.1080/03610918.2020.1772302',
    'Xue, Yao y Xiang (2024), EM con aprendizaje automático': 'https://doi.org/10.1007/s00180-024-01482-5',
    'Jobst (2024), regresión de mezcla con boosting': 'https://doi.org/10.48550/arxiv.2412.09583',
    'März (2019), XGBoostLSS': 'https://doi.org/10.48550/arxiv.1907.03178',
    'Hajjem, Bellavance y Larocque (2014), bosque aleatorio de efectos mixtos': 'https://doi.org/10.1080/00949655.2012.741599',
    'Sela y Simonoff (2012), árboles RE-EM': 'https://doi.org/10.1007/s10994-011-5258-3',
    'Sigrist (2022), GPBoost': 'https://doi.org/10.48550/arxiv.2004.02653',
}


def pedir(url: str, intentos: int = 6) -> bytes:
    """Descarga una URL; ante un fallo pasajero (límite de consultas, corte de red) espera y reintenta."""
    for i in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if i == intentos - 1:
                raise
            time.sleep(6 * (i + 1))
    raise RuntimeError('sin respuesta')


def resumen_openalex(indice: dict | None) -> str:
    """OpenAlex guarda el resumen como índice invertido (palabra → posiciones): se reconstruye el texto."""
    if not indice:
        return ''
    palabras = sorted((pos, palabra) for palabra, posiciones in indice.items() for pos in posiciones)
    return ' '.join(p for _, p in palabras)


def openalex(cadena: str) -> dict:
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search': cadena, 'per-page': REVISADOS})
    d = json.loads(pedir(url))
    return {'total': d['meta']['count'],
            'primeros': [{'titulo': w.get('title'), 'anio': w.get('publication_year'), 'id': w.get('doi') or w.get('id'),
                          'citas': w.get('cited_by_count'), 'resumen': resumen_openalex(w.get('abstract_inverted_index'))[:1200]}
                         for w in d['results']]}


def arxiv(cadena: str) -> dict:
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': cadena, 'max_results': REVISADOS})
    raiz = ElementTree.fromstring(pedir(url))
    ns = {'a': 'http://www.w3.org/2005/Atom', 'o': 'http://a9.com/-/spec/opensearch/1.1/'}
    return {'total': int(raiz.find('o:totalResults', ns).text),
            'primeros': [{'titulo': ' '.join(e.find('a:title', ns).text.split()), 'anio': int(e.find('a:published', ns).text[:4]),
                          'id': e.find('a:id', ns).text, 'resumen': ' '.join(e.find('a:summary', ns).text.split())[:1200]}
                         for e in raiz.findall('a:entry', ns)]}


def crossref(cadena: str) -> dict:
    url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({
        'query.bibliographic': cadena, 'rows': REVISADOS, 'select': 'title,issued,DOI,type'})
    d = json.loads(pedir(url))['message']
    return {'total': int(d['total-results']),
            'primeros': [{'titulo': (w.get('title') or [''])[0], 'anio': (w.get('issued', {}).get('date-parts') or [[None]])[0][0],
                          'id': 'https://doi.org/' + w.get('DOI', '')} for w in d.get('items', [])]}


def citas_sobre_fuga(doi: str) -> dict:
    """Trabajos que citan a ``doi`` y cuyo título o resumen habla de fuga de clientes (búsqueda por citación)."""
    obra = json.loads(pedir('https://api.openalex.org/works/' + urllib.parse.quote(doi, safe=':/')))
    ident = obra['id'].rsplit('/', 1)[-1]
    filtro = f'cites:{ident},title_and_abstract.search:churn'
    d = json.loads(pedir('https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': filtro, 'per-page': REVISADOS})))
    return {'titulo': obra.get('title'), 'anio': obra.get('publication_year'), 'citado_por': obra.get('cited_by_count'),
            'citan_y_tratan_fuga': d['meta']['count'],
            'primeros': [{'titulo': w.get('title'), 'anio': w.get('publication_year'), 'id': w.get('doi') or w.get('id'),
                          'resumen': resumen_openalex(w.get('abstract_inverted_index'))[:1200]} for w in d['results']]}


CERCANOS = {
    'Xue, Yao y Xiang (2024), EM con aprendizaje automático para mezclas de regresiones': 'https://doi.org/10.1007/s00180-024-01482-5',
    'Hepp y otros (2020), regresión distribucional con clases latentes': 'https://doi.org/10.1186/s12859-020-03853-3',
    'Rügamer y otros (2023), mezcla de expertos con regresión distribucional': 'https://doi.org/10.1007/s10182-023-00486-8',
    'De Caigny, Coussement y De Bock (2018), híbrido de árboles y logística para fuga': 'https://doi.org/10.1016/j.ejor.2018.02.009',
    'Guo y vanden Broucke (2024), predicción conforme para fuga': 'https://doi.org/10.1109/icmla61862.2024.00008',
    # añadidos en la segunda consulta
    'Hou, Li y Gao (2025), algoritmo Expectation-Boosting para mezclas': 'https://doi.org/10.1016/j.insmatheco.2024.12.007',
    'Jobst (2024), regresión de mezcla con boosting': 'https://doi.org/10.48550/arxiv.2412.09583',
}
"""Trabajos más cercanos hallados con las cadenas: de ellos sale la búsqueda por citación hacia atrás."""

REPOSITORIOS = ['mixture regression boosting', 'latent class gradient boosting', 'finite mixture boosting EM',
                'churn mixture model boosting', 'expectation boosting mixture', 'mixed effects boosting churn']
"""Consultas a GitHub para la revisión de repositorios de código afines."""

TERMINOS_MEZCLA = ('mixture', 'latent class', 'latent variable', 'heterogeneity', 'finite mixture', 'mezcla')


def referencias_de(doi: str) -> dict:
    """Búsqueda por citación hacia atrás: las referencias de ``doi`` que tratan de mezclas o clases latentes."""
    obra = json.loads(pedir('https://api.openalex.org/works/' + urllib.parse.quote(doi, safe=':/')))
    ids = [r.rsplit('/', 1)[-1] for r in obra.get('referenced_works', [])]
    titulos = []
    for i in range(0, len(ids), 50):
        filtro = 'openalex_id:' + '|'.join(ids[i:i + 50])
        d = json.loads(pedir('https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': filtro, 'per-page': 50, 'select': 'title,publication_year,doi'})))
        titulos += [(w.get('title') or '', w.get('publication_year'), w.get('doi')) for w in d['results']]
        time.sleep(0.5)
    de_mezcla = [x for x in titulos if any(k in x[0].lower() for k in TERMINOS_MEZCLA)]
    de_fuga = [x for x in titulos if any(k in x[0].lower() for k in ('churn', 'attrition', 'fuga'))]
    return {'referencias': len(ids), 'revisadas': len(titulos), 'sobre_mezclas_o_clases_latentes': len(de_mezcla), 'sobre_fuga': len(de_fuga),
            'que_combinan_mezcla_y_fuga': len([x for x in de_mezcla if x in de_fuga]),
            'titulos_de_mezcla': [{'titulo': x[0], 'anio': x[1], 'id': x[2]} for x in de_mezcla[:40]]}


def repositorios(consulta: str) -> dict:
    """Repositorios públicos de GitHub que responden a la consulta, ordenados por estrellas."""
    url = 'https://api.github.com/search/repositories?' + urllib.parse.urlencode({'q': consulta, 'sort': 'stars', 'per_page': 10})
    d = json.loads(pedir(url))
    return {'total': d.get('total_count'),
            'primeros': [{'nombre': r['full_name'], 'estrellas': r['stargazers_count'], 'descripcion': (r.get('description') or '')[:200]}
                         for r in d.get('items', [])]}


def _pendiente(entrada: dict | None) -> bool:
    """Una consulta está pendiente si no se ha hecho hoy o si falló."""
    return entrada is None or 'error' in entrada


def _guardar(carpeta: Path, salida: dict) -> None:
    """Escribe la copia con fecha y la deja también como la más reciente (``busqueda.json`` y ``matriz.csv``)."""
    fecha = salida['fecha_de_consulta']
    filas = [[base, clave, c['tema'], c[base]['cadena'], c[base].get('total'), min(len(c[base].get('primeros', [])), REVISADOS), fecha]
             for clave, c in salida['cadenas'].items() for base in ('openalex', 'arxiv', 'crossref') if base in c]
    for nombre in (f'busqueda_{fecha}.json', 'busqueda.json'):
        (carpeta / nombre).write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding='utf-8')
    for nombre in (f'matriz_{fecha}.csv', 'matriz.csv'):
        with (carpeta / nombre).open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['base', 'cadena', 'tema', 'consulta', 'resultados', 'revisados', 'fecha'])
            w.writerows(filas)


def main() -> None:
    carpeta = CARPETA_RESULTADOS / 'literatura'
    carpeta.mkdir(parents=True, exist_ok=True)
    fecha = time.strftime('%Y-%m-%d')
    ruta = carpeta / f'busqueda_{fecha}.json'
    previa = json.loads(ruta.read_text(encoding='utf-8')) if ruta.exists() else {}      # lo que ya se consultó hoy
    anteriores = sorted(x.stem.split('_', 1)[1] for x in carpeta.glob('busqueda_*.json') if x.stem.split('_', 1)[1] != fecha)
    salida = {'fecha_de_consulta': fecha, 'consultas_anteriores': anteriores, 'revisados_por_cadena': REVISADOS,
              'cadenas': previa.get('cadenas', {}), 'citacion': previa.get('citacion', {}),
              'citacion_hacia_atras': previa.get('citacion_hacia_atras', {}), 'repositorios': previa.get('repositorios', {})}
    for clave, c in CADENAS.items():
        entrada = salida['cadenas'].setdefault(clave, {'tema': c['tema']})
        for base, funcion in (('openalex', openalex), ('arxiv', arxiv), ('crossref', crossref)):
            if not _pendiente(entrada.get(base)):
                continue
            try:
                r = funcion(c[base])
            except Exception as e:  # noqa: BLE001
                r = {'total': None, 'primeros': [], 'error': f'{type(e).__name__}: {e}'[:200]}
            entrada[base] = {'cadena': c[base], **r}
            print(f'{clave} {base:9s} {r["total"]}' + (f'  [{r["error"]}]' if 'error' in r else ''), flush=True)
            time.sleep(3.5 if base == 'arxiv' else 1.0)
        _guardar(carpeta, salida)
    for nombre, doi in SEMILLAS.items():
        if not _pendiente(salida['citacion'].get(nombre)):
            continue
        try:
            salida['citacion'][nombre] = {'doi': doi, **citas_sobre_fuga(doi)}
        except Exception as e:  # noqa: BLE001
            salida['citacion'][nombre] = {'doi': doi, 'error': f'{type(e).__name__}: {e}'[:200]}
        print(nombre, '->', salida['citacion'][nombre].get('citan_y_tratan_fuga'), salida['citacion'][nombre].get('error', ''), flush=True)
        time.sleep(1.0)
    _guardar(carpeta, salida)
    for nombre, doi in CERCANOS.items():
        if not _pendiente(salida['citacion_hacia_atras'].get(nombre)):
            continue
        try:
            salida['citacion_hacia_atras'][nombre] = {'doi': doi, **referencias_de(doi)}
        except Exception as e:  # noqa: BLE001
            salida['citacion_hacia_atras'][nombre] = {'doi': doi, 'error': f'{type(e).__name__}: {e}'[:200]}
        r = salida['citacion_hacia_atras'][nombre]
        print(nombre, '<-', r.get('referencias'), r.get('sobre_mezclas_o_clases_latentes'), r.get('sobre_fuga'), r.get('que_combinan_mezcla_y_fuga'),
              r.get('error', ''), flush=True)
    _guardar(carpeta, salida)
    for consulta in REPOSITORIOS:
        if not _pendiente(salida['repositorios'].get(consulta)):
            continue
        try:
            salida['repositorios'][consulta] = repositorios(consulta)
        except Exception as e:  # noqa: BLE001
            salida['repositorios'][consulta] = {'error': f'{type(e).__name__}: {e}'[:200]}
        print('GitHub', consulta, '->', salida['repositorios'][consulta].get('total'), flush=True)
        time.sleep(7)
    _guardar(carpeta, salida)
    fallidas = sum(_pendiente(c.get(b)) for c in salida['cadenas'].values() for b in ('openalex', 'arxiv', 'crossref'))
    fallidas += sum(_pendiente(x) for grupo in ('citacion', 'citacion_hacia_atras', 'repositorios') for x in salida[grupo].values())
    print(f'Consulta del {fecha}: {fallidas} consultas con error' + (' (vuelva a ejecutar el guion para repetirlas)' if fallidas else ''))


if __name__ == '__main__':
    main()
