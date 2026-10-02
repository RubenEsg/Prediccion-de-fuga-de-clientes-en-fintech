"""Lanza (o reanuda) las 140 corridas del pipeline combinatorio sobre los datos completos.

Uso, desde la carpeta del proyecto::

    correr_140.bat                                        (doble clic: ventana aparte, prioridad baja)
    .venv\\Scripts\\python.exe -u correr_140.py           (lo mismo, en la consola actual)
    .venv\\Scripts\\python.exe -u correr_140.py --prueba  (ensayo de 2 corridas sobre una submuestra)
    .venv\\Scripts\\python.exe -u correr_140.py --plan modelo_nuevo   (las 20 corridas de la EBM)

Con ``--plan modelo_nuevo`` ejecuta las corridas del modelo nuevo (EBM: 4 balanceos × 4
optimizadores en clasificación y 4 optimizadores en regresión) sobre la misma tabla maestra, con
su propia copia congelada del código en ``_codigo_modelo_nuevo/``. Las 140 ya hechas no se tocan.

Pensado para una ejecución de muchas horas en un PC de uso diario:

- **Reanudable.** Cada corrida se escribe en ``resultados/experimentos.parquet`` al terminar, con
  escritura atómica, y las ya registradas se saltan; las que fallaron se reintentan. Si el PC se
  apaga o se cierra la ventana, se pierde solo la corrida en curso: basta con volver a lanzar.
- **Código congelado.** La primera vez copia ``src/`` a ``_codigo_corrida/src`` y ejecuta siempre
  desde esa copia, de modo que editar ``src/`` mientras corre, o entre reanudaciones, no mezcla
  versiones en la tabla maestra. Para correr con código nuevo hay que borrar ``_codigo_corrida`` y
  también ``resultados/experimentos.parquet`` (empezar de cero).
- **Prioridad por debajo de lo normal**, que heredan los procesos hijos: el PC sigue usable.
- **Sin suspensión** por inactividad mientras corre.
- **No compite con otros experimentos**: antes de cada corrida espera a que termine cualquier
  ``benchmark_*.py`` o guion de ``exploracion/`` que esté en marcha, para no contaminar los tiempos
  por evaluación que la guía (§3.4) compara entre optimizadores.
- **Parada limpia**: crear el archivo ``resultados/DETENER``; termina al acabar la corrida en curso.
- **Orden**: regresión y luego clasificación, en cada tarea de los modelos baratos a los caros, con
  Random Forest al final, para que la tabla maestra se llene pronto.
- **Bitácora** con tiempo transcurrido y restante estimado en ``resultados/progreso.log``.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

ORDEN: list[tuple[str, list[str]]] = [
    ('regresion', ['ridge', 'lasso', 'arbol', 'svr', 'xgboost', 'knn', 'random_forest']),
    ('clasificacion', ['naive_bayes', 'logistica', 'arbol', 'svm', 'xgboost', 'knn', 'random_forest']),
]
"""Orden de ejecución: de lo barato a lo caro en cada tarea."""

HORAS_POR_CORRIDA: dict[tuple[str, str], float] = {
    ('regresion', 'ridge'): 0.02, ('regresion', 'lasso'): 0.11, ('regresion', 'arbol'): 0.05,
    ('regresion', 'svr'): 0.026, ('regresion', 'xgboost'): 0.17, ('regresion', 'knn'): 0.17,
    ('regresion', 'random_forest'): 0.12,
    ('clasificacion', 'naive_bayes'): 0.02, ('clasificacion', 'logistica'): 0.094,
    ('clasificacion', 'arbol'): 0.066, ('clasificacion', 'svm'): 0.074, ('clasificacion', 'xgboost'): 0.18,
    ('clasificacion', 'knn'): 0.29, ('clasificacion', 'random_forest'): 0.17,
    ('clasificacion', 'ebm'): 0.45, ('regresion', 'ebm'): 0.5,
}
"""Horas por corrida, medidas el 29-sep con ``exploracion/medir_costes_140.py`` (coste real de cada
ajuste sobre un pliegue interno) × 1,5 de sobrecarga observada en las corridas ya hechas (Ridge,
Lasso y árbol de regresión son tiempos reales). La bitácora las recalibra a medida que avanza.
La EBM se midió el 1-oct sobre un pliegue interno de 20.788 filas: 4-26 s por ajuste en
clasificación y 8-18 s en regresión con 2 bolsas y un proceso."""

BALANCEOS_MODELO_NUEVO: list[str] = ['ninguno', 'class_weight', 'smote', 'adasyn']
"""Orden del plan del modelo nuevo: primero lo más informativo; el sobremuestreo, que cuesta más
y en las 140 no mejoró a los modelos de árboles, al final."""

PESO_BALANCEO: dict[str, float] = {'ninguno': 0.8, 'smote': 1.2, 'adasyn': 1.2, 'class_weight': 0.8}
"""SMOTE y ADASYN agrandan el pliegue de entrenamiento (~1,5 veces las filas con la etiqueta del
percentil 75)."""

MINIMO_H_POR_CORRIDA: float = 0.02
"""Suelo de la estimación (≈ 1,2 min): en los modelos baratos domina la sobrecarga fija de las seis
búsquedas de cada corrida anidada (reparto en paralelo, ensayos de Optuna), no el ajuste. Medido el
29-sep: Ridge tardó entre 48 y 85 s por corrida aunque cada ajuste dura décimas de segundo."""

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


class Bitacora:
    """Escribe cada mensaje con marca de tiempo en la consola y en un archivo."""

    def __init__(self, ruta: Path) -> None:
        self.archivo = open(ruta, 'a', encoding='utf-8')

    def __call__(self, mensaje: str) -> None:
        linea = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {mensaje}'
        print(linea, flush=True)
        self.archivo.write(linea + '\n')
        self.archivo.flush()


def congelar_codigo(destino: Path, refrescar: bool) -> str:
    """Copia ``src/`` a ``destino/src`` (si no existe) y devuelve su huella.

    Parameters
    ----------
    destino : Path
    refrescar : bool
        Borrar una copia previa y volver a copiar.

    Returns
    -------
    str
        Primeros 12 caracteres del SHA-256 de los ``.py`` copiados.
    """
    if refrescar and destino.exists():
        shutil.rmtree(destino)
    nueva = not (destino / 'src').exists()
    if nueva:
        shutil.copytree(RAIZ / 'src', destino / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    h = hashlib.sha256()
    for f in sorted((destino / 'src').glob('*.py')):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    huella = h.hexdigest()[:12]
    if nueva:
        (destino / 'VERSION.txt').write_text(
            f'huella {huella}\ncongelado el {datetime.now():%Y-%m-%d %H:%M:%S} desde {RAIZ / "src"}\n',
            encoding='utf-8')
    return huella


def otros_experimentos() -> list[str]:
    """Guiones de ``exploracion/`` o ``benchmark_*.py`` en marcha fuera de este proceso.

    Returns
    -------
    list of str
        Una descripción corta por proceso encontrado (vacía si no hay ninguno o si ``psutil``
        no está disponible).
    """
    try:
        import psutil
    except ImportError:
        return []
    propios = {os.getpid()}
    try:
        yo = psutil.Process()
        propios |= {p.pid for p in yo.children(recursive=True)}
        propios |= {p.pid for p in yo.parents()}
    except psutil.Error:
        pass
    hallados = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if p.info['pid'] in propios or not (p.info['name'] or '').lower().startswith('python'):
                continue
            cmd = ' '.join(p.info['cmdline'] or [])
            if 'benchmark_' in cmd or 'exploracion' in cmd or 'comparar_entrega2' in cmd:
                hallados.append(f'pid {p.info["pid"]}: {cmd[-90:]}')
        except psutil.Error:
            continue
    return hallados


def esperar_cpu_libre(log: Bitacora, detener: Path) -> None:
    """Bloquea mientras haya otros experimentos en marcha (revisa cada minuto)."""
    avisado = False
    while not detener.exists():
        otros = otros_experimentos()
        if not otros:
            if avisado:
                log('los otros experimentos terminaron; se continúa')
            return
        if not avisado:
            log(f'en espera: hay otros experimentos en marcha que contaminarían los tiempos: {otros}')
            avisado = True
        time.sleep(60)


def horas(segundos: float) -> str:
    """Formato ``h:mm`` legible."""
    return str(timedelta(seconds=int(segundos)))[:-3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--prueba', action='store_true',
                    help='ensayo de 2 corridas sobre 1.500 filas (resultados en resultados_prueba/)')
    ap.add_argument('--plan', choices=['140', 'modelo_nuevo'], default='140',
                    help='las 140 corridas de la guía o las 20 del modelo nuevo (EBM)')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    carpeta = RAIZ / ('resultados_prueba' if args.prueba else 'resultados')
    carpeta.mkdir(exist_ok=True)
    detener = carpeta / 'DETENER'
    log = Bitacora(carpeta / 'progreso.log')
    if args.prueba:
        destino = RAIZ / '_codigo_prueba'
    else:
        destino = RAIZ / ('_codigo_corrida' if args.plan == '140' else '_codigo_modelo_nuevo')
    huella = congelar_codigo(destino, refrescar=args.prueba)
    sys.path.insert(0, str(destino))
    warnings.filterwarnings('ignore')

    import src
    from src import datos, experimento
    from src.config import K_EXT, K_INT, SEED, fijar_semillas
    from src.registro import Registro
    if Path(src.__file__).resolve().parent.parent != destino.resolve():
        raise RuntimeError(f'se importó src desde {src.__file__}, no desde la copia congelada {destino}')

    kernel32 = ctypes.windll.kernel32
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    t_inicio = time.perf_counter()
    try:
        log('=' * 78)
        log(f'inicio {"(PRUEBA) " if args.prueba else ""}· plan {args.plan} · pid {os.getpid()} · código '
            f'congelado en {destino.name} (huella {huella}) · prioridad baja · suspensión bloqueada')
        fijar_semillas(SEED)
        p = datos.preparar_todo()
        c, e = p['conjuntos'], p['etiqueta']
        X = c.X_train
        objetivos = {'clasificacion': p['y_train'], 'regresion': c.y_train_cont}
        if args.prueba:
            X = X.iloc[:1500]
            objetivos = {k: v.iloc[:1500] for k, v in objetivos.items()}
        log(f'datos: {len(X):,} filas × {X.shape[1]} columnas de entrenamiento · etiqueta {e.metodo} '
            f'q={e.q} umbral {e.umbral:.6f} · riesgo alto {objetivos["clasificacion"].mean() * 100:.2f} %')

        from src.optimizacion import OPTIMIZADORES
        if args.prueba:
            plan = ([('regresion', 'ridge', 'ninguno', 'grid'), ('clasificacion', 'random_forest', 'smote', 'genetica')]
                    if args.plan == '140' else
                    [('regresion', 'ebm', 'ninguno', 'bayesiana'), ('clasificacion', 'ebm', 'class_weight', 'genetica')])
            kw = dict(k_ext=2, k_int=2, presupuesto=3)
        else:
            if args.plan == '140':
                plan = [comb for tarea, mods in ORDEN for m in mods for comb in experimento.combinaciones(tarea, [m])]
                esperado = 140
            else:
                plan = ([('clasificacion', 'ebm', 'ninguno', o) for o in OPTIMIZADORES]
                        + [('regresion', 'ebm', 'ninguno', o) for o in OPTIMIZADORES]
                        + [('clasificacion', 'ebm', b, o) for b in BALANCEOS_MODELO_NUEVO[1:] for o in OPTIMIZADORES])
                esperado = 20
            kw = dict(k_ext=K_EXT, k_int=K_INT, presupuesto=None)
            if len(plan) != esperado:
                raise RuntimeError(f'el plan tiene {len(plan)} corridas, no {esperado}')

        from src.registro import dueno_del_cerrojo, ruta_cerrojo
        tabla = carpeta / 'experimentos.parquet'
        dueno = dueno_del_cerrojo(tabla)
        if dueno is not None:
            log(f'ya hay otra ejecución en marcha sobre {tabla.name} (pid {dueno}); esta se cierra sin hacer nada')
            return
        cerrojo = ruta_cerrojo(tabla)
        cerrojo.write_text(f'{os.getpid()} {datetime.now():%Y-%m-%d %H:%M:%S} correr_140.py\n', encoding='utf-8')
        registro = Registro(tabla)
        etiqueta_meta = {'metodo': e.metodo, 'umbral': e.umbral}

        def estimado_h(comb) -> float:
            t, m, b, _ = comb
            base = HORAS_POR_CORRIDA[(t, m)] * (PESO_BALANCEO[b] if t == 'clasificacion' else 1.0)
            return max(base, MINIMO_H_POR_CORRIDA)

        def pendiente(comb) -> bool:
            t, m, b, o = comb
            return registro.estado(tarea=t, modelo=m, balanceo=b, optimizador=o, semilla=SEED) not in ('ok', 'no_aplica')

        pendientes = [comb for comb in plan if pendiente(comb)]
        log(f'plan: {len(plan)} corridas · ya hechas {len(plan) - len(pendientes)} · pendientes {len(pendientes)} · '
            f'estimación a priori {sum(map(estimado_h, pendientes)):.1f} h')

        real_s, est_s, hechas = 0.0, 0.0, 0
        for i, comb in enumerate(plan, 1):
            if not pendiente(comb):
                continue
            if detener.exists():
                log(f'encontrado {detener.name}: se detiene antes de la corrida {i}/{len(plan)}')
                break
            esperar_cpu_libre(log, detener)
            if detener.exists():
                log(f'encontrado {detener.name}: se detiene antes de la corrida {i}/{len(plan)}')
                break
            t, m, b, o = comb
            t1 = time.perf_counter()
            filas = experimento.correr_todo(X, objetivos[t], t, registro, modelos=[m], balanceos=[b],
                                            optimizadores=[o], etiqueta=etiqueta_meta,
                                            carpeta_modelos=carpeta / 'modelos', informar=lambda s: None, **kw)
            dur = time.perf_counter() - t1
            fila = filas[0] if filas else {'estado': '?', 'motivo': 'no se ejecutó'}
            if fila['estado'] == 'ok':
                real_s += dur
                est_s += estimado_h(comb) * 3600
                hechas += 1
                resultado = (f'{fila["metrica_principal"]} = {abs(fila["media_ext"]):.4f} ± {fila["desv_ext"]:.4f} · '
                             f'{fila["n_evaluaciones"]} evals · {horas(dur)}')
            else:
                resultado = f'{fila["estado"].upper()}: {fila.get("motivo", "")[:160]}'
            factor = real_s / est_s if est_s > 0 else 1.0
            resto_h = factor * sum(estimado_h(x) for x in plan[i:] if pendiente(x))
            log(f'[{i:3d}/{len(plan)}] {t} · {m} · {b} · {o}: {resultado} · transcurrido '
                f'{horas(time.perf_counter() - t_inicio)} · restante estimado {resto_h:.1f} h '
                f'(ritmo real/estimado ×{factor:.2f}) · fin previsto '
                f'{datetime.now() + timedelta(hours=resto_h):%a %d %H:%M}')

        df = registro.cargar(decodificar=False)
        conteo = df['estado'].value_counts().to_dict() if 'estado' in df.columns else {}
        log(f'fin de la sesión · {hechas} corridas ejecutadas en {horas(time.perf_counter() - t_inicio)} · '
            f'plan {args.plan}: {sum(not pendiente(x) for x in plan)} de {len(plan)} terminadas · '
            f'tabla maestra: {conteo} ({len(df)} filas)')
    except BaseException:
        log('ERROR no controlado:\n' + traceback.format_exc())
        raise
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        cerrojo = carpeta / 'experimentos.parquet.en_uso'
        try:
            if cerrojo.exists() and cerrojo.read_text(encoding='utf-8').split()[0] == str(os.getpid()):
                cerrojo.unlink()
        except OSError:
            pass
        log.archivo.close()


if __name__ == '__main__':
    main()
