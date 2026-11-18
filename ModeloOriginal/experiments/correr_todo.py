"""Ejecuta, en orden, todos los experimentos del protocolo. Cada paso salta lo que ya está hecho, así que se puede
interrumpir y volver a lanzar.

    python experiments/correr_todo.py                 # la cola principal, desde donde se quedó
    python experiments/correr_todo.py --desde 05      # desde un paso
    python experiments/correr_todo.py --cola b --esperar 06m   # la cola secundaria (ablación y robustez), en otra terminal

Hay dos colas para aprovechar el equipo: la principal usa todos los núcleos con los benchmarks; la secundaria corre a la
vez los estudios que solo ajustan el modelo original (que usa un núcleo por pliegue). Los estudios que **miden tiempos**
(estabilidad por semilla y cómputo, curva de aprendizaje) están en la cola principal en momentos en que no compiten con
la secundaria, para que los tiempos sean comparables.

El orden pone primero lo que da un resultado utilizable (semilla 42, prueba, pruebas de hipótesis) y deja para el final
las dos repeticiones nuevas de la validación anidada de los diez modelos, que son lo más largo. Cada paso deja su salida
en ``resultados/`` y una línea en ``resultados/cola.log``.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
COLAS = {
    'a': [
        ('02a', ['02_validacion_anidada.py', '--semillas', '42'], 'validación anidada, semilla 42 (LightGBM y el original)'),
        ('03', ['03_prueba.py'], 'evaluación en prueba'),
        ('04b', ['04b_techo.py'], 'techo teórico'),
        ('04', ['04_pruebas_hipotesis.py'], 'pruebas de hipótesis (con las repeticiones disponibles)'),
        ('03b', ['03b_semillas.py'], 'estabilidad por semilla y cómputo'),
        ('03c', ['03c_memoria.py'], 'memoria de cada modelo e inferencia según las filas'),
        ('03d', ['03d_recomendados.py'], 'EBM y LightGBM con sus hiperparámetros recomendados'),
        ('06m', ['06_robustez.py', 'muestra'], 'curva de aprendizaje y tiempos'),
        ('05', ['05_conjuntos_adicionales.py'], 'conjuntos de datos adicionales'),
        ('02r', ['02_validacion_anidada.py', '--semillas', '43', '44', '--tareas', 'regresion'], 'repeticiones 2 y 3, regresión'),
        ('04r', ['04_pruebas_hipotesis.py'], 'pruebas de hipótesis (regresión completa)'),
        ('02c', ['02_validacion_anidada.py', '--semillas', '43', '44', '--tareas', 'clasificacion'], 'repeticiones 2 y 3, clasificación'),
        ('04f', ['04_pruebas_hipotesis.py'], 'pruebas de hipótesis (tres repeticiones)'),
        ('04c', ['04c_optimo.py'], 'las mismas pruebas para el predictor óptimo (análisis de la causa)'),
        ('04d', ['04d_empates_campana.py'], 'sensibilidad de la campaña del 25 % y de P4 a los empates'),
        ('04e', ['04e_referencias_homogeneas.py'], 'referencias homogéneas: MAE con la mediana y conjunto por densidad del benchmark'),
        ('09', ['09_rutas_relativas.py'], 'rutas de las tablas maestras relativas al repositorio'),
        ('08', ['08_tablas_libro.py'], 'tablas y figuras del libro'),
    ],
    'b': [
        ('07', ['07_ablacion.py'], 'ablación'),
        ('06', ['06_robustez.py', 'ciudades', 'ruido', 'sensibilidad', 'mezcla'], 'robustez (ciudades, ruido, sensibilidad y curva sin mezcla)'),
    ],
}


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--cola', default='a', choices=list(COLAS))
    a.add_argument('--desde', default=None, help='clave del primer paso a ejecutar')
    a.add_argument('--solo', nargs='+', default=None, help='claves de los pasos a ejecutar')
    a.add_argument('--esperar', default=None, help='no empezar hasta que la cola principal termine este paso (p. ej. 06m)')
    args = a.parse_args()
    pasos = COLAS[args.cola]
    claves = [c for c, *_ in pasos]
    inicio = claves.index(args.desde) if args.desde else 0
    bitacora = RAIZ / 'resultados' / 'cola.log'
    bitacora.parent.mkdir(parents=True, exist_ok=True)
    while args.esperar and f'[a] {args.esperar} ' not in (bitacora.read_text(encoding='utf-8') if bitacora.exists() else ''):
        time.sleep(30)
    for clave, orden, descripcion in pasos[inicio:]:
        if args.solo and clave not in args.solo:
            continue
        t0 = time.time()
        print(f'\n===== {clave}: {descripcion} =====', flush=True)
        codigo = subprocess.run([sys.executable, '-B', str(AQUI / orden[0]), *orden[1:]], cwd=str(RAIZ)).returncode
        linea = (f'{time.strftime("%Y-%m-%d %H:%M")} [{args.cola}] {clave} {descripcion}: '
                 f'{"bien" if codigo == 0 else f"FALLÓ ({codigo})"} en {(time.time() - t0) / 60:.1f} min')
        print(linea, flush=True)
        with bitacora.open('a', encoding='utf-8') as f:
            f.write(linea + '\n')


if __name__ == '__main__':
    main()
