"""Auditoría de fuga de información y efecto de la corrección de la recencia.

Observación del docente: «la recencia se calcula con la fecha máxima de todo el detalle, que incluye el periodo de
prueba». La corrección (``src/datos.py``) calcula la fecha de corte solo con los clientes de entrenamiento. Este guion
comprueba, con los datos:

1. que la partición es la misma del Entregable 3;
2. cuál era la fecha de corte antes (todo el detalle) y cuál es ahora (solo entrenamiento), y cuántas celdas de los
   predictores cambian;
3. que, aunque la fecha hubiera cambiado, los modelos no lo habrían notado: todos estandarizan la recencia dentro del
   pipeline y la recencia estandarizada no depende de la fecha de corte (se demuestra y se verifica numéricamente);
4. que la fecha de corte tampoco cambia dentro de ningún pliegue de la validación anidada;
5. el resto de la lista de fuga de la guía (§3): preprocesamiento, variables que codifican la respuesta, duplicados
   entre particiones, umbral de la etiqueta y mezcla temporal.

Salida: ``resultados/auditoria_fuga.json``. Uso: ``python experiments/01_auditoria_fuga.py``.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import datos                                                      # noqa: E402
from src.config import CARPETA_RESULTADOS, K_EXT, K_INT, SEED, SEMILLAS_REPETICION, TEST_SIZE  # noqa: E402
from src.optimizacion import construir_cv                                  # noqa: E402


def huella(df: pd.DataFrame) -> str:
    """Resumen SHA-256 del contenido de una tabla (valores e índice)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def main() -> None:
    p = datos.preparar_todo()
    tx, clientes, part, c = p['tx'], p['clientes'], p['particion'], p['conjuntos']
    ids_train = clientes.loc[part.idx_train, 'customer_id']
    ultima = tx.groupby('customer_id')['date'].max()
    informe = {'fecha': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

    # 1. la partición es la del Entregable 3 (que partía la tabla ya unida: mismo índice, misma semilla)
    idx_tr, idx_te = train_test_split(p['df'].index, test_size=TEST_SIZE, random_state=SEED)
    informe['particion'] = {'n_train': len(part.idx_train), 'n_test': len(part.idx_test),
                            'igual_que_entregable_3': bool(part.idx_train.equals(pd.Index(idx_tr)) and part.idx_test.equals(pd.Index(idx_te))),
                            'clientes_en_ambas': int(len(set(part.idx_train) & set(part.idx_test)))}

    # 2. fecha de corte antes y después, y celdas que cambian
    corte_todo, corte_train = p['fecha_corte_todo'], p['fecha_corte']
    recencia_antes = (corte_todo - ultima).dt.days
    recencia_ahora = p['df'].set_index('customer_id')['recencia_dias']
    X_antes_train, X_antes_test = c.X_train.copy(), c.X_test.copy()
    X_antes_train['recencia_dias'] = recencia_antes.reindex(ids_train).values
    X_antes_test['recencia_dias'] = recencia_antes.reindex(clientes.loc[part.idx_test, 'customer_id']).values
    informe['fecha_de_corte'] = {
        'entregable_3_todo_el_detalle': str(corte_todo.date()), 'correccion_solo_entrenamiento': str(corte_train.date()),
        'solo_prueba': str(tx.loc[~tx['customer_id'].isin(ids_train), 'date'].max().date()),
        'coinciden': bool(corte_todo == corte_train),
        'clientes_con_ultima_transaccion_posterior_al_corte': p['recencias_recortadas'],
        'celdas_de_recencia_que_cambian': int((recencia_antes.reindex(recencia_ahora.index) != recencia_ahora).sum()),
        'huella_X_train_antes': huella(X_antes_train), 'huella_X_train_ahora': huella(c.X_train),
        'huella_X_test_antes': huella(X_antes_test), 'huella_X_test_ahora': huella(c.X_test),
        'predictores_identicos': bool(huella(X_antes_train) == huella(c.X_train) and huella(X_antes_test) == huella(c.X_test)),
    }

    # 3. invarianza: recencia = corte − última ⇒ estandarizada con media y desviación de entrenamiento,
    #    z = (corte − u − (corte − ū)) / s_u = −(u − ū) / s_u, que no contiene la fecha de corte.
    u_train = ultima.reindex(ids_train)
    u_test = ultima.reindex(clientes.loc[part.idx_test, 'customer_id'])
    diferencias = {}
    for dias in (-30, -1, 0, 7, 365):
        corte = corte_train + pd.Timedelta(days=dias)
        escala = StandardScaler().fit((corte - u_train).dt.days.to_numpy(dtype=float).reshape(-1, 1))
        z_train = escala.transform((corte - u_train).dt.days.to_numpy(dtype=float).reshape(-1, 1))
        z_test = escala.transform((corte - u_test).dt.days.to_numpy(dtype=float).reshape(-1, 1))
        if dias == 0:
            ref_train, ref_test = z_train, z_test
        diferencias[dias] = (z_train, z_test)
    informe['invarianza_de_la_recencia_estandarizada'] = {
        f'corte desplazado {d:+d} días': float(max(np.abs(zt - ref_train).max(), np.abs(ze - ref_test).max()))
        for d, (zt, ze) in diferencias.items()}

    # 4. dentro de la validación anidada: fecha de corte de las filas de entrenamiento de cada pliegue
    distintos, revisados = 0, 0
    fechas_train = u_train.reset_index(drop=True)
    for semilla in SEMILLAS_REPETICION:
        for tarea, y in (('clasificacion', p['y_train'].reset_index(drop=True)), ('regresion', c.y_train_cont.reset_index(drop=True))):
            for i_ext, _ in construir_cv(tarea, K_EXT, semilla).split(fechas_train, y):
                revisados += 1
                distintos += int(fechas_train.iloc[i_ext].max() != corte_train)
                for i_int, _ in construir_cv(tarea, K_INT, semilla).split(fechas_train.iloc[i_ext], y.iloc[i_ext]):
                    revisados += 1
                    distintos += int(fechas_train.iloc[i_ext].iloc[i_int].max() != corte_train)
    informe['pliegues'] = {'pliegues_revisados': revisados, 'pliegues_con_otra_fecha_de_corte': distintos,
                           'clientes_de_entrenamiento_con_transaccion_en_la_fecha_de_corte': int((u_train == corte_train).sum())}

    # 5. resto de la lista de la guía
    X_todo = pd.concat([c.X_train, c.X_test])
    duplicadas = X_todo.duplicated(keep=False)
    informe['lista_de_fuga'] = {
        'preprocesamiento': 'StandardScaler y OneHotEncoder viven dentro de cada pipeline (src/preprocesamiento.py): se '
                            'ajustan solo con las filas de entrenamiento de cada pliegue externo e interno.',
        'variables_que_codifican_la_respuesta': {
            'excluidas': datos.COLUMNAS_A_ELIMINAR,
            'r_active_products_con_el_objetivo_en_entrenamiento': round(p['diagnosticos']['r_active_products'], 4)},
        'duplicados_entre_particiones': {
            'clientes_repetidos': int(clientes['customer_id'].duplicated().sum()),
            'filas_de_predictores_identicas_en_entrenamiento_y_prueba': int(
                (duplicadas.loc[c.X_train.index].sum() > 0) and (duplicadas.loc[c.X_test.index].sum() > 0)
                and len(pd.merge(c.X_train, c.X_test, how='inner')))},
        'umbral_de_la_etiqueta': {'valor': p['etiqueta'].umbral, 'estimado_con': 'entrenamiento', 'n': p['etiqueta'].n_train},
        'mezcla_temporal': 'El conjunto es un corte transversal de 2023: una fila por cliente y partición aleatoria por '
                           f'cliente. El detalle va del {tx["date"].min().date()} al {tx["date"].max().date()}; el único '
                           'parámetro temporal común es la fecha de corte de la recencia, corregida aquí.',
        'modelo_original': 'El puntaje continuo entra como parámetro de ajuste (modelo__puntaje): scikit-learn lo recorta a '
                           'las filas de entrenamiento de cada pliegue; las filas de validación y de prueba nunca lo '
                           'aportan (tests/test_marco.py). Sus repartos internos (ajuste, validación, calibración) se '
                           'hacen dentro de las filas de entrenamiento que recibe.',
    }
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    (CARPETA_RESULTADOS / 'auditoria_fuga.json').write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in informe.items() if k != 'lista_de_fuga'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
