"""Tabla de líderes · ¿Cuál es la mejor combinación de cada modelo?

La primera vista de Modelos. Para cada uno de los diez modelos de la tarea (el modelo original, BDL; los dos benchmarks
recientes, la EBM y LightGBM; y los siete modelos de clase), la combinación (balanceo, solo en clasificación, y optimizador)
con que compite, en una tabla interactiva (dash_table.DataTable, que se ordena y se filtra): la métrica externa como media ±
desviación de los pliegues externos, la de prueba con su intervalo BCa, el costo (inferencia, ajuste final y tamaño) y si
el modelo queda entre los finalistas de la comparación estadística (a menos de la diferencia crítica de Nemenyi del mejor
rango medio de los pliegues externos, datos/cd.json, en las dos tareas). Debajo, la mejor combinación global, dónde queda
el modelo original con su veredicto y, para el modelo elegido (al pulsar una fila o con el selector), cómo se eligió su
combinación, lo que quedó en el modelo final y los enlaces a su predicción en vivo y a su diagnóstico.

Las filas tienen dos orígenes, y la tabla lo dice:
  · Los ocho modelos del Entregable 3 traen la mejor de sus combinaciones con la regla del proyecto (Entregable 3,
    src/analisis.py::mejores_por_modelo): entre las corridas «ok» de cada modelo, la de mayor media_ext (en regresión es el
    RMSE con el signo cambiado); con un empate exacto, la primera de la tabla. La vista la vuelve a aplicar sobre
    datos/entrega3/corridas.csv y avisa si no da la combinación del artefacto o si los archivos no cuadran.
  · BDL y LightGBM vienen de la corrección (datos/original/, que arma preparar_original.py), con una sola combinación,
    fijada en el protocolo pre-registrado antes de ejecutar. LightGBM no tiene artefacto: está en la tabla, no en vivo.
La métrica externa de todos sale de los mismos pliegues (D.lideres: datos/original/pliegues_<tarea>.csv cuando existe), los
que usa el veredicto del modelo original. Ninguna cifra se escribe a mano: todas salen de datos/ o de artifacts/, y el
veredicto, de D.frase_veredicto (veredicto.json y condicion4.json).

    /modelos/lideres?tarea=regresion&modelo=bdl      abre con esa tarea y ese modelo en «Cómo se eligió cada fila»
"""
import json
import math

import pandas as pd
from dash import Input, Output, State, callback, ctx, dash_table, html, no_update
from dash.dash_table.Format import Format, Group, Scheme

import app
import datos_app as D
import formato as F
import modelos_vivos as V
import ui

VISTA = "lideres"
PRE = "ld"
PANTALLA = "comparacion"            # la pantalla cuyo icono lleva la cabecera (la balanza de la tabla de líderes)
O = D.ORIGINAL
E3 = "Entregable 3"                 # el valor de «origen» de las ocho filas del informe en D.lideres
TOLERANCIA = 1e-9                   # dos copias de la misma métrica (csv y metadata.json) pueden diferir en el redondeo del csv

CORRIDAS = "entrega3/corridas.csv"
COSTO = "resultados/costo.csv"
CD = "cd.json"
METADATA = "artifacts/{tarea}/{modelo}/metadata.json"
PROTOCOLO = "ModeloOriginal/experiments/protocolo.md (protocolo pre-registrado: modelos y presupuesto de ajuste)"

# Lo que cambia entre las dos tareas: columnas de datos/resultados, decimales del informe y nombres
TAREA = {
    "clasificacion": {"metrica": "AUC", "d": 4, "externa": "auc_ext_media", "externa_desv": "auc_ext_desv",
                      "externa_nombre": "AUC externo", "prueba": "auc_prueba", "inf": "auc_bca_inf", "sup": "auc_bca_sup",
                      "mayor_es_mejor": True, "que": "clasificación", "filtro": ">", "media": "el AUC medio",
                      "mejor": "mayor AUC medio"},
    "regresion": {"metrica": "RMSE", "d": 5, "externa": "rmse_ext_media", "externa_desv": "rmse_ext_desv",
                  "externa_nombre": "RMSE externo", "prueba": "rmse_prueba", "inf": "rmse_bca_inf", "sup": "rmse_bca_sup",
                  "mayor_es_mejor": False, "que": "regresión", "filtro": "<", "media": "el RMSE medio",
                  "mejor": "menor RMSE medio"},
}
# (Como el informe, «el AUC» va en masculino: «el AUC externo», «el mayor AUC medio».)
D_INFERENCIA, D_AJUSTE, D_TAMANO, D_CORRIDA, D_RANGO = 3, 2, 3, 1, 1     # decimales de la tabla de costo del informe
CIFRAS_HIPERPARAMETRO = 4           # cifras significativas de un hiperparámetro real (0,02361 · 621)
CARDINAL = {1: "un", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
            12: "doce", 15: "quince", 16: "dieciséis"}
ORDINAL_M = {1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto", 5: "quinto", 6: "sexto", 7: "séptimo", 8: "octavo", 9: "noveno",
             10: "décimo"}
FUENTES_TABLA = ([D.fuente(f"resultados/{t}_prueba.csv") for t in TAREA] + [D.fuente_original("lideres_<tarea>.csv"),
                 D.fuente_original("pliegues_<tarea>.csv"), D.fuente(COSTO), D.fuente_original("costo.csv"), D.fuente(CD),
                 D.fuente(CORRIDAS), METADATA])


# --- Textos ---------------------------------------------------------------------------------------------------------------
def _mayuscula(texto):
    return texto[:1].upper() + texto[1:]


def _tras(preposicion, frase):
    """«a» o «de» delante de un nombre en frase, con su contracción: «al árbol de decisión», «del SVM lineal», «de la EBM»."""
    if frase.startswith("el "):
        return {"a": "al", "de": "del"}[preposicion] + frase[2:]
    return f"{preposicion} {frase}"


def _lista(modelos):
    """Varios modelos dentro de una frase: «XGBoost, Random Forest y el árbol de decisión»."""
    return F.enumerar(D.en_frase(m) for m in modelos)


def _cardinal(n):
    return CARDINAL.get(n, F.entero(n))


def _p(p):
    """Un valor p para una frase o una celda: «p = 0,072» o «p < 0,001»."""
    return F.p_valor(p)


def _tarea(tarea):
    return tarea if tarea in TAREA else "clasificacion"


def _combinacion(tarea, balanceo, optimizador):
    """La combinación dentro de una frase: «con ponderación y Genética», «sin balanceo y con Grid» o, en regresión, «con TPE»."""
    opt = D.NOMBRE_OPTIMIZADOR[optimizador]
    if tarea == "regresion":
        return f"con {opt}"
    if balanceo == "ninguno":
        return f"sin balanceo y con {opt}"
    return f"con {D.NOMBRE_BALANCEO[balanceo]} y {opt}"


def _rotulo_combinacion(tarea, balanceo, optimizador):
    """La combinación en un rótulo: «ponderación · Genética» o, en regresión, «TPE»."""
    opt = D.NOMBRE_OPTIMIZADOR[optimizador]
    return opt if tarea == "regresion" else f"{D.NOMBRE_BALANCEO[balanceo]} · {opt}"


def _valor_hiperparametro(v):
    """Un hiperparámetro en es-CO: enteros con punto de miles, reales con CIFRAS_HIPERPARAMETRO cifras significativas."""
    if isinstance(v, bool):
        return "sí" if v else "no"
    if v is None:
        return "ninguno"
    if isinstance(v, int):
        return F.entero(v)
    if isinstance(v, float):
        if v == 0 or not math.isfinite(v):
            return ui.num(v, 0)
        d = max(0, CIFRAS_HIPERPARAMETRO - 1 - math.floor(math.log10(abs(v))))
        texto = ui.num(v, d)
        return texto.rstrip("0").rstrip(",") if "," in texto else texto
    return "automático (BIC)" if v == "auto" else str(v)


def _pliegues_externos(n, externos):
    """«los cinco pliegues externos» o, con varias repeticiones, «los quince pliegues externos (tres repeticiones de cinco)»."""
    texto = f"los {_cardinal(n)} pliegues externos"
    if n > externos and n % externos == 0:
        texto += f" ({_cardinal(n // externos)} repeticiones de {_cardinal(externos)})"
    return texto


# --- Números para la tabla interactiva ------------------------------------------------------------------------------------
def _redondo(x, d):
    """El número redondeado como lo escribe la app (mitad hacia arriba, formato.fmt). La tabla lo muestra con `d` decimales,
    así que da el mismo dígito que ui.num y que el informe, y lo ordena y lo filtra como número."""
    texto = F.fmt(x, d)
    return None if texto == F.RAYA else float(texto.replace(".", "").replace(",", "."))


def _formato(d):
    """Formato es-CO de una columna numérica de la tabla: coma decimal, punto de miles y `d` decimales fijos."""
    return Format(precision=d, scheme=Scheme.fixed, decimal_delimiter=",", group_delimiter=".", group=Group.yes, nully=F.RAYA)


def _num(clave, nombre, d):
    # El sufijo «_num» alinea la columna a la derecha con cifras tabulares (assets/tablas.css)
    return {"id": f"{clave}_num", "name": nombre, "type": "numeric", "format": _formato(d)}


def _texto(clave, nombre):
    return {"id": clave, "name": nombre, "type": "text"}


# --- Datos ----------------------------------------------------------------------------------------------------------------
def _corridas(tarea, modelo):
    c = D.tabla(CORRIDAS)
    return c[(c["tarea"] == tarea) & (c["modelo"] == modelo)]


def _eleccion(tarea, modelo):
    """Cómo se eligió la combinación de un modelo del Entregable 3 entre sus corridas, con la regla del proyecto: la de mayor
    media_ext entre las «ok» (idxmax: con un empate exacto, la primera de la tabla). Devuelve la elegida, la segunda, el
    margen entre las dos, cuántas empatan con la elegida, cuántas quedan a menos de una desviación de ella y las que no se
    aplican. None para BDL y LightGBM, que tienen una sola combinación, la del protocolo de la corrección."""
    todas = _corridas(tarea, modelo)
    if todas.empty:
        return None
    ok = todas[todas["estado"] == "ok"]
    elegida = ok.loc[ok["media_ext"].idxmax()]
    resto = ok.drop(index=elegida.name)
    segunda = resto.loc[resto["media_ext"].idxmax()] if not resto.empty else None
    margen = float(elegida["media_ext"] - segunda["media_ext"]) if segunda is not None else None
    empates = int((ok["media_ext"] == elegida["media_ext"]).sum())
    cerca = int(((elegida["media_ext"] - ok["media_ext"]) < elegida["desv_ext"]).sum()) - 1     # sin contar a la elegida
    return {"todas": todas, "ok": ok, "elegida": elegida, "segunda": segunda, "margen": margen, "empates": empates,
            "cerca": cerca, "no_aplica": todas[todas["estado"] != "ok"]}


def _finalistas(tarea):
    """Los finalistas de la comparación estadística, en las dos tareas: los modelos a menos de la diferencia crítica de
    Nemenyi del mejor rango medio en los pliegues externos (datos/cd.json, con todos los modelos de la tabla cuando la
    corrección ya los dejó; dos modelos difieren si sus rangos medios se separan al menos esa diferencia). `valor` es el
    rango medio de cada modelo."""
    n = D.json(CD)[tarea]
    rangos = pd.Series(n["rango_medio"], dtype=float)
    mejor = float(rangos.min())
    return {"dentro": [m for m in rangos.index if rangos[m] - mejor < n["diferencia_critica"]], "valor": rangos.to_dict(),
            "mejor": mejor, "dc": float(n["diferencia_critica"]), "cd": n}


def _lideres(tarea):
    """Una fila por modelo con su combinación, en el orden del bucle externo (puesto 1 = la mejor métrica externa), y la
    lista de lo que no cuadra entre los archivos (vacía si todo coincide)."""
    c = TAREA[tarea]
    t = D.lideres(tarea)
    costo = D.tabla(COSTO)
    costo = costo[costo["tarea"] == tarea].set_index("modelo")
    fin = _finalistas(tarea)
    en_vivo = set(V.modelos(tarea))
    externos, _internos = _pliegues(tarea)
    filas, problemas = [], []
    for _, f in t.iterrows():
        m = f["modelo"]
        nombre = D.NOMBRE_MODELO[m]
        meta = V.metadata(tarea, m) if m in en_vivo else None
        balanceo = str(f["balanceo"]) if "balanceo" in f.index else (str(meta["balanceo"]) if meta else "ninguno")
        optimizador = str(f["optimizador"])
        e = _eleccion(tarea, m) if f["origen"] == E3 else None
        if f["origen"] == E3:
            elegida = (str(e["elegida"]["balanceo"]), str(e["elegida"]["optimizador"]))
            if not (elegida == (balanceo, optimizador) == (meta["balanceo"], meta["optimizador"])):
                problemas.append(f"{nombre}: la regla de elección da {_rotulo_combinacion(tarea, *elegida)}, la tabla de resultados "
                                 f"{_rotulo_combinacion(tarea, balanceo, optimizador)} y el artefacto "
                                 f"{_rotulo_combinacion(tarea, meta['balanceo'], meta['optimizador'])}.")
            if abs(e["elegida"]["metrica_externa"] - f[c["externa"]]) > TOLERANCIA:
                problemas.append(f"{nombre}: la métrica externa no coincide entre {D.fuente(f'resultados/{tarea}_prueba.csv')} y "
                                 f"{D.fuente(CORRIDAS)}.")
            k = costo.loc[m] if m in costo.index else None
        else:
            if meta is not None and (balanceo, optimizador) != (meta["balanceo"], meta["optimizador"]):
                problemas.append(f"{nombre}: la tabla de la corrección da {_rotulo_combinacion(tarea, balanceo, optimizador)} y el "
                                 f"artefacto {_rotulo_combinacion(tarea, meta['balanceo'], meta['optimizador'])}.")
            k = D.costo_original(tarea, m)
        if meta is not None:
            externa = meta["metricas_bucle_externo"]
            if (abs(externa["media"] - f[c["externa"]]) > TOLERANCIA or abs(externa["desviacion"] - f[c["externa_desv"]]) > TOLERANCIA):
                problemas.append(f"{nombre}: la métrica externa de la tabla de resultados no coincide con la de su metadata.json.")
        if int(f["externa_pliegues"]) == externos and abs(f["externa_media"] - f[c["externa"]]) > TOLERANCIA:
            problemas.append(f"{nombre}: la métrica externa de {D.fuente_original(f'pliegues_{tarea}.csv')} no coincide con la de la "
                             "tabla de resultados.")
        if k is None:
            problemas.append(f"{nombre}: no está en {D.fuente(COSTO) if f['origen'] == E3 else D.fuente_original('costo.csv')}.")
            continue
        filas.append({"modelo": m, "origen": f["origen"], "en_vivo": m in en_vivo, "balanceo": balanceo, "optimizador": optimizador,
                      "externa": float(f["externa_media"]), "desv": float(f["externa_desv"]), "pliegues": int(f["externa_pliegues"]),
                      "prueba": float(f[c["prueba"]]), "inf": float(f[c["inf"]]), "sup": float(f[c["sup"]]),
                      "inferencia": float(k["inferencia_s"]), "ajuste": float(k["ajuste_s"]), "tamano": float(k["tamano_mb"]),
                      "parametros": None if pd.isna(k["parametros"]) else int(k["parametros"]), "corrida": float(k["corrida_min"]),
                      "n_ok": len(e["ok"]) if e else 1, "n_todas": len(e["todas"]) if e else 1, "empates": e["empates"] if e else 1,
                      "margen": e["margen"] if e else None,
                      "finalista": (m in fin["dentro"]) if m in fin["valor"] else None,
                      "criterio": float(fin["valor"][m]) if m in fin["valor"] else None})
    L = pd.DataFrame(filas).sort_values("externa", ascending=not c["mayor_es_mejor"], kind="stable").reset_index(drop=True)
    L["puesto"] = L["externa"].rank(ascending=not c["mayor_es_mejor"], method="min").astype(int)
    return L, problemas


def _motivo(tarea, f):
    """Por qué esa fila, en pocas palabras: la mejor de cuántas combinaciones y con qué margen sobre la segunda; para BDL y
    LightGBM, la única, fijada en el protocolo."""
    d = TAREA[tarea]["d"] + 1
    if f["origen"] != E3:
        return "Única, fijada en el protocolo"
    if f["empates"] > 1:
        return f"Empate exacto de {f['empates']}; la primera"
    margen = "" if pd.isna(f["margen"]) else f"; margen {ui.num(f['margen'], d)}"
    return f"Mejor de {f['n_ok']}{margen}"


def _finalista_texto(f):
    if f["finalista"] is None or pd.isna(f["criterio"]):
        return F.RAYA
    return f"{'sí' if f['finalista'] else 'no'} (rango {ui.num(f['criterio'], D_RANGO)})"


def _columnas(tarea):
    """El modelo va primero: la tabla solo pone el texto de ayuda del filtro en la primera columna."""
    c = TAREA[tarea]
    cols = [_texto("modelo", "Modelo"), _num("puesto", "Puesto", 0)]
    if tarea == "clasificacion":
        cols.append(_texto("balanceo", "Balanceo"))
    cols += [_texto("optimizador", "Optimizador"), _num("externa", c["externa_nombre"], c["d"]), _num("desv", "± desv.", c["d"]),
             _texto("finalista", "Finalista (Nemenyi)"), _texto("prueba", f"{c['metrica']} de prueba [IC 95 %]"),
             _num("inferencia", "Inferencia (s)", D_INFERENCIA), _num("ajuste", "Ajuste (s)", D_AJUSTE),
             _num("tamano", "Tamaño (MB)", D_TAMANO), _texto("motivo", "Por qué esta combinación")]
    return cols


def _datos(tarea, L):
    """Las filas de la tabla: los números ya redondeados como en el informe (la tabla les pone el formato es-CO) y el «id»
    de cada fila, que es la clave del modelo (llega en active_cell como row_id aunque se ordene o se filtre)."""
    d = TAREA[tarea]["d"]
    filas = []
    for _, f in L.iterrows():
        filas.append({"id": f["modelo"], "puesto_num": int(f["puesto"]), "modelo": D.NOMBRE_MODELO[f["modelo"]],
                      "balanceo": D.NOMBRE_BALANCEO.get(f["balanceo"], f["balanceo"]),
                      "optimizador": D.NOMBRE_OPTIMIZADOR[f["optimizador"]], "externa_num": _redondo(f["externa"], d),
                      "desv_num": _redondo(f["desv"], d), "motivo": _motivo(tarea, f), "finalista": _finalista_texto(f),
                      # la métrica y su intervalo en un solo texto: con el mismo formato, ordenarlo como texto es ordenarlo por valor
                      "prueba": f"{ui.num(f['prueba'], d)} {F.intervalo(f['inf'], f['sup'], d)}",
                      "inferencia_num": _redondo(f["inferencia"], D_INFERENCIA), "ajuste_num": _redondo(f["ajuste"], D_AJUSTE),
                      "tamano_num": _redondo(f["tamano"], D_TAMANO)})
    return filas


def _opciones(L):
    return [{"label": D.NOMBRE_MODELO[m], "value": m} for m in L["modelo"]]


def _pliegues(tarea):
    """Pliegues externos e internos de cada repetición de la validación anidada (los mismos en todos los artefactos)."""
    externa = V.metadata(tarea, "ebm")["metricas_bucle_externo"]
    return int(externa["pliegues_externos"]), int(externa["pliegues_internos"])


def _foco(L):
    """El modelo cuya posición cuenta el encabezado: el original si ya está en la tarea; si no, la EBM."""
    return O if O in set(L["modelo"]) else "ebm"


# --- El encabezado y la tabla ---------------------------------------------------------------------------------------------
def _veredicto(tarea):
    return f"Con la regla de superioridad de la guía, el veredicto de BDL es: {D.frase_veredicto(tarea)}."


def _respuesta(tarea):
    """El subtítulo de la pantalla: la mejor combinación global, si es finalista, y dónde queda el modelo original."""
    c = TAREA[tarea]
    L, _ = _lideres(tarea)
    g, s = L.iloc[0], L.iloc[1]
    externa = f"{c['externa_nombre']} de {ui.num(g['externa'], c['d'])} ± {ui.num(g['desv'], c['d'])}"
    estado = "está entre los finalistas de Nemenyi" if g["finalista"] else "no está entre los finalistas de Nemenyi"
    texto = (f"En {c['que']}, la mejor combinación global es {D.en_frase(g['modelo'])} {_combinacion(tarea, g['balanceo'], g['optimizador'])} "
             f"({externa}), {ui.num(abs(g['externa'] - s['externa']), c['d'] + 1)} por delante {_tras('de', D.en_frase(s['modelo']))}, "
             f"y {estado}.")
    if O not in set(L["modelo"]):
        return texto + " El modelo original, BDL, todavía no tiene resultados en esta tarea."
    if g["modelo"] == O:
        return f"{texto} Es el modelo original del proyecto. {_veredicto(tarea)}"
    e = L[L["modelo"] == O].iloc[0]
    finalista = "también es finalista" if e["finalista"] and g["finalista"] else ("es finalista" if e["finalista"] else "no es finalista")
    return (f"{texto} El modelo original, BDL, queda {ORDINAL_M.get(int(e['puesto']), F.entero(e['puesto']))} "
            f"({ui.num(e['externa'], c['d'])} ± {ui.num(e['desv'], c['d'])}) y {finalista}. {_veredicto(tarea)}")


def _textos_tabla(tarea):
    """Título, subtítulo y notas de la tarjeta de la tabla."""
    c = TAREA[tarea]
    L, _ = _lideres(tarea)
    fin = _finalistas(tarea)
    g = L.iloc[0]
    externos, internos = _pliegues(tarea)
    n_fin = int(sum(bool(x) for x in L["finalista"]))
    titulo = (f"{D.NOMBRE_MODELO[g['modelo']]} encabeza el bucle externo con {ui.num(g['externa'], c['d'])} de {c['metrica']}; "
              f"{_cardinal(n_fin)} de los {_cardinal(len(L))} modelos quedan a menos de la diferencia crítica de Nemenyi del mejor rango "
              "medio")
    combinaciones = (f"{len(D.ORDEN_BALANCEOS)} balanceos × {len(D.ORDEN_OPTIMIZADORES)} optimizadores" if tarea == "clasificacion"
                     else f"{len(D.ORDEN_OPTIMIZADORES)} optimizadores; en regresión no hay balanceo")
    nuevos = list(L.loc[L["origen"] != E3, "modelo"])
    sub = (f"Una fila por modelo, con la combinación con que compite. Los {_cardinal(len(L) - len(nuevos))} modelos del Entregable 3 "
           f"traen la mejor de las suyas ({combinaciones}), la de {c['mejor']} en los {_cardinal(externos)} pliegues externos de la "
           f"validación anidada {externos} × {internos}"
           + (f"; {_lista(nuevos)}, la única que fijó el protocolo de la corrección" if nuevos else "")
           + ". Así se eligió, sin mirar la prueba; la prueba se midió una sola vez, con el modelo final.")
    foco = L.loc[L["modelo"] == _foco(L), "externa"].iloc[0]
    ejemplo = math.floor(foco * 100) / 100 if c["mayor_es_mejor"] else math.ceil(foco * 100) / 100
    texto_ejemplo = D.NOMBRE_OPTIMIZADOR["genetica"]
    n_pliegues = int(L["pliegues"].iloc[0])
    costos = ("Los dos tiempos y el tamaño de los modelos del Entregable 3 son los de la tabla de costo computacional del informe; los de "
              f"{_lista(nuevos)} los midió la corrección ({D.fuente_original('costo.csv')}), en el mismo equipo." if nuevos else
              "Los dos tiempos y el tamaño son los de la tabla de costo computacional del informe, en el equipo de referencia.")
    notas = [ui.nota(f"Ordene con la flecha de cada encabezado y filtre escribiendo en la fila de debajo: un texto («{texto_ejemplo}», "
                     f"«sí») o, en las columnas numéricas, una condición con punto decimal («{c['filtro']} {ejemplo:.2f}»). Pulse una fila "
                     "para ver más abajo cómo se eligió, con los enlaces a su predicción en vivo y a su diagnóstico."),
             ui.nota(f"{c['externa_nombre']} y ± desv.: media y desviación de {_pliegues_externos(n_pliegues, externos)}, los mismos para "
                     f"todos los modelos. Finalista: a menos de la diferencia crítica de Nemenyi ({ui.num(fin['dc'], 2)} rangos) del mejor "
                     f"rango medio de los pliegues ({ui.num(fin['mejor'], D_RANGO)}); entre paréntesis, el rango medio del modelo. "
                     f"{c['metrica']} de prueba con su intervalo bootstrap BCa del 95 %. Inferencia: puntuar los "
                     f"{F.entero(D.resumen()['n_prueba'])} clientes de prueba; ajuste: entrenar el modelo final. {costos} Por qué esta "
                     "combinación: entre cuántas se eligió y su margen sobre la segunda."
                     + (" LightGBM no se sirve en vivo: no tiene artefacto en la app." if "lightgbm" in set(L["modelo"]) else "")),
             html.Div(ui.boton("Ir a cómo se eligió cada fila", href=f"#{PRE}-eleccion", icono="flecha-abajo", pequeno=True),
                      className="botonera")]
    return titulo, sub, notas


def _avisos(problemas):
    if not problemas:
        return None
    return ui.aviso(problemas, "critico", "Los archivos no cuadran: revise antes de usar esta tabla")


# --- La mejor combinación global y los finalistas -------------------------------------------------------------------------
def _global(tarea):
    """Dos tarjetas: la mejor combinación global (con su margen sobre la segunda y si es finalista) y los finalistas, con la
    lectura de lo que significa serlo y el veredicto del modelo original."""
    c = TAREA[tarea]
    L, _ = _lideres(tarea)
    fin = _finalistas(tarea)
    g, s = L.iloc[0], L.iloc[1]
    d = c["d"]
    margen = abs(g["externa"] - s["externa"])
    if ui.num(g["externa"], d) == ui.num(s["externa"], d):
        lugar = (f"{_mayuscula(D.en_frase(s['modelo']))}, segundo, queda a {ui.num(margen, d + 1)}: con {_cardinal(d)} decimales los "
                 f"dos dan {ui.num(g['externa'], d)}.")
    else:
        lugar = f"{ui.num(margen, d + 1)} por delante {_tras('de', D.en_frase(s['modelo']))} ({ui.num(s['externa'], d)})."
    debil = ("El margen es menor que la desviación entre pliegues: el primer puesto no es una ventaja establecida."
             if margen < g["desv"] else None)
    externa = ui.cifra(c["externa_nombre"], ui.num(g["externa"], d), f"± {ui.num(g['desv'], d)}", variacion=lugar, nota=debil, mini=True)
    estado = ui.chip("finalista", "bien") if g["finalista"] else ui.chip("no es finalista")
    r = fin["valor"][g["modelo"]]
    iguales = [m for m, x in fin["valor"].items() if x == fin["mejor"] and m != g["modelo"]]
    if r == fin["mejor"]:
        lugar_rango = "Es el mejor rango medio" + (f", empatado con {_lista(iguales)}" if iguales else "")
    else:
        lugar_rango = f"A {ui.num(r - fin['mejor'], D_RANGO)} del mejor rango medio"
    segunda = ui.cifra("Rango medio en los pliegues", ui.num(r, D_RANGO), chip=estado, mini=True,
                       variacion=f"{lugar_rango}; la diferencia crítica de Nemenyi es {ui.num(fin['dc'], 2)}.")
    prueba = ui.nota(f"La métrica externa eligió sin mirar la prueba. En la prueba, {D.en_frase(g['modelo'])} obtiene "
                     f"{ui.num(g['prueba'], d)} de {c['metrica']}, con intervalo BCa del 95 % {F.intervalo(g['inf'], g['sup'], d)}.")
    titulo_a = (f"La mejor combinación global, {D.en_frase(g['modelo'])} {_combinacion(tarea, g['balanceo'], g['optimizador'])}, "
                f"{'está' if g['finalista'] else 'no está'} entre los finalistas")
    fuente_fila = D.fuente(f"resultados/{tarea}_prueba.csv") if g["origen"] == E3 else D.fuente_original(f"lideres_{tarea}.csv")
    tarjeta_a = ui.tarjeta([ui.cifras([externa, segunda]), prueba], titulo=titulo_a, fuentes=[fuente_fila, D.fuente(CD)])

    dentro = [m for m in L["modelo"] if m in fin["dentro"]]
    fuera = [m for m in L["modelo"] if m not in fin["dentro"]]
    n = fin["cd"]
    hay_original = O in set(L["modelo"])
    titulo_b = f"{_mayuscula(_cardinal(len(dentro)))} de los {_cardinal(len(L))} modelos quedan entre los finalistas de Nemenyi"
    if hay_original:
        titulo_b += ", BDL entre ellos" if O in dentro else "; BDL no"
    rechaza = n["p_friedman"] < D.ALFA
    lectura = [f"Finalista es el modelo que queda a menos de la diferencia crítica ({ui.num(fin['dc'], 2)} rangos) del mejor rango medio "
               f"({ui.num(fin['mejor'], D_RANGO)}). Friedman {'rechaza' if rechaza else 'no rechaza'} que los "
               f"{_cardinal(int(n['modelos']))} modelos rindan igual en los pliegues ({_p(n['p_friedman'])}), pero con "
               f"{_cardinal(int(n['pliegues']))} pliegues esa diferencia es grande"
               + (f": solo {_lista(fuera)} {'quedan' if len(fuera) > 1 else 'queda'} fuera." if fuera else ": ningún modelo queda fuera.")
               + " Ser finalista es un filtro débil."]
    if hay_original:
        lectura.append("Lo que decide si el modelo original supera a los benchmarks es la regla de superioridad de la guía, con las "
                       f"pruebas por pares frente al mejor benchmark. Con ella, el veredicto de BDL es: {D.frase_veredicto(tarea)}.")
    chips = html.Div([ui.chip(D.NOMBRE_CORTO[m], "bien") for m in dentro], className="chips")
    botones = [ui.boton("Ver la comparación estadística", href=app.ruta_vista("comparacion") + f"?tarea={tarea}", icono="balanza")]
    if hay_original:
        botones.append(ui.boton("Ver el modelo original", href=app.ruta_vista("original") + f"?tarea={tarea}", icono="modelo"))
    fuentes_b = [D.fuente(CD)] + ([D.fuente_original("veredicto.json"), D.fuente_original("condicion4.json")] if hay_original else [])
    tarjeta_b = ui.tarjeta([chips] + [ui.parrafo(x) for x in lectura] + [html.Div(botones, className="botonera")],
                           titulo=titulo_b, fuentes=fuentes_b)
    return ui.rejilla([tarjeta_a, tarjeta_b], "1-1")


def _sub_global(tarea):
    return ("La mejor combinación global es la primera fila de la tabla. Los finalistas son los modelos que la prueba de Friedman y el "
            "post hoc de Nemenyi, sobre los rangos de cada pliegue externo, no separan del mejor.")


# --- Cómo se eligió cada fila ---------------------------------------------------------------------------------------------
def _rejilla_combinaciones(tarea, e):
    """La métrica externa de cada combinación del modelo: balanceo × optimizador en clasificación, un optimizador por fila en
    regresión. La elegida va en negrita con la palabra «elegida»."""
    c = TAREA[tarea]
    d = c["d"]
    todas, elegida = e["todas"], e["elegida"]
    optimizadores = [o for o in D.ORDEN_OPTIMIZADORES if o in set(todas["optimizador"])]
    filas, mejor = [], []
    if tarea == "clasificacion":
        balanceos = [b for b in D.ORDEN_BALANCEOS if b in set(todas["balanceo"])]
        for i, b in enumerate(balanceos):
            fila = [_mayuscula(D.NOMBRE_BALANCEO[b])]
            for j, o in enumerate(optimizadores, 1):
                r = todas[(todas["balanceo"] == b) & (todas["optimizador"] == o)]
                fila.append("no se aplica" if r.empty or r.iloc[0]["estado"] != "ok" else ui.num(r.iloc[0]["metrica_externa"], d))
                if (b, o) == (elegida["balanceo"], elegida["optimizador"]):
                    mejor.append((i, j))
            filas.append(fila)
        encabezados = ["Balanceo"] + [D.NOMBRE_OPTIMIZADOR[o] for o in optimizadores]
    else:
        for i, o in enumerate(optimizadores):
            r = todas[todas["optimizador"] == o].iloc[0]
            filas.append([D.NOMBRE_OPTIMIZADOR[o], ui.num(r["metrica_externa"], d), ui.num(r["desv_ext"], d),
                          ui.num(r["tiempo_total_s"] / 60, D_CORRIDA)])
            if o == elegida["optimizador"]:
                mejor.append((i, 1))
        encabezados = ["Optimizador", c["externa_nombre"], "Desviación", "Corrida (min)"]
    return ui.tabla(encabezados, filas, mejor=mejor, mejor_texto="elegida", compacta=True)


def _fidelidad(e):
    """La búsqueda de dos fidelidades (Random Forest y la EBM), dicha con precisión: no es Successive Halving."""
    texto = e["elegida"].get("fidelidad")
    if not isinstance(texto, str) or not texto.strip():
        return None
    f = json.loads(texto)

    def pares(dic):
        return F.enumerar(f"{k.split('__', 1)[-1]} = {_valor_hiperparametro(v)}" for k, v in dic.items())
    return ui.nota(f"La búsqueda evaluó cada configuración con {pares(f.get('busqueda', {}))} y el modelo final se reajustó con "
                   f"{pares(f.get('final', {}))}: son dos fidelidades fijas, no Successive Halving (no hay rondas que descarten "
                   "configuraciones con un factor de reducción).")


def _enlaces(tarea, modelo, en_vivo):
    """A las otras vistas de ese modelo, con la tarea y el modelo en la dirección. Un modelo sin artefacto no tiene
    predicción en vivo; el original lleva, además, a su vista."""
    c = TAREA[tarea]
    consulta = f"?tarea={tarea}&modelo={modelo}"
    icono = {v: i for v, _n, _p, _g, _m, i in app.VISTAS}
    nombre_vista = {v: n_ for v, n_, *_ in app.VISTAS}
    frase = D.en_frase(modelo)
    botones = []
    if en_vivo:
        botones.append(ui.boton(f"{nombre_vista['prediccion']} con {frase}", href=app.ruta_vista("prediccion") + consulta, tipo="primario",
                                icono=icono["prediccion"], titulo=f"Abre la vista de predicción en {c['que']}, con este modelo ya elegido"))
    botones.append(ui.boton(f"{nombre_vista['diagnostico']} {_tras('de', frase)}", href=app.ruta_vista("diagnostico") + consulta,
                            icono=icono["diagnostico"], titulo=f"Abre el diagnóstico de este modelo en la prueba, en {c['que']}"))
    if modelo == O:
        botones.append(ui.boton(nombre_vista["original"], href=app.ruta_vista("original") + f"?tarea={tarea}", icono=icono["original"],
                                titulo="Abre la vista del modelo original: qué es, cómo aprende y su veredicto"))
    return html.Div(botones, className="botonera")


def _eleccion_entregable3(tarea, f, e):
    """Título, tabla y frases de cómo se eligió la combinación de un modelo del Entregable 3 entre sus corridas."""
    c = TAREA[tarea]
    d = c["d"]
    nombre = D.NOMBRE_MODELO[f["modelo"]]
    el = e["elegida"]
    combinacion = _rotulo_combinacion(tarea, el["balanceo"], el["optimizador"])
    externos, internos = _pliegues(tarea)
    n_ok = len(e["ok"])
    if e["empates"] > 1:
        titulo = (f"{nombre}: {_cardinal(e['empates'])} combinaciones empatan en {ui.num(el['metrica_externa'], d)} de "
                  f"{c['externa_nombre']}; queda {combinacion}, la primera de la tabla de corridas")
    elif e["segunda"] is not None:
        titulo = (f"{nombre}: {combinacion}, la mejor de sus {_cardinal(n_ok)} combinaciones por {ui.num(e['margen'], d + 1)} de "
                  f"{c['externa_nombre']}")
    else:
        titulo = f"{nombre}: {combinacion}, su única combinación"
    frases = [f"Cada celda es {c['media']} de los {_cardinal(externos)} pliegues externos de la validación anidada "
              f"{externos} × {internos}; gana el {'mayor' if c['mayor_es_mejor'] else 'menor'}, sin mirar la prueba."]
    if e["empates"] > 1:
        iguales = e["ok"][e["ok"]["media_ext"] == el["media_ext"]]
        hp = sorted(set(iguales["hiperparametros"]))
        frases.append(f"Las {_cardinal(e['empates'])} empatadas dan la misma métrica hasta el último decimal"
                      + (" porque la búsqueda llegó a los mismos hiperparámetros" if len(hp) == 1 else "")
                      + "; la regla del proyecto deja la primera.")
    redondean = int((e["ok"]["metrica_externa"].map(lambda x: ui.num(x, d)) == ui.num(el["metrica_externa"], d)).sum()) - e["empates"]
    if redondean > 0:
        frases.append(f"{_mayuscula(_cardinal(redondean))} {'combinación más da' if redondean == 1 else 'combinaciones más dan'} "
                      f"{ui.num(el['metrica_externa'], d)} con {_cardinal(d)} decimales, pero {'queda' if redondean == 1 else 'quedan'} "
                      "por debajo en los decimales siguientes.")
    if e["segunda"] is not None:
        s = e["segunda"]
        otra = _rotulo_combinacion(tarea, s["balanceo"], s["optimizador"])
        if e["cerca"] == n_ok - 1:
            cuantas = f"las otras {_cardinal(n_ok - 1)} combinaciones quedan todas a menos de una desviación de ella"
        else:
            cuantas = (f"{e['cerca']} de las otras {n_ok - 1} combinaciones {'queda' if e['cerca'] == 1 else 'quedan'} a menos de una "
                       "desviación de ella")
        frases.append(f"La segunda es {otra} ({ui.num(s['metrica_externa'], d)}). La desviación entre pliegues de la elegida es "
                      f"{ui.num(el['desv_ext'], d)}: {cuantas}; entre esas, la diferencia es menor que la variación entre pliegues."
                      if e["cerca"] > 0 else
                      f"La segunda es {otra} ({ui.num(s['metrica_externa'], d)}), a más de una desviación entre pliegues "
                      f"({ui.num(el['desv_ext'], d)}).")
    if not e["no_aplica"].empty:
        motivos = sorted({str(x) for x in e["no_aplica"]["motivo"].dropna()})
        balanceos = sorted({D.NOMBRE_BALANCEO[b] for b in e["no_aplica"]["balanceo"]})
        frases.append(f"Las {len(e['no_aplica'])} corridas con {F.enumerar(balanceos)} no se aplican"
                      + (f" (motivo registrado: «{'; '.join(motivos)}»)." if motivos else "."))
    if f["pliegues"] > externos:
        frases.append(f"La tabla de líderes promedia {_pliegues_externos(int(f['pliegues']), externos)}; la combinación se eligió con la "
                      f"repetición de la semilla {F.entero(V.metadata(tarea, f['modelo'])['semilla'])}.")
    return titulo, _rejilla_combinaciones(tarea, e), frases, [D.fuente(CORRIDAS), METADATA.format(tarea=tarea, modelo=f["modelo"])]


def _eleccion_correccion(tarea, f):
    """Lo mismo para BDL y LightGBM: una sola combinación, fijada en el protocolo pre-registrado antes de ejecutar."""
    c = TAREA[tarea]
    d = c["d"]
    m = f["modelo"]
    nombre = D.NOMBRE_MODELO[m]
    combinacion = _rotulo_combinacion(tarea, f["balanceo"], f["optimizador"])
    externos, internos = _pliegues(tarea)
    k = D.costo_original(tarea, m)
    corrida = D.corrida_original(tarea, m)
    titulo = f"{nombre}: {combinacion}, la única combinación, fijada en el protocolo antes de ejecutar"
    fila = [combinacion if tarea == "regresion" else _mayuscula(combinacion), ui.num(f["externa"], d), ui.num(f["desv"], d),
            F.entero(k["presupuesto"]), F.entero(k["evaluaciones"]), ui.num(f["corrida"], D_CORRIDA)]
    # Encabezados cortos: los de una tabla no parten línea y con los largos la última columna no cabía en la tarjeta
    tabla = ui.tabla(["Combinación", c["externa_nombre"], "Desviación", "Configuraciones", "Evaluaciones", "Corrida (min)"], [fila],
                     compacta=True, ocultar_movil=(4,))
    otros = len(D.tabla(f"resultados/{tarea}_prueba.csv"))
    frases = [f"La corrección fijó en su protocolo pre-registrado, antes de ejecutar, una sola combinación para {D.en_frase(m)}: "
              f"{_combinacion(tarea, f['balanceo'], f['optimizador']).removeprefix('con ')}, con {F.entero(k['presupuesto'])} "
              "configuraciones en cada búsqueda"
              + (" (el mínimo que recibió un benchmark: el original no tuvo más presupuesto que ninguno)." if m == O else ".")
              + f" Los {_cardinal(otros)} modelos del Entregable 3 compiten con la mejor de sus combinaciones. En la tabla, "
              "«Configuraciones» son las que prueba cada búsqueda y «Evaluaciones», las de toda la corrida.",
              f"La métrica es {c['media']} de {_pliegues_externos(int(f['pliegues']), externos)} de la validación anidada "
              f"{externos} × {internos}, sin mirar la prueba."]
    if m == O and corrida is not None and corrida.get("params_por_pliegue"):
        sin = sum(1 for p in corrida["params_por_pliegue"] if p.get("modelo__usar_puntaje", True) is False)
        if sin:
            frases.append(f"En {F.entero(sin)} de los {F.entero(len(corrida['params_por_pliegue']))} pliegues externos de la semilla "
                          f"{F.entero(corrida['semilla'])}, la búsqueda interna eligió la variante que no usa el puntaje continuo "
                          "(usar_puntaje = no): la vista del modelo original mide cuánto pesa eso en el resultado.")
    if not f["en_vivo"]:
        frases.append(f"{nombre} no se sirve en vivo: no tiene artefacto en la app, para no añadir una dependencia al servidor. Sus "
                      "cifras salen de las predicciones que guardó la corrección.")
    fuentes = [D.fuente_original(f"lideres_{tarea}.csv"), D.fuente_original("experimentos.csv"), D.fuente_original("costo.csv"), PROTOCOLO]
    return titulo, tabla, frases, fuentes


def _detalle(tarea, modelo):
    """Para un modelo: cómo se eligió su combinación, lo que quedó en el modelo final y los enlaces a su predicción en vivo y
    a su diagnóstico."""
    L, _ = _lideres(tarea)
    if modelo not in set(L["modelo"]):
        modelo = L.iloc[0]["modelo"]
    f = L[L["modelo"] == modelo].iloc[0]
    e = _eleccion(tarea, modelo) if f["origen"] == E3 else None
    meta = V.metadata(tarea, modelo) if f["en_vivo"] else None
    combinacion = _rotulo_combinacion(tarea, f["balanceo"], f["optimizador"])
    titulo, tabla, frases, fuentes = _eleccion_entregable3(tarea, f, e) if e is not None else _eleccion_correccion(tarea, f)
    if f["finalista"] is None:
        chip_fin = ui.chip("sin rango en la comparación")
    else:
        chip_fin = ui.chip("finalista", "bien") if f["finalista"] else ui.chip("no es finalista")
    chips = [chip_fin, ui.chip_valor("Puesto en la tabla:", F.entero(f["puesto"]))]
    if modelo == O:
        chips.append(ui.chip("modelo original", "acento"))
    izquierda = ui.tarjeta([html.Div(chips, className="chips"), tabla, ui.parrafo(" ".join(frases)),
                            _enlaces(tarea, modelo, bool(f["en_vivo"]))], titulo=titulo, fuentes=fuentes)

    # --- lo que quedó en el modelo final
    n = F.entero(D.resumen()["n_prueba"])
    corrida = None if e is not None else D.corrida_original(tarea, modelo)
    hiperparametros = meta["hiperparametros"] if meta is not None else ((corrida["hiperparametros"] or {}) if corrida is not None else {})
    estimador = str(meta["estimador"]).rsplit(".", 1)[-1] if meta is not None else f"{D.NOMBRE_MODELO[modelo]} (sin artefacto en la app)"
    semilla = meta["semilla"] if meta is not None else (corrida["semilla"] if corrida is not None else None)
    pares = [("Combinación", combinacion), ("Estimador", estimador)]
    pares += [(k.split("__", 1)[-1], _valor_hiperparametro(v)) for k, v in hiperparametros.items()]
    pares += ([("Semilla", str(int(semilla)))] if semilla is not None else [])
    pares += [("Ajuste del modelo final", f"{ui.num(f['ajuste'], D_AJUSTE)} s"),
              (f"Inferencia de {n} clientes", f"{ui.num(f['inferencia'], D_INFERENCIA)} s"),
              ("Corrida anidada completa", f"{ui.num(f['corrida'], D_CORRIDA)} min"),
              ("Tamaño serializado", f"{ui.num(f['tamano'], D_TAMANO)} MB")]
    hay_parametros = f["parametros"] is not None and not pd.isna(f["parametros"])
    if hay_parametros:
        pares.append(("Parámetros almacenados", F.entero(f["parametros"])))
    final_cuerpo = [ui.ficha(pares)]
    fidelidad = _fidelidad(e) if e is not None else None
    if fidelidad is not None:
        final_cuerpo.append(fidelidad)
    if e is not None:
        final_cuerpo.append(ui.nota("Los nombres de los hiperparámetros son los de la librería; los tiempos, el tamaño y los parámetros "
                                    "son los de la tabla de costo computacional del informe."))
        fuentes_final = [METADATA.format(tarea=tarea, modelo=modelo), D.fuente(COSTO)]
    else:
        k = D.costo_original(tarea, modelo)
        final_cuerpo.append(ui.nota(("Los hiperparámetros son los del código del modelo (src/original_model.py)" if modelo == O
                                     else "Los nombres de los hiperparámetros son los de la librería")
                                    + f"; los tiempos y el tamaño los midió la corrección: {k['de']}."))
        fuentes_final = ([METADATA.format(tarea=tarea, modelo=modelo)] if meta is not None else [D.fuente_original("experimentos.csv")]) \
            + [D.fuente_original("costo.csv")]
    guarda = (f"El modelo final guarda {F.entero(f['parametros'])} parámetros" if hay_parametros
              else f"El modelo final ocupa {ui.num(f['tamano'], D_TAMANO)} MB")
    final = ui.tarjeta(final_cuerpo, titulo=f"{guarda} y puntúa los {n} clientes de prueba en {ui.num(f['inferencia'], D_INFERENCIA)} s",
                       fuentes=fuentes_final)
    return ui.rejilla([izquierda, final], "2-1")


def _sub_eleccion(tarea):
    externos, _internos = _pliegues(tarea)
    return (f"Elija un modelo (o pulse su fila en la tabla). En los modelos del Entregable 3, cada combinación es una corrida de la "
            f"validación anidada, con {TAREA[tarea]['media']} de los {_cardinal(externos)} pliegues externos, y gana la mejor, con la regla "
            "del proyecto; BDL y LightGBM tienen una sola, la del protocolo de la corrección.")


# --- La pantalla ----------------------------------------------------------------------------------------------------------
def layout(tarea=None, modelo=None, **_):
    """`/modelos/lideres?tarea=regresion&modelo=bdl` abre con esa tarea y ese modelo (un valor desconocido cae en
    clasificación y en la mejor combinación global). Cuando la dirección pide una tarea, el control no repone el valor
    recordado en la sesión: manda la dirección."""
    pedida = tarea in TAREA
    inicial = tarea if pedida else "clasificacion"
    L, problemas = _lideres(inicial)
    elegido = modelo if modelo in set(L["modelo"]) else L.iloc[0]["modelo"]
    control_tarea = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", [{"label": D.NOMBRE_TAREA[t], "value": t} for t in TAREA],
                                                       inicial, persistir=not pedida))
    selector = ui.control("Modelo:", ui.selector(f"{PRE}-modelo", _opciones(L), elegido, persistir=False))
    tabla = dash_table.DataTable(
        id=f"{PRE}-tabla", columns=_columnas(inicial), data=_datos(inicial, L), sort_action="native", sort_mode="single",
        filter_action="native", filter_options={"case": "insensitive", "placeholder_text": "Filtro"}, page_action="none",
        style_as_list_view=True)
    titulo, sub, notas = _textos_tabla(inicial)
    return ui.pantalla([
        ui.encabezado(app.pregunta(VISTA), html.Span(_respuesta(inicial), id=f"{PRE}-respuesta"), [control_tarea], pantalla=PANTALLA),
        ui.seccion("La tabla de líderes", [
            html.Div(_avisos(problemas), id=f"{PRE}-avisos"),
            ui.tarjeta([html.Div(tabla, className="tabla-datos"), html.Div(notas, id=f"{PRE}-tabla-nota")],
                       titulo=html.Span(titulo, id=f"{PRE}-tabla-titulo"), subtitulo=html.Span(sub, id=f"{PRE}-tabla-sub"),
                       fuentes=FUENTES_TABLA),
        ], subtitulo=f"La combinación de cada uno de los {_cardinal(len(L))} modelos de la tarea (el modelo original, los dos benchmarks "
                     "recientes y los siete de clase), con su costo y su lugar en la comparación estadística."),
        ui.seccion("La mejor combinación global y los finalistas", [html.Div(_global(inicial), id=f"{PRE}-global")],
                   subtitulo=html.Span(_sub_global(inicial), id=f"{PRE}-s2-sub")),
        ui.seccion("Cómo se eligió cada fila", [html.Div(_detalle(inicial, elegido), id=f"{PRE}-detalle")],
                   subtitulo=html.Span(_sub_eleccion(inicial), id=f"{PRE}-s3-sub"), controles=[selector], id=f"{PRE}-eleccion"),
    ])


@callback(Output(f"{PRE}-respuesta", "children"), Output(f"{PRE}-avisos", "children"), Output(f"{PRE}-tabla-titulo", "children"),
          Output(f"{PRE}-tabla-sub", "children"), Output(f"{PRE}-tabla-nota", "children"), Output(f"{PRE}-tabla", "columns"),
          Output(f"{PRE}-tabla", "data"), Output(f"{PRE}-tabla", "sort_by"), Output(f"{PRE}-tabla", "filter_query"),
          Output(f"{PRE}-tabla", "active_cell"), Output(f"{PRE}-tabla", "selected_cells"), Output(f"{PRE}-global", "children"),
          Output(f"{PRE}-s2-sub", "children"), Output(f"{PRE}-s3-sub", "children"), Input(f"{PRE}-tarea", "value"))
def _por_tarea(tarea):
    """Todo lo que depende solo de la tarea. Al cambiarla se borran el orden, el filtro y la celda activa: las columnas
    cambian (en regresión no hay balanceo)."""
    tarea = _tarea(tarea)
    L, problemas = _lideres(tarea)
    titulo, sub, notas = _textos_tabla(tarea)
    return (_respuesta(tarea), _avisos(problemas), titulo, sub, notas, _columnas(tarea), _datos(tarea, L), [], "", None, [],
            _global(tarea), _sub_global(tarea), _sub_eleccion(tarea))


@callback(Output(f"{PRE}-modelo", "options"), Output(f"{PRE}-modelo", "value"), Input(f"{PRE}-tarea", "value"),
          Input(f"{PRE}-tabla", "active_cell"), State(f"{PRE}-modelo", "value"))
def _elegir(tarea, celda, actual):
    """El modelo de «Cómo se eligió cada fila»: el de la fila pulsada (su id llega en active_cell aunque la tabla esté
    ordenada o filtrada); si no, el que ya estaba, si es de la tarea; si no, la mejor combinación global."""
    L, _ = _lideres(_tarea(tarea))
    modelos = list(L["modelo"])
    if f"{PRE}-tabla" in ctx.triggered_prop_ids.values() and isinstance(celda, dict) and celda.get("row_id") in modelos:
        valor = celda["row_id"]
    elif actual in modelos:
        valor = actual
    else:
        valor = modelos[0]
    return _opciones(L), (no_update if valor == actual else valor)


@callback(Output(f"{PRE}-detalle", "children"), Input(f"{PRE}-modelo", "value"), Input(f"{PRE}-tarea", "value"))
def _por_modelo(modelo, tarea):
    return _detalle(_tarea(tarea), modelo)
