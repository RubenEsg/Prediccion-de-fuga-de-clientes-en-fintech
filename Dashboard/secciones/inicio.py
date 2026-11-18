"""Inicio · ¿Qué encontramos? (pantalla de la versión de nueve; hoy sus piezas las usa la sección Contexto)

La respuesta del informe del Entregable 3 en una pantalla, con la EBM en foco: la cifra principal con lo que la acota, la
EBM frente a cada referencia y lo que hay detrás del resultado. La EBM es hoy un benchmark; el modelo original es BDL y la
cifra principal de Contexto es la suya (paginas/contexto.py::_heroe_original). Contexto sigue usando de aquí TAREA,
NIVEL_MCS y FUENTES_PRUEBAS, y el héroe de la EBM (_heroe, con _saldo) solo como respaldo, mientras la corrección no haya
dejado al original en una tarea. El layout de abajo ya no se monta y su callback se retiró. Es también el ejemplo de
referencia de CONTRATO.md §3: toda cifra sale de datos/ o del motor, cada tarjeta dice sus fuentes y los colores salen de
los roles (aquí, los de las vistas de la EBM: D.ROL_VISTA_EBM).
"""
import dash
import plotly.graph_objects as go
from dash import html

import app
import datos_app as D
import formato as F
import motor as M
import ui


PRE = "in"
RIVAL = "xgboost"                     # el mejor de los modelos de árboles, con el que se compara siempre a la EBM
NIVEL_MCS = 0.90                      # nivel del Model Confidence Set del informe
MILESIMA = 0.001                      # por debajo, una diferencia de AUC se dice «menos de una milésima»
ORDINAL = {1: "primera", 2: "segunda", 3: "tercera", 4: "cuarta", 5: "quinta", 6: "sexta", 7: "séptima", 8: "octava"}

# Lo que cambia entre las dos tareas: nombres de columnas de datos/resultados y decimales del informe
TAREA = {
    "clasificacion": {"metrica": "AUC", "d": 4, "prueba": "auc_prueba", "inf": "auc_bca_inf", "sup": "auc_bca_sup",
                      "externa": "auc_ext_media", "delta": "delta_auc", "p_fijada": "p_delong_holm", "fijada": "DeLong",
                      "rope": "auc", "lineal": "logistica", "e2": "logistica_e2", "mayor_es_mejor": True,
                      "que": "al ordenar a los clientes por riesgo"},
    "regresion": {"metrica": "RMSE", "d": 5, "prueba": "rmse_prueba", "inf": "rmse_bca_inf", "sup": "rmse_bca_sup",
                  "externa": "rmse_ext_media", "delta": "delta_rmse", "p_fijada": "p_dm_holm", "fijada": "Diebold-Mariano",
                  "rope": "rmse", "lineal": "lasso", "e2": "lasso_e2", "mayor_es_mejor": False,
                  "que": "al pronosticar el puntaje de cada cliente"},
}
FUENTES_PRUEBAS = ([D.fuente(f"resultados/{t}_prueba.csv") for t in TAREA] + [D.fuente(f"resultados/pruebas_{t}.csv") for t in TAREA]
                   + [D.fuente("resultados/resumen.json")])


def _puesto_externo(tarea):
    """Puesto de la EBM en la validación anidada (1 = la mejor) y la métrica externa de la mejor."""
    c = TAREA[tarea]
    t = D.prueba(tarea).sort_values(c["externa"], ascending=not c["mayor_es_mejor"]).reset_index(drop=True)
    return int(t.index[t["modelo"] == "ebm"][0]) + 1, float(t[c["externa"]].iloc[0])


def _saldo(tarea, modelo):
    """La EBM frente a una referencia: la diferencia con su signo, su intervalo y lo que dicen las pruebas."""
    c = TAREA[tarea]
    f = D.fila(D.pruebas(tarea), modelo=modelo)
    v = D.veredicto(f, tarea)
    intervalo = F.intervalo(f["delta_bca_inf"], f["delta_bca_sup"], c["d"], signo=True)
    if v == "empata":
        # En regresión el Model Confidence Set sí separa a la referencia que queda sola en él: se dice, como el informe
        t = D.prueba(tarea)
        solo_mcs = (tarea == "regresion" and bool(D.fila(t, modelo=modelo)["en_mcs"]) and not bool(D.fila(t, modelo="ebm")["en_mcs"]))
        mcs = (f"; solo el Model Confidence Set al {F.pct(NIVEL_MCS, 0)} detecta la ventaja de {D.en_frase(modelo)}" if solo_mcs else "")
        texto = (f"Intervalo del 95 %: {intervalo}. Ni la t corregida ni {c['fijada']} encuentran diferencia tras la corrección "
                 f"de Holm{mcs}. Probabilidad de equivalencia práctica: {F.prob(f['p_rope'])}.")
    elif v == "supera":
        texto = (f"Intervalo del 95 %: {intervalo}. La t corregida y {c['fijada']} rechazan la igualdad tras la corrección "
                 f"de Holm ({F.p_valor(max(f['p_t_holm'], f[c['p_fijada']]))} ajustado).")
    else:
        texto = f"Intervalo del 95 %: {intervalo}. {D.VEREDICTO_TEXTO[v].capitalize()}."
    # El signo es la ventaja de la EBM: positivo, queda por delante; negativo, por detrás
    lado = "delante" if f[c["delta"]] > 0 else "detrás"
    return ui.saldo(f"Diferencia con {D.en_frase(modelo)}", f[c["delta"]], texto, unidad=f"de {c['metrica']}: la EBM queda por {lado}",
                    d=c["d"], palabras=(D.VEREDICTO_TEXTO["supera"], D.VEREDICTO_TEXTO["pierde"]),
                    extremo=None if v in ("supera", "pierde") else ("empate" if v == "empata" else D.VEREDICTO_TEXTO[v]))


def _heroe(tarea):
    c = TAREA[tarea]
    t = D.prueba(tarea)
    e, x = D.fila(t, modelo="ebm"), D.fila(t, modelo=RIVAL)
    puesto, mejor_externa = _puesto_externo(tarea)
    v = D.veredictos(tarea)
    superadas, empates = [m for m in v.index if v[m] == "supera"], [m for m in v.index if v[m] == "empata"]
    if tarea == "clasificacion":
        margen = e[c["prueba"]] - x[c["prueba"]]
        cuanto = "menos de una milésima" if margen < MILESIMA else ui.num(margen, c["d"])
        lectura = (f"Es el mayor de los modelos del Entregable 3, por {cuanto} sobre {D.NOMBRE_MODELO[RIVAL]} ({ui.num(x[c['prueba']], c['d'])}). "
                   f"En la validación anidada la EBM queda {ORDINAL[puesto]} ({ui.num(e[c['externa']], c['d'])} frente a "
                   f"{ui.num(mejor_externa, c['d'])} de la mejor).")
    else:
        lectura = (f"Es el segundo menor: {D.NOMBRE_MODELO[RIVAL]} tiene {ui.num(x[c['prueba']], c['d'])}. El Model Confidence Set "
                   f"al {F.pct(NIVEL_MCS, 0)} deja solo a {D.NOMBRE_MODELO[RIVAL]}; la EBM es la última en salir ({F.p_valor(e['p_mcs'])}).")
    lado = [ui.cifra("Referencias que supera", f"{len(superadas)} de {len(v)}", nota=F.enumerar(D.NOMBRE_MODELO[m] for m in superadas) + ".", mini=True),
            ui.cifra("Referencias con las que empata", str(len(empates)), nota=F.enumerar(D.NOMBRE_MODELO[m] for m in empates) + ".", mini=True)]
    return ui.heroe(f"{c['metrica']} de prueba de la EBM, en los {F.entero(M.meta()['n_prueba'])} clientes que el modelo no vio",
                    ui.num(e[c["prueba"]], c["d"]), f"intervalo del 95 %: {F.intervalo(e[c['inf']], e[c['sup']], c['d'])}",
                    lectura, [_saldo(tarea, RIVAL), _saldo(tarea, c["lineal"])], lado=lado,
                    pie=ui.boton("Ver todas las pruebas", href="/modelos/comparacion" + ("?tarea=regresion" if tarea == "regresion" else ""),
                                 icono="balanza"))


def _diferencias(tarea):
    """Figura, título, subtítulo, tabla y etiqueta del gráfico de la EBM frente a cada referencia."""
    c = TAREA[tarea]
    t = D.pruebas(tarea)
    v = D.veredictos(tarea)
    rope = D.resumen()["rope"][c["rope"]]
    nombres = [D.NOMBRE_MODELO[m] for m in t["modelo"]]
    fig = go.Figure()
    for rol, forma in (("rival", "barra"), ("base", "rayada")):          # rosa sólido y gris rayado: nunca la misma forma
        s = t[t["modelo"].map(D.ROL_VISTA_EBM) == rol]
        if s.empty:
            continue
        textos = [f"{D.NOMBRE_MODELO[f['modelo']]}: <b>{ui.con_signo(f[c['delta']], c['d'])}</b> "
                  f"{F.intervalo(f['delta_bca_inf'], f['delta_bca_sup'], c['d'], signo=True)} · {D.VEREDICTO_TEXTO[v[f['modelo']]]}"
                  for _, f in s.iterrows()]
        fig.add_trace(go.Bar(
            y=[D.NOMBRE_MODELO[m] for m in s["modelo"]], x=s[c["delta"]], orientation="h", **ui.traza(rol, forma),
            error_x={"type": "data", "symmetric": False, "array": s["delta_bca_sup"] - s[c["delta"]],
                     "arrayminus": s[c["delta"]] - s["delta_bca_inf"], "color": ui.ROL["tinta_2"], "thickness": ui.ANCHO_LINEA},
            **ui.hover_texto(textos)))
    ui.franja(fig, -rope, rope)
    ui.linea_cero(fig, "x")
    ui.unidad_eje(fig, f"ventaja de la EBM en {c['metrica']}")
    ui.formato_eje(fig, "x", 2 if tarea == "clasificacion" else 3)
    fig.update_layout(hovermode="closest", bargap=0.45)
    fig.update_xaxes(showgrid=True, gridcolor=ui.ROL["rejilla"], showline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=nombres[::-1], tickfont=ui.fuente_svg("tinta_2", "--t-s"))

    n_sup, n_emp = int((v == "supera").sum()), int((v == "empata").sum())
    titulo = f"La EBM supera a {n_sup} de las {len(v)} referencias y empata con las otras {n_emp}"
    sub = (f"Diferencia de {c['metrica']} de prueba entre la EBM y cada modelo, con su intervalo del 95 %. La franja es la banda de equivalencia práctica "
           f"(±{ui.num(rope, 2 if tarea == 'clasificacion' else 4)}): lo que cae dentro no cambia una decisión.")
    filas = [[D.NOMBRE_MODELO[f["modelo"]], ui.con_signo(f[c["delta"]], c["d"]),
              F.intervalo(f["delta_bca_inf"], f["delta_bca_sup"], c["d"], signo=True), F.p_valor(f["p_t_holm"]),
              F.p_valor(f[c["p_fijada"]]), F.prob(f["p_rope"]), D.VEREDICTO_TEXTO[v[f["modelo"]]]] for _, f in t.iterrows()]
    tabla = ui.tabla(["Referencia", "Diferencia", "Intervalo del 95 %", "t corregida (p ajustado)", f"{c['fijada']} (p ajustado)",
                      "Equivalencia práctica", "Veredicto"], filas, compacta=True, muestras=[D.ROL_VISTA_EBM[m] for m in t["modelo"]],
                     num=[False, True, True, True, True, True, False], ocultar_movil=(3, 4))
    etiqueta = f"{titulo}. " + "; ".join(f"{r[0]}: {r[1]}, {r[6]}" for r in filas) + "."
    return fig, titulo, sub, tabla, etiqueta


def _fichas(tarea):
    """Lo que hay detrás del resultado: la entrega anterior, las variables que mandan, el costo y los experimentos."""
    c = TAREA[tarea]
    e2 = D.tabla("resultados/entrega2.csv")
    antes, despues = D.fila(e2, tarea=tarea, modelo=c["e2"]), D.fila(e2, tarea=tarea, modelo="ebm")
    techo = D.techo()
    aparte = "La EBM de esta comparación se afinó aparte; por eso no coincide en la última cifra con el modelo final."
    if tarea == "clasificacion":
        cubre = (despues["metrica"] - 0.5) / (techo["auc_mediana"] - 0.5)
        titulo_entrega = "AUC con la etiqueta de la entrega anterior (la mediana)"
        nota_techo = (f"Otra etiqueta: no se compara con el AUC de arriba ({F.p_valor(despues['p_holm'])} ajustado). Cubre el "
                      f"{F.pct(cubre, 0)} del techo de información ({ui.num(techo['auc_mediana'], 4)}). {aparte}")
    else:
        cubre = despues["secundaria"] / techo["r2"]
        titulo_entrega = "RMSE de prueba frente a la entrega anterior"
        nota_techo = (f"{F.p_valor(despues['p_holm'])} ajustado. Su R² ({ui.num(despues['secundaria'], 4)}) cubre el {F.pct(cubre, 0)} "
                      f"del techo de información ({ui.num(techo['r2'], 4)}). {aparte}")
    cambio = despues["metrica"] - antes["metrica"]
    entrega = ui.antes_despues(titulo_entrega, ui.num(antes["metrica"], c["d"]),
                               ui.num(despues["metrica"], c["d"]), cambio=ui.chip_cambio(cambio, ui.con_signo(cambio, c["d"])),
                               nota=nota_techo, hoy_rotulo=D.NOMBRE_MODELO[c["e2"]], plan_rotulo="EBM")

    imp = M.importancia(tarea)
    cuatro = imp.iloc[:4]
    variables = ui.cifra("Cuatro variables concentran la importancia", F.pct(cuatro.sum(), 0),
                         nota=F.enumerar(M.nombre(v).lower() for v in cuatro.index).capitalize() + ".")

    costo = D.tabla("resultados/costo.csv")
    ce, cx = D.fila(costo, tarea=tarea, modelo="ebm"), D.fila(costo, tarea=tarea, modelo=RIVAL)
    ajuste = ui.cifra("Ajuste de la EBM", ui.num(ce["ajuste_s"], 1), "s",
                      variacion=f"{ui.num(ce['ajuste_s'] / cx['ajuste_s'], 0)} veces lo que tarda {D.NOMBRE_MODELO[RIVAL]}",
                      nota=f"Predice a los {F.entero(M.meta()['n_prueba'])} clientes de prueba en {ui.num(ce['inferencia_s'], 3)} s "
                           f"y ocupa {ui.num(ce['tamano_mb'], 1)} MB.")

    corridas = D.tabla("entrega3/corridas.csv")
    experimentos = ui.cifra("Experimentos", F.entero(len(corridas)), "corridas",
                            nota=f"Validación cruzada anidada {int(corridas['k_ext'].iloc[0])} × {int(corridas['k_int'].iloc[0])} y "
                                 f"{ui.num(corridas['tiempo_total_s'].sum() / 3600, 1)} horas de cómputo.")
    return [entrega, variables, ajuste, experimentos]


def _subtitulo():
    t = D.prueba("clasificacion")
    e, x = D.fila(t, modelo="ebm"), D.fila(t, modelo=RIVAL)
    return (f"Un modelo que se puede leer, la EBM, ordena a los clientes por riesgo igual de bien que los mejores modelos de "
            f"árboles (AUC de prueba {ui.num(e['auc_prueba'], 4)} frente a {ui.num(x['auc_prueba'], 4)} de {D.NOMBRE_MODELO[RIVAL]}) "
            f"y supera a los lineales, a k-NN y a Naive Bayes. Alcanza la precisión de los modelos de árboles; no la supera.")


def _glosario():
    """Los términos que la pantalla usa sin explicar, en una línea cada uno (las cifras, de los datos)."""
    meta, corridas, resumen = M.meta(), D.tabla("entrega3/corridas.csv"), D.resumen()
    return ui.desplegable("Qué significan estos términos", [ui.ficha([
        ("AUC", f"Qué tan bien ordena el modelo a los clientes por riesgo: {ui.num(M.sigmoide(0), 1)} es ordenar al azar y 1, sin error."),
        ("RMSE", "Error típico del pronóstico del puntaje, en las unidades del puntaje; menor es mejor."),
        ("Prueba", f"Los {F.entero(meta['n_prueba'])} clientes apartados antes de entrenar; cada modelo final los toca una sola vez."),
        ("Validación anidada", f"Dentro del entrenamiento, {int(corridas['k_ext'].iloc[0])} pliegues externos miden y "
                               f"{int(corridas['k_int'].iloc[0])} internos eligen los hiperparámetros: así se elige la mejor "
                               f"combinación de cada modelo sin mirar la prueba."),
        ("Intervalo del 95 %", f"Rango en que cabe esperar la cifra, con {F.entero(resumen['n_boot'])} remuestreos de los clientes de "
                               f"prueba (bootstrap BCa)."),
        ("Corrección de Holm", "Ajuste de los valores p cuando se hacen siete comparaciones a la vez, para no declarar diferencias por azar."),
        ("Equivalencia práctica", f"Una diferencia menor que ±{ui.num(resumen['rope']['auc'], 2)} de AUC o ±{ui.num(resumen['rope']['rmse'], 4)} "
                                  f"de RMSE no cambiaría una decisión; la probabilidad de equivalencia es la de que la diferencia caiga ahí."),
        ("Riesgo alto", "El cuarto superior del puntaje de propensión que trae el conjunto de datos; no es fuga observada."),
    ])], pequeno=False)


def _destinos():
    C, meta = M.clientes(), M.meta()
    fila = meta["ejemplos"]["riesgo bajo con mayor puntaje"]
    cl = C.loc[fila]
    campana = M.resumen_capacidad(C["riesgo_alto"], C["p_ebm"], M.CAPACIDAD_INFORME)
    vivos = [
        ui.destino(f"/modelos/cliente?fila={fila}", "cliente", "Un cliente y su explicación",
                   f"El modelo le da un puntaje de {ui.num(cl['p_ebm'], 4)} a un cliente de {int(cl['age'])} años que no es de riesgo "
                   f"alto: es su falsa alarma más clara. Vea qué variables lo explican y cambie una."),
        ui.destino("/modelos/campana", "campana", "Una campaña con capacidad limitada",
                   f"Con capacidad para el {F.pct(M.CAPACIDAD_INFORME, 0)} de los clientes, la EBM alcanza a {F.entero(campana['vp'])} "
                   f"de los {F.entero(campana['positivos'])} de riesgo alto. Mueva la capacidad."),
        ui.destino("/modelos/decide", "modelo", "Las curvas del modelo",
                   f"Las {F.entero(M.ebm().n_parametros)} cifras del modelo, variable por variable."),
    ]
    resto = [ui.destino(*d) for d in app.DESTINOS if d[1] in ("datos", "comparacion", "robustez", "errores")]
    return vivos, resto


def layout(**_):
    vivos, resto = _destinos()
    tarea = ui.control("Tarea:", ui.segmentado(f"{PRE}-tarea", [{"label": "Clasificación", "value": "clasificacion"},
                                                               {"label": "Regresión", "value": "regresion"}], "clasificacion"))
    return ui.pantalla([
        ui.encabezado("¿Qué encontramos?", _subtitulo(), [tarea], pantalla="inicio"),
        ui.seccion("El resultado, con lo que lo acota", [
            html.Div(id=f"{PRE}-heroe"),
        ], subtitulo="La cifra principal nunca va sola: debajo, la diferencia con el mejor modelo de árboles y con el mejor lineal.",
            fuentes=FUENTES_PRUEBAS),
        ui.seccion("La EBM frente a cada referencia", [
            ui.rejilla([
                ui.grafico(f"{PRE}-dif", fuentes=FUENTES_PRUEBAS,
                           leyenda=[("Modelos de árboles", "rival", "barra"), ("Lineales, k-NN y Naive Bayes", "base", "rayada")]),
                ui.tarjeta_insight("La alcanza, no la supera",
                                   "Un modelo aditivo que se puede leer llega a la precisión de los modelos de árboles en estos datos.",
                                   "Es más lento de ajustar, igual de rápido al predecir y cada variable aporta una curva que se puede "
                                   "graficar y auditar.", chip_="Conclusión del informe"),
            ], "2-1"),
            _glosario(),
        ]),
        ui.seccion("Lo que hay detrás del resultado", [ui.cifras([], id=f"{PRE}-fichas")],
                   fuentes=[D.fuente("resultados/entrega2.csv"), D.fuente("entrega3/comparacion_entrega2/techo.json"),
                            D.fuente("modelos/ebm_clasificacion.json"), D.fuente("resultados/costo.csv"), D.fuente("entrega3/corridas.csv")]),
        ui.seccion("El modelo en vivo y el resto de la historia", [ui.destinos(vivos + resto)],
                   subtitulo="La app evalúa las tablas del modelo guardado por el proyecto: puntúa y explica cada cliente en el momento."),
        ui.pie([ui.nota("Datos: COFINFAD, Colombian Fintech Financial Analytics Dataset (Mendeley Data, licencia CC BY 4.0). «Riesgo "
                        "alto» es el cuarto superior del puntaje de propensión del conjunto de datos, no fuga observada."),
                ui.nota("Proyecto final de Machine Learning · Pregrado en Ciencia de Datos, Universidad del Norte · Valerie Martínez "
                        "Pulido, Abrahan Basto Martínez y Rubén Esguerra Fernández.")]),
    ])
