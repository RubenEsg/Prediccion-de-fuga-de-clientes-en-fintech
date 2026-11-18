# Sistema de diseño de la app de fuga de clientes

> Fuente única: el bloque `:root` de `assets/estilos.css`. `tokens.py` lo lee y da a Python los roles (`ROL`, `VAR`);
> `ui.py` arma con ellos la plantilla de Plotly y los componentes; las páginas (`paginas/*.py`) solo componen esas piezas y
> esos roles. `verificar_estilo.py` comprueba todo lo de abajo, en el código y en el navegador.
>
> Procedencia: es el sistema de diseño del laboratorio Dash del Data Challenge (proyecto Sobusa), portado a esta app. Se
> conservaron los colores, la escala y los componentes; cambiaron los nombres de los roles (§2), se quitó todo lo propio de
> aquel proyecto y se añadieron las piezas que pide una app de métricas entre 0 y 1 (§4 y §5).

## 1. Dirección: la referencia que eligió el usuario

La referencia es el tablero **«Dashboard for payment and business analytics»** de Outcrowd en Dribbble:
<https://dribbble.com/shots/26446179-Dashboard-for-payment-and-business-analytics>. De ella se toma:

- página gris claro y **tarjetas casi blancas** muy redondeadas (radio 26 px), con borde fino y sombra difusa;
- **navegación arriba**: logo + «Fuga de clientes» a la izquierda, las nueve pantallas como pestañas de texto (Inicio,
  Datos, Modelo, Cliente, Campaña, Comparación, Robustez, Errores y Límites; la activa en **píldora negra**) y el chip
  «Modelo en vivo» a la derecha;
- **título enorme en peso normal** (la pregunta de la pantalla, hasta 60 px, interletrado cerrado) y, en la misma fila a la
  derecha, los controles de la pantalla, **cada uno en su píldora gris** con el segmento elegido en blanco; si no caben
  junto al título bajan a la derecha; en el teléfono va un control por línea;
- cada tarjeta con su título a 24 px y el **botón redondo «…»** arriba a la derecha (abre la tabla gemela y las fuentes);
- **cifras muy grandes en peso normal** (44 px; la protagonista hasta 72 px), lo secundario en gris;
- **chips en píldora blanca** con sombra suave («▲ +0,08», «Umbral: 0,25»);
- **barras rayadas** en diagonal (el color sobre su tinte) con la columna en foco **sólida** y una banda azul clara detrás;
  escalón con relleno de rayas verticales; barras de progreso rayadas; matriz de puntos; tarjeta de lectura con degradado
  naranja → azul con grano y texto blanco.

**La barra con nueve pestañas** (medida en Edge sin interfaz). A su tamaño, marca + pestañas + chip ocupan 1.232 px;
compacta, 1.071 px. Contando 32 px de márgenes y hasta 17 px de la barra de desplazamiento de la página: desde 1300 px va
todo a su tamaño; de 1140 a 1299 px va compacta (menos relleno, el nombre de la marca a 20 px) sin ocultar nada; de 1101 a
1139 px, además, el chip queda solo con su icono (el texto sigue en `title` y `aria-label`); el nombre de la marca no se
oculta nunca. Por debajo de 1101 px la navegación baja a una segunda fila y las pestañas se desplazan en horizontal, con
la activa centrada a la vista.

Tipografía: **Inter** (SIL Open Font License 1.1; «Copyright (c) 2016 The Inter Project Authors»), variable, en
`assets/fuentes/` (`InterVariable.woff2` y la licencia en `OFL.txt`), declarada con `@font-face` y `font-display: swap`:
funciona sin internet. Respaldo: Segoe UI Variable Display, Segoe UI, system-ui.

## 2. Colores por rol

Cada entidad tiene un rol y su color es el mismo en las nueve pantallas. Nunca se escribe un color en `paginas/*.py`: el
color entra como `ui.traza(rol, forma)`, `ui.ROL[rol]`, `ui.VAR[rol]`, `ui.fuente_svg(rol, paso)` o una clase del CSS.

| Rol (`ui.ROL[…]`) | Qué es en esta app | Token | Hex | Cómo se dibuja |
|---|---|---|---|---|
| `modelo` | el modelo en foco: el modelo original, BDL, en las vistas de los diez modelos y en la suya (`D.ROL_MODELO`); la EBM en las vistas de su grupo y en la mitad del Entregable 3 de la comparación (`D.ROL_VISTA_EBM`). La leyenda dice cuál es | `--serie-modelo` | `#2457f0` | sólida cuando es la serie en foco; rayada azul para lo demás de su familia |
| `rival` | los benchmarks de la parte alta de la tabla: la EBM, LightGBM, XGBoost, Random Forest y el árbol de decisión (en las vistas de la EBM, los modelos de árboles con los que empata y, si aparece, BDL) | `--serie-rival` | `#b8235f` | barras rayadas en rosa; línea rosa |
| `base` | modelos lineales, k-NN, Naive Bayes y la Entrega 2 | `--serie-base` | `#858585` | barras rayadas en gris; línea gris **a trazos** |
| `real` | lo observado | `--serie-real` | `#141414` | línea de 2,5 px o puntos |
| `contexto` | el contexto (el azar, lo que rodea) | `--serie-contexto` | `#c8c8c8` | línea fina o barras; 1,6:1 sobre la tarjeta: siempre con rótulo o tabla |
| `modelo_tinte` · `rival_tinte` · `base_tinte` | fondo del rayado de cada familia | `--serie-…-tinte` | `#dce5fd` · `#f6dbe6` · `#e3e3e3` | nunca una serie por sí solos; el azul claro es también el fondo de los aciertos de la matriz de confusión |
| `bien` · `atencion` · `critico` | estados | `--bien` · `--atencion` (`--atencion-texto`) · `--critico` | `#18753a` · `#b26f0b` (`#925b08`) · `#ba2b2e` | solo estados, siempre icono + palabra; nunca una serie |

El orden fijo de la leyenda es `modelo`, `rival`, `base`, `real`, `contexto` (`tokens.SERIES`). `--franja` es el azul al
9 % (banda de foco o de rango). `--acento` es el mismo azul del modelo en foco: enlaces, foco del teclado, tramo del deslizador.
Varios rivales o varios modelos de la base en un mismo gráfico comparten color: se distinguen por la forma, por un rótulo
directo o en una tabla; no hay una paleta de cuatro tonos para categorías sin rol (balanceo, optimizador).

Neutros: página `--fondo #efefef`, tarjeta `--superficie #fafafa`, grupo de píldoras `--superficie-2 #e9e9e9`,
píldoras y anidados `--blanco #ffffff`, borde `--linea #e3e3e3`, texto `--tinta #141414`, `--tinta-2 #555555`,
`--tinta-3 #666666` (los grises claros de la referencia, oscurecidos lo justo para 4,5:1), `--tinta-tenue #8a8a8a`
(solo texto de 24 px o más: 3,3:1), rejilla `#ebebeb`.

**Validación** (`validate_palette.js` del método dataviz, superficie `#fafafa`, todos los pares; la corre
`verificar_estilo.py` si están `node` y la variable `VALIDADOR_PALETA`):

| Conjunto que se ve junto | Daltonismo (peor par, ΔE) | Visión normal (ΔE) | Contraste con la tarjeta |
|---|---|---|---|
| modelo en foco + rivales | pasa, 23,7 | pasa, 32,0 | pasa |
| modelo en foco + base | pasa, 22,0 | pasa, 25,1 | pasa |
| base + rivales | pasa, **8,9** (deuteranopía) | pasa, 21,0 | pasa |
| modelo en foco + rivales + base | pasa, 8,9 | pasa, 21,0 | pasa |
| observado + modelo en foco + contexto | pasa, 33,6 | pasa, 38,5 | aviso: el gris claro, 1,6:1 (pide rótulo o tabla) |
| modelo en foco + base + observado + rivales | pasa, 8,9 | pasa, 21,0 | pasa |

Los grises «fallan» el piso de croma por diseño (son grises). El par más justo es gris–rosa: de ahí la regla **gris y rosa
nunca con la misma forma**. Contraste de texto: 32 pares con su mínimo (4,5:1; 3:1 para texto de 24 px o más y marcas); el
más justo, blanco sobre `--insight-3` (3,02:1, solo texto grande). Los aciertos de la matriz (tinta sobre el azul claro)
dan 14,6:1 y 5,9:1.

## 3. Escala

Letra: `--t-xs 12` · `--t-s 14` · `--t-m 16` (texto) · `--t-l 20` · `--t-xl 24` (título de tarjeta) · `--t-2xl 32`
(título de sección, matriz, cifras que acotan) · `--t-3xl 40–60` (título de pantalla) · `--t-cifra 44` · `--t-hero 52–72`.
En el teléfono (≤ 600 px): 36, 26, 22, 36 y 46 para los cinco mayores. Pesos: 400 en títulos y cifras (`--peso-titulo`,
`--peso-cifra`), 500 en pestañas, chips y botones, 600 solo en rótulos pequeños y en el mejor valor de una tabla; nunca 700.
**Hasta 8 tamaños por pantalla** (el de 12 px se cuela con facilidad: las ayudas de los campos van a 14).
Radios: 3 (puntos), 10, 16 (anidados), 26 (tarjetas), 28, píldora. Sombras: `--sombra-2` tarjetas, `--sombra-4` chips y
segmento elegido, `--sombra-3` panel «…» y lista del desplegable. Espaciado en pasos de 8 con medio paso (`--e-1` … `--e-9`).

## 4. Componentes (`ui.py`)

Estructura de la pantalla
- `pantalla(hijos)` → `encabezado(titulo, subtitulo, controles, respuesta, pantalla=id)` (o `arriba(cabecera(…), controles,
  respuesta)`): la pregunta enorme (h1) a la izquierda, el subtítulo que la responde con la cifra y los controles a la
  derecha, cada uno en su píldora gris.
- `seccion(titulo, hijos, subtitulo, fuentes, controles, id, intro)`: h2 con el mensaje. `rejilla(hijos, "2-1" | "1-2" |
  "1-1" | "3")`: el tablero de tarjetas (una columna por debajo de 900 px).
- `tarjeta(cuerpo, titulo=…, subtitulo=…, detalles=…, fuentes=…)` / `bloque(titulo, cuerpo, detalles=…)`: tarjeta (h3) con
  «…» si hay detalles o fuentes. `parrafo`, `nota`, `pie`, `fuentes`, `desplegable`, `destino` / `destinos`.
- Encabezados sin saltos: h1 (pantalla) → h2 (sección, o uno con la clase `oculto`) → h3 (tarjeta o gráfico).

Cifras
- `cifra(etiqueta, valor, unidad, variacion, tono, nota, flecha=…, chip=…, tenue=…, mini=…)` y `cifras(…)`. `tenue=True`
  apaga el valor: la cifra de comparación.
- `heroe(etiqueta, valor, unidad, texto, saldos, lado, pie)`: la cifra protagonista, como mucho una por pantalla y siempre
  con las **dos cifras que la acotan**: `saldo(titulo, valor, definicion, unidad="", d=0, palabras=("a favor", "en contra"),
  extremo=None)`. También vale una `tarjeta(…, clase="heroe cifra--hero")` con las mismas partes, si hace falta título y «…».
- `antes_despues(etiqueta, hoy, plan, cambio, nota, plan_rotulo="Después", hoy_rotulo="Antes")` y `cambios(…)`: dos niveles
  comparados («Entrega 2» → «EBM» pasando los dos rótulos).
- `chip(texto, tipo)`, `chip_cambio(valor, texto, tono)` (▲/▼), `chip_valor(rotulo, valor)`.
- `barra_rayas(etiqueta, valor, fraccion, rol)` (una probabilidad, la importancia de una variable), `puntos(cuentas, foco,
  rol)`, `tarjeta_insight(valor, titulo, texto)` (una por pantalla como mucho), `aviso(texto, tipo, titulo)`.
- `matriz(vp, fp, fn, vn, rotulos=None, id=None)`: matriz de confusión de dos por dos. Filas = lo que pasó («Riesgo alto
  real», «Riesgo bajo real»); columnas = lo que se hizo («Contactado», «No contactado»); cada celda con su nombre
  («acierto», «falsa alarma», «se escapa», «bien descartado») y los dos aciertos sobre el azul claro. Los números llegan ya
  formateados. Lleva `role="img"` con una etiqueta que dice las cuatro cifras.
- `ficha(pares, id=None)`: lista de pares rótulo–valor (los datos de un cliente).

Gráficos y tablas
- `grafico(id_o_fig, titulo, subtitulo, etiqueta, tabla, leyenda, nota, alto, id, fuentes=…)`: título-mensaje, leyenda
  HTML, lienzo con `role="img"` y, en el «…», la tabla gemela y las fuentes. Con un id, los callbacks escriben en
  `<id>-titulo`, `-sub`, `-leyenda`, `-tabla`, `-nota` (su contenido) y `-lienzo` (su `aria-label`); el contenedor es
  `<id>-figura`.
- `leyenda([(texto, rol, forma)])`: siempre en el orden fijo de los roles.
- `tabla(encabezados, filas, total, mejor, compacta, muestras, ocultar_movil, num, alta, id, mejor_texto="mejor")`: el mejor
  valor va **en negrita y en tinta** con la palabra «mejor» debajo (14 px, gris), nunca en el color de una serie: el
  mejor puede ser un rival.

Controles
- `segmentado(id, opciones, valor)` (2 a 6 opciones cortas), `casilla`, `control(rotulo, componente, ayuda)` (rótulo en
  línea), `boton` (el primario es la píldora negra), `boton_redondo`.
- `deslizador(id, rotulo, minimo, maximo, paso, valor, marcas, ayuda, persistir, valor_inicial_texto=None,
  updatemode="mouseup")`: el valor se muestra en `<id>-valor`; nace con `valor_inicial_texto` y lo mantiene un callback
  (`drag_value` + `value`). Con `updatemode="drag"` recalcula mientras se arrastra (el simulador).
- `campo_numero(id, valor, minimo, maximo, paso, rotulo=None, ayuda=None, persistir=True)`: número en píldora, con el
  rótulo encima; avisa al salir del campo o con Intro.
- `selector(id, opciones, valor, rotulo=None, buscar=False, persistir=True, placeholder=None)`: desplegable de una sola
  elección para listas largas (un cliente, una variable), con sus textos en español; con rótulo va encima, sin él se pone
  en línea con `control`.
- El esqueleto de un simulador son clases del CSS: `laboratorio` (panel de 320 px + resultados; por debajo de 1200 px una
  columna, con el panel debajo), `panel-parametros`, `panel__cab`, `panel__titulo`, `panel__grupo`, `resultados`,
  `estado-calculo` y `chips`.

Reglas de disposición en `estilos.css`: `.tarjeta` y `.grafico` llevan `align-content: start` (el alto que sobra en una
fila queda abajo, no entre sus partes); las dos cifras que acotan a la protagonista comparten filas con `subgrid`; por
debajo de 420 px las celdas de las tablas tienen menos relleno lateral.

Prefijos reservados para los `id` de cada página: Inicio `in-`, Datos `da-`, Modelo `mo-`, Cliente `cl-`, Campaña `ca-`,
Comparación `co-`, Robustez `ro-`, Errores `er-`, Límites `li-`. Los de la barra son `nav-0` … `nav-8`; los del esqueleto,
`url` y `contenido`.

Piezas del recorrido (en `app.py` y `assets/estilos.css`, no en las páginas):

- **Pie «Siguiente»**: `app.py` añade al pie de cada pantalla el enlace a la que sigue en `NAV` (en la última, la vuelta al
  inicio). Las páginas no ponen el suyo.
- **Clase `pegada`**: una tarjeta de controles con esa clase se queda a la vista bajo la barra mientras se recorre lo que
  mueve (solo desde 1101 px de ancho y 640 px de alto). La usa el deslizador de Campaña.
- **Enlaces internos**: `.seccion`, `.tarjeta` y `.grafico` llevan `scroll-margin-top`, así que un enlace a `#id` deja la
  pieza debajo de la barra y no tapada por ella.
- **Página 404**: `paginas/not_found_404.py` (Dash la busca por ese nombre) lista las pantallas en castellano.
- **Nombres en frases**: `datos_app.en_frase(modelo)` da el nombre con su artículo («la regresión logística», «el árbol de
  decisión»); en rótulos, tablas y leyendas va `datos_app.NOMBRE_MODELO`.

## 5. Gráficos

Plantilla `fuga` (se aplica sola con `import ui`): Inter, rótulos de los ejes en `--tinta-3`, rejilla muy clara, sin
títulos de eje (la unidad con `unidad_eje`), etiqueta flotante en píldora blanca con borde fino, separadores `,.` y el
menos «−» en el texto ya formateado. Formas por rol con `traza(rol, forma)`: `foco` / `barra` (sólida), `rayada` (rayas
diagonales sobre el tinte), `linea`, `gruesa` (2,5 px), `trazos`, `punteada`, `escalon`, `escalon_rayado`, `area`,
`punto`. Sin forma, cada rol trae la suya: `modelo`, `rival` y `contexto` en línea, `base` a trazos, `real` gruesa.
Ayudas: `barras_foco(rol, categorias, foco)` (una columna sólida, las demás rayadas), `banda_foco`, `etiquetas_columnas`,
`hover` y `hover_texto` (el texto ya formateado en `customdata`), `linea_cero`, `referencia` (sólida de 1 px), `banda`
(un intervalo), `franja`, `ancho_barra`, `figura_vacia`, `fuente_svg`.

Métricas y probabilidades entre 0 y 1 (la plantilla trae enteros en el eje y):
- `formato_eje(fig, eje="y", d=2, sufijo="")` fija los decimales con coma y un sufijo opcional (« %»);
- `marcas(lo, hi, paso)` da las marcas en es-CO y deduce los decimales del paso (0,25 → dos; 0,5 → uno; 5 → ninguno);
- `paso_redondo(tope)` elige un paso redondo también por debajo de 1 (0,87 → 0,5; 0,06 → 0,02);
- `ui.num(x)` redondea a entero si no se le dan decimales: en esta app se pasan siempre (`ui.num(0.8289, 4)`);
- `ui.pct(x)` recibe una **fracción** (`ui.pct(0.2573)` → «25,7 %»); los números salen de `formato.py`.

Reglas: un solo eje y; leyenda HTML cuando hay dos series o más (un gráfico de un solo rol no la lleva: lo nombra el
título); **gris y rosa nunca con la misma forma**; el rayado y las bandas salen solo de `ui.py`; referencias sólidas de
1 px, nunca a trazos; la cascada es `go.Bar` con `base`; sin curvas suavizadas; tabla gemela en el «…» de cada gráfico;
el título de un gráfico dice la conclusión con su cifra.

## 6. Reglas que no cambian

- Las cifras salen de `datos/`: ninguna se escribe a mano en una página, y cada tarjeta dice de qué archivo sale en su «…».
- Formato es-CO en todo, también en Plotly: miles con «.», decimales con «,», menos «−».
- El color solo codifica significado; verde, ámbar y rojo no son series. Los estados van siempre con icono y palabra, y el
  tono de una diferencia (verde o rojo) solo cuando su sentido es inequívoco.
- La cifra grande va con lo que la acota: una protagonista por pantalla como mucho, y con sus dos cifras al lado. Si son
  los bordes de un rango, se presentan como tales (chip neutro), no como un resultado en rojo.
- Cada pantalla abre con su pregunta (h1) y un subtítulo que la responde con la cifra. Un control nunca va a la derecha del
  título de una tarjeta.
- Lo que falla se dice en un aviso; no rompe la pantalla ni inventa datos. Un canje o un límite va en un aviso a la vista,
  nunca escondido en un desplegable.
- Solo modo claro. Fuente incluida: la app funciona sin internet.
- Del disco solo se sirven tres PDF del proyecto, por lista blanca (`/documento/informe`, `/documento/metodologia`,
  `/documento/resultados`), y solo si están junto a la carpeta de la app.

## 7. Verificación

    python verificar_estilo.py                 en el código: contraste, roles, CSS sin literales, Python sin colores ni
                                               tamaños, construcciones prohibidas, cifras a mano en las páginas, paleta
    python verificar_estilo.py --vivo 8061     además, en Edge sin interfaz (Playwright), con la app corriendo en ese puerto

En vivo revisa las nueve pantallas a 1366 y a 400 px: consola sin errores, hasta 8 tamaños de letra y todos de la escala,
pesos 400, 500 y 600, cada color calculado dentro de la paleta, encabezados sin saltos, como mucho una cifra protagonista
(`.cifra--hero`) y, si la hay, con exactamente dos `.saldo` dentro, y nada que se salga a 400 px. Termina con código 1 si
algo falla. La app tiene que estar ya corriendo (`PORT=8061 python app.py`); el verificador no usa los puertos 8000, 8020
ni 8050. La parte en vivo necesita un Python con Playwright y Edge instalado; la paleta, `node` y la variable
`VALIDADOR_PALETA` con la ruta de `validate_palette.js` (si faltan, lo dice en una nota y sigue).

El CSS está escrito contra el DOM de **Dash 4.5.0** (segmentado, casilla, deslizador, campo y desplegable): con otra
versión mayor esos controles pueden perder el estilo sin dar error. Por eso `requirements.txt` fija `dash==4.5.0` y
`plotly==7.1.0`.

## Tres secciones, subnavegación y tablas (Tercer Entregable)

La barra fija de arriba tiene tres secciones (Contexto, EDA, Modelos), como pide la guía. Modelos abre una subnavegación
(`.subnav`): tres grupos de píldoras con rótulo («Los diez modelos», «El modelo original: BDL» y «La EBM, benchmark
interpretable»); la vista actual en píldora negra y, para lectores de pantalla, «(vista actual)». Por debajo de 640 px cada
grupo se desplaza en horizontal. Las vistas del grupo de la EBM llevan, bajo su cabecera, un aviso que dice que es un
benchmark y no el modelo original (`paginas/modelos.py::aviso_benchmark`).

La vista «Modelo original» escribe sus fórmulas con `dcc.Markdown(mathjax=True)`: en línea dentro de los párrafos y, las
largas, en un bloque con desplazamiento propio (`.tabla-marco`), para que no ensanchen la página en el teléfono. Los
nombres largos de pruebas y de funciones van en `html.Code`, que parte la palabra donde haga falta. Sus contenedores que
se rellenan por callback usan la clase `rejilla` (una columna que no se desborda), no `resultados`. Un bloque cuyo archivo
todavía no existe se muestra con `ui.aviso(…, "atencion", "Pendiente")`: qué falta, qué archivo lo traerá y quién lo
produce.

Las tablas interactivas (`dash_table.DataTable`: la tabla de líderes, el diccionario de variables, las de la EDA) van
dentro de `html.Div(className="tabla-datos")` y toman su estilo de `assets/tablas.css` (tokens del sistema; nada escrito
en la tabla). Todo se alinea a la izquierda; una columna cuyo id termina en `_num`, a la derecha con cifras tabulares. Con
filtros, `filter_options={"case": "insensitive"}`: el botón de mayúsculas se oculta. Sin paginación.

Las fórmulas van con `dcc.Markdown(..., mathjax=True)`; cada `$$` en su línea. El verificador en vivo ignora lo que hay
dentro de `mjx-container` (el MathML oculto para lectores de pantalla tiene tamaños propios).
