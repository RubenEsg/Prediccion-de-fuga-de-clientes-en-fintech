# Informe final del proyecto (Entregables 1 a 3)

Informe final del proyecto de Machine Learning (predicción de fuga de clientes con COFINFAD) en un solo
documento: introducción, la sección de metodología escrita con la guía del curso (`../Metodologia Entregable 3/`),
la sección de resultados escrita con su guía (`../Resultados Entregable 3/`) y las conclusiones, con un resumen,
un índice y una sola lista de referencias.

| Archivo | Qué es |
|---|---|
| `main.pdf` | El informe compilado (31 páginas). |
| `main.tex` | Portada, resumen, índice e inclusión de las cuatro secciones; el preámbulo reúne el de las dos secciones. |
| `introduccion.tex`, `conclusiones.tex` | Las dos secciones escritas para el informe; sus cifras provienen de las otras dos. |
| `metodologia.tex`, `resultados.tex`, `tablas/`, `figures/`, `referencias.bib` | Copias que genera `armar.py` desde las carpetas de las secciones. No se editan aquí: el cambio se hace en la carpeta de origen y se vuelve a armar. |
| `armar.py` | Copia las dos secciones con sus tablas y figuras, une las referencias (comprueba que las entradas comunes sean idénticas) y resuelve la única colisión de etiquetas (`tab:optimizadores` de los resultados pasa a `tab:optimizadores_res`). |
| `compilar.py` | Compila con Tectonic, cuenta las palabras de cada sección con la regla de su guía y revisa el registro (cajas desbordadas, referencias o citas sin definir, etiquetas repetidas). |

## Cómo regenerar

Desde esta carpeta, con cualquier Python 3 (solo usa la biblioteca estándar; `pypdf` es opcional para contar las
páginas):

```powershell
$env:PYTHONIOENCODING = "utf-8"
python armar.py       # copia las secciones, sus tablas y figuras y une las referencias
python compilar.py    # main.pdf, conteo de palabras y revisión del registro
```

`compilar.py` busca Tectonic en el PATH, en la variable de entorno `TECTONIC`, en
`../entrega 3/articulo/herramientas/tectonic.exe` (carpeta de trabajo) o en
`../../Entrega3/articulo/herramientas/tectonic.exe` (repositorio); `armar.py` busca las secciones en
`../Metodologia Entregable 3` y `../Resultados Entregable 3` (carpeta de trabajo) o en `../metodologia` y
`../resultados` (repositorio, carpeta `InformeFinal/`).

## Qué se une y qué se añade

- La **metodología** (ocho subsecciones, 20 referencias, hasta 3.500 palabras sin ecuaciones) y los
  **resultados** (ocho subsecciones, hasta 3.000 palabras) entran tal cual; la única frase que cambia es la
  primera de la metodología, que ahora remite a la sección de resultados por su número.
- La **introducción** presenta el problema, las tres entregas, los dos hallazgos que condicionan el trabajo (la
  fuga de información de `active_products` y la no linealidad de las cuatro variables con señal), el modelo
  propuesto, la pregunta, las contribuciones y la organización del informe.
- Las **conclusiones** responden a la pregunta con las cifras de la sección de resultados (equivalencia con los
  modelos de árboles, superioridad sobre los lineales, techo de información), cierran las cinco hipótesis de la
  ablación, extraen las lecciones prácticas (capacidad de campaña, recalibración, costo de ajuste frente a
  inferencia, legibilidad), recogen las limitaciones y proponen el trabajo futuro.
- Las figuras son los doce PDF vectoriales de las dos secciones (texto en inglés, 12 pt, 6,3 in de ancho) y las
  tablas, las quince que genera `generar.py` en la carpeta de resultados.
