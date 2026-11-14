# Informe final del proyecto

Las tres piezas del informe final del proyecto de Machine Learning (Entregables 1 a 3), cada una con su propio
`README.md` que explica los archivos y cómo regenerarlos:

| Carpeta | Contenido |
|---|---|
| [`informe/`](informe/) | **El informe final** ([`informe/main.pdf`](informe/main.pdf), 31 páginas): resumen, introducción, metodología, resultados, conclusiones y una sola lista de referencias. `armar.py` lo arma copiando las dos secciones siguientes y `compilar.py` lo compila con Tectonic. |
| [`metodologia/`](metodologia/) | La sección de metodología escrita con la guía del curso ([`metodologia/main.pdf`](metodologia/main.pdf)): ocho subsecciones, 20 referencias, tres figuras que genera `figuras.py` desde los datos. |
| [`resultados/`](resultados/) | La sección de resultados escrita con la guía del curso ([`resultados/main.pdf`](resultados/main.pdf)): ocho subsecciones; `generar.py` reproduce cada prueba estadística, tabla y figura desde `../../Entrega3/resultados/` y deja toda cifra en `datos/*.csv`. |

Las dos secciones describen lo entregado en el Entregable 3 (la Explainable Boosting Machine como modelo propuesto
frente a los modelos del curso) a partir de sus resultados guardados, sin volver a ejecutar las 160 corridas; el
informe añade la introducción y las conclusiones. Los guiones buscan la carpeta del Entregable 3 en `../../Entrega3`
y los datos comprimidos en `../../datos`, así que se ejecutan desde este repositorio tal cual (los modelos
serializados de `Entrega3/resultados/modelos/`, 490 MB, no se incluyen: sin ellos `generar.py` solo puede
reescribir tablas y figuras con `--tablas` y `--figuras`).
