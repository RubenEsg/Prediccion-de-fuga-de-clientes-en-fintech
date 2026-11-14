# Sección de metodología del proyecto (Entregables 1 a 3)

Sección «Metodología» del proyecto de Machine Learning (predicción de fuga de clientes con COFINFAD), escrita
según la guía del curso *Cómo escribir la sección de metodología* (ocho subsecciones con presupuesto de palabras,
figuras en PDF vectorial con texto en inglés, ecuaciones numeradas, pseudocódigo, complejidad, protocolo de
evaluación y diseño de ablación). Describe lo entregado hasta el Entregable 3: los datos y el preprocesamiento
del Entregable 1, los modelos del curso del Entregable 2 como referencias y la Explainable Boosting Machine
(EBM) del Entregable 3 como modelo propuesto. Esta carpeta está fuera de los repositorios del proyecto y no
modifica nada de `entrega 3/`; solo lee sus datos y resultados.

| Archivo | Qué es |
|---|---|
| `main.pdf` | El documento compilado (portada, la sección con sus ocho subsecciones y las referencias). |
| `main.tex` | Preámbulo (clase `article`, español, `natbib` con estilo `elsarticle-harv` como el de la guía) e inclusión de la sección. |
| `metodologia.tex` | La sección: texto, tablas, ecuaciones y el algoritmo. Es lo que se copia al informe completo. |
| `referencias.bib` | Las 20 referencias citadas (subconjunto de `entrega 3/articulo/sn-bibliography.bib`). |
| `figuras.py` | Genera las tres figuras en `figures/` desde `entrega 3/datos/customer_data.csv` (misma partición 80/20 con semilla 42), con el formato de la guía (12 pt, sans-serif, PDF vectorial, texto en inglés). Al guardar cada PDF comprueba que no supere las 6,3 in de ancho (si las superara, LaTeX lo reduciría y el texto quedaría por debajo de 12 pt) y que ningún texto se salga del lienzo. |
| `figures/` | `fig_target.pdf`, `fig_ebm_architecture.pdf`, `fig_pipeline_nested_cv.pdf`. |
| `compilar.py` | Compila con Tectonic y cuenta las palabras por subsección frente al presupuesto de la guía. |

## Cómo regenerar

Desde esta carpeta, con el Python del entorno `entrega 3/.venv` (tiene pandas, scikit-learn y matplotlib):

```powershell
$env:PYTHONIOENCODING = "utf-8"
& "..\entrega 3\.venv\Scripts\python.exe" figuras.py     # figuras (segundos)
& "..\entrega 3\.venv\Scripts\python.exe" compilar.py    # main.pdf + conteo de palabras
```

En el repositorio publicado esta carpeta es `InformeFinal/metodologia/`: los datos están comprimidos en
`../../datos/` (el guion los descomprime la primera vez; también puede fijarse la variable de entorno `COFINFAD`)
y el intérprete es el del entorno creado en `Entrega3/` (`..\..\Entrega3\.venv\Scripts\python.exe`).

`compilar.py` busca Tectonic en el PATH, en la variable de entorno `TECTONIC`, en
`../entrega 3/articulo/herramientas/tectonic.exe` o en `../../Entrega3/articulo/herramientas/tectonic.exe`; Tectonic descarga los paquetes de LaTeX la primera vez y corre
BibTeX solo. Para una vista previa en PNG de las figuras: `$env:FIGURAS_PNG = "<carpeta>"`.

## De dónde salen las cifras

Todas las cifras vienen del Entregable 3 sin volver a ejecutar ningún experimento: `entrega 3/articulo/articulo.tex`
y sus tablas (`articulo/tablas/`), el libro de Jupyter (`libro/fuentes/`), `resultados/` (tabla maestra, métricas
de prueba, comparación con la Entrega 2, cómputo y complementos), `src/` (configuración, espacios de búsqueda y
pesos) y `requirements.txt` (versiones).
