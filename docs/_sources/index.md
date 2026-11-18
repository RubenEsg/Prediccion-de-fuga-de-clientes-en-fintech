# Predicción de fuga de clientes en una fintech colombiana

**Universidad del Norte · Pregrado en Ciencia de Datos · Machine Learning · Proyecto final**

Martínez Pulido, Valerie · Basto Martínez, Abrahan · Esguerra Fernández, Rubén

Este Jupyter Book es el informe técnico del proyecto final sobre el conjunto público COFINFAD. La primera parte,
**Proyecto final**, sigue los capítulos de la guía (del 0 al 10): el dashboard desplegado, el modelo original y su
evaluación. Las partes siguientes son las tres entregas anteriores, en el orden en que se hicieron, y el código con que
se produjo cada resultado.

- **Dashboard desplegado:** <https://fuga-de-clientes-dash.onrender.com>
- **Repositorio:** <https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech>

| Parte | Contenido |
|---|---|
| **Proyecto final** | Capítulos 0 a 10: introducción, trazabilidad de las correcciones, datos y auditoría de fuga, benchmarks, revisión de literatura y novedad, el modelo original (BDL), protocolo y validación estadística, evaluación ampliada y robustez, dashboard y despliegue, conclusiones y reproducibilidad. Al final, el código completo del modelo original, sus pruebas y el protocolo pre-registrado |
| **Entrega 1** | Análisis exploratorio de datos: el cuaderno ejecutado de la primera entrega (artículo en `Entrega1/`) |
| **Entrega 2** | Preparación de datos y modelos base: el cuaderno ejecutado de la segunda entrega (artículo en `Entrega2/`) |
| **Entrega 3** | El tercer entregable en cinco partes: Data, Metodología, Modelos y Resultados, El modelo nuevo, y Comparación estadística y conclusiones. El artículo en formato Springer, el notebook en PDF y un cuaderno único con todo están en `Entrega3/` |
| **Código fuente** | Todo el código utilizado en la Entrega 3, archivo por archivo |

## Resultados principales del proyecto final

```{include} ProyectoFinal/_generado/resultados_principales.md
```

## Resultados de las entregas anteriores

- **Pipeline de la guía**: 140 corridas (136 entrenadas y 4 que no aplican), con validación cruzada anidada 5 × 3. Mejor modelo de clasificación: XGBoost (AUC externa 0,8332); de regresión: XGBoost (RMSE externo 0,0590).
- **Modelo nuevo de la Entrega 3 (EBM) frente a la Entrega 2**, en la misma prueba y con la misma etiqueta: AUC 0,6832 → 0,7187 (DeLong p < 10⁻¹⁸) y R² 0,1511 → 0,2159 (Diebold-Mariano con HLN p < 10⁻³⁹). Empata con XGBoost (p = 0,196 en AUC) y queda cerca del techo de información (0,7223 de AUC y 0,2197 de R²). En el proyecto final la EBM es un benchmark reciente: proviene de una librería y no cuenta como modelo original.

Toda cifra del libro se genera a partir de los resultados. Los datos van comprimidos en `datos/` y el entorno se fija con
`Entrega3/requirements.txt` (Python 3.13.3); la semilla del proyecto es 42.
