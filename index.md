# Predicción de fuga de clientes en una fintech colombiana

**Universidad del Norte · Pregrado en Ciencia de Datos · Machine Learning · Proyecto final**

Martínez Pulido, Valerie · Basto Martínez, Abrahan · Esguerra Fernández, Rubén

Este Jupyter Book reúne las tres entregas del proyecto sobre el conjunto público COFINFAD, en el orden en que se hicieron, y
el código con que se produjo cada resultado. Repositorio: <https://github.com/RubenEsg/Prediccion-de-fuga-de-clientes-en-fintech>.

| Parte | Contenido |
|---|---|
| **Entrega 1** | Análisis exploratorio de datos: el cuaderno ejecutado de la primera entrega (artículo en `Entrega1/`) |
| **Entrega 2** | Preparación de datos y modelos base: el cuaderno ejecutado de la segunda entrega (artículo en `Entrega2/`) |
| **Entrega 3** | El entregable final en cinco partes: Data, Metodología, Modelos y Resultados, El modelo nuevo, y Comparación estadística y conclusiones. El artículo en formato Springer, el notebook en PDF y un cuaderno único con todo están en `Entrega3/` |
| **Código fuente** | Todo el código utilizado, archivo por archivo |

- **Pipeline de la guía**: 140 corridas (136 entrenadas y 4 que no aplican), con validación cruzada anidada 5 × 3. Mejor modelo de clasificación: XGBoost (AUC externa 0,8332); de regresión: XGBoost (RMSE externo 0,0590).
- **Modelo nuevo (EBM) frente a la Entrega 2**, en la misma prueba y con la misma etiqueta: AUC 0,6832 → 0,7187 (DeLong p < 10⁻¹⁸) y R² 0,1511 → 0,2159 (Diebold-Mariano con HLN p < 10⁻³⁹). Empata con XGBoost (p = 0,196 en AUC) y queda cerca del techo de información (0,7223 de AUC y 0,2197 de R²).

Toda cifra del notebook se genera a partir de los resultados. Los datos van comprimidos en `datos/` y el entorno se fija con
`Entrega3/requirements.txt` (Python 3.13.3); la semilla del proyecto es 42.
