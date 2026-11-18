| Aspecto | Declaración |
| :--- | :--- |
| Conjunto | COFINFAD: *Colombian Fintech Financial Analytics Dataset* |
| Procedencia | Muñoz Guerrero, Ceballos y Trejos Rojas; publicado en Mendeley Data, versión 1, 2025; DOI [10.17632/mhb4zn3258.1](https://doi.org/10.17632/mhb4zn3258.1) |
| Licencia | CC BY 4.0: se puede usar y redistribuir citando a los autores |
| Composición | 48.723 clientes, uno por fila, con 54 columnas, y 3.159.157 transacciones de 2023 en una tabla aparte |
| Naturaleza | **Sintético**, generado con reglas. Sus huellas: la edad solo toma 6 valores (29, 30, 44, 45, 59 y 60); el puntaje baja en escalones casi idénticos con el número de productos activos (r de Pearson = −0,876); y las 12 variables derivadas de las transacciones casi no se asocian con el puntaje (ρ de Spearman de a lo sumo 0,018 en valor absoluto) |
| Objetivo | `churn_probability`, un puntaje de propensión entre 0,114 y 0,501 en entrenamiento: nunca vale 0 ni 1 y no es una fuga observada |
| Sesgos conocidos | Las cohortes de edad tienen tasas de riesgo alto muy distintas (del 8,7 % al 45,2 %); el archivo está ordenado por cohortes, y por eso la partición es aleatoria y no por posición; el uso del cupo falta exactamente cuando el cliente no tiene tarjeta de crédito (37,4 % de los clientes de entrenamiento): es un «no aplica», no un dato perdido, y se imputa con 0; las columnas de texto (`complaint_topics`, `feature_requests`) tienen entre el 33 % y el 50 % de faltantes y se excluyen |
| Usos adecuados | Enseñar y comparar métodos de aprendizaje supervisado con un protocolo reproducible; estudiar fugas de información, calibración y modelos legibles |
| Usos no recomendados | Sacar conclusiones sobre clientes reales, estimar tasas reales de abandono (el objetivo es un puntaje, no la fuga observada) o tomar decisiones sobre personas |
