# Declaración de los datos: COFINFAD

Declaración de los datos (en la línea de Bender y Friedman, 2018, y Gebru et al., 2021) del conjunto que usa el proyecto «Predicción de fuga de clientes en una fintech colombiana: dashboard predictivo, modelo original (Boosting con Desplazamiento Latente) y evaluación rigurosa». La genera `generar_fichas.py` desde `datos/`: ninguna cifra se escribe a mano.

## Resumen

Qué conjunto es, de dónde viene y con qué licencia.

### El conjunto

- **Nombre:** COFINFAD: Colombian Fintech Financial Analytics Dataset
- **Autores:** Muñoz Guerrero, Luis Eduardo; Ceballos, Yony Fernando; Trejos Rojas, Luis David
- **Publicación:** Mendeley Data, versión 1, 2025
- **DOI:** 10.17632/mhb4zn3258.1 (https://doi.org/10.17632/mhb4zn3258.1)
- **Licencia:** CC BY 4.0: se puede usar y redistribuir citando a los autores
- **Periodo que describe:** 2023, según la descripción del conjunto que recoge el informe final
- **Tamaño:** 48.723 clientes con 54 columnas y 3.159.157 transacciones en una tabla aparte

Fuentes: CITATION.cff · datos/eda/etl.json · Informe final (metodologia.tex, resultados.tex, conclusiones.tex)

## Cómo se generaron los datos

Lo que se sabe de su origen sale del análisis exploratorio del entrenamiento.

### Datos sintéticos, generados con reglas

Los datos son sintéticos: describen a los clientes de una fintech colombiana, pero llevan las huellas de una generación por reglas, no de una muestra de clientes reales. Las conclusiones que se saquen de ellos describen esas reglas.

- La edad solo toma 6 valores (29, 30, 44, 45, 59, 60) en 3 cohortes de tamaño casi igual, y el archivo está ordenado por bloques que las siguen.
- El puntaje churn_probability baja en una escalera de pasos casi idénticos con el número de productos activos (r de Pearson −0,876: explica el 76,8 % de su varianza): la firma de una fórmula.
- Las 12 variables transaccionales derivadas no tienen señal (|ρ| ≤ 0,018).
- El uso del cupo falta exactamente cuando el cliente no tiene tarjeta de crédito (37,4 % de los clientes de entrenamiento): es un «no aplica», no un dato perdido.

Fuentes: datos/eda/asociacion.csv · datos/eda/cohortes.csv · datos/eda/faltantes.csv · datos/variables.csv

### Estructura

- **Unidad:** un cliente por fila; las transacciones, en una tabla aparte que se resume por cliente
- **Objetivo:** churn_probability, un puntaje de propensión entre 0,114 y 0,501 en el entrenamiento: nunca vale 0 ni 1, no es un evento observado
- **Etiqueta del proyecto:** riesgo alto = puntaje por encima de su percentil 75 del entrenamiento (0,3927)
- **Predictores:** 48: 32 numéricos, 7 booleanos y 9 categóricos (78 columnas tras el preprocesamiento)
- **Excluidas:** 17 columnas: customer_id, active_products, customer_lifetime_value, clv_segment, average_transaction_value, total_transaction_volume, avg_daily_transactions, monthly_transaction_count, first_transaction_date, last_transaction_date, nps_score, occupation, complaint_topics, feature_requests, last_survey_date, first_tx, last_tx

Fuentes: datos/eda/etl.json · datos/eda/entrenamiento.parquet · datos/metadatos.json

## Sesgos conocidos

Lo que hay que tener en cuenta al usarlos.

### Sesgos y particularidades

- Cohortes de edad con tasas de riesgo alto muy distintas en el entrenamiento: 29–30 años, 19,8 %; 44–45 años, 8,7 %; 59–60 años, 45,2 %.
- El archivo está ordenado por cohortes: una partición por posición daría conjuntos distintos; por eso se parte al azar.
- El puntaje se construyó a partir de los productos activos: usarlos como entrada es una fuga de información.
- Las categóricas casi no se asocian con el puntaje (η² ≤ 0,005), tampoco el género ni la ubicación.
- Columnas de texto con 33 % de faltantes (feature_requests) y 50 % de faltantes (complaint_topics), sin mecanismo ni señal: se excluyen.

Fuentes: datos/eda/cohortes.csv · datos/eda/asociacion.csv · datos/eda/faltantes.csv

### Preprocesamiento y partición

- Auditoría de las transacciones: 102 duplicadas eliminadas; montos no positivos, fechas nulas y transacciones huérfanas: 0.
- 12 variables de comportamiento derivadas de las transacciones (n_tx, monto_total, monto_medio, monto_mediano, recencia_dias, dias_activo, volatilidad_monto, meses_activos, prop_Deposit, prop_Payment, prop_Transfer, prop_Withdrawal), sin usar el objetivo.
- 17 columnas eliminadas (identificador, la variable que generó el puntaje, el segundo objetivo, duplicados, fechas sin señal, una colineal y textos con faltantes).
- El uso del cupo se imputa con 0 cuando no hay tarjeta; las categóricas se codifican con indicadoras y las numéricas se estandarizan dentro de cada pliegue de cada validación cruzada.
- Partición aleatoria por índices, 80/20, con semilla 42, sin estratificar: 38.978 clientes de entrenamiento y 9.745 de prueba. El umbral de la etiqueta y todo el análisis exploratorio usan solo el entrenamiento.
- La recencia se mide hasta una fecha de corte que sale solo de los clientes de entrenamiento (29 de diciembre de 2023); antes salía de todo el detalle de transacciones. La auditoría de la corrección comprueba que cambian 0 celdas y que los predictores de entrenamiento y de prueba son idénticos antes y después.

Fuentes: datos/eda/etl.json · datos/metadatos.json · datos/original/auditoria_fuga.json

## Usos adecuados e inadecuados

Para qué sirven estos datos y para qué no.

### Adecuados

- Enseñar y comparar métodos de aprendizaje supervisado con un protocolo reproducible.
- Estudiar fugas de información, desbalance, calibración y modelos legibles.
- Reproducir los resultados de este proyecto.

Fuentes: Informe final (metodologia.tex, resultados.tex, conclusiones.tex)

### Inadecuados

- Sacar conclusiones sobre clientes reales de una fintech colombiana.
- Estimar tasas reales de abandono: el objetivo es un puntaje, no la fuga observada.
- Tomar decisiones sobre personas o leer las asociaciones como causas.
- Predecir el puntaje con los productos activos: el modelo recuperaría la regla con que se generó.

Fuentes: Informe final (metodologia.tex, resultados.tex, conclusiones.tex)
