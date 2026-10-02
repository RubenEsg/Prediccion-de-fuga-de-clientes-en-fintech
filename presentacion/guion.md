# Sustentación del Entregable 3 — guion de 10 minutos

La guía (§8) pide cubrir: ETL, EDA, modelos implementados y resultados, comparación de los métodos de
optimización, y resultados finales y conclusiones. Sugiere mostrar la evolución de las métricas
durante la optimización, enfatizar las decisiones y su justificación, y usar gráficos comparativos.

**Las diapositivas definitivas están en `Entregable3_sustentacion.pptx`** (y en PDF, por si el equipo de la
sala no tiene PowerPoint). Las cifras son las del libro final del 1 de octubre, y las notas del orador de
cada diapositiva traen el guion de ese minuto; la última lleva además las preguntas probables. Se
regeneran con `generador/generar.js` (ver su cabecera).

| # | Diapositiva | Mensaje central (una frase) | Figura o tabla del libro | Tiempo |
|---|---|---|---|---|
| 1 | Título | Fuga de clientes en una *fintech* colombiana: 140 + 20 modelos, un modelo nuevo y lo que los datos permiten | — | 0:15 |
| 2 | El problema y los datos (ETL) | COFINFAD: 48.723 clientes, 3,16 M de transacciones; 12 variables derivadas; partición 80/20 **antes** de mirar el objetivo | Página 01, secciones 1 y 2 (tabla de auditoría) | 0:45 |
| 3 | Hallazgo 1: la etiqueta | `churn_probability` es un puntaje generado desde `active_products` (escalera de paso constante): se excluye, y queda un techo de información | Página 01, escalera; página 08, tabla del techo | 1:00 |
| 4 | EDA: dónde está la señal | Cuatro variables con efectos no lineales; la edad solo tiene tres cohortes; las transacciones no aportan; corrección del CLV | Página 01, secciones 5 a 7 | 0:45 |
| 5 | Diseño experimental | Pipeline dentro de la validación anidada 5 × 3; 7 × 4 × 4 + 7 × 4 = 140 corridas; mismo presupuesto por optimizador; multi-fidelidad | Página 02, sección 1 y tabla del catálogo | 0:50 |
| 6 | Resultados de las 140 corridas | Los modelos de árboles encabezan y no se distinguen entre sí; el balanceo no mejora el AUC | Página 03, mapa de calor de clasificación | 0:50 |
| 7 | Comparación de optimizadores | La bayesiana es la más eficiente por evaluación; la métrica final no los separa; en reloj, las secuenciales son más lentas | Página 04, curvas *anytime* y diagramas CD | 1:10 |
| 8 | Cómputo | `hist` frente a `exact`, GPU, FAISS frente a KD-Tree en 78 dimensiones, SVM con núcleo inviable, multi-fidelidad validada | Página 05, tabla estándar frente a optimizado | 0:50 |
| 9 | Evaluación | El AUC de prueba coincide con el de la validación; el umbral de 0,5 engaña; la calibración se corrige con isotónica | Página 06, ROC y diagrama de confiabilidad | 0:50 |
| 10 | El modelo nuevo frente a la Entrega 2 | La EBM supera a la Entrega 2 en las dos tareas (DeLong y Diebold-Mariano), empata con XGBoost y es legible | Página 08, tabla de contrastes y funciones de forma | 1:15 |
| 11 | Interpretabilidad y estadística | SHAP y la EBM cuentan la misma historia; Friedman, Nemenyi, DeLong, MCS: los finalistas no se distinguen | Página 07, SHAP; página 09, diagrama CD | 0:45 |
| 12 | Conclusiones | Los datos fijan el techo; la EBM lo alcanza con interpretabilidad; decisiones justificadas con mediciones | Página 10 | 0:45 |

**Total**: 10:00. Reparto sugerido entre los tres integrantes: diapositivas 1-4 (datos), 5-9
(pipeline, optimización, cómputo y evaluación) y 10-12 (modelo nuevo, estadística y conclusiones).

## Preguntas probables y respuesta corta

- **¿Por qué excluir `active_products` si mejora tanto el modelo?** Porque la etiqueta se construyó a
  partir de ella: con ella el modelo reproduce la fórmula (R² ≈ 0,999), no predice la fuga.
- **¿Por qué el percentil 75 y no la mediana?** Con la mediana las clases quedan al 50 % y el balanceo
  no tiene nada que hacer (ADASYN no genera muestras). La comparación con la Entrega 2 sí usa la mediana.
- **¿Por qué la métrica final no distingue a los optimizadores?** Los espacios tienen 1 a 7
  hiperparámetros y el óptimo es una meseta; la diferencia está en cuánto presupuesto se necesita para
  llegar, y ahí gana la bayesiana.
- **¿La EBM es mejor que XGBoost?** No: empatan (diferencias no significativas). Lo que aporta es la
  legibilidad, y supera a la Entrega 2 con significación estadística.
- **¿Por qué no se llega a AUC 0,80 con la mediana?** Por el techo de información: ni un modelo que
  conoce el mecanismo de generación, sin `active_products`, pasa de unas 0,72.
