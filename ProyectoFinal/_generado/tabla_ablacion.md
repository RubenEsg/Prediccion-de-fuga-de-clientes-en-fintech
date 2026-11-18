**Regresión (RMSE, menor es mejor).**

| Variante | RMSE externo | RMSE en prueba | Aporte del componente (externo) | p (*t* corregida, Holm) | Aporte en prueba | p (prueba, Holm) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BDL completo** | 0,05887 ± 0,00041 | 0,05912 | — | — | — | — |
| BDL sin mezcla (K = 1) | 0,05898 ± 0,00041 | 0,05910 | 0,000107 (5/5) | 0,040 | −0,000016 | 0,625 |

**Clasificación (AUC, mayor es mejor).**

| Variante | AUC externo | AUC en prueba | Aporte del componente (externo) | p (*t* corregida, Holm) | Aporte en prueba | p (prueba, Holm) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BDL completo** | 0,8320 ± 0,0026 | 0,8311 | — | — | — | — |
| BDL sin mezcla (K = 1) | 0,8299 ± 0,0009 | 0,8311 | 0,00211 (5/5) | 0,652 | −0,00009 | 1,000 |
| BDL sin puntaje (aprende de la etiqueta) | 0,8292 ± 0,0028 | 0,8253 | 0,00284 (4/5) | 0,652 | 0,00578 | 0,065 |
| BDL con salida conforme empírica | 0,8313 ± 0,0024 | 0,8315 | 0,00075 (3/5) | 0,729 | −0,00044 | 1,000 |

Aporte: lo que mejora el modelo completo respecto de la variante (positivo: el componente ayuda). En prueba, la comparación es DeLong en clasificación y permutación pareada en regresión.

**Lectura.** En regresión (RMSE), retirar la mezcla ($K = 1$) cuesta 0,000107 en la media externa (el modelo completo va por delante en 5 de 5 pliegues; p = 0,040) y en prueba la variante queda 0,000016 por delante (p = 0,625). En clasificación (AUC), retirar la mezcla ($K = 1$) cuesta 0,00211 en la media externa (el modelo completo va por delante en 5 de 5 pliegues; p = 0,652) y en prueba la variante queda 0,00009 por delante (p = 1,000); aprender de la etiqueta en lugar del puntaje cuesta 0,00284 en la media externa (el modelo completo va por delante en 4 de 5 pliegues; p = 0,652) y en prueba cuesta 0,00578 (p = 0,065); cambiar la salida de la mezcla por la conforme empírica cuesta 0,00075 en la media externa (el modelo completo va por delante en 3 de 5 pliegues; p = 0,729) y en prueba la variante queda 0,00044 por delante (p = 1,000). Tras el ajuste de Holm, en los pliegues externos solo es significativo el costo de retirar la mezcla ($K = 1$) en regresión; en prueba, ninguno. En clasificación, lo que más pesa en prueba es aprender del puntaje continuo y no de su dicotomización: es la diferencia entre un AUC de 0,8311 y uno de 0,8253. Que los aportes sean tan pequeños es coherente con la sección 6.4: con todas las filas, un boosting sin mezcla ya está casi en el límite del conjunto. En regresión la mezcla no cambia lo que el modelo estima, que es la media condicional del puntaje, sino con cuántos datos lo consigue (sección 7.5). Lo que añade es lo que el modelo sabe del error: cuántos niveles hay, dónde están y con qué probabilidad (sección 7.8).
