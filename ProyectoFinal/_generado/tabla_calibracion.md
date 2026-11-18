| Modelo | AUC | Precisión media | Captura al 25 % | Pérdida log. | Brier | ECE | Pérdida log. (isotónica) | Brier (isotónica) | ECE (isotónica) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **BDL (original)** | 0,8311 | 0,5103 | 51,4 % | 0,3688 | 0,1301 | 0,0081 | 0,3683 | 0,1301 | 0,0067 |
| LightGBM | 0,8296 | 0,5535 | 51,0 % | 0,4571 | 0,1667 | 0,1488 | 0,3698 | 0,1315 | 0,0219 |
| EBM | 0,8289 | 0,5512 | 51,2 % | 0,5214 | 0,1732 | 0,1698 | 0,3863 | 0,1338 | 0,0050 |
| XGBoost | 0,8283 | 0,5480 | 51,0 % | 0,5621 | 0,1875 | 0,2161 | 0,3675 | 0,1305 | 0,0066 |
| Random Forest | 0,8260 | 0,5444 | 49,4 % | 0,3735 | 0,1323 | 0,0211 | 0,3676 | 0,1307 | 0,0107 |
| Árbol de decisión | 0,8243 | 0,5304 | 49,5 % | 0,4524 | 0,1690 | 0,1442 | 0,3678 | 0,1308 | 0,0111 |
| k-NN | 0,7874 | 0,5017 | 50,8 % | 0,5045 | 0,1660 | 0,1079 | 0,4549 | 0,1510 | 0,0077 |
| Regresión logística | 0,7709 | 0,5079 | 51,1 % | 0,5747 | 0,1977 | 0,1993 | 0,4586 | 0,1521 | 0,0053 |
| SVM lineal | 0,7706 | 0,5067 | 51,1 % | — | — | — | 0,4624 | 0,1523 | 0,0072 |
| Naive Bayes | 0,6662 | 0,3627 | 40,8 % | 3,2131 | 0,6600 | 0,6647 | 0,5250 | 0,1736 | 0,0072 |

ECE: error de calibración esperado con diez tramos de igual número de clientes; el dashboard lo calcula con diez tramos de igual ancho, como el Entregable 3, y por eso sus cifras no coinciden exactamente con las de esta tabla. Isotónica: recalibración ajustada con las predicciones fuera de pliegue de entrenamiento. El SVM lineal no da probabilidades: solo se reporta recalibrado.

**Dos columnas dependen de los empates.** La probabilidad del original toma 219 valores distintos, y 4.354 clientes comparten la del corte de la campaña (Tabla 6.11). Su captura al 25 % es la del desempate del protocolo; con un desempate al azar se espera 51,3 % ± 0,6 puntos. Su precisión media es la de la convención de scikit-learn, que cuenta a los empatados como un solo punto; con los empates rotos al azar es 0,5507 ± 0,0046.
