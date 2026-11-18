**Regresión (RMSE, menor es mejor).** El original es el mejor de los tres en 16 de 18 ciudades.

| Ciudad que se deja fuera | Clientes | BDL (original) | XGBoost | LightGBM |
| :--- | ---: | ---: | ---: | ---: |
| Armenia, Quindío | 405 | 0,06048 | 0,06048 | 0,06060 |
| Barranquilla, Atlántico | 3.069 | 0,05873 | 0,05879 | 0,05887 |
| Bogotá, Cundinamarca | 13.613 | 0,05944 | 0,05950 | 0,05950 |
| Bucaramanga, Santander | 1.512 | 0,05897 | 0,05913 | 0,05907 |
| Cali, Valle del Cauca | 4.669 | 0,05817 | 0,05825 | 0,05827 |
| Cartagena, Bolívar | 1.898 | 0,05878 | 0,05888 | 0,05887 |
| Cúcuta, Norte de Santander | 1.556 | 0,05879 | 0,05890 | 0,05896 |
| Ibagué, Tolima | 849 | 0,05793 | 0,05819 | 0,05819 |
| Manizales, Caldas | 782 | 0,05883 | 0,05891 | 0,05902 |
| Medellín, Antioquia | 5.909 | 0,05895 | 0,05908 | 0,05909 |
| Montería, Córdoba | 172 | 0,05396 | 0,05413 | 0,05423 |
| Neiva, Huila | 395 | 0,06065 | 0,06075 | 0,06067 |
| Pasto, Nariño | 786 | 0,05862 | 0,05876 | 0,05879 |
| Pereira, Risaralda | 1.131 | 0,05810 | 0,05824 | 0,05822 |
| Santa Marta, Magdalena | 1.243 | 0,05730 | 0,05732 | 0,05738 |
| Sincelejo, Sucre | 175 | 0,06123 | 0,06116 | 0,06123 |
| Valledupar, Cesar | 406 | 0,05780 | 0,05788 | 0,05782 |
| Villavicencio, Meta | 408 | 0,05751 | 0,05751 | 0,05755 |
| **Media** | — | **0,05857** | **0,05866** | **0,05869** |

**Clasificación (AUC, mayor es mejor).** El original es el mejor de los tres en 6 de 18 ciudades.

| Ciudad que se deja fuera | Clientes | BDL (original) | XGBoost | LightGBM |
| :--- | ---: | ---: | ---: | ---: |
| Armenia, Quindío | 405 | 0,8572 | 0,8571 | 0,8483 |
| Barranquilla, Atlántico | 3.069 | 0,8236 | 0,8191 | 0,8170 |
| Bogotá, Cundinamarca | 13.613 | 0,8345 | 0,8342 | 0,8376 |
| Bucaramanga, Santander | 1.512 | 0,8312 | 0,8398 | 0,8287 |
| Cali, Valle del Cauca | 4.669 | 0,8443 | 0,8482 | 0,8418 |
| Cartagena, Bolívar | 1.898 | 0,8245 | 0,8253 | 0,8141 |
| Cúcuta, Norte de Santander | 1.556 | 0,8256 | 0,8283 | 0,8282 |
| Ibagué, Tolima | 849 | 0,8434 | 0,8516 | 0,8465 |
| Manizales, Caldas | 782 | 0,8331 | 0,8457 | 0,8515 |
| Medellín, Antioquia | 5.909 | 0,8301 | 0,8318 | 0,8283 |
| Montería, Córdoba | 172 | 0,8453 | 0,8505 | 0,8522 |
| Neiva, Huila | 395 | 0,7898 | 0,7861 | 0,7691 |
| Pasto, Nariño | 786 | 0,8443 | 0,8387 | 0,8339 |
| Pereira, Risaralda | 1.131 | 0,8477 | 0,8399 | 0,8423 |
| Santa Marta, Magdalena | 1.243 | 0,8328 | 0,8330 | 0,8238 |
| Sincelejo, Sucre | 175 | 0,8413 | 0,8471 | 0,8447 |
| Valledupar, Cesar | 406 | 0,8412 | 0,8262 | 0,8139 |
| Villavicencio, Meta | 408 | 0,8291 | 0,8257 | 0,8399 |
| **Media** | — | **0,8344** | **0,8349** | **0,8312** |

**Lectura.** En regresión, la media simple de las 18 ciudades es 0,05857 para el original y 0,05866 para XGBoost; con pliegues al azar y la misma configuración, el original obtiene 0,05888 ± 0,00041: evaluar en una ciudad que no se vio no empeora su resultado. En clasificación, la media simple de las 18 ciudades es 0,8344 para el original y 0,8349 para XGBoost; con pliegues al azar y la misma configuración, el original obtiene 0,8339 ± 0,0030: evaluar en una ciudad que no se vio no empeora su resultado. Las ciudades pequeñas pesan lo mismo que las grandes en esa media, y en las de pocos cientos de clientes (6 de 18 tienen menos de 500) las diferencias entre modelos son ruido de muestreo. Como dejar fuera una ciudad no degrada a ninguno de los tres modelos, en COFINFAD la relación entre los predictores y el puntaje no parece cambiar de una ciudad a otra: este estudio no es una prueba exigente de cambio de distribución.
