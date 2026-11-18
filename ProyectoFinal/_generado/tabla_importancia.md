| N.º | Variable | Aporte medio absoluto al puntaje | Parte del total |
| ---: | :--- | ---: | ---: |
| 1 | `age` | 0,02084 | 45,5 % |
| 2 | `app_logins_frequency` | 0,01512 | 33,0 % |
| 3 | `satisfaction_score` | 0,00956 | 20,9 % |
| 4 | `tx_count` | 0,00017 | 0,4 % |
| 5 | `n_tx` | 0,00003 | 0,1 % |
| 6 | `base_satisfaction` | 0,00002 | 0,1 % |
| 7 | `total_tx_volume` | 0,00001 | 0,0 % |
| 8 | `prop_Withdrawal` | 0,00001 | 0,0 % |

Las cinco variables de más peso en BDL son `age`, `app_logins_frequency`, `satisfaction_score`, `tx_count`, `n_tx`; en el mejor benchmark de regresión (XGBoost, valores SHAP de 500 clientes de prueba), `age`, `app_logins_frequency`, `satisfaction_score`, `base_satisfaction`, `monto_mediano`.
