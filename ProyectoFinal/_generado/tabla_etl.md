| Paso | Resultado |
| :--- | :--- |
| 1. Carga | 48.723 clientes con 54 columnas; el detalle de transacciones, en una tabla aparte |
| 2. Auditoría de las transacciones | 102 filas exactamente duplicadas, que se eliminan (quedan 3.159.055); 0 montos no positivos, 0 fechas nulas, 0 transacciones sin cliente y 0 clientes sin transacciones |
| 3. Partición | Entrenamiento y prueba (80/20, semilla 42), antes de mirar el objetivo y antes de derivar variables |
| 4. Variables derivadas | 12, del detalle de transacciones y sin usar el objetivo: `n_tx`, `monto_total`, `monto_medio`, `monto_mediano`, `recencia_dias`, `dias_activo`, `volatilidad_monto`, `meses_activos`, `prop_Deposit`, `prop_Payment`, `prop_Transfer`, `prop_Withdrawal` |
| 5. Columnas eliminadas | 17: `customer_id`, `active_products`, `customer_lifetime_value`, `clv_segment`, `average_transaction_value`, `total_transaction_volume`, `avg_daily_transactions`, `monthly_transaction_count`, `first_transaction_date`, `last_transaction_date`, `nps_score`, `occupation`, `complaint_topics`, `feature_requests`, `last_survey_date`, `first_tx`, `last_tx` |
| 6. Predictores | 48: 32 numéricos, 7 booleanos y 9 categóricos |
