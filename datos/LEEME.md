# Datos: COFINFAD

*Colombian Fintech Financial Analytics Dataset*, publicado en [Mendeley Data](https://data.mendeley.com/datasets/mhb4zn3258/1)
(DOI 10.17632/mhb4zn3258.1, licencia CC BY 4.0). Dos tablas de 2023: `customer_data.csv` (48.723 clientes, 54 columnas) y
`transactions_data.csv` (3.159.157 transacciones).

Aquí están **comprimidas** (`customer_data.zip` y `transactions_data.zip`), porque `transactions_data.csv` pesa 114 MB y
GitHub no admite archivos de más de 100 MB. El código las descomprime solo la primera vez que las necesita
(`Entrega3/src/config.py`, función `localizar_datos`): no hay que hacer nada. Si se prefiere, se extraen a mano en esta
misma carpeta.
