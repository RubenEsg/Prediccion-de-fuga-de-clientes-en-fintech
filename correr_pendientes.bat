@echo off
title Pendientes Entregable 3 - NO CERRAR
rem Experimentos que van despues de las corridas del modelo nuevo, uno detras de otro:
rem   computo.py      seccion 4 de la guia (tiempos: necesita el PC sin otros experimentos)
rem   complementos.py sensibilidad a semillas, calibracion, predicciones fuera de pliegue y SHAP
rem Cada guion espera por si mismo a que terminen la comparacion y las 20 corridas de la EBM.
rem Si el PC se apaga, basta con volver a ejecutar este archivo: lo ya hecho se salta.
rem Progreso: resultados\computo\progreso.log y resultados\complementos\progreso.log
cd /d "%~dp0"
".venv\Scripts\python.exe" -u computo.py --esperar
".venv\Scripts\python.exe" -u complementos.py --esperar
