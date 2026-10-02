@echo off
title Modelo nuevo Entregable 3 - NO CERRAR
rem Modelo nuevo del Entregable 3 (EBM): comparacion con la Entrega 2 y sus 20 corridas en el pipeline,
rem una detras de otra en esta ventana y con prioridad baja.
rem Si el PC se apaga o se cierra la ventana, basta con volver a ejecutar este archivo: la comparacion
rem ya hecha se salta, igual que las corridas terminadas; las fallidas se repiten.
rem Para detenerlo limpiamente: crear el archivo resultados\DETENER (termina al acabar la corrida en curso).
rem Progreso: resultados\comparacion_entrega2\progreso.log y resultados\progreso.log
cd /d "%~dp0"
".venv\Scripts\python.exe" -u comparar_entrega2.py
".venv\Scripts\python.exe" -u correr_140.py --plan modelo_nuevo
