@echo off
rem Lanza o reanuda las 140 corridas del Entregable 3 en una ventana aparte, con prioridad baja.
rem Si el PC se apago o se cerro la ventana, basta con volver a ejecutar este archivo: las corridas
rem terminadas se saltan y las que fallaron se repiten.
rem Para detenerlo limpiamente: crear el archivo resultados\DETENER (termina al acabar la corrida en curso).
rem Progreso: resultados\progreso.log
cd /d "%~dp0"
start "Corridas Entregable 3 - NO CERRAR" /belownormal /min ".venv\Scripts\python.exe" -u correr_140.py
