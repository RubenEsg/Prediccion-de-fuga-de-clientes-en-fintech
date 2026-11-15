@echo off
rem Fuga de clientes - Laboratorio del modelo (Dash), para Windows.
rem Doble clic: la primera vez crea el entorno .venv en esta carpeta e instala requirements.txt (unos minutos, con internet);
rem despues abre http://127.0.0.1:8060 en el navegador. Para detener la app, cierre esta ventana.
title Fuga de clientes - Laboratorio del modelo
cd /d "%~dp0"
set "PORT=8060"

if exist ".venv\Scripts\python.exe" goto correr

echo Primera vez: creando el entorno de Python en .venv ...
set "PYBASE="
where py >nul 2>nul && set "PYBASE=py -3"
if not defined PYBASE where python >nul 2>nul && set "PYBASE=python"
if not defined PYBASE goto sin_python
%PYBASE% -m venv .venv
if errorlevel 1 goto sin_python
echo Instalando las dependencias de requirements.txt ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fallo_pip

:correr
echo.
echo Iniciando la app. El navegador se abre solo en http://127.0.0.1:8060
echo Para detener la app, cierre esta ventana.
echo.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 240; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri 'http://127.0.0.1:8060/' | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }; Start-Process 'http://127.0.0.1:8060/'"
".venv\Scripts\python.exe" -B app.py
echo.
echo La app se detuvo.
pause
exit /b 0

:sin_python
echo No se encontro Python 3.11 o mas reciente. Instalelo desde https://www.python.org y vuelva a intentar.
if exist ".venv" rmdir /s /q ".venv"
pause
exit /b 1

:fallo_pip
echo Fallo la instalacion de las dependencias. Revise la conexion a internet y vuelva a intentar.
if exist ".venv" rmdir /s /q ".venv"
pause
exit /b 1
