@echo off
setlocal
cd /d "%~dp0"
if defined MYPICKPLACE_PYTHON goto run
set "MYPICKPLACE_PYTHON=%~dp0.venv\python.exe"
if exist "%MYPICKPLACE_PYTHON%" goto run
set "MYPICKPLACE_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%MYPICKPLACE_PYTHON%" goto run
echo Set MYPICKPLACE_PYTHON to your environment's python.exe.
exit /b 1
:run
for %%I in ("%MYPICKPLACE_PYTHON%") do set "PYTHON_DIR=%%~dpI"
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%Library\bin;%PYTHON_DIR%Scripts;%PATH%"
set "PYTHONIOENCODING=utf-8"
"%MYPICKPLACE_PYTHON%" "%~dp0MyPickPlace-ManiSkill\MyPickPlace-ManiSkill\scripts\dispatch.py" %*
set "result=%errorlevel%"
exit /b %result%
