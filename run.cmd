@echo off
setlocal
cd /d "%~dp0"
set "PATH=%~dp0.venv;%~dp0.venv\Library\bin;%~dp0.venv\Scripts;%PATH%"
"%~dp0.venv\python.exe" "%~dp0MyPickPlace-ManiSkill\MyPickPlace-ManiSkill\scripts\dispatch.py" %*
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
