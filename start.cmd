@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-llama.ps1"
exit /b %ERRORLEVEL%
