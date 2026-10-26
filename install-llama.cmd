@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-llama.ps1"
exit /b %ERRORLEVEL%
