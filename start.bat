@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Please run install.bat first.
  pause
  exit /b 1
)
start "SoundScribe" .venv\Scripts\pythonw.exe main.py

