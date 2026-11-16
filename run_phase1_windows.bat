@echo off
cd /d "%~dp0"
title EvoMonitor Phase 1
where py >nul 2>nul
if errorlevel 1 (echo Python not found. Install Python 3.11 from https://www.python.org/downloads/ and select Add Python to PATH. & pause & exit /b 1)
if not exist .venv (py -3.11 -m venv .venv || py -m venv .venv)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (echo Installation failed. Send the full error screenshot. & pause & exit /b 1)
python -m streamlit run app.py
pause
