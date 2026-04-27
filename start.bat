@echo off
REM One-click launcher for the Paper QA web UI on Windows.
REM Activates the local venv and starts the Gradio app on http://127.0.0.1:7860

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run: python -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python src\web.py

REM Keep the window open if the script exits with an error.
if errorlevel 1 pause
