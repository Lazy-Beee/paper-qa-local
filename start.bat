@echo off
REM One-click launcher for the Paper QA web UI on Windows.
REM Activates the local venv and starts the Gradio app on http://127.0.0.1:7860

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. First-time setup:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "config.toml" (
    if exist "config.example.toml" (
        echo [INFO] config.toml not found - seeding from config.example.toml
        copy /Y "config.example.toml" "config.toml" >nul
    ) else (
        echo [ERROR] Neither config.toml nor config.example.toml exists.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python src\web.py
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] web.py exited with code %EXITCODE%.
    echo Common causes:
    echo   - LM Studio is not running ^(default endpoint: http://localhost:1234^)
    echo   - A required model is not loaded ^(check [llm] / [reranker] in config.toml^)
    echo   - Port 7860 is already in use
    echo See the latest log/web_*.log for details.
    pause
)
exit /b %EXITCODE%
