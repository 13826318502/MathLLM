@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Create .venv and install requirements.txt first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m app.tools.config_ui --env-file ".env.local"
echo.
echo Configuration closed. Run start_local.bat to launch MathLLM.
pause
