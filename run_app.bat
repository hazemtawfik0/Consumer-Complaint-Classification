@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment was not found.
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)

echo.
echo Starting Consumer Complaint Classifier...
echo The browser should open at http://127.0.0.1:7860
echo Keep this terminal window open while using the application.
echo Press Ctrl+C to stop the server.
echo.

".venv\Scripts\python.exe" app.py

echo.
echo Application stopped.
pause
