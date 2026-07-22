@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  Consumer Complaint Classifier - Windows Setup
echo ============================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" was not found.
    echo Install Python 3.11 64-bit, then run this file again.
    pause
    exit /b 1
)

py -3.11 --version >nul 2>nul
if errorlevel 1 (
    echo Python 3.11 was not found.
    echo Install Python 3.11 64-bit, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 goto :error
)

echo Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Installing application packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup completed successfully.
echo Next, double-click run_app.bat.
echo.
pause
exit /b 0

:error
echo.
echo Setup failed. Review the error message above.
pause
exit /b 1
