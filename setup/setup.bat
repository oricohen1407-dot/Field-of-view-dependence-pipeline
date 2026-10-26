@echo off
REM Sets up the Python environment for Phase_Retrieval_stand_alone (the GUI pipeline) on
REM Windows. Creates Phase_Retrieval_stand_alone\.venv and installs requirements.txt.
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%..\Phase_Retrieval_stand_alone"
set "VENV_DIR=%PROJECT_DIR%\.venv"

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_BIN=python"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_BIN=py"
    ) else (
        echo Could not find a Python interpreter on PATH. Install Python 3.10+ and try again.
        exit /b 1
    )
)

if not exist "%VENV_DIR%" (
    echo Creating virtual environment at %VENV_DIR% ...
    %PYTHON_BIN% -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

REM Always go through "python -m pip", never the bare pip.exe: Windows security tools
REM (e.g. Device Guard) can block pip.exe right after it upgrades/replaces itself.
python -m pip install --upgrade pip
python -m pip install -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 exit /b 1

echo.
echo Setup complete.
echo To run the GUI:
echo   cd /d "%PROJECT_DIR%"
echo   .venv\Scripts\activate.bat
echo   python main.py

endlocal
