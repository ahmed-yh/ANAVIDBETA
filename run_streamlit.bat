@echo off
REM Streamlit App Launcher for ANAVID Queue Intelligence System
REM Windows batch script - Uses virtual environment with CUDA support

echo.
echo =====================================
echo   ANAVID Streamlit App Launcher
echo =====================================
echo.

REM Check for virtual environment
if exist "anavid_py311\Scripts\python.exe" (
    echo [OK] Using virtual environment: anavid_py311
    set PYTHON_CMD=anavid_py311\Scripts\python.exe
    set PIP_CMD=anavid_py311\Scripts\pip.exe
) else if exist "anavid\Scripts\python.exe" (
    echo [OK] Using virtual environment: anavid
    set PYTHON_CMD=anavid\Scripts\python.exe
    set PIP_CMD=anavid\Scripts\pip.exe
) else (
    echo [WARNING] Virtual environment not found, using system Python
    echo [NOTE] System Python may not have CUDA support
    set PYTHON_CMD=python
    set PIP_CMD=pip
)

REM Check if Python is installed
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

%PYTHON_CMD% --version
echo.

REM Check if streamlit is installed
%PYTHON_CMD% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Streamlit not found, installing dependencies...
    %PIP_CMD% install -r requirements.txt
    echo.
)

REM Check CUDA availability
echo [INFO] Checking CUDA availability...
%PYTHON_CMD% -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo.

echo [OK] Dependencies ready
echo.

REM Create necessary directories
if not exist "data\input" mkdir data\input
if not exist "data\output\segments" mkdir data\output\segments
if not exist "results" mkdir results

echo [OK] Directories created
echo.

REM Launch Streamlit app
echo Launching Streamlit app on http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

%PYTHON_CMD% -m streamlit run streamlit_app.py --logger.level=info --server.maxMessageSize=200

pause
