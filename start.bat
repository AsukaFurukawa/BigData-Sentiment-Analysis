@echo off
REM Advanced Sentiment Analysis Engine Startup Script
REM --------------------------------------------------

echo === Sentiment Analysis Engine Startup ===
echo.

REM Check for Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Install dependencies
echo Checking dependencies...
python install_dependencies.py
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies may not be installed correctly
    echo The system will try to continue, but some features may not work
    timeout /t 5
)

echo.
echo What would you like to do?
echo 1. Launch Dashboard
echo 2. Collect Data
echo 3. Setup Databases
echo 4. Run Complete Pipeline
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Launching Dashboard...
    python run_advanced.py dashboard
) else if "%choice%"=="2" (
    echo.
    echo Starting Data Collection...
    python run_advanced.py collect
) else if "%choice%"=="3" (
    echo.
    echo Setting up databases...
    python setup_databases.py
) else if "%choice%"=="4" (
    echo.
    echo Running complete pipeline...
    python setup_databases.py
    if %errorlevel% neq 0 (
        echo Database setup failed. Cannot continue.
        pause
        exit /b 1
    )
    
    echo.
    echo Collecting data...
    python run_advanced.py collect
    
    echo.
    echo Launching dashboard...
    python run_advanced.py dashboard
) else if "%choice%"=="5" (
    echo Exiting...
    exit /b 0
) else (
    echo.
    echo Invalid choice. Please enter a number between 1 and 5.
    pause
    exit /b 1
)

echo.
echo Process completed.
echo Thank you for using the Sentiment Analysis Engine.
pause 