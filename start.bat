@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ==========================================
echo   Chat-Bot (Electron) Starting...
echo ==========================================
echo.

:: Find conda activate script
set CONDA_BAT=
if exist "D:\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=D:\miniconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
)

if "%CONDA_BAT%"=="" (
    echo [WARNING] conda activate script not found. Trying PATH...
    call conda activate new
) else (
    call "%CONDA_BAT%" new
)

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate conda env "new"
    pause
    exit /b 1
)

echo [OK] conda env "new" activated

cd electron
if not exist node_modules (
    echo [INFO] Installing npm dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed
        pause
        exit /b 1
    )
)

call npm start
if errorlevel 1 (
    echo.
    echo [ERROR] App exited with an error.
    pause
    exit /b 1
)

exit /b 0
