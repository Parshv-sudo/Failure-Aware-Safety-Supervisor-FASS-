@echo off
title FASS Accurate Autonomous Simulation
set PYTHON_EXE=C:\Users\parsh\AppData\Local\Programs\Python\Python37\python.exe
set ROOT_DIR=C:\Users\parsh\Desktop\CARLA 0.9.15\WindowsNoEditor\PythonAPI
set EXAMPLES_DIR=%ROOT_DIR%\examples

echo ========================================================
echo   FASS (Failure-Aware Safety Supervisor) Integration
echo   Random Map + Traffic + Stress-Test Scenarios
echo ========================================================
echo.

REM 1. Clean up leftover actors from previous runs
echo [STEP 1] Cleaning up leftover actors from previous runs...
cd /d "%EXAMPLES_DIR%"
%PYTHON_EXE% cleanup_world.py
timeout /t 3 /nobreak > nul

REM 2. Launch FASS Demo (random map, inline traffic, infinite loop)
echo [STEP 2] Launching FASS Demo (random map, ESC to quit)...
cd /d "%ROOT_DIR%"
set PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%

set CHECKPOINT=%ROOT_DIR%\fass_checkpoints\best_model.pt
if not exist "%CHECKPOINT%" (
    echo [WARNING] No ML checkpoint found. Running with heuristics...
    %PYTHON_EXE% -m fass_ml.demo_carla_loop
) else (
    echo [SUCCESS] Found ML checkpoint!
    %PYTHON_EXE% -m fass_ml.demo_carla_loop --checkpoint "%CHECKPOINT%"
)

echo.
echo Simulation session ended.
pause
