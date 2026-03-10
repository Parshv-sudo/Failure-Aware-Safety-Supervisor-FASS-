@echo off
title FASS Integrated Autonomous Simulation
set PYTHON_EXE=C:\Users\parsh\AppData\Local\Programs\Python\Python37\python.exe
set CARLA_DIR=C:\Users\parsh\Desktop\CARLA 0.9.15
set ADAS_DIR=%CARLA_DIR%\adas_supervision_project
set EXAMPLES_DIR=%CARLA_DIR%\WindowsNoEditor\PythonAPI\examples

echo =========================================================================
echo   FASS x ADAS Supervision Integration
echo   Evaluating AI Risk Models with Level 2 Human-in-the-loop State Machine
echo =========================================================================
echo.

REM 1. Clean up leftover actors from previous runs
echo [STEP 1] Cleaning up leftover actors from previous runs...
cd /d "%EXAMPLES_DIR%"
%PYTHON_EXE% cleanup_world.py
timeout /t 3 /nobreak > nul

REM 2. Map Selection
echo.
echo [STEP 2] Select the CARLA Map to load:
echo 1. Town01 (Basic, small town)
echo 2. Town03 (Complex intersection, roundabout)
echo 3. Town04 (Infinity highway)
echo 4. Town05 (Multiple lanes, intersections)
choice /n /c 1234 /m "Enter your choice [1-4]: "
if errorlevel 4 set MAP_NAME=Town05 & goto map_selected
if errorlevel 3 set MAP_NAME=Town04 & goto map_selected
if errorlevel 2 set MAP_NAME=Town03 & goto map_selected
if errorlevel 1 set MAP_NAME=Town01 & goto map_selected
:map_selected
echo Selected Map: %MAP_NAME%

REM 3. Launch Integrated FASS + ADAS Demo
echo.
echo [STEP 3] Launching Integrated Simulation...
echo          (Press Ctrl+C to stop the simulation when finished)
echo.
cd /d "%ADAS_DIR%"
%PYTHON_EXE% main.py --config config/fass_integration_config.yaml --map %MAP_NAME%

REM 4. Automatic Visualization Post-Run
echo.
echo [STEP 4] Simulation sequence ended. 
echo          Searching for the latest blackbox log to generate visualization...
echo.

REM Find the most recently created .jsonl file in the logs directory
FOR /F "delims=|" %%I IN ('DIR "logs\fass_integration_runs\*_blackbox.jsonl" /B /O:D /T:C') DO SET LATEST_LOG=%%I

if "%LATEST_LOG%"=="" (
    echo [WARNING] No blackbox log found in logs\fass_integration_runs\
    echo           Did the simulation run successfully?
) else (
    echo [SUCCESS] Found log: %LATEST_LOG%
    echo           Launching Telemetry and Risk Dashboard...
    %PYTHON_EXE% visualize_log.py "logs\fass_integration_runs\%LATEST_LOG%"
)

echo.
echo Evaluation session complete.
pause
