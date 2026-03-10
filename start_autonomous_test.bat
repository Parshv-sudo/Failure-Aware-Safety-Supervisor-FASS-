@echo off
title CARLA Autonomous Traffic Test
set PYTHON_EXE=C:\Users\parsh\AppData\Local\Programs\Python\Python37\python.exe
set EXAMPLES_DIR=%~dp0WindowsNoEditor\PythonAPI\examples

echo ==========================================
echo   CARLA Autonomous Driving Test Setup
echo ==========================================
echo.

REM 1. Start Traffic and Pedestrians in a separate window
echo Starting 50 vehicles and 30 pedestrians...
start "CARLA Traffic Generator" /D "%EXAMPLES_DIR%" %PYTHON_EXE% generate_traffic.py -n 50 -w 30 --asynch

REM 2. Wait a few seconds for actors to spawn
timeout /t 5 /nobreak > nul

REM 3. Start the Autonomous Agent
echo Starting Autonomous Agent (Behavior Mode)...
cd /d "%EXAMPLES_DIR%"
%PYTHON_EXE% automatic_control.py --agent Behavior --loop

echo.
echo Simulation stopped.
pause
