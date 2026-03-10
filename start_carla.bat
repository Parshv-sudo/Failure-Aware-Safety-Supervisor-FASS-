@echo off
title CARLA 0.9.15 Server (Background)

echo ===============================
echo Starting CARLA in background
echo ===============================

REM Go to the directory where this .bat lives
cd /d "%~dp0"

REM Enter CARLA packaged build folder
cd WindowsNoEditor

REM Start CARLA in background with minimal GPU usage
start /B CarlaUE4.exe ^
  -RenderOffScreen ^
  -quality-level=low ^
  -ResX=640 ^
  -ResY=480 ^
  -nosound

echo CARLA started in background.
echo You can close it from Task Manager when needed.