@echo off
title FASS CARLA Environment Setup
echo ========================================================
echo   FASS Environment Setup - CARLA 0.9.15 Downloader
echo ========================================================
echo.
echo This script will download the official CARLA 0.9.15 Windows 
echo release (approx. 15GB) and set up your Python environment.
echo.
echo Please ensure you have at least 40GB of free disk space.
echo.
pause

echo.
echo [1/3] Downloading CARLA 0.9.15 Windows Release...
echo This will take a while depending on your internet connection.
curl -L -# -o CARLA_0.9.15.zip "https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/CARLA_0.9.15.zip"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to download CARLA. Please download manually from:
    echo https://github.com/carla-simulator/carla/releases/tag/0.9.15
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Extracting CARLA into the current directory...
tar -xf CARLA_0.9.15.zip
del CARLA_0.9.15.zip

echo.
echo [3/3] Installing Python dependencies (PyTorch, Pygame, Numpy)...
pip install pygame numpy torch torchvision torchaudio
if %ERRORLEVEL% neq 0 (
    echo [WARNING] PIP install had some issues. Ensure Python 3.7+ is installed.
)

echo.
echo ========================================================
echo Setup Complete! 
echo ========================================================
echo 1. Run "WindowsNoEditor\CarlaUE4.exe" to start the simulator.
echo 2. Run "start_fass_autonomous.bat" to launch the AI demo.
echo ========================================================
pause
