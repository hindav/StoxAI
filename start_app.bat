@echo off
echo ===============================================================================
echo       STOCK PREDICTION APP LAUNCHER
echo ===============================================================================
echo.
echo 1. Activating Anaconda Environment (base)...
call %USERPROFILE%\anaconda3\Scripts\activate.bat base

echo.
echo 2. Starting Flask Server...
echo    The server will run in a new window.
echo    Do not close that window while using the app.
start "Stock Prediction Server" cmd /k python flask_app.py

echo.
echo 3. Waiting for server to initialize (3 seconds)...
timeout /t 3 /nobreak >nul

echo.
echo 4. Opening Web Application...
start http://localhost:5000

echo.
echo ===============================================================================
echo     App started successfully!
echo    You can minimize this window.
echo ===============================================================================
pause
