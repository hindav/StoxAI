@echo off
echo ===============================================================================
echo       STOCK PREDICTION DASHBOARD (Streamlit)
echo ===============================================================================
echo.
echo 1. Activating Anaconda Environment (base)...
call %USERPROFILE%\anaconda3\Scripts\activate.bat base

echo.
echo 2. Starting Unified API Backend (Port 8001)...
start "Unified Stock API" cmd /k python Models\unified_predictor.py

echo.
echo 3. Waiting for API to initialize (3 seconds)...
timeout /t 3 /nobreak >nul

echo.
echo 4. Starting Streamlit Dashboard...
start "Streamlit Dashboard" cmd /k streamlit run streamlit_ui.py

echo.
echo ===============================================================================
echo     Dashboard started!
echo     Access it at local URL provided in the Streamlit window.
echo ===============================================================================
pause
