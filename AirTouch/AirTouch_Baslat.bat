@echo off
if exist "%~dp0AirTouch\app.py" (
    cd /d "%~dp0"
) else (
    cd /d "%~dp0.."
)
chcp 65001 > nul
title AirTouch - Temassiz Masaustu Kontrol Merkezi
color 0b

set "PYTHONUTF8=1"
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

set "APP_PORT=8501"
for /f "usebackq delims=" %%P in (`""%PYTHON_EXE%" -m AirTouch.core.port_utils"`) do set "APP_PORT=%%P"

echo ============================================================
echo    AirTouch: Temassiz Medya, Sunum ve Sistem Kontrol Merkezi
echo    Laptop Varsayilan Kamerasi + PyTorch Iskelet AI
echo ============================================================
echo.
echo [1/2] Python: %PYTHON_EXE%
if not "%APP_PORT%"=="8501" echo [BILGI] 8501 portu kullanimda oldugu icin %APP_PORT% portu secildi.
echo [2/2] Streamlit Kontrol Merkezi Baslatiliyor...
echo.
echo Tarayici Adresi: http://localhost:%APP_PORT%
echo Cikmak icin bu pencereyi kapatabilir veya Ctrl+C yapabilirsiniz.
echo ============================================================
echo.

"%PYTHON_EXE%" -m streamlit run AirTouch/app.py --server.port %APP_PORT%
if errorlevel 1 (
    echo.
    echo [HATA] Uygulama baslatilirken bir sorun olustu!
    pause
)
