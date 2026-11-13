@echo off
if exist "%~dp0AirTouch\live_desktop_cam.py" (
    cd /d "%~dp0"
) else (
    cd /d "%~dp0.."
)
chcp 65001 > nul
title AirTouch Pro - Canli Kamera, Sesli Komut ve Cift Ekran HUD
color 0a

set "PYTHONUTF8=1"
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo =====================================================================
echo   AirTouch v2.1 Pro: El Iskeleti + Sesli Komut + Cift Ekran Kontrol
echo   Laptop Varsayilan Kamerasi (0) + Cevrimdisi Vosk Turkce Ses AI
echo =====================================================================
echo.
echo [1/2] MediaPipe Iskelet AI + Vosk Turkce Ses Motoru yukleniyor...
echo [2/2] Laptop kamerasi (0) ve masaustu HUD baslatiliyor...
echo.
echo Hangi Parmak Ne Yapar? (HUD uzerindeki [G] paneli canli gosterir)
echo  - Ornek Ses Komutlari : 'video dursun', 'video baslasin', 'sesi artir',
echo                          'youtube da tarkan ac', 'tam ekran', 'ileri sar',
echo                          'diger ekran', 'ekrana tasi', 'uygulama degistir'
echo.
echo Kisayollar:
echo  [K] El Kapisi (Avuc/Direkt/Ses) / [G] Rehber / [V] Ses / [C] Mini PiP
echo  [W] Ekran Tasi / [M] Profil (Medya/YT/CiftEkran/Odak) / [SPACE] Bekle / [Q] Cik
echo =====================================================================
echo.

"%PYTHON_EXE%" AirTouch/live_desktop_cam.py --camera 0
if errorlevel 1 (
    echo.
    echo [HATA] Canli kamera penceresi hata ile kapandi!
    pause
)
