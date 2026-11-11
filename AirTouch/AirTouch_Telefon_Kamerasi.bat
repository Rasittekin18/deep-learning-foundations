@echo off
chcp 65001 > nul
title AirTouch - Telefon Kamerasını Bağla (192.168.1.15)
color 0a

echo ============================================================
echo   📱 AirTouch: Telefon Kamerasından Canlı 60 FPS Takip
echo   Bağlantı: http://192.168.1.15:8080/video
echo ============================================================
echo.
echo [1/2] Telefon kamerasına bağlanılıyor...
echo [2/2] Canlı video penceresi açılıyor...
echo.
echo İpuçları:
echo  - ✊ Yumruk yapın: Oynat / Durdur ⏯️
echo  - ☝️ İşaret parmağı: Ses Aç (+4%%) 🔊
echo  - 🖐️ Açık el: Ses Kıs (-4%%) 🔉
echo  - Kapatmak için video penceresindeyken 'Q' tuşuna basın.
echo ============================================================
echo.

"C:\Users\Rasit\AppData\Local\Programs\Python\Python312\python.exe" AirTouch/live_desktop_cam.py "http://192.168.1.15:8080/video"

pause
