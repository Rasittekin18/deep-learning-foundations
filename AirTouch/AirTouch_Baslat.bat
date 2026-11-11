@echo off
chcp 65001 > nul
title AirTouch - Temassız Medya Kontrol Merkezi
color 0b

echo ============================================================
echo    🖐️ AirTouch: Temassız Medya ve Sistem Kontrol Merkezi
echo    Deep Learning & PyTorch CNN Tabanlı Masaüstü Asistanı
echo ============================================================
echo.
echo [1/2] Model ve Sistem Motorları Hazırlanıyor...
echo [2/2] Streamlit Web Arayüzü Başlatılıyor...
echo.
echo Tarayıcınız otomatik olarak açılacaktır (http://localhost:8501)
echo Çıkmak için bu pencereyi kapatabilir veya Ctrl+C yapabilirsiniz.
echo ============================================================
echo.

streamlit run AirTouch/app.py

pause

