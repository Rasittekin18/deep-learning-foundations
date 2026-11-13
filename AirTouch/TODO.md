# 🖐️ AirTouch Projesi: Mimari ve Geliştirme Durumu (v2.0)

> **Proje Amacı:** Normal laptop varsayılan web kamerası karşısında el hareketleri yaparak klavyeye veya fareye dokunmadan YouTube, Spotify, PowerPoint ve Windows ses/medya/tarayıcı kontrollerini temassız yöneten; PyTorch Deep Learning + MediaPipe 21 Nokta El İskeleti tabanlı masaüstü yapay zeka asistanı.

---

## 🔧 Tespit Edilen Hatalar ve Yapılan Mimari İyileştirmeler

1. **Kamera & Çalışma Ortamı Düzeltmesi:**
   - Telefon IP kamerası (`192.168.1.15:8080`) varsayılan bağımlılığı kaldırıldı; tüm akışlar **laptop varsayılan kamerası (`index 0`, DirectShow düşük gecikme)** üzerine geçirildi.
   - Hatalı sabit `Python312` yolları temizlenerek `sys.executable` ve aktif Python 3.11 ortamına bağlandı.
2. **El Tespit & Alan Uyumsuzluğu (Domain Gap) Çözümü:**
   - Yüzü el sanan eski YCrCb renk eşiği yerine **MediaPipe `HandLandmarker` (21 3D eklem)** entegre edildi (`core/hand_tracker.py`).
   - 88 boyutlu pozisyon/ölçek/el-tarafı değişmez öznitelik çıkarıcı ve **`AirTouchLandmarkNet`** PyTorch Residual MLP modeli eklendi (`core/model.py`, `core/landmark_trainer.py`).
   - `X.npy` Sign Language veri setindeki hatalı sınıf etiketleri düzeltildi (`SIGN_LANGUAGE_DIGIT_MAP`: `0→9, 1→0, 2→7, 3→6, 4→1, 5→8, 6→4, 7→3, 8→2, 9→5`).
3. **Arayüz, HUD & Senkronizasyon:**
   - `app.py` içindeki tek kare fotoğraf çekme (`st.camera_input`) kısıtlaması kaldırılarak gerçek zamanlı canlı laptop kamera akışı eklendi; eski `use_column_width` hatası ve PIL/OpenCV Türkçe karakter (`[]` / `?`) bozulmaları `core/hud_renderer.py` (TrueType `Segoe UI`) ile tamamen giderildi.
   - `core/settings.py` (`data/settings.json`) ile Streamlit arayüzü ve Masaüstü HUD penceresi arasında canlı ayar/profil senkronizasyonu sağlandı.
   - **Medya**, **Sunum** ve **Tarayıcı** profilleri, Bekleme Modu (`SPACE`), Her Zaman Üstte Modu (`A`), Loş Işık İyileştirme (`CLAHE`) ve **Özel Jest Veri Toplama & Yeniden Eğitim Stüdyosu** eklendi.

---

## 📌 Bileşen Tablosu

| Bileşen | Dosya | Durum |
| :--- | :--- | :---: |
| Yapılandırma & Profiller | `AirTouch/config.py` | ✅ Aktif |
| Kalıcı Ayar & Hot-Reload | `AirTouch/core/settings.py` | ✅ Aktif |
| Laptop Kamera & CLAHE | `AirTouch/core/camera.py` | ✅ Aktif |
| MediaPipe 21-Nokta Takip | `AirTouch/core/hand_tracker.py` | ✅ Aktif |
| PyTorch CNN & LandmarkNet | `AirTouch/core/model.py` | ✅ Aktif |
| Landmark Eğitim Motoru | `AirTouch/core/landmark_trainer.py` | ✅ Aktif |
| Türkçe TrueType HUD | `AirTouch/core/hud_renderer.py` | ✅ Aktif |
| Birleşik Karar Motoru | `AirTouch/core/engine.py` | ✅ Aktif |
| Win32 SendInput Kontrolcü | `AirTouch/core/system_controller.py` | ✅ Aktif |
| Masaüstü Canlı HUD | `AirTouch/live_desktop_cam.py` | ✅ Aktif |
| Streamlit Kontrol Merkezi | `AirTouch/app.py` | ✅ Aktif |
