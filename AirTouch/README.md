# 🖐️ AirTouch v2.1 Pro: El İskeleti, Türkçe Sesli Komut & Çift Ekran Kontrol Merkezi

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-21%203D%20Landmarks-0097A7?logo=google&logoColor=white)
![Vosk](https://img.shields.io/badge/Vosk-Offline%20Turkish%20ASR-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/pytest-52%20passed%20(100%25)-brightgreen)

<br/>

**AirTouch v2.1 Pro**, normal laptop varsayılan kamerası ve mikrofonu üzerinden elinize temas etmeden **YouTube, medya oynatıcılar, tarayıcı sekmeleri ve çift monitör** pencerelerini yönetmenizi sağlayan çevrimdışı yapay zeka asistanıdır.

</div>

---

## 🌟 Öne Çıkan Özellikler (v2.1 Pro)

- **🦴 21 Noktalı 3D El İskeleti (`MediaPipe HandLandmarker` + `AirTouchLandmarkNet`):** Yüzü/arka planı el sanan renk eşiklemesi yerine el boyutundan ve konumundan bağımsız 88 boyutlu biyomekanik öznitelik çıkarıcı + PyTorch Residual MLP.
- **🎤 Çevrimdışı Türkçe Sesli Komut Motoru (`Vosk tr`):**
  - Windows `Stereo Karışımı` gibi loopback aygıtlarını otomatik eler, gerçek mikrofonu (`Mikrofon Dizisi` vb.) seçer ve `48 kHz → 16 kHz` dönüşümü + otomatik kazanç (AGC) + hızlı VAD uygular.
  - *"video dursun"*, *"video başlasın"*, *"sesi artır"*, *"tam ekran"*, *"diğer ekran"* komutlarını tanır.
  - **Dinamik YouTube Video Açma:** *"youtube'da barış manço aç"*, *"youtube tarkan kuzum açılsın"* cümlelerinde doğrudan ilk video sonucunu açar.
  - **Kamerasız Sesli Asistan:** Streamlit arayüzündeki **🎙️ Sadece Sesli Asistanı Aç** butonuyla kamera açmadan da tüm komutları kullanabilirsiniz.
- **🛡️ 4 Katmanlı Kazara Tetikleme Koruması (Anti-Spam):**
  1. **Avuç Kapısı (`[K]` / `palm_arm`):** Komut vermeden önce kısa süre *Açık El* göstermeden kilitli kalır.
  2. **Komut Sonrası Nötr Sıfırlama (`strict_neutral_reset`):** Parmak değiştirerek arka arkaya zincirleme komut üretimini keser.
  3. **El Hareket Hızı Filtresi (`max_hand_motion`):** El sallama / hızlı geçiş anlarını `MOVING` olarak bloke eder.
  4. **Yüksek Güven & Odak Profili:** Varsayılan `%78` güven, `6` kararlı kare, `1.25 sn` bekleme ve 3 temel jestli **🛡️ Odak Modu (`focus`)**.
- **🖥️ Çift Ekran & Mini Köşe PiP HUD (`live_desktop_cam.py`):** Parmağınıza göre canlı eylem rehberi (`[G]`), Mini PiP (`[C]`), monitörler arası taşıma (`[W]`) ve Windows sistem tepsisi desteği.
- **🎭 6 Kullanım Profili:** `Medya`, `YouTube`, `Çift Ekran`, `Sunum`, `Tarayıcı`, `Odak (3 Jest)` + Web arayüzünden Özel Jest Kayıt ve Yeniden Eğitim Stüdyosu.

---

## 🖐️ Varsayılan İskelet Jestleri (`Medya` Profili)

| Sınıf | El Pozu | Parmak Durumu | Varsayılan Eylem |
| :---: | :--- | :--- | :--- |
| **0** | ✋ **Açık El** | 5 parmak açık | **Nötr / El Kapısını Aç (Arm)** |
| **1** | ✊ **Yumruk** | 4 ana parmak kapalı | **Oynat / Durdur** |
| **2** | ☝️ **1 Parmak** | İşaret açık | **Sesi Artır (+4%)** |
| **3** | ✌️ **2 Parmak (V)** | İşaret + orta açık | **Sesi Azalt (-4%)** |
| **4** | 🤟 **3 Parmak** | İşaret + orta + yüzük | **Sonraki Parça / İleri** |
| **5** | 🖖 **4 Parmak** | 4 ana parmak açık | **Önceki Parça / Geri** |
| **6** | 👌 **OK / Çimdik** | Baş + işaret birleşik | **Sessize Al (Mute)** |
| **7** | 👍 **Başparmak Yukarı** | Yalnızca başparmak yukarı | **Tam Ekran (`F`)** |

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin
```bash
pip install -r AirTouch/requirements.txt
```
*(İlk çalıştırmada MediaPipe `hand_landmarker.task` otomatik indirilir. Çevrimdışı Türkçe ses tanıma için [Vosk `vosk-model-small-tr-0.3`](https://alphacephei.com/vosk/models) arşivini `AirTouch/models/vosk-model-small-tr-0.3/` klasörüne çıkarabilirsiniz.)*

### 2. Başlatma
- **Streamlit Kontrol Merkezi (Web + Kamerasız Sesli Asistan + Özel Jest Eğitimi):**
  ```bash
  streamlit run AirTouch/app.py
  ```
  *(Windows'ta doğrudan `AirTouch_Baslat.bat` dosyasına çift tıklayabilirsiniz.)*
- **Masaüstü Canlı HUD (Düşük Gecikme, Çift Ekran & Sesli Komut):**
  ```bash
  python AirTouch/live_desktop_cam.py --camera 0
  ```
  *(Windows'ta `AirTouch_Kesintisiz_Canli_Kamera.bat` dosyasına çift tıklayabilirsiniz.)*

**Masaüstü HUD Kısayolları:** `[K]` El Kapısı (Avuç/Direkt/Ses) • `[G]` Parmak Rehberi • `[V]` Sesli Komut • `[C]` Mini PiP • `[W]` Ekran Taşı • `[M]` Profil Değiştir • `[SPACE]` Bekleme • `[Q]` Çıkış.

---

## 🧪 Testler
```bash
pytest AirTouch/tests -v
```
Tüm veri önişleme, CNN/LandmarkNet model, Win32 sistem kontrolcü, port seçici, çift ekran pencere yöneticisi, dinamik YouTube sesli komut ayrıştırıcı, `48 kHz → 16 kHz` AGC ve anti-spam durum makinesi **52/52 pytest testi** ile doğrulanmıştır.
