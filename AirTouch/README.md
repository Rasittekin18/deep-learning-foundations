# 🖐️ AirTouch: Temassız Medya Kontrol Asistanı (Touchless Media Controller)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-60%20FPS%20Realtime-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/pytest-25%20passed%20(100%25)-brightgreen)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-%2598.71-success)

<br/>

**AirTouch**, web kamerası veya telefon kamerası (IP Webcam) akışı üzerinden el hareketlerini (Sign Language & Custom CNN) sıfır gecikmeyle tanıyan, YouTube, Spotify ve Windows sistem ses/medya kontrollerini temassız olarak yönetmenizi sağlayan yeni nesil bir yapay zeka masaüstü kontrol merkezidir.

[Özellikler](#-öne-çıkan-özellikler) • [Mimari](#-4-katmanlı-akıllı-karar-mimarisi) • [Jest Tablosu](#-el-hareketleri-ve-medya-eşleşmeleri) • [Hızlı Başlangıç](#-kurulum-ve-çalıştırma) • [Testler](#-yazılım-mimarisi-ve-testler)

</div>

---

## 🌟 Öne Çıkan Özellikler

- **🤖 Otomatik El Takibi (Auto-Hand Tracking):** Elinizi ekrandaki sabit bir kutuya denk getirmeye çalışma zahmeti yoktur. Kaldırılan el kare içinde otomatik olarak bulunur ve takip kutusu akıcı (EMA filtreli) olarak elinize kilitlenir.
- **🛡️ 4 Katmanlı Akıllı Karar Mimarisi (Anti-Spam):** Boş odada, duvarda veya kaşınma anlarında yanlış tetiklemeyi %100 engelleyen el varlık kapısı ve tutma doğrulaması.
- **🔒 Tekil Kilitlenme (One-Shot Latch):** Oynat/Durdur ve şarkı geçişlerinde tek bir jest için yalnızca 1 kez tuşa basar; el indirilene kadar tekrar basmaz (makineli tüfek etkisini önler).
- **📱 Telefon IP Kamerası & Web Kamerası Desteği:** İster dahili laptop kamerasını, ister Android/iOS telefonunuzu kablosuz yüksek çözünürlüklü (1080p -> 960x540 normalize) 60 FPS kamera olarak kullanın.
- **🎮 Çift Arayüz Modu:** 
  - **Masaüstü 60 FPS Modu (`live_desktop_cam.py`):** Sıfır gecikmeli, tam ekran/pencere OpenCV HUD deneyimi.
  - **Modern Web Dashboard (`app.py`):** Streamlit + Plotly ile canlı model teşhisi, olasılık dağılımı ve tuş atama paneli.
- **⭐ %98.71 Doğrulama Doğruluğu:** 2062 işaret dili verisiyle eğitilmiş, Batch Normalization ve Dropout regülarizasyonlu özel `AirTouchCNN` mimarisi.

---

## 🧠 4 Katmanlı Akıllı Karar Mimarisi

```
               [Kamera Karesi]
                      │
                      ▼
┌───────────────────────────────────────────┐
│  1. Katman: El Varlığı Kapısı (YCrCb)     │ ──(Kutuda El Yoksa)──> [0 Komut / Sessiz]
│  (Chai & Ngan Universal Skin Color Model) │
└───────────────────────────────────────────┘
                      │ (Ten Oranı >= %10)
                      ▼
┌───────────────────────────────────────────┐
│  2. Katman: Güven & Marj Doğrulaması      │ ──(Model Kararsızsa)─> [Yoksay / Bekle]
│  (Top-1 >= %75 ve Top-1 - Top-2 >= %28)   │
└───────────────────────────────────────────┘
                      │ (Yüksek Kaliteli Tahmin)
                      ▼
┌───────────────────────────────────────────┐
│  3. Katman: Hold-to-Confirm Sayacı        │ ──(Anlık Geçiş)──────> [Sayaç Sıfırlanır]
│  (5 Ardışık Kare Boyunca Sabit Tutma)     │
└───────────────────────────────────────────┘
                      │ (Doğrulandı: %100)
                      ▼
┌───────────────────────────────────────────────────────────┐
│  4. Katman: One-Shot Kilitlenme (State Latch)             │
│  • Oynat/Durdur, Sonraki, Önceki, Mute: SADECE 1 KEZ BAS! │
│  • Ses Aç/Kıs: 1'er saniye aralıklarla adımlı kontrol     │
└───────────────────────────────────────────────────────────┘
```

---

## 🖐️ El Hareketleri ve Medya Eşleşmeleri

| Jest No | El Hareketi | Tetiklenen Windows Eylemi | Açıklama |
| :---: | :--- | :--- | :--- |
| **0** | **Boşta / Nötr** | *İşlem Yok* | Kutu boşken veya el indirilmişken. |
| **1** | **Yumruk Yapmak** | **Oynat / Durdur (Play/Pause) ⏯️** | YouTube/Spotify'da videoyu/müziği başlatıp durdurur. |
| **2** | **İşaret Parmağı Yukarı** | **Sesi Artır (+4%) 🔊** | 1 saniye aralıklarla Windows ana sesini açar. |
| **3** | **İki Parmak (V / Barış)** | **Sesi Azalt (-4%) 🔉** | 1 saniye aralıklarla Windows ana sesini kısar. |
| **4** | **3 Parmak Göstermek** | **Sonraki Parça ⏭️** | Çalma listesindeki sonraki şarkıya/videoya geçer. |
| **5** | **Açık El Göstermek** | **Sessize Al (Mute) 🔇** | Sesi anında tamamen kapatır veya açar. |

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Çalıştırma Yöntemleri

#### Seçenek A: Kesintisiz Canlı Masaüstü Takibi (60 FPS — Tavsiye Edilen)
```bash
# Web Kamerası için:
python live_desktop_cam.py 0

# Telefon Kamerası (IP Webcam) için:
python live_desktop_cam.py "http://192.168.1.15:8080/video"
```
*(Windows kullanıcıları doğrudan `AirTouch_Telefon_Kamerasi.bat` dosyasına çift tıklayabilir).*

**Masaüstü Canlı Kısayolları:**
- `[SPACE]` : Otomatik El Takibini Aç / Kapat
- `[1]`, `[2]`, `[3]` : Kutu Boyutu (1: 280px, 2: 340px Geniş, 3: 420px Devasa)
- `[+]` / `[-]` : Boyutu serbestçe büyüt / küçült
- `[W]`, `[A]`, `[S]`, `[D]` : Manuel modda kutuyu taşı
- `[T]` : Güven eşiğini değiştir (%70 / %75 / %80)
- `[Q]` veya `[ESC]` : Çıkış

#### Seçenek B: Streamlit Web Dashboard
```bash
streamlit run app.py
```
*(Tarayıcınızda otomatik olarak `http://localhost:8501` açılır).*

---

## 🧪 Yazılım Mimarisi ve Testler

AirTouch; nesne yönelimli, modüler ve sağlam bir yazılım mimarisi üzerine inşa edilmiştir:
- `core/system_controller.py`: Doğrudan Windows Win32 API (`ctypes.windll.user32.keybd_event`) ile sıfır gecikmeli sanal donanım tuş basımı.
- `core/model.py`: 3 Bloklu PyTorch Conv2D + BatchNorm + MaxPool + Dropout mimarisi.
- `core/dataset.py`: Dinamik veri artırma ve adaptif kontrast gerdirme ön işleme hattı.
- `core/engine.py`: Çıkarım, yumuşatma ve HUD overlay motoru.

Tüm bu mimari bileşenler, **25 adet kapsamlı pytest mimari testiyle** doğrulanmıştır:
```bash
pytest tests/ -v
```
```text
tests/test_dataset.py::TestDatasetAndPreprocessingArchitecture PASSED [ 32%]
tests/test_engine.py::TestEngineArchitectureAndDebounceState   PASSED [ 60%]
tests/test_model.py::TestModelArchitectureAndGradientFlow     PASSED [ 80%]
tests/test_system_controller.py::TestSystemControllerArchitecture PASSED [100%]

============================= 25 passed in 3.06s =============================
```

---

## 👨‍💻 Geliştirici

**Raşit Tekin**  
*Deep Learning & Computer Vision Projesi*  
- GitHub: [@Rasittekin18](https://github.com/Rasittekin18)
