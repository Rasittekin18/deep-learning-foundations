# 🖐️ AirTouch Projesi: Canlı Takip ve Görev Listesi (TODO)

> **Proje Amacı:** Bilgisayar kamerası karşısında el hareketleri yaparak (Sign Language & CNN) klavyeye dokunmadan YouTube, Spotify ve Windows medya/ses kontrollerini temassız olarak yöneten, Streamlit tabanlı modern bir web dashboard'a sahip masaüstü yapay zeka asistanı.

---

## 📌 Aşama Takip Tablosu

| Aşama | Başlık | Durum | Detay |
| :---: | :--- | :---: | :--- |
| **1** | **Proje İskeleti & Windows Medya Motoru** | ✅ **TAMAMLANDI** | Yerel Windows API (`ctypes`) ile sıfır gecikmeli medya kontrolleri kuruldu ve test edildi. |
| **2** | **Veri Hattı (Dataset & Preprocessing)** | ✅ **TAMAMLANDI** | Kurstaki 2062 el verisi bağlandı, veri artırma ve canlı kamera kare ön işleme pipeline'ı test edildi. |
| **3** | **PyTorch CNN Model Mimarisi & Eğitimi** | ✅ **TAMAMLANDI** | `AirTouchCNN` modeli eğitildi, %98.71 doğrulama doğruluğu (val acc) elde edildi ve ağırlıklar kaydedildi. |
| **4** | **Akıllı Tetikleme Motoru (Debounce & Cooldown)** | ✅ **TAMAMLANDI** | Güven eşiği, bekleme süresi (cooldown) ve hareket kararlılığı (smoothing) filtreleme motoru kodlandı ve test edildi. |
| **5** | **Streamlit Web Dashboard (Modern Arayüz)** | ✅ **TAMAMLANDI** | Canlı kamera önizlemesi, model teşhis kartı, Plotly canlı olasılık grafiği, tuş atama ve metrik sekmeleri kodlandı. |
| **TEST** | **Kapsamlı Mimari ve Birim Testleri (Pytest)** | ✅ **TAMAMLANDI** | 4 ana modülde 24 adet mimari test (gradyan akışı, tensör değişmezleri, debounce, Windows API) %100 başarıyla geçti. |
| **6** | **Canlı Test, Otomatik El Takibi & Akıllı Filtreleme** | ✅ **TAMAMLANDI** | Otomatik El Takibi (Auto-Track), %85 büyütülmüş BBox (340px) ve 4 katmanlı anti-spam mimarisiyle sıfır zahmetli kontrol sağlandı. |

---

## 📝 Detaylı Görev Listesi

### ✅ Aşama 1: Proje İskeleti ve Windows Medya Motoru (Tamamlandı)
- [x] Proje dizin yapısının kurulması (`AirTouch/core/`, `AirTouch/data/`, `AirTouch/models/`)
- [x] Temel yapılandırma dosyasının (`config.py`) oluşturulması
- [x] Yerel Windows API (`ctypes.windll.user32.keybd_event`) medya motorunun yazılması:
  - Play / Pause (Oynat / Durdur)
  - Volume Up / Down (Ses Aç / Kıs)
  - Next / Previous Track (Sonraki / Önceki Şarkı)
  - Mute (Sessize Al)
- [x] Medya motorunun donanım seviyesinde test edilip doğrulanması

---

### ✅ Aşama 2: Veri Hattı ve Canlı Kamera Ön İşleme (Tamamlandı)
- [x] Kurstaki hazır `X.npy` (2062 el resmi, $64 \times 64$) ve `Y.npy` veri setinin projeye bağlanması
- [x] PyTorch `GestureDataset` sınıfının yazılması:
  - Normalizasyon ($[0, 1]$ aralığı) ve tensör dönüşümleri
  - Veri Artırma (Data Augmentation: Rastgele parlaklık, hafif kaydırma/shift)
  - Eğitim / Doğrulama (%85 Train - %15 Val) ayrımı
- [x] Kameradan gelen ham renkli kareleri modelin beklediği $64 \times 64$ gri tonlamalı tensöre dönüştüren ön işleme fonksiyonu (`preprocess_frame`)

---

### ✅ Aşama 3: PyTorch CNN Model Mimarisi ve Eğitimi (Tamamlandı)
- [x] `AirTouchCNN` model sınıfının kodlanması:
  - 3 adet Conv2D Katmanı ($3 \times 3$ filtreler, ReLU aktivasyonu, BatchNorm)
  - MaxPool2D ve Dropout (%20-%25) katmanları
  - Tam Bağlantılı (Fully Connected / Dense) sınıflandırıcı katman
- [x] Model eğitim fonksiyonu (`trainer.py`):
  - CrossEntropyLoss ve Adam optimizer
  - Epoch bazında train/validation loss ve accuracy hesaplama
  - 15 epoch sonunda **%98.71 doğrulama doğruluğu (Validation Accuracy)** elde edildi!
  - En yüksek başarılı ağırlıklar [`models/airtouch_model.pth`](file:///c:/Users/Rasit/OneDrive/Masaüstü/Yazılım/Yapay%20Zeka/1-Udemy/yarım/5-Deep%20Learning%20ve%20Python%20A'dan%20Z'ye%20Derin%20Öğrenme%20ok/AirTouch/models/airtouch_model.pth) olarak kaydedildi.
  - Eğitim geçmişi [`models/training_history.json`](file:///c:/Users/Rasit/OneDrive/Masaüstü/Yazılım/Yapay%20Zeka/1-Udemy/yarım/5-Deep%20Learning%20ve%20Python%20A'dan%20Z'ye%20Derin%20Öğrenme%20ok/AirTouch/models/training_history.json) dosyasına yazıldı.

---

### ✅ Aşama 4: Akıllı Tetikleme Motoru (Tamamlandı)
- [x] Canlı tahmin motorunun (`AirTouch/core/engine.py`) kodlanması:
  - Kameradan gelen karenin modele verilmesi ve Softmax olasılıklarının çıkarılması
  - **Güven Eşiği (Confidence Threshold):** Örneğin %80'in altındaki tahminlerin yok sayılması (Boş ekranda yanlış tetiklemeyi önleme)
  - **Bekleme Süresi (Cooldown Timer):** Eylemler arasına bekleme süresi koyarak sesin aniden fırlamasını engelleme
  - **Kararlılık Penceresi (Smoothing Window):** Titremeyi ve anlık parazitleri önleyen kayan pencere filtresi
  - Test edildi ve doğrulandı!

---

### ✅ Aşama 5: Streamlit Web Dashboard (Tamamlandı)
- [x] Streamlit ana uygulamasının ([`AirTouch/app.py`](file:///c:/Users/Rasit/OneDrive/Masaüstü/Yazılım/Yapay%20Zeka/1-Udemy/yarım/5-Deep%20Learning%20ve%20Python%20A'dan%20Z'ye%20Derin%20Öğrenme%20ok/AirTouch/app.py)) hazırlanması:
  - **Sekme 1: Canlı Kontrol Odası (Live Control):**
    - Kameradan anlık görüntü alma (`st.camera_input`)
    - Modelin ne gördüğü (Ön işlenmiş $64 \times 64$ el karesi önizlemesi)
    - Plotly ile canlı olasılık çubuk grafiği (Hangi hareketten ne kadar emin?)
    - Anlık tetiklenen eylem rozeti ("Ses Açıldı 🔊", "Oynatıldı ⏯️") ve bekleme süresi sayacı
  - **Sekme 2: Tuş Atamaları & Hareketler:**
    - Hangi jestin hangi tuşa basacağını arayüzden dinamik seçme
  - **Sekme 3: Model Eğitimi & Metrikler:**
    - Epoch bazında Loss ve %98.71 Doğruluk eğrilerini interaktif Plotly grafiğiyle inceleme
  - **Sekme 4: Proje ve Kurs Rehberi Bağlantısı**

---

### ✅ Aşama 6: Canlı Test, Otomatik El Takibi & Akıllı Filtreleme (Tamamlandı)
- [x] **🤖 Otomatik El Takibi (Auto-Tracking):** Kullanıcının elini sabit bir kutuya denk getirmeye çalışma zahmeti ortadan kaldırıldı; sistem kaldırılan eli otomatik bulup yeşil kutuyu yumuşakça (EMA filtresiyle) elin üzerine kilitler.
- [x] **📐 Genişletilmiş Bounding Box:** Varsayılan kutu boyutu 250px'ten **340px'e (%85 daha geniş)** çıkarıldı; klavyeden `[1/2/3]` tuşlarıyla anında 280px, 340px veya 420px (devasa) boyutlara geçiş desteği eklendi.
- [x] **🛡️ 4 Katmanlı Akıllı Karar Mimarisi:**
  - El Varlığı Kapısı (YCrCb ten rengi doğrulaması)
  - Güven & Marj Kalitesi ($\ge \%75$ güven ve $\ge \%28$ sınıf ayrımı)
  - Zaman Tabanlı Tutma (Hold-to-Confirm: 5 kararlı kare)
  - One-Shot Kilitlenme (Tekil tetikleme: Oynat/Durdur ve şarkı geçişlerinde 1 kez tetikleyip el indirilene kadar bekler)
- [x] Streamlit Web Dashboard ve Masaüstü 60 FPS başlatıcıların güncellenmesi

---

## 💡 İleride Eklenebilecek Opsiyonel Fikirler (Geliştirmeye Açık)
- [ ] **Masaüstü Bildirimleri:** Eylem tetiklendiğinde Windows sağ alt köşesinde ufak bir bildirim çıkması.
- [ ] **Özel Tuş Kombinasyonu Ekleme:** İsteğe bağlı olarak PowerPoint sunum geçişi (Sağ/Sol Ok) ya da Chrome sekme kapatma (Ctrl+W) jesti ekleme.
- [ ] **Gece Modu / Düşük Işık Desteği:** Kamerada ışık yetersiz olduğunda otomatik histogram eşitleme (histogram equalization) ile görüntüyü parlatma.

