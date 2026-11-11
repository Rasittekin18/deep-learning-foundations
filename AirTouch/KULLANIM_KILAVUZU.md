# 🖐️ AirTouch: Hızlı Kullanım ve Canlı Test Kılavuzu

Tebrikler! **"Deep Learning ve Python: A'dan Z'ye Derin Öğrenme"** kursunda öğrendiğiniz teorik ve pratik bilgileri, günlük hayatınızda bilgisayarınızı temassız yönetebileceğiniz, **%98.71 doğruluk oranına sahip** ve **24 yazılım mimari testinden geçmiş** uçtan uca bir ürüne dönüştürdük.

---

## 🚀 1. Uygulamayı Başlatma Yolları

### Yol A: Kesintisiz Canlı Masaüstü Takibi (60 FPS — Tavsiye Edilen)
Telefon kameranızla kesintisiz, 60 FPS canlı jest algılama için:
👉 **[`AirTouch_Telefon_Kamerasi.bat`](file:///c:/Users/Rasit/OneDrive/Masaüstü/Yazılım/Yapay%20Zeka/1-Udemy/yarım/5-Deep%20Learning%20ve%20Python%20A'dan%20Z'ye%20Derin%20Öğrenme%20ok/AirTouch_Telefon_Kamerasi.bat)** dosyasına çift tıklayın.
- **Pencere Boyutu:** Standart ideal 960x540 HD pencerede açılır (ekranı kaplamaz).
- **Hedef El Kutusu (BBox) Kontrolleri:**
  - `[W]`, `[A]`, `[S]`, `[D]`: Yeşil hedef kutusunu yukarı / sola / aşağı / sağa taşır.
  - `[+]`, `[-]`: Kutuyu büyütür veya küçültür.
  - `[C]`: Kutuyu tam merkeze sıfırlar.
  - `[Q]` veya `[ESC]`: Pencereyi kapatır.

### Yol B: Streamlit Web Dashboard (Tarayıcı Arayüzü)
👉 **[`AirTouch_Baslat.bat`](file:///c:/Users/Rasit/OneDrive/Masaüstü/Yazılım/Yapay%20Zeka/1-Udemy/yarım/5-Deep%20Learning%20ve%20Python%20A'dan%20Z'ye%20Derin%20Öğrenme%20ok/AirTouch_Baslat.bat)** dosyasına çift tıklayın veya tarayıcınızda:
👉 **`http://localhost:8501`** adresini açın.
- **Kamera Kaynağı Seçimi:**
  - **📱 Telefon IP Kamerası:** Telefondaki canlı MJPEG video akışını tarayıcıda doğrudan gösterir. *"Telefondan Kare Al & Analiz Et"* butonu veya *"Otomatik Döngü"* ile jestleri web üzerinde test edebilirsiniz.
  - **💻 Laptop Web Kamerası:** Laptop kamerasından fotoğraf çekerek tek kare analiz yapar.

---

## 📱 Telefon Kamerasını Bağlama Adımları (IP Webcam)
1. Telefonunuzun Android Google Play mağazasından ücretsiz **"IP Webcam"** (Pavel Khlebovich) uygulamasını indirin.
2. Telefonunuz ile bilgisayarınızın **aynı Wi-Fi ağına** bağlı olduğundan emin olun.
3. Uygulamayı açıp en alta inin ve **"Start server"** butonuna dokunun.
4. Telefon ekranının altında çıkan adresi kontrol edin (Örn: `http://192.168.1.15:8080`).
5. Bu adresi AirTouch arayüzüne veya `.bat` dosyasına girmeniz yeterlidir!

---

## 🖐️ 2. El Hareketleri (Jest) ve Medya Eylemleri

| Jest No | El Hareketi | Tetiklenen Eylem | Kullanım Senaryosu |
| :---: | :--- | :--- | :--- |
| **0** | **Boşta / Nötr** | *İşlem Yok* | Elinizi kaldırmadığınız veya beklediğiniz anlar. |
| **1** | **Yumruk Yapmak** | **Oynat / Durdur (Play/Pause) ⏯️** | YouTube veya Spotify'da videoyu/müziği durdurup başlatmak. |
| **2** | **İşaret Parmağı Yukarı** | **Sesi Artır (+4%) 🔊** | Windows ana sesini açmak. |
| **3** | **Açık El Göstermek** | **Sesi Azalt (-4%) 🔉** | Windows ana sesini kısmak. |
| **4** | **Sağa İşaret Etmek** | **Sonraki Parça ⏭️** | Çalma listesindeki sonraki şarkıya veya videoya geçmek. |

> 💡 **İpucu:** Bu atamaları web arayüzündeki **"Tuş Atamaları & Hareketler"** sekmesinden istediğiniz zaman değiştirebilirsiniz.

---

## 🧪 3. Canlı Test Senaryosu (Adım Adım)

1. Arka planda bir **YouTube videosu**, **Spotify** veya herhangi bir film/müzik açın.
2. Tarayıcınızda açılan AirTouch web sayfasında sol üstteki **"📷 Web Kamerası"** kutusundan kamerayı başlatın.
3. Elinizi kameranın ortasına gösterip bir jest yapın (örneğin: **Yumruk** veya **İşaret Parmağı**).
4. Ekranda anlık olarak:
   - Modelin hareketinizi tanıdığını (%95+ güvenle),
   - Yeşil rozetle *"Ses Artırıldı 🔊"* veya *"Oynat/Durdur ⏯️"* yazdığını,
   - Bilgisayarınızın sesinin açıldığını ya da videonun anında durup başladığını göreceksiniz!
5. Elinizi kameranın önünde tutmaya devam ettiğinizde **1 saniyelik mola sayacının (cooldown)** devreye girdiğini ve sesin aniden %100'e fırlamasının engellendiğini gözlemleyin.

---

## ⚙️ 4. İnce Ayarlar (Dashboard Üzerinden)
- **Güven Eşiği (Threshold):** Varsayılan olarak **%80**'dir. Odanız loşsa veya hareketleri daha kolay algılamasını isterseniz sol panelden **%70**'e çekebilirsiniz.
- **Bekleme Süresi (Cooldown):** Bir hareket yapıldıktan sonra ikinci harekete izin verme süresidir (Varsayılan: **1.0 saniye**).
- **Canlı Tuş Tetikleme:** Arayüzün solundaki toggle butonunu kapatarak tuşlara basmadan sadece modelin tahminlerini test edebilirsiniz.

