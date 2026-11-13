# 🖐️ AirTouch v2.1 Pro — Kullanım Kılavuzu (El İskeleti + Sesli Komut + Çift Ekran)

**AirTouch Pro**, bilgisayarınızın varsayılan laptop kamerası (`index 0`) ve mikrofonu üzerinden çalışan; **MediaPipe 21 Nokta El İskeleti + PyTorch `AirTouchLandmarkNet`** ile **Vosk Çevrimdışı Türkçe Sesli Komut** motorunu birleştiren profesyonel masaüstü kontrol asistanıdır.

---

## 🚀 1. Uygulamayı Başlatma Yolları

### Yol A: Masaüstü Canlı HUD (YouTube & Çift Ekran İçin Önerilen)
- **`AirTouch_Kesintisiz_Canli_Kamera.bat`** dosyasına çift tıklayın.
- Sağ kenarda açık olan **"Hangi Parmak Ne Yapar?"** rehberi aktif profilinize göre canlı güncellenir ve yaptığınız hareketi o anda vurgular.
- **Pencere Kısayolları:**
  - `[K]`: **El Güvenlik Kapısı** (**Avuçla Uyandır** → **Doğrudan + Kilit** → **Sadece Ses**)
  - `[G]`: Sağ kenar **Parmak Rehberi** panelini Aç / Kapat
  - `[V]`: **Sesli Komut** dinlemeyi Aç / Kapat
  - `[C]`: **Mini PiP Köşe Modu** (`420×255`, her zaman üstte) ↔ Tam Görünüm
  - `[W]`: AirTouch HUD penceresini **Ekran 1 ↔ Ekran 2** arasında taşı
  - `[M]`: Profil değiştir (**Medya** → **YouTube** → **Çift Ekran** → **Sunum** → **Tarayıcı** → **Odak**)
  - `[X]`: Aktif pencereye göre **Otomatik Bağlam Profili** Aç / Kapat
  - `[Z]`: Pencereyi Sistem Tepsisine (sağ alt saat yanı) Gizle / Göster
  - `[A]`: Pencereyi Her Zaman Üstte Tut
  - `[SPACE]`: El takibini Bekleme Moduna al / Devam et
  - `[Q]` veya `[ESC]`: Çıkış

### Yol B: Streamlit Kontrol & Eğitim Merkezi
- **`AirTouch_Baslat.bat`** dosyasına çift tıklayın (8501 doluysa otomatik bir sonraki boş portu seçer ve konsolda adresi yazar).
- Canlı Odada kamerasız kontrol için **"🎙️ Sadece Sesli Asistanı Aç"** veya masaüstü penceresi için **"🖥️ MASAÜSTÜ HUD'U BAŞLAT"** butonlarını kullanabilirsiniz.

> 🛡️ **Kazara & Arka Arkaya El Tetikleme Koruması:** Varsayılan **Avuç Kapısı (`[K]`)** modunda sıradan el hareketleri yoksayılır; komut vermek için kısaca **Açık El (5 Parmak)** gösterdikten sonra istediğiniz jesti tutarsınız. Hareket eden el (`MOVING`) tetiklemez ve tek komut sonrası el Açık El'e dönene kadar ikinci bir işlem yapılmaz.

---

## 🖐️ 2. Hangi Parmak Ne Yapar? (6 Hazır Profil)

| Jest | Parmak Şeması | 🎬 Medya | ▶️ YouTube | 🖥️ Çift Ekran | 📊 Sunum | 🌐 Tarayıcı | 🛡️ Odak (3 Jest) |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **✋** | 5 Parmak Açık | *Hazır / Uyandır* | *Hazır / Uyandır* | *Hazır / Uyandır* | *Hazır / Uyandır* | *Hazır / Uyandır* | *Hazır / Uyandır* |
| **✊** | Yumruk (0 Parmak) | Oynat / Durdur | Oynat / Durdur | Uygulama Değiştir | Boşluk (Space) | Boşluk (Space) | Oynat / Durdur |
| **☝️** | 1 Parmak (İşaret) | Ses Artır (+4%) | Ses Artır (+4%) | Diğer Ekrana Geç | Sonraki Slayt | Yukarı Kaydır | Ses Artır (+4%) |
| **✌️** | 2 Parmak (Zafer) | Ses Kıs (-4%) | Ses Kıs (-4%) | Pencereyi Taşı | Önceki Slayt | Aşağı Kaydır | Ses Kıs (-4%) |
| **🤟** | 3 Parmak | Sonraki Parça | YT +10 sn (L) | Sonraki Sekme | Aşağı Kaydır | Sonraki Sekme | *Pasif* |
| **🖖** | 4 Parmak | Önceki Parça | YT -10 sn (J) | Önceki Sekme | Yukarı Kaydır | Önceki Sekme | *Pasif* |
| **👌** | OK / Çimdik | Sessize Al / Aç | Sessize Al / Aç | Sessize Al / Aç | ESC (Çıkış) | Sessize Al / Aç | *Pasif* |
| **👍** | Başparmak Yukarı | Tam Ekran (F) | Tam Ekran (F) | YouTube'a Odaklan | Tam Ekran (F) | Tam Ekran (F) | *Pasif* |

> 💡 **Odak Koruması:** YouTube/PowerPoint/tarayıcı kısayolları (`F`, `J`, `L`, `Ctrl+Tab`, `Win+Shift+Sağ`) çalıştırılırken AirTouch odaktaysa komut doğrudan son aktif kullanıcı penceresine yönlendirilir.

---

## 🎤 3. Çevrimdışı Türkçe Sesli Komutlar (Vosk)

İnternet gerektirmez; Windows'ta `Stereo Karışımı` seçili olsa bile otomatik olarak gerçek mikrofonunuzu (`Mikrofon Dizisi`) seçer.

- **Doğrudan YouTube Video Açma:** `"youtube'da [şarkı / video adı] aç"`, `"youtube tarkan kuzum açılsın"`, `"youtube python dersleri aç"`, `"barış manço gülpembe şarkısını çal"`
- **YouTube & Medya:** `"video dursun"`, `"video başlasın"`, `"oynat"`, `"durdur"`, `"sesi artır"`, `"sesi azalt"`, `"sesi kapat"`, `"tam ekran"`, `"ekrandan çık"`, `"ileri sar"`, `"geri sar"`, `"sonraki video"`, `"önceki video"`, `"altyazı aç"`
- **Çift Ekran & Uygulama Geçişi:** `"diğer ekran"`, `"ekran değiştir"`, `"ekrana taşı"`, `"uygulama değiştir"`, `"pencere değiştir"`, `"sonraki sekme"`, `"önceki sekme"`, `"youtube geç"`, `"spotify geç"`, `"sola yasla"`, `"sağa yasla"`
- **AirTouch Mod Kontrolü:** `"odak modu"`, `"youtube modu"`, `"çift ekran modu"`, `"medya modu"`, `"sunum modu"`, `"mini mod"`, `"büyük mod"`, `"bekleme modu"`, `"devam et"`, `"rehber"`, `"kontrolü diğer ekrana al"`
