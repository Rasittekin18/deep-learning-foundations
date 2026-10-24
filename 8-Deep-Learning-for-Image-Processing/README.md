# Deep Learning for Image Processing 🚀📸

Bu depo, **OpenCV**, **Keras / TensorFlow** ve **Python** kullanılarak gerçekleştirilen geleneksel nesne tespiti, nesne takibi (object tracking) ve derin öğrenme tabanlı evrişimsel sinir ağları (CNN) uygulamalarını, proje kodlarını ve teknik analizleri içermektedir.

Geleneksel bilgisayarla görme (Computer Vision) yöntemlerinden modern derin öğrenme mimarilerine (ResNet, R-CNN, YOLO, SSD) uzanan geniş bir yelpazede nesne algılama ve takip algoritmaları uygulamalı olarak ele alınmıştır.

---

## 📌 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Öğrenilen ve Uygulanan Konular](#-öğrenilen-ve-uygulanan-konular)
  - [1. OpenCV ile Nesne Tespiti (Geleneksel Yöntemler)](#1-opencv-ile-nesne-tespiti-geleneksel-yöntemler)
  - [2. OpenCV ile Nesne Takibi (Object Tracking)](#2-opencv-ile-nesne-takibi-object-tracking)
  - [3. Evrişimsel Sinir Ağları (CNN) & Projeler](#3-evrişimsel-sinir-ağları-cnn--projeler)
  - [4. Derin Öğrenme Tabanlı Nesne Tespiti](#4-derin-öğrenme-tabanlı-nesne-tespiti)
- [Öne Çıkan Projeler](#-öne-çıkan-projeler)
- [Kullanılan Teknolojiler ve Kütüphaneler](#-kullanılan-teknolojiler-ve-kütüphaneler)
- [Gereksinimler ve Kurulum](#-gereksinimler-ve-kurulum)
- [İletişim ve Yazar](#-iletişim-ve-yazar)

---

## 💡 Proje Hakkında

Bilgisayarla Görme (Computer Vision) disiplininde nesneleri tanımlama, konumlandırma ve zaman içerisinde takip etme süreçleri kritik rol oynar. Bu projede:
* **Geleneksel Yöntemler:** Kenar, köşe, kontur algılama, HSV renk filtresi, Haar Cascade ve HOG (Histogram of Oriented Gradients) gibi yöntemler incelenmiştir.
* **Nesne Takibi:** Meanshift, MIL, KCF, CSRT, GOTURN, MOSSE ve MedianFlow algoritmalarının veri setleri üzerindeki performansları kıyaslanmıştır.
* **CNN & Derin Öğrenme:** Sıfırdan CNN mimarileri tasarlanmış, T-Rex otomat oyunu ve gerçek zamanlı rakam sınıflandırma projeleri geliştirilmiştir.
* **İleri Seviye Nesne Tespiti:** Image Pyramid, Sliding Window, Non-Maximum Suppression (NMS), Selective Search, R-CNN, YOLO ve SSD mimarileri uygulamalı olarak kodlanmıştır.

---

## 📚 Öğrenilen ve Uygulanan Konular

### 1. OpenCV ile Nesne Tespiti (Geleneksel Yöntemler)
* **Kenar & Köşe Tespiti:** Canny Edge Detector, Harris Corner Detection ve Shi-Tomasi (`goodFeaturesToTrack`) yöntemleri.
* **Kontur Analizi & Renk ile Nesne Tespiti:** HSV renk uzayında maskeleme, erozyon/genişleme ve kontur tespiti ile nesne konumlandırma.
* **Şablon & Özellik Eşleştirme:** `cv2.matchTemplate`, SIFT ve ORB tanımlayıcıları ile Brute-Force Matcher pratikleri.
* **Havza (Watershed) Algoritması:** Dokunan ve çakışan nesnelerin (bozuk paralar) segmentasyonu ve ayrıştırılması.
* **Haar Cascade Sınıflandırıcıları:** Özel Cascade eğitimi (Positive/Negative dataset hazırlama), yüz, kedi yüzü ve yaya tespiti.

### 2. OpenCV ile Nesne Takibi (Object Tracking)
* **Meanshift / Camshift:** Piksel yoğunluk moduna dayalı nesne takibi ve Back-Projection uygulaması.
* **Çoklu Takip Algoritmaları (MOT):** Boosting, MIL, KCF, TLD, MedianFlow, MOSSE ve CSRT algoritmalarının MOT17 veri seti üzerindeki Öklid mesafesi ve doğruluk analizleri.
* **Çoklu Nesne Takibi (Multi-Object Tracking):** `cv2.MultiTracker` yapısı ile eş zamanlı çoklu yaya ve araç takibi.

### 3. Evrişimsel Sinir Ağları (CNN) & Projeler
* **CNN Mimarisi:** Convolution, ReLU, Max Pooling, Flatten, Dropout ve Dense katmanlarının teorisi.
* **T-Rex Oyunu Otomasyonu:**
  * MSS kütüphanesi ile anlık ekran görüntüsü alma ve veri seti oluşturma.
  * Keras ile model eğitimi ve modelin oyunu oynaması (Auto-Pilot).
* **Gerçek Zamanlı Rakam Sınıflandırma:**
  * `MyData` rakam veri seti ile CNN modeli eğitimi.
  * Kameradan alınan canlı görüntüde yazılan rakamların anlık sınıflandırılması.

### 4. Derin Öğrenme Tabanlı Nesne Tespiti
* **Image Pyramid & Sliding Window:** Görsel ölçekleme ve pencere kaydırma mekanizması.
* **Non-Maximum Suppression (NMS):** Çakışan Bounding Box'ların elenmesi ve IoU (Intersection over Union) hesabı.
* **Selective Search & Region Proposals:** Bölge önerisi tabanlı tespit stratejileri.
* **R-CNN, YOLO ve SSD:** Bölge tabanlı ve tek aşamalı (Single Shot) derin öğrenme detektörlerinin mimari karşılaştırması.

---

## 🏆 Öne Çıkan Projeler

| Proje Adı | Açıklama | Kullanılan Teknolojiler |
| :--- | :--- | :--- |
| **T-Rex Auto-Player** | Tarayıcı üzerindeki T-Rex oyununu ekran görüntüsü alıp CNN ile canlı oynayan yapay zeka ajanı. | Keras, MSS, Keyboard, OpenCV |
| **Real-Time Digit Classifier** | Kamera akışındaki el yazısı rakamları anlık tespit edip sınıflandıran derin öğrenme modeli. | Keras, OpenCV, NumPy |
| **Custom Cascade Object Detection** | Kendi özel veri setiyle eğitilmiş Cascade Classifier ile nesne algılama. | OpenCV, Cascade Trainer GUI |
| **MOT17 Vehicle & Pedestrian Tracker** | MOT17 Benchmark verisi üzerinde çoklu nesne takip algoritmalarının performans testi. | OpenCV MultiTracker, Pandas, Matplotlib |

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

- **Programlama Dili:** Python 3.x
- **Görüntü İşleme & Bilgisayarlı Görme:** OpenCV (`opencv-python`, `opencv-contrib-python`)
- **Derin Öğrenme Frameworkleri:** TensorFlow / Keras
- **Veri Manipülasyonu & Analiz:** NumPy, Pandas
- **Görselleştirme:** Matplotlib, Seaborn
- **Sistem & Ekran Yakalama:** MSS, Keyboard, Pickle

---

## ⚙️ Gereksinimler ve Kurulum

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/Rasittekin18/8-Deep-Learning-for-Image-Processing.git](https://github.com/Rasittekin18/8-Deep-Learning-for-Image-Processing.git)
cd 8-Deep-Learning-for-Image-Processing