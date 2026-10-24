# İleri Seviye Derin Öğrenme Mimarileri ve Projeleri

Bu depo, ileri seviye **Derin Öğrenme (Deep Learning)** ve **Bilgisayarlı Görü (Computer Vision)** mimarilerini, teorik alt yapıları ve PyTorch ile Keras kütüphaneleri kullanılarak geliştirilmiş uçtan uca uygulamalı projeleri içermektedir.

Eğitim sürecinde geliştirilen modeller; sınıflandırma, boyut indirgeme, sentetik veri üretimi ve transfer öğrenme gibi geniş bir yelpazedeki problemleri çözmek üzere kurgulanmıştır.

---

## 🛠️ Kullanılan Teknolojiler & Kütüphaneler

* **Derin Öğrenme Çerçeveleri:** PyTorch, Keras, TensorFlow
* **Veri İşleme & Görselleştirme:** NumPy, Pandas, OpenCV, PIL (Pillow), Matplotlib
* **Model & Veri İhracı:** JSON, Codecs, H5

---

## 📚 Repo İçeriği ve Mimariler

### 1. Convolutional Neural Networks (CNN)
* **Teorik Altyapı:** Konvolüsyon katmanları, aktivasyon fonksiyonları (ReLU, Softmax), Pooling (Max Pooling), Dropout ile regülasyon ve Fully Connected katman yapısı.
* **Geliştirilen Projeler:**
  * **Fruit360 Dataset:** Keras kullanılarak görüntü sınıflandırma; `ImageDataGenerator` ile veri artırma (Data Augmentation - Shear, Zoom, Horizontal Flip) ve normalizasyon.
  * **MNIST Dataset:** Keras ile el yazısı rakam sınıflandırma projesi ve `to_categorical` ile One-Hot Encoding uygulaması.

### 2. Deep Residual Networks (ResNets)
* **Teorik Altyapı:** Derin ağlarda yaşanan **Vanishing Gradient (Kaybolan Gradyan)** problemi ve bu sorunu aşmak için kullanılan **Skip Connections / Shortcuts** (Kestirme Bağlantılar) mekanizması.
* **Geliştirilen Proje:**
  * **IR Pedestrian Dataset:** Kızılötesi (IR) görüntüler üzerinden yaya tespiti. PyTorch kullanılarak nesne yönelimli mimariyle sıfırdan **Custom CNN** ve **Basic Block** içeren modüler **ResNet** mimarisi tasarlanmış, Custom Loss & Optimizer konfigürasyonlarıyla eğitilmiştir.

### 3. Transfer Learning (Transfer Öğrenme)
* **Teorik Altyapı:** ImageNet gibi devasa veri setlerinde önceden eğitilmiş (pre-trained) ağırlıkların kullanımı, Fine-Tuning ve Feature Extraction yaklaşımları.
* **Geliştirilen Projeler:**
  * **Fruit360 Dataset:** Pre-trained **VGG16** mimarisi üzerine özel katmanlar eklenerek sınıflandırma.
  * **CIFAR-10 Dataset:** Pre-trained **VGG19** mimarisi kullanımı; OpenCV kütüphanesi ile düşük çözünürlüklü görsellerin (32x32) model giriş boyutuna (48x48) yeniden boyutlandırılması (Resizing) ve sınıflandırılması.

### 4. Autoencoders (Oto-Kodlayıcılar)
* **Teorik Altyapı:** Gözetimsiz öğrenme (Unsupervised Learning) yaklaşımı, Encoder-Decoder yapısı, Latent Space Representation (gizil alan gösterimi) ve boyut indirgeme.
* **Geliştirilen Proje:**
  * **Fashion MNIST Dataset:** Keras ile giysi görsellerinin öznitelik çıkarımı (Feature Extraction) ve görsel rekonstrüksiyonu (Image Reconstruction).

### 5. Generative Adversarial Networks (GANs)
* **Teorik Altyapı:** Çekişmeli Eğitim (Adversarial Training) mantığı; üreteç (**Generator**) ve ayırt edici (**Discriminator**) ağların eş zamanlı eğitimi.
* **Geliştirilen Proje:**
  * **MNIST Dataset:** Keras kullanılarak rastgele gürültüden (Random Noise) gerçekçi sentetik el yazısı rakamların üretilmesi.

---

## 📂 Proje Yapısı

```bash
.
├── 1- Convolutional Neural Networks (CNN)/
├── 2- Deep Residual Networks (ResNets)/
├── 3- Transfer Learning/
├── 4- Autoencoders/
└── 5- Generative Adversarial Networks (GANs)/
```
