# 🧠 Deep Learning Foundations & Architectures

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-teal)

<br/>

**A comprehensive Deep Learning monorepo covering Neural Network Foundations (ANN, CNN, RNN), Advanced Architectures (ResNet, GANs, AutoEncoders, Transfer Learning), and Custom MNIST Vision Models.**  
Developed by **[Raşit Tekin](https://github.com/Rasittekin18)**.

[Modüller Tablosu](#-modüller-ve-koleksiyonlar) • [Kurulum & Çalıştırma](#-kurulum-ve-kullanım) • [Teknolojiler](#-kullanılan-teknolojiler) • [İletişim](#-yazar)

</div>

---

## 📌 Genel Bakış (Overview)

Bu depo; yapay sinir ağlarının matematiksel ve algoritmik temellerinden başlayarak modern derin öğrenme mimarilerine uzanan kapsamlı bir kaynaktır:
- **Temel Mimariler:** Yapay Sinir Ağları (ANN), Evrişimli Sinir Ağları (CNN), Tekrarlayan Sinir Ağları (RNN) ve Zaman Serisi Analizi.
- **İleri Düzey Mimariler:** Derin Artık Ağlar (ResNet), Önceden Eğitilmiş Modeller ile Transfer Learning, Oto-kodlayıcılar (AutoEncoders) ve Çekişmeli Üretici Ağlar (GANs).
- **Görüntü İşleme & Nesne Tespiti:** CNN tabanlı nesne algılama ve gerçek zamanlı nesne takibi.
- **Özel Vision Modelleri:** Gürültülü MNIST görüntülerini temizleyen evrişimli oto-kodlayıcı (Denoising AutoEncoder) ve çoklu çıktılı rakam farkı kestirim ağı.

---

## 📂 Modüller ve Koleksiyonlar

| # | Modül / Proje | Odak Alanı & İçerik | Mimari & Yöntemler | Klasör Linki |
| :-: | :--- | :--- | :--- | :-: |
| 01 | **Deep Learning Fundamentals** | Lojistik Regresyonun ANN olarak modellenmesi, çok katmanlı algılayıcılar (MLP), Evrişimli Sinir Ağları (CNN) ve borsa/yolcu verileriyle Tekrarlayan Sinir Ağları (RNN / LSTM). | ANN, CNN, RNN, Keras, TensorFlow | [`/5-Deep_Learning`](./5-Deep_Learning) |
| 02 | **Advanced Deep Learning** | İleri CNN teknikleri, artık bağlantılar (Residual Connections) ile ResNet, VGG/ResNet ile Transfer Learning, latent uzay temsili için AutoEncoders ve sentetik veri üretimi için GANs. | ResNet, GANs, AutoEncoders, Transfer Learning | [`/5-Deep_LearningAdvanced`](./5-Deep_LearningAdvanced) |
| 03 | **Deep Learning for Image Processing** | OpenCV ile nesne tespiti ve takibi, CNN tabanlı özellik çıkarımı ve nesne sınıflandırma modelleri. | CNN, Object Tracking, OpenCV, PyTorch | [`/8-Deep-Learning-for-Image-Processing`](./8-Deep-Learning-for-Image-Processing) |
| 04 | **MNIST Trainable Denoiser** | Gaussian veya rastgele gürültü eklenmiş el yazısı rakam görüntülerini temizleyen ve orijinal haline yeniden yapılandıran Evrişimli Oto-kodlayıcı (Convolutional Denoising AutoEncoder). | AutoEncoder, Encoder-Decoder, MSE Loss | [`/MNIST-Trainable-Denoiser`](./MNIST-Trainable-Denoiser) |
| 05 | **MNIST Digit Difference Predictor** | İki el yazısı rakamı girdi alarak hem her iki rakamı aynı anda tanıyan hem de aralarındaki mutlak matematiksel farkı tahmin eden çok görevli (multi-task) sinir ağı. | Multi-output Neural Network, Classification & Regression | [`/mnist-digit-diff`](./mnist-digit-diff) |

---

## 🚀 Kurulum ve Kullanım

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/Rasittekin18/deep-learning-foundations.git
cd deep-learning-foundations
```

### 2. Sanal Ortam Oluşturun
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükleyin
```bash
pip install tensorflow torch torchvision keras numpy pandas matplotlib opencv-python jupyter
```

### 4. Jupyter Notebook'ları Başlatın
```bash
jupyter notebook
```
İlgili klasöre giderek eğitim ve test adımlarını interaktif olarak yürütebilirsiniz.

---

## 🛠️ Kullanılan Teknolojiler

- **Derin Öğrenme Çatıları:** TensorFlow, Keras, PyTorch, Torchvision
- **Mimariler:** ANN, CNN, RNN, LSTM, ResNet, AutoEncoders, GANs
- **Görüntü & Sayısal İşleme:** OpenCV, NumPy, SciPy
- **Görselleştirme:** Matplotlib, Seaborn
- **Değerlendirme:** Karmaşıklık matrisleri, Loss/Accuracy eğrileri, PSNR

---

## 👨‍💻 Yazar

**Raşit Tekin**  
*Electronics Engineer \| AI & Data Science*  
- GitHub: [@Rasittekin18](https://github.com/Rasittekin18)  
- E-posta: [rasittekin18@gmail.com](mailto:rasittekin18@gmail.com)  
- Konum: İstanbul, Türkiye

---

## 📄 Lisans

Bu projedeki tüm çalışmalar **MIT Lisansı** altında sunulmaktadır.
