"""
AirTouch Veri Hattı ve Ön İşleme Modülü (dataset.py)
Kursun Sign Language veri setini PyTorch Dataset yapısına bağlar ve
canlı kamera karelerini model formatına dönüştürür.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

# Kök dizini sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import IMAGE_SIZE, DATA_DIR

class GestureDataset(Dataset):
    """
    İşaret Dili ve El Jestleri için PyTorch Veri Kümesi
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        """
        images: (N, 64, 64) veya (N, 1, 64, 64) float32 [0.0 - 1.0]
        labels: (N,) int64 sınıf indeksleri
        augment: Eğitim sırasında hafif veri artırma uygulansın mı?
        """
        self.images = images.astype(np.float32)
        if len(self.images.shape) == 3:
            # (N, 64, 64) -> (N, 1, 64, 64) Kanal boyutu ekle
            self.images = np.expand_dims(self.images, axis=1)
            
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx] # (1, 64, 64)
        label = self.labels[idx]

        if self.augment:
            img = self._apply_augmentation(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """
        PyTorch/NumPy tabanlı hafif veri artırma (torchvision gerektirmez)
        - Hafif parlaklık/kontrast değişimi
        - Hafif gürültü (gaussian noise)
        - Hafif kaydırma (roll / shift)
        """
        aug_img = img.copy()
        
        # 1. Parlaklık değişimi (%10)
        brightness_factor = np.random.uniform(0.9, 1.1)
        aug_img = aug_img * brightness_factor
        
        # 2. Rastgele 1-2 piksel kaydırma (Shift)
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        aug_img[0] = np.roll(aug_img[0], shift_x, axis=1)
        aug_img[0] = np.roll(aug_img[0], shift_y, axis=0)
        
        # 3. Min-Max kırpma [0.0, 1.0]
        aug_img = np.clip(aug_img, 0.0, 1.0)
        return aug_img


def load_course_dataset(data_dir: Path = None):
    """
    Kurs klasöründeki veya proje verisindeki X.npy ve Y.npy dosyalarını yükler.
    """
    possible_paths = [
        Path("4-Logistic Regression ok/kodlar"),
        Path("5-Artificial Neural Network ( Yapay Sinir Ağları ) ok/kodlar"),
        DATA_DIR
    ]
    
    x_path, y_path = None, None
    for p in possible_paths:
        if (p / "X.npy").exists() and (p / "Y.npy").exists():
            x_path = p / "X.npy"
            y_path = p / "Y.npy"
            break
            
    if not x_path or not y_path:
        raise FileNotFoundError("X.npy ve Y.npy veri seti bulunamadı!")
        
    print(f"Veri seti yükleniyor: {x_path}")
    X = np.load(str(x_path)) # (2062, 64, 64)
    Y = np.load(str(y_path)) # (2062, 10) One-hot
    
    # One-hot'ı tamsayı sınıf indeksine dönüştür (0 - 9)
    if len(Y.shape) > 1 and Y.shape[1] > 1:
        labels = np.argmax(Y, axis=1)
    else:
        labels = Y.ravel().astype(np.int64)
        
    return X, labels


def get_data_loaders(batch_size: int = 64, test_size: float = 0.15, random_state: int = 42):
    """
    Eğitim ve Doğrulama (Validation) PyTorch DataLoader'larını üretir.
    """
    X, y = load_course_dataset()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_dataset = GestureDataset(X_train, y_train, augment=True)
    val_dataset = GestureDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train.shape[0], X_val.shape[0]


def preprocess_frame(frame, bbox=None) -> torch.Tensor:
    """
    Web kamerasından gelen görüntüyü modelin beklediği (1, 1, 64, 64) tensörüne dönüştürür.
    bbox: (x1, y1, x2, y2) El hedef kutusu koordinatları. Verilirse sadece bu bölge kırpılır!
    Girdi: PIL Image veya NumPy array (RGB / BGR)
    Çıktı: PyTorch Tensor (1, 1, 64, 64), float32 [0.0 - 1.0]
    """
    # PIL Image'e çevir
    if isinstance(frame, np.ndarray):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            pil_img = Image.fromarray(frame).convert('L')
        else:
            pil_img = Image.fromarray(frame)
    elif isinstance(frame, Image.Image):
        pil_img = frame.convert('L')
    else:
        raise TypeError("Desteklenmeyen resim tipi!")

    # BBox verilmişse doğrudan el kutusunu kırp (Böylece yüz ve oda elenir!)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        width, height = pil_img.size
        # Sınır kontrolleri
        x1 = max(0, min(int(x1), width - 10))
        y1 = max(0, min(int(y1), height - 10))
        x2 = max(x1 + 10, min(int(x2), width))
        y2 = max(y1 + 10, min(int(y2), height))
        pil_img = pil_img.crop((x1, y1, x2, y2))
    else:
        # Fallback: Merkezden kare kırpma
        width, height = pil_img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        pil_img = pil_img.crop((left, top, left + min_dim, top + min_dim))

    # 64x64 boyutuna küçült
    pil_img = pil_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    # NumPy array
    img_array = np.array(pil_img, dtype=np.float32)

    # Adaptif Kontrast Gerdirme (Düşük Işık ve Oda Işığı Adaptasyonu)
    p_min, p_max = np.percentile(img_array, (2, 98))
    if p_max > p_min + 1e-4:
        img_array = (img_array - p_min) / (p_max - p_min)
        img_array = np.clip(img_array, 0.0, 1.0)
    else:
        img_array = img_array / 255.0

    # (64, 64) -> (1, 1, 64, 64)
    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


if __name__ == "__main__":
    print("Veri Hattı Test Ediliyor...")
    train_loader, val_loader, n_train, n_val = get_data_loaders(batch_size=32)
    print(f"Toplam Eğitim Verisi: {n_train}, Doğrulama Verisi: {n_val}")
    
    for batch_imgs, batch_labels in train_loader:
        print(f"Girdi Batch Boyutu: {batch_imgs.shape}")
        print(f"Etiket Batch Boyutu: {batch_labels.shape}")
        print(f"Piksel Değer Aralığı: [{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]")
        break
        
    # Örnek kare ön işleme testi
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    processed = preprocess_frame(dummy_frame)
    print(f"Kamera Ön İşleme Çıktı Tensörü: {processed.shape}")
    print("Veri Hattı Başarıyla Test Edildi.")
