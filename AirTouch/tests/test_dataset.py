"""
Veri Hattı ve Kamera Ön İşleme Mimari Testleri (test_dataset.py)
"""
import pytest
import numpy as np
import torch
from PIL import Image

from core.dataset import (
    GestureDataset, 
    preprocess_frame, 
    load_course_dataset, 
    get_data_loaders
)
from config import IMAGE_SIZE

class TestDatasetAndPreprocessingArchitecture:

    def test_preprocess_frame_shape_and_dtype_invariants(self, dummy_rgb_frame):
        """Kamera ön işleme çıktısı daima (1, 1, 64, 64) ve float32 olmalıdır."""
        tensor = preprocess_frame(dummy_rgb_frame)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
        assert tensor.dtype == torch.float32

    def test_preprocess_frame_normalization_bounds(self, dummy_rgb_frame, dummy_black_frame, dummy_white_frame):
        """Piksel değerleri kesinlikle [0.0, 1.0] aralığında olmalıdır (Asla negatif veya >1 olamaz)."""
        for frame in [dummy_rgb_frame, dummy_black_frame, dummy_white_frame]:
            tensor = preprocess_frame(frame)
            assert tensor.min().item() >= 0.0, "Piksel değeri 0'dan küçük olamaz!"
            assert tensor.max().item() <= 1.0, "Piksel değeri 1'den büyük olamaz!"
            assert not torch.isnan(tensor).any(), "Tensörde NaN tespit edildi!"
            assert not torch.isinf(tensor).any(), "Tensörde Inf tespit edildi!"

    def test_preprocess_frame_supports_multiple_formats(self, dummy_rgb_frame, dummy_pil_frame):
        """Fonksiyon hem NumPy array hem de PIL Image formatlarını desteklemelidir."""
        t_numpy = preprocess_frame(dummy_rgb_frame)
        t_pil = preprocess_frame(dummy_pil_frame)
        assert t_numpy.shape == t_pil.shape == (1, 1, 64, 64)

    def test_preprocess_frame_unsupported_type_raises(self):
        """Desteklenmeyen bir veri tipi verildiğinde TypeError fırlatmalıdır."""
        with pytest.raises(TypeError):
            preprocess_frame("invalid_string_input")

    def test_preprocess_frame_center_crop_aspect_ratios(self):
        """Geniş (1920x1080) ve dikey (1080x1920) görüntüler bozulmadan ortadan kare kırpılmalıdır."""
        wide_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        tall_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        
        t_wide = preprocess_frame(wide_frame)
        t_tall = preprocess_frame(tall_frame)
        
        assert t_wide.shape == (1, 1, 64, 64)
        assert t_tall.shape == (1, 1, 64, 64)

    def test_dataset_augmentation_stability(self):
        """Veri artırma işlemi (augmentation) veriyi bozmamalı, pikseller [0, 1] aralığında kalmalıdır."""
        dummy_images = np.random.uniform(0.1, 0.9, (10, 64, 64)).astype(np.float32)
        dummy_labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        
        ds = GestureDataset(dummy_images, dummy_labels, augment=True)
        
        for i in range(len(ds)):
            img, label = ds[i]
            assert img.shape == (1, 64, 64)
            assert img.min().item() >= 0.0
            assert img.max().item() <= 1.0
            assert not torch.isnan(img).any()

    def test_course_dataset_loading_integrity(self):
        """Kurs veri seti X.npy ve Y.npy doğru boyutlarda ve 10 sınıflı olarak yüklenmelidir."""
        X, labels = load_course_dataset()
        assert X.shape[0] == 2062, "Sign Language veri seti 2062 örnek olmalı!"
        assert X.shape[1:] == (64, 64), "Görüntü boyutu 64x64 olmalı!"
        assert len(labels) == 2062
        assert set(np.unique(labels)) == set(range(10)), "0-9 arası tüm 10 sınıf mevcut olmalı!"

    def test_dataloader_batch_dimensions(self):
        """DataLoader partileri (batch) modelin beklediği (B, 1, 64, 64) tensörleri üretmelidir."""
        train_loader, val_loader, n_train, n_val = get_data_loaders(batch_size=16)
        assert n_train > 0 and n_val > 0
        
        images, labels = next(iter(train_loader))
        assert images.shape == (16, 1, 64, 64)
        assert labels.shape == (16,)
        assert labels.dtype == torch.int64

