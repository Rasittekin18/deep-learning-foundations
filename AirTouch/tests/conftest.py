"""
Pytest Test Yapılandırması ve Ortak Fikstürler (conftest.py)
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image

# Proje kök dizinini test yoluna ekle
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import IMAGE_SIZE, NUM_CLASSES
from core.model import AirTouchCNN

@pytest.fixture
def dummy_rgb_frame():
    """640x480 rastgele renkli kamera karesi (NumPy uint8)."""
    np.random.seed(42)
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def dummy_pil_frame():
    """PIL formatında test karesi."""
    return Image.fromarray(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8))

@pytest.fixture
def dummy_black_frame():
    """Siyah (sıfır piksel) test karesi."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def dummy_white_frame():
    """Beyaz (255 piksel) test karesi."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 255

@pytest.fixture
def initialized_model():
    """Test için örnek AirTouchCNN modeli."""
    model = AirTouchCNN(num_classes=NUM_CLASSES)
    model.eval()
    return model

