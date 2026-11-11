"""
AirTouch Yapılandırma ve Varsayılan Ayarlar (config.py)
"""
import os
from pathlib import Path

# Dizin Yolları
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CUSTOM_GESTURES_DIR = DATA_DIR / "custom_gestures"

# Dizinleri otomatik oluştur
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_GESTURES_DIR.mkdir(parents=True, exist_ok=True)

# Girdi ve Model Parametreleri
IMAGE_SIZE = (64, 64)  # Kursun Sign Language boyutuna tam uyumlu
NUM_CLASSES = 10       # 0'dan 9'a 10 sınıf

# Sınıf İsimleri ve Varsayılan Windows Eylem Eşleşmeleri
GESTURE_NAMES = {
    0: "Yumruk (Oynat / Durdur) ⏯️",
    1: "İşaret Parmağı (Ses Aç) 🔊",
    2: "İki Parmak / V (Ses Kıs) 🔉",
    3: "Üç Parmak (Sonraki Şarkı) ⏭️",
    4: "Dört Parmak (Önceki Şarkı) ⏮️",
    5: "Açık El (Sessize Al) 🔇",
    6: "İşaret 6",
    7: "İşaret 7",
    8: "İşaret 8",
    9: "İşaret 9"
}

GESTURE_ACTIONS = {
    0: "play_pause",
    1: "volume_up",
    2: "volume_down",
    3: "next_track",
    4: "prev_track",
    5: "mute",
    6: None,
    7: None,
    8: None,
    9: None
}

# Canlı Tetikleme Hassasiyeti (Debounce & Cooldown)
DEFAULT_CONFIDENCE_THRESHOLD = 0.75  # Aşırı tetiklemeyi önleyen ideal güven eşiği (%75)
DEFAULT_COOLDOWN_SECONDS = 1.00      # Ardışık basmayı önleyen bekleme süresi (1.0 sn)

