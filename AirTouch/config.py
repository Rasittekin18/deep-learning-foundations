"""
AirTouch Yapılandırma, Jest Tanımları, Eylem Profilleri ve Varsayılan Ayarlar (config.py)
"""
from pathlib import Path

# Dizin Yolları
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CUSTOM_GESTURES_DIR = DATA_DIR / "custom_gestures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_GESTURES_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = DATA_DIR / "settings.json"
USER_LANDMARK_SAMPLES_FILE = CUSTOM_GESTURES_DIR / "user_samples.npz"

# Model Dosyaları
CNN_MODEL_PATH = MODELS_DIR / "airtouch_model.pth"
CNN_HISTORY_PATH = MODELS_DIR / "training_history.json"
LANDMARK_MODEL_PATH = MODELS_DIR / "airtouch_landmark_model.pth"
LANDMARK_HISTORY_PATH = MODELS_DIR / "landmark_training_history.json"
HAND_LANDMARKER_TASK_PATH = MODELS_DIR / "hand_landmarker.task"
VOSK_TR_MODEL_DIR = MODELS_DIR / "vosk-model-small-tr-0.3"

# Girdi ve Model Parametreleri
IMAGE_SIZE = (64, 64)   # Kurs Sign Language CNN boyutu
NUM_CLASSES = 10        # Kurs CNN 10 sınıf (0-9)
LANDMARK_FEATURE_DIM = 88
LANDMARK_NUM_CLASSES = 8

# ==================== 1. AKILLI İSKELET AI (LANDMARK) JEST SÖZLÜĞÜ ====================
# Açık El (5 parmak) bilinçli olarak "Nötr / Hazır" bırakılmıştır:
# Kullanıcı elini kaldırdığında veya hareketi bitirip elini gevşettiğinde yanlış tetikleme olmaz,
# tek-atış (one-shot) kilidi el indirilmeden temizce sıfırlanır.
LANDMARK_GESTURE_NAMES = {
    0: "Açık El (Hazır / Nötr)",
    1: "Yumruk (Kapalı El)",
    2: "1 Parmak (İşaret)",
    3: "2 Parmak (Zafer / V)",
    4: "3 Parmak",
    5: "4 Parmak",
    6: "OK / Çimdik",
    7: "Başparmak Yukarı",
}

LANDMARK_GESTURE_SHORT = {
    0: "Acik El (Hazir)",
    1: "Yumruk",
    2: "1 Parmak",
    3: "2 Parmak",
    4: "3 Parmak",
    5: "4 Parmak",
    6: "OK / Cimdik",
    7: "Basparmak Yukari",
}

LANDMARK_GESTURE_ICONS = {
    0: "✋",
    1: "✊",
    2: "☝️",
    3: "✌️",
    4: "🤟",
    5: "🖖",
    6: "👌",
    7: "👍",
}

# Parmak durum şemaları [Başparmak, İşaret, Orta, Yüzük, Serçe]:
# Kullanıcıya "hangi parmak ne yapar" görseli üretmek için HUD ve Web arayüzünde kullanılır.
# Değerler: 1=Açık, 0=Kapalı, 2=Çimdik/Özel
LANDMARK_FINGER_PATTERNS = {
    0: {"dots": [1, 1, 1, 1, 1], "pose": "5 parmağı da açın", "role": "Hazır / Kilit Sıfırla"},
    1: {"dots": [0, 0, 0, 0, 0], "pose": "Elinizi yumruk yapın", "role": "Ana Oynat / İşlem"},
    2: {"dots": [0, 1, 0, 0, 0], "pose": "Yalnızca işaret parmağı", "role": "Birincil İleri / Ses +"},
    3: {"dots": [0, 1, 1, 0, 0], "pose": "İşaret + orta parmak (V)", "role": "Birincil Geri / Ses -"},
    4: {"dots": [0, 1, 1, 1, 0], "pose": "İşaret + orta + yüzük", "role": "Sonraki / Geçiş"},
    5: {"dots": [0, 1, 1, 1, 1], "pose": "Başparmak hariç 4 parmak", "role": "Önceki / Geri Geçiş"},
    6: {"dots": [2, 2, 1, 1, 1], "pose": "Baş ve işaret ucunu birleştir", "role": "Sessiz / ESC / Özel"},
    7: {"dots": [2, 0, 0, 0, 0], "pose": "Sadece başparmağı yukarı kaldır", "role": "Tam Ekran / Odak"},
}

# ==================== 2. KURS SIGN LANGUAGE CNN DOĞRU SINIF HARİTASI ====================
# X.npy / Y.npy veri setindeki indekslerin gerçek işaret dili rakam karşılıkları:
# 0->9, 1->0, 2->7, 3->6, 4->1, 5->8, 6->4, 7->3, 8->2, 9->5
SIGN_LANGUAGE_DIGIT_MAP = {
    0: 9,
    1: 0,
    2: 7,
    3: 6,
    4: 1,
    5: 8,
    6: 4,
    7: 3,
    8: 2,
    9: 5,
}

SIGN_LANGUAGE_CLASS_NAMES = {
    0: "Sayı 9 / OK İşareti",
    1: "Sayı 0 / Kapalı O",
    2: "Sayı 7",
    3: "Sayı 6",
    4: "Sayı 1 (Tek Parmak)",
    5: "Sayı 8",
    6: "Sayı 4 (Dört Parmak)",
    7: "Sayı 3 (Üç Parmak)",
    8: "Sayı 2 (Zafer / V)",
    9: "Sayı 5 (Açık El)",
}

# ==================== 3. SİSTEM EYLEM KATALOĞU ====================
# mode: "one_shot" -> tek kez tetikler ve el nötre/boşa dönene kadar kilitlenir
#       "repeat"   -> el tutulduğu sürece cooldown aralığıyla tekrarlar (ses/kaydırma)
#       "idle"     -> işlem yapmaz, kilit sıfırlar
AVAILABLE_ACTIONS = {
    "none": {
        "label": "İşlem Yok (Hazır / Kilit Sıfırla)",
        "mode": "idle",
        "icon": "✋",
    },
    "play_pause": {
        "label": "Oynat / Durdur (Play/Pause)",
        "mode": "one_shot",
        "icon": "⏯️",
    },
    "volume_up": {
        "label": "Sesi Artır (+4%)",
        "mode": "repeat",
        "icon": "🔊",
    },
    "volume_down": {
        "label": "Sesi Azalt (-4%)",
        "mode": "repeat",
        "icon": "🔉",
    },
    "next_track": {
        "label": "Sonraki Parça",
        "mode": "one_shot",
        "icon": "⏭️",
    },
    "prev_track": {
        "label": "Önceki Parça",
        "mode": "one_shot",
        "icon": "⏮️",
    },
    "mute": {
        "label": "Sesi Kapat / Aç (Mute)",
        "mode": "one_shot",
        "icon": "🔇",
    },
    "fullscreen": {
        "label": "Tam Ekran (F Tuşu)",
        "mode": "one_shot",
        "icon": "⛶",
    },
    "next_slide": {
        "label": "Sonraki Slayt / İleri (Sağ Ok)",
        "mode": "one_shot",
        "icon": "➡️",
    },
    "prev_slide": {
        "label": "Önceki Slayt / Geri (Sol Ok)",
        "mode": "one_shot",
        "icon": "⬅️",
    },
    "space_play": {
        "label": "Boşluk Tuşu (Space)",
        "mode": "one_shot",
        "icon": "␣",
    },
    "scroll_up": {
        "label": "Sayfayı Yukarı Kaydır",
        "mode": "repeat",
        "icon": "⬆️",
    },
    "scroll_down": {
        "label": "Sayfayı Aşağı Kaydır",
        "mode": "repeat",
        "icon": "⬇️",
    },
    "esc": {
        "label": "Tam Ekrandan Çık / İptal (ESC)",
        "short_label": "ESC / Çık",
        "mode": "one_shot",
        "icon": "⎋",
    },
    # --- YouTube Akıllı Kontrolleri ---
    "yt_forward_10": {
        "label": "YouTube +10 sn İleri Sar (L)",
        "short_label": "YT +10sn (L)",
        "mode": "one_shot",
        "icon": "⏩",
    },
    "yt_rewind_10": {
        "label": "YouTube -10 sn Geri Sar (J)",
        "short_label": "YT -10sn (J)",
        "mode": "one_shot",
        "icon": "⏪",
    },
    "yt_next_video": {
        "label": "YouTube Sonraki Video (Shift+N)",
        "short_label": "YT Sonraki",
        "mode": "one_shot",
        "icon": "⏭️",
    },
    "yt_captions": {
        "label": "YouTube Altyazı Aç / Kapat (C)",
        "short_label": "YT Altyazı",
        "mode": "one_shot",
        "icon": "💬",
    },
    # --- Çift Ekran & Uygulamalar Arası Geçiş ---
    "next_app": {
        "label": "Sonraki Uygulamaya Geç (Alt+Tab)",
        "short_label": "Uygulama Geç",
        "mode": "one_shot",
        "icon": "🪟",
    },
    "prev_app": {
        "label": "Önceki Uygulamaya Dön (Alt+Shift+Tab)",
        "short_label": "Önceki Uyg.",
        "mode": "one_shot",
        "icon": "🔙",
    },
    "focus_other_screen": {
        "label": "Diğer Ekrandaki Uygulamaya Odaklan",
        "short_label": "Diğer Ekran",
        "mode": "one_shot",
        "icon": "🖥️",
    },
    "move_window_other_screen": {
        "label": "Pencereyi Diğer Ekrana Taşı (Win+Shift+Sağ)",
        "short_label": "Ekrana Taşı",
        "mode": "one_shot",
        "icon": "🔀",
    },
    "browser_next_tab": {
        "label": "Tarayıcı Sonraki Sekme (Ctrl+Tab)",
        "short_label": "Sonraki Sekme",
        "mode": "one_shot",
        "icon": "🗂️",
    },
    "browser_prev_tab": {
        "label": "Tarayıcı Önceki Sekme (Ctrl+Shift+Tab)",
        "short_label": "Önceki Sekme",
        "mode": "one_shot",
        "icon": "📑",
    },
    "focus_youtube": {
        "label": "YouTube Penceresine Odaklan",
        "short_label": "YouTube Odak",
        "mode": "one_shot",
        "icon": "▶️",
    },
    "focus_spotify": {
        "label": "Spotify Penceresine Odaklan",
        "short_label": "Spotify Odak",
        "mode": "one_shot",
        "icon": "🎵",
    },
    "snap_left": {
        "label": "Pencereyi Sola Yasla (Win+Sol)",
        "short_label": "Sola Yasla",
        "mode": "one_shot",
        "icon": "◧",
    },
    "snap_right": {
        "label": "Pencereyi Sağa Yasla (Win+Sağ)",
        "short_label": "Sağa Yasla",
        "mode": "one_shot",
        "icon": "◨",
    },
}

# ==================== 4. HAZIR KULLANIM PROFİLLERİ ====================
ACTION_PROFILES = {
    "media": {
        "name": "🎬 Medya Kontrolü (Genel / Spotify / VLC)",
        "short_name": "Medya",
        "actions": {
            0: "none",
            1: "play_pause",
            2: "volume_up",
            3: "volume_down",
            4: "next_track",
            5: "prev_track",
            6: "mute",
            7: "fullscreen",
        },
    },
    "youtube": {
        "name": "▶️ YouTube Akıllı Mod (+10sn / Ses / Tam Ekran)",
        "short_name": "YouTube",
        "actions": {
            0: "none",
            1: "play_pause",
            2: "volume_up",
            3: "volume_down",
            4: "yt_forward_10",
            5: "yt_rewind_10",
            6: "mute",
            7: "fullscreen",
        },
    },
    "workspace": {
        "name": "🖥️ Çift Ekran & Uygulama Geçişi",
        "short_name": "Cift Ekran",
        "actions": {
            0: "none",
            1: "next_app",
            2: "focus_other_screen",
            3: "move_window_other_screen",
            4: "browser_next_tab",
            5: "browser_prev_tab",
            6: "mute",
            7: "focus_youtube",
        },
    },
    "presentation": {
        "name": "📊 Sunum Kontrolü (PowerPoint / PDF)",
        "short_name": "Sunum",
        "actions": {
            0: "none",
            1: "space_play",
            2: "next_slide",
            3: "prev_slide",
            4: "scroll_down",
            5: "scroll_up",
            6: "esc",
            7: "fullscreen",
        },
    },
    "browser": {
        "name": "🌐 Tarayıcı & Sekme Okuma Modu",
        "short_name": "Tarayici",
        "actions": {
            0: "none",
            1: "space_play",
            2: "scroll_up",
            3: "scroll_down",
            4: "browser_next_tab",
            5: "browser_prev_tab",
            6: "mute",
            7: "fullscreen",
        },
    },
    "focus": {
        "name": "🛡️ Odak Modu (Yalnızca 3 Temel Jest)",
        "short_name": "Odak (3 Jest)",
        "actions": {
            0: "none",
            1: "play_pause",
            2: "volume_up",
            3: "volume_down",
            4: "none",
            5: "none",
            6: "none",
            7: "none",
        },
    },
}

# Geriye dönük uyumluluk sabitleri (Kurs CNN ve eski modüller için)
GESTURE_NAMES = SIGN_LANGUAGE_CLASS_NAMES

GESTURE_ACTIONS = {
    0: "mute",         # Sayı 9 / OK
    1: "play_pause",   # Sayı 0 / Kapalı O
    2: None,           # Sayı 7
    3: None,           # Sayı 6
    4: "volume_up",    # Sayı 1
    5: None,           # Sayı 8
    6: "prev_track",   # Sayı 4
    7: "next_track",   # Sayı 3
    8: "volume_down",  # Sayı 2 (V)
    9: None,           # Sayı 5 (Açık El - Nötr)
}

# Varsayılan Tetikleme Hassasiyeti ve Kamera Ayarları
DEFAULT_CAMERA_SOURCE = 0            # Normal laptop varsayılan kamerası!
DEFAULT_ENGINE_MODE = "landmark"     # "landmark" (Önerilen) veya "cnn"
DEFAULT_PROFILE = "media"
DEFAULT_CONFIDENCE_THRESHOLD = 0.78
DEFAULT_MIN_MARGIN = 0.24
DEFAULT_COOLDOWN_SECONDS = 1.25
DEFAULT_STABLE_FRAMES = 6

# Kazara / Arka Arkaya El Tetiklemesini Önleyen Güvenlik Kapısı Varsayılanları
DEFAULT_GESTURE_GUARD_MODE = "palm_arm"  # "palm_arm" | "strict_reset" | "voice_only"
DEFAULT_STRICT_NEUTRAL_RESET = True
DEFAULT_MAX_HAND_MOTION = 0.16
DEFAULT_PALM_ARM_SECONDS = 6.0

# Sesli Komut, Çift Ekran & HUD Varsayılanları
DEFAULT_VOICE_ENABLED = True
DEFAULT_VOICE_WAKE_REQUIRED = False
DEFAULT_VOICE_ENGINE = "vosk"          # "vosk" (Çevrimdışı Türkçe) veya "google"
DEFAULT_SOUND_FEEDBACK = True
DEFAULT_SHOW_GESTURE_GUIDE = True      # HUD sağ tarafında canlı "Hangi parmak ne yapar" paneli
DEFAULT_HUD_COMPACT_MODE = False       # Mini PiP köşe penceresi modu
DEFAULT_HUD_MONITOR_INDEX = 1          # 1: Birincil ekran, 2: İkinci ekran
DEFAULT_AUTO_CONTEXT_PROFILE = False   # Aktif uygulamaya göre otomatik profil önerisi/geçişi
