"""
AirTouch v2.0 İskelet Öznitelikleri, Ayar Yöneticisi, Kamera Ön İşleme,
HUD Çizimi ve Birleşik Motor Durum Makinesi Testleri (test_landmark_and_settings.py)
"""
import time
import numpy as np
import pytest
import torch

from config import (
    ACTION_PROFILES,
    LANDMARK_FEATURE_DIM,
    LANDMARK_MODEL_PATH,
    LANDMARK_NUM_CLASSES,
    SIGN_LANGUAGE_DIGIT_MAP,
)
from core.camera import apply_low_light_boost, parse_camera_source
from core.engine import AirTouchEngine
from core.hand_tracker import analyze_finger_states, extract_hand_features
from core.hud_renderer import render_hud_frame
from core.landmark_trainer import generate_synthetic_hand_landmarks
from core.settings import SettingsManager


class _FakeHandTracker:
    """Deterministik birim testleri için sahte MediaPipe takipçisi."""

    def __init__(self, landmarks=None, detected=True):
        self.landmarks = landmarks
        self.detected = detected
        self.is_available = True

    def detect(self, rgb_image):
        if not self.detected or self.landmarks is None:
            return {
                "detected": False,
                "landmarks": None,
                "pixel_landmarks": None,
                "bbox": None,
                "handedness": None,
                "score": 0.0,
                "features": None,
                "finger_states": None,
            }
        h, w = rgb_image.shape[:2]
        px = np.stack([self.landmarks[:, 0] * w, self.landmarks[:, 1] * h], axis=1)
        return {
            "detected": True,
            "landmarks": self.landmarks,
            "pixel_landmarks": px,
            "bbox": (100, 80, 320, 300),
            "handedness": "Right",
            "score": 0.98,
            "features": extract_hand_features(self.landmarks, aspect_ratio=w / max(h, 1)),
            "finger_states": analyze_finger_states(self.landmarks, aspect_ratio=w / max(h, 1)),
        }

    def reset_smoothing(self):
        pass

    def close(self):
        pass


class TestLandmarkFeaturesAndInvariance:

    def test_feature_vector_shape_and_invariance(self):
        """Öznitelik vektörü öteleme, ölçek ve sağ/sol yatay aynalamaya karşı değişmez olmalıdır."""
        rng = np.random.default_rng(7)
        lm = generate_synthetic_hand_landmarks(1, rng)
        f_base = extract_hand_features(lm, aspect_ratio=1.0)
        assert f_base.shape == (LANDMARK_FEATURE_DIM,)
        assert not np.isnan(f_base).any()

        # Öteleme ve ölçekleme
        lm_shifted = lm * 1.8 + np.array([0.22, -0.15, 0.0], dtype=np.float32)
        f_shifted = extract_hand_features(lm_shifted, aspect_ratio=1.0)
        assert np.allclose(f_base, f_shifted, atol=1e-4)

        # Yatay aynalama (Sağ el <-> Sol el simetrisi)
        lm_mirrored = lm.copy()
        lm_mirrored[:, 0] = 1.0 - lm_mirrored[:, 0]
        f_mirrored = extract_hand_features(lm_mirrored, aspect_ratio=1.0)
        assert np.allclose(f_base, f_mirrored, atol=1e-4)

    def test_biomech_finger_states_match_poses(self):
        """Biyomekanik parmak analizi temel el pozlarını doğru özetlemelidir."""
        rng = np.random.default_rng(11)
        fs_open = analyze_finger_states(generate_synthetic_hand_landmarks(0, rng))
        assert fs_open["four_count"] == 4
        assert fs_open["extended"]["thumb"] is True

        fs_fist = analyze_finger_states(generate_synthetic_hand_landmarks(1, rng))
        assert fs_fist["four_count"] == 0

        fs_one = analyze_finger_states(generate_synthetic_hand_landmarks(2, rng))
        assert fs_one["extended"]["index"] is True
        assert fs_one["four_count"] == 1

        fs_pinch = analyze_finger_states(generate_synthetic_hand_landmarks(6, rng))
        assert fs_pinch["pinch_ratio"] < 0.25

        fs_thumb_up = analyze_finger_states(generate_synthetic_hand_landmarks(7, rng))
        assert fs_thumb_up["four_count"] == 0
        assert fs_thumb_up["thumb_up_score"] > 0.50

    def test_sign_language_digit_map_is_bijection(self):
        """X.npy sınıf indeks haritası 0-9 arası tüm rakamları birebir kapsamalıdır."""
        assert set(SIGN_LANGUAGE_DIGIT_MAP.keys()) == set(range(10))
        assert set(SIGN_LANGUAGE_DIGIT_MAP.values()) == set(range(10))


class TestSettingsAndCameraHelpers:

    def test_settings_persistence_and_profiles(self, tmp_path):
        """Ayarlar diske kaydedilmeli, profiller yüklenmeli ve hot-reload çalışmalıdır."""
        cfg_file = tmp_path / "settings.json"
        mgr = SettingsManager(settings_path=cfg_file)
        assert cfg_file.exists()
        assert mgr.data["camera_source"] == 0

        mgr.apply_profile("presentation")
        assert mgr.data["active_profile"] == "presentation"
        assert mgr.get_actions_map("landmark")[2] == "next_slide"

        # Harici değişiklik algılama (hot-reload)
        mgr2 = SettingsManager(settings_path=cfg_file)
        mgr2.data["confidence_threshold"] = 0.88
        time.sleep(0.02)
        mgr2.save(mgr2.data)
        assert mgr.reload_if_changed() is True
        assert abs(mgr.data["confidence_threshold"] - 0.88) < 1e-5

        mgr.reset()
        assert mgr.data["active_profile"] == "media"

    def test_camera_source_parsing_and_low_light_boost(self):
        """Kamera kaynağı varsayılan 0 olmalı ve loş kareler CLAHE ile iyileştirilmelidir."""
        assert parse_camera_source(None) == 0
        assert parse_camera_source("") == 0
        assert parse_camera_source("0") == 0
        assert parse_camera_source("1") == 1

        dark_bgr = np.full((120, 160, 3), 30, dtype=np.uint8)
        bright_bgr = np.full((120, 160, 3), 180, dtype=np.uint8)
        _, boosted_dark = apply_low_light_boost(dark_bgr)
        _, boosted_bright = apply_low_light_boost(bright_bgr)
        assert boosted_dark is True
        assert boosted_bright is False


class TestUnifiedLandmarkEngineAndHUD:

    def test_landmark_hold_confirm_and_one_shot_latch(self, monkeypatch):
        """One-shot eylem N kare onaydan sonra 1 kez tetiklenip Açık El gösterilene kadar kilitlenmelidir."""
        rng = np.random.default_rng(99)
        lm_fist = generate_synthetic_hand_landmarks(1, rng)
        lm_open = generate_synthetic_hand_landmarks(0, rng)

        tracker = _FakeHandTracker(landmarks=lm_fist, detected=True)
        eng = AirTouchEngine(
            model_path=LANDMARK_MODEL_PATH,
            engine_mode="landmark",
            confidence_threshold=0.65,
            cooldown_seconds=0.10,
            required_stable_frames=3,
            smoothing_window=3,
            hand_tracker=tracker,
        )

        # Gerçek Windows tuşuna basmayı engelle
        executed = []
        from core import engine as engine_mod
        monkeypatch.setattr(
            engine_mod.system_controller,
            "execute_action",
            lambda name: executed.append(name) or f"Mock:{name}",
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # İlk 2 kare: CONFIRMING (henüz tetiklememeli)
        r1 = eng.process_frame(frame, trigger_enabled=True)
        r2 = eng.process_frame(frame, trigger_enabled=True)
        assert r1["action_executed"] is None and r1["status_state"] == "CONFIRMING"
        assert r2["action_executed"] is None and r2["status_state"] == "CONFIRMING"

        # 3. kare: TRIGGERED (play_pause)
        r3 = eng.process_frame(frame, trigger_enabled=True)
        assert r3["action_executed"] == "Mock:play_pause"
        assert executed == ["play_pause"]

        # 4. kare (el hala yumruk): LATCHED (tekrar tetiklenmemeli!)
        time.sleep(0.15)
        r4 = eng.process_frame(frame, trigger_enabled=True)
        assert r4["status_state"] == "LATCHED"
        assert r4["action_executed"] is None
        assert len(executed) == 1

        # Kullanıcı Açık El (0 - Nötr) gösterir -> kilit açılır
        tracker.landmarks = lm_open
        for _ in range(4):
            eng.process_frame(frame, trigger_enabled=True)

        # Tekrar Yumruk -> smoothing (3) + onay (3) sonrası yeniden tetiklenmeli!
        tracker.landmarks = lm_fist
        for _ in range(4):
            r_again = eng.process_frame(frame, trigger_enabled=True)
        assert len(executed) == 2
        assert r_again["annotated_frame"].size == (640, 480)

    def test_paused_standby_blocks_all_triggers(self, monkeypatch):
        """Bekleme modu (paused=True) açıkken hiçbir Windows tuşu tetiklenmemelidir."""
        rng = np.random.default_rng(5)
        tracker = _FakeHandTracker(landmarks=generate_synthetic_hand_landmarks(2, rng), detected=True)
        eng = AirTouchEngine(
            model_path=LANDMARK_MODEL_PATH,
            engine_mode="landmark",
            required_stable_frames=1,
            hand_tracker=tracker,
        )
        eng.paused = True

        called = []
        from core import engine as engine_mod
        monkeypatch.setattr(engine_mod.system_controller, "execute_action", lambda name: called.append(name))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rep = eng.process_frame(frame, trigger_enabled=True)
        assert rep["status_state"] == "PAUSED"
        assert rep["action_executed"] is None
        assert called == []

    def test_turkish_hud_rendering_returns_valid_image(self):
        """HUD çizimi Türkçe karakterlerle hatasız görüntü üretmelidir."""
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        report = {
            "status_state": "TRIGGERED",
            "is_confident": True,
            "is_cooldown_active": False,
            "action_executed": "Ses Artırıldı (+4%)",
            "class_name": "1 Parmak (İşaret)",
            "confidence": 0.96,
            "has_hand": True,
            "confirm_progress": 1.0,
        }
        img = render_hud_frame(
            frame,
            report,
            bbox=(200, 100, 420, 320),
            profile_label="Medya",
            fps=30.0,
            footer_hint="[SPACE] Duraklat | [Q] Çıkış",
        )
        assert img.size == (800, 450)
        assert img.mode == "RGB"
