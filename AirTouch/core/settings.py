"""
AirTouch Kalıcı Ayar ve Profil Yöneticisi (settings.py)
Streamlit Kontrol Merkezi ve Masaüstü Canlı Kamera Penceresi arasında
ayarları, aktif profili ve özel tuş atamalarını senkronize tutar.
"""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from config import (
    ACTION_PROFILES,
    DEFAULT_AUTO_CONTEXT_PROFILE,
    DEFAULT_CAMERA_SOURCE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_ENGINE_MODE,
    DEFAULT_GESTURE_GUARD_MODE,
    DEFAULT_HUD_COMPACT_MODE,
    DEFAULT_HUD_MONITOR_INDEX,
    DEFAULT_MAX_HAND_MOTION,
    DEFAULT_MIN_MARGIN,
    DEFAULT_PALM_ARM_SECONDS,
    DEFAULT_PROFILE,
    DEFAULT_SHOW_GESTURE_GUIDE,
    DEFAULT_SOUND_FEEDBACK,
    DEFAULT_STABLE_FRAMES,
    DEFAULT_STRICT_NEUTRAL_RESET,
    DEFAULT_VOICE_ENABLED,
    DEFAULT_VOICE_ENGINE,
    DEFAULT_VOICE_WAKE_REQUIRED,
    GESTURE_ACTIONS,
    SETTINGS_FILE,
)


def default_settings() -> Dict[str, Any]:
    """Varsayılan AirTouch yapılandırma sözlüğünü döndürür."""
    media_actions = dict(ACTION_PROFILES[DEFAULT_PROFILE]["actions"])
    cnn_actions = {k: (v if v is not None else "none") for k, v in GESTURE_ACTIONS.items()}
    return {
        "camera_source": DEFAULT_CAMERA_SOURCE,
        "engine_mode": DEFAULT_ENGINE_MODE,
        "active_profile": DEFAULT_PROFILE,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "min_margin": DEFAULT_MIN_MARGIN,
        "cooldown_seconds": DEFAULT_COOLDOWN_SECONDS,
        "required_stable_frames": DEFAULT_STABLE_FRAMES,
        "gesture_guard_mode": DEFAULT_GESTURE_GUARD_MODE,
        "strict_neutral_reset": DEFAULT_STRICT_NEUTRAL_RESET,
        "max_hand_motion": DEFAULT_MAX_HAND_MOTION,
        "palm_arm_seconds": DEFAULT_PALM_ARM_SECONDS,
        "trigger_enabled": True,
        "mirror_camera": True,
        "show_skeleton": True,
        "low_light_boost": True,
        "always_on_top": False,
        "show_gesture_guide": DEFAULT_SHOW_GESTURE_GUIDE,
        "hud_compact_mode": DEFAULT_HUD_COMPACT_MODE,
        "hud_monitor_index": DEFAULT_HUD_MONITOR_INDEX,
        "auto_context_profile": DEFAULT_AUTO_CONTEXT_PROFILE,
        "voice_enabled": DEFAULT_VOICE_ENABLED,
        "voice_wake_required": DEFAULT_VOICE_WAKE_REQUIRED,
        "voice_engine": DEFAULT_VOICE_ENGINE,
        "mic_device_index": None,
        "sound_feedback": DEFAULT_SOUND_FEEDBACK,
        "landmark_actions": {str(k): v for k, v in media_actions.items()},
        "cnn_actions": {str(k): v for k, v in cnn_actions.items()},
    }


def _normalize_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Eksik anahtarları varsayılanlarla tamamlar ve tipleri doğrular."""
    cfg = default_settings()
    if not isinstance(raw, dict):
        return cfg

    for key, val in raw.items():
        if key in ("landmark_actions", "cnn_actions") and isinstance(val, dict):
            merged = dict(cfg[key])
            for gid, act in val.items():
                merged[str(gid)] = act if act is not None else "none"
            cfg[key] = merged
        elif key in cfg:
            cfg[key] = val

    # Kamera kaynağı tamsayı indeks ise int yap
    cam = cfg["camera_source"]
    if isinstance(cam, str) and cam.strip().isdigit():
        cfg["camera_source"] = int(cam.strip())

    cfg["confidence_threshold"] = float(max(0.40, min(0.95, float(cfg["confidence_threshold"]))))
    cfg["cooldown_seconds"] = float(max(0.20, min(4.00, float(cfg["cooldown_seconds"]))))
    cfg["required_stable_frames"] = int(max(1, min(15, int(cfg["required_stable_frames"]))))
    cfg["min_margin"] = float(max(0.05, min(0.60, float(cfg.get("min_margin", DEFAULT_MIN_MARGIN)))))
    if cfg.get("gesture_guard_mode") not in ("palm_arm", "strict_reset", "voice_only"):
        cfg["gesture_guard_mode"] = DEFAULT_GESTURE_GUARD_MODE
    cfg["strict_neutral_reset"] = bool(cfg.get("strict_neutral_reset", True))
    cfg["max_hand_motion"] = float(max(0.05, min(0.60, float(cfg.get("max_hand_motion", DEFAULT_MAX_HAND_MOTION)))))
    cfg["palm_arm_seconds"] = float(max(2.0, min(20.0, float(cfg.get("palm_arm_seconds", DEFAULT_PALM_ARM_SECONDS)))))
    cfg["hud_monitor_index"] = int(max(1, min(8, int(cfg.get("hud_monitor_index", 1)))))
    if cfg.get("voice_engine") not in ("vosk", "google"):
        cfg["voice_engine"] = "vosk"
    mic = cfg.get("mic_device_index")
    if mic is not None:
        try:
            cfg["mic_device_index"] = int(mic)
        except (TypeError, ValueError):
            cfg["mic_device_index"] = None
    return cfg


class SettingsManager:
    """JSON tabanlı kalıcı ayar ve canlı yeniden yükleme (hot-reload) yöneticisi."""

    def __init__(self, settings_path: Path = None):
        self.settings_path = Path(settings_path or SETTINGS_FILE)
        self._last_mtime = 0.0
        self.data = self.load()

    def load(self) -> Dict[str, Any]:
        """Ayarları diskten yükler; dosya yoksa varsayılanları oluşturur."""
        if self.settings_path.exists():
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._last_mtime = self.settings_path.stat().st_mtime
                self.data = _normalize_settings(raw)
                return deepcopy(self.data)
            except Exception:
                pass
        self.data = default_settings()
        self.save(self.data)
        return deepcopy(self.data)

    def save(self, new_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ayarları diske atomik olarak yazar."""
        if new_data is not None:
            self.data = _normalize_settings(new_data)
        else:
            self.data = _normalize_settings(self.data)

        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.settings_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.settings_path)
        self._last_mtime = self.settings_path.stat().st_mtime
        return deepcopy(self.data)

    def reload_if_changed(self) -> bool:
        """Disk üzerindeki ayar dosyası dışarıdan güncellendiyse yeniden yükler."""
        if not self.settings_path.exists():
            return False
        mtime = self.settings_path.stat().st_mtime
        if mtime > self._last_mtime:
            self.load()
            return True
        return False

    def apply_profile(self, profile_key: str) -> Dict[str, Any]:
        """Hazır bir kullanım profilini aktif eder ve kaydeder."""
        if profile_key in ACTION_PROFILES:
            self.data["active_profile"] = profile_key
            profile_actions = ACTION_PROFILES[profile_key]["actions"]
            self.data["landmark_actions"] = {str(k): v for k, v in profile_actions.items()}
            return self.save(self.data)
        return deepcopy(self.data)

    def get_actions_map(self, engine_mode: str = None) -> Dict[int, str]:
        """Seçili motor modu için {int(class_id): action_name} sözlüğü döndürür."""
        mode = engine_mode or self.data.get("engine_mode", DEFAULT_ENGINE_MODE)
        raw_map = self.data.get("cnn_actions" if mode == "cnn" else "landmark_actions", {})
        result: Dict[int, str] = {}
        for k, v in raw_map.items():
            try:
                result[int(k)] = None if v in (None, "none", "") else str(v)
            except ValueError:
                continue
        return result

    def reset(self) -> Dict[str, Any]:
        """Ayarları fabrika varsayılanlarına döndürür."""
        self.data = default_settings()
        return self.save(self.data)

