"""
AirTouch v2.1 Pro Türkçe Sesli Komut, Çift Ekran/Pencere Bağlamı,
Parmak Rehberli HUD & Mini PiP Mimari Testleri (test_voice_and_multiscreen.py)
"""
import io
import time
import numpy as np
import pytest

from config import ACTION_PROFILES, LANDMARK_FINGER_PATTERNS, LANDMARK_MODEL_PATH, LANDMARK_NUM_CLASSES, VOSK_TR_MODEL_DIR
from core.engine import AirTouchEngine
from core.hand_tracker import analyze_finger_states, extract_hand_features
from core.hud_renderer import render_hud_frame
from core.landmark_trainer import generate_synthetic_hand_landmarks
from core.settings import SettingsManager
from core.system_controller import resolve_youtube_video_url
from core.voice_commander import (
    _is_non_mic_device,
    _resample_to_16k_int16,
    get_cpp_safe_model_path,
    match_voice_command,
    normalize_voice_text,
)
from core.window_manager import classify_window_context, is_airtouch_window, window_manager


class TestVoiceCommandParserAndVosk:

    def test_turkish_normalization_handles_combining_dot(self):
        """Türkçe 'İ'/'i̇' birleştirici nokta ve aksan karakterleri tutarlı normalize edilmelidir."""
        assert normalize_voice_text("İleri Sar!") == "ileri sar"
        assert normalize_voice_text("i\u0307leri sar") == "ileri sar"
        assert normalize_voice_text("ÇİFT EKRAN MODU") == "cift ekran modu"

    def test_voice_command_intents_youtube_and_multiscreen(self):
        """Medya, YouTube, çift ekran geçişi, sekme ve HUD meta komutları doğru eşlenmelidir."""
        cases = [
            ("videoyu oynat", "action", "play_pause"),
            ("durdur", "action", "play_pause"),
            ("sesi artır", "action", "volume_up"),
            ("sesini azalt", "action", "volume_down"),
            ("sesi kapat", "action", "mute"),
            ("tam ekran", "action", "fullscreen"),
            ("i\u0307leri sar", "action", "yt_forward_10"),
            ("geri sar", "action", "yt_rewind_10"),
            ("sonraki video", "action", "yt_next_video"),
            ("diğer ekrana geç", "action", "focus_other_screen"),
            ("ekrana taşı", "action", "move_window_other_screen"),
            ("uygulama değiştir", "action", "next_app"),
            ("sonraki set mi", "action", "browser_next_tab"),
            ("youtube geç", "action", "focus_youtube"),
            ("video dursun", "action", "play_pause"),
            ("videoyu durdur", "action", "play_pause"),
            ("video başlasın", "action", "play_pause"),
            ("video oynasın", "action", "play_pause"),
            ("youtube modu", "profile", "youtube"),
            ("çift ekran modu", "profile", "workspace"),
            ("odak modu", "profile", "focus"),
            ("şunun modu", "profile", "presentation"),
            ("minimum", "meta", "compact_on"),
            ("bekleme modu", "meta", "pause"),
            ("devam et", "meta", "resume"),
            ("parmak rehberi", "meta", "guide_toggle"),
        ]
        for spoken, exp_type, exp_target in cases:
            res = match_voice_command(spoken, wake_required=False)
            assert res is not None, f"Komut çözülemedi: {spoken}"
            assert (res["type"], res["target"]) == (exp_type, exp_target), f"{spoken} -> {res}"

    def test_wake_word_mode_prevents_speaker_false_triggers(self):
        """Uyandırma kelimesi açıkken hoparlörden gelen yalın kelimeler tetiklememelidir."""
        assert match_voice_command("videoyu oynat", wake_required=True) is None
        assert match_voice_command("tam ekran", wake_required=True) is None

        ok = match_voice_command("asistan diğer ekrana geç", wake_required=True)
        assert ok is not None and ok["target"] == "focus_other_screen"

        ok2 = match_voice_command("air sesi artır", wake_required=True)
        assert ok2 is not None and ok2["target"] == "volume_up"

    def test_vosk_turkish_model_loads_via_safe_path(self):
        """Türkçe klasör yolu altında Vosk çevrimdışı model güvenli yoldan yüklenebilmelidir."""
        if not VOSK_TR_MODEL_DIR.exists():
            pytest.skip("Vosk Türkçe modeli kurulu değil.")
        import vosk

        safe_path = get_cpp_safe_model_path(VOSK_TR_MODEL_DIR)
        assert safe_path is not None and safe_path.isascii()
        vosk.SetLogLevel(-1)
        model = vosk.Model(safe_path)
        assert model is not None


class TestWindowManagerAndProfiles:

    def test_context_classification_and_airtouch_filter(self):
        """Aktif pencere başlığı ve süreç adı doğru bağlama sınıflandırılmalıdır."""
        assert classify_window_context("Python Dersleri - YouTube - Google Chrome", "chrome.exe") == "youtube"
        assert classify_window_context("Sunum1.pptx - PowerPoint", "POWERPNT.EXE") == "presentation"
        assert classify_window_context("Daily Mix", "Spotify.exe") == "media"
        assert classify_window_context("Dokümantasyon - Microsoft Edge", "msedge.exe") == "browser"
        assert classify_window_context("app.py - Visual Studio Code", "Code.exe") == "workspace"

        assert is_airtouch_window("AirTouch - Canli Masaustu Kontrol") is True
        assert is_airtouch_window("YouTube - Google Chrome") is False

    def test_monitors_list_returns_primary(self):
        """Monitör taraması en az 1 ekran ve geçerli dikdörtgen sınırları döndürmelidir."""
        mons = window_manager.get_monitors()
        assert len(mons) >= 1
        assert mons[0]["index"] == 1
        l, t, r, b = mons[0]["bounds"]
        assert r > l and b > t

    def test_five_profiles_and_finger_patterns_complete(self):
        """Tüm profiller 8 sınıfı eksiksiz içermeli; parmak şemaları 5 parmak uzunluğunda olmalıdır."""
        assert {"media", "youtube", "workspace", "presentation", "browser", "focus"}.issubset(ACTION_PROFILES.keys())
        for gid in range(LANDMARK_NUM_CLASSES):
            pat = LANDMARK_FINGER_PATTERNS[gid]
            assert len(pat["dots"]) == 5
        for pkey, pdata in ACTION_PROFILES.items():
            assert set(pdata["actions"].keys()) == set(range(LANDMARK_NUM_CLASSES)), pkey

    def test_settings_normalizes_voice_and_multiscreen_fields(self, tmp_path):
        """Yeni ses, monitör ve mini HUD alanları settings.json'da doğrulanıp saklanmalıdır."""
        mgr = SettingsManager(settings_path=tmp_path / "settings.json")
        mgr.data["hud_monitor_index"] = 2
        mgr.data["hud_compact_mode"] = True
        mgr.data["voice_enabled"] = True
        mgr.data["voice_wake_required"] = True
        mgr.save(mgr.data)

        reloaded = SettingsManager(settings_path=tmp_path / "settings.json")
        assert reloaded.data["hud_monitor_index"] == 2
        assert reloaded.data["hud_compact_mode"] is True
        assert reloaded.data["voice_wake_required"] is True


class TestHudFingerGuideAndCompactMode:

    def test_hud_with_finger_guide_and_voice_status_preserves_size(self):
        """Sağ kenar 'Hangi Parmak Ne Yapar?' rehberli tam HUD çözünürlüğü koruyarak çizilmelidir."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        report = {
            "status_state": "CONFIRMING",
            "is_confident": True,
            "is_cooldown_active": False,
            "class_id": 2,
            "class_name": "1 Parmak (İşaret)",
            "confidence": 0.94,
            "has_hand": True,
            "confirm_progress": 0.75,
            "engine_mode": "landmark",
        }
        img = render_hud_frame(
            frame,
            report,
            bbox=(120, 100, 360, 340),
            profile_label="YouTube",
            fps=30.0,
            footer_hint="[G] Rehber | [V] Ses | [W] Ekran:1",
            action_map=ACTION_PROFILES["youtube"]["actions"],
            voice_status={"running": True, "available": True, "mic_level": 0.45, "last_heard": "sesi artır"},
            target_context={"title": "Lo-Fi Beats - YouTube", "monitor_index": 2},
            show_guide=True,
            compact_mode=False,
        )
        assert img.size == (960, 540)
        assert img.mode == "RGB"

    def test_hud_compact_pip_mode_preserves_size(self):
        """Mini PiP köşe HUD modu 420x255 boyutunda temiz çıktı üretmelidir."""
        frame = np.zeros((255, 420, 3), dtype=np.uint8)
        report = {
            "status_state": "READY",
            "is_confident": True,
            "is_cooldown_active": False,
            "class_id": 0,
            "class_name": "Açık El",
            "confidence": 0.91,
            "has_hand": True,
            "confirm_progress": 1.0,
            "engine_mode": "landmark",
        }
        img = render_hud_frame(
            frame,
            report,
            profile_label="Cift Ekran",
            voice_status={"running": True, "available": True, "mic_level": 0.25},
            target_context={"title": "YouTube", "monitor_index": 1},
            show_guide=True,
            compact_mode=True,
        )
        assert img.size == (420, 255)


def test_voice_dynamic_youtube_queries_and_ambiguity():
    """Doğal YouTube arama-açma cümleleri ile sabit aksiyon/profil cümleleri doğru ayrışmalıdır."""
    from core.voice_commander import match_voice_command

    evt1 = match_voice_command("youtube'da barış manço aç")
    assert evt1 is not None and evt1["type"] == "youtube_play"
    assert evt1["target"] == "youtube_search_play"
    assert evt1["query"] == "baris manco"

    evt2 = match_voice_command("youtube tarkan kuzum açılsın")
    assert evt2 is not None and evt2["type"] == "youtube_play"
    assert evt2["query"] == "tarkan kuzum"

    evt3 = match_voice_command("youtube'da fayton dersleri aç")
    assert evt3 is not None and evt3["type"] == "youtube_play"
    assert evt3["query"] == "python dersleri"

    evt4 = match_voice_command("barış manço gülpembe şarkısını çal")
    assert evt4 is not None and evt4["type"] == "youtube_play"
    assert evt4["query"] == "baris manco gulpembe"

    evt_focus = match_voice_command("youtube şu video açılsın")
    assert evt_focus is not None and evt_focus["type"] == "action"
    assert evt_focus["target"] == "focus_youtube"

    evt_prof = match_voice_command("youtube modu")
    assert evt_prof is not None and evt_prof["type"] == "profile"
    assert evt_prof["target"] == "youtube"


def test_resolve_youtube_video_url_html_parsing(monkeypatch):
    """YouTube arama HTML cevabından ilk videoRenderer ID'sini güvenle ayıklamalıdır."""
    import urllib.request
    from core.system_controller import resolve_youtube_video_url

    class _FakeResp:
        def __init__(self, payload: bytes):
            self._payload = payload

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_html = b'<html>"videoRenderer":{"videoId":"dQw4w9WgXcQ","title":{"runs":[{"text":"Test"}]}}</html>'
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=4.0: _FakeResp(fake_html))

    url, vid = resolve_youtube_video_url("baris manco gulpembe", timeout=2.0)
    assert vid == "dQw4w9WgXcQ"
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_voice_device_filter_and_resample_agc():
    """Loopback (Stereo Karışımı) cihazları elenmeli; 48kHz -> 16kHz dönüşüm + AGC çalışmalıdır."""
    import numpy as np
    from core.voice_commander import _is_non_mic_device, _resample_to_16k_int16

    assert _is_non_mic_device("Stereo Karışımı (Realtek(R) Audio)") is True
    assert _is_non_mic_device("Stereo Mix (Realtek)") is True
    assert _is_non_mic_device("Mikrofon Dizisi (Realtek(R) Audio)") is False

    # 48000 Hz, 0.1s düşük genlikli konuşma sinyali (tepe=600)
    t = np.linspace(0.0, 0.1, 4800, endpoint=False)
    samples = (np.sin(2.0 * np.pi * 440.0 * t) * 600.0).astype(np.int16)
    out_bytes, level = _resample_to_16k_int16(
        samples.tobytes(),
        src_rate=48000,
        apply_agc=True,
        noise_floor=0.003,
    )
    out_arr = np.frombuffer(out_bytes, dtype=np.int16)
    assert len(out_arr) == 1600
    assert len(out_bytes) == 3200
    assert int(np.max(np.abs(out_arr))) > 600
    assert level > 0.0


class _FakeGuardHandTracker:
    is_available = True

    def __init__(self):
        self.landmarks = None
        self.features = None
        self.finger_states = None

    def set_pose(self, gesture_id: int, dx: float = 0.0, seed: int = 77):
        import numpy as np
        from core.hand_tracker import analyze_finger_states, extract_hand_features
        from core.landmark_trainer import generate_synthetic_hand_landmarks

        rng = np.random.default_rng(seed + gesture_id)
        lm = generate_synthetic_hand_landmarks(gesture_id, rng).astype(np.float32)
        lm_img = lm * 0.18 + np.array([0.50 + dx, 0.76, 0.0], dtype=np.float32)
        self.landmarks = lm_img
        self.features = extract_hand_features(lm_img)
        self.finger_states = analyze_finger_states(lm_img)

    def detect(self, frame_rgb):
        import numpy as np

        if self.landmarks is None:
            return {"detected": False}
        h, w = frame_rgb.shape[:2]
        px = (self.landmarks[:, :2] * np.array([w, h], dtype=np.float32)).astype(np.int32)
        x1, y1 = int(np.min(px[:, 0])), int(np.min(px[:, 1]))
        x2, y2 = int(np.max(px[:, 0])), int(np.max(px[:, 1]))
        return {
            "detected": True,
            "landmarks": self.landmarks.copy(),
            "pixel_landmarks": px,
            "bbox": (x1, y1, x2, y2),
            "handedness": "Right",
            "score": 0.99,
            "features": self.features.copy(),
            "finger_states": dict(self.finger_states),
        }

    def reset_smoothing(self):
        return None

    def close(self):
        return None


def test_engine_palm_arm_and_strict_neutral_reset_prevent_spam(monkeypatch):
    """Avuç Kapısı + Nötr Sıfırlama: Açık El görmeden komut atmaz, parmak değiştirerek zincir tetiklemez."""
    import numpy as np
    from config import LANDMARK_MODEL_PATH
    from core import engine as engine_mod
    from core.engine import AirTouchEngine

    tracker = _FakeGuardHandTracker()
    actions_executed = []
    monkeypatch.setattr(
        engine_mod.system_controller,
        "execute_action",
        lambda name: actions_executed.append(name) or f"Mock:{name}",
    )

    engine = AirTouchEngine(
        model_path=LANDMARK_MODEL_PATH,
        engine_mode="landmark",
        confidence_threshold=0.70,
        min_margin=0.15,
        cooldown_seconds=0.05,
        smoothing_window=3,
        required_stable_frames=3,
        guard_mode="palm_arm",
        strict_neutral_reset=True,
        max_hand_motion=0.20,
        palm_arm_seconds=6.0,
        hand_tracker=tracker,
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1) Doğrudan Yumruk (1) göster -> Avuç Kapısı kilitli kalmalı (tetiklenmemeli)
    tracker.set_pose(1, dx=0.0)
    for _ in range(5):
        rep = engine.process_frame(blank, trigger_enabled=True)
    assert actions_executed == []
    assert rep["status_state"] == "LOCKED_WAIT_PALM"

    # 2) Açık El (0) göster -> Kapı açılmalı (ARMED)
    tracker.set_pose(0, dx=0.0)
    for _ in range(5):
        rep = engine.process_frame(blank, trigger_enabled=True)
    assert rep["status_state"] in {"ARMED", "READY"}
    assert actions_executed == []

    # 3) Yumruk (1) sabit tut -> Tek atım tetiklenmeli ve kilitlenmeli
    tracker.set_pose(1, dx=0.0)
    statuses = []
    for _ in range(6):
        rep = engine.process_frame(blank, trigger_enabled=True)
        statuses.append(rep["status_state"])
    assert "TRIGGERED" in statuses
    assert actions_executed == ["play_pause"]

    # 4) Açık El'e dönmeden doğrudan 1 Parmak (2) pozuna geç -> zincir tetikleme engellenmeli
    tracker.set_pose(2, dx=0.0)
    for _ in range(6):
        rep_chain = engine.process_frame(blank, trigger_enabled=True)
    assert len(actions_executed) == 1
    assert rep_chain["status_state"] in {"LATCHED", "LOCKED_WAIT_PALM"}


def test_engine_motion_gate_and_voice_only_mode(monkeypatch):
    """Hızlı el hareketi MOVING ile bloke edilmeli; voice_only modunda el tetiklemesi kapalı kalmalıdır."""
    import numpy as np
    from config import LANDMARK_MODEL_PATH
    from core import engine as engine_mod
    from core.engine import AirTouchEngine

    tracker = _FakeGuardHandTracker()
    actions_executed = []
    monkeypatch.setattr(
        engine_mod.system_controller,
        "execute_action",
        lambda name: actions_executed.append(name) or f"Mock:{name}",
    )

    engine = AirTouchEngine(
        model_path=LANDMARK_MODEL_PATH,
        engine_mode="landmark",
        confidence_threshold=0.70,
        min_margin=0.15,
        cooldown_seconds=0.05,
        smoothing_window=3,
        required_stable_frames=3,
        guard_mode="strict_reset",
        strict_neutral_reset=True,
        max_hand_motion=0.12,
        hand_tracker=tracker,
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    # Önce frame0 konumu oturt, sonra max_hand_motion (0.12) üstü sıçrat (dx=0.30)
    tracker.set_pose(1, dx=0.0)
    engine.process_frame(blank, trigger_enabled=True)
    tracker.set_pose(1, dx=0.30)
    rep_move = engine.process_frame(blank, trigger_enabled=True)
    assert rep_move["status_state"] == "MOVING"
    assert actions_executed == []

    # voice_only modu: el ne gösterirse göstersin VOICE_ONLY döner ve 0 aksiyon çalıştırır
    engine.guard_mode = "voice_only"
    tracker.set_pose(1, dx=0.30)
    for _ in range(6):
        rep_vo = engine.process_frame(blank, trigger_enabled=True)
    assert rep_vo["status_state"] == "VOICE_ONLY"
    assert actions_executed == []

