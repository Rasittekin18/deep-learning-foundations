"""
Windows Medya ve Sistem Denetleyicisi Mimari Testleri (test_system_controller.py)
"""
import pytest
from core.system_controller import (
    WindowsSystemController, 
    system_controller,
    VK_VOLUME_UP, 
    VK_VOLUME_DOWN, 
    VK_MEDIA_PLAY_PAUSE,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PREV_TRACK,
    VK_VOLUME_MUTE
)

class TestSystemControllerArchitecture:
    
    def test_singleton_instance_exists(self):
        """Sistem kontrolcüsü tekil (singleton) nesne olarak başlatılmış olmalıdır."""
        assert system_controller is not None
        assert isinstance(system_controller, WindowsSystemController)

    def test_virtual_key_constants_integrity(self):
        """Windows API sanal tuş kodları standart Microsoft değerleriyle uyuşmalıdır."""
        assert VK_VOLUME_MUTE == 0xAD
        assert VK_VOLUME_DOWN == 0xAE
        assert VK_VOLUME_UP == 0xAF
        assert VK_MEDIA_NEXT_TRACK == 0xB0
        assert VK_MEDIA_PREV_TRACK == 0xB1
        assert VK_MEDIA_PLAY_PAUSE == 0xB3

    def test_execute_action_dispatch_valid(self, monkeypatch):
        """execute_action fonksiyonu tüm medya, YouTube, sunum ve çift ekran eylemlerini desteklemelidir."""
        ctrl = WindowsSystemController()
        # Donanıma gerçekten basmayı ve pencere odağını değiştirmeyi mockla (güvenli birim testi)
        monkeypatch.setattr(ctrl, "_press_key", lambda code: None)
        monkeypatch.setattr(ctrl, "_press_combo", lambda mods, code: None)
        monkeypatch.setattr(ctrl, "_scroll_wheel", lambda delta: None)
        monkeypatch.setattr(ctrl, "_prepare_user_focus", lambda: None)

        from core import system_controller as sc_mod
        monkeypatch.setattr(sc_mod.window_manager, "switch_next_app", lambda reverse=False: (True, "Mock:AppSwitch"))
        monkeypatch.setattr(sc_mod.window_manager, "focus_other_screen", lambda: (True, "Mock:OtherScreen"))
        monkeypatch.setattr(sc_mod.window_manager, "focus_app_by_keywords", lambda kws, display_name="Uyg": (True, f"Mock:{display_name}"))

        valid_actions = [
            "play_pause", "volume_up", "volume_down", "next_track",
            "prev_track", "mute", "fullscreen", "next_slide",
            "prev_slide", "space_play", "scroll_up", "scroll_down", "esc",
            "yt_forward_10", "yt_rewind_10", "yt_next_video", "yt_captions",
            "next_app", "prev_app", "focus_other_screen", "move_window_other_screen",
            "browser_next_tab", "browser_prev_tab", "focus_youtube", "focus_spotify",
            "snap_left", "snap_right",
        ]
        for action in valid_actions:
            result = ctrl.execute_action(action)
            assert result is not None
            assert isinstance(result, str)

        assert ctrl.execute_action("none") is None

    def test_execute_action_dispatch_invalid(self):
        """Tanımsız bir eylem istendiğinde hata fırlatmadan None dönmelidir."""
        ctrl = WindowsSystemController()
        result = ctrl.execute_action("non_existent_action_xyz")
        assert result is None

    def test_volume_steps_multiplier(self, monkeypatch):
        """Ses artırma ve kısma metotları verilen adım sayısı kadar çağrılmalıdır."""
        ctrl = WindowsSystemController()
        press_count = 0
        
        def mock_press(code):
            nonlocal press_count
            press_count += 1

        monkeypatch.setattr(ctrl, "_press_key", mock_press)
        
        # 3 adım artır -> 3 kez tuş basılmalı
        ctrl.volume_up(steps=3)
        assert press_count == 3
        
        press_count = 0
        ctrl.volume_down(steps=5)
        assert press_count == 5

