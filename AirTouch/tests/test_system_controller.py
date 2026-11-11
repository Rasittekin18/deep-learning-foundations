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
        """execute_action fonksiyonu geçerli tüm eylemleri ilgili metotlara yönlendirmelidir."""
        ctrl = WindowsSystemController()
        # Donanıma gerçekten basmayı mockla (güvenli birim testi)
        monkeypatch.setattr(ctrl, "_press_key", lambda code: None)
        
        valid_actions = ["play_pause", "volume_up", "volume_down", "next_track", "prev_track", "mute"]
        for action in valid_actions:
            result = ctrl.execute_action(action)
            assert result is not None
            assert isinstance(result, str)

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

