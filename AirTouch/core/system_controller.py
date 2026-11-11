"""
AirTouch Sistem & Medya Denetleyicisi (system_controller.py)
Windows yerel API'si (ctypes) kullanılarak sıfır gecikmeli medya kontrolü sağlar.
"""
import sys
import time

# Windows sanal tuş kodları (Virtual Key Codes)
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_KEYUP = 0x0002

class WindowsSystemController:
    """Windows işletim sistemi için donanım seviyesinde medya denetleyicisi."""
    
    def __init__(self):
        self.is_windows = sys.platform == "win32"
        if self.is_windows:
            import ctypes
            self._user32 = ctypes.windll.user32
        else:
            self._user32 = None

    def _press_key(self, vk_code: int):
        """Tuşa bas ve hemen bırak."""
        if not self.is_windows or not self._user32:
            print(f"[Simülasyon] Tuşa basıldı: {hex(vk_code)}")
            return
        # Key Down
        self._user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.05)
        # Key Up
        self._user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

    def play_pause(self):
        """Medyayı oynat veya duraklat (YouTube, Spotify, VLC vb.)."""
        self._press_key(VK_MEDIA_PLAY_PAUSE)
        return "Oynat / Durdur Tetiklendi ⏯️"

    def volume_up(self, steps: int = 2):
        """Windows ana sesini artır."""
        for _ in range(steps):
            self._press_key(VK_VOLUME_UP)
            time.sleep(0.03)
        return f"Ses Artırıldı (+{steps * 2}%) 🔊"

    def volume_down(self, steps: int = 2):
        """Windows ana sesini azalt."""
        for _ in range(steps):
            self._press_key(VK_VOLUME_DOWN)
            time.sleep(0.03)
        return f"Ses Kısıldı (-{steps * 2}%) 🔉"

    def next_track(self):
        """Sonraki şarkıya veya videoya geç."""
        self._press_key(VK_MEDIA_NEXT_TRACK)
        return "Sonraki Parçaya Geçildi ⏭️"

    def prev_track(self):
        """Önceki şarkıya veya videoya dön."""
        self._press_key(VK_MEDIA_PREV_TRACK)
        return "Önceki Parçaya Dönüldü ⏮️"

    def mute(self):
        """Sesi tamamen kapat / aç."""
        self._press_key(VK_VOLUME_MUTE)
        return "Sessize Alındı / Açıldı 🔇"

    def execute_action(self, action_name: str):
        """Eylem ismine göre ilgili metodu dinamik çalıştırır."""
        actions = {
            "play_pause": self.play_pause,
            "volume_up": self.volume_up,
            "volume_down": self.volume_down,
            "next_track": self.next_track,
            "prev_track": self.prev_track,
            "mute": self.mute
        }
        handler = actions.get(action_name)
        if handler:
            return handler()
        return None

# Singleton örneği
system_controller = WindowsSystemController()

if __name__ == "__main__":
    print("Windows Medya Kontrolcüsü Test Ediliyor...")
    ctrl = WindowsSystemController()
    print("1. Ses artırma testi:")
    print(ctrl.volume_up())
    time.sleep(0.5)
    print("2. Ses kısma testi:")
    print(ctrl.volume_down())
    print("Medya motoru başarıyla test edildi.")

