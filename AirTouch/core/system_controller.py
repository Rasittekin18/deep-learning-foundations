"""
AirTouch Sistem, Medya, YouTube, Sunum & Çift Ekran Denetleyicisi (system_controller.py)
Windows yerel SendInput API'si ve WindowManager entegrasyonu ile düşük gecikmeli medya,
YouTube, sunum, sekme ve çoklu monitör uygulama geçiş kontrolü sağlar.
"""
import ctypes
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from typing import Iterable, Optional, Tuple

from core.window_manager import window_manager

# Windows sanal tuş kodları (Virtual Key Codes)
VK_TAB = 0x09
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12          # ALT
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_C = 0x43
VK_F = 0x46
VK_J = 0x4A
VK_K = 0x4B
VK_L = 0x4C
VK_N = 0x4E
VK_LWIN = 0x5B

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_WHEEL = 0x0800
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

EXTENDED_VKS = {
    VK_LEFT,
    VK_UP,
    VK_RIGHT,
    VK_DOWN,
    VK_LWIN,
    VK_VOLUME_MUTE,
    VK_VOLUME_DOWN,
    VK_VOLUME_UP,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PREV_TRACK,
    VK_MEDIA_STOP,
    VK_MEDIA_PLAY_PAUSE,
}


if sys.platform == "win32":
    from ctypes import wintypes

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", _MOUSEINPUT),
            ("ki", _KEYBDINPUT),
            ("hi", _HARDWAREINPUT),
        ]

    class _INPUT(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.DWORD),
            ("union", _INPUT_UNION),
        ]


_YT_RENDERER_RE = re.compile(r'"videoRenderer"\s*:\s*\{\s*"videoId"\s*:\s*"([a-zA-Z0-9_-]{11})"')
_YT_GENERIC_ID_RE = re.compile(r'"videoId"\s*:\s*"([a-zA-Z0-9_-]{11})"')


def resolve_youtube_video_url(query: str, timeout: float = 4.0) -> Tuple[str, Optional[str]]:
    """
    Verilen arama sorgusu için en üst YouTube video izleme URL'sini (watch?v=...) çözer.
    İnternet yoksa veya video ID çıkarılamazsa güvenli şekilde arama sonuç sayfasına düşer.
    """
    clean_q = " ".join((query or "").split())
    if not clean_q:
        return "https://www.youtube.com", None

    search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(clean_q)
    try:
        req = urllib.request.Request(
            search_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        m = _YT_RENDERER_RE.search(html) or _YT_GENERIC_ID_RE.search(html)
        if m:
            vid = m.group(1)
            return f"https://www.youtube.com/watch?v={vid}", vid
    except Exception:
        pass
    return search_url, None


class WindowsSystemController:
    """Windows için donanım seviyesinde medya, YouTube ve çoklu ekran denetleyicisi."""

    def __init__(self):
        self.is_windows = sys.platform == "win32"
        self.auto_restore_focus = True
        if self.is_windows:
            self._user32 = ctypes.windll.user32
        else:
            self._user32 = None

    def _build_key_input(self, vk_code: int, key_up: bool = False):
        ext_flag = KEYEVENTF_EXTENDEDKEY if vk_code in EXTENDED_VKS else 0
        flags = ext_flag | (KEYEVENTF_KEYUP if key_up else 0)
        scan = self._user32.MapVirtualKeyW(vk_code, 0)
        inp = _INPUT(type=INPUT_KEYBOARD)
        inp.union.ki = _KEYBDINPUT(
            wVk=vk_code,
            wScan=scan,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        return inp

    def _prepare_user_focus(self):
        """AirTouch HUD odaktaysa uygulama tuşu göndermeden önce kullanıcı penceresine döner."""
        if not self.is_windows or not self.auto_restore_focus:
            return
        try:
            fg = window_manager.get_foreground_context()
            if fg.get("is_airtouch") and window_manager.ensure_user_focus():
                time.sleep(0.03)
        except Exception:
            pass

    def _press_key(self, vk_code: int):
        """Tek tuşa bas ve hemen bırak (SendInput kullanır; medya/yön tuşları için extended flag içerir)."""
        if not self.is_windows or not self._user32:
            print(f"[Simülasyon] Tuşa basıldı: {hex(vk_code)}")
            return

        down = self._build_key_input(vk_code, key_up=False)
        up = self._build_key_input(vk_code, key_up=True)
        inputs = (_INPUT * 2)(down, up)
        sent = self._user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(_INPUT))
        if sent == 0:
            ext_flag = KEYEVENTF_EXTENDEDKEY if vk_code in EXTENDED_VKS else 0
            scan = self._user32.MapVirtualKeyW(vk_code, 0)
            self._user32.keybd_event(vk_code, scan, ext_flag, 0)
            time.sleep(0.03)
            self._user32.keybd_event(vk_code, scan, ext_flag | KEYEVENTF_KEYUP, 0)

    def _press_combo(self, modifiers: Iterable[int], vk_code: int):
        """Çoklu tuş kombinasyonu gönderir (örn. Alt+Tab, Ctrl+Tab, Win+Shift+Sağ)."""
        mods = list(modifiers)
        if not self.is_windows or not self._user32:
            print(f"[Simülasyon] Combo: {[hex(m) for m in mods]} + {hex(vk_code)}")
            return

        seq = [self._build_key_input(m, key_up=False) for m in mods]
        seq.append(self._build_key_input(vk_code, key_up=False))
        seq.append(self._build_key_input(vk_code, key_up=True))
        for m in reversed(mods):
            seq.append(self._build_key_input(m, key_up=True))

        arr = (_INPUT * len(seq))(*seq)
        self._user32.SendInput(len(seq), ctypes.byref(arr), ctypes.sizeof(_INPUT))

    def _scroll_wheel(self, delta: int):
        """Fare tekerleğini yukarı (>0) veya aşağı (<0) kaydırır."""
        if not self.is_windows or not self._user32:
            print(f"[Simülasyon] Scroll: {delta}")
            return
        evt = _INPUT(type=INPUT_MOUSE)
        evt.union.mi = _MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=ctypes.c_uint32(delta).value,
            dwFlags=MOUSEEVENTF_WHEEL,
            time=0,
            dwExtraInfo=0,
        )
        self._user32.SendInput(1, ctypes.byref(evt), ctypes.sizeof(_INPUT))

    # --- Genel Medya Kontrolleri ---
    def play_pause(self):
        """Medyayı oynat veya duraklat (YouTube, Spotify, VLC vb.)."""
        self._press_key(VK_MEDIA_PLAY_PAUSE)
        return "Oynat / Durdur Tetiklendi"

    def volume_up(self, steps: int = 2):
        """Windows ana sesini artır."""
        for i in range(steps):
            self._press_key(VK_VOLUME_UP)
            if i + 1 < steps:
                time.sleep(0.02)
        return f"Ses Artırıldı (+{steps * 2}%)"

    def volume_down(self, steps: int = 2):
        """Windows ana sesini azalt."""
        for i in range(steps):
            self._press_key(VK_VOLUME_DOWN)
            if i + 1 < steps:
                time.sleep(0.02)
        return f"Ses Kısıldı (-{steps * 2}%)"

    def next_track(self):
        """Sonraki şarkıya veya videoya geç."""
        self._press_key(VK_MEDIA_NEXT_TRACK)
        return "Sonraki Parçaya Geçildi"

    def prev_track(self):
        """Önceki şarkıya veya videoya dön."""
        self._press_key(VK_MEDIA_PREV_TRACK)
        return "Önceki Parçaya Dönüldü"

    def mute(self):
        """Sesi tamamen kapat / aç."""
        self._press_key(VK_VOLUME_MUTE)
        return "Sessize Alındı / Açıldı"

    # --- Uygulama Odaklı Kontroller (YouTube / Sunum / Tarayıcı) ---
    def fullscreen(self):
        """Aktif video oynatıcıda tam ekran aç/kapat (F tuşu)."""
        self._prepare_user_focus()
        self._press_key(VK_F)
        return "Tam Ekran Değiştirildi (F)"

    def next_slide(self):
        """Sunumda/tarayıcıda sonraki slayt veya ileri sarma (Sağ Ok)."""
        self._prepare_user_focus()
        self._press_key(VK_RIGHT)
        return "Sonraki Slayt / İleri (Sağ Ok)"

    def prev_slide(self):
        """Sunumda/tarayıcıda önceki slayt veya geri sarma (Sol Ok)."""
        self._prepare_user_focus()
        self._press_key(VK_LEFT)
        return "Önceki Slayt / Geri (Sol Ok)"

    def space_play(self):
        """Space tuşu (Video duraklat / sunum ilerlet)."""
        self._prepare_user_focus()
        self._press_key(VK_SPACE)
        return "Boşluk Tuşu (Space) Basıldı"

    def scroll_up(self, steps: int = 3):
        """Sayfayı yukarı kaydırır."""
        self._prepare_user_focus()
        self._scroll_wheel(120 * steps)
        return "Sayfa Yukarı Kaydırıldı"

    def scroll_down(self, steps: int = 3):
        """Sayfayı aşağı kaydırır."""
        self._prepare_user_focus()
        self._scroll_wheel(-120 * steps)
        return "Sayfa Aşağı Kaydırıldı"

    def esc(self):
        """ESC tuşuna basar."""
        self._prepare_user_focus()
        self._press_key(VK_ESCAPE)
        return "ESC Tuşuna Basıldı"

    # --- YouTube Akıllı Kontroller ---
    def yt_forward_10(self):
        """YouTube videosunu 10 saniye ileri sarar (L tuşu)."""
        self._prepare_user_focus()
        self._press_key(VK_L)
        return "YouTube +10 sn İleri (L)"

    def yt_rewind_10(self):
        """YouTube videosunu 10 saniye geri sarar (J tuşu)."""
        self._prepare_user_focus()
        self._press_key(VK_J)
        return "YouTube -10 sn Geri (J)"

    def yt_next_video(self):
        """YouTube oynatma listesinde sonraki videoya geçer (Shift+N)."""
        self._prepare_user_focus()
        self._press_combo((VK_SHIFT,), VK_N)
        return "YouTube Sonraki Video (Shift+N)"

    def yt_captions(self):
        """YouTube altyazılarını açar/kapatır (C tuşu)."""
        self._prepare_user_focus()
        self._press_key(VK_C)
        return "YouTube Altyazı Değiştirildi (C)"

    # --- Çift Ekran & Uygulamalar Arası Geçişler ---
    def next_app(self):
        """Açık uygulamalar arasında ileri geçiş yapar."""
        ok, msg = window_manager.switch_next_app(reverse=False)
        if not ok and self.is_windows:
            self._press_combo((VK_MENU,), VK_TAB)
            return "Sonraki Uygulamaya Geçildi (Alt+Tab)"
        return msg

    def prev_app(self):
        """Açık uygulamalar arasında geri geçiş yapar."""
        ok, msg = window_manager.switch_next_app(reverse=True)
        if not ok and self.is_windows:
            self._press_combo((VK_MENU, VK_SHIFT), VK_TAB)
            return "Önceki Uygulamaya Dönüldü"
        return msg

    def focus_other_screen(self):
        """İki ekranlı kurulumda odağı diğer monitördeki pencereye taşır."""
        _, msg = window_manager.focus_other_screen()
        return msg

    def move_window_other_screen(self):
        """Aktif kullanıcı penceresini diğer ekrana taşır (Win+Shift+Sağ)."""
        self._prepare_user_focus()
        self._press_combo((VK_LWIN, VK_SHIFT), VK_RIGHT)
        return "Pencere Diğer Ekrana Taşındı"

    def browser_next_tab(self):
        """Tarayıcıda sonraki sekmeye geçer (Ctrl+Tab)."""
        self._prepare_user_focus()
        self._press_combo((VK_CONTROL,), VK_TAB)
        return "Sonraki Sekmeye Geçildi"

    def browser_prev_tab(self):
        """Tarayıcıda önceki sekmeye geçer (Ctrl+Shift+Tab)."""
        self._prepare_user_focus()
        self._press_combo((VK_CONTROL, VK_SHIFT), VK_TAB)
        return "Önceki Sekmeye Geçildi"

    def focus_youtube(self):
        """YouTube veya tarayıcı penceresini bulup odaklanır; açık değilse YouTube'u açar."""
        ok, msg = window_manager.focus_app_by_keywords(("youtube",), display_name="YouTube")
        if not ok:
            ok_br, msg = window_manager.focus_app_by_keywords(("chrome", "msedge", "brave", "firefox"), display_name="Tarayıcı")
            if not ok_br:
                webbrowser.open("https://www.youtube.com")
                return "YouTube Tarayıcıda Açıldı"
        return msg

    def open_youtube_video(self, query: str = "", block: bool = False) -> str:
        """
        Sesli veya metinli YouTube sorgusu için en üst videoyu tarayıcıda doğrudan açar.
        Kamerayı dondurmamak için ağ çözümlemesini varsayılan olarak arka plan iş parçacığında yürütür.
        """
        clean_q = " ".join((query or "").split())
        if not clean_q:
            return self.focus_youtube()

        def _worker():
            url, _ = resolve_youtube_video_url(clean_q)
            try:
                webbrowser.open(url, new=2)
            except Exception:
                pass

        if block:
            _worker()
        else:
            threading.Thread(target=_worker, daemon=True).start()
        return f"YouTube'da Açılıyor: {clean_q}"

    def focus_spotify(self):
        """Spotify penceresini bulup odaklanır."""
        _, msg = window_manager.focus_app_by_keywords(("spotify",), display_name="Spotify")
        return msg

    def snap_left(self):
        """Aktif pencereyi ekranın sol yarısına yaslar (Win+Sol)."""
        self._prepare_user_focus()
        self._press_combo((VK_LWIN,), VK_LEFT)
        return "Pencere Sola Yaslandı"

    def snap_right(self):
        """Aktif pencereyi ekranın sağ yarısına yaslar (Win+Sağ)."""
        self._prepare_user_focus()
        self._press_combo((VK_LWIN,), VK_RIGHT)
        return "Pencere Sağa Yaslandı"

    def execute_action(self, action_name: str) -> Optional[str]:
        """Eylem ismine göre ilgili metodu dinamik çalıştırır."""
        if not action_name or action_name == "none":
            return None
        actions = {
            "play_pause": self.play_pause,
            "volume_up": self.volume_up,
            "volume_down": self.volume_down,
            "next_track": self.next_track,
            "prev_track": self.prev_track,
            "mute": self.mute,
            "fullscreen": self.fullscreen,
            "next_slide": self.next_slide,
            "prev_slide": self.prev_slide,
            "space_play": self.space_play,
            "scroll_up": self.scroll_up,
            "scroll_down": self.scroll_down,
            "esc": self.esc,
            "yt_forward_10": self.yt_forward_10,
            "yt_rewind_10": self.yt_rewind_10,
            "yt_next_video": self.yt_next_video,
            "yt_captions": self.yt_captions,
            "next_app": self.next_app,
            "prev_app": self.prev_app,
            "focus_other_screen": self.focus_other_screen,
            "move_window_other_screen": self.move_window_other_screen,
            "browser_next_tab": self.browser_next_tab,
            "browser_prev_tab": self.browser_prev_tab,
            "focus_youtube": self.focus_youtube,
            "focus_spotify": self.focus_spotify,
            "snap_left": self.snap_left,
            "snap_right": self.snap_right,
        }
        handler = actions.get(action_name)
        if handler:
            return handler()
        return None


# Singleton örneği
system_controller = WindowsSystemController()
