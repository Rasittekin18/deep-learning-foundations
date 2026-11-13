"""
AirTouch Çoklu Ekran, Pencere ve Uygulama Bağlam Yöneticisi (window_manager.py)
Windows masaüstünde aktif uygulama penceresini ve bağlı monitörleri (Ekran 1 / Ekran 2 vb.)
gerçek zamanlı takip eder; YouTube, Sunum, Tarayıcı ve Çift Ekran geçişlerini yönetir.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


# AirTouch kendi pencerelerini kullanıcı uygulamalarından ayırmak için başlık önekleri
AIRTOUCH_WINDOW_MARKERS = ("airtouch",)

MEDIA_PROCESSES = {"spotify.exe", "vlc.exe", "wmplayer.exe", "musicbee.exe", "mpc-hc64.exe"}
BROWSER_PROCESSES = {"chrome.exe", "msedge.exe", "brave.exe", "firefox.exe", "opera.exe", "vivaldi.exe"}
PRESENTATION_PROCESSES = {"powerpnt.exe", "acrobat.exe", "acrord32.exe", "foxitpdfreader.exe"}


def classify_window_context(title: str, process_name: str = "") -> str:
    """Pencere başlığı ve süreç adından kullanım bağlamını (youtube/media/presentation/browser/workspace) çıkarır."""
    tl = (title or "").lower()
    pl = (process_name or "").lower()

    if "youtube" in tl or "twitch" in tl or "netflix" in tl or "disney+" in tl:
        return "youtube"
    if pl in MEDIA_PROCESSES or any(k in tl for k in ("spotify", "vlc media", "windows media")):
        return "media"
    if pl in PRESENTATION_PROCESSES or any(k in tl for k in ("powerpoint", ".pptx", "google slaytlar", "google slides", "slayt")):
        return "presentation"
    if pl in BROWSER_PROCESSES:
        return "browser"
    return "workspace"


def is_airtouch_window(title: str) -> bool:
    """Verilen pencere başlığının AirTouch'ın kendi HUD/terminal penceresine ait olup olmadığını döndürür."""
    tl = (title or "").lower()
    return any(marker in tl for marker in AIRTOUCH_WINDOW_MARKERS)


if sys.platform == "win32":
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_APPWINDOW = 0x00040000
    SW_RESTORE = 9
    SW_SHOW = 5
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOZORDER = 0x0004
    SWP_NOACTIVATE = 0x0010
    MONITOR_DEFAULTTONEAREST = 2

    class _RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class _MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", _RECT),
            ("rcWork", _RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    _MONITORENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(_RECT),
        wintypes.LPARAM,
    )
    _WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)


class WindowManager:
    """Çoklu monitör konumlandırma, pencere odaklama ve odak koruma yöneticisi."""

    def __init__(self):
        self.is_windows = sys.platform == "win32"
        self.last_user_hwnd: Optional[int] = None
        self.last_user_context: Dict[str, Any] = {
            "hwnd": None,
            "title": "Masaüstü",
            "process_name": "",
            "monitor_index": 1,
            "context": "workspace",
        }

    def get_monitors(self) -> List[Dict[str, Any]]:
        """Bağlı monitörleri birincil ekran ilk sırada olacak şekilde listeler."""
        if not self.is_windows:
            return [
                {
                    "index": 1,
                    "hmonitor": 1,
                    "is_primary": True,
                    "device": "DISPLAY1",
                    "bounds": (0, 0, 1920, 1080),
                    "work": (0, 0, 1920, 1040),
                }
            ]

        monitors: List[Dict[str, Any]] = []

        def _cb(hmon, hdc, lprect, lparam):
            info = _MONITORINFOEXW()
            info.cbSize = ctypes.sizeof(_MONITORINFOEXW)
            if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                monitors.append(
                    {
                        "hmonitor": int(hmon),
                        "is_primary": bool(info.dwFlags & 1),
                        "device": info.szDevice,
                        "bounds": (
                            int(info.rcMonitor.left),
                            int(info.rcMonitor.top),
                            int(info.rcMonitor.right),
                            int(info.rcMonitor.bottom),
                        ),
                        "work": (
                            int(info.rcWork.left),
                            int(info.rcWork.top),
                            int(info.rcWork.right),
                            int(info.rcWork.bottom),
                        ),
                    }
                )
            return True

        user32.EnumDisplayMonitors(None, None, _MONITORENUMPROC(_cb), 0)
        # Birincil ekranı index 1 yap, diğerlerini soldan sağa sırala
        monitors.sort(key=lambda m: (not m["is_primary"], m["bounds"][0], m["bounds"][1]))
        for idx, m in enumerate(monitors, start=1):
            m["index"] = idx
        return monitors or [
            {
                "index": 1,
                "hmonitor": 0,
                "is_primary": True,
                "device": "DISPLAY1",
                "bounds": (0, 0, 1920, 1080),
                "work": (0, 0, 1920, 1040),
            }
        ]

    def _get_window_title(self, hwnd: int) -> str:
        if not self.is_windows or not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value.strip()

    def _get_process_name(self, hwnd: int) -> str:
        if not self.is_windows or not hwnd or psutil is None:
            return ""
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""
        try:
            return psutil.Process(pid.value).name()
        except Exception:
            return ""

    def _get_monitor_index_for_hwnd(self, hwnd: int, monitors: Optional[List[Dict[str, Any]]] = None) -> int:
        if not self.is_windows or not hwnd:
            return 1
        mons = monitors if monitors is not None else self.get_monitors()
        hmon = int(user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST))
        for m in mons:
            if m["hmonitor"] == hmon:
                return int(m["index"])
        return 1

    def get_foreground_context(self) -> Dict[str, Any]:
        """Şu anda odakta olan pencere, süreç, monitör ve uygulama bağlam bilgisini döndürür."""
        if not self.is_windows:
            return dict(self.last_user_context)

        hwnd = int(user32.GetForegroundWindow() or 0)
        title = self._get_window_title(hwnd)
        proc = self._get_process_name(hwnd)
        mon_idx = self._get_monitor_index_for_hwnd(hwnd)
        own = is_airtouch_window(title)
        ctx = classify_window_context(title, proc)

        info = {
            "hwnd": hwnd,
            "title": title or (proc or "Masaüstü"),
            "process_name": proc,
            "monitor_index": mon_idx,
            "context": ctx,
            "is_airtouch": own,
        }
        if hwnd and title and not own:
            self.last_user_hwnd = hwnd
            self.last_user_context = dict(info)
        return info

    def update_last_user_window(self) -> Dict[str, Any]:
        """Ön plandaki gerçek kullanıcı penceresini takip eder (AirTouch HUD odağını yok sayar)."""
        info = self.get_foreground_context()
        if not info.get("is_airtouch"):
            return info
        return dict(self.last_user_context)

    def _activate_hwnd(self, hwnd: int) -> bool:
        """Verilen Windows penceresini güvenli şekilde ön plana ve odağa getirir."""
        if not self.is_windows or not hwnd or not user32.IsWindow(hwnd):
            return False
        try:
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, SW_RESTORE)
            fg_hwnd = user32.GetForegroundWindow()
            fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
            cur_tid = kernel32.GetCurrentThreadId()
            tgt_tid = user32.GetWindowThreadProcessId(hwnd, None)

            if fg_tid and fg_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, fg_tid, True)
            if tgt_tid and tgt_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, tgt_tid, True)

            user32.BringWindowToTop(hwnd)
            ok = bool(user32.SetForegroundWindow(hwnd))
            if not ok:
                user32.SwitchToThisWindow(hwnd, True)
                ok = True
        except Exception:
            ok = False
        finally:
            try:
                fg_hwnd = user32.GetForegroundWindow()
                fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
                cur_tid = kernel32.GetCurrentThreadId()
                tgt_tid = user32.GetWindowThreadProcessId(hwnd, None)
                if fg_tid and fg_tid != cur_tid:
                    user32.AttachThreadInput(cur_tid, fg_tid, False)
                if tgt_tid and tgt_tid != cur_tid:
                    user32.AttachThreadInput(cur_tid, tgt_tid, False)
            except Exception:
                pass
        return ok

    def ensure_user_focus(self) -> bool:
        """
        Eğer o an AirTouch penceresi odaktaysa tuşların boşa gitmemesi için odağı
        son kullanıcı penceresine (YouTube, PowerPoint, IDE vb.) geri taşır.
        """
        if not self.is_windows:
            return False
        fg = self.get_foreground_context()
        if not fg.get("is_airtouch"):
            return True

        if self.last_user_hwnd and user32.IsWindow(self.last_user_hwnd):
            return self._activate_hwnd(self.last_user_hwnd)

        windows = self.list_taskbar_windows()
        if windows:
            return self._activate_hwnd(windows[0]["hwnd"])
        return False

    def list_taskbar_windows(self, include_airtouch: bool = False) -> List[Dict[str, Any]]:
        """Z-sırasına göre gerçek görünür masaüstü uygulama pencerelerini listeler."""
        if not self.is_windows:
            return []

        mons = self.get_monitors()
        results: List[Dict[str, Any]] = []
        skip_titles = {"program manager", "windows input experience", "settings", "microsoft text input application"}

        def _cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if (ex_style & WS_EX_TOOLWINDOW) and not (ex_style & WS_EX_APPWINDOW):
                return True
            title = self._get_window_title(hwnd)
            if not title or title.lower() in skip_titles:
                return True
            if not include_airtouch and is_airtouch_window(title):
                return True
            rect = _RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if (rect.right - rect.left) < 120 or (rect.bottom - rect.top) < 80:
                return True

            proc = self._get_process_name(hwnd)
            mon_idx = self._get_monitor_index_for_hwnd(hwnd, monitors=mons)
            results.append(
                {
                    "hwnd": int(hwnd),
                    "title": title,
                    "process_name": proc,
                    "monitor_index": mon_idx,
                    "context": classify_window_context(title, proc),
                    "is_minimized": bool(user32.IsIconic(hwnd)),
                }
            )
            return True

        user32.EnumWindows(_WNDENUMPROC(_cb), 0)
        return results

    def focus_other_screen(self) -> Tuple[bool, str]:
        """Odağı aktif pencereden farklı bir monitördeki (Ekran 1 <-> Ekran 2) uygulamaya taşır."""
        if not self.is_windows:
            return False, "Diğer Ekrana Geçildi (Simülasyon)"

        mons = self.get_monitors()
        if len(mons) < 2:
            # Tek ekran varsa sonraki uygulamaya geç
            return self.switch_next_app()

        cur_ctx = self.update_last_user_window()
        cur_mon = int(cur_ctx.get("monitor_index", 1))
        windows = self.list_taskbar_windows()

        # Önce diğer monitörde minimize olmayan pencerelere öncelik ver
        other_non_min = [w for w in windows if w["monitor_index"] != cur_mon and not w["is_minimized"]]
        other_any = [w for w in windows if w["monitor_index"] != cur_mon]
        target = (other_non_min or other_any or [None])[0]

        if target is None:
            return False, "Diğer ekranda açık pencere bulunamadı"

        if self._activate_hwnd(target["hwnd"]):
            self.last_user_hwnd = target["hwnd"]
            self.last_user_context = dict(target)
            short_title = target["title"][:36]
            return True, f"Ekran {target['monitor_index']} Odaklandı: {short_title}"
        return False, "Diğer ekrana odaklanılamadı"

    def switch_next_app(self, reverse: bool = False) -> Tuple[bool, str]:
        """Açık uygulamalar arasında doğrudan pencere geçişi yapar."""
        if not self.is_windows:
            return False, "Uygulama Değiştirildi (Simülasyon)"

        windows = self.list_taskbar_windows()
        if len(windows) < 2:
            if windows:
                self._activate_hwnd(windows[0]["hwnd"])
                return True, f"Odaklandı: {windows[0]['title'][:36]}"
            return False, "Geçiş yapılacak açık pencere yok"

        cur = self.update_last_user_window()
        cur_hwnd = cur.get("hwnd")
        hwnds = [w["hwnd"] for w in windows]
        if cur_hwnd in hwnds:
            idx = hwnds.index(cur_hwnd)
            step = -1 if reverse else 1
            target = windows[(idx + step) % len(windows)]
        else:
            target = windows[1] if not reverse else windows[-1]

        if self._activate_hwnd(target["hwnd"]):
            self.last_user_hwnd = target["hwnd"]
            self.last_user_context = dict(target)
            return True, f"Geçildi (Ekran {target['monitor_index']}): {target['title'][:36]}"
        return False, "Uygulama değiştirilemedi"

    def focus_app_by_keywords(self, keywords: Tuple[str, ...], display_name: str = "Uygulama") -> Tuple[bool, str]:
        """Başlık veya süreç adına göre belirli bir uygulamayı bulup ön plana getirir."""
        if not self.is_windows:
            return False, f"{display_name} Odaklandı (Simülasyon)"

        windows = self.list_taskbar_windows()
        kws = tuple(k.lower() for k in keywords)
        # İlk geçiş: pencere başlığında geçenler (örn. "YouTube"), sonra süreç adı
        matches = [
            w for w in windows if any(k in w["title"].lower() for k in kws)
        ] or [
            w for w in windows if any(k in w["process_name"].lower() for k in kws)
        ]

        if not matches:
            return False, f"Açık {display_name} penceresi bulunamadı"

        target = matches[0]
        if self._activate_hwnd(target["hwnd"]):
            self.last_user_hwnd = target["hwnd"]
            self.last_user_context = dict(target)
            return True, f"{display_name} (Ekran {target['monitor_index']}) Odaklandı"
        return False, f"{display_name} penceresine odaklanılamadı"

    def position_window_on_monitor(
        self,
        window_title: str,
        monitor_index: int = 1,
        width: int = 960,
        height: int = 540,
        corner: str = "bottom_right",
    ) -> int:
        """OpenCV HUD penceresini odağı çalmadan seçilen monitörün köşesine yerleştirir."""
        if not self.is_windows:
            return 1

        mons = self.get_monitors()
        idx = ((int(monitor_index) - 1) % len(mons)) + 1
        mon = next((m for m in mons if m["index"] == idx), mons[0])
        wl, wt, wr, wb = mon["work"]
        avail_w = max(320, wr - wl)
        avail_h = max(240, wb - wt)
        win_w = min(width, avail_w - 20)
        win_h = min(height, avail_h - 40)

        if corner == "top_right":
            x = wr - win_w - 16
            y = wt + 16
        elif corner == "center":
            x = wl + max(0, (avail_w - win_w) // 2)
            y = wt + max(0, (avail_h - win_h) // 2)
        else:  # bottom_right
            x = wr - win_w - 16
            y = wb - win_h - 16

        hwnd = user32.FindWindowW(None, window_title)
        if hwnd:
            user32.SetWindowPos(hwnd, 0, int(x), int(y), int(win_w), int(win_h), SWP_NOZORDER | SWP_NOACTIVATE)
        return idx


window_manager = WindowManager()

