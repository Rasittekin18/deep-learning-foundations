"""
AirTouch Dizüstü Kamera & Görüntü Akış Yöneticisi (camera.py)
Laptop varsayılan web kamerasını (index 0) Windows üzerinde düşük gecikmeyle (DirectShow)
açar, arka planda kesintisiz kare okur ve loş ortamlarda otomatik ışık iyileştirme (CLAHE) sunar.
"""
import sys
import threading
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


def parse_camera_source(source: Union[int, str, None]) -> Union[int, str]:
    """Kamera kaynağını normalize eder (varsayılan: 0 - Laptop kamerası)."""
    if source is None:
        return 0
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    return s


def open_video_capture(
    source: Union[int, str] = 0,
    width: int = 960,
    height: int = 540,
) -> Optional[cv2.VideoCapture]:
    """
    Kamera kaynağını en düşük gecikmeli backend ile açar.
    Laptop kameralarında Windows'ta önce CAP_DSHOW denenir, başarısız olursa varsayılan backend kullanılır.
    """
    src = parse_camera_source(source)
    backends = []
    if isinstance(src, int) and sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(src, backend)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(src, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
            # İlk karenin okunabildiğini doğrula
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return cap
            cap.release()
        elif cap is not None:
            cap.release()
    return None


def list_available_cameras(max_index: int = 3) -> List[int]:
    """Sistemde bağlı ve erişilebilir kamera indekslerini tarar."""
    found: List[int] = []
    for idx in range(max_index):
        cap = open_video_capture(idx)
        if cap is not None:
            found.append(idx)
            cap.release()
    return found


def apply_low_light_boost(bgr_frame: np.ndarray, brightness_threshold: float = 95.0) -> Tuple[np.ndarray, bool]:
    """
    Ortam karanlıksa (ortalama parlaklık < 95) sadece parlaklık (L) kanalına
    yumuşak CLAHE uygulayarak el konturlarını belirginleştirir.
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return bgr_frame, False
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    mean_l = float(np.mean(l_chan))
    if mean_l >= brightness_threshold:
        return bgr_frame, False
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_boosted = clahe.apply(l_chan)
    # Aşırı yapay görüntüyü önlemek için %65 karışım
    l_merged = cv2.addWeighted(l_chan, 0.35, l_boosted, 0.65, 0)
    merged_lab = cv2.merge((l_merged, a_chan, b_chan))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR), True


class CameraStream:
    """
    Kuyruk birikmesini (buffer lag) ortadan kaldıran çoklu iş parçacıklı (threaded) kamera okuyucusu.
    """

    def __init__(self, source: Union[int, str] = 0, width: int = 960, height: int = 540):
        self.source = parse_camera_source(source)
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._last_read_time = 0.0

    def start(self) -> bool:
        if self._running:
            return True
        self._cap = open_video_capture(self.source, width=self.width, height=self.height)
        if self._cap is None:
            return False
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._latest_frame = frame
            self._last_read_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def _reader_loop(self):
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._last_read_time = time.time()
            else:
                time.sleep(0.02)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    @property
    def is_running(self) -> bool:
        return self._running and self._cap is not None

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        with self._lock:
            self._latest_frame = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

