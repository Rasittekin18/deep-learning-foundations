"""
AirTouch El İskeleti Takip ve Biyomekanik Öznitelik Çıkarıcı (hand_tracker.py)
MediaPipe HandLandmarker ile 21 adet 3D el eklemini gerçek zamanlı takip eder,
ölçek/konum/aynalama bağımsız 88 boyutlu öznitelik vektörü ve parmak durumları üretir.
"""
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import HAND_LANDMARKER_TASK_PATH

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# 21 nokta el iskeleti kemik bağlantıları
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Başparmak
    (0, 5), (5, 6), (6, 7), (7, 8),        # İşaret parmağı
    (0, 9), (9, 10), (10, 11), (11, 12),   # Orta parmak
    (0, 13), (13, 14), (14, 15), (15, 16), # Yüzük parmağı
    (0, 17), (17, 18), (18, 19), (19, 20), # Serçe parmak
    (5, 9), (9, 13), (13, 17),             # Avuç içi kemikleri
]

# Her parmağın eklem zinciri (kök -> uç)
FINGER_CHAINS = [
    (1, 2, 3, 4),      # Başparmak
    (5, 6, 7, 8),      # İşaret
    (9, 10, 11, 12),   # Orta
    (13, 14, 15, 16),  # Yüzük
    (17, 18, 19, 20),  # Serçe
]


def _cosine_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def _canonicalize_landmarks(landmarks: np.ndarray, aspect_ratio: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    (21, 3) normalize MediaPipe koordinatlarını en-boy oranından arındırır,
    bileğe (0) merkezler, avuç ölçeğine böler ve sağ/sol el için X-simetrisini normalize eder.
    """
    pts = np.asarray(landmarks, dtype=np.float32).copy()
    if pts.shape != (21, 3):
        raise ValueError(f"Beklenen landmark şekli (21, 3), gelen: {pts.shape}")

    # Görüntü en-boy oranını (width / height) düzelt ki geometri izotropik olsun
    pts[:, 0] *= float(aspect_ratio)
    pts[:, 2] *= float(aspect_ratio)

    # Bileği orijin (0, 0, 0) yap
    rel = pts - pts[0]

    # Sabit avuç iskelet ölçeği (MCP eklemlerinin bileğe ortalama uzaklığı)
    mcp_dists = np.linalg.norm(rel[[5, 9, 13, 17], :2], axis=1)
    palm_scale = float(max(np.mean(mcp_dists), 1e-4))
    norm = rel / palm_scale

    # Sağ/sol el simetri standardizasyonu: İşaret MCP (5) her zaman +X tarafında olsun
    if (norm[5, 0] - norm[17, 0]) < -0.05:
        norm[:, 0] = -norm[:, 0]

    return norm, palm_scale


def extract_hand_features(landmarks: np.ndarray, aspect_ratio: float = 1.0) -> np.ndarray:
    """
    (21, 3) el eklemlerinden 88 boyutlu ölçek/konum/el-tarafı değişmez öznitelik vektörü üretir:
      - 63: Normalize ve simetrikleştirilmiş 3D eklem koordinatları
      -  5: Parmak ucu/bilek vs MCP/bilek uzama oranları
      -  5: Parmak uçlarının avuç merkezine (9) veya serçe köküne (17) uzaklıkları
      - 10: Proksimal ve distal eklem bükülme kosinüsleri (5 parmak x 2)
      -  4: Başparmak ucu (4) ile diğer 4 parmak ucu arası çimdik mesafeleri
      -  1: Başparmak dikey yönelimi (Başparmak yukarı/aşağı tespiti)
    """
    norm, _ = _canonicalize_landmarks(landmarks, aspect_ratio=aspect_ratio)
    xy = norm[:, :2]

    feats = [norm.reshape(-1)]  # 63

    # 1. Uzama oranları: d(tip, 0) / d(mcp, 0) -> 5 değer
    tip_wrist_ratios = []
    for _, mcp, _, tip in FINGER_CHAINS:
        d_tip = float(np.linalg.norm(xy[tip]))
        d_mcp = float(max(np.linalg.norm(xy[mcp]), 1e-4))
        tip_wrist_ratios.append(d_tip / d_mcp)
    feats.append(np.array(tip_wrist_ratios, dtype=np.float32))

    # 2. Avuç içine kapanma mesafeleri -> 5 değer
    # Başparmak için serçe kökü (17), diğer 4 parmak için bilek/orta-MCP ortası referans alınır
    palm_center = 0.5 * (xy[0] + xy[9])
    palm_dists = [float(np.linalg.norm(xy[4] - xy[17]))]
    for tip in (8, 12, 16, 20):
        palm_dists.append(float(np.linalg.norm(xy[tip] - palm_center)))
    feats.append(np.array(palm_dists, dtype=np.float32))

    # 3. Eklem bükülme açıları (kosinüs) -> 10 değer
    joint_cosines = []
    for j0, j1, j2, j3 in FINGER_CHAINS:
        c_prox = _cosine_angle(xy[j1] - xy[j0], xy[j2] - xy[j1])
        c_dist = _cosine_angle(xy[j2] - xy[j1], xy[j3] - xy[j2])
        joint_cosines.extend([c_prox, c_dist])
    feats.append(np.array(joint_cosines, dtype=np.float32))

    # 4. Başparmak ucu - diğer parmak uçları mesafesi -> 4 değer
    pinch_dists = [float(np.linalg.norm(xy[4] - xy[t])) for t in (8, 12, 16, 20)]
    feats.append(np.array(pinch_dists, dtype=np.float32))

    # 5. Başparmak dikey yönelimi (negatif = görüntüde yukarı) -> 1 değer
    thumb_vec = xy[4] - xy[2]
    thumb_len = float(max(np.linalg.norm(thumb_vec), 1e-4))
    thumb_up_dir = float(thumb_vec[1] / thumb_len)
    feats.append(np.array([thumb_up_dir], dtype=np.float32))

    vec = np.concatenate(feats).astype(np.float32)
    return vec


def analyze_finger_states(landmarks: np.ndarray, aspect_ratio: float = 1.0) -> Dict[str, Any]:
    """
    21 el ekleminden parmakların açık/kapalı durumunu, çimdik mesafesini ve
    biyomekanik doğrulama sinyallerini çıkarır.
    """
    norm, _ = _canonicalize_landmarks(landmarks, aspect_ratio=aspect_ratio)
    xy = norm[:, :2]

    # Başparmak açıklığı: uç (4) hem serçe kökünden (17) hem işaret MCP'den (5) uzaksa açık
    d_thumb_pinky = float(np.linalg.norm(xy[4] - xy[17]))
    d_thumb_index_mcp = float(np.linalg.norm(xy[4] - xy[5]))
    thumb_straight = _cosine_angle(xy[2] - xy[1], xy[4] - xy[2])
    thumb_open = (d_thumb_pinky > 0.92 and d_thumb_index_mcp > 0.42) or (
        d_thumb_index_mcp > 0.55 and thumb_straight > 0.45
    )

    # 4 ana parmak: tip->wrist uzaklığı PIP->wrist uzaklığından belirgin büyükse açık
    names = ["index", "middle", "ring", "pinky"]
    chains_4 = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    extended: Dict[str, bool] = {"thumb": bool(thumb_open)}
    for name, (mcp, pip, tip) in zip(names, chains_4):
        d_tip = float(np.linalg.norm(xy[tip]))
        d_pip = float(np.linalg.norm(xy[pip]))
        d_mcp = float(max(np.linalg.norm(xy[mcp]), 1e-4))
        straight = _cosine_angle(xy[pip] - xy[mcp], xy[tip] - xy[pip])
        is_ext = (d_tip > d_pip * 1.08) and (d_tip / d_mcp > 1.32) and (straight > 0.15)
        extended[name] = bool(is_ext)

    pinch_ratio = float(np.linalg.norm(xy[4] - xy[8]))
    thumb_vec = xy[4] - xy[2]
    thumb_up_score = float(-thumb_vec[1] / max(np.linalg.norm(thumb_vec), 1e-4))

    four_count = sum(1 for n in names if extended[n])

    return {
        "extended": extended,
        "four_count": four_count,
        "total_extended": four_count + (1 if thumb_open else 0),
        "pinch_ratio": pinch_ratio,
        "thumb_up_score": thumb_up_score,
    }


def ensure_hand_landmarker_task(task_path: Path = None) -> Optional[Path]:
    """hand_landmarker.task model dosyasının varlığını doğrular; yoksa indirir."""
    target = Path(task_path or HAND_LANDMARKER_TASK_PATH)
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".download")
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, str(tmp))
        tmp.replace(target)
        return target
    except Exception:
        return None


class HandTracker:
    """
    MediaPipe HandLandmarker tabanlı yüksek kararlılıklı el tespit ve takip motoru.
    Yumuşatılmış BBox (EMA), 21 eklem koordinatı ve 88 boyutlu öznitelik çıkarır.
    """

    def __init__(
        self,
        running_mode: str = "IMAGE",
        task_path: Path = None,
        min_detection_confidence: float = 0.55,
        min_presence_confidence: float = 0.55,
        min_tracking_confidence: float = 0.50,
        bbox_ema_alpha: float = 0.45,
        auto_download: bool = True,
    ):
        self.running_mode = running_mode.upper()
        self.bbox_ema_alpha = bbox_ema_alpha
        self._smooth_box: Optional[np.ndarray] = None
        self._last_ts_ms = 0
        self._landmarker = None
        self._mp = None
        self._vision = None

        model_file = Path(task_path or HAND_LANDMARKER_TASK_PATH)
        if not model_file.exists() and auto_download:
            model_file = ensure_hand_landmarker_task(model_file) or model_file

        if model_file.exists():
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                self._mp = mp
                self._vision = vision
                mode_enum = (
                    vision.RunningMode.VIDEO
                    if self.running_mode == "VIDEO"
                    else vision.RunningMode.IMAGE
                )
                model_bytes = model_file.read_bytes()
                options = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
                    running_mode=mode_enum,
                    num_hands=1,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_presence_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
            except Exception as ex:
                print(f"[HandTracker Uyarı] MediaPipe başlatılamadı, yedek moda geçiliyor: {ex}")
                self._landmarker = None

    @property
    def is_available(self) -> bool:
        return self._landmarker is not None

    def reset_smoothing(self):
        self._smooth_box = None

    def detect(self, rgb_image: np.ndarray) -> Dict[str, Any]:
        """
        RGB uint8 görüntüde eli tespit eder.
        Döndürür: detected, landmarks (21,3), pixel_landmarks (21,2), bbox, handedness,
                  score, features (88,), finger_states
        """
        h, w = rgb_image.shape[:2]
        empty = {
            "detected": False,
            "landmarks": None,
            "pixel_landmarks": None,
            "bbox": None,
            "handedness": None,
            "score": 0.0,
            "features": None,
            "finger_states": None,
        }
        if self._landmarker is None or rgb_image is None or rgb_image.size == 0:
            self._smooth_box = None
            return empty

        mp_img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_image),
        )

        if self.running_mode == "VIDEO":
            ts_ms = int(time.monotonic() * 1000)
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms
            result = self._landmarker.detect_for_video(mp_img, ts_ms)
        else:
            result = self._landmarker.detect(mp_img)

        if not result.hand_landmarks:
            self._smooth_box = None
            return empty

        lm_list = result.hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)

        handedness = "Right"
        score = 0.9
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name or "Right"
            score = float(result.handedness[0][0].score)

        px = np.stack([landmarks[:, 0] * w, landmarks[:, 1] * h], axis=1)

        # Kare ve cömert dolgulu el kutusu hesapla
        min_x, min_y = float(np.min(px[:, 0])), float(np.min(px[:, 1]))
        max_x, max_y = float(np.max(px[:, 0])), float(np.max(px[:, 1]))
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        side = max(max_x - min_x, max_y - min_y) * 1.42
        side = max(120.0, min(float(min(w, h) - 16), side))

        raw_box = np.array([cx, cy, side], dtype=np.float32)
        if self._smooth_box is None:
            self._smooth_box = raw_box
        else:
            a = self.bbox_ema_alpha
            self._smooth_box = (1.0 - a) * self._smooth_box + a * raw_box

        scx, scy, sside = [float(v) for v in self._smooth_box]
        half = sside / 2.0
        x1 = int(max(0, min(w - sside, scx - half)))
        y1 = int(max(0, min(h - sside, scy - half)))
        x2 = int(min(w, x1 + sside))
        y2 = int(min(h, y1 + sside))

        aspect = float(w) / float(max(h, 1))
        features = extract_hand_features(landmarks, aspect_ratio=aspect)
        finger_states = analyze_finger_states(landmarks, aspect_ratio=aspect)

        return {
            "detected": True,
            "landmarks": landmarks,
            "pixel_landmarks": px,
            "bbox": (x1, y1, x2, y2),
            "handedness": handedness,
            "score": score,
            "features": features,
            "finger_states": finger_states,
        }

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def __del__(self):
        self.close()

