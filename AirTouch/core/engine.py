"""
AirTouch Birleşik Akıllı Çıkarım ve Tetikleme Motoru (engine.py)
Hem 21 eklemli İskelet AI (MediaPipe + AirTouchLandmarkNet) hem de
Kurs CNN (AirTouchCNN) modlarını tek kararlı durum makinesinde birleştirir.

4 Katmanlı Anti-Spam Mimarisi:
  1. El Varlık Kapısı (MediaPipe HandLandmarker veya Ten Rengi Kapısı)
  2. Güven & Sınıf Ayrım Marjı Kapısı (Confidence + Top1-Top2 Margin + Biyomekanik Doğrulama)
  3. Kararlılık Onayı (Hold-to-Confirm: N kararlı kare)
  4. Akıllı Tetikleme (Tek-Atış Kilitlenme / One-Shot Latch + Bekleme Süresi / Cooldown)
"""
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ACTION_PROFILES,
    AVAILABLE_ACTIONS,
    CNN_MODEL_PATH,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_ENGINE_MODE,
    DEFAULT_MAX_HAND_MOTION,
    DEFAULT_MIN_MARGIN,
    DEFAULT_PALM_ARM_SECONDS,
    DEFAULT_STABLE_FRAMES,
    DEFAULT_STRICT_NEUTRAL_RESET,
    GESTURE_ACTIONS,
    GESTURE_NAMES,
    LANDMARK_FEATURE_DIM,
    LANDMARK_GESTURE_NAMES,
    LANDMARK_MODEL_PATH,
    LANDMARK_NUM_CLASSES,
    NUM_CLASSES,
)
from core.dataset import preprocess_frame
from core.hud_renderer import render_hud_frame
from core.model import AirTouchCNN, AirTouchLandmarkNet
from core.system_controller import system_controller


def _compute_biomech_weights(finger_states: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    21 eklemden çıkarılan parmak açıklığı / çimdik / başparmak yönelim sinyallerini
    kullanarak 8 sınıf için yumuşak biyomekanik uyumluluk katsayıları üretir.
    """
    weights = np.ones(LANDMARK_NUM_CLASSES, dtype=np.float32)
    if not finger_states:
        return weights

    ext = finger_states["extended"]
    four_cnt = finger_states["four_count"]
    pinch = finger_states["pinch_ratio"]
    thumb_up = finger_states["thumb_up_score"]
    thumb_open = ext["thumb"]

    # 0: Açık El -> en az 3-4 parmak açık ve çimdik yok
    if four_cnt < 3 or pinch < 0.28:
        weights[0] = 0.25

    # 1: Yumruk -> 4 parmak kapalı ve belirgin başparmak-yukarı yok
    if four_cnt > 1:
        weights[1] = 0.20
    elif thumb_open and thumb_up > 0.78:
        weights[1] = 0.35

    # 2: 1 Parmak -> işaret açık ve toplam açık parmak <= 2
    if not ext["index"] or four_cnt > 2:
        weights[2] = 0.22

    # 3: 2 Parmak -> işaret & orta açık, yüzük & serçe kapalı
    if not (ext["index"] and ext["middle"]) or four_cnt > 3:
        weights[3] = 0.22

    # 4: 3 Parmak -> 2-4 arası açık parmak
    if four_cnt < 2 or four_cnt > 4:
        weights[4] = 0.28

    # 5: 4 Parmak -> 4 ana parmak açık ve başparmak kapalı
    if four_cnt < 3 or thumb_open:
        weights[5] = 0.30

    # 6: OK / Çimdik -> başparmak & işaret ucu yakın ve en az 2 diğer parmak açık
    if pinch > 0.42 or four_cnt < 2:
        weights[6] = 0.20

    # 7: Başparmak Yukarı -> 4 parmak kapalı ve başparmak yukarı yönlü
    if four_cnt > 1 or thumb_up < 0.35:
        weights[7] = 0.22

    return weights


def _check_skin_ratio(rgb_array: np.ndarray, bbox=None) -> float:
    """Yedek el varlık kapısı (CNN modunda veya tracker yokken kullanılır)."""
    arr = np.asarray(rgb_array)
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        arr = arr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if arr.size == 0 or arr.ndim < 3:
        return 0.0
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(np.mean(mask))


class AirTouchEngine:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        smoothing_window: int = 3,
        device: Optional[str] = None,
        engine_mode: Optional[str] = None,
        required_stable_frames: Optional[int] = None,
        min_margin: float = DEFAULT_MIN_MARGIN,
        running_mode: str = "IMAGE",
        hand_tracker: Optional[Any] = None,
        profile_label: str = "Medya",
        enable_latch: Optional[bool] = None,
        guard_mode: str = "strict_reset",
        strict_neutral_reset: bool = DEFAULT_STRICT_NEUTRAL_RESET,
        max_hand_motion: float = DEFAULT_MAX_HAND_MOTION,
        palm_arm_seconds: float = DEFAULT_PALM_ARM_SECONDS,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = float(confidence_threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.min_margin = float(min_margin)
        self.profile_label = profile_label

        # model_path açıkça verilmiş ve landmark modeli değilse CNN modunu algıla (test uyumluluğu)
        if engine_mode is not None:
            self.engine_mode = engine_mode
        elif model_path is not None and "landmark" not in str(model_path).lower():
            self.engine_mode = "cnn"
        else:
            self.engine_mode = DEFAULT_ENGINE_MODE

        if required_stable_frames is not None:
            self.required_stable_frames = max(1, int(required_stable_frames))
        else:
            self.required_stable_frames = (
                1 if self.engine_mode == "cnn" else DEFAULT_STABLE_FRAMES
            )

        self.enable_latch = (
            bool(enable_latch)
            if enable_latch is not None
            else (self.engine_mode == "landmark")
        )
        self.guard_mode = guard_mode if guard_mode in ("palm_arm", "strict_reset", "voice_only") else "strict_reset"
        self.strict_neutral_reset = bool(strict_neutral_reset)
        self.max_hand_motion = float(max_hand_motion)
        self.palm_arm_seconds = float(palm_arm_seconds)

        self.history = deque(maxlen=max(1, int(smoothing_window)))

        # Durum makinesi & zamanlayıcılar
        self.last_trigger_time = 0.0
        self.last_action_message: Optional[str] = None
        self.total_triggers = 0
        self.paused = False
        self.show_skeleton = True

        self._candidate_class: Optional[int] = None
        self._stable_count = 0
        self._latched = False
        self._latched_gesture: Optional[int] = None
        self._empty_frames = 0
        self._armed_until = 0.0
        self._last_repeat_class: Optional[int] = None
        self._prev_palm_center: Optional[np.ndarray] = None
        self._motion_ema = 0.0

        # Modelleri yükle
        self.cnn_model = self._load_cnn_model(
            model_path if self.engine_mode == "cnn" else None
        )
        self.landmark_model = self._load_landmark_model(
            model_path if self.engine_mode == "landmark" else None
        )

        # Geriye dönük uyumluluk: engine.model aktif moda göre işaret eder
        self.model = (
            self.landmark_model if self.engine_mode == "landmark" else self.cnn_model
        )

        # Sınıf adları ve eylem haritaları
        self._init_mappings()

        # MediaPipe HandTracker (yalnızca gerektiğinde başlatılır)
        self._running_mode = running_mode
        self._hand_tracker = hand_tracker

    def _init_mappings(self):
        if self.engine_mode == "landmark":
            self.gesture_names = dict(LANDMARK_GESTURE_NAMES)
            self.gesture_actions = dict(ACTION_PROFILES["media"]["actions"])
        else:
            self.gesture_names = dict(GESTURE_NAMES)
            self.gesture_actions = dict(GESTURE_ACTIONS)

    def set_mode(self, engine_mode: str):
        """Motor modunu ('landmark' veya 'cnn') güvenli şekilde değiştirir."""
        if engine_mode not in ("landmark", "cnn") or engine_mode == self.engine_mode:
            return
        self.engine_mode = engine_mode
        self.model = self.landmark_model if engine_mode == "landmark" else self.cnn_model
        self.reset_state()
        self._init_mappings()

    def reset_state(self):
        self.history.clear()
        self._candidate_class = None
        self._stable_count = 0
        self._latched = False
        self._latched_gesture = None
        self._empty_frames = 0
        self._armed_until = 0.0
        self._last_repeat_class = None
        self._prev_palm_center = None
        self._motion_ema = 0.0
        if self._hand_tracker is not None:
            self._hand_tracker.reset_smoothing()

    def reload_landmark_model(self):
        """Özel jest eğitimi sonrası güncellenen LandmarkNet ağırlıklarını belleğe alır."""
        self.landmark_model = self._load_landmark_model(LANDMARK_MODEL_PATH)
        if self.engine_mode == "landmark":
            self.model = self.landmark_model

    def _get_tracker(self):
        if self._hand_tracker is None:
            from core.hand_tracker import HandTracker

            self._hand_tracker = HandTracker(running_mode=self._running_mode)
        return self._hand_tracker

    def _load_cnn_model(self, model_path: Optional[Path] = None) -> AirTouchCNN:
        target = Path(model_path or CNN_MODEL_PATH)
        model = AirTouchCNN(num_classes=NUM_CLASSES).to(self.device)
        if target.exists():
            checkpoint = torch.load(str(target), map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_landmark_model(self, model_path: Optional[Path] = None) -> AirTouchLandmarkNet:
        target = Path(model_path or LANDMARK_MODEL_PATH)
        model = AirTouchLandmarkNet(
            input_dim=LANDMARK_FEATURE_DIM,
            num_classes=LANDMARK_NUM_CLASSES,
        ).to(self.device)
        if target.exists():
            checkpoint = torch.load(str(target), map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    def _to_rgb_array(self, frame) -> np.ndarray:
        if isinstance(frame, Image.Image):
            return np.array(frame.convert("RGB"))
        arr = np.asarray(frame)
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        if arr.shape[2] == 4:
            return arr[..., :3]
        return arr

    def _update_motion(self, landmarks: Optional[np.ndarray], has_hand: bool) -> float:
        """El merkezinin avuç ölçeğine göre kareler arası hızını hesaplar (el sallamayı süzer)."""
        if not has_hand or landmarks is None or len(landmarks) < 10:
            self._prev_palm_center = None
            self._motion_ema = 0.0
            return 0.0
        center = 0.5 * (landmarks[0, :2] + landmarks[9, :2])
        palm_scale = max(float(np.linalg.norm(landmarks[9, :2] - landmarks[0, :2])), 1e-4)
        if self._prev_palm_center is None:
            inst = 0.0
        else:
            inst = float(np.linalg.norm(center - self._prev_palm_center) / palm_scale)
        self._prev_palm_center = center
        if inst >= self._motion_ema:
            self._motion_ema = max(inst, 0.45 * self._motion_ema + 0.55 * inst)
        elif inst < 0.015:
            # El sabitlendiğinde gecikmesiz serbest bırak (tek karelik sıçrama artığı bırakma)
            self._motion_ema = inst
        else:
            self._motion_ema = 0.25 * self._motion_ema + 0.75 * inst
        return self._motion_ema

    def _run_decision_state_machine(
        self,
        probabilities: np.ndarray,
        has_hand: bool,
        trigger_enabled: bool,
        now: float,
        margin: float,
        motion: float = 0.0,
    ) -> Dict[str, Any]:
        num_cls = len(probabilities)
        pred_class = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class])

        if not has_hand:
            self._empty_frames += 1
            self.history.clear()
            self._candidate_class = None
            self._stable_count = 0
            self._last_repeat_class = None
            if self._empty_frames >= 3:
                self._latched = False
                self._latched_gesture = None
                self._armed_until = 0.0
            time_since_last = now - self.last_trigger_time
            cooldown_remaining = max(0.0, self.cooldown_seconds - time_since_last)
            return {
                "stable_class": 0,
                "raw_class_id": pred_class,
                "confidence": 0.0,
                "is_confident": False,
                "action_name": None,
                "action_executed": None,
                "is_cooldown_active": cooldown_remaining > 0.0,
                "cooldown_remaining": round(cooldown_remaining, 2),
                "status_state": "PAUSED" if self.paused else "EMPTY",
                "confirm_progress": 0.0,
                "arm_remaining": 0.0,
                "arm_just_activated": False,
                "guard_mode": self.guard_mode,
            }

        self._empty_frames = 0

        # Yumuşatma (Smoothing / Majority Voting)
        self.history.append((pred_class, confidence))
        classes_in_window = [
            c for c, conf in self.history if conf >= self.confidence_threshold
        ]
        if classes_in_window:
            stable_class = max(set(classes_in_window), key=classes_in_window.count)
        else:
            stable_class = pred_class

        if stable_class >= num_cls:
            stable_class = pred_class

        is_confident = confidence >= self.confidence_threshold
        is_quality = is_confident and (margin >= self.min_margin)
        is_moving = (self.engine_mode == "landmark") and (motion > self.max_hand_motion)

        action_name = self.gesture_actions.get(stable_class)
        if action_name == "none":
            action_name = None

        action_meta = AVAILABLE_ACTIONS.get(action_name or "none", {"mode": "idle"})
        action_mode = action_meta.get("mode", "one_shot")

        time_since_last = now - self.last_trigger_time
        cooldown_remaining = max(0.0, self.cooldown_seconds - time_since_last)
        is_cooldown_active = cooldown_remaining > 0.0
        action_executed = None
        confirm_progress = 0.0
        arm_just_activated = False

        if self.paused:
            self._stable_count = 0
            status_state = "PAUSED"
        elif self.guard_mode == "voice_only":
            self._stable_count = 0
            status_state = "VOICE_ONLY"
        elif is_moving:
            self._stable_count = max(0, self._stable_count - 1)
            status_state = "MOVING"
        elif not is_quality:
            self._stable_count = max(0, self._stable_count - 1)
            status_state = "UNCERTAIN"
        elif action_name is None:
            # Nötr veya pasif jest (ör. Açık El): kilidi temizle ve Güvenli Kapı'da eli hazırla!
            self._last_repeat_class = None
            if self._candidate_class == stable_class:
                self._stable_count += 1
            else:
                self._candidate_class = stable_class
                self._stable_count = 1
            if self._stable_count >= 2:
                self._latched = False
                self._latched_gesture = None
                if self.guard_mode == "palm_arm" and stable_class == 0:
                    was_armed = now <= self._armed_until
                    self._armed_until = now + self.palm_arm_seconds
                    if not was_armed:
                        arm_just_activated = True
                    status_state = "ARMED"
                else:
                    status_state = "READY"
            else:
                status_state = "READY"
            confirm_progress = 1.0
        else:
            # Tekrarlı bir eylemden (ses/scroll) başka bir jeste geçilirse kazara zincir tetiklemeyi kes
            if self._last_repeat_class is not None and stable_class != self._last_repeat_class:
                if self.guard_mode == "palm_arm":
                    self._armed_until = 0.0
                if self.strict_neutral_reset and self.enable_latch:
                    self._latched = True
                    self._latched_gesture = self._last_repeat_class
                self._last_repeat_class = None

            is_one_shot = self.enable_latch and (action_mode == "one_shot")
            armed_ok = (self.guard_mode != "palm_arm") or (now <= self._armed_until)

            if self._latched and self.enable_latch and (self.strict_neutral_reset or self._latched_gesture == stable_class):
                status_state = "LATCHED"
                confirm_progress = 1.0
            elif not armed_ok:
                self._candidate_class = stable_class
                self._stable_count = 0
                status_state = "LOCKED_WAIT_PALM"
            else:
                if self._candidate_class == stable_class:
                    self._stable_count += 1
                else:
                    self._candidate_class = stable_class
                    self._stable_count = 1
                    if not self.strict_neutral_reset and is_one_shot:
                        self._latched = False

                confirm_progress = min(
                    1.0, float(self._stable_count) / float(self.required_stable_frames)
                )

                if self._stable_count < self.required_stable_frames:
                    status_state = "CONFIRMING"
                elif is_cooldown_active:
                    status_state = "COOLDOWN"
                else:
                    status_state = "READY"
                    if trigger_enabled:
                        action_result = system_controller.execute_action(action_name)
                        if action_result:
                            self.last_trigger_time = now
                            self.last_action_message = action_result
                            action_executed = action_result
                            self.total_triggers += 1
                            cooldown_remaining = self.cooldown_seconds
                            is_cooldown_active = True
                            status_state = "TRIGGERED"
                            if is_one_shot:
                                self._latched = True
                                self._latched_gesture = stable_class
                                self._last_repeat_class = None
                                if self.guard_mode == "palm_arm":
                                    self._armed_until = 0.0
                            else:
                                # Tekrarlı eylem (ses/scroll): aynı jest sabit tutulursa sonraki adıma hazırlan
                                self._last_repeat_class = stable_class
                                self._stable_count = (
                                    max(1, self.required_stable_frames - 1)
                                    if self.engine_mode == "cnn"
                                    else max(1, self.required_stable_frames // 2)
                                )
                                if self.guard_mode == "palm_arm":
                                    self._armed_until = now + max(2.5, self.cooldown_seconds + 1.5)

        arm_remaining = max(0.0, self._armed_until - now) if self.guard_mode == "palm_arm" else 0.0
        return {
            "stable_class": stable_class,
            "raw_class_id": pred_class,
            "confidence": confidence,
            "is_confident": is_confident,
            "action_name": action_name,
            "action_executed": action_executed,
            "is_cooldown_active": is_cooldown_active,
            "cooldown_remaining": round(cooldown_remaining, 2),
            "status_state": status_state,
            "confirm_progress": confirm_progress,
            "arm_remaining": round(arm_remaining, 1),
            "arm_just_activated": arm_just_activated,
            "guard_mode": self.guard_mode,
        }

    def process_frame(
        self,
        frame,
        bbox=None,
        trigger_enabled: bool = True,
        fps: Optional[float] = None,
        footer_hint: Optional[str] = None,
        voice_status: Optional[Dict[str, Any]] = None,
        target_context: Optional[Dict[str, Any]] = None,
        show_guide: bool = False,
        compact_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Tek bir kamera karesini (RGB uint8 array veya PIL Image) analiz eder,
        eylem durum makinesini çalıştırır ve HUD çizilmiş kareyi döndürür.
        """
        now = time.time()
        rgb_arr = self._to_rgb_array(frame)

        pixel_landmarks = None
        landmarks = None
        features = None
        finger_states = None
        active_bbox = bbox
        has_hand = True
        processed_tensor = np.zeros((64, 64), dtype=np.float32)

        if self.engine_mode == "landmark":
            tracker = self._get_tracker()
            track_res = tracker.detect(rgb_arr)
            has_hand = bool(track_res["detected"])
            if has_hand:
                landmarks = track_res["landmarks"]
                pixel_landmarks = track_res["pixel_landmarks"]
                features = track_res["features"]
                finger_states = track_res["finger_states"]
                if active_bbox is None:
                    active_bbox = track_res["bbox"]

                feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_probs = (
                        self.landmark_model.predict_probabilities(feat_tensor)
                        .cpu()
                        .numpy()[0]
                    )
                bio_w = _compute_biomech_weights(finger_states)
                fused = raw_probs * bio_w
                denom = float(np.sum(fused))
                probabilities = (fused / denom) if denom > 1e-6 else raw_probs
            else:
                probabilities = np.zeros(LANDMARK_NUM_CLASSES, dtype=np.float32)
                probabilities[0] = 1.0
        else:
            # Kurs CNN Modu: MediaPipe el kutusu bulabiliyorsa kullan, yoksa verilen bbox / merkez kırpma
            if active_bbox is None and min(rgb_arr.shape[:2]) >= 160:
                tracker = self._get_tracker()
                if tracker.is_available:
                    track_res = tracker.detect(rgb_arr)
                    if track_res["detected"]:
                        active_bbox = track_res["bbox"]
                        pixel_landmarks = track_res["pixel_landmarks"]
                        landmarks = track_res["landmarks"]
                    else:
                        skin_ratio = _check_skin_ratio(rgb_arr, bbox=None)
                        has_hand = True if min(rgb_arr.shape[:2]) <= 128 else (skin_ratio >= 0.08)

            tensor = preprocess_frame(rgb_arr, bbox=active_bbox).to(self.device)
            processed_tensor = tensor.squeeze().cpu().numpy()
            with torch.no_grad():
                probabilities = (
                    self.cnn_model.predict_probabilities(tensor).cpu().numpy()[0]
                )

        motion = self._update_motion(landmarks, has_hand)
        sorted_probs = np.sort(probabilities)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

        decision = self._run_decision_state_machine(
            probabilities=probabilities,
            has_hand=has_hand,
            trigger_enabled=trigger_enabled,
            now=now,
            margin=margin,
            motion=motion,
        )

        stable_class = decision["stable_class"]
        class_name = (
            self.gesture_names.get(stable_class, f"İşaret {stable_class}")
            if has_hand
            else "El Bekleniyor"
        )

        report: Dict[str, Any] = {
            "class_id": stable_class,
            "raw_class_id": decision["raw_class_id"],
            "class_name": class_name,
            "confidence": decision["confidence"],
            "margin": margin,
            "is_confident": decision["is_confident"],
            "has_hand": has_hand,
            "action_name": decision["action_name"],
            "action_executed": decision["action_executed"],
            "last_action_message": self.last_action_message,
            "is_cooldown_active": decision["is_cooldown_active"],
            "cooldown_remaining": decision["cooldown_remaining"],
            "status_state": decision["status_state"],
            "confirm_progress": decision["confirm_progress"],
            "arm_remaining": decision["arm_remaining"],
            "arm_just_activated": decision["arm_just_activated"],
            "guard_mode": decision["guard_mode"],
            "hand_motion": round(motion, 3),
            "all_probabilities": probabilities.tolist(),
            "total_triggers": self.total_triggers,
            "processed_tensor": processed_tensor,
            "bbox": active_bbox,
            "landmarks": landmarks,
            "pixel_landmarks": pixel_landmarks,
            "features": features,
            "finger_states": finger_states,
            "paused": self.paused,
            "engine_mode": self.engine_mode,
        }

        report["annotated_frame"] = render_hud_frame(
            frame_rgb=rgb_arr,
            report=report,
            bbox=active_bbox,
            pixel_landmarks=pixel_landmarks,
            show_skeleton=self.show_skeleton,
            profile_label=self.profile_label,
            fps=fps,
            footer_hint=footer_hint,
            action_map=self.gesture_actions,
            voice_status=voice_status,
            target_context=target_context,
            show_guide=show_guide,
            compact_mode=compact_mode,
        )
        return report

    def close(self):
        if self._hand_tracker is not None:
            self._hand_tracker.close()
            self._hand_tracker = None
