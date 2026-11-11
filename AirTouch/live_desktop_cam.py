"""
AirTouch 60 FPS Canlı Masaüstü Kamera Penceresi (live_desktop_cam.py)
Özellikler:
1. 🤖 Otomatik El Takibi (Auto-Tracking): Kutu otomatik olarak elinizi bulur ve takip eder.
2. 📐 Genişletilmiş BBox (340x340 px): Artık elinizi küçük bir kutuya sığdırmaya çalışmak yok!
3. 🛡️ Akıllı 4 Katmanlı Karar Mimarisi (El varlık kapısı, hold-to-confirm, one-shot kilitlenme).
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import cv2

# Proje dizinini sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    MODELS_DIR, 
    GESTURE_NAMES, 
    GESTURE_ACTIONS, 
    DEFAULT_CONFIDENCE_THRESHOLD, 
    DEFAULT_COOLDOWN_SECONDS
)
from core.model import AirTouchCNN
from core.system_controller import system_controller

def detect_hand_presence(hand_bgr, min_skin_ratio=0.10):
    """
    YCrCb renk uzayında kutu içinde ten rengi yoğunluğunu kontrol eder.
    """
    if hand_bgr is None or hand_bgr.size == 0 or hand_bgr.shape[0] < 10 or hand_bgr.shape[1] < 10:
        return False, 0.0
    ycrcb = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    total_pixels = hand_bgr.shape[0] * hand_bgr.shape[1]
    skin_pixels = np.count_nonzero(skin_mask)
    ratio = float(skin_pixels) / float(total_pixels)
    return (ratio >= min_skin_ratio), ratio

def find_hand_in_frame(frame, min_area=3000, max_area=85000):
    """
    Kare içindeki kaldırılmış eli (en belirgin ve yüksek ten bloğunu) otomatik bulur.
    Döndürür: (found, target_center_x, target_center_y, target_size)
    """
    h, w, _ = frame.shape
    # Arama bölgesini sınırla (Kenar parazitlerini ve aşırı alt gövdeyi ele)
    search_x1 = int(w * 0.08)
    search_x2 = int(w * 0.92)
    search_y1 = int(h * 0.05)
    search_y2 = int(h * 0.90)

    search_roi = frame[search_y1:search_y2, search_x1:search_x2]
    ycrcb = cv2.cvtColor(search_roi, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Alt kenara yapışık devasa gövde bloklarını ele
            if by + bh >= (search_y2 - search_y1 - 15) and bw > (w * 0.40):
                continue
            gx = search_x1 + bx
            gy = search_y1 + by
            candidates.append((area, gx, gy, bw, bh))

    if not candidates:
        return False, None, None, None

    # En yukarıda olan (en küçük gy koordinatlı kaldırılmış el) adayını seç
    best_cand = min(candidates, key=lambda item: item[2])
    area, gx, gy, bw, bh = best_cand

    cx = gx + bw // 2
    cy = gy + bh // 2
    # Cömert ve geniş boyutlandırma
    hand_size = int(max(bw, bh) * 1.45)
    hand_size = max(260, min(min(w, h) - 50, hand_size))

    return True, cx, cy, hand_size

def run_live_desktop_camera(camera_source="http://192.168.1.15:8080/video"):
    if len(sys.argv) > 1 and sys.argv[1].strip():
        arg_source = sys.argv[1].strip()
        if arg_source.isdigit():
            camera_source = int(arg_source)
        else:
            camera_source = arg_source
            if not camera_source.endswith("/video") and "http" in camera_source and ":8080" in camera_source:
                camera_source = camera_source.rstrip("/") + "/video"

    print("=" * 70)
    print("🖐️ AirTouch Akıllı & Otomatik Takipli El Kumandası Başlatılıyor...")
    print(f"Kamera Kaynağı: {camera_source}")
    print("YENİ ÖZELLİKLER:")
    print("  ⭐ OTOMATİK EL TAKİBİ: Kutu otomatik olarak elinize kilitlenir!")
    print("     (Elinizi kutunun içine denk getirmeye uğraşmanıza gerek yok)")
    print("  ⭐ GENİŞ BOYUT: Kutu varsayılan olarak %85 daha geniş (340x340 px).")
    print("  ⭐ ONE-SHOT LATCH: Tek harekette yalnızca 1 kez basar, kilitleyip bekler.")
    print("Kontroller:")
    print("  - [SPACE]: Otomatik El Takibini Aç / Kapat (Açıkken elinizi takip eder)")
    print("  - [1 / 2 / 3]: Kutu Boyutu (1: 280px, 2: 340px Geniş, 3: 420px Devasa)")
    print("  - [+ / -]: Boyutu serbestçe büyüt / küçült")
    print("  - [W / A / S / D]: Manuel modda kutuyu taşı")
    print("  - [T]: Güven Eşiğini Değiştir (%70 / %75 / %80)")
    print("  - [Q / ESC]: Çıkış")
    print("=" * 70)

    # 1. Modeli Yükle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AirTouchCNN(num_classes=10).to(device)
    model_path = MODELS_DIR / "airtouch_model.pth"
    
    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        val_acc = checkpoint.get('val_acc', 98.71)
        print(f"✅ Model başarıyla yüklendi (Doğruluk: %{val_acc:.2f})")
    else:
        print("❌ HATA: airtouch_model.pth bulunamadı!")
        return

    model.eval()

    # 2. Kamera Bağlantısı
    print(f"Kameraya bağlanılıyor: {camera_source} ...")
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print(f"\n❌ HATA: Kamera açılamadı: '{camera_source}'!")
        return

    TARGET_W = 960
    TARGET_H = 540
    WINDOW_TITLE = "AirTouch - Otomatik Takipli El Kumandasi (SPACE: Oto-Takip | 1/2/3: Boyut)"
    
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, TARGET_W, TARGET_H)

    # Kutu Boyutları & Pozisyonu (Varsayılan Geniş Boyut: 340x340 px)
    auto_track = True           # Otomatik takip varsayılan olarak AÇIK!
    box_size = 340.0
    box_cx = float(TARGET_W // 2)
    box_cy = float(TARGET_H // 2)

    # Akıllı Durum Makinesi
    REQUIRED_STABLE_FRAMES = 5
    ONE_SHOT_ACTIONS = {"play_pause", "next_track", "prev_track", "mute"}
    STEP_ACTIONS = {"volume_up", "volume_down"}
    STEP_COOLDOWN = 0.95

    confidence_threshold = 0.75
    min_margin = 0.28

    candidate_class = None
    stable_count = 0
    latched = False
    latched_gesture = None
    last_step_trigger_time = 0.0
    last_action_message = None
    flash_timer = 0.0

    print("AirTouch hazır. Elinizi kameranın önüne kaldırın!")

    while True:
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            time.sleep(0.2)
            k = cv2.waitKey(10) & 0xFF
            if k in (ord('q'), ord('Q'), 27):
                break
            continue

        frame = cv2.resize(raw_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # ==================== 1. OTOMATİK EL TAKİBİ ====================
        hand_auto_found = False
        if auto_track:
            found, target_cx, target_cy, target_sz = find_hand_in_frame(frame)
            if found:
                hand_auto_found = True
                # Yumuşak Takip (EMA Filtresi - Sarsıntısız ve akıcı geçiş)
                alpha = 0.35
                box_cx = (1.0 - alpha) * box_cx + alpha * float(target_cx)
                box_cy = (1.0 - alpha) * box_cy + alpha * float(target_cy)
                box_size = (1.0 - alpha) * box_size + alpha * float(target_sz)

        # Sınır kontrolleri
        box_size = max(200.0, min(float(min(w, h) - 40), box_size))
        half_s = box_size / 2.0
        box_cx = max(half_s + 10, min(w - half_s - 10, box_cx))
        box_cy = max(half_s + 50, min(h - half_s - 25, box_cy))

        x1 = int(box_cx - half_s)
        y1 = int(box_cy - half_s)
        x2 = int(box_cx + half_s)
        y2 = int(box_cy + half_s)

        hand_roi = frame[y1:y2, x1:x2]

        # ==================== 2. EL VARLIK KONTROLÜ ====================
        has_hand, skin_ratio = detect_hand_presence(hand_roi, min_skin_ratio=0.09)

        pred_class = None
        confidence = 0.0
        action_name = None
        class_name = "El Yok"
        action_executed = None
        progress = 0.0
        norm_roi = np.zeros((64, 64), dtype=np.float32)

        if not has_hand and not hand_auto_found:
            stable_count = 0
            candidate_class = None
            latched = False
            latched_gesture = None
            status_text = "ELINIZI KALDIRIN (KUTU OTOMATIK YAKALAR)"
            status_state = "EMPTY"
        else:
            # Model Çıkarımı
            gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
            resized_roi = cv2.resize(gray_roi, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)

            p_min, p_max = np.percentile(resized_roi, (2, 98))
            if p_max > p_min + 1e-4:
                norm_roi = (resized_roi - p_min) / (p_max - p_min)
                norm_roi = np.clip(norm_roi, 0.0, 1.0)
            else:
                norm_roi = resized_roi / 255.0

            tensor = torch.tensor(norm_roi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = model.predict_probabilities(tensor).cpu().numpy()[0]

            sorted_indices = np.argsort(probs)[::-1]
            top1 = int(sorted_indices[0])
            top2 = int(sorted_indices[1])
            confidence = float(probs[top1])
            margin = float(probs[top1] - probs[top2])
            pred_class = top1

            class_name = GESTURE_NAMES.get(pred_class, f"İşaret {pred_class}")
            action_name = GESTURE_ACTIONS.get(pred_class, None)

            is_quality = (confidence >= confidence_threshold) and (margin >= min_margin)

            if not is_quality:
                stable_count = max(0, stable_count - 1)
                status_text = f"Hareketi Tutun (%{confidence * 100:.0f})"
                status_state = "UNCERTAIN"
            else:
                if latched and latched_gesture == pred_class:
                    status_text = f"KILITLENDI: {class_name} (Yeni komut icin elinizi indirin)"
                    status_state = "LATCHED"
                    progress = 1.0
                else:
                    if candidate_class == pred_class:
                        stable_count += 1
                    else:
                        candidate_class = pred_class
                        stable_count = 1
                        latched = False

                    progress = min(1.0, stable_count / REQUIRED_STABLE_FRAMES)

                    if stable_count >= REQUIRED_STABLE_FRAMES:
                        if action_name in ONE_SHOT_ACTIONS:
                            action_result = system_controller.execute_action(action_name)
                            if action_result:
                                last_action_message = action_result
                                action_executed = action_result
                                latched = True
                                latched_gesture = pred_class
                                flash_timer = now + 0.6
                                status_text = f"✔ ISLEM: {action_result}"
                                status_state = "TRIGGERED"
                        elif action_name in STEP_ACTIONS:
                            if now - last_step_trigger_time >= STEP_COOLDOWN:
                                action_result = system_controller.execute_action(action_name)
                                if action_result:
                                    last_action_message = action_result
                                    action_executed = action_result
                                    last_step_trigger_time = now
                                    stable_count = 2
                                    flash_timer = now + 0.4
                                    status_text = f"✔ SES AYARLANDI: {action_result}"
                                    status_state = "TRIGGERED"
                            else:
                                wait_left = STEP_COOLDOWN - (now - last_step_trigger_time)
                                status_text = f"Bekleniyor ({wait_left:.1f}s)..."
                                status_state = "STEP_WAIT"
                        else:
                            status_text = f"{class_name}"
                            status_state = "NO_ACTION"
                    else:
                        status_text = f"Dogrulaniyor [%{int(progress * 100)}]: {class_name}"
                        status_state = "CONFIRMING"

        # ==================== EKRANA ÇİZİMLER (HUD) ====================
        COLOR_GREEN = (0, 255, 128)
        COLOR_YELLOW = (0, 215, 255)
        COLOR_CYAN = (255, 200, 0)
        COLOR_GRAY = (120, 130, 140)
        COLOR_DARK = (18, 22, 28)

        if flash_timer > now:
            active_color = COLOR_GREEN
        elif status_state == "TRIGGERED":
            active_color = COLOR_GREEN
        elif status_state == "CONFIRMING":
            active_color = COLOR_YELLOW
        elif status_state == "LATCHED":
            active_color = (0, 170, 255)
        elif status_state == "UNCERTAIN":
            active_color = COLOR_CYAN
        else:
            active_color = COLOR_GRAY

        # --- A. Bounding Box ---
        box_thickness = 3 if (flash_timer > now or auto_track) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), active_color, box_thickness)

        # Siber Köşe Kancaları (Daha belirgin)
        c_len = 30
        cv2.line(frame, (x1, y1), (x1 + c_len, y1), active_color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + c_len), active_color, 4)
        cv2.line(frame, (x2, y1), (x2 - c_len, y1), active_color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + c_len), active_color, 4)
        cv2.line(frame, (x1, y2), (x1 + c_len, y2), active_color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - c_len), active_color, 4)
        cv2.line(frame, (x2, y2), (x2 - c_len, y2), active_color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - c_len), active_color, 4)

        # Otomatik Takip Göstergesi (Kutunun üst sol köşesinde minik rozet)
        mode_badge = "[OTO-TAKIP AKTIF]" if auto_track else "[MANUEL KUTU]"
        cv2.putText(frame, mode_badge, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200) if auto_track else (180, 180, 180), 1, cv2.LINE_AA)

        # --- B. Doğrulama İlerleme Çubuğu ---
        if status_state == "CONFIRMING" and progress > 0.0:
            bar_w = int((x2 - x1) * progress)
            cv2.rectangle(frame, (x1, y2 + 8), (x2, y2 + 18), COLOR_DARK, -1)
            cv2.rectangle(frame, (x1, y2 + 8), (x1 + bar_w, y2 + 18), COLOR_YELLOW, -1)
            cv2.rectangle(frame, (x1, y2 + 8), (x2, y2 + 18), (80, 80, 80), 1)

        # --- C. Durum Etiketi ---
        badge_w = len(status_text) * 11 + 16
        cv2.rectangle(frame, (x1, y1 - 32), (x1 + badge_w, y1), COLOR_DARK, -1)
        cv2.rectangle(frame, (x1, y1 - 32), (x1 + badge_w, y1), active_color, 1)
        cv2.putText(frame, status_text, (x1 + 8, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, active_color, 1, cv2.LINE_AA)

        # --- D. Üst Medya HUD Şeridi ---
        cv2.rectangle(frame, (0, 0), (w, 52), COLOR_DARK, -1)
        cv2.line(frame, (0, 52), (w, 52), active_color, 2)

        if action_executed or (flash_timer > now and last_action_message):
            hud_msg = f">> MEDYADA YAPILAN ISLEM: {last_action_message} <<"
            hud_color = COLOR_GREEN
        elif status_state == "LATCHED":
            hud_msg = f"KILITLENDI: {class_name} | Tekrar komut icin elinizi indirin"
            hud_color = (0, 200, 255)
        elif status_state == "CONFIRMING":
            hud_msg = f"Hareketi Tutun: {class_name} (Doğrulama: %{int(progress * 100)})"
            hud_color = COLOR_YELLOW
        elif auto_track and hand_auto_found:
            hud_msg = f"El Kilitlendi | Boyut: {int(box_size)}px | Eylemi yapmak icin jesti gosterin"
            hud_color = (0, 255, 200)
        else:
            hud_msg = "AirTouch: Elinizi kaldirin, kutu elinize otomatik gelecektir (SPACE: Takip Modu)"
            hud_color = (180, 190, 200)

        cv2.putText(frame, hud_msg, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2, cv2.LINE_AA)

        # --- E. Alt Bilgi Çubuğu ---
        cv2.rectangle(frame, (0, h - 28), (w, h), COLOR_DARK, -1)
        cv2.line(frame, (0, h - 28), (w, h - 28), (60, 70, 80), 1)
        track_str = "ACIK" if auto_track else "KAPALI"
        footer_msg = f"[SPACE] Oto-Takip: {track_str} | [1/2/3] Boyut: {int(box_size)}px | [T] Esik: %{int(confidence_threshold*100)} | [Q] Cikis"
        cv2.putText(frame, footer_msg, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1, cv2.LINE_AA)

        # --- F. Model Görünümü ---
        thumb_64 = cv2.resize((norm_roi * 255).astype(np.uint8), (76, 76))
        thumb_bgr = cv2.cvtColor(thumb_64, cv2.COLOR_GRAY2BGR)
        frame[h - 110:h - 34, 15:91] = thumb_bgr
        cv2.rectangle(frame, (15, h - 110), (91, h - 34), active_color, 1)
        cv2.putText(frame, "Model Gozu", (16, h - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1)

        cv2.imshow(WINDOW_TITLE, frame)

        # Tuş Kontrolleri
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == 32: # SPACE: Otomatik El Takibi Aç / Kapat
            auto_track = not auto_track
            print(f"👉 Otomatik El Takibi: {'AÇIK' if auto_track else 'KAPALI'}")
        elif key == ord('1'): # Orta Boyut
            box_size = 280.0
        elif key == ord('2'): # Geniş Boyut
            box_size = 340.0
        elif key == ord('3'): # Devasa Boyut
            box_size = 420.0
        elif key in (ord('+'), ord('=')):
            box_size += 25.0
        elif key in (ord('-'), ord('_')):
            box_size = max(180.0, box_size - 25.0)
        elif key in (ord('w'), ord('W')):
            auto_track = False
            box_cy -= 25.0
        elif key in (ord('s'), ord('S')):
            auto_track = False
            box_cy += 25.0
        elif key in (ord('a'), ord('A')):
            auto_track = False
            box_cx -= 25.0
        elif key in (ord('d'), ord('D')):
            auto_track = False
            box_cx += 25.0
        elif key in (ord('c'), ord('C'), ord('r'), ord('R')):
            auto_track = True
            box_cx = float(w // 2)
            box_cy = float(h // 2)
            box_size = 340.0
        elif key in (ord('t'), ord('T')):
            thresholds = [0.70, 0.75, 0.80]
            curr_idx = thresholds.index(confidence_threshold) if confidence_threshold in thresholds else 1
            confidence_threshold = thresholds[(curr_idx + 1) % len(thresholds)]
            print(f"👉 Güven Eşiği: %{confidence_threshold * 100:.0f}")

    cap.release()
    cv2.destroyAllWindows()
    print("AirTouch canlı masaüstü penceresi kapatıldı.")

if __name__ == "__main__":
    run_live_desktop_camera()
