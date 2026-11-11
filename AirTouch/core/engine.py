"""
AirTouch Akıllı Çıkarım ve Tetikleme Motoru (engine.py)
Canlı kamera karelerini analiz eder, güven eşiği (threshold),
bekleme süresi (cooldown) ve hareket kararlılığı (smoothing) uygulayarak
Windows medya eylemlerini güvenli ve kararlı şekilde tetikler.
"""
import sys
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
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
from core.dataset import preprocess_frame
from core.system_controller import system_controller

class AirTouchEngine:
    def __init__(
        self,
        model_path: Path = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        smoothing_window: int = 3,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Titremeyi önleyen hareket geçmişi penceresi (Smoothing)
        self.history = deque(maxlen=smoothing_window)
        
        # Zamanlayıcılar ve durum
        self.last_trigger_time = 0.0
        self.last_action_message = None
        self.total_triggers = 0
        
        # Özel tuş eşleşmeleri (Varsayılan ayarlardan kopyala)
        self.gesture_actions = dict(GESTURE_ACTIONS)
        self.gesture_names = dict(GESTURE_NAMES)
        
        # Modeli yükle
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Path = None) -> AirTouchCNN:
        if model_path is None:
            model_path = MODELS_DIR / "airtouch_model.pth"
            
        model = AirTouchCNN(num_classes=10).to(self.device)
        
        if model_path.exists():
            checkpoint = torch.load(str(model_path), map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                val_acc = checkpoint.get('val_acc', 0.0)
                print(f"AirTouch modeli yüklendi: {model_path} (Doğruluk: %{val_acc:.2f})")
            else:
                model.load_state_dict(checkpoint)
                print(f"AirTouch modeli yüklendi: {model_path}")
        else:
            print(f"Uyarı: Model dosyası bulunamadı ({model_path}), eğitimsiz ağırlıklar devrede.")
            
        model.eval()
        return model

    def process_frame(self, frame, bbox=None, trigger_enabled: bool = True) -> dict:
        """
        Kameradan gelen tek bir kareyi işler.
        frame: PIL Image veya NumPy array
        bbox: (x1, y1, x2, y2) El hedef kutusu
        trigger_enabled: Windows tuşlarına gerçekten basılsın mı?
        """
        now = time.time()
        
        # 1. Ön İşleme (Sadece Bounding Box bölgesini kırp!)
        tensor = preprocess_frame(frame, bbox=bbox).to(self.device)
        
        # 2. Model Tahmini
        with torch.no_grad():
            probabilities = self.model.predict_probabilities(tensor).cpu().numpy()[0]
            
        pred_class = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class])
        
        # 3. Kararlılık Penceresi (Smoothing / Majority Voting)
        self.history.append((pred_class, confidence))
        
        # Son pencerelerdeki en baskın sınıfı bul
        classes_in_window = [c for c, conf in self.history if conf >= self.confidence_threshold]
        if classes_in_window:
            stable_class = max(set(classes_in_window), key=classes_in_window.count)
        else:
            stable_class = pred_class

        # 4. Eylemi Belirle ve Güven Eşiğini Kontrol Et
        is_confident = confidence >= self.confidence_threshold
        action_name = self.gesture_actions.get(stable_class, None)
        action_executed = None
        
        time_since_last = now - self.last_trigger_time
        cooldown_remaining = max(0.0, self.cooldown_seconds - time_since_last)
        is_cooldown_active = cooldown_remaining > 0.0

        # 5. Tetikleme Şartı: Güven yüksek + Bekleme süresi dolmuş + Eylem tanımlı + Tetikleme açık
        if is_confident and not is_cooldown_active and action_name and trigger_enabled:
            # Eylemi çalıştır
            action_result = system_controller.execute_action(action_name)
            if action_result:
                self.last_trigger_time = now
                self.last_action_message = action_result
                action_executed = action_result
                self.total_triggers += 1
                cooldown_remaining = self.cooldown_seconds
                is_cooldown_active = True

        # 6. Rapor Paketi
        report = {
            "class_id": stable_class,
            "raw_class_id": pred_class,
            "class_name": self.gesture_names.get(stable_class, f"İşaret {stable_class}"),
            "confidence": confidence,
            "is_confident": is_confident,
            "action_name": action_name,
            "action_executed": action_executed,
            "last_action_message": self.last_action_message,
            "is_cooldown_active": is_cooldown_active,
            "cooldown_remaining": round(cooldown_remaining, 2),
            "all_probabilities": probabilities.tolist(),
            "total_triggers": self.total_triggers,
            "processed_tensor": tensor.squeeze().cpu().numpy(), # 64x64 görselleştirme için
            "bbox": bbox
        }
        
        # 7. Kamera üzerine Bounding Box ve Medya Eylemi HUD Çizimi
        annotated_image = self.draw_hud_overlay(frame, bbox, report)
        report["annotated_frame"] = annotated_image
        
        return report

    def draw_hud_overlay(self, frame, bbox, report):
        """
        Kamera karesinin üzerine:
        1. El hedef Bounding Box'ını (köşe kancalarıyla)
        2. Algılanan el hareketini (etiket)
        3. Medyada yapılan işlemi (üst HUD şeridi) çizer.
        """
        from PIL import Image, ImageDraw, ImageFont

        # PIL kopyasını al
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame).convert("RGB")
        elif isinstance(frame, Image.Image):
            img = frame.copy().convert("RGB")
        else:
            return frame

        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Renk paleti
        COLOR_CONFIDENT = (0, 255, 128)   # Neon Zümrüt Yeşili (Algılandı)
        COLOR_COOLDOWN = (255, 180, 0)    # Kehribar Sarısı (Beklemede)
        COLOR_NORMAL = (0, 210, 255)      # Elektrik Camgöbeği (Boşta)

        is_confident = report.get("is_confident", False)
        is_cooldown = report.get("is_cooldown_active", False)
        action_executed = report.get("action_executed", None)
        last_action = report.get("last_action_message", None)
        class_name = report.get("class_name", "Bilinmiyor")
        confidence = report.get("confidence", 0.0)

        # Aktif renk
        if action_executed:
            active_color = COLOR_CONFIDENT
        elif is_cooldown:
            active_color = COLOR_COOLDOWN
        elif is_confident:
            active_color = COLOR_CONFIDENT
        else:
            active_color = COLOR_NORMAL

        # --- 1. ÜST HUD ŞERİDİ (MEDYADA YAPILAN İŞLEM) ---
        hud_height = 48
        draw.rectangle([0, 0, width, hud_height], fill=(15, 20, 28))
        
        # Alt çizgi vurgusu
        draw.line([(0, hud_height), (width, hud_height)], fill=active_color, width=2)

        if action_executed:
            hud_text = f"⚡ MEDYADA YAPILAN İŞLEM: {action_executed}"
            text_color = COLOR_CONFIDENT
        elif is_cooldown and last_action:
            hud_text = f"⏳ BEKLEME MODU ({report.get('cooldown_remaining')} sn) | Son İşlem: {last_action}"
            text_color = COLOR_COOLDOWN
        elif is_confident:
            hud_text = f"🎯 HAREKET ALGILANDI: {class_name} (Güven: %{confidence * 100:.1f})"
            text_color = COLOR_CONFIDENT
        else:
            hud_text = "🖐️ AirTouch: Elinizi kutunun içine gösterin (Yumruk: Oynat/Durdur | 1 Parmak: Ses Aç)"
            text_color = (200, 215, 230)

        draw.text((15, 14), hud_text, fill=text_color)

        # --- 2. BOUNDING BOX (HEDEF EL KUTUSU) ---
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Ana kutu çerçevesi
            box_width = 3 if is_confident else 2
            draw.rectangle([x1, y1, x2, y2], outline=active_color, width=box_width)

            # Köşe Kancaları (Futuristic HUD Corner Brackets)
            corner_len = min(25, (x2 - x1) // 4)
            corner_w = 4
            # Sol Üst
            draw.line([(x1, y1), (x1 + corner_len, y1)], fill=active_color, width=corner_w)
            draw.line([(x1, y1), (x1, y1 + corner_len)], fill=active_color, width=corner_w)
            # Sağ Üst
            draw.line([(x2, y1), (x2 - corner_len, y1)], fill=active_color, width=corner_w)
            draw.line([(x2, y1), (x2, y1 + corner_len)], fill=active_color, width=corner_w)
            # Sol Alt
            draw.line([(x1, y2), (x1 + corner_len, y2)], fill=active_color, width=corner_w)
            draw.line([(x1, y2), (x1, y2 - corner_len)], fill=active_color, width=corner_w)
            # Sağ Alt
            draw.line([(x2, y2), (x2 - corner_len, y2)], fill=active_color, width=corner_w)
            draw.line([(x2, y2), (x2, y2 - corner_len)], fill=active_color, width=corner_w)

            # --- 3. BOUNDING BOX ETİKETİ ---
            if is_confident:
                label_text = f" {class_name} | %{confidence * 100:.1f} "
            else:
                label_text = " ELİNİZİ BURAYA YERLEŞTİRİN "

            # Etiket arka plan hapı
            label_y = max(hud_height + 4, y1 - 24)
            label_w = len(label_text) * 7 + 10
            draw.rectangle([x1, label_y, x1 + label_w, label_y + 20], fill=(15, 20, 28))
            draw.rectangle([x1, label_y, x1 + label_w, label_y + 20], outline=active_color, width=1)
            draw.text((x1 + 4, label_y + 3), label_text, fill=active_color)

        return img


if __name__ == "__main__":
    print("AirTouch Engine Test Ediliyor...")
    engine = AirTouchEngine(confidence_threshold=0.70, cooldown_seconds=0.5)
    
    # Boş bir kare testi
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    result = engine.process_frame(dummy_frame, trigger_enabled=False)
    
    print(f"Tahmin Edilen Sınıf: {result['class_name']} ({result['class_id']})")
    print(f"Güven Skoru: %{result['confidence'] * 100:.2f}")
    print(f"Olasılık Dağılımı Uzunluğu: {len(result['all_probabilities'])}")
    print(f"Tetikleme Eşiği Üstünde mi: {result['is_confident']}")
    print("Motor Başarıyla Test Edildi.")

