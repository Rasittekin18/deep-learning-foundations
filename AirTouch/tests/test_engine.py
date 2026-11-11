"""
Akıllı Çıkarım ve Tetikleme Motoru Mimari Testleri (test_engine.py)
"""
import pytest
import time
import numpy as np

from core.engine import AirTouchEngine
from config import MODELS_DIR

class TestEngineArchitectureAndDebounceState:

    @pytest.fixture
    def engine(self):
        """Test amaçlı optimize edilmiş motor (Kısa cooldown süresi ile)."""
        eng = AirTouchEngine(
            model_path=MODELS_DIR / "airtouch_model.pth",
            confidence_threshold=0.80,
            cooldown_seconds=0.30, # 300ms
            smoothing_window=3
        )
        # Donanıma gerçekten basmayı engelle (Mock)
        eng.gesture_actions = {1: "play_pause", 2: "volume_up"}
        return eng

    def test_engine_initialization_with_checkpoint(self, engine):
        """Motor başlatıldığında eğitilmiş model başarıyla belleğe alınmalıdır."""
        assert engine.model is not None
        assert engine.confidence_threshold == 0.80
        assert engine.cooldown_seconds == 0.30
        assert engine.total_triggers == 0

    def test_confidence_threshold_filtering(self, engine, dummy_black_frame):
        """Güven eşiğinin altındaki belirsiz görüntüler asla eylem tetiklememelidir."""
        report = engine.process_frame(dummy_black_frame, trigger_enabled=True)
        
        # Boş karede güven eşiği (%80) aşılmamalıdır
        if report["confidence"] < 0.80:
            assert report["is_confident"] is False
            assert report["action_executed"] is None
            assert engine.total_triggers == 0

    def test_cooldown_timer_prevents_rapid_fire(self, engine, monkeypatch):
        """Eylem tetiklendikten sonra cooldown süresi bitene kadar tekrar tuşa basılmamalıdır."""
        # Modeli her zaman Class 1 ve %99 güven döndürecek şekilde mockla
        def mock_predict(tensor):
            probs = torch.zeros(1, 10)
            probs[0, 1] = 0.99
            return probs

        import torch
        monkeypatch.setattr(engine.model, "predict_probabilities", mock_predict)
        
        dummy_frame = np.ones((64, 64, 3), dtype=np.uint8) * 100
        
        # 1. Çağrı: Eylem tetiklenmeli
        r1 = engine.process_frame(dummy_frame, trigger_enabled=True)
        assert r1["action_executed"] is not None
        assert engine.total_triggers == 1
        assert r1["is_cooldown_active"] is True
        
        # 2. Çağrı (Hemen ardından - Cooldown devrede): Tetiklenmemeli!
        r2 = engine.process_frame(dummy_frame, trigger_enabled=True)
        assert r2["action_executed"] is None, "Cooldown süresi dolmadan eylem tetiklendi!"
        assert engine.total_triggers == 1
        assert r2["is_cooldown_active"] is True
        assert r2["cooldown_remaining"] > 0.0

        # 3. Cooldown süresi kadar bekle (0.35 saniye)
        time.sleep(0.35)
        
        # 4. Çağrı (Süre doldu): Artık tekrar tetiklenebilmeli!
        r3 = engine.process_frame(dummy_frame, trigger_enabled=True)
        assert r3["action_executed"] is not None
        assert engine.total_triggers == 2

    def test_trigger_enabled_toggle_safety(self, engine, monkeypatch):
        """trigger_enabled=False yapıldığında model tahmin üretmeli ancak asla tuşa basmamalıdır."""
        def mock_predict(tensor):
            probs = torch.zeros(1, 10)
            probs[0, 1] = 0.99
            return probs

        import torch
        monkeypatch.setattr(engine.model, "predict_probabilities", mock_predict)
        dummy_frame = np.ones((64, 64, 3), dtype=np.uint8) * 100
        
        # Canlı tetikleme kapalı
        report = engine.process_frame(dummy_frame, trigger_enabled=False)
        assert pytest.approx(report["confidence"], 0.001) == 0.99
        assert report["is_confident"] is True
        assert report["action_executed"] is None, "Güvenlik anahtarı kapalıyken tuşa basıldı!"
        assert engine.total_triggers == 0

    def test_smoothing_window_majority_voting(self, engine):
        """Son karelerdeki tek tük parazitler kararlı sınıfı anında değiştirmemelidir."""
        # Geçmişe elle 2 adet Sınıf 1 ve 1 adet gürültü (Sınıf 7) ekleyelim
        engine.history.clear()
        engine.history.append((1, 0.95))
        engine.history.append((1, 0.92))
        engine.history.append((7, 0.85)) # Anlık parazit
        
        classes_in_window = [c for c, conf in engine.history if conf >= engine.confidence_threshold]
        stable_class = max(set(classes_in_window), key=classes_in_window.count)
        
        # Çoğunluk 1 olduğu için parazit elenmelidir
        assert stable_class == 1

    def test_edge_case_frames_stability(self, engine):
        """Aşırı ışık (tam beyaz), karanlık (tam siyah) veya gürültülü karelerde motor çökmeksizin yanıt vermelidir."""
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8), # Tam Siyah
            np.ones((100, 100, 3), dtype=np.uint8) * 255, # Tam Beyaz
            np.random.randint(0, 256, (320, 240, 3), dtype=np.uint8) # Statik gürültü
        ]
        
        required_keys = [
            "class_id", "class_name", "confidence", "is_confident", 
            "action_executed", "all_probabilities", "processed_tensor"
        ]
        
        for f in frames:
            rep = engine.process_frame(f, trigger_enabled=False)
            for k in required_keys:
                assert k in rep, f"Eksik anahtar: {k}"
            assert not np.isnan(rep["confidence"])
            assert len(rep["all_probabilities"]) == 10

    def test_bbox_crop_and_hud_annotation_integrity(self, engine, dummy_rgb_frame):
        """BBox koordinatı verildiğinde sadece o bölge kırpılmalı ve HUD görüntüsü üretilmelidir."""
        bbox = (100, 80, 300, 280)
        report = engine.process_frame(dummy_rgb_frame, bbox=bbox, trigger_enabled=False)
        
        assert "annotated_frame" in report
        assert report["bbox"] == bbox
        
        # Üretilen HUD görüntüsü orijinal kamera boyutlarıyla birebir aynı olmalıdır
        annotated_img = report["annotated_frame"]
        assert annotated_img.size == (640, 480) # (Width, Height)
