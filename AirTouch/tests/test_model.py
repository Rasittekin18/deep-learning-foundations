"""
PyTorch CNN Model Mimarisi ve Gradyan Akışı Testleri (test_model.py)
"""
import pytest
from pathlib import Path
import torch
import torch.nn as nn

from core.model import AirTouchCNN, AirTouchLandmarkNet
from config import (
    LANDMARK_FEATURE_DIM,
    LANDMARK_MODEL_PATH,
    LANDMARK_NUM_CLASSES,
    MODELS_DIR,
    NUM_CLASSES,
)

class TestModelArchitectureAndGradientFlow:

    def test_forward_tensor_dimension_propagation(self, initialized_model):
        """Tensör boyutları katmanlar boyunca doğru dönüşmelidir: (B, 1, 64, 64) -> (B, 10)."""
        batch_sizes = [1, 4, 16]
        for b in batch_sizes:
            x = torch.randn(b, 1, 64, 64)
            logits = initialized_model(x)
            assert logits.shape == (b, NUM_CLASSES)
            assert not torch.isnan(logits).any()

    def test_gradient_flow_across_all_layers(self):
        """
        Geri Yayılım (Backpropagation) Doğrulaması:
        Her bir katmanın ağırlığı sıfır olmayan bir gradyan almalıdır (Ölü katman veya gradyan yok olması olmamalı).
        """
        model = AirTouchCNN(num_classes=NUM_CLASSES)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        dummy_input = torch.randn(8, 1, 64, 64)
        dummy_target = torch.randint(0, NUM_CLASSES, (8,))
        
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        
        # Tüm parametrelerin gradyanını kontrol et
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} katmanı gradyan almadı!"
                grad_norm = param.grad.norm().item()
                assert not torch.isnan(param.grad).any(), f"{name} katmanında NaN gradyan!"
                assert grad_norm > 0.0, f"{name} katmanında gradyan 0 (Ölü katman)!"

    def test_softmax_probabilities_mathematical_property(self, initialized_model):
        """Softmax olasılıkları [0, 1] aralığında olmalı ve toplamları kesinlikle 1.0 yapmalıdır."""
        x = torch.randn(8, 1, 64, 64)
        probs = initialized_model.predict_probabilities(x)
        
        assert probs.shape == (8, NUM_CLASSES)
        assert (probs >= 0.0).all().item()
        assert (probs <= 1.0).all().item()
        
        # Olasılık toplamları her örnek için 1.0 olmalıdır
        sums = probs.sum(dim=1)
        for s in sums:
            assert abs(s.item() - 1.0) < 1e-5, f"Olasılık toplamı 1.0 değil: {s.item()}"

    def test_eval_mode_determinism(self, initialized_model):
        """Eval modunda Dropout kapalı olduğu için aynı girdiye iki kez aynı çıktı üretilmelidir."""
        x = torch.randn(2, 1, 64, 64)
        out1 = initialized_model(x)
        out2 = initialized_model(x)
        assert torch.allclose(out1, out2, atol=1e-7), "Eval modunda model deterministik davranmadı!"

    def test_saved_checkpoint_integrity_and_accuracy(self):
        """Kaydedilen airtouch_model.pth dosyası modeli eksiksiz yüklemeli ve yüksek başarı bildirmelidir."""
        model_path = MODELS_DIR / "airtouch_model.pth"
        assert model_path.exists(), "Model ağırlık dosyası bulunamadı!"
        
        checkpoint = torch.load(str(model_path), map_location="cpu")
        assert 'model_state_dict' in checkpoint
        assert 'val_acc' in checkpoint
        
        # Doğruluk %90'ın üzerinde olmalıdır (biz %98.71 elde etmiştik)
        assert checkpoint['val_acc'] >= 90.0, f"Doğruluk beklenenden düşük: %{checkpoint['val_acc']}"
        
        # Ağırlıkları sıfır hatayla modele yükleyebilmeliyiz
        model = AirTouchCNN(num_classes=10)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'])
        assert len(missing_keys) == 0, f"Eksik katmanlar var: {missing_keys}"
        assert len(unexpected_keys) == 0, f"Beklenmeyen katmanlar var: {unexpected_keys}"


class TestLandmarkNetArchitectureAndCheckpoint:

    def test_landmark_forward_and_probabilities(self):
        """AirTouchLandmarkNet (B, 88) -> (B, 8) tensör boyutları ve Softmax toplamı 1.0 olmalıdır."""
        net = AirTouchLandmarkNet(input_dim=LANDMARK_FEATURE_DIM, num_classes=LANDMARK_NUM_CLASSES)
        net.eval()
        x = torch.randn(5, LANDMARK_FEATURE_DIM)
        logits = net(x)
        probs = net.predict_probabilities(x)
        assert logits.shape == (5, LANDMARK_NUM_CLASSES)
        assert probs.shape == (5, LANDMARK_NUM_CLASSES)
        assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)

    def test_landmark_gradient_flow(self):
        """AirTouchLandmarkNet geri yayılımda tüm eğitilebilir katmanlara sıfır olmayan gradyan iletmelidir."""
        net = AirTouchLandmarkNet(input_dim=LANDMARK_FEATURE_DIM, num_classes=LANDMARK_NUM_CLASSES)
        net.train()
        x = torch.randn(16, LANDMARK_FEATURE_DIM)
        y = torch.randint(0, LANDMARK_NUM_CLASSES, (16,))
        loss = nn.CrossEntropyLoss()(net(x), y)
        loss.backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} gradyan almadı!"
                assert param.grad.norm().item() > 0.0, f"{name} gradyanı sıfır!"

    def test_saved_landmark_checkpoint_integrity(self):
        """Kaydedilen airtouch_landmark_model.pth eksiksiz yüklenmeli ve >=%95 val_acc sunmalıdır."""
        assert LANDMARK_MODEL_PATH.exists(), "Landmark modeli bulunamadı!"
        ckpt = torch.load(str(LANDMARK_MODEL_PATH), map_location="cpu")
        assert ckpt["val_acc"] >= 95.0
        net = AirTouchLandmarkNet(input_dim=LANDMARK_FEATURE_DIM, num_classes=LANDMARK_NUM_CLASSES)
        missing, unexpected = net.load_state_dict(ckpt["model_state_dict"])
        assert len(missing) == 0 and len(unexpected) == 0


