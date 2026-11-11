"""
AirTouch PyTorch CNN Model Mimarisi (model.py)
Gerçek zamanlı kamera akışında düşük gecikme ve yüksek doğruluk için
optimize edilmiş evrişimli sinir ağı (Convolutional Neural Network).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AirTouchCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AirTouchCNN, self).__init__()
        
        # 1. Evrişim Bloğu: (1, 64, 64) -> (16, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.20)
        
        # 2. Evrişim Bloğu: (16, 32, 32) -> (32, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 3. Evrişim Bloğu: (32, 16, 16) -> (64, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Tam Bağlantılı (Fully Connected / Dense) Sınıflandırıcı
        # 64 kanal * 8 * 8 = 4096 öznitelik
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.40)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Blok 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Blok 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # Blok 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Flatten (Düzleştirme)
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout_fc(x)
        logits = self.fc2(x)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """İleri yayılım sonrası her sınıf için [0, 1] arası Softmax olasılıkları üretir."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


if __name__ == "__main__":
    print("AirTouchCNN Mimarisi Test Ediliyor...")
    model = AirTouchCNN(num_classes=10)
    print(model)
    
    # Test Tensörü (Batch: 4, Kanal: 1, Boyut: 64x64)
    dummy_input = torch.randn(4, 1, 64, 64)
    output = model(dummy_input)
    probs = model.predict_probabilities(dummy_input)
    
    print(f"Girdi Şekli: {dummy_input.shape}")
    print(f"Çıktı Logits Şekli: {output.shape}")
    print(f"Olasılıklar Toplamı (Batch 0): {probs[0].sum().item():.4f}")
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam Eğitilebilir Parametre Sayısı: {total_params:,}")
    print("Model Mimarisi Başarıyla Doğrulandı.")

