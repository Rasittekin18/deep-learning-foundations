"""
AirTouch PyTorch Derin Öğrenme Modelleri (model.py)
1. AirTouchCNN: Kurs Sign Language (64x64 gri tonlamalı) CNN sınıflandırıcısı.
2. AirTouchLandmarkNet: 21 nokta 3D el iskeleti + geometrik özniteliklerden (88 boyut)
   ışık/arka plan bağımsız canlı jest tanıyan PyTorch Residual MLP ağı.
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
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.40)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout_fc(x)
        return self.fc2(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """İleri yayılım sonrası her sınıf için [0, 1] arası Softmax olasılıkları üretir."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class AirTouchLandmarkNet(nn.Module):
    """
    21 el eklemi (63 normalize 3D koordinat) + 25 biyomekanik açı/oran özniteliği
    üzerinden çalışan hafif ve yüksek doğruluklu PyTorch Residual MLP sınıflandırıcısı.
    """

    def __init__(self, input_dim: int = 88, num_classes: int = 8):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )
        # Residual blok
        self.res_fc1 = nn.Linear(128, 128)
        self.res_bn1 = nn.BatchNorm1d(128)
        self.res_drop = nn.Dropout(0.20)
        self.res_fc2 = nn.Linear(128, 128)
        self.res_bn2 = nn.BatchNorm1d(128)

        # Sınıflandırma başlığı
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        residual = h
        out = F.relu(self.res_bn1(self.res_fc1(h)))
        out = self.res_drop(out)
        out = self.res_bn2(self.res_fc2(out))
        h = F.relu(out + residual)
        return self.head(h)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """İleri yayılım sonrası Softmax olasılık dağılımını döndürür."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


if __name__ == "__main__":
    print("AirTouch Modelleri Test Ediliyor...")
    cnn = AirTouchCNN(num_classes=10)
    lnet = AirTouchLandmarkNet(input_dim=88, num_classes=8)

    x_img = torch.randn(4, 1, 64, 64)
    x_lm = torch.randn(4, 88)
    print("CNN çıktı:", cnn(x_img).shape, "Olasılık toplamı:", cnn.predict_probabilities(x_img)[0].sum().item())
    print("LandmarkNet çıktı:", lnet(x_lm).shape, "Olasılık toplamı:", lnet.predict_probabilities(x_lm)[0].sum().item())
