"""
AirTouch Model Eğitim Motoru (trainer.py)
PyTorch CNN modelini eğitir, validation performansını ölçer ve
en yüksek doğruluktaki ağırlıkları kaydeder.
"""
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODELS_DIR
from core.model import AirTouchCNN
from core.dataset import get_data_loaders

def train_model(epochs: int = 15, batch_size: int = 32, learning_rate: float = 0.001, device: str = None):
    """
    AirTouchCNN modelini eğitir ve en iyi ağırlıkları kaydeder.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Eğitim cihazı: {device.upper()}")

    # Veri yükleyicileri al
    train_loader, val_loader, n_train, n_val = get_data_loaders(batch_size=batch_size)

    # Model, Kayıp Fonksiyonu ve Optimizer
    model = AirTouchCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": epochs
    }

    best_val_acc = 0.0
    model_save_path = MODELS_DIR / "airtouch_model.pth"
    history_save_path = MODELS_DIR / "training_history.json"

    start_time = time.time()
    print("=" * 60)
    print(f"AirTouch CNN Eğitimi Başlıyor ({epochs} Epoch)...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # --- EĞİTİM ADIMI ---
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # --- DOĞRULAMA (VALIDATION) ADIMI ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        scheduler.step(epoch_val_acc)

        # Geçmişe kaydet
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 2))

        # En iyi modeli kaydet
        saved_marker = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'num_classes': 10
            }, str(model_save_path))
            saved_marker = "⭐ (En İyi Model Kaydedildi)"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: %{epoch_train_acc:.2f} || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: %{epoch_val_acc:.2f} {saved_marker}")

    total_duration = time.time() - start_time
    print("=" * 60)
    print(f"Eğitim Tamamlandı! Süre: {total_duration:.1f} saniye")
    print(f"En Yüksek Doğruluk (Validation Accuracy): %{best_val_acc:.2f}")
    print(f"Ağırlıklar Kaydedildi: {model_save_path}")

    # Geçmişi JSON olarak kaydet
    with open(str(history_save_path), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    return model, history

if __name__ == "__main__":
    train_model(epochs=15, batch_size=32)

