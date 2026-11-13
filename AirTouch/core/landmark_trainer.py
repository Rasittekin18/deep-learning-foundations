"""
AirTouch İskelet AI (LandmarkNet) Veri Üretici & PyTorch Eğitim Motoru (landmark_trainer.py)
21 eklemli 3D biyomekanik el kinematik modeli ile 8 temel jest için sentetik eğitim verisi
üretir, kullanıcının web kameradan kaydettiği gerçek örneklerle birleştirir ve
AirTouchLandmarkNet PyTorch modelini eğitir.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    LANDMARK_FEATURE_DIM,
    LANDMARK_HISTORY_PATH,
    LANDMARK_MODEL_PATH,
    LANDMARK_NUM_CLASSES,
    USER_LANDMARK_SAMPLES_FILE,
)
from core.hand_tracker import extract_hand_features
from core.model import AirTouchLandmarkNet


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def _build_finger_chain(
    mcp_pos: np.ndarray,
    splay_angle: float,
    lengths: Tuple[float, float, float],
    curl: float,
) -> List[np.ndarray]:
    """
    Tek bir parmağın MCP'den uca (PIP, DIP, TIP) 3D eklemlerini üretir.
    curl=0.0 -> tam açık (yukarı -Y), curl=1.0 -> avuç içine tam kıvrılmış.
    """
    # Temel yönelim yukarı (-Y) ve splay_angle kadar yana açılma
    # Kıvrılma arttıkça Y yönünden Z (öne/avuca) ve +Y (aşağı) yönüne bükülür
    a1 = curl * np.deg2rad(88.0)   # MCP bükülmesi
    a2 = curl * np.deg2rad(105.0)  # PIP bükülmesi
    a3 = curl * np.deg2rad(70.0)   # DIP bükülmesi

    pts = [mcp_pos]
    angles = [a1, a1 + a2, a1 + a2 + a3]
    for length, theta in zip(lengths, angles):
        # splay_angle küçük yan açılma, theta öne (+Z/-Z) ve aşağı kıvrılma
        dx = np.sin(splay_angle) * length * np.cos(theta * 0.5)
        dy = -np.cos(splay_angle) * length * np.cos(theta)
        dz = -length * np.sin(theta) * 0.85
        pts.append(pts[-1] + np.array([dx, dy, dz], dtype=np.float32))
    return pts


def generate_synthetic_hand_landmarks(
    gesture_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Verilen jest sınıfı (0..7) için biyomekanik olarak gerçekçi (21, 3) el eklemi üretir.
    """
    # El boyut oranlarında doğal varyasyon
    palm_w = rng.uniform(0.92, 1.08)
    palm_h = rng.uniform(0.92, 1.08)

    wrist = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    idx_mcp = np.array([0.28 * palm_w, -0.95 * palm_h, 0.0], dtype=np.float32)
    mid_mcp = np.array([0.02 * palm_w, -1.02 * palm_h, 0.0], dtype=np.float32)
    rng_mcp = np.array([-0.22 * palm_w, -0.96 * palm_h, 0.0], dtype=np.float32)
    pnk_mcp = np.array([-0.45 * palm_w, -0.82 * palm_h, 0.0], dtype=np.float32)

    cmc = np.array([0.36 * palm_w, -0.28 * palm_h, -0.04], dtype=np.float32)
    th_mcp = np.array([0.58 * palm_w, -0.58 * palm_h, -0.08], dtype=np.float32)

    # Sınıf bazlı parmak kıvrılma hedefleri: [index, middle, ring, pinky]
    if gesture_id == 0:  # Açık El
        curls = [rng.uniform(0.0, 0.14) for _ in range(4)]
        thumb_mode = "open"
        splays = [rng.uniform(0.14, 0.28), rng.uniform(-0.04, 0.06), rng.uniform(-0.18, -0.08), rng.uniform(-0.34, -0.20)]
    elif gesture_id == 1:  # Yumruk
        curls = [rng.uniform(0.82, 1.08) for _ in range(4)]
        thumb_mode = rng.choice(["tucked", "closed_side"])
        splays = [rng.uniform(-0.04, 0.06) for _ in range(4)]
    elif gesture_id == 2:  # 1 Parmak (İşaret)
        curls = [rng.uniform(0.0, 0.14), rng.uniform(0.82, 1.06), rng.uniform(0.84, 1.06), rng.uniform(0.84, 1.06)]
        thumb_mode = rng.choice(["tucked", "closed_side"])
        splays = [rng.uniform(-0.04, 0.08), 0.0, -0.04, -0.08]
    elif gesture_id == 3:  # 2 Parmak (Zafer / V)
        curls = [rng.uniform(0.0, 0.14), rng.uniform(0.0, 0.14), rng.uniform(0.84, 1.06), rng.uniform(0.84, 1.06)]
        thumb_mode = rng.choice(["tucked", "closed_side", "relaxed_open"])
        splays = [rng.uniform(0.12, 0.25), rng.uniform(-0.14, -0.03), -0.05, -0.10]
    elif gesture_id == 4:  # 3 Parmak (İşaret+Orta+Yüzük VEYA Başparmak+İşaret+Orta)
        if rng.random() < 0.72:
            curls = [rng.uniform(0.0, 0.14), rng.uniform(0.0, 0.14), rng.uniform(0.0, 0.15), rng.uniform(0.82, 1.05)]
            thumb_mode = "tucked"
            splays = [rng.uniform(0.10, 0.20), rng.uniform(-0.02, 0.04), rng.uniform(-0.18, -0.08), -0.10]
        else:
            curls = [rng.uniform(0.0, 0.14), rng.uniform(0.0, 0.14), rng.uniform(0.84, 1.05), rng.uniform(0.84, 1.05)]
            thumb_mode = "open"
            splays = [rng.uniform(0.08, 0.18), rng.uniform(-0.12, -0.02), -0.05, -0.10]
    elif gesture_id == 5:  # 4 Parmak
        curls = [rng.uniform(0.0, 0.15) for _ in range(4)]
        thumb_mode = "tucked"
        splays = [rng.uniform(0.10, 0.20), rng.uniform(-0.02, 0.05), rng.uniform(-0.14, -0.05), rng.uniform(-0.28, -0.15)]
    elif gesture_id == 6:  # OK / Çimdik
        curls = [rng.uniform(0.56, 0.72), rng.uniform(0.02, 0.18), rng.uniform(0.02, 0.18), rng.uniform(0.02, 0.20)]
        thumb_mode = "pinch"
        splays = [-0.04, rng.uniform(-0.03, 0.06), rng.uniform(-0.14, -0.05), rng.uniform(-0.28, -0.15)]
    elif gesture_id == 7:  # Başparmak Yukarı (Thumbs Up)
        curls = [rng.uniform(0.84, 1.06) for _ in range(4)]
        thumb_mode = "thumbs_up"
        splays = [0.0, 0.0, 0.0, 0.0]
    else:
        raise ValueError(f"Geçersiz jest sınıfı: {gesture_id}")

    f_lens = [
        (0.42, 0.27, 0.22),
        (0.46, 0.30, 0.24),
        (0.42, 0.28, 0.22),
        (0.34, 0.22, 0.18),
    ]
    mcps = [idx_mcp, mid_mcp, rng_mcp, pnk_mcp]
    finger_pts = [
        _build_finger_chain(m, s, ln, c) for m, s, ln, c in zip(mcps, splays, f_lens, curls)
    ]

    # Başparmak eklemleri (IP=3, TIP=4)
    if thumb_mode == "open":
        th_ip = th_mcp + np.array([rng.uniform(0.20, 0.30), rng.uniform(-0.28, -0.16), -0.03], dtype=np.float32)
        th_tip = th_ip + np.array([rng.uniform(0.16, 0.26), rng.uniform(-0.24, -0.12), -0.02], dtype=np.float32)
    elif thumb_mode == "closed_side":
        th_ip = th_mcp + np.array([0.05, -0.22, -0.06], dtype=np.float32)
        th_tip = th_ip + np.array([-0.14, -0.15, -0.05], dtype=np.float32)
    elif thumb_mode == "relaxed_open":
        th_ip = th_mcp + np.array([rng.uniform(0.14, 0.22), rng.uniform(-0.20, -0.10), -0.03], dtype=np.float32)
        th_tip = th_ip + np.array([rng.uniform(0.08, 0.16), rng.uniform(-0.16, -0.06), -0.02], dtype=np.float32)
    elif thumb_mode == "tucked":
        th_ip = th_mcp + np.array([-0.24, -0.26, -0.10], dtype=np.float32)
        th_tip = th_ip + np.array([-0.34, -0.10, -0.06], dtype=np.float32)
    elif thumb_mode == "pinch":
        idx_tip = finger_pts[0][3]
        target = idx_tip + rng.normal(0.0, 0.04, size=3).astype(np.float32)
        th_ip = 0.52 * th_mcp + 0.48 * target + np.array([0.05, 0.04, -0.05], dtype=np.float32)
        th_tip = target
    elif thumb_mode == "thumbs_up":
        th_ip = th_mcp + np.array([rng.uniform(0.16, 0.26), rng.uniform(-0.32, -0.20), -0.02], dtype=np.float32)
        th_tip = th_ip + np.array([rng.uniform(0.12, 0.22), rng.uniform(-0.28, -0.18), -0.01], dtype=np.float32)
    else:
        th_ip = th_mcp + np.array([0.1, -0.2, 0.0], dtype=np.float32)
        th_tip = th_ip + np.array([0.1, -0.2, 0.0], dtype=np.float32)

    pts = np.vstack([
        wrist,
        cmc, th_mcp, th_ip, th_tip,
        *finger_pts[0],
        *finger_pts[1],
        *finger_pts[2],
        *finger_pts[3],
    ]).astype(np.float32)

    # 3D Rotasyon (Başparmak Yukarı doğal olarak yatay eksende döner!)
    if gesture_id == 7:
        # Avuç yatay (+X veya -X yönlü), başparmak yukarı (-Y)!
        base_rz = np.deg2rad(-72.0 + rng.uniform(-16.0, 16.0))
        rx = np.deg2rad(rng.uniform(-18.0, 18.0))
        ry = np.deg2rad(rng.uniform(-22.0, 22.0))
        R = _rotation_matrix(rx, ry, base_rz)
    else:
        rx = np.deg2rad(rng.uniform(-18.0, 18.0))
        ry = np.deg2rad(rng.uniform(-24.0, 24.0))
        rz = np.deg2rad(rng.uniform(-20.0, 20.0))
        R = _rotation_matrix(rx, ry, rz)

    pts = (pts @ R.T).astype(np.float32)

    # Sağ / Sol el rastgele aynalama
    if rng.random() < 0.5:
        pts[:, 0] = -pts[:, 0]

    # Ölçüm gürültüsü + ekran koordinatına projeksiyon
    pts += rng.normal(0.0, 0.016, size=pts.shape).astype(np.float32)
    scale = rng.uniform(0.08, 0.20)
    offset = np.array([rng.uniform(0.25, 0.75), rng.uniform(0.35, 0.80), 0.0], dtype=np.float32)
    pts = pts * scale + offset
    return pts


def build_landmark_dataset(
    samples_per_class: int = 750,
    seed: int = 42,
    include_user_samples: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Sentetik biyomekanik veriyi ve varsa kullanıcının gerçek kamera örneklerini birleştirir.
    Döndürür: (X_features (N, 88), y_labels (N,), user_sample_count)
    """
    rng = np.random.default_rng(seed)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for gid in range(LANDMARK_NUM_CLASSES):
        for _ in range(samples_per_class):
            lm = generate_synthetic_hand_landmarks(gid, rng)
            feat = extract_hand_features(lm, aspect_ratio=1.0)
            X_list.append(feat)
            y_list.append(gid)

    user_count = 0
    if include_user_samples and USER_LANDMARK_SAMPLES_FILE.exists():
        try:
            data = np.load(str(USER_LANDMARK_SAMPLES_FILE))
            u_x = data["features"].astype(np.float32)
            u_y = data["labels"].astype(np.int64)
            user_count = int(len(u_y))
            if user_count > 0:
                # Kullanıcının gerçek el örneklerini hafif gürültüyle ağırlıklandır
                repeats = max(3, min(12, 150 // max(1, user_count // LANDMARK_NUM_CLASSES)))
                for _ in range(repeats):
                    noisy = u_x.copy()
                    noisy[:, :63] += rng.normal(0.0, 0.012, size=(user_count, 63)).astype(np.float32)
                    X_list.extend(list(noisy))
                    y_list.extend(list(u_y))
        except Exception as ex:
            print(f"[LandmarkTrainer] Kullanıcı örnekleri okunamadı: {ex}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, user_count


def save_user_landmark_samples(new_features: np.ndarray, new_labels: np.ndarray) -> int:
    """Kullanıcının kameradan kaydettiği gerçek örnekleri diskteki dosyaya ekler."""
    USER_LANDMARK_SAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_f = np.asarray(new_features, dtype=np.float32)
    new_l = np.asarray(new_labels, dtype=np.int64)

    if USER_LANDMARK_SAMPLES_FILE.exists():
        try:
            old = np.load(str(USER_LANDMARK_SAMPLES_FILE))
            old_f = old["features"].astype(np.float32)
            old_l = old["labels"].astype(np.int64)
            merged_f = np.vstack([old_f, new_f])
            merged_l = np.concatenate([old_l, new_l])
        except Exception:
            merged_f, merged_l = new_f, new_l
    else:
        merged_f, merged_l = new_f, new_l

    np.savez_compressed(str(USER_LANDMARK_SAMPLES_FILE), features=merged_f, labels=merged_l)
    return int(len(merged_l))


def get_user_sample_counts() -> Dict[int, int]:
    """Kayıtlı kullanıcı örneklerinin sınıf başına adetlerini döndürür."""
    counts = {i: 0 for i in range(LANDMARK_NUM_CLASSES)}
    if USER_LANDMARK_SAMPLES_FILE.exists():
        try:
            data = np.load(str(USER_LANDMARK_SAMPLES_FILE))
            labels = data["labels"].astype(np.int64)
            vals, cnts = np.unique(labels, return_counts=True)
            for v, c in zip(vals, cnts):
                counts[int(v)] = int(c)
        except Exception:
            pass
    return counts


def train_landmark_model(
    epochs: int = 28,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    samples_per_class: int = 750,
    device: Optional[str] = None,
) -> Tuple[AirTouchLandmarkNet, Dict]:
    """AirTouchLandmarkNet PyTorch modelini eğitir ve en iyi ağırlıkları kaydeder."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, user_count = build_landmark_dataset(samples_per_class=samples_per_class)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AirTouchLandmarkNet(
        input_dim=LANDMARK_FEATURE_DIM,
        num_classes=LANDMARK_NUM_CLASSES,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": epochs,
        "user_samples": user_count,
        "total_samples": int(len(y)),
    }

    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * bx.size(0)
            tr_correct += (out.argmax(dim=1) == by).sum().item()
            tr_total += by.size(0)

        scheduler.step()
        e_tr_loss = tr_loss / tr_total
        e_tr_acc = 100.0 * tr_correct / tr_total

        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                va_loss += loss.item() * bx.size(0)
                va_correct += (out.argmax(dim=1) == by).sum().item()
                va_total += by.size(0)

        e_va_loss = va_loss / va_total
        e_va_acc = 100.0 * va_correct / va_total

        history["train_loss"].append(round(e_tr_loss, 4))
        history["train_acc"].append(round(e_tr_acc, 2))
        history["val_loss"].append(round(e_va_loss, 4))
        history["val_acc"].append(round(e_va_acc, 2))

        if e_va_acc >= best_val_acc:
            best_val_acc = e_va_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": round(best_val_acc, 2),
                    "input_dim": LANDMARK_FEATURE_DIM,
                    "num_classes": LANDMARK_NUM_CLASSES,
                    "user_samples": user_count,
                },
                str(LANDMARK_MODEL_PATH),
            )

    history["best_val_acc"] = round(best_val_acc, 2)
    history["duration_sec"] = round(time.time() - t0, 2)

    with open(str(LANDMARK_HISTORY_PATH), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(
        f"LandmarkNet Eğitimi Tamamlandı ({history['duration_sec']}s) | "
        f"Val Acc: %{best_val_acc:.2f} | Kullanıcı Örnekleri: {user_count}"
    )
    return model, history


if __name__ == "__main__":
    train_landmark_model()
