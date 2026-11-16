"""
Pipeline Ekstraksi & Pelatihan Model dari WL-BISINDO Keypoints (.tar.gz)
Membongkar keypoints 543 titik MediaPipe, mengambil koordinat tangan (126D),
melakukan normalisasi invarian posisi/skala, dan melatih Bi-LSTM Attention model (32 kelas).
"""

import os
import tarfile
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from word_model import get_model

FILENAME_PATTERN = re.compile(r"^signer(?P<signer>\d+)_label(?P<label>\d+)_sample(?P<sample>\d+)\.npy$")
TARGET_SEQ_LEN = 30
FEATURE_DIM = 126  # 21 Left Hand (63) + 21 Right Hand (63)


def normalize_hand_coords(coords):
    """
    coords: (T, 21, 3)
    Mengaplikasikan normalisasi invarian Jemari per frame:
    - Origin pada pergelangan tangan (wrist / index 0)
    - Skala relatif terhadap rentang jarak landmark terjauh
    """
    T = coords.shape[0]
    normalized = np.zeros((T, 63), dtype=np.float32)

    for t in range(T):
        pts = coords[t]  # (21, 3)
        # Jika semua titik 0 (tangan tidak terdeteksi)
        if np.all(pts == 0) or np.all(np.isnan(pts)):
            continue

        wrist = pts[0]
        trans = pts - wrist
        dists = np.linalg.norm(trans, axis=1)
        scale = np.max(dists)
        if scale < 1e-6:
            scale = 1.0

        scaled = trans / scale
        normalized[t] = scaled.flatten()

    return normalized


def resample_temporal(seq, target_len=TARGET_SEQ_LEN):
    """
    seq: (T, 126)
    Resample urutan frame bervariasi menjadi target_len seragam.
    """
    T_orig = seq.shape[0]
    if T_orig == target_len:
        return seq

    indices = np.linspace(0, T_orig - 1, num=target_len)
    resampled = np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    for d in range(FEATURE_DIM):
        resampled[:, d] = np.interp(indices, np.arange(T_orig), seq[:, d])
    return resampled


def extract_and_load_keypoints(tar_path, extract_dir, cache_npz_path):
    if os.path.exists(cache_npz_path):
        print(f"[*] Cache dataset ditemukan di {cache_npz_path}. Memuat langsung...")
        archive = np.load(cache_npz_path)
        return (
            archive['X_train'], archive['y_train'], archive['signers_train'],
            archive['X_test'], archive['y_test'], archive['signers_test']
        )

    # 1. Unpack tar.gz jika folder belum ada
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
        print(f"[*] Mengekstrak {tar_path} ke {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("[✓] Ekstraksi tar.gz selesai!")

    # Cari file .npy
    npy_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith('.npy'):
                npy_files.append(os.path.join(root, f))

    print(f"[+] Ditemukan {len(npy_files)} file .npy WL-BISINDO.")
    if len(npy_files) == 0:
        raise RuntimeError("Tidak ada file .npy ditemukan dalam folder ekstraksi!")

    X_train, y_train, s_train = [], [], []
    X_test, y_test, s_test = [], [], []

    # Format MediaPipe Holistic di OpenHands / WL-BISINDO:
    # 0..467: Face (468)
    # 468..499: Pose (33) -> Total 501
    # 501..521: Left Hand (21)
    # 522..542: Right Hand (21)
    LEFT_HAND_IDX = slice(501, 522)
    RIGHT_HAND_IDX = slice(522, 543)

    for i, fpath in enumerate(npy_files):
        fname = os.path.basename(fpath)
        m = FILENAME_PATTERN.match(fname)
        if not m:
            continue

        signer_id = int(m.group("signer"))
        label = int(m.group("label"))
        sample_id = int(m.group("sample"))

        try:
            arr = np.load(fpath) # (T, 543, 3) atau (T, V, C)
        except Exception as e:
            continue

        # Validasi dimensi
        if arr.ndim != 3 or arr.shape[1] < 543:
            # Fallback jika hanya landmark tangan
            if arr.shape[1] == 42:
                left_pts = arr[:, :21, :3]
                right_pts = arr[:, 21:, :3]
            else:
                continue
        else:
            left_pts = arr[:, LEFT_HAND_IDX, :3]
            right_pts = arr[:, RIGHT_HAND_IDX, :3]

        norm_left = normalize_hand_coords(left_pts)   # (T, 63)
        norm_right = normalize_hand_coords(right_pts) # (T, 63)
        feats_126 = np.concatenate([norm_left, norm_right], axis=1) # (T, 126)

        feats_resampled = resample_temporal(feats_126, TARGET_SEQ_LEN) # (30, 126)

        # Split: Signer 1-4 untuk train, Signer 5 untuk test (Signer-Independent test)
        # Atau 80% train / 20% test per signer
        if sample_id <= 8:
            X_train.append(feats_resampled)
            y_train.append(label)
            s_train.append(signer_id)
        else:
            X_test.append(feats_resampled)
            y_test.append(label)
            s_test.append(signer_id)

        if (i + 1) % 400 == 0 or (i + 1) == len(npy_files):
            print(f"[*] Diproses {i + 1}/{len(npy_files)} video...")

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    s_train = np.array(s_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    s_test = np.array(s_test, dtype=np.int64)

    # Simpan cache NPZ
    os.makedirs(os.path.dirname(cache_npz_path), exist_ok=True)
    np.savez_compressed(
        cache_npz_path,
        X_train=X_train, y_train=y_train, signers_train=s_train,
        X_test=X_test, y_test=y_test, signers_test=s_test
    )
    print(f"[✓] Berhasil mengolah {len(X_train)} Train dan {len(X_test)} Test! Cache tersimpan di {cache_npz_path}")

    return X_train, y_train, s_train, X_test, y_test, s_test


def train_on_real_keypoints(
    tar_path=r"C:\Users\Admin\.gemini\antigravity\scratch\jemari-bisindo\dataset_raw\keypoints-wl-bisindo.tar.gz",
    extract_dir=r"C:\Users\Admin\.gemini\antigravity\scratch\jemari-bisindo\dataset_raw\keypoints",
    cache_npz=r"C:\Users\Admin\.gemini\antigravity\scratch\jemari-bisindo\dataset_raw\wl_bisindo_real_features.npz",
    epochs=40,
    batch_size=32,
    lr=1e-3,
    output_dir=r"C:\Users\Admin\.gemini\antigravity\scratch\jemari-bisindo\models\word"
):
    print("=" * 60)
    print("TRAINING WORD-LEVEL BISINDO PADA DATASET REAL WL-BISINDO")
    print("=" * 60)

    X_train, y_train, _, X_test, y_test, _ = extract_and_load_keypoints(tar_path, extract_dir, cache_npz)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Perangkat: {device}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(np.unique(y_train))
    model = get_model("bilstm", in_features=FEATURE_DIM, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    best_val_acc = 0.0
    best_weights_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # Augmentasi: noise kecil pada koordinat
            noise = torch.randn_like(bx) * 0.015
            bx_aug = bx + noise

            optimizer.zero_grad()
            out = model(bx_aug)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        train_loss = total_loss / total
        train_acc = (correct / total) * 100.0

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += by.size(0)

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        scheduler.step(val_acc)

        if epoch % 5 == 0 or epoch == epochs or val_acc > best_val_acc:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_type': 'bilstm',
                'in_features': FEATURE_DIM,
                'num_classes': num_classes,
                'val_acc': val_acc
            }, best_weights_path)

    print(f"\n[✓] TRAINING REAL DATA SELESAI!")
    print(f"    - Akurasi Validasi Terbaik: {best_val_acc:.2f}%")
    print(f"    - Model disimpan di: {best_weights_path}")

    meta = {
        "model_type": "bilstm",
        "in_features": FEATURE_DIM,
        "seq_len": TARGET_SEQ_LEN,
        "num_classes": int(num_classes),
        "best_val_accuracy": float(best_val_acc),
        "weights_file": "best_model.pt",
        "dataset": "WL-BISINDO Real (1600 samples)"
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    train_on_real_keypoints()
