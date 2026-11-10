"""
Ekstraksi Landmark Spatio-Temporal untuk Dataset WL-BISINDO
Menggunakan MediaPipe Holistic / Hands untuk mengekstrak urutan koordinat tangan
(Tangan Kiri 21 titik x 3 = 63, Tangan Kanan 21 titik x 3 = 63 -> Total 126 fitur per frame).

Normalisasi spasial:
1. Centering pergelangan tangan (wrist-relative).
2. Scale-invariance (dibagi jarak terpanjang landmark).
3. Resampling temporal seragam ke panjang sekuens tetap (misal T=30 frame).
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path

# Cek apakah mediapipe tersedia
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception as e:
    mp_hands = None

TARGET_SEQ_LEN = 30  # Jumlah frame tetap per gestur kata
FEATURE_DIM = 126    # 21 titik * 3 (kiri) + 21 titik * 3 (kanan)


def normalize_hand(landmarks):
    """
    Normalisasi koordinat 21 titik satu tangan.
    Sesuai prinsip invariant Jemari:
    - Terjemahkan sehingga wrist (titik 0) ada di (0,0,0).
    - Bagi dengan jarak Euclidean terjauh.
    """
    if landmarks is None or len(landmarks) == 0:
        return np.zeros(63, dtype=np.float32)

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32) # (21, 3)
    wrist = coords[0]
    translated = coords - wrist

    dists = np.linalg.norm(translated, axis=1)
    scale = np.max(dists)
    if scale < 1e-6:
        scale = 1.0

    normalized = translated / scale
    return normalized.flatten()


def extract_video_features(video_path, max_frames=TARGET_SEQ_LEN):
    """
    Ekstrak urutan fitur 126D dari satu file video .mp4.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Gagal membuka video: {video_path}")
        return None

    raw_frames_features = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Konversi warna BGR -> RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            left_hand_features = np.zeros(63, dtype=np.float32)
            right_hand_features = np.zeros(63, dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label # 'Left' atau 'Right'
                    norm_features = normalize_hand(hand_landmarks.landmark)
                    if label == 'Left':
                        left_hand_features = norm_features
                    else:
                        right_hand_features = norm_features

            frame_features = np.concatenate([left_hand_features, right_hand_features])
            raw_frames_features.append(frame_features)

    cap.release()

    if len(raw_frames_features) == 0:
        return np.zeros((max_frames, FEATURE_DIM), dtype=np.float32)

    raw_array = np.array(raw_frames_features, dtype=np.float32) # (T_orig, 126)

    # Resample temporal seragam ke TARGET_SEQ_LEN
    T_orig = raw_array.shape[0]
    if T_orig == max_frames:
        return raw_array
    else:
        # Interpolasi linier 1D di sepanjang dimensi waktu
        indices = np.linspace(0, T_orig - 1, num=max_frames)
        resampled = np.zeros((max_frames, FEATURE_DIM), dtype=np.float32)
        for d in range(FEATURE_DIM):
            resampled[:, d] = np.interp(indices, np.arange(T_orig), raw_array[:, d])
        return resampled


def process_dataset(organized_dir, output_file):
    """
    Mengekstrak seluruh video di direktori terorganisasi (train/test) dan
    menyimpan menjadi file arsip numpy terkompresi .npz.
    """
    X_train, y_train, ids_train = [], [], []
    X_test, y_test, ids_test = [], [], []

    for split in ['train', 'test']:
        split_dir = os.path.join(organized_dir, split)
        if not os.path.exists(split_dir):
            continue

        for label_str in sorted(os.listdir(split_dir)):
            label_dir = os.path.join(split_dir, label_str)
            if not os.path.isdir(label_dir):
                continue
            try:
                label_int = int(label_str)
            except ValueError:
                continue

            video_files = [f for f in os.listdir(label_dir) if f.endswith('.mp4')]
            print(f"[*] Memproses {split} - Label {label_int}: {len(video_files)} video...")

            for vid in video_files:
                vpath = os.path.join(label_dir, vid)
                feats = extract_video_features(vpath)
                if feats is not None:
                    if split == 'train':
                        X_train.append(feats)
                        y_train.append(label_int)
                        ids_train.append(vid)
                    else:
                        X_test.append(feats)
                        y_test.append(label_int)
                        ids_test.append(vid)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        ids_train=np.array(ids_train),
        X_test=X_test,
        y_test=y_test,
        ids_test=np.array(ids_test)
    )
    print(f"\n[✓] Ekstraksi selesai dan tersimpan di: {output_file}")
    print(f"    - Train: {X_train.shape}, Label: {y_train.shape}")
    print(f"    - Test:  {X_test.shape}, Label: {y_test.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract MediaPipe features from WL-BISINDO videos")
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                        help='Direktori dataset yang telah di-organize')
    parser.add_argument('--output', type=str, default='wl_bisindo_features.npz',
                        help='File output .npz')

    args = parser.parse_args()
    process_dataset(args.dataset_dir, args.output)
