"""
Evaluasi Komprehensif Model Kosakata WL-BISINDO (32 Kelas)
Menghitung Akurasi, Macro-F1, dan menghasilkan Confusion Matrix Plot
untuk presentasi lomba / paper ilmiah.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from inference import get_engine

def run_evaluation():
    print("=" * 60)
    print("EVALUASI PERFORMA MODEL BISINDO (32 KELAS)")
    print("=" * 60)

    engine = get_engine()
    data_path = os.path.join(os.path.dirname(__file__), '..', 'dataset_raw', 'wl_bisindo_real_features.npz')
    if not os.path.exists(data_path):
        print(f"[!] File dataset {data_path} tidak ditemukan!")
        return

    data = np.load(data_path)
    X_test, y_test = data['X_test'], data['y_test']
    labels = engine.labels
    class_names = [item['display'] for item in labels]

    print(f"[*] Menguji {len(X_test)} sampel uji (Test Set)...")

    y_pred = []
    y_true = list(y_test)

    for i in range(len(X_test)):
        res = engine.predict(X_test[i])
        y_pred.append(res['class_id'])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    acc = np.mean(y_pred == y_true) * 100.0
    print(f"\n[✓] OVERALL TEST ACCURACY: {acc:.2f}%\n")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Buat direktori docs untuk asset lomba/paper
    docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    os.makedirs(docs_dir, exist_ok=True)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'WL-BISINDO 32-Classes Confusion Matrix (Akurasi: {acc:.2f}%)', fontsize=14, pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    plt.tight_layout()
    plt.ylabel('Ground Truth (Kelas Asli)', fontsize=11)
    plt.xlabel('Predicted Label (Hasil Prediksi)', fontsize=11)

    cm_path = os.path.join(docs_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[✓] Confusion Matrix Plot tersimpan di: {cm_path}")

    # Simpan ringkasan metrik
    summary = {
        "test_accuracy": round(float(acc), 2),
        "total_test_samples": len(y_test),
        "num_classes": len(labels)
    }
    with open(os.path.join(docs_dir, 'eval_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    run_evaluation()
