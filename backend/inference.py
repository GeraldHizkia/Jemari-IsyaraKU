"""
Mesin Inferensi Real-Time untuk Pengenalan Kosakata BISINDO
Memuat checkpoint PyTorch dan melayani prediksi sekuens temporal.
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from word_model import get_model

TARGET_SEQ_LEN = 30
FEATURE_DIM = 126


class WordRecognizer:
    def __init__(self, model_path=None, labels_path=None, device="cpu"):
        self.device = torch.device(device)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        if model_path is None:
            model_path = os.path.join(self.base_dir, "..", "models", "word", "best_model.pt")
        if labels_path is None:
            labels_path = os.path.join(self.base_dir, "..", "models", "word", "labels.json")

        self.model_path = model_path
        self.labels_path = labels_path

        self.labels = self._load_labels(labels_path)
        self.model = self._load_model(model_path)

    def _load_labels(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Default fallback 32 classes
        return [{"id": i, "gloss": f"word_{i}", "display": f"Kata {i}"} for i in range(32)]

    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"[!] Checkpoint model tidak ditemukan di {path}. Menginisialisasi model default.")
            m = get_model("bilstm", in_features=FEATURE_DIM, num_classes=len(self.labels))
            m.to(self.device).eval()
            return m

        checkpoint = torch.load(path, map_location=self.device)
        model_type = checkpoint.get("model_type", "bilstm")
        in_features = checkpoint.get("in_features", FEATURE_DIM)
        num_classes = checkpoint.get("num_classes", len(self.labels))

        model = get_model(model_type, in_features=in_features, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        print(f"[✓] Model {model_type.upper()} ({num_classes} kelas) berhasil dimuat dari {path}")
        return model

    def resample_sequence(self, raw_seq, target_len=TARGET_SEQ_LEN):
        """
        Resampling urutan frame sembarang panjang menjadi tepat TARGET_SEQ_LEN
        menggunakan interpolasi linier sepanjang dimensi waktu.
        """
        raw_seq = np.asarray(raw_seq, dtype=np.float32)
        T_orig = raw_seq.shape[0]

        if T_orig == 0:
            return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)

        if T_orig == target_len:
            return raw_seq

        indices = np.linspace(0, T_orig - 1, num=target_len)
        resampled = np.zeros((target_len, raw_seq.shape[1]), dtype=np.float32)
        for d in range(raw_seq.shape[1]):
            resampled[:, d] = np.interp(indices, np.arange(T_orig), raw_seq[:, d])
        return resampled

    def predict(self, raw_sequence, top_k=3):
        """
        Melakukan prediksi kosakata dari buffer sekuens temporal.
        raw_sequence: List atau ndarray (T, 126)
        """
        if len(raw_sequence) < 5:
            return {
                "error": "Sekuens terlalu pendek (minimal 5 frame dibutuhkan)",
                "label": None,
                "confidence": 0.0
            }

        seq = self.resample_sequence(raw_sequence, TARGET_SEQ_LEN)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device) # (1, T, 126)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0) # (num_classes,)

        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.labels)))

        top_probs = top_probs.cpu().tolist()
        top_indices = top_indices.cpu().tolist()

        best_id = top_indices[0]
        best_conf = top_probs[0]
        best_meta = next((item for item in self.labels if item["id"] == best_id), {
            "id": best_id, "gloss": f"word_{best_id}", "display": f"Kata {best_id}"
        })

        alternatives = []
        for conf, idx in zip(top_probs, top_indices):
            m = next((item for item in self.labels if item["id"] == idx), {"gloss": str(idx), "display": str(idx)})
            alternatives.append({
                "id": idx,
                "gloss": m.get("gloss", ""),
                "display": m.get("display", ""),
                "category": m.get("category", ""),
                "confidence": round(conf, 4)
            })

        return {
            "class_id": best_id,
            "gloss": best_meta.get("gloss", ""),
            "label": best_meta.get("display", best_meta.get("gloss", "")),
            "category": best_meta.get("category", ""),
            "confidence": round(best_conf, 4),
            "top_k": alternatives
        }


# Singleton instance untuk serving
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = WordRecognizer()
    return _engine


if __name__ == "__main__":
    engine = get_engine()
    # Dummy sequence 25 frames
    test_seq = np.random.randn(25, 126).tolist()
    res = engine.predict(test_seq)
    print("Inference test result:", json.dumps(res, indent=2))
