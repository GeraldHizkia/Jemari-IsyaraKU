"""
Definisi Arsitektur Model Pengenalan Kosakata BISINDO (32 Kelas)
Mendukung:
1. SignBiLSTMAttention (Baseline Efisien & Cepat untuk CPU & Edge Browser)
2. SignTransformer (Transformer Temporal berbasis SPOTER)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Mekanisme perhatian temporal terbobot untuk merangkum representasi sekuens
    menjadi satu vektor representasi gerakan tunggal (gesture embedding).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.proj(x).squeeze(-1) # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (B, T, 1)
        context = torch.sum(x * weights, dim=1) # (B, D)
        return context, weights


class SignBiLSTMAttention(nn.Module):
    """
    Arsitektur Bi-LSTM dengan Proyeksi Fitur Awal, Dropout, dan Temporal Attention.
    Sangat cocok untuk inferensi real-time CPU/Web karena latensi sangat rendah (<5ms).
    """
    def __init__(self, in_features=126, hidden_dim=128, num_layers=2, num_classes=32, dropout=0.3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Proyeksi dimensi fitur spasial per-frame
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Sekuens temporal dua arah
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = TemporalAttention(hidden_dim * 2)

        # Klasifikasi akhir
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None):
        # x: (B, T, in_features)
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        proj = self.feature_proj(x_flat).view(B, T, -1)

        lstm_out, _ = self.lstm(proj) # (B, T, hidden_dim * 2)
        context, attn_weights = self.attention(lstm_out, mask)
        logits = self.classifier(context) # (B, num_classes)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class SignTransformer(nn.Module):
    """
    Arsitektur Transformer berbasis SPOTER untuk pemrosesan sekuens skeleton/landmark.
    """
    def __init__(self, in_features=126, d_model=128, nhead=4, num_layers=3, num_classes=32, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, in_features)
        B, T, _ = x.shape
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, T+1, d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        cls_out = out[:, 0, :] # ambil output token [CLS]
        logits = self.fc(cls_out)
        return logits


def get_model(model_type="bilstm", in_features=126, num_classes=32):
    if model_type.lower() == "transformer":
        return SignTransformer(in_features=in_features, num_classes=num_classes)
    return SignBiLSTMAttention(in_features=in_features, num_classes=num_classes)


if __name__ == "__main__":
    # Smoke test model
    dummy_input = torch.randn(4, 30, 126) # 4 sampel, 30 frame, 126 dimensi fitur
    m1 = get_model("bilstm", in_features=126, num_classes=32)
    m2 = get_model("transformer", in_features=126, num_classes=32)

    out1 = m1(dummy_input)
    out2 = m2(dummy_input)
    print("SignBiLSTM output shape:", out1.shape)
    print("SignTransformer output shape:", out2.shape)
    assert out1.shape == (4, 32)
    assert out2.shape == (4, 32)
    print("Smoke test passed successfully!")
