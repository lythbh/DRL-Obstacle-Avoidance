"""
CNN-based object detection model for CNN-LiDAR-SLAM.

Architecture (Figure 2 / Section III-D-2 of the paper):
    Input  →  3 × Conv1D(GELU, BatchNorm)  →  BiLSTM  →  Attention  →  Dense
    Output: 12 values = (cx_i, cy_i, r_i) for i = 1..4 object landmarks.

The 26-D feature vector per frame (scan stats + object geometry + IMU) is
treated as a sequence of length 26 with 1 input channel for the convolutional
stage; multiple frames form a temporal sequence for the BiLSTM.

Equations from the paper:
    (15)  F_l = σ(W_l * F_{l-1} + b_l)          -- Conv1D layers (GELU)
    (16)  α_t = softmax(e_t),  F̂ = Σ α_t F_t   -- attention
    (17)  output = Dense(F̂)                      -- final regression head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Attention module
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over the time axis.

    Equation (16):
        e_t = v^T tanh(W_a F_t + b_a)
        α_t = exp(e_t) / Σ_k exp(e_k)
        F̂   = Σ_t α_t F_t
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, seq_len, hidden_size) BiLSTM hidden states.

        Returns:
            context: (batch, hidden_size) weighted sum.
            weights: (batch, seq_len)     attention weights.
        """
        # e_t = v^T tanh(W_a h_t + b_a)
        energy = self.v(torch.tanh(self.W_a(h))).squeeze(-1)  # (B, T)
        weights = F.softmax(energy, dim=-1)                    # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return context, weights


# ─────────────────────────────────────────────────────────────────────────────
# Main CNN model
# ─────────────────────────────────────────────────────────────────────────────

class CNNObjectDetector(nn.Module):
    """
    Regression CNN for estimating up to 4 object centroids + radii.

    Input shape:  (batch, seq_len, 26)
                  seq_len = number of consecutive frames (temporal window).
    Output shape: (batch, 12)   = [cx1, cy1, r1, cx2, cy2, r2, cx3, cy3, r3,
                                    cx4, cy4, r4]

    Architecture:
      Per-frame encoder
        3 × (Conv1D → BatchNorm → GELU → GlobalAvgPool)
      Temporal modelling
        BiLSTM(hidden=lstm_hidden)
      Attention
        TemporalAttention
      Regression head
        Dense(128) → GELU → Dropout → Dense(12)
    """

    INPUT_DIM = 26
    OUTPUT_DIM = 12  # 4 objects × (cx, cy, r)

    def __init__(
        self,
        conv_channels: Tuple[int, int, int] = (64, 128, 64),
        lstm_hidden: int = 128,
        dense_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ── Per-frame 1D convolutional encoder  (Eq. 15) ────────────────────
        # Treat the 26-D feature vector as a length-26 sequence with 1 channel
        ch_in = 1
        conv_layers = []
        for ch_out in conv_channels:
            conv_layers += [
                nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch_out),
                nn.GELU(),
            ]
            ch_in = ch_out
        self.conv_encoder = nn.Sequential(*conv_layers)

        # Global average pool collapses the length dimension → (B*T, ch_out)
        self.gap = nn.AdaptiveAvgPool1d(1)

        conv_out_dim = conv_channels[-1]

        # ── Bi-directional LSTM  (temporal modelling) ───────────────────────
        self.bilstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        bilstm_out = lstm_hidden * 2  # bidirectional

        # ── Attention  (Eq. 16) ──────────────────────────────────────────────
        self.attention = TemporalAttention(hidden_size=bilstm_out)

        # ── Regression head  (Eq. 17) ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(bilstm_out, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, self.OUTPUT_DIM),
        )

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 26) float tensor.

        Returns:
            (batch, 12) predicted landmark parameters.
        """
        B, T, _ = x.shape

        # ── Conv encoder (applied frame-by-frame) ────────────────────────────
        # Reshape: (B*T, 1, 26)
        x_flat = x.view(B * T, 1, self.INPUT_DIM)
        feat = self.conv_encoder(x_flat)       # (B*T, C, 26)
        feat = self.gap(feat).squeeze(-1)       # (B*T, C)
        feat = feat.view(B, T, -1)             # (B, T, C)

        # ── BiLSTM ───────────────────────────────────────────────────────────
        h_all, _ = self.bilstm(feat)            # (B, T, 2*H)

        # ── Attention ────────────────────────────────────────────────────────
        context, _weights = self.attention(h_all)  # (B, 2*H)

        # ── Regression head ──────────────────────────────────────────────────
        out = self.head(context)                # (B, 12)
        return out

    # ── Convenience methods ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, feature_seq: np.ndarray) -> np.ndarray:
        """
        Run inference on a numpy feature sequence.

        Args:
            feature_seq: (seq_len, 26) float32 array.

        Returns:
            (12,) predicted values = [cx1,cy1,r1, …, cx4,cy4,r4].
        """
        self.eval()
        t = torch.from_numpy(feature_seq).float().unsqueeze(0)  # (1, T, 26)
        out = self(t)
        return out.squeeze(0).cpu().numpy()

    def get_landmarks(self, feature_seq: np.ndarray):
        """
        Predict and parse up to 4 landmark tuples (cx, cy, r).

        Args:
            feature_seq: (seq_len, 26) float32.

        Returns:
            List of (cx, cy, r) tuples (length 4, may contain zeros).
        """
        raw = self.predict(feature_seq)
        landmarks = []
        for i in range(4):
            cx = float(raw[i * 3])
            cy = float(raw[i * 3 + 1])
            r = float(abs(raw[i * 3 + 2]))
            landmarks.append((cx, cy, r))
        return landmarks


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkLoss(nn.Module):
    """
    Smooth-L1 (Huber) loss over the 12 landmark outputs.

    Using Huber loss instead of MSE improves robustness to annotation
    outliers in the training data.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


def build_model(
    conv_channels: Tuple[int, int, int] = (64, 128, 64),
    lstm_hidden: int = 128,
    dense_hidden: int = 128,
    dropout: float = 0.3,
) -> CNNObjectDetector:
    """Factory function matching paper hyper-parameters."""
    return CNNObjectDetector(
        conv_channels=conv_channels,
        lstm_hidden=lstm_hidden,
        dense_hidden=dense_hidden,
        dropout=dropout,
    )


def train_one_epoch(
    model: CNNObjectDetector,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Args:
        model:      CNNObjectDetector.
        dataloader: yields (feature_seq, targets) tensors.
        optimizer:  e.g. Adam with lr=1e-4 (as in paper Section III-F).
        device:     CPU or CUDA.

    Returns:
        Mean loss over the epoch.
    """
    model.train()
    criterion = LandmarkLoss()
    total_loss = 0.0
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)
