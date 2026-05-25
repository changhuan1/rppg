"""RT-MRCPNet for contactless stress recognition.

The model consumes multi-region RGB temporal sequences with shape
``[batch, num_rois, clip_len, channels]`` and predicts stress/non-stress.
"""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Selects informative time positions within each ROI sequence."""

    def __init__(self, hidden_dim):
        super().__init__()
        attn_dim = max(hidden_dim // 2, 1)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x):
        # x: [B*K, T, D]
        score = self.attn(x).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        out = torch.sum(x * weight.unsqueeze(-1), dim=1)
        return out, weight


class RegionAttention(nn.Module):
    """Selects informative facial regions for each clip."""

    def __init__(self, hidden_dim):
        super().__init__()
        attn_dim = max(hidden_dim // 2, 1)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x):
        # x: [B, K, D]
        score = self.attn(x).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        out = torch.sum(x * weight.unsqueeze(-1), dim=1)
        return out, weight


class RTMRCPNet(nn.Module):
    """Region-temporal attention multi-region color pulse network."""

    def __init__(
        self,
        num_rois=5,
        in_channels=3,
        hidden_dim=128,
        num_classes=2,
        dropout=0.5,
        use_temporal_attention=True,
        use_region_attention=True,
        use_region_residual=True,
    ):
        super().__init__()
        self.num_rois = num_rois
        self.use_temporal_attention = use_temporal_attention
        self.use_region_attention = use_region_attention
        self.use_region_residual = use_region_residual

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_attn = TemporalAttention(hidden_dim)
        self.region_attn = RegionAttention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, K, T, C]
        if x.dim() != 4:
            raise ValueError(f"RTMRCPNet expects [B, K, T, C], got {tuple(x.shape)}")

        batch_size, num_rois, clip_len, channels = x.shape
        x = x.reshape(batch_size * num_rois, clip_len, channels).permute(0, 2, 1)

        h = self.temporal_encoder(x)  # [B*K, D, T]
        h = h.permute(0, 2, 1)  # [B*K, T, D]

        if self.use_temporal_attention:
            h, temporal_weight = self.temporal_attn(h)  # [B*K, D], [B*K, T]
        else:
            temporal_weight = torch.full(
                (h.shape[0], clip_len),
                1.0 / clip_len,
                dtype=h.dtype,
                device=h.device,
            )
            h = torch.mean(h, dim=1)
        h = h.reshape(batch_size, num_rois, -1)  # [B, K, D]

        z_avg = torch.mean(h, dim=1)
        if self.use_region_attention:
            z_attn, region_weight = self.region_attn(h)  # [B, D], [B, K]
            z = z_attn + z_avg if self.use_region_residual else z_attn
        else:
            region_weight = torch.full(
                (batch_size, num_rois),
                1.0 / num_rois,
                dtype=h.dtype,
                device=h.device,
            )
            z = z_avg

        logits = self.classifier(z)
        temporal_weight = temporal_weight.reshape(batch_size, num_rois, clip_len)
        return logits, region_weight, temporal_weight


class LightweightCNNLSTM(nn.Module):
    """Compact CNN-LSTM baseline for ROI RGB temporal sequences."""

    def __init__(self, num_rois=5, in_channels=3, hidden_dim=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.num_rois = num_rois
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(num_rois * in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, K, T, C]
        if x.dim() != 4:
            raise ValueError(f"LightweightCNNLSTM expects [B, K, T, C], got {tuple(x.shape)}")
        batch_size, num_rois, clip_len, channels = x.shape
        h = x.permute(0, 1, 3, 2).reshape(batch_size, num_rois * channels, clip_len)
        h = self.temporal_cnn(h).permute(0, 2, 1)
        h, _ = self.lstm(h)
        z = h.mean(dim=1)
        logits = self.classifier(z)
        region_weight = torch.full(
            (batch_size, num_rois), 1.0 / num_rois, dtype=x.dtype, device=x.device
        )
        temporal_weight = torch.full(
            (batch_size, num_rois, clip_len), 1.0 / clip_len, dtype=x.dtype, device=x.device
        )
        return logits, region_weight, temporal_weight


class Lightweight3DCNN(nn.Module):
    """Small 3D convolution baseline over time, region, and color axes."""

    def __init__(self, num_rois=5, in_channels=3, hidden_dim=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.num_rois = num_rois
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: [B, K, T, C] -> [B, 1, T, K, C]
        if x.dim() != 4:
            raise ValueError(f"Lightweight3DCNN expects [B, K, T, C], got {tuple(x.shape)}")
        batch_size, num_rois, clip_len, _ = x.shape
        h = x.permute(0, 2, 1, 3).unsqueeze(1)
        logits = self.classifier(self.backbone(h))
        region_weight = torch.full(
            (batch_size, num_rois), 1.0 / num_rois, dtype=x.dtype, device=x.device
        )
        temporal_weight = torch.full(
            (batch_size, num_rois, clip_len), 1.0 / clip_len, dtype=x.dtype, device=x.device
        )
        return logits, region_weight, temporal_weight


class RPPGStressHead(nn.Module):
    """rPPG-oriented temporal backbone with a classification head."""

    def __init__(self, num_rois=5, in_channels=3, hidden_dim=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.num_rois = num_rois
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Uses the face-level mean RGB trace as an rPPG-style input.
        if x.dim() != 4:
            raise ValueError(f"RPPGStressHead expects [B, K, T, C], got {tuple(x.shape)}")
        batch_size, num_rois, clip_len, _ = x.shape
        face_trace = x.mean(dim=1).permute(0, 2, 1)
        h = self.backbone(face_trace).mean(dim=-1)
        logits = self.classifier(h)
        region_weight = torch.full(
            (batch_size, num_rois), 1.0 / num_rois, dtype=x.dtype, device=x.device
        )
        temporal_weight = torch.full(
            (batch_size, num_rois, clip_len), 1.0 / clip_len, dtype=x.dtype, device=x.device
        )
        return logits, region_weight, temporal_weight
