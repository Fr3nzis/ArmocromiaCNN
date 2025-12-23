import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Network.blocks import ConvBlock


class CNNMultiTask(nn.Module):
    """
    CNN multitask per:
    - stagione (4 classi)
    - sottotipo (6 classi), condizionato soft sulla stagione

    Output:
        - logits_stagione:  (B, num_stagioni)
        - logits_sottotipo: (B, num_sottotipi)
    """

    def __init__(self, num_stagioni: int = 4, num_sottotipi: int = 6):
        super().__init__()

        self.num_stagioni = num_stagioni
        self.num_sottotipi = num_sottotipi

        # -------------------------------------------------
        # FEATURE EXTRACTOR
        # -------------------------------------------------
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )

        # Global Average Pooling per ridurre (C,H,W) → (C,1,1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Calcolo dimensione vettore flatten dopo GAP
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out = self.features(dummy)
            out = self.gap(out)               # (1, C, 1, 1)
            flatten_dim = out.shape[1]        # C
        self.flatten_dim = flatten_dim       # dovrebbe essere 64

        # -------------------------------------------------
        # TESTA STAGIONE
        # -------------------------------------------------
        self.fc_stagione = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_stagioni)
        )

        # -------------------------------------------------
        # TESTA SOTTOTIPO
        # -------------------------------------------------
        # Input = feature flatten + prob_stagione (soft) → dim = flatten_dim + num_stagioni
        self.fc_sottotipo = nn.Sequential(
            nn.Linear(self.flatten_dim + num_stagioni, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_sottotipi)
        )

    # -------------------------------------------------
    # FORWARD
    # -------------------------------------------------
    def forward(self, x):
        """
        x: immagini (B, 3, 128, 128)

        Ritorna:
            logits_stagione, logits_sottotipo
        """

        # 1) estrazione feature
        f = self.features(x)          # (B, C, H, W)
        f = self.gap(f)               # (B, C, 1, 1)
        f = torch.flatten(f, 1)       # (B, C) → vettore per immagine

        # 2) testa stagione
        logits_stagione = self.fc_stagione(f)          # (B, num_stagioni)
        prob_stagione = F.softmax(logits_stagione, 1)  # (B, num_stagioni)

        # 3) testa sottotipo condizionata sulla stagione (soft)
        cond_input = torch.cat([f, prob_stagione], dim=1)   # (B, C + num_stagioni)
        logits_sottotipo = self.fc_sottotipo(cond_input)    # (B, num_sottotipi)

        return logits_stagione, logits_sottotipo
