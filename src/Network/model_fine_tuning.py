import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


# ============================================================
# CLASSIFIER HEAD
# ============================================================
class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# MODELLO MULTITASK SOFT-CONDITIONAL
# ============================================================
class CNNMultiTask(nn.Module):
    """
    - Backbone FaceNet → embedding 512
    - Head stagione → 4 classi
    - Head sottotipo → 6 classi
      condizionata sulle PROBABILITÀ della stagione

    Nessun vincolo hard.
    Forward identico in train e test.
    """

    def __init__(
        self,
        num_seasons: int = 4,
        num_subtypes: int = 6,
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.num_seasons = num_seasons
        self.num_subtypes = num_subtypes

        # -----------------------------
        # Backbone FaceNet
        # -----------------------------
        self.backbone = InceptionResnetV1(
            pretrained="vggface2",
            classify=False
        )

        embedding_dim = 512

        # -----------------------------
        # Heads
        # -----------------------------
        self.head_season = ClassifierHead(
            embedding_dim,
            num_seasons
        )

        # sottotipo condizionato soft
        self.head_subtype = ClassifierHead(
            embedding_dim + num_seasons,
            num_subtypes
        )

        if freeze_backbone:
            self.freeze_backbone()

    # =====================================================
    # FREEZE BACKBONE
    # =====================================================
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    # =====================================================
    # FINE-TUNING LEGGERO
    # =====================================================
    def unfreeze_high_layers(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.backbone.repeat_3.parameters():
            p.requires_grad = True
        for p in self.backbone.block8.parameters():
            p.requires_grad = True
        for p in self.backbone.last_linear.parameters():
            p.requires_grad = True
        for p in self.backbone.last_bn.parameters():
            p.requires_grad = True

        print(">>> Fine-tuning: repeat_3, block8, last_linear, last_bn")

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, x):
        """
        Ritorna:
            logits_season  : [B, 4]
            logits_subtype : [B, 6]
        """

        # 1) Embedding facciale
        emb = self.backbone(x)          # [B, 512]

        # 2) Stagione
        logits_season = self.head_season(emb)
        prob_season = F.softmax(logits_season, dim=1)   # [B, 4]

        # 3) Sottotipo condizionato (soft)
        cond_input = torch.cat([emb, prob_season], dim=1)  # [B, 516]
        logits_sub = self.head_subtype(cond_input)

        return logits_season, logits_sub