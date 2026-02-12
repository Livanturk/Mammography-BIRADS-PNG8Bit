"""
Tam Model: Multi-View Hierarchical BI-RADS Classifier
=======================================================
4 seviyeyi birleştiren ana model sınıfı.

Model Akışı:
    ┌─────────────────────────────────────────────────────┐
    │  Girdi: 4 Görüntü (RCC, LCC, RMLO, LMLO)          │
    │                                                      │
    │  Seviye 1 — Backbone (Weight-Shared)                │
    │  ├── RCC  → f_rcc   (512-dim)                       │
    │  ├── LCC  → f_lcc   (512-dim)                       │
    │  ├── RMLO → f_rmlo  (512-dim)                       │
    │  └── LMLO → f_lmlo  (512-dim)                       │
    │                                                      │
    │  Seviye 2 — Lateral Cross-Attention                  │
    │  ├── Right: CrossAttn(RCC, RMLO) → f_right (512-d)  │
    │  └── Left:  CrossAttn(LCC, LMLO) → f_left  (512-d)  │
    │                                                      │
    │  Seviye 3 — Bilateral Fusion                         │
    │  ├── f_diff = f_left - f_right                       │
    │  ├── f_avg  = (f_left + f_right) / 2                 │
    │  └── SelfAttn([f_L, f_R, f_diff, f_avg]) → f_pat    │
    │                                                      │
    │  Seviye 4 — Multi-Head Classification                │
    │  ├── Binary:  f_pat → Benign/Malign                  │
    │  ├── Benign:  f_pat → BIRADS 1/2                     │
    │  ├── Malign:  f_pat → BIRADS 4/5                     │
    │  └── Full:    f_pat → BIRADS 1/2/4/5                 │
    │  └── Uncertainty: Temperature-scaled confidence      │
    └─────────────────────────────────────────────────────┘

Ablation Desteği:
    Config'deki ablation ayarlarına göre modüller devre dışı bırakılabilir.
    Baseline: Sadece backbone + mean pooling + full_head.
"""

import torch
import torch.nn as nn

from models.backbone import MultiViewBackbone
from models.lateral_fusion import BilateralLateralFusion
from models.bilateral_fusion import BilateralFusion
from models.classification_heads import HierarchicalClassifier


class MammographyClassifier(nn.Module):
    """
    Multi-view hierarchical mammografi sınıflandırıcı.

    Config dosyasından tüm ayarları okuyarak modeli oluşturur.
    Ablation çalışmaları için modüller seçici olarak etkinleştirilebilir.

    Args:
        config: Parsed YAML konfigürasyon sözlüğü.
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config["model"]
        ablation_cfg = config.get("ablation", {})

        self.projection_dim = model_cfg["projection_dim"]

        # Ablation bayrakları
        self.use_lateral = ablation_cfg.get("use_lateral_fusion", True)
        self.use_bilateral = ablation_cfg.get("use_bilateral_fusion", True)
        self.use_binary_head = ablation_cfg.get("use_binary_head", True)
        self.use_subgroup_head = ablation_cfg.get("use_subgroup_head", True)
        self.use_uncertainty = ablation_cfg.get("use_uncertainty", True)

        # ============================================================
        # Seviye 1: Backbone (her zaman aktif)
        # ============================================================
        backbone_cfg = model_cfg["backbone"]
        self.backbone = MultiViewBackbone(
            backbone_name=backbone_cfg["name"],
            pretrained=backbone_cfg["pretrained"],
            projection_dim=self.projection_dim,
            freeze_layers=backbone_cfg.get("freeze_layers", 0),
        )

        # ============================================================
        # Seviye 2: Lateral Fusion (opsiyonel)
        # ============================================================
        if self.use_lateral:
            lat_cfg = model_cfg["lateral_fusion"]
            self.lateral_fusion = BilateralLateralFusion(
                dim=self.projection_dim,
                num_heads=lat_cfg["num_heads"],
                dropout=lat_cfg["dropout"],
                num_layers=lat_cfg.get("num_layers", 2),
            )
        else:
            # Lateral fusion yoksa: CC ve MLO'yu basitçe topla
            self.simple_lateral_proj = nn.Sequential(
                nn.Linear(self.projection_dim * 2, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
                nn.GELU(),
            )

        # ============================================================
        # Seviye 3: Bilateral Fusion (opsiyonel)
        # ============================================================
        if self.use_bilateral:
            bil_cfg = model_cfg["bilateral_fusion"]
            self.bilateral_fusion = BilateralFusion(
                dim=self.projection_dim,
                num_heads=bil_cfg["num_heads"],
                dropout=bil_cfg["dropout"],
                use_diff=bil_cfg.get("use_diff", True),
                use_avg=bil_cfg.get("use_avg", True),
            )
        else:
            # Bilateral fusion yoksa: left ve right'ı basitçe topla
            self.simple_bilateral_proj = nn.Sequential(
                nn.Linear(self.projection_dim * 2, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
                nn.GELU(),
            )

        # ============================================================
        # Seviye 4: Classification Heads
        # ============================================================
        cls_cfg = model_cfg["classification"]
        self.classifier = HierarchicalClassifier(
            input_dim=self.projection_dim,
            hidden_dim=cls_cfg["hidden_dim"],
            dropout=cls_cfg["dropout"],
            temperature=cls_cfg.get("temperature", 1.5),
        )

        # Toplam parametre sayısını yazdır
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] Toplam parametre: {total_params:,}")
        print(f"[MODEL] Eğitilebilir parametre: {trainable_params:,}")
        print(f"[MODEL] Lateral fusion: {'AÇIK' if self.use_lateral else 'KAPALI'}")
        print(f"[MODEL] Bilateral fusion: {'AÇIK' if self.use_bilateral else 'KAPALI'}")

    def forward(self, images: torch.Tensor) -> dict:
        """
        Tam ileri geçiş (forward pass).

        Args:
            images: (B, 4, 3, 384, 384) — 4 mammografi görüntüsü.

        Returns:
            dict: Sınıflandırma sonuçları.
                - "binary_logits": (B, 2)
                - "benign_sub_logits": (B, 2)
                - "malign_sub_logits": (B, 2)
                - "full_logits": (B, 4)
                - "confidence": (B,)
                - "patient_features": (B, projection_dim) — Grad-CAM için
        """
        # --- Seviye 1: Her görüntüden öznitelik çıkar ---
        view_features = self.backbone(images)
        # view_features = {"RCC": (B, dim), "LCC": ..., "RMLO": ..., "LMLO": ...}

        # --- Seviye 2: Lateral Fusion ---
        if self.use_lateral:
            lateral_features = self.lateral_fusion(view_features)
            # {"right": (B, dim), "left": (B, dim)}
        else:
            # Basit birleştirme: concat + projection
            right_concat = torch.cat(
                [view_features["RCC"], view_features["RMLO"]], dim=-1
            )
            left_concat = torch.cat(
                [view_features["LCC"], view_features["LMLO"]], dim=-1
            )
            lateral_features = {
                "right": self.simple_lateral_proj(right_concat),
                "left": self.simple_lateral_proj(left_concat),
            }

        # --- Seviye 3: Bilateral Fusion ---
        if self.use_bilateral:
            patient_feat = self.bilateral_fusion(
                left_feat=lateral_features["left"],
                right_feat=lateral_features["right"],
            )
        else:
            # Basit birleştirme
            bilateral_concat = torch.cat(
                [lateral_features["left"], lateral_features["right"]], dim=-1
            )
            patient_feat = self.simple_bilateral_proj(bilateral_concat)

        # --- Seviye 4: Sınıflandırma ---
        outputs = self.classifier(patient_feat)
        outputs["patient_features"] = patient_feat

        return outputs

    def get_backbone_extractor(self):
        """Grad-CAM için backbone erişimi sağlar."""
        return self.backbone.backbone


def build_model(config: dict) -> MammographyClassifier:
    """
    Config'den model oluşturur.

    Args:
        config: YAML konfigürasyon sözlüğü.

    Returns:
        MammographyClassifier instance.
    """
    model = MammographyClassifier(config)
    return model


def build_baseline_config(config: dict) -> dict:
    """
    Baseline deney için ablation ayarlarını düzenler.

    Baseline = Sadece backbone + ortalama pooling + full_head.
    Lateral ve bilateral fusion kapatılır.

    Args:
        config: Orijinal konfigürasyon.

    Returns:
        Baseline konfigürasyonu (orijinal değişmez, kopya döner).
    """
    import copy
    baseline_cfg = copy.deepcopy(config)
    baseline_cfg["ablation"] = {
        "use_lateral_fusion": False,
        "use_bilateral_fusion": False,
        "use_binary_head": False,
        "use_subgroup_head": False,
        "use_uncertainty": False,
    }
    return baseline_cfg
