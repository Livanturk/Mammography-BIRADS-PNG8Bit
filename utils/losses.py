"""
Multi-Head Loss Function
=========================
Hiyerarşik çoklu-kafa sınıflandırıcı için bileşik kayıp fonksiyonu.

Kayıp Bileşenleri:
    L_total = w1 * L_binary + w2 * L_subgroup + w3 * L_full

    L_binary:   CrossEntropy(binary_logits, binary_labels)
                Benign vs Malign ayrımı için.

    L_subgroup: CrossEntropy(benign_sub_logits, benign_labels) +
                CrossEntropy(malign_sub_logits, malign_labels)
                Alt grup ayrımları için. Sadece ilgili örneklere uygulanır.

    L_full:     CrossEntropy(full_logits, full_labels)
                4-sınıf direkt tahmin için.

Class Weights:
    Sınıf dengesizliğini telafi etmek için ağırlıklar kullanılır.
    Az temsil edilen sınıflar daha yüksek ağırlık alır.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classification_heads import HierarchicalClassifier


class MultiHeadLoss(nn.Module):
    """
    Hiyerarşik çoklu-kafa kayıp fonksiyonu.

    Args:
        loss_weights: Her head'in toplam kayba katkı ağırlığı.
            {"binary_head": 0.3, "subgroup_head": 0.3, "full_head": 0.4}
        class_weights_4: 4-sınıf için sınıf ağırlıkları (tensor).
        class_weights_binary: Binary için sınıf ağırlıkları (tensor).
        class_weights_benign_sub: Benign subgroup (BIRADS 1 vs 2) ağırlıkları.
        class_weights_malign_sub: Malign subgroup (BIRADS 4 vs 5) ağırlıkları.
        use_binary: Binary head kaybını hesapla.
        use_subgroup: Subgroup head kaybını hesapla.
        label_smoothing: Etiket yumuşatma (overfitting'i azaltır).
    """

    def __init__(
        self,
        loss_weights: dict,
        class_weights_4: Optional[torch.Tensor] = None,
        class_weights_binary: Optional[torch.Tensor] = None,
        class_weights_benign_sub: Optional[torch.Tensor] = None,
        class_weights_malign_sub: Optional[torch.Tensor] = None,
        use_binary: bool = True,
        use_subgroup: bool = True,
        label_smoothing: float = 0.05,
    ):
        super().__init__()

        self.w_binary = loss_weights.get("binary_head", 0.3)
        self.w_subgroup = loss_weights.get("subgroup_head", 0.3)
        self.w_full = loss_weights.get("full_head", 0.4)
        self.use_binary = use_binary
        self.use_subgroup = use_subgroup

        # 4-sınıf CrossEntropy
        self.full_criterion = nn.CrossEntropyLoss(
            weight=class_weights_4,
            label_smoothing=label_smoothing,
        )

        # Binary CrossEntropy
        self.binary_criterion = nn.CrossEntropyLoss(
            weight=class_weights_binary,
            label_smoothing=label_smoothing,
        )

        # Subgroup CrossEntropy — her alt grup için ayrı ağırlıklar
        self.benign_sub_criterion = nn.CrossEntropyLoss(
            weight=class_weights_benign_sub,
            label_smoothing=label_smoothing,
        )
        self.malign_sub_criterion = nn.CrossEntropyLoss(
            weight=class_weights_malign_sub,
            label_smoothing=label_smoothing,
        )

    def forward(
        self, outputs: dict, labels: torch.Tensor
    ) -> dict:
        """
        Toplam kaybı hesaplar.

        Args:
            outputs: Model çıkışları (binary_logits, subgroup logits, full_logits).
            labels: (B,) 4-sınıf etiketleri.

        Returns:
            dict:
                - "total_loss": Toplam ağırlıklı kayıp.
                - "binary_loss": Binary head kaybı.
                - "benign_sub_loss": Benign subgroup kaybı.
                - "malign_sub_loss": Malign subgroup kaybı.
                - "full_loss": Full head kaybı.
        """
        # Etiketleri her head için dönüştür
        label_dict = HierarchicalClassifier.convert_labels(labels)

        losses = {}
        total_loss = torch.tensor(0.0, device=labels.device)

        # --- Full Head Loss (her zaman aktif) ---
        full_loss = self.full_criterion(outputs["full_logits"], label_dict["full"])
        losses["full_loss"] = full_loss
        total_loss = total_loss + self.w_full * full_loss

        # --- Binary Head Loss ---
        if self.use_binary:
            binary_loss = self.binary_criterion(
                outputs["binary_logits"], label_dict["binary"]
            )
            losses["binary_loss"] = binary_loss
            total_loss = total_loss + self.w_binary * binary_loss

        # --- Subgroup Head Loss ---
        # Sadece ilgili örneklere uygulanır (benign → benign head, malign → malign head)
        if self.use_subgroup:
            benign_sub_loss = torch.tensor(0.0, device=labels.device)
            malign_sub_loss = torch.tensor(0.0, device=labels.device)

            # Benign alt grubu (BIRADS 1 ve 2 olan örnekler)
            benign_mask = label_dict["benign_mask"]
            if benign_mask.any():
                benign_logits = outputs["benign_sub_logits"][benign_mask]
                benign_labels = label_dict["benign_sub"]
                benign_sub_loss = self.benign_sub_criterion(benign_logits, benign_labels)

            # Malign alt grubu (BIRADS 4 ve 5 olan örnekler)
            malign_mask = label_dict["malign_mask"]
            if malign_mask.any():
                malign_logits = outputs["malign_sub_logits"][malign_mask]
                malign_labels = label_dict["malign_sub"]
                malign_sub_loss = self.malign_sub_criterion(malign_logits, malign_labels)

            subgroup_loss = (benign_sub_loss + malign_sub_loss) / 2.0
            losses["benign_sub_loss"] = benign_sub_loss
            losses["malign_sub_loss"] = malign_sub_loss
            total_loss = total_loss + self.w_subgroup * subgroup_loss

        losses["total_loss"] = total_loss
        return losses


def build_loss_function(config: dict, device: torch.device) -> MultiHeadLoss:
    """
    Config'den loss fonksiyonu oluşturur.

    Args:
        config: YAML konfigürasyonu.
        device: CUDA/CPU cihazı.

    Returns:
        MultiHeadLoss instance.
    """
    train_cfg = config["training"]
    ablation_cfg = config.get("ablation", {})

    # 4-sınıf ağırlıkları
    cw = train_cfg.get("class_weights", [1.0, 1.0, 1.0, 1.0])
    class_weights_4 = torch.tensor(cw, dtype=torch.float32).to(device)

    # Binary ağırlıklar (benign örnek sayısı vs malign)
    # BIRADS 1+2 = 3932, BIRADS 4+5 = 3625 → yaklaşık dengeli
    class_weights_binary = torch.tensor([1.0, 1.08], dtype=torch.float32).to(device)

    # Subgroup ağırlıkları (sqrt-inverse frequency)
    # Benign: BIRADS 1 (1428) vs BIRADS 2 (2504) → sqrt(2504/1428)=1.32, 1.0
    class_weights_benign_sub = torch.tensor([1.32, 1.0], dtype=torch.float32).to(device)
    # Malign: BIRADS 4 (1648) vs BIRADS 5 (1977) → sqrt(1977/1648)=1.10, 1.0
    class_weights_malign_sub = torch.tensor([1.10, 1.0], dtype=torch.float32).to(device)

    return MultiHeadLoss(
        loss_weights=train_cfg["loss_weights"],
        class_weights_4=class_weights_4,
        class_weights_binary=class_weights_binary,
        class_weights_benign_sub=class_weights_benign_sub,
        class_weights_malign_sub=class_weights_malign_sub,
        use_binary=ablation_cfg.get("use_binary_head", True),
        use_subgroup=ablation_cfg.get("use_subgroup_head", True),
    )
