"""
Seviye 1: Image-Level Feature Extraction (Backbone)
=====================================================
Weight-shared backbone ağı ile her mammografi görüntüsünden
öznitelik vektörü (feature vector) çıkarır.

Weight Sharing Nedir?
    4 farklı görüntü (RCC, LCC, RMLO, LMLO) aynı backbone ağından geçirilir.
    Böylece tüm görüntüler aynı öznitelik uzayında (feature space) temsil edilir
    ve parametre sayısı 4 kat azaltılmış olur.

Desteklenen Backbone'lar:
    - ResNet50: Klasik, güvenilir. Feature dim = 2048.
    - EfficientNet-B0/B3/B5: Hafif ve etkili. Feature dim = 1280/1536/2048.
    - ConvNeXt-Tiny/Small: Modern CNN. Feature dim = 768/768.
"""

from typing import Optional

import torch
import torch.nn as nn
import timm


class BackboneFeatureExtractor(nn.Module):
    """
    Pretrained backbone + projeksiyon katmanı.

    Akış:
        Görüntü (3, 384, 384) → Backbone → Global Avg Pool → Projection → Feature (projection_dim,)

    Args:
        backbone_name: timm kütüphanesinden model adı.
        pretrained: ImageNet ağırlıklarını kullan.
        projection_dim: Çıkış öznitelik boyutu.
        freeze_layers: İlk N katmanı dondur (fine-tuning stratejisi).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        projection_dim: int = 512,
        freeze_layers: int = 0,
    ):
        super().__init__()

        # timm ile backbone oluştur (son sınıflandırma katmanı olmadan)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,      # Sınıflandırma kafasını kaldır
            global_pool="avg",  # Global Average Pooling
        )

        # Backbone'un çıkış boyutunu öğren
        # timm modelleri num_features attribute'unu sağlar
        backbone_dim = self.backbone.num_features

        # Projeksiyon katmanı: backbone_dim → projection_dim
        # Bu katman, farklı backbone'ların çıkışlarını ortak bir boyuta indirger
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # İsteğe bağlı: İlk katmanları dondur
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        self.backbone_dim = backbone_dim
        self.projection_dim = projection_dim

    def _freeze_layers(self, n: int):
        """
        Backbone'un ilk N katmanını dondurur (gradyan hesaplanmaz).
        Bu, düşük seviyeli özelliklerin (kenarlar, dokular) korunmasını sağlar
        ve aşırı öğrenmeyi (overfitting) önlemeye yardımcı olur.
        """
        params = list(self.backbone.parameters())
        for param in params[:n]:
            param.requires_grad = False
        print(f"[BİLGİ] Backbone'un ilk {n} parametresi donduruldu.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) boyutunda görüntü tensörü.

        Returns:
            (B, projection_dim) boyutunda öznitelik vektörü.
        """
        features = self.backbone(x)     # (B, backbone_dim)
        projected = self.projection(features)  # (B, projection_dim)
        return projected

    def get_last_conv_layer(self) -> nn.Module:
        """
        Grad-CAM için backbone'un son konvolüsyon katmanını döndürür.
        timm modellerinin yapısına göre doğru katmanı bulur.
        """
        # Yaygın timm mimarileri için son konvolüsyon katmanı
        if hasattr(self.backbone, "conv_head"):
            # EfficientNet ailesi
            return self.backbone.conv_head
        elif hasattr(self.backbone, "layer4"):
            # ResNet ailesi
            return self.backbone.layer4[-1]
        elif hasattr(self.backbone, "stages"):
            # ConvNeXt ailesi
            return self.backbone.stages[-1]
        else:
            # Genel yaklaşım: Son çocuk modülü bul
            children = list(self.backbone.children())
            for child in reversed(children):
                if isinstance(child, (nn.Conv2d, nn.Sequential)):
                    return child
            raise ValueError(
                f"Grad-CAM katmanı otomatik bulunamadı: {type(self.backbone).__name__}. "
                f"Lütfen config.yaml'da target_layer'ı manuel belirtin."
            )


class MultiViewBackbone(nn.Module):
    """
    4 mammografi görüntüsünü tek bir weight-shared backbone'dan geçirir.

    Weight Sharing:
        Aynı BackboneFeatureExtractor nesnesi 4 kez kullanılır.
        Bu, parametrelerin paylaşılması anlamına gelir:
        - 4 ayrı backbone → ~4x parametre (kötü, overfitting riski)
        - 1 shared backbone → 1x parametre (verimli, genelleme iyi)

    Args:
        backbone_name: timm model adı.
        pretrained: ImageNet pretrained ağırlıklar.
        projection_dim: Çıkış boyutu.
        freeze_layers: Dondurulacak katman sayısı.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        projection_dim: int = 512,
        freeze_layers: int = 0,
    ):
        super().__init__()

        # TEK backbone — tüm görüntüler bundan geçer (weight sharing)
        self.backbone = BackboneFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            projection_dim=projection_dim,
            freeze_layers=freeze_layers,
        )

    def forward(
        self, images: torch.Tensor
    ) -> dict:
        """
        4 mammografi görüntüsünü işler.

        Args:
            images: (B, 4, C, H, W) boyutunda tensor.
                    Kanal sırası: [RCC, LCC, RMLO, LMLO]

        Returns:
            dict: Her görüntünün öznitelik vektörü.
                {
                    "RCC":  (B, projection_dim),
                    "LCC":  (B, projection_dim),
                    "RMLO": (B, projection_dim),
                    "LMLO": (B, projection_dim),
                }
        """
        B, num_views, C, H, W = images.shape
        assert num_views == 4, f"4 görüntü bekleniyor, {num_views} alındı."

        features = {}
        view_names = ["RCC", "LCC", "RMLO", "LMLO"]

        for i, name in enumerate(view_names):
            # Her görüntüyü ayrı ayrı backbone'dan geçir
            view_img = images[:, i]          # (B, C, H, W)
            features[name] = self.backbone(view_img)  # (B, projection_dim)

        return features
