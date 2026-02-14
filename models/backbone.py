"""
Seviye 1: Image-Level Feature Extraction (Backbone)
=====================================================
Weight-shared backbone ağı ile her mammografi görüntüsünden
spatial öznitelik haritası (feature map) çıkarır.

Weight Sharing Nedir?
    4 farklı görüntü (RCC, LCC, RMLO, LMLO) aynı backbone ağından geçirilir.
    Böylece tüm görüntüler aynı öznitelik uzayında (feature space) temsil edilir
    ve parametre sayısı 4 kat azaltılmış olur.

Spatial Feature Map:
    Global Average Pooling YAPILMAZ. Backbone çıkışı (B, C, H, W) olarak korunur
    ve (B, H*W, projection_dim) şeklinde spatial token dizisine dönüştürülür.
    Bu sayede Lateral Fusion'da gerçek cross-attention uygulanabilir:
    CC'deki her spatial bölge, MLO'daki ilgili bölgelere dikkat edebilir.

Desteklenen Backbone'lar:
    - ResNet50: Klasik, güvenilir. Feature dim = 2048.
    - EfficientNet-B0/B3/B5: Hafif ve etkili. Feature dim = 1280/1536/2048.
    - ConvNeXt-Tiny/Small/Large: Modern CNN. Feature dim = 768/768/1536.
    - MaxViT: Hibrit ViT+CNN. Feature dim = 768.
"""

from typing import Optional

import torch
import torch.nn as nn
import timm


class BackboneFeatureExtractor(nn.Module):
    """
    Pretrained backbone + spatial projeksiyon katmanı.

    Akış:
        Görüntü (3, 384, 384)
            → Backbone (global pool yok)
            → Spatial Feature Map (B, C, H, W)
            → Reshape (B, H*W, C)
            → Projection (B, H*W, projection_dim)

    Args:
        backbone_name: timm kütüphanesinden model adı.
        pretrained: ImageNet ağırlıklarını kullan.
        projection_dim: Çıkış öznitelik boyutu.
        freeze_layers: İlk N katmanı dondur (fine-tuning stratejisi).
        projection_dropout: Projeksiyon katmanı dropout oranı.
        image_size: Girdi görüntü boyutu (spatial token sayısını hesaplamak için).
    """

    def __init__(
        self,
        backbone_name: str = "convnext_large",
        pretrained: bool = True,
        projection_dim: int = 512,
        freeze_layers: int = 0,
        projection_dropout: float = 0.2,
        image_size: int = 384,
    ):
        super().__init__()

        # timm ile backbone oluştur — global pooling YOK, spatial feature map korunur
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",     # Spatial feature map koru (global avg pool yapma)
        )

        # Backbone'un çıkış boyutunu öğren
        backbone_dim = self.backbone.num_features

        # Spatial token sayısını hesapla (dummy forward ile)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            dummy_out = self.backbone(dummy)
            if dummy_out.dim() == 4:
                if dummy_out.shape[1] == backbone_dim:
                    # Channels-first: (1, C, H, W) — CNN (ConvNeXt, ResNet, EfficientNet)
                    self.num_spatial_tokens = dummy_out.shape[2] * dummy_out.shape[3]
                else:
                    # Channels-last: (1, H, W, C) — Swin Transformer vb.
                    self.num_spatial_tokens = dummy_out.shape[1] * dummy_out.shape[2]
            else:
                # ViT çıkışı: (1, N, C)
                self.num_spatial_tokens = dummy_out.shape[1]

        # Projeksiyon katmanı: backbone_dim → projection_dim (her spatial token için)
        # nn.Linear son boyut üzerinde çalışır, (B, S, backbone_dim) → (B, S, projection_dim)
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )

        # İsteğe bağlı: İlk katmanları dondur
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        self.backbone_dim = backbone_dim
        self.projection_dim = projection_dim

        print(f"[BACKBONE] {backbone_name}: {backbone_dim}-dim → {projection_dim}-dim projeksiyon")
        print(f"[BACKBONE] Spatial token sayısı: {self.num_spatial_tokens}")

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
        Spatial öznitelik haritası çıkarır.

        Args:
            x: (B, 3, H, W) boyutunda görüntü tensörü.

        Returns:
            (B, num_spatial_tokens, projection_dim) boyutunda spatial öznitelik dizisi.
        """
        features = self.backbone(x)

        if features.dim() == 4:
            if features.shape[1] == self.backbone_dim:
                # Channels-first: (B, C, H, W) — CNN (ConvNeXt, ResNet, EfficientNet)
                B, C, H, W = features.shape
                features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            else:
                # Channels-last: (B, H, W, C) — Swin Transformer vb.
                B, H, W, C = features.shape
                features = features.reshape(B, H * W, C)
        # 3D çıkış (B, N, C) zaten doğru formatta — ViT modelleri

        # Her spatial token için projeksiyon
        projected = self.projection(features)   # (B, S, projection_dim)
        return projected

    def get_last_conv_layer(self) -> nn.Module:
        """
        Grad-CAM için backbone'un son konvolüsyon katmanını döndürür.
        timm modellerinin yapısına göre doğru katmanı bulur.
        """
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
    Her görüntü için spatial öznitelik dizisi döndürür.

    Weight Sharing:
        Aynı BackboneFeatureExtractor nesnesi 4 kez kullanılır.
        - 4 ayrı backbone → ~4x parametre (kötü, overfitting riski)
        - 1 shared backbone → 1x parametre (verimli, genelleme iyi)

    Args:
        backbone_name: timm model adı.
        pretrained: ImageNet pretrained ağırlıklar.
        projection_dim: Çıkış boyutu.
        freeze_layers: Dondurulacak katman sayısı.
        projection_dropout: Projeksiyon dropout oranı.
        image_size: Girdi görüntü boyutu.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_large",
        pretrained: bool = True,
        projection_dim: int = 512,
        freeze_layers: int = 0,
        projection_dropout: float = 0.2,
        image_size: int = 384,
    ):
        super().__init__()

        # TEK backbone — tüm görüntüler bundan geçer (weight sharing)
        self.backbone = BackboneFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            projection_dim=projection_dim,
            freeze_layers=freeze_layers,
            projection_dropout=projection_dropout,
            image_size=image_size,
        )

        self.num_spatial_tokens = self.backbone.num_spatial_tokens

    def forward(
        self, images: torch.Tensor
    ) -> dict:
        """
        4 mammografi görüntüsünü işler, spatial öznitelik dizileri döndürür.

        Args:
            images: (B, 4, C, H, W) boyutunda tensor.
                    Kanal sırası: [RCC, LCC, RMLO, LMLO]

        Returns:
            dict: Her görüntünün spatial öznitelik dizisi.
                {
                    "RCC":  (B, num_spatial_tokens, projection_dim),
                    "LCC":  (B, num_spatial_tokens, projection_dim),
                    "RMLO": (B, num_spatial_tokens, projection_dim),
                    "LMLO": (B, num_spatial_tokens, projection_dim),
                }
        """
        B, num_views, C, H, W = images.shape
        assert num_views == 4, f"4 görüntü bekleniyor, {num_views} alındı."

        features = {}
        view_names = ["RCC", "LCC", "RMLO", "LMLO"]

        for i, name in enumerate(view_names):
            view_img = images[:, i]                     # (B, C, H, W)
            features[name] = self.backbone(view_img)    # (B, S, projection_dim)

        return features
