"""
Seviye 2: Lateral-Level Fusion (Bi-directional Cross-Attention)
================================================================
Aynı taraftaki CC ve MLO görüntülerini cross-attention ile birleştirir.

Neden Cross-Attention?
    CC (üstten) ve MLO (yandan) görüntüleri aynı memeyi farklı açılardan gösterir.
    Cross-attention, bir görüntüdeki bilginin diğer görüntüdeki ilgili bölgelere
    "dikkat etmesini" sağlar. Örneğin, CC'de görülen bir kitle MLO'daki
    karşılık gelen bölge ile eşleştirilir.

    Bi-directional: Hem CC→MLO hem MLO→CC yönünde dikkat hesaplanır.

Matematiksel Detay:
    Cross-Attention(Q, K, V):
        Q = CC feature, K = V = MLO feature (veya tam tersi)
        Attention(Q, K, V) = softmax(QK^T / √d_k) × V

    Birleştirme:
        CC_enhanced = CrossAttn(CC→MLO) + CC    (residual bağlantı)
        MLO_enhanced = CrossAttn(MLO→CC) + MLO

        Lateral_Feature = LayerNorm(CC_enhanced + MLO_enhanced)
"""

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    Tek yönlü Cross-Attention bloğu.

    Bir kaynak (source) özniteliğin, bir hedef (target) özniteliğe
    dikkat etmesini sağlar.

    Akış:
        Query = Linear(source)
        Key   = Linear(target)
        Value = Linear(target)
        Output = MultiHeadAttention(Q, K, V) + source  (residual)

    Args:
        dim: Öznitelik boyutu.
        num_heads: Dikkat başlığı sayısı (dim'in tam böleni olmalı).
        attention_dropout: Attention weight dropout oranı.
        ffn_dropout: Feed-forward network dropout oranı.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.15,
        ffn_dropout: float = 0.2,
    ):
        super().__init__()

        assert dim % num_heads == 0, (
            f"dim ({dim}), num_heads ({num_heads}) ile tam bölünmeli."
        )

        # Multi-Head Attention: PyTorch'un yerleşik implementasyonu
        # batch_first=True: Giriş boyutu (B, seq_len, dim) olur
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # Layer normalization: Eğitimi stabilize eder
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward network (attention sonrası)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),    # Genişletme (4x standart)
            nn.GELU(),                   # Aktivasyon
            nn.Dropout(ffn_dropout),
            nn.Linear(dim * 4, dim),    # Geri daraltma
            nn.Dropout(ffn_dropout),
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention uygular: source, target'a dikkat eder.

        Args:
            source: (B, dim) — Sorgu (query) kaynağı.
            target: (B, dim) — Anahtar ve değer kaynağı.

        Returns:
            (B, dim) — Zenginleştirilmiş source özniteliği.
        """
        # Vektörleri sequence formatına çevir: (B, dim) → (B, 1, dim)
        # MultiheadAttention sequence boyutu bekler
        src = source.unsqueeze(1)
        tgt = target.unsqueeze(1)

        # Cross-attention: source (Q), target (K, V)
        # attn_output: (B, 1, dim)
        attn_output, _ = self.cross_attn(
            query=src,
            key=tgt,
            value=tgt,
        )

        # Residual bağlantı + Layer Norm
        src = self.norm1(src + attn_output)

        # Feed-forward + Residual + Layer Norm
        ffn_output = self.ffn(src)
        src = self.norm2(src + ffn_output)

        # Sequence boyutunu kaldır: (B, 1, dim) → (B, dim)
        return src.squeeze(1)


class LateralFusion(nn.Module):
    """
    Bir taraftaki CC ve MLO görüntülerini bi-directional cross-attention
    ile birleştirip tek bir lateral öznitelik vektörü üretir.

    Akış:
        1. CC → CrossAttn → CC' (MLO bilgisi ile zenginleştirilmiş)
        2. MLO → CrossAttn → MLO' (CC bilgisi ile zenginleştirilmiş)
        3. Lateral = Projection(CC' + MLO')

    Bu işlem N kez tekrarlanır (num_layers), böylece bilgi
    iki görüntü arasında giderek daha fazla paylaşılır.

    Args:
        dim: Öznitelik boyutu.
        num_heads: Cross-attention başlık sayısı.
        dropout: Dropout oranı.
        num_layers: Cross-attention katman sayısı (daha fazla = daha derin etkileşim).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.15,
        ffn_dropout: float = 0.2,
        projection_dropout: float = 0.2,
        num_layers: int = 2,
    ):
        super().__init__()

        # Birden fazla cross-attention katmanı: bilgi derinleşir
        self.cc_to_mlo_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, attention_dropout, ffn_dropout)
            for _ in range(num_layers)
        ])
        self.mlo_to_cc_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, attention_dropout, ffn_dropout)
            for _ in range(num_layers)
        ])

        # Son birleştirme projeksiyon katmanı
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )

    def forward(
        self, cc_feat: torch.Tensor, mlo_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        CC ve MLO özniteliklerini birleştirir.

        Args:
            cc_feat:  (B, dim) — CC görüntüsünün öznitelik vektörü.
            mlo_feat: (B, dim) — MLO görüntüsünün öznitelik vektörü.

        Returns:
            (B, dim) — Birleştirilmiş lateral öznitelik.
        """
        cc_enhanced = cc_feat
        mlo_enhanced = mlo_feat

        # Her katmanda çift yönlü bilgi alışverişi
        for cc2mlo, mlo2cc in zip(self.cc_to_mlo_layers, self.mlo_to_cc_layers):
            # CC, MLO'ya dikkat eder → CC zenginleşir
            cc_new = cc2mlo(cc_enhanced, mlo_enhanced)
            # MLO, CC'ye dikkat eder → MLO zenginleşir
            mlo_new = mlo2cc(mlo_enhanced, cc_enhanced)

            cc_enhanced = cc_new
            mlo_enhanced = mlo_new

        # İki zenginleştirilmiş vektörü birleştir
        fused = torch.cat([cc_enhanced, mlo_enhanced], dim=-1)  # (B, dim*2)
        lateral = self.fusion_projection(fused)                  # (B, dim)

        return lateral


class BilateralLateralFusion(nn.Module):
    """
    Sağ ve sol meme için ayrı ayrı lateral fusion uygular.

    Sağ meme: RCC + RMLO → Right Lateral Feature
    Sol meme: LCC + LMLO → Left Lateral Feature

    Not: İki taraf için AYNI ağırlıklar paylaşılır (weight sharing).
    Çünkü sağ ve sol meme anatomik olarak simetrik yapılardır,
    dolayısıyla aynı öznitelik çıkarma stratejisi uygulanabilir.

    Args:
        dim: Öznitelik boyutu.
        num_heads: Cross-attention başlık sayısı.
        dropout: Dropout oranı.
        num_layers: Cross-attention katman sayısı.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.15,
        ffn_dropout: float = 0.2,
        projection_dropout: float = 0.2,
        num_layers: int = 2,
    ):
        super().__init__()

        # Weight-shared lateral fusion (sağ ve sol aynı ağırlıkları paylaşır)
        self.lateral_fusion = LateralFusion(
            dim=dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            projection_dropout=projection_dropout,
            num_layers=num_layers,
        )

    def forward(self, view_features: dict) -> dict:
        """
        Args:
            view_features: Backbone çıkışları.
                {
                    "RCC":  (B, dim),
                    "LCC":  (B, dim),
                    "RMLO": (B, dim),
                    "LMLO": (B, dim),
                }

        Returns:
            dict:
                {
                    "right": (B, dim) — Sağ meme lateral özniteliği.
                    "left":  (B, dim) — Sol meme lateral özniteliği.
                }
        """
        # Sağ meme: RCC + RMLO
        right_lateral = self.lateral_fusion(
            cc_feat=view_features["RCC"],
            mlo_feat=view_features["RMLO"],
        )

        # Sol meme: LCC + LMLO
        left_lateral = self.lateral_fusion(
            cc_feat=view_features["LCC"],
            mlo_feat=view_features["LMLO"],
        )

        return {"right": right_lateral, "left": left_lateral}
