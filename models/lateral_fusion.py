"""
Seviye 2: Lateral-Level Fusion (Spatial Cross-Attention)
=========================================================
Aynı taraftaki CC ve MLO görüntülerinin spatial öznitelik haritaları
arasında cross-attention uygular.

Neden Spatial Cross-Attention?
    CC (üstten) ve MLO (yandan) görüntüleri aynı memeyi farklı açılardan gösterir.
    Radyolog CC'de gördüğü bir kitlenin MLO'daki karşılığını arar.

    Spatial cross-attention bu süreci modeller:
    CC'deki HER spatial bölge (token), MLO'daki TÜM spatial bölgelere
    dikkat ederek en ilgili bölgeleri bulur. Bu sayede:
    - CC'de bir kitle varsa, MLO'daki karşılık gelen bölge vurgulanır
    - Multi-head yapı farklı ilişki türlerini paralel öğrenir
    - Birden fazla katman ile bilgi giderek daha fazla paylaşılır

Matematiksel Detay:
    Spatial tokenlar: CC = (B, S, dim), MLO = (B, S, dim)
    S = H × W (örn: 12×12 = 144 token, 384×384 girdi, stride=32)

    Cross-Attention (Pre-LN):
        Q = LayerNorm(CC),  K = V = LayerNorm(MLO)
        CC' = CC + MultiHeadAttn(Q, K, V)    (residual)
        CC'' = CC' + FFN(LayerNorm(CC'))       (residual)

    Bi-directional: Hem CC→MLO hem MLO→CC yönünde.

    Attention Pooling:
        S adet spatial token → tek lateral vektör (dim boyutlu)
        Her token için öğrenilen önem skoru hesaplanır.
"""

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    Pre-LN Spatial Cross-Attention bloğu.

    Source dizisindeki her token, target dizisindeki tüm token'lara
    dikkat eder. Pre-LN (norm-first) kullanılır — daha stabil eğitim sağlar.

    Akış (Pre-LN):
        h = source + MultiHeadAttn(LN(source), LN(target), LN(target))
        output = h + FFN(LN(h))

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

        # Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # Pre-LN: Normalizasyon attention/FFN ÖNCESİNDE uygulanır
        self.norm_q = nn.LayerNorm(dim)      # Query (source) normalizasyonu
        self.norm_kv = nn.LayerNorm(dim)     # Key/Value (target) normalizasyonu
        self.norm_ffn = nn.LayerNorm(dim)    # FFN öncesi normalizasyon

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Spatial cross-attention uygular.

        Args:
            source: (B, S_q, dim) — Sorgu kaynağı (spatial token dizisi).
            target: (B, S_kv, dim) — Anahtar/değer kaynağı (spatial token dizisi).
                    Genelde S_q == S_kv (aynı backbone, aynı spatial boyut).

        Returns:
            (B, S_q, dim) — Cross-attention ile zenginleştirilmiş source dizisi.
        """
        # Pre-LN Cross-Attention
        src_norm = self.norm_q(source)
        tgt_norm = self.norm_kv(target)

        attn_output, _ = self.cross_attn(
            query=src_norm,
            key=tgt_norm,
            value=tgt_norm,
        )
        source = source + attn_output       # Residual bağlantı

        # Pre-LN Feed-Forward
        ffn_output = self.ffn(self.norm_ffn(source))
        source = source + ffn_output        # Residual bağlantı

        return source


class LateralFusion(nn.Module):
    """
    Bir taraftaki CC ve MLO spatial özniteliklerini bi-directional
    cross-attention ile birleştirip tek bir lateral vektör üretir.

    Akış:
        1. Positional embedding ekle (spatial konum bilgisi)
        2. N katman bi-directional cross-attention:
           CC' = CrossAttn(CC → MLO)   — CC, MLO'ya dikkat eder
           MLO' = CrossAttn(MLO → CC)  — MLO, CC'ye dikkat eder
        3. Attention pooling: spatial token'ları tek vektöre indirge
        4. Fusion: concat([CC_pooled, MLO_pooled]) → projeksiyon

    Args:
        dim: Öznitelik boyutu.
        num_spatial_tokens: Spatial token sayısı (H × W).
        num_heads: Cross-attention başlık sayısı.
        attention_dropout: Attention dropout oranı.
        ffn_dropout: FFN dropout oranı.
        projection_dropout: Fusion projeksiyon dropout oranı.
        num_layers: Cross-attention katman sayısı.
    """

    def __init__(
        self,
        dim: int,
        num_spatial_tokens: int,
        num_heads: int = 8,
        attention_dropout: float = 0.15,
        ffn_dropout: float = 0.2,
        projection_dropout: float = 0.2,
        num_layers: int = 2,
    ):
        super().__init__()

        # Learnable positional embedding — spatial konum bilgisi
        # CC ve MLO aynı backbone'dan geçtiği için aynı spatial grid'e sahip,
        # dolayısıyla aynı positional embedding paylaşılır
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_spatial_tokens, dim) * 0.02
        )

        # Bi-directional cross-attention katmanları
        self.cc_to_mlo_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, attention_dropout, ffn_dropout)
            for _ in range(num_layers)
        ])
        self.mlo_to_cc_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, attention_dropout, ffn_dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention sonrası normalizasyon
        self.final_norm = nn.LayerNorm(dim)

        # Attention pooling: spatial token'ları tek vektöre indirger
        # Her token için öğrenilen önem skoru hesaplar
        self.attention_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

        # Son birleştirme projeksiyon katmanı
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )

    def _pool_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention pooling: spatial token dizisini tek vektöre indirger.

        Args:
            x: (B, S, dim) — Spatial token dizisi.

        Returns:
            (B, dim) — Ağırlıklı toplam ile elde edilen tek vektör.
        """
        scores = self.attention_pool(x)             # (B, S, 1)
        weights = torch.softmax(scores, dim=1)      # (B, S, 1) — normalize
        pooled = (weights * x).sum(dim=1)           # (B, dim)
        return pooled

    def forward(
        self, cc_feat: torch.Tensor, mlo_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        CC ve MLO spatial özniteliklerini cross-attention ile birleştirir.

        Args:
            cc_feat:  (B, S, dim) — CC görüntüsünün spatial öznitelik dizisi.
            mlo_feat: (B, S, dim) — MLO görüntüsünün spatial öznitelik dizisi.

        Returns:
            (B, dim) — Birleştirilmiş lateral öznitelik vektörü.
        """
        # Positional embedding ekle — spatial konum bilgisi
        cc_enhanced = cc_feat + self.pos_embed
        mlo_enhanced = mlo_feat + self.pos_embed

        # Her katmanda çift yönlü spatial cross-attention
        for cc2mlo, mlo2cc in zip(self.cc_to_mlo_layers, self.mlo_to_cc_layers):
            # CC'deki her token, MLO'daki tüm token'lara dikkat eder
            cc_new = cc2mlo(cc_enhanced, mlo_enhanced)
            # MLO'daki her token, CC'deki tüm token'lara dikkat eder
            mlo_new = mlo2cc(mlo_enhanced, cc_enhanced)

            cc_enhanced = cc_new
            mlo_enhanced = mlo_new

        # Final normalizasyon
        cc_enhanced = self.final_norm(cc_enhanced)
        mlo_enhanced = self.final_norm(mlo_enhanced)

        # Attention pooling: (B, S, dim) → (B, dim)
        cc_pooled = self._pool_spatial(cc_enhanced)
        mlo_pooled = self._pool_spatial(mlo_enhanced)

        # İki pooled vektörü birleştir ve projeksiyon uygula
        fused = torch.cat([cc_pooled, mlo_pooled], dim=-1)   # (B, dim*2)
        lateral = self.fusion_projection(fused)               # (B, dim)

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
        num_spatial_tokens: Spatial token sayısı (H × W).
        num_heads: Cross-attention başlık sayısı.
        attention_dropout: Attention dropout oranı.
        ffn_dropout: FFN dropout oranı.
        projection_dropout: Fusion projeksiyon dropout oranı.
        num_layers: Cross-attention katman sayısı.
    """

    def __init__(
        self,
        dim: int,
        num_spatial_tokens: int,
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
            num_spatial_tokens=num_spatial_tokens,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            projection_dropout=projection_dropout,
            num_layers=num_layers,
        )

    def forward(self, view_features: dict) -> dict:
        """
        Args:
            view_features: Backbone çıkışları (spatial).
                {
                    "RCC":  (B, S, dim),
                    "LCC":  (B, S, dim),
                    "RMLO": (B, S, dim),
                    "LMLO": (B, S, dim),
                }

        Returns:
            dict:
                {
                    "right": (B, dim) — Sağ meme lateral özniteliği.
                    "left":  (B, dim) — Sol meme lateral özniteliği.
                }
        """
        # Sağ meme: RCC + RMLO → spatial cross-attention → pooled vektör
        right_lateral = self.lateral_fusion(
            cc_feat=view_features["RCC"],
            mlo_feat=view_features["RMLO"],
        )

        # Sol meme: LCC + LMLO → spatial cross-attention → pooled vektör
        left_lateral = self.lateral_fusion(
            cc_feat=view_features["LCC"],
            mlo_feat=view_features["LMLO"],
        )

        return {"right": right_lateral, "left": left_lateral}
