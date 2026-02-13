"""
Seviye 3: Patient-Level Bilateral Fusion
==========================================
Sol ve sağ meme özniteliklerini birleştirerek hasta düzeyinde temsil oluşturur.
Asimetri sinyallerini yakalamak için özel operasyonlar içerir.

Klinik Motivasyon:
    Radyologlar mamografi okurken iki memeyi karşılaştırır.
    İki meme arasındaki FARKLILIKLAR (asimetri) genellikle
    malignite (kötü huylu tümör) belirtisidir.
    Bu modül bu karşılaştırmayı matematiksel olarak modeller.

Asimetri Vektörleri:
    F_diff = F_left - F_right
        → İki meme arasındaki farkı yakalar.
        → Eğer bir tarafta kitle varsa, fark vektörü bunu vurgular.

    F_avg = (F_left + F_right) / 2
        → Ortak (paylaşılan) özellikleri temsil eder.
        → Meme dokusu yoğunluğu gibi genel bilgileri içerir.

Birleştirme Stratejisi:
    [F_left, F_right, F_diff, F_avg] → Self-Attention → Patient Feature

    Self-Attention burada 4 vektörü birbirleriyle ilişkilendirir.
    Örneğin, fark vektörü sol memenin özniteliğini vurgulayabilir.
"""

import torch
import torch.nn as nn


class BilateralFusion(nn.Module):
    """
    Sol ve sağ meme özniteliklerini asimetri sinyalleriyle birleştirir.

    Akış:
        1. F_diff = F_left - F_right  (asimetri farkı)
        2. F_avg = (F_left + F_right) / 2  (ortak özellikler)
        3. tokens = [F_left, F_right, F_diff, F_avg]  (4 vektör)
        4. Self-Attention(tokens)  → ilişkili temsil
        5. Projection → Patient-level feature

    Args:
        dim: Giriş öznitelik boyutu.
        num_heads: Self-attention başlık sayısı.
        dropout: Dropout oranı.
        use_diff: F_diff kullanılsın mı.
        use_avg: F_avg kullanılsın mı.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.2,
        output_dropout: float = 0.25,
        use_diff: bool = True,
        use_avg: bool = True,
    ):
        super().__init__()

        self.use_diff = use_diff
        self.use_avg = use_avg

        # Kaç token var? Minimum 2 (left, right) + opsiyonel diff ve avg
        self.num_tokens = 2 + int(use_diff) + int(use_avg)

        # Self-Attention: Tüm vektörlerin birbirleriyle etkileşimini sağlar
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,       # (B, seq_len, dim) formatı
            norm_first=True,        # Pre-LN (daha stabil eğitim)
        )

        # Birden fazla transformer katmanı için encoder kullan
        self.transformer_encoder = nn.TransformerEncoder(
            self.self_attention,
            num_layers=2,
        )

        # Attention pooling: 4 token'ı tek bir vektöre indirger
        # Learned attention weights ile hangi token'ların daha önemli
        # olduğu öğrenilir
        self.attention_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),     # Her token için skaler skor
        )

        # Son projeksiyon — sınıflandırma öncesi son darboğaz, daha yüksek
        # dropout ile modelin sadece en discriminative feature'ları taşımasını zorlar
        self.output_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(output_dropout),
        )

    def forward(
        self, left_feat: torch.Tensor, right_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Sol ve sağ meme özniteliklerini birleştirir.

        Args:
            left_feat:  (B, dim) — Sol meme lateral özniteliği.
            right_feat: (B, dim) — Sağ meme lateral özniteliği.

        Returns:
            (B, dim) — Hasta düzeyinde birleşik öznitelik vektörü.
        """
        B, dim = left_feat.shape

        # --- Asimetri Vektörleri ---
        # F_diff: İki meme arasındaki fark
        # Eğer bir tarafta anormallik varsa, bu vektör bunu vurgular
        tokens = [left_feat, right_feat]

        if self.use_diff:
            f_diff = left_feat - right_feat         # (B, dim)
            tokens.append(f_diff)

        if self.use_avg:
            f_avg = (left_feat + right_feat) / 2.0  # (B, dim)
            tokens.append(f_avg)

        # Token'ları sequence olarak yığ: (B, num_tokens, dim)
        token_sequence = torch.stack(tokens, dim=1)

        # --- Self-Attention ---
        # 4 vektör birbirleriyle etkileşir
        # Örn: diff vektörü, left vektörünü vurgulayabilir
        attended = self.transformer_encoder(token_sequence)  # (B, num_tokens, dim)

        # --- Attention Pooling ---
        # Her token için önem skoru hesapla
        attn_scores = self.attention_pool(attended)      # (B, num_tokens, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) # (B, num_tokens, 1)

        # Ağırlıklı toplam: önemli token'lar daha çok katkıda bulunur
        # (B, num_tokens, 1) * (B, num_tokens, dim) → sum → (B, dim)
        pooled = (attn_weights * attended).sum(dim=1)    # (B, dim)

        # Son projeksiyon
        output = self.output_projection(pooled)          # (B, dim)

        return output
