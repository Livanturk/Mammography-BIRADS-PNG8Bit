"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
======================================================
Modelin hangi bölgelere bakarak karar verdiğini görselleştirir.

Grad-CAM Nasıl Çalışır?
    1. Son konvolüsyon katmanının çıkışını (activation map) yakalar.
    2. Hedef sınıf için gradyanları hesaplar (backpropagation).
    3. Gradyanları ortalayarak her kanalın önemini bulur (channel weights).
    4. Activation map'i bu ağırlıklarla çarparak ısı haritası oluşturur.
    5. ReLU uygular (sadece pozitif katkılar).
    6. Orijinal görüntü boyutuna büyütür (upsample).

Klinik Önemi:
    Radyologlar modelin dikkat ettiği bölgeleri görerek:
    - Modelin doğru lezyona bakıp bakmadığını doğrulayabilir.
    - Yanlış tahminlerin nedenini anlayabilir.
    - Modele güven duyup duymayacağına karar verebilir.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # GUI olmadan kaydetmek için


class GradCAM:
    """
    Grad-CAM görselleştirme sınıfı.

    Kullanım:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(image, target_class)
        gradcam.visualize(image, heatmap, save_path)

    Args:
        model: PyTorch modeli.
        target_layer: Grad-CAM'in uygulanacağı konvolüsyon katmanı.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        # Hook ile activation ve gradient'ları yakala
        self.activations = None
        self.gradients = None

        # Forward hook: Katmanın çıkışını kaydet
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )
        # Backward hook: Katmana gelen gradyanı kaydet
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        """Forward geçişte activation map'i kaydet."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward geçişte gradyanları kaydet."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Tek bir görüntü için Grad-CAM ısı haritası üretir.

        Adımlar:
            1. Forward pass → activation map ve logit'leri al.
            2. Hedef sınıf için backward pass → gradyanları al.
            3. Global Average Pooling ile kanal ağırlıklarını hesapla.
            4. Ağırlıklı toplam → ısı haritası.
            5. ReLU → sadece pozitif katkılar.
            6. Normalize [0, 1] aralığına.

        Args:
            input_tensor: (1, 3, H, W) tek görüntü tensörü.
            target_class: Hangi sınıf için Grad-CAM üretilecek.
                          None ise en yüksek olasılıklı sınıf kullanılır.

        Returns:
            (H, W) boyutunda numpy array — ısı haritası [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Full head logit'leri kullan
        if isinstance(output, dict):
            logits = output["full_logits"]
        else:
            logits = output

        # Hedef sınıf belirtilmemişse, en yüksek olasılıklı sınıfı kullan
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass: Hedef sınıfın skoru için gradyan hesapla
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        # Gradyanlar ve aktivasyonlar
        gradients = self.gradients       # (1, C, h, w)
        activations = self.activations   # (1, C, h, w)

        # Global Average Pooling: Her kanalın önem ağırlığı
        # Gradyanın spatial boyutları üzerinden ortalama alınır
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Ağırlıklı toplam: önemli kanallar daha çok katkıda bulunur
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU: Sadece pozitif katkılar (negatifler sınıfı baskılıyor demektir)
        cam = F.relu(cam)

        # Orijinal görüntü boyutuna büyüt
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Normalize: [0, 1] aralığına çek
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def cleanup(self):
        """Hook'ları temizle (bellek sızıntısını önler)."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        self.cleanup()


class _GradCAMBackboneWrapper(nn.Module):
    """
    Backbone'un spatial çıkışını GradCAM için pool'layan wrapper.

    Backbone artık (B, S, dim) spatial token dizisi döndürüyor.
    GradCAM ise (B, dim) boyutunda bir çıkış bekliyor (gradyan hesabı için).
    Bu wrapper spatial token'ları average pool ile tek vektöre indirger.
    """

    def __init__(self, backbone_extractor):
        super().__init__()
        self.backbone = backbone_extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.backbone(x)          # (B, S, dim)
        return spatial.mean(dim=1)          # (B, dim)


def generate_gradcam_for_patient(
    model: nn.Module,
    images: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
) -> dict:
    """
    Bir hastanın 4 görüntüsü için Grad-CAM ısı haritaları üretir.

    Args:
        model: Tam model (MammographyClassifier).
        images: (1, 4, 3, H, W) tek hastanın görüntüleri.
        target_class: Hedef sınıf indeksi.
        target_layer: Grad-CAM katmanı (None ise otomatik bulunur).

    Returns:
        dict: Her görüntü için ısı haritası.
            {"RCC": ndarray, "LCC": ndarray, "RMLO": ndarray, "LMLO": ndarray}
    """
    view_names = ["RCC", "LCC", "RMLO", "LMLO"]

    # Backbone'un son konvolüsyon katmanını bul
    if target_layer is None:
        backbone_extractor = model.get_backbone_extractor()
        target_layer = backbone_extractor.get_last_conv_layer()

    # Backbone artık spatial çıkış veriyor (B, S, dim).
    # GradCAM için spatial token'ları pool'layan bir wrapper kullanıyoruz.
    backbone_wrapper = _GradCAMBackboneWrapper(model.backbone.backbone)
    backbone_wrapper.eval()

    heatmaps = {}
    for i, name in enumerate(view_names):
        gradcam = GradCAM(backbone_wrapper, target_layer)
        single_img = images[0, i].unsqueeze(0)  # (1, 3, H, W)
        heatmap = gradcam.generate(single_img, target_class)
        heatmaps[name] = heatmap
        gradcam.cleanup()

    return heatmaps


def save_gradcam_visualization(
    original_images: dict,
    heatmaps: dict,
    patient_id: str,
    true_label: int,
    pred_label: int,
    confidence: float,
    save_dir: str,
    inverse_normalize=None,
):
    """
    4 görüntü ve ısı haritalarını yan yana kaydeder.

    Üst satır: Orijinal görüntüler.
    Alt satır: Grad-CAM overlay'li görüntüler.

    Args:
        original_images: Orijinal görüntüler (normalize edilmiş tensor).
        heatmaps: Grad-CAM ısı haritaları.
        patient_id: Hasta ID'si (dosya adı için).
        true_label: Gerçek etiket.
        pred_label: Tahmin edilen etiket.
        confidence: Güven skoru.
        save_dir: Kayıt dizini.
        inverse_normalize: Normalize geri alma transform'u.
    """
    from pathlib import Path
    from data.transforms import get_inverse_normalize
    from data.dataset import INDEX_TO_BIRADS

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if inverse_normalize is None:
        inverse_normalize = get_inverse_normalize()

    view_names = ["RCC", "LCC", "RMLO", "LMLO"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for col, name in enumerate(view_names):
        # Orijinal görüntü
        img = original_images[name]
        if isinstance(img, torch.Tensor):
            img = inverse_normalize(img).clamp(0, 1)
            img = img.permute(1, 2, 0).cpu().numpy()

        # Üst satır: Orijinal
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(name, fontsize=14, fontweight="bold")
        axes[0, col].axis("off")

        # Alt satır: Grad-CAM overlay
        if name in heatmaps:
            axes[1, col].imshow(img, cmap="gray")
            axes[1, col].imshow(
                heatmaps[name], cmap="jet", alpha=0.4, vmin=0, vmax=1
            )
        axes[1, col].axis("off")

    # Başlık
    true_birads = INDEX_TO_BIRADS.get(true_label, "?")
    pred_birads = INDEX_TO_BIRADS.get(pred_label, "?")
    status = "DOĞRU" if true_label == pred_label else "YANLIŞ"
    fig.suptitle(
        f"Hasta: {patient_id} | Gerçek: BI-RADS {true_birads} | "
        f"Tahmin: BI-RADS {pred_birads} ({status}) | Güven: {confidence:.2%}",
        fontsize=16,
        fontweight="bold",
        color="green" if true_label == pred_label else "red",
    )

    plt.tight_layout()
    save_path = Path(save_dir) / f"gradcam_{patient_id}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)
