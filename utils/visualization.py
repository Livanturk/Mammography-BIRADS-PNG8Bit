"""
Görselleştirme Modülü
======================
Eğitim sonrası üretilen grafikler ve raporlar.
- Confusion Matrix (karışıklık matrisi)
- Eğitim eğrileri (loss, accuracy, F1)
- Sınıflandırma raporu
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Confusion matrix görselleştirmesi.

    Args:
        cm: (N, N) confusion matrix (sklearn çıkışı).
        class_names: Sınıf isimleri.
        title: Grafik başlığı.
        save_path: Kayıt yolu (None ise kaydedilmez).
        normalize: Satır bazlı normalize et (oranlar göster).

    Returns:
        Matplotlib Figure nesnesi.
    """
    if class_names is None:
        class_names = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 4", "BI-RADS 5"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        # Satır toplamına böl: Her sınıfın doğru/yanlış oranını gösterir
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        # Gerçek sayıları da göster
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.75,
                    f"({cm[i, j]})",
                    ha="center", va="center",
                    fontsize=8, color="gray",
                )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

    ax.set_ylabel("Gerçek Etiket", fontsize=12)
    ax.set_xlabel("Tahmin Edilen", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GÖRSEL] Confusion matrix kaydedildi: {save_path}")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Eğitim ve doğrulama metriklerinin epoch bazlı grafiğini çizer.

    Args:
        history: Metrik geçmişi.
            {
                "train_loss": [0.8, 0.6, ...],
                "val_loss": [0.9, 0.7, ...],
                "train_full_f1_macro": [...],
                "val_full_f1_macro": [...],
                ...
            }
        save_path: Kayıt yolu.

    Returns:
        Matplotlib Figure nesnesi.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history.get("train_loss", [1])) + 1)

    # 1. Loss grafiği
    ax = axes[0, 0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", color="red", linestyle="--")
    ax.set_title("Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Full F1 Macro grafiği
    ax = axes[0, 1]
    if "train_full_f1_macro" in history:
        ax.plot(epochs, history["train_full_f1_macro"], label="Train F1", color="blue")
    if "val_full_f1_macro" in history:
        ax.plot(epochs, history["val_full_f1_macro"], label="Val F1", color="red", linestyle="--")
    ax.set_title("F1-Score (Macro)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Accuracy grafiği
    ax = axes[1, 0]
    if "train_full_accuracy" in history:
        ax.plot(epochs, history["train_full_accuracy"], label="Train Acc", color="blue")
    if "val_full_accuracy" in history:
        ax.plot(epochs, history["val_full_accuracy"], label="Val Acc", color="red", linestyle="--")
    ax.set_title("Accuracy", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Binary F1 grafiği
    ax = axes[1, 1]
    if "train_binary_f1" in history:
        ax.plot(epochs, history["train_binary_f1"], label="Train Binary F1", color="blue")
    if "val_binary_f1" in history:
        ax.plot(epochs, history["val_binary_f1"], label="Val Binary F1", color="red", linestyle="--")
    ax.set_title("Binary F1 (Benign vs Malign)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Eğitim Metrikleri", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GÖRSEL] Eğitim eğrileri kaydedildi: {save_path}")

    return fig


def save_classification_report(
    report_text: str,
    save_path: str,
    extra_info: Optional[dict] = None,
):
    """
    Sınıflandırma raporunu metin dosyası olarak kaydeder.

    Args:
        report_text: sklearn classification_report çıkışı.
        save_path: Kayıt yolu.
        extra_info: Ek bilgiler (model adı, epoch, vb.).
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BI-RADS Classification Report\n")
        f.write("=" * 60 + "\n\n")

        if extra_info:
            for key, value in extra_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "-" * 60 + "\n\n")

        f.write(report_text)
        f.write("\n")

    print(f"[RAPOR] Sınıflandırma raporu kaydedildi: {save_path}")
