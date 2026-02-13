"""
Weights & Biases (WandB) Deney Takip Modülü
=============================================
Eğitim sürecindeki tüm metrikleri, parametreleri ve
artifact'ları (model ağırlıkları, grafikler) WandB'ye kaydeder.

MLFlow logger ile birlikte çalışır.

Kullanım:
    wandb_logger = WandbLogger(config)
    wandb_logger.start_run(run_name="full_maxvit_base")
    wandb_logger.log_params(config)
    wandb_logger.log_metrics({"accuracy": 0.95}, step=10)
    wandb_logger.log_artifact("model.pt")
    wandb_logger.end_run()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import wandb


class WandbLogger:
    """
    WandB tabanlı deney takip sınıfı.

    Args:
        config: YAML konfigürasyon sözlüğü.
    """

    def __init__(self, config: dict):
        self.config = config
        self.wandb_cfg = config.get("wandb", {})
        self.run = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[list] = None):
        """
        Yeni WandB run başlatır.

        Args:
            run_name: Run'a verilecek isim (örn: "baseline_resnet50").
            tags: Ek etiketler.
        """
        project = self.wandb_cfg.get("project", "mammography-birads-classification")
        entity = self.wandb_cfg.get("entity", None)
        # Boş string'i None'a çevir
        if entity == "":
            entity = None

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            reinit=True,
        )
        print(f"[WandB] Run başlatıldı: {self.run.id}")
        return self.run

    def log_params_flat(self, config: dict, prefix: str = ""):
        """
        İç içe config sözlüğünü düzleştirerek WandB config'e kaydeder.

        Örnek:
            {"model": {"backbone": {"name": "resnet50"}}}
            → "model.backbone.name" = "resnet50"
        """
        flat = {}
        self._flatten_dict(config, flat, prefix)
        wandb.config.update(flat, allow_val_change=True)

    def _flatten_dict(self, d: dict, flat: dict, prefix: str = ""):
        """Nested dict'i düzleştirir."""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_dict(value, flat, full_key)
            else:
                flat[full_key] = value

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Metrikleri kaydeder.

        Args:
            metrics: Metrik sözlüğü {"accuracy": 0.95, "loss": 0.1}.
            step: Eğitim adımı veya epoch numarası.
        """
        log_data = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_data[key] = value

        if step is not None:
            log_data["epoch"] = step

        wandb.log(log_data)

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """
        Dosyayı artifact olarak kaydeder (model, grafik, rapor vb.).

        Args:
            file_path: Kaydedilecek dosyanın yolu.
            artifact_path: Artifact türü/adı.
        """
        artifact_name = artifact_path or Path(file_path).stem
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_path or "file",
        )
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

    def log_model(self, model, artifact_path: str = "model"):
        """PyTorch modelini artifact olarak kaydeder."""
        artifact = wandb.Artifact(name=artifact_path, type="model")
        # Geçici dosyaya kaydet
        tmp_path = "tmp_wandb_model.pt"
        import torch
        torch.save(model.state_dict(), tmp_path)
        artifact.add_file(tmp_path)
        wandb.log_artifact(artifact)
        os.remove(tmp_path)

    def log_figure(self, fig, artifact_file: str):
        """Matplotlib figure'ını WandB'ye kaydeder."""
        wandb.log({artifact_file: wandb.Image(fig)})

    def log_text(self, text: str, artifact_file: str):
        """Metin dosyasını artifact olarak kaydeder."""
        artifact = wandb.Artifact(name=Path(artifact_file).stem, type="text")
        with artifact.new_file(artifact_file) as f:
            f.write(text)
        wandb.log_artifact(artifact)

    def end_run(self):
        """WandB run'ı sonlandırır."""
        if self.run:
            wandb.finish()
            print(f"[WandB] Run sonlandırıldı: {self.run.id}")
            self.run = None

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False
