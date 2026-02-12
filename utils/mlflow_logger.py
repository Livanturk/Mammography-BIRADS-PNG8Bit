"""
MLFlow + DagsHub Deney Takip Modülü
=====================================
Eğitim sürecindeki tüm metrikleri, parametreleri ve
artifact'ları (model ağırlıkları, grafikler) kaydeder.

DagsHub Entegrasyonu:
    DagsHub, MLFlow sunucusu olarak kullanılır.
    Ücretsiz plan ile sınırsız deney takibi yapılabilir.
    https://dagshub.com adresinden repo oluşturun.

Kullanım:
    logger = ExperimentLogger(config)
    logger.log_params(config)
    logger.log_metrics({"accuracy": 0.95}, step=10)
    logger.log_artifact("model.pt")
    logger.end_run()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch


class ExperimentLogger:
    """
    MLFlow tabanlı deney takip sınıfı.

    Args:
        config: YAML konfigürasyon sözlüğü.
    """

    def __init__(self, config: dict):
        self.config = config
        mlflow_cfg = config.get("mlflow", {})

        # DagsHub token ayarla (env variable veya config'den)
        dagshub_token = mlflow_cfg.get("dagshub_token", "")
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = "token"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # MLFlow tracking URI
        tracking_uri = mlflow_cfg.get(
            "tracking_uri", "mlruns"  # Varsayılan: yerel dizin
        )
        mlflow.set_tracking_uri(tracking_uri)

        # Experiment oluştur veya seç
        experiment_name = mlflow_cfg.get(
            "experiment_name", "mammography-birads"
        )
        mlflow.set_experiment(experiment_name)

        self.run = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[dict] = None):
        """
        Yeni MLFlow run başlatır.

        Args:
            run_name: Run'a verilecek isim (örn: "baseline_resnet50").
            tags: Ek etiketler.
        """
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        print(f"[MLFlow] Run başlatıldı: {self.run.info.run_id}")
        return self.run

    def log_params_flat(self, config: dict, prefix: str = ""):
        """
        İç içe config sözlüğünü düzleştirerek MLFlow'a kaydeder.

        Örnek:
            {"model": {"backbone": {"name": "resnet50"}}}
            → "model.backbone.name" = "resnet50"
        """
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self.log_params_flat(value, prefix=full_key)
            else:
                # MLFlow parametre anahtarı max 250 karakter
                try:
                    mlflow.log_param(full_key[:250], value)
                except Exception as e:
                    print(f"[MLFlow UYARI] Parametre loglanamadı: {full_key}: {e}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Metrikleri kaydeder.

        Args:
            metrics: Metrik sözlüğü {"accuracy": 0.95, "loss": 0.1}.
            step: Eğitim adımı veya epoch numarası.
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """
        Dosyayı artifact olarak kaydeder (model, grafik, rapor vb.).

        Args:
            file_path: Kaydedilecek dosyanın yolu.
            artifact_path: MLFlow'daki hedef alt dizin.
        """
        mlflow.log_artifact(file_path, artifact_path)

    def log_model(self, model, artifact_path: str = "model"):
        """PyTorch modelini artifact olarak kaydeder."""
        mlflow.pytorch.log_model(model, artifact_path)

    def log_figure(self, fig, artifact_file: str):
        """Matplotlib figure'ını artifact olarak kaydeder."""
        mlflow.log_figure(fig, artifact_file)

    def log_text(self, text: str, artifact_file: str):
        """Metin dosyasını artifact olarak kaydeder."""
        mlflow.log_text(text, artifact_file)

    def end_run(self):
        """MLFlow run'ı sonlandırır."""
        if self.run:
            mlflow.end_run()
            print(f"[MLFlow] Run sonlandırıldı: {self.run.info.run_id}")
            self.run = None

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False
