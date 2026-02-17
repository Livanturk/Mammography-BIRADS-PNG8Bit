"""
Ana Eğitim Scripti
====================
Multi-view mammografi BI-RADS sınıflandırma modelini eğitir.

Kullanım:
    python train.py                          # Varsayılan config ile
    python train.py --config my_config.yaml  # Özel config ile
    python train.py --baseline               # Baseline deney

Eğitim Döngüsü:
    1. Config yüklenir, seed ayarlanır.
    2. DataLoader'lar oluşturulur.
    3. Model, optimizer ve scheduler oluşturulur.
    4. Her epoch'ta:
       a. Train: Forward → Loss → Backward → Optimizer step
       b. Validation: Forward → Metrik hesaplama
       c. Early stopping kontrolü
       d. MLFlow'a loglama
    5. En iyi model ile test seti değerlendirilir.
    6. Grad-CAM, confusion matrix ve rapor üretilir.
"""

import argparse
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

from data.dataset import create_dataloaders
from models.full_model import build_model, build_baseline_config
from models.gradcam import generate_gradcam_for_patient, save_gradcam_visualization
from utils.losses import build_loss_function
from utils.metrics import MetricTracker
from utils.mlflow_logger import ExperimentLogger
from utils.wandb_logger import WandbLogger
from utils.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    save_classification_report,
)


def set_seed(seed: int):
    """Tekrarlanabilirlik için tüm rastgelelik kaynaklarını sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """YAML config dosyasını yükler."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _get_param_groups(model: nn.Module, config: dict) -> list:
    """
    Layer-wise weight decay ve discriminative learning rate ile
    parametre grupları oluşturur.

    Prensipler:
    - Backbone (pretrained): Düşük LR + yüksek WD → orijinalden uzaklaşmayı sınırla
    - Fusion katmanları: Normal LR + orta WD → domain-specific öğrenme
    - Classification heads: Normal LR + düşük WD → dropout zaten regularize ediyor
    - LayerNorm/Bias: WD=0 → scale/shift parametrelerine decay uygulanmamalı
    """
    opt_cfg = config["training"]["optimizer"]
    base_lr = opt_cfg["lr"]
    backbone_lr = base_lr * opt_cfg.get("backbone_lr_scale", 0.1)

    # Weight decay değerleri (geriye uyumluluk: eski config tek float olabilir)
    wd_cfg = opt_cfg["weight_decay"]
    if isinstance(wd_cfg, (int, float)):
        wd_backbone = float(wd_cfg)
        wd_fusion = float(wd_cfg)
        wd_head = float(wd_cfg)
    else:
        wd_backbone = wd_cfg.get("backbone", 0.1)
        wd_fusion = wd_cfg.get("fusion", 0.05)
        wd_head = wd_cfg.get("head", 0.01)

    # Parametre kategorileri
    backbone_decay = []
    backbone_no_decay = []
    fusion_decay = []
    fusion_no_decay = []
    head_decay = []
    head_no_decay = []

    # LayerNorm ve bias parametrelerini ayırt etme fonksiyonu
    no_decay_keywords = {"bias", "LayerNorm", "layernorm", "layer_norm",
                         "norm", "ln", "log_temperature"}

    def is_no_decay(name):
        return any(kw in name for kw in no_decay_keywords)

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Backbone parametreleri
        if param_name.startswith("backbone.backbone.backbone"):
            if is_no_decay(param_name):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        # Backbone projection
        elif param_name.startswith("backbone.backbone.projection"):
            if is_no_decay(param_name):
                fusion_no_decay.append(param)
            else:
                fusion_decay.append(param)
        # Fusion katmanları (lateral + bilateral)
        elif "lateral_fusion" in param_name or "bilateral_fusion" in param_name:
            if is_no_decay(param_name):
                fusion_no_decay.append(param)
            else:
                fusion_decay.append(param)
        # Basit projection (baseline mod)
        elif "simple_lateral_proj" in param_name or "simple_bilateral_proj" in param_name:
            if is_no_decay(param_name):
                fusion_no_decay.append(param)
            else:
                fusion_decay.append(param)
        # Classification heads
        elif "classifier" in param_name:
            if is_no_decay(param_name):
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            # Bilinmeyen parametreler → fusion grubuna ata
            if is_no_decay(param_name):
                fusion_no_decay.append(param)
            else:
                fusion_decay.append(param)

    param_groups = []
    if backbone_decay:
        param_groups.append({
            "params": backbone_decay,
            "lr": backbone_lr,
            "weight_decay": wd_backbone,
            "group_name": "backbone_decay",
        })
    if backbone_no_decay:
        param_groups.append({
            "params": backbone_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
            "group_name": "backbone_no_decay",
        })
    if fusion_decay:
        param_groups.append({
            "params": fusion_decay,
            "lr": base_lr,
            "weight_decay": wd_fusion,
            "group_name": "fusion_decay",
        })
    if fusion_no_decay:
        param_groups.append({
            "params": fusion_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "group_name": "fusion_no_decay",
        })
    if head_decay:
        param_groups.append({
            "params": head_decay,
            "lr": base_lr,
            "weight_decay": wd_head,
            "group_name": "head_decay",
        })
    if head_no_decay:
        param_groups.append({
            "params": head_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "group_name": "head_no_decay",
        })

    # Özet yazdır
    total = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"\n[OPTİMİZER] Layer-wise parametre grupları:")
    for g in param_groups:
        count = sum(p.numel() for p in g["params"])
        print(f"  {g['group_name']:25s}: {count:>10,} params | lr={g['lr']:.2e} | wd={g['weight_decay']:.3f}")
    print(f"  {'TOPLAM':25s}: {total:>10,} params")

    return param_groups


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Config'e göre layer-wise weight decay ile optimizer oluşturur."""
    opt_cfg = config["training"]["optimizer"]
    name = opt_cfg["name"].lower()

    param_groups = _get_param_groups(model, config)

    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    elif name == "adam":
        return torch.optim.Adam(
            param_groups,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=0.9,
            nesterov=True,
        )
    else:
        raise ValueError(f"Desteklenmeyen optimizer: {name}")


def build_scheduler(optimizer, config: dict, steps_per_epoch: int = None):
    """Config'e göre learning rate scheduler oluşturur.

    Args:
        optimizer: Optimizer instance.
        config: Tam config dict.
        steps_per_epoch: Epoch başına batch sayısı (OneCycleLR için gerekli).
    """
    sched_cfg = config["training"]["scheduler"]
    name = sched_cfg["name"].lower()
    epochs = config["training"]["epochs"]

    if name == "cosine_warmup":
        warmup_epochs = sched_cfg.get("warmup_epochs", 5)
        min_lr = sched_cfg.get("min_lr", 1e-6)

        def warmup_cosine_fn(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return max(
                    min_lr / config["training"]["optimizer"]["lr"],
                    0.5 * (1 + np.cos(np.pi * progress))
                )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_fn)

    elif name == "cosine_warm_restarts":
        T_0 = sched_cfg.get("T_0", 20)
        T_mult = sched_cfg.get("T_mult", 2)
        eta_min = sched_cfg.get("eta_min", 1e-7)

        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
        )

    elif name == "onecycle":
        assert steps_per_epoch is not None, \
            "OneCycleLR için steps_per_epoch gerekli"

        max_lr = sched_cfg.get("max_lr", 5e-4)
        backbone_lr_scale = config["training"]["optimizer"].get("backbone_lr_scale", 0.2)

        # Her param grup için max_lr: backbone grupları scaled, diğerleri tam
        max_lrs = []
        for group in optimizer.param_groups:
            group_name = group.get("group_name", "")
            if "backbone" in group_name:
                max_lrs.append(max_lr * backbone_lr_scale)
            else:
                max_lrs.append(max_lr)

        # Efektif optimizer step sayısı (gradient accumulation dahil)
        grad_accum = config["training"].get("gradient_accumulation_steps", 1)
        effective_steps = (steps_per_epoch + grad_accum - 1) // grad_accum

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=epochs,
            steps_per_epoch=effective_steps,
            pct_start=sched_cfg.get("pct_start", 0.3),
            anneal_strategy=sched_cfg.get("anneal_strategy", "cos"),
            div_factor=sched_cfg.get("div_factor", 10.0),
            final_div_factor=sched_cfg.get("final_div_factor", 100.0),
        )

    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 30),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    else:
        raise ValueError(f"Desteklenmeyen scheduler: {name}")


class EarlyStopping:
    """
    Early stopping: Validation metriki iyileşmezse eğitimi durdurur.

    Args:
        patience: Kaç epoch iyileşme olmazsa durdurulur.
        mode: "max" (metrik artmalı) veya "min" (metrik azalmalı).
        min_delta: Minimum iyileşme miktarı.
    """

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: GradScaler,
    config: dict,
    tracker: MetricTracker,
    scheduler=None,
) -> dict:
    """
    Bir epoch eğitim döngüsü.

    Gradient Accumulation:
        Büyük batch boyutlarını simüle etmek için gradyanlar
        biriktrilir ve belirli aralıklarla güncellenir.
        Efektif batch = batch_size × gradient_accumulation_steps
    """
    model.train()
    tracker.reset()

    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="  Train", leave=False, ncols=100)
    for step, batch in enumerate(pbar):
        images = batch["images"].to(device)     # (B, 4, 3, H, W)
        labels = batch["label"].to(device)       # (B,)

        # Mixed Precision Training: float16 ile hesaplama (daha hızlı)
        with autocast():
            outputs = model(images)
            loss_dict = criterion(outputs, labels)
            loss = loss_dict["total_loss"] / grad_accum

        # Backward pass (scaled gradients)
        scaler.scale(loss).backward()

        # Gradient accumulation: Her grad_accum adımda güncelle
        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            # Gradient clipping: Patlayan gradyanları önle
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Step-level scheduler (OneCycleLR)
            if scheduler is not None:
                scheduler.step()

        # Metrikleri birikdir
        tracker.update(outputs, labels, loss_dict)
        pbar.set_postfix(loss=f"{loss_dict['total_loss'].item():.4f}")

    pbar.close()
    return tracker.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    tracker: MetricTracker,
) -> dict:
    """Validation veya test döngüsü (gradyan hesaplanmaz)."""
    model.eval()
    tracker.reset()

    pbar = tqdm(dataloader, desc="  Val  ", leave=False, ncols=100)
    for batch in pbar:
        images = batch["images"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss_dict = criterion(outputs, labels)

        tracker.update(outputs, labels, loss_dict)
        pbar.set_postfix(loss=f"{loss_dict['total_loss'].item():.4f}")

    pbar.close()
    return tracker.compute()


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    save_path: str,
):
    """Model checkpoint'ını kaydeder."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        },
        save_path,
    )


def main(config_path: str, baseline: bool = False, device_id: int = 0):
    """Ana eğitim fonksiyonu."""

    # --- Konfigürasyon ---
    config = load_config(config_path)

    if baseline:
        config = build_baseline_config(config)
        print("[BİLGİ] Baseline modu aktif: Lateral ve Bilateral fusion kapalı.")

    seed = config["project"]["seed"]
    set_seed(seed)

    # --- Cihaz ---
    if torch.cuda.is_available() and device_id >= 0:
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    print(f"[BİLGİ] Cihaz: {device}")
    if device.type == "cuda":
        print(f"[BİLGİ] GPU: {torch.cuda.get_device_name(device_id)}")

    # --- Veri ---
    print("\n[1/5] Veri yükleniyor...")
    dataloaders = create_dataloaders(config)

    # --- Model ---
    print("\n[2/5] Model oluşturuluyor...")
    model = build_model(config).to(device)

    # --- Loss, Optimizer, Scheduler ---
    print("\n[3/5] Eğitim bileşenleri hazırlanıyor...")
    criterion = build_loss_function(config, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(dataloaders["train"]))
    scaler = GradScaler()  # Mixed precision

    # Early stopping
    es_cfg = config["training"].get("early_stopping", {})
    early_stopping = None
    if es_cfg.get("enabled", True):
        early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 15),
            mode=es_cfg.get("mode", "max"),
        )

    # --- MLFlow Logger ---
    logger = ExperimentLogger(config)
    run_name = f"{'baseline' if baseline else 'full'}_{config['model']['backbone']['name']}"
    logger.start_run(run_name=run_name)
    logger.log_params_flat(config)

    # --- WandB Logger ---
    wandb_logger = WandbLogger(config)
    wandb_logger.start_run(run_name=run_name)
    wandb_logger.log_params_flat(config)

    # --- Eğitim Döngüsü ---
    print("\n[4/5] Eğitim başlıyor...")
    train_tracker = MetricTracker()
    val_tracker = MetricTracker()
    history = {}
    best_val_f1 = 0.0
    best_model_state = None

    epochs = config["training"]["epochs"]
    checkpoint_dir = config["checkpoint"]["save_dir"]

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # OneCycleLR step-level çalışır, diğerleri epoch-level
        is_step_scheduler = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

        # Eğitim
        train_metrics = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer,
            device, scaler, config, train_tracker,
            scheduler=scheduler if is_step_scheduler else None,
        )

        # Doğrulama
        val_metrics = evaluate(
            model, dataloaders["val"], criterion, device, val_tracker,
        )

        # Epoch-level scheduler step (OneCycleLR hariç — o train_one_epoch içinde step eder)
        if not is_step_scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get("full_f1_macro", 0))
            else:
                scheduler.step()

        # Metrikleri logla
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        log_metrics = {}
        for k, v in train_metrics.items():
            log_metrics[f"train_{k}"] = v
        for k, v in val_metrics.items():
            log_metrics[f"val_{k}"] = v
        log_metrics["lr"] = current_lr
        log_metrics["epoch_time_s"] = epoch_time

        logger.log_metrics(log_metrics, step=epoch)
        wandb_logger.log_metrics(log_metrics, step=epoch)

        # History'ye ekle
        for k, v in log_metrics.items():
            if isinstance(v, (int, float)):
                history.setdefault(k, []).append(v)

        # En iyi model
        val_f1 = val_metrics.get("full_f1_macro", 0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(checkpoint_dir, "best_model.pt"),
            )

        # Konsol çıktısı
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics.get('total_loss', 0):.4f} | "
            f"Val Loss: {val_metrics.get('total_loss', 0):.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Acc: {val_metrics.get('full_accuracy', 0):.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Süre: {epoch_time:.1f}s"
        )

        # Early stopping
        if early_stopping and early_stopping.step(val_f1):
            print(f"\n[BİLGİ] Early stopping: {early_stopping.patience} epoch iyileşme yok.")
            break

    # --- Test Değerlendirmesi ---
    print("\n[5/5] Test değerlendirmesi yapılıyor...")

    # En iyi modeli yükle
    if best_model_state:
        model.load_state_dict(best_model_state)

    test_tracker = MetricTracker()
    test_metrics = evaluate(
        model, dataloaders["test"], criterion, device, test_tracker,
    )

    # Test metriklerini logla
    test_log = {f"test_{k}": v for k, v in test_metrics.items()}
    logger.log_metrics(test_log)
    wandb_logger.log_metrics(test_log)

    print("\n" + "=" * 60)
    print("TEST SONUÇLARI")
    print("=" * 60)
    print(f"  Accuracy:    {test_metrics.get('full_accuracy', 0):.4f}")
    print(f"  F1 (Macro):  {test_metrics.get('full_f1_macro', 0):.4f}")
    print(f"  AUC-ROC:     {test_metrics.get('full_auc_roc', 0):.4f}")
    print(f"  Binary F1:   {test_metrics.get('binary_f1', 0):.4f}")

    # --- Görselleştirmeler ---
    vis_cfg = config.get("visualization", {})

    # Confusion Matrix
    cm = test_tracker.get_confusion_matrix()
    cm_path = os.path.join(vis_cfg.get("confusion_matrix", {}).get("save_dir", "outputs/plots"), "confusion_matrix.png")
    cm_fig = plot_confusion_matrix(cm, save_path=cm_path)
    logger.log_artifact(cm_path, "plots")
    wandb_logger.log_artifact(cm_path, "plots")

    # Classification Report
    report = test_tracker.get_classification_report()
    report_path = os.path.join(vis_cfg.get("classification_report", {}).get("save_dir", "outputs/reports"), "classification_report.txt")
    save_classification_report(report, report_path, extra_info={
        "Model": config["model"]["backbone"]["name"],
        "Best Val F1": f"{best_val_f1:.4f}",
        "Baseline": str(baseline),
    })
    logger.log_artifact(report_path, "reports")
    wandb_logger.log_artifact(report_path, "reports")
    print(f"\n{report}")

    # Eğitim eğrileri
    curves_path = os.path.join(vis_cfg.get("confusion_matrix", {}).get("save_dir", "outputs/plots"), "training_curves.png")
    plot_training_curves(history, save_path=curves_path)
    logger.log_artifact(curves_path, "plots")
    wandb_logger.log_artifact(curves_path, "plots")

    # Grad-CAM (opsiyonel)
    gradcam_cfg = vis_cfg.get("gradcam", {})
    if gradcam_cfg.get("enabled", False):
        print("\n[BİLGİ] Grad-CAM görselleştirmeleri üretiliyor...")
        gradcam_dir = gradcam_cfg.get("save_dir", "outputs/gradcam")
        num_samples = gradcam_cfg.get("num_samples", 20)

        model.eval()
        test_loader = dataloaders["test"]
        num_classes = config["model"]["classification"]["num_classes"]
        samples_per_class = num_samples // num_classes
        class_counts = {c: 0 for c in range(num_classes)}
        samples_done = 0

        for batch in test_loader:
            if samples_done >= num_samples:
                break

            images = batch["images"].to(device)
            labels = batch["label"]
            patient_ids = batch["patient_id"]

            with torch.no_grad():
                outputs = model(images)
                preds = outputs["full_logits"].argmax(dim=-1)
                confidences = outputs.get("confidence", torch.zeros(len(labels)))

            for i in range(len(labels)):
                if samples_done >= num_samples:
                    break

                label_class = labels[i].item()
                if class_counts[label_class] >= samples_per_class:
                    continue

                try:
                    heatmaps = generate_gradcam_for_patient(
                        model=model,
                        images=images[i:i+1],
                        target_class=preds[i].item(),
                    )

                    original_views = {
                        name: images[i, j]
                        for j, name in enumerate(["RCC", "LCC", "RMLO", "LMLO"])
                    }

                    save_gradcam_visualization(
                        original_images=original_views,
                        heatmaps=heatmaps,
                        patient_id=patient_ids[i],
                        true_label=labels[i].item(),
                        pred_label=preds[i].item(),
                        confidence=confidences[i].item(),
                        save_dir=gradcam_dir,
                    )
                    class_counts[label_class] += 1
                    samples_done += 1
                except Exception as e:
                    print(f"[UYARI] Grad-CAM hatası ({patient_ids[i]}): {e}")

        # Grad-CAM dosyalarını artifact olarak logla
        if os.path.exists(gradcam_dir):
            for f in os.listdir(gradcam_dir):
                logger.log_artifact(os.path.join(gradcam_dir, f), "gradcam")
                wandb_logger.log_artifact(os.path.join(gradcam_dir, f), "gradcam")

    # Temizlik
    logger.end_run()
    wandb_logger.end_run()
    print("\n[BİLGİ] Eğitim tamamlandı!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-View Mammography BI-RADS Classification Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config dosyası yolu",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline deney (lateral/bilateral fusion kapalı)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Kullanılacak GPU indeksi (örn: 0, 1, 2, 3)",
    )

    args = parser.parse_args()
    main(config_path=args.config, baseline=args.baseline, device_id=args.device)
