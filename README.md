# Multi-View Mammography BI-RADS Classification

Hierarchical multi-head sınıflandırma sistemi. 4 mammografi görüntüsünden (RCC, LCC, RMLO, LMLO) BI-RADS 1, 2, 4, 5 kategorilerini tahmin eder.

## Proje Yapısı

```
mammography_birads/
├── config.yaml              # Tüm ayarlar (backbone, lr, loss weights, vb.)
├── train.py                 # Ana eğitim scripti
├── requirements.txt         # Bağımlılıklar
│
├── models/
│   ├── backbone.py          # Seviye 1: Weight-shared backbone (ResNet, EfficientNet, ConvNeXt)
│   ├── lateral_fusion.py    # Seviye 2: Bi-directional cross-attention (CC ↔ MLO)
│   ├── bilateral_fusion.py  # Seviye 3: Asimetri tespiti (Left ↔ Right)
│   ├── classification_heads.py  # Seviye 4: Multi-head classifier + uncertainty
│   ├── full_model.py        # 4 seviyeyi birleştiren ana model
│   └── gradcam.py           # Grad-CAM görselleştirme
│
├── data/
│   ├── dataset.py           # Hasta bazlı Dataset + DataLoader
│   └── transforms.py        # Augmentation pipeline'ları
│
└── utils/
    ├── losses.py            # Multi-head loss function
    ├── metrics.py           # Metrik hesaplama (F1, AUC, CM)
    ├── mlflow_logger.py     # MLFlow + DagsHub entegrasyonu
    └── visualization.py     # Grafikler ve raporlar
```

## Hızlı Başlangıç

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt

# 2. Config'i düzenle
#    - data.root_dir: Veri seti yolunu ayarla
#    - mlflow.tracking_uri: DagsHub MLFlow URI'sini ayarla

# 3. Baseline deney (sadece backbone + 4-class head)
python train.py --baseline

# 4. Tam model eğitimi
python train.py
```

## Veri Seti Yapısı

Etiketler klasör yapısından otomatik olarak okunur — ayrı CSV dosyası gerekmez.

```
dataset_root/
├── BI-RADS1/
│   ├── patient_001/
│   │   ├── RCC.png         # Sağ Craniocaudal (384x384, 8-bit)
│   │   ├── LCC.png         # Sol Craniocaudal
│   │   ├── RMLO.png        # Sağ Mediolateral Oblique
│   │   └── LMLO.png        # Sol Mediolateral Oblique
│   ├── patient_002/
│   │   └── ...
│   └── ...
├── BI-RADS2/
│   └── ...
├── BI-RADS4/
│   └── ...
└── BI-RADS5/
    └── ...
```

Sınıf dağılımı: BI-RADS 1 (1678), BI-RADS 2 (2754), BI-RADS 4 (1898), BI-RADS 5 (2274).

## Model Mimarisi (4 Seviye)

```
Görüntüler (4x)  →  Backbone  →  Cross-Attention  →  Bilateral Fusion  →  Multi-Head
   RCC ─────────┐                                                         ├─ Binary (B/M)
   RMLO ────────┤   Weight-     Right Lateral ──┐                         ├─ Benign (1/2)
                ├── Shared   →  (CC ↔ MLO)      ├── Asymmetry  →         ├─ Malign (4/5)
   LCC ─────────┤   EfficientNet Left Lateral ──┘    Detection            └─ Full (1/2/4/5)
   LMLO ────────┘
```

## Ablation Study Rehberi

Projenin temel amacı, her bileşenin katkısını sistematik olarak ölçmektir. Aşağıdaki sırayla deneyleri çalıştır:

### Adım 1: Baseline (Backbone Only)

```bash
python train.py --baseline
```

`config.yaml` içinde otomatik olarak şu ayarlar uygulanır:
- Lateral fusion: KAPALI
- Bilateral fusion: KAPALI
- Binary/Subgroup head: KAPALI
- Sadece backbone → mean pooling → full_head

### Adım 2: Backbone Karşılaştırması

`config.yaml`'daki `model.backbone.name` değerini değiştirerek farklı backbone'ları dene:

| Backbone | Feature Dim | Config Değeri |
|----------|-------------|---------------|
| ResNet50 | 2048 | `resnet50` |
| EfficientNet-B0 | 1280 | `efficientnet_b0` |
| EfficientNet-B3 | 1536 | `efficientnet_b3` |
| ConvNeXt-Tiny | 768 | `convnext_tiny` |

> **Not:** `model.backbone.feature_dim` değerini de backbone'a göre güncelle.

### Adım 3: Lateral Fusion Ekleme

En iyi backbone'u seçtikten sonra:

```yaml
ablation:
  use_lateral_fusion: true       # AÇIK
  use_bilateral_fusion: false    # KAPALI
  use_binary_head: false
  use_subgroup_head: false
```

### Adım 4: Bilateral Fusion Ekleme

```yaml
ablation:
  use_lateral_fusion: true       # AÇIK
  use_bilateral_fusion: true     # AÇIK
  use_binary_head: false
  use_subgroup_head: false
```

### Adım 5: Multi-Head Classification

```yaml
ablation:
  use_lateral_fusion: true
  use_bilateral_fusion: true
  use_binary_head: true          # AÇIK
  use_subgroup_head: true        # AÇIK
  use_uncertainty: true          # AÇIK
```

### Adım 6: Loss Weight Optimizasyonu

Son adımda loss ağırlıklarını dene:

```yaml
training:
  loss_weights:
    binary_head: 0.2    # veya 0.3, 0.4
    subgroup_head: 0.3   # veya 0.2, 0.4
    full_head: 0.5       # veya 0.4, 0.3
```

### Beklenen Sonuç Tablosu

| Deney | Lateral | Bilateral | Multi-Head | Val F1 |
|-------|---------|-----------|------------|--------|
| Baseline | ✗ | ✗ | ✗ | ? |
| + Lateral | ✓ | ✗ | ✗ | ? |
| + Bilateral | ✓ | ✓ | ✗ | ? |
| + Multi-Head | ✓ | ✓ | ✓ | ? |

## MLFlow ile Deney Takibi

1. [DagsHub](https://dagshub.com) hesabı oluştur
2. Yeni repo oluştur
3. `config.yaml`'daki MLFlow ayarlarını güncelle:

```yaml
mlflow:
  tracking_uri: "https://dagshub.com/<username>/<repo>.mlflow"
  experiment_name: "birads-classification"
```

4. DagsHub token'ı environment variable olarak ayarla:
```bash
export MLFLOW_TRACKING_USERNAME=token
export MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

## Çıktılar

Eğitim sonrası `outputs/` dizininde:
- `checkpoints/best_model.pt` — En iyi model ağırlıkları
- `plots/confusion_matrix.png` — Karışıklık matrisi
- `plots/training_curves.png` — Eğitim eğrileri
- `reports/classification_report.txt` — Detaylı sınıflandırma raporu
- `gradcam/` — Grad-CAM görselleştirmeleri
