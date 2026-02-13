"""
Mammography Image Transforms
=============================
Eğitim ve doğrulama/test için görüntü dönüşüm pipeline'ları.

Mamografi görüntüleri için özel dikkat edilmesi gerekenler:
- Dikey çevirme genelde yapılmaz (anatomik yönelimi bozar).
- Aşırı renk/kontrast değişikliği diagnostik bilgiyi bozabilir.
- Grayscale görüntüler 3 kanala kopyalandığı için, 3 kanala da aynı
  normalizasyon değeri uygulanmalıdır. ImageNet'in grayscale karşılığı
  (luminance formula: 0.299R + 0.587G + 0.114B) kullanılır.
"""

from torchvision import transforms


# Grayscale-uyumlu ImageNet normalizasyon değerleri
# Grayscale→RGB kopyalama yapıldığı için 3 kanal identik olmalı.
# Değerler ImageNet luminance ortalamasından türetilmiştir:
#   mean = 0.299*0.485 + 0.587*0.456 + 0.114*0.406 ≈ 0.449
#   std  = sqrt(0.299*0.229² + 0.587*0.224² + 0.114*0.225²) ≈ 0.226
IMAGENET_MEAN = [0.449, 0.449, 0.449]
IMAGENET_STD = [0.226, 0.226, 0.226]


def get_train_transforms(data_cfg: dict) -> transforms.Compose:
    """
    Eğitim seti için augmentation pipeline'ı oluşturur.

    Config'den augmentation parametrelerini okur ve
    mamografi görüntülerine uygun dönüşümler uygular.
    """
    aug = data_cfg.get("augmentation", {})
    img_size = data_cfg["image_size"]

    transform_list = [
        transforms.Resize((img_size, img_size)),
    ]

    if aug.get("enabled", True):
        # Yatay çevirme: CC ve MLO görüntülerinde kullanılabilir
        if aug.get("horizontal_flip", 0) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=aug["horizontal_flip"])
            )

        # Hafif döndürme: Pozisyonlama farklılıklarını simüle eder
        if aug.get("rotation_degrees", 0) > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=aug["rotation_degrees"])
            )

        # Parlaklık ve kontrast değişimi: Farklı cihazları simüle eder
        brightness = aug.get("brightness", 0)
        contrast = aug.get("contrast", 0)
        if brightness > 0 or contrast > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                )
            )

        # Affine dönüşüm: Hafif zoom ve kaydırma
        transform_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Random erasing: Modelin küçük bölgelere bağımlılığını azaltır
    if aug.get("enabled", True) and aug.get("random_erasing", 0) > 0:
        transform_list.append(
            transforms.RandomErasing(p=aug["random_erasing"])
        )

    return transforms.Compose(transform_list)


def get_val_transforms(data_cfg: dict) -> transforms.Compose:
    """
    Doğrulama ve test seti için transform pipeline'ı.
    Augmentation uygulanmaz, sadece resize ve normalize yapılır.
    """
    img_size = data_cfg["image_size"]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inverse_normalize() -> transforms.Normalize:
    """
    Normalize işlemini tersine çevirir (Grad-CAM görselleştirmesi için).

    Formül: original = (normalized * std) + mean
    Ters normalize: value = (value - (-mean/std)) / (1/std)
    """
    inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    inv_std = [1.0 / s for s in IMAGENET_STD]
    return transforms.Normalize(mean=inv_mean, std=inv_std)
