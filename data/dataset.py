"""
Mammography Multi-View Patient-Level Dataset
=============================================
Bu modül hasta bazlı mammografi veri setini yükler.
Her hasta için 4 görüntü (RCC, LCC, RMLO, LMLO) tek bir örnek olarak döndürülür.

Klasör Yapısı:
    root_dir/
        BI-RADS1/
            patient_001/
                RCC.png     (Sağ Craniocaudal)
                LCC.png     (Sol Craniocaudal)
                RMLO.png    (Sağ Mediolateral Oblique)
                LMLO.png    (Sol Mediolateral Oblique)
            patient_002/
                ...
        BI-RADS2/
            ...
        BI-RADS4/
            ...
        BI-RADS5/
            ...

    Etiketler klasör yapısından otomatik olarak çıkarılır.
    BI-RADS klasör adı → sınıf etiketi dönüşümü:
        BI-RADS1 → 0, BI-RADS2 → 1, BI-RADS4 → 2, BI-RADS5 → 3

Önemli:
    - Görüntüler 384x384, 8-bit PNG formatında olmalı.
    - Her hastanın 4 görüntüsü de mevcut olmalıdır.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.transforms import get_train_transforms, get_val_transforms


# Her mammografi görüntüsünün standart adı
VIEW_NAMES = ["RCC", "LCC", "RMLO", "LMLO"]


class MammographyDataset(Dataset):
    """
    Hasta bazlı mammografi veri seti.

    Her __getitem__ çağrısında bir hastanın 4 görüntüsünü ve etiketini döndürür.

    Args:
        patient_dirs: Hasta klasör yollarının listesi.
        labels: Her hasta için BI-RADS etiketi (0-indexed: 0=BIRADS1, 1=BIRADS2, 2=BIRADS4, 3=BIRADS5).
        transform: Torchvision transform pipeline.
        view_names: Görüntü dosya adları listesi (varsayılan: RCC, LCC, RMLO, LMLO).

    Returns:
        dict:
            - "images": (4, C, H, W) boyutunda tensor — 4 görüntü yığını.
            - "label": Skaler tensor — BI-RADS sınıf indeksi.
            - "patient_id": str — Hasta klasör adı.
    """

    def __init__(
        self,
        patient_dirs: List[str],
        labels: List[int],
        transform=None,
        view_names: List[str] = None,
    ):
        assert len(patient_dirs) == len(labels), (
            f"Hasta sayısı ({len(patient_dirs)}) ve etiket sayısı ({len(labels)}) eşleşmiyor!"
        )

        self.patient_dirs = patient_dirs
        self.labels = labels
        self.transform = transform
        self.view_names = view_names or VIEW_NAMES

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_dir = Path(self.patient_dirs[idx])
        label = self.labels[idx]
        patient_id = patient_dir.name

        images = []
        for view_name in self.view_names:
            img_path = patient_dir / f"{view_name}.png"

            if not img_path.exists():
                raise FileNotFoundError(
                    f"Görüntü bulunamadı: {img_path}\n"
                    f"Hasta klasöründe {view_name}.png dosyası eksik."
                )

            # PNG'yi grayscale olarak oku, 3 kanala genişlet (pretrained backbone için)
            img = Image.open(img_path).convert("L")  # Grayscale
            img = Image.merge("RGB", [img, img, img])  # 3-kanal kopyası

            if self.transform:
                img = self.transform(img)

            images.append(img)

        # (4, C, H, W) boyutunda tensor oluştur
        images_tensor = torch.stack(images, dim=0)

        return {
            "images": images_tensor,            # (4, 3, 384, 384)
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
        }


# BI-RADS klasör adlarını 0-indexed sınıf indekslerine çevirme haritası
# Klasör adı → sınıf indeksi
BIRADS_FOLDER_TO_INDEX = {
    "BI-RADS_1": 0,
    "BI-RADS_2": 1,
    "BI-RADS_4": 2,
    "BI-RADS_5": 3,
}

# Sınıf indeksi → BI-RADS numarası (görselleştirme için)
INDEX_TO_BIRADS = {0: 1, 1: 2, 2: 4, 3: 5}

# Geriye uyumluluk için eski harita da korunur
BIRADS_TO_INDEX = {1: 0, 2: 1, 4: 2, 5: 3}

# Binary (Benign/Malign) etiket haritası
BIRADS_TO_BINARY = {1: 0, 2: 0, 4: 1, 5: 1}  # 0=Benign, 1=Malign


def scan_dataset_from_folders(root_dir: str) -> Tuple[List[str], List[int]]:
    """
    Klasör yapısından hasta dizinlerini ve etiketlerini tarar.

    Beklenen yapı:
        root_dir/
            BI-RADS1/
                patient_001/   (içinde RCC.png, LCC.png, RMLO.png, LMLO.png)
                patient_002/
            BI-RADS2/
                ...
            BI-RADS4/
                ...
            BI-RADS5/
                ...

    Args:
        root_dir: Veri setinin kök dizini (BI-RADS klasörlerini içeren).

    Returns:
        tuple: (patient_dirs, labels)
            - patient_dirs: Hasta klasör yollarının listesi.
            - labels: Her hasta için 0-indexed sınıf etiketi.

    Raises:
        FileNotFoundError: BI-RADS klasörleri bulunamazsa.
        ValueError: Hasta klasöründe eksik görüntü varsa.
    """
    patient_dirs = []
    labels = []
    skipped = 0

    # Beklenen BI-RADS klasörleri
    expected_folders = list(BIRADS_FOLDER_TO_INDEX.keys())
    found_folders = []

    for birads_folder, class_idx in BIRADS_FOLDER_TO_INDEX.items():
        birads_path = os.path.join(root_dir, birads_folder)

        if not os.path.isdir(birads_path):
            print(f"[UYARI] BI-RADS klasörü bulunamadı: {birads_path}")
            continue

        found_folders.append(birads_folder)

        # Her hasta klasörünü tara
        for patient_id in sorted(os.listdir(birads_path)):
            patient_path = os.path.join(birads_path, patient_id)

            if not os.path.isdir(patient_path):
                continue  # Dosyaları atla, sadece klasörler

            # 4 görüntünün varlığını kontrol et
            missing_views = []
            for view_name in VIEW_NAMES:
                img_path = os.path.join(patient_path, f"{view_name}.png")
                if not os.path.isfile(img_path):
                    missing_views.append(view_name)

            if missing_views:
                print(
                    f"[UYARI] Eksik görüntü, hasta atlanıyor: {patient_path} "
                    f"(eksik: {', '.join(missing_views)})"
                )
                skipped += 1
                continue

            patient_dirs.append(patient_path)
            labels.append(class_idx)

    # Özet bilgi
    if not found_folders:
        raise FileNotFoundError(
            f"Hiçbir BI-RADS klasörü bulunamadı: {root_dir}\n"
            f"Beklenen klasörler: {expected_folders}\n"
            f"Mevcut içerik: {os.listdir(root_dir) if os.path.isdir(root_dir) else 'DİZİN YOK'}"
        )

    print(f"\n[BİLGİ] Veri seti tarama sonucu:")
    print(f"  Kök dizin: {root_dir}")
    print(f"  Bulunan BI-RADS klasörleri: {found_folders}")
    for folder in found_folders:
        count = labels.count(BIRADS_FOLDER_TO_INDEX[folder])
        print(f"    {folder}: {count} hasta")
    print(f"  Toplam hasta: {len(patient_dirs)}")
    if skipped > 0:
        print(f"  Atlanan (eksik görüntü): {skipped}")

    return patient_dirs, labels


def prepare_patient_split(
    root_dir: str,
    test_dir: str,
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Hasta bazlı stratified train/val split uygular, test seti ayrı klasörden okunur.

    Train ve val, root_dir içindeki veriden stratified split ile ayrılır.
    Test seti ise test_dir klasöründen bağımsız olarak yüklenir.

    Args:
        root_dir: Train/Val veri setinin kök dizini (BI-RADS1/, BI-RADS2/, ... içeren).
        test_dir: Sabit test seti kök dizini (aynı BI-RADS alt yapısı).
        train_ratio: Eğitim seti oranı (root_dir içinden).
        val_ratio: Doğrulama seti oranı (root_dir içinden).
        seed: Rastgelelik tohumu (tekrarlanabilirlik için).

    Returns:
        dict: Her split için (patient_dirs, labels) tuple'ları.
            {
                "train": ([dir1, dir2, ...], [label1, label2, ...]),
                "val": ([dir1, dir2, ...], [label1, label2, ...]),
                "test": ([dir1, dir2, ...], [label1, label2, ...]),
            }
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, (
        "Train ve val oranları toplamı 1.0 olmalıdır."
    )

    # Train/Val: Klasör yapısından hastaları ve etiketleri tara
    patient_dirs, labels = scan_dataset_from_folders(root_dir)
    labels_arr = np.array(labels)

    # Train vs Val bölme
    train_dirs, val_dirs, train_labels, val_labels = train_test_split(
        patient_dirs, labels_arr,
        test_size=val_ratio,
        stratify=labels_arr,
        random_state=seed,
    )

    # Test: Ayrı klasörden oku
    test_dirs, test_labels = scan_dataset_from_folders(test_dir)

    print(f"\n[BİLGİ] Veri bölme tamamlandı:")
    print(f"  Train: {len(train_dirs)} hasta (kaynak: {root_dir})")
    print(f"  Val:   {len(val_dirs)} hasta (kaynak: {root_dir})")
    print(f"  Test:  {len(test_dirs)} hasta (kaynak: {test_dir})")

    return {
        "train": (train_dirs, train_labels.tolist()),
        "val": (val_dirs, val_labels.tolist()),
        "test": (test_dirs, test_labels),
    }


def create_dataloaders(
    config: dict,
) -> Dict[str, DataLoader]:
    """
    Config dosyasına göre train, val ve test DataLoader'larını oluşturur.

    Args:
        config: Parsed YAML konfigürasyon sözlüğü.

    Returns:
        dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Veri setini böl (train/val root_dir'den, test ayrı test_dir'den)
    splits = prepare_patient_split(
        root_dir=data_cfg["root_dir"],
        test_dir=data_cfg["test_dir"],
        train_ratio=data_cfg["split"]["train"],
        val_ratio=data_cfg["split"]["val"],
        seed=config["project"]["seed"],
    )

    # Transform pipeline'ları oluştur
    train_transform = get_train_transforms(data_cfg)
    val_transform = get_val_transforms(data_cfg)

    # Dataset nesneleri
    train_dataset = MammographyDataset(
        patient_dirs=splits["train"][0],
        labels=splits["train"][1],
        transform=train_transform,
    )
    val_dataset = MammographyDataset(
        patient_dirs=splits["val"][0],
        labels=splits["val"][1],
        transform=val_transform,
    )
    test_dataset = MammographyDataset(
        patient_dirs=splits["test"][0],
        labels=splits["test"][1],
        transform=val_transform,
    )

    # DataLoader'lar
    # Not: WeightedRandomSampler kaldırıldı. Sınıf dengesizliği yalnızca
    # loss fonksiyonundaki class weights ile ele alınır (sqrt-inverse frequency).
    # Sampler + class weights birlikte kullanıldığında çarpımsal aşırı telafi
    # oluşuyor ve az örnekli sınıfların ezberlenmesine yol açıyordu.
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
        ),
    }

    return dataloaders
