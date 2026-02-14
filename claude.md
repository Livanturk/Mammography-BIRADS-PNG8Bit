# 📑 Proje Teknik Dökümantasyonu: Mamografi BI-RADS Tahmini

Bu döküman, projenin veri yapısını, istatistiksel özelliklerini ve modelleme kısıtlarını Claude'a tanıtmak amacıyla oluşturulmuştur. **Claude, tüm önerilerini bu bağlama dayandırmalıdır.**

## 1. Genel Yapı ve Görüntü Özellikleri

| Özellik | Değer |
| --- | --- |
| **Görüntü Formatı** | 8-bit PNG, Grayscale |
| **Çözünürlük** | 384 × 384 piksel |
| **Kaynak** | DICOM → 8-bit PNG dönüşümü (windowing uygulanmış) |
| **View Sayısı** | 4 (RCC, LCC, RMLO, LMLO) |
| **Birim (Unit)** | **Hasta bazlı** (1 hasta = 1 klasör = 4 görüntü) |
| **Sınıf Sayısı** | 4 (BI-RADS 1, 2, 4, 5) — **BI-RADS 3 yoktur.** |

## 2. Veri Dağılımı ve Split Stratejisi

Veri seti, test aşamasında objektif bir değerlendirme için dengelenmiştir.

### Sınıf Dağılım Tablosu

| Sınıf | Klinik Anlam | Train/Val | Test | Toplam |
| --- | --- | --- | --- | --- |
| **BI-RADS 1** | Negatif (Normal) | 1,428 | 250 | 1,678 |
| **BI-RADS 2** | Benign (İyi huylu) | 2,504 | 250 | 2,754 |
| **BI-RADS 4** | Şüpheli Malignite | 1,648 | 250 | 1,898 |
| **BI-RADS 5** | Yüksek Olasılıklı | 1,977 | 250 | 2,227 |
| **Toplam** |  | **7,557** | **1,000** | **8,557** |

### Split Detayları

* **Train (%85):** ~6,423 görüntü. Stratified random split (seed=42).
* **Val (%15):** ~1,134 görüntü. Stratified random split (seed=42).
* **Test (Sabit):** 1,000 görüntü. Bağımsız holdout (Her sınıftan tam 250 adet).

---

## 3. Piksel İstatistikleri ve Yoğunluk Analizi

* **Değer Aralığı:** [0, 255]
* **Global Ort / Std:** 21.77 (0.0854) / 37.13 (0.1456)
* **Sıfır Piksel Oranı:** %66 (Arka plan)
* **95. - 99. Yüzdelik:** 104 - 145

### Sınıflar Arası Farklar

| Sınıf | Ortalama Piksel | Sıfır Oranı | Yorum |
| --- | --- | --- | --- |
| BI-RADS 1 | ~18 | %69 | En az yoğun doku |
| BI-RADS 2 | ~20 | %66 | Düşük yoğunluk |
| BI-RADS 4 | ~24 | %65 | Daha yoğun doku |
| BI-RADS 5 | ~25 | %63 | En yoğun doku |

> ** Kritik Not:** Malign sınıflarda doku yoğunluğu (parlaklık) daha yüksektir. Modelin morfolojik özellikleri öğrenmek yerine parlaklığı bir "kısayol" (shortcut) olarak öğrenme riski mevcuttur.

---

## 4. Eğitim Metodolojisi

* **Dengeleme:** `Sqrt-inverse frequency class weights` kullanılmaktadır.
* **Mevcut Ağırlıklar:** `[1.32, 1.0, 1.23, 1.13]`
* **Preprocessing:** Histogram normalizasyonu veya maskeleme uygulanmamıştır. 8-bit dönüşümü sabittir.

### Histogram Dağılımı (Kuyruk Analizi)

```text
[  0- 16]: ████████████████████ (%67 — arka plan)
[ 16-145]: █████                (%31 — doku bilgisi)
[145-255]: ▏                    (%2  — uzun kuyruk)

```

---

## 5. Claude İçin Operasyonel Kurallar
4. **Anomali:** Test setinin dengeli olması sebebiyle Test F1 > Val F1 durumunun normal olduğunu unutma.
