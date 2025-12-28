"""
Test Script - Tek görüntü üzerinde test
"""

import os
import sys

# DRIVE veri seti yolu - KENDİ YOLUNU YAZ
DRIVE_PATH = "DRIVE"
# Ana modülü import et
from main import *


def test_single_image():
    """Tek bir görüntü üzerinde test yap"""

    print("=" * 60)
    print("DRIVE Veri Seti Kontrolü")
    print("=" * 60)

    # Klasörleri kontrol et
    training_images = os.path.join(DRIVE_PATH, "training", "images")
    training_manual = os.path.join(DRIVE_PATH, "training", "1st_manual")
    training_mask = os.path.join(DRIVE_PATH, "training", "mask")

    print(f"\nKlasör kontrolleri:")
    print(
        f"  training/images: {'✓' if os.path.exists(training_images) else '✗'}"
    )
    print(
        f"  training/1st_manual: {'✓' if os.path.exists(training_manual) else '✗'}"
    )
    print(f"  training/mask: {'✓' if os.path.exists(training_mask) else '✗'}")

    if not all(
        [
            os.path.exists(p)
            for p in [training_images, training_manual, training_mask]
        ]
    ):
        print("\nHATA: Bazı klasörler bulunamadı!")
        print("Lütfen DRIVE_PATH değişkenini kontrol et.")
        return

    # Dosyaları listele
    image_files = sorted(os.listdir(training_images))
    manual_files = sorted(os.listdir(training_manual))
    mask_files = sorted(os.listdir(training_mask))

    print(f"\nDosya sayıları:")
    print(f"  Görüntüler: {len(image_files)}")
    print(f"  Ground Truth: {len(manual_files)}")
    print(f"  Maskeler: {len(mask_files)}")

    print(f"\nÖrnek dosya isimleri:")
    print(f"  Görüntü: {image_files[0] if image_files else 'YOK'}")
    print(f"  Ground Truth: {manual_files[0] if manual_files else 'YOK'}")
    print(f"  Maske: {mask_files[0] if mask_files else 'YOK'}")

    # İlk görüntüyü işle
    if image_files:
        print("\n" + "=" * 60)
        print("İlk görüntü işleniyor...")
        print("=" * 60)

        # Dosya isimlerinden numara çıkar
        img_file = image_files[0]
        img_num = img_file.split("_")[0]

        # Dosya yolları
        img_path = os.path.join(training_images, img_file)

        # Maske ve GT dosya isimlerini bul
        mask_file = None
        gt_file = None

        for f in mask_files:
            if f.startswith(img_num):
                mask_file = f
                break

        for f in manual_files:
            if f.startswith(img_num):
                gt_file = f
                break

        if mask_file and gt_file:
            mask_path = os.path.join(training_mask, mask_file)
            gt_path = os.path.join(training_manual, gt_file)

            print(f"\nDosyalar:")
            print(f"  Görüntü: {img_path}")
            print(f"  Maske: {mask_path}")
            print(f"  Ground Truth: {gt_path}")

            # Çıktı klasörü
            output_dir = os.path.join(DRIVE_PATH, "output", img_num)

            # İşle
            try:
                metrics, classifier, results = process_single_image(
                    img_path, mask_path, gt_path, save_dir=output_dir
                )

                print("\n" + "=" * 60)
                print("TEST BAŞARILI!")
                print("=" * 60)
                print(f"\nSonuçlar:")
                print(f"  Sensitivity: {metrics['Sensitivity']:.4f}")
                print(f"  Specificity: {metrics['Specificity']:.4f}")
                print(f"  Accuracy: {metrics['Accuracy']:.4f}")
                print(f"  PPV: {metrics['PPV']:.4f}")
                print(f"\nÇıktılar kaydedildi: {output_dir}")

            except Exception as e:
                print(f"\nHATA: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"\nHATA: Maske veya GT dosyası bulunamadı!")
            print(f"  Aranan maske: {img_num}_*")
            print(f"  Aranan GT: {img_num}_*")


if __name__ == "__main__":
    test_single_image()
