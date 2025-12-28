"""
CENG 391 Final Project - Full Pipeline
=======================================
Bu script:
1. Tüm DRIVE test görüntülerini işler
2. Table 1'deki metrikleri hesaplar (Sen, Spe, PPV, Acc)
3. Figure 3-12'deki ara görüntüleri kaydeder

Kullanım:
    python run_full_pipeline.py
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime

# Ana modülü import et
from main import *

# DRIVE veri seti yolu
DRIVE_PATH = "DRIVE"


def save_intermediate_figures(
    preprocessed, all_features, results, save_dir, img_name
):
    """
    Makale Figure 3-12'deki ara görüntüleri kaydet

    Figure 3: G, Y, L kanalları
    Figure 4: CLAHE sonrası G, Y, L kanalları
    Figure 5: Gabor filter çıktıları
    Figure 6: Binary Gabor görüntüleri
    Figure 7: Top Hat özelliği
    Figure 8: Shade Corrected özelliği
    Figure 9: Bit Plane Slicing
    Figure 10: FCM kümeleme sonucu
    Figure 11: FCM + Classification sonucu
    Figure 12: Final sonuç
    """
    os.makedirs(save_dir, exist_ok=True)

    # Figure 3 & 4: Kanallar (CLAHE sonrası)
    cv2.imwrite(
        os.path.join(save_dir, f"{img_name}_fig3_G_channel.png"),
        preprocessed["G"],
    )
    cv2.imwrite(
        os.path.join(save_dir, f"{img_name}_fig3_Y_channel.png"),
        preprocessed["Y"],
    )
    cv2.imwrite(
        os.path.join(save_dir, f"{img_name}_fig3_L_channel.png"),
        preprocessed["L"],
    )

    # Figure 5 & 6: Gabor özellikleri (ilk 9 özellik)
    wavelengths = [9, 10, 11]
    channels = ["G", "Y", "L"]
    for i, feature in enumerate(all_features[:9]):
        wl = wavelengths[i // 3]
        ch = channels[i % 3]
        cv2.imwrite(
            os.path.join(save_dir, f"{img_name}_fig5_gabor_{ch}_wl{wl}.png"),
            feature,
        )

    # Figure 7: Top Hat (index 10)
    if len(all_features) > 10:
        cv2.imwrite(
            os.path.join(save_dir, f"{img_name}_fig7_tophat.png"),
            all_features[10],
        )

    # Figure 8: Shade Corrected (index 11)
    if len(all_features) > 11:
        cv2.imwrite(
            os.path.join(save_dir, f"{img_name}_fig8_shade_corrected.png"),
            all_features[11],
        )

    # Figure 9: BPS (index 12)
    if len(all_features) > 12:
        cv2.imwrite(
            os.path.join(save_dir, f"{img_name}_fig9_bps.png"),
            all_features[12],
        )

    # Figure 10: FCM sonucu
    cv2.imwrite(
        os.path.join(save_dir, f"{img_name}_fig10_fcm_result.png"),
        results["fcm_result"],
    )

    # Figure 11: FCM + DT sonucu
    combined = np.maximum(results["fcm_result"], results["dt_result"])
    cv2.imwrite(
        os.path.join(save_dir, f"{img_name}_fig11_combined_result.png"),
        combined,
    )

    # Figure 12: Final sonuç
    cv2.imwrite(
        os.path.join(save_dir, f"{img_name}_fig12_final_result.png"),
        results["final_result"],
    )

    # Ground truth (karşılaştırma için)
    cv2.imwrite(
        os.path.join(save_dir, f"{img_name}_ground_truth.png"),
        results["ground_truth"],
    )

    # Input görüntü
    cv2.imwrite(
        os.path.join(save_dir, f"{img_name}_input.png"),
        preprocessed["cropped_image"],
    )


def process_dataset(dataset_type="training"):
    """
    Tüm veri setini işle

    dataset_type: 'training' veya 'test'
    """
    print("=" * 70)
    print(f"DRIVE {dataset_type.upper()} Veri Seti İşleniyor")
    print("=" * 70)

    # Klasör yolları
    images_dir = os.path.join(DRIVE_PATH, dataset_type, "images")
    mask_dir = os.path.join(DRIVE_PATH, dataset_type, "mask")

    if dataset_type == "training":
        gt_dir = os.path.join(DRIVE_PATH, dataset_type, "1st_manual")
    else:
        # Test seti için ground truth training klasöründe olabilir
        gt_dir = os.path.join(DRIVE_PATH, "test", "1st_manual")
        if not os.path.exists(gt_dir):
            gt_dir = os.path.join(DRIVE_PATH, "training", "1st_manual")

    output_dir = os.path.join(DRIVE_PATH, "output", dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    # Dosyaları listele
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith(".tif")]
    )

    print(f"\nToplam görüntü sayısı: {len(image_files)}")
    print("-" * 70)

    # Tüm metrikleri topla
    all_metrics = []
    classifier = None  # İlk görüntüde eğitilecek, sonra tekrar kullanılacak

    for idx, img_file in enumerate(image_files):
        img_num = img_file.split("_")[0]

        # Dosya yollarını bul
        img_path = os.path.join(images_dir, img_file)

        # Maske dosyasını bul
        mask_file = None
        for f in os.listdir(mask_dir):
            if f.startswith(img_num):
                mask_file = f
                break

        if mask_file is None:
            print(f"UYARI: {img_num} için maske bulunamadı, atlanıyor...")
            continue

        mask_path = os.path.join(mask_dir, mask_file)

        # Ground truth dosyasını bul
        gt_file = None
        if os.path.exists(gt_dir):
            for f in os.listdir(gt_dir):
                if f.startswith(img_num):
                    gt_file = f
                    break

        if gt_file is None:
            print(
                f"UYARI: {img_num} için ground truth bulunamadı, atlanıyor..."
            )
            continue

        gt_path = os.path.join(gt_dir, gt_file)

        print(f"\n[{idx+1}/{len(image_files)}] İşleniyor: {img_file}")

        try:
            # Görüntüyü işle
            metrics, classifier, results, preprocessed, all_features = (
                process_single_image_full(
                    img_path,
                    mask_path,
                    gt_path,
                    classifier=classifier,
                    save_dir=os.path.join(output_dir, img_num),
                )
            )

            all_metrics.append(metrics)

            # Ara görüntüleri kaydet
            save_intermediate_figures(
                preprocessed,
                all_features,
                results,
                os.path.join(output_dir, img_num),
                img_num,
            )

            print(
                f"    Sen: {metrics['Sensitivity']:.4f} | "
                f"Spe: {metrics['Specificity']:.4f} | "
                f"Acc: {metrics['Accuracy']:.4f} | "
                f"PPV: {metrics['PPV']:.4f}"
            )

        except Exception as e:
            print(f"    HATA: {e}")
            import traceback

            traceback.print_exc()

    return all_metrics, output_dir


def process_single_image_full(
    image_path, mask_path, gt_path, classifier=None, save_dir=None
):
    """
    Tek bir görüntüyü işle - tüm ara sonuçları da döndür
    """
    # 1. Yükle
    image = load_image(image_path)
    mask = load_mask(mask_path)
    ground_truth = load_ground_truth(gt_path)

    # 2. Ön işleme
    preprocessed = preprocess_image(image, mask)

    # Ground truth'u da kırp
    crop = preprocessed["crop_coords"]
    gt_cropped = ground_truth[
        crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2]
    ]

    # 3. Özellik çıkarma
    feature_vectors, pixel_coords, all_features = construct_feature_vector(
        preprocessed, preprocessed["mask"]
    )

    # 4. PCA
    pca_features, pca_model = apply_pca(feature_vectors)

    # 5. FCM Kümeleme
    fcm_labels, fcm_probs, vessel_cluster, fcm_vessel_mask = apply_fcm(
        pca_features, threshold=0.60
    )

    # FCM sonucunu görüntüye dönüştür
    h, w = preprocessed["mask"].shape
    fcm_result = np.zeros((h, w), dtype=np.uint8)
    for idx, (y, x) in enumerate(pixel_coords):
        if fcm_vessel_mask[idx]:
            fcm_result[y, x] = 255

    # 6. Karar Ağacı Sınıflandırma
    if classifier is None:
        X_train, y_train = prepare_training_data(
            pca_features,
            pixel_coords,
            gt_cropped,
            preprocessed["mask"],
            fcm_vessel_mask,
        )
        if len(X_train) > 0:
            classifier = train_decision_tree(X_train, y_train)

    # Sadece FCM'in arka plan dediği pikselleri sınıflandır
    dt_result = np.zeros((h, w), dtype=np.uint8)

    if classifier is not None:
        non_vessel_features = []
        non_vessel_coords = []

        for idx, (y, x) in enumerate(pixel_coords):
            if not fcm_vessel_mask[idx]:
                non_vessel_features.append(pca_features[idx])
                non_vessel_coords.append((y, x))

        if len(non_vessel_features) > 0:
            non_vessel_features = np.array(non_vessel_features)
            predictions_proba = classifier.predict_proba(non_vessel_features)
            threshold = 0.58
            predictions = (predictions_proba[:, 1] > threshold).astype(int)

            for idx, (y, x) in enumerate(non_vessel_coords):
                if predictions[idx] == 1:
                    dt_result[y, x] = 255

    # 7. FCM ve DT sonuçlarını birleştir
    combined_result = np.maximum(fcm_result, dt_result)

    # 8. Son işleme
    final_result = post_process(combined_result, preprocessed["mask"])

    # 9. Değerlendirme
    metrics = calculate_metrics(final_result, gt_cropped, preprocessed["mask"])

    results = {
        "fcm_result": fcm_result,
        "dt_result": dt_result,
        "final_result": final_result,
        "ground_truth": gt_cropped,
    }

    return metrics, classifier, results, preprocessed, all_features


def print_table1(all_metrics):
    """
    Table 1 formatında sonuçları yazdır
    """
    print("\n")
    print("=" * 70)
    print("TABLE 1: Performance Measures for DRIVE Database")
    print("=" * 70)

    # Ortalama hesapla
    avg_sen = np.mean([m["Sensitivity"] for m in all_metrics])
    avg_spe = np.mean([m["Specificity"] for m in all_metrics])
    avg_ppv = np.mean([m["PPV"] for m in all_metrics])
    avg_acc = np.mean([m["Accuracy"] for m in all_metrics])

    # Standart sapma
    std_sen = np.std([m["Sensitivity"] for m in all_metrics])
    std_spe = np.std([m["Specificity"] for m in all_metrics])
    std_ppv = np.std([m["PPV"] for m in all_metrics])
    std_acc = np.std([m["Accuracy"] for m in all_metrics])

    print(f"\n{'Method':<25} {'Sen':<12} {'Spe':<12} {'PPV':<12} {'Acc':<12}")
    print("-" * 70)
    print(
        f"{'Proposed Method':<25} {avg_sen:.4f}       {avg_spe:.4f}       {avg_ppv:.4f}       {avg_acc:.4f}"
    )
    print(
        f"{'(Std Dev)':<25} ({std_sen:.4f})     ({std_spe:.4f})     ({std_ppv:.4f})     ({std_acc:.4f})"
    )
    print("-" * 70)
    print(
        f"{'Paper Results':<25} {'0.7830':<12} {'0.9800':<12} {'0.8594':<12} {'0.9531':<12}"
    )
    print("=" * 70)

    return {
        "Sensitivity": avg_sen,
        "Specificity": avg_spe,
        "PPV": avg_ppv,
        "Accuracy": avg_acc,
    }


def save_results_to_file(all_metrics, avg_metrics, output_dir):
    """
    Sonuçları dosyaya kaydet
    """
    results_file = os.path.join(output_dir, "results.txt")

    with open(results_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CENG 391 Final Project - Retinal Blood Vessel Extraction\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("TABLE 1: Performance Measures for DRIVE Database\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Method':<25} {'Sen':<12} {'Spe':<12} {'PPV':<12} {'Acc':<12}\n"
        )
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Proposed Method':<25} {avg_metrics['Sensitivity']:.4f}       "
            f"{avg_metrics['Specificity']:.4f}       "
            f"{avg_metrics['PPV']:.4f}       "
            f"{avg_metrics['Accuracy']:.4f}\n"
        )
        f.write(
            f"{'Paper Results':<25} {'0.7830':<12} {'0.9800':<12} {'0.8594':<12} {'0.9531':<12}\n"
        )
        f.write("-" * 70 + "\n\n")

        f.write("Individual Image Results:\n")
        f.write("-" * 70 + "\n")
        for i, m in enumerate(all_metrics):
            f.write(
                f"Image {i+1:2d}: Sen={m['Sensitivity']:.4f} | "
                f"Spe={m['Specificity']:.4f} | "
                f"Acc={m['Accuracy']:.4f} | "
                f"PPV={m['PPV']:.4f}\n"
            )

    print(f"\nSonuçlar kaydedildi: {results_file}")


def main():
    """
    Ana fonksiyon
    """
    print("\n" + "=" * 70)
    print("CENG 391 Final Project")
    print("Retinal Blood Vessel Extraction")
    print("Hashemzadeh & Adlpour Azar (2019) Implementation")
    print("=" * 70)

    # Veri setini kontrol et
    if not os.path.exists(DRIVE_PATH):
        print(f"\nHATA: DRIVE klasörü bulunamadı: {DRIVE_PATH}")
        print("Lütfen DRIVE_PATH değişkenini kontrol edin.")
        return

    # Training setini işle
    print("\n" + "=" * 70)
    print("Training seti işleniyor...")
    print("=" * 70)

    all_metrics, output_dir = process_dataset("training")

    if all_metrics:
        # Table 1'i yazdır
        avg_metrics = print_table1(all_metrics)

        # Sonuçları dosyaya kaydet
        save_results_to_file(all_metrics, avg_metrics, output_dir)

        print("\n" + "=" * 70)
        print("İŞLEM TAMAMLANDI!")
        print("=" * 70)
        print(f"\nÇıktılar: {output_dir}")
        print("\nKaydedilen dosyalar:")
        print("  - Her görüntü için Figure 3-12 (ara görüntüler)")
        print("  - results.txt (tüm metrikler)")
    else:
        print("\nHATA: Hiçbir görüntü işlenemedi!")


if __name__ == "__main__":
    main()
