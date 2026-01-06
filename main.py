import cv2
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
import skfuzzy as fuzz
from tabulate import tabulate


def load_image(path, is_grayscale=False):
    if not os.path.exists(path):
        return None
    try:
        if path.lower().endswith(".gif"):
            img = Image.open(path).convert("L" if is_grayscale else "RGB")
            return np.array(img)
        img = cv2.imread(path)
        if img is None:
            return None
        return (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if is_grayscale
            else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
    except:
        return None


def custom_automatic_threshold(img_8u):
    """
    Gri tonlu görüntü için basit ortalama tabanli eşik değeri hesaplar.
    Hesaplama: T = sum(i * (Yi / size)) şeklinde her seviye için beklenen
    yoğunluk alinir. Küçük bir epsilon, sifira bölünmeyi önler.
    """
    hist = cv2.calcHist([img_8u], [0], None, [256], [0, 256]).flatten()
    total_pixels = img_8u.size + 1e-7
    return np.sum(np.arange(256) * (hist / total_pixels))


# --- AŞAMA 1: ÖN-İŞLEME ---
def pre_processing_stage(img, mask, output_path, save=False):
    # FOV (field-of-view) dışındaki bölgeleri maskeleyip sadece ilgi alanını al
    fov_only = cv2.bitwise_and(img, img, mask=(mask > 0).astype(np.uint8))

    # Kanallar: yeşil kanal, Y (luma) kanalı, Lab L kanalı — damar yapıları için faydalı
    g = fov_only[:, :, 1]
    y = cv2.cvtColor(fov_only, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    l = cv2.cvtColor(fov_only, cv2.COLOR_RGB2Lab)[:, :, 0]

    if save:
        # Figure-3
        cv2.imwrite(f"{output_path}/Fig3_Channels.png", np.hstack([g, y, l]))

    # Yerel kontrastı arttırmak için CLAHE uygula (her kanal için ayrı)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enh, y_enh, l_enh = clahe.apply(g), clahe.apply(y), clahe.apply(l)

    if save:
        # Figure-4
        cv2.imwrite(
            f"{output_path}/Fig4_CLAHE_Results.png",
            np.hstack([g_enh, y_enh, l_enh]),
        )
    return [g_enh, y_enh, l_enh]


# --- AŞAMA 2: ÖZELLİK ÇIKARIMI (13 Özellik) ---
def extract_13_features(channels, mask, output_path, save_figs=False):
    g_enh = channels[0]

    # features: her piksel için çıkarılan özelliklerin saklanacağı liste
    # gabor_res_grid, auto_thr_grid: görselleştirme için geçici saklama
    features, gabor_res_grid, auto_thr_grid = [], [], []
    mask_bin = (mask > 0).astype(np.uint8)  # FOV maskesinin ikili hali

    # Gabor filtreleri: farklı dalga boyları (wavelengths) ve açılar (angles)
    # Gabor, belirli frekanslarda ve açılardaki çizgisel/damar benzeri yapıları vurgular.
    wavelengths = [9, 10, 11]
    angles = np.arange(0, 180, 15)  # 0..165 arası 15 derecelik adımlarla
    aspect_ratio = 0.5

    # Her kanal (g_enh, y_enh, l_enh) için Gabor tabanlı özellikler çıkarılır
    for ch in channels:
        # kayan pencere işlemleri için float32 kullan
        ch_float = ch.astype(np.float32)
        row_res, row_thr = [], []  # görselleştirme satırları (her lambda için)

        for lambd in wavelengths:
            # Sigma, dalga boyuna göre ayarlanır; kernel boyutu sigma'ya bağlı
            sigma = 0.56 * lambd
            k_size = int(3 * sigma) * 2 + 1  # tek sayı kernel

            # scale_max: aynı boyuttaki bir matris, tüm açılardaki maksimum Gabor cevabını tutar
            scale_max = np.zeros_like(ch_float)

            # Her açı için Gabor kernel oluştur, filtre uygula, maksimumu al
            for theta_deg in angles:
                theta = np.deg2rad(theta_deg)
                kernel = cv2.getGaborKernel(
                    (k_size, k_size),
                    sigma,
                    theta,
                    lambd,
                    aspect_ratio,
                    0,
                    ktype=cv2.CV_32F,
                )

                # filtre uygula; borderType ile kenarlarda kopyalama yapıyoruz
                filtered = cv2.filter2D(
                    ch_float,
                    cv2.CV_32F,
                    kernel,
                    borderType=cv2.BORDER_REPLICATE,
                )

                # Farklı açılardaki cevabın piksel bazında maksimumunu al
                scale_max = np.maximum(scale_max, filtered)

            # scale_max artık bu lambda için tek bir özellik haritasıdır
            features.append(scale_max.flatten())

            if save_figs:
                # Görselleştirme: normalize et, laplace ile kenar vurgusu ekle
                norm_res = cv2.normalize(
                    scale_max, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                row_res.append(norm_res)

                # Laplacian ile kenar bilgisi alıyoruz, ardından ağırlıklı birleştirme
                laplace = cv2.Laplacian(
                    norm_res.astype(np.float64), cv2.CV_64F, ksize=3
                )
                v_enh = cv2.addWeighted(
                    norm_res, 0.75, cv2.convertScaleAbs(laplace), 0.25, 0
                )

                # Kenar vurgusundan otomatik eşik hesapla ve FOV ile kırp
                t_val = custom_automatic_threshold(v_enh)
                _, binary = cv2.threshold(v_enh, t_val, 255, cv2.THRESH_BINARY)
                row_thr.append(cv2.bitwise_and(binary, mask_bin * 255))

        # Her kanal için görselleştirme satırlarını topla
        if save_figs:
            gabor_res_grid.append(np.hstack(row_res))
            auto_thr_grid.append(np.hstack(row_thr))

    # Eğer görseller kaydedilecekse, grid'leri kaydet
    if save_figs:
        # Figure-5
        cv2.imwrite(
            f"{output_path}/Fig5_Gabor_Responses.png",
            np.vstack(gabor_res_grid),
        )
        # Figure-6
        cv2.imwrite(
            f"{output_path}/Fig6_Automatic_Thresholding.png",
            np.vstack(auto_thr_grid),
        )

    # Özellik 10: CLAHE uygulanmış yeşil kanalın pikselleri (daha sonra kullanılıyor)
    features.append(g_enh.flatten())

    # Özellik 11: Top-Hat / Blackhat morfolojik işlemler
    # Amaç: ince çizgisel yapıların (damar benzeri) vurgulanması
    th_max = np.zeros_like(g_enh)
    for angle in np.arange(0, 180, 22.5):
        # Çizgisel yapı için küçük çizgisel kernel oluştur ve döndür
        kernel = np.zeros((15, 15), dtype=np.uint8)
        cv2.line(kernel, (0, 7), (14, 7), 1, 1)
        rot_k = cv2.warpAffine(
            kernel, cv2.getRotationMatrix2D((7, 7), angle, 1), (15, 15)
        )

        # Blackhat (morfo) ile koyu yapıların (çizgi) arka plandan farkını al
        th_max = cv2.max(
            th_max, cv2.morphologyEx(g_enh, cv2.MORPH_BLACKHAT, rot_k)
        )

    features.append(
        cv2.normalize(th_max, None, 0, 255, cv2.NORM_MINMAX).flatten()
    )
    if save_figs:
        # Figure-7
        cv2.imwrite(
            f"{output_path}/Fig7_TopHat.png",
            cv2.normalize(th_max, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            ),
        )

    # Özellik 12: Gölgeleme düzeltmesi — homojen olmayan aydınlanmayı azaltmak için
    # Büyük bir median blur alıp orijinalden çıkarma + offset ile arkaplanı normalize ediyoruz
    sc = cv2.addWeighted(cv2.medianBlur(g_enh, 25), 1.0, g_enh, -1.0, 128)
    features.append(cv2.normalize(sc, None, 0, 255, cv2.NORM_MINMAX).flatten())
    if save_figs:
        # Figure-8
        cv2.imwrite(
            f"{output_path}/Fig8_ShadeCorrection.png",
            cv2.normalize(sc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        )

    # Özellik 13: Bit Plane Slicing (BPS) — düşük seviyeli bit düzeylerindeki desenleri kullan
    bit_planes = []
    _, fov_mask = cv2.threshold(g_enh, 10, 255, cv2.THRESH_BINARY)
    for i in range(8):
        mask_val = 1 << i
        plane = ((g_enh & mask_val) > 0).astype(np.uint8) * 255
        # Eğer bit düzeyi FOV içinde genel olarak parlaksa, tersleyip FOV ile kırp
        if cv2.mean(plane, mask=fov_mask)[0] > 127:
            plane = cv2.bitwise_and(cv2.bitwise_not(plane), fov_mask)
        bit_planes.append(plane)

    # BPS için yüksek anlamlı iki bit düzeyini birleştir (ör: 6 ve 7)
    bps_feature = cv2.bitwise_or(bit_planes[6], bit_planes[7])
    features.append(bps_feature.flatten())
    if save_figs:
        # Figure-9
        panel = [g_enh] + bit_planes + [bps_feature]
        cv2.imwrite(
            f"{output_path}/Fig9_BPS_Panel.png",
            np.vstack([np.hstack(panel[:5]), np.hstack(panel[5:])]),
        )

    # Sonuç: özellik matrisini döndür (N_pixels x N_features)
    return np.array(features).T


# --- AŞAMA 3: SON İŞLEME ---
def post_processing_stage(binary_img, mask):
    """Gürültü giderme ve FOV (görüş alanı) maskesinin uygulanması.
    - Küçük gürültü bileşenlerini kaldırmak için morfolojik açma uygular.
    - Bağlı bileşenleri büyüklüğe göre filtreleyip küçük parçaları atar.
    - Sonuçta FOV içinde kalan final ikili haritayı döndürür.
    """
    m_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, m_kernel)
    nb, output, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    final = np.zeros(output.shape, dtype=np.uint8)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= 40:
            final[output == i] = 255
    kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel_mask, iterations=2)
    return cv2.bitwise_and(
        final, final, mask=(eroded_mask > 0).astype(np.uint8)
    )


if __name__ == "__main__":
    BASE = "DRIVE"
    MAIN_OUT = "output_results"
    if not os.path.exists(MAIN_OUT):
        os.makedirs(MAIN_OUT)
    # Amaç: özellikler arasındaki lineer korelasyonu azaltma
    pca = PCA(n_components=13)
    clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=20, random_state=42
    )

    # --- FAZ 1: EĞİTİM (training görüntüleri) ---
    print("--- Phase 1: Training Started ---")
    train_files = sorted(
        [
            f
            for f in os.listdir(os.path.join(BASE, "training", "images"))
            if f.endswith(".tif")
        ]
    )
    X_train_list, y_train_list = [], []
    for f in train_files[:15]:
        img_id = f.split("_")[0]
        img = load_image(os.path.join(BASE, "training", "images", f))
        mask = load_image(
            os.path.join(
                BASE, "training", "mask", f"{img_id}_training_mask.gif"
            ),
            True,
        )
        gt = load_image(
            os.path.join(
                BASE, "training", "1st_manual", f"{img_id}_manual1.gif"
            ),
            True,
        )
        if img is not None:
            feats = extract_13_features(
                pre_processing_stage(img, mask, MAIN_OUT),
                mask,
                MAIN_OUT,
            )
            v_idx, b_idx = (
                np.where(gt.flatten() > 0)[0],
                np.where(gt.flatten() == 0)[0],
            )
            # Rastgele örnekleme: pozitif ve negatiften eşit sayıda örnek al
            s_idx = np.concatenate(
                [np.random.choice(v_idx, 8000), np.random.choice(b_idx, 8000)]
            )
            X_train_list.append(feats[s_idx])
            y_train_list.append((gt.flatten() > 0)[s_idx])
    # PCA yalnızca eğitim verisi üzerinde fit edilir (sızıntıyı önlemek için).
    # Burada tüm eğitim örnekleri (rastgele örneklenmiş pikseller) birleştirilir
    # ve PCA'nin dönüşüm matrisi öğrenilir.
    pca.fit(np.vstack(X_train_list))

    # Daha sonra eğitim verisi PCA uzayına dönüştürülüp sınıflandırıcı bu projeksiyon
    # üzerinde eğitilir. Bu sayede sınıflandırıcı korelasyonlu ham özellikler yerine
    # ana bileşenleri (daha 'düzenli' özellikler) kullanır.
    clf.fit(pca.transform(np.vstack(X_train_list)), np.hstack(y_train_list))

    # --- FAZ 2: PERFORMANS ÖLÇÜTLERİ (Eğitim seti doğrulama) ---
    print("\n--- Phase 2: Validation on remaining Training Images ---")
    val_results = []
    for f in train_files[15:]:
        img_id = f.split("_")[0]
        img = load_image(os.path.join(BASE, "training", "images", f))
        mask = load_image(
            os.path.join(
                BASE, "training", "mask", f"{img_id}_training_mask.gif"
            ),
            True,
        )
        gt = load_image(
            os.path.join(
                BASE, "training", "1st_manual", f"{img_id}_manual1.gif"
            ),
            True,
        )
        # Use the PCA fitted on training set to project validation features.
        X_p = pca.transform(
            extract_13_features(
                pre_processing_stage(img, mask, MAIN_OUT),
                mask,
                MAIN_OUT,
            )
        )
        cntr, u, *_ = fuzz.cluster.cmeans(
            X_p.T, 2, 2, error=1e-6, maxiter=2000
        )
        # FCM outputs: `cntr`=centroids, `u`=membership matrix
        # Select vessel cluster by centroid mean
        fcm_idx = np.argmax(np.mean(cntr, axis=1))

        # Combine FCM membership and classifier probability (hybrid score)
        hybrid_prob = 0.4 * u[fcm_idx] + 0.6 * clf.predict_proba(X_p)[:, 1]

        # Threshold hybrid score and apply post-processing to clean result
        res = post_processing_stage(
            (hybrid_prob.reshape(img.shape[:2]) > 0.58).astype(np.uint8) * 255,
            mask,
        )
        tn, fp, fn, tp = confusion_matrix(
            (gt[mask > 0] > 0).flatten(), (res[mask > 0] > 0).flatten()
        ).ravel()
        val_results.append(
            [
                tp / (tp + fn + 1e-7),
                tn / (tn + fp + 1e-7),
                tp / (tp + fp + 1e-7),
                (tp + tn) / (tp + tn + fp + fn),
                roc_auc_score(
                    (gt[mask > 0] > 0).flatten(),
                    hybrid_prob[mask.flatten() > 0],
                ),
            ]
        )
    print(
        tabulate(
            [
                [
                    "DRIVE-Train",
                    "Proposed",
                    *[f"{v:.4f}" for v in np.mean(val_results, axis=0)],
                ]
            ],
            headers=["DB", "Method", "Sen", "Spe", "PPV", "ACC", "AUC"],
            tablefmt="grid",
        )
    )

    # --- FAZ 3: TEST SETİ İŞLEME (1st_manual referans kullanılarak) ---
    print("\n--- Phase 3: Processing Official Test Set with Ground Truth ---")
    test_results = []
    test_files = sorted(
        [
            f
            for f in os.listdir(os.path.join(BASE, "test", "images"))
            if f.endswith(".tif")
        ]
    )

    for f in test_files:
        img_id = f.split("_")[0]
        sub = os.path.join(MAIN_OUT, img_id)
        if not os.path.exists(sub):
            os.makedirs(sub)

        img = load_image(os.path.join(BASE, "test", "images", f))
        mask = load_image(
            os.path.join(BASE, "test", "mask", f"{img_id}_test_mask.gif"), True
        )
        gt = load_image(
            os.path.join(BASE, "test", "1st_manual", f"{img_id}_manual1.gif"),
            True,
        )

        if gt is not None:
            # Step 3.3.1: Feature Extraction
            # Project test features into the PCA space learned from training.
            # This ensures the clustering/classifier operate on the same
            # transformed feature space as during training/validation.
            X_p_test = pca.transform(
                extract_13_features(
                    pre_processing_stage(img, mask, sub, save=True),
                    mask,
                    sub,
                    save_figs=True,
                )
            )

            # Adım: Gözetimsiz kümeleme (FCM)
            cntr, u, *_ = fuzz.cluster.cmeans(
                X_p_test.T, 2, 2, error=1e-6, maxiter=2000
            )
            # FCM outputs: `cntr`=centroids, `u`=membership matrix
            # Select vessel cluster by centroid mean
            fcm_idx = np.argmax(np.mean(cntr, axis=1))

            # fcm_prob: seçilen kümenin her için üyelik değeri (0..1 arası),
            # reshape ile görüntü boyutuna döndürülür.
            fcm_prob = u[fcm_idx].reshape(img.shape[:2])

            # Basit eşikleme ile FCM tabanlı ikili harita elde edilir.
            fcm_map = (fcm_prob > 0.60).astype(np.uint8) * 255

            # Figure-10
            # --- FIG 10 KAYDI ---
            cv2.imwrite(f"{sub}/Fig10_InitialClustering.png", fcm_map)

            # Adım: Gözetimli sınıflandırma ve hibrit olasılık birleşimi
            dt_prob = clf.predict_proba(X_p_test)[:, 1].reshape(img.shape[:2])
            # Hibrit birleşim: FCM olasılıkları ile sınıflandırıcı olasılıklarının ağırlıklı ortalaması
            hybrid_prob_flat = (
                0.4 * u[fcm_idx] + 0.6 * clf.predict_proba(X_p_test)[:, 1]
            )
            hybrid_map = (
                hybrid_prob_flat.reshape(img.shape[:2]) > 0.58
            ).astype(np.uint8) * 255

            # Figure-11
            # --- FIG 11 KAYDI (Kümeleme ve Sınıflandırma birleşimi) ---
            # Makaledeki gibi karşılaştırmalı görmek için yan yana eklenebilir
            cv2.imwrite(f"{sub}/Fig11_HybridOutput.png", hybrid_map)

            # Step 3.4: Post-Processing
            final_v = post_processing_stage(hybrid_map, mask)

            # Figure-12
            # FIG 12 KAYDI
            cv2.imwrite(f"{sub}/Fig12_FinalResult.png", final_v)

            # Metrik Hesaplamaları
            y_true = (gt[mask > 0] > 0).flatten()
            y_pred = (final_v[mask > 0] > 0).flatten()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            sen, spe = tp / (tp + fn + 1e-7), tn / (tn + fp + 1e-7)
            ppv, acc = tp / (tp + fp + 1e-7), (tp + tn) / (tp + tn + fp + fn)
            auc = roc_auc_score(y_true, hybrid_prob_flat[mask.flatten() > 0])

            test_results.append([img_id, sen, spe, ppv, acc, auc])
            print(f" [+] Image {img_id} processed | ACC: {acc:.4f}")
