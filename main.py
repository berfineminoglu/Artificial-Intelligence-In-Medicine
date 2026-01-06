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


# --- PAPER STEP 3.3.1.1: AUTOMATIC THRESHOLDING ---
def custom_automatic_threshold(img_8u):
    """T = sum(i * (Yi / size)) [cite: 289-291]"""
    hist = cv2.calcHist([img_8u], [0], None, [256], [0, 256]).flatten()
    total_pixels = img_8u.size + 1e-7
    return np.sum(np.arange(256) * (hist / total_pixels))


# --- STAGE 1: PRE-PROCESSING (Step 3.1 & 3.2) ---
def pre_processing_stage(img, mask, img_id, output_path, save=False):
    fov_only = cv2.bitwise_and(img, img, mask=(mask > 0).astype(np.uint8))
    g = fov_only[:, :, 1]
    y = cv2.cvtColor(fov_only, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    l = cv2.cvtColor(fov_only, cv2.COLOR_RGB2Lab)[:, :, 0]

    if save:
        cv2.imwrite(f"{output_path}/Fig3_Channels.png", np.hstack([g, y, l]))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enh, y_enh, l_enh = clahe.apply(g), clahe.apply(y), clahe.apply(l)

    if save:
        cv2.imwrite(
            f"{output_path}/Fig4_CLAHE_Results.png",
            np.hstack([g_enh, y_enh, l_enh]),
        )
    return [g_enh, y_enh, l_enh]


# --- STAGE 2: FEATURE EXTRACTION (13 Features) ---
def extract_13_features(channels, mask, img_id, output_path, save_figs=False):
    g_enh = channels[0]
    features, gabor_res_grid, auto_thr_grid = [], [], []
    mask_bin = (mask > 0).astype(np.uint8)

    wavelengths = [9, 10, 11]  # [cite: 275]
    angles = np.arange(0, 180, 15)
    aspect_ratio = 0.5

    for ch in channels:
        ch_float = ch.astype(np.float32)
        row_res, row_thr = [], []
        for lambd in wavelengths:
            sigma = 0.56 * lambd
            k_size = int(3 * sigma) * 2 + 1
            scale_max = np.zeros_like(ch_float)
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
                filtered = cv2.filter2D(
                    ch_float,
                    cv2.CV_32F,
                    kernel,
                    borderType=cv2.BORDER_REPLICATE,
                )
                scale_max = np.maximum(scale_max, filtered)

            features.append(scale_max.flatten())
            if save_figs:
                norm_res = cv2.normalize(
                    scale_max, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                row_res.append(norm_res)
                laplace = cv2.Laplacian(
                    norm_res.astype(np.float64), cv2.CV_64F, ksize=3
                )
                v_enh = cv2.addWeighted(
                    norm_res, 0.75, cv2.convertScaleAbs(laplace), 0.25, 0
                )  # [cite: 283-284]
                t_val = custom_automatic_threshold(v_enh)
                _, binary = cv2.threshold(v_enh, t_val, 255, cv2.THRESH_BINARY)
                row_thr.append(cv2.bitwise_and(binary, mask_bin * 255))

        if save_figs:
            gabor_res_grid.append(np.hstack(row_res))
            auto_thr_grid.append(np.hstack(row_thr))

    if save_figs:
        cv2.imwrite(
            f"{output_path}/Fig5_Gabor_Responses.png",
            np.vstack(gabor_res_grid),
        )
        cv2.imwrite(
            f"{output_path}/Fig6_Automatic_Thresholding.png",
            np.vstack(auto_thr_grid),
        )

    features.append(g_enh.flatten())  # Feature 10 [cite: 264]

    # Feature 11: Top-Hat (TH) [cite: 296-316]
    th_max = np.zeros_like(g_enh)
    for angle in np.arange(0, 180, 22.5):
        kernel = np.zeros((15, 15), dtype=np.uint8)
        cv2.line(kernel, (0, 7), (14, 7), 1, 1)
        rot_k = cv2.warpAffine(
            kernel, cv2.getRotationMatrix2D((7, 7), angle, 1), (15, 15)
        )
        th_max = cv2.max(
            th_max, cv2.morphologyEx(g_enh, cv2.MORPH_BLACKHAT, rot_k)
        )
    features.append(
        cv2.normalize(th_max, None, 0, 255, cv2.NORM_MINMAX).flatten()
    )
    if save_figs:
        cv2.imwrite(
            f"{output_path}/Fig7_TopHat.png",
            cv2.normalize(th_max, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            ),
        )

    # Feature 12: Shade Correction (SC) [cite: 319-327]
    sc = cv2.addWeighted(cv2.medianBlur(g_enh, 25), 1.0, g_enh, -1.0, 128)
    features.append(cv2.normalize(sc, None, 0, 255, cv2.NORM_MINMAX).flatten())
    if save_figs:
        cv2.imwrite(
            f"{output_path}/Fig8_ShadeCorrection.png",
            cv2.normalize(sc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        )

    # Feature 13: Bit Plane Slicing (BPS) [cite: 328-334]
    bit_planes = []
    _, fov_mask = cv2.threshold(g_enh, 10, 255, cv2.THRESH_BINARY)
    for i in range(8):
        mask_val = 1 << i
        plane = ((g_enh & mask_val) > 0).astype(np.uint8) * 255
        if cv2.mean(plane, mask=fov_mask)[0] > 127:
            plane = cv2.bitwise_and(cv2.bitwise_not(plane), fov_mask)
        bit_planes.append(plane)
    bps_feature = cv2.bitwise_or(bit_planes[6], bit_planes[7])
    features.append(bps_feature.flatten())
    if save_figs:
        panel = [g_enh] + bit_planes + [bps_feature]
        cv2.imwrite(
            f"{output_path}/Fig9_BPS_Panel.png",
            np.vstack([np.hstack(panel[:5]), np.hstack(panel[5:])]),
        )

    return np.array(features).T


# --- STAGE 3: POST-PROCESSING ---
def post_processing_stage(binary_img, mask):
    """Step 3.4: Noise removal and FOV Masking [cite: 411-416]"""
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
    BASE = "DRIVE"  # Klasör yapını 'DRIVE' olarak ayarla
    MAIN_OUT = "output_results"
    if not os.path.exists(MAIN_OUT):
        os.makedirs(MAIN_OUT)
    pca = PCA(n_components=13)
    clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=20, random_state=42
    )

    # --- PHASE 1: TRAINING (Images 21-35) ---
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
                pre_processing_stage(img, mask, img_id, MAIN_OUT),
                mask,
                img_id,
                MAIN_OUT,
            )
            v_idx, b_idx = (
                np.where(gt.flatten() > 0)[0],
                np.where(gt.flatten() == 0)[0],
            )
            # [cite: 437]
            s_idx = np.concatenate(
                [np.random.choice(v_idx, 8000), np.random.choice(b_idx, 8000)]
            )
            X_train_list.append(feats[s_idx])
            y_train_list.append((gt.flatten() > 0)[s_idx])
    pca.fit(np.vstack(X_train_list))
    clf.fit(pca.transform(np.vstack(X_train_list)), np.hstack(y_train_list))

    # --- PHASE 2: PERFORMANCE METRICS (Training Set Validation) ---
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
        X_p = pca.transform(
            extract_13_features(
                pre_processing_stage(img, mask, img_id, MAIN_OUT),
                mask,
                img_id,
                MAIN_OUT,
            )
        )
        cntr, u, *_ = fuzz.cluster.cmeans(
            X_p.T, 2, 2, error=1e-6, maxiter=2000
        )
        fcm_idx = np.argmax(np.mean(cntr, axis=1))
        hybrid_prob = 0.4 * u[fcm_idx] + 0.6 * clf.predict_proba(X_p)[:, 1]
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
                    "DRIVE-VAL",
                    "Proposed",
                    *[f"{v:.4f}" for v in np.mean(val_results, axis=0)],
                ]
            ],
            headers=["DB", "Method", "Sen", "Spe", "PPV", "ACC", "AUC"],
            tablefmt="grid",
        )
    )

    # --- PHASE 3: TEST SET PERFORMANCE (USING 1st_manual Ground Truth) ---
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
            X_p_test = pca.transform(
                extract_13_features(
                    pre_processing_stage(img, mask, img_id, sub, save=True),
                    mask,
                    img_id,
                    sub,
                    save_figs=True,
                )
            )

            # Step 3.3.3: Unsupervised Clustering (FCM)
            cntr, u, *_ = fuzz.cluster.cmeans(
                X_p_test.T, 2, 2, error=1e-6, maxiter=2000
            )
            fcm_idx = np.argmax(
                np.mean(cntr, axis=1)
            )  # Damar kümesini belirle [cite: 392]
            fcm_prob = u[fcm_idx].reshape(img.shape[:2])
            fcm_map = (fcm_prob > 0.60).astype(
                np.uint8
            ) * 255  # Eşikleme [cite: 387]

            # --- FIG 10 KAYDI ---
            cv2.imwrite(f"{sub}/Fig10_InitialClustering.png", fcm_map)

            # Step 3.3.4: Supervised Classification & Hybrid Combination
            dt_prob = clf.predict_proba(X_p_test)[:, 1].reshape(img.shape[:2])
            # Makaledeki hibrit birleştirme mantığı [cite: 408]
            hybrid_prob_flat = (
                0.4 * u[fcm_idx] + 0.6 * clf.predict_proba(X_p_test)[:, 1]
            )
            hybrid_map = (
                hybrid_prob_flat.reshape(img.shape[:2]) > 0.58
            ).astype(np.uint8) * 255

            # --- FIG 11 KAYDI (Kümeleme ve Sınıflandırma birleşimi) ---
            # Makaledeki gibi karşılaştırmalı görmek için yan yana eklenebilir
            cv2.imwrite(f"{sub}/Fig11_HybridOutput.png", hybrid_map)

            # Step 3.4: Post-Processing
            final_v = post_processing_stage(hybrid_map, mask)

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
