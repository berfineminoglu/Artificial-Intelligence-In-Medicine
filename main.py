"""
Retinal Blood Vessel Extraction - Compact Version
Hashemzadeh & Adlpour Azar (2019) Implementation
CENG 391 Final Project
"""

import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# ==================== PREPROCESSING ====================


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    return img


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def load_ground_truth(path):
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    return gt


def preprocess(image, mask):
    # Crop to FOV
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    img_crop = image[y : y + h, x : x + w]
    mask_crop = mask[y : y + h, x : x + w]

    # Extract G, Y, L channels
    G = img_crop[:, :, 1]
    Y = cv2.cvtColor(img_crop, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    L = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)[:, :, 0]

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    G_enh = clahe.apply(G)
    Y_enh = clahe.apply(Y)
    L_enh = clahe.apply(L)

    return {
        "G": G_enh,
        "Y": Y_enh,
        "L": L_enh,
        "mask": mask_crop,
        "crop": (x, y, w, h),
        "image": img_crop,
    }


# ==================== FEATURE EXTRACTION ====================


def gabor_filter(image, wavelength):
    """Apply Gabor filter at multiple orientations"""
    result = np.zeros_like(image, dtype=np.float64)
    for theta in np.arange(0, np.pi, np.pi / 12):
        kernel = cv2.getGaborKernel(
            (21, 21), 4.0, theta, wavelength, 0.5, 0, ktype=cv2.CV_64F
        )
        filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
        result = np.maximum(result, np.abs(filtered))
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )


def top_hat(image):
    """Multi-directional top-hat transform"""
    inv = 255 - image
    result = np.zeros_like(inv, dtype=np.float64)
    for angle in range(0, 180, 15):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
        M = cv2.getRotationMatrix2D((10, 0), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (21, 21))
        opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
        result = np.maximum(result, inv.astype(np.float64) - opened)
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )


def shade_correct(image):
    """Remove uneven illumination"""
    bg = cv2.medianBlur(image, 25)
    sc = cv2.subtract(image, bg)
    return cv2.normalize(sc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def bit_plane_slice(image):
    """Extract bit planes 7 and 8"""
    return ((image >> 6) & 1) * 127 + ((image >> 7) & 1) * 127


def extract_features(prep, mask):
    """Extract 13-dimensional feature vector for each pixel"""
    G, Y, L = prep["G"], prep["Y"], prep["L"]

    # 9 Gabor features (3 channels x 3 wavelengths)
    features = []
    for ch in [G, Y, L]:
        for wl in [8, 10, 12]:
            features.append(gabor_filter(ch, wl))

    # G channel, Top-hat, Shade-corrected, BPS
    features.extend([G, top_hat(G), shade_correct(G), bit_plane_slice(G)])

    # Build feature vectors for FOV pixels only
    h, w = mask.shape
    vectors, coords = [], []
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                vectors.append([float(f[y, x]) for f in features])
                coords.append((y, x))

    return np.array(vectors), coords, features


# ==================== CLUSTERING & CLASSIFICATION ====================


def kmeans_segment(features, threshold=0.5):
    """K-Means clustering for initial vessel segmentation"""
    # Normalize features
    feat_norm = (features - features.mean(axis=0)) / (
        features.std(axis=0) + 1e-8
    )

    # K-Means with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feat_norm)

    # Vessel cluster has fewer pixels
    if np.sum(labels == 0) < np.sum(labels == 1):
        vessel_label = 0
    else:
        vessel_label = 1

    # Calculate distances to cluster centers for confidence
    distances = kmeans.transform(feat_norm)
    vessel_dist = distances[:, vessel_label]
    other_dist = distances[:, 1 - vessel_label]

    # Confidence score
    confidence = other_dist / (vessel_dist + other_dist + 1e-8)
    vessel_mask = confidence > threshold

    return vessel_mask, confidence


def train_classifier(X, y):
    """Train Random Forest classifier"""
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def prepare_training_data(features, coords, gt, kmeans_mask, n_samples=20000):
    """Prepare balanced training data from non-vessel pixels"""
    vessel_idx, non_vessel_idx = [], []

    for i, (y, x) in enumerate(coords):
        if not kmeans_mask[i]:  # Only from K-Means background
            if gt[y, x] > 0:
                vessel_idx.append(i)
            else:
                non_vessel_idx.append(i)

    # Balance classes
    n_each = min(n_samples // 2, len(vessel_idx), len(non_vessel_idx))
    if n_each < 100:
        return np.array([]), np.array([])

    np.random.seed(42)
    v_idx = np.random.choice(vessel_idx, n_each, replace=False)
    nv_idx = np.random.choice(non_vessel_idx, n_each, replace=False)

    all_idx = np.concatenate([v_idx, nv_idx])
    X = features[all_idx]
    y = np.array(
        [1 if gt[coords[i][0], coords[i][1]] > 0 else 0 for i in all_idx]
    )

    return X, y


# ==================== POST-PROCESSING ====================


def post_process(vessel_img, mask):
    """Clean up the segmentation result"""
    # Erode mask to remove border artifacts
    kernel = np.ones((11, 11), np.uint8)
    mask_eroded = cv2.erode(mask, kernel)
    result = vessel_img & mask_eroded

    # Morphological opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_open)

    # Remove small components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
    result_clean = np.zeros_like(result)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 8:
            result_clean[labels == i] = 255

    return result_clean


# ==================== METRICS ====================


def calculate_metrics(pred, gt, mask):
    """Calculate evaluation metrics"""
    pred_m = (pred[mask > 0] > 0).astype(int)
    gt_m = (gt[mask > 0] > 0).astype(int)

    TP = np.sum((pred_m == 1) & (gt_m == 1))
    TN = np.sum((pred_m == 0) & (gt_m == 0))
    FP = np.sum((pred_m == 1) & (gt_m == 0))
    FN = np.sum((pred_m == 0) & (gt_m == 1))

    return {
        "Sensitivity": TP / (TP + FN + 1e-8),
        "Specificity": TN / (TN + FP + 1e-8),
        "Accuracy": (TP + TN) / (TP + TN + FP + FN + 1e-8),
        "PPV": TP / (TP + FP + 1e-8),
    }


# ==================== MAIN PROCESSING ====================


def process_image(
    img_path, mask_path, gt_path, classifier=None, save_dir=None
):
    """Process a single retinal image"""
    # Load
    image = load_image(img_path)
    mask = load_mask(mask_path)
    gt = load_ground_truth(gt_path)

    # Preprocess
    prep = preprocess(image, mask)
    crop = prep["crop"]
    gt_crop = gt[crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2]]

    # Extract features
    features, coords, feat_images = extract_features(prep, prep["mask"])

    # PCA
    pca = PCA(n_components=13)
    features_pca = pca.fit_transform(features)

    # K-Means clustering (unsupervised - thick vessels)
    kmeans_mask, confidence = kmeans_segment(features_pca, threshold=0.45)

    h, w = prep["mask"].shape
    kmeans_result = np.zeros((h, w), dtype=np.uint8)
    for i, (y, x) in enumerate(coords):
        if kmeans_mask[i]:
            kmeans_result[y, x] = 255

    # Train classifier if needed
    if classifier is None:
        X_train, y_train = prepare_training_data(
            features_pca, coords, gt_crop, kmeans_mask
        )
        if len(X_train) > 0:
            classifier = train_classifier(X_train, y_train)

    # Classification (supervised - thin vessels)
    clf_result = np.zeros((h, w), dtype=np.uint8)
    if classifier is not None:
        non_vessel_feat, non_vessel_coords = [], []
        for i, (y, x) in enumerate(coords):
            if not kmeans_mask[i]:
                non_vessel_feat.append(features_pca[i])
                non_vessel_coords.append((y, x))

        if len(non_vessel_feat) > 0:
            proba = classifier.predict_proba(np.array(non_vessel_feat))
            preds = (proba[:, 1] > 0.6).astype(int)  # Daha katı threshold
            for i, (y, x) in enumerate(non_vessel_coords):
                if preds[i] == 1:
                    clf_result[y, x] = 255

    # Combine results
    combined = np.maximum(kmeans_result, clf_result)
    final = post_process(combined, prep["mask"])

    # Calculate metrics
    metrics = calculate_metrics(final, gt_crop, prep["mask"])

    # Save outputs
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        name = os.path.basename(img_path).split("_")[0]
        cv2.imwrite(f"{save_dir}/{name}_input.png", prep["image"])
        cv2.imwrite(f"{save_dir}/{name}_G.png", prep["G"])
        cv2.imwrite(f"{save_dir}/{name}_tophat.png", feat_images[10])
        cv2.imwrite(f"{save_dir}/{name}_sc.png", feat_images[11])
        cv2.imwrite(f"{save_dir}/{name}_bps.png", feat_images[12])
        cv2.imwrite(f"{save_dir}/{name}_kmeans.png", kmeans_result)
        cv2.imwrite(f"{save_dir}/{name}_clf.png", clf_result)
        cv2.imwrite(f"{save_dir}/{name}_final.png", final)
        cv2.imwrite(f"{save_dir}/{name}_gt.png", gt_crop)

    results = {
        "kmeans_result": kmeans_result,
        "clf_result": clf_result,
        "final_result": final,
        "ground_truth": gt_crop,
        "features": feat_images,
        "preprocessed": prep,
    }

    return metrics, classifier, results


def run_pipeline(drive_path, dataset="training"):
    """Run full pipeline on dataset"""
    print("=" * 60)
    print("Retinal Blood Vessel Extraction")
    print("=" * 60)

    img_dir = os.path.join(drive_path, dataset, "images")
    mask_dir = os.path.join(drive_path, dataset, "mask")
    gt_dir = os.path.join(drive_path, dataset, "1st_manual")
    out_dir = os.path.join(drive_path, "output", dataset)

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    print(f"\nProcessing {len(files)} images...\n")

    all_metrics = []

    for i, f in enumerate(files):
        clf = None  # Her görüntü için yeniden eğit
        num = f.split("_")[0]
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f"{num}_{dataset}_mask.gif")
        gt_path = os.path.join(gt_dir, f"{num}_manual1.gif")

        if not all(os.path.exists(p) for p in [img_path, mask_path, gt_path]):
            print(f"[{i+1}/{len(files)}] {f} - SKIPPED (missing files)")
            continue

        try:
            m, clf, _ = process_image(
                img_path, mask_path, gt_path, clf, f"{out_dir}/{num}"
            )
            all_metrics.append(m)
            print(
                f"[{i+1}/{len(files)}] {f} - Sen: {m['Sensitivity']:.4f} | Spe: {m['Specificity']:.4f} | Acc: {m['Accuracy']:.4f}"
            )
        except Exception as e:
            print(f"[{i+1}/{len(files)}] {f} - ERROR: {e}")

    # Print summary
    if all_metrics:
        print("\n" + "=" * 60)
        print("RESULTS (Table 1)")
        print("=" * 60)
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"Sensitivity: {avg['Sensitivity']:.4f}")
        print(f"Specificity: {avg['Specificity']:.4f}")
        print(f"Accuracy:    {avg['Accuracy']:.4f}")
        print(f"PPV:         {avg['PPV']:.4f}")
        print("=" * 60)

        # Save results
        with open(f"{out_dir}/results.txt", "w") as f:
            f.write(f"Sen: {avg['Sensitivity']:.4f}\n")
            f.write(f"Spe: {avg['Specificity']:.4f}\n")
            f.write(f"Acc: {avg['Accuracy']:.4f}\n")
            f.write(f"PPV: {avg['PPV']:.4f}\n")

    return all_metrics


if __name__ == "__main__":
    run_pipeline("DRIVE", "training")
