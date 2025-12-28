"""
Retinal Blood Vessel Extraction
================================
Makale: "Retinal blood vessel extraction employing effective image features
        and combination of supervised and unsupervised machine learning methods"

Yazar: Hashemzadeh & Adlpour Azar (2019)

Bu kod CENG 391 Final Projesi için hazırlanmıştır.
"""

import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from skimage.filters import gabor
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# ADIM 1: ÖN İŞLEME (PREPROCESSING)
# =============================================================================
"""
Ön işleme adımları:
1. FOV (Field of View) çıkarma - görüntünün sadece göz kısmını al
2. Renk uzayı dönüşümü - RGB'den L*a*b ve YCbCr'ye
3. Kanal seçimi - G, Y, L kanallarını seç
4. CLAHE ile kontrast iyileştirme
"""


def load_image(image_path):
    """Görüntüyü yükle"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    return img


def load_mask(mask_path):
    """FOV maskesini yükle"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Maske yüklenemedi: {mask_path}")
    return mask


def load_ground_truth(gt_path):
    """Ground truth görüntüsünü yükle"""
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise ValueError(f"Ground truth yüklenemedi: {gt_path}")
    # Binary hale getir
    _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    return gt


def crop_to_fov(image, mask):
    """
    FOV (Field of View) bölgesine kırp
    Makale Section 3.2'de açıklanıyor
    """
    # Maskenin sınırlarını bul
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # Görüntüyü ve maskeyi kırp
    cropped_image = image[y : y + h, x : x + w]
    cropped_mask = mask[y : y + h, x : x + w]

    return cropped_image, cropped_mask, (x, y, w, h)


def extract_channels(image):
    """
    Üç farklı renk uzayından en uygun kanalları çıkar
    - RGB'den G (Green) kanalı
    - YCbCr'den Y kanalı
    - L*a*b'den L kanalı

    Makale Section 3.2 ve Figure 3'te açıklanıyor
    """
    # G kanalı (RGB'den)
    G_channel = image[:, :, 1]  # OpenCV BGR formatında, index 1 = Green

    # Y kanalı (YCbCr'den)
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y_channel = ycbcr[:, :, 0]

    # L kanalı (L*a*b'den)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L_channel = lab[:, :, 0]

    return G_channel, Y_channel, L_channel


def apply_clahe(channel, clip_limit=0.01, tile_size=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) uygula

    Makale Section 3.2'de:
    - tile size: 8x8
    - clip limit: 0.01
    - histogram bins: 256
    - histogram shape: uniform

    CLAHE nedir? Normal histogram equalization tüm görüntüye uygulanır,
    CLAHE ise küçük bölgelere (tiles) ayrı ayrı uygular. Bu sayede
    lokal kontrast iyileşir ve damarlar daha görünür olur.
    """
    # CLAHE nesnesini oluştur
    # clipLimit: kontrast sınırlaması (çok yüksek olursa gürültü artar)
    clahe = cv2.createCLAHE(clipLimit=clip_limit * 256, tileGridSize=tile_size)

    # Uygula ve döndür
    enhanced = clahe.apply(channel)
    return enhanced


def preprocess_image(image, mask):
    """
    Tüm ön işleme adımlarını uygula
    """
    # 1. FOV'a kırp
    cropped_img, cropped_mask, crop_coords = crop_to_fov(image, mask)

    # 2. Kanalları çıkar
    G, Y, L = extract_channels(cropped_img)

    # 3. CLAHE uygula
    G_enhanced = apply_clahe(G)
    Y_enhanced = apply_clahe(Y)
    L_enhanced = apply_clahe(L)

    return {
        "G": G_enhanced,
        "Y": Y_enhanced,
        "L": L_enhanced,
        "G_original": G,
        "mask": cropped_mask,
        "crop_coords": crop_coords,
        "cropped_image": cropped_img,
    }


# =============================================================================
# ADIM 2: ÖZELLİK ÇIKARMA (FEATURE EXTRACTION)
# =============================================================================
"""
13 özellik çıkarılıyor:
- 9 Gabor özelliği (3 kanal × 3 wavelength)
- 1 G kanalı (histogram equalized)
- 1 Top Hat (TH) özelliği
- 1 Shade Corrected (SC) özelliği
- 1 Bit Plane Slicing (BPS) özelliği
"""


def apply_gabor_filter(image, wavelength, orientation_count=12):
    """
    Gabor filtresi uygula

    Gabor filtresi nedir? Sinüzoidal bir dalga ile Gaussian'ın çarpımıdır.
    Kan damarları gibi çizgisel yapıları tespit etmek için idealdir.

    Makale Section 3.3.1.1'de parametreler:
    - wavelength: 9, 10, 11 (damar kalınlığına göre)
    - aspect ratio: 0.5
    - bandwidth: 1
    - orientation: 0'dan başlayıp 15 derece aralıklarla (12 yön)
    """
    # Tüm yönlerdeki maksimum yanıtı al
    max_response = np.zeros_like(image, dtype=np.float64)

    for i in range(orientation_count):
        theta = i * np.pi / orientation_count  # 0 ile π arasında yönler

        # Gabor filtresi uygula
        # sigma = wavelength * 0.5 / π ≈ wavelength * 0.16
        sigma = wavelength * 0.56  # bandwidth=1 için yaklaşık değer

        # OpenCV Gabor kernel
        kernel_size = int(wavelength * 3) | 1  # Tek sayı olmalı
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma=sigma,
            theta=theta,
            lambd=wavelength,
            gamma=0.5,  # aspect ratio
            psi=0,
        )

        # Filtrele
        filtered = cv2.filter2D(image, cv2.CV_64F, kernel)

        # Maksimum yanıtı güncelle
        max_response = np.maximum(max_response, filtered)

    return max_response


def extract_gabor_features(G, Y, L):
    """
    9 Gabor özelliği çıkar (3 kanal × 3 wavelength)

    Makale Figure 5 ve 6'da gösteriliyor
    """
    wavelengths = [9, 10, 11]
    gabor_features = []

    channels = {"G": G, "Y": Y, "L": L}

    for wavelength in wavelengths:
        for name, channel in channels.items():
            gabor_response = apply_gabor_filter(channel, wavelength)
            gabor_features.append(gabor_response)

    return gabor_features


def automatic_threshold(image):
    """
    Otomatik eşikleme

    Makale Section 3.3.1.1'de açıklanan yöntem:
    1. Her piksel değerinin kaç kez tekrarladığını say
    2. Her değeri tekrar sayısıyla çarp
    3. Toplamı al - bu eşik değeri olur

    Bu yöntem görüntünün istatistiksel dağılımına göre eşik belirler.
    """
    # Normalize et
    img_normalized = cv2.normalize(
        image, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Histogram hesapla
    hist = cv2.calcHist([img_normalized], [0], None, [256], [0, 256]).flatten()

    # Toplam piksel sayısı
    total_pixels = image.size

    # Eşik değerini hesapla
    threshold = 0
    for i in range(256):
        threshold += i * (hist[i] / total_pixels)

    # Laplacian ekle ve eşikle
    laplacian = cv2.Laplacian(img_normalized, cv2.CV_64F)
    combined = img_normalized.astype(np.float64) + laplacian
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    _, binary = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)

    return binary


def binarize_gabor_features(gabor_features):
    """
    Gabor özelliklerini binary görüntülere dönüştür
    Makale Figure 6'da gösteriliyor
    """
    binary_features = []
    for gabor_img in gabor_features:
        binary = automatic_threshold(gabor_img)
        binary_features.append(binary)
    return binary_features


def extract_tophat_feature(G_channel):
    """
    Top Hat özelliği çıkar

    Makale Section 3.3.1.2.1'de açıklanıyor:
    TopHat(A) = A - (A ∘ B)

    Top Hat nedir? Opening işleminden sonra orijinal görüntüden çıkarılır.
    Bu işlem, yapısal elemandan küçük parlak nesneleri (damarları) ortaya çıkarır.

    Parametreler:
    - Yapısal eleman: çizgisel (linear)
    - Uzunluk: 21 piksel
    - 9 farklı yön (0'dan başlayıp 22.5 derece aralıklarla)
    """
    # G kanalını ters çevir (damarlar koyu, ters çevirince parlak olur)
    inverted = 255 - G_channel

    # 9 farklı yönde Top Hat uygula ve maksimumu al
    max_tophat = np.zeros_like(inverted, dtype=np.float64)

    for i in range(9):
        angle = i * 22.5  # 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180

        # Çizgisel yapısal eleman oluştur
        length = 21
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

        # Döndür
        M = cv2.getRotationMatrix2D((length // 2, 0), angle, 1)
        kernel_rotated = cv2.warpAffine(kernel, M, (length, length))

        # Top Hat uygula (opening sonrası çıkarma)
        opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel_rotated)
        tophat = inverted.astype(np.float64) - opened.astype(np.float64)

        # Maksimumu güncelle
        max_tophat = np.maximum(max_tophat, tophat)

    return cv2.normalize(max_tophat, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )


def extract_shade_corrected_feature(G_channel):
    """
    Shade Corrected (SC) özelliği çıkar

    Makale Section 3.3.1.2.2'de açıklanıyor:
    SC = Orijinal - Arka plan tahmini

    Arka plan tahmini median filtre ile yapılır (25×25 pencere)

    Bu neden önemli? Retina görüntülerinde ışık dağılımı düzensiz olabilir.
    SC bu düzensizliği giderir ve damarları daha net gösterir.
    """
    # Median filtre ile arka planı tahmin et
    background = cv2.medianBlur(G_channel, 25)

    # Orijinalden çıkar
    sc = G_channel.astype(np.float64) - background.astype(np.float64)

    # Normalize et
    sc = cv2.normalize(sc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return sc


def extract_bps_feature(G_channel):
    """
    Bit Plane Slicing (BPS) özelliği çıkar

    Makale Section 3.3.1.2.3'te açıklanıyor:
    - Her piksel 8 bit'ten oluşur
    - En anlamlı bitler (MSB) görüntünün ana yapısını içerir
    - Son iki bit düzlemi (7. ve 8.) en fazla bilgiyi taşır
    - BPS = 7. bit düzlemi + 8. bit düzlemi

    Bu özellik damarların yön haritasını ve şeklini verir.
    """
    # 7. bit düzlemi (0-indexed: bit 6)
    bit_plane_7 = (G_channel >> 6) & 1

    # 8. bit düzlemi (0-indexed: bit 7)
    bit_plane_8 = (G_channel >> 7) & 1

    # İkisini topla
    bps = (bit_plane_7 + bit_plane_8) * 127  # 0, 127, 254 değerleri

    return bps.astype(np.uint8)


def construct_feature_vector(preprocessed, mask):
    """
    13 boyutlu özellik vektörünü oluştur

    Her piksel için 13 özellik:
    - 9 Gabor binary özellik
    - 1 G kanalı (enhanced)
    - 1 Top Hat
    - 1 Shade Corrected
    - 1 Bit Plane Slicing
    """
    G, Y, L = preprocessed["G"], preprocessed["Y"], preprocessed["L"]
    G_original = preprocessed["G_original"]

    print("  Gabor özellikleri çıkarılıyor...")
    gabor_features = extract_gabor_features(G, Y, L)
    binary_gabor = binarize_gabor_features(gabor_features)

    print("  Top Hat özelliği çıkarılıyor...")
    tophat = extract_tophat_feature(G)

    print("  Shade Corrected özelliği çıkarılıyor...")
    sc = extract_shade_corrected_feature(G)

    print("  BPS özelliği çıkarılıyor...")
    bps = extract_bps_feature(G)

    # Tüm özellikleri birleştir
    all_features = binary_gabor + [G, tophat, sc, bps]

    # Sadece FOV içindeki pikselleri al
    h, w = mask.shape
    feature_vectors = []
    pixel_coords = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:  # FOV içinde
                fv = [f[y, x] for f in all_features]
                feature_vectors.append(fv)
                pixel_coords.append((y, x))

    return (
        np.array(feature_vectors, dtype=np.float64),
        pixel_coords,
        all_features,
    )


# =============================================================================
# ADIM 3: PCA (Principal Component Analysis)
# =============================================================================
"""
PCA neden kullanılıyor?
1. Özellikler arasındaki korelasyonu kaldırır
2. Veriyi yeni bir uzaya dönüştürür
3. Makale Section 3.3.2'de açıklanıyor

NOT: Makale boyut azaltma yapmıyor, sadece korelasyonu kaldırıyor.
"""


def apply_pca(feature_vectors, n_components=None):
    """
    PCA uygula

    n_components=None: tüm bileşenleri koru, sadece dönüşüm yap
    """
    if n_components is None:
        n_components = feature_vectors.shape[1]

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(feature_vectors)

    return transformed, pca


# =============================================================================
# ADIM 4: FCM KÜMELEME (Unsupervised - Kalın Damarlar)
# =============================================================================
"""
FCM (Fuzzy C-Means) nedir?
- Her nokta birden fazla kümeye ait olabilir (fuzzy)
- Üyelik dereceleri 0 ile 1 arasında
- K-Means'in yumuşak versiyonu

Makale Section 3.3.3'te parametreler:
- m (fuzziness): 2
- max iterations: 2000
- min improvement: 0.000001

Neden FCM? Kalın ve net damarları hızlıca ayırmak için.
"""


def fcm_clustering(data, n_clusters=2, m=2, max_iter=2000, error=1e-6):
    """
    Fuzzy C-Means kümeleme

    m: fuzziness parametresi (m=1 ise K-Means, m>1 ise fuzzy)
    """
    n_samples = data.shape[0]

    # Rastgele üyelik matrisi başlat
    np.random.seed(42)
    U = np.random.rand(n_samples, n_clusters)
    U = U / U.sum(axis=1, keepdims=True)  # Normalize et

    for iteration in range(max_iter):
        U_old = U.copy()

        # Küme merkezlerini hesapla
        Um = U**m
        centers = (Um.T @ data) / Um.sum(axis=0, keepdims=True).T

        # Uzaklıkları hesapla
        distances = np.zeros((n_samples, n_clusters))
        for j in range(n_clusters):
            diff = data - centers[j]
            distances[:, j] = np.sqrt(np.sum(diff**2, axis=1))

        # Sıfır uzaklıkları küçük bir değerle değiştir
        distances = np.maximum(distances, 1e-10)

        # Üyelik matrisini güncelle
        for j in range(n_clusters):
            denominator = np.zeros(n_samples)
            for k in range(n_clusters):
                denominator += (distances[:, j] / distances[:, k]) ** (
                    2 / (m - 1)
                )
            U[:, j] = 1 / denominator

        # Yakınsama kontrolü
        if np.max(np.abs(U - U_old)) < error:
            print(f"  FCM yakınsadı: {iteration + 1} iterasyon")
            break

    return U, centers


def apply_fcm(feature_vectors):
    """
    FCM uygula ve damar/arka plan kümelerini belirle
    """
    print("  FCM kümeleme uygulanıyor...")

    # FCM uygula
    U, centers = fcm_clustering(feature_vectors, n_clusters=2)

    # Her piksel için en yüksek üyelikli kümeyi bul
    labels = np.argmax(U, axis=1)

    # Hangi küme damar? (daha az piksel içeren küme damar kümesidir)
    unique, counts = np.unique(labels, return_counts=True)
    vessel_cluster = unique[np.argmin(counts)]

    # Damar olasılıkları
    vessel_probs = U[:, vessel_cluster]

    return labels, vessel_probs, vessel_cluster


# =============================================================================
# ADIM 5: KARAR AĞACI SINIFLANDIRMA (Supervised - İnce Damarlar)
# =============================================================================
"""
Neden karar ağacı?
- FCM kalın damarları iyi bulur ama ince damarları kaçırabilir
- Karar ağacı, etiketlenmiş veriden öğrenerek ince damarları da bulur

Makale Section 3.3.4'te "root guided decision tree" kullanılıyor.
Biz sklearn'ün DecisionTreeClassifier'ını kullanacağız.
"""


def train_decision_tree(X_train, y_train):
    """
    Karar ağacı eğit
    """
    clf = DecisionTreeClassifier(
        random_state=42, max_depth=20, min_samples_split=10, min_samples_leaf=5
    )
    clf.fit(X_train, y_train)
    return clf


def prepare_training_data(
    feature_vectors,
    pixel_coords,
    ground_truth,
    mask,
    fcm_vessel_mask,
    n_samples=10000,
):
    """
    Eğitim verisi hazırla

    Makale'ye göre:
    - DRIVE için her görüntüden 10,000 piksel
    - Sadece FCM'in arka plan olarak belirlediği piksellerden
    """
    # FCM'in arka plan dediği pikselleri filtrele
    non_vessel_indices = []

    for idx, (y, x) in enumerate(pixel_coords):
        if not fcm_vessel_mask[idx]:  # FCM'e göre arka plan
            non_vessel_indices.append(idx)

    # Bu piksellerden eğitim verisi oluştur
    X_train = []
    y_train = []

    for idx in non_vessel_indices:
        y, x = pixel_coords[idx]
        X_train.append(feature_vectors[idx])
        # Ground truth'a göre etiketle
        if ground_truth[y, x] > 0:
            y_train.append(1)  # Damar
        else:
            y_train.append(0)  # Arka plan

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Örnekle (çok fazla veri varsa)
    if len(X_train) > n_samples:
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    return X_train, y_train


# =============================================================================
# ADIM 6: SON İŞLEME (POST-PROCESSING)
# =============================================================================


def post_process(vessel_mask, mask, reduction=5):
    """
    Son işleme

    Makale Section 3.4'te açıklanıyor:
    FOV kenarındaki yanlış pozitif tespitleri kaldır
    """
    # Maskeyi küçült
    kernel = np.ones((reduction * 2 + 1, reduction * 2 + 1), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Küçültülmüş maskeyi uygula
    result = vessel_mask & eroded_mask

    return result


# =============================================================================
# ADIM 7: DEĞERLENDİRME METRİKLERİ
# =============================================================================


def calculate_metrics(predicted, ground_truth, mask):
    """
    Değerlendirme metriklerini hesapla

    Makale Section 4.2'de açıklanan metrikler:
    - Sensitivity (Sen): TP / (TP + FN) - Damarları bulma oranı
    - Specificity (Spe): TN / (TN + FP) - Arka planı doğru tanıma oranı
    - Accuracy (Acc): (TP + TN) / (TP + FP + FN + TN) - Genel doğruluk
    - PPV: TP / (TP + FP) - Pozitif tahmin değeri
    """
    # Sadece FOV içindeki pikselleri değerlendir
    pred_masked = predicted[mask > 0]
    gt_masked = ground_truth[mask > 0]

    # Binary yap
    pred_binary = (pred_masked > 0).astype(int)
    gt_binary = (gt_masked > 0).astype(int)

    # TP, TN, FP, FN hesapla
    TP = np.sum((pred_binary == 1) & (gt_binary == 1))
    TN = np.sum((pred_binary == 0) & (gt_binary == 0))
    FP = np.sum((pred_binary == 1) & (gt_binary == 0))
    FN = np.sum((pred_binary == 0) & (gt_binary == 1))

    # Metrikleri hesapla
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    accuracy = (
        (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    )
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "PPV": ppv,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }


# =============================================================================
# ANA FONKSİYON
# =============================================================================


def process_single_image(
    image_path, mask_path, gt_path, classifier=None, save_dir=None
):
    """
    Tek bir görüntüyü işle
    """
    print(f"\nGörüntü işleniyor: {os.path.basename(image_path)}")

    # 1. Yükle
    print("  Görüntüler yükleniyor...")
    image = load_image(image_path)
    mask = load_mask(mask_path)
    ground_truth = load_ground_truth(gt_path)

    # 2. Ön işleme
    print("  Ön işleme yapılıyor...")
    preprocessed = preprocess_image(image, mask)

    # Ground truth'u da kırp
    crop = preprocessed["crop_coords"]
    gt_cropped = ground_truth[
        crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2]
    ]

    # 3. Özellik çıkarma
    print("  Özellikler çıkarılıyor...")
    feature_vectors, pixel_coords, all_features = construct_feature_vector(
        preprocessed, preprocessed["mask"]
    )

    # 4. PCA
    print("  PCA uygulanıyor...")
    pca_features, pca_model = apply_pca(feature_vectors)

    # 5. FCM Kümeleme
    print("  FCM kümeleme yapılıyor...")
    fcm_labels, fcm_probs, vessel_cluster = apply_fcm(pca_features)

    # FCM sonucunu görüntüye dönüştür
    h, w = preprocessed["mask"].shape
    fcm_vessel_mask = fcm_labels == vessel_cluster
    fcm_result = np.zeros((h, w), dtype=np.uint8)
    for idx, (y, x) in enumerate(pixel_coords):
        if fcm_vessel_mask[idx]:
            fcm_result[y, x] = 255

    # 6. Karar Ağacı Sınıflandırma
    if classifier is None:
        print("  Karar ağacı eğitiliyor...")
        X_train, y_train = prepare_training_data(
            pca_features,
            pixel_coords,
            gt_cropped,
            preprocessed["mask"],
            fcm_vessel_mask,
        )
        classifier = train_decision_tree(X_train, y_train)

    print("  Karar ağacı sınıflandırması yapılıyor...")
    # Sadece FCM'in arka plan dediği pikselleri sınıflandır
    dt_result = np.zeros((h, w), dtype=np.uint8)

    non_vessel_features = []
    non_vessel_coords = []

    for idx, (y, x) in enumerate(pixel_coords):
        if not fcm_vessel_mask[idx]:
            non_vessel_features.append(pca_features[idx])
            non_vessel_coords.append((y, x))

    if len(non_vessel_features) > 0:
        non_vessel_features = np.array(non_vessel_features)
        predictions = classifier.predict(non_vessel_features)

        for idx, (y, x) in enumerate(non_vessel_coords):
            if predictions[idx] == 1:
                dt_result[y, x] = 255

    # 7. FCM ve DT sonuçlarını birleştir
    combined_result = np.maximum(fcm_result, dt_result)

    # 8. Son işleme
    print("  Son işleme yapılıyor...")
    final_result = post_process(combined_result, preprocessed["mask"])

    # 9. Değerlendirme
    metrics = calculate_metrics(final_result, gt_cropped, preprocessed["mask"])

    print(f"  Sonuçlar:")
    print(f"    Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"    Specificity: {metrics['Specificity']:.4f}")
    print(f"    Accuracy: {metrics['Accuracy']:.4f}")
    print(f"    PPV: {metrics['PPV']:.4f}")

    # Ara görüntüleri kaydet
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        cv2.imwrite(
            os.path.join(save_dir, f"{base_name}_input.png"),
            preprocessed["cropped_image"],
        )
        cv2.imwrite(
            os.path.join(save_dir, f"{base_name}_G_channel.png"),
            preprocessed["G"],
        )
        cv2.imwrite(
            os.path.join(save_dir, f"{base_name}_fcm_result.png"), fcm_result
        )
        cv2.imwrite(
            os.path.join(save_dir, f"{base_name}_dt_result.png"), dt_result
        )
        cv2.imwrite(
            os.path.join(save_dir, f"{base_name}_final_result.png"),
            final_result,
        )
        cv2.imwrite(
            os.path.join(save_dir, f"{base_name}_ground_truth.png"), gt_cropped
        )

    return (
        metrics,
        classifier,
        {
            "fcm_result": fcm_result,
            "dt_result": dt_result,
            "final_result": final_result,
            "ground_truth": gt_cropped,
            "all_features": all_features,
        },
    )


def main(drive_path):
    """
    Ana fonksiyon - tüm DRIVE veri setini işle
    """
    print("=" * 60)
    print("Retinal Blood Vessel Extraction")
    print("=" * 60)

    # Klasör yolları
    train_images = os.path.join(drive_path, "training", "images")
    train_manual = os.path.join(drive_path, "training", "1st_manual")
    train_mask = os.path.join(drive_path, "training", "mask")

    test_images = os.path.join(drive_path, "test", "images")
    test_mask = os.path.join(drive_path, "test", "mask")

    # Çıktı klasörü
    output_dir = os.path.join(drive_path, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Eğitim görüntülerini listele
    train_image_files = sorted(
        [f for f in os.listdir(train_images) if f.endswith(".tif")]
    )

    print(f"\nEğitim görüntüleri: {len(train_image_files)}")

    # İlk görüntüyü işle (demo)
    if train_image_files:
        img_file = train_image_files[0]
        img_num = img_file.split("_")[0]

        # Dosya yolları
        img_path = os.path.join(train_images, img_file)
        mask_path = os.path.join(train_mask, f"{img_num}_training_mask.gif")
        gt_path = os.path.join(train_manual, f"{img_num}_manual1.gif")

        # Dosyaların varlığını kontrol et
        if (
            os.path.exists(img_path)
            and os.path.exists(mask_path)
            and os.path.exists(gt_path)
        ):
            metrics, classifier, results = process_single_image(
                img_path,
                mask_path,
                gt_path,
                save_dir=os.path.join(output_dir, img_num),
            )

            print("\n" + "=" * 60)
            print("İşlem tamamlandı!")
            print(f"Sonuçlar kaydedildi: {os.path.join(output_dir, img_num)}")
            print("=" * 60)
        else:
            print("Hata: Dosyalar bulunamadı!")
            print(
                f"  Image: {img_path} - {'VAR' if os.path.exists(img_path) else 'YOK'}"
            )
            print(
                f"  Mask: {mask_path} - {'VAR' if os.path.exists(mask_path) else 'YOK'}"
            )
            print(
                f"  GT: {gt_path} - {'VAR' if os.path.exists(gt_path) else 'YOK'}"
            )


if __name__ == "__main__":
    # DRIVE veri seti yolunu belirt
    DRIVE_PATH = "/path/to/DRIVE"  # Bu yolu kendi yolunla değiştir

    print("Bu dosyayı çalıştırmak için:")
    print("1. DRIVE_PATH değişkenini kendi veri seti yolunla değiştir")
    print("2. python main.py komutunu çalıştır")
    print("\nVeya Python'da:")
    print(">>> from main import main")
    print(">>> main('/path/to/DRIVE')")
