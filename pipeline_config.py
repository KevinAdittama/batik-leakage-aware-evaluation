"""Konfigurasi tunggal pipeline penelitian batik vs non-batik."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT_CONFIG_DIR = PROJECT_DIR / "audit_config"
DEVELOPMENT_EXCLUSIONS_FILE = AUDIT_CONFIG_DIR / "development_exclusions.csv"
SOURCE_GROUPS_FILE = AUDIT_CONFIG_DIR / "source_groups.csv"
DATASET_DIR = PROJECT_DIR / "dataset_batik"
EXTERNAL_DIR = PROJECT_DIR / "uji_eksternal"
AUGMENTED_DIR = PROJECT_DIR / "dataset_aug"
RESULTS_DIR = PROJECT_DIR / "hasil_paper"

AUDIT_DIR = RESULTS_DIR / "01_audit"
PREVIEW_ORIGINAL_DIR = RESULTS_DIR / "02_preview_original"
AUGMENTATION_DIR = RESULTS_DIR / "03_augmentasi"
PREVIEW_AUGMENTED_DIR = RESULTS_DIR / "04_preview_augmented"
FEATURE_DIR = RESULTS_DIR / "05_fitur"
FEATURE_ANALYSIS_DIR = RESULTS_DIR / "06_analisis_fitur"
CV_DIR = RESULTS_DIR / "07_cross_validation"
EXTERNAL_RESULT_DIR = RESULTS_DIR / "08_uji_eksternal"
TABLE_DIR = RESULTS_DIR / "09_tabel_paper"
ERROR_DIR = RESULTS_DIR / "10_analisis_error"
DEEP_LEARNING_DIR = RESULTS_DIR / "11_deep_learning_baseline"
LOG_DIR = RESULTS_DIR / "00_logs"

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_TO_LABEL = {"non_batik": 0, "batik": 1}
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}

MODEL_FEATURES = [
    "motif_complexity_score",
    "small_contour_count",
    "glcm_entropy",
    "glcm_homogeneity",
    "lbp_entropy",
    "fft_peak_ratio",
]

FEATURE_GROUPS = {
    "Edge/Morfologi": ["motif_complexity_score", "small_contour_count"],
    "Tekstur": ["glcm_entropy", "glcm_homogeneity", "lbp_entropy"],
    "Frekuensi": ["fft_peak_ratio"],
    "Gabungan 6 Fitur": MODEL_FEATURES,
}

EDGE_IMG_SIZE = 512
TEX_IMG_SIZE = 256
GLCM_LEVELS = 16
TARGET_PER_KELAS = 200
CV_POOL_PER_KELAS = 300
N_SPLITS = 5
RANDOM_SEED = 42
PREVIEW_PER_CLASS = 6
