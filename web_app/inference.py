"""Validasi citra dan inferensi yang memakai pipeline penelitian secara langsung."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from pipeline_common import extract_six
from pipeline_config import LABEL_TO_CLASS, MODEL_FEATURES
from pipeline_models import batik_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
CV_RESULT_PATH = PROJECT_DIR / "hasil_paper" / "07_cross_validation" / "best_cv_model.json"
EXTERNAL_RESULT_PATH = (
    PROJECT_DIR / "hasil_paper" / "08_uji_eksternal" / "selected_model_result.json"
)
MODEL_DIR = PROJECT_DIR / "hasil_paper" / "08_uji_eksternal"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
MIN_IMAGE_SIDE = 32


FEATURE_LABELS = {
    "motif_complexity_score": "Kompleksitas motif",
    "small_contour_count": "Jumlah kontur kecil",
    "glcm_entropy": "Entropi GLCM",
    "glcm_homogeneity": "Homogenitas GLCM",
    "lbp_entropy": "Entropi LBP",
    "fft_peak_ratio": "Rasio puncak FFT",
}

FEATURE_DOMAINS = {
    "motif_complexity_score": "Edge / morfologi",
    "small_contour_count": "Edge / morfologi",
    "glcm_entropy": "Tekstur",
    "glcm_homogeneity": "Tekstur",
    "lbp_entropy": "Tekstur",
    "fft_peak_ratio": "Frekuensi",
}

FEATURE_DESCRIPTIONS = {
    "motif_complexity_score": "Kepadatan struktur motif setelah tepi kasar disaring.",
    "small_contour_count": "Banyaknya kontur kecil yang menangkap detail lokal.",
    "glcm_entropy": "Keragaman pasangan intensitas pada tekstur citra.",
    "glcm_homogeneity": "Kedekatan distribusi intensitas terhadap diagonal GLCM.",
    "lbp_entropy": "Keragaman pola mikro permukaan berbasis Local Binary Pattern.",
    "fft_peak_ratio": "Kekuatan puncak frekuensi terhadap rerata spektrum.",
}


@dataclass(frozen=True)
class ModelBundle:
    model: object
    model_name: str
    model_slug: str
    cv_result: dict
    external_result: dict
    model_path: Path


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: int
    predicted_class: str
    score_batik: float
    confidence: float
    features: dict[str, float]
    visualizations: dict[str, np.ndarray]


def load_model_bundle() -> ModelBundle:
    """Muat model yang dipilih secara formal dari hasil 5-fold CV."""
    missing = [
        path for path in (CV_RESULT_PATH, EXTERNAL_RESULT_PATH) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Artefak penelitian belum lengkap. Jalankan run_all.py terlebih dahulu: "
            + ", ".join(str(path) for path in missing)
        )

    cv_result = json.loads(CV_RESULT_PATH.read_text(encoding="utf-8"))
    external_result = json.loads(EXTERNAL_RESULT_PATH.read_text(encoding="utf-8"))
    model_slug = str(cv_result["model_slug"])
    model_path = MODEL_DIR / f"model_{model_slug}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model final tidak ditemukan: {model_path}")

    model = joblib.load(model_path)
    return ModelBundle(
        model=model,
        model_name=str(cv_result["model"]),
        model_slug=model_slug,
        cv_result=cv_result,
        external_result=external_result,
        model_path=model_path,
    )


def decode_uploaded_image(data: bytes) -> np.ndarray:
    """Validasi berkas unggahan lalu kembalikan citra BGR untuk OpenCV."""
    if not data:
        raise ValueError("Berkas kosong. Pilih citra JPG, PNG, WEBP, atau BMP.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Ukuran citra melebihi batas 10 MB.")

    try:
        with Image.open(BytesIO(data)) as opened:
            width, height = opened.size
            if min(width, height) < MIN_IMAGE_SIDE:
                raise ValueError("Resolusi citra terlalu kecil; minimal 32 × 32 piksel.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Resolusi citra terlalu besar; maksimal 30 megapiksel.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            rgb = np.asarray(image)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Berkas tidak dapat dikenali sebagai citra yang valid.") from error

    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)


def _colorize_gray(image: np.ndarray, color_map: int) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized.astype(np.uint8), color_map)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _prepare_visualizations(bgr: np.ndarray, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fft = np.asarray(raw["fft"], dtype=np.float32)
    upper = float(np.percentile(fft, 99.5))
    fft = np.clip(fft, 0, upper if upper > 0 else 1)
    return {
        "original": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "motif_edge": raw["motif_edge"],
        "lbp": _colorize_gray(raw["lbp"], cv2.COLORMAP_INFERNO),
        "fft": _colorize_gray(fft, cv2.COLORMAP_VIRIDIS),
    }


def predict_image(bgr: np.ndarray, bundle: ModelBundle) -> PredictionResult:
    """Ekstrak enam fitur dalam urutan formal lalu lakukan prediksi model final."""
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Citra internal harus berupa array BGR tiga kanal.")

    features, visualization = extract_six(bgr, want_viz=True)
    values = np.asarray([[features[name] for name in MODEL_FEATURES]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Ekstraksi citra menghasilkan nilai fitur yang tidak valid.")

    predicted_label = int(bundle.model.predict(values)[0])
    score_batik = float(batik_score(bundle.model, values)[0])
    confidence = score_batik if predicted_label == 1 else 1.0 - score_batik
    return PredictionResult(
        predicted_label=predicted_label,
        predicted_class=LABEL_TO_CLASS[predicted_label],
        score_batik=score_batik,
        confidence=confidence,
        features={name: float(features[name]) for name in MODEL_FEATURES},
        visualizations=_prepare_visualizations(bgr, visualization),
    )


def model_feature_importance(bundle: ModelBundle) -> dict[str, float]:
    """Ambil importance global bila estimator formal menyediakannya."""
    model = bundle.model
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None:
        return {}
    return {
        name: float(value) for name, value in zip(MODEL_FEATURES, importance, strict=True)
    }

