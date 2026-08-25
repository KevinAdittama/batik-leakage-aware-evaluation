"""Fungsi data, citra, dan enam fitur yang dipakai seluruh pipeline."""

from __future__ import annotations

import hashlib
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline_config import (
    CLASS_TO_LABEL,
    EDGE_IMG_SIZE,
    GLCM_LEVELS,
    MODEL_FEATURES,
    PROJECT_DIR,
    TEX_IMG_SIZE,
    VALID_EXT,
)


def reset_directory(path: Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_class_name(value: str) -> str:
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    if value not in CLASS_TO_LABEL:
        raise ValueError(
            f"Kelas tidak valid: {value!r}; harus tepat batik/non_batik."
        )
    return value


def infer_subtype_from_filename(path: Path) -> str:
    value = path.stem.lower()
    value = re.sub(r"\s*\(\d+\)\s*$", "", value)
    value = re.sub(r"\d+$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    for prefix in ("non_batik_", "nonbatik_", "batik_", "flanel_", "kain_"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return value or "tidak_diketahui"


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return result or "image"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def dataset_records(root: Path) -> list[dict]:
    """Label berasal hanya dari folder teratas batik/non_batik."""
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")
    unexpected = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.lower() not in CLASS_TO_LABEL
    ]
    if unexpected:
        raise ValueError(f"Folder kelas tak dikenal di {root}: {sorted(unexpected)}")

    rows = []
    for class_name in ("batik", "non_batik"):
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Folder kelas wajib tidak ada: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VALID_EXT:
                continue
            relative_class = path.relative_to(class_dir)
            subjenis = (
                relative_class.parent.as_posix()
                if relative_class.parent != Path(".")
                else infer_subtype_from_filename(path)
            )
            portable = path.relative_to(PROJECT_DIR).as_posix()
            rows.append(
                {
                    "source_id": stable_id(portable),
                    "path": portable,
                    "kelas": class_name,
                    "label": CLASS_TO_LABEL[class_name],
                    "subjenis": subjenis,
                }
            )
    return rows


def read_image_color(path: Path):
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        with Image.open(path) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError):
        return None


def write_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise OSError(f"Gagal encode citra: {path}")
    encoded.tofile(str(path))


def clahe_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def make_fine_edge(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)


def make_coarse_edge(gray: np.ndarray) -> np.ndarray:
    coarse = cv2.Canny(cv2.GaussianBlur(gray, (9, 9), 0), 100, 220)
    return cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def filter_large_contours(edge: np.ndarray, min_area=300, min_length=80):
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(edge)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area or cv2.arcLength(contour, False) >= min_length:
            cv2.drawContours(filtered, [contour], -1, 255, 1)
    return filtered


def make_motif_edge(fine: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    mask = cv2.dilate(coarse, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.bitwise_and(fine, cv2.bitwise_not(mask))


def count_small_contours(edge: np.ndarray, threshold=300) -> int:
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(cv2.contourArea(contour) < threshold for contour in contours)


def edge_features(bgr: np.ndarray, want_viz=False):
    resized = cv2.resize(bgr, (EDGE_IMG_SIZE, EDGE_IMG_SIZE))
    gray = clahe_gray(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    fine = make_fine_edge(gray)
    coarse = filter_large_contours(make_coarse_edge(gray))
    motif = make_motif_edge(fine, coarse)
    small = count_small_contours(fine)
    features = {
        "motif_complexity_score": float(np.count_nonzero(motif) / motif.size * small),
        "small_contour_count": float(small),
    }
    viz = {"edge_gray": gray, "fine_edge": fine, "motif_edge": motif} if want_viz else None
    return features, viz


def glcm_features(gray: np.ndarray, levels=GLCM_LEVELS):
    quantized = (gray.astype(np.float32) / 256.0 * levels).astype(np.int32)
    quantized[quantized >= levels] = levels - 1
    height, width = quantized.shape
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    values = []
    for dy, dx in [(0, 1), (-1, 1), (-1, 0), (-1, -1)]:
        y0, y1 = max(0, -dy), height - max(0, dy)
        x0, x1 = max(0, -dx), width - max(0, dx)
        first = quantized[y0:y1, x0:x1]
        second = quantized[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (first.ravel(), second.ravel()), 1)
        matrix += matrix.T
        if matrix.sum() > 0:
            matrix /= matrix.sum()
        homogeneity = np.sum(matrix / (1.0 + (ii - jj) ** 2))
        nonzero = matrix[matrix > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero))) if nonzero.size else 0.0
        values.append((homogeneity, entropy))
    mean = np.mean(values, axis=0)
    return {"glcm_homogeneity": float(mean[0]), "glcm_entropy": float(mean[1])}


def lbp_code_and_features(gray: np.ndarray):
    image = gray.astype(np.int32)
    center = image[1:-1, 1:-1]
    neighbors = [
        image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:],
        image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2],
    ]
    code = np.zeros_like(center)
    for index, neighbor in enumerate(neighbors):
        code += (neighbor >= center).astype(np.int32) << index
    histogram = np.bincount(code.ravel(), minlength=256).astype(float)
    histogram /= max(histogram.sum(), 1)
    nonzero = histogram[histogram > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero))) if nonzero.size else 0.0
    return code.astype(np.uint8), {"lbp_entropy": entropy}


def fft_spectrum_and_feature(gray: np.ndarray):
    image = gray.astype(np.float32) - gray.mean()
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(image * window)))
    height, width = magnitude.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
    selected = magnitude[radius > 8]
    ratio = float(selected.max() / (selected.mean() + 1e-9)) if selected.size else 0.0
    return np.log1p(magnitude), {"fft_peak_ratio": ratio}


def texture_features(bgr: np.ndarray, want_viz=False):
    resized = cv2.resize(bgr, (TEX_IMG_SIZE, TEX_IMG_SIZE))
    gray = clahe_gray(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    features = glcm_features(gray)
    lbp, lbp_feature = lbp_code_and_features(gray)
    spectrum, fft_feature = fft_spectrum_and_feature(gray)
    features.update(lbp_feature)
    features.update(fft_feature)
    viz = {"texture_gray": gray, "lbp": lbp, "fft": spectrum} if want_viz else None
    return features, viz


def extract_six(bgr: np.ndarray, want_viz=False):
    edge, edge_viz = edge_features(bgr, want_viz)
    texture, texture_viz = texture_features(bgr, want_viz)
    features = {**edge, **texture}
    if any(not math.isfinite(float(features[name])) for name in MODEL_FEATURES):
        raise ValueError("Ekstraksi menghasilkan fitur non-finite.")
    viz = {**edge_viz, **texture_viz} if want_viz else None
    return features, viz


def diagnostic_views(bgr: np.ndarray):
    features, viz = extract_six(bgr, want_viz=True)
    resized = cv2.resize(bgr, (TEX_IMG_SIZE, TEX_IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    gray = viz["texture_gray"]
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edge_small = cv2.resize(viz["fine_edge"], (TEX_IMG_SIZE, TEX_IMG_SIZE))
    lines = cv2.HoughLinesP(edge_small, 1, np.pi / 180, 60, minLineLength=35, maxLineGap=8)
    line_overlay = cv2.cvtColor(edge_small, cv2.COLOR_GRAY2RGB)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_overlay, (x1, y1), (x2, y2), (255, 80, 20), 1)
    curve_overlay = rgb.copy()
    contours, _ = cv2.findContours(edge_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        arc = cv2.arcLength(contour, False)
        if arc < 25:
            continue
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (arc * arc + 1e-9)
        vertices = len(cv2.approxPolyDP(contour, 0.03 * arc, False))
        curved = circularity > 0.08 or vertices >= 6
        color = (220, 40, 40) if curved else (30, 180, 80)
        cv2.drawContours(curve_overlay, [contour], -1, color, 1)
    return {
        "Original": rgb,
        "CLAHE Gray": gray,
        "Fine Edge": viz["fine_edge"],
        "Motif Edge": viz["motif_edge"],
        "Otsu": otsu,
        "Straight-line diagnostic": line_overlay,
        "Curve diagnostic": curve_overlay,
        "LBP": viz["lbp"],
        "FFT spectrum": viz["fft"],
    }, features
