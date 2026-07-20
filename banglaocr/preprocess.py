"""
Preprocessing stage: deskew, denoise, contrast-enhance old/degraded newspaper scans.
Pure OpenCV/numpy - no model downloads required, works fully offline.
"""
from __future__ import annotations

import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_borders(gray: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """
    Remove dark borders/rulers from edges.
    Applies adaptive thresholding and finds the largest central bounding box.
    """
    # Create adaptive threshold as suggested
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    
    # Projection profiles to find the content area
    row_sum = np.sum(bw, axis=1)
    col_sum = np.sum(bw, axis=0)
    
    h, w = gray.shape
    h_m, w_m = int(h * margin), int(w * margin)
    
    # 5% threshold of max row/col density
    row_thresh = np.max(row_sum[h_m:h-h_m]) * 0.05
    col_thresh = np.max(col_sum[w_m:w-w_m]) * 0.05
    
    valid_rows = np.where(row_sum > row_thresh)[0]
    valid_cols = np.where(col_sum > col_thresh)[0]
    
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return gray
        
    y_min, y_max = max(0, valid_rows[0] - 10), min(h, valid_rows[-1] + 10)
    x_min, x_max = max(0, valid_cols[0] - 10), min(w, valid_cols[-1] + 10)
    
    # Fallback if crop is too aggressive
    if (y_max - y_min) < h * 0.5 or (x_max - x_min) < w * 0.5:
        return gray
        
    return gray[y_min:y_max, x_min:x_max]


def denoise(gray: np.ndarray) -> np.ndarray:
    """Remove scan noise / foxing spots while preserving text edges."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE: brings out faded ink without blowing out bright paper background."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate page rotation via minAreaRect on thresholded text mass."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV's minAreaRect angle convention needs normalizing to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def deskew(gray: np.ndarray, angle: float | None = None) -> np.ndarray:
    if angle is None:
        angle = estimate_skew_angle(gray)
    if abs(angle) < 0.1:
        return gray
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold - handles uneven lighting/aging better than a global threshold."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )


def upscale(gray: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Simple cubic upscale for low-DPI scans. Swap this out for a real super-resolution
    model (Real-ESRGAN / SwinIR) later if accuracy on small/blurry text needs more help -
    those require GPU + model weights so they're intentionally not wired in by default.
    """
    if scale == 1.0:
        return gray
    h, w = gray.shape
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def preprocess_page(
    path: str,
    do_crop: bool = True,
    do_denoise: bool = True,
    do_contrast: bool = True,
    do_deskew: bool = True,
    do_upscale: bool = False,
    upscale_factor: float = 2.0,
) -> np.ndarray:
    """Full preprocessing chain. Returns a clean grayscale image ready for segmentation."""
    gray = load_grayscale(path)

    if do_crop:
        gray = crop_borders(gray)
    if do_upscale:
        gray = upscale(gray, upscale_factor)
    if do_denoise:
        gray = denoise(gray)
    if do_contrast:
        gray = enhance_contrast(gray)
    if do_deskew:
        gray = deskew(gray)

    return gray


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input_image> <output_image>")
        sys.exit(1)
    result = preprocess_page(sys.argv[1])
    cv2.imwrite(sys.argv[2], result)
    print(f"Wrote preprocessed image to {sys.argv[2]}")
