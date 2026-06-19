import os
import sys
import argparse
import json
import shutil
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List, Dict

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  WARNING: opencv-python not installed. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# NEW: Optional SciPy guard for texture calculation
try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Enhanced configuration with content-aware preprocessing"""

    pass_threshold: float = 300.0
    min_resolution: int = 400

    # Enable/Disable preprocessing steps
    enable_clahe: bool = True
    enable_denoise: bool = True
    enable_sharpen: bool = True
    enable_morphology: bool = True
    enable_shadow_correction: bool = True
    enable_gamma_correction: bool = True
    enable_white_balance: bool = True

    # Quality safeguards
    max_sharpness_loss: float = 0.30  # Don't lose more than 30% sharpness
    max_clipping_ratio: float = 2.0   # Don't create 2x more clipped pixels

    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # Advanced denoising parameters
    denoise_h: float = 10.0
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Unsharp mask parameters
    sharpen_radius: float = 2.0
    sharpen_amount: float = 1.5
    sharpen_threshold: int = 0

    # Morphological operations
    morph_kernel_size: int = 3
    morph_operation: str = "closing"

    # Shadow/lighting correction
    shadow_kernel_size: int = 51

    # Gamma correction
    gamma_value: float = 1.2

    # Analysis
    analysis_size: int = 1024

    # Output
    default_output_dir: str = "./output"
    original_subdir: str = "original"
    preprocessed_subdir: str = "preprocessed"
    save_by_default: bool = True

CFG = Config()


# ============================================================================
# SUPPORTED FILE EXTENSIONS
# ============================================================================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
PDF_EXTENSION = '.pdf'


# ============================================================================
# BLUR DETECTION
# ============================================================================

def variance_of_laplacian(gray: np.ndarray) -> float:
    """Primary blur detection metric"""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


# ============================================================================
# IMAGE TYPE DETECTION
# ============================================================================

def detect_image_type(gray: np.ndarray, initial_vol_full: float) -> str:
    """
    Detect image type to apply appropriate preprocessing:
    - clean_document: High quality scans/photos (minimal processing)
    - medical_photo: Wound/medical images (preserve details)
    - dark_scene: Fire damage, dark photos (preserve darkness)
    - underexposed_phone: Poor quality phone photos (full pipeline)
    - general: Unknown/mixed content (moderate processing)
    """
    mean_bright = gray.mean()
    contrast = gray.std()

    # Calculate texture richness (with SciPy if available, otherwise OpenCV fallback)
    kernel_size = 15
    gray_float = gray.astype(np.float32)

    if HAS_SCIPY:
        local_mean = uniform_filter(gray_float, size=kernel_size)
        local_sq_mean = uniform_filter(gray_float ** 2, size=kernel_size)
    else:
        # Fallback using box filter via OpenCV
        k = (kernel_size, kernel_size)
        local_mean = cv2.blur(gray_float, k)
        local_sq_mean = cv2.blur(gray_float ** 2, k)

    local_var = local_sq_mean - local_mean ** 2
    texture_score = float(local_var.mean())

    # Image characteristics
    is_very_dark = mean_bright < 100
    is_bright = mean_bright > 180
    is_high_contrast = contrast > 50
    is_textured = texture_score > 200
    is_sharp = initial_vol_full > 200

    # Check for uneven lighting
    h, w = gray.shape
    top_half = gray[:h // 2, :].mean()
    bottom_half = gray[h // 2:, :].mean()
    left_half = gray[:, :w // 2].mean()
    right_half = gray[:, w // 2:].mean()
    lighting_variance = max(abs(top_half - bottom_half), abs(left_half - right_half))
    has_uneven_lighting = lighting_variance > 30

    # Classify image type
    if is_bright and is_sharp and contrast > 40 and not has_uneven_lighting:
        return "clean_document"
    elif is_textured and is_high_contrast and 100 < mean_bright < 200:
        return "medical_photo"
    elif is_very_dark and is_high_contrast and texture_score > 150:
        return "dark_scene"
    elif mean_bright < 120 and contrast < 35 and has_uneven_lighting:
        return "underexposed_phone"
    else:
        return "general"


# ============================================================================
# ADVANCED PREPROCESSING TECHNIQUES
# ============================================================================

def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Enhanced contrast using CLAHE"""
    clahe = cv2.createCLAHE(
        clipLimit=CFG.clahe_clip_limit,
        tileGridSize=(CFG.clahe_tile_size, CFG.clahe_tile_size)
    )
    return clahe.apply(gray)


def apply_advanced_denoise(gray: np.ndarray, use_bilateral: bool = False) -> np.ndarray:
    """
    Advanced denoising with two options:
    1. Non-Local Means: Better quality, preserves details
    2. Bilateral Filter: Faster, preserves edges
    """
    if use_bilateral:
        return cv2.bilateralFilter(
            gray,
            d=CFG.bilateral_d,
            sigmaColor=CFG.bilateral_sigma_color,
            sigmaSpace=CFG.bilateral_sigma_space
        )
    else:
        return cv2.fastNlMeansDenoising(
            gray,
            h=CFG.denoise_h,
            templateWindowSize=CFG.denoise_template_window,
            searchWindowSize=CFG.denoise_search_window
        )


def apply_unsharp_mask(gray: np.ndarray, radius: float = None, amount: float = None,
                       threshold: int = None) -> np.ndarray:
    """Advanced sharpening using Unsharp Mask technique"""
    if radius is None:
        radius = CFG.sharpen_radius
    if amount is None:
        amount = CFG.sharpen_amount
    if threshold is None:
        threshold = CFG.sharpen_threshold

    blurred = cv2.GaussianBlur(gray, (0, 0), radius)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

    if threshold > 0:
        low_contrast_mask = np.abs(gray - blurred) < threshold
        np.copyto(sharpened, gray, where=low_contrast_mask)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_morphological_operations(gray: np.ndarray, operation: str = None) -> np.ndarray:
    """Apply morphological operations"""
    if operation is None:
        operation = CFG.morph_operation

    kernel = np.ones((CFG.morph_kernel_size, CFG.morph_kernel_size), np.uint8)

    operations = {
        'closing': cv2.MORPH_CLOSE,
        'opening': cv2.MORPH_OPEN,
        'dilation': cv2.MORPH_DILATE,
        'erosion': cv2.MORPH_ERODE,
        'gradient': cv2.MORPH_GRADIENT
    }

    if operation not in operations:
        return gray

    return cv2.morphologyEx(gray, operations[operation], kernel)


def apply_shadow_correction(gray: np.ndarray) -> np.ndarray:
    """Remove uneven lighting and shadows"""
    kernel_size = CFG.shadow_kernel_size
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    diff = cv2.subtract(background, gray)
    corrected = cv2.normalize(diff, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    corrected = 255 - corrected

    return corrected


def apply_gamma_correction(gray: np.ndarray, gamma: float = None) -> np.ndarray:
    """Adjust mid-tones without affecting whites/blacks"""
    if gamma is None:
        gamma = CFG.gamma_value

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in range(256)]).astype(np.uint8)

    return cv2.LUT(gray, table)


def apply_white_balance(gray: np.ndarray) -> np.ndarray:
    """Simple gray world white balance for grayscale images"""
    p2, p98 = np.percentile(gray, (2, 98))
    # SAFE GUARD: avoid division by zero on flat images
    if p98 <= p2:
        return gray.copy()
    stretched = np.clip((gray - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    return stretched


# ============================================================================
# CONTENT-AWARE PREPROCESSING
# ============================================================================

def preprocess_clean_document(gray: np.ndarray, initial_vol_full: float) -> Tuple[np.ndarray, List[str]]:
    """Minimal processing for high-quality documents"""
    processed = gray.copy()
    steps = []

    # Only light sharpening if actually needed
    if initial_vol_full < 250:
        processed = apply_unsharp_mask(processed, radius=1.0, amount=0.5)
        steps.append("light_sharpen")

    return processed, steps


def preprocess_medical_photo(gray: np.ndarray, initial_vol_full: float) -> Tuple[np.ndarray, List[str]]:
    """Preserve details for medical/wound images"""
    processed = gray.copy()
    steps = []

    # Light CLAHE only if very low contrast
    if gray.std() < 35:
        processed = apply_clahe(processed)
        steps.append("clahe")

    # NO denoising - kills medical texture
    # NO shadow correction - real lighting matters

    # Very gentle sharpening only if blurry
    if initial_vol_full < 150:
        processed = apply_unsharp_mask(processed, radius=1.2, amount=0.6)
        steps.append("gentle_sharpen")

    return processed, steps


def preprocess_dark_scene(gray: np.ndarray, initial_vol_full: float) -> Tuple[np.ndarray, List[str]]:
    """Preserve darkness as content, enhance details only"""
    processed = gray.copy()
    steps = []

    # DO NOT brighten - darkness is content
    # DO NOT apply shadow correction

    # Light local contrast enhancement
    processed = apply_clahe(processed)
    steps.append("clahe")

    # Sharpen details
    processed = apply_unsharp_mask(processed, radius=1.5, amount=0.8)
    steps.append("sharpen")

    return processed, steps


def preprocess_underexposed_phone(gray: np.ndarray, initial_vol_full: float) -> Tuple[np.ndarray, List[str]]:
    """Full pipeline for poor quality phone photos"""
    processed = gray.copy()
    steps = []

    # Shadow correction for uneven lighting
    if CFG.enable_shadow_correction:
        processed = apply_shadow_correction(processed)
        steps.append("shadow_correction")

    # Gamma correction for brightness
    if processed.mean() < 100:
        processed = apply_gamma_correction(processed, 1.3)
        steps.append("gamma_1.3")

    # White balance
    if CFG.enable_white_balance and (processed.max() - processed.min()) < 150:
        processed = apply_white_balance(processed)
        steps.append("white_balance")

    # CLAHE for contrast
    processed = apply_clahe(processed)
    steps.append("clahe")

    # Denoise
    if CFG.enable_denoise:
        processed = apply_advanced_denoise(processed, use_bilateral=True)
        steps.append("bilateral_denoise")

    # Morphology if low contrast
    if CFG.enable_morphology and processed.std() < 40:
        processed = apply_morphological_operations(processed, "closing")
        steps.append("morph_closing")

    # Sharpen
    if CFG.enable_sharpen:
        processed = apply_unsharp_mask(processed)
        steps.append("unsharp_mask")

    return processed, steps


def preprocess_general(gray: np.ndarray, initial_vol_full: float) -> Tuple[np.ndarray, List[str]]:
    """Conservative processing for unknown content"""
    processed = gray.copy()
    steps = []

    # Only apply corrections if clearly needed
    if gray.mean() < 80:
        processed = apply_gamma_correction(processed, 1.2)
        steps.append("gamma_1.2")

    if gray.std() < 40:
        processed = apply_clahe(processed)
        steps.append("clahe")

    # Light sharpening if blurry
    if initial_vol_full < 200:
        processed = apply_unsharp_mask(processed, radius=1.5, amount=0.8)
        steps.append("unsharp_mask")

    return processed, steps


def preprocess_image(gray_full: np.ndarray, initial_vol_full: float) -> Tuple[np.ndarray, List[str]]:
    """
    Content-aware preprocessing with quality safeguards
    - IMPORTANT FIX: use *full-resolution* VoL for both before/after in safeguards
    """
    # Detect image type (on full-res for stability)
    image_type = detect_image_type(gray_full, initial_vol_full)

    # Apply appropriate preprocessing
    if image_type == "clean_document":
        processed, steps = preprocess_clean_document(gray_full, initial_vol_full)
    elif image_type == "medical_photo":
        processed, steps = preprocess_medical_photo(gray_full, initial_vol_full)
    elif image_type == "dark_scene":
        processed, steps = preprocess_dark_scene(gray_full, initial_vol_full)
    elif image_type == "underexposed_phone":
        processed, steps = preprocess_underexposed_phone(gray_full, initial_vol_full)
    else:
        processed, steps = preprocess_general(gray_full, initial_vol_full)

    # Add image type detection to steps
    steps.insert(0, f"type:{image_type}")

    # Quality check: Did preprocessing hurt the image?
    final_vol_full = variance_of_laplacian(processed)

    # Check sharpness loss (FULL vs FULL)
    if final_vol_full < initial_vol_full * (1 - CFG.max_sharpness_loss):
        # Lost too much sharpness - return original
        return gray_full, ["preprocessing_skipped_sharpness_loss"]

    # Check for excessive clipping
    orig_clipped = int(np.sum(gray_full == 0) + np.sum(gray_full == 255))
    proc_clipped = int(np.sum(processed == 0) + np.sum(processed == 255))

    if proc_clipped > orig_clipped * CFG.max_clipping_ratio and proc_clipped > 1000:
        # Too much clipping - return original
        return gray_full, ["preprocessing_skipped_excessive_clipping"]

    return processed, steps


# ============================================================================
# QUALITY ASSESSMENT - TWO-TIER
# ============================================================================

@dataclass
class QualityResult:
    """Quality assessment result for single image"""
    original_path: str
    original_saved_path: str
    preprocessed_path: Optional[str]

    tier: str
    passed: bool

    vol_original: float
    vol_processed: float

    preprocessing_steps: List[str]
    image_type: str
    message: str
    suggestion: str

    resolution: str
    file_size_kb: int

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PDFQualityResult:
    """Aggregated PDF quality result"""
    pdf_path: str
    total_pages: int

    overall_tier: str
    passed: bool

    pass_pages: int
    warning_pages: int
    warning_page_numbers: List[int]

    avg_vol_original: float
    avg_vol_processed: float

    original_saved_path: str
    preprocessed_pages_dir: str

    message: str
    suggestion: str

    page_results: List[QualityResult]

    def to_dict(self) -> Dict:
        return {
            'pdf_path': self.pdf_path,
            'total_pages': self.total_pages,
            'overall_tier': self.overall_tier,
            'passed': self.passed,
            'pass_pages': self.pass_pages,
            'warning_pages': self.warning_pages,
            'warning_page_numbers': self.warning_page_numbers,
            'avg_vol_original': self.avg_vol_original,
            'avg_vol_processed': self.avg_vol_processed,
            'original_saved_path': self.original_saved_path,
            'preprocessed_pages_dir': self.preprocessed_pages_dir,
            'message': self.message,
            'suggestion': self.suggestion
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def determine_tier_absolute(vol: float) -> Tuple[str, str, str]:
    """Determine tier using absolute VoL values"""
    if vol >= CFG.pass_threshold:
        return (
            "PASS",
            "✅ Image quality is good.",
            "Recommended for use."
        )
    else:
        return (
            "WARNING",
            "⚠️  Image quality acceptable but not optimal.",
            "Optional: Retake for better quality."
        )


def assess_image_quality(
    image_path: str,
    output_dir: Optional[str] = None,
    save_images: bool = None,
    preprocessed_dir_override: Optional[str] = None  # NEW: allow directing preprocessed outputs
) -> QualityResult:
    """Main quality gate with content-aware preprocessing"""

    if output_dir is None:
        output_dir = CFG.default_output_dir
    if save_images is None:
        save_images = CFG.save_by_default

    original_dir = os.path.join(output_dir, CFG.original_subdir)
    preprocessed_dir = preprocessed_dir_override or os.path.join(output_dir, CFG.preprocessed_subdir)

    if save_images:
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(preprocessed_dir, exist_ok=True)

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert('L')

    orig_w, orig_h = img.size
    long_side = max(orig_w, orig_h)

    filename = Path(image_path).name
    file_stem = Path(image_path).stem

    if long_side < CFG.min_resolution:
        return QualityResult(
            original_path=image_path,
            original_saved_path="",
            preprocessed_path=None,
            tier="WARNING",
            passed=True,
            vol_original=0.0,
            vol_processed=0.0,
            preprocessing_steps=[],
            image_type="low_resolution",
            message=f"⚠️  Resolution low ({long_side}px)",
            suggestion=f"Recommended minimum: {CFG.min_resolution}px. Consider retaking.",
            resolution=f"{orig_w}x{orig_h}",
            file_size_kb=int(os.path.getsize(image_path) / 1024)
        )

    # Create downscaled copy for analysis if needed
    scale = min(1.0, CFG.analysis_size / long_side)
    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_small = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        new_w, new_h = orig_w, orig_h
        img_small = img

    gray_small = np.asarray(img_small)
    gray_full = np.asarray(img)

    # IMPORTANT FIX: compute VoL on both small and full to use consistently
    vol_original_small = variance_of_laplacian(gray_small)
    vol_original_full = variance_of_laplacian(gray_full)

    # Preprocess on FULL and run safeguards vs FULL
    preprocessed_full, preproc_steps = preprocess_image(gray_full, vol_original_full)

    # Create small version of processed image for tiering to stay consistent with original behavior
    if scale < 1.0:
        preprocessed_small = cv2.resize(preprocessed_full, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        preprocessed_small = preprocessed_full

    vol_processed_small = variance_of_laplacian(preprocessed_small)

    tier, message, suggestion = determine_tier_absolute(vol_processed_small)

    # Extract image type from steps
    image_type = "unknown"
    for step in list(preproc_steps):
        if step.startswith("type:"):
            image_type = step.split(":", 1)[1]
            preproc_steps.remove(step)
            break

    original_saved_path = ""
    preprocessed_path = None

    if save_images:
        original_saved_path = os.path.join(original_dir, filename)
        shutil.copy2(image_path, original_saved_path)

        preprocessed_filename = f"{file_stem}_preprocessed.png"
        preprocessed_path = os.path.join(preprocessed_dir, preprocessed_filename)
        Image.fromarray(preprocessed_full).save(preprocessed_path)

    return QualityResult(
        original_path=image_path,
        original_saved_path=original_saved_path,
        preprocessed_path=preprocessed_path,
        tier=tier,
        passed=True,
        vol_original=vol_original_small,
        vol_processed=vol_processed_small,
        preprocessing_steps=preproc_steps if preproc_steps else ["none"],
        image_type=image_type,
        message=message,
        suggestion=suggestion,
        resolution=f"{orig_w}x{orig_h}",
        file_size_kb=int(os.path.getsize(image_path) / 1024)
    )


def check_quality(image_path):
    """
    Wrapper for assess_image_quality to return simple dict format for API
    """
    try:
        result = assess_image_quality(image_path, save_images=False)
        
        # Load image to measure lighting separately
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert('L')
        gray = np.asarray(img)
        
        # Blur score from original VOL (higher VoL = sharper = LESS blur)
        vol_orig = result.vol_original
        blur_score = vol_orig / (vol_orig + CFG.pass_threshold) if vol_orig > 0 else 0.0
        
        # Lighting score from brightness and contrast analysis
        mean_brightness = gray.mean()
        std_brightness = gray.std()
        
        # Good lighting: balanced brightness (not too dark/bright) + good contrast
        lighting_score = 1.0
        if mean_brightness < 50 or mean_brightness > 220:  # Too dark or bright
            lighting_score *= 0.6
        if std_brightness < 30:  # Low contrast
            lighting_score *= 0.7
        
        # Overall quality: weighted combination (sharpness 60%, lighting 40%)
        quality_score = ((1-blur_score) * 0.6 + lighting_score * 0.4)
        
        return {
            'quality_score': round(min(1.0, max(0.0, quality_score)), 3),
            'blur_score': round(min(1.0, max(0.0, blur_score)), 3),
            'lighting_score': round(min(1.0, max(0.0, lighting_score)), 3)
        }
    except Exception as e:
        print(f"[ERROR] Quality check failed for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'quality_score': 0.0,
            'blur_score': 0.0,
            'lighting_score': 0.0,
            'error': str(e)
        }
