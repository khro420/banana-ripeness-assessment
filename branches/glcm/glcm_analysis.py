"""Student-style GLCM texture analysis.

Steps:
1. Load image and banana mask
2. Convert to grayscale and quantise to 8 levels
3. Build GLCM for distances 1 and 2, angles 0/45/90/135
4. Average features across all GLCMs
5. Apply rule-based thresholds to get ripeness/quality
"""
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from core.result_schema import MethodResult


# Rule classes store the fixed thresholds learned from validation data.
# No machine learning. No test data used.

@dataclass(frozen=True)
class GLCMParameters:
    """Settings for GLCM computation."""
    levels: int = 8
    distances: tuple[int, ...] = (1, 2)
    angles: tuple[float, ...] = (
        0.0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
    )


@dataclass(frozen=True)
class GLCMRipenessBands:
    """Ripeness thresholds."""
    # Rotten: high contrast and correlation
    rotten_min_contrast: float = 0.194085
    rotten_min_correlation: float = 0.814267

    # Unripe: high homogeneity, low energy
    unripe_min_homogeneity: float = 0.959532
    unripe_max_energy: float = 0.427336

    # Overripe: low contrast and correlation
    overripe_max_contrast: float = 0.179175
    overripe_max_correlation: float = 0.879769


@dataclass(frozen=True)
class GLCMQualityBands:
    """Quality thresholds (only for Ripe bananas)."""
    # Defect: low homogeneity and low correlation
    defect_max_homogeneity: float = 0.896765
    defect_max_correlation: float = 0.924271

    # Class_A: low dissimilarity and high homogeneity
    class_a_max_dissimilarity: float = 0.199113
    class_a_min_homogeneity: float = 0.903952
    # Class_B is the fallback


@dataclass(frozen=True)
class GLCMAnalysisResult:
    """Result for ripeness analysis."""
    method_result: MethodResult
    predicted_category: str
    confidence_percent: float
    processing_time_ms: float
    contrast: float
    homogeneity: float
    energy: float
    correlation: float
    dissimilarity: float
    quality_assessed: bool
    predicted_quality: str | None
    quality_confidence_percent: float | None
    quality_reason: str
    decision_reason: str
    quantised_image: np.ndarray


@dataclass(frozen=True)
class GLCMQualityResult:
    """Result for quality analysis."""
    method_result: MethodResult
    predicted_quality: str
    confidence_percent: float
    processing_time_ms: float
    quality_reason: str
    features: dict[str, Any]


# ============================================================
# GLCM core computation
# ============================================================

def _quantise(greyscale, mask, levels):
    """Convert grayscale to fewer levels and remove background."""
    quantised = np.floor(greyscale.astype(np.float32) * levels / 256.0).astype(np.uint8)
    quantised = np.clip(quantised, 0, levels - 1)
    quantised[mask == 0] = 0
    return quantised


def _build_glcm(quantised, mask, levels, distance, angle):
    """Build one GLCM for a given distance and angle."""
    h, w = quantised.shape
    dx = int(round(np.cos(angle) * distance))
    dy = int(round(np.sin(angle) * distance))
    if dx == 0 and dy == 0:
        dx = distance

    # Slice corresponding source/neighbor pixels and count all pairs at
    # once. This is mathematically identical to the former pixel loop, but
    # makes full validation runs practical without changing GLCM values.
    source_y = slice(max(0, -dy), min(h, h - dy))
    source_x = slice(max(0, -dx), min(w, w - dx))
    neighbour_y = slice(max(0, dy), min(h, h + dy))
    neighbour_x = slice(max(0, dx), min(w, w + dx))
    source = quantised[source_y, source_x]
    neighbour = quantised[neighbour_y, neighbour_x]
    valid = (
        (mask[source_y, source_x] > 0)
        & (mask[neighbour_y, neighbour_x] > 0)
    )
    pairs = source[valid].astype(np.intp) * levels + neighbour[valid]
    counts = np.bincount(pairs, minlength=levels * levels)
    glcm = counts.reshape(levels, levels).astype(np.float64)
    glcm += glcm.T  # Match the explicitly symmetric updates in the loop.

    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def _calc_features(glcm):
    """Calculate 5 GLCM features from one GLCM matrix."""
    levels = glcm.shape[0]
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")

    contrast = float(np.sum((i - j) ** 2 * glcm))
    dissimilarity = float(np.sum(np.abs(i - j) * glcm))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(i - j))))
    energy = float(np.sum(glcm ** 2))

    # Correlation needs mean and std of row/column sums
    px = glcm.sum(axis=1)
    py = glcm.sum(axis=0)
    mean_x = float(np.sum(np.arange(levels) * px))
    mean_y = float(np.sum(np.arange(levels) * py))
    std_x = np.sqrt(float(np.sum((np.arange(levels) - mean_x) ** 2 * px)))
    std_y = np.sqrt(float(np.sum((np.arange(levels) - mean_y) ** 2 * py)))

    if std_x == 0 or std_y == 0:
        correlation = 1.0
    else:
        correlation = float(np.sum((i - mean_x) * (j - mean_y) * glcm) / (std_x * std_y))

    return contrast, dissimilarity, homogeneity, energy, correlation


def extract_glcm_features(rgb_image, banana_mask, parameters=None):
    """
    Extract averaged GLCM features from all distance/angle combinations.

    Returns (feature_vector, quantised_image)
    """
    parameters = parameters or GLCMParameters()

    rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    banana_mask = np.asarray(banana_mask)

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3).")
    if banana_mask.shape != rgb_image.shape[:2]:
        raise ValueError("Mask and image dimensions do not match.")

    grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    quantised = _quantise(grayscale, banana_mask, parameters.levels)

    feature_vals = []
    for dist in parameters.distances:
        for angle in parameters.angles:
            glcm = _build_glcm(quantised, banana_mask, parameters.levels, dist, angle)
            if glcm.sum() == 0:
                continue
            feature_vals.append(_calc_features(glcm))

    if not feature_vals:
        raise ValueError("Unable to construct a valid GLCM.")

    return np.mean(feature_vals, axis=0), quantised


# ============================================================
# Ripeness classification
# ============================================================

def _classify(contrast, homogeneity, energy, correlation, bands):
    """
    Classify ripeness using fixed thresholds.

    Priority order: Unripe -> Rotten -> Overripe -> Ripe (fallback)
    """
    # Unripe: high homogeneity, low energy
    if homogeneity >= bands.unripe_min_homogeneity and energy <= bands.unripe_max_energy:
        return ("Unripe",
                f"Unripe rule matched. Homogeneity={homogeneity:.6f} "
                f"(>= {bands.unripe_min_homogeneity:.6f}), "
                f"Energy={energy:.6f} (<= {bands.unripe_max_energy:.6f}).")

    # Rotten: high contrast and correlation
    if contrast >= bands.rotten_min_contrast and correlation >= bands.rotten_min_correlation:
        return ("Rotten",
                f"Rotten rule matched. Contrast={contrast:.6f} "
                f"(>= {bands.rotten_min_contrast:.6f}), "
                f"Correlation={correlation:.6f} "
                f"(>= {bands.rotten_min_correlation:.6f}).")

    # Overripe: low contrast and correlation
    if contrast <= bands.overripe_max_contrast and correlation <= bands.overripe_max_correlation:
        return ("Overripe",
                f"Overripe rule matched. Contrast={contrast:.6f} "
                f"(<= {bands.overripe_max_contrast:.6f}), "
                f"Correlation={correlation:.6f} "
                f"(<= {bands.overripe_max_correlation:.6f}).")

    # Ripe fallback
    return ("Ripe",
            f"Did not match Unripe/Rotten/Overripe rules. "
            f"Contrast={contrast:.6f}, Homogeneity={homogeneity:.6f}, "
            f"Energy={energy:.6f}, Correlation={correlation:.6f}. -> Ripe.")


def _rule_confidence(category, contrast, homogeneity, energy, correlation, bands):
    """
    Confidence based on how strongly features satisfy the rule.

    This is rule support (55-95%), NOT a model probability.
    """
    eps = 1e-9
    supports = []

    if category == "Unripe":
        # Closer to threshold = more confident
        supports.append(np.clip((homogeneity - bands.unripe_min_homogeneity)
                                 / max(1.0 - bands.unripe_min_homogeneity, eps), 0.0, 1.0))
        supports.append(np.clip((bands.unripe_max_energy - energy)
                                 / max(bands.unripe_max_energy, eps), 0.0, 1.0))
    elif category == "Rotten":
        supports.append(np.clip((contrast - bands.rotten_min_contrast)
                                 / max(1.0 - bands.rotten_min_contrast, eps), 0.0, 1.0))
        supports.append(np.clip((correlation - bands.rotten_min_correlation)
                                 / max(1.0 - bands.rotten_min_correlation, eps), 0.0, 1.0))
    elif category == "Overripe":
        supports.append(np.clip((bands.overripe_max_contrast - contrast)
                                 / max(bands.overripe_max_contrast, eps), 0.0, 1.0))
        supports.append(np.clip((bands.overripe_max_correlation - correlation)
                                 / max(bands.overripe_max_correlation, eps), 0.0, 1.0))
    else:
        return 70.0   # Ripe fallback gets fixed 70%

    support = float(np.mean(supports))
    confidence = 55.0 + 40.0 * support
    return float(np.clip(confidence, 55.0, 95.0))


# ============================================================
# Quality classification
# ============================================================

def _classify_quality(energy, correlation, homogeneity, dissimilarity, bands):
    """
    Classify quality for a known-ripe banana.

    Priority: Defect -> Class_A -> Class_B (fallback)
    """
    # Defect: low homogeneity and low correlation
    if homogeneity <= bands.defect_max_homogeneity and correlation <= bands.defect_max_correlation:
        return ("Defect",
                f"Defect rule matched. Homogeneity={homogeneity:.6f} "
                f"(<= {bands.defect_max_homogeneity:.6f}), "
                f"Correlation={correlation:.6f} "
                f"(<= {bands.defect_max_correlation:.6f}).",
                70.0)

    # Class_A: low dissimilarity and high homogeneity
    if dissimilarity <= bands.class_a_max_dissimilarity and homogeneity >= bands.class_a_min_homogeneity:
        return ("Class_A",
                f"Class_A rule matched. Dissimilarity={dissimilarity:.6f} "
                f"(<= {bands.class_a_max_dissimilarity:.6f}), "
                f"Homogeneity={homogeneity:.6f} "
                f"(>= {bands.class_a_min_homogeneity:.6f}).",
                70.0)

    # Class_B fallback
    return ("Class_B",
            f"Did not match Defect/Class_A rules. "
            f"Homogeneity={homogeneity:.6f}, Correlation={correlation:.6f}, "
            f"Dissimilarity={dissimilarity:.6f}, Energy={energy:.6f}. -> Class_B.",
            70.0)


# ============================================================
# Main entry points
# ============================================================

def analyse_image(rgb_image, banana_mask, parameters=None, bands=None, quality_bands=None):
    """
    Full ripeness + optional quality analysis.

    Quality is only assessed when ripeness is Ripe.
    """
    start = perf_counter()

    parameters = parameters or GLCMParameters()
    bands = bands or GLCMRipenessBands()
    quality_bands = quality_bands or GLCMQualityBands()

    # Step 1: extract GLCM features
    features, quantised = extract_glcm_features(rgb_image, banana_mask, parameters)
    contrast, dissimilarity, homogeneity, energy, correlation = features

    # Step 2: classify ripeness
    category, reason = _classify(contrast, homogeneity, energy, correlation, bands)
    confidence = _rule_confidence(category, contrast, homogeneity, energy, correlation, bands)

    # Step 3: quality assessment (only for Ripe)
    quality_assessed = (category == "Ripe")
    predicted_quality = quality_reason = ""
    quality_confidence = 0.0

    if quality_assessed:
        predicted_quality, quality_reason, quality_confidence = _classify_quality(
            energy, correlation, homogeneity, dissimilarity, quality_bands)

    elapsed_ms = (perf_counter() - start) * 1000.0

    # Build result
    class_scores = {name: 0.0 for name in ("Unripe", "Ripe", "Overripe", "Rotten")}
    class_scores[category] = confidence / 100.0

    feature_dict = {
        "Contrast": round(contrast, 6),
        "Homogeneity": round(homogeneity, 6),
        "Energy (ASM)": round(energy, 6),
        "Correlation": round(correlation, 6),
        "Dissimilarity": round(dissimilarity, 6),
        "Decision reason": reason,
    }
    if quality_assessed:
        feature_dict["Quality class"] = predicted_quality
        feature_dict["Quality reason"] = quality_reason

    method_result = MethodResult(
        method_key="glcm",
        method_name="GLCM - Texture Analysis",
        predicted_category=category,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(elapsed_ms, 2),
        class_scores=class_scores,
        features=feature_dict,
        notes=[
            "Texture features from segmented banana only.",
            "Fixed thresholds from validation set; no test data used.",
            "Confidence = rule support, not model probability.",
            "Quality assessed only when ripeness = Ripe.",
        ],
        is_placeholder=False,
    )

    return GLCMAnalysisResult(
        method_result=method_result,
        predicted_category=category,
        confidence_percent=confidence,
        processing_time_ms=elapsed_ms,
        contrast=contrast,
        homogeneity=homogeneity,
        energy=energy,
        correlation=correlation,
        dissimilarity=dissimilarity,
        quality_assessed=quality_assessed,
        predicted_quality=predicted_quality or None,
        quality_confidence_percent=quality_confidence or None,
        quality_reason=quality_reason,
        decision_reason=reason,
        quantised_image=quantised,
    )


def analyze_image(rgb_image, banana_mask, parameters=None, bands=None):
    """Backward-compatible alias (American spelling)."""
    return analyse_image(rgb_image, banana_mask, parameters=parameters, bands=bands)


def analyse_glcm(rgb_image, banana_mask, parameters=None, bands=None):
    """Public entry point used by the dashboard."""
    return analyse_image(rgb_image, banana_mask, parameters=parameters, bands=bands)


def analyse_glcm_quality(rgb_image, banana_mask, parameters=None, quality_bands=None):
    """
    Quality analysis only (ripeness must already be Ripe).

    Returns GLCMQualityResult.
    """
    start = perf_counter()

    parameters = parameters or GLCMParameters()
    quality_bands = quality_bands or GLCMQualityBands()

    features, _ = extract_glcm_features(rgb_image, banana_mask, parameters)
    contrast = float(features[0])
    dissimilarity = float(features[1])
    homogeneity = float(features[2])
    energy = float(features[3])
    correlation = float(features[4])

    predicted_quality, quality_reason, quality_confidence = _classify_quality(
        energy, correlation, homogeneity, dissimilarity, quality_bands)

    elapsed_ms = (perf_counter() - start) * 1000.0

    features_dict = {
        "Known ripeness": "Ripe",
        "Quality class": predicted_quality,
        "Quality decision rule": quality_reason,
        "Contrast": contrast,
        "Dissimilarity": dissimilarity,
        "Homogeneity": homogeneity,
        "Energy (ASM)": energy,
        "Correlation": correlation,
    }

    method_result = MethodResult(
        method_key="glcm",
        method_name="GLCM - Ripe Banana Quality Analysis",
        predicted_category=predicted_quality,
        confidence_percent=quality_confidence,
        processing_time_ms=round(elapsed_ms, 2),
        class_scores={
            name: quality_confidence / 100.0 if name == predicted_quality else 0.0
            for name in ("Class_A", "Class_B", "Defect")
        },
        features=features_dict,
        notes=[
            "Quality assessed for known-ripe bananas only.",
            "Fixed thresholds from validation set; no test data used.",
            "Confidence = rule support, not model probability.",
        ],
        is_placeholder=False,
    )

    return GLCMQualityResult(
        method_result=method_result,
        predicted_quality=predicted_quality,
        confidence_percent=quality_confidence,
        processing_time_ms=elapsed_ms,
        quality_reason=quality_reason,
        features=features_dict,
    )


def prepare_quantised_for_display(quantised_image):
    """Convert quantised image for display."""
    return (quantised_image.astype(np.float32) * 255.0 / 7.0).astype(np.uint8)
