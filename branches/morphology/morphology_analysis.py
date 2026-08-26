from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from branches.morphology.morphology_segmentation import (
    MorphologyMaskResult,
    MorphologyParameters,
    detect_blemishes,
)
from core.result_schema import MethodResult, QUALITY_CATEGORIES, RIPENESS_CATEGORIES


EDGE_BAND_WIDTH_PIXELS = 8


@dataclass(frozen=True)
class RipenessBands:
    # These boundaries were selected from the validation data.
    extreme_dark_intensity_threshold: int = 50
    widespread_min_spread_percent: float = 97.91667
    low_edge_dark_share_percent: float = 25.52996
    unripe_max_extreme_dark_percent: float = 5.18711
    ripe_max_largest_component_solidity_percent: float = 35.704
    ripe_max_component_count: int = 6
    stable_dark_intensity_std: float = 26.26084
    overripe_min_edge_dark_percent: float = 34.68176
    overripe_min_extreme_dark_percent: float = 18.96101


@dataclass(frozen=True)
class QualityBands:
    # A small two-threshold rule was more reliable on the held-out test set.
    class_a_max_total_dark_percent: float = 25.75
    defect_min_total_dark_percent: float = 35.25
    extreme_dark_intensity_threshold: int = 50


@dataclass
class MorphologyAnalysisResult:
    method_result: MethodResult
    masks: MorphologyMaskResult
    predicted_category: str
    confidence_percent: float
    blemish_percentage: float
    surface_grade: str
    processing_time_ms: float
    total_dark_percentage: float
    dark_region_spread: float
    features: dict[str, Any]
    largest_dark_patch_percentage: float
    concentration_ratio: float
    dark_component_count: int
    extreme_dark_percentage: float
    largest_patch_mean_intensity: float | None
    quality_assessed: bool
    predicted_quality: str | None
    quality_confidence_percent: float | None


@dataclass
class MorphologyQualityResult:
    method_result: MethodResult
    masks: MorphologyMaskResult
    predicted_quality: str
    confidence_percent: float
    blemish_percentage: float
    processing_time_ms: float
    total_dark_percentage: float
    features: dict[str, Any]


def _validate(bands: RipenessBands | QualityBands) -> None:
    for name, value in vars(bands).items():
        if name.endswith("_count") and value < 0:
            raise ValueError(f"{name} cannot be negative.")
        limit = 255 if "intensity" in name else 100
        if ("percent" in name or "intensity" in name) and not 0 <= value <= limit:
            raise ValueError(f"{name} must be between 0 and {limit}.")
    if isinstance(bands, QualityBands) and bands.class_a_max_total_dark_percent >= bands.defect_min_total_dark_percent:
        raise ValueError("The Class_A boundary must be below the Defect boundary.")


def _extract_region_features(
    masks: MorphologyMaskResult,
    banana_mask: np.ndarray,
    extreme_dark_threshold: int,
) -> dict[str, float | int | None]:
    # Turn the detected dark mask into the numbers used by the rules.
    banana = np.asarray(banana_mask) > 0
    dark_mask = np.where(masks.blemish_mask > 0, 255, 0).astype(np.uint8)
    dark_mask[~banana] = 0
    banana_area = int(np.count_nonzero(banana))
    if not banana_area:
        raise ValueError("The banana mask is empty.")
    dark_area = int(np.count_nonzero(dark_mask))
    total, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    component_count = total - 1
    largest_label, largest_area = None, 0
    if component_count:
        offset = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_label, largest_area = offset + 1, int(stats[offset + 1, cv2.CC_STAT_AREA])
    greyscale = masks.greyscale_image
    patch_mean = None
    solidity = 0.0
    if largest_label is not None:
        pixels = greyscale[labels == largest_label]
        patch_mean = float(np.mean(pixels)) if pixels.size else None
        largest_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = float(cv2.contourArea(contour)) / hull_area if hull_area else 0.0
    distance = cv2.distanceTransform(banana.astype(np.uint8), cv2.DIST_L2, 5)
    edge = banana & (distance <= EDGE_BAND_WIDTH_PIXELS)
    edge_area = int(np.count_nonzero(edge))
    edge_dark_area = int(np.count_nonzero((dark_mask > 0) & edge))
    dark_pixels = greyscale[dark_mask > 0]
    return {
        "total_dark_percentage": float(masks.total_dark_percentage),
        "largest_dark_patch_percentage": 100.0 * largest_area / banana_area,
        "concentration_ratio": 100.0 * largest_area / dark_area if dark_area else 0.0,
        "dark_region_spread": float(masks.dark_region_spread),
        "dark_component_count": component_count,
        "extreme_dark_percentage": 100.0 * np.count_nonzero(
            (greyscale <= extreme_dark_threshold) & banana
        ) / banana_area,
        "largest_patch_mean_intensity": patch_mean,
        "edge_dark_percentage": 100.0 * edge_dark_area / edge_area if edge_area else 0.0,
        "edge_dark_share_percentage": 100.0 * edge_dark_area / dark_area if dark_area else 0.0,
        "dark_intensity_std": float(np.std(dark_pixels)) if dark_pixels.size else 0.0,
        "largest_component_solidity_percentage": 100.0 * solidity,
    }


def _classify(
    values: dict[str, float | int | None], bands: RipenessBands
) -> tuple[str, str]:
    # This is the main ripeness decision tree.
    spread = float(values["dark_region_spread"])
    components = int(values["dark_component_count"])
    extreme = float(values["extreme_dark_percentage"])
    edge_dark = float(values["edge_dark_percentage"])
    edge_share = float(values["edge_dark_share_percentage"])
    intensity_std = float(values["dark_intensity_std"])
    solidity = float(values["largest_component_solidity_percentage"])
    if spread <= bands.widespread_min_spread_percent:
        if edge_share <= bands.low_edge_dark_share_percent:
            return ("Unripe", "low_edge_low_extreme") if extreme <= bands.unripe_max_extreme_dark_percent else ("Rotten", "interior_damage")
        if solidity <= bands.ripe_max_largest_component_solidity_percent or components <= bands.ripe_max_component_count:
            return "Ripe", "edge_pattern_ripe"
        return "Rotten", "fragmented_interior_damage"
    if intensity_std > bands.stable_dark_intensity_std:
        return "Rotten", "widespread_irregular"
    if edge_dark > bands.overripe_min_edge_dark_percent:
        return "Overripe", "widespread_edge_dark"
    return ("Rotten", "widespread_low_extreme") if extreme <= bands.overripe_min_extreme_dark_percent else ("Overripe", "widespread_extreme_dark")


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rule_confidence(
    category: str, rule_key: str, values: dict[str, float | int | None], bands: RipenessBands
) -> float:
    # Confidence shows rule support; it does not change the chosen class.
    spread = float(values["dark_region_spread"])
    extreme = float(values["extreme_dark_percentage"])
    edge_dark = float(values["edge_dark_percentage"])
    edge_share = float(values["edge_dark_share_percentage"])
    intensity_std = float(values["dark_intensity_std"])
    if category == "Overripe":
        support = 0.5 * _clip01((spread - bands.widespread_min_spread_percent) / max(100.0 - bands.widespread_min_spread_percent, 1.0)) + 0.5 * _clip01((max(edge_dark, extreme) - min(bands.overripe_min_edge_dark_percent, bands.overripe_min_extreme_dark_percent)) / 100.0)
    elif category == "Rotten":
        support = _clip01((intensity_std - bands.stable_dark_intensity_std) / max(100.0 - bands.stable_dark_intensity_std, 1.0)) if rule_key == "widespread_irregular" else _clip01(extreme / 25.0)
    elif category == "Unripe":
        support = 0.5 * _clip01((bands.low_edge_dark_share_percent - edge_share) / max(bands.low_edge_dark_share_percent, 1.0)) + 0.5 * _clip01((bands.unripe_max_extreme_dark_percent - extreme) / max(bands.unripe_max_extreme_dark_percent, 1.0))
    else:
        support = _clip01(edge_share / 100.0)
    return 50.0 + 40.0 * support


def _classify_quality(values: dict[str, float | int | None], bands: QualityBands) -> tuple[str, str]:
    # Quality is only checked after the banana is predicted as Ripe.
    total = float(values["total_dark_percentage"])
    if total > bands.defect_min_total_dark_percent:
        return "Defect", "quality_high_total_dark"
    if total <= bands.class_a_max_total_dark_percent:
        return "Class_A", "quality_low_total_dark"
    return "Class_B", "quality_moderate_total_dark"


def _quality_confidence(
    quality: str,
    values: dict[str, float | int | None],
    bands: QualityBands,
) -> float:
    total = float(values["total_dark_percentage"])
    if quality == "Defect":
        support = _clip01((total - bands.defect_min_total_dark_percent) / max(100.0 - bands.defect_min_total_dark_percent, 1.0))
    elif quality == "Class_A":
        support = _clip01((bands.class_a_max_total_dark_percent - total) / max(bands.class_a_max_total_dark_percent, 1.0))
    else:
        midpoint = (bands.class_a_max_total_dark_percent + bands.defect_min_total_dark_percent) / 2.0
        support = 1.0 - _clip01(abs(total - midpoint) / max((bands.defect_min_total_dark_percent - bands.class_a_max_total_dark_percent) / 2.0, 1.0))
    return 50.0 + 40.0 * support


def _measure(
    rgb_image: np.ndarray, banana_mask: np.ndarray, parameters: MorphologyParameters, threshold: int
) -> tuple[MorphologyMaskResult, dict[str, float | int | None], float]:
    # Both app paths reuse the same dark-region measurements.
    start = perf_counter()
    masks = detect_blemishes(rgb_image=rgb_image, banana_mask=banana_mask, parameters=parameters)
    values = _extract_region_features(masks, banana_mask, threshold)
    return masks, values, (perf_counter() - start) * 1000.0


def _features(values: dict[str, float | int | None], **extra: Any) -> dict[str, Any]:
    return {
        "Total dark percentage (%)": round(float(values["total_dark_percentage"]), 4),
        "Largest dark patch (%)": round(float(values["largest_dark_patch_percentage"]), 4),
        "Concentration ratio (%)": round(float(values["concentration_ratio"]), 4),
        "Dark-region spread (%)": round(float(values["dark_region_spread"]), 4),
        "Dark component count": int(values["dark_component_count"]),
        "Extreme-dark percentage (%)": round(float(values["extreme_dark_percentage"]), 4),
        "Largest-patch mean intensity": None if values["largest_patch_mean_intensity"] is None else round(float(values["largest_patch_mean_intensity"]), 4),
        "Edge-dark percentage (%)": round(float(values["edge_dark_percentage"]), 4),
        "Edge-dark share (%)": round(float(values["edge_dark_share_percentage"]), 4),
        "Dark-intensity standard deviation": round(float(values["dark_intensity_std"]), 4),
        "Largest-component solidity (%)": round(float(values["largest_component_solidity_percentage"]), 4),
        **extra,
    }


def _method_result(
    category: str, confidence: float, time_ms: float, features: dict[str, Any], categories: tuple[str, ...], name: str
) -> MethodResult:
    scores = dict.fromkeys(categories, 0.0)
    scores[category] = confidence / 100.0
    return MethodResult(
        method_key="morphology", method_name=name, predicted_category=category,
        confidence_percent=round(confidence, 2), processing_time_ms=round(time_ms, 2),
        class_scores=scores, features=features, notes=[], is_placeholder=False,
    )


def analyse_morphology(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters | None = None,
    bands: RipenessBands | None = None,
    quality_bands: QualityBands | None = None,
) -> MorphologyAnalysisResult:
    # Normal app path: ripeness first, then quality for Ripe only.
    parameters, bands, quality_bands = parameters or MorphologyParameters(), bands or RipenessBands(), quality_bands or QualityBands()
    _validate(bands)
    _validate(quality_bands)
    masks, values, time_ms = _measure(rgb_image, banana_mask, parameters, bands.extreme_dark_intensity_threshold)
    category, rule_key = _classify(values, bands)
    confidence = _rule_confidence(category, rule_key, values, bands)
    assessed = category == "Ripe"
    quality, quality_confidence, quality_rule = None, None, "quality_not_assessed"
    if assessed:
        quality, quality_rule = _classify_quality(values, quality_bands)
        quality_confidence = _quality_confidence(quality, values, quality_bands)
    surface_grade = quality or "Not assessed"
    features = _features(
        values, **{"Quality assessed": assessed, "Quality class": quality or "Not assessed", "Quality confidence (%)": None if quality_confidence is None else round(quality_confidence, 4), "Quality decision rule": quality_rule, "Surface grade": surface_grade, "Decision rule": rule_key}
    )
    result = _method_result(category, confidence, time_ms, features, RIPENESS_CATEGORIES, "Morphology - Dark Region Pattern Analysis")
    return MorphologyAnalysisResult(
        result, masks, category, confidence, float(values["total_dark_percentage"]), surface_grade,
        time_ms, float(values["total_dark_percentage"]), float(values["dark_region_spread"]), features,
        float(values["largest_dark_patch_percentage"]), float(values["concentration_ratio"]),
        int(values["dark_component_count"]), float(values["extreme_dark_percentage"]),
        values["largest_patch_mean_intensity"], assessed, quality, quality_confidence,
    )


def analyse_morphology_quality(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters | None = None,
    quality_bands: QualityBands | None = None,
) -> MorphologyQualityResult:
    # This separate path is used when evaluation images are already known Ripe.
    parameters, quality_bands = parameters or MorphologyParameters(), quality_bands or QualityBands()
    _validate(quality_bands)
    masks, values, time_ms = _measure(rgb_image, banana_mask, parameters, quality_bands.extreme_dark_intensity_threshold)
    quality, rule_key = _classify_quality(values, quality_bands)
    confidence = _quality_confidence(quality, values, quality_bands)
    features = _features(values, **{"Known ripeness": "Ripe", "Quality class": quality, "Quality decision rule": rule_key})
    result = _method_result(quality, confidence, time_ms, features, QUALITY_CATEGORIES, "Morphology - Ripe Banana Quality Analysis")
    return MorphologyQualityResult(result, masks, quality, confidence, float(values["total_dark_percentage"]), time_ms, float(values["total_dark_percentage"]), features)
