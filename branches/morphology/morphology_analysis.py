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
from core.result_schema import (
    MethodResult,
    QUALITY_CATEGORIES,
    RIPENESS_CATEGORIES,
)


CATEGORIES = RIPENESS_CATEGORIES


@dataclass(frozen=True)
class RipenessBands:
    """Rules selected from the validation feature distributions."""

    high_total_dark_percent: float = 28.0
    low_concentration_percent: float = 83.0

    almost_no_extreme_dark_percent: float = 0.01
    unripe_max_component_count: int = 3
    ripe_max_component_count: int = 14

    very_small_patch_percent: float = 1.5
    small_patch_percent: float = 6.0
    localised_spread_percent: float = 48.0

    overripe_min_spread_percent: float = 98.0
    overripe_max_component_count: int = 19
    overripe_max_patch_mean_intensity: float = 68.0

    extreme_dark_intensity_threshold: int = 50

    # Compatibility aliases for older Streamlit code. The improved page does
    # not use these because the new classifier is not a four-band classifier.
    @property
    def unripe_max_total_dark_percent(self) -> float:
        return self.high_total_dark_percent

    @property
    def rotten_min_total_dark_percent(self) -> float:
        return self.high_total_dark_percent

    @property
    def overripe_min_total_dark_percent(self) -> float:
        return self.high_total_dark_percent


@dataclass(frozen=True)
class QualityBands:
    """Quality rules selected from the ripe-only quality validation data."""

    class_a_max_total_dark_percent: float = 23.50
    defect_min_total_dark_percent: float = 37.25
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
    decision_reason: str
    features: dict[str, Any]

    # Additional measurements retained for evaluation and the UI.
    largest_dark_patch_percentage: float
    concentration_ratio: float
    dark_component_count: int
    extreme_dark_percentage: float
    largest_patch_mean_intensity: float | None

    # Quality is deliberately conditional on a Ripe ripeness prediction.
    quality_assessed: bool
    predicted_quality: str | None
    quality_confidence_percent: float | None
    quality_reason: str

    @property
    def concentration_ratio_percentage(self) -> float:
        return self.concentration_ratio


@dataclass
class MorphologyQualityResult:
    """Quality-only output for a dataset containing known-ripe bananas."""

    method_result: MethodResult
    masks: MorphologyMaskResult
    predicted_quality: str
    confidence_percent: float
    blemish_percentage: float
    processing_time_ms: float
    total_dark_percentage: float
    quality_reason: str
    features: dict[str, Any]


def _validate_bands(bands: RipenessBands) -> None:
    percentage_fields = (
        "high_total_dark_percent",
        "low_concentration_percent",
        "almost_no_extreme_dark_percent",
        "very_small_patch_percent",
        "small_patch_percent",
        "localised_spread_percent",
        "overripe_min_spread_percent",
    )

    for name in percentage_fields:
        value = getattr(bands, name)
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be between 0 and 100.")

    if bands.very_small_patch_percent >= bands.small_patch_percent:
        raise ValueError(
            "very_small_patch_percent must be below small_patch_percent."
        )

    if bands.unripe_max_component_count < 0:
        raise ValueError("unripe_max_component_count cannot be negative.")
    if bands.ripe_max_component_count < bands.unripe_max_component_count:
        raise ValueError(
            "ripe_max_component_count must not be below the Unripe limit."
        )
    if bands.overripe_max_component_count < 0:
        raise ValueError("overripe_max_component_count cannot be negative.")
    if not 0 <= bands.overripe_max_patch_mean_intensity <= 255:
        raise ValueError(
            "overripe_max_patch_mean_intensity must be between 0 and 255."
        )
    if not 0 <= bands.extreme_dark_intensity_threshold <= 255:
        raise ValueError(
            "extreme_dark_intensity_threshold must be between 0 and 255."
        )


def _validate_quality_bands(bands: QualityBands) -> None:
    if not 0.0 <= bands.class_a_max_total_dark_percent <= 100.0:
        raise ValueError(
            "class_a_max_total_dark_percent must be between 0 and 100."
        )
    if not 0.0 <= bands.defect_min_total_dark_percent <= 100.0:
        raise ValueError(
            "defect_min_total_dark_percent must be between 0 and 100."
        )
    if (
        bands.class_a_max_total_dark_percent
        >= bands.defect_min_total_dark_percent
    ):
        raise ValueError(
            "The Class_A boundary must be below the Defect boundary."
        )
    if not 0 <= bands.extreme_dark_intensity_threshold <= 255:
        raise ValueError(
            "extreme_dark_intensity_threshold must be between 0 and 255."
        )


def _extract_region_features(
    masks: MorphologyMaskResult,
    banana_mask: np.ndarray,
    extreme_dark_threshold: int,
) -> dict[str, float | int | None]:
    banana = np.asarray(banana_mask) > 0
    dark_mask = np.where(masks.blemish_mask > 0, 255, 0).astype(np.uint8)
    dark_mask[~banana] = 0

    banana_area = int(np.count_nonzero(banana))
    dark_area = int(np.count_nonzero(dark_mask))

    if banana_area == 0:
        raise ValueError("The banana mask is empty.")

    component_total, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_mask,
        connectivity=8,
    )
    component_count = max(0, component_total - 1)

    largest_label = None
    largest_area = 0

    if component_count:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_offset = int(np.argmax(component_areas))
        largest_label = largest_offset + 1
        largest_area = int(component_areas[largest_offset])

    largest_patch_percentage = 100.0 * largest_area / banana_area
    concentration_ratio = (
        100.0 * largest_area / dark_area if dark_area else 0.0
    )

    greyscale = masks.greyscale_image
    extreme_dark_area = int(
        np.count_nonzero((greyscale <= extreme_dark_threshold) & banana)
    )
    extreme_dark_percentage = 100.0 * extreme_dark_area / banana_area

    patch_mean_intensity = None
    if largest_label is not None:
        patch_pixels = greyscale[labels == largest_label]
        if patch_pixels.size:
            patch_mean_intensity = float(np.mean(patch_pixels))

    return {
        "total_dark_percentage": float(masks.total_dark_percentage),
        "largest_dark_patch_percentage": largest_patch_percentage,
        "concentration_ratio": concentration_ratio,
        "dark_region_spread": float(masks.dark_region_spread),
        "dark_component_count": component_count,
        "extreme_dark_percentage": extreme_dark_percentage,
        "largest_patch_mean_intensity": patch_mean_intensity,
    }


def _classify(
    values: dict[str, float | int | None],
    bands: RipenessBands,
) -> tuple[str, str, str]:
    total_dark = float(values["total_dark_percentage"])
    largest_patch = float(values["largest_dark_patch_percentage"])
    concentration = float(values["concentration_ratio"])
    spread = float(values["dark_region_spread"])
    components = int(values["dark_component_count"])
    extreme_dark = float(values["extreme_dark_percentage"])
    patch_mean_value = values["largest_patch_mean_intensity"]
    patch_mean = 255.0 if patch_mean_value is None else float(patch_mean_value)

    # Later-stage bananas are separated first. Validation showed that
    # Overripe samples normally contain widespread connected darkening,
    # whereas Rotten samples are more variable and fragmented.
    if total_dark > bands.high_total_dark_percent:
        if (
            spread >= bands.overripe_min_spread_percent
            and components <= bands.overripe_max_component_count
            and patch_mean <= bands.overripe_max_patch_mean_intensity
        ):
            return (
                "Overripe",
                "late_widespread",
                f"Darkness is widespread ({spread:.2f}% spread), the largest "
                f"patch is strongly dark ({patch_mean:.2f}/255), and the "
                f"mask contains {components} connected dark regions.",
            )

        return (
            "Rotten",
            "late_irregular",
            f"Total darkness is high ({total_dark:.2f}%), but its spread, "
            "fragmentation, or patch intensity does not match the validated "
            "widespread Overripe pattern.",
        )

    # When the largest patch does not dominate the dark mask, component count
    # and extreme darkness are the most useful low-stage separators.
    if concentration <= bands.low_concentration_percent:
        if (
            extreme_dark <= bands.almost_no_extreme_dark_percent
            and components <= bands.unripe_max_component_count
        ):
            return (
                "Unripe",
                "low_sparse_unripe",
                "Almost no extremely dark peel is present and only "
                f"{components} dark components were retained.",
            )

        if (
            extreme_dark > bands.almost_no_extreme_dark_percent
            and components <= bands.ripe_max_component_count
        ):
            return (
                "Ripe",
                "low_moderate_ripe",
                f"Darkness remains below {bands.high_total_dark_percent:.1f}% "
                f"with {components} retained components.",
            )

        return (
            "Rotten",
            "low_fragmented_rotten",
            f"Although total darkness is below {bands.high_total_dark_percent:.1f}%, "
            f"the mask is unusually fragmented ({components} components).",
        )

    # High concentration at a low total-dark level is interpreted using the
    # physical size and spread of the dominant patch.
    if largest_patch <= bands.very_small_patch_percent:
        return (
            "Unripe",
            "dominant_tiny_unripe",
            f"The dominant dark patch covers only {largest_patch:.2f}% of "
            "the banana.",
        )

    if largest_patch <= bands.small_patch_percent:
        return (
            "Ripe",
            "dominant_small_ripe",
            f"The dominant patch is limited to {largest_patch:.2f}% of the "
            "banana surface.",
        )

    if spread <= bands.localised_spread_percent:
        return (
            "Unripe",
            "dominant_localised_unripe",
            f"The dominant patch is concentrated, but darkness reaches only "
            f"{spread:.2f}% of valid grid cells.",
        )

    return (
        "Rotten",
        "dominant_spread_rotten",
        f"A dominant patch is accompanied by {spread:.2f}% dark-region "
        "spread, which is inconsistent with the validated low-stage patterns.",
    )


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rule_confidence(
    category: str,
    rule_key: str,
    values: dict[str, float | int | None],
    bands: RipenessBands,
) -> float:
    """Return a conservative rule-support score, not a probability."""

    total_dark = float(values["total_dark_percentage"])
    largest_patch = float(values["largest_dark_patch_percentage"])
    concentration = float(values["concentration_ratio"])
    spread = float(values["dark_region_spread"])
    components = int(values["dark_component_count"])

    if category == "Overripe":
        spread_support = _clip01(
            (spread - bands.overripe_min_spread_percent)
            / max(100.0 - bands.overripe_min_spread_percent, 1.0)
        )
        darkness_support = _clip01(
            (total_dark - bands.high_total_dark_percent)
            / max(100.0 - bands.high_total_dark_percent, 1.0)
        )
        support = 0.65 * spread_support + 0.35 * darkness_support

    elif category == "Rotten":
        if rule_key == "late_irregular":
            support = _clip01(
                (total_dark - bands.high_total_dark_percent)
                / max(60.0 - bands.high_total_dark_percent, 1.0)
            )
        else:
            support = _clip01(components / 25.0)

    elif category == "Unripe":
        concentration_support = _clip01(
            (concentration - bands.low_concentration_percent)
            / max(100.0 - bands.low_concentration_percent, 1.0)
        )
        spread_support = _clip01(
            (bands.localised_spread_percent - spread)
            / max(bands.localised_spread_percent, 1.0)
        )
        support = 0.5 * concentration_support + 0.5 * spread_support

    else:
        patch_support = 1.0 - _clip01(
            abs(largest_patch - bands.small_patch_percent)
            / max(bands.small_patch_percent, 1.0)
        )
        total_support = 1.0 - _clip01(
            total_dark / max(bands.high_total_dark_percent, 1.0)
        )
        support = 0.5 * patch_support + 0.5 * total_support

    return 50.0 + 40.0 * support


def _classify_quality(
    values: dict[str, float | int | None],
    bands: QualityBands,
) -> tuple[str, str, str]:
    """Classify a Ripe banana into the three quality-dataset labels."""

    total_dark = float(values["total_dark_percentage"])
    if total_dark > bands.defect_min_total_dark_percent:
        return (
            "Defect",
            "quality_high_total_dark",
            f"Total dark area is {total_dark:.2f}%, above the validated "
            f"Defect boundary of {bands.defect_min_total_dark_percent:.2f}%.",
        )

    if total_dark <= bands.class_a_max_total_dark_percent:
        return (
            "Class_A",
            "quality_low_total_dark",
            f"Total dark area is {total_dark:.2f}%, at or below the "
            f"validated Class_A boundary of "
            f"{bands.class_a_max_total_dark_percent:.2f}%.",
        )

    return (
        "Class_B",
        "quality_moderate_total_dark",
        f"Total dark area ({total_dark:.2f}%) lies between the validated "
        "Class_A and Defect boundaries.",
    )


def _quality_rule_confidence(
    quality: str,
    values: dict[str, float | int | None],
    bands: QualityBands,
) -> float:
    """Return conservative support for the selected quality rule."""

    total_dark = float(values["total_dark_percentage"])
    if quality == "Defect":
        support = _clip01(
            (total_dark - bands.defect_min_total_dark_percent)
            / max(100.0 - bands.defect_min_total_dark_percent, 1.0)
        )
    elif quality == "Class_A":
        support = _clip01(
            (bands.class_a_max_total_dark_percent - total_dark)
            / max(bands.class_a_max_total_dark_percent, 1.0)
        )
    else:
        midpoint = (
            bands.class_a_max_total_dark_percent
            + bands.defect_min_total_dark_percent
        ) / 2.0
        half_width = max(
            (bands.defect_min_total_dark_percent
             - bands.class_a_max_total_dark_percent)
            / 2.0,
            1.0,
        )
        support = 1.0 - _clip01(
            abs(total_dark - midpoint) / half_width
        )

    return 50.0 + 40.0 * support


def analyse_morphology(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters | None = None,
    bands: RipenessBands | None = None,
    quality_bands: QualityBands | None = None,
) -> MorphologyAnalysisResult:
    """Classify ripeness and conditionally grade a Ripe banana's quality."""

    start_time = perf_counter()
    parameters = parameters or MorphologyParameters()
    bands = bands or RipenessBands()
    quality_bands = quality_bands or QualityBands()
    _validate_bands(bands)
    _validate_quality_bands(quality_bands)

    masks = detect_blemishes(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
    )

    values = _extract_region_features(
        masks=masks,
        banana_mask=banana_mask,
        extreme_dark_threshold=bands.extreme_dark_intensity_threshold,
    )
    category, rule_key, reason = _classify(values, bands)
    confidence = _rule_confidence(category, rule_key, values, bands)

    quality_assessed = category == "Ripe"
    predicted_quality = None
    quality_confidence = None
    quality_rule_key = "quality_not_assessed"

    if quality_assessed:
        predicted_quality, quality_rule_key, quality_reason = _classify_quality(
            values,
            quality_bands,
        )
        quality_confidence = _quality_rule_confidence(
            predicted_quality,
            values,
            quality_bands,
        )
        surface_grade = predicted_quality
    else:
        quality_reason = (
            f"Quality assessment was skipped because the predicted ripeness "
            f"category is {category}. Only Ripe bananas are quality graded."
        )
        surface_grade = "Not assessed"

    processing_time_ms = (perf_counter() - start_time) * 1000.0

    patch_mean = values["largest_patch_mean_intensity"]
    features = {
        "Total dark percentage (%)": round(
            float(values["total_dark_percentage"]), 4
        ),
        "Largest dark patch (%)": round(
            float(values["largest_dark_patch_percentage"]), 4
        ),
        "Concentration ratio (%)": round(
            float(values["concentration_ratio"]), 4
        ),
        "Dark-region spread (%)": round(
            float(values["dark_region_spread"]), 4
        ),
        "Dark component count": int(values["dark_component_count"]),
        "Extreme-dark percentage (%)": round(
            float(values["extreme_dark_percentage"]), 4
        ),
        "Largest-patch mean intensity": (
            None if patch_mean is None else round(float(patch_mean), 4)
        ),
        "Quality assessed": quality_assessed,
        "Quality class": predicted_quality or "Not assessed",
        "Quality confidence (%)": (
            None
            if quality_confidence is None
            else round(float(quality_confidence), 4)
        ),
        "Quality decision rule": quality_rule_key,
        "Quality decision reason": quality_reason,
        # Retained so older consumers that read this feature do not break.
        "Surface grade": surface_grade,
        "Decision rule": rule_key,
        "Decision reason": reason,
    }

    class_scores = {name: 0.0 for name in CATEGORIES}
    class_scores[category] = confidence / 100.0

    method_result = MethodResult(
        method_key="morphology",
        method_name="Morphology - Dark Region Pattern Analysis",
        predicted_category=category,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(processing_time_ms, 2),
        class_scores=class_scores,
        features=features,
        notes=[
            "All measurements come from thresholded and morphologically cleaned dark regions.",
            "Decision thresholds were selected using the validation split and must now be frozen.",
            "Confidence is deterministic rule support, not a learned probability.",
            "Morphology remains weakest for visually clean Unripe versus Ripe bananas.",
            "Quality is assessed only after the ripeness result is Ripe.",
            "The quality stage reuses the existing morphology measurements; it does not process the image twice.",
        ],
        is_placeholder=False,
    )

    return MorphologyAnalysisResult(
        method_result=method_result,
        masks=masks,
        predicted_category=category,
        confidence_percent=confidence,
        blemish_percentage=float(values["total_dark_percentage"]),
        surface_grade=surface_grade,
        processing_time_ms=processing_time_ms,
        total_dark_percentage=float(values["total_dark_percentage"]),
        dark_region_spread=float(values["dark_region_spread"]),
        decision_reason=reason,
        features=features,
        largest_dark_patch_percentage=float(
            values["largest_dark_patch_percentage"]
        ),
        concentration_ratio=float(values["concentration_ratio"]),
        dark_component_count=int(values["dark_component_count"]),
        extreme_dark_percentage=float(values["extreme_dark_percentage"]),
        largest_patch_mean_intensity=(
            None if patch_mean is None else float(patch_mean)
        ),
        quality_assessed=quality_assessed,
        predicted_quality=predicted_quality,
        quality_confidence_percent=quality_confidence,
        quality_reason=quality_reason,
    )


def analyse_morphology_quality(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters | None = None,
    quality_bands: QualityBands | None = None,
) -> MorphologyQualityResult:
    """
    Grade a banana whose ripeness is already known to be Ripe.

    This path intentionally skips the four-class ripeness classifier. It is
    used by the quality-dataset evaluation, where every source image is a
    ripe banana and the only target is Class_A, Class_B, or Defect.
    """

    start_time = perf_counter()
    parameters = parameters or MorphologyParameters()
    quality_bands = quality_bands or QualityBands()
    _validate_quality_bands(quality_bands)

    masks = detect_blemishes(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
    )
    values = _extract_region_features(
        masks=masks,
        banana_mask=banana_mask,
        extreme_dark_threshold=(
            quality_bands.extreme_dark_intensity_threshold
        ),
    )
    quality, rule_key, reason = _classify_quality(values, quality_bands)
    confidence = _quality_rule_confidence(quality, values, quality_bands)
    processing_time_ms = (perf_counter() - start_time) * 1000.0

    patch_mean = values["largest_patch_mean_intensity"]
    features = {
        "Known ripeness": "Ripe",
        "Total dark percentage (%)": round(
            float(values["total_dark_percentage"]), 4
        ),
        "Largest dark patch (%)": round(
            float(values["largest_dark_patch_percentage"]), 4
        ),
        "Concentration ratio (%)": round(
            float(values["concentration_ratio"]), 4
        ),
        "Dark-region spread (%)": round(
            float(values["dark_region_spread"]), 4
        ),
        "Dark component count": int(values["dark_component_count"]),
        "Extreme-dark percentage (%)": round(
            float(values["extreme_dark_percentage"]), 4
        ),
        "Largest-patch mean intensity": (
            None if patch_mean is None else round(float(patch_mean), 4)
        ),
        "Quality class": quality,
        "Quality decision rule": rule_key,
        "Quality decision reason": reason,
    }

    class_scores = {name: 0.0 for name in QUALITY_CATEGORIES}
    class_scores[quality] = confidence / 100.0

    method_result = MethodResult(
        method_key="morphology",
        method_name="Morphology - Ripe Banana Quality Analysis",
        predicted_category=quality,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(processing_time_ms, 2),
        class_scores=class_scores,
        features=features,
        notes=[
            "The input is treated as Ripe because the quality dataset "
            "contains only ripe bananas.",
            "No ripeness category is predicted in this analysis path.",
            "Quality is determined from morphologically cleaned dark-area "
            "measurements.",
            "Confidence is deterministic rule support, not a learned "
            "probability.",
        ],
        is_placeholder=False,
    )

    return MorphologyQualityResult(
        method_result=method_result,
        masks=masks,
        predicted_quality=quality,
        confidence_percent=confidence,
        blemish_percentage=float(values["total_dark_percentage"]),
        processing_time_ms=processing_time_ms,
        total_dark_percentage=float(values["total_dark_percentage"]),
        quality_reason=reason,
        features=features,
    )
