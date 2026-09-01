from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from branches.hsv.hsv_segmentation import (
    HSVMaskResult,
    HSVParameters,
    segment_hsv_colours,
)
from core.result_schema import (
    MethodResult,
    QUALITY_CATEGORIES,
    RIPENESS_CATEGORIES,
)


@dataclass(frozen=True)
class HSVRipenessBands:
    # Ripeness thresholds
    unripe_min_green_percent: float = 50.0
    unripe_max_brown_percent: float = 1.0
    ripe_min_yellow_percent: float = 75.0
    ripe_max_brown_percent: float = 7.5
    ripe_secondary_min_yellow_percent: float = 54.0
    ripe_secondary_max_brown_percent: float = 6.0
    overripe_min_dark_percent: float = 22.5
    overripe_max_other_percent: float = 23.5


@dataclass(frozen=True)
class HSVQualityBands:
    # Quality thresholds
    defect_min_deteriorated_percent: float = 35.9685
    class_b_max_brown_percent: float = 15.0296
    class_b_max_yellow_deterioration_ratio: float = 6.0109


@dataclass
class HSVAnalysisResult:
    method_result: MethodResult
    masks: HSVMaskResult
    predicted_category: str
    confidence_percent: float
    surface_grade: str
    processing_time_ms: float
    green_percentage: float
    yellow_percentage: float
    brown_percentage: float
    dark_percentage: float
    deteriorated_percentage: float
    other_percentage: float
    decision_reason: str
    features: dict[str, Any]
    quality_assessed: bool
    predicted_quality: str | None
    quality_confidence_percent: float | None
    quality_reason: str
    yellow_deterioration_ratio: float


@dataclass
class HSVQualityResult:
    method_result: MethodResult
    masks: HSVMaskResult
    predicted_quality: str
    confidence_percent: float
    processing_time_ms: float
    green_percentage: float
    yellow_percentage: float
    brown_percentage: float
    dark_percentage: float
    deteriorated_percentage: float
    other_percentage: float
    yellow_deterioration_ratio: float
    quality_reason: str
    features: dict[str, Any]


def _validate_ripeness(b: HSVRipenessBands) -> None:
    for name, value in vars(b).items():
        if not 0 <= value <= 100:
            raise ValueError(f"{name} must be between 0 and 100.")

    if b.ripe_secondary_min_yellow_percent > b.ripe_min_yellow_percent:
        raise ValueError("Secondary Ripe yellow threshold cannot exceed primary.")

    if b.ripe_secondary_max_brown_percent > b.ripe_max_brown_percent:
        raise ValueError("Secondary Ripe brown threshold cannot exceed primary.")


def _validate_quality(b: HSVQualityBands) -> None:
    if not 0 <= b.defect_min_deteriorated_percent <= 100:
        raise ValueError("Defect threshold must be between 0 and 100.")
    if not 0 <= b.class_b_max_brown_percent <= 100:
        raise ValueError("Class_B brown threshold must be between 0 and 100.")
    if b.class_b_max_yellow_deterioration_ratio <= 0:
        raise ValueError("Class_B ratio must be above zero.")


# Main ripeness rules
def _classify(green, yellow, brown, dark, other, b):
    if green >= b.unripe_min_green_percent and brown <= b.unripe_max_brown_percent:
        return "Unripe", f"Green {green:.2f}% and brown {brown:.2f}% match Unripe."

    if yellow >= b.ripe_min_yellow_percent and brown <= b.ripe_max_brown_percent:
        return "Ripe", f"Yellow {yellow:.2f}% and brown {brown:.2f}% match Ripe."

    if dark >= b.overripe_min_dark_percent and other <= b.overripe_max_other_percent:
        return "Overripe", f"Dark {dark:.2f}% and other {other:.2f}% match Overripe."

    if (
        yellow >= b.ripe_secondary_min_yellow_percent
        and brown <= b.ripe_secondary_max_brown_percent
    ):
        return "Ripe", f"Yellow {yellow:.2f}% and brown {brown:.2f}% match secondary Ripe."

    return "Rotten", "No Unripe, Ripe or Overripe rule was satisfied."


def _clip(value):
    return float(np.clip(value, 0.0, 1.0))


def _ripeness_confidence(category, green, yellow, brown, dark, other):
    if category == "Unripe":
        support = green / 100.0
    elif category == "Ripe":
        support = yellow / 100.0
    elif category == "Overripe":
        support = dark / 100.0
    else:
        support = (brown + dark + other) / 100.0

    return 50.0 + 45.0 * _clip(support)


def _ratio(yellow, deteriorated):
    return float("inf") if deteriorated <= 0 else yellow / deteriorated


# Main quality rules
def _classify_quality(yellow, brown, dark, b):
    deteriorated = brown + dark
    ratio = _ratio(yellow, deteriorated)

    if deteriorated > b.defect_min_deteriorated_percent:
        return (
            "Defect",
            "quality_high_deterioration",
            f"Brown + dark {deteriorated:.2f}% exceeds Defect threshold.",
            ratio,
        )

    if (
        brown <= b.class_b_max_brown_percent
        and ratio <= b.class_b_max_yellow_deterioration_ratio
    ):
        return (
            "Class_B",
            "quality_class_b_colour_balance",
            f"Brown {brown:.2f}% and ratio {ratio:.2f} match Class_B.",
            ratio,
        )

    return (
        "Class_A",
        "quality_class_a_colour_balance",
        "Non-defect colour balance does not match Class_B.",
        ratio,
    )


def _quality_confidence(quality, brown, dark, ratio, b):
    deteriorated = brown + dark

    if quality == "Defect":
        support = _clip(
            (deteriorated - b.defect_min_deteriorated_percent)
            / max(100.0 - b.defect_min_deteriorated_percent, 1.0)
        )

    elif quality == "Class_B":
        brown_support = _clip(
            (b.class_b_max_brown_percent - brown)
            / max(b.class_b_max_brown_percent, 1.0)
        )
        ratio_support = _clip(
            (b.class_b_max_yellow_deterioration_ratio - ratio)
            / max(b.class_b_max_yellow_deterioration_ratio, 1.0)
        )
        support = 0.5 * brown_support + 0.5 * ratio_support

    else:
        defect_margin = _clip(
            (b.defect_min_deteriorated_percent - deteriorated)
            / max(b.defect_min_deteriorated_percent, 1.0)
        )
        class_b_margin = max(
            _clip(
                (brown - b.class_b_max_brown_percent)
                / max(100.0 - b.class_b_max_brown_percent, 1.0)
            ),
            _clip(
                (ratio - b.class_b_max_yellow_deterioration_ratio)
                / max(b.class_b_max_yellow_deterioration_ratio, 1.0)
            ),
        )
        support = 0.65 * defect_margin + 0.35 * class_b_margin

    return 50.0 + 40.0 * support


def _features(masks: HSVMaskResult, **extra):
    ratio = _ratio(
        masks.yellow_percentage,
        masks.deteriorated_percentage,
    )

    return {
        "Green peel (%)": round(masks.green_percentage, 4),
        "Yellow peel (%)": round(masks.yellow_percentage, 4),
        "Brown peel (%)": round(masks.brown_percentage, 4),
        "Dark peel (%)": round(masks.dark_percentage, 4),
        "Brown + dark (%)": round(masks.deteriorated_percentage, 4),
        "Other peel (%)": round(masks.other_percentage, 4),
        "Yellow / deterioration ratio": (
            None if not np.isfinite(ratio) else round(ratio, 4)
        ),
        **extra,
    }


def _method_result(category, confidence, time_ms, categories, name, features):
    scores = dict.fromkeys(categories, 0.0)
    scores[category] = confidence / 100.0

    return MethodResult(
        method_key="hsv",
        method_name=name,
        predicted_category=category,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(time_ms, 2),
        class_scores=scores,
        features=features,
        notes=[],
        is_placeholder=False,
    )


def analyse_hsv(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: HSVParameters | None = None,
    bands: HSVRipenessBands | None = None,
    quality_bands: HSVQualityBands | None = None,
) -> HSVAnalysisResult:
    start = perf_counter()

    parameters = parameters or HSVParameters()
    bands = bands or HSVRipenessBands()
    quality_bands = quality_bands or HSVQualityBands()

    _validate_ripeness(bands)
    _validate_quality(quality_bands)

    masks = segment_hsv_colours(
        rgb_image,
        banana_mask,
        parameters,
    )

    green = masks.green_percentage
    yellow = masks.yellow_percentage
    brown = masks.brown_percentage
    dark = masks.dark_percentage
    other = masks.other_percentage
    deteriorated = masks.deteriorated_percentage

    category, reason = _classify(
        green,
        yellow,
        brown,
        dark,
        other,
        bands,
    )

    confidence = _ripeness_confidence(
        category,
        green,
        yellow,
        brown,
        dark,
        other,
    )

    quality_assessed = category == "Ripe"
    predicted_quality = None
    quality_confidence = None
    quality_rule = "quality_not_assessed"
    quality_reason = f"Quality not assessed because ripeness is {category}."
    ratio = _ratio(yellow, deteriorated)

    if quality_assessed:
        (
            predicted_quality,
            quality_rule,
            quality_reason,
            ratio,
        ) = _classify_quality(
            yellow,
            brown,
            dark,
            quality_bands,
        )

        quality_confidence = _quality_confidence(
            predicted_quality,
            brown,
            dark,
            ratio,
            quality_bands,
        )

    surface_grade = predicted_quality or "Not assessed"
    time_ms = (perf_counter() - start) * 1000.0

    features = _features(
        masks,
        **{
            "Quality assessed": quality_assessed,
            "Quality class": predicted_quality or "Not assessed",
            "Quality confidence (%)": (
                None if quality_confidence is None else round(quality_confidence, 4)
            ),
            "Quality decision rule": quality_rule,
            "Quality decision reason": quality_reason,
            "Surface grade": surface_grade,
            "Decision reason": reason,
        },
    )

    result = _method_result(
        category,
        confidence,
        time_ms,
        RIPENESS_CATEGORIES,
        "HSV - Colour Segmentation",
        features,
    )

    return HSVAnalysisResult(
        method_result=result,
        masks=masks,
        predicted_category=category,
        confidence_percent=confidence,
        surface_grade=surface_grade,
        processing_time_ms=time_ms,
        green_percentage=green,
        yellow_percentage=yellow,
        brown_percentage=brown,
        dark_percentage=dark,
        deteriorated_percentage=deteriorated,
        other_percentage=other,
        decision_reason=reason,
        features=features,
        quality_assessed=quality_assessed,
        predicted_quality=predicted_quality,
        quality_confidence_percent=quality_confidence,
        quality_reason=quality_reason,
        yellow_deterioration_ratio=ratio,
    )


def analyse_hsv_quality(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: HSVParameters | None = None,
    quality_bands: HSVQualityBands | None = None,
) -> HSVQualityResult:
    start = perf_counter()

    parameters = parameters or HSVParameters()
    quality_bands = quality_bands or HSVQualityBands()
    _validate_quality(quality_bands)

    masks = segment_hsv_colours(
        rgb_image,
        banana_mask,
        parameters,
    )

    quality, rule, reason, ratio = _classify_quality(
        masks.yellow_percentage,
        masks.brown_percentage,
        masks.dark_percentage,
        quality_bands,
    )

    confidence = _quality_confidence(
        quality,
        masks.brown_percentage,
        masks.dark_percentage,
        ratio,
        quality_bands,
    )

    time_ms = (perf_counter() - start) * 1000.0

    features = _features(
        masks,
        **{
            "Known ripeness": "Ripe",
            "Quality class": quality,
            "Quality decision rule": rule,
            "Quality decision reason": reason,
        },
    )

    result = _method_result(
        quality,
        confidence,
        time_ms,
        QUALITY_CATEGORIES,
        "HSV - Ripe Banana Quality Analysis",
        features,
    )

    return HSVQualityResult(
        method_result=result,
        masks=masks,
        predicted_quality=quality,
        confidence_percent=confidence,
        processing_time_ms=time_ms,
        green_percentage=masks.green_percentage,
        yellow_percentage=masks.yellow_percentage,
        brown_percentage=masks.brown_percentage,
        dark_percentage=masks.dark_percentage,
        deteriorated_percentage=masks.deteriorated_percentage,
        other_percentage=masks.other_percentage,
        yellow_deterioration_ratio=ratio,
        quality_reason=reason,
        features=features,
    )