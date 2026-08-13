from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from branches.hsv.hsv_segmentation import (
    HSVMaskResult,
    HSVParameters,
    segment_hsv_colours,
)
from core.result_schema import MethodResult


CATEGORIES = ("Unripe", "Ripe", "Overripe", "Rotten")


@dataclass(frozen=True)
class HSVRipenessBands:
    """Decision thresholds calibrated on the validation split. Re-run validation after changes, then freeze before test evaluation."""

    unripe_min_green_percent: float = 50.0
    unripe_max_brown_percent: float = 1.0

    ripe_min_yellow_percent: float = 75.0
    ripe_max_brown_percent: float = 7.5

    ripe_secondary_min_yellow_percent: float = 54.0
    ripe_secondary_max_brown_percent: float = 6.0

    overripe_min_dark_percent: float = 22.5
    overripe_max_other_percent: float = 23.5


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


def _validate_bands(bands: HSVRipenessBands) -> None:
    for name, value in vars(bands).items():
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be between 0 and 100.")

    if bands.ripe_secondary_min_yellow_percent > bands.ripe_min_yellow_percent:
        raise ValueError(
            "The secondary Ripe yellow threshold must not be higher than "
            "the primary Ripe yellow threshold."
        )

    if bands.ripe_secondary_max_brown_percent > bands.ripe_max_brown_percent:
        raise ValueError(
            "The secondary Ripe brown threshold must not be higher than "
            "the primary Ripe brown threshold."
        )


def _classify(
    green: float,
    yellow: float,
    brown: float,
    dark: float,
    other: float,
    bands: HSVRipenessBands,
) -> tuple[str, str]:
    """Apply validation-calibrated HSV rules in frozen priority order."""

    if (
        green >= bands.unripe_min_green_percent
        and brown <= bands.unripe_max_brown_percent
    ):
        return (
            "Unripe",
            f"Green peel covers {green:.2f}% of the banana while brown peel "
            f"is only {brown:.2f}%, matching the calibrated Unripe rule.",
        )

    if (
        yellow >= bands.ripe_min_yellow_percent
        and brown <= bands.ripe_max_brown_percent
    ):
        return (
            "Ripe",
            f"Yellow peel covers {yellow:.2f}% of the banana while brown peel "
            f"is {brown:.2f}%, matching the primary Ripe rule.",
        )

    if (
        dark >= bands.overripe_min_dark_percent
        and other <= bands.overripe_max_other_percent
    ):
        return (
            "Overripe",
            f"Dark peel covers {dark:.2f}% and unclassified peel is "
            f"{other:.2f}%, matching the calibrated Overripe rule.",
        )

    if (
        yellow >= bands.ripe_secondary_min_yellow_percent
        and brown <= bands.ripe_secondary_max_brown_percent
    ):
        return (
            "Ripe",
            f"Yellow peel covers {yellow:.2f}% while brown peel remains "
            f"low at {brown:.2f}%, matching the secondary Ripe rule.",
        )

    return (
        "Rotten",
        f"The peel contains {brown:.2f}% brown, {dark:.2f}% dark and "
        f"{other:.2f}% unclassified regions. It does not satisfy the "
        "calibrated Unripe, Ripe or Overripe rules, so it is classified Rotten.",
    )


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rule_confidence(
    category: str,
    green: float,
    yellow: float,
    brown: float,
    dark: float,
    other: float,
) -> float:
    """Return a 50-95 rule-support score, not a learned probability."""

    if category == "Unripe":
        support = green / 100.0
    elif category == "Ripe":
        support = yellow / 100.0
    elif category == "Overripe":
        support = dark / 100.0
    else:
        support = (brown + dark + other) / 100.0

    return 50.0 + 45.0 * _clip(support)


def _surface_grade(category: str, deteriorated: float) -> str:
    if category == "Rotten":
        return "Reject - Severe peel deterioration"
    if category == "Overripe":
        return "Grade C - High peel deterioration"
    if deteriorated < 10.0:
        return "Grade A - Low peel deterioration"
    return "Grade B - Moderate peel deterioration"


def analyse_hsv(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: HSVParameters | None = None,
    bands: HSVRipenessBands | None = None,
) -> HSVAnalysisResult:
    """Classify banana ripeness using HSV peel-colour percentages."""

    start_time = perf_counter()
    parameters = parameters or HSVParameters()
    bands = bands or HSVRipenessBands()
    _validate_bands(bands)

    masks = segment_hsv_colours(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
    )

    green = masks.green_percentage
    yellow = masks.yellow_percentage
    brown = masks.brown_percentage
    dark = masks.dark_percentage
    other = masks.other_percentage
    deteriorated = masks.deteriorated_percentage

    category, reason = _classify(
        green=green,
        yellow=yellow,
        brown=brown,
        dark=dark,
        other=other,
        bands=bands,
    )

    confidence = _rule_confidence(
        category=category,
        green=green,
        yellow=yellow,
        brown=brown,
        dark=dark,
        other=other,
    )
    surface_grade = _surface_grade(category, deteriorated)
    processing_time_ms = (perf_counter() - start_time) * 1000.0

    class_scores = {name: 0.0 for name in CATEGORIES}
    class_scores[category] = confidence / 100.0

    features = {
        "Green peel (%)": round(green, 4),
        "Yellow peel (%)": round(yellow, 4),
        "Brown peel (%)": round(brown, 4),
        "Dark peel (%)": round(dark, 4),
        "Brown + dark (%)": round(deteriorated, 4),
        "Other peel (%)": round(other, 4),
        "Surface grade": surface_grade,
        "Decision reason": reason,
    }

    method_result = MethodResult(
        method_key="hsv",
        method_name="HSV - Colour Segmentation",
        predicted_category=category,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(processing_time_ms, 2),
        class_scores=class_scores,
        features=features,
        notes=[
            "Classification uses green, yellow, brown, dark and unclassified peel percentages.",
            "Only pixels inside the shared banana-segmentation mask are analysed.",
            "Decision thresholds were calibrated on the validation split and must be frozen before test evaluation.",
            "Confidence is rule support, not a statistical probability.",
        ],
        is_placeholder=False,
    )

    return HSVAnalysisResult(
        method_result=method_result,
        masks=masks,
        predicted_category=category,
        confidence_percent=confidence,
        surface_grade=surface_grade,
        processing_time_ms=processing_time_ms,
        green_percentage=green,
        yellow_percentage=yellow,
        brown_percentage=brown,
        dark_percentage=dark,
        deteriorated_percentage=deteriorated,
        other_percentage=other,
        decision_reason=reason,
        features=features,
    )