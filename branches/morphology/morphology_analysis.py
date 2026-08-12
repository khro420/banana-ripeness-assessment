from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from branches.morphology.morphology_segmentation import (
    MorphologyMaskResult,
    MorphologyParameters,
    detect_blemishes,
)
from core.result_schema import MethodResult


CATEGORIES = ("Unripe", "Ripe", "Overripe", "Rotten")


@dataclass(frozen=True)
class RipenessBands:
    """Thresholds selected using the validation split and then frozen."""

    unripe_max_total_dark_percent: float = 9.0
    rotten_min_total_dark_percent: float = 26.0
    overripe_min_total_dark_percent: float = 42.5
    overripe_min_spread_percent: float = 100.0


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


def _validate_bands(bands: RipenessBands) -> None:
    for name, value in vars(bands).items():
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be between 0 and 100.")

    if not (
        bands.unripe_max_total_dark_percent
        < bands.rotten_min_total_dark_percent
        < bands.overripe_min_total_dark_percent
    ):
        raise ValueError("Total-dark thresholds must be in increasing order.")


def _classify(
    total_dark: float,
    spread: float,
    bands: RipenessBands,
) -> tuple[str, str]:
    """Apply the frozen two-feature rules in priority order."""

    if (
        total_dark >= bands.overripe_min_total_dark_percent
        and spread >= bands.overripe_min_spread_percent
    ):
        return (
            "Overripe",
            f"Darkness covers {total_dark:.2f}% of the peel and reaches "
            f"{spread:.2f}% of valid grid cells, so it is widespread.",
        )

    if total_dark >= bands.rotten_min_total_dark_percent:
        return (
            "Rotten",
            f"Darkness covers {total_dark:.2f}% of the peel but reaches only "
            f"{spread:.2f}% of valid grid cells, so it is less widespread.",
        )

    if total_dark <= bands.unripe_max_total_dark_percent:
        return (
            "Unripe",
            f"Only {total_dark:.2f}% of the visible peel is dark.",
        )

    return (
        "Ripe",
        f"The dark area is {total_dark:.2f}%, between the frozen Unripe and "
        "Rotten thresholds.",
    )


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rule_confidence(
    category: str,
    total_dark: float,
    spread: float,
    bands: RipenessBands,
) -> float:
    """Return a 50-95 rule-support score, not a probability."""

    if category == "Unripe":
        support = _clip(
            (bands.unripe_max_total_dark_percent - total_dark)
            / max(bands.unripe_max_total_dark_percent, 1.0)
        )
    elif category == "Ripe":
        lower = bands.unripe_max_total_dark_percent
        upper = bands.rotten_min_total_dark_percent
        half_width = (upper - lower) / 2.0
        support = _clip(
            1.0 - abs(total_dark - (lower + upper) / 2.0) / half_width
        )
    elif category == "Overripe":
        support = _clip(
            (total_dark - bands.overripe_min_total_dark_percent)
            / max(100.0 - bands.overripe_min_total_dark_percent, 1.0)
        )
    else:
        total_support = _clip(
            (total_dark - bands.rotten_min_total_dark_percent)
            / max(
                bands.overripe_min_total_dark_percent
                - bands.rotten_min_total_dark_percent,
                1.0,
            )
        )
        localisation_support = _clip(
            (bands.overripe_min_spread_percent - spread)
            / max(bands.overripe_min_spread_percent, 1.0)
        )
        support = 0.7 * total_support + 0.3 * localisation_support

    return 50.0 + 45.0 * support


def _surface_grade(
    category: str,
    total_dark: float,
    bands: RipenessBands,
) -> str:
    if category == "Rotten":
        return "Reject - Severe dark pattern"
    if total_dark <= bands.unripe_max_total_dark_percent:
        return "Grade A - Low dark area"
    if total_dark < bands.rotten_min_total_dark_percent:
        return "Grade B - Moderate dark area"
    if total_dark < bands.overripe_min_total_dark_percent:
        return "Grade C - High dark area"
    return "Reject - Extensive dark area"


def analyse_morphology(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters | None = None,
    bands: RipenessBands | None = None,
) -> MorphologyAnalysisResult:
    """Classify ripeness using morphology-derived darkness and spread."""

    start_time = perf_counter()
    parameters = parameters or MorphologyParameters()
    bands = bands or RipenessBands()
    _validate_bands(bands)

    masks = detect_blemishes(rgb_image, banana_mask, parameters)
    total_dark = masks.total_dark_percentage
    spread = masks.dark_region_spread

    category, reason = _classify(total_dark, spread, bands)
    confidence = _rule_confidence(category, total_dark, spread, bands)
    surface_grade = _surface_grade(category, total_dark, bands)
    processing_time_ms = (perf_counter() - start_time) * 1000.0

    class_scores = {name: 0.0 for name in CATEGORIES}
    class_scores[category] = confidence / 100.0

    features = {
        "Total dark percentage (%)": round(total_dark, 4),
        "Dark-region spread (%)": round(spread, 4),
        "Surface grade": surface_grade,
        "Decision reason": reason,
    }

    method_result = MethodResult(
        method_key="morphology",
        method_name="Morphology - Dark Region Analysis",
        predicted_category=category,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(processing_time_ms, 2),
        class_scores=class_scores,
        features=features,
        notes=[
            "Classification uses total dark percentage and dark-region spread.",
            "Thresholds were selected on the validation split and are frozen.",
            "Confidence is rule support, not a statistical probability.",
            "Dark morphology cannot reliably separate spotless green and yellow peel.",
        ],
        is_placeholder=False,
    )

    return MorphologyAnalysisResult(
        method_result=method_result,
        masks=masks,
        predicted_category=category,
        confidence_percent=confidence,
        blemish_percentage=total_dark,
        surface_grade=surface_grade,
        processing_time_ms=processing_time_ms,
        total_dark_percentage=total_dark,
        dark_region_spread=spread,
        decision_reason=reason,
        features=features,
    )
