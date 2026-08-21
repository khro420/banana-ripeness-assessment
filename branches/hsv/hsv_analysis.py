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


CATEGORIES = RIPENESS_CATEGORIES


@dataclass(frozen=True)
class HSVRipenessBands:
    """Ripeness rules calibrated on the ripeness validation split."""

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
    """
    HSV-only surface-quality rules calibrated from dataset/quality/valid.

    The quality branch uses only HSV-derived colour measurements:
    yellow percentage, brown percentage, dark percentage and
    deteriorated percentage (brown + dark).
    """

    defect_min_deteriorated_percent: float = 31.85
    class_b_max_brown_percent: float = 13.75
    class_b_max_yellow_deterioration_ratio: float = 7.65


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

    # Surface quality is deliberately assessed only when ripeness is Ripe.
    quality_assessed: bool
    predicted_quality: str | None
    quality_confidence_percent: float | None
    quality_reason: str
    yellow_deterioration_ratio: float


@dataclass
class HSVQualityResult:
    """HSV-only output for the known-ripe quality dataset."""

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


def _validate_quality_bands(bands: HSVQualityBands) -> None:
    percentage_fields = (
        "defect_min_deteriorated_percent",
        "class_b_max_brown_percent",
    )

    for name in percentage_fields:
        value = getattr(bands, name)
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be between 0 and 100.")

    if bands.class_b_max_yellow_deterioration_ratio <= 0.0:
        raise ValueError(
            "class_b_max_yellow_deterioration_ratio must be above zero."
        )


def _classify(
    green: float,
    yellow: float,
    brown: float,
    dark: float,
    other: float,
    bands: HSVRipenessBands,
) -> tuple[str, str]:
    """Apply validation-calibrated HSV ripeness rules in frozen order."""

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
    """Return a 50-95 ripeness rule-support score, not a probability."""

    if category == "Unripe":
        support = green / 100.0
    elif category == "Ripe":
        support = yellow / 100.0
    elif category == "Overripe":
        support = dark / 100.0
    else:
        support = (brown + dark + other) / 100.0

    return 50.0 + 45.0 * _clip(support)


def _yellow_deterioration_ratio(
    yellow: float,
    deteriorated: float,
) -> float:
    """Measure yellow coverage relative to brown + dark HSV coverage."""

    if deteriorated <= 0.0:
        return float("inf")

    return yellow / deteriorated


def _classify_quality(
    yellow: float,
    brown: float,
    dark: float,
    bands: HSVQualityBands,
) -> tuple[str, str, str, float]:
    """
    Classify the surface quality of a known-ripe banana using HSV only.

    No morphology, texture or connected-component measurements are used.
    """

    deteriorated = brown + dark
    ratio = _yellow_deterioration_ratio(
        yellow=yellow,
        deteriorated=deteriorated,
    )

    # Strong combined brown + dark coverage is the clearest Defect separator.
    if deteriorated > bands.defect_min_deteriorated_percent:
        return (
            "Defect",
            "quality_high_deterioration",
            f"Brown + dark peel covers {deteriorated:.2f}%, above the "
            f"validated Defect boundary of "
            f"{bands.defect_min_deteriorated_percent:.2f}%.",
            ratio,
        )

    # Within the non-defect group, Class_B validation images were best
    # separated by lower brown coverage together with the relative balance
    # between yellow peel and deteriorated peel.
    if (
        brown <= bands.class_b_max_brown_percent
        and ratio <= bands.class_b_max_yellow_deterioration_ratio
    ):
        return (
            "Class_B",
            "quality_class_b_colour_balance",
            f"Brown peel is {brown:.2f}% and the yellow-to-deterioration "
            f"ratio is {ratio:.2f}, matching the calibrated Class_B "
            f"HSV colour pattern.",
            ratio,
        )

    return (
        "Class_A",
        "quality_class_a_colour_balance",
        f"Brown + dark peel remains below the Defect boundary, and the "
        f"yellow/brown colour balance does not match the calibrated Class_B "
        f"pattern. The banana is classified Class_A.",
        ratio,
    )


def _quality_rule_confidence(
    quality: str,
    yellow: float,
    brown: float,
    dark: float,
    ratio: float,
    bands: HSVQualityBands,
) -> float:
    """Return conservative HSV quality rule support, not a probability."""

    deteriorated = brown + dark

    if quality == "Defect":
        support = _clip(
            (
                deteriorated
                - bands.defect_min_deteriorated_percent
            )
            / max(
                100.0 - bands.defect_min_deteriorated_percent,
                1.0,
            )
        )

    elif quality == "Class_B":
        brown_support = _clip(
            (
                bands.class_b_max_brown_percent
                - brown
            )
            / max(
                bands.class_b_max_brown_percent,
                1.0,
            )
        )

        ratio_support = _clip(
            (
                bands.class_b_max_yellow_deterioration_ratio
                - ratio
            )
            / max(
                bands.class_b_max_yellow_deterioration_ratio,
                1.0,
            )
        )

        support = 0.5 * brown_support + 0.5 * ratio_support

    else:
        defect_margin = _clip(
            (
                bands.defect_min_deteriorated_percent
                - deteriorated
            )
            / max(
                bands.defect_min_deteriorated_percent,
                1.0,
            )
        )

        class_b_margin = max(
            _clip(
                (
                    brown - bands.class_b_max_brown_percent
                )
                / max(
                    100.0 - bands.class_b_max_brown_percent,
                    1.0,
                )
            ),
            _clip(
                (
                    ratio
                    - bands.class_b_max_yellow_deterioration_ratio
                )
                / max(
                    bands.class_b_max_yellow_deterioration_ratio,
                    1.0,
                )
            ),
        )

        support = 0.65 * defect_margin + 0.35 * class_b_margin

    return 50.0 + 40.0 * support


def analyse_hsv(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: HSVParameters | None = None,
    bands: HSVRipenessBands | None = None,
    quality_bands: HSVQualityBands | None = None,
) -> HSVAnalysisResult:
    """
    Classify banana ripeness using HSV and conditionally grade Ripe quality.

    The quality stage reuses the same HSV colour masks. The image is not
    processed by morphology for the HSV quality result.
    """

    start_time = perf_counter()

    parameters = parameters or HSVParameters()
    bands = bands or HSVRipenessBands()
    quality_bands = quality_bands or HSVQualityBands()

    _validate_bands(bands)
    _validate_quality_bands(quality_bands)

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

    quality_assessed = category == "Ripe"
    predicted_quality = None
    quality_confidence = None
    quality_rule_key = "quality_not_assessed"
    ratio = _yellow_deterioration_ratio(
        yellow=yellow,
        deteriorated=deteriorated,
    )

    if quality_assessed:
        (
            predicted_quality,
            quality_rule_key,
            quality_reason,
            ratio,
        ) = _classify_quality(
            yellow=yellow,
            brown=brown,
            dark=dark,
            bands=quality_bands,
        )

        quality_confidence = _quality_rule_confidence(
            quality=predicted_quality,
            yellow=yellow,
            brown=brown,
            dark=dark,
            ratio=ratio,
            bands=quality_bands,
        )

        surface_grade = predicted_quality

    else:
        quality_reason = (
            f"Quality assessment was skipped because the predicted ripeness "
            f"category is {category}. Only Ripe bananas are quality graded."
        )
        surface_grade = "Not assessed"

    processing_time_ms = (
        perf_counter()
        - start_time
    ) * 1000.0

    class_scores = {
        name: 0.0
        for name in CATEGORIES
    }
    class_scores[category] = confidence / 100.0

    features = {
        "Green peel (%)": round(green, 4),
        "Yellow peel (%)": round(yellow, 4),
        "Brown peel (%)": round(brown, 4),
        "Dark peel (%)": round(dark, 4),
        "Brown + dark (%)": round(deteriorated, 4),
        "Other peel (%)": round(other, 4),
        "Yellow / deterioration ratio": (
            None
            if not np.isfinite(ratio)
            else round(float(ratio), 4)
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
        # Compatibility for existing pages/components.
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
            (
                "Ripeness classification uses green, yellow, brown, dark "
                "and unclassified HSV peel percentages."
            ),
            (
                "Surface quality is assessed only when the HSV ripeness "
                "prediction is Ripe."
            ),
            (
                "HSV quality uses only yellow, brown and dark colour "
                "measurements; no morphology measurements are used."
            ),
            (
                "Only pixels inside the shared banana-segmentation mask "
                "are analysed."
            ),
            (
                "Decision thresholds are calibrated using validation data "
                "and must be frozen before final test evaluation."
            ),
            (
                "Confidence is deterministic rule support, not a "
                "statistical probability."
            ),
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
    """
    Grade a banana whose ripeness is already known to be Ripe using HSV only.

    This function is intended for dataset/quality, where the target labels are
    Class_A, Class_B and Defect. It deliberately skips the four-class ripeness
    classifier.
    """

    start_time = perf_counter()

    parameters = parameters or HSVParameters()
    quality_bands = quality_bands or HSVQualityBands()

    _validate_quality_bands(quality_bands)

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

    (
        quality,
        rule_key,
        reason,
        ratio,
    ) = _classify_quality(
        yellow=yellow,
        brown=brown,
        dark=dark,
        bands=quality_bands,
    )

    confidence = _quality_rule_confidence(
        quality=quality,
        yellow=yellow,
        brown=brown,
        dark=dark,
        ratio=ratio,
        bands=quality_bands,
    )

    processing_time_ms = (
        perf_counter()
        - start_time
    ) * 1000.0

    class_scores = {
        name: 0.0
        for name in QUALITY_CATEGORIES
    }
    class_scores[quality] = confidence / 100.0

    features = {
        "Known ripeness": "Ripe",
        "Green peel (%)": round(green, 4),
        "Yellow peel (%)": round(yellow, 4),
        "Brown peel (%)": round(brown, 4),
        "Dark peel (%)": round(dark, 4),
        "Brown + dark (%)": round(deteriorated, 4),
        "Other peel (%)": round(other, 4),
        "Yellow / deterioration ratio": (
            None
            if not np.isfinite(ratio)
            else round(float(ratio), 4)
        ),
        "Quality class": quality,
        "Quality decision rule": rule_key,
        "Quality decision reason": reason,
    }

    method_result = MethodResult(
        method_key="hsv",
        method_name="HSV - Ripe Banana Quality Analysis",
        predicted_category=quality,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(processing_time_ms, 2),
        class_scores=class_scores,
        features=features,
        notes=[
            (
                "The input is treated as Ripe because the quality dataset "
                "contains known-ripe bananas."
            ),
            (
                "No four-class ripeness prediction is performed in this "
                "quality-only analysis path."
            ),
            (
                "Quality is determined only from HSV-derived yellow, brown "
                "and dark peel measurements."
            ),
            (
                "No morphology, GLCM or K-means measurements are used."
            ),
            (
                "Confidence is deterministic rule support, not a "
                "statistical probability."
            ),
        ],
        is_placeholder=False,
    )

    return HSVQualityResult(
        method_result=method_result,
        masks=masks,
        predicted_quality=quality,
        confidence_percent=confidence,
        processing_time_ms=processing_time_ms,
        green_percentage=green,
        yellow_percentage=yellow,
        brown_percentage=brown,
        dark_percentage=dark,
        deteriorated_percentage=deteriorated,
        other_percentage=other,
        yellow_deterioration_ratio=ratio,
        quality_reason=reason,
        features=features,
    )
