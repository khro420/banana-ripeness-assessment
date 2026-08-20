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
from core.result_schema import MethodResult


CATEGORIES = ("Unripe", "Ripe", "Overripe", "Rotten")


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

    @property
    def concentration_ratio_percentage(self) -> float:
        return self.concentration_ratio


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


def _surface_grade(total_dark: float) -> str:
    if total_dark <= 5.0:
        return "Grade A - Very low visible dark area"
    if total_dark <= 15.0:
        return "Grade B - Low visible dark area"
    if total_dark <= 30.0:
        return "Grade C - High visible dark area"
    return "Reject - Severe visible dark area"


def analyse_morphology(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters | None = None,
    bands: RipenessBands | None = None,
) -> MorphologyAnalysisResult:
    """Classify ripeness using validation-calibrated morphology rules."""

    start_time = perf_counter()
    parameters = parameters or MorphologyParameters()
    bands = bands or RipenessBands()
    _validate_bands(bands)

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
    surface_grade = _surface_grade(float(values["total_dark_percentage"]))
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
    )