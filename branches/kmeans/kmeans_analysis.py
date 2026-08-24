from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from branches.kmeans.kmeans_segmentation import (
    KMeansParameters,
    KMeansSegmentationResult,
    segment_kmeans_colours,
)
from core.result_schema import MethodResult, QUALITY_CATEGORIES


CATEGORIES = ("Unripe", "Ripe", "Overripe", "Rotten")
COLOURS = ("Green", "Yellow", "Brown", "Dark", "Other")


@dataclass(frozen=True)
class KMeansRipenessBands:
    unripe_green_score_min: float = 0.55
    ripe_yellow_score_min: float = 0.55
    overripe_brown_score_min: float = 0.18
    rotten_dark_score_min: float = 0.18


@dataclass(frozen=True)
class KMeansQualityBands:
    class_a_max_damage_percent: float = 10.0
    defect_min_damage_percent: float = 25.0


@dataclass
class KMeansAnalysisResult:
    method_result: MethodResult
    segmentation: KMeansSegmentationResult
    predicted_category: str
    confidence_percent: float
    processing_time_ms: float
    dominant_colour: str
    decision_reason: str
    features: dict[str, Any]
    quality_assessed: bool
    predicted_quality: str | None
    quality_confidence_percent: float | None
    quality_reason: str
    quality_damage_percent: float | None


@dataclass
class KMeansQualityResult:
    method_result: MethodResult
    segmentation: KMeansSegmentationResult
    predicted_quality: str
    confidence_percent: float
    processing_time_ms: float
    damage_percent: float
    quality_reason: str
    features: dict[str, Any]


def _identify_cluster_colour(rgb_centroid: np.ndarray) -> str:
    """Give one K-means RGB centroid a simple colour label."""
    pixel = np.uint8([[rgb_centroid]])
    h, s, v = map(int, cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0])

    if v <= 65:
        return "Dark"
    if 5 <= h <= 17 and s >= 35 and v <= 190:
        return "Brown"
    if 18 <= h <= 34 and s >= 35 and v >= 70:
        return "Yellow"
    if 35 <= h <= 85 and s >= 35 and v >= 35:
        return "Green"
    return "Other"


def _calculate_colour_scores(segmentation: KMeansSegmentationResult) -> dict[str, float]:
    """Combine cluster percentages by interpreted colour."""
    scores = dict.fromkeys(COLOURS, 0.0)
    for centre, percentage in zip(
        segmentation.centres_rgb,
        segmentation.cluster_percentages,
    ):
        scores[_identify_cluster_colour(centre)] += float(percentage) / 100.0
    return scores


def _classify(
    scores: dict[str, float],
    bands: KMeansRipenessBands,
) -> tuple[str, str]:
    """Apply the original K=4 ripeness rules without changing their order."""
    green = scores["Green"]
    yellow = scores["Yellow"]
    brown = scores["Brown"]
    dark = scores["Dark"]
    deteriorated = brown + dark

    if dark >= bands.rotten_dark_score_min and deteriorated >= 0.30:
        return (
            "Rotten",
            "K-means detected a substantial proportion of dark and deteriorated peel regions.",
        )

    if brown >= bands.overripe_brown_score_min or deteriorated >= 0.20:
        return (
            "Overripe",
            "K-means detected noticeable brown or deteriorated peel regions.",
        )

    if green >= bands.unripe_green_score_min and green > yellow:
        return (
            "Unripe",
            "K-means detected green as the dominant banana-peel colour.",
        )

    if yellow >= bands.ripe_yellow_score_min and deteriorated < 0.20:
        return (
            "Ripe",
            "K-means detected predominantly yellow peel with little brown or dark deterioration.",
        )

    if dark >= 0.15 and deteriorated >= 0.25:
        return (
            "Rotten",
            "Dark peel regions together with substantial deterioration indicate an advanced stage.",
        )

    if deteriorated >= 0.15:
        return (
            "Overripe",
            "Brown and dark peel regions indicate increasing deterioration.",
        )

    if green > yellow:
        return (
            "Unripe",
            "Green was the strongest remaining K-means peel-colour group.",
        )

    return (
        "Ripe",
        "Yellow was the strongest remaining K-means peel-colour group.",
    )


def _confidence(category: str, scores: dict[str, float]) -> float:
    """Return the same 50-95 rule-support score as the original code."""
    if category == "Unripe":
        support = scores["Green"]
    elif category == "Ripe":
        support = scores["Yellow"]
    elif category == "Overripe":
        support = scores["Brown"] + 0.30 * scores["Dark"]
    else:
        support = scores["Dark"] + 0.30 * scores["Brown"]
    return 50.0 + 45.0 * float(np.clip(support, 0.0, 1.0))


def _quality_damage_percent(scores: dict[str, float]) -> float:
    damage = (scores["Dark"] + 0.50 * scores["Brown"]) * 100.0
    return float(np.clip(damage, 0.0, 100.0))


def _classify_quality(
    damage_percent: float,
    bands: KMeansQualityBands,
) -> tuple[str, str]:
    if damage_percent <= bands.class_a_max_damage_percent:
        return "Class_A", f"K-means found low visible damage ({damage_percent:.2f}%)."
    if damage_percent >= bands.defect_min_damage_percent:
        return "Defect", f"K-means found high visible damage ({damage_percent:.2f}%)."
    return "Class_B", f"K-means found moderate visible damage ({damage_percent:.2f}%)."


def _quality_confidence(
    quality: str,
    damage_percent: float,
    bands: KMeansQualityBands,
) -> float:
    if quality == "Class_A":
        support = (
            bands.class_a_max_damage_percent - damage_percent
        ) / max(bands.class_a_max_damage_percent, 1.0)
    elif quality == "Defect":
        support = (
            damage_percent - bands.defect_min_damage_percent
        ) / max(100.0 - bands.defect_min_damage_percent, 1.0)
    else:
        midpoint = (
            bands.class_a_max_damage_percent + bands.defect_min_damage_percent
        ) / 2.0
        half_width = max(
            (bands.defect_min_damage_percent - bands.class_a_max_damage_percent) / 2.0,
            1.0,
        )
        support = 1.0 - abs(damage_percent - midpoint) / half_width

    return 55.0 + 35.0 * float(np.clip(support, 0.0, 1.0))


def _quality_values(
    scores: dict[str, float],
    bands: KMeansQualityBands,
) -> tuple[str, float, float, str]:
    damage = _quality_damage_percent(scores)
    quality, reason = _classify_quality(damage, bands)
    confidence = _quality_confidence(quality, damage, bands)
    return quality, damage, confidence, reason


def analyse_kmeans(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: KMeansParameters | None = None,
    bands: KMeansRipenessBands | None = None,
    quality_bands: KMeansQualityBands | None = None,
) -> KMeansAnalysisResult:
    start = perf_counter()
    parameters = parameters or KMeansParameters()
    bands = bands or KMeansRipenessBands()
    quality_bands = quality_bands or KMeansQualityBands()

    segmentation = segment_kmeans_colours(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
    )
    scores = _calculate_colour_scores(segmentation)
    category, reason = _classify(scores, bands)
    confidence = _confidence(category, scores)

    quality_assessed = category == "Ripe"
    predicted_quality = None
    quality_confidence = None
    quality_damage = None

    if quality_assessed:
        predicted_quality, quality_damage, quality_confidence, quality_reason = (
            _quality_values(scores, quality_bands)
        )
    else:
        quality_reason = (
            f"Quality was not checked because ripeness was predicted as {category}. "
            "Quality checking only applies to Ripe bananas."
        )

    processing_time_ms = (perf_counter() - start) * 1000.0
    dominant_colour = max(scores, key=scores.get)

    features = {
        "Dominant cluster colour": dominant_colour,
        "Green proportion": round(scores["Green"] * 100.0, 2),
        "Yellow proportion": round(scores["Yellow"] * 100.0, 2),
        "Brown proportion": round(scores["Brown"] * 100.0, 2),
        "Dark proportion": round(scores["Dark"] * 100.0, 2),
        "Other proportion": round(scores["Other"] * 100.0, 2),
        "Quality assessed": quality_assessed,
        "Quality class": predicted_quality or "Not assessed",
        "Quality damage (%)": None if quality_damage is None else round(quality_damage, 2),
        "Quality confidence (%)": (
            None if quality_confidence is None else round(quality_confidence, 2)
        ),
        "Quality reason": quality_reason,
        "Number of clusters": parameters.k,
        "K-means compactness": round(segmentation.compactness, 4),
        "Decision reason": reason,
    }

    class_scores = {name: 0.0 for name in CATEGORIES}
    class_scores[category] = confidence / 100.0

    method_result = MethodResult(
        method_key="kmeans",
        method_name="K-means - Colour Clustering",
        predicted_category=category,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(processing_time_ms, 2),
        class_scores=class_scores,
        features=features,
        notes=[
            "K-means clustering is applied only to pixels inside the shared banana mask.",
            "The algorithm automatically groups similar banana-peel colours.",
            "The resulting K-means cluster centroids are interpreted as green, yellow, brown, dark or other.",
            "The proportions of the interpreted clusters are used to determine the ripeness category.",
            "HSV is used only to interpret the K-means cluster centroids and is not used to perform the clustering.",
            "The final category is Unripe, Ripe, Overripe or Rotten.",
            "Quality is checked only when the K-means ripeness prediction is Ripe.",
            "Quality reuses the same Brown and Dark K-means clusters; no morphology blemish mask is used.",
            "Confidence represents rule support rather than a statistical probability.",
        ],
        is_placeholder=False,
    )

    return KMeansAnalysisResult(
        method_result=method_result,
        segmentation=segmentation,
        predicted_category=category,
        confidence_percent=confidence,
        processing_time_ms=processing_time_ms,
        dominant_colour=dominant_colour,
        decision_reason=reason,
        features=features,
        quality_assessed=quality_assessed,
        predicted_quality=predicted_quality,
        quality_confidence_percent=quality_confidence,
        quality_reason=quality_reason,
        quality_damage_percent=quality_damage,
    )


def analyse_kmeans_quality(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: KMeansParameters | None = None,
    quality_bands: KMeansQualityBands | None = None,
) -> KMeansQualityResult:
    start = perf_counter()
    parameters = parameters or KMeansParameters(k=4)
    quality_bands = quality_bands or KMeansQualityBands()

    segmentation = segment_kmeans_colours(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
    )
    scores = _calculate_colour_scores(segmentation)
    quality, damage, confidence, reason = _quality_values(scores, quality_bands)
    processing_time_ms = (perf_counter() - start) * 1000.0

    features = {
        "Known ripeness": "Ripe",
        "Brown proportion (%)": round(scores["Brown"] * 100.0, 2),
        "Dark proportion (%)": round(scores["Dark"] * 100.0, 2),
        "Quality damage (%)": round(damage, 2),
        "Quality class": quality,
        "Quality reason": reason,
        "Number of clusters": parameters.k,
    }

    class_scores = {name: 0.0 for name in QUALITY_CATEGORIES}
    class_scores[quality] = confidence / 100.0

    method_result = MethodResult(
        method_key="kmeans",
        method_name="K-means - Ripe Banana Quality Analysis",
        predicted_category=quality,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(processing_time_ms, 2),
        class_scores=class_scores,
        features=features,
        notes=[
            "The quality dataset is treated as already Ripe.",
            "Quality is derived from K-means Brown and Dark clusters.",
            "Dark clusters count fully and Brown clusters count half toward the quality-damage score.",
            "No morphology blemish mask is used.",
        ],
        is_placeholder=False,
    )

    return KMeansQualityResult(
        method_result=method_result,
        segmentation=segmentation,
        predicted_quality=quality,
        confidence_percent=confidence,
        processing_time_ms=processing_time_ms,
        damage_percent=damage,
        quality_reason=reason,
        features=features,
    )
