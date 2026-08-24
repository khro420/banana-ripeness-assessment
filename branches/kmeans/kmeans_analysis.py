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


# =========================================================
# Ripeness categories
# =========================================================

CATEGORIES = (
    "Unripe",
    "Ripe",
    "Overripe",
    "Rotten",
)


# =========================================================
# Decision thresholds
# =========================================================

@dataclass(frozen=True)
class KMeansRipenessBands:
    """
    Thresholds used to interpret the colour groups produced
    by K-means.

    K-means performs the clustering first. These thresholds
    are then used to classify the resulting colour
    distribution into a banana ripeness category.
    """

    unripe_green_score_min: float = 0.55

    ripe_yellow_score_min: float = 0.55

    overripe_brown_score_min: float = 0.18

    rotten_dark_score_min: float = 0.18


@dataclass(frozen=True)
class KMeansQualityBands:
    """Simple K-means quality thresholds for Ripe bananas only."""

    class_a_max_damage_percent: float = 10.0
    defect_min_damage_percent: float = 25.0


# =========================================================
# Analysis result
# =========================================================

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

    # Quality is checked only when ripeness is Ripe.
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


# =========================================================
# Convert K-means centroid from RGB to HSV
# =========================================================

def _centroid_to_hsv(
    rgb_centroid: np.ndarray,
) -> tuple[int, int, int]:
    """
    Convert a K-means RGB cluster centroid into HSV.

    HSV is NOT used to perform the clustering.

    K-means already produced the centroid. HSV is used only
    here to give that centroid a meaningful colour label.
    """

    rgb_pixel = np.uint8(
        [[rgb_centroid]]
    )

    hsv_pixel = cv2.cvtColor(
        rgb_pixel,
        cv2.COLOR_RGB2HSV,
    )[0][0]

    return (
        int(hsv_pixel[0]),
        int(hsv_pixel[1]),
        int(hsv_pixel[2]),
    )


# =========================================================
# Interpret each K-means cluster
# =========================================================

def _identify_cluster_colour(
    rgb_centroid: np.ndarray,
) -> str:
    """
    Interpret a K-means cluster centroid as:

    Green
    Yellow
    Brown
    Dark
    Other

    K-means itself discovers the clusters automatically.
    This function only interprets the discovered centroid.
    """

    h, s, v = _centroid_to_hsv(
        rgb_centroid
    )

    # -----------------------------------------------------
    # Dark / black deteriorated peel
    # -----------------------------------------------------

    if v <= 65:
        return "Dark"

    # -----------------------------------------------------
    # Brown peel
    # -----------------------------------------------------

    if (
        5 <= h <= 17
        and s >= 35
        and v <= 190
    ):
        return "Brown"

    # -----------------------------------------------------
    # Yellow peel
    # -----------------------------------------------------

    if (
        18 <= h <= 34
        and s >= 35
        and v >= 70
    ):
        return "Yellow"

    # -----------------------------------------------------
    # Green peel
    # -----------------------------------------------------

    if (
        35 <= h <= 85
        and s >= 35
        and v >= 35
    ):
        return "Green"

    return "Other"


# =========================================================
# Calculate proportion of each interpreted colour
# =========================================================

def _calculate_colour_scores(
    segmentation: KMeansSegmentationResult,
) -> dict[str, float]:
    """
    Calculate the total banana proportion represented by
    green, yellow, brown, dark and other K-means clusters.

    Example:

        Green  = 0.10
        Yellow = 0.65
        Brown  = 0.18
        Dark   = 0.04
        Other  = 0.03

    These values are based on K-means cluster percentages.
    """

    scores = {
        "Green": 0.0,
        "Yellow": 0.0,
        "Brown": 0.0,
        "Dark": 0.0,
        "Other": 0.0,
    }

    for cluster_id, centre in enumerate(
        segmentation.centres_rgb
    ):

        colour_name = _identify_cluster_colour(
            centre
        )

        percentage = float(
            segmentation.cluster_percentages[
                cluster_id
            ]
        )

        # Convert percentage from 0-100 into 0-1.
        scores[colour_name] += (
            percentage / 100.0
        )

    return scores


# =========================================================
# Ripeness classification
# =========================================================

def _classify(
    scores: dict[str, float],
    bands: KMeansRipenessBands,
) -> tuple[str, str]:
    """
    Classify banana ripeness from the colour distribution
    discovered by K-means.

    Decision order:

        Rotten
        -> Overripe
        -> Unripe
        -> Ripe
        -> fallback rules
    """

    green = scores["Green"]
    yellow = scores["Yellow"]
    brown = scores["Brown"]
    dark = scores["Dark"]

    # Combined evidence of deterioration.
    deteriorated = brown + dark

    # =====================================================
    # 1. ROTTEN
    # =====================================================

    # Rotten bananas should contain a meaningful amount
    # of dark peel together with substantial deterioration.

    if (
        dark >= bands.rotten_dark_score_min
        and deteriorated >= 0.30
    ):
        return (
            "Rotten",
            (
                "K-means detected a substantial proportion "
                "of dark and deteriorated peel regions."
            ),
        )

    # =====================================================
    # 2. OVERRIPE
    # =====================================================

    # Brown regions are the main evidence for overripe
    # bananas. Combined brown + dark regions can also
    # indicate deterioration.

    if (
        brown >= bands.overripe_brown_score_min
        or deteriorated >= 0.20
    ):
        return (
            "Overripe",
            (
                "K-means detected noticeable brown or "
                "deteriorated peel regions."
            ),
        )

    # =====================================================
    # 3. UNRIPE
    # =====================================================

    if (
        green >= bands.unripe_green_score_min
        and green > yellow
    ):
        return (
            "Unripe",
            (
                "K-means detected green as the dominant "
                "banana-peel colour."
            ),
        )

    # =====================================================
    # 4. RIPE
    # =====================================================

    if (
        yellow >= bands.ripe_yellow_score_min
        and deteriorated < 0.20
    ):
        return (
            "Ripe",
            (
                "K-means detected predominantly yellow peel "
                "with little brown or dark deterioration."
            ),
        )

    # =====================================================
    # FALLBACK RULES
    # =====================================================

    # Strong dark evidence that narrowly missed the main
    # rotten condition.

    if (
        dark >= 0.15
        and deteriorated >= 0.25
    ):
        return (
            "Rotten",
            (
                "Dark peel regions together with substantial "
                "deterioration indicate an advanced stage."
            ),
        )

    # Moderate deterioration.
    if deteriorated >= 0.15:
        return (
            "Overripe",
            (
                "Brown and dark peel regions indicate "
                "increasing deterioration."
            ),
        )

    # Remaining green-dominant cases.
    if green > yellow:
        return (
            "Unripe",
            (
                "Green was the strongest remaining "
                "K-means peel-colour group."
            ),
        )

    # Remaining cases are treated as ripe.
    return (
        "Ripe",
        (
            "Yellow was the strongest remaining "
            "K-means peel-colour group."
        ),
    )


# =========================================================
# Confidence calculation
# =========================================================

def _confidence(
    category: str,
    scores: dict[str, float],
) -> float:
    """
    Calculate a simple rule-support confidence.

    IMPORTANT:
    This is not a machine-learning probability.
    """

    green = scores["Green"]
    yellow = scores["Yellow"]
    brown = scores["Brown"]
    dark = scores["Dark"]

    if category == "Unripe":

        support = green

    elif category == "Ripe":

        support = yellow

    elif category == "Overripe":

        # Overripe is supported mainly by brown,
        # with some contribution from dark peel.
        support = (
            brown
            + (0.30 * dark)
        )

    else:

        # Rotten is supported mainly by dark peel,
        # with brown contributing additional evidence.
        support = (
            dark
            + (0.30 * brown)
        )

    support = float(
        np.clip(
            support,
            0.0,
            1.0,
        )
    )

    return (
        50.0
        + 45.0 * support
    )



# =========================================================
# Ripe-only quality checking
# =========================================================

def _quality_damage_percent(
    scores: dict[str, float],
) -> float:
    """
    Estimate visible quality damage from the SAME K-means clusters.

    Dark clusters count fully as damage.
    Brown clusters count half because some brown colour can occur
    during normal ripening.
    """

    damage = (
        scores["Dark"]
        + 0.50 * scores["Brown"]
    ) * 100.0

    return float(np.clip(damage, 0.0, 100.0))


def _classify_quality(
    damage_percent: float,
    bands: KMeansQualityBands,
) -> tuple[str, str]:
    """Classify a Ripe banana as Class_A, Class_B or Defect."""

    if damage_percent <= bands.class_a_max_damage_percent:
        return (
            "Class_A",
            f"K-means found low visible damage ({damage_percent:.2f}%).",
        )

    if damage_percent >= bands.defect_min_damage_percent:
        return (
            "Defect",
            f"K-means found high visible damage ({damage_percent:.2f}%).",
        )

    return (
        "Class_B",
        f"K-means found moderate visible damage ({damage_percent:.2f}%).",
    )


def _quality_confidence(
    quality: str,
    damage_percent: float,
    bands: KMeansQualityBands,
) -> float:
    """Simple rule-support confidence for the quality result."""

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
            bands.class_a_max_damage_percent
            + bands.defect_min_damage_percent
        ) / 2.0
        half_width = max(
            (
                bands.defect_min_damage_percent
                - bands.class_a_max_damage_percent
            ) / 2.0,
            1.0,
        )
        support = 1.0 - abs(damage_percent - midpoint) / half_width

    support = float(np.clip(support, 0.0, 1.0))
    return 55.0 + 35.0 * support


# =========================================================
# Main K-means analysis function
# =========================================================

def analyse_kmeans(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: KMeansParameters | None = None,
    bands: KMeansRipenessBands | None = None,
    quality_bands: KMeansQualityBands | None = None,
) -> KMeansAnalysisResult:
    """
    Analyse banana ripeness using K-means colour clustering.

    Processing pipeline:

        Banana image
              |
              v
        Banana mask
              |
              v
        K-means clustering
              |
              v
        Cluster centroids
              |
              v
        Interpret centroid colours
              |
              v
        Calculate colour proportions
              |
              v
        Apply ripeness rules
              |
              v
        Final prediction
    """

    start_time = perf_counter()

    # -----------------------------------------------------
    # Default parameters
    # -----------------------------------------------------

    parameters = (
        parameters
        or KMeansParameters()
    )

    bands = (
        bands
        or KMeansRipenessBands()
    )

    quality_bands = (
        quality_bands
        or KMeansQualityBands()
    )

    # -----------------------------------------------------
    # Perform K-means colour clustering
    # -----------------------------------------------------

    segmentation = segment_kmeans_colours(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
    )

    # -----------------------------------------------------
    # Interpret K-means clusters
    # -----------------------------------------------------

    scores = _calculate_colour_scores(
        segmentation
    )

    # -----------------------------------------------------
    # Determine ripeness category
    # -----------------------------------------------------

    category, reason = _classify(
        scores=scores,
        bands=bands,
    )

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = _confidence(
        category=category,
        scores=scores,
    )

    # -----------------------------------------------------
    # Ripe-only quality checking
    # -----------------------------------------------------

    quality_assessed = category == "Ripe"
    predicted_quality = None
    quality_confidence = None
    quality_damage = None

    if quality_assessed:
        quality_damage = _quality_damage_percent(scores)
        predicted_quality, quality_reason = _classify_quality(
            quality_damage,
            quality_bands,
        )
        quality_confidence = _quality_confidence(
            predicted_quality,
            quality_damage,
            quality_bands,
        )
    else:
        quality_reason = (
            f"Quality was not checked because ripeness was predicted as "
            f"{category}. Quality checking only applies to Ripe bananas."
        )

    # -----------------------------------------------------
    # Processing time
    # -----------------------------------------------------

    processing_time_ms = (
        perf_counter()
        - start_time
    ) * 1000.0

    # -----------------------------------------------------
    # Dominant interpreted colour
    # -----------------------------------------------------

    dominant_colour = max(
        scores,
        key=scores.get,
    )

    # -----------------------------------------------------
    # Features displayed / stored
    # -----------------------------------------------------

    features = {

        "Dominant cluster colour":
            dominant_colour,

        "Green proportion":
            round(
                scores["Green"] * 100.0,
                2,
            ),

        "Yellow proportion":
            round(
                scores["Yellow"] * 100.0,
                2,
            ),

        "Brown proportion":
            round(
                scores["Brown"] * 100.0,
                2,
            ),

        "Dark proportion":
            round(
                scores["Dark"] * 100.0,
                2,
            ),

        "Other proportion":
            round(
                scores["Other"] * 100.0,
                2,
            ),

        "Quality assessed":
            quality_assessed,

        "Quality class":
            predicted_quality or "Not assessed",

        "Quality damage (%)":
            (
                None
                if quality_damage is None
                else round(quality_damage, 2)
            ),

        "Quality confidence (%)":
            (
                None
                if quality_confidence is None
                else round(quality_confidence, 2)
            ),

        "Quality reason":
            quality_reason,

        "Number of clusters":
            parameters.k,

        "K-means compactness":
            round(
                segmentation.compactness,
                4,
            ),

        "Decision reason":
            reason,
    }

    # -----------------------------------------------------
    # Class scores
    # -----------------------------------------------------

    class_scores = {
        name: 0.0
        for name in CATEGORIES
    }

    class_scores[category] = (
        confidence / 100.0
    )

    # -----------------------------------------------------
    # Standard MethodResult
    # -----------------------------------------------------

    method_result = MethodResult(

        method_key="kmeans",

        method_name=(
            "K-means - Colour Clustering"
        ),

        predicted_category=category,

        confidence_percent=round(
            confidence,
            2,
        ),

        processing_time_ms=round(
            processing_time_ms,
            2,
        ),

        class_scores=class_scores,

        features=features,

        notes=[
            (
                "K-means clustering is applied only "
                "to pixels inside the shared banana mask."
            ),
            (
                "The algorithm automatically groups "
                "similar banana-peel colours."
            ),
            (
                "The resulting K-means cluster centroids "
                "are interpreted as green, yellow, brown, "
                "dark or other."
            ),
            (
                "The proportions of the interpreted "
                "clusters are used to determine the "
                "ripeness category."
            ),
            (
                "HSV is used only to interpret the "
                "K-means cluster centroids and is not "
                "used to perform the clustering."
            ),
            (
                "The final category is Unripe, Ripe, "
                "Overripe or Rotten."
            ),
            (
                "Quality is checked only when the K-means ripeness "
                "prediction is Ripe."
            ),
            (
                "Quality reuses the same Brown and Dark K-means clusters; "
                "no morphology blemish mask is used."
            ),
            (
                "Confidence represents rule support "
                "rather than a statistical probability."
            ),
        ],

        is_placeholder=False,
    )

    # -----------------------------------------------------
    # Return complete analysis
    # -----------------------------------------------------

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


# =========================================================
# Quality-dataset analysis
# =========================================================

def analyse_kmeans_quality(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: KMeansParameters | None = None,
    quality_bands: KMeansQualityBands | None = None,
) -> KMeansQualityResult:
    """
    Grade a banana whose ripeness is already known to be Ripe.

    This is used by dataset/Quality/test, where only
    Class_A, Class_B and Defect are evaluated.
    """

    start_time = perf_counter()

    parameters = parameters or KMeansParameters(k=4)
    quality_bands = quality_bands or KMeansQualityBands()

    segmentation = segment_kmeans_colours(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
    )

    scores = _calculate_colour_scores(segmentation)
    damage_percent = _quality_damage_percent(scores)

    quality, reason = _classify_quality(
        damage_percent,
        quality_bands,
    )

    confidence = _quality_confidence(
        quality,
        damage_percent,
        quality_bands,
    )

    processing_time_ms = (
        perf_counter()
        - start_time
    ) * 1000.0

    features = {
        "Known ripeness": "Ripe",
        "Brown proportion (%)": round(scores["Brown"] * 100.0, 2),
        "Dark proportion (%)": round(scores["Dark"] * 100.0, 2),
        "Quality damage (%)": round(damage_percent, 2),
        "Quality class": quality,
        "Quality reason": reason,
        "Number of clusters": parameters.k,
    }

    class_scores = {
        name: 0.0
        for name in QUALITY_CATEGORIES
    }
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
            "Dark clusters count fully and Brown clusters count half "
            "toward the quality-damage score.",
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
        damage_percent=damage_percent,
        quality_reason=reason,
        features=features,
    )
