from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from core.result_schema import MethodResult


CATEGORIES = (
    "Unripe",
    "Ripe",
    "Overripe",
    "Rotten",
)

METHOD_NAMES = {
    "morphology": "Morphology",
    "hsv": "HSV",
    "kmeans": "K-means",
    "glcm": "GLCM Texture",
}

METHOD_ORDER = (
    "morphology",
    "hsv",
    "kmeans",
    "glcm",
)


# Starting reliability weights.
#
# These reflect the expected strengths of each technique:
# - HSV and K-means: Unripe and Ripe colour information.
# - Morphology: widespread dark regions for Overripe.
# - HSV, GLCM and Morphology: Rotten deterioration evidence.
#
# Calibrate these weights using validation data only.
# Do not calculate them from the final test results.
CLASS_RELIABILITY_WEIGHTS = {
    "Unripe": {
        "morphology": 0.05,
        "hsv": 0.45,
        "kmeans": 0.40,
        "glcm": 0.10,
    },
    "Ripe": {
        "morphology": 0.05,
        "hsv": 0.50,
        "kmeans": 0.35,
        "glcm": 0.10,
    },
    "Overripe": {
        "morphology": 0.45,
        "hsv": 0.40,
        "kmeans": 0.05,
        "glcm": 0.10,
    },
    "Rotten": {
        "morphology": 0.25,
        "hsv": 0.40,
        "kmeans": 0.05,
        "glcm": 0.30,
    },
}


# Used only when categories receive exactly equal scores.
TIE_BREAK_METHOD_ORDER = (
    "hsv",
    "morphology",
    "glcm",
    "kmeans",
)


@dataclass
class HybridAnalysisResult:
    method_result: MethodResult

    predicted_category: str
    confidence_percent: float
    processing_time_ms: float

    method_results: dict[str, MethodResult]

    class_scores: dict[str, float]
    raw_class_scores: dict[str, float]

    effective_weights: dict[str, dict[str, float]]
    method_contributions: dict[str, dict[str, float]]

    agreement_count: int
    methods_used: int
    agreement_percent: float
    winning_margin_percent: float

    decision_reason: str


def _validate_weights(
    weights: Mapping[str, Mapping[str, float]],
) -> None:
    for category in CATEGORIES:
        if category not in weights:
            raise ValueError(
                f"Hybrid weights are missing category: {category}."
            )

        for method_key in METHOD_ORDER:
            value = weights[category].get(method_key)

            if value is None:
                raise ValueError(
                    f"Hybrid weights are missing "
                    f"{method_key} for {category}."
                )

            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "Hybrid reliability weights must be "
                    "finite and non-negative."
                )

        if sum(weights[category].values()) <= 0:
            raise ValueError(
                f"Hybrid weights for {category} must "
                "contain a positive value."
            )


def _collect_usable_results(
    method_results: Sequence[MethodResult],
) -> dict[str, MethodResult]:
    usable_results: dict[str, MethodResult] = {}

    for result in method_results:
        if not isinstance(result, MethodResult):
            raise TypeError(
                "Every hybrid input must be a MethodResult."
            )

        if result.is_placeholder:
            continue

        if result.method_key not in METHOD_ORDER:
            raise ValueError(
                f"Unsupported hybrid method: {result.method_key}."
            )

        if result.method_key in usable_results:
            raise ValueError(
                f"Duplicate hybrid method: {result.method_key}."
            )

        if result.predicted_category not in CATEGORIES:
            continue

        usable_results[result.method_key] = result

    if len(usable_results) < 2:
        raise ValueError(
            "Hybrid analysis requires at least two "
            "successful approaches."
        )

    return usable_results


def _get_category_support(
    result: MethodResult,
    category: str,
) -> float:
    """
    Return one approach's support for a category.

    Current approach modules store rule support inside
    MethodResult.class_scores. If unavailable, the predicted
    category and confidence are used as a fallback.
    """

    score = result.class_scores.get(category)

    if (
        score is not None
        and np.isfinite(score)
        and score > 0
    ):
        return float(
            np.clip(score, 0.0, 1.0)
        )

    if result.predicted_category != category:
        return 0.0

    confidence = result.confidence_percent

    if (
        confidence is None
        or not np.isfinite(confidence)
    ):
        return 1.0

    return float(
        np.clip(
            confidence / 100.0,
            0.0,
            1.0,
        )
    )


def _resolve_tie(
    tied_categories: list[str],
    results: Mapping[str, MethodResult],
) -> str:
    for method_key in TIE_BREAK_METHOD_ORDER:
        result = results.get(method_key)

        if (
            result is not None
            and result.predicted_category in tied_categories
        ):
            return result.predicted_category

    return next(
        category
        for category in CATEGORIES
        if category in tied_categories
    )


def _get_surface_features(
    results: Mapping[str, MethodResult],
) -> dict[str, object]:
    morphology_result = results.get("morphology")

    if morphology_result is None:
        return {}

    selected_features = {}

    for feature_name in (
        "Total dark percentage (%)",
        "Dark-region spread (%)",
        "Surface grade",
    ):
        if feature_name in morphology_result.features:
            selected_features[feature_name] = (
                morphology_result.features[feature_name]
            )

    return selected_features


def combine_method_results(
    method_results: Sequence[MethodResult],
    class_weights: (
        Mapping[str, Mapping[str, float]] | None
    ) = None,
) -> HybridAnalysisResult:
    """
    Combine approach predictions using class-specific weighted fusion.

    The function operates at decision level. Each approach supplies
    its predicted class and rule-support score. Reliability weights
    represent how suitable each approach is for each class.
    """

    fusion_start = perf_counter()

    weights = (
        class_weights
        or CLASS_RELIABILITY_WEIGHTS
    )

    _validate_weights(weights)

    results = _collect_usable_results(
        method_results
    )

    raw_scores = {
        category: 0.0
        for category in CATEGORIES
    }

    effective_weights = {
        category: {
            method_key: 0.0
            for method_key in results
        }
        for category in CATEGORIES
    }

    contributions = {
        method_key: {
            category: 0.0
            for category in CATEGORIES
        }
        for method_key in results
    }

    for category in CATEGORIES:
        available_weight = sum(
            weights[category][method_key]
            for method_key in results
        )

        if available_weight <= 0:
            continue

        for method_key, result in results.items():
            effective_weight = (
                weights[category][method_key]
                / available_weight
            )

            support = _get_category_support(
                result=result,
                category=category,
            )

            contribution = (
                effective_weight
                * support
            )

            effective_weights[
                category
            ][method_key] = effective_weight

            contributions[
                method_key
            ][category] = contribution

            raw_scores[category] += contribution

    highest_raw_score = max(
        raw_scores.values()
    )

    tied_categories = [
        category
        for category, score in raw_scores.items()
        if np.isclose(
            score,
            highest_raw_score,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    predicted_category = _resolve_tie(
        tied_categories,
        results,
    )

    total_score = sum(
        raw_scores.values()
    )

    if total_score <= 0:
        raise ValueError(
            "The approaches did not provide usable "
            "class-support scores."
        )

    class_scores = {
        category: (
            raw_scores[category]
            / total_score
        )
        for category in CATEGORIES
    }

    ranked_scores = sorted(
        class_scores.values(),
        reverse=True,
    )

    winning_score = class_scores[
        predicted_category
    ]

    runner_up_score = (
        ranked_scores[1]
        if len(ranked_scores) > 1
        else 0.0
    )

    winning_margin = max(
        0.0,
        winning_score - runner_up_score,
    )

    agreement_count = sum(
        result.predicted_category
        == predicted_category
        for result in results.values()
    )

    methods_used = len(results)

    agreement_ratio = (
        agreement_count
        / methods_used
    )

    # Hybrid confidence combines:
    # 50% winning score
    # 30% method agreement
    # 20% winning margin
    #
    # The final range is kept between 50% and 95%.
    confidence_support = (
        0.50 * winning_score
        + 0.30 * agreement_ratio
        + 0.20 * min(
            1.0,
            winning_margin * 2.0,
        )
    )

    confidence_percent = (
        50.0
        + 45.0
        * float(
            np.clip(
                confidence_support,
                0.0,
                1.0,
            )
        )
    )

    branch_processing_time = sum(
        float(
            result.processing_time_ms
            or 0.0
        )
        for result in results.values()
    )

    fusion_processing_time = (
        perf_counter()
        - fusion_start
    ) * 1000.0

    processing_time_ms = (
        branch_processing_time
        + fusion_processing_time
    )

    supporting_methods = [
        METHOD_NAMES[method_key]
        for method_key, result in results.items()
        if (
            result.predicted_category
            == predicted_category
        )
    ]

    support_text = (
        ", ".join(supporting_methods)
        or "weighted evidence"
    )

    decision_reason = (
        f"{predicted_category} obtained the highest "
        f"normalised hybrid score of "
        f"{winning_score * 100.0:.2f}%. "
        f"{agreement_count} of {methods_used} approaches "
        f"agreed with the final result: {support_text}."
    )

    features = {
        "Methods used": methods_used,
        "Method agreement (%)": round(
            agreement_ratio * 100.0,
            2,
        ),
        "Winning class score (%)": round(
            winning_score * 100.0,
            2,
        ),
        "Winning margin (%)": round(
            winning_margin * 100.0,
            2,
        ),
        "Decision reason": decision_reason,
        **_get_surface_features(results),
    }

    method_result = MethodResult(
        method_key="hybrid",
        method_name=(
            "Hybrid - Class-Specific Weighted Fusion"
        ),
        predicted_category=predicted_category,
        confidence_percent=round(
            confidence_percent,
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
                "Each approach is weighted differently "
                "for each ripeness category."
            ),
            (
                "Weights are re-normalised when an "
                "approach fails."
            ),
            (
                "Weights must be selected using validation "
                "data and frozen before final testing."
            ),
            (
                "Hybrid confidence is a rule-support score, "
                "not a probability."
            ),
        ],
        is_placeholder=False,
    )

    return HybridAnalysisResult(
        method_result=method_result,
        predicted_category=predicted_category,
        confidence_percent=confidence_percent,
        processing_time_ms=processing_time_ms,
        method_results=dict(results),
        class_scores=class_scores,
        raw_class_scores=raw_scores,
        effective_weights=effective_weights,
        method_contributions=contributions,
        agreement_count=agreement_count,
        methods_used=methods_used,
        agreement_percent=(
            agreement_ratio * 100.0
        ),
        winning_margin_percent=(
            winning_margin * 100.0
        ),
        decision_reason=decision_reason,
    )


def analyse_hybrid(
    method_results: Sequence[MethodResult],
    class_weights: (
        Mapping[str, Mapping[str, float]] | None
    ) = None,
) -> HybridAnalysisResult:
    """Alternative name for evaluation-module integration."""

    return combine_method_results(
        method_results=method_results,
        class_weights=class_weights,
    )


def hybrid_status() -> str:
    return (
        "Hybrid class-specific weighted fusion "
        "is connected."
    )