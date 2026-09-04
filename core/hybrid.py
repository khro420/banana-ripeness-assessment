from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from core.result_schema import (
    MethodResult,
    QUALITY_CATEGORIES,
    RIPENESS_CATEGORIES,
)


CATEGORIES = RIPENESS_CATEGORIES
QUALITY_HYBRID_CATEGORIES = QUALITY_CATEGORIES
METHOD_NAMES = {"morphology": "Morphology", "hsv": "HSV", "kmeans": "K-means", "glcm": "GLCM Texture"}
METHOD_ORDER = tuple(METHOD_NAMES)
TIE_BREAK_METHOD_ORDER = ("hsv", "morphology", "glcm", "kmeans")

# Each row is calibrated per ripeness class, then re-normalised if a method fails.
CLASS_RELIABILITY_WEIGHTS = {
    "Unripe": {"morphology": 0.0135, "hsv": 0.4360, "kmeans": 0.4264, "glcm": 0.1241},
    "Ripe": {"morphology": 0.1123, "hsv": 0.4491, "kmeans": 0.2814, "glcm": 0.1572},
    "Overripe": {"morphology": 0.5702, "hsv": 0.3468, "kmeans": 0.0153, "glcm": 0.0677},
    "Rotten": {"morphology": 0.1869, "hsv": 0.6080, "kmeans": 0.0066, "glcm": 0.1984},
}

# Quality fusion is intentionally separate from ripeness fusion: its inputs are
# the four rules calibrated for bananas already known to be ripe.
# These conservative class-specific priors keep the stronger dark-region
# evidence prominent while still allowing colour and texture to change a vote.
QUALITY_CLASS_RELIABILITY_WEIGHTS = {
    "Class_A": {"morphology": 0.35, "hsv": 0.25, "kmeans": 0.25, "glcm": 0.15},
    "Class_B": {"morphology": 0.35, "hsv": 0.30, "kmeans": 0.20, "glcm": 0.15},
    "Defect": {"morphology": 0.35, "hsv": 0.25, "kmeans": 0.20, "glcm": 0.20},
}


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
    categories: tuple[str, ...]


def _validate_weights(
    weights: Mapping[str, Mapping[str, float]],
    categories: tuple[str, ...],
) -> None:
    for category in categories:
        if category not in weights:
            raise ValueError(f"Hybrid weights are missing category: {category}.")
        values = [weights[category].get(method) for method in METHOD_ORDER]
        if any(value is None for value in values):
            raise ValueError(f"Hybrid weights are missing a method for {category}.")
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("Hybrid reliability weights must be finite and non-negative.")
        if sum(values) <= 0:
            raise ValueError(f"Hybrid weights for {category} must contain a positive value.")


def _usable_results(
    method_results: Sequence[MethodResult],
    categories: tuple[str, ...],
) -> dict[str, MethodResult]:
    usable = {}
    for result in method_results:
        if not isinstance(result, MethodResult):
            raise TypeError("Every hybrid input must be a MethodResult.")
        if result.is_placeholder:
            continue
        if result.method_key not in METHOD_ORDER:
            raise ValueError(f"Unsupported hybrid method: {result.method_key}.")
        if result.method_key in usable:
            raise ValueError(f"Duplicate hybrid method: {result.method_key}.")
        if result.predicted_category in categories:
            usable[result.method_key] = result
    if len(usable) < 2:
        raise ValueError("Hybrid analysis requires at least two successful approaches.")
    return usable


def _support(result: MethodResult, category: str) -> float:
    score = result.class_scores.get(category)
    if score is not None and np.isfinite(score) and score > 0:
        return float(np.clip(score, 0.0, 1.0))
    if result.predicted_category != category:
        return 0.0
    confidence = result.confidence_percent
    return 1.0 if confidence is None or not np.isfinite(confidence) else float(np.clip(confidence / 100.0, 0.0, 1.0))


def _resolve_tie(
    categories: list[str],
    results: Mapping[str, MethodResult],
    category_order: tuple[str, ...],
) -> str:
    for method in TIE_BREAK_METHOD_ORDER:
        if method in results and results[method].predicted_category in categories:
            return results[method].predicted_category
    return next(category for category in category_order if category in categories)


def _surface_features(results: Mapping[str, MethodResult]) -> dict[str, object]:
    morphology = results.get("morphology")
    if morphology is None:
        return {}
    names = ("Total dark percentage (%)", "Dark-region spread (%)", "Surface grade")
    return {name: morphology.features[name] for name in names if name in morphology.features}


def combine_method_results(
    method_results: Sequence[MethodResult],
    class_weights: Mapping[str, Mapping[str, float]] | None = None,
    categories: tuple[str, ...] | None = None,
) -> HybridAnalysisResult:
    """Fuse valid branch predictions with class-specific weighted support."""
    start = perf_counter()
    if categories is None:
        categories = (
            QUALITY_HYBRID_CATEGORIES
            if class_weights is QUALITY_CLASS_RELIABILITY_WEIGHTS
            else CATEGORIES
        )
    categories = tuple(categories)
    if categories not in {CATEGORIES, QUALITY_HYBRID_CATEGORIES}:
        raise ValueError("Hybrid categories must be ripeness or quality categories.")
    weights = class_weights or (
        QUALITY_CLASS_RELIABILITY_WEIGHTS
        if categories == QUALITY_HYBRID_CATEGORIES
        else CLASS_RELIABILITY_WEIGHTS
    )
    _validate_weights(weights, categories)
    results = _usable_results(method_results, categories)
    raw_scores = dict.fromkeys(categories, 0.0)
    effective_weights = {category: dict.fromkeys(results, 0.0) for category in categories}
    contributions = {method: dict.fromkeys(categories, 0.0) for method in results}

    for category in categories:
        available = sum(weights[category][method] for method in results)
        if available <= 0:
            continue
        for method, result in results.items():
            effective = weights[category][method] / available
            contribution = effective * _support(result, category)
            effective_weights[category][method] = effective
            contributions[method][category] = contribution
            raw_scores[category] += contribution

    total = sum(raw_scores.values())
    if total <= 0:
        raise ValueError("The approaches did not provide usable class-support scores.")
    highest = max(raw_scores.values())
    tied = [category for category, score in raw_scores.items() if np.isclose(score, highest, rtol=0.0, atol=1e-12)]
    predicted = _resolve_tie(tied, results, categories)
    class_scores = {category: raw_scores[category] / total for category in categories}
    ranked = sorted(class_scores.values(), reverse=True)
    winning_score, margin = class_scores[predicted], max(0.0, class_scores[predicted] - ranked[1])
    agreement = sum(result.predicted_category == predicted for result in results.values())
    agreement_ratio = agreement / len(results)
    confidence = 50.0 + 45.0 * float(np.clip(
        0.50 * winning_score + 0.30 * agreement_ratio + 0.20 * min(1.0, margin * 2.0), 0.0, 1.0
    ))
    processing_time = sum(float(result.processing_time_ms or 0.0) for result in results.values()) + (perf_counter() - start) * 1000.0
    supporters = ", ".join(METHOD_NAMES[method] for method, result in results.items() if result.predicted_category == predicted) or "weighted evidence"
    reason = f"{predicted} obtained the highest normalised hybrid score of {winning_score * 100.0:.2f}%. {agreement} of {len(results)} approaches agreed with the final result: {supporters}."
    features = {
        "Methods used": len(results),
        "Method agreement (%)": round(agreement_ratio * 100.0, 2),
        "Winning class score (%)": round(winning_score * 100.0, 2),
        "Winning margin (%)": round(margin * 100.0, 2),
        "Decision reason": reason,
        **_surface_features(results),
    }
    method_result = MethodResult(
        method_key="hybrid",
        method_name="Hybrid - Class-Specific Weighted Fusion",
        predicted_category=predicted,
        confidence_percent=round(confidence, 2),
        processing_time_ms=round(processing_time, 2),
        class_scores=class_scores,
        features=features,
        notes=[
            "Each approach is weighted differently for each output category.",
            "Weights are re-normalised when an approach fails.",
            "Weights must be selected using validation data and frozen before final testing.",
            "Hybrid confidence is a rule-support score, not a probability.",
        ],
        is_placeholder=False,
    )
    return HybridAnalysisResult(
        method_result, predicted, confidence, processing_time, dict(results), class_scores, raw_scores,
        effective_weights, contributions, agreement, len(results), agreement_ratio * 100.0, margin * 100.0, reason,
        categories,
    )


def combine_quality_method_results(
    method_results: Sequence[MethodResult],
    class_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> HybridAnalysisResult:
    """Fuse the four known-ripe quality assessments."""
    return combine_method_results(
        method_results,
        class_weights=class_weights or QUALITY_CLASS_RELIABILITY_WEIGHTS,
        categories=QUALITY_HYBRID_CATEGORIES,
    )


def analyse_hybrid(
    method_results: Sequence[MethodResult],
    class_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> HybridAnalysisResult:
    return combine_method_results(method_results, class_weights)


def hybrid_status() -> str:
    return "Hybrid class-specific weighted fusion is connected."
