from dataclasses import dataclass, field
from typing import Any


RIPENESS_CATEGORIES = ("Unripe", "Ripe", "Overripe", "Rotten")
QUALITY_CATEGORIES = ("Class_A", "Class_B", "Defect")

# Backward-compatible name used by the existing approach modules.
CATEGORIES = RIPENESS_CATEGORIES

METHOD_NAMES = {
    "morphology": "Morphology",
    "hsv": "HSV",
    "kmeans": "K-means",
    "glcm": "GLCM Texture",
    "hybrid": "Hybrid",
}


@dataclass
class MethodResult:
    """Common output format expected from every approach."""

    method_key: str
    method_name: str
    predicted_category: str | None = None
    confidence_percent: float | None = None
    processing_time_ms: float | None = None
    class_scores: dict[str, float | None] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    is_placeholder: bool = True


@dataclass
class EvaluationSnapshot:
    """Latest evaluation information displayed by the dashboard."""

    generated_at: str | None
    dataset_name: str
    dataset_split: str
    image_count: int
    failed_count: int
    status_message: str
    overall_rows: list[dict[str, Any]]
    class_rows: list[dict[str, Any]]
    confusion_matrices: dict[str, list[list[int]]]
    is_placeholder: bool = True


def placeholder_method_result(method_key: str) -> MethodResult:
    """Return an honest empty result for UI development."""

    if method_key not in METHOD_NAMES:
        raise ValueError(f"Unknown method key: {method_key}")

    return MethodResult(
        method_key=method_key,
        method_name=METHOD_NAMES[method_key],
        predicted_category=None,
        confidence_percent=None,
        processing_time_ms=None,
        class_scores={category: None for category in CATEGORIES},
        features={},
        notes=[
            "The interface is working.",
            "The image-processing approach has not been connected.",
        ],
        is_placeholder=True,
    )


def empty_evaluation_snapshot(
    evaluation_mode: str = "ripeness",
) -> EvaluationSnapshot:
    """Create the initial dashboard structure without fabricated results."""

    if evaluation_mode == "ripeness":
        categories = RIPENESS_CATEGORIES
        dataset_name = "Banana ripeness classification dataset"
        dataset_split = "Fixed test set"
    elif evaluation_mode == "quality":
        categories = QUALITY_CATEGORIES
        dataset_name = "Ripe banana quality dataset"
        dataset_split = "Quality dataset"
    else:
        raise ValueError(
            "evaluation_mode must be either 'ripeness' or 'quality'."
        )

    overall_rows = [
        {
            "Approach": approach,
            "Overall accuracy": None,
            "Macro F1": None,
            "Average processing time (ms)": None,
        }
        for approach in METHOD_NAMES.values()
    ]

    class_rows = [
        {
            "Approach": approach,
            "Category": category,
            "Precision": None,
            "Recall": None,
            "F1": None,
            "Support": None,
        }
        for approach in METHOD_NAMES.values()
        for category in categories
    ]

    confusion_matrices = {
        approach: [[0 for _ in categories] for _ in categories]
        for approach in METHOD_NAMES.values()
    }

    return EvaluationSnapshot(
        generated_at=None,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        image_count=0,
        failed_count=0,
        status_message="No real evaluation has been performed.",
        overall_rows=overall_rows,
        class_rows=class_rows,
        confusion_matrices=confusion_matrices,
        is_placeholder=True,
    )
