import csv
import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from branches.morphology.morphology_analysis import (
    RipenessBands,
    analyse_morphology,
)
from branches.morphology.morphology_segmentation import (
    MorphologyParameters,
)
from core.banana_segmentation import segment_banana
from core.image_handling import standardise_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = PROJECT_ROOT / "dataset" / "test"

RESULT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_results"
)

LATEST_RESULT_FILE = (
    RESULT_DIRECTORY
    / "latest_evaluation.json"
)

PREDICTION_FILE = (
    RESULT_DIRECTORY
    / "morphology_predictions.csv"
)

CATEGORIES = (
    "Unripe",
    "Ripe",
    "Overripe",
    "Rotten",
)

FOLDER_TO_CATEGORY = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "overripe": "Overripe",
    "rotten": "Rotten",
}

METHODS = {
    "morphology": "Morphology",
    "hsv": "HSV",
    "kmeans": "K-means",
    "glcm": "GLCM Texture",
    "hybrid": "Hybrid",
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

ProgressCallback = Callable[
    [int, int, str],
    None,
]


class EvaluationError(RuntimeError):
    """Raised when fixed-dataset evaluation cannot proceed."""


def _empty_class_metrics() -> dict[str, dict[str, Any]]:
    return {
        category: {
            "precision": None,
            "recall": None,
            "f1": None,
            "support": None,
        }
        for category in CATEGORIES
    }


def _empty_method_result(
    method_key: str,
    implemented: bool,
) -> dict[str, Any]:
    return {
        "method_key": method_key,
        "approach": METHODS[method_key],
        "implemented": implemented,
        "status": (
            "Not evaluated"
            if implemented
            else "Not implemented"
        ),
        "overall_accuracy": None,
        "macro_f1": None,
        "average_processing_time_ms": None,
        "successful_images": None,
        "failed_images": None,
        "per_class": _empty_class_metrics(),
        "confusion_matrix": None,
    }


def create_empty_evaluation() -> dict[str, Any]:
    """Return an empty dashboard report without fabricated values."""

    methods = {
        method_key: _empty_method_result(
            method_key=method_key,
            implemented=(
                method_key == "morphology"
            ),
        )
        for method_key in METHODS
    }

    return {
        "schema_version": 1,
        "status": "No evaluation has been run.",
        "generated_at": None,
        "dataset_split": "test",
        "dataset_directory": "dataset/test",
        "image_count": 0,
        "successful_images": 0,
        "failed_images": 0,
        "class_counts": {
            category: 0
            for category in CATEGORIES
        },
        "methods": methods,
        "predictions_file": None,
        "configuration": None,
    }


def load_latest_evaluation() -> dict[str, Any]:
    """Load the most recently saved evaluation report."""

    if not LATEST_RESULT_FILE.exists():
        return create_empty_evaluation()

    try:
        with LATEST_RESULT_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError):
        report = create_empty_evaluation()
        report["status"] = (
            "The saved evaluation report could not be read."
        )

        return report


def _discover_test_images() -> list[dict[str, Any]]:
    """Find labelled images in the fixed test directory."""

    if not TEST_DIRECTORY.exists():
        raise EvaluationError(
            f"Test directory not found: {TEST_DIRECTORY}"
        )

    discovered_images: list[dict[str, Any]] = []

    for folder_name, category in FOLDER_TO_CATEGORY.items():
        category_directory = (
            TEST_DIRECTORY / folder_name
        )

        if not category_directory.exists():
            raise EvaluationError(
                "Required test class directory is missing: "
                f"{category_directory}"
            )

        image_paths = sorted(
            path
            for path in category_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )

        if not image_paths:
            raise EvaluationError(
                f"No test images found for {category}."
            )

        for image_path in image_paths:
            discovered_images.append(
                {
                    "path": image_path,
                    "actual_category": category,
                }
            )

    if not discovered_images:
        raise EvaluationError(
            "No supported images were found in dataset/test."
        )

    return discovered_images


def _open_rgb_image(image_path: Path) -> Image.Image:
    """Open one test image and correct its EXIF orientation."""

    try:
        with Image.open(image_path) as opened_image:
            image = ImageOps.exif_transpose(
                opened_image
            )
            image = image.convert("RGB")
            image.load()

            return image.copy()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise EvaluationError(
            f"Cannot read image: {image_path.name}"
        ) from error


def _evaluate_morphology_image(
    image_path: Path,
    actual_category: str,
    parameters: MorphologyParameters,
    bands: RipenessBands,
) -> dict[str, Any]:
    """Evaluate morphology on one labelled image."""

    relative_path = str(
        image_path.relative_to(PROJECT_ROOT)
    )

    record = {
        "image_path": relative_path,
        "actual_category": actual_category,
        "predicted_category": "Failed",
        "correct": False,
        "confidence_percent": None,
        "blemish_percentage": None,
        "preprocessing_time_ms": None,
        "segmentation_time_ms": None,
        "morphology_time_ms": None,
        "total_processing_time_ms": None,
        "status": "Failed",
        "error": None,
    }

    processing_start = None

    try:
        # Disk-reading time is intentionally excluded.
        image = _open_rgb_image(image_path)

        processing_start = perf_counter()

        preprocessing_start = perf_counter()

        prepared = standardise_image(
            image=image,
            upload_metadata={
                "filename": image_path.name,
            },
            target_size=(416, 416),
        )

        record["preprocessing_time_ms"] = (
            perf_counter() - preprocessing_start
        ) * 1000.0

        segmentation_start = perf_counter()

        banana_segmentation = segment_banana(
            rgb_image=prepared.working_rgb,
            content_mask=prepared.content_mask,
        )

        record["segmentation_time_ms"] = (
            perf_counter() - segmentation_start
        ) * 1000.0

        if not banana_segmentation.success:
            record["error"] = (
                "Banana segmentation failed: "
                f"{banana_segmentation.message}"
            )

            record["total_processing_time_ms"] = (
                perf_counter() - processing_start
            ) * 1000.0

            return record

        analysis = analyse_morphology(
            rgb_image=prepared.working_rgb,

            # The complete banana, including its tips.
            banana_mask=(
                banana_segmentation.final_mask
            ),

            parameters=parameters,
            bands=bands,
        )

        record.update(
            {
                "predicted_category": (
                    analysis.predicted_category
                ),
                "correct": (
                    analysis.predicted_category
                    == actual_category
                ),
                "confidence_percent": (
                    analysis.confidence_percent
                ),
                "blemish_percentage": (
                    analysis.blemish_percentage
                ),
                "morphology_time_ms": (
                    analysis.processing_time_ms
                ),
                "status": "Completed",
                "error": None,
            }
        )

        record["total_processing_time_ms"] = (
            perf_counter() - processing_start
        ) * 1000.0

        return record

    except Exception as error:
        record["error"] = str(error)

        if processing_start is not None:
            record["total_processing_time_ms"] = (
                perf_counter() - processing_start
            ) * 1000.0

        return record


def _calculate_morphology_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate metrics while counting failures as incorrect."""

    actual_values = [
        record["actual_category"]
        for record in records
    ]

    predicted_values = [
        record["predicted_category"]
        for record in records
    ]

    overall_accuracy = accuracy_score(
        actual_values,
        predicted_values,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual_values,
            predicted_values,
            labels=list(CATEGORIES),
            zero_division=0,
        )
    )

    macro_f1 = f1_score(
        actual_values,
        predicted_values,
        labels=list(CATEGORIES),
        average="macro",
        zero_division=0,
    )

    # A fifth predicted column exposes processing failures.
    predicted_labels = [
        *CATEGORIES,
        "Failed",
    ]

    full_matrix = confusion_matrix(
        actual_values,
        predicted_values,
        labels=predicted_labels,
    )

    # Actual labels are always one of the four real categories.
    outcome_matrix = full_matrix[
        :len(CATEGORIES),
        :,
    ]

    per_class = {}

    for index, category in enumerate(CATEGORIES):
        per_class[category] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    measured_times = [
        float(
            record["total_processing_time_ms"]
        )
        for record in records
        if (
            record["total_processing_time_ms"]
            is not None
        )
    ]

    successful_count = sum(
        record["status"] == "Completed"
        for record in records
    )

    failed_count = len(records) - successful_count

    return {
        "method_key": "morphology",
        "approach": METHODS["morphology"],
        "implemented": True,
        "status": (
            "Completed"
            if failed_count == 0
            else f"Completed with {failed_count} failures"
        ),
        "overall_accuracy": float(
            overall_accuracy
        ),
        "macro_f1": float(macro_f1),
        "average_processing_time_ms": (
            float(np.mean(measured_times))
            if measured_times
            else None
        ),
        "successful_images": successful_count,
        "failed_images": failed_count,
        "per_class": per_class,
        "confusion_matrix": {
            "actual_labels": list(CATEGORIES),
            "predicted_labels": predicted_labels,
            "values": outcome_matrix.tolist(),
        },
    }


def _save_predictions(
    records: list[dict[str, Any]],
) -> None:
    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "image_path",
        "actual_category",
        "predicted_category",
        "correct",
        "confidence_percent",
        "blemish_percentage",
        "preprocessing_time_ms",
        "segmentation_time_ms",
        "morphology_time_ms",
        "total_processing_time_ms",
        "status",
        "error",
    ]

    with PREDICTION_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


def _save_report(report: dict[str, Any]) -> None:
    """Save the report atomically."""

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = LATEST_RESULT_FILE.with_suffix(
        ".json.tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_file.replace(
        LATEST_RESULT_FILE
    )


def run_fixed_dataset_evaluation(
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Evaluate the current morphology implementation on dataset/test.

    Other approaches remain explicitly unimplemented and empty.
    """

    discovered_images = _discover_test_images()

    parameters = MorphologyParameters()
    bands = RipenessBands()

    records = []

    total_images = len(discovered_images)

    for index, item in enumerate(
        discovered_images,
        start=1,
    ):
        record = _evaluate_morphology_image(
            image_path=item["path"],
            actual_category=item[
                "actual_category"
            ],
            parameters=parameters,
            bands=bands,
        )

        records.append(record)

        if progress_callback is not None:
            progress_callback(
                index,
                total_images,
                (
                    f"Evaluating {index}/{total_images}: "
                    f"{item['path'].name}"
                ),
            )

    morphology_metrics = (
        _calculate_morphology_metrics(records)
    )

    methods = {
        "morphology": morphology_metrics,
        "hsv": _empty_method_result(
            "hsv",
            implemented=False,
        ),
        "kmeans": _empty_method_result(
            "kmeans",
            implemented=False,
        ),
        "glcm": _empty_method_result(
            "glcm",
            implemented=False,
        ),
        "hybrid": _empty_method_result(
            "hybrid",
            implemented=False,
        ),
    }

    class_counts = {
        category: sum(
            record["actual_category"] == category
            for record in records
        )
        for category in CATEGORIES
    }

    report = {
        "schema_version": 1,
        "status": "Evaluation completed.",
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "dataset_split": "test",
        "dataset_directory": "dataset/test",
        "image_count": len(records),
        "successful_images": (
            morphology_metrics["successful_images"]
        ),
        "failed_images": (
            morphology_metrics["failed_images"]
        ),
        "class_counts": class_counts,
        "methods": methods,
        "predictions_file": str(
            PREDICTION_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "configuration": {
            "morphology_parameters": asdict(
                parameters
            ),
            "ripeness_bands": asdict(bands),
        },
    }

    _save_predictions(records)
    _save_report(report)

    return report