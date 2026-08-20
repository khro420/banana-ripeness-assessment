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

from branches.kmeans.kmeans_analysis import (
    KMeansRipenessBands,
    analyse_kmeans,
)

from branches.kmeans.kmeans_segmentation import (
    KMeansParameters,
)

from branches.hsv.hsv_analysis import (
    HSVRipenessBands,
    analyse_hsv,
)
from branches.hsv.hsv_segmentation import HSVParameters
from branches.morphology.morphology_analysis import (
    RipenessBands,
    analyse_morphology,
)
from branches.morphology.morphology_segmentation import (
    MorphologyParameters,
)

from branches.glcm.glcm_analysis import (
    GLCMRipenessBands,
    analyse_glcm,
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
    / "evaluation_predictions.csv"
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

    implemented_methods = {
        "morphology",
        "hsv",
        "kmeans",
    }

    methods = {
        method_key: _empty_method_result(
            method_key=method_key,
            implemented=(
                method_key in implemented_methods
            ),
        )
        for method_key in METHODS
    }

    return {
        "schema_version": 2,
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
        category_directory = TEST_DIRECTORY / folder_name

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
            image = ImageOps.exif_transpose(opened_image)
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


def _new_record(
    image_path: Path,
    actual_category: str,
) -> dict[str, Any]:
    relative_path = str(
        image_path.relative_to(PROJECT_ROOT)
    )

    return {
        "image_path": relative_path,
        "actual_category": actual_category,
        "preprocessing_time_ms": None,
        "segmentation_time_ms": None,
        "shared_status": "Failed",
        "shared_error": None,

        "morphology_predicted_category": "Failed",
        "morphology_correct": False,
        "morphology_confidence_percent": None,
        "blemish_percentage": None,
        "morphology_time_ms": None,
        "morphology_total_processing_time_ms": None,
        "morphology_status": "Failed",
        "morphology_error": None,

        "hsv_predicted_category": "Failed",
        "hsv_correct": False,
        "hsv_confidence_percent": None,
        "green_percentage": None,
        "yellow_percentage": None,
        "brown_percentage": None,
        "dark_percentage": None,
        "deteriorated_percentage": None,
        "other_percentage": None,
        "hsv_time_ms": None,
        "hsv_total_processing_time_ms": None,
        "hsv_status": "Failed",
        "hsv_error": None,

        "kmeans_predicted_category": "Failed",
        "kmeans_correct": False,
        "kmeans_confidence_percent": None,
        "kmeans_time_ms": None,
        "kmeans_total_processing_time_ms": None,
        "kmeans_status": "Failed",
        "kmeans_error": None,

        "glcm_predicted_category": "Failed",
        "glcm_correct": False,
        "glcm_confidence_percent": None,
        "glcm_time_ms": None,
        "glcm_total_processing_time_ms": None,
        "glcm_status": "Failed",
        "glcm_error": None,
    }


def _evaluate_image(
    image_path: Path,
    actual_category: str,
    morphology_parameters: MorphologyParameters,
    morphology_bands: RipenessBands,
    hsv_parameters: HSVParameters,
    hsv_bands: HSVRipenessBands,
    kmeans_parameters: KMeansParameters,
    kmeans_bands: KMeansRipenessBands,
    glcm_bands: GLCMRipenessBands,
) -> dict[str, Any]:
    """Evaluate all currently implemented methods on one image."""

    record = _new_record(
        image_path=image_path,
        actual_category=actual_category,
    )

    try:
        image = _open_rgb_image(image_path)

        preprocessing_start = perf_counter()
        prepared = standardise_image(
            image=image,
            upload_metadata={
                "filename": image_path.name,
            },
            target_size=(416, 416),
        )
        preprocessing_time_ms = (
            perf_counter() - preprocessing_start
        ) * 1000.0
        record["preprocessing_time_ms"] = preprocessing_time_ms

        segmentation_start = perf_counter()
        banana_segmentation = segment_banana(
            rgb_image=prepared.working_rgb,
            content_mask=prepared.content_mask,
        )
        segmentation_time_ms = (
            perf_counter() - segmentation_start
        ) * 1000.0
        record["segmentation_time_ms"] = segmentation_time_ms

        shared_time_ms = (
            preprocessing_time_ms
            + segmentation_time_ms
        )

        if not banana_segmentation.success:
            message = (
                "Banana segmentation failed: "
                f"{banana_segmentation.message}"
            )
            record["shared_error"] = message
            record["morphology_error"] = message
            record["hsv_error"] = message
            record["kmeans_error"] = message
            record["glcm_error"] = message
            record["morphology_total_processing_time_ms"] = shared_time_ms
            record["hsv_total_processing_time_ms"] = shared_time_ms
            record["kmeans_total_processing_time_ms"] = shared_time_ms
            record["glcm_total_processing_time_ms"] = shared_time_ms

            return record

        record["shared_status"] = "Completed"

        # Morphology analysis
        try:
            morphology = analyse_morphology(
                rgb_image=prepared.working_rgb,
                banana_mask=banana_segmentation.final_mask,
                parameters=morphology_parameters,
                bands=morphology_bands,
            )

            record.update(
                {
                    "morphology_predicted_category": morphology.predicted_category,
                    "morphology_correct": (
                        morphology.predicted_category
                        == actual_category
                    ),
                    "morphology_confidence_percent": morphology.confidence_percent,
                    "blemish_percentage": morphology.blemish_percentage,
                    "morphology_time_ms": morphology.processing_time_ms,
                    "morphology_total_processing_time_ms": (
                        shared_time_ms
                        + morphology.processing_time_ms
                    ),
                    "morphology_status": "Completed",
                    "morphology_error": None,
                }
            )

        except Exception as error:
            record["morphology_error"] = str(error)
            record["morphology_total_processing_time_ms"] = shared_time_ms

        # HSV analysis
        try:
            hsv = analyse_hsv(
                rgb_image=prepared.working_rgb,
                banana_mask=banana_segmentation.final_mask,
                parameters=hsv_parameters,
                bands=hsv_bands,
            )

            record.update(
                {
                    "hsv_predicted_category": hsv.predicted_category,
                    "hsv_correct": (
                        hsv.predicted_category
                        == actual_category
                    ),
                    "hsv_confidence_percent": hsv.confidence_percent,
                    "green_percentage": hsv.green_percentage,
                    "yellow_percentage": hsv.yellow_percentage,
                    "brown_percentage": hsv.brown_percentage,
                    "dark_percentage": hsv.dark_percentage,
                    "deteriorated_percentage": hsv.deteriorated_percentage,
                    "other_percentage": hsv.other_percentage,
                    "hsv_time_ms": hsv.processing_time_ms,
                    "hsv_total_processing_time_ms": (
                        shared_time_ms
                        + hsv.processing_time_ms
                    ),
                    "hsv_status": "Completed",
                    "hsv_error": None,
                }
            )

        except Exception as error:
            record["hsv_error"] = str(error)
            record["hsv_total_processing_time_ms"] = shared_time_ms

        # K-means analysis
        try:
            kmeans = analyse_kmeans(
                rgb_image=prepared.working_rgb,
                banana_mask=banana_segmentation.final_mask,
                parameters=kmeans_parameters,
                bands=kmeans_bands,
            )

            record.update(
                {
                    "kmeans_predicted_category": kmeans.predicted_category,
                    "kmeans_correct": (
                        kmeans.predicted_category == actual_category
                    ),
                    "kmeans_confidence_percent": kmeans.confidence_percent,
                    "kmeans_time_ms": kmeans.processing_time_ms,
                    "kmeans_total_processing_time_ms": (
                        shared_time_ms + kmeans.processing_time_ms
                    ),
                    "kmeans_status": "Completed",
                    "kmeans_error": None,
                }
            )

        except Exception as error:
            record["kmeans_error"] = str(error)
            record["kmeans_total_processing_time_ms"] = shared_time_ms

        # GLCM analysis
        try:
            glcm = analyse_glcm(
                rgb_image=prepared.working_rgb,
                banana_mask=banana_segmentation.final_mask,
                bands=glcm_bands,
            )

            record.update(
                {
                    "glcm_predicted_category": glcm.predicted_category,
                    "glcm_correct": (
                        glcm.predicted_category == actual_category
                    ),
                    "glcm_confidence_percent": (
                        glcm.confidence_percent
                    ),
                    "glcm_time_ms": glcm.processing_time_ms,
                    "glcm_total_processing_time_ms": (
                        shared_time_ms + glcm.processing_time_ms
                    ),
                    "glcm_status": "Completed",
                    "glcm_error": None,
                }
            )

        except Exception as error:
            record["glcm_error"] = str(error)
            record["glcm_total_processing_time_ms"] = shared_time_ms
        return record

    except Exception as error:
        message = str(error)
        record["shared_error"] = message
        record["morphology_error"] = message
        record["hsv_error"] = message
        record["kmeans_error"] = message
        record["glcm_error"] = message
        return record


def _calculate_method_metrics(
    records: list[dict[str, Any]],
    method_key: str,
) -> dict[str, Any]:
    """Calculate metrics while counting method failures as incorrect."""

    predicted_field = f"{method_key}_predicted_category"
    status_field = f"{method_key}_status"
    time_field = f"{method_key}_total_processing_time_ms"

    actual_values = [
        record["actual_category"]
        for record in records
    ]
    predicted_values = [
        record[predicted_field]
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

    predicted_labels = [
        *CATEGORIES,
        "Failed",
    ]

    full_matrix = confusion_matrix(
        actual_values,
        predicted_values,
        labels=predicted_labels,
    )

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
        float(record[time_field])
        for record in records
        if record[time_field] is not None
    ]

    successful_count = sum(
        record[status_field] == "Completed"
        for record in records
    )
    failed_count = len(records) - successful_count

    return {
        "method_key": method_key,
        "approach": METHODS[method_key],
        "implemented": True,
        "status": (
            "Completed"
            if failed_count == 0
            else f"Completed with {failed_count} failures"
        ),
        "overall_accuracy": float(overall_accuracy),
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

    fieldnames = list(records[0].keys()) if records else []

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

    temporary_file.replace(LATEST_RESULT_FILE)


def run_fixed_dataset_evaluation(
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Evaluate Morphology, HSV and K-means on dataset/test.

    Decision thresholds should remain frozen during final test evaluation.
    """

    discovered_images = _discover_test_images()

    morphology_parameters = MorphologyParameters()
    morphology_bands = RipenessBands()
    hsv_parameters = HSVParameters()
    hsv_bands = HSVRipenessBands()
    kmeans_parameters = KMeansParameters()
    kmeans_bands = KMeansRipenessBands()
    glcm_bands = GLCMRipenessBands()

    records = []
    total_images = len(discovered_images)

    for index, item in enumerate(
        discovered_images,
        start=1,
    ):
        record = _evaluate_image(
            image_path=item["path"],
            actual_category=item["actual_category"],
            morphology_parameters=morphology_parameters,
            morphology_bands=morphology_bands,
            hsv_parameters=hsv_parameters,
            hsv_bands=hsv_bands,
            kmeans_parameters=kmeans_parameters,
            kmeans_bands=kmeans_bands,
            glcm_bands=glcm_bands,
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

    morphology_metrics = _calculate_method_metrics(
        records=records,
        method_key="morphology",
    )
    hsv_metrics = _calculate_method_metrics(
        records=records,
        method_key="hsv",
    )

    kmeans_metrics = _calculate_method_metrics(
        records=records,
        method_key="kmeans",
    )
    glcm_metrics = _calculate_method_metrics(
        records=records,
        method_key="glcm",
)

    methods = {
        "morphology": morphology_metrics,
        "hsv": hsv_metrics,
        "kmeans": kmeans_metrics,
        "glcm": glcm_metrics,
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

    successful_images = sum(
    record["morphology_status"] == "Completed"
    and record["hsv_status"] == "Completed"
    and record["kmeans_status"] == "Completed"
    and record["glcm_status"] == "Completed"
    for record in records
)
    failed_images = len(records) - successful_images

    report = {
        "schema_version": 2,
        "status": (
            "Evaluation completed for Morphology, HSV and K-means and GLCM."
        ),
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "dataset_split": "test",
        "dataset_directory": "dataset/test",
        "image_count": len(records),
        "successful_images": successful_images,
        "failed_images": failed_images,
        "class_counts": class_counts,
        "methods": methods,
        "predictions_file": str(
            PREDICTION_FILE.relative_to(PROJECT_ROOT)
        ),
        "configuration": {
            "morphology_parameters": asdict(
                morphology_parameters
            ),
            "morphology_ripeness_bands": asdict(
                morphology_bands
            ),
            "hsv_parameters": asdict(
                hsv_parameters
            ),
            "hsv_ripeness_bands": asdict(
                hsv_bands
            ),
            "kmeans_parameters": asdict(
                kmeans_parameters
            ),
            "kmeans_ripeness_bands": asdict(
                kmeans_bands
            ),
            "glcm_ripeness_bands": asdict(
                glcm_bands
            ),
        },
    }

    _save_predictions(records)
    _save_report(report)

    return report
