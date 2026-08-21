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

from branches.glcm.glcm_analysis import (
    GLCMParameters,
    GLCMRipenessBands,
    analyse_glcm,
)
from branches.hsv.hsv_analysis import HSVRipenessBands, analyse_hsv
from branches.hsv.hsv_segmentation import HSVParameters
from branches.kmeans.kmeans_analysis import (
    KMeansRipenessBands,
    analyse_kmeans,
)
from branches.kmeans.kmeans_segmentation import KMeansParameters
from branches.morphology.morphology_analysis import (
    RipenessBands,
    analyse_morphology,
)
from branches.morphology.morphology_segmentation import MorphologyParameters
from core.banana_segmentation import segment_banana
from core.hybrid import CLASS_RELIABILITY_WEIGHTS, combine_method_results
from core.image_handling import standardise_image
from core.result_schema import CATEGORIES, METHOD_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = PROJECT_ROOT / "dataset" / "test"
RESULT_DIRECTORY = PROJECT_ROOT / "outputs" / "evaluation_results"
LATEST_RESULT_FILE = RESULT_DIRECTORY / "latest_evaluation.json"

FOLDER_TO_CATEGORY = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "overripe": "Overripe",
    "rotten": "Rotten",
}

METHODS = dict(METHOD_NAMES)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ProgressCallback = Callable[[int, int, str], None]


class EvaluationError(RuntimeError):
    """Raised when fixed-dataset evaluation cannot proceed."""


def _empty_method_result(method_key: str) -> dict[str, Any]:
    return {
        "method_key": method_key,
        "approach": METHODS[method_key],
        "implemented": True,
        "status": "Not evaluated",
        "overall_accuracy": None,
        "macro_f1": None,
        "average_processing_time_ms": None,
        "successful_images": None,
        "failed_images": None,
        "per_class": {
            category: {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": None,
            }
            for category in CATEGORIES
        },
        "confusion_matrix": None,
    }


def create_empty_evaluation() -> dict[str, Any]:
    """Create an honest empty report for the dashboard."""

    return {
        "schema_version": 4,
        "status": "No evaluation has been run.",
        "generated_at": None,
        "dataset_split": "test",
        "dataset_directory": "dataset/test",
        "image_count": 0,
        "successful_images": 0,
        "failed_images": 0,
        "class_counts": {category: 0 for category in CATEGORIES},
        "methods": {
            key: _empty_method_result(key)
            for key in METHODS
        },
        "configuration": None,
    }


def load_latest_evaluation() -> dict[str, Any]:
    if not LATEST_RESULT_FILE.exists():
        return create_empty_evaluation()

    try:
        with LATEST_RESULT_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        report = create_empty_evaluation()
        report["status"] = "The saved evaluation report could not be read."
        return report


def _discover_test_images() -> list[dict[str, Any]]:
    if not TEST_DIRECTORY.exists():
        raise EvaluationError(f"Test directory not found: {TEST_DIRECTORY}")

    images = []

    for folder_name, category in FOLDER_TO_CATEGORY.items():
        folder = TEST_DIRECTORY / folder_name

        if not folder.exists():
            raise EvaluationError(
                f"Required test class directory is missing: {folder}"
            )

        paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not paths:
            raise EvaluationError(f"No test images found for {category}.")

        images.extend(
            {"path": path, "actual_category": category}
            for path in paths
        )

    return images


def _open_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise EvaluationError(f"Cannot read image: {path.name}") from error


def _new_record(path: Path, actual: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "image_path": str(path.relative_to(PROJECT_ROOT)),
        "actual_category": actual,
        "preprocessing_time_ms": None,
        "segmentation_time_ms": None,
        "shared_status": "Failed",
        "shared_error": None,
    }

    for key in METHODS:
        record.update(
            {
                f"{key}_predicted_category": "Failed",
                f"{key}_correct": False,
                f"{key}_confidence_percent": None,
                f"{key}_time_ms": None,
                f"{key}_total_processing_time_ms": None,
                f"{key}_status": "Failed",
                f"{key}_error": None,
            }
        )

    record.update(
        {
            "blemish_percentage": None,
            "green_percentage": None,
            "yellow_percentage": None,
            "brown_percentage": None,
            "dark_percentage": None,
            "deteriorated_percentage": None,
            "other_percentage": None,
            "hybrid_methods_used": None,
            "hybrid_agreement_count": None,
            "hybrid_agreement_percent": None,
            "hybrid_winning_margin_percent": None,
        }
    )

    return record


def _fail_all(
    record: dict[str, Any],
    message: str,
    shared_time_ms: float | None,
) -> None:
    record["shared_error"] = message

    for key in METHODS:
        record[f"{key}_error"] = message
        record[f"{key}_total_processing_time_ms"] = shared_time_ms


def _store_result(
    record: dict[str, Any],
    key: str,
    analysis: Any,
    actual: str,
    shared_time_ms: float,
) -> None:
    result = analysis.method_result
    method_time_ms = float(analysis.processing_time_ms)

    record.update(
        {
            f"{key}_predicted_category": result.predicted_category,
            f"{key}_correct": result.predicted_category == actual,
            f"{key}_confidence_percent": result.confidence_percent,
            f"{key}_time_ms": method_time_ms,
            f"{key}_total_processing_time_ms": (
                shared_time_ms + method_time_ms
            ),
            f"{key}_status": "Completed",
            f"{key}_error": None,
        }
    )

    if key == "morphology":
        record["blemish_percentage"] = analysis.blemish_percentage
    elif key == "hsv":
        record.update(
            {
                "green_percentage": analysis.green_percentage,
                "yellow_percentage": analysis.yellow_percentage,
                "brown_percentage": analysis.brown_percentage,
                "dark_percentage": analysis.dark_percentage,
                "deteriorated_percentage": analysis.deteriorated_percentage,
                "other_percentage": analysis.other_percentage,
            }
        )
    elif key == "hybrid":
        record.update(
            {
                "hybrid_methods_used": analysis.methods_used,
                "hybrid_agreement_count": analysis.agreement_count,
                "hybrid_agreement_percent": analysis.agreement_percent,
                "hybrid_winning_margin_percent": (
                    analysis.winning_margin_percent
                ),
            }
        )


def _evaluate_image(
    image_path: Path,
    actual_category: str,
    morphology_parameters: MorphologyParameters,
    morphology_bands: RipenessBands,
    hsv_parameters: HSVParameters,
    hsv_bands: HSVRipenessBands,
    kmeans_parameters: KMeansParameters,
    kmeans_bands: KMeansRipenessBands,
    glcm_parameters: GLCMParameters,
    glcm_bands: GLCMRipenessBands,
) -> dict[str, Any]:
    record = _new_record(image_path, actual_category)

    try:
        image = _open_rgb_image(image_path)

        start = perf_counter()
        prepared = standardise_image(
            image=image,
            upload_metadata={"filename": image_path.name},
            target_size=(416, 416),
        )
        preprocessing_ms = (perf_counter() - start) * 1000.0
        record["preprocessing_time_ms"] = preprocessing_ms

        start = perf_counter()
        segmentation = segment_banana(
            rgb_image=prepared.working_rgb,
            content_mask=prepared.content_mask,
        )
        segmentation_ms = (perf_counter() - start) * 1000.0
        record["segmentation_time_ms"] = segmentation_ms
        shared_time_ms = preprocessing_ms + segmentation_ms

        if not segmentation.success:
            _fail_all(
                record,
                f"Banana segmentation failed: {segmentation.message}",
                shared_time_ms,
            )
            return record

        record["shared_status"] = "Completed"

        jobs = {
            "morphology": lambda: analyse_morphology(
                rgb_image=prepared.working_rgb,
                banana_mask=segmentation.final_mask,
                parameters=morphology_parameters,
                bands=morphology_bands,
            ),
            "hsv": lambda: analyse_hsv(
                rgb_image=prepared.working_rgb,
                banana_mask=segmentation.final_mask,
                parameters=hsv_parameters,
                bands=hsv_bands,
            ),
            "kmeans": lambda: analyse_kmeans(
                rgb_image=prepared.working_rgb,
                banana_mask=segmentation.final_mask,
                parameters=kmeans_parameters,
                bands=kmeans_bands,
            ),
            "glcm": lambda: analyse_glcm(
                rgb_image=prepared.working_rgb,
                banana_mask=segmentation.final_mask,
                parameters=glcm_parameters,
                bands=glcm_bands,
            ),
        }

        analyses = {}

        for key, job in jobs.items():
            branch_start = perf_counter()

            try:
                analysis = job()
                analyses[key] = analysis
                _store_result(
                    record,
                    key,
                    analysis,
                    actual_category,
                    shared_time_ms,
                )
            except Exception as error:
                failed_branch_ms = (
                    perf_counter() - branch_start
                ) * 1000.0
                record[f"{key}_error"] = str(error)
                record[f"{key}_total_processing_time_ms"] = (
                    shared_time_ms + failed_branch_ms
                )

        successful_branch_ms = sum(
            float(analysis.processing_time_ms)
            for analysis in analyses.values()
        )

        try:
            hybrid = combine_method_results(
                [analysis.method_result for analysis in analyses.values()]
            )
            _store_result(
                record,
                "hybrid",
                hybrid,
                actual_category,
                shared_time_ms,
            )
        except Exception as error:
            record["hybrid_error"] = str(error)
            record["hybrid_total_processing_time_ms"] = (
                shared_time_ms + successful_branch_ms
            )

        return record

    except Exception as error:
        _fail_all(record, str(error), None)
        return record


def _calculate_method_metrics(
    records: list[dict[str, Any]],
    method_key: str,
) -> dict[str, Any]:
    predicted_field = f"{method_key}_predicted_category"
    status_field = f"{method_key}_status"
    time_field = f"{method_key}_total_processing_time_ms"

    actual = [record["actual_category"] for record in records]
    predicted = [record[predicted_field] for record in records]

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(CATEGORIES),
        zero_division=0,
    )

    labels = [*CATEGORIES, "Failed"]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    matrix = matrix[: len(CATEGORIES), :]

    times = [
        float(record[time_field])
        for record in records
        if record[time_field] is not None
    ]
    successes = sum(
        record[status_field] == "Completed"
        for record in records
    )
    failures = len(records) - successes

    return {
        "method_key": method_key,
        "approach": METHODS[method_key],
        "implemented": True,
        "status": (
            "Completed"
            if failures == 0
            else f"Completed with {failures} failures"
        ),
        "overall_accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=list(CATEGORIES),
                average="macro",
                zero_division=0,
            )
        ),
        "average_processing_time_ms": (
            float(np.mean(times)) if times else None
        ),
        "successful_images": successes,
        "failed_images": failures,
        "per_class": {
            category: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, category in enumerate(CATEGORIES)
        },
        "confusion_matrix": {
            "actual_labels": list(CATEGORIES),
            "predicted_labels": labels,
            "values": matrix.tolist(),
        },
    }


def _save_report(report: dict[str, Any]) -> None:
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary = LATEST_RESULT_FILE.with_suffix(".json.tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    temporary.replace(LATEST_RESULT_FILE)


def run_fixed_dataset_evaluation(
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Evaluate all individual approaches and the hybrid on dataset/test."""

    images = _discover_test_images()

    morphology_parameters = MorphologyParameters()
    morphology_bands = RipenessBands()
    hsv_parameters = HSVParameters()
    hsv_bands = HSVRipenessBands()
    kmeans_parameters = KMeansParameters()
    kmeans_bands = KMeansRipenessBands()
    glcm_parameters = GLCMParameters()
    glcm_bands = GLCMRipenessBands()

    records = []

    for index, item in enumerate(images, start=1):
        records.append(
            _evaluate_image(
                image_path=item["path"],
                actual_category=item["actual_category"],
                morphology_parameters=morphology_parameters,
                morphology_bands=morphology_bands,
                hsv_parameters=hsv_parameters,
                hsv_bands=hsv_bands,
                kmeans_parameters=kmeans_parameters,
                kmeans_bands=kmeans_bands,
                glcm_parameters=glcm_parameters,
                glcm_bands=glcm_bands,
            )
        )

        if progress_callback is not None:
            progress_callback(
                index,
                len(images),
                f"Evaluating {index}/{len(images)}: {item['path'].name}",
            )

    method_metrics = {
        key: _calculate_method_metrics(records, key)
        for key in METHODS
    }

    successful_images = sum(
        all(
            record[f"{key}_status"] == "Completed"
            for key in METHODS
        )
        for record in records
    )

    report = {
        "schema_version": 4,
        "status": (
            "Evaluation completed for Morphology, HSV, K-means, "
            "GLCM Texture and Hybrid."
        ),
        "generated_at": (
            datetime.now().astimezone().isoformat(timespec="seconds")
        ),
        "dataset_split": "test",
        "dataset_directory": "dataset/test",
        "image_count": len(records),
        "successful_images": successful_images,
        "failed_images": len(records) - successful_images,
        "class_counts": {
            category: sum(
                record["actual_category"] == category
                for record in records
            )
            for category in CATEGORIES
        },
        "methods": method_metrics,
        "configuration": {
            "morphology_parameters": asdict(morphology_parameters),
            "morphology_ripeness_bands": asdict(morphology_bands),
            "hsv_parameters": asdict(hsv_parameters),
            "hsv_ripeness_bands": asdict(hsv_bands),
            "kmeans_parameters": asdict(kmeans_parameters),
            "kmeans_ripeness_bands": asdict(kmeans_bands),
            "glcm_parameters": asdict(glcm_parameters),
            "glcm_ripeness_bands": asdict(glcm_bands),
            "hybrid_class_reliability_weights": (
                CLASS_RELIABILITY_WEIGHTS
            ),
        },
    }

    _save_report(report)

    return report