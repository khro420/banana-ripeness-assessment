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
    GLCMQualityBands,
    GLCMRipenessBands,
    analyse_glcm,
    analyse_glcm_quality,
)
from branches.hsv.hsv_analysis import (
    HSVQualityBands,
    HSVRipenessBands,
    analyse_hsv,
    analyse_hsv_quality,
)
from branches.hsv.hsv_segmentation import HSVParameters
from branches.kmeans.kmeans_analysis import (
    KMeansQualityBands,
    KMeansRipenessBands,
    analyse_kmeans,
    analyse_kmeans_quality,
)
from branches.kmeans.kmeans_segmentation import KMeansParameters
from branches.morphology.morphology_analysis import (
    QualityBands,
    RipenessBands,
    analyse_morphology,
    analyse_morphology_quality,
)
from branches.morphology.morphology_segmentation import MorphologyParameters
from core.banana_segmentation import segment_banana
from core.hybrid import CLASS_RELIABILITY_WEIGHTS, combine_method_results
from core.image_handling import standardise_image
from core.result_schema import (
    CATEGORIES,
    METHOD_NAMES,
    QUALITY_CATEGORIES,
    RIPENESS_CATEGORIES,
)


EVALUATION_MODE_RIPENESS = "ripeness"
EVALUATION_MODE_QUALITY = "quality"
EVALUATION_MODE_LABELS = {
    EVALUATION_MODE_RIPENESS: "Ripeness",
    EVALUATION_MODE_QUALITY: "Quality",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RIPENESS_TEST_DIRECTORY = PROJECT_ROOT / "dataset" / "ripeness" / "test"
QUALITY_DIRECTORY = PROJECT_ROOT / "dataset" / "quality" / "test"
RESULT_DIRECTORY = PROJECT_ROOT / "outputs" / "evaluation_results"

LATEST_RESULT_FILES = {
    EVALUATION_MODE_RIPENESS: (
        RESULT_DIRECTORY / "latest_ripeness_evaluation.json"
    ),
    EVALUATION_MODE_QUALITY: (
        RESULT_DIRECTORY / "latest_quality_evaluation.json"
    ),
}
LEGACY_RIPENESS_RESULT_FILE = RESULT_DIRECTORY / "latest_evaluation.json"

RIPENESS_FOLDER_TO_CATEGORY = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "overripe": "Overripe",
    "rotten": "Rotten",
}
QUALITY_FOLDER_TO_CATEGORY = {
    "class_a": "Class_A",
    "class_b": "Class_B",
    "defect": "Defect",
}

METHODS = dict(METHOD_NAMES)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ProgressCallback = Callable[[int, int, str], None]


class EvaluationError(RuntimeError):
    """Raised when a fixed-dataset evaluation cannot proceed."""


def _normalise_mode(mode: str) -> str:
    normalised = str(mode).strip().lower()
    if normalised not in EVALUATION_MODE_LABELS:
        raise ValueError("mode must be either 'ripeness' or 'quality'.")
    return normalised


def _categories_for_mode(mode: str) -> tuple[str, ...]:
    if mode == EVALUATION_MODE_QUALITY:
        return QUALITY_CATEGORIES
    return RIPENESS_CATEGORIES


def _implemented_methods(mode: str) -> set[str]:
    if mode == EVALUATION_MODE_QUALITY:
        return {"morphology", "hsv", "kmeans", "glcm"}
    return set(METHODS)


def _dataset_definition(
    mode: str,
) -> tuple[Path, dict[str, str], str, str]:
    if mode == EVALUATION_MODE_QUALITY:
        return (
            QUALITY_DIRECTORY,
            QUALITY_FOLDER_TO_CATEGORY,
            "dataset/quality/test",
            "test",
        )
    return (
        RIPENESS_TEST_DIRECTORY,
        RIPENESS_FOLDER_TO_CATEGORY,
        "dataset/ripeness/test",
        "test",
    )


def _empty_method_result(
    method_key: str,
    categories: tuple[str, ...],
    implemented: bool,
) -> dict[str, Any]:
    return {
        "method_key": method_key,
        "approach": METHODS[method_key],
        "implemented": implemented,
        "status": "Not evaluated" if implemented else "Not implemented",
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
            for category in categories
        },
        "confusion_matrix": None,
    }


def create_empty_evaluation(
    mode: str = EVALUATION_MODE_RIPENESS,
) -> dict[str, Any]:
    """Create an honest empty report for one dashboard mode."""

    mode = _normalise_mode(mode)
    categories = _categories_for_mode(mode)
    implemented = _implemented_methods(mode)
    _, _, dataset_directory, dataset_split = _dataset_definition(mode)

    return {
        "schema_version": 6,
        "evaluation_mode": mode,
        "categories": list(categories),
        "status": "No evaluation has been run.",
        "generated_at": None,
        "dataset_split": dataset_split,
        "dataset_directory": dataset_directory,
        "image_count": 0,
        "successful_images": 0,
        "failed_images": 0,
        "class_counts": {category: 0 for category in categories},
        "methods": {
            key: _empty_method_result(
                key,
                categories,
                implemented=key in implemented,
            )
            for key in METHODS
        },
        "configuration": None,
    }


def _normalise_loaded_report(
    report: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    categories = _categories_for_mode(mode)
    implemented = _implemented_methods(mode)

    report.setdefault("schema_version", 6)
    report.setdefault("evaluation_mode", mode)
    report.setdefault("categories", list(categories))
    report.setdefault("methods", {})

    for key in METHODS:
        report["methods"].setdefault(
            key,
            _empty_method_result(
                key,
                categories,
                implemented=key in implemented,
            ),
        )

        method = report["methods"][key]
        has_result = isinstance(method.get("overall_accuracy"), (int, float))

        if not has_result:
            if key in implemented:
                method["implemented"] = True
                if method.get("status") == "Not implemented":
                    method["status"] = "Not evaluated"
            else:
                method["implemented"] = False
                method["status"] = "Not implemented"

    return report


def load_latest_evaluation(
    mode: str = EVALUATION_MODE_RIPENESS,
) -> dict[str, Any]:
    mode = _normalise_mode(mode)
    result_file = LATEST_RESULT_FILES[mode]

    if (
        mode == EVALUATION_MODE_RIPENESS
        and not result_file.exists()
        and LEGACY_RIPENESS_RESULT_FILE.exists()
    ):
        result_file = LEGACY_RIPENESS_RESULT_FILE

    if not result_file.exists():
        return create_empty_evaluation(mode)

    try:
        with result_file.open("r", encoding="utf-8") as file:
            report = json.load(file)
        return _normalise_loaded_report(report, mode)
    except (OSError, json.JSONDecodeError, TypeError):
        report = create_empty_evaluation(mode)
        report["status"] = "The saved evaluation report could not be read."
        return report


def _discover_images(mode: str) -> list[dict[str, Any]]:
    directory, folder_mapping, _, _ = _dataset_definition(mode)
    if not directory.exists():
        raise EvaluationError(f"Dataset directory not found: {directory}")

    images: list[dict[str, Any]] = []
    for folder_name, category in folder_mapping.items():
        folder = directory / folder_name
        if not folder.exists():
            raise EvaluationError(
                f"Required class directory is missing: {folder}"
            )

        paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            raise EvaluationError(f"No images found for {category}.")

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


def _new_record(
    path: Path,
    actual: str,
    implemented_methods: set[str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "image_path": str(path.relative_to(PROJECT_ROOT)),
        "actual_category": actual,
        "preprocessing_time_ms": None,
        "segmentation_time_ms": None,
        "shared_status": "Failed",
        "shared_error": None,
    }

    for key in METHODS:
        is_implemented = key in implemented_methods
        record.update(
            {
                f"{key}_predicted_category": (
                    "Failed" if is_implemented else None
                ),
                f"{key}_correct": False if is_implemented else None,
                f"{key}_confidence_percent": None,
                f"{key}_time_ms": None,
                f"{key}_total_processing_time_ms": None,
                f"{key}_status": (
                    "Failed" if is_implemented else "Not implemented"
                ),
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


def _fail_implemented(
    record: dict[str, Any],
    message: str,
    shared_time_ms: float | None,
    implemented_methods: set[str],
) -> None:
    record["shared_error"] = message
    for key in implemented_methods:
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

    if key == "morphology" and hasattr(analysis, "blemish_percentage"):
        record["blemish_percentage"] = analysis.blemish_percentage
    elif key == "hsv" and hasattr(analysis, "green_percentage"):
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


def _prepare_and_segment(
    record: dict[str, Any],
    image_path: Path,
    implemented_methods: set[str],
) -> tuple[Any, Any, float] | None:
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
            _fail_implemented(
                record,
                f"Banana segmentation failed: {segmentation.message}",
                shared_time_ms,
                implemented_methods,
            )
            return None

        record["shared_status"] = "Completed"
        return prepared, segmentation, shared_time_ms

    except Exception as error:
        _fail_implemented(
            record,
            str(error),
            None,
            implemented_methods,
        )
        return None


def _evaluate_ripeness_image(
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
    implemented = _implemented_methods(EVALUATION_MODE_RIPENESS)
    record = _new_record(image_path, actual_category, implemented)
    shared = _prepare_and_segment(record, image_path, implemented)
    if shared is None:
        return record

    prepared, segmentation, shared_time_ms = shared
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
    analyses: dict[str, Any] = {}

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
            failed_branch_ms = (perf_counter() - branch_start) * 1000.0
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


def _evaluate_quality_image(
    image_path: Path,
    actual_quality: str,
    morphology_parameters: MorphologyParameters,
    quality_bands: QualityBands,
    hsv_parameters: HSVParameters,
    hsv_quality_bands: HSVQualityBands,
    kmeans_parameters: KMeansParameters,
    kmeans_quality_bands: KMeansQualityBands,
    glcm_parameters: GLCMParameters | None = None,
    glcm_quality_bands: GLCMQualityBands | None = None,
) -> dict[str, Any]:
    """Evaluate implemented quality methods on one known-ripe image."""

    implemented = _implemented_methods(EVALUATION_MODE_QUALITY)
    record = _new_record(image_path, actual_quality, implemented)
    shared = _prepare_and_segment(record, image_path, implemented)

    if shared is None:
        return record

    prepared, segmentation, shared_time_ms = shared

    glcm_parameters = glcm_parameters or GLCMParameters()
    glcm_quality_bands = glcm_quality_bands or GLCMQualityBands()

    jobs = {
        "morphology": lambda: analyse_morphology_quality(
            rgb_image=prepared.working_rgb,
            banana_mask=segmentation.final_mask,
            parameters=morphology_parameters,
            quality_bands=quality_bands,
        ),
        "hsv": lambda: analyse_hsv_quality(
            rgb_image=prepared.working_rgb,
            banana_mask=segmentation.final_mask,
            parameters=hsv_parameters,
            quality_bands=hsv_quality_bands,
        ),
        "kmeans": lambda: analyse_kmeans_quality(
            rgb_image=prepared.working_rgb,
            banana_mask=segmentation.final_mask,
            parameters=kmeans_parameters,
            quality_bands=kmeans_quality_bands,
        ),
        "glcm": lambda: analyse_glcm_quality(
            rgb_image=prepared.working_rgb,
            banana_mask=segmentation.final_mask,
            parameters=glcm_parameters,
            quality_bands=glcm_quality_bands,
        ),
    }

    for key, job in jobs.items():
        branch_start = perf_counter()
        try:
            analysis = job()
            _store_result(
                record,
                key,
                analysis,
                actual_quality,
                shared_time_ms,
            )
        except Exception as error:
            failed_branch_ms = (perf_counter() - branch_start) * 1000.0
            record[f"{key}_error"] = str(error)
            record[f"{key}_total_processing_time_ms"] = (
                shared_time_ms + failed_branch_ms
            )

    return record


def _calculate_method_metrics(
    records: list[dict[str, Any]],
    method_key: str,
    categories: tuple[str, ...],
) -> dict[str, Any]:
    predicted_field = f"{method_key}_predicted_category"
    status_field = f"{method_key}_status"
    time_field = f"{method_key}_total_processing_time_ms"
    actual = [record["actual_category"] for record in records]
    predicted = [record[predicted_field] for record in records]

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(categories),
        zero_division=0,
    )
    labels = [*categories, "Failed"]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    matrix = matrix[: len(categories), :]
    times = [
        float(record[time_field])
        for record in records
        if record[time_field] is not None
    ]
    successes = sum(
        record[status_field] == "Completed" for record in records
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
                labels=list(categories),
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
            for index, category in enumerate(categories)
        },
        "confusion_matrix": {
            "actual_labels": list(categories),
            "predicted_labels": labels,
            "values": matrix.tolist(),
        },
    }


def _write_json_atomically(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _save_report(report: dict[str, Any], mode: str) -> None:
    _write_json_atomically(LATEST_RESULT_FILES[mode], report)
    if mode == EVALUATION_MODE_RIPENESS:
        _write_json_atomically(LEGACY_RIPENESS_RESULT_FILE, report)


def run_fixed_dataset_evaluation(
    progress_callback: ProgressCallback | None = None,
    mode: str = EVALUATION_MODE_RIPENESS,
) -> dict[str, Any]:
    """Run either ripeness-test or ripe-only quality evaluation."""

    mode = _normalise_mode(mode)
    categories = _categories_for_mode(mode)
    implemented = _implemented_methods(mode)
    images = _discover_images(mode)
    morphology_parameters = MorphologyParameters()
    records: list[dict[str, Any]] = []

    if mode == EVALUATION_MODE_RIPENESS:
        morphology_bands = RipenessBands()
        hsv_parameters = HSVParameters()
        hsv_bands = HSVRipenessBands()
        kmeans_parameters = KMeansParameters()
        kmeans_bands = KMeansRipenessBands()
        glcm_parameters = GLCMParameters()
        glcm_bands = GLCMRipenessBands()

        for index, item in enumerate(images, start=1):
            records.append(
                _evaluate_ripeness_image(
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
                    f"Evaluating ripeness {index}/{len(images)}: "
                    f"{item['path'].name}",
                )

        configuration = {
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
        }
        status = (
            "Ripeness evaluation completed for Morphology, HSV, K-means, "
            "GLCM Texture and Hybrid."
        )

    else:
        quality_bands = QualityBands()
        hsv_parameters = HSVParameters()
        hsv_quality_bands = HSVQualityBands()
        kmeans_parameters = KMeansParameters()
        kmeans_quality_bands = KMeansQualityBands()
        glcm_parameters = GLCMParameters()
        glcm_quality_bands = GLCMQualityBands()

        for index, item in enumerate(images, start=1):
            records.append(
                _evaluate_quality_image(
                    image_path=item["path"],
                    actual_quality=item["actual_category"],
                    morphology_parameters=morphology_parameters,
                    quality_bands=quality_bands,
                    hsv_parameters=hsv_parameters,
                    hsv_quality_bands=hsv_quality_bands,
                    kmeans_parameters=kmeans_parameters,
                    kmeans_quality_bands=kmeans_quality_bands,
                    glcm_parameters=glcm_parameters,
                    glcm_quality_bands=glcm_quality_bands,
                )
            )

            if progress_callback is not None:
                progress_callback(
                    index,
                    len(images),
                    f"Evaluating quality {index}/{len(images)}: "
                    f"{item['path'].name}",
                )

        configuration = {
            "known_ripeness": "Ripe",
            "morphology_parameters": asdict(morphology_parameters),
            "morphology_quality_bands": asdict(quality_bands),
            "hsv_parameters": asdict(hsv_parameters),
            "hsv_quality_bands": asdict(hsv_quality_bands),
            "kmeans_parameters": asdict(kmeans_parameters),
            "kmeans_quality_bands": asdict(kmeans_quality_bands),
            "glcm_parameters": asdict(glcm_parameters),
            "glcm_quality_bands": asdict(glcm_quality_bands),
        }

        status = (
            "Quality evaluation completed for Morphology, HSV, K-means "
            "and GLCM Texture."
        )

    method_metrics = {
        key: (
            _calculate_method_metrics(records, key, categories)
            if key in implemented
            else _empty_method_result(key, categories, implemented=False)
        )
        for key in METHODS
    }
    successful_images = sum(
        all(
            record[f"{key}_status"] == "Completed"
            for key in implemented
        )
        for record in records
    )
    _, _, dataset_directory, dataset_split = _dataset_definition(mode)

    report = {
        "schema_version": 6,
        "evaluation_mode": mode,
        "categories": list(categories),
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "dataset_split": dataset_split,
        "dataset_directory": dataset_directory,
        "image_count": len(records),
        "successful_images": successful_images,
        "failed_images": len(records) - successful_images,
        "class_counts": {
            category: sum(
                record["actual_category"] == category for record in records
            )
            for category in categories
        },
        "methods": method_metrics,
        "configuration": configuration,
    }
    _save_report(report, mode)
    return report
