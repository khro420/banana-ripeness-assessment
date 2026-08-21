from pathlib import Path
from time import perf_counter

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from branches.morphology.morphology_analysis import (
    QualityBands,
    RipenessBands,
    analyse_morphology,
)
from branches.morphology.morphology_segmentation import MorphologyParameters
from core.banana_segmentation import segment_banana
from core.evaluation import (
    PROJECT_ROOT,
    discover_quality_split_images,
    ensure_quality_split_manifest,
)
from core.image_handling import standardise_image


OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "morphology_quality_validation"
DETAIL_FILE = OUTPUT_DIRECTORY / "morphology_quality_validation_features.csv"
FEATURE_SUMMARY_FILE = OUTPUT_DIRECTORY / "morphology_quality_feature_summary.csv"
CATEGORY_SUMMARY_FILE = OUTPUT_DIRECTORY / "morphology_quality_category_summary.csv"

CHECKPOINT_INTERVAL = 25
FEATURE_COLUMNS = (
    "total_dark_percentage",
    "largest_dark_patch_percentage",
    "concentration_ratio_percentage",
    "dark_region_spread_percentage",
    "dark_component_count",
    "extreme_dark_percentage",
    "largest_patch_mean_intensity",
)
QUALITY_ORDER = ("Class_A", "Class_B", "Defect")


class QualityValidationError(RuntimeError):
    """Raised when quality validation feature extraction cannot continue."""


def _open_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise QualityValidationError(f"Cannot read image: {path.name}") from error


def _empty_record(path: Path, actual_quality: str) -> dict:
    return {
        "image_path": str(path.relative_to(PROJECT_ROOT)),
        "filename": path.name,
        "actual_quality": actual_quality,
        "predicted_quality": "Failed",
        "correct": False,
        "total_dark_percentage": None,
        "largest_dark_patch_percentage": None,
        "concentration_ratio_percentage": None,
        "dark_region_spread_percentage": None,
        "dark_component_count": None,
        "extreme_dark_percentage": None,
        "largest_patch_mean_intensity": None,
        "quality_confidence_percent": None,
        "banana_area_percent": None,
        "preprocessing_time_ms": None,
        "segmentation_time_ms": None,
        "morphology_time_ms": None,
        "total_processing_time_ms": None,
        "status": "Failed",
        "error": None,
    }


def _analyse_image(
    path: Path,
    actual_quality: str,
    morphology_parameters: MorphologyParameters,
    ripeness_bands: RipenessBands,
    quality_bands: QualityBands,
) -> dict:
    record = _empty_record(path, actual_quality)
    total_start = perf_counter()

    try:
        image = _open_rgb_image(path)

        start = perf_counter()
        prepared = standardise_image(
            image=image,
            upload_metadata={"filename": path.name},
            target_size=(416, 416),
        )
        record["preprocessing_time_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        segmentation = segment_banana(
            prepared.working_rgb,
            prepared.content_mask,
        )
        record["segmentation_time_ms"] = (perf_counter() - start) * 1000.0

        if not segmentation.success:
            record["error"] = f"Foreground segmentation failed: {segmentation.message}"
            return record

        analysis = analyse_morphology(
            rgb_image=prepared.working_rgb,
            banana_mask=segmentation.final_mask,
            parameters=morphology_parameters,
            bands=ripeness_bands,
            quality_bands=quality_bands,
            assume_ripe_for_quality=True,
        )

        record.update(
            {
                "predicted_quality": analysis.quality_category,
                "correct": analysis.quality_category == actual_quality,
                "total_dark_percentage": analysis.total_dark_percentage,
                "largest_dark_patch_percentage": (
                    analysis.largest_dark_patch_percentage
                ),
                "concentration_ratio_percentage": analysis.concentration_ratio,
                "dark_region_spread_percentage": analysis.dark_region_spread,
                "dark_component_count": analysis.dark_component_count,
                "extreme_dark_percentage": analysis.extreme_dark_percentage,
                "largest_patch_mean_intensity": (
                    analysis.largest_patch_mean_intensity
                ),
                "quality_confidence_percent": (
                    analysis.quality_confidence_percent
                ),
                "banana_area_percent": segmentation.banana_area_percent,
                "morphology_time_ms": analysis.processing_time_ms,
                "status": "Completed",
                "error": None,
            }
        )
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["total_processing_time_ms"] = (
            perf_counter() - total_start
        ) * 1000.0

    return record


def _save_detail(records: list[dict]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(DETAIL_FILE, index=False, float_format="%.4f")


def _feature_summary(completed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in QUALITY_ORDER:
        class_data = completed[completed["actual_quality"] == category]
        for feature in FEATURE_COLUMNS:
            values = pd.to_numeric(class_data[feature], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "actual_quality": category,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "minimum": values.min(),
                    "q25": values.quantile(0.25),
                    "median": values.median(),
                    "q75": values.quantile(0.75),
                    "maximum": values.max(),
                }
            )
    return pd.DataFrame(rows)


def _category_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in QUALITY_ORDER:
        class_data = results[results["actual_quality"] == category]
        completed = class_data[class_data["status"] == "Completed"]
        rows.append(
            {
                "actual_quality": category,
                "total_images": len(class_data),
                "completed_images": len(completed),
                "failed_images": len(class_data) - len(completed),
                "correct_predictions": int(completed["correct"].sum()),
                "recall_percent": (
                    completed["correct"].mean() * 100.0
                    if not completed.empty
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest = ensure_quality_split_manifest()
    images = discover_quality_split_images("validation")
    morphology_parameters = MorphologyParameters()
    ripeness_bands = RipenessBands()
    quality_bands = QualityBands()
    records: list[dict] = []

    print("Morphology ripe-banana quality validation exporter", flush=True)
    print(f"Fixed split manifest: {manifest}", flush=True)
    print(f"Validation images: {len(images)}", flush=True)
    print("The manifest test rows are not read by this script.\n", flush=True)

    try:
        for index, item in enumerate(images, start=1):
            print(
                f"[{index:>4}/{len(images)}] "
                f"{item['actual_label']:<8} {item['path'].name}",
                flush=True,
            )
            records.append(
                _analyse_image(
                    item["path"],
                    item["actual_label"],
                    morphology_parameters,
                    ripeness_bands,
                    quality_bands,
                )
            )
            if index == 1 or index % CHECKPOINT_INTERVAL == 0:
                _save_detail(records)
    except KeyboardInterrupt:
        _save_detail(records)
        print(f"\nStopped. Partial results saved to {DETAIL_FILE}")
        return

    _save_detail(records)
    results = pd.DataFrame(records)
    completed = results[results["status"] == "Completed"].copy()
    failed = results[results["status"] != "Completed"].copy()
    if completed.empty:
        raise QualityValidationError(
            f"No image completed. Inspect the error column in {DETAIL_FILE}."
        )

    _feature_summary(completed).to_csv(
        FEATURE_SUMMARY_FILE,
        index=False,
        float_format="%.4f",
    )
    _category_summary(results).to_csv(
        CATEGORY_SUMMARY_FILE,
        index=False,
        float_format="%.4f",
    )

    print("\nValidation predictions by actual quality:")
    print(
        pd.crosstab(
            completed["actual_quality"],
            completed["predicted_quality"],
            margins=True,
        ).to_string()
    )
    print(f"\nCompleted: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Current validation accuracy: {completed['correct'].mean():.2%}")
    print(f"\nSaved: {DETAIL_FILE}")
    print(f"Saved: {FEATURE_SUMMARY_FILE}")
    print(f"Saved: {CATEGORY_SUMMARY_FILE}")
    print(
        "\nChoose and freeze QualityBands from these validation results only. "
        "Then run the dashboard once for the held-out quality test rows."
    )


if __name__ == "__main__":
    main()
