import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from branches.morphology.morphology_analysis import (
    RipenessBands,
    analyse_morphology,
)
from branches.morphology.morphology_segmentation import MorphologyParameters
from core.banana_segmentation import segment_banana
from core.image_handling import standardise_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
VALID_DIRECTORY = PROJECT_ROOT / "dataset" / "ripeness" / "valid"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "development" / "morphology_validation"

DETAIL_FILE = OUTPUT_DIRECTORY / "morphology_validation_features.csv"
FEATURE_SUMMARY_FILE = (
    OUTPUT_DIRECTORY / "morphology_validation_feature_summary.csv"
)
CATEGORY_SUMMARY_FILE = (
    OUTPUT_DIRECTORY / "morphology_validation_category_summary.csv"
)

CHECKPOINT_INTERVAL = 25
EXTREME_DARK_THRESHOLD = 50

FOLDER_TO_CATEGORY = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "overripe": "Overripe",
    "rotten": "Rotten",
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

FEATURE_COLUMNS = [
    "total_dark_percentage",
    "largest_dark_patch_percentage",
    "concentration_ratio_percentage",
    "dark_region_spread_percentage",
    "dark_component_count",
    "extreme_dark_percentage",
    "largest_patch_mean_intensity",
]


class ValidationAnalysisError(RuntimeError):
    """Raised when morphology validation cannot continue."""


def discover_validation_images() -> list[dict]:
    if not VALID_DIRECTORY.exists():
        raise ValidationAnalysisError(
            f"Validation directory not found: {VALID_DIRECTORY}"
        )

    images = []

    for folder_name, category in FOLDER_TO_CATEGORY.items():
        category_directory = VALID_DIRECTORY / folder_name

        if not category_directory.exists():
            raise ValidationAnalysisError(
                f"Missing validation class directory: {category_directory}"
            )

        image_paths = sorted(
            path
            for path in category_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_paths:
            raise ValidationAnalysisError(
                f"No validation images found for {category}."
            )

        for image_path in image_paths:
            images.append(
                {
                    "path": image_path,
                    "actual_category": category,
                }
            )

    return images


def open_rgb_image(image_path: Path) -> Image.Image:
    try:
        with Image.open(image_path) as opened_image:
            image = ImageOps.exif_transpose(opened_image)
            image = image.convert("RGB")
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValidationAnalysisError(
            f"Cannot read image: {image_path.name}"
        ) from error


def extract_component_features(
    dark_mask: np.ndarray,
    greyscale_image: np.ndarray,
    banana_mask: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure the size, concentration and intensity of dark regions."""

    binary_dark = np.where(dark_mask > 0, 255, 0).astype(np.uint8)
    binary_banana = banana_mask > 0
    binary_dark[~binary_banana] = 0

    banana_area = int(np.count_nonzero(binary_banana))
    dark_area = int(np.count_nonzero(binary_dark))

    if banana_area == 0:
        raise ValidationAnalysisError("The supplied banana mask is empty.")

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_dark,
        connectivity=8,
    )

    dark_component_count = max(0, component_count - 1)
    largest_label = None
    largest_area = 0

    if dark_component_count > 0:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_offset = int(np.argmax(component_areas))
        largest_label = largest_offset + 1
        largest_area = int(component_areas[largest_offset])

    largest_patch_percentage = 100.0 * largest_area / banana_area
    concentration_ratio = (
        100.0 * largest_area / dark_area if dark_area else 0.0
    )

    extreme_dark_area = int(
        np.count_nonzero(
            (greyscale_image <= EXTREME_DARK_THRESHOLD)
            & binary_banana
        )
    )
    extreme_dark_percentage = 100.0 * extreme_dark_area / banana_area

    largest_patch_mean_intensity = None
    if largest_label is not None:
        largest_patch_pixels = greyscale_image[labels == largest_label]
        if largest_patch_pixels.size:
            largest_patch_mean_intensity = float(
                np.mean(largest_patch_pixels)
            )

    return {
        "largest_dark_patch_percentage": largest_patch_percentage,
        "concentration_ratio_percentage": concentration_ratio,
        "dark_component_count": dark_component_count,
        "extreme_dark_percentage": extreme_dark_percentage,
        "largest_patch_mean_intensity": largest_patch_mean_intensity,
    }


def empty_record(image_path: Path, actual_category: str) -> dict:
    return {
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "filename": image_path.name,
        "actual_category": actual_category,
        "predicted_category": "Failed",
        "correct": False,
        "total_dark_percentage": None,
        "largest_dark_patch_percentage": None,
        "concentration_ratio_percentage": None,
        "dark_region_spread_percentage": None,
        "dark_component_count": None,
        "extreme_dark_percentage": None,
        "largest_patch_mean_intensity": None,
        "confidence_percent": None,
        "surface_grade": None,
        "banana_area_pixels": None,
        "preprocessing_time_ms": None,
        "segmentation_time_ms": None,
        "morphology_time_ms": None,
        "total_processing_time_ms": None,
        "status": "Failed",
        "error": None,
    }


def analyse_one_image(
    image_path: Path,
    actual_category: str,
    parameters: MorphologyParameters,
    bands: RipenessBands,
) -> dict:
    record = empty_record(image_path, actual_category)
    processing_start = perf_counter()

    try:
        image = open_rgb_image(image_path)

        preprocessing_start = perf_counter()
        prepared = standardise_image(
            image=image,
            upload_metadata={"filename": image_path.name},
            target_size=(416, 416),
        )
        record["preprocessing_time_ms"] = (
            perf_counter() - preprocessing_start
        ) * 1000.0

        segmentation_start = perf_counter()
        segmentation = segment_banana(
            rgb_image=prepared.working_rgb,
            content_mask=prepared.content_mask,
        )
        record["segmentation_time_ms"] = (
            perf_counter() - segmentation_start
        ) * 1000.0

        if not segmentation.success:
            record["error"] = (
                "Banana segmentation failed: "
                f"{segmentation.message}"
            )
            record["total_processing_time_ms"] = (
                perf_counter() - processing_start
            ) * 1000.0
            return record

        analysis = analyse_morphology(
            rgb_image=prepared.working_rgb,
            banana_mask=segmentation.final_mask,
            parameters=parameters,
            bands=bands,
        )

        extra_features = extract_component_features(
            dark_mask=analysis.masks.blemish_mask,
            greyscale_image=analysis.masks.greyscale_image,
            banana_mask=segmentation.final_mask,
        )

        record.update(
            {
                "predicted_category": analysis.predicted_category,
                "correct": analysis.predicted_category == actual_category,
                "total_dark_percentage": analysis.total_dark_percentage,
                "dark_region_spread_percentage": analysis.dark_region_spread,
                "confidence_percent": analysis.confidence_percent,
                "surface_grade": analysis.surface_grade,
                "banana_area_pixels": analysis.masks.banana_area_pixels,
                "morphology_time_ms": analysis.processing_time_ms,
                "status": "Completed",
                "error": None,
                **extra_features,
            }
        )

    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"

    record["total_processing_time_ms"] = (
        perf_counter() - processing_start
    ) * 1000.0

    return record


def save_detail_checkpoint(records: list[dict]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        DETAIL_FILE,
        index=False,
        float_format="%.4f",
    )


def create_feature_summary(completed: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category in FOLDER_TO_CATEGORY.values():
        class_data = completed[
            completed["actual_category"] == category
        ]

        for feature in FEATURE_COLUMNS:
            values = pd.to_numeric(
                class_data[feature],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            rows.append(
                {
                    "actual_category": category,
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


def create_category_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category in FOLDER_TO_CATEGORY.values():
        class_data = results[results["actual_category"] == category]
        completed = class_data[class_data["status"] == "Completed"]

        rows.append(
            {
                "actual_category": category,
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


def print_feature_overview(completed: pd.DataFrame) -> None:
    print("\nMORPHOLOGY VALIDATION FEATURE OVERVIEW")
    print("=" * 88)

    for category in FOLDER_TO_CATEGORY.values():
        class_data = completed[
            completed["actual_category"] == category
        ]
        if class_data.empty:
            continue

        print(f"\n{category} ({len(class_data)} completed images)")
        print("-" * 88)

        for feature in FEATURE_COLUMNS:
            values = pd.to_numeric(
                class_data[feature],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue

            label = feature.replace("_", " ").title()
            print(
                f"{label:<38} "
                f"mean={values.mean():7.2f}  "
                f"median={values.median():7.2f}  "
                f"min={values.min():7.2f}  "
                f"max={values.max():7.2f}"
            )


def print_failure_overview(failed: pd.DataFrame) -> None:
    if failed.empty:
        return

    print("\nFAILURE OVERVIEW")
    print("=" * 88)

    error_counts = failed["error"].fillna("Unknown error").value_counts()
    for error, count in error_counts.head(10).items():
        print(f"{count:>4} x {error}")


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Morphology validation feature exporter", flush=True)
    print(f"Validation directory: {VALID_DIRECTORY}", flush=True)
    print(f"Output directory: {OUTPUT_DIRECTORY}", flush=True)
    print("The test split is not used by this script.\n", flush=True)

    images = discover_validation_images()
    total_images = len(images)
    print(f"Found {total_images} validation images.\n", flush=True)

    parameters = MorphologyParameters()
    bands = RipenessBands()
    records = []

    try:
        for index, item in enumerate(images, start=1):
            image_path = item["path"]
            actual_category = item["actual_category"]

            print(
                f"[{index:>4}/{total_images}] "
                f"{actual_category:<9} {image_path.name}",
                flush=True,
            )

            records.append(
                analyse_one_image(
                    image_path=image_path,
                    actual_category=actual_category,
                    parameters=parameters,
                    bands=bands,
                )
            )

            if index == 1 or index % CHECKPOINT_INTERVAL == 0:
                save_detail_checkpoint(records)
                print(
                    f"Checkpoint saved: {DETAIL_FILE}",
                    flush=True,
                )

    except KeyboardInterrupt:
        save_detail_checkpoint(records)
        print(
            f"\nStopped by user. Partial results saved to:\n{DETAIL_FILE}",
            flush=True,
        )
        return

    save_detail_checkpoint(records)
    results = pd.DataFrame(records)

    completed = results[results["status"] == "Completed"].copy()
    failed = results[results["status"] != "Completed"].copy()

    if completed.empty:
        print_failure_overview(failed)
        raise ValidationAnalysisError(
            "No validation images were successfully processed. "
            f"Inspect the error column in {DETAIL_FILE}."
        )

    feature_summary = create_feature_summary(completed)
    feature_summary.to_csv(
        FEATURE_SUMMARY_FILE,
        index=False,
        float_format="%.4f",
    )

    category_summary = create_category_summary(results)
    category_summary.to_csv(
        CATEGORY_SUMMARY_FILE,
        index=False,
        float_format="%.4f",
    )

    accuracy = completed["correct"].mean() * 100.0
    print_feature_overview(completed)

    print("\n" + "=" * 88)
    print("CURRENT-RULE VALIDATION RESULT")
    print("=" * 88)
    print(f"Completed images    : {len(completed)}")
    print(f"Failed images       : {len(failed)}")
    print(f"Validation accuracy : {accuracy:.2f}%")

    print("\nPredictions by actual class:")
    confusion = pd.crosstab(
        completed["actual_category"],
        completed["predicted_category"],
        margins=True,
    )
    print(confusion.to_string())

    print_failure_overview(failed)

    print("\nSaved per-image measurements to:")
    print(DETAIL_FILE)
    print("\nSaved per-class feature statistics to:")
    print(FEATURE_SUMMARY_FILE)
    print("\nSaved category results to:")
    print(CATEGORY_SUMMARY_FILE)
    print(
        "\nUse these validation distributions to choose morphology rules. "
        "Do not tune them using dataset/test."
    )


if __name__ == "__main__":
    main()
