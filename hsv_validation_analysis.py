from pathlib import Path
from time import perf_counter

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from branches.hsv.hsv_analysis import HSVRipenessBands, analyse_hsv
from branches.hsv.hsv_segmentation import HSVParameters
from core.banana_segmentation import segment_banana
from core.image_handling import standardise_image


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_DIRECTORY = PROJECT_ROOT / "dataset" / "valid"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "hsv_validation"

DETAIL_FILE = OUTPUT_DIRECTORY / "hsv_validation_features.csv"
SUMMARY_FILE = OUTPUT_DIRECTORY / "hsv_validation_summary.csv"

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
    "green_percentage",
    "yellow_percentage",
    "brown_percentage",
    "dark_percentage",
    "deteriorated_percentage",
    "other_percentage",
]


class ValidationAnalysisError(RuntimeError):
    """Raised when the HSV validation analysis cannot continue."""


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


def analyse_one_image(
    image_path: Path,
    actual_category: str,
    parameters: HSVParameters,
    bands: HSVRipenessBands,
) -> dict:
    record = {
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "filename": image_path.name,
        "actual_category": actual_category,
        "predicted_category": "Failed",
        "correct": False,
        "green_percentage": None,
        "yellow_percentage": None,
        "brown_percentage": None,
        "dark_percentage": None,
        "deteriorated_percentage": None,
        "other_percentage": None,
        "confidence_percent": None,
        "preprocessing_time_ms": None,
        "segmentation_time_ms": None,
        "hsv_time_ms": None,
        "total_processing_time_ms": None,
        "status": "Failed",
        "error": None,
    }

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

        analysis = analyse_hsv(
            rgb_image=prepared.working_rgb,
            banana_mask=segmentation.final_mask,
            parameters=parameters,
            bands=bands,
        )

        record.update(
            {
                "predicted_category": analysis.predicted_category,
                "correct": analysis.predicted_category == actual_category,
                "green_percentage": analysis.green_percentage,
                "yellow_percentage": analysis.yellow_percentage,
                "brown_percentage": analysis.brown_percentage,
                "dark_percentage": analysis.dark_percentage,
                "deteriorated_percentage": analysis.deteriorated_percentage,
                "other_percentage": analysis.other_percentage,
                "confidence_percent": analysis.confidence_percent,
                "hsv_time_ms": analysis.processing_time_ms,
                "status": "Completed",
                "error": None,
            }
        )

    except Exception as error:
        # Keep the exact exception type so a future integration error is easy
        # to diagnose from the CSV/terminal instead of ending with a vague
        # "0 images processed" message.
        record["error"] = f"{type(error).__name__}: {error}"

    record["total_processing_time_ms"] = (
        perf_counter() - processing_start
    ) * 1000.0

    return record


def create_summary(completed: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category in FOLDER_TO_CATEGORY.values():
        class_data = completed[
            completed["actual_category"] == category
        ]

        if class_data.empty:
            continue

        for feature in FEATURE_COLUMNS:
            values = class_data[feature].dropna()
            if values.empty:
                continue

            rows.append(
                {
                    "actual_category": category,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "min": values.min(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "max": values.max(),
                }
            )

    return pd.DataFrame(rows)


def print_class_overview(completed: pd.DataFrame) -> None:
    print("\nHSV VALIDATION FEATURE OVERVIEW")
    print("=" * 78)

    for category in FOLDER_TO_CATEGORY.values():
        class_data = completed[
            completed["actual_category"] == category
        ]
        if class_data.empty:
            continue

        print(f"\n{category} ({len(class_data)} images)")
        print("-" * 78)

        for feature in FEATURE_COLUMNS:
            values = class_data[feature].dropna()
            if values.empty:
                continue

            label = feature.replace("_percentage", "").replace("_", " ").title()
            print(
                f"{label:<15} "
                f"mean={values.mean():6.2f}%  "
                f"median={values.median():6.2f}%  "
                f"min={values.min():6.2f}%  "
                f"max={values.max():6.2f}%"
            )


def print_failure_overview(failed: pd.DataFrame) -> None:
    if failed.empty:
        return

    print("\nFAILURE OVERVIEW")
    print("=" * 78)

    error_counts = failed["error"].fillna("Unknown error").value_counts()
    for error, count in error_counts.head(10).items():
        print(f"{count:>4} x {error}")


def main() -> None:
    print("HSV validation analyser")
    print(f"Validation directory: {VALID_DIRECTORY}")
    print("The test split is not used by this script.\n")

    images = discover_validation_images()
    total_images = len(images)
    print(f"Found {total_images} validation images.\n")

    parameters = HSVParameters()
    bands = HSVRipenessBands()
    records = []

    for index, item in enumerate(images, start=1):
        image_path = item["path"]
        actual_category = item["actual_category"]

        print(
            f"[{index:>4}/{total_images}] "
            f"{actual_category:<9} {image_path.name}"
        )

        records.append(
            analyse_one_image(
                image_path=image_path,
                actual_category=actual_category,
                parameters=parameters,
                bands=bands,
            )
        )

    results = pd.DataFrame(records)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results.to_csv(DETAIL_FILE, index=False, float_format="%.4f")

    completed = results[results["status"] == "Completed"].copy()
    failed = results[results["status"] != "Completed"].copy()

    if completed.empty:
        print_failure_overview(failed)
        print(f"\nDetailed failure records were saved to:\n{DETAIL_FILE}")
        raise ValidationAnalysisError(
            "No validation images were successfully processed. "
            "See the failure overview above for the actual cause."
        )

    summary = create_summary(completed)
    summary.to_csv(SUMMARY_FILE, index=False, float_format="%.4f")

    accuracy = completed["correct"].mean() * 100.0

    print_class_overview(completed)

    print("\n" + "=" * 78)
    print("FROZEN-RULE VALIDATION RESULT")
    print("=" * 78)
    print(f"Completed images : {len(completed)}")
    print(f"Failed images    : {len(failed)}")
    print(f"Validation accuracy: {accuracy:.2f}%")

    print("\nPredictions by actual class:")
    confusion = pd.crosstab(
        completed["actual_category"],
        completed["predicted_category"],
        margins=True,
    )
    print(confusion.to_string())

    per_class = (
        completed.groupby("actual_category")["correct"]
        .mean()
        .mul(100.0)
    )
    print("\nRecall by actual class:")
    for category in FOLDER_TO_CATEGORY.values():
        if category in per_class.index:
            print(f"{category:<10}: {per_class[category]:6.2f}%")

    print_failure_overview(failed)

    print("\nSaved detailed measurements to:")
    print(DETAIL_FILE)
    print("\nSaved per-class statistics to:")
    print(SUMMARY_FILE)
    print(
        "\nThese rules are now treated as frozen. Do not use dataset/test "
        "to tune them."
    )


if __name__ == "__main__":
    main()
