from pathlib import Path
from time import perf_counter

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from branches.hsv.hsv_analysis import (
    HSVQualityBands,
    analyse_hsv_quality,
)
from branches.hsv.hsv_segmentation import HSVParameters
from core.banana_segmentation import segment_banana
from core.image_handling import standardise_image


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_DIRECTORY = PROJECT_ROOT / "dataset" / "quality" / "valid"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "hsv_quality_validation"

DETAIL_FILE = OUTPUT_DIRECTORY / "hsv_quality_validation_features.csv"
SUMMARY_FILE = OUTPUT_DIRECTORY / "hsv_quality_validation_summary.csv"

FOLDER_TO_CATEGORY = {
    "class_a": "Class_A",
    "class_b": "Class_B",
    "defect": "Defect",
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
    "yellow_deterioration_ratio",
]


class HSVQualityValidationError(RuntimeError):
    """Raised when HSV quality validation cannot continue."""


def discover_validation_images() -> list[dict]:
    if not VALID_DIRECTORY.exists():
        raise HSVQualityValidationError(
            f"Quality validation directory not found: {VALID_DIRECTORY}"
        )

    images = []

    for folder_name, category in FOLDER_TO_CATEGORY.items():
        category_directory = VALID_DIRECTORY / folder_name

        if not category_directory.exists():
            raise HSVQualityValidationError(
                f"Missing quality validation directory: {category_directory}"
            )

        image_paths = sorted(
            path
            for path in category_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_paths:
            raise HSVQualityValidationError(
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

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise HSVQualityValidationError(
            f"Cannot read image: {image_path.name}"
        ) from error


def analyse_one_image(
    image_path: Path,
    actual_category: str,
    parameters: HSVParameters,
    quality_bands: HSVQualityBands,
) -> dict:
    record = {
        "image_path": str(
            image_path.relative_to(PROJECT_ROOT)
        ),
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
        "yellow_deterioration_ratio": None,

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
        image = open_rgb_image(
            image_path
        )

        preprocessing_start = perf_counter()

        prepared = standardise_image(
            image=image,
            upload_metadata={
                "filename": image_path.name
            },
            target_size=(416, 416),
        )

        record["preprocessing_time_ms"] = (
            perf_counter()
            - preprocessing_start
        ) * 1000.0

        segmentation_start = perf_counter()

        segmentation = segment_banana(
            rgb_image=prepared.working_rgb,
            content_mask=prepared.content_mask,
        )

        record["segmentation_time_ms"] = (
            perf_counter()
            - segmentation_start
        ) * 1000.0

        if not segmentation.success:
            record["error"] = (
                "Banana segmentation failed: "
                f"{segmentation.message}"
            )
            record["total_processing_time_ms"] = (
                perf_counter()
                - processing_start
            ) * 1000.0
            return record

        analysis = analyse_hsv_quality(
            rgb_image=prepared.working_rgb,
            banana_mask=segmentation.final_mask,
            parameters=parameters,
            quality_bands=quality_bands,
        )

        record.update(
            {
                "predicted_category": analysis.predicted_quality,
                "correct": (
                    analysis.predicted_quality
                    == actual_category
                ),
                "green_percentage": analysis.green_percentage,
                "yellow_percentage": analysis.yellow_percentage,
                "brown_percentage": analysis.brown_percentage,
                "dark_percentage": analysis.dark_percentage,
                "deteriorated_percentage": (
                    analysis.deteriorated_percentage
                ),
                "other_percentage": analysis.other_percentage,
                "yellow_deterioration_ratio": (
                    analysis.yellow_deterioration_ratio
                ),
                "confidence_percent": analysis.confidence_percent,
                "hsv_time_ms": analysis.processing_time_ms,
                "status": "Completed",
                "error": None,
            }
        )

    except Exception as error:
        record["error"] = (
            f"{type(error).__name__}: {error}"
        )

    record["total_processing_time_ms"] = (
        perf_counter()
        - processing_start
    ) * 1000.0

    return record


def create_summary(
    completed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for category in FOLDER_TO_CATEGORY.values():
        class_data = completed[
            completed["actual_category"]
            == category
        ]

        if class_data.empty:
            continue

        for feature in FEATURE_COLUMNS:
            values = class_data[
                feature
            ].dropna()

            if values.empty:
                continue

            rows.append(
                {
                    "actual_category": category,
                    "feature": feature,
                    "count": int(
                        values.count()
                    ),
                    "mean": values.mean(),
                    "median": values.median(),
                    "min": values.min(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "max": values.max(),
                }
            )

    return pd.DataFrame(
        rows
    )


def print_class_overview(
    completed: pd.DataFrame,
) -> None:
    print(
        "\nHSV QUALITY VALIDATION FEATURE OVERVIEW"
    )
    print(
        "=" * 78
    )

    for category in FOLDER_TO_CATEGORY.values():
        class_data = completed[
            completed["actual_category"]
            == category
        ]

        if class_data.empty:
            continue

        print(
            f"\n{category} "
            f"({len(class_data)} images)"
        )
        print(
            "-" * 78
        )

        for feature in FEATURE_COLUMNS:
            values = class_data[
                feature
            ].dropna()

            if values.empty:
                continue

            label = (
                feature
                .replace(
                    "_percentage",
                    "",
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

            suffix = (
                ""
                if feature
                == "yellow_deterioration_ratio"
                else "%"
            )

            print(
                f"{label:<30} "
                f"mean={values.mean():7.2f}{suffix}  "
                f"median={values.median():7.2f}{suffix}  "
                f"min={values.min():7.2f}{suffix}  "
                f"max={values.max():7.2f}{suffix}"
            )


def print_failure_overview(
    failed: pd.DataFrame,
) -> None:
    if failed.empty:
        return

    print(
        "\nFAILURE OVERVIEW"
    )
    print(
        "=" * 78
    )

    error_counts = (
        failed["error"]
        .fillna(
            "Unknown error"
        )
        .value_counts()
    )

    for error, count in (
        error_counts
        .head(10)
        .items()
    ):
        print(
            f"{count:>4} x {error}"
        )


def main() -> None:
    print(
        "HSV surface-quality validation analyser"
    )
    print(
        f"Validation directory: "
        f"{VALID_DIRECTORY}"
    )
    print(
        "Only dataset/quality/valid is used. "
        "dataset/quality/test is not used by this script.\n"
    )

    images = discover_validation_images()
    total_images = len(
        images
    )

    print(
        f"Found {total_images} "
        f"quality validation images.\n"
    )

    parameters = HSVParameters()
    quality_bands = HSVQualityBands()

    print(
        "Current HSV quality rules:"
    )
    print(
        f"  Defect : Brown + Dark > "
        f"{quality_bands.defect_min_deteriorated_percent:.2f}%"
    )
    print(
        f"  Class_B: otherwise Brown <= "
        f"{quality_bands.class_b_max_brown_percent:.2f}% "
        f"AND Yellow/(Brown+Dark) <= "
        f"{quality_bands.class_b_max_yellow_deterioration_ratio:.2f}"
    )
    print(
        "  Class_A: all remaining non-defect cases\n"
    )

    records = []

    for index, item in enumerate(
        images,
        start=1,
    ):
        image_path = item[
            "path"
        ]
        actual_category = item[
            "actual_category"
        ]

        print(
            f"[{index:>4}/{total_images}] "
            f"{actual_category:<8} "
            f"{image_path.name}"
        )

        records.append(
            analyse_one_image(
                image_path=image_path,
                actual_category=actual_category,
                parameters=parameters,
                quality_bands=quality_bands,
            )
        )

    results = pd.DataFrame(
        records
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        DETAIL_FILE,
        index=False,
        float_format="%.4f",
    )

    completed = results[
        results["status"]
        == "Completed"
    ].copy()

    failed = results[
        results["status"]
        != "Completed"
    ].copy()

    if completed.empty:
        print_failure_overview(
            failed
        )

        raise HSVQualityValidationError(
            "No quality validation images "
            "were successfully processed."
        )

    summary = create_summary(
        completed
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        float_format="%.4f",
    )

    accuracy = (
        completed[
            "correct"
        ].mean()
        * 100.0
    )

    print_class_overview(
        completed
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "HSV QUALITY VALIDATION RESULT"
    )
    print(
        "=" * 78
    )

    print(
        f"Completed images : "
        f"{len(completed)}"
    )
    print(
        f"Failed images    : "
        f"{len(failed)}"
    )
    print(
        f"Validation accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        "\nPredictions by actual class:"
    )

    confusion = pd.crosstab(
        completed[
            "actual_category"
        ],
        completed[
            "predicted_category"
        ],
        margins=True,
    )

    print(
        confusion.to_string()
    )

    per_class = (
        completed
        .groupby(
            "actual_category"
        )["correct"]
        .mean()
        .mul(100.0)
    )

    print(
        "\nRecall by actual class:"
    )

    for category in (
        FOLDER_TO_CATEGORY.values()
    ):
        if category in per_class.index:
            print(
                f"{category:<8}: "
                f"{per_class[category]:6.2f}%"
            )

    print_failure_overview(
        failed
    )

    print(
        "\nSaved detailed measurements to:"
    )
    print(
        DETAIL_FILE
    )

    print(
        "\nSaved per-class statistics to:"
    )
    print(
        SUMMARY_FILE
    )

    print(
        "\nDo not use dataset/quality/test "
        "to tune these thresholds."
    )


if __name__ == "__main__":
    main()
