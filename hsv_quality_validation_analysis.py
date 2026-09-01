from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from branches.hsv.hsv_analysis import (
    HSVQualityBands,
    analyse_hsv_quality,
)
from branches.hsv.hsv_segmentation import HSVParameters
from core.banana_segmentation import segment_banana
from core.image_handling import standardise_image


PROJECT_ROOT = Path(__file__).resolve().parent

VALID_DIRECTORY = (
    PROJECT_ROOT
    / "dataset"
    / "quality"
    / "valid"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "hsv_quality_validation"
)

DETAIL_FILE = (
    OUTPUT_DIRECTORY
    / "hsv_quality_validation_features.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIRECTORY
    / "hsv_quality_validation_summary.csv"
)

THRESHOLD_FILE = (
    OUTPUT_DIRECTORY
    / "hsv_quality_threshold_search_top10.csv"
)

FINE_THRESHOLD_FILE = (
    OUTPUT_DIRECTORY
    / "hsv_quality_threshold_fine_search_top10.csv"
)


FOLDER_TO_CATEGORY = {
    "class_a": "Class_A",
    "class_b": "Class_B",
    "defect": "Defect",
}

QUALITY_CATEGORIES = [
    "Class_A",
    "Class_B",
    "Defect",
]

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
    pass


def discover_validation_images() -> list[dict]:

    if not VALID_DIRECTORY.exists():

        raise HSVQualityValidationError(
            f"Quality validation directory not found: "
            f"{VALID_DIRECTORY}"
        )

    images = []

    for folder_name, category in (
        FOLDER_TO_CATEGORY.items()
    ):

        category_directory = (
            VALID_DIRECTORY
            / folder_name
        )

        if not category_directory.exists():

            raise HSVQualityValidationError(
                f"Missing quality validation directory: "
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

            raise HSVQualityValidationError(
                f"No validation images found for "
                f"{category}."
            )

        for image_path in image_paths:

            images.append(
                {
                    "path": image_path,
                    "actual_category": category,
                }
            )

    return images


def open_rgb_image(
    image_path: Path,
) -> Image.Image:

    try:

        with Image.open(
            image_path
        ) as opened_image:

            image = ImageOps.exif_transpose(
                opened_image
            )

            image = image.convert(
                "RGB"
            )

            image.load()

            return image.copy()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:

        raise HSVQualityValidationError(
            f"Cannot read image: "
            f"{image_path.name}"
        ) from error


def analyse_one_image(
    image_path: Path,
    actual_category: str,
    parameters: HSVParameters,
    quality_bands: HSVQualityBands,
) -> dict:

    record = {

        "image_path":
            str(
                image_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "filename":
            image_path.name,

        "actual_category":
            actual_category,

        "predicted_category":
            "Failed",

        "correct":
            False,

        "green_percentage":
            None,

        "yellow_percentage":
            None,

        "brown_percentage":
            None,

        "dark_percentage":
            None,

        "deteriorated_percentage":
            None,

        "other_percentage":
            None,

        "yellow_deterioration_ratio":
            None,

        "confidence_percent":
            None,

        "preprocessing_time_ms":
            None,

        "segmentation_time_ms":
            None,

        "hsv_time_ms":
            None,

        "total_processing_time_ms":
            None,

        "status":
            "Failed",

        "error":
            None,
    }

    processing_start = perf_counter()

    try:

        image = open_rgb_image(
            image_path
        )

        preprocessing_start = (
            perf_counter()
        )

        prepared = standardise_image(
            image=image,

            upload_metadata={
                "filename":
                    image_path.name
            },

            target_size=(
                416,
                416,
            ),
        )

        record[
            "preprocessing_time_ms"
        ] = (
            perf_counter()
            - preprocessing_start
        ) * 1000.0


        segmentation_start = (
            perf_counter()
        )

        segmentation = segment_banana(

            rgb_image=
                prepared.working_rgb,

            content_mask=
                prepared.content_mask,
        )

        record[
            "segmentation_time_ms"
        ] = (
            perf_counter()
            - segmentation_start
        ) * 1000.0


        if not segmentation.success:

            record["error"] = (
                "Banana segmentation failed: "
                f"{segmentation.message}"
            )

            record[
                "total_processing_time_ms"
            ] = (
                perf_counter()
                - processing_start
            ) * 1000.0

            return record


        analysis = analyse_hsv_quality(

            rgb_image=
                prepared.working_rgb,

            banana_mask=
                segmentation.final_mask,

            parameters=
                parameters,

            quality_bands=
                quality_bands,
        )


        record.update(
            {
                "predicted_category":
                    analysis.predicted_quality,

                "correct":
                    (
                        analysis.predicted_quality
                        == actual_category
                    ),

                "green_percentage":
                    analysis.green_percentage,

                "yellow_percentage":
                    analysis.yellow_percentage,

                "brown_percentage":
                    analysis.brown_percentage,

                "dark_percentage":
                    analysis.dark_percentage,

                "deteriorated_percentage":
                    analysis.deteriorated_percentage,

                "other_percentage":
                    analysis.other_percentage,

                "yellow_deterioration_ratio":
                    analysis.yellow_deterioration_ratio,

                "confidence_percent":
                    analysis.confidence_percent,

                "hsv_time_ms":
                    analysis.processing_time_ms,

                "status":
                    "Completed",

                "error":
                    None,
            }
        )

    except Exception as error:

        record["error"] = (
            f"{type(error).__name__}: "
            f"{error}"
        )


    record[
        "total_processing_time_ms"
    ] = (
        perf_counter()
        - processing_start
    ) * 1000.0

    return record


def create_summary(
    completed: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for category in (
        FOLDER_TO_CATEGORY.values()
    ):

        class_data = completed[
            completed[
                "actual_category"
            ]
            == category
        ]

        if class_data.empty:
            continue

        for feature in FEATURE_COLUMNS:

            values = (
                class_data[
                    feature
                ]
                .dropna()
            )

            if values.empty:
                continue

            rows.append(
                {
                    "actual_category":
                        category,

                    "feature":
                        feature,

                    "count":
                        int(
                            values.count()
                        ),

                    "mean":
                        values.mean(),

                    "median":
                        values.median(),

                    "min":
                        values.min(),

                    "q25":
                        values.quantile(
                            0.25
                        ),

                    "q75":
                        values.quantile(
                            0.75
                        ),

                    "max":
                        values.max(),
                }
            )

    return pd.DataFrame(
        rows
    )


def print_class_overview(
    completed: pd.DataFrame,
) -> None:

    print(
        "\nHSV QUALITY VALIDATION "
        "FEATURE OVERVIEW"
    )

    print(
        "=" * 78
    )

    for category in (
        FOLDER_TO_CATEGORY.values()
    ):

        class_data = completed[
            completed[
                "actual_category"
            ]
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

            values = (
                class_data[
                    feature
                ]
                .dropna()
            )

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
                f"mean="
                f"{values.mean():7.2f}"
                f"{suffix}  "
                f"median="
                f"{values.median():7.2f}"
                f"{suffix}  "
                f"min="
                f"{values.min():7.2f}"
                f"{suffix}  "
                f"max="
                f"{values.max():7.2f}"
                f"{suffix}"
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
        failed[
            "error"
        ]
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
            f"{count:>4} x "
            f"{error}"
        )


# ================================================================
# VALIDATION-ONLY THRESHOLD SEARCH
# ================================================================

def predict_with_thresholds(
    completed: pd.DataFrame,
    defect_threshold: float,
    class_b_brown_threshold: float,
    class_b_ratio_threshold: float,
) -> np.ndarray:

    deteriorated = completed[
        "deteriorated_percentage"
    ].to_numpy(
        dtype=float
    )

    brown = completed[
        "brown_percentage"
    ].to_numpy(
        dtype=float
    )

    ratio = completed[
        "yellow_deterioration_ratio"
    ].to_numpy(
        dtype=float
    )

    # Start with Class_A.
    predictions = np.full(
        len(completed),
        "Class_A",
        dtype=object,
    )

    # Rule 1: Defect
    defect = (
        deteriorated
        > defect_threshold
    )

    predictions[
        defect
    ] = "Defect"

    # Rule 2: Class_B
    class_b = (
        ~defect
        & (
            brown
            <= class_b_brown_threshold
        )
        & (
            ratio
            <= class_b_ratio_threshold
        )
    )

    predictions[
        class_b
    ] = "Class_B"

    return predictions


def candidate_values(
    values: pd.Series,
    count: int = 40,
) -> np.ndarray:

    values = (
        values
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .astype(float)
    )

    if values.empty:

        raise HSVQualityValidationError(
            "Cannot generate threshold "
            "candidates from an empty "
            "feature column."
        )

    # Broad search uses quantiles.
    quantiles = np.linspace(
        0.02,
        0.98,
        count,
    )

    candidates = np.quantile(
        values.to_numpy(),
        quantiles,
    )

    return np.unique(
        np.round(
            candidates,
            4,
        )
    )


def search_grid(
    completed: pd.DataFrame,
    defect_candidates: np.ndarray,
    brown_candidates: np.ndarray,
    ratio_candidates: np.ndarray,
) -> dict:

    actual = completed[
        "actual_category"
    ].to_numpy()

    best = None
    all_results = []

    for defect_threshold in (
        defect_candidates
    ):

        for brown_threshold in (
            brown_candidates
        ):

            for ratio_threshold in (
                ratio_candidates
            ):

                predicted = (
                    predict_with_thresholds(
                        completed,
                        defect_threshold,
                        brown_threshold,
                        ratio_threshold,
                    )
                )

                macro_f1 = f1_score(
                    actual,
                    predicted,
                    labels=
                        QUALITY_CATEGORIES,
                    average=
                        "macro",
                    zero_division=
                        0,
                )

                accuracy = (
                    accuracy_score(
                        actual,
                        predicted,
                    )
                )

                result = {

                    "defect_min_deteriorated_percent":
                        float(
                            defect_threshold
                        ),

                    "class_b_max_brown_percent":
                        float(
                            brown_threshold
                        ),

                    "class_b_max_yellow_deterioration_ratio":
                        float(
                            ratio_threshold
                        ),

                    "macro_f1":
                        float(
                            macro_f1
                        ),

                    "accuracy":
                        float(
                            accuracy
                        ),
                }

                all_results.append(
                    result
                )

                # Macro F1 first.
                # Accuracy breaks ties.
                score = (
                    macro_f1,
                    accuracy,
                )

                if (
                    best is None
                    or score
                    > best["score"]
                ):

                    best = {

                        "score":
                            score,

                        "result":
                            result,

                        "predicted":
                            predicted.copy(),
                    }

    top_results = sorted(

        all_results,

        key=lambda row: (
            row[
                "macro_f1"
            ],
            row[
                "accuracy"
            ],
        ),

        reverse=True,

    )[:10]

    return {

        "best":
            best,

        "top":
            top_results,
    }


def search_quality_thresholds(
    completed: pd.DataFrame,
) -> dict:

    print(
        "\nSearching validation-only "
        "HSV quality thresholds..."
    )

    defect_candidates = (
        candidate_values(
            completed[
                "deteriorated_percentage"
            ],
            count=40,
        )
    )

    brown_candidates = (
        candidate_values(
            completed[
                "brown_percentage"
            ],
            count=40,
        )
    )

    ratio_candidates = (
        candidate_values(
            completed[
                "yellow_deterioration_ratio"
            ],
            count=40,
        )
    )

    combination_count = (
        len(defect_candidates)
        * len(brown_candidates)
        * len(ratio_candidates)
    )

    print(
        f"Testing "
        f"{combination_count} "
        f"candidate combinations."
    )

    return search_grid(
        completed,
        defect_candidates,
        brown_candidates,
        ratio_candidates,
    )


def fine_search_quality_thresholds(
    completed: pd.DataFrame,
    broad_result: dict,
) -> dict:

    print(
        "\nStarting second-stage "
        "fine search..."
    )

    defect_center = (
        broad_result[
            "defect_min_deteriorated_percent"
        ]
    )

    brown_center = (
        broad_result[
            "class_b_max_brown_percent"
        ]
    )

    ratio_center = (
        broad_result[
            "class_b_max_yellow_deterioration_ratio"
        ]
    )

    # ±1.0 around the best Defect threshold.
    # 41 values = step of 0.05.
    fine_defect_candidates = (
        np.linspace(
            defect_center - 1.0,
            defect_center + 1.0,
            41,
        )
    )

    # ±1.0 around the best Brown threshold.
    # 41 values = step of 0.05.
    fine_brown_candidates = (
        np.linspace(
            brown_center - 1.0,
            brown_center + 1.0,
            41,
        )
    )

    # ±0.5 around the best Ratio threshold.
    # 41 values = step of 0.025.
    fine_ratio_candidates = (
        np.linspace(
            ratio_center - 0.5,
            ratio_center + 0.5,
            41,
        )
    )

    fine_defect_candidates = (
        np.unique(
            np.round(
                fine_defect_candidates,
                4,
            )
        )
    )

    fine_brown_candidates = (
        np.unique(
            np.round(
                fine_brown_candidates,
                4,
            )
        )
    )

    fine_ratio_candidates = (
        np.unique(
            np.round(
                fine_ratio_candidates,
                4,
            )
        )
    )

    combination_count = (
        len(fine_defect_candidates)
        * len(fine_brown_candidates)
        * len(fine_ratio_candidates)
    )

    print(
        f"Testing "
        f"{combination_count} "
        f"fine combinations."
    )

    return search_grid(
        completed,
        fine_defect_candidates,
        fine_brown_candidates,
        fine_ratio_candidates,
    )


def print_confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
) -> None:

    matrix = confusion_matrix(

        actual,

        predicted,

        labels=
            QUALITY_CATEGORIES,
    )

    matrix = pd.DataFrame(

        matrix,

        index=[
            "Actual Class_A",
            "Actual Class_B",
            "Actual Defect",
        ],

        columns=[
            "Pred Class_A",
            "Pred Class_B",
            "Pred Defect",
        ],
    )

    print(
        f"\n{title}"
    )

    print(
        matrix.to_string()
    )


def print_threshold_search(
    completed: pd.DataFrame,
    quality_bands: HSVQualityBands,
) -> None:

    actual = completed[
        "actual_category"
    ].to_numpy()

    # ------------------------------------------------------------
    # CURRENT THRESHOLDS
    # ------------------------------------------------------------

    current_predictions = (
        predict_with_thresholds(

            completed,

            quality_bands
            .defect_min_deteriorated_percent,

            quality_bands
            .class_b_max_brown_percent,

            quality_bands
            .class_b_max_yellow_deterioration_ratio,
        )
    )

    current_accuracy = (
        accuracy_score(
            actual,
            current_predictions,
        )
    )

    current_macro_f1 = (
        f1_score(

            actual,

            current_predictions,

            labels=
                QUALITY_CATEGORIES,

            average=
                "macro",

            zero_division=
                0,
        )
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "HSV QUALITY THRESHOLD SEARCH"
    )

    print(
        "=" * 78
    )

    print(
        "\nCURRENT THRESHOLDS"
    )

    print(
        "Defect deterioration : "
        f"{quality_bands.defect_min_deteriorated_percent:.4f}"
    )

    print(
        "Class_B brown        : "
        f"{quality_bands.class_b_max_brown_percent:.4f}"
    )

    print(
        "Class_B ratio        : "
        f"{quality_bands.class_b_max_yellow_deterioration_ratio:.4f}"
    )

    print(
        "Validation accuracy  : "
        f"{current_accuracy * 100:.2f}%"
    )

    print(
        "Validation Macro F1  : "
        f"{current_macro_f1 * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # STAGE 1: BROAD QUANTILE SEARCH
    # ------------------------------------------------------------

    broad_search = (
        search_quality_thresholds(
            completed
        )
    )

    broad_best = (
        broad_search[
            "best"
        ]
    )

    broad_result = (
        broad_best[
            "result"
        ]
    )

    broad_predicted = (
        broad_best[
            "predicted"
        ]
    )

    print(
        "\nBEST VALIDATION THRESHOLDS"
    )

    print(
        "defect_min_deteriorated_percent = "
        f"{broad_result['defect_min_deteriorated_percent']:.4f}"
    )

    print(
        "class_b_max_brown_percent = "
        f"{broad_result['class_b_max_brown_percent']:.4f}"
    )

    print(
        "class_b_max_yellow_deterioration_ratio = "
        f"{broad_result['class_b_max_yellow_deterioration_ratio']:.4f}"
    )

    print(
        "Validation accuracy : "
        f"{broad_result['accuracy'] * 100:.2f}%"
    )

    print(
        "Validation Macro F1 : "
        f"{broad_result['macro_f1'] * 100:.2f}%"
    )

    print_confusion_matrix(
        actual,
        broad_predicted,
        "BEST VALIDATION CONFUSION MATRIX",
    )

    pd.DataFrame(
        broad_search[
            "top"
        ]
    ).to_csv(

        THRESHOLD_FILE,

        index=False,

        float_format=
            "%.4f",
    )

    # ------------------------------------------------------------
    # STAGE 2: FINE SEARCH AROUND BROAD WINNER
    # ------------------------------------------------------------

    fine_search = (
        fine_search_quality_thresholds(
            completed,
            broad_result,
        )
    )

    fine_best = (
        fine_search[
            "best"
        ]
    )

    fine_result = (
        fine_best[
            "result"
        ]
    )

    fine_predicted = (
        fine_best[
            "predicted"
        ]
    )

    print(
        "\nBEST FINE-SEARCH "
        "VALIDATION THRESHOLDS"
    )

    print(
        "defect_min_deteriorated_percent = "
        f"{fine_result['defect_min_deteriorated_percent']:.4f}"
    )

    print(
        "class_b_max_brown_percent = "
        f"{fine_result['class_b_max_brown_percent']:.4f}"
    )

    print(
        "class_b_max_yellow_deterioration_ratio = "
        f"{fine_result['class_b_max_yellow_deterioration_ratio']:.4f}"
    )

    print(
        "Validation accuracy : "
        f"{fine_result['accuracy'] * 100:.2f}%"
    )

    print(
        "Validation Macro F1 : "
        f"{fine_result['macro_f1'] * 100:.2f}%"
    )

    print_confusion_matrix(
        actual,
        fine_predicted,
        "BEST FINE-SEARCH "
        "VALIDATION CONFUSION MATRIX",
    )

    pd.DataFrame(
        fine_search[
            "top"
        ]
    ).to_csv(

        FINE_THRESHOLD_FILE,

        index=False,

        float_format=
            "%.4f",
    )

    print(
        "\nBroad-search top 10 "
        "combinations saved to:"
    )

    print(
        THRESHOLD_FILE
    )

    print(
        "\nFine-search top 10 "
        "combinations saved to:"
    )

    print(
        FINE_THRESHOLD_FILE
    )

    print(
        "\nIMPORTANT: Both search stages "
        "were selected using validation "
        "data only. Use the BEST FINE-SEARCH "
        "thresholds, copy them into "
        "HSVQualityBands, freeze them, "
        "and only then run "
        "dataset/quality/test."
    )


def main() -> None:

    print(
        "HSV surface-quality "
        "validation analyser"
    )

    print(
        f"Validation directory: "
        f"{VALID_DIRECTORY}"
    )

    print(
        "Only dataset/quality/valid "
        "is used. dataset/quality/test "
        "is not used by this script.\n"
    )

    images = (
        discover_validation_images()
    )

    total_images = len(
        images
    )

    print(
        f"Found {total_images} "
        f"quality validation images.\n"
    )

    parameters = (
        HSVParameters()
    )

    quality_bands = (
        HSVQualityBands()
    )

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
        "  Class_A: all remaining "
        "non-defect cases\n"
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

                image_path=
                    image_path,

                actual_category=
                    actual_category,

                parameters=
                    parameters,

                quality_bands=
                    quality_bands,
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

        float_format=
            "%.4f",
    )

    completed = results[
        results[
            "status"
        ]
        == "Completed"
    ].copy()

    failed = results[
        results[
            "status"
        ]
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

        float_format=
            "%.4f",
    )

    current_accuracy = (

        completed[
            "correct"
        ].mean()

        * 100.0
    )

    # Feature distributions
    print_class_overview(
        completed
    )

    # Broad + fine validation-only search
    print_threshold_search(
        completed,
        quality_bands,
    )

    # Current classifier result
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
        f"{current_accuracy:.2f}%"
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

        .mul(
            100.0
        )
    )

    print(
        "\nRecall by actual class:"
    )

    for category in (
        FOLDER_TO_CATEGORY.values()
    ):

        if category in (
            per_class.index
        ):

            print(
                f"{category:<8}: "
                f"{per_class[category]:6.2f}%"
            )

    print_failure_overview(
        failed
    )

    print(
        "\nSaved detailed "
        "measurements to:"
    )

    print(
        DETAIL_FILE
    )

    print(
        "\nSaved per-class "
        "statistics to:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nSaved broad-search top "
        "threshold combinations to:"
    )

    print(
        THRESHOLD_FILE
    )

    print(
        "\nSaved fine-search top "
        "threshold combinations to:"
    )

    print(
        FINE_THRESHOLD_FILE
    )

    print(
        "\nDo not use "
        "dataset/quality/test "
        "to tune these thresholds."
    )


if __name__ == "__main__":
    main()