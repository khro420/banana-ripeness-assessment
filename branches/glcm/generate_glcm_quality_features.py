from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core.banana_segmentation import segment_banana

from branches.glcm.glcm_analysis import (
    GLCMParameters,
    build_glcm,
    calculate_glcm_features,
    quantise_image,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUALITY_VALIDATION_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "quality"
    / "valid"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "glcm_quality_validation_features.csv"
)


# ============================================================
# DISSIMILARITY
# ============================================================

def calculate_dissimilarity(
    glcm: np.ndarray,
) -> float:
    """
    Calculate GLCM dissimilarity.

    Dissimilarity measures the average absolute
    difference between paired gray-level values.
    """

    levels = glcm.shape[0]

    i = np.arange(
        levels,
        dtype=np.float64,
    )

    j = np.arange(
        levels,
        dtype=np.float64,
    )

    difference = np.abs(
        i[:, None] - j[None, :]
    )

    return float(
        np.sum(
            difference * glcm
        )
    )


# ============================================================
# QUALITY GLCM FEATURES
# ============================================================

def calculate_quality_glcm_features(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: GLCMParameters | None = None,
):
    """
    Extract the GLCM features used for quality classification.

    Features:
        - Contrast
        - Dissimilarity
        - Homogeneity
        - Energy
        - Correlation

    The GLCM construction uses the existing implementation
    from branches.glcm.glcm_analysis.
    """

    parameters = (
        parameters
        or GLCMParameters()
    )

    # --------------------------------------------------------
    # Convert inputs to NumPy arrays
    # --------------------------------------------------------

    rgb_image = np.asarray(
        rgb_image,
        dtype=np.uint8,
    )

    banana_mask = np.asarray(
        banana_mask
    )

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if rgb_image.ndim != 3:
        raise ValueError(
            "rgb_image must have shape "
            "(height, width, 3)."
        )

    if rgb_image.shape[2] != 3:
        raise ValueError(
            "rgb_image must contain 3 channels."
        )

    # --------------------------------------------------------
    # Validate mask
    # --------------------------------------------------------

    if banana_mask.shape != rgb_image.shape[:2]:
        raise ValueError(
            "Banana mask and image dimensions "
            "do not match."
        )

    # --------------------------------------------------------
    # RGB → grayscale
    # --------------------------------------------------------

    grayscale = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY,
    )

    # --------------------------------------------------------
    # Quantise grayscale
    # --------------------------------------------------------

    quantised = quantise_image(
        grayscale,
        banana_mask,
        parameters.levels,
    )

    # --------------------------------------------------------
    # Calculate GLCM features
    # for every distance / angle combination
    # --------------------------------------------------------

    feature_values = []

    dissimilarity_values = []

    for distance in parameters.distances:

        for angle in parameters.angles:

            glcm = build_glcm(
                quantised=quantised,
                mask=banana_mask,
                levels=parameters.levels,
                distance=distance,
                angle=angle,
            )

            # Existing four GLCM features
            features = calculate_glcm_features(
                glcm
            )

            feature_values.append(
                features
            )

            # Dissimilarity
            dissimilarity_values.append(
                calculate_dissimilarity(
                    glcm
                )
            )

    # --------------------------------------------------------
    # Convert feature results to arrays
    # --------------------------------------------------------

    feature_values = np.array(
        feature_values,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Average the four existing GLCM features
    # --------------------------------------------------------

    final_features = np.mean(
        feature_values,
        axis=0,
    )

    # --------------------------------------------------------
    # Average dissimilarity
    # --------------------------------------------------------

    dissimilarity = float(
        np.mean(
            dissimilarity_values
        )
    )

    # --------------------------------------------------------
    # Return:
    #
    # calculate_glcm_features() returns:
    #   Contrast
    #   Homogeneity
    #   Energy
    #   Correlation
    #
    # We add Dissimilarity as the second feature.
    # --------------------------------------------------------

    return {
        "Contrast": float(
            final_features[0]
        ),
        "Dissimilarity": dissimilarity,
        "Homogeneity": float(
            final_features[1]
        ),
        "Energy": float(
            final_features[2]
        ),
        "Correlation": float(
            final_features[3]
        ),
    }


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path: Path,
    category: str,
):
    """
    Process one quality-validation image.

    The category passed here is already normalized to:
        Class_A
        Class_B
        Defect
    """

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        raise ValueError(
            f"Unable to read image: "
            f"{image_path}"
        )

    # --------------------------------------------------------
    # OpenCV BGR → RGB
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------
    # Segment banana
    # --------------------------------------------------------

    segmentation_result = segment_banana(
        image_rgb
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # segment_banana() returns a SegmentationResult
    # object, NOT a NumPy mask directly.
    #
    # Therefore we use:
    #     segmentation_result.final_mask
    # --------------------------------------------------------

    if segmentation_result is None:
        raise ValueError(
            "Segmentation returned None."
        )

    if not segmentation_result.success:
        raise ValueError(
            "Banana segmentation failed: "
            f"{segmentation_result.message}"
        )

    banana_mask = (
        segmentation_result.final_mask
    )

    # --------------------------------------------------------
    # Validate mask
    # --------------------------------------------------------

    if banana_mask is None:
        raise ValueError(
            "Segmentation returned no final mask."
        )

    banana_mask = np.asarray(
        banana_mask
    )

    if banana_mask.shape != image_rgb.shape[:2]:
        raise ValueError(
            "Segmentation mask shape does not "
            "match image dimensions."
        )

    # --------------------------------------------------------
    # Extract GLCM features
    # --------------------------------------------------------

    features = calculate_quality_glcm_features(
        rgb_image=image_rgb,
        banana_mask=banana_mask,
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    features["category"] = category

    features["filename"] = (
        image_path.name
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "GLCM QUALITY VALIDATION FEATURE EXTRACTION"
    )

    print("=" * 70)

    print()

    print("Dataset:")

    print(
        QUALITY_VALIDATION_DIR
    )

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not QUALITY_VALIDATION_DIR.exists():

        raise FileNotFoundError(
            "Quality validation directory "
            "was not found:\n"
            f"{QUALITY_VALIDATION_DIR}"
        )

    # --------------------------------------------------------
    # NEW DATASET FOLDER NAMES
    #
    # Physical folder name → CSV category name
    # --------------------------------------------------------

    category_map = {
        "class_a": "Class_A",
        "class_b": "Class_B",
        "defect": "Defect",
    }

    results = []

    total_processed = 0

    total_failed = 0

    # --------------------------------------------------------
    # PROCESS EACH CATEGORY
    # --------------------------------------------------------

    for folder_name, category in category_map.items():

        category_dir = (
            QUALITY_VALIDATION_DIR
            / folder_name
        )

        if not category_dir.exists():

            print()

            print(
                f"WARNING: {category_dir} "
                "does not exist."
            )

            continue

        # ----------------------------------------------------
        # Find image files
        # ----------------------------------------------------

        image_files = []

        for extension in (
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.bmp",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
            "*.BMP",
        ):

            image_files.extend(
                category_dir.glob(
                    extension
                )
            )

        image_files = sorted(
            set(image_files)
        )

        print()

        print(
            f"{category}: "
            f"{len(image_files)} images"
        )

        processed_category = 0

        failed_category = 0

        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        for image_path in image_files:

            try:

                result = process_image(
                    image_path,
                    category,
                )

                if result is not None:

                    results.append(
                        result
                    )

                    processed_category += 1

                    total_processed += 1

                    # Progress every 50 images
                    if (
                        processed_category % 50
                        == 0
                    ):

                        print(
                            f"  Processed "
                            f"{processed_category}/"
                            f"{len(image_files)}"
                        )

            except Exception as error:

                failed_category += 1

                total_failed += 1

                print()

                print(
                    f"Error processing "
                    f"{image_path.name}:"
                )

                print(error)

        # ----------------------------------------------------
        # Category summary
        # ----------------------------------------------------

        print()

        print(
            f"  {category} complete: "
            f"{processed_category}/"
            f"{len(image_files)} successful"
        )

        if failed_category > 0:

            print(
                f"  Failed: "
                f"{failed_category}"
            )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not results:

        raise RuntimeError(
            "No images were successfully "
            "processed."
        )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    dataframe = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Ensure column order
    # --------------------------------------------------------

    column_order = [
        "filename",
        "category",
        "Contrast",
        "Dissimilarity",
        "Homogeneity",
        "Energy",
        "Correlation",
    ]

    dataframe = dataframe[
        column_order
    ]

    # ========================================================
    # SAVE CSV
    # ========================================================

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "EXTRACTION COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Validation images processed: "
        f"{total_processed}"
    )

    print(
        f"Validation images failed: "
        f"{total_failed}"
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        dataframe[
            "category"
        ].value_counts()
    )

    print()

    print(
        "Features:"
    )

    print(
        "  - Contrast"
    )

    print(
        "  - Dissimilarity"
    )

    print(
        "  - Homogeneity"
    )

    print(
        "  - Energy"
    )

    print(
        "  - Correlation"
    )

    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    # ========================================================
    # SAFETY CHECK
    # ========================================================
    #
    # We expect:
    #   Class_A = 354
    #   Class_B = 203
    #   Defect  = 282
    #   TOTAL   = 839
    #
    # If this does not happen, STOP before calibration.
    # ========================================================

    expected_counts = {
        "Class_A": 354,
        "Class_B": 203,
        "Defect": 282,
    }

    actual_counts = (
        dataframe[
            "category"
        ]
        .value_counts()
        .to_dict()
    )

    print("=" * 70)

    print(
        "DATASET COUNT CHECK"
    )

    print("=" * 70)

    count_check_passed = True

    for category, expected in (
        expected_counts.items()
    ):

        actual = actual_counts.get(
            category,
            0,
        )

        status = (
            "OK"
            if actual == expected
            else "MISMATCH"
        )

        print(
            f"{category:<10} "
            f"Expected: {expected:<4} "
            f"Actual: {actual:<4} "
            f"{status}"
        )

        if actual != expected:
            count_check_passed = False

    print()

    if (
        total_processed == 839
        and count_check_passed
    ):

        print(
            "✓ ALL 839 VALIDATION IMAGES "
            "WERE PROCESSED SUCCESSFULLY."
        )

        print()

        print(
            "Safe to proceed to GLCM "
            "rule calibration."
        )

    else:

        print(
            "WARNING: The expected 839 images "
            "were NOT all processed."
        )

        print(
            "DO NOT calibrate the rules yet."
        )

        print(
            "Check the errors above first."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()