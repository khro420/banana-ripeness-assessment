from pathlib import Path
from collections import defaultdict
import sys

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.image_handling import standardise_image
from core.banana_segmentation import segment_banana

from branches.glcm.glcm_analysis import (
    extract_glcm_features,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_DIRECTORY = (
    PROJECT_ROOT
    / "dataset"
    / "ripeness"
    / "valid"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "glcm_validation_features.csv"
)

CATEGORIES = (
    "unripe",
    "ripe",
    "overripe",
    "rotten",
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# IMAGE LOADING
# ============================================================

def open_rgb_image(image_path: Path) -> Image.Image:
    """
    Open an image and convert it to RGB.
    """

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

        raise ValueError(
            f"Cannot read image: {image_path}"
        ) from error


# ============================================================
# FIND VALIDATION IMAGES
# ============================================================

def find_images():
    """
    Find images inside:

        dataset/ripeness/valid/

    The folder name is used as the ground-truth
    ripeness category.
    """

    image_records = []

    for category in CATEGORIES:

        category_directory = (
            VALID_DIRECTORY / category
        )

        if not category_directory.exists():

            print(
                f"WARNING: Missing folder: "
                f"{category_directory}"
            )

            continue

        for image_path in sorted(
            category_directory.rglob("*")
        ):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                image_records.append(
                    (
                        image_path,
                        category,
                    )
                )

    return image_records


# ============================================================
# CALCULATE FEATURES
# ============================================================

def analyse_one_image(
    image_path: Path,
):
    """
    Run the same preprocessing, banana segmentation
    and GLCM feature extraction used by the project.
    """

    image = open_rgb_image(
        image_path
    )

    # --------------------------------------------------------
    # Image standardisation
    # --------------------------------------------------------

    prepared = standardise_image(
        image=image,
        upload_metadata={
            "filename": image_path.name,
        },
        target_size=(416, 416),
    )

    # --------------------------------------------------------
    # Shared banana segmentation
    # --------------------------------------------------------

    segmentation = segment_banana(
        rgb_image=prepared.working_rgb,
        content_mask=prepared.content_mask,
    )

    if not segmentation.success:

        raise ValueError(
            "Banana segmentation failed: "
            f"{segmentation.message}"
        )

    banana_mask = (
        segmentation.final_mask
    )

    # --------------------------------------------------------
    # GLCM feature extraction
    # --------------------------------------------------------

    features, _ = extract_glcm_features(
        rgb_image=prepared.working_rgb,
        banana_mask=banana_mask,
    )

    features = np.asarray(
        features,
        dtype=np.float64,
    )

    if features.size < 4:

        raise ValueError(
            "GLCM feature extraction did not "
            "return four features."
        )

    contrast = float(
        features[0]
    )

    homogeneity = float(
        features[1]
    )

    energy = float(
        features[2]
    )

    correlation = float(
        features[3]
    )

    return (
        contrast,
        homogeneity,
        energy,
        correlation,
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        file.write(
            "category,image,contrast,"
            "homogeneity,energy,correlation\n"
        )

        for record in records:

            file.write(
                f"{record['category']},"
                f"\"{record['image']}\","
                f"{record['contrast']:.8f},"
                f"{record['homogeneity']:.8f},"
                f"{record['energy']:.8f},"
                f"{record['correlation']:.8f}\n"
            )


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(
    grouped_records,
):

    print()
    print("=" * 75)
    print("GLCM VALIDATION DATASET STATISTICS")
    print("=" * 75)

    feature_names = (
        "contrast",
        "homogeneity",
        "energy",
        "correlation",
    )

    for category in CATEGORIES:

        values = grouped_records.get(
            category,
            [],
        )

        print()
        print(
            f"========== "
            f"{category.upper()} "
            f"({len(values)} images) =========="
        )

        if not values:

            print("No successful images.")

            continue

        for feature_name in feature_names:

            feature_values = np.array(
                [
                    record[feature_name]
                    for record in values
                ],
                dtype=np.float64,
            )

            print()
            print(
                f"{feature_name.capitalize()}:"
            )

            print(
                f"  Min    : "
                f"{np.min(feature_values):.6f}"
            )

            print(
                f"  10%    : "
                f"{np.percentile(feature_values, 10):.6f}"
            )

            print(
                f"  25%    : "
                f"{np.percentile(feature_values, 25):.6f}"
            )

            print(
                f"  Mean   : "
                f"{np.mean(feature_values):.6f}"
            )

            print(
                f"  Median : "
                f"{np.median(feature_values):.6f}"
            )

            print(
                f"  75%    : "
                f"{np.percentile(feature_values, 75):.6f}"
            )

            print(
                f"  90%    : "
                f"{np.percentile(feature_values, 90):.6f}"
            )

            print(
                f"  Max    : "
                f"{np.max(feature_values):.6f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "Starting GLCM validation calibration..."
    )

    print(
        f"Validation directory:"
        f"\n{VALID_DIRECTORY}"
    )

    if not VALID_DIRECTORY.exists():

        raise FileNotFoundError(
            "Validation directory was not found:\n"
            f"{VALID_DIRECTORY}"
        )

    image_records = find_images()

    print()
    print(
        f"Found {len(image_records)} "
        f"validation images."
    )

    if not image_records:

        raise RuntimeError(
            "No validation images were found."
        )

    results = []

    grouped_records = defaultdict(list)

    failed_count = 0

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, (
        image_path,
        category,
    ) in enumerate(
        image_records,
        start=1,
    ):

        print(
            f"[{index}/{len(image_records)}] "
            f"{category}: "
            f"{image_path.name}"
        )

        try:

            (
                contrast,
                homogeneity,
                energy,
                correlation,
            ) = analyse_one_image(
                image_path
            )

            record = {
                "category": category,
                "image": str(
                    image_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "contrast": contrast,
                "homogeneity": homogeneity,
                "energy": energy,
                "correlation": correlation,
            }

            results.append(
                record
            )

            grouped_records[
                category
            ].append(
                record
            )

        except Exception as error:

            failed_count += 1

            print(
                "  FAILED:"
                f" {error}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_csv(
        results
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print_statistics(
        grouped_records
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("CALIBRATION FINISHED")
    print("=" * 75)

    print(
        f"Successful images : "
        f"{len(results)}"
    )

    print(
        f"Failed images     : "
        f"{failed_count}"
    )

    print()
    print(
        "Feature data saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These results are from the validation "
        "set only."
    )

    print(
        "Do NOT use dataset/ripeness/test/ "
        "to determine thresholds."
    )


if __name__ == "__main__":

    main()