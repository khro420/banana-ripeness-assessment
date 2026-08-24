from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class KMeansParameters:
    """Settings for K-means clustering."""

    k: int = 4
    attempts: int = 10
    max_iterations: int = 100
    epsilon: float = 0.2


@dataclass
class KMeansSegmentationResult:
    """Result produced by K-means clustering."""

    segmented_image_rgb: np.ndarray
    cluster_map: np.ndarray

    centres_rgb: np.ndarray
    centres_lab: np.ndarray

    cluster_sizes: np.ndarray
    cluster_percentages: np.ndarray

    banana_area_pixels: int
    compactness: float


def _binary_mask(
    mask: np.ndarray,
) -> np.ndarray:
    """Convert mask into 0 and 255."""

    return np.where(
        mask > 0,
        255,
        0,
    ).astype(np.uint8)


def segment_kmeans_colours(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: KMeansParameters | None = None,
) -> KMeansSegmentationResult:
    """
    Perform K-means clustering on banana pixels.

    LAB is used because:
    L = brightness
    A = green/red information
    B = blue/yellow information
    """

    parameters = (
        parameters
        or KMeansParameters()
    )

    rgb_image = np.asarray(
        rgb_image,
        dtype=np.uint8,
    )

    banana_mask = _binary_mask(
        np.asarray(banana_mask)
    )


    # Check image
    if (
        rgb_image.ndim != 3
        or rgb_image.shape[2] != 3
    ):
        raise ValueError(
            "rgb_image must have shape "
            "(height, width, 3)."
        )


    if (
        banana_mask.shape
        != rgb_image.shape[:2]
    ):
        raise ValueError(
            "Banana mask and image dimensions "
            "do not match."
        )


    # -------------------------------------------------
    # Convert RGB to LAB
    # -------------------------------------------------

    lab_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2LAB,
    )


    # Only take banana pixels
    banana_pixels_lab = (
        lab_image[banana_mask > 0]
    )

    banana_area_pixels = len(
        banana_pixels_lab
    )


    if banana_area_pixels < 100:
        raise ValueError(
            "Banana region is too small "
            "for K-means analysis."
        )


    # -------------------------------------------------
    # Prepare K-means
    # -------------------------------------------------

    pixel_data = np.float32(
        banana_pixels_lab
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        parameters.max_iterations,
        parameters.epsilon,
    )


    # Same result when evaluation is repeated
    cv2.setRNGSeed(42)


    # -------------------------------------------------
    # Run K-means
    # -------------------------------------------------

    compactness, labels, centres_lab = cv2.kmeans(
        pixel_data,
        parameters.k,
        None,
        criteria,
        parameters.attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.flatten()


    centres_lab = np.uint8(
        np.clip(
            centres_lab,
            0,
            255,
        )
    )


    # Convert centres back to RGB for display
    centres_rgb = cv2.cvtColor(
        centres_lab.reshape(
            1,
            -1,
            3,
        ),
        cv2.COLOR_LAB2RGB,
    ).reshape(
        -1,
        3,
    )


    # -------------------------------------------------
    # Calculate cluster sizes
    # -------------------------------------------------

    cluster_sizes = np.array(
        [
            np.count_nonzero(
                labels == cluster_id
            )
            for cluster_id
            in range(parameters.k)
        ],
        dtype=np.int32,
    )


    cluster_percentages = (
        cluster_sizes
        / banana_area_pixels
        * 100.0
    )


    # -------------------------------------------------
    # Full cluster map
    # -------------------------------------------------

    cluster_map = np.full(
        banana_mask.shape,
        -1,
        dtype=np.int32,
    )

    cluster_map[
        banana_mask > 0
    ] = labels


    # -------------------------------------------------
    # Clustered image
    # -------------------------------------------------

    segmented_image_rgb = (
        rgb_image.copy()
    )


    segmented_image_rgb[
        banana_mask > 0
    ] = centres_rgb[labels]


    # Darken background
    background = (
        banana_mask == 0
    )

    segmented_image_rgb[
        background
    ] = (
        segmented_image_rgb[
            background
        ].astype(np.float32)
        * 0.25
    ).astype(np.uint8)


    return KMeansSegmentationResult(
        segmented_image_rgb=(
            segmented_image_rgb
        ),
        cluster_map=cluster_map,

        centres_rgb=centres_rgb,
        centres_lab=centres_lab,

        cluster_sizes=cluster_sizes,
        cluster_percentages=(
            cluster_percentages
        ),

        banana_area_pixels=(
            banana_area_pixels
        ),

        compactness=float(
            compactness
        ),
    )