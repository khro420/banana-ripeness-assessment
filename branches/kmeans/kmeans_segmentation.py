from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class KMeansParameters:
    """Parameters used for K-means colour clustering."""

    k: int = 4
    attempts: int = 10
    max_iterations: int = 100
    epsilon: float = 0.2


@dataclass
class KMeansSegmentationResult:
    """Intermediate results from K-means colour clustering."""

    segmented_image_rgb: np.ndarray
    cluster_map: np.ndarray
    centres_rgb: np.ndarray
    cluster_sizes: np.ndarray
    cluster_percentages: np.ndarray
    banana_area_pixels: int
    compactness: float


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def segment_kmeans_colours(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: KMeansParameters | None = None,
) -> KMeansSegmentationResult:
    """
    Apply K-means clustering only to pixels inside the banana mask.
    """

    parameters = parameters or KMeansParameters()

    rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    banana_mask = _binary_mask(np.asarray(banana_mask))

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(
            "rgb_image must have shape (height, width, 3)."
        )

    if banana_mask.shape != rgb_image.shape[:2]:
        raise ValueError(
            "The banana mask and image dimensions do not match."
        )

    banana_pixels = rgb_image[banana_mask > 0]

    banana_area_pixels = len(banana_pixels)

    if banana_area_pixels < 100:
        raise ValueError(
            "The banana region is too small for K-means analysis."
        )

    if parameters.k < 2:
        raise ValueError("K must be at least 2.")

    if parameters.k > banana_area_pixels:
        raise ValueError(
            "K cannot be larger than the number of banana pixels."
        )

    # OpenCV K-means requires float32 input
    pixel_data = np.float32(banana_pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        parameters.max_iterations,
        parameters.epsilon,
    )

    compactness, labels, centres = cv2.kmeans(
        pixel_data,
        parameters.k,
        None,
        criteria,
        parameters.attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.flatten()

    # Convert cluster centres back to normal RGB values
    centres_rgb = np.uint8(
        np.clip(centres, 0, 255)
    )

    cluster_sizes = np.array(
        [
            np.count_nonzero(labels == cluster_id)
            for cluster_id in range(parameters.k)
        ],
        dtype=np.int32,
    )

    cluster_percentages = (
        cluster_sizes
        / banana_area_pixels
        * 100.0
    )

    # Create full image-sized cluster map
    cluster_map = np.full(
        banana_mask.shape,
        -1,
        dtype=np.int32,
    )

    cluster_map[banana_mask > 0] = labels

    # Create visual clustered banana image
    segmented_image_rgb = rgb_image.copy()

    clustered_pixels = centres_rgb[labels]

    segmented_image_rgb[banana_mask > 0] = clustered_pixels

    # Darken background
    background = banana_mask == 0

    segmented_image_rgb[background] = (
        segmented_image_rgb[background]
        .astype(np.float32)
        * 0.25
    ).astype(np.uint8)

    return KMeansSegmentationResult(
        segmented_image_rgb=segmented_image_rgb,
        cluster_map=cluster_map,
        centres_rgb=centres_rgb,
        cluster_sizes=cluster_sizes,
        cluster_percentages=cluster_percentages,
        banana_area_pixels=banana_area_pixels,
        compactness=float(compactness),
    )