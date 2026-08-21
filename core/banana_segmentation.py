from dataclasses import dataclass

import cv2
import numpy as np


MINIMUM_AREA_RATIO = 0.02
MAXIMUM_AREA_RATIO = 0.95

DARK_PADDING_THRESHOLD = 45
MAXIMUM_PADDING_RATIO = 0.48


@dataclass
class SegmentationResult:
    """Shared foreground-segmentation output."""

    success: bool
    message: str
    final_mask: np.ndarray
    overlay_rgb: np.ndarray


def _normalise_content_mask(
    content_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if content_mask is None:
        return np.full(shape, 255, dtype=np.uint8)

    content_mask = np.asarray(content_mask)

    if content_mask.shape != shape:
        raise ValueError(
            "content_mask must match the image dimensions."
        )

    return np.where(
        content_mask > 0,
        255,
        0,
    ).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground object."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if count <= 1:
        return np.zeros_like(mask)

    largest_label = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )

    return np.where(
        labels == largest_label,
        255,
        0,
    ).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes inside the foreground object."""

    padded = cv2.copyMakeBorder(
        mask,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    flooded = padded.copy()

    cv2.floodFill(
        flooded,
        None,
        (0, 0),
        255,
    )

    holes = cv2.bitwise_not(
        flooded
    )[1:-1, 1:-1]

    return cv2.bitwise_or(mask, holes)

def _remove_dark_border_padding(
    rgb_image: np.ndarray,
    content_mask: np.ndarray,
) -> np.ndarray:
    """
    Remove near-black regions connected to the image boundary.

    This handles black triangular corners caused by rotated images
    without removing dark blemishes inside the banana.
    """

    maximum_channel = np.max(
        rgb_image,
        axis=2,
    )

    dark_mask = np.where(
        (
            maximum_channel
            <= DARK_PADDING_THRESHOLD
        )
        & (content_mask > 0),
        255,
        0,
    ).astype(np.uint8)

    count, labels, _, _ = (
        cv2.connectedComponentsWithStats(
            dark_mask,
            connectivity=8,
        )
    )

    if count <= 1:
        return content_mask.copy()

    edge_labels = np.unique(
        np.concatenate(
            (
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            )
        )
    )

    # Label zero represents normal background.
    edge_labels = edge_labels[
        edge_labels > 0
    ]

    if edge_labels.size == 0:
        return content_mask.copy()

    padding_mask = (
        np.isin(
            labels,
            edge_labels,
        ).astype(np.uint8)
        * 255
    )

    # Include anti-aliased pixels beside the black corners.
    fringe_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    padding_mask = cv2.dilate(
        padding_mask,
        fringe_kernel,
        iterations=1,
    )

    padding_mask[
        content_mask == 0
    ] = 0

    content_pixels = max(
        1,
        int(
            np.count_nonzero(
                content_mask
            )
        ),
    )

    padding_ratio = (
        np.count_nonzero(
            padding_mask
        )
        / content_pixels
    )

    # If most of the photograph is dark, it is probably a real
    # dark background rather than rotated-image padding.
    if (
        padding_ratio
        > MAXIMUM_PADDING_RATIO
    ):
        return content_mask.copy()

    cleaned_content = (
        content_mask.copy()
    )

    cleaned_content[
        padding_mask > 0
    ] = 0

    return cleaned_content


def _separate_foreground(
    rgb_image: np.ndarray,
    content_mask: np.ndarray,
) -> np.ndarray:
    """
    Separate the banana from its surrounding background.

    Black border-connected padding is removed first. Background
    colour is then estimated from the usable image boundary.
    Lightness differences are downweighted to suppress shadows.
    """

    if (
        np.count_nonzero(
            content_mask
        )
        == 0
    ):
        return np.zeros_like(
            content_mask
        )

    usable_content = (
        _remove_dark_border_padding(
            rgb_image,
            content_mask,
        )
    )

    if (
        np.count_nonzero(
            usable_content
        )
        == 0
    ):
        return np.zeros_like(
            content_mask
        )

    height, width = (
        usable_content.shape
    )

    border_width = max(
        3,
        int(
            round(
                min(height, width)
                * 0.06
            )
        ),
    )

    border_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                border_width * 2 + 1,
                border_width * 2 + 1,
            ),
        )
    )

    inner_content = cv2.erode(
        usable_content,
        border_kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    border_mask = cv2.subtract(
        usable_content,
        inner_content,
    )

    blurred = cv2.GaussianBlur(
        rgb_image,
        (5, 5),
        0,
    )

    lab_image = cv2.cvtColor(
        blurred,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    # Prefer non-black border pixels when estimating the
    # surrounding paper or table colour.
    border_brightness = np.max(
        blurred,
        axis=2,
    )

    preferred_border = (
        (border_mask > 0)
        & (
            border_brightness
            > DARK_PADDING_THRESHOLD
        )
    )

    minimum_samples = max(
        20,
        int(
            np.count_nonzero(
                border_mask
            )
            * 0.10
        ),
    )

    if (
        np.count_nonzero(
            preferred_border
        )
        >= minimum_samples
    ):
        border_pixels = (
            lab_image[
                preferred_border
            ]
        )
    else:
        border_pixels = (
            lab_image[
                border_mask > 0
            ]
        )

    if border_pixels.size == 0:
        border_pixels = (
            lab_image[
                usable_content > 0
            ]
        )

    background_colour = (
        np.median(
            border_pixels,
            axis=0,
        )
    )

    # Use the complete Lab difference.
    # This preserves dark and discoloured banana regions.
    colour_distance = np.linalg.norm(
        lab_image - background_colour,
        axis=2,
    )

    valid_distances = (
        colour_distance[
            usable_content > 0
        ]
    )

    scale_limit = max(
        1.0,
        float(
            np.percentile(
                valid_distances,
                99,
            )
        ),
    )

    distance_image = np.clip(
        colour_distance
        * 255.0
        / scale_limit,
        0,
        255,
    ).astype(np.uint8)

    otsu_threshold, _ = (
        cv2.threshold(
            distance_image[
                usable_content > 0
            ].reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )
    )

    threshold = max(
        8.0,
        float(
            otsu_threshold
        )
        * 0.90,
    )

    mask = np.where(
        (
            distance_image
            >= threshold
        )
        & (
            usable_content > 0
        ),
        255,
        0,
    ).astype(np.uint8)

    opening_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )
    )

    closing_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )

    mask[
        usable_content == 0
    ] = 0

    mask = _largest_component(
        mask
    )

    mask = _fill_holes(
        mask
    )

    mask[
        usable_content == 0
    ] = 0

    return mask


def segment_banana(
    rgb_image: np.ndarray,
    content_mask: np.ndarray | None = None,
) -> SegmentationResult:
    """
    Extract the main foreground object using classical
    image-processing techniques.
    """

    rgb_image = np.asarray(rgb_image)

    if (
        rgb_image.ndim != 3
        or rgb_image.shape[2] != 3
    ):
        raise ValueError(
            "rgb_image must have shape "
            "(height, width, 3)."
        )

    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(
            rgb_image,
            0,
            255,
        ).astype(np.uint8)

    rgb_image = np.ascontiguousarray(
        rgb_image
    )

    height, width = rgb_image.shape[:2]

    valid_content = _normalise_content_mask(
        content_mask,
        (height, width),
    )

    final_mask = _separate_foreground(
        rgb_image,
        valid_content,
    )

    foreground_pixels = int(
        np.count_nonzero(final_mask)
    )

    content_pixels = max(
        1,
        int(
            np.count_nonzero(
                valid_content
            )
        ),
    )

    area_ratio = (
        foreground_pixels
        / content_pixels
    )

    if foreground_pixels == 0:
        success = False
        message = (
            "No foreground object was detected."
        )

    elif area_ratio < MINIMUM_AREA_RATIO:
        success = False
        message = (
            "The detected foreground region "
            "is too small."
        )

    elif area_ratio > MAXIMUM_AREA_RATIO:
        success = False
        message = (
            "The detected foreground covers "
            "almost the entire image."
        )

    else:
        success = True
        message = (
            "The main foreground region was "
            "extracted using classical image processing."
        )

    overlay_rgb = rgb_image.copy()

    contours, _ = cv2.findContours(
        final_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay_rgb,
        contours,
        -1,
        (0, 255, 0),
        2,
    )

    return SegmentationResult(
        success=success,
        message=message,
        final_mask=final_mask,
        overlay_rgb=overlay_rgb,
    )