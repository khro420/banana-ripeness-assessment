from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

@dataclass
class SegmentationResult:
    """Outputs from the shared banana segmentation pipeline."""

    success: bool
    message: str

    initial_candidate_mask: np.ndarray
    grabcut_mask: np.ndarray
    final_mask: np.ndarray
    inner_mask: np.ndarray

    segmented_rgb: np.ndarray
    foreground_rgba: np.ndarray
    overlay_rgb: np.ndarray

    banana_area_pixels: int
    banana_area_percent: float
    bounding_box: tuple[int, int, int, int] | None

    diagnostics: dict[str, Any]


def _normalise_binary_mask(
    mask: np.ndarray | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape

    if mask is None:
        return np.full((height, width), 255, dtype=np.uint8)

    mask = np.asarray(mask)

    if mask.shape != (height, width):
        raise ValueError(
            "The content mask must match the image dimensions."
        )

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _odd_kernel_size(value: int) -> int:
    value = max(3, int(value))

    if value % 2 == 0:
        value += 1

    return value


def _mask_area_percent(
    mask: np.ndarray,
    content_mask: np.ndarray,
) -> float:
    content_pixels = max(1, np.count_nonzero(content_mask))
    foreground_pixels = np.count_nonzero(mask)

    return 100.0 * foreground_pixels / content_pixels


def _content_border_ring(
    content_mask: np.ndarray,
) -> np.ndarray:
    height, width = content_mask.shape

    kernel_size = _odd_kernel_size(
        round(min(height, width) * 0.07)
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    inner_content = cv2.erode(content_mask, kernel)

    return cv2.subtract(content_mask, inner_content)

def _create_initial_candidate_mask(
    rgb_image: np.ndarray,
    content_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Create banana-colour seeds without treating every background difference
    as foreground.

    Clear white and near-black background pixels are excluded. Background
    difference is only allowed to expand around an existing banana-colour
    region.
    """

    blurred_rgb = cv2.GaussianBlur(
        rgb_image,
        (5, 5),
        sigmaX=1.2,
        sigmaY=1.2,
    )

    hsv = cv2.cvtColor(
        blurred_rgb,
        cv2.COLOR_RGB2HSV,
    )

    lab = cv2.cvtColor(
        blurred_rgb,
        cv2.COLOR_RGB2LAB,
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    lab_a = lab[:, :, 1].astype(np.int16)
    lab_b = lab[:, :, 2].astype(np.int16)

    chroma_distance = np.sqrt(
        np.square(lab_a - 128)
        + np.square(lab_b - 128)
    )

    # Obvious background in this dataset:
    # white/light paper and almost-pure black corners.
    obvious_light_background = (
        (saturation <= 45)
        & (value >= 170)
    )

    obvious_dark_background = value <= 10

    obvious_background = (
        obvious_light_background
        | obvious_dark_background
        | (content_mask == 0)
    )

    # Broad brown, yellow and green banana-colour family.
    hsv_colour_family = (
        (hue <= 100)
        & (saturation >= 30)
        & (value >= 24)
    )

    lab_colour_family = (
        (
            (lab_b >= 132)
            | (lab_a <= 123)
        )
        & (chroma_distance >= 10)
        & (value >= 24)
    )

    colour_core = (
        hsv_colour_family
        | lab_colour_family
    ) & ~obvious_background

    colour_core_mask = np.where(
        colour_core,
        255,
        0,
    ).astype(np.uint8)

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    colour_core_mask = cv2.morphologyEx(
        colour_core_mask,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1,
    )

    # Only inspect background differences close to a banana-colour seed.
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31),
    )

    colour_support = cv2.dilate(
        colour_core_mask,
        support_kernel,
        iterations=1,
    )

    border_ring = _content_border_ring(content_mask)
    border_pixels = lab[border_ring > 0]

    if border_pixels.size == 0:
        background_lab = np.median(
            lab.reshape(-1, 3),
            axis=0,
        )
    else:
        background_lab = np.median(
            border_pixels,
            axis=0,
        )

    lab_difference = (
        lab.astype(np.float32)
        - background_lab.astype(np.float32)
    )

    background_distance = np.linalg.norm(
        lab_difference,
        axis=2,
    )

    border_distances = background_distance[
        border_ring > 0
    ]

    if border_distances.size == 0:
        background_threshold = 18.0
    else:
        background_threshold = max(
            16.0,
            float(np.percentile(border_distances, 95)) + 5.0,
        )

    different_from_background = (
        background_distance >= background_threshold
    )

    supported_difference = (
        different_from_background
        & (colour_support > 0)
        & ~obvious_background
    )

    # Recover dark spots, stem regions and rotten peel only when they are
    # spatially connected to a banana-colour neighbourhood.
    nearby_dark_region = (
        (colour_support > 0)
        & (value > 10)
        & (value < 120)
        & ~obvious_light_background
        & (content_mask > 0)
    )

    initial_candidate = (
        (colour_core_mask > 0)
        | supported_difference
        | nearby_dark_region
    )

    initial_mask = np.where(
        initial_candidate & (content_mask > 0),
        255,
        0,
    ).astype(np.uint8)

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    initial_mask = cv2.morphologyEx(
        initial_mask,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1,
    )

    initial_mask = cv2.morphologyEx(
        initial_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )

    # The safest definite foreground pixels are inside the colour core.
    strong_seed = cv2.erode(
        colour_core_mask,
        opening_kernel,
        iterations=1,
    )

    if np.count_nonzero(strong_seed) < 40:
        strong_seed = cv2.erode(
            initial_mask,
            opening_kernel,
            iterations=1,
        )

    return (
        initial_mask,
        strong_seed,
        background_threshold,
    )

def _content_rectangle(
    content_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(content_mask)

    if points is None:
        raise ValueError("The content mask is empty.")

    x, y, width, height = cv2.boundingRect(points)

    margin_x = max(1, round(width * 0.03))
    margin_y = max(1, round(height * 0.03))

    rectangle_x = x + margin_x
    rectangle_y = y + margin_y
    rectangle_width = max(2, width - 2 * margin_x)
    rectangle_height = max(2, height - 2 * margin_y)

    return (
        rectangle_x,
        rectangle_y,
        rectangle_width,
        rectangle_height,
    )

def _run_grabcut(
    rgb_image: np.ndarray,
    content_mask: np.ndarray,
    initial_mask: np.ndarray,
    strong_seed: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, bool]:
    """
    Refine the banana candidate using mask-initialised GrabCut.

    The foreground is restricted to a generous neighbourhood around the
    initial banana-colour candidate.
    """

    bgr_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )

    height, width = initial_mask.shape

    support_kernel_size = _odd_kernel_size(
        round(min(height, width) * 0.10)
    )

    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            support_kernel_size,
            support_kernel_size,
        ),
    )

    foreground_support = cv2.dilate(
        initial_mask,
        support_kernel,
        iterations=1,
    )

    foreground_support[
        content_mask == 0
    ] = 0

    initial_coverage = _mask_area_percent(
        initial_mask,
        content_mask,
    )

    # Reject an invalid initial mask before sending it to GrabCut.
    if (
        np.count_nonzero(strong_seed) < 20
        or initial_coverage < 0.5
        or initial_coverage > 85.0
    ):
        return initial_mask.copy(), True

    # Begin with definite background everywhere.
    grabcut_labels = np.full(
        initial_mask.shape,
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    # The expanded candidate is possible foreground.
    grabcut_labels[
        (foreground_support > 0)
        & (content_mask > 0)
    ] = cv2.GC_PR_FGD

    # The original candidate is more likely to be foreground.
    grabcut_labels[
        initial_mask > 0
    ] = cv2.GC_PR_FGD

    # Eroded colour regions are definite foreground.
    grabcut_labels[
        strong_seed > 0
    ] = cv2.GC_FGD

    grabcut_labels[
        content_mask == 0
    ] = cv2.GC_BGD

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    try:
        cv2.grabCut(
            bgr_image,
            grabcut_labels,
            None,
            background_model,
            foreground_model,
            max(2, int(iterations)),
            cv2.GC_INIT_WITH_MASK,
        )

        binary_mask = np.where(
            (grabcut_labels == cv2.GC_FGD)
            | (grabcut_labels == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        # Do not allow GrabCut to escape far outside the seed area.
        binary_mask[
            foreground_support == 0
        ] = 0

        binary_mask[
            content_mask == 0
        ] = 0

        binary_mask = cv2.bitwise_or(
            binary_mask,
            strong_seed,
        )

        coverage = _mask_area_percent(
            binary_mask,
            content_mask,
        )

        if coverage < 0.5 or coverage > 85.0:
            return initial_mask.copy(), True

        return binary_mask, False

    except cv2.error:
        return initial_mask.copy(), True

def _select_primary_component(
    binary_mask: np.ndarray,
    colour_seed: np.ndarray,
    content_mask: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Select the component most likely to be the main banana object."""

    component_count, labels, statistics, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    actual_component_count = max(0, component_count - 1)

    if actual_component_count == 0:
        return np.zeros_like(binary_mask), 0, 0.0

    content_points = cv2.findNonZero(content_mask)

    if content_points is None:
        return np.zeros_like(binary_mask), 0, 0.0

    content_x, content_y, content_width, content_height = (
        cv2.boundingRect(content_points)
    )

    content_area = max(1, np.count_nonzero(content_mask))

    content_centre_x = content_x + content_width / 2
    content_centre_y = content_y + content_height / 2

    maximum_distance = max(
        1.0,
        np.hypot(content_width / 2, content_height / 2),
    )

    best_label = None
    best_score = float("-inf")

    for label_number in range(1, component_count):
        area = int(
            statistics[
                label_number,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < 25:
            continue

        component = labels == label_number

        area_ratio = area / content_area

        colour_overlap = (
            np.count_nonzero(
                component & (colour_seed > 0)
            )
            / area
        )

        centre_x, centre_y = centroids[label_number]

        centre_distance = np.hypot(
            centre_x - content_centre_x,
            centre_y - content_centre_y,
        )

        centre_score = 1.0 - min(
            centre_distance / maximum_distance,
            1.0,
        )

        size_score = min(
            area_ratio / 0.45,
            1.0,
        )

        component_x = int(
            statistics[
                label_number,
                cv2.CC_STAT_LEFT,
            ]
        )
        component_y = int(
            statistics[
                label_number,
                cv2.CC_STAT_TOP,
            ]
        )
        component_width = int(
            statistics[
                label_number,
                cv2.CC_STAT_WIDTH,
            ]
        )
        component_height = int(
            statistics[
                label_number,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        touched_sides = sum(
            [
                component_x <= content_x + 1,
                component_y <= content_y + 1,
                (
                    component_x + component_width
                    >= content_x + content_width - 1
                ),
                (
                    component_y + component_height
                    >= content_y + content_height - 1
                ),
            ]
        )

        edge_penalty = touched_sides / 4.0
        oversize_penalty = max(
            0.0,
            (area_ratio - 0.75) / 0.25,
        )

        component_score = (
            0.45 * size_score
            + 0.35 * colour_overlap
            + 0.20 * centre_score
            - 0.25 * edge_penalty
            - 0.80 * oversize_penalty
        )

        if component_score > best_score:
            best_score = component_score
            best_label = label_number

    if best_label is None:
        return (
            np.zeros_like(binary_mask),
            actual_component_count,
            0.0,
        )

    selected_mask = np.where(
        labels == best_label,
        255,
        0,
    ).astype(np.uint8)

    return selected_mask, actual_component_count, best_score


def _fill_internal_holes(
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Fill background regions fully enclosed by the foreground."""

    padded = cv2.copyMakeBorder(
        binary_mask,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    flood_filled = padded.copy()
    flood_mask = np.zeros(
        (
            padded.shape[0] + 2,
            padded.shape[1] + 2,
        ),
        dtype=np.uint8,
    )

    cv2.floodFill(
        flood_filled,
        flood_mask,
        seedPoint=(0, 0),
        newVal=255,
    )

    holes = cv2.bitwise_not(flood_filled)
    filled = cv2.bitwise_or(padded, holes)

    return filled[1:-1, 1:-1]


def _create_inner_mask(
    final_mask: np.ndarray,
    bounding_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Erode the mask slightly to reduce boundary contamination."""

    if bounding_box is None:
        return np.zeros_like(final_mask)

    _, _, width, height = bounding_box

    kernel_size = _odd_kernel_size(
        round(min(width, height) * 0.025)
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    inner_mask = cv2.erode(
        final_mask,
        kernel,
        iterations=1,
    )

    if np.count_nonzero(inner_mask) == 0:
        return final_mask.copy()

    return inner_mask


def _create_output_images(
    rgb_image: np.ndarray,
    final_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create isolated, transparent and boundary-overlay views."""

    foreground = final_mask > 0

    segmented_rgb = np.full_like(
        rgb_image,
        245,
    )
    segmented_rgb[foreground] = rgb_image[foreground]

    foreground_rgba = np.dstack(
        [rgb_image, final_mask]
    )

    overlay_rgb = rgb_image.copy()

    green_tint = np.zeros_like(rgb_image)
    green_tint[:, :, 1] = 255

    if np.any(foreground):
        blended_foreground = (
            0.82 * rgb_image[foreground].astype(np.float32)
            + 0.18 * green_tint[foreground].astype(np.float32)
        )

        overlay_rgb[foreground] = np.clip(
            blended_foreground,
            0,
            255,
        ).astype(np.uint8)

    contours, _ = cv2.findContours(
        final_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay_rgb,
        contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2,
    )

    return segmented_rgb, foreground_rgba, overlay_rgb


def segment_banana(
    rgb_image: np.ndarray,
    content_mask: np.ndarray | None = None,
    grabcut_iterations: int = 5,
    minimum_area_ratio: float = 0.02,
) -> SegmentationResult:
    """
    Segment the primary banana object from one standardised RGB image.

    Assumption:
        The banana is the main object and is reasonably close to the centre.
    """

    rgb_image = np.asarray(rgb_image)

    if (
        rgb_image.ndim != 3
        or rgb_image.shape[2] != 3
    ):
        raise ValueError(
            "rgb_image must have shape (height, width, 3)."
        )

    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(
            rgb_image,
            0,
            255,
        ).astype(np.uint8)

    rgb_image = np.ascontiguousarray(rgb_image)

    height, width = rgb_image.shape[:2]

    content_mask = _normalise_binary_mask(
        content_mask,
        (height, width),
    )

    initial_mask, strong_seed, background_threshold = (
        _create_initial_candidate_mask(
            rgb_image,
            content_mask,
        )
    )

    grabcut_mask, used_fallback = _run_grabcut(
        rgb_image=rgb_image,
        content_mask=content_mask,
        initial_mask=initial_mask,
        strong_seed=strong_seed,
        iterations=grabcut_iterations,
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    cleaned_mask = cv2.morphologyEx(
        grabcut_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )

    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1,
    )

    cleaned_mask[content_mask == 0] = 0

    primary_mask, component_count, component_score = (
        _select_primary_component(
            cleaned_mask,
            strong_seed,
            content_mask,
        )
    )

    final_mask = _fill_internal_holes(primary_mask)

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1,
    )

    final_mask[content_mask == 0] = 0

    contours, _ = cv2.findContours(
        final_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    bounding_box = None

    if contours:
        largest_contour = max(
            contours,
            key=cv2.contourArea,
        )
        bounding_box = cv2.boundingRect(largest_contour)

    banana_area_pixels = int(
        np.count_nonzero(final_mask)
    )

    content_area = max(
        1,
        int(np.count_nonzero(content_mask)),
    )

    banana_area_ratio = (
        banana_area_pixels / content_area
    )

    banana_area_percent = (
        banana_area_ratio * 100.0
    )

    if banana_area_ratio < minimum_area_ratio:
        success = False
        message = (
            "No sufficiently large banana region was detected."
        )
    elif banana_area_ratio > 0.85:
        success = False
        message = (
            "The detected region covers most of the image and is "
            "probably background."
        )
    elif bounding_box is None:
        success = False
        message = "No valid banana contour was detected."
    else:
        success = True
        message = "A candidate banana region was detected."

    inner_mask = _create_inner_mask(
        final_mask,
        bounding_box,
    )

    segmented_rgb, foreground_rgba, overlay_rgb = (
        _create_output_images(
            rgb_image,
            final_mask,
        )
    )

    diagnostics = {
        "working_width": width,
        "working_height": height,
        "initial_candidate_percent": round(
            _mask_area_percent(
                initial_mask,
                content_mask,
            ),
            3,
        ),
        "grabcut_foreground_percent": round(
            _mask_area_percent(
                grabcut_mask,
                content_mask,
            ),
            3,
        ),
        "final_foreground_percent": round(
            banana_area_percent,
            3,
        ),
        "connected_components_found": component_count,
        "selected_component_score": round(
            float(component_score),
            4,
        ),
        "background_distance_threshold": round(
            float(background_threshold),
            3,
        ),
        "grabcut_fallback_used": used_fallback,
        "grabcut_iterations": grabcut_iterations,
        "minimum_area_ratio": minimum_area_ratio,
    }

    return SegmentationResult(
        success=success,
        message=message,
        initial_candidate_mask=initial_mask,
        grabcut_mask=grabcut_mask,
        final_mask=final_mask,
        inner_mask=inner_mask,
        segmented_rgb=segmented_rgb,
        foreground_rgba=foreground_rgba,
        overlay_rgb=overlay_rgb,
        banana_area_pixels=banana_area_pixels,
        banana_area_percent=banana_area_percent,
        bounding_box=bounding_box,
        diagnostics=diagnostics,
    )


def segmentation_status() -> str:
    return (
        "Shared GrabCut and morphology segmentation is connected."
    )