from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MorphologyParameters:
    # These settings control how dark peel regions are detected.
    gaussian_kernel_size: int = 5
    blackhat_kernel_size: int = 21
    blackhat_threshold: int = 15
    absolute_dark_threshold: int = 75
    local_dark_intensity_ceiling: int = 115
    opening_kernel_size: int = 3
    closing_kernel_size: int = 5
    minimum_component_area_ratio: float = 0.0003
    spread_grid_size: int = 5
    spread_cell_dark_ratio: float = 0.05
    minimum_grid_banana_ratio: float = 0.01


@dataclass
class MorphologyMaskResult:
    # Keep only images and measurements used by the analysis or page.
    greyscale_image: np.ndarray
    blemish_mask: np.ndarray
    blemish_overlay_rgb: np.ndarray
    banana_area_pixels: int
    dark_area_pixels: int
    total_dark_percentage: float
    dark_region_spread: float


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _validate_parameters(parameters: MorphologyParameters) -> None:
    # Check settings once so a bad value does not give misleading results.
    for name in ("gaussian_kernel_size", "blackhat_kernel_size", "opening_kernel_size", "closing_kernel_size"):
        value = getattr(parameters, name)
        if value < 3 or value % 2 == 0:
            raise ValueError(f"{name} must be an odd number of at least 3.")
    for name in ("blackhat_threshold", "absolute_dark_threshold", "local_dark_intensity_ceiling"):
        if not 0 <= getattr(parameters, name) <= 255:
            raise ValueError(f"{name} must be between 0 and 255.")
    if not 0 <= parameters.minimum_component_area_ratio <= 1:
        raise ValueError("minimum_component_area_ratio must be between 0 and 1.")
    if parameters.spread_grid_size < 2:
        raise ValueError("spread_grid_size must be at least 2.")
    if not 0 < parameters.spread_cell_dark_ratio <= 1:
        raise ValueError("spread_cell_dark_ratio must be above 0 and at most 1.")
    if not 0 <= parameters.minimum_grid_banana_ratio < 1:
        raise ValueError("minimum_grid_banana_ratio must be at least 0 and below 1.")


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    # Small isolated pixels are usually noise, not peel damage.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def _measure_spread(
    dark_mask: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters,
) -> float:
    # Split the banana into a grid and count cells with enough dark pixels.
    points = cv2.findNonZero(banana_mask)
    if points is None:
        return 0.0
    x, y, width, height = cv2.boundingRect(points)
    size = parameters.spread_grid_size
    x_edges = np.linspace(x, x + width, size + 1, dtype=int)
    y_edges = np.linspace(y, y + height, size + 1, dtype=int)
    minimum_area = max(10, round(np.count_nonzero(banana_mask) * parameters.minimum_grid_banana_ratio))
    active = valid = 0
    for row in range(size):
        for column in range(size):
            x1, x2 = x_edges[column], x_edges[column + 1]
            y1, y2 = y_edges[row], y_edges[row + 1]
            banana_area = int(np.count_nonzero(banana_mask[y1:y2, x1:x2]))
            if banana_area < minimum_area:
                continue
            valid += 1
            dark_ratio = np.count_nonzero(dark_mask[y1:y2, x1:x2]) / banana_area
            active += int(dark_ratio >= parameters.spread_cell_dark_ratio)
    return 100.0 * active / valid if valid else 0.0


def _dark_overlay(
    rgb_image: np.ndarray, banana_mask: np.ndarray, dark_mask: np.ndarray
) -> np.ndarray:
    # This is the one diagnostic visual kept for the demo.
    overlay = rgb_image.copy()
    overlay[banana_mask == 0] = (overlay[banana_mask == 0] * 0.30).astype(np.uint8)
    dark = dark_mask > 0
    overlay[dark] = (0.5 * rgb_image[dark] + np.array([127.5, 0, 0])).astype(np.uint8)
    return overlay


def detect_blemishes(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters | None = None,
) -> MorphologyMaskResult:
    """Find cleaned dark regions inside an already segmented banana."""
    parameters = parameters or MorphologyParameters()
    _validate_parameters(parameters)
    rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    banana_mask = _binary_mask(np.asarray(banana_mask))
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3).")
    if banana_mask.shape != rgb_image.shape[:2]:
        raise ValueError("The banana mask and image dimensions do not match.")
    banana_area = int(np.count_nonzero(banana_mask))
    if banana_area < 100:
        raise ValueError("The banana region is too small for morphology analysis.")

    # Convert to grey, then use black-hat to highlight local dark patches.
    greyscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(greyscale, (parameters.gaussian_kernel_size,) * 2, sigmaX=1.2)
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (parameters.blackhat_kernel_size,) * 2
    )
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, blackhat_kernel)
    local_dark = (blackhat >= parameters.blackhat_threshold) & (blurred <= parameters.local_dark_intensity_ceiling) & (banana_mask > 0)
    absolute_dark = (blurred <= parameters.absolute_dark_threshold) & (banana_mask > 0)
    raw_mask = np.where(local_dark | absolute_dark, 255, 0).astype(np.uint8)

    # Opening removes specks; closing joins nearby dark pixels.
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (parameters.opening_kernel_size,) * 2)
    closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (parameters.closing_kernel_size,) * 2)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, opening)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, closing)
    cleaned = _remove_small_components(cleaned, max(5, round(banana_area * parameters.minimum_component_area_ratio)))
    cleaned = cv2.bitwise_and(cleaned, banana_mask)

    dark_area = int(np.count_nonzero(cleaned))
    greyscale[banana_mask == 0] = 0
    return MorphologyMaskResult(
        greyscale_image=greyscale,
        blemish_mask=cleaned,
        blemish_overlay_rgb=_dark_overlay(rgb_image, banana_mask, cleaned),
        banana_area_pixels=banana_area,
        dark_area_pixels=dark_area,
        total_dark_percentage=100.0 * dark_area / banana_area,
        dark_region_spread=_measure_spread(cleaned, banana_mask, parameters),
    )
