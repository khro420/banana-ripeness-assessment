from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MorphologyParameters:
    """Dark-mask settings frozen after validation calibration."""

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
    greyscale_image: np.ndarray
    blackhat_response: np.ndarray
    raw_blemish_mask: np.ndarray
    blemish_mask: np.ndarray
    blemish_overlay_rgb: np.ndarray
    spread_overlay_rgb: np.ndarray
    banana_area_pixels: int
    dark_area_pixels: int
    total_dark_percentage: float
    dark_region_spread: float
    active_spread_cells: int
    valid_spread_cells: int

    # These aliases preserve compatibility with the evaluator.
    @property
    def blemish_area_pixels(self) -> int:
        return self.dark_area_pixels

    @property
    def total_dark_area_pixels(self) -> int:
        return self.dark_area_pixels

    @property
    def blemish_percentage(self) -> float:
        return self.total_dark_percentage

    @property
    def normal_surface_percentage(self) -> float:
        return max(0.0, 100.0 - self.total_dark_percentage)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _validate_parameters(parameters: MorphologyParameters) -> None:
    kernel_names = (
        "gaussian_kernel_size",
        "blackhat_kernel_size",
        "opening_kernel_size",
        "closing_kernel_size",
    )
    for name in kernel_names:
        value = getattr(parameters, name)
        if value < 3 or value % 2 == 0:
            raise ValueError(f"{name} must be an odd number of at least 3.")

    threshold_names = (
        "blackhat_threshold",
        "absolute_dark_threshold",
        "local_dark_intensity_ceiling",
    )
    for name in threshold_names:
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
) -> tuple[float, int, int, list[tuple[int, int, int, int, bool]]]:
    points = cv2.findNonZero(banana_mask)
    if points is None:
        return 0.0, 0, 0, []

    x, y, width, height = cv2.boundingRect(points)
    grid_size = parameters.spread_grid_size
    x_edges = np.linspace(x, x + width, grid_size + 1, dtype=int)
    y_edges = np.linspace(y, y + height, grid_size + 1, dtype=int)
    banana_area = int(np.count_nonzero(banana_mask))
    minimum_cell_area = max(
        10,
        round(banana_area * parameters.minimum_grid_banana_ratio),
    )

    active_cells = 0
    valid_cells = 0
    cells = []

    for row in range(grid_size):
        for column in range(grid_size):
            x1, x2 = int(x_edges[column]), int(x_edges[column + 1])
            y1, y2 = int(y_edges[row]), int(y_edges[row + 1])
            cell_banana = banana_mask[y1:y2, x1:x2]
            cell_dark = dark_mask[y1:y2, x1:x2]
            cell_area = int(np.count_nonzero(cell_banana))

            if cell_area < minimum_cell_area:
                continue

            valid_cells += 1
            dark_ratio = np.count_nonzero(cell_dark) / cell_area
            is_active = dark_ratio >= parameters.spread_cell_dark_ratio
            active_cells += int(is_active)
            cells.append((x1, y1, x2, y2, is_active))

    spread = 100.0 * active_cells / valid_cells if valid_cells else 0.0
    return spread, active_cells, valid_cells, cells


def _dim_background(rgb_image: np.ndarray, banana_mask: np.ndarray) -> np.ndarray:
    overlay = rgb_image.copy()
    background = banana_mask == 0
    overlay[background] = (
        overlay[background].astype(np.float32) * 0.30
    ).astype(np.uint8)
    return overlay


def _dark_overlay(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    dark_mask: np.ndarray,
) -> np.ndarray:
    overlay = _dim_background(rgb_image, banana_mask)
    dark_pixels = dark_mask > 0
    red = np.zeros_like(rgb_image)
    red[:, :, 0] = 255
    overlay[dark_pixels] = (
        0.5 * rgb_image[dark_pixels].astype(np.float32)
        + 0.5 * red[dark_pixels].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def _spread_overlay(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    cells: list[tuple[int, int, int, int, bool]],
) -> np.ndarray:
    overlay = _dim_background(rgb_image, banana_mask)

    for x1, y1, x2, y2, is_active in cells:
        colour = (0, 255, 0) if is_active else (30, 120, 255)
        cv2.rectangle(
            overlay,
            (x1, y1),
            (max(x1, x2 - 1), max(y1, y2 - 1)),
            colour,
            1,
        )

    return overlay


def detect_blemishes(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: MorphologyParameters | None = None,
) -> MorphologyMaskResult:
    """
    Detect dark peel regions with greyscale morphology.

    The full supplied banana mask is used, including both tips. No colour
    classifier or machine-learning model is used here.
    """

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

    greyscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(
        greyscale,
        (parameters.gaussian_kernel_size, parameters.gaussian_kernel_size),
        sigmaX=1.2,
    )

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (parameters.blackhat_kernel_size, parameters.blackhat_kernel_size),
    )
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, blackhat_kernel)

    local_dark = (
        (blackhat >= parameters.blackhat_threshold)
        & (blurred <= parameters.local_dark_intensity_ceiling)
        & (banana_mask > 0)
    )
    absolute_dark = (
        (blurred <= parameters.absolute_dark_threshold)
        & (banana_mask > 0)
    )
    raw_mask = np.where(local_dark | absolute_dark, 255, 0).astype(np.uint8)

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (parameters.opening_kernel_size, parameters.opening_kernel_size),
    )
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (parameters.closing_kernel_size, parameters.closing_kernel_size),
    )
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, opening_kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, closing_kernel)
    minimum_area = max(
        5,
        round(banana_area * parameters.minimum_component_area_ratio),
    )
    cleaned_mask = _remove_small_components(cleaned_mask, minimum_area)
    cleaned_mask = cv2.bitwise_and(cleaned_mask, banana_mask)

    dark_area = int(np.count_nonzero(cleaned_mask))
    total_dark = 100.0 * dark_area / banana_area
    spread, active_cells, valid_cells, cells = _measure_spread(
        cleaned_mask,
        banana_mask,
        parameters,
    )

    displayed_greyscale = greyscale.copy()
    displayed_blackhat = blackhat.copy()
    displayed_greyscale[banana_mask == 0] = 0
    displayed_blackhat[banana_mask == 0] = 0

    return MorphologyMaskResult(
        greyscale_image=displayed_greyscale,
        blackhat_response=displayed_blackhat,
        raw_blemish_mask=raw_mask,
        blemish_mask=cleaned_mask,
        blemish_overlay_rgb=_dark_overlay(rgb_image, banana_mask, cleaned_mask),
        spread_overlay_rgb=_spread_overlay(rgb_image, banana_mask, cells),
        banana_area_pixels=banana_area,
        dark_area_pixels=dark_area,
        total_dark_percentage=total_dark,
        dark_region_spread=spread,
        active_spread_cells=active_cells,
        valid_spread_cells=valid_cells,
    )
