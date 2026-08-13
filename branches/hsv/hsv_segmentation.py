from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVParameters:
    """HSV thresholds used to separate visible banana-peel colours."""

    # OpenCV HSV ranges: H = 0-179, S/V = 0-255.
    green_hue_min: int = 35
    green_hue_max: int = 85
    green_saturation_min: int = 35
    green_value_min: int = 35

    yellow_hue_min: int = 18
    yellow_hue_max: int = 34
    yellow_saturation_min: int = 35
    yellow_value_min: int = 70

    brown_hue_min: int = 5
    brown_hue_max: int = 17
    brown_saturation_min: int = 35
    brown_value_min: int = 35
    brown_value_max: int = 180

    # Very dark pixels have unreliable hue, so Value is used directly.
    dark_value_max: int = 70


@dataclass
class HSVMaskResult:
    hsv_image: np.ndarray
    green_mask: np.ndarray
    yellow_mask: np.ndarray
    brown_mask: np.ndarray
    dark_mask: np.ndarray
    other_mask: np.ndarray
    colour_overlay_rgb: np.ndarray

    banana_area_pixels: int
    green_area_pixels: int
    yellow_area_pixels: int
    brown_area_pixels: int
    dark_area_pixels: int
    other_area_pixels: int

    green_percentage: float
    yellow_percentage: float
    brown_percentage: float
    dark_percentage: float
    other_percentage: float
    deteriorated_percentage: float


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _validate_parameters(parameters: HSVParameters) -> None:
    hue_names = (
        "green_hue_min",
        "green_hue_max",
        "yellow_hue_min",
        "yellow_hue_max",
        "brown_hue_min",
        "brown_hue_max",
    )
    for name in hue_names:
        value = getattr(parameters, name)
        if not 0 <= value <= 179:
            raise ValueError(f"{name} must be between 0 and 179.")

    channel_names = (
        "green_saturation_min",
        "green_value_min",
        "yellow_saturation_min",
        "yellow_value_min",
        "brown_saturation_min",
        "brown_value_min",
        "brown_value_max",
        "dark_value_max",
    )
    for name in channel_names:
        value = getattr(parameters, name)
        if not 0 <= value <= 255:
            raise ValueError(f"{name} must be between 0 and 255.")

    if parameters.green_hue_min > parameters.green_hue_max:
        raise ValueError("Green minimum hue cannot exceed maximum hue.")
    if parameters.yellow_hue_min > parameters.yellow_hue_max:
        raise ValueError("Yellow minimum hue cannot exceed maximum hue.")
    if parameters.brown_hue_min > parameters.brown_hue_max:
        raise ValueError("Brown minimum hue cannot exceed maximum hue.")
    if parameters.brown_value_min > parameters.brown_value_max:
        raise ValueError("Brown minimum value cannot exceed maximum value.")


def _percentage(area_pixels: int, banana_area_pixels: int) -> float:
    if banana_area_pixels <= 0:
        return 0.0
    return 100.0 * area_pixels / banana_area_pixels


def _dim_background(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
) -> np.ndarray:
    overlay = rgb_image.copy()
    background = banana_mask == 0
    overlay[background] = (
        overlay[background].astype(np.float32) * 0.30
    ).astype(np.uint8)
    return overlay


def _create_colour_overlay(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    green_mask: np.ndarray,
    yellow_mask: np.ndarray,
    brown_mask: np.ndarray,
    dark_mask: np.ndarray,
) -> np.ndarray:
    overlay = _dim_background(rgb_image, banana_mask)

    regions = (
        (green_mask, np.array([0, 255, 0], dtype=np.float32)),
        (yellow_mask, np.array([255, 220, 0], dtype=np.float32)),
        (brown_mask, np.array([165, 85, 25], dtype=np.float32)),
        (dark_mask, np.array([255, 0, 0], dtype=np.float32)),
    )

    for mask, colour in regions:
        pixels = mask > 0
        if not np.any(pixels):
            continue
        overlay[pixels] = (
            0.45 * rgb_image[pixels].astype(np.float32)
            + 0.55 * colour
        ).astype(np.uint8)

    return overlay


def segment_hsv_colours(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: HSVParameters | None = None,
) -> HSVMaskResult:
    """Separate the segmented banana into green/yellow/brown/dark regions."""

    parameters = parameters or HSVParameters()
    _validate_parameters(parameters)

    rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    banana_mask = _binary_mask(np.asarray(banana_mask))

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3).")
    if banana_mask.shape != rgb_image.shape[:2]:
        raise ValueError("The banana mask and image dimensions do not match.")

    banana_area_pixels = int(np.count_nonzero(banana_mask))
    if banana_area_pixels < 100:
        raise ValueError("The banana region is too small for HSV analysis.")

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]
    value = hsv_image[:, :, 2]
    inside_banana = banana_mask > 0

    # Dark first because hue is unreliable when brightness is very low.
    dark_pixels = (
        (value <= parameters.dark_value_max)
        & inside_banana
    )

    green_pixels = (
        (hue >= parameters.green_hue_min)
        & (hue <= parameters.green_hue_max)
        & (saturation >= parameters.green_saturation_min)
        & (value >= parameters.green_value_min)
        & inside_banana
        & ~dark_pixels
    )

    yellow_pixels = (
        (hue >= parameters.yellow_hue_min)
        & (hue <= parameters.yellow_hue_max)
        & (saturation >= parameters.yellow_saturation_min)
        & (value >= parameters.yellow_value_min)
        & inside_banana
        & ~dark_pixels
    )

    brown_pixels = (
        (hue >= parameters.brown_hue_min)
        & (hue <= parameters.brown_hue_max)
        & (saturation >= parameters.brown_saturation_min)
        & (value >= parameters.brown_value_min)
        & (value <= parameters.brown_value_max)
        & inside_banana
        & ~dark_pixels
    )

    classified_pixels = (
        green_pixels | yellow_pixels | brown_pixels | dark_pixels
    )
    other_pixels = inside_banana & ~classified_pixels

    green_mask = np.where(green_pixels, 255, 0).astype(np.uint8)
    yellow_mask = np.where(yellow_pixels, 255, 0).astype(np.uint8)
    brown_mask = np.where(brown_pixels, 255, 0).astype(np.uint8)
    dark_mask = np.where(dark_pixels, 255, 0).astype(np.uint8)
    other_mask = np.where(other_pixels, 255, 0).astype(np.uint8)

    green_area_pixels = int(np.count_nonzero(green_mask))
    yellow_area_pixels = int(np.count_nonzero(yellow_mask))
    brown_area_pixels = int(np.count_nonzero(brown_mask))
    dark_area_pixels = int(np.count_nonzero(dark_mask))
    other_area_pixels = int(np.count_nonzero(other_mask))

    green_percentage = _percentage(green_area_pixels, banana_area_pixels)
    yellow_percentage = _percentage(yellow_area_pixels, banana_area_pixels)
    brown_percentage = _percentage(brown_area_pixels, banana_area_pixels)
    dark_percentage = _percentage(dark_area_pixels, banana_area_pixels)
    other_percentage = _percentage(other_area_pixels, banana_area_pixels)
    deteriorated_percentage = brown_percentage + dark_percentage

    colour_overlay_rgb = _create_colour_overlay(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        green_mask=green_mask,
        yellow_mask=yellow_mask,
        brown_mask=brown_mask,
        dark_mask=dark_mask,
    )

    return HSVMaskResult(
        hsv_image=hsv_image,
        green_mask=green_mask,
        yellow_mask=yellow_mask,
        brown_mask=brown_mask,
        dark_mask=dark_mask,
        other_mask=other_mask,
        colour_overlay_rgb=colour_overlay_rgb,
        banana_area_pixels=banana_area_pixels,
        green_area_pixels=green_area_pixels,
        yellow_area_pixels=yellow_area_pixels,
        brown_area_pixels=brown_area_pixels,
        dark_area_pixels=dark_area_pixels,
        other_area_pixels=other_area_pixels,
        green_percentage=green_percentage,
        yellow_percentage=yellow_percentage,
        brown_percentage=brown_percentage,
        dark_percentage=dark_percentage,
        other_percentage=other_percentage,
        deteriorated_percentage=deteriorated_percentage,
    )
