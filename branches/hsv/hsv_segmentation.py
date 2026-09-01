from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class HSVParameters:
    # HSV thresholds
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


def _validate(p: HSVParameters) -> None:
    hue_fields = (
        "green_hue_min", "green_hue_max",
        "yellow_hue_min", "yellow_hue_max",
        "brown_hue_min", "brown_hue_max",
    )
    channel_fields = (
        "green_saturation_min", "green_value_min",
        "yellow_saturation_min", "yellow_value_min",
        "brown_saturation_min", "brown_value_min",
        "brown_value_max", "dark_value_max",
    )

    for name in hue_fields:
        if not 0 <= getattr(p, name) <= 179:
            raise ValueError(f"{name} must be between 0 and 179.")

    for name in channel_fields:
        if not 0 <= getattr(p, name) <= 255:
            raise ValueError(f"{name} must be between 0 and 255.")

    if p.green_hue_min > p.green_hue_max:
        raise ValueError("Green minimum hue cannot exceed maximum hue.")
    if p.yellow_hue_min > p.yellow_hue_max:
        raise ValueError("Yellow minimum hue cannot exceed maximum hue.")
    if p.brown_hue_min > p.brown_hue_max:
        raise ValueError("Brown minimum hue cannot exceed maximum hue.")
    if p.brown_value_min > p.brown_value_max:
        raise ValueError("Brown minimum value cannot exceed maximum value.")


def _mask(condition: np.ndarray) -> np.ndarray:
    return np.where(condition, 255, 0).astype(np.uint8)


def _percentage(mask: np.ndarray, banana_area: int) -> float:
    return 100.0 * np.count_nonzero(mask) / banana_area


def _overlay(rgb, banana_mask, green, yellow, brown, dark):
    output = rgb.copy()
    background = banana_mask == 0
    output[background] = (output[background].astype(np.float32) * 0.30).astype(np.uint8)

    regions = (
        (green, np.array([0, 255, 0], dtype=np.float32)),
        (yellow, np.array([255, 220, 0], dtype=np.float32)),
        (brown, np.array([165, 85, 25], dtype=np.float32)),
        (dark, np.array([255, 0, 0], dtype=np.float32)),
    )

    for mask, colour in regions:
        pixels = mask > 0
        if np.any(pixels):
            output[pixels] = (
                0.45 * rgb[pixels].astype(np.float32) + 0.55 * colour
            ).astype(np.uint8)

    return output


def segment_hsv_colours(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: HSVParameters | None = None,
) -> HSVMaskResult:
    p = parameters or HSVParameters()
    _validate(p)

    rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    banana_mask = _mask(np.asarray(banana_mask) > 0)

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3).")
    if banana_mask.shape != rgb_image.shape[:2]:
        raise ValueError("The banana mask and image dimensions do not match.")

    banana_area = int(np.count_nonzero(banana_mask))
    if banana_area < 100:
        raise ValueError("The banana region is too small for HSV analysis.")

    # RGB to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    inside = banana_mask > 0

    # Colour segmentation
    dark = (value <= p.dark_value_max) & inside

    green = (
        (hue >= p.green_hue_min)
        & (hue <= p.green_hue_max)
        & (saturation >= p.green_saturation_min)
        & (value >= p.green_value_min)
        & inside
        & ~dark
    )

    yellow = (
        (hue >= p.yellow_hue_min)
        & (hue <= p.yellow_hue_max)
        & (saturation >= p.yellow_saturation_min)
        & (value >= p.yellow_value_min)
        & inside
        & ~dark
    )

    brown = (
        (hue >= p.brown_hue_min)
        & (hue <= p.brown_hue_max)
        & (saturation >= p.brown_saturation_min)
        & (value >= p.brown_value_min)
        & (value <= p.brown_value_max)
        & inside
        & ~dark
    )

    other = inside & ~(green | yellow | brown | dark)

    green_mask = _mask(green)
    yellow_mask = _mask(yellow)
    brown_mask = _mask(brown)
    dark_mask = _mask(dark)
    other_mask = _mask(other)

    green_area = int(np.count_nonzero(green_mask))
    yellow_area = int(np.count_nonzero(yellow_mask))
    brown_area = int(np.count_nonzero(brown_mask))
    dark_area = int(np.count_nonzero(dark_mask))
    other_area = int(np.count_nonzero(other_mask))

    green_pct = _percentage(green_mask, banana_area)
    yellow_pct = _percentage(yellow_mask, banana_area)
    brown_pct = _percentage(brown_mask, banana_area)
    dark_pct = _percentage(dark_mask, banana_area)
    other_pct = _percentage(other_mask, banana_area)

    return HSVMaskResult(
        hsv_image=hsv,
        green_mask=green_mask,
        yellow_mask=yellow_mask,
        brown_mask=brown_mask,
        dark_mask=dark_mask,
        other_mask=other_mask,
        colour_overlay_rgb=_overlay(
            rgb_image, banana_mask, green_mask, yellow_mask, brown_mask, dark_mask
        ),
        banana_area_pixels=banana_area,
        green_area_pixels=green_area,
        yellow_area_pixels=yellow_area,
        brown_area_pixels=brown_area,
        dark_area_pixels=dark_area,
        other_area_pixels=other_area,
        green_percentage=green_pct,
        yellow_percentage=yellow_pct,
        brown_percentage=brown_pct,
        dark_percentage=dark_pct,
        other_percentage=other_pct,
        deteriorated_percentage=brown_pct + dark_pct,
    )