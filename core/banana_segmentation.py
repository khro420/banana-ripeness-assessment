from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


MODEL_NAME = "u2net"
MASK_THRESHOLD = 128
MINIMUM_AREA_RATIO = 0.02
MAXIMUM_AREA_RATIO = 0.95


@dataclass
class SegmentationResult:
    """Shared foreground-segmentation output."""

    success: bool
    message: str
    final_mask: np.ndarray
    overlay_rgb: np.ndarray


@lru_cache(maxsize=1)
def _get_rembg():
    """Load the foreground-removal model once."""

    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise RuntimeError(
            "rembg is not installed. Run: "
            "pip install -r requirements.txt"
        ) from exc

    return remove, new_session(MODEL_NAME)


def _largest_component(
    mask: np.ndarray,
) -> np.ndarray:
    count, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if count <= 1:
        return np.zeros_like(mask)

    largest_label = 1 + int(
        np.argmax(
            statistics[1:, cv2.CC_STAT_AREA]
        )
    )

    return np.where(
        labels == largest_label,
        255,
        0,
    ).astype(np.uint8)


def segment_banana(
    rgb_image: np.ndarray,
    content_mask: np.ndarray | None = None,
) -> SegmentationResult:
    """Extract the main foreground object using pretrained U2-Net."""

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

    if content_mask is None:
        valid_content = np.full(
            (height, width),
            255,
            dtype=np.uint8,
        )

    else:
        content_mask = np.asarray(content_mask)

        if content_mask.shape != (height, width):
            raise ValueError(
                "content_mask must match the image dimensions."
            )

        valid_content = np.where(
            content_mask > 0,
            255,
            0,
        ).astype(np.uint8)

    remove, session = _get_rembg()

    model_mask = np.asarray(
        remove(
            rgb_image,
            session=session,
            only_mask=True,
        )
    )

    if model_mask.ndim == 3:
        model_mask = model_mask[:, :, 0]

    if model_mask.shape != (height, width):
        model_mask = cv2.resize(
            model_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    model_mask = np.where(
        model_mask >= MASK_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)

    model_mask[valid_content == 0] = 0

    final_mask = _largest_component(
        model_mask
    )

    foreground_pixels = int(
        np.count_nonzero(final_mask)
    )

    content_pixels = max(
        1,
        int(np.count_nonzero(valid_content)),
    )

    area_ratio = (
        foreground_pixels
        / content_pixels
    )

    if foreground_pixels == 0:
        success = False
        message = "No foreground object was detected."

    elif area_ratio < MINIMUM_AREA_RATIO:
        success = False
        message = (
            "The detected foreground region is too small."
        )

    elif area_ratio > MAXIMUM_AREA_RATIO:
        success = False
        message = (
            "The detected foreground covers almost "
            "the entire image."
        )

    else:
        success = True
        message = (
            "The main foreground region was extracted."
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