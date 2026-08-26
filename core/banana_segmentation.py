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


def _normalise_content_mask(content_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if content_mask is None:
        return np.full(shape, 255, dtype=np.uint8)
    content_mask = np.asarray(content_mask)
    if content_mask.shape != shape:
        raise ValueError("content_mask must match the image dimensions.")
    return np.where(content_mask > 0, 255, 0).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground object."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes inside the banana mask."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flooded)[1:-1, 1:-1])


def _remove_dark_border_padding(rgb_image: np.ndarray, content_mask: np.ndarray) -> np.ndarray:
    """Discard black, border-connected padding from rotated images."""
    dark_mask = np.where(
        (np.max(rgb_image, axis=2) <= DARK_PADDING_THRESHOLD) & (content_mask > 0), 255, 0
    ).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    if count <= 1:
        return content_mask.copy()

    edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    edge_labels = edge_labels[edge_labels > 0]
    if not edge_labels.size:
        return content_mask.copy()

    padding = np.isin(labels, edge_labels).astype(np.uint8) * 255
    padding = cv2.dilate(padding, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    padding[content_mask == 0] = 0
    content_pixels = max(1, int(np.count_nonzero(content_mask)))
    if np.count_nonzero(padding) / content_pixels > MAXIMUM_PADDING_RATIO:
        return content_mask.copy()

    cleaned = content_mask.copy()
    cleaned[padding > 0] = 0
    return cleaned


def _separate_foreground(rgb_image: np.ndarray, content_mask: np.ndarray) -> np.ndarray:
    """Find the banana as the main colour-distinct object in the usable image."""
    if not np.count_nonzero(content_mask):
        return np.zeros_like(content_mask)
    usable = _remove_dark_border_padding(rgb_image, content_mask)
    if not np.count_nonzero(usable):
        return np.zeros_like(content_mask)

    height, width = usable.shape
    border_width = max(3, int(round(min(height, width) * 0.06)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (border_width * 2 + 1,) * 2)
    border = cv2.subtract(usable, cv2.erode(usable, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0))

    blurred = cv2.GaussianBlur(rgb_image, (5, 5), 0)
    lab_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB).astype(np.float32)
    preferred_border = (border > 0) & (np.max(blurred, axis=2) > DARK_PADDING_THRESHOLD)
    minimum_samples = max(20, int(np.count_nonzero(border) * 0.10))
    border_pixels = lab_image[preferred_border] if np.count_nonzero(preferred_border) >= minimum_samples else lab_image[border > 0]
    if not border_pixels.size:
        border_pixels = lab_image[usable > 0]

    background_colour = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab_image - background_colour, axis=2)
    valid_distances = distance[usable > 0]
    scale_limit = max(1.0, float(np.percentile(valid_distances, 99)))
    distance_image = np.clip(distance * 255.0 / scale_limit, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        distance_image[usable > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(8.0, float(otsu_threshold) * 0.90)
    mask = np.where((distance_image >= threshold) & (usable > 0), 255, 0).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    mask[usable == 0] = 0
    mask = _fill_holes(_largest_component(mask))
    mask[usable == 0] = 0
    return mask


def segment_banana(rgb_image: np.ndarray, content_mask: np.ndarray | None = None) -> SegmentationResult:
    """Extract the main banana with colour distance and mask cleanup."""
    rgb_image = np.asarray(rgb_image)
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3).")
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    rgb_image = np.ascontiguousarray(rgb_image)

    valid_content = _normalise_content_mask(content_mask, rgb_image.shape[:2])
    final_mask = _separate_foreground(rgb_image, valid_content)
    foreground_pixels = int(np.count_nonzero(final_mask))
    area_ratio = foreground_pixels / max(1, int(np.count_nonzero(valid_content)))
    if not foreground_pixels:
        success, message = False, "No foreground object was detected."
    elif area_ratio < MINIMUM_AREA_RATIO:
        success, message = False, "The detected foreground region is too small."
    elif area_ratio > MAXIMUM_AREA_RATIO:
        success, message = False, "The detected foreground covers almost the entire image."
    else:
        success, message = True, "The main foreground region was extracted using classical image processing."

    overlay_rgb = rgb_image.copy()
    contours, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, contours, -1, (0, 255, 0), 2)
    return SegmentationResult(success, message, final_mask, overlay_rgb)
