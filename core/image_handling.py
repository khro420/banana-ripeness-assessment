from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_WORKING_SIZE = (416, 416)  # Width, height


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be used."""


@dataclass
class PreparedImage:
    """Image data shared by the processing approaches."""

    original_pil: Image.Image
    original_rgb: np.ndarray
    working_rgb: np.ndarray
    content_mask: np.ndarray
    metadata: dict[str, Any]
    scale: float
    padding: tuple[int, int, int, int]


def load_uploaded_image(
    uploaded_file: Any,
) -> tuple[Image.Image, dict[str, Any]]:
    """Read an uploaded Streamlit file as an RGB Pillow image."""

    if uploaded_file is None:
        raise ImageValidationError("No image was uploaded.")

    file_size = int(getattr(uploaded_file, "size", 0))

    if file_size <= 0:
        raise ImageValidationError("The uploaded file is empty.")

    if file_size > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            "The image exceeds the 20 MB upload limit."
        )

    try:
        uploaded_file.seek(0)

        with Image.open(uploaded_file) as opened_image:
            image = ImageOps.exif_transpose(opened_image)
            image = image.convert("RGB")
            image.load()

        uploaded_file.seek(0)

    except (UnidentifiedImageError, OSError) as error:
        raise ImageValidationError(
            "The uploaded file is not a valid JPG, JPEG or PNG image."
        ) from error

    if image.width < 50 or image.height < 50:
        raise ImageValidationError(
            "The image is too small. Use an image of at least 50 × 50 pixels."
        )

    metadata = {
        "filename": uploaded_file.name,
        "file_type": getattr(uploaded_file, "type", "Unknown"),
        "file_size_kb": round(file_size / 1024, 1),
        "original_width": image.width,
        "original_height": image.height,
        "colour_mode": image.mode,
    }

    return image, metadata


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image into a contiguous uint8 RGB array."""

    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.uint8)

    return np.ascontiguousarray(array)


def standardise_image(
    image: Image.Image,
    upload_metadata: dict[str, Any] | None = None,
    target_size: tuple[int, int] = DEFAULT_WORKING_SIZE,
    padding_colour: tuple[int, int, int] = (114, 114, 114),
) -> PreparedImage:
    """
    Resize while preserving aspect ratio and add letterbox padding.

    The content mask identifies real image pixels and excludes padding.
    """

    target_width, target_height = target_size

    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive.")

    original_rgb = pil_to_rgb_array(image)
    original_height, original_width = original_rgb.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )

    resized_rgb = cv2.resize(
        original_rgb,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    pad_right = target_width - resized_width - pad_left
    pad_bottom = target_height - resized_height - pad_top

    working_rgb = np.full(
        (target_height, target_width, 3),
        padding_colour,
        dtype=np.uint8,
    )

    working_rgb[
        pad_top:pad_top + resized_height,
        pad_left:pad_left + resized_width,
    ] = resized_rgb

    content_mask = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )

    content_mask[
        pad_top:pad_top + resized_height,
        pad_left:pad_left + resized_width,
    ] = 255

    metadata = dict(upload_metadata or {})
    metadata.update(
        {
            "working_width": target_width,
            "working_height": target_height,
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "pad_right": pad_right,
            "pad_bottom": pad_bottom,
        }
    )

    return PreparedImage(
        original_pil=image.copy(),
        original_rgb=original_rgb,
        working_rgb=working_rgb,
        content_mask=content_mask,
        metadata=metadata,
        scale=scale,
        padding=(
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        ),
    )


def prepare_uploaded_image(
    uploaded_file: Any,
    target_size: tuple[int, int] = DEFAULT_WORKING_SIZE,
) -> PreparedImage:
    """Validate, load and standardise one uploaded image."""

    image, metadata = load_uploaded_image(uploaded_file)

    return standardise_image(
        image=image,
        upload_metadata=metadata,
        target_size=target_size,
    )


def image_fingerprint(uploaded_file: Any) -> str:
    """Identify an upload so stale results are not displayed."""

    return (
        f"{getattr(uploaded_file, 'name', '')}:"
        f"{getattr(uploaded_file, 'size', 0)}:"
        f"{getattr(uploaded_file, 'type', '')}"
    )