from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from core.result_schema import MethodResult


CATEGORIES = (
    "Unripe",
    "Ripe",
    "Overripe",
    "Rotten",
)


@dataclass(frozen=True)
class GLCMParameters:
    # Number of gray levels used for GLCM
    levels: int = 8

    # Pixel distances
    distances: tuple[int, ...] = (1, 2)

    # 0, 45, 90 and 135 degrees
    angles: tuple[float, ...] = (
        0.0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
    )


@dataclass(frozen=True)
class GLCMRipenessBands:
    """
    Frozen thresholds obtained from the validation dataset.
    """

    unripe_max_contrast: float = 0.116461
    ripe_max_contrast: float = 0.161108
    overripe_min_contrast: float = 0.161108
    rotten_min_contrast: float = 0.212873


@dataclass
class GLCMAnalysisResult:
    # Result used by the main application
    method_result: MethodResult

    # Prediction
    predicted_category: str
    confidence_percent: float

    # Processing time
    processing_time_ms: float

    # GLCM features
    contrast: float
    homogeneity: float
    energy: float
    correlation: float

    # Explanation of the classification
    decision_reason: str

    # Quantised image for display
    quantised_image: np.ndarray


def validate_parameters(
    parameters: GLCMParameters,
) -> None:

    if parameters.levels < 2:
        raise ValueError(
            "Number of GLCM gray levels must be at least 2."
        )

    if not parameters.distances:
        raise ValueError(
            "At least one GLCM distance is required."
        )

    if not parameters.angles:
        raise ValueError(
            "At least one GLCM angle is required."
        )

    for distance in parameters.distances:
        if distance < 1:
            raise ValueError(
                "GLCM distance must be at least 1."
            )


def quantise_image(
    greyscale: np.ndarray,
    mask: np.ndarray,
    levels: int,
) -> np.ndarray:

    quantised = np.floor(
        greyscale.astype(np.float32)
        * levels
        / 256.0
    ).astype(np.uint8)

    quantised = np.clip(
        quantised,
        0,
        levels - 1,
    )

    # Remove background
    quantised[mask == 0] = 0

    return quantised


def build_glcm(
    quantised: np.ndarray,
    mask: np.ndarray,
    levels: int,
    distance: int,
    angle: float,
) -> np.ndarray:

    height, width = quantised.shape

    dx = int(
        round(
            np.cos(angle) * distance
        )
    )

    dy = int(
        round(
            np.sin(angle) * distance
        )
    )

    if dx == 0 and dy == 0:
        dx = distance

    glcm = np.zeros(
        (levels, levels),
        dtype=np.float64,
    )

    for y in range(height):
        for x in range(width):

            neighbour_x = x + dx
            neighbour_y = y + dy

            if (
                neighbour_x < 0
                or neighbour_x >= width
                or neighbour_y < 0
                or neighbour_y >= height
            ):
                continue

            if mask[y, x] == 0:
                continue

            if mask[
                neighbour_y,
                neighbour_x
            ] == 0:
                continue

            current_level = int(
                quantised[y, x]
            )

            neighbour_level = int(
                quantised[
                    neighbour_y,
                    neighbour_x
                ]
            )

            glcm[
                current_level,
                neighbour_level
            ] += 1

            # Make the GLCM symmetric
            glcm[
                neighbour_level,
                current_level
            ] += 1

    total = glcm.sum()

    if total > 0:
        glcm = glcm / total

    return glcm


def calculate_glcm_features(
    glcm: np.ndarray,
) -> tuple[float, float, float, float]:

    levels = glcm.shape[0]

    gray_levels = np.arange(
        levels
    )

    i, j = np.meshgrid(
        gray_levels,
        gray_levels,
        indexing="ij",
    )

    # Contrast
    contrast = np.sum(
        ((i - j) ** 2) * glcm
    )

    # Homogeneity
    homogeneity = np.sum(
        glcm
        / (
            1.0
            + np.abs(i - j)
        )
    )

    # Energy / ASM
    energy = np.sum(
        glcm ** 2
    )

    # Correlation
    px = glcm.sum(
        axis=1
    )

    py = glcm.sum(
        axis=0
    )

    mean_x = np.sum(
        gray_levels * px
    )

    mean_y = np.sum(
        gray_levels * py
    )

    std_x = np.sqrt(
        np.sum(
            (
                gray_levels
                - mean_x
            ) ** 2
            * px
        )
    )

    std_y = np.sqrt(
        np.sum(
            (
                gray_levels
                - mean_y
            ) ** 2
            * py
        )
    )

    if std_x == 0 or std_y == 0:
        correlation = 1.0

    else:
        correlation = np.sum(
            (
                (i - mean_x)
                * (j - mean_y)
                * glcm
            )
        ) / (
            std_x * std_y
        )

    return (
        float(contrast),
        float(homogeneity),
        float(energy),
        float(correlation),
    )


def extract_glcm_features(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: GLCMParameters | None = None,
):

    parameters = (
        parameters
        or GLCMParameters()
    )

    validate_parameters(
        parameters
    )

    rgb_image = np.asarray(
        rgb_image,
        dtype=np.uint8,
    )

    banana_mask = np.asarray(
        banana_mask
    )

    if rgb_image.ndim != 3:
        raise ValueError(
            "rgb_image must have shape "
            "(height, width, 3)."
        )

    if rgb_image.shape[2] != 3:
        raise ValueError(
            "rgb_image must contain 3 channels."
        )

    if banana_mask.shape != rgb_image.shape[:2]:
        raise ValueError(
            "Banana mask and image dimensions "
            "do not match."
        )

    # Convert RGB to grayscale
    grayscale = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY,
    )

    # Quantise grayscale image
    quantised = quantise_image(
        grayscale,
        banana_mask,
        parameters.levels,
    )

    feature_values = []

    # Create GLCMs for all distances and angles
    for distance in parameters.distances:

        for angle in parameters.angles:

            glcm = build_glcm(
                quantised=quantised,
                mask=banana_mask,
                levels=parameters.levels,
                distance=distance,
                angle=angle,
            )

            if glcm.sum() == 0:
                continue

            features = calculate_glcm_features(
                glcm
            )

            feature_values.append(
                features
            )

    if not feature_values:
        raise ValueError(
            "Unable to construct a valid GLCM."
        )

    feature_values = np.asarray(
        feature_values,
        dtype=np.float64,
    )

    # Average the features from all GLCMs
    final_features = np.mean(
        feature_values,
        axis=0,
    )

    return (
        final_features,
        quantised,
    )


def _classify(
    contrast: float,
    bands: GLCMRipenessBands,
) -> tuple[str, str]:

    if contrast < bands.unripe_max_contrast:

        return (
            "Unripe",
            (
                f"Contrast is {contrast:.6f}, "
                f"below the Unripe threshold "
                f"of {bands.unripe_max_contrast:.6f}."
            ),
        )

    if contrast < bands.ripe_max_contrast:

        return (
            "Ripe",
            (
                f"Contrast is {contrast:.6f}, "
                f"within the Ripe range."
            ),
        )

    if contrast < bands.rotten_min_contrast:

        return (
            "Overripe",
            (
                f"Contrast is {contrast:.6f}, "
                f"within the Overripe range."
            ),
        )

    return (
        "Rotten",
        (
            f"Contrast is {contrast:.6f}, "
            f"above the Rotten threshold "
            f"of {bands.rotten_min_contrast:.6f}."
        ),
    )


def _rule_confidence(
    category: str,
    contrast: float,
    bands: GLCMRipenessBands,
) -> float:

    if category == "Unripe":

        distance = abs(
            bands.unripe_max_contrast
            - contrast
        )

    elif category == "Ripe":

        distance = min(
            abs(
                contrast
                - bands.unripe_max_contrast
            ),
            abs(
                bands.ripe_max_contrast
                - contrast
            ),
        )

    elif category == "Overripe":

        distance = min(
            abs(
                contrast
                - bands.ripe_max_contrast
            ),
            abs(
                bands.rotten_min_contrast
                - contrast
            ),
        )

    else:

        distance = abs(
            contrast
            - bands.rotten_min_contrast
        )

    # This is a rule-support score, not a probability
    confidence = 50.0 + min(
        distance * 100.0,
        45.0,
    )

    return float(
        np.clip(
            confidence,
            50.0,
            95.0,
        )
    )


def analyse_image(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: GLCMParameters | None = None,
    bands: GLCMRipenessBands | None = None,
) -> GLCMAnalysisResult:

    start_time = perf_counter()

    parameters = (
        parameters
        or GLCMParameters()
    )

    bands = (
        bands
        or GLCMRipenessBands()
    )

    features, quantised = extract_glcm_features(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
    )

    contrast = float(features[0])
    homogeneity = float(features[1])
    energy = float(features[2])
    correlation = float(features[3])

    category, reason = _classify(
        contrast=contrast,
        bands=bands,
    )

    confidence = _rule_confidence(
        category=category,
        contrast=contrast,
        bands=bands,
    )

    processing_time_ms = (
        perf_counter() - start_time
    ) * 1000.0

    class_scores = {
        name: 0.0
        for name in CATEGORIES
    }

    class_scores[category] = (
        confidence / 100.0
    )

    feature_dict = {
        "Contrast": round(
            contrast,
            6,
        ),
        "Homogeneity": round(
            homogeneity,
            6,
        ),
        "Energy (ASM)": round(
            energy,
            6,
        ),
        "Correlation": round(
            correlation,
            6,
        ),
        "Decision reason": reason,
    }

    method_result = MethodResult(
        method_key="glcm",
        method_name="GLCM - Texture Analysis",
        predicted_category=category,
        confidence_percent=round(
            confidence,
            2,
        ),
        processing_time_ms=round(
            processing_time_ms,
            2,
        ),
        class_scores=class_scores,
        features=feature_dict,
        notes=[
            "GLCM features are calculated from the segmented banana.",
            "Four texture features are extracted.",
            "Classification uses fixed thresholds from the validation set.",
            "The thresholds are kept unchanged during test evaluation.",
            "Confidence shows rule support, not model probability.",
        ],
        is_placeholder=False,
    )

    return GLCMAnalysisResult(
        method_result=method_result,
        predicted_category=category,
        confidence_percent=confidence,
        processing_time_ms=processing_time_ms,
        contrast=contrast,
        homogeneity=homogeneity,
        energy=energy,
        correlation=correlation,
        decision_reason=reason,
        quantised_image=quantised,
    )


def analyze_image(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: GLCMParameters | None = None,
    bands: GLCMRipenessBands | None = None,
) -> GLCMAnalysisResult:

    return analyse_image(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
        bands=bands,
    )


def analyse_glcm(
    rgb_image: np.ndarray,
    banana_mask: np.ndarray,
    parameters: GLCMParameters | None = None,
    bands: GLCMRipenessBands | None = None,
) -> GLCMAnalysisResult:

    return analyse_image(
        rgb_image=rgb_image,
        banana_mask=banana_mask,
        parameters=parameters,
        bands=bands,
    )


def prepare_quantised_for_display(
    quantised_image: np.ndarray,
) -> np.ndarray:

    return (
        quantised_image.astype(
            np.float32
        )
        * 255.0
        / 7.0
    ).astype(
        np.uint8
    )