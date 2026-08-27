"""Bulk input handling and shared hybrid-analysis workflow."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import PurePosixPath
from time import perf_counter
from typing import Any, Callable, Iterable
from zipfile import BadZipFile, ZipFile

from branches.glcm.glcm_analysis import (
    GLCMParameters,
    GLCMRipenessBands,
    analyse_glcm,
)
from branches.hsv.hsv_analysis import HSVRipenessBands, analyse_hsv
from branches.hsv.hsv_segmentation import HSVParameters
from branches.kmeans.kmeans_analysis import KMeansRipenessBands, analyse_kmeans
from branches.kmeans.kmeans_segmentation import KMeansParameters
from branches.morphology.morphology_analysis import RipenessBands, analyse_morphology
from branches.morphology.morphology_segmentation import MorphologyParameters
from core.banana_segmentation import segment_banana
from core.hybrid import METHOD_NAMES, combine_method_results
from core.image_handling import ImageValidationError, prepare_uploaded_image


ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_BATCH_IMAGES = 100
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_EXPANDED_ARCHIVE_BYTES = 250 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1_000
QUALITY_ORDER = ("Class_A", "Class_B", "Defect")
APPROACH_NAMES = {
    "morphology": "Morphology",
    "hsv": "HSV",
    "kmeans": "K-means",
    "glcm": "GLCM Texture",
    "hybrid": "Hybrid",
}


class BulkInputError(ValueError):
    """Raised when a bulk input cannot be processed safely."""


class InMemoryUpload(BytesIO):
    """Small UploadedFile-compatible wrapper for images read from a ZIP file."""

    def __init__(self, data: bytes, name: str, media_type: str) -> None:
        super().__init__(data)
        self.name = name
        self.type = media_type
        self.size = len(data)


def _media_type(filename: str) -> str:
    suffix = PurePosixPath(filename).suffix.lower()
    return "image/png" if suffix == ".png" else "image/jpeg"


def validate_image_selection(uploaded_files: Iterable[Any] | None) -> list[Any]:
    """Validate a list returned by Streamlit's multi-image uploader."""

    files = list(uploaded_files or [])
    if not files:
        return []
    if len(files) > MAX_BATCH_IMAGES:
        raise BulkInputError(
            f"A maximum of {MAX_BATCH_IMAGES} images can be analysed in one batch."
        )
    return files


def load_images_from_zip(uploaded_archive: Any) -> list[InMemoryUpload]:
    """Read JPG, JPEG and PNG files from a folder supplied as a ZIP archive."""

    if uploaded_archive is None:
        return []

    archive_size = int(getattr(uploaded_archive, "size", 0))
    if archive_size <= 0:
        raise BulkInputError("The uploaded ZIP file is empty.")
    if archive_size > MAX_ARCHIVE_BYTES:
        raise BulkInputError("The ZIP file exceeds the 200 MB upload limit.")

    try:
        uploaded_archive.seek(0)
        with ZipFile(uploaded_archive) as archive:
            candidates = [
                item
                for item in archive.infolist()
                if not item.is_dir()
                and PurePosixPath(item.filename).suffix.lower() in ALLOWED_IMAGE_SUFFIXES
                and "__MACOSX" not in PurePosixPath(item.filename).parts
                and not PurePosixPath(item.filename).name.startswith(".")
            ]

            if len(candidates) > MAX_BATCH_IMAGES:
                raise BulkInputError(
                    f"The ZIP file contains more than {MAX_BATCH_IMAGES} supported images."
                )
            if not candidates:
                raise BulkInputError("The ZIP file contains no JPG, JPEG or PNG images.")

            expanded_size = sum(item.file_size for item in candidates)
            if expanded_size > MAX_EXPANDED_ARCHIVE_BYTES:
                raise BulkInputError("The images expand beyond the 250 MB batch limit.")

            uploads: list[InMemoryUpload] = []
            for item in sorted(candidates, key=lambda value: value.filename.casefold()):
                if item.flag_bits & 0x1:
                    raise BulkInputError("Password-protected ZIP files are not supported.")
                if item.compress_size == 0 and item.file_size > 0:
                    raise BulkInputError("The ZIP file contains an invalid compressed entry.")
                if item.compress_size and item.file_size / item.compress_size > MAX_COMPRESSION_RATIO:
                    raise BulkInputError("The ZIP file has an unsafe compression ratio.")

                data = archive.read(item)
                uploads.append(
                    InMemoryUpload(
                        data=data,
                        name=item.filename,
                        media_type=_media_type(item.filename),
                    )
                )

        uploaded_archive.seek(0)
        return uploads
    except BulkInputError:
        raise
    except BadZipFile as error:
        raise BulkInputError("The uploaded file is not a valid ZIP archive.") from error
    except (OSError, RuntimeError) as error:
        raise BulkInputError(f"The ZIP file could not be read: {error}") from error


def batch_fingerprint(files: Iterable[Any]) -> str:
    """Identify the selected batch so stale results are not displayed."""

    return "|".join(
        f"{getattr(item, 'name', '')}:{getattr(item, 'size', 0)}:"
        f"{getattr(item, 'type', '')}"
        for item in files
    )


def _quality_consensus(analyses: dict[str, Any], ripeness: str) -> tuple[str, str]:
    """Summarise the available ripe-only quality votes without inventing a score."""

    if ripeness != "Ripe":
        return "Not assessed", "Ripeness was not Ripe."

    votes = [
        analysis.predicted_quality
        for analysis in analyses.values()
        if getattr(analysis, "quality_assessed", False)
        and getattr(analysis, "predicted_quality", None) in QUALITY_ORDER
    ]
    if not votes:
        return "Not available", "No branch produced a ripe-only quality result."

    counts = Counter(votes)
    highest = max(counts.values())
    quality = next(category for category in QUALITY_ORDER if counts[category] == highest)
    vote_text = ", ".join(f"{category}: {counts[category]}" for category in QUALITY_ORDER)
    return quality, vote_text


def _single_quality(analysis: Any, ripeness: str, approach: str) -> tuple[str, str]:
    """Return the selected branch's own ripe-only quality result."""

    approach_name = APPROACH_NAMES[approach]
    if ripeness != "Ripe":
        return "Not assessed", "Ripeness was not Ripe."
    if not getattr(analysis, "quality_assessed", False):
        return "Not available", f"{approach_name} does not provide a quality result."

    quality = getattr(analysis, "predicted_quality", None)
    if quality not in QUALITY_ORDER:
        return "Not available", f"{approach_name} did not return a valid quality class."
    return quality, f"{approach_name}: {quality}"


def _failed_row(
    filename: str,
    message: str,
    processing_time_ms: float,
    approach: str,
) -> dict[str, Any]:
    return {
        "file_name": filename,
        "analysis_approach": APPROACH_NAMES[approach],
        "status": "Failed",
        "predicted_category": None,
        "confidence_percent": None,
        "quality_class": "Not assessed",
        "quality_votes": "",
        "visible_blemish_percent": None,
        "method_agreement": None,
        "branch_predictions": "",
        "processing_time_ms": round(processing_time_ms, 2),
        "decision_reason": "",
        "error": message,
    }


def _method_jobs() -> dict[str, tuple[Any, Any, Any]]:
    """Construct fresh parameters for every independent analysis run."""

    return {
        "morphology": (analyse_morphology, MorphologyParameters(), RipenessBands()),
        "hsv": (analyse_hsv, HSVParameters(), HSVRipenessBands()),
        "kmeans": (analyse_kmeans, KMeansParameters(k=4), KMeansRipenessBands()),
        "glcm": (analyse_glcm, GLCMParameters(), GLCMRipenessBands()),
    }


def analyse_one_upload(uploaded_file: Any, approach: str = "hybrid") -> dict[str, Any]:
    """Run the selected branch or the four-branch hybrid for one image."""

    if approach not in APPROACH_NAMES:
        raise BulkInputError(f"Unknown analysis approach: {approach}.")

    started = perf_counter()
    filename = str(getattr(uploaded_file, "name", "Unnamed image"))

    try:
        prepared = prepare_uploaded_image(uploaded_file, target_size=(416, 416))
        segmentation = segment_banana(
            rgb_image=prepared.working_rgb,
            content_mask=prepared.content_mask,
        )
        if not segmentation.success:
            return _failed_row(
                filename,
                segmentation.message,
                (perf_counter() - started) * 1_000.0,
                approach,
            )

        jobs = _method_jobs()
        requested_methods = tuple(METHOD_NAMES) if approach == "hybrid" else (approach,)
        analyses: dict[str, Any] = {}
        branch_errors: dict[str, str] = {}

        for method_key in requested_methods:
            analyser, parameters, bands = jobs[method_key]
            try:
                analyses[method_key] = analyser(
                    rgb_image=prepared.working_rgb,
                    banana_mask=segmentation.final_mask,
                    parameters=parameters,
                    bands=bands,
                )
            except Exception as error:  # Continue when an independent branch fails.
                branch_errors[method_key] = str(error)

        if approach == "hybrid":
            hybrid = combine_method_results(
                [analysis.method_result for analysis in analyses.values()]
            )
            predicted_category = hybrid.predicted_category
            confidence_percent = hybrid.confidence_percent
            quality, quality_votes = _quality_consensus(analyses, predicted_category)
            method_agreement = f"{hybrid.agreement_count}/{hybrid.methods_used}"
            decision_reason = hybrid.decision_reason
            branch_predictions = "; ".join(
                f"{METHOD_NAMES[key]}: {analysis.predicted_category}"
                for key, analysis in analyses.items()
            )
            if branch_errors:
                branch_predictions += "; excluded - " + ", ".join(
                    METHOD_NAMES.get(key, key) for key in branch_errors
                )
        else:
            if approach not in analyses:
                message = branch_errors.get(
                    approach,
                    f"{APPROACH_NAMES[approach]} did not return a result.",
                )
                return _failed_row(
                    filename,
                    message,
                    (perf_counter() - started) * 1_000.0,
                    approach,
                )

            analysis = analyses[approach]
            method_result = analysis.method_result
            predicted_category = str(method_result.predicted_category)
            confidence_percent = float(method_result.confidence_percent or 0.0)
            quality, quality_votes = _single_quality(
                analysis,
                predicted_category,
                approach,
            )
            method_agreement = "Not applicable"
            branch_predictions = f"{APPROACH_NAMES[approach]}: {predicted_category}"
            decision_reason = str(
                getattr(analysis, "decision_reason", "")
                or method_result.features.get("Decision reason", "")
                or (
                    f"{APPROACH_NAMES[approach]} classified the image as "
                    f"{predicted_category} using its calibrated decision rules."
                )
            )

        morphology = analyses.get("morphology")

        return {
            "file_name": filename,
            "analysis_approach": APPROACH_NAMES[approach],
            "status": "Completed",
            "predicted_category": predicted_category,
            "confidence_percent": round(confidence_percent, 2),
            "quality_class": quality,
            "quality_votes": quality_votes,
            "visible_blemish_percent": (
                None
                if morphology is None
                else round(float(morphology.total_dark_percentage), 2)
            ),
            "method_agreement": method_agreement,
            "branch_predictions": branch_predictions,
            "processing_time_ms": round((perf_counter() - started) * 1_000.0, 2),
            "decision_reason": decision_reason,
            "error": "",
        }
    except ImageValidationError as error:
        return _failed_row(
            filename,
            str(error),
            (perf_counter() - started) * 1_000.0,
            approach,
        )
    except BulkInputError:
        raise
    except Exception as error:
        return _failed_row(
            filename,
            f"Analysis failed unexpectedly: {error}",
            (perf_counter() - started) * 1_000.0,
            approach,
        )


def analyse_batch(
    uploaded_files: Iterable[Any],
    approach: str = "hybrid",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Analyse a bounded batch while keeping per-image failures isolated."""

    if approach not in APPROACH_NAMES:
        raise BulkInputError(f"Unknown analysis approach: {approach}.")
    files = validate_image_selection(uploaded_files)
    if not files:
        raise BulkInputError("Select at least one image before starting the analysis.")

    results = []
    total = len(files)
    for index, uploaded_file in enumerate(files, start=1):
        filename = str(getattr(uploaded_file, "name", f"Image {index}"))
        results.append(analyse_one_upload(uploaded_file, approach=approach))
        if progress_callback is not None:
            progress_callback(index, total, filename)
    return results
