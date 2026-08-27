from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.bulk_analysis import (
    APPROACH_NAMES,
    BulkInputError,
    analyse_batch,
    batch_fingerprint,
    load_images_from_zip,
    validate_image_selection,
)


RESULT_VERSION = 1
APPROACH_DESCRIPTIONS = {
    "morphology": "Each image uses dark-region morphology, including blemish area and ripe-only quality grading.",
    "hsv": "Each image uses HSV colour segmentation, including ripe-only quality grading.",
    "kmeans": "Each image uses K-means colour clustering, including ripe-only quality grading.",
    "glcm": "Each image uses GLCM texture analysis. GLCM provides ripeness but not a quality class.",
    "hybrid": "Each image uses class-specific fusion of Morphology, HSV, K-means and GLCM Texture.",
}


def _display_columns(approach: str) -> dict[str, str]:
    columns = {
        "file_name": "File",
        "status": "Status",
        "predicted_category": "Ripeness",
        "confidence_percent": "Confidence (%)",
        "quality_class": "Quality",
    }
    if approach in {"morphology", "hybrid"}:
        columns["visible_blemish_percent"] = "Visible blemish area (%)"
    if approach == "hybrid":
        columns["method_agreement"] = "Method agreement"
    columns.update(
        {
            "processing_time_ms": "Processing time (ms)",
            "error": "Error",
        }
    )
    return columns


def _load_multiple_files(key_prefix: str) -> tuple[list[object], str]:
    source_mode = st.radio(
        "Multiple-image source",
        options=("Select multiple images", "Upload an image folder as ZIP"),
        horizontal=True,
        key=f"{key_prefix}_multiple_source",
    )

    try:
        if source_mode == "Select multiple images":
            uploads = st.file_uploader(
                "Select up to 100 banana images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"{key_prefix}_multiple_upload",
            )
            files = validate_image_selection(uploads)
        else:
            archive = st.file_uploader(
                "Upload one ZIP file containing JPG, JPEG or PNG images",
                type=["zip"],
                accept_multiple_files=False,
                key=f"{key_prefix}_folder_upload",
            )
            files = load_images_from_zip(archive) if archive else []
    except BulkInputError as error:
        st.error(str(error))
        st.stop()

    return files, source_mode


def _render_summary(results: list[dict], approach: str) -> None:
    completed = [row for row in results if row["status"] == "Completed"]
    failed = [row for row in results if row["status"] != "Completed"]
    average_time = (
        sum(row["processing_time_ms"] for row in completed) / len(completed)
        if completed
        else 0.0
    )

    st.divider()
    st.subheader(f"{APPROACH_NAMES[approach]} multiple-image results")
    metrics = st.columns(4)
    metrics[0].metric("Images submitted", len(results))
    metrics[1].metric("Completed", len(completed))
    metrics[2].metric("Failed", len(failed))
    metrics[3].metric("Mean processing time", f"{average_time:.2f} ms")

    summary_tab, results_tab = st.tabs(["Summary", "Image results"])

    with summary_tab:
        if completed:
            order = ("Unripe", "Ripe", "Overripe", "Rotten")
            counts = Counter(row["predicted_category"] for row in completed)
            distribution = pd.DataFrame(
                {"Ripeness": order, "Images": [counts[name] for name in order]}
            ).set_index("Ripeness")
            st.markdown("#### Ripeness distribution")
            st.bar_chart(distribution, color="#2E7D32")

            predominant = max(counts, key=lambda name: counts[name])
            confidences = [float(row["confidence_percent"]) for row in completed]
            st.markdown("#### Analysis findings")
            st.write(
                f"- {len(completed)} of {len(results)} images completed successfully.\n"
                f"- **{predominant}** was the most frequent ripeness result "
                f"({counts[predominant]} image(s)).\n"
                f"- Mean {APPROACH_NAMES[approach]} rule-support confidence was "
                f"**{sum(confidences) / len(confidences):.2f}%**."
            )

            quality_order = ("Class_A", "Class_B", "Defect")
            quality_counts = Counter(
                row["quality_class"]
                for row in completed
                if row["quality_class"] in quality_order
            )
            if quality_counts:
                quality_frame = pd.DataFrame(
                    {
                        "Quality": quality_order,
                        "Images": [quality_counts[name] for name in quality_order],
                    }
                ).set_index("Quality")
                st.markdown("#### Ripe-only quality distribution")
                st.bar_chart(quality_frame, color="#F9A825")
        else:
            st.warning("No image completed successfully. Review the failure messages.")

    with results_tab:
        display_columns = _display_columns(approach)
        frame = pd.DataFrame(results)
        st.dataframe(
            frame[list(display_columns)].rename(columns=display_columns),
            hide_index=True,
            width="stretch",
            column_config={
                "Confidence (%)": st.column_config.NumberColumn(format="%.2f"),
                "Visible blemish area (%)": st.column_config.NumberColumn(format="%.2f"),
                "Processing time (ms)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        if failed:
            with st.expander("Failure details", expanded=True):
                for row in failed:
                    st.error(f"{row['file_name']}: {row['error']}")


def render_image_input_mode(approach: str, key_prefix: str) -> None:
    """Render the shared selector; return only when single-image mode is chosen."""

    if approach not in APPROACH_NAMES:
        raise ValueError(f"Unknown analysis approach: {approach}.")

    mode = st.radio(
        "Image input mode",
        options=("Upload one image", "Upload multiple images"),
        horizontal=True,
        key=f"{key_prefix}_image_input_mode",
    )
    if mode == "Upload one image":
        return

    st.info(APPROACH_DESCRIPTIONS[approach])
    files, source_mode = _load_multiple_files(key_prefix)
    if not files:
        st.caption("Add images to prepare the multiple-image analysis.")
        st.stop()

    selection = pd.DataFrame(
        {
            "File": [getattr(item, "name", "Unnamed image") for item in files],
            "Size (KB)": [round(getattr(item, "size", 0) / 1024, 1) for item in files],
        }
    )
    st.success(f"{len(files)} image(s) ready for {APPROACH_NAMES[approach]} analysis.")
    with st.expander("Review selected images", expanded=len(files) <= 10):
        st.dataframe(selection, hide_index=True, width="stretch")

    fingerprint = f"{approach}|{source_mode}|{batch_fingerprint(files)}"
    state_key = f"{key_prefix}_multiple_image_result"
    button_label = f"Run {APPROACH_NAMES[approach]} on all images"

    if st.button(button_label, type="primary", width="stretch", key=f"{key_prefix}_multiple_run"):
        progress = st.progress(0)
        message = st.empty()

        def update_progress(current: int, total: int, filename: str) -> None:
            progress.progress(int(current / max(total, 1) * 100))
            message.caption(f"Processed {current} of {total}: {filename}")

        try:
            results = analyse_batch(
                files,
                approach=approach,
                progress_callback=update_progress,
            )
            st.session_state[state_key] = {
                "version": RESULT_VERSION,
                "fingerprint": fingerprint,
                "generated_at": datetime.now(timezone.utc),
                "results": results,
            }
            progress.progress(100)
            message.success("Multiple-image analysis completed.")
        except BulkInputError as error:
            message.empty()
            st.error(str(error))
        except Exception as error:
            message.empty()
            st.error(f"Multiple-image analysis stopped unexpectedly: {error}")

    saved = st.session_state.get(state_key)
    if (
        saved is None
        or saved.get("version") != RESULT_VERSION
        or saved.get("fingerprint") != fingerprint
    ):
        st.caption(f"Press **{button_label}** to continue.")
        st.stop()

    _render_summary(saved["results"], approach)
    st.stop()
