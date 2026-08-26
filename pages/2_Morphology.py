from time import perf_counter

import streamlit as st

from branches.morphology.morphology_analysis import (
    QualityBands,
    RipenessBands,
    analyse_morphology,
)
from branches.morphology.morphology_segmentation import MorphologyParameters
from core.banana_segmentation import segment_banana
from core.image_handling import (
    ImageValidationError,
    image_fingerprint,
    prepare_uploaded_image,
)
from ui.components import apply_app_styles, page_header
from ui.result_display import render_result_summary


MORPHOLOGY_RESULT_VERSION = 10


apply_app_styles()
page_header(
    "Morphological Dark-Region Analysis",
    "Classify ripeness, then grade quality only when the result is Ripe.",
)

parameters = MorphologyParameters()
bands = RipenessBands()
quality_bands = QualityBands()

# The user supplies one image, then starts analysis manually.
uploaded_file = st.file_uploader(
    "Upload one banana image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="morphology_upload",
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    prepared_image = prepare_uploaded_image(
        uploaded_file,
        target_size=(416, 416),
    )
except ImageValidationError as error:
    st.error(str(error))
    st.stop()

st.image(prepared_image.original_pil, caption=uploaded_file.name, width=420)
fingerprint = image_fingerprint(uploaded_file)

# Keep the button so results are only calculated when requested.
if st.button("Run morphology analysis", type="primary", width="stretch"):
    segmentation_start = perf_counter()
    segmentation = segment_banana(
        rgb_image=prepared_image.working_rgb,
        content_mask=prepared_image.content_mask,
    )
    segmentation_time_ms = (perf_counter() - segmentation_start) * 1000.0

    analysis = None
    error_message = None
    if segmentation.success:
        try:
            analysis = analyse_morphology(
                rgb_image=prepared_image.working_rgb,
                banana_mask=segmentation.final_mask,
                parameters=parameters,
                bands=bands,
                quality_bands=quality_bands,
            )
        except ValueError as error:
            error_message = str(error)
    else:
        error_message = "Morphology analysis stopped because banana segmentation failed."

    st.session_state["morphology_result"] = {
        "result_version": MORPHOLOGY_RESULT_VERSION,
        "fingerprint": fingerprint,
        "segmentation": segmentation,
        "segmentation_time_ms": segmentation_time_ms,
        "analysis": analysis,
        "error": error_message,
    }

saved = st.session_state.get("morphology_result")
# Avoid showing an old result after the user uploads a different image.
if (
    saved is None
    or saved.get("result_version") != MORPHOLOGY_RESULT_VERSION
    or saved.get("fingerprint") != fingerprint
):
    st.caption("Press **Run morphology analysis** to continue.")
    st.stop()

segmentation = saved["segmentation"]
analysis = saved["analysis"]

if not segmentation.success:
    st.error(saved["error"])
    st.image(
        segmentation.overlay_rgb,
        caption="Rejected segmentation",
        width="stretch",
    )
    st.stop()

if analysis is None:
    st.error(saved["error"] or "Morphology analysis did not return a result.")
    st.stop()

render_result_summary(analysis.method_result)

# Quality is only relevant when the ripeness result is Ripe.
st.subheader("Conditional quality assessment")
if analysis.quality_assessed:
    quality_columns = st.columns(2)
    quality_columns[0].metric("Quality class", analysis.predicted_quality)
    quality_columns[1].metric(
        "Quality confidence",
        f"{analysis.quality_confidence_percent:.2f}%",
    )
else:
    st.caption("Quality classification is available for Ripe bananas only.")

metrics = st.columns(4)
metrics[0].metric("Total dark area", f"{analysis.total_dark_percentage:.2f}%")
metrics[1].metric(
    "Largest dark patch",
    f"{analysis.largest_dark_patch_percentage:.2f}%",
)
metrics[2].metric("Dark-region spread", f"{analysis.dark_region_spread:.2f}%")
metrics[3].metric("Dark components", analysis.dark_component_count)

st.subheader("Analysis visualisation")
# These two images briefly show the main image-processing steps.
first, second = st.columns(2)
with first:
    st.image(
        segmentation.overlay_rgb,
        caption="Banana segmentation",
        width="stretch",
    )
with second:
    st.image(
        analysis.masks.blemish_overlay_rgb,
        caption="Detected dark regions",
        width="stretch",
    )
