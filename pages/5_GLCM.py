"""Streamlit page for GLCM texture analysis."""
from time import perf_counter

import streamlit as st

from branches.glcm.glcm_analysis import (
    analyse_image,
    GLCMQualityBands,
)
from core.banana_segmentation import segment_banana
from core.image_handling import (
    ImageValidationError,
    image_fingerprint,
    prepare_uploaded_image,
)
from ui.components import apply_app_styles, page_header
from ui.multiple_image_mode import render_image_input_mode


GLCM_RESULT_VERSION = 2

# Page setup
apply_app_styles()

page_header(
    "GLCM Texture Analysis",
    "Classify banana ripeness using texture features extracted "
    "from the Gray-Level Co-occurrence Matrix.",
)

# Info about the method
st.info(
    "GLCM measures banana surface texture. "
    "Five features are extracted: Contrast, Dissimilarity, "
    "Homogeneity, Energy and Correlation. "
    "Features are averaged across distances 1 and 2, "
    "and angles 0/45/90/135 degrees. "
    "Ripeness is determined using fixed thresholds "
    "calibrated from the validation dataset."
)

st.warning(
    "Limitation: GLCM focuses on texture rather than colour. "
    "Images with similar surface texture may be hard to distinguish."
)

# Image input
render_image_input_mode("glcm", "glcm")

# Settings info
with st.expander("View GLCM settings"):
    st.markdown(
        """
        The GLCM rule-based model uses:

        - **Gray levels:** 8
        - **Distances:** 1 and 2 pixels
        - **Angles:** 0, 45, 90 and 135 degrees
        - **Features:** Contrast, Dissimilarity, Homogeneity, Energy and Correlation

        The five features are averaged across the selected
        distances and angles before being evaluated against
        the fixed validation-derived thresholds.

        No machine-learning model is used.
        Classification is fully rule-based.
        """
    )

# Upload image
uploaded_file = st.file_uploader(
    "Upload one banana image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="glcm_upload",
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

# Load image
try:
    prepared_image = prepare_uploaded_image(
        uploaded_file,
        target_size=(416, 416),
    )
except ImageValidationError as error:
    st.error(str(error))
    st.stop()

# Show original image
st.image(
    prepared_image.original_pil,
    caption=uploaded_file.name,
    width=420,
)

# Fingerprint for caching
fingerprint = image_fingerprint(uploaded_file)

# Run analysis button
if st.button("Run GLCM analysis", type="primary", use_container_width=True):
    seg_start = perf_counter()
    segmentation = segment_banana(
        rgb_image=prepared_image.working_rgb,
        content_mask=prepared_image.content_mask,
    )
    seg_time_ms = (perf_counter() - seg_start) * 1000.0

    analysis = None
    error_message = None

    if segmentation.success:
        try:
            analysis = analyse_image(
                prepared_image.working_rgb,
                segmentation.final_mask,
                quality_bands=GLCMQualityBands(),
            )
        except (ValueError, AttributeError, FileNotFoundError, TypeError) as error:
            error_message = str(error)
    else:
        error_message = "GLCM analysis stopped because banana segmentation failed."

    st.session_state["glcm_result"] = {
        "result_version": GLCM_RESULT_VERSION,
        "fingerprint": fingerprint,
        "segmentation": segmentation,
        "segmentation_time_ms": seg_time_ms,
        "analysis": analysis,
        "error": error_message,
    }

# Get saved result
saved = st.session_state.get("glcm_result")

# Check if we have a valid result
if (saved is None
        or saved.get("result_version") != GLCM_RESULT_VERSION
        or saved.get("fingerprint") != fingerprint):
    st.caption("Press **Run GLCM analysis** to continue.")
    st.stop()

segmentation = saved["segmentation"]
analysis = saved["analysis"]

# Handle segmentation failure
if not segmentation.success:
    st.error(saved["error"])
    st.image(segmentation.overlay_rgb, caption="Rejected segmentation", use_container_width=True)
    st.stop()

# Handle analysis failure
if analysis is None:
    st.error(saved["error"] or "GLCM analysis did not return a result.")
    st.stop()

# Result header
st.subheader("Analysis Result")

result_columns = st.columns(3)
result_columns[0].metric("Predicted Category", analysis.predicted_category)
result_columns[1].metric("Rule-Based Confidence", f"{analysis.confidence_percent:.2f}%")
result_columns[2].metric("GLCM Analysis Time", f"{analysis.processing_time_ms:.2f} ms")

# Quality assessment (only for Ripe)
st.subheader("Conditional quality assessment")

if analysis.quality_assessed:
    quality_columns = st.columns(2)
    quality_columns[0].metric("Quality class", analysis.predicted_quality)
    quality_columns[1].metric("Quality confidence", f"{analysis.quality_confidence_percent:.2f}%")
    st.success(analysis.quality_reason)
else:
    st.caption("Quality classification is available for Ripe bananas only.")
    st.caption(analysis.quality_reason)

# Feature values
st.subheader("GLCM Feature Values")

feature_columns = st.columns(5)
feature_columns[0].metric("Contrast", f"{analysis.contrast:.6f}")
feature_columns[1].metric("Dissimilarity", f"{analysis.dissimilarity:.6f}")
feature_columns[2].metric("Homogeneity", f"{analysis.homogeneity:.6f}")
feature_columns[3].metric("Energy (ASM)", f"{analysis.energy:.6f}")
feature_columns[4].metric("Correlation", f"{analysis.correlation:.6f}")

# Tabs for more details
overview_tab, texture_tab, details_tab = st.tabs(
    ["Overview", "GLCM texture", "Decision details"],
)

with overview_tab:
    first, second = st.columns(2)

    with first:
        st.markdown("#### Banana segmentation")
        st.image(segmentation.overlay_rgb, use_container_width=True)

    with second:
        st.markdown("#### Quantised grayscale image")
        display_quantised = (analysis.quantised_image * 255 // 7).astype("uint8")
        st.image(display_quantised, clamp=True, use_container_width=True)

    st.caption("The banana segmentation is shared with the other image-processing methods.")
    st.caption(f"Shared banana-segmentation time: {saved['segmentation_time_ms']:.2f} ms")

with texture_tab:
    st.markdown("#### GLCM feature values")
    feature_columns = st.columns(5)
    feature_columns[0].metric("Contrast", f"{analysis.contrast:.6f}")
    feature_columns[1].metric("Dissimilarity", f"{analysis.dissimilarity:.6f}")
    feature_columns[2].metric("Homogeneity", f"{analysis.homogeneity:.6f}")
    feature_columns[3].metric("Energy (ASM)", f"{analysis.energy:.6f}")
    feature_columns[4].metric("Correlation", f"{analysis.correlation:.6f}")

    st.markdown("#### Quantised grayscale")
    display_quantised = (analysis.quantised_image * 255 // 7).astype("uint8")
    st.image(display_quantised, clamp=True, use_container_width=True)
    st.caption("The grayscale image is reduced to 8 gray levels (0-7) before the GLCM is calculated.")

with details_tab:
    st.write(f"**Prediction:** {analysis.predicted_category}")
    st.write(f"**Confidence:** {analysis.confidence_percent:.2f}%")
    st.write(f"**Processing time:** {analysis.processing_time_ms:.2f} ms")

    st.markdown("#### Class decision scores")

    class_order = ["Unripe", "Ripe", "Overripe", "Rotten"]
    score_columns = st.columns(4)

    for index, class_name in enumerate(class_order):
        with score_columns[index]:
            score = analysis.method_result.class_scores.get(class_name, 0.0)
            st.metric(class_name, f"{score * 100:.2f}%")

    st.caption("The class scores show the support for the selected GLCM classification rule.")
