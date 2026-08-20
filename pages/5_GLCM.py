from time import perf_counter

import streamlit as st

from branches.glcm.glcm_analysis import analyse_image
from core.banana_segmentation import segment_banana
from core.image_handling import (
    ImageValidationError,
    image_fingerprint,
    prepare_uploaded_image,
)
from ui.components import apply_app_styles, page_header


# ============================================================
# GLCM RESULT VERSION
# ============================================================

GLCM_RESULT_VERSION = 2


# ============================================================
# PAGE STYLE
# ============================================================

apply_app_styles()

page_header(
    "GLCM Texture Analysis",
    "Classify banana ripeness using texture features extracted "
    "from the Gray-Level Co-occurrence Matrix.",
)


# ============================================================
# INFORMATION
# ============================================================

st.info(
    "GLCM analysis measures the texture of the banana surface. "
    "The four texture features extracted are "
    "Contrast, Homogeneity, Energy and Correlation."
    "The features are averaged across the selected"
    "distances and angles. The final ripeness category"
    "is determined using fixed Contrast thresholds"
    "calibrated from the validation dataset."
)

st.warning(
    "Limitation: GLCM focuses on texture rather than colour. "
    "Images with similar surface texture may therefore be "
    "difficult to distinguish between ripeness categories."
)


# ============================================================
# GLCM INFORMATION
# ============================================================

with st.expander("View GLCM settings"):

    st.markdown(
        """
        The trained GLCM model uses:

        - **Gray levels:** 8
        - **Distances:** 1 and 2 pixels
        - **Angles:** 0°, 45°, 90° and 135°
        - **Features:** Contrast, Homogeneity, Energy and Correlation

        The four features are averaged across the selected
        distances and angles before being given to the
        trained Random Forest model.
        """
    )

    st.caption(
        "The GLCM settings are kept the same as the settings "
        "used during model training."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload one banana image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="glcm_upload",
)


# ============================================================
# WAIT FOR IMAGE
# ============================================================

if uploaded_file is None:

    st.info("Upload an image to begin.")

    st.stop()


# ============================================================
# PREPARE IMAGE
# ============================================================

try:

    prepared_image = prepare_uploaded_image(
        uploaded_file,
        target_size=(416, 416),
    )

except ImageValidationError as error:

    st.error(str(error))

    st.stop()


# ============================================================
# SHOW ORIGINAL IMAGE
# ============================================================

st.image(
    prepared_image.original_pil,
    caption=uploaded_file.name,
    width=420,
)


# ============================================================
# IMAGE FINGERPRINT
# ============================================================

fingerprint = image_fingerprint(
    uploaded_file
)


# ============================================================
# RUN GLCM ANALYSIS
# ============================================================

if st.button(
    "Run GLCM analysis",
    type="primary",
    use_container_width=True,
):

    # --------------------------------------------------------
    # Shared banana segmentation
    # --------------------------------------------------------

    segmentation_start = perf_counter()

    segmentation = segment_banana(
        rgb_image=prepared_image.working_rgb,
        content_mask=prepared_image.content_mask,
    )

    segmentation_time_ms = (
        perf_counter() - segmentation_start
    ) * 1000.0


    # --------------------------------------------------------
    # Prepare analysis result
    # --------------------------------------------------------

    analysis = None
    error_message = None


    # --------------------------------------------------------
    # Run GLCM
    # --------------------------------------------------------

    if segmentation.success:

        try:

            # GLCM analysis uses the segmented banana image
            analysis = analyse_image(
                prepared_image.working_rgb,
                segmentation.final_mask,
            )

        except (ValueError, AttributeError, FileNotFoundError, TypeError) as error:

            error_message = str(error)

    else:

        error_message = (
            "GLCM analysis stopped because banana segmentation failed."
        )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    st.session_state["glcm_result"] = {

        "result_version": GLCM_RESULT_VERSION,

        "fingerprint": fingerprint,

        "segmentation": segmentation,

        "segmentation_time_ms": segmentation_time_ms,

        "analysis": analysis,

        "error": error_message,
    }


# ============================================================
# GET SAVED RESULT
# ============================================================

saved = st.session_state.get(
    "glcm_result"
)


# ============================================================
# CHECK RESULT
# ============================================================

if (
    saved is None
    or saved.get("result_version") != GLCM_RESULT_VERSION
    or saved.get("fingerprint") != fingerprint
):

    st.caption(
        "Press **Run GLCM analysis** to continue."
    )

    st.stop()


# ============================================================
# GET RESULTS
# ============================================================

segmentation = saved["segmentation"]

analysis = saved["analysis"]


# ============================================================
# SEGMENTATION FAILURE
# ============================================================

if not segmentation.success:

    st.error(
        saved["error"]
    )

    st.image(
        segmentation.overlay_rgb,
        caption="Rejected segmentation",
        use_container_width=True,
    )

    st.stop()


# ============================================================
# GLCM FAILURE
# ============================================================

if analysis is None:

    st.error(
        saved["error"]
        or "GLCM analysis did not return a result."
    )

    st.stop()


# ============================================================
# PREDICTION RESULT
# ============================================================

st.subheader("Analysis Result")


result_columns = st.columns(3)


with result_columns[0]:

    st.metric(
        "Predicted Category",
        analysis.predicted_category,
    )


with result_columns[1]:

    st.metric(
        "Rule-Based Confidence",
        f"{analysis.confidence_percent:.2f}%",
    )


with result_columns[2]:

    st.metric(
        "GLCM Analysis Time",
        f"{analysis.processing_time_ms:.2f} ms",
    )


# ============================================================
# GLCM FEATURES
# ============================================================

st.subheader("GLCM Feature Values")


feature_columns = st.columns(4)


with feature_columns[0]:

    st.metric(
        "Contrast",
        f"{analysis.contrast:.6f}",
    )


with feature_columns[1]:

    st.metric(
        "Homogeneity",
        f"{analysis.homogeneity:.6f}",
    )


with feature_columns[2]:

    st.metric(
        "Energy",
        f"{analysis.energy:.6f}",
    )


with feature_columns[3]:

    st.metric(
        "Correlation",
        f"{analysis.correlation:.6f}",
    )


# ============================================================
# TABS
# ============================================================

overview_tab, texture_tab, details_tab = st.tabs(
    [
        "Overview",
        "GLCM texture",
        "Decision details",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:

    first, second = st.columns(2)


    with first:

        st.markdown(
            "#### Banana segmentation"
        )

        st.image(
            segmentation.overlay_rgb,
            use_container_width=True,
        )


    with second:

        st.markdown(
            "#### Quantised grayscale image"
        )

        # Convert the 8 gray levels into a visible image.
        display_quantised = (
            analysis.quantised_image * 255 // 7
        ).astype("uint8")

        st.image(
            display_quantised,
            clamp=True,
            use_container_width=True,
        )


    st.caption(
        "The banana segmentation is shared with the other "
        "image-processing methods in the project."
    )

    st.caption(
        f"Shared banana-segmentation time: "
        f"{saved['segmentation_time_ms']:.2f} ms"
    )


# ============================================================
# GLCM TEXTURE TAB
# ============================================================

with texture_tab:

    st.markdown(
        "#### GLCM feature values"
    )


    feature_columns = st.columns(4)


    with feature_columns[0]:

        st.metric(
            "Contrast",
            f"{analysis.contrast:.6f}",
        )


    with feature_columns[1]:

        st.metric(
            "Homogeneity",
            f"{analysis.homogeneity:.6f}",
        )


    with feature_columns[2]:

        st.metric(
            "Energy (ASM)",
            f"{analysis.energy:.6f}",
        )


    with feature_columns[3]:

        st.metric(
            "Correlation",
            f"{analysis.correlation:.6f}",
        )


    st.markdown(
        "#### Quantised grayscale"
    )


    display_quantised = (
        analysis.quantised_image * 255 // 7
    ).astype("uint8")


    st.image(
        display_quantised,
        clamp=True,
        use_container_width=True,
    )


    st.caption(
        "The grayscale image is reduced to 8 gray levels "
        "(0-7) before the GLCM is calculated."
    )


# ============================================================
# DECISION DETAILS TAB
# ============================================================

with details_tab:

    st.write(
        f"**Prediction:** "
        f"{analysis.predicted_category}"
    )


    st.write(
        f"**Confidence:** "
        f"{analysis.confidence_percent :.2f}%"
    )


    st.write(
        f"**Processing time:** "
        f"{analysis.processing_time_ms:.2f} ms"
    )


    st.markdown(
        "#### Class decision scores"
    )


    class_order = [
        "Unripe",
        "Ripe",
        "Overripe",
        "Rotten",
    ]


    score_columns = st.columns(4)


    for index, class_name in enumerate(class_order):

        with score_columns[index]:

            score = analysis.method_result.class_scores.get(
                class_name,
                0.0,
            )

            st.metric(
                class_name,
                f"{score * 100:.2f}%",
            )


    st.caption(
    "The class scores show the support for the selected GLCM classification rule."
)


# ============================================================
# END
# ============================================================
