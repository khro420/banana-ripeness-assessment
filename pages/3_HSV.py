from time import perf_counter

import streamlit as st

from branches.hsv.hsv_analysis import HSVRipenessBands, analyse_hsv
from branches.hsv.hsv_segmentation import HSVParameters
from core.banana_segmentation import segment_banana
from core.image_handling import (
    ImageValidationError,
    image_fingerprint,
    prepare_uploaded_image,
)
from ui.components import apply_app_styles, page_header
from ui.result_display import render_result_summary


HSV_RESULT_VERSION = 4


apply_app_styles()

page_header(
    "HSV Colour Analysis",
    "Classify banana ripeness from green, yellow, brown and dark peel regions.",
)

st.info(
    "HSV analysis converts the segmented banana from RGB to HSV colour space. "
    "Green, yellow, brown and very dark peel regions are measured as a "
    "percentage of the visible banana surface."
)

st.warning(
    "Limitation: HSV colour measurements can be affected by illumination, "
    "shadows and camera colour differences. The classification thresholds "
    "below were calibrated on the validation split and should remain frozen "
    "during final test evaluation."
)


parameters = HSVParameters()
bands = HSVRipenessBands()


with st.expander("View frozen HSV decision rules"):
    st.markdown(
        f"""
        Rules are checked in this order:

        1. **Unripe:** green area ≥
           {bands.unripe_min_green_percent:.1f}% and brown area ≤
           {bands.unripe_max_brown_percent:.1f}%.

        2. **Ripe (primary):** yellow area ≥
           {bands.ripe_min_yellow_percent:.1f}% and brown area ≤
           {bands.ripe_max_brown_percent:.1f}%.

        3. **Overripe:** dark area ≥
           {bands.overripe_min_dark_percent:.1f}% and unclassified area ≤
           {bands.overripe_max_other_percent:.1f}%.

        4. **Ripe (secondary):** yellow area ≥
           {bands.ripe_secondary_min_yellow_percent:.1f}% and brown area ≤
           {bands.ripe_secondary_max_brown_percent:.1f}%.

        5. **Rotten:** all remaining cases.
        """
    )

    st.caption(
        "These thresholds were selected using the validation split. Do not "
        "change them after viewing final test results."
    )


uploaded_file = st.file_uploader(
    "Upload one banana image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="hsv_upload",
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


st.image(
    prepared_image.original_pil,
    caption=uploaded_file.name,
    width=420,
)


fingerprint = image_fingerprint(uploaded_file)


if st.button(
    "Run HSV analysis",
    type="primary",
    use_container_width=True,
):
    # ---------------------------------------------------------
    # Shared banana segmentation
    # ---------------------------------------------------------
    segmentation_start = perf_counter()

    segmentation = segment_banana(
        rgb_image=prepared_image.working_rgb,
        content_mask=prepared_image.content_mask,
    )

    segmentation_time_ms = (
        perf_counter() - segmentation_start
    ) * 1000.0

    analysis = None
    error_message = None

    # ---------------------------------------------------------
    # HSV analysis
    # ---------------------------------------------------------
    if segmentation.success:
        try:
            analysis = analyse_hsv(
                rgb_image=prepared_image.working_rgb,
                banana_mask=segmentation.final_mask,
                parameters=parameters,
                bands=bands,
            )

        except ValueError as error:
            error_message = str(error)

    else:
        error_message = (
            "HSV analysis stopped because segmentation failed."
        )

    # ---------------------------------------------------------
    # Save current result in Streamlit session state
    # ---------------------------------------------------------
    st.session_state["hsv_result"] = {
        "result_version": HSV_RESULT_VERSION,
        "fingerprint": fingerprint,
        "segmentation": segmentation,
        "segmentation_time_ms": segmentation_time_ms,
        "analysis": analysis,
        "error": error_message,
    }


# -------------------------------------------------------------
# Retrieve saved result
# -------------------------------------------------------------
saved = st.session_state.get("hsv_result")


if (
    saved is None
    or saved.get("result_version") != HSV_RESULT_VERSION
    or saved.get("fingerprint") != fingerprint
):
    st.caption(
        "Press **Run HSV analysis** to continue."
    )
    st.stop()


segmentation = saved["segmentation"]
analysis = saved["analysis"]


# -------------------------------------------------------------
# Shared segmentation failure
# -------------------------------------------------------------
if not segmentation.success:
    st.error(saved["error"])

    st.image(
        segmentation.overlay_rgb,
        caption="Rejected segmentation",
        use_container_width=True,
    )

    st.stop()


# -------------------------------------------------------------
# HSV-analysis failure
# -------------------------------------------------------------
if analysis is None:
    st.error(
        saved["error"]
        or "HSV analysis did not return a result."
    )

    st.stop()


# -------------------------------------------------------------
# Common result summary
# -------------------------------------------------------------
render_result_summary(
    analysis.method_result
)

st.info(
    analysis.decision_reason
)

st.caption(
    "The confidence value shows rule support. "
    "It is not a learned probability."
)


# -------------------------------------------------------------
# Main HSV measurements
# -------------------------------------------------------------
metric_columns = st.columns(5)

metric_columns[0].metric(
    "Green area",
    f"{analysis.green_percentage:.2f}%",
)

metric_columns[1].metric(
    "Yellow area",
    f"{analysis.yellow_percentage:.2f}%",
)

metric_columns[2].metric(
    "Brown area",
    f"{analysis.brown_percentage:.2f}%",
)

metric_columns[3].metric(
    "Dark area",
    f"{analysis.dark_percentage:.2f}%",
)

metric_columns[4].metric(
    "HSV time",
    f"{analysis.processing_time_ms:.2f} ms",
)


# -------------------------------------------------------------
# Tabs
# -------------------------------------------------------------
overview_tab, masks_tab, rules_tab = st.tabs(
    [
        "Overview",
        "HSV colour masks",
        "Decision details",
    ]
)


# =============================================================
# OVERVIEW TAB
# =============================================================
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
            "#### HSV colour regions"
        )

        st.image(
            analysis.masks.colour_overlay_rgb,
            use_container_width=True,
        )

    st.caption(
        "Green marks green peel, yellow marks yellow peel, brown marks brown "
        "peel and red marks very dark peel. Pixels outside the shared banana "
        "mask are excluded from the HSV measurements."
    )

    st.caption(
        f"Shared banana-segmentation time: "
        f"{saved['segmentation_time_ms']:.2f} ms"
    )


# =============================================================
# HSV COLOUR MASKS TAB
# =============================================================
with masks_tab:
    first, second = st.columns(2)

    with first:
        st.markdown(
            "#### Green peel mask"
        )

        st.image(
            analysis.masks.green_mask,
            clamp=True,
            use_container_width=True,
        )

    with second:
        st.markdown(
            "#### Yellow peel mask"
        )

        st.image(
            analysis.masks.yellow_mask,
            clamp=True,
            use_container_width=True,
        )

    third, fourth = st.columns(2)

    with third:
        st.markdown(
            "#### Brown peel mask"
        )

        st.image(
            analysis.masks.brown_mask,
            clamp=True,
            use_container_width=True,
        )

    with fourth:
        st.markdown(
            "#### Dark peel mask"
        )

        st.image(
            analysis.masks.dark_mask,
            clamp=True,
            use_container_width=True,
        )

    st.markdown(
        "#### Unclassified peel mask"
    )

    st.image(
        analysis.masks.other_mask,
        clamp=True,
        use_container_width=True,
    )

    st.caption(
        "White pixels belong to the named region. The unclassified mask "
        "contains banana pixels that do not satisfy the frozen green, "
        "yellow, brown or dark HSV ranges."
    )


# =============================================================
# DECISION DETAILS TAB
# =============================================================
with rules_tab:
    st.write(
        f"**Prediction:** "
        f"{analysis.predicted_category}"
    )

    st.write(
        f"**Surface grade:** "
        f"{analysis.surface_grade}"
    )

    st.write(
        f"**Reason:** "
        f"{analysis.decision_reason}"
    )

    first, second, third = st.columns(3)

    first.metric(
        "Green",
        f"{analysis.green_percentage:.2f}%",
    )

    second.metric(
        "Yellow",
        f"{analysis.yellow_percentage:.2f}%",
    )

    third.metric(
        "Brown",
        f"{analysis.brown_percentage:.2f}%",
    )

    fourth, fifth, sixth = st.columns(3)

    fourth.metric(
        "Dark",
        f"{analysis.dark_percentage:.2f}%",
    )

    fifth.metric(
        "Brown + dark",
        f"{analysis.deteriorated_percentage:.2f}%",
    )

    sixth.metric(
        "Other",
        f"{analysis.other_percentage:.2f}%",
    )

    st.caption(
        "Only pixels inside the shared banana segmentation mask contribute "
        "to these percentages."
    )

    st.caption(
        "The classifier applies the frozen rules from top to bottom. "
        "The first matching rule becomes the final prediction."
    )