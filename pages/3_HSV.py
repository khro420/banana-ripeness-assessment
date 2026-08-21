from time import perf_counter

import streamlit as st

from branches.hsv.hsv_analysis import (
    HSVQualityBands,
    HSVRipenessBands,
    analyse_hsv,
)
from branches.hsv.hsv_segmentation import HSVParameters
from core.banana_segmentation import segment_banana
from core.image_handling import (
    ImageValidationError,
    image_fingerprint,
    prepare_uploaded_image,
)
from ui.components import apply_app_styles, page_header
from ui.result_display import render_result_summary


HSV_RESULT_VERSION = 5


apply_app_styles()

page_header(
    "HSV Colour Analysis",
    (
        "Classify banana ripeness and assess Ripe banana "
        "surface quality using HSV colour measurements."
    ),
)

st.info(
    "HSV analysis converts the segmented banana from RGB to HSV colour "
    "space. Green, yellow, brown and dark peel regions are measured as "
    "percentages of the visible banana surface."
)

st.warning(
    "Ripeness and quality thresholds must be calibrated using their "
    "respective validation splits. The test splits must not be used for "
    "threshold tuning."
)

parameters = HSVParameters()
bands = HSVRipenessBands()
quality_bands = HSVQualityBands()


with st.expander("View HSV ripeness decision rules"):
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


with st.expander("View HSV surface-quality decision rules"):
    st.markdown(
        f"""
        Surface quality is assessed **only when the HSV ripeness result is Ripe**.

        1. **Defect:** brown + dark area >
           {quality_bands.defect_min_deteriorated_percent:.2f}%.

        2. **Class B:** otherwise, brown area ≤
           {quality_bands.class_b_max_brown_percent:.2f}% and
           yellow / (brown + dark) ≤
           {quality_bands.class_b_max_yellow_deterioration_ratio:.2f}.

        3. **Class A:** all remaining non-defect Ripe cases.

        The surface-quality branch uses only HSV-derived measurements.
        It does not use morphology measurements.
        """
    )


uploaded_file = st.file_uploader(
    "Upload one banana image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    accept_multiple_files=False,
    key="hsv_upload",
)


if uploaded_file is None:
    st.info(
        "Upload an image to begin."
    )
    st.stop()


try:
    prepared_image = prepare_uploaded_image(
        uploaded_file,
        target_size=(416, 416),
    )

except ImageValidationError as error:
    st.error(
        str(error)
    )
    st.stop()


st.image(
    prepared_image.original_pil,
    caption=uploaded_file.name,
    width=420,
)

fingerprint = image_fingerprint(
    uploaded_file
)


if st.button(
    "Run HSV analysis",
    type="primary",
    use_container_width=True,
):
    segmentation_start = perf_counter()

    segmentation = segment_banana(
        rgb_image=prepared_image.working_rgb,
        content_mask=prepared_image.content_mask,
    )

    segmentation_time_ms = (
        perf_counter()
        - segmentation_start
    ) * 1000.0

    analysis = None
    error_message = None

    if segmentation.success:
        try:
            analysis = analyse_hsv(
                rgb_image=prepared_image.working_rgb,
                banana_mask=segmentation.final_mask,
                parameters=parameters,
                bands=bands,
                quality_bands=quality_bands,
            )

        except ValueError as error:
            error_message = str(
                error
            )

    else:
        error_message = (
            "HSV analysis stopped because "
            "banana segmentation failed."
        )

    st.session_state[
        "hsv_result"
    ] = {
        "result_version": HSV_RESULT_VERSION,
        "fingerprint": fingerprint,
        "segmentation": segmentation,
        "segmentation_time_ms": segmentation_time_ms,
        "analysis": analysis,
        "error": error_message,
    }


saved = st.session_state.get(
    "hsv_result"
)


if (
    saved is None
    or saved.get(
        "result_version"
    ) != HSV_RESULT_VERSION
    or saved.get(
        "fingerprint"
    ) != fingerprint
):
    st.caption(
        "Press **Run HSV analysis** to continue."
    )
    st.stop()


segmentation = saved[
    "segmentation"
]
analysis = saved[
    "analysis"
]


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


if analysis is None:
    st.error(
        saved["error"]
        or "HSV analysis did not return a result."
    )
    st.stop()


render_result_summary(
    analysis.method_result
)

st.info(
    analysis.decision_reason
)


if analysis.quality_assessed:
    st.success(
        f"Surface quality: "
        f"{analysis.predicted_quality} "
        f"({analysis.quality_confidence_percent:.2f}% rule support)"
    )

    st.caption(
        analysis.quality_reason
    )

else:
    st.info(
        "Surface quality: Not assessed"
    )

    st.caption(
        analysis.quality_reason
    )


metric_columns = st.columns(
    5
)

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


overview_tab, masks_tab, rules_tab = st.tabs(
    [
        "Overview",
        "HSV colour masks",
        "Decision details",
    ]
)


with overview_tab:
    first, second = st.columns(
        2
    )

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
        "Green marks green peel, yellow marks yellow peel, brown marks "
        "brown peel and red marks very dark peel. Pixels outside the "
        "shared banana mask are excluded."
    )

    st.caption(
        f"Shared banana-segmentation time: "
        f"{saved['segmentation_time_ms']:.2f} ms"
    )


with masks_tab:
    first, second = st.columns(
        2
    )

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

    third, fourth = st.columns(
        2
    )

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


with rules_tab:
    st.write(
        f"**Ripeness prediction:** "
        f"{analysis.predicted_category}"
    )

    st.write(
        f"**Ripeness reason:** "
        f"{analysis.decision_reason}"
    )

    if analysis.quality_assessed:
        st.write(
            f"**Surface quality:** "
            f"{analysis.predicted_quality}"
        )

        st.write(
            f"**Quality rule support:** "
            f"{analysis.quality_confidence_percent:.2f}%"
        )

        st.write(
            f"**Quality reason:** "
            f"{analysis.quality_reason}"
        )

    else:
        st.write(
            "**Surface quality:** Not assessed"
        )

    first, second, third = st.columns(
        3
    )

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

    fourth, fifth, sixth = st.columns(
        3
    )

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

    if analysis.quality_assessed:
        st.metric(
            "Yellow / deterioration ratio",
            f"{analysis.yellow_deterioration_ratio:.2f}",
        )

    st.caption(
        "All quality measurements on this page are derived from the HSV "
        "colour masks. No morphology measurements are used by the HSV "
        "surface-quality branch."
    )
