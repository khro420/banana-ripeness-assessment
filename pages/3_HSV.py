from time import perf_counter

import streamlit as st

from branches.hsv.hsv_analysis import (
    HSVQualityBands,
    HSVRipenessBands,
    analyse_hsv,
)

from branches.hsv.hsv_segmentation import (
    HSVParameters,
)

from core.banana_segmentation import (
    segment_banana,
)

from core.image_handling import (
    ImageValidationError,
    image_fingerprint,
    prepare_uploaded_image,
)

from ui.components import (
    apply_app_styles,
    page_header,
)

from ui.multiple_image_mode import (
    render_image_input_mode,
)

from ui.result_display import (
    render_result_summary,
)


HSV_RESULT_VERSION = 6


apply_app_styles()

page_header(
    "HSV Colour Analysis",
    "Classify ripeness, then grade surface quality only when the result is Ripe.",
)

render_image_input_mode(
    "hsv",
    "hsv",
)


parameters = HSVParameters()

bands = HSVRipenessBands()

quality_bands = HSVQualityBands()


# ================================================================
# IMAGE UPLOAD
# ================================================================

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


# ================================================================
# RUN ANALYSIS
# ================================================================

if st.button(
    "Run HSV analysis",
    type="primary",
    use_container_width=True,
):

    # ------------------------------------------------------------
    # 1. SHARED BANANA SEGMENTATION
    # ------------------------------------------------------------

    segmentation_start = perf_counter()

    segmentation = segment_banana(
        rgb_image=
            prepared_image.working_rgb,

        content_mask=
            prepared_image.content_mask,
    )

    segmentation_time_ms = (
        perf_counter()
        - segmentation_start
    ) * 1000.0


    analysis = None
    error_message = None


    # ------------------------------------------------------------
    # 2. HSV ANALYSIS
    # ------------------------------------------------------------

    if segmentation.success:

        try:

            analysis = analyse_hsv(

                rgb_image=
                    prepared_image.working_rgb,

                banana_mask=
                    segmentation.final_mask,

                parameters=
                    parameters,

                bands=
                    bands,

                quality_bands=
                    quality_bands,
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

        "result_version":
            HSV_RESULT_VERSION,

        "fingerprint":
            fingerprint,

        "segmentation":
            segmentation,

        "segmentation_time_ms":
            segmentation_time_ms,

        "analysis":
            analysis,

        "error":
            error_message,
    }


# ================================================================
# LOAD SAVED RESULT
# ================================================================

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


# ================================================================
# FINAL RIPENESS RESULT
# ================================================================

render_result_summary(
    analysis.method_result
)

st.caption(
    analysis.decision_reason
)


# ================================================================
# CONDITIONAL SURFACE QUALITY
# ================================================================

st.subheader(
    "Conditional quality assessment"
)


if analysis.quality_assessed:

    quality_columns = st.columns(
        2
    )

    quality_columns[0].metric(
        "Quality class",
        analysis.predicted_quality,
    )

    quality_columns[1].metric(
        "Quality confidence",
        f"{analysis.quality_confidence_percent:.2f}%",
    )

    st.caption(
        analysis.quality_reason
    )

else:

    st.caption(
        "Quality classification is available "
        "for Ripe bananas only."
    )


# ================================================================
# MAIN HSV MEASUREMENTS
# ================================================================

metrics = st.columns(
    5
)

metrics[0].metric(
    "Green",
    f"{analysis.green_percentage:.2f}%",
)

metrics[1].metric(
    "Yellow",
    f"{analysis.yellow_percentage:.2f}%",
)

metrics[2].metric(
    "Brown",
    f"{analysis.brown_percentage:.2f}%",
)

metrics[3].metric(
    "Dark",
    f"{analysis.dark_percentage:.2f}%",
)

metrics[4].metric(
    "Other",
    f"{analysis.other_percentage:.2f}%",
)


if analysis.quality_assessed:

    st.caption(
        f"Brown + dark: "
        f"{analysis.deteriorated_percentage:.2f}%"
        f" | Yellow / deterioration: "
        f"{analysis.yellow_deterioration_ratio:.2f}"
    )


st.caption(
    f"HSV processing time: "
    f"{analysis.processing_time_ms:.2f} ms"
    f" | Shared segmentation time: "
    f"{saved['segmentation_time_ms']:.2f} ms"
)


# ================================================================
# IMPORTANT DEMO VISUALS ONLY
# ================================================================

st.subheader(
    "Analysis visualisation"
)


first, second = st.columns(
    2
)


with first:

    st.image(
        segmentation.overlay_rgb,
        caption="Banana segmentation",
        use_container_width=True,
    )


with second:

    st.image(
        analysis.masks.colour_overlay_rgb,
        caption="HSV colour regions",
        use_container_width=True,
    )


st.caption(
    "Overlay: green = green peel, "
    "yellow = yellow peel, "
    "brown = brown peel, "
    "red = very dark peel. "
    "Background pixels are excluded."
)