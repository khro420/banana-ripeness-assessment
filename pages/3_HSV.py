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
from ui.multiple_image_mode import render_image_input_mode
from ui.result_display import render_result_summary


HSV_RESULT_VERSION = 7


apply_app_styles()

page_header(
    "HSV Colour Analysis",
    "Classify ripeness, then grade surface quality only when the result is Ripe.",
)

render_image_input_mode("hsv", "hsv")

parameters = HSVParameters()
bands = HSVRipenessBands()
quality_bands = HSVQualityBands()


# The user supplies one image, then starts analysis manually.
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


# Keep the button so results are only calculated when requested.
if st.button(
    "Run HSV analysis",
    type="primary",
    width="stretch",
):
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
            error_message = str(error)
    else:
        error_message = (
            "HSV analysis stopped because banana segmentation failed."
        )

    st.session_state["hsv_result"] = {
        "result_version": HSV_RESULT_VERSION,
        "fingerprint": fingerprint,
        "segmentation": segmentation,
        "segmentation_time_ms": segmentation_time_ms,
        "analysis": analysis,
        "error": error_message,
    }


saved = st.session_state.get("hsv_result")


# Avoid showing an old result after another image is uploaded.
if (
    saved is None
    or saved.get("result_version") != HSV_RESULT_VERSION
    or saved.get("fingerprint") != fingerprint
):
    st.caption("Press **Run HSV analysis** to continue.")
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
    st.error(
        saved["error"]
        or "HSV analysis did not return a result."
    )
    st.stop()


# Final ripeness result.
render_result_summary(analysis.method_result)

st.caption(analysis.decision_reason)


# Quality is only relevant when the ripeness result is Ripe.
st.subheader("Conditional quality assessment")

if analysis.quality_assessed:
    quality_columns = st.columns(2)

    quality_columns[0].metric(
        "Quality class",
        analysis.predicted_quality,
    )

    quality_columns[1].metric(
        "Quality confidence",
        f"{analysis.quality_confidence_percent:.2f}%",
    )

    st.caption(analysis.quality_reason)

else:
    st.caption(
        "Quality classification is available for Ripe bananas only."
    )


# Main HSV measurements.
metrics = st.columns(5)

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


# Main processing visualisations.
st.subheader("Analysis visualisation")

first, second = st.columns(2)

with first:
    st.image(
        segmentation.overlay_rgb,
        caption="Banana segmentation",
        width="stretch",
    )

with second:
    st.image(
        analysis.masks.colour_overlay_rgb,
        caption="HSV colour regions",
        width="stretch",
    )


st.caption(
    "Overlay: green = green peel, "
    "yellow = yellow peel, "
    "brown = brown peel, "
    "red = very dark peel. "
    "Background pixels are excluded."
)


# Individual HSV masks.
st.subheader("HSV colour masks")

first, second, third = st.columns(3)

with first:
    st.image(
        analysis.masks.green_mask,
        caption="Green peel mask",
        clamp=True,
        width="stretch",
    )

with second:
    st.image(
        analysis.masks.yellow_mask,
        caption="Yellow peel mask",
        clamp=True,
        width="stretch",
    )

with third:
    st.image(
        analysis.masks.brown_mask,
        caption="Brown peel mask",
        clamp=True,
        width="stretch",
    )


first, second = st.columns(2)

with first:
    st.image(
        analysis.masks.dark_mask,
        caption="Dark peel mask",
        clamp=True,
        width="stretch",
    )

with second:
    st.image(
        analysis.masks.other_mask,
        caption="Unclassified peel mask",
        clamp=True,
        width="stretch",
    )


# Extra information is available without cluttering the main demo page.
with st.expander("Decision details"):
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
            f"**Quality confidence:** "
            f"{analysis.quality_confidence_percent:.2f}%"
        )

        st.write(
            f"**Quality reason:** "
            f"{analysis.quality_reason}"
        )

        st.write(
            f"**Yellow / deterioration ratio:** "
            f"{analysis.yellow_deterioration_ratio:.2f}"
        )

    else:
        st.write(
            "**Surface quality:** Not assessed"
        )

    st.write(
        f"**HSV processing time:** "
        f"{analysis.processing_time_ms:.2f} ms"
    )

    st.write(
        f"**Shared segmentation time:** "
        f"{saved['segmentation_time_ms']:.2f} ms"
    )

    st.caption(
        "All HSV quality measurements are derived from the HSV "
        "colour masks. No morphology measurements are used by "
        "the HSV quality branch."
    )