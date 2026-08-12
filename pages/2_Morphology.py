from time import perf_counter

import streamlit as st

from branches.morphology.morphology_analysis import (
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


MORPHOLOGY_RESULT_VERSION = 4


apply_app_styles()
page_header(
    "Morphological Dark-Region Analysis",
    "Classify banana ripeness from total dark area and its spatial spread.",
)

st.info(
    "Overripe means extensive darkness spread throughout the peel. Rotten "
    "means substantial darkness that is less widely distributed."
)
st.warning(
    "Limitation: dark-region morphology cannot reliably separate a spotless "
    "green banana from a spotless yellow banana. Unripe and Ripe are therefore "
    "the weakest classes for this approach."
)

parameters = MorphologyParameters()
bands = RipenessBands()

with st.expander("View frozen decision rules"):
    st.markdown(
        f"""
        Rules are checked in this order:

        1. **Overripe:** total dark area ≥
           {bands.overripe_min_total_dark_percent:.1f}% and spread ≥
           {bands.overripe_min_spread_percent:.1f}%.
        2. **Rotten:** total dark area ≥
           {bands.rotten_min_total_dark_percent:.1f}% after the Overripe rule.
        3. **Unripe:** total dark area ≤
           {bands.unripe_max_total_dark_percent:.1f}%.
        4. **Ripe:** all remaining cases.
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
    key="morphology_upload",
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    prepared_image = prepare_uploaded_image(uploaded_file, target_size=(416, 416))
except ImageValidationError as error:
    st.error(str(error))
    st.stop()

st.image(prepared_image.original_pil, caption=uploaded_file.name, width=420)
fingerprint = image_fingerprint(uploaded_file)

if st.button("Run morphology analysis", type="primary", use_container_width=True):
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
            )
        except ValueError as error:
            error_message = str(error)
    else:
        error_message = "Morphology analysis stopped because segmentation failed."

    st.session_state["morphology_result"] = {
        "result_version": MORPHOLOGY_RESULT_VERSION,
        "fingerprint": fingerprint,
        "segmentation": segmentation,
        "segmentation_time_ms": segmentation_time_ms,
        "analysis": analysis,
        "error": error_message,
    }


saved = st.session_state.get("morphology_result")
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
        use_container_width=True,
    )
    st.stop()

if analysis is None:
    st.error(saved["error"] or "Morphology analysis did not return a result.")
    st.stop()


render_result_summary(analysis.method_result)
st.info(analysis.decision_reason)
st.caption(
    "The confidence value shows rule support. It is not a learned probability."
)

metric_columns = st.columns(4)
metric_columns[0].metric(
    "Total dark area",
    f"{analysis.total_dark_percentage:.2f}%",
)
metric_columns[1].metric(
    "Dark-region spread",
    f"{analysis.dark_region_spread:.2f}%",
)
metric_columns[2].metric("Surface grade", analysis.surface_grade)
metric_columns[3].metric(
    "Morphology time",
    f"{analysis.processing_time_ms:.2f} ms",
)

overview_tab, masks_tab, rules_tab = st.tabs(
    ["Overview", "Morphology masks", "Decision details"]
)

with overview_tab:
    first, second, third = st.columns(3)

    with first:
        st.markdown("#### Banana segmentation")
        st.image(segmentation.overlay_rgb, use_container_width=True)

    with second:
        st.markdown("#### Detected dark regions")
        st.image(analysis.masks.blemish_overlay_rgb, use_container_width=True)

    with third:
        st.markdown("#### Spatial spread grid")
        st.image(analysis.masks.spread_overlay_rgb, use_container_width=True)

    st.caption(
        "Red marks detected dark pixels. Green grid cells pass the dark-cell "
        "threshold; blue cells contain banana pixels but do not pass it. The "
        "full segmented banana, including both tips, is analysed."
    )
    st.caption(
        f"Shared banana-segmentation time: "
        f"{saved['segmentation_time_ms']:.2f} ms"
    )

with masks_tab:
    first, second = st.columns(2)
    with first:
        st.markdown("#### Greyscale banana")
        st.image(
            analysis.masks.greyscale_image,
            clamp=True,
            use_container_width=True,
        )
    with second:
        st.markdown("#### Black-hat response")
        st.image(
            analysis.masks.blackhat_response,
            clamp=True,
            use_container_width=True,
        )

    third, fourth = st.columns(2)
    with third:
        st.markdown("#### Raw dark candidates")
        st.image(
            analysis.masks.raw_blemish_mask,
            clamp=True,
            use_container_width=True,
        )
    with fourth:
        st.markdown("#### Cleaned dark mask")
        st.image(
            analysis.masks.blemish_mask,
            clamp=True,
            use_container_width=True,
        )

with rules_tab:
    st.write(f"**Prediction:** {analysis.predicted_category}")
    st.write(f"**Surface grade:** {analysis.surface_grade}")
    st.write(f"**Reason:** {analysis.decision_reason}")

    spread_columns = st.columns(3)
    spread_columns[0].metric(
        "Active dark cells",
        analysis.masks.active_spread_cells,
    )
    spread_columns[1].metric(
        "Valid banana cells",
        analysis.masks.valid_spread_cells,
    )
    spread_columns[2].metric(
        "Spread",
        f"{analysis.dark_region_spread:.2f}%",
    )

    st.caption(
        "Only two classification measurements are retained: total dark "
        "percentage and dark-region spread. Largest-patch and concentration "
        "features were removed because validation showed that they added "
        "complexity without improving the selected rule set."
    )
