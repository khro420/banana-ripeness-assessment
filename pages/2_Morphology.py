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


MORPHOLOGY_RESULT_VERSION = 5


apply_app_styles()
page_header(
    "Morphological Dark-Region Analysis",
    "Classify ripeness from the amount, connectivity and spread of dark peel regions.",
)

st.info(
    "Validation showed that Overripe bananas usually have widespread connected "
    "darkening. Rotten samples are more variable and frequently contain a more "
    "fragmented or irregular dark-region pattern."
)
st.warning(
    "Limitation: morphology still cannot directly see the difference between "
    "clean green and clean yellow peel. Unripe versus Ripe remains the weakest "
    "part of this approach."
)

parameters = MorphologyParameters()
bands = RipenessBands()

with st.expander("View frozen validation-derived rules"):
    st.markdown(
        f"""
        **Later-stage decision**

        - When total darkness is above **{bands.high_total_dark_percent:.1f}%**,
          predict **Overripe** only when spread is at least
          **{bands.overripe_min_spread_percent:.1f}%**, the number of dark
          components is at most **{bands.overripe_max_component_count}**, and
          the largest patch's mean intensity is at most
          **{bands.overripe_max_patch_mean_intensity:.0f}/255**.
        - Otherwise, the high-darkness sample is **Rotten**.

        **Lower-darkness decision**

        - If concentration is at most **{bands.low_concentration_percent:.1f}%**,
          extreme darkness and connected-component count separate the classes.
        - If concentration is higher, the dominant patch size and spread are
          used: **≤ {bands.very_small_patch_percent:.1f}%**, then
          **≤ {bands.small_patch_percent:.1f}%**, followed by a spread boundary
          of **{bands.localised_spread_percent:.1f}%**.
        """
    )
    st.caption(
        "These rules were selected from dataset/valid. Freeze them before the "
        "final test evaluation; do not tune them using dataset/test."
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
    prepared_image = prepare_uploaded_image(
        uploaded_file,
        target_size=(416, 416),
    )
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
        error_message = (
            "Morphology analysis stopped because banana segmentation failed."
        )

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
    "Confidence indicates deterministic rule support. It is not a learned "
    "probability."
)

first_metrics = st.columns(4)
first_metrics[0].metric(
    "Total dark area",
    f"{analysis.total_dark_percentage:.2f}%",
)
first_metrics[1].metric(
    "Largest dark patch",
    f"{analysis.largest_dark_patch_percentage:.2f}%",
)
first_metrics[2].metric(
    "Concentration ratio",
    f"{analysis.concentration_ratio:.2f}%",
)
first_metrics[3].metric(
    "Dark-region spread",
    f"{analysis.dark_region_spread:.2f}%",
)

patch_intensity = (
    "N/A"
    if analysis.largest_patch_mean_intensity is None
    else f"{analysis.largest_patch_mean_intensity:.2f}/255"
)

second_metrics = st.columns(4)
second_metrics[0].metric(
    "Dark components",
    analysis.dark_component_count,
)
second_metrics[1].metric(
    "Extreme-dark area",
    f"{analysis.extreme_dark_percentage:.2f}%",
)
second_metrics[2].metric(
    "Largest-patch intensity",
    patch_intensity,
)
second_metrics[3].metric(
    "Morphology time",
    f"{analysis.processing_time_ms:.2f} ms",
)

st.metric("Surface quality", analysis.surface_grade)

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
        "Red marks cleaned dark pixels. Green grid cells pass the dark-cell "
        "threshold; blue cells contain banana pixels but do not pass it."
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

    grid_columns = st.columns(3)
    grid_columns[0].metric(
        "Active dark cells",
        analysis.masks.active_spread_cells,
    )
    grid_columns[1].metric(
        "Valid banana cells",
        analysis.masks.valid_spread_cells,
    )
    grid_columns[2].metric(
        "Spread",
        f"{analysis.dark_region_spread:.2f}%",
    )

    st.caption(
        "The classifier uses seven measurements derived from the same cleaned "
        "dark mask. No colour-space classifier or trained ripeness model is "
        "used by this branch."
    )