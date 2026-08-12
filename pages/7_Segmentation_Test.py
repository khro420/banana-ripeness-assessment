from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from core.banana_segmentation import segment_banana
from core.image_handling import (
    ImageValidationError,
    image_fingerprint,
    prepare_uploaded_image,
)
from ui.components import apply_app_styles, page_header


def to_png_bytes(array) -> bytes:
    """Encode an RGB, RGBA or greyscale array as PNG."""

    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG")

    return output.getvalue()


apply_app_styles()

page_header(
    "Segmentation Test",
    (
        "Development page for inspecting the shared banana foreground "
        "segmentation before connecting the four analysis approaches."
    ),
    kicker="Development and quality assurance",
)

st.warning(
    "This is a development page, not an additional classification approach. "
    "Its purpose is to expose segmentation failures."
)

with st.expander("Segmentation settings", expanded=False):
    grabcut_iterations = st.slider(
        "GrabCut iterations",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help=(
            "More iterations may improve the boundary but increase "
            "processing time."
        ),
    )

    minimum_area_percent = st.slider(
        "Minimum accepted foreground area",
        min_value=1.0,
        max_value=20.0,
        value=2.0,
        step=0.5,
        format="%.1f%%",
    )

uploaded_file = st.file_uploader(
    "Upload a banana image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="segmentation_test_upload",
)

if uploaded_file is None:
    st.info(
        "Upload one image to inspect the segmentation pipeline."
    )
    st.stop()

try:
    prepared_image = prepare_uploaded_image(
        uploaded_file,
        target_size=(416, 416),
    )
except ImageValidationError as error:
    st.error(str(error))
    st.stop()

current_fingerprint = (
    f"{image_fingerprint(uploaded_file)}:"
    f"{grabcut_iterations}:"
    f"{minimum_area_percent}"
)

input_column, standardised_column = st.columns(2)

with input_column:
    st.markdown("#### Original upload")
    st.image(
        prepared_image.original_pil,
        use_container_width=True,
    )

with standardised_column:
    st.markdown("#### Standardised working image")
    st.image(
        prepared_image.working_rgb,
        use_container_width=True,
    )
    st.caption(
        "Aspect ratio is preserved. Padding is excluded using the "
        "content mask."
    )

run_segmentation = st.button(
    "Run banana segmentation",
    type="primary",
    use_container_width=True,
)

if run_segmentation:
    with st.spinner("Segmenting the banana region..."):
        try:
            segmentation_result = segment_banana(
                rgb_image=prepared_image.working_rgb,
                content_mask=prepared_image.content_mask,
                grabcut_iterations=grabcut_iterations,
                minimum_area_ratio=(
                    minimum_area_percent / 100.0
                ),
            )
        except Exception as error:
            st.error(
                f"Segmentation failed with an unexpected error: {error}"
            )
            st.exception(error)
            st.stop()

    st.session_state["segmentation_test_result"] = (
        segmentation_result
    )
    st.session_state["segmentation_test_prepared"] = (
        prepared_image
    )
    st.session_state["segmentation_test_fingerprint"] = (
        current_fingerprint
    )

saved_result = st.session_state.get(
    "segmentation_test_result"
)
saved_prepared = st.session_state.get(
    "segmentation_test_prepared"
)
saved_fingerprint = st.session_state.get(
    "segmentation_test_fingerprint"
)

if (
    saved_result is None
    or saved_prepared is None
    or saved_fingerprint != current_fingerprint
):
    st.caption(
        "Press **Run banana segmentation** to generate the masks."
    )
    st.stop()

result = saved_result
prepared_image = saved_prepared

if result.success:
    st.success(result.message)
else:
    st.error(result.message)

status_column, area_column, box_column, fallback_column = (
    st.columns(4)
)

status_column.metric(
    "Segmentation status",
    "Accepted" if result.success else "Rejected",
)

area_column.metric(
    "Detected area",
    f"{result.banana_area_percent:.2f}%",
)

box_column.metric(
    "Bounding box",
    (
        "None"
        if result.bounding_box is None
        else " × ".join(
            map(
                str,
                result.bounding_box[2:],
            )
        )
    ),
)

fallback_column.metric(
    "GrabCut fallback",
    (
        "Used"
        if result.diagnostics["grabcut_fallback_used"]
        else "No"
    ),
)

input_tab, final_tab, masks_tab, diagnostics_tab = st.tabs(
    [
        "Input preparation",
        "Final output",
        "Pipeline masks",
        "Diagnostics",
    ]
)

with input_tab:
    first, second, third = st.columns(3)

    with first:
        st.markdown("#### Working image")
        st.image(
            prepared_image.working_rgb,
            use_container_width=True,
        )

    with second:
        st.markdown("#### Content mask")
        st.image(
            prepared_image.content_mask,
            clamp=True,
            use_container_width=True,
        )

    with third:
        st.markdown("#### Inner peel mask")
        st.image(
            result.inner_mask,
            clamp=True,
            use_container_width=True,
        )

with final_tab:
    first, second, third = st.columns(3)

    with first:
        st.markdown("#### Boundary overlay")
        st.image(
            result.overlay_rgb,
            use_container_width=True,
        )

    with second:
        st.markdown("#### Isolated banana")
        st.image(
            result.segmented_rgb,
            use_container_width=True,
        )

    with third:
        st.markdown("#### Transparent background")
        st.image(
            result.foreground_rgba,
            use_container_width=True,
        )

    file_stem = Path(uploaded_file.name).stem

    download_mask_column, download_image_column = st.columns(2)

    with download_mask_column:
        st.download_button(
            "Download final mask",
            data=to_png_bytes(result.final_mask),
            file_name=f"{file_stem}_banana_mask.png",
            mime="image/png",
            use_container_width=True,
        )

    with download_image_column:
        st.download_button(
            "Download segmented image",
            data=to_png_bytes(result.foreground_rgba),
            file_name=f"{file_stem}_segmented.png",
            mime="image/png",
            use_container_width=True,
        )

with masks_tab:
    first, second = st.columns(2)

    with first:
        st.markdown("#### Initial candidate mask")
        st.image(
            result.initial_candidate_mask,
            clamp=True,
            use_container_width=True,
        )

    with second:
        st.markdown("#### GrabCut foreground mask")
        st.image(
            result.grabcut_mask,
            clamp=True,
            use_container_width=True,
        )

    third, fourth = st.columns(2)

    with third:
        st.markdown("#### Final primary component")
        st.image(
            result.final_mask,
            clamp=True,
            use_container_width=True,
        )

    with fourth:
        st.markdown("#### Inner analysis region")
        st.image(
            result.inner_mask,
            clamp=True,
            use_container_width=True,
        )

with diagnostics_tab:
    diagnostic_rows = [
        {
            "Diagnostic": key.replace("_", " ").title(),
            "Value": value,
        }
        for key, value in result.diagnostics.items()
    ]

    st.dataframe(
        pd.DataFrame(diagnostic_rows),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Image standardisation")

    standardisation_rows = [
        {
            "Property": "Original dimensions",
            "Value": (
                f"{prepared_image.metadata['original_width']} × "
                f"{prepared_image.metadata['original_height']}"
            ),
        },
        {
            "Property": "Working dimensions",
            "Value": (
                f"{prepared_image.metadata['working_width']} × "
                f"{prepared_image.metadata['working_height']}"
            ),
        },
        {
            "Property": "Resize scale",
            "Value": round(prepared_image.scale, 5),
        },
        {
            "Property": "Padding (left, top, right, bottom)",
            "Value": str(prepared_image.padding),
        },
    ]

    st.dataframe(
        pd.DataFrame(standardisation_rows),
        hide_index=True,
        use_container_width=True,
    )