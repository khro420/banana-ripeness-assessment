import streamlit as st

from core.image_handling import (
    ImageValidationError,
    image_fingerprint,
    load_uploaded_image,
)
from core.result_schema import placeholder_method_result
from ui.components import (
    apply_app_styles,
    page_header,
    render_confidence_explanation,
    render_image_metadata,
    render_method_scope,
    render_placeholder_box,
)
from ui.result_display import (
    render_method_outputs,
    render_result_summary,
)


def render_single_method_page(
    method_key: str,
    title: str,
    description: str,
    scope_items: list[str],
) -> None:
    """Render the common upload-and-analysis interface."""

    apply_app_styles()
    page_header(title, description)
    render_method_scope(scope_items)

    st.subheader("Upload one banana image")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key=f"{method_key}_uploader",
        help="Maximum file size: 20 MB.",
    )

    if uploaded_file is None:
        render_placeholder_box(
            "No image selected",
            "Upload one banana image to activate the analysis button.",
        )
        render_confidence_explanation()
        return

    try:
        image, metadata = load_uploaded_image(uploaded_file)
    except ImageValidationError as error:
        st.error(str(error))
        return

    image_column, information_column = st.columns([1.25, 1])

    with image_column:
        st.markdown("#### Uploaded image")
        st.image(
            image,
            caption=metadata["filename"],
            use_container_width=True,
        )

    with information_column:
        st.markdown("#### Image information")
        render_image_metadata(metadata)

        st.info(
            "The image is currently only being validated and displayed. "
            "No banana segmentation or classification is running."
        )

    current_fingerprint = image_fingerprint(uploaded_file)
    result_state_key = f"{method_key}_result"
    fingerprint_state_key = f"{method_key}_result_fingerprint"

    analyse_clicked = st.button(
        "Analyse image",
        type="primary",
        use_container_width=True,
        key=f"{method_key}_analyse_button",
    )

    if analyse_clicked:
        st.session_state[result_state_key] = placeholder_method_result(
            method_key
        )
        st.session_state[fingerprint_state_key] = current_fingerprint

        st.toast(
            "UI flow completed. Processing logic is still pending.",
            icon="🍌",
        )

    saved_result = st.session_state.get(result_state_key)
    saved_fingerprint = st.session_state.get(fingerprint_state_key)

    if saved_result is not None and saved_fingerprint == current_fingerprint:
        st.divider()

        summary_tab, evidence_tab = st.tabs(
            ["Result summary", "Approach evidence"]
        )

        with summary_tab:
            render_result_summary(saved_result)
            render_confidence_explanation()

        with evidence_tab:
            render_method_outputs(saved_result)

    else:
        st.caption(
            "Press **Analyse image** to preview the result layout."
        )