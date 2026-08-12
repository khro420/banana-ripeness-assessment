from typing import Any

import streamlit as st


def apply_app_styles() -> None:
    """Apply restrained styling shared by every page."""

    st.markdown(
        """
        <style>
            .page-kicker {
                color: #8A6500;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.08rem;
                margin-bottom: 0.25rem;
                text-transform: uppercase;
            }

            .page-description {
                color: #5F6368;
                font-size: 1rem;
                margin-bottom: 1.4rem;
                max-width: 900px;
            }

            .placeholder-box {
                border: 1px dashed #B8A56A;
                border-radius: 0.65rem;
                min-height: 150px;
                padding: 1.25rem;
                text-align: center;
            }

            .placeholder-title {
                font-weight: 650;
                margin-bottom: 0.4rem;
            }

            .placeholder-copy {
                color: #6B6B6B;
                font-size: 0.9rem;
            }

            div[data-testid="stMetric"] {
                border: 1px solid rgba(49, 51, 63, 0.15);
                border-radius: 0.65rem;
                padding: 0.8rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    description: str,
    kicker: str = "Banana ripeness assessment",
) -> None:
    st.markdown(
        f'<div class="page-kicker">{kicker}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<div class="page-description">{description}</div>',
        unsafe_allow_html=True,
    )


def render_placeholder_box(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="placeholder-box">
            <div class="placeholder-title">{title}</div>
            <div class="placeholder-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_metadata(metadata: dict[str, Any]) -> None:
    first, second, third = st.columns(3)

    first.metric(
        "Dimensions",
        f"{metadata['width']} × {metadata['height']}",
    )
    second.metric(
        "File size",
        f"{metadata['file_size_kb']} KB",
    )
    third.metric(
        "Colour mode",
        metadata["colour_mode"],
    )

    st.caption(
        f"File: {metadata['filename']} · Type: {metadata['file_type']}"
    )


def render_confidence_explanation() -> None:
    with st.expander("What does rule-based confidence mean?"):
        st.write(
            "Rule-based confidence will represent the relative strength of "
            "the winning decision score compared with the other class scores."
        )
        st.warning(
            "It is not a statistical probability and must not be described "
            "as prediction accuracy."
        )


def render_method_scope(items: list[str]) -> None:
    with st.expander("What this approach will eventually show"):
        for item in items:
            st.markdown(f"- {item}")