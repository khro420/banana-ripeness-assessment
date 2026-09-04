import pandas as pd
import streamlit as st

from core.result_schema import MethodResult
from ui.components import render_placeholder_box


def _display_value(value: object, suffix: str = "") -> str:
    if value is None:
        return "Pending"

    return f"{value}{suffix}"


def render_result_summary(result: MethodResult) -> None:
    """Display the common output required from every approach."""

    st.subheader("Analysis result")

    if result.is_placeholder:
        st.warning(
            "UI placeholder only. No image-processing prediction has been "
            "calculated."
        )

    first, second, third = st.columns(3)

    first.metric(
        "Predicted category",
        result.predicted_category or "Pending",
    )
    second.metric(
        "Rule-based confidence",
        _display_value(result.confidence_percent, "%"),
    )
    third.metric(
        "Processing time",
        _display_value(result.processing_time_ms, " ms"),
    )

    score_rows = [
        {
            "Category": category,
            "Decision score": result.class_scores.get(category),
        }
        for category in result.class_scores
    ]

    st.markdown("#### Class decision scores")
    st.dataframe(
        pd.DataFrame(score_rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Category": st.column_config.TextColumn("Category"),
            "Decision score": st.column_config.NumberColumn(
                "Decision score",
                format="%.3f",
            ),
        },
    )


def _render_morphology_outputs() -> None:
    st.markdown("### Morphology-specific outputs")

    first, second, third = st.columns(3)

    with first:
        render_placeholder_box(
            "CIELAB candidate masks",
            "Green, yellow, brown and dark candidate masks will appear here.",
        )

    with second:
        render_placeholder_box(
            "Cleaned morphological masks",
            "Opening, closing and connected-component results will appear here.",
        )

    with third:
        render_placeholder_box(
            "Blemish mask",
            "Detected surface blemishes will appear here.",
        )

    metrics = st.columns(3)
    metrics[0].metric("Blemish percentage", "Pending")
    metrics[1].metric("Surface-quality grade", "Pending")
    metrics[2].metric("Dark components", "Pending")


def _render_hsv_outputs() -> None:
    st.markdown("### HSV-specific outputs")

    first, second = st.columns(2)

    with first:
        render_placeholder_box(
            "HSV colour masks",
            "Green, yellow, brown and dark HSV masks will appear here.",
        )

    with second:
        render_placeholder_box(
            "Segmented peel regions",
            "The colour-labelled banana region will appear here.",
        )

    colour_table = pd.DataFrame(
        {
            "Colour region": ["Green", "Yellow", "Brown", "Dark"],
            "Area percentage": [None, None, None, None],
        }
    )

    st.dataframe(
        colour_table,
        hide_index=True,
        use_container_width=True,
    )


def _render_kmeans_outputs() -> None:
    st.markdown("### K-means-specific outputs")

    first, second = st.columns(2)

    with first:
        render_placeholder_box(
            "Clustered image",
            "The banana image recoloured using cluster centroids will appear here.",
        )

    with second:
        render_placeholder_box(
            "Cluster palette",
            "The dominant cluster colours will appear here.",
        )

    cluster_table = pd.DataFrame(
        {
            "Cluster": ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"],
            "Mapped colour": ["Pending"] * 4,
            "Pixel percentage": [None] * 4,
        }
    )

    st.dataframe(
        cluster_table,
        hide_index=True,
        use_container_width=True,
    )


def _render_glcm_outputs() -> None:
    st.markdown("### GLCM texture-specific outputs")

    first, second = st.columns(2)

    with first:
        render_placeholder_box(
            "Greyscale banana ROI",
            "The masked and quantised greyscale banana region will appear here.",
        )

    with second:
        render_placeholder_box(
            "Texture visualisation",
            "A texture or GLCM visualisation will appear here.",
        )

    feature_table = pd.DataFrame(
        {
            "Texture feature": [
                "Contrast",
                "Homogeneity",
                "Energy",
                "Correlation",
            ],
            "Feature value": [None, None, None, None],
        }
    )

    st.dataframe(
        feature_table,
        hide_index=True,
        use_container_width=True,
    )


def _render_hybrid_outputs() -> None:
    st.markdown("### Hybrid-specific outputs")

    vote_table = pd.DataFrame(
        {
            "Approach": [
                "Morphology",
                "HSV",
                "K-means",
                "GLCM Texture",
            ],
            "Prediction": ["Pending"] * 4,
            "Confidence": [None] * 4,
            "Weighted vote": [None] * 4,
        }
    )

    st.dataframe(
        vote_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Confidence": st.column_config.NumberColumn(
                "Rule-based confidence",
                format="%.2f%%",
            ),
            "Weighted vote": st.column_config.NumberColumn(
                "Weighted vote",
                format="%.3f",
            ),
        },
    )

    first, second, third = st.columns(3)
    first.metric("Method agreement", "Pending")
    second.metric("Blemish percentage", "Pending")
    third.metric("Surface-quality grade", "Pending")

    render_placeholder_box(
        "Weighted class scores",
        "The four hybrid class scores will be displayed here.",
    )


def render_method_outputs(result: MethodResult) -> None:
    """Display information that differs between approaches."""

    handlers = {
        "morphology": _render_morphology_outputs,
        "hsv": _render_hsv_outputs,
        "kmeans": _render_kmeans_outputs,
        "glcm": _render_glcm_outputs,
        "hybrid": _render_hybrid_outputs,
    }

    handler = handlers.get(result.method_key)

    if handler is None:
        st.error(f"No UI output handler exists for {result.method_key}.")
        return

    handler()
