import streamlit as st

from branches.kmeans.kmeans_analysis import (
    KMeansQualityBands,
    KMeansRipenessBands,
    analyse_kmeans,
)
from branches.kmeans.kmeans_segmentation import KMeansParameters
from core.banana_segmentation import segment_banana
from core.image_handling import prepare_uploaded_image
from ui.components import apply_app_styles, page_header
from ui.multiple_image_mode import render_image_input_mode
from ui.result_display import render_result_summary


apply_app_styles()

page_header(
    "K-means Colour Clustering",
    "Classify banana ripeness using automatically generated colour clusters.",
)


st.info(
    "K-means analysis groups similar banana-peel pixels into colour clusters. "
    "The detected cluster colours are then analysed to determine whether the "
    "banana is Unripe, Ripe, Overripe or Rotten."
)


st.warning(
    "Limitation: K-means clustering can be affected by lighting, shadows, "
    "reflections and the selected number of clusters. The decision thresholds "
    "should be calibrated using the validation dataset before final testing."
)


render_image_input_mode("kmeans", "kmeans")

parameters = KMeansParameters(k=4)
bands = KMeansRipenessBands()
quality_bands = KMeansQualityBands()


with st.expander("View K-means decision rules"):
    st.markdown(
        f"""
        Rules are checked in this order:

        1. **Rotten:** dark cluster score ≥
           {bands.rotten_dark_score_min * 100:.1f}%.

        2. **Overripe:** brown cluster score ≥
           {bands.overripe_brown_score_min * 100:.1f}%.

        3. **Unripe:** green cluster score ≥
           {bands.unripe_green_score_min * 100:.1f}%.

        4. **Ripe:** yellow cluster score ≥
           {bands.ripe_yellow_score_min * 100:.1f}%.

        **Ripe-only quality**

        - Damage ≤ {quality_bands.class_a_max_damage_percent:.1f}% → **Class_A**
        - Damage ≥ {quality_bands.defect_min_damage_percent:.1f}% → **Defect**
        - Otherwise → **Class_B**

        Quality damage = Dark cluster % + 50% of Brown cluster %.
        """
    )


uploaded_file = st.file_uploader(
    "Upload one banana image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="kmeans_upload",
)


if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()


prepared_image = prepare_uploaded_image(
    uploaded_file,
    target_size=(416, 416),
)

st.image(
    prepared_image.original_pil,
    caption="Uploaded Banana",
    width=420,
)


if st.button(
    "Run K-means analysis",
    type="primary",
    use_container_width=True,
):

    segmentation = segment_banana(
        rgb_image=prepared_image.working_rgb,
        content_mask=prepared_image.content_mask,
    )

    if not segmentation.success:
        st.error("Banana segmentation failed.")
        st.stop()


    analysis = analyse_kmeans(
        rgb_image=prepared_image.working_rgb,
        banana_mask=segmentation.final_mask,
        parameters=parameters,
        bands=bands,
        quality_bands=quality_bands,
    )


    # Final prediction
    render_result_summary(
        analysis.method_result
    )

    st.info(
        analysis.decision_reason
    )


    # Metric cards
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Banana condition",
        analysis.predicted_category,
    )

    col2.metric(
        "Dominant colour",
        analysis.dominant_colour,
    )

    col3.metric(
        "K-means time",
        f"{analysis.processing_time_ms:.2f} ms",
    )


    # Ripe-only quality
    st.markdown("### Conditional quality assessment")

    if analysis.quality_assessed:
        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Quality class",
            analysis.predicted_quality,
        )

        q2.metric(
            "Quality damage",
            f"{analysis.quality_damage_percent:.2f}%",
        )

        q3.metric(
            "Quality confidence",
            f"{analysis.quality_confidence_percent:.2f}%",
        )

        st.success(analysis.quality_reason)

    else:
        st.info(analysis.quality_reason)


    # Tabs
    overview_tab, clusters_tab, decision_tab = st.tabs(
        [
            "Overview",
            "K-means clusters",
            "Decision details",
        ]
    )


    # OVERVIEW
    with overview_tab:

        first, second = st.columns(2)

        with first:
            st.markdown("#### Banana segmentation")

            st.image(
                segmentation.overlay_rgb,
                use_container_width=True,
            )

        with second:
            st.markdown("#### K-means result")

            st.image(
                analysis.segmentation.segmented_image_rgb,
                use_container_width=True,
            )

        st.caption(
            "The first image shows the detected banana region. "
            "The second image shows the banana pixels grouped "
            "into similar colour clusters using K-means."
        )


    # K-MEANS CLUSTERS
    with clusters_tab:

        st.markdown(
            "#### Detected cluster colours"
        )

        centres = analysis.segmentation.centres_rgb

        columns = st.columns(len(centres))

        for i, centre in enumerate(centres):

            r = int(centre[0])
            g = int(centre[1])
            b = int(centre[2])

            with columns[i]:

                st.markdown(
                    f"**Cluster {i + 1}**"
                )

                st.markdown(
                    f"""
                    <div style="
                        height:80px;
                        border-radius:8px;
                        background-color:rgb({r},{g},{b});
                        border:1px solid gray;
                    ">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"RGB ({r}, {g}, {b})"
                )


    # DECISION DETAILS
    with decision_tab:

        st.write(
            f"**Prediction:** {analysis.predicted_category}"
        )

        st.write(
            f"**Confidence:** {analysis.confidence_percent:.2f}%"
        )

        st.write(
            f"**Dominant colour:** {analysis.dominant_colour}"
        )

        st.write(
            f"**Reason:** {analysis.decision_reason}"
        )

        if analysis.quality_assessed:
            st.write(
                f"**Quality:** {analysis.predicted_quality}"
            )
            st.write(
                f"**Quality damage:** "
                f"{analysis.quality_damage_percent:.2f}%"
            )
            st.write(
                f"**Quality reason:** {analysis.quality_reason}"
            )
        else:
            st.write("**Quality:** Not assessed")

        st.caption(
            "K-means first groups similar banana-peel colours. "
            "The cluster centres are then interpreted to determine "
            "whether the banana is Unripe, Ripe, Overripe or Rotten."
        )
