import numpy as np
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
    "Classify ripeness, then grade quality only when the result is Ripe.",
)

render_image_input_mode("kmeans", "kmeans")

parameters = KMeansParameters(k=4)
bands = KMeansRipenessBands()
quality_bands = KMeansQualityBands()

with st.expander("View K-means decision rules"):
    st.markdown(
        f"""
        Rules are checked in this order:

        1. **Unripe:** green cluster coverage ≥
           {bands.unripe_min_green_percent:.1f}% and brown coverage ≤
           {bands.unripe_max_brown_percent:.1f}%.

        2. **Primary Ripe:** yellow cluster coverage ≥
           {bands.ripe_min_yellow_percent:.1f}% and brown coverage ≤
           {bands.ripe_max_brown_percent:.1f}%.

        3. **Overripe:** dark cluster coverage ≥
           {bands.overripe_min_dark_percent:.1f}% and unclassified coverage ≤
           {bands.overripe_max_other_percent:.1f}%.

        4. **Secondary Ripe:** yellow cluster coverage ≥
           {bands.ripe_secondary_min_yellow_percent:.1f}% and brown coverage ≤
           {bands.ripe_secondary_max_brown_percent:.1f}%.

        5. **Rotten:** fallback for the remaining mixed, deteriorated colour
           patterns.

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
    overview_tab, mask_tab, clusters_tab, decision_tab = st.tabs(
        [
            "Overview",
            "K-means masks",
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

    # K-MEANS CLUSTER MASKS
    with mask_tab:

        st.markdown("#### K-means cluster masks")

        st.caption(
            "White pixels belong to the selected K-means cluster; "
            "black pixels belong to other clusters or the background."
        )

        cluster_map = analysis.segmentation.cluster_map
        centres = analysis.segmentation.centres_rgb
        percentages = analysis.segmentation.cluster_percentages

        for row_start in range(0, len(centres), 2):
            mask_columns = st.columns(2)

            for column_index, cluster_index in enumerate(
                range(row_start, min(row_start + 2, len(centres)))
            ):
                cluster_mask = np.where(
                    cluster_map == cluster_index,
                    255,
                    0,
                ).astype(np.uint8)

                centre = centres[cluster_index]
                centre_rgb = tuple(int(value) for value in centre)

                with mask_columns[column_index]:
                    st.image(
                        cluster_mask,
                        caption=(
                            f"Cluster {cluster_index + 1} mask — "
                            f"{percentages[cluster_index]:.2f}% of banana pixels; "
                            f"centre RGB {centre_rgb}"
                        ),
                        use_container_width=True,
                    )

        with st.expander("View shared binary banana mask"):
            st.image(
                segmentation.final_mask,
                caption=(
                    "White pixels represent the detected banana region; "
                    "black pixels represent the excluded background."
                ),
                use_container_width=True,
            )

            st.caption(
                "This shared mask limits K-means clustering to banana pixels."
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