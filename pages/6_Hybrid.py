from time import perf_counter

import pandas as pd
import streamlit as st

from branches.glcm.glcm_analysis import (
    GLCMParameters,
    GLCMRipenessBands,
    analyse_glcm,
    prepare_quantised_for_display,
)
from branches.hsv.hsv_analysis import (
    HSVRipenessBands,
    analyse_hsv,
)
from branches.hsv.hsv_segmentation import (
    HSVParameters,
)
from branches.kmeans.kmeans_analysis import (
    KMeansRipenessBands,
    analyse_kmeans,
)
from branches.kmeans.kmeans_segmentation import (
    KMeansParameters,
)
from branches.morphology.morphology_analysis import (
    RipenessBands,
    analyse_morphology,
)
from branches.morphology.morphology_segmentation import (
    MorphologyParameters,
)
from core.banana_segmentation import segment_banana
from core.hybrid import (
    CATEGORIES,
    CLASS_RELIABILITY_WEIGHTS,
    METHOD_NAMES,
    METHOD_ORDER,
    combine_method_results,
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
from ui.multiple_image_mode import render_image_input_mode
from ui.result_display import (
    render_result_summary,
)


HYBRID_RESULT_VERSION = 1


apply_app_styles()

page_header(
    "Hybrid Banana Assessment",
    (
        "Combine Morphology, HSV, K-means and GLCM "
        "Texture using class-specific weighted fusion."
    ),
)

st.info(
    "HSV and K-means provide colour evidence for Unripe "
    "and Ripe bananas. Morphology supports widespread dark "
    "region detection for Overripe bananas, while HSV, "
    "Morphology and GLCM contribute deterioration evidence "
    "for Rotten bananas."
)

st.warning(
    "The current hybrid weights are a starting configuration. "
    "Calibrate them with validation data and freeze them before "
    "the final test-set evaluation. Do not tune them using the "
    "test results."
)


render_image_input_mode("hybrid", "hybrid")


with st.expander(
    "View hybrid reliability weights"
):
    weight_rows = []

    for category in CATEGORIES:
        row = {
            "Category": category,
        }

        for method_key in METHOD_ORDER:
            row[
                METHOD_NAMES[method_key]
            ] = (
                CLASS_RELIABILITY_WEIGHTS[
                    category
                ][method_key]
                * 100.0
            )

        weight_rows.append(row)

    st.dataframe(
        pd.DataFrame(weight_rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            METHOD_NAMES[
                method_key
            ]: st.column_config.NumberColumn(
                format="%.1f%%"
            )
            for method_key in METHOD_ORDER
        },
    )


uploaded_file = st.file_uploader(
    "Upload one banana image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    accept_multiple_files=False,
    key="hybrid_upload",
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
    st.error(str(error))
    st.stop()


st.image(
    prepared_image.original_pil,
    caption=uploaded_file.name,
    width=420,
)


fingerprint = image_fingerprint(
    uploaded_file
)


if st.button(
    "Run hybrid assessment",
    type="primary",
    use_container_width=True,
):
    segmentation = None
    analyses = {}
    errors = {}
    hybrid = None

    segmentation_time_ms = 0.0

    try:
        segmentation_start = perf_counter()

        segmentation = segment_banana(
            rgb_image=(
                prepared_image.working_rgb
            ),
            content_mask=(
                prepared_image.content_mask
            ),
        )

        segmentation_time_ms = (
            perf_counter()
            - segmentation_start
        ) * 1000.0

        if not segmentation.success:
            errors["segmentation"] = (
                segmentation.message
            )

        else:
            method_jobs = (
                (
                    "morphology",
                    analyse_morphology,
                    MorphologyParameters(),
                    RipenessBands(),
                ),
                (
                    "hsv",
                    analyse_hsv,
                    HSVParameters(),
                    HSVRipenessBands(),
                ),
                (
                    "kmeans",
                    analyse_kmeans,
                    KMeansParameters(k=4),
                    KMeansRipenessBands(),
                ),
                (
                    "glcm",
                    analyse_glcm,
                    GLCMParameters(),
                    GLCMRipenessBands(),
                ),
            )

            for (
                method_key,
                analyser,
                parameters,
                bands,
            ) in method_jobs:
                try:
                    analyses[method_key] = analyser(
                        rgb_image=(
                            prepared_image.working_rgb
                        ),
                        banana_mask=(
                            segmentation.final_mask
                        ),
                        parameters=parameters,
                        bands=bands,
                    )

                except Exception as error:
                    # One failed branch should not crash
                    # the remaining hybrid pipeline.
                    errors[method_key] = str(error)

            try:
                hybrid = combine_method_results(
                    [
                        analysis.method_result
                        for analysis
                        in analyses.values()
                    ]
                )

            except (
                TypeError,
                ValueError,
            ) as error:
                errors["hybrid"] = str(error)

    except Exception as error:
        errors["segmentation"] = str(error)

    st.session_state[
        "hybrid_result"
    ] = {
        "result_version": (
            HYBRID_RESULT_VERSION
        ),
        "fingerprint": fingerprint,
        "segmentation": segmentation,
        "segmentation_time_ms": (
            segmentation_time_ms
        ),
        "analyses": analyses,
        "hybrid": hybrid,
        "errors": errors,
    }


saved = st.session_state.get(
    "hybrid_result"
)


if (
    saved is None
    or saved.get("result_version")
    != HYBRID_RESULT_VERSION
    or saved.get("fingerprint")
    != fingerprint
):
    st.caption(
        "Press **Run hybrid assessment** to continue."
    )
    st.stop()


segmentation = saved["segmentation"]
analyses = saved["analyses"]
hybrid = saved["hybrid"]
errors = saved["errors"]


if (
    segmentation is None
    or not segmentation.success
):
    st.error(
        errors.get(
            "segmentation",
            "Shared banana segmentation failed.",
        )
    )

    if segmentation is not None:
        st.image(
            segmentation.overlay_rgb,
            caption="Rejected segmentation",
            use_container_width=True,
        )

    st.stop()


if hybrid is None:
    st.error(
        errors.get(
            "hybrid",
            "Hybrid analysis did not return a result.",
        )
    )

    for method_key, message in errors.items():
        if method_key != "hybrid":
            st.caption(
                f"{METHOD_NAMES.get(method_key, method_key)}: "
                f"{message}"
            )

    st.stop()


failed_methods = [
    METHOD_NAMES.get(
        method_key,
        method_key,
    )
    for method_key in errors
    if method_key not in {
        "segmentation",
        "hybrid",
    }
]


if failed_methods:
    st.warning(
        "The following approach failed and was excluded: "
        + ", ".join(failed_methods)
    )


render_result_summary(
    hybrid.method_result
)

st.info(
    hybrid.decision_reason
)

st.caption(
    "Hybrid confidence measures weighted rule support, "
    "method agreement and the winning margin. It is not "
    "a statistical probability."
)


total_time_ms = (
    saved["segmentation_time_ms"]
    + hybrid.processing_time_ms
)


metric_columns = st.columns(4)

metric_columns[0].metric(
    "Final category",
    hybrid.predicted_category,
)

metric_columns[1].metric(
    "Hybrid confidence",
    f"{hybrid.confidence_percent:.2f}%",
)

metric_columns[2].metric(
    "Method agreement",
    (
        f"{hybrid.agreement_count}/"
        f"{hybrid.methods_used}"
    ),
)

metric_columns[3].metric(
    "Total processing time",
    f"{total_time_ms:.2f} ms",
)


morphology_result = analyses.get(
    "morphology"
)


if morphology_result is not None:
    surface_columns = st.columns(2)

    surface_columns[0].metric(
        "Visible dark / blemished area",
        (
            f"{morphology_result.blemish_percentage:.2f}%"
        ),
    )

    surface_columns[1].metric(
        "Surface-quality grade",
        morphology_result.surface_grade,
    )


(
    overview_tab,
    methods_tab,
    evidence_tab,
    decision_tab,
) = st.tabs(
    [
        "Overview",
        "Method comparison",
        "Visual evidence",
        "Decision details",
    ]
)


with overview_tab:
    first_column, second_column = (
        st.columns(2)
    )

    with first_column:
        st.markdown(
            "#### Uploaded image"
        )

        st.image(
            prepared_image.original_pil,
            use_container_width=True,
        )

    with second_column:
        st.markdown(
            "#### Shared banana segmentation"
        )

        st.image(
            segmentation.overlay_rgb,
            use_container_width=True,
        )

    score_dataframe = pd.DataFrame(
        {
            "Hybrid score (%)": {
                category: (
                    hybrid.class_scores[
                        category
                    ]
                    * 100.0
                )
                for category in CATEGORIES
            }
        }
    )

    st.markdown(
        "#### Final class scores"
    )

    st.bar_chart(
        score_dataframe,
        use_container_width=True,
    )


with methods_tab:
    comparison_rows = []

    for method_key in METHOD_ORDER:
        method_result = (
            hybrid.method_results.get(
                method_key
            )
        )

        if method_result is None:
            comparison_rows.append(
                {
                    "Approach": (
                        METHOD_NAMES[
                            method_key
                        ]
                    ),
                    "Prediction": "Failed",
                    "Rule support (%)": None,
                    "Class reliability (%)": None,
                    "Weighted contribution": None,
                }
            )

            continue

        predicted_category = (
            method_result.predicted_category
        )

        comparison_rows.append(
            {
                "Approach": (
                    METHOD_NAMES[
                        method_key
                    ]
                ),
                "Prediction": (
                    predicted_category
                ),
                "Rule support (%)": (
                    method_result.confidence_percent
                ),
                "Class reliability (%)": (
                    hybrid.effective_weights[
                        predicted_category
                    ][method_key]
                    * 100.0
                ),
                "Weighted contribution": (
                    hybrid.method_contributions[
                        method_key
                    ][predicted_category]
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            comparison_rows
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rule support (%)": (
                st.column_config.NumberColumn(
                    format="%.2f"
                )
            ),
            "Class reliability (%)": (
                st.column_config.NumberColumn(
                    format="%.2f"
                )
            ),
            "Weighted contribution": (
                st.column_config.NumberColumn(
                    format="%.4f"
                )
            ),
        },
    )


with evidence_tab:
    first_column, second_column = (
        st.columns(2)
    )

    with first_column:
        st.markdown(
            "#### Morphological dark regions"
        )

        morphology = analyses.get(
            "morphology"
        )

        if morphology is not None:
            st.image(
                morphology.masks.blemish_overlay_rgb,
                use_container_width=True,
            )
        else:
            st.info(
                "Morphology result unavailable."
            )

    with second_column:
        st.markdown(
            "#### HSV peel regions"
        )

        hsv = analyses.get(
            "hsv"
        )

        if hsv is not None:
            st.image(
                hsv.masks.colour_overlay_rgb,
                use_container_width=True,
            )
        else:
            st.info(
                "HSV result unavailable."
            )

    third_column, fourth_column = (
        st.columns(2)
    )

    with third_column:
        st.markdown(
            "#### K-means clusters"
        )

        kmeans = analyses.get(
            "kmeans"
        )

        if kmeans is not None:
            st.image(
                (
                    kmeans
                    .segmentation
                    .segmented_image_rgb
                ),
                use_container_width=True,
            )
        else:
            st.info(
                "K-means result unavailable."
            )

    with fourth_column:
        st.markdown(
            "#### GLCM quantised image"
        )

        glcm = analyses.get(
            "glcm"
        )

        if glcm is not None:
            st.image(
                prepare_quantised_for_display(
                    glcm.quantised_image
                ),
                clamp=True,
                use_container_width=True,
            )
        else:
            st.info(
                "GLCM result unavailable."
            )


with decision_tab:
    decision_rows = [
        {
            "Category": category,
            "Raw weighted score": (
                hybrid.raw_class_scores[
                    category
                ]
            ),
            "Normalised score (%)": (
                hybrid.class_scores[
                    category
                ]
                * 100.0
            ),
        }
        for category in CATEGORIES
    ]

    st.dataframe(
        pd.DataFrame(
            decision_rows
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Raw weighted score": (
                st.column_config.NumberColumn(
                    format="%.4f"
                )
            ),
            "Normalised score (%)": (
                st.column_config.NumberColumn(
                    format="%.2f"
                )
            ),
        },
    )

    st.write(
        hybrid.decision_reason
    )

    st.caption(
        f"Winning margin: "
        f"{hybrid.winning_margin_percent:.2f}% · "
        f"Fusion used {hybrid.methods_used} approaches."
    )
