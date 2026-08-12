import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.evaluation import (
    CATEGORIES,
    METHODS,
    PROJECT_ROOT,
    EvaluationError,
    load_latest_evaluation,
    run_fixed_dataset_evaluation,
)
from ui.components import apply_app_styles, page_header


apply_app_styles()

page_header(
    "Evaluation Dashboard",
    (
        "Display the latest fixed test-set results for all approaches. "
        "Only implemented approaches contain evaluation values."
    ),
)

st.warning(
    "Do not adjust thresholds after inspecting test-set results. "
    "Threshold calibration must use dataset/valid; dataset/test should "
    "be reserved for final evaluation."
)

if "evaluation_report" not in st.session_state:
    st.session_state["evaluation_report"] = (
        load_latest_evaluation()
    )

report = st.session_state["evaluation_report"]

top_columns = st.columns(4)

top_columns[0].metric(
    "Dataset split",
    report["dataset_split"],
)

top_columns[1].metric(
    "Test images",
    report["image_count"],
)

top_columns[2].metric(
    "Completed images",
    report["successful_images"],
)

top_columns[3].metric(
    "Failed images",
    report["failed_images"],
)

st.caption(
    f"Last evaluation: "
    f"{report['generated_at'] or 'Never'}"
)

st.caption(report["status"])

if st.button(
    "Re-run fixed test-set evaluation",
    type="primary",
    use_container_width=True,
):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_progress(
        current: int,
        total: int,
        message: str,
    ) -> None:
        percentage = int(
            current / max(1, total) * 100
        )

        progress_bar.progress(percentage)
        progress_text.caption(message)

    try:
        with st.spinner(
            "Evaluating the fixed test set..."
        ):
            report = run_fixed_dataset_evaluation(
                progress_callback=update_progress
            )

        st.session_state["evaluation_report"] = report

        progress_bar.progress(100)
        progress_text.success(
            "Evaluation completed and saved."
        )

    except EvaluationError as error:
        progress_text.empty()
        st.error(str(error))

    except Exception as error:
        progress_text.empty()
        st.error(
            f"Evaluation stopped unexpectedly: {error}"
        )
        st.exception(error)

report = st.session_state["evaluation_report"]

st.divider()
st.subheader("Overall approach comparison")

overall_rows = []

for method_key, method_name in METHODS.items():
    method = report["methods"][method_key]

    overall_rows.append(
        {
            "Approach": method_name,
            "Status": method["status"],
            "Overall accuracy": (
                method["overall_accuracy"]
            ),
            "Macro F1": method["macro_f1"],
            "Average processing time (ms)": (
                method[
                    "average_processing_time_ms"
                ]
            ),
            "Successful images": (
                method["successful_images"]
            ),
            "Failed images": (
                method["failed_images"]
            ),
        }
    )

overall_dataframe = pd.DataFrame(
    overall_rows
)

st.dataframe(
    overall_dataframe,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Overall accuracy": (
            st.column_config.NumberColumn(
                "Overall accuracy",
                format="%.4f",
            )
        ),
        "Macro F1": (
            st.column_config.NumberColumn(
                "Macro F1",
                format="%.4f",
            )
        ),
        "Average processing time (ms)": (
            st.column_config.NumberColumn(
                "Average processing time (ms)",
                format="%.2f",
            )
        ),
        "Successful images": (
            st.column_config.NumberColumn(
                "Successful images",
                format="%d",
            )
        ),
        "Failed images": (
            st.column_config.NumberColumn(
                "Failed images",
                format="%d",
            )
        ),
    },
)

st.subheader("Per-class evaluation")

class_rows = []

for method_key, method_name in METHODS.items():
    method = report["methods"][method_key]

    for category in CATEGORIES:
        category_metrics = method[
            "per_class"
        ][category]

        class_rows.append(
            {
                "Approach": method_name,
                "Category": category,
                "Precision": (
                    category_metrics["precision"]
                ),
                "Recall": (
                    category_metrics["recall"]
                ),
                "F1": category_metrics["f1"],
                "Support": (
                    category_metrics["support"]
                ),
            }
        )

class_dataframe = pd.DataFrame(
    class_rows
)

st.dataframe(
    class_dataframe,
    hide_index=True,
    use_container_width=True,
    height=650,
    column_config={
        "Precision": (
            st.column_config.NumberColumn(
                "Precision",
                format="%.4f",
            )
        ),
        "Recall": (
            st.column_config.NumberColumn(
                "Recall",
                format="%.4f",
            )
        ),
        "F1": (
            st.column_config.NumberColumn(
                "F1",
                format="%.4f",
            )
        ),
        "Support": (
            st.column_config.NumberColumn(
                "Support",
                format="%d",
            )
        ),
    },
)

st.subheader("Confusion matrix")

selected_method_key = st.selectbox(
    "Select an approach",
    options=list(METHODS.keys()),
    format_func=lambda key: METHODS[key],
)

selected_method = report["methods"][
    selected_method_key
]

matrix_information = selected_method[
    "confusion_matrix"
]

if matrix_information is None:
    st.info(
        f"{METHODS[selected_method_key]} has not been "
        "implemented or evaluated."
    )
else:
    matrix_dataframe = pd.DataFrame(
        matrix_information["values"],
        index=[
            f"Actual {label}"
            for label in matrix_information[
                "actual_labels"
            ]
        ],
        columns=[
            f"Predicted {label}"
            for label in matrix_information[
                "predicted_labels"
            ]
        ],
    )

    st.dataframe(
        matrix_dataframe,
        use_container_width=True,
    )

    st.caption(
        "The Failed column contains images where preprocessing, "
        "banana segmentation or morphology analysis could not return "
        "a prediction."
    )

st.subheader("Dataset composition")

composition_rows = [
    {
        "Category": category,
        "Test images": report[
            "class_counts"
        ].get(category, 0),
    }
    for category in CATEGORIES
]

st.dataframe(
    pd.DataFrame(composition_rows),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Download results")

download_columns = st.columns(2)

with download_columns[0]:
    st.download_button(
        "Download evaluation report",
        data=json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="latest_evaluation.json",
        mime="application/json",
        use_container_width=True,
    )

with download_columns[1]:
    predictions_file = report.get(
        "predictions_file"
    )

    if predictions_file:
        predictions_path = (
            PROJECT_ROOT
            / Path(predictions_file)
        )

        if predictions_path.exists():
            st.download_button(
                "Download image predictions",
                data=predictions_path.read_bytes(),
                file_name=(
                    "morphology_predictions.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button(
                "Predictions file unavailable",
                disabled=True,
                use_container_width=True,
            )
    else:
        st.button(
            "No predictions generated",
            disabled=True,
            use_container_width=True,
        )