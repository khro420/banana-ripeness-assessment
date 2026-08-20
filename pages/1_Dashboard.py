from datetime import datetime
from numbers import Real

import pandas as pd
import plotly.express as px
import streamlit as st

from core.evaluation import (
    CATEGORIES,
    METHODS,
    EvaluationError,
    load_latest_evaluation,
    run_fixed_dataset_evaluation,
)
from ui.components import apply_app_styles, page_header


PLOT_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}

METHOD_COLOURS = {
    "Morphology": "#8C6D31",
    "HSV": "#2E8B57",
    "K-means": "#3B82F6",
    "GLCM Texture": "#8B5CF6",
    "Hybrid": "#D97706",
}

METRIC_COLOURS = {
    "Overall accuracy": "#2563EB",
    "Macro F1": "#D97706",
    "Precision": "#2563EB",
    "Recall": "#16A34A",
    "F1": "#D97706",
}


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not pd.isna(value)


def _evaluated_method_keys(report: dict) -> list[str]:
    methods = report.get("methods", {})
    return [
        key
        for key in METHODS
        if key in methods
        and _is_number(methods[key].get("overall_accuracy"))
    ]


def _overall_dataframe(
    report: dict,
    method_keys: list[str],
) -> pd.DataFrame:
    rows = []

    for key in method_keys:
        result = report["methods"][key]
        rows.append(
            {
                "Approach": METHODS[key],
                "Status": result.get("status"),
                "Overall accuracy": result.get("overall_accuracy"),
                "Macro F1": result.get("macro_f1"),
                "Average processing time (ms)": result.get(
                    "average_processing_time_ms"
                ),
                "Successful images": result.get("successful_images"),
                "Failed images": result.get("failed_images"),
            }
        )

    return pd.DataFrame(rows)


def _class_dataframe(
    report: dict,
    method_keys: list[str],
) -> pd.DataFrame:
    rows = []

    for key in method_keys:
        per_class = report["methods"][key].get("per_class", {})

        for category in CATEGORIES:
            metrics = per_class.get(category, {})
            rows.append(
                {
                    "Approach": METHODS[key],
                    "Category": category,
                    "Precision": metrics.get("precision"),
                    "Recall": metrics.get("recall"),
                    "F1": metrics.get("f1"),
                    "Support": metrics.get("support"),
                }
            )

    return pd.DataFrame(rows)


def _style_figure(figure, height: int = 430):
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=25, b=10),
        legend_title_text="",
        font=dict(size=14),
    )
    return figure


def _format_generated_at(value: str | None) -> str:
    if not value:
        return "Never"

    try:
        generated = datetime.fromisoformat(value)
        return generated.strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return value


apply_app_styles()

page_header(
    "Evaluation Dashboard",
    (
        "Definitive fixed test-set results for the four individual "
        "approaches and the integrated hybrid approach."
    ),
)

st.warning(
    "Final-test results are for reporting only. Do not modify thresholds or "
    "hybrid weights after viewing them; calibration belongs to dataset/valid."
)

if "evaluation_report" not in st.session_state:
    st.session_state["evaluation_report"] = load_latest_evaluation()

report = st.session_state["evaluation_report"]
evaluated_keys = _evaluated_method_keys(report)

best_key = None
if evaluated_keys:
    best_key = max(
        evaluated_keys,
        key=lambda key: report["methods"][key]["overall_accuracy"],
    )

summary_columns = st.columns(4)
summary_columns[0].metric(
    "Dataset split",
    str(report.get("dataset_split", "test")).title(),
)
summary_columns[1].metric(
    "Test images",
    int(report.get("image_count", 0)),
)
summary_columns[2].metric(
    "Best approach",
    METHODS[best_key] if best_key else "Not evaluated",
)
summary_columns[3].metric(
    "Best accuracy",
    (
        f"{report['methods'][best_key]['overall_accuracy']:.2%}"
        if best_key
        else "—"
    ),
)

st.caption(
    "Last evaluation: "
    f"{_format_generated_at(report.get('generated_at'))} · "
    f"{report.get('status', 'No evaluation status available.')}"
)

if st.button(
    "Re-run fixed test-set evaluation",
    type="primary",
    use_container_width=True,
):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_progress(current: int, total: int, message: str) -> None:
        progress_bar.progress(int(current / max(total, 1) * 100))
        progress_text.caption(message)

    try:
        with st.spinner("Evaluating the fixed test set..."):
            report = run_fixed_dataset_evaluation(
                progress_callback=update_progress
            )

        st.session_state["evaluation_report"] = report
        progress_bar.progress(100)
        progress_text.success("Evaluation completed and saved.")
        evaluated_keys = _evaluated_method_keys(report)

    except EvaluationError as error:
        progress_text.empty()
        st.error(str(error))

    except Exception as error:
        progress_text.empty()
        st.error(f"Evaluation stopped unexpectedly: {error}")
        st.exception(error)

report = st.session_state["evaluation_report"]
evaluated_keys = _evaluated_method_keys(report)

if not evaluated_keys:
    st.info(
        "No completed evaluation is available. Run the fixed test-set "
        "evaluation to generate the result visualisations."
    )
    st.stop()

overall = _overall_dataframe(report, evaluated_keys)
per_class = _class_dataframe(report, evaluated_keys)

st.divider()
st.subheader("Overall performance")

performance_long = overall.melt(
    id_vars="Approach",
    value_vars=["Overall accuracy", "Macro F1"],
    var_name="Metric",
    value_name="Score",
)

performance_figure = px.bar(
    performance_long,
    x="Approach",
    y="Score",
    color="Metric",
    barmode="group",
    color_discrete_map=METRIC_COLOURS,
    category_orders={
        "Approach": [METHODS[key] for key in evaluated_keys],
        "Metric": ["Overall accuracy", "Macro F1"],
    },
)
performance_figure.update_traces(
    texttemplate="%{y:.1%}",
    textposition="outside",
    cliponaxis=False,
)
performance_figure.update_yaxes(
    title="Score",
    range=[0, 1.08],
    tickformat=".0%",
)
performance_figure.update_xaxes(title=None)
_style_figure(performance_figure)

st.plotly_chart(
    performance_figure,
    use_container_width=True,
    config=PLOT_CONFIG,
)

st.caption(
    "Overall accuracy measures the proportion of correctly classified test "
    "images. Macro F1 gives equal importance to all four ripeness classes."
)

st.subheader("Per-class F1 comparison")

method_order = [METHODS[key] for key in evaluated_keys]
f1_matrix = (
    per_class.pivot(index="Approach", columns="Category", values="F1")
    .reindex(index=method_order, columns=list(CATEGORIES))
)

f1_figure = px.imshow(
    f1_matrix,
    zmin=0,
    zmax=1,
    text_auto=".3f",
    aspect="auto",
    color_continuous_scale="YlGn",
    labels={
        "x": "Ripeness category",
        "y": "Approach",
        "color": "F1",
    },
)
f1_figure.update_xaxes(side="top")
_style_figure(f1_figure, height=max(360, 78 * len(method_order)))

st.plotly_chart(
    f1_figure,
    use_container_width=True,
    config=PLOT_CONFIG,
)

st.caption(
    "The heatmap exposes class-specific strengths and weaknesses that a "
    "single overall score would hide."
)

st.subheader("Detailed approach results")

default_detail_index = (
    evaluated_keys.index("hybrid")
    if "hybrid" in evaluated_keys
    else 0
)

selected_key = st.selectbox(
    "Approach",
    options=evaluated_keys,
    index=default_detail_index,
    format_func=lambda key: METHODS[key],
)

selected_name = METHODS[selected_key]
selected_rows = per_class[per_class["Approach"] == selected_name]
detail_long = selected_rows.melt(
    id_vars=["Approach", "Category", "Support"],
    value_vars=["Precision", "Recall", "F1"],
    var_name="Metric",
    value_name="Score",
)

detail_figure = px.bar(
    detail_long,
    x="Category",
    y="Score",
    color="Metric",
    barmode="group",
    color_discrete_map=METRIC_COLOURS,
    category_orders={
        "Category": list(CATEGORIES),
        "Metric": ["Precision", "Recall", "F1"],
    },
)
detail_figure.update_traces(
    texttemplate="%{y:.2f}",
    textposition="outside",
    cliponaxis=False,
)
detail_figure.update_yaxes(
    title="Score",
    range=[0, 1.08],
    tickformat=".0%",
)
detail_figure.update_xaxes(title=None)
_style_figure(detail_figure)

st.plotly_chart(
    detail_figure,
    use_container_width=True,
    config=PLOT_CONFIG,
)

st.subheader("Confusion matrix")

matrix_information = report["methods"][selected_key].get("confusion_matrix")

if matrix_information is None:
    st.info(f"No confusion matrix is available for {selected_name}.")
else:
    predicted_labels = list(matrix_information["predicted_labels"])
    values = matrix_information["values"]

    if (
        "Failed" in predicted_labels
        and all(row[predicted_labels.index("Failed")] == 0 for row in values)
    ):
        failed_index = predicted_labels.index("Failed")
        predicted_labels.pop(failed_index)
        values = [
            row[:failed_index] + row[failed_index + 1 :]
            for row in values
        ]

    matrix_dataframe = pd.DataFrame(
        values,
        index=matrix_information["actual_labels"],
        columns=predicted_labels,
    )

    matrix_figure = px.imshow(
        matrix_dataframe,
        text_auto="d",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted category",
            "y": "Actual category",
            "color": "Images",
        },
    )
    matrix_figure.update_xaxes(side="top")
    _style_figure(matrix_figure, height=470)

    st.plotly_chart(
        matrix_figure,
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    st.caption(
        "Diagonal cells are correct predictions. Off-diagonal cells show "
        "which ripeness categories were confused with one another."
    )

st.subheader("Average processing time")

time_data = overall.dropna(subset=["Average processing time (ms)"]).copy()
time_data = time_data.sort_values("Average processing time (ms)")

time_figure = px.bar(
    time_data,
    x="Approach",
    y="Average processing time (ms)",
    color="Approach",
    color_discrete_map=METHOD_COLOURS,
    category_orders={"Approach": time_data["Approach"].tolist()},
)
time_figure.update_traces(
    texttemplate="%{y:.0f} ms",
    textposition="outside",
    cliponaxis=False,
)
time_figure.update_layout(showlegend=False)
time_figure.update_xaxes(title=None)
time_figure.update_yaxes(title="Average processing time (ms)")
_style_figure(time_figure)

st.plotly_chart(
    time_figure,
    use_container_width=True,
    config=PLOT_CONFIG,
)

support_text = " · ".join(
    f"{category}: {report.get('class_counts', {}).get(category, 0)}"
    for category in CATEGORIES
)
st.caption(f"Fixed test-set composition — {support_text}")

with st.expander("View exact evaluation tables"):
    st.markdown("#### Overall metrics")
    st.dataframe(
        overall,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Overall accuracy": st.column_config.NumberColumn(
                "Overall accuracy", format="%.4f"
            ),
            "Macro F1": st.column_config.NumberColumn(
                "Macro F1", format="%.4f"
            ),
            "Average processing time (ms)": st.column_config.NumberColumn(
                "Average processing time (ms)", format="%.2f"
            ),
            "Successful images": st.column_config.NumberColumn(
                "Successful images", format="%d"
            ),
            "Failed images": st.column_config.NumberColumn(
                "Failed images", format="%d"
            ),
        },
    )

    st.markdown("#### Per-class metrics")
    st.dataframe(
        per_class,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Precision": st.column_config.NumberColumn(
                "Precision", format="%.4f"
            ),
            "Recall": st.column_config.NumberColumn(
                "Recall", format="%.4f"
            ),
            "F1": st.column_config.NumberColumn("F1", format="%.4f"),
            "Support": st.column_config.NumberColumn(
                "Support", format="%d"
            ),
        },
    )