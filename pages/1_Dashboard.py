from numbers import Real

import pandas as pd
import plotly.express as px
import streamlit as st

from core.evaluation import (
    EVALUATION_MODE_QUALITY,
    EVALUATION_MODE_RIPENESS,
    METHODS,
    EvaluationError,
    load_latest_evaluation,
    run_fixed_dataset_evaluation,
)
from core.result_schema import QUALITY_CATEGORIES, RIPENESS_CATEGORIES
from ui.components import apply_app_styles


PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}
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
    return [key for key in METHODS if key in methods and _is_number(methods[key].get("overall_accuracy"))]


def _overall_dataframe(report: dict, method_keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Approach": METHODS[key],
            "Overall accuracy": report["methods"][key].get("overall_accuracy"),
            "Macro F1": report["methods"][key].get("macro_f1"),
        }
        for key in method_keys
    )


def _class_dataframe(report: dict, method_keys: list[str], categories: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for key in method_keys:
        for category in categories:
            metrics = report["methods"][key].get("per_class", {}).get(category, {})
            rows.append(
                {
                    "Approach": METHODS[key],
                    "Category": category,
                    "Precision": metrics.get("precision"),
                    "Recall": metrics.get("recall"),
                    "F1": metrics.get("f1"),
                }
            )
    return pd.DataFrame(rows)


def _processing_time_dataframe(
    report: dict,
    method_keys: list[str],
) -> pd.DataFrame:
    """Return the reported average end-to-end time for each approach."""
    return pd.DataFrame(
        {
            "Approach": METHODS[key],
            "Average processing time (ms)": report["methods"][key].get(
                "average_processing_time_ms"
            ),
        }
        for key in method_keys
        if _is_number(
            report["methods"][key].get("average_processing_time_ms")
        )
    )


def _style_figure(figure, height: int = 400):
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=25, b=10),
        legend_title_text="",
        font=dict(size=14),
    )
    return figure


def _render_evaluation_report(report: dict, mode_label: str, evaluation_mode: str) -> None:
    """Expose one report for the complete dashboard evaluation."""

    with st.container(border=True):
        st.subheader("Automated Evaluation Reporting")
        st.write(
            f"Download one PDF that documents the complete **{mode_label.lower()} "
            "evaluation** and compares every evaluated image-processing approach."
        )

        try:
            from core.reporting import build_evaluation_pdf_report

            pdf_bytes = build_evaluation_pdf_report(report)
        except ModuleNotFoundError as error:
            if error.name == "reportlab":
                st.error(
                    "PDF reporting requires ReportLab. Install the project "
                    "dependencies with `./.venv/bin/python -m pip install -r "
                    "requirements.txt`, then restart Streamlit."
                )
            else:
                raise
        except Exception as error:
            st.error(f"The evaluation PDF could not be generated: {error}")
        else:
            st.download_button(
                "Download dashboard evaluation report (PDF)",
                data=pdf_bytes,
                file_name=f"banana_{evaluation_mode}_evaluation_report.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
                key=f"{evaluation_mode}_evaluation_pdf_report",
            )
            st.caption(
                "The report includes the dataset summary, class distribution, "
                "approach comparison, automated findings, per-class metrics and "
                "confusion matrices."
            )

apply_app_styles()
st.title("Evaluation dashboard")

mode_label = st.radio(
    "Evaluation mode",
    options=("Ripeness", "Quality"),
    horizontal=True,
    key="evaluation_mode_selector",
)
evaluation_mode = EVALUATION_MODE_QUALITY if mode_label == "Quality" else EVALUATION_MODE_RIPENESS
categories = QUALITY_CATEGORIES if evaluation_mode == EVALUATION_MODE_QUALITY else RIPENESS_CATEGORIES
class_axis_title = "Quality class" if evaluation_mode == EVALUATION_MODE_QUALITY else "Ripeness category"

report_state_key = f"evaluation_report_{evaluation_mode}"
if report_state_key not in st.session_state:
    st.session_state[report_state_key] = load_latest_evaluation(mode=evaluation_mode)
report = st.session_state[report_state_key]

summary_columns = st.columns(3)
summary_columns[0].metric("Evaluation mode", mode_label)
summary_columns[1].metric("Images", int(report.get("image_count", 0)))
summary_columns[2].metric("Methods evaluated", len(_evaluated_method_keys(report)))

button_label = "Run quality evaluation" if evaluation_mode == EVALUATION_MODE_QUALITY else "Run ripeness evaluation"
spinner_label = "Evaluating quality..." if evaluation_mode == EVALUATION_MODE_QUALITY else "Evaluating ripeness..."

if st.button(button_label, type="primary", width="stretch"):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_progress(current: int, total: int, message: str) -> None:
        progress_bar.progress(int(current / max(total, 1) * 100))
        progress_text.caption(message)

    try:
        with st.spinner(spinner_label):
            report = run_fixed_dataset_evaluation(progress_callback=update_progress, mode=evaluation_mode)
        st.session_state[report_state_key] = report
        progress_bar.progress(100)
        progress_text.success("Evaluation completed and saved.")
    except EvaluationError as error:
        progress_text.empty()
        st.error(str(error))
    except Exception as error:
        progress_text.empty()
        st.error(f"Evaluation stopped unexpectedly: {error}")
        st.exception(error)

report = st.session_state[report_state_key]
evaluated_keys = _evaluated_method_keys(report)
if not evaluated_keys:
    st.info(f"No completed {mode_label.lower()} evaluation is available. Run the evaluation to generate results.")
    st.stop()

overall = _overall_dataframe(report, evaluated_keys)
per_class = _class_dataframe(report, evaluated_keys, categories)
processing_time = _processing_time_dataframe(report, evaluated_keys)

st.divider()
_render_evaluation_report(report, mode_label, evaluation_mode)

st.divider()
st.subheader("Approach results")

default_index = evaluated_keys.index("morphology") if "morphology" in evaluated_keys else 0
selected_key = st.selectbox(
    "Approach",
    options=evaluated_keys,
    index=default_index,
    format_func=lambda key: METHODS[key],
)
selected_name = METHODS[selected_key]
selected_result = report["methods"][selected_key]
selected_metrics = st.columns(3)
selected_metrics[0].metric("Overall accuracy", f"{selected_result['overall_accuracy']:.1%}")
selected_metrics[1].metric("Macro F1", f"{selected_result['macro_f1']:.1%}")
selected_metrics[2].metric("Images processed", int(selected_result.get("successful_images") or 0))

selected_rows = per_class[per_class["Approach"] == selected_name]
detail_long = selected_rows.melt(
    id_vars=["Approach", "Category"],
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
    category_orders={"Category": list(categories), "Metric": ["Precision", "Recall", "F1"]},
)
detail_figure.update_traces(texttemplate="%{y:.2f}", textposition="outside", cliponaxis=False)
detail_figure.update_yaxes(title="Score", range=[0, 1.08], tickformat=".0%")
detail_figure.update_xaxes(title=None)
st.plotly_chart(_style_figure(detail_figure), width="stretch", config=PLOT_CONFIG)

st.divider()
st.subheader("Approach comparison")

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
    category_orders={"Approach": [METHODS[key] for key in evaluated_keys], "Metric": ["Overall accuracy", "Macro F1"]},
)
performance_figure.update_traces(texttemplate="%{y:.1%}", textposition="outside", cliponaxis=False)
performance_figure.update_yaxes(title="Score", range=[0, 1.08], tickformat=".0%")
performance_figure.update_xaxes(title=None)
st.plotly_chart(_style_figure(performance_figure), width="stretch", config=PLOT_CONFIG)

st.markdown("#### Processing time comparison")
if processing_time.empty:
    st.info("Processing-time data is not available for this evaluation yet.")
else:
    st.caption(
        "Average end-to-end time per image, including shared preprocessing "
        "and banana segmentation."
    )
    st.bar_chart(
        processing_time,
        x="Approach",
        y="Average processing time (ms)",
        horizontal=True,
        sort="-Average processing time (ms)",
        color="orange",
        height=max(260, 58 * len(processing_time)),
    )

st.markdown("#### Per-class F1-score comparison")
st.caption(
    "F1-score comparison across approaches for each "
    f"{class_axis_title.lower()}."
)

method_order = [METHODS[key] for key in evaluated_keys]
f1_matrix = per_class.pivot(index="Approach", columns="Category", values="F1").reindex(index=method_order, columns=list(categories))
f1_figure = px.imshow(
    f1_matrix,
    zmin=0,
    zmax=1,
    text_auto=".3f",
    aspect="auto",
    color_continuous_scale="YlGn",
    labels={"x": class_axis_title, "y": "Approach", "color": "F1"},
)
f1_figure.update_xaxes(side="top")
st.plotly_chart(_style_figure(f1_figure, height=max(330, 74 * len(method_order))), width="stretch", config=PLOT_CONFIG)
