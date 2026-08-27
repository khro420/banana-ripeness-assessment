from __future__ import annotations

from datetime import datetime
from io import BytesIO
from numbers import Real
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.result_schema import METHOD_NAMES


REPORT_TITLE = "Banana Ripeness Assessment - Dashboard Evaluation Report"
BRAND_GREEN = colors.HexColor("#2E7D32")
BRAND_YELLOW = colors.HexColor("#F9A825")
LIGHT_GREEN = colors.HexColor("#E8F5E9")
LIGHT_GREY = colors.HexColor("#F5F5F5")
TEXT_GREY = colors.HexColor("#455A64")
GRID_GREY = colors.HexColor("#CFD8DC")
MODE_LABELS = {"ripeness": "Ripeness", "quality": "Quality"}


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _percent(value: Any, decimals: int = 1) -> str:
    if not _is_number(value):
        return "-"
    return f"{float(value):.{decimals}%}"


def _number(value: Any, suffix: str = "", decimals: int = 2) -> str:
    if not _is_number(value):
        return "-"
    return f"{float(value):.{decimals}f}{suffix}"


def _timestamp(value: Any) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return str(value)
    if parsed.tzinfo is not None:
        return parsed.strftime("%d %B %Y, %H:%M %Z")
    return parsed.strftime("%d %B %Y, %H:%M")


def _footer(canvas, document) -> None:
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#D7E3D8"))
    canvas.line(15 * mm, 12 * mm, width - 15 * mm, 12 * mm)
    canvas.setFillColor(TEXT_GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(15 * mm, 7.5 * mm, "Banana Ripeness Assessment")
    canvas.drawRightString(width - 15 * mm, 7.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def _report_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=26,
            textColor=BRAND_GREEN,
            alignment=TA_CENTER,
            spaceAfter=4 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=TEXT_GREY,
            alignment=TA_CENTER,
            spaceAfter=3 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=BRAND_GREEN,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MethodTitle",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=TEXT_GREY,
            spaceBefore=2.5 * mm,
            spaceAfter=1.5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=TEXT_GREY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Finding",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            leftIndent=5 * mm,
            firstLineIndent=-3 * mm,
            spaceAfter=0.8 * mm,
        )
    )
    return styles


def _styled_table(
    data: list[list[Any]],
    col_widths: list[float],
    *,
    header_color=BRAND_GREEN,
    font_size: float = 8.5,
) -> Table:
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ("GRID", (0, 0), (-1, -1), 0.25, GRID_GREY),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _evaluated_methods(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    methods = report.get("methods", {})
    return [
        (key, methods[key])
        for key in METHOD_NAMES
        if key in methods and _is_number(methods[key].get("overall_accuracy"))
    ]


def _performance_chart(
    evaluated: list[tuple[str, dict[str, Any]]],
) -> Drawing:
    drawing = Drawing(720, 170)
    chart = VerticalBarChart()
    chart.x = 58
    chart.y = 30
    chart.height = 100
    chart.width = 625
    chart.data = [
        [float(method["overall_accuracy"]) * 100 for _, method in evaluated],
        [float(method["macro_f1"]) * 100 for _, method in evaluated],
    ]
    chart.categoryAxis.categoryNames = [
        str(method.get("approach") or METHOD_NAMES[key])
        for key, method in evaluated
    ]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 20
    chart.valueAxis.labels.fontSize = 8
    chart.valueAxis.labelTextFormat = "%d%%"
    chart.bars[0].fillColor = BRAND_GREEN
    chart.bars[0].strokeColor = BRAND_GREEN
    chart.bars[1].fillColor = BRAND_YELLOW
    chart.bars[1].strokeColor = BRAND_YELLOW
    chart.barWidth = 22
    chart.barSpacing = 2
    chart.groupSpacing = 10
    drawing.add(chart)

    legend = Legend()
    legend.x = 245
    legend.y = 160
    legend.fontName = "Helvetica"
    legend.fontSize = 8.5
    legend.dx = 9
    legend.dy = 9
    legend.deltax = 120
    legend.colorNamePairs = [
        (BRAND_GREEN, "Overall accuracy"),
        (BRAND_YELLOW, "Macro F1"),
    ]
    drawing.add(legend)
    return drawing


def _automated_findings(
    report: dict[str, Any],
    evaluated: list[tuple[str, dict[str, Any]]],
    categories: list[str],
) -> list[str]:
    image_count = int(report.get("image_count") or 0)
    successful = int(report.get("successful_images") or 0)
    failed = int(report.get("failed_images") or 0)

    best_accuracy_key, best_accuracy = max(
        evaluated,
        key=lambda item: float(item[1]["overall_accuracy"]),
    )
    best_f1_key, best_f1 = max(
        evaluated,
        key=lambda item: float(item[1]["macro_f1"]),
    )
    timed = [
        item
        for item in evaluated
        if _is_number(item[1].get("average_processing_time_ms"))
    ]

    findings = [
        f"The fixed test-set evaluation contained {image_count} images; {successful} completed across all implemented approaches and {failed} failed.",
        f"{best_accuracy.get('approach') or METHOD_NAMES[best_accuracy_key]} achieved the highest overall accuracy at {_percent(best_accuracy['overall_accuracy'])}.",
        f"{best_f1.get('approach') or METHOD_NAMES[best_f1_key]} achieved the highest macro F1 score at {_percent(best_f1['macro_f1'])}.",
    ]

    if timed:
        fastest_key, fastest = min(
            timed,
            key=lambda item: float(item[1]["average_processing_time_ms"]),
        )
        findings.append(
            f"{fastest.get('approach') or METHOD_NAMES[fastest_key]} had the lowest average processing time at {_number(fastest['average_processing_time_ms'], ' ms')} per image."
        )

    for category in categories:
        class_candidates = []
        for key, method in evaluated:
            f1_value = method.get("per_class", {}).get(category, {}).get("f1")
            if _is_number(f1_value):
                class_candidates.append((key, method, float(f1_value)))
        if class_candidates:
            key, method, score = max(class_candidates, key=lambda item: item[2])
            findings.append(
                f"For {category}, {method.get('approach') or METHOD_NAMES[key]} produced the strongest class F1 score at {_percent(score)}."
            )

    not_evaluated = [
        method.get("approach") or METHOD_NAMES[key]
        for key, method in report.get("methods", {}).items()
        if key in METHOD_NAMES and not _is_number(method.get("overall_accuracy"))
    ]
    if not_evaluated:
        findings.append(
            "Not evaluated in this mode: " + ", ".join(map(str, not_evaluated)) + "."
        )
    return findings


def _confusion_matrix_table(method: dict[str, Any]) -> Table | None:
    matrix = method.get("confusion_matrix")
    if not isinstance(matrix, dict):
        return None

    actual_labels = list(matrix.get("actual_labels") or [])
    predicted_labels = list(matrix.get("predicted_labels") or [])
    values = list(matrix.get("values") or [])
    if not actual_labels or not predicted_labels or len(values) != len(actual_labels):
        return None

    data: list[list[Any]] = [["Actual / Predicted", *predicted_labels]]
    for actual, row in zip(actual_labels, values):
        data.append([actual, *row])

    remaining_width = 208 * mm
    predicted_width = remaining_width / len(predicted_labels)
    table = _styled_table(
        data,
        [48 * mm, *([predicted_width] * len(predicted_labels))],
        header_color=BRAND_YELLOW,
        font_size=8,
    )
    table.setStyle(
        TableStyle(
            [
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ]
        )
    )
    return table


def build_evaluation_pdf_report(
    report: dict[str, Any],
    generated_at: datetime | None = None,
) -> bytes:
    """Build one PDF report for the complete dashboard evaluation."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")

    evaluated = _evaluated_methods(report)
    if not evaluated:
        raise ValueError("A completed dashboard evaluation is required.")

    mode = str(report.get("evaluation_mode") or "ripeness").lower()
    mode_label = MODE_LABELS.get(mode, mode.title())
    categories = [str(item) for item in report.get("categories", [])]
    if not categories:
        raise ValueError("The evaluation report does not define any categories.")

    export_time = generated_at or datetime.now().astimezone()
    methods = report.get("methods", {})
    image_count = int(report.get("image_count") or 0)
    styles = _report_styles()

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=17 * mm,
        title=REPORT_TITLE,
        author="Banana Ripeness Assessment System",
    )

    story: list[Any] = [
        Paragraph(REPORT_TITLE, styles["ReportTitle"]),
        Paragraph(
            f"{escape(mode_label)} evaluation across all evaluated image-processing approaches",
            styles["ReportSubtitle"],
        ),
        Paragraph(
            f"Evaluation completed: {escape(_timestamp(report.get('generated_at')))}<br/>"
            f"Report exported: {escape(_timestamp(export_time.isoformat()))}",
            styles["SmallBody"],
        ),
        Spacer(1, 2.5 * mm),
    ]

    status_text = str(report.get("status") or "Evaluation completed.")
    status_table = Table(
        [[Paragraph(f"<b>Status:</b> {escape(status_text)}", styles["SmallBody"])]],
        colWidths=[256 * mm],
    )
    status_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREEN),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7C9B8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([status_table, Paragraph("Evaluation summary", styles["SectionTitle"])])

    summary_data = [
        ["Mode", "Images", "Successful", "Failed", "Methods evaluated"],
        [
            mode_label,
            str(image_count),
            str(int(report.get("successful_images") or 0)),
            str(int(report.get("failed_images") or 0)),
            str(len(evaluated)),
        ],
    ]
    summary_table = _styled_table(
        summary_data,
        [48 * mm, 48 * mm, 52 * mm, 48 * mm, 60 * mm],
        font_size=9,
    )
    summary_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(summary_table)

    dataset_data = [
        ["Dataset information", "Value"],
        ["Directory", str(report.get("dataset_directory") or "-")],
        ["Split", str(report.get("dataset_split") or "-")],
    ]
    dataset_table = _styled_table(dataset_data, [52 * mm, 204 * mm])
    story.extend([Paragraph("Dataset and class distribution", styles["SectionTitle"]), dataset_table])

    class_counts = report.get("class_counts", {})
    class_data = [["Category", "Images", "Share of dataset"]]
    for category in categories:
        count = int(class_counts.get(category) or 0)
        share = count / image_count if image_count else None
        class_data.append([category, str(count), _percent(share)])
    class_table = _styled_table(class_data, [86 * mm, 70 * mm, 100 * mm])
    class_table.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "CENTER")]))
    story.append(class_table)

    comparison_data = [[
        "Approach",
        "Status",
        "Accuracy",
        "Macro F1",
        "Avg. time",
        "Successful",
        "Failed",
    ]]
    for key in METHOD_NAMES:
        method = methods.get(key, {})
        comparison_data.append(
            [
                str(method.get("approach") or METHOD_NAMES[key]),
                str(method.get("status") or "Not evaluated"),
                _percent(method.get("overall_accuracy")),
                _percent(method.get("macro_f1")),
                _number(method.get("average_processing_time_ms"), " ms"),
                str(method.get("successful_images") if method.get("successful_images") is not None else "-"),
                str(method.get("failed_images") if method.get("failed_images") is not None else "-"),
            ]
        )
    comparison_table = _styled_table(
        comparison_data,
        [43 * mm, 42 * mm, 31 * mm, 31 * mm, 38 * mm, 38 * mm, 33 * mm],
        font_size=8,
    )
    comparison_table.setStyle(TableStyle([("ALIGN", (2, 1), (-1, -1), "CENTER")]))
    story.append(
        KeepTogether(
            [
                Paragraph("Approach comparison", styles["SectionTitle"]),
                comparison_table,
                Spacer(1, 1.5 * mm),
                _performance_chart(evaluated),
            ]
        )
    )

    story.append(Paragraph("Automated findings", styles["SectionTitle"]))
    for finding in _automated_findings(report, evaluated, categories):
        story.append(
            Paragraph(f"&#8226;&nbsp; {escape(finding)}", styles["Finding"])
        )

    story.extend([PageBreak(), Paragraph("Per-class performance", styles["SectionTitle"])])
    story.append(
        Paragraph(
            "Precision, recall and F1 are reported for each class using the same fixed test split.",
            styles["SmallBody"],
        )
    )
    per_class_data = [["Approach", "Category", "Precision", "Recall", "F1", "Support"]]
    for key, method in evaluated:
        approach_name = str(method.get("approach") or METHOD_NAMES[key])
        for category in categories:
            metrics = method.get("per_class", {}).get(category, {})
            per_class_data.append(
                [
                    approach_name,
                    category,
                    _percent(metrics.get("precision")),
                    _percent(metrics.get("recall")),
                    _percent(metrics.get("f1")),
                    str(metrics.get("support") if metrics.get("support") is not None else "-"),
                ]
            )
    per_class_table = _styled_table(
        per_class_data,
        [52 * mm, 48 * mm, 39 * mm, 39 * mm, 39 * mm, 39 * mm],
        font_size=8,
    )
    per_class_table.setStyle(TableStyle([("ALIGN", (2, 1), (-1, -1), "CENTER")]))
    story.append(per_class_table)

    story.append(Paragraph("Confusion matrices", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Rows are actual classes and columns are predicted classes. A Failed column records images for which an approach did not return a prediction.",
            styles["SmallBody"],
        )
    )
    for key, method in evaluated:
        matrix_table = _confusion_matrix_table(method)
        if matrix_table is None:
            continue
        story.append(
            KeepTogether(
                [
                    Paragraph(
                        escape(str(method.get("approach") or METHOD_NAMES[key])),
                        styles["MethodTitle"],
                    ),
                    matrix_table,
                    Spacer(1, 2 * mm),
                ]
            )
        )

    story.append(Paragraph("Interpretation notes", styles["SectionTitle"]))
    notes = [
        "Overall accuracy is the proportion of correctly classified images.",
        "Macro F1 gives equal importance to every class, including minority classes.",
        "Performance values should be interpreted together with the class-level metrics and confusion matrices.",
        "This report is generated from the saved Dashboard evaluation and does not change the evaluation results.",
    ]
    for note in notes:
        story.append(Paragraph(f"&#8226;&nbsp; {escape(note)}", styles["Finding"]))

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output.getvalue()