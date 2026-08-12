from ui.single_method_page import render_single_method_page


render_single_method_page(
    method_key="hsv",
    title="HSV Colour Analysis",
    description=(
        "Upload one banana image for ripeness assessment using HSV colour "
        "segmentation and colour-area ratios."
    ),
    scope_items=[
        "HSV colour-space conversion",
        "Green, yellow, brown and dark masks",
        "Colour-region area percentages",
        "Four-class ripeness prediction",
        "Rule-based confidence level",
    ],
)