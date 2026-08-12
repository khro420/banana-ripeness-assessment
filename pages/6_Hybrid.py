from ui.single_method_page import render_single_method_page


render_single_method_page(
    method_key="hybrid",
    title="Hybrid Banana Assessment",
    description=(
        "Upload one banana image to combine Morphology, HSV, K-means and "
        "GLCM Texture results."
    ),
    scope_items=[
        "Four independent approach predictions",
        "Validation-F1 weighted class voting",
        "Final four-class ripeness prediction",
        "Method agreement level",
        "Hybrid rule-based confidence",
        "Blemish percentage and surface-quality grade",
    ],
)