from ui.single_method_page import render_single_method_page


render_single_method_page(
    method_key="glcm",
    title="GLCM Texture Analysis",
    description=(
        "Upload one banana image for ripeness assessment using greyscale "
        "texture characteristics."
    ),
    scope_items=[
        "Masked greyscale banana region",
        "Grey-level quantisation",
        "GLCMs at selected distances and angles",
        "Contrast, homogeneity, energy and correlation",
        "Four-class ripeness prediction",
        "Rule-based confidence level",
    ],
)