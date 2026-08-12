from ui.single_method_page import render_single_method_page


render_single_method_page(
    method_key="kmeans",
    title="K-means Colour Clustering",
    description=(
        "Upload one banana image for ripeness assessment using colour "
        "clusters and cluster proportions."
    ),
    scope_items=[
        "Banana-region pixel clustering",
        "Cluster centroid colours",
        "Cluster-to-peel-colour mapping",
        "Cluster pixel percentages",
        "Four-class ripeness prediction",
        "Rule-based confidence level",
    ],
)