from pathlib import Path

import streamlit as st


ROOT = Path(__file__).parent

st.set_page_config(
    page_title="Banana Ripeness Assessment",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(
        ROOT / "pages" / "1_Dashboard.py",
        title="Dashboard",
        icon=":material/dashboard:",
        default=True,
    ),
    st.Page(
        ROOT / "pages" / "2_Morphology.py",
        title="Morphology",
        icon=":material/grain:",
    ),
    st.Page(
        ROOT / "pages" / "3_HSV.py",
        title="HSV",
        icon=":material/palette:",
    ),
    st.Page(
        ROOT / "pages" / "4_KMeans.py",
        title="K-means",
        icon=":material/bubble_chart:",
    ),
    st.Page(
        ROOT / "pages" / "5_GLCM.py",
        title="GLCM Texture",
        icon=":material/texture:",
    ),
    st.Page(
        ROOT / "pages" / "6_Hybrid.py",
        title="Hybrid",
        icon=":material/hub:",
    ),
    st.Page(
    ROOT / "pages" / "7_Segmentation_Test.py",
    title="Segmentation Test",
    icon=":material/filter_center_focus:",
),
]

with st.sidebar:
    st.markdown("## 🍌 Banana Assessment")
    st.caption("Classical image-processing prototype")
    st.divider()
    st.caption(
        "UI prototype mode: image-processing and evaluation logic "
        "have not been connected yet."
    )

navigation = st.navigation(pages, position="sidebar", expanded=True)
navigation.run()