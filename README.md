# Banana Ripeness and Surface Quality Assessment System

This Streamlit application compares four classical image-processing approaches - Morphology, HSV, K-means and GLCM Texture - and combines their outputs through Hybrid weighted fusion.

## Run the application

1. Use Python 3.11 or a compatible version.
2. Install dependencies with `pip install -r requirements.txt`.
3. Place the cited datasets under `dataset/ripeness/` and `dataset/quality/` using the `valid` and `test` folder structures described in the documentation.
4. Start the application with `streamlit run app.py`.

The datasets are intentionally excluded from the source-code submission, as required by the assignment specification. The application can still be reviewed without them, but fixed-dataset evaluation requires the corresponding local dataset folders.

## Repository layout

- `app.py`, `pages/`, `core/`, `branches/`, `ui/`: application runtime source.
- `outputs/evaluation_results/`: latest final ripeness and quality evaluation evidence.
- `outputs/hybrid_tuning/`: retained Hybrid calibration reports.
- `outputs/quality_split_manifest_70_30.json`: reproducible quality validation/test split record.
- `scripts/`: optional validation and calibration utilities.
- `outputs/development/`: generated development artefacts; excluded from submission.
