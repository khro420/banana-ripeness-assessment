"""Export current ripeness branch predictions for Hybrid weight calibration."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branches.glcm.glcm_analysis import GLCMParameters, GLCMRipenessBands, analyse_glcm
from branches.hsv.hsv_analysis import HSVRipenessBands, analyse_hsv
from branches.hsv.hsv_segmentation import HSVParameters
from branches.kmeans.kmeans_analysis import KMeansRipenessBands, analyse_kmeans
from branches.kmeans.kmeans_segmentation import KMeansParameters
from branches.morphology.morphology_analysis import RipenessBands, analyse_morphology
from branches.morphology.morphology_segmentation import MorphologyParameters
from core.banana_segmentation import segment_banana
from core.evaluation import RIPENESS_FOLDER_TO_CATEGORY, _open_rgb_image
from core.image_handling import standardise_image


DATASET = ROOT / "dataset" / "ripeness" / "valid"
OUTPUT = ROOT / "outputs" / "development" / "hybrid_tuning" / "validation_branch_predictions.jsonl"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images(dataset: Path) -> list[tuple[Path, str]]:
    return [
        (path, RIPENESS_FOLDER_TO_CATEGORY[folder.name])
        for folder in sorted(dataset.iterdir())
        if folder.is_dir() and folder.name in RIPENESS_FOLDER_TO_CATEGORY
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]


def run(path: Path, actual: str) -> dict:
    image = _open_rgb_image(path)
    prepared = standardise_image(image, {"filename": path.name}, target_size=(416, 416))
    segmented = segment_banana(prepared.working_rgb, prepared.content_mask)
    if not segmented.success:
        return {"path": str(path.relative_to(ROOT)), "actual": actual, "error": segmented.message}

    analyses = (
        analyse_morphology(prepared.working_rgb, segmented.final_mask, MorphologyParameters(), RipenessBands()),
        analyse_hsv(prepared.working_rgb, segmented.final_mask, HSVParameters(), HSVRipenessBands()),
        analyse_kmeans(prepared.working_rgb, segmented.final_mask, KMeansParameters(), KMeansRipenessBands()),
        analyse_glcm(prepared.working_rgb, segmented.final_mask, GLCMParameters(), GLCMRipenessBands()),
    )
    results = {}
    for analysis in analyses:
        result = analysis.method_result
        results[result.method_key] = {
            "predicted": result.predicted_category,
            "confidence": result.confidence_percent,
            "scores": result.class_scores,
        }
    return {"path": str(path.relative_to(ROOT)), "actual": actual, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    selected = images(args.dataset.resolve())[args.start:args.start + args.count]
    with args.output.open("a", encoding="utf-8") as stream:
        for number, (path, actual) in enumerate(selected, args.start + 1):
            try:
                record = run(path, actual)
            except Exception as error:
                record = {"path": str(path.relative_to(ROOT)), "actual": actual, "error": str(error)}
            stream.write(json.dumps(record) + "\n")
            print(f"{number}: {path.name}")


if __name__ == "__main__":
    main()
