"""Calibrate known-ripe quality Hybrid weights using validation data only.

The script is restartable: completed branch predictions are retained in a
JSONL file and only missing validation images are processed on the next run.
It never reads the held-out ``dataset/quality/test`` partition.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branches.glcm.glcm_analysis import (
    GLCMParameters,
    GLCMQualityBands,
    analyse_glcm_quality,
)
from branches.hsv.hsv_analysis import (
    HSVQualityBands,
    analyse_hsv_quality,
)
from branches.hsv.hsv_segmentation import HSVParameters
from branches.kmeans.kmeans_analysis import (
    KMeansQualityBands,
    analyse_kmeans_quality,
)
from branches.kmeans.kmeans_segmentation import KMeansParameters
from branches.morphology.morphology_analysis import (
    QualityBands,
    analyse_morphology_quality,
)
from branches.morphology.morphology_segmentation import MorphologyParameters
from core.banana_segmentation import segment_banana
from core.evaluation import _open_rgb_image
from core.hybrid import (
    METHOD_ORDER,
    QUALITY_CLASS_RELIABILITY_WEIGHTS,
    QUALITY_HYBRID_CATEGORIES,
)
from core.image_handling import standardise_image


VALIDATION_DIRECTORY = ROOT / "dataset" / "quality" / "valid"
ARTIFACT_DIRECTORY = ROOT / "outputs" / "development" / "hybrid_tuning"
REPORT_DIRECTORY = ROOT / "outputs" / "hybrid_tuning"
PREDICTIONS_FILE = ARTIFACT_DIRECTORY / "quality_validation_predictions.jsonl"
REPORT_FILE = REPORT_DIRECTORY / "quality_weight_calibration_report.json"
FOLDER_TO_CATEGORY = {
    "class_a": "Class_A",
    "class_b": "Class_B",
    "defect": "Defect",
}
CATEGORY_INDEX = {
    category: index for index, category in enumerate(QUALITY_HYBRID_CATEGORIES)
}
RANDOM_STATE = 2026


def _images() -> list[tuple[Path, str]]:
    return [
        (path, category)
        for folder_name, category in FOLDER_TO_CATEGORY.items()
        for path in sorted((VALIDATION_DIRECTORY / folder_name).glob("*"))
        if path.is_file()
    ]


def _existing_paths() -> set[str]:
    if not PREDICTIONS_FILE.exists():
        return set()
    return {
        row["path"]
        for line in PREDICTIONS_FILE.read_text(encoding="utf-8").splitlines()
        if line and (row := json.loads(line)).get("results")
    }


def _analyse(path: Path, actual: str) -> dict:
    image = _open_rgb_image(path)
    prepared = standardise_image(image, {"filename": path.name}, target_size=(416, 416))
    segmentation = segment_banana(prepared.working_rgb, prepared.content_mask)
    if not segmentation.success:
        return {"path": str(path.relative_to(ROOT)), "actual": actual, "error": segmentation.message}

    analyses = (
        analyse_morphology_quality(
            prepared.working_rgb, segmentation.final_mask,
            MorphologyParameters(), QualityBands(),
        ),
        analyse_hsv_quality(
            prepared.working_rgb, segmentation.final_mask,
            HSVParameters(), HSVQualityBands(),
        ),
        analyse_kmeans_quality(
            prepared.working_rgb, segmentation.final_mask,
            KMeansParameters(), KMeansQualityBands(),
        ),
        analyse_glcm_quality(
            prepared.working_rgb, segmentation.final_mask,
            GLCMParameters(), GLCMQualityBands(),
        ),
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "actual": actual,
        "results": {
            analysis.method_result.method_key: {
                "predicted": analysis.method_result.predicted_category,
                "confidence": analysis.method_result.confidence_percent,
                "scores": analysis.method_result.class_scores,
            }
            for analysis in analyses
        },
    }


def _load_records() -> list[dict]:
    if not PREDICTIONS_FILE.exists():
        return []
    rows_by_path = {
        row["path"]: row
        for line in PREDICTIONS_FILE.read_text(encoding="utf-8").splitlines()
        if line and (row := json.loads(line)).get("results")
        and all(method in row["results"] for method in METHOD_ORDER)
    }
    return list(rows_by_path.values())


def _as_array(weights: dict) -> np.ndarray:
    return np.array(
        [
            [weights[category][method] for method in METHOD_ORDER]
            for category in QUALITY_HYBRID_CATEGORIES
        ],
        dtype=float,
    )


def _as_dict(weights: np.ndarray) -> dict:
    return {
        category: {
            method: round(float(weights[row, column]), 6)
            for column, method in enumerate(METHOD_ORDER)
        }
        for row, category in enumerate(QUALITY_HYBRID_CATEGORIES)
    }


def _supports_and_labels(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    supports = np.zeros(
        (len(records), len(QUALITY_HYBRID_CATEGORIES), len(METHOD_ORDER)),
        dtype=float,
    )
    labels = np.zeros(len(records), dtype=int)
    for row_index, record in enumerate(records):
        labels[row_index] = CATEGORY_INDEX[record["actual"]]
        for method_index, method in enumerate(METHOD_ORDER):
            result = record["results"][method]
            prediction = result["predicted"]
            score = result.get("scores", {}).get(prediction, 0.0)
            if score is None or score <= 0:
                score = float(result.get("confidence") or 0.0) / 100.0
            supports[row_index, CATEGORY_INDEX[prediction], method_index] = score
    return supports, labels


def _predict(supports: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.argmax(np.einsum("ncm,cm->nc", supports, weights), axis=1)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 6),
        "macro_f1": round(
            float(f1_score(actual, predicted, average="macro", zero_division=0)),
            6,
        ),
    }


def _candidate_weights(baseline: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    candidates = [baseline, np.full_like(baseline, 1.0 / len(METHOD_ORDER))]
    for concentration in (8.0, 20.0, 50.0, 120.0):
        candidates.extend(
            np.array(
                [rng.dirichlet(np.maximum(row * concentration, 0.35)) for row in baseline]
            )
            for _ in range(12_500)
        )
    return candidates


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    images = _images()
    completed_paths = _existing_paths()
    missing = [
        (path, actual)
        for path, actual in images
        if str(path.relative_to(ROOT)) not in completed_paths
    ]
    print(f"Quality validation images: {len(images)}; remaining: {len(missing)}", flush=True)
    with PREDICTIONS_FILE.open("a", encoding="utf-8") as output:
        for index, (path, actual) in enumerate(missing, start=1):
            try:
                record = _analyse(path, actual)
            except Exception as error:
                record = {
                    "path": str(path.relative_to(ROOT)),
                    "actual": actual,
                    "error": f"{type(error).__name__}: {error}",
                }
            output.write(json.dumps(record) + "\n")
            output.flush()
            if index == 1 or index % 25 == 0 or index == len(missing):
                print(f"Processed {index}/{len(missing)}", flush=True)

    records = _load_records()
    if len(records) != len(images):
        raise RuntimeError(
            f"Only {len(records)} of {len(images)} validation images completed. "
            "Re-run to resume failed rows after resolving their errors."
        )

    supports, labels = _supports_and_labels(records)
    indices = np.arange(len(labels))
    calibration, holdout = train_test_split(
        indices,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    baseline = _as_array(QUALITY_CLASS_RELIABILITY_WEIGHTS)
    candidates = _candidate_weights(baseline)
    baseline_calibration = _metrics(
        labels[calibration],
        _predict(supports[calibration], baseline),
    )
    scored_candidates = [
        (
            _metrics(labels[calibration], _predict(supports[calibration], weights)),
            weights,
        )
        for weights in candidates
    ]
    best_metrics, best = max(
        scored_candidates,
        key=lambda item: (item[0]["macro_f1"], item[0]["accuracy"]),
    )
    balanced_metrics, balanced = max(
        (
            item
            for item in scored_candidates
            if item[0]["macro_f1"] >= baseline_calibration["macro_f1"] + 0.005
        ),
        key=lambda item: (item[0]["accuracy"], item[0]["macro_f1"]),
    )
    baseline_holdout = _metrics(labels[holdout], _predict(supports[holdout], baseline))
    candidate_holdout = _metrics(labels[holdout], _predict(supports[holdout], best))
    balanced_holdout = _metrics(labels[holdout], _predict(supports[holdout], balanced))
    accepted = (
        candidate_holdout["macro_f1"] >= baseline_holdout["macro_f1"] + 0.005
        and candidate_holdout["accuracy"] >= baseline_holdout["accuracy"]
    )
    balanced_accepted = (
        balanced_holdout["macro_f1"] >= baseline_holdout["macro_f1"] + 0.005
        and balanced_holdout["accuracy"] >= baseline_holdout["accuracy"]
    )
    report = {
        "records": len(records),
        "split": {
            "calibration": len(calibration),
            "holdout": len(holdout),
            "random_state": RANDOM_STATE,
        },
        "baseline": {
            "weights": _as_dict(baseline),
            "calibration": baseline_calibration,
            "holdout": baseline_holdout,
        },
        "candidate": {
            "weights": _as_dict(best),
            "calibration": best_metrics,
            "holdout": candidate_holdout,
        },
        "balanced_candidate": {
            "weights": _as_dict(balanced),
            "calibration": balanced_metrics,
            "holdout": balanced_holdout,
        },
        "accepted": accepted,
        "balanced_candidate_accepted": balanced_accepted,
        "acceptance_rule": (
            "Holdout macro F1 improves by at least 0.5 percentage points "
            "with no holdout accuracy decrease."
        ),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
