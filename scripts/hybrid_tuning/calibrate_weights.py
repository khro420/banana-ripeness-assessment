"""Tune compact Hybrid weights on a calibration partition, then verify holdout."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.hybrid import CATEGORIES, METHOD_ORDER, CLASS_RELIABILITY_WEIGHTS


ARTIFACT_DIRECTORY = ROOT / "outputs" / "development" / "hybrid_tuning"
REPORT = ROOT / "outputs" / "hybrid_tuning" / "weight_calibration_report.json"
INDEX = {name: number for number, name in enumerate(CATEGORIES)}


def load() -> tuple[np.ndarray, np.ndarray]:
    rows_by_path = {}
    for file in ARTIFACT_DIRECTORY.glob("*_predictions.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "results" in row and all(method in row["results"] for method in METHOD_ORDER):
                rows_by_path[row["path"]] = row
    rows = list(rows_by_path.values())
    supports = np.zeros((len(rows), len(CATEGORIES), len(METHOD_ORDER)), dtype=float)
    labels = np.zeros(len(rows), dtype=int)
    for row_number, row in enumerate(rows):
        labels[row_number] = INDEX[row["actual"]]
        for method_number, method in enumerate(METHOD_ORDER):
            result = row["results"][method]
            category = result["predicted"]
            score = result.get("scores", {}).get(category, 0.0)
            supports[row_number, INDEX[category], method_number] = max(float(score or 0.0), 0.0)
    return supports, labels


def as_array(weights: dict) -> np.ndarray:
    return np.array([[weights[category][method] for method in METHOD_ORDER] for category in CATEGORIES], dtype=float)


def as_dict(weights: np.ndarray) -> dict:
    return {
        category: {method: round(float(weights[i, j]), 6) for j, method in enumerate(METHOD_ORDER)}
        for i, category in enumerate(CATEGORIES)
    }


def predict(supports: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.argmax(np.einsum("ncm,cm->nc", supports, weights), axis=1)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 6),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 6),
    }


def main() -> None:
    supports, labels = load()
    indices = np.arange(len(labels))
    calibration, holdout = train_test_split(indices, test_size=0.25, random_state=2026, stratify=labels)
    baseline = as_array(CLASS_RELIABILITY_WEIGHTS)
    rng = np.random.default_rng(2026)
    candidates = [baseline]
    for _ in range(20000):
        candidates.append(np.array([rng.dirichlet(np.maximum(row * 70.0, 0.8)) for row in baseline]))

    best = max(
        candidates,
        key=lambda weights: f1_score(labels[calibration], predict(supports[calibration], weights), average="macro", zero_division=0),
    )
    baseline_holdout = metrics(labels[holdout], predict(supports[holdout], baseline))
    candidate_holdout = metrics(labels[holdout], predict(supports[holdout], best))
    accepted = (
        candidate_holdout["macro_f1"] >= baseline_holdout["macro_f1"] + 0.005
        and candidate_holdout["accuracy"] >= baseline_holdout["accuracy"]
    )
    report = {
        "records": int(len(labels)),
        "split": {"calibration": int(len(calibration)), "holdout": int(len(holdout)), "random_state": 2026},
        "baseline": {"weights": as_dict(baseline), "calibration": metrics(labels[calibration], predict(supports[calibration], baseline)), "holdout": baseline_holdout},
        "candidate": {"weights": as_dict(best), "calibration": metrics(labels[calibration], predict(supports[calibration], best)), "holdout": candidate_holdout},
        "accepted": accepted,
        "acceptance_rule": "Holdout macro F1 improves by at least 0.5 percentage points with no holdout accuracy decrease.",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
