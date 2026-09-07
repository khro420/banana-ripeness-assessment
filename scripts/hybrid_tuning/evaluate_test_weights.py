"""Report the frozen Hybrid weights on the untouched ripeness test set."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.hybrid import CATEGORIES, METHOD_ORDER, CLASS_RELIABILITY_WEIGHTS


ARTIFACT_DIRECTORY = ROOT / "outputs" / "development" / "hybrid_tuning"
REPORT_DIRECTORY = ROOT / "outputs" / "hybrid_tuning"
REPORT = REPORT_DIRECTORY / "test_weight_evaluation.json"
INDEX = {name: number for number, name in enumerate(CATEGORIES)}


def rows() -> list[dict]:
    records = []
    for file in ARTIFACT_DIRECTORY.glob("test_*_predictions.jsonl"):
        records.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines())
    return [row for row in records if "results" in row and all(method in row["results"] for method in METHOD_ORDER)]


def predict(records: list[dict], weights: dict) -> np.ndarray:
    output = []
    for row in records:
        scores = dict.fromkeys(CATEGORIES, 0.0)
        for category in CATEGORIES:
            for method in METHOD_ORDER:
                result = row["results"][method]
                scores[category] += weights[category][method] * float(result["scores"].get(category, 0.0) or 0.0)
        output.append(max(CATEGORIES, key=scores.get))
    return np.array(output)


def measure(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 6),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 6),
        "per_class_f1": {
            category: round(float(f1_score(actual == category, predicted == category, zero_division=0)), 6)
            for category in CATEGORIES
        },
    }


def main() -> None:
    records = rows()
    actual = np.array([row["actual"] for row in records])
    validation_report = json.loads((REPORT_DIRECTORY / "weight_calibration_report.json").read_text(encoding="utf-8"))
    old_weights = validation_report["baseline"]["weights"]
    report = {
        "records": len(records),
        "old_weights": measure(actual, predict(records, old_weights)),
        "frozen_validation_selected_weights": measure(actual, predict(records, CLASS_RELIABILITY_WEIGHTS)),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
