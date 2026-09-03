"""
GLCM test evaluation script.

Evaluates the locked ripeness rules on the test dataset.
No test data is used to tune the rules.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_FILE = PROJECT_ROOT / "glcm_test_features.csv"

# Locked rules from validation calibration
RULES = {
    "rotten": {"contrast_min": 0.164410, "homogeneity_max": 0.932841, "energy_max": 0.327074, "correlation_max": 0.958117},
    "unripe": {"contrast_max": 0.100742, "homogeneity_min": 0.953566, "energy_min": 0.118257, "correlation_min": 0.938309},
    "overripe": {"contrast_min": 0.074217, "homogeneity_max": 0.953566, "energy_max": 0.381516, "correlation_max": 0.915206},
}

CLASSES = ["unripe", "ripe", "overripe", "rotten"]


def load_data():
    df = pd.read_csv(TEST_FILE)
    return df


def classify(row):
    c, h, e, corr = row["contrast"], row["homogeneity"], row["energy"], row["correlation"]

    # Priority: rotten -> unripe -> overripe -> ripe fallback
    r = RULES["rotten"]
    if c >= r["contrast_min"] and h <= r["homogeneity_max"] and e <= r["energy_max"] and corr <= r["correlation_max"]:
        return "rotten"

    u = RULES["unripe"]
    if c <= u["contrast_max"] and h >= u["homogeneity_min"] and e >= u["energy_min"] and corr >= u["correlation_min"]:
        return "unripe"

    o = RULES["overripe"]
    if c >= o["contrast_min"] and h <= o["homogeneity_max"] and e <= o["energy_max"] and corr <= o["correlation_max"]:
        return "overripe"

    return "ripe"


def main():
    df = load_data()
    actual = df["category"].values
    predicted = np.array([classify(row) for _, row in df.iterrows()])

    accuracy = float(np.mean(actual == predicted))
    print(f"Test accuracy: {accuracy:.4f}")

    cm = np.zeros((4, 4), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    for a, p in zip(actual, predicted):
        if a in class_to_idx and p in class_to_idx:
            cm[class_to_idx[a], class_to_idx[p]] += 1

    print("Confusion matrix:")
    print("          " + "  ".join(f"{c:>8}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>8}  " + "  ".join(f"{cm[i][j]:>8}" for j in range(4)))

    print("\nPer-class metrics:")
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        print(f"  {cls}: P={prec:.4f} R={rec:.4f} F1={f1:.4f} support={cm[i,:].sum()}")


if __name__ == "__main__":
    main()
