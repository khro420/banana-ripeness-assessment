"""
Simple GLCM ripeness calibration.

No machine learning. Searches for threshold rules on the
VALIDATION dataset only. Rules are saved to a report file.

Steps:
1. Search best rule for each non-ripe class
2. Use priority order to get 4-class accuracy
3. Save report
"""
from pathlib import Path
import sys
from itertools import combinations

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_FILE = PROJECT_ROOT / "glcm_validation_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "glcm_ripeness_calibration_report.txt"

FEATURES = ["contrast", "homogeneity", "energy", "correlation"]
CLASSES = ["unripe", "ripe", "overripe", "rotten"]
PERCENTILES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def load_data():
    """Load validation CSV."""
    df = pd.read_csv(VALIDATION_FILE)
    required = ["category"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def accuracy_score(actual, predicted):
    """Simple accuracy."""
    return float(np.mean(np.asarray(actual) == np.asarray(predicted)))


def search_best_rule_for_class(df, positive_class):
    """
    Find best simple rule for one class vs all others.

    Tries single, 2-feature, 3-feature, and 4-feature AND rules.
    """
    target = (df["category"].values == positive_class).astype(bool)
    best = None

    # Precompute values and thresholds
    feature_values = {f: df[f].values for f in FEATURES}
    feature_thresholds = {f: np.unique(np.percentile(feature_values[f], PERCENTILES)) for f in FEATURES}

    def evaluate(preds):
        acc = float(np.mean(preds == target))
        tp = int(np.sum(preds & target))
        fp = int(np.sum(preds & ~target))
        fn = int(np.sum(~preds & target))
        return acc, tp, fp, fn

    # Single-feature rules
    for feat in FEATURES:
        vals = feature_values[feat]
        ths = feature_thresholds[feat]
        for d in [">=", "<="]:
            preds = vals[:, None] >= ths[None, :] if d == ">=" else vals[:, None] <= ths[None, :]
            for j in range(len(ths)):
                acc, tp, fp, fn = evaluate(preds[:, j])
                if best is None or acc > best["accuracy"]:
                    best = {"type": "single", "feature": feat, "direction": d,
                            "threshold": float(ths[j]), "accuracy": acc}

    # 2-feature AND rules
    for f1, f2 in combinations(FEATURES, 2):
        t1 = feature_thresholds[f1]
        t2 = feature_thresholds[f2]
        v1 = feature_values[f1]
        v2 = feature_values[f2]
        for d1 in [">=", "<="]:
            r1 = v1[:, None, None] >= t1[None, :, None] if d1 == ">=" else v1[:, None, None] <= t1[None, :, None]
            for d2 in [">=", "<="]:
                r2 = v2[:, None, None] >= t2[None, None, :] if d2 == ">=" else v2[:, None, None] <= t2[None, None, :]
                preds = r1 & r2
                for ii in range(len(t1)):
                    for jj in range(len(t2)):
                        acc, tp, fp, fn = evaluate(preds[:, ii, jj])
                        if acc > best["accuracy"]:
                            best = {"type": "two", "feature1": f1, "direction1": d1,
                                    "threshold1": float(t1[ii]), "feature2": f2, "direction2": d2,
                                    "threshold2": float(t2[jj]), "accuracy": acc}

    # 3-feature AND rules
    for f1, f2, f3 in combinations(FEATURES, 3):
        t1 = feature_thresholds[f1]
        t2 = feature_thresholds[f2]
        t3 = feature_thresholds[f3]
        v1 = feature_values[f1]
        v2 = feature_values[f2]
        v3 = feature_values[f3]
        for d1 in [">=", "<="]:
            r1 = v1[:, None, None, None] >= t1[None, :, None, None] if d1 == ">=" else v1[:, None, None, None] <= t1[None, :, None, None]
            for d2 in [">=", "<="]:
                r2 = v2[:, None, None, None] >= t2[None, None, :, None] if d2 == ">=" else v2[:, None, None, None] <= t2[None, None, :, None]
                for d3 in [">=", "<="]:
                    r3 = v3[:, None, None, None] >= t3[None, None, None, :] if d3 == ">=" else v3[:, None, None, None] <= t3[None, None, None, :]
                    preds = r1 & r2 & r3
                    for ii in range(len(t1)):
                        for jj in range(len(t2)):
                            for kk in range(len(t3)):
                                acc, tp, fp, fn = evaluate(preds[:, ii, jj, kk])
                                if acc > best["accuracy"]:
                                    best = {"type": "three", "feature1": f1, "direction1": d1,
                                            "threshold1": float(t1[ii]), "feature2": f2, "direction2": d2,
                                            "threshold2": float(t2[jj]), "feature3": f3, "direction3": d3,
                                            "threshold3": float(t3[kk]), "accuracy": acc}

    # 4-feature AND rules
    t1 = feature_thresholds[FEATURES[0]]
    t2 = feature_thresholds[FEATURES[1]]
    t3 = feature_thresholds[FEATURES[2]]
    t4 = feature_thresholds[FEATURES[3]]
    v1 = feature_values[FEATURES[0]]
    v2 = feature_values[FEATURES[1]]
    v3 = feature_values[FEATURES[2]]
    v4 = feature_values[FEATURES[3]]
    for d1 in [">=", "<="]:
        r1 = v1[:, None, None, None, None] >= t1[None, :, None, None, None] if d1 == ">=" else v1[:, None, None, None, None] <= t1[None, :, None, None, None]
        for d2 in [">=", "<="]:
            r2 = v2[:, None, None, None, None] >= t2[None, None, :, None, None] if d2 == ">=" else v2[:, None, None, None, None] <= t2[None, None, :, None, None]
            for d3 in [">=", "<="]:
                r3 = v3[:, None, None, None, None] >= t3[None, None, None, :, None] if d3 == ">=" else v3[:, None, None, None, None] <= t3[None, None, None, :, None]
                for d4 in [">=", "<="]:
                    r4 = v4[:, None, None, None, None] >= t4[None, None, None, None, :] if d4 == ">=" else v4[:, None, None, None, None] <= t4[None, None, None, None, :]
                    preds = r1 & r2 & r3 & r4
                    for ii in range(len(t1)):
                        for jj in range(len(t2)):
                            for kk in range(len(t3)):
                                for ll in range(len(t4)):
                                    acc, tp, fp, fn = evaluate(preds[:, ii, jj, kk, ll])
                                    if acc > best["accuracy"]:
                                        best = {"type": "four", "feature1": FEATURES[0], "direction1": d1,
                                                "threshold1": float(t1[ii]), "feature2": FEATURES[1], "direction2": d2,
                                                "threshold2": float(t2[jj]), "feature3": FEATURES[2], "direction3": d3,
                                                "threshold3": float(t3[kk]), "feature4": FEATURES[3], "direction4": d4,
                                                "threshold4": float(t4[ll]), "accuracy": acc}
    return best


def format_rule(rule):
    """Format rule dict as readable string."""
    if rule["type"] == "single":
        return f"  {rule['feature']} {rule['direction']} {rule['threshold']:.6f}"
    elif rule["type"] == "two":
        return (f"  {rule['feature1']} {rule['direction1']} {rule['threshold1']:.6f}\n"
                f"  AND {rule['feature2']} {rule['direction2']} {rule['threshold2']:.6f}")
    elif rule["type"] == "three":
        return (f"  {rule['feature1']} {rule['direction1']} {rule['threshold1']:.6f}\n"
                f"  AND {rule['feature2']} {rule['direction2']} {rule['threshold2']:.6f}\n"
                f"  AND {rule['feature3']} {rule['direction3']} {rule['threshold3']:.6f}")
    elif rule["type"] == "four":
        return (f"  {rule['feature1']} {rule['direction1']} {rule['threshold1']:.6f}\n"
                f"  AND {rule['feature2']} {rule['direction2']} {rule['threshold2']:.6f}\n"
                f"  AND {rule['feature3']} {rule['direction3']} {rule['threshold3']:.6f}\n"
                f"  AND {rule['feature4']} {rule['direction4']} {rule['threshold4']:.6f}")
    return "  (no rule)"


def apply_rule(row, rule):
    """Apply rule dict to one dataframe row."""
    if rule is None:
        return False
    if rule["type"] == "single":
        val = row[rule["feature"]]
        return val >= rule["threshold"] if rule["direction"] == ">=" else val <= rule["threshold"]
    elif rule["type"] == "two":
        r1 = row[rule["feature1"]] >= rule["threshold1"] if rule["direction1"] == ">=" else row[rule["feature1"]] <= rule["threshold1"]
        r2 = row[rule["feature2"]] >= rule["threshold2"] if rule["direction2"] == ">=" else row[rule["feature2"]] <= rule["threshold2"]
        return bool(r1 and r2)
    elif rule["type"] == "three":
        r1 = row[rule["feature1"]] >= rule["threshold1"] if rule["direction1"] == ">=" else row[rule["feature1"]] <= rule["threshold1"]
        r2 = row[rule["feature2"]] >= rule["threshold2"] if rule["direction2"] == ">=" else row[rule["feature2"]] <= rule["threshold2"]
        r3 = row[rule["feature3"]] >= rule["threshold3"] if rule["direction3"] == ">=" else row[rule["feature3"]] <= rule["threshold3"]
        return bool(r1 and r2 and r3)
    elif rule["type"] == "four":
        r1 = row[rule["feature1"]] >= rule["threshold1"] if rule["direction1"] == ">=" else row[rule["feature1"]] <= rule["threshold1"]
        r2 = row[rule["feature2"]] >= rule["threshold2"] if rule["direction2"] == ">=" else row[rule["feature2"]] <= rule["threshold2"]
        r3 = row[rule["feature3"]] >= rule["threshold3"] if rule["direction3"] == ">=" else row[rule["feature3"]] <= rule["threshold3"]
        r4 = row[rule["feature4"]] >= rule["threshold4"] if rule["direction4"] == ">=" else row[rule["feature4"]] <= rule["threshold4"]
        return bool(r1 and r2 and r3 and r4)
    return False


def main():
    df = load_data()
    print(f"Loaded {len(df)} validation images")
    print("\nClass counts:")
    print(df["category"].value_counts())

    # Search best rule for each non-ripe class
    priority_classes = ["unripe", "rotten", "overripe"]
    found_rules = {}

    for cls in priority_classes:
        print(f"\nSearching best rule for: {cls}")
        best = search_best_rule_for_class(df, cls)
        if best is None:
            print(f"  No rule found for {cls}")
            continue
        print(f"  Best rule type: {best['type']}-feature")
        print(f"  {format_rule(best)}")
        print(f"  Binary accuracy on validation: {best['accuracy']*100:.2f}%")
        found_rules[cls] = best

    # Evaluate full 4-class accuracy with priority order
    predictions = []
    for _, row in df.iterrows():
        pred = "ripe"
        for cls in priority_classes:
            if cls in found_rules and apply_rule(row, found_rules[cls]):
                pred = cls
                break
        predictions.append(pred)

    df["predicted"] = predictions
    overall = accuracy_score(df["category"].values, np.array(predictions))

    print("\n" + "=" * 70)
    print(f"Validation accuracy (4-class): {overall*100:.2f}%")
    print("=" * 70)

    for cls in CLASSES:
        mask = df["category"] == cls
        if mask.sum() == 0:
            continue
        correct = (df.loc[mask, "predicted"] == cls).sum()
        total = mask.sum()
        print(f"  {cls:10s}: {correct}/{total} = {correct/total*100:.1f}%")

    # Save report
    lines = [
        "GLCM RIPENESS CALIBRATION REPORT",
        "=" * 70, "",
        f"Validation accuracy: {overall*100:.2f}%", "",
        "RULES FOUND:", "-" * 70,
    ]

    for cls in priority_classes:
        if cls not in found_rules:
            continue
        rule = found_rules[cls]
        lines.append(f"\n{cls.upper()} RULE ({rule['type']}-feature):")
        lines.append(format_rule(rule))
        lines.append(f"  Binary accuracy: {rule['accuracy']*100:.2f}%")

    lines.extend([
        "\n" + "=" * 70,
        "NOTE: These rules are from the VALIDATION set only.",
        "Do NOT use the test set to choose thresholds.",
        "=" * 70,
    ])

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
