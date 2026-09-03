"""
GLCM quality calibration.

No machine learning. Searches for simple threshold rules on the
quality validation dataset.

Steps:
1. Search best rule for Defect vs non-Defect
2. Search best rule for Class_A vs Class_B (among non-Defect)
3. Priority: Defect -> Class_A -> Class_B fallback
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_FILE = PROJECT_ROOT / "glcm_quality_validation_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "glcm_quality_calibration_report.txt"

FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
PERCENTILES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def load_data():
    """Load quality validation CSV."""
    df = pd.read_csv(VALIDATION_FILE)
    # Force feature columns to lowercase to match FEATURES list.
    df.columns = [c if c.lower() == "category" else c.lower() for c in df.columns]
    required = ["category"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def f1_score(tp, fp, fn):
    """F1 from counts."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def search_best_rule(target_mask, feature_matrix):
    """
    Find best simple rule for a target class.

    Tries single-feature and 2-feature AND rules.
    Optimizes for F1 to balance precision and recall.
    """
    target = target_mask.astype(bool)
    best = None

    def evaluate(preds):
        tp = int(np.sum(preds & target))
        fp = int(np.sum(preds & ~target))
        fn = int(np.sum(~preds & target))
        supp = int(np.sum(preds))
        acc = float(np.mean(preds == target))
        f1 = f1_score(tp, fp, fn)
        prec = tp / supp if supp > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return acc, prec, rec, f1, supp

    # Precompute percentiles for each feature
    thresholds = {}
    for idx, feat in enumerate(FEATURES):
        vals = feature_matrix[:, idx]
        thresholds[feat] = np.unique(np.percentile(vals, PERCENTILES))

    # Single-feature rules
    for idx, feat in enumerate(FEATURES):
        vals = feature_matrix[:, idx]
        ths = thresholds[feat]
        for d in [">=", "<="]:
            preds = vals[:, None] >= ths[None, :] if d == ">=" else vals[:, None] <= ths[None, :]
            for j in range(len(ths)):
                acc, prec, rec, f1, supp = evaluate(preds[:, j])
                if supp == 0:
                    continue
                if best is None or (f1, supp, acc) > (best["f1"], best["support"], best["accuracy"]):
                    best = {"type": "single", "feature": feat, "direction": d,
                            "threshold": float(ths[j]), "accuracy": acc,
                            "precision": prec, "recall": rec, "f1": f1, "support": supp}

    # 2-feature AND rules
    for i1, f1 in enumerate(FEATURES):
        for i2, f2 in enumerate(FEATURES):
            if i1 == i2:
                continue
            v1 = feature_matrix[:, i1]
            v2 = feature_matrix[:, i2]
            t1 = thresholds[f1]
            t2 = thresholds[f2]
            for d1 in [">=", "<="]:
                r1 = v1[:, None, None] >= t1[None, :, None] if d1 == ">=" else v1[:, None, None] <= t1[None, :, None]
                for d2 in [">=", "<="]:
                    r2 = v2[:, None, None] >= t2[None, None, :] if d2 == ">=" else v2[:, None, None] <= t2[None, None, :]
                    preds = r1 & r2
                    for ii in range(len(t1)):
                        for jj in range(len(t2)):
                            acc, prec, rec, rule_f1, supp = evaluate(preds[:, ii, jj])
                            if supp == 0:
                                continue
                            if best is None or (rule_f1, supp, acc) > (best["f1"], best["support"], best["accuracy"]):
                                best = {"type": "two", "feature1": f1, "direction1": d1,
                                        "threshold1": float(t1[ii]), "feature2": f2, "direction2": d2,
                                        "threshold2": float(t2[jj]), "accuracy": acc,
                                        "precision": prec, "recall": rec, "f1": rule_f1, "support": supp}
    return best


def format_rule(rule):
    """Format rule as readable string."""
    if rule["type"] == "single":
        return f"  {rule['feature']} {rule['direction']} {rule['threshold']:.6f}"
    elif rule["type"] == "two":
        return (f"  {rule['feature1']} {rule['direction1']} {rule['threshold1']:.6f}\n"
                f"  AND {rule['feature2']} {rule['direction2']} {rule['threshold2']:.6f}")
    return "  (no rule)"


def apply_rule(features, rule):
    """Apply a rule dict to a feature vector."""
    if rule is None:
        return False
    if rule["type"] == "single":
        idx = FEATURES.index(rule["feature"])
        val = features[idx]
        return val >= rule["threshold"] if rule["direction"] == ">=" else val <= rule["threshold"]
    elif rule["type"] == "two":
        i1 = FEATURES.index(rule["feature1"])
        i2 = FEATURES.index(rule["feature2"])
        v1, v2 = features[i1], features[i2]
        r1 = v1 >= rule["threshold1"] if rule["direction1"] == ">=" else v1 <= rule["threshold1"]
        r2 = v2 >= rule["threshold2"] if rule["direction2"] == ">=" else v2 <= rule["threshold2"]
        return bool(r1 and r2)
    return False


def main():
    df = load_data()
    feature_matrix, labels = df[FEATURES].values, df["category"].str.lower().values
    total = len(df)
    print(f"Loaded {total} validation samples")

    # Stage 1: Defect vs non-Defect
    print("\n=== Stage 1: Defect vs non-Defect ===")
    defect_mask = labels == "defect"
    defect_rule = search_best_rule(defect_mask, feature_matrix)
    if defect_rule:
        print(f"Best Defect rule (F1={defect_rule['f1']:.4f}, support={defect_rule['support']}):")
        print(format_rule(defect_rule))
    else:
        print("No Defect rule found")

    # Stage 2: Class_A vs Class_B among non-Defect
    print("\n=== Stage 2: Class_A vs Class_B (non-Defect only) ===")
    class_a_mask = labels == "class_a"
    target_ca = class_a_mask & ~defect_mask

    if defect_rule:
        defect_preds = np.array([apply_rule(feature_matrix[i], defect_rule) for i in range(total)])
        non_defect_idx = np.where(~defect_preds)[0]
    else:
        non_defect_idx = np.where(~defect_mask)[0]

    if len(non_defect_idx) == 0:
        print("No non-Defect samples found")
        ca_rule = None
    else:
        ca_rule = search_best_rule(target_ca[non_defect_idx], feature_matrix[non_defect_idx])
        if ca_rule:
            print(f"Best Class_A rule (F1={ca_rule['f1']:.4f}, support={ca_rule['support']}):")
            print(format_rule(ca_rule))
        else:
            print("No Class_A rule found")

    # Evaluate overall accuracy with priority rules
    print("\n=== Overall Validation Accuracy ===")
    predictions = []
    for i in range(total):
        feats = feature_matrix[i]
        if defect_rule and apply_rule(feats, defect_rule):
            predictions.append("defect")
        elif ca_rule and apply_rule(feats, ca_rule):
            predictions.append("class_a")
        else:
            predictions.append("class_b")

    predictions = np.array(predictions)
    accuracy = float(np.mean(predictions == labels))

    for cls in ["defect", "class_a", "class_b"]:
        mask = labels == cls
        tp = int(np.sum((predictions == cls) & mask))
        fp = int(np.sum((predictions == cls) & ~mask))
        fn = int(np.sum((predictions != cls) & mask))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = f1_score(tp, fp, fn)
        print(f"  {cls}: precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}, support={mask.sum()}")

    print(f"\nOverall validation accuracy: {accuracy:.4f}")

    # Save report
    lines = [
        "GLCM QUALITY CALIBRATION REPORT",
        "=" * 60, "",
        f"Validation accuracy: {accuracy:.4f}",
        f"Total samples: {total}", "",
    ]

    if defect_rule:
        lines.extend(["Defect rule:", format_rule(defect_rule),
                      f"  F1: {defect_rule['f1']:.4f}, Support: {defect_rule['support']}", ""])
    if ca_rule:
        lines.extend(["Class_A rule:", format_rule(ca_rule),
                      f"  F1: {ca_rule['f1']:.4f}, Support: {ca_rule['support']}", ""])

    lines.extend([
        "Priority order:",
        "  1. Defect",
        "  2. Class_A",
        "  3. Class_B fallback",
        "", "=" * 60,
    ])

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT_FILE}")
    return defect_rule, ca_rule


if __name__ == "__main__":
    main()
