"""Recompute the metrics of own_eval.py from trial_scores_own.csv (trial level, per-fold thresholds)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import glob
T = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("own_scores/*.csv"))]); T["alarm"] = T.score >= T.thr
rows = []
for m, d in T.groupby("model", sort=False):
    f, ly, nm = d[d.label == 1], d[d.task == "lie_down"], d[d.hours.notna()]
    tp, fp = int(f.alarm.sum()), int((d.alarm & (d.label == 0)).sum()); tn = int((d.label == 0).sum()) - fp
    p, r = tp / max(1, tp + fp), tp / len(f)
    rows.append({"model": m, "Accuracy": 100 * (tp + tn) / len(d), "Precision": 100 * p, "Recall": 100 * r, "F1": 100 * 2 * p * r / max(1e-9, p + r),
                 "AUC": 100 * roc_auc_score(d.label, d.score), "AP": 100 * average_precision_score(d.label, d.score),
                 "falls": f"{tp}/{len(f)}", "lying FA": f"{int(ly.alarm.sum())}/{len(ly)}", "FA/h": nm.alarm.sum() / nm.hours.sum()})
R = pd.DataFrame(rows).round(2); R.to_csv("results_own.csv", index=False)
pd.set_option("display.width", 220); print(R.to_string(index=False))
