"""One table: every method x every dataset, 5 metrics (Accuracy, Precision, Recall, F1, AUC) + specificity and false alarms/h, trial level.
Sources:
  rules + frozen ML : imu_classifier/benchmark/results_<ds>.csv   (fixed rules, no training; AUC undefined for a yes/no rule)
  learned models    : trial_scores.csv  (train_eval.py, pooled 5-fold by subject over the 3 datasets)
  own-data protocol : own_scores/*.csv  (own_eval.py, 5-fold by person on our data, threshold @ 1 FA/h)"""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.join(HERE, "..", "imu_classifier", "benchmark")


def metrics(label, alarm, score=None, hours=None):
    label, alarm = np.asarray(label), np.asarray(alarm, bool); tp = int((alarm & (label == 1)).sum()); fp = int((alarm & (label == 0)).sum())
    neg = label == 0; tn = int(neg.sum()) - fp; p = tp / max(1, tp + fp); r = tp / max(1, (label == 1).sum())
    out = dict(Accuracy=100 * (tp + tn) / len(label), Precision=100 * p, Recall=100 * r, F1=100 * 2 * p * r / max(1e-9, p + r),
               AUC=100 * roc_auc_score(label, score) if score is not None else np.nan)
    if hours is not None:
        h = ~np.isnan(hours); out["Spec (ADL trials)"] = 100 * (~alarm[neg & ~h]).mean() if (neg & ~h).any() else np.nan
        out["FA/h"] = alarm[h].sum() / hours[h].sum() if h.any() else np.nan
    return out


rows = []
for ds in ["Own", "UNIVRFall", "SisFall"]:
    R = pd.read_csv(os.path.join(BENCH, f"results_{ds}.csv"))
    for m, d in R.groupby("method", sort=False):
        kind = "frozen ML (trained on Own)" if m.startswith("ML") else "rule"
        rows.append(dict(dataset=ds, group=kind, method=m, **metrics(d.label, d.alarm, None, d.hours.values)))

T = pd.read_csv(os.path.join(HERE, "trial_scores.csv")); T = T[T.protocol == "cv5_subject"]; T["ds"] = T.dataset.replace({"Own-normal": "Own"})
for (ds, m), d in T.groupby(["ds", "model"]):
    aucs = [roc_auc_score(f.label, f.score) for _, f in d.groupby("fold") if f.label.nunique() == 2]
    r = metrics(d.label, d.score >= d.thr, None, d.hours.values); r["AUC"] = 100 * np.mean(aucs)
    rows.append(dict(dataset=ds, group="learned, pooled 5-fold by subject", method=m, **r))

for p in sorted(glob.glob(os.path.join(HERE, "own_scores", "*.csv"))):
    d = pd.read_csv(p); rows.append(dict(dataset="Own", group="learned, own-data 5-fold by person @1 FA/h", method=d.model.iloc[0],
                                         **metrics(d.label, d.score >= d.thr, d.score, d.hours.values)))

S = pd.DataFrame(rows).round(2); S.to_csv(os.path.join(HERE, "results_all_5metrics.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45)
for ds in ["Own", "UNIVRFall", "SisFall"]:
    print(f"\n===== {ds}"); print(S[S.dataset == ds].drop(columns="dataset").to_string(index=False))
