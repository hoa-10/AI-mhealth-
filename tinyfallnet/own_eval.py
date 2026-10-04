"""Evaluation centred on OUR data (danh_gia): which model is best for our device?

Outer: 5-fold GroupKFold by person over our trials (67 falls, 10 lying-down, 368 normal recordings ~6 h).
Threshold: from inner 3-fold (by person) out-of-fold scores on the TRAINING part only, set to 1 false alarm / hour on
training normal recordings (same idea as the deployed RF's 2 FA/h threshold).  false_alarm is not used.
Public data (UNIVRFall + SisFall, clipped to +-2 g) can be added to training as extra examples; our data is oversampled x5.
usage: python own_eval.py"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from data import load_all
from model import TinyFallNet, count_macs
from train_eval import fit_nn, fit_rf
warnings.filterwarnings("ignore")
FA_H = 1.0
OWN_REPEAT = 5

X, M, F = load_all()
own = M.dataset.isin(["Own", "Own-normal"]).values
pub_rows = M.index[~own & M.role.isin(["pos", "neg"]).values]

VARIANTS = {
    "RF 43 features (own)": dict(kind="rf", public=False),
    "TinyFallNet (own)": dict(kind="nn", make=lambda: TinyFallNet(False), public=False),
    "TinyFallNet (public+own)": dict(kind="nn", make=lambda: TinyFallNet(False), public=True),
    "TinyFallNet-Phys (own)": dict(kind="nn", make=lambda: TinyFallNet(True), public=False),
    "TinyFallNet-Phys (public+own)": dict(kind="nn", make=lambda: TinyFallNet(True), public=True),
}


def fit_predict(v, tr_rows, te_rows, seed):
    """tr_rows: our training candidate rows (pos/neg).  returns scores for te_rows."""
    rows = np.concatenate([np.repeat(tr_rows, OWN_REPEAT if v["public"] else 1), pub_rows if v["public"] else []]).astype(int)
    y = (M.role.values[rows] == "pos").astype(int)
    if v["kind"] == "rf": return fit_rf(F[rows], y)(F[te_rows])
    return fit_nn(v["make"], X[rows], y, epochs=30, seed=seed)(X[te_rows])


def trial_table(idx, scores):
    s = pd.Series(0.0, index=idx); s[scores.index] = scores.values
    return M.loc[idx].assign(s=s).groupby("trial").agg(score=("s", "max"), label=("label", "first"), task=("task", "first"),
                                                         subject=("subject", "first"), hours=("hours", "first"))


def thr_for_fa(T, fa_h=FA_H):
    n = T[T.hours.notna()]; hrs = n.hours.sum(); k = int(np.floor(fa_h * hrs))      # allow k alarms in hrs hours
    sc = np.sort(n.score.values)[::-1]
    return float(sc[k]) + 1e-9 if k < len(sc) else 0.0


def scored(idx):
    return idx[M.loc[idx, "role"].isin(["pos", "neg", "ign"]).values]


if __name__ == "__main__":
    own_idx = M.index[own]; Mo = M.loc[own_idx]
    rows = []; allT = []
    pick = sys.argv[1:]
    for name, v in VARIANTS.items():
        if pick and not any(p in name for p in pick): continue
        t0 = time.time(); Ts = []
        for fold, (tr, te) in enumerate(GroupKFold(5).split(Mo, groups=Mo.subject)):
            tr_idx, te_idx = own_idx[tr], own_idx[te]
            # inner OOF on the training part -> threshold
            oof = pd.Series(dtype=float); Mi = M.loc[tr_idx]
            for itr, ite in GroupKFold(3).split(Mi, groups=Mi.subject):
                a, b = tr_idx[itr], tr_idx[ite]; a = a[M.loc[a, "role"].isin(["pos", "neg"]).values]; b = scored(b)
                oof = pd.concat([oof, pd.Series(fit_predict(v, a, b, fold), index=b)])
            thr = thr_for_fa(trial_table(tr_idx, oof))
            a = tr_idx[M.loc[tr_idx, "role"].isin(["pos", "neg"]).values]; b = scored(te_idx)
            T = trial_table(te_idx, pd.Series(fit_predict(v, a, b, fold), index=b)).assign(thr=thr, fold=fold); Ts.append(T)
        T = pd.concat(Ts); T["alarm"] = T.score >= T.thr; allT.append(T.assign(model=name))
        os.makedirs(os.path.join(HERE, "own_scores"), exist_ok=True)
        T.assign(model=name).to_csv(os.path.join(HERE, "own_scores", name.replace(" ", "_").replace("/", "") + ".csv"))
        f, ly, nm = T[T.label == 1], T[T.task == "lie_down"], T[T.hours.notna()]
        tp, fp = int(f.alarm.sum()), int((T.alarm & (T.label == 0)).sum()); tn = int((T.label == 0).sum()) - fp
        prec = tp / max(1, tp + fp); rec = tp / len(f)
        rows.append({"model": name, "Accuracy": 100 * (tp + tn) / len(T), "Precision": 100 * prec, "Recall": 100 * rec,
                     "F1": 100 * 2 * prec * rec / max(1e-9, prec + rec), "AUC": 100 * roc_auc_score(T.label, T.score), "AP": 100 * average_precision_score(T.label, T.score),
                     "falls caught": f"{tp}/{len(f)}", "lying alarms": f"{int(ly.alarm.sum())}/{len(ly)}", "FA per h": round(nm.alarm.sum() / nm.hours.sum(), 2), "time_s": round(time.time() - t0)})
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows).round(2); R.to_csv(os.path.join(HERE, "results_own.csv"), index=False)
    pd.concat(allT).to_csv(os.path.join(HERE, "trial_scores_own.csv"))
    for mk in (False, True): print("MACs TinyFallNet phys=%s: %d" % (mk, count_macs(TinyFallNet(mk))))
    pd.set_option("display.width", 220); print(R.to_string(index=False))
