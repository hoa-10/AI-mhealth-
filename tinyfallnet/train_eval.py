"""Fair comparison of TinyFallNet vs the deployed RF (43 features) vs 5 published DL architectures, SAME candidates, SAME folds.

Protocols (all split by SUBJECT, never by window):
  cv   : pooled UNIVRFall + SisFall + Own, 5-fold GroupKFold by subject
  lodo : leave-one-dataset-out (train on 2 datasets, test on the 3rd)
Inside every training set, 15% of subjects are held out as validation: the alarm threshold is set there
(97% trial-level specificity) and never touched on the test fold.  false_alarm is NOT used here (final.py only).
Unit of evaluation = trial: alarm if any of its candidates scores >= threshold.  Own normal recordings -> false alarms per hour.
GPU (env hoa).  usage: python train_eval.py cv lodo"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_archive_old_experiments", "sota_deep_learning_benchmark_2024"))
import models as sota
from data import load_all, ACC_CLIP
from model import TinyFallNet, Wrap, count_macs, N_CH
warnings.filterwarnings("ignore")
dev = torch.device("cuda"); assert torch.cuda.is_available(), "GPU required (env hoa)"
SPEC = 0.97


# ---------------------------------------------------------------- neural nets
def augment(xb):
    s = int(np.random.randint(-3, 4)); xb = torch.roll(xb, s, 1)                          # trigger jitter +-60 ms
    B = xb.shape[0]; sc = 0.95 + 0.1 * torch.rand(B, 1, 1, device=xb.device)
    a = xb[..., :3] * sc + 0.01 * torch.randn_like(xb[..., :3]); g = xb[..., 3:] * sc + 2.0 * torch.randn_like(xb[..., 3:])
    return torch.cat([a.clamp(-ACC_CLIP, ACC_CLIP), g], -1)


def fit_nn(make, X, y, epochs=30, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * int(np.ceil(len(X) / 256)))
    Xt = torch.from_numpy(X).to(dev); yt = torch.from_numpy(y.astype(np.float32)).to(dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(10.0, (y == 0).sum() / max(1, y.sum())), device=dev))
    for _ in range(epochs):
        net.train(); idx = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            if len(b) < 2: continue
            opt.zero_grad(); lossf(net(augment(Xt[b])), yt[b]).backward(); opt.step(); sch.step()
    net.eval()

    def predict(Xe):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xe), 4096): out.append(torch.sigmoid(net(torch.from_numpy(Xe[i:i + 4096]).to(dev))).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
    predict.net = net
    return predict


def fit_rf(F, y):
    m = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=0)).fit(F, y)
    return lambda Fe: m.predict_proba(Fe)[:, 1]


MODELS = {
    "TinyFallNet (proposed)": lambda: TinyFallNet(),
    "MultiScale1DCNN": lambda: Wrap(sota.MultiScale1DCNN(N_CH, 1)),
    "CNN-BiLSTM": lambda: Wrap(sota.CNN_BiLSTM(N_CH, 64, 1)),
    "Transformer": lambda: Wrap(sota.FallTransformer(N_CH, 64, 4, 2, 1)),
    "Dilated TCN": lambda: Wrap(sota.DilatedTCN(N_CH, [32, 48, 64, 64], 3, 0.2, 1)),
    "SE-EdgeNet1D": lambda: Wrap(sota.SE_EdgeNet1D(N_CH, 1)),
    "RF 43 features (deployed)": None,
}


# ---------------------------------------------------------------- evaluation
def trial_scores(M, s):
    return M.assign(s=s).groupby("trial").agg(score=("s", "max"), label=("label", "first"), dataset=("dataset", "first"), hours=("hours", "first"))


def pick_threshold(T):
    neg = np.sort(T[T.label == 0].score.values)
    return float(neg[int(np.ceil(SPEC * len(neg))) - 1]) + 1e-9


def report(T, thr):
    rows = []
    for ds, d in T.groupby("dataset"):
        f = d[d.label == 1]; n = d[(d.label == 0) & d.hours.isna()]; h = d[(d.label == 0) & d.hours.notna()]
        r = dict(test=ds, n_fall=len(f), sens=100 * (f.score >= thr).mean() if len(f) else np.nan,
                 n_adl=len(n), spec=100 * (n.score < thr).mean() if len(n) else np.nan,
                 fa_per_h=(h.score >= thr).sum() / h.hours.sum() if len(h) else np.nan)
        neg_all = d[d.label == 0]
        r["auc"] = 100 * roc_auc_score(d.label, d.score) if len(f) and len(neg_all) else np.nan
        rows.append(r)
    return rows


def run_split(name, X, M, F, tr_mask, te_mask, protocol, fold):
    """train on tr_mask trials (85% train / 15% val by subject), evaluate on te_mask."""
    Mtr = M[tr_mask]; gss = GroupShuffleSplit(1, test_size=0.15, random_state=fold)
    fit_i, val_i = next(gss.split(Mtr, groups=Mtr.subject)); fit_idx, val_idx = Mtr.index[fit_i], Mtr.index[val_i]
    use = fit_idx[M.loc[fit_idx, "role"].isin(["pos", "neg"]).values]; y = (M.loc[use, "role"] == "pos").values.astype(int)
    val_rows = val_idx[(M.loc[val_idx, "role"] != "none").values]; te_idx = M.index[te_mask]; te_rows = te_idx[(M.loc[te_idx, "role"] != "none").values]
    t0 = time.time()
    if MODELS[name] is None:
        pr = fit_rf(F[use], y); sv, st = pr(F[val_rows]), pr(F[te_rows])
    else:
        pr = fit_nn(MODELS[name], X[use], y, seed=fold); sv, st = pr(X[val_rows]), pr(X[te_rows])
    s_val = pd.Series(0.0, index=val_idx); s_val[val_rows] = sv
    s_te = pd.Series(0.0, index=te_idx); s_te[te_rows] = st
    thr = pick_threshold(trial_scores(M.loc[val_idx], s_val))
    T = trial_scores(M.loc[te_idx], s_te)
    print(f"  {protocol} fold {fold} {name}: {time.time() - t0:.0f} s, thr {thr:.3f}", flush=True)
    return [dict(protocol=protocol, fold=fold, model=name, thr=thr, **r) for r in report(T, thr)], T.assign(model=name, protocol=protocol, fold=fold, thr=thr)


def summarize(R):
    def agg(d):   # pool counts across folds (weighted by n)
        f, n = d[d.n_fall > 0], d[d.n_adl > 0]
        return pd.Series(dict(sens=np.average(f.sens, weights=f.n_fall) if len(f) else np.nan,
                              spec=np.average(n.spec, weights=n.n_adl) if len(n) else np.nan,
                              fa_per_h=d.fa_per_h.mean(), auc=d.auc.mean()))
    return R.groupby(["protocol", "model", "test"]).apply(agg).round(2).reset_index()


if __name__ == "__main__":
    X, M, F = load_all()
    which = sys.argv[1:] or ["cv", "lodo"]
    R, Ts = [], []
    for name in MODELS:
        if MODELS[name] is not None:
            net = MODELS[name](); print(f"{name}: params {sum(p.numel() for p in net.parameters()):,}, MACs {count_macs(net):,}", flush=True)
        if "cv" in which:
            for fold, (tr, te) in enumerate(GroupKFold(5).split(M, groups=M.subject)):
                tr_m = np.zeros(len(M), bool); tr_m[tr] = True
                r, T = run_split(name, X, M, F, tr_m, ~tr_m, "cv5_subject", fold); R += r; Ts.append(T)
        if "lodo" in which:
            for fold, ds in enumerate(["UNIVRFall", "SisFall", "Own"]):
                te_m = (M.dataset == ds).values | ((ds == "Own") & (M.dataset == "Own-normal")).values
                r, T = run_split(name, X, M, F, ~te_m, te_m, f"lodo", fold); R += r; Ts.append(T)
        pd.DataFrame(R).to_csv(os.path.join(HERE, "results_folds.csv"), index=False)
    S = summarize(pd.DataFrame(R)); S.to_csv(os.path.join(HERE, "results_summary.csv"), index=False)
    pd.concat(Ts).to_csv(os.path.join(HERE, "trial_scores.csv"))
    pd.set_option("display.width", 200); print(S.to_string(index=False))
