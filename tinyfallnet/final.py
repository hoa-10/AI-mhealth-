"""Deployable TinyFallNet, configuration chosen by own_eval.py (best on OUR data): TinyFallNet, no physics scalars,
trained on public (UNIVRFall + SisFall, clipped to +-2 g) + our data oversampled x5.
Threshold = 1 false alarm / hour on our normal recordings, from 3-fold-by-person out-of-fold scores (as in own_eval.py).
Then: ONE test on false_alarm (never used for any choice), export to C, save model.pt + config.json.
usage: python final.py"""
import os, sys, json, numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import own_eval as OE
from own_eval import M, X, VARIANTS, fit_predict, trial_table, thr_for_fa, scored, FA_H
from data import build, false_alarm
from model import TinyFallNet, count_macs
from train_eval import fit_nn
from export_c import export

V = VARIANTS["TinyFallNet (public+own)"]
own_idx = M.index[M.dataset.isin(["Own", "Own-normal"]).values]; Mo = M.loc[own_idx]
train_rows = own_idx[Mo.role.isin(["pos", "neg"]).values]

# threshold from out-of-fold scores on our data
oof = pd.Series(dtype=float)
for tr, te in GroupKFold(3).split(Mo, groups=Mo.subject):
    a = own_idx[tr]; a = a[M.loc[a, "role"].isin(["pos", "neg"]).values]; b = scored(own_idx[te])
    oof = pd.concat([oof, pd.Series(fit_predict(V, a, b, 7), index=b)])
T = trial_table(own_idx, oof); thr = thr_for_fa(T)
f, ly, nm = T[T.label == 1], T[T.task == "lie_down"], T[T.hours.notna()]
print(f"threshold {thr:.4f} @ {FA_H} FA/h  | OOF: falls {(f.score >= thr).sum()}/{len(f)}, lying {(ly.score >= thr).sum()}/{len(ly)}, "
      f"FA/h {(nm.score >= thr).sum() / nm.hours.sum():.2f}")

# final model on everything
rows = np.concatenate([np.repeat(train_rows, OE.OWN_REPEAT), OE.pub_rows]).astype(int)
y = (M.role.values[rows] == "pos").astype(int)
pr = fit_nn(V["make"], X[rows], y, epochs=30, seed=7)

# ---- final test on false_alarm, done once
Xf, Mf, _ = build("FalseAlarm", [false_alarm()])
sf = np.where(Mf.role.values == "none", 0.0, pr(Xf)); per_file = Mf.assign(score=sf).groupby("task").score.max()
man = pd.read_csv(os.path.join(os.path.dirname(HERE), "..", "dataset_IMU_collection", "danh_gia", "false_alarm", "manifest.csv")).set_index("file").scenario
hit = per_file[per_file >= thr]
print(f"\nFALSE_ALARM test: {len(hit)}/{len(per_file)} files flagged")
for fn, s in hit.items(): print(f"  {fn}  {s:.3f}  {man.get(fn, '')}")

net = pr.net.cpu(); torch.save(net.state_dict(), os.path.join(HERE, "model.pt"))
export(net, thr, X[train_rows][M.role.values[train_rows] == "pos"][0])
json.dump(dict(model="TinyFallNet (public+own)", threshold=thr, fa_per_h_target=FA_H, macs=count_macs(TinyFallNet()), params=sum(p.numel() for p in net.parameters()),
               window=dict(pre=50, post=100, fs=50), trigger="local max |a| >= 1.6 g, >= 1 s apart", acc_clip_g=2.0,
               false_alarm_flagged=int(len(hit)), false_alarm_files=int(len(per_file))),
          open(os.path.join(HERE, "config.json"), "w"), indent=2)
