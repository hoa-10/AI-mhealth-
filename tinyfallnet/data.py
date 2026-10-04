"""Impact-candidate windows from all datasets, in one format, cached.

Candidate = local max of |a| >= 1.6 g, >= 1 s apart (same trigger as the deployed detector).
Window   = 150 samples (3 s) = [k-50, k+100) at 50 Hz, raw acc (g) + gyro (deg/s), shape (150, 6).
Sensor match: our device's accelerometer is +-2 g per axis (every fall saturates at 1.996 g), so every dataset is
clipped per axis to +-2 g and +-2000 deg/s BEFORE candidate search -> public data looks like what the ESP32 sees.

Also stores the 43 hand-crafted features of the deployed RF+HistGB, so the RF baseline sees exactly the same candidates.
Roles: pos = the impact of a fall trial (nearest labelled impact for UNIVRFall, else the largest peak);
       ign = other candidates of a fall trial (not used for training);  neg = every candidate of a non-fall trial.
false_alarm (= test_anomaly copies) is built separately and is used ONLY for the final test."""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(SRC, "imu_classifier", "benchmark"))
import datasets as DS
from fall_detector_logic.features import extract

FEATS = json.load(open(os.path.join(SRC, "fall_detector_logic", "artifacts", "config.json")))["features"]
PRE, POST, WIN = 50, 100, 150
ACC_CLIP, GYR_CLIP = 2.0, 2000.0
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)


def sensor_match(a, g):
    return np.clip(a, -ACC_CLIP, ACC_CLIP), np.clip(g, -GYR_CLIP, GYR_CLIP)


def candidates(a):
    s = np.linalg.norm(a, axis=1); pk, _ = find_peaks(s, height=1.6, distance=50)
    return [int(k) for k in pk if k - PRE >= 0 and k + POST <= len(a)], s


def window(a, g, k):
    return np.hstack([a[k - PRE:k + POST], g[k - PRE:k + POST]]).astype(np.float32)


def build(name, gens, with_feats=True):
    p = os.path.join(CACHE, name + ".npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return z["X"], pd.DataFrame(z["meta"].tolist()), z["F"]
    X, meta, F, tid = [], [], [], 0
    for gen in gens:
        for tr in gen:
            a, g = sensor_match(*tr["padded"]); pk, s = candidates(a); pos = None
            if tr["label"] == 1 and pk:
                pos = min(pk, key=lambda k: abs(k - tr["impact"])) if tr.get("impact") is not None else max(pk, key=lambda k: s[k])
            base = dict(trial=tid, dataset=tr["dataset"], subject=str(tr["subject"]), label=tr["label"], task=str(tr["task"]), hours=tr.get("hours", np.nan))
            for k in pk:
                X.append(window(a, g, k)); meta.append(dict(base, role="pos" if k == pos else ("ign" if tr["label"] == 1 else "neg"), k=k))
                if with_feats:
                    f = extract(a[k - PRE:k + POST + 1], g[k - PRE:k + POST + 1], 50.0, PRE); F.append([f.get(c, np.nan) for c in FEATS])
            if not pk:      # trial without a candidate: can never alarm, but must still be counted
                meta.append(dict(base, role="none", k=-1)); X.append(np.zeros((WIN, 6), np.float32)); F.append([np.nan] * len(FEATS))
            tid += 1
            if tid % 500 == 0: print(f"  {name}: {tid} trials", flush=True)
    X, F = np.stack(X), np.array(F, dtype=np.float32) if F else np.zeros((len(X), 0), np.float32)
    np.savez_compressed(p, X=X, meta=np.array(meta, dtype=object), F=F)
    return X, pd.DataFrame(meta), F


def false_alarm():
    """test-only: the 64 hard false-alarm scenarios (copies of test_anomaly)."""
    d = os.path.join(DS.ROOT, "danh_gia", "false_alarm")
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        if os.path.basename(p) == "manifest.csv": continue
        a, g = DS._read(p)
        yield dict(dataset="FalseAlarm", subject="test", label=0, task=os.path.basename(p), padded=DS._pad(a, g))


def load_all():
    parts = {"UNIVRFall": [DS.univrfall()], "SisFall": [DS.sisfall()], "Own": [DS.own_falls(), DS.own_lying(), DS.own_normal()]}
    Xs, Ms, Fs, off = [], [], [], 0
    for n, gens in parts.items():
        X, M, F = build(n, gens); M = M.copy(); M["trial"] += off; off = M.trial.max() + 1
        M["subject"] = M.dataset.str[:1] + "_" + M.subject           # unique across datasets
        Xs.append(X); Ms.append(M); Fs.append(F); print(f"{n}: {M.trial.nunique()} trials, {len(M)} rows, pos {(M.role == 'pos').sum()}, neg {(M.role == 'neg').sum()}", flush=True)
    return np.concatenate(Xs), pd.concat(Ms, ignore_index=True), np.concatenate(Fs)


if __name__ == "__main__":
    load_all(); build("FalseAlarm", [false_alarm()])
