"""Replay recordings through the exact firmware state machine (esp32_tinyfallnet/src/main.cpp) with the final model.
NOTE: the final model was trained on all of our data, so this checks the PIPELINE (trigger/window), not generalisation."""
import json, sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imu_classifier", "benchmark"))
import datasets as DS
from model import TinyFallNet
net = TinyFallNet(); net.load_state_dict(torch.load("model.pt")); net.eval(); thr = json.load(open("config.json"))["threshold"]

def firmware(a, g):
    """candidate c = sample with |a| >= 1.6 g that is the max of |a| over [c-50, c+50] (checked at k = c+50);
    evaluated at k = c+99 on window [c-50, c+100).  Mirrors main.cpp."""
    a = np.clip(a, -2, 2); s = np.linalg.norm(a, axis=1); pending, alarms = [], []
    for k in range(len(a)):
        c = k - 50
        if c >= 50 and s[c] >= 1.6 and s[c] >= s[c - 50:k + 1].max() and np.argmax(s[c - 50:k + 1]) == 50: pending.append(c)
        for c in [p for p in pending if k == p + 99]:
            w = np.hstack([a[c - 50:c + 100], g[c - 50:c + 100]]).astype(np.float32)
            with torch.no_grad(): p = float(torch.sigmoid(net(torch.from_numpy(w[None]))))
            alarms.append(p >= thr); pending.remove(c)
    return any(alarms)

for name, gen in [("falls", DS.own_falls()), ("lying", DS.own_lying()), ("normal", DS.own_normal())]:
    r = [(firmware(*tr["padded"]), tr.get("hours", 0)) for tr in gen]
    k = sum(x for x, _ in r); print(name, f"{k}/{len(r)} alarms" + (f", {k / sum(h for _, h in r):.2f} FA/h" if name == "normal" else ""))
