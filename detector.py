"""Streaming fall detector (50 Hz IMU).  Pipeline: impact trigger -> 3 s window [-1.0 s, +2.0 s] -> physics gate -> ML ensemble -> physics rule -> tiered alert.
 push(t_ms, acc[3], gyr[3]) is called for every sample; alerts are returned ~2.0 s after the impact (the window needs the 'after' part).
Gap handling (missing packets): gaps <= 0.4 s are linearly interpolated, longer gaps reset the buffer (an impact waiting for its window is dropped)."""
import os, json
from dataclasses import dataclass, field
import numpy as np, pandas as pd, joblib
from .features import extract

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 50.0
GAP_MS, FILL_MAX_MS = 150.0, 400.0


@dataclass
class Alert:
    t_impact_ms: float            # timestamp of the impact (SVM peak)
    t_alert_ms: float             # timestamp when the alert was issued (impact + window 'after' part)
    tier: str                     # 'FALL' | 'SUSPECTED'
    score: float                  # ensemble probability
    tilt_deg: float               # posture change before->after the impact
    peak_g: float
    model_hit: bool
    rule_hit: bool
    extra: dict = field(default_factory=dict)


class FallDetector:
    def __init__(self, artifacts_dir=None, mode="tiered", fp_budget=2):
        adir = artifacts_dir or os.path.join(HERE, "artifacts")
        self.cfg = json.load(open(os.path.join(adir, "config.json")))
        pack = joblib.load(os.path.join(adir, "model.joblib"))
        self.models, self.features = list(pack["models"].values()), pack["features"]
        self.pre = int(round(self.cfg["window"]["pre_s"] * FS)); self.post = int(round(self.cfg["window"]["post_s"] * FS))
        self.trigger_g = self.cfg["trigger_peak_g"]; self.refractory = int(round(self.cfg["trigger_min_gap_s"] * FS)) - 1
        self.gate = self.cfg["gate_tilt_deg"]; self.rule_tilt = self.cfg["physics_rule"]["tilt_deg"]
        self.thr = {1: self.cfg["threshold_prob_1FPh"], 2: self.cfg["threshold_prob_2FPh"], 4: self.cfg["threshold_prob_4FPh"]}[fp_budget]
        assert mode in ("tiered", "model", "rule"); self.mode = mode
        self.reset()

    # ---------------- stream state ----------------
    def reset(self):
        self._a, self._g, self._t = [], [], []
        self._offset = 0                 # global index of buffer[0]
        self._prev_t = None
        self._last_eval = -10 ** 9       # global index of the last evaluated impact
        if not hasattr(self, 'trace'): self.trace = []

    def push(self, t_ms, acc, gyr):
        alerts = []
        if self._prev_t is not None:
            dt = t_ms - self._prev_t
            if dt > FILL_MAX_MS:
                self.reset()
            elif dt > GAP_MS:
                m = int(round(dt / 20.0)) - 1
                a0, g0, t0 = self._a[-1], self._g[-1], self._prev_t
                for j in range(1, m + 1):
                    w = j / (m + 1)
                    alerts += self._append(t0 + w * dt, a0 * (1 - w) + np.asarray(acc, float) * w, g0 * (1 - w) + np.asarray(gyr, float) * w)
        self._prev_t = t_ms
        return alerts + self._append(t_ms, np.asarray(acc, float), np.asarray(gyr, float))

    def _append(self, t_ms, a, g):
        self._a.append(a); self._g.append(g); self._t.append(t_ms)
        n = len(self._a); j = n - 1 - self.post
        out = []
        if j >= 1:
            al = self._maybe_event(j, n)
            if al: out.append(al)
        if n > 1200:                      # trim, keep ~24 s
            cut = n - 800; self._a, self._g, self._t = self._a[cut:], self._g[cut:], self._t[cut:]; self._offset += cut
        return out

    def flush(self):
        """end of recording: evaluate impacts whose 'after' part is shorter than the window (uses what is available)."""
        out = []; n = len(self._a)
        for j in range(max(1, n - self.post), n - 1):
            al = self._maybe_event(j, n, truncated=True)
            if al: out.append(al)
        return out

    # ---------------- event logic ----------------
    def _maybe_event(self, j, n, truncated=False):
        A = np.asarray(self._a[max(0, j - self.refractory):min(n, j + self.refractory + 1)]); svm_win = np.linalg.norm(A, axis=1)
        s = float(np.linalg.norm(self._a[j]))
        if s < self.trigger_g or s < svm_win.max() - 1e-12 or (j + self._offset) - self._last_eval <= self.refractory:
            return None
        if np.argmax(svm_win) != (j - max(0, j - self.refractory)):          # first maximum wins ties
            return None
        if not (s >= np.linalg.norm(self._a[j - 1]) and (j + 1 >= n or s > np.linalg.norm(self._a[j + 1]))):
            return None
        self._last_eval = j + self._offset
        lo, hi = max(0, j - self.pre), min(n, j + self.post + 1)
        a = np.asarray(self._a[lo:hi]); g = np.asarray(self._g[lo:hi])
        f = extract(a, g, FS, j - lo)
        tilt = f.get("O_tilt_A", np.nan); tilt = 0.0 if tilt != tilt else float(tilt)
        score = 0.0
        if tilt >= self.gate:                                       # physics gate: the ML ensemble only runs when the posture changed a little at least
            X = pd.DataFrame([[f.get(c, np.nan) for c in self.features]], columns=self.features)
            score = float(np.mean([m.predict_proba(X)[:, 1][0] for m in self.models]))
        model_hit = bool(tilt >= self.gate and score >= self.thr); rule_hit = bool(s >= self.cfg["physics_rule"]["peak_g"] and tilt >= self.rule_tilt)
        if self.mode == "tiered":
            tier = "FALL" if (model_hit and rule_hit) else ("SUSPECTED" if model_hit else None)
        elif self.mode == "model":
            tier = "FALL" if model_hit else None
        else:
            tier = "FALL" if rule_hit else None
        self.trace.append(dict(t_impact_ms=self._t[j], peak_g=s, tilt_deg=tilt, score=score, model_hit=model_hit, rule_hit=rule_hit, tier=tier))
        if tier is None:
            return None
        return Alert(t_impact_ms=self._t[j], t_alert_ms=self._t[min(n - 1, j + self.post)], tier=tier, score=score, tilt_deg=tilt, peak_g=s, model_hit=model_hit, rule_hit=rule_hit,
                     extra=dict(truncated=truncated))

    # ---------------- convenience ----------------
    def process_arrays(self, t_ms, acc, gyr):
        self.reset(); out = []
        for t, a, g in zip(t_ms, acc, gyr):
            out += self.push(float(t), a, g)
        return out + self.flush()

    def process_csv(self, path):
        d = pd.read_csv(path)
        return self.process_arrays(d.timestamp_ms.values, d[["accX_g", "accY_g", "accZ_g"]].values, d[["gyrX_dps", "gyrY_dps", "gyrZ_dps"]].values)
