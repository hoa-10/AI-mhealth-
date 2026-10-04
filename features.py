"""Hand-crafted feature bank for ONE impact candidate (window: -1.5 s .. +3 s around the SVM peak).
All features are mounting-invariant (magnitudes, angles between gravity vectors,
vertical/horizontal split relative to the PRE-impact gravity)."""
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial.transform import Rotation as R

S4 = np.sqrt(3)
DB4_LO = np.array([(1 + S4), (3 + S4), (3 - S4), (1 - S4)]) / (4 * np.sqrt(2))  # Daubechies-4 low-pass
DB4_HI = np.array([DB4_LO[3], -DB4_LO[2], DB4_LO[1], -DB4_LO[0]])


def ang(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def dwt_energy(x, levels=4):
    """db4 DWT: relative detail-energy per level (+ approximation) and wavelet entropy."""
    x = np.asarray(x, float)
    x = x - x.mean()
    en = []
    for _ in range(levels):
        if len(x) < 8:
            en.append(0.0)
            continue
        hi = np.convolve(x, DB4_HI[::-1], mode="valid")[::2]
        lo = np.convolve(x, DB4_LO[::-1], mode="valid")[::2]
        en.append(float((hi ** 2).sum()))
        x = lo
    en.append(float((x ** 2).sum()))
    e = np.array(en)
    r = e / (e.sum() + 1e-9)
    return r, float(-(r[r > 0] * np.log(r[r > 0])).sum())


def spec(x, fs, nfft=128):
    x = np.asarray(x, float)
    x = (x - x.mean()) * np.hanning(len(x))
    return np.fft.rfftfreq(nfft, 1 / fs), np.abs(np.fft.rfft(x, nfft)) ** 2


def spec_feats(x, fs, tag):
    keys = ["domf", "domratio", "entropy", "b_low", "b_mid", "b_high", "centroid"]
    if len(x) < 12:
        return {f"{tag}_{k}": np.nan for k in keys}
    f, P = spec(x, fs)
    m = (f >= 0.4) & (f <= fs / 2 - 0.5)
    tot = P[m].sum() + 1e-9
    out = {f"{tag}_domf": float(f[m][np.argmax(P[m])]), f"{tag}_domratio": float(P[m].max() / tot)}
    p = P[m] / tot
    out[f"{tag}_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    for nm, (lo, hi) in {"b_low": (0.4, 3), "b_mid": (3, 8), "b_high": (8, fs / 2)}.items():
        out[f"{tag}_{nm}"] = float(P[(f >= lo) & (f < hi)].sum() / tot)
    out[f"{tag}_centroid"] = float((f[m] * P[m]).sum() / tot)
    return out


def autocorr_peak(x, fs, lo=0.3, hi=1.2):
    x = np.asarray(x, float)
    x = x - x.mean()
    v = (x ** 2).sum()
    if len(x) < int(hi * fs) + 5 or v < 1e-6:
        return np.nan
    return float(max((x[:-l] * x[l:]).sum() / v for l in range(int(lo * fs), int(hi * fs))))


def extract(a, g, fs, k):
    n = len(a)
    s = lambda t: int(round(t * fs))
    svm = np.linalg.norm(a, axis=1)
    gm = np.linalg.norm(g, axis=1)
    F = {}
    nan = np.nan

    def seg(t0, t1):
        return slice(max(0, k + s(t0)), min(n, k + s(t1)))

    def ok(sl, m=4):
        return (sl.stop - sl.start) >= m

    pre, pre0 = seg(-2.5, -0.5), seg(-1.5, -0.2)
    post, late = seg(0.3, 1.3), seg(1.3, 3.0)

    # ---------- T: time-domain impact features ----------
    lo = max(0, k - s(0.7))
    F["T_peak"] = svm[k]
    F["T_dipmin"] = svm[lo:k + 1].min()
    F["T_t_dip2peak"] = (k - (lo + int(np.argmin(svm[lo:k + 1])))) / fs
    bs = seg(-2.5, -0.7)
    F["T_dipdepth"] = (np.median(svm[bs]) - F["T_dipmin"]) if ok(bs) else nan
    w = seg(-0.5, 0.5)
    F["T_jerk"] = float((np.abs(np.diff(svm[w])) * fs).max()) if ok(w) else nan
    F["T_std_imp"] = svm[w].std()
    F["T_range_imp"] = float(np.ptp(svm[w]))
    F["T_skew_imp"] = float(skew(svm[w]))
    F["T_kurt_imp"] = float(kurtosis(svm[w]))
    F["T_gyro_peak"] = gm[w].max()
    F["T_gyro_mean_imp"] = gm[w].mean()
    F["T_impulse"] = float(((svm[seg(-0.3, 0.3)] - 1) ** 2).sum() / fs)
    F["T_acc_gyro_lag"] = (int(np.argmax(gm[w])) - int(np.argmax(svm[w]))) / fs

    # ---------- O: orientation / gravity (mounting-invariant) ----------
    gp = a[pre].mean(0) if ok(pre, 6) else None
    for nm, sl in [("A", seg(0.3, 1.3)), ("B", seg(1.0, 3.0)), ("C", seg(2.0, 4.0))]:
        F["O_tilt_" + nm] = ang(gp, a[sl].mean(0)) if (gp is not None and ok(sl, 10)) else nan
    r0, r1 = max(0, k - s(0.5)), min(n, k + s(1.5))
    if r1 - r0 > s(1.0):
        Rt = R.identity()
        for q in R.from_rotvec(np.radians(g[r0:r1]) / fs):
            Rt = Rt * q
        F["O_rot_net"] = float(np.degrees(np.linalg.norm(Rt.as_rotvec())))
    else:
        F["O_rot_net"] = nan
    if gp is not None:
        u = gp / np.linalg.norm(gp)
        v = a @ u
        h = np.linalg.norm(a - np.outer(v, u), axis=1)
        wi = seg(-0.3, 0.3)
        F["O_peak_v"] = float(v[wi].max())
        F["O_peak_h"] = float(h[wi].max())
        F["O_h_over_v"] = F["O_peak_h"] / (abs(F["O_peak_v"]) + 1e-3)
        F["O_post_v"] = float(v[post].mean()) if ok(post) else nan
        F["O_post_h"] = float(h[post].mean()) if ok(post) else nan
        F["O_late_v"] = float(v[late].mean()) if ok(late) else nan
        F["O_v_min_pre"] = float(v[seg(-0.7, 0)].min())
    else:
        for c in ["peak_v", "peak_h", "h_over_v", "post_v", "post_h", "late_v", "v_min_pre"]:
            F["O_" + c] = nan
    ps = a[pre]
    hh = len(ps) // 2
    F["O_pre_drift"] = ang(ps[:hh].mean(0), ps[hh:].mean(0)) if hh > 3 else nan
    pa, pb = seg(0.3, 1.3), seg(1.3, 2.3)
    F["O_post_drift"] = ang(a[pa].mean(0), a[pb].mean(0)) if (ok(pa, 10) and ok(pb, 10)) else nan

    # ---------- P: phase statistics (pre / post / late) ----------
    for nm, sl in [("pre", pre0), ("post", post), ("late", late)]:
        if ok(sl, 6):
            F[f"P_{nm}_svm_mean"] = svm[sl].mean()
            F[f"P_{nm}_svm_std"] = svm[sl].std()
            F[f"P_{nm}_gyro_mean"] = gm[sl].mean()
            F[f"P_{nm}_gyro_std"] = gm[sl].std()
            F[f"P_{nm}_still"] = float((gm[sl] < 30).mean())
        else:
            for c in ["svm_mean", "svm_std", "gyro_mean", "gyro_std", "still"]:
                F[f"P_{nm}_{c}"] = nan
    F["P_post_over_pre_std"] = F["P_post_svm_std"] / (F["P_pre_svm_std"] + 1e-3)
    F["P_post_over_pre_gyro"] = F["P_post_gyro_mean"] / (F["P_pre_gyro_mean"] + 1)
    pw = svm[seg(0.3, 2.5)]
    F["P_post_peaks"] = float((pw > 1.5).sum()) if len(pw) > s(1.0) else nan
    qw = svm[seg(-2.5, -0.3)]
    F["P_pre_peaks"] = float((qw > 1.5).sum()) if len(qw) > s(1.0) else nan

    # ---------- F: frequency domain ----------
    F.update(spec_feats(svm[pre0], fs, "F_pre_svm"))
    F.update(spec_feats(svm[seg(0.3, 2.3)], fs, "F_post_svm"))
    F.update(spec_feats(gm[pre0], fs, "F_pre_gyro"))
    F.update(spec_feats(gm[seg(0.3, 2.3)], fs, "F_post_gyro"))
    isl = seg(-0.5, 0.6)
    if ok(isl, 12):
        f_, P_ = spec(svm[isl], fs)
        F["F_imp_highfrac"] = float(P_[f_ >= 8].sum() / (P_.sum() + 1e-9))
    else:
        F["F_imp_highfrac"] = nan
    F["F_pre_rhythm"] = autocorr_peak(svm[seg(-2.5, -0.3)], fs)
    F["F_post_rhythm"] = autocorr_peak(svm[seg(0.3, 2.5)], fs)

    # ---------- W: wavelet (db4, 4 levels) over +-1 s ----------
    wsl = seg(-1.0, 1.0)
    for tag, x in [("svm", svm), ("gyro", gm)]:
        if ok(wsl, 32):
            r, e = dwt_energy(x[wsl], 4)
            for i, val in enumerate(r):
                F[f"W_{tag}_L{i + 1}"] = float(val)
            F[f"W_{tag}_entropy"] = e
        else:
            for i in range(5):
                F[f"W_{tag}_L{i + 1}"] = nan
            F[f"W_{tag}_entropy"] = nan
    return F
