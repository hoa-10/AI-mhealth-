"""Replay the deployed detector sample-by-sample over danh_gia data.
usage:  python -m fall_detector.replay_eval  [--normal-limit N]
 - fall_data_crop5s/person*  : a fall counts as detected if a FALL-tier alert has its impact within +-1 s of the crop's largest SVM peak (strict) / any FALL alert (lenient)
 - data_normal               : FALL-tier and SUSPECTED-tier alerts per hour
 - false_alarm               : files with a FALL-tier / SUSPECTED-tier alert
 - parity check              : events seen by the detector vs events used in training (same files)"""
import sys, glob, os, time, argparse, json
import numpy as np, pandas as pd
from . import FallDetector

ROOT = r"C:\Users\Admin\Desktop\IMU_data"
DG = os.path.join(ROOT, "dataset_IMU_collection", "danh_gia")
if not os.path.exists(DG):
    DG = os.path.join(ROOT, "danh_gia")
ap = argparse.ArgumentParser(); ap.add_argument("--normal-limit", type=int, default=0); ap.add_argument("--mode", default="tiered"); args = ap.parse_args()
det = FallDetector(mode=args.mode)
t0 = time.time()


def run(path):
    d = pd.read_csv(path); det.trace = []
    al = det.process_arrays(d.timestamp_ms.values, d[["accX_g", "accY_g", "accZ_g"]].values, d[["gyrX_dps", "gyrY_dps", "gyrZ_dps"]].values)
    return d, al, list(det.trace)


# ---------------- falls ----------------
rows = []
for p in sorted(glob.glob(DG + r"\fall_data_crop5s\person*\*.csv")):
    d, al, tr = run(p); a = d[["accX_g", "accY_g", "accZ_g"]].values; tpk = d.timestamp_ms.values[int(np.argmax(np.linalg.norm(a, axis=1)))]
    fall = [x for x in al if x.tier == "FALL"]; sus = [x for x in al if x.tier == "SUSPECTED"]
    rows.append(dict(person=os.path.basename(os.path.dirname(p)), file=os.path.basename(p), n_events=len(tr), lenient=bool(fall), strict=any(abs(x.t_impact_ms - tpk) <= 1000 for x in fall),
                     suspected_only=(not fall) and any(abs(x.t_impact_ms - tpk) <= 1000 for x in sus), latency_s=np.mean([(x.t_alert_ms - x.t_impact_ms) / 1000 for x in fall]) if fall else np.nan))
F = pd.DataFrame(rows)
print(f"[falls] {len(F)} crops | FALL-tier strict {int(F.strict.sum())}/{len(F)} ({F.strict.mean() * 100:.1f}%) | lenient {int(F.lenient.sum())} | only SUSPECTED {int(F.suspected_only.sum())} | median latency {F.latency_s.median():.2f} s")
print(F.groupby("person").agg(files=("file", "size"), strict=("strict", "sum"), susp_only=("suspected_only", "sum")).T.to_string())
print("missed:", F[~F.strict].file.tolist(), flush=True)

# ---------------- normals ----------------
files = [p for p in sorted(glob.glob(DG + r"\data_normal\*.csv")) if os.path.basename(p) != "manifest.csv"]
if args.normal_limit: files = files[::max(1, len(files) // args.normal_limit)]
hours = nfall = nsus = 0; rowsN = []; evN = 0
for p in files:
    d, al, tr = run(p); h = len(d) / 50 / 3600; hours += h; evN += len(tr)
    nfall += sum(x.tier == "FALL" for x in al); nsus += sum(x.tier == "SUSPECTED" for x in al)
    rowsN.append(dict(file=os.path.basename(p), activity=os.path.basename(p).split("_")[0], fall_alerts=sum(x.tier == "FALL" for x in al), susp=sum(x.tier == "SUSPECTED" for x in al)))
N = pd.DataFrame(rowsN)
print(f"\n[normal] {len(files)} files, {hours:.2f} h | FALL-tier alerts {nfall} = {nfall / hours:.2f}/h | SUSPECTED-tier {nsus} = {nsus / hours:.2f}/h | events evaluated {evN}")
print("FALL-tier alerts by activity:", N.groupby("activity").fall_alerts.sum().to_dict(), "| SUSPECTED by activity:", N.groupby("activity").susp.sum().to_dict(), flush=True)

# ---------------- false alarms ----------------
man = pd.read_csv(DG + r"\false_alarm\manifest.csv").set_index("file").scenario; rowsH = []
for p in sorted(glob.glob(DG + r"\false_alarm\*.csv")):
    if os.path.basename(p) == "manifest.csv": continue
    d, al, tr = run(p); rowsH.append(dict(file=os.path.basename(p), scenario=man.get(os.path.basename(p), ""), FALL=any(x.tier == "FALL" for x in al), SUSP=any(x.tier == "SUSPECTED" for x in al)))
Hh = pd.DataFrame(rowsH)
print(f"\n[false_alarm] {len(Hh)} files | FALL-tier {int(Hh.FALL.sum())}/{len(Hh)} | SUSPECTED-tier {int(Hh.SUSP.sum())}/{len(Hh)} | any {int((Hh.FALL | Hh.SUSP).sum())}/{len(Hh)}")
print(Hh.groupby("scenario").agg(files=("file", "size"), FALL=("FALL", "sum"), SUSPECTED=("SUSP", "sum")).to_string())
print(f"\nelapsed {time.time() - t0:.0f} s")
F.to_csv(os.path.dirname(os.path.abspath(__file__)) + r"\replay_falls.csv", index=False); Hh.to_csv(os.path.dirname(os.path.abspath(__file__)) + r"\replay_false_alarm.csv", index=False)
