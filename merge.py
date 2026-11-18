#!/usr/bin/env python3

import os, re, glob, warnings
import numpy as np
import pandas as pd

# ---------------- config ----------------
MAT_PATH   = "Propeller_fault_data.mat"
FORCE_DIR  = "Data_19_01_23_force"
OUT_CSV    = "merged.csv"
FS_ELEC    = 312497.5     # Hz, electrical capture
POLE_PAIRS = 7            # T200 is 12N14P -> mech rev/s = elec Hz / 7  (VERIFY)
TRIM       = 0.20         # drop first/last 20% of each record (step transients)
# ----------------------------------------

def load_mat(path):
    from pymatreader import read_mat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = read_mat(path)
    return {k: v for k, v in d.items() if not k.startswith("__")}

def steady(x, trim=TRIM):
    """middle portion of a record, avoiding start/stop transients."""
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 10:
        return x
    a = int(x.size * trim); b = int(x.size * (1 - trim))
    return x[a:b]

def digits(s):
    m = re.findall(r"\d+", s)
    return int(m[-1]) if m else None

# ---------------- force side ----------------
def read_force_dir(force_dir):
    """-> DataFrame [direction, propeller, throttle, thrust_N, thrust_std, n_force]"""
    rows = []
    for d in sorted(os.listdir(force_dir)):
        sub = os.path.join(force_dir, d)
        if not os.path.isdir(sub):
            continue
        m = re.fullmatch(r"(CCV|CV)(\d+)", d)
        if not m:
            continue
        direction, prop = m.group(1), int(m.group(2))
        for f in sorted(glob.glob(os.path.join(sub, "StairThruster*.mat"))):
            try:
                md = load_mat(f)
            except Exception as e:
                print(f"  [skip] {os.path.basename(f)}: {e}"); continue
            if "MeasForce" not in md or "seqThrottle" not in md:
                print(f"  [skip] {os.path.basename(f)}: missing MeasForce/seqThrottle"); continue
            thr = np.asarray(md["seqThrottle"]).ravel()
            thr = int(np.round(np.median(thr.astype(float))))  * 10  # constant per file
            fseg = steady(md["MeasForce"])
            rows.append({"direction": direction, "propeller": prop, "throttle": thr,
                         "thrust_N": float(np.mean(fseg)),
                         "thrust_std": float(np.std(fseg)),
                         "n_force": int(fseg.size),
                         "force_file": os.path.basename(f)})
    return pd.DataFrame(rows)

# ---------------- electrical side ----------------
def fundamental_hz(x, fs=FS_ELEC):
    x = steady(x, 0.25)
    if x.size < 64:
        return np.nan
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    fr = np.fft.rfftfreq(x.size, 1.0 / fs)
    X[(fr <= 5) | (fr > 600)] = 0
    return float(fr[np.argmax(X)])

def read_electrical(mat_path):
    """-> DataFrame [direction, propeller, throttle, power_W, I_rms, speed_rps, ...]"""
    data = load_mat(mat_path)
    rows = []
    for direction in ("CV", "CCV"):
        if direction not in data:
            continue
        for run, runv in data[direction].items():
            if run.lower() == "omega" or not isinstance(runv, dict):
                continue
            prop = digits(run)
            for ikey, ipt in runv.items():
                if not isinstance(ipt, dict) or "MeasCurrA" not in ipt:
                    continue
                thr = digits(ikey)  * 10                    # in110 -> 110
                pw, irms = 0.0, []
                for c, v in (("MeasCurrA","MeasVoltA"),
                             ("MeasCurrB","MeasVoltB"),
                             ("MeasCurrC","MeasVoltC")):
                    if c in ipt and v in ipt:
                        I = np.asarray(ipt[c], float).ravel()
                        V = np.asarray(ipt[v], float).ravel()
                        n = min(I.size, V.size)
                        pw += float(np.mean(I[:n] * V[:n]))
                        irms.append(float(np.sqrt(np.mean(I**2))))
                f_elec = fundamental_hz(np.asarray(ipt["MeasCurrA"], float).ravel())
                rows.append({"direction": direction, "propeller": prop, "throttle": thr,
                             "power_W": abs(pw),
                             "I_rms": float(np.mean(irms)) if irms else np.nan,
                             "f_elec_Hz": f_elec,
                             "speed_rps": f_elec / POLE_PAIRS if np.isfinite(f_elec) else np.nan})
    return pd.DataFrame(rows)

# ---------------- merge ----------------
def main():
    print("reading force files ...")
    F = read_force_dir(FORCE_DIR)
    print(f"  force rows: {len(F)}  propellers: {F.groupby(['direction','propeller']).ngroups}")
    print(f"  throttles seen: {sorted(F.throttle.unique())}")

    print("reading electrical dataset ...")
    E = read_electrical(MAT_PATH)
    print(f"  electrical rows: {len(E)}  propellers: {E.groupby(['direction','propeller']).ngroups}")
    print(f"  throttles seen: {sorted(E.throttle.unique())}")

    M = E.merge(F, on=["direction", "propeller", "throttle"], how="outer", indicator=True)

    both = M[M._merge == "both"].copy()
    print(f"\nmatched rows: {len(both)}")
    only_e = M[M._merge == "left_only"]; only_f = M[M._merge == "right_only"]
    if len(only_e):
        print(f"  electrical with NO force: {len(only_e)} -> "
              f"{sorted(set(zip(only_e.direction, only_e.propeller)))[:8]}")
    if len(only_f):
        print(f"  force with NO electrical: {len(only_f)} -> "
              f"{sorted(set(zip(only_f.direction, only_f.propeller)))[:8]}")

    # derived quantities
    both["speed_rad_s"]  = both.speed_rps * 2 * np.pi
    both["abs_thrust_N"] = both.thrust_N.abs()
    both["efficiency"]   = both.abs_thrust_N / both.power_W.replace(0, np.nan)   # N per W
    both = both.drop(columns=["_merge"]).sort_values(
        ["direction", "propeller", "throttle"]).reset_index(drop=True)
    both.to_csv(OUT_CSV, index=False)

    print(f"\n-> {OUT_CSV}  ({len(both)} rows, {both.shape[1]} cols)")
    print("\nsanity: mean thrust by throttle")
    print(both.groupby("throttle")["thrust_N"].mean().round(2).to_string())
    print("\nsanity: mean speed (rev/s) by throttle")
    print(both.groupby("throttle")["speed_rps"].mean().round(2).to_string())
    print("\nfirst rows:")
    print(both.head(10)[["direction","propeller","throttle","thrust_N",
                         "power_W","speed_rps","efficiency"]].to_string(index=False))

if __name__ == "__main__":
    main()