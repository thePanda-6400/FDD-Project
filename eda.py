#!/usr/bin/env python3
"""
Exploratory Data Analysis for the AAU incipient propeller-fault dataset
(Propeller_fault_data.mat): groups CCV and CV, each a struct of numbered runs
plus 'omega'. Signal fields may be nested inside sub-structs, so this version
DISCOVERS them recursively and never assumes a field is a flat numeric vector.

Run:
    conda activate rov-fdd
    python eda.py

Then paste back the printed "STRUCTURE OF FIRST RUN" and "OMEGA" trees plus the
"WHAT TO TELL CLAUDE" block so the signals, sampling rate, and labels can be fixed.
"""

import os
import sys
import warnings
import numpy as np

# --------------------------- CONFIG ---------------------------
MAT_PATH      = "Propeller_fault_data.mat"
OUT_DIR       = "eda_out"
FS_HZ         = None    # set sampling rate (Hz) if known; else inferred
TIME_FIELD    = None    # name (or path suffix) of a time vector, if known
MAX_PLOT_RUNS = 9
MAX_SIGNALS   = 12      # cap signals plotted per group
# --------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAVE_PD = True
except Exception:
    HAVE_PD = False

_LINES = []
def log(msg=""):
    print(msg)
    _LINES.append(str(msg))


# --------------------------- loading ---------------------------
def load_mat(path):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    if os.path.getsize(path) < 100_000 and b"git-lfs" in open(path, "rb").read(80):
        sys.exit("This is a Git LFS pointer, not the data. Run `git lfs pull`.")
    from pymatreader import read_mat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = read_mat(path)
    return {k: v for k, v in data.items() if not k.startswith("__")}


# --------------------------- structure helpers ---------------------------
def describe_tree(obj, name, depth=0, max_depth=4):
    """Print a nested view of a dict/array so we can see the real layout."""
    pad = "  " * (depth + 1)
    if isinstance(obj, dict):
        log(f"{pad}{name}/  (dict, {len(obj)} keys: {list(obj)[:12]})")
        if depth < max_depth:
            for k, v in obj.items():
                describe_tree(v, k, depth + 1, max_depth)
        return
    a = np.asarray(obj)
    if a.dtype == object:
        log(f"{pad}{name}: object-array shape={a.shape}")
    elif np.issubdtype(a.dtype, np.number):
        a2 = np.squeeze(a)
        if a2.ndim == 0:
            log(f"{pad}{name} = {float(a2):.5g}  (scalar)")
        else:
            log(f"{pad}{name}: shape={a2.shape} mean={np.nanmean(a2):.4g} "
                f"std={np.nanstd(a2):.4g} min={np.nanmin(a2):.4g} max={np.nanmax(a2):.4g}")
    else:
        log(f"{pad}{name}: dtype={a.dtype} shape={a.shape}")


def flatten_signals(obj, prefix=""):
    """Recurse through nested dicts; return {dotted_path: 1d float array}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_signals(v, f"{prefix}{k}."))
    else:
        a = np.asarray(obj)
        if np.issubdtype(a.dtype, np.number) and a.ndim >= 1 and a.size >= 8:
            out[prefix.rstrip(".")] = np.squeeze(a).astype(float)
    return out


def list_runs(group):
    for k, v in group.items():
        if k.lower() == "omega":
            continue
        yield k, v


def infer_fs(sig):
    if FS_HZ:
        return FS_HZ, "config"
    cands = [k for k in sig if k.split(".")[-1].lower() in ("t", "time", "time_s", "tvec", "tt", "sec")]
    if TIME_FIELD:
        cands = [k for k in sig if k.endswith(TIME_FIELD)] or cands
    for name in cands:
        t = sig[name]
        if t.size > 2:
            d = np.diff(t)
            if np.all(d > 0) and np.std(d) / (abs(np.median(d)) + 1e-12) < 0.05:
                return 1.0 / np.median(d), f"time field '{name}'"
    for name, v in sig.items():
        if v.size > 50:
            d = np.diff(v)
            if np.all(d > 0) and np.std(d) / (abs(np.median(d)) + 1e-12) < 0.02:
                return 1.0 / np.median(d), f"inferred from '{name}'"
    return None, "unknown"


# --------------------------- main ---------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_mat(MAT_PATH)

    log("=" * 70); log("TOP-LEVEL STRUCTURE"); log("=" * 70)
    groups = {}
    for k, v in data.items():
        n = len(v) if isinstance(v, dict) else np.asarray(v).size
        log(f"  {k:12s} type={type(v).__name__:8s} entries/size={n}")
        if isinstance(v, dict) and any(kk.lower() != "omega" for kk in v):
            groups[k] = v

    # ---- omega, as a tree (it is a nested struct, not a vector) ----
    log("\n" + "=" * 70); log("OMEGA (nested structure)"); log("=" * 70)
    for gname, g in groups.items():
        key = next((kk for kk in g if kk.lower() == "omega"), None)
        if key is None:
            continue
        log(f"\n  [{gname}.omega]")
        om = g[key]
        if isinstance(om, dict):
            describe_tree(om, "omega", depth=1)
        else:
            a = np.squeeze(np.asarray(om))
            if np.issubdtype(a.dtype, np.number):
                log(f"     {a.size} values: {np.round(a, 3)}")
            else:
                log(f"     non-numeric: dtype={a.dtype} shape={a.shape}")

    # ---- full tree of the first run of each group (reveals leaf signals) ----
    log("\n" + "=" * 70); log("STRUCTURE OF FIRST RUN (paste this back)"); log("=" * 70)
    for gname, g in groups.items():
        runs = list(list_runs(g))
        if not runs:
            continue
        rname, run = runs[0]
        log(f"\n  [{gname}] first run = {rname}")
        describe_tree(run, rname, depth=1)

    # ---- inventory over all runs using flattened signal paths ----
    log("\n" + "=" * 70); log("SIGNAL INVENTORY (flattened paths)"); log("=" * 70)
    inventory, field_union, fs_report = [], {}, {}
    for gname, g in groups.items():
        runs = list(list_runs(g))
        s0 = flatten_signals(runs[0][1]) if runs else {}
        field_union[gname] = sorted(s0.keys())
        fs, how = infer_fs(s0)
        fs_report[gname] = (fs, how)
        log(f"\n  [{gname}] {len(runs)} runs; {len(field_union[gname])} signal paths")
        log(f"     paths: {field_union[gname]}")
        log(f"     sampling rate: {('%.3f Hz' % fs) if fs else 'UNKNOWN'} ({how})")
        for rname, run in runs:
            for path, arr in flatten_signals(run).items():
                inventory.append((gname, rname, path, arr.size,
                                  float(np.nanmean(arr)), float(np.nanstd(arr)),
                                  float(np.nanmin(arr)), float(np.nanmax(arr)),
                                  int(np.isnan(arr).sum())))

    if HAVE_PD and inventory:
        df = pd.DataFrame(inventory, columns=["group", "run", "path", "n",
                                              "mean", "std", "min", "max", "n_nan"])
        p = os.path.join(OUT_DIR, "signal_inventory.csv")
        df.to_csv(p, index=False)
        log(f"\n  {len(df)} rows -> {p}")

    # ---- mode inference: within-run CV per signal path ----
    log("\n" + "=" * 70)
    log("MODE INFERENCE (within-run coefficient of variation per path)")
    log("  Flat (low CV) in one group but variable in the other = controlled variable.")
    log("=" * 70)
    for gname, g in groups.items():
        agg = {}
        for _, run in list_runs(g):
            for path, arr in flatten_signals(run).items():
                cv = np.nanstd(arr) / (abs(np.nanmean(arr)) + 1e-12)
                agg.setdefault(path, []).append(cv)
        log(f"\n  [{gname}]")
        for path in field_union.get(gname, sorted(agg)):
            if path in agg:
                log(f"     {path:24s} CV={np.mean(agg[path]):.4f}")

    # ---- plots: each signal path, runs overlaid ----
    for gname, g in groups.items():
        try:
            paths = field_union.get(gname, [])[:MAX_SIGNALS]
            if not paths:
                continue
            ncol = min(3, len(paths)); nrow = int(np.ceil(len(paths) / ncol))
            fig, ax = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow), squeeze=False)
            for i, path in enumerate(paths):
                a = ax[i // ncol][i % ncol]
                for rname, run in list(list_runs(g))[:MAX_PLOT_RUNS]:
                    s = flatten_signals(run)
                    if path in s:
                        a.plot(s[path], lw=0.7, alpha=0.7, label=rname)
                a.set_title(f"{gname}: {path}", fontsize=8); a.set_xlabel("sample")
            if len(paths) <= 6:
                ax[0][0].legend(fontsize=6, ncol=2)
            for j in range(len(paths), nrow * ncol):
                ax[j // ncol][j % ncol].axis("off")
            fig.tight_layout()
            p = os.path.join(OUT_DIR, f"timeseries_{gname}.png")
            fig.savefig(p, dpi=110); plt.close(fig)
            log(f"  figure -> {p}")
        except Exception as e:
            log(f"  [plot warning {gname}] {e}")

    # ---- spectral first look on the most-varying signal of one run ----
    log("\n" + "=" * 70); log("SPECTRAL FIRST LOOK"); log("=" * 70)
    for gname, g in groups.items():
        try:
            fs, _ = fs_report[gname]
            rname, run = list(list_runs(g))[0]
            s = flatten_signals(run)
            if not s:
                continue
            path = max(s, key=lambda k: s[k].size * (np.nanstd(s[k]) + 1e-9))
            x = s[path]; x = x[np.isfinite(x)]
            seg = x[len(x)//4: 3*len(x)//4]; seg = seg - np.mean(seg)
            d = 1.0/fs if fs else 1.0
            freqs = np.fft.rfftfreq(seg.size, d=d); psd = np.abs(np.fft.rfft(seg))**2
            unit = "Hz" if fs else "cycles/sample"
            peaks = freqs[np.argsort(psd)[-6:][::-1]]
            log(f"\n  [{gname}] {rname} '{path}' top peaks ({unit}): {np.round(peaks,4)}")
            fig, a = plt.subplots(figsize=(7, 4))
            a.semilogy(freqs[1:], psd[1:] + 1e-9, lw=0.8)
            a.set_title(f"{gname} {rname} '{path}' PSD"); a.set_xlabel(f"freq ({unit})")
            p = os.path.join(OUT_DIR, f"psd_{gname}.png"); fig.savefig(p, dpi=110); plt.close(fig)
            log(f"  figure -> {p}")
        except Exception as e:
            log(f"  [spectrum warning {gname}] {e}")

    log("\n" + "=" * 70); log("WHAT TO TELL CLAUDE (paste back)"); log("=" * 70)
    log("  1. The 'STRUCTURE OF FIRST RUN' tree (real leaf signal names).")
    log("  2. The 'OMEGA' tree.")
    log("  3. The inferred sampling rate (correct it if you know the true fs).")
    log("  4. The 'MODE INFERENCE' CV tables.")
    log("  5. Anything in README.md defining CV/CCV and the run index.")

    with open(os.path.join(OUT_DIR, "eda_summary.txt"), "w") as f:
        f.write("\n".join(_LINES))
    log(f"\n  Full summary -> {os.path.join(OUT_DIR, 'eda_summary.txt')}")


if __name__ == "__main__":
    main()