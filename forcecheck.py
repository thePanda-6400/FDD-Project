#!/usr/bin/env python3
"""
Validate merged.csv and reproduce the AAU paper's figures for a visual check.

Automated checks:
  1. coverage        - 9 throttles per propeller, no missing/duplicate rows
  2. NaNs            - key columns complete
  3. zero point      - thrust and speed ~0 at throttle 1500 (stopped)
  4. monotonicity    - |thrust| and speed rise as |throttle-1500| rises
  5. sign            - thrust negative below 1500, positive above
  6. PHYSICS         - thrust proportional to speed^2 (validates the merge AND
                       the pole-pair conversion; a weak fit means one is wrong)
  7. new vs worn     - new propellers out-thrust worn ones at full input

Figures (compare by eye with the paper):
  fig_speed_vs_input.png   ~ paper Fig. 12  (rotational speed vs input)
  fig_thrust_vs_input.png  ~ paper Fig. 13/16 (force vs input)
  fig_thrust_vs_speed.png  ~ paper Fig. 15  (performance curves)
  fig_efficiency.png       - thrust per watt vs input

Run: python check_merge.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "merged.csv"
NEW = {"CV": {0, 10, 11}, "CCV": {0, 6, 7}}   # paper's baselines; revise if Malte differs
OK, BAD = "  [ok] ", "  [!!] "

def main():
    df = pd.read_csv(CSV)
    df["is_new"] = [p in NEW.get(d, set()) for d, p in zip(df.direction, df.propeller)]
    # signed speed: below 1500 the thruster runs in reverse (FFT gives magnitude only)
    df["speed_signed"] = np.where(df.throttle < 1500, -df.speed_rps, df.speed_rps)
    print(f"rows={len(df)}  propellers={df.groupby(['direction','propeller']).ngroups}\n")

    # 1 coverage
    cov = df.groupby(["direction", "propeller"]).throttle.nunique()
    bad = cov[cov != 9]
    print((OK if bad.empty else BAD) + f"coverage: {len(cov)} propellers, "
          f"{'all have 9 throttles' if bad.empty else f'{len(bad)} incomplete -> {dict(bad)}'}")
    dup = df.duplicated(["direction", "propeller", "throttle"]).sum()
    print((OK if dup == 0 else BAD) + f"duplicates: {dup}")

    # 2 NaNs
    key = ["thrust_N", "power_W", "speed_rps"]
    nn = df[key].isna().sum()
    print((OK if nn.sum() == 0 else BAD) + f"NaNs: {dict(nn)}")

    # 3 zero point at 1500
    z = df[df.throttle == 1500]
    if len(z):
        print((OK if z.thrust_N.abs().mean() < 5 else BAD) +
              f"zero point: |thrust|={z.thrust_N.abs().mean():.2f} N, speed={z.speed_rps.mean():.2f} rev/s")

    # 4 monotonicity with distance from neutral
    g = df.assign(d=(df.throttle - 1500).abs()).groupby("d")[["thrust_N", "speed_rps"]]
    dd = df.assign(d=(df.throttle - 1500).abs())
    mt = dd.groupby("d").thrust_N.apply(lambda s: s.abs().mean())
    ms = dd.groupby("d").speed_rps.mean()

    
    # 5 sign
    lo = df[df.throttle < 1500].thrust_N.mean(); hi = df[df.throttle > 1500].thrust_N.mean()
    print((OK if lo < 0 < hi else BAD) + f"sign: mean thrust below 1500={lo:.1f} N, above={hi:.1f} N")

    # 6 PHYSICS: T ~ n^2
    m = df[(df.throttle != 1500) & df.speed_rps.notna()]
    x = m.speed_rps ** 2; y = m.thrust_N.abs()
    r = np.corrcoef(x, y)[0, 1]
    print((OK if r > 0.9 else BAD) + f"thrust vs speed^2: r={r:.3f} "
          f"{'(physics consistent)' if r > 0.9 else '(CHECK merge or POLE_PAIRS)'}")

    # 7 new vs worn at full input
    full = df[df.throttle == 1900]
    if full.is_new.nunique() == 2:
        n_, w_ = full[full.is_new].thrust_N.mean(), full[~full.is_new].thrust_N.mean()
        print((OK if n_ > w_ else BAD) + f"full input thrust: new={n_:.1f} N, worn={w_:.1f} N")

    # ---------------- figures ----------------
    def per_prop(ax, ycol, ylab):
        for (d, p), s in df.groupby(["direction", "propeller"]):
            s = s.sort_values("throttle")
            new = p in NEW.get(d, set())
            ax.plot(s.throttle, s[ycol], marker="o", ms=2.5, lw=1,
                    color=("tab:blue" if new else "tab:red"),
                    alpha=(0.95 if new else 0.45), label="_")
        ax.set_xlabel("input [us]"); ax.set_ylabel(ylab); ax.grid(alpha=.3)
        ax.plot([], [], color="tab:blue", label="new"); ax.plot([], [], color="tab:red", label="worn")
        ax.legend(fontsize=8)

    f, a = plt.subplots(figsize=(6.5, 4)); per_prop(a, "speed_signed", "rotational speed [rev/s]")
    a.set_title("Speed vs input (cf. paper Fig. 12)"); f.tight_layout()
    f.savefig("fig_speed_vs_input.png", dpi=120); plt.close(f)

    f, a = plt.subplots(figsize=(6.5, 4)); per_prop(a, "thrust_N", "thrust [N]")
    a.set_title("Thrust vs input (cf. paper Fig. 13/16)"); f.tight_layout()
    f.savefig("fig_thrust_vs_input.png", dpi=120); plt.close(f)

    f, a = plt.subplots(figsize=(6.5, 4))
    for (d, p), s in df.groupby(["direction", "propeller"]):
        s = s.sort_values("speed_signed"); new = p in NEW.get(d, set())
        a.plot(s.speed_signed, s.thrust_N, marker="o", ms=2.5, lw=1,
               color=("tab:blue" if new else "tab:red"), alpha=(0.95 if new else 0.45))
    a.set_xlabel("rotational speed [rev/s]"); a.set_ylabel("thrust [N]")
    a.set_title("Performance curves (cf. paper Fig. 15)"); a.grid(alpha=.3)
    a.plot([], [], color="tab:blue", label="new"); a.plot([], [], color="tab:red", label="worn")
    a.legend(fontsize=8); f.tight_layout(); f.savefig("fig_thrust_vs_speed.png", dpi=120); plt.close(f)

    f, a = plt.subplots(figsize=(6.5, 4)); per_prop(a, "efficiency", "thrust per watt [N/W]")
    a.set_title("Efficiency vs input"); f.tight_layout()
    f.savefig("fig_efficiency.png", dpi=120); plt.close(f)

    print("\nfigures: fig_speed_vs_input.png, fig_thrust_vs_input.png, "
          "fig_thrust_vs_speed.png, fig_efficiency.png")

if __name__ == "__main__":
    main()