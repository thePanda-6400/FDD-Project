#!/usr/bin/env python3
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- config ----------------
CSV       = "merged.csv"
D         = 0.076      # T200 propeller diameter [m]   -- VERIFY with Malte
RHO       = 998.0      # fresh water lab tank [kg/m^3] -- VERIFY with Malte
ETA_MOTOR = 1.0        # unknown; keeps K_Q on a relative scale
NEW       = {"CV": {0, 10, 11}, "CCV": {0, 6, 7}}   # paper's baselines -- VERIFY
MIN_SPEED = 2.0        # rev/s
NEUTRAL   = 1500       # us, motor stopped
# ----------------------------------------


def kf_constant(y, q=1e-8):
    """1-state Kalman filter for a constant parameter -> (estimate, variance)."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.nan, np.nan
    x, P = y[0], np.var(y) + 1e-12
    R = np.var(y) + 1e-12
    for z in y[1:]:
        P += q                       # predict: parameter is constant
        K = P / (P + R)              # update
        x += K * (z - x)
        P *= (1 - K)
    return float(x), float(P)


def main():
    df = pd.read_csv(CSV)
    df["power_W"] = df.power_W.abs()      # probe polarity inverted; magnitude ok
    df["is_new"] = [p in NEW.get(d, set()) for d, p in zip(df.direction, df.propeller)]

    # ---------- 1. per-point coefficients ----------
    d = df[(df.throttle != NEUTRAL) &
           (df.speed_rps.abs() >= MIN_SPEED) &
           (df.power_W > 0)].copy()
    n = d.speed_rps.abs()
    d["n"]      = n
    d["K_T"]    = d.thrust_N.abs() / (RHO * n**2 * D**4)
    d["Q_est"]  = ETA_MOTOR * d.power_W / (2 * np.pi * n)     # = 9549*P[kW]/RPM
    d["K_Q"]    = d.Q_est / (RHO * n**2 * D**5)
    d["regime"] = np.where(d.throttle < NEUTRAL, "reverse", "forward")
    d["uid"]    = d.direction + d.propeller.astype(str)
    print(f"points used: {len(d)} / {len(df)}")

    # ---------- 2. KF coefficients per propeller ----------
    rows = []
    for (dirn, prop, reg), s in d.groupby(["direction", "propeller", "regime"]):
        kt, ktv = kf_constant(s.K_T.values)
        kq, kqv = kf_constant(s.K_Q.values)
        rows.append({"direction": dirn, "propeller": prop, "regime": reg,
                     "uid": f"{dirn}{prop}", "is_new": bool(s.is_new.iloc[0]),
                     "n_points": len(s),
                     "K_T": kt, "K_T_sd": np.sqrt(ktv),
                     "K_Q": kq, "K_Q_sd": np.sqrt(kqv),
                     "KT_over_KQ": kt / kq if kq else np.nan})
    C = pd.DataFrame(rows)

    # K_T relative to the mean of the new propellers, within direction+regime
    C["K_T_rel"] = np.nan
    for (dirn, reg), s in C.groupby(["direction", "regime"]):
        base = s.loc[s.is_new, "K_T"].mean()
        if np.isfinite(base) and base > 0:
            C.loc[s.index, "K_T_rel"] = s.K_T / base
    C.to_csv("stage2_coeffs.csv", index=False)

    print("\nK_T by propeller (forward), relative to new baseline:")
    print(C[C.regime == "forward"].sort_values(["direction", "propeller"])
          [["uid", "is_new", "K_T", "K_T_sd", "K_T_rel", "KT_over_KQ"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for (dirn, reg), s in C.groupby(["direction", "regime"]):
        if s.is_new.nunique() == 2:
            a, b = s[s.is_new].K_T.mean(), s[~s.is_new].K_T.mean()
            print(f"  {dirn} {reg}: K_T new={a:.4f} worn={b:.4f} drop={100*(1-b/a):.1f}%")

    # ---------- 3. KF-reconstructed thrust (consistency check only) ----------
    key = C.set_index(["uid", "regime"])
    d["K_T_kf"] = [key.K_T.get((u, r), np.nan) for u, r in zip(d.uid, d.regime)]
    d["K_Q_kf"] = [key.K_Q.get((u, r), np.nan) for u, r in zip(d.uid, d.regime)]
    d["T_kf"]   = d.K_T_kf * RHO * d.n**2 * D**4
    d["Q_kf"]   = d.K_Q_kf * RHO * d.n**2 * D**5
    print(f"\n[3] KF thrust reconstruction (consistency, not independent): "
          f"R2={r2_score(d.thrust_N.abs(), d.T_kf):.3f}  "
          f"MAE={mean_absolute_error(d.thrust_N.abs(), d.T_kf):.2f} N")

    # ---------- 4. ELECTRICAL-ONLY thrust estimation (LOPO) ----------
    # regressors computable WITHOUT a load cell:
    #   x1 = rho*n^2*D^4 (thrust scale)   x2 = Q_est/D (torque-derived proxy)
    d["x1"] = RHO * d.n**2 * D**4
    d["x2"] = d.Q_est / D
    X = d[["x1", "x2"]].values
    y = d.thrust_N.abs().values
    groups = d.uid.values

    pred = np.full(len(d), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    d["T_pred_elec"] = pred

    # naive baseline: assume every propeller is healthy (mean new K_T)
    base_kt = C[C.is_new].groupby("regime").K_T.mean()
    d["T_naive"] = [base_kt.get(r, np.nan) * RHO * nn**2 * D**4
                    for r, nn in zip(d.regime, d.n)]

    print(f"[4] electrical-only thrust (LOPO): "
          f"R2={r2_score(y, pred):.3f}  MAE={mean_absolute_error(y, pred):.2f} N")
    print(f"    naive 'all healthy' baseline:  "
          f"R2={r2_score(y, d.T_naive):.3f}  MAE={mean_absolute_error(y, d.T_naive):.2f} N")

    # the comparison that matters: error on the WORN propellers only,
    # where the healthy-assumption baseline should fail hardest
    w = ~d.is_new.values
    if w.sum() > 5:
        print(f"    worn propellers only -> estimator MAE={mean_absolute_error(y[w], pred[w]):.2f} N"
              f"   naive MAE={mean_absolute_error(y[w], d.T_naive.values[w]):.2f} N")

    # ---------- 5. degradation index ----------
    fwd = C[C.regime == "forward"].copy()
    fwd["degradation"] = 1 - fwd.K_T_rel          # 0 = healthy, higher = more worn
    print("\n[5] degradation index (forward), higher = more worn:")
    print(fwd.sort_values("degradation", ascending=False)
          [["uid", "is_new", "K_T", "K_T_rel", "degradation", "KT_over_KQ"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fwd.to_csv("stage2_degradation.csv", index=False)
    d.to_csv("stage2_points.csv", index=False)

    # ---------- figures ----------
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, reg in enumerate(["forward", "reverse"]):
        s = C[C.regime == reg].sort_values(["direction", "propeller"])
        ax[i].bar(range(len(s)), s.K_T, yerr=s.K_T_sd, capsize=2,
                  color=["tab:blue" if v else "tab:red" for v in s.is_new])
        ax[i].set_xticks(range(len(s)))
        ax[i].set_xticklabels(s.uid, rotation=90, fontsize=7)
        ax[i].set_ylabel("$K_T$")
        ax[i].set_title(f"Thrust coefficient, {reg}")
        ax[i].grid(alpha=.3, axis="y")
    f.tight_layout(); f.savefig("fig_KT_per_propeller.png", dpi=130); plt.close(f)

    # headline: estimated vs measured thrust
    f, a = plt.subplots(figsize=(5.2, 5))
    a.scatter(y, pred, s=14, alpha=.6,
              c=["tab:blue" if v else "tab:red" for v in d.is_new])
    lim = [0, max(np.nanmax(y), np.nanmax(pred)) * 1.05]
    a.plot(lim, lim, "k--", lw=.8); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("measured thrust [N]")
    a.set_ylabel("estimated thrust [N], electrical only")
    a.set_title(f"Electrical-only thrust estimate (LOPO)\n$R^2$={r2_score(y, pred):.3f}, "
                f"MAE={mean_absolute_error(y, pred):.2f} N")
    a.grid(alpha=.3); f.tight_layout()
    f.savefig("fig_thrust_pred_vs_meas.png", dpi=130); plt.close(f)

    # K_T vs K_Q, and the K_T/K_Q separation
    f, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    s = C[C.regime == "forward"]
    ax[0].scatter(s.K_Q, s.K_T, s=35,
                  c=["tab:blue" if v else "tab:red" for v in s.is_new])
    for r in s.itertuples():
        ax[0].annotate(r.uid, (r.K_Q, r.K_T), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("$K_Q$ (relative scale)"); ax[0].set_ylabel("$K_T$")
    ax[0].set_title("$K_T$ vs $K_Q$"); ax[0].grid(alpha=.3)

    srt = s.sort_values("KT_over_KQ")
    ax[1].barh(range(len(srt)), srt.KT_over_KQ,
               color=["tab:blue" if v else "tab:red" for v in srt.is_new])
    ax[1].set_yticks(range(len(srt))); ax[1].set_yticklabels(srt.uid, fontsize=7)
    ax[1].set_xlabel("$K_T/K_Q$  (thrust per unit torque)")
    ax[1].set_title("Thrust-per-torque ratio"); ax[1].grid(alpha=.3, axis="x")
    f.tight_layout(); f.savefig("fig_KT_vs_KQ.png", dpi=130); plt.close(f)

    print("\n-> stage2_coeffs.csv, stage2_points.csv, stage2_degradation.csv")
    print("   fig_KT_per_propeller.png, fig_thrust_pred_vs_meas.png, fig_KT_vs_KQ.png")


if __name__ == "__main__":
    main()