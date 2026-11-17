# Thruster FDD & FTC — Handoff Doc

Context for continuing the project (feed to a new chat or a local coding model like Qwen Coder).
Workflow: Claude for planning/methodology/review; local model for code iteration.

## Project
Fault detection & fault-tolerant control of worn ROV thrusters, with Dr. Simon Pedersen's group
(AAU Energy, Esbjerg). Goal: a co-authored paper toward a Danish Industrial PhD (deadline late Feb).
Constraints: MacBook, no NVIDIA GPU, no MATLAB. Python only. Local models via Ollama + Qwen Coder.

Three-stage pipeline:
1. Detection — identify worn vs new thrusters from 3-phase motor current. WORKING.
2. Estimation — quantify wear (drag/efficiency) with a Kalman filter. BLOCKED (needs force data).
3. Fault-tolerant control — adaptive thrust allocation on BlueROV2 Heavy, derate-to-weakest. STARTED.

## Dataset (Propeller_fault_data.mat, from von Benzon et al., OCEANS 2024)
- Real offshore T200 propellers, bench rig, steady-state per operating point (no transient dynamics).
- Two groups: CV = clockwise, CCV = counter-clockwise.
- Each propeller (run) has 9 PWM operating points in110..in190 (= 1100..1900 us).
- Each point: MeasCurrA/B/C, MeasVoltA/B/C, TriggerCurr/Volt, time. 125k samples, ~312.5 kHz.
- NO force field in the published file (confirmed by tree-wide search). Load-cell force (Table I) is
  in a separate file, requested from AAU.
- Labels (paper Fig.3/4, Table III): new/healthy = CV{0,10,11}, CCV{0,6,7}; rest are worn (NOT in
  severity order). Severity target = Performance Parameter PP = thrust/power (1.0=new, lower=worse).
- TO CONFIRM with Malte: CV<n> <-> CW<n> index mapping (labels depend on it); where force lives.

## What the paper found (and didn't)
FFT + 5th-harmonic sidebands + rotational-speed variance + PP. It is descriptive: showed wear leaves
electrical signatures but found NO clean relationship between harmonic cue and severity, and no
online/real-time method, estimator, or control. That gap is our runway.

## Progress
- Stage 1 detection WORKS: current features (RMS, kurtosis, phase imbalance, harmonics) -> RandomForest,
  leave-one-propeller-out (no propeller in both train/test). New-vs-worn AUC ~0.81 pooled, up to 0.99
  per operating point (best ~1100/1400 us). Signature is NON-LINEAR (logistic regression fails) and
  DISTRIBUTED across features. Rotational-speed variance did NOT correlate with severity as computed.
- Severity estimation is weak (PP R2~0.08) — expected, needs force. Reproduces the paper's difficulty.
- Stage 3 STARTED: BlueROV2 Heavy 8-thruster allocation matrix + PID -> desired wrench -> two
  allocators (baseline pseudo-inverse vs FTC derate-to-weakest), fault injected, tracking-error plot.
  NOTE: currently runs a SIMPLE fallback 6-DOF integrator, NOT the real simulator.

## Key domain-correctness notes (keep these decisions with Claude, not the local model)
- Detection must use current/voltage only (force is offline, validation-only).
- KF for Stage 2 = PARAMETER estimation (estimate a constant drag/thrust coefficient from steady-state
  data), NOT a dynamic-residual observer (no transients in this data). Model e.g. T = kT * n^2; estimate
  kT per propeller; lower kT = more worn; validate kT vs PP.
- Severity labels are only ~20 (one PP per propeller). Per-capture force would give ~207 labels.
- Stage 3 allocation matrix geometry is currently representative — swap in AAU's exact BlueROV2 Heavy
  values. Derate-to-weakest = scale all thruster commands to the weakest healthy capacity.

## Open code tasks (specs for the local model)
1. Wire the REAL simulator into Stage 3. bluerov2_gym is a GitHub repo using uv (not pip):
   `git clone`, `uv sync`, run inside it with `uv run python`, or `uv pip install -e .` into the env.
   Confirm the registered env id (script currently assumes gym.make("BlueRov-v0")) and whether a Heavy
   8-thruster config exists; if not, keep the custom allocation matrix on top of its dynamics.
2. Finalize Stage 1: feature-importance confirmation, per-input models, clean report/plots.
3. Stage 2 (when force arrives): augmented-state / parameter KF for kT; efficiency = force/power regressed
   from current-only features; validate vs PP.
4. Contingency if no force: relative unitless wear index from current+speed as the degradation factor
   feeding Stage 3; demonstrate FTC in sim. Stage 1 + Stage 3 then stand as a complete result.

## Files so far (Python, conda env rov-fdd)
- eda.py — dataset structure explorer
- features.py / kalman_harmonics.py — Stage 1 feature extraction (FFT + KF harmonic estimator)
- stage1.py / stage1b.py / stage1_final.py — detection classifier + severity regression, LOPO
- stage2.py — KF thrust-coefficient estimator (blocked on force; also a force locator)
- stage3.py — BlueROV2 Heavy allocation + derate-to-weakest FTC (fallback dynamics)

## Environment gotchas
- Keep project in ~/code (NOT Desktop/Documents/iCloud) to avoid macOS file-permission errors.
- The .mat is a Git LFS file (~310 MB); use `git lfs pull` to get the real data, not the pointer.
- conda env rov-fdd: numpy, scipy, pandas, scikit-learn, matplotlib, pymatreader.
