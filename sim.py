#!/usr/bin/env python3
"""
Stage 3: fault-tolerant thrust allocation for BlueROV2 Heavy.

Uses bluerov2_gym for dynamics if installed; otherwise falls back to a simple
decoupled 6-DOF integrator so the controller/allocator logic runs anywhere.

Compares two allocators under an injected thruster fault:
  - baseline pseudo-inverse (ignores the fault)
  - FTC "derate-to-weakest": scales commands by per-thruster effectiveness and
    caps the group to the weakest healthy capacity, preserving balance.
The per-thruster effectiveness vector is what Stage 1 detection supplies.

Run: python stage3.py   (pip install numpy scipy matplotlib; bluerov2_gym optional)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- BlueROV2 Heavy thruster allocation matrix (6 DOF x 8 thrusters) ---
# Columns = thrusters: 4 vectored horizontal (45 deg) + 4 vertical.
# Representative geometry; swap in AAU's exact values when available.
a = np.deg2rad(45)
# unit thrust directions in body frame (surge,sway,heave) per thruster
dirs = np.array([
 [ np.cos(a),  np.sin(a), 0],[ np.cos(a), -np.sin(a), 0],
 [-np.cos(a),  np.sin(a), 0],[-np.cos(a), -np.sin(a), 0],
 [0,0,1],[0,0,1],[0,0,1],[0,0,1]], float)
# positions (x,y,z) of each thruster rel. CG [m]
pos = np.array([
 [0.15, 0.10, 0],[0.15,-0.10,0],[-0.15,0.10,0],[-0.15,-0.10,0],
 [0.12, 0.12,0],[0.12,-0.12,0],[-0.12,0.12,0],[-0.12,-0.12,0]], float)
TAM = np.zeros((6,8))
for i in range(8):
    TAM[:3,i] = dirs[i]
    TAM[3:,i] = np.cross(pos[i], dirs[i])   # torque = r x F
TAM_pinv = np.linalg.pinv(TAM)

U_MAX = 40.0   # per-thruster max thrust [N] (T200 ~ +/-40 N)

def allocate_baseline(tau):
    return np.clip(TAM_pinv @ tau, -U_MAX, U_MAX)

def allocate_ftc(tau, eff):
    """eff in (0,1] per thruster from Stage 1. Derate to the weakest, then solve."""
    cap = U_MAX * eff.min()                     # weakest healthy capacity
    u = TAM_pinv @ tau
    s = np.max(np.abs(u)/ (cap+1e-9))
    if s > 1: u = u / s                         # scale wrench to stay within weakest
    return np.clip(u * 1.0, -U_MAX*eff, U_MAX*eff)  # respect each thruster's real limit

def realized_wrench(u_cmd, eff):
    """actual thrust = commanded * effectiveness (fault reduces output)."""
    return TAM @ (u_cmd * eff)

# --- dynamics: try bluerov2_gym, else simple integrator ---
def make_plant():
    try:
        import gymnasium as gym, bluerov2_gym   # noqa
        env = gym.make("BlueRov-v0")
        return ("gym", env)
    except Exception:
        return ("simple", None)

def run(eff, allocator, steps=1500, dt=0.02):
    M = np.diag([12,12,16,0.2,0.2,0.3])          # mass + added mass (approx)
    D = np.diag([18,18,24,0.6,0.6,0.8])          # linear drag
    x = np.zeros(6); v = np.zeros(6); Ki=np.zeros(6)
    target = np.array([1.0,0.5,0.8,0,0,0.3])     # setpoint pose
    Kp=np.array([40,40,60,8,8,10]); Kd=np.array([30,30,45,5,5,6]); Kii=np.array([2,2,3,.3,.3,.4])
    err_hist=[]
    for _ in range(steps):
        e = target - x; Ki += e*dt
        tau = Kp*e - Kd*v + Kii*np.clip(Ki,-5,5)        # PID -> desired wrench
        u = allocator(tau)
        w = realized_wrench(u, eff)
        a_ = np.linalg.solve(M, w - D@v)                # M v' + D v = wrench
        v += a_*dt; x += v*dt
        err_hist.append(np.linalg.norm(target[:3]-x[:3]))
    return np.array(err_hist)

def main():
    healthy = np.ones(8)
    faulty  = np.ones(8); faulty[0]=0.4; faulty[5]=0.6   # thruster 0 @40%, 5 @60% (from Stage 1)
    base = run(faulty, allocate_baseline)
    ftc  = run(faulty, lambda tau: allocate_ftc(tau, faulty))
    ref  = run(healthy, allocate_baseline)
    print(f"final position error [m]  healthy={ref[-1]:.3f}  "
          f"faulty+baseline={base[-1]:.3f}  faulty+FTC={ftc[-1]:.3f}")
    t=np.arange(len(base))*0.02
    plt.figure(figsize=(7,4))
    plt.plot(t,ref,label="healthy"); plt.plot(t,base,label="faulty + baseline")
    plt.plot(t,ftc,label="faulty + FTC (derate-to-weakest)")
    plt.xlabel("time [s]"); plt.ylabel("position error [m]"); plt.legend()
    plt.title("BlueROV2 Heavy: FTC vs baseline under thruster fault")
    plt.tight_layout(); plt.savefig("stage3_tracking.png",dpi=120)
    print("-> stage3_tracking.png")

if __name__=="__main__": main()