# Thruster Propeller Wear Estimation and Fault-Tolerant Allocation

Analysis and control code for estimating propeller wear in underwater
thrusters from motor-current measurements, and using that estimate to drive
fault-tolerant thrust allocation.

Work in collaboration with AAU Energy, Aalborg University (Esbjerg).

## Data

The dataset is **not** included in this repository. It is published by the
AAU Subsea Research Lab:

    https://github.com/Subsea-Research-Lab-AAU/Incipient-Propeller-Fault-Data

To reproduce, clone that repository (it uses Git LFS) and place
`Propeller_fault_data.mat` and the force-measurement folder alongside these
scripts.

    git lfs install
    git clone https://github.com/Subsea-Research-Lab-AAU/Incipient-Propeller-Fault-Data.git
    git lfs pull

## Environment

    conda env create -f environment.yml
    conda activate rov-fdd

## Scripts

| File | Purpose |
|---|---|
| `eda.py` | Explores the dataset structure (groups, runs, signals, sampling rate) |
| `merge.py` | Joins the force recordings to the electrical records by throttle level |
| `forcecheck.py` | Validation checks on the merge, plus figures reproducing the source paper |
| `Stage1.py`, `stage1_final.py` | Current-signature detection of worn propellers, leave-one-propeller-out |
| `stage2.py` | Propeller coefficients, Kalman parameter estimation, degradation index |
| `sim.py` | Thrust allocation and fault-tolerant control experiments |

## Pipeline

1. Detect that a propeller is worn, from three-phase motor current.
2. Quantify the wear as a physical degradation parameter.
3. Feed that parameter to a thrust allocator as a per-thruster effectiveness factor.
