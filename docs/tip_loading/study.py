"""
Parametric study of the sectional lift near the tip of a rectangular wing with tip shedding.

Compares the original formulation (docs/tip_loading/VLMSolver_original.py, solver as of commit 2a30236)
with the revised formulation (src/VLMSolver.py). Rectangular wing, AR = 1, alpha = 10 deg, M = 5 chordwise
panels, cosine spanwise spacing unless stated, simulation length u_inf*T/c = 3.

Run from the repository root:  python docs/tip_loading/study.py
Writes docs/tip_loading/study.json and docs/tip_loading/tsweep.json.
"""
import sys, os, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "docs", "tip_loading"))
from VLMSurface import VLMSurface
from VLMSolver import VLMSolver as Revised
from VLMSolver_original import VLMSolver as Original

CORE = 0.05     # revised: vortex core radius / chord
CUT  = 0.4      # original: cutoff / minimum panel width (also a_ratio for both)

def run(solver, N=15, M=5, T=3.0, dt=0.1, ratio=None, shed="both", space=True, alpha=10.0, AR=1.0):
    if ratio is None:
        ratio = CORE if solver is Revised else CUT
    s = VLMSurface(np.zeros(3), 1, False, shed, 1.0, np.ones(N+1)/AR, np.deg2rad(alpha), 0, 0, 0, 0, True, space, N, M)
    s._build_wing()
    v = solver([s], np.array([1.,0,0]), False, ratio, CUT)
    v._time_sim(T, dt, "classic"); v._kuttas_loads()
    ctrl = np.array([p.ctr for p in s.wing_panels["real"]])
    y = np.mean(ctrl.reshape(M, 2*N, 3), axis=0)[:, 1]          # spanwise position of each strip
    return dict(y=y.tolist(), Cl=(-s.Cl_2d).tolist(), CL=float(-2*s.loads[2]*AR), N=N, dt=dt, ratio=ratio, shed=shed, space=space, T=T)

if __name__ == "__main__":
    out = {"original": [], "revised": []}
    t0 = time.time()
    for name, solver in [("original", Original), ("revised", Revised)]:
        sweep = [0.4, 0.6, 1.0] if solver is Original else [0.02, 0.05, 0.1]
        for r in sweep:                                     # regularisation parameter sensitivity, N=15
            out[name].append(dict(case="ratio", **run(solver, ratio=r)))
        for N in [8, 15, 25, 40]:                           # spanwise refinement
            out[name].append(dict(case="N", **run(solver, N=N)))
        for N, dt in [(15,0.05),(15,0.025),(25,0.05),(25,0.025)]:   # time-step refinement
            out[name].append(dict(case="dt", **run(solver, N=N, dt=dt)))
        for N in [8, 15, 25, 40]:                           # uniform spacing
            out[name].append(dict(case="uniform", **run(solver, N=N, space=False)))
        out[name].append(dict(case="none", **run(solver, shed="none")))
        print(name, "study done", f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(ROOT, "docs", "tip_loading", "study.json"), "w"))
    ts = {"original": [], "revised": []}
    for name, solver in [("original", Original), ("revised", Revised)]:
        for N in [15, 40]:
            for T in [1, 2, 3, 5, 8]:
                ts[name].append(dict(case="T", **run(solver, N=N, T=float(T))))
        print(name, "time sweep done", f"{time.time()-t0:.0f}s", flush=True)
    json.dump(ts, open(os.path.join(ROOT, "docs", "tip_loading", "tsweep.json"), "w"))
    for name in out:
        print("=====", name)
        for d in out[name] + ts[name]:
            print(f"{d['case']:8s} N={2*d['N']:3d} dt={d['dt']:5.3f} T={d['T']:3.0f} par={d['ratio']:4.2f} shed={d['shed']:4s} uniform={not d['space']!s:5s} tipCl={d['Cl'][0]:7.3f} Cl[1]={d['Cl'][1]:6.3f} mid={d['Cl'][d['N']-1]:6.3f} CL={d['CL']:.4f}")
