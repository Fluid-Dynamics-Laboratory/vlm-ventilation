"""
Template for comparing the code with digitised measurements.

Digitised data live in docs/validation/data/ as CSV files, one per source figure, with the
columns of the template docs/validation/data/TEMPLATE.csv (see docs/report/methods_and_validation.md,
Sect. 10). This script reads every CSV whose `quantity` is a lift coefficient of a surface-piercing
strut in the fully wetted or fully ventilated regime, runs the code at the same conditions, and
prints and plots the comparison. Extend it case by case as data arrive.

Run from the repository root:  python docs/validation/compare_template.py
"""
import sys, os, csv, glob, warnings, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from VLMSurface import VLMSurface
from VLMCavity import CavitySolver

warnings.simplefilter("ignore")
G = 9.81


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("alpha_deg", "Fn_h", "AR_h", "value", "uncertainty", "chord_m"):
            if k in r and r[k] not in ("", None):
                r[k] = float(r[k])
    return rows


def run_strut(alpha_deg, fnh, ar_h, chord, regime, N=12, M=8):
    h = ar_h*chord; u = fnh*np.sqrt(G*h)
    s = VLMSurface(np.zeros(3), 0, True, "both", 2*h, np.ones(N + 1)*chord, 0.0, np.deg2rad(alpha_deg), 0, 0, 0, False, True, N, M)
    s._build_wing()
    v = CavitySolver([s], np.array([u, 0, 0]), "antisymmetric", 0.05, 0.4, g=G, rate=10.0)
    if regime == "FV":
        v.vent.force(s, None)
    v._time_sim(3.0*chord/u, 0.05*chord/u, "classic"); v._kuttas_loads()
    return 2*abs(s.loads[1])/(u**2*h*chord)


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "docs", "validation", "data", "*.csv")))
    files = [f for f in files if not os.path.basename(f).startswith("TEMPLATE")]
    if not files:
        print("no digitised data yet: put CSV files in docs/validation/data/ following TEMPLATE.csv"); return
    for f in files:
        rows = [r for r in read(f) if r.get("quantity") == "CL" and r.get("regime") in ("FW", "FV")]
        if not rows:
            continue
        print(f"\n{os.path.basename(f)}  ({rows[0].get('source', '')})")
        print(" alpha   Fn_h  AR_h  regime  measured  computed  difference")
        for r in rows:
            cl = run_strut(r["alpha_deg"], r["Fn_h"], r["AR_h"], r.get("chord_m", 0.2794), r["regime"])
            print(f" {r['alpha_deg']:5.1f} {r['Fn_h']:6.2f} {r['AR_h']:5.2f}  {r['regime']:5s} {r['value']:9.3f} {cl:9.3f} {cl - r['value']:+11.3f}")


if __name__ == "__main__":
    main()
