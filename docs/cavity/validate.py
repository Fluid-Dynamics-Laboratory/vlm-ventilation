"""
Validation of the cavity and regime model against the scalar values stated in the literature.

Strut of Harwood, Young & Ceccio (J. Fluid Mech. 800, 2016): rectangular section, immersed
aspect ratio AR_h = 1, chord 0.2794 m (assumed), atmospheric cavity (dsigma = 0), antisymmetric
free-surface image. 12 sections over the immersion, 8 chordwise panels.

Case A  Fully ventilated branch imposed at Fn_h = 2.5, alpha = 6, 10, 14, 18 deg: ratio of
        ventilated to wetted lift, centre of pressure wetted and ventilated, cavity planform,
        mean closure angle. Published scalars: lift loss of 50 to 70 % (Breslin & Skalak 1959;
        Harwood et al. 2016; Young et al. 2017), centre of pressure of a supercavitating section
        at 3c/16 = 0.1875 forward of mid-chord; the panel method of Viola's bem-fem-fsi gives
        ratios 0.544 to 0.575 and e = 0.08 to 0.19 at the same conditions.

Case B  Closure angle at alpha = 20 deg, Fn_h = 1.5, AR_h = 1: Harwood et al. measured a mean
        closure angle of 40.75 deg at washout and proposed 45 deg as the stability limit.

Case C  Hysteresis at Fn_h = 2.5: incidence swept up from a wetted state with the stall gate at
        14.5 deg (the incidence-only formation boundary of the paper) and back down from the
        ventilated state until washout by the re-entrant-jet criterion. Published: bi-stable
        band from the bifurcation angle up to stall; the paper's Fig. 16.

Run from the repository root:  python docs/cavity/validate.py   (about 10 min)
Writes docs/cavity/validate.json.
"""
import sys, os, json, warnings, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from VLMSurface import VLMSurface
from VLMCavity import CavitySolver, stall_gate, carry_state
import vent_section as vs

warnings.simplefilter("ignore")
G = 9.81; C = 0.2794; N = 12; M = 8


def strut(alpha_deg, fnh, ar_h=1.0, T=3.0, inception=None, rate=0.2):
    h = ar_h*C; u = fnh*np.sqrt(G*h)
    s = VLMSurface(np.zeros(3), 0, True, "both", 2*h, np.ones(N + 1)*C, 0.0, np.deg2rad(alpha_deg), 0, 0, 0, False, True, N, M)
    s._build_wing()
    v = CavitySolver([s], np.array([u, 0, 0]), "antisymmetric", 0.05, 0.4, g=G, rate=rate, inception=inception)
    return s, v, u, h


def run(s, v, u, T=3.0):
    v._time_sim(T*C/u, 0.05*C/u, "classic"); v._kuttas_loads()
    return float(2*abs(s.loads[1])/(u**2*C*C))          # CL on h c with h = c (AR_h = 1); lift is the y force of a strut


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o


def main():
    out = {}
    # ---- Case A
    print("Case A: fully ventilated branch, Fn_h = 2.5, AR_h = 1")
    print(" alpha  CL wet   CL vent  ratio  D-S 2019   e wet   e vent   L range (root..tip)   phi_mid  branches")
    caseA = []
    def damley_strnad_ratio(alpha_deg, fnh, ar_h=1.0):
        # Damley-Strnad, Harwood & Young (smp'19) eq. (7), atmospheric cavity: psi = sigma_c/alpha with the
        # mean-depth cavitation number 1/Fn_h^2
        psi = (1.0/fnh**2)/np.deg2rad(alpha_deg)
        return (1 - (ar_h - 1)/(2*ar_h + 1)*np.exp(-psi))*(psi**2 - 0.935*psi + 1)/(psi**2 - 1.535*psi + 2)
    for a in [6.0, 10.0, 14.0, 18.0]:
        s, v, u, h = strut(a, 2.5); cl_w = run(s, v, u)
        e_w = float(np.nanmean(v.vent.state[id(s)]["xcp"]))
        s, v, u, h = strut(a, 2.5, rate=10.0); v.vent.force(s, None); cl_v = run(s, v, u)
        st = v.vent.state[id(s)]
        e_v = float(np.nanmean(st["xcp"][st["active"]])) if np.any(st["active"]) else float("nan")
        L = st["L"]; phi = np.degrees(st["phi_bar"]) if st["phi_bar"] is not None else float("nan")
        br = {k: int(np.sum(st["branch"] == k)) for k in ("wet", "lattice", "sectional")}
        ds = float(damley_strnad_ratio(a, 2.5))
        caseA.append(dict(alpha=a, CL_wet=cl_w, CL_vent=cl_v, ratio=cl_v/cl_w, ratio_damley_strnad=ds, e_wet=e_w, e_vent=e_v, L=L,
                          phi_mid=phi, regime=st["regime"], branches=br, depth_sec=v._cavity_geo[id(s)]["depth_sec"]/C))
        print(f" {a:5.1f} {cl_w:7.4f} {cl_v:8.4f} {cl_v/cl_w:6.3f} {ds:8.3f} {e_w:7.3f} {e_v:8.3f}   {L[0]:.2f} .. {L[-1]:.2f}            {phi:6.1f}   {br}  {st['regime']}")
    out["caseA"] = caseA
    # ---- Case B
    print("Case B: closure angle at alpha = 20 deg, Fn_h = 1.5 (measured 40.75 deg, criterion 45 deg)")
    caseB = []
    for fnh in [1.5, 2.0, 2.5]:
        s, v, u, h = strut(20.0, fnh, rate=10.0); v.vent.force(s, None); cl = run(s, v, u)
        st = v.vent.state[id(s)]
        phi = np.degrees(st["phi_bar"]) if st["phi_bar"] is not None else float("nan")
        fn_wash = float(vs.washout_froude(cl, 1.0))
        caseB.append(dict(Fn_h=fnh, phi_mid=phi, CL=cl, L=st["L"], Fn_washout_of_CL=fn_wash, regime=st["regime"]))
        print(f"  Fn_h = {fnh:.1f}: phi(mid-depth) = {phi:5.1f} deg  CL = {cl:.3f}  washout Fn_h(CL) of Harwood = {fn_wash:.2f}  regime {st['regime']}")
    out["caseB"] = caseB
    # ---- Case C: hysteresis
    print("Case C: hysteresis at Fn_h = 2.5, stall gate 14.5 deg")
    gate = stall_gate(14.5)
    up = []; prev = None
    for a in np.arange(4.0, 20.1, 2.0):
        s, v, u, h = strut(a, 2.5, inception=gate, rate=3.0)         # rate 3: the cavity reaches equilibrium within the 4 chords of each step
        if prev is not None:
            carry_state(prev[1], prev[0], v, s)
        cl = run(s, v, u, T=4.0); st = v.vent.state[id(s)]
        up.append(dict(alpha=a, CL=cl, regime=st["regime"], active=float(np.mean(st["active"])), route=st["route"]))
        print(f"  up   alpha = {a:4.1f}  CL = {cl:.4f}  regime {st['regime']}  ventilated fraction {np.mean(st['active']):.2f}  route {st['route']}")
        prev = (s, v)
    down = []
    for a in np.arange(20.0, 1.9, -2.0):
        s, v, u, h = strut(a, 2.5, inception=gate, rate=3.0)
        carry_state(prev[1], prev[0], v, s)
        cl = run(s, v, u, T=4.0); st = v.vent.state[id(s)]
        phi = np.degrees(st["phi_bar"]) if st["phi_bar"] is not None else float("nan")
        down.append(dict(alpha=a, CL=cl, regime=st["regime"], active=float(np.mean(st["active"])), phi_bar=phi, route=st["route"]))
        print(f"  down alpha = {a:4.1f}  CL = {cl:.4f}  regime {st['regime']}  ventilated fraction {np.mean(st['active']):.2f}  phi_bar {phi:5.1f}  route {st['route']}")
        prev = (s, v)
    out["caseC"] = dict(up=up, down=down)
    json.dump(clean(out), open(os.path.join(ROOT, "docs", "cavity", "validate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
