"""
Validation of the inception model against the scalar values stated in the literature.

Case 1  Surface-piercing strut of Harwood, Young & Ceccio (J. Fluid Mech. 800, 2016):
        rectangular NACA 0009-type section, chord 0.2794 m (assumed from the paper's 11 in
        model), immersed aspect ratio AR_h = 1, depth Froude numbers 1.5 to 3.5, atmospheric
        pressure. The model predicts the incidence at which the nose route opens. Published
        scalar: spontaneous (stall-induced) inception at about 14.5 deg, weakly dependent on
        Fn_h in this range (the vertical boundary of the paper's regime map, Fig. 16, as used
        by Viola's bem-fem-fsi). Aguiar Ferreira et al. (J. Fluid Mech. 2026) report nose
        inception at 15 to above 25 deg for a NACA 0010-34 at Fr 0.5 to 2.5, AR 1 to 1.5.

Case 2  The same strut with a blunt trailing edge: the tail route and its Froude threshold
        Fn_h > AR_h**(-1/2) (Aguiar Ferreira et al. 2026). The threshold is implemented as
        published, so this case checks the plumbing, not the physics.

Case 3  A generic swept L-foil horizontal element (rectangular, sweep 20 deg, aspect ratio 6,
        submerged half a chord) at 6 deg: the tip-vortex route. UNVALIDATED: reported only.

Run from the repository root:  python docs/inception/validate.py
Writes docs/inception/validate.json.
"""
import sys, os, json, warnings, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from VLMSurface import VLMSurface
from VLMSolver import VLMSolver
from section_data import SectionData
import VLMInception as vi

G = 9.81; NU = 1.0e-6
data = SectionData.naca0009()


def strut(alpha_deg, fnh, ar_h=1.0, c=0.2794, N=12, M=5, T=3.0, blunt=False):
    """Vertical surface-piercing strut with the antisymmetric image; alpha is the drift angle."""
    h = ar_h*c
    u = fnh*np.sqrt(G*h)
    s = VLMSurface(np.zeros(3), 0, True, "both", 2*h, np.ones(N + 1)*c, 0.0, np.deg2rad(alpha_deg), 0, 0, 0,
                   False, True, N, M)          # span 2h with sym=False: the strut occupies z in [0, h]
    s.blunt_te = blunt
    s._build_wing()
    v = VLMSolver([s], np.array([u, 0, 0]), "antisymmetric", 0.05, 0.4)
    v._time_sim(T*c/u, 0.05*c/u, "classic"); v._kuttas_loads()
    return s, v


def lfoil(alpha_deg, depth_c=0.5, c=0.3, AR=6.0, sweep_deg=20.0, N=10, M=4, T=4.0, u=6.0):
    """Horizontal swept element of an L-foil, submerged depth_c chords below the free surface."""
    b = AR*c
    s = VLMSurface(np.array([0, 0, depth_c*c]), 1, True, "both", b, np.ones(N + 1)*c, np.deg2rad(alpha_deg), 0,
                   np.deg2rad(sweep_deg), 0, 0, True, True, N, M)
    s._build_wing()
    v = VLMSolver([s], np.array([u, 0, 0]), "antisymmetric", 0.05, 0.4)
    v._time_sim(T*c/u, 0.05*c/u, "classic"); v._kuttas_loads()
    return s, v


def main():
    out = {}
    warnings.simplefilter("ignore")
    # Case 1: inception incidence against Fn_h
    print("Case 1: nose route, NACA 0009 strut, AR_h = 1")
    case1 = []
    for fnh in [1.5, 2.0, 2.5, 3.5]:
        alpha_inc = None; rows = []
        for a in np.arange(8.0, 30.1, 1.0):
            s, v = strut(a, fnh)
            r = vi.assess(s, v, data, G, NU, band=0.5)
            rows.append(dict(alpha=a, prone_fraction=r["prone_fraction"], nose=r["routes"]["nose"][0],
                             alpha_eff_max=float(r["state"]["alpha_eff"].max()), Re=float(r["state"]["Re"].mean()),
                             alpha_sep=float(r["state"]["alpha_sep"].mean())))
            if r["routes"]["nose"][0] and alpha_inc is None:
                alpha_inc = float(a)
        case1.append(dict(Fn_h=fnh, alpha_incep=alpha_inc, rows=rows))
        print(f"  Fn_h = {fnh:.1f}  Re = {rows[0]['Re']:.2e}  alpha_sep(Re) = {rows[0]['alpha_sep']:.1f} deg  "
              f"predicted nose inception at alpha = {alpha_inc} deg")
    out["case1"] = case1
    # Case 1b: sensitivity of the predicted inception incidence to the waterline band, Fn_h = 2.5
    print("Case 1b: sensitivity to the waterline band, Fn_h = 2.5")
    case1b = []
    for band in [0.25, 0.5, 1.0]:
        alpha_inc = None
        for a in np.arange(8.0, 30.1, 1.0):
            s, v = strut(a, 2.5)
            r = vi.assess(s, v, data, G, NU, band=band)
            if r["routes"]["nose"][0]:
                alpha_inc = float(a); break
        case1b.append(dict(band=band, alpha_incep=alpha_inc))
        print(f"  band = {band:.2f} c  predicted nose inception at alpha = {alpha_inc} deg")
    out["case1b"] = case1b
    # Case 1c: the geometric separation gate (Harwood's incidence-only boundary)
    print("Case 1c: geometric separation gate, nose route")
    case1c = []
    for fnh in [1.5, 2.5, 3.5]:
        alpha_inc = None
        for a in np.arange(8.0, 30.1, 0.5):
            s, v = strut(a, fnh)
            r = vi.assess(s, v, data, G, NU, band=0.5, seal="geometric")
            if r["routes"]["nose"][0]:
                alpha_inc = float(a); break
        case1c.append(dict(Fn_h=fnh, alpha_incep=alpha_inc))
        print(f"  Fn_h = {fnh:.1f}  predicted nose inception at alpha = {alpha_inc} deg  (Harwood et al. 2016: about 14.5 deg)")
    out["case1c"] = case1c
    # Case 2: tail route
    print("Case 2: tail route, blunt trailing edge, alpha = 10 deg")
    case2 = []
    for fnh in [0.8, 1.0, 1.5, 2.5]:
        s, v = strut(10.0, fnh, blunt=True)
        r = vi.assess(s, v, data, G, NU)
        t = r["routes"]["tail"]
        case2.append(dict(Fn_h=fnh, open=t[0], **t[1]))
        print(f"  Fn_h = {fnh:.1f}  threshold AR_h^-1/2 = {t[1]['Fn_h_threshold']:.2f}  tail route {'OPEN' if t[0] else 'closed'}  {t[1]}")
    out["case2"] = case2
    # Case 3: tip-vortex route on a generic L-foil element
    print("Case 3: tip-vortex route, generic swept L-foil element (UNVALIDATED)")
    case3 = []
    for depth_c in [0.1, 0.25, 0.5, 1.0]:
        s, v = lfoil(6.0, depth_c=depth_c)
        r = vi.assess(s, v, data, G, NU, submerged=True)
        t = r["routes"]["tip"]
        case3.append(dict(depth_c=depth_c, open=t[0], **t[1], CL=float(-2*s.loads[2]/(0.3*1.8)/36.0)))
        print(f"  depth/c = {depth_c:.2f}  Gamma_v = {t[1]['gamma_v']:.3f}  Cp_core = {t[1]['cp_core']:.2f}  "
              f"min depth of tip filament = {t[1]['depth_min']:.3f} m  r_c = {t[1]['core_radius']:.3f} m  route {'OPEN' if t[0] else 'closed'}")
    out["case3"] = case3
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    json.dump(clean(out), open(os.path.join(ROOT, "docs", "inception", "validate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
