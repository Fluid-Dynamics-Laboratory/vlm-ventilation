"""
End-to-end example: inception model -> regime machine -> cavity loads, on the strut of
Harwood, Young & Ceccio (2016) at Fn_h = 2.5, AR_h = 1.

The incidence is swept up from 4 deg and back down to 2 deg; the ventilation state is carried
from one incidence to the next. Inception is decided by VLMInception.assess with the geometric
separation gate (Sect. 6 of the report), the cavity and the loads by VLMCavity.CavitySolver.

Run from the repository root:  python docs/report/example_end_to_end.py   (about 5 min)
Writes docs/report/example_end_to_end.json.
"""
import sys, os, json, warnings, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from VLMSurface import VLMSurface
from VLMCavity import CavitySolver, carry_state
from section_data import SectionData
import VLMInception as vi

warnings.simplefilter("ignore")
G, NU = 9.81, 1.0e-6                 # gravity [m/s^2], kinematic viscosity of water [m^2/s]
C, AR_H, FN_H = 0.2794, 1.0, 2.5    # chord [m], immersed aspect ratio, depth Froude number
N, M = 12, 8                         # sections over the immersion, chordwise panels
data = SectionData.naca0009()        # viscous section data (NeuralFoil)


def inception(surface, solver):
    """The predicate the regime machine calls: the inception assessment with the geometric gate."""
    return vi.assess(surface, solver, data, g=G, nu=NU, band=0.5, seal="geometric")


def strut(alpha_deg):
    h = AR_H*C; u = FN_H*np.sqrt(G*h)
    s = VLMSurface(np.zeros(3), 0, True, "both", 2*h, np.ones(N + 1)*C, 0.0, np.deg2rad(alpha_deg), 0, 0, 0,
                   False, True, N, M)
    s._build_wing()
    v = CavitySolver([s], np.array([u, 0, 0]), "antisymmetric", 0.05, 0.4, g=G, rate=3.0, inception=inception)
    return s, v, u


def main():
    rows = []; prev = None
    for a in list(np.arange(4.0, 20.1, 2.0)) + list(np.arange(18.0, 1.9, -2.0)):
        s, v, u = strut(a)
        if prev is not None:
            carry_state(prev[1], prev[0], v, s)
        v._time_sim(4.0*C/u, 0.05*C/u, "classic"); v._kuttas_loads()
        st = v.vent.state[id(s)]
        cl = 2*abs(s.loads[1])/(u**2*C*C)
        rows.append(dict(alpha=float(a), CL=float(cl), regime=st["regime"], route=st["route"],
                         ventilated_fraction=float(np.mean(st["active"])),
                         L_mid=float(st["L"][N//2]), e_mid=float(st["xcp"][N//2]),
                         phi_mid_deg=float(np.degrees(st["phi_bar"])) if st["phi_bar"] is not None else None))
        print(f"alpha = {a:4.1f}  CL = {cl:.3f}  regime {st['regime']:2s}  route {str(st['route']):8s}  "
              f"ventilated {np.mean(st['active']):.2f}  L(mid-depth) = {st['L'][N//2]:.2f}  e(mid-depth) = {st['xcp'][N//2]:.3f}")
        prev = (s, v)
    json.dump(rows, open(os.path.join(ROOT, "docs", "report", "example_end_to_end.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
