"""
Verification of the lattice cavity against the linearised two-dimensional theory.

A rectangular wing of aspect ratio 40 at 4 deg with a uniform, prescribed cavitation number
(g = 0, dsigma = sigma) approximates a two-dimensional flat plate with a partial cavity from
the leading edge. The mid-span section is compared with Acosta (1955): cavity length L against
the cavity parameter Psi = sigma/(2 alpha), and the lift slope a0 = Cl/alpha against Acosta's
pi (1 + 1/sqrt(1 - L)). The closed forms come from vent_section.py.

Also checks the source panel kernel: normal velocity +-1/2 on either side of the panel, and
the far field of a point source.

Run from the repository root:  python docs/cavity/verify_acosta.py
"""
import sys, os, warnings, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from VLMSurface import VLMSurface
from VLMCavity import CavitySolver, source_panel_velocity
import vent_section as vs

warnings.simplefilter("ignore")


def kernel_checks():
    corners = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], float)   # unit square, LE at x=0
    normal = np.array([0, 0, 1.0])
    c = corners.mean(axis=0)
    v = source_panel_velocity(corners, normal, np.array([c + [0, 0, 1e-6], c - [0, 0, 1e-6]]))
    far = np.array([[30.0, 20.0, 10.0]])
    vf = source_panel_velocity(corners, normal, far)[0]
    r = far[0] - c; point = r/(4*np.pi*np.linalg.norm(r)**3)          # unit-area point source
    print(f"kernel: w(+) = {v[0, 2]:+.6f}  w(-) = {v[1, 2]:+.6f}  (expected +-0.5)")
    print(f"kernel: far field / point source = {np.linalg.norm(vf)/np.linalg.norm(point):.4f}  direction error = {np.linalg.norm(vf/np.linalg.norm(vf) - r/np.linalg.norm(r)):.2e}")
    assert abs(v[0, 2] - 0.5) < 1e-3 and abs(v[1, 2] + 0.5) < 1e-3
    assert abs(np.linalg.norm(vf)/np.linalg.norm(point) - 1) < 2e-2


def plate(alpha_deg, sigma, AR=40.0, N=12, M=16, T=8.0, force=True):
    c = 1.0; b = AR*c; u = 1.0
    s = VLMSurface(np.zeros(3), 1, False, "none", b, np.ones(N + 1)*c, np.deg2rad(alpha_deg), 0, 0, 0, 0, True, False, N, M)
    s._build_wing()
    v = CavitySolver([s], np.array([u, 0, 0]), False, 0.05, 0.4, g=0.0, dsigma=sigma, rate=10.0, washout=False)
    if force:
        v.vent.force(s, None)
    v._time_sim(T, 0.1, "classic"); v._kuttas_loads()
    return s, v


def main():
    kernel_checks()
    alpha = 4.0; a = np.deg2rad(alpha)
    # wetted reference at the same travel, so that the starting-vortex transient cancels in the ratios
    s0, v0 = plate(alpha, 1e6, force=False)
    j = s0.N//2
    a0_wet = -s0.Cl_2d[j]/a
    e_wet = v0.vent.state[id(s0)]["xcp"][j]
    print(f"\n wetted reference, AR = 40, alpha = 4 deg, mid-span: a0 = {a0_wet:.3f} (2 pi = {2*np.pi:.3f}, finite travel and span), e_cp = {e_wet:.3f} (thin aerofoil 0.25)")
    rows = []
    print(" Psi      L lattice   L Acosta    a0/a0_wet lattice   a0(L)/2pi Acosta   a0(L)/2pi fit   e_cp lattice  e_cp fit")
    for psi in [40.0, 20.0, 12.0, 8.0, 6.0, 4.0, 2.0]:
        sigma = 2*a*psi
        s, v = plate(alpha, sigma)
        st = v.vent.state[id(s)]
        L = st["L"][j]
        ratio = -s.Cl_2d[j]/a/a0_wet
        L_ac = float(vs.acosta_length(psi)) if psi >= float(vs.acosta_psi(0.5)) else float("nan")
        a0_ac = float(vs.acosta_lift_slope(L))/(2*np.pi) if L < 0.5 else float("nan")
        rows.append(dict(psi=psi, L=float(L), L_acosta=L_ac, a0_ratio=float(ratio), a0_acosta_ratio=a0_ac,
                         a0_fit_ratio=float(vs.lift_slope(L))/(2*np.pi), e_cp=float(st["xcp"][j]),
                         e_cp_fit=float(vs.centre_of_pressure(L)), open=bool(st["open"][j]), closure=float(st["closure"][j])))
        print(f" {psi:4.1f} {L:11.3f} {L_ac:10.3f} {ratio:19.3f} {a0_ac:18.3f} {vs.lift_slope(L)/(2*np.pi):15.3f} {st['xcp'][j]:13.3f} {vs.centre_of_pressure(L):9.3f}"
              + ("   open" if st["open"][j] else ""))
    json.dump(rows, open(os.path.join(ROOT, "docs", "cavity", "verify_acosta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
