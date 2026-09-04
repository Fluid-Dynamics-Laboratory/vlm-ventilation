"""
Unit checks of the cavity and regime model. Run from the repository root:  python tests/test_cavity.py
Fast cases only; the verification against Acosta is docs/cavity/verify_acosta.py and the validation
against the strut of Harwood et al. is docs/cavity/validate.py.
"""
import sys, os, warnings, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from VLMSurface import VLMSurface
from VLMSolver import VLMSolver
from VLMCavity import CavitySolver, source_panel_velocity, stall_gate, carry_state, L_MAX
import vent_section as vs

warnings.simplefilter("ignore")


def test_source_kernel():
    corners = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], float); n = np.array([0, 0, 1.0])
    c = corners.mean(axis=0)
    v = source_panel_velocity(corners, n, np.array([c + [0, 0, 1e-6], c - [0, 0, 1e-6], [30.0, 20.0, 10.0]]))
    assert abs(v[0, 2] - 0.5) < 1e-3 and abs(v[1, 2] + 0.5) < 1e-3, "normal velocity jump +-1/2"
    r = np.array([30.0, 20.0, 10.0]) - c
    assert abs(np.linalg.norm(v[2])*4*np.pi*np.linalg.norm(r)**2 - 1) < 2e-2, "far field of a point source"
    # in-plane velocity of a long strip against the two-dimensional sheet
    long = np.array([[0, -200, 0], [0, 200, 0], [1, 200, 0], [1, -200, 0]], float)
    u = source_panel_velocity(long, n, np.array([[1.5, 0.0, 1e-9]]))[0, 0]
    assert abs(u - np.log(1.5/0.5)/(2*np.pi)) < 1e-4


def test_wetted_unchanged():
    """A CavitySolver with no cavity reproduces the base solver."""
    s1 = VLMSurface(np.zeros(3), 1, False, "both", 1.0, np.ones(5), np.deg2rad(10), 0, 0, 0, 0, True, True, 4, 3); s1._build_wing()
    s2 = VLMSurface(np.zeros(3), 1, False, "both", 1.0, np.ones(5), np.deg2rad(10), 0, 0, 0, 0, True, True, 4, 3); s2._build_wing()
    v1 = VLMSolver([s1], np.array([1., 0, 0]), False, 0.05, 0.4); v1._time_sim(0.5, 0.1, "classic"); v1._kuttas_loads()
    v2 = CavitySolver([s2], np.array([1., 0, 0]), False, 0.05, 0.4); v2._time_sim(0.5, 0.1, "classic"); v2._kuttas_loads()
    assert np.allclose(s1.loads, s2.loads) and v2.vent.state[id(s2)]["regime"] == "FW"


def test_cavity_lowers_suction_and_lift_on_long_branch():
    """With a supercavity on every section (sigma small) the lift falls to the sectional slope ratio."""
    N, M = 4, 6; a = np.deg2rad(6.0)
    s = VLMSurface(np.zeros(3), 1, False, "none", 8.0, np.ones(N + 1), a, 0, 0, 0, 0, True, False, N, M); s._build_wing()
    v0 = CavitySolver([s], np.array([1., 0, 0]), False, 0.05, 0.4, g=0.0, dsigma=0.02, washout=False)
    v0._time_sim(3.0, 0.1, "classic"); v0._kuttas_loads(); cl_w = -s.Cl_2d[N]
    s2 = VLMSurface(np.zeros(3), 1, False, "none", 8.0, np.ones(N + 1), a, 0, 0, 0, 0, True, False, N, M); s2._build_wing()
    v = CavitySolver([s2], np.array([1., 0, 0]), False, 0.05, 0.4, g=0.0, dsigma=0.02, rate=10.0, washout=False)
    v.vent.force(s2, None); v._time_sim(3.0, 0.1, "classic"); v._kuttas_loads()
    st = v.vent.state[id(s2)]
    assert np.all(st["L"] > L_MAX) and np.all(st["branch"] == "sectional")
    ratio = -s2.Cl_2d[N]/cl_w
    expect = float(vs.lift_slope(st["L"][N]))/(2*np.pi)
    assert abs(ratio - expect) < 0.05, f"long-branch lift ratio {ratio:.3f} against sectional {expect:.3f}"


def test_partial_cavity_lattice_branch():
    """Psi = 8 at 4 deg: L = 0.25 on the lattice branch, lift ratio within 10 % of Acosta."""
    N, M = 4, 16; a = np.deg2rad(4.0); psi = 8.0
    def plate(force):
        s = VLMSurface(np.zeros(3), 1, False, "none", 40.0, np.ones(N + 1), a, 0, 0, 0, 0, True, False, N, M); s._build_wing()
        v = CavitySolver([s], np.array([1., 0, 0]), False, 0.05, 0.4, g=0.0, dsigma=2*a*psi, rate=10.0, washout=False)
        if force:
            v.vent.force(s, None)
        v._time_sim(4.0, 0.1, "classic"); v._kuttas_loads(); return s, v
    s0, _ = plate(False); s1, v1 = plate(True)
    st = v1.vent.state[id(s1)]
    L = st["L"][N]
    assert 0.15 < L <= L_MAX and st["branch"][N] == "lattice"
    ratio = s1.Cl_2d[N]/s0.Cl_2d[N]
    assert abs(ratio/(float(vs.acosta_lift_slope(L))/(2*np.pi)) - 1) < 0.10


def test_regime_and_hysteresis():
    """Stall gate: wetted below, ventilated above; the cavity persists when the incidence is lowered."""
    N, M = 6, 5; C = 0.28; h = C; u = 2.5*np.sqrt(9.81*h)
    def strut(alpha):
        s = VLMSurface(np.zeros(3), 0, True, "both", 2*h, np.ones(N + 1)*C, 0.0, np.deg2rad(alpha), 0, 0, 0, False, True, N, M); s._build_wing()
        v = CavitySolver([s], np.array([u, 0, 0]), "antisymmetric", 0.05, 0.4, g=9.81, inception=stall_gate(14.5), rate=1.0)
        return s, v
    s, v = strut(10.0); v._time_sim(1.0*C/u, 0.1*C/u, "classic")
    assert v.vent.state[id(s)]["regime"] == "FW"
    s2, v2 = strut(16.0); carry_state(v, s, v2, s2); v2._time_sim(2.0*C/u, 0.1*C/u, "classic")
    st2 = v2.vent.state[id(s2)]
    assert st2["regime"] in ("PV", "FV") and np.any(st2["active"]), "the stall gate must open the cavity"
    s3, v3 = strut(10.0); carry_state(v2, s2, v3, s3); v3._time_sim(2.0*C/u, 0.1*C/u, "classic")
    st3 = v3.vent.state[id(s3)]
    assert np.any(st3["active"]) or st3["route"] == "washout", "below stall the cavity either persists or washes out by the jet criterion, never by the gate"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok ", name)
