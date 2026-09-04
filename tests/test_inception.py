"""
Unit checks of the inception model. Run from the repository root:  python tests/test_inception.py
Fast cases only (a few seconds); the validation against published values is docs/inception/validate.py.
"""
import sys, os, warnings, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from VLMSurface import VLMSurface
from VLMSolver import VLMSolver
from section_data import SectionData
import VLMInception as vi

warnings.simplefilter("ignore")
G, NU = 9.81, 1.0e-6


def small_strut(alpha_deg, u, blunt=False, N=4, M=2):
    c, h = 0.28, 0.28
    s = VLMSurface(np.zeros(3), 0, True, "both", 2*h, np.ones(N + 1)*c, 0.0, np.deg2rad(alpha_deg), 0, 0, 0,
                   False, True, N, M)
    s.blunt_te = blunt
    s._build_wing()
    v = VLMSolver([s], np.array([u, 0, 0]), "antisymmetric", 0.05, 0.4)
    v._time_sim(0.6*c/u, 0.1*c/u, "classic"); v._kuttas_loads()
    return s, v


def test_section_data():
    d = SectionData.naca0009()
    assert d.cp_min(0.0) < 0 and d.cp_min(10.0) < d.cp_min(5.0), "suction must grow with incidence"
    assert d.alpha_sep(1e5) < d.alpha_sep(1e6) < d.alpha_sep(1e7), "separation incidence must grow with Re"
    re_lo, re_hi = 10**d._re_sep.min(), 10**d._re_sep.max()
    assert d.alpha_sep(re_lo/100) == d.alpha_sep(re_lo) and d.alpha_sep(re_hi*100) == d.alpha_sep(re_hi), "held constant outside the table"
    assert d.separated(20.0, 1e6) and not d.separated(2.0, 1e6)
    assert 12.0 <= d.alpha_sep(1e6) <= 14.0, "NACA 0009 stalls at 13 deg near Re 1e6 (published 12 to 14.5)"
    assert -6 < d.cp_min(10.0, 1e6) < -3, "viscous suction peak at 10 deg is about -4.5"
    assert np.all(d.confidence(10.0, [5e5, 1e6, 3e6]) > 0.8) and d.confidence(10.0, 1e5) < 0.8
    # the provisional table still loads and answers the same questions
    q = SectionData.naca0009("provisional")
    assert not q.viscous and q.cp_min(10.0) < d.cp_min(10.0, 1e6), "the inviscid suction bounds the viscous one"


def test_condition_a_and_nose():
    d = SectionData.naca0009()
    s, v = small_strut(4.0, 4.0)
    r = vi.assess(s, v, d, G, NU)
    assert not r["incepts"] and r["prone_fraction"] == 0.0, "an attached section cannot be prone"
    s, v = small_strut(20.0, 4.0)
    r = vi.assess(s, v, d, G, NU, seal="geometric")
    assert r["routes"]["nose"][0] and r["route"] == "nose", "a stalled strut at the waterline must open the nose route"
    st = r["state"]
    assert np.all(st["sigma_c"] >= 0) and np.all(np.diff(st["depth"]) > 0), "depth and sigma_c increase along the strut"


def test_tail_threshold():
    d = SectionData.naca0009()
    for u, expect in [(0.8*np.sqrt(G*0.28), False), (1.5*np.sqrt(G*0.28), True)]:
        s, v = small_strut(6.0, u, blunt=True)
        r = vi.assess(s, v, d, G, NU)
        assert r["routes"]["tail"][0] == expect, f"tail route at Fn_h = {r['state']['Fn_h']:.2f}"
    s, v = small_strut(6.0, 1.5*np.sqrt(G*0.28), blunt=False)
    assert not vi.assess(s, v, d, G, NU)["routes"]["tail"][0], "no tail route without a blunt trailing edge"


def test_tip_route_reports_unvalidated():
    d = SectionData.naca0009()
    c = 0.3
    s = VLMSurface(np.array([0, 0, 0.5*c]), 1, True, "both", 6*c, np.ones(5)*c, np.deg2rad(6.0), 0, 0, 0, 0,
                   True, True, 4, 2); s._build_wing()
    v = VLMSolver([s], np.array([6.0, 0, 0]), "antisymmetric", 0.05, 0.4)
    v._time_sim(0.6*c/6.0, 0.1*c/6.0, "classic"); v._kuttas_loads()
    r = vi.assess(s, v, d, G, NU, submerged=True)
    t = r["routes"]["tip"]
    assert t is not None and t[1]["validated"] is False and t[1]["cp_core"] < 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok ", name)
