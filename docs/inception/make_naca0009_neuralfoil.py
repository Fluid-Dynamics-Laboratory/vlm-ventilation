"""
Build the viscous section table for a NACA 0009 with NeuralFoil: src/section_data/naca0009_neuralfoil.json

NeuralFoil (P. Sharpe, https://github.com/peterdsharpe/NeuralFoil, MIT licence) is a pure Python
surrogate of XFOIL trained on about eight million XFOIL runs. It runs offline in milliseconds
and returns, besides the force coefficients, the boundary-layer edge velocity and shape factor
at 32 chordwise stations and a confidence value. `SectionData.from_neuralfoil` turns these
into the quantities the inception model needs; this script applies it to the NACA 0009 and
prints the derived separation incidence against the provisional table and against the
published values, then a check of the confidence over the Reynolds range of the experiments.

Run from the repository root:  python docs/inception/make_naca0009_neuralfoil.py
"""
import sys, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from section_data import SectionData


def naca00xx(t, n=100):
    """NACA 00xx with a closed trailing edge in Selig order (TE -> upper -> LE -> lower -> TE)."""
    beta = np.linspace(0, np.pi, n); x = 0.5*(1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    return np.vstack([np.column_stack([x[::-1], yt[::-1]]), np.column_stack([x[1:], -yt[1:]])])


def main():
    out = os.path.join(ROOT, "src", "section_data", "naca0009_neuralfoil.json")
    d = SectionData.from_neuralfoil(naca00xx(0.09), "NACA 0009", save=out)
    p = SectionData.naca0009("provisional")
    print("written", out)
    print(" Re        alpha_sep NeuralFoil   provisional table   confidence at alpha_sep   Cp_min at 10 deg (visc / inviscid)")
    for re in [1e5, 2e5, 5e5, 7e5, 1e6, 1.6e6, 3e6, 6e6, 1e7]:
        a = float(d.alpha_sep(re))
        print(f" {re:8.1e} {a:14.1f} {float(p.alpha_sep(re)):19.1f} {float(d.confidence(a, re)):21.2f}"
              f" {float(d.cp_min(10.0, re)):16.2f} / {float(p.cp_min(10.0)):.2f}")
    print("published: 13 to 15 deg at Re 3e6 to 9e6 (Abbott & von Doenhoff); 9 to 12 deg at Re 1e5 to 1e6 (Sheldahl & Klimas);")
    print("           Harwood et al. (2016) formed cavities by stall at about 14.5 deg at Re about 1e6 on a modified NACA 0009.")


if __name__ == "__main__":
    main()
