"""
Build the provisional section table for a NACA 0009 section: src/section_data/naca0009.json

Two quantities are tabulated, both as functions the inception model needs.

1. Minimum pressure coefficient Cp_min(alpha) of the section in inviscid flow, from a
   two-dimensional linear-strength vortex panel method (Kuethe & Chow, Foundations of
   Aerodynamics, 5th ed., Sect. 5.10) on the NACA 0009 thickness form with a closed
   trailing edge (Abbott & von Doenhoff, Theory of Wing Sections, Eq. 6.2 with the
   closing coefficient). The inviscid suction peak bounds from above the suction the
   real section can hold, so it is a conservative measure of sub-atmospheric flow.

2. Separation angle alpha_sep(Re): the geometric incidence at which the suction side of a
   NACA 0009 is separated over a large part of the chord (stall, or a bursting leading-edge
   bubble). Taken from published lift curves: Sheldahl & Klimas (1981), Sandia report
   SAND80-2114, for Re from 1e5 to 1e6, and Abbott & von Doenhoff (1959) for Re from 3e6 to
   9e6. The values are the incidences of maximum lift, read to the nearest half degree.

PROVISIONAL. Replace with XFOIL polars (separation location, transition, Cp_min with
boundary layer) when available; the table format is documented in src/section_data.py.

Run from the repository root:  python docs/inception/make_naca0009_table.py
"""
import json, os, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def naca00xx(t, n=160):
    """NACA 00xx coordinates with a closed trailing edge, in the Kuethe & Chow order: from the
    trailing edge along the LOWER surface to the leading edge, then along the upper surface back. -> (2n-1, 2)"""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5*(1 - np.cos(beta))                                  # cosine clustering
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)   # -0.1036 closes the TE
    xl, yl = x[::-1], -yt[::-1]
    xu, yu = x[1:], yt[1:]
    return np.column_stack([np.concatenate([xl, xu]), np.concatenate([yl, yu])])


def vortex_panel_cp(coords, alpha):
    """
    Linear-strength vortex panel method (Kuethe & Chow). Returns the pressure coefficient at
    the panel control points and the lift coefficient.
    """
    xb, yb = coords[:, 0], coords[:, 1]
    m = len(xb) - 1
    x = 0.5*(xb[:-1] + xb[1:]); y = 0.5*(yb[:-1] + yb[1:])
    s = np.hypot(np.diff(xb), np.diff(yb))
    theta = np.arctan2(np.diff(yb), np.diff(xb))
    sn, cs = np.sin(theta), np.cos(theta)
    rhs = np.zeros(m + 1)
    rhs[:m] = np.sin(theta - alpha)
    cn1 = np.zeros((m, m)); cn2 = np.zeros((m, m)); ct1 = np.zeros((m, m)); ct2 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                cn1[i, j] = -1.0; cn2[i, j] = 1.0; ct1[i, j] = 0.5*np.pi; ct2[i, j] = 0.5*np.pi
                continue
            a = -(x[i] - xb[j])*cs[j] - (y[i] - yb[j])*sn[j]
            b = (x[i] - xb[j])**2 + (y[i] - yb[j])**2
            c = np.sin(theta[i] - theta[j]); d = np.cos(theta[i] - theta[j])
            e = (x[i] - xb[j])*sn[j] - (y[i] - yb[j])*cs[j]
            f = np.log(1 + s[j]*(s[j] + 2*a)/b)
            g = np.arctan2(e*s[j], b + a*s[j])
            p = (x[i] - xb[j])*np.sin(theta[i] - 2*theta[j]) + (y[i] - yb[j])*np.cos(theta[i] - 2*theta[j])
            q = (x[i] - xb[j])*np.cos(theta[i] - 2*theta[j]) - (y[i] - yb[j])*np.sin(theta[i] - 2*theta[j])
            cn2[i, j] = d + 0.5*q*f/s[j] - (a*c + d*e)*g/s[j]
            cn1[i, j] = 0.5*d*f + c*g - cn2[i, j]
            ct2[i, j] = c + 0.5*p*f/s[j] + (a*d - c*e)*g/s[j]
            ct1[i, j] = 0.5*c*f - d*g - ct2[i, j]
    an = np.zeros((m + 1, m + 1)); at = np.zeros((m, m + 1))
    an[:m, 0] = cn1[:, 0]; an[:m, m] = cn2[:, m - 1]
    an[:m, 1:m] = cn1[:, 1:] + cn2[:, :-1]
    an[m, 0] = 1.0; an[m, m] = 1.0                       # Kutta condition
    at[:, 0] = ct1[:, 0]; at[:, m] = ct2[:, m - 1]
    at[:, 1:m] = ct1[:, 1:] + ct2[:, :-1]
    gamma = np.linalg.solve(an, rhs)
    v = np.cos(theta - alpha) + at @ gamma
    cp = 1 - v**2
    # lift from the circulation: Gamma = 2 pi sum(gamma_avg s) in Kuethe-Chow normalisation (gamma = gamma'/(2 pi U))
    gam_avg = 0.5*(gamma[:-1] + gamma[1:])
    cl = 4*np.pi*np.sum(gam_avg*s)
    return cp, cl


def main():
    coords = naca00xx(0.09, n=120)
    alphas = np.arange(0.0, 26.0, 1.0)
    cp_min, cl = [], []
    for a in alphas:
        cp, c = vortex_panel_cp(coords, np.deg2rad(a))
        cp_min.append(float(cp.min())); cl.append(float(c))
    # separation (maximum-lift) incidence versus Reynolds number, provisional, degrees
    alpha_sep = {"Re": [1.0e5, 1.6e5, 3.6e5, 7.0e5, 1.0e6, 3.0e6, 6.0e6, 9.0e6],
                 "alpha_deg": [8.5, 9.0, 10.0, 11.0, 12.0, 13.5, 14.0, 15.0],
                 "source": ["Sheldahl & Klimas 1981 (Re 1e5-1e6, NACA 0009, read to 0.5 deg)",
                            "Abbott & von Doenhoff 1959 (Re 3e6-9e6, NACA 0009, incidence of maximum lift)"]}
    table = {
        "section": "NACA 0009",
        "status": "PROVISIONAL - inviscid Cp_min from a vortex panel method; separation angles read from published lift curves. Replace with XFOIL polars.",
        "thickness": 0.09,
        "alpha_deg": alphas.tolist(),
        "cp_min_inviscid": cp_min,
        "cl_inviscid": cl,
        "alpha_sep": alpha_sep,
        "cp_base_blunt_te": -0.3,
        "cp_base_note": "Base pressure coefficient behind a blunt trailing edge, typical value for a truncated section (Hoerner, Fluid-Dynamic Drag, Ch. 3). Used only when the surface is flagged blunt_te.",
        "lift_slope_2d": 2*np.pi,
    }
    out = os.path.join(ROOT, "src", "section_data", "naca0009.json")
    json.dump(table, open(out, "w"), indent=1)
    print("written", out)
    print("alpha  Cp_min  Cl (inviscid)")
    for a, c, l in zip(alphas, cp_min, cl):
        print(f"{a:5.1f} {c:8.3f} {l:7.3f}")


if __name__ == "__main__":
    main()
