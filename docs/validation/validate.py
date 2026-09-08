"""
Validation of the code against the digitised curves in docs/validation/data/ (template format,
see docs/report/methods_and_validation.md, Sect. 10, and data/raw/README.md for the provenance).

Case A  Wetted solver without free surface: lift curves of rectangular wings of aspect ratio 1
        and 3 against the free-wake vortex lattice of Rodriguez (1990) and the experiments
        reported by Lamar (1974); 45 deg swept wing of aspect ratio 5 against the straight-wake
        vortex lattice of Bertin and Smith (1998) / Tornado and the experiment of Weber and
        Brebner (1958).
Case B  Wetted strut with the antisymmetric image: C_L(alpha) at AR_h = 0.5, 1, 1.5 against
        the towing-tank data of Harwood, Young and Ceccio (2016) at every Froude number, and
        against their lifting-line model. Two numerical variants at AR_h = 1: no vortex shedding
        from the waterline edge, and the symmetric (rigid-wall) image.
Case C  Fully ventilated strut: C_L(alpha) at every (AR_h, Fn_h) of the data.
Case D  Cavity profile L(z) at alpha = 10 deg, Fn_h = 1.5, AR_h = 1 against the photograph and
        the lifting-line model of Harwood et al., with the sectional lift, effective incidence,
        lift slope and cavitation number of the same model.

Run from the repository root:  python docs/validation/validate.py          (about 25 min)
                               python docs/validation/validate.py --plot   (figures from validate.json only)
                               python docs/validation/validate.py --only=C (rerun one case, keep the others)
Writes docs/validation/validate.json and fig1_wings.png ... fig4_profile.png.
"""
import sys, os, csv, glob, json, time, warnings, subprocess, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from VLMSurface import VLMSurface
from VLMSolver import VLMSolver
from VLMCavity import CavitySolver

warnings.simplefilter("ignore")
G = 9.81; C = 0.2794; N_STRUT = 12; M_STRUT = 8


# ------------------------------------------------------------------ data
def read(name):
    with open(os.path.join(DATA, name), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("alpha_deg", "Fn_h", "AR_h", "value", "uncertainty", "chord_m"):
            r[k] = float(r[k]) if r.get(k, "") not in ("", None) else None
        r["z_over_h"] = None
        for tok in (r.get("notes") or "").split(";"):
            tok = tok.strip()
            if tok.startswith("z_over_h="):
                r["z_over_h"] = float(tok.split("=")[1])
    return rows


def curve(rows, **sel):
    """(alpha, value) arrays of the rows matching the selection, sorted by alpha."""
    pts = [(r["alpha_deg"], r["value"]) for r in rows if all(abs((r[k] or 0) - v) < 1e-9 if isinstance(v, float) else r[k] == v for k, v in sel.items())]
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def profile(rows, quantity):
    pts = sorted((r["z_over_h"], r["value"]) for r in rows if r["quantity"] == quantity)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def slope_through_origin(a_deg, cl, amax=10.0):
    """Lift slope per radian from a least-squares line through the origin over alpha <= amax."""
    m = (a_deg <= amax + 1e-9) & (a_deg > 0)
    a = np.deg2rad(a_deg[m]); y = cl[m]
    return float((a @ y)/(a @ a)) if a.size else float("nan")


def compare(a_code, cl_code, a_data, cl_data):
    """Code against data interpolated at the computed incidences within the data range."""
    m = (a_code >= a_data.min() - 1e-9) & (a_code <= a_data.max() + 1e-9)
    if not np.any(m):
        return dict(n=0)
    d = np.interp(a_code[m], a_data, cl_data)
    rel = (cl_code[m] - d)/np.where(np.abs(d) > 1e-9, d, np.nan)
    return dict(n=int(m.sum()), alpha=a_code[m].tolist(), data=d.tolist(), code=cl_code[m].tolist(),
                mean_rel_diff=float(np.nanmean(rel)), max_abs_diff=float(np.max(np.abs(cl_code[m] - d))),
                slope_code=slope_through_origin(a_code, cl_code), slope_data=slope_through_origin(a_data, cl_data))


# ------------------------------------------------------------------ runs
def run_wing(AR, alpha_deg, sweep_deg=0.0, N=15, M=5, T_chords=None, dt_chords=0.05, ratio=0.05, shed="both"):
    """Rectangular (optionally swept) wing, span 1, no free surface, tip shedding as given. -> CL"""
    c = 1.0/AR
    T = (T_chords if T_chords is not None else max(3.0, 2.0*AR))*c          # at least three chords and two spans of travel
    s = VLMSurface(np.zeros(3), 1, False, shed, 1.0, np.ones(N + 1)*c, np.deg2rad(alpha_deg), 0, np.deg2rad(sweep_deg), 0, 0, True, True, N, M)
    s._build_wing()
    v = VLMSolver([s], np.array([1.0, 0, 0]), False, ratio, 0.4)
    v._time_sim(T, dt_chords*c, "classic"); v._kuttas_loads()
    return float(-2*s.loads[2]*AR)


def strut(alpha_deg, fnh, ar_h, shedding="both", image="antisymmetric", N=N_STRUT, M=M_STRUT, rate=10.0):
    h = ar_h*C; u = fnh*np.sqrt(G*h)
    s = VLMSurface(np.zeros(3), 0, True, shedding, 2*h, np.ones(N + 1)*C, 0.0, np.deg2rad(alpha_deg), 0, 0, 0, False, True, N, M)
    s._build_wing()
    v = CavitySolver([s], np.array([u, 0, 0]), image, 0.05, 0.4, g=G, rate=rate)
    return s, v, u, h


def run_strut(alpha_deg, fnh, ar_h, regime="FW", T=3.0, **kw):
    s, v, u, h = strut(alpha_deg, fnh, ar_h, **kw)
    if regime == "FV":
        v.vent.force(s, None)
    v._time_sim(T*C/u, 0.05*C/u, "classic"); v._kuttas_loads()
    cl = float(2*abs(s.loads[1])/(u**2*h*C))
    st = v.vent.state[id(s)]; g = v._cavity_geo[id(s)]
    return dict(CL=cl, regime=st["regime"], active=float(np.mean(st["active"])), L=st["L"].tolist(), z_over_h=(g["depth_sec"]/h).tolist(),
                Cl_2d=np.abs(s.Cl_2d).tolist(), sigma=g["sigma_sec"].tolist())


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ cases
def case_A(out):
    print("Case A: wings without free surface")
    rod = read("rodriguez1990_VLM_CL.csv"); lam = read("lamar1974_exp_CL.csv")
    tor = read("bertin1998_tornado_VLM_CL.csv"); web = read("weber1958_exp_CL.csv")
    res = {}
    # the rectangular wings shed from the tips (side-edge separation of a low aspect ratio wing); the swept wing of aspect
    # ratio 5 does not: with tip shedding and cosine spacing its solution diverges at alpha <= 4 deg (recorded below)
    for key, AR, sweep, shed, alphas in (("rect_AR1", 1.0, 0.0, "both", np.arange(2.0, 16.1, 2.0)), ("rect_AR3", 3.0, 0.0, "both", np.arange(2.0, 12.1, 2.0)),
                                         ("swept45_AR5", 5.0, 45.0, "none", np.arange(2.0, 10.1, 2.0))):
        t = time.time(); cl = np.array([run_wing(AR, a, sweep, shed=shed) for a in alphas])
        # travel check at 8 deg: twice the default travel
        cl_long = run_wing(AR, 8.0, sweep, T_chords=2*max(3.0, 2.0*AR), shed=shed)
        r = dict(AR=AR, sweep_deg=sweep, shedding=shed, alpha=alphas.tolist(), CL=cl.tolist(), CL_8deg_double_travel=cl_long, CL_8deg=float(cl[list(alphas).index(8.0)]))
        if sweep != 0:
            r["CL_tip_shedding"] = [run_wing(AR, a, sweep, shed="both") for a in alphas]
            print(f"  {key} with tip shedding: CL = {np.round(r['CL_tip_shedding'], 3).tolist()}")
        if sweep == 0:
            for name, rows in (("rodriguez", rod), ("lamar", lam)):
                a, y = curve(rows, AR_h=AR); r[name] = dict(alpha=a.tolist(), CL=y.tolist(), cmp=compare(alphas, cl, a, y))
        else:
            for name, rows in (("tornado", tor), ("weber", web)):
                a, y = curve(rows, AR_h=AR); r[name] = dict(alpha=a.tolist(), CL=y.tolist(), cmp=compare(alphas, cl, a, y))
        res[key] = r
        print(f"  {key}: slope {slope_through_origin(alphas, cl):.3f}/rad; CL(8 deg) {r['CL_8deg']:.4f}, double travel {cl_long:.4f}; "
              + "; ".join(f"{k} slope {r[k]['cmp']['slope_data']:.3f}/rad, mean diff {100*r[k]['cmp']['mean_rel_diff']:+.1f} %" for k in r if isinstance(r[k], dict) and "cmp" in r[k])
              + f"  ({time.time() - t:.0f} s)")
    out["caseA"] = res


def case_B(out):
    print("Case B: wetted strut, antisymmetric image")
    fw = read("harwood2016_CL_FW.csv"); llm = read("harwood2016_LLM_CL.csv")
    alphas = np.arange(2.5, 15.1, 2.5); res = {}
    for ar in (0.5, 1.0, 1.5):
        t = time.time(); runs = [run_strut(a, 2.5, ar) for a in alphas]; cl = np.array([r["CL"] for r in runs])
        r = dict(AR_h=ar, alpha=alphas.tolist(), CL=cl.tolist(), Cl_2d_10deg=runs[3]["Cl_2d"], z_over_h=runs[3]["z_over_h"], data={})
        fns = sorted({row["Fn_h"] for row in fw if abs(row["AR_h"] - ar) < 1e-9})
        for fn in fns:
            a, y = curve(fw, AR_h=ar, Fn_h=fn); r["data"][f"{fn:.1f}"] = dict(alpha=a.tolist(), CL=y.tolist(), cmp=compare(alphas, cl, a, y))
        if ar == 1.0:
            a, y = curve(llm, regime="FW"); r["llm"] = dict(alpha=a.tolist(), CL=y.tolist(), cmp=compare(alphas, cl, a, y))
        res[f"AR{ar}"] = r
        print(f"  AR_h = {ar}: slope {slope_through_origin(alphas, cl):.3f}/rad; " + "; ".join(f"Fn {k}: data slope {v['cmp']['slope_data']:.3f}, mean diff {100*v['cmp']['mean_rel_diff']:+.1f} %" for k, v in r["data"].items()) + f"  ({time.time() - t:.0f} s)")
    # numerical variants at AR_h = 1
    var = {}
    for key, kw in (("no_waterline_shedding", dict(shedding="right")), ("symmetric_image", dict(image="symmetric")), ("no_tip_shedding", dict(shedding="none")), ("N24_M8", dict(N=24))):
        runs = [run_strut(a, 2.5, 1.0, **kw) for a in alphas]; cl = np.array([r["CL"] for r in runs])
        var[key] = dict(alpha=alphas.tolist(), CL=cl.tolist(), Cl_2d_10deg=runs[3]["Cl_2d"], z_over_h=runs[3]["z_over_h"], slope=slope_through_origin(alphas, cl))
        print(f"  variant {key}: slope {var[key]['slope']:.3f}/rad, CL(10 deg) = {cl[3]:.4f}")
    res["variants_AR1"] = var
    out["caseB"] = res


def case_C(out):
    print("Case C: fully ventilated strut")
    fv = read("harwood2016_CL_FV.csv"); llm = read("harwood2016_LLM_CL.csv")
    alphas = np.arange(5.0, 30.1, 5.0); res = {}
    combos = sorted({(row["AR_h"], row["Fn_h"]) for row in fv})
    for ar, fn in combos:
        t = time.time(); runs = [run_strut(a, fn, ar, regime="FV") for a in alphas]; cl = np.array([r["CL"] for r in runs])
        a, y = curve(fv, AR_h=ar, Fn_h=fn)
        r = dict(AR_h=ar, Fn_h=fn, alpha=alphas.tolist(), CL=cl.tolist(), regime=[x["regime"] for x in runs], active=[x["active"] for x in runs],
                 L_mid=[x["L"][len(x["L"])//2] for x in runs], data=dict(alpha=a.tolist(), CL=y.tolist()), cmp=compare(alphas, cl, a, y))
        if ar == 1.0 and abs(fn - 1.5) < 1e-9:
            a2, y2 = curve(llm, regime="FV"); r["llm"] = dict(alpha=a2.tolist(), CL=y2.tolist(), cmp=compare(alphas, cl, a2, y2))
        res[f"AR{ar}_Fn{fn}"] = r
        print(f"  AR_h = {ar}, Fn_h = {fn}: CL = {np.round(cl, 3).tolist()} regimes {r['regime']}; mean diff {100*r['cmp']['mean_rel_diff']:+.1f} %, max {r['cmp']['max_abs_diff']:.3f}  ({time.time() - t:.0f} s)")
    out["caseC"] = res


def case_D(out):
    print("Case D: cavity profile, alpha = 10 deg, Fn_h = 1.5, AR_h = 1")
    prof = read("harwood2016_cavity_profile.csv"); llm = read("harwood2016_LLM_profiles.csv")
    res = dict(experiment=dict(zip(("z_over_h", "L"), [x.tolist() for x in profile(prof, "L_over_c")])),
               llm={q: dict(zip(("z_over_h", "value"), [x.tolist() for x in profile(llm, q)])) for q in ("L_over_c", "a0_over_2pi", "alpha_eff_over_alpha", "Cl_section", "sigma_c")})
    for key, kw in (("N12", dict()), ("N24", dict(N=24))):
        wet = run_strut(10.0, 1.5, 1.0, **kw); vent = run_strut(10.0, 1.5, 1.0, regime="FV", T=4.0, **kw)
        z = np.array(vent["z_over_h"]); L = np.array(vent["L"]); clv = np.array(vent["Cl_2d"]); clw = np.array(wet["Cl_2d"])
        a_eff = clw/(2*np.pi)                                    # effective incidence of the wetted solution, as the code uses it for Psi
        a0 = np.where(a_eff > 1e-9, clv/a_eff, np.nan)/(2*np.pi)
        res[key] = dict(z_over_h=z.tolist(), L=L.tolist(), Cl_vent=clv.tolist(), Cl_wet=clw.tolist(), alpha_eff_over_alpha=(a_eff/np.deg2rad(10)).tolist(),
                        a0_over_2pi=a0.tolist(), sigma=vent["sigma"], CL_wet=wet["CL"], CL_vent=vent["CL"], regime=vent["regime"], active=vent["active"])
        # experiment interpolated at the sections
        ze, Le = profile(prof, "L_over_c"); zl, Ll = profile(llm, "L_over_c")
        res[key]["L_exp_at_sections"] = np.interp(z, ze, Le, left=np.nan, right=np.nan).tolist()
        res[key]["L_llm_at_sections"] = np.interp(z, zl, Ll, left=np.nan, right=np.nan).tolist()
        print(f"  {key}: CL wet {wet['CL']:.3f}, vent {vent['CL']:.3f}, regime {vent['regime']}, ventilated fraction {vent['active']:.2f}")
        print("   z/h    L code  L exp   L llm   Cl_v   Cl_w  a0/2pi")
        for j in range(len(z)):
            print(f"   {z[j]:.3f} {L[j]:7.3f} {res[key]['L_exp_at_sections'][j]:7.3f} {res[key]['L_llm_at_sections'][j]:7.3f} {clv[j]:6.3f} {clw[j]:6.3f} {a0[j]:7.3f}")
    out["caseD"] = res


# ------------------------------------------------------------------ figures
def figures(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
                         "axes.grid": True, "grid.color": "#d8d8d8", "grid.linewidth": 0.5, "legend.frameon": False,
                         "savefig.dpi": 170, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
    BLK, BLU, ORG, GRN, PUR, GRY = "#1a1a1a", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8c8c8c"
    FNCOL = {"1.0": PUR, "1.5": BLU, "2.0": GRN, "2.5": ORG, "3.0": GRY, "3.5": BLK, "4.5": "#E69F00"}
    # Fig. 1 wings
    A = out["caseA"]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, key, ttl in zip(axs, ("rect_AR1", "rect_AR3", "swept45_AR5"), ("rectangular, $AR$ = 1", "rectangular, $AR$ = 3", "45° swept, $AR$ = 5")):
        r = A[key]
        ax.plot(r["alpha"], r["CL"], "-o", color=BLK, ms=4, lw=1.4, label="this code")
        if "CL_tip_shedding" in r:
            ts = np.array(r["CL_tip_shedding"]); ok = np.abs(ts) < 2
            ax.plot(np.array(r["alpha"])[ok], ts[ok], "x", color=GRY, ms=5, label="code with tip shedding (diverges below 6°)")
        for name, lab, col, mk in (("rodriguez", "Rodriguez (1990) VLM", BLU, "s"), ("lamar", "experiment (Lamar, 1974)", ORG, "^"),
                                   ("tornado", "Bertin & Smith / Tornado VLM", BLU, "s"), ("weber", "Weber & Brebner (1958)", ORG, "^")):
            if name in r:
                ax.plot(r[name]["alpha"], r[name]["CL"], mk, color=col, ms=4, mfc="none", label=lab)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_xlim(0, None); ax.set_ylim(0, None); ax.legend(loc="upper left", fontsize=7)
    axs[0].set_ylabel(r"$C_L$", rotation=0, labelpad=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_wings.png")); plt.close(fig)
    # Fig. 2 wetted strut
    B = out["caseB"]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, ar in zip(axs, ("0.5", "1.0", "1.5")):
        r = B[f"AR{ar}"]
        for fn, d in sorted(r["data"].items()):
            ax.plot(d["alpha"], d["CL"], "o", color=FNCOL.get(fn, GRY), ms=4, mfc="none", label=fr"experiment, $Fn_h$ = {fn}")
        if "llm" in r:
            ax.plot(r["llm"]["alpha"], r["llm"]["CL"], "--", color=GRY, lw=1.2, label="lifting line, Harwood et al.")
        ax.plot(r["alpha"], r["CL"], "-", color=BLK, lw=1.6, label="this code")
        if ar == "1.0":
            v = B["variants_AR1"]["no_waterline_shedding"]; ax.plot(v["alpha"], v["CL"], ":", color=BLK, lw=1.4, label="code, no shedding at the waterline")
        ax.set_title(fr"$AR_h$ = {ar}", fontsize=9); ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_xlim(0, 20); ax.set_ylim(0, 0.7); ax.legend(loc="upper left", fontsize=6.5)
    axs[0].set_ylabel(r"$C_L$", rotation=0, labelpad=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_strut_FW.png")); plt.close(fig)
    # Fig. 3 ventilated strut
    Cc = out["caseC"]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, ar in zip(axs, (0.5, 1.0, 1.5)):
        for key, r in sorted(Cc.items(), key=lambda kv: kv[1]["Fn_h"]):
            if abs(r["AR_h"] - ar) > 1e-9:
                continue
            col = FNCOL.get(f"{r['Fn_h']:.1f}", GRY)
            ax.plot(r["data"]["alpha"], r["data"]["CL"], "o", color=col, ms=4, mfc="none")
            ax.plot(r["alpha"], r["CL"], "-", color=col, lw=1.4, label=fr"$Fn_h$ = {r['Fn_h']:.1f}")
        ax.set_title(fr"$AR_h$ = {ar}, fully ventilated", fontsize=9); ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_xlim(0, 32); ax.set_ylim(0, 1.1); ax.legend(loc="upper left", fontsize=7)
    axs[0].set_ylabel(r"$C_L$", rotation=0, labelpad=12)
    axs[2].text(31, 0.05, "lines: this code\ncircles: Harwood et al. (2016)", ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_strut_FV.png")); plt.close(fig)
    # Fig. 4 cavity profile and sectional quantities
    D = out["caseD"]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    ax = axs[0]
    ax.plot(D["experiment"]["L"], D["experiment"]["z_over_h"], "o", color=ORG, ms=4, mfc="none", label="photograph, Harwood et al.")
    ax.plot(D["llm"]["L_over_c"]["value"], D["llm"]["L_over_c"]["z_over_h"], "--", color=GRY, lw=1.2, label="lifting line, Harwood et al.")
    ax.plot(np.minimum(D["N12"]["L"], 2.5), D["N12"]["z_over_h"], "-s", color=BLK, ms=4, lw=1.4, label="this code, 12 sections")
    ax.plot(np.minimum(D["N24"]["L"], 2.5), D["N24"]["z_over_h"], "-", color=BLU, lw=1.0, label="this code, 24 sections")
    ax.set_xlim(0, 2.5); ax.set_ylim(1, 0); ax.set_xlabel(r"$L/c$ (code values above 2.5 clipped)"); ax.set_ylabel(r"$z/h$", rotation=0, labelpad=10); ax.legend(loc="lower right", fontsize=6.5); ax.set_title("cavity length", fontsize=9)
    ax = axs[1]
    ax.plot(D["llm"]["Cl_section"]["value"], D["llm"]["Cl_section"]["z_over_h"], "--", color=GRY, lw=1.2, label="lifting line, ventilated")
    ax.plot(D["N12"]["Cl_vent"], D["N12"]["z_over_h"], "-s", color=BLK, ms=4, lw=1.4, label="this code, ventilated")
    ax.plot(D["N12"]["Cl_wet"], D["N12"]["z_over_h"], ":", color=BLK, lw=1.2, label="this code, wetted")
    ax.set_xlim(0, 0.5); ax.set_ylim(1, 0); ax.set_xlabel(r"$C_l$"); ax.legend(loc="lower right", fontsize=6.5); ax.set_title("sectional lift", fontsize=9)
    ax = axs[2]
    ax.plot(D["llm"]["alpha_eff_over_alpha"]["value"], D["llm"]["alpha_eff_over_alpha"]["z_over_h"], "--", color=GRY, lw=1.2, label=r"lifting line, $\alpha_\mathrm{eff}/\alpha$")
    ax.plot(D["N12"]["alpha_eff_over_alpha"], D["N12"]["z_over_h"], "-s", color=BLK, ms=4, lw=1.4, label=r"this code, $\alpha_\mathrm{eff}/\alpha$")
    ax.plot(D["llm"]["a0_over_2pi"]["value"], D["llm"]["a0_over_2pi"]["z_over_h"], "--", color=ORG, lw=1.2, label=r"lifting line, $a_0/2\pi$")
    ax.plot(D["N12"]["a0_over_2pi"], D["N12"]["z_over_h"], "-^", color=ORG, ms=4, lw=1.4, label=r"this code, $a_0/2\pi$")
    ax.set_xlim(0, 1.4); ax.set_ylim(1, 0); ax.legend(loc="upper right", fontsize=6.5); ax.set_title("effective incidence and lift slope", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_profile.png")); plt.close(fig)
    print("figures written")


def main():
    path = os.path.join(HERE, "validate.json")
    if "--plot" in sys.argv:
        figures(json.load(open(path))); return
    only = [a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")]          # --only=A,C reruns those cases, keeps the others
    only = only[0] if only else ["A", "B", "C", "D"]
    out = json.load(open(path)) if (os.path.exists(path) and len(only) < 4) else {}
    out.update(code_commit=git_head(), N_strut=N_STRUT, M_strut=M_STRUT, chord_m=C)
    t0 = time.time()
    for letter, case in zip("ABCD", (case_A, case_B, case_C, case_D)):
        if letter in only:
            case(out); json.dump(clean(out), open(path, "w"), indent=1)
    print(f"total {time.time() - t0:.0f} s")
    figures(out)


if __name__ == "__main__":
    main()
