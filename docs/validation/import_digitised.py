"""
Convert the raw digitised curves of docs/validation/data/raw/ (L. Orain, repository
vortex_lattice_method-cavity_flow, branch wake_rel) into the template format of
docs/validation/data/TEMPLATE.csv, one CSV per source and quantity. See raw/README.md for the
meaning of each raw file.

Run from the repository root:  python docs/validation/import_digitised.py
"""
import os, re, csv, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw"); OUT = os.path.join(HERE, "data")
COLS = ["source", "figure", "quantity", "regime", "path", "alpha_deg", "Fn_h", "AR_h", "chord_m", "section", "value", "uncertainty", "notes"]
HARWOOD = "Harwood Young Ceccio 2016 JFM 800"; HARWOOD_LLM = "Harwood Young Ceccio 2016 JFM 800, lifting-line model"
DIGIT = "digitised by L. Orain (2026), wake_rel"
CHORD = 0.2794; SECTION = "modified NACA 0009"


def raw(name):
    rows = []
    for line in open(os.path.join(RAW, name), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        a, b = line.split(";")
        rows.append((float(a.replace(",", ".")), float(b.replace(",", "."))))
    return np.array(rows)


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    print(f"{name}: {len(rows)} rows")


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    # 1. Harwood et al. (2016): lift curves of the strut, fully wetted and fully ventilated
    for tag, regime in (("w", "FW"), ("v", "FV")):
        rows = []
        for f in sorted(glob.glob(os.path.join(RAW, f"cl_ar*_fn*_{tag}.csv"))):
            m = re.match(r"cl_ar(\d+)_fn(\d+)_[wv]\.csv", os.path.basename(f))
            ar = int(m.group(1))/10 if len(m.group(1)) > 1 else float(m.group(1))     # ar05 -> 0.5, ar1 -> 1, ar15 -> 1.5
            fn = int(m.group(2))/10 if len(m.group(2)) > 1 else float(m.group(2))     # fn15 -> 1.5, fn3 -> 3
            for a, cl in raw(os.path.basename(f))[np.argsort(raw(os.path.basename(f))[:, 0])]:
                rows.append(dict(source=HARWOOD, figure=DIGIT, quantity="CL", regime=regime, path="steady", alpha_deg=fmt(a, 2),
                                 Fn_h=fmt(fn, 2), AR_h=fmt(ar, 2), chord_m=CHORD, section=SECTION, value=fmt(cl), uncertainty="0.01",
                                 notes="CL on the immersed area h c"))
        write(f"harwood2016_CL_{regime}.csv", rows)
    # 2. Harwood et al. (2016): cavity profile from the photograph
    rows = []
    for L, z in raw("cavity.csv")[np.argsort(raw("cavity.csv")[:, 1])]:
        rows.append(dict(source=HARWOOD, figure=DIGIT + ", photograph", quantity="L_over_c", regime="FV", path="steady", alpha_deg="10.0",
                         Fn_h="1.50", AR_h="1.00", chord_m=CHORD, section=SECTION, value=fmt(L, 3), uncertainty="0.05",
                         notes=f"z_over_h={z:.3f}; curve clipped at L/c = 2"))
    write("harwood2016_cavity_profile.csv", rows)
    # 3. Harwood et al. (2016): their lifting-line model, sectional profiles at the same condition
    rows = []
    for fname, q in (("Lc.csv", "L_over_c"), ("a0.csv", "a0_over_2pi"), ("alpha.csv", "alpha_eff_over_alpha"), ("cl.csv", "Cl_section"), ("sigma.csv", "sigma_c")):
        d = raw(fname); d = d[np.argsort(d[:, 1])]
        for x, z in d:
            rows.append(dict(source=HARWOOD_LLM, figure=DIGIT, quantity=q, regime="FV", path="steady", alpha_deg="10.0", Fn_h="1.50",
                             AR_h="1.00", chord_m=CHORD, section=SECTION, value=fmt(x), uncertainty="", notes=f"z_over_h={z:.3f}"))
    write("harwood2016_LLM_profiles.csv", rows)
    # 4. Harwood et al. (2016): their lifting-line model, lift curves (Froude number not recorded)
    rows = []
    for fname, regime in (("lifting_FW.csv", "FW"), ("lifting_FV.csv", "FV")):
        d = raw(fname); d = d[np.argsort(d[:, 0])]
        for a, cl in d:
            rows.append(dict(source=HARWOOD_LLM, figure=DIGIT, quantity="CL", regime=regime, path="steady", alpha_deg=fmt(a, 2), Fn_h="1.50",
                             AR_h="1.00", chord_m=CHORD, section=SECTION, value=fmt(cl), uncertainty="0.01",
                             notes="Fn_h not recorded in the digitisation, 1.5 assumed (the profiles of the same set)"))
    write("harwood2016_LLM_CL.csv", rows)
    # 5. Rectangular wings without free surface: Rodriguez (1990) VLM and the experiments reported by Lamar (1974)
    for src, pre, out in (("Rodriguez 1990 MSc thesis VPI, free-wake VLM", "rodriguez_vlm", "rodriguez1990_VLM_CL.csv"),
                          ("Lamar 1974 NASA TN D-7921, experiment", "lamar_exp", "lamar1974_exp_CL.csv")):
        rows = []
        for ar in (1, 3):
            d = raw(f"{pre}_{ar}.csv"); d = d[np.argsort(d[:, 0])]
            for a, cl in d:
                rows.append(dict(source=src, figure=DIGIT, quantity="CL", regime="FW", path="steady", alpha_deg=fmt(a, 2), Fn_h="",
                                 AR_h=fmt(ar, 2), chord_m="", section="flat rectangular wing", value=fmt(cl), uncertainty="0.01",
                                 notes="no free surface; AR_h is the wing aspect ratio"))
        write(out, rows)
    # 6. Swept wing: Bertin & Smith / Tornado VLM and Weber & Brebner (1958)
    for src, fname, out in (("Bertin Smith 1998 analytic VLM and Tornado (Melin 2000), straight wake", "Tornado_vlm.csv", "bertin1998_tornado_VLM_CL.csv"),
                            ("Weber Brebner 1958 ARC R&M 2882, experiment", "experiment_weber.csv", "weber1958_exp_CL.csv")):
        d = raw(fname); d = d[np.argsort(d[:, 0])]
        rows = [dict(source=src, figure=DIGIT, quantity="CL", regime="FW", path="steady", alpha_deg=fmt(a, 2), Fn_h="", AR_h="5.00", chord_m="",
                     section="rectangular wing, 45 deg sweep", value=fmt(cl), uncertainty="0.01", notes="no free surface; sweep_deg=45; AR_h is the wing aspect ratio")
                for a, cl in d]
        write(out, rows)


if __name__ == "__main__":
    main()
