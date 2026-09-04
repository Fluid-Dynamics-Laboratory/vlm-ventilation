"""Figures of the methods report from the verification and validation results already in the repository."""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 10,
                     "axes.grid": True, "grid.color": "#d8d8d8", "grid.linewidth": 0.5, "legend.frameon": False,
                     "savefig.dpi": 170, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
BLK, BLU, ORG, GRN = "#1a1a1a", "#0072B2", "#D55E00", "#009E73"

# Figure A: lattice cavity against Acosta
rows = json.load(open(os.path.join(ROOT, "docs", "cavity", "verify_acosta.json")))
L = np.linspace(0.005, 0.5, 100)
acosta = np.pi*(1 + 1/np.sqrt(1 - L))/(2*np.pi)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot(L, acosta, color=BLK, lw=1.4, label="Acosta (1955)")
lat = [r for r in rows if 0 < r["L"] <= 0.5]
ax.plot([r["L"] for r in lat], [r["a0_ratio"] for r in lat], "o", color=BLU, ms=6, label="lattice")
ax.set_xlabel(r"$L/c$"); ax.set_ylabel(r"$a_0/2\pi$", rotation=0, labelpad=14)
ax.set_xlim(0, 0.5); ax.set_ylim(1.0, 1.3); ax.set_xticks([0, 0.25, 0.5]); ax.set_yticks([1.0, 1.1, 1.2, 1.3])
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "figA_acosta.png")); plt.close(fig)

# Figure B: hysteresis loop on the strut
val = json.load(open(os.path.join(ROOT, "docs", "cavity", "validate.json")))
up = val["caseC"]["up"]; down = val["caseC"]["down"]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot([r["alpha"] for r in up], [r["CL"] for r in up], "-o", color=BLK, ms=4, lw=1.4, label="incidence increasing")
ax.plot([r["alpha"] for r in down], [r["CL"] for r in down], "--s", color=ORG, ms=4, lw=1.4, label="incidence decreasing")
ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$C_L$", rotation=0, labelpad=12)
ax.set_xlim(0, 20); ax.set_ylim(0, 0.5); ax.set_xticks([0, 5, 10, 15, 20]); ax.set_yticks([0, 0.25, 0.5])
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "figB_hysteresis.png")); plt.close(fig)

# Figure C: inception incidence against Froude number, two gates
inc = json.load(open(os.path.join(ROOT, "docs", "inception", "validate.json")))
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot([r["Fn_h"] for r in inc["case1"]], [r["alpha_incep"] for r in inc["case1"]], "-o", color=BLK, ms=5, lw=1.4, label="effective gate")
ax.plot([r["Fn_h"] for r in inc["case1c"]], [r["alpha_incep"] for r in inc["case1c"]], "--s", color=BLU, ms=5, lw=1.4, label="geometric gate")
ax.axhspan(20, 26, color=ORG, alpha=0.12, lw=0); ax.text(1.55, 20.6, "Aguiar Ferreira et al. (2026), steady", fontsize=8, color=ORG)
ax.axhline(14.5, color=GRN, ls=":", lw=1.2); ax.text(1.55, 15.0, "Harwood et al. (2016)", fontsize=8, color=GRN)
ax.set_xlabel(r"$Fn_h$"); ax.set_ylabel(r"$\alpha_\mathrm{incep}$ [deg]", rotation=90)
ax.set_xlim(1, 4); ax.set_ylim(0, 30); ax.set_xticks([1, 2, 3, 4]); ax.set_yticks([0, 10, 20, 30])
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "figC_inception.png")); plt.close(fig)
print("figures written")
