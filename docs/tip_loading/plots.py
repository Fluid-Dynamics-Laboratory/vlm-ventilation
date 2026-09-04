"""Figures of the tip-loading study. Run from the repository root after study.py."""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "study.json")))
T = json.load(open(os.path.join(HERE, "tsweep.json")))
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"stix","font.size":10,
                     "axes.grid":True,"grid.color":"#d8d8d8","grid.linewidth":0.5,"legend.frameon":False,
                     "savefig.dpi":170,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
BLK, BLU, ORG, GRN = "#1a1a1a", "#0072B2", "#D55E00", "#009E73"
def pick(name, **kw): return [d for d in D[name] if all(d.get(k)==v for k,v in kw.items())]
def panel_titles(): return ["(a) original formulation", "(b) revised formulation"]

# Figure 1: sectional lift for three values of the regularisation parameter
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
sty = [dict(color=BLK, ls="-"), dict(color=BLU, ls="--"), dict(color=ORG, ls="-.")]
for ax, name, ttl in zip(axs, ["original","revised"], panel_titles()):
    for d, st in zip(pick(name, case="ratio"), sty):
        lab = rf"$d/b_\mathrm{{min}}={d['ratio']}$" if name=="original" else rf"$r_\mathrm{{c}}/c={d['ratio']}$"
        ax.plot(d["y"], d["Cl"], label=lab, lw=1.4, **st)
    ax.set_xlabel(r"$y/b$"); ax.set_xlim(-0.5, 0.5); ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5]); ax.set_title(ttl, fontsize=10)
    ax.legend(loc="lower center", fontsize=9)
axs[0].set_ylabel(r"$C_l$", rotation=0, labelpad=12); axs[0].set_ylim(0, 0.5); axs[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_regularisation.png")); plt.close(fig)

# Figure 2: spanwise refinement (outer half of the span)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
cols = {8:ORG, 15:BLK, 25:BLU, 40:GRN}; lss = {8:":", 15:"-", 25:"--", 40:"-."}
for ax, name, ttl in zip(axs, ["original","revised"], panel_titles()):
    for d in pick(name, case="N"):
        y = np.array(d["y"]); c = np.array(d["Cl"]); k = y > 0
        ax.plot(y[k], c[k], color=cols[d["N"]], ls=lss[d["N"]], lw=1.4, label=rf"$N={2*d['N']}$")
    ax.set_xlabel(r"$y/b$"); ax.set_xlim(0, 0.5); ax.set_xticks([0, 0.25, 0.5]); ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel(r"$C_l$", rotation=0, labelpad=12); axs[0].set_ylim(0, 0.5); axs[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
axs[1].legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_refinement.png")); plt.close(fig)

# Figure 3: time-step sensitivity of the tip section
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, name, ttl in zip(axs, ["original","revised"], panel_titles()):
    for N, col, ls in [(15, BLK, "-"), (25, BLU, "--")]:
        rows = sorted([d for d in D[name] if d["N"]==N and d["shed"]=="both" and d["space"] and d["case"] in ("N","dt")], key=lambda d: d["dt"])
        ax.plot([d["dt"] for d in rows], [d["Cl"][0] for d in rows], color=col, ls=ls, marker="o", ms=4, lw=1.4, label=rf"$N={2*N}$")
    ax.set_xlabel(r"$u_\infty\,\Delta t/c$"); ax.set_xlim(0, 0.12); ax.set_xticks([0, 0.05, 0.1]); ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel(r"$C_{l,\mathrm{tip}}$", rotation=0, labelpad=18); axs[0].set_ylim(-0.3, 0.6); axs[0].set_yticks([-0.3, 0, 0.3, 0.6])
axs[1].legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_timestep.png")); plt.close(fig)

# Figure 4: lift coefficient versus simulation time
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, name, ttl in zip(axs, ["original","revised"], panel_titles()):
    for N, col, ls in [(15, BLK, "-"), (40, GRN, "-.")]:
        rows = sorted([d for d in T[name] if d["N"]==N], key=lambda d: d["T"])
        ax.plot([d["T"] for d in rows], [d["CL"] for d in rows], color=col, ls=ls, marker="o", ms=4, lw=1.4, label=rf"$N={2*N}$")
    ax.set_xlabel(r"$u_\infty t/c$"); ax.set_xlim(0, 8); ax.set_xticks([0, 2, 4, 6, 8]); ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel(r"$C_L$", rotation=0, labelpad=12); axs[0].set_ylim(0.3, 0.4); axs[0].set_yticks([0.3, 0.35, 0.4])
axs[1].legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_time.png")); plt.close(fig)
print("figures written to", HERE)
