#!/usr/bin/env python3
"""Dual-axis time-history plot of Ma_max and U_bulk vs FTT."""

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ── style ──────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         12,
    "axes.labelsize":    14,
    "axes.titlesize":    15,
    "legend.fontsize":   11,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  5,
    "ytick.major.size":  5,
    "xtick.minor.size":  3,
    "ytick.minor.size":  3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth":    1.0,
    "lines.linewidth":   1.2,
    "savefig.dpi":       300,
})

# ── load data ──────────────────────────────────────────────
data = np.loadtxt("../Ustar_Force_record.dat")
ftt    = data[:, 0]
u_star = data[:, 1]   # U_bulk / U_ref
ma_max = data[:, 3]

# ── colours ────────────────────────────────────────────────
c_ma = "#D62728"   # red family
c_ub = "#1F77B4"   # blue family

# ── figure ─────────────────────────────────────────────────
fig, ax_ma = plt.subplots(figsize=(7, 3.8))
ax_ub = ax_ma.twinx()

ln1 = ax_ma.plot(ftt, ma_max, color=c_ma, lw=1.2, label=r"$\mathrm{Ma}_{\max}$")
ln2 = ax_ub.plot(ftt, u_star, color=c_ub, lw=1.2, label=r"$U_b\,/\,U_{\mathrm{ref}}$")

# ── axes labels ────────────────────────────────────────────
ax_ma.set_xlabel(r"$\mathrm{FTT}$")
ax_ma.set_ylabel(r"$\mathrm{Ma}_{\max}$", color=c_ma)
ax_ub.set_ylabel(r"$U_b\,/\,U_{\mathrm{ref}}$", color=c_ub)

# ── tick colours ───────────────────────────────────────────
ax_ma.tick_params(axis="y", colors=c_ma)
ax_ub.tick_params(axis="y", colors=c_ub)
ax_ma.spines["left"].set_color(c_ma)
ax_ma.spines["right"].set_color(c_ub)
ax_ub.spines["left"].set_color(c_ma)
ax_ub.spines["right"].set_color(c_ub)

# ── independent y-ranges (add ~5% padding) ─────────────────
def padded_range(v, pad=0.08):
    lo, hi = v.min(), v.max()
    margin = (hi - lo) * pad if hi > lo else 0.01
    return lo - margin, hi + margin

ax_ma.set_ylim(*padded_range(ma_max))
ax_ub.set_ylim(*padded_range(u_star))

# ── combined legend ────────────────────────────────────────
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax_ma.legend(lns, labs, loc="upper right", frameon=True,
             edgecolor="0.7", fancybox=False)

fig.tight_layout()
fig.savefig("monitor_MaMa_Ubulk.png")
fig.savefig("monitor_MaMa_Ubulk.pdf")
print("Saved: monitor_MaMa_Ubulk.png / .pdf")
