#!/usr/bin/env python3
"""
Monitor Convergence Plot — Ub/Uref, Ma_max, RS check, TKE check
================================================================
Reads Ustar_Force_record.dat (4-col legacy or 7-col new format).

Usage:
  python3 4.Ma_U_Time.py               # default Re=700
  python3 4.Ma_U_Time.py --Re 5600     # specify Re

Output:
  monitor_convergence_Re{N}.pdf / .png
"""

import os, sys, argparse
import numpy as np

# Windows console UTF-8 support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import matplotlib as mpl
if not os.environ.get('DISPLAY') and sys.platform != 'win32':
    mpl.use('Agg')
import matplotlib.pyplot as plt

# ── Re argument ───────────────────────────────────────────
parser = argparse.ArgumentParser(description="Monitor convergence plot")
parser.add_argument('--Re', type=int, default=700, help='Reynolds number (default 700)')
args, _ = parser.parse_known_args()
Re = args.Re

# ── style ─────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         11,
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "legend.fontsize":   10,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  5,
    "ytick.major.size":  5,
    "xtick.minor.size":  3,
    "ytick.minor.size":  3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.0,
    "savefig.dpi":       300,
})

# ── load data ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
filepath = os.path.join(SCRIPT_DIR, "..", "Ustar_Force_record.dat")
if not os.path.isfile(filepath):
    sys.exit(f"[ERROR] File not found: {filepath}")

# Handle mixed column counts (4-col legacy → 7-col new mid-file)
rows = []
with open(filepath, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        vals = line.split()
        try:
            rows.append([float(v) for v in vals])
        except ValueError:
            continue
max_cols = max(len(r) for r in rows)
# Pad short rows with NaN
data = np.array([r + [np.nan] * (max_cols - len(r)) for r in rows])
if data.ndim == 1:
    data = data.reshape(1, -1)
ncols = max_cols

FTT     = data[:, 0]
Ub_Uref = data[:, 1]
Force   = data[:, 2]
Ma_max  = data[:, 3]

has_rs = (ncols >= 7)
if has_rs:
    accu_count  = data[:, 4]
    uu_RS_check = data[:, 5]
    k_check     = data[:, 6]
    print(f"[INFO] New format (7 cols): {len(FTT)} rows, Re={Re}")
else:
    print(f"[INFO] Legacy format (4 cols): {len(FTT)} rows — only Ub/Uref and Ma_max")

# ── convergence analysis helper ───────────────────────────
def convergence_analysis(ax, ftt, values, label_name):
    """Add convergence band + CV text for the last 10 FTT."""
    if len(ftt) < 2:
        return
    ftt_end = ftt[-1]
    mask = (ftt >= ftt_end - 10) & (values > 0)
    last10 = values[mask]
    if len(last10) < 10:
        return

    mean_val = np.mean(last10)
    std_val  = np.std(last10)
    cv = std_val / abs(mean_val) * 100 if abs(mean_val) > 1e-30 else 0.0

    # Mean line + ±1σ band
    ax.axhline(mean_val, color='blue', ls='--', alpha=0.4, lw=0.8,
               label=f'mean = {mean_val:.4e}')
    ftt_band = ftt[mask]
    ax.fill_between(ftt_band, mean_val - std_val, mean_val + std_val,
                    alpha=0.12, color='blue')

    # Convergence text
    if cv < 1.0:
        txt, clr = f'CONVERGED (CV={cv:.1f}%)', 'green'
    elif cv < 5.0:
        txt, clr = f'NEAR CONVERGED (CV={cv:.1f}%)', 'orange'
    else:
        txt, clr = f'NOT CONVERGED (CV={cv:.1f}%)', 'red'
    ax.text(0.98, 0.92, txt, transform=ax.transAxes, ha='right', va='top',
            color=clr, fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# ── figure ────────────────────────────────────────────────
n_rows = 4 if has_rs else 2
fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.0 * n_rows), sharex=True)
if n_rows == 2:
    axes = list(axes)

# Row 1: Ub/Uref
ax1 = axes[0]
ax1.plot(FTT, Ub_Uref, color='#1F77B4', lw=1.0)
ax1.axhline(1.0, color='#D62728', ls='--', lw=0.8, alpha=0.7, label=r'$U_b/U_{ref}=1$')
ax1.set_ylabel(r"$U_b \,/\, U_{ref}$")
ax1.set_title(r"Bulk Velocity Convergence")
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Row 2: Ma_max
ax2 = axes[1]
ax2.plot(FTT, Ma_max, color='#D62728', lw=1.0)
ax2.axhline(0.3, color='gray', ls='--', lw=0.8, alpha=0.7, label=r'$Ma = 0.3$')
ax2.set_ylabel(r"$Ma_{\max}$")
ax2.set_title(r"Maximum Mach Number")
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

if has_rs:
    # FTT_STATS_START line
    ftt_stats_start = 20.0  # from variables.h
    mask_stats = (FTT >= ftt_stats_start) & (uu_RS_check > 0)

    # Row 3: uu_RS_check
    ax3 = axes[2]
    if np.any(mask_stats):
        ax3.plot(FTT[mask_stats], uu_RS_check[mask_stats], color='#2CA02C', lw=1.0)
        convergence_analysis(ax3, FTT[mask_stats], uu_RS_check[mask_stats], 'uu_RS')
    ax3.axvline(ftt_stats_start, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax3.text(ftt_stats_start + 0.3, ax3.get_ylim()[1] * 0.95 if ax3.get_ylim()[1] > 0 else 0,
             f'FTT={ftt_stats_start:.0f}', fontsize=8, color='gray', va='top')
    ax3.set_ylabel(r"$\langle u'u' \rangle \,/\, U_b^2$")
    ax3.set_title(r"Streamwise Reynolds Stress at check point ($x/h=2,\, y/h=1$)")
    ax3.grid(True, alpha=0.3)

    # Row 4: k_check
    ax4 = axes[3]
    if np.any(mask_stats):
        ax4.plot(FTT[mask_stats], k_check[mask_stats], color='#9467BD', lw=1.0)
        convergence_analysis(ax4, FTT[mask_stats], k_check[mask_stats], 'k')
    ax4.axvline(ftt_stats_start, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax4.text(ftt_stats_start + 0.3, ax4.get_ylim()[1] * 0.95 if ax4.get_ylim()[1] > 0 else 0,
             f'FTT={ftt_stats_start:.0f}', fontsize=8, color='gray', va='top')
    ax4.set_ylabel(r"$k \,/\, U_b^2$")
    ax4.set_title(r"TKE at check point ($x/h=2,\, y/h=1$)")
    ax4.grid(True, alpha=0.3)

# Shared x-axis label
axes[-1].set_xlabel(r"FTT (Flow-Through Time)")

fig.suptitle(f"Periodic Hill Flow Monitor — Re = {Re}", fontsize=15, y=1.01)
fig.tight_layout()

# ── save ──────────────────────────────────────────────────
out_pdf = os.path.join(SCRIPT_DIR, f"monitor_convergence_Re{Re}.pdf")
out_png = os.path.join(SCRIPT_DIR, f"monitor_convergence_Re{Re}.png")
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"[OK] Saved: {os.path.basename(out_pdf)}")
print(f"[OK] Saved: {os.path.basename(out_png)}")
plt.close(fig)
