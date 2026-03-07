#!/usr/bin/env python3
"""
Monitor Convergence Plot — Ub/Uref, Ma_max, rho, RS check, TKE check
=====================================================================
Reads Ustar_Force_record.dat (4/7/9-col formats).

Panel 1: Ma_max + Ub/Uref
Panel 2: rho_crest + rho_L1  (9-col only)
Panel 3: RS + TKE            (7+ col only)

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

# Handle mixed column counts (4-col legacy → 7-col → 9-col mid-file)
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

has_rs  = (ncols >= 7)
has_rho = (ncols >= 9)

if has_rs:
    accu_count  = data[:, 4]
    uu_RS_check = data[:, 5]
    k_check     = data[:, 6]
if has_rho:
    rho_crest = data[:, 7]
    rho_L1    = data[:, 8]
    print(f"[INFO] 9-col format: {len(FTT)} rows, Re={Re}")
elif has_rs:
    print(f"[INFO] 7-col format: {len(FTT)} rows, Re={Re}")
else:
    print(f"[INFO] Legacy 4-col format: {len(FTT)} rows — only Ub/Uref and Ma_max")

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
    ax.text(0.98, 0.78, txt, transform=ax.transAxes, ha='right', va='top',
            color=clr, fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# ── Helper: draw FTT accumulation start marker ───────────
def mark_ftt_start(ax, ftt_val=20.0):
    ax.axvline(ftt_val, color='black', ls='-', lw=1.2, alpha=0.8)
    ax.text(ftt_val, 0.92, f' FTT={ftt_val:.0f}\n accumulate start',
            fontsize=9, color='black', fontweight='bold', va='top', ha='left',
            transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=2, boxstyle='round,pad=0.3'))

# ── Determine number of panels ───────────────────────────
ftt_stats_start = 20.0  # accumulation start

n_rows = 1  # always have panel 1 (Ma + Ub)
if has_rho:
    n_rows += 1  # panel 2 (rho)
if has_rs:
    n_rows += 1  # panel 3 (RS + TKE)

fig, all_axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
if n_rows == 1:
    all_axes = [all_axes]

panel_idx = 0

# === Panel 1: Ub/Uref + Ma_max ===
ax1 = all_axes[panel_idx]
color_ub = '#1F77B4'
color_ma = '#D62728'

ln1 = ax1.plot(FTT, Ub_Uref, color=color_ub, lw=1.0, label=r'$U_b / U_{ref}$')
ax1.axhline(1.0, color=color_ub, ls='--', lw=0.8, alpha=0.5)
ax1.set_ylabel(r"$U_b \,/\, U_{ref}$", color=color_ub)
ax1.tick_params(axis='y', labelcolor=color_ub)
ax1.grid(True, alpha=0.3)

ax1b = ax1.twinx()
ln2 = ax1b.plot(FTT, Ma_max, color=color_ma, lw=1.0, label=r'$Ma_{\max}$')
ax1b.set_ylabel(r"$Ma_{\max}$", color=color_ma)
ax1b.tick_params(axis='y', labelcolor=color_ma)

# --- Ma = 0.2 danger zone ---
# Ensure y-axis shows at least up to 0.25 so the danger zone is visible
ma_ymin, ma_ymax = ax1b.get_ylim()
ma_upper = max(ma_ymax, 0.25)
ax1b.set_ylim(ma_ymin, ma_upper)

# FREEZE zone (Ma >= 0.20): red shading
ax1b.axhspan(0.20, ma_upper, alpha=0.10, color='red', zorder=0)
ax1b.axhline(0.20, color='red', ls='--', lw=1.2, alpha=0.8)

# CAUTION zone (0.18 <= Ma < 0.20): yellow shading
ax1b.axhspan(0.18, 0.20, alpha=0.10, color='#FFA500', zorder=0)
ax1b.axhline(0.18, color='#FFA500', ls=':', lw=0.8, alpha=0.6)

# Annotations on right edge
ax1b.text(FTT[-1], 0.203, r' $Ma \geq 0.20$: FREEZE',
          fontsize=8, color='red', fontweight='bold', va='bottom', ha='right',
          bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', pad=1.5, boxstyle='round,pad=0.2'))
ax1b.text(FTT[-1], 0.183, r' CAUTION',
          fontsize=7, color='#CC7000', va='bottom', ha='right',
          bbox=dict(facecolor='white', alpha=0.7, edgecolor='#FFA500', pad=1, boxstyle='round,pad=0.2'))

mark_ftt_start(ax1, ftt_stats_start)
lns = ln1 + ln2
ax1.legend(lns, [l.get_label() for l in lns], loc='upper right', fontsize=9)
ax1.set_title("Bulk Velocity & Mach Number", fontsize=12)
panel_idx += 1

# === Panel 2: rho_crest + rho_L1 (only if 9-col data) ===
if has_rho:
    ax2 = all_axes[panel_idx]
    color_rho_c = '#E67E22'   # orange
    color_rho_l = '#8E44AD'   # purple

    mask_rho = np.isfinite(rho_crest) & np.isfinite(rho_L1)

    ln_rc = ax2.plot(FTT[mask_rho], rho_crest[mask_rho], color=color_rho_c, lw=1.0,
                     label=r'$\rho_{\mathrm{crest}}$')
    ax2.axhline(1.0, color=color_rho_c, ls='--', lw=0.8, alpha=0.5)
    ax2.set_ylabel(r"$\rho_{\mathrm{crest}}$", color=color_rho_c)
    ax2.tick_params(axis='y', labelcolor=color_rho_c)
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ln_rl = ax2b.plot(FTT[mask_rho], rho_L1[mask_rho], color=color_rho_l, lw=1.0,
                      label=r'$\|\rho - 1\|_{L_1}$')
    ax2b.set_ylabel(r"$\sum |\rho_i - 1|$", color=color_rho_l)
    ax2b.tick_params(axis='y', labelcolor=color_rho_l)

    mark_ftt_start(ax2, ftt_stats_start)
    lns_rho = ln_rc + ln_rl
    ax2.legend(lns_rho, [l.get_label() for l in lns_rho], loc='upper right', fontsize=9)
    ax2.set_title("Density Monitor (hill crest point & global L1 norm)", fontsize=12)
    panel_idx += 1

# === Panel 3: RS + TKE (only if 7+ col data) ===
if has_rs:
    mask_valid = np.isfinite(uu_RS_check) & np.isfinite(k_check)
    mask_stats = mask_valid & (FTT >= ftt_stats_start) & (uu_RS_check > 0)

    ax3 = all_axes[panel_idx]
    color_uu = '#006400'
    color_k  = '#4B0082'

    if np.any(mask_stats):
        ln3 = ax3.plot(FTT[mask_stats], uu_RS_check[mask_stats], color=color_uu, lw=1.0,
                       label=r"$\langle u'u' \rangle / U_{ref}^2$")
        convergence_analysis(ax3, FTT[mask_stats], uu_RS_check[mask_stats], 'uu_RS')
    else:
        ln3 = ax3.plot([], [], color=color_uu, label=r"$\langle u'u' \rangle / U_{ref}^2$")
    ax3.set_ylabel(r"$\langle u'u' \rangle \,/\, U_{ref}^2$", color=color_uu)
    ax3.tick_params(axis='y', labelcolor=color_uu)
    ax3.grid(True, alpha=0.3)

    ax3b = ax3.twinx()
    if np.any(mask_stats):
        ln4 = ax3b.plot(FTT[mask_stats], k_check[mask_stats], color=color_k, lw=1.0,
                        label=r"$k / U_{ref}^2$")
        convergence_analysis(ax3b, FTT[mask_stats], k_check[mask_stats], 'k')
    else:
        ln4 = ax3b.plot([], [], color=color_k, label=r"$k / U_{ref}^2$")
    ax3b.set_ylabel(r"$k \,/\, U_{ref}^2$", color=color_k)
    ax3b.tick_params(axis='y', labelcolor=color_k)

    mark_ftt_start(ax3, ftt_stats_start)
    lns2 = ln3 + ln4
    ax3.legend(lns2, [l.get_label() for l in lns2], loc='upper right', fontsize=9)
    ax3.set_title(r"RS & TKE Convergence at check point ($x/h\approx 2,\, y/h\approx 1$)", fontsize=12)
    panel_idx += 1

all_axes[-1].set_xlabel(r"FTT (Flow-Through Time)")
fig.suptitle(f"Periodic Hill Flow Monitor — Re = {Re}", fontsize=15)
fig.tight_layout()

# ── Explanation note (2 lines) ────────────────
note = (
    r"$U_{ref}$ = target reference velocity = 0.0583 (fixed, equivalent to $U_b$ in benchmark literature)"
    "\n"
    r"$U_b$ = instantaneous cross-section-averaged velocity (fluctuates each step, driven toward $U_{ref}$ by PD controller)"
)
fig.text(0.5, -0.01, note, ha='center', va='top', fontsize=9,
         style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFDD', edgecolor='0.7', alpha=0.9))

out_pdf = os.path.join(SCRIPT_DIR, f"monitor_convergence_Re{Re}.pdf")
out_png = os.path.join(SCRIPT_DIR, f"monitor_convergence_Re{Re}.png")
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"[OK] Saved: {os.path.basename(out_png)}")
plt.close(fig)
