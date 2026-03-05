#!/usr/bin/env python3
"""
Periodic Hill Benchmark Comparison: GILBM vs Breuer et al. (2009) DNS Re=700
=============================================================================
用法 (三種方式):
  1. pvpython paraview_benchmark_comparison.py   (ParaView Python)
  2. python3  paraview_benchmark_comparison.py   (需要 numpy + matplotlib)
  3. 在 ParaView GUI 中 Tools → Python Shell → Run Script

輸出:
  benchmark_Umean_Re700.png / .pdf  — U/Ub 比對圖 (offset profile format)

Benchmark 資料來源:
  Breuer, M., Peller, N., Rapp, Ch., Manhart, M. (2009).
  "Flow over periodic hills – Numerical and experimental study in a wide
   range of Reynolds numbers." Computers & Fluids, 38, 433–457.
  下載: https://kbwiki.ercoftac.org/w/index.php?title=Abstr:2D_Periodic_Hill_Flow
        (ERCOFTAC UFR 3-30, 選 Re=700 DNS 資料)

  請將下載的 benchmark 資料放在 result/benchmark/ 目錄下，
  檔名格式: breuer_re700_xh{X}.dat  (如 breuer_re700_xh05.dat)
  每個檔案格式: 兩欄 (y/h  U/Ub), 空格分隔, 無表頭

座標映射:
  Code x → spanwise  (LX = 4.5)
  Code y → streamwise (LY = 9.0 = 9h, h=1)
  Code z → wall-normal (LZ = 3.036)
  VTK U_mean = <v_code> / Uref  ≈ U / Ub (when Ub ≈ Uref)
"""

import os, sys, glob
import numpy as np

# ── attempt to use Agg backend (no GUI needed) ────────────────
try:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found; will export CSV only.")

# ================================================================
# 設定區
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

# VTK 檔案 (自動使用最新的 velocity_merged_*.vtk)
VTK_DIR = SCRIPT_DIR
VTK_PATTERN = "velocity_merged_*.vtk"

# Benchmark 資料目錄
BENCH_DIR = os.path.join(SCRIPT_DIR, "benchmark")

# 要提取的 x/h 站位 (streamwise = code y, hill-to-hill = 9h, h=1)
XH_STATIONS = [0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

# 物理參數
H_HILL = 1.0       # hill height
LY     = 9.0       # streamwise periodic length
LZ     = 3.036     # channel height (top wall z)

# ── Hill profile (needed for benchmark coordinate transform + bottom decoration)
def hill_function(Y):
    """Standard periodic hill geometry, h=1, period=9h."""
    Y = np.asarray(Y, dtype=float).copy()
    Y = np.where(Y < 0, Y + LY, Y)
    Y = np.where(Y > LY, Y - LY, Y)
    model = np.zeros_like(Y)
    t = Y * 28.0
    seg1 = Y <= (54.0/28.0)*(9.0/54.0)
    model = np.where(seg1, (1.0/28.0)*np.minimum(28.0, 28.0 + 0.006775070969851*t*t - 0.0021245277758000*t*t*t), model)
    seg2 = (Y > (54.0/28.0)*(9.0/54.0)) & (Y <= (54.0/28.0)*(14.0/54.0))
    model = np.where(seg2, 1.0/28.0*(25.07355893131 + 0.9754803562315*t - 0.1016116352781*t*t + 0.001889794677828*t*t*t), model)
    seg3 = (Y > (54.0/28.0)*(14.0/54.0)) & (Y <= (54.0/28.0)*(20.0/54.0))
    model = np.where(seg3, 1.0/28.0*(25.79601052357 + 0.8206693007457*t - 0.09055370274339*t*t + 0.001626510569859*t*t*t), model)
    seg4 = (Y > (54.0/28.0)*(20.0/54.0)) & (Y <= (54.0/28.0)*(30.0/54.0))
    model = np.where(seg4, 1.0/28.0*(40.46435022819 - 1.379581654948*t + 0.019458845041284*t*t - 0.0002070318932190*t*t*t), model)
    seg5 = (Y > (54.0/28.0)*(30.0/54.0)) & (Y <= (54.0/28.0)*(40.0/54.0))
    model = np.where(seg5, 1.0/28.0*(17.92461334664 + 0.8743920332081*t - 0.05567361123058*t*t + 0.0006277731764683*t*t*t), model)
    seg6 = (Y > (54.0/28.0)*(40.0/54.0)) & (Y <= (54.0/28.0)*(54.0/54.0))
    model = np.where(seg6, 1.0/28.0*np.maximum(0.0, 56.39011190988 - 2.010520359035*t + 0.01644919857549*t*t + 0.00002674976141766*t*t*t), model)
    Yr = LY - Y; tr = Yr * 28.0; rseg = (Y >= LY - (54.0/28.0))
    model = np.where(rseg & (Yr <= (54.0/28.0)*(9.0/54.0)), (1.0/28.0)*np.minimum(28.0, 28.0 + 0.006775070969851*tr*tr - 0.0021245277758000*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(9.0/54.0)) & (Yr <= (54.0/28.0)*(14.0/54.0)), 1.0/28.0*(25.07355893131 + 0.9754803562315*tr - 0.1016116352781*tr*tr + 0.001889794677828*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(14.0/54.0)) & (Yr <= (54.0/28.0)*(20.0/54.0)), 1.0/28.0*(25.79601052357 + 0.8206693007457*tr - 0.09055370274339*tr*tr + 0.001626510569859*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(20.0/54.0)) & (Yr <= (54.0/28.0)*(30.0/54.0)), 1.0/28.0*(40.46435022819 - 1.379581654948*tr + 0.019458845041284*tr*tr - 0.0002070318932190*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(30.0/54.0)) & (Yr <= (54.0/28.0)*(40.0/54.0)), 1.0/28.0*(17.92461334664 + 0.8743920332081*tr - 0.05567361123058*tr*tr + 0.0006277731764683*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(40.0/54.0)) & (Yr <= (54.0/28.0)*(54.0/54.0)), 1.0/28.0*np.maximum(0.0, 56.39011190988 - 2.010520359035*tr + 0.01644919857549*tr*tr + 0.00002674976141766*tr*tr*tr), model)
    return model

# ================================================================
# 1. 找出最新的 VTK 檔案
# ================================================================
vtk_files = sorted(glob.glob(os.path.join(VTK_DIR, VTK_PATTERN)),
                    key=lambda f: int(''.join(c for c in os.path.basename(f) if c.isdigit()) or '0'))
if not vtk_files:
    sys.exit(f"[ERROR] No VTK files matching '{VTK_PATTERN}' found in {VTK_DIR}")
vtk_path = vtk_files[-1]  # latest by timestep number
vtk_name = os.path.basename(vtk_path)
print(f"[INFO] Loading VTK: {vtk_name}")

# ================================================================
# 2. Parse ASCII Structured-Grid VTK
# ================================================================
def parse_vtk(filepath):
    """Read points, velocity, and scalar fields from ASCII STRUCTURED_GRID VTK."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Header
    dims = None
    npts = 0
    npts_from_dims = 0
    points = []
    scalars = {}
    current_scalar = None
    section = None
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()

        if line.startswith("DIMENSIONS"):
            dims = tuple(int(v) for v in line.split()[1:4])  # (nx, ny, nz)
            npts_from_dims = dims[0] * dims[1] * dims[2]

        elif line.startswith("POINT_DATA"):
            npts = int(line.split()[1])

        elif line.startswith("POINTS"):
            npts = int(line.split()[1])
            section = "points"
            idx += 1
            raw = []
            while len(raw) < npts * 3 and idx < len(lines):
                vals = lines[idx].strip().split()
                if not vals or vals[0].startswith(("SCALARS", "VECTORS", "POINT_DATA")):
                    break
                raw.extend(float(v) for v in vals)
                idx += 1
            raw = raw[:npts * 3]
            points = list(zip(raw[0::3], raw[1::3], raw[2::3]))
            continue

        elif line.startswith("VECTORS"):
            # Skip vector data block (3 components per point)
            idx += 1
            remaining = npts * 3
            while remaining > 0 and idx < len(lines):
                vals = lines[idx].strip().split()
                if not vals or vals[0].startswith(("SCALARS", "VECTORS", "POINT_DATA")):
                    break
                remaining -= len(vals)
                idx += 1
            continue

        elif line.startswith("SCALARS"):
            if npts == 0 and npts_from_dims > 0:
                npts = npts_from_dims  # fallback from DIMENSIONS
            parts = line.split()
            current_scalar = parts[1]
            scalars[current_scalar] = []
            section = "scalar"
            idx += 1  # skip LOOKUP_TABLE line
            idx += 1
            count = 0
            while count < npts and idx < len(lines):
                vals = lines[idx].strip().split()
                if not vals or vals[0].startswith("SCALARS") or vals[0].startswith("VECTORS"):
                    break
                for v in vals:
                    if count < npts:
                        scalars[current_scalar].append(float(v))
                        count += 1
                idx += 1
            continue

        idx += 1

    points = np.array(points)  # (npts, 3)
    for key in scalars:
        scalars[key] = np.array(scalars[key])

    return dims, points, scalars

def check_vtk_completeness(filepath):
    """Pre-parse scan: verify VTK file has all required sections.
    Returns (ok, diagnostics_str). If not ok, diagnostics_str explains what's missing."""
    markers = {
        "DIMENSIONS": False,
        "POINTS":     False,
        "POINT_DATA": False,
        "U_mean":     False,
        "W_mean":     False,
        "V_mean":     False,
    }
    total_lines = 0
    grid_npts = 0

    with open(filepath, "r") as f:
        for line in f:
            total_lines += 1
            stripped = line.strip()
            if stripped.startswith("DIMENSIONS"):
                markers["DIMENSIONS"] = True
                parts = stripped.split()
                if len(parts) >= 4:
                    grid_npts = int(parts[1]) * int(parts[2]) * int(parts[3])
            elif stripped.startswith("POINTS"):
                markers["POINTS"] = True
            elif stripped.startswith("POINT_DATA"):
                markers["POINT_DATA"] = True
            elif stripped.startswith("SCALARS U_mean"):
                markers["U_mean"] = True
            elif stripped.startswith("SCALARS W_mean"):
                markers["W_mean"] = True
            elif stripped.startswith("SCALARS V_mean"):
                markers["V_mean"] = True

    # Expected line count estimate:
    #   header ~5 + POINTS section (npts lines) + POINT_DATA + per scalar (2 header + npts data)
    #   Plus VECTORS blocks. Conservative lower bound: header + POINTS + 3 scalars
    expected_min = 5 + grid_npts + 3 * (grid_npts + 2)  # ~4 * npts

    missing = [k for k, v in markers.items() if not v]
    diag_lines = []
    diag_lines.append(f"  檔案: {os.path.basename(filepath)}")
    diag_lines.append(f"  總行數: {total_lines:,}")
    if grid_npts > 0:
        diag_lines.append(f"  網格點數: {grid_npts:,}")
        diag_lines.append(f"  預估最少行數: ~{expected_min:,}")
        pct = total_lines / expected_min * 100
        diag_lines.append(f"  完成度: ~{pct:.1f}%")

    ok = len(missing) == 0
    if not ok:
        diag_lines.append(f"  缺少區段: {', '.join(missing)}")
        if grid_npts > 0 and total_lines < expected_min:
            diag_lines.append(f"  → 檔案傳輸未完成 (行數不足)，請等待同步完成後重試")
        elif not markers["U_mean"]:
            if markers["POINTS"] and not markers["POINT_DATA"]:
                diag_lines.append(f"  → 檔案在 POINTS 區段後被截斷")
            elif markers["POINT_DATA"] and not markers["U_mean"]:
                diag_lines.append(f"  → 檔案在 POINT_DATA 後被截斷，缺少時間平均場")
            else:
                diag_lines.append(f"  → 此 VTK 不含時間平均資料 (U_mean)，可能尚未開始統計")

    return ok, "\n".join(diag_lines)

# ── 先檢查檔案完整性 ──
vtk_ok, vtk_diag = check_vtk_completeness(vtk_path)
if not vtk_ok:
    print(f"\n[ERROR] VTK 檔案輸出不完全！")
    print(vtk_diag)
    sys.exit(1)

dims, points, scalars = parse_vtk(vtk_path)
nx, ny, nz = dims
print(f"[INFO] Grid: {nx} × {ny} × {nz} = {nx*ny*nz} points")
print(f"[INFO] Available scalars: {list(scalars.keys())}")

if "U_mean" not in scalars:
    sys.exit("[ERROR] U_mean field not found after parsing. File may be corrupted.")

# Reshape to 3D arrays: VTK order is (i fastest, j, k slowest)
# points[k*ny*nx + j*nx + i] = (x, y, z)
pts_3d = points.reshape(nz, ny, nx, 3)  # [k, j, i, xyz]
U_mean_3d = scalars["U_mean"].reshape(nz, ny, nx)  # [k, j, i]
W_mean_3d = scalars.get("W_mean", np.zeros_like(U_mean_3d)).reshape(nz, ny, nx)

# Extract coordinate arrays
x_3d = pts_3d[:, :, :, 0]  # spanwise
y_3d = pts_3d[:, :, :, 1]  # streamwise
z_3d = pts_3d[:, :, :, 2]  # wall-normal

# y stations (streamwise, constant for all i at fixed j,k)
y_stations = y_3d[0, :, 0]  # shape (ny,)

# ================================================================
# 3. Extract spanwise-averaged profiles at each x/h station
# ================================================================
def extract_profile(xh, y_stations, z_3d, U_3d):
    """Extract spanwise-averaged vertical profile at streamwise station x/h."""
    y_target = xh * H_HILL  # physical y coordinate

    # Find nearest j index
    j_idx = np.argmin(np.abs(y_stations - y_target))
    y_actual = y_stations[j_idx]

    # z coordinates at this j (spanwise-averaged should all be same)
    z_profile = z_3d[:, j_idx, 0]  # [k], z at i=0 (same for all i)

    # Spanwise average of U_mean over all i
    U_profile = np.mean(U_3d[:, j_idx, :], axis=1)  # average over i → [k]

    # Wall height at this station (minimum z at k=0)
    z_wall = z_profile[0]

    # Normalize: (z - z_wall) / h
    z_norm = (z_profile - z_wall) / H_HILL

    return z_norm, U_profile, y_actual, z_wall

profiles = {}
print(f"\n{'x/h':>6s}  {'j_idx':>5s}  {'y_actual':>8s}  {'z_wall':>6s}  {'U_max':>8s}")
print("-" * 48)
for xh in XH_STATIONS:
    z_n, U_p, y_a, z_w = extract_profile(xh, y_stations, z_3d, U_mean_3d)
    # Store absolute z/h so profiles start from the hill surface
    z_abs = z_n + z_w / H_HILL
    profiles[xh] = (z_abs, U_p, z_w)
    print(f"{xh:6.1f}  {np.argmin(np.abs(y_stations - xh)):5d}  {y_a:8.4f}  {z_w:6.4f}  {U_p.max():8.5f}")

# ================================================================
# 4. Load benchmark data (Breuer Re700, ERCOFTAC UFR 3-30)
# ================================================================
# ERCOFTAC file-number → x/h mapping
_ERCOFTAC_XH = {
    "001": 0.05, "002": 0.5,  "003": 1.0,  "004": 2.0,  "005": 3.0,
    "006": 4.0,  "007": 5.0,  "008": 6.0,  "009": 7.0,  "010": 8.0,
}

benchmark = {}
if os.path.isdir(BENCH_DIR):
    for xh in XH_STATIONS:
        # Determine ERCOFTAC file number for this x/h
        file_num = None
        for num, x in _ERCOFTAC_XH.items():
            if abs(x - xh) < 1e-6:
                file_num = num
                break

        # Try multiple naming conventions
        xh_str = f"{xh:.1f}".replace(".", "")
        candidates = [
            f"breuer_re700_xh{xh_str}.dat",
            f"breuer_re700_xh{xh:.0f}.dat",
            f"Re700_x{xh:.1f}.dat",
            f"xh{xh_str}.dat",
        ]
        if file_num:
            candidates.insert(0, f"UFR3-30_C_700_data_MB-{file_num}.dat")

        for cand in candidates:
            fpath = os.path.join(BENCH_DIR, cand)
            if os.path.exists(fpath):
                # ERCOFTAC format: 7 columns (y/h  U/Ub  V/Ub  uu  vv  uv  k)
                # with # comment header lines; y/h is absolute coordinate
                data = np.loadtxt(fpath, comments="#")
                y_abs = data[:, 0]   # absolute y/h
                U_bench = data[:, 1] # U/Ub

                # Convert absolute y/h → (y - y_wall) / h
                z_wall_bench = hill_function(np.array([xh]))[0]
                # Keep absolute y/h (profiles start from hill surface)
                benchmark[xh] = (y_abs, U_bench)
                print(f"[INFO] Loaded benchmark x/h={xh}: {cand} ({len(data)} pts, "
                      f"y/h=[{y_abs.min():.3f},{y_abs.max():.3f}], z_wall={z_wall_bench:.4f})")
                break
    if not benchmark:
        print("[WARN] No benchmark files found in benchmark/. Plotting simulation only.")
else:
    print(f"[WARN] Benchmark directory {BENCH_DIR} not found. Creating it...")
    os.makedirs(BENCH_DIR, exist_ok=True)
    print(f"       Place benchmark data files in: {BENCH_DIR}")

# ================================================================
# 5. Plot: offset profile format (standard CFD benchmark style)
# ================================================================
if not HAS_MPL:
    # If no matplotlib, export CSVs
    for xh in XH_STATIONS:
        z_n, U_p = profiles[xh]
        out = os.path.join(SCRIPT_DIR, f"profile_xh{xh:.1f}.csv")
        np.savetxt(out, np.column_stack([z_n, U_p]), header="z/h  U/Uref", fmt="%.8f")
        print(f"[INFO] Exported {out}")
    sys.exit(0)

# ── Matplotlib style ───────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         11,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
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
    "lines.linewidth":   1.2,
    "savefig.dpi":       300,
})

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.0))

# Profile scale factor (how wide each profile appears in x/h units)
SCALE = 0.8

# Colors
c_sim  = "#1F77B4"
c_ref  = "#D62728"

# Step 1: fill hill geometry at the bottom
yh_fine = np.linspace(0, LY, 2000)
zh_fine = hill_function(yh_fine)
ax.fill_between(yh_fine / H_HILL, 0, zh_fine / H_HILL, color="0.90", zorder=0)
ax.plot(yh_fine / H_HILL, zh_fine / H_HILL, color="0.45", lw=1.2, zorder=1)

# Upper wall
ax.axhline(y=LZ / H_HILL, color="0.45", lw=1.2, zorder=1)

# Step 2: plot each profile (absolute z/h, starting from hill surface)
for xh in XH_STATIONS:
    z_abs, U_p, z_w = profiles[xh]

    # Offset: U_p * SCALE + x/h
    x_plot = U_p * SCALE + xh
    ax.plot(x_plot, z_abs, "-", color=c_sim, lw=1.0, zorder=5)

    # Zero reference line (thin dashed, from wall to top)
    ax.plot([xh, xh], [z_abs[0], z_abs[-1]], "--", color="0.3", lw=0.4, zorder=2)

    # Benchmark data (hollow scatter)
    if xh in benchmark:
        z_b, U_b = benchmark[xh]
        x_b_plot = U_b * SCALE + xh
        ax.scatter(x_b_plot, z_b, s=12, facecolors="none", edgecolors=c_ref,
                   linewidths=0.5, zorder=6, marker="o")

# Step 3: legend (placed in whitespace area)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=c_sim, lw=1.3, label="Present (GILBM)"),
]
if benchmark:
    legend_elements.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=c_ref, markersize=5, markeredgewidth=0.8,
               label="Breuer et al. (2009) DNS")
    )
ax.legend(handles=legend_elements, loc="lower center", frameon=True,
          edgecolor="0.7", fancybox=False, ncol=2, fontsize=10,
          bbox_to_anchor=(0.5, -0.28))

# Step 4: labels and limits — frame tightly around hill range
ax.set_ylabel(r"$y\,/\,h$", fontsize=13)
ax.set_xlabel(r"$x\,/\,h$", fontsize=13)

# x ticks at integer positions 0..9
ax.set_xticks(range(10))
ax.set_xlim(0, 9)
ax.set_ylim(0, LZ / H_HILL)
ax.set_yticks([0, 1, 2, 3])
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
out_png = os.path.join(SCRIPT_DIR, "benchmark_Umean_Re700.png")
out_pdf = os.path.join(SCRIPT_DIR, "benchmark_Umean_Re700.pdf")
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"\n[OK] Saved: {os.path.basename(out_png)}, {os.path.basename(out_pdf)}")
