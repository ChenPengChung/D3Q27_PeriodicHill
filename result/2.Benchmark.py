#!/usr/bin/env python3
"""
ERCOFTAC UFR 3-30 Periodic Hill Benchmark Comparison
GILBM vs Breuer et al. (2009) DNS Re=700
=====================================================
用法:
  python3 2.Benchmark.py   (需要 numpy + matplotlib)

輸出:
  benchmark_Umean_Re700.png   — <U>/Ub offset profile 比對圖
  benchmark_RS_Re700.png      — RS + k + <V>/Ub 5-panel 比對圖
  benchmark_profiles_*.png    — 6 頁 per-station profile (2×5 layout)

Benchmark 資料來源:
  Breuer et al. (2009), Computers & Fluids, 38, 433-457.
  ERCOFTAC UFR 3-30, Re=700 DNS data.
  請將下載的 benchmark 資料放在 result/benchmark/ 目錄下，
  檔名格式: UFR3-30_C_700_data_MB-{NNN}.dat

座標映射 (Code ↔ VTK ↔ ERCOFTAC):
  Code x (i) = spanwise     → VTK v   → ERCOFTAC w
  Code y (j) = streamwise   → VTK u   → ERCOFTAC u
  Code z (k) = wall-normal  → VTK w   → ERCOFTAC v

  ERCOFTAC col 2: <U>/Ub     = VTK U_mean   (streamwise = code v)
  ERCOFTAC col 3: <V>/Ub     = VTK W_mean   (wall-normal = code w, NOT V_mean!)
  ERCOFTAC col 4: <u'u'>/Ub² = VTK uu_RS    (stream×stream)
  ERCOFTAC col 5: <v'v'>/Ub² = VTK ww_RS    (wallnorm×wallnorm, NOT vv_RS!)
  ERCOFTAC col 6: <u'v'>/Ub² = VTK uw_RS    (stream×wallnorm, NOT uv_RS!)
  ERCOFTAC col 7: k/Ub²      = VTK k_TKE
"""

import os, sys, glob
import numpy as np

# ── auto-detect backend (Agg for headless, TkAgg for GUI) ─────
try:
    import matplotlib as mpl
    if not os.environ.get('DISPLAY') and sys.platform != 'win32':
        mpl.use('Agg')
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
W_mean_3d = scalars.get("W_mean", None)

# Reynolds stresses — VTK Level 0 fields only
# VTK naming: u=streamwise, v=spanwise, w=wall-normal
# ERCOFTAC naming: u=streamwise, v=wall-normal
#   VTK uu_RS = ERCOFTAC <u'u'>/Ub²
#   VTK uw_RS = ERCOFTAC <u'v'>/Ub²  (stream×wallnorm)
#   VTK ww_RS = ERCOFTAC <v'v'>/Ub²  (wallnorm×wallnorm)
#   VTK k_TKE = ERCOFTAC k/Ub²
uu_RS_3d = scalars.get("uu_RS", None)
uw_RS_3d = scalars.get("uw_RS", None)
ww_RS_3d = scalars.get("ww_RS", None)
k_TKE_3d = scalars.get("k_TKE", None)

# Reshape available fields
if W_mean_3d is not None: W_mean_3d = W_mean_3d.reshape(nz, ny, nx)
if uu_RS_3d is not None: uu_RS_3d = uu_RS_3d.reshape(nz, ny, nx)
if uw_RS_3d is not None: uw_RS_3d = uw_RS_3d.reshape(nz, ny, nx)
if ww_RS_3d is not None: ww_RS_3d = ww_RS_3d.reshape(nz, ny, nx)
if k_TKE_3d is not None: k_TKE_3d = k_TKE_3d.reshape(nz, ny, nx)

HAS_RS = (uu_RS_3d is not None)
if HAS_RS:
    print(f"[INFO] Reynolds stress fields found — will plot RS comparison")
else:
    print(f"[INFO] No Reynolds stress fields — plotting mean velocity only")

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

def extract_scalar_profile(xh, y_stations, field_3d):
    """Extract spanwise-averaged vertical profile of any scalar field."""
    if field_3d is None:
        return None
    y_target = xh * H_HILL
    j_idx = np.argmin(np.abs(y_stations - y_target))
    return np.mean(field_3d[:, j_idx, :], axis=1)  # average over i → [k]

profiles = {}
print(f"\n{'x/h':>6s}  {'j_idx':>5s}  {'y_actual':>8s}  {'z_wall':>6s}  {'U_max':>8s}")
print("-" * 48)
for xh in XH_STATIONS:
    z_n, U_p, y_a, z_w = extract_profile(xh, y_stations, z_3d, U_mean_3d)
    # Store absolute z/h so profiles start from the hill surface
    z_abs = z_n + z_w / H_HILL
    # Extract additional scalar profiles (spanwise-averaged)
    W_p = extract_scalar_profile(xh, y_stations, W_mean_3d)          # V_ERCOFTAC (wall-normal mean)
    uu_p = extract_scalar_profile(xh, y_stations, uu_RS_3d)          # uu (stream×stream)
    ww_p = extract_scalar_profile(xh, y_stations, ww_RS_3d)          # ERCOFTAC vv (wallnorm×wallnorm)
    uw_p = extract_scalar_profile(xh, y_stations, uw_RS_3d)          # ERCOFTAC uv (stream×wallnorm)
    k_p  = extract_scalar_profile(xh, y_stations, k_TKE_3d)          # k/Ub²
    profiles[xh] = {
        "z_abs": z_abs, "U": U_p, "z_w": z_w,
        "W": W_p, "uu": uu_p, "ww": ww_p, "uw": uw_p, "k": k_p,
    }
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

        # Search in root, old path, and new named path
        search_dirs = [
            BENCH_DIR,
            os.path.join(BENCH_DIR, "LESOCC (Breuer et al. 2009)", "Re700"),
            os.path.join(BENCH_DIR, "LESOCC", "Re700"),
        ]
        for cand in candidates:
            fpath = None
            for sdir in search_dirs:
                p = os.path.join(sdir, cand)
                if os.path.exists(p):
                    fpath = p
                    break
            if fpath:
                # ERCOFTAC format: 7 columns
                #   y/h   U/Ub   V/Ub   <u'u'>/Ub²   <v'v'>/Ub²   <u'v'>/Ub²   k/Ub²
                # ERCOFTAC u=streamwise, v=wall-normal (NOT spanwise!)
                data = np.loadtxt(fpath, comments="#")
                y_abs = data[:, 0]   # absolute y/h
                benchmark[xh] = {
                    "y":  y_abs,
                    "U":  data[:, 1],  # U/Ub (streamwise mean)
                    "V":  data[:, 2],  # V/Ub (wall-normal mean, ERCOFTAC v)
                    "uu": data[:, 3],  # <u'u'>/Ub² (stream×stream)
                    "vv": data[:, 4],  # <v'v'>/Ub² (wallnorm×wallnorm, ERCOFTAC v)
                    "uv": data[:, 5],  # <u'v'>/Ub² (stream×wallnorm)
                    "k":  data[:, 6],  # k/Ub²
                }
                print(f"[INFO] Loaded benchmark x/h={xh}: {cand} ({len(data)} pts, "
                      f"y/h=[{y_abs.min():.3f},{y_abs.max():.3f}])")
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
        p = profiles[xh]
        out = os.path.join(SCRIPT_DIR, f"profile_xh{xh:.1f}.csv")
        np.savetxt(out, np.column_stack([p["z_abs"], p["U"]]), header="z/h  U/Uref", fmt="%.8f")
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

# Colors
c_sim  = "#1F77B4"
c_ref  = "#D62728"
from matplotlib.lines import Line2D

# ── Helper: draw one offset-profile subplot ────────────────────
def plot_offset_panel(ax, profiles, benchmark, field_sim, field_bench,
                      scale, title, xlabel, ylabel=r"$y\,/\,h$"):
    """Generic offset-profile plotter for one subplot.
    field_sim:   key in profiles[xh] dict (e.g. "U", "uu", "uw")
    field_bench: key in benchmark[xh] dict (e.g. "U", "uu", "uv")
    """
    # Hill geometry
    yh_fine = np.linspace(0, LY, 2000)
    zh_fine = hill_function(yh_fine)
    ax.fill_between(yh_fine / H_HILL, 0, zh_fine / H_HILL, color="0.90", zorder=0)
    ax.plot(yh_fine / H_HILL, zh_fine / H_HILL, color="0.45", lw=1.0, zorder=1)
    ax.axhline(y=LZ / H_HILL, color="0.45", lw=1.0, zorder=1)

    for xh in XH_STATIONS:
        p = profiles[xh]
        z_abs = p["z_abs"]
        data_sim = p.get(field_sim)
        if data_sim is None:
            continue

        x_plot = data_sim * scale + xh
        ax.plot(x_plot, z_abs, "-", color=c_sim, lw=1.0, zorder=5)
        ax.plot([xh, xh], [z_abs[0], z_abs[-1]], "--", color="0.3", lw=0.3, zorder=2)

        if xh in benchmark and field_bench in benchmark[xh]:
            z_b = benchmark[xh]["y"]
            d_b = benchmark[xh][field_bench]
            x_b_plot = d_b * scale + xh
            ax.scatter(x_b_plot, z_b, s=10, facecolors="none", edgecolors=c_ref,
                       linewidths=0.4, zorder=6, marker="o")

    ax.set_xticks(range(10))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, LZ / H_HILL)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12, pad=4)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

# ================================================================
# 5a. Figure 1: Mean Velocity — <U>/Ub  (original plot)
# ================================================================
fig1, ax1 = plt.subplots(figsize=(10, 4.0))
SCALE_U = 0.8
plot_offset_panel(ax1, profiles, benchmark,
                  field_sim="U", field_bench="U",
                  scale=SCALE_U,
                  title=r"$\langle U \rangle / U_b$  (Re = 700, DNS)",
                  xlabel=r"$x\,/\,h$")

legend_elements = [
    Line2D([0], [0], color=c_sim, lw=1.3, label="Present (GILBM)"),
]
if benchmark:
    legend_elements.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=c_ref, markersize=5, markeredgewidth=0.8,
               label="Breuer et al. (2009) DNS")
    )
ax1.legend(handles=legend_elements, loc="lower center", frameon=True,
           edgecolor="0.7", fancybox=False, ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.28))
fig1.tight_layout()
out1 = os.path.join(SCRIPT_DIR, "benchmark_Umean_Re700.png")
fig1.savefig(out1, bbox_inches="tight")
print(f"\n[OK] Saved: {os.path.basename(out1)}")

# ================================================================
# 5b. Figure 2: 6-panel Reynolds stress + k + V_mean comparison
# ================================================================
if HAS_RS:
    # Define the 6 quantities to compare
    # (field_sim, field_bench, scale, title)
    panels = [
        ("uu", "uu", 30, r"$\langle u^\prime u^\prime \rangle / U_b^2$"),
        ("ww", "vv", 30, r"$\langle v^\prime v^\prime \rangle / U_b^2$  (wall-normal)"),
        ("uw", "uv", 60, r"$\langle u^\prime v^\prime \rangle / U_b^2$  (shear stress)"),
        ("k",  "k",  20, r"$k / U_b^2$  (TKE)"),
        ("W",  "V",   3, r"$\langle V \rangle / U_b$  (wall-normal mean)"),
    ]

    nrows, ncols = 3, 2
    fig2, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
    axes_flat = axes.flatten()

    for idx, (fs, fb, sc, ttl) in enumerate(panels):
        plot_offset_panel(axes_flat[idx], profiles, benchmark,
                          field_sim=fs, field_bench=fb,
                          scale=sc, title=ttl, xlabel=r"$x\,/\,h$")

    # Hide unused subplot (6th panel)
    axes_flat[-1].set_visible(False)

    # Single shared legend at the bottom
    legend_elements2 = [
        Line2D([0], [0], color=c_sim, lw=1.3, label="Present (GILBM)"),
    ]
    if benchmark:
        legend_elements2.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=c_ref, markersize=5, markeredgewidth=0.8,
                   label="Breuer et al. (2009) DNS Re=700")
        )
    fig2.legend(handles=legend_elements2, loc="lower center", frameon=True,
                edgecolor="0.7", fancybox=False, ncol=2, fontsize=12,
                bbox_to_anchor=(0.5, 0.01))

    fig2.suptitle("Periodic Hill Re = 700 — Reynolds Stress Comparison", fontsize=15, y=0.98)
    fig2.tight_layout(rect=[0, 0.04, 1, 0.96])
    out2 = os.path.join(SCRIPT_DIR, "benchmark_RS_Re700.png")
    fig2.savefig(out2, bbox_inches="tight")
    print(f"[OK] Saved: {os.path.basename(out2)}")
else:
    print("[INFO] No Reynolds stress data in VTK — skipping RS comparison plot.")

# ================================================================
# 5c. Figure 3: Per-station profiles (6 pages × 2×5 subplots)
# ================================================================
# Each page shows one physical quantity at all 10 x/h stations.
# X axis = physical quantity, Y axis = y/h (wall-normal).
# Red line = simulation, black dots = ERCOFTAC benchmark.
per_station_quantities = [
    # (field_sim, field_bench, xlabel,                           filename_suffix)
    ("U",  "U",  r"$\langle U \rangle / U_b$",                 "U"),
    ("W",  "V",  r"$\langle V \rangle / U_b$ (wall-normal)",   "V"),
    ("uu", "uu", r"$\langle u^\prime u^\prime \rangle / U_b^2$","uu"),
    ("ww", "vv", r"$\langle v^\prime v^\prime \rangle / U_b^2$","vv"),
    ("uw", "uv", r"$\langle u^\prime v^\prime \rangle / U_b^2$","uv"),
    ("k",  "k",  r"$k / U_b^2$",                                "k"),
]

for fs, fb, xlbl, suffix in per_station_quantities:
    # Check if simulation data exists for this field
    if profiles[XH_STATIONS[0]].get(fs) is None:
        print(f"[INFO] Skipping per-station page for '{suffix}' — no data in VTK")
        continue

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes_flat = axes.flatten()

    for idx, xh in enumerate(XH_STATIONS):
        ax = axes_flat[idx]
        p = profiles[xh]
        z_abs = p["z_abs"]
        data_sim = p.get(fs)

        # Simulation profile
        if data_sim is not None:
            ax.plot(data_sim, z_abs, "-", color=c_sim, lw=1.2, label="GILBM")

        # Benchmark profile
        if xh in benchmark and fb in benchmark[xh]:
            z_b = benchmark[xh]["y"]
            d_b = benchmark[xh][fb]
            ax.plot(d_b, z_b, "ko", ms=2.5, mfc="none", mew=0.6, label="Breuer DNS")

        ax.set_title(f"$x/h = {xh}$", fontsize=11)
        ax.set_xlabel(xlbl, fontsize=9)
        if idx % 5 == 0:
            ax.set_ylabel(r"$y\,/\,h$", fontsize=11)
        ax.set_ylim(0, LZ / H_HILL)
        ax.tick_params(labelsize=9)

    # Shared legend (from first subplot)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=11,
               frameon=True, edgecolor="0.7", fancybox=False,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f"Periodic Hill Re = 700 — {xlbl}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_ps = os.path.join(SCRIPT_DIR, f"benchmark_profiles_{suffix}.png")
    fig.savefig(out_ps, bbox_inches="tight")
    print(f"[OK] Saved: {os.path.basename(out_ps)}")
    plt.close(fig)
