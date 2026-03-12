#!/usr/bin/env python3
"""
ERCOFTAC UFR 3-30 Periodic Hill Benchmark Comparison
GILBM vs multiple benchmark sources (DNS/LES/Experiment)
=====================================================
用法:
  python3 2.Benchmark.py               # 互動詢問 Re
  python3 2.Benchmark.py --Re 700      # 指定 Re=700
  python3 2.Benchmark.py --Re 5600     # 指定 Re=5600

輸出:
  benchmark_Umean_Re{N}.png           — <U>/Ub offset profile
  benchmark_RS_Re{N}.png              — RS + k + <V>/Ub 5-panel offset
  benchmark_all_Re{N}.pdf/png         — 6x10 per-station 全比較圖

Benchmark 資料來源:
  Breuer et al. (2009), Computers & Fluids, 38, 433-457.
  Rapp & Manhart (2011), Experiments in Fluids.
  ERCOFTAC UFR 3-30 database.
  請將 benchmark 資料放在 result/benchmark/ 目錄下.

座標映射 (Code <-> VTK <-> ERCOFTAC):
  Code x (i) = spanwise     -> VTK v   -> ERCOFTAC w
  Code y (j) = streamwise   -> VTK u   -> ERCOFTAC u
  Code z (k) = wall-normal  -> VTK w   -> ERCOFTAC v

  ERCOFTAC col 2: <U>/Ub     = VTK U_mean   (streamwise = code v)
  ERCOFTAC col 3: <V>/Ub     = VTK W_mean   (wall-normal = code w, NOT V_mean!)
  ERCOFTAC col 4: <u'u'>/Ub² = VTK uu_RS    (stream x stream)
  ERCOFTAC col 5: <v'v'>/Ub² = VTK ww_RS    (wallnorm x wallnorm, NOT vv_RS!)
  ERCOFTAC col 6: <u'v'>/Ub² = VTK uw_RS    (stream x wallnorm, NOT uv_RS!)
  ERCOFTAC col 7: k/Ub²      = VTK k_TKE
"""

import os, sys, glob, argparse
import numpy as np

# Windows console UTF-8 support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── auto-detect backend ─────────────────────────────────────────
try:
    import matplotlib as mpl
    if not os.environ.get('DISPLAY') and sys.platform != 'win32':
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found; will export CSV only.")

# ================================================================
# Configuration
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
VTK_DIR = SCRIPT_DIR
VTK_PATTERN = "velocity_merged_*.vtk"
BENCH_DIR = os.path.join(SCRIPT_DIR, "benchmark")

XH_STATIONS = [0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_STN_TO_XH = {1: 0.05, 2: 0.5, 3: 1.0, 4: 2.0, 5: 3.0,
              6: 4.0, 7: 5.0, 8: 6.0, 9: 7.0, 10: 8.0}

H_HILL = 1.0
LY     = 9.0
LZ     = 3.036

# ================================================================
# Benchmark Sources Definition
# ================================================================
BENCHMARK_SOURCES = {
    'LESOCC': {
        'dir_name':  'LESOCC (Breuer et al. 2009)',
        'label':     'LESOCC (Breuer et al. 2009)',
        'delimiter': None,       # whitespace
        'color':     '#1f77b4',  # blue
        'marker':    's',
        'markersize': 3,
    },
    'MGLET': {
        'dir_name':  'MGLET (Breuer et al. 2009)',
        'label':     'MGLET (Breuer et al. 2009)',
        'delimiter': None,       # whitespace
        'color':     '#2ca02c',  # green
        'marker':    '^',
        'markersize': 3,
    },
    'Experiment': {
        'dir_name':  'Experiment (Rapp & Manhart 2011)',
        'label':     'Experiment (Rapp & Manhart 2011)',
        'delimiter': ',',        # comma-separated
        'color':     '#000000',  # black
        'marker':    'o',
        'markersize': 2.5,
    },
    'LBM': {
        'dir_name':  'LBM',
        'label':     'LBM (reference)',
        'delimiter': None,
        'color':     '#ff7f0e',  # orange
        'marker':    'D',
        'markersize': 3,
        'format':    'tecplot',  # Tecplot format with y-station offset
    },
}

# ================================================================
# Re Argument
# ================================================================
parser = argparse.ArgumentParser(description="ERCOFTAC benchmark comparison")
parser.add_argument('--Re', type=int, default=None, help='Reynolds number')
parser.add_argument('--Uref', type=float, default=None, help='Reference velocity (auto-detect from variables.h if omitted)')
args, _ = parser.parse_known_args()

# ---- Auto-detect Uref from variables.h ----
def _read_uref_from_header():
    """Parse '#define Uref <value>' from ../variables.h."""
    import re as _re
    header = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'variables.h')
    if not os.path.isfile(header):
        return None
    with open(header) as fh:
        for line in fh:
            m = _re.match(r'^\s*#\s*define\s+Uref\s+([\d.eE+\-]+)', line)
            if m:
                return float(m.group(1))
    return None

if args.Uref is not None:
    Uref = args.Uref
else:
    Uref = _read_uref_from_header()
    if Uref is None:
        Uref = 0.0583          # fallback default
        print(f"[WARN] Cannot read Uref from variables.h — using default {Uref}")
    else:
        print(f"[INFO] Uref = {Uref}  (from variables.h)")

if args.Re is not None:
    Re = args.Re
else:
    try:
        Re = int(input("Reynolds number (default 700): ") or "700")
    except (ValueError, EOFError):
        Re = 700
LAMINAR = (Re <= 100)
if Re <= 100:
    XH_STATIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # No 0.05, 0.5 — 對應 LBM 分布
print(f"[INFO] Re = {Re}  {'(laminar mode)' if LAMINAR else '(turbulent mode)'}")

# ================================================================
# Hill Function
# ================================================================
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
# VTK Parsing
# ================================================================
def parse_vtk(filepath):
    """Read points, velocity, and scalar fields from ASCII STRUCTURED_GRID VTK."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    dims = None
    npts = 0
    npts_from_dims = 0
    points = []
    scalars = {}
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()

        if line.startswith("DIMENSIONS"):
            dims = tuple(int(v) for v in line.split()[1:4])
            npts_from_dims = dims[0] * dims[1] * dims[2]

        elif line.startswith("POINT_DATA"):
            npts = int(line.split()[1])

        elif line.startswith("POINTS"):
            npts = int(line.split()[1])
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
            if npts == 0 and npts_from_dims > 0:
                npts = npts_from_dims
            parts = line.split()
            vec_name = parts[1] if len(parts) > 1 else "velocity"
            idx += 1
            raw_vec = []
            while len(raw_vec) < npts * 3 and idx < len(lines):
                vals = lines[idx].strip().split()
                if not vals:
                    idx += 1
                    continue
                if vals[0].startswith(("SCALARS", "VECTORS", "POINT_DATA")):
                    break
                raw_vec.extend(float(v) for v in vals)
                idx += 1
            raw_vec = raw_vec[:npts * 3]
            if len(raw_vec) == npts * 3:
                arr = np.array(raw_vec)
                scalars[f"{vec_name}_x"] = arr[0::3]
                scalars[f"{vec_name}_y"] = arr[1::3]
                scalars[f"{vec_name}_z"] = arr[2::3]
            continue

        elif line.startswith("SCALARS"):
            if npts == 0 and npts_from_dims > 0:
                npts = npts_from_dims
            parts = line.split()
            current_scalar = parts[1]
            scalars[current_scalar] = []
            idx += 1
            # skip LOOKUP_TABLE line if present
            if idx < len(lines) and lines[idx].strip().startswith("LOOKUP_TABLE"):
                idx += 1
            count = 0
            while count < npts and idx < len(lines):
                vals = lines[idx].strip().split()
                if not vals:
                    idx += 1
                    continue
                if vals[0].startswith("SCALARS") or vals[0].startswith("VECTORS") or vals[0].startswith("POINT_DATA"):
                    break
                for v in vals:
                    if count < npts:
                        scalars[current_scalar].append(float(v))
                        count += 1
                idx += 1
            continue

        idx += 1

    points = np.array(points)
    for key in scalars:
        scalars[key] = np.array(scalars[key])
    return dims, points, scalars


def check_vtk_completeness(filepath):
    """Pre-parse scan: verify VTK file has all required sections."""
    markers = {"DIMENSIONS": False, "POINTS": False, "POINT_DATA": False,
               "U_mean": False, "W_mean": False}
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

    expected_min = 5 + grid_npts + 3 * (grid_npts + 2)
    missing = [k for k, v in markers.items() if not v]
    diag = [f"  檔案: {os.path.basename(filepath)}", f"  總行數: {total_lines:,}"]
    if grid_npts > 0:
        diag.append(f"  網格點數: {grid_npts:,}")
        diag.append(f"  完成度: ~{total_lines / expected_min * 100:.1f}%")
    ok = len(missing) == 0
    if not ok:
        diag.append(f"  缺少區段: {', '.join(missing)}")
        if grid_npts > 0 and total_lines < expected_min:
            diag.append(f"  -> 檔案傳輸未完成 (行數不足)")
        elif not markers["U_mean"]:
            diag.append(f"  -> 此 VTK 不含時間平均資料 (U_mean)")
    return ok, "\n".join(diag)


# ================================================================
# Benchmark Scanning & Loading
# ================================================================
def find_re_directory(source_dir, target_re):
    """Find Re subdirectory matching target_re (exact or within 5%)."""
    if not os.path.isdir(source_dir):
        return None, None
    re_dirs = {}
    for d in os.listdir(source_dir):
        if d.startswith("Re") and os.path.isdir(os.path.join(source_dir, d)):
            try:
                re_dirs[int(d[2:])] = d
            except ValueError:
                pass
    if target_re in re_dirs:
        return os.path.join(source_dir, re_dirs[target_re]), target_re
    for re_val, dirname in sorted(re_dirs.items(), key=lambda x: abs(x[0] - target_re)):
        if abs(re_val - target_re) / max(target_re, 1) < 0.05:
            return os.path.join(source_dir, dirname), re_val
    return None, None


def find_station_files(re_dir):
    """Find station files by new naming convention.

    Naming: {Source}_Re{Re}_{xh}.dat  (e.g. LESOCC_Re700_0.05.dat)
    Also:   {Source}_Re{Re}_{xh}.DAT  (e.g. LBM_Re100_0.DAT)

    Returns {xh_float: filepath}.
    """
    files = sorted(glob.glob(os.path.join(re_dir, "*.dat")) +
                   glob.glob(os.path.join(re_dir, "*.DAT")))
    xh_map = {}
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        parts = base.rsplit('_', 1)
        if len(parts) == 2:
            xh_str = parts[1]
            if xh_str == 'wall':
                continue  # skip wall data files (MGLET)
            try:
                xh = float(xh_str)
                xh_map[xh] = f
            except ValueError:
                pass
    return xh_map


def load_station_file(filepath, delimiter=None, fmt=None, xh_station=0.0):
    """Load one station .dat file.

    fmt='tecplot': Tecplot format (LBM), columns: z, uavg, vavg, wavg
                   vavg/wavg contain y-station offset → subtract xh_station.
    fmt=None:      ERCOFTAC format, columns: y/h, U, V, uu, vv, uv [, k]

    Returns dict {y, U, V, uu, vv, uv, k} or None.
    """
    try:
        if fmt == 'tecplot':
            data = np.loadtxt(filepath, skiprows=3)
            if data.ndim < 2 or data.shape[1] < 4:
                return None
            # LBM: z, uavg(col1), vavg(col2), wavg(col3)
            # U=streamwise=vavg-offset, V=wall-normal=wavg-offset
            return {
                "y":  data[:, 0],                         # z (abs wall-normal)
                "U":  data[:, 2] - xh_station,             # vavg - offset = U/Ub (streamwise)
                "V":  data[:, 3] - xh_station,             # wavg - offset = V/Ub (wall-normal)
                "uu": None, "vv": None, "uv": None, "k": None,
            }
        else:
            data = np.loadtxt(filepath, comments="#", delimiter=delimiter)
    except Exception:
        return None
    if data.ndim < 2 or data.shape[1] < 6:
        return None  # skip non-profile files (e.g. MGLET file-11 with 3 cols)
    result = {
        "y":  data[:, 0],
        "U":  data[:, 1],
        "V":  data[:, 2],
        "uu": data[:, 3],
        "vv": data[:, 4],
        "uv": data[:, 5],
    }
    result["k"] = data[:, 6] if data.shape[1] >= 7 else None
    return result


def scan_and_load_benchmarks(bench_dir, target_re):
    """Scan benchmark dir for all available sources at target Re.
    Returns list of (source_id, info, data_dict) where data_dict = {xh: {y,U,V,uu,...}}
    """
    results = []
    print(f"\n{'='*60}")
    print(f"掃描 Re={target_re} 的 benchmark 數據源...")
    print(f"{'='*60}")

    for src_id, info in BENCHMARK_SOURCES.items():
        src_dir = os.path.join(bench_dir, info['dir_name'])
        re_dir, matched_re = find_re_directory(src_dir, target_re)

        if re_dir is None:
            # List what Re values ARE available
            if os.path.isdir(src_dir):
                avail = sorted([d for d in os.listdir(src_dir)
                               if d.startswith("Re") and os.path.isdir(os.path.join(src_dir, d))])
                print(f"  \u274c {info['label']} \u2014 Re{target_re} 無數據 (有: {', '.join(avail)})")
            else:
                print(f"  \u274c {info['label']} \u2014 目錄不存在")
            continue

        xh_files = find_station_files(re_dir)
        if not xh_files:
            print(f"  \u26a0\ufe0f  {info['label']} \u2014 Re{matched_re} 目錄存在但無 .dat 檔案")
            continue

        data = {}
        src_fmt = info.get('format')  # None or 'tecplot'
        for xh, fpath in sorted(xh_files.items()):
            stn_data = load_station_file(fpath, info.get('delimiter'),
                                         fmt=src_fmt, xh_station=xh)
            if stn_data is not None:
                data[xh] = stn_data

        if data:
            re_suffix = f" (matched Re{matched_re})" if matched_re != target_re else ""
            print(f"  \u2705 {info['label']} \u2014 Re{matched_re}{re_suffix}: "
                  f"{len(data)} 站位")
            results.append((src_id, info, data))
        else:
            print(f"  \u26a0\ufe0f  {info['label']} \u2014 Re{matched_re} 檔案載入失敗")

    print(f"\n可用數據源: {len(results)} 個")
    return results


# ================================================================
# 1. Load VTK (optional — benchmark-only mode if U_mean absent)
# ================================================================
HAS_VTK = False
vtk_files = sorted(glob.glob(os.path.join(VTK_DIR, VTK_PATTERN)),
                    key=lambda f: int(''.join(c for c in os.path.basename(f) if c.isdigit()) or '0'))
if not vtk_files:
    print(f"[WARN] No VTK files matching '{VTK_PATTERN}' found — benchmark-only mode.")
else:
    vtk_path = vtk_files[-1]
    print(f"[INFO] Loading VTK: {os.path.basename(vtk_path)}")
    dims, points, scalars = parse_vtk(vtk_path)
    nx, ny, nz = dims
    print(f"[INFO] Grid: {nx} x {ny} x {nz} = {nx*ny*nz} points")
    print(f"[INFO] Available scalars: {list(scalars.keys())}")

    # Determine which velocity fields to use
    expected_npts = nx * ny * nz
    vel_u_key, vel_w_key = None, None
    if LAMINAR:
        for u_candidate, w_candidate in [
            ("u_inst", "w_inst"),
            ("velocity_y", "velocity_z"),   # VECTORS velocity → y=streamwise, z=wall-normal
            ("U_mean", "W_mean"),
        ]:
            if u_candidate in scalars and len(scalars[u_candidate]) == expected_npts:
                vel_u_key, vel_w_key = u_candidate, w_candidate
                break
        if vel_u_key is None:
            print(f"[WARN] No complete velocity field found (expected {expected_npts} pts).")
            for k, v in scalars.items():
                print(f"       {k}: {len(v)} pts")
    else:
        vel_u_key, vel_w_key = "U_mean", "W_mean"

    if vel_u_key is not None and vel_u_key in scalars:
        HAS_VTK = True
        # Reshape to 3D: VTK order is (i fastest, j, k slowest)
        pts_3d    = points.reshape(nz, ny, nx, 3)
        U_mean_3d = scalars[vel_u_key].reshape(nz, ny, nx)
        W_mean_3d = scalars.get(vel_w_key)

        if LAMINAR:
            # Laminar: no RS/TKE fields
            uu_RS_3d = uw_RS_3d = ww_RS_3d = k_TKE_3d = None
        else:
            uu_RS_3d  = scalars.get("uu_RS")
            uw_RS_3d  = scalars.get("uw_RS")
            ww_RS_3d  = scalars.get("ww_RS")
            k_TKE_3d  = scalars.get("k_TKE")

        if W_mean_3d is not None: W_mean_3d = W_mean_3d.reshape(nz, ny, nx)
        if uu_RS_3d  is not None: uu_RS_3d  = uu_RS_3d.reshape(nz, ny, nx)
        if uw_RS_3d  is not None: uw_RS_3d  = uw_RS_3d.reshape(nz, ny, nx)
        if ww_RS_3d  is not None: ww_RS_3d  = ww_RS_3d.reshape(nz, ny, nx)
        if k_TKE_3d  is not None: k_TKE_3d  = k_TKE_3d.reshape(nz, ny, nx)

        # ---- Normalize raw velocity fields by Uref ----
        if vel_u_key == "velocity_y":
            print(f"[INFO] Normalizing {vel_u_key}/{vel_w_key} by Uref = {Uref}")
            U_mean_3d = U_mean_3d / Uref
            if W_mean_3d is not None:
                W_mean_3d = W_mean_3d / Uref

        HAS_RS = (uu_RS_3d is not None)
        print(f"[INFO] Using VTK fields: {vel_u_key}, {vel_w_key}")
        if not LAMINAR:
            print(f"[INFO] {'RS fields found' if HAS_RS else 'No RS fields'}")

        y_3d = pts_3d[:, :, :, 1]  # streamwise
        z_3d = pts_3d[:, :, :, 2]  # wall-normal
        y_stations = y_3d[0, :, 0]
    else:
        avail = list(scalars.keys())
        print(f"[WARN] No usable velocity field found in VTK — benchmark-only mode.")
        if not avail:
            print(f"[WARN] VTK file contains 0 scalar/vector fields — file may be incomplete.")

if not HAS_VTK:
    HAS_RS = False

# ================================================================
# 2. Extract spanwise-averaged profiles (only if VTK available)
# ================================================================
def extract_profile(xh):
    """Extract spanwise-averaged U profile at station x/h."""
    j_idx = np.argmin(np.abs(y_stations - xh * H_HILL))
    y_actual = y_stations[j_idx]
    z_profile = z_3d[:, j_idx, 0]
    U_profile = np.mean(U_mean_3d[:, j_idx, :], axis=1)
    z_wall = z_profile[0]
    z_norm = (z_profile - z_wall) / H_HILL
    return z_norm, U_profile, y_actual, z_wall

def extract_scalar_profile(xh, field_3d):
    """Extract spanwise-averaged vertical profile of any scalar field."""
    if field_3d is None:
        return None
    j_idx = np.argmin(np.abs(y_stations - xh * H_HILL))
    return np.mean(field_3d[:, j_idx, :], axis=1)

profiles = {}
if HAS_VTK:
    print(f"\n{'x/h':>6s}  {'j_idx':>5s}  {'y_actual':>8s}  {'z_wall':>6s}  {'U_max':>8s}")
    print("-" * 48)
    for xh in XH_STATIONS:
        z_n, U_p, y_a, z_w = extract_profile(xh)
        z_abs = z_n + z_w / H_HILL
        profiles[xh] = {
            "z_abs": z_abs, "U": U_p, "z_w": z_w,
            "W":  extract_scalar_profile(xh, W_mean_3d),
            "uu": extract_scalar_profile(xh, uu_RS_3d),
            "ww": extract_scalar_profile(xh, ww_RS_3d),
            "uw": extract_scalar_profile(xh, uw_RS_3d),
            "k":  extract_scalar_profile(xh, k_TKE_3d),
        }
        print(f"{xh:6.2f}  {np.argmin(np.abs(y_stations - xh)):5d}  "
              f"{y_a:8.4f}  {z_w:6.4f}  {U_p.max():8.5f}")
else:
    print("[INFO] No VTK time-averaged data — skipping profile extraction.")

# ================================================================
# 3. Load benchmark data (multi-source)
# ================================================================
bench_sources = scan_and_load_benchmarks(BENCH_DIR, Re)

if not bench_sources and not HAS_VTK:
    sys.exit("[ERROR] No benchmark data found and no VTK data. Nothing to plot.")

# Collect ALL unique x/H stations from benchmarks + VTK
ALL_XH_STATIONS = sorted(set(XH_STATIONS))
for _, _, bdata in bench_sources:
    ALL_XH_STATIONS = sorted(set(ALL_XH_STATIONS) | set(bdata.keys()))

# ================================================================
# 4. Plotting
# ================================================================
if not HAS_MPL:
    if HAS_VTK:
        for xh in XH_STATIONS:
            p = profiles[xh]
            out = os.path.join(SCRIPT_DIR, f"profile_xh{xh:.1f}_Re{Re}.csv")
            np.savetxt(out, np.column_stack([p["z_abs"], p["U"]]),
                       header="z/h  U/Uref", fmt="%.8f")
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

c_sim = "#BB1792"  # red for simulation


def make_legend_elements():
    """Build legend entries: simulation + all available benchmark sources."""
    elems = []
    if HAS_VTK:
        elems.append(Line2D([0], [0], color=c_sim, lw=1.3, label="GILBM (present)"))
    for _, info, _ in bench_sources:
        elems.append(
            Line2D([0], [0], marker=info['marker'], color="none",
                   markerfacecolor="none", markeredgecolor=info['color'],
                   markersize=5, markeredgewidth=0.8, label=info['label'])
        )
    return elems


# ── Helper: offset-profile subplot (multi-source) ─────────────
def plot_offset_panel(ax, field_sim, field_bench, scale, title, xlabel, xlim_range=None):
    """Offset-profile plotter with multi-source benchmark overlay.
    xlim_range: (xmin, xmax) adaptive limits; None → default [0,9].
    """
    xl = xlim_range[0] * H_HILL if xlim_range else 0
    xr = xlim_range[1] * H_HILL if xlim_range else LY
    # Hill shape always drawn within one period [0, 9h]
    yh_fine = np.linspace(0, LY, 3000)
    zh_fine = hill_function(yh_fine)
    ax.fill_between(yh_fine / H_HILL, 0, zh_fine / H_HILL, color="0.90", zorder=0)
    ax.plot(yh_fine / H_HILL, zh_fine / H_HILL, color="0.45", lw=1.0, zorder=1)
    ax.axhline(y=LZ / H_HILL, color="0.45", lw=1.0, zorder=1)

    for xh in ALL_XH_STATIONS:
        # VTK simulation profile
        if HAS_VTK and xh in profiles:
            p = profiles[xh]
            data_sim = p.get(field_sim)
            if data_sim is not None:
                ax.plot(data_sim * scale + xh, p["z_abs"], "-", color=c_sim, lw=1.0, zorder=5)
            ax.plot([xh, xh], [p["z_abs"][0], p["z_abs"][-1]],
                    "--", color="0.3", lw=0.3, zorder=2)
        else:
            # Draw baseline even without VTK
            ax.plot([xh, xh], [0, LZ / H_HILL],
                    "--", color="0.3", lw=0.3, zorder=2)

        # Benchmark scatter
        for _, info, bdata in bench_sources:
            if xh in bdata and field_bench in bdata[xh]:
                d_b = bdata[xh][field_bench]
                if d_b is None:
                    continue
                z_b = bdata[xh]["y"]
                ax.scatter(d_b * scale + xh, z_b, s=info['markersize']**2,
                           facecolors="none", edgecolors=info['color'],
                           linewidths=0.4, zorder=6, marker=info['marker'])

    if xlim_range is not None:
        ax.set_xlim(xlim_range)
        ax.set_xticks(range(int(np.floor(xlim_range[0])), int(np.ceil(xlim_range[1])) + 1))
    else:
        ax.set_xticks(range(10))
        ax.set_xlim(0, 9)
    ax.set_ylim(0, LZ / H_HILL)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12, pad=4)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r"$y\,/\,h$", fontsize=11)


def compute_offset_extent(field_sim, field_bench, scale, padding=0.3):
    """Compute the x-axis data extent for an offset profile panel."""
    x_min = min(ALL_XH_STATIONS)
    x_max = max(ALL_XH_STATIONS)
    for xh in ALL_XH_STATIONS:
        if HAS_VTK and xh in profiles:
            p = profiles[xh]
            data_sim = p.get(field_sim)
            if data_sim is not None:
                vals = data_sim * scale + xh
                x_min = min(x_min, float(np.min(vals)))
                x_max = max(x_max, float(np.max(vals)))
        for _, info, bdata in bench_sources:
            if xh in bdata and field_bench in bdata[xh]:
                d_b = bdata[xh][field_bench]
                if d_b is not None:
                    vals = d_b * scale + xh
                    x_min = min(x_min, float(np.min(vals)))
                    x_max = max(x_max, float(np.max(vals)))
    return x_min - padding, x_max + padding


# ================================================================
# Figure 1: U offset profile
# ================================================================
xlim_plot = (-1, 10)
fig1_width = max(10.0, 10.0 * (xlim_plot[1] - xlim_plot[0]) / 9.0)
fig1, ax1 = plt.subplots(figsize=(fig1_width, 4.0))
if LAMINAR:
    u_title = r"$U / U_{\mathrm{ref}}$  (Re = %d, laminar)" % Re
    u_xlabel = r"$x\,/\,h$"
else:
    u_title = r"$\langle U \rangle / U_b$  (Re = %d)" % Re
    u_xlabel = r"$x\,/\,h$"
plot_offset_panel(ax1, "U", "U", scale=0.8,
                  title=u_title, xlabel=u_xlabel, xlim_range=xlim_plot)
ncol_leg = min(len(bench_sources) + 1, 4)
ax1.legend(handles=make_legend_elements(), loc="lower center", frameon=True,
           edgecolor="0.7", fancybox=False, ncol=ncol_leg, fontsize=9,
           bbox_to_anchor=(0.5, -0.28))
fig1.tight_layout()
out1 = os.path.join(SCRIPT_DIR, f"benchmark_Umean_Re{Re}.png")
fig1.savefig(out1, bbox_inches="tight")
print(f"\n[OK] Saved: {os.path.basename(out1)}")
plt.close(fig1)

# ================================================================
# Figure 2: V (vertical/wall-normal) offset profile
# ================================================================
fig2, ax2 = plt.subplots(figsize=(fig1_width, 4.0))
if LAMINAR:
    v_title = r"$V / U_{\mathrm{ref}}$  (wall-normal, Re = %d, laminar)" % Re
else:
    v_title = r"$\langle V \rangle / U_b$  (wall-normal, Re = %d)" % Re
plot_offset_panel(ax2, "W", "V", scale=0.8,
                  title=v_title, xlabel=r"$x\,/\,h$", xlim_range=xlim_plot)
ax2.legend(handles=make_legend_elements(), loc="lower center", frameon=True,
           edgecolor="0.7", fancybox=False, ncol=ncol_leg, fontsize=9,
           bbox_to_anchor=(0.5, -0.28))
fig2.tight_layout()
out2 = os.path.join(SCRIPT_DIR, f"benchmark_Vmean_Re{Re}.png")
fig2.savefig(out2, bbox_inches="tight")
print(f"[OK] Saved: {os.path.basename(out2)}")
plt.close(fig2)

# ================================================================
# Figure 3: RS + k offset profiles (5 panels) — turbulent only
# ================================================================
if LAMINAR:
    print("[INFO] Laminar mode — skipping RS comparison plot.")
elif HAS_RS:
    panels = [
        ("uu", "uu", 30, r"$\langle u^\prime u^\prime \rangle / U_b^2$"),
        ("ww", "vv", 30, r"$\langle v^\prime v^\prime \rangle / U_b^2$  (wall-normal)"),
        ("uw", "uv", 60, r"$\langle u^\prime v^\prime \rangle / U_b^2$  (shear stress)"),
        ("k",  "k",  20, r"$k / U_b^2$  (TKE)"),
        ("W",  "V",   3, r"$\langle V \rangle / U_b$  (wall-normal mean)"),
    ]
    xlim_rs = (-1, 10)
    x_range_rs = xlim_rs[1] - xlim_rs[0]
    fig3_rs_width = max(18.0, 18.0 * x_range_rs / 9.0)

    fig3_rs, axes = plt.subplots(3, 2, figsize=(fig3_rs_width, 12))
    axes_flat = axes.flatten()
    for idx, (fs, fb, sc, ttl) in enumerate(panels):
        plot_offset_panel(axes_flat[idx], fs, fb, scale=sc, title=ttl,
                          xlabel=r"$x\,/\,h$", xlim_range=xlim_rs)
    # Place legend in the empty 6th cell instead of wasting bottom space
    ax_leg = axes_flat[-1]
    ax_leg.axis("off")
    ax_leg.legend(handles=make_legend_elements(), loc="center", frameon=True,
                  edgecolor="0.7", fancybox=False,
                  ncol=1, fontsize=11)
    fig3_rs.suptitle(f"Periodic Hill Re = {Re} \u2014 Reynolds Stress Comparison",
                     fontsize=15, y=0.98)
    fig3_rs.tight_layout(rect=[0, 0, 1, 0.96])
    out3_rs = os.path.join(SCRIPT_DIR, f"benchmark_RS_Re{Re}.png")
    fig3_rs.savefig(out3_rs, bbox_inches="tight")
    print(f"[OK] Saved: {os.path.basename(out3_rs)}")
    plt.close(fig3_rs)
else:
    print("[INFO] No RS data in VTK \u2014 skipping RS comparison plot.")

