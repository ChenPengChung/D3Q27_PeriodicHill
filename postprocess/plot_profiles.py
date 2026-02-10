#!/usr/bin/env python3
"""
Publication-quality velocity profile plots for Periodic Hill LBM-LES
Compares simulation results with Breuer et al. (2009) DNS reference data (Re=700)
from ERCOFTAC database (UFR 3-30, Case 1).

Usage:
    python plot_profiles.py --vtk ../result/velocity_merged_500001.vtk
    python plot_profiles.py --vtk ../result/velocity_merged_500001.vtk --save
    python plot_profiles.py --vtk-dir ../result/ --step 500001
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import os
import re
import argparse
import sys

# ============================================================================
# Configuration
# ============================================================================

# Physical parameters
LX = 4.5     # spanwise
LY = 9.0     # streamwise
LZ = 3.036   # wall-normal
H  = 1.0     # hill height

# Profile extraction positions (x/h)
PROFILE_POSITIONS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

# ERCOFTAC DNS reference data URLs for Re=700 (Case 1, LESOCC)
# Data columns: y/h, <U/Ub>, <V/Ub>, <u'u'/Ub^2>, <v'v'/Ub^2>, <u'v'/Ub^2>, <k/Ub^2>
REFERENCE_URLS = {
    '001': 'https://kbwiki-images.s3.amazonaws.com/1/14/UFR3-30_C_700_data_MB-001.dat',  # x/h=0.05
    '002': 'https://kbwiki-images.s3.amazonaws.com/d/d3/UFR3-30_C_700_data_MB-002.dat',  # x/h=0.5
    '003': 'https://kbwiki-images.s3.amazonaws.com/a/a1/UFR3-30_C_700_data_MB-003.dat',  # x/h=1.0
    '004': 'https://kbwiki-images.s3.amazonaws.com/6/6d/UFR3-30_C_700_data_MB-004.dat',  # x/h=2.0
    '005': 'https://kbwiki-images.s3.amazonaws.com/c/ce/UFR3-30_C_700_data_MB-005.dat',  # x/h=3.0
    '006': 'https://kbwiki-images.s3.amazonaws.com/4/4a/UFR3-30_C_700_data_MB-006.dat',  # x/h=4.0
    '007': 'https://kbwiki-images.s3.amazonaws.com/3/36/UFR3-30_C_700_data_MB-007.dat',  # x/h=5.0
    '008': 'https://kbwiki-images.s3.amazonaws.com/1/11/UFR3-30_C_700_data_MB-008.dat',  # x/h=6.0
    '009': 'https://kbwiki-images.s3.amazonaws.com/b/b4/UFR3-30_C_700_data_MB-009.dat',  # x/h=7.0
    '010': 'https://kbwiki-images.s3.amazonaws.com/d/d8/UFR3-30_C_700_data_MB-010.dat',  # x/h=8.0
}

# Map file index to x/h position
FILE_TO_XH = {
    '001': 0.05, '002': 0.5, '003': 1.0, '004': 2.0, '005': 3.0,
    '006': 4.0,  '007': 5.0, '008': 6.0, '009': 7.0, '010': 8.0,
}


# ============================================================================
# Hill geometry (same polynomial as in model.h)
# ============================================================================

def hill_function(y):
    """Compute hill height at streamwise position y (physical units).
    Replicates HillFunction in model.h exactly — uses independent if
    statements (not elif) so the flat region (model=0) is preserved.
    """
    Yb = y % LY
    model = 0.0

    # Boundary positions
    b1 = 54./28. * ( 9./54.)  # ≈ 0.3214
    b2 = 54./28. * (14./54.)  # = 0.5
    b3 = 54./28. * (20./54.)  # ≈ 0.7143
    b4 = 54./28. * (30./54.)  # ≈ 1.0714
    b5 = 54./28. * (40./54.)  # ≈ 1.4286
    b6 = 54./28. * (54./54.)  # ≈ 1.9286

    S = Yb * 28.0  # scaled coordinate for left hill

    # Left hill (independent if, same as model.h)
    if Yb <= b1:
        model = 1./28. * min(28., 28. + 0.006775070969851*S*S - 0.0021245277758*S*S*S)
    if Yb > b1 and Yb <= b2:
        model = 1./28. * (25.07355893131 + 0.9754803562315*S - 0.1016116352781*S*S + 0.001889794677828*S*S*S)
    if Yb > b2 and Yb <= b3:
        model = 1./28. * (25.79601052357 + 0.8206693007457*S - 0.09055370274339*S*S + 0.001626510569859*S*S*S)
    if Yb > b3 and Yb <= b4:
        model = 1./28. * (40.46435022819 - 1.379581654948*S + 0.019458845041284*S*S - 0.0002070318932190*S*S*S)
    if Yb > b4 and Yb <= b5:
        model = 1./28. * (17.92461334664 + 0.8743920332081*S - 0.05567361123058*S*S + 0.0006277731764683*S*S*S)
    if Yb > b5 and Yb <= b6:
        model = 1./28. * max(0., 56.39011190988 - 2.010520359035*S + 0.01644919857549*S*S + 0.00002674976141766*S*S*S)

    # Right hill (mirrored, independent if)
    R = (LY - Yb) * 28.0  # scaled coordinate for right hill
    if Yb < LY - b5 and Yb >= LY - b6:
        model = 1./28. * max(0., 56.39011190988 - 2.010520359035*R + 0.01644919857549*R*R + 0.00002674976141766*R*R*R)
    if Yb < LY - b4 and Yb >= LY - b5:
        model = 1./28. * (17.92461334664 + 0.8743920332081*R - 0.05567361123058*R*R + 0.0006277731764683*R*R*R)
    if Yb < LY - b3 and Yb >= LY - b4:
        model = 1./28. * (40.46435022819 - 1.379581654948*R + 0.019458845041284*R*R - 0.0002070318932190*R*R*R)
    if Yb < LY - b2 and Yb >= LY - b3:
        model = 1./28. * (25.79601052357 + 0.8206693007457*R - 0.09055370274339*R*R + 0.001626510569859*R*R*R)
    if Yb < LY - b1 and Yb >= LY - b2:
        model = 1./28. * (25.07355893131 + 0.9754803562315*R - 0.1016116352781*R*R + 0.001889794677828*R*R*R)
    if Yb >= LY - b1:
        model = 1./28. * min(28., 28. + 0.006775070969851*R*R - 0.0021245277758*R*R*R)

    return model


# ============================================================================
# VTK Reader
# ============================================================================

def read_vtk(filepath):
    """Read ASCII STRUCTURED_GRID VTK file.
    Returns coordinates (x, y, z) and velocity (u, v, w) arrays.

    In the simulation coordinate system:
    - x = spanwise  (LX = 4.5)
    - y = streamwise (LY = 9.0)
    - z = wall-normal (LZ = 3.036)

    So: streamwise velocity = v, wall-normal velocity = w
    """
    print(f"Reading VTK file: {filepath}")
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    idx = 0
    dims = None
    npoints = 0

    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith('DIMENSIONS'):
            dims = list(map(int, line.split()[1:4]))
            print(f"  Grid dimensions: {dims[0]} x {dims[1]} x {dims[2]}")
        elif line.startswith('POINTS'):
            parts = line.split()
            npoints = int(parts[1])
            idx += 1
            break
        idx += 1

    if dims is None:
        raise ValueError("Could not find DIMENSIONS in VTK file")

    nx, ny, nz = dims

    # Read coordinates
    coords = []
    while len(coords) < npoints * 3:
        parts = lines[idx].strip().split()
        coords.extend([float(p) for p in parts])
        idx += 1

    coords = np.array(coords).reshape(npoints, 3)
    x_coord = coords[:, 0]  # spanwise
    y_coord = coords[:, 1]  # streamwise
    z_coord = coords[:, 2]  # wall-normal

    # Find VECTORS velocity
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith('VECTORS'):
            idx += 1
            break
        idx += 1

    # Read velocity data
    vel = []
    while len(vel) < npoints * 3:
        parts = lines[idx].strip().split()
        vel.extend([float(p) for p in parts])
        idx += 1

    vel = np.array(vel).reshape(npoints, 3)
    u_vel = vel[:, 0]  # spanwise velocity
    v_vel = vel[:, 1]  # streamwise velocity
    w_vel = vel[:, 2]  # wall-normal velocity

    # Reshape to 3D: VTK STRUCTURED_GRID order is (i fastest, j, k slowest)
    # i=x(spanwise), j=y(streamwise), k=z(wall-normal)
    x_3d = x_coord.reshape(nz, ny, nx)
    y_3d = y_coord.reshape(nz, ny, nx)
    z_3d = z_coord.reshape(nz, ny, nx)
    u_3d = u_vel.reshape(nz, ny, nx)
    v_3d = v_vel.reshape(nz, ny, nx)
    w_3d = w_vel.reshape(nz, ny, nx)

    return x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, dims


def _extract_step(filename):
    """Extract timestep number from VTK filename."""
    m = re.search(r'velocity_merged_(\d+)\.vtk', str(filename))
    return int(m.group(1)) if m else 0


def _read_vtk_fast(filepath):
    """Fast VTK reader — only extracts velocity fields, skips coordinate parsing."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    idx = 0
    dims = None
    npoints = 0

    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith('DIMENSIONS'):
            dims = list(map(int, line.split()[1:4]))
            npoints = dims[0] * dims[1] * dims[2]
        elif line.startswith('POINTS'):
            idx += 1
            # Skip coordinate data (don't parse floats)
            coord_count = 0
            while coord_count < npoints * 3:
                parts = lines[idx].strip().split()
                coord_count += len(parts)
                idx += 1
            continue  # Don't increment idx again
        elif line.startswith('VECTORS'):
            idx += 1
            break
        idx += 1

    nx, ny, nz = dims

    # Read velocity data
    vel = []
    while len(vel) < npoints * 3:
        parts = lines[idx].strip().split()
        vel.extend([float(p) for p in parts])
        idx += 1

    vel = np.array(vel).reshape(npoints, 3)
    v_3d = vel[:, 1].reshape(nz, ny, nx)  # streamwise
    w_3d = vel[:, 2].reshape(nz, ny, nx)  # wall-normal

    return v_3d, w_3d, dims


def time_average_vtk(vtk_dir, start_step=None, end_step=None, every=1):
    """Compute time-averaged velocity fields from multiple VTK snapshots.

    Uses online (running) mean for memory efficiency.
    Also computes <v²>, <w²>, <vw> for Reynolds stress calculation:
        <v'v'> = <v²> - <v>²

    Parameters:
        vtk_dir:     Directory containing velocity_merged_*.vtk files
        start_step:  First timestep to include (skip initial transient)
        end_step:    Last timestep to include
        every:       Process every N-th file (default 1 = all)

    Returns:
        x_3d, y_3d, z_3d: coordinate arrays (from first file)
        mean_v, mean_w:    time-averaged velocity fields
        dims:              grid dimensions [nx, ny, nz]
        stress_vv, stress_ww, stress_vw: Reynolds stress fields
        n_samples:         number of snapshots averaged
    """
    vtk_path = Path(vtk_dir)
    all_files = sorted(vtk_path.glob('velocity_merged_*.vtk'))

    # Filter by step range
    selected = []
    for f in all_files:
        step = _extract_step(f)
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        selected.append(f)

    # Apply skip
    selected = selected[::every]

    if not selected:
        raise ValueError("No VTK files found in the specified range")

    n_total = len(selected)
    step_first = _extract_step(selected[0])
    step_last  = _extract_step(selected[-1])
    print(f"\n{'='*60}")
    print(f"Time averaging: {n_total} snapshots")
    print(f"  Step range : {step_first} → {step_last}")
    print(f"  Skip (every): {every}")
    print(f"{'='*60}")

    # ---- First file: full read (need coordinates) ----
    print(f"  [1/{n_total}] {selected[0].name} (full read)")
    x_3d, y_3d, z_3d, _, v_3d, w_3d, dims = read_vtk(str(selected[0]))

    # Initialize accumulators
    mean_v  = v_3d.copy()
    mean_w  = w_3d.copy()
    mean_vv = v_3d ** 2
    mean_ww = w_3d ** 2
    mean_vw = v_3d * w_3d

    # ---- Remaining files: fast read (velocity only) ----
    for n, vtk_file in enumerate(selected[1:], start=2):
        if n % 20 == 0 or n == n_total:
            pct = n / n_total * 100
            print(f"  [{n}/{n_total}] {vtk_file.name}  ({pct:.0f}%)")

        v_n, w_n, _ = _read_vtk_fast(str(vtk_file))

        # Welford online mean update: mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
        mean_v  += (v_n       - mean_v)  / n
        mean_w  += (w_n       - mean_w)  / n
        mean_vv += (v_n ** 2  - mean_vv) / n
        mean_ww += (w_n ** 2  - mean_ww) / n
        mean_vw += (v_n * w_n - mean_vw) / n

    # Reynolds stresses: <u'u'> = <u²> - <u>²
    stress_vv = mean_vv - mean_v ** 2
    stress_ww = mean_ww - mean_w ** 2
    stress_vw = mean_vw - mean_v * mean_w

    print(f"  Time averaging complete — {n_total} snapshots.")
    return (x_3d, y_3d, z_3d, mean_v, mean_w, dims,
            stress_vv, stress_ww, stress_vw, n_total)


def extract_profiles(y_3d, z_3d, v_3d, w_3d, dims, positions_xh,
                     stress_vv=None, stress_ww=None, stress_vw=None,
                     spanwise_avg=False):
    """Extract velocity profiles for comparison with benchmark data.

    Parameters:
        y_3d: streamwise coordinates (nz, ny, nx)
        z_3d: wall-normal coordinates (nz, ny, nx)
        v_3d: streamwise velocity (time-averaged or instantaneous)
        w_3d: wall-normal velocity
        dims: [nx, ny, nz]
        positions_xh: list of x/h positions to extract
        stress_vv: <v'v'> Reynolds stress field (optional)
        stress_ww: <w'w'> Reynolds stress field (optional)
        stress_vw: <v'w'> Reynolds stress field (optional)
        spanwise_avg: if True, average over spanwise direction;
                      otherwise use mid-plane slice

    Returns:
        dict of {x/h: {'y_h', 'U_Ub', 'V_Ub', ...}}, Ub
    """
    nx, ny, nz = dims
    has_stress = stress_vv is not None

    if spanwise_avg:
        print(f"  Using spanwise average (over {nx} points)")
        v_2d = np.mean(v_3d, axis=2)   # (nz, ny)
        w_2d = np.mean(w_3d, axis=2)
        z_2d = z_3d[:, :, 0]           # z invariant along spanwise
        if has_stress:
            svv_2d = np.mean(stress_vv, axis=2)
            sww_2d = np.mean(stress_ww, axis=2)
            svw_2d = np.mean(stress_vw, axis=2)
    else:
        i_mid = nx // 2
        print(f"  Extracting mid-plane at spanwise index i={i_mid}/{nx}")
        v_2d = v_3d[:, :, i_mid]
        w_2d = w_3d[:, :, i_mid]
        z_2d = z_3d[:, :, i_mid]
        if has_stress:
            svv_2d = stress_vv[:, :, i_mid]
            sww_2d = stress_ww[:, :, i_mid]
            svw_2d = stress_vw[:, :, i_mid]

    y_1d = y_3d[0, :, 0]  # streamwise positions

    # Compute bulk velocity above the hill surface
    v_valid = []
    for j in range(ny):
        y_pos = y_1d[j]
        h_val = hill_function(y_pos)
        for k in range(nz):
            z_val = z_2d[k, j]
            if z_val > h_val and z_val < LZ:
                v_valid.append(v_2d[k, j])

    Ub = np.mean(v_valid) if v_valid else 1.0
    print(f"  Bulk velocity Ub = {Ub:.6f}")

    profiles = {}
    for xh in positions_xh:
        y_target = xh * H

        j_idx = np.argmin(np.abs(y_1d - y_target))

        z_profile = z_2d[:, j_idx]
        v_profile = v_2d[:, j_idx]
        w_profile = w_2d[:, j_idx]

        h_local = hill_function(y_1d[j_idx])
        y_h = z_profile / H

        prof = {
            'y_h': y_h,
            'U_Ub': v_profile / Ub,
            'V_Ub': w_profile / Ub,
            'h_local': h_local,
        }

        if has_stress:
            prof['uu'] = svv_2d[:, j_idx] / Ub ** 2
            prof['vv'] = sww_2d[:, j_idx] / Ub ** 2
            prof['uv'] = svw_2d[:, j_idx] / Ub ** 2

        profiles[xh] = prof

    return profiles, Ub


# ============================================================================
# Reference data download
# ============================================================================

def download_reference_data(cache_dir='reference_data'):
    """Download Breuer et al. (2009) DNS data for Re=700 from ERCOFTAC."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    ref_data = {}
    for key, url in REFERENCE_URLS.items():
        xh = FILE_TO_XH[key]
        fname = cache_path / f"Re700_xh_{xh:.2f}.dat"

        if not fname.exists():
            print(f"  Downloading reference data for x/h={xh}...")
            try:
                urllib.request.urlretrieve(url, fname)
            except Exception as e:
                print(f"  Warning: Could not download {url}: {e}")
                continue

        if fname.exists():
            try:
                data = np.loadtxt(str(fname), comments='#')
                if data.ndim == 2 and data.shape[1] >= 3:
                    ref_data[xh] = {
                        'y_h': data[:, 0],
                        'U_Ub': data[:, 1],
                        'V_Ub': data[:, 2],
                    }
                    if data.shape[1] >= 6:
                        ref_data[xh]['uu'] = data[:, 3]
                        ref_data[xh]['vv'] = data[:, 4]
                        ref_data[xh]['uv'] = data[:, 5]
            except Exception as e:
                print(f"  Warning: Could not parse {fname}: {e}")

    return ref_data


# ============================================================================
# Publication-quality plotting
# ============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 1.2,
    })


def draw_hill(ax, y_range=(0, 9)):
    """Draw the periodic hill geometry (like Fig. 20 style)."""
    y_pts = np.linspace(y_range[0], y_range[1], 1000)
    z_pts = np.array([hill_function(y) for y in y_pts])

    # Fill hill region (light gray like Fig. 20)
    ax.fill_between(y_pts / H, 0, z_pts / H,
                     facecolor='#D0D0D0', edgecolor='none', zorder=1)

    # Draw hill outline (solid gray)
    ax.plot(y_pts / H, z_pts / H, color='gray', linewidth=1.2, zorder=2)

    # Draw bottom wall (y/h = 0)
    ax.axhline(y=0, color='gray', linewidth=0.8, zorder=2)


def plot_velocity_profiles(sim_profiles, ref_data, Ub, save=False, output_dir='.'):
    """Create publication-quality velocity profile comparison plot.

    Creates a two-panel figure like Fig. 20 in the reference paper:
    (a) Mean streamwise velocity <U>/Ub
    (b) Mean vertical velocity <V>/Ub
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Scaling factor for plotting (offset profiles at each x/h position)
    u_scale = 1.0  # How much to scale U/Ub for visual clarity
    v_scale = 4.0  # V/Ub is much smaller, scale up for visibility

    # === Panel (a): Streamwise velocity ===
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14,
             fontweight='bold', va='top')

    draw_hill(ax1)

    for xh in PROFILE_POSITIONS:
        # Plot simulation
        if xh in sim_profiles:
            prof = sim_profiles[xh]
            u_plot = xh + prof['U_Ub'] * u_scale
            ax1.plot(u_plot, prof['y_h'], 'b-', linewidth=1.2, zorder=3)

        # Plot reference (DNS by Breuer et al.)
        if xh in ref_data:
            ref = ref_data[xh]
            u_ref_plot = xh + ref['U_Ub'] * u_scale
            ax1.plot(u_ref_plot, ref['y_h'], 'r--', linewidth=1.0, zorder=3)

        # Draw vertical dashed line at each x/h
        ax1.axvline(x=xh, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    ax1.set_xlim(-0.5, 9.5)
    ax1.set_ylim(0, 3.2)
    ax1.set_ylabel(r'$y/h$')
    ax1.set_title(r'Mean streamwise velocity $\langle U \rangle / U_b$', fontsize=13)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=1.2, label='Present LBM-LES'),
        Line2D([0], [0], color='r', linewidth=1.0, linestyle='--',
               label='DNS, Breuer et al. (2009)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # === Panel (b): Wall-normal velocity ===
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14,
             fontweight='bold', va='top')

    draw_hill(ax2)

    for xh in PROFILE_POSITIONS:
        if xh in sim_profiles:
            prof = sim_profiles[xh]
            v_plot = xh + prof['V_Ub'] * v_scale
            ax2.plot(v_plot, prof['y_h'], 'b-', linewidth=1.2, zorder=3)

        if xh in ref_data:
            ref = ref_data[xh]
            v_ref_plot = xh + ref['V_Ub'] * v_scale
            ax2.plot(v_ref_plot, ref['y_h'], 'r--', linewidth=1.0, zorder=3)

        ax2.axvline(x=xh, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    ax2.set_xlim(-0.5, 9.5)
    ax2.set_ylim(0, 3.2)
    ax2.set_xlabel(r'$x/h$')
    ax2.set_ylabel(r'$y/h$')
    ax2.set_title(r'Mean wall-normal velocity $\langle V \rangle / U_b$', fontsize=13)

    plt.tight_layout()

    if save:
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True)
        fig.savefig(out_path / 'velocity_profiles_Re700.png', dpi=300, bbox_inches='tight')
        fig.savefig(out_path / 'velocity_profiles_Re700.pdf', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {out_path / 'velocity_profiles_Re700.png'}")
        print(f"Saved: {out_path / 'velocity_profiles_Re700.pdf'}")
    else:
        plt.show()

    plt.close()


def plot_reynolds_stresses(sim_profiles, ref_data, save=False, output_dir='.'):
    """Plot Reynolds stress profiles (reference + simulation if available)."""
    setup_plot_style()

    # Check if any data has stress components
    has_ref_stress = any('uu' in ref_data.get(xh, {}) for xh in PROFILE_POSITIONS)
    has_sim_stress = any('uu' in sim_profiles.get(xh, {}) for xh in PROFILE_POSITIONS)
    if not has_ref_stress and not has_sim_stress:
        print("No Reynolds stress data available, skipping stress plots.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    stress_scale = 20.0  # Scale factor for visibility

    for panel, (ax, key, title) in enumerate(zip(
        [ax1, ax2, ax3],
        ['uu', 'vv', 'uv'],
        [r"$\langle u'u' \rangle / U_b^2$",
         r"$\langle v'v' \rangle / U_b^2$",
         r"$\langle u'v' \rangle / U_b^2$"]
    )):
        label = chr(ord('a') + panel)
        ax.text(0.02, 0.95, f'({label})', transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='top')
        draw_hill(ax)

        for xh in PROFILE_POSITIONS:
            # Simulation stresses (blue solid)
            if xh in sim_profiles and key in sim_profiles[xh]:
                prof = sim_profiles[xh]
                val_plot = xh + prof[key] * stress_scale
                ax.plot(val_plot, prof['y_h'], 'b-', linewidth=1.2, zorder=3)

            # Reference stresses (red dashed)
            if xh in ref_data and key in ref_data[xh]:
                ref = ref_data[xh]
                val_plot = xh + ref[key] * stress_scale
                ax.plot(val_plot, ref['y_h'], 'r--', linewidth=1.0, zorder=3)

            ax.axvline(x=xh, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(0, 3.2)
        ax.set_ylabel(r'$y/h$')
        ax.set_title(f'Reynolds stress {title}', fontsize=13)

    # Legend on first panel
    from matplotlib.lines import Line2D
    legend_items = []
    if has_sim_stress:
        legend_items.append(Line2D([0], [0], color='b', linewidth=1.2,
                                   label='Present LBM-LES'))
    if has_ref_stress:
        legend_items.append(Line2D([0], [0], color='r', linewidth=1.0,
                                   linestyle='--', label='DNS, Breuer et al. (2009)'))
    if legend_items:
        ax1.legend(handles=legend_items, loc='upper right', framealpha=0.9)

    ax3.set_xlabel(r'$x/h$')
    plt.tight_layout()

    if save:
        out_path = Path(output_dir)
        fig.savefig(out_path / 'reynolds_stresses_Re700.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path / 'reynolds_stresses_Re700.png'}")

    plt.close()


def plot_single_profiles(sim_profiles, ref_data, save=False, output_dir='.'):
    """Plot individual profiles at each x/h position (detailed view)."""
    setup_plot_style()

    positions = [xh for xh in PROFILE_POSITIONS if xh in sim_profiles or xh in ref_data]
    if not positions:
        print("No profile data available for individual plots, skipping.")
        return
    ncols = 3
    nrows = (len(positions) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows), sharey=True)
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for idx, xh in enumerate(positions):
        ax = axes[idx]

        if xh in sim_profiles:
            prof = sim_profiles[xh]
            ax.plot(prof['U_Ub'], prof['y_h'], 'b-', linewidth=1.5,
                    label='LBM-LES')

        if xh in ref_data:
            ref = ref_data[xh]
            ax.plot(ref['U_Ub'], ref['y_h'], 'ro', markersize=3,
                    markerfacecolor='none', label='DNS (Breuer)')

        h_local = hill_function(xh * H)
        ax.axhline(y=h_local/H, color='gray', linestyle=':', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.5)

        ax.set_title(f'$x/h = {xh:.1f}$', fontsize=12)
        ax.set_xlabel(r'$\langle U \rangle / U_b$')
        if idx % ncols == 0:
            ax.set_ylabel(r'$y/h$')
        ax.set_ylim(0, 3.2)
        ax.set_xlim(-0.3, 1.5)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=9, loc='upper left')

    # Hide unused subplots
    for idx in range(len(positions), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(r'Streamwise velocity profiles, $Re = 700$', fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        out_path = Path(output_dir)
        fig.savefig(out_path / 'individual_profiles_Re700.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path / 'individual_profiles_Re700.png'}")

    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot velocity profiles for Periodic Hill LBM-LES')
    parser.add_argument('--vtk', type=str, help='Path to merged VTK file')
    parser.add_argument('--vtk-dir', type=str, default='../result',
                        help='Directory containing VTK files')
    parser.add_argument('--step', type=int, default=None,
                        help='Time step to plot (e.g., 500001)')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files instead of showing')
    parser.add_argument('--output-dir', type=str, default='./figures',
                        help='Output directory for saved figures')
    parser.add_argument('--ref-only', action='store_true',
                        help='Plot only reference data (no VTK needed)')
    parser.add_argument('--no-download', action='store_true',
                        help='Skip downloading reference data')

    # ---- Time-averaging options ----
    parser.add_argument('--time-avg', action='store_true',
                        help='Enable time averaging over multiple VTK files')
    parser.add_argument('--start-step', type=int, default=None,
                        help='First timestep to include (skip initial transient)')
    parser.add_argument('--end-step', type=int, default=None,
                        help='Last timestep to include')
    parser.add_argument('--every', type=int, default=1,
                        help='Process every N-th VTK file (default: 1 = all)')
    parser.add_argument('--spanwise-avg', action='store_true', default=False,
                        help='Average over spanwise direction '
                             '(auto-enabled with --time-avg)')

    args = parser.parse_args()

    # --time-avg implies --spanwise-avg
    spanwise_avg = args.spanwise_avg or args.time_avg

    # Download reference data
    ref_data = {}
    if not args.no_download:
        print("Downloading ERCOFTAC DNS reference data (Re=700)...")
        script_dir = Path(__file__).parent
        ref_data = download_reference_data(str(script_dir / 'reference_data'))
        print(f"  Loaded reference data at x/h = {sorted(ref_data.keys())}")

    # === Read simulation data ===
    sim_profiles = {}
    Ub = 1.0

    if args.ref_only:
        pass  # No simulation data

    elif args.time_avg:
        # ------ Time-averaged mode ------
        vtk_dir = args.vtk_dir
        if not Path(vtk_dir).exists():
            print(f"ERROR: VTK directory not found: {vtk_dir}")
            if not ref_data:
                sys.exit(1)
        else:
            (x_3d, y_3d, z_3d, mean_v, mean_w, dims,
             stress_vv, stress_ww, stress_vw, n_snap) = \
                time_average_vtk(vtk_dir,
                                 start_step=args.start_step,
                                 end_step=args.end_step,
                                 every=args.every)

            sim_profiles, Ub = extract_profiles(
                y_3d, z_3d, mean_v, mean_w, dims, PROFILE_POSITIONS,
                stress_vv=stress_vv,
                stress_ww=stress_ww,
                stress_vw=stress_vw,
                spanwise_avg=spanwise_avg)

    else:
        # ------ Single-snapshot mode ------
        vtk_file = None
        if args.vtk:
            vtk_file = args.vtk
        elif args.step:
            vtk_file = os.path.join(args.vtk_dir,
                                    f'velocity_merged_{args.step:06d}.vtk')

        if vtk_file:
            if not os.path.exists(vtk_file):
                print(f"ERROR: VTK file not found: {vtk_file}")
                if not ref_data:
                    sys.exit(1)
                print("Plotting reference data only...")
            else:
                x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, dims = read_vtk(vtk_file)
                sim_profiles, Ub = extract_profiles(
                    y_3d, z_3d, v_3d, w_3d, dims, PROFILE_POSITIONS,
                    spanwise_avg=spanwise_avg)
        else:
            # Try to find the latest VTK file
            vtk_dir = Path(args.vtk_dir)
            if vtk_dir.exists():
                vtk_files = sorted(vtk_dir.glob('velocity_merged_*.vtk'))
                if vtk_files:
                    vtk_file = str(vtk_files[-1])
                    print(f"Using latest VTK file: {vtk_file}")
                    x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, dims = read_vtk(
                        vtk_file)
                    sim_profiles, Ub = extract_profiles(
                        y_3d, z_3d, v_3d, w_3d, dims, PROFILE_POSITIONS,
                        spanwise_avg=spanwise_avg)
                else:
                    print("No VTK files found. Plotting reference data only.")

    # Create plots
    print("\nGenerating publication-quality plots...")

    # 1. Main comparison figure (like Fig. 20 in paper)
    plot_velocity_profiles(sim_profiles, ref_data, Ub,
                          save=args.save, output_dir=args.output_dir)

    # 2. Reynolds stress comparison
    plot_reynolds_stresses(sim_profiles, ref_data,
                          save=args.save, output_dir=args.output_dir)

    # 3. Individual detailed profiles
    plot_single_profiles(sim_profiles, ref_data,
                        save=args.save, output_dir=args.output_dir)

    if not args.save:
        print("\nUse --save to save figures to files.")

    print("Done!")


if __name__ == '__main__':
    main()
