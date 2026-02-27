"""
Periodic Hill Curvilinear Mesh ↔ Computational Space Mapping Visualization
===========================================================================
Replicates Imamura (2005) Fig. 1 concept for the periodic hill geometry:
  Left panel  — Physical space (y, z): curvilinear mesh with DISTORTED
                D2Q9 velocity vectors via J^{-1} transform (varying length
                and angle reflecting local cell size and stretching)
  Right panel — Computational space (ξ, ζ): uniform grid with STANDARD
                D2Q9 velocity vectors (uniform, axis-aligned)

Physics: GILBM solves the LBE in computational space where streaming is
uniform. To visualize what particles "see" in physical space, we apply
J^{-1} = ∂(y,z)/∂(ξ,ζ) to convert the standard lattice velocities.

Hill profile: exact ERCOFTAC piecewise cubic (Mellen et al. 2000),
              ported verbatim from model.h
Stretching:   tanh wall-normal clustering matching initializationTool.h
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ══════════════════════════════════════════════════════════════════════
# 1. Exact ERCOFTAC Hill Profile (from model.h)
# ══════════════════════════════════════════════════════════════════════

def hill_function(Y, LY=9.0):
    """Exact piecewise cubic hill profile from model.h (Mellen et al. 2000).
    Input:  Y — streamwise coordinate (scalar or array)
    Output: z_bottom — hill surface height
    """
    Y = np.asarray(Y, dtype=float)
    scalar = Y.ndim == 0
    Y = np.atleast_1d(Y).copy()

    # Periodic wrapping
    Y = np.where(Y < 0, Y + LY, Y)
    Y = np.where(Y > LY, Y - LY, Y)

    model = np.zeros_like(Y)

    # --- Left hill (6 cubic segments) ---
    Yb = Y  # alias for readability (matches model.h's Yb)
    t = Yb * 28.0  # scaled variable used in polynomial coefficients

    seg1 = Yb <= (54.0 / 28.0) * (9.0 / 54.0)
    model = np.where(seg1,
        (1.0 / 28.0) * np.minimum(28.0, 28.0
            + 0.006775070969851 * t * t
            - 0.0021245277758000 * t * t * t),
        model)

    seg2 = (Yb > (54.0 / 28.0) * (9.0 / 54.0)) & (Yb <= (54.0 / 28.0) * (14.0 / 54.0))
    model = np.where(seg2,
        1.0 / 28.0 * (25.07355893131
            + 0.9754803562315 * t
            - 0.1016116352781 * t * t
            + 0.001889794677828 * t * t * t),
        model)

    seg3 = (Yb > (54.0 / 28.0) * (14.0 / 54.0)) & (Yb <= (54.0 / 28.0) * (20.0 / 54.0))
    model = np.where(seg3,
        1.0 / 28.0 * (25.79601052357
            + 0.8206693007457 * t
            - 0.09055370274339 * t * t
            + 0.001626510569859 * t * t * t),
        model)

    seg4 = (Yb > (54.0 / 28.0) * (20.0 / 54.0)) & (Yb <= (54.0 / 28.0) * (30.0 / 54.0))
    model = np.where(seg4,
        1.0 / 28.0 * (40.46435022819
            - 1.379581654948 * t
            + 0.019458845041284 * t * t
            - 0.0002070318932190 * t * t * t),
        model)

    seg5 = (Yb > (54.0 / 28.0) * (30.0 / 54.0)) & (Yb <= (54.0 / 28.0) * (40.0 / 54.0))
    model = np.where(seg5,
        1.0 / 28.0 * (17.92461334664
            + 0.8743920332081 * t
            - 0.05567361123058 * t * t
            + 0.0006277731764683 * t * t * t),
        model)

    seg6 = (Yb > (54.0 / 28.0) * (40.0 / 54.0)) & (Yb <= (54.0 / 28.0) * (54.0 / 54.0))
    model = np.where(seg6,
        1.0 / 28.0 * np.maximum(0.0, 56.39011190988
            - 2.010520359035 * t
            + 0.01644919857549 * t * t
            + 0.00002674976141766 * t * t * t),
        model)

    # --- Right hill (mirror-symmetric, 6 cubic segments) ---
    Yr = LY - Y  # mirrored coordinate
    tr = Yr * 28.0

    rseg6 = (Yr >= 0) & (Yr <= (54.0 / 28.0) * (54.0 / 54.0)) & (Y >= LY - (54.0 / 28.0))
    model = np.where(rseg6 & (Yr <= (54.0 / 28.0) * (9.0 / 54.0)),
        (1.0 / 28.0) * np.minimum(28.0, 28.0
            + 0.006775070969851 * tr * tr
            - 0.0021245277758000 * tr * tr * tr),
        model)

    model = np.where(rseg6
        & (Yr > (54.0 / 28.0) * (9.0 / 54.0))
        & (Yr <= (54.0 / 28.0) * (14.0 / 54.0)),
        1.0 / 28.0 * (25.07355893131
            + 0.9754803562315 * tr
            - 0.1016116352781 * tr * tr
            + 0.001889794677828 * tr * tr * tr),
        model)

    model = np.where(rseg6
        & (Yr > (54.0 / 28.0) * (14.0 / 54.0))
        & (Yr <= (54.0 / 28.0) * (20.0 / 54.0)),
        1.0 / 28.0 * (25.79601052357
            + 0.8206693007457 * tr
            - 0.09055370274339 * tr * tr
            + 0.001626510569859 * tr * tr * tr),
        model)

    model = np.where(rseg6
        & (Yr > (54.0 / 28.0) * (20.0 / 54.0))
        & (Yr <= (54.0 / 28.0) * (30.0 / 54.0)),
        1.0 / 28.0 * (40.46435022819
            - 1.379581654948 * tr
            + 0.019458845041284 * tr * tr
            - 0.0002070318932190 * tr * tr * tr),
        model)

    model = np.where(rseg6
        & (Yr > (54.0 / 28.0) * (30.0 / 54.0))
        & (Yr <= (54.0 / 28.0) * (40.0 / 54.0)),
        1.0 / 28.0 * (17.92461334664
            + 0.8743920332081 * tr
            - 0.05567361123058 * tr * tr
            + 0.0006277731764683 * tr * tr * tr),
        model)

    model = np.where(rseg6
        & (Yr > (54.0 / 28.0) * (40.0 / 54.0))
        & (Yr <= (54.0 / 28.0) * (54.0 / 54.0)),
        1.0 / 28.0 * np.maximum(0.0, 56.39011190988
            - 2.010520359035 * tr
            + 0.01644919857549 * tr * tr
            + 0.00002674976141766 * tr * tr * tr),
        model)

    return model.item() if scalar else model


# ══════════════════════════════════════════════════════════════════════
# 2. Tanh Stretching (from initializationTool.h)
# ══════════════════════════════════════════════════════════════════════

def tanh_stretch(L, a, j, N):
    """Symmetric tanh wall stretching — matches tanhFunction_wall macro.
    Maps j ∈ [0, N] → physical coord ∈ [0, L].
    """
    ratio = np.log((1.0 + a) / (1.0 - a))
    return L / 2.0 + (L / 2.0 / a) * np.tanh((-1.0 + 2.0 * j / N) / 2.0 * ratio)


def find_stretch_param(L, N, target_dz):
    """Bisection to find stretching parameter a such that first cell ≈ target_dz.
    Matches GetNonuniParameter() logic in initializationTool.h.
    """
    a_lo, a_hi = 1e-6, 1.0 - 1e-6
    for _ in range(200):
        a_mid = (a_lo + a_hi) / 2.0
        dz_first = tanh_stretch(L, a_mid, 1, N) - tanh_stretch(L, a_mid, 0, N)
        if dz_first < target_dz:
            a_hi = a_mid
        else:
            a_lo = a_mid
    return (a_lo + a_hi) / 2.0


# ══════════════════════════════════════════════════════════════════════
# 3. Grid Generation
# ══════════════════════════════════════════════════════════════════════

def generate_grid(Ny, Nz, LY=9.0, LZ=3.036, H_HILL=1.0):
    """Body-fitted curvilinear grid matching initialization.h logic.
    Ny: streamwise points (including endpoints)
    Nz: wall-normal points (including walls)
    Returns Y[Ny, Nz], Z[Ny, Nz] in physical coordinates.
    """
    y_1d = np.linspace(0, LY, Ny)
    z_bottom = hill_function(y_1d)

    # Find stretching parameter for the minimum-height column (hill crest)
    # Use the geometry at the valley floor (max vertical span) for the stretch param
    # to match the code's single-a approach
    total_max = LZ - 0.0   # flat bottom total height
    # Target first-cell spacing scaled to this coarse grid
    # In the real code: NZ6-6 = 128, minSize = (LZ-1.0)/128*0.6 ≈ 0.00954
    # For coarse grid: scale proportionally
    CFL = 0.6
    min_size_coarse = (LZ - H_HILL) / (Nz - 1) * CFL
    a = find_stretch_param(total_max, Nz - 1, min_size_coarse)

    Y = np.zeros((Ny, Nz))
    Z = np.zeros((Ny, Nz))

    for i in range(Ny):
        total = LZ - z_bottom[i]
        Y[i, :] = y_1d[i]
        for j in range(Nz):
            Z[i, j] = z_bottom[i] + tanh_stretch(total, a, j, Nz - 1)

    return Y, Z


# ══════════════════════════════════════════════════════════════════════
# 4. D2Q9 Velocity Set & Jacobian Transform
# ══════════════════════════════════════════════════════════════════════

# D2Q9 lattice velocities in PHYSICAL (Cartesian) space:
#   These are the actual particle velocities — axis-aligned, uniform magnitude.
#   In physical space they form a standard square lattice pattern.
e_cart = np.array([
    [0,  0],   # 0: rest
    [1,  0],   # 1: +y
    [0,  1],   # 2: +z
    [-1, 0],   # 3: -y
    [0, -1],   # 4: -z
    [1,  1],   # 5: +y+z
    [-1, 1],   # 6: -y+z
    [-1,-1],   # 7: -y-z
    [1, -1],   # 8: +y-z
], dtype=float)


def compute_jacobian(Y, Z, i, j):
    """Compute forward Jacobian J = ∂(ξ,ζ)/∂(y,z) at grid node (i, j).

    J transforms Cartesian velocities → contravariant (computational) velocities:
        ẽ_α = J · e_α

    J^{-1} = [[dy/dξ, dy/dζ],     is computed from central differences on the
              [dz/dξ, dz/dζ]]     physical grid, then inverted to get J.

    For our geometry (y uniform, z body-fitted):
        J = [[1/dy,           0          ],
             [-(dz/dξ)/(dy·dz/dζ),  1/(dz/dζ)]]
    """
    Ny, Nz = Y.shape
    # ∂/∂ξ (streamwise index i)
    im = max(i - 1, 0)
    ip = min(i + 1, Ny - 1)
    dy_dxi = (Y[ip, j] - Y[im, j]) / float(ip - im)
    dz_dxi = (Z[ip, j] - Z[im, j]) / float(ip - im)

    # ∂/∂ζ (wall-normal index j)
    jm = max(j - 1, 0)
    jp = min(j + 1, Nz - 1)
    dy_dzeta = (Y[i, jp] - Y[i, jm]) / float(jp - jm)
    dz_dzeta = (Z[i, jp] - Z[i, jm]) / float(jp - jm)

    # J^{-1}
    J_inv = np.array([[dy_dxi, dy_dzeta],
                      [dz_dxi, dz_dzeta]])
    # J = (J^{-1})^{-1}
    J = np.linalg.inv(J_inv)
    return J


def compute_contravariant_velocities(Y, Z, i, j):
    """Transform D2Q9 Cartesian velocities → contravariant velocities at node (i,j).

    Physical space:      e_α = standard D2Q9 (Cartesian, uniform)
    Computational space:  ẽ_α = J · e_α  (varying magnitude & direction)

    Contravariant velocities are longer where physical cells are small
    (near walls: large dk/dz) and tilted on hill slopes (dk/dy ≠ 0).
    """
    J = compute_jacobian(Y, Z, i, j)
    e_contra = np.zeros((9, 2))
    for alpha in range(9):
        e_contra[alpha] = J @ e_cart[alpha]
    return e_contra


def compute_physical_velocities(Y, Z, i, j):
    """Physical-space displacement vectors for each D2Q9 direction.

    e_phys[α] = J^{-1} · e_α  where J^{-1} = ∂(y,z)/∂(ξ,ζ)

    For standard D2Q9 direction e_α (unit steps in ξ,ζ), this gives the
    actual physical displacement (Δy, Δz) per computational time step.
    Near walls where physical cells are small, ζ-direction arrows shrink.
    On slopes where dk/dy ≠ 0, arrows tilt.
    """
    J = compute_jacobian(Y, Z, i, j)
    J_inv = np.linalg.inv(J)
    e_phys = np.zeros((9, 2))
    for alpha in range(9):
        e_phys[alpha] = J_inv @ e_cart[alpha]
    return e_phys


# ══════════════════════════════════════════════════════════════════════
# 5. Figure Construction
# ══════════════════════════════════════════════════════════════════════

def main():
    # --- Grid parameters ---
    Ny = 25   # streamwise (coarse for clarity)
    Nz = 17   # wall-normal
    LY = 9.0
    LZ = 3.036

    Y, Z = generate_grid(Ny, Nz, LY=LY, LZ=LZ)

    # --- Select 5 representative nodes ---
    # Format: (label, i_index, j_index, color)
    # Positions match user specifications for physical coordinates.
    nodes = [
        ('A', 3,   5,  '#2166AC'),  # Hill slope left, y≈1, z≈1
        ('B', 8,   1,  '#1B7837'),  # Valley floor, y≈3, z≈0
        ('C', 11,  8,  '#762A83'),  # Mid-channel above valley, y≈4, z≈1.5
        ('D', 20,  1,  '#E08214'),  # Right hill crest, y≈7.5, z≈0.1
        ('E', 9,  14,  '#C51B7D'),  # Near top wall, y≈3.5, z≈2.8
    ]

    # --- Figure setup ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.0,
    })

    fig = plt.figure(figsize=(20, 9))

    # Layout: two panels side by side with gap for mapping arrows
    ax_phys = fig.add_axes([0.04, 0.08, 0.42, 0.84])   # left panel
    ax_comp = fig.add_axes([0.56, 0.08, 0.42, 0.84])    # right panel

    # ═══════════════════════════════════════
    # LEFT PANEL: Physical Space (y, z)
    # ═══════════════════════════════════════

    # Light gray domain fill
    ax_phys.fill_between([0, LY], 0, LZ, color='#F0F0F0', zorder=0)

    # Grid lines: constant-ζ lines (horizontal-ish)
    for j in range(Nz):
        ax_phys.plot(Y[:, j], Z[:, j], color='#CCCCCC', linewidth=0.4, zorder=1)
    # Grid lines: constant-ξ lines (vertical-ish)
    for i in range(Ny):
        ax_phys.plot(Y[i, :], Z[i, :], color='#CCCCCC', linewidth=0.4, zorder=1)

    # Bottom wall (thick black) + hill fill
    y_fine = np.linspace(0, LY, 500)
    z_wall = hill_function(y_fine)
    ax_phys.fill_between(y_fine, 0, z_wall, color='#D8D8D8', zorder=2)
    ax_phys.plot(y_fine, z_wall, color='black', linewidth=2.0, zorder=5)
    # Top wall (thinner)
    ax_phys.axhline(y=LZ, color='black', linewidth=1.0, zorder=5)

    # ── Physical space: J^{-1}·e arrows showing metric distortion ──
    # Each node shows 8 arrows representing the physical displacement
    # for each D2Q9 lattice direction: e_phys[α] = J^{-1}·e_α
    # Near walls: ζ-direction arrows shrink (compressed cells),
    # on slopes: arrows tilt (dk/dy ≠ 0).  Per-node normalization
    # ensures all 8 arrows are visible; |ẽ|_max is annotated.
    star_radius_phys = 0.50  # max arrow length in physical coords (y/h)

    for (label, ni, nj, color) in nodes:
        yc, zc = Y[ni, nj], Z[ni, nj]
        e_phys = compute_physical_velocities(Y, Z, ni, nj)

        # Per-node normalization
        lengths = [np.linalg.norm(e_phys[alpha]) for alpha in range(1, 9)]
        local_max = max(lengths)
        local_scale = star_radius_phys / local_max if local_max > 0 else 1.0

        for alpha in range(1, 9):
            dy_arr = e_phys[alpha, 0] * local_scale
            dz_arr = e_phys[alpha, 1] * local_scale
            ax_phys.annotate('',
                xy=(yc + dy_arr, zc + dz_arr),
                xytext=(yc, zc),
                arrowprops=dict(arrowstyle='->', color='#CC0000',
                                lw=0.8, mutation_scale=8),
                zorder=10)

        # Node marker (size ~60 in scatter = markersize ~8 in plot)
        ax_phys.plot(yc, zc, 'o', color=color, markersize=8,
                     markeredgecolor='black', markeredgewidth=0.6, zorder=15)
        # Label with |ẽ|_max annotation
        ax_phys.annotate(
            f'{label}\n' + r'$|\tilde{{e}}|_{{\max}}$' + f' = {local_max:.2f}',
            (yc, zc), textcoords='offset points',
            xytext=(10, 10), fontsize=9, fontweight='bold',
            color=color, zorder=15,
            path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

    ax_phys.set_xlabel(r'$y / h$', fontsize=14)
    ax_phys.set_ylabel(r'$z / h$', fontsize=14)
    ax_phys.set_title(r'Physical Space $(y,\, z)$', fontsize=16, pad=10)
    ax_phys.set_xlim(-0.3, LY + 0.3)
    ax_phys.set_ylim(-0.15, LZ + 0.25)
    ax_phys.set_aspect('equal')
    ax_phys.tick_params(labelsize=11)

    # ═══════════════════════════════════════
    # RIGHT PANEL: Computational Space (ξ, ζ)
    # ═══════════════════════════════════════

    # Light gray domain fill
    ax_comp.fill_between([0, Ny - 1], 0, Nz - 1, color='#F0F0F0', zorder=0)

    # Uniform Cartesian grid
    for j in range(Nz):
        ax_comp.plot([0, Ny - 1], [j, j], color='#CCCCCC', linewidth=0.4, zorder=1)
    for i in range(Ny):
        ax_comp.plot([i, i], [0, Nz - 1], color='#CCCCCC', linewidth=0.4, zorder=1)

    # Domain boundary
    rect_y = [0, Ny - 1, Ny - 1, 0, 0]
    rect_z = [0, 0, Nz - 1, Nz - 1, 0]
    ax_comp.plot(rect_y, rect_z, color='black', linewidth=1.0, zorder=5)

    # ── Computational space: standard D2Q9 arrows (uniform) ──
    # All nodes show the SAME 8-pointed star: standard D2Q9 lattice
    # velocities in index space (ξ, ζ).  Scaled up for visibility.
    arrow_scale_comp = 0.75   # ~75% of grid spacing (= 1 in index units)

    for (label, ni, nj, color) in nodes:
        xi_c, zeta_c = float(ni), float(nj)

        for alpha in range(1, 9):
            dxi   = e_cart[alpha, 0] * arrow_scale_comp
            dzeta = e_cart[alpha, 1] * arrow_scale_comp
            ax_comp.annotate('',
                xy=(xi_c + dxi, zeta_c + dzeta),
                xytext=(xi_c, zeta_c),
                arrowprops=dict(arrowstyle='->', color='#CC0000',
                                lw=0.8, mutation_scale=8),
                zorder=10)

        # Node marker
        ax_comp.plot(xi_c, zeta_c, 'o', color=color, markersize=8,
                     markeredgecolor='black', markeredgewidth=0.6, zorder=15)
        # Label
        ax_comp.annotate(label, (xi_c, zeta_c), textcoords='offset points',
                         xytext=(10, 10), fontsize=11, fontweight='bold',
                         color=color, zorder=15,
                         path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

    ax_comp.set_xlabel(r'$\xi$', fontsize=14)
    ax_comp.set_ylabel(r'$\zeta$', fontsize=14)
    ax_comp.set_title(r'Computational Space $(\xi,\, \zeta)$', fontsize=16, pad=10)
    ax_comp.set_xlim(-1.5, Ny + 0.5)
    ax_comp.set_ylim(-1.5, Nz + 0.5)
    ax_comp.set_aspect('equal')
    ax_comp.tick_params(labelsize=11)

    # ═══════════════════════════════════════
    # MAPPING ARROWS between panels
    # ═══════════════════════════════════════

    for (label, ni, nj, color) in nodes:
        # Physical space point → figure coordinates
        phys_pt = ax_phys.transData.transform((Y[ni, nj], Z[ni, nj]))
        comp_pt = ax_comp.transData.transform((float(ni), float(nj)))

        # Convert to figure coordinates
        phys_fig = fig.transFigure.inverted().transform(phys_pt)
        comp_fig = fig.transFigure.inverted().transform(comp_pt)

        arrow = FancyArrowPatch(
            phys_fig, comp_fig,
            transform=fig.transFigure,
            arrowstyle='<->', color='#888888',
            linestyle='--', linewidth=0.8,
            mutation_scale=10, zorder=0,
            connectionstyle='arc3,rad=0.12',
            shrinkA=10, shrinkB=10,
        )
        fig.patches.append(arrow)

    # Central labels between panels — Coordinate Transformation + Jacobian formulas
    fig.text(0.50, 0.72, 'Coordinate Transformation',
             ha='center', va='center', fontsize=13, color='#555555',
             fontstyle='italic')
    fig.text(0.50, 0.64,
             r'$(y,\, z) \;\longleftrightarrow\; (\xi,\, \zeta)$',
             ha='center', va='center', fontsize=12, color='#333333')
    # Jacobian transformation formulas (simplified, no pmatrix)
    fig.text(0.50, 0.50,
             r'$\tilde{\mathbf{e}}_\alpha = J \cdot \mathbf{e}_\alpha$',
             ha='center', va='center', fontsize=11, color='#333333')
    fig.text(0.50, 0.40,
             r'$J = \left[\frac{\partial(\xi,\zeta)}{\partial(y,z)}\right]$',
             ha='center', va='center', fontsize=11, color='#333333')
    fig.text(0.50, 0.28,
             r'$\mathbf{e}_\alpha^{\mathrm{phys}} = J^{-1} \cdot \mathbf{e}_\alpha$',
             ha='center', va='center', fontsize=11, color='#333333')

    # ═══════════════════════════════════════
    # Save
    # ═══════════════════════════════════════

    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))

    fig.savefig(os.path.join(out_dir, 'curvilinear_mesh_mapping.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(out_dir, 'curvilinear_mesh_mapping.pdf'),
                bbox_inches='tight', facecolor='white')

    print(f"Saved: curvilinear_mesh_mapping.png (300 dpi)")
    print(f"Saved: curvilinear_mesh_mapping.pdf")
    plt.close(fig)


if __name__ == '__main__':
    main()
