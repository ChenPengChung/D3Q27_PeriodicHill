"""
Periodic Hill Curvilinear Mesh <-> Computational Space Mapping Visualization
===========================================================================
  Left panel  -- Physical Space (y, z): curvilinear mesh with UNIFORM
                 D2Q9 velocity vectors (standard lattice velocities e_alpha)
  Right panel -- Computational Space (xi, zeta): uniform square grid with
                 IRREGULAR D2Q9 velocity vectors e_tilde = J . e_alpha
                 (Jacobian-distorted, varying length & angle at each node)

Physics: The D2Q9 lattice velocities are defined in Cartesian (physical)
space as unit-step vectors on a regular lattice -> uniform arrows.
When transformed to computational space via the Jacobian J = d(xi,zeta)/d(y,z),
the arrows become distorted depending on local cell shape.

Hill profile: exact ERCOFTAC piecewise cubic (Mellen et al. 2000),
              ported verbatim from model.h
Stretching:   tanh wall-normal clustering matching initializationTool.h
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os

# ======================================================================
# 1. Exact ERCOFTAC Hill Profile (from model.h)
# ======================================================================

def hill_function(Y, LY=9.0):
    """Exact piecewise cubic hill profile from model.h (Mellen et al. 2000)."""
    Y = np.asarray(Y, dtype=float)
    scalar = Y.ndim == 0
    Y = np.atleast_1d(Y).copy()
    Y = np.where(Y < 0, Y + LY, Y)
    Y = np.where(Y > LY, Y - LY, Y)
    model = np.zeros_like(Y)

    Yb = Y; t = Yb * 28.0

    seg1 = Yb <= (54.0/28.0)*(9.0/54.0)
    model = np.where(seg1,
        (1.0/28.0)*np.minimum(28.0, 28.0
            + 0.006775070969851*t*t
            - 0.0021245277758000*t*t*t), model)

    seg2 = (Yb > (54.0/28.0)*(9.0/54.0)) & (Yb <= (54.0/28.0)*(14.0/54.0))
    model = np.where(seg2,
        1.0/28.0*(25.07355893131 + 0.9754803562315*t
            - 0.1016116352781*t*t + 0.001889794677828*t*t*t), model)

    seg3 = (Yb > (54.0/28.0)*(14.0/54.0)) & (Yb <= (54.0/28.0)*(20.0/54.0))
    model = np.where(seg3,
        1.0/28.0*(25.79601052357 + 0.8206693007457*t
            - 0.09055370274339*t*t + 0.001626510569859*t*t*t), model)

    seg4 = (Yb > (54.0/28.0)*(20.0/54.0)) & (Yb <= (54.0/28.0)*(30.0/54.0))
    model = np.where(seg4,
        1.0/28.0*(40.46435022819 - 1.379581654948*t
            + 0.019458845041284*t*t - 0.0002070318932190*t*t*t), model)

    seg5 = (Yb > (54.0/28.0)*(30.0/54.0)) & (Yb <= (54.0/28.0)*(40.0/54.0))
    model = np.where(seg5,
        1.0/28.0*(17.92461334664 + 0.8743920332081*t
            - 0.05567361123058*t*t + 0.0006277731764683*t*t*t), model)

    seg6 = (Yb > (54.0/28.0)*(40.0/54.0)) & (Yb <= (54.0/28.0)*(54.0/54.0))
    model = np.where(seg6,
        1.0/28.0*np.maximum(0.0, 56.39011190988 - 2.010520359035*t
            + 0.01644919857549*t*t + 0.00002674976141766*t*t*t), model)

    # Right hill (mirror-symmetric)
    Yr = LY - Y; tr = Yr * 28.0
    rseg = (Y >= LY - (54.0/28.0))

    model = np.where(rseg & (Yr <= (54.0/28.0)*(9.0/54.0)),
        (1.0/28.0)*np.minimum(28.0, 28.0
            + 0.006775070969851*tr*tr - 0.0021245277758000*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(9.0/54.0)) & (Yr <= (54.0/28.0)*(14.0/54.0)),
        1.0/28.0*(25.07355893131 + 0.9754803562315*tr
            - 0.1016116352781*tr*tr + 0.001889794677828*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(14.0/54.0)) & (Yr <= (54.0/28.0)*(20.0/54.0)),
        1.0/28.0*(25.79601052357 + 0.8206693007457*tr
            - 0.09055370274339*tr*tr + 0.001626510569859*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(20.0/54.0)) & (Yr <= (54.0/28.0)*(30.0/54.0)),
        1.0/28.0*(40.46435022819 - 1.379581654948*tr
            + 0.019458845041284*tr*tr - 0.0002070318932190*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(30.0/54.0)) & (Yr <= (54.0/28.0)*(40.0/54.0)),
        1.0/28.0*(17.92461334664 + 0.8743920332081*tr
            - 0.05567361123058*tr*tr + 0.0006277731764683*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(40.0/54.0)) & (Yr <= (54.0/28.0)*(54.0/54.0)),
        1.0/28.0*np.maximum(0.0, 56.39011190988 - 2.010520359035*tr
            + 0.01644919857549*tr*tr + 0.00002674976141766*tr*tr*tr), model)

    return model.item() if scalar else model


# ======================================================================
# 2. Tanh Stretching (from initializationTool.h)
# ======================================================================

def tanh_stretch(L, a, j, N):
    ratio = np.log((1.0 + a) / (1.0 - a))
    return L/2.0 + (L/2.0/a)*np.tanh((-1.0 + 2.0*j/N)/2.0 * ratio)

def find_stretch_param(L, N, target_dz):
    a_lo, a_hi = 1e-6, 1.0 - 1e-6
    for _ in range(200):
        a_mid = (a_lo + a_hi) / 2.0
        dz_first = tanh_stretch(L, a_mid, 1, N) - tanh_stretch(L, a_mid, 0, N)
        if dz_first < target_dz:
            a_hi = a_mid
        else:
            a_lo = a_mid
    return (a_lo + a_hi) / 2.0


# ======================================================================
# 3. Grid Generation
# ======================================================================

def generate_grid(Ny, Nz, LY=9.0, LZ=3.036, H_HILL=1.0):
    y_1d = np.linspace(0, LY, Ny)
    z_bottom = hill_function(y_1d)
    total_max = LZ - 0.0
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


# ======================================================================
# 4. D2Q9 Velocity Set & Jacobian Transform
# ======================================================================

# D2Q9 lattice velocities -- defined in PHYSICAL Cartesian space
e_cart = np.array([
    [0,  0],    # 0: rest
    [1,  0],    # 1: +y
    [0,  1],    # 2: +z
    [-1, 0],    # 3: -y
    [0, -1],    # 4: -z
    [1,  1],    # 5: +y+z
    [-1, 1],    # 6: -y+z
    [-1,-1],    # 7: -y-z
    [1, -1],    # 8: +y-z
], dtype=float)


def compute_jacobian(Y, Z, i, j):
    """Compute forward Jacobian J = d(xi,zeta)/d(y,z) at grid node (i,j)."""
    Ny, Nz = Y.shape
    im, ip = max(i-1, 0), min(i+1, Ny-1)
    jm, jp = max(j-1, 0), min(j+1, Nz-1)

    dy_dxi   = (Y[ip, j] - Y[im, j]) / float(ip - im)
    dz_dxi   = (Z[ip, j] - Z[im, j]) / float(ip - im)
    dy_dzeta = (Y[i, jp] - Y[i, jm]) / float(jp - jm)
    dz_dzeta = (Z[i, jp] - Z[i, jm]) / float(jp - jm)

    J_inv = np.array([[dy_dxi, dy_dzeta],
                      [dz_dxi, dz_dzeta]])
    J = np.linalg.inv(J_inv)
    return J


def compute_contravariant_velocities(Y, Z, i, j):
    """Transform D2Q9 physical velocities -> contravariant (computational).
    e_tilde_alpha = J . e_alpha  (varying magnitude & direction)
    """
    J = compute_jacobian(Y, Z, i, j)
    e_contra = np.zeros((9, 2))
    for alpha in range(9):
        e_contra[alpha] = J @ e_cart[alpha]
    return e_contra


# ======================================================================
# 5. Figure Construction
# ======================================================================

def main():
    # --- Grid parameters ---
    Ny = 25
    Nz = 17
    LY = 9.0
    LZ = 3.036

    Y, Z = generate_grid(Ny, Nz, LY=LY, LZ=LZ)

    # --- Select 5 representative nodes ---
    nodes = [
        ('A', 3,   1,  '#2166AC'),  # Hill slope left (near wall), blue
        ('B', 8,   1,  '#1B7837'),  # Valley floor (near wall), green
        ('C', 11,  8,  '#762A83'),  # Mid-channel, purple
        ('D', 20,  1,  '#E08214'),  # Right hill slope (near wall), orange
        ('E', 9,  14,  '#C51B7D'),  # Near top wall, magenta
    ]

    # --- Figure setup ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.0,
    })

    fig = plt.figure(figsize=(20, 9))
    ax_phys = fig.add_axes([0.04, 0.08, 0.43, 0.84])   # left panel
    ax_comp = fig.add_axes([0.55, 0.08, 0.43, 0.84])    # right panel

    # ===================================
    # LEFT PANEL: Physical Space (y, z)
    #   Curvilinear grid + UNIFORM D2Q9 arrows
    # ===================================

    ax_phys.set_facecolor('#F2F2F2')

    # Grid lines
    for j in range(Nz):
        ax_phys.plot(Y[:, j], Z[:, j], color='#CCCCCC', linewidth=0.4, zorder=1)
    for i in range(Ny):
        ax_phys.plot(Y[i, :], Z[i, :], color='#CCCCCC', linewidth=0.4, zorder=1)

    # Hill wall
    y_fine = np.linspace(0, LY, 500)
    z_wall = hill_function(y_fine)
    ax_phys.fill_between(y_fine, 0, z_wall, color='#D8D8D8', zorder=2)
    ax_phys.plot(y_fine, z_wall, color='black', linewidth=2.0, zorder=5)
    ax_phys.axhline(y=LZ, color='black', linewidth=1.0, zorder=5)

    # UNIFORM D2Q9 arrows -- same at every node (standard lattice velocities)
    arrow_len_phys = 0.30  # uniform length in axis units

    for (label, ni, nj, color) in nodes:
        yc, zc = Y[ni, nj], Z[ni, nj]

        # Draw 8 uniform arrows (skip rest direction alpha=0)
        for alpha in range(1, 9):
            ey, ez = e_cart[alpha]
            norm = np.sqrt(ey**2 + ez**2)
            dy_arr = ey / norm * arrow_len_phys
            dz_arr = ez / norm * arrow_len_phys
            ax_phys.annotate('',
                xy=(yc + dy_arr, zc + dz_arr),
                xytext=(yc, zc),
                arrowprops=dict(arrowstyle='->', color='#CC0000',
                                lw=0.8, mutation_scale=8),
                zorder=10)

        # Node marker
        ax_phys.plot(yc, zc, 'o', color=color, markersize=8,
                     markeredgecolor='black', markeredgewidth=0.6, zorder=15)

        # Label (bold, color-matched)
        ax_phys.annotate(label, (yc, zc), textcoords='offset points',
                         xytext=(12, 10), fontsize=11, fontweight='bold',
                         color=color, zorder=15,
                         path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    ax_phys.set_xlabel(r'$y\,/\,h$', fontsize=14)
    ax_phys.set_ylabel(r'$z\,/\,h$', fontsize=14)
    ax_phys.set_title(r'Physical Space $(y,\, z)$', fontsize=16, pad=10)
    ax_phys.set_xlim(-0.3, LY + 0.3)
    ax_phys.set_ylim(-0.15, LZ + 0.25)
    ax_phys.set_aspect('equal')
    ax_phys.tick_params(labelsize=11)

    # ===================================
    # RIGHT PANEL: Computational Space (xi, zeta)
    #   Uniform square grid + IRREGULAR D2Q9 arrows (e_tilde = J . e)
    # ===================================

    ax_comp.set_facecolor('white')

    # Uniform grid lines
    for j in range(Nz):
        ax_comp.plot([0, Ny-1], [j, j], color='#CCCCCC', linewidth=0.4, zorder=1)
    for i in range(Ny):
        ax_comp.plot([i, i], [0, Nz-1], color='#CCCCCC', linewidth=0.4, zorder=1)

    # Domain boundary
    ax_comp.plot([0, Ny-1, Ny-1, 0, 0], [0, 0, Nz-1, Nz-1, 0],
                 color='black', linewidth=1.0, zorder=5)

    # Compute global max |e_tilde| for consistent scaling
    all_max_e = []
    for (label, ni, nj, color) in nodes:
        e_contra = compute_contravariant_velocities(Y, Z, ni, nj)
        magnitudes = np.array([np.linalg.norm(e_contra[a]) for a in range(1, 9)])
        all_max_e.append(magnitudes.max())
    global_max_e = max(all_max_e)
    arrow_scale = 1.80 / global_max_e  # largest arrow ~ 1.80 grid spacings (enlarged for visibility)

    for (label, ni, nj, color) in nodes:
        xi_c, zeta_c = float(ni), float(nj)

        # Compute contravariant (Jacobian-transformed) velocities
        e_contra = compute_contravariant_velocities(Y, Z, ni, nj)
        magnitudes = np.array([np.linalg.norm(e_contra[a]) for a in range(1, 9)])
        local_max = magnitudes.max()

        # Draw IRREGULAR arrows (varying length & angle)
        for alpha in range(1, 9):
            dxi   = e_contra[alpha, 0] * arrow_scale
            dzeta = e_contra[alpha, 1] * arrow_scale
            ax_comp.annotate('',
                xy=(xi_c + dxi, zeta_c + dzeta),
                xytext=(xi_c, zeta_c),
                arrowprops=dict(arrowstyle='->', color='#CC0000',
                                lw=0.8, mutation_scale=8),
                zorder=10)

        # Node marker
        ax_comp.plot(xi_c, zeta_c, 'o', color=color, markersize=8,
                     markeredgecolor='black', markeredgewidth=0.6, zorder=15)

        # Label (bold, color-matched)
        ax_comp.annotate(label, (xi_c, zeta_c), textcoords='offset points',
                         xytext=(12, 10), fontsize=11, fontweight='bold',
                         color=color, zorder=15,
                         path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

        # |e_tilde|_max below label
        ax_comp.annotate(
            r'$|\tilde{e}|_{\max}$' + f' = {local_max:.2f}',
            (xi_c, zeta_c), textcoords='offset points',
            xytext=(12, -8), fontsize=9, color=color, zorder=15,
            path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

    ax_comp.set_xlabel(r'$\xi$', fontsize=14)
    ax_comp.set_ylabel(r'$\zeta$', fontsize=14)
    ax_comp.set_title(r'Computational Space $(\xi,\, \zeta)$', fontsize=16, pad=10)
    ax_comp.set_xlim(-1.5, Ny + 0.5)
    ax_comp.set_ylim(-1.5, Nz + 0.5)
    ax_comp.set_aspect('equal')
    ax_comp.tick_params(labelsize=11)

    # ===================================
    # 中央文字（兩個面板之間的間隙）
    # ===================================
    # 不使用虛線箭頭 -- 僅標籤匹配 (A-E)

    # fig.text(0.495, 0.60, r'$\mathit{Coordinate\;\; Transformation}$',
    #          ha='center', va='center', fontsize=13, color='#444444')
    # fig.text(0.495, 0.53,
    #          r'$(y,\, z) \;\longleftrightarrow\; (\xi,\, \zeta)$',
    #          ha='center', va='center', fontsize=12, color='#333333')

    # ===================================
    # BOTTOM-LEFT: Transformation formula & correspondence
    # ===================================

    # formula_lines = (
    #     r'$\tilde{e}_\xi = \dfrac{\partial\xi}{\partial y}\,e_y'
    #     r' + \dfrac{\partial\xi}{\partial z}\,e_z$'
    #     '\n\n'
    #     r'$\tilde{e}_\zeta = \dfrac{\partial\zeta}{\partial y}\,e_y'
    #     r' + \dfrac{\partial\zeta}{\partial z}\,e_z$'
    #     '\n\n'
    #     r'$\tilde{\mathbf{e}}_\alpha = J \cdot \mathbf{e}_\alpha$'
    #     r'$,\quad J = \dfrac{\partial(\xi,\zeta)}{\partial(y,z)}$'
    # )

    # fig.text(0.495, 0.35, formula_lines,
    #          ha='center', va='center', fontsize=11, color='#333333',
    #          linespacing=1.6,
    #          bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAFAFA',
    #                    edgecolor='#BBBBBB', alpha=0.9))

    # # Correspondence note
    # fig.text(0.495, 0.15,
    #          'Node correspondence:  A \u2194 A,  B \u2194 B,  C \u2194 C,  D \u2194 D,  E \u2194 E',
    #          ha='center', va='center', fontsize=10, color='#666666',
    #          fontstyle='italic')
    # ===================================
    # Save
    # ===================================

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, 'curvilinear_mesh_mapping.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(out_dir, 'curvilinear_mesh_mapping.pdf'),
                bbox_inches='tight', facecolor='white')
    print("Saved: curvilinear_mesh_mapping.png (300 dpi)")
    print("Saved: curvilinear_mesh_mapping.pdf")
    plt.close(fig)


if __name__ == '__main__':
    main()
