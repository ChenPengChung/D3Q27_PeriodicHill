"""
Periodic Hill Curvilinear Mesh <-> Computational Space Mapping Visualization
===========================================================================
v4 – Global arrow normalization (proportional lengths) with overlap-aware
     node placement. Boundary nodes kept at j=1; inner nodes placed at
     higher j where arrows are shorter and won't clash.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import os

# ======================================================================
# Hill Profile
# ======================================================================

def hill_function(Y, LY=9.0):
    Y = np.asarray(Y, dtype=float)
    scalar = Y.ndim == 0
    Y = np.atleast_1d(Y).copy()
    Y = np.where(Y < 0, Y + LY, Y)
    Y = np.where(Y > LY, Y - LY, Y)
    model = np.zeros_like(Y)
    Yb = Y; t = Yb * 28.0
    seg1 = Yb <= (54.0/28.0)*(9.0/54.0)
    model = np.where(seg1, (1.0/28.0)*np.minimum(28.0, 28.0 + 0.006775070969851*t*t - 0.0021245277758000*t*t*t), model)
    seg2 = (Yb > (54.0/28.0)*(9.0/54.0)) & (Yb <= (54.0/28.0)*(14.0/54.0))
    model = np.where(seg2, 1.0/28.0*(25.07355893131 + 0.9754803562315*t - 0.1016116352781*t*t + 0.001889794677828*t*t*t), model)
    seg3 = (Yb > (54.0/28.0)*(14.0/54.0)) & (Yb <= (54.0/28.0)*(20.0/54.0))
    model = np.where(seg3, 1.0/28.0*(25.79601052357 + 0.8206693007457*t - 0.09055370274339*t*t + 0.001626510569859*t*t*t), model)
    seg4 = (Yb > (54.0/28.0)*(20.0/54.0)) & (Yb <= (54.0/28.0)*(30.0/54.0))
    model = np.where(seg4, 1.0/28.0*(40.46435022819 - 1.379581654948*t + 0.019458845041284*t*t - 0.0002070318932190*t*t*t), model)
    seg5 = (Yb > (54.0/28.0)*(30.0/54.0)) & (Yb <= (54.0/28.0)*(40.0/54.0))
    model = np.where(seg5, 1.0/28.0*(17.92461334664 + 0.8743920332081*t - 0.05567361123058*t*t + 0.0006277731764683*t*t*t), model)
    seg6 = (Yb > (54.0/28.0)*(40.0/54.0)) & (Yb <= (54.0/28.0)*(54.0/54.0))
    model = np.where(seg6, 1.0/28.0*np.maximum(0.0, 56.39011190988 - 2.010520359035*t + 0.01644919857549*t*t + 0.00002674976141766*t*t*t), model)
    Yr = LY - Y; tr = Yr * 28.0; rseg = (Y >= LY - (54.0/28.0))
    model = np.where(rseg & (Yr <= (54.0/28.0)*(9.0/54.0)), (1.0/28.0)*np.minimum(28.0, 28.0 + 0.006775070969851*tr*tr - 0.0021245277758000*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(9.0/54.0)) & (Yr <= (54.0/28.0)*(14.0/54.0)), 1.0/28.0*(25.07355893131 + 0.9754803562315*tr - 0.1016116352781*tr*tr + 0.001889794677828*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(14.0/54.0)) & (Yr <= (54.0/28.0)*(20.0/54.0)), 1.0/28.0*(25.79601052357 + 0.8206693007457*tr - 0.09055370274339*tr*tr + 0.001626510569859*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(20.0/54.0)) & (Yr <= (54.0/28.0)*(30.0/54.0)), 1.0/28.0*(40.46435022819 - 1.379581654948*tr + 0.019458845041284*tr*tr - 0.0002070318932190*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(30.0/54.0)) & (Yr <= (54.0/28.0)*(40.0/54.0)), 1.0/28.0*(17.92461334664 + 0.8743920332081*tr - 0.05567361123058*tr*tr + 0.0006277731764683*tr*tr*tr), model)
    model = np.where(rseg & (Yr > (54.0/28.0)*(40.0/54.0)) & (Yr <= (54.0/28.0)*(54.0/54.0)), 1.0/28.0*np.maximum(0.0, 56.39011190988 - 2.010520359035*tr + 0.01644919857549*tr*tr + 0.00002674976141766*tr*tr*tr), model)
    return model.item() if scalar else model

def tanh_stretch(L, a, j, N):
    ratio = np.log((1.0 + a) / (1.0 - a))
    return L/2.0 + (L/2.0/a)*np.tanh((-1.0 + 2.0*j/N)/2.0 * ratio)

def find_stretch_param(L, N, target_dz):
    a_lo, a_hi = 1e-6, 1.0 - 1e-6
    for _ in range(200):
        a_mid = (a_lo + a_hi) / 2.0
        dz_first = tanh_stretch(L, a_mid, 1, N) - tanh_stretch(L, a_mid, 0, N)
        if dz_first < target_dz: a_hi = a_mid
        else: a_lo = a_mid
    return (a_lo + a_hi) / 2.0

def generate_grid(Ny, Nz, LY=9.0, LZ=3.036, H_HILL=1.0):
    y_1d = np.linspace(0, LY, Ny)
    z_bottom = hill_function(y_1d)
    total_max = LZ; CFL = 0.6
    min_size_coarse = (LZ - H_HILL) / (Nz - 1) * CFL
    a = find_stretch_param(total_max, Nz - 1, min_size_coarse)
    Y = np.zeros((Ny, Nz)); Z = np.zeros((Ny, Nz))
    for i in range(Ny):
        total = LZ - z_bottom[i]; Y[i, :] = y_1d[i]
        for j in range(Nz):
            Z[i, j] = z_bottom[i] + tanh_stretch(total, a, j, Nz - 1)
    return Y, Z

e_cart = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

def compute_jacobian(Y, Z, i, j):
    Ny, Nz = Y.shape
    im, ip = max(i-1, 0), min(i+1, Ny-1)
    jm, jp = max(j-1, 0), min(j+1, Nz-1)
    dy_dxi   = (Y[ip, j] - Y[im, j]) / float(ip - im)
    dz_dxi   = (Z[ip, j] - Z[im, j]) / float(ip - im)
    dy_dzeta = (Y[i, jp] - Y[i, jm]) / float(jp - jm)
    dz_dzeta = (Z[i, jp] - Z[i, jm]) / float(jp - jm)
    J_inv = np.array([[dy_dxi, dy_dzeta], [dz_dxi, dz_dzeta]])
    return np.linalg.inv(J_inv)

def compute_contravariant_velocities(Y, Z, i, j):
    J = compute_jacobian(Y, Z, i, j)
    e_contra = np.zeros((9, 2))
    for alpha in range(9):
        e_contra[alpha] = J @ e_cart[alpha]
    return e_contra

def draw_arrows(ax, xc, yc, vectors, scale,
                color='#CC0000', lw=0.9, zorder=10):
    for alpha in range(1, 9):
        dx = vectors[alpha, 0] * scale
        dy = vectors[alpha, 1] * scale
        if dx**2 + dy**2 < 1e-16:
            continue
        arrow = FancyArrowPatch(
            (xc, yc), (xc + dx, yc + dy),
            arrowstyle='->,head_width=3.0,head_length=4.0',
            color=color, linewidth=lw, zorder=zorder,
            mutation_scale=1.0, shrinkA=0, shrinkB=0,
            clip_on=False,
        )
        ax.add_patch(arrow)

# ======================================================================
# Main
# ======================================================================

def main():
    Ny, Nz = 25, 17
    LY, LZ = 9.0, 3.036
    Y, Z = generate_grid(Ny, Nz, LY=LY, LZ=LZ)

    # ------------------------------------------------------------------
    # Node placement strategy (global normalization):
    #   Boundary (j=1): long arrows → space ξ ≥ 7 apart
    #   Mid-layer (j=6): moderate arrows → ξ ≥ 5 apart, ζ≥5 from boundary
    #   Core (j=10): short arrows → flexible
    #   Top (j=14): moderate arrows
    # ------------------------------------------------------------------
    nodes = [
        # Boundary layer j=1 — large |ẽ|, need wide ξ spacing
        ('A',   3,   1,  '#2166AC',  (-42,  8),  (-55,  8)),
        ('B',  10,   1,  '#1B7837',  ( 14, 10),  ( 14, 10)),
        ("B'", 14,   1,  '#1B7837',  ( 14, 10),  ( 14, 10)),   # mirror of B
        ('C',  21,   1,  '#D94701',  ( 14, 10),  ( 14, 10)),

        # Mid-height j=6 — moderate |ẽ|
        ('D',   3,   6,  '#4393C3',  (-42,  8),  (-55,  8)),
        ('E',  12,   6,  '#762A83',  ( 14,  8),  ( 14,  8)),
        ('F',  21,   6,  '#E08214',  ( 14,  8),  ( 14,  8)),

        # Channel core j=10 — small |ẽ|
        ('G',   6,  10,  '#0571B0',  ( 14,  8),  ( 14,  8)),
        ('H',  18,  10,  '#B35806',  ( 14,  8),  ( 14,  8)),

        # Near top wall j=14
        ('I',  12,  14,  '#C51B7D',  ( 14,  8),  ( 14,  8)),
    ]

    # Precompute
    node_data = []
    for (label, ni, nj, *_) in nodes:
        e_contra = compute_contravariant_velocities(Y, Z, ni, nj)
        mags = np.array([np.linalg.norm(e_contra[a]) for a in range(1, 9)])
        node_data.append((e_contra, mags, mags.max()))

    global_max_e = max(nd[2] for nd in node_data)
    global_arrow_scale = 2.8 / global_max_e

    for idx, (label, ni, nj, *_) in enumerate(nodes):
        lm = node_data[idx][2]
        print(f"  {label} (i={ni:2d},j={nj:2d}): |ẽ|_max={lm:6.2f}  "
              f"→ arrow={lm*global_arrow_scale:.2f} grid")

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 13,
        'mathtext.fontset': 'cm', 'axes.linewidth': 1.0,
    })

    fig = plt.figure(figsize=(22, 8.5))
    ax_phys = fig.add_axes([0.03, 0.09, 0.38, 0.82])
    ax_comp = fig.add_axes([0.56, 0.09, 0.42, 0.82])

    # =================================================================
    # LEFT PANEL
    # =================================================================
    ax_phys.set_facecolor('#F5F5F5')
    for sp in ax_phys.spines.values():
        sp.set_linewidth(2.0)

    for j in range(Nz):
        ax_phys.plot(Y[:, j], Z[:, j], color='#C0C0C0', lw=0.5, zorder=1)
    for i in range(Ny):
        ax_phys.plot(Y[i, :], Z[i, :], color='#C0C0C0', lw=0.5, zorder=1)

    y_fine = np.linspace(0, LY, 500)
    z_wall = hill_function(y_fine)
    ax_phys.fill_between(y_fine, 0, z_wall, color='#D8D8D8', zorder=2)
    ax_phys.plot(y_fine, z_wall, color='black', lw=2.0, zorder=5)
    ax_phys.plot([0, LY], [LZ, LZ], color='black', lw=2.0, zorder=5, clip_on=False)

    arrow_base = 0.22

    for (label, ni, nj, color, phys_off, comp_off) in nodes:
        yc, zc = Y[ni, nj], Z[ni, nj]
        e_square = e_cart * arrow_base
        draw_arrows(ax_phys, yc, zc, e_square, scale=1.0,
                    color='#CC0000', lw=0.9, zorder=10)
        ax_phys.plot(yc, zc, 'o', color=color, markersize=7,
                     markeredgecolor='black', markeredgewidth=0.7, zorder=15)
        ax_phys.annotate(label, (yc, zc), textcoords='offset points',
                         xytext=phys_off, fontsize=11, fontweight='bold',
                         color=color, zorder=20,
                         path_effects=[pe.withStroke(linewidth=2.8, foreground='white')])

    ax_phys.set_xlabel(r'$y\,/\,h$', fontsize=14)
    ax_phys.set_ylabel(r'$z\,/\,h$', fontsize=14)
    ax_phys.set_title(r'Physical Space $(y,\, z)$', fontsize=16, pad=10)
    ax_phys.set_xlim(0, LY)
    ax_phys.set_ylim(0, LZ)
    ax_phys.set_aspect('equal')
    ax_phys.tick_params(labelsize=11)

    # =================================================================
    # RIGHT PANEL — GLOBAL normalisation
    # =================================================================
    ax_comp.set_facecolor('white')
    for sp in ax_comp.spines.values():
        sp.set_linewidth(2.0)

    for j in range(Nz):
        ax_comp.plot([0, Ny-1], [j, j], color='#D0D0D0', lw=0.5, zorder=1)
    for i in range(Ny):
        ax_comp.plot([i, i], [0, Nz-1], color='#D0D0D0', lw=0.5, zorder=1)

    for idx, (label, ni, nj, color, phys_off, comp_off) in enumerate(nodes):
        xi_c, zeta_c = float(ni), float(nj)
        e_contra, mags, local_max = node_data[idx]

        draw_arrows(ax_comp, xi_c, zeta_c, e_contra, scale=global_arrow_scale,
                    color='#CC0000', lw=0.9, zorder=10)

        ax_comp.plot(xi_c, zeta_c, 'o', color=color, markersize=7,
                     markeredgecolor='black', markeredgewidth=0.7, zorder=15)
        ax_comp.annotate(label, (xi_c, zeta_c), textcoords='offset points',
                         xytext=comp_off, fontsize=11, fontweight='bold',
                         color=color, zorder=20,
                         path_effects=[pe.withStroke(linewidth=2.8, foreground='white')])
        mag_off = (comp_off[0], comp_off[1] - 14)
        ax_comp.annotate(
            r'$|\tilde{e}|_{\max}$' + f'={local_max:.1f}',
            (xi_c, zeta_c), textcoords='offset points',
            xytext=mag_off, fontsize=8, color=color, zorder=20,
            path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    ax_comp.set_xlabel(r'$\xi$', fontsize=14)
    ax_comp.set_ylabel(r'$\zeta$', fontsize=14)
    ax_comp.set_title(r'Computational Space $(\xi,\, \zeta)$', fontsize=16, pad=10)
    ax_comp.set_xlim(0, Ny - 1)
    ax_comp.set_ylim(0, Nz - 1)
    ax_comp.set_aspect('equal')
    ax_comp.tick_params(labelsize=11)

    # =================================================================
    # Central annotation
    # =================================================================
    formula = (
        r'$\tilde{e}_\xi = \dfrac{\partial\xi}{\partial y}\,e_y'
        r' + \dfrac{\partial\xi}{\partial z}\,e_z$'
        '\n\n'
        r'$\tilde{e}_\zeta = \dfrac{\partial\zeta}{\partial y}\,e_y'
        r' + \dfrac{\partial\zeta}{\partial z}\,e_z$'
    )
    fig.text(0.475, 0.55, formula,
             ha='center', va='center', fontsize=13, color='black',
             linespacing=2.0,
             bbox=dict(boxstyle='round,pad=0.55', facecolor='#FAFAFA',
                       edgecolor='#AAAAAA', alpha=0.95))
    fig.text(0.475, 0.38,
             r'$(y,\, z) \;\longrightarrow\; (\xi,\, \zeta)$',
             ha='center', va='center', fontsize=13, color='black')

    # Save
    out = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out, 'curvilinear_mesh_mapping.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(out, 'curvilinear_mesh_mapping.pdf'),
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved to: {out}")
    print("  - curvilinear_mesh_mapping.png")
    print("  - curvilinear_mesh_mapping.pdf")
    plt.close(fig)

if __name__ == '__main__':
    main()