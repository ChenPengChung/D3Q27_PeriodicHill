#ifndef GILBM_BOUNDARY_CONDITIONS_H
#define GILBM_BOUNDARY_CONDITIONS_H

// Phase 1: Chapman-Enskog BC for GILBM (Imamura 2005 Eq. A.9, no-slip wall u=0)
//
// Direction criterion (Section 1.1.E):
//   Bottom wall k=2: e_tilde_k > 0 -> upwind point outside domain -> need C-E BC
//   Top wall k=NZ6-3: e_tilde_k < 0 -> upwind point outside domain -> need C-E BC
//
// C-E BC formula at no-slip wall (u=0):
//   f_alpha|wall = w_alpha * rho_wall * (1 + C_alpha)
//   C_alpha = -omega*dt * sum_i du_i/dk * [(9*e_i*e_j - delta_ij) * dk/dx_j]
//   where 9 = 3/c^2 (c^2 = 1/3 for D3Q19)
//
// Wall velocity gradient: 2nd-order one-sided finite difference (u[wall]=0):
//   du/dk|wall = (4*u[k=3] - u[k=4]) / 2     (bottom wall k=2)
//   du/dk|wall = (4*u[k=NZ6-4] - u[k=NZ6-5]) / 2  (top wall, reversed sign)
//
// Wall density: rho_wall = rho[k=3] (zero normal pressure gradient, Imamura S3.2)

// Check if direction alpha needs BC at this wall point
// Uses GILBM_e from __constant__ memory (defined in evolution_gilbm.h)
//
// 判定準則：ẽ^ζ_α = e_y[α]·dk_dy + e_z[α]·dk_dz（ζ 方向逆變速度分量）
//   底壁 (k=2):   ẽ^ζ_α > 0 → streaming 出發點 k_dep = k - δζ < 2（壁外）→ 需要 BC
//   頂壁 (k=NZ6-3): ẽ^ζ_α < 0 → 出發點 k_dep > NZ6-3（壁外）→ 需要 BC
//
// 返回 true 時：該 α 由 Chapman-Enskog BC 處理，跳過 streaming。
// 對應的 delta_eta[α] / delta_xi[α] / delta_zeta[α,j,k] 不被讀取。
//
// 平坦底壁 BC 方向: α={5,11,12,15,16}（共 5 個，皆 e_z > 0）
// 斜面底壁 (slope<45°): 額外加入 e_y 分量方向，共 8 個 BC 方向
__device__ __forceinline__ bool NeedsBoundaryCondition(
    int alpha,
    double dk_dy_val, double dk_dz_val,
    bool is_bottom_wall
) {
    double e_tilde_k = GILBM_e[alpha][1] * dk_dy_val + GILBM_e[alpha][2] * dk_dz_val;
    return is_bottom_wall ? (e_tilde_k > 0.0) : (e_tilde_k < 0.0);
}

// Chapman-Enskog BC: compute f_alpha at no-slip wall
__device__ double ChapmanEnskogBC(
    int alpha,
    double rho_wall,
    double du_x_dk, double du_y_dk, double du_z_dk,  // velocity gradients at wall
    double dk_dy_val, double dk_dz_val,
    double omega_val, double dt_val
) {
    double ex = GILBM_e[alpha][0];
    double ey = GILBM_e[alpha][1];
    double ez = GILBM_e[alpha][2];

    // C-E correction: C = -omega*dt * sum_i du_i/dk * [(9*e_i*e_j - delta_ij) * dk/dx_j]
    // dk/dx = 0, dk/dy = dk_dy, dk/dz = dk_dz
    double C_alpha = 0.0;

    // i = x component: delta_{x,y}=0, delta_{x,z}=0
    C_alpha += du_x_dk * (
        (9.0 * ex * ey) * dk_dy_val +
        (9.0 * ex * ez) * dk_dz_val
    );

    // i = y component: delta_{y,y}=1, delta_{y,z}=0
    C_alpha += du_y_dk * (
        (9.0 * ey * ey - 1.0) * dk_dy_val +
        (9.0 * ey * ez) * dk_dz_val
    );

    // i = z component: delta_{z,y}=0, delta_{z,z}=1
    C_alpha += du_z_dk * (
        (9.0 * ez * ey) * dk_dy_val +
        (9.0 * ez * ez - 1.0) * dk_dz_val
    );

    C_alpha *= -omega_val * dt_val;

    // f_alpha = w_alpha * rho_wall * (1 + C_alpha)
    return GILBM_W[alpha] * rho_wall * (1.0 + C_alpha);
}

#endif
