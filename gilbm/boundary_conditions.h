#ifndef GILBM_BOUNDARY_CONDITIONS_H
#define GILBM_BOUNDARY_CONDITIONS_H

// Phase 1: Chapman-Enskog BC for GILBM (Imamura 2005 Eq. A.9, no-slip wall u=0)
//
// Direction criterion (Section 1.1.E):
//   Bottom wall k=2: e_tilde_k > 0 -> upwind point outside domain -> need C-E BC
//   Top wall k=NZ6-3: e_tilde_k < 0 -> upwind point outside domain -> need C-E BC
//
// C-E BC formula at no-slip wall (u=v=w=0), Imamura Eq.(A.9):
//   f_i|wall = w_i * rho_wall * (1 + C_i)
//
//   Eq.(A.9) 張量: (c_{iα} - u_α)(c_{iβ} - u_β) / c_s^4 - δ_{αβ}/c_s^2
//   壁面 u=0 → 退化為: 9·c_{iα}·c_{iβ} - δ_{αβ}  (1/c_s^4 = 9)
//   α = 1~3 (x,y,z 速度分量),  β = 2~3 (ξ,ζ 方向; β=1(η) 因 dk/dx=0 消去)
//
//   C_i = -ω·Δt · Σ_α Σ_{β=y,z} [9·c_{iα}·c_{iβ} - δ_{αβ}] · (∂u_α/∂x_β)
//
//   壁面 chain rule: ∂u_α/∂x_β = (du_α/dk)·(dk/dx_β)，展開 3α × 2β = 6 項：
//
//   C_i = -ω·Δt × {
//     ① 9·c_{ix}·c_{iy} · (du/dk)·(dk/dy)        α=x, β=y  (δ_{xy}=0)
//   + ② 9·c_{ix}·c_{iz} · (du/dk)·(dk/dz)        α=x, β=z  (δ_{xz}=0)
//   + ③ (9·c_{iy}²−1)   · (dv/dk)·(dk/dy)        α=y, β=y  (δ_{yy}=1)
//   + ④ 9·c_{iy}·c_{iz} · (dv/dk)·(dk/dz)        α=y, β=z  (δ_{yz}=0)
//   + ⑤ 9·c_{iz}·c_{iy} · (dw/dk)·(dk/dy)        α=z, β=y  (δ_{zy}=0)
//   + ⑥ (9·c_{iz}²−1)   · (dw/dk)·(dk/dz)        α=z, β=z  (δ_{zz}=1)
//   }
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
    // if e_tilde_k > 0.0 then 回傳 is_bottom_wall
    // if e_tilde_k < 0.0 then 回傳 !is_bottom_wall
}

// Chapman-Enskog BC: compute f_alpha at no-slip wall
__device__ double ChapmanEnskogBC(
    int alpha,
    double rho_wall,
    double du_dk, double dv_dk, double dw_dk,  // velocity gradients at wall
    double dk_dy_val, double dk_dz_val,
    double omega_val, double localtimestep
) {
    double ex = GILBM_e[alpha][0];
    double ey = GILBM_e[alpha][1];
    double ez = GILBM_e[alpha][2];

    // 展開 6 項 (dk/dx=0，僅 β=y,z 存活)
    double C_alpha = 0.0;

    // α=x: ①② 項
    C_alpha += (
        (9.0 * ex * ey) * du_dk * dk_dy_val +       // ① 9·c_x·c_y · (du/dk)·(dk/dy)//x->u;y->y
        (9.0 * ex * ez) * du_dk * dk_dz_val          // ② 9·c_x·c_z · (du/dk)·(dk/dz)//x->u;z->z
    );

    // α=y: ③④ 項
    C_alpha += (
        (9.0 * ey * ey - 1.0) * dv_dk * dk_dy_val + // ③ (9·c_y²−1) · (dv/dk)·(dk/dy)//y->v;y->y
        (9.0 * ey * ez) * dv_dk * dk_dz_val          // ④ 9·c_y·c_z · (dv/dk)·(dk/dz)//y->v;z->z
    );

    // α=z: ⑤⑥ 項
    C_alpha += (
        (9.0 * ez * ey) * dw_dk * dk_dy_val +       // ⑤ 9·c_z·c_y · (dw/dk)·(dk/dy)//z->w;y->y
        (9.0 * ez * ez - 1.0) * dw_dk * dk_dz_val   // ⑥ (9·c_z²−1) · (dw/dk)·(dk/dz)//z->w;z->z
    );

    C_alpha *= -omega_val * localtimestep;
    // equibilirium distribution function = GILBM_W[alpha] * rho_wall 
    // f_alpha = equibilirium distribution function * (C_alpha)
    double f_eq_atwall = GILBM_W[alpha] * rho_wall;
    return f_eq_atwall * (1.0 + C_alpha) ;  //計算壁面上的插值後分佈函數

#endif
