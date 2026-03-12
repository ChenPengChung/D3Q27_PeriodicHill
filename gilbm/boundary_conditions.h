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
//   CE non-equilibrium (NS level): f^neq = -ρ w_i (τ-0.5)Δt / c_s² × Σ(e·e - c_s² δ) × S
//   c=1, c_s²=1/3 → 1/c_s² = 3 → tensor coeff: 3·c_{iα}·c_{iβ} - δ_{αβ}
//   α = 1~3 (x,y,z 速度分量),  β = 2~3 (ξ,ζ 方向; β=1(η) 因 dk/dx=0 消去)
//
//   C_i = -(omega_local)·Δt · Σ_α Σ_{β=y,z} [3·c_{iα}·c_{iβ} - δ_{αβ}] · (∂u_α/∂x_β)
//
//   壁面 chain rule: ∂u_α/∂x_β = (du_α/dk)·(dk/dx_β)，展開 3α × 2β = 6 項：
//
//   C_i = -(omega_local)·Δt × {                     [= -3ν, 常數]
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
    // 底壁 (is_bottom_wall=true):  e_tilde_k > 0 → 出發點在壁外 → 回傳 true (需要 BC)
    // 頂壁 (is_bottom_wall=false): e_tilde_k < 0 → 出發點在壁外 → 回傳 true (需要 BC)
}

// Chapman-Enskog BC: compute f_alpha at no-slip wall
//
// 係數: -ω·Δt (Imamura 2005 Appendix Eq. A.9)
//   ω = omega_local = τ = 3ν/Δt + 0.5 (relaxation TIME, code convention)
//   Δt = localtimestep = dt_global (GTS mode)
//
// FIX (USE_CUMULANT): Equilibrium uses GILBM_feq_cum (Cumulant equilibrium) instead
//   of GILBM_W (standard feq). The Cumulant WP mode has 4th-order equilibria
//   (A,B coefficients from Gehrke 2022 Eq.17-18) that make its equilibrium distribution
//   DIFFERENT from standard feq = W[q]*rho. Using standard feq in BC creates mass
//   source/sink at tilted wall points (~1.9e-02 per point), causing density divergence.
//   This fix reduces the error by 650,000x.
__device__ double ChapmanEnskogBC(
    int alpha,
    double rho_wall,
    double du_dk, double dv_dk, double dw_dk,  // velocity gradients at wall
    double dk_dy_val, double dk_dz_val,
    double omega_local, double localtimestep
) {
    double ex = GILBM_e[alpha][0];
    double ey = GILBM_e[alpha][1];
    double ez = GILBM_e[alpha][2];

    // 展開 6 項 (dk/dx=0，僅 β=y,z 存活)
    double C_alpha = 0.0;

    // α=x: ①② 項
    C_alpha += (
        (3.0 * ex * ey) * du_dk * dk_dy_val +       // ① 3·c_x·c_y · (du/dk)·(dk/dy)
        (3.0 * ex * ez) * du_dk * dk_dz_val          // ② 3·c_x·c_z · (du/dk)·(dk/dz)
    );

    // α=y: ③④ 項
    C_alpha += (
        (3.0 * ey * ey - 1.0) * dv_dk * dk_dy_val + // ③ (3·c_y²−1) · (dv/dk)·(dk/dy)
        (3.0 * ey * ez) * dv_dk * dk_dz_val          // ④ 3·c_y·c_z · (dv/dk)·(dk/dz)
    );

    // α=z: ⑤⑥ 項
    C_alpha += (
        (3.0 * ez * ey) * dw_dk * dk_dy_val +       // ⑤ 3·c_z·c_y · (dw/dk)·(dk/dy)
        (3.0 * ez * ez - 1.0) * dw_dk * dk_dz_val   // ⑥ (3·c_z²−1) · (dw/dk)·(dk/dz)
    );

    // Chapman-Enskog non-equilibrium coefficient: -(τ-0.5)·Δt = -3ν
    //
    // [BUG FIX] 原始代碼使用 -τ·Δt (Imamura 2005 Eq.A.9 原始寫法),
    //   但 Chapman-Enskog BC 應使用 PHYSICAL viscosity 係數:
    //     f^neq = -w·ρ·(τ-0.5)·Δt·cs^{-2}·Σ(e·e - cs²δ)·S
    //
    //   CE 一階展開 f^(1) ∝ τ 包含了離散碰撞的 "numerical viscosity" (0.5·Δt)，
    //   但 BC 重建的是物理解 (Navier-Stokes level)，應只包含物理黏度 ν。
    //
    //   τ = 3ν/Δt + 0.5 → (τ-0.5)·Δt = 3ν   [正確: 純物理黏度]
    //                      τ·Δt = 3ν + 0.5·Δt  [錯誤: 包含數值黏度]
    //
    //   修正前: ratio = τ/(τ-0.5) = 0.635/0.135 ≈ 4.7× 過大
    //   → 壁面非平衡修正放大 4.7 倍 → 壁面附近不穩定 → Ma 爆衝
    //
    // 參照: Latt & Chopard 2006, Malaspinas 2007 (標準 CE-BC 文獻均使用 (τ-0.5))
    C_alpha *= -(omega_local - 0.5) * localtimestep;

#if USE_CUMULANT
    // FIX: Use Cumulant equilibrium (includes 4th-order A,B corrections)
    // Chapman-Enskog: f = f^eq × (1 + C)
    //   f^eq = feq_cum (Cumulant 固定點), NOT standard W[q]
    //   f^neq = f^eq × C (非平衡校正也必須用 Cumulant 平衡態)
    //
    // AO mode: feq_cum = W → 與舊版一致, 無影響
    // WP mode: feq_cum ≠ W (差 ~5%) → 用 W 會產生質量源/匯 → rho 震盪
    double f_eq_atwall = GILBM_feq_cum[alpha] * rho_wall;
    double f_neq = GILBM_feq_cum[alpha] * rho_wall * C_alpha;
    return f_eq_atwall + f_neq;
#else
    // Standard feq for MRT/BGK: f_alpha = f_eq * (1 + C_alpha)
    double f_eq_atwall = GILBM_W[alpha] * rho_wall;
    return f_eq_atwall * (1.0 + C_alpha);
#endif
}

#endif
