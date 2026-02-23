#ifndef GILBM_INTERPOLATION_H
#define GILBM_INTERPOLATION_H

// 7-point Lagrange weighted sum
#define Intrpl7(f1, a1, f2, a2, f3, a3, f4, a4, f5, a5, f6, a6, f7, a7) \
    ((f1)*(a1)+(f2)*(a2)+(f3)*(a3)+(f4)*(a4)+(f5)*(a5)+(f6)*(a6)+(f7)*(a7))

// Compute 1D 7-point Lagrange interpolation coefficients
// Nodes at integer positions 0,1,2,3,4,5,6; evaluate at position t
__device__ __forceinline__ void lagrange_7point_coeffs(double t, double a[7]) {
    for (int k = 0; k < 7; k++) {
        double L = 1.0;
        for (int j = 0; j < 7; j++) {
            if (j != k) L *= (t - (double)j) / (double)(k - j);
        }
        a[k] = L;
    }
}

// ── 平衡態分佈函數 (標準 D3Q19 BGK 公式) ─────────────────────────
// f^eq_α = w_α · ρ · (1 + 3·(e_α·u) + 4.5·(e_α·u)² − 1.5·|u|²)
//
// ★ 此公式在 GILBM 曲線坐標系中仍然正確，無需 Jacobian 修正。
//
// 原因：GILBM 的分佈函數 f_i 定義在物理直角坐標的速度空間中：
//   - GILBM_e[i] = 物理直角坐標系的離散速度向量 (e_x, e_y, e_z)
//     → 標準 D3Q19 整數向量 {0, ±1}，不是曲線坐標的逆變速度分量
//   - (u, v, w) = 物理直角坐標系的宏觀速度
//     → 由 Σ f_i·e_i / ρ 直接得到，因為 e_i 就是物理向量
//   - 曲線坐標映射 (Jacobian) 只進入 streaming 步驟的空間位移量：
//     δη = dt·e_x/dx,  δξ = dt·e_y/dy,  δζ = dt·(e_y·dk_dy + e_z·dk_dz)
//     → 度量項 (dk_dy, dk_dz) 乘在位移上，不乘在 e_i 上
//
// 參考：Imamura 2005
//   Eq. 2:  c_i = c × e_i     — 物理速度 (直角坐標)
//   Eq. 13: c̃ = c·e·∂ξ/∂x   — 逆變速度 (僅用於計算 streaming 位移)
//   → 碰撞算子 (feq, MRT) 始終在物理速度空間中執行
__device__ __forceinline__ double compute_feq_alpha(
    int alpha, double rho, double u, double v, double w
) {
    double eu = GILBM_e[alpha][0]*u + GILBM_e[alpha][1]*v + GILBM_e[alpha][2]*w;
    double udot = u*u + v*v + w*w;
    return GILBM_W[alpha] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*udot);
}

#endif
 