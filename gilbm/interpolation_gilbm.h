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
        //t為非物理空間計算點的座標
        a[k] = L;//逐點輸出插值權重
    }
}

// ============================================================================
// IDW (Inverse Distance Weighting) 7-point interpolation coefficients
// ============================================================================
// Nodes at integer positions 0,1,2,3,4,5,6; evaluate at position t
// w_k = 1 / |t - k|^p,  normalized so Σ w_k = 1
//
// Properties:
//   - All weights ≥ 0 and Σ=1 → result ∈ [min(f), max(f)] (monotone)
//   - No Runge oscillation possible (unlike Lagrange)
//   - When t is exactly at a node, that node gets weight=1 (exact reproduction)
//   - Accuracy: O(h^2) for p=2, lower than Lagrange O(h^6)
//   - Suitable for steady-state solvers where stability > transient accuracy
__device__ __forceinline__ void idw_7point_coeffs(double t, double p, double a[7]) {
    const double eps = 1.0e-12;  // tolerance for "on node" detection
    double sum = 0.0;

    // Check if t is very close to a node → snap to that node
    for (int k = 0; k < 7; k++) { //k為內插成員計算點的stencil座標 
        double d = t - (double)k; //t為中心點 (非物理空間計算點)的stencil 座標 
        if (d < 0.0) d = -d;  // fabs//距離沒有負值 
        if (d < eps) {
            // Exactly on node k: delta function
            for (int m = 0; m < 7; m++) a[m] = 0.0;
            a[k] = 1.0;//最靠近的計算點權重設為1 
            return;
        }
    }

    // General case: compute IDW weights
    for (int k = 0; k < 7; k++) {//k為內插成員計算點的'stencil座標 
        double d = t - (double)k;
        if (d < 0.0) d = -d;  // fabs
        // w_k = 1 / d^p
        // For p=2: 1/d^2; for integer p, pow is exact
        double w = 1.0;//fake value 
        for (int ip = 0; ip < (int)p; ip++) w /= d;  // w = 1/d^p (integer p)
        // For non-integer p, use: w = 1.0 / pow(d, p);
        a[k] = w;
        sum += w;//從一維方向逐點加總權重 
    }

    // Normalize
    double inv_sum = 1.0 / sum;
    for (int k = 0; k < 7; k++) a[k] *= inv_sum;
}

// Host-side version for precomputation
static inline void idw_7point_coeffs_host(double t, double p, double a[7]) {
    const double eps = 1.0e-12;
    double sum = 0.0;

    for (int k = 0; k < 7; k++) {
        double d = fabs(t - (double)k);
        if (d < eps) {
            for (int m = 0; m < 7; m++) a[m] = 0.0;
            a[k] = 1.0;
            return;
        }
    }

    for (int k = 0; k < 7; k++) {
        double d = fabs(t - (double)k);
        double w = 1.0 / pow(d, p);
        a[k] = w;
        sum += w;
    }

    double inv_sum = 1.0 / sum;
    for (int k = 0; k < 7; k++) a[k] *= inv_sum;
}

// ============================================================================
// True 3D IDW interpolation on 7×7×7 = 343 stencil points
// ============================================================================
// 直接在 343 個 stencil 點上用三維歐氏距離計算 IDW 權重:
//   d_{ijk} = sqrt( (t_η - si)² + (t_ξ - sj)² + (t_ζ - sk)² )
//   w_{ijk} = 1 / d_{ijk}^p
//   Σ_{343} w = 1 (歸一化)
//
// 性質:
//   - 全部 343 個權重 ≥ 0 且 Σ = 1 → 結果 ∈ [min(f), max(f)] → monotone
//   - 用真實 3D 距離，角落點自然獲得更小的權重 (比 separable 版更物理)
//   - 不可分離 → 無法用三階段 reduction，必須 343 次乘加
//   - 若 departure point 恰好在某 node 上，回傳該 node 值 (exact)
//
// Parameters:
//   f_ptr:   f[q] 的 device pointer (全域陣列)
//   bi,bj,bk: stencil 起始索引 (global index)
//   t_eta, t_xi, t_zeta: departure point 在 stencil 座標系中的位置 [0,6]
//   p:       IDW 冪次
//   nface:   NX6 * NZ6 (j-stride)
//   NX6_val: NX6 (k-stride 的 i 維度)
__device__ __forceinline__ double idw_3d_interpolate(
    double *f_ptr,
    int bi, int bj, int bk,
    double t_eta, double t_xi, double t_zeta,
    double p,
    int nface, int NX6_val
) {
    const double eps = 1.0e-24;  // d² < eps → 在 node 上
    double sum_w  = 0.0;
    double sum_wf = 0.0;

    for (int si = 0; si < 7; si++) {
        double d_eta  = t_eta - (double)si;
        double d_eta2 = d_eta * d_eta;
        int gi = bi + si;

        for (int sj = 0; sj < 7; sj++) {
            double d_xi  = t_xi - (double)sj;
            double d_xi2 = d_xi * d_xi;
            int gj = bj + sj;

            for (int sk = 0; sk < 7; sk++) {
                double d_zeta  = t_zeta - (double)sk;
                double d2 = d_eta2 + d_xi2 + d_zeta * d_zeta;

                int gk  = bk + sk;
                int idx = gj * nface + gk * NX6_val + gi;
                double f_val = f_ptr[idx];

                // 距離 ≈ 0 → 恰好在此 node → 直接回傳
                if (d2 < eps) return f_val;

                // w = 1 / d^p = 1 / (d²)^(p/2)
                // p=2 (常見): w = 1/d², 不需 sqrt
                // p=4: w = 1/(d²)²
                // 一般: w = 1/(d²)^(p/2)
                double w;
                if (p == 2.0) {
                    w = 1.0 / d2;
                } else if (p == 4.0) {
                    w = 1.0 / (d2 * d2);
                } else {
                    w = 1.0 / pow(d2, p * 0.5);
                }

                sum_w  += w;
                sum_wf += w * f_val;
            }
        }
    }

    return sum_wf / sum_w;
}

// ============================================================================
// True 3D IDW for f_pc (pre-copied stencil) SoA memory layout
// ============================================================================
// f_pc 的記憶體格局:
//   f_pc[(q * STENCIL_VOL + si * 49 + sj * 7 + sk) * GRID_SIZE + index]
// 其中 STENCIL_VOL = 343 = 7×7×7
//
// 和 idw_3d_interpolate 完全相同的 IDW 邏輯，只是資料讀取方式不同。
__device__ __forceinline__ double idw_3d_interpolate_fpc(
    double *f_pc,
    int q, int index,
    double t_eta, double t_xi, double t_zeta,
    double p,
    int STENCIL_VOL_val, int GRID_SIZE_val
) {
    const double eps = 1.0e-24;
    double sum_w  = 0.0;
    double sum_wf = 0.0;
    const int q_off = q * STENCIL_VOL_val;

    for (int si = 0; si < 7; si++) {
        double d_eta  = t_eta - (double)si;
        double d_eta2 = d_eta * d_eta;

        for (int sj = 0; sj < 7; sj++) {
            double d_xi  = t_xi - (double)sj;
            double d_xi2 = d_xi * d_xi;
            int jk_flat_base = sj * 7;

            for (int sk = 0; sk < 7; sk++) {
                double d_zeta = t_zeta - (double)sk;
                double d2 = d_eta2 + d_xi2 + d_zeta * d_zeta;

                int stencil_idx = q_off + si * 49 + jk_flat_base + sk;
                double f_val = f_pc[stencil_idx * GRID_SIZE_val + index];

                if (d2 < eps) return f_val;

                double w;
                if (p == 2.0) {
                    w = 1.0 / d2;
                } else if (p == 4.0) {
                    w = 1.0 / (d2 * d2);
                } else {
                    w = 1.0 / pow(d2, p * 0.5);
                }

                sum_w  += w;
                sum_wf += w * f_val;
            }
        }
    }

    return sum_wf / sum_w;
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
//   Eq. 13: c̃ = c·e·∂ξ/∂x     — 逆變速度 (僅用於計算 streaming 位移)
//   → 碰撞算子 (feq, MRT)        始終在物理速度空間中執行
__device__ __forceinline__ double compute_feq_alpha(
    int alpha, double rho, double u, double v, double w
) {
    double eu = GILBM_e[alpha][0]*u + GILBM_e[alpha][1]*v + GILBM_e[alpha][2]*w;
    double udot = u*u + v*v + w*w;
    return GILBM_W[alpha] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*udot);
}

#endif
 