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
// with direct physical-space distance from coordinate arrays
// ============================================================================
//
// 在曲線坐標系 (η,ξ,ζ) 中，計算空間的歐氏距離不等於物理空間距離。
// 例如壁面附近 dz_physical << dx_physical，但計算空間中 Δk 與 Δi 量級相同。
// 若用計算空間距離，x 方向的遠距鄰點被高估權重，稀釋壁面法向資訊。
//
// 方法: 直接查表 z_d[] 取得 stencil 節點的精確物理 z 座標
//
//   ds² = dx²·Δη² + dy²·Δξ² + (z_node - z_dep)²
//
// 其中:
//   dx²·Δη²:  x 均勻 → 精確 (不需查表)
//   dy²·Δξ²:  y 均勻 → 精確 (不需查表)
//   z_node:   從 z_d[(bj+sj)*NZ6 + (bk+sk)] 直接讀取 → 精確
//   z_dep:    departure point 的 z, 由 caller 雙線性插值 z_d 得到
//
// 對比度量張量方法:
//   度量張量用中心點的 dk_dz 線性外推 z, 壁面 tanh stretching 下
//   stencil 邊緣誤差可達 4.5% (權重誤差 21.5%)。
//   直接查表消除此線性化誤差，只在 z_dep (CFL<1, 偏移<1格) 有微小近似。
//
// 性質:
//   - 全部 343 個權重 ≥ 0 且 Σ = 1 → 結果 ∈ [min(f), max(f)] → monotone
//   - ds² = |Δr|² 在歐氏空間中嚴格正定
//   - 若 departure point 恰好在某 node 上，回傳該 node 值 (exact)
//   - 均勻網格時退化為標準各向同性 IDW
//
// Parameters:
//   f_ptr:     f[q] 的 device pointer (全域陣列)
//   bi,bj,bk:  stencil 起始索引 (global index)
//   t_eta, t_xi, t_zeta: departure point 在 stencil 座標系中的位置 [0,6]
//   p:         IDW 冪次
//   nface:     NX6 * NZ6 (j-stride)
//   NX6_val:   NX6 (k-stride 的 i 維度)
//   z_dep:     departure point 的物理 z 座標 (由 caller 計算)
//   z_d:       z_d[j*NZ6+k] 物理 z 座標陣列 (device, 大小 NYD6×NZ6)
__device__ __forceinline__ double idw_3d_interpolate(
    double *f_ptr,
    int bi, int bj, int bk,
    double t_eta, double t_xi, double t_zeta,
    double p,
    int nface, int NX6_val,
    double z_dep, double *z_d
) {
    const double eps = 1.0e-24;  // ds² < eps → 在 node 上
    // x, y 方向均勻: 物理距離 = grid_spacing × index_offset
    const double dx2 = (LX / (double)NX) * (LX / (double)NX);  // dx_phys²
    const double dy2 = (LY / (double)NY) * (LY / (double)NY);  // dy_phys²
    double sum_w  = 0.0;
    double sum_wf = 0.0;

    for (int si = 0; si < 7; si++) {
        double d_eta = t_eta - (double)si;
        double dx2_term = dx2 * d_eta * d_eta;  // Δx² (精確, x 均勻)
        int gi = bi + si;

        for (int sj = 0; sj < 7; sj++) {
            double d_xi = t_xi - (double)sj;
            double dy2_term = dy2 * d_xi * d_xi;  // Δy² (精確, y 均勻)
            int gj = bj + sj;
            // z_d 的 j-stride 偏移 (同一 sj 的所有 sk 共用)
            int z_base = gj * NZ6;

            for (int sk = 0; sk < 7; sk++) {
                int gk = bk + sk;
                // z_node: 從 z_d 直接讀取 — 精確物理座標
                double dz = z_d[z_base + gk] - z_dep;

                // 物理空間距離平方: ds² = Δx² + Δy² + Δz²
                double d2 = dx2_term + dy2_term + dz * dz;

                int idx = gj * nface + gk * NX6_val + gi;
                double f_val = f_ptr[idx];

                // 距離 ≈ 0 → 恰好在此 node → 直接回傳
                if (d2 < eps) return f_val;

                // w = 1 / ds^p = 1 / (ds²)^(p/2)
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
 