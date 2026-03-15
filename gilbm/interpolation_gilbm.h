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

// ============================================================================
// MLS (Moving Least Squares) 3D interpolation with quadratic basis (m=10)
// ============================================================================
//
// 在 7×7×7 = 343 stencil 上，以物理空間距離加權擬合局部二次多項式：
//   f̃(x₀) = pᵀ(0)·a = a[0]
// 其中 a 由正規方程 (PᵀWP)·a = PᵀW·f 求解。
//
// 基底: p = [1, δx, δy, δz, δx², δy², δz², δxδy, δxδz, δyδz]
// 權重: Wendland C² — w(r) = (1-r/h)⁴(4r/h+1), 緊致支撐, C² 連續
//
// 性質:
//   - C² 連續插值 → 精確重建梯度 (梯度誤差 < 2%)
//   - 距離加權 → 自然處理各向異性網格 (壁面 dx/dz ~ 50)
//   - 座標歸一化 → PᵀWP 矩陣良態 (cond ~ O(1))
//   - Cholesky 10×10 → 全部在 register 中操作, ~440 FLOP
//   - Fallback: Cholesky 失敗時退化為 IDW
//
// 代價: ~30K FLOP/方向 (~10× IDW, ~58× Lagrange)
// ============================================================================

// ── Wendland C² 權重函數 ────────────────────────────────────────
// w(r) = (1 - r/h)⁴ · (4r/h + 1)  for r < h; 0 for r ≥ h
// 輸入: d2 = 歸一化距離², h2 = 歸一化支撐半徑²
// 性質: w(0)=1, w(h)=0, w'(h)=0, w''(h)=0 → C² 連續
__device__ __forceinline__ double wendland_c2_weight(double d2, double h2) {
    if (d2 >= h2) return 0.0;
    double q = sqrt(d2 / h2);  // q = r/h ∈ [0, 1)
    double t = 1.0 - q;
    double t2 = t * t;
    return t2 * t2 * (4.0 * q + 1.0);
}

// ── 10×10 Cholesky 求解器 (packed upper triangle, register-only) ──
// A[55]: 上三角存儲, A[i,j] at index i*(21-i)/2 + j-i  (j ≥ i)
//   Row 0: indices 0..9    (10 elements)
//   Row 1: indices 10..18  (9 elements)
//   ...
//   Row 9: index 54        (1 element)
// b[10]: RHS 輸入, 解 x 輸出 (in-place)
// 返回: true = 成功, false = 矩陣不正定 (需 fallback)
//
// 演算法: A = RᵀR (R 上三角), 前代 Rᵀy=b, 後代 Rx=y
// FLOP: ~440 (分解 ~330 + 前/後代換 ~110)
__device__ __forceinline__ bool cholesky_solve_10x10(double A[55], double b[10]) {
    const int N = 10;
    const double tol = 1.0e-20;

    // ── Phase 1: Cholesky 分解 A = RᵀR (R 上三角, in-place 覆蓋 A) ──
    for (int j = 0; j < N; j++) {
        int jj = j * (21 - j) / 2;  // A[j,j] 的 packed index
        double diag = A[jj];
        for (int k = 0; k < j; k++) {
            double rkj = A[k * (21 - k) / 2 + j - k];  // R[k,j]
            diag -= rkj * rkj;
        }
        if (diag <= tol) return false;  // 不正定
        A[jj] = sqrt(diag);             // R[j,j]
        double inv_rjj = 1.0 / A[jj];

        for (int i = j + 1; i < N; i++) {
            int ji = j * (21 - j) / 2 + i - j;  // A[j,i]
            double sum = A[ji];
            for (int k = 0; k < j; k++) {
                sum -= A[k * (21 - k) / 2 + j - k]    // R[k,j]
                     * A[k * (21 - k) / 2 + i - k];   // R[k,i]
            }
            A[ji] = sum * inv_rjj;  // R[j,i]
        }
    }

    // ── Phase 2: 前代 Rᵀy = b (Rᵀ 下三角) ──
    for (int i = 0; i < N; i++) {
        double sum = b[i];
        for (int k = 0; k < i; k++) {
            sum -= A[k * (21 - k) / 2 + i - k] * b[k];  // Rᵀ[i,k] = R[k,i]
        }
        b[i] = sum / A[i * (21 - i) / 2];  // R[i,i]
    }

    // ── Phase 3: 後代 Rx = y (R 上三角) ──
    for (int i = N - 1; i >= 0; i--) {
        double sum = b[i];
        for (int k = i + 1; k < N; k++) {
            sum -= A[i * (21 - i) / 2 + k - i] * b[k];  // R[i,k]
        }
        b[i] = sum / A[i * (21 - i) / 2];  // R[i,i]
    }

    return true;
}

// ── MLS 3D 插值 (全域陣列版, 物理空間距離) ──────────────────────
// 從 f_ptr (全域 f[q] 陣列) 讀取 7×7×7 stencil 資料。
// 使用物理空間距離 + 座標歸一化確保 PᵀWP 良態。
//
// 歸一化策略:
//   sx = 6·dx, sy = 6·dy (compile-time 常數, x/y 均勻)
//   sz = max|z_node - z_dep| over 7×7 stencil (runtime, 49 次比較)
//   → 所有歸一化座標 ∈ [-1, 1], d²_norm ≤ 3
//   → 支撐半徑 h=2.0 (h²=4.0) 確保全部 343 點有非零權重
//
// Parameters:
//   f_ptr:     f[q] device pointer (全域陣列)
//   bi,bj,bk:  stencil 起始索引 (global index)
//   t_eta, t_xi, t_zeta: departure point 在 stencil 座標中的位置 [0,6]
//   nface:     NX6 * NZ6 (j-stride)
//   NX6_val:   NX6 (i-stride)
//   z_dep:     departure point 的物理 z 座標
//   z_d:       z_d[j*NZ6+k] 物理 z 座標陣列
__device__ __forceinline__ double mls_3d_interpolate(
    double *f_ptr,
    int bi, int bj, int bk,
    double t_eta, double t_xi, double t_zeta,
    int nface, int NX6_val,
    double z_dep, double *z_d
) {
    // 物理網格間距 (compile-time 常數)
    const double dx = LX / (double)NX;
    const double dy = LY / (double)NY;

    // 歸一化尺度
    const double sx = 6.0 * dx;  // x 方向 stencil 跨度
    const double sy = 6.0 * dy;  // y 方向 stencil 跨度

    // 預掃描: 找 z 方向最大跨度 (7×7 = 49 次, 極低開銷)
    double sz = 0.0;
    for (int sj = 0; sj < 7; sj++) {
        int gj = bj + sj;
        for (int sk = 0; sk < 7; sk++) {
            double dz = z_d[gj * NZ6 + bk + sk] - z_dep;
            if (dz < 0.0) dz = -dz;
            if (dz > sz) sz = dz;
        }
    }
    if (sz < 1.0e-30) sz = 1.0;  // 安全: 避免除以零

    // 歸一化空間支撐半徑: h = 2.0
    // 所有 343 節點 d_norm ≤ √3 ≈ 1.73 < 2.0 → 全部有非零權重
    const double h2 = 4.0;  // h² = 2.0²

    // 累積 PᵀWP (55 upper triangle) 和 PᵀWf (10)
    double A[55];
    double bv[10];
    for (int m = 0; m < 55; m++) A[m] = 0.0;
    for (int m = 0; m < 10; m++) bv[m] = 0.0;

    int active = 0;

    for (int si = 0; si < 7; si++) {
        double px = ((double)si - t_eta) * dx / sx;  // 歸一化 Δx
        int gi = bi + si;

        for (int sj = 0; sj < 7; sj++) {
            double py = ((double)sj - t_xi) * dy / sy;  // 歸一化 Δy
            int gj = bj + sj;
            int z_base = gj * NZ6;

            for (int sk = 0; sk < 7; sk++) {
                int gk = bk + sk;
                double pz = (z_d[z_base + gk] - z_dep) / sz;  // 歸一化 Δz

                double d2 = px * px + py * py + pz * pz;
                double w = wendland_c2_weight(d2, h2);
                if (w < 1.0e-30) continue;

                active++;
                int idx = gj * nface + gk * NX6_val + gi;
                double fval = f_ptr[idx];

                // 二次基底 (m=10): [1, x, y, z, x², y², z², xy, xz, yz]
                double p[10];
                p[0] = 1.0;  p[1] = px;  p[2] = py;  p[3] = pz;
                p[4] = px*px; p[5] = py*py; p[6] = pz*pz;
                p[7] = px*py; p[8] = px*pz; p[9] = py*pz;

                // 累積 PᵀWP (上三角) 和 PᵀWf
                for (int a = 0; a < 10; a++) {
                    double wpa = w * p[a];
                    bv[a] += wpa * fval;
                    int base_a = a * (21 - a) / 2;
                    for (int bb = a; bb < 10; bb++) {
                        A[base_a + bb - a] += wpa * p[bb];
                    }
                }
            }
        }
    }

    // 至少需要 m=10 個有效節點 (quadratic basis 的自由度)
    if (active < 10) {
        return idw_3d_interpolate(f_ptr, bi, bj, bk,
            t_eta, t_xi, t_zeta, 2.0, nface, NX6_val, z_dep, z_d);
    }

    // Cholesky 求解 (PᵀWP)·a = PᵀWf
    if (!cholesky_solve_10x10(A, bv)) {
        return idw_3d_interpolate(f_ptr, bi, bj, bk,
            t_eta, t_xi, t_zeta, 2.0, nface, NX6_val, z_dep, z_d);
    }

    // f̃(x₀) = a[0] (因為 p(0) = [1, 0, ..., 0])
    return bv[0];
}

// ── MLS 3D 插值 (f_pc SoA 版, 計算空間歸一化距離) ──────────────
// f_pc 的記憶體格局:
//   f_pc[(q * STENCIL_VOL + si * 49 + sj * 7 + sk) * GRID_SIZE + index]
//
// 此版本使用計算空間歸一化距離 (無需 z_d)。
// 各方向均勻歸一化: Δx_norm = (si - t_eta)/6, 同 Δy, Δz。
// 梯度精度仍遠優於 IDW (二次多項式 vs 加權平均)，
// 但壁面附近各向異性修正不如物理空間版。
__device__ __forceinline__ double mls_3d_interpolate_fpc(
    double *f_pc,
    int q, int index,
    double t_eta, double t_xi, double t_zeta,
    int STENCIL_VOL_val, int GRID_SIZE_val
) {
    const double h2 = 4.0;  // 歸一化支撐半徑²
    double A[55];
    double bv[10];
    for (int m = 0; m < 55; m++) A[m] = 0.0;
    for (int m = 0; m < 10; m++) bv[m] = 0.0;

    const int q_off = q * STENCIL_VOL_val;

    for (int si = 0; si < 7; si++) {
        double px = ((double)si - t_eta) / 6.0;  // 歸一化 ∈ [-1, 1]

        for (int sj = 0; sj < 7; sj++) {
            double py = ((double)sj - t_xi) / 6.0;
            int jk_base = sj * 7;

            for (int sk = 0; sk < 7; sk++) {
                double pz = ((double)sk - t_zeta) / 6.0;

                double d2 = px * px + py * py + pz * pz;
                double w = wendland_c2_weight(d2, h2);
                if (w < 1.0e-30) continue;

                int stencil_idx = q_off + si * 49 + jk_base + sk;
                double fval = f_pc[stencil_idx * GRID_SIZE_val + index];

                double p[10];
                p[0] = 1.0;  p[1] = px;  p[2] = py;  p[3] = pz;
                p[4] = px*px; p[5] = py*py; p[6] = pz*pz;
                p[7] = px*py; p[8] = px*pz; p[9] = py*pz;

                for (int a = 0; a < 10; a++) {
                    double wpa = w * p[a];
                    bv[a] += wpa * fval;
                    int base_a = a * (21 - a) / 2;
                    for (int bb = a; bb < 10; bb++) {
                        A[base_a + bb - a] += wpa * p[bb];
                    }
                }
            }
        }
    }

    if (!cholesky_solve_10x10(A, bv)) {
        return idw_3d_interpolate_fpc(f_pc, q, index,
            t_eta, t_xi, t_zeta, 2.0, STENCIL_VOL_val, GRID_SIZE_val);
    }

    return bv[0];
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
 