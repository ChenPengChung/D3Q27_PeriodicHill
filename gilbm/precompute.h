#ifndef GILBM_PRECOMPUTE_H
#define GILBM_PRECOMPUTE_H

// Phase 1.5: GILBM RK2 upwind displacement precomputation (Imamura 2005 Eq. 19-20)
// Split into ξ-direction (constant for uniform y) and ζ-direction (space-varying).
//
// δξ[α] = dt · ẽ^ξ_α = dt · e_y[α] / dy   → constant per alpha (19 values)
// δζ[α,j,k] = dt · ẽ^ζ_α(k_half)           → RK2 midpoint evaluation [19*NYD6*NZ6]
//
// Called once during initialization; results copied to GPU.

// ============================================================================
// PrecomputeGILBM_DeltaXi: ξ-direction displacement (constant for uniform y)
// ============================================================================
// For uniform y-grid: dξ/dy = 1/dy (constant), dξ/dz = 0
// → ẽ^ξ_α = e_y[α] · (1/dy) = e_y[α] / dy
// → δξ[α] = dt · e_y[α] / dy
// No RK2 correction needed (metric is constant → midpoint = endpoint).
//
// When y becomes non-uniform, promote delta_xi from [19] to [19*NYD6*NZ6]
// and add RK2 midpoint interpolation in j-direction.
void PrecomputeGILBM_DeltaXi(
    double *delta_xi_h,    // 輸出: [19]，ξ 方向位移量（常數）
    double dy_val          // 輸入: uniform grid spacing dy = LY/(NY6-7)
) {
    // D3Q19 離散速度集（與 initialization.h 中一致）
    double e_y[19] = {
        0,
        0, 0, 1, -1, 0, 0,
        1, 1, -1, -1,
        0, 0, 0, 0,
        1, -1, 1, -1
    };

    for (int alpha = 0; alpha < 19; alpha++) {
        delta_xi_h[alpha] = dt * e_y[alpha] / dy_val;
    }
}

// ============================================================================
// PrecomputeGILBM_DeltaZeta: ζ-direction RK2 displacement (space-varying)
// ============================================================================
// Imamura 2005 Eq. 19-20:
//   Step 1: ẽ^ζ_α(k) = e_y·(dk/dy) + e_z·(dk/dz)  at current point
//   Step 2: k_half = k - 0.5·dt·ẽ^ζ_α(k)            RK2 midpoint
//   Step 3: Interpolate dk/dy, dk/dz at k_half
//   Step 4: δζ[α] = dt · ẽ^ζ_α(k_half)              full RK2 displacement
void PrecomputeGILBM_DeltaZeta(
    double *delta_zeta_h,    // 輸出: [19 * NYD6 * NZ6]，預計算的位移量
    const double *dk_dz_h,   // 輸入: 度量項 dk/dz [NYD6*NZ6]
    const double *dk_dy_h,   // 輸入: 度量項 dk/dy [NYD6*NZ6]
    int NYD6_local,
    int NZ6_local
) {
    // D3Q19 離散速度集
    double e[19][3] = {
        {0,0,0},
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };

    int sz = NYD6_local * NZ6_local;

    // 將整個輸出陣列初始化為零
    for (int n = 0; n < 19 * sz; n++) {
        delta_zeta_h[n] = 0.0;
    }

    // 遍歷所有離散速度方向（alpha=0 為靜止方向，跳過）
    for (int alpha = 1; alpha < 19; alpha++) {
        // 若該方向在 y 和 z 分量皆為零，則逆變速度 e_tilde_zeta 恆為零
        if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;

        for (int j = 3; j < NYD6_local - 4; j++) { // 跳過 y 方向 buffer layer
            for (int k = 2; k < NZ6_local - 2; k++) {
                int idx_jk = j * NZ6_local + k;

                // 步驟一：計算當前格點 (j, k) 處的 ζ 方向逆變速度
                double e_tilde_zeta0 = e[alpha][1] * dk_dy_h[idx_jk]
                                     + e[alpha][2] * dk_dz_h[idx_jk];

                // 步驟二：RK2 半步位移與中間點位置
                double dk_half = 0.5 * dt * e_tilde_zeta0;
                double k_half = (double)k - dk_half;

                // 在中間點位置進行度量項的線性插值
                int k_lo = (int)floor(k_half);
                if (k_lo < 2) k_lo = 2;
                if (k_lo > NZ6_local - 4) k_lo = NZ6_local - 4;
                double frac = k_half - (double)k_lo;
                if (frac < 0.0) frac = 0.0;
                if (frac > 1.0) frac = 1.0;

                int idx_lo = j * NZ6_local + k_lo;
                int idx_hi = j * NZ6_local + k_lo + 1;

                // 插值得到中間點處的度量項
                double dk_dy_half = (1.0 - frac) * dk_dy_h[idx_lo]
                                  + frac * dk_dy_h[idx_hi];
                double dk_dz_half = (1.0 - frac) * dk_dz_h[idx_lo]
                                  + frac * dk_dz_h[idx_hi];

                // 步驟三：計算中間點處的逆變速度
                double e_tilde_zeta_half = e[alpha][1] * dk_dy_half
                                         + e[alpha][2] * dk_dz_half;

                // 步驟四：完整 RK2 位移量（Imamura 2005 Eq.20）
                delta_zeta_h[alpha * sz + idx_jk] = dt * e_tilde_zeta_half;
            }
        }
    }
}

// ============================================================================
// Wrapper: precompute both ξ and ζ displacements
// ============================================================================
void PrecomputeGILBM_DeltaXiZeta(
    double *delta_xi_h,      // 輸出: [19]，ξ 方向位移量
    double *delta_zeta_h,    // 輸出: [19 * NYD6 * NZ6]，ζ 方向位移量
    const double *dk_dz_h,   // 輸入: 度量項 dk/dz [NYD6*NZ6]
    const double *dk_dy_h,   // 輸入: 度量項 dk/dy [NYD6*NZ6]
    int NYD6_local,
    int NZ6_local
) {
    double dy_val = LY / (double)(NY6 - 7);

    PrecomputeGILBM_DeltaXi(delta_xi_h, dy_val);
    PrecomputeGILBM_DeltaZeta(delta_zeta_h, dk_dz_h, dk_dy_h, NYD6_local, NZ6_local);
}

#endif
