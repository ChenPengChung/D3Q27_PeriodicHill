#ifndef GILBM_PRECOMPUTE_H
#define GILBM_PRECOMPUTE_H

// GILBM displacement precomputation (Imamura 2005 Eq. 19-20)
// Three directions, called once during initialization; results copied to GPU.
//
// δη[α]     = dt · e_x[α] / dx             → constant per alpha (19 values, uniform x)
// δξ[α]     = dt · e_y[α] / dy             → constant per alpha (19 values, uniform y)
// δζ[α,j,k] = dt · ẽ^ζ_α(k_half)          → RK2 midpoint evaluation [19*NYD6*NZ6]
//
// Entry point: PrecomputeGILBM_DeltaAll() computes all three in one call.

// ============================================================================
// PrecomputeGILBM_DeltaEta: η-direction displacement (constant for uniform x)
// ============================================================================
// For uniform x-grid: dη/dx = 1/dx (constant), dη/dy = dη/dz = 0
// → ẽ^η_α = e_x[α] · (1/dx) = e_x[α] / dx
// → δη[α] = dt · e_x[α] / dx
// No RK2 correction needed (metric is constant → midpoint = endpoint).
//
// When x becomes non-uniform, promote delta_eta from [19] to [19*NYD6*NZ6]
// and add RK2 midpoint interpolation in i-direction.
void PrecomputeGILBM_DeltaEta(
    double *delta_eta_h,   // 輸出: [19]，η 方向位移量（常數）
    double dx_val          // 輸入: uniform grid spacing dx = LX/(NX6-7)
) {
    // D3Q19 離散速度集（與 initialization.h 中一致）
    double e_x[19] = {
        0,
        1, -1, 0, 0, 0, 0,
        1, -1, 1, -1,
        1, -1, 1, -1,
        0, 0, 0, 0
    };

    for (int alpha = 0; alpha < 19; alpha++) {
        delta_eta_h[alpha] = dt * e_x[alpha] / dx_val;
    }
}

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
        if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue; // 若 e_y 和 e_z 皆為零（純 x 方向），則 ẽ^ζ_α ≡ 0，跳過此方向

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
                if (k_lo < 2) k_lo = 2; //k_low最小為2
                if (k_lo > NZ6_local - 4) k_lo = NZ6_local - 4;//k_lo最大為NZ6-4
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
// Wrapper: precompute all three direction displacements (η, ξ, ζ)
// ============================================================================
// η: δη[α] = dt · e_x[α] / dx   (constant, [19])
// ξ: δξ[α] = dt · e_y[α] / dy   (constant, [19])
// ζ: δζ[α,j,k] = dt · ẽ^ζ(k_half)  (RK2, [19*NYD6*NZ6])
void PrecomputeGILBM_DeltaAll(
    double *delta_xi_h,      // 輸出: [19]，ξ 方向位移量
    double *delta_eta_h,     // 輸出: [19]，η 方向位移量
    double *delta_zeta_h,    // 輸出: [19 * NYD6 * NZ6]，ζ 方向位移量
    const double *dk_dz_h,   // 輸入: 度量項 dk/dz [NYD6*NZ6]
    const double *dk_dy_h,   // 輸入: 度量項 dk/dy [NYD6*NZ6]
    int NYD6_local,
    int NZ6_local
) {
    double dx_val = LX / (double)(NX6 - 7);
    double dy_val = LY / (double)(NY6 - 7);

    PrecomputeGILBM_DeltaEta(delta_eta_h, dx_val);
    PrecomputeGILBM_DeltaXi(delta_xi_h, dy_val);
    PrecomputeGILBM_DeltaZeta(delta_zeta_h, dk_dz_h, dk_dy_h, NYD6_local, NZ6_local);
}

// ============================================================================
// Phase 3: Imamura's Global Time Step (Imamura 2005 Eq. 22)
// ============================================================================
// Δt_g = λ · min_{i,α,j,k} [ 1 / |c̃_{i,α}|_{j,k} ]
//      = λ / max_{i,α,j,k} |c̃_{i,α}|_{j,k}
//
// where c̃ is the contravariant velocity in each computational direction:
//   η: |c̃^η_α| = |e_x[α]| / dx          (uniform x)
//   ξ: |c̃^ξ_α| = |e_y[α]| / dy          (uniform y)
//   ζ: |c̃^ζ_α| = |e_y·dk_dy + e_z·dk_dz| (space-varying)
//
// This ensures CFL < 1 in ALL directions at ALL grid points.
double ComputeGlobalTimeStep(
    const double *dk_dz_h,
    const double *dk_dy_h,
    double dx_val,
    double dy_val,
    int NYD6_local,
    int NZ6_local,
    double cfl_lambda,
    int myid_local
) {
    double e[19][3] = {
        {0,0,0},
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };

    double max_c_tilde = 0.0;
    int max_dir = -1;  // 0=eta, 1=xi, 2=zeta
    int max_alpha = -1, max_j = -1, max_k = -1;

    // η-direction (uniform x): max|c̃^η| = 1/dx (for |e_x|=1 directions)
    double c_eta = 1.0 / dx_val;
    if (c_eta > max_c_tilde) {
        max_c_tilde = c_eta;
        max_dir = 0; max_alpha = 1; max_j = -1; max_k = -1;
    }

    // ξ-direction (uniform y): max|c̃^ξ| = 1/dy (for |e_y|=1 directions)
    double c_xi = 1.0 / dy_val;
    if (c_xi > max_c_tilde) {
        max_c_tilde = c_xi;
        max_dir = 1; max_alpha = 3; max_j = -1; max_k = -1;
    }

    // ζ-direction (non-uniform z): scan all interior fluid points
    for (int j = 3; j < NYD6_local - 4; j++) {
        for (int k = 3; k < NZ6_local - 3; k++) {
            int idx_jk = j * NZ6_local + k;
            double dk_dy_val = dk_dy_h[idx_jk];
            double dk_dz_val = dk_dz_h[idx_jk];

            for (int alpha = 1; alpha < 19; alpha++) {
                if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;

                double c_zeta = fabs(e[alpha][1] * dk_dy_val
                                   + e[alpha][2] * dk_dz_val);
                if (c_zeta > max_c_tilde) {
                    max_c_tilde = c_zeta;
                    max_dir = 2; max_alpha = alpha;
                    max_j = j; max_k = k;
                }
            }
        }
    }

    double dt_g = cfl_lambda / max_c_tilde;

    if (myid_local == 0) {
        const char *dir_name[] = {"eta (x)", "xi (y)", "zeta (z)"};
        printf("\n=============================================================\n");
        printf("  Phase 3: Imamura Global Time Step (Eq. 22)\n");
        printf("  CFL lambda = %.4f\n", cfl_lambda);
        printf("=============================================================\n");
        printf("  max|c_tilde| = %.6f in %s direction\n",
               max_c_tilde, dir_name[max_dir]);
        if (max_dir == 2) {
            printf("    at alpha=%d (e_y=%+.0f, e_z=%+.0f), j=%d, k=%d\n",
                   max_alpha, e[max_alpha][1], e[max_alpha][2], max_j, max_k);
        }
        printf("  dt_g = lambda / max|c_tilde| = %.6e\n", dt_g);
        printf("  dt_old = minSize = %.6e\n", (double)minSize);
        printf("  ratio dt_g / minSize = %.4f\n", dt_g / (double)minSize);
        printf("  Speedup cost: %.1fx more timesteps per physical time\n",
               (double)minSize / dt_g);
        printf("=============================================================\n\n");
    }

    return dt_g;
}

// ============================================================================
// Phase 4: Local Time Step (Imamura 2005 Eq. 28)
// ============================================================================
// dt_local(j,k) = λ / max_α |c̃_α(j,k)|
// tau_local(j,k) = 0.5 + 3·ν / dt_local(j,k)
// a(j,k) = dt_local / dt_global  (acceleration factor, ≥ 1)
//
// Each (j,k) gets its own CFL-limited time step. Near walls where dk_dz is
// large, dt_local ≈ dt_global. At channel center where dk_dz is small,
// dt_local >> dt_global → faster convergence to steady state.
void ComputeLocalTimeStep(
    double *dt_local_h,          // 輸出: [NYD6*NZ6]
    double *tau_local_h,         // 輸出: [NYD6*NZ6]
    double *tau_dt_product_h,    // 輸出: [NYD6*NZ6] tau*dt for re-estimation
    const double *dk_dz_h,
    const double *dk_dy_h,
    double dx_val, double dy_val,
    double niu_val, double dt_global,
    int NYD6_local, int NZ6_local,
    double cfl_lambda, int myid_local
) {
    double e[19][3] = {
        {0,0,0},
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };

    
    // Constant contributions from η and ξ directions
    double c_eta_max = 1.0 / dx_val;   // max|e_x|/dx = 1/dx
    double c_xi_max  = 1.0 / dy_val;   // max|e_y|/dy = 1/dy
    double c_uniform = (c_eta_max > c_xi_max) ? c_eta_max : c_xi_max;

    int sz = NYD6_local * NZ6_local;
    double a_min = 1e30, a_max = 0.0, a_sum = 0.0;
    int a_count = 0;
    int a_max_j = -1, a_max_k = -1;

    // Fill all points (including halo) with global dt as default
    for (int idx = 0; idx < sz; idx++) {
        dt_local_h[idx] = dt_global;
        tau_local_h[idx] = 0.5 + 3.0 * niu_val / dt_global;
        tau_dt_product_h[idx] = tau_local_h[idx] * dt_global;
    }

    // Compute local dt at interior fluid points
    for (int j = 3; j < NYD6_local - 4; j++) {
        for (int k = 3; k < NZ6_local - 3; k++) {
            int idx_jk = j * NZ6_local + k;
            double dk_dy_val = dk_dy_h[idx_jk];
            double dk_dz_val = dk_dz_h[idx_jk];

            // Find max|c̃| across all directions at this (j,k)
            double max_c = c_uniform;
            for (int alpha = 1; alpha < 19; alpha++) {
                if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;
                double c_zeta = fabs(e[alpha][1] * dk_dy_val
                                   + e[alpha][2] * dk_dz_val);
                if (c_zeta > max_c) max_c = c_zeta;
            }

            double dt_l = cfl_lambda / max_c;
            double tau_l = 0.5 + 3.0 * niu_val / dt_l;

            dt_local_h[idx_jk] = dt_l;
            tau_local_h[idx_jk] = tau_l;
            tau_dt_product_h[idx_jk] = tau_l * dt_l;

            // Track acceleration factor statistics
            double a = dt_l / dt_global;
            if (a < a_min) a_min = a;
            if (a > a_max) { a_max = a; a_max_j = j; a_max_k = k; }
            a_sum += a;
            a_count++;
        }
    }

    if (myid_local == 0 && a_count > 0) {
        printf("\n=============================================================\n");
        printf("  Phase 4: Local Time Step (Imamura 2005 Eq. 28)\n");
        printf("=============================================================\n");
        printf("  dt_global = %.6e\n", dt_global);
        printf("  Acceleration factor a(j,k) = dt_local / dt_global:\n");
        printf("    min(a)  = %.4f  (near wall, CFL-limited)\n", a_min);
        printf("    max(a)  = %.4f  at j=%d, k=%d (channel center)\n",
               a_max, a_max_j, a_max_k);
        printf("    mean(a) = %.4f  (%d interior points)\n",
               a_sum / a_count, a_count);
        printf("  tau range: [%.4f, %.4f]\n",
               0.5 + 3.0 * niu_val / (a_max * dt_global),
               0.5 + 3.0 * niu_val / (a_min * dt_global));
        printf("  dt_local range: [%.6e, %.6e]\n",
               a_min * dt_global, a_max * dt_global);

        // Print k-profile at middle j
        int j_mid = NYD6_local / 2;
        printf("\n  k-profile at j=%d:\n", j_mid);
        printf("  %4s  %12s  %8s  %8s\n", "k", "dt_local", "tau_loc", "a");
        for (int k = 3; k < NZ6_local - 3; k += 3) {
            int idx = j_mid * NZ6_local + k;
            printf("  %4d  %12.6e  %8.4f  %8.4f\n",
                   k, dt_local_h[idx], tau_local_h[idx],
                   dt_local_h[idx] / dt_global);
        }
        printf("=============================================================\n\n");
    }
}

// ============================================================================
// PrecomputeGILBM_DeltaZeta_Local: ζ-direction RK2 with LOCAL dt
// ============================================================================
// Same as PrecomputeGILBM_DeltaZeta but uses dt_local(j,k) instead of global dt.
// The RK2 midpoint k_half depends on dt, so this is NOT a simple scaling.
void PrecomputeGILBM_DeltaZeta_Local(
    double *delta_zeta_h,        // 輸出: [19 * NYD6 * NZ6]
    const double *dk_dz_h,      // 輸入: [NYD6 * NZ6]
    const double *dk_dy_h,      // 輸入: [NYD6 * NZ6]
    const double *dt_local_h,   // 輸入: [NYD6 * NZ6] local time step
    int NYD6_local,
    int NZ6_local
) {
    double e_y[19] = {
        0, 0,0,1,-1,0,0, 1,1,-1,-1, 0,0,0,0, 1,-1,1,-1
    };
    double e_z[19] = {
        0, 0,0,0,0,1,-1, 0,0,0,0, 1,1,-1,-1, 1,1,-1,-1
    };

    int sz = NYD6_local * NZ6_local;

    for (int alpha = 0; alpha < 19; alpha++) {
        if (e_y[alpha] == 0.0 && e_z[alpha] == 0.0) {
            // alpha=0 (rest) or pure x-direction: δζ = 0
            for (int idx = 0; idx < sz; idx++)
                delta_zeta_h[alpha * sz + idx] = 0.0;
            continue;
        }

        for (int j = 0; j < NYD6_local; j++) {
            for (int k = 0; k < NZ6_local; k++) {
                int idx_jk = j * NZ6_local + k;

                // Skip non-interior points (keep zero default)
                if (k < 2 || k >= NZ6_local - 2) {
                    delta_zeta_h[alpha * sz + idx_jk] = 0.0;
                    continue;
                }

                double dk_dy_val = dk_dy_h[idx_jk];
                double dk_dz_val = dk_dz_h[idx_jk];
                double dt_l = dt_local_h[idx_jk];

                // Step 1: contravariant velocity at current point
                double e_tilde_zeta0 = e_y[alpha] * dk_dy_val
                                     + e_z[alpha] * dk_dz_val;

                // Step 2: RK2 midpoint with LOCAL dt
                double k_half = (double)k - 0.5 * dt_l * e_tilde_zeta0;

                // Clamp midpoint to valid interpolation range
                if (k_half < 2.0) k_half = 2.0;
                if (k_half > (double)(NZ6_local - 3)) k_half = (double)(NZ6_local - 3);

                // Step 3: interpolate dk_dy, dk_dz at midpoint
                int k_base = (int)k_half;
                if (k_base < 2) k_base = 2;
                if (k_base >= NZ6_local - 3) k_base = NZ6_local - 4;
                double frac = k_half - (double)k_base;

                int idx0 = j * NZ6_local + k_base;
                int idx1 = j * NZ6_local + k_base + 1;

                double dk_dy_half = dk_dy_h[idx0] * (1.0 - frac) + dk_dy_h[idx1] * frac;
                double dk_dz_half = dk_dz_h[idx0] * (1.0 - frac) + dk_dz_h[idx1] * frac;

                // Step 4: full RK2 displacement with LOCAL dt
                double e_tilde_zeta_half = e_y[alpha] * dk_dy_half
                                         + e_z[alpha] * dk_dz_half;

                delta_zeta_h[alpha * sz + idx_jk] = dt_l * e_tilde_zeta_half;
            }
        }
    }
}

#endif
/*
在曲線座標下的遷移距離計算分三個方向：
Delta[alpha][0=η] = dt * e_x[alpha] / dx        (uniform x, __constant__[19])
Delta[alpha][1=ξ] = dt * e_y[alpha] / dy        (uniform y, __constant__[19])
Delta[alpha][2=ζ][idx_jk] = dt * ẽ^ζ(k_half)   (RK2, device[19*NYD6*NZ6])
Note: 不需要額外乘 minSize — dt 本身就是物理位移（c=1 約定，見 Section 22）
*/
