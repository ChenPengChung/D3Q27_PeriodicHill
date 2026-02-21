#ifndef GILBM_PRECOMPUTE_H
#define GILBM_PRECOMPUTE_H
//phase 1.5：此章節為預計算各個物理空間計算點的非物理空間計算點的空間位置：
//對於同一個物理空間計算點，一共有3*18個量需要計算
//配置三種偏移量二維陣列
//分別為： delta_eta_h[alpha]、delta_xi_h[alpha]、delta_zeta_h[alpha][idx_jk <= NYD6*NZ6]
//先處理後面兩個計算的陣列 ：
//公式：
// δη[α]     = dt · e_x[α] / dx             → constant per alpha (19 values, uniform x)
// δξ[α]     = dt · e_y[α] / dy             → constant per alpha (19 values, uniform y)
// δζ[α,j,k] = dt · ẽ^ζ_α(k_half)          → RK2 midpoint evaluation [19*NYD6*NZ6]



// ============================================================================
// PrecomputeGILBM_DeltaEta: η-direction displacement (constant for uniform x)
// ============================================================================
// For uniform x-grid: dη/dx = 1/dx (constant), dη/dy = dη/dz = 0
// → ẽ^η_α = e_x[α] · (1/dx) = e_x[α] / dx
// → δη[α] = dt · e_x[α] / dx
// No RK2 correction needed (metric is constant → midpoint = endpoint).
//
// NOTE: 壁面邊界節點（k=2 底壁, k=NZ6-3 頂壁）上，若某方向 α 的 ζ 方向
// 逆變速度 ẽ^ζ_α = e_y·dk_dy + e_z·dk_dz > 0（底壁）或 < 0（頂壁），
// 則該 α 由 Chapman-Enskog BC 處理，不走 streaming。
// 此時 delta_eta[α] 雖已計算，但在該壁面節點上不會被 streaming 讀取。
// η 方向為均勻網格，δη 為常數陣列 [19]，不區分壁面/內部，統一預計算。
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
// NOTE: 壁面邊界節點上，BC 方向（ẽ^ζ_α > 0 底壁 / < 0 頂壁）的分佈函數
// 由 Chapman-Enskog BC 處理，不走 streaming，因此 delta_xi[α] 在壁面節點
// 對該 α 無效（不被讀取）。ξ 為均勻網格，統一預計算所有 α。
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
/*void PrecomputeGILBM_DeltaXi 說明：
- 對一維陣列寫入資料
- 計算曲線座標系，各個編號的xi方向偏移距離
- 輸出一維陣列 delta_xi_h[19]，每個 alpha 對應一個常數偏移量
- 輸入物理空間的dy : dy_val = LY/(NY6-7)，用於計算偏移距離
- 公式：
→ ẽ^ξ_α = e_y[α] · (1/dy) = e_y[α] / dy
→ δξ[α] = dt · e_y[α] / dy
*/

// ============================================================================
// PrecomputeGILBM_DeltaZeta: ζ-direction RK2 displacement (space-varying)
// ============================================================================
// Imamura 2005 Eq. 19-20:
//   Step 1: ẽ^ζ_α(k) = e_y·(dk/dy) + e_z·(dk/dz)  at current point
//   Step 2: k_half = k - 0.5·dt·ẽ^ζ_α(k)            RK2 midpoint
//   Step 3: Interpolate dk/dy, dk/dz at k_half
//   Step 4: δζ[α] = dt · ẽ^ζ_α(k_half)              full RK2 displacement
//
// 注意：壁面邊界節點（k=2 底壁, k=NZ6-3 頂壁）的 BC/streaming 分類：
//   ẽ^ζ_α = e_y[α]·dk_dy + e_z[α]·dk_dz
//   底壁：ẽ^ζ_α > 0 → 出發點在壁面以下 → BC 方向（Chapman-Enskog 處理）
//   底壁：ẽ^ζ_α ≤ 0 → 出發點在流體內部 → streaming 方向（使用 δζ 插值）
//   頂壁：ẽ^ζ_α < 0 → BC 方向；ẽ^ζ_α ≥ 0 → streaming 方向
//
//   平坦底壁 (dk_dy=0) 的 BC 方向: α={5,11,12,15,16}（e_z>0，共 5 個）
//   斜面底壁 (dk_dy≠0, slope<45°): BC 方向增至 8 個（額外含 e_y 分量方向）
//
// δζ 對所有 19 方向統一計算（含 BC 方向），但 BC 方向的值不被 streaming 讀取。
// 統一計算原因：
//(1) BC 方向隨 (j,k) 的 dk_dy 變化，條件判斷反增複雜度；
// (2) D3Q19 的 (e_y,e_z)↔(-e_y,-e_z) 對稱性保證 BC 方向的 δζ 值有限且無害。
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
    /*分類：
    ẽ^ζ_α(k) = e_y·(dk/dy) + e_z·(dk/dz)  at current point
    e_y=0 者：alpha = 1,2,5,6,11,12,13,14 ; (其中，e_z=0 ： 1,2.)
    e_y!=0 者：alpha = 3,4,7,8,9,10,15,16,17,18 ; (其中，e_z=0 ： 3,4,7,8,9,10)
    策略：先計算半步偏移位置
    再計算逆變速度分量e_zeta在該半步偏移位置的量
    e_y·(dk/dy) + e_z·(dk/dz)
    其中，需要線性插值：離散度量項 dk/dz 和 dk/dy 在半步位置的值
    */
    for(int alpha = 1 ; alpha <= 18 ; alpha++){
        //若y方向速度分量=0且z方向速度分量=0，則跳過不特別寫入資料；
        if(e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;//就是指alpha = 1,2
       for (int j = 3; j < NYD6_local - 4; j++) { // 跳過 y 方向 buffer layer
            for (int k = 2; k < NZ6_local - 2; k++) {
                int idx_xi = j * NZ6_local + k ; 
                //step1:計算該位置點該編號尿變速度zeta分量：
                //For (j,k) For alpha=
                double e_alpha_k = e[alpha][1] * dk_dy_h[idx_xi] + e[alpha][2] * dk_dz_h[idx_xi];
                //但是我們要算的不是當前計算點上的值，而是半步長位置點的值
                //step2:對於每一個空間點對於每一個編號計算半步長位置點
                double k_half = (double)k - 0.5 * dt * e_alpha_k;
                //但是半步長位置點作為一個非物理空間計算點，沒有設定值，所以需要做線性插值，插直到該半步長非物理空間計算點
                //公式e_alpha_k = e[alpha][1] * dk_dy_half + e[alpha][2] * dk_dz_half
                //根據上述公式，我們插值的對象為dk_dy_half和dk_dz_half
                //當前座標為(j,k) , alpha編號下，半步長位置點為(j,k_half), 
                //step3:尋找要內插的物理空間計算點位置
                int k_low = (int)floor(k_half); //k_low為半步長位置點所在的物理空間計算點的下界位置
                //給k_low設定最小為2，最大為NZ6_local-4，確保內插的兩個物理空間計算點都在有效範圍內
                if(k_low < 2) k_low = 2;
                if(k_low > NZ6_local - 4) k_low = NZ6_local - 3 -1 ; 
                double frac = k_half - (double)k_lo;
                if (frac < 0.0) frac = 0.0;
                if (frac > 1.0) frac = 1.0;
                int idx_xi_low = j * NZ6_local + k_low;
                int idx_xi_high = j * NZ6_local + k_low + 1;
                double dk_dy_half = dk_dy_h[idx_xi_low] * (1.0 - frac) + dk_dy_h[idx_xi_high] * frac;
                double dk_dz_half = dk_dz_h[idx_xi_low] * (1.0 - frac) + dk_dz_h[idx_xi_high] * frac;
                //step4:結合起來計算半步長位置點的逆變速度（對於alpha編號，對於空間點j,k_half)
                double e_alpha_k_half = e[alpha][1] * dk_dy_half + e[alpha][2] * dk_dz_half;
                //step5:寫入陣列，對於該空間點(j,k)對於編號alpha，zeta方向的非物理空間計算點的偏移量
                delta_zeta_h[alpha * NYD6_local * NZ6_local + idx_xi] = dt * e_alpha_k_half;
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
//目標：求最大的速度分量c̃
    //比較維度：各個空間計算點(j,k),各個速度編號alpha,三個分量，求最大值
    //比較順序：分量->編號->空間點
    //step1:初始化：
    double max_c_tilde = 0.0 ; 
    int max_component = -1 ; //1.0:eta, 1:xi, 2:zeta
    int max_alpha = -1; //2.
    int max_j = -1, max_k = -1;//3.
    
    // η-direction (uniform x): max|c̃^η| = 1/dx (for |e_x|=1 directions)
    double c_eta = 1.0 / dx_val;
    if (c_eta > max_c_tilde) {
        max_c_tilde = c_eta;
        max_component = 0; //1.
        max_alpha = 1; //2.
        max_j = -1; max_k = -1; //3.個點相同
    }
    // ξ-direction (uniform y): max|c̃^ξ| = 1/dy (for |e_y|=1 directions)
    double c_xi = 1.0 / dy_val;
    if (c_xi > max_c_tilde) {
        max_c_tilde = c_xi;
        max_component = 1; //1.
        max_alpha = 3; //2.
        max_j = -1; max_k = -1; //3.個點相同
    }
    // ζ-direction (non-uniform z): scan all fluid points including wall (k=2, k=NZ6-3)
    // 壁面 dk_dz 最大（tanh 拉伸最密處），是全場 CFL 最嚴格的約束點。
    // 雖然壁面有部分方向由 BC 處理（不做 streaming），但 D3Q19 的
    // (e_y,e_z)↔(-e_y,-e_z) 對稱性保證 max|c̃^ζ| 在 streaming 方向與
    // BC 方向完全相同，因此掃描所有方向不會高估 CFL 約束。
    for (int j = 3 ; j < NYD6_local-3 ; j++){
        for(int k = 2 ; k <= NZ6_local-3 ; k++){
            int idx_jk = j * NZ6_local + k;
            double dk_dy_val = dk_dy_h[idx_jk];
            double dk_dz_val = dk_dz_h[idx_jk];
            for(int alpha = 1 ; alpha <19 ; alpha++){
                if(e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;
                double c_zeta = fabs(e[alpha][1] * dk_dy_val + e[alpha][2] * dk_dz_val);
                if(c_zeta > max_c_tilde){
                    max_c_tilde = c_zeta;
                    max_component = 2; //1.
                    max_alpha = alpha; //2.
                    max_j = j; max_k = k; //3.
                }
            }
        }
    }

    double dt_g = cfl_lambda / max_c_tilde;

    if (myid_local == 0) {
        const char *dir_name[] = {"eta (x)", "xi (y)", "zeta (z)"};
        std::cout << "\n=============================================================\n"
                  << "  Phase 3: Imamura Global Time Step (Eq. 25)\n"
                  << "  CFL lambda = " << std::fixed << std::setprecision(4) << cfl_lambda << "\n"
                  << "=============================================================\n"
                  << "  max|c_tilde| = " << std::setprecision(6) << max_c_tilde
                  << " in " << dir_name[max_dir] << " direction\n";
        if (max_dir == 2) {
        std::cout << "    at alpha=" << max_alpha
                  << " (e_y=" << std::showpos << std::setprecision(0) << std::fixed << (double)e[max_alpha][1]
                  << ", e_z=" << (double)e[max_alpha][2] << std::noshowpos
                  << "), j=" << max_j << ", k=" << max_k << "\n";
        }
        std::cout << std::noshowpos
                  << "  dt_g = lambda / max|c_tilde| = " << std::scientific << std::setprecision(6) << dt_g << "\n"
                  << "  dt_old = minSize = " << (double)minSize << "\n"
                  << std::fixed << std::setprecision(4)
                  << "  ratio dt_g / minSize = " << dt_g / (double)minSize << "\n"
                  << std::setprecision(1)
                  << "  Speedup cost: " << (double)minSize / dt_g << "x more timesteps per physical time\n"
                  << "=============================================================\n\n";
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

    // Compute local dt at all fluid points including wall (k=2, k=NZ6-3)
    // 壁面 dk_dz 最大 → dt_local 最小（CFL 最嚴格），acceleration factor ≈ 1。
    // D3Q19 對稱性：max|c̃^ζ| 不受 BC/streaming 方向過濾影響（見 ComputeGlobalTimeStep 註解）。
    for (int j = 3; j < NYD6_local - 4; j++) {
        for (int k = 2; k < NZ6_local - 2; k++) {
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
//
// 壁面 BC/streaming 分類同 PrecomputeGILBM_DeltaZeta：BC 方向的 δζ 被計算但
// 不被 streaming 使用（由 Chapman-Enskog BC 處理）。
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
