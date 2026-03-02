#ifndef GILBM_PRECOMPUTE_H
#define GILBM_PRECOMPUTE_H

#include <iostream>
#include <iomanip>
//phase 1.5：此章節為預計算各個物理空間計算點的非物理空間計算點的空間位置：
//對於同一個物理空間計算點，一共有3*18個量需要計算
//配置三種偏移量二維陣列
//分別為： delta_eta_h[alpha]、delta_xi_h[alpha]、delta_zeta_h[alpha][idx_jk <= NYD6*NZ6]
//先處理後面兩個計算的陣列 ：
//公式：
// δη[α]     = dt_global · e_x[α] / dx           → constant per alpha (19 values, uniform x)
// δξ[α]     = dt_global · e_y[α] / dy           → constant per alpha (19 values, uniform y)
// δζ[α,j,k] = dt_local(j,k) · ẽ^ζ_α(k_half)   → RK2 midpoint evaluation [19*NYD6*NZ6]



// ============================================================================
// PrecomputeGILBM_DeltaEta: η-direction displacement (constant for uniform x)
// ============================================================================
// For uniform x-grid: dη/dx = 1/dx (constant), dη/dy = dη/dz = 0
// → ẽ^η_α = e_x[α] · (1/dx) = e_x[α] / dx
// → δη[α] = dt_global · e_x[α] / dx
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
    double dx_val,         // 輸入: uniform grid spacing dx = LX/(NX6-7)
    double dt_val          // 輸入: 參考時間步長 (dt_global)
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
        delta_eta_h[alpha] = dt_val * e_x[alpha] / dx_val;
    }
}

// ============================================================================
// PrecomputeGILBM_DeltaXi: ξ-direction displacement (constant for uniform y)
// ============================================================================
// For uniform y-grid: dξ/dy = 1/dy (constant), dξ/dz = 0
// → ẽ^ξ_α = e_y[α] · (1/dy) = e_y[α] / dy
// → δξ[α] = dt_global · e_y[α] / dy
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
    double dy_val,         // 輸入: uniform grid spacing dy = LY/(NY6-7)
    double dt_val          // 輸入: 參考時間步長 (dt_global)
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
        delta_xi_h[alpha] = dt_val * e_y[alpha] / dy_val;
    }
}
/*void PrecomputeGILBM_DeltaXi 說明：
- 對一維陣列寫入資料
- 計算曲線座標系，各個編號的xi方向偏移距離
- 輸出一維陣列 delta_xi_h[19]，每個 alpha 對應一個常數偏移量
- 輸入物理空間的dy : dy_val = LY/(NY6-7)，用於計算偏移距離
- 公式：
→ ẽ^ξ_α = e_y[α] · (1/dy) = e_y[α] / dy
→ δξ[α] = dt_global · e_y[α] / dy
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
// 注意：壁面邊界節點（k=3 底壁, k=NZ6-4 頂壁）的 BC/streaming 分類：
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
    int NZ6_local,
    double dt_val 
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
       for (int j = 3; j < NYD6_local - 3; j++) { // 跳過 y 方向 buffer layer (含 MPI 重疊點)
            for (int k = 3; k < NZ6_local - 3; k++) {  // Buffer=3: k=3(壁面)..NZ6-4(頂壁)
                int idx_xi = j * NZ6_local + k ;
                double e_alpha_k = e[alpha][1] * dk_dy_h[idx_xi] + e[alpha][2] * dk_dz_h[idx_xi];
                double k_half = (double)k - 0.5 * dt_val * e_alpha_k;
                int k_low = (int)floor(k_half);
                // Buffer=3: clamp 到有效度量項範圍 [3, NZ6-5]
                if(k_low < 3) k_low = 3;
                if(k_low > NZ6_local - 5) k_low = NZ6_local - 4 - 1;
                double frac = k_half - (double)k_low;
                if (frac < 0.0) frac = 0.0;
                if (frac > 1.0) frac = 1.0;
                int idx_xi_low = j * NZ6_local + k_low;
                int idx_xi_high = j * NZ6_local + k_low + 1;
                double dk_dy_half = dk_dy_h[idx_xi_low] * (1.0 - frac) + dk_dy_h[idx_xi_high] * frac;
                double dk_dz_half = dk_dz_h[idx_xi_low] * (1.0 - frac) + dk_dz_h[idx_xi_high] * frac;
                //step4:結合起來計算半步長位置點的逆變速度（對於alpha編號，對於空間點j,k_half)
                double e_alpha_k_half = e[alpha][1] * dk_dy_half + e[alpha][2] * dk_dz_half;
                //step5:寫入陣列，對於該空間點(j,k)對於編號alpha，zeta方向的非物理空間計算點的偏移量
                delta_zeta_h[alpha * NYD6_local * NZ6_local + idx_xi] = dt_val * e_alpha_k_half;
            }
        }
    }
}

// ============================================================================
// Wrapper: precompute all three direction displacements (η, ξ, ζ)
// ============================================================================
// η: δη[α] = dt_global · e_x[α] / dx   (constant, [19])
// ξ: δξ[α] = dt_global · e_y[α] / dy   (constant, [19])
// ζ: δζ[α,j,k] = dt_local(j,k) · ẽ^ζ(k_half)  (RK2, [19*NYD6*NZ6])
void PrecomputeGILBM_DeltaAll(
    double *delta_xi_h,      // 輸出: [19]，ξ 方向位移量
    double *delta_eta_h,     // 輸出: [19]，η 方向位移量
    double *delta_zeta_h,    // 輸出: [19 * NYD6 * NZ6]，ζ 方向位移量
    const double *dk_dz_h,   // 輸入: 度量項 dk/dz [NYD6*NZ6]
    const double *dk_dy_h,   // 輸入: 度量項 dk/dy [NYD6*NZ6]
    int NYD6_local,
    int NZ6_local,
    double dt_val            // 輸入: 參考時間步長 (dt_global)
) {
    double dx_val = LX / (double)(NX6 - 7);
    double dy_val = LY / (double)(NY6 - 7);

    PrecomputeGILBM_DeltaEta(delta_eta_h, dx_val, dt_val);
    PrecomputeGILBM_DeltaXi(delta_xi_h, dy_val, dt_val);
    PrecomputeGILBM_DeltaZeta(delta_zeta_h, dk_dz_h, dk_dy_h, NYD6_local, NZ6_local, dt_val);
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
    int myid_local,
    int nprocs_local
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
    int max_component = -1 ; //0:eta, 1:xi, 2:zeta
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
    // ζ-direction (non-uniform z): scan all fluid points including wall (k=3, k=NZ6-4)
    // Buffer=3: 壁面在 k=3 和 k=NZ6-4
    for (int j = 3 ; j < NYD6_local-3 ; j++){
        for(int k = 3 ; k <= NZ6_local-4 ; k++){
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

    // --- Per-rank sequential output (MPI_Barrier 確保輸出順序) ---
    const char *dir_name[] = {"eta (x)", "xi (y)", "zeta (z)"};
    if (myid_local == 0) {
        std::cout << "\n=============================================================\n"
                  << "  Phase 3: Imamura Global Time Step (Eq. 25)\n"
                  << "  CFL lambda = " << std::fixed << std::setprecision(4) << cfl_lambda << "\n"
                  << "=============================================================\n";
    }
    for (int r = 0; r < nprocs_local; r++) {
        CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
        if (myid_local == r) {
            std::cout << "  Rank " << r << ": max|c_tilde| = "
                      << std::fixed << std::setprecision(6) << max_c_tilde
                      << " in " << dir_name[max_component] << " direction";
            if (max_component == 2) {
                std::cout << " at alpha=" << max_alpha
                          << " (e_y=" << std::showpos << std::setprecision(0) << std::fixed
                          << (double)e[max_alpha][1]
                          << ", e_z=" << (double)e[max_alpha][2] << std::noshowpos
                          << "), j=" << max_j << ", k=" << max_k;
            }
            std::cout << ", dt_rank = " << std::scientific << std::setprecision(6) << dt_g
                      << std::endl;
        }
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    return dt_g;
}

// ============================================================================
// Phase 4: Local Time Step (Imamura 2005 Eq. 28)
// ============================================================================
// dt_local(j,k) = λ / max_α |c̃_α(j,k)|
// omega_local(j,k) = 0.5 + 3·ν / dt_local(j,k)
// a(j,k) = dt_local / dt_global  (acceleration factor, ≥ 1)
//
// Each (j,k) gets its own CFL-limited time step. Near walls where dk_dz is
// large, dt_local ≈ dt_global. At channel center where dk_dz is small,
// dt_local >> dt_global → faster convergence to steady state.
void ComputeLocalTimeStep(
    double *dt_local_h,          // 輸出: [NYD6*NZ6]
    double *omega_local_h,         // 輸出: [NYD6*NZ6]
    double *omegadt_local_h,    // 輸出: [NYD6*NZ6] omega*dt for re-estimation
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
    double c_uniform = (c_eta_max > c_xi_max) ? c_eta_max : c_xi_max; //先比較兩者中較大者

    int sz = NYD6_local * NZ6_local;//k-j平面總計算點數量
    double a_min = 1e30, a_max = 0.0, a_sum = 0.0; //加速因子記錄器
    int a_count = 0 ; //加速因子累加器
    int a_max_j = -1, a_max_k = -1;

    // Fill all points (including halo) with global dt as default
    for (int idx_xi = 0; idx_xi < sz; idx_xi++) {
        dt_local_h[idx_xi] = dt_global;
        omega_local_h[idx_xi] = 0.5 + 3.0 * niu_val / dt_global; // ω = 3ν/Δt + 0.5
        omegadt_local_h[idx_xi] = omega_local_h[idx_xi] * dt_global;
    }

    // Compute local dt at all fluid points including wall (k=3, k=NZ6-4)
    // Buffer=3: 壁面在 k=3 和 k=NZ6-4
    for (int j = 3; j < NYD6_local - 3; j++) {  // 含 MPI 重疊點 j=NYD6-4
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

            double dt_local = cfl_lambda / max_c;
            double omega_local = 0.5 + 3.0 * niu_val / dt_local; // ω_local = 3ν/Δt_local + 0.5
            
            dt_local_h[idx_jk] = dt_local; //一個點存一個最小值，形成local time step 
            omega_local_h[idx_jk] = omega_local;
            omegadt_local_h[idx_jk] = omega_local * dt_local; //這才是真正的relaxation time，一般定義

            // Track acceleration factor statistics
            double a = dt_local / dt_global;//加速因子計算公式
            if (a < a_min) a_min = a;
            if (a > a_max) { a_max = a; a_max_j = j; a_max_k = k; }
            a_sum += a;
            a_count++;
        }
    }

    if (myid_local == 0 && a_count > 0) {
        std::cout << "\n=============================================================\n"
                  << "  Phase 4: Local Time Step Calulating (Imamura 2005 Eq. 28)\n"
                  << "=============================================================\n"
                  << "  dt_global = " << std::scientific << std::setprecision(6) << dt_global << "\n"
                  << "  Acceleration factor a(j,k) = dt_local / dt_global:\n"
                  << std::fixed << std::setprecision(4)
                  << "    min(a)  = " << a_min << "  (near wall, CFL-limited)\n"
                  << "    max(a)  = " << a_max
                  << "    at j=" << a_max_j << ", k=" << a_max_k << " (channel center)\n"
                  << "    mean(a) = " << a_sum / a_count
                  << "  (" << a_count << " interior points)\n"
                  << "  omega_local range: ["
                  << 0.5 + 3.0 *niu/ (a_max * dt_global) << ", "
                  << 0.5 + 3.0 *niu/ (a_min * dt_global) << "]\n"
                  << "  dt_local range: ["
                  << std::scientific << std::setprecision(6)
                  << a_min * dt_global << ", "
                  << a_max * dt_global << "]\n";

        // Print k-profile at middle j
        int j_mid = NYD6_local / 2;
        std::cout << "\n  k-profile at j=" << j_mid << ":\n"
                  << "  " << std::setw(4) << "k"
                  << "  " << std::setw(12) << "dt_local"
                  << "  " << std::setw(8) << "omega_local"
                  << "  " << std::setw(8) << "a" << "\n";
        for (int k = 4; k < NZ6_local - 4; k += 3) {  // Buffer=3: 從第一內點 k=4 開始
            int idx = j_mid * NZ6_local + k;
            std::cout << "  " << std::setw(4) << k
                      << "  " << std::scientific << std::setprecision(6) << std::setw(12) << dt_local_h[idx]
                      << "  " << std::fixed << std::setprecision(4) << std::setw(8) << omega_local_h[idx]
                      << "  " << std::setw(8) << dt_local_h[idx] / dt_global << "\n";
        }
        std::cout << "=============================================================\n\n";
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
void PrecomputeGILBM_DeltaZeta_Local( //函數是在main.cu中被呼叫的，目的是為了計算每個空間點(j,k)在每個速度方向alpha下的zeta方向的位移量delta_zeta
    double *delta_zeta_h,        // 輸出: [19 * NYD6 * NZ6]
    const double *dk_dz_h,      // 輸入: [NYD6 * NZ6]
    const double *dk_dy_h,      // 輸入: [NYD6 * NZ6]
    const double *dt_local_h,   // 輸入: [NYD6 * NZ6] local time step
    int NYD6_local,
    int NZ6_local
) {
    double e_y[19] = {
        0, 0,0,1,-1,0,0, 1,1,-1,-1, 0,0,0,0, 1,-1,1,-1
    };//正規化離離散化粒子速度集y分量 
    double e_z[19] = {
        0, 0,0,0,0,1,-1, 0,0,0,0, 1,1,-1,-1, 1,1,-1,-1
    };//正規化離離散化粒子速度集Z分量 

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
                // Buffer=3: 計算範圍 k=3..NZ6-4
                if (k < 3 || k >= NZ6_local - 3) {
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
                // Buffer=3: 有效範圍 [3, NZ6-4]
                if (k_half < 3.0) k_half = 3.0;
                if (k_half > (double)(NZ6_local - 4)) k_half = (double)(NZ6_local - 4);

                // Step 3: interpolate dk_dy, dk_dz at midpoint
                int k_base = (int)k_half;
                if (k_base < 3) k_base = 3;
                if (k_base >= NZ6_local - 4) k_base = NZ6_local - 5;
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

// ============================================================================
// PrecomputeGILBM_DeltaEta_Local: η-direction displacement with LOCAL dt
// ============================================================================
// Replaces the old __constant__[19] approach (dt_global × e_x/dx, scaled by a_local in kernel).
// New: precompute directly with dt_local(j,k) → [19 × NYD6 × NZ6]
// Formula: delta_eta_local[q, j, k] = dt_local(j,k) · e_x[q] / dx
// Eliminates runtime a_local computation in kernel.
void PrecomputeGILBM_DeltaEta_Local(
    double *delta_eta_local_h,   // output: [19 * NYD6 * NZ6]
    const double *dt_local_h,    // input:  [NYD6 * NZ6]
    double dx_val,
    int NYD6_local, int NZ6_local
) {
    double e_x[19] = {
        0, 1,-1,0,0,0,0, 1,-1,1,-1, 1,-1,1,-1, 0,0,0,0
    };
    int sz = NYD6_local * NZ6_local;
    for (int q = 0; q < 19; q++) {
        if (e_x[q] == 0.0) {
            // Pure zero: q=0,3,4,5,6,15,16,17,18
            for (int idx = 0; idx < sz; idx++)
                delta_eta_local_h[q * sz + idx] = 0.0;
        } else {
            double e_over_dx = e_x[q] / dx_val;
            for (int idx = 0; idx < sz; idx++)
                delta_eta_local_h[q * sz + idx] = dt_local_h[idx] * e_over_dx;
        }
    }
}

// ============================================================================
// PrecomputeGILBM_DeltaXi_Local: ξ-direction displacement with LOCAL dt
// ============================================================================
// Same as DeltaEta_Local but for y-direction.
// Formula: delta_xi_local[q, j, k] = dt_local(j,k) · e_y[q] / dy
void PrecomputeGILBM_DeltaXi_Local(
    double *delta_xi_local_h,    // output: [19 * NYD6 * NZ6]
    const double *dt_local_h,    // input:  [NYD6 * NZ6]
    double dy_val,
    int NYD6_local, int NZ6_local
) {
    double e_y[19] = {
        0, 0,0,1,-1,0,0, 1,1,-1,-1, 0,0,0,0, 1,-1,1,-1
    };
    int sz = NYD6_local * NZ6_local;
    for (int q = 0; q < 19; q++) {
        if (e_y[q] == 0.0) {
            for (int idx = 0; idx < sz; idx++)
                delta_xi_local_h[q * sz + idx] = 0.0;
        } else {
            double e_over_dy = e_y[q] / dy_val;
            for (int idx = 0; idx < sz; idx++)
                delta_xi_local_h[q * sz + idx] = dt_local_h[idx] * e_over_dy;
        }
    }
}

// ============================================================================
// Host-side Lagrange 7-point interpolation coefficients
// ============================================================================
// Identical logic to the __device__ version in interpolation_gilbm.h,
// but callable from host code for precomputation.
static inline void lagrange_7point_coeffs_host(double t, double a[7]) {
    for (int k = 0; k < 7; k++) {
        double L = 1.0;
        for (int j = 0; j < 7; j++) {
            if (j != k) L *= (t - (double)j) / (double)(k - j);
        }
        a[k] = L;
    }
}

// ============================================================================
// PrecomputeGILBM_LagrangeWeights: precompute all Lagrange interpolation weights
// ============================================================================
// For each streaming direction q at each spatial point (j,k), the Lagrange
// parameter t depends only on (q,j,k), NOT on the spatial index i:
//   η: t_eta = 3 - delta_eta_local[q,j,k]     (bi=i-3 for all executed i)
//   ξ: t_xi  = 3 - delta_xi_local[q,j,k]      (bj=j-3 for all executed j)
//   ζ: t_zeta = up_k - bk(k)                   (with stencil base clamping)
//
// This eliminates 3 × lagrange_7point_coeffs calls (each 42 divisions) per q
// from the kernel hot path → replaced by 21 cached memory reads per q.
//
// Memory layout: w[q * 7 * NYD6 * NZ6 + c * NYD6 * NZ6 + j * NZ6 + k]
//   where q ∈ [0,18] direction, c ∈ [0,6] Lagrange basis index (q outermost).
// Total per array: 7 × 19 × NYD6 × NZ6 × 8 bytes ≈ 5.6 MB.
void PrecomputeGILBM_LagrangeWeights(
    double *w_eta_h,                   // output: [7 * 19 * NYD6 * NZ6]
    double *w_xi_h,                    // output: [7 * 19 * NYD6 * NZ6]
    double *w_zeta_h,                  // output: [7 * 19 * NYD6 * NZ6]
    const double *delta_eta_local_h,   // input: [19 * NYD6 * NZ6]
    const double *delta_xi_local_h,    // input: [19 * NYD6 * NZ6]
    const double *delta_zeta_h,        // input: [19 * NYD6 * NZ6]
    int NYD6_local, int NZ6_local
) {
    int sz = NYD6_local * NZ6_local;
    size_t total = (size_t)7 * 19 * sz;

    // Initialize to zero (for q=0 rest direction, BC directions, and ghost zones)
    memset(w_eta_h,  0, total * sizeof(double));
    memset(w_xi_h,   0, total * sizeof(double));
    memset(w_zeta_h, 0, total * sizeof(double));

    for (int q = 1; q < 19; q++) {
        for (int j = 3; j < NYD6_local - 3; j++) {
            for (int k = 3; k < NZ6_local - 3; k++) {
                int idx_jk = j * NZ6_local + k;
                int q_base = q * 7 * sz;  // base index for this q in [q][c][idx] layout

                // ── η weights ──
                // t_eta = ci - delta_eta_local, where ci = 3 for all executed i
                // (bi = i-3 for i ∈ [3, NX6-4], never clamped)
                double d_eta = delta_eta_local_h[q * sz + idx_jk];
                double t_eta = 3.0 - d_eta;
                // Safety clamp (CFL < 1 guarantees this won't trigger)
                if (t_eta < 0.0) t_eta = 0.0;
                if (t_eta > 6.0) t_eta = 6.0;
                double a_eta[7];
                lagrange_7point_coeffs_host(t_eta, a_eta);
                for (int c = 0; c < 7; c++)
                    w_eta_h[q_base + c * sz + idx_jk] = a_eta[c];

                // ── ξ weights ──
                // t_xi = cj - delta_xi_local, where cj = 3 for all executed j
                // (bj = j-3 for j ∈ [3, NYD6-4], never clamped)
                double d_xi = delta_xi_local_h[q * sz + idx_jk];
                double t_xi = 3.0 - d_xi;
                if (t_xi < 0.0) t_xi = 0.0;
                if (t_xi > 6.0) t_xi = 6.0;
                double a_xi[7];
                lagrange_7point_coeffs_host(t_xi, a_xi);
                for (int c = 0; c < 7; c++)
                    w_xi_h[q_base + c * sz + idx_jk] = a_xi[c];

                // ── ζ weights (with stencil base clamping) ──
                // Reproduce exact kernel logic: compute_stencil_base + departure clamp
                double d_zeta = delta_zeta_h[q * sz + idx_jk];
                double up_k = (double)k - d_zeta;
                // Departure point clamping (same as kernel L298-299)
                if (up_k < 3.0)                          up_k = 3.0;
                if (up_k > (double)(NZ6_local - 4))      up_k = (double)(NZ6_local - 4);
                // Stencil base clamping (same as compute_stencil_base L67,72-73)
                int bk = k - 3;
                if (bk < 3)                    bk = 3;
                if (bk + 6 > NZ6_local - 4)   bk = NZ6_local - 10;
                double t_zeta = up_k - (double)bk;
                double a_zeta[7];
                lagrange_7point_coeffs_host(t_zeta, a_zeta);
                for (int c = 0; c < 7; c++)
                    w_zeta_h[q_base + c * sz + idx_jk] = a_zeta[c];
            }
        }
    }
}

// ============================================================================
// PrecomputeGILBM_StencilBaseK: precompute z-direction stencil base with wall clamping
// ============================================================================
// bk depends ONLY on k (not on q, j, or i), so a 1D array [NZ6] suffices.
// Kernel access: bk_precomp_d[k] (direct indexing, no offset).
// bk_h[0,1,2] and bk_h[NZ6-3..NZ6-1] are ghost/buffer — kernel guard skips them.
// Reproduces compute_stencil_base() z-clamping logic:
//   bk = k - 3
//   if (bk < 3)           bk = 3            (bottom wall: stencil starts at k=3)
//   if (bk + 6 > NZ6 - 4) bk = NZ6 - 10    (top wall: stencil ends at k=NZ6-4)
void PrecomputeGILBM_StencilBaseK(
    int *bk_h,          // output: [NZ6] (indexed directly by k)
    int NZ6_local
) {
    for (int k = 0; k < NZ6_local; k++) {
        int bk = k - 3;
        if (bk < 3)                    bk = 3;
        if (bk + 6 > NZ6_local - 4)   bk = NZ6_local - 10;
        bk_h[k] = bk;
    }
}

#endif
/*
在曲線座標下的遷移距離計算分三個方向：
Delta[alpha][0=η][j,k] = dt_local(j,k) * e_x[alpha] / dx    (uniform x, device[19*NYD6*NZ6])
Delta[alpha][1=ξ][j,k] = dt_local(j,k) * e_y[alpha] / dy    (uniform y, device[19*NYD6*NZ6])
Delta[alpha][2=ζ][j,k] = dt_local(j,k) * ẽ^ζ(k_half)        (RK2, device[19*NYD6*NZ6])
所有位移量均以 dt_local 預計算，kernel 中不做任何位移縮放。

Lagrange 權重以 [19×7×NYD6×NZ6] 預計算 (q outermost, c middle)，kernel 中直接讀取，不做任何 lagrange_7point_coeffs 計算。
bk 預計算為 int [NZ6]，kernel 中直接讀取，不需 compute_stencil_base()。
*/
