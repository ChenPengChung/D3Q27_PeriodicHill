#ifndef GILB4ru4nj04_PRECOMPUTE_H
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
void PrecomputeGILBM_DeltaXi(
    double *delta_xi_h,    // 輸出: [19]，ξ 方向位移量（常數）
    double dy_val          // 輸入: uniform grid spacing dy = LY/(NY6-7)
) {
    // D3Q19 標準晶格離散化粒子速度集的y分量
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
//: 評述：只有變換後的zeta方向需要做離散積分求得
void PrecomputeGILBM_Deltazeta(
    double *delta_zeta_h,    // 輸出: [19 * NYD6 * NZ6]，預計算的位移量
    const double *dk_dz_h,   // 輸入: 度量項 dk/dz [NYD6*NZ6]
    const double *dk_dy_h,   // 輸入: 度量項 dk/dy [NYD6*NZ6]
    int NYD6_local,
    int NZ6_local
){
    //完整的D3Q19離散速度集
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
    PrecomputeGILBM_Deltazeta(delta_zeta_h, dk_dz_h, dk_dy_h, NYD6_local, NZ6_local);
}

// ============================================================================
// Phase 3: Imamura's Global Time Step (Imamura 2005 Eq. 25)
// ============================================================================
//Global time step 為便利每一個物理空間計算點每一個編號，每一個分量比較下的結果
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
    // ζ-direction (non-uniform z): scan all interior fluid points
    
  


