#ifndef GILBM_PRECOMPUTE_H
#define GILBM_PRECOMPUTE_H

// Phase 1: GILBM RK2 upwind displacement precomputation (Imamura 2005 Eq. 19-20)
//預計算曲線坐標系中，非物理空間計算點的位置，分開編號分開分量計算 
// Precomputes delta_k[alpha][j*NZ6+k] = dt * e_tilde_k(k_half)
// where k_half = k - 0.5*dt*e_tilde_k(k) is the RK2 midpoint.
//RK2 minPoint is \xi vector - \Delta \xi(1)
// Called once during initialization; results copied to GPU.
// This saves ~50% kernel time vs computing on-the-fly (Imamura p.650).

void PrecomputeGILBM_DeltaK(
    double *delta_k_h,       // 輸出: [19 * NYD6 * NZ6]，預計算的位移量
    const double *dk_dz_h,   // 輸入: 度量項 dk/dz [NYD6*NZ6]
    const double *dk_dy_h,   // 輸入: 度量項 dk/dy [NYD6*NZ6]
    int NYD6_local,
    int NZ6_local
) {
    // D3Q19 離散速度集（與 initialization.h 中一致）
    double e[19][3] = {
        {0,0,0},
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };

    // 計算單層平面的大小（j-k 平面的節點總數）
    int sz = NYD6_local * NZ6_local;

    // 將整個輸出陣列初始化為零
    for (int n = 0; n < 19 * sz; n++) {
        delta_k_h[n] = 0.0;
    }

    // 遍歷所有離散速度方向（alpha=0 為靜止方向，跳過）
    for (int alpha = 1; alpha < 19; alpha++) {
        // 若該方向在 y 和 z 分量皆為零，則逆變速度 e_tilde_k 恆為零，直接跳過
        // e[alpha][1] = e_y, e[alpha][2] = e_z
        if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;

        for (int j = 3 ; j < NYD6_local -4 ; j++) { //修改：y方向應該跳過BufferLayer
            // k 方向僅遍歷內部節點（避開邊界各留 2 層//k方向buffer layer只有下兩層上兩層
            for (int k = 2; k < NZ6_local - 2; k++) {
                int idx_jk = j * NZ6_local + k;

                // 步驟一：計算當前格點 (j, k) 處的 k 方向逆變速度
                // e_tilde_k = e_y * (dk/dy) + e_z * (dk/dz)
                double e_tilde_k0 = e[alpha][1] * dk_dy_h[idx_jk]
                                  + e[alpha][2] * dk_dz_h[idx_jk];

                // 步驟二：RK2 半步位移與中間點位置
                // Δk(1) = 0.5 * dt * e_tilde_k(k)
                // k_half = k - Δk(1)  （Imamura 2005 Eq.19）
                double dk_half = 0.5 * dt * e_tilde_k0;
                double k_half = (double)k - dk_half;

                // 在中間點位置進行度量項的線性插值
                // 將插值索引限制在安全範圍 [2, NZ6_local-3] 內
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

                // 步驟三：計算中間點處（非物理空間計算點）的逆變速度
                // e_tilde_k(k_half) = e_y * (dk/dy)|_{k_half} + e_z * (dk/dz)|_{k_half}
                double e_tilde_k_half = e[alpha][1] * dk_dy_half
                                      + e[alpha][2] * dk_dz_half;

                // 步驟四：完整 RK2 位移量（精度為 O(dt^3)）
                // Δk = dt * e_tilde_k(k_half)  （Imamura 2005 Eq.20）
                delta_k_h[alpha * sz + idx_jk] = dt * e_tilde_k_half;
            }
        }
    }
}

#endif

提出異議：你的遷移前計算點應該分成\xi分量以及\zeta分量計算，這樣在做內插時，才可以配合函數計算，可以直接套用函數是，來計算垂直方向剋插全重陣列：