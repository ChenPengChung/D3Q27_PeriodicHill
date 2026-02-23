#ifndef GILBM_EVOLUTION_H
#define GILBM_EVOLUTION_H 
//=============================
//GILBM核心演算法流程
//步驟一: Interpolation Lagrange插值 + Streaming 取值的內插點為上衣時間步所更新的碰撞後分佈函數陣列
//步驟二: 以插值後的分佈函數輸出為當前計算點的f_new，以及 計算物理空間計算點的平衡分佈函數，宏觀參數
//-------更新專數於當前計算點的陣列
//步驟三: 更新物理空間計算點的重估一般態分佈函數陣列
//步驟四: 更新物理空間計算點的 碰撞後一般態分佈函數陣列
//=============================
 
// __constant__ device memory for D3Q19 velocity set and weights
__constant__ double GILBM_e[19][3] = {
    {0,0,0},
    {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
    {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
    {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
    {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
};

__constant__ double GILBM_W[19] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};


// Runtime parameters (set once via cudaMemcpyToSymbol)
__constant__ double GILBM_dt; //global time step 
//計算計算空間中，\eta , \xi分量的偏移量。
// Precomputed displacement arrays (constant for uniform x and y)
__constant__ double GILBM_delta_eta[19];
__constant__ double GILBM_delta_xi[19];

// Include sub-modules (after __constant__ declarations they depend on)
#include "gilbm/interpolation_gilbm.h"
#include "gilbm/boundary_conditions.h"
constexpr int STENCIL_SIZE = 7;
constexpr int STENCIL_VOL  = 343;  // 7*7*7//內插成員物理空間計算點總數
// Grid size for f_pc_d / feq_d indexing
#define GRID_SIZE (NX6 * NYD6 * NZ6)
// ============================================================================
// Helper:zeta方向內插成員起始點編號設計(預防越界)
// ============================================================================
__device__ __forceinline__ void compute_stencil_base(
    int i, int j, int k,
    int &bi, int &bj, int &bk
) {
    bi = i - 3;
    bj = j - 3;
    bk = k - 3;
    if (bi < 0)           bi = 0;
    if (bi + 6 >= NX6)    bi = NX6 - STENCIL_SIZE;
    if (bj < 0)           bj = 0;
    if (bj + 6 >= NYD6)   bj = NYD6 - STENCIL_SIZE;
    if (bk < 2)           bk = 2;
    if (bk + 6 > NZ6 - 3) bk = NZ6 - 9;
}
// ============================================================================
// Helper: 在此利用位址運算子進行pass by reference 傳址呼叫，此技巧為 回傳多值得方法
// ============================================================================
__device__ __forceinline__ void compute_macroscopic_at(
    double *f_ptrs[19], int idx,
    double &rho_out, double &ux_out, double &uy_out, double &uz_out
) {
    double f[19];
    for (int q = 0; q < 19; q++) f[q] = f_ptrs[q][idx];

    rho_out = f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]+f[8]+f[9]
             +f[10]+f[11]+f[12]+f[13]+f[14]+f[15]+f[16]+f[17]+f[18];
    ux_out = (f[1]+f[7]+f[9]+f[11]+f[13] - (f[2]+f[8]+f[10]+f[12]+f[14])) / rho_out;
    uy_out = (f[3]+f[7]+f[8]+f[15]+f[17] - (f[4]+f[9]+f[10]+f[16]+f[18])) / rho_out;
    uz_out = (f[5]+f[11]+f[12]+f[15]+f[16] - (f[6]+f[13]+f[14]+f[17]+f[18])) / rho_out;
}

__device__ void gilbm_compute_point(
    int i, int j, int k,//計算空間座標點
    double *f_new_ptrs[19],
    double *f_pc,
    double *feq_d,
    double *omega_dt_d,
    double *dk_dz_d, double *dk_dy_d, double *delta_zeta_d,
    double *dt_local_d, double *tau_local_d,
    double *u_out, double *v_out, double *w_out, double *rho_out_arr,
    double *Force, double *rho_modify
) {
    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    // Local dt and tau at point A
    const double dt_A    = dt_local_d[idx_jk];  // Δt_A (local time step)
    const double tau_A   = tau_local_d[idx_jk]; // ω_A (Imamura無因次鬆弛時間 ≡ τ/Δt, Eq.1)
    const double omegadt_A = omega_dt_d[index];  // ω_A × Δt_A = τ_A (教科書鬆弛時間)

    // LTS acceleration factor for eta/xi displacement scaling
    const double a_local = dt_A / GILBM_dt;//計算該點上的加速因子，此參數為loca的值，此值隨空間變化 

    // Stencil base with boundary clamping
    int bi, bj, bk;
    compute_stencil_base(i, j, k, bi, bj, bk);
    //計算當前計算點在整個343stencil節點組的相對位置
    const int ci = i - bi;//eta方向相對座標
    const int cj = j - bj;//xi方向相對座標
    const int ck = k - bk;
    //定義下邊界計算底計算空間座標以及上邊界
    bool is_bottom = (k == 2);
    bool is_top    = (k == NZ6 - 3);
    //對當前點寫入度量係數
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];
    //邊界處理:計算邊界上法向度梯度(三個分量)以及(邊界上的密度取值)
    double rho_wall = 0.0, du_dk = 0.0, dv_dk = 0.0, dw_dk = 0.0;
    if (is_bottom) {
        // k=2 為底壁，用 k=3, k=4 兩層做二階外推
        int idx3 = j * nface + i *NZ6 + 3 ;
        int idx4 = j * nface + i *NZ6 + 4;
        double rho3, u3, v3, w3, rho4, u4, v4, w4;
        compute_macroscopic_at(f_new_ptrs, idx3, rho3, u3, v3, w3);
        compute_macroscopic_at(f_new_ptrs, idx4, rho4, u4, v4, w4);
        du_dk = (4.0 * u3 - u4) / 2.0;  // ∂u/∂k|_wall //採用二階精度單邊差分計算法向速度梯度
        dv_dk = (4.0 * v3 - v4) / 2.0;  // ∂v/∂k|_wall //採用二階精度單邊差分計算法向速度梯度
        dw_dk = (4.0 * w3 - w4) / 2.0;  // ∂w/∂k|_wall //採用二階精度單邊差分計算法向速度梯度
        rho_wall = rho3;  // 零法向壓力梯度近似 (Imamura S3.2)
    } else if (is_top) {
        // k=NZ6-3 為頂壁，用 k=NZ6-4, k=NZ6-5 兩層 (反向差分)
        int idxm1 = j * nface + i *NZ6 + (NZ6 - 4);
        int idxm2 = j * nface + i *NZ6 + (NZ6 - 5);
        double rhom1, um1, vm1, wm1, rhom2, um2, vm2, wm2;
        compute_macroscopic_at(f_new_ptrs, idxm1, rhom1, um1, vm1, wm1);
        compute_macroscopic_at(f_new_ptrs, idxm2, rhom2, um2, vm2, wm2);
        du_dk = -(4.0 * um1 - um2) / 2.0;  // ∂u/∂k|_wall (頂壁法向反向)
        dv_dk = -(4.0 * vm1 - vm2) / 2.0;  // ∂v/∂k|_wall (頂壁法向反向)
        dw_dk = -(4.0 * wm1 - wm2) / 2.0;  // ∂w/∂k|_wall (頂壁法向反向)
        rho_wall = rhom1;
    }
    
    




