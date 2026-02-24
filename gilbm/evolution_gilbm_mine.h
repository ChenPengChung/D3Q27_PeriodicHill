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
    const int index = j * nface + k * NX6 + i;//當前空間座標位置 
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
    // ── Wall BC pre-computation ──────────────────────────────────────
    // Chapman-Enskog BC 需要物理空間速度梯度張量 ∂u_α/∂x_β。
    // 由 chain rule:
    //   ∂u_α/∂x_β = ∂u_α/∂η · ∂η/∂x_β + ∂u_α/∂ξ · ∂ξ/∂x_β + ∂u_α/∂ζ · ∂ζ/∂x_β
    // 一般情況需要 9 個計算座標梯度 (3 速度分量 × 3 計算座標方向)。
    //
    // 但在 no-slip 壁面 (等 k 面) 上，u=v=w=0 對所有 (η,ξ) 恆成立，因此：
    //   ∂u_α/∂η = 0,  ∂u_α/∂ξ = 0   (切向微分為零)
    //   ∂u_α/∂ζ ≠ 0                   (唯一非零：法向梯度)
    // 9 個量退化為 3 個：du/dk, dv/dk, dw/dk
    //
    // Chain rule 簡化為：∂u_α/∂x_β = (∂u_α/∂k) · (∂k/∂x_β)
    // 度量係數 ∂k/∂x_β 由 dk_dy, dk_dz 提供 (dk_dx 目前假設為 0)。
    // 二階單邊差分 (壁面 u=0): du/dk|_wall = (4·u_{k±1} - u_{k±2}) / 2
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
    //為了計算邊界上的六個值 = d(u,v,w)/dy , d(u,v,w)/dz，這六個值在壁面邊界條件的計算中會用到，這裡先行計算好以供後續使用
    //你需要計算六個偏微分在曲線坐標系中，d(u,v,w)/dk三個值 以及 兩個度量係數 dk_dy , dk_dz 一共五個資訊你需要知道，才可以處理"邊界問題"
    //===========================================================
    //         第一步驟(Lagrange Interpolation+Streaming)
    //============================================================
    double rho_stream = 0.0, mx_stream = 0.0, my_stream = 0.0, mz_stream = 0.0; 
    //stream = 這些值來自「遷移步驟完成後」的分佈函數，是碰撞步驟的輸入。
    //(ci,cj,ck):物理空間計算點的內插系統空間座標
    //f_pc:陣列元素物理命名意義:1.pc=post-collision 
    //2.f_pc[(q * 343 + flat) * GRID_SIZE + index]
    //        ↑編號(1~18) ↑stencil內位置      ↑物理空間計算點A   ->這就是post-collision 的命名意義
    //在迴圈之外，對於某一個空間點
    for (int q = 0; q < 19; q++) {
    //在迴圈內部，對於某一個空間點，對於某一個離散度方向
        double f_streamed;
        if(q == 0){
            int center_flat = ci * 49 + cj * 7 + ck; //當前計算點的內差系統位置轉換為一維座標 
            //不需要經過差過程，直接取上一輪碰撞後分佈函數post-collision的中心點分佈函數作為該計算點上的"插值後分佈函數"
            //在f_pc的命名意義有有三層資訊需納入考慮，1.index(當前座標網格點)2.alpha(編號)3.stencil(內插系統座標)，為了不要讓這三層資訊彼此混淆而誤用同一直，採如下命名方式:
            f_streamed = f_pc[(q * STENCIL_VOL + center_flat) * GRID_SIZE + index];//其中，index<=GRID_SIZE ; center_flat<=STENCIL_VOL
        }else{
             bool need_bc = false ; //初始化邊界條件判斷子為否
             //如果是"下邊界計算點" ，且編號對應的粒子速度zeta分量>0則把 false 改為 true 
            if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true); //>0 -> true 
            else if (is_top) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false); //<0 -> true 
            if (need_bc) { f_streamed = ChapmanEnskogBC(q, rho_wall,
                    du_dk, dv_dk, dw_dk,
                    dk_dy_val, dk_dz_val, 
                    tau_A, dt_A //權重係數//localtimestep 
                );
            } else {
                // Load 343 values from f_pc into local stencil
                double f_stencil[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
                for (int si = 0; si < 7; si++){
                    for (int sj = 0; sj < 7; sj++){
                        for (int sk = 0; sk < 7; sk++) {
                            int interpolation = si * 49 + sj * 7 + sk; //遍歷內插成員系統的每一個點 
                            f_stencil[si][sj][sk] = f_pc[(q * STENCIL_VOL + interpolation) * GRID_SIZE + index];//拿個桶子紀錄本計算點上相對應的內插成員系統的分佈函數
                        }
                    }
                }
                // Departure point  //a_local 為本地local accerleration factor
                //計算local timed stepo 版本的偏移量
                double delta_eta_loc    = a_local * GILBM_delta_eta[q];
                double delta_xi_loc   = a_local * GILBM_delta_xi[q];
                double delta_zeta_loc = delta_zeta_d[q * NYD6 * NZ6 + idx_jk];
                //(up_i,up_j,up_k)為非物理空間計算點的空間座標//因為整個streaming過程為流入計算點，所以要減去偏移量
                //對於某一格物理計算點，對於某一個離散速度編號Q而言，
                double up_i = (double)i - delta_eta_loc;
                double up_j = (double)j - delta_xi_loc;
                double up_k = (double)k - delta_zeta_loc;
                //i,j,k為計算區域內的物理聰間計算點(不含buffer layer)
                if (up_i < 1.0)               up_i = 1.0; //up_i >= 2 
                if (up_i > (double)(NX6 - 3)) up_i = (double)(NX6 - 3);//up_i <= NX6-3
                if (up_j < 1.0)               up_j = 1.0; //up_j >= 2
                if (up_j > (double)(NYD6 - 3))up_j = (double)(NYD6 - 3);//up_j <= NYD6-3
                if (up_k < 2.0)               up_k = 2.0; //up_k >=2 因為如果小於2對於該點該方向已邊界條件做處理 
                if (up_k > (double)(NZ6 - 3)) up_k = (double)(NZ6 - 3); //up_k <= NZ6-3 因為如果大於NZ6-3對於該點該方向已邊界條件做處理 
                //上面這段if語句的作用是確保up_i, up_j, up_k的值在物理空間計算區域內，避免越界訪問f_pc陣列
                // Lagrange weights relative to stencil base
                //(t_i,t_j,t_k)為 非物理空間計算點在內插成員系統中的座標位置
                //先轉換標為當前非物理空間計算點的座標為內插成員系統座標
                double t_i = up_i - (double)bi;
                double t_j = up_j - (double)bj;
                double t_k = up_k - (double)bk;
                //一維插值權重陣列
                double Lagrangarray_xi[7], Lagrangarray_eta[7], Lagrangarray_zeta[7];
                lagrange_7point_coeffs(t_i, Lagrangarray_xi);
                lagrange_7point_coeffs(t_j, Lagrangarray_eta);
                lagrange_7point_coeffs(t_k, Lagrangarray_zeta);
                //(si,sj,sk)為內插成員座標系統編號
                //Tensor-product interpolation
                //StepA: 一維eta方向Lagrange內插結果，配合一維內插權重陣列
                double interpolation1order[7][7];
                //StepB: 一維xi方向Lagrange內插結果，配合一維內插權重陣列
                double interpolation2order[7];
                for (int sj = 0; sj < 7; sj++)
                    for (int sk = 0; sk < 7; sk++)
                        interpolation1order[sj][sk] = Intrpl7(
                            f_stencil[0][sj][sk], Lagrangarray_xi[0],
                            f_stencil[1][sj][sk], Lagrangarray_xi[1],
                            f_stencil[2][sj][sk], Lagrangarray_xi[2],
                            f_stencil[3][sj][sk], Lagrangarray_xi[3],
                            f_stencil[4][sj][sk], Lagrangarray_xi[4],
                            f_stencil[5][sj][sk], Lagrangarray_xi[5],
                            f_stencil[6][sj][sk], Lagrangarray_xi[6]);
                double interpolation2order[7];
                for (int sk = 0; sk < 7; sk++)
                    interpolation2order[sk] = Intrpl7(
                        interpolation1order[0][sk], Lagrangarray_eta[0],
                        interpolation1order[1][sk], Lagrangarray_eta[1],
                        interpolation1order[2][sk], Lagrangarray_eta[2],
                        interpolation1order[3][sk], Lagrangarray_eta[3],
                        interpolation1order[4][sk], Lagrangarray_eta[4],
                        interpolation1order[5][sk], Lagrangarray_eta[5],
                        interpolation1order[6][sk], Lagrangarray_eta[6]);
                // Step C: zeta reduction -> scalar
                f_streamed = Intrpl7(
                    interpolation2order[0], Lagrangarray_zeta[0],
                    interpolation2order[1], Lagrangarray_zeta[1],
                    interpolation2order[2], Lagrangarray_zeta[2],
                    interpolation2order[3], Lagrangarray_zeta[3],
                    interpolation2order[4], Lagrangarray_zeta[4],
                    interpolation2order[5], Lagrangarray_zeta[5],
                    interpolation2order[6], Lagrangarray_zeta[6]);        
        }
    }
    // Write post-streaming to f_new (this IS streaming)
        f_new_ptrs[q][index] = f_streamed;

        // ── 宏觀量累加 (物理直角坐標) ────────────────────────────
        // ρ  = Σ_q f_q             (密度)
        // ρu = Σ_q e_{q,x} · f_q  (x-動量)
        // ρv = Σ_q e_{q,y} · f_q  (y-動量)
        // ρw = Σ_q e_{q,z} · f_q  (z-動量)
        //
        // GILBM_e[q] = 物理直角坐標系的離散速度 (e_x, e_y, e_z)，
        // 不是曲線坐標系的逆變速度分量。f_i 的速度空間定義不受座標映射影響。
        // 曲線坐標映射只影響 streaming 步驟 (位移 δη, δξ, δζ 含度量項)。
        // → Σ f_i·e_i 直接得到物理直角坐標的動量，不需要 Jacobian 映射。
        rho_stream += f_streamed; //累加不同編號計算宏觀參數
        mx_stream  += GILBM_e[q][0] * f_streamed;
        my_stream  += GILBM_e[q][1] * f_streamed;
        mz_stream  += GILBM_e[q][2] * f_streamed;
    }
    

     




