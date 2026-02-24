#ifndef GILBM_EVOLUTION_H
#define GILBM_EVOLUTION_H

// ============================================================================
// GILBM Two-Pass Evolution Kernel (Imamura 2005)
//
// Single kernel, 4 steps per grid point A:
//   Step 1:   Read f_pc_d (private stencil), interpolate → f_new (post-streaming)
//   Step 1.5: Compute rho, u, feq from f_new → write feq_d, rho_out, u_out
//   Step 2:   Read f_new[B], feq_d[B] → Eq.35 re-estimation → f_buf (local)
//   Step 3:   Collision with tau_A on f_buf → write back to f_pc_d (private)
//
// Double-buffer: cudaMemcpy(f_new, f_old) BEFORE kernel launch.
//   Kernel only touches f_new. f_old not passed to kernel.
// ============================================================================

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
__constant__ double GILBM_dt;


// Precomputed displacement arrays (constant for uniform x and y)
__constant__ double GILBM_delta_eta[19];
__constant__ double GILBM_delta_xi[19];

// Include sub-modules (after __constant__ declarations they depend on)
#include "gilbm/interpolation_gilbm.h"
#include "gilbm/boundary_conditions.h"

constexpr int STENCIL_SIZE = 7;
constexpr int STENCIL_VOL  = 343;  // 7*7*7

// Grid size for f_pc_d / feq_d indexing
#define GRID_SIZE (NX6 * NYD6 * NZ6)



// ============================================================================
// Helper: compute stencil base with boundary clamping
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
// Helper: compute macroscopic from 19 f values at a given index
// ============================================================================
__device__ __forceinline__ void compute_macroscopic_at(
    double *f_ptrs[19], int idx,
    double &rho_out, double &u_out, double &v_out, double &w_out
) {
    double f[19];
    for (int q = 0; q < 19; q++) f[q] = f_ptrs[q][idx];

    rho_out = f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]+f[8]+f[9]
             +f[10]+f[11]+f[12]+f[13]+f[14]+f[15]+f[16]+f[17]+f[18];
    u_out = (f[1]+f[7]+f[9]+f[11]+f[13] - (f[2]+f[8]+f[10]+f[12]+f[14])) / rho_out;
    v_out = (f[3]+f[7]+f[8]+f[15]+f[17] - (f[4]+f[9]+f[10]+f[16]+f[18])) / rho_out;
    w_out = (f[5]+f[11]+f[12]+f[15]+f[16] - (f[6]+f[13]+f[14]+f[17]+f[18])) / rho_out;
}

// ============================================================================
// Core GILBM 4-step logic (shared by Buffer and Full kernels)
// ============================================================================
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

    // A's position within stencil
    const int ci = i - bi;//X
    const int cj = j - bj;
    const int ck = k - bk;

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
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

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

    // ==================================================================
    // STEP 1: Interpolation + Streaming (all q)
    // ==================================================================
    double rho_stream = 0.0, mx_stream = 0.0, my_stream = 0.0, mz_stream = 0.0;

    for (int q = 0; q < 19; q++) {
        double f_streamed;

        if (q == 0) {
            // Rest direction: read center value from f_pc (no interpolation)
            int center_flat = ci * 49 + cj * 7 + ck; //當前計算點的內差系統位置轉換為一維座標 
            f_streamed = f_pc[(q * STENCIL_VOL + center_flat) * GRID_SIZE + index];
        } else {
            bool need_bc = false;
            if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
            else if (is_top) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
            if (need_bc) {
                f_streamed = ChapmanEnskogBC(q, rho_wall,
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
                double delta_eta_loc    = a_local * GILBM_delta_eta[q];
                double delta_xi_loc   = a_local * GILBM_delta_xi[q];
                double delta_zeta_loc = delta_zeta_d[q * NYD6 * NZ6 + idx_jk];

                double up_i = (double)i - delta_eta_loc;
                double up_j = (double)j - delta_xi_loc;
                double up_k = (double)k - delta_zeta_loc;
                
                if (up_i < 1.0)               up_i = 1.0;
                if (up_i > (double)(NX6 - 3)) up_i = (double)(NX6 - 3);
                if (up_j < 1.0)               up_j = 1.0;
                if (up_j > (double)(NYD6 - 3))up_j = (double)(NYD6 - 3);
                if (up_k < 2.0)               up_k = 2.0;
                if (up_k > (double)(NZ6 - 3)) up_k = (double)(NZ6 - 3);

                // Lagrange weights relative to stencil base
                double t_i = up_i - (double)bi;
                double t_j = up_j - (double)bj;
                double t_k = up_k - (double)bk;

                double Lagrangarray_xi[7], Lagrangarray_eta[7], Lagrangarray_zeta[7];
                lagrange_7point_coeffs(t_i, Lagrangarray_xi);
                lagrange_7point_coeffs(t_j, Lagrangarray_eta);
                lagrange_7point_coeffs(t_k, Lagrangarray_zeta);

                // Tensor-product interpolation
                // Step A: xi reduction -> interpolation1order[7][7]
                double interpolation1order[7][7];
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

                // Step B: eta reduction -> interpolation2order[7]
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
        rho_stream += f_streamed;
        mx_stream  += GILBM_e[q][0] * f_streamed;
        my_stream  += GILBM_e[q][1] * f_streamed;
        mz_stream  += GILBM_e[q][2] * f_streamed;
    }

    // ==================================================================
    // STEP 1.5: Macroscopic + feq -> persistent arrays
    // ==================================================================
    // Mass correction
    rho_stream += rho_modify[0];
    f_new_ptrs[0][index] += rho_modify[0];
    // ── Audit 結論：此處不需要映射回直角坐標系 ─────────────────
    // (u_A, v_A, w_A) 已是物理直角坐標系的速度分量，可直接代入 feq。
    //
    // 理由：GILBM 中 f_i 的離散速度 e_i 始終是物理直角坐標向量：
    //   GILBM_e[q] = {0,±1} (D3Q19 標準整數向量)
    //   → mx_stream = Σ e_{q,x}·f_q = 物理 x-動量 (非曲線坐標分量)
    //
    // 曲線坐標映射只進入 streaming 位移 (precompute.h):
    //   δη = dt · e_x / dx           ← 度量項在此
    //   δξ = dt · e_y / dy           ← 度量項在此
    //   δζ = dt · (e_y·dk_dy + e_z·dk_dz)  ← 度量項在此
    //   → 位移量 = dt × 逆變速度 (e_i × ∂ξ/∂x)
    //   → e_i 本身不被座標映射修改
    //
    // 驗證：
    //   (1) initialization.h 用相同公式初始化 feq，無映射
    //   (2) fileIO.h 將 u,v,w 直接輸出為 VTK 物理速度，無映射
    //   (3) Imamura 2005 Eq. 2: c_i = c·e_i (物理速度)
    //       Eq. 13: c̃ = c·e·∂ξ/∂x (逆變速度僅用於位移)
    //       碰撞算子始終在物理速度空間執行
    double rho_A = rho_stream;
    double u_A   = mx_stream / rho_A;
    double v_A   = my_stream / rho_A;
    double w_A   = mz_stream / rho_A;
    // 計算平衡態分佈函數 (物理直角坐標，標準 D3Q19 BGK 公式)
    // feq_α = w_α · ρ · (1 + 3·(e_α·u) + 4.5·(e_α·u)² − 1.5·|u|²)
    // 此處 (ρ_A, u_A, v_A, w_A) 皆為物理量，feq 公式無需曲線坐標修正
    // Write feq to persistent global array
    for (int q = 0; q < 19; q++) {
        feq_d[q * GRID_SIZE + index] = compute_feq_alpha(q, rho_A, u_A, v_A, w_A);
    }
    //計算宏觀參數
    // Write macroscopic output
    rho_out_arr[index] = rho_A;
    u_out[index] = u_A;
    v_out[index] = v_A;
    w_out[index] = w_A;

    // ==================================================================
    // STEPS 2+3: Re-estimation (Eq.35) + Collision (Eq.3) per q
    //   Eq.35: f̃_B = feq_B + (f_B - feq_B) × R_AB
    //          R_AB = (ω_A·Δt_A)/(ω_B·Δt_B) = omegadt_A / omegadt_B
    //   Eq.3:  f*_B = f̃_B - (1/ω_A)(f̃_B - feq_B)
    //   ω_A = tau_A (code variable), feq_B is per-stencil-node B.
    // ==================================================================
    for (int q = 0; q < 19; q++) {
        // Skip BC directions: f_pc not needed, f_new already has BC value
        bool need_bc = false;
        if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
        //我要讓他>0為true<0為false
        else if (is_top) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
        if (need_bc) continue;

        for (int si = 0; si < 7; si++) {
            int gi = bi + si;
            for (int sj = 0; sj < 7; sj++) {
                int gj = bj + sj;
                for (int sk = 0; sk < 7; sk++) {
                    int gk = bk + sk;
                    int idx_B = gj * nface + gk * NX6 + gi;
                    int flat  = si * 49 + sj * 7 + sk;

                    // Read f_B from f_new (Gauss-Seidel)
                    double f_B = f_new_ptrs[q][idx_B];

                    // Read feq_B with ghost zone fallback
                    double feq_B;
                    if (gj < 3 || gj >= NYD6 - 3) {
                        // Ghost zone: compute on-the-fly
                        double rho_B, u_B, v_B, w_B;
                        compute_macroscopic_at(f_new_ptrs, idx_B,
                                               rho_B, u_B, v_B, w_B);
                        feq_B = compute_feq_alpha(q, rho_B, u_B, v_B, w_B);
                    } else {
                        feq_B = feq_d[q * GRID_SIZE + idx_B];
                    }

                    // Read omega_dt at B
                    double omegadt_B = omega_dt_d[idx_B];
                    // Eq.35: R_AB = (ω_A·Δt_A)/(ω_B·Δt_B) = omegadt_A / omegadt_B
                    double R_AB = omegadt_A / omegadt_B;

                    // Eq.35: Re-estimation
                    double f_re = feq_B + (f_B - feq_B) * R_AB;

                    // Eq.3: Collision with ω_A → f* = f - (1/ω_A)(f - feq_B)
                    f_re -= (1.0 / tau_A) * (f_re - feq_B);

                    // Write to A's PRIVATE f_pc
                    f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index] = f_re;
                }
            }
        }
    }
}

// ============================================================================
// Full-grid kernel (no start offset)
// ============================================================================
__global__ void GILBM_StreamCollide_Kernel(
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new,
    double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new,
    double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new,
    double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *f_pc, double *feq_d, double *omega_dt_d,
    double *dk_dz_d, double *dk_dy_d, double *delta_zeta_d,
    double *dt_local_d, double *tau_local_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *Force, double *rho_modify
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 2 || i >= NX6 - 3 || k <= 1 || k >= NZ6 - 2) return;

    double *f_new_ptrs[19] = {
        f0_new, f1_new, f2_new, f3_new, f4_new, f5_new, f6_new,
        f7_new, f8_new, f9_new, f10_new, f11_new, f12_new,
        f13_new, f14_new, f15_new, f16_new, f17_new, f18_new
    };

    gilbm_compute_point(i, j, k, f_new_ptrs,
        f_pc, feq_d, omega_dt_d,
        dk_dz_d, dk_dy_d, delta_zeta_d,
        dt_local_d, tau_local_d,
        u_out, v_out, w_out, rho_out,
        Force, rho_modify);
}

// ============================================================================
// Buffer kernel (processes buffer j-rows with start offset)
// ============================================================================
__global__ void GILBM_StreamCollide_Buffer_Kernel(
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new,
    double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new,
    double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new,
    double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *f_pc, double *feq_d, double *omega_dt_d,
    double *dk_dz_d, double *dk_dy_d, double *delta_zeta_d,
    double *dt_local_d, double *tau_local_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *Force, double *rho_modify, int start
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + start;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 2 || i >= NX6 - 3 || k <= 1 || k >= NZ6 - 2) return;

    double *f_new_ptrs[19] = {
        f0_new, f1_new, f2_new, f3_new, f4_new, f5_new, f6_new,
        f7_new, f8_new, f9_new, f10_new, f11_new, f12_new,
        f13_new, f14_new, f15_new, f16_new, f17_new, f18_new
    };

    gilbm_compute_point(i, j, k, f_new_ptrs,
        f_pc, feq_d, omega_dt_d,
        dk_dz_d, dk_dy_d, delta_zeta_d,
        dt_local_d, tau_local_d,
        u_out, v_out, w_out, rho_out,
        Force, rho_modify);
}

// ============================================================================
// Initialization kernel: fill f_pc_d from initial f arrays
// ============================================================================
__global__ void Init_FPC_Kernel(
    double *f0, double *f1, double *f2, double *f3, double *f4,
    double *f5, double *f6, double *f7, double *f8, double *f9,
    double *f10, double *f11, double *f12, double *f13, double *f14,
    double *f15, double *f16, double *f17, double *f18,
    double *f_pc
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX6 || j >= NYD6 || k >= NZ6) return;

    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;

    double *f_ptrs[19] = {
        f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
        f10, f11, f12, f13, f14, f15, f16, f17, f18
    };

    // Compute stencil base with clamping
    int bi, bj, bk;
    compute_stencil_base(i, j, k, bi, bj, bk);

    // Fill f_pc for all q and all stencil positions
    for (int q = 0; q < 19; q++) {
        for (int si = 0; si < 7; si++) {
            int gi = bi + si;
            for (int sj = 0; sj < 7; sj++) {
                int gj = bj + sj;
                for (int sk = 0; sk < 7; sk++) {
                    int gk = bk + sk;
                    int idx_B = gj * nface + gk * NX6 + gi;
                    int flat  = si * 49 + sj * 7 + sk;
                    f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index] = f_ptrs[q][idx_B];
                }
            }
        }
    }
}

// ============================================================================
// Initialization kernel: compute feq_d from initial f arrays
// ============================================================================
__global__ void Init_Feq_Kernel(
    double *f0, double *f1, double *f2, double *f3, double *f4,
    double *f5, double *f6, double *f7, double *f8, double *f9,
    double *f10, double *f11, double *f12, double *f13, double *f14,
    double *f15, double *f16, double *f17, double *f18,
    double *feq_d
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX6 || j >= NYD6 || k >= NZ6) return;

    const int index = j * NX6 * NZ6 + k * NX6 + i;

    double f[19];
    f[0]=f0[index]; f[1]=f1[index]; f[2]=f2[index]; f[3]=f3[index]; f[4]=f4[index];
    f[5]=f5[index]; f[6]=f6[index]; f[7]=f7[index]; f[8]=f8[index]; f[9]=f9[index];
    f[10]=f10[index]; f[11]=f11[index]; f[12]=f12[index]; f[13]=f13[index]; f[14]=f14[index];
    f[15]=f15[index]; f[16]=f16[index]; f[17]=f17[index]; f[18]=f18[index];

    double rho = f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]+f[8]+f[9]
                +f[10]+f[11]+f[12]+f[13]+f[14]+f[15]+f[16]+f[17]+f[18];
    double u = (f[1]+f[7]+f[9]+f[11]+f[13] - (f[2]+f[8]+f[10]+f[12]+f[14])) / rho;
    double v = (f[3]+f[7]+f[8]+f[15]+f[17] - (f[4]+f[9]+f[10]+f[16]+f[18])) / rho;
    double w = (f[5]+f[11]+f[12]+f[15]+f[16] - (f[6]+f[13]+f[14]+f[17]+f[18])) / rho;

    for (int q = 0; q < 19; q++) {
        feq_d[q * GRID_SIZE + index] = compute_feq_alpha(q, rho, u, v, w);
    }
}

// ============================================================================
// Initialization kernel: compute omega_dt_d from dt_local and tau_local
// ============================================================================
__global__ void Init_OmegaDt_Kernel(
    double *dt_local_d, double *tau_local_d, double *omega_dt_d
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX6 || j >= NYD6 || k >= NZ6) return;

    const int index = j * NX6 * NZ6 + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    // Imamura Eq.1: ω ≡ τ/Δt (無因次鬆弛時間, 碰撞項分母)
    // tau_local = ω, dt_local = Δt
    // omega_dt = ω × Δt = τ (教科書鬆弛時間)
    // Eq.35: R_AB = omegadt_A / omegadt_B = (ω_A·Δt_A)/(ω_B·Δt_B)
    omega_dt_d[index] = tau_local_d[idx_jk] * dt_local_d[idx_jk];
}

#endif
