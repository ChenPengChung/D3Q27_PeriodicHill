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

// REMOVED: __constant__ GILBM_dt, GILBM_delta_eta[19], GILBM_delta_xi[19]
// Replaced by precomputed Lagrange weights: lagrange_eta_d, lagrange_xi_d, lagrange_zeta_d
// Kernel no longer computes a_local, departure points, or lagrange_7point_coeffs at runtime.

#if USE_MRT
// MRT transformation matrix M[19][19] and inverse M⁻¹[19][19]
// Values from MRT_Matrix.h (d'Humières 2002 D3Q19)
__constant__ double GILBM_M[19][19];
__constant__ double GILBM_Mi[19][19];
#endif

// Include sub-modules (after __constant__ declarations they depend on)
#include "interpolation_gilbm.h"
#include "boundary_conditions.h"

#define STENCIL_SIZE 7
#define STENCIL_VOL  343  // 7*7*7

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
    if (bk < 3)           bk = 3;                    // Buffer=3: 壁面在 k=3
    if (bk + 6 > NZ6 - 4) bk = NZ6 - 10;             // 確保 bk+6 ≤ NZ6-4 (頂壁)
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
//使用在 計算a粒子碰撞前插植後一般態分佈函數重估陣列 後面
#if USE_MRT
// ============================================================================
// MRT collision device function for GILBM with Local Time Stepping
//
// Standard MRT: f* = f̃ - M⁻¹ S (M·f̃ - M·feq)
//   where S = diag(s0..s18) is the relaxation rate matrix
//
// LTS localization: only viscosity-related moments use local tau
//   s9 = s11 = s13 = s14 = s15 = 1/tau_A  (LOCAL, from omega_A)
//   all other s_i = fixed global constants (same as MRT_Matrix.h)
//
// Body force: first-order Guo (consistent with original MRT_Process.h)
//   f_q += w_q * 3 * e_y[q] * Force * dt_A
// ============================================================================
__device__ void gilbm_mrt_collision(
    double f_re[19],          // in/out: re-estimated distribution → post-collision
    const double feq_B[19],   // input: equilibrium distribution at node B
    double s_visc,            // 1/omega_A = 1/tau_A (local viscosity relaxation rate)
    double dt_A,              // local time step at point A (for body force scaling)
    double Force0             // body force magnitude (y-direction streamwise)
) {
    // ---- Step 3a: Compute non-equilibrium moments ----
    // m_neq[i] = Σ_q M[i][q] × (f̃[q] - feq[q])
    double m_neq[19];
    for (int i = 0; i < 19; i++) {
        double sum = 0.0;
        for (int q = 0; q < 19; q++)
            sum += GILBM_M[i][q] * (f_re[q] - feq_B[q]);
        m_neq[i] = sum;
    }

    // ---- Step 3b: Apply per-moment relaxation rates ----
    // dm[i] = s_i × m_neq[i]
    // Conserved moments (s=0): dm[0]=dm[3]=dm[5]=dm[7]=0
    // Viscosity moments: s_visc = 1/tau_A (LOCAL)
    // Other moments: fixed global values (MRT_Matrix.h Relaxation)
    double dm[19];
    dm[0]  = 0.0;                    // s0  = 0.0 (conserved: density)
    dm[1]  = 1.19  * m_neq[1];      // s1  = 1.19 (energy)
    dm[2]  = 1.4   * m_neq[2];      // s2  = 1.4  (energy square)
    dm[3]  = 0.0;                    // s3  = 0.0 (conserved: momentum-x)
    dm[4]  = 1.2   * m_neq[4];      // s4  = 1.2  (energy flux)
    dm[5]  = 0.0;                    // s5  = 0.0 (conserved: momentum-y)
    dm[6]  = 1.2   * m_neq[6];      // s6  = 1.2  (energy flux)
    dm[7]  = 0.0;                    // s7  = 0.0 (conserved: momentum-z)
    dm[8]  = 1.2   * m_neq[8];      // s8  = 1.2  (energy flux)
    dm[9]  = s_visc * m_neq[9];     // s9  = 1/tau_A ★ LOCAL (stress p_xx-p_yy)
    dm[10] = 1.4   * m_neq[10];     // s10 = 1.4
    dm[11] = s_visc * m_neq[11];    // s11 = 1/tau_A ★ LOCAL (stress p_ww)
    dm[12] = 1.4   * m_neq[12];     // s12 = 1.4
    dm[13] = s_visc * m_neq[13];    // s13 = 1/tau_A ★ LOCAL (stress p_xy)
    dm[14] = s_visc * m_neq[14];    // s14 = 1/tau_A ★ LOCAL (stress p_yz)
    dm[15] = s_visc * m_neq[15];    // s15 = 1/tau_A ★ LOCAL (stress p_xz)
    dm[16] = 1.5   * m_neq[16];     // s16 = 1.5 (kinetic 3rd-order)
    dm[17] = 1.5   * m_neq[17];     // s17 = 1.5
    dm[18] = 1.5   * m_neq[18];     // s18 = 1.5

    // ---- Step 3c: Inverse transform + body force ----
    // f*[q] = f̃[q] - Σ_i Mi[q][i] × dm[i] + force_source[q]
    for (int q = 0; q < 19; q++) {
        double correction = 0.0;
        for (int i = 0; i < 19; i++)
            correction += GILBM_Mi[q][i] * dm[i];
        f_re[q] -= correction;
        // Body force: w_q × 3 × e_y[q] × Force × dt_A (y=streamwise)//adding the discrete force term for each alpha index F_i delta_t
        f_re[q] += GILBM_W[q] * 3.0 * GILBM_e[q][1] * Force0 * dt_A;
    }
}
#endif // USE_MRT

// ============================================================================
// Core GILBM 4-step logic (shared by Buffer and Full kernels)
// ============================================================================
__device__ void gilbm_compute_point(
    int i, int j, int k,//計算空間座標點
    double *f_new_ptrs[19],
    double *f_pc,
    double *feq_d,
    double *omegadt_local_d,
    double *dk_dz_d, double *dk_dy_d,
    double *dt_local_d, double *omega_local_d,
    double *lagrange_eta_d, double *lagrange_xi_d, double *lagrange_zeta_d,  // 預計算 Lagrange 權重
    int *bk_precomp_d,  // 預計算 stencil base k [NZ6], 直接用 k 索引
    double *u_out, double *v_out, double *w_out, double *rho_out_arr,
    double *Force, double *rho_modify
) {
    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    // Local dt and tau at point A
    const double dt_A    = dt_local_d[idx_jk];  // Δt_A (local time step)
    const double omega_A   = omega_local_d[idx_jk]; // ω_A (Imamura無因次鬆弛時間 ≡ τ/Δt, Eq.1)
    const double omegadt_A = omegadt_local_d[index];  // ω_A × Δt_A = τ_A (教科書鬆弛時間)

    // REMOVED: a_local = dt_A / GILBM_dt — no longer needed, Lagrange weights precomputed

    // Stencil base: bi/bj never clamped for executed points, bk precomputed with wall clamping
    const int bi = i - 3;  // i ∈ [3, NX6-4] → bi ∈ [0, NX6-7], no clamping needed
    const int bj = j - 3;  // j ∈ [3, NYD6-4] → bj ∈ [0, NYD6-7], no clamping needed
    const int bk = bk_precomp_d[k];  // precomputed: max(3, min(NZ6-10, k-3))

    // A's position within stencil
    const int ci = i - bi;  // = 3 (always, for executed i)
    const int cj = j - bj;  // = 3 (always, for executed j)
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
    bool is_bottom = (k == 3);       // Buffer=3: 底壁在 k=3
    bool is_top    = (k == NZ6 - 4); // Buffer=3: 頂壁在 k=NZ6-4
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

    double rho_wall = 0.0, du_dk = 0.0, dv_dk = 0.0, dw_dk = 0.0;
    if (is_bottom) {
        // k=3 為底壁，用 k=4, k=5 兩層做二階外推
        int idx3 = j * nface + 4 * NX6 + i;
        int idx4 = j * nface + 5 * NX6 + i;
        double rho3, u3, v3, w3, rho4, u4, v4, w4;
        compute_macroscopic_at(f_new_ptrs, idx3, rho3, u3, v3, w3);
        compute_macroscopic_at(f_new_ptrs, idx4, rho4, u4, v4, w4);
        du_dk = (u3) ;  // ∂u/∂k|_wall // 先用一階，待 CE BC 修正驗證後再升階
        dv_dk = (v3) ;  // ∂v/∂k|_wall //
        dw_dk = (w3) ;  // ∂w/∂k|_wall //
        /*du_dk = (4.0 * u3 - u4) / 2.0;  // ∂u/∂k|_wall //採用二階精度單邊差分計算法向速度梯度
        dv_dk = (4.0 * v3 - v4) / 2.0;  // ∂v/∂k|_wall //採用二階精度單邊差分計算法向速度梯度
        dw_dk = (4.0 * w3 - w4) / 2.0;  // ∂w/∂k|_wall //採用二階精度單邊差分計算法向速度梯度*/
        rho_wall = rho3;  // 零法向壓力梯度近似 (Imamura S3.2)
    } else if (is_top) {
        // k=NZ6-4 為頂壁，用 k=NZ6-5, k=NZ6-6 兩層 (反向差分)
        int idxm1 = j * nface + (NZ6 - 5) * NX6 + i;
        int idxm2 = j * nface + (NZ6 - 6) * NX6 + i;
        double rhom1, um1, vm1, wm1, rhom2, um2, vm2, wm2;
        compute_macroscopic_at(f_new_ptrs, idxm1, rhom1, um1, vm1, wm1);
        compute_macroscopic_at(f_new_ptrs, idxm2, rhom2, um2, vm2, wm2);
        du_dk = -(um1) ;  // ∂u/∂k|_wall // 先用一階
        dv_dk = -(vm1) ;  // ∂v/∂k|_wall //
        dw_dk = -(wm1) ;  // ∂w/∂k|_wall //
        /*du_dk = -(4.0 * um1 - um2) / 2.0;  // ∂u/∂k|_wall (頂壁法向反向)
        dv_dk = -(4.0 * vm1 - vm2) / 2.0;  // ∂v/∂k|_wall (頂壁法向反向)
        dw_dk = -(4.0 * wm1 - wm2) / 2.0;  // ∂w/∂k|_wall (頂壁法向反向)*/
        rho_wall = rhom1;
    }

    //stream = 這些值來自「遷移步驟完成後」的分佈函數，是碰撞步驟的輸入。
    //(ci,cj,ck):物理空間計算點的內插系統空間座標
    //f_pc:陣列元素物理命名意義:1.pc=post-collision 
    //2.f_pc[(q * 343 + flat) * GRID_SIZE + index]
    //        ↑編號(1~18) ↑stencil內位置      ↑物理空間計算點A   ->這就是post-collision 的命名意義                                                                             
    //在迴圈之外，對於某一個空間點
    double rho_stream = 0.0, mx_stream = 0.0, my_stream = 0.0, mz_stream = 0.0;

    for (int q = 0; q < 19; q++) {
    //在迴圈內部，對於某一個空間點，對於某一個離散度方向
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
                    omega_A, dt_A //權重係數//localtimestep
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
                // ── Read precomputed Lagrange weights (replaces runtime computation) ──
                // All weights depend only on (q, j, k), NOT on i.
                // Precomputed in PrecomputeGILBM_LagrangeWeights() on host.
                // Layout: w[q*7*NYD6*NZ6 + c*NYD6*NZ6 + idx_jk], q outermost, c middle
                const int q_base = q * 7 * NYD6 * NZ6;
                const int sz_jk = NYD6 * NZ6;
                double Lagrangarray_eta[7], Lagrangarray_xi[7], Lagrangarray_zeta[7];
                for (int c = 0; c < 7; c++) {
                    Lagrangarray_eta[c]  = lagrange_eta_d[q_base + c * sz_jk + idx_jk];
                    Lagrangarray_xi[c]   = lagrange_xi_d[q_base + c * sz_jk + idx_jk];
                    Lagrangarray_zeta[c] = lagrange_zeta_d[q_base + c * sz_jk + idx_jk];
                }

                // Tensor-product interpolation
                // Step A: η (i) reduction -> interpolation1order[7][7]
                double interpolation1order[7][7];
                for (int sj = 0; sj < 7; sj++)
                    for (int sk = 0; sk < 7; sk++)
                        interpolation1order[sj][sk] = Intrpl7(
                            f_stencil[0][sj][sk], Lagrangarray_eta[0],
                            f_stencil[1][sj][sk], Lagrangarray_eta[1],
                            f_stencil[2][sj][sk], Lagrangarray_eta[2],
                            f_stencil[3][sj][sk], Lagrangarray_eta[3],
                            f_stencil[4][sj][sk], Lagrangarray_eta[4],
                            f_stencil[5][sj][sk], Lagrangarray_eta[5],
                            f_stencil[6][sj][sk], Lagrangarray_eta[6]);

                // Step B: ξ (j) reduction -> interpolation2order[7]
                double interpolation2order[7];
                for (int sk = 0; sk < 7; sk++)
                    interpolation2order[sk] = Intrpl7(
                        interpolation1order[0][sk], Lagrangarray_xi[0],
                        interpolation1order[1][sk], Lagrangarray_xi[1],
                        interpolation1order[2][sk], Lagrangarray_xi[2],
                        interpolation1order[3][sk], Lagrangarray_xi[3],
                        interpolation1order[4][sk], Lagrangarray_xi[4],
                        interpolation1order[5][sk], Lagrangarray_xi[5],
                        interpolation1order[6][sk], Lagrangarray_xi[6]);

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
    //   δη = dt_global · e_x / dx           ← 度量項在此 (kernel 中由 a_local 縮放至 dt_local)
    //   δξ = dt_global · e_y / dy           ← 度量項在此 (kernel 中由 a_local 縮放至 dt_local)
    //   δζ = dt_local · (e_y·dk_dy + e_z·dk_dz)  ← 度量項在此 (已用 dt_local 預計算)
    //   → 位移量 = dt_local × 逆變速度 (e_i × ∂ξ/∂x)
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
    // STEPS 2+3: Re-estimation (Eq.35) + Collision
    // 計算 重估陣列 計算 碰撞後陣列 for one point 
    //   Eq.35: f̃_B = feq_B + (f_B - feq_B) × R_AB
    //          R_AB = (ω_A·Δt_A)/(ω_B·Δt_B) = omegadt_A / omegadt_B
    //   BGK Eq.3:  f*_B = f̃_B - (1/ω_A)(f̃_B - feq_B)
    //   MRT:       f*   = f̃ - M⁻¹ S (M·f̃ - M·feq_B)
    //   ω_A = omega_A (code variable), feq_B is per-stencil-node B.
    // ==================================================================

#if USE_MRT
    // ========== MRT collision: loop structure = for B { all 19 q } ==========
    // MRT requires all 19 f values at each stencil node B for moment transformation.
    // Re-estimation stays in distribution space (same R_AB as BGK).
    // Collision uses M⁻¹ S (m - meq) with local s_visc for viscosity moments.

    // Pre-check BC directions for all 19 q
    bool need_bc_arr[19];
    need_bc_arr[0] = false;  // q=0 (rest) is never BC
    for (int q = 1; q < 19; q++) {
        need_bc_arr[q] = false;
        if (is_bottom) need_bc_arr[q] = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
        else if (is_top) need_bc_arr[q] = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
    }

    double s_visc = 1.0 / omega_A ; //omega_A 作為 relaxation time 直接使用在碰撞矩陣中，作為分佈函數與平衡態分佈函數的前綴

    for (int si = 0; si < 7; si++) {
        int gi = bi + si; //計算內插成員座標點具體位置 
        for (int sj = 0; sj < 7; sj++) {
            int gj = bj + sj; //計算內插成員座標點具體位置 
            for (int sk = 0; sk < 7; sk++) {
                int gk = bk + sk;//計算內插成員座標點具體位置  
                //==========for each interpolation node position 
                //==========便歷每一個內插成員座標點位置 

                int idx_B = gj * nface + gk * NX6 + gi;
                int flat  = si * 49 + sj * 7 + sk;

                // ---- Gather all 19 f_B and feq_B at stencil node B ----
                double f_re_mrt[19], feq_B_arr[19];
                //在stencil 內部的每點，先寫入19個編號的分布佈函數與平衡態分佈函數 
                bool ghost_j = (gj < 3 || gj >= NYD6 - 3);
                     
                // Ghost zone: compute macroscopic once for all 19 feq
                double rho_B_g, u_B_g, v_B_g, w_B_g;
                //若為buffer layer,  則有一個 時序缺陷 | f_new 在 ghost zone 是舊值（MPI 還沒交換）→ feq 滯後一步 |
                if (ghost_j)
                    compute_macroscopic_at(f_new_ptrs, idx_B, rho_B_g, u_B_g, v_B_g, w_B_g);
                //如果是buffer layer 則重新計算，若為interrior ，則直接讀取 
                for (int q = 0; q < 19; q++) {
                    f_re_mrt[q] = f_new_ptrs[q][idx_B];
                    feq_B_arr[q] = ghost_j
                        ? compute_feq_alpha(q, rho_B_g, u_B_g, v_B_g, w_B_g)
                        : feq_d[q * GRID_SIZE + idx_B];
                }
                //===========此區為逐點操作，但是是所有編號同時一起操作===========
                // ---- Step 2: Re-estimation (distribution space, same as BGK) ----
                double omegadt_B = omegadt_local_d[idx_B];
                double R_AB = omegadt_A / omegadt_B;
                for (int q = 0; q < 19; q++)
                    f_re_mrt[q] = feq_B_arr[q] + (f_re_mrt[q] - feq_B_arr[q]) * R_AB;

                // ---- Step 3: MRT collision ----
                gilbm_mrt_collision(f_re_mrt, feq_B_arr, s_visc, dt_A, Force[0]);
                //===========此區為逐點操作，但是是所有編號同時一起操作===========
                // ---- Write back to A's PRIVATE f_pc (skip BC directions) ----
                //同一個內插成員點要寫回19筆資料 
                for (int q = 0; q < 19; q++) {
                    if (!need_bc_arr[q])
                        f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index] = f_re_mrt[q];
                }
            }
        }
    }

#else
    // ========== Original BGK/SRT collision (unchanged) ==========
    for (int q = 0; q < 19; q++) {
        // Skip BC directions: f_pc not needed, f_new already has BC value
        bool need_bc = false;
        if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
        else if (is_top) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
        if (need_bc) continue;

        // Body force source term for this q (y-direction pressure gradient)
        double force_source_q = GILBM_W[q] * 3.0 * (double)GILBM_e[q][1] * Force[0] * dt_A;

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
                        double rho_B, u_B, v_B, w_B;
                        compute_macroscopic_at(f_new_ptrs, idx_B,
                                               rho_B, u_B, v_B, w_B);
                        feq_B = compute_feq_alpha(q, rho_B, u_B, v_B, w_B);
                    } else {
                        feq_B = feq_d[q * GRID_SIZE + idx_B];
                    }

                    // Read omega_dt at B
                    double omegadt_B = omegadt_local_d[idx_B];
                    double R_AB = omegadt_A / omegadt_B;

                    // Eq.35: Re-estimation
                    double f_re = feq_B + (f_B - feq_B) * R_AB;

                    // Eq.3: BGK Collision
                    f_re -= (1.0 / omega_A) * (f_re - feq_B);

                    // Add body force source term
                    f_re += force_source_q;

                    // Write to A's PRIVATE f_pc
                    f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index] = f_re;
                }
            }
        }
    }
#endif // USE_MRT
}

// ============================================================================
// Step 2+3 only: Re-estimation + Collision (extracted for correction kernel)
// Re-runs after MPI exchange to fix stale ghost zone data at boundary rows.
// ============================================================================
__device__ void gilbm_step23_point(
    int index, int nface,
    int bi, int bj, int bk,
    double omega_A, double omegadt_A, double dt_A,
    double dk_dy_val, double dk_dz_val,
    bool is_bottom, bool is_top,
    double *f_new_ptrs[19],
    double *f_pc,
    double *feq_d,
    double *omegadt_local_d,
    double Force0
) {
#if USE_MRT
    // ========== MRT collision: for B { all 19 q } ==========
    bool need_bc_arr[19];
    need_bc_arr[0] = false;
    for (int q = 1; q < 19; q++) {
        need_bc_arr[q] = false;
        if (is_bottom) need_bc_arr[q] = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
        else if (is_top) need_bc_arr[q] = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
    }
    double s_visc = 1.0 / omega_A;

    for (int si = 0; si < 7; si++) {
        int gi = bi + si;
        for (int sj = 0; sj < 7; sj++) {
            int gj = bj + sj;
            for (int sk = 0; sk < 7; sk++) {
                int gk = bk + sk;
                int idx_B = gj * nface + gk * NX6 + gi;
                int flat  = si * 49 + sj * 7 + sk;

                double f_re_mrt[19], feq_B_arr[19];
                bool ghost_j = (gj < 3 || gj >= NYD6 - 3);

                double rho_B_g, u_B_g, v_B_g, w_B_g;
                if (ghost_j)
                    compute_macroscopic_at(f_new_ptrs, idx_B, rho_B_g, u_B_g, v_B_g, w_B_g);

                for (int q = 0; q < 19; q++) {
                    f_re_mrt[q] = f_new_ptrs[q][idx_B];
                    feq_B_arr[q] = ghost_j
                        ? compute_feq_alpha(q, rho_B_g, u_B_g, v_B_g, w_B_g)
                        : feq_d[q * GRID_SIZE + idx_B];
                }

                double omegadt_B = omegadt_local_d[idx_B];
                double R_AB = omegadt_A / omegadt_B;
                for (int q = 0; q < 19; q++)
                    f_re_mrt[q] = feq_B_arr[q] + (f_re_mrt[q] - feq_B_arr[q]) * R_AB;

                gilbm_mrt_collision(f_re_mrt, feq_B_arr, s_visc, dt_A, Force0);

                for (int q = 0; q < 19; q++) {
                    if (!need_bc_arr[q])
                        f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index] = f_re_mrt[q];
                }
            }
        }
    }
#else
    // ========== BGK/SRT collision ==========
    for (int q = 0; q < 19; q++) {
        bool need_bc = false;
        if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
        else if (is_top) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
        if (need_bc) continue;

        double force_source_q = GILBM_W[q] * 3.0 * (double)GILBM_e[q][1] * Force0 * dt_A;

        for (int si = 0; si < 7; si++) {
            int gi = bi + si;
            for (int sj = 0; sj < 7; sj++) {
                int gj = bj + sj;
                for (int sk = 0; sk < 7; sk++) {
                    int gk = bk + sk;
                    int idx_B = gj * nface + gk * NX6 + gi;
                    int flat  = si * 49 + sj * 7 + sk;

                    double f_B = f_new_ptrs[q][idx_B];

                    double feq_B;
                    if (gj < 3 || gj >= NYD6 - 3) {
                        double rho_B, u_B, v_B, w_B;
                        compute_macroscopic_at(f_new_ptrs, idx_B, rho_B, u_B, v_B, w_B);
                        feq_B = compute_feq_alpha(q, rho_B, u_B, v_B, w_B);
                    } else {
                        feq_B = feq_d[q * GRID_SIZE + idx_B];
                    }

                    double omegadt_B = omegadt_local_d[idx_B];
                    double R_AB = omegadt_A / omegadt_B;

                    double f_re = feq_B + (f_B - feq_B) * R_AB;
                    f_re -= (1.0 / omega_A) * (f_re - feq_B);
                    f_re += force_source_q;

                    f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index] = f_re;
                }
            }
        }
    }
#endif
}

// ============================================================================
// Correction kernel: re-run Step 2+3 for MPI boundary rows AFTER ghost exchange
// Fixes stale ghost zone f_new data used by the initial buffer kernel pass.
// Launch for start=3 (left band, 3 rows) and start=NYD6-6 (right band, 3 rows).
// ============================================================================
__global__ void GILBM_Correction_Kernel(
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new,
    double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new,
    double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new,
    double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *f_pc, double *feq_d, double *omegadt_local_d,
    double *dk_dz_d, double *dk_dy_d,
    double *dt_local_d, double *omega_local_d,
    int *bk_precomp_d,
    double *Force,
    int start
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + start;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 2 || i >= NX6 - 3 || k <= 2 || k >= NZ6 - 3) return;
    if (j < 3 || j >= NYD6 - 3) return;  // safety guard

    double *f_new_ptrs[19] = {
        f0_new, f1_new, f2_new, f3_new, f4_new, f5_new, f6_new,
        f7_new, f8_new, f9_new, f10_new, f11_new, f12_new,
        f13_new, f14_new, f15_new, f16_new, f17_new, f18_new
    };

    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    const double dt_A      = dt_local_d[idx_jk];
    const double omega_A   = omega_local_d[idx_jk];
    const double omegadt_A = omegadt_local_d[index];

    const int bi = i - 3;
    const int bj = j - 3;
    const int bk = bk_precomp_d[k];

    bool is_bottom = (k == 3);
    bool is_top    = (k == NZ6 - 4);
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

    gilbm_step23_point(index, nface, bi, bj, bk,
                       omega_A, omegadt_A, dt_A,
                       dk_dy_val, dk_dz_val,
                       is_bottom, is_top,
                       f_new_ptrs, f_pc, feq_d, omegadt_local_d,
                       Force[0]);
}

// ============================================================================
// Full-grid kernel (no start offset)
// ============================================================================
__global__ void GILBM_StreamCollide_Kernel(
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new,
    double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new,
    double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new,
    double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *f_pc, double *feq_d, double *omegadt_local_d,
    double *dk_dz_d, double *dk_dy_d,
    double *dt_local_d, double *omega_local_d,
    double *lagrange_eta_d, double *lagrange_xi_d, double *lagrange_zeta_d,
    int *bk_precomp_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *Force, double *rho_modify
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    // j-guard: Full kernel 只計算 j∈[7, NYD6-8]，邊界行由 Buffer kernel 負責
    // Buffer kernel 計算 j∈{3..6, 32..35}，避免兩個 stream 寫入重疊導致 race condition
    // Buffer=3: 計算範圍 k=3..NZ6-4
    if (i <= 2 || i >= NX6 - 3 || j <= 6 || j >= NYD6 - 7 || k <= 2 || k >= NZ6 - 3) return;

    double *f_new_ptrs[19] = {
        f0_new, f1_new, f2_new, f3_new, f4_new, f5_new, f6_new,
        f7_new, f8_new, f9_new, f10_new, f11_new, f12_new,
        f13_new, f14_new, f15_new, f16_new, f17_new, f18_new
    };

    gilbm_compute_point(i, j, k, f_new_ptrs,
        f_pc, feq_d, omegadt_local_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        lagrange_eta_d, lagrange_xi_d, lagrange_zeta_d,
        bk_precomp_d,
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
    double *f_pc, double *feq_d, double *omegadt_local_d,
    double *dk_dz_d, double *dk_dy_d,
    double *dt_local_d, double *omega_local_d,
    double *lagrange_eta_d, double *lagrange_xi_d, double *lagrange_zeta_d,
    int *bk_precomp_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *Force, double *rho_modify, int start
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + start;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Buffer=3: 計算範圍 k=3..NZ6-4
    if (i <= 2 || i >= NX6 - 3 || k <= 2 || k >= NZ6 - 3) return;

    double *f_new_ptrs[19] = {
        f0_new, f1_new, f2_new, f3_new, f4_new, f5_new, f6_new,
        f7_new, f8_new, f9_new, f10_new, f11_new, f12_new,
        f13_new, f14_new, f15_new, f16_new, f17_new, f18_new
    };

    gilbm_compute_point(i, j, k, f_new_ptrs,
        f_pc, feq_d, omegadt_local_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        lagrange_eta_d, lagrange_xi_d, lagrange_zeta_d,
        bk_precomp_d,
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
// Initialization kernel: compute omegadt_local_d from dt_local_d and omega_local_d
// ============================================================================
__global__ void Init_OmegaDt_Kernel(
    double *dt_local_d, double *omega_local_d, double *omegadt_local_d
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX6 || j >= NYD6 || k >= NZ6) return;

    const int index = j * NX6 * NZ6 + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    omegadt_local_d[index] = omega_local_d[idx_jk] * dt_local_d[idx_jk];
}

#endif
