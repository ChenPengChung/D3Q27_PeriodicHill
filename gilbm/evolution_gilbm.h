#ifndef GILBM_EVOLUTION_H
#define GILBM_EVOLUTION_H

//=============================
//GILBM核心演算法流程
//步驟一: Interpolation Lagrange插值 + Streaming 取值的內插點為上衣時間步所更新的碰撞後分佈函數陣列
//步驟二: 以插值後的分佈函數輸出為當前計算點的f_new，以及 計算物理空間計算點的平衡分佈函數，宏觀參數
////更新專數於當前計算點的陣列
////步驟三: 更新物理空間計算點的重估一般態分佈函數陣列
//步驟四: 更新物理空間計算點的 碰撞後一般態分佈函數 單點
//=============================


// __constant__ device memory for D3Q27 velocity set and weights
// Indices 0-18: same ordering as old D3Q19; indices 19-26: corner velocities
__constant__ double GILBM_e[NQ][3] = {
    { 0, 0, 0},                                    // 0: rest
    { 1, 0, 0}, {-1, 0, 0},                        // 1-2: ±x
    { 0, 1, 0}, { 0,-1, 0},                        // 3-4: ±y
    { 0, 0, 1}, { 0, 0,-1},                        // 5-6: ±z
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0}, // 7-10: xy edges
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1}, // 11-14: xz edges
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1}, // 15-18: yz edges
    { 1, 1, 1}, {-1, 1, 1}, { 1,-1, 1}, {-1,-1, 1}, // 19-22: corners +z
    { 1, 1,-1}, {-1, 1,-1}, { 1,-1,-1}, {-1,-1,-1}  // 23-26: corners -z
};

__constant__ double GILBM_W[NQ] = {
    8.0/27.0,                                       // rest
    2.0/27.0, 2.0/27.0, 2.0/27.0,                  // face ±x, ±y, ±z
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,        // edge
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,    // corner
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

__constant__ double GILBM_dt;             // global time step
__constant__ double GILBM_delta_eta[NQ];  // η displacements (precomputed with dt_global)
__constant__ double GILBM_delta_xi[NQ];   // ξ displacements (precomputed with dt_global)

#if USE_MRT
// D3Q27 MRT transformation matrix M[27][27] and inverse M⁻¹[27][27]
// Computed at initialization via Gram-Schmidt (MRT_Matrix_D3Q27.h)
__constant__ double GILBM_M[NQ][NQ];
__constant__ double GILBM_Mi[NQ][NQ];
// Diagonal relaxation rates S[27]
__constant__ double GILBM_S[NQ];
#endif

// Include sub-modules (after __constant__ declarations they depend on)
#include "interpolation_gilbm.h"
#include "boundary_conditions.h"

#if USE_CUMULANT
// D3Q27 Cumulant collision (AO or WP mode, selected by USE_WP_CUMULANT)
#include "../Cumulants/cumulant_collision.h"
#endif

#define STENCIL_SIZE 7
#define STENCIL_VOL  343  // 7*7*7




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
// Helper: compute macroscopic from NQ=27 f values at a given index
// ============================================================================
__device__ __forceinline__ void compute_macroscopic_at(
    double *f_ptrs[NQ], int idx,
    double &rho_out, double &u_out, double &v_out, double &w_out
) {
    double rho = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
    for (int q = 0; q < NQ; q++) {
        double fq = f_ptrs[q][idx];
        rho += fq;
        mx  += GILBM_e[q][0] * fq;
        my  += GILBM_e[q][1] * fq;
        mz  += GILBM_e[q][2] * fq;
    }
    rho_out = rho;
    u_out = mx / rho;
    v_out = my / rho;
    w_out = mz / rho;
}
//使用在 計算a粒子碰撞前插植後一般態分佈函數重估陣列 後面
#if USE_MRT
// ============================================================================
// D3Q27 MRT collision with Guo forcing (Suga et al. 2015)
//
// MRT-LBM equation:
//   f* = f̃ - M⁻¹ Ŝ (M·f̃ - M·feq) + M⁻¹(I - Ŝ/2)·M·|F>·δt
//
// Ŝ = diag(s0..s26) diagonal relaxation matrix (from GPU constant GILBM_S)
//   s0-3 = 0 (conserved: ρ, jx, jy, jz)
//   s4-9 = 1/τ (viscosity-related stress moments)
//   s10+ = fixed ghost relaxation rates
//
// Guo forcing (2nd-order accurate):
//   F_α = w_α × ρ × [ξ_α·a/cs² × (1 + ξ_α·u/cs²) - a·u/cs²]
//   where a = (F_x/ρ, 0, 0) body acceleration (y=streamwise)
//
// Half-force velocity correction applied BEFORE calling this function:
//   u_corrected = (Σ ξ_α f_α + 0.5×F×dt) / ρ
// ============================================================================
__device__ void gilbm_mrt_collision(
    double f_re[NQ],          // in/out: distribution → post-collision
    const double feq_B[NQ],   // input: equilibrium distribution
    double rho,               // density at this node
    double ux, double uy, double uz,  // velocity (with half-force correction)
    double dt_A,              // time step (for force scaling)
    double Force0             // body force magnitude (y-direction streamwise)
) {
    // ---- Step 1: Compute Guo forcing in velocity space ----
    // a = (0, Force0/rho, 0) — acceleration in y (streamwise) direction
    // F_α = w_α × ρ × [ (ξ_α·a)/cs² × (1 + (ξ_α·u)/cs²) - (a·u)/cs² ]
    // With cs²=1/3: 1/cs²=3, 1/cs⁴=9
    double ax = 0.0, ay = Force0 / rho, az = 0.0;
    double a_dot_u = ax*ux + ay*uy + az*uz;  // = ay*uy

    double F_vel[NQ];
    for (int q = 0; q < NQ; q++) {
        double e_dot_a = GILBM_e[q][0]*ax + GILBM_e[q][1]*ay + GILBM_e[q][2]*az;
        double e_dot_u = GILBM_e[q][0]*ux + GILBM_e[q][1]*uy + GILBM_e[q][2]*uz;
        F_vel[q] = GILBM_W[q] * rho * (3.0 * e_dot_a * (1.0 + 3.0 * e_dot_u) - 3.0 * a_dot_u);
    }

    // ---- Step 2: Transform f and feq to moment space ----
    // m[n] = Σ_α M[n][α] × f_α
    // meq[n] = Σ_α M[n][α] × feq_α
    // MF[n] = Σ_α M[n][α] × F_α
    double m[NQ], meq[NQ], MF[NQ];
    for (int n = 0; n < NQ; n++) {
        double s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (int a = 0; a < NQ; a++) {
            double Mna = GILBM_M[n][a];
            s1 += Mna * f_re[a];
            s2 += Mna * feq_B[a];
            s3 += Mna * F_vel[a];
        }
        m[n]   = s1;
        meq[n] = s2;
        MF[n]  = s3;
    }

    // ---- Step 3: Collision in moment space ----
    // m_post[n] = m[n] - S[n] × (m[n] - meq[n]) + (1 - S[n]/2) × MF[n] × dt
    double m_post[NQ];
    for (int n = 0; n < NQ; n++) {
        double sn = GILBM_S[n];
        m_post[n] = m[n] - sn * (m[n] - meq[n])
                  + (1.0 - 0.5 * sn) * MF[n] * dt_A;
    }

    // ---- Step 4: Inverse transform back to velocity space ----
    // f*[α] = Σ_n Mi[α][n] × m_post[n]
    for (int q = 0; q < NQ; q++) {
        double sum = 0.0;
        for (int n = 0; n < NQ; n++)
            sum += GILBM_Mi[q][n] * m_post[n];
        f_re[q] = sum;
    }
}
#endif // USE_MRT

// ============================================================================
// [GTS] Core GILBM with Global Time Stepping (no f_pc, no Re-estimation)
// ============================================================================
// Algorithm:
//   Step 1: 7-point Lagrange interpolation from f_old (previous time step)
//           + Chapman-Enskog BC for wall boundary directions
//   Step 1.5: Compute macroscopic (rho,u,v,w) + feq from streamed f
//   Step 2: Point-wise collision at A only (BGK or MRT) using omega_global
//           No 343-loop, no Re-estimation (R_AB=1 when omega/dt uniform)
//
// Memory: reads f_old_ptrs[NQ], writes f_new_ptrs[NQ] (double-buffer, no race)
//         Eliminates f_pc (NQ×343×GRID_SIZE ≈ 5.5 GB)
// ============================================================================
__device__ void gilbm_compute_point_gts(
    int i, int j, int k,
    double *f_old_ptrs[NQ],   // previous time step (read-only)
    double *f_new_ptrs[NQ],   // new time step (write)
    double *feq_d,
    double *dk_dz_d, double *dk_dy_d,
    double *delta_zeta_d,
    int *bk_precomp_d,
    double *u_out, double *v_out, double *w_out, double *rho_out_arr,
    double *Force, double *rho_modify,
    double omega_global
) {
    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    // Stencil base: bi/bj never clamped for executed points, bk precomputed
    const int bi = i - 3;
    const int bj = j - 3;
    const int bk = bk_precomp_d[k];
    const int ci = i - bi;  // = 3 (always, for executed i ∈ [3, NX6-4])
    const int cj = j - bj;  // = 3 (always, for executed j ∈ [3, NYD6-4])

    // ── Wall BC pre-computation ──
    bool is_bottom = (k == 3);
    bool is_top    = (k == NZ6 - 4);
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

    double rho_wall = 0.0, du_dk = 0.0, dv_dk = 0.0, dw_dk = 0.0;
    if (is_bottom) {
        // 2nd-order one-sided FD at wall k=2 (no-slip u[wall]=0):
        //   du/dk|wall = (4*u[k=3] - u[k=4]) / 2   (Imamura Eq. A.9)
        int idx_k3 = j * nface + 3 * NX6 + i;   // k=3 (first interior)
        int idx_k4 = j * nface + 4 * NX6 + i;   // k=4 (second interior)
        double rho3, u3, v3, w3;
        double rho4, u4, v4, w4;
        compute_macroscopic_at(f_old_ptrs, idx_k3, rho3, u3, v3, w3);
        compute_macroscopic_at(f_old_ptrs, idx_k4, rho4, u4, v4, w4);
        du_dk = (4.0 * u3 - u4) * 0.5;
        dv_dk = (4.0 * v3 - v4) * 0.5;
        dw_dk = (4.0 * w3 - w4) * 0.5;
        rho_wall = rho3;  // zero normal pressure gradient (Imamura S3.2)
    } else if (is_top) {
        // 2nd-order one-sided FD at wall k=NZ6-3 (no-slip u[wall]=0):
        //   du/dk|wall = -(4*u[k=NZ6-4] - u[k=NZ6-5]) / 2
        int idx_km1 = j * nface + (NZ6 - 4) * NX6 + i;  // k=NZ6-4 (first interior from top)
        int idx_km2 = j * nface + (NZ6 - 5) * NX6 + i;  // k=NZ6-5 (second interior from top)
        double rhom1, um1, vm1, wm1;
        double rhom2, um2, vm2, wm2;
        compute_macroscopic_at(f_old_ptrs, idx_km1, rhom1, um1, vm1, wm1);
        compute_macroscopic_at(f_old_ptrs, idx_km2, rhom2, um2, vm2, wm2);
        du_dk = -(4.0 * um1 - um2) * 0.5;
        dv_dk = -(4.0 * vm1 - vm2) * 0.5;
        dw_dk = -(4.0 * wm1 - wm2) * 0.5;
        rho_wall = rhom1;  // zero normal pressure gradient (Imamura S3.2)
    }

    // ── STEP 1: Interpolation + Streaming (from f_old, no a_local scaling) ──
    double f_streamed_all[NQ];
    double rho_stream = 0.0, mx_stream = 0.0, my_stream = 0.0, mz_stream = 0.0;

    for (int q = 0; q < NQ; q++) {
        double f_streamed;

        if (q == 0) {
            // Rest direction: no streaming, read center from f_old
            f_streamed = f_old_ptrs[0][index];
        } else {
            bool need_bc = false;
            if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
            else if (is_top) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
            if (need_bc) {
                // Chapman-Enskog BC with global omega and dt
                f_streamed = ChapmanEnskogBC(q, rho_wall,
                    du_dk, dv_dk, dw_dk,
                    dk_dy_val, dk_dz_val,
                    omega_global, GILBM_dt);
            } else {
                // GTS: no a_local scaling (a_local ≡ 1)
                double t_eta = (double)ci - GILBM_delta_eta[q];
                if (t_eta < 0.0) t_eta = 0.0; if (t_eta > 6.0) t_eta = 6.0;
                double t_xi  = (double)cj - GILBM_delta_xi[q];
                if (t_xi  < 0.0) t_xi  = 0.0; if (t_xi  > 6.0) t_xi  = 6.0;
                double delta_zeta = delta_zeta_d[q * NYD6 * NZ6 + idx_jk];
                double up_k = (double)k - delta_zeta;
                if (up_k < 3.0)                up_k = 3.0;
                if (up_k > (double)(NZ6 - 4))  up_k = (double)(NZ6 - 4);
                double t_zeta = up_k - (double)bk;
                double Lagrangarray_eta[7], Lagrangarray_xi[7], Lagrangarray_zeta[7];
                lagrange_7point_coeffs(t_eta,  Lagrangarray_eta);
                lagrange_7point_coeffs(t_xi,   Lagrangarray_xi);
                lagrange_7point_coeffs(t_zeta, Lagrangarray_zeta);

                // ── Fused load + η-reduction from f_old (not f_pc!) ──
                // For each (sj,sk), read si=0..6 from f_old_ptrs[q] directly
                // x is contiguous in memory → coalesced reads
                double interpolation1order[7][7];
                for (int sj = 0; sj < 7; sj++) {
                    int gj = bj + sj;
                    for (int sk = 0; sk < 7; sk++) {
                        int gk = bk + sk;
                        int base_jk = gj * nface + gk * NX6 + bi;
                        interpolation1order[sj][sk] = Intrpl7(
                            f_old_ptrs[q][base_jk + 0], Lagrangarray_eta[0],
                            f_old_ptrs[q][base_jk + 1], Lagrangarray_eta[1],
                            f_old_ptrs[q][base_jk + 2], Lagrangarray_eta[2],
                            f_old_ptrs[q][base_jk + 3], Lagrangarray_eta[3],
                            f_old_ptrs[q][base_jk + 4], Lagrangarray_eta[4],
                            f_old_ptrs[q][base_jk + 5], Lagrangarray_eta[5],
                            f_old_ptrs[q][base_jk + 6], Lagrangarray_eta[6]);
                    }
                }

                // Step B: ξ (j) reduction → 7 values
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

                // Step C: ζ reduction → scalar
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

        f_streamed_all[q] = f_streamed;
        rho_stream += f_streamed;
        mx_stream  += GILBM_e[q][0] * f_streamed;
        my_stream  += GILBM_e[q][1] * f_streamed;
        mz_stream  += GILBM_e[q][2] * f_streamed;
    }

    // ── STEP 1.5: Macroscopic + feq ──
    // Mass correction only for counted points (avoid overlap overcorrection)
    if (i < NX6 - 4 && j < NYD6 - 4) {
        rho_stream += rho_modify[0];
        f_streamed_all[0] += rho_modify[0];
    }
    double rho_A = rho_stream;
    // Half-force velocity correction (Guo 2002): u = (Σξf + 0.5*F*dt) / ρ
    double u_A = (mx_stream + 0.0) / rho_A;             // no x-force
    double v_A = (my_stream + 0.5 * Force[0] * GILBM_dt) / rho_A;  // y=streamwise
    double w_A = (mz_stream + 0.0) / rho_A;             // no z-force

    double feq_A[NQ];
    for (int q = 0; q < NQ; q++) {
        feq_A[q] = compute_feq_alpha(q, rho_A, u_A, v_A, w_A);
        feq_d[q * GRID_SIZE + index] = feq_A[q];
    }
    rho_out_arr[index] = rho_A;
    u_out[index] = u_A;
    v_out[index] = v_A;
    w_out[index] = w_A;

    // ── STEP 2: Point-wise collision at A (no 343-loop, no Re-estimation) ──
    // 三種碰撞算子: Cumulant (AO/WP) > MRT > BGK
#if USE_CUMULANT
    // ================================================================
    // Cumulant collision (D3Q27 native)
    // ── AO or WP mode selected by USE_WP_CUMULANT in variables.h ──
    //
    // Interface: f_in[27], ω₁, dt, Fx, Fy, Fz → f_out[27], ρ, ux, uy, uz
    // Guo forcing (half-force velocity correction) handled INSIDE cumulant.
    // rho_A / u_A / v_A / w_A are RE-WRITTEN by the cumulant kernel
    // (they include the half-force correction from the Chimera transform).
    // ================================================================
    {
        double f_post[NQ];
        double rho_cum, ux_cum, uy_cum, uz_cum;
        cumulant_collision_D3Q27(
            f_streamed_all, omega_global, GILBM_dt,
            0.0, Force[0], 0.0,       // Fx=0, Fy=streamwise force, Fz=0
            f_post, &rho_cum, &ux_cum, &uy_cum, &uz_cum);

        for (int q = 0; q < NQ; q++)
            f_new_ptrs[q][index] = f_post[q];

        // Overwrite macroscopic outputs with cumulant's half-force-corrected values
        rho_out_arr[index] = rho_cum;
        u_out[index] = ux_cum;
        v_out[index] = uy_cum;
        w_out[index] = uz_cum;
    }
#elif USE_MRT
    // D3Q27 MRT collision with Guo forcing (2nd-order accurate)
    gilbm_mrt_collision(f_streamed_all, feq_A, rho_A, u_A, v_A, w_A, GILBM_dt, Force[0]);
    for (int q = 0; q < NQ; q++)
        f_new_ptrs[q][index] = f_streamed_all[q];
#else
    // BGK collision: f* = f̃ - (1/ω)(f̃ - feq) + force
    double inv_omega = 1.0 / omega_global;
    for (int q = 0; q < NQ; q++) {
        double f_post = f_streamed_all[q] - inv_omega * (f_streamed_all[q] - feq_A[q]);
        f_post += GILBM_W[q] * 3.0 * GILBM_e[q][1] * Force[0] * GILBM_dt;
        f_new_ptrs[q][index] = f_post;
    }
#endif
}

#if 0  // ── LTS functions removed for GTS conversion ──
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
    double *delta_zeta_d,  // ζ 方向 RK2 位移 [19*NYD6*NZ6] (space-varying, 1 read/q)
    int *bk_precomp_d,  // 預計算 stencil base k [NZ6], 直接用 k 索引
    double *u_out, double *v_out, double *w_out, double *rho_out_arr,
    double *Force, double *rho_modify
) {
    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    // Local dt and tau at point A
    const double dt_A    = dt_local_d[idx_jk];  // Δt_A (local time step)
    // const double omega_A = omega_local_d[idx_jk]; // [GTS] LTS-only, unused (nvcc #177-D)
    const double omegadt_A = omegadt_local_d[index];  // ω_A × Δt_A = τ_A (教科書鬆弛時間)

    const double a_local = dt_A / GILBM_dt;  // LTS acceleration factor

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
                // ── Runtime departure point + Lagrange weight computation ──
                // Computed BEFORE load to enable fused load+η-reduction
                double t_eta = (double)ci - a_local * GILBM_delta_eta[q];
                if (t_eta < 0.0) t_eta = 0.0; if (t_eta > 6.0) t_eta = 6.0;
                double t_xi  = (double)cj - a_local * GILBM_delta_xi[q];
                if (t_xi  < 0.0) t_xi  = 0.0; if (t_xi  > 6.0) t_xi  = 6.0;
                double delta_zeta = delta_zeta_d[q * NYD6 * NZ6 + idx_jk];
                double up_k = (double)k - delta_zeta;
                if (up_k < 3.0)              up_k = 3.0;
                if (up_k > (double)(NZ6 - 4)) up_k = (double)(NZ6 - 4);
                double t_zeta = up_k - (double)bk;
                double Lagrangarray_eta[7], Lagrangarray_xi[7], Lagrangarray_zeta[7];
                lagrange_7point_coeffs(t_eta,  Lagrangarray_eta);
                lagrange_7point_coeffs(t_xi,   Lagrangarray_xi);
                lagrange_7point_coeffs(t_zeta, Lagrangarray_zeta);

                // ── Fused load + η-reduction (eliminates f_stencil[7][7][7]) ──
                // Old: load 343 values → f_stencil[7][7][7] (2744 bytes → register spill)
                //      then η-reduce from f_stencil → interpolation1order[7][7]
                // New: for each (sj,sk), load si=0..6 directly from f_pc into Intrpl7
                //      → interpolation1order[7][7] (392 bytes, fits in registers)
                // Same 343 global reads, same FMA count, but 7× less local storage.
                const int q_off = q * STENCIL_VOL;
                double interpolation1order[7][7];
                for (int sj = 0; sj < 7; sj++) {
                    for (int sk = 0; sk < 7; sk++) {
                        const int jk_flat = sj * 7 + sk;
                        interpolation1order[sj][sk] = Intrpl7(
                            f_pc[(q_off +   0 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[0],
                            f_pc[(q_off +  49 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[1],
                            f_pc[(q_off +  98 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[2],
                            f_pc[(q_off + 147 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[3],
                            f_pc[(q_off + 196 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[4],
                            f_pc[(q_off + 245 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[5],
                            f_pc[(q_off + 294 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[6]);
                    }
                }

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

                // Step C: ζ reduction -> scalar
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
// Step 2+3 combined MRT operator (P5 optimization)
// Algebraically combines re-estimation (Eq.35) + MRT collision (Eq.3) into
// a single 19×19 matvec: f*[q] = feq_B[q] + R_AB × Σ C_eff[q][q'] × δ[q'] + force[q]
// where C_eff = I - M⁻¹SM is precomputed in shared memory per block.
// Saves ~40% FMAs vs separate m_neq=M×δ + dm=S×m_neq + f*=f̃-Mi×dm.
// ============================================================================
#if USE_MRT
__device__ void gilbm_step23_combined(
    int index, int nface,
    int bi, int bj, int bk,
    double omegadt_A, double dt_A,
    double dk_dy_val, double dk_dz_val,
    bool is_bottom, bool is_top,
    double *f_new_ptrs[19],
    double *f_pc,
    double *feq_d,
    double *omegadt_local_d,
    double Force0,
    const double *C_eff   // 19×19 combined operator from shared memory (row-major)
) {
    // BC detection
    bool need_bc_arr[19];
    need_bc_arr[0] = false;
    for (int q = 1; q < 19; q++) {
        need_bc_arr[q] = false;
        if (is_bottom) need_bc_arr[q] = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
        else if (is_top) need_bc_arr[q] = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
    }

    // Precompute body force per direction (constant for all 343 stencil points)
    double force_q[19];
    for (int q = 0; q < 19; q++)
        force_q[q] = GILBM_W[q] * 3.0 * GILBM_e[q][1] * Force0 * dt_A;

    // Triple loop over 7×7×7 stencil points
    for (int si = 0; si < 7; si++) {
        int gi = bi + si;
        for (int sj = 0; sj < 7; sj++) {
            int gj = bj + sj;
            for (int sk = 0; sk < 7; sk++) {
                int gk = bk + sk;
                int idx_B = gj * nface + gk * NX6 + gi;
                int flat  = si * 49 + sj * 7 + sk;

                // Load all 19 f_B and feq_B, compute δ = f_B - feq_B
                double delta[19], feq_B_arr[19];
                bool ghost_j = (gj < 3 || gj >= NYD6 - 3);
                double rho_B_g, u_B_g, v_B_g, w_B_g;
                if (ghost_j)
                    compute_macroscopic_at(f_new_ptrs, idx_B, rho_B_g, u_B_g, v_B_g, w_B_g);

                for (int q = 0; q < 19; q++) {
                    double f_B = f_new_ptrs[q][idx_B];
                    feq_B_arr[q] = ghost_j
                        ? compute_feq_alpha(q, rho_B_g, u_B_g, v_B_g, w_B_g)
                        : feq_d[q * GRID_SIZE + idx_B];
                    delta[q] = f_B - feq_B_arr[q];
                }

                // R_AB ratio
                double omegadt_B = omegadt_local_d[idx_B];
                double R_AB = omegadt_A / omegadt_B;

                // Combined: f*[q] = feq_B[q] + R_AB × (C_eff × δ)[q] + force[q]
                for (int q = 0; q < 19; q++) {
                    if (need_bc_arr[q]) continue;
                    double Cdelta = 0.0;
                    for (int qp = 0; qp < 19; qp++)
                        Cdelta += C_eff[q * 19 + qp] * delta[qp];
                    f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index] =
                        feq_B_arr[q] + R_AB * Cdelta + force_q[q];
                }
            }
        }
    }
}
#endif

// ============================================================================
// Step 1 + 1.5 only: Interpolation/Streaming + Macroscopic/feq
// Extracted from gilbm_compute_point for split-kernel approach (Kernel A).
// Reads f_pc → writes f_new, feq_d, rho/u/v/w.  Does NOT touch Step 2+3.
// ============================================================================
__device__ void gilbm_step1_point(
    int i, int j, int k,
    double *f_new_ptrs[19],
    double *f_pc,
    double *feq_d,
    double *dk_dz_d, double *dk_dy_d,
    double *dt_local_d, double *omega_local_d,
    double *delta_zeta_d,
    int *bk_precomp_d,
    double *u_out, double *v_out, double *w_out, double *rho_out_arr,
    double *rho_modify
) {
    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    const double dt_A    = dt_local_d[idx_jk];
    const double omega_A = omega_local_d[idx_jk];
    const double a_local = dt_A / GILBM_dt;

    const int bi = i - 3;
    const int bj = j - 3;
    const int bk = bk_precomp_d[k];
    const int ci = i - bi;
    const int cj = j - bj;
    const int ck = k - bk;

    // ── Wall BC pre-computation ──
    bool is_bottom = (k == 3);
    bool is_top    = (k == NZ6 - 4);
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

    double rho_wall = 0.0, du_dk = 0.0, dv_dk = 0.0, dw_dk = 0.0;
    if (is_bottom) {
        int idx3 = j * nface + 4 * NX6 + i;
        int idx4 = j * nface + 5 * NX6 + i;
        double rho3, u3, v3, w3, rho4, u4, v4, w4;
        compute_macroscopic_at(f_new_ptrs, idx3, rho3, u3, v3, w3);
        compute_macroscopic_at(f_new_ptrs, idx4, rho4, u4, v4, w4);
        du_dk = u3;  dv_dk = v3;  dw_dk = w3;
        rho_wall = rho3;
    } else if (is_top) {
        int idxm1 = j * nface + (NZ6 - 5) * NX6 + i;
        int idxm2 = j * nface + (NZ6 - 6) * NX6 + i;
        double rhom1, um1, vm1, wm1, rhom2, um2, vm2, wm2;
        compute_macroscopic_at(f_new_ptrs, idxm1, rhom1, um1, vm1, wm1);
        compute_macroscopic_at(f_new_ptrs, idxm2, rhom2, um2, vm2, wm2);
        du_dk = -(um1);  dv_dk = -(vm1);  dw_dk = -(wm1);
        rho_wall = rhom1;
    }

    // ── STEP 1: Interpolation + Streaming ──
    double rho_stream = 0.0, mx_stream = 0.0, my_stream = 0.0, mz_stream = 0.0;

    for (int q = 0; q < 19; q++) {
        double f_streamed;

        if (q == 0) {
            int center_flat = ci * 49 + cj * 7 + ck;
            f_streamed = f_pc[(q * STENCIL_VOL + center_flat) * GRID_SIZE + index];
        } else {
            bool need_bc = false;
            if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
            else if (is_top) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);
            if (need_bc) {
                f_streamed = ChapmanEnskogBC(q, rho_wall,
                    du_dk, dv_dk, dw_dk,
                    dk_dy_val, dk_dz_val,
                    omega_A, dt_A);
            } else {
                // ── Fused load + η-reduction ──
                double t_eta = (double)ci - a_local * GILBM_delta_eta[q];
                if (t_eta < 0.0) t_eta = 0.0; if (t_eta > 6.0) t_eta = 6.0;
                double t_xi  = (double)cj - a_local * GILBM_delta_xi[q];
                if (t_xi  < 0.0) t_xi  = 0.0; if (t_xi  > 6.0) t_xi  = 6.0;
                double delta_zeta = delta_zeta_d[q * NYD6 * NZ6 + idx_jk];
                double up_k = (double)k - delta_zeta;
                if (up_k < 3.0)              up_k = 3.0;
                if (up_k > (double)(NZ6 - 4)) up_k = (double)(NZ6 - 4);
                double t_zeta = up_k - (double)bk;
                double Lagrangarray_eta[7], Lagrangarray_xi[7], Lagrangarray_zeta[7];
                lagrange_7point_coeffs(t_eta,  Lagrangarray_eta);
                lagrange_7point_coeffs(t_xi,   Lagrangarray_xi);
                lagrange_7point_coeffs(t_zeta, Lagrangarray_zeta);

                const int q_off = q * STENCIL_VOL;
                double interpolation1order[7][7];
                for (int sj = 0; sj < 7; sj++) {
                    for (int sk = 0; sk < 7; sk++) {
                        const int jk_flat = sj * 7 + sk;
                        interpolation1order[sj][sk] = Intrpl7(
                            f_pc[(q_off +   0 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[0],
                            f_pc[(q_off +  49 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[1],
                            f_pc[(q_off +  98 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[2],
                            f_pc[(q_off + 147 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[3],
                            f_pc[(q_off + 196 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[4],
                            f_pc[(q_off + 245 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[5],
                            f_pc[(q_off + 294 + jk_flat) * GRID_SIZE + index], Lagrangarray_eta[6]);
                    }
                }

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

        f_new_ptrs[q][index] = f_streamed;
        rho_stream += f_streamed;
        mx_stream  += GILBM_e[q][0] * f_streamed;
        my_stream  += GILBM_e[q][1] * f_streamed;
        mz_stream  += GILBM_e[q][2] * f_streamed;
    }

    // ── STEP 1.5: Macroscopic + feq ──
    // Apply rho_modify ONLY to points counted by ReduceRhoSum (i<NX6-4, j<NYD6-4)
    // to avoid 9.57% overcorrection at x-periodic overlap (i=NX6-4) and
    // MPI overlap (j=NYD6-4) which receive correction but aren't counted.
    if (i < NX6 - 4 && j < NYD6 - 4) {
        rho_stream += rho_modify[0];
        f_new_ptrs[0][index] += rho_modify[0];
    }
    double rho_A = rho_stream;
    double u_A   = mx_stream / rho_A;
    double v_A   = my_stream / rho_A;
    double w_A   = mz_stream / rho_A;
    for (int q = 0; q < 19; q++)
        feq_d[q * GRID_SIZE + index] = compute_feq_alpha(q, rho_A, u_A, v_A, w_A);
    rho_out_arr[index] = rho_A;
    u_out[index] = u_A;
    v_out[index] = v_A;
    w_out[index] = w_A;
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
    // const double omega_A = omega_local_d[idx_jk]; // [GTS] LTS-only, unused (nvcc #177-D)
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
    double *delta_zeta_d,
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
        delta_zeta_d,
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
    double *delta_zeta_d,
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
        delta_zeta_d,
        bk_precomp_d,
        u_out, v_out, w_out, rho_out,
        Force, rho_modify);
}

// ============================================================================
// Split-kernel Kernel A: Step 1 + 1.5 for all interior j-rows
// Pure Jacobi: all f_new written before any Step 2+3 reads them.
// ============================================================================
__global__ void GILBM_Step1_Kernel(
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new,
    double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new,
    double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new,
    double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *f_pc, double *feq_d,
    double *dk_dz_d, double *dk_dy_d,
    double *dt_local_d, double *omega_local_d,
    double *delta_zeta_d,
    int *bk_precomp_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *rho_modify
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 2 || i >= NX6 - 3 || j < 3 || j >= NYD6 - 3 || k <= 2 || k >= NZ6 - 3) return;

    double *f_new_ptrs[19] = {
        f0_new, f1_new, f2_new, f3_new, f4_new, f5_new, f6_new,
        f7_new, f8_new, f9_new, f10_new, f11_new, f12_new,
        f13_new, f14_new, f15_new, f16_new, f17_new, f18_new
    };

    gilbm_step1_point(i, j, k, f_new_ptrs,
        f_pc, feq_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        delta_zeta_d, bk_precomp_d,
        u_out, v_out, w_out, rho_out,
        rho_modify);
}

// ============================================================================
// Split-kernel Kernel B: Step 2+3 for all interior j-rows
// Runs AFTER MPI exchange + periodicSW, so all f_new/feq ghost zones are valid.
// P5 optimization: builds C_eff = C0 - s_visc × C1 in shared memory per block,
// then uses combined operator: f* = feq_B + R_AB × C_eff × δ + force
// ============================================================================
__global__ void GILBM_Step23_Full_Kernel(
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new,
    double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new,
    double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new,
    double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *f_pc, double *feq_d, double *omegadt_local_d,
    double *dk_dz_d, double *dk_dy_d,
    double *dt_local_d, double *omega_local_d,
    int *bk_precomp_d,
    double *Force
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;   // blockDim.y = 1 → j = blockIdx.y
    const int k = blockIdx.z;   // blockDim.z = 1 → k = blockIdx.z

#if USE_MRT
    // ── Build C_eff in shared memory (ALL threads participate, before any return) ──
    // C_eff = C0 - s_visc × C1, same for all threads in block (same j,k → same s_visc)
    __shared__ double C_eff_shared[19 * 19];
    {
        double s_visc_val = 0.0;
        if (j >= 3 && j < NYD6 - 3 && k > 2 && k < NZ6 - 3) {
            int idx_jk_block = j * NZ6 + k;
            s_visc_val = 1.0 / omega_local_d[idx_jk_block];
        }
        // First 19 threads compute one row each of C_eff (19×19 = 361 entries)
        if (threadIdx.x < 19) {
            int q = threadIdx.x;
            for (int qp = 0; qp < 19; qp++)
                C_eff_shared[q * 19 + qp] = GILBM_C0[q][qp] - s_visc_val * GILBM_C1[q][qp];
        }
    }
    __syncthreads();
#endif

    // ── Boundary guard (after syncthreads to avoid deadlock) ──
    if (i <= 2 || i >= NX6 - 3 || j < 3 || j >= NYD6 - 3 || k <= 2 || k >= NZ6 - 3) return;

    double *f_new_ptrs[19] = {
        f0_new, f1_new, f2_new, f3_new, f4_new, f5_new, f6_new,
        f7_new, f8_new, f9_new, f10_new, f11_new, f12_new,
        f13_new, f14_new, f15_new, f16_new, f17_new, f18_new
    };

    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;

    const double dt_A      = dt_local_d[idx_jk];
    // const double omega_A = omega_local_d[idx_jk]; // [GTS] LTS-only, unused (nvcc #177-D)
    const double omegadt_A = omegadt_local_d[index];

    const int bi = i - 3;
    const int bj = j - 3;
    const int bk = bk_precomp_d[k];

    bool is_bottom = (k == 3);
    bool is_top    = (k == NZ6 - 4);
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

#if USE_MRT
    gilbm_step23_combined(index, nface, bi, bj, bk,
                          omegadt_A, dt_A,
                          dk_dy_val, dk_dz_val,
                          is_bottom, is_top,
                          f_new_ptrs, f_pc, feq_d, omegadt_local_d,
                          Force[0], C_eff_shared);
#else
    gilbm_step23_point(index, nface, bi, bj, bk,
                       omega_A, omegadt_A, dt_A,
                       dk_dy_val, dk_dz_val,
                       is_bottom, is_top,
                       f_new_ptrs, f_pc, feq_d, omegadt_local_d,
                       Force[0]);
#endif
}
#endif  // ── end LTS kernels ──


// ============================================================================
// [GTS] Single-pass kernel: interpolation + collision (D3Q27)
// Uses device pointer arrays instead of 27 individual args.
// Double-buffer: reads f_old (ft), writes f_new (fd), then swap in host code.
// ============================================================================
__global__ void GILBM_GTS_Kernel(
    double **f_old_d,         // device array of NQ pointers (previous timestep)
    double **f_new_d,         // device array of NQ pointers (new timestep)
    double *feq_d,
    double *dk_dz_d, double *dk_dy_d,
    double *delta_zeta_d,
    int *bk_precomp_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *Force, double *rho_modify,
    double omega_global
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Buffer=3: compute i∈[3,NX6-4], j∈[3,NYD6-4], k∈[3,NZ6-4]
    if (i <= 2 || i >= NX6 - 3 || j < 3 || j >= NYD6 - 3 || k <= 2 || k >= NZ6 - 3) return;

    // Load pointer arrays from global memory to registers
    double *f_old_ptrs[NQ], *f_new_ptrs[NQ];
    for (int q = 0; q < NQ; q++) {
        f_old_ptrs[q] = f_old_d[q];
        f_new_ptrs[q] = f_new_d[q];
    }

    gilbm_compute_point_gts(i, j, k,
        f_old_ptrs, f_new_ptrs,
        feq_d,
        dk_dz_d, dk_dy_d,
        delta_zeta_d, bk_precomp_d,
        u_out, v_out, w_out, rho_out,
        Force, rho_modify,
        omega_global);
}


// ============================================================================
// Initialization kernel: compute feq_d from initial f arrays (D3Q27)
// ============================================================================
__global__ void Init_Feq_Kernel(
    double **f_d,             // device array of NQ pointers
    double *feq_d
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX6 || j >= NYD6 || k >= NZ6) return;

    const int index = j * NX6 * NZ6 + k * NX6 + i;

    double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
    for (int q = 0; q < NQ; q++) {
        double fq = f_d[q][index];
        rho += fq;
        ux  += GILBM_e[q][0] * fq;
        uy  += GILBM_e[q][1] * fq;
        uz  += GILBM_e[q][2] * fq;
    }
    ux /= rho;
    uy /= rho;
    uz /= rho;

    for (int q = 0; q < NQ; q++) {
        feq_d[q * GRID_SIZE + index] = compute_feq_alpha(q, rho, ux, uy, uz);
    }
}

#if 0  // [GTS] Init_OmegaDt_Kernel removed (LTS only)
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
#endif  // end Init_OmegaDt_Kernel

#endif
