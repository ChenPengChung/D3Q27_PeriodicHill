#ifndef GILBM_EVOLUTION_H
#define GILBM_EVOLUTION_H

// Phase 1: GILBM streaming + MRT collision kernel (Imamura 2005)
//
// Replaces ISLBM stream_collide_Buffer / stream_collide.
// Key differences from ISLBM:
//   - Streaming: contravariant velocity RK2 + 2nd-order quadratic interpolation (vs 7th-order Lagrange)
//   - Wall BC: Chapman-Enskog (vs bounce-back + BFL)
//   - No Xi coordinate, no BFL parameters
//   - Collision: MRT unchanged (reuses MRT_Matrix.h + MRT_Process.h)

// __constant__ device memory for D3Q19 velocity set and weights
__constant__ double GILBM_e[19][3] = {
    {0,0,0},
    {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
    {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
    {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
    {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
};

// Phase 3: Runtime dt and tau from Imamura global time step
__constant__ double GILBM_dt;
__constant__ double GILBM_tau;

// Precomputed displacement arrays (constant for uniform x and y)
// NOTE: 壁面邊界節點上，ẽ^ζ_α > 0（底壁）或 < 0（頂壁）的方向由
// Chapman-Enskog BC 處理，不走 streaming，因此對該 α 不讀取 delta_eta/delta_xi。
// 判定邏輯見 NeedsBoundaryCondition()（boundary_conditions.h）。
__constant__ double GILBM_delta_eta[19]; // δη[α] = dt · e_x[α] / dx
__constant__ double GILBM_delta_xi[19];  // δξ[α] = dt · e_y[α] / dy

__constant__ double GILBM_W[19] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// Include GILBM sub-modules (after __constant__ declarations they depend on)
#include "gilbm/interpolation_gilbm.h"
#include "gilbm/boundary_conditions.h"

// Helper: compute macroscopic quantities from f at a given 3D index
__device__ __forceinline__ void compute_macroscopic_at(
    double *f0, double *f1, double *f2, double *f3, double *f4,
    double *f5, double *f6, double *f7, double *f8, double *f9,
    double *f10, double *f11, double *f12, double *f13, double *f14,
    double *f15, double *f16, double *f17, double *f18,
    int idx, double &rho_out, double &ux_out, double &uy_out, double &uz_out
) {
    double f[19];
    f[0]=f0[idx]; f[1]=f1[idx]; f[2]=f2[idx]; f[3]=f3[idx]; f[4]=f4[idx];
    f[5]=f5[idx]; f[6]=f6[idx]; f[7]=f7[idx]; f[8]=f8[idx]; f[9]=f9[idx];
    f[10]=f10[idx]; f[11]=f11[idx]; f[12]=f12[idx]; f[13]=f13[idx]; f[14]=f14[idx];
    f[15]=f15[idx]; f[16]=f16[idx]; f[17]=f17[idx]; f[18]=f18[idx];

    rho_out = f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]+f[8]+f[9]
             +f[10]+f[11]+f[12]+f[13]+f[14]+f[15]+f[16]+f[17]+f[18];
    ux_out = (f[1]+f[7]+f[9]+f[11]+f[13] - (f[2]+f[8]+f[10]+f[12]+f[14])) / rho_out;
    uy_out = (f[3]+f[7]+f[8]+f[15]+f[17] - (f[4]+f[9]+f[10]+f[16]+f[18])) / rho_out;
    uz_out = (f[5]+f[11]+f[12]+f[15]+f[16] - (f[6]+f[13]+f[14]+f[17]+f[18])) / rho_out;
}

// ============================================================================
// GILBM Buffer kernel (processes buffer j-rows for MPI overlap, with start offset)
// ============================================================================
__global__ void stream_collide_GILBM_Buffer(
    double *f0_old, double *f1_old, double *f2_old, double *f3_old, double *f4_old,
    double *f5_old, double *f6_old, double *f7_old, double *f8_old, double *f9_old,
    double *f10_old, double *f11_old, double *f12_old, double *f13_old, double *f14_old,
    double *f15_old, double *f16_old, double *f17_old, double *f18_old,
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new,
    double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new,
    double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new,
    double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *dk_dz_d, double *dk_dy_d, double *delta_zeta_d,
    double *dt_local_d, double *tau_local_d, double *tau_dt_product_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *Force, double *rho_modify, int start
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + start;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 2 || i >= NX6 - 3 || k <= 1 || k >= NZ6 - 2) return;

    const int index = j * NX6 * NZ6 + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;
    const int nface = NX6 * NZ6;

    // Phase 4: local dt and tau from per-point fields
    const double dt = dt_local_d[idx_jk];
    const double tau = tau_local_d[idx_jk];
    const double omega = 1.0 / tau;

    // MRT variables
    double F0_in, F1_in, F2_in, F3_in, F4_in, F5_in, F6_in, F7_in, F8_in, F9_in;
    double F10_in, F11_in, F12_in, F13_in, F14_in, F15_in, F16_in, F17_in, F18_in;
    double m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18;
    double s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18;
    double meq0,meq1,meq2,meq3,meq4,meq5,meq6,meq7,meq8,meq9,meq10,meq11,meq12,meq13,meq14,meq15,meq16,meq17,meq18;

    Matrix;
    Inverse_Matrix;
    Relaxation;

    // Pointers array for interpolation access
    double *f_old_ptrs[19] = {
        f0_old, f1_old, f2_old, f3_old, f4_old, f5_old, f6_old,
        f7_old, f8_old, f9_old, f10_old, f11_old, f12_old,
        f13_old, f14_old, f15_old, f16_old, f17_old, f18_old
    };

    // ===== F0: rest direction, no streaming =====
    F0_in = f0_old[index];

    // ===== GILBM Streaming for alpha=1..18 =====
    bool is_bottom = (k == 2);
    bool is_top = (k == NZ6 - 3);
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

    // Pre-compute wall data only if at wall (shared across all alpha)
    double rho_wall = 0.0, du_x_dk = 0.0, du_y_dk = 0.0, du_z_dk = 0.0;
    if (is_bottom) {
        // Read macroscopic at k=3 and k=4 for one-sided gradient
        int idx3 = j * nface + 3 * NX6 + i;
        int idx4 = j * nface + 4 * NX6 + i;
        double rho3, ux3, uy3, uz3, rho4, ux4, uy4, uz4;
        compute_macroscopic_at(f0_old, f1_old, f2_old, f3_old, f4_old,
            f5_old, f6_old, f7_old, f8_old, f9_old,
            f10_old, f11_old, f12_old, f13_old, f14_old,
            f15_old, f16_old, f17_old, f18_old,
            idx3, rho3, ux3, uy3, uz3);
        compute_macroscopic_at(f0_old, f1_old, f2_old, f3_old, f4_old,
            f5_old, f6_old, f7_old, f8_old, f9_old,
            f10_old, f11_old, f12_old, f13_old, f14_old,
            f15_old, f16_old, f17_old, f18_old,
            idx4, rho4, ux4, uy4, uz4);
        // 2nd-order one-sided: du/dk = (4*u[3] - u[4]) / 2  (u[wall]=0)
        du_x_dk = (4.0 * ux3 - ux4) / 2.0;
        du_y_dk = (4.0 * uy3 - uy4) / 2.0;
        du_z_dk = (4.0 * uz3 - uz4) / 2.0;
        rho_wall = rho3;
    } else if (is_top) {
        int idxm1 = j * nface + (NZ6 - 4) * NX6 + i;
        int idxm2 = j * nface + (NZ6 - 5) * NX6 + i;
        double rhom1, uxm1, uym1, uzm1, rhom2, uxm2, uym2, uzm2;
        compute_macroscopic_at(f0_old, f1_old, f2_old, f3_old, f4_old,
            f5_old, f6_old, f7_old, f8_old, f9_old,
            f10_old, f11_old, f12_old, f13_old, f14_old,
            f15_old, f16_old, f17_old, f18_old,
            idxm1, rhom1, uxm1, uym1, uzm1);
        compute_macroscopic_at(f0_old, f1_old, f2_old, f3_old, f4_old,
            f5_old, f6_old, f7_old, f8_old, f9_old,
            f10_old, f11_old, f12_old, f13_old, f14_old,
            f15_old, f16_old, f17_old, f18_old,
            idxm2, rhom2, uxm2, uym2, uzm2);
        // Top wall: du/dk = -(4*u[NZ6-4] - u[NZ6-5]) / 2  (u[wall]=0, inward direction)
        du_x_dk = -(4.0 * uxm1 - uxm2) / 2.0;
        du_y_dk = -(4.0 * uym1 - uym2) / 2.0;
        du_z_dk = -(4.0 * uzm1 - uzm2) / 2.0;
        rho_wall = rhom1;
    }

    // Streaming for each non-rest direction
    double F_in_arr[19];
    F_in_arr[0] = F0_in;

    // Phase 4: LTS acceleration factor and re-estimation prefactor
    const double a_local = dt / GILBM_dt;               // ≥ 1 (local dt / global dt)
    const double tau_A_minus1_dt_A = (tau - 1.0) * dt;  // for re-estimation Eq. 36

    // ====================================================================
    // Step 1 插值 + Step 2 Re-estimation + Step 3 Streaming（合併迴圈）
    // ====================================================================
    // 【注意事項 Step 1 插值】
    //   插值在 departure point D 的 stencil 鄰近點執行，非在計算點 A，
    //   插值誤差直接影響後續所有步驟精度。
    //
    // 【注意事項 Step 2 Re-estimation】
    //   此處的 f_eq 使用 stencil 點 B 的宏觀量計算（非 A 點），
    //   對應 Imamura Eq.36 的 LTS 重估項：
    //   f̃_B = feq_B + (f_B - feq_B) × R_AB，
    //   R_AB = (τ_A - 1)dt_A / (τ_B · dt_B) 處理 τ 空間不均勻性。
    //
    // 【注意事項 Step 3 Streaming】
    //   Streaming 本質為資料傳遞，不涉及物理計算。
    //   雖然 Imamura 原文將碰撞寫在 streaming 之前，兩者在數學上等價，
    //   因為碰撞固定使用 A 點的 τ_A，與傳遞順序無關。
    //
    // 判定準則（見 NeedsBoundaryCondition, boundary_conditions.h）：
    //   ẽ^ζ_α = e_y[α]·dk_dy + e_z[α]·dk_dz
    //   底壁 (k=2):   ẽ^ζ_α > 0 → 出發點在壁外 → BC（Chapman-Enskog）
    //   頂壁 (k=NZ6-3): ẽ^ζ_α < 0 → 出發點在壁外 → BC（Chapman-Enskog）
    //   其他情況 → streaming（使用 delta_eta, delta_xi, delta_zeta 計算出發點）
    //
    // BC 方向不讀取 delta_eta[α] / delta_xi[α] / delta_zeta[α,j,k]，
    // 這些值雖在 precompute 階段已計算，但對壁面 BC 方向無效。
    // ====================================================================
    for (int alpha = 1; alpha < 19; alpha++) {
        bool need_bc = false;
        if (is_bottom) {
            need_bc = NeedsBoundaryCondition(alpha, dk_dy_val, dk_dz_val, true);
        } else if (is_top) {
            need_bc = NeedsBoundaryCondition(alpha, dk_dy_val, dk_dz_val, false);
        }

        if (need_bc) {
            // BC 方向：ẽ^ζ_α 指向壁外 → 跳過 streaming → Chapman-Enskog 重建 f_α
            // delta_eta[α], delta_xi[α], delta_zeta[α,...] 對此 α 不使用
            F_in_arr[alpha] = ChapmanEnskogBC(alpha, rho_wall,
                du_x_dk, du_y_dk, du_z_dk,
                dk_dy_val, dk_dz_val, omega, dt);
        } else {
            // Streaming 方向：ẽ^ζ_α 指向流體內部（或為零）→ 使用位移量計算出發點
            // Phase 4: scale η/ξ displacement by local acceleration factor
            double delta_i = a_local * GILBM_delta_eta[alpha];
            double delta_xi_val = a_local * GILBM_delta_xi[alpha];
            // ζ displacement already precomputed with local dt
            double delta_zeta_val = delta_zeta_d[alpha * NYD6 * NZ6 + idx_jk];

            double up_i = (double)i - delta_i;
            double up_j = (double)j - delta_xi_val;
            double up_k = (double)k - delta_zeta_val;

            // Clamp upwind point to valid interpolation range
            if (up_i < 1.0) up_i = 1.0;
            if (up_i > (double)(NX6 - 3)) up_i = (double)(NX6 - 3);
            if (up_j < 1.0) up_j = 1.0;
            if (up_j > (double)(NYD6 - 3)) up_j = (double)(NYD6 - 3);
            if (up_k < 2.0) up_k = 2.0;
            if (up_k > (double)(NZ6 - 5)) up_k = (double)(NZ6 - 5);

            // Step 1 插值 + Step 2 Re-estimation（均在 interpolate_quadratic_3d_lts 內執行）
            // Step 3 Streaming：下行賦值即為 f 從 departure point D 傳遞至計算點 A
            F_in_arr[alpha] = interpolate_quadratic_3d_lts(
                up_i, up_j, up_k,
                f_old_ptrs, tau_dt_product_d,
                alpha, tau_A_minus1_dt_A,
                NX6, NZ6);
        }
    }

    // Assign to named variables for MRT macros
    F0_in = F_in_arr[0];   F1_in = F_in_arr[1];   F2_in = F_in_arr[2];
    F3_in = F_in_arr[3];   F4_in = F_in_arr[4];   F5_in = F_in_arr[5];
    F6_in = F_in_arr[6];   F7_in = F_in_arr[7];   F8_in = F_in_arr[8];
    F9_in = F_in_arr[9];   F10_in = F_in_arr[10];  F11_in = F_in_arr[11];
    F12_in = F_in_arr[12];  F13_in = F_in_arr[13];  F14_in = F_in_arr[14];
    F15_in = F_in_arr[15];  F16_in = F_in_arr[16];  F17_in = F_in_arr[17];
    F18_in = F_in_arr[18];

    // ====================================================================
    // Step 3 Streaming 完成：F_in_arr[α] 已將各方向 f 從 departure point D 傳遞至計算點 A
    // ====================================================================

    // ===== Global mass correction =====
    F0_in = F0_in + rho_modify[0];

    // ====================================================================
    // Step 4：碰撞 Collision（在計算點 A，使用 τ_A）
    // ====================================================================
    // 【注意事項 Step 4 碰撞】
    //   碰撞使用計算點 A 的 τ_A（非 departure point 的 τ_D），
    //   此為 GILBM 與標準 LBM 的關鍵差異。此步的 f_eq 使用 A 點宏觀量重新計算，
    //   確保 Σf_i = ρ 在碰撞步精確成立（離散守恆性）。
    // ====================================================================

    // ===== Macroscopic quantities =====
    double rho_s = F0_in + F1_in + F2_in + F3_in + F4_in + F5_in + F6_in + F7_in + F8_in + F9_in
                 + F10_in + F11_in + F12_in + F13_in + F14_in + F15_in + F16_in + F17_in + F18_in;
    double u1 = (F1_in + F7_in + F9_in + F11_in + F13_in - (F2_in + F8_in + F10_in + F12_in + F14_in)) / rho_s;
    double v1 = (F3_in + F7_in + F8_in + F15_in + F17_in - (F4_in + F9_in + F10_in + F16_in + F18_in)) / rho_s;
    double w1 = (F5_in + F11_in + F12_in + F15_in + F16_in - (F6_in + F13_in + F14_in + F17_in + F18_in)) / rho_s;
    double udot = u1 * u1 + v1 * v1 + w1 * w1;

    // ===== Equilibrium =====
    const double F0_eq  = (1./3.)  * rho_s * (1.0 - 1.5 * udot);
    const double F1_eq  = (1./18.) * rho_s * (1.0 + 3.0 * u1 + 4.5 * u1 * u1 - 1.5 * udot);
    const double F2_eq  = (1./18.) * rho_s * (1.0 - 3.0 * u1 + 4.5 * u1 * u1 - 1.5 * udot);
    const double F3_eq  = (1./18.) * rho_s * (1.0 + 3.0 * v1 + 4.5 * v1 * v1 - 1.5 * udot);
    const double F4_eq  = (1./18.) * rho_s * (1.0 - 3.0 * v1 + 4.5 * v1 * v1 - 1.5 * udot);
    const double F5_eq  = (1./18.) * rho_s * (1.0 + 3.0 * w1 + 4.5 * w1 * w1 - 1.5 * udot);
    const double F6_eq  = (1./18.) * rho_s * (1.0 - 3.0 * w1 + 4.5 * w1 * w1 - 1.5 * udot);
    const double F7_eq  = (1./36.) * rho_s * (1.0 + 3.0 * (u1 + v1) + 4.5 * (u1 + v1) * (u1 + v1) - 1.5 * udot);
    const double F8_eq  = (1./36.) * rho_s * (1.0 + 3.0 * (-u1 + v1) + 4.5 * (-u1 + v1) * (-u1 + v1) - 1.5 * udot);
    const double F9_eq  = (1./36.) * rho_s * (1.0 + 3.0 * (u1 - v1) + 4.5 * (u1 - v1) * (u1 - v1) - 1.5 * udot);
    const double F10_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-u1 - v1) + 4.5 * (-u1 - v1) * (-u1 - v1) - 1.5 * udot);
    const double F11_eq = (1./36.) * rho_s * (1.0 + 3.0 * (u1 + w1) + 4.5 * (u1 + w1) * (u1 + w1) - 1.5 * udot);
    const double F12_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-u1 + w1) + 4.5 * (-u1 + w1) * (-u1 + w1) - 1.5 * udot);
    const double F13_eq = (1./36.) * rho_s * (1.0 + 3.0 * (u1 - w1) + 4.5 * (u1 - w1) * (u1 - w1) - 1.5 * udot);
    const double F14_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-u1 - w1) + 4.5 * (-u1 - w1) * (-u1 - w1) - 1.5 * udot);
    const double F15_eq = (1./36.) * rho_s * (1.0 + 3.0 * (v1 + w1) + 4.5 * (v1 + w1) * (v1 + w1) - 1.5 * udot);
    const double F16_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-v1 + w1) + 4.5 * (-v1 + w1) * (-v1 + w1) - 1.5 * udot);
    const double F17_eq = (1./36.) * rho_s * (1.0 + 3.0 * (v1 - w1) + 4.5 * (v1 - w1) * (v1 - w1) - 1.5 * udot);
    const double F18_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-v1 - w1) + 4.5 * (-v1 - w1) * (-v1 - w1) - 1.5 * udot);

    // ===== MRT Collision (reused from existing code) =====
    m_matrix;
    meq;
    collision;

    // ===== Write output =====
    __syncthreads();
    f0_new[index]  = F0_in;   f1_new[index]  = F1_in;   f2_new[index]  = F2_in;
    f3_new[index]  = F3_in;   f4_new[index]  = F4_in;   f5_new[index]  = F5_in;
    f6_new[index]  = F6_in;   f7_new[index]  = F7_in;   f8_new[index]  = F8_in;
    f9_new[index]  = F9_in;   f10_new[index] = F10_in;  f11_new[index] = F11_in;
    f12_new[index] = F12_in;  f13_new[index] = F13_in;  f14_new[index] = F14_in;
    f15_new[index] = F15_in;  f16_new[index] = F16_in;  f17_new[index] = F17_in;
    f18_new[index] = F18_in;
    u_out[index] = u1;  v_out[index] = v1;  w_out[index] = w1;  rho_out[index] = rho_s;
}

// ============================================================================
// GILBM full-grid kernel (no start offset, for main grid computation)
// ============================================================================
__global__ void stream_collide_GILBM(
    double *f0_old, double *f1_old, double *f2_old, double *f3_old, double *f4_old,
    double *f5_old, double *f6_old, double *f7_old, double *f8_old, double *f9_old,
    double *f10_old, double *f11_old, double *f12_old, double *f13_old, double *f14_old,
    double *f15_old, double *f16_old, double *f17_old, double *f18_old,
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new,
    double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new,
    double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new,
    double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *dk_dz_d, double *dk_dy_d, double *delta_zeta_d,
    double *dt_local_d, double *tau_local_d, double *tau_dt_product_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *Force, double *rho_modify
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 2 || i >= NX6 - 3 || k <= 1 || k >= NZ6 - 2) return;

    const int index = j * NX6 * NZ6 + k * NX6 + i;
    const int idx_jk = j * NZ6 + k;
    const int nface = NX6 * NZ6;

    // Phase 4: local dt and tau from per-point fields
    const double dt = dt_local_d[idx_jk];
    const double tau = tau_local_d[idx_jk];
    const double omega = 1.0 / tau;

    double F0_in, F1_in, F2_in, F3_in, F4_in, F5_in, F6_in, F7_in, F8_in, F9_in;
    double F10_in, F11_in, F12_in, F13_in, F14_in, F15_in, F16_in, F17_in, F18_in;
    double m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18;
    double s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18;
    double meq0,meq1,meq2,meq3,meq4,meq5,meq6,meq7,meq8,meq9,meq10,meq11,meq12,meq13,meq14,meq15,meq16,meq17,meq18;

    Matrix;
    Inverse_Matrix;
    Relaxation;

    double *f_old_ptrs[19] = {
        f0_old, f1_old, f2_old, f3_old, f4_old, f5_old, f6_old,
        f7_old, f8_old, f9_old, f10_old, f11_old, f12_old,
        f13_old, f14_old, f15_old, f16_old, f17_old, f18_old
    };

    F0_in = f0_old[index];

    bool is_bottom = (k == 2);
    bool is_top = (k == NZ6 - 3);
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

    double rho_wall = 0.0, du_x_dk = 0.0, du_y_dk = 0.0, du_z_dk = 0.0;
    if (is_bottom) {
        int idx3 = j * nface + 3 * NX6 + i;
        int idx4 = j * nface + 4 * NX6 + i;
        double rho3, ux3, uy3, uz3, rho4, ux4, uy4, uz4;
        compute_macroscopic_at(f0_old, f1_old, f2_old, f3_old, f4_old,
            f5_old, f6_old, f7_old, f8_old, f9_old,
            f10_old, f11_old, f12_old, f13_old, f14_old,
            f15_old, f16_old, f17_old, f18_old,
            idx3, rho3, ux3, uy3, uz3);
        compute_macroscopic_at(f0_old, f1_old, f2_old, f3_old, f4_old,
            f5_old, f6_old, f7_old, f8_old, f9_old,
            f10_old, f11_old, f12_old, f13_old, f14_old,
            f15_old, f16_old, f17_old, f18_old,
            idx4, rho4, ux4, uy4, uz4);
        du_x_dk = (4.0 * ux3 - ux4) / 2.0;
        du_y_dk = (4.0 * uy3 - uy4) / 2.0;
        du_z_dk = (4.0 * uz3 - uz4) / 2.0;
        rho_wall = rho3;
    } else if (is_top) {
        int idxm1 = j * nface + (NZ6 - 4) * NX6 + i;
        int idxm2 = j * nface + (NZ6 - 5) * NX6 + i;
        double rhom1, uxm1, uym1, uzm1, rhom2, uxm2, uym2, uzm2;
        compute_macroscopic_at(f0_old, f1_old, f2_old, f3_old, f4_old,
            f5_old, f6_old, f7_old, f8_old, f9_old,
            f10_old, f11_old, f12_old, f13_old, f14_old,
            f15_old, f16_old, f17_old, f18_old,
            idxm1, rhom1, uxm1, uym1, uzm1);
        compute_macroscopic_at(f0_old, f1_old, f2_old, f3_old, f4_old,
            f5_old, f6_old, f7_old, f8_old, f9_old,
            f10_old, f11_old, f12_old, f13_old, f14_old,
            f15_old, f16_old, f17_old, f18_old,
            idxm2, rhom2, uxm2, uym2, uzm2);
        du_x_dk = -(4.0 * uxm1 - uxm2) / 2.0;
        du_y_dk = -(4.0 * uym1 - uym2) / 2.0;
        du_z_dk = -(4.0 * uzm1 - uzm2) / 2.0;
        rho_wall = rhom1;
    }

    double F_in_arr[19];
    F_in_arr[0] = F0_in;

    // Phase 4: LTS acceleration factor and re-estimation prefactor
    const double a_local = dt / GILBM_dt;
    const double tau_A_minus1_dt_A = (tau - 1.0) * dt;

    // ====================================================================
    // Step 1 插值 + Step 2 Re-estimation + Step 3 Streaming（合併迴圈）
    // ====================================================================
    // 【注意事項 Step 1 插值】
    //   插值在 departure point D 的 stencil 鄰近點執行，非在計算點 A，
    //   插值誤差直接影響後續所有步驟精度。
    //
    // 【注意事項 Step 2 Re-estimation】
    //   此處的 f_eq 使用 stencil 點 B 的宏觀量計算（非 A 點），
    //   對應 Imamura Eq.36 的 LTS 重估項：
    //   f̃_B = feq_B + (f_B - feq_B) × R_AB，
    //   R_AB = (τ_A - 1)dt_A / (τ_B · dt_B) 處理 τ 空間不均勻性。
    //
    // 【注意事項 Step 3 Streaming】
    //   Streaming 本質為資料傳遞，不涉及物理計算。
    //   雖然 Imamura 原文將碰撞寫在 streaming 之前，兩者在數學上等價，
    //   因為碰撞固定使用 A 點的 τ_A，與傳遞順序無關。
    //
    // （詳細判定準則見第一個 kernel 同名區段）
    // ====================================================================
    for (int alpha = 1; alpha < 19; alpha++) {
        bool need_bc = false;
        if (is_bottom) {
            need_bc = NeedsBoundaryCondition(alpha, dk_dy_val, dk_dz_val, true);
        } else if (is_top) {
            need_bc = NeedsBoundaryCondition(alpha, dk_dy_val, dk_dz_val, false);
        }

        if (need_bc) {
            // BC 方向：跳過 streaming，Chapman-Enskog 重建（delta_eta/xi/zeta 不使用）
            F_in_arr[alpha] = ChapmanEnskogBC(alpha, rho_wall,
                du_x_dk, du_y_dk, du_z_dk,
                dk_dy_val, dk_dz_val, omega, dt);
        } else {
            // Streaming 方向：使用位移量計算出發點
            double delta_i = a_local * GILBM_delta_eta[alpha];
            double delta_xi_val = a_local * GILBM_delta_xi[alpha];
            double delta_zeta_val = delta_zeta_d[alpha * NYD6 * NZ6 + idx_jk];

            double up_i = (double)i - delta_i;
            double up_j = (double)j - delta_xi_val;
            double up_k = (double)k - delta_zeta_val;

            if (up_i < 1.0) up_i = 1.0;
            if (up_i > (double)(NX6 - 3)) up_i = (double)(NX6 - 3);
            if (up_j < 1.0) up_j = 1.0;
            if (up_j > (double)(NYD6 - 3)) up_j = (double)(NYD6 - 3);
            if (up_k < 2.0) up_k = 2.0;
            if (up_k > (double)(NZ6 - 5)) up_k = (double)(NZ6 - 5);

            // Step 1 插值 + Step 2 Re-estimation（均在 interpolate_quadratic_3d_lts 內執行）
            // Step 3 Streaming：下行賦值即為 f 從 departure point D 傳遞至計算點 A
            F_in_arr[alpha] = interpolate_quadratic_3d_lts(
                up_i, up_j, up_k,
                f_old_ptrs, tau_dt_product_d,
                alpha, tau_A_minus1_dt_A,
                NX6, NZ6);
        }
    }

    F0_in = F_in_arr[0];   F1_in = F_in_arr[1];   F2_in = F_in_arr[2];
    F3_in = F_in_arr[3];   F4_in = F_in_arr[4];   F5_in = F_in_arr[5];
    F6_in = F_in_arr[6];   F7_in = F_in_arr[7];   F8_in = F_in_arr[8];
    F9_in = F_in_arr[9];   F10_in = F_in_arr[10];  F11_in = F_in_arr[11];
    F12_in = F_in_arr[12];  F13_in = F_in_arr[13];  F14_in = F_in_arr[14];
    F15_in = F_in_arr[15];  F16_in = F_in_arr[16];  F17_in = F_in_arr[17];
    F18_in = F_in_arr[18];

    // ====================================================================
    // Step 3 Streaming 完成：F_in_arr[α] 已將各方向 f 從 departure point D 傳遞至計算點 A
    // ====================================================================

    F0_in = F0_in + rho_modify[0];

    // ====================================================================
    // Step 4：碰撞 Collision（在計算點 A，使用 τ_A）
    // ====================================================================
    // 【注意事項 Step 4 碰撞】
    //   碰撞使用計算點 A 的 τ_A（非 departure point 的 τ_D），
    //   此為 GILBM 與標準 LBM 的關鍵差異。此步的 f_eq 使用 A 點宏觀量重新計算，
    //   確保 Σf_i = ρ 在碰撞步精確成立（離散守恆性）。
    // ====================================================================

    double rho_s = F0_in + F1_in + F2_in + F3_in + F4_in + F5_in + F6_in + F7_in + F8_in + F9_in
                 + F10_in + F11_in + F12_in + F13_in + F14_in + F15_in + F16_in + F17_in + F18_in;
    double u1 = (F1_in + F7_in + F9_in + F11_in + F13_in - (F2_in + F8_in + F10_in + F12_in + F14_in)) / rho_s;
    double v1 = (F3_in + F7_in + F8_in + F15_in + F17_in - (F4_in + F9_in + F10_in + F16_in + F18_in)) / rho_s;
    double w1 = (F5_in + F11_in + F12_in + F15_in + F16_in - (F6_in + F13_in + F14_in + F17_in + F18_in)) / rho_s;
    double udot = u1 * u1 + v1 * v1 + w1 * w1;

    const double F0_eq  = (1./3.)  * rho_s * (1.0 - 1.5 * udot);
    const double F1_eq  = (1./18.) * rho_s * (1.0 + 3.0 * u1 + 4.5 * u1 * u1 - 1.5 * udot);
    const double F2_eq  = (1./18.) * rho_s * (1.0 - 3.0 * u1 + 4.5 * u1 * u1 - 1.5 * udot);
    const double F3_eq  = (1./18.) * rho_s * (1.0 + 3.0 * v1 + 4.5 * v1 * v1 - 1.5 * udot);
    const double F4_eq  = (1./18.) * rho_s * (1.0 - 3.0 * v1 + 4.5 * v1 * v1 - 1.5 * udot);
    const double F5_eq  = (1./18.) * rho_s * (1.0 + 3.0 * w1 + 4.5 * w1 * w1 - 1.5 * udot);
    const double F6_eq  = (1./18.) * rho_s * (1.0 - 3.0 * w1 + 4.5 * w1 * w1 - 1.5 * udot);
    const double F7_eq  = (1./36.) * rho_s * (1.0 + 3.0 * (u1 + v1) + 4.5 * (u1 + v1) * (u1 + v1) - 1.5 * udot);
    const double F8_eq  = (1./36.) * rho_s * (1.0 + 3.0 * (-u1 + v1) + 4.5 * (-u1 + v1) * (-u1 + v1) - 1.5 * udot);
    const double F9_eq  = (1./36.) * rho_s * (1.0 + 3.0 * (u1 - v1) + 4.5 * (u1 - v1) * (u1 - v1) - 1.5 * udot);
    const double F10_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-u1 - v1) + 4.5 * (-u1 - v1) * (-u1 - v1) - 1.5 * udot);
    const double F11_eq = (1./36.) * rho_s * (1.0 + 3.0 * (u1 + w1) + 4.5 * (u1 + w1) * (u1 + w1) - 1.5 * udot);
    const double F12_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-u1 + w1) + 4.5 * (-u1 + w1) * (-u1 + w1) - 1.5 * udot);
    const double F13_eq = (1./36.) * rho_s * (1.0 + 3.0 * (u1 - w1) + 4.5 * (u1 - w1) * (u1 - w1) - 1.5 * udot);
    const double F14_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-u1 - w1) + 4.5 * (-u1 - w1) * (-u1 - w1) - 1.5 * udot);
    const double F15_eq = (1./36.) * rho_s * (1.0 + 3.0 * (v1 + w1) + 4.5 * (v1 + w1) * (v1 + w1) - 1.5 * udot);
    const double F16_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-v1 + w1) + 4.5 * (-v1 + w1) * (-v1 + w1) - 1.5 * udot);
    const double F17_eq = (1./36.) * rho_s * (1.0 + 3.0 * (v1 - w1) + 4.5 * (v1 - w1) * (v1 - w1) - 1.5 * udot);
    const double F18_eq = (1./36.) * rho_s * (1.0 + 3.0 * (-v1 - w1) + 4.5 * (-v1 - w1) * (-v1 - w1) - 1.5 * udot);

    m_matrix;
    meq;
    collision;

    __syncthreads();
    f0_new[index]  = F0_in;   f1_new[index]  = F1_in;   f2_new[index]  = F2_in;
    f3_new[index]  = F3_in;   f4_new[index]  = F4_in;   f5_new[index]  = F5_in;
    f6_new[index]  = F6_in;   f7_new[index]  = F7_in;   f8_new[index]  = F8_in;
    f9_new[index]  = F9_in;   f10_new[index] = F10_in;  f11_new[index] = F11_in;
    f12_new[index] = F12_in;  f13_new[index] = F13_in;  f14_new[index] = F14_in;
    f15_new[index] = F15_in;  f16_new[index] = F16_in;  f17_new[index] = F17_in;
    f18_new[index] = F18_in;
    u_out[index] = u1;  v_out[index] = v1;  w_out[index] = w1;  rho_out[index] = rho_s;
}

#endif
