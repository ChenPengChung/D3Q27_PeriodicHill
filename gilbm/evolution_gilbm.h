#ifndef GILBM_EVOLUTION_H
#define GILBM_EVOLUTION_H

// Imamura GILBM evolution kernel — 4-step procedure
//
// For each grid point A and velocity direction alpha:
//
//   Step 1+2 (combined): Interpolation + Streaming
//       7-point, 6th-order Lagrange interpolation of f_old at departure point
//       → produces f_streamed(A, alpha) (used for macroscopic output)
//
//   Step 3: Re-estimation with local τ_B
//       For 7×7×7 stencil nodes B around A:
//         f_re[si][sj][sk] = f(B) + (1/τ_B)(feq_B - f(B))
//       Each node B uses its OWN τ_B at B's position
//
//   Step 4: Collision with point A's τ_A
//       For all 343 entries:
//         f_re[si][sj][sk] += (1/τ_A)(feq_A - f_re[si][sj][sk])
//       feq_A uses macroscopic at A from f_old
//
//   Output: f_new(A, alpha) = f_re at A's position in stencil (post-collision)
//           Macroscopic output from f_streamed (post-streaming)
//
// Wall BC: Chapman-Enskog (Imamura 2005 Eq. A.9) for no-slip walls

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

constexpr int STENCIL_SIZE = 7;
constexpr int LAGRANGE_ORDER = 6;   // polynomial degree = STENCIL_SIZE - 1

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

    // Distribution function variables (post-streaming, pre-output)
    double F0_in, F1_in, F2_in, F3_in, F4_in, F5_in, F6_in, F7_in, F8_in, F9_in;
    double F10_in, F11_in, F12_in, F13_in, F14_in, F15_in, F16_in, F17_in, F18_in;

    // Pointers array for interpolation access
    double *f_old_ptrs[19] = {
        f0_old, f1_old, f2_old, f3_old, f4_old, f5_old, f6_old,
        f7_old, f8_old, f9_old, f10_old, f11_old, f12_old,
        f13_old, f14_old, f15_old, f16_old, f17_old, f18_old
    };

    // ====================================================================
    // Shared stencil: centered at arrival point (i,j,k), used by ALL alpha
    // Boundary-adaptive clamping in zeta (like ISLBM cell_z)
    // ====================================================================
    int bi = i - 3;
    int bj = j - 3;
    int bk = k - 3;
    if (bi < 0)              bi = 0;
    if (bi + 6 >= NX6)       bi = NX6 - STENCIL_SIZE;
    if (bj < 0)              bj = 0;
    if (bj + 6 >= NYD6)      bj = NYD6 - STENCIL_SIZE;
    if (bk < 2)              bk = 2;
    if (bk + 6 > NZ6 - 3)    bk = NZ6 - 9;

    // Pre-compute macroscopic (rho, ux, uy, uz) at ALL 343 stencil nodes ONCE
    // Shared by all 18 velocity directions -> ~10x fewer global memory reads
    double rho_s[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
    double ux_s[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
    double uy_s[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
    double uz_s[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
    double inv_tau_s[STENCIL_SIZE][STENCIL_SIZE];  // tau varies only in (j,k)

    for (int sn = 0; sn < STENCIL_SIZE; sn++) {
        int kB = bk + sn;
        for (int sm = 0; sm < STENCIL_SIZE; sm++) {
            int jB = bj + sm;
            inv_tau_s[sm][sn] = 1.0 / tau_local_d[jB * NZ6 + kB];

            for (int sl = 0; sl < STENCIL_SIZE; sl++) {
                int iB = bi + sl;
                int idx_B = jB * NZ6 * NX6 + kB * NX6 + iB;

                double rho_B = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                for (int a = 0; a < 19; a++) {
                    double fa = f_old_ptrs[a][idx_B];
                    rho_B += fa;
                    mx += GILBM_e[a][0] * fa;
                    my += GILBM_e[a][1] * fa;
                    mz += GILBM_e[a][2] * fa;
                }
                rho_s[sl][sm][sn] = rho_B;
                ux_s[sl][sm][sn] = mx / rho_B;
                uy_s[sl][sm][sn] = my / rho_B;
                uz_s[sl][sm][sn] = mz / rho_B;
            }
        }
    }

    // A's position within the shared stencil (for extracting f_re at center)
    const int ci = i - bi;  // always 3 in interior
    const int cj = j - bj;  // always 3 in interior
    const int ck = k - bk;  // varies near walls (0..6)

    // Wall BC pre-computation
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

    // LTS acceleration factor for η/ξ displacement scaling
    const double a_local = dt / GILBM_dt;  // ≥ 1 (local dt / global dt)

    // ====================================================================
    // Imamura GILBM 4-Step Algorithm
    // ====================================================================
    // All alpha share ONE 7×7×7 stencil centered at arrival (i,j,k).
    // Macroscopic (rho, ux, uy, uz) pre-computed ONCE at 343 nodes above.
    //
    // Per-alpha steps:
    //   Step 1+2: Read f_old into f_re, interpolate at departure → f_streamed
    //   Step 3:   Re-estimation: f_re += (1/τ_B)(feq_B - f_re) at each node B
    //   Step 4:   Collision:     f_re += (1/τ_A)(feq_A - f_re) using τ_A at point A
    //   Output:   F_in_arr = f_re[ci][cj][ck] (post-collision at A's center)
    //             Accumulate f_streamed for macroscopic output
    //
    // BC: e_zeta = e_y*dk_dy + e_z*dk_dz
    //   bottom (k=2): e_zeta > 0 → Chapman-Enskog BC
    //   top (k=NZ6-3): e_zeta < 0 → Chapman-Enskog BC
    // ====================================================================

    double F_in_arr[19];
    double rho_stream = 0.0, mx_stream = 0.0, my_stream = 0.0, mz_stream = 0.0;

    for (int alpha = 0; alpha < 19; alpha++) {
        double f_streamed;

        if (alpha == 0) {
            // ── Rest direction: no streaming (Imamura Step 1+2) ──
            f_streamed = f0_old[index];

            // ── Steps 3+4 at center only (single point, B=A) ──
            double feq_A0 = compute_feq_alpha(0,
                rho_s[ci][cj][ck], ux_s[ci][cj][ck],
                uy_s[ci][cj][ck], uz_s[ci][cj][ck]);
            // Step 3: re-estimation with τ_B (= τ_A since B=A)
            double f_re_A = f_streamed + inv_tau_s[cj][ck] * (feq_A0 - f_streamed);
            // Step 4: collision with τ_A
            f_re_A += omega * (feq_A0 - f_re_A);
            F_in_arr[0] = f_re_A;

        } else {
            bool need_bc = false;
            if (is_bottom) {
                need_bc = NeedsBoundaryCondition(alpha, dk_dy_val, dk_dz_val, true);
            } else if (is_top) {
                need_bc = NeedsBoundaryCondition(alpha, dk_dy_val, dk_dz_val, false);
            }

            if (need_bc) {
                // Chapman-Enskog BC overrides normal procedure
                f_streamed = ChapmanEnskogBC(alpha, rho_wall,
                    du_x_dk, du_y_dk, du_z_dk,
                    dk_dy_val, dk_dz_val, omega, dt);
                F_in_arr[alpha] = f_streamed;

            } else {
                // ── Imamura Step 1+2: Read f_old into f_re + interpolate at departure ──
                double f_re[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
                for (int sn = 0; sn < STENCIL_SIZE; sn++) {
                    int kB = bk + sn;
                    for (int sm = 0; sm < STENCIL_SIZE; sm++) {
                        int jB = bj + sm;
                        for (int sl = 0; sl < STENCIL_SIZE; sl++) {
                            int iB = bi + sl;
                            int idx_B = jB * NZ6 * NX6 + kB * NX6 + iB;
                            f_re[sl][sm][sn] = f_old_ptrs[alpha][idx_B];
                        }
                    }
                }

                // Departure point (same as before)
                double delta_i_val = a_local * GILBM_delta_eta[alpha];
                double delta_xi_val = a_local * GILBM_delta_xi[alpha];
                double delta_zeta_val = delta_zeta_d[alpha * NYD6 * NZ6 + idx_jk];

                double up_i = (double)i - delta_i_val;
                double up_j = (double)j - delta_xi_val;
                double up_k = (double)k - delta_zeta_val;

                if (up_i < 1.0) up_i = 1.0;
                if (up_i > (double)(NX6 - 3)) up_i = (double)(NX6 - 3);
                if (up_j < 1.0) up_j = 1.0;
                if (up_j > (double)(NYD6 - 3)) up_j = (double)(NYD6 - 3);
                if (up_k < 2.0) up_k = 2.0;
                if (up_k > (double)(NZ6 - 5)) up_k = (double)(NZ6 - 5);

                // Lagrange weights relative to shared stencil base
                double t_i = up_i - (double)bi;
                double t_j = up_j - (double)bj;
                double t_k = up_k - (double)bk;

                double Lxi[STENCIL_SIZE], Leta[STENCIL_SIZE], Lzeta[STENCIL_SIZE];
                lagrange_7point_coeffs(t_i, Lxi);
                lagrange_7point_coeffs(t_j, Leta);
                lagrange_7point_coeffs(t_k, Lzeta);

                // Tensor-product interpolation of f_re (= f_old) at departure
                double val_xi[STENCIL_SIZE], val_eta[STENCIL_SIZE];
                for (int sn = 0; sn < STENCIL_SIZE; sn++) {
                    for (int sm = 0; sm < STENCIL_SIZE; sm++) {
                        val_xi[sm] = Intrpl7(
                            f_re[0][sm][sn], Lxi[0], f_re[1][sm][sn], Lxi[1],
                            f_re[2][sm][sn], Lxi[2], f_re[3][sm][sn], Lxi[3],
                            f_re[4][sm][sn], Lxi[4], f_re[5][sm][sn], Lxi[5],
                            f_re[6][sm][sn], Lxi[6]);
                    }
                    val_eta[sn] = Intrpl7(
                        val_xi[0], Leta[0], val_xi[1], Leta[1],
                        val_xi[2], Leta[2], val_xi[3], Leta[3],
                        val_xi[4], Leta[4], val_xi[5], Leta[5],
                        val_xi[6], Leta[6]);
                }
                f_streamed = Intrpl7(
                    val_eta[0], Lzeta[0], val_eta[1], Lzeta[1],
                    val_eta[2], Lzeta[2], val_eta[3], Lzeta[3],
                    val_eta[4], Lzeta[4], val_eta[5], Lzeta[5],
                    val_eta[6], Lzeta[6]);

                // ── Imamura Step 3: Re-estimation with τ_B at each stencil node ──
                for (int sn = 0; sn < STENCIL_SIZE; sn++) {
                    for (int sm = 0; sm < STENCIL_SIZE; sm++) {
                        double inv_tau_B = inv_tau_s[sm][sn];
                        for (int sl = 0; sl < STENCIL_SIZE; sl++) {
                            double feq_B = compute_feq_alpha(alpha,
                                rho_s[sl][sm][sn], ux_s[sl][sm][sn],
                                uy_s[sl][sm][sn], uz_s[sl][sm][sn]);
                            f_re[sl][sm][sn] += inv_tau_B * (feq_B - f_re[sl][sm][sn]);
                        }
                    }
                }

                // ── Imamura Step 4: Collision with τ_A at point A ──
                double feq_A = compute_feq_alpha(alpha,
                    rho_s[ci][cj][ck], ux_s[ci][cj][ck],
                    uy_s[ci][cj][ck], uz_s[ci][cj][ck]);
                for (int sn = 0; sn < STENCIL_SIZE; sn++) {
                    for (int sm = 0; sm < STENCIL_SIZE; sm++) {
                        for (int sl = 0; sl < STENCIL_SIZE; sl++) {
                            f_re[sl][sm][sn] += omega * (feq_A - f_re[sl][sm][sn]);
                        }
                    }
                }

                // f_new at A = f_re at A's center position in stencil
                F_in_arr[alpha] = f_re[ci][cj][ck];
            }
        }

        // Accumulate macroscopic from f_streamed (post-streaming, for output)
        rho_stream += f_streamed;
        mx_stream += GILBM_e[alpha][0] * f_streamed;
        my_stream += GILBM_e[alpha][1] * f_streamed;
        mz_stream += GILBM_e[alpha][2] * f_streamed;
    }

    // ====================================================================
    // Imamura 4-step complete:
    //   F_in_arr[alpha] = post-collision f_re at A (Steps 3+4) → written to f_new
    //   rho_stream/mx/my/mz = post-streaming macroscopic → written to output
    // ====================================================================

    // Global mass correction
    F_in_arr[0] += rho_modify[0];
    rho_stream += rho_modify[0];

    // Macroscopic at A from post-streaming f_streamed
    double rho_A = rho_stream;
    double u1 = mx_stream / rho_A;
    double v1 = my_stream / rho_A;
    double w1 = mz_stream / rho_A;

    // Assign to named variables
    F0_in = F_in_arr[0];   F1_in = F_in_arr[1];   F2_in = F_in_arr[2];
    F3_in = F_in_arr[3];   F4_in = F_in_arr[4];   F5_in = F_in_arr[5];
    F6_in = F_in_arr[6];   F7_in = F_in_arr[7];   F8_in = F_in_arr[8];
    F9_in = F_in_arr[9];   F10_in = F_in_arr[10];  F11_in = F_in_arr[11];
    F12_in = F_in_arr[12];  F13_in = F_in_arr[13];  F14_in = F_in_arr[14];
    F15_in = F_in_arr[15];  F16_in = F_in_arr[16];  F17_in = F_in_arr[17];
    F18_in = F_in_arr[18];

    // Write f_new (post-collision from Steps 3+4) and macroscopic output (post-streaming)
    __syncthreads();
    f0_new[index]  = F0_in;   f1_new[index]  = F1_in;   f2_new[index]  = F2_in;
    f3_new[index]  = F3_in;   f4_new[index]  = F4_in;   f5_new[index]  = F5_in;
    f6_new[index]  = F6_in;   f7_new[index]  = F7_in;   f8_new[index]  = F8_in;
    f9_new[index]  = F9_in;   f10_new[index] = F10_in;  f11_new[index] = F11_in;
    f12_new[index] = F12_in;  f13_new[index] = F13_in;  f14_new[index] = F14_in;
    f15_new[index] = F15_in;  f16_new[index] = F16_in;  f17_new[index] = F17_in;
    f18_new[index] = F18_in;
    u_out[index] = u1;  v_out[index] = v1;  w_out[index] = w1;  rho_out[index] = rho_A;
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

    double *f_old_ptrs[19] = {
        f0_old, f1_old, f2_old, f3_old, f4_old, f5_old, f6_old,
        f7_old, f8_old, f9_old, f10_old, f11_old, f12_old,
        f13_old, f14_old, f15_old, f16_old, f17_old, f18_old
    };

    // ====================================================================
    // Shared stencil: centered at arrival point (i,j,k), used by ALL alpha
    // Boundary-adaptive clamping in zeta (like ISLBM cell_z)
    // ====================================================================
    int bi = i - 3;
    int bj = j - 3;
    int bk = k - 3;
    if (bi < 0)              bi = 0;
    if (bi + 6 >= NX6)       bi = NX6 - STENCIL_SIZE;
    if (bj < 0)              bj = 0;
    if (bj + 6 >= NYD6)      bj = NYD6 - STENCIL_SIZE;
    if (bk < 2)              bk = 2;
    if (bk + 6 > NZ6 - 3)    bk = NZ6 - 9;

    // Pre-compute macroscopic (rho, ux, uy, uz) at ALL 343 stencil nodes ONCE
    // Shared by all 18 velocity directions -> ~10x fewer global memory reads
    double rho_s[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
    double ux_s[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
    double uy_s[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
    double uz_s[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
    double inv_tau_s[STENCIL_SIZE][STENCIL_SIZE];  // tau varies only in (j,k)

    for (int sn = 0; sn < STENCIL_SIZE; sn++) {
        int kB = bk + sn;
        for (int sm = 0; sm < STENCIL_SIZE; sm++) {
            int jB = bj + sm;
            inv_tau_s[sm][sn] = 1.0 / tau_local_d[jB * NZ6 + kB];

            for (int sl = 0; sl < STENCIL_SIZE; sl++) {
                int iB = bi + sl;
                int idx_B = jB * NZ6 * NX6 + kB * NX6 + iB;

                double rho_B = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                for (int a = 0; a < 19; a++) {
                    double fa = f_old_ptrs[a][idx_B];
                    rho_B += fa;
                    mx += GILBM_e[a][0] * fa;
                    my += GILBM_e[a][1] * fa;
                    mz += GILBM_e[a][2] * fa;
                }
                rho_s[sl][sm][sn] = rho_B;
                ux_s[sl][sm][sn] = mx / rho_B;
                uy_s[sl][sm][sn] = my / rho_B;
                uz_s[sl][sm][sn] = mz / rho_B;
            }
        }
    }

    // A's position within the shared stencil (for extracting f_re at center)
    const int ci = i - bi;  // always 3 in interior
    const int cj = j - bj;  // always 3 in interior
    const int ck = k - bk;  // varies near walls (0..6)

    // Wall BC pre-computation
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

    // LTS acceleration factor for η/ξ displacement scaling
    const double a_local = dt / GILBM_dt;  // ≥ 1 (local dt / global dt)

    // ====================================================================
    // Imamura GILBM 4-Step Algorithm
    // ====================================================================
    // All alpha share ONE 7×7×7 stencil centered at arrival (i,j,k).
    // Macroscopic (rho, ux, uy, uz) pre-computed ONCE at 343 nodes above.
    //
    // Per-alpha steps:
    //   Step 1+2: Read f_old into f_re, interpolate at departure → f_streamed
    //   Step 3:   Re-estimation: f_re += (1/τ_B)(feq_B - f_re) at each node B
    //   Step 4:   Collision:     f_re += (1/τ_A)(feq_A - f_re) using τ_A at point A
    //   Output:   F_in_arr = f_re[ci][cj][ck] (post-collision at A's center)
    //             Accumulate f_streamed for macroscopic output
    //
    // BC: e_zeta = e_y*dk_dy + e_z*dk_dz
    //   bottom (k=2): e_zeta > 0 → Chapman-Enskog BC
    //   top (k=NZ6-3): e_zeta < 0 → Chapman-Enskog BC
    // ====================================================================

    double F_in_arr[19];
    double rho_stream = 0.0, mx_stream = 0.0, my_stream = 0.0, mz_stream = 0.0;

    for (int alpha = 0; alpha < 19; alpha++) {
        double f_streamed;

        if (alpha == 0) {
            // ── Rest direction: no streaming (Imamura Step 1+2) ──
            f_streamed = f0_old[index];

            // ── Steps 3+4 at center only (single point, B=A) ──
            double feq_A0 = compute_feq_alpha(0,
                rho_s[ci][cj][ck], ux_s[ci][cj][ck],
                uy_s[ci][cj][ck], uz_s[ci][cj][ck]);
            // Step 3: re-estimation with τ_B (= τ_A since B=A)
            double f_re_A = f_streamed + inv_tau_s[cj][ck] * (feq_A0 - f_streamed);
            // Step 4: collision with τ_A
            f_re_A += omega * (feq_A0 - f_re_A);
            F_in_arr[0] = f_re_A;

        } else {
            bool need_bc = false;
            if (is_bottom) {
                need_bc = NeedsBoundaryCondition(alpha, dk_dy_val, dk_dz_val, true);
            } else if (is_top) {
                need_bc = NeedsBoundaryCondition(alpha, dk_dy_val, dk_dz_val, false);
            }

            if (need_bc) {
                // Chapman-Enskog BC overrides normal procedure
                f_streamed = ChapmanEnskogBC(alpha, rho_wall,
                    du_x_dk, du_y_dk, du_z_dk,
                    dk_dy_val, dk_dz_val, omega, dt);
                F_in_arr[alpha] = f_streamed;

            } else {
                // ── Imamura Step 1+2: Read f_old into f_re + interpolate at departure ──
                double f_re[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
                for (int sn = 0; sn < STENCIL_SIZE; sn++) {
                    int kB = bk + sn;
                    for (int sm = 0; sm < STENCIL_SIZE; sm++) {
                        int jB = bj + sm;
                        for (int sl = 0; sl < STENCIL_SIZE; sl++) {
                            int iB = bi + sl;
                            int idx_B = jB * NZ6 * NX6 + kB * NX6 + iB;
                            f_re[sl][sm][sn] = f_old_ptrs[alpha][idx_B];
                        }
                    }
                }

                // Departure point (same as before)
                double delta_i_val = a_local * GILBM_delta_eta[alpha];
                double delta_xi_val = a_local * GILBM_delta_xi[alpha];
                double delta_zeta_val = delta_zeta_d[alpha * NYD6 * NZ6 + idx_jk];

                double up_i = (double)i - delta_i_val;
                double up_j = (double)j - delta_xi_val;
                double up_k = (double)k - delta_zeta_val;

                if (up_i < 1.0) up_i = 1.0;
                if (up_i > (double)(NX6 - 3)) up_i = (double)(NX6 - 3);
                if (up_j < 1.0) up_j = 1.0;
                if (up_j > (double)(NYD6 - 3)) up_j = (double)(NYD6 - 3);
                if (up_k < 2.0) up_k = 2.0;
                if (up_k > (double)(NZ6 - 5)) up_k = (double)(NZ6 - 5);

                // Lagrange weights relative to shared stencil base
                double t_i = up_i - (double)bi;
                double t_j = up_j - (double)bj;
                double t_k = up_k - (double)bk;

                double Lxi[STENCIL_SIZE], Leta[STENCIL_SIZE], Lzeta[STENCIL_SIZE];
                lagrange_7point_coeffs(t_i, Lxi);
                lagrange_7point_coeffs(t_j, Leta);
                lagrange_7point_coeffs(t_k, Lzeta);

                // Tensor-product interpolation of f_re (= f_old) at departure
                double val_xi[STENCIL_SIZE], val_eta[STENCIL_SIZE];
                for (int sn = 0; sn < STENCIL_SIZE; sn++) {
                    for (int sm = 0; sm < STENCIL_SIZE; sm++) {
                        val_xi[sm] = Intrpl7(
                            f_re[0][sm][sn], Lxi[0], f_re[1][sm][sn], Lxi[1],
                            f_re[2][sm][sn], Lxi[2], f_re[3][sm][sn], Lxi[3],
                            f_re[4][sm][sn], Lxi[4], f_re[5][sm][sn], Lxi[5],
                            f_re[6][sm][sn], Lxi[6]);
                    }
                    val_eta[sn] = Intrpl7(
                        val_xi[0], Leta[0], val_xi[1], Leta[1],
                        val_xi[2], Leta[2], val_xi[3], Leta[3],
                        val_xi[4], Leta[4], val_xi[5], Leta[5],
                        val_xi[6], Leta[6]);
                }
                f_streamed = Intrpl7(
                    val_eta[0], Lzeta[0], val_eta[1], Lzeta[1],
                    val_eta[2], Lzeta[2], val_eta[3], Lzeta[3],
                    val_eta[4], Lzeta[4], val_eta[5], Lzeta[5],
                    val_eta[6], Lzeta[6]);

                // ── Imamura Step 3: Re-estimation with τ_B at each stencil node ──
                for (int sn = 0; sn < STENCIL_SIZE; sn++) {
                    for (int sm = 0; sm < STENCIL_SIZE; sm++) {
                        double inv_tau_B = inv_tau_s[sm][sn];
                        for (int sl = 0; sl < STENCIL_SIZE; sl++) {
                            double feq_B = compute_feq_alpha(alpha,
                                rho_s[sl][sm][sn], ux_s[sl][sm][sn],
                                uy_s[sl][sm][sn], uz_s[sl][sm][sn]);
                            f_re[sl][sm][sn] += inv_tau_B * (feq_B - f_re[sl][sm][sn]);
                        }
                    }
                }

                // ── Imamura Step 4: Collision with τ_A at point A ──
                double feq_A = compute_feq_alpha(alpha,
                    rho_s[ci][cj][ck], ux_s[ci][cj][ck],
                    uy_s[ci][cj][ck], uz_s[ci][cj][ck]);
                for (int sn = 0; sn < STENCIL_SIZE; sn++) {
                    for (int sm = 0; sm < STENCIL_SIZE; sm++) {
                        for (int sl = 0; sl < STENCIL_SIZE; sl++) {
                            f_re[sl][sm][sn] += omega * (feq_A - f_re[sl][sm][sn]);
                        }
                    }
                }

                // f_new at A = f_re at A's center position in stencil
                F_in_arr[alpha] = f_re[ci][cj][ck];
            }
        }

        // Accumulate macroscopic from f_streamed (post-streaming, for output)
        rho_stream += f_streamed;
        mx_stream += GILBM_e[alpha][0] * f_streamed;
        my_stream += GILBM_e[alpha][1] * f_streamed;
        mz_stream += GILBM_e[alpha][2] * f_streamed;
    }

    // ====================================================================
    // Imamura 4-step complete:
    //   F_in_arr[alpha] = post-collision f_re at A (Steps 3+4) → written to f_new
    //   rho_stream/mx/my/mz = post-streaming macroscopic → written to output
    // ====================================================================

    // Global mass correction
    F_in_arr[0] += rho_modify[0];
    rho_stream += rho_modify[0];

    // Macroscopic at A from post-streaming f_streamed
    double rho_A = rho_stream;
    double u1 = mx_stream / rho_A;
    double v1 = my_stream / rho_A;
    double w1 = mz_stream / rho_A;

    // Assign to named variables
    F0_in = F_in_arr[0];   F1_in = F_in_arr[1];   F2_in = F_in_arr[2];
    F3_in = F_in_arr[3];   F4_in = F_in_arr[4];   F5_in = F_in_arr[5];
    F6_in = F_in_arr[6];   F7_in = F_in_arr[7];   F8_in = F_in_arr[8];
    F9_in = F_in_arr[9];   F10_in = F_in_arr[10];  F11_in = F_in_arr[11];
    F12_in = F_in_arr[12];  F13_in = F_in_arr[13];  F14_in = F_in_arr[14];
    F15_in = F_in_arr[15];  F16_in = F_in_arr[16];  F17_in = F_in_arr[17];
    F18_in = F_in_arr[18];

    // Write f_new (post-collision from Steps 3+4) and macroscopic output (post-streaming)
    __syncthreads();
    f0_new[index]  = F0_in;   f1_new[index]  = F1_in;   f2_new[index]  = F2_in;
    f3_new[index]  = F3_in;   f4_new[index]  = F4_in;   f5_new[index]  = F5_in;
    f6_new[index]  = F6_in;   f7_new[index]  = F7_in;   f8_new[index]  = F8_in;
    f9_new[index]  = F9_in;   f10_new[index] = F10_in;  f11_new[index] = F11_in;
    f12_new[index] = F12_in;  f13_new[index] = F13_in;  f14_new[index] = F14_in;
    f15_new[index] = F15_in;  f16_new[index] = F16_in;  f17_new[index] = F17_in;
    f18_new[index] = F18_in;
    u_out[index] = u1;  v_out[index] = v1;  w_out[index] = w1;  rho_out[index] = rho_A;
}

#endif
