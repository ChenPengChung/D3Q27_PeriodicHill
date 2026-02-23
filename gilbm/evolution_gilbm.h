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
__constant__ double GILBM_tau;

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
// Helper: compute zeta offset for boundary-adaptive stencil
// ============================================================================
__device__ __forceinline__ int compute_zeta_offset(int k) {
    const int kmin = 2;
    const int kmax = NZ6 - 3;
    if (k - 3 < kmin)      return k - kmin;        // 0,1,2 near bottom
    else if (k + 3 > kmax) return 6 - (kmax - k);  // 4,5,6 near top
    else                    return 3;               // standard centered
}

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
    const double dt_A    = dt_local_d[idx_jk];//A點的local time step 
    const double tau_A   = tau_local_d[idx_jk];//A點的真實鬆弛時間 relaxation time 
    const double omegadt_A = omega_dt_d[index];

    // LTS acceleration factor for eta/xi displacement scaling
    const double a_local = dt_A / GILBM_dt;

    // Stencil base with boundary clamping
    int bi, bj, bk;
    compute_stencil_base(i, j, k, bi, bj, bk);

    // A's position within stencil
    const int ci = i - bi;
    const int cj = j - bj;
    const int ck = k - bk;

    // Wall BC pre-computation
    bool is_bottom = (k == 2);
    bool is_top    = (k == NZ6 - 3);
    double dk_dy_val = dk_dy_d[idx_jk];
    double dk_dz_val = dk_dz_d[idx_jk];

    double rho_wall = 0.0, du_x_dk = 0.0, du_y_dk = 0.0, du_z_dk = 0.0;
    if (is_bottom) {
        int idx3 = j * nface + 3 * NX6 + i;
        int idx4 = j * nface + 4 * NX6 + i;
        double rho3, ux3, uy3, uz3, rho4, ux4, uy4, uz4;
        compute_macroscopic_at(f_new_ptrs, idx3, rho3, ux3, uy3, uz3);
        compute_macroscopic_at(f_new_ptrs, idx4, rho4, ux4, uy4, uz4);
        du_x_dk = (4.0 * ux3 - ux4) / 2.0;
        du_y_dk = (4.0 * uy3 - uy4) / 2.0;
        du_z_dk = (4.0 * uz3 - uz4) / 2.0;
        rho_wall = rho3;
    } else if (is_top) {
        int idxm1 = j * nface + (NZ6 - 4) * NX6 + i;
        int idxm2 = j * nface + (NZ6 - 5) * NX6 + i;
        double rhom1, uxm1, uym1, uzm1, rhom2, uxm2, uym2, uzm2;
        compute_macroscopic_at(f_new_ptrs, idxm1, rhom1, uxm1, uym1, uzm1);
        compute_macroscopic_at(f_new_ptrs, idxm2, rhom2, uxm2, uym2, uzm2);
        du_x_dk = -(4.0 * uxm1 - uxm2) / 2.0;
        du_y_dk = -(4.0 * uym1 - uym2) / 2.0;
        du_z_dk = -(4.0 * uzm1 - uzm2) / 2.0;
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
            int center_flat = ci * 49 + cj * 7 + ck;
            f_streamed = f_pc[(q * STENCIL_VOL + center_flat) * GRID_SIZE + index];
        } else {
            bool need_bc = false;
            if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
            else if (is_top) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, false);

            if (need_bc) {
                f_streamed = ChapmanEnskogBC(q, rho_wall,
                    du_x_dk, du_y_dk, du_z_dk,
                    dk_dy_val, dk_dz_val, omega_A, dt_A);
            } else {
                // Load 343 values from f_pc into local stencil
                double f_stencil[STENCIL_SIZE][STENCIL_SIZE][STENCIL_SIZE];
                for (int si = 0; si < 7; si++)
                    for (int sj = 0; sj < 7; sj++)
                        for (int sk = 0; sk < 7; sk++) {
                            int flat = si * 49 + sj * 7 + sk;
                            f_stencil[si][sj][sk] = f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index];
                        }

                // Departure point
                double delta_i_val   = a_local * GILBM_delta_eta[q];
                double delta_xi_val  = a_local * GILBM_delta_xi[q];
                double delta_zeta_val = delta_zeta_d[q * NYD6 * NZ6 + idx_jk];

                double up_i = (double)i - delta_i_val;
                double up_j = (double)j - delta_xi_val;
                double up_k = (double)k - delta_zeta_val;

                if (up_i < 1.0)               up_i = 1.0;
                if (up_i > (double)(NX6 - 3))  up_i = (double)(NX6 - 3);
                if (up_j < 1.0)               up_j = 1.0;
                if (up_j > (double)(NYD6 - 3)) up_j = (double)(NYD6 - 3);
                if (up_k < 2.0)               up_k = 2.0;
                if (up_k > (double)(NZ6 - 5))  up_k = (double)(NZ6 - 5);

                // Lagrange weights relative to stencil base
                double t_i = up_i - (double)bi;
                double t_j = up_j - (double)bj;
                double t_k = up_k - (double)bk;

                double Lxi[7], Leta[7], Lzeta[7];
                lagrange_7point_coeffs(t_i, Lxi);
                lagrange_7point_coeffs(t_j, Leta);
                lagrange_7point_coeffs(t_k, Lzeta);

                // Tensor-product interpolation
                // Step A: xi reduction -> val_ez[7][7]
                double val_ez[7][7];
                for (int sj = 0; sj < 7; sj++)
                    for (int sk = 0; sk < 7; sk++)
                        val_ez[sj][sk] = Intrpl7(
                            f_stencil[0][sj][sk], Lxi[0],
                            f_stencil[1][sj][sk], Lxi[1],
                            f_stencil[2][sj][sk], Lxi[2],
                            f_stencil[3][sj][sk], Lxi[3],
                            f_stencil[4][sj][sk], Lxi[4],
                            f_stencil[5][sj][sk], Lxi[5],
                            f_stencil[6][sj][sk], Lxi[6]);

                // Step B: eta reduction -> val_z[7]
                double val_z[7];
                for (int sk = 0; sk < 7; sk++)
                    val_z[sk] = Intrpl7(
                        val_ez[0][sk], Leta[0],
                        val_ez[1][sk], Leta[1],
                        val_ez[2][sk], Leta[2],
                        val_ez[3][sk], Leta[3],
                        val_ez[4][sk], Leta[4],
                        val_ez[5][sk], Leta[5],
                        val_ez[6][sk], Leta[6]);

                // Step C: zeta reduction -> scalar
                f_streamed = Intrpl7(
                    val_z[0], Lzeta[0],
                    val_z[1], Lzeta[1],
                    val_z[2], Lzeta[2],
                    val_z[3], Lzeta[3],
                    val_z[4], Lzeta[4],
                    val_z[5], Lzeta[5],
                    val_z[6], Lzeta[6]);
            }
        }

        // Write post-streaming to f_new (this IS streaming)
        f_new_ptrs[q][index] = f_streamed;

        // Accumulate macroscopic
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

    double rho_A = rho_stream;
    double ux_A  = mx_stream / rho_A;
    double uy_A  = my_stream / rho_A;
    double uz_A  = mz_stream / rho_A;

    // Write feq to persistent global array
    for (int q = 0; q < 19; q++) {
        feq_d[q * GRID_SIZE + index] = compute_feq_alpha(q, rho_A, ux_A, uy_A, uz_A);
    }

    // Write macroscopic output
    rho_out_arr[index] = rho_A;
    u_out[index] = ux_A;
    v_out[index] = uy_A;
    w_out[index] = uz_A;

    // ==================================================================
    // STEPS 2+3: Re-estimation (Eq.35) + Collision (Eq.36) per q
    //   Eq.35: f̃_B = feq_B + (f_B - feq_B) * R_AB
    //   Eq.36: f̂_B = f̃_B - (1/ω_A)(f̃_B - feq_B)
    //        = f̃_B + ω_A * (feq_B - f̃_B)
    //   ω_A shared across all 343 nodes; feq_B is per-node B.
    // ==================================================================
    for (int q = 0; q < 19; q++) {
        // Skip BC directions: f_pc not needed, f_new already has BC value
        bool need_bc = false;
        if (is_bottom) need_bc = NeedsBoundaryCondition(q, dk_dy_val, dk_dz_val, true);
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
                        double rho_B, ux_B, uy_B, uz_B;
                        compute_macroscopic_at(f_new_ptrs, idx_B,
                                               rho_B, ux_B, uy_B, uz_B);
                        feq_B = compute_feq_alpha(q, rho_B, ux_B, uy_B, uz_B);
                    } else {
                        feq_B = feq_d[q * GRID_SIZE + idx_B];
                    }

                    // Read omega_dt at B
                    double omegadt_B = omega_dt_d[idx_B];
                    double R_AB = omegadt_A*dt_A / omegadt_B*dt_B;

                    // Eq.35: Re-estimation
                    double f_re = feq_B + (f_B - feq_B) * R_AB;

                    // Eq.36: Collision with ω_A and feq_B
                    f_re += 1/(omegadt_A) * (feq_B - f_re);

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
    double ux = (f[1]+f[7]+f[9]+f[11]+f[13] - (f[2]+f[8]+f[10]+f[12]+f[14])) / rho;
    double uy = (f[3]+f[7]+f[8]+f[15]+f[17] - (f[4]+f[9]+f[10]+f[16]+f[18])) / rho;
    double uz = (f[5]+f[11]+f[12]+f[15]+f[16] - (f[6]+f[13]+f[14]+f[17]+f[18])) / rho;

    for (int q = 0; q < 19; q++) {
        feq_d[q * GRID_SIZE + index] = compute_feq_alpha(q, rho, ux, uy, uz);
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

    // Imamura convention: ω = τ (dimensionless relaxation TIME, in denominator of collision)
    // omega_dt = ω × Δt = τ × Δt
    // R_35 = omegadt_A / omegadt_B = (τ_A·Δt_A)/(τ_B·Δt_B)
    // Combined Eq.35+36 gives: (τ_A−1)·Δt_A/(τ_B·Δt_B) ✓
    omega_dt_d[index] = tau_local_d[idx_jk] * dt_local_d[idx_jk];
}

#endif
