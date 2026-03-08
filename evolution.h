#ifndef EVOLUTION_FILE
#define EVOLUTION_FILE

#include "gilbm/evolution_gilbm.h"
#include "MRT_Process.h"
#include "MRT_Matrix.h"
__global__ void periodicSW(
    double *f0_old, double *f1_old, double *f2_old, double *f3_old, double *f4_old, double *f5_old, double *f6_old, double *f7_old, double *f8_old, double *f9_old, double *f10_old, double *f11_old, double *f12_old, double *f13_old, double *f14_old, double *f15_old, double *f16_old, double *f17_old, double *f18_old,
    double *f0_new, double *f1_new, double *f2_new, double *f3_new, double *f4_new, double *f5_new, double *f6_new, double *f7_new, double *f8_new, double *f9_new, double *f10_new, double *f11_new, double *f12_new, double *f13_new, double *f14_new, double *f15_new, double *f16_new, double *f17_new, double *f18_new,
    double *y_d,       double *x_d,      double *z_d,
    double *u,         double *v,        double *w,         double *rho_d,
    double *feq_d_arg)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
          int idx, idx_buffer;
          int buffer = 3;
    const int grid_size = NX6 * NYD6 * NZ6;

    if( j >= NYD6 || k >= NZ6 ) return;

    idx_buffer = j*NZ6*NX6 + k*NX6 + i;
    idx = idx_buffer + (NX6-2*buffer-1);

    f0_new[idx_buffer]  = f0_new[idx];    f1_new[idx_buffer]  = f1_new[idx];    f2_new[idx_buffer]  = f2_new[idx];
    f3_new[idx_buffer]  = f3_new[idx];    f4_new[idx_buffer]  = f4_new[idx];    f5_new[idx_buffer]  = f5_new[idx];
    f6_new[idx_buffer]  = f6_new[idx];    f7_new[idx_buffer]  = f7_new[idx];    f8_new[idx_buffer]  = f8_new[idx];
    f9_new[idx_buffer]  = f9_new[idx];    f10_new[idx_buffer] = f10_new[idx];   f11_new[idx_buffer] = f11_new[idx];
    f12_new[idx_buffer] = f12_new[idx];   f13_new[idx_buffer] = f13_new[idx];   f14_new[idx_buffer] = f14_new[idx];
    f15_new[idx_buffer] = f15_new[idx];   f16_new[idx_buffer] = f16_new[idx];   f17_new[idx_buffer] = f17_new[idx];
    f18_new[idx_buffer] = f18_new[idx];
    u[idx_buffer] = u[idx];
    v[idx_buffer] = v[idx];
    w[idx_buffer] = w[idx];
    rho_d[idx_buffer] = rho_d[idx];
    // feq_d periodic copy (19 planes)
    for (int q = 0; q < 19; q++)
        feq_d_arg[q * grid_size + idx_buffer] = feq_d_arg[q * grid_size + idx];

    idx_buffer = j*NX6*NZ6 + k*NX6 + (NX6-1-i);
    idx = idx_buffer - (NX6-2*buffer-1);

    f0_new[idx_buffer]  = f0_new[idx];    f1_new[idx_buffer]  = f1_new[idx];    f2_new[idx_buffer]  = f2_new[idx];
    f3_new[idx_buffer]  = f3_new[idx];    f4_new[idx_buffer]  = f4_new[idx];    f5_new[idx_buffer]  = f5_new[idx];
    f6_new[idx_buffer]  = f6_new[idx];    f7_new[idx_buffer]  = f7_new[idx];    f8_new[idx_buffer]  = f8_new[idx];
    f9_new[idx_buffer]  = f9_new[idx];    f10_new[idx_buffer] = f10_new[idx];   f11_new[idx_buffer] = f11_new[idx];
    f12_new[idx_buffer] = f12_new[idx];   f13_new[idx_buffer] = f13_new[idx];   f14_new[idx_buffer] = f14_new[idx];
    f15_new[idx_buffer] = f15_new[idx];   f16_new[idx_buffer] = f16_new[idx];   f17_new[idx_buffer] = f17_new[idx];
    f18_new[idx_buffer] = f18_new[idx];
    u[idx_buffer] = u[idx];
    v[idx_buffer] = v[idx];
    w[idx_buffer] = w[idx];
    rho_d[idx_buffer] = rho_d[idx];
    // feq_d periodic copy (19 planes)
    for (int q = 0; q < 19; q++)
        feq_d_arg[q * grid_size + idx_buffer] = feq_d_arg[q * grid_size + idx];

}


// ===== GPU reduction kernel: sum rho_d over interior points =====
// Maps 1D thread index to interior (i,j,k), writes partial block sums to partial_sums_d.
// Host sums the partial results (typically 256-512 doubles = 2-4 KB) instead of
// transferring the entire rho field (1.6 MB per rank) to CPU.
__global__ void ReduceRhoSum_Kernel(double *rho_d, double *partial_sums_d) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Interior dimensions: i∈[3,NX6-4), j∈[3,NYD6-4), k∈[3,NZ6-3)
    const int ni = NX6 - 7;
    const int nk = NZ6 - 6;
    const int nj = NYD6 - 7;
    const int total = ni * nj * nk;

    double val = 0.0;
    if (gid < total) {
        int j = gid / (ni * nk) + 3;
        int rem = gid % (ni * nk);
        int k = rem / ni + 3;
        int i = rem % ni + 3;
        val = rho_d[j * NX6 * NZ6 + k * NX6 + i];
    }

    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums_d[blockIdx.x] = sdata[0];
}

// ===== Time-average accumulation kernel (GPU-side, FTT-gated in main.cu) =====
// Accumulates all 3 velocity components: u(spanwise), v(streamwise), w(wall-normal)
__global__ void AccumulateTavg_Kernel(double *u_tavg, double *v_tavg, double *w_tavg,
                                      const double *u_src, const double *v_src, const double *w_src, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        u_tavg[idx] += u_src[idx];
        v_tavg[idx] += v_src[idx];
        w_tavg[idx] += w_src[idx];
    }
}

void Launch_AccumulateTavg() {
    const int N = NX6 * NYD6 * NZ6;
    const int block = 256;
    const int grid = (N + block - 1) / block;
    AccumulateTavg_Kernel<<<grid, block>>>(u_tavg_d, v_tavg_d, w_tavg_d, u, v, w, N);
}

// ===== Vorticity accumulation kernel (FTT >= FTT_STATS_START, same window as velocity mean) =====
// Curvilinear vorticity:
//   omega_x = dw/dy - dv/dz = (1/dy)*dw_dj + dk_dy*dw_dk - dk_dz*dv_dk
//   omega_y = du/dz - dw/dx = dk_dz*du_dk - (1/dx)*dw_di
//   omega_z = dv/dx - du/dy = (1/dx)*dv_di - (1/dy)*du_dj - dk_dy*du_dk
__global__ void AccumulateVorticity_Kernel(
    double *ox_tavg, double *oy_tavg, double *oz_tavg,
    const double *u_in, const double *v_in, const double *w_in,
    const double *dk_dz_in, const double *dk_dy_in,
    double dx_inv, double dy_inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 2 || i >= NX6-3 || j <= 2 || j >= NYD6-3 || k <= 3 || k >= NZ6-4) return;

    const int nface = NX6 * NZ6;
    const int index = j * nface + k * NX6 + i;

    // 2nd-order central differences in computational coordinates
    double du_dj = (u_in[index + nface] - u_in[index - nface]) * 0.5;
    double du_dk = (u_in[index + NX6]   - u_in[index - NX6])   * 0.5;

    double dv_di = (v_in[index + 1]     - v_in[index - 1])     * 0.5;
    double dv_dk = (v_in[index + NX6]   - v_in[index - NX6])   * 0.5;

    double dw_di = (w_in[index + 1]     - w_in[index - 1])     * 0.5;
    double dw_dj = (w_in[index + nface] - w_in[index - nface]) * 0.5;
    double dw_dk = (w_in[index + NX6]   - w_in[index - NX6])   * 0.5;

    double dkdz = dk_dz_in[j * NZ6 + k];
    double dkdy = dk_dy_in[j * NZ6 + k];

    ox_tavg[index] += dy_inv * dw_dj + dkdy * dw_dk - dkdz * dv_dk;
    oy_tavg[index] += dkdz * du_dk - dx_inv * dw_di;
    oz_tavg[index] += dx_inv * dv_di - dy_inv * du_dj - dkdy * du_dk;
}

void Launch_AccumulateVorticity() {
    dim3 grid(NX6/NT+1, NYD6, NZ6);
    dim3 block(NT, 1, 1);
    double dx_inv = (double)(NX6 - 7) / (double)LX;
    double dy_inv = (double)(NY6 - 7) / (double)LY;
    AccumulateVorticity_Kernel<<<grid, block>>>(
        ox_tavg_d, oy_tavg_d, oz_tavg_d,
        u, v, w, dk_dz_d, dk_dy_d, dx_inv, dy_inv);
}

__global__ void AccumulateUbulk(double *Ub_avg, double *v)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + 3;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if( i <= 2 || i >= NX6-3 || k <= 2 || k >= NZ6-3 ) return;

    // Store pure velocity — area weighting done on host with correct 2D z_h
    Ub_avg[k*NX6+i] = v[j*NZ6*NX6+k*NX6+i];
}

void Launch_CollisionStreaming(double *f_old[19], double *f_new[19]) {
    dim3 griddimSW(  1,      NYD6/NT+1, NZ6);
    dim3 blockdimSW( 3, NT,        1 );

    dim3 griddim(  NX6/NT+1, NYD6, NZ6);
    dim3 blockdim( NT, 1, 1);

    // [GTS] Single-pass: interpolation + collision in one kernel
    // Double-buffer: reads f_old, writes f_new (no race condition)
    GILBM_GTS_Kernel<<<griddim, blockdim, 0, stream0>>>(
        f_old[0], f_old[1], f_old[2], f_old[3], f_old[4], f_old[5], f_old[6],
        f_old[7], f_old[8], f_old[9], f_old[10], f_old[11], f_old[12],
        f_old[13], f_old[14], f_old[15], f_old[16], f_old[17], f_old[18],
        f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6],
        f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12],
        f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
        feq_d,
        dk_dz_d, dk_dy_d,
        delta_zeta_d, bk_precomp_d,
        u, v, w, rho_d,
        Force_d, rho_modify_d,
        omega_global
    );
    CHECK_CUDA( cudaStreamSynchronize(stream0) );

    // MPI exchange: 19 f_new directions (y-direction halo)
    ISend_LtRtBdry( f_new, iToLeft,    l_nbr, itag_f4, 0, 19,   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18  );
    IRecv_LtRtBdry( f_new, iFromRight, r_nbr, itag_f4, 1, 19,   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18  );
    ISend_LtRtBdry( f_new, iToRight,   r_nbr, itag_f3, 2, 19,   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18  );
    IRecv_LtRtBdry( f_new, iFromLeft,  l_nbr, itag_f3, 3, 19,   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18  );
    for( int i = 0;  i < 19; i++ ){
        CHECK_MPI( MPI_Waitall(4, request[i], status[i]) );
    }
    for( int i = 19; i < 23; i++ ){
        CHECK_MPI( MPI_Waitall(4, request[i], status[i]) );
    }

    // x-direction periodic BC (f_new + u/v/w/rho + feq_d)
    periodicSW<<<griddimSW, blockdimSW, 0, stream0>>>(
        f_old[0] ,f_old[1] ,f_old[2] ,f_old[3] ,f_old[4] ,f_old[5] ,f_old[6] ,f_old[7] ,f_old[8] ,f_old[9] ,f_old[10] ,f_old[11] ,f_old[12] ,f_old[13] ,f_old[14] ,f_old[15] ,f_old[16] ,f_old[17] ,f_old[18],
        f_new[0] ,f_new[1] ,f_new[2] ,f_new[3] ,f_new[4] ,f_new[5] ,f_new[6] ,f_new[7] ,f_new[8] ,f_new[9] ,f_new[10] ,f_new[11] ,f_new[12] ,f_new[13] ,f_new[14] ,f_new[15] ,f_new[16] ,f_new[17] ,f_new[18],
        y_d, x_d, z_d, u, v, w, rho_d, feq_d
    );
}

void Launch_ModifyForcingTerm()
{
    // ====== Instantaneous Ub: zero → accumulate once → read ======
    const size_t nBytes = NX6 * NZ6 * sizeof(double);
    CHECK_CUDA( cudaMemset(Ub_avg_d, 0, nBytes) );   // always clean before single-shot

    dim3 griddim_Ubulk(NX6/NT+1, 1, NZ6);
    dim3 blockdim_Ubulk(NT, 1, 1);
    AccumulateUbulk<<<griddim_Ubulk, blockdim_Ubulk>>>(Ub_avg_d, v);
    CHECK_CUDA( cudaDeviceSynchronize() );

    CHECK_CUDA( cudaMemcpy(Ub_avg_h, Ub_avg_d, nBytes, cudaMemcpyDeviceToHost) );

    // Bilinear cell-average integration: Σ v_cell × dx_cell × dz_cell / A_total
    // v_cell = 4-point average; area from host arrays x_h (1D), z_h (2D at j=3)
    double Ub_avg = 0.0;
    for( int k = 3; k < NZ6-4; k++ ){
    for( int i = 3; i < NX6-4; i++ ){
        double v_cell = (Ub_avg_h[k*NX6+i] + Ub_avg_h[(k+1)*NX6+i]
                       + Ub_avg_h[k*NX6+i+1] + Ub_avg_h[(k+1)*NX6+i+1]) / 4.0;
        Ub_avg += v_cell * (x_h[i+1] - x_h[i]) * (z_h[3*NZ6+k+1] - z_h[3*NZ6+k]);
    }}
    Ub_avg /= (double)(LX*(LZ-1.0));

    // ★ 只有 rank 0 的 j=3 = 山丘頂入口截面，具有物理意義
    CHECK_MPI( MPI_Bcast(&Ub_avg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD) );
    Ub_avg_global = Ub_avg;

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    double Ma_now = Ub_avg / (double)cs;
    double Ma_max = ComputeMaMax();  // all ranks participate (MPI_Allreduce)
    double FTT    = step * dt_global / (double)flow_through_time;

    // ====== 雙階段外力控制器 (Gehrke & Rung 2022) ======
    // Re% = (Ub - Uref) / Uref × 100
    double Re_pct = (Ub_avg - (double)Uref) / (double)Uref * 100.0;
    bool use_gehrke = (fabs(Re_pct) <= FORCE_SWITCH_THRESHOLD) && (FTT >= FTT_GEHRKE_FORCE);
    const char *ctrl_mode;

    if (use_gehrke) {
        // Phase 2: Gehrke multiplicative controller
        // Dead zone: |Re%| < 1.5% → no adjustment
        if (fabs(Re_pct) < 1.5) {
            ctrl_mode = "Gehrke-HOLD";
        } else {
            // Multiplicative correction: F *= (1 + 0.01 * Re%)
            // Re% < 0 → Ub too low → increase Force
            // Re% > 0 → Ub too high → decrease Force
            double correction = 1.0 - 0.01 * Re_pct;
            Force_h[0] *= correction;
            ctrl_mode = "Gehrke-MULT";
        }
    } else {
        // Phase 1: P-additive controller (cold start / far from target)
        double beta  = max(0.001, force_alpha / (double)Re);
        double error = (double)Uref - Ub_avg;
        Force_h[0] += beta * error * (double)Uref / (double)LY;
        ctrl_mode = "P-additive";
    }

    // Force 非負 clamp
    if (Force_h[0] < 0.0) {
        Force_h[0] = 0.0;
    }

    // Ma 安全檢查 (使用 local Ma_max — LBM 穩定性取決於局部最大 Ma, 非 bulk 平均)
    if (Ma_max > 0.35) {
        Force_h[0] *= 0.05;
        if (myid == 0)
            printf("[CRITICAL] Ma_max=%.4f > 0.35, Force reduced to 5%%: %.5E\n", Ma_max, Force_h[0]);
    } else if (Ma_max > 0.3) {
        Force_h[0] *= 0.5;
        if (myid == 0)
            printf("[WARNING] Ma_max=%.4f > 0.3, Force halved to %.5E\n", Ma_max, Force_h[0]);
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    double U_star = Ub_avg / (double)Uref;
    double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);
    double Re_now = Ub_avg / ((double)Uref / (double)Re);

    const char *status_tag = "";
    if (Ma_max > 0.35)       status_tag = " [WARNING: Ma_max>0.35, reduce Uref]";
    else if (U_star > 1.2)   status_tag = " [OVERSHOOT!]";
    else if (U_star > 1.05)  status_tag = " [OVERSHOOT]";

    if (myid == 0) {
        printf("[Step %d | FTT=%.2f] Ub=%.6f  U*=%.4f  Re%%=%.2f%%  Force=%.5E  F*=%.4f  Re=%.1f  Ma=%.4f  Ma_max=%.4f  [%s]%s\n",
               step, FTT, Ub_avg, U_star, Re_pct, Force_h[0], F_star, Re_now, Ma_now, Ma_max, ctrl_mode, status_tag);
    }

    if (Ma_max > 0.35 && myid == 0) {
        printf("  >>> BGK stability limit: Ma < 0.3. Current Ma_max=%.4f at hill crest.\n", Ma_max);
        printf("  >>> Recommended: reduce Uref to %.4f (target Ma_max<0.25)\n", (double)Uref * 0.25 / Ma_max);
    }

    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif
