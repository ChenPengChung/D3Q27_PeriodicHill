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

// AccumulateTavg removed — velocity sums now handled by MeanVars (shared accu_count)

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

#if USE_P6
    // ===== P6 flow: Step1 → sync → MPI → periodicSW → Step23 =====

    // Kernel A: Step 1 + 1.5 (all interior j=3..NYD6-4)
    // Pure Jacobi: reads f_pc (private per point) → writes f_new, feq_d, u/v/w/rho.
    // No pre-copy needed — f_pc is self-contained; f_new is fully overwritten.
    GILBM_Step1_Kernel<<<griddim, blockdim, 0, stream0>>>(
        f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6],
        f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12],
        f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
        f_pc_d, feq_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        delta_zeta_d, bk_precomp_d,
        u, v, w, rho_d, rho_modify_d
    );
    CHECK_CUDA( cudaStreamSynchronize(stream0) );

    // MPI exchange: all 19 f_new directions (y-direction halo)
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

    // Kernel B: Step 2+3 (all interior j=3..NYD6-4)
    // Ghost zone f_new valid after MPI + periodicSW; feq at ghost computed on-the-fly.
    GILBM_Step23_Full_Kernel<<<griddim, blockdim, 0, stream0>>>(
        f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6],
        f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12],
        f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
        f_pc_d, feq_d, omegadt_local_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        bk_precomp_d, Force_d
    );

#else
    // ===== Old flow (P5 only, no P6 kernel split): pre-copy → Buffer → Full → MPI → periodicSW → Correction =====

    // 1. Double-buffer pre-copy: f_new ← f_old (Step 2+3 reads stencil from f_new)
    const size_t grid_bytes = NX6 * NYD6 * NZ6 * sizeof(double);
    for (int q = 0; q < 19; q++)
        CHECK_CUDA( cudaMemcpy(f_new[q], f_old[q], grid_bytes, cudaMemcpyDeviceToDevice) );

    // 2. Buffer kernel: MPI boundary rows (j=3..6 and j=NYD6-7..NYD6-4)
    //    Must run BEFORE Full kernel to avoid race condition on overlapping j-rows.
    dim3 griddim_buf(NX6/NT+1, 1, NZ6);
    dim3 blockdim_buf(NT, 4, 1);
    GILBM_StreamCollide_Buffer_Kernel<<<griddim_buf, blockdim_buf, 0, stream0>>>(
        f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6],
        f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12],
        f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
        f_pc_d, feq_d, omegadt_local_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        delta_zeta_d, bk_precomp_d,
        u, v, w, rho_d, Force_d, rho_modify_d, 3
    );
    GILBM_StreamCollide_Buffer_Kernel<<<griddim_buf, blockdim_buf, 0, stream0>>>(
        f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6],
        f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12],
        f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
        f_pc_d, feq_d, omegadt_local_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        delta_zeta_d, bk_precomp_d,
        u, v, w, rho_d, Force_d, rho_modify_d, NYD6-7
    );

    // 3. Full kernel: interior rows (j=7..NYD6-8, guarded inside kernel)
    GILBM_StreamCollide_Kernel<<<griddim, blockdim, 0, stream0>>>(
        f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6],
        f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12],
        f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
        f_pc_d, feq_d, omegadt_local_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        delta_zeta_d, bk_precomp_d,
        u, v, w, rho_d, Force_d, rho_modify_d
    );
    CHECK_CUDA( cudaStreamSynchronize(stream0) );

    // 4. MPI exchange: all 19 f_new directions (y-direction halo)
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

    // 5. x-direction periodic BC (f_new + u/v/w/rho + feq_d)
    periodicSW<<<griddimSW, blockdimSW, 0, stream0>>>(
        f_old[0] ,f_old[1] ,f_old[2] ,f_old[3] ,f_old[4] ,f_old[5] ,f_old[6] ,f_old[7] ,f_old[8] ,f_old[9] ,f_old[10] ,f_old[11] ,f_old[12] ,f_old[13] ,f_old[14] ,f_old[15] ,f_old[16] ,f_old[17] ,f_old[18],
        f_new[0] ,f_new[1] ,f_new[2] ,f_new[3] ,f_new[4] ,f_new[5] ,f_new[6] ,f_new[7] ,f_new[8] ,f_new[9] ,f_new[10] ,f_new[11] ,f_new[12] ,f_new[13] ,f_new[14] ,f_new[15] ,f_new[16] ,f_new[17] ,f_new[18],
        y_d, x_d, z_d, u, v, w, rho_d, feq_d
    );

    // 6. Correction kernel: re-run Step 2+3 for 6 MPI boundary rows (j=3..5, j=NYD6-6..NYD6-4)
    //    These rows had stale ghost zone data during the initial Buffer pass.
    dim3 griddim_corr(NX6/NT+1, 1, NZ6);
    dim3 blockdim_corr(NT, 3, 1);
    GILBM_Correction_Kernel<<<griddim_corr, blockdim_corr, 0, stream0>>>(
        f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6],
        f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12],
        f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
        f_pc_d, feq_d, omegadt_local_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        bk_precomp_d, Force_d, 3
    );
    GILBM_Correction_Kernel<<<griddim_corr, blockdim_corr, 0, stream0>>>(
        f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6],
        f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12],
        f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
        f_pc_d, feq_d, omegadt_local_d,
        dk_dz_d, dk_dy_d,
        dt_local_d, omega_local_d,
        bk_precomp_d, Force_d, NYD6-6
    );

#endif
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

    // EMA smoothing: filter vortex shedding noise from Ub before feeding to controller
    // α=0.3 → effective window ~3 samples (3000 steps), DC gain = 1 (no steady-state bias)
    // All ranks execute identically (same Ub_avg after Bcast) → Ub_ema synchronized
    static double Ub_ema = -1.0;           // sentinel: uninitialized
    const double ema_alpha = 0.3;
    if (Ub_ema < 0.0)
        Ub_ema = Ub_avg;                   // first call: seed with raw measurement
    else
        Ub_ema = ema_alpha * Ub_avg + (1.0 - ema_alpha) * Ub_ema;

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    double beta = max(0.001, force_alpha/(double)Re);
    double Ma_now = Ub_avg / (double)cs;
    double Ma_max = ComputeMaMax();  // all ranks participate (MPI_Allreduce)

    double error = (double)Uref - Ub_ema;  // ← use EMA-filtered Ub for controller
    double gain = beta;
    Force_h[0] = Force_h[0] + gain * error * (double)Uref / (double)LY;

    // Force 非負 clamp
    if (Force_h[0] < 0.0) {
        Force_h[0] = 0.0;
    }

    // Ma 安全檢查 (使用 local Ma_max — LBM 穩定性取決於局部最大 Ma, 非 bulk 平均)
    if (Ma_max > 0.35) {
        Force_h[0] *= 0.05;
        if (myid == 0)
            printf("[CRITICAL] Ma_max=%.4f > 0.35, Force reduced to 5%%: %.5E\n", Ma_max, Force_h[0]);
    } else if (Ma_max > 0.27) {
        Force_h[0] *= 0.5;
        if (myid == 0)
            printf("[WARNING] Ma_max=%.4f > 0.27, Force halved to %.5E\n", Ma_max, Force_h[0]);
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    double FTT    = step * dt_global / (double)flow_through_time;
    double U_star_ema = Ub_ema / (double)Uref;   // controller sees this //Ub_ema為濾波過後的涮間速度平均
    double U_star_raw = Ub_avg / (double)Uref;   // instantaneous measurement
    double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);
    double Re_now = Ub_ema / ((double)Uref / (double)Re);

    const char *status_tag = "";
    if (Ma_max > 0.35)          status_tag = " [WARNING: Ma_max>0.35, reduce Uref]";
    else if (U_star_ema > 1.2)  status_tag = " [OVERSHOOT!]";
    else if (U_star_ema > 1.05) status_tag = " [OVERSHOOT]";

    if (myid == 0) {
        printf("[Step %d | FTT=%.2f] Ub_inst=%.6f  U*=%.4f(ema=%.4f)  Force=%.5E  F*=%.4f  Re=%.1f  Ma=%.4f  Ma_max=%.4f%s\n",
               step, FTT, Ub_avg, U_star_raw, U_star_ema, Force_h[0], F_star, Re_now, Ma_now, Ma_max, status_tag);
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
