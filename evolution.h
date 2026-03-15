#ifndef EVOLUTION_FILE
#define EVOLUTION_FILE

#include "MRT_Matrix_D3Q27.h"
#include "gilbm/evolution_gilbm.h"
__global__ void periodicSW(
    double **f_new_d,         // device array of NQ pointers
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

    // Left periodic copy
    idx_buffer = j*NZ6*NX6 + k*NX6 + i;
    idx = idx_buffer + (NX6-2*buffer-1);

    for (int q = 0; q < NQ; q++)
        f_new_d[q][idx_buffer] = f_new_d[q][idx];
    u[idx_buffer] = u[idx];
    v[idx_buffer] = v[idx];
    w[idx_buffer] = w[idx];
    rho_d[idx_buffer] = rho_d[idx];
    for (int q = 0; q < NQ; q++)
        feq_d_arg[q * grid_size + idx_buffer] = feq_d_arg[q * grid_size + idx];

    // Right periodic copy
    idx_buffer = j*NX6*NZ6 + k*NX6 + (NX6-1-i);
    idx = idx_buffer - (NX6-2*buffer-1);

    for (int q = 0; q < NQ; q++)
        f_new_d[q][idx_buffer] = f_new_d[q][idx];
    u[idx_buffer] = u[idx];
    v[idx_buffer] = v[idx];
    w[idx_buffer] = w[idx];
    rho_d[idx_buffer] = rho_d[idx];
    for (int q = 0; q < NQ; q++)
        feq_d_arg[q * grid_size + idx_buffer] = feq_d_arg[q * grid_size + idx];

}


// [方案 C 已移除] Odd-Even Filter kernels removed — replaced by 方案 A (Matrix moment transform)
// 濾波器因降低有效 Re (Δν/ν = nσdt/6ν) 而放棄

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

void Launch_CollisionStreaming(double *f_old[NQ], double *f_new[NQ]) {
    dim3 griddimSW(  1,      NYD6/NT+1, NZ6);
    dim3 blockdimSW( 3, NT,        1 );

    dim3 griddim(  NX6/NT+1, NYD6, NZ6);
    dim3 blockdim( NT, 1, 1);

    // Upload pointer arrays to device (small: NQ*8 = 216 bytes each)
    CHECK_CUDA( cudaMemcpy(ft_ptrs_d, f_old, NQ * sizeof(double*), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(fd_ptrs_d, f_new, NQ * sizeof(double*), cudaMemcpyHostToDevice) );

    // [GTS] Single-pass: interpolation + collision in one kernel
    GILBM_GTS_Kernel<<<griddim, blockdim, 0, stream0>>>(
        ft_ptrs_d, fd_ptrs_d,
        feq_d,
        dk_dz_d, dk_dy_d,
        delta_zeta_d, bk_precomp_d,
        u, v, w, rho_d,
        Force_d, rho_modify_d,
        omega_global
    );
    CHECK_CUDA( cudaStreamSynchronize(stream0) );

    // MPI exchange: NQ f_new directions (y-direction halo)
    ISend_LtRtBdry( f_new, iToLeft,    l_nbr, itag_f4, 0, NQ,   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26  );
    IRecv_LtRtBdry( f_new, iFromRight, r_nbr, itag_f4, 1, NQ,   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26  );
    ISend_LtRtBdry( f_new, iToRight,   r_nbr, itag_f3, 2, NQ,   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26  );
    IRecv_LtRtBdry( f_new, iFromLeft,  l_nbr, itag_f3, 3, NQ,   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26  );
    for( int i = 0;  i < NQ; i++ ){
        CHECK_MPI( MPI_Waitall(4, request[i], status[i]) );
    }
    for( int i = NQ; i < NQ+4; i++ ){
        CHECK_MPI( MPI_Waitall(4, request[i], status[i]) );
    }

    // x-direction periodic BC (f_new + u/v/w/rho + feq_d)
    CHECK_CUDA( cudaMemcpy(fd_ptrs_d, f_new, NQ * sizeof(double*), cudaMemcpyHostToDevice) );
    periodicSW<<<griddimSW, blockdimSW, 0, stream0>>>(
        fd_ptrs_d,
        y_d, x_d, z_d, u, v, w, rho_d, feq_d
    );

    // [方案 C 已移除] Filter launch code removed
}

// ────────────────────────────────────────────────────────────────
// 舊版 P-additive + Gehrke 控制器已移除 (2026-03-15)
// 已替換為下方 PID + Gehrke hybrid controller (對齊 D3Q19 Edit3)
// ────────────────────────────────────────────────────────────────

void Launch_ModifyForcingTerm()
{
    // ====== Instantaneous Ub: zero → accumulate once → read ======
    const size_t nBytes = NX6 * NZ6 * sizeof(double);
    CHECK_CUDA( cudaMemset(Ub_avg_d, 0, nBytes) );

    dim3 griddim_Ubulk(NX6/NT+1, 1, NZ6);
    dim3 blockdim_Ubulk(NT, 1, 1);
    AccumulateUbulk<<<griddim_Ubulk, blockdim_Ubulk>>>(Ub_avg_d, v);
    CHECK_CUDA( cudaDeviceSynchronize() );

    CHECK_CUDA( cudaMemcpy(Ub_avg_h, Ub_avg_d, nBytes, cudaMemcpyDeviceToHost) );

    // Bilinear cell-average integration: Σ v_cell × dx_cell × dz_cell / A_total
    double Ub_avg = 0.0;
    for( int k = 3; k < NZ6-4; k++ ){
    for( int i = 3; i < NX6-4; i++ ){
        double v_cell = (Ub_avg_h[k*NX6+i] + Ub_avg_h[(k+1)*NX6+i]
                       + Ub_avg_h[k*NX6+i+1] + Ub_avg_h[(k+1)*NX6+i+1]) / 4.0;
        Ub_avg += v_cell * (x_h[i+1] - x_h[i]) * (z_h[3*NZ6+k+1] - z_h[3*NZ6+k]);
    }}
    Ub_avg /= (double)(LX*(LZ-1.0));

    // Only rank 0's j=3 (hill-crest inlet cross-section) is physically meaningful
    CHECK_MPI( MPI_Bcast(&Ub_avg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD) );
    Ub_avg_global = Ub_avg;

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    double Ma_now = Ub_avg / (double)cs;
    double Ma_max = ComputeMaMax();  // all ranks participate (MPI_Allreduce)

    // ====================================================================
    // Hybrid Dual-Stage Force Controller (PID + Gehrke multiplicative)
    // ====================================================================
    // Phase 1 (PID):    |Re%| > SWITCH_THRESHOLD — 冷啟動/遠離目標安全加速
    // Phase 2 (Gehrke): |Re%| ≤ SWITCH_THRESHOLD — 穩態乘法微調
    // Gehrke ref: Gehrke & Rung (2020) Int J Numer Meth Fluids, Sec 3.1
    //   原文: F *= (1 - 0.1 × Re%)  當 |Re%| > 1.5%, 每 FTT 更新 10 次
    // 連續 Mach brake 在兩模式之上統一適用
    // ====================================================================

    double error = (double)Uref - Ub_avg;  // 正 = 需加速, 負 = 需減速
    double Re_pct = (Ub_avg - (double)Uref) / (double)Uref * 100.0;
    const char *ctrl_mode;

    // ── 持久狀態 (跨 force update) ──
    static double Force_integral = 0.0;      // PID 積分項
    static double error_prev = 0.0;          // PID 微分項
    static bool   controller_initialized = false;
    static bool   gehrke_activated = false;   // Gehrke 模式旗標
    if (!controller_initialized) {
        Force_integral = 0.0;
        error_prev = error;
        controller_initialized = true;
    }

    // ── 控制器參數 (從 variables.h #define 讀取) ──
    double Kp = (double)FORCE_KP;
    double Ki = (double)FORCE_KI;
    double Kd = (double)FORCE_KD;
    double norm = (double)Uref * (double)Uref / (double)LY;

    // Poiseuille force 估計 (Gehrke floor + Force cap 用)
    double h_eff = (double)LZ - (double)H_HILL;
    double F_Poiseuille = 8.0 * (double)niu * (double)Uref / (h_eff * h_eff);
    double F_floor = (double)FORCE_GEHRKE_FLOOR * F_Poiseuille;
    double F_cap  = (double)FORCE_CAP_MULT * F_Poiseuille;  // Force 上限

    // ── 模式選擇 ──
    bool use_gehrke = (fabs(Re_pct) <= (double)FORCE_SWITCH_THRESHOLD);

    // Phase transition logging
    if (use_gehrke && !gehrke_activated) {
        gehrke_activated = true;
        if (myid == 0)
            printf("\n=== [Step %d | FTT=%.2f] Gehrke ACTIVATED (Re%%=%.2f%%) ===\n\n",
                   step, step * dt_global / (double)flow_through_time, Re_pct);
    } else if (!use_gehrke && gehrke_activated) {
        gehrke_activated = false;
        // ★ Gehrke → PID 回切: 同步積分項 = 當前 Force, 避免跳變
        Force_integral = fmax(0.0, Force_h[0]);
        if (myid == 0)
            printf("\n=== [Step %d | FTT=%.2f] Gehrke DEACTIVATED -> PID (Re%%=%.2f%%) ===\n\n",
                   step, step * dt_global / (double)flow_through_time, Re_pct);
    }

    if (use_gehrke) {
        // ============================================================
        // Phase 2: Gehrke 乘法控制器
        // F *= (1 - GEHRKE_GAIN × Re%)
        // Re% > 0 → Ub 太高 → correction < 1 → 減力
        // Re% < 0 → Ub 太低 → correction > 1 → 加力
        // ============================================================
        if (fabs(Re_pct) < (double)FORCE_GEHRKE_DEADZONE) {
            ctrl_mode = "GEHRKE-HOLD";
            // 死區: 不調整, 維持現有 Force
        } else {
            double correction = 1.0 - (double)FORCE_GEHRKE_GAIN * Re_pct;
            // 安全 clamp: SWITCH_THRESHOLD=5% 時理論極值 = [0.5, 1.5]
            // ★ 上界 1.5 而非 2.0: 防止 Re%=-5% 時每步 ×1.5 造成指數增長
            if (correction < 0.5) correction = 0.5;
            if (correction > 1.5) correction = 1.5;
            Force_h[0] *= correction;
            ctrl_mode = (Re_pct > 0) ? "GEHRKE-DEC" : "GEHRKE-INC";
        }

        // Gehrke floor: 防止 Force → 0 陷阱
        if (Force_h[0] < F_floor) {
            Force_h[0] = F_floor;
            if (myid == 0)
                printf("[GEHRKE-FLOOR] Force clamped to %.1f%% Poiseuille = %.5E\n",
                       (double)FORCE_GEHRKE_FLOOR * 100.0, F_floor);
        }

        // ★ 同步 PID 積分項: 追蹤 Gehrke 的 Force 值
        // 這樣如果切回 PID, 積分項 = Gehrke 最後設定的力, 無跳變
        Force_integral = Force_h[0];
        error_prev = error;  // 同步微分項基準

    } else {
        // ============================================================
        // Phase 1: PID 控制器 (冷啟動 / 遠離目標)
        // Force = Kp*error*norm + integral + Kd*d_error*norm
        // ============================================================

        // 微分項
        double d_error = error - error_prev;
        error_prev = error;

        // 積分項累加
        Force_integral += Ki * error * norm;

        // Conditional decay: overshoot 時快速衰減
        if (error < 0.0 && Force_integral > 0.0) {
            Force_integral *= 0.5;
        }

        // Anti-windup: integral ∈ [0, 10×norm]
        double Force_max = 10.0 * norm;
        if (Force_integral > Force_max) Force_integral = Force_max;
        if (Force_integral < 0.0) Force_integral = 0.0;

        // PID 合成
        Force_h[0] = Kp * error * norm + Force_integral + Kd * d_error * norm;

        // Back-calculation anti-windup: Force < 0 → clamp + 回算 integral
        if (Force_h[0] < 0.0) {
            Force_h[0] = 0.0;
            double integral_target = fmax(0.0, -Kp * error * norm);
            if (Force_integral > integral_target)
                Force_integral = integral_target;
        }

        ctrl_mode = (fabs(Re_pct) < 1.5) ? "PID-steady" :
                    (error > 0)           ? "PID-accel"  : "PID-decel";
    }

    // ====== Force Magnitude Cap (兩模式共用) ======
    // 防止任何模式下 Force 失控 (e.g., Gehrke 指數增長, PID windup 殘留)
    if (Force_h[0] > F_cap) {
        if (myid == 0)
            printf("[FORCE-CAP] Force=%.5E > cap=%.5E (%.0f×Poiseuille), clamped!\n",
                   Force_h[0], F_cap, (double)FORCE_CAP_MULT);
        Force_h[0] = F_cap;
        Force_integral = fmin(Force_integral, F_cap);  // 同步 integral
    }

    // ── Cold start ramp: 已關閉 (對齊 D3Q19 Edit3，FORCE_RAMP_STEPS=0) ──
    // 若需重新啟用，將 FORCE_RAMP_STEPS 設為 >0 即可
#if FORCE_RAMP_STEPS > 0
    if (step < FORCE_RAMP_STEPS) {
        double ramp = (double)(step + 1) / (double)FORCE_RAMP_STEPS;
        double F_ramp_limit = ramp * (double)FORCE_RAMP_CAP * F_Poiseuille;
        if (Force_h[0] > F_ramp_limit) {
            Force_h[0] = F_ramp_limit;
            ctrl_mode = "PID-RAMP";
        }
    }
#endif

    // ====== Continuous Mach Safety Brake (兩模式共用) ======
    // 閾值自動跟隨 Uref 縮放
    double Ma_bulk_ref  = (double)Uref / (double)cs;       // 目標 bulk Ma
    double Ma_threshold = (double)MA_BRAKE_MULT_THRESHOLD * Ma_bulk_ref;  // 連續二次衰減開始
    double Ma_critical  = (double)MA_BRAKE_MULT_CRITICAL  * Ma_bulk_ref;  // 緊急歸零

    // Ma 增長率偵測
    static double Ma_max_prev = 0.0;
    double Ma_growth_rate = 0.0;
    if (Ma_max_prev > 1e-10) {
        Ma_growth_rate = (Ma_max - Ma_max_prev) / Ma_max_prev;
    }
    Ma_max_prev = Ma_max;

    double Ma_factor = 1.0;

    // 連續二次衰減
    if (Ma_max > Ma_threshold && Ma_max <= Ma_critical) {
        double excess = (Ma_max - Ma_threshold) / (Ma_critical - Ma_threshold);
        Ma_factor = (1.0 - excess) * (1.0 - excess);
        if (myid == 0)
            printf("[Ma-BRAKE] Ma_max=%.4f > %.3f, factor=%.4f\n",
                   Ma_max, Ma_threshold, Ma_factor);
    }

    // 緊急歸零 + integral reset
    if (Ma_max > Ma_critical) {
        Ma_factor = 0.0;
        Force_integral = 0.0;
        if (myid == 0)
            printf("[CRITICAL] Ma_max=%.4f > %.3f, Force=0, integral reset!\n",
                   Ma_max, Ma_critical);
    }

    // 急速增長率煞車
    if (Ma_growth_rate > (double)MA_BRAKE_GROWTH_LIMIT && Ma_max > Ma_bulk_ref * 1.5) {
        Ma_factor *= 0.3;
        Force_integral *= 0.5;
        if (myid == 0)
            printf("[RATE-BRAKE] Ma growth=%.1f%%, extra brake applied\n",
                   Ma_growth_rate * 100.0);
    }

    Force_h[0] *= Ma_factor;
    Force_integral *= Ma_factor;

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    double FTT    = step * dt_global / (double)flow_through_time;
    double U_star = Ub_avg / (double)Uref;
    double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);
    double Re_now = Ub_avg / ((double)Uref / (double)Re);

    const char *status_tag = "";
    if (Ma_max > 0.35)       status_tag = " [WARNING: Ma_max>0.35, reduce Uref]";
    else if (U_star > 1.2)   status_tag = " [OVERSHOOT!]";
    else if (U_star > 1.05)  status_tag = " [OVERSHOOT]";

    if (myid == 0) {
        printf("[Step %d | FTT=%.2f] Ub=%.6f  U*=%.4f  Re%%=%.2f%%  Force=%.5E  F*=%.4f  Re=%.1f  Ma=%.4f  Ma_max=%.6e  [%s]%s\n",
               step, FTT, Ub_avg, U_star, Re_pct, Force_h[0], F_star, Re_now, Ma_now, Ma_max, ctrl_mode, status_tag);

        // Append to force_control.dat for post-processing
        FILE *flog = fopen("force_control.dat", "a");
        if (flog) {
            fprintf(flog, "%d\t%.4f\t%.6f\t%.6e\t%.4f\t%.1f\t%.6e\t%s\n",
                    step, FTT, Ub_avg, Force_h[0], U_star, Re_now, Ma_max, ctrl_mode);
            fclose(flog);
        }
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
