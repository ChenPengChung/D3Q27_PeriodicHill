#ifndef MONITOR_FILE
#define MONITOR_FILE

// ================================================================
// GPU reduction kernel: Σ|rho - 1.0| over interior points (L1 density deviation)
// Same structure as ReduceRhoSum_Kernel in evolution.h
// ================================================================
__global__ void ReduceRhoL1_Kernel(double *rho_d, double *partial_sums_d) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

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
        val = fabs(rho_d[j * NX6 * NZ6 + k * NX6 + i] - 1.0);
    }

    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums_d[blockIdx.x] = sdata[0];
}

// ================================================================
// RS convergence check point
// ================================================================
// ERCOFTAC x/h=2.0 (streamwise, hill-rear shear layer, RS 峰值區)
// ERCOFTAC y/h=1.0 (wall-normal, above hill bottom)
// Code mapping: streamwise=y(j), wall-normal=z(k), spanwise=x(i)
int rs_check_rank = -1;
int rs_check_i = -1, rs_check_j = -1, rs_check_k = -1;

// 初始化代表點 index (在 GenerateMesh 之後呼叫)
void InitMonitorCheckPoint() {
    // Target: ERCOFTAC x/h=2.0 → code y=2.0, y/h=1.0 → code z = hill(2.0) + 1.0
    double y_target = 2.0 * (double)H_HILL;
    double z_above  = 1.0 * (double)H_HILL;

    // j_global from uniform y grid: dy = LY/NY
    double dy = (double)LY / (double)NY;
    int j_global = (int)round(y_target / dy) + 3;  // +3 for ghost/buffer
    if (j_global < 3)      j_global = 3;
    if (j_global > NY + 2) j_global = NY + 2;

    // Determine owning rank
    int stride = NYD6 - 7;
    rs_check_rank = (j_global - 3) / stride;
    if (rs_check_rank >= jp) rs_check_rank = jp - 1;
    int j_local = j_global - rs_check_rank * stride;

    if (myid == rs_check_rank) {
        rs_check_j = j_local;
        rs_check_i = NX6 / 2;  // mid-span

        // z_wall at this j (hill height at x/h=2.0)
        double z_wall = z_h[j_local * NZ6 + 3];
        double z_target = z_wall + z_above;

        // Find closest k to z_target
        double min_dist = 1e30;
        rs_check_k = 3;
        for (int k = 3; k < NZ6 - 3; k++) {
            double dist = fabs(z_h[j_local * NZ6 + k] - z_target);
            if (dist < min_dist) {
                min_dist = dist;
                rs_check_k = k;
            }
        }
    }

    // Broadcast check point info to all ranks (for MPI_Bcast root)
    MPI_Bcast(&rs_check_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Print check point info
    if (myid == rs_check_rank) {
        printf("[Monitor] RS check point: rank=%d  j=%d  k=%d  i=%d\n",
               myid, rs_check_j, rs_check_k, rs_check_i);
        printf("          y=%.4f (target=%.1f)  z=%.4f (target=hill+%.1f)  hill=%.4f\n",
               y_h[rs_check_j], y_target,
               z_h[rs_check_j * NZ6 + rs_check_k], z_above,
               z_h[rs_check_j * NZ6 + 3]);
    }

    // Print density monitoring point (hill crest: rank 0, j=3, k=4, mid-span)
    if (myid == 0) {
        printf("[Monitor] Density crest point: rank=0  j=3  k=4  i=%d\n", NX6 / 2);
        printf("          y=%.4f (hill crest)  z=%.4f (1st interior above wall)\n",
               y_h[3], z_h[3 * NZ6 + 4]);
    }

    // Write header comment to monitor file (9 columns)
    if (myid == 0) {
        FILE *f = fopen("Ustar_Force_record.dat", "a");
        fprintf(f, "# FTT\tUb/Uref\tForce\tMa_max\taccu_count\tuu_RS_check\tk_check\trho_crest\trho_L1\n");
        fclose(f);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// 計算全場最大 Ma 數 (所有 rank 參與, MPI_Allreduce MAX)
// 從 GPU 拷貝 u,v,w → 掃描內部計算點 → 取全域最大 |V| / cs
double ComputeMaMax(){
    double local_max_sq = 0.0;
    const int gs = NX6 * NYD6 * NZ6;
    double *u_h = (double*)malloc(gs * sizeof(double));
    double *v_h = (double*)malloc(gs * sizeof(double));
    double *w_h = (double*)malloc(gs * sizeof(double));
    CHECK_CUDA( cudaMemcpy(u_h, u, gs * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(v_h, v, gs * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(w_h, w, gs * sizeof(double), cudaMemcpyDeviceToHost) );

    for (int j = 3; j < NYD6-3; j++)
    for (int k = 3; k < NZ6-3; k++)
    for (int i = 3; i < NX6-3; i++) {
        int idx = j*NX6*NZ6 + k*NX6 + i;
        double sq = u_h[idx]*u_h[idx] + v_h[idx]*v_h[idx] + w_h[idx]*w_h[idx];
        if (sq > local_max_sq) local_max_sq = sq;
    }
    free(u_h); free(v_h); free(w_h);

    double local_max_mag = sqrt(local_max_sq);
    double global_max_mag;
    MPI_Allreduce(&local_max_mag, &global_max_mag, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max_mag / (double)cs;
}

void Launch_Monitor(){
    // --- 1. 計算瞬時 Ub (rank 0, j=3 hill-crest section) ---
    double Ub_inst = 0.0;
    if (myid == 0) {
        double *v_slice = (double*)malloc(NX6 * NZ6 * sizeof(double));
        CHECK_CUDA( cudaMemcpy(v_slice, &v[3*NX6*NZ6], NX6*NZ6*sizeof(double), cudaMemcpyDeviceToHost) );
        //輸出入口端瞬時，空間平均速度場
        // Bilinear cell-average: Σ v_cell × dx_cell × dz_cell / A_total
        for (int k = 3; k < NZ6-4; k++) {
        for (int i = 3; i < NX6-4; i++) {
            double v_cell = (v_slice[k*NX6+i] + v_slice[(k+1)*NX6+i]
                           + v_slice[k*NX6+i+1] + v_slice[(k+1)*NX6+i+1]) / 4.0;
            Ub_inst += v_cell * (x_h[i+1] - x_h[i]) * (z_h[3*NZ6+k+1] - z_h[3*NZ6+k]);
        }}
        Ub_inst /= (double)(LX * (LZ - 1.0));
        free(v_slice);
    }

    // --- 2. 計算全場 Ma_max (all ranks) ---
    double Ma_max = ComputeMaMax();

    // --- 3. RS 收斂檢查 (代表點單點值) ---
    double uu_RS_check = 0.0;
    double k_check_val = 0.0;

    if (accu_count > 0 && rs_check_rank >= 0 && (int)TBSWITCH) {
        double check_vals[2] = {0.0, 0.0};

        if (myid == rs_check_rank) {
            int idx = rs_check_j * NX6 * NZ6 + rs_check_k * NX6 + rs_check_i;
            double v_sum, vv_sum, u_sum, uu_sum, w_sum, ww_sum;

            // 從 GPU 複製 6 個單點累積值
            CHECK_CUDA( cudaMemcpy(&v_sum,  &V[idx],  sizeof(double), cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(&vv_sum, &VV[idx], sizeof(double), cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(&u_sum,  &U[idx],  sizeof(double), cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(&uu_sum, &UU[idx], sizeof(double), cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(&w_sum,  &W[idx],  sizeof(double), cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(&ww_sum, &WW[idx], sizeof(double), cudaMemcpyDeviceToHost) );

            double N = (double)accu_count;
            double inv_Uref2 = 1.0 / ((double)Uref * (double)Uref);

            // 流向 RS: <u'u'>/Ub² = (<v²>-<v>²)/Uref² (code v = streamwise)
            double v_mean = v_sum / N;
            double uu_phys = (vv_sum / N - v_mean * v_mean) * inv_Uref2;

            // 展向 RS: <w'w'>/Ub² = (<u²>-<u>²)/Uref² (code u = spanwise)
            double u_mean = u_sum / N;
            double vv_phys = (uu_sum / N - u_mean * u_mean) * inv_Uref2;

            // 法向 RS: <v'v'>/Ub² = (<w²>-<w>²)/Uref² (code w = wall-normal)
            double w_mean = w_sum / N;
            double ww_phys = (ww_sum / N - w_mean * w_mean) * inv_Uref2;

            check_vals[0] = uu_phys;                                     // uu_RS_check
            check_vals[1] = 0.5 * (uu_phys + vv_phys + ww_phys);        // k_check
        }

        MPI_Bcast(check_vals, 2, MPI_DOUBLE, rs_check_rank, MPI_COMM_WORLD);
        uu_RS_check = check_vals[0];
        k_check_val = check_vals[1];
    }

    // --- 4. 密度監測：hill crest 單點 + L1 norm ---
    // rho_crest: rank 0, j=3 (hill crest y=0), k=4 (1st interior), i=NX6/2
    double rho_crest = 1.0;
    if (myid == 0) {
        int crest_idx = 3 * NX6 * NZ6 + 4 * NX6 + NX6 / 2;
        CHECK_CUDA( cudaMemcpy(&rho_crest, &rho_d[crest_idx], sizeof(double), cudaMemcpyDeviceToHost) );
    }

    // rho_L1: Σ|rho_i - 1.0| over all interior points (GPU reduction per rank → MPI sum)
    double rho_L1 = 0.0;
    {
        const int rho_total_l1 = (NX6 - 7) * (NYD6 - 7) * (NZ6 - 6);
        const int rho_threads_l1 = 256;
        const int rho_blocks_l1 = (rho_total_l1 + rho_threads_l1 - 1) / rho_threads_l1;

        ReduceRhoL1_Kernel<<<rho_blocks_l1, rho_threads_l1, rho_threads_l1 * sizeof(double)>>>(rho_d, rho_partial_d);
        CHECK_CUDA( cudaMemcpy(rho_partial_h, rho_partial_d, rho_blocks_l1 * sizeof(double), cudaMemcpyDeviceToHost) );

        double rho_L1_local = 0.0;
        for (int b = 0; b < rho_blocks_l1; b++) rho_L1_local += rho_partial_h[b];

        MPI_Reduce(&rho_L1_local, &rho_L1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // --- 5. 輸出 Ustar_Force_record.dat (9 columns) ---
    double FTT = step * dt_global / (double)flow_through_time;
    double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);

    // 格式: FTT  Ub/Uref  Force  Ma_max  accu_count  uu_RS_check  k_check  rho_crest  rho_L1
    if (myid == 0) {
        FILE *fhist = fopen("Ustar_Force_record.dat", "a");
        fprintf(fhist, "%.6f\t%.10f\t%.10f\t%.6f\t%d\t%.6e\t%.6e\t%.10f\t%.6e\n",
                FTT, Ub_inst/(double)Uref, F_star, Ma_max,
                accu_count, uu_RS_check, k_check_val,
                rho_crest, rho_L1);
        fclose(fhist);
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif
