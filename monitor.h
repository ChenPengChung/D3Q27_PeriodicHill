#ifndef MONITOR_FILE
#define MONITOR_FILE

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

    // Write header comment to monitor file
    if (myid == 0) {
        FILE *f = fopen("Ustar_Force_record.dat", "a");
        fprintf(f, "# FTT\tUb/Uref\tForce\tMa_max\taccu_count\tuu_RS_check\tk_check\n");
        fclose(f);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// 計算全場最大 Ma 數 (所有 rank 參與, MPI_Allreduce MAX)
// 從 GPU 拷貝 u,v,w → 掃描內部計算點 → 取全域最大 |V| / cs
// 增加位置追蹤: 輸出 Ma_max 發生的 (i,j,k) 座標和 rank
double ComputeMaMax(){
    double local_max_sq = 0.0;
    int loc_i = -1, loc_j = -1, loc_k = -1;  // Ma_max location tracking
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
        if (sq > local_max_sq) {
            local_max_sq = sq;
            loc_i = i; loc_j = j; loc_k = k;
        }
    }
    free(u_h); free(v_h); free(w_h);

    double local_max_mag = sqrt(local_max_sq);
    double global_max_mag;
    MPI_Allreduce(&local_max_mag, &global_max_mag, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Determine which rank has the global max and print location
    // Use MPI_MAXLOC to find rank with maximum
    struct { double val; int rank; } local_vr, global_vr;
    local_vr.val = local_max_mag;
    local_vr.rank = myid;
    MPI_Allreduce(&local_vr, &global_vr, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    // The rank that owns the max prints its location (j is local to that rank)
    if (myid == global_vr.rank && global_max_mag / (double)cs > 0.05) {
        printf("  [Ma_max loc] rank=%d, i=%d, j_local=%d, k=%d, |V|=%.6f, Ma=%.4f\n",
               myid, loc_i, loc_j, loc_k, global_max_mag, global_max_mag / (double)cs);
    }

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

    // --- 4. 輸出 Ustar_Force_record.dat ---
    double FTT = step * dt_global / (double)flow_through_time;
    double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);

    // 格式: FTT  Ub/Uref  Force  Ma_max  accu_count  uu_RS_check  k_check
    if (myid == 0) {
        FILE *fhist = fopen("Ustar_Force_record.dat", "a");
        fprintf(fhist, "%.6f\t%.10f\t%.10f\t%.6f\t%d\t%.6e\t%.6e\n",
                FTT, Ub_inst/(double)Uref, F_star, Ma_max,
                accu_count, uu_RS_check, k_check_val);
        fclose(fhist);
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif
