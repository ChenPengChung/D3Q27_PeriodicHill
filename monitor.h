#ifndef MONITOR_FILE
#define MONITOR_FILE

// 計算全場最大 Ma 數 (所有 rank 參與, MPI_Allreduce MAXLOC)
// 從 GPU 拷貝 u,v,w → 掃描內部計算點 → 取全域最大 |V| / cs
// 同時追蹤最大 Ma 點的位置 (rank, i, j, k) 和速度分量
double ComputeMaMax(){
    double local_max_sq = 0.0;
    int loc_i = 0, loc_j = 0, loc_k = 0;
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

    double local_max_mag = sqrt(local_max_sq);

    // MPI_MAXLOC: find global max AND which rank owns it
    struct { double val; int rank; } local_pair, global_pair;
    local_pair.val = local_max_mag;
    local_pair.rank = myid;
    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    double Ma = global_pair.val / (double)cs;

    // When Ma is high, print diagnostic location (from the rank that owns the max)
    if (Ma > 0.20) {
        // Broadcast location + velocity from max-owning rank
        int info[3] = {loc_i, loc_j, loc_k};
        double vel[3];
        if (myid == global_pair.rank) {
            int idx = loc_j*NX6*NZ6 + loc_k*NX6 + loc_i;
            vel[0] = u_h[idx]; vel[1] = v_h[idx]; vel[2] = w_h[idx];
        }
        MPI_Bcast(info, 3, MPI_INT, global_pair.rank, MPI_COMM_WORLD);
        MPI_Bcast(vel,  3, MPI_DOUBLE, global_pair.rank, MPI_COMM_WORLD);

        int j_global = global_pair.rank * (NYD6 - 7) + info[1];
        if (myid == 0) {
            printf("[Ma_diag] Ma=%.4f at rank=%d i=%d j_local=%d(j_global=%d) k=%d  "
                   "u=%.6f v=%.6f w=%.6f\n",
                   Ma, global_pair.rank, info[0], info[1], j_global, info[2],
                   vel[0], vel[1], vel[2]);
        }
    }

    free(u_h); free(v_h); free(w_h);
    return Ma;
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

    // --- 2. 計算全場 Ma_max (all ranks, with location diagnostics) ---
    double Ma_max = ComputeMaMax();

    // --- 3. Emergency Ma brake (every NDTMIT steps, much faster than NDTFRC) ---
    // 補救 Launch_ModifyForcingTerm 每 1000 步才檢查的時間差
    if (Ma_max > 0.35) {
        Force_h[0] *= 0.05;
        CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
        if (myid == 0)
            printf("[EMERGENCY] Ma_max=%.4f > 0.35, Force reduced to 5%%: %.5E (monitor brake)\n",
                   Ma_max, Force_h[0]);
    } else if (Ma_max > 0.27) {
        Force_h[0] *= 0.5;
        CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
        if (myid == 0)
            printf("[WARNING] Ma_max=%.4f > 0.27, Force halved: %.5E (monitor brake)\n",
                   Ma_max, Force_h[0]);
    }

    // --- 4. 輸出 Ustar_Force_record.dat ---
    double FTT = step * dt_global / (double)flow_through_time;
    double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);

    // 格式: FTT  U*  F*  Ma_max
    if (myid == 0) {
        FILE *fhist = fopen("Ustar_Force_record.dat", "a");
        fprintf(fhist, "%.6f\t%.10f\t%.10f\t%.6f\n", FTT, Ub_inst/(double)Uref, F_star, Ma_max);
        fclose(fhist);
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif
