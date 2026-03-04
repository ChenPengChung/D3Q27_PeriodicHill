#ifndef MONITOR_FILE
#define MONITOR_FILE

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

    double local_Ma = sqrt(local_max_sq) / (double)cs;
    double global_Ma;
    MPI_Allreduce(&local_Ma, &global_Ma, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    free(u_h); free(v_h); free(w_h);
    return global_Ma;
}

// Returns 0 = normal, 1 = emergency stop (Ma > 0.40)
int Launch_Monitor(){
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

    // --- 2. 計算全場 Ma_max ---
    double Ma_max = ComputeMaMax();

    // --- 3. Emergency Ma brake (every NDTMIT steps, faster than NDTFRC) ---
    // 閾值: 0.30→輕煞車, 0.35→重煞車, 0.40→不可恢復→停止模擬
    if (Ma_max > 0.40) {
        Force_h[0] = 0.0;
        CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
        if (myid == 0)
            printf("[FATAL] Ma_max=%.4f > 0.40 — unrecoverable, stopping simulation (monitor brake)\n", Ma_max);
        return 1;  // signal emergency stop
    } else if (Ma_max > 0.35) {
        Force_h[0] *= 0.1;
        CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
        if (myid == 0)
            printf("[CRITICAL] Ma_max=%.4f > 0.35, Force reduced to 10%%: %.5E (monitor brake)\n",
                   Ma_max, Force_h[0]);
    } else if (Ma_max > 0.30) {
        Force_h[0] *= 0.8;
        CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
        if (myid == 0)
            printf("[WARNING] Ma_max=%.4f > 0.30, Force reduced to 80%%: %.5E (monitor brake)\n",
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
    return 0;  // normal
}

#endif
