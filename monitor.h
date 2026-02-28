#ifndef MONITOR_FILE
#define MONITOR_FILE

void Launch_Monitor(){
    // 計算瞬時 Ub: 從 GPU 讀取 rank 0 的 v(j=3) 截面，面積加權平均
    // 同 AccumulateUbulk kernel 公式: Σ v(j=3,k,i) * dx * dz / (LX*(LZ-1))
    double Ub_inst = 0.0;
    if (myid == 0) {
        double *v_slice = (double*)malloc(NX6 * NZ6 * sizeof(double));
        CHECK_CUDA( cudaMemcpy(v_slice, &v[3*NX6*NZ6], NX6*NZ6*sizeof(double), cudaMemcpyDeviceToHost) );

        for (int k = 3; k < NZ6-3; k++) {
        for (int i = 3; i < NX6-4; i++) {
            double dx_loc = (x_h[i+1] - x_h[i-1]) / 2.0;
            double dz_loc = (z_h[3*NZ6 + k+1] - z_h[3*NZ6 + k-1]) / 2.0;
            Ub_inst += v_slice[k*NX6 + i] * dx_loc * dz_loc;
        }}
        Ub_inst /= (double)(LX * (LZ - 1.0));
        free(v_slice);
    }

    // FTT: 使用全域 step（絕對時間），與 checkrho.dat 對齊
    double FTT = step * dt_global / (double)flow_through_time;
    double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);

    // Ustar_Force_record.dat: FTT, U*, F* (每 NDTMIT 步輸出)
    if (myid == 0) {
        FILE *fhist = fopen("Ustar_Force_record.dat", "a");
        fprintf(fhist, "%.6f\t%.10f\t%.10f\n", FTT, Ub_inst/(double)Uref, F_star);
        fclose(fhist);
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif
