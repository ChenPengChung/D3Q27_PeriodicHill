#ifndef MONITOR_FILE
#define MONITOR_FILE

void Launch_Monitor( const int step ){
    const int i = NX6 / 2;
    const int j = NYD6 / 2;
    const int k = NZ6 / 2;

    const int index = j*NX6*NZ6 + k*NX6 +i;

    if( myid == jp / 2 ){
        double v_monitor;
        double u_monitor;
        double w_monitor;
        CHECK_CUDA( cudaMemcpy( &v_monitor, &v[index], sizeof(double), cudaMemcpyDeviceToHost ) );
        CHECK_CUDA( cudaMemcpy( &u_monitor, &u[index], sizeof(double), cudaMemcpyDeviceToHost ) );
        CHECK_CUDA( cudaMemcpy( &w_monitor, &w[index], sizeof(double), cudaMemcpyDeviceToHost ) );
        double FTT = step * dt_global / (double)flow_through_time;
        double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);
        //Ustar_Force_record.dat  每 NDTMIT 步輸出 同頻率
        FILE *fhist = fopen("Ustar_Force_record.dat", "a");
        fprintf(fhist, "%.6f\t%.10f\t%.10f\n", FTT, Ub_avg_global/Uref, F_star);
        fclose(fhist);
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    return;
}

#endif