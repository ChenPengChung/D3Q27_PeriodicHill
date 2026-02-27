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
        double FTT_mon = step * dt_global / (double)flow_through_time;
        double F_star_mon = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);
        FILE *Monitor;
        Monitor = fopen("Monitor.dat","a");
        // 欄位: step | FTT | v(streamwise) | u(spanwise) | w(wall-normal) | Force | F*
        fprintf( Monitor, "%d\t %.6f\t %.15lf\t %.15lf\t %.15lf\t %.15lf\t %.6f\n",
                 step, FTT_mon, v_monitor, u_monitor, w_monitor, Force_h[0], F_star_mon);
        fclose( Monitor );
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    return;
}

#endif