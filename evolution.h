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


__global__ void AccumulateUbulk(
    double *Ub_avg,     double *v,
    double *x,          double *z  )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + 3;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if( i <= 2 || i >= NX6-3 || k <= 2 || k >= NZ6-3 ) return;  // 包含壁面 k=3,NZ6-4

    double dx = ( x[i+1] - x[i-1] ) / 2.0;
    double dz = ( z[j*NZ6+k+1] - z[j*NZ6+k-1] ) / 2.0;  // k=3 時讀 z[k=2](extrap) 和 z[k=4]

    Ub_avg[k*NX6+i] += v[j*NZ6*NX6+k*NX6+i] * dx * dz;
}

void Launch_CollisionStreaming(double *f_old[19], double *f_new[19]) {
    int buffer = 3;

    // Option B double-buffer: pre-copy f_old → f_new so kernel starts with last iteration's values
    const size_t grid_bytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
    for (int q = 0; q < 19; q++)
        CHECK_CUDA( cudaMemcpyAsync(f_new[q], f_old[q], grid_bytes, cudaMemcpyDeviceToDevice, stream0) );
    CHECK_CUDA( cudaStreamSynchronize(stream0) );

    dim3 griddimSW(  1,      NYD6/NT+1, NZ6);
    dim3 blockdimSW( buffer, NT,        1 );

    dim3 griddim(  NX6/NT+1, NYD6, NZ6);
    dim3 blockdim( NT, 1, 1);

    dim3 griddimBuf(NX6/NT+1, 1, NZ6);
    dim3 blockdimBuf(NT, 4, 1);

    // ===== GILBM two-pass kernel dispatch =====
    GILBM_StreamCollide_Buffer_Kernel<<<griddimBuf, blockdimBuf, 0, stream1>>>(
    f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6], f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12], f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
    f_pc_d, feq_d, omegadt_local_d,
    dk_dz_d, dk_dy_d, delta_zeta_d,
    dt_local_d, omega_local_d,
    u, v, w, rho_d, Force_d, rho_modify_d, 3
    );
    GILBM_StreamCollide_Buffer_Kernel<<<griddimBuf, blockdimBuf, 0, stream1>>>(
    f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6], f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12], f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
    f_pc_d, feq_d, omegadt_local_d,
    dk_dz_d, dk_dy_d, delta_zeta_d,
    dt_local_d, omega_local_d,
    u, v, w, rho_d, Force_d, rho_modify_d, NYD6-7
    );

    dim3 griddim_Ubulk(  NX6/NT+1, 1, NZ6);
    dim3 blockdim_Ubulk( NT, 1, 1);
    AccumulateUbulk<<<griddim_Ubulk, blockdim_Ubulk, 0, stream1>>>(
    Ub_avg_d, v, x_d, z_d
    );

    GILBM_StreamCollide_Kernel<<<griddim, blockdim, 0, stream0>>>(
    f_new[0], f_new[1], f_new[2], f_new[3], f_new[4], f_new[5], f_new[6], f_new[7], f_new[8], f_new[9], f_new[10], f_new[11], f_new[12], f_new[13], f_new[14], f_new[15], f_new[16], f_new[17], f_new[18],
    f_pc_d, feq_d, omegadt_local_d,
    dk_dz_d, dk_dy_d, delta_zeta_d,
    dt_local_d, omega_local_d,
    u, v, w, rho_d, Force_d, rho_modify_d
    );

    CHECK_CUDA( cudaStreamSynchronize(stream1) );

    // GILBM: 必須交換全部 19 個 f 方向（不只 y-moving 的 10 個）
    // 原因: Step 2+3 讀取 ghost zone 所有 q 的 f_new[q][idx_B] 做重估+碰撞
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

    CHECK_CUDA( cudaStreamSynchronize(stream0) );

    periodicSW<<<griddimSW, blockdimSW, 0, stream0>>>(
        f_old[0] ,f_old[1] ,f_old[2] ,f_old[3] ,f_old[4] ,f_old[5] ,f_old[6] ,f_old[7] ,f_old[8] ,f_old[9] ,f_old[10] ,f_old[11] ,f_old[12] ,f_old[13] ,f_old[14] ,f_old[15] ,f_old[16] ,f_old[17] ,f_old[18],
        f_new[0] ,f_new[1] ,f_new[2] ,f_new[3] ,f_new[4] ,f_new[5] ,f_new[6] ,f_new[7] ,f_new[8] ,f_new[9] ,f_new[10] ,f_new[11] ,f_new[12] ,f_new[13] ,f_new[14] ,f_new[15] ,f_new[16] ,f_new[17] ,f_new[18],
        y_d, x_d, z_d, u, v, w, rho_d, feq_d
    );
}

void Launch_ModifyForcingTerm()
{
    const size_t nBytes = NX6 * NZ6 * sizeof(double);
    CHECK_CUDA( cudaMemcpy(Ub_avg_h, Ub_avg_d, nBytes, cudaMemcpyDeviceToHost) );
    
    double Ub_avg = 0.0;
    for( int k = 3; k < NZ6-3; k++ ){    // 包含壁面計算點
    for( int i = 3; i < NX6-4; i++ ){
        Ub_avg = Ub_avg + Ub_avg_h[k*NX6+i];
        Ub_avg_h[k*NX6+i] = 0.0;
    }}
    Ub_avg = Ub_avg / (double)(LX*(LZ-1.0))/NDTFRC;

    CHECK_CUDA( cudaMemcpy(Ub_avg_d, Ub_avg_h, nBytes, cudaMemcpyHostToDevice) );

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    
    double beta = max(0.001, force_alpha/(double)Re);
    Force_h[0] = Force_h[0] + beta*(Uref - Ub_avg)*Uref/(double)LY; 

    double force_avg = 0.0;
    CHECK_MPI( MPI_Reduce( (void*)Force_h, (void*)&force_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD ) );
    CHECK_MPI( MPI_Barrier( MPI_COMM_WORLD ) );

    if( myid == 0 ){
        force_avg = force_avg / (double)jp;
        Force_h[0] = force_avg;
    }

    CHECK_MPI( MPI_Bcast( (void*)Force_h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD ) );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    // 無因次化量 (論文 Fig.5)
    double FTT    = step * dt_global / (double)flow_through_time;  // T*Uref/L (用 dt_global, 非 minSize)
    double U_star = Ub_avg / (double)Uref;                          // U* = Ub/Uref
    double F_star = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);  // F* = F*L/(rho*Uref^2), rho=1
    double Re_now = Ub_avg / (double)niu;
    double Ma_now = Ub_avg / (double)cs;

    printf("[Step %d | FTT=%.2f] Ub=%.6f  U*=%.4f  Force=%.5E  F*=%.4f  Re(now)=%.1f  Ma=%.4f\n",
           step, FTT, Ub_avg, U_star, Force_h[0], F_star, Ub_avg / ((double)Uref/(double)Re), Ma_now);

    // 寫入 ForcingHistory.dat (Rank 0 only, 對應論文 Fig.5 的兩條曲線)
    if (myid == 0) {
        FILE *fhist = fopen("ForcingHistory.dat", "a");
        fprintf(fhist, "%.6f\t %.10f\t %.10f\n", FTT, U_star, F_star);
        fclose(fhist);
    }

    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
    
    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    
}

#endif
