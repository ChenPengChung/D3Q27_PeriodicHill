#ifndef COMMUNICATION_FILE
#define COMMUNICATION_FILE

void CreateDataType() {

    CHECK_MPI( MPI_Type_vector(icount_sw, 1, 1, MPI_DOUBLE, &DataSideways) );
    CHECK_MPI( MPI_Type_commit(&DataSideways) );

}

void ISend_LtRtBdry(
    double *f_new[19], 
    const int istart,       const int nbr,      const int itag[23], 
    const int req,          const int num_arrays, ...)
{
    va_list args;
    va_start( args, num_arrays );

    for( int i = 0; i < num_arrays; i++ ) {
        const int dir = va_arg(args, int);
        CHECK_MPI(
            MPI_Isend((void *)&f_new[dir][istart],     1,   DataSideways,   nbr,   itag[i],
                      MPI_COMM_WORLD,   &(request[i][req]))
        );
    }

    va_end( args );

    CHECK_MPI(
        MPI_Isend((void *)&u[istart],           1,      DataSideways,      nbr,     itag[19],
                  MPI_COMM_WORLD,   &(request[19][req]))
    );
    CHECK_MPI(
        MPI_Isend((void *)&v[istart],           1,      DataSideways,      nbr,     itag[20],
                  MPI_COMM_WORLD,   &(request[20][req]))
    );
    CHECK_MPI(
        MPI_Isend((void *)&w[istart],           1,      DataSideways,      nbr,     itag[21],
                  MPI_COMM_WORLD,   &(request[21][req]))
    );
    CHECK_MPI(
        MPI_Isend((void *)&rho_d[istart],       1,      DataSideways,      nbr,     itag[22],
                  MPI_COMM_WORLD,   &(request[22][req]))
    );

}

void IRecv_LtRtBdry(
    double *f_new[19],
    const int istart,       const int nbr,      const int itag[23],
    const int req,          const int num_arrays, ...)
{
    va_list args;
    va_start( args, num_arrays );

    for( int i = 0; i < num_arrays; i++ ) {
        const int dir = va_arg(args, int);
        CHECK_MPI(
            MPI_Irecv((void *)&f_new[dir][istart],     1,    DataSideways,   nbr,  itag[i],
                      MPI_COMM_WORLD,   &(request[i][req]))
        );
    }

    va_end( args );

    CHECK_MPI(
        MPI_Irecv((void *)&u[istart],           1,      DataSideways,      nbr,     itag[19],
                  MPI_COMM_WORLD,   &(request[19][req]))
    );
    CHECK_MPI(
        MPI_Irecv((void *)&v[istart],           1,      DataSideways,      nbr,     itag[20],
                  MPI_COMM_WORLD,   &(request[20][req]))
    );
    CHECK_MPI(
        MPI_Irecv((void *)&w[istart],           1,      DataSideways,      nbr,     itag[21],
                  MPI_COMM_WORLD,   &(request[21][req]))
    );
    CHECK_MPI(
        MPI_Irecv((void *)&rho_d[istart],       1,      DataSideways,      nbr,     itag[22],
                  MPI_COMM_WORLD,   &(request[22][req]))
    );
}

void Isend_Sideways(const int istart, const int sw_nbr, const int itag_sw[23], MPI_Request reqSideways[23], const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int i = 0; i < num_arrays; i++ ) {
        const int dir = va_arg(args, int);
        CHECK_MPI(
            MPI_Isend((void *)&fh_p[dir][istart],     1,   DataSideways,   sw_nbr,   itag_sw[i],
                      MPI_COMM_WORLD,   &reqSideways[i])
        );
    }

    va_end( args );

    CHECK_MPI(
        MPI_Isend((void *)&u_h_p[istart],       1,      DataSideways,   sw_nbr,     itag_sw[19],
                  MPI_COMM_WORLD,   &reqSideways[19])
    );
    CHECK_MPI(
        MPI_Isend((void *)&v_h_p[istart],       1,      DataSideways,   sw_nbr,     itag_sw[20],
                  MPI_COMM_WORLD,   &reqSideways[20])
    );
    CHECK_MPI(
        MPI_Isend((void *)&w_h_p[istart],       1,      DataSideways,   sw_nbr,     itag_sw[21],
                  MPI_COMM_WORLD,   &reqSideways[21])
    );
    CHECK_MPI(
        MPI_Isend((void *)&rho_h_p[istart],     1,      DataSideways,   sw_nbr,     itag_sw[22],
                  MPI_COMM_WORLD,   &reqSideways[22])
    );

}

void Irecv_Sideways(const int istart, const int sw_nbr, const int itag_sw[23], MPI_Request reqSideways[23], const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int i = 0; i < num_arrays; i++ ) {
        const int dir = va_arg(args, int);
        CHECK_MPI(
            MPI_Irecv((void *)&fh_p[dir][istart],     1,    DataSideways,   sw_nbr,  itag_sw[i],
                      MPI_COMM_WORLD,   &reqSideways[i])
        );
    }

    va_end( args );

    CHECK_MPI(
        MPI_Irecv((void *)&u_h_p[istart],       1,      DataSideways,   sw_nbr,     itag_sw[19],
                  MPI_COMM_WORLD,   &reqSideways[19])
    );
    CHECK_MPI(
        MPI_Irecv((void *)&v_h_p[istart],       1,      DataSideways,   sw_nbr,     itag_sw[20],
                  MPI_COMM_WORLD,   &reqSideways[20])
    );
    CHECK_MPI(
        MPI_Irecv((void *)&w_h_p[istart],       1,      DataSideways,   sw_nbr,     itag_sw[21],
                  MPI_COMM_WORLD,   &reqSideways[21])
    );
    CHECK_MPI(
        MPI_Irecv((void *)&rho_h_p[istart],     1,      DataSideways,   sw_nbr,     itag_sw[22],
                  MPI_COMM_WORLD,   &reqSideways[22])
    );

}

void Wait_Sideways(
    double *f_new[19], const int iend_sw, 
    MPI_Request reqSend[23], MPI_Request reqRecv[23], const int transfsize, cudaStream_t stream0,
    const int num_arrays, ...)
{    
    va_list args;
    va_start( args, num_arrays );

    for( int i = 0; i < num_arrays; i++ ) {
        const int dir = va_arg(args, int);

        CHECK_MPI( MPI_Wait(&reqSend[i], istat) );
        CHECK_MPI( MPI_Wait(&reqRecv[i], istat) );
        CHECK_CUDA(
            cudaMemcpyAsync((void *)&f_new[dir][iend_sw],	(void *)&fh_p[dir][iend_sw],    transfsize*sizeof(double),
					        cudaMemcpyHostToDevice,		stream0)
        );
    }

    va_end( args );

    CHECK_MPI( MPI_Wait(&reqSend[19], istat) );
    CHECK_MPI( MPI_Wait(&reqRecv[19], istat) );
    CHECK_CUDA(
        cudaMemcpyAsync((void *)&u[iend_sw],    (void *)&u_h_p[iend_sw],   transfsize*sizeof(double),
                        cudaMemcpyHostToDevice,     stream0)
    );
    CHECK_MPI( MPI_Wait(&reqSend[20], istat) );
    CHECK_MPI( MPI_Wait(&reqRecv[20], istat) );
    CHECK_CUDA(
        cudaMemcpyAsync((void *)&v[iend_sw],    (void *)&v_h_p[iend_sw],   transfsize*sizeof(double),
                        cudaMemcpyHostToDevice,     stream0)
    );
    CHECK_MPI( MPI_Wait(&reqSend[21], istat) );
    CHECK_MPI( MPI_Wait(&reqRecv[21], istat) );
    CHECK_CUDA(
        cudaMemcpyAsync((void *)&w[iend_sw],    (void *)&w_h_p[iend_sw],   transfsize*sizeof(double),
                        cudaMemcpyHostToDevice,     stream0)
    );
    CHECK_MPI( MPI_Wait(&reqSend[22], istat) );
    CHECK_MPI( MPI_Wait(&reqRecv[22], istat) );
    CHECK_CUDA(
        cudaMemcpyAsync((void *)&rho_d[iend_sw],(void *)&rho_h_p[iend_sw], transfsize*sizeof(double),
                        cudaMemcpyHostToDevice,     stream0)
    );

}

void SendBdryToCPU_Sideways(cudaStream_t stream, double *f_new[19], const int istart, const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    const size_t nBytes = 3 * NX6 * NZ6 * sizeof(double);

    for( int i = 0; i < num_arrays; i++ ) {
        const int dir = va_arg(args, int);
        CHECK_CUDA( cudaMemcpyAsync((void *)&fh_p[dir][istart],(void *)&f_new[dir][istart], nBytes, cudaMemcpyDeviceToHost, stream) );
    }

    CHECK_CUDA( cudaMemcpyAsync((void *)&u_h_p[istart],   (void *)&u[istart],     nBytes, cudaMemcpyDeviceToHost, stream) );
    CHECK_CUDA( cudaMemcpyAsync((void *)&v_h_p[istart],   (void *)&v[istart],     nBytes, cudaMemcpyDeviceToHost, stream) );
    CHECK_CUDA( cudaMemcpyAsync((void *)&w_h_p[istart],   (void *)&w[istart],     nBytes, cudaMemcpyDeviceToHost, stream) );
    CHECK_CUDA( cudaMemcpyAsync((void *)&rho_h_p[istart], (void *)&rho_d[istart], nBytes, cudaMemcpyDeviceToHost, stream) );

    CHECK_CUDA( cudaStreamSynchronize(stream) );

    va_end( args );
}

void SendSrcToGPU(const size_t nBytes, const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int i = 0; i < num_arrays; i++ ) {
        const int dir = va_arg(args, int);
        CHECK_CUDA( cudaMemcpy( (void*)fd[dir], (void*)fh_p[dir], nBytes, cudaMemcpyHostToDevice ) );
        CHECK_CUDA( cudaMemcpy( (void*)ft[dir], (void*)fh_p[dir], nBytes, cudaMemcpyHostToDevice ) );
    }

    CHECK_CUDA( cudaDeviceSynchronize() );
    va_end( args );
}

void SendSrcToCPU(double *f_new[19], const size_t nBytes, const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int i = 0; i < num_arrays; i++ ) {
        const int dir = va_arg(args, int);
        CHECK_CUDA( cudaMemcpy(fh_p[dir],f_new[dir],nBytes,cudaMemcpyDeviceToHost); )
    }

    va_end( args );
}

void SendDataToGPU() {
    const size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);

    CHECK_CUDA( cudaMemcpy(rho_d, rho_h_p, nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(u,     u_h_p,   nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(v,     v_h_p,   nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(w,     w_h_p,   nBytes, cudaMemcpyHostToDevice) );

    SendSrcToGPU(nBytes, 19, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18);

    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

}

void SendDataToCPU(double *f_new[19]) {
    const size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);

    CHECK_CUDA( cudaMemcpy(u_h_p,   u,     nBytes, cudaMemcpyDeviceToHost) );
	CHECK_CUDA( cudaMemcpy(v_h_p,   v,     nBytes, cudaMemcpyDeviceToHost) );
	CHECK_CUDA( cudaMemcpy(w_h_p,   w,     nBytes, cudaMemcpyDeviceToHost) );
	CHECK_CUDA( cudaMemcpy(rho_h_p, rho_d, nBytes, cudaMemcpyDeviceToHost) );

    SendSrcToCPU(f_new, nBytes, 19, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18);

    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

}

#endif