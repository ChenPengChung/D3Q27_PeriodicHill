#ifndef STATISTICS_FILE
#define STATISTICS_FILE

/* __global__ void MeanVars(
    double *DUDX2,  double *DUDY2,  double *DUDZ2,
    double *DVDX2,  double *DVDY2,  double *DVDZ2,
    double *DWDX2,  double *DWDY2,  double *DWDZ2,
    double *KT,     double *DISS,
    double *u,      double *v,      double *w,
    double *x,      double *y,      double *z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;

    const int index  = j*NX6*NZ6 + k*NX6 + i;
    const int idx_tb = j*NX6*NZ6 + k*NX6 + i;

    double dudx = 0.0, dudy = 0.0, dudz = 0.0, dvdx = 0.0, dvdy = 0.0, dvdz = 0.0, dwdx = 0.0, dwdy = 0.0, dwdz = 0.0;

    if( i <= 2 || i >= NX6-3 || j <= 2 || j >= NYD6-3 || k <= 2 || k >= NZ6-3 ) return;

    const double u1 = u[index];
    const double v1 = v[index];
    const double w1 = w[index];

    const double x1 = x[i+1] - x[i];
    const double x2 = x[i+2] - x[i];
    const double y1 = y[j+1] - y[j];
    const double y2 = y[j+2] - y[j];
    const double z1 = z[k+1] - z[k];
    const double z2 = z[k+2] - z[k];

    KT[idx_tb] = 0.5*(u1*u1 + v1*v1 + w1*w1);

    dudx = (x2*x2*x2*( u[index+1] - u[index-1] ) - x1*x1*x1*( u[index+2] - u[index-2] )) / (2.0*x1*x2*(x2*x2-x1*x1));
    dvdx = (x2*x2*x2*( v[index+1] - v[index-1] ) - x1*x1*x1*( v[index+2] - v[index-2] )) / (2.0*x1*x2*(x2*x2-x1*x1));
    dwdx = (x2*x2*x2*( w[index+1] - w[index-1] ) - x1*x1*x1*( w[index+2] - w[index-2] )) / (2.0*x1*x2*(x2*x2-x1*x1));

    dudy = (y2*y2*y2*( u[index+NX6*NZ6] - u[index-NX6*NZ6] ) - y1*y1*y1*( u[index+2*NX6*NZ6] - u[index-2*NX6*NZ6] )) / (2.0*y1*y2*(y2*y2-y1*y1));
    dvdy = (y2*y2*y2*( v[index+NX6*NZ6] - v[index-NX6*NZ6] ) - y1*y1*y1*( v[index+2*NX6*NZ6] - v[index-2*NX6*NZ6] )) / (2.0*y1*y2*(y2*y2-y1*y1));
    dwdy = (y2*y2*y2*( w[index+NX6*NZ6] - w[index-NX6*NZ6] ) - y1*y1*y1*( w[index+2*NX6*NZ6] - w[index-2*NX6*NZ6] )) / (2.0*y1*y2*(y2*y2-y1*y1));

    dudz = (z2*z2*z2*( u[index+NX6] - u[index-NX6] ) - z1*z1*z1*( u[index+2*NX6] - u[index-2*NX6] )) / (2.0*z1*z2*(z2*z2-z1*z1));
    dvdz = (z2*z2*z2*( v[index+NX6] - v[index-NX6] ) - z1*z1*z1*( v[index+2*NX6] - v[index-2*NX6] )) / (2.0*z1*z2*(z2*z2-z1*z1));
    dwdz = (z2*z2*z2*( w[index+NX6] - w[index-NX6] ) - z1*z1*z1*( w[index+2*NX6] - w[index-2*NX6] )) / (2.0*z1*z2*(z2*z2-z1*z1));

    //dudx = (8.0*( u[index+1] - u[index-1] ) - ( u[index+2] - u[index-2] )) / 12.0 / ( x[i+1] - x[i] );
	//dvdx = (8.0*( v[index+1] - v[index-1] ) - ( v[index+2] - v[index-2] )) / 12.0 / ( x[i+1] - x[i] );
	//dwdx = (8.0*( w[index+1] - w[index-1] ) - ( w[index+2] - w[index-2] )) / 12.0 / ( x[i+1] - x[i] );

	//dudy = (8.0*( u[index+NX6*NZ6] - u[index-NX6*NZ6] ) - ( u[index+2*NX6*NZ6] - u[index-2*NX6*NZ6] )) / 12.0 / ( y[j+1] - y[j] );
	//dvdy = (8.0*( v[index+NX6*NZ6] - v[index-NX6*NZ6] ) - ( v[index+2*NX6*NZ6] - v[index-2*NX6*NZ6] )) / 12.0 / ( y[j+1] - y[j] );
	//dwdy = (8.0*( w[index+NX6*NZ6] - w[index-NX6*NZ6] ) - ( w[index+2*NX6*NZ6] - w[index-2*NX6*NZ6] )) / 12.0 / ( y[j+1] - y[j] );

	//dudz = (8.0*( u[index+NX6] - u[index-NX6] ) - ( u[index+2*NX6] - u[index-2*NX6] )) / 12.0 / ( z[k+1] - z[k] );
	//dvdz = (8.0*( v[index+NX6] - v[index-NX6] ) - ( v[index+2*NX6] - v[index-2*NX6] )) / 12.0 / ( z[k+1] - z[k] );
	//dwdz = (8.0*( w[index+NX6] - w[index-NX6] ) - ( w[index+2*NX6] - w[index-2*NX6] )) / 12.0 / ( z[k+1] - z[k] );

    DISS[idx_tb] = dudx*dudx + dvdx*dvdx + dwdx*dwdx +
                   dudy*dudy + dvdy*dvdy + dwdy*dwdy +
                   dudz*dudz + dvdz*dvdz + dwdz*dwdz;

    __syncthreads();

}

void Launch_TurbulentSum(double *f_new[19]) {
    dim3 griddimTB( NX6, NYD6, NZ6 );
    dim3 blockdimTB(1,   1,   1);

    MeanVars<<<griddimTB, blockdimTB>>>(
        DUDX2, DUDY2, DUDZ2, DVDX2, DVDY2, DVDZ2, DWDX2, DWDY2, DWDZ2,
        KT, DISS, u, v, w,
        x_d, y_d, z_d
    );

    double *KT_h, *DISS_h;
    CHECK_CUDA( cudaMallocHost((void**)&KT_h,   NX6*NYD6*NZ6*sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&DISS_h, NX6*NYD6*NZ6*sizeof(double)) );
    memset( KT_h,   0.0, NX6*NYD6*NZ6*sizeof(double) );
    memset( DISS_h, 0.0, NX6*NYD6*NZ6*sizeof(double) );

    CHECK_CUDA( cudaMemcpy( KT_h,   KT,   NX6*NYD6*NZ6*sizeof(double), cudaMemcpyDeviceToHost ) );
    CHECK_CUDA( cudaMemcpy( DISS_h, DISS, NX6*NYD6*NZ6*sizeof(double), cudaMemcpyDeviceToHost ) );

    double kt = 0.0, diss = 0.0;
    int n = 0;
    for( int k = 3; k < NZ6-4;  k=k+2 ){
    for( int j = 3; j < NYD6-4; j=j+2 ){
    for( int i = 3; i < NX6-4;  i=i+2 ){
        kt = kt + KT_h[j*NX6*NZ6 + k*NX6 + i];
        diss = diss + DISS_h[j*NX6*NZ6 + k*NX6 + i];
        n = n+1;
    }}}
    kt = kt / (double)n;
    diss = diss / (double)n;
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    CHECK_CUDA( cudaFreeHost( KT_h ) );
    CHECK_CUDA( cudaFreeHost( DISS_h ) );

    double Err1 = 0.0, Err2 = 0.0;
    CHECK_MPI( MPI_Reduce( (void*)&kt,   (void*)&Err1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD ) );
    CHECK_MPI( MPI_Reduce( (void*)&diss, (void*)&Err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD ) );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    if( myid == 0 ){
        Err1 = Err1 / (double)jp;
        Err2 = Err2 / (double)jp;
        FILE *tke;
        tke = fopen("Kinetic energy.dat","a");
        fprintf( tke, "%lf\t ",(step*dt*2.0*pi*U_0/LX) );
        fprintf( tke, "%.15lf\t", (Err1/U_0/U_0) );
        fprintf( tke, "%.15lf\n", (niu*Err2/U_0/U_0/U_0*LX/2.0/pi) );
        fclose( tke );
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

} */

__global__ void MeanVars(
          double *U,        double *V,        double *W,        double *P,
          double *UU,       double *UV,       double *UW,       double *VV,       double *VW,       double *WW,
          double *PU,       double *PV,       double *PW,       double *PP,
          double *KT,
          double *UUU,      double *UUV,      double *UUW,
          double *VVU,      double *VVV,      double *VVW,
          double *WWU,      double *WWV,      double *WWW,
    const double *u,  const double *v,  const double *w,  const double *rho  )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;

    const int index  = j*NX6*NZ6 + k*NX6 + i;

    if( i <= 2 || i >= NX6-3 || j <= 2 || j >= NYD6-3 || k <= 2 || k >= NZ6-3 ) return;

    const double u1 = u[index];
    const double v1 = v[index];
    const double w1 = w[index];
    const double p1 = 1.0/3.0*rho[index] - 1.0/3.0;

    U[index] += u1;
    V[index] += v1;
    W[index] += w1;
    P[index] += p1;

    UU[index] += u1 * u1;
    UV[index] += u1 * v1;
    UW[index] += u1 * w1;
    VV[index] += v1 * v1;
    VW[index] += v1 * w1;
    WW[index] += w1 * w1;

    PP[index] += p1 * p1;
    PU[index] += p1 * u1;
    PV[index] += p1 * v1;
    PW[index] += p1 * w1;

    KT[index] += 0.5*(u1*u1 + v1*v1 + w1*w1);

    UUU[index] += u1 * u1 * u1;
    UUV[index] += u1 * u1 * v1;
    UUW[index] += u1 * u1 * w1;
    VVU[index] += v1 * v1 * u1;
    VVV[index] += v1 * v1 * v1;
    VVW[index] += v1 * v1 * w1;
    WWU[index] += w1 * w1 * u1;
    WWV[index] += w1 * w1 * v1;
    WWW[index] += w1 * w1 * w1;

}

__global__ void MeanDerivatives(
          double *DUDX2,	    double *DUDY2,	      double *DUDZ2,
		  double *DVDX2,	    double *DVDY2,	      double *DVDZ2,
		  double *DWDX2,	    double *DWDY2,	      double *DWDZ2,
          double *SlpPara_0,    double *SlpPara_1,    double *SlpPara_2,    double *SlpPara_3,    double *SlpPara_4,
	const double *u,      const double *v,      const double *w,      const double *rho,
	const double *x,      const double *y,      const double *z  )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    const int index = j*NX6*NZ6 + k*NX6 + i;

    double dudx = 0.0, dudy = 0.0, dudz = 0.0, dvdx = 0.0, dvdy = 0.0, dvdz = 0.0, dwdx = 0.0, dwdy = 0.0, dwdz = 0.0;

    if( i <= 2 || i >= NX6-3 || j <= 2 || j >= NYD6-3 || k <= 2 || k >= NZ6-3 ) return;

    dudx = (8.0*( u[index+1] - u[index-1] ) - ( u[index+2] - u[index-2] )) / 12.0 / ( x[i+1] - x[i] );
	dvdx = (8.0*( v[index+1] - v[index-1] ) - ( v[index+2] - v[index-2] )) / 12.0 / ( x[i+1] - x[i] );
	dwdx = (8.0*( w[index+1] - w[index-1] ) - ( w[index+2] - w[index-2] )) / 12.0 / ( x[i+1] - x[i] );

    dudy = (8.0*( u[index+NX6*NZ6] - u[index-NX6*NZ6] ) - ( u[index+2*NX6*NZ6] - u[index-2*NX6*NZ6] )) / 12.0 / ( y[j+1] - y[j] );
	dvdy = (8.0*( v[index+NX6*NZ6] - v[index-NX6*NZ6] ) - ( v[index+2*NX6*NZ6] - v[index-2*NX6*NZ6] )) / 12.0 / ( y[j+1] - y[j] );
	dwdy = (8.0*( w[index+NX6*NZ6] - w[index-NX6*NZ6] ) - ( w[index+2*NX6*NZ6] - w[index-2*NX6*NZ6] )) / 12.0 / ( y[j+1] - y[j] );

    int n = k-2;
    if( k <= 4 ) n = 3;
    if( k >= NZ6-5 ) n = NZ6-8;
    int idx = j*NX6*NZ6 + n*NX6 + i;
    dudz = u[idx]*SlpPara_0[k] + u[idx+NX6]*SlpPara_1[k] + u[idx+2*NX6]*SlpPara_2[k] + u[idx+3*NX6]*SlpPara_3[k] + u[idx+4*NX6]*SlpPara_4[k];
    dvdz = v[idx]*SlpPara_0[k] + v[idx+NX6]*SlpPara_1[k] + v[idx+2*NX6]*SlpPara_2[k] + v[idx+3*NX6]*SlpPara_3[k] + v[idx+4*NX6]*SlpPara_4[k];
    dwdz = w[idx]*SlpPara_0[k] + w[idx+NX6]*SlpPara_1[k] + w[idx+2*NX6]*SlpPara_2[k] + w[idx+3*NX6]*SlpPara_3[k] + w[idx+4*NX6]*SlpPara_4[k];

    DUDX2[index] += dudx * dudx;
    DUDY2[index] += dudy * dudy;
    DUDZ2[index] += dudz * dudz;

    DVDX2[index] += dvdx * dvdx;
    DVDY2[index] += dvdy * dvdy;
    DVDZ2[index] += dvdz * dvdz;

    DWDX2[index] += dwdx * dwdx;
    DWDY2[index] += dwdy * dwdy;
    DWDZ2[index] += dwdz * dwdz;

}

void Launch_TurbulentSum(double *f_new[19]) {
    dim3 griddimTB( NX6/NT+1, NYD6, NZ6 );
    dim3 blockdimTB(NT,     1,      1);

    MeanVars<<<griddimTB, blockdimTB, 0, tbsum_stream[0]>>>(
        U,   V,   W,   P,
        UU,  UV,  UW,  VV,  VW,  WW,  PU,  PV,  PW,  PP,  KT,
        UUU, UUV, UUW, VVU, VVV, VVW, WWU, WWV, WWW,
        u, v, w, rho_d
    );
    CHECK_CUDA( cudaGetLastError() );

    MeanDerivatives<<<griddimTB, blockdimTB, 0, tbsum_stream[1]>>>(
        DUDX2, DUDY2, DUDZ2, DVDX2, DVDY2, DVDZ2, DWDX2, DWDY2, DWDZ2,
        ZSlopePara_d[0], ZSlopePara_d[1], ZSlopePara_d[2], ZSlopePara_d[3], ZSlopePara_d[4],
		u, v, w, rho_d,
		x_d, y_d, z_d
    );
    CHECK_CUDA( cudaGetLastError() );

    for( int i = 0; i < 2; i++ ){
        CHECK_CUDA( cudaStreamSynchronize(tbsum_stream[i]) );
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif