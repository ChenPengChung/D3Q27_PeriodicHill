#ifndef STATISTICS_FILE
#define STATISTICS_FILE

// MeanVars: accumulate 1st-order (4+3), 2nd-order (6), pressure (3), 3rd-order (10) = 26 fields
// PP and KT removed — KT derived in VTK as 0.5*(uu+vv+ww), PP not needed
// UVW (u*v*w) added for complete symmetric 3rd-order tensor
__global__ void MeanVars(
          double *U,        double *V,        double *W,        double *P,
          double *UU,       double *UV,       double *UW,       double *VV,       double *VW,       double *WW,
          double *PU,       double *PV,       double *PW,
          double *UUU,      double *UUV,      double *UUW,
          double *VVU,      double *UVW_arr,  double *WWU,
          double *VVV,      double *VVW,      double *WWV,      double *WWW,
    const double *u,  const double *v,  const double *w,  const double *rho  )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;

    const int index  = j*NX6*NZ6 + k*NX6 + i;

    if( i <= 2 || i >= NX6-3 || j <= 2 || j >= NYD6-3 || k <= 3 || k >= NZ6-4 ) return;

    const double u1 = u[index];
    const double v1 = v[index];
    const double w1 = w[index];
    const double p1 = 1.0/3.0*rho[index] - 1.0/3.0;

    // 一階矩 (4)
    U[index] += u1;
    V[index] += v1;
    W[index] += w1;
    P[index] += p1;

    // 二階矩 (6, symmetric)
    UU[index] += u1 * u1;
    UV[index] += u1 * v1;
    UW[index] += u1 * w1;
    VV[index] += v1 * v1;
    VW[index] += v1 * w1;
    WW[index] += w1 * w1;

    // 壓力交叉 (3)
    PU[index] += p1 * u1;
    PV[index] += p1 * v1;
    PW[index] += p1 * w1;

    // 三階矩 (10, complete symmetric tensor)
    UUU[index]     += u1 * u1 * u1;
    UUV[index]     += u1 * u1 * v1;
    UUW[index]     += u1 * u1 * w1;
    VVU[index]     += u1 * v1 * v1;  // = uvv
    UVW_arr[index] += u1 * v1 * w1;  // NEW: uvw
    WWU[index]     += u1 * w1 * w1;  // = uww
    VVV[index]     += v1 * v1 * v1;
    VVW[index]     += v1 * v1 * w1;
    WWV[index]     += v1 * w1 * w1;  // = vww
    WWW[index]     += w1 * w1 * w1;
}

// MeanDerivatives: accumulate squared velocity gradients (9) + vorticity sums (3) = 12 fields
// Uses curvilinear coordinate metrics for correct physical-space derivatives
__global__ void MeanDerivatives(
          double *DUDX2,      double *DUDY2,        double *DUDZ2,
          double *DVDX2,      double *DVDY2,        double *DVDZ2,
          double *DWDX2,      double *DWDY2,        double *DWDZ2,
          double *OMEGA_U_SUM, double *OMEGA_V_SUM,  double *OMEGA_W_SUM,
    const double *dk_dz_in,   const double *dk_dy_in,
    const double *u,    const double *v,    const double *w,
    const double *x,    const double *y  )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    const int index = j*NX6*NZ6 + k*NX6 + i;

    if( i <= 2 || i >= NX6-3 || j <= 2 || j >= NYD6-3 || k <= 3 || k >= NZ6-4 ) return;

    // x-direction: ∂φ/∂x = (1/dx) · ∂φ/∂η  (4th-order central diff)
    double dx_inv = 1.0 / (x[i+1] - x[i]);
    double dudx = (8.0*(u[index+1] - u[index-1]) - (u[index+2] - u[index-2])) / 12.0 * dx_inv;
    double dvdx = (8.0*(v[index+1] - v[index-1]) - (v[index+2] - v[index-2])) / 12.0 * dx_inv;
    double dwdx = (8.0*(w[index+1] - w[index-1]) - (w[index+2] - w[index-2])) / 12.0 * dx_inv;

    // k-direction finite difference: ∂φ/∂ζ (4th-order, shared by y and z)
    double du_dk = (8.0*(u[index+NX6] - u[index-NX6]) - (u[index+2*NX6] - u[index-2*NX6])) / 12.0;
    double dv_dk = (8.0*(v[index+NX6] - v[index-NX6]) - (v[index+2*NX6] - v[index-2*NX6])) / 12.0;
    double dw_dk = (8.0*(w[index+NX6] - w[index-NX6]) - (w[index+2*NX6] - w[index-2*NX6])) / 12.0;

    // j-direction finite difference: ∂φ/∂ξ (4th-order)
    const int nface = NX6 * NZ6;
    double du_dj = (8.0*(u[index+nface] - u[index-nface]) - (u[index+2*nface] - u[index-2*nface])) / 12.0;
    double dv_dj = (8.0*(v[index+nface] - v[index-nface]) - (v[index+2*nface] - v[index-2*nface])) / 12.0;
    double dw_dj = (8.0*(w[index+nface] - w[index-nface]) - (w[index+2*nface] - w[index-2*nface])) / 12.0;

    double dkdy = dk_dy_in[j*NZ6 + k];
    double dkdz = dk_dz_in[j*NZ6 + k];
    double dy_inv = 1.0 / (y[j+1] - y[j]);

    // y-direction: ∂φ/∂y = (1/dy)·∂φ/∂ξ + dk_dy·∂φ/∂ζ
    double dudy = du_dj * dy_inv + dkdy * du_dk;
    double dvdy = dv_dj * dy_inv + dkdy * dv_dk;
    double dwdy = dw_dj * dy_inv + dkdy * dw_dk;

    // z-direction: ∂φ/∂z = dk_dz·∂φ/∂ζ
    double dudz = dkdz * du_dk;
    double dvdz = dkdz * dv_dk;
    double dwdz = dkdz * dw_dk;

    // Accumulate squared gradients (9)
    DUDX2[index] += dudx * dudx;
    DUDY2[index] += dudy * dudy;
    DUDZ2[index] += dudz * dudz;

    DVDX2[index] += dvdx * dvdx;
    DVDY2[index] += dvdy * dvdy;
    DVDZ2[index] += dvdz * dvdz;

    DWDX2[index] += dwdx * dwdx;
    DWDY2[index] += dwdy * dwdy;
    DWDZ2[index] += dwdz * dwdz;

    // Accumulate vorticity components (3)
    // omega_x = ∂w/∂y - ∂v/∂z
    // omega_y = ∂u/∂z - ∂w/∂x
    // omega_z = ∂v/∂x - ∂u/∂y
    OMEGA_U_SUM[index] += (dwdy - dvdz);
    OMEGA_V_SUM[index] += (dudz - dwdx);
    OMEGA_W_SUM[index] += (dvdx - dudy);
}

void Launch_TurbulentSum(double *f_new[19]) {
    dim3 griddimTB( NX6/NT+1, NYD6, NZ6 );
    dim3 blockdimTB(NT,     1,      1);

    MeanVars<<<griddimTB, blockdimTB, 0, tbsum_stream[0]>>>(
        U,   V,   W,   P,
        UU,  UV,  UW,  VV,  VW,  WW,  PU,  PV,  PW,
        UUU, UUV, UUW, VVU, UVW, WWU, VVV, VVW, WWV, WWW,
        u, v, w, rho_d
    );
    CHECK_CUDA( cudaGetLastError() );

    MeanDerivatives<<<griddimTB, blockdimTB, 0, tbsum_stream[1]>>>(
        DUDX2, DUDY2, DUDZ2, DVDX2, DVDY2, DVDZ2, DWDX2, DWDY2, DWDZ2,
        OMEGA_U_SUM, OMEGA_V_SUM, OMEGA_W_SUM,
        dk_dz_d, dk_dy_d,
        u, v, w,
        x_d, y_d
    );
    CHECK_CUDA( cudaGetLastError() );

    for( int i = 0; i < 2; i++ ){
        CHECK_CUDA( cudaStreamSynchronize(tbsum_stream[i]) );
    }

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif
