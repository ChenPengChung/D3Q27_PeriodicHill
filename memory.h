#ifndef MEMORY_FILE
#define MEMORY_FILE

void AllocateHostArray(const size_t nBytes, const int num_arrays, ...) {
	va_list args;
	va_start( args, num_arrays );

	for( int i = 0; i < num_arrays; i++ ) {
        double **tmp = va_arg(args, double**);
		CHECK_CUDA( cudaMallocHost( (void**)tmp, nBytes) );
	}

	va_end( args );
}

void AllocateDeviceArray(const size_t nBytes, const int num_arrays, ...) {
	va_list args;
	va_start( args, num_arrays );

	for( int i = 0; i < num_arrays; i++ ) {
        double **tmp = va_arg(args, double**);
		CHECK_CUDA( cudaMalloc( (void**)tmp, nBytes) );
	}

	va_end( args );
}

void FreeHostArray(const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int i = 0; i < num_arrays; i++ ) {
        CHECK_CUDA( cudaFreeHost( (void*)(va_arg(args, double*)) ) );
    }

    va_end( args );

}

void FreeDeviceArray(const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int  i = 0; i < num_arrays; i++ ) {
        CHECK_CUDA( cudaFree( (void*)(va_arg(args, double*)) ) );
    }

    va_end( args );
}

void AllocateMemory() {
    size_t nBytes;

    nBytes = NX6 * NYD6 * NZ6 * sizeof(double);

    AllocateHostArray( nBytes, 4, &rho_h_p, &u_h_p, &v_h_p, &w_h_p );
    for( int i = 0; i < 19; i++ ) {
        CHECK_CUDA( cudaMallocHost( (void**)&fh_p[i], nBytes ) );
        memset(fh_p[i], 0.0, nBytes);
    }

    AllocateDeviceArray(nBytes, 4,  &rho_d, &u, &v, &w);
    for( int i = 0; i < 19; i++ ) {
        CHECK_CUDA( cudaMalloc( &fd[i], nBytes ) );     CHECK_CUDA( cudaMemset( fd[i], 0.0, nBytes ) );
        CHECK_CUDA( cudaMalloc( &ft[i], nBytes ) );     CHECK_CUDA( cudaMemset( ft[i], 0.0, nBytes ) );
    }

    if( TBSWITCH ) {
        //AllocateDeviceArray(nBytes, 2,  &KT, &DISS);
        //AllocateDeviceArray(nBytes, 9,  &DUDX2, &DUDY2, &DUDZ2, &DVDX2, &DVDY2, &DVDZ2, &DWDX2, &DWDY2, &DWDZ2);
        AllocateDeviceArray(nBytes, 4,  &U,  &V,  &W,  &P);
        AllocateDeviceArray(nBytes, 10, &UU, &UV, &UW, &VV, &VW, &WW, &PU, &PV, &PW, &PP);
        AllocateDeviceArray(nBytes, 1,  &KT);
        AllocateDeviceArray(nBytes, 9,  &DUDX2, &DUDY2, &DUDZ2, &DVDX2, &DVDY2, &DVDZ2, &DWDX2, &DWDY2, &DWDZ2);
    	AllocateDeviceArray(nBytes, 9,  &UUU,   &UUV,   &UUW,   &VVU,   &VVV,   &VVW,   &WWU,   &WWV,   &WWW);
    }

    nBytes = NYD6 * sizeof(double);
    AllocateHostArray(  nBytes, 4,  &y_h, &Ydep_h[0], &Ydep_h[1], &Ydep_h[2]);
    AllocateDeviceArray(nBytes, 4,  &y_d, &Ydep_d[0], &Ydep_d[1], &Ydep_d[2]);
    nBytes = NX6 * sizeof(double);
    AllocateHostArray(  nBytes, 4,  &x_h, &Xdep_h[0], &Xdep_h[1], &Xdep_h[2]);
    AllocateDeviceArray(nBytes, 4,  &x_d, &Xdep_d[0], &Xdep_d[1], &Xdep_d[2]);

    nBytes = NYD6 * NZ6 * sizeof(double);
    AllocateHostArray(  nBytes, 4,  &z_h, &Zdep_h[0], &Zdep_h[1], &Zdep_h[2]);
    AllocateDeviceArray(nBytes, 4,  &z_d, &Zdep_d[0], &Zdep_d[1], &Zdep_d[2]);
    // GILBM 度量項：∂ζ/∂z 和 ∂ζ/∂y（與 z_h 同大小 [NYD6*NZ6]）
    AllocateHostArray(  nBytes, 2,  &dk_dz_h, &dk_dy_h);
    AllocateDeviceArray(nBytes, 2,  &dk_dz_d, &dk_dy_d);
    // GILBM 預計算 RK2 ζ 方向位移 [19 * NYD6 * NZ6]
    nBytes = 19 * NYD6 * NZ6 * sizeof(double);
    AllocateHostArray(  nBytes, 1, &delta_zeta_h);
    AllocateDeviceArray(nBytes, 1, &delta_zeta_d);
    // Phase 4 LTS: local dt, tau, tau*dt fields [NYD6 * NZ6]
    nBytes = NYD6 * NZ6 * sizeof(double);
    AllocateHostArray(  nBytes, 3, &dt_local_h, &tau_local_h, &tau_dt_product_h);
    AllocateDeviceArray(nBytes, 3, &dt_local_d, &tau_local_d, &tau_dt_product_d);

    nBytes = NZ6 * sizeof(double);
    CHECK_CUDA( cudaMallocHost( (void**)&xi_h, nBytes ) );
    CHECK_CUDA( cudaMalloc( &xi_d, nBytes ) );

    nBytes = NZ6 * NX6 * sizeof(double);
    CHECK_CUDA( cudaMallocHost( (void**)&Ub_avg_h, nBytes ) );
    CHECK_CUDA( cudaMalloc( &Ub_avg_d, nBytes ) );

    nBytes = sizeof(double);
    CHECK_CUDA( cudaMallocHost( (void**)&Force_h, nBytes ) );
    CHECK_CUDA( cudaMalloc( &Force_d, nBytes ) );
    CHECK_CUDA( cudaMallocHost( (void**)&rho_modify_h, nBytes ) );
    CHECK_CUDA( cudaMalloc( &rho_modify_d, nBytes ) );

    CHECK_CUDA( cudaStreamCreate( &stream0 ) );
    CHECK_CUDA( cudaStreamCreate( &stream1 ) );
    CHECK_CUDA( cudaStreamCreate( &stream2 ) );
    for( int i = 0; i < 2; i++ )
        CHECK_CUDA( cudaStreamCreate( &tbsum_stream[i] ) );

    CHECK_CUDA( cudaEventCreate( &start  ) );
    CHECK_CUDA( cudaEventCreate( &stop   ) );
    CHECK_CUDA( cudaEventCreate( &start1 ) );
    CHECK_CUDA( cudaEventCreate( &stop1  ) );
}

void FreeSource() {

    for( int i = 0; i < 19; i++ )
        CHECK_CUDA( cudaFreeHost( fh_p[i] ) );
        
    FreeHostArray(  3,  rho_h_p, u_h_p, v_h_p);

    for( int i = 0; i < 19; i++ ) {
        CHECK_CUDA( cudaFree( ft[i] ) );
        CHECK_CUDA( cudaFree( fd[i] ) );
    }
    FreeDeviceArray(4,  rho_d, u, v, w);

    if( TBSWITCH ) {
        //FreeDeviceArray(2,  KT, DISS);
        //FreeDeviceArray(9,  DUDX2, DUDY2, DUDZ2, DVDX2, DVDY2, DVDZ2, DWDX2, DWDY2, DWDZ2);
        FreeDeviceArray(4,  U,  V,  W,  P);
        FreeDeviceArray(10, UU, UV, UW, VV, VW, WW, PU, PV, PW, PP);
        FreeDeviceArray(1,  KT);
        FreeDeviceArray(9,  DUDX2, DUDY2, DUDZ2, DVDX2, DVDY2, DVDZ2, DWDX2, DWDY2, DWDZ2);
        FreeDeviceArray(9,  UUU, UUV, UUW, VVU, VVV, VVW, WWU, WWV, WWW);
    }

    FreeHostArray(  4,  x_h, y_h, z_h, xi_h);
    FreeDeviceArray(4,  x_d, y_d, z_d, xi_d);
    // GILBM 度量項
    FreeHostArray(  2,  dk_dz_h, dk_dy_h);
    FreeDeviceArray(2,  dk_dz_d, dk_dy_d);
    FreeHostArray(  1,  delta_zeta_h);
    FreeDeviceArray(1,  delta_zeta_d);
    // Phase 4 LTS
    FreeHostArray(  3,  dt_local_h, tau_local_h, tau_dt_product_h);
    FreeDeviceArray(3,  dt_local_d, tau_local_d, tau_dt_product_d);

    for( int i = 0; i < 3; i++ ){
        FreeHostArray(  3,  Xdep_h[i], Ydep_h[i], Zdep_h[i]);
        FreeDeviceArray(3,  Xdep_d[i], Ydep_d[i], Zdep_d[i]);
    }

    CHECK_CUDA( cudaFreeHost( Ub_avg_h ) );
    CHECK_CUDA( cudaFree( Ub_avg_d ) );

    CHECK_CUDA( cudaFreeHost( Force_h ) );
    CHECK_CUDA( cudaFree( Force_d ) );
    
    CHECK_CUDA( cudaStreamDestroy( stream0 ) );
    CHECK_CUDA( cudaStreamDestroy( stream1 ) );
    CHECK_CUDA( cudaStreamDestroy( stream2 ) );
    for( int i = 0; i < 2; i++ )
        CHECK_CUDA( cudaStreamDestroy( tbsum_stream[i] ) );

    CHECK_CUDA( cudaEventDestroy( start  ) );
    CHECK_CUDA( cudaEventDestroy( stop   ) );
    CHECK_CUDA( cudaEventDestroy( start1 ) );
    CHECK_CUDA( cudaEventDestroy( stop1  ) );
}

#endif
