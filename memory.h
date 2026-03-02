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
        AllocateDeviceArray(nBytes, 4,  &U,  &V,  &W,  &P);
        CHECK_CUDA(cudaMemset(U, 0, nBytes));  CHECK_CUDA(cudaMemset(V, 0, nBytes));
        CHECK_CUDA(cudaMemset(W, 0, nBytes));  CHECK_CUDA(cudaMemset(P, 0, nBytes));

        AllocateDeviceArray(nBytes, 10, &UU, &UV, &UW, &VV, &VW, &WW, &PU, &PV, &PW, &PP);
        CHECK_CUDA(cudaMemset(UU, 0, nBytes)); CHECK_CUDA(cudaMemset(UV, 0, nBytes));
        CHECK_CUDA(cudaMemset(UW, 0, nBytes)); CHECK_CUDA(cudaMemset(VV, 0, nBytes));
        CHECK_CUDA(cudaMemset(VW, 0, nBytes)); CHECK_CUDA(cudaMemset(WW, 0, nBytes));
        CHECK_CUDA(cudaMemset(PU, 0, nBytes)); CHECK_CUDA(cudaMemset(PV, 0, nBytes));
        CHECK_CUDA(cudaMemset(PW, 0, nBytes)); CHECK_CUDA(cudaMemset(PP, 0, nBytes));

        AllocateDeviceArray(nBytes, 1,  &KT);
        CHECK_CUDA(cudaMemset(KT, 0, nBytes));

        AllocateDeviceArray(nBytes, 9,  &DUDX2, &DUDY2, &DUDZ2, &DVDX2, &DVDY2, &DVDZ2, &DWDX2, &DWDY2, &DWDZ2);
        CHECK_CUDA(cudaMemset(DUDX2, 0, nBytes)); CHECK_CUDA(cudaMemset(DUDY2, 0, nBytes));
        CHECK_CUDA(cudaMemset(DUDZ2, 0, nBytes)); CHECK_CUDA(cudaMemset(DVDX2, 0, nBytes));
        CHECK_CUDA(cudaMemset(DVDY2, 0, nBytes)); CHECK_CUDA(cudaMemset(DVDZ2, 0, nBytes));
        CHECK_CUDA(cudaMemset(DWDX2, 0, nBytes)); CHECK_CUDA(cudaMemset(DWDY2, 0, nBytes));
        CHECK_CUDA(cudaMemset(DWDZ2, 0, nBytes));

    	AllocateDeviceArray(nBytes, 9,  &UUU,   &UUV,   &UUW,   &VVU,   &VVV,   &VVW,   &WWU,   &WWV,   &WWW);
        CHECK_CUDA(cudaMemset(UUU, 0, nBytes)); CHECK_CUDA(cudaMemset(UUV, 0, nBytes));
        CHECK_CUDA(cudaMemset(UUW, 0, nBytes)); CHECK_CUDA(cudaMemset(VVU, 0, nBytes));
        CHECK_CUDA(cudaMemset(VVV, 0, nBytes)); CHECK_CUDA(cudaMemset(VVW, 0, nBytes));
        CHECK_CUDA(cudaMemset(WWU, 0, nBytes)); CHECK_CUDA(cudaMemset(WWV, 0, nBytes));
        CHECK_CUDA(cudaMemset(WWW, 0, nBytes));
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
    // GILBM 預計算 RK2 ζ 方向位移 [19 * NYD6 * NZ6] (host-only, kernel uses precomputed weights)
    nBytes = 19 * NYD6 * NZ6 * sizeof(double);
    AllocateHostArray(  nBytes, 1, &delta_zeta_h);
    // Part A: space-varying δη/δξ with dt_local [19 * NYD6 * NZ6] (host-only)
    AllocateHostArray(  nBytes, 2, &delta_eta_local_h, &delta_xi_local_h);
    // Part B: precomputed Lagrange weights [19 * 7 * NYD6 * NZ6] (q outermost, c middle)
    nBytes = 7 * 19 * NYD6 * NZ6 * sizeof(double);
    AllocateHostArray(  nBytes, 3, &lagrange_eta_h, &lagrange_xi_h, &lagrange_zeta_h);
    AllocateDeviceArray(nBytes, 3, &lagrange_eta_d, &lagrange_xi_d, &lagrange_zeta_d);
    // Precomputed stencil base k [NZ6] (int array, wall-clamped, direct k indexing)
    CHECK_CUDA( cudaMallocHost((void**)&bk_precomp_h, NZ6 * sizeof(int)) );
    CHECK_CUDA( cudaMalloc(&bk_precomp_d, NZ6 * sizeof(int)) );
    // Phase 4 LTS: local dt, omega fields [NYD6 * NZ6]
    nBytes = NYD6 * NZ6 * sizeof(double);
    AllocateHostArray(  nBytes, 3, &dt_local_h, &omega_local_h, &omegadt_local_h);
    AllocateDeviceArray(nBytes, 2, &dt_local_d, &omega_local_d);

    // GILBM two-pass: f_pc_d [19*343*grid], feq_d [19*grid], omegadt_local_d [grid]
    {
        size_t grid_size = (size_t)NX6 * NYD6 * NZ6;
        size_t f_pc_bytes = 19ULL * 343ULL * grid_size * sizeof(double);
        CHECK_CUDA( cudaMalloc(&f_pc_d, f_pc_bytes) );
        CHECK_CUDA( cudaMemset(f_pc_d, 0, f_pc_bytes) );

        size_t feq_bytes = 19ULL * grid_size * sizeof(double);
        CHECK_CUDA( cudaMalloc(&feq_d, feq_bytes) );
        CHECK_CUDA( cudaMemset(feq_d, 0, feq_bytes) );

        size_t omega_bytes = grid_size * sizeof(double);
        CHECK_CUDA( cudaMalloc(&omegadt_local_d, omega_bytes) );
        CHECK_CUDA( cudaMemset(omegadt_local_d, 0, omega_bytes) );
    }

    // Time-average accumulation (GPU-side): u_tavg(spanwise), v_tavg(streamwise), w_tavg(wall-normal)
    {
        size_t tavg_bytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
        CHECK_CUDA( cudaMalloc(&u_tavg_d, tavg_bytes) );
        CHECK_CUDA( cudaMalloc(&v_tavg_d, tavg_bytes) );
        CHECK_CUDA( cudaMalloc(&w_tavg_d, tavg_bytes) );
        CHECK_CUDA( cudaMemset(u_tavg_d, 0, tavg_bytes) );
        CHECK_CUDA( cudaMemset(v_tavg_d, 0, tavg_bytes) );
        CHECK_CUDA( cudaMemset(w_tavg_d, 0, tavg_bytes) );
    }

    nBytes = NZ6 * sizeof(double);
    CHECK_CUDA( cudaMallocHost( (void**)&xi_h, nBytes ) );
    CHECK_CUDA( cudaMalloc( &xi_d, nBytes ) );

    nBytes = NZ6 * NX6 * sizeof(double);
    CHECK_CUDA( cudaMallocHost( (void**)&Ub_avg_h, nBytes ) );
    CHECK_CUDA( cudaMalloc( &Ub_avg_d, nBytes ) );
    CHECK_CUDA( cudaMemset(Ub_avg_d, 0, nBytes) );
    

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
        
    FreeHostArray(  4,  rho_h_p, u_h_p, v_h_p, w_h_p);

    for( int i = 0; i < 19; i++ ) {
        CHECK_CUDA( cudaFree( ft[i] ) );
        CHECK_CUDA( cudaFree( fd[i] ) );
    }
    FreeDeviceArray(4,  rho_d, u, v, w);

    if( TBSWITCH ) {
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
    // Part A: space-varying δη/δξ (host-only)
    FreeHostArray(  2,  delta_eta_local_h, delta_xi_local_h);
    // Part B: precomputed Lagrange weights
    FreeHostArray(  3,  lagrange_eta_h, lagrange_xi_h, lagrange_zeta_h);
    FreeDeviceArray(3,  lagrange_eta_d, lagrange_xi_d, lagrange_zeta_d);
    // Precomputed stencil base k
    CHECK_CUDA( cudaFreeHost(bk_precomp_h) );
    CHECK_CUDA( cudaFree(bk_precomp_d) );
    // Phase 4 LTS
    FreeHostArray(  3,  dt_local_h, omega_local_h, omegadt_local_h);
    FreeDeviceArray(2,  dt_local_d, omega_local_d);
    // GILBM two-pass arrays
    FreeDeviceArray(3,  f_pc_d, feq_d, omegadt_local_d);
    // Time-average accumulation (GPU)
    FreeDeviceArray(3,  u_tavg_d, v_tavg_d, w_tavg_d);

    for( int i = 0; i < 3; i++ ){
        FreeHostArray(  3,  Xdep_h[i], Ydep_h[i], Zdep_h[i]);
        FreeDeviceArray(3,  Xdep_d[i], Ydep_d[i], Zdep_d[i]);
    }

    CHECK_CUDA( cudaFreeHost( Ub_avg_h ) );
    CHECK_CUDA( cudaFree( Ub_avg_d ) );

    CHECK_CUDA( cudaFreeHost( Force_h ) );
    CHECK_CUDA( cudaFree( Force_d ) );
    CHECK_CUDA( cudaFreeHost( rho_modify_h ) );
    CHECK_CUDA( cudaFree( rho_modify_d ) );
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
