#ifndef INITIALIZATION_FILE
#define INITIALIZATION_FILE

#include "initializationTool.h"

void InitialUsingDftFunc() {
    double e[19][3]={{0.0,0.0,0.0},{1.0,0.0,0.0},{-1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,-1.0,0.0},{0.0,0.0,1.0},{0.0,0.0,-1.0},
					{1.0,1.0,0.0},{-1.0,1.0,0.0},{1.0,-1.0,0.0},{-1.0,-1.0,0.0},{1.0,0.0,1.0},{-1.0,0.0,1.0},{1.0,0.0,-1.0},
					{-1.0,0.0,-1.0},{0.0,1.0,1.0},{0.0,-1.0,1.0},{0.0,1.0,-1.0},{0.0,-1.0,-1.0}}; 
    double W[19]={(1.0/3.0),(1.0/18.0),(1.0/18.0),(1.0/18.0),(1.0/18.0),(1.0/18.0),(1.0/18.0),(1.0/36.0),(1.0/36.0)
				  ,(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0)
				  ,(1.0/36.0)};

    double udot;

    for( int k = 0; k < NZ6;  k++ ) {
    for( int j = 0; j < NYD6; j++ ) {
    for( int i = 0; i < NX6;  i++ ) {
    
        const int index = j*NX6*NZ6 + k*NX6 + i;

        /* Initial condition for 3-D Taylor-Green vortex */
        //rho_h_p[index] = 1.0 + 3.0*U_0*U_0/16.0*(cos(2.0*2.0*pi*x_h[i]/LX)+cos(2.0*2.0*pi*y_h[j]/LY))*(2.0*cos(2.0*2.0*pi*z_h[k]/LZ));
        //u_h_p[index] = U_0*sin(2.0*pi*x_h[i]/LX)*cos(2.0*pi*y_h[j]/LY)*cos(2.0*pi*z_h[k]/LZ);
        //v_h_p[index] = -U_0*cos(2.0*pi*x_h[i]/LX)*sin(2.0*pi*y_h[j]/LY)*cos(2.0*pi*z_h[k]/LZ);
        //w_h_p[index] = 0.0;

        /* Initial condition for channel flow && periodic hills */
        rho_h_p[index] = 1.0;
        u_h_p[index] = 0.0;
        v_h_p[index] = 0.0;
        w_h_p[index] = 0.0;

        udot = u_h_p[index]*u_h_p[index] + v_h_p[index]*v_h_p[index] + w_h_p[index]*w_h_p[index];

        fh_p[0][index] = W[0]*rho_h_p[index]*(1.0-1.5*udot);
        for( int dir = 1; dir <= 18; dir++ ) {
            fh_p[dir][index] = W[dir] * rho_h_p[index] *( 1.0 + 
                                                          3.0 *( e[dir][0] * u_h_p[index] + e[dir][1] * v_h_p[index] + e[dir][2] * w_h_p[index])+ 
                                                          4.5 *( e[dir][0] * u_h_p[index] + e[dir][1] * v_h_p[index] + e[dir][2] * w_h_p[index] )*( e[dir][0] * u_h_p[index] + e[dir][1] * v_h_p[index] + e[dir][2] * w_h_p[index] )- 
                                                          1.5*udot );
        }
    
    }}}

    Force_h[0] =  (8.0*niu*Uref)/(LZ*LZ)*5.0; //0.0001;
    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );

}

void GenerateMesh_X() {
    double dx;
    int bfr = 3;

    if( Uniform_In_Xdir ){
		dx = LX / (double)(NX6-2*bfr-1);
		for( int i = 0; i < NX6; i++ ){
			x_h[i]  = dx*((double)(i-bfr));
		}
	} else {
        printf("Mesh needs to be uniform in periodic hill problem, exit...\n");
        exit(0);
    }

    FILE *meshX;
	meshX = fopen("meshX.DAT","w");
	for( int i = 0 ; i < NX6 ; i++ ){
		fprintf( meshX, "%.15lf\n", x_h[i]);
	}
	fclose(meshX);

    CHECK_CUDA( cudaMemcpy(x_d,  x_h,  NX6*sizeof(double), cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaDeviceSynchronize() );
}

void GenerateMesh_Y() {
    double dy;
    double y_global[NY6];
    int bfr = 3;

    if( Uniform_In_Ydir ){
        dy = LY / (double)(NY6-2*bfr-1);
        for( int i = 0; i < NY6; i++ ){
            y_global[i] = dy * ((double)(i-bfr));
        }

        for( int j = 0; j < NYD6; j++ ) {
            int j_global = myid * (NYD6-2*bfr-1) + j;
            y_h[j] = y_global[j_global];
        }

    } else {
        printf("Mesh needs to be uniform in periodic hill problem, exit...\n");
        exit(0);
    }

    FILE *meshY;
	meshY = fopen("meshY.DAT","w");
	for( int j = 0 ; j < NY6 ; j++ ){
		fprintf( meshY, "%.15lf\n", y_global[j]);
	}
	fclose(meshY);

    CHECK_CUDA( cudaMemcpy(y_d,  y_h,  NYD6*sizeof(double), cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaDeviceSynchronize() );
}

void GenerateMesh_Z() {
    int bfr = 3;

    if( Uniform_In_Zdir ){
        printf("Mesh needs to be non-uniform in z-direction in periodic hill problem, exit...\n");
        exit(0);
    }

    double a = GetNonuniParameter();

    // Buffer=3: tanh 起點=壁面(k=3)=Hill(y), 終點=(k=NZ6-4)=LZ
    for( int j = 0; j < NYD6; j++ ){
        double total = LZ - HillFunction( y_h[j] );
        for( int k = bfr; k < NZ6-bfr; k++ ){  // k=3..NZ6-4
            z_h[j*NZ6+k] = tanhFunction_wall( total, a, (k-3), (NZ6-7) ) +
                           HillFunction( y_h[j] );
        }
        // k=3 = Hill(y) (tanh_wall at j=0 = 0),  k=NZ6-4 = LZ (tanh_wall at j=N = total)

        // 外插 buffer 層 (k=2 和 k=NZ6-3)
        z_h[j*NZ6+2]       = 2.0 * z_h[j*NZ6+3] - z_h[j*NZ6+4];
        z_h[j*NZ6+(NZ6-3)] = 2.0 * z_h[j*NZ6+(NZ6-4)] - z_h[j*NZ6+(NZ6-5)];

        // Ghost z values (linear extrapolation)
        z_h[j*NZ6+1]       = 2.0 * z_h[j*NZ6+2] - z_h[j*NZ6+3];
        z_h[j*NZ6+0]       = 2.0 * z_h[j*NZ6+1] - z_h[j*NZ6+2];
        z_h[j*NZ6+(NZ6-2)] = 2.0 * z_h[j*NZ6+(NZ6-3)] - z_h[j*NZ6+(NZ6-4)];
        z_h[j*NZ6+(NZ6-1)] = 2.0 * z_h[j*NZ6+(NZ6-2)] - z_h[j*NZ6+(NZ6-3)];
    }

    // 計算座標 xi_h: tanh_wall 映射 k=3→0, k=NZ6-4→LXi
    for( int k = bfr; k < NZ6-bfr; k++ ){
        xi_h[k] = tanhFunction_wall( LXi, a, (k-3), (NZ6-7) );
    }
    // Buffer xi values (linear extrapolation)
    xi_h[2]     = 2.0 * xi_h[3] - xi_h[4];
    xi_h[NZ6-3] = 2.0 * xi_h[NZ6-4] - xi_h[NZ6-5];


    double y_global[NY6];
    double z_global[NY6*NZ6];
    for( int j = 0; j < NY6; j++ ){
        double dy = LY / (double)(NY6-2*bfr-1);
        y_global[j] = dy * ((double)(j-bfr));
        double total = LZ - HillFunction( y_global[j] );
        for( int k = bfr; k < NZ6-bfr; k++ ){
            z_global[j*NZ6+k] = tanhFunction_wall( total, a, (k-3), (NZ6-7) ) +
                                HillFunction( y_global[j] );
        }

        // 外插 buffer 層
        z_global[j*NZ6+2]       = 2.0 * z_global[j*NZ6+3] - z_global[j*NZ6+4];
        z_global[j*NZ6+(NZ6-3)] = 2.0 * z_global[j*NZ6+(NZ6-4)] - z_global[j*NZ6+(NZ6-5)];

        // Ghost z values (linear extrapolation)
        z_global[j*NZ6+1]       = 2.0 * z_global[j*NZ6+2] - z_global[j*NZ6+3];
        z_global[j*NZ6+0]       = 2.0 * z_global[j*NZ6+1] - z_global[j*NZ6+2];
        z_global[j*NZ6+(NZ6-2)] = 2.0 * z_global[j*NZ6+(NZ6-3)] - z_global[j*NZ6+(NZ6-4)];
        z_global[j*NZ6+(NZ6-1)] = 2.0 * z_global[j*NZ6+(NZ6-2)] - z_global[j*NZ6+(NZ6-3)];
    }

    FILE *meshZ;
    meshZ = fopen("meshZ.DAT","w");
    for( int k = 0; k < NZ6; k++ ){
    for( int j = 0; j < NY6; j++ ){
         
        fprintf( meshZ, "%.15lf\n", z_global[j*NZ6+k] );
    }}
    fclose( meshZ );

    FILE *meshXi;
    meshXi = fopen("meshXi.DAT","w");
    for( int k = 0; k < NZ6; k++ ){
        fprintf( meshXi, "%.15lf\n", xi_h[k] );
    }
    fclose( meshXi );

    CHECK_CUDA( cudaMemcpy(z_d,   z_h,   NZ6*NYD6*sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(xi_d,  xi_h,  NZ6*sizeof(double), cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaDeviceSynchronize() );
}

// NOTE: GetXiParameter, GetIntrplParameter_X/Y/Xi 已移除
// (舊 ISLBM 插值係數計算，GILBM 改用 7-point Lagrange 插值，不再需要)

#endif
