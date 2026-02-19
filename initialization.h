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

    for( int j = 0; j < NYD6; j++ ){
        double total = LZ - HillFunction( y_h[j] ) - minSize;
        for( int k = bfr; k < NZ6-bfr; k++ ){
            z_h[j*NZ6+k] = tanhFunction( total, minSize, a, (k-3), (NZ6-7) ) + 
                           HillFunction( y_h[j] );
        }
        z_h[j*NZ6+2] = HillFunction( y_h[j] );
        z_h[j*NZ6+(NZ6-3)] = (double)LZ;

        // Ghost z values (linear extrapolation, for difference stencils at k=2 and k=NZ6-3)
        z_h[j*NZ6+1]       = 2.0 * z_h[j*NZ6+2] - z_h[j*NZ6+3]; //透過線性外插取得bufferlayer 層的位置
        z_h[j*NZ6+(NZ6-2)] = 2.0 * z_h[j*NZ6+(NZ6-3)] - z_h[j*NZ6+(NZ6-4)];
    }

    for( int k = bfr; k < NZ6-bfr; k++ ){
        xi_h[k] = tanhFunction( LXi, minSize, a, (k-3), (NZ6-7) ) - minSize/2.0;
    }
    // Wall xi values (linear extrapolation for k=2 and k=NZ6-3 computation points)
    xi_h[2]     = 2.0 * xi_h[3] - xi_h[4];
    xi_h[NZ6-3] = 2.0 * xi_h[NZ6-4] - xi_h[NZ6-5];


    double y_global[NY6];
    double z_global[NY6*NZ6];
    for( int j = 0; j < NY6; j++ ){
        double dy = LY / (double)(NY6-2*bfr-1);
        y_global[j] = dy * ((double)(j-bfr));
        double total = LZ - HillFunction( y_global[j] ) - minSize;
        for( int k = bfr; k < NZ6-bfr; k++ ){
            z_global[j*NZ6+k] = tanhFunction( total, minSize, a, (k-3), (NZ6-7) ) + 
                                HillFunction( y_global[j] );
        }
        z_global[j*NZ6+2] = HillFunction( y_global[j] );
        z_global[j*NZ6+(NZ6-3)] = (double)LZ;

        // Ghost z values (linear extrapolation)
        z_global[j*NZ6+1]       = 2.0 * z_global[j*NZ6+2] - z_global[j*NZ6+3];
        z_global[j*NZ6+(NZ6-2)] = 2.0 * z_global[j*NZ6+(NZ6-3)] - z_global[j*NZ6+(NZ6-4)];
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




#endif
