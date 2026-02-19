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
        z_h[j*NZ6+1]       = 2.0 * z_h[j*NZ6+2] - z_h[j*NZ6+3];
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

void GetXiParameter(
    double *XiPara_h[7],    double pos_z,       double pos_y,
    double *Pos_xi,         int IdxToStore,     int k  )
{
    double L = LZ - HillFunction(pos_y) - minSize;
    double pos_xi = LXi * (pos_z - (HillFunction(pos_y)+minSize/2.0)) / L;
    double a = GetNonuniParameter();
    //double pos_xi = atanh((pos_z-HillFunction(pos_y)-minSize/2.0-L/2.0)/((L/2.0)/a))/log((1.0+a)/(1.0-a))*2.0;

    if( k >= 3 && k <= 6 ){
        GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, 3 );
    } else if ( k >= NZ6-7 && k <= NZ6-4 ) {
        GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, NZ6-10 );
    } else {
        GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, k-3 );
    }    
}

void GetIntrplParameter_X() {

    for( int i = 3; i < NX6-3; i++ ){
        GetParameter_6th( XPara0_h, x_h[i]-minSize, x_h, i, i-3 );
        GetParameter_6th( XPara2_h, x_h[i]+minSize, x_h, i, i-3 );
    }

    for( int i = 0; i < 7; i++ ){
        CHECK_CUDA( cudaMemcpy(XPara0_d[i], XPara0_h[i], NX6*sizeof(double), cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XPara2_d[i], XPara2_h[i], NX6*sizeof(double), cudaMemcpyHostToDevice) );
    }
    CHECK_CUDA( cudaDeviceSynchronize() );
}

void GetIntrplParameter_Y() {

    for( int i = 3; i < NYD6-3; i++ ){
        GetParameter_6th( YPara0_h, y_h[i]-minSize, y_h, i, i-3 );
        GetParameter_6th( YPara2_h, y_h[i]+minSize, y_h, i, i-3 );
    }

    for( int i = 0; i < 7; i++ ){
        CHECK_CUDA( cudaMemcpy(YPara0_d[i], YPara0_h[i], NYD6*sizeof(double), cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(YPara2_d[i], YPara2_h[i], NYD6*sizeof(double), cudaMemcpyHostToDevice) );
    }
    CHECK_CUDA( cudaDeviceSynchronize() );
}

void GetIntrplParameter_Xi() {

    for( int j = 3; j < NYD6-3; j++ ){
    for( int k = 3; k < NZ6-3;  k++ ){
        GetXiParameter( XiParaF3_h,  z_h[j*NZ6+k],         y_h[j]-minSize, xi_h, j*NZ6+k, k );
        GetXiParameter( XiParaF4_h,  z_h[j*NZ6+k],         y_h[j]+minSize, xi_h, j*NZ6+k, k );
        GetXiParameter( XiParaF5_h,  z_h[j*NZ6+k]-minSize, y_h[j],         xi_h, j*NZ6+k, k );
        GetXiParameter( XiParaF6_h,  z_h[j*NZ6+k]+minSize, y_h[j],         xi_h, j*NZ6+k, k );
        GetXiParameter( XiParaF15_h, z_h[j*NZ6+k]-minSize, y_h[j]-minSize, xi_h, j*NZ6+k, k );
        GetXiParameter( XiParaF16_h, z_h[j*NZ6+k]-minSize, y_h[j]+minSize, xi_h, j*NZ6+k, k );
        GetXiParameter( XiParaF17_h, z_h[j*NZ6+k]+minSize, y_h[j]-minSize, xi_h, j*NZ6+k, k );
        GetXiParameter( XiParaF18_h, z_h[j*NZ6+k]+minSize, y_h[j]+minSize, xi_h, j*NZ6+k, k );
    }}

    const size_t nBytes = NYD6 * NZ6 * sizeof(double);
    for( int i = 0; i < 7; i++ ){
        CHECK_CUDA( cudaMemcpy(XiParaF3_d[i],  XiParaF3_h[i],  nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiParaF4_d[i],  XiParaF4_h[i],  nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiParaF5_d[i],  XiParaF5_h[i],  nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiParaF6_d[i],  XiParaF6_h[i],  nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiParaF15_d[i], XiParaF15_h[i], nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiParaF16_d[i], XiParaF16_h[i], nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiParaF17_d[i], XiParaF17_h[i], nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiParaF18_d[i], XiParaF18_h[i], nBytes, cudaMemcpyHostToDevice) );
    }

    CHECK_CUDA( cudaDeviceSynchronize() );
}

void BFLInitialization() {
    size_t nBytes;

    for( int k = 0; k < 2;      k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
        BFLReqF3_h[k*NYD6+j]  = IsBFLBCNeeded(y_h[j]-minSize, z_h[j*NZ6+k+3]);
        BFLReqF4_h[k*NYD6+j]  = IsBFLBCNeeded(y_h[j]+minSize, z_h[j*NZ6+k+3]);
        BFLReqF15_h[k*NYD6+j] = IsBFLBCNeeded(y_h[j]-minSize, z_h[j*NZ6+k+3]-minSize);
        BFLReqF16_h[k*NYD6+j] = IsBFLBCNeeded(y_h[j]+minSize, z_h[j*NZ6+k+3]-minSize);
    }}

    nBytes = 2 * NYD6 * sizeof(int);
    CHECK_CUDA( cudaMemcpy(BFLReqF3_d,  BFLReqF3_h,  nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(BFLReqF4_d,  BFLReqF4_h,  nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(BFLReqF15_d, BFLReqF15_h, nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(BFLReqF16_d, BFLReqF16_h, nBytes, cudaMemcpyHostToDevice) );

    CHECK_CUDA( cudaDeviceSynchronize() );

    double delta;
    for( int k = 0; k < 2;      k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
        //BFL initialization for F3, F7, and F8
        if( BFLReqF3_h[k*NYD6+j] == 1 ){
            delta = GetDeltaHorizontal(z_h[j*NZ6+k+3], y_h[j]-minSize, y_h[j], y_h[j]);
            Q3_h[k*NYD6+j]=(minSize-delta)/2.0/minSize;
            GetParameter_6th(XBFLParaF38_h, x_h[NX6/2]-delta, x_h, k*NYD6+j, NX6/2-3);  //X
            GetParameter_6th(XBFLParaF37_h, x_h[NX6/2]+delta, x_h, k*NYD6+j, NX6/2-3);  //X
            GetParameter_6th(YBFLParaF378_h, y_h[j]+delta, y_h, k*NYD6+j, j-3);     //Y
            GetXiParameter(XiBFLParaF378_h, z_h[j*NZ6+k+3], y_h[j]+delta, xi_h, k*NYD6+j, k+3);

            /* GetParameter_2nd(XBFLParaF38_h, x_h[NX6/2]-delta, x_h, k*NYD6+j, NX6/2-1);  //X
            GetParameter_2nd(XBFLParaF37_h, x_h[NX6/2]+delta, x_h, k*NYD6+j, NX6/2-1);  //X
            GetParameter_2nd(YBFLParaF378_h, y_h[j]+delta, y_h, k*NYD6+j, j-1);     //Y
            GetBFLXiParameter(XiBFLParaF378_h, z_h[j*NZ6+k+3], y_h[j]+delta, xi_h, k*NYD6+j, k+3);    */  
        }

        //BFL initialization for F4, F9, and F10
        if( BFLReqF4_h[k*NYD6+j] == 1){
            delta = GetDeltaHorizontal(z_h[j*NZ6+k+3], y_h[j]+minSize, y_h[j], y_h[j]);
            Q4_h[k*NYD6+j]=(minSize-delta)/2.0/minSize;
            GetParameter_6th(XBFLParaF49_h,  x_h[NX6/2]+delta, x_h, k*NYD6+j, NX6/2-3); //X
            GetParameter_6th(XBFLParaF410_h, x_h[NX6/2]-delta, x_h, k*NYD6+j, NX6/2-3); //X
            GetParameter_6th(YBFLParaF4910_h, y_h[j]-delta, y_h, k*NYD6+j, j-3);    //Y
            GetXiParameter(XiBFLParaF4910_h, z_h[j*NZ6+k+3], y_h[j]-delta, xi_h, k*NYD6+j, k+3);

            /* GetParameter_2nd(XBFLParaF49_h,  x_h[NX6/2]+delta, x_h, k*NYD6+j, NX6/2-1); //X
            GetParameter_2nd(XBFLParaF410_h, x_h[NX6/2]-delta, x_h, k*NYD6+j, NX6/2-1); //X
            GetParameter_2nd(YBFLParaF4910_h, y_h[j]-delta, y_h, k*NYD6+j, j-1);    //Y
            GetBFLXiParameter(XiBFLParaF4910_h, z_h[j*NZ6+k+3], y_h[j]-delta, xi_h, k*NYD6+j, k+3); */
        }

        //BFL initialization for F15
        if( BFLReqF15_h[k*NYD6+j] == 1 ){
            delta = GetDelta45Degree(z_h[j*NZ6+k+3], y_h[j], y_h[j], y_h[j]-minSize);
            Q15_h[k*NYD6+j]=(minSize-delta)/2.0/minSize;
            GetParameter_6th(YBFLParaF15_h, y_h[j]+delta, y_h, k*NYD6+j, j-3);
            GetXiParameter(XiBFLParaF15_h, z_h[j*NZ6+k+3]+delta, y_h[j]+delta, xi_h, k*NYD6+j, k+3);

            /* GetParameter_2nd(YBFLParaF15_h, y_h[j]+delta, y_h, k*NYD6+j, j-1);
            GetBFLXiParameter(XiBFLParaF15_h, z_h[j*NZ6+k+3]+delta, y_h[j]+delta, xi_h, k*NYD6+j, k+3); */
        }

        //BFL initialization for F16
        if( BFLReqF16_h[k*NYD6+j] == 1 ){
            delta = GetDelta45Degree(z_h[j*NZ6+k+3], y_h[j], y_h[j], y_h[j]+minSize);
            Q16_h[k*NYD6+j]=(minSize-delta)/2.0/minSize;
            GetParameter_6th(YBFLParaF16_h, y_h[j]-delta, y_h, k*NYD6+j, j-3);
            GetXiParameter(XiBFLParaF16_h, z_h[j*NZ6+k+3]+delta, y_h[j]-delta, xi_h, k*NYD6+j, k+3);

            /* GetParameter_2nd(YBFLParaF16_h, y_h[j]-delta, y_h, k*NYD6+j, j-1);
            GetBFLXiParameter(XiBFLParaF16_h, z_h[j*NZ6+k+3]+delta, y_h[j]-delta, xi_h, k*NYD6+j, k+3); */
        }
    }}
    
    nBytes = 2 * NYD6 * sizeof(double);
    CHECK_CUDA( cudaMemcpy(Q3_d,    Q3_h,    nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(Q4_d,    Q4_h,    nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(Q15_d,   Q15_h,   nBytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(Q16_d,   Q16_h,   nBytes, cudaMemcpyHostToDevice) );

    for( int i = 0; i < 7; i++ ){
        CHECK_CUDA( cudaMemcpy(XBFLParaF37_d[i],   XBFLParaF37_h[i],   nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XBFLParaF38_d[i],   XBFLParaF38_h[i],   nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(YBFLParaF378_d[i],  YBFLParaF378_h[i],  nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiBFLParaF378_d[i], XiBFLParaF378_h[i], nBytes, cudaMemcpyHostToDevice) );

        CHECK_CUDA( cudaMemcpy(XBFLParaF49_d[i],    XBFLParaF49_h[i],    nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XBFLParaF410_d[i],   XBFLParaF410_h[i],   nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(YBFLParaF4910_d[i],  YBFLParaF4910_h[i],  nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiBFLParaF4910_d[i], XiBFLParaF4910_h[i], nBytes, cudaMemcpyHostToDevice) );

        CHECK_CUDA( cudaMemcpy(YBFLParaF15_d[i],  YBFLParaF15_h[i],  nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiBFLParaF15_d[i], XiBFLParaF15_h[i], nBytes, cudaMemcpyHostToDevice) );

        CHECK_CUDA( cudaMemcpy(YBFLParaF16_d[i],  YBFLParaF16_h[i],  nBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(XiBFLParaF16_d[i], XiBFLParaF16_h[i], nBytes, cudaMemcpyHostToDevice) );
    }

    CHECK_CUDA( cudaDeviceSynchronize() );
}

#endif
