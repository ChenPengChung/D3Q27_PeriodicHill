#ifndef FILEIO_FILE
#define FILEIO_FILE

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>


void wirte_ASCII_of_str(char * str, FILE *file);
//function: check whether a file exist
//input: 	file relative path. ex: "./backup/u0.bkp"
//return: 	if exsit return 1; else return 0
bool FileExist(const char *fileName) {
    std::ifstream file(fileName);
    return file.good();
}

//function: check whether a dir exist or not, if not, creat it
//input:	dir name. ex: "result"
void ExistOrCreateDir(const char *dir) {
	std::string path(dir);
	path = "./" + path;
	if (access(path.c_str(), F_OK) != 0) {
		if (mkdir(path.c_str(), S_IRWXU) == 0)
			std::cout << "folder " << path << " not exist, created"<< std::endl;
	}
}

//check whether the folder "result", "backup", "statistics"
//exist or not. if not, create them
void PreCheckDir() {
	ExistOrCreateDir("result");
	ExistOrCreateDir("backup");
	//ExistOrCreateDir("./backup/0");
	//ExistOrCreateDir("./backup/1");
	ExistOrCreateDir("statistics");
	//ExistOrCreateDir("./statistics/monitor");
    if ( TBSWITCH ) {
		const int num_files = 35;
		std::string name[num_files] = {"U","V","W","P","UU","UV","UW","VV","VW","WW","PU","PV","PW","KT","DUDX2","DUDY2","DUDZ2","DVDX2","DVDY2","DVDZ2","DWDX2","DWDY2","DWDZ2","UUU","UUV","UUW","VVU","VVV","VVW","WWU","WWV","WWW","OMEGA_X","OMEGA_Y","OMEGA_Z"};
		for( int i = 0; i < num_files; i++ ) {
			std::string fname = "./statistics/" + name[i];
			ExistOrCreateDir(fname.c_str());
		}
	}
}

void OutputData(
    double *arr_h,
    const char *fname,      const int myid  )
{
    char path[100];
    sprintf( path, "./result/%s_%d.bin", fname, myid );

    FILE *data;
    if((data = fopen(path, "wb")) == NULL) {
        printf("Output data error, exit...\n");
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    fwrite( arr_h, sizeof(double), NX6*NZ6*NYD6, data );
    fclose( data );
}

void fileIO_velocity() {
    FILE *outptr1;
    char result[50];

    sprintf( result, "%dx%dx%d velocity_%d.dat",(int)NX6, (int)NYD6, (int)NZ6, myid );
    outptr1 = fopen( result, "w" );
    fprintf( outptr1, "VARIABLES = \"X\", \"Y\", \"Z\", \"U\", \"V\", \"W\"\n" );
    fprintf( outptr1, "ZONE T = \"velocity\", F = POINT\n" );
    fprintf( outptr1, "I = %d, J = %d, K = %d\n", NX6-6, NYD6-6, NZ6-6 );

    for( int k = 3; k < NZ6-3; k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        int index = j*NZ6*NX6 + k*NX6 + i;
        fprintf( outptr1, "%.4lf\t %.4lf\t %.4lf\t ",x_h[i], y_h[j], z_h[j*NZ6+k] );
        fprintf( outptr1, "%.15lf\t %.15lf\t %.15lf\n", u_h_p[index], v_h_p[index], w_h_p[index] );
    }}}

    fclose( outptr1 );

    FILE *outptr2;
    sprintf( result, "%dx%d Parameter_%d.dat",(int)NYD6, (int)NZ6, myid );
    outptr2 = fopen( result, "w" );
    fprintf( outptr2, "VARIABLES = \"Y\", \"Z\", \"BFLF3\", \"BFLF4\", \"BFLF15\", \"BFLF16\", \"Q3\", \"Q4\", \"Q15\", \"Q16\"\n" );
    fprintf( outptr2, "ZONE T = \"parameter\", F = POINT\n" );
    fprintf( outptr2, "J = %d, K = %d\n", NYD6-6, NZ6-6 );

    for( int k = 3; k < NZ6-3; k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
        //int index = j*NZ6*NX6 + k*NX6 + i;
        int f3 = 0, f4 = 0, f15 = 0, f16 = 0;
        double q3 = 0.0, q4 = 0.0, q15 = 0.0, q16 = 0.0;
        if( k == 3 || k == 4 ){
            f3 = BFLReqF3_h[(k-3)*NYD6+j];
            f4 = BFLReqF4_h[(k-3)*NYD6+j];
            f15 = BFLReqF15_h[(k-3)*NYD6+j];
            f16 = BFLReqF16_h[(k-3)*NYD6+j];
            q3 = Q3_h[(k-3)*NYD6+j];
            q4 = Q4_h[(k-3)*NYD6+j];
            q15 = Q15_h[(k-3)*NYD6+j];;
            q16 = Q16_h[(k-3)*NYD6+j];;
        }
        fprintf( outptr2, "%.4lf\t %.4lf\t ",y_h[j], z_h[j*NZ6+k] );
        fprintf( outptr2, "%d\t %d\t %d\t %d\t %lf\t %lf\t %lf\t %lf\n", f3, f4, f15, f16, q3, q4, q15, q16 );
    }}

    fclose( outptr2 );


    /* FILE *fout1,*fout2,*fout3,*fout4,*fout5,*fout6,*fout7,*fout8,*fout9;

    fout1 = fopen("velocity_y=0.DAT","w+t");		
	fprintf( fout1, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout1, "ZONE T=\"y=0\", F=POINT\n", Re );
	fprintf( fout1, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){
		int j = 0; 
		int idx = 0.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	

		fprintf( fout1, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout1);

    fout2 = fopen("velocity_y=1.DAT","w+t");		
	fprintf( fout2, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout2, "ZONE T=\"y=1\", F=POINT\n", Re );
	fprintf( fout2, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){

		int j = 1.0/9.0*NY6 ;
		int idx = 1.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	

		fprintf( fout2, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout2);

    fout3 = fopen("velocity_y=2.DAT","w+t");		
	fprintf( fout3, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout3, "ZONE T=\"y=2\", F=POINT\n", Re );
	fprintf( fout3, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){

		int j = 2.0/9.0*NY6 ;
		int idx = 2.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	

		fprintf( fout3, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout3);

    fout4 = fopen("velocity_y=3.DAT","w+t");		
	fprintf( fout4, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout4, "ZONE T=\"y=3\", F=POINT\n", Re );
	fprintf( fout4, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){
    
		int j = 3.0/9.0*NY6 ;
		int idx = 3.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	

		fprintf( fout4, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout4);

    fout5 = fopen("velocity_y=4.DAT","w+t");		
	fprintf( fout5, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout5, "ZONE T=\"y=4\", F=POINT\n", Re );
	fprintf( fout5, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){

		int j = 4.0/9.0*NY6 ;
		int idx = 4.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	

		fprintf( fout5, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout5);

    fout6 = fopen("velocity_y=5.DAT","w+t");		
	fprintf( fout6, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout6, "ZONE T=\"y=5\", F=POINT\n", Re );
	fprintf( fout6, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){

		int j = 5.0/9.0*NY6 ;
		int idx = 5.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	

		fprintf( fout6, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout6);

    fout7 = fopen("velocity_y=6.DAT","w+t");		
	fprintf( fout7, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout7, "ZONE T=\"y=6\", F=POINT\n", Re );
	fprintf( fout7, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){

		int j = 6.0/9.0*NY6 ;
		int idx = 6.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	
		fprintf( fout7, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout7);

    fout8 = fopen("velocity_y=7.DAT","w+t");		
	fprintf( fout8, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout8, "ZONE T=\"y=7\", F=POINT\n", Re );
	fprintf( fout8, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){

		int j = 7.0/9.0*NY6 ;
		int idx = 7.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	
		fprintf( fout8, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout8);

    fout9 = fopen("velocity_y=8.DAT","w+t");		
	fprintf( fout9, "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n" );
	fprintf( fout9, "ZONE T=\"y=8\", F=POINT\n", Re );
	fprintf( fout9, "K=%d\n",NZ6-6 );	
	for ( int k = 3 ; k < NZ6-3 ; k++ ){

		int j = 8.0/9.0*NY6 ;
		int idx = 8.0/9.0*NY6*NX6*NZ6  + k*NX6 + NX6/2.0;	
		fprintf( fout9, "%.5f %.15f %.15f %.15f \n", 
		z_h[j*NZ6+k], (u_h_p[idx]/Uref),(v_h_p[idx]/Uref),4*(w_h_p[idx]/Uref));
        
	}
	fclose(fout9); */
    /* if( myid == 0 ){
        for( int k = 3; k < NZ6-3; k++ ){
            int i = NX6/2, j = NYD6/2;
            int index = j*NZ6*NX6 + k*NX6 + i;
            double v_exact = Force/2.0/niu*z_h[j*NZ6+k]*(LZ-z_h[j*NZ6+k]);

            printf("%lf\t %lf\t %lf\n", z_h[j*NZ6+k], v_exact, v_h_p[index]);
        }
    } */

    printf("\n----------- Start Output, myid = %d ----------\n", myid);

    if( myid == 0 ) {
        FILE *fp_gg;
        fp_gg = fopen("./result/0_force.dat","w");
        fprintf( fp_gg, "%.15lf", Force_h[0] );
        fclose( fp_gg );
    }

    OutputData(rho_h_p, "rho", myid);
    OutputData(u_h_p,   "u",   myid);
    OutputData(v_h_p,   "v",   myid);
    OutputData(w_h_p,   "w",   myid);
}

void fileIO_PDF()
{
    OutputData(fh_p[0],  "f0",  myid);
    OutputData(fh_p[1],  "f1",  myid);
    OutputData(fh_p[2],  "f2",  myid);
    OutputData(fh_p[3],  "f3",  myid);
    OutputData(fh_p[4],  "f4",  myid);
    OutputData(fh_p[5],  "f5",  myid);
    OutputData(fh_p[6],  "f6",  myid);
    OutputData(fh_p[7],  "f7",  myid);
    OutputData(fh_p[8],  "f8",  myid);
    OutputData(fh_p[9],  "f9",  myid);
    OutputData(fh_p[10], "f10", myid);
    OutputData(fh_p[11], "f11", myid);
    OutputData(fh_p[12], "f12", myid);
    OutputData(fh_p[13], "f13", myid);
    OutputData(fh_p[14], "f14", myid);
    OutputData(fh_p[15], "f15", myid);
    OutputData(fh_p[16], "f16", myid);
    OutputData(fh_p[17], "f17", myid);
    OutputData(fh_p[18], "f18", myid);
}

void ReadData(
    double *arr_h,
    const char *folder,     const char *fname,      const int myid  )
{
    char path[100];
    sprintf( path, "./%s/%s_%d.bin", folder, fname, myid );

    FILE *data;
    if((data = fopen(path, "rb")) == NULL) {
        printf("Read data error, exit...\n");
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    fread( arr_h, sizeof(double), NX6*NZ6*NYD6, data );
    fclose( data );
}

void InitialUsingBkpData()
{
    PreCheckDir();

    char result[30];

    sprintf( result, "result" );

    FILE *fp_gg;
    fp_gg = fopen("./result/0_force.dat","r");
    fscanf( fp_gg, "%lf", &Force_h[0] );
    fclose( fp_gg );

    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );

    ReadData(rho_h_p, result, "rho", myid);
    ReadData(u_h_p,   result, "u",   myid);
    ReadData(v_h_p,   result, "v",   myid);
    ReadData(w_h_p,   result, "w",   myid);

    ReadData(fh_p[0],  result, "f0",  myid);
    ReadData(fh_p[1],  result, "f1",  myid);
    ReadData(fh_p[2],  result, "f2",  myid);
    ReadData(fh_p[3],  result, "f3",  myid);
    ReadData(fh_p[4],  result, "f4",  myid);
    ReadData(fh_p[5],  result, "f5",  myid);
    ReadData(fh_p[6],  result, "f6",  myid);
    ReadData(fh_p[7],  result, "f7",  myid);
    ReadData(fh_p[8],  result, "f8",  myid);
    ReadData(fh_p[9],  result, "f9",  myid);
    ReadData(fh_p[10], result, "f10", myid);
    ReadData(fh_p[11], result, "f11", myid);
    ReadData(fh_p[12], result, "f12", myid);
    ReadData(fh_p[13], result, "f13", myid);
    ReadData(fh_p[14], result, "f14", myid);
    ReadData(fh_p[15], result, "f15", myid);
    ReadData(fh_p[16], result, "f16", myid);
    ReadData(fh_p[17], result, "f17", myid);
    ReadData(fh_p[18], result, "f18", myid);
}

void OutputTBData(
    double *arr_d,
    const char *fname,      const int myid  )
{
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    const size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)malloc(nBytes);

    CHECK_CUDA( cudaMemcpy(arr_h, arr_d, nBytes, cudaMemcpyDeviceToHost) );

    char path[100];
    sprintf( path, "./statistics/%s/%s_%d.bin", fname, fname, myid );

    FILE *fp = NULL;
    fp = fopen( path, "wb" );

    for( int k = 3; k < NZ6-3;  k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3;  i++ ){
        int index = j*NX6*NZ6 + k*NX6 + i;
        fwrite( &arr_h[index], sizeof(double), 1, fp );
    }}}

    fclose( fp );
    free( arr_h );
}

void Launch_OutputTB()
{
    if( myid == 0 ) {
        FILE *fp_accu;
        fp_accu = fopen("./statistics/accu.dat","w");
        fprintf( fp_accu, "%d", accu_num );
        fclose( fp_accu );
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    OutputTBData(U, "U", myid);
	OutputTBData(V, "V", myid);
	OutputTBData(W, "W", myid);
	OutputTBData(P, "P", myid);
	OutputTBData(UU, "UU", myid);
	OutputTBData(UV, "UV", myid);
	OutputTBData(UW, "UW", myid);
	OutputTBData(VV, "VV", myid);
	OutputTBData(VW, "VW", myid);
	OutputTBData(WW, "WW", myid);
	OutputTBData(PU, "PU", myid);
	OutputTBData(PV, "PV", myid);
	OutputTBData(PW, "PW", myid);
	OutputTBData(KT, "KT", myid);
	OutputTBData(DUDX2, "DUDX2", myid);
	OutputTBData(DUDY2, "DUDY2", myid);
	OutputTBData(DUDZ2, "DUDZ2", myid);
	OutputTBData(DVDX2, "DVDX2", myid);
	OutputTBData(DVDY2, "DVDY2", myid);
	OutputTBData(DVDZ2, "DVDZ2", myid);
	OutputTBData(DWDX2, "DWDX2", myid);
	OutputTBData(DWDY2, "DWDY2", myid);
	OutputTBData(DWDZ2, "DWDZ2", myid);
	OutputTBData(UUU, "UUU", myid);
	OutputTBData(UUV, "UUV", myid);
	OutputTBData(UUW, "UUW", myid);
	OutputTBData(VVU, "VVU", myid);
	OutputTBData(VVV, "VVV", myid);
	OutputTBData(VVW, "VVW", myid);
	OutputTBData(WWU, "WWU", myid);
	OutputTBData(WWV, "WWV", myid);
	OutputTBData(WWW, "WWW", myid);
	//OutputTBData(OMEGA_X, "OMEGA_X", myid);
	//OutputTBData(OMEGA_Y, "OMEGA_Y", myid);
	//OutputTBData(OMEGA_Z, "OMEGA_Z", myid);
}

void ReadTBData(
    double * arr_d,
    const char *fname,      const int myid  )
{
    const size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)malloc(nBytes);

    char result[100];
    sprintf( result, "./statistics/%s/%s_%d.bin", fname, fname, myid );
    FILE *fp = NULL;
    fp = fopen(result, "rb");

    for( int k = 3; k < NZ6-3;  k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3;  i++ ){
        const int index = j*NX6*NZ6 + k*NX6 + i;
        fread( &arr_h[index], sizeof(double), 1, fp );
    }}}

    fclose( fp );
    CHECK_CUDA( cudaMemcpy(arr_d, arr_h, nBytes, cudaMemcpyHostToDevice) );
    free( arr_h );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

void InitialTBUsingBkpData() {

    FILE *fp_accu;
    fp_accu = fopen("./statistics/accu.dat","r");
    fscanf( fp_accu, "%d", &accu_num );
    fclose( fp_accu );

    ReadTBData(U, "U", myid);
	ReadTBData(V, "V", myid);
	ReadTBData(W, "W", myid);
	ReadTBData(P, "P", myid);
	ReadTBData(UU, "UU", myid);
	ReadTBData(UV, "UV", myid);
	ReadTBData(UW, "UW", myid);
	ReadTBData(VV, "VV", myid);
	ReadTBData(VW, "VW", myid);
	ReadTBData(WW, "WW", myid);
	ReadTBData(PU, "PU", myid);
	ReadTBData(PV, "PV", myid);
	ReadTBData(PW, "PW", myid);
	ReadTBData(KT, "KT", myid);
	ReadTBData(DUDX2, "DUDX2", myid);
	ReadTBData(DUDY2, "DUDY2", myid);
	ReadTBData(DUDZ2, "DUDZ2", myid);
	ReadTBData(DVDX2, "DVDX2", myid);
	ReadTBData(DVDY2, "DVDY2", myid);
	ReadTBData(DVDZ2, "DVDZ2", myid);
	ReadTBData(DWDX2, "DWDX2", myid);
	ReadTBData(DWDY2, "DWDY2", myid);
	ReadTBData(DWDZ2, "DWDZ2", myid);
	ReadTBData(UUU, "UUU", myid);
	ReadTBData(UUV, "UUV", myid);
	ReadTBData(UUW, "UUW", myid);
	ReadTBData(VVU, "VVU", myid);
	ReadTBData(VVV, "VVV", myid);
	ReadTBData(VVW, "VVW", myid);
	ReadTBData(WWU, "WWU", myid);
	ReadTBData(WWV, "WWV", myid);
	ReadTBData(WWW, "WWW", myid);
	//ReadTBData(OMEGA_X, "OMEGA_X", myid);
	//ReadTBData(OMEGA_Y, "OMEGA_Y", myid);
	//ReadTBData(OMEGA_Z, "OMEGA_Z", myid);

	CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

void Output3Dvelocity(){

    char filename_E2[300];

    sprintf(filename_E2, "test_%d.plt",myid);
    printf("%s\n", filename_E2);

    FILE *fpE3;

    fpE3 = fopen(filename_E2, "wb");

    int IMax = NX6-6;

    int JMax = NYD6-6;

    int KMax = NZ6-6;

    char Title[] = "Particle intensity";

    char Varname1[] = "X";

    char Varname2[] = "Y";

    char Varname3[] = "Z";

    //char Varname4[] = "Intensity";
    
    char Varname5[] = "U";
	
	char Varname6[] = "V"; 
	
	char Varname7[] = "W";

    char Zonename1[] = "Zone 001";

    float ZONEMARKER = 299.0;

    float EOHMARKER = 357.0;

    //==============Header Secontion =================//
    //------1.1 Magic number, Version number
    char MagicNumber[] = "#!TDV101";
    //cout << "watchout" << sizeof(MagicNumber) << endl;
    fwrite(MagicNumber, 8, 1, fpE3);

    //---- - 1.2.Integer value of 1.----------------------------------------------------------
    int IntegerValue = 1;
    fwrite(&IntegerValue, sizeof(IntegerValue), 1, fpE3);

    //---- - 1.3.Title and variable names.------------------------------------------------ -
    //---- - 1.3.1.The TITLE.
    wirte_ASCII_of_str(Title, fpE3);

    //---- - 1.3.2 Number of variables(NumVar) in the c_strfile.
    int NumVar = 6;
    fwrite(&NumVar, sizeof(NumVar), 1, fpE3);

    //------1.3.3 Variable names.N = L[1] + L[2] + ....L[NumVar]
    wirte_ASCII_of_str(Varname1, fpE3);
    wirte_ASCII_of_str(Varname2, fpE3);
    wirte_ASCII_of_str(Varname3, fpE3);
    //wirte_ASCII_of_str(Varname4, fpE3);
    wirte_ASCII_of_str(Varname5, fpE3);
    wirte_ASCII_of_str(Varname6, fpE3);
	wirte_ASCII_of_str(Varname7, fpE3);
    //---- - 1.4.Zones------------------------------------------------------------------ -
    //--------Zone marker.Value = 299.0
    fwrite(&ZONEMARKER, 1, sizeof(ZONEMARKER), fpE3);

    //--------Zone name.
    wirte_ASCII_of_str(Zonename1, fpE3);

    //--------Zone color
    int ZoneColor = -1;
    fwrite(&ZoneColor, sizeof(ZoneColor), 1, fpE3);

    //--------ZoneType
    int ZoneType = 0;
    fwrite(&ZoneType, sizeof(ZoneType), 1, fpE3);

    //--------DaraPacking 0=Block, 1=Point
    int DaraPacking = 1;
    fwrite(&DaraPacking, sizeof(DaraPacking), 1, fpE3);

    //--------Specify Var Location. 0 = Don't specify, all c_str is located at the nodes. 1 = Specify
    int SpecifyVarLocation = 0;
    fwrite(&SpecifyVarLocation, sizeof(SpecifyVarLocation), 1, fpE3);

    //--------Number of user defined face neighbor connections(value >= 0)
    int NumOfNeighbor = 0;
    fwrite(&NumOfNeighbor, sizeof(NumOfNeighbor), 1, fpE3);

    //-------- - IMax, JMax, KMax
    fwrite(&IMax, sizeof(IMax), 1, fpE3);
    fwrite(&JMax, sizeof(JMax), 1, fpE3);
    fwrite(&KMax, sizeof(KMax), 1, fpE3);

    //----------// -1 = Auxiliary name / value pair to follow 0 = No more Auxiliar name / value pairs.
    int AuxiliaryName = 0;
    fwrite(&AuxiliaryName, sizeof(AuxiliaryName), 1, fpE3);

    //----I HEADER OVER--------------------------------------------------------------------------------------------

    //=============================Geometries section=======================
    //=============================Text section======================
    // EOHMARKER, value = 357.0
    fwrite(&EOHMARKER, sizeof(EOHMARKER), 1, fpE3);

    //================II.Data section===============//
    //------ 2.1 zone---------------------------------------------------------------------- -
    fwrite(&ZONEMARKER, sizeof(ZONEMARKER), 1, fpE3);

    //--------variable c_str format, 1 = Float, 2 = Double, 3 = LongInt, 4 = ShortInt, 5 = Byte, 6 = Bit
    int fomat1 = 2;
    int fomat2 = 2;
    int fomat3 = 2;
    int fomat5 = 2;
    int fomat6 = 2;
	int fomat7 = 2;
    fwrite(&fomat1, sizeof(fomat1), 1, fpE3);
    fwrite(&fomat2, sizeof(fomat2), 1, fpE3);
    fwrite(&fomat3, sizeof(fomat3), 1, fpE3);
    fwrite(&fomat5, sizeof(fomat5), 1, fpE3);
    fwrite(&fomat6, sizeof(fomat6), 1, fpE3);
	fwrite(&fomat7, sizeof(fomat7), 1, fpE3);
    //--------Has variable sharing 0 = no, 1 = yes.
    int HasVarSharing = 0;
    fwrite(&HasVarSharing, sizeof(HasVarSharing), 1, fpE3);

    //----------Zone number to share connectivity list with(-1 = no sharing).
    int ZoneNumToShareConnectivity = -1;
    fwrite(&ZoneNumToShareConnectivity, sizeof(ZoneNumToShareConnectivity), 1, fpE3);

    //----------Zone c_str.Each variable is in c_str format asspecified above.
    for (int k = 3; k < NZ6-3; k++){
    for (int j = 3; j < NYD6-3; j++){
    for (int i = 3; i < NX6-3; i++){

    double VarToWrite1 = x_h[i];
    double VarToWrite2 = y_h[j];
    double VarToWrite3 = z_h[j*NZ6+k];

    int index = j*NZ6*NX6 + k*NX6 + i;
	double Uvelocity = u_h_p[index]/Uref;
	double Vvelocity = v_h_p[index]/Uref;
	double Wvelocity = w_h_p[index]/Uref; 
    fwrite(&VarToWrite1, sizeof(VarToWrite1), 1, fpE3);
    fwrite(&VarToWrite2, sizeof(VarToWrite2), 1, fpE3);
    fwrite(&VarToWrite3, sizeof(VarToWrite3), 1, fpE3);
	fwrite(&Uvelocity, sizeof(Uvelocity), 1, fpE3);
	fwrite(&Vvelocity, sizeof(Vvelocity), 1, fpE3);
	fwrite(&Wvelocity, sizeof(Wvelocity), 1, fpE3);
    }}}

    fclose(fpE3);

}

void wirte_ASCII_of_str(char * str, FILE * file)
{
    int value = 0;

    while ((*str) != '\0'){
        value = (int)*str;
        fwrite(&value, sizeof(int), 1, file);
        str++;
    }

    char null_char[] = "";

    value = (int)*null_char;

    fwrite(&value, sizeof(int), 1, file);
}
#endif