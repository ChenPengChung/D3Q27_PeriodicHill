#include <time.h>
#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdarg.h>

#include "variables.h"

/************************** Host Variables **************************/
double  *fh_p[19];
double  *rho_h_p,  *u_h_p,  *v_h_p,  *w_h_p;


/************************** Device Variables **************************/
double  *ft[19], *fd[19];
double  *rho_d,  *u,  *v,  *w;

/* double  *KT,    *DISS,
        *DUDX2, *DUDY2, *DUDZ2,
        *DVDX2, *DVDY2, *DVDZ2,
        *DWDX2, *DWDY2, *DWDZ2; */
double  *U,  *V,  *W,  *P, 
        *UU, *UV, *UW, *VV, *VW, *WW, *PU, *PV, *PW, *PP,
        *KT,
        *DUDX2, *DUDY2, *DUDZ2, 
        *DVDX2, *DVDY2, *DVDZ2, 
        *DWDX2, *DWDY2, *DWDZ2,
        *UUU, *UUV, *UUW,
        *VVU, *VVV, *VVW,
        *WWU, *WWV, *WWW;

/************************** Other Variables **************************/
double  *x_h, *y_h, *z_h, *xi_h,
        *x_d, *y_d, *z_d, *xi_d;
double  *Xdep_h[3], *Ydep_h[3], *Zdep_h[3],
        *Xdep_d[3], *Ydep_d[3], *Zdep_d[3];
/* double  *XPara0_h[7], *YPara0_h[7], *ZPara0_h[7],
        *XPara2_h[7], *YPara2_h[7], *ZPara2_h[7],
        *XPara0_d[7], *YPara0_d[7], *ZPara0_d[7],
        *XPara2_d[7], *YPara2_d[7], *ZPara2_d[7]; */

//Variables for interpolation process
double  *XPara0_h[7],    *XPara0_d[7],    *XPara2_h[7],    *XPara2_d[7],
        *YPara0_h[7],    *YPara0_d[7],    *YPara2_h[7],    *YPara2_d[7],
        *XiParaF3_h[7],  *XiParaF3_d[7],  *XiParaF4_h[7],  *XiParaF4_d[7],
        *XiParaF5_h[7],  *XiParaF5_d[7],  *XiParaF6_h[7],  *XiParaF6_d[7],
        *XiParaF15_h[7], *XiParaF15_d[7], *XiParaF16_h[7], *XiParaF16_d[7],
        *XiParaF17_h[7], *XiParaF17_d[7], *XiParaF18_h[7], *XiParaF18_d[7];

//Variables for BFL boundary condition
//If BFL boundary condition is required. 1: yes, 0: no.
int     *BFLReqF3_h,    *BFLReqF4_h,    *BFLReqF15_h,   *BFLReqF16_h,
        *BFLReqF3_d,    *BFLReqF4_d,    *BFLReqF15_d,   *BFLReqF16_d;
//Parameters of interpolation process.
double  *XBFLParaF37_h[7],      *XBFLParaF38_h[7],      *YBFLParaF378_h[7],     *XiBFLParaF378_h[7],
        *XBFLParaF49_h[7],      *XBFLParaF410_h[7],     *YBFLParaF4910_h[7],    *XiBFLParaF4910_h[7],
        *YBFLParaF15_h[7],      *XiBFLParaF15_h[7],
        *YBFLParaF16_h[7],      *XiBFLParaF16_h[7];
double  *XBFLParaF37_d[7],      *XBFLParaF38_d[7],      *YBFLParaF378_d[7],     *XiBFLParaF378_d[7],
        *XBFLParaF49_d[7],      *XBFLParaF410_d[7],     *YBFLParaF4910_d[7],    *XiBFLParaF4910_d[7],
        *YBFLParaF15_d[7],      *XiBFLParaF15_d[7],
        *YBFLParaF16_d[7],      *XiBFLParaF16_d[7];
double  *ZSlopePara_h[5],
        *ZSlopePara_d[5];

//Variables for forcing term modification.
double  *Ub_avg_h,  *Ub_avg_d;

double  *Force_h,   *Force_d;

double *rho_modify_h, *rho_modify_d;
//Variables for BFL 
double *Q3_h, *Q3_d, *Q4_h, *Q4_d, *Q15_h, *Q15_d, *Q16_h, *Q16_d;


int nProcs, myid;

int step;
int accu_num = 0;

int l_nbr, r_nbr;

MPI_Status    istat[8];

MPI_Request   request[23][4];
MPI_Status    status[23][4];

MPI_Datatype  DataSideways;

cudaStream_t  stream0, stream1, stream2;
cudaStream_t  tbsum_stream[2];
cudaEvent_t   start,   stop;
cudaEvent_t   start1,  stop1;

int Buffer     = 3;
int icount_sw  = Buffer * NX6 * NZ6;
int iToLeft    = (Buffer+1) * NX6 * NZ6;
int iFromLeft  = 0;
int iToRight   = NX6 * NYD6 * NZ6 - (Buffer*2+1) * NX6 * NZ6;
int iFromRight = iToRight + (Buffer+1) * NX6 * NZ6;

MPI_Request reqToLeft[23], reqToRight[23],   reqFromLeft[23], reqFromRight[23];
MPI_Request reqToTop[23],  reqToBottom[23],  reqFromTop[23],  reqFromBottom[23];
int itag_f3[23] = {250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272};
int itag_f4[23] = {200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222};
int itag_f5[23] = {300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322};
int itag_f6[23] = {400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422};

#include "common.h"

#include "model.h"
#include "memory.h"
#include "initialization.h"
#include "communication.h"
#include "monitor.h"
#include "statistics.h"
#include "evolution.h"
#include "fileIO.h"
#include "MRT_Matrix.h"
#include "MRT_Process.h"
int main(int argc, char *argv[])
{
    CHECK_MPI( MPI_Init(&argc, &argv) );
    CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &nProcs) );
    CHECK_MPI( MPI_Comm_rank(MPI_COMM_WORLD, &myid) );
	
	l_nbr = myid - 1;       r_nbr = myid + 1;
    if (myid == 0)    l_nbr = jp-1;
	if (myid == jp-1) r_nbr = 0;

	int iDeviceCount = 0;
    CHECK_CUDA( cudaGetDeviceCount( &iDeviceCount ) );
    CHECK_CUDA( cudaSetDevice( myid % iDeviceCount ) );

    if (myid == 0)  printf("\n%s running with %d GPUs...\n\n", argv[0], (int)(jp));          CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    printf( "[ Info ] Rank Rank %2d/%2d, localrank: %d/%d\n", myid, nProcs-1, myid, iDeviceCount );

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    AllocateMemory();
    //pre-check whether the directories exit or not
    PreCheckDir();
    CreateDataType();
    //generate mesh and coordinates of each point
	GenerateMesh_X();
    GenerateMesh_Y();
    GenerateMesh_Z();

    GetIntrplParameter_X();
    GetIntrplParameter_Y();
    GetIntrplParameter_Xi();    

    BFLInitialization();

    if ( INIT == 0 ) {
        printf("Initializing by default function...\n");
        InitialUsingDftFunc();
    } else if ( INIT == 1 ) {
        printf("Initializing by backup data...\n");
        result_readbin_velocityandf();
        if( TBINIT && TBSWITCH ) statistics_readbin_stress();
    }

    SendDataToGPU();

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    CHECK_CUDA( cudaEventRecord(start,0) );
	CHECK_CUDA( cudaEventRecord(start1,0) );
    for( step = 0 ; step < loop ; step++, accu_num++ ) {

        Launch_CollisionStreaming( ft, fd );

        if( (int)TBSWITCH ) { Launch_TurbulentSum( fd ); }

        CHECK_CUDA( cudaDeviceSynchronize() );
        step += 1;
        accu_num += 1;

        //Launch_ModifyForcingTerm();
        Launch_CollisionStreaming( fd, ft );

        if( (int)TBSWITCH ) { Launch_TurbulentSum( ft ); }

        CHECK_CUDA( cudaDeviceSynchronize() );

        if ( myid == 0 && step%5000 == 1 ) {
            CHECK_CUDA( cudaEventRecord( stop1,0 ) );
            CHECK_CUDA( cudaEventSynchronize( stop1 ) );
			float cudatime1;
			CHECK_CUDA( cudaEventElapsedTime( &cudatime1,start1,stop1 ) );

            printf("+----------------------------------------------------------------+\n");
			printf("| Step = %d \n",step);
            printf("|%s running with %4dx%4dx%4d grids            \n", argv[0], (int)NX6, (int)NY6, (int)NZ6 );
            printf("| Running %6f mins                                           \n", (cudatime1/60/1000) ),
            printf("+----------------------------------------------------------------+\n");

            cudaEventRecord(start1,0);
        }

        if (  (step%(int)NDTFRC == 1) ) {
            Launch_ModifyForcingTerm();
        }

		if ( step%(int)NDTMIT == 1 ) {
			Launch_Monitor( accu_num );
		}

        // 每1000步輸出合併 VTK 追蹤流場 (所有GPU合併為單一檔案)
        if ( step % 1000 == 0 ) {
            SendDataToCPU( ft );
            fileIO_velocity_vtk_merged( step );
        }

        cudaDeviceSynchronize();
        cudaMemcpy(Force_h, Force_d, sizeof(double), cudaMemcpyDeviceToHost);   
        //Global Mass Conservation Modify
        
            SendDataToCPU( ft );
            double rho_LocalSum  = 0.0;
            double rho_GlobalSum = 0.0;
            double rho_global;
            for( int j = 3 ; j < NYD6-4; j++){
            for( int k = 3 ; k < NZ6-3; k++){
            for( int i = 3 ; i < NX6-4; i++){ 
                int index = j*NX6*NZ6 + k*NX6 + i;
                rho_LocalSum =  rho_LocalSum + rho_h_p[index];
            }}}
            MPI_Reduce((void *)&rho_LocalSum, (void *)&rho_GlobalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if ( myid == 0){
                rho_global = 1.0*(NX6-7)*(NY6-7)*(NZ6-6);
                rho_modify_h[0] =( rho_global - rho_GlobalSum ) / ((NX6-7)*(NY6-7)*(NZ6-6));
                cudaMemcpy(rho_modify_d, rho_modify_h, sizeof(double), cudaMemcpyHostToDevice);
            }
        
        
        //Check Mass Conservation
        if ( step % 100 == 1){
            SendDataToCPU( ft ); 
            double rho_LocalSum = 0;
            double rho_GlobalSum = 0;
            double rho_initial = 1.0 ;
            for( int j = 3 ; j < NYD6-4; j++){
            for( int k = 3 ; k < NZ6-3; k++){
            for( int i = 3 ; i < NX6-4; i++){ 
                int index = j*NX6*NZ6 + k*NX6 + i;
                rho_LocalSum =  rho_LocalSum + rho_h_p[index] ;
            }}}
            double rho_LocalAvg;
            rho_LocalAvg = rho_LocalSum / ((NX6-7)*(NYD6-7)*(NZ6-6));
            MPI_Reduce((void *)&rho_LocalAvg, (void *)&rho_GlobalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if( myid ==0 ){
                FILE *checkrho;
                checkrho = fopen("checkrho.dat","a");
                fprintf(checkrho,"%d\t %lf\t %lf\n",step, rho_initial, rho_GlobalSum/(double)jp );
                fclose (checkrho);
            }
        }
        //fprintf(checkrho, "Step =%d\tRho_inital=%lf\tRho_avg=%lf\n",step, rho_inital, rho_avg );
         
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    SendDataToCPU( ft );
    result_writebin_velocityandf();
    //Output3Dvelocity();
    if( TBSWITCH ) {
        statistics_writebin_stress();
    }
    
    /* for( step = 0 ; step < loop ; step++ ){
        double rrhhoo = 0;
        for( int j = 0 ; j < NYD6; j++){
            for( int k = 0; k < NZ6; k++){
                for( int i = 0 ; i < NX6 ; i++){ 
                 
                int index = j*NX6*NZ6 + k*NX6 + i;
                //printf(" i = %d \t j = %d \t k = %d \t rho = %lf \n", i, j, k, rho_h_p[index]);
                rrhhoo =  rrhhoo + rho_h_p[index] ;
                }
            }
        }   
        double rho_avg;
        rho_avg = rrhhoo / (NX6*NYD6*NZ6) ;
        printf(" step = %d \t rho = %lf \n",step, rho_avg );
    } */
    FreeSource();
    MPI_Finalize();

    return 0;
}
