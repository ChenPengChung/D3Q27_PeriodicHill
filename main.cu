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

// ZSlopePara retained for MeanDerivatives kernel in statistics.h (TBSWITCH=0, dead path)
double  *ZSlopePara_d[5];



//======== GILBM 度量項（Imamura 2005 左側元素）========
// 座標變換 (x,y,z) → 計算空間 (η=i, ξ=j, ζ=k)
// 度量項矩陣（∂計算/∂物理）：
//   | ∂η/∂x  ∂η/∂y  ∂η/∂z |   | 1/dx   0      0      |
//   | ∂ξ/∂x  ∂ξ/∂y  ∂ξ/∂z | = | 0      1/dy   0      |  ← 常數，不需陣列
//   | ∂ζ/∂x  ∂ζ/∂y  ∂ζ/∂z |   | 0      dk_dy  dk_dz  |  ← 隨空間變化
//
// 只需 2 個空間變化的度量項（大小 [NYD6*NZ6]，與 z_h 相同）
double *dk_dz_h, *dk_dz_d;   // ∂ζ/∂z = 1/(∂z/∂k)
double *dk_dy_h, *dk_dy_d;   // ∂ζ/∂y = -(∂z/∂j)/(dy·∂z/∂k)
double *delta_zeta_h, *delta_zeta_d;  // GILBM RK2 ζ-direction displacement [19*NYD6*NZ6]
double delta_xi_h[19];               // GILBM ξ-direction displacement (constant for uniform y)

// Phase 3: Imamura global time step — runtime parameters (extern'd in variables.h)
double dt;
double tau;

// Phase 4: Local Time Step fields [NYD6*NZ6]
double *dt_local_h, *dt_local_d;
double *tau_local_h, *tau_local_d;
double *tau_dt_product_h, *tau_dt_product_d;
//
// 逆變速度在 GPU kernel 中即時計算（不需全場存儲）：
//   ẽ_α_η = e[α][0] / dx           (常數)
//   ẽ_α_ξ = e[α][1] / dy           (常數)
//   ẽ_α_ζ = e[α][1]*dk_dy + e[α][2]*dk_dz  (從度量項即時算)
//
// RK2 上風點座標是 kernel 局部變量，不需全場存儲


//Variables for forcing term modification.
double  *Ub_avg_h,  *Ub_avg_d;

double  *Force_h,   *Force_d;

double *rho_modify_h, *rho_modify_d;



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
#include "gilbm/metric_terms.h"
#include "gilbm/precompute.h"
#include "gilbm/diagnostic_gilbm.h"
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

    // Phase 0: 計算離散 Jacobian 度量項並輸出診斷文件
    DiagnoseMetricTerms(myid);

    // GILBM Phase 1: 計算各 rank 的區域度量項
    ComputeMetricTerms(dk_dz_h, dk_dy_h, z_h, y_h, NYD6, NZ6);

    // Phase 3: Imamura's global time step (Eq. 22)
    // Compute ν_target from original parameters: tau_orig=0.6833, dt_orig=minSize
    double niu_target = (0.6833 - 0.5) / 3.0 * (double)minSize;
    double dx_val = LX / (double)(NX6 - 7);
    double dy_val = LY / (double)(NY6 - 7);

    dt = ComputeGlobalTimeStep(dk_dz_h, dk_dy_h, dx_val, dy_val, NYD6, NZ6, CFL, myid);
    tau = 0.5 + 3.0 * niu_target / dt;

    if (myid == 0) {
        printf("Phase 3: Imamura CFL time step\n");
        printf("  dt  = %.6e (was minSize = %.6e, ratio = %.4f)\n",
               dt, (double)minSize, dt / (double)minSize);
        printf("  tau = %.6f (was 0.6833)\n", tau);
        printf("  niu = %.6e (target = %.6e, match = %s)\n",
               niu, niu_target, fabs(niu - niu_target) < 1e-15 ? "YES" : "NO");
        printf("  Re  = %.1f, Uref = %.6e\n", (double)Re, Uref);
    }

    // Precompute delta_eta (η-direction, constant, like delta_xi)
    double delta_eta_h[19];
    PrecomputeGILBM_DeltaEta(delta_eta_h, dx_val, dt);

    // GILBM Phase 1.5: 預計算 δξ (常數) 和 δζ (RK2 空間變化) 位移 (using runtime dt)
    PrecomputeGILBM_DeltaXiZeta(delta_xi_h, delta_zeta_h, dk_dz_h, dk_dy_h, NYD6, NZ6);

    // Phase 4: Local Time Step — per-point dt, tau, tau*dt
    ComputeLocalTimeStep(dt_local_h, tau_local_h, tau_dt_product_h,
                         dk_dz_h, dk_dy_h, dx_val, dy_val,
                         niu_target, dt, NYD6, NZ6, CFL, myid);

    // Phase 4: Recompute delta_zeta with local dt (overwrites global-dt values)
    PrecomputeGILBM_DeltaZeta_Local(delta_zeta_h, dk_dz_h, dk_dy_h,
                                     dt_local_h, NYD6, NZ6);

    // Phase 2: CFL validation — departure point safety check (should now PASS)
    bool cfl_ok = ValidateDepartureCFL(delta_zeta_h, dk_dy_h, dk_dz_h, NYD6, NZ6, myid);
    if (!cfl_ok && myid == 0) {
        fprintf(stderr,
            "[WARNING] CFL_zeta >= 1.0 still detected after Imamura time step.\n"
            "  This should not happen — check ComputeGlobalTimeStep logic.\n");
    }

    // 拷貝度量項、delta 陣列、runtime 參數到 GPU
    CHECK_CUDA( cudaMemcpy(dk_dz_d,   dk_dz_h,   NYD6*NZ6*sizeof(double),      cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dk_dy_d,   dk_dy_h,   NYD6*NZ6*sizeof(double),      cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(delta_zeta_d, delta_zeta_h, 19*NYD6*NZ6*sizeof(double),   cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_delta_xi,  delta_xi_h,  19*sizeof(double)) );
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_delta_eta, delta_eta_h, 19*sizeof(double)) );
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_dt,  &dt,  sizeof(double)) );
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_tau, &tau, sizeof(double)) );
    // Phase 4: LTS fields to GPU
    CHECK_CUDA( cudaMemcpy(dt_local_d,        dt_local_h,        NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(tau_local_d,       tau_local_h,       NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(tau_dt_product_d,  tau_dt_product_h,  NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice) );

    if (myid == 0) printf("GILBM: dt/tau/delta_eta/delta_xi (__constant__) + delta_zeta/LTS (field) copied to GPU.\n");

    if ( INIT == 0 ) {
        printf("Initializing by default function...\n");
        InitialUsingDftFunc();
    } else if ( INIT == 1 ) {
        printf("Initializing by backup data...\n");
        result_readbin_velocityandf();
        if( TBINIT && TBSWITCH ) statistics_readbin_stress();
    }

    // Phase 1.5 acceptance diagnostic: delta_xi, delta_zeta range, interpolation, C-E BC
    DiagnoseGILBM_Phase1(delta_xi_h, delta_zeta_h, dk_dz_h, dk_dy_h, fh_p, NYD6, NZ6, myid);

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
        // 注意: step 在迴圈中是奇數 (1,3,5...)，所以用 % 1000 == 1
        if ( step % 1000 == 1 ) {
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
            for( int k = 2 ; k < NZ6-2; k++){     // 包含壁面 k=2 和 k=NZ6-3
            for( int i = 3 ; i < NX6-4; i++){
                int index = j*NX6*NZ6 + k*NX6 + i;
                rho_LocalSum =  rho_LocalSum + rho_h_p[index];
            }}}
            MPI_Reduce((void *)&rho_LocalSum, (void *)&rho_GlobalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if ( myid == 0){
                rho_global = 1.0*(NX6-7)*(NY6-7)*(NZ6-4);   // 66 個 k 計算點 (k=2..NZ6-3)
                rho_modify_h[0] =( rho_global - rho_GlobalSum ) / ((NX6-7)*(NY6-7)*(NZ6-4));
                cudaMemcpy(rho_modify_d, rho_modify_h, sizeof(double), cudaMemcpyHostToDevice);
            }
        
        
        //Check Mass Conservation
        if ( step % 100 == 1){
            SendDataToCPU( ft ); 
            double rho_LocalSum = 0;
            double rho_GlobalSum = 0;
            double rho_initial = 1.0 ;
            for( int j = 3 ; j < NYD6-4; j++){
            for( int k = 2 ; k < NZ6-2; k++){     // 包含壁面 k=2 和 k=NZ6-3
            for( int i = 3 ; i < NX6-4; i++){
                int index = j*NX6*NZ6 + k*NX6 + i;
                rho_LocalSum =  rho_LocalSum + rho_h_p[index] ;
            }}}
            double rho_LocalAvg;
            rho_LocalAvg = rho_LocalSum / ((NX6-7)*(NYD6-7)*(NZ6-4));  // 66 個 k 計算點
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
