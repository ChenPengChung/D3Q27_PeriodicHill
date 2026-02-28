#include <time.h>
#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdarg.h>
#include "variables.h"
using namespace std;
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
double delta_eta_h[19];              // GILBM η-direction displacement (constant for uniform x)

// Phase 3: Curvilinear global time step (runtime, from CFL on contravariant velocities)
// NOTE: dt (= minSize) is a compile-time macro in variables.h for defining ν.
//       dt_global is the actual curvilinear time step, computed at runtime.
double dt_global;
double omega_global;     // = 3·niu/dt_global + 0.5 (dimensionless relaxation time)
double omegadt_global;   // = omega_global · dt_global (dimensional relaxation time τ)

// Phase 4: Local Time Step fields [NYD6*NZ6]
double *dt_local_h, *dt_local_d;
double *omega_local_h, *omega_local_d;       // ω_local = 3·niu/dt_local + 0.5
double *omegadt_local_h;                      // ω·Δt (host only, diagnostic)

// GILBM two-pass architecture: persistent global arrays
double *f_pc_d;           // Per-point post-collision stencil data [19 * 343 * NX6*NYD6*NZ6]
double *feq_d;            // Equilibrium distribution [19 * NX6*NYD6*NZ6]
double *omegadt_local_d;  // Precomputed ω·Δt at each grid point [NX6*NYD6*NZ6] (3D broadcast)
//
// 逆變速度在 GPU kernel 中即時計算（不需全場存儲）：
//   ẽ_α_η = e[α][0] / dx           (常數)
//   ẽ_α_ξ = e[α][1] / dy           (常數)
//   ẽ_α_ζ = e[α][1]*dk_dy + e[α][2]*dk_dz  (從度量項即時算)
//
// RK2 上風點座標是 kernel 局部變量，不需全場存儲


//Variables for forcing term modification.
double  *Ub_avg_h,  *Ub_avg_d;
double  Ub_avg_global = 0.0;   // Bcast 後的全場代表 u_bulk (rank 0 入口截面)

double  *Force_h,   *Force_d;

double *rho_modify_h, *rho_modify_d;

// Time-average accumulation (host-side, for VTK output)
// v = streamwise, w = wall-normal; accumulated every outer iteration
double *v_tavg_h = NULL, *w_tavg_h = NULL;
int time_avg_count = 0;

int nProcs, myid;

int step;
int restart_step = 0;  // 續跑起始步 (INIT=2 時從 VTK header 解析)
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
    double dx_val = LX / (double)(NX6 - 7);
    double dy_val = LY / (double)(NY6 - 7);
    //dt_global 取為遍歷每一格空間計算點，每一個分量，每一個編號下的速度分量最大值，定義而成
    //dt_global 指的就是global time step
    // 每個 rank 先計算自己的 dt_rank，再取全域 MIN
    double dt_rank = ComputeGlobalTimeStep(dk_dz_h, dk_dy_h, dx_val, dy_val, NYD6, NZ6, CFL, myid, nProcs);
    CHECK_MPI( MPI_Allreduce(&dt_rank, &dt_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD) );

    //可以計算omega_global. ;
    omega_global = (3*niu/dt_global) + 0.5 ; 
    omegadt_global = omega_global*dt_global;
   
    if (myid == 0) {
        printf("  ─────────────────────────────────────────────────────────\n");
        printf("  dt_global = MIN(all ranks) = %.6e\n", dt_global);
        printf("  dt_old = minSize = %.6e\n", (double)minSize);
        printf("  ratio dt_global / minSize = %.4f\n", dt_global / (double)minSize);
        printf("  Speedup cost: %.1fx more timesteps per physical time\n", (double)minSize / dt_global);
        printf("  omega_global = %.6f, 1/omega_global = %.6f\n", omega_global, 1.0/omega_global);
        printf("  =============================================================\n\n");
    }

    // GILBM: 預計算三方向位移 δη (常數), δξ (常數), δζ (RK2 空間變化)
    PrecomputeGILBM_DeltaAll(delta_xi_h, delta_eta_h, delta_zeta_h,
                              dk_dz_h, dk_dy_h, NYD6, NZ6, dt_global );

                              
    // Phase 4: Local Time Step — per-point dt, omega, omega*dt
    ComputeLocalTimeStep(dt_local_h, omega_local_h, omegadt_local_h,
                         dk_dz_h, dk_dy_h, dx_val, dy_val,
                         niu, dt_global, NYD6, NZ6, CFL, myid);

    // Bug 3 fix: Exchange dt_local_h and omega_local_h ghost zones between MPI ranks.
    // ComputeLocalTimeStep only fills j=3..NYD6-4 with actual local values;
    // ghost j=0,1,2 and j=NYD6-3..NYD6-1 retain dt_global defaults.
    // Without this fix, omegadt_local_d at ghost j is wrong → R_AB ratio in
    // GILBM Step 2+3 is incorrect at MPI boundaries → causes divergence.
    {
        int ghost_count = Buffer * NZ6;  // 3 j-slices × NZ6 doubles
        int j_send_left  = (Buffer + 1) * NZ6;            // j=4
        int j_recv_right = (NYD6 - Buffer) * NZ6;         // j=36
        int j_send_right = (NYD6 - 2*Buffer - 1) * NZ6;   // j=32
        int j_recv_left  = 0;                               // j=0

        // dt_local_h: send j=4..6 to left, receive from right into j=36..38
        MPI_Sendrecv(&dt_local_h[j_send_left],  ghost_count, MPI_DOUBLE, l_nbr, 500,
                     &dt_local_h[j_recv_right], ghost_count, MPI_DOUBLE, r_nbr, 500,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // dt_local_h: send j=32..34 to right, receive from left into j=0..2
        MPI_Sendrecv(&dt_local_h[j_send_right], ghost_count, MPI_DOUBLE, r_nbr, 501,
                     &dt_local_h[j_recv_left],  ghost_count, MPI_DOUBLE, l_nbr, 501,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // omega_local_h: same exchange pattern
        MPI_Sendrecv(&omega_local_h[j_send_left],  ghost_count, MPI_DOUBLE, l_nbr, 502,
                     &omega_local_h[j_recv_right], ghost_count, MPI_DOUBLE, r_nbr, 502,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&omega_local_h[j_send_right], ghost_count, MPI_DOUBLE, r_nbr, 503,
                     &omega_local_h[j_recv_left],  ghost_count, MPI_DOUBLE, l_nbr, 503,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Recompute omegadt_local_h at exchanged ghost zones for host-side consistency
        for (int j = 0; j < Buffer; j++)
            for (int k = 0; k < NZ6; k++) {
                int idx = j * NZ6 + k;
                omegadt_local_h[idx] = omega_local_h[idx] * dt_local_h[idx];
            }
        for (int j = NYD6 - Buffer; j < NYD6; j++)
            for (int k = 0; k < NZ6; k++) {
                int idx = j * NZ6 + k;
                omegadt_local_h[idx] = omega_local_h[idx] * dt_local_h[idx];
            }

        if (myid == 0) printf("GILBM: dt_local/omega_local ghost zones exchanged between MPI ranks.\n");
    }
    
    // Phase 4: Recompute delta_zeta with local dt (overwrites global-dt values)
    PrecomputeGILBM_DeltaZeta_Local(delta_zeta_h, dk_dz_h, dk_dy_h,
                                     dt_local_h, NYD6, NZ6);
    //在初始化階段就已計算zeta方向的偏移量以local time step 為單位
    // Phase 2: CFL validation — departure point safety check (should now PASS)
    bool cfl_ok = ValidateDepartureCFL(delta_zeta_h, dk_dy_h, dk_dz_h, NYD6, NZ6, myid);
    if (!cfl_ok && myid == 0) {
        fprintf(stderr,
            "[WARNING] CFL_zeta >= 1.0 still detected after Imamura time step.\n"
            "  This should not happen — check ComputeGlobalTimeStep logic.\n");
    }

    // 拷貝度量項、位移量陣列、runtime 參數到 GPU
    // NOTE: delta_eta/delta_xi/delta_zeta 對所有 19 方向統一上傳。
    // 壁面節點上 BC 方向（ẽ^ζ_α > 0 底壁 / < 0 頂壁）的值雖已上傳，
    // 但 kernel 中 NeedsBoundaryCondition() 判定後跳過 streaming，不讀取。
    CHECK_CUDA( cudaMemcpy(dk_dz_d,   dk_dz_h,   NYD6*NZ6*sizeof(double),      cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dk_dy_d,   dk_dy_h,   NYD6*NZ6*sizeof(double),      cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(delta_zeta_d, delta_zeta_h, 19*NYD6*NZ6*sizeof(double),   cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_delta_xi,  delta_xi_h,  19*sizeof(double)) );
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_delta_eta, delta_eta_h, 19*sizeof(double)) );
    // GILBM_dt = dt_global, matching delta_eta/xi precomputation.
    // Kernel's a_local = dt_A / GILBM_dt scales delta_eta/xi to local dt.
    { double gilbm_dt_val = dt_global;
      CHECK_CUDA( cudaMemcpyToSymbol(GILBM_dt, &gilbm_dt_val, sizeof(double)) ); }
    // Phase 4: LTS fields to GPU (omega_local is 2D; omegadt_local_d is 3D, filled by Init_OmegaDt_Kernel)
    CHECK_CUDA( cudaMemcpy(dt_local_d,      dt_local_h,      NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(omega_local_d,   omega_local_h,   NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice) );

    if (myid == 0) printf("GILBM: GILBM_dt/delta_eta/delta_xi (__constant__) + delta_zeta/dk/LTS (field) copied to GPU.\n");

    if ( INIT == 0 ) {
        printf("Initializing by default function...\n");
        InitialUsingDftFunc();
    } else if ( INIT == 1 ) {
        printf("Initializing by backup data...\n");
        result_readbin_velocityandf();
        if( TBINIT && TBSWITCH ) statistics_readbin_stress();
    } else if ( INIT == 2 ) {
        printf("Initializing from merged VTK: %s\n", RESTART_VTK_FILE);
        InitFromMergedVTK(RESTART_VTK_FILE);
    }
     
    // Phase 1.5 acceptance diagnostic: delta_xi, delta_zeta range, interpolation, C-E BC
    DiagnoseGILBM_Phase1(delta_xi_h, delta_zeta_h, dk_dz_h, dk_dy_h, fh_p, NYD6, NZ6, myid, dt_global);

    SendDataToGPU();

    // GILBM two-pass initialization: omega_dt, feq, f_pc
    {
        dim3 init_block(8, 8, 4);
        dim3 init_grid((NX6 + init_block.x - 1) / init_block.x,
                       (NYD6 + init_block.y - 1) / init_block.y,
                       (NZ6 + init_block.z - 1) / init_block.z);

        Init_OmegaDt_Kernel<<<init_grid, init_block>>>(
            dt_local_d, omega_local_d, omegadt_local_d
        );

        Init_Feq_Kernel<<<init_grid, init_block>>>(
            fd[0], fd[1], fd[2], fd[3], fd[4], fd[5], fd[6], fd[7], fd[8], fd[9],
            fd[10], fd[11], fd[12], fd[13], fd[14], fd[15], fd[16], fd[17], fd[18],
            feq_d
        );

        Init_FPC_Kernel<<<init_grid, init_block>>>(
            fd[0], fd[1], fd[2], fd[3], fd[4], fd[5], fd[6], fd[7], fd[8], fd[9],
            fd[10], fd[11], fd[12], fd[13], fd[14], fd[15], fd[16], fd[17], fd[18],
            f_pc_d
        );

        CHECK_CUDA( cudaDeviceSynchronize() );
        if (myid == 0) printf("GILBM two-pass: omega_dt, feq, f_pc initialized.\n");
    }
    
    // ---- GILBM Initialization Parameter Summary ----
    {
        // Find dt_local max/min over interior computational points
        double dt_local_max_loc = 0.0;
        double dt_local_min_loc = 1e30;
        for (int j = 3; j < NYD6 - 3; j++) {
            for (int k = 3; k < NZ6 - 3; k++) {
                int idx = j * NZ6 + k;
                if (dt_local_h[idx] > dt_local_max_loc) dt_local_max_loc = dt_local_h[idx];
                if (dt_local_h[idx] < dt_local_min_loc) dt_local_min_loc = dt_local_h[idx];
            }
        }
        double dt_local_max_g, dt_local_min_g;
        MPI_Allreduce(&dt_local_max_loc, &dt_local_max_g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&dt_local_min_loc, &dt_local_min_g, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        if (myid == 0) {
            double omega_dtmax = 3.0 * niu / dt_local_max_g + 0.5;
            double tau_dtmax   = 3.0 * niu + 0.5 * dt_local_max_g;
            double omega_dtmin = 3.0 * niu / dt_local_min_g + 0.5;
            double tau_dtmin   = 3.0 * niu + 0.5 * dt_local_min_g;

            printf("\n+================================================================+\n");
            printf("| GILBM Initialization Parameter Summary                         |\n");
            printf("+================================================================+\n");
            printf("| [Input]  Re               = %d\n", (int)Re);
            printf("| [Input]  Uref             = %.6f\n", (double)Uref);
            printf("| [Output] niu              = %.6e\n", (double)niu);
            printf("+----------------------------------------------------------------+\n");
            printf("| [Output] dt_global        = %.6e\n", dt_global);
            printf("|   -> Omega(dt_global)     = 3*niu/dt_global + 0.5     = %.6f\n", omega_global);
            printf("|   -> tau(dt_global)       = 3*niu + 0.5*dt_global     = %.6e  (\"omegadt_global\")\n", omegadt_global);
            printf("+----------------------------------------------------------------+\n");
            printf("| [Output] dt_local_max     = %.6e  (channel center)\n", dt_local_max_g);
            printf("|   -> Omega(dt_local_max)  = 3*niu/dt_local_max + 0.5  = %.6f\n", omega_dtmax);
            printf("|   -> tau(dt_local_max)    = 3*niu + 0.5*dt_local_max  = %.6e\n", tau_dtmax);
            printf("+----------------------------------------------------------------+\n");
            printf("| [Output] dt_local_min     = %.6e  (wall)\n", dt_local_min_g);
            printf("|   -> Omega(dt_local_min)  = 3*niu/dt_local_min + 0.5  = %.6f\n", omega_dtmin);
            printf("|   -> tau(dt_local_min)    = 3*niu + 0.5*dt_local_min  = %.6e\n", tau_dtmin);
            printf("+================================================================+\n\n");
        }
    } 
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    // Allocate time-average accumulation arrays (zero-initialized via calloc)
    {
        size_t nTotal = (size_t)NX6 * NYD6 * NZ6;
        v_tavg_h = (double*)calloc(nTotal, sizeof(double));
        w_tavg_h = (double*)calloc(nTotal, sizeof(double));
        time_avg_count = 0;
        if (myid == 0) printf("Time-average arrays allocated (%.1f MB each).\n",
                               nTotal * sizeof(double) / 1.0e6);
    }

    CHECK_CUDA( cudaEventRecord(start,0) );
	CHECK_CUDA( cudaEventRecord(start1,0) );
    // 續跑初始狀態輸出: 從 CPU 資料計算 Ub，完整顯示重啟狀態
    if (restart_step > 0) {
        // Compute Ub from CPU data (rank 0 only, j=3 hill-crest cross-section)
        // 同 Launch_ModifyForcingTerm 的公式: Σ v(j=3,k,i) / (LX*(LZ-1))
        double Ub_init = 0.0;
        if (myid == 0) {
            for (int k = 3; k < NZ6-3; k++) {
            for (int i = 3; i < NX6-4; i++) {
                int index = 3 * NX6 * NZ6 + k * NX6 + i;
                Ub_init += v_h_p[index];
            }}
            Ub_init /= (double)(LX * (LZ - 1.0));
        }
        MPI_Bcast(&Ub_init, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Ub_avg_global = Ub_init;

        if (myid == 0) {
            double FTT_init = (double)restart_step * dt_global / (double)flow_through_time;
            double Ustar = Ub_init / (double)Uref;
            double Fstar = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);
            double Re_now = Ub_init * (double)H_HILL / (double)niu;
            double Ma_init = Ub_init / (double)cs;

            printf("+----------------------------------------------------------------+\n");
            printf("| Step = %d    FTT = %.2f\n", restart_step, FTT_init);
            printf("|%s running with %4dx%4dx%4d grids\n", argv[0], (int)NX6, (int)NY6, (int)NZ6);
            printf("| Loop %d more steps, end at step %d\n", (int)loop, restart_step + 1 + (int)loop);
            printf("+----------------------------------------------------------------+\n");
            printf("[Step %d | FTT=%.2f] Ub=%.6f  U*=%.4f  Force=%.5E  F*=%.4f  Re(now)=%.1f  Ma=%.4f\n",
                   restart_step, FTT_init, Ub_init, Ustar, Force_h[0], Fstar, Re_now, Ma_init);
        }
        // 輸出初始 VTK (驗證重啟載入正確 + 使用修正後的 stride mapping)
        fileIO_velocity_vtk_merged(restart_step);
    }
    // VTK step 為奇數 (step%1000==1), for-loop 須從偶數開始
    // 以確保 step+=1 後 monitoring 在奇數步 (step%N==1) 正確觸發
    int loop_start = (restart_step > 0) ? restart_step + 1 : 0;
    for( step = loop_start ; step < loop_start + loop ; step++, accu_num++ ) {

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

            double FTT_now = step * dt_global / (double)flow_through_time;
            printf("+----------------------------------------------------------------+\n");
			printf("| Step = %d    FTT = %.2f \n", step, FTT_now);
            printf("|%s running with %4dx%4dx%4d grids            \n", argv[0], (int)NX6, (int)NY6, (int)NZ6 );
            printf("| Running %6f mins                                           \n", (cudatime1/60/1000) ),
            printf("+----------------------------------------------------------------+\n");

            cudaEventRecord(start1,0);
        }

        if ( (step%(int)NDTFRC == 1) ) {
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
            for( int k = 3 ; k < NZ6-3; k++){     // 包含壁面 k=3 和 k=NZ6-4
            for( int i = 3 ; i < NX6-4; i++){
                int index = j*NX6*NZ6 + k*NX6 + i;
                rho_LocalSum =  rho_LocalSum + rho_h_p[index];
            }}}
            MPI_Reduce((void *)&rho_LocalSum, (void *)&rho_GlobalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if ( myid == 0){
                rho_global = 1.0*(NX6-7)*(NY6-7)*(NZ6-6);   // 64 個 k 計算點 (k=3..NZ6-4)
                rho_modify_h[0] =( rho_global - rho_GlobalSum ) / ((NX6-7)*(NY6-7)*(NZ6-6));
                cudaMemcpy(rho_modify_d, rho_modify_h, sizeof(double), cudaMemcpyHostToDevice);
            }

        // Time-average accumulation (每 outer iteration = 每 2 physical steps)
        // v_h_p/w_h_p 已由上方 SendDataToCPU(ft) 更新至 CPU
        {
            const int nTotal = NX6 * NYD6 * NZ6;
            for (int idx = 0; idx < nTotal; idx++) {
                v_tavg_h[idx] += v_h_p[idx];
                w_tavg_h[idx] += w_h_p[idx];
            }
            time_avg_count++;
        }

        //Check Mass Conservation + NaN early stop
        if ( step % 100 == 1){
            SendDataToCPU( ft );
            double rho_LocalSum = 0;
            double rho_GlobalSum = 0;
            double rho_initial = 1.0 ;
            for( int j = 3 ; j < NYD6-4; j++){
            for( int k = 3 ; k < NZ6-3; k++){     // 包含壁面 k=3 和 k=NZ6-4
            for( int i = 3 ; i < NX6-4; i++){
                int index = j*NX6*NZ6 + k*NX6 + i;
                rho_LocalSum =  rho_LocalSum + rho_h_p[index] ;
            }}}
            double rho_LocalAvg;
            rho_LocalAvg = rho_LocalSum / ((NX6-7)*(NYD6-7)*(NZ6-6));  // 64 個 k 計算點
            MPI_Reduce((void *)&rho_LocalAvg, (void *)&rho_GlobalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            // NaN / divergence early stop
            int nan_flag = 0;
            if (myid == 0) {
                double rho_avg_check = rho_GlobalSum / (double)jp;
                if (isnan(rho_avg_check) || isinf(rho_avg_check) || fabs(rho_avg_check - 1.0) > 0.01) {
                    printf("[FATAL] Divergence detected at step %d: rho_avg = %.6e, stopping.\n", step, rho_avg_check);
                    nan_flag = 1;
                }
            }
            MPI_Bcast(&nan_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (nan_flag) {
                // 輸出最後一步 VTK 供分析
                SendDataToCPU( ft );
                fileIO_velocity_vtk_merged( step );
                break;
            }

            if( myid ==0 ){
                double FTT_rho = step * dt_global / (double)flow_through_time;
                FILE *checkrho;
                checkrho = fopen("checkrho.dat","a");
                fprintf(checkrho,"%d\t %.4f\t %lf\t %lf\n",step, FTT_rho, rho_initial, rho_GlobalSum/(double)jp );
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
    free(v_tavg_h);
    free(w_tavg_h);
    FreeSource();
    MPI_Finalize();

    return 0;
}
