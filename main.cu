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
double  *fh_p[19]; //主機端一般態分佈函數
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

// ZSlopePara removed — MeanDerivatives now uses dk_dz_d/dk_dy_d metric terms



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
double *delta_zeta_h;  // GILBM RK2 ζ-direction displacement [19*NYD6*NZ6] (host-only)
double delta_xi_h[19];               // GILBM ξ-direction displacement (global dt, for initial CFL check)
double delta_eta_h[19];              // GILBM η-direction displacement (global dt, for initial CFL check)

// Part A: space-varying δη/δξ with dt_local [19*NYD6*NZ6] (host-only, feeds Lagrange precompute)
double *delta_eta_local_h;
double *delta_xi_local_h;

// Part B: precomputed Lagrange interpolation weights [19*7*NYD6*NZ6]
// Layout: w[q*7*NYD6*NZ6 + c*NYD6*NZ6 + j*NZ6 + k], q outermost, c middle
double *lagrange_eta_h,  *lagrange_eta_d;
double *lagrange_xi_h,   *lagrange_xi_d;
double *lagrange_zeta_h, *lagrange_zeta_d;

// Precomputed stencil base k [NZ6] (int, wall-clamped)
int *bk_precomp_h, *bk_precomp_d;

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

// Time-average accumulation (FTT-gated)
// u=spanwise, v=streamwise, w=wall-normal; GPU-side accumulation
double *u_tavg_h = NULL, *v_tavg_h = NULL, *w_tavg_h = NULL;   // host (for VTK output)
double *u_tavg_d = NULL, *v_tavg_d = NULL, *w_tavg_d = NULL;   // device (accumulated on GPU)
int vel_avg_count = 0;      // Stage 1: mean velocity accumulation count (FTT >= FTT_STAGE1)
int rey_avg_count = 0;      // Stage 2: Reynolds stress accumulation count (FTT >= FTT_STAGE2)
bool stage1_announced = false;
bool stage2_announced = false;

int nProcs, myid;

int step;
int restart_step = 0;  // 續跑起始步 (INIT=2 時從 VTK header 解析)
int accu_num = 0;
// ub_accu_count removed — Launch_ModifyForcingTerm now uses instantaneous Ub

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

    // Allocate time-average accumulation arrays early (before possible VTK restart read)
    {
        size_t nTotal = (size_t)NX6 * NYD6 * NZ6;
        u_tavg_h = (double*)calloc(nTotal, sizeof(double));
        v_tavg_h = (double*)calloc(nTotal, sizeof(double));
        w_tavg_h = (double*)calloc(nTotal, sizeof(double));
        vel_avg_count = 0;
        rey_avg_count = 0;
    }

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

    // Part A: Precompute space-varying δη/δξ with dt_local (replaces __constant__[19] + a_local)
    PrecomputeGILBM_DeltaEta_Local(delta_eta_local_h, dt_local_h, dx_val, NYD6, NZ6);
    PrecomputeGILBM_DeltaXi_Local(delta_xi_local_h, dt_local_h, dy_val, NYD6, NZ6);

    // Phase 2: CFL validation — departure point safety check (should now PASS)
    bool cfl_ok = ValidateDepartureCFL(delta_zeta_h, dk_dy_h, dk_dz_h, NYD6, NZ6, myid);
    if (!cfl_ok && myid == 0) {
        fprintf(stderr,
            "[WARNING] CFL_zeta >= 1.0 still detected after Imamura time step.\n"
            "  This should not happen — check ComputeGlobalTimeStep logic.\n");
    }

    // Part B: Precompute Lagrange interpolation weights for all 3 directions
    // Eliminates 3 × lagrange_7point_coeffs (each 42 divisions) per q per grid point from kernel
    PrecomputeGILBM_LagrangeWeights(lagrange_eta_h, lagrange_xi_h, lagrange_zeta_h,
                                     delta_eta_local_h, delta_xi_local_h, delta_zeta_h,
                                     NYD6, NZ6);

    // Precompute stencil base k (wall-clamped, depends only on k)
    PrecomputeGILBM_StencilBaseK(bk_precomp_h, NZ6);

    // Part C: Comprehensive diagnostic cross-check (rank 0 only)
    // Layout: w[q*7*sz + c*sz + idx_jk], q outermost, c middle
    // Four tests verify precomputed Lagrange weights + bk against multiple criteria:
    //   Test 1: Self-consistency — recompute from same delta arrays, expect bit-identical
    //   Test 2: Old-path equivalence — compare with a_local × GILBM_delta[q] path (FP rounding ~1e-16)
    //   Test 3: Partition of unity — Σ weights = 1.0 (Lagrange property)
    //   Test 4: bk_precomp consistency — compare with compute_stencil_base logic
    if (myid == 0) {
        int sz = NYD6 * NZ6;
        int test_count = 0, pass_count = 0;

        printf("\n=============================================================\n");
        printf("  Lagrange Weight Precomputation: Comprehensive Cross-Check\n");
        printf("  Layout: w[q*7*sz + c*sz + idx_jk] (q outermost, c middle)\n");
        printf("  Memory: 3 arrays x 19x7x%dx%d x 8 bytes = %.1f MB\n",
               NYD6, NZ6, 3.0 * 7 * 19 * NYD6 * NZ6 * 8.0 / 1e6);
        printf("  bk_precomp: %d x 4 bytes = %d bytes\n", NZ6, NZ6 * 4);
        printf("=============================================================\n");

        // ── Sample printout (quick visual inspection) ──
        {
            int q_test = 1, j_mid = NYD6 / 2, k_mid = NZ6 / 2;
            int idx = j_mid * NZ6 + k_mid;
            printf("  Sample (q=%d, j=%d, k=%d):\n", q_test, j_mid, k_mid);
            printf("    eta: ");
            for (int c = 0; c < 7; c++) printf("%.6f ", lagrange_eta_h[q_test*7*sz + c*sz + idx]);
            printf("\n    xi:  ");
            for (int c = 0; c < 7; c++) printf("%.6f ", lagrange_xi_h[q_test*7*sz + c*sz + idx]);
            printf("\n    zeta:");
            for (int c = 0; c < 7; c++) printf("%.6f ", lagrange_zeta_h[q_test*7*sz + c*sz + idx]);
            printf("\n    bk[%d]=%d, bk[3]=%d, bk[%d]=%d\n",
                   k_mid, bk_precomp_h[k_mid], bk_precomp_h[3], NZ6-4, bk_precomp_h[NZ6-4]);
        }

        // ── Test 1: Self-consistency ──
        // Recompute weights from the SAME delta arrays using the SAME host function.
        // Expected: bit-identical (max|err| = 0.0)
        {
            double max_err_eta = 0.0, max_err_xi = 0.0, max_err_zeta = 0.0;
            for (int q = 1; q < 19; q++) {
                int q_base = q * 7 * sz;
                for (int j = 3; j < NYD6 - 3; j++) {
                    for (int k = 3; k < NZ6 - 3; k++) {
                        int idx_jk = j * NZ6 + k;
                        double a_chk[7];

                        // η
                        double t_eta = 3.0 - delta_eta_local_h[q * sz + idx_jk];
                        if (t_eta < 0.0) t_eta = 0.0;
                        if (t_eta > 6.0) t_eta = 6.0;
                        lagrange_7point_coeffs_host(t_eta, a_chk);
                        for (int c = 0; c < 7; c++) {
                            double err = fabs(lagrange_eta_h[q_base + c*sz + idx_jk] - a_chk[c]);
                            if (err > max_err_eta) max_err_eta = err;
                        }

                        // ξ
                        double t_xi = 3.0 - delta_xi_local_h[q * sz + idx_jk];
                        if (t_xi < 0.0) t_xi = 0.0;
                        if (t_xi > 6.0) t_xi = 6.0;
                        lagrange_7point_coeffs_host(t_xi, a_chk);
                        for (int c = 0; c < 7; c++) {
                            double err = fabs(lagrange_xi_h[q_base + c*sz + idx_jk] - a_chk[c]);
                            if (err > max_err_xi) max_err_xi = err;
                        }

                        // ζ (with stencil base clamping)
                        double d_zeta = delta_zeta_h[q * sz + idx_jk];
                        double up_k_val = (double)k - d_zeta;
                        if (up_k_val < 3.0)              up_k_val = 3.0;
                        if (up_k_val > (double)(NZ6 - 4)) up_k_val = (double)(NZ6 - 4);
                        int bk_ref = k - 3;
                        if (bk_ref < 3)              bk_ref = 3;
                        if (bk_ref + 6 > NZ6 - 4)   bk_ref = NZ6 - 10;
                        double t_zeta = up_k_val - (double)bk_ref;
                        lagrange_7point_coeffs_host(t_zeta, a_chk);
                        for (int c = 0; c < 7; c++) {
                            double err = fabs(lagrange_zeta_h[q_base + c*sz + idx_jk] - a_chk[c]);
                            if (err > max_err_zeta) max_err_zeta = err;
                        }
                    }
                }
            }
            bool pass = (max_err_eta == 0.0 && max_err_xi == 0.0 && max_err_zeta == 0.0);
            printf("  Test 1: Self-consistency (recompute from same delta arrays)\n");
            printf("    max|err| eta=%.2e  xi=%.2e  zeta=%.2e  %s\n",
                   max_err_eta, max_err_xi, max_err_zeta, pass ? "[PASS]" : "[FAIL]");
            test_count++; if (pass) pass_count++;
        }

        // ── Test 2: Old-path equivalence ──
        // Compare precomputed weights against the OLD kernel computation path:
        //   OLD: delta_eta_old = (dt_local/dt_global) × (dt_global × e_x[q]/dx) = a_local × delta_eta_h[q]
        //   NEW: delta_eta_new = dt_local × (e_x[q]/dx) = delta_eta_local_h[q*sz+idx]
        // These are mathematically identical but differ by ~1 ULP due to FP associativity.
        // For ζ: both use same delta_zeta_h → should be bit-identical (not tested here).
        // Tolerance: 1e-14 (100× machine epsilon, generous margin)
        {
            double max_err_eta = 0.0, max_err_xi = 0.0;
            double max_t_diff_eta = 0.0, max_t_diff_xi = 0.0;
            for (int q = 1; q < 19; q++) {
                int q_base = q * 7 * sz;
                for (int j = 3; j < NYD6 - 3; j++) {
                    for (int k = 3; k < NZ6 - 3; k++) {
                        int idx_jk = j * NZ6 + k;
                        double dt_loc = dt_local_h[idx_jk];
                        double a_local = dt_loc / dt_global;
                        double a_old[7];

                        // η: old path
                        double delta_eta_old = a_local * delta_eta_h[q];
                        double t_eta_old = 3.0 - delta_eta_old;
                        double t_eta_new = 3.0 - delta_eta_local_h[q * sz + idx_jk];
                        double t_diff = fabs(t_eta_old - t_eta_new);
                        if (t_diff > max_t_diff_eta) max_t_diff_eta = t_diff;
                        lagrange_7point_coeffs_host(t_eta_old, a_old);
                        for (int c = 0; c < 7; c++) {
                            double err = fabs(lagrange_eta_h[q_base + c*sz + idx_jk] - a_old[c]);
                            if (err > max_err_eta) max_err_eta = err;
                        }

                        // ξ: old path
                        double delta_xi_old = a_local * delta_xi_h[q];
                        double t_xi_old = 3.0 - delta_xi_old;
                        double t_xi_new = 3.0 - delta_xi_local_h[q * sz + idx_jk];
                        t_diff = fabs(t_xi_old - t_xi_new);
                        if (t_diff > max_t_diff_xi) max_t_diff_xi = t_diff;
                        lagrange_7point_coeffs_host(t_xi_old, a_old);
                        for (int c = 0; c < 7; c++) {
                            double err = fabs(lagrange_xi_h[q_base + c*sz + idx_jk] - a_old[c]);
                            if (err > max_err_xi) max_err_xi = err;
                        }
                    }
                }
            }
            bool pass = (max_err_eta < 1e-14 && max_err_xi < 1e-14);
            printf("  Test 2: Old-path equivalence (a_local x delta_h[q] vs precomputed)\n");
            printf("    max|t_diff| eta=%.2e  xi=%.2e  (FP associativity)\n",
                   max_t_diff_eta, max_t_diff_xi);
            printf("    max|w_err|  eta=%.2e  xi=%.2e  %s\n",
                   max_err_eta, max_err_xi, pass ? "[PASS]" : "[FAIL]");
            printf("    (zeta uses same delta_zeta_h in both paths -> bit-identical, not tested)\n");
            test_count++; if (pass) pass_count++;
        }

        // ── Test 3: Partition of unity ──
        // Lagrange interpolation property: Σ_{c=0}^{6} L_c(t) = 1 for any t.
        // Numerical sum should be 1.0 ± machine epsilon accumulation.
        // Tolerance: 1e-12 (generous for 7-term sum)
        {
            double max_sum_err_eta = 0.0, max_sum_err_xi = 0.0, max_sum_err_zeta = 0.0;
            for (int q = 1; q < 19; q++) {
                int q_base = q * 7 * sz;
                for (int j = 3; j < NYD6 - 3; j++) {
                    for (int k = 3; k < NZ6 - 3; k++) {
                        int idx_jk = j * NZ6 + k;
                        double s_eta = 0.0, s_xi = 0.0, s_zeta = 0.0;
                        for (int c = 0; c < 7; c++) {
                            s_eta  += lagrange_eta_h [q_base + c*sz + idx_jk];
                            s_xi   += lagrange_xi_h  [q_base + c*sz + idx_jk];
                            s_zeta += lagrange_zeta_h[q_base + c*sz + idx_jk];
                        }
                        double e1 = fabs(s_eta  - 1.0); if (e1 > max_sum_err_eta)  max_sum_err_eta  = e1;
                        double e2 = fabs(s_xi   - 1.0); if (e2 > max_sum_err_xi)   max_sum_err_xi   = e2;
                        double e3 = fabs(s_zeta - 1.0); if (e3 > max_sum_err_zeta) max_sum_err_zeta = e3;
                    }
                }
            }
            bool pass = (max_sum_err_eta < 1e-12 && max_sum_err_xi < 1e-12 && max_sum_err_zeta < 1e-12);
            printf("  Test 3: Partition of unity (sum of 7 weights = 1.0)\n");
            printf("    max|sum-1| eta=%.2e  xi=%.2e  zeta=%.2e  %s\n",
                   max_sum_err_eta, max_sum_err_xi, max_sum_err_zeta, pass ? "[PASS]" : "[FAIL]");
            test_count++; if (pass) pass_count++;
        }

        // ── Test 4: bk_precomp consistency ──
        // Verify bk_precomp_h[k] matches compute_stencil_base z-clamping logic
        {
            bool pass = true;
            int fail_k = -1, fail_expected = -1, fail_got = -1;
            for (int k = 3; k < NZ6 - 3; k++) {
                int bk_ref = k - 3;
                if (bk_ref < 3)              bk_ref = 3;
                if (bk_ref + 6 > NZ6 - 4)   bk_ref = NZ6 - 10;
                if (bk_precomp_h[k] != bk_ref) {
                    pass = false;
                    fail_k = k; fail_expected = bk_ref; fail_got = bk_precomp_h[k];
                    break;
                }
            }
            printf("  Test 4: bk_precomp consistency (wall clamping)\n");
            if (pass) {
                printf("    All k=[3,%d] correct  [PASS]\n", NZ6-4);
            } else {
                printf("    MISMATCH at k=%d: expected=%d, got=%d  [FAIL]\n",
                       fail_k, fail_expected, fail_got);
            }
            test_count++; if (pass) pass_count++;
        }

        // ── Test 5: Displacement statistics + NaN/Inf + clamp count ──
        {
            const char *dir_names[3] = {"eta", "xi", "zeta"};
            const double *delta_arrays[3] = {delta_eta_local_h, delta_xi_local_h, delta_zeta_h};
            bool pass = true;
            int total_nan_inf = 0;

            printf("  Test 5: Displacement statistics & data integrity\n");
            for (int d = 0; d < 3; d++) {
                double d_min = 1e30, d_max = -1e30;
                int d_min_q = -1, d_min_j = -1, d_min_k = -1;
                int d_max_q = -1, d_max_j = -1, d_max_k = -1;
                int nan_inf_count = 0;
                int clamp_count = 0;  // departure points that needed clamping
                double t_min = 1e30, t_max = -1e30;

                for (int q = 1; q < 19; q++) {
                    for (int j = 3; j < NYD6 - 3; j++) {
                        for (int k = 3; k < NZ6 - 3; k++) {
                            int idx_jk = j * NZ6 + k;
                            double dval = delta_arrays[d][q * sz + idx_jk];

                            // NaN/Inf check
                            if (dval != dval || dval == 1.0/0.0 || dval == -1.0/0.0) {
                                nan_inf_count++;
                                continue;
                            }

                            // Min/max tracking
                            if (dval < d_min) { d_min = dval; d_min_q = q; d_min_j = j; d_min_k = k; }
                            if (dval > d_max) { d_max = dval; d_max_q = q; d_max_j = j; d_max_k = k; }

                            // Lagrange parameter and clamp check
                            double t_val;
                            if (d < 2) {
                                // η/ξ: t = 3 - delta
                                t_val = 3.0 - dval;
                                if (t_val < 0.0 || t_val > 6.0) clamp_count++;
                            } else {
                                // ζ: t = up_k - bk, with departure + stencil clamping
                                double up_k_val = (double)k - dval;
                                bool clamped = false;
                                if (up_k_val < 3.0 || up_k_val > (double)(NZ6 - 4)) clamped = true;
                                if (up_k_val < 3.0) up_k_val = 3.0;
                                if (up_k_val > (double)(NZ6 - 4)) up_k_val = (double)(NZ6 - 4);
                                int bk_ref = k - 3;
                                if (bk_ref < 3) bk_ref = 3;
                                if (bk_ref + 6 > NZ6 - 4) bk_ref = NZ6 - 10;
                                t_val = up_k_val - (double)bk_ref;
                                if (clamped) clamp_count++;
                            }
                            if (t_val < t_min) t_min = t_val;
                            if (t_val > t_max) t_max = t_val;
                        }
                    }
                }
                total_nan_inf += nan_inf_count;
                bool dir_ok = (nan_inf_count == 0 && clamp_count == 0);
                if (!dir_ok) pass = false;

                printf("    %4s: delta=[%+.4e, %+.4e]  t=[%.3f, %.3f]  "
                       "NaN/Inf=%d  clamp=%d  %s\n",
                       dir_names[d], d_min, d_max, t_min, t_max,
                       nan_inf_count, clamp_count, dir_ok ? "OK" : "WARN");
                if (d_min_q >= 0)
                    printf("          min@(q=%d,j=%d,k=%d)  max@(q=%d,j=%d,k=%d)\n",
                           d_min_q, d_min_j, d_min_k, d_max_q, d_max_j, d_max_k);
            }
            printf("    Overall: NaN/Inf=%d, clamp=%d  %s\n",
                   total_nan_inf, total_nan_inf, pass ? "[PASS]" : "[FAIL]");
            test_count++; if (pass) pass_count++;
        }

        // ── Overall ──
        printf("  -------------------------------------------------------------\n");
        printf("  Overall: %d/%d tests passed  %s\n",
               pass_count, test_count, (pass_count == test_count) ? "[ALL PASS]" : "[SOME FAILED]");
        printf("=============================================================\n\n");
    }

    // ──── Upload to GPU ────
    // 度量項
    CHECK_CUDA( cudaMemcpy(dk_dz_d,   dk_dz_h,   NYD6*NZ6*sizeof(double),      cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dk_dy_d,   dk_dy_h,   NYD6*NZ6*sizeof(double),      cudaMemcpyHostToDevice) );
    // Precomputed Lagrange weights → GPU (layout: [q][c][idx_jk])
    {
        size_t w_bytes = (size_t)7 * 19 * NYD6 * NZ6 * sizeof(double);
        CHECK_CUDA( cudaMemcpy(lagrange_eta_d,  lagrange_eta_h,  w_bytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(lagrange_xi_d,   lagrange_xi_h,   w_bytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(lagrange_zeta_d, lagrange_zeta_h, w_bytes, cudaMemcpyHostToDevice) );
    }
    // Precomputed stencil base k → GPU
    CHECK_CUDA( cudaMemcpy(bk_precomp_d, bk_precomp_h, NZ6*sizeof(int), cudaMemcpyHostToDevice) );

#if USE_MRT
    // Phase 3.5: MRT transformation matrices → __constant__ memory
    {
        Matrix;           // MRT_Matrix.h 嬉集 → double M[19][19] = { ... };
        Inverse_Matrix;   // MRT_Matrix.h 嬉集 → double Mi[19][19] = { ... };
        CHECK_CUDA( cudaMemcpyToSymbol(GILBM_M,  M,  sizeof(M)) );
        CHECK_CUDA( cudaMemcpyToSymbol(GILBM_Mi, Mi, sizeof(Mi)) );
        if (myid == 0) printf("GILBM-MRT: M[19x19] and Mi[19x19] copied to __constant__ memory (from MRT_Matrix.h).\n");
    }
#endif

    // Phase 4: LTS fields to GPU (omega_local is 2D; omegadt_local_d is 3D, filled by Init_OmegaDt_Kernel)
    CHECK_CUDA( cudaMemcpy(dt_local_d,      dt_local_h,      NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(omega_local_d,   omega_local_h,   NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice) );

    if (myid == 0) printf("GILBM: Lagrange weights + bk_precomp + dk + LTS fields copied to GPU.\n");

    if ( INIT == 0 ) {
        printf("Initializing by default function...\n");
        InitialUsingDftFunc();
    } else if ( INIT == 1 ) {
        printf("Initializing by backup data...\n");
        result_readbin_velocityandf();
        if( TBINIT && TBSWITCH ) statistics_readbin_merged_stress();
    } else if ( INIT == 2 ) {
        printf("Initializing from merged VTK: %s\n", RESTART_VTK_FILE);
        InitFromMergedVTK(RESTART_VTK_FILE);
    }

    // ---- Perturbation injection: break spanwise symmetry to trigger 3D turbulence ----//加入擾動量
    // 使用 additive δfeq 方法: f[q] += feq(ρ, u+δu) - feq(ρ, u)
    // 保留已發展流場的非平衡部分 (viscous stress), 只注入速度擾動
#if PERTURB_INIT
    {
        double e_lbm[19][3] = {
            {0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
            {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},{1,0,1},{-1,0,1},{1,0,-1},
            {-1,0,-1},{0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}};
        double W_lbm[19] = {
            1.0/3,  1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18,
            1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36,
            1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36};

        double amp = (PERTURB_PERCENT / 100.0) * (double)Uref;
        // 每個 rank 用不同的 seed → 不同的擾動 pattern
        srand(42 + myid * 13579);

        int count = 0;
        for (int j = 3; j < NYD6 - 3; j++)
        for (int k = 3; k < NZ6 - 3; k++)
        for (int i = 3; i < NX6 - 3; i++) { //遍歷每一個物理空間計算點 
            int index = j * NX6 * NZ6 + k * NX6 + i;

            // 壁面距離 envelope: sin(π·z_norm), 壁面=0, 中心=1
            double z_bot  = z_h[j * NZ6 + 3];
            double z_top  = z_h[j * NZ6 + (NZ6 - 4)];
            double z_norm = (z_h[j * NZ6 + k] - z_bot) / (z_top - z_bot);
            double envelope = sin(pi * z_norm);

            // 三分量隨機擾動 [-amp, +amp] × envelope
            double du = amp * envelope * (2.0 * rand() / (double)RAND_MAX - 1.0);
            double dv = amp * envelope * (2.0 * rand() / (double)RAND_MAX - 1.0);
            double dw = amp * envelope * (2.0 * rand() / (double)RAND_MAX - 1.0);

            double rho_p = rho_h_p[index];
            double u_old = u_h_p[index], v_old = v_h_p[index], w_old = w_h_p[index];
            double u_new = u_old + du,    v_new = v_old + dv,    w_new = w_old + dw;

            // 更新宏觀速度
            u_h_p[index] = u_new;
            v_h_p[index] = v_new;
            w_h_p[index] = w_new;

            // Additive δfeq: 保留 f_neq, 只加入擾動的平衡態差值
            //S_{i}= (feq(ρ, u+δu) - feq(ρ, u)) 相當於一個外力進去，理論根據 : Kupershtokh2004-
            double udot_old = u_old * u_old + v_old * v_old + w_old * w_old;
            double udot_new = u_new * u_new + v_new * v_new + w_new * w_new;
            for (int q = 0; q < 19; q++) {
                double eu_old = e_lbm[q][0]*u_old + e_lbm[q][1]*v_old + e_lbm[q][2]*w_old;
                double eu_new = e_lbm[q][0]*u_new + e_lbm[q][1]*v_new + e_lbm[q][2]*w_new;
                double feq_old = W_lbm[q] * rho_p * (1.0 + 3.0*eu_old + 4.5*eu_old*eu_old - 1.5*udot_old);
                double feq_new = W_lbm[q] * rho_p * (1.0 + 3.0*eu_new + 4.5*eu_new*eu_new - 1.5*udot_new);
                fh_p[q][index] += (feq_new - feq_old);
            }
            count++;
        }
        if (myid == 0)
            printf("Perturbation injected: amp=%.2e (%d%% Uref), %d interior points/rank, envelope=sin(pi*z_norm)\n",
                   amp, (int)PERTURB_PERCENT, count);
    }
#endif

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

    // Restore time-average from VTK restart (if available)
    if (restart_step > 0 && vel_avg_count > 0) {
        // tavg_h arrays contain averaged values from VTK; multiply by count to get cumulative sums
        const size_t nTotal = (size_t)NX6 * NYD6 * NZ6;
        for (size_t idx = 0; idx < nTotal; idx++) {
            u_tavg_h[idx] *= (double)vel_avg_count;
            v_tavg_h[idx] *= (double)vel_avg_count;
            w_tavg_h[idx] *= (double)vel_avg_count;
        }
        // Copy accumulated sums to GPU
        const size_t tavg_bytes = nTotal * sizeof(double);
        CHECK_CUDA( cudaMemcpy(u_tavg_d, u_tavg_h, tavg_bytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(v_tavg_d, v_tavg_h, tavg_bytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(w_tavg_d, w_tavg_h, tavg_bytes, cudaMemcpyHostToDevice) );
        if (myid == 0)
            printf("Velocity time-average restored from VTK: vel_avg_count=%d, copied to GPU (%.1f MB each).\n",
                   vel_avg_count, tavg_bytes / 1.0e6);
        stage1_announced = true;
    } else {
        if (myid == 0) {
            size_t nTotal = (size_t)NX6 * NYD6 * NZ6;
            printf("Time-average arrays allocated (%.1f MB each), starting fresh.\n",
                   nTotal * sizeof(double) / 1.0e6);
        }
    }

    // Restore Reynolds stress from merged binary checkpoint (if available)
    if (restart_step > 0 && rey_avg_count > 0 && (int)TBSWITCH) {
        statistics_readbin_merged_stress();
        stage2_announced = true;
        if (myid == 0)
            printf("Reynolds stress restored from binary checkpoint (rey_avg_count=%d)\n", rey_avg_count);
    }

    // FTT-gate check: discard old statistics if restart FTT is below threshold
    // 防止從舊版 VTK (無 FTT gating) 繼承的污染數據在 FTT < threshold 時持續輸出
    if (restart_step > 0) {
        double FTT_restart = (double)restart_step * dt_global / (double)flow_through_time;
        const size_t nTotal_gate = (size_t)NX6 * NYD6 * NZ6;
        const size_t tavg_bytes_gate = nTotal_gate * sizeof(double);

        if (FTT_restart < FTT_STAGE1 && vel_avg_count > 0) {
            if (myid == 0)
                printf("[FTT-GATE] FTT_restart=%.2f < FTT_STAGE1=%.1f: discarding old velocity averages (vel_avg_count=%d -> 0)\n",
                       FTT_restart, FTT_STAGE1, vel_avg_count);
            vel_avg_count = 0;
            stage1_announced = false;
            memset(u_tavg_h, 0, tavg_bytes_gate);
            memset(v_tavg_h, 0, tavg_bytes_gate);
            memset(w_tavg_h, 0, tavg_bytes_gate);
            CHECK_CUDA( cudaMemset(u_tavg_d, 0, tavg_bytes_gate) );
            CHECK_CUDA( cudaMemset(v_tavg_d, 0, tavg_bytes_gate) );
            CHECK_CUDA( cudaMemset(w_tavg_d, 0, tavg_bytes_gate) );
        }
        if (FTT_restart < FTT_STAGE2 && rey_avg_count > 0) {
            if (myid == 0)
                printf("[FTT-GATE] FTT_restart=%.2f < FTT_STAGE2=%.1f: discarding old RS data (rey_avg_count=%d -> 0)\n",
                       FTT_restart, FTT_STAGE2, rey_avg_count);
            rey_avg_count = 0;
            stage2_announced = false;
            // TBSWITCH arrays already cudaMemset'd to 0 in AllocateMemory; readbin was skipped or data is stale
        }
    }

    CHECK_CUDA( cudaEventRecord(start,0) );
	CHECK_CUDA( cudaEventRecord(start1,0) );
    // 續跑初始狀態輸出: 從 CPU 資料計算 Ub，完整顯示重啟狀態
    if (restart_step > 0) {
        // Compute Ub from CPU data (rank 0 only, j=3 hill-crest cross-section)
        // 同 AccumulateUbulk kernel: Σ v(j=3,k,i) * dx * dz / (LX*(LZ-1))
        double Ub_init = 0.0;
        if (myid == 0) {
            for (int k = 3; k < NZ6-3; k++) {
            for (int i = 3; i < NX6-4; i++) {
                int index = 3 * NX6 * NZ6 + k * NX6 + i;
                double dx_loc = (x_h[i+1] - x_h[i-1]) / 2.0;
                double dz_loc = (z_h[3*NZ6 + k+1] - z_h[3*NZ6 + k-1]) / 2.0;
                Ub_init += v_h_p[index] * dx_loc * dz_loc;
            }}
            Ub_init /= (double)(LX * (LZ - 1.0));
        }
        MPI_Bcast(&Ub_init, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Ub_avg_global = Ub_init;

        // Ma_max: 需所有 rank 參與 MPI_Allreduce，放在 if(myid==0) 之外
        double Ma_max_init = ComputeMaMax();

        if (myid == 0) {
            double FTT_init = (double)restart_step * dt_global / (double)flow_through_time;
            double Ustar = Ub_init / (double)Uref;
            double Fstar = Force_h[0] * (double)LY / ((double)Uref * (double)Uref);
            double Re_now = Ub_init / ((double)Uref / (double)Re);
            double Ma_init = Ub_init / (double)cs;

            printf("+----------------------------------------------------------------+\n");
            printf("| Step = %d    FTT = %.2f\n", restart_step, FTT_init);
            printf("|%s running with %4dx%4dx%4d grids\n", argv[0], (int)NX6, (int)NY6, (int)NZ6);
            printf("| Loop %d more steps, end at step %d\n", (int)loop, restart_step + 1 + (int)loop);
            printf("+----------------------------------------------------------------+\n");
            printf("[Step %d | FTT=%.2f] Ub=%.6f  U*=%.4f  Force=%.5E  F*=%.4f  Re(now)=%.1f  Ma=%.4f  Ma_max=%.4f\n",
                   restart_step, FTT_init, Ub_init, Ustar, Force_h[0], Fstar, Re_now, Ma_init, Ma_max_init);

            if (Ma_max_init > 0.35)
                printf("  >>> [WARNING] Ma_max=%.4f > 0.35 — BGK stability risk, consider reducing Uref\n", Ma_max_init);

            if (Ustar > 1.3)
                printf("  >>> [NOTE] U*=%.4f >> 1.0 — VTK velocity from old Uref, flow will decelerate to new target\n", Ustar);
        }
        // Anti-windup Force cap (顯示原始狀態後才套用，避免前 NDTFRC 步用過高外力)
        {
            double h_eff = (double)LZ - (double)H_HILL;
            double Force_initial  = 8.0 * (double)niu * (double)Uref / (h_eff * h_eff) * 3.0;
            if (Force_h[0] > Force_initial) {
                if (myid == 0)
                    printf("[ANTI-WINDUP] Force capped: %.5E -> %.5E (max=3x Poiseuille)\n",
                           Force_h[0], Force_initial);
                Force_h[0] = Force_initial;
                CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
            }
        }
        // 輸出初始 VTK (驗證重啟載入正確 + 使用修正後的 stride mapping)
        fileIO_velocity_vtk_merged(restart_step);
    }
    // VTK step 為奇數 (step%1000==1), for-loop 須從偶數開始
    // 以確保 step+=1 後 monitoring 在奇數步 (step%N==1) 正確觸發
    int loop_start = (restart_step > 0) ? restart_step + 1 : 0;




    //從此開始進入迴圈 (FTT-gated two-stage time averaging)
    for( step = loop_start ; step < loop_start + loop ; step++, accu_num++ ) {
        double FTT_now = step * dt_global / (double)flow_through_time;

        // ===== Sub-step 1: even step (ft → fd) =====
        Launch_CollisionStreaming( ft, fd );

        // Stage 2: MeanVars + MeanDerivatives (FTT >= FTT_STAGE2)
        if (FTT_now >= FTT_STAGE2 && (int)TBSWITCH) {
            Launch_TurbulentSum( fd );
            rey_avg_count++;
        }

        CHECK_CUDA( cudaDeviceSynchronize() );

        // Stage 1: velocity mean (FTT >= FTT_STAGE1)
        if (FTT_now >= FTT_STAGE1 && step > 0) {
            Launch_AccumulateTavg();
            vel_avg_count++;
        }

        // Stage transition messages
        if (!stage1_announced && FTT_now >= FTT_STAGE1) {
            stage1_announced = true;
            if (myid == 0) printf("\n>>> [FTT=%.2f] STAGE 1: Mean velocity accumulation STARTED <<<\n\n", FTT_now);
        }
        if (!stage2_announced && FTT_now >= FTT_STAGE2) {
            stage2_announced = true;
            if (myid == 0) printf("\n>>> [FTT=%.2f] STAGE 2: Reynolds stress accumulation STARTED <<<\n\n", FTT_now);
        }

        // ===== Sub-step 2: odd step (fd → ft) =====
        step += 1;
        accu_num += 1;
        FTT_now = step * dt_global / (double)flow_through_time;

        Launch_CollisionStreaming( fd, ft );

        if (FTT_now >= FTT_STAGE2 && (int)TBSWITCH) {
            Launch_TurbulentSum( ft );
            rey_avg_count++;
        }

        CHECK_CUDA( cudaDeviceSynchronize() );

        if (FTT_now >= FTT_STAGE1) {
            Launch_AccumulateTavg();
            vel_avg_count++;
        }

        // ===== Status display (every 5000 steps) =====
        if ( myid == 0 && step%5000 == 1 ) {
            CHECK_CUDA( cudaEventRecord( stop1,0 ) );
            CHECK_CUDA( cudaEventSynchronize( stop1 ) );
			float cudatime1;
			CHECK_CUDA( cudaEventElapsedTime( &cudatime1,start1,stop1 ) );

            printf("+----------------------------------------------------------------+\n");
			printf("| Step = %d    FTT = %.2f \n", step, FTT_now);
            printf("|%s running with %4dx%4dx%4d grids            \n", argv[0], (int)NX6, (int)NY6, (int)NZ6 );
            printf("| Running %6f mins                                           \n", (cudatime1/60/1000) );
            printf("| Stage: %s | %s\n",
                   (FTT_now >= FTT_STAGE1) ? "VelAvg ON" : "VelAvg OFF",
                   (FTT_now >= FTT_STAGE2) ? "RS ON" : "RS OFF");
            printf("| vel_avg_count=%d  rey_avg_count=%d\n", vel_avg_count, rey_avg_count);
            printf("+----------------------------------------------------------------+\n");

            cudaEventRecord(start1,0);
        }

        // ===== Force modification (every NDTFRC steps) =====
        if ( (step%(int)NDTFRC == 1) ) {
            Launch_ModifyForcingTerm();
        }

		if ( step%(int)NDTMIT == 1 ) {
			Launch_Monitor();
		}

        // ===== VTK output + binary checkpoint (every 1000 steps) =====
        if ( step % 1000 == 1 ) {
            SendDataToCPU( ft );
            // Copy GPU tavg → host for VTK output
            const size_t tavg_bytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
            CHECK_CUDA( cudaMemcpy(u_tavg_h, u_tavg_d, tavg_bytes, cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(v_tavg_h, v_tavg_d, tavg_bytes, cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(w_tavg_h, w_tavg_d, tavg_bytes, cudaMemcpyDeviceToHost) );

            // VTK-step status
            double Ma_max_vtk = ComputeMaMax();
            if (myid == 0) {
                double Ub_vtk = 0.0;
                for (int kk = 3; kk < NZ6-3; kk++)
                for (int ii = 3; ii < NX6-4; ii++) {
                    int idx = 3*NX6*NZ6 + kk*NX6 + ii;
                    double dx_loc = (x_h[ii+1] - x_h[ii-1]) / 2.0;
                    double dz_loc = (z_h[3*NZ6+kk+1] - z_h[3*NZ6+kk-1]) / 2.0;
                    Ub_vtk += v_h_p[idx] * dx_loc * dz_loc;
                }
                Ub_vtk /= (double)(LX * (LZ - 1.0));
                printf("[Step %d | FTT=%.2f] Ub=%.6f  U*=%.4f  Force=%.5E  F*=%.4f  Re=%.1f  Ma=%.4f  Ma_max=%.4f  vel=%d  rey=%d\n",
                       step, FTT_now,
                       Ub_vtk, Ub_vtk / (double)Uref, Force_h[0],
                       Force_h[0] * (double)LY / ((double)Uref * (double)Uref),
                       Ub_vtk / ((double)Uref / (double)Re),
                       Ub_vtk / (double)cs, Ma_max_vtk, vel_avg_count, rey_avg_count);
            }

            fileIO_velocity_vtk_merged( step );
            // Note: BIN checkpoint removed from periodic output.
            // Merged BIN is written ONLY on program stop (final exit block).
        }

        // ===== Global Mass Conservation Modify =====
        cudaDeviceSynchronize();
        cudaMemcpy(Force_h, Force_d, sizeof(double), cudaMemcpyDeviceToHost);
        {
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
        }

        // ===== Mass Conservation Check + NaN early stop (every 100 steps) =====
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
                SendDataToCPU( ft );
                fileIO_velocity_vtk_merged( step );
                break;
            }

            if( myid ==0 ){
                double FTT_rho = step * dt_global / (double)flow_through_time;
                FILE *checkrho;
                checkrho = fopen("checkrho.dat","a");
                fprintf(checkrho,"%d\t %.4f\t %lf\t %lf\n",step, FTT_rho, rho_initial, rho_GlobalSum / (double)jp);
                fclose (checkrho);
            }
        }

        // ===== FTT stopping criterion =====
        if (FTT_now >= FTT_STOP) {
            if (myid == 0)
                printf("\n[FTT-STOP] FTT=%.2f >= FTT_STOP=%.1f at step %d. Ending simulation.\n",
                       FTT_now, FTT_STOP, step);
            break;
        }
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    // ===== Final exit checkpoint: always save state =====
    {
        double FTT_final = step * dt_global / (double)flow_through_time;
        SendDataToCPU( ft );
        result_writebin_velocityandf();

        // Copy GPU tavg → host for final VTK
        const size_t tavg_bytes_final = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
        CHECK_CUDA( cudaMemcpy(u_tavg_h, u_tavg_d, tavg_bytes_final, cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaMemcpy(v_tavg_h, v_tavg_d, tavg_bytes_final, cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaMemcpy(w_tavg_h, w_tavg_d, tavg_bytes_final, cudaMemcpyDeviceToHost) );
        fileIO_velocity_vtk_merged( step );

        // Write Reynolds stress statistics if accumulated (FTT > FTT_STAGE2)
        if (rey_avg_count > 0 && (int)TBSWITCH) {
            double FTT_rs = (double)rey_avg_count * dt_global / (double)flow_through_time;
            if (myid == 0) {
                printf("\n========================================================\n");
                printf("[FINAL OUTPUT] FTT = %.3f (timestep = %d)\n", FTT_final, step);
                printf("  -> Velocity mean accumulation: %d steps\n", vel_avg_count);
                printf("  -> Reynolds stress accumulation: %.3f FTTs (%d steps)\n", FTT_rs, rey_avg_count);
                printf("  -> Writing merged statistics (35 arrays)\n");
                printf("========================================================\n\n");
            }
            statistics_writebin_merged_stress();
        } else if (myid == 0) {
            printf("\n[FINAL] FTT=%.2f, step=%d. No RS data to write (rey_avg_count=%d).\n",
                   FTT_final, step, rey_avg_count);
        }
    }

    free(u_tavg_h);
    free(v_tavg_h);
    free(w_tavg_h);
    FreeSource();
    MPI_Finalize();

    return 0;
}
