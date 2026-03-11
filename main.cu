#include <time.h>
#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdarg.h>
#include "variables.h"
#include "MRT_Matrix_D3Q27.h"   // NQ=27, D3Q27_ex/ey/ez/W (needed by all host-side headers)
#if USE_WP_CUMULANT
#include "Cumulants/cumulant_wp_diagnostic.h"
#endif
using namespace std;
/************************** Host Variables **************************/
double  *fh_p[NQ]; //主機端一般態分佈函數 (D3Q27)
double  *rho_h_p,  *u_h_p,  *v_h_p,  *w_h_p;


/************************** Device Variables **************************/
double  *ft[NQ], *fd[NQ];
double  **ft_ptrs_d, **fd_ptrs_d;  // Device pointer arrays for kernel calls
double  *rho_d,  *u,  *v,  *w;

/* double  *KT,    *DISS,
        *DUDX2, *DUDY2, *DUDZ2,
        *DVDX2, *DVDY2, *DVDZ2,
        *DWDX2, *DWDY2, *DWDZ2; */
double  *U,  *V,  *W,  *P, 
        *UU, *UV, *UW, *VV, *VW, *WW, *PU, *PV, *PW, *PP,
        *DUDX2, *DUDY2, *DUDZ2,
        *DVDX2, *DVDY2, *DVDZ2, 
        *DWDX2, *DWDY2, *DWDZ2,
        *UUU, *UUV, *UUW, *UVW,
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
double *delta_zeta_h, *delta_zeta_d;  // GILBM RK2 ζ-direction displacement [NQ*NYD6*NZ6]
double delta_xi_h[NQ];               // GILBM ξ-direction displacement (global dt, for initial CFL check)
double delta_eta_h[NQ];              // GILBM η-direction displacement (global dt, for initial CFL check)

// Precomputed stencil base k [NZ6] (int, wall-clamped)
int *bk_precomp_h, *bk_precomp_d;

// Phase 3: Curvilinear global time step (runtime, from CFL on contravariant velocities)
// NOTE: dt (= minSize) is a compile-time macro in variables.h for defining ν.
//       dt_global is the actual curvilinear time step, computed at runtime.
double dt_global;
double omega_global;     // = 3·niu/dt_global + 0.5 (dimensionless relaxation time)
double omegadt_global;   // = omega_global · dt_global (dimensional relaxation time τ)
int force_check_interval;  // FT/10 in steps (Re%-based force control check interval)

// GILBM two-pass architecture: persistent global arrays
double *feq_d;            // Equilibrium distribution [NQ * NX6*NYD6*NZ6]
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

// GPU reduction partial sums for mass conservation (replaces SendDataToCPU every step)
double *rho_partial_h, *rho_partial_d;

// Time-average accumulation (FTT-gated)
// u=spanwise, v=streamwise, w=wall-normal; GPU-side accumulation
double *u_tavg_h = NULL, *v_tavg_h = NULL, *w_tavg_h = NULL;   // host (for VTK output)
double *u_tavg_d = NULL, *v_tavg_d = NULL, *w_tavg_d = NULL;   // device (accumulated on GPU)
// Vorticity mean accumulation (same Stage 1 window as velocity mean)
double *ox_tavg_h = NULL, *oy_tavg_h = NULL, *oz_tavg_h = NULL; // host
double *ox_tavg_d = NULL, *oy_tavg_d = NULL, *oz_tavg_d = NULL; // device
int accu_count = 0;         // Unified statistics accumulation count (FTT >= FTT_STATS_START)
bool stage1_announced = false;

int nProcs, myid;

int step;
int restart_step = 0;  // 續跑起始步 (INIT=2 時從 VTK header 解析)
int accu_num = 0;
// ub_accu_count removed — Launch_ModifyForcingTerm now uses instantaneous Ub

int l_nbr, r_nbr;

MPI_Status    istat[8];

MPI_Request   request[NQ+4][4];
MPI_Status    status[NQ+4][4];

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

MPI_Request reqToLeft[NQ+4], reqToRight[NQ+4],   reqFromLeft[NQ+4], reqFromRight[NQ+4];
MPI_Request reqToTop[NQ+4],  reqToBottom[NQ+4],  reqFromTop[NQ+4],  reqFromBottom[NQ+4];
// MPI tags: NQ direction tags + 4 macroscopic (u,v,w,rho)
int itag_f3[NQ+4], itag_f4[NQ+4], itag_f5[NQ+4], itag_f6[NQ+4];
// Initialize tag arrays (done before main via static init helper)
struct TagInit {
    TagInit() {
        for (int i = 0; i < NQ+4; i++) {
            itag_f3[i] = 250 + i;
            itag_f4[i] = 200 + i;
            itag_f5[i] = 300 + i;
            itag_f6[i] = 400 + i;
        }
    }
} _tag_init;


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
int main(int argc, char *argv[])
{
    CHECK_MPI( MPI_Init(&argc, &argv) );
    CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &nProcs) );
    CHECK_MPI( MPI_Comm_rank(MPI_COMM_WORLD, &myid) );

    // Safety check: compiled jp must match runtime MPI rank count
    if (nProcs != jp) {
        if (myid == 0)
            fprintf(stderr, "FATAL: nProcs=%d but compiled with jp=%d. Recompile with correct jp!\n", nProcs, jp);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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
        ox_tavg_h = (double*)calloc(nTotal, sizeof(double));
        oy_tavg_h = (double*)calloc(nTotal, sizeof(double));
        oz_tavg_h = (double*)calloc(nTotal, sizeof(double));
        accu_count = 0;
    }

    //pre-check whether the directories exit or not
    PreCheckDir();
    CreateDataType();
    //generate mesh and coordinates of each point
	GenerateMesh_X();
    GenerateMesh_Y();
    GenerateMesh_Z();

    // 初始化 monitor RS 代表點 (需在 GenerateMesh_Z 之後, z_h 已填充)
    InitMonitorCheckPoint();

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

    // Force check interval: NDTFRC steps
    force_check_interval = NDTFRC;

    if (myid == 0) {
        printf("  ─────────────────────────────────────────────────────────\n");
        printf("  dt_global = MIN(all ranks) = %.6e\n", dt_global);
        printf("  dt_old = minSize = %.6e\n", (double)minSize);
        printf("  ratio dt_global / minSize = %.4f\n", dt_global / (double)minSize);
        printf("  Speedup cost: %.1fx more timesteps per physical time\n", (double)minSize / dt_global);
        printf("  omega_global = %.6f, 1/omega_global = %.6f\n", omega_global, 1.0/omega_global);
        printf("  Force check interval = %d steps (NDTFRC)\n", force_check_interval);
        printf("  =============================================================\n\n");
    }

    // GILBM: 預計算三方向位移 δη (常數), δξ (常數), δζ (RK2 空間變化, 使用 dt_global)
    PrecomputeGILBM_DeltaAll(delta_xi_h, delta_eta_h, delta_zeta_h,
                              dk_dz_h, dk_dy_h, NYD6, NZ6, dt_global );

    // GTS: CFL validation — departure point safety check
    bool cfl_ok = ValidateDepartureCFL(delta_zeta_h, dk_dy_h, dk_dz_h, NYD6, NZ6, myid);
    if (!cfl_ok && myid == 0) {
        fprintf(stderr,
            "[WARNING] CFL_zeta >= 1.0 still detected after Imamura time step.\n"
            "  This should not happen — check ComputeGlobalTimeStep logic.\n");
    }

    // Precompute stencil base k (wall-clamped, depends only on k)
    PrecomputeGILBM_StencilBaseK(bk_precomp_h, NZ6);

    // ──── Upload to GPU ────
    // 度量項
    CHECK_CUDA( cudaMemcpy(dk_dz_d,   dk_dz_h,   NYD6*NZ6*sizeof(double),      cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dk_dy_d,   dk_dy_h,   NYD6*NZ6*sizeof(double),      cudaMemcpyHostToDevice) );
    // delta_zeta → GPU (ζ displacements, space-varying, used by kernel Step 1)
    CHECK_CUDA( cudaMemcpy(delta_zeta_d, delta_zeta_h, NQ*NYD6*NZ6*sizeof(double), cudaMemcpyHostToDevice) );
    // __constant__ symbols: dt_global, delta_eta[NQ], delta_xi[NQ]
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_dt,        &dt_global,   sizeof(double)) );
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_delta_eta, delta_eta_h,  NQ*sizeof(double)) );
    CHECK_CUDA( cudaMemcpyToSymbol(GILBM_delta_xi,  delta_xi_h,   NQ*sizeof(double)) );
    // Precomputed stencil base k → GPU
    CHECK_CUDA( cudaMemcpy(bk_precomp_d, bk_precomp_h, NZ6*sizeof(int), cudaMemcpyHostToDevice) );

#if USE_MRT
    // D3Q27 MRT: compute transformation matrices via Gram-Schmidt, then upload
    {
        double M_h[NQ][NQ], Mi_h[NQ][NQ];
        ComputeD3Q27_MRT_Matrices(M_h, Mi_h);

        // Verify M × M_inv = I
        double id_err = VerifyMRT27_Identity(M_h, Mi_h);
        if (myid == 0) printf("D3Q27 MRT: M*Mi identity check: max_err = %.2e\n", id_err);
        if (id_err > 1e-12 && myid == 0)
            printf("[WARNING] MRT matrix identity error > 1e-12!\n");

        CHECK_CUDA( cudaMemcpyToSymbol(GILBM_M,  M_h,  sizeof(M_h)) );
        CHECK_CUDA( cudaMemcpyToSymbol(GILBM_Mi, Mi_h, sizeof(Mi_h)) );

        // Setup relaxation rates: s_visc = 1/tau = 1/(3*nu/dt + 0.5)
        double s_visc = 1.0 / omega_global;
        double S_h[NQ];
        SetupD3Q27_Relaxation(S_h, s_visc);
        CHECK_CUDA( cudaMemcpyToSymbol(GILBM_S, S_h, sizeof(S_h)) );

        if (myid == 0) {
            printf("D3Q27 MRT: M[27x27], Mi[27x27], S[27] -> __constant__ memory.\n");
            printf("  s_visc = 1/omega = %.6f (viscosity moments 4-9)\n", s_visc);
        }
    }
#endif

#if USE_CUMULANT
    if (myid == 0) {
#if USE_WP_CUMULANT
        printf("D3Q27 Cumulant (WP mode): omega=%.6f, lambda=%.2e\n",
               omega_global, (double)CUM_LAMBDA);
        printf("  WP: omega3-5 parameterized (Eq.14-16), A/B coeffs (Eq.17-18), lambda-limiter (Eq.20-26)\n");
#else
        printf("D3Q27 Cumulant (AO mode): omega=%.6f\n", omega_global);
        printf("  AO: omega2-10 = 1.0 (All-One, Geier 2015)\n");
#endif
    }

#if USE_WP_CUMULANT
    // ── 預計算奇異點診斷 (host-side, rank 0 only) ──
    // 自動從 {Re, Uref, dt_global} 計算所有鬆弛率並檢查是否接近奇異點
    // omega2 = CUM_OMEGA2 (variables.h 全域巨集，碰撞與診斷統一)
    CumulantWP_DiagnoseOmega((int)Re, (double)Uref, dt_global, (double)CUM_OMEGA2, myid);
#endif
#endif

    if (myid == 0) printf("GILBM: delta_zeta + __constant__(dt,eta,xi) + bk_precomp + dk copied to GPU.\n");

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
    } else if ( INIT == 3 ) {
        printf("Initializing from binary checkpoint: %s\n", RESTART_BIN_DIR);
        LoadBinaryCheckpoint(RESTART_BIN_DIR);
    }

    // Force sanity guard (applies to all restart paths: INIT=1 binary, INIT=2 VTK)
    // INIT=2 has additional anti-windup cap later (after restart display); this covers INIT=1
    if (INIT > 0) {
        if (isnan(Force_h[0]) || isinf(Force_h[0])) {
            if (myid == 0) printf("[FORCE-GUARD] Invalid Force=%.5E (NaN/Inf), reset to 0\n", Force_h[0]);
            Force_h[0] = 0.0;
            CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
        } else if (Force_h[0] < 0.0) {
            if (myid == 0) printf("[FORCE-GUARD] Negative Force=%.5E, clamped to 0\n", Force_h[0]);
            Force_h[0] = 0.0;
            CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
        }
        // Anti-windup cap for binary restarts (INIT=1 or INIT=3)
        if (INIT == 1 || INIT == 3) {
            double h_eff = (double)LZ - (double)H_HILL;
            double Force_Poiseuille = 8.0 * (double)niu * (double)Uref / (h_eff * h_eff);
            double Force_cap = Force_Poiseuille * 100.0;
            if (Force_h[0] > Force_cap) {
                if (myid == 0)
                    printf("[ANTI-WINDUP] Binary restart Force capped: %.5E -> %.5E (100x Poiseuille)\n",
                           Force_h[0], Force_cap);
                Force_h[0] = Force_cap;
                CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
            }
        }
    }

    // ---- Perturbation injection: break spanwise symmetry to trigger 3D turbulence ----//加入擾動量
    // 使用 additive δfeq 方法: f[q] += feq(ρ, u+δu) - feq(ρ, u)
    // 保留已發展流場的非平衡部分 (viscous stress), 只注入速度擾動
#if PERTURB_INIT
    {
        // Use D3Q27 velocity vectors and weights from MRT_Matrix_D3Q27.h
        const int    *ex_lbm = D3Q27_ex;
        const int    *ey_lbm = D3Q27_ey;
        const int    *ez_lbm = D3Q27_ez;
        const double *W_lbm  = D3Q27_W;

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
            for (int q = 0; q < NQ; q++) {
                double eu_old = (double)ex_lbm[q]*u_old + (double)ey_lbm[q]*v_old + (double)ez_lbm[q]*w_old;
                double eu_new = (double)ex_lbm[q]*u_new + (double)ey_lbm[q]*v_new + (double)ez_lbm[q]*w_new;
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
    DiagnoseGILBM_Phase1(delta_xi_h, delta_zeta_h, dk_dz_h, dk_dy_h, fh_p, NYD6, NZ6, myid, dt_global, INIT);

    SendDataToGPU();

    // === Ub integration self-test (runs every startup, aborts on failure) ===
    {
        int ub_test_fail = 0;
        if (myid == 0) {
            // Test 1: Σ dx_cell × dz_cell must = LX × (LZ - 1.0) (telescoping sum identity)
            double A_sum = 0.0;
            for (int k = 3; k < NZ6-4; k++)
            for (int i = 3; i < NX6-4; i++)
                A_sum += (x_h[i+1] - x_h[i]) * (z_h[3*NZ6+k+1] - z_h[3*NZ6+k]);
            double A_exact = LX * (LZ - 1.0);
            double area_err = fabs(A_sum - A_exact) / A_exact;
            printf("[Ub-CHECK] Test 1 — Area:    Sum=%.12f  Exact=%.12f  rel_err=%.2e  %s\n",
                   A_sum, A_exact, area_err, (area_err < 1e-12) ? "PASS" : "FAIL");
            if (area_err >= 1e-12) {
                fprintf(stderr, "[FATAL] Ub area sum mismatch: x_h or z_h boundary values are wrong.\n");
                ub_test_fail = 1;
            }

            // Test 2: Uniform v=Uref → Ub must = Uref (integration + normalization check)
            double Ub_test = 0.0;
            for (int k = 3; k < NZ6-4; k++)
            for (int i = 3; i < NX6-4; i++)
                Ub_test += (double)Uref * (x_h[i+1] - x_h[i]) * (z_h[3*NZ6+k+1] - z_h[3*NZ6+k]);
            Ub_test /= (double)(LX * (LZ - 1.0));
            double uniform_err = fabs(Ub_test - (double)Uref) / (double)Uref;
            printf("[Ub-CHECK] Test 2 — Uniform: Ub=%.15f  Uref=%.15f  rel_err=%.2e  %s\n",
                   Ub_test, (double)Uref, uniform_err, (uniform_err < 1e-12) ? "PASS" : "FAIL");
            if (uniform_err >= 1e-12) {
                fprintf(stderr, "[FATAL] Ub uniform field test failed: integration formula or normalization bug.\n");
                ub_test_fail = 1;
            }

            // Test 3: Actual field Ub (sanity: 0 < U* < 2)
            double Ub_actual = 0.0;
            for (int k = 3; k < NZ6-4; k++)
            for (int i = 3; i < NX6-4; i++) {
                double v00 = v_h_p[3*NX6*NZ6 + k*NX6 + i];
                double v10 = v_h_p[3*NX6*NZ6 + (k+1)*NX6 + i];
                double v01 = v_h_p[3*NX6*NZ6 + k*NX6 + (i+1)];
                double v11 = v_h_p[3*NX6*NZ6 + (k+1)*NX6 + (i+1)];
                double v_cell = (v00 + v10 + v01 + v11) / 4.0;
                Ub_actual += v_cell * (x_h[i+1] - x_h[i]) * (z_h[3*NZ6+k+1] - z_h[3*NZ6+k]);
            }
            Ub_actual /= (double)(LX * (LZ - 1.0));
            double Ustar_actual = Ub_actual / (double)Uref;
            int t3_ok = (Ub_actual >= 0.0 && Ustar_actual < 2.0) || (INIT == 0);  // cold start: Ub=0 is OK
            printf("[Ub-CHECK] Test 3 — Field:   Ub=%.10f  U*=%.6f  %s\n",
                   Ub_actual, Ustar_actual, t3_ok ? "PASS" : "WARNING");
            if (!t3_ok) {
                fprintf(stderr, "[WARNING] Ub field test: U*=%.4f outside [0,2). VTK data may be corrupted.\n", Ustar_actual);
                // Warning only — don't abort (flow might just not have developed yet)
            }
        }
        MPI_Bcast(&ub_test_fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ub_test_fail) {
            if (myid == 0) fprintf(stderr, "[FATAL] Ub self-test FAILED. Aborting.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // GILBM initialization: feq from initial f arrays
    {
        dim3 init_block(8, 8, 4);
        dim3 init_grid((NX6 + init_block.x - 1) / init_block.x,
                       (NYD6 + init_block.y - 1) / init_block.y,
                       (NZ6 + init_block.z - 1) / init_block.z);

        // Upload fd pointer array to device for Init_Feq_Kernel
        CHECK_CUDA( cudaMemcpy(fd_ptrs_d, fd, NQ * sizeof(double*), cudaMemcpyHostToDevice) );
        Init_Feq_Kernel<<<init_grid, init_block>>>(fd_ptrs_d, feq_d);

        CHECK_CUDA( cudaDeviceSynchronize() );
        if (myid == 0) printf("GILBM: feq initialized.\n");
    }

    // [GTS] Initial MPI exchange: fill y-direction ghost zones on GPU
    // Before the first time step, ft/fd ghost zones may be uninitialized
    // (restart files may store only interior points).
    // Exchange both ft and fd so that the first Launch_CollisionStreaming
    // (which reads ft as f_old) has valid ghost zones for interpolation.
    {
        ISend_LtRtBdry( ft, iToLeft,    l_nbr, itag_f4, 0, NQ, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 );
        IRecv_LtRtBdry( ft, iFromRight, r_nbr, itag_f4, 1, NQ, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 );
        ISend_LtRtBdry( ft, iToRight,   r_nbr, itag_f3, 2, NQ, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 );
        IRecv_LtRtBdry( ft, iFromLeft,  l_nbr, itag_f3, 3, NQ, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 );
        for (int i = 0; i < NQ+4; i++)
            CHECK_MPI( MPI_Waitall(4, request[i], status[i]) );

        ISend_LtRtBdry( fd, iToLeft,    l_nbr, itag_f4, 0, NQ, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 );
        IRecv_LtRtBdry( fd, iFromRight, r_nbr, itag_f4, 1, NQ, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 );
        ISend_LtRtBdry( fd, iToRight,   r_nbr, itag_f3, 2, NQ, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 );
        IRecv_LtRtBdry( fd, iFromLeft,  l_nbr, itag_f3, 3, NQ, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 );
        for (int i = 0; i < NQ+4; i++)
            CHECK_MPI( MPI_Waitall(4, request[i], status[i]) );

        // x-direction periodic BC for both buffers
        dim3 griddimSW(1, NYD6/NT+1, NZ6);
        dim3 blockdimSW(3, NT, 1);
        // Upload ft/fd pointer arrays to device for periodicSW
        CHECK_CUDA( cudaMemcpy(ft_ptrs_d, ft, NQ * sizeof(double*), cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(fd_ptrs_d, fd, NQ * sizeof(double*), cudaMemcpyHostToDevice) );

        periodicSW<<<griddimSW, blockdimSW>>>(
            ft_ptrs_d, y_d, x_d, z_d, u, v, w, rho_d, feq_d
        );
        periodicSW<<<griddimSW, blockdimSW>>>(
            fd_ptrs_d, y_d, x_d, z_d, u, v, w, rho_d, feq_d
        );
        CHECK_CUDA( cudaDeviceSynchronize() );

        if (myid == 0) printf("GILBM: Initial MPI ghost exchange completed (ft + fd).\n");
    }
    
    // ---- GILBM GTS Parameter Summary ----
    if (myid == 0) {
        printf("\n+================================================================+\n");
        printf("| GILBM Initialization Parameter Summary (GTS)                   |\n");
        printf("+================================================================+\n");
        printf("| [Input]  Re               = %d\n", (int)Re);
        printf("| [Input]  Uref             = %.6f\n", (double)Uref);
        printf("| [Output] niu              = %.6e\n", (double)niu);
        printf("+----------------------------------------------------------------+\n");
        printf("| [Output] dt_global        = %.6e\n", dt_global);
        printf("|   -> omega_global         = 3*niu/dt_global + 0.5     = %.6f\n", omega_global);
        printf("|   -> omegadt_global       = omega*dt                  = %.6e\n", omegadt_global);
        printf("+----------------------------------------------------------------+\n");
#if USE_CUMULANT
#if USE_WP_CUMULANT
        printf("| Collision = Cumulant-WP (Gehrke 2022), lambda=%.2e      |\n", (double)CUM_LAMBDA);
#else
        printf("| Collision = Cumulant-AO (Geier 2015)                       |\n");
#endif
#elif USE_MRT
        printf("| Collision = MRT (Suga 2015)                                |\n");
#else
        printf("| Collision = BGK/SRT                                        |\n");
#endif
        printf("+================================================================+\n\n");
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    // Restore time-average from restart (if available)
    if (restart_step > 0 && accu_count > 0) {
        const size_t nTotal = (size_t)NX6 * NYD6 * NZ6;
        const size_t tavg_bytes = nTotal * sizeof(double);
        if (INIT == 2) {
            // VTK stores averaged values (÷count÷Uref → ×Uref already done in reader)
            // Multiply by count to get cumulative sums
            for (size_t idx = 0; idx < nTotal; idx++) {
                u_tavg_h[idx] *= (double)accu_count;
                v_tavg_h[idx] *= (double)accu_count;
                w_tavg_h[idx] *= (double)accu_count;
            }
        }
        // INIT=3: binary checkpoint stores raw cumulative sums, 直接使用不需乘
        // Copy accumulated sums to GPU
        CHECK_CUDA( cudaMemcpy(u_tavg_d, u_tavg_h, tavg_bytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(v_tavg_d, v_tavg_h, tavg_bytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(w_tavg_d, w_tavg_h, tavg_bytes, cudaMemcpyHostToDevice) );
        if (myid == 0)
            printf("Statistics restored: accu_count=%d, copied to GPU (%.1f MB each). [%s]\n",
                   accu_count, tavg_bytes / 1.0e6, (INIT==3) ? "binary" : "VTK");
        stage1_announced = true;
    } else {
        if (myid == 0) {
            size_t nTotal = (size_t)NX6 * NYD6 * NZ6;
            printf("Time-average arrays allocated (%.1f MB each), starting fresh.\n",
                   nTotal * sizeof(double) / 1.0e6);
        }
    }

    // Restore Reynolds stress from merged binary (INIT=1,2 only; INIT=3 已在 LoadBinaryCheckpoint 內讀取)
    if (restart_step > 0 && accu_count > 0 && (int)TBSWITCH && INIT != 3) {
        statistics_readbin_merged_stress();
        if (myid == 0)
            printf("Reynolds stress restored from ./statistics/ (accu_count=%d)\n", accu_count);
    }

    // FTT-gate check: discard old statistics if restart FTT is below threshold
    if (restart_step > 0) {
        double FTT_restart = (double)restart_step * dt_global / (double)flow_through_time;
        const size_t nTotal_gate = (size_t)NX6 * NYD6 * NZ6;
        const size_t tavg_bytes_gate = nTotal_gate * sizeof(double);

        if (FTT_restart < FTT_STATS_START && accu_count > 0) {
            if (myid == 0)
                printf("[FTT-GATE] FTT_restart=%.2f < FTT_STATS_START=%.1f: discarding ALL old statistics (accu_count=%d -> 0)\n",
                       FTT_restart, FTT_STATS_START, accu_count);
            accu_count = 0;
            stage1_announced = false;
            memset(u_tavg_h, 0, tavg_bytes_gate);
            memset(v_tavg_h, 0, tavg_bytes_gate);
            memset(w_tavg_h, 0, tavg_bytes_gate);
            CHECK_CUDA( cudaMemset(u_tavg_d, 0, tavg_bytes_gate) );
            CHECK_CUDA( cudaMemset(v_tavg_d, 0, tavg_bytes_gate) );
            CHECK_CUDA( cudaMemset(w_tavg_d, 0, tavg_bytes_gate) );
            // Also clear vorticity mean
            if (ox_tavg_h) {
                memset(ox_tavg_h, 0, tavg_bytes_gate);
                memset(oy_tavg_h, 0, tavg_bytes_gate);
                memset(oz_tavg_h, 0, tavg_bytes_gate);
                CHECK_CUDA( cudaMemset(ox_tavg_d, 0, tavg_bytes_gate) );
                CHECK_CUDA( cudaMemset(oy_tavg_d, 0, tavg_bytes_gate) );
                CHECK_CUDA( cudaMemset(oz_tavg_d, 0, tavg_bytes_gate) );
            }
            // TBSWITCH arrays already cudaMemset'd to 0 in AllocateMemory
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
            // Bilinear cell-average: Σ v_cell × dx_cell × dz_cell / A_total
            for (int k = 3; k < NZ6-4; k++) {
            for (int i = 3; i < NX6-4; i++) {
                double v00 = v_h_p[3*NX6*NZ6 + k*NX6 + i];
                double v10 = v_h_p[3*NX6*NZ6 + (k+1)*NX6 + i];
                double v01 = v_h_p[3*NX6*NZ6 + k*NX6 + (i+1)];
                double v11 = v_h_p[3*NX6*NZ6 + (k+1)*NX6 + (i+1)];
                double v_cell = (v00 + v10 + v01 + v11) / 4.0;
                Ub_init += v_cell * (x_h[i+1] - x_h[i]) * (z_h[3*NZ6+k+1] - z_h[3*NZ6+k]);
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
            printf("| Simulation will run until FTT >= %.1f\n", (double)FTT_STOP);
            printf("+----------------------------------------------------------------+\n");
            printf("[Step %d | FTT=%.2f] Ub=%.6f  U*=%.4f  Force=%.5E  F*=%.4f  Re(now)=%.1f  Ma=%.4f  Ma_max=%.4f\n",
                   restart_step, FTT_init, Ub_init, Ustar, Force_h[0], Fstar, Re_now, Ma_init, Ma_max_init);

            if (Ma_max_init > 0.35)
                printf("  >>> [WARNING] Ma_max=%.4f > 0.35 — BGK stability risk, consider reducing Uref\n", Ma_max_init);

            if (Ustar > 1.3)
                printf("  >>> [NOTE] U*=%.4f >> 1.0 — VTK velocity from old Uref, flow will decelerate to new target\n", Ustar);
        }
        // Anti-windup Force cap (顯示原始狀態後才套用，避免前 NDTFRC 步用過高外力)
        // 週期性山丘流場因山丘阻力，所需外力遠高於 Poiseuille (典型 ~10-100×)
        {
            double h_eff = (double)LZ - (double)H_HILL;
            double Force_Poiseuille = 8.0 * (double)niu * (double)Uref / (h_eff * h_eff);
            double Force_cap = Force_Poiseuille * 100.0;  // 100× Poiseuille (hill drag >> flat channel)
            if (Force_h[0] > Force_cap) {
                if (myid == 0)
                    printf("[ANTI-WINDUP] Force capped: %.5E -> %.5E (max=100x Poiseuille=%.5E)\n",
                           Force_h[0], Force_cap, Force_Poiseuille);
                Force_h[0] = Force_cap;
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
    for( step = loop_start ; ; step++, accu_num++ ) {  // 無上限，以 FTT_STOP break 為終止
        double FTT_now = step * dt_global / (double)flow_through_time;

        // ===== Sub-step 1: even step (ft → fd) =====
        Launch_CollisionStreaming( ft, fd );

        // Statistics accumulation (FTT >= FTT_STATS_START): mean + RS + derivatives
        if (FTT_now >= FTT_STATS_START && step > 0) {
            if ((int)TBSWITCH) Launch_TurbulentSum( fd );
            CHECK_CUDA( cudaDeviceSynchronize() );
            Launch_AccumulateTavg();
            Launch_AccumulateVorticity();
            accu_count++;
        } else {
            CHECK_CUDA( cudaDeviceSynchronize() );
        }

        // Stage transition message
        if (!stage1_announced && FTT_now >= FTT_STATS_START) {
            stage1_announced = true;
            if (myid == 0) printf("\n>>> [FTT=%.2f] Statistics accumulation STARTED (accu_count=%d) <<<\n\n", FTT_now, accu_count);
        }

        // ===== Mid-step mass correction (between even and odd) =====
        // Ensures odd step uses up-to-date rho_modify computed from even step's density.
        // Without this, rho_modify is applied 2× per iteration but computed 1× → ρ̄ = 1+ε offset.
        // With this, each half-step gets its own correction → ρ̄ → 1.0 exactly at steady state.
        {
            const int rho_total_mid = (NX6 - 7) * (NYD6 - 7) * (NZ6 - 6);
            const int rho_threads_mid = 256;
            const int rho_blocks_mid = (rho_total_mid + rho_threads_mid - 1) / rho_threads_mid;

            ReduceRhoSum_Kernel<<<rho_blocks_mid, rho_threads_mid, rho_threads_mid * sizeof(double)>>>(rho_d, rho_partial_d);
            CHECK_CUDA( cudaMemcpy(rho_partial_h, rho_partial_d, rho_blocks_mid * sizeof(double), cudaMemcpyDeviceToHost) );

            double rho_LocalSum_mid = 0.0;
            for (int b = 0; b < rho_blocks_mid; b++) rho_LocalSum_mid += rho_partial_h[b];

            double rho_GlobalSum_mid = 0.0;
            MPI_Reduce(&rho_LocalSum_mid, &rho_GlobalSum_mid, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (myid == 0) {
                double rho_global_mid = 1.0 * (NX6 - 7) * (NY6 - 7) * (NZ6 - 6);
                rho_modify_h[0] = (rho_global_mid - rho_GlobalSum_mid) / ((NX6 - 7) * (NY6 - 7) * (NZ6 - 6));
            }
            MPI_Bcast(rho_modify_h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            cudaMemcpy(rho_modify_d, rho_modify_h, sizeof(double), cudaMemcpyHostToDevice);
        }

        // ===== Sub-step 2: odd step (fd → ft) =====
        step += 1;
        accu_num += 1;
        FTT_now = step * dt_global / (double)flow_through_time;

        Launch_CollisionStreaming( fd, ft );

        // Statistics accumulation (FTT >= FTT_STATS_START)
        if (FTT_now >= FTT_STATS_START) {
            if ((int)TBSWITCH) Launch_TurbulentSum( ft );
            CHECK_CUDA( cudaDeviceSynchronize() );
            Launch_AccumulateTavg();
            Launch_AccumulateVorticity();
            accu_count++;
        } else {
            CHECK_CUDA( cudaDeviceSynchronize() );
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
            printf("| Stats: %s  accu_count=%d\n",
                   (FTT_now >= FTT_STATS_START) ? "ON" : "OFF", accu_count);
            printf("+----------------------------------------------------------------+\n");

            cudaEventRecord(start1,0);
        }

        // ===== Force modification (every NDTFRC steps, Re%-based adaptive) =====
        // NOTE: step is always ODD here (after step+=1), so use == 1 like all other periodic checks
        if ( step > 0 && (step % NDTFRC == 1) ) { //根據文獻每一格FTT/10更新一次外力 
            Launch_ModifyForcingTerm();
        }

		if ( step%(int)NDTMIT == 1 ) {
			Launch_Monitor();
		}

        // ===== VTK output (every NDTVTK steps) + binary checkpoint (every NDTBIN steps) =====
        if ( step % NDTVTK == 1 ) {
            SendDataToCPU( ft );
            const size_t tavg_bytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
            CHECK_CUDA( cudaMemcpy(u_tavg_h, u_tavg_d, tavg_bytes, cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(v_tavg_h, v_tavg_d, tavg_bytes, cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(w_tavg_h, w_tavg_d, tavg_bytes, cudaMemcpyDeviceToHost) );

            // VTK-step status
            double Ma_max_vtk = ComputeMaMax();
            if (myid == 0) {
                // Bilinear cell-average: Σ v_cell × dx_cell × dz_cell / A_total
                double Ub_vtk = 0.0;
                for (int kk = 3; kk < NZ6-4; kk++)
                for (int ii = 3; ii < NX6-4; ii++) {
                    double v00 = v_h_p[3*NX6*NZ6 + kk*NX6 + ii];
                    double v10 = v_h_p[3*NX6*NZ6 + (kk+1)*NX6 + ii];
                    double v01 = v_h_p[3*NX6*NZ6 + kk*NX6 + (ii+1)];
                    double v11 = v_h_p[3*NX6*NZ6 + (kk+1)*NX6 + (ii+1)];
                    double v_cell = (v00 + v10 + v01 + v11) / 4.0;
                    Ub_vtk += v_cell * (x_h[ii+1] - x_h[ii]) * (z_h[3*NZ6+kk+1] - z_h[3*NZ6+kk]);
                }
                Ub_vtk /= (double)(LX * (LZ - 1.0));
                printf("[Step %d | FTT=%.2f] Ub=%.6f  U*=%.4f  Force=%.5E  F*=%.4f  Re=%.1f  Ma=%.4f  Ma_max=%.4f  accu=%d\n",
                       step, FTT_now,
                       Ub_vtk, Ub_vtk / (double)Uref, Force_h[0],
                       Force_h[0] * (double)LY / ((double)Uref * (double)Uref),
                       Ub_vtk / ((double)Uref / (double)Re),
                       Ub_vtk / (double)cs, Ma_max_vtk, accu_count);
            }

            fileIO_velocity_vtk_merged( step );

            // Binary checkpoint (every NDTBIN steps, piggyback on VTK's SendDataToCPU)
            if (step % NDTBIN == 1) {
                SaveBinaryCheckpoint( step );
            }
        }

        // ===== Global Mass Conservation Modify (GPU reduction — no SendDataToCPU) =====
        cudaDeviceSynchronize();
        cudaMemcpy(Force_h, Force_d, sizeof(double), cudaMemcpyDeviceToHost);
        {
            const int rho_total = (NX6 - 7) * (NYD6 - 7) * (NZ6 - 6);
            const int rho_threads = 256;
            const int rho_blocks = (rho_total + rho_threads - 1) / rho_threads;

            ReduceRhoSum_Kernel<<<rho_blocks, rho_threads, rho_threads * sizeof(double)>>>(rho_d, rho_partial_d);
            CHECK_CUDA( cudaMemcpy(rho_partial_h, rho_partial_d, rho_blocks * sizeof(double), cudaMemcpyDeviceToHost) );

            double rho_LocalSum = 0.0;
            for (int b = 0; b < rho_blocks; b++) rho_LocalSum += rho_partial_h[b];

            double rho_GlobalSum = 0.0;
            MPI_Reduce(&rho_LocalSum, &rho_GlobalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (myid == 0) {
                double rho_global = 1.0 * (NX6 - 7) * (NY6 - 7) * (NZ6 - 6);
                rho_modify_h[0] = (rho_global - rho_GlobalSum) / ((NX6 - 7) * (NY6 - 7) * (NZ6 - 6));
            }
            // Broadcast mass correction to ALL ranks (Bug #16 fix: was rank 0 only)
            MPI_Bcast(rho_modify_h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            cudaMemcpy(rho_modify_d, rho_modify_h, sizeof(double), cudaMemcpyHostToDevice);
        }

        // ===== Mass Conservation Check + NaN early stop (every 100 steps, GPU reduction) =====
        if (step % 100 == 1) {
            const int rho_total_chk = (NX6 - 7) * (NYD6 - 7) * (NZ6 - 6);
            const int rho_threads_chk = 256;
            const int rho_blocks_chk = (rho_total_chk + rho_threads_chk - 1) / rho_threads_chk;

            ReduceRhoSum_Kernel<<<rho_blocks_chk, rho_threads_chk, rho_threads_chk * sizeof(double)>>>(rho_d, rho_partial_d);
            CHECK_CUDA( cudaMemcpy(rho_partial_h, rho_partial_d, rho_blocks_chk * sizeof(double), cudaMemcpyDeviceToHost) );

            double rho_LocalSum_chk = 0.0;
            for (int b = 0; b < rho_blocks_chk; b++) rho_LocalSum_chk += rho_partial_h[b];
            double rho_LocalAvg = rho_LocalSum_chk / ((NX6 - 7) * (NYD6 - 7) * (NZ6 - 6));

            double rho_GlobalSum_chk = 0.0;
            MPI_Reduce(&rho_LocalAvg, &rho_GlobalSum_chk, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            int nan_flag = 0;
            if (myid == 0) {
                double rho_avg_check = rho_GlobalSum_chk / (double)jp;
                if (isnan(rho_avg_check) || isinf(rho_avg_check) || fabs(rho_avg_check - 1.0) > 0.01) {
                    printf("[FATAL] Divergence detected at step %d: rho_avg = %.6e, stopping.\n", step, rho_avg_check);
                    nan_flag = 1;
                }
            }
            MPI_Bcast(&nan_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (nan_flag) {
                SendDataToCPU(ft);
                fileIO_velocity_vtk_merged(step);
                break;
            }

            if (myid == 0) {
                double FTT_rho = step * dt_global / (double)flow_through_time;
                FILE *checkrho = fopen("checkrho.dat", "a");
                fprintf(checkrho, "%d\t %.4f\t %lf\t %lf\n", step, FTT_rho, 1.0, rho_GlobalSum_chk / (double)jp);
                fclose(checkrho);
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
        result_writebin_velocityandf();   // legacy: ./result/ 平面 checkpoint

        // Copy GPU tavg → host for final VTK + binary checkpoint
        const size_t tavg_bytes_final = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
        CHECK_CUDA( cudaMemcpy(u_tavg_h, u_tavg_d, tavg_bytes_final, cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaMemcpy(v_tavg_h, v_tavg_d, tavg_bytes_final, cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaMemcpy(w_tavg_h, w_tavg_d, tavg_bytes_final, cudaMemcpyDeviceToHost) );
        fileIO_velocity_vtk_merged( step );
        SaveBinaryCheckpoint( step );     // binary checkpoint (f^neq + tavg + RS + metadata)

        // Write merged statistics to ./statistics/ (backward compat for Python analysis scripts)
        if (accu_count > 0 && (int)TBSWITCH) {
            if (myid == 0) {
                printf("\n========================================================\n");
                printf("[FINAL OUTPUT] FTT = %.3f (timestep = %d)\n", FTT_final, step);
                printf("  -> Statistics accumulation: %d steps\n", accu_count);
                printf("  -> Writing merged statistics (33 arrays)\n");
                printf("========================================================\n\n");
            }
            statistics_writebin_merged_stress();
        } else if (myid == 0) {
            printf("\n[FINAL] FTT=%.2f, step=%d. No statistics to write (accu_count=%d).\n",
                   FTT_final, step, accu_count);
        }
    }

    free(u_tavg_h);
    free(v_tavg_h);
    free(w_tavg_h);
    free(ox_tavg_h);
    free(oy_tavg_h);
    free(oz_tavg_h);
    FreeSource();
    MPI_Finalize();

    return 0;
}
