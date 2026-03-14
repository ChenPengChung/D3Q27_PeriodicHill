// ============================================================================
// CUDA Unit Test: MRT-RM vs MRT-CM collision operator
// Verifies:
//   1. Both modes compile and run without errors
//   2. MRT-CM reduces to MRT-RM when u=0 (degeneracy)
//   3. Both conserve mass and momentum
//   4. feq is a fixed point for both
//   5. MRT-CM shift operator round-trip T^{-1}·T = I
//   6. Stability comparison at high velocity (Ma ~ 0.3)
//
// Compile (standalone, no MPI):
//   nvcc -O2 -arch=sm_35 test_mrt_cm_cuda.cu -o test_mrt_cm -DUSE_MRT=1 -DUSE_MRT_CM=1
//   nvcc -O2 -arch=sm_35 test_mrt_cm_cuda.cu -o test_mrt_rm -DUSE_MRT=1 -DUSE_MRT_CM=0
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// ── Minimal defines needed for the GILBM headers ──
// We don't need the full grid; just enough for the collision function
#ifndef USE_MRT
#define USE_MRT 1
#endif
#ifndef USE_MRT_CM
#define USE_MRT_CM 1
#endif

// Minimal grid defines (single-point test, no real grid needed)
#define NX6 39
#define NYD6 39
#define NZ6 134
#define NX 32
#define NY 128
#define NZ 128
#define jp 4
#define LX 4.5
#define LY 9.0
#define LZ 3.036
#define CFL 0.5
#define minSize ((LZ-1.0)/(NZ6-6)*CFL)
#define NT 32
#define Re 700
#define Uref 0.0583
#define niu (Uref/Re)
#define dt minSize
#define pi 3.14159265358979323846264338327950
#define cs (1.0/1.732050807568877)

// ── D3Q19 velocity set (host) ──
static const double h_e[19][3] = {
    {0,0,0},
    {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
    {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
    {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
    {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
};
static const double h_W[19] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// ── d'Humières M matrix (host) ──
static const double h_M[19][19] = {
    {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
    {-30,-11,-11,-11,-11,-11,-11,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8},
    { 12, -4, -4, -4, -4, -4, -4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
    {  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0},
    {  0, -4,  4,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0},
    {  0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1},
    {  0,  0,  0, -4,  4,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1},
    {  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1},
    {  0,  0,  0,  0,  0, -4,  4,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1},
    {  0,  2,  2, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2},
    {  0, -4, -4,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2},
    {  0,  0,  0,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0},
    {  0,  0,  0, -2, -2,  2,  2,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0},
    {  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0},
    {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1},
    {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0},
    {  0,  0,  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1, -1,  1,  0,  0,  0,  0},
    {  0,  0,  0,  0,  0,  0,  0, -1, -1,  1,  1,  0,  0,  0,  0,  1, -1,  1, -1},
    {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1, -1, -1,  1,  1}
};

// ── Device constants (matching evolution_gilbm.h) ──
__constant__ double GILBM_e[19][3];
__constant__ double GILBM_W[19];
__constant__ double GILBM_M[19][19];
__constant__ double GILBM_Mi[19][19];
__constant__ double GILBM_dt;

// ── Include the actual collision code ──
// We include the sub-modules directly
#include "gilbm/interpolation_gilbm.h"
#include "gilbm/boundary_conditions.h"
#if USE_MRT && USE_MRT_CM
#include "gilbm/MRT_CM_ShiftOperator.h"
#endif

// ── Copy collision functions from evolution_gilbm.h ──
// (Device functions only, no kernel wrappers)
__device__ void test_gilbm_mrt_collision(
    double f_re[19], const double feq_B[19],
    double s_visc, double dt_A, double Force0
) {
    double m_neq[19];
    for (int i = 0; i < 19; i++) {
        double sum = 0.0;
        for (int q = 0; q < 19; q++)
            sum += GILBM_M[i][q] * (f_re[q] - feq_B[q]);
        m_neq[i] = sum;
    }
    double dm[19];
    dm[0]  = 0.0;
    dm[1]  = 1.19  * m_neq[1];
    dm[2]  = 1.4   * m_neq[2];
    dm[3]  = 0.0;
    dm[4]  = 1.2   * m_neq[4];
    dm[5]  = 0.0;
    dm[6]  = 1.2   * m_neq[6];
    dm[7]  = 0.0;
    dm[8]  = 1.2   * m_neq[8];
    dm[9]  = s_visc * m_neq[9];
    dm[10] = 1.4   * m_neq[10];
    dm[11] = s_visc * m_neq[11];
    dm[12] = 1.4   * m_neq[12];
    dm[13] = s_visc * m_neq[13];
    dm[14] = s_visc * m_neq[14];
    dm[15] = s_visc * m_neq[15];
    dm[16] = 1.5   * m_neq[16];
    dm[17] = 1.5   * m_neq[17];
    dm[18] = 1.5   * m_neq[18];
    for (int q = 0; q < 19; q++) {
        double correction = 0.0;
        for (int i = 0; i < 19; i++)
            correction += GILBM_Mi[q][i] * dm[i];
        f_re[q] -= correction;
        f_re[q] += GILBM_W[q] * 3.0 * GILBM_e[q][1] * Force0 * dt_A;
    }
}

#if USE_MRT_CM
__device__ void test_gilbm_mrt_cm_collision(
    double f_re[19], const double feq_B[19],
    double s_visc, double dt_A, double Force0,
    double ux, double uy, double uz
) {
    double m_neq[19];
    for (int i = 0; i < 19; i++) {
        double sum = 0.0;
        for (int q = 0; q < 19; q++)
            sum += GILBM_M[i][q] * (f_re[q] - feq_B[q]);
        m_neq[i] = sum;
    }
    double k_neq[19];
    raw_to_central_dH(m_neq, ux, uy, uz, k_neq);
    double dk[19];
    dk[0]  = 0.0;
    dk[1]  = 1.19  * k_neq[1];
    dk[2]  = 1.4   * k_neq[2];
    dk[3]  = 0.0;
    dk[4]  = 1.2   * k_neq[4];
    dk[5]  = 0.0;
    dk[6]  = 1.2   * k_neq[6];
    dk[7]  = 0.0;
    dk[8]  = 1.2   * k_neq[8];
    dk[9]  = s_visc * k_neq[9];
    dk[10] = 1.4   * k_neq[10];
    dk[11] = s_visc * k_neq[11];
    dk[12] = 1.4   * k_neq[12];
    dk[13] = s_visc * k_neq[13];
    dk[14] = s_visc * k_neq[14];
    dk[15] = s_visc * k_neq[15];
    dk[16] = 1.5   * k_neq[16];
    dk[17] = 1.5   * k_neq[17];
    dk[18] = 1.5   * k_neq[18];
    double dm[19];
    central_to_raw_dH(dk, ux, uy, uz, dm);
    for (int q = 0; q < 19; q++) {
        double correction = 0.0;
        for (int i = 0; i < 19; i++)
            correction += GILBM_Mi[q][i] * dm[i];
        f_re[q] -= correction;
        f_re[q] += GILBM_W[q] * 3.0 * GILBM_e[q][1] * Force0 * dt_A;
    }
}
#endif

// ── Host feq computation ──
void host_compute_feq(double *feq, double rho, double ux, double uy, double uz) {
    for (int q = 0; q < 19; q++) {
        double eu = h_e[q][0]*ux + h_e[q][1]*uy + h_e[q][2]*uz;
        double u2 = ux*ux + uy*uy + uz*uz;
        feq[q] = h_W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
    }
}

// ============================================================================
// Test kernel: single-thread, runs all tests
// ============================================================================
__global__ void test_kernel(int *results, double *error_vals) {
    if (threadIdx.x != 0) return;
    int tid = 0;  // test index

    double f[19], feq[19];
    double rho, ux, uy, uz;
    double s_visc = 1.0 / 0.55;
    double dt_A = 0.001;

    // ========== Test 1: MRT-RM feq is fixed point ==========
    rho = 1.1; ux = 0.05; uy = -0.03; uz = 0.02;
    for (int q = 0; q < 19; q++) {
        double eu = GILBM_e[q][0]*ux + GILBM_e[q][1]*uy + GILBM_e[q][2]*uz;
        double u2 = ux*ux + uy*uy + uz*uz;
        feq[q] = GILBM_W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        f[q] = feq[q];
    }
    test_gilbm_mrt_collision(f, feq, s_visc, dt_A, 0.0);
    double err = 0.0;
    for (int q = 0; q < 19; q++) err = fmax(err, fabs(f[q] - feq[q]));
    results[tid] = (err < 1e-12) ? 1 : 0;
    error_vals[tid] = err;
    tid++;

    // ========== Test 2: MRT-RM conserves mass & momentum ==========
    rho = 1.0; ux = 0.0; uy = 0.0; uz = 0.0;
    for (int q = 0; q < 19; q++) {
        double eu = GILBM_e[q][0]*ux + GILBM_e[q][1]*uy + GILBM_e[q][2]*uz;
        double u2 = ux*ux + uy*uy + uz*uz;
        feq[q] = GILBM_W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        f[q] = feq[q] + 0.001 * ((q * 7 + 13) % 19 - 9) * 0.001;
    }
    // Normalize to preserve mass
    double sum0 = 0.0; for (int q = 0; q < 19; q++) sum0 += f[q];
    for (int q = 0; q < 19; q++) f[q] *= rho / sum0;
    double mass_pre = 0.0, momx_pre = 0.0;
    for (int q = 0; q < 19; q++) {
        mass_pre += f[q];
        momx_pre += f[q] * GILBM_e[q][0];
    }
    test_gilbm_mrt_collision(f, feq, s_visc, dt_A, 0.0);
    double mass_post = 0.0, momx_post = 0.0;
    for (int q = 0; q < 19; q++) {
        mass_post += f[q];
        momx_post += f[q] * GILBM_e[q][0];
    }
    err = fmax(fabs(mass_post - mass_pre), fabs(momx_post - momx_pre));
    results[tid] = (err < 1e-12) ? 1 : 0;
    error_vals[tid] = err;
    tid++;

#if USE_MRT_CM
    // ========== Test 3: MRT-CM feq is fixed point ==========
    rho = 1.1; ux = 0.05; uy = -0.03; uz = 0.02;
    for (int q = 0; q < 19; q++) {
        double eu = GILBM_e[q][0]*ux + GILBM_e[q][1]*uy + GILBM_e[q][2]*uz;
        double u2 = ux*ux + uy*uy + uz*uz;
        feq[q] = GILBM_W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        f[q] = feq[q];
    }
    test_gilbm_mrt_cm_collision(f, feq, s_visc, dt_A, 0.0, ux, uy, uz);
    err = 0.0;
    for (int q = 0; q < 19; q++) err = fmax(err, fabs(f[q] - feq[q]));
    results[tid] = (err < 1e-12) ? 1 : 0;
    error_vals[tid] = err;
    tid++;

    // ========== Test 4: MRT-CM conserves mass & momentum ==========
    rho = 1.0; ux = 0.0; uy = 0.0; uz = 0.0;
    for (int q = 0; q < 19; q++) {
        double eu = GILBM_e[q][0]*ux + GILBM_e[q][1]*uy + GILBM_e[q][2]*uz;
        double u2 = ux*ux + uy*uy + uz*uz;
        feq[q] = GILBM_W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        f[q] = feq[q] + 0.001 * ((q * 7 + 13) % 19 - 9) * 0.001;
    }
    sum0 = 0.0; for (int q = 0; q < 19; q++) sum0 += f[q];
    for (int q = 0; q < 19; q++) f[q] *= rho / sum0;
    mass_pre = 0.0; momx_pre = 0.0;
    for (int q = 0; q < 19; q++) {
        mass_pre += f[q];
        momx_pre += f[q] * GILBM_e[q][0];
    }
    test_gilbm_mrt_cm_collision(f, feq, s_visc, dt_A, 0.0, ux, uy, uz);
    mass_post = 0.0; momx_post = 0.0;
    for (int q = 0; q < 19; q++) {
        mass_post += f[q];
        momx_post += f[q] * GILBM_e[q][0];
    }
    err = fmax(fabs(mass_post - mass_pre), fabs(momx_post - momx_pre));
    results[tid] = (err < 1e-12) ? 1 : 0;
    error_vals[tid] = err;
    tid++;

    // ========== Test 5: u=0 degeneracy (CM = RM) ==========
    rho = 1.0; ux = 0.0; uy = 0.0; uz = 0.0;
    double f_rm[19], f_cm[19];
    for (int q = 0; q < 19; q++) {
        double eu = GILBM_e[q][0]*ux + GILBM_e[q][1]*uy + GILBM_e[q][2]*uz;
        double u2 = ux*ux + uy*uy + uz*uz;
        feq[q] = GILBM_W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        f_rm[q] = feq[q] + 0.005 * ((q * 11 + 3) % 19 - 9) * 0.001;
        f_cm[q] = f_rm[q];
    }
    test_gilbm_mrt_collision(f_rm, feq, s_visc, dt_A, 0.0);
    test_gilbm_mrt_cm_collision(f_cm, feq, s_visc, dt_A, 0.0, 0.0, 0.0, 0.0);
    err = 0.0;
    for (int q = 0; q < 19; q++) err = fmax(err, fabs(f_rm[q] - f_cm[q]));
    results[tid] = (err < 1e-12) ? 1 : 0;
    error_vals[tid] = err;
    tid++;

    // ========== Test 6: Shift operator round-trip T^{-1}·T = I ==========
    double m_test[19], k_test[19], m_back[19];
    for (int q = 0; q < 19; q++) m_test[q] = 0.1 * ((q * 13 + 7) % 19 - 9);
    ux = 0.1; uy = -0.08; uz = 0.05;
    raw_to_central_dH(m_test, ux, uy, uz, k_test);
    central_to_raw_dH(k_test, ux, uy, uz, m_back);
    err = 0.0;
    for (int q = 0; q < 19; q++) err = fmax(err, fabs(m_back[q] - m_test[q]));
    results[tid] = (err < 1e-12) ? 1 : 0;
    error_vals[tid] = err;
    tid++;

    // ========== Test 7: CM stability at Ma~0.3 (100 iterations) ==========
    rho = 1.0; ux = 0.1; uy = 0.1; uz = 0.0;  // Ma ≈ 0.3
    for (int q = 0; q < 19; q++) {
        double eu = GILBM_e[q][0]*ux + GILBM_e[q][1]*uy + GILBM_e[q][2]*uz;
        double u2 = ux*ux + uy*uy + uz*uz;
        feq[q] = GILBM_W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        f[q] = feq[q] + 0.01 * ((q * 7 + 5) % 19 - 9) * 0.001;
    }
    bool cm_stable = true;
    for (int iter = 0; iter < 100; iter++) {
        // Recompute feq from current f
        double rho_c = 0, mx = 0, my = 0, mz = 0;
        for (int q = 0; q < 19; q++) {
            rho_c += f[q]; mx += f[q]*GILBM_e[q][0];
            my += f[q]*GILBM_e[q][1]; mz += f[q]*GILBM_e[q][2];
        }
        double ucx = mx/rho_c, ucy = my/rho_c, ucz = mz/rho_c;
        for (int q = 0; q < 19; q++) {
            double eu = GILBM_e[q][0]*ucx + GILBM_e[q][1]*ucy + GILBM_e[q][2]*ucz;
            double u2 = ucx*ucx + ucy*ucy + ucz*ucz;
            feq[q] = GILBM_W[q] * rho_c * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        }
        test_gilbm_mrt_cm_collision(f, feq, s_visc, dt_A, 0.0, ucx, ucy, ucz);
        for (int q = 0; q < 19; q++) {
            if (isnan(f[q]) || isinf(f[q]) || fabs(f[q]) > 100.0) {
                cm_stable = false; break;
            }
        }
        if (!cm_stable) break;
    }
    results[tid] = cm_stable ? 1 : 0;
    error_vals[tid] = cm_stable ? 0.0 : -1.0;
    tid++;

    // ========== Test 8: RM stability at Ma~0.3 (100 iterations) ==========
    rho = 1.0; ux = 0.1; uy = 0.1; uz = 0.0;
    for (int q = 0; q < 19; q++) {
        double eu = GILBM_e[q][0]*ux + GILBM_e[q][1]*uy + GILBM_e[q][2]*uz;
        double u2 = ux*ux + uy*uy + uz*uz;
        feq[q] = GILBM_W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        f[q] = feq[q] + 0.01 * ((q * 7 + 5) % 19 - 9) * 0.001;
    }
    bool rm_stable = true;
    for (int iter = 0; iter < 100; iter++) {
        double rho_c = 0, mx = 0, my = 0, mz = 0;
        for (int q = 0; q < 19; q++) {
            rho_c += f[q]; mx += f[q]*GILBM_e[q][0];
            my += f[q]*GILBM_e[q][1]; mz += f[q]*GILBM_e[q][2];
        }
        double ucx = mx/rho_c, ucy = my/rho_c, ucz = mz/rho_c;
        for (int q = 0; q < 19; q++) {
            double eu = GILBM_e[q][0]*ucx + GILBM_e[q][1]*ucy + GILBM_e[q][2]*ucz;
            double u2 = ucx*ucx + ucy*ucy + ucz*ucz;
            feq[q] = GILBM_W[q] * rho_c * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        }
        test_gilbm_mrt_collision(f, feq, s_visc, dt_A, 0.0);
        for (int q = 0; q < 19; q++) {
            if (isnan(f[q]) || isinf(f[q]) || fabs(f[q]) > 100.0) {
                rm_stable = false; break;
            }
        }
        if (!rm_stable) break;
    }
    results[tid] = rm_stable ? 1 : 0;
    error_vals[tid] = rm_stable ? 0.0 : -1.0;
    tid++;

#else
    // Non-CM mode: fill remaining slots
    for (int t = tid; t < 8; t++) {
        results[t] = -1;  // skipped
        error_vals[t] = 0.0;
    }
#endif
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("======================================\n");
#if USE_MRT_CM
    printf("  CUDA Unit Test: MRT-CM mode\n");
#else
    printf("  CUDA Unit Test: MRT-RM mode\n");
#endif
    printf("======================================\n\n");

    // Compute M inverse on host
    // Simple Gauss-Jordan for 19×19
    double h_Mi[19][19];
    double aug[19][38];
    for (int i = 0; i < 19; i++) {
        for (int j = 0; j < 19; j++) {
            aug[i][j] = h_M[i][j];
            aug[i][j+19] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < 19; col++) {
        // Find pivot
        int pivot = col;
        for (int row = col+1; row < 19; row++)
            if (fabs(aug[row][col]) > fabs(aug[pivot][col])) pivot = row;
        if (pivot != col)
            for (int j = 0; j < 38; j++) { double t = aug[col][j]; aug[col][j] = aug[pivot][j]; aug[pivot][j] = t; }
        double diag = aug[col][col];
        for (int j = 0; j < 38; j++) aug[col][j] /= diag;
        for (int row = 0; row < 19; row++) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (int j = 0; j < 38; j++) aug[row][j] -= factor * aug[col][j];
        }
    }
    for (int i = 0; i < 19; i++)
        for (int j = 0; j < 19; j++)
            h_Mi[i][j] = aug[i][j+19];

    // Copy to device constants
    double h_dt = 0.001;
    cudaMemcpyToSymbol(GILBM_e, h_e, sizeof(h_e));
    cudaMemcpyToSymbol(GILBM_W, h_W, sizeof(h_W));
    cudaMemcpyToSymbol(GILBM_M, h_M, sizeof(h_M));
    cudaMemcpyToSymbol(GILBM_Mi, h_Mi, sizeof(h_Mi));
    cudaMemcpyToSymbol(GILBM_dt, &h_dt, sizeof(double));

    // Allocate device results
    int *d_results;
    double *d_errors;
    cudaMalloc(&d_results, 8 * sizeof(int));
    cudaMalloc(&d_errors, 8 * sizeof(double));
    cudaMemset(d_results, 0, 8 * sizeof(int));

    // Launch
    test_kernel<<<1, 1>>>(d_results, d_errors);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy results back
    int h_results[8];
    double h_errors[8];
    cudaMemcpy(h_results, d_results, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_errors, d_errors, 8 * sizeof(double), cudaMemcpyDeviceToHost);

    const char *test_names[] = {
#if USE_MRT_CM
        "RM feq fixed point",
        "RM conservation (mass+momentum)",
        "CM feq fixed point",
        "CM conservation (mass+momentum)",
        "u=0 degeneracy (CM=RM)",
        "Shift round-trip T^-1·T = I",
        "CM stability (100 iter, Ma~0.3)",
        "RM stability (100 iter, Ma~0.3)"
#else
        "RM feq fixed point",
        "RM conservation (mass+momentum)",
        "CM feq fixed point (SKIPPED)",
        "CM conservation (SKIPPED)",
        "u=0 degeneracy (SKIPPED)",
        "Shift round-trip (SKIPPED)",
        "CM stability (SKIPPED)",
        "RM stability (SKIPPED)"
#endif
    };

    int total_pass = 0, total_run = 0;
    for (int t = 0; t < 8; t++) {
        if (h_results[t] == -1) {
            printf("  [SKIP] Test %d: %s\n", t+1, test_names[t]);
            continue;
        }
        total_run++;
        bool pass = (h_results[t] == 1);
        if (pass) total_pass++;
        printf("  [%s] Test %d: %s (err=%.2e)\n",
               pass ? "PASS" : "FAIL", t+1, test_names[t], h_errors[t]);
    }

    printf("\n  Result: %d/%d passed\n", total_pass, total_run);
    printf("======================================\n");

    cudaFree(d_results);
    cudaFree(d_errors);

    return (total_pass == total_run) ? 0 : 1;
}
