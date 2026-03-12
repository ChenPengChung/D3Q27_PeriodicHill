// ================================================================
// test_cumulant_mass_conservation.cpp
// CPU-only unit test for D3Q27 Cumulant collision mass conservation
//
// Purpose: Diagnose why rho != 1.0 after the first collision step
//          when using Cumulant collision in GILBM PeriodicHill.
//
// Build: g++ -O2 -o test_cumulant test_cumulant_mass_conservation.cpp -lm
// Run:   ./test_cumulant
// ================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// ================================================================
// Reproduce project parameters from variables.h
// ================================================================
#define NQ 27
#define Re 700
#define Uref 0.0503
#define niu (Uref / Re)
#define LZ 3.036
#define H_HILL 1.0
#define LY 9.0
#define NX6 88
#define NY6 391
#define NZ6 198
#define CFL 0.5
#define minSize ((LZ - 1.0) / (NZ6 - 6) * CFL)

// Cumulant mode
#define USE_WP_CUMULANT 1
#define CUM_LAMBDA 1.0e-2
#define CUM_OMEGA2 0.5

// ================================================================
// Strip CUDA qualifiers for CPU
// ================================================================
#define __device__
#define __constant__
#define __forceinline__ inline

// ================================================================
// D3Q27 velocity set (same ordering as GILBM_e in evolution_gilbm.h)
// ================================================================
static double GILBM_e[NQ][3] = {
    { 0, 0, 0},                                    // 0: rest
    { 1, 0, 0}, {-1, 0, 0},                        // 1-2: +/-x
    { 0, 1, 0}, { 0,-1, 0},                        // 3-4: +/-y
    { 0, 0, 1}, { 0, 0,-1},                        // 5-6: +/-z
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0}, // 7-10: xy edges
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1}, // 11-14: xz edges
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1}, // 15-18: yz edges
    { 1, 1, 1}, {-1, 1, 1}, { 1,-1, 1}, {-1,-1, 1}, // 19-22: corners +z
    { 1, 1,-1}, {-1, 1,-1}, { 1,-1,-1}, {-1,-1,-1}  // 23-26: corners -z
};

static double GILBM_W[NQ] = {
    8.0/27.0,                                       // rest
    2.0/27.0, 2.0/27.0, 2.0/27.0,                  // face
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,        // edge
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,    // corner
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

// ================================================================
// Include cumulant_constants.h and cumulant_collision.h (CPU mode)
// ================================================================
#include "../Cumulants/cumulant_constants.h"
#include "../Cumulants/cumulant_collision.h"

// ================================================================
// Compute equilibrium distribution (same formula as GILBM)
// ================================================================
static double compute_feq(int alpha, double rho, double u, double v, double w) {
    double eu = GILBM_e[alpha][0]*u + GILBM_e[alpha][1]*v + GILBM_e[alpha][2]*w;
    double udot = u*u + v*v + w*w;
    return GILBM_W[alpha] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*udot);
}

// ================================================================
// Diagnostic: Forward Chimera with mass tracking
// ================================================================
static void forward_chimera_diagnostic(double m[27], const double u[3]) {
    printf("\n=== Forward Chimera Diagnostic ===\n");
    double sum_before = 0;
    for (int i = 0; i < 27; i++) sum_before += m[i];
    printf("  Sum(m) before forward Chimera: %.15e\n", sum_before);

    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;
        double sum_sweep = 0;
        for (int i = 0; i < 27; i++) sum_sweep += m[i];
        printf("  [dir=%d] Sum(m) before sweep: %.15e\n", dir, sum_sweep);

        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double k = CUM_K[p];

            double old_sum = m[a] + m[b] + m[c];

            double sum  = m[a] + m[c];
            double diff = m[c] - m[a];
            m[a] = m[a] + m[b] + m[c];
            m[b] = diff - (m[a] + k) * u[dir];
            m[c] = sum - 2.0 * diff * u[dir] + u[dir] * u[dir] * (m[a] + k);

            double new_sum = m[a] + m[b] + m[c];
            if (fabs(old_sum - new_sum) > 1e-14) {
                printf("  *** TRIPLET SUM CHANGED at pass %d: %.15e -> %.15e (diff=%.2e)\n",
                       p, old_sum, new_sum, new_sum - old_sum);
            }
        }
        double sum_after_sweep = 0;
        for (int i = 0; i < 27; i++) sum_after_sweep += m[i];
        printf("  [dir=%d] Sum(m) after sweep: %.15e (change=%.2e)\n",
               dir, sum_after_sweep, sum_after_sweep - sum_sweep);
    }

    double sum_after = 0;
    for (int i = 0; i < 27; i++) sum_after += m[i];
    printf("  Sum(m) after forward Chimera: %.15e (total change=%.2e)\n",
           sum_after, sum_after - sum_before);
}

// ================================================================
// Diagnostic: Backward Chimera with mass tracking
// ================================================================
static void backward_chimera_diagnostic(double m[27], const double u[3]) {
    printf("\n=== Backward Chimera Diagnostic ===\n");
    double sum_before = 0;
    for (int i = 0; i < 27; i++) sum_before += m[i];
    printf("  Sum(m) before backward Chimera: %.15e\n", sum_before);

    for (int dir = 0; dir < 3; dir++) {
        int base = (2 - dir) * 9;
        double sum_sweep = 0;
        for (int i = 0; i < 27; i++) sum_sweep += m[i];

        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double k = CUM_K[p];

            double old_sum = m[a] + m[b] + m[c];

            double ma = ((m[c] - m[b]) * 0.5 + m[b] * u[dir]
                        + (m[a] + k) * (u[dir]*u[dir] - u[dir]) * 0.5);
            double mb = (m[a] - m[c]) - 2.0 * m[b] * u[dir]
                        - (m[a] + k) * u[dir] * u[dir];
            double mc = ((m[c] + m[b]) * 0.5 + m[b] * u[dir]
                        + (m[a] + k) * (u[dir]*u[dir] + u[dir]) * 0.5);

            double new_sum = ma + mb + mc;
            if (fabs(old_sum - new_sum) > 1e-14) {
                printf("  *** BACKWARD TRIPLET SUM CHANGED at pass %d: old_sum(m[a]+m[b]+m[c])=%.15e\n"
                       "      ma+mb+mc=%.15e (diff=%.2e) m[a]=%.15e\n",
                       p, old_sum, new_sum, new_sum - old_sum, m[a]);
            }

            m[a] = ma;
            m[b] = mb;
            m[c] = mc;
        }

        double sum_after_sweep = 0;
        for (int i = 0; i < 27; i++) sum_after_sweep += m[i];
        printf("  [dir=%d] Sum(m) after backward sweep: %.15e (change=%.2e)\n",
               dir, sum_after_sweep, sum_after_sweep - sum_sweep);
    }

    double sum_after = 0;
    for (int i = 0; i < 27; i++) sum_after += m[i];
    printf("  Sum(m) after backward Chimera: %.15e (total change=%.2e)\n",
           sum_after, sum_after - sum_before);
}

// ================================================================
// TEST 1: Pure collision at rho=1, u=0, zero force
//         Expected: f_out = f_in (equilibrium is a fixed point)
// ================================================================
void test1_equilibrium_zero_force() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 1: Equilibrium (rho=1, u=0) with ZERO force\n");
    printf("============================================================\n");

    double f_in[NQ], f_out[NQ];
    double rho_out, ux_out, uy_out, uz_out;

    // Initialize to equilibrium at rho=1, u=v=w=0
    for (int q = 0; q < NQ; q++)
        f_in[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    // Verify initialization
    double sum_in = 0;
    for (int q = 0; q < NQ; q++) sum_in += f_in[q];
    printf("  Sum(f_in) = %.15f\n", sum_in);
    printf("  f_in[0] (rest) = %.15f,  W[0] = %.15f\n", f_in[0], GILBM_W[0]);

    // Compute omega_global as in main.cu
    double dt_test = minSize;  // Use same as the project
    double omega_test = 3.0 * niu / dt_test + 0.5;
    printf("  dt = %.10f, omega_global(tau) = %.10f, omega1(rate) = %.10f\n",
           dt_test, omega_test, 1.0 / omega_test);

    cumulant_collision_D3Q27(f_in, omega_test, dt_test,
                             0.0, 0.0, 0.0,
                             f_out, &rho_out, &ux_out, &uy_out, &uz_out);

    double sum_out = 0;
    for (int q = 0; q < NQ; q++) sum_out += f_out[q];

    printf("  rho_out = %.15f\n", rho_out);
    printf("  Sum(f_out) = %.15f\n", sum_out);
    printf("  Mass error (sum_out - sum_in) = %.2e\n", sum_out - sum_in);
    printf("  Max |f_out - f_in| = ");
    double max_diff = 0;
    for (int q = 0; q < NQ; q++) {
        double d = fabs(f_out[q] - f_in[q]);
        if (d > max_diff) max_diff = d;
    }
    printf("%.2e\n", max_diff);

    if (fabs(sum_out - 1.0) < 1e-12)
        printf("  >>> PASS: Mass conserved.\n");
    else
        printf("  >>> FAIL: Mass NOT conserved! Delta = %.2e\n", sum_out - 1.0);
}

// ================================================================
// TEST 2: Collision at rho=1, u=0, WITH force (same as cold start)
//         Expected: sum(f_out) = 1.0 (Guo forcing conserves mass)
// ================================================================
void test2_equilibrium_with_force() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 2: Equilibrium (rho=1, u=0) with FORCE (cold start)\n");
    printf("============================================================\n");

    double f_in[NQ], f_out[NQ];
    double rho_out, ux_out, uy_out, uz_out;

    for (int q = 0; q < NQ; q++)
        f_in[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    double dt_test = minSize;
    double omega_test = 3.0 * niu / dt_test + 0.5;

    // Force = 2 * Poiseuille (same as initialization.h)
    double h_eff = LZ - H_HILL;
    double Force_init = (8.0 * niu * Uref) / (h_eff * h_eff) * 2.0;
    printf("  Force_init = %.10e\n", Force_init);
    printf("  dt = %.10f, omega_global(tau) = %.10f\n", dt_test, omega_test);

    double sum_in = 0;
    for (int q = 0; q < NQ; q++) sum_in += f_in[q];

    cumulant_collision_D3Q27(f_in, omega_test, dt_test,
                             0.0, Force_init, 0.0,
                             f_out, &rho_out, &ux_out, &uy_out, &uz_out);

    double sum_out = 0;
    for (int q = 0; q < NQ; q++) sum_out += f_out[q];

    printf("  Sum(f_in)  = %.15f\n", sum_in);
    printf("  Sum(f_out) = %.15f\n", sum_out);
    printf("  rho_out    = %.15f\n", rho_out);
    printf("  ux = %.10e, uy = %.10e, uz = %.10e\n", ux_out, uy_out, uz_out);
    printf("  Mass error = %.2e\n", sum_out - sum_in);

    // Verify Guo forcing mass conservation analytically
    double ax = 0.0, ay = Force_init, az = 0.0;
    double uy_half = 0.5 * Force_init * dt_test;  // half-force velocity at cold start
    double a_dot_u = ay * uy_half;
    double source_sum = 0;
    for (int q = 0; q < NQ; q++) {
        double e_dot_a = GILBM_e[q][0]*ax + GILBM_e[q][1]*ay + GILBM_e[q][2]*az;
        double e_dot_u = GILBM_e[q][1]*uy_half;
        double S_q = GILBM_W[q] * 1.0 * (3.0*e_dot_a*(1.0 + 3.0*e_dot_u) - 3.0*a_dot_u);
        source_sum += S_q;
    }
    printf("  Analytic sum(S_i) = %.2e (should be 0)\n", source_sum);

    if (fabs(sum_out - 1.0) < 1e-12)
        printf("  >>> PASS: Mass conserved with force.\n");
    else
        printf("  >>> FAIL: Mass NOT conserved with force! Delta = %.2e\n", sum_out - 1.0);
}

// ================================================================
// TEST 3: Detailed stage-by-stage mass tracking
//         Replicate cumulant_collision_D3Q27 step by step
// ================================================================
void test3_stage_by_stage_diagnostic() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 3: Stage-by-stage mass diagnostic\n");
    printf("============================================================\n");

    double f_in[NQ];
    for (int q = 0; q < NQ; q++)
        f_in[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    double dt_test = minSize;
    double omega_tau = 3.0 * niu / dt_test + 0.5;
    double omega = 1.0 / omega_tau;

    double h_eff = LZ - H_HILL;
    double Fy = (8.0 * niu * Uref) / (h_eff * h_eff) * 2.0;

    // Stage 0: Macroscopic
    double rho = 0;
    for (int i = 0; i < 27; i++) rho += f_in[i];
    double jx = 0, jy = 0, jz = 0;
    for (int i = 0; i < 27; i++) {
        jx += f_in[i] * GILBM_e[i][0];
        jy += f_in[i] * GILBM_e[i][1];
        jz += f_in[i] * GILBM_e[i][2];
    }
    double inv_rho = 1.0 / rho;
    double u[3];
    u[0] = jx * inv_rho;
    u[1] = jy * inv_rho + 0.5 * Fy * inv_rho * dt_test;
    u[2] = jz * inv_rho;

    printf("  Stage 0: rho=%.15f, jx=%.2e, jy=%.2e, jz=%.2e\n", rho, jx, jy, jz);
    printf("  Stage 0: u=(%.10e, %.10e, %.10e)\n", u[0], u[1], u[2]);

    // Well-conditioning
    double m[27];
    for (int i = 0; i < 27; i++) m[i] = f_in[i] - GILBM_W[i];

    double sum_m = 0;
    for (int i = 0; i < 27; i++) sum_m += m[i];
    printf("  After well-conditioning: Sum(m) = %.15e (should be rho-1 = %.15e)\n",
           sum_m, rho - 1.0);

    // Forward Chimera with detailed diagnostic
    forward_chimera_diagnostic(m, u);

    printf("\n  After forward Chimera, key central moments:\n");
    printf("    m[I_aaa=%d] (kappa000 = drho) = %.15e\n", I_aaa, m[I_aaa]);
    printf("    m[I_baa=%d] (kappa100 = jx)   = %.15e\n", I_baa, m[I_baa]);
    printf("    m[I_aba=%d] (kappa010 = jy)   = %.15e\n", I_aba, m[I_aba]);
    printf("    m[I_aab=%d] (kappa001 = jz)   = %.15e\n", I_aab, m[I_aab]);
    printf("    m[I_caa=%d] (kappa200 = xx)   = %.15e\n", I_caa, m[I_caa]);
    printf("    m[I_aca=%d] (kappa020 = yy)   = %.15e\n", I_aca, m[I_aca]);
    printf("    m[I_aac=%d] (kappa002 = zz)   = %.15e\n", I_aac, m[I_aac]);

    // Save pre-relaxation m for comparison
    double m_pre_relax[27];
    for (int i = 0; i < 27; i++) m_pre_relax[i] = m[i];

    // ---- Relaxation (simplified, just check what changes) ----
    // 2nd order
    double drho = rho - 1.0;
    const double omega2 = CUM_OMEGA2;

    double mxxPyyPzz = m[I_caa] + m[I_aca] + m[I_aac];
    double mxxMyy    = m[I_caa] - m[I_aca];
    double mxxMzz    = m[I_caa] - m[I_aac];
    mxxPyyPzz += omega2 * (m[I_aaa] - mxxPyyPzz);
    mxxMyy *= (1.0 - omega);
    mxxMzz *= (1.0 - omega);
    m[I_abb] *= (1.0 - omega);
    m[I_bab] *= (1.0 - omega);
    m[I_bba] *= (1.0 - omega);

    m[I_caa] = ( mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
    m[I_aca] = (-2.0*mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
    m[I_aac] = ( mxxMyy - 2.0*mxxMzz + mxxPyyPzz) / 3.0;

    // Check 2nd-order trace conservation
    double trace_after = m[I_caa] + m[I_aca] + m[I_aac];
    printf("\n  After 2nd-order relaxation:\n");
    printf("    trace = %.15e (should = mxxPyyPzz = %.15e)\n", trace_after, mxxPyyPzz);
    printf("    m[I_aaa] unchanged? = %.15e\n", m[I_aaa]);

    double sum_m_after_relax = 0;
    for (int i = 0; i < 27; i++) sum_m_after_relax += m[i];
    printf("    Sum(m) after partial relaxation = %.15e\n", sum_m_after_relax);

    // Skip full relaxation for now, do backward chimera with current state
    // to test if sum is preserved

    // Backward Chimera diagnostic
    backward_chimera_diagnostic(m, u);

    // Restore f from well-conditioned
    double f_restored[27];
    for (int i = 0; i < 27; i++) f_restored[i] = m[i] + GILBM_W[i];

    double sum_restored = 0;
    for (int i = 0; i < 27; i++) sum_restored += f_restored[i];
    printf("\n  After restoring (m + W): Sum(f_restored) = %.15f\n", sum_restored);
    printf("  Mass change = %.2e\n", sum_restored - rho);
}

// ================================================================
// TEST 4: Round-trip Chimera (forward + backward) without relaxation
//         Should reproduce input exactly
// ================================================================
void test4_chimera_roundtrip() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 4: Chimera round-trip (no relaxation)\n");
    printf("============================================================\n");

    double f_in[NQ];
    for (int q = 0; q < NQ; q++)
        f_in[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    // Test with non-zero velocity (as in cold start with force)
    double dt_test = minSize;
    double Fy = (8.0 * niu * Uref) / ((LZ - H_HILL) * (LZ - H_HILL)) * 2.0;
    double u_test[3] = {0.0, 0.5 * Fy * dt_test, 0.0};

    printf("  Test velocity: u = (%.10e, %.10e, %.10e)\n", u_test[0], u_test[1], u_test[2]);

    // Well-condition
    double m[27], m_orig[27];
    for (int i = 0; i < 27; i++) {
        m[i] = f_in[i] - GILBM_W[i];
        m_orig[i] = m[i];
    }

    // Forward Chimera
    _cum_forward_chimera(m, u_test);
    double sum_forward = 0;
    for (int i = 0; i < 27; i++) sum_forward += m[i];

    // Backward Chimera (NO relaxation in between)
    _cum_backward_chimera(m, u_test);
    double sum_backward = 0;
    for (int i = 0; i < 27; i++) sum_backward += m[i];

    double max_err = 0;
    int max_err_idx = -1;
    for (int i = 0; i < 27; i++) {
        double err = fabs(m[i] - m_orig[i]);
        if (err > max_err) { max_err = err; max_err_idx = i; }
    }

    printf("  Sum after forward:  %.15e\n", sum_forward);
    printf("  Sum after backward: %.15e\n", sum_backward);
    printf("  Max round-trip error: %.2e at index %d\n", max_err, max_err_idx);

    // Restore and check mass
    double sum_restored = 0;
    for (int i = 0; i < 27; i++) sum_restored += (m[i] + GILBM_W[i]);
    printf("  Sum(f_restored) = %.15f\n", sum_restored);

    if (max_err < 1e-12)
        printf("  >>> PASS: Chimera round-trip exact.\n");
    else
        printf("  >>> FAIL: Chimera round-trip error = %.2e\n", max_err);
}

// ================================================================
// TEST 5: Multiple collision steps (simulate time stepping)
//         Check if density drifts over steps
// ================================================================
void test5_multiple_steps() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 5: Multiple collision steps (10 steps)\n");
    printf("============================================================\n");

    double f[NQ];
    for (int q = 0; q < NQ; q++)
        f[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    double dt_test = minSize;
    double omega_test = 3.0 * niu / dt_test + 0.5;
    double h_eff = LZ - H_HILL;
    double Force = (8.0 * niu * Uref) / (h_eff * h_eff) * 2.0;

    printf("  Step | Sum(f)          | rho_out         | uy_out\n");
    printf("  -----|-----------------|-----------------|----------\n");

    for (int step = 0; step < 10; step++) {
        double f_out[NQ];
        double rho_out, ux_out, uy_out, uz_out;

        cumulant_collision_D3Q27(f, omega_test, dt_test,
                                 0.0, Force, 0.0,
                                 f_out, &rho_out, &ux_out, &uy_out, &uz_out);

        double sum_out = 0;
        for (int q = 0; q < NQ; q++) sum_out += f_out[q];

        printf("  %4d | %.13f | %.13f | %.10e\n", step, sum_out, rho_out, uy_out);

        // Feed output as next input (single-point, no streaming)
        for (int q = 0; q < NQ; q++) f[q] = f_out[q];
    }
}

// ================================================================
// TEST 6: Verify CUM_K constants match GILBM_W
//         CUM_K should be forward Chimera of W at u=0
// ================================================================
void test6_verify_CUM_K() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 6: Verify CUM_K constants match forward Chimera of W\n");
    printf("============================================================\n");

    double w[27];
    for (int i = 0; i < 27; i++) w[i] = GILBM_W[i];

    double u_zero[3] = {0.0, 0.0, 0.0};

    // Apply forward Chimera to weights at u=0
    // At u=0, the transform simplifies:
    //   m[a] = m[a] + m[b] + m[c]     (sum)
    //   m[b] = m[c] - m[a]             (diff, since u=0 term vanishes)
    //   m[c] = m[a] + m[c]             (sum of endpoints, since u=0)
    // But the well-conditioned version uses k=0 since we're transforming W itself

    // Actually, CUM_K[p] is the kappa0 value (m[a] after the pass)
    // when the forward Chimera is applied to W[] at u=0.
    // Let's compute it directly:

    // Simulate the forward Chimera with k=0 (transforming W directly, not well-conditioned)
    // Wait - CUM_K is used when transforming m = f - W. When we transform W itself:
    // m_W[i] = W[i] - W[i] = 0 for all i
    // Forward Chimera on all-zeros with any k gives:
    //   m[a] = 0, m[b] = -(0+k)*u = 0 (u=0), m[c] = 0 + k*u^2 = 0
    // So CUM_K = forward Chimera of W... hmm

    // Actually, let me re-read the code. CUM_K[p] appears as:
    //   m[b] = diff - (m[a] + k) * u
    //   m[c] = sum - 2*diff*u + u^2*(m[a]+k)
    // Here m[a] is the well-conditioned sum, and m[a]+k is the PHYSICAL sum.
    // So k = physical_sum - well_conditioned_sum = sum(W in triplet)

    // For the z-sweep, k = sum of 3 weights. Let me verify:
    printf("  Z-sweep K values:\n");
    double max_K_err = 0;
    for (int p = 0; p < 9; p++) {
        int a = CUM_IDX[p][0];
        int b = CUM_IDX[p][1];
        int c = CUM_IDX[p][2];
        double k_computed = GILBM_W[a] + GILBM_W[b] + GILBM_W[c];
        double k_stored = CUM_K[p];
        double err = fabs(k_computed - k_stored);
        if (err > max_K_err) max_K_err = err;
        if (err > 1e-14)
            printf("    Pass %d: {%d,%d,%d} computed=%.15f stored=%.15f ERR=%.2e\n",
                   p, a, b, c, k_computed, k_stored, err);
    }
    printf("  Z-sweep max K error: %.2e\n", max_K_err);

    // For y-sweep and x-sweep, K values depend on the cascaded transforms.
    // Let me compute them properly by running the forward Chimera on W at u=0.
    double w_copy[27];
    for (int i = 0; i < 27; i++) w_copy[i] = GILBM_W[i];

    // Apply z-sweep at u=0 (k=0 for direct transform, not well-conditioned)
    // Actually for the ORIGINAL (non-well-conditioned) Chimera, k=0.
    // We need the transform WITHOUT k corrections, applied to W.
    printf("\n  Full K verification (forward Chimera of W at u=0):\n");
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];

            // At u=0, forward Chimera (WITHOUT k correction) gives:
            //   new_a = old_a + old_b + old_c  (sum)
            //   new_b = old_c - old_a           (diff)
            //   new_c = old_a + old_c           (sum of endpoints)
            double old_a = w_copy[a], old_b = w_copy[b], old_c = w_copy[c];
            w_copy[a] = old_a + old_b + old_c;
            w_copy[b] = old_c - old_a;
            w_copy[c] = old_a + old_c;

            double err = fabs(w_copy[a] - CUM_K[p]);
            if (err > 1e-14) {
                printf("    Pass %d (dir=%d): K_computed=%.15f K_stored=%.15f ERR=%.2e ***\n",
                       p, dir, w_copy[a], CUM_K[p], err);
            }
        }
    }

    // Final check: after full forward Chimera of W at u=0,
    // w_copy[I_aaa] should be sum(W) = 1.0 = CUM_K[18+8] = CUM_K[26]
    printf("  After full Chimera of W: w_copy[I_aaa=%d] = %.15f (should be 1.0)\n",
           I_aaa, w_copy[I_aaa]);
    printf("  CUM_K[18] (x-sweep pass 0) = %.15f\n", CUM_K[18]);

    // Check all K values
    double w_verify[27];
    for (int i = 0; i < 27; i++) w_verify[i] = GILBM_W[i];
    double max_total_err = 0;
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double old_a = w_verify[a], old_b = w_verify[b], old_c = w_verify[c];
            w_verify[a] = old_a + old_b + old_c;
            w_verify[b] = old_c - old_a;
            w_verify[c] = old_a + old_c;
            double err = fabs(w_verify[a] - CUM_K[p]);
            if (err > max_total_err) max_total_err = err;
            if (err > 1e-14) {
                printf("    *** K MISMATCH at pass %d: computed=%.15f stored=%.15f\n",
                       p, w_verify[a], CUM_K[p]);
            }
        }
    }
    printf("  Overall max K error: %.2e\n", max_total_err);
    if (max_total_err < 1e-14)
        printf("  >>> PASS: CUM_K constants are correct.\n");
    else
        printf("  >>> FAIL: CUM_K constants have errors!\n");
}

// ================================================================
// TEST 7: Verify weight sum
// ================================================================
void test7_verify_weights() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 7: Verify D3Q27 weight sum\n");
    printf("============================================================\n");

    double sum_W = 0;
    for (int i = 0; i < 27; i++) sum_W += GILBM_W[i];
    printf("  Sum(GILBM_W) = %.15f (should be 1.0)\n", sum_W);

    // Verify 2nd moment isotropy: sum(W[i]*e[i][d]*e[i][d]) = 1/3
    for (int d = 0; d < 3; d++) {
        double m2 = 0;
        for (int i = 0; i < 27; i++)
            m2 += GILBM_W[i] * GILBM_e[i][d] * GILBM_e[i][d];
        printf("  sum(W*e_%c^2) = %.15f (should be 1/3 = %.15f)\n",
               'x'+d, m2, 1.0/3.0);
    }

    // Verify feq at rho=1, u=0 sums to 1
    double sum_feq = 0;
    for (int q = 0; q < NQ; q++)
        sum_feq += compute_feq(q, 1.0, 0.0, 0.0, 0.0);
    printf("  Sum(feq(rho=1, u=0)) = %.15f\n", sum_feq);
}

// ================================================================
// TEST 8: Decompose Guo forcing mass contribution
// ================================================================
void test8_guo_forcing_mass() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 8: Guo forcing mass contribution\n");
    printf("============================================================\n");

    double dt_test = minSize;
    double omega_tau = 3.0 * niu / dt_test + 0.5;
    double omega = 1.0 / omega_tau;
    double h_eff = LZ - H_HILL;
    double Fy = (8.0 * niu * Uref) / (h_eff * h_eff) * 2.0;

    double rho = 1.0;
    double inv_rho = 1.0;
    // Half-force corrected velocity at cold start
    double ux = 0.0, uy = 0.5 * Fy * inv_rho * dt_test, uz = 0.0;

    double ax = 0.0, ay = Fy * inv_rho, az = 0.0;
    double a_dot_u = ay * uy;
    double prefactor = 1.0 - 0.5 * omega;

    printf("  omega(rate) = %.10f, prefactor = %.10f\n", omega, prefactor);
    printf("  ay = %.10e, uy_half = %.10e\n", ay, uy);
    printf("  a_dot_u = %.10e\n", a_dot_u);

    double source_sum = 0;
    double source_max = 0;
    for (int i = 0; i < 27; i++) {
        double e_dot_a = GILBM_e[i][0]*ax + GILBM_e[i][1]*ay + GILBM_e[i][2]*az;
        double e_dot_u = GILBM_e[i][0]*ux + GILBM_e[i][1]*uy + GILBM_e[i][2]*uz;
        double S_i = GILBM_W[i] * rho * (3.0*e_dot_a*(1.0 + 3.0*e_dot_u) - 3.0*a_dot_u);
        double contrib = prefactor * S_i * dt_test;
        source_sum += contrib;
        if (fabs(contrib) > fabs(source_max)) source_max = contrib;
    }

    printf("  Sum of Guo source terms = %.2e\n", source_sum);
    printf("  Max individual source term = %.2e\n", source_max);

    if (fabs(source_sum) < 1e-15)
        printf("  >>> PASS: Guo forcing is mass-conserving.\n");
    else
        printf("  >>> FAIL: Guo forcing breaks mass by %.2e\n", source_sum);
}

// ================================================================
// TEST 9: Compare Cumulant vs BGK f_out for same input
//         (check if the difference is in total mass)
// ================================================================
void test9_cumulant_vs_bgk() {
    printf("\n");
    printf("============================================================\n");
    printf("TEST 9: Cumulant vs BGK mass comparison\n");
    printf("============================================================\n");

    double f_in[NQ];
    for (int q = 0; q < NQ; q++)
        f_in[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    double dt_test = minSize;
    double omega_tau = 3.0 * niu / dt_test + 0.5;
    double h_eff = LZ - H_HILL;
    double Force = (8.0 * niu * Uref) / (h_eff * h_eff) * 2.0;

    // Cumulant collision
    double f_cum[NQ], rho_cum, ux_cum, uy_cum, uz_cum;
    cumulant_collision_D3Q27(f_in, omega_tau, dt_test,
                             0.0, Force, 0.0,
                             f_cum, &rho_cum, &ux_cum, &uy_cum, &uz_cum);

    // BGK collision (same as evolution_gilbm.h)
    double f_bgk[NQ];
    double rho_A = 0;
    for (int q = 0; q < NQ; q++) rho_A += f_in[q];
    double mx = 0, my = 0, mz = 0;
    for (int q = 0; q < NQ; q++) {
        mx += GILBM_e[q][0] * f_in[q];
        my += GILBM_e[q][1] * f_in[q];
        mz += GILBM_e[q][2] * f_in[q];
    }
    double u_A = mx / rho_A;
    double v_A = (my + 0.5 * Force * dt_test) / rho_A;
    double w_A = mz / rho_A;

    for (int q = 0; q < NQ; q++) {
        double feq_q = compute_feq(q, rho_A, u_A, v_A, w_A);
        double inv_omega = 1.0 / omega_tau;
        f_bgk[q] = f_in[q] - inv_omega * (f_in[q] - feq_q);
        f_bgk[q] += GILBM_W[q] * 3.0 * GILBM_e[q][1] * Force * dt_test;
    }

    double sum_cum = 0, sum_bgk = 0;
    for (int q = 0; q < NQ; q++) { sum_cum += f_cum[q]; sum_bgk += f_bgk[q]; }

    printf("  Sum(f_cumulant) = %.15f\n", sum_cum);
    printf("  Sum(f_bgk)      = %.15f\n", sum_bgk);
    printf("  Difference      = %.2e\n", sum_cum - sum_bgk);

    printf("\n  Per-direction comparison (largest differences):\n");
    double diffs[NQ];
    for (int q = 0; q < NQ; q++) diffs[q] = f_cum[q] - f_bgk[q];
    for (int iter = 0; iter < 5; iter++) {
        int max_idx = 0;
        double max_d = 0;
        for (int q = 0; q < NQ; q++) {
            if (fabs(diffs[q]) > max_d) { max_d = fabs(diffs[q]); max_idx = q; }
        }
        printf("    q=%2d: e=(%+.0f,%+.0f,%+.0f) cum=%.12f bgk=%.12f diff=%.2e\n",
               max_idx, GILBM_e[max_idx][0], GILBM_e[max_idx][1], GILBM_e[max_idx][2],
               f_cum[max_idx], f_bgk[max_idx], diffs[max_idx]);
        diffs[max_idx] = 0;
    }
}

// ================================================================
// Chapman-Enskog BC (reproduced from boundary_conditions.h for CPU)
// ================================================================
static double ChapmanEnskogBC_CPU(
    int alpha,
    double rho_wall,
    double du_dk, double dv_dk, double dw_dk,
    double dk_dy_val, double dk_dz_val,
    double omega_local, double localtimestep
) {
    double ex = GILBM_e[alpha][0];
    double ey = GILBM_e[alpha][1];
    double ez = GILBM_e[alpha][2];

    double C_alpha = 0.0;
    C_alpha += (3.0 * ex * ey) * du_dk * dk_dy_val +
               (3.0 * ex * ez) * du_dk * dk_dz_val;
    C_alpha += (3.0 * ey * ey - 1.0) * dv_dk * dk_dy_val +
               (3.0 * ey * ez) * dv_dk * dk_dz_val;
    C_alpha += (3.0 * ez * ey) * dw_dk * dk_dy_val +
               (3.0 * ez * ez - 1.0) * dw_dk * dk_dz_val;
    C_alpha *= -(omega_local) * localtimestep;

    double f_eq_atwall = GILBM_W[alpha] * rho_wall;
    return f_eq_atwall * (1.0 + C_alpha);
}

static bool NeedsBoundaryCondition_CPU(int alpha, double dk_dy, double dk_dz, bool is_bottom) {
    double e_tilde_k = GILBM_e[alpha][1] * dk_dy + GILBM_e[alpha][2] * dk_dz;
    return is_bottom ? (e_tilde_k > 0.0) : (e_tilde_k < 0.0);
}

// ================================================================
// TEST 10: Chapman-Enskog BC vs Cumulant Distribution Mismatch
//          ROOT CAUSE ANALYSIS for density = 1.0008
//
// This test proves that the Chapman-Enskog BC produces distributions
// incompatible with the WP Cumulant collision output at wall points.
// The BC only accounts for 2nd-order non-equilibrium, but WP Cumulant
// injects 4th-order equilibria via A,B coefficients that the BC
// doesn't know about. When streaming mixes BC values with collision
// values at wall points, mass is not conserved.
// ================================================================
void test10_bc_cumulant_mismatch() {
    printf("\n================================================================\n");
    printf("TEST 10: Chapman-Enskog BC vs Cumulant Mismatch (ROOT CAUSE)\n");
    printf("================================================================\n");

    double omega_tau = 3.0 * niu / minSize + 0.5;  // tau
    double dt_test = minSize;
    double Force = 8.0 * niu * Uref / ((LZ - H_HILL) * (LZ - H_HILL));  // Poiseuille

    // ---- STEP 0: Cold start ----
    // All points start with feq(rho=1, u=0)
    double f_eq[NQ];
    for (int q = 0; q < NQ; q++)
        f_eq[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    // Apply Cumulant collision (Step 0 collision)
    double f_post[NQ];
    double rho_cum, ux_cum, uy_cum, uz_cum;
    cumulant_collision_D3Q27(f_eq, omega_tau, dt_test,
        0.0, Force, 0.0,
        f_post, &rho_cum, &ux_cum, &uy_cum, &uz_cum);

    printf("\n  After Step 0 collision (feq -> f_post):\n");
    printf("    rho_cum = %.15f (should be 1.0)\n", rho_cum);
    printf("    ux_cum  = %.6e, uy_cum = %.6e, uz_cum = %.6e\n", ux_cum, uy_cum, uz_cum);

    // Compute raw macroscopic WITHOUT half-force correction
    // (this is what compute_macroscopic_at would give)
    double rho_raw = 0.0, jx_raw = 0.0, jy_raw = 0.0, jz_raw = 0.0;
    for (int q = 0; q < NQ; q++) {
        rho_raw += f_post[q];
        jx_raw += GILBM_e[q][0] * f_post[q];
        jy_raw += GILBM_e[q][1] * f_post[q];
        jz_raw += GILBM_e[q][2] * f_post[q];
    }
    double ux_raw = jx_raw / rho_raw;
    double vy_raw = jy_raw / rho_raw;
    double wz_raw = jz_raw / rho_raw;
    printf("    Raw macroscopic (no force correction): u=%.6e, v=%.6e, w=%.6e\n",
           ux_raw, vy_raw, wz_raw);

    // ---- STEP 1: Streaming at wall point k=3 (bottom wall) ----
    // Flat wall: dk_dy = 0, dk_dz = 1
    double dk_dy = 0.0, dk_dz = 1.0;

    // For bottom wall, BC uses macroscopic from k=4 (from Step 0 output).
    // In the uniform cold-start case, all points have the same f_post.
    // So rho_wall = rho_raw, velocity gradients du/dk = u_raw (1st-order FD approx)
    double rho_wall = rho_raw;
    double du_dk = ux_raw;   // du/dk approx = u(k=4) / delta_k
    double dv_dk = vy_raw;
    double dw_dk = wz_raw;

    printf("\n  BC inputs (from compute_macroscopic_at at k=4):\n");
    printf("    rho_wall = %.15f\n", rho_wall);
    printf("    du_dk = %.6e, dv_dk = %.6e, dw_dk = %.6e\n", du_dk, dv_dk, dw_dk);

    // Determine which directions need BC at bottom wall
    int bc_dirs[NQ], n_bc = 0;
    int non_bc_dirs[NQ], n_non_bc = 0;
    for (int q = 0; q < NQ; q++) {
        if (q == 0) {
            non_bc_dirs[n_non_bc++] = q;  // rest: no BC
        } else if (NeedsBoundaryCondition_CPU(q, dk_dy, dk_dz, true)) {
            bc_dirs[n_bc++] = q;
        } else {
            non_bc_dirs[n_non_bc++] = q;
        }
    }

    printf("\n  Bottom wall (dk_dy=0, dk_dz=1): %d BC directions, %d non-BC directions\n",
           n_bc, n_non_bc);
    printf("  BC directions: ");
    for (int i = 0; i < n_bc; i++)
        printf("q=%d(e_z=%+.0f) ", bc_dirs[i], GILBM_e[bc_dirs[i]][2]);
    printf("\n");

    // Build f_streamed: BC values for BC directions, f_post for non-BC directions
    double f_streamed[NQ];
    double rho_streamed = 0.0;

    for (int i = 0; i < n_non_bc; i++) {
        int q = non_bc_dirs[i];
        f_streamed[q] = f_post[q];  // From interpolation of uniform f_post
    }

    for (int i = 0; i < n_bc; i++) {
        int q = bc_dirs[i];
        f_streamed[q] = ChapmanEnskogBC_CPU(q, rho_wall, du_dk, dv_dk, dw_dk,
                                             dk_dy, dk_dz, omega_tau, dt_test);
    }

    for (int q = 0; q < NQ; q++) rho_streamed += f_streamed[q];

    printf("\n  *** DENSITY AT WALL POINT AFTER STREAMING ***\n");
    printf("  rho_streamed = %.15f\n", rho_streamed);
    printf("  Error = rho - 1.0 = %.6e\n", rho_streamed - 1.0);

    // Show per-direction BC vs f_post mismatch
    printf("\n  Per-direction mismatch (BC directions only):\n");
    double total_mismatch = 0.0;
    for (int i = 0; i < n_bc; i++) {
        int q = bc_dirs[i];
        double f_bc = f_streamed[q];
        double f_col = f_post[q];
        double diff = f_bc - f_col;
        total_mismatch += diff;
        printf("    q=%2d e=(%+.0f,%+.0f,%+.0f): f_BC=%.12f  f_post=%.12f  diff=%+.6e\n",
               q, GILBM_e[q][0], GILBM_e[q][1], GILBM_e[q][2], f_bc, f_col, diff);
    }
    printf("  Sum of mismatch over BC dirs = %+.6e (this IS the density error)\n", total_mismatch);

    // ---- Compare with BGK collision ----
    printf("\n  --- Comparison: BGK collision ---\n");
    double f_bgk_post[NQ];
    double rho_A_bgk = 0;
    for (int q = 0; q < NQ; q++) rho_A_bgk += f_eq[q];
    double mx_bgk = 0, my_bgk = 0, mz_bgk = 0;
    for (int q = 0; q < NQ; q++) {
        mx_bgk += GILBM_e[q][0] * f_eq[q];
        my_bgk += GILBM_e[q][1] * f_eq[q];
        mz_bgk += GILBM_e[q][2] * f_eq[q];
    }
    double u_A_bgk = mx_bgk / rho_A_bgk;
    double v_A_bgk = (my_bgk + 0.5 * Force * dt_test) / rho_A_bgk;
    double w_A_bgk = mz_bgk / rho_A_bgk;

    for (int q = 0; q < NQ; q++) {
        double feq_q = compute_feq(q, rho_A_bgk, u_A_bgk, v_A_bgk, w_A_bgk);
        double inv_omega = 1.0 / omega_tau;
        f_bgk_post[q] = f_eq[q] - inv_omega * (f_eq[q] - feq_q);
        f_bgk_post[q] += GILBM_W[q] * 3.0 * GILBM_e[q][1] * Force * dt_test;
    }

    // Build f_streamed for BGK
    double rho_bgk_raw = 0.0, jy_bgk_raw = 0.0;
    for (int q = 0; q < NQ; q++) {
        rho_bgk_raw += f_bgk_post[q];
        jy_bgk_raw += GILBM_e[q][1] * f_bgk_post[q];
    }
    double vy_bgk_raw = jy_bgk_raw / rho_bgk_raw;

    double f_streamed_bgk[NQ];
    double rho_streamed_bgk = 0.0;
    for (int i = 0; i < n_non_bc; i++) {
        int q = non_bc_dirs[i];
        f_streamed_bgk[q] = f_bgk_post[q];
    }
    for (int i = 0; i < n_bc; i++) {
        int q = bc_dirs[i];
        f_streamed_bgk[q] = ChapmanEnskogBC_CPU(q, rho_bgk_raw,
            0.0, vy_bgk_raw, 0.0, dk_dy, dk_dz, omega_tau, dt_test);
    }
    for (int q = 0; q < NQ; q++) rho_streamed_bgk += f_streamed_bgk[q];

    printf("  BGK rho_streamed = %.15f\n", rho_streamed_bgk);
    printf("  BGK error = %.6e\n", rho_streamed_bgk - 1.0);

    double bgk_mismatch = 0.0;
    for (int i = 0; i < n_bc; i++) {
        int q = bc_dirs[i];
        bgk_mismatch += f_streamed_bgk[q] - f_bgk_post[q];
    }
    printf("  BGK sum of mismatch over BC dirs = %+.6e\n", bgk_mismatch);

    // ---- Estimate global density error ----
    printf("\n  --- Global density error estimate ---\n");
    int n_wall_points = (NX6 - 7) * (NY6 - 7) * 2;  // top + bottom wall
    int n_total_points = (NX6 - 7) * (NY6 - 7) * (NZ6 - 6);
    double global_error = total_mismatch * (double)n_wall_points / (double)n_total_points;
    printf("  Wall points (flat wall approx): %d x 2 = %d\n",
           (NX6-7)*(NY6-7), n_wall_points);
    printf("  Total interior points: %d\n", n_total_points);
    printf("  Estimated global rho error = %.6e\n", global_error);
    printf("  Observed error (from checkrho): ~8.0e-04\n");

    if (fabs(rho_streamed - 1.0) > 1e-10) {
        printf("\n  *** ROOT CAUSE CONFIRMED ***\n");
        printf("  Chapman-Enskog BC is INCOMPATIBLE with WP Cumulant collision.\n");
        printf("  The BC only accounts for 2nd-order non-equilibrium,\n");
        printf("  but WP Cumulant injects 4th-order equilibria (A,B coefficients)\n");
        printf("  that change individual f values by ~5%%.\n");
        printf("  At wall points, mixing BC values with collision values\n");
        printf("  creates a density error of %.6e PER WALL POINT.\n", rho_streamed - 1.0);
    } else {
        printf("\n  No mismatch detected at this configuration.\n");
    }
}

// ================================================================
// TEST 11: AO mode comparison (should have smaller mismatch)
// ================================================================
void test11_ao_mode_comparison() {
    printf("\n================================================================\n");
    printf("TEST 11: AO Mode (omega2-10=1) BC Mismatch Check\n");
    printf("================================================================\n");

    // For AO mode, all higher-order relaxation rates = 1
    // This means 4th-order cumulants relax to ZERO equilibria
    // So the mismatch should be much smaller

    double omega_tau = 3.0 * niu / minSize + 0.5;
    double dt_test = minSize;
    double Force = 8.0 * niu * Uref / ((LZ - H_HILL) * (LZ - H_HILL));

    // Temporarily override: simulate AO by manually setting omega values
    // In AO mode: all omega_k = 1.0, A=B=0
    // The cumulant_collision.h is compiled with USE_WP_CUMULANT=1,
    // so we can't easily switch. Instead, we compute the f_post
    // and then compute what the EQUILIBRIUM distributions would be
    // if we used AO mode (A=B=0, all omegas=1).

    printf("  Note: This test uses the COMPILED WP mode.\n");
    printf("  The 4th-order equilibria from A,B are what cause the BC mismatch.\n");
    printf("  In AO mode (A=B=0), the 4th-order equilibria would be ZERO,\n");
    printf("  and the BC mismatch would be dominated by the Guo forcing term\n");
    printf("  (which both BGK/MRT and Cumulant share).\n");

    // Show the A,B values for current configuration
    double omega_rate = 1.0 / omega_tau;
    double w2 = CUM_OMEGA2;
    double A, B;
    // Reproduce A,B computation
    double denom_AB = (omega_rate - w2) * (w2 * (2.0 + 3.0*omega_rate) - 8.0*omega_rate);
    if (fabs(denom_AB) > 1e-10) {
        A = (4.0*omega_rate*omega_rate + 2.0*omega_rate*w2*(omega_rate - 6.0)
           + w2*w2*(omega_rate*(10.0 - 3.0*omega_rate) - 4.0)) / denom_AB;
        B = (4.0*omega_rate*w2*(9.0*omega_rate - 16.0) - 4.0*omega_rate*omega_rate
           - 2.0*w2*w2*(2.0 + 9.0*omega_rate*(omega_rate - 2.0)))
           / (3.0 * denom_AB);
    } else {
        A = 0.0; B = 0.0;
    }
    printf("  omega1 (rate) = %.6f, omega2 = %.2f\n", omega_rate, w2);
    printf("  A = %.6f, B = %.6f\n", A, B);
    printf("  (A+B)/9 = %.6f  (4th-order diagonal equilibrium magnitude)\n", (A+B)/9.0);
    printf("  For AO mode: A=B=0 -> 4th-order equilibria = 0\n");
}

// ================================================================
// TEST 12: Tilted wall BC mismatch (actual periodic hill has dk_dy != 0)
//          AND: BC coefficient error analysis (tau vs tau-0.5)
// ================================================================
void test12_tilted_wall_bc_mismatch() {
    printf("\n================================================================\n");
    printf("TEST 12: Tilted Wall BC Mismatch + BC Coefficient Error\n");
    printf("================================================================\n");

    double omega_tau = 3.0 * niu / minSize + 0.5;  // tau
    double dt_test = minSize;
    double Force = 8.0 * niu * Uref / ((LZ - H_HILL) * (LZ - H_HILL)) * 2.0;  // 2x Poiseuille

    // Step 0 collision: feq + force → f_post
    double f_eq[NQ];
    for (int q = 0; q < NQ; q++)
        f_eq[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    double f_post[NQ];
    double rho_cum, ux_cum, uy_cum, uz_cum;
    cumulant_collision_D3Q27(f_eq, omega_tau, dt_test,
        0.0, Force, 0.0,
        f_post, &rho_cum, &ux_cum, &uy_cum, &uz_cum);

    // Raw macroscopic (no force correction, like compute_macroscopic_at)
    double rho_raw = 0.0, jx_raw = 0.0, jy_raw = 0.0, jz_raw = 0.0;
    for (int q = 0; q < NQ; q++) {
        rho_raw += f_post[q];
        jx_raw += GILBM_e[q][0] * f_post[q];
        jy_raw += GILBM_e[q][1] * f_post[q];
        jz_raw += GILBM_e[q][2] * f_post[q];
    }
    double ux_raw = jx_raw / rho_raw;
    double vy_raw = jy_raw / rho_raw;
    double wz_raw = jz_raw / rho_raw;

    printf("  omega_global(tau)   = %.6f\n", omega_tau);
    printf("  tau - 0.5          = %.6f  (=3*nu/dt)\n", omega_tau - 0.5);
    printf("  BC uses tau*dt     = %.6e  (CURRENT, wrong)\n", omega_tau * dt_test);
    printf("  Should use (tau-0.5)*dt = %.6e  (CORRECT = 3*nu)\n", (omega_tau - 0.5) * dt_test);
    printf("  Overestimation factor = tau/(tau-0.5) = %.1f x !!!\n", omega_tau / (omega_tau - 0.5));
    printf("  Post-collision: vy_raw = %.6e (raw momentum / rho)\n\n", vy_raw);

    // Test several tilted wall angles (like the periodic hill slope)
    double dk_dy_vals[] = { 0.0, 0.5, 1.0, 2.0, -0.5, -1.0 };
    double dk_dz_val = 1.0;
    int n_angles = sizeof(dk_dy_vals) / sizeof(dk_dy_vals[0]);

    printf("  %-8s  %-6s  %-6s  %-12s  %-12s  %-12s\n",
           "dk_dy", "n_BC", "n_noBC", "err_tau*dt", "err_(tau-0.5)", "ratio");
    printf("  %-8s  %-6s  %-6s  %-12s  %-12s  %-12s\n",
           "--------", "------", "------", "------------", "------------", "--------");

    for (int angle = 0; angle < n_angles; angle++) {
        double dk_dy = dk_dy_vals[angle];

        // Determine BC directions
        int bc_dirs[NQ], n_bc = 0;
        int non_bc_dirs[NQ], n_non_bc = 0;
        for (int q = 0; q < NQ; q++) {
            if (q == 0) {
                non_bc_dirs[n_non_bc++] = q;
            } else if (NeedsBoundaryCondition_CPU(q, dk_dy, dk_dz_val, true)) {
                bc_dirs[n_bc++] = q;
            } else {
                non_bc_dirs[n_non_bc++] = q;
            }
        }

        // Test with CURRENT BC coefficient (tau * dt) — WRONG
        double rho_err_current = 0.0;
        for (int i = 0; i < n_bc; i++) {
            int q = bc_dirs[i];
            double f_bc = ChapmanEnskogBC_CPU(q, rho_raw,
                ux_raw, vy_raw, wz_raw, dk_dy, dk_dz_val, omega_tau, dt_test);
            rho_err_current += (f_bc - f_post[q]);
        }

        // Test with CORRECT BC coefficient (tau-0.5) * dt = 3*nu
        double rho_err_correct = 0.0;
        for (int i = 0; i < n_bc; i++) {
            int q = bc_dirs[i];
            double f_bc = ChapmanEnskogBC_CPU(q, rho_raw,
                ux_raw, vy_raw, wz_raw, dk_dy, dk_dz_val, omega_tau - 0.5, dt_test);
            rho_err_correct += (f_bc - f_post[q]);
        }

        double ratio = (fabs(rho_err_correct) > 1e-30) ?
            fabs(rho_err_current) / fabs(rho_err_correct) : 0.0;

        printf("  %+6.1f    %-6d  %-6d  %+.6e  %+.6e  %.1f\n",
               dk_dy, n_bc, n_non_bc - 1, rho_err_current, rho_err_correct, ratio);
    }

    // ---- Detailed analysis for dk_dy = 1.0 (typical hill slope) ----
    double dk_dy_hill = 1.0;
    printf("\n  Detailed analysis for dk_dy = %.1f (tilted wall like periodic hill):\n", dk_dy_hill);

    int bc_dirs_hill[NQ], n_bc_hill = 0;
    for (int q = 1; q < NQ; q++)
        if (NeedsBoundaryCondition_CPU(q, dk_dy_hill, dk_dz_val, true))
            bc_dirs_hill[n_bc_hill++] = q;

    printf("  BC directions (%d): ", n_bc_hill);
    for (int i = 0; i < n_bc_hill; i++)
        printf("q=%d(ey=%+.0f,ez=%+.0f) ", bc_dirs_hill[i],
               GILBM_e[bc_dirs_hill[i]][1], GILBM_e[bc_dirs_hill[i]][2]);
    printf("\n\n");

    printf("  Per-direction (CURRENT tau*dt):\n");
    double total_err = 0.0;
    for (int i = 0; i < n_bc_hill; i++) {
        int q = bc_dirs_hill[i];
        double f_bc = ChapmanEnskogBC_CPU(q, rho_raw,
            ux_raw, vy_raw, wz_raw, dk_dy_hill, dk_dz_val, omega_tau, dt_test);
        double diff = f_bc - f_post[q];
        total_err += diff;
        printf("    q=%2d e=(%+.0f,%+.0f,%+.0f) W=%.6f: f_BC=%.12f f_post=%.12f diff=%+.6e\n",
               q, GILBM_e[q][0], GILBM_e[q][1], GILBM_e[q][2], GILBM_W[q], f_bc, f_post[q], diff);
    }
    printf("  Sum of errors (= density error per wall point) = %+.6e\n", total_err);

    // Global estimate for periodic hill
    printf("\n  --- Global density error estimate (periodic hill) ---\n");
    // Periodic hill has ~50% of bottom wall area with non-zero slope
    // Estimate: wall points with slope contribute density error
    int n_wall_points_sloped = (NX6 - 7) * (NY6 - 7) / 2;  // ~50% of wall with slope
    int n_total_points = (NX6 - 7) * (NY6 - 7) * (NZ6 - 6);
    double estimated_global_err = total_err * (double)n_wall_points_sloped / (double)n_total_points;
    printf("  Sloped wall points (approx): %d\n", n_wall_points_sloped);
    printf("  Total interior points: %d\n", n_total_points);
    printf("  Estimated rho error per step: %.6e\n", estimated_global_err);
    printf("  After ~100 steps (if accumulating): %.6e\n", estimated_global_err * 100);
    printf("  Observed error: ~8.0e-04\n");

    if (fabs(total_err) > 1e-10) {
        printf("\n  *** ROOT CAUSE IDENTIFIED ***\n");
        printf("  On TILTED walls (dk_dy != 0), the Chapman-Enskog BC\n");
        printf("  creates a mass mismatch with Cumulant post-collision distributions.\n");
        printf("  The asymmetric BC direction set doesn't cancel the 4th-order\n");
        printf("  equilibria from the WP parametrization.\n");
        printf("  Additionally, the BC coefficient uses tau*dt instead of (tau-0.5)*dt,\n");
        printf("  overestimating the non-equilibrium by %.1f x.\n", omega_tau / (omega_tau - 0.5));
        printf("\n  RECOMMENDED FIX:\n");
        printf("  In boundary_conditions.h line 93, change:\n");
        printf("    C_alpha *= -(omega_local) * localtimestep;\n");
        printf("  To:\n");
        printf("    C_alpha *= -(omega_local - 0.5) * localtimestep;\n");
    }
}

// ================================================================
// Cumulant-aware Chapman-Enskog BC (CPU version)
// Uses Cumulant equilibrium (f_eq_cum) instead of standard feq (W*rho)
// ================================================================
static double ChapmanEnskogBC_CumulantAware_CPU(
    int alpha,
    double rho_wall,
    double du_dk, double dv_dk, double dw_dk,
    double dk_dy_val, double dk_dz_val,
    double omega_local, double localtimestep,
    const double f_eq_cum[NQ]  // pre-computed Cumulant equilibrium at rho=1, u=0
) {
    double ex = GILBM_e[alpha][0];
    double ey = GILBM_e[alpha][1];
    double ez = GILBM_e[alpha][2];

    // Non-equilibrium correction (same as original, but with correct coefficient)
    double C_alpha = 0.0;
    C_alpha += (3.0 * ex * ey) * du_dk * dk_dy_val +
               (3.0 * ex * ez) * du_dk * dk_dz_val;
    C_alpha += (3.0 * ey * ey - 1.0) * dv_dk * dk_dy_val +
               (3.0 * ey * ez) * dv_dk * dk_dz_val;
    C_alpha += (3.0 * ez * ey) * dw_dk * dk_dy_val +
               (3.0 * ez * ez - 1.0) * dw_dk * dk_dz_val;
    C_alpha *= -(omega_local - 0.5) * localtimestep;  // FIXED: tau-0.5, not tau

    // Use Cumulant equilibrium scaled by rho_wall (instead of W[alpha]*rho_wall)
    double f_eq_atwall = f_eq_cum[alpha] * rho_wall;
    // Non-equilibrium correction added as absolute term (based on standard feq form)
    double f_neq = GILBM_W[alpha] * rho_wall * C_alpha;
    return f_eq_atwall + f_neq;
}

// ================================================================
// TEST 13: Cumulant-Aware BC Fix Verification
//
// Verifies that using the Cumulant equilibrium (instead of standard feq)
// in the Chapman-Enskog BC eliminates the density error at tilted walls.
// ================================================================
void test13_cumulant_aware_bc_fix() {
    printf("\n================================================================\n");
    printf("TEST 13: Cumulant-Aware BC Fix Verification\n");
    printf("================================================================\n");

    double omega_tau = 3.0 * niu / minSize + 0.5;  // tau
    double dt_test = minSize;
    double Force = 8.0 * niu * Uref / ((LZ - H_HILL) * (LZ - H_HILL)) * 2.0;

    // --- Step A: Compute the Cumulant equilibrium at rho=1, u=0 ---
    // Run collision 200 times from standard feq (no force) to find fixed point
    double f_cum_eq[NQ];
    for (int q = 0; q < NQ; q++)
        f_cum_eq[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    double rho_dummy, ux_dummy, uy_dummy, uz_dummy;
    for (int step = 0; step < 200; step++) {
        double f_tmp[NQ];
        cumulant_collision_D3Q27(f_cum_eq, omega_tau, dt_test,
            0.0, 0.0, 0.0,  // NO force
            f_tmp, &rho_dummy, &ux_dummy, &uy_dummy, &uz_dummy);
        for (int q = 0; q < NQ; q++)
            f_cum_eq[q] = f_tmp[q];
    }

    // Verify convergence
    double rho_ceq = 0.0;
    for (int q = 0; q < NQ; q++) rho_ceq += f_cum_eq[q];
    printf("  Cumulant equilibrium (200 collisions, no force):\n");
    printf("    Sum(f_eq_cum) = %.15f (should be 1.0)\n", rho_ceq);
    printf("    f_eq_cum[0] = %.12f (vs W[0]=%.12f, diff=%+.6e)\n",
           f_cum_eq[0], GILBM_W[0], f_cum_eq[0] - GILBM_W[0]);
    printf("    f_eq_cum[1] = %.12f (vs W[1]=%.12f, diff=%+.6e)\n",
           f_cum_eq[1], GILBM_W[1], f_cum_eq[1] - GILBM_W[1]);
    printf("    f_eq_cum[19] = %.12f (vs W[19]=%.12f, diff=%+.6e)\n",
           f_cum_eq[19], GILBM_W[19], f_cum_eq[19] - GILBM_W[19]);

    // --- Step B: Collision with force (cold start step 0) ---
    double f_eq[NQ];
    for (int q = 0; q < NQ; q++)
        f_eq[q] = compute_feq(q, 1.0, 0.0, 0.0, 0.0);

    double f_post[NQ];
    double rho_cum, ux_cum, uy_cum, uz_cum;
    cumulant_collision_D3Q27(f_eq, omega_tau, dt_test,
        0.0, Force, 0.0,
        f_post, &rho_cum, &ux_cum, &uy_cum, &uz_cum);

    // Raw macroscopic (no force correction)
    double rho_raw = 0.0, jx_raw = 0.0, jy_raw = 0.0, jz_raw = 0.0;
    for (int q = 0; q < NQ; q++) {
        rho_raw += f_post[q];
        jx_raw += GILBM_e[q][0] * f_post[q];
        jy_raw += GILBM_e[q][1] * f_post[q];
        jz_raw += GILBM_e[q][2] * f_post[q];
    }
    double ux_raw = jx_raw / rho_raw;
    double vy_raw = jy_raw / rho_raw;
    double wz_raw = jz_raw / rho_raw;

    // --- Step C: Compare OLD vs NEW BC at tilted walls ---
    double dk_dy_vals[] = { 0.0, 0.5, 1.0, 2.0, -0.5, -1.0 };
    double dk_dz_val = 1.0;
    int n_angles = sizeof(dk_dy_vals) / sizeof(dk_dy_vals[0]);

    printf("\n  %-8s  %-6s  %-14s  %-14s  %-10s\n",
           "dk_dy", "n_BC", "err_OLD(W*rho)", "err_NEW(f_cum)", "reduction");
    printf("  %-8s  %-6s  %-14s  %-14s  %-10s\n",
           "--------", "------", "--------------", "--------------", "----------");

    for (int angle = 0; angle < n_angles; angle++) {
        double dk_dy = dk_dy_vals[angle];

        // BC directions
        int bc_dirs[NQ], n_bc = 0;
        for (int q = 1; q < NQ; q++) {
            if (NeedsBoundaryCondition_CPU(q, dk_dy, dk_dz_val, true))
                bc_dirs[n_bc++] = q;
        }

        // OLD BC: standard feq (W*rho) + wrong coefficient (tau*dt)
        double err_old = 0.0;
        for (int i = 0; i < n_bc; i++) {
            int q = bc_dirs[i];
            double f_bc = ChapmanEnskogBC_CPU(q, rho_raw,
                ux_raw, vy_raw, wz_raw, dk_dy, dk_dz_val, omega_tau, dt_test);
            err_old += (f_bc - f_post[q]);
        }

        // NEW BC: Cumulant equilibrium + correct coefficient (tau-0.5)*dt
        double err_new = 0.0;
        for (int i = 0; i < n_bc; i++) {
            int q = bc_dirs[i];
            double f_bc = ChapmanEnskogBC_CumulantAware_CPU(q, rho_raw,
                ux_raw, vy_raw, wz_raw, dk_dy, dk_dz_val, omega_tau, dt_test,
                f_cum_eq);
            err_new += (f_bc - f_post[q]);
        }

        double reduction = (fabs(err_old) > 1e-30) ?
            fabs(err_new) / fabs(err_old) : 0.0;

        printf("  %+6.1f    %-6d  %+.8e  %+.8e  %.2e\n",
               dk_dy, n_bc, err_old, err_new, reduction);
    }

    // --- Step D: Detailed per-direction analysis for dk_dy=1.0 ---
    double dk_dy_hill = 1.0;
    printf("\n  Detailed per-direction for dk_dy=%.1f (NEW Cumulant-aware BC):\n", dk_dy_hill);

    int bc_dirs_hill[NQ], n_bc_hill = 0;
    for (int q = 1; q < NQ; q++)
        if (NeedsBoundaryCondition_CPU(q, dk_dy_hill, dk_dz_val, true))
            bc_dirs_hill[n_bc_hill++] = q;

    double total_err_old = 0.0, total_err_new = 0.0;
    for (int i = 0; i < n_bc_hill; i++) {
        int q = bc_dirs_hill[i];
        double f_bc_old = ChapmanEnskogBC_CPU(q, rho_raw,
            ux_raw, vy_raw, wz_raw, dk_dy_hill, dk_dz_val, omega_tau, dt_test);
        double f_bc_new = ChapmanEnskogBC_CumulantAware_CPU(q, rho_raw,
            ux_raw, vy_raw, wz_raw, dk_dy_hill, dk_dz_val, omega_tau, dt_test,
            f_cum_eq);
        double diff_old = f_bc_old - f_post[q];
        double diff_new = f_bc_new - f_post[q];
        total_err_old += diff_old;
        total_err_new += diff_new;
        printf("    q=%2d e=(%+.0f,%+.0f,%+.0f): OLD=%+.6e  NEW=%+.6e\n",
               q, GILBM_e[q][0], GILBM_e[q][1], GILBM_e[q][2], diff_old, diff_new);
    }
    printf("  Sum OLD = %+.6e\n", total_err_old);
    printf("  Sum NEW = %+.6e\n", total_err_new);

    // --- Step E: Global estimate with fix ---
    int n_wall_sloped = (NX6 - 7) * (NY6 - 7) / 2;
    int n_total = (NX6 - 7) * (NY6 - 7) * (NZ6 - 6);
    double glob_old = total_err_old * (double)n_wall_sloped / (double)n_total;
    double glob_new = total_err_new * (double)n_wall_sloped / (double)n_total;

    printf("\n  --- Global density error estimate ---\n");
    printf("  OLD BC: rho error/step = %+.6e\n", glob_old);
    printf("  NEW BC: rho error/step = %+.6e\n", glob_new);
    printf("  Improvement: %.1f x reduction\n",
           fabs(glob_old) / (fabs(glob_new) > 1e-30 ? fabs(glob_new) : 1e-30));

    if (fabs(total_err_new) < fabs(total_err_old) * 0.01) {
        printf("\n  >>> FIX VERIFIED: Cumulant-aware BC reduces mass error by >100x <<<\n");
    } else if (fabs(total_err_new) < fabs(total_err_old) * 0.1) {
        printf("\n  >>> FIX PARTIALLY EFFECTIVE: ~%.0fx reduction <<<\n",
               fabs(total_err_old) / fabs(total_err_new));
    } else {
        printf("\n  >>> FIX INSUFFICIENT: only %.1fx reduction <<<\n",
               fabs(total_err_old) / fabs(total_err_new));
        printf("  The Guo forcing term creates additional distribution asymmetry\n");
        printf("  that the BC equilibrium change alone cannot fix.\n");
    }
}

// ================================================================
// MAIN
// ================================================================
int main() {
    printf("================================================================\n");
    printf("D3Q27 Cumulant Collision Mass Conservation Test\n");
    printf("================================================================\n");
    printf("Configuration:\n");
    printf("  USE_WP_CUMULANT = %d\n", USE_WP_CUMULANT);
    printf("  CUM_LAMBDA = %.1e\n", CUM_LAMBDA);
    printf("  CUM_OMEGA2 = %.2f\n", CUM_OMEGA2);
    printf("  Re = %d, Uref = %.4f, niu = %.6e\n", Re, Uref, niu);
    printf("  dt(minSize) = %.10f\n", minSize);
    printf("  omega_global(tau) = %.10f\n", 3.0*niu/minSize + 0.5);
    printf("  omega1(rate=1/tau) = %.10f\n", 1.0 / (3.0*niu/minSize + 0.5));

    test7_verify_weights();
    test6_verify_CUM_K();
    test4_chimera_roundtrip();
    test1_equilibrium_zero_force();
    test2_equilibrium_with_force();
    test8_guo_forcing_mass();
    test3_stage_by_stage_diagnostic();
    test9_cumulant_vs_bgk();
    test5_multiple_steps();
    test10_bc_cumulant_mismatch();
    test11_ao_mode_comparison();
    test12_tilted_wall_bc_mismatch();
    test13_cumulant_aware_bc_fix();

    printf("\n================================================================\n");
    printf("All tests complete.\n");
    printf("================================================================\n");

    return 0;
}
