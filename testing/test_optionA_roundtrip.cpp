// ================================================================
// test_optionA_roundtrip.cpp
// Proves that Option A (global M/S separation) breaks round-trip
// consistency: backward(forward(f)) ≠ f
//
// Compile: g++ -O2 -o test_optionA test_optionA_roundtrip.cpp -lm
// ================================================================
#include <cstdio>
#include <cmath>
#include <cstring>

// ── Reproduce the constants from cumulant_constants.h (host version) ──
static int CUM_IDX[27][3] = {
    // z-sweep (0-8)
    { 6,  0,  5}, {13,  1, 11}, {14,  2, 12},
    {17,  3, 15}, {18,  4, 16}, {23,  7, 19},
    {24,  8, 20}, {25,  9, 21}, {26, 10, 22},
    // y-sweep (9-17)
    {18,  6, 17}, { 4,  0,  3}, {16,  5, 15},
    {25, 13, 23}, { 9,  1,  7}, {21, 11, 19},
    {26, 14, 24}, {10,  2,  8}, {22, 12, 20},
    // x-sweep (18-26)
    {26, 18, 25}, {14,  6, 13}, {24, 17, 23},
    {10,  4,  9}, { 2,  0,  1}, { 8,  3,  7},
    {22, 16, 21}, {12,  5, 11}, {20, 15, 19}
};

static double CUM_K[27] = {
    // z-sweep
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    // y-sweep
    2.0/3.0, 0.0, 2.0/9.0,
    1.0/6.0, 0.0, 1.0/18.0,
    1.0/6.0, 0.0, 1.0/18.0,
    // x-sweep
    1.0, 0.0, 1.0/3.0,
    0.0, 0.0, 0.0,
    1.0/3.0, 0.0, 1.0/9.0
};

// D3Q27 weights
static double W[27] = {
    8.0/27.0,   // 0: (0,0,0)
    2.0/27.0, 2.0/27.0,  // 1-2: (1,0,0), (-1,0,0)
    2.0/27.0, 2.0/27.0,  // 3-4: (0,1,0), (0,-1,0)
    2.0/27.0, 2.0/27.0,  // 5-6: (0,0,1), (0,0,-1)
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,  // 7-10: xy-edges
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,  // 11-14: xz-edges
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,  // 15-18: yz-edges
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,  // 19-22: corners
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0   // 23-26: corners
};

// ════════════════════════════════════════════════════════════════
// ORIGINAL CHIMERA (known correct)
// ════════════════════════════════════════════════════════════════
void chimera_forward(double m[27], const double u[3]) {
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double k = CUM_K[p];
            double sum  = m[a] + m[c];
            double diff = m[c] - m[a];
            m[a] = m[a] + m[b] + m[c];
            m[b] = diff - (m[a] + k) * u[dir];
            m[c] = sum - 2.0 * diff * u[dir] + u[dir]*u[dir] * (m[a] + k);
        }
    }
}

void chimera_backward(double m[27], const double u[3]) {
    for (int dir = 0; dir < 3; dir++) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double k = CUM_K[p];
            double ma = ((m[c] - m[b]) * 0.5 + m[b] * u[dir]
                        + (m[a] + k) * (u[dir]*u[dir] - u[dir]) * 0.5);
            double mb = (m[a] - m[c]) - 2.0 * m[b] * u[dir]
                        - (m[a] + k) * u[dir] * u[dir];
            double mc = ((m[c] + m[b]) * 0.5 + m[b] * u[dir]
                        + (m[a] + k) * (u[dir]*u[dir] + u[dir]) * 0.5);
            m[a] = ma; m[b] = mb; m[c] = mc;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// OPTION A: Global separation (CURRENT IMPLEMENTATION — BUGGY)
// ════════════════════════════════════════════════════════════════
void matrix_forward(double m[27], const double u[3]) {
    // Phase 1: ALL raw moments (z->y->x)
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double fa = m[a], fb = m[b], fc = m[c];
            m[a] = fa + fb + fc;
            m[b] = fc - fa;
            m[c] = fa + fc;
        }
    }
    // Phase 2: ALL shifts (z->y->x)
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double k = CUM_K[p];
            double m0 = m[a], m1 = m[b], m2 = m[c];
            m[b] = m1 - (m0 + k) * u[dir];
            m[c] = m2 - 2.0 * m1 * u[dir] + (m0 + k) * u[dir] * u[dir];
        }
    }
}

void matrix_backward(double m[27], const double u[3]) {
    // Phase 1: ALL inverse shifts (x->y->z)
    for (int dir = 0; dir < 3; dir++) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double k = CUM_K[p];
            double k0 = m[a], k1 = m[b], k2 = m[c];
            m[b] = k1 + (k0 + k) * u[dir];
            m[c] = k2 + 2.0 * k1 * u[dir] + (k0 + k) * u[dir] * u[dir];
        }
    }
    // Phase 2: ALL inverse raws (x->y->z)
    for (int dir = 0; dir < 3; dir++) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double m0 = m[a], m1 = m[b], m2 = m[c];
            m[a] = (-m1 + m2) * 0.5;
            m[b] = m0 - m2;
            m[c] = (m1 + m2) * 0.5;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// OPTION A-FIX: Per-direction separation (M then S within each dir)
// ════════════════════════════════════════════════════════════════
void matrix_perdir_forward(double m[27], const double u[3]) {
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;
        // Sub-phase 1: Raw moments for THIS direction only
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double fa = m[a], fb = m[b], fc = m[c];
            m[a] = fa + fb + fc;
            m[b] = fc - fa;
            m[c] = fa + fc;
        }
        // Sub-phase 2: Shift for THIS direction only
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double k = CUM_K[p];
            double m0 = m[a], m1 = m[b], m2 = m[c];
            m[b] = m1 - (m0 + k) * u[dir];
            m[c] = m2 - 2.0 * m1 * u[dir] + (m0 + k) * u[dir] * u[dir];
        }
    }
}

void matrix_perdir_backward(double m[27], const double u[3]) {
    for (int dir = 0; dir < 3; dir++) {
        int base = (2 - dir) * 9;
        // Sub-phase 1: Inverse shift for THIS direction only
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double k = CUM_K[p];
            double k0 = m[a], k1 = m[b], k2 = m[c];
            m[b] = k1 + (k0 + k) * u[dir];
            m[c] = k2 + 2.0 * k1 * u[dir] + (k0 + k) * u[dir] * u[dir];
        }
        // Sub-phase 2: Inverse raw for THIS direction only
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0], b = CUM_IDX[p][1], c = CUM_IDX[p][2];
            double m0 = m[a], m1 = m[b], m2 = m[c];
            m[a] = (-m1 + m2) * 0.5;
            m[b] = m0 - m2;
            m[c] = (m1 + m2) * 0.5;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Test helper: generate a non-trivial f near equilibrium
// ════════════════════════════════════════════════════════════════
void make_test_f(double f[27], double rho, double ux, double uy, double uz) {
    // Simple feq + small perturbation
    double usq = ux*ux + uy*uy + uz*uz;
    int ex[27] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0,
                  1,-1, 1,-1, 1,-1, 1,-1};
    int ey[27] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1, 1,-1,-1,
                  1, 1,-1,-1, 1, 1,-1,-1};
    int ez[27] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1,-1, 1,-1,
                  1, 1, 1, 1,-1,-1,-1,-1};
    for (int i = 0; i < 27; i++) {
        double eu = ex[i]*ux + ey[i]*uy + ez[i]*uz;
        f[i] = W[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq);
    }
}

// ════════════════════════════════════════════════════════════════
int main() {
    printf("================================================================\n");
    printf("  Option A Round-Trip Test: backward(forward(f)) == f ?\n");
    printf("================================================================\n\n");

    double rho = 1.02;
    double ux = 0.05, uy = 0.03, uz = -0.02;
    double u[3] = {ux, uy, uz};

    double f_orig[27], m[27];
    make_test_f(f_orig, rho, ux, uy, uz);

    // ── TEST 1: Chimera round-trip (known correct) ──
    printf("TEST 1: Chimera round-trip\n");
    for (int i = 0; i < 27; i++) m[i] = f_orig[i] - W[i];
    chimera_forward(m, u);
    chimera_backward(m, u);
    for (int i = 0; i < 27; i++) m[i] += W[i];  // restore
    double max_err1 = 0;
    for (int i = 0; i < 27; i++) {
        double err = fabs(m[i] - f_orig[i]);
        if (err > max_err1) max_err1 = err;
    }
    printf("  max|f_roundtrip - f_orig| = %.2e  %s\n\n",
           max_err1, max_err1 < 1e-14 ? "PASS" : "*** FAIL ***");

    // ── TEST 2: Option A GLOBAL round-trip (BUGGY) ──
    printf("TEST 2: Option A GLOBAL separation round-trip\n");
    for (int i = 0; i < 27; i++) m[i] = f_orig[i] - W[i];
    matrix_forward(m, u);
    matrix_backward(m, u);
    for (int i = 0; i < 27; i++) m[i] += W[i];
    double max_err2 = 0;
    int worst2 = -1;
    for (int i = 0; i < 27; i++) {
        double err = fabs(m[i] - f_orig[i]);
        if (err > max_err2) { max_err2 = err; worst2 = i; }
    }
    printf("  max|f_roundtrip - f_orig| = %.2e at i=%d  %s\n",
           max_err2, worst2, max_err2 < 1e-14 ? "PASS" : "*** FAIL ***");
    if (max_err2 > 1e-14) {
        printf("  >>> BUG CONFIRMED: Global M/S separation breaks round-trip!\n");
        printf("  >>> This explains Ma_max=0 (velocity destroyed each step).\n");
    }
    printf("\n");

    // ── TEST 3: Option A PER-DIRECTION round-trip ──
    printf("TEST 3: Option A PER-DIRECTION separation round-trip\n");
    for (int i = 0; i < 27; i++) m[i] = f_orig[i] - W[i];
    matrix_perdir_forward(m, u);
    matrix_perdir_backward(m, u);
    for (int i = 0; i < 27; i++) m[i] += W[i];
    double max_err3 = 0;
    for (int i = 0; i < 27; i++) {
        double err = fabs(m[i] - f_orig[i]);
        if (err > max_err3) max_err3 = err;
    }
    printf("  max|f_roundtrip - f_orig| = %.2e  %s\n\n",
           max_err3, max_err3 < 1e-14 ? "PASS" : "*** FAIL ***");

    // ── TEST 4: Compare forward transform outputs ──
    printf("TEST 4: Compare forward outputs (Chimera vs Option A)\n");
    double m_chi[27], m_glo[27], m_per[27];
    for (int i = 0; i < 27; i++) {
        m_chi[i] = f_orig[i] - W[i];
        m_glo[i] = f_orig[i] - W[i];
        m_per[i] = f_orig[i] - W[i];
    }
    chimera_forward(m_chi, u);
    matrix_forward(m_glo, u);
    matrix_perdir_forward(m_per, u);

    printf("  Chimera vs GLOBAL:\n");
    double max_cg = 0;
    for (int i = 0; i < 27; i++) {
        double err = fabs(m_chi[i] - m_glo[i]);
        if (err > max_cg) max_cg = err;
    }
    printf("    max|chimera - global| = %.2e  %s\n",
           max_cg, max_cg < 1e-14 ? "MATCH" : "*** DIFFER ***");

    printf("  Chimera vs PER-DIRECTION:\n");
    double max_cp = 0;
    for (int i = 0; i < 27; i++) {
        double err = fabs(m_chi[i] - m_per[i]);
        if (err > max_cp) max_cp = err;
    }
    printf("    max|chimera - perdir| = %.2e  %s\n",
           max_cp, max_cp < 1e-14 ? "MATCH" : "*** DIFFER ***");

    if (max_cg > 1e-14 && max_cp < 1e-14) {
        printf("\n  >>> CONCLUSION: GLOBAL separation gives WRONG central moments.\n");
        printf("  >>> PER-DIRECTION separation is IDENTICAL to Chimera.\n");
    }

    // ── TEST 5: Show per-element differences for global ──
    printf("\nTEST 5: Per-element difference (Chimera vs GLOBAL forward)\n");
    printf("  idx   Chimera_val          Global_val           diff\n");
    for (int i = 0; i < 27; i++) {
        double diff = m_chi[i] - m_glo[i];
        if (fabs(diff) > 1e-15) {
            printf("  %2d   %+.12e  %+.12e  %+.2e\n",
                   i, m_chi[i], m_glo[i], diff);
        }
    }

    // ── TEST 6: Effect on extracted velocity ──
    printf("\nTEST 6: Velocity extraction after forward transform\n");
    // After forward Chimera: m[I_baa]=κ100, m[I_aba]=κ010, m[I_aab]=κ001
    // At equilibrium: κ100 ≈ 0 (well-conditioned), κ010 ≈ 0, κ001 ≈ 0
    // The MACROSCOPIC velocity comes from Stage 0 (before transform),
    // but the 1st-order central moments should be ~0 at equilibrium.
    printf("  Chimera: κ100=%.6e, κ010=%.6e, κ001=%.6e\n",
           m_chi[18], m_chi[14], m_chi[10]);
    printf("  GLOBAL:  κ100=%.6e, κ010=%.6e, κ001=%.6e\n",
           m_glo[18], m_glo[14], m_glo[10]);
    printf("  PERDIR:  κ100=%.6e, κ010=%.6e, κ001=%.6e\n",
           m_per[18], m_per[14], m_per[10]);

    // ── TEST 7: At u=0, global should be correct ──
    printf("\nTEST 7: Round-trip at u=0 (trivial case, S=I)\n");
    double u0[3] = {0, 0, 0};
    for (int i = 0; i < 27; i++) m[i] = f_orig[i] - W[i];
    matrix_forward(m, u0);
    matrix_backward(m, u0);
    for (int i = 0; i < 27; i++) m[i] += W[i];
    double max_err7 = 0;
    for (int i = 0; i < 27; i++) {
        double err = fabs(m[i] - f_orig[i]);
        if (err > max_err7) max_err7 = err;
    }
    printf("  max|f_roundtrip - f_orig| at u=0 = %.2e  %s\n",
           max_err7, max_err7 < 1e-14 ? "PASS" : "*** FAIL ***");
    printf("  (At u=0, S=I, so global separation should work.)\n");

    printf("\n================================================================\n");
    printf("  SUMMARY\n");
    printf("================================================================\n");
    printf("  Chimera round-trip:   %.2e  %s\n", max_err1,
           max_err1 < 1e-14 ? "PASS" : "FAIL");
    printf("  GLOBAL round-trip:    %.2e  %s\n", max_err2,
           max_err2 < 1e-14 ? "PASS" : "FAIL");
    printf("  PERDIR round-trip:    %.2e  %s\n", max_err3,
           max_err3 < 1e-14 ? "PASS" : "FAIL");
    printf("  Chi==GLOBAL forward:  %.2e  %s\n", max_cg,
           max_cg < 1e-14 ? "MATCH" : "DIFFER");
    printf("  Chi==PERDIR forward:  %.2e  %s\n", max_cp,
           max_cp < 1e-14 ? "MATCH" : "DIFFER");
    printf("  GLOBAL at u=0:        %.2e  %s\n", max_err7,
           max_err7 < 1e-14 ? "PASS" : "FAIL");

    if (max_err2 > 1e-10) {
        printf("\n  *** ROOT CAUSE OF Ma_max=0 IDENTIFIED ***\n");
        printf("  The global separation of Phase 1 (all M) and Phase 2 (all S)\n");
        printf("  is INCORRECT because:\n");
        printf("    1. The y-sweep in Phase 1 OVERWRITES z-sweep positions.\n");
        printf("    2. Phase 2 z-shift then reads WRONG values (y-raw, not z-raw).\n");
        printf("    3. K constants are calibrated for interleaved M+S per direction,\n");
        printf("       not for globally separated M then S.\n");
        printf("    4. Forward and backward are no longer inverse operations.\n");
        printf("    5. The collision destroys velocity every step → Ma_max=0.\n");
        printf("\n  FIX: Use per-direction separation (M then S within each dir),\n");
        printf("  which is mathematically IDENTICAL to the Chimera.\n");
    }

    return 0;
}
