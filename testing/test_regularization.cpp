// ============================================================================
// Unit Test: Pre-collision Regularization (RLBM)
// Reference: Latt & Chopard, "Lattice Boltzmann method with regularized
//            pre-collision distribution functions", 2006
//
// 驗證項目:
//   1. f_reg = f^eq(ρ,u) + f^(1)(Π^neq)  matches Latt Eq.(10)+(11)
//   2. Moment preservation: ρ, j_α (exact), Π_αβ (exact)
//   3. 3rd-order moments: regularized → equilibrium values (= 0 for central 3rd)
//   4. Formula verification: f^(1)_i = (t_i / 2cs⁴) · Q_iαβ · Π^neq_αβ
//
// Notation mapping (Latt → code):
//   t_i  →  GILBM_W[q]          (lattice weight)
//   cs²  →  1/3
//   cs⁴  →  1/9
//   Q_iαβ = e_iα·e_iβ - cs²·δ_αβ
//   Π^neq_αβ = Σ f_i·e_iα·e_iβ - ρ·u_α·u_β - ρ·cs²·δ_αβ   (Latt Eq.6+4)
//   f^(1)_i = (t_i / 2cs⁴) · Q_iαβ · Π^neq_αβ               (Latt Eq.10)
//   f_reg_i = f^eq_i + f^(1)_i                                 (Latt Eq.11)
// ============================================================================

#include <cstdio>
#include <cmath>
#include <cstdlib>

static const int NQ = 27;

// D3Q27 velocity set (same ordering as GILBM code)
static const double e[NQ][3] = {
    { 0, 0, 0},                                         // 0: rest
    { 1, 0, 0}, {-1, 0, 0},                             // 1-2: ±x
    { 0, 1, 0}, { 0,-1, 0},                             // 3-4: ±y
    { 0, 0, 1}, { 0, 0,-1},                             // 5-6: ±z
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},    // 7-10: xy edges
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},    // 11-14: xz edges
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},    // 15-18: yz edges
    { 1, 1, 1}, {-1, 1, 1}, { 1,-1, 1}, {-1,-1, 1},    // 19-22: corners +z
    { 1, 1,-1}, {-1, 1,-1}, { 1,-1,-1}, {-1,-1,-1}     // 23-26: corners -z
};

// D3Q27 weights
static const double W[NQ] = {
    8.0/27.0,                                            // rest
    2.0/27.0, 2.0/27.0, 2.0/27.0,                       // face
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,             // edge
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,          // corner
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

// Kronecker delta
static inline double kron(int a, int b) { return (a == b) ? 1.0 : 0.0; }

// ── Compute f^eq (standard 2nd-order D3Q27 equilibrium) ──
static double compute_feq(int q, double rho, double ux, double uy, double uz) {
    double eu = e[q][0]*ux + e[q][1]*uy + e[q][2]*uz;
    double usq = ux*ux + uy*uy + uz*uz;
    return W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq);
}

// ── Compute Π^neq_αβ from distributions (Latt Eq.6 + Eq.4) ──
//    Π^neq_αβ = Σ f_i·e_iα·e_iβ - Π^eq_αβ
//    Π^eq_αβ = ρ·u_α·u_β + ρ·cs²·δ_αβ    (Latt Eq.4)
static void compute_Pi_neq(const double f[NQ], double rho,
                           double ux, double uy, double uz,
                           double Pi_neq[3][3]) {
    double u[3] = {ux, uy, uz};

    // Raw 2nd moment: Σ f·eα·eβ
    double Pi_raw[3][3] = {};
    for (int q = 0; q < NQ; q++)
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                Pi_raw[a][b] += f[q] * e[q][a] * e[q][b];

    // Subtract equilibrium: Π^eq = ρ·u·u + (ρ/3)·δ
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            Pi_neq[a][b] = Pi_raw[a][b] - rho * u[a] * u[b] - rho / 3.0 * kron(a, b);
}

// ── Compute f^(1) via Latt Eq.10 ──
//    f^(1)_i = (t_i / 2cs⁴) · Q_iαβ · Π^neq_αβ
//    where Q_iαβ = e_iα·e_iβ - cs²·δ_αβ, cs² = 1/3, cs⁴ = 1/9
//
//    t_i / (2·cs⁴) = W[q] / (2/9) = W[q] · 9/2 = 4.5 · W[q]
static double compute_f1_latt(int q, const double Pi_neq[3][3]) {
    double result = 0.0;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
            double Q_ab = e[q][a] * e[q][b] - kron(a, b) / 3.0;
            result += Q_ab * Pi_neq[a][b];
        }
    // prefactor = t_i / (2·cs⁴) = W[q] · 9/2
    return W[q] * 4.5 * result;
}

// ── Our code's regularization (copied from evolution_gilbm.h STEP 1.75) ──
static void regularize_code(double f[NQ], double rho,
                            double ux, double uy, double uz) {
    // Compute Π^neq (same as code)
    double Pxx=0, Pyy=0, Pzz=0, Pxy=0, Pxz=0, Pyz=0;
    for (int q = 0; q < NQ; q++) {
        double ex = e[q][0], ey = e[q][1], ez = e[q][2];
        Pxx += f[q] * ex * ex;
        Pyy += f[q] * ey * ey;
        Pzz += f[q] * ez * ez;
        Pxy += f[q] * ex * ey;
        Pxz += f[q] * ex * ez;
        Pyz += f[q] * ey * ez;
    }
    Pxx -= rho * ux * ux + rho / 3.0;
    Pyy -= rho * uy * uy + rho / 3.0;
    Pzz -= rho * uz * uz + rho / 3.0;
    Pxy -= rho * ux * uy;
    Pxz -= rho * ux * uz;
    Pyz -= rho * uy * uz;

    // Reconstruct
    for (int q = 0; q < NQ; q++) {
        double ex = e[q][0], ey = e[q][1], ez = e[q][2];
        double eu = ex*ux + ey*uy + ez*uz;
        double usq = ux*ux + uy*uy + uz*uz;

        double feq_bare = W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq);
        double fneq_2nd = W[q] * 4.5 * (
            (ex*ex - 1.0/3.0) * Pxx +
            (ey*ey - 1.0/3.0) * Pyy +
            (ez*ez - 1.0/3.0) * Pzz +
            2.0 * ex * ey * Pxy +
            2.0 * ex * ez * Pxz +
            2.0 * ey * ez * Pyz
        );

        f[q] = feq_bare + fneq_2nd;
    }
}

// ── Build a non-trivial test distribution ──
// Start from equilibrium, add a known non-equilibrium perturbation
// that has content at ALL moment orders (2nd, 3rd, 4th, etc.)
static void build_test_distribution(double f[NQ],
                                    double &rho_out, double &ux_out, double &uy_out, double &uz_out) {
    // Physical parameters
    double rho = 1.02;   // slightly above 1
    double ux  = 0.03;   // low Ma
    double uy  = 0.05;
    double uz  = -0.02;

    rho_out = rho;
    ux_out  = ux;
    uy_out  = uy;
    uz_out  = uz;

    // Start with equilibrium
    for (int q = 0; q < NQ; q++)
        f[q] = compute_feq(q, rho, ux, uy, uz);

    // Add a realistic non-equilibrium perturbation (simulating interpolation noise)
    // Method: add perturbation proportional to e³ (gives 3rd-order content)
    //         and e⁴ (gives 4th-order content), plus 2nd-order stress perturbation
    double eps2 = 0.005;   // 2nd-order perturbation magnitude (stress)
    double eps3 = 0.002;   // 3rd-order perturbation magnitude
    double eps4 = 0.001;   // 4th-order perturbation magnitude

    for (int q = 0; q < NQ; q++) {
        double ex = e[q][0], ey = e[q][1], ez = e[q][2];

        // 2nd-order: asymmetric stress perturbation
        double pert2 = eps2 * W[q] * (ex*ey + 0.5*ey*ez - 0.3*ex*ez);

        // 3rd-order: e_x·e_y·e_z type perturbation (should be removed by regularization)
        double pert3 = eps3 * W[q] * ex * ey * ez;

        // 4th-order: e_x²·e_y² type perturbation (should be removed by regularization)
        double pert4 = eps4 * W[q] * (ex*ex*ey*ey - 1.0/9.0);

        // Random-ish noise at all orders (simulating interpolation artifacts)
        double pert_noise = 1.0e-4 * W[q] * sin(3.7 * q + 0.5);

        f[q] += pert2 + pert3 + pert4 + pert_noise;
    }
}

// ── Compute arbitrary moment Σ f · eα^a · eβ^b · eγ^c ──
static double compute_moment(const double f[NQ], int a, int b, int c) {
    double m = 0.0;
    for (int q = 0; q < NQ; q++) {
        double val = f[q];
        for (int i = 0; i < a; i++) val *= e[q][0];
        for (int i = 0; i < b; i++) val *= e[q][1];
        for (int i = 0; i < c; i++) val *= e[q][2];
        m += val;
    }
    return m;
}

// ── Compute equilibrium moment Σ f^eq · eα^a · eβ^b · eγ^c ──
static double compute_eq_moment(double rho, double ux, double uy, double uz,
                                int a, int b, int c) {
    double m = 0.0;
    for (int q = 0; q < NQ; q++) {
        double feq = compute_feq(q, rho, ux, uy, uz);
        double val = feq;
        for (int i = 0; i < a; i++) val *= e[q][0];
        for (int i = 0; i < b; i++) val *= e[q][1];
        for (int i = 0; i < c; i++) val *= e[q][2];
        m += val;
    }
    return m;
}

// ============================================================================
// TESTS
// ============================================================================

int pass_count = 0, fail_count = 0;

void CHECK(bool cond, const char* name) {
    if (cond) {
        printf("  PASS: %s\n", name);
        pass_count++;
    } else {
        printf("  FAIL: %s\n", name);
        fail_count++;
    }
}

void CHECK_CLOSE(double a, double b, double tol, const char* name) {
    double err = fabs(a - b);
    double ref = fabs(b) > 1e-15 ? fabs(b) : 1.0;
    bool ok = err < tol || err / ref < tol;
    if (ok) {
        printf("  PASS: %s (err=%.2e)\n", name, err);
        pass_count++;
    } else {
        printf("  FAIL: %s (got=%.10e, expect=%.10e, err=%.2e)\n", name, a, b, err);
        fail_count++;
    }
}

int main() {
    printf("=== RLBM Pre-collision Regularization Unit Test ===\n");
    printf("Reference: Latt & Chopard (2006), Eq.6, 10, 11\n\n");

    // ── Build test distribution with known non-equilibrium content ──
    double f_orig[NQ], f_code[NQ], f_latt[NQ];
    double rho_build, ux_build, uy_build, uz_build;
    build_test_distribution(f_orig, rho_build, ux_build, uy_build, uz_build);

    // Compute ACTUAL ρ, u from the perturbed distribution
    // (perturbations shift the true moments away from the build values)
    // This is what the simulation code does: compute moments FROM f_streamed
    double rho = compute_moment(f_orig, 0, 0, 0);
    double jx  = compute_moment(f_orig, 1, 0, 0);
    double jy  = compute_moment(f_orig, 0, 1, 0);
    double jz  = compute_moment(f_orig, 0, 0, 1);
    double ux  = jx / rho;
    double uy  = jy / rho;
    double uz  = jz / rho;

    printf("  Actual moments from perturbed f:\n");
    printf("    ρ  = %.10f (build: %.10f, diff: %.2e)\n", rho, rho_build, rho-rho_build);
    printf("    ux = %.10f (build: %.10f)\n", ux, ux_build);
    printf("    uy = %.10f (build: %.10f)\n", uy, uy_build);
    printf("    uz = %.10f (build: %.10f)\n\n", uz, uz_build);

    // Save copies
    for (int q = 0; q < NQ; q++) {
        f_code[q] = f_orig[q];
        f_latt[q] = f_orig[q];
    }

    // ── Compute Π^neq from ORIGINAL distribution (before regularization) ──
    // Uses ACTUAL ρ, u (not build values) — same as simulation code
    double Pi_neq[3][3];
    compute_Pi_neq(f_orig, rho, ux, uy, uz, Pi_neq);

    printf("── Test 0: Verify test distribution has non-trivial content ──\n");
    {
        // 3rd-order moment should be non-zero before regularization
        double m_xyz = compute_moment(f_orig, 1, 1, 1);
        double m_xyz_eq = compute_eq_moment(rho, ux, uy, uz, 1, 1, 1);
        printf("  m_xyz (original)    = %.6e\n", m_xyz);
        printf("  m_xyz (equilibrium) = %.6e\n", m_xyz_eq);
        printf("  difference          = %.6e\n", fabs(m_xyz - m_xyz_eq));
        CHECK(fabs(m_xyz - m_xyz_eq) > 1e-8, "3rd-order moment has non-equilibrium content");

        // 4th-order moment
        double m_xxyy = compute_moment(f_orig, 2, 2, 0);
        double m_xxyy_eq = compute_eq_moment(rho, ux, uy, uz, 2, 2, 0);
        printf("  m_xxyy (original)    = %.6e\n", m_xxyy);
        printf("  m_xxyy (equilibrium) = %.6e\n", m_xxyy_eq);
        CHECK(fabs(m_xxyy - m_xxyy_eq) > 1e-8, "4th-order moment has non-equilibrium content");
    }

    // ── Apply regularizations ──
    // Method 1: Our code (from evolution_gilbm.h STEP 1.75)
    regularize_code(f_code, rho, ux, uy, uz);

    // Method 2: Latt Eq.10+11 directly
    for (int q = 0; q < NQ; q++) {
        double feq = compute_feq(q, rho, ux, uy, uz);
        double f1  = compute_f1_latt(q, Pi_neq);
        f_latt[q] = feq + f1;
    }

    // ================================================================
    printf("\n── Test 1: Code regularization matches Latt Eq.10+11 ──\n");
    // ================================================================
    {
        double max_err = 0.0;
        for (int q = 0; q < NQ; q++) {
            double err = fabs(f_code[q] - f_latt[q]);
            if (err > max_err) max_err = err;
        }
        printf("  max |f_code - f_latt| = %.2e\n", max_err);
        CHECK(max_err < 1e-14, "Code matches Latt Eq.10+11 (machine precision)");
    }

    // ================================================================
    printf("\n── Test 2: Moment preservation (ρ, j) ──\n");
    // ================================================================
    {
        // 0th moment: density
        double rho_orig = compute_moment(f_orig, 0, 0, 0);
        double rho_reg  = compute_moment(f_code, 0, 0, 0);
        CHECK_CLOSE(rho_reg, rho_orig, 1e-13, "ρ preserved (0th moment)");

        // 1st moments: momentum
        double jx_orig = compute_moment(f_orig, 1, 0, 0);
        double jx_reg  = compute_moment(f_code, 1, 0, 0);
        CHECK_CLOSE(jx_reg, jx_orig, 1e-13, "jx preserved (1st moment)");

        double jy_orig = compute_moment(f_orig, 0, 1, 0);
        double jy_reg  = compute_moment(f_code, 0, 1, 0);
        CHECK_CLOSE(jy_reg, jy_orig, 1e-13, "jy preserved (1st moment)");

        double jz_orig = compute_moment(f_orig, 0, 0, 1);
        double jz_reg  = compute_moment(f_code, 0, 0, 1);
        CHECK_CLOSE(jz_reg, jz_orig, 1e-13, "jz preserved (1st moment)");
    }

    // ================================================================
    printf("\n── Test 3: Stress tensor Π preserved (2nd moment) ──\n");
    // ================================================================
    {
        // All 6 independent components of Π_αβ
        const char* labels[6] = {"Π_xx", "Π_yy", "Π_zz", "Π_xy", "Π_xz", "Π_yz"};
        int aa[6] = {2,0,0,1,1,0};
        int bb[6] = {0,2,0,1,0,1};
        int cc[6] = {0,0,2,0,1,1};

        for (int s = 0; s < 6; s++) {
            double m_orig = compute_moment(f_orig, aa[s], bb[s], cc[s]);
            double m_reg  = compute_moment(f_code, aa[s], bb[s], cc[s]);
            char buf[64];
            snprintf(buf, sizeof(buf), "%s preserved", labels[s]);
            CHECK_CLOSE(m_reg, m_orig, 1e-13, buf);
        }
    }

    // ================================================================
    printf("\n── Test 4: 3rd-order moments → equilibrium values ──\n");
    // ================================================================
    {
        // All 10 independent 3rd-order moments: (3,0,0),(0,3,0),(0,0,3),
        //   (2,1,0),(2,0,1),(0,2,1),(1,2,0),(1,0,2),(0,1,2),(1,1,1)
        int third_a[10] = {3,0,0, 2,2,0,1,1,0, 1};
        int third_b[10] = {0,3,0, 1,0,2,2,0,1, 1};
        int third_c[10] = {0,0,3, 0,1,1,0,2,2, 1};
        const char* names[10] = {
            "m_300","m_030","m_003",
            "m_210","m_201","m_021","m_120","m_102","m_012",
            "m_111"
        };

        int pass3 = 0;
        for (int s = 0; s < 10; s++) {
            double m_reg = compute_moment(f_code, third_a[s], third_b[s], third_c[s]);
            double m_eq  = compute_eq_moment(rho, ux, uy, uz, third_a[s], third_b[s], third_c[s]);
            double m_orig = compute_moment(f_orig, third_a[s], third_b[s], third_c[s]);
            double diff_before = fabs(m_orig - m_eq);
            double diff_after  = fabs(m_reg  - m_eq);

            char buf[128];
            snprintf(buf, sizeof(buf), "%s: regularized → equilibrium", names[s]);
            CHECK_CLOSE(m_reg, m_eq, 1e-13, buf);
            if (diff_before > 1e-10) {
                printf("         (was %.2e off equilibrium, now %.2e)\n", diff_before, diff_after);
            }
        }
    }

    // ================================================================
    printf("\n── Test 5: 4th-order central moments: noise removed, f^(1) retained ──\n");
    // ================================================================
    // KEY INSIGHT (Latt & Chopard):
    //   f_reg = f^eq + f^(1)  where f^(1) is SOLELY determined by Π^neq.
    //   f^(1) DOES contribute to 4th-order central moments (via lattice tensors).
    //   This is NOT noise — it's a deterministic consequence of 2nd-order stress.
    //
    //   The original f has: κ₄ = κ₄^eq + κ₄(f^(1)) + κ₄(NOISE)
    //   After regularization:  κ₄ = κ₄^eq + κ₄(f^(1))   ← noise removed
    //
    //   We verify: regularized 4th central moments = those of (feq + f^(1)_Latt)
    {
        int fourth_a[6] = {4,0,0, 2,2,0};
        int fourth_b[6] = {0,4,0, 2,0,2};
        int fourth_c[6] = {0,0,4, 0,2,2};
        const char* names[6] = {"κ_xxxx","κ_yyyy","κ_zzzz","κ_xxyy","κ_xxzz","κ_yyzz"};

        for (int s = 0; s < 6; s++) {
            double cm_orig = 0.0, cm_reg = 0.0, cm_latt = 0.0, cm_eq = 0.0;
            for (int q = 0; q < NQ; q++) {
                double dx = e[q][0]-ux, dy = e[q][1]-uy, dz = e[q][2]-uz;
                double factor = 1.0;
                for (int i = 0; i < fourth_a[s]; i++) factor *= dx;
                for (int i = 0; i < fourth_b[s]; i++) factor *= dy;
                for (int i = 0; i < fourth_c[s]; i++) factor *= dz;
                cm_orig += f_orig[q] * factor;
                cm_reg  += f_code[q] * factor;
                cm_latt += f_latt[q] * factor;
                cm_eq   += compute_feq(q, rho, ux, uy, uz) * factor;
            }
            char buf[128];
            snprintf(buf, sizeof(buf), "%s: code matches Latt (feq+f^(1))", names[s]);
            CHECK_CLOSE(cm_reg, cm_latt, 1e-14, buf);
            printf("         eq=%.6e  f^(1)_contrib=%.2e  orig_noise=%.2e (removed)\n",
                   cm_eq, cm_latt - cm_eq, cm_orig - cm_latt);
        }
    }

    // ================================================================
    printf("\n── Test 6: Latt Eq.10 formula verification ──\n");
    // f^(1)_i = (t_i / 2cs⁴) · Q_iαβ · Π^neq_αβ
    // Verify component-by-component
    // ================================================================
    {
        printf("  Π^neq tensor (from original f, Latt Eq.6):\n");
        printf("    Π_xx = %+.6e\n", Pi_neq[0][0]);
        printf("    Π_yy = %+.6e\n", Pi_neq[1][1]);
        printf("    Π_zz = %+.6e\n", Pi_neq[2][2]);
        printf("    Π_xy = %+.6e\n", Pi_neq[0][1]);
        printf("    Π_xz = %+.6e\n", Pi_neq[0][2]);
        printf("    Π_yz = %+.6e\n", Pi_neq[1][2]);
        printf("    trace = %+.6e (should be small for incompressible)\n",
               Pi_neq[0][0]+Pi_neq[1][1]+Pi_neq[2][2]);

        // Verify symmetry
        CHECK_CLOSE(Pi_neq[0][1], Pi_neq[1][0], 1e-15, "Π^neq symmetric: xy=yx");
        CHECK_CLOSE(Pi_neq[0][2], Pi_neq[2][0], 1e-15, "Π^neq symmetric: xz=zx");
        CHECK_CLOSE(Pi_neq[1][2], Pi_neq[2][1], 1e-15, "Π^neq symmetric: yz=zy");

        // Verify f^(1) sums: Σ f^(1) = 0, Σ f^(1)·e = 0
        double sum_f1 = 0.0, jx_f1 = 0.0, jy_f1 = 0.0, jz_f1 = 0.0;
        for (int q = 0; q < NQ; q++) {
            double f1 = compute_f1_latt(q, Pi_neq);
            sum_f1 += f1;
            jx_f1 += f1 * e[q][0];
            jy_f1 += f1 * e[q][1];
            jz_f1 += f1 * e[q][2];
        }
        CHECK_CLOSE(sum_f1, 0.0, 1e-15, "Σ f^(1) = 0 (mass neutral)");
        CHECK_CLOSE(jx_f1, 0.0, 1e-15, "Σ f^(1)·ex = 0 (momentum neutral, x)");
        CHECK_CLOSE(jy_f1, 0.0, 1e-15, "Σ f^(1)·ey = 0 (momentum neutral, y)");
        CHECK_CLOSE(jz_f1, 0.0, 1e-15, "Σ f^(1)·ez = 0 (momentum neutral, z)");

        // Verify f^(1) reproduces Π^neq: Σ f^(1)·eα·eβ = Π^neq_αβ
        double Pi_from_f1[3][3] = {};
        for (int q = 0; q < NQ; q++) {
            double f1 = compute_f1_latt(q, Pi_neq);
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    Pi_from_f1[a][b] += f1 * e[q][a] * e[q][b];
        }
        CHECK_CLOSE(Pi_from_f1[0][0], Pi_neq[0][0], 1e-14, "Σ f^(1)·ex·ex = Π_xx");
        CHECK_CLOSE(Pi_from_f1[1][1], Pi_neq[1][1], 1e-14, "Σ f^(1)·ey·ey = Π_yy");
        CHECK_CLOSE(Pi_from_f1[2][2], Pi_neq[2][2], 1e-14, "Σ f^(1)·ez·ez = Π_zz");
        CHECK_CLOSE(Pi_from_f1[0][1], Pi_neq[0][1], 1e-14, "Σ f^(1)·ex·ey = Π_xy");
        CHECK_CLOSE(Pi_from_f1[0][2], Pi_neq[0][2], 1e-14, "Σ f^(1)·ex·ez = Π_xz");
        CHECK_CLOSE(Pi_from_f1[1][2], Pi_neq[1][2], 1e-14, "Σ f^(1)·ey·ez = Π_yz");
    }

    // ================================================================
    printf("\n── Test 7: Idempotency (regularizing twice = regularizing once) ──\n");
    // ================================================================
    {
        double f_twice[NQ];
        for (int q = 0; q < NQ; q++) f_twice[q] = f_code[q];
        regularize_code(f_twice, rho, ux, uy, uz);

        double max_diff = 0.0;
        for (int q = 0; q < NQ; q++) {
            double d = fabs(f_twice[q] - f_code[q]);
            if (d > max_diff) max_diff = d;
        }
        printf("  max |f_reg_twice - f_reg_once| = %.2e\n", max_diff);
        CHECK(max_diff < 1e-14, "Regularization is idempotent");
    }

    // ================================================================
    printf("\n── Test 8: Equilibrium input unchanged ──\n");
    // ================================================================
    {
        double f_eq[NQ], f_eq_reg[NQ];
        for (int q = 0; q < NQ; q++) {
            f_eq[q] = compute_feq(q, rho, ux, uy, uz);
            f_eq_reg[q] = f_eq[q];
        }
        regularize_code(f_eq_reg, rho, ux, uy, uz);

        double max_diff = 0.0;
        for (int q = 0; q < NQ; q++) {
            double d = fabs(f_eq_reg[q] - f_eq[q]);
            if (d > max_diff) max_diff = d;
        }
        printf("  max |f_eq_reg - f_eq| = %.2e\n", max_diff);
        CHECK(max_diff < 1e-14, "Pure equilibrium unchanged by regularization");
    }

    // ================================================================
    printf("\n══════════════════════════════════════\n");
    printf("Results: %d PASS, %d FAIL, %d total\n", pass_count, fail_count, pass_count + fail_count);
    printf("══════════════════════════════════════\n");

    return fail_count;
}
