#ifndef GILBM_DIAGNOSTIC_H
#define GILBM_DIAGNOSTIC_H

// Phase 1.5 Acceptance Diagnostic (Imamura 2005 GILBM)
// Call ONCE after initialization, before main time loop.
// Prints: (0) delta_xi validation, (1) delta_zeta statistics,
//         (2) interpolation spot-check, (3) C-E BC spot-check.
// All computation is host-side — no GPU kernel needed.

void DiagnoseGILBM_Phase1(
    const double *delta_xi_h,    // [19] ξ-direction displacement
    const double *delta_zeta_h,  // [19*NYD6*NZ6] ζ-direction displacement
    const double *dk_dz_h,
    const double *dk_dy_h,
    double **fh_p_local,     // host distribution pointers [19]
    int NYD6_local,
    int NZ6_local,
    int myid_local
) {
    if (myid_local != 0) return;

    // D3Q19 velocity set (host copy, matches GILBM_e in evolution_gilbm.h)
    double e[19][3] = {
        {0,0,0},
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };
    double W[19] = {
        1.0/3.0,
        1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
    };

    int sz = NYD6_local * NZ6_local;
    double dy_val = LY / (double)(NY6 - 7);

    printf("\n");
    printf("=============================================================\n");
    printf("  GILBM Phase 1.5 Acceptance Diagnostic (Rank 0, t=0)\n");
    printf("  NYD6=%d, NZ6=%d, NX6=%d, dt=%.6e, tau=%.4f\n",
           NYD6_local, NZ6_local, (int)NX6, dt, tau);
    printf("=============================================================\n");

    // ==================================================================
    // TEST 0: delta_xi validation (constant for uniform y)
    // ==================================================================
    printf("\n[Test 0] delta_xi validation: delta_xi[a] == dt*e_y[a]/dy\n");
    printf("  dy = %.10f, dt = %.10f\n", dy_val, dt);
    printf("  %5s  %5s  %14s  %14s  %10s\n",
           "alpha", "e_y", "delta_xi", "dt*e_y/dy", "error");

    double max_xi_err = 0.0;
    for (int alpha = 0; alpha < 19; alpha++) {
        double expected = dt * e[alpha][1] / dy_val;
        double err = fabs(delta_xi_h[alpha] - expected);
        if (err > max_xi_err) max_xi_err = err;
        printf("  %5d  %+4.0f  %+14.10e  %+14.10e  %10.2e\n",
               alpha, e[alpha][1], delta_xi_h[alpha], expected, err);
    }
    printf("\n  max|error| = %.2e  %s\n", max_xi_err,
           max_xi_err < 1e-15 ? "PASS" : "FAIL");

    // ==================================================================
    // TEST 1: delta_zeta range (min/max across all alpha, j, k)
    // ==================================================================
    printf("\n[Test 1] delta_zeta range (19 dirs x %d j-k points)\n", sz);

    double dk_gmin = 1e30, dk_gmax = -1e30;
    int gmin_a = 0, gmin_j = 0, gmin_k = 0;
    int gmax_a = 0, gmax_j = 0, gmax_k = 0;
    int nonzero = 0;

    for (int alpha = 0; alpha < 19; alpha++) {
        for (int j = 0; j < NYD6_local; j++) {
            for (int k = 2; k < NZ6_local - 2; k++) {
                double val = delta_zeta_h[alpha * sz + j * NZ6_local + k];
                if (val < dk_gmin) { dk_gmin = val; gmin_a = alpha; gmin_j = j; gmin_k = k; }
                if (val > dk_gmax) { dk_gmax = val; gmax_a = alpha; gmax_j = j; gmax_k = k; }
                if (val != 0.0) nonzero++;
            }
        }
    }

    printf("  min(delta_zeta) = %+.6e  at alpha=%2d, j=%3d, k=%3d\n",
           dk_gmin, gmin_a, gmin_j, gmin_k);
    printf("  max(delta_zeta) = %+.6e  at alpha=%2d, j=%3d, k=%3d\n",
           dk_gmax, gmax_a, gmax_j, gmax_k);
    printf("  max|delta_zeta| = %.6e\n", fabs(dk_gmin) > fabs(dk_gmax) ? fabs(dk_gmin) : fabs(dk_gmax));
    printf("  nonzero entries: %d / %d\n", nonzero, 19 * sz);

    // Per-direction breakdown (only directions with e_y or e_z != 0)
    printf("\n  Per-direction max|delta_zeta|:\n");
    printf("  %5s  %8s  %12s  %12s  %12s\n", "alpha", "e(y,z)", "max|dz|", "min(dz)", "max(dz)");
    for (int alpha = 1; alpha < 19; alpha++) {
        if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;
        double amax = 0.0, amin = 1e30, amx = -1e30;
        for (int j = 0; j < NYD6_local; j++) {
            for (int k = 2; k < NZ6_local - 2; k++) {
                double val = delta_zeta_h[alpha * sz + j * NZ6_local + k];
                if (fabs(val) > amax) amax = fabs(val);
                if (val < amin) amin = val;
                if (val > amx) amx = val;
            }
        }
        printf("  %5d  (%+.0f,%+.0f)  %12.6f  %+12.6f  %+12.6f\n",
               alpha, e[alpha][1], e[alpha][2], amax, amin, amx);
    }

    // CFL safety check
    double absmax = fabs(dk_gmin) > fabs(dk_gmax) ? fabs(dk_gmin) : fabs(dk_gmax);
    if (absmax > 2.0) {
        printf("\n  ** WARNING: max|delta_zeta|=%.3f > 2.0 cells! "
               "3-point stencil stretched.\n", absmax);
    }
    if (absmax > 3.0) {
        printf("  ** CRITICAL: max|delta_zeta|=%.3f > 3.0 cells! "
               "Quadratic interpolation insufficient!\n", absmax);
    }

    // Symmetry check: alpha=5 (e_z=+1) vs alpha=6 (e_z=-1)
    int jmid = NYD6_local / 2;
    int kmid = NZ6_local / 2;
    double dk5 = delta_zeta_h[5 * sz + jmid * NZ6_local + kmid];
    double dk6 = delta_zeta_h[6 * sz + jmid * NZ6_local + kmid];
    printf("\n  Symmetry check at j=%d, k=%d:\n", jmid, kmid);
    printf("    alpha=5 (e_z=+1): delta_zeta = %+.8e\n", dk5);
    printf("    alpha=6 (e_z=-1): delta_zeta = %+.8e\n", dk6);
    printf("    |dz5 + dz6| = %.2e  (should be ~0 for symmetric metric)\n",
           fabs(dk5 + dk6));

    // Spot-check: compare delta_zeta vs dt*e_tilde_zeta for one point
    int idx_jk_spot = jmid * NZ6_local + kmid;
    double dk_dz_spot = dk_dz_h[idx_jk_spot];
    double dk_dy_spot = dk_dy_h[idx_jk_spot];
    double etk5 = e[5][1] * dk_dy_spot + e[5][2] * dk_dz_spot;
    printf("    dt*e_tilde_zeta(alpha=5) = %.8e  (1st-order, no RK2)\n", dt * etk5);
    printf("    delta_zeta[5]            = %.8e  (RK2)\n", dk5);
    printf("    Difference (RK2 correction) = %.2e\n", fabs(dk5 - dt * etk5));

    // ==================================================================
    // TEST 2: Interpolation spot-check (host-side)
    // ==================================================================
    printf("\n[Test 2] Interpolation spot-check (equilibrium f, u=0, rho=1)\n");
    printf("  At t=0: f_alpha = w_alpha everywhere.\n");
    printf("  Interpolating at any upwind point should return w_alpha exactly.\n\n");

    int ti = (int)NX6 / 2;
    int tj = NYD6_local / 2;
    int tk = NZ6_local / 2;
    printf("  Test point: i=%d, j=%d, k=%d\n", ti, tj, tk);

    double dx_val = LX / (double)(NX6 - 7);
    double max_err = 0.0;

    printf("  %5s  %14s  %14s  %10s\n", "alpha", "interpolated", "expected(w_a)", "error");
    for (int alpha = 1; alpha < 19; alpha++) {
        int idx_jk = tj * NZ6_local + tk;

        double delta_i = dt * e[alpha][0] / dx_val;
        double delta_xi_val = delta_xi_h[alpha];
        double delta_zeta_val = delta_zeta_h[alpha * sz + idx_jk];

        double up_i = (double)ti - delta_i;
        double up_j = (double)tj - delta_xi_val;
        double up_k = (double)tk - delta_zeta_val;

        // Clamp (same as kernel)
        if (up_i < 1.0) up_i = 1.0;
        if (up_i > (double)(NX6 - 3)) up_i = (double)(NX6 - 3);
        if (up_j < 1.0) up_j = 1.0;
        if (up_j > (double)(NYD6_local - 3)) up_j = (double)(NYD6_local - 3);
        if (up_k < 2.0) up_k = 2.0;
        if (up_k > (double)(NZ6_local - 5)) up_k = (double)(NZ6_local - 5);

        // Host-side quadratic interpolation (replica of interpolate_quadratic_3d)
        int bi = (int)floor(up_i);
        int bj = (int)floor(up_j);
        int bk = (int)floor(up_k);
        double fi = up_i - (double)bi;
        double fj = up_j - (double)bj;
        double fk = up_k - (double)bk;

        double ai[3], aj[3], ak[3];
        ai[0] = 0.5*(fi-1.0)*(fi-2.0); ai[1] = -fi*(fi-2.0); ai[2] = 0.5*fi*(fi-1.0);
        aj[0] = 0.5*(fj-1.0)*(fj-2.0); aj[1] = -fj*(fj-2.0); aj[2] = 0.5*fj*(fj-1.0);
        ak[0] = 0.5*(fk-1.0)*(fk-2.0); ak[1] = -fk*(fk-2.0); ak[2] = 0.5*fk*(fk-1.0);

        double result = 0.0;
        for (int n = 0; n < 3; n++) {
            for (int m = 0; m < 3; m++) {
                double wjk = aj[m] * ak[n];
                for (int l = 0; l < 3; l++) {
                    int idx = (bj+m)*NZ6_local*NX6 + (bk+n)*NX6 + (bi+l);
                    result += ai[l] * wjk * fh_p_local[alpha][idx];
                }
            }
        }

        double err = fabs(result - W[alpha]);
        if (err > max_err) max_err = err;

        printf("  %5d  %14.10e  %14.10e  %10.2e\n", alpha, result, W[alpha], err);
    }

    printf("\n  max interpolation error (18 dirs): %.2e\n", max_err);
    if (max_err > 1e-12) {
        printf("  ** WARNING: Error > 1e-12 on uniform equilibrium!\n");
        printf("     Possible cause: array layout mismatch or uninitialized ghost cells.\n");
    } else {
        printf("  PASS: Interpolation reproduces constant field exactly.\n");
    }

    // ==================================================================
    // TEST 3: Chapman-Enskog BC spot-check (bottom wall k=2)
    // ==================================================================
    printf("\n[Test 3] Chapman-Enskog BC spot-check (bottom wall, k=2)\n");
    printf("  At t=0: u=0 everywhere -> du/dk=0 -> C_alpha=0\n");
    printf("  Expected: f_CE = w_alpha * rho_wall\n\n");

    int bc_i = (int)NX6 / 2;
    int bc_j = NYD6_local / 2;
    int bc_k = 2;  // bottom wall
    int bc_idx_jk = bc_j * NZ6_local + bc_k;
    double bc_dk_dy = dk_dy_h[bc_idx_jk];
    double bc_dk_dz = dk_dz_h[bc_idx_jk];

    printf("  Wall point: i=%d, j=%d, k=%d\n", bc_i, bc_j, bc_k);
    printf("  Metric: dk/dy = %+.6e, dk/dz = %+.6e\n", bc_dk_dy, bc_dk_dz);

    // Compute macroscopic at k=3 and k=4
    int idx3 = bc_j * NX6 * NZ6_local + 3 * NX6 + bc_i;
    int idx4 = bc_j * NX6 * NZ6_local + 4 * NX6 + bc_i;

    double rho3 = 0.0, rho4 = 0.0;
    double f3[19], f4[19];
    for (int a = 0; a < 19; a++) { f3[a] = fh_p_local[a][idx3]; rho3 += f3[a]; }
    for (int a = 0; a < 19; a++) { f4[a] = fh_p_local[a][idx4]; rho4 += f4[a]; }

    double ux3 = (f3[1]+f3[7]+f3[9]+f3[11]+f3[13] - (f3[2]+f3[8]+f3[10]+f3[12]+f3[14])) / rho3;
    double uy3 = (f3[3]+f3[7]+f3[8]+f3[15]+f3[17] - (f3[4]+f3[9]+f3[10]+f3[16]+f3[18])) / rho3;
    double uz3 = (f3[5]+f3[11]+f3[12]+f3[15]+f3[16] - (f3[6]+f3[13]+f3[14]+f3[17]+f3[18])) / rho3;

    double ux4 = (f4[1]+f4[7]+f4[9]+f4[11]+f4[13] - (f4[2]+f4[8]+f4[10]+f4[12]+f4[14])) / rho4;
    double uy4 = (f4[3]+f4[7]+f4[8]+f4[15]+f4[17] - (f4[4]+f4[9]+f4[10]+f4[16]+f4[18])) / rho4;
    double uz4 = (f4[5]+f4[11]+f4[12]+f4[15]+f4[16] - (f4[6]+f4[13]+f4[14]+f4[17]+f4[18])) / rho4;

    double du_x_dk = (4.0*ux3 - ux4) / 2.0;
    double du_y_dk = (4.0*uy3 - uy4) / 2.0;
    double du_z_dk = (4.0*uz3 - uz4) / 2.0;

    printf("  rho_wall (from k=3) = %.10f\n", rho3);
    printf("  du/dk at wall: (%.6e, %.6e, %.6e)\n", du_x_dk, du_y_dk, du_z_dk);
    printf("  [At t=0 with u=0 init, du/dk should be ~0]\n");

    double omega = 1.0 / tau;

    printf("\n  C-E BC per direction needing BC at bottom wall:\n");
    printf("  %5s  %6s  %6s  %12s  %12s  %12s  %12s\n",
           "alpha", "e_y", "e_z", "e_tilde_zeta", "C_alpha", "f_CE", "w_alpha");

    int bc_count = 0;
    double sum_f_CE = 0.0;
    for (int alpha = 1; alpha < 19; alpha++) {
        double e_tilde_zeta = e[alpha][1] * bc_dk_dy + e[alpha][2] * bc_dk_dz;
        if (e_tilde_zeta <= 0.0) continue;  // doesn't need BC at bottom wall
        bc_count++;

        // Host-side replica of ChapmanEnskogBC
        double ex = e[alpha][0], ey = e[alpha][1], ez = e[alpha][2];
        double C_alpha = 0.0;
        C_alpha += du_x_dk * ((3.0*ex*ey)*bc_dk_dy + (3.0*ex*ez)*bc_dk_dz);
        C_alpha += du_y_dk * ((3.0*ey*ey - 1.0)*bc_dk_dy + (3.0*ey*ez)*bc_dk_dz);
        C_alpha += du_z_dk * ((3.0*ez*ey)*bc_dk_dy + (3.0*ez*ez - 1.0)*bc_dk_dz);
        C_alpha *= -omega * dt;

        double f_CE = W[alpha] * rho3 * (1.0 + C_alpha);
        sum_f_CE += f_CE;

        printf("  %5d  %+5.0f  %+5.0f  %+12.4f  %+12.6e  %12.8f  %12.8f\n",
               alpha, ey, ez, e_tilde_zeta, C_alpha, f_CE, W[alpha]);
    }

    printf("\n  Directions needing BC: %d / 18\n", bc_count);
    printf("  Sum(f_CE) across BC directions: %.10f\n", sum_f_CE);
    printf("  Sum(w_alpha) for same directions: ");
    double sum_w = 0.0;
    for (int alpha = 1; alpha < 19; alpha++) {
        double e_tilde_zeta = e[alpha][1] * bc_dk_dy + e[alpha][2] * bc_dk_dz;
        if (e_tilde_zeta > 0.0) sum_w += W[alpha];
    }
    printf("%.10f\n", sum_w);
    printf("  Difference: %.2e  (should be ~0 at t=0)\n", fabs(sum_f_CE - sum_w * rho3));

    // Summary
    printf("\n=============================================================\n");
    printf("  Phase 1.5 Acceptance Summary:\n");
    printf("  [0] max|delta_xi error| = %.2e  %s\n", max_xi_err,
           max_xi_err < 1e-15 ? "PASS" : "FAIL");
    printf("  [1] max|delta_zeta| = %.4f cells  %s\n", absmax,
           absmax <= 2.5 ? "OK" : (absmax <= 3.0 ? "WARNING" : "CRITICAL"));
    printf("  [2] Interpolation error = %.2e  %s\n", max_err,
           max_err < 1e-12 ? "PASS" : "FAIL");
    printf("  [3] C-E BC consistency = %.2e  %s\n",
           fabs(sum_f_CE - sum_w * rho3),
           fabs(sum_f_CE - sum_w * rho3) < 1e-12 ? "PASS" : "FAIL");
    printf("=============================================================\n\n");
}

// ==============================================================================
// Phase 2: Departure Point CFL Validation
// ==============================================================================
// Checks whether |δζ| < 1 at first interior nodes k=3 (bottom) and k=NZ6-4 (top).
// In our GILBM framework ζ = k (integer index), so Δζ = 1 between adjacent points.
// CFL_ζ = |δζ| / 1 = |δζ|.
// If CFL_ζ ≥ 1.0, the departure point crosses into the solid/ghost region.
//
// Mathematical background:
//   tanhFunction(L, minSize, a, j=0, N) = minSize/2  (structural identity)
//   → z[k=3] - z[k=2] = minSize/2
//   → dz_dk(k=3) = (z[k=4]-z[k=2])/2 = 3·minSize/4  (central difference)
//   → dk_dz(k=3) = 4/(3·minSize)
//   → CFL_ζ = dt · dk_dz = minSize · 4/(3·minSize) = 4/3 ≈ 1.333
//   This is INDEPENDENT of CFL parameter or NZ — it's structural.

bool ValidateDepartureCFL(
    const double *delta_zeta_h,  // [19*NYD6*NZ6] precomputed RK2 displacement
    const double *dk_dy_h,       // [NYD6*NZ6] metric terms
    const double *dk_dz_h,       // [NYD6*NZ6] metric terms
    int NYD6_local,
    int NZ6_local,
    int myid_local
) {
    if (myid_local != 0) return true;

    double e[19][3] = {
        {0,0,0},
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };

    int sz = NYD6_local * NZ6_local;
    bool valid = true;

    printf("\n");
    printf("=============================================================\n");
    printf("  Phase 2: Departure Point CFL Validation (Rank 0)\n");
    printf("  dt=%.6e, minSize=%.6e, NZ6=%d, NYD6=%d\n",
           dt, (double)minSize, NZ6_local, NYD6_local);
    printf("  Theoretical CFL at flat floor: dt·dk_dz(k=3) = 4/3 ≈ 1.333\n");
    printf("=============================================================\n");

    // ====== Bottom wall: first interior k=3 ======
    printf("\n[Bottom Wall] CFL check at k=3 (first interior, k=2 is wall)\n");

    double bot_raw_max = 0.0, bot_eff_max = 0.0;
    int bot_raw_j = -1, bot_raw_a = -1;
    int bot_eff_j = -1, bot_eff_a = -1;
    int bot_violations = 0;

    for (int j = 3; j < NYD6_local - 3; j++) {
        int idx3 = j * NZ6_local + 3;
        double dk_dy_val = dk_dy_h[idx3];
        double dk_dz_val = dk_dz_h[idx3];

        for (int alpha = 1; alpha < 19; alpha++) {
            if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;

            // Raw CFL: from metric terms at current point (no RK2)
            double e_tilde_zeta = e[alpha][1] * dk_dy_val + e[alpha][2] * dk_dz_val;
            double raw_cfl = fabs(e_tilde_zeta) * dt;

            // Effective CFL: from precomputed delta_zeta (with RK2 midpoint)
            double eff_cfl = fabs(delta_zeta_h[alpha * sz + idx3]);

            if (raw_cfl > bot_raw_max) {
                bot_raw_max = raw_cfl; bot_raw_j = j; bot_raw_a = alpha;
            }
            if (eff_cfl > bot_eff_max) {
                bot_eff_max = eff_cfl; bot_eff_j = j; bot_eff_a = alpha;
            }

            if (eff_cfl >= 1.0) {
                bot_violations++;
                if (bot_violations <= 5) {
                    printf("  [VIOLATION] j=%d, alpha=%2d (e_y=%+.0f,e_z=%+.0f): "
                           "CFL_raw=%.4f, CFL_eff=%.4f, delta_zeta=%+.6f\n",
                           j, alpha, e[alpha][1], e[alpha][2],
                           raw_cfl, eff_cfl, delta_zeta_h[alpha * sz + idx3]);
                }
            }
        }
    }
    if (bot_violations > 5) {
        printf("  ... and %d more violations (total: %d)\n",
               bot_violations - 5, bot_violations);
    }

    printf("\n  Max raw  CFL (metric-based): %.6f at j=%d, alpha=%d\n",
           bot_raw_max, bot_raw_j, bot_raw_a);
    printf("  Max eff  CFL (RK2 delta_z):  %.6f at j=%d, alpha=%d\n",
           bot_eff_max, bot_eff_j, bot_eff_a);
    printf("  RK2 correction: %.2e (raw - eff at worst)\n",
           bot_raw_max - bot_eff_max);

    // ====== Top wall: first interior k=NZ6-4 ======
    int k_top = NZ6_local - 4;
    printf("\n[Top Wall] CFL check at k=%d (first interior, k=%d is wall)\n",
           k_top, NZ6_local - 3);

    double top_raw_max = 0.0, top_eff_max = 0.0;
    int top_raw_j = -1, top_raw_a = -1;
    int top_eff_j = -1, top_eff_a = -1;
    int top_violations = 0;

    for (int j = 3; j < NYD6_local - 3; j++) {
        int idx_top = j * NZ6_local + k_top;
        double dk_dy_val = dk_dy_h[idx_top];
        double dk_dz_val = dk_dz_h[idx_top];

        for (int alpha = 1; alpha < 19; alpha++) {
            if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;

            double e_tilde_zeta = e[alpha][1] * dk_dy_val + e[alpha][2] * dk_dz_val;
            double raw_cfl = fabs(e_tilde_zeta) * dt;
            double eff_cfl = fabs(delta_zeta_h[alpha * sz + idx_top]);

            if (raw_cfl > top_raw_max) {
                top_raw_max = raw_cfl; top_raw_j = j; top_raw_a = alpha;
            }
            if (eff_cfl > top_eff_max) {
                top_eff_max = eff_cfl; top_eff_j = j; top_eff_a = alpha;
            }

            if (eff_cfl >= 1.0) {
                top_violations++;
                if (top_violations <= 5) {
                    printf("  [VIOLATION] j=%d, alpha=%2d (e_y=%+.0f,e_z=%+.0f): "
                           "CFL_raw=%.4f, CFL_eff=%.4f, delta_zeta=%+.6f\n",
                           j, alpha, e[alpha][1], e[alpha][2],
                           raw_cfl, eff_cfl, delta_zeta_h[alpha * sz + idx_top]);
                }
            }
        }
    }
    if (top_violations > 5) {
        printf("  ... and %d more violations (total: %d)\n",
               top_violations - 5, top_violations);
    }

    printf("\n  Max raw  CFL (metric-based): %.6f at j=%d, alpha=%d\n",
           top_raw_max, top_raw_j, top_raw_a);
    printf("  Max eff  CFL (RK2 delta_z):  %.6f at j=%d, alpha=%d\n",
           top_eff_max, top_eff_j, top_eff_a);

    // ====== Per-j profile (condensed) ======
    printf("\n[Per-j CFL Profile] (every 4th j, bottom k=3)\n");
    printf("  %5s  %12s  %12s  %6s\n", "j", "max_raw_CFL", "max_eff_CFL", "status");
    for (int j = 3; j < NYD6_local - 3; j += 4) {
        int idx3 = j * NZ6_local + 3;
        double j_raw_max = 0.0, j_eff_max = 0.0;
        for (int alpha = 1; alpha < 19; alpha++) {
            if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;
            double e_tilde = e[alpha][1] * dk_dy_h[idx3] + e[alpha][2] * dk_dz_h[idx3];
            double rc = fabs(e_tilde) * dt;
            double ec = fabs(delta_zeta_h[alpha * sz + idx3]);
            if (rc > j_raw_max) j_raw_max = rc;
            if (ec > j_eff_max) j_eff_max = ec;
        }
        const char *status = j_eff_max < 0.8 ? "PASS" :
                             (j_eff_max < 1.0 ? "WARN" : "FAIL");
        printf("  %5d  %12.6f  %12.6f  %6s\n", j, j_raw_max, j_eff_max, status);
    }

    // ====== Summary ======
    double overall_max = bot_eff_max > top_eff_max ? bot_eff_max : top_eff_max;
    int total_violations = bot_violations + top_violations;

    if (overall_max >= 1.0) valid = false;

    printf("\n=============================================================\n");
    printf("  Phase 2 CFL Summary:\n");
    printf("  [Bottom] max CFL_zeta = %.4f at j=%d, alpha=%d  %s\n",
           bot_eff_max, bot_eff_j, bot_eff_a,
           bot_eff_max < 0.8 ? "PASS" : (bot_eff_max < 1.0 ? "WARNING" : "FAIL"));
    printf("  [Top]    max CFL_zeta = %.4f at j=%d, alpha=%d  %s\n",
           top_eff_max, top_eff_j, top_eff_a,
           top_eff_max < 0.8 ? "PASS" : (top_eff_max < 1.0 ? "WARNING" : "FAIL"));
    printf("  Total violations (CFL >= 1.0): %d\n", total_violations);

    if (!valid) {
        printf("\n  *** CFL VIOLATION DETECTED ***\n");
        printf("  Root cause: tanhFunction(j=0) = minSize/2 → z[3]-z[2] = minSize/2\n");
        printf("  With dt = minSize: CFL = dt · 4/(3·minSize) = 4/3 ≈ 1.333\n");
        printf("  This is STRUCTURAL — independent of CFL parameter or NZ.\n");
        printf("  Remedies:\n");
        printf("    A. Decouple dt: dt = timeCFL * minSize (timeCFL < 3/4)\n");
        printf("    B. Extend C-E BC to k=3 for violating directions\n");
        printf("    C. Modify tanh stretching so z[3]-z[2] = minSize\n");
    }
    printf("=============================================================\n\n");

    return valid;
}

#endif
