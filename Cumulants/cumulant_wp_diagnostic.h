// ================================================================
// cumulant_wp_diagnostic.h
// D3Q27 Cumulant-WP Pre-computation Singularity Diagnostic (HOST-SIDE)
//
// Purpose: Before simulation, compute all relaxation rates and check singularities
//   INPUT:  Re, Uref, dt_global, omega2 (from main.cu)
//   OUTPUT: Full diagnostic report (printf)
//
// Usage:
//   (A) Call in main.cu after omega_global is computed:
//       CumulantWP_DiagnoseOmega(Re, Uref, dt_global, omega2, myid);
//
//   (B) Standalone CLI tool (test_cumulant_wp_diagnostic.cpp):
//       cin >> Re >> Uref >> omega2
//       dt_global from precompute.h ComputeGlobalTimeStep
//
// Singularity overview (omega2 = 1.0):
//   w1: input parameter, determined by {Re, Uref, Jacobian->dt_global}, no singularity
//       w1 = 1/tau, tau = 3*nu/dt + 0.5, nu = Uref/Re
//       Physics guarantee: w1 in (0, 2)
//   w2: user-specified (default 1.0, bulk viscosity relaxation rate)
//   w3 (Eq.14): den3 = (9*w2-16)*w1^2 + (40-26*w2)*w1 + 8*w2
//       Roots: w1 ~ 2.46, -0.46 -> always outside (0,2), SAFE
//       Proof: den3(0)=8*w2>0, den3(2)=8*(2-w2)>0, opens downward -> always positive
//   w4 (Eq.15): den4 = 9*w2*w1^2 - (42*w2+8)*w1 + 56*w2
//       Roots: w1 = [(42*w2+8) - sqrt((42*w2+8)^2 - 4*9*w2*56*w2)] / (18*w2)
//       w2=1: w1 = 14/9 ~ 1.5556 <- inside (0,2), the only dangerous singularity
//   w5 (Eq.16): den5 is a cubic polynomial in w1
//       w2 in [0.5, 1.5]: no roots in (0,2), SAFE
//   A,B (Eq.17-18): denom = (w1-w2)*(w2*(2+3*w1) - 8*w1)
//       Root 1: w1 = w2 (inside (0,2) when w2=1)
//       Root 2: w1 = 2*w2/(8-3*w2) (~ 0.4 when w2=1)
//       Fallback: A=B=0 (degenerates to AO 4th-order equilibrium), no blow-up
//
// Stable intervals:
//   w3, w4, w5: open interval (0, 2) -- endpoints unusable
//     w=0 -> cumulant not relaxed -> exponential blow-up
//     w=2 -> over-relaxation boundary -> oscillatory instability
//   w6~w10: closed interval [0, 2] -- endpoints usable (higher-order insensitive)
//
// References:
//   [GR22] Gehrke & Rung, Int. J. Numer. Meth. Fluids 94, 1111-1154, 2022
// ================================================================
#ifndef CUMULANT_WP_DIAGNOSTIC_H
#define CUMULANT_WP_DIAGNOSTIC_H

#include <cstdio>
#include <cmath>
#include <cstdlib>

// ================================================================
// Core computation: given w1, w2, compute all relaxation rates + singularity info
// ================================================================
struct CumWP_OmegaReport {
    // Input
    double w1, w2;
    double tau, dt_global, niu;
    int    Re;
    double Uref;

    // Eq.14: w3
    double w3_raw, w3_used, den3;
    double w3_sing;          // singularity position (-1 means none in [0,2])
    double w3_sing_dist;     // distance to singularity

    // Eq.15: w4
    double w4_raw, w4_used, den4;
    double w4_sing;
    double w4_sing_dist;

    // Eq.16: w5
    double w5_raw, w5_used, den5;
    double w5_sing;          // numerically solved (-1 means none in [0,2])
    double w5_sing_dist;

    // Eq.17-18: A, B
    double A, B, denom_AB;
    double AB_sing1, AB_sing2;  // two roots
    double AB_min_dist;

    // Overall verdict
    int any_fallback;        // whether any omega was fallback-ed
    int any_danger;          // whether any omega is within SAFE_MARGIN of singularity
};

static const double CUM_DIAG_SAFE_MARGIN = 0.15;
static const double CUM_DIAG_DEN_EPS     = 1.0e-10;

// ================================================================
// Compute function (pure computation, no printing)
// ================================================================
static CumWP_OmegaReport CumulantWP_ComputeReport(
    int Re_in, double Uref_in, double dt_global_in, double omega2_in)
{
    CumWP_OmegaReport r;
    r.Re = Re_in;
    r.Uref = Uref_in;
    r.dt_global = dt_global_in;
    r.w2 = omega2_in;
    r.niu = Uref_in / (double)Re_in;
    r.tau = 3.0 * r.niu / dt_global_in + 0.5;
    r.w1 = 1.0 / r.tau;

    const double w1 = r.w1;
    const double w2 = r.w2;

    // -- w3 (Eq.14) ---------------------------------------------------
    double num3 = 8.0*(w1-2.0)*(w2*(3.0*w1-1.0)-5.0*w1);
    r.den3 = 8.0*(5.0-2.0*w1)*w1 + w2*(8.0+w1*(9.0*w1-26.0));
    r.w3_raw = (fabs(r.den3) > CUM_DIAG_DEN_EPS) ? num3/r.den3 : 1.0;
    r.w3_used = (r.w3_raw >= 0.0 && r.w3_raw <= 2.0) ? r.w3_raw : 1.0;

    // w3 singularity: (9*w2-16)*w1^2 + (40-26*w2)*w1 + 8*w2 = 0
    r.w3_sing = -1.0;
    {
        double a = 9.0*w2-16.0, b = 40.0-26.0*w2, c = 8.0*w2;
        double disc = b*b - 4.0*a*c;
        if (disc >= 0.0 && fabs(a) > CUM_DIAG_DEN_EPS) {
            double r1 = (-b-sqrt(disc))/(2.0*a);
            double r2 = (-b+sqrt(disc))/(2.0*a);
            if (r1 > 0.0 && r1 < 2.0) r.w3_sing = r1;
            else if (r2 > 0.0 && r2 < 2.0) r.w3_sing = r2;
        }
    }
    r.w3_sing_dist = (r.w3_sing > 0.0) ? fabs(w1 - r.w3_sing) : 99.0;

    // -- w4 (Eq.15) ---------------------------------------------------
    double num4 = 8.0*(w1-2.0)*(w1+w2*(3.0*w1-7.0));
    r.den4 = w2*(56.0-42.0*w1+9.0*w1*w1) - 8.0*w1;
    r.w4_raw = (fabs(r.den4) > CUM_DIAG_DEN_EPS) ? num4/r.den4 : 1.0;
    r.w4_used = (r.w4_raw >= 0.0 && r.w4_raw <= 2.0) ? r.w4_raw : 1.0;

    // w4 singularity: 9*w2*w1^2 - (42*w2+8)*w1 + 56*w2 = 0
    r.w4_sing = -1.0;
    {
        double a = 9.0*w2;
        double b = -(42.0*w2+8.0);
        double c = 56.0*w2;
        double disc = b*b - 4.0*a*c;
        if (disc >= 0.0) {
            double r1 = (-b-sqrt(disc))/(2.0*a);
            double r2 = (-b+sqrt(disc))/(2.0*a);
            if (r1 > 0.0 && r1 < 2.0) r.w4_sing = r1;
            else if (r2 > 0.0 && r2 < 2.0) r.w4_sing = r2;
        }
    }
    r.w4_sing_dist = (r.w4_sing > 0.0) ? fabs(w1 - r.w4_sing) : 99.0;

    // -- w5 (Eq.16) ---------------------------------------------------
    double num5 = 24.0*(w1-2.0)*(4.0*w1*w1
                + w1*w2*(18.0-13.0*w1)
                + w2*w2*(2.0+w1*(6.0*w1-11.0)));
    r.den5 = 16.0*w1*w1*(w1-6.0)
           - 2.0*w1*w2*(216.0+5.0*w1*(9.0*w1-46.0))
           + w2*w2*(w1*(3.0*w1-10.0)*(15.0*w1-28.0)-48.0);
    r.w5_raw = (fabs(r.den5) > CUM_DIAG_DEN_EPS) ? num5/r.den5 : 1.0;
    r.w5_used = (r.w5_raw >= 0.0 && r.w5_raw <= 2.0) ? r.w5_raw : 1.0;

    // w5 singularity: cubic polynomial, numerical scan in (0, 2)
    r.w5_sing = -1.0;
    {
        const int N = 10000;
        double prev = 0.0;
        for (int i = 0; i <= N; i++) {
            double x = 0.001 + 1.998 * (double)i / (double)N;
            double d5 = 16.0*x*x*(x-6.0)
                      - 2.0*x*w2*(216.0+5.0*x*(9.0*x-46.0))
                      + w2*w2*(x*(3.0*x-10.0)*(15.0*x-28.0)-48.0);
            if (i > 0 && prev * d5 < 0.0) {
                // linear interpolation to find root
                double x0 = 0.001 + 1.998*(double)(i-1)/(double)N;
                double root = x0 - prev*(x-x0)/(d5-prev);
                if (root > 0.0 && root < 2.0) {
                    r.w5_sing = root;
                    break;
                }
            }
            prev = d5;
        }
    }
    r.w5_sing_dist = (r.w5_sing > 0.0) ? fabs(w1 - r.w5_sing) : 99.0;

    // -- A, B (Eq.17-18) ----------------------------------------------
    r.denom_AB = (w1-w2) * (w2*(2.0+3.0*w1) - 8.0*w1);
    if (fabs(r.denom_AB) > CUM_DIAG_DEN_EPS) {
        r.A = (4.0*w1*w1 + 2.0*w1*w2*(w1-6.0)
             + w2*w2*(w1*(10.0-3.0*w1)-4.0)) / r.denom_AB;
        r.B = (4.0*w1*w2*(9.0*w1-16.0) - 4.0*w1*w1
             - 2.0*w2*w2*(2.0+9.0*w1*(w1-2.0))) / (3.0*r.denom_AB);
    } else {
        r.A = 0.0;
        r.B = 0.0;
    }
    r.AB_sing1 = w2;   // ω₁ = ω₂
    r.AB_sing2 = (fabs(8.0-3.0*w2) > CUM_DIAG_DEN_EPS)
               ? 2.0*w2/(8.0-3.0*w2) : -1.0;
    double d1 = fabs(w1 - r.AB_sing1);
    double d2 = (r.AB_sing2 > 0.0 && r.AB_sing2 < 2.0)
              ? fabs(w1 - r.AB_sing2) : 99.0;
    r.AB_min_dist = (d1 < d2) ? d1 : d2;

    // -- Overall verdict -----------------------------------------------
    r.any_fallback = (r.w3_raw < 0.0 || r.w3_raw > 2.0)
                   || (r.w4_raw < 0.0 || r.w4_raw > 2.0)
                   || (r.w5_raw < 0.0 || r.w5_raw > 2.0)
                   || (fabs(r.denom_AB) <= CUM_DIAG_DEN_EPS);
    r.any_danger = (r.w3_sing_dist < CUM_DIAG_SAFE_MARGIN)
                 || (r.w4_sing_dist < CUM_DIAG_SAFE_MARGIN)
                 || (r.w5_sing_dist < CUM_DIAG_SAFE_MARGIN)
                 || (r.AB_min_dist  < CUM_DIAG_SAFE_MARGIN);

    return r;
}

// ================================================================
// Formatted output (caller: main.cu or standalone CLI)
// ================================================================
static void CumulantWP_PrintReport(const CumWP_OmegaReport &r)
{
    printf("\n");
    printf("  +--------------------------------------------------------------------+\n");
    printf("  |        Cumulant-WP Pre-Simulation Singularity Diagnostic           |\n");
    printf("  +--------------------------------------------------------------------+\n");
    printf("  |  Input Parameters:                                                 |\n");
    printf("  |    Re        = %d\n", r.Re);
    printf("  |    Uref      = %.6f\n", r.Uref);
    printf("  |    niu       = Uref/Re = %.6e\n", r.niu);
    printf("  |    dt_global = %.6e  (from Jacobian / Imamura GTS)\n", r.dt_global);
    printf("  |    omega2    = %.4f  (bulk viscosity, user-specified)\n", r.w2);
    printf("  +--------------------------------------------------------------------+\n");
    printf("  |  Derived:                                                          |\n");
    printf("  |    tau   = 3*niu/dt + 0.5 = %.6f\n", r.tau);
    printf("  |    omega1 = 1/tau         = %.6f    (shear relaxation rate)\n", r.w1);
    printf("  |    omega1 range: (0, 2), guaranteed by physics                     |\n");
    printf("  +--------------------------------------------------------------------+\n");

    // -- w3 --
    printf("  |  omega3 (Eq.14, symmetric 3rd-order cumulant):                     |\n");
    printf("  |    raw = %+10.6f   used = %8.6f   den = %+12.4f\n", r.w3_raw, r.w3_used, r.den3);
    if (r.w3_sing > 0.0) {
        const char *tag = (r.w3_sing_dist < CUM_DIAG_SAFE_MARGIN) ? "DANGER" : "OK";
        printf("  |    singularity at w1 = %.4f (dist = %.4f)  [%s]\n", r.w3_sing, r.w3_sing_dist, tag);
    } else {
        printf("  |    singularity: NONE in (0,2)  [SAFE - proven: den3>0 for all w1]\n");
    }
    if (r.w3_raw < 0.0 || r.w3_raw > 2.0)
        printf("  |    >>> FALLBACK: omega3 out of [0,2] -> AO fallback = 1.0\n");
    else
        printf("  |    >>> STATUS: omega3 in stable range (0,2)  [OK]\n");

    printf("  |                                                                    |\n");

    // -- w4 --
    printf("  |  omega4 (Eq.15, antisymmetric 3rd-order cumulant):                 |\n");
    printf("  |    raw = %+10.6f   used = %8.6f   den = %+12.4f\n", r.w4_raw, r.w4_used, r.den4);
    if (r.w4_sing > 0.0) {
        const char *tag = (r.w4_sing_dist < CUM_DIAG_SAFE_MARGIN) ? "DANGER" : "OK";
        printf("  |    singularity at w1 = %.4f (dist = %.4f)  [%s]\n", r.w4_sing, r.w4_sing_dist, tag);
        if (r.w4_sing_dist < CUM_DIAG_SAFE_MARGIN) {
            printf("  |    *** WARNING: omega1 = %.4f is dangerously close! ***\n", r.w1);
            printf("  |    *** den4 = %.4f -> omega4_raw = %.4f (divergent) ***\n", r.den4, r.w4_raw);
        }
    } else {
        printf("  |    singularity: NONE in (0,2)  [SAFE]\n");
    }
    if (r.w4_raw < 0.0 || r.w4_raw > 2.0)
        printf("  |    >>> FALLBACK: omega4 out of [0,2] -> AO fallback = 1.0\n");
    else
        printf("  |    >>> STATUS: omega4 in stable range (0,2)  [OK]\n");

    printf("  |                                                                    |\n");

    // -- w5 --
    printf("  |  omega5 (Eq.16, C111 cumulant):                                    |\n");
    printf("  |    raw = %+10.6f   used = %8.6f   den = %+12.4f\n", r.w5_raw, r.w5_used, r.den5);
    if (r.w5_sing > 0.0) {
        const char *tag = (r.w5_sing_dist < CUM_DIAG_SAFE_MARGIN) ? "DANGER" : "OK";
        printf("  |    singularity at w1 = %.4f (dist = %.4f)  [%s]\n", r.w5_sing, r.w5_sing_dist, tag);
    } else {
        printf("  |    singularity: NONE in (0,2)  [SAFE]\n");
    }
    if (r.w5_raw < 0.0 || r.w5_raw > 2.0)
        printf("  |    >>> FALLBACK: omega5 out of [0,2] -> AO fallback = 1.0\n");
    else
        printf("  |    >>> STATUS: omega5 in stable range (0,2)  [OK]\n");

    printf("  |                                                                    |\n");

    // -- A, B --
    printf("  |  A, B (Eq.17-18, 4th-order equilibrium coefficients):              |\n");
    printf("  |    A = %+10.6f   B = %+10.6f   denom = %+12.6f\n", r.A, r.B, r.denom_AB);
    printf("  |    singularity 1: w1 = w2 = %.4f (dist = %.4f)\n", r.AB_sing1, fabs(r.w1-r.AB_sing1));
    if (r.AB_sing2 > 0.0 && r.AB_sing2 < 2.0)
        printf("  |    singularity 2: w1 = %.4f     (dist = %.4f)\n", r.AB_sing2, fabs(r.w1-r.AB_sing2));
    else
        printf("  |    singularity 2: w1 = %.4f     (outside (0,2), safe)\n", r.AB_sing2);
    if (fabs(r.denom_AB) <= CUM_DIAG_DEN_EPS)
        printf("  |    >>> FALLBACK: A=B=0 (degenerate to AO 4th-order eq)\n");
    else if (r.AB_min_dist < CUM_DIAG_SAFE_MARGIN)
        printf("  |    >>> WARNING: near A/B singularity (dist=%.4f)\n", r.AB_min_dist);
    else
        printf("  |    >>> STATUS: A,B well-defined  [OK]\n");

    // -- Overall verdict --
    printf("  +--------------------------------------------------------------------+\n");
    if (!r.any_danger && !r.any_fallback) {
        printf("  |  VERDICT: ALL CLEAR                                                |\n");
        printf("  |    All omega in stable range (0,2). WP optimization fully active.  |\n");
    } else if (r.any_fallback && !r.any_danger) {
        printf("  |  VERDICT: FALLBACK ACTIVE (WP degraded to partial AO)              |\n");
        printf("  |    Simulation is STABLE, but WP accuracy advantage is lost.        |\n");
    } else {
        printf("  |  VERDICT: *** DANGER -- SINGULARITY PROXIMITY ***                   |\n");
        printf("  |    One or more omega near denominator zero.                        |\n");
        printf("  |    Fallback to AO=1.0 active, but WP benefit fully lost.          |\n");
    }

    // -- Root cause + suggestions --
    if (r.any_danger || r.any_fallback) {
        printf("  +--------------------------------------------------------------------+\n");
        printf("  |  Root Cause:                                                       |\n");
        printf("  |    {Re=%d, Uref=%.4f, Jacobian->dt=%.4e, omega2=%.2f}\n",
               r.Re, r.Uref, r.dt_global, r.w2);
        printf("  |    -> tau = %.6f -> omega1 = %.6f\n", r.tau, r.w1);

        if (r.w4_sing > 0.0 && r.w4_sing_dist < CUM_DIAG_SAFE_MARGIN)
            printf("  |    -> omega4 Eq.15 den=0 at w1=%.4f (dist=%.4f, CRITICAL)\n",
                   r.w4_sing, r.w4_sing_dist);
        if (r.w3_sing > 0.0 && r.w3_sing_dist < CUM_DIAG_SAFE_MARGIN)
            printf("  |    -> omega3 Eq.14 den=0 at w1=%.4f (dist=%.4f)\n",
                   r.w3_sing, r.w3_sing_dist);
        if (r.w5_sing > 0.0 && r.w5_sing_dist < CUM_DIAG_SAFE_MARGIN)
            printf("  |    -> omega5 Eq.16 den=0 at w1=%.4f (dist=%.4f)\n",
                   r.w5_sing, r.w5_sing_dist);
        if (r.AB_min_dist < CUM_DIAG_SAFE_MARGIN)
            printf("  |    -> A/B Eq.17-18 denom=0 (dist=%.4f)\n", r.AB_min_dist);

        printf("  |                                                                    |\n");
        printf("  |  Suggested Fixes:                                                  |\n");
        printf("  |    (a) Adjust Uref: change tau away from singularity               |\n");

        // Compute safe tau range
        if (r.w4_sing > 0.0 && r.w4_sing_dist < 0.3) {
            double tau_safe_hi = 1.0 / (r.w4_sing - CUM_DIAG_SAFE_MARGIN);
            double tau_safe_lo = 1.0 / (r.w4_sing + CUM_DIAG_SAFE_MARGIN);
            printf("  |        target: tau > %.4f (w1 < %.4f)\n",
                   tau_safe_hi, r.w4_sing - CUM_DIAG_SAFE_MARGIN);
            printf("  |             or tau < %.4f (w1 > %.4f)\n",
                   tau_safe_lo, r.w4_sing + CUM_DIAG_SAFE_MARGIN);
            // Corresponding Uref
            // tau = 3*(Uref/Re)/dt + 0.5
            // Uref = Re * (tau - 0.5) * dt / 3
            double Uref_safe_hi = (double)r.Re * (tau_safe_hi - 0.5) * r.dt_global / 3.0;
            double Uref_safe_lo = (double)r.Re * (tau_safe_lo - 0.5) * r.dt_global / 3.0;
            printf("  |        -> Uref > %.6f or Uref < %.6f\n", Uref_safe_hi, Uref_safe_lo);
            printf("  |        (Re=%d, dt_global=%.4e fixed)\n", r.Re, r.dt_global);
        }

        printf("  |    (b) Change omega2 to shift singularity position                |\n");
        // Try a few omega2 values
        double w2_try[] = {0.5, 0.7, 0.8, 1.2, 1.5};
        for (int t = 0; t < 5; t++) {
            double w2t = w2_try[t];
            double a4 = 9.0*w2t, b4 = -(42.0*w2t+8.0), c4 = 56.0*w2t;
            double disc = b4*b4-4.0*a4*c4;
            if (disc >= 0.0) {
                double root = (-b4-sqrt(disc))/(2.0*a4);
                if (root > 0.0 && root < 2.0) {
                    double d = fabs(r.w1 - root);
                    printf("  |        omega2=%.1f -> w4 sing at %.4f (dist=%.4f) %s\n",
                           w2t, root, d, (d > 0.3) ? "[SAFE]" : (d > 0.15) ? "[OK]" : "[CLOSE]");
                }
            }
        }

        printf("  |    (c) Add smooth blending (code fix in _cum_wp_compute_omega345)  |\n");
    }

    printf("  +--------------------------------------------------------------------+\n\n");
}

// ================================================================
// One-line call interface for main.cu
// ================================================================
static void CumulantWP_DiagnoseOmega(
    int Re_in, double Uref_in, double dt_global_in, double omega2_in,
    int myid)
{
    if (myid != 0) return;  // only output on rank 0
    CumWP_OmegaReport rep = CumulantWP_ComputeReport(Re_in, Uref_in, dt_global_in, omega2_in);
    CumulantWP_PrintReport(rep);
}

#endif // CUMULANT_WP_DIAGNOSTIC_H
