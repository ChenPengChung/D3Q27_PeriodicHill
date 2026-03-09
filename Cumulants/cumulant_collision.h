// ================================================================
// cumulant_collision.h
// D3Q27 Cumulant Collision  Standalone CUDA Device Function
//
// TWO MODES (compile-time selection via USE_WP_CUMULANT):
//
//   USE_WP_CUMULANT = 0  ->  AO (All-One, Geier 2015)
//     All w2w10 = 1.  Simple, stable at high Re via damping.
//     Equivalent to original Geier et al. 2015 approach.
//
//   USE_WP_CUMULANT = 1  ->  WP (Well-conditioned Parameterized, Geier 2017)
//     w3,w4,w5 optimized from w1,w2 (Eq.14-16 of Gehrke & Rung 2022).
//     4th-order equilibria use A,B coefficients (Eq.17-18).
//     Regularization limiter lambda (Eq.20-26) for stability control.
//     Superior accuracy at moderate grids; lambda tunes DNS<->VLES blend.
//
// Interface:
//   INPUT:  f_in[27], omega(=w1), delta_t, Fx, Fy, Fz
//   OUTPUT: f_out[27], rho, ux, uy, uz
//
// References:
//   [G15] Geier et al., Comp. Math. Appl. 70(4), 507-547, 2015
//   [G17] Geier et al., J. Comput. Phys. 348, 862-888, 2017
//   [GR22] Gehrke & Rung, Int. J. Numer. Meth. Fluids 94, 1111-1154, 2022
// ================================================================
#ifndef CUMULANT_COLLISION_H
#define CUMULANT_COLLISION_H

#include "cumulant_constants.h"

// ================================================================
// Default compile-time mode (override in variables.h BEFORE including)
// ================================================================
#ifndef USE_WP_CUMULANT
#define USE_WP_CUMULANT 0   // 0 = AO (All-One), 1 = WP (Parameterized)
#endif

// ================================================================
// WP regularization parameter lambda  (only used when USE_WP_CUMULANT=1)
//   lambda_def = 1e-2  (Gehrke default, good for most cases)
//   lambda = 1e-6      effectively -> AO (regularization off)
//   lambda = 1e-1~1e0  optimal for Re>=10600 on medium grids (Table 7, GR22)
// ================================================================
#ifndef CUM_LAMBDA
#define CUM_LAMBDA 1.0e-2
#endif

// Forward declarations
__device__ static void _cum_forward_chimera(double m[27], const double u[3]);
__device__ static void _cum_backward_chimera(double m[27], const double u[3]);

#if USE_WP_CUMULANT
// ================================================================
// WP helper: compute parameterized w3,w4,w5 from w1,w2
//   [GR22] Eq. 14-16, derived from 4th-order diffusion error optimization
// ================================================================
__device__ static void _cum_wp_compute_omega345(
    const double w1, const double w2,
    double *w3, double *w4, double *w5)
{
    // Denominator singularity guard:
    //   den4 = 0 at w1 = 14/9 ~ 1.5556 (w2=1), w4 diverges
    //   den3, den5 have no zeros in (0,2) for w2=1, but guard conservatively
    //   When |den| < eps, corresponding omega falls back to 1.0 (neutral)
    const double DEN_EPS = 1.0e-10;

    // Eq. 14: w3
    double num3  = 8.0 * (w1 - 2.0) * (w2 * (3.0*w1 - 1.0) - 5.0*w1);
    double den3  = 8.0 * (5.0 - 2.0*w1) * w1
                 + w2 * (8.0 + w1 * (9.0*w1 - 26.0));
    *w3 = (fabs(den3) > DEN_EPS) ? num3 / den3 : 1.0;

    // Eq. 15: w4  (den4=0 at w1=14/9)
    double num4  = 8.0 * (w1 - 2.0) * (w1 + w2 * (3.0*w1 - 7.0));
    double den4  = w2 * (56.0 - 42.0*w1 + 9.0*w1*w1) - 8.0*w1;
    *w4 = (fabs(den4) > DEN_EPS) ? num4 / den4 : 1.0;

    // Eq. 16: w5
    double num5  = 24.0 * (w1 - 2.0) * (4.0*w1*w1
                 + w1*w2*(18.0 - 13.0*w1)
                 + w2*w2*(2.0 + w1*(6.0*w1 - 11.0)));
    double den5  = 16.0*w1*w1*(w1 - 6.0)
                 - 2.0*w1*w2*(216.0 + 5.0*w1*(9.0*w1 - 46.0))
                 + w2*w2*(w1*(3.0*w1 - 10.0)*(15.0*w1 - 28.0) - 48.0);
    *w5 = (fabs(den5) > DEN_EPS) ? num5 / den5 : 1.0;
}

// ================================================================
// WP helper: compute A, B coefficients for 4th-order equilibria
//   [GR22] Eq. 17-18
// ================================================================
__device__ static void _cum_wp_compute_AB(
    const double w1, const double w2,
    double *A, double *B)
{
    // Denominator zeros: denom = (w1-w2)*(w2*(2+3*w1)-8*w1)
    //   Zero 1: w1 = w2 (= 1.0 when w2=1)
    //   Zero 2: w1 = 2*w2/(8-3*w2) ~ 0.4 (when w2=1)
    //   Fallback: A=B=0 (degenerates to AO mode, 4th-order eq = 0)
    const double DEN_EPS = 1.0e-10;
    double denom = (w1 - w2) * (w2 * (2.0 + 3.0*w1) - 8.0*w1);

    if (fabs(denom) > DEN_EPS) {
        // Eq. 17: A
        *A = (4.0*w1*w1 + 2.0*w1*w2*(w1 - 6.0)
           + w2*w2*(w1*(10.0 - 3.0*w1) - 4.0)) / denom;

        // Eq. 18: B
        *B = (4.0*w1*w2*(9.0*w1 - 16.0) - 4.0*w1*w1
           - 2.0*w2*w2*(2.0 + 9.0*w1*(w1 - 2.0)))
           / (3.0 * denom);
    } else {
        // Degenerate: w1=w2 or near-singular -> A=B=0 (4th-order eq = 0, same as AO)
        *A = 0.0;
        *B = 0.0;
    }
}

// ================================================================
// WP helper: apply lambda-limiter to a single relaxation rate
//   omega^lambda = omega_base + (1 - omega_base)|C_mag| / (rholambda + |C_mag|)
//   [GR22] Eq. 20-26 pattern
// ================================================================
__device__ static double _cum_wp_limit(
    const double omega_base, const double C_mag,
    const double rho, const double lambda)
{
    double absC = fabs(C_mag);
    return omega_base + (1.0 - omega_base) * absC / (rho * lambda + absC);
}
#endif // USE_WP_CUMULANT


// ================================================================
//
//  * Main entry point *
//
//  Usage:
//    cumulant_collision_D3Q27(
//        f_streamed, omega_global, GILBM_dt,
//        0.0, Force[0], 0.0,       // Fy=streamwise
//        f_post, rho, ux, uy, uz);
//
// ================================================================
__device__ void cumulant_collision_D3Q27(
    const double f_in[27],   // INPUT:  post-streaming distributions
    const double omega,      // INPUT:  shear relaxation rate w1
    const double delta_t,    // INPUT:  time step (for half-force correction) [avoid macro collision with #define dt]
    const double Fx,         // INPUT:  body force x
    const double Fy,         // INPUT:  body force y
    const double Fz,         // INPUT:  body force z
    double       f_out[27],  // OUTPUT: post-collision distributions
    double      *rho_out,    // OUTPUT: density
    double      *ux_out,     // OUTPUT: velocity x (half-force corrected)
    double      *uy_out,     // OUTPUT: velocity y
    double      *uz_out      // OUTPUT: velocity z
)
{
    // ==============================================================
    // STAGE 0: Macroscopic Quantities + Well-Conditioning
    // ==============================================================

    // 0a. Density: rho = Sum f_alpha
    double rho = 0.0;
    for (int i = 0; i < 27; i++) rho += f_in[i];

    // 0b. Momentum: j = Sum f_alpha  e_alpha
    double jx = 0.0, jy = 0.0, jz = 0.0;
    for (int i = 0; i < 27; i++) {
        jx += f_in[i] * GILBM_e[i][0];
        jy += f_in[i] * GILBM_e[i][1];
        jz += f_in[i] * GILBM_e[i][2];
    }

    // 0c. Half-force corrected velocity (Guo 2002)
    double inv_rho = 1.0 / rho;
    double u[3];
    u[0] = jx * inv_rho + 0.5 * Fx * inv_rho * delta_t;
    u[1] = jy * inv_rho + 0.5 * Fy * inv_rho * delta_t;
    u[2] = jz * inv_rho + 0.5 * Fz * inv_rho * delta_t;

    // 0d. Well-conditioning: f = f - w
    double m[27];
    for (int i = 0; i < 27; i++) {
        m[i] = f_in[i] - GILBM_W[i];
    }

    // 0e. Auxiliary
    double drho = rho - 1.0;

    // ==============================================================
    // STAGE 1: Forward Chimera Transform (z -> y -> x)
    //          f[27] -> kappa[27] (central moments)
    //          [G15] Appendix J, Eq. J.4J.12
    // ==============================================================
    _cum_forward_chimera(m, u);

    // ==============================================================
    // STAGE 2: Central Moments -> Cumulants
    //          Orders 03: C = kappa (identical)
    //          Orders 46: subtract lower-order products
    //          [G15] Appendix J, Eq. J.16J.19
    // ==============================================================

    // --- 4th order off-diagonal (Eq. J.16) ---
    double CUMcbb = m[I_cbb] - ((m[I_caa] + 1.0/3.0) * m[I_abb]
                    + 2.0 * m[I_bba] * m[I_bab]) * inv_rho;
    double CUMbcb = m[I_bcb] - ((m[I_aca] + 1.0/3.0) * m[I_bab]
                    + 2.0 * m[I_bba] * m[I_abb]) * inv_rho;
    double CUMbbc = m[I_bbc] - ((m[I_aac] + 1.0/3.0) * m[I_bba]
                    + 2.0 * m[I_bab] * m[I_abb]) * inv_rho;

    // --- 4th order diagonal (Eq. J.17) ---
    double CUMcca = m[I_cca] - (((m[I_caa]*m[I_aca] + 2.0*m[I_bba]*m[I_bba])
                    + 1.0/3.0*(m[I_caa]+m[I_aca])) * inv_rho
                    - 1.0/9.0*(drho*inv_rho));
    double CUMcac = m[I_cac] - (((m[I_caa]*m[I_aac] + 2.0*m[I_bab]*m[I_bab])
                    + 1.0/3.0*(m[I_caa]+m[I_aac])) * inv_rho
                    - 1.0/9.0*(drho*inv_rho));
    double CUMacc = m[I_acc] - (((m[I_aac]*m[I_aca] + 2.0*m[I_abb]*m[I_abb])
                    + 1.0/3.0*(m[I_aac]+m[I_aca])) * inv_rho
                    - 1.0/9.0*(drho*inv_rho));

    // --- 5th order (Eq. J.18) ---
    double CUMbcc = m[I_bcc] - ((m[I_aac]*m[I_bca] + m[I_aca]*m[I_bac]
                    + 4.0*m[I_abb]*m[I_bbb]
                    + 2.0*(m[I_bab]*m[I_acb] + m[I_bba]*m[I_abc]))
                    + 1.0/3.0*(m[I_bca]+m[I_bac])) * inv_rho;
    double CUMcbc = m[I_cbc] - ((m[I_aac]*m[I_cba] + m[I_caa]*m[I_abc]
                    + 4.0*m[I_bab]*m[I_bbb]
                    + 2.0*(m[I_abb]*m[I_cab] + m[I_bba]*m[I_bac]))
                    + 1.0/3.0*(m[I_cba]+m[I_abc])) * inv_rho;
    double CUMccb = m[I_ccb] - ((m[I_caa]*m[I_acb] + m[I_aca]*m[I_cab]
                    + 4.0*m[I_bba]*m[I_bbb]
                    + 2.0*(m[I_bab]*m[I_bca] + m[I_abb]*m[I_cba]))
                    + 1.0/3.0*(m[I_acb]+m[I_cab])) * inv_rho;

    // --- 6th order (Eq. J.19) ---
    // NOTE: The well-conditioning correction term for 4th-order central moments
    //       (acc+cac+cca) uses coefficient -1/3, consistent with the 4th-order
    //       diagonal well-conditioning structure (cf. Eq. J.17 uses +1/3).
    double CUMccc = m[I_ccc]
        + ((-4.0*m[I_bbb]*m[I_bbb]
            - (m[I_caa]*m[I_acc] + m[I_aca]*m[I_cac] + m[I_aac]*m[I_cca])
            - 4.0*(m[I_abb]*m[I_cbb] + m[I_bab]*m[I_bcb] + m[I_bba]*m[I_bbc])
            - 2.0*(m[I_bca]*m[I_bac] + m[I_cba]*m[I_abc] + m[I_cab]*m[I_acb]))
                * inv_rho
        + (4.0*(m[I_bab]*m[I_bab]*m[I_aca]
              + m[I_abb]*m[I_abb]*m[I_caa]
              + m[I_bba]*m[I_bba]*m[I_aac])
          + 2.0*m[I_caa]*m[I_aca]*m[I_aac]
          + 16.0*m[I_bba]*m[I_bab]*m[I_abb])
                * inv_rho * inv_rho
        - 1.0/3.0*(m[I_acc]+m[I_cac]+m[I_cca]) * inv_rho
        - 1.0/9.0*(m[I_caa]+m[I_aca]+m[I_aac]) * inv_rho
        + (2.0*(m[I_bab]*m[I_bab] + m[I_abb]*m[I_abb] + m[I_bba]*m[I_bba])
          + (m[I_aac]*m[I_aca] + m[I_aac]*m[I_caa] + m[I_aca]*m[I_caa])
          + 1.0/3.0*(m[I_aac]+m[I_aca]+m[I_caa]))
                * inv_rho * inv_rho * 2.0/3.0
        + 1.0/27.0*((drho*drho - drho) * inv_rho * inv_rho));

    // ==============================================================
    // STAGE 3: Relaxation (Collision in Cumulant Space)
    // ==============================================================
    //
    // 
    //   AO mode (USE_WP_CUMULANT=0):                          
    //     w2 = w3 = w4 = ... = w10 = 1.0                    
    //     All non-equilibrium parts fully damped.              
    //     4th-order cumulant equilibria = 0.                   
    //     Simple, robust, but accuracy-limited. [G15]          
    //                                                          
    //   WP mode (USE_WP_CUMULANT=1):                          
    //     w2 = 1.0 (bulk viscosity, user choice)              
    //     w3,w4,w5 = f(w1,w2) [GR22 Eq.14-16]              
    //     -> then lambda-limited [GR22 Eq.20-26]                    
    //     4th-order equilibria != 0, use A,B [GR22 Eq.17-18]   
    //     w6=w7=w8 = 1.0, w9 = 1.0, w10 = 1.0             
    // 

    // ---- w2: bulk viscosity (same for both modes) ----
    const double omega2 = 1.0;

    // ---- w6=w7=w8: 4th order, w9: 5th order, w10: 6th order ----
    const double omega6  = 1.0;
    const double omega9  = 1.0;
    const double omega10 = 1.0;

#if USE_WP_CUMULANT
    // ============================================================
    //  WP MODE: Parameterized relaxation [GR22]
    // ============================================================

    // Step A: Compute base w3,w4,w5 from w1,w2 (Eq.14-16)
    double omega3_base, omega4_base, omega5_base;
    _cum_wp_compute_omega345(omega, omega2, &omega3_base, &omega4_base, &omega5_base);

    // Safety clamp: Eq.15 denominator at w1~1.45-1.75 range crosses zero, causing w4 divergence
    // clamped to [0, 2] ensuring base relaxation rate within stability boundary, lambda-limiter then fine-tuned from this baseline
    omega3_base = fmax(0.0, fmin(2.0, omega3_base));
    omega4_base = fmax(0.0, fmin(2.0, omega4_base));
    omega5_base = fmax(0.0, fmin(2.0, omega5_base));

    // Step B: Compute A, B for 4th-order equilibria (Eq.17-18)
    double coeff_A, coeff_B;
    _cum_wp_compute_AB(omega, omega2, &coeff_A, &coeff_B);

    // Step C: Extract raw 3rd-order cumulants BEFORE symmetric/antisymmetric
    //         decomposition (needed for lambda-limiter magnitude, Eq.20-26)
    //
    //   * IMPORTANT: lambda-limiter magnitude parameter must use[original/raw]cumulant values, not symmetric/antisymmetric decomposed values
    //     After 3rd-order moments extracted here, after 2nd-order relaxation (only 2nd-order moments modified) they are not modified,
    //     until line 360+ then sym/antisym decomposition
    //     Orders  3: cumulant = central moment (C = kappa), no extra conversion needed
    double C_120 = m[I_bca];   // kappa120 = C120
    double C_102 = m[I_bac];   // kappa102 = C102
    double C_210 = m[I_cba];   // kappa210 = C210
    double C_012 = m[I_abc];   // kappa012 = C012
    double C_201 = m[I_cab];   // kappa201 = C201
    double C_021 = m[I_acb];   // kappa021 = C021
    double C_111 = m[I_bbb];   // kappa111 = C111

    // Step D: Apply lambda-limiter to get effective omega^lambda (Eq.20-26)
    const double lam = CUM_LAMBDA;

    // Eq.20: omega^lambda_{3,1} for (C120 + C102) symmetric pair
    double omega3_1 = _cum_wp_limit(omega3_base, C_120 + C_102, rho, lam);
    // Eq.21: omega^lambda_{4,1} for (C120 - C102) antisymmetric pair
    double omega4_1 = _cum_wp_limit(omega4_base, C_120 - C_102, rho, lam);
    // Eq.22: omega^lambda_{3,2} for (C210 + C012)
    double omega3_2 = _cum_wp_limit(omega3_base, C_210 + C_012, rho, lam);
    // Eq.23: omega^lambda_{4,2} for (C210 - C012)
    double omega4_2 = _cum_wp_limit(omega4_base, C_210 - C_012, rho, lam);
    // Eq.24: omega^lambda_{3,3} for (C201 + C021)
    double omega3_3 = _cum_wp_limit(omega3_base, C_201 + C_021, rho, lam);
    // Eq.25: omega^lambda_{4,3} for (C201 - C021)
    double omega4_3 = _cum_wp_limit(omega4_base, C_201 - C_021, rho, lam);
    // Eq.26: omega^lambda_5 for C111
    double omega5_lim = _cum_wp_limit(omega5_base, C_111, rho, lam);

#else
    // ============================================================
    //  AO MODE: All-One relaxation [G15]
    // ============================================================
    // All 3rd-order relaxation rates = 1 (full damping to equilibrium)
    // No per-pair distinction, no limiter.
#endif

    // ---- 2nd order: decompose into orthogonal modes ----
    double mxxPyyPzz = m[I_caa] + m[I_aca] + m[I_aac];  // trace (bulk)
    double mxxMyy    = m[I_caa] - m[I_aca];              // deviatoric 1
    double mxxMzz    = m[I_caa] - m[I_aac];              // deviatoric 2

    // Relax trace with w2 toward rho equilibrium
    mxxPyyPzz += omega2 * (m[I_aaa] - mxxPyyPzz);
    // Relax deviatorics with w1 toward 0
    mxxMyy *= (1.0 - omega);
    mxxMzz *= (1.0 - omega);

    // Off-diagonal 2nd order with w1
    m[I_abb] *= (1.0 - omega);  // C011
    m[I_bab] *= (1.0 - omega);  // C101
    m[I_bba] *= (1.0 - omega);  // C110

    // ---- 3rd order: symmetric / antisymmetric decomposition ----
    double mxxyPyzz = m[I_cba] + m[I_abc];   // C210 + C012
    double mxxyMyzz = m[I_cba] - m[I_abc];   // C210 - C012
    double mxxzPyyz = m[I_cab] + m[I_acb];   // C201 + C021
    double mxxzMyyz = m[I_cab] - m[I_acb];   // C201 - C021
    double mxyyPxzz = m[I_bca] + m[I_bac];   // C120 + C102
    double mxyyMxzz = m[I_bca] - m[I_bac];   // C120 - C102

#if USE_WP_CUMULANT
    // WP: each pair has its OWN lambda-limited relaxation rate
    m[I_bbb]  *= (1.0 - omega5_lim);    // C111 (Eq.26)
    mxxyPyzz  *= (1.0 - omega3_2);      // (C210+C012) symmetric (Eq.22)
    mxxyMyzz  *= (1.0 - omega4_2);      // (C210-C012) antisymmetric (Eq.23)
    mxxzPyyz  *= (1.0 - omega3_3);      // (C201+C021) symmetric (Eq.24)
    mxxzMyyz  *= (1.0 - omega4_3);      // (C201-C021) antisymmetric (Eq.25)
    mxyyPxzz  *= (1.0 - omega3_1);      // (C120+C102) symmetric (Eq.20)
    mxyyMxzz  *= (1.0 - omega4_1);      // (C120-C102) antisymmetric (Eq.21)
#else
    // AO: all = 1 -> (1-omega) = 0 -> full relaxation to zero
    m[I_bbb]  = 0.0;
    mxxyPyzz  = 0.0;
    mxxyMyzz  = 0.0;
    mxxzPyyz  = 0.0;
    mxxzMyyz  = 0.0;
    mxyyPxzz  = 0.0;
    mxyyMxzz  = 0.0;
#endif

    // ---- Reconstruct 2nd order individual moments ----
    m[I_caa] = ( mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
    m[I_aca] = (-2.0*mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
    m[I_aac] = ( mxxMyy - 2.0*mxxMzz + mxxPyyPzz) / 3.0;

    // ---- Reconstruct 3rd order ----
    m[I_cba] = ( mxxyMyzz + mxxyPyzz) * 0.5;
    m[I_abc] = (-mxxyMyzz + mxxyPyzz) * 0.5;
    m[I_cab] = ( mxxzMyyz + mxxzPyyz) * 0.5;
    m[I_acb] = (-mxxzMyyz + mxxzPyyz) * 0.5;
    m[I_bca] = ( mxyyMxzz + mxyyPxzz) * 0.5;
    m[I_bac] = (-mxyyMxzz + mxyyPxzz) * 0.5;

    // ---- 4th order relaxation ----
#if USE_WP_CUMULANT
    // WP: 4th-order diagonal cumulants relax toward NON-ZERO equilibria
    //   C^eq_{alphaalpha,betabeta} involves products of 2nd-order cumulants via A, B
    //   [GR22] Eq.17-18, Appendix B.2
    //
    // 2nd-order cumulants (post-relaxation, well-conditioned):
    //   Dxx = m[I_caa], Dyy = m[I_aca], Dzz = m[I_aac]
    //   Dxy = m[I_bba], Dxz = m[I_bab], Dyz = m[I_abb]
    //
    // Diagonal 4th-order equilibria (Geier 2017, Appendix B.2):
    //   C^eq_{2200} = (A+B)(DxxDyy + Dxy^2) + (A-B)(DxxDyy - Dxy^2)
    //               = A(DxxDyy + Dxy^2) + B(DxxDyy - Dxy^2)  ... WRONG
    //   Actually per Geier:
    //   C^eq_{xxyy} / rho = A(DxxDyy/rho + Dxy^2/rho) + B(DxxDyy/rho - Dxy^2/rho)  (with /rho)
    //   But in well-conditioned form these are already normalized.
    //   The OpenLB implementation uses:
    double Dxx = m[I_caa], Dyy = m[I_aca], Dzz = m[I_aac];
    double Dxy = m[I_bba], Dxz = m[I_bab], Dyz = m[I_abb];

    // 4th-order diagonal equilibria (A,B from optimized parameterization):
    double CUMcca_eq = (coeff_A * (Dxx*Dyy + Dxy*Dxy) + coeff_B * (Dxx*Dyy - Dxy*Dxy)) * inv_rho;
    double CUMcac_eq = (coeff_A * (Dxx*Dzz + Dxz*Dxz) + coeff_B * (Dxx*Dzz - Dxz*Dxz)) * inv_rho;
    double CUMacc_eq = (coeff_A * (Dyy*Dzz + Dyz*Dyz) + coeff_B * (Dyy*Dzz - Dyz*Dyz)) * inv_rho;

    // Relax diagonal 4th-order toward non-zero equilibria
    CUMcca += omega6 * (CUMcca_eq - CUMcca);
    CUMcac += omega6 * (CUMcac_eq - CUMcac);
    CUMacc += omega6 * (CUMacc_eq - CUMacc);

    // Off-diagonal 4th-order: equilibria = 0 (same as AO)
    CUMbbc *= (1.0 - omega6);
    CUMbcb *= (1.0 - omega6);
    CUMcbb *= (1.0 - omega6);
#else
    // AO: all 4th-order cumulants relax toward 0
    CUMacc *= (1.0 - omega6);
    CUMcac *= (1.0 - omega6);
    CUMcca *= (1.0 - omega6);
    CUMbbc *= (1.0 - omega6);
    CUMbcb *= (1.0 - omega6);
    CUMcbb *= (1.0 - omega6);
#endif

    // ---- 5th order: equilibria = 0 for both modes ----
    CUMbcc *= (1.0 - omega9);
    CUMcbc *= (1.0 - omega9);
    CUMccb *= (1.0 - omega9);

    // ---- 6th order: equilibria = 0 for both modes ----
    CUMccc *= (1.0 - omega10);

    // ==============================================================
    // STAGE 4: Cumulants -> Central Moments (Inverse of Stage 2)
    //          [G15] Appendix J, Eq. J.16J.19 inverse
    // ==============================================================

    // --- 4th order off-diagonal inverse (Eq. J.16) ---
    m[I_cbb] = CUMcbb + ((m[I_caa] + 1.0/3.0)*m[I_abb]
               + 2.0*m[I_bba]*m[I_bab]) * inv_rho;
    m[I_bcb] = CUMbcb + ((m[I_aca] + 1.0/3.0)*m[I_bab]
               + 2.0*m[I_bba]*m[I_abb]) * inv_rho;
    m[I_bbc] = CUMbbc + ((m[I_aac] + 1.0/3.0)*m[I_bba]
               + 2.0*m[I_bab]*m[I_abb]) * inv_rho;

    // --- 4th order diagonal inverse (Eq. J.17) ---
    m[I_cca] = CUMcca + (((m[I_caa]*m[I_aca]+2.0*m[I_bba]*m[I_bba])*9.0
               + 3.0*(m[I_caa]+m[I_aca])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;
    m[I_cac] = CUMcac + (((m[I_caa]*m[I_aac]+2.0*m[I_bab]*m[I_bab])*9.0
               + 3.0*(m[I_caa]+m[I_aac])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;
    m[I_acc] = CUMacc + (((m[I_aac]*m[I_aca]+2.0*m[I_abb]*m[I_abb])*9.0
               + 3.0*(m[I_aac]+m[I_aca])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;

    // --- 5th order inverse (Eq. J.18) ---
    m[I_bcc] = CUMbcc + ((m[I_aac]*m[I_bca] + m[I_aca]*m[I_bac]
               + 4.0*m[I_abb]*m[I_bbb]
               + 2.0*(m[I_bab]*m[I_acb] + m[I_bba]*m[I_abc]))
               + 1.0/3.0*(m[I_bca]+m[I_bac])) * inv_rho;
    m[I_cbc] = CUMcbc + ((m[I_aac]*m[I_cba] + m[I_caa]*m[I_abc]
               + 4.0*m[I_bab]*m[I_bbb]
               + 2.0*(m[I_abb]*m[I_cab] + m[I_bba]*m[I_bac]))
               + 1.0/3.0*(m[I_cba]+m[I_abc])) * inv_rho;
    m[I_ccb] = CUMccb + ((m[I_caa]*m[I_acb] + m[I_aca]*m[I_cab]
               + 4.0*m[I_bba]*m[I_bbb]
               + 2.0*(m[I_bab]*m[I_bca] + m[I_abb]*m[I_cba]))
               + 1.0/3.0*(m[I_acb]+m[I_cab])) * inv_rho;

    // --- 6th order inverse (Eq. J.19) ---
    // * BUG FIX: coefficient for (acc+cac+cca)/rho is -1/3, matching forward *
    m[I_ccc] = CUMccc
        - ((-4.0*m[I_bbb]*m[I_bbb]
            - (m[I_caa]*m[I_acc]+m[I_aca]*m[I_cac]+m[I_aac]*m[I_cca])
            - 4.0*(m[I_abb]*m[I_cbb]+m[I_bab]*m[I_bcb]+m[I_bba]*m[I_bbc])
            - 2.0*(m[I_bca]*m[I_bac]+m[I_cba]*m[I_abc]+m[I_cab]*m[I_acb]))
                * inv_rho
        + (4.0*(m[I_bab]*m[I_bab]*m[I_aca]
              + m[I_abb]*m[I_abb]*m[I_caa]
              + m[I_bba]*m[I_bba]*m[I_aac])
          + 2.0*(m[I_caa]*m[I_aca]*m[I_aac])
          + 16.0*m[I_bba]*m[I_bab]*m[I_abb])
                * inv_rho * inv_rho
        - 1.0/3.0*(m[I_acc]+m[I_cac]+m[I_cca]) * inv_rho    // * FIXED: was 1/9
        - 1.0/9.0*(m[I_caa]+m[I_aca]+m[I_aac]) * inv_rho
        + (2.0*(m[I_bab]*m[I_bab]+m[I_abb]*m[I_abb]+m[I_bba]*m[I_bba])
          + (m[I_aac]*m[I_aca]+m[I_aac]*m[I_caa]+m[I_aca]*m[I_caa])
          + 1.0/3.0*(m[I_aac]+m[I_aca]+m[I_caa]))
                * inv_rho * inv_rho * 2.0/3.0
        + 1.0/27.0*((drho*drho - drho) * inv_rho * inv_rho));

    // --- Force correction: sign flip of 1st-order moments (Eq. 85-87) ---
    m[I_baa] = -m[I_baa];
    m[I_aba] = -m[I_aba];
    m[I_aab] = -m[I_aab];

    // ==============================================================
    // STAGE 5: Backward Chimera Transform (x -> y -> z)
    //          [G15] Appendix J, Eq. J.20J.28
    // ==============================================================
    _cum_backward_chimera(m, u);

    // Restore from well-conditioned: f* = f* + w
    for (int i = 0; i < 27; i++) {
        f_out[i] = m[i] + GILBM_W[i];
    }

    // Output macroscopic quantities
    *rho_out = rho;
    *ux_out  = u[0];
    *uy_out  = u[1];
    *uz_out  = u[2];
}


// ================================================================
// Internal: Forward Chimera (Stage 1)
// Sweep order: z(dir=2) -> y(dir=1) -> x(dir=0)
// [G15] Appendix J, Eq. J.4J.12
// ================================================================
__device__ static void _cum_forward_chimera(
    double m[27], const double u[3])
{
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double k = CUM_K[p];

            double sum  = m[a] + m[c];
            double diff = m[c] - m[a];

            m[a] = m[a] + m[b] + m[c];
            m[b] = diff - (m[a] + k) * u[dir];
            m[c] = sum - 2.0 * diff * u[dir]
                   + u[dir] * u[dir] * (m[a] + k);
        }
    }
}

// ================================================================
// Internal: Backward Chimera (Stage 5)
// Sweep order: x(dir=0) -> y(dir=1) -> z(dir=2)
// [G15] Appendix J, Eq. J.20J.28
// ================================================================
__device__ static void _cum_backward_chimera(
    double m[27], const double u[3])
{
    for (int dir = 0; dir < 3; dir++) {
        int base = (2 - dir) * 9;
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double k = CUM_K[p];

            double ma = ((m[c] - m[b]) * 0.5 + m[b] * u[dir]
                        + (m[a] + k) * (u[dir]*u[dir] - u[dir]) * 0.5);
            double mb = (m[a] - m[c]) - 2.0 * m[b] * u[dir]
                        - (m[a] + k) * u[dir] * u[dir];
            double mc = ((m[c] + m[b]) * 0.5 + m[b] * u[dir]
                        + (m[a] + k) * (u[dir]*u[dir] + u[dir]) * 0.5);

            m[a] = ma;
            m[b] = mb;
            m[c] = mc;
        }
    }
}

#endif // CUMULANT_COLLISION_H
