// ================================================================
// cumulant_collision.h
// D3Q27 Cumulant Collision — Standalone CUDA Device Function
//
// TWO MODES (compile-time selection via USE_WP_CUMULANT):
//
//   USE_WP_CUMULANT = 0  →  AO (All-One, Geier 2015)
//     All ω₂–ω₁₀ = 1.  Simple, stable at high Re via damping.
//     Equivalent to original Geier et al. 2015 approach.
//
//   USE_WP_CUMULANT = 1  →  WP (Well-conditioned Parameterized, Geier 2017)
//     ω₃,ω₄,ω₅ optimized from ω₁,ω₂ (Eq.14-16 of Gehrke & Rung 2022).
//     4th-order equilibria use A,B coefficients (Eq.17-18).
//     Regularization limiter λ (Eq.20-26) for stability control.
//     Superior accuracy at moderate grids; λ tunes DNS↔VLES blend.
//
// Interface:
//   INPUT:  f_in[27], omega(=ω₁), dt, Fx, Fy, Fz
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
// WP regularization parameter λ  (only used when USE_WP_CUMULANT=1)
//   λ_def = 1e-2  (Gehrke default, good for most cases)
//   λ = 1e-6      effectively → AO (regularization off)
//   λ = 1e-1~1e0  optimal for Re≥10600 on medium grids (Table 7, GR22)
// ================================================================
#ifndef CUM_LAMBDA
#define CUM_LAMBDA 1.0e-2
#endif

// Forward declarations
__device__ static void _cum_forward_chimera(double m[27], const double u[3]);
__device__ static void _cum_backward_chimera(double m[27], const double u[3]);

#if USE_WP_CUMULANT
// ================================================================
// WP helper: compute parameterized ω₃,ω₄,ω₅ from ω₁,ω₂
//   [GR22] Eq. 14-16, derived from 4th-order diffusion error optimization
// ================================================================
__device__ static void _cum_wp_compute_omega345(
    const double w1, const double w2,
    double& w3, double& w4, double& w5)
{
    // Eq. 14: ω₃
    double num3  = 8.0 * (w1 - 2.0) * (w2 * (3.0*w1 - 1.0) - 5.0*w1);
    double den3  = 8.0 * (5.0 - 2.0*w1) * w1
                 + w2 * (8.0 + w1 * (9.0*w1 - 26.0));
    w3 = num3 / den3;

    // Eq. 15: ω₄
    double num4  = 8.0 * (w1 - 2.0) * (w1 + w2 * (3.0*w1 - 7.0));
    double den4  = w2 * (56.0 - 42.0*w1 + 9.0*w1*w1) - 8.0*w1;
    w4 = num4 / den4;

    // Eq. 16: ω₅
    double num5  = 24.0 * (w1 - 2.0) * (4.0*w1*w1
                 + w1*w2*(18.0 - 13.0*w1)
                 + w2*w2*(2.0 + w1*(6.0*w1 - 11.0)));
    double den5  = 16.0*w1*w1*(w1 - 6.0)
                 - 2.0*w1*w2*(216.0 + 5.0*w1*(9.0*w1 - 46.0))
                 + w2*w2*(w1*(3.0*w1 - 10.0)*(15.0*w1 - 28.0) - 48.0);
    w5 = num5 / den5;
}

// ================================================================
// WP helper: compute A, B coefficients for 4th-order equilibria
//   [GR22] Eq. 17-18
// ================================================================
__device__ static void _cum_wp_compute_AB(
    const double w1, const double w2,
    double& A, double& B)
{
    double denom = (w1 - w2) * (w2 * (2.0 + 3.0*w1) - 8.0*w1);

    // Eq. 17: A
    A = (4.0*w1*w1 + 2.0*w1*w2*(w1 - 6.0)
       + w2*w2*(w1*(10.0 - 3.0*w1) - 4.0)) / denom;

    // Eq. 18: B
    B = (4.0*w1*w2*(9.0*w1 - 16.0) - 4.0*w1*w1
       - 2.0*w2*w2*(2.0 + 9.0*w1*(w1 - 2.0)))
       / (3.0 * denom);
}

// ================================================================
// WP helper: apply λ-limiter to a single relaxation rate
//   ω^λ = ω_base + (1 - ω_base)|C_mag| / (ρλ + |C_mag|)
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
//  ★ Main entry point ★
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
    const double omega,      // INPUT:  shear relaxation rate ω₁
    const double dt,         // INPUT:  time step (for half-force correction)
    const double Fx,         // INPUT:  body force x
    const double Fy,         // INPUT:  body force y
    const double Fz,         // INPUT:  body force z
    double       f_out[27],  // OUTPUT: post-collision distributions
    double&      rho_out,    // OUTPUT: density
    double&      ux_out,     // OUTPUT: velocity x (half-force corrected)
    double&      uy_out,     // OUTPUT: velocity y
    double&      uz_out      // OUTPUT: velocity z
)
{
    // ==============================================================
    // STAGE 0: Macroscopic Quantities + Well-Conditioning
    // ==============================================================

    // 0a. Density: ρ = Σ f_α
    double rho = 0.0;
    for (int i = 0; i < 27; i++) rho += f_in[i];

    // 0b. Momentum: j = Σ f_α · e_α
    double jx = 0.0, jy = 0.0, jz = 0.0;
    for (int i = 0; i < 27; i++) {
        jx += f_in[i] * GILBM_e[i][0];
        jy += f_in[i] * GILBM_e[i][1];
        jz += f_in[i] * GILBM_e[i][2];
    }

    // 0c. Half-force corrected velocity (Guo 2002)
    double inv_rho = 1.0 / rho;
    double u[3];
    u[0] = jx * inv_rho + 0.5 * Fx * inv_rho * dt;
    u[1] = jy * inv_rho + 0.5 * Fy * inv_rho * dt;
    u[2] = jz * inv_rho + 0.5 * Fz * inv_rho * dt;

    // 0d. Well-conditioning: f̄ = f - w
    double m[27];
    for (int i = 0; i < 27; i++) {
        m[i] = f_in[i] - GILBM_W[i];
    }

    // 0e. Auxiliary
    double drho = rho - 1.0;

    // ==============================================================
    // STAGE 1: Forward Chimera Transform (z → y → x)
    //          f̄[27] → κ[27] (central moments)
    //          [G15] Appendix J, Eq. J.4–J.12
    // ==============================================================
    _cum_forward_chimera(m, u);

    // ==============================================================
    // STAGE 2: Central Moments → Cumulants
    //          Orders 0–3: C = κ (identical)
    //          Orders 4–6: subtract lower-order products
    //          [G15] Appendix J, Eq. J.16–J.19
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
    // ┌─────────────────────────────────────────────────────────┐
    // │  AO mode (USE_WP_CUMULANT=0):                          │
    // │    ω₂ = ω₃ = ω₄ = ... = ω₁₀ = 1.0                    │
    // │    All non-equilibrium parts fully damped.              │
    // │    4th-order cumulant equilibria = 0.                   │
    // │    Simple, robust, but accuracy-limited. [G15]          │
    // │                                                         │
    // │  WP mode (USE_WP_CUMULANT=1):                          │
    // │    ω₂ = 1.0 (bulk viscosity, user choice)              │
    // │    ω₃,ω₄,ω₅ = f(ω₁,ω₂) [GR22 Eq.14-16]              │
    // │    → then λ-limited [GR22 Eq.20-26]                    │
    // │    4th-order equilibria ≠ 0, use A,B [GR22 Eq.17-18]   │
    // │    ω₆=ω₇=ω₈ = 1.0, ω₉ = 1.0, ω₁₀ = 1.0             │
    // └─────────────────────────────────────────────────────────┘

    // ---- ω₂: bulk viscosity (same for both modes) ----
    const double omega2 = 1.0;

    // ---- ω₆=ω₇=ω₈: 4th order, ω₉: 5th order, ω₁₀: 6th order ----
    const double omega6  = 1.0;
    const double omega9  = 1.0;
    const double omega10 = 1.0;

#if USE_WP_CUMULANT
    // ============================================================
    //  WP MODE: Parameterized relaxation [GR22]
    // ============================================================

    // Step A: Compute base ω₃,ω₄,ω₅ from ω₁,ω₂ (Eq.14-16)
    double omega3_base, omega4_base, omega5_base;
    _cum_wp_compute_omega345(omega, omega2, omega3_base, omega4_base, omega5_base);

    // Step B: Compute A, B for 4th-order equilibria (Eq.17-18)
    double coeff_A, coeff_B;
    _cum_wp_compute_AB(omega, omega2, coeff_A, coeff_B);

    // Step C: Extract raw 3rd-order cumulants BEFORE decomposition
    //         (needed for λ-limiter, Eq.20-26)
    double C_120 = m[I_bca];   // κ₁₂₀ = C₁₂₀ (orders ≤3: C=κ)
    double C_102 = m[I_bac];   // κ₁₀₂ = C₁₀₂
    double C_210 = m[I_cba];   // κ₂₁₀ = C₂₁₀
    double C_012 = m[I_abc];   // κ₀₁₂ = C₀₁₂
    double C_201 = m[I_cab];   // κ₂₀₁ = C₂₀₁
    double C_021 = m[I_acb];   // κ₀₂₁ = C₀₂₁
    double C_111 = m[I_bbb];   // κ₁₁₁ = C₁₁₁

    // Step D: Apply λ-limiter to get effective ω^λ (Eq.20-26)
    const double lam = CUM_LAMBDA;

    // Eq.20: ω^λ_{3,1} for (C₁₂₀ + C₁₀₂) symmetric pair
    double omega3_1 = _cum_wp_limit(omega3_base, C_120 + C_102, rho, lam);
    // Eq.21: ω^λ_{4,1} for (C₁₂₀ - C₁₀₂) antisymmetric pair
    double omega4_1 = _cum_wp_limit(omega4_base, C_120 - C_102, rho, lam);
    // Eq.22: ω^λ_{3,2} for (C₂₁₀ + C₀₁₂)
    double omega3_2 = _cum_wp_limit(omega3_base, C_210 + C_012, rho, lam);
    // Eq.23: ω^λ_{4,2} for (C₂₁₀ - C₀₁₂)
    double omega4_2 = _cum_wp_limit(omega4_base, C_210 - C_012, rho, lam);
    // Eq.24: ω^λ_{3,3} for (C₂₀₁ + C₀₂₁)
    double omega3_3 = _cum_wp_limit(omega3_base, C_201 + C_021, rho, lam);
    // Eq.25: ω^λ_{4,3} for (C₂₀₁ - C₀₂₁)
    double omega4_3 = _cum_wp_limit(omega4_base, C_201 - C_021, rho, lam);
    // Eq.26: ω^λ_5 for C₁₁₁
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

    // Relax trace with ω₂ toward δρ equilibrium
    mxxPyyPzz += omega2 * (m[I_aaa] - mxxPyyPzz);
    // Relax deviatorics with ω₁ toward 0
    mxxMyy *= (1.0 - omega);
    mxxMzz *= (1.0 - omega);

    // Off-diagonal 2nd order with ω₁
    m[I_abb] *= (1.0 - omega);  // C₀₁₁
    m[I_bab] *= (1.0 - omega);  // C₁₀₁
    m[I_bba] *= (1.0 - omega);  // C₁₁₀

    // ---- 3rd order: symmetric / antisymmetric decomposition ----
    double mxxyPyzz = m[I_cba] + m[I_abc];   // C₂₁₀ + C₀₁₂
    double mxxyMyzz = m[I_cba] - m[I_abc];   // C₂₁₀ - C₀₁₂
    double mxxzPyyz = m[I_cab] + m[I_acb];   // C₂₀₁ + C₀₂₁
    double mxxzMyyz = m[I_cab] - m[I_acb];   // C₂₀₁ - C₀₂₁
    double mxyyPxzz = m[I_bca] + m[I_bac];   // C₁₂₀ + C₁₀₂
    double mxyyMxzz = m[I_bca] - m[I_bac];   // C₁₂₀ - C₁₀₂

#if USE_WP_CUMULANT
    // WP: each pair has its OWN λ-limited relaxation rate
    m[I_bbb]  *= (1.0 - omega5_lim);    // C₁₁₁ (Eq.26)
    mxxyPyzz  *= (1.0 - omega3_2);      // (C₂₁₀+C₀₁₂) symmetric (Eq.22)
    mxxyMyzz  *= (1.0 - omega4_2);      // (C₂₁₀-C₀₁₂) antisymmetric (Eq.23)
    mxxzPyyz  *= (1.0 - omega3_3);      // (C₂₀₁+C₀₂₁) symmetric (Eq.24)
    mxxzMyyz  *= (1.0 - omega4_3);      // (C₂₀₁-C₀₂₁) antisymmetric (Eq.25)
    mxyyPxzz  *= (1.0 - omega3_1);      // (C₁₂₀+C₁₀₂) symmetric (Eq.20)
    mxyyMxzz  *= (1.0 - omega4_1);      // (C₁₂₀-C₁₀₂) antisymmetric (Eq.21)
#else
    // AO: all = 1 → (1-ω) = 0 → full relaxation to zero
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
    //   C^eq_{αα,ββ} involves products of 2nd-order cumulants via A, B
    //   [GR22] Eq.17-18, Appendix B.2
    //
    // 2nd-order cumulants (post-relaxation, well-conditioned):
    //   Dxx = m[I_caa], Dyy = m[I_aca], Dzz = m[I_aac]
    //   Dxy = m[I_bba], Dxz = m[I_bab], Dyz = m[I_abb]
    //
    // Diagonal 4th-order equilibria (Geier 2017, Appendix B.2):
    //   C^eq_{2200} = (A+B)(Dxx·Dyy + Dxy²) + (A-B)(Dxx·Dyy - Dxy²)
    //               = A·(Dxx·Dyy + Dxy²) + B·(Dxx·Dyy - Dxy²)  ... WRONG
    //   Actually per Geier:
    //   C^eq_{xxyy} / ρ = A·(Dxx·Dyy/ρ + Dxy²/ρ) + B·(Dxx·Dyy/ρ - Dxy²/ρ)  (with /ρ)
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
    // STAGE 4: Cumulants → Central Moments (Inverse of Stage 2)
    //          [G15] Appendix J, Eq. J.16–J.19 inverse
    // ==============================================================

    // --- 4th order off-diagonal inverse (Eq. J.16⁻¹) ---
    m[I_cbb] = CUMcbb + ((m[I_caa] + 1.0/3.0)*m[I_abb]
               + 2.0*m[I_bba]*m[I_bab]) * inv_rho;
    m[I_bcb] = CUMbcb + ((m[I_aca] + 1.0/3.0)*m[I_bab]
               + 2.0*m[I_bba]*m[I_abb]) * inv_rho;
    m[I_bbc] = CUMbbc + ((m[I_aac] + 1.0/3.0)*m[I_bba]
               + 2.0*m[I_bab]*m[I_abb]) * inv_rho;

    // --- 4th order diagonal inverse (Eq. J.17⁻¹) ---
    m[I_cca] = CUMcca + (((m[I_caa]*m[I_aca]+2.0*m[I_bba]*m[I_bba])*9.0
               + 3.0*(m[I_caa]+m[I_aca])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;
    m[I_cac] = CUMcac + (((m[I_caa]*m[I_aac]+2.0*m[I_bab]*m[I_bab])*9.0
               + 3.0*(m[I_caa]+m[I_aac])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;
    m[I_acc] = CUMacc + (((m[I_aac]*m[I_aca]+2.0*m[I_abb]*m[I_abb])*9.0
               + 3.0*(m[I_aac]+m[I_aca])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;

    // --- 5th order inverse (Eq. J.18⁻¹) ---
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

    // --- 6th order inverse (Eq. J.19⁻¹) ---
    // ★ BUG FIX: coefficient for (acc+cac+cca)/ρ is -1/3, matching forward ★
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
        - 1.0/3.0*(m[I_acc]+m[I_cac]+m[I_cca]) * inv_rho    // ★ FIXED: was 1/9
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
    // STAGE 5: Backward Chimera Transform (x → y → z)
    //          [G15] Appendix J, Eq. J.20–J.28
    // ==============================================================
    _cum_backward_chimera(m, u);

    // Restore from well-conditioned: f* = f̄* + w
    for (int i = 0; i < 27; i++) {
        f_out[i] = m[i] + GILBM_W[i];
    }

    // Output macroscopic quantities
    rho_out = rho;
    ux_out  = u[0];
    uy_out  = u[1];
    uz_out  = u[2];
}


// ================================================================
// Internal: Forward Chimera (Stage 1)
// Sweep order: z(dir=2) → y(dir=1) → x(dir=0)
// [G15] Appendix J, Eq. J.4–J.12
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
// Sweep order: x(dir=0) → y(dir=1) → z(dir=2)
// [G15] Appendix J, Eq. J.20–J.28
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
