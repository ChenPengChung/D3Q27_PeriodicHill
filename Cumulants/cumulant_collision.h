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
//     w3,w4,w5 optimized from w1,w2 (Eq.3.103-3.105 / [GR22] Eq.14-16).
//     4th-order diagonal: A·velocity_derivative form (Thesis Eq.3.83-3.85).
//     4th-order off-diagonal: B·velocity_derivative form (Thesis Eq.3.86-3.88).
//     Regularization limiter lambda (Eq.3.111-3.114 / [GR22] Eq.20-26).
//     ω₆,ω₇,ω₈ retained as parameters (override via CUM_OMEGA6/7/8).
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

// ── Compile-time diagnostic: verify macro visibility ──
#if USE_WP_CUMULANT
  #ifdef CUM_WP_OMEGA_MIN
    #pragma message(">>> CUM_WP_OMEGA_MIN is DEFINED = active")
  #else
    #pragma message(">>> CUM_WP_OMEGA_MIN is NOT defined = original WP behavior")
  #endif
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

// Forward declarations (Chimera transform — restored from Option A revert)
__device__ static void _cum_forward_chimera(double m[27], const double u[3]);
__device__ static void _cum_backward_chimera(double m[27], const double u[3]);

#if USE_WP_CUMULANT
// ================================================================
// WP helper: compute parameterized w3,w4,w5 from w1,w2
//   [GR22] Eq. 14-16, derived from 4th-order diffusion error optimization
//
//   SINGULARITY HANDLING:
//   Eq.15 denominator = 0 at w1 = 14/9 ≈ 1.5556 (for w2=1).
//   Near this pole, the parametrization formula produces omega values
//   far outside the stability interval [0, 2].
//
//   Strategy: compute raw omega from Eq.14-16. If the result is outside
//   [0, 2], the parametrization is in its divergent regime and cannot
//   be trusted. Fall back to omega=1.0 (AO neutral value = full damping
//   to equilibrium). This is the SAFE default that Geier 2015 uses for
//   ALL higher-order rates.
//
//   The fallback is SMOOTH: we use a blending function that transitions
//   from the parametrized value to 1.0 as we approach the instability
//   boundary, rather than a hard switch.
//
//   Blending: for omega_raw ∈ [0, 2], use it directly.
//   For omega_raw outside [0, 2]:
//     dist = min(|omega_raw - 0|, |omega_raw - 2|) / BLEND_WIDTH
//     omega = lerp(omega_raw_clamped, 1.0, saturate(dist))
//   This provides C0 continuity at the boundary.
// ================================================================
__device__ static double _cum_wp_clamp_with_ao_fallback(double omega_raw)
{
    // Outside [0,2]: fall back to AO neutral (1.0)
    // The parametrization is unreliable at this w1.
    if (omega_raw < 0.0 || omega_raw > 2.0) {
        return 1.0;
    }

    // ── GILBM safety: enforce minimum damping for 3rd-order cumulants ──
    // When ω₁ crosses the ω₄ pole at 14/9 ≈ 1.556, the Eq.15 formula
    // produces ω₄ → 0 (≈0.067 at ω₁=1.80), meaning antisymmetric
    // 3rd-order modes retain 93% per step — almost no damping.
    // In GILBM, interpolation noise continuously feeds 3rd-order cumulants.
    // Without sufficient damping, this noise accumulates → divergence.
    //
    // Fix: clamp ω₃,ω₄,ω₅ to [CUM_WP_OMEGA_MIN, 2.0].
    // CUM_WP_OMEGA_MIN = 0.5 → at least 50% damping per step.
    // This preserves the WP parametrization where it's well-behaved,
    // and falls back toward AO-like damping in the near-pole region.
#ifdef CUM_WP_OMEGA_MIN
    if (omega_raw < CUM_WP_OMEGA_MIN) {
        return CUM_WP_OMEGA_MIN;
    }
#endif

    return omega_raw;
}

__device__ static void _cum_wp_compute_omega345(
    const double w1, const double w2,
    double *w3, double *w4, double *w5)
{
    // Denominator guard: only for exact zero (numerical safety)
    const double DEN_EPS = 1.0e-10;

    // Eq. 14: w3
    double num3  = 8.0 * (w1 - 2.0) * (w2 * (3.0*w1 - 1.0) - 5.0*w1);
    double den3  = 8.0 * (5.0 - 2.0*w1) * w1
                 + w2 * (8.0 + w1 * (9.0*w1 - 26.0)); //分母為0，w1 = 2.46 和 -0.46
    double w3_raw = (fabs(den3) > DEN_EPS) ? num3 / den3 : 1.0;
    *w3 = _cum_wp_clamp_with_ao_fallback(w3_raw);

    // Eq. 15: w4  (den4=0 at w1=14/9 ≈ 1.5556 for w2=1)
    double num4  = 8.0 * (w1 - 2.0) * (w1 + w2 * (3.0*w1 - 7.0));
    double den4  = w2 * (56.0 - 42.0*w1 + 9.0*w1*w1) - 8.0*w1;
    double w4_raw = (fabs(den4) > DEN_EPS) ? num4 / den4 : 1.0;
    *w4 = _cum_wp_clamp_with_ao_fallback(w4_raw);

    // Eq. 16: w5
    double num5  = 24.0 * (w1 - 2.0) * (4.0*w1*w1
                 + w1*w2*(18.0 - 13.0*w1)
                 + w2*w2*(2.0 + w1*(6.0*w1 - 11.0)));
    double den5  = 16.0*w1*w1*(w1 - 6.0)
                 - 2.0*w1*w2*(216.0 + 5.0*w1*(9.0*w1 - 46.0))
                 + w2*w2*(w1*(3.0*w1 - 10.0)*(15.0*w1 - 28.0) - 48.0);
    double w5_raw = (fabs(den5) > DEN_EPS) ? num5 / den5 : 1.0;
    *w5 = _cum_wp_clamp_with_ao_fallback(w5_raw);
}
//統一以 omega2 = 1.0 為基準 (預設值，用於零點分析)
//統計穩定度設計 : 該因子的分母不可以為0
//輸入參數 : Re , U_ref , dt_global 輸出 w1 , w3 , w4 , w5  , A , B , 並輸出 : "分母為0發散風險 , 需要重新設置參數 (U_ref基本要調整 )(因為Re 以及 變換度量 固定於此模型)"
//omega1 = Re , U_{ref} , Jacobian 決定 ， 本身無奇異點, 物理保證 w1 ∈ (0,2)
//omega2 = CUM_OMEGA2 (由 variables.h 統一指定), 以下零點以 w2=1.0 為基準列出
//omega_3 = 分母零點為 w1 = 2.46 和 -0.46        → 皆在 (0,2) 外, ★安全★
//omega_4 = 分母零點為 w1 = 14/9 ≈ 1.5556 (w2=1)  → 在 (0,2) 內, ★★危險★★ 唯一需注意的奇異點
//omega_5 = 分母零點為 w1 = -0.256 (實根) 和兩個共軛複數根 → 皆在 (0,2) 外, ★安全★
//           den5 = -29*w1³ + 130*w1² - 152*w1 - 48 (w2=1 代入)
//A,B 共用分母 = (w1 - w2)*(w2*(2+3*w1) - 8*w1)
//  A ; 分母零點為 w1 = w2 = 1.0  和  w1 = 2*w2/(8-3*w2) = 0.4 (w2=1)  → 皆在 (0,2) 內, ★需注意★
//  B ; 分母零點同 A (共用分母), 額外除以 3 不影響零點位置
//  ► 實際 omega2 由 CUM_OMEGA2 決定，零點位置會偏移，詳見 test_cumulant_wp_diagnostic 診斷輸出
//  ► A,B 奇異點距離: |w1 - 1.0| 和 |w1 - 0.4|, 目前 w1≈0.63 → 距 0.4 僅 0.23, 距 1.0 為 0.37
//  ► fallback: denom < DEN_EPS 時 A=B=0 (退化為 AO 模式, 4 階平衡態為零)
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
    const double omega_tau,  // INPUT:  relaxation TIME τ = 3ν/dt + 0.5 (convention: same as BGK's omega_global)
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
    // CONVENTION FIX: omega_global from main.cu is τ (relaxation TIME),
    //   τ = 3ν/dt + 0.5.  BGK uses 1/τ; cumulant needs ω₁ = 1/τ.
    //   Before this fix, code used τ directly as ω₁, giving:
    //     (1-τ) ≈ 0.28 instead of (1-1/τ) ≈ -0.39
    //     → ν_eff ≈ 4×ν → Re_eff ≈ Re/4  (wrong physics, but stable)
    // ==============================================================
    //omega_tau為Imamura鬆弛時間
    const double omega = 1.0 / omega_tau;   // ω₁ = 1/τ (shear relaxation RATE)
    //設置omega1在這裡
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
    //          Combined raw moment + binomial shift per triplet.
    //          [G15] Appendix J / Gehrke Thesis §3.1.3
    //
    //          NOTE: Option A (global M/S separation) was tested and proven
    //          INCORRECT — the in-place array overwrites between directions
    //          invalidate K constants, producing wrong central moments.
    //          Reverted to original Chimera (2025-03-15).
    // ==============================================================
    _cum_forward_chimera(m, u);

    // ==============================================================
    // STAGE 2: Central Moments -> Cumulants
    //          Orders 03: C = kappa (identical)
    //          Orders 46: subtract lower-order products
    //          [G15] Appendix J, Eq. J.16J.19
    // ==============================================================

    // --- 4th order off-diagonal (Eq. J.16) ---
    // Both modes need these for:
    //   AO: standard relaxation + back-conversion
    //   WP: general ω₈ formula (Eq.3.86-3.88) + back-conversion (Eq.J.16 inverse)
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
    // 使用 variables.h 中的全域巨集 CUM_OMEGA2
    // (預設 0.5，可在 variables.h 統一修改)
#ifdef CUM_OMEGA2
    const double omega2 = CUM_OMEGA2;
#else
    const double omega2 = 0.5;   // fallback: verify/ 測試檔未 include variables.h 時
#endif

    // ---- ω₆/ω₇/ω₈: 4th-order relaxation rates [Gehrke Thesis p.44-45, Eq.3.83-3.88] ----
    //   ω₆: 4th-order deviatoric (Eq.3.83-3.85, deviatoric linear combos)
    //   ω₇: 4th-order trace      (Eq.3.83-3.85, trace linear combo)
    //   ω₈: 4th-order off-diagonal (Eq.3.86-3.88)
    //   Thesis §3.2.3 (p.47): WP variant sets ω_{C>5}=1.
    //   Override via CUM_OMEGA6/7/8 in variables.h for non-unity values.
#ifdef CUM_OMEGA6
    const double omega6  = CUM_OMEGA6;
#else
    const double omega6  = 1.0;    // 4th-order deviatoric (default: thesis WP/AO)
#endif
#ifdef CUM_OMEGA7
    const double omega7  = CUM_OMEGA7;
#else
    const double omega7  = 1.0;    // 4th-order trace (default: thesis WP/AO)
#endif
#ifdef CUM_OMEGA8
    const double omega8  = CUM_OMEGA8;
#else
    const double omega8  = 1.0;    // 4th-order off-diagonal (default: thesis WP/AO)
#endif
    // ---- ω₉: 5th order, ω₁₀: 6th order ----
#ifdef CUM_OMEGA9
    const double omega9  = CUM_OMEGA9;
#else
    const double omega9  = 1.0;
#endif
#ifdef CUM_OMEGA10
    const double omega10 = CUM_OMEGA10;
#else
    const double omega10 = 1.0;
#endif

#if USE_WP_CUMULANT
    // ============================================================
    //  WP MODE: Parameterized relaxation [GR22]
    // ============================================================

    // Step A: Compute base w3,w4,w5 from w1,w2 (Eq.14-16)
    //   _cum_wp_compute_omega345 now includes AO-fallback for singularity:
    //   when Eq.14-16 produce values outside [0,2], falls back to 1.0 (AO neutral).
    //   No additional [0,2] clamp needed here.
    double omega3_base, omega4_base, omega5_base;
    _cum_wp_compute_omega345(omega, omega2, &omega3_base, &omega4_base, &omega5_base);

    // Step B: Compute A, B for 4th-order equilibria (Eq.17-18)
    double coeff_A, coeff_B;
    _cum_wp_compute_AB(omega, omega2, &coeff_A, &coeff_B);

    // ── ONE-TIME DIAGNOSTIC: print WP parameters at first call ──
    // Thread (0,0,0) in block (0,0,0) prints once, then flag prevents repeats.
    {
        __shared__ int printed;
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printed = 0;
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
            && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            if (atomicCAS(&printed, 0, 1) == 0) {
                printf("[CUM_WP_DIAG] omega1=%.6f omega2=%.6f\n", omega, omega2);
                printf("[CUM_WP_DIAG] omega3_base=%.6f omega4_base=%.6f omega5_base=%.6f\n",
                       omega3_base, omega4_base, omega5_base);
                printf("[CUM_WP_DIAG] A=%.6f B=%.6f\n", coeff_A, coeff_B);
                printf("[CUM_WP_DIAG] rho=%.6f u=(%.6e, %.6e, %.6e) Ma=%.6f\n",
                       rho, u[0], u[1], u[2],
                       sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]) / (1.0/1.732050807568877));
#ifdef CUM_WP_OMEGA_MIN
                printf("[CUM_WP_DIAG] CUM_WP_OMEGA_MIN=%.4f *** ACTIVE ***\n", (double)CUM_WP_OMEGA_MIN);
#else
                printf("[CUM_WP_DIAG] CUM_WP_OMEGA_MIN not defined (original WP)\n");
#endif
#ifdef CUM_ODDEVEN_SIGMA
                printf("[CUM_WP_DIAG] CUM_ODDEVEN_SIGMA=%.4f\n", (double)CUM_ODDEVEN_SIGMA);
#endif
            }
        }
    }

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

    // ---- Galilean correction [Gehrke Thesis p.56, Eq.3.70-3.75] ----
    // Velocity derivatives from PRE-relaxation 2nd-order cumulants.
    // Using well-conditioned variables directly: the ±1/3 shifts cancel
    // in both ω₁ and ω₂ terms, so m[I_xxx] can be used as-is.
    //
    // ★ GILBM WARNING: Galilean correction contains u² terms. When u has
    //   interpolation noise δu, the correction injects energy ~ u·δu into
    //   2nd-order moments → positive feedback → exponential instability.
    //   Controlled by CUM_GALILEAN in variables.h (0=disable for GILBM stability).
    //
    // Eq.3.73-3.75 (walberla CellwiseSweep.impl.h lines 270-272):
    // Velocity derivatives needed by Galilean correction AND/OR WP 4th-order
#if CUM_GALILEAN || USE_WP_CUMULANT
    double Dxux = -0.5*omega*inv_rho*(2.0*m[I_caa] - m[I_aca] - m[I_aac])
                 - 0.5*omega2*inv_rho*(mxxPyyPzz - drho);
    double Dyuy = Dxux + 1.5*omega*inv_rho*(m[I_caa] - m[I_aca]);
    double Dzuz = Dxux + 1.5*omega*inv_rho*(m[I_caa] - m[I_aac]);
#endif

#if CUM_GALILEAN
    // Eq.3.70-3.72: Galilean correction terms
    // (walberla CellwiseSweep.impl.h lines 274-276)
    double GalCorr_dev1  = -3.0*rho*(1.0-0.5*omega )*(Dxux*u[0]*u[0] - Dyuy*u[1]*u[1]);
    double GalCorr_dev2  = -3.0*rho*(1.0-0.5*omega )*(Dxux*u[0]*u[0] - Dzuz*u[2]*u[2]);
    double GalCorr_trace = -3.0*rho*(1.0-0.5*omega2)*(Dxux*u[0]*u[0] + Dyuy*u[1]*u[1] + Dzuz*u[2]*u[2]);

    // Relax with Galilean correction
    mxxMyy    = (1.0-omega)*mxxMyy + GalCorr_dev1;      // deviatoric 1 + Gal corr
    mxxMzz    = (1.0-omega)*mxxMzz + GalCorr_dev2;      // deviatoric 2 + Gal corr
    mxxPyyPzz = omega2*m[I_aaa] + (1.0-omega2)*mxxPyyPzz + GalCorr_trace;  // trace + Gal corr
#else
    // No Galilean correction: standard relaxation (safe for GILBM)
    mxxMyy    = (1.0-omega)*mxxMyy;
    mxxMzz    = (1.0-omega)*mxxMzz;
    mxxPyyPzz = omega2*m[I_aaa] + (1.0-omega2)*mxxPyyPzz;
#endif

    // Off-diagonal 2nd order with w1
#if USE_WP_CUMULANT
    // [GR22 B26-B28 FIX] Save pre-relaxation off-diagonal 2nd-order values.
    // Paper B26: C*_{211} = (1-w1/2)*B*C_{011}  uses PRE-relaxation C_{011}.
    const double saved_C011 = m[I_abb];  // pre-relaxation C_{yz}
    const double saved_C101 = m[I_bab];  // pre-relaxation C_{xz}
    const double saved_C110 = m[I_bba];  // pre-relaxation C_{xy}
#endif
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
    // ================================================================
    // WP: 4th-order DIAGONAL collision — Thesis Eq. 3.83-3.85 (original form)
    //
    //   Uses velocity derivatives (∂_xu, ∂_yv, ∂_zw) and parameter A only.
    //   This is the GENERAL form valid for ANY ω₆, ω₇ values.
    //
    //   Structure per mode (deviatoric/trace):
    //     mode* = equilibrium_term(A, vel_derivs) + (1-ω)·mode_pre
    //
    //   The velocity derivatives Dxux, Dyuy, Dzuz were computed at Eq.3.73-3.75
    //   BEFORE 2nd-order relaxation, using PRE-relaxation cumulants (correct).
    // ================================================================
    {
        // Velocity-derivative equilibrium coefficient: (1/ω₁ - 1/2)·A·ρ
        const double vel_eq_A = (1.0/omega - 0.5) * coeff_A * rho;

        // Eq. 3.83-3.85 equilibrium terms for each orthogonal mode:
        //   deviatoric 1: 2/3·vel_eq_A·(∂_xu - 2∂_yv + ∂_zw)
        //   deviatoric 2: 2/3·vel_eq_A·(∂_xu + ∂_yv - 2∂_zw)
        //   trace:       -4/3·vel_eq_A·(∂_xu + ∂_yv + ∂_zw)
        const double eq4_dev1  =  2.0/3.0 * vel_eq_A * (Dxux - 2.0*Dyuy + Dzuz);
        const double eq4_dev2  =  2.0/3.0 * vel_eq_A * (Dxux + Dyuy - 2.0*Dzuz);
        const double eq4_trace = -4.0/3.0 * vel_eq_A * (Dxux + Dyuy + Dzuz);

        // Pre-collision deviatoric/trace decomposition of 4th-order diagonal cumulants
        double cum4_dev1  = CUMcca - 2.0*CUMcac + CUMacc;
        double cum4_dev2  = CUMcca + CUMcac - 2.0*CUMacc;
        double cum4_trace = CUMcca + CUMcac + CUMacc;

        // Thesis Eq. 3.83-3.85 (original form, NO simplification):
        //   mode* = eq_term + (1 - ω)·mode_pre
        cum4_dev1  = eq4_dev1  + (1.0 - omega6) * cum4_dev1;    // deviatoric (ω₆)
        cum4_dev2  = eq4_dev2  + (1.0 - omega6) * cum4_dev2;    // deviatoric (ω₆)
        cum4_trace = eq4_trace + (1.0 - omega7) * cum4_trace;   // trace (ω₇)

        // Reconstruct individual 4th-order diagonal cumulants
        CUMcca = (cum4_dev1 + cum4_dev2 + cum4_trace) / 3.0;
        CUMcac = (cum4_trace - cum4_dev1) / 3.0;
        CUMacc = (cum4_trace - cum4_dev2) / 3.0;
    }

    // Off-diagonal 4th-order: General Eq. 3.86-3.88 [Gehrke Thesis p.45]
    //   C*_αβγ = -1/3(ω₁/2-1)·ω₈·B·ρ·(velocity_derivative) + (1-ω₈)·C_αβγ
    //
    // Step 1: Velocity derivatives from Eq. 3.89 (CORRECTED subscripts):
    //   Thesis Eq. 3.89 has C_110 ↔ C_011 SWAPPED (typo).
    //   Correct mapping (from Chapman-Enskog analysis):
    //     ∂_y w + ∂_z v = -3ω₁/ρ · C_011  (yz shear → C_011)
    //     ∂_x w + ∂_z u = -3ω₁/ρ · C_101  (xz shear → C_101)
    //     ∂_x v + ∂_y u = -3ω₁/ρ · C_110  (xy shear → C_110)
    const double dywPdzv = -3.0 * omega * inv_rho * saved_C011;  // ∂_yw+∂_zv (Eq.3.89 corrected)
    const double dxwPdzu = -3.0 * omega * inv_rho * saved_C101;  // ∂_xw+∂_zu
    const double dxvPdyu = -3.0 * omega * inv_rho * saved_C110;  // ∂_xv+∂_yu (Eq.3.89 corrected)

    // Step 2: General collision Eq. 3.86-3.88 with ω₈ retained
    //   When ω₈=1: (1-ω₈)·C = 0, reduces to Eq. 3.119-3.121
    const double eq_offdiag_coeff = -1.0/3.0 * (omega * 0.5 - 1.0) * omega8 * coeff_B * rho;
    const double wp_C211_star = eq_offdiag_coeff * dywPdzv + (1.0 - omega8) * CUMcbb;  // Eq.3.86
    const double wp_C121_star = eq_offdiag_coeff * dxwPdzu + (1.0 - omega8) * CUMbcb;  // Eq.3.87
    const double wp_C112_star = eq_offdiag_coeff * dxvPdyu + (1.0 - omega8) * CUMbbc;  // Eq.3.88
#else
    // AO: all 4th-order cumulants relax toward 0
    // Diagonal: ω₆ (deviatoric) and ω₇ (trace) [Thesis Eq.3.83-3.85]
    // Since ω₆=ω₇=1 in AO, all → 0
    {
        double cum4_dev1  = CUMcca - 2.0*CUMcac + CUMacc;
        double cum4_dev2  = CUMcca + CUMcac - 2.0*CUMacc;
        double cum4_trace = CUMcca + CUMcac + CUMacc;

        cum4_dev1  *= (1.0 - omega6);   // deviatoric (ω₆)
        cum4_dev2  *= (1.0 - omega6);   // deviatoric (ω₆)
        cum4_trace *= (1.0 - omega7);   // trace (ω₇)

        CUMcca = (cum4_dev1 + cum4_dev2 + cum4_trace) / 3.0;
        CUMcac = (cum4_trace - cum4_dev1) / 3.0;
        CUMacc = (cum4_trace - cum4_dev2) / 3.0;
    }
    // Off-diagonal: ω₈ [Thesis Eq.3.86-3.88, with eq=0 since A=B=0 in AO]
    CUMbbc *= (1.0 - omega8);
    CUMbcb *= (1.0 - omega8);
    CUMcbb *= (1.0 - omega8);
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
#if USE_WP_CUMULANT
    // [Thesis Eq.3.86-3.88 + Eq.J.16 inverse] Cumulant → central moment
    // FIXED: was missing low-order product terms (cumulant≠central moment for order≥4)
    m[I_cbb] = wp_C211_star + ((m[I_caa] + 1.0/3.0)*m[I_abb]
               + 2.0*m[I_bba]*m[I_bab]) * inv_rho;
    m[I_bcb] = wp_C121_star + ((m[I_aca] + 1.0/3.0)*m[I_bab]
               + 2.0*m[I_bba]*m[I_abb]) * inv_rho;
    m[I_bbc] = wp_C112_star + ((m[I_aac] + 1.0/3.0)*m[I_bba]
               + 2.0*m[I_bab]*m[I_abb]) * inv_rho;
#else
    m[I_cbb] = CUMcbb + ((m[I_caa] + 1.0/3.0)*m[I_abb]
               + 2.0*m[I_bba]*m[I_bab]) * inv_rho;
    m[I_bcb] = CUMbcb + ((m[I_aca] + 1.0/3.0)*m[I_bab]
               + 2.0*m[I_bba]*m[I_abb]) * inv_rho;
    m[I_bbc] = CUMbbc + ((m[I_aac] + 1.0/3.0)*m[I_bba]
               + 2.0*m[I_bab]*m[I_abb]) * inv_rho;
#endif

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

    // --- 1st-order moments: Sign flip for Strang-splitting body force ---
    // [Gehrke Thesis p.55, §3.2.1]:
    //   "the three first-order central moments (κ_ō=1) have to change sign
    //    prior to the back transformation in this case."
    //
    // Strang splitting (Geier 2015 Eq.85-87):
    //   1) Forward Chimera uses ũ = u + F·Δt/(2ρ)   → done at Stage 0c
    //   2) After collision, flip κ*_100, κ*_010, κ*_001 before backward Chimera
    //
    // ★ GILBM WARNING: Sign flip assumes exact lattice-shift streaming.
    //   GILBM interpolation streaming breaks this assumption. Use Guo explicit
    //   source (CUM_GUO_SRC=1, CUM_SIGNFLIP=0) for GILBM compatibility.
    //
    // Controlled by CUM_SIGNFLIP in variables.h
#if CUM_SIGNFLIP
    m[I_baa] = -m[I_baa];  // κ*_100 (x-momentum)
    m[I_aba] = -m[I_aba];  // κ*_010 (y-momentum)
    m[I_aab] = -m[I_aab];  // κ*_001 (z-momentum)
#endif

    // ==============================================================
    // STAGE 5: Backward Chimera Transform (x -> y -> z)
    //          kappa[27] -> f[27] (distributions)
    //          Combined inverse shift + inverse raw moment per triplet.
    //          [G15] Appendix J / Gehrke Thesis §3.1.3
    // ==============================================================
    _cum_backward_chimera(m, u);

    // Restore from well-conditioned: f* = f* + w
    for (int i = 0; i < 27; i++) {
        f_out[i] = m[i] + GILBM_W[i];
    }

    // ==============================================================
    // STAGE 6: Guo Forcing Source Term — DISABLED
    // ==============================================================
    // [Gehrke Thesis p.48, Eq.3.35-3.36]:
    //   "the body force is implicitly applied" through the modified
    //   equilibrium velocity ũ = u + F·Δt/(2ρ) entering the collision.
    //   No explicit Guo source term for MRT/cumulant models.
    //
    // Thesis p.48, Eq.3.36: f*_ζ(x,t) = f_ζ(x,t) + Ω_ζ(ρ,ũ)
    //   → collision uses ũ directly (Stage 0c), combined with sign flip
    //     of κ_ō=1 (above) to achieve Strang splitting.
    //
    // Previously this block added explicit Guo source: f* += (1-ω/2)·S·Δt
    // which DOUBLE-COUNTED the force when combined with sign flip → divergence.
    // ==============================================================
#if CUM_GUO_SRC
    // ── Guo explicit source (compatible with GILBM interpolation streaming) ──
    // Unlike sign-flip (Strang splitting), Guo source doesn't assume exact advection.
    // The (1-ω/2) prefactor handles the temporal offset between collision and streaming.
    //
    // For GILBM: CUM_SIGNFLIP=0, CUM_GUO_SRC=1 is recommended.
    // The half-force velocity ũ is still used in Stage 0c for the equilibrium,
    // but the force is added EXPLICITLY here rather than implicitly via sign flip.
    //
    // Note: u[] here is the half-force velocity from Stage 0c. For Guo source,
    // ideally we should use the bare momentum velocity u_bare = j/ρ. However,
    // the difference is O(F²·dt²) which is negligible.
    {
        double ax = Fx * inv_rho;
        double ay = Fy * inv_rho;
        double az = Fz * inv_rho;
        double a_dot_u = ax*u[0] + ay*u[1] + az*u[2];
        double prefactor = 1.0 - 0.5 * omega;

        for (int i = 0; i < 27; i++) {
            double e_dot_a = GILBM_e[i][0]*ax + GILBM_e[i][1]*ay + GILBM_e[i][2]*az;
            double e_dot_u = GILBM_e[i][0]*u[0] + GILBM_e[i][1]*u[1] + GILBM_e[i][2]*u[2];
            double S_i = GILBM_W[i] * rho * (3.0*e_dot_a*(1.0 + 3.0*e_dot_u) - 3.0*a_dot_u);
            f_out[i] += prefactor * S_i * delta_t;
        }
    }
#endif

    // Output macroscopic quantities
    *rho_out = rho;
    *ux_out  = u[0];
    *uy_out  = u[1];
    *uz_out  = u[2];
}


// ================================================================
// Internal: Forward Chimera Transform (Stage 1)
// Combined raw moment + binomial shift per triplet, swept z→y→x.
//
// For each triplet {a, b, c} (= {minus, zero, plus} in sweep direction):
//   sum  = f_a + f_c
//   diff = f_c - f_a
//   m[a] = f_a + f_b + f_c               (raw 0th moment = partial density)
//   m[b] = diff - (m[a] + K) · u         (1st central moment)
//   m[c] = sum - 2·diff·u + (m[a]+K)·u²  (2nd central moment)
//
// K[p] = well-conditioning constant calibrated for interleaved Chimera.
// After all 27 passes (9 z + 9 y + 9 x), m[27] holds central moments.
//
// Reference: [G15] Appendix J, Gehrke Thesis §3.1.3 Eq.3.23-3.29
//
// NOTE: Option A (global M/S separation) was tested 2025-03-15 and proven
// INCORRECT: global separation produces different central moments (O(1e-2))
// because in-place array overwrites between directions invalidate K constants.
// See testing/test_optionA_roundtrip.cpp for proof.
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
            m[a] = m[a] + m[b] + m[c];                              // raw m0
            m[b] = diff - (m[a] + k) * u[dir];                      // κ1
            m[c] = sum - 2.0 * diff * u[dir]
                   + (m[a] + k) * u[dir] * u[dir];                  // κ2
        }
    }
}

// ================================================================
// Internal: Backward Chimera Transform (Stage 5)
// Combined inverse shift + inverse raw moment per triplet, swept x→y→z.
//
// For each triplet {a, b, c} in reverse order:
//   fa = ((κ2 - κ1)/2 + κ1·u + (κ0+K)·(u²-u)/2)
//   fb = κ0 - κ2 - 2·κ1·u - (κ0+K)·u²
//   fc = ((κ2 + κ1)/2 + κ1·u + (κ0+K)·(u²+u)/2)
//
// This is S⁻¹(u)·M⁻¹_1d combined in a single pass.
// Reference: [G15] Appendix J, Gehrke Thesis §3.1.3
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
