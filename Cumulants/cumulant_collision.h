// ================================================================
// cumulant_collision.h
// D3Q27 Cumulant Collision — Standalone CUDA Device Function
//
// Interface:
//   INPUT:  f_in[27] (post-streaming), omega, dt, Fx, Fy, Fz
//   OUTPUT: f_out[27] (post-collision), rho, ux, uy, uz
//
// No feq needed. No M/M⁻¹ matrix. No external dependencies.
//
// Adapted from OpenLB collisionCUM.h (GPL v2+), Geier et al. 2015.
// Constants re-derived for this project's D3Q27 velocity ordering.
// ================================================================
#ifndef CUMULANT_COLLISION_H
#define CUMULANT_COLLISION_H

#include "cumulant_constants.h"

// Forward declarations (internal, do not call directly)
__device__ static void _cum_forward_chimera(double m[27], const double u[3]);
__device__ static void _cum_backward_chimera(double m[27], const double u[3]);

// ================================================================
//
//  ★ Main entry point — replace gilbm_mrt_collision() with this ★
//
//  Usage in evolution_gilbm.h:
//
//    double f_post[NQ];
//    double rho, ux, uy, uz;
//    cumulant_collision_D3Q27(
//        f_streamed_all,      // post-streaming f[27]
//        omega_global,        // 1/(3ν/dt + 0.5)
//        GILBM_dt,            // global time step
//        0.0, Force[0], 0.0,  // Fx=0, Fy=streamwise, Fz=0
//        f_post, rho, ux, uy, uz
//    );
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

    // 0c. Half-force corrected velocity (Guo 2002, time-symmetric)
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
    //          (Geier 2015, Appendix J, Eq. J.4–J.12)
    // ==============================================================
    _cum_forward_chimera(m, u);

    // After this, m[27] holds central moments κ_αβγ
    // accessed via I_xxx index aliases from cumulant_constants.h

    // ==============================================================
    // STAGE 2: Central Moments → Cumulants
    //          κ_αβγ → C_αβγ
    //          Orders 0–3: C = κ (identical, no conversion needed)
    //          Orders 4–6: subtract lower-order products
    //          (Geier 2015, Appendix J, Eq. J.16–J.19)
    // ==============================================================

    // --- 4th order off-diagonal cumulants (Eq. J.16) ---
    double CUMcbb = m[I_cbb] - ((m[I_caa] + 1.0/3.0) * m[I_abb]
                    + 2.0 * m[I_bba] * m[I_bab]) * inv_rho;
    double CUMbcb = m[I_bcb] - ((m[I_aca] + 1.0/3.0) * m[I_bab]
                    + 2.0 * m[I_bba] * m[I_abb]) * inv_rho;
    double CUMbbc = m[I_bbc] - ((m[I_aac] + 1.0/3.0) * m[I_bba]
                    + 2.0 * m[I_bab] * m[I_abb]) * inv_rho;

    // --- 4th order diagonal cumulants (Eq. J.17) ---
    double CUMcca = m[I_cca] - (((m[I_caa]*m[I_aca] + 2.0*m[I_bba]*m[I_bba])
                    + 1.0/3.0*(m[I_caa]+m[I_aca])) * inv_rho
                    - 1.0/9.0*(drho*inv_rho));
    double CUMcac = m[I_cac] - (((m[I_caa]*m[I_aac] + 2.0*m[I_bab]*m[I_bab])
                    + 1.0/3.0*(m[I_caa]+m[I_aac])) * inv_rho
                    - 1.0/9.0*(drho*inv_rho));
    double CUMacc = m[I_acc] - (((m[I_aac]*m[I_aca] + 2.0*m[I_abb]*m[I_abb])
                    + 1.0/3.0*(m[I_aac]+m[I_aca])) * inv_rho
                    - 1.0/9.0*(drho*inv_rho));

    // --- 5th order cumulants (Eq. J.18) ---
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

    // --- 6th order cumulant (Eq. J.19) ---
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
    //
    //  ★ Only omega (= ω₁) affects physical viscosity: ν = cs²(1/ω₁ - 0.5)dt
    //  ★ All other ω₂–ω₁₀ = 1.0 (full relaxation to equilibrium)
    //
    //  (Geier 2015, Section 4.3, Eq. 55–80)
    // ==============================================================
    const double omega2  = 1.0;  // ω₂: bulk viscosity (Eq. 63)
    const double omega3  = 1.0;  // ω₃: 3rd order symmetric (Eq. 64-66)
    const double omega4  = 1.0;  // ω₄: 3rd order antisymmetric (Eq. 67-70)
    const double omega6  = 1.0;  // ω₆=ω₇=ω₈: 4th order (Eq. 71-76)
    const double omega7  = 1.0;  // ω₉: 5th order (Eq. 77-79)
    const double omega10 = 1.0;  // ω₁₀: 6th order (Eq. 80)

    // --- 2nd order: decompose into orthogonal modes ---
    double mxxPyyPzz = m[I_caa] + m[I_aca] + m[I_aac];  // trace (bulk)
    double mxxMyy    = m[I_caa] - m[I_aca];              // deviatoric 1
    double mxxMzz    = m[I_caa] - m[I_aac];              // deviatoric 2

    // (Eq. 63) Relax trace with ω₂
    mxxPyyPzz += omega2 * (m[I_aaa] - mxxPyyPzz);
    // (Eq. 61-62) Relax deviatorics with ω₁
    mxxMyy *= (1.0 - omega);
    mxxMzz *= (1.0 - omega);

    // (Eq. 55-57) Off-diagonal 2nd order
    m[I_abb] *= (1.0 - omega);  // C₀₁₁
    m[I_bab] *= (1.0 - omega);  // C₁₀₁
    m[I_bba] *= (1.0 - omega);  // C₁₁₀

    // --- 3rd order ---
    double mxxyPyzz = m[I_cba] + m[I_abc];
    double mxxyMyzz = m[I_cba] - m[I_abc];
    double mxxzPyyz = m[I_cab] + m[I_acb];
    double mxxzMyyz = m[I_cab] - m[I_acb];
    double mxyyPxzz = m[I_bca] + m[I_bac];
    double mxyyMxzz = m[I_bca] - m[I_bac];

    m[I_bbb]  *= (1.0 - omega4);    // (Eq. 70) C₁₁₁
    mxxyPyzz  *= (1.0 - omega3);    // (Eq. 64)
    mxxyMyzz  *= (1.0 - omega4);    // (Eq. 67)
    mxxzPyyz  *= (1.0 - omega3);    // (Eq. 65)
    mxxzMyyz  *= (1.0 - omega4);    // (Eq. 68)
    mxyyPxzz  *= (1.0 - omega3);    // (Eq. 66)
    mxyyMxzz  *= (1.0 - omega4);    // (Eq. 69)

    // --- Reconstruct 2nd order individual moments ---
    m[I_caa] = (mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
    m[I_aca] = (-2.0*mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
    m[I_aac] = (mxxMyy - 2.0*mxxMzz + mxxPyyPzz) / 3.0;

    // --- Reconstruct 3rd order ---
    m[I_cba] = (mxxyMyzz + mxxyPyzz) * 0.5;
    m[I_abc] = (-mxxyMyzz + mxxyPyzz) * 0.5;
    m[I_cab] = (mxxzMyyz + mxxzPyyz) * 0.5;
    m[I_acb] = (-mxxzMyyz + mxxzPyyz) * 0.5;
    m[I_bca] = (mxyyMxzz + mxyyPxzz) * 0.5;
    m[I_bac] = (-mxyyMxzz + mxyyPxzz) * 0.5;

    // --- 4th order relaxation (Eq. 71-76) ---
    CUMacc *= (1.0 - omega6);
    CUMcac *= (1.0 - omega6);
    CUMcca *= (1.0 - omega6);
    CUMbbc *= (1.0 - omega6);
    CUMbcb *= (1.0 - omega6);
    CUMcbb *= (1.0 - omega6);

    // --- 5th order relaxation (Eq. 77-79) ---
    CUMbcc *= (1.0 - omega7);
    CUMcbc *= (1.0 - omega7);
    CUMccb *= (1.0 - omega7);

    // --- 6th order relaxation (Eq. 80) ---
    CUMccc *= (1.0 - omega10);

    // ==============================================================
    // STAGE 4: Cumulants → Central Moments (Inverse of Stage 2)
    //          (Geier 2015, Appendix J, Eq. J.16–J.19 inverse)
    // ==============================================================

    // --- 4th order inverse (Eq. J.16 inverse) ---
    m[I_cbb] = CUMcbb + 1.0/3.0*((3.0*m[I_caa]+1.0)*m[I_abb]
               + 6.0*m[I_bba]*m[I_bab]) * inv_rho;
    m[I_bcb] = CUMbcb + 1.0/3.0*((3.0*m[I_aca]+1.0)*m[I_bab]
               + 6.0*m[I_bba]*m[I_abb]) * inv_rho;
    m[I_bbc] = CUMbbc + 1.0/3.0*((3.0*m[I_aac]+1.0)*m[I_bba]
               + 6.0*m[I_bab]*m[I_abb]) * inv_rho;

    // --- 4th order diagonal inverse (Eq. J.17 inverse) ---
    m[I_cca] = CUMcca + (((m[I_caa]*m[I_aca]+2.0*m[I_bba]*m[I_bba])*9.0
               + 3.0*(m[I_caa]+m[I_aca])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;
    m[I_cac] = CUMcac + (((m[I_caa]*m[I_aac]+2.0*m[I_bab]*m[I_bab])*9.0
               + 3.0*(m[I_caa]+m[I_aac])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;
    m[I_acc] = CUMacc + (((m[I_aac]*m[I_aca]+2.0*m[I_abb]*m[I_abb])*9.0
               + 3.0*(m[I_aac]+m[I_aca])) * inv_rho
               - (drho*inv_rho)) * 1.0/9.0;

    // --- 5th order inverse (Eq. J.18 inverse) ---
    m[I_bcc] = CUMbcc + 1.0/3.0*(3.0*(m[I_aac]*m[I_bca] + m[I_aca]*m[I_bac]
               + 4.0*m[I_abb]*m[I_bbb]
               + 2.0*(m[I_bab]*m[I_acb] + m[I_bba]*m[I_abc]))
               + (m[I_bca]+m[I_bac])) * inv_rho;
    m[I_cbc] = CUMcbc + 1.0/3.0*(3.0*(m[I_aac]*m[I_cba] + m[I_caa]*m[I_abc]
               + 4.0*m[I_bab]*m[I_bbb]
               + 2.0*(m[I_abb]*m[I_cab] + m[I_bba]*m[I_bac]))
               + (m[I_cba]+m[I_abc])) * inv_rho;
    m[I_ccb] = CUMccb + 1.0/3.0*(3.0*(m[I_caa]*m[I_acb] + m[I_aca]*m[I_cab]
               + 4.0*m[I_bba]*m[I_bbb]
               + 2.0*(m[I_bab]*m[I_bca] + m[I_abb]*m[I_cba]))
               + (m[I_acb]+m[I_cab])) * inv_rho;

    // --- 6th order inverse (Eq. J.19 inverse) ---
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
        - 1.0/9.0*(m[I_acc]+m[I_cac]+m[I_cca]) * inv_rho
        - 1.0/9.0*(m[I_caa]+m[I_aca]+m[I_aac]) * inv_rho
        + (2.0*(m[I_bab]*m[I_bab]+m[I_abb]*m[I_abb]+m[I_bba]*m[I_bba])
          + (m[I_aac]*m[I_aca]+m[I_aac]*m[I_caa]+m[I_aca]*m[I_caa])
          + 1.0/3.0*(m[I_aac]+m[I_aca]+m[I_caa]))
                * inv_rho * inv_rho * 2.0/3.0
        + 1.0/27.0*((drho*drho - drho) * inv_rho * inv_rho));

    // --- Force correction: sign flip of 1st order (Eq. 85-87) ---
    // This is the second half of the time-symmetric force splitting.
    // Combined with the half-force velocity correction in Stage 0,
    // this gives 2nd-order temporal accuracy.
    m[I_baa] = -m[I_baa];   // (Eq. 85) κ*₁₀₀ = -κ₁₀₀
    m[I_aba] = -m[I_aba];   // (Eq. 86) κ*₀₁₀ = -κ₀₁₀
    m[I_aab] = -m[I_aab];   // (Eq. 87) κ*₀₀₁ = -κ₀₀₁

    // ==============================================================
    // STAGE 5: Backward Chimera Transform (x → y → z)
    //          κ*[27] → f̄*[27]
    //          (Geier 2015, Appendix J, Eq. J.20–J.28)
    // ==============================================================
    _cum_backward_chimera(m, u);

    // --- Restore from well-conditioned: f* = f̄* + w ---
    for (int i = 0; i < 27; i++) {
        f_out[i] = m[i] + GILBM_W[i];
    }

    // --- Output macroscopic quantities ---
    rho_out = rho;
    ux_out  = u[0];
    uy_out  = u[1];
    uz_out  = u[2];
}


// ================================================================
// Internal: Forward Chimera (Stage 1)
// Sweep order: z(dir=2) → y(dir=1) → x(dir=0)
// (Geier 2015, Appendix J, Eq. J.4–J.12)
// ================================================================
__device__ static void _cum_forward_chimera(
    double m[27], const double u[3])
{
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;  // z→0, y→9, x→18
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double k = CUM_K[p];

            double sum  = m[a] + m[c];      // f₋₁ + f₊₁
            double diff = m[c] - m[a];      // f₊₁ - f₋₁

            // (Eq. J.4/J.7/J.10) κ₀ = f₋₁ + f₀ + f₊₁
            m[a] = m[a] + m[b] + m[c];
            // (Eq. J.5/J.8/J.11) κ₁ = (f₊₁ - f₋₁) - u·(κ₀ + K)
            m[b] = diff - (m[a] + k) * u[dir];
            // (Eq. J.6/J.9/J.12) κ₂ = (f₋₁+f₊₁) - 2u·(f₊₁-f₋₁) + u²·(κ₀+K)
            m[c] = sum - 2.0 * diff * u[dir]
                   + u[dir] * u[dir] * (m[a] + k);
        }
    }
}

// ================================================================
// Internal: Backward Chimera (Stage 5)
// Sweep order: x(dir=0) → y(dir=1) → z(dir=2)
// (Geier 2015, Appendix J, Eq. J.20–J.28)
// ================================================================
__device__ static void _cum_backward_chimera(
    double m[27], const double u[3])
{
    for (int dir = 0; dir < 3; dir++) {
        // Backward reverses the sweep order:
        //   dir=0 (x first) → use x-passes 18-26
        //   dir=1 (y second) → use y-passes 9-17
        //   dir=2 (z last) → use z-passes 0-8
        int base = (2 - dir) * 9;

        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double k = CUM_K[p];

            // At this point: m[a]=κ₀, m[b]=κ₁, m[c]=κ₂
            // Recover: f₋₁(→a), f₀(→b), f₊₁(→c)

            // (Eq. J.21/J.24/J.27) f̄₋₁ = ((κ₀+K)(u²-u) + κ₁(2u-1) + κ₂) / 2
            double ma = ((m[c] - m[b]) * 0.5 + m[b] * u[dir]
                        + (m[a] + k) * (u[dir]*u[dir] - u[dir]) * 0.5);
            // (Eq. J.20/J.23/J.26) f̄₀ = κ₀(1-u²) - 2u·κ₁ - κ₂
            double mb = (m[a] - m[c]) - 2.0 * m[b] * u[dir]
                        - (m[a] + k) * u[dir] * u[dir];
            // (Eq. J.22/J.25/J.28) f̄₊₁ = ((κ₀+K)(u²+u) + κ₁(2u+1) + κ₂) / 2
            double mc = ((m[c] + m[b]) * 0.5 + m[b] * u[dir]
                        + (m[a] + k) * (u[dir]*u[dir] + u[dir]) * 0.5);

            m[a] = ma;
            m[b] = mb;
            m[c] = mc;
        }
    }
}

#endif // CUMULANT_COLLISION_H
