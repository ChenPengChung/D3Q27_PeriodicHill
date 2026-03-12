#ifndef CUMULANT_EQUILIBRIUM_HOST_H
#define CUMULANT_EQUILIBRIUM_HOST_H

// ================================================================
// Host-side computation of Cumulant WP equilibrium at rho=1, u=0
//
// The Cumulant WP collision operator has 4th-order equilibria (A,B)
// that make its equilibrium distribution DIFFERENT from the standard
// LBM feq = W[q] * rho. This function computes the Cumulant-specific
// equilibrium for use in Chapman-Enskog boundary conditions.
//
// Usage: Call ComputeCumulantEquilibrium_Host(omega_global, feq_cum)
//        in main.cu initialization, then upload feq_cum to __constant__ memory.
// ================================================================

#include <cmath>
#include <cstdio>

// Local copies of D3Q27 constants (CPU-accessible, matching GPU ordering)
static const double HOST_GILBM_W[27] = {
    8.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

static const double HOST_GILBM_e[27][3] = {
    { 0, 0, 0},
    { 1, 0, 0}, {-1, 0, 0},
    { 0, 1, 0}, { 0,-1, 0},
    { 0, 0, 1}, { 0, 0,-1},
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 1}, {-1, 1, 1}, { 1,-1, 1}, {-1,-1, 1},
    { 1, 1,-1}, {-1, 1,-1}, { 1,-1,-1}, {-1,-1,-1}
};

static const int HOST_CUM_IDX[27][3] = {
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

static const double HOST_CUM_K[27] = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    2.0/3.0, 0.0, 2.0/9.0, 1.0/6.0, 0.0, 1.0/18.0,
    1.0/6.0, 0.0, 1.0/18.0,
    1.0, 0.0, 1.0/3.0, 0.0, 0.0, 0.0, 1.0/3.0, 0.0, 1.0/9.0
};

// Moment index aliases (same as cumulant_constants.h)
#define H_I_aaa 26
#define H_I_baa 18
#define H_I_aba 14
#define H_I_aab 10
#define H_I_caa 25
#define H_I_aca 24
#define H_I_aac 22
#define H_I_bba  6
#define H_I_bab  4
#define H_I_abb  2
#define H_I_bbb  0
#define H_I_cba 13
#define H_I_bca 17
#define H_I_cab  9
#define H_I_acb  8
#define H_I_bac 16
#define H_I_abc 12
#define H_I_cbb  1
#define H_I_bcb  3
#define H_I_bbc  5
#define H_I_cca 23
#define H_I_cac 21
#define H_I_acc 20
#define H_I_ccb  7
#define H_I_bcc 15
#define H_I_cbc 11
#define H_I_ccc 19

// Host version of WP helper: compute A, B (Eq.17-18)
static void host_cum_wp_compute_AB(double w1, double w2, double *A, double *B) {
    const double DEN_EPS = 1.0e-10;
    double denom = (w1 - w2) * (w2 * (2.0 + 3.0*w1) - 8.0*w1);
    if (fabs(denom) > DEN_EPS) {
        *A = (4.0*w1*w1 + 2.0*w1*w2*(w1 - 6.0)
           + w2*w2*(w1*(10.0 - 3.0*w1) - 4.0)) / denom;
        *B = (4.0*w1*w2*(9.0*w1 - 16.0) - 4.0*w1*w1
           - 2.0*w2*w2*(2.0 + 9.0*w1*(w1 - 2.0))) / (3.0 * denom);
    } else {
        *A = 0.0;
        *B = 0.0;
    }
}

// ================================================================
// Compute Cumulant WP equilibrium at rho=1, u=v=w=0
//
// Method: Iterates the full collision (no force) 200 times from
// standard feq to converge to the Cumulant fixed point.
//
// Parameters:
//   omega_global: tau = 3*nu/dt + 0.5 (relaxation TIME, same as GPU)
//   feq_cum[27]:  output array, Cumulant equilibrium distribution
// ================================================================
static void ComputeCumulantEquilibrium_Host(
    double omega_global,   // tau (relaxation time)
    double feq_cum[27])
{
    const double omega2 = CUM_OMEGA2;
    const double omega = 1.0 / omega_global;  // omega rate = 1/tau
    const double omega6 = 1.0;
    const double omega9 = 1.0;
    const double omega10 = 1.0;
    const double inv_rho = 1.0;  // rho = 1.0 at equilibrium

    // Initialize with standard feq at rho=1, u=0
    for (int q = 0; q < 27; q++)
        feq_cum[q] = HOST_GILBM_W[q];

    // Iterate collision 200 times (no force, u=0) to find fixed point
    for (int iter = 0; iter < 200; iter++) {
        double m[27];

        // Well-conditioning: m = f - W
        for (int q = 0; q < 27; q++)
            m[q] = feq_cum[q] - HOST_GILBM_W[q];

        // Forward Chimera at u=0 (simplified: velocity shifts are zero)
        for (int dir = 2; dir >= 0; dir--) {
            int base = (2 - dir) * 9;
            for (int j = 0; j < 9; j++) {
                int p = base + j;
                int a = HOST_CUM_IDX[p][0];
                int b = HOST_CUM_IDX[p][1];
                int c = HOST_CUM_IDX[p][2];
                double old_a = m[a], old_b = m[b], old_c = m[c];
                // At u=0: no velocity shift (k = HOST_CUM_K[p] not needed)
                m[a] = old_a + old_b + old_c;
                m[b] = old_c - old_a;
                m[c] = old_a + old_c;
            }
        }

        // Central moments -> Cumulants (Stage 2, orders 4+)
        double CUMcca = m[H_I_cca] - (m[H_I_caa]*m[H_I_aca] + 2.0*m[H_I_bba]*m[H_I_bba]) * inv_rho;
        double CUMcac = m[H_I_cac] - (m[H_I_caa]*m[H_I_aac] + 2.0*m[H_I_bab]*m[H_I_bab]) * inv_rho;
        double CUMacc = m[H_I_acc] - (m[H_I_aca]*m[H_I_aac] + 2.0*m[H_I_abb]*m[H_I_abb]) * inv_rho;

        double CUMcbb = m[H_I_cbb] - (m[H_I_caa]*m[H_I_abb] + 2.0*m[H_I_bba]*m[H_I_bab]) * inv_rho;
        double CUMbcb = m[H_I_bcb] - (m[H_I_aca]*m[H_I_bab] + 2.0*m[H_I_bba]*m[H_I_abb]) * inv_rho;
        double CUMbbc = m[H_I_bbc] - (m[H_I_aac]*m[H_I_bba] + 2.0*m[H_I_bab]*m[H_I_abb]) * inv_rho;

        double CUMccb = m[H_I_ccb]
            - (m[H_I_caa]*m[H_I_acb] + m[H_I_aca]*m[H_I_cab]
               + 4.0*m[H_I_bba]*m[H_I_bbb] + 2.0*m[H_I_bab]*m[H_I_abc]
               + 2.0*m[H_I_abb]*m[H_I_cba]) * inv_rho
            + 2.0*(m[H_I_caa]*m[H_I_aca]*m[H_I_aab]
                  + 2.0*m[H_I_bba]*m[H_I_bba]*m[H_I_aab]
                  + 2.0*m[H_I_bba]*m[H_I_bab]*m[H_I_aba]
                  + 2.0*m[H_I_bba]*m[H_I_abb]*m[H_I_baa]) * inv_rho * inv_rho;

        double CUMbcc = m[H_I_bcc]
            - (m[H_I_aca]*m[H_I_bac] + m[H_I_aac]*m[H_I_bca]
               + 4.0*m[H_I_abb]*m[H_I_bbb] + 2.0*m[H_I_bba]*m[H_I_abc]
               + 2.0*m[H_I_bab]*m[H_I_acb]) * inv_rho
            + 2.0*(m[H_I_aca]*m[H_I_aac]*m[H_I_baa]
                  + 2.0*m[H_I_abb]*m[H_I_abb]*m[H_I_baa]
                  + 2.0*m[H_I_abb]*m[H_I_bab]*m[H_I_aab]
                  + 2.0*m[H_I_abb]*m[H_I_bba]*m[H_I_aba]) * inv_rho * inv_rho;

        double CUMcbc = m[H_I_cbc]
            - (m[H_I_caa]*m[H_I_abc] + m[H_I_aac]*m[H_I_cba]
               + 4.0*m[H_I_bab]*m[H_I_bbb] + 2.0*m[H_I_bba]*m[H_I_bac]
               + 2.0*m[H_I_abb]*m[H_I_cab]) * inv_rho
            + 2.0*(m[H_I_caa]*m[H_I_aac]*m[H_I_aba]
                  + 2.0*m[H_I_bab]*m[H_I_bab]*m[H_I_aba]
                  + 2.0*m[H_I_bab]*m[H_I_bba]*m[H_I_aab]
                  + 2.0*m[H_I_bab]*m[H_I_abb]*m[H_I_baa]) * inv_rho * inv_rho;

        double CUMccc = m[H_I_ccc]
            - (  m[H_I_cca]*m[H_I_aac] + m[H_I_cac]*m[H_I_aca] + m[H_I_acc]*m[H_I_caa]
               + 2.0*(m[H_I_ccb]*m[H_I_aab] + m[H_I_cbc]*m[H_I_aba] + m[H_I_bcc]*m[H_I_baa])
               + 8.0*m[H_I_bbb]*m[H_I_bbb]
               + 4.0*(m[H_I_cbb]*m[H_I_abb] + m[H_I_bcb]*m[H_I_bab] + m[H_I_bbc]*m[H_I_bba])) * inv_rho
            + 4.0*(  m[H_I_caa]*m[H_I_abb]*m[H_I_abb]
                   + m[H_I_aca]*m[H_I_bab]*m[H_I_bab]
                   + m[H_I_aac]*m[H_I_bba]*m[H_I_bba]
                   + 4.0*m[H_I_bba]*m[H_I_bab]*m[H_I_abb]
                   + m[H_I_caa]*m[H_I_aca]*m[H_I_aac]) * inv_rho * inv_rho
            - 16.0*(m[H_I_bba]*m[H_I_bab]*m[H_I_abb]*m[H_I_baa]*inv_rho
                   + m[H_I_caa]*m[H_I_aca]*m[H_I_aac]*m[H_I_baa]*inv_rho) * inv_rho * inv_rho;

        // Relaxation
        // 2nd order
        double mxxPyyPzz = m[H_I_caa] + m[H_I_aca] + m[H_I_aac];
        double mxxMyy = m[H_I_caa] - m[H_I_aca];
        double mxxMzz = m[H_I_caa] - m[H_I_aac];
        mxxPyyPzz += omega2 * (m[H_I_aaa] - mxxPyyPzz);
        mxxMyy *= (1.0 - omega);
        mxxMzz *= (1.0 - omega);

#if USE_WP_CUMULANT
        // Save pre-relaxation off-diagonal 2nd-order values for WP B26-B28
        const double saved_C011 = m[H_I_abb];
        const double saved_C101 = m[H_I_bab];
        const double saved_C110 = m[H_I_bba];
#endif
        m[H_I_abb] *= (1.0 - omega);
        m[H_I_bab] *= (1.0 - omega);
        m[H_I_bba] *= (1.0 - omega);

        // 3rd order (at u=0 with no force, all are ~0, relax toward 0)
        m[H_I_bbb] = 0.0;
        m[H_I_cba] = 0.0; m[H_I_abc] = 0.0;
        m[H_I_cab] = 0.0; m[H_I_acb] = 0.0;
        m[H_I_bca] = 0.0; m[H_I_bac] = 0.0;

        // Reconstruct 2nd order
        m[H_I_caa] = ( mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
        m[H_I_aca] = (-2.0*mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
        m[H_I_aac] = ( mxxMyy - 2.0*mxxMzz + mxxPyyPzz) / 3.0;

        // 4th order diagonal: relax toward WP equilibria
#if USE_WP_CUMULANT
        double Dxx = m[H_I_caa], Dyy = m[H_I_aca], Dzz = m[H_I_aac];
        double Dxy = m[H_I_bba], Dxz = m[H_I_bab], Dyz = m[H_I_abb];
        double Sxx = Dxx + 1.0/3.0, Syy = Dyy + 1.0/3.0, Szz = Dzz + 1.0/3.0;

        double coeff_A, coeff_B;
        host_cum_wp_compute_AB(omega, omega2, &coeff_A, &coeff_B);

        double CUMcca_eq = (coeff_A * (Sxx*Syy + Dxy*Dxy) + coeff_B * (Sxx*Syy - Dxy*Dxy)) * inv_rho;
        double CUMcac_eq = (coeff_A * (Sxx*Szz + Dxz*Dxz) + coeff_B * (Sxx*Szz - Dxz*Dxz)) * inv_rho;
        double CUMacc_eq = (coeff_A * (Syy*Szz + Dyz*Dyz) + coeff_B * (Syy*Szz - Dyz*Dyz)) * inv_rho;

        CUMcca += omega6 * (CUMcca_eq - CUMcca);
        CUMcac += omega6 * (CUMcac_eq - CUMcac);
        CUMacc += omega6 * (CUMacc_eq - CUMacc);

        // Off-diagonal 4th order (B26-B28)
        double wp_offdiag_coeff = (1.0 - omega * 0.5) * coeff_B;
        double wp_C211 = wp_offdiag_coeff * saved_C011;
        double wp_C121 = wp_offdiag_coeff * saved_C101;
        double wp_C112 = wp_offdiag_coeff * saved_C110;
#else
        CUMcca *= (1.0 - omega6);
        CUMcac *= (1.0 - omega6);
        CUMacc *= (1.0 - omega6);
        CUMcbb *= (1.0 - omega6);
        CUMbcb *= (1.0 - omega6);
        CUMbbc *= (1.0 - omega6);
#endif

        // 5th order
        CUMbcc *= (1.0 - omega9);
        CUMcbc *= (1.0 - omega9);
        CUMccb *= (1.0 - omega9);

        // 6th order
        CUMccc *= (1.0 - omega10);

        // Cumulants -> Central moments (Stage 4)
        m[H_I_cca] = CUMcca + (m[H_I_caa]*m[H_I_aca] + 2.0*m[H_I_bba]*m[H_I_bba]) * inv_rho;
        m[H_I_cac] = CUMcac + (m[H_I_caa]*m[H_I_aac] + 2.0*m[H_I_bab]*m[H_I_bab]) * inv_rho;
        m[H_I_acc] = CUMacc + (m[H_I_aca]*m[H_I_aac] + 2.0*m[H_I_abb]*m[H_I_abb]) * inv_rho;

#if USE_WP_CUMULANT
        m[H_I_cbb] = wp_C211;
        m[H_I_bcb] = wp_C121;
        m[H_I_bbc] = wp_C112;
#else
        m[H_I_cbb] = CUMcbb + (m[H_I_caa]*m[H_I_abb] + 2.0*m[H_I_bba]*m[H_I_bab]) * inv_rho;
        m[H_I_bcb] = CUMbcb + (m[H_I_aca]*m[H_I_bab] + 2.0*m[H_I_bba]*m[H_I_abb]) * inv_rho;
        m[H_I_bbc] = CUMbbc + (m[H_I_aac]*m[H_I_bba] + 2.0*m[H_I_bab]*m[H_I_abb]) * inv_rho;
#endif

        m[H_I_ccb] = CUMccb
            + (m[H_I_caa]*m[H_I_acb] + m[H_I_aca]*m[H_I_cab]
               + 4.0*m[H_I_bba]*m[H_I_bbb] + 2.0*m[H_I_bab]*m[H_I_abc]
               + 2.0*m[H_I_abb]*m[H_I_cba]) * inv_rho
            - 2.0*(m[H_I_caa]*m[H_I_aca]*m[H_I_aab]
                  + 2.0*m[H_I_bba]*m[H_I_bba]*m[H_I_aab]
                  + 2.0*m[H_I_bba]*m[H_I_bab]*m[H_I_aba]
                  + 2.0*m[H_I_bba]*m[H_I_abb]*m[H_I_baa]) * inv_rho * inv_rho;

        m[H_I_bcc] = CUMbcc
            + (m[H_I_aca]*m[H_I_bac] + m[H_I_aac]*m[H_I_bca]
               + 4.0*m[H_I_abb]*m[H_I_bbb] + 2.0*m[H_I_bba]*m[H_I_abc]
               + 2.0*m[H_I_bab]*m[H_I_acb]) * inv_rho
            - 2.0*(m[H_I_aca]*m[H_I_aac]*m[H_I_baa]
                  + 2.0*m[H_I_abb]*m[H_I_abb]*m[H_I_baa]
                  + 2.0*m[H_I_abb]*m[H_I_bab]*m[H_I_aab]
                  + 2.0*m[H_I_abb]*m[H_I_bba]*m[H_I_aba]) * inv_rho * inv_rho;

        m[H_I_cbc] = CUMcbc
            + (m[H_I_caa]*m[H_I_abc] + m[H_I_aac]*m[H_I_cba]
               + 4.0*m[H_I_bab]*m[H_I_bbb] + 2.0*m[H_I_bba]*m[H_I_bac]
               + 2.0*m[H_I_abb]*m[H_I_cab]) * inv_rho
            - 2.0*(m[H_I_caa]*m[H_I_aac]*m[H_I_aba]
                  + 2.0*m[H_I_bab]*m[H_I_bab]*m[H_I_aba]
                  + 2.0*m[H_I_bab]*m[H_I_bba]*m[H_I_aab]
                  + 2.0*m[H_I_bab]*m[H_I_abb]*m[H_I_baa]) * inv_rho * inv_rho;

        m[H_I_ccc] = CUMccc
            + (  m[H_I_cca]*m[H_I_aac] + m[H_I_cac]*m[H_I_aca] + m[H_I_acc]*m[H_I_caa]
               + 2.0*(m[H_I_ccb]*m[H_I_aab] + m[H_I_cbc]*m[H_I_aba] + m[H_I_bcc]*m[H_I_baa])
               + 8.0*m[H_I_bbb]*m[H_I_bbb]
               + 4.0*(m[H_I_cbb]*m[H_I_abb] + m[H_I_bcb]*m[H_I_bab] + m[H_I_bbc]*m[H_I_bba])) * inv_rho
            - 4.0*(  m[H_I_caa]*m[H_I_abb]*m[H_I_abb]
                   + m[H_I_aca]*m[H_I_bab]*m[H_I_bab]
                   + m[H_I_aac]*m[H_I_bba]*m[H_I_bba]
                   + 4.0*m[H_I_bba]*m[H_I_bab]*m[H_I_abb]
                   + m[H_I_caa]*m[H_I_aca]*m[H_I_aac]) * inv_rho * inv_rho
            + 16.0*(m[H_I_bba]*m[H_I_bab]*m[H_I_abb]*m[H_I_baa]*inv_rho
                   + m[H_I_caa]*m[H_I_aca]*m[H_I_aac]*m[H_I_baa]*inv_rho) * inv_rho * inv_rho;

        // Backward Chimera at u=0
        for (int dir = 0; dir < 3; dir++) {
            int base = (2 - dir) * 9;
            for (int j = 0; j < 9; j++) {
                int p = base + j;
                int a = HOST_CUM_IDX[p][0];
                int b = HOST_CUM_IDX[p][1];
                int c = HOST_CUM_IDX[p][2];
                double old_a = m[a], old_b = m[b], old_c = m[c];
                // At u=0: simplified backward Chimera
                m[a] = (old_c - old_b) * 0.5;
                m[b] = old_a - old_c;
                m[c] = (old_c + old_b) * 0.5;
            }
        }

        // Restore: f = m + W
        for (int q = 0; q < 27; q++)
            feq_cum[q] = m[q] + HOST_GILBM_W[q];
    }

    // Verify mass conservation
    double sum = 0.0;
    for (int q = 0; q < 27; q++) sum += feq_cum[q];
    if (fabs(sum - 1.0) > 1e-12) {
        printf("[WARNING] Cumulant equilibrium mass error: sum = %.15e (should be 1.0)\n", sum);
    }
}

// Cleanup local macros
#undef H_I_aaa
#undef H_I_baa
#undef H_I_aba
#undef H_I_aab
#undef H_I_caa
#undef H_I_aca
#undef H_I_aac
#undef H_I_bba
#undef H_I_bab
#undef H_I_abb
#undef H_I_bbb
#undef H_I_cba
#undef H_I_bca
#undef H_I_cab
#undef H_I_acb
#undef H_I_bac
#undef H_I_abc
#undef H_I_cbb
#undef H_I_bcb
#undef H_I_bbc
#undef H_I_cca
#undef H_I_cac
#undef H_I_acc
#undef H_I_ccb
#undef H_I_bcc
#undef H_I_cbc
#undef H_I_ccc

#endif // CUMULANT_EQUILIBRIUM_HOST_H
