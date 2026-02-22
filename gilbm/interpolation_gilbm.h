#ifndef GILBM_INTERPOLATION_H
#define GILBM_INTERPOLATION_H

// 7-point Lagrange weighted sum (from interpolationHillISLBM.h)
#define Intrpl7(f1, a1, f2, a2, f3, a3, f4, a4, f5, a5, f6, a6, f7, a7) \
    ((f1)*(a1)+(f2)*(a2)+(f3)*(a3)+(f4)*(a4)+(f5)*(a5)+(f6)*(a6)+(f7)*(a7))

// Phase 1: GILBM 2nd-order quadratic Lagrange interpolation (Imamura 2005 Eq. 23-24)
//
// 1D coefficients: 3-point stencil at base = floor(upwind_position)
//   t = upwind_position - base  in [0, 1)
//   a0(t) = 0.5*(t-1)*(t-2)    <- base point
//   a1(t) = -t*(t-2)           <- neighbor
//   a2(t) = 0.5*t*(t-1)        <- far point
//
// 3D: tensor product g(up) = sum_{l,m,n} a_l^i * a_m^j * a_n^k * g[base_i+l, base_j+m, base_k+n]

// Compute 1D quadratic interpolation coefficients
__device__ __forceinline__ void quadratic_coeffs(
    double t,           // fractional position in [0, 1)
    double &a0, double &a1, double &a2
) {
    a0 = 0.5 * (t - 1.0) * (t - 2.0);
    a1 = -t * (t - 2.0);
    a2 = 0.5 * t * (t - 1.0);
}

// Compute 1D 7-point Lagrange interpolation coefficients
// Nodes at integer positions 0,1,2,3,4,5,6; evaluate at position t
// t is typically in [3,4) for a centered stencil, or shifted near boundaries
__device__ __forceinline__ void lagrange_7point_coeffs(double t, double a[7]) {
    for (int k = 0; k < 7; k++) {
        double L = 1.0;
        for (int j = 0; j < 7; j++) {
            if (j != k) L *= (t - (double)j) / (double)(k - j);
        }
        a[k] = L;
    }
}

// Compute 7-point Lagrange weights with explicit offset (Imamura GILBM)
//   alpha_frac: fractional position in [0,1) = up - floor(up)
//   offset: stencil points before floor(up) (3=centered, adapts near zeta walls)
//   Equivalence: compute_lagrange_weights(alpha, offset, w) == lagrange_7point_coeffs(alpha+offset, w)
__device__ __forceinline__ void compute_lagrange_weights(
    double alpha_frac, int offset, double weights[7]
) {
    for (int k = 0; k < 7; k++) {
        double L = 1.0;
        for (int j = 0; j < 7; j++) {
            if (j != k) {
                L *= (alpha_frac - (double)(j - offset)) / (double)(k - j);
            }
        }
        weights[k] = L;
    }
}

// Compute equilibrium distribution for a single alpha at given macroscopic state
// Used by LTS re-estimation (Imamura 2005 Eq. 36)
__device__ __forceinline__ double compute_feq_alpha(
    int alpha, double rho, double ux, double uy, double uz
) {
    double eu = GILBM_e[alpha][0]*ux + GILBM_e[alpha][1]*uy + GILBM_e[alpha][2]*uz;
    double udot = ux*ux + uy*uy + uz*uz;
    return GILBM_W[alpha] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*udot);
}

// 3D quadratic upwind interpolation (tensor product)
// up_i, up_j, up_k: upwind point coordinates in computational space
// f_alpha: distribution function array for this direction [NYD6 * NZ6 * NX6]
// Array layout: index = j*NZ6*NX6 + k*NX6 + i
__device__ double interpolate_quadratic_3d(
    double up_i, double up_j, double up_k,
    const double *f_alpha,
    int NX6_val, int NZ6_val
) {
    // Base indices (floor)
    int bi = (int)floor(up_i);
    int bj = (int)floor(up_j);
    int bk = (int)floor(up_k);

    // Fractional parts
    double ti = up_i - (double)bi;
    double tj = up_j - (double)bj;
    double tk = up_k - (double)bk;

    // Interpolation coefficients for each dimension
    double ai[3], aj[3], ak[3];
    quadratic_coeffs(ti, ai[0], ai[1], ai[2]);
    quadratic_coeffs(tj, aj[0], aj[1], aj[2]);
    quadratic_coeffs(tk, ak[0], ak[1], ak[2]);

    // Tensor product summation: 3x3x3 = 27 points max
    double result = 0.0;
    for (int n = 0; n < 3; n++) {         // k direction
        for (int m = 0; m < 3; m++) {     // j direction
            double wjk = aj[m] * ak[n];
            for (int l = 0; l < 3; l++) { // i direction
                int idx = (bj + m) * NZ6_val * NX6_val
                        + (bk + n) * NX6_val
                        + (bi + l);
                result += ai[l] * wjk * f_alpha[idx];
            }
        }
    }
    return result;
}

// Phase 4: 3D quadratic interpolation with LTS re-estimation (Imamura 2005 Eq. 36)
//
// At each stencil point B, the non-equilibrium part is rescaled to match
// the target point A's relaxation scale:
//   f̃*_α|_B = f^eq_α|_B + (f_α|_B - f^eq_α|_B) * (τ_A-1)·dt_A / (τ_B·dt_B)
//
// This ensures correct interpolation when neighboring cells have different
// local time steps (and hence different τ·dt products).
__device__ double interpolate_quadratic_3d_lts(
    double up_i, double up_j, double up_k,
    double * const *f_old_all,          // [19] pointers to all f arrays
    const double *tau_dt_field,         // [NYD6*NZ6] precomputed τ·dt at each (j,k)
    int alpha,
    double tau_A_minus1_dt_A,           // (τ_A - 1) * dt_A at target point
    int NX6_val, int NZ6_val
) {
    int bi = (int)floor(up_i);
    int bj = (int)floor(up_j);
    int bk = (int)floor(up_k);

    double ti = up_i - (double)bi;
    double tj = up_j - (double)bj;
    double tk = up_k - (double)bk;

    double ai[3], aj[3], ak[3];
    quadratic_coeffs(ti, ai[0], ai[1], ai[2]);
    quadratic_coeffs(tj, aj[0], aj[1], aj[2]);
    quadratic_coeffs(tk, ak[0], ak[1], ak[2]);

    double result = 0.0;
    for (int n = 0; n < 3; n++) {
        for (int m = 0; m < 3; m++) {
            double wjk = aj[m] * ak[n];
            int jB = bj + m;
            int kB = bk + n;
            int idx_jk_B = jB * NZ6_val + kB;
            double tau_dt_B = tau_dt_field[idx_jk_B];
            double R_AB = tau_A_minus1_dt_A / tau_dt_B;

            for (int l = 0; l < 3; l++) {
                int idx_B = jB * NZ6_val * NX6_val
                          + kB * NX6_val
                          + (bi + l);

                // Read f at B for this alpha
                double f_B = f_old_all[alpha][idx_B];

                // Compute macroscopic at B from all 19 f's
                double rho_B = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                for (int a = 0; a < 19; a++) {
                    double fa = f_old_all[a][idx_B];
                    rho_B += fa;
                    mx += GILBM_e[a][0] * fa;
                    my += GILBM_e[a][1] * fa;
                    mz += GILBM_e[a][2] * fa;
                }
                double ux_B = mx / rho_B;
                double uy_B = my / rho_B;
                double uz_B = mz / rho_B;

                // f_eq at B for this alpha
                double feq_B = compute_feq_alpha(alpha, rho_B, ux_B, uy_B, uz_B);

                // Re-estimation: rescale non-equilibrium part
                double f_tilde_B = feq_B + (f_B - feq_B) * R_AB;

                result += ai[l] * wjk * f_tilde_B;
            }
        }
    }
    return result;
}

// Phase 5: 7-point Lagrange interpolation with per-stencil BGK collision
//
// Algorithm (3 steps inside this function):
//   Step 2: For each of the 7³=343 stencil nodes B, compute post-BGK value:
//             f_post_B = f_B - (1/τ_B) × (f_B - feq_B)
//           stored in internal buffer f_re[7][7][7]
//   Step 3: Lagrange interpolation f_re → arrival point A
//
// Stencil base: bi = floor(up_i) - 3  (symmetric center)
// Boundary-adaptive clamping ensures stencil stays within valid grid cells.
// τ_B from tau_local_field[jB*NZ6+kB] (per (j,k) column, uniform in i).
__device__ double interpolate_lagrange7_3d_bgk(
    double up_i, double up_j, double up_k,
    double * const *f_old_all,       // [19] pointers to all f arrays
    const double *tau_local_field,   // τ_B at each (j,k) — for per-stencil BGK
    int alpha,
    int NX6_val, int NYD6_val, int NZ6_val
) {
    // Stencil base: symmetric center, floor(up) - 3
    int bi = (int)floor(up_i) - 3;
    int bj = (int)floor(up_j) - 3;
    int bk = (int)floor(up_k) - 3;

    // Boundary-adaptive clamping (wall cells: k=2 and k=NZ6-3)
    if (bi < 0)                bi = 0;
    if (bi + 6 >= NX6_val)    bi = NX6_val - 7;
    if (bj < 0)                bj = 0;
    if (bj + 6 >= NYD6_val)   bj = NYD6_val - 7;
    if (bk < 2)                bk = 2;
    if (bk + 6 > NZ6_val - 3) bk = NZ6_val - 9;  // bk+6 <= NZ6-3

    // Fractional position relative to stencil base
    double ti = up_i - (double)bi;
    double tj = up_j - (double)bj;
    double tk = up_k - (double)bk;

    // 7-point Lagrange coefficients per direction
    double ai[7], aj[7], ak[7];
    lagrange_7point_coeffs(ti, ai);
    lagrange_7point_coeffs(tj, aj);
    lagrange_7point_coeffs(tk, ak);

    // Per-stencil BGK buffer: f_re[l][m][n] = f_post at node (bi+l, bj+m, bk+n)
    double f_re[7][7][7];

    // Step 2: Per-stencil BGK — compute f_post at all 343 stencil nodes
    for (int n = 0; n < 7; n++) {
        int kB = bk + n;
        for (int m = 0; m < 7; m++) {
            int jB = bj + m;
            int idx_jk_B = jB * NZ6_val + kB;
            double inv_tau_B = 1.0 / tau_local_field[idx_jk_B];

            for (int l = 0; l < 7; l++) {
                int iB = bi + l;
                int idx_B = jB * NZ6_val * NX6_val + kB * NX6_val + iB;

                // Macroscopic at B (from all 19 f values)
                double rho_B = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                for (int a = 0; a < 19; a++) {
                    double fa = f_old_all[a][idx_B];
                    rho_B += fa;
                    mx += GILBM_e[a][0] * fa;
                    my += GILBM_e[a][1] * fa;
                    mz += GILBM_e[a][2] * fa;
                }
                double ux_B = mx / rho_B;
                double uy_B = my / rho_B;
                double uz_B = mz / rho_B;

                // f_eq at B for this alpha
                double feq_B = compute_feq_alpha(alpha, rho_B, ux_B, uy_B, uz_B);

                // Per-stencil BGK: f_post = f_B - (1/τ_B)(f_B - feq_B)
                double f_B = f_old_all[alpha][idx_B];
                f_re[l][m][n] = f_B - inv_tau_B * (f_B - feq_B);
            }
        }
    }

    // Step 3: Lagrange interpolation from f_re to arrival point A
    double result = 0.0;
    for (int n = 0; n < 7; n++) {
        for (int m = 0; m < 7; m++) {
            double wjk = aj[m] * ak[n];
            for (int l = 0; l < 7; l++) {
                result += ai[l] * wjk * f_re[l][m][n];
            }
        }
    }
    return result;
}

#endif
