#ifndef GILBM_INTERPOLATION_H
#define GILBM_INTERPOLATION_H

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

#endif
