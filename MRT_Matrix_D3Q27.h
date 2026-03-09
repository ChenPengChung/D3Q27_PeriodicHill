#ifndef MRT_MATRIX_D3Q27_FILE
#define MRT_MATRIX_D3Q27_FILE

// ============================================================================
// D3Q27 MRT Transformation Matrix (Suga et al. 2015, Comp. Math. Appl. 69)
// ============================================================================
//
// Construction: Tensor-product orthogonal basis from 1D polynomials
//   p0(xi) = 1,  p1(xi) = xi,  p2(xi) = xi^2 - 1/3
// These are orthogonal w.r.t. 1D weights w(-1)=1/6, w(0)=2/3, w(1)=1/6.
//
// 3D moments M[n][alpha] = p_a(xix_alpha) * p_b(xiy_alpha) * p_c(xiz_alpha)
// where (a,b,c) is the polynomial index triplet for moment n.
//
// Inverse: M_inv[alpha][n] = w_alpha * M[n][alpha] / ||m_n||^2_w
// where ||m_n||^2_w = ||p_a||^2 * ||p_b||^2 * ||p_c||^2
//
// Norms: ||p0||^2=1, ||p1||^2=1/3, ||p2||^2=2/9
//
// The moment ordering is chosen to match physical interpretation:
//   0: density (conserved)
//   1-3: momentum (conserved)
//   4-9: stress tensor (viscosity-related, s = 1/tau)
//   10-15: energy flux (3rd order)
//   16: xixxiyxiz (3rd order)
//   17-26: higher-order ghost moments
// ============================================================================

// Number of discrete velocities
#define NQ 27

// D3Q27 velocity vectors (matching D3Q19 ordering for indices 0-18, corners at 19-26)
static const int D3Q27_ex[NQ] = { 0,  1,-1, 0, 0, 0, 0,  1,-1, 1,-1,  1,-1, 1,-1, 0, 0, 0, 0,   1,-1, 1,-1, 1,-1, 1,-1};
static const int D3Q27_ey[NQ] = { 0,  0, 0, 1,-1, 0, 0,  1, 1,-1,-1,  0, 0, 0, 0, 1,-1, 1,-1,   1, 1,-1,-1, 1, 1,-1,-1};
static const int D3Q27_ez[NQ] = { 0,  0, 0, 0, 0, 1,-1,  0, 0, 0, 0,  1, 1,-1,-1, 1, 1,-1,-1,   1, 1, 1, 1,-1,-1,-1,-1};

// D3Q27 weights (factored: w = wx * wy * wz, where w(0)=2/3, w(+/-1)=1/6)
static const double D3Q27_W[NQ] = {
    8.0/27.0,                                       // rest: (2/3)^3
    2.0/27.0, 2.0/27.0, 2.0/27.0,                  // face: (1/6)(2/3)(2/3) * 6
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,        // edge: (1/6)(1/6)(2/3) * 12
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,    // corner: (1/6)^3 * 8
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

// ============================================================================
// Moment ordering: (a,b,c) polynomial index triplet
// ============================================================================
// Row n of M: M[n][alpha] = p_a(xix_alpha) * p_b(xiy_alpha) * p_c(xiz_alpha)
//
// Index | (a,b,c) | Physical meaning                  | Relaxation
// ------|---------|-----------------------------------|----------
//   0   | (0,0,0) | rho  (density)                      | s=0 (conserved)
//   1   | (1,0,0) | jx (x-momentum)                   | s=0 (conserved)
//   2   | (0,1,0) | jy (y-momentum)                   | s=0 (conserved)
//   3   | (0,0,1) | jz (z-momentum)                   | s=0 (conserved)
//   4   | (2,0,0) | sigmaxx = xix^2-1/3 (normal stress)     | s=1/tau (viscosity)
//   5   | (0,2,0) | sigmayy = xiy^2-1/3 (normal stress)     | s=1/tau (viscosity)
//   6   | (0,0,2) | sigmazz = xiz^2-1/3 (normal stress)     | s=1/tau (viscosity)
//   7   | (1,1,0) | sigmaxy = xixxiy    (shear stress)      | s=1/tau (viscosity)
//   8   | (1,0,1) | sigmaxz = xixxiz    (shear stress)      | s=1/tau (viscosity)
//   9   | (0,1,1) | sigmayz = xiyxiz    (shear stress)      | s=1/tau (viscosity)
//  10   | (2,1,0) | qx(y) = (xix^2-1/3)xiy              | s_q (energy flux)
//  11   | (2,0,1) | qx(z) = (xix^2-1/3)xiz              | s_q
//  12   | (1,2,0) | qy(x) = xix(xiy^2-1/3)              | s_q
//  13   | (0,2,1) | qy(z) = (xiy^2-1/3)xiz              | s_q
//  14   | (1,0,2) | qz(x) = xix(xiz^2-1/3)              | s_q
//  15   | (0,1,2) | qz(y) = xiy(xiz^2-1/3)              | s_q
//  16   | (1,1,1) | xixxiyxiz                            | s_16
//  17   | (2,2,0) | (xix^2-1/3)(xiy^2-1/3)               | s_ghost
//  18   | (2,0,2) | (xix^2-1/3)(xiz^2-1/3)               | s_ghost
//  19   | (0,2,2) | (xiy^2-1/3)(xiz^2-1/3)               | s_ghost
//  20   | (2,1,1) | (xix^2-1/3)xiyxiz                    | s_ghost
//  21   | (1,2,1) | xix(xiy^2-1/3)xiz                    | s_ghost
//  22   | (1,1,2) | xixxiy(xiz^2-1/3)                    | s_ghost
//  23   | (2,2,1) | (xix^2-1/3)(xiy^2-1/3)xiz             | s_ghost
//  24   | (2,1,2) | (xix^2-1/3)xiy(xiz^2-1/3)             | s_ghost
//  25   | (1,2,2) | xix(xiy^2-1/3)(xiz^2-1/3)             | s_ghost
//  26   | (2,2,2) | (xix^2-1/3)(xiy^2-1/3)(xiz^2-1/3)      | s_ghost

// Polynomial index triplets in the above ordering
static const int MRT27_abc[NQ][3] = {
    {0,0,0},
    {1,0,0}, {0,1,0}, {0,0,1},
    {2,0,0}, {0,2,0}, {0,0,2},
    {1,1,0}, {1,0,1}, {0,1,1},
    {2,1,0}, {2,0,1}, {1,2,0}, {0,2,1}, {1,0,2}, {0,1,2},
    {1,1,1},
    {2,2,0}, {2,0,2}, {0,2,2},
    {2,1,1}, {1,2,1}, {1,1,2},
    {2,2,1}, {2,1,2}, {1,2,2},
    {2,2,2}
};

// 1D polynomial norms squared: ||p0||^2=1, ||p1||^2=1/3, ||p2||^2=2/9
static const double poly_norm2[3] = { 1.0, 1.0/3.0, 2.0/9.0 };

// 1D orthogonal polynomial: p0(xi)=1, p1(xi)=xi, p2(xi)=xi^2-1/3
static inline double _mrt_poly1D(int order, int xi) {
    switch (order) {
        case 0: return 1.0;
        case 1: return (double)xi;
        case 2: return (double)(xi * xi) - 1.0/3.0;
        default: return 0.0;
    }
}

// ============================================================================
// Host-side function: compute M[27][27] and M_inv[27][27]
// Call once at initialization, then copy to GPU constant memory.
// ============================================================================
static void ComputeD3Q27_MRT_Matrices(double M_out[NQ][NQ], double Mi_out[NQ][NQ])
{

    // Build M[n][alpha] = p_a(xix_alpha) * p_b(xiy_alpha) * p_c(xiz_alpha)
    for (int n = 0; n < NQ; n++) {
        int a = MRT27_abc[n][0];
        int b = MRT27_abc[n][1];
        int c = MRT27_abc[n][2];
        for (int alpha = 0; alpha < NQ; alpha++) {
            M_out[n][alpha] = _mrt_poly1D(a, D3Q27_ex[alpha])
                            * _mrt_poly1D(b, D3Q27_ey[alpha])
                            * _mrt_poly1D(c, D3Q27_ez[alpha]);
        }
    }

    // Build M_inv[alpha][n] = w_alpha * M[n][alpha] / ||m_n||^2_w
    // where ||m_n||^2_w = ||p_a||^2 * ||p_b||^2 * ||p_c||^2
    for (int alpha = 0; alpha < NQ; alpha++) {
        for (int n = 0; n < NQ; n++) {
            int a = MRT27_abc[n][0];
            int b = MRT27_abc[n][1];
            int c = MRT27_abc[n][2];
            double norm2_n = poly_norm2[a] * poly_norm2[b] * poly_norm2[c];
            Mi_out[alpha][n] = D3Q27_W[alpha] * M_out[n][alpha] / norm2_n;
        }
    }
}

// ============================================================================
// Verify M * M_inv = I (host-side unit test)
// Returns max absolute error (should be < 1e-14)
// ============================================================================
static double VerifyMRT27_Identity(const double M[NQ][NQ], const double Mi[NQ][NQ])
{
    double max_err = 0.0;
    for (int i = 0; i < NQ; i++) {
        for (int j = 0; j < NQ; j++) {
            double sum = 0.0;
            for (int k = 0; k < NQ; k++)
                sum += M[i][k] * Mi[k][j];
            double expected = (i == j) ? 1.0 : 0.0;
            double err = fabs(sum - expected);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// ============================================================================
// Setup relaxation rates S_diag[27] per Suga et al. (2015) / Kuwata & Suga (2017)
//
// s_visc = 1/tau where tau = 3nu/dt + 0.5
// For tensor-product basis:
//   - Moments 0-3 (conserved): s = 0
//   - Moments 4-9 (stress): s = s_visc = 1/tau
//   - Moments 10-15 (energy flux): s = s_q = 1.5
//   - Moment 16 (xixxiyxiz): s = 1.4
//   - Moments 17-19 (4th order): s = 1.54
//   - Moments 20-22 (mixed 4th): s = 1.83
//   - Moments 23-25 (5th order): s = 1.98
//   - Moment 26 (6th order): s = 1.74
// ============================================================================
static void SetupD3Q27_Relaxation(double S_diag[NQ], double s_visc)
{
    // Conserved moments (density + momentum)
    S_diag[0]  = 0.0;
    S_diag[1]  = 0.0;
    S_diag[2]  = 0.0;
    S_diag[3]  = 0.0;

    // Stress tensor moments (viscosity-related)
    S_diag[4]  = s_visc;  // sigmaxx
    S_diag[5]  = s_visc;  // sigmayy
    S_diag[6]  = s_visc;  // sigmazz
    S_diag[7]  = s_visc;  // sigmaxy
    S_diag[8]  = s_visc;  // sigmaxz
    S_diag[9]  = s_visc;  // sigmayz

    // Energy flux moments (3rd order)
    S_diag[10] = 1.5;
    S_diag[11] = 1.5;
    S_diag[12] = 1.5;
    S_diag[13] = 1.5;
    S_diag[14] = 1.5;
    S_diag[15] = 1.5;

    // xixxiyxiz (3rd order)
    S_diag[16] = 1.4;

    // 4th order ghost moments
    S_diag[17] = 1.54;
    S_diag[18] = 1.54;
    S_diag[19] = 1.54;

    // Mixed 4th order
    S_diag[20] = 1.83;
    S_diag[21] = 1.83;
    S_diag[22] = 1.83;

    // 5th order
    S_diag[23] = 1.98;
    S_diag[24] = 1.98;
    S_diag[25] = 1.98;

    // 6th order
    S_diag[26] = 1.74;
}

#endif // MRT_MATRIX_D3Q27_FILE
