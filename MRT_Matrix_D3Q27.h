#ifndef MRT_MATRIX_D3Q27_FILE
#define MRT_MATRIX_D3Q27_FILE

// ============================================================================
// D3Q27 MRT Transformation Matrix (Suga et al. 2015, Comp. Math. Appl. 69)
// ============================================================================
//
// Construction: Tensor-product orthogonal basis from 1D polynomials
//   p0(ξ) = 1,  p1(ξ) = ξ,  p2(ξ) = ξ² - 1/3
// These are orthogonal w.r.t. 1D weights w(-1)=1/6, w(0)=2/3, w(1)=1/6.
//
// 3D moments M[n][α] = p_a(ξx_α) × p_b(ξy_α) × p_c(ξz_α)
// where (a,b,c) is the polynomial index triplet for moment n.
//
// Inverse: M_inv[α][n] = w_α × M[n][α] / ||m_n||²_w
// where ||m_n||²_w = ||p_a||² × ||p_b||² × ||p_c||²
//
// Norms: ||p0||²=1, ||p1||²=1/3, ||p2||²=2/9
//
// The moment ordering is chosen to match physical interpretation:
//   0: density (conserved)
//   1-3: momentum (conserved)
//   4-9: stress tensor (viscosity-related, s = 1/τ)
//   10-15: energy flux (3rd order)
//   16: ξxξyξz (3rd order)
//   17-26: higher-order ghost moments
// ============================================================================

// Number of discrete velocities
#define NQ 27

// D3Q27 velocity vectors (matching D3Q19 ordering for indices 0-18, corners at 19-26)
static const int D3Q27_ex[NQ] = { 0,  1,-1, 0, 0, 0, 0,  1,-1, 1,-1,  1,-1, 1,-1, 0, 0, 0, 0,   1,-1, 1,-1, 1,-1, 1,-1};
static const int D3Q27_ey[NQ] = { 0,  0, 0, 1,-1, 0, 0,  1, 1,-1,-1,  0, 0, 0, 0, 1,-1, 1,-1,   1, 1,-1,-1, 1, 1,-1,-1};
static const int D3Q27_ez[NQ] = { 0,  0, 0, 0, 0, 1,-1,  0, 0, 0, 0,  1, 1,-1,-1, 1, 1,-1,-1,   1, 1, 1, 1,-1,-1,-1,-1};

// D3Q27 weights (factored: w = wx × wy × wz, where w(0)=2/3, w(±1)=1/6)
static const double D3Q27_W[NQ] = {
    8.0/27.0,                                       // rest: (2/3)^3
    2.0/27.0, 2.0/27.0, 2.0/27.0,                  // face: (1/6)(2/3)(2/3) × 6
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,        // edge: (1/6)(1/6)(2/3) × 12
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,    // corner: (1/6)^3 × 8
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

// ============================================================================
// Moment ordering: (a,b,c) polynomial index triplet
// ============================================================================
// Row n of M: M[n][α] = p_a(ξx_α) × p_b(ξy_α) × p_c(ξz_α)
//
// Index | (a,b,c) | Physical meaning                  | Relaxation
// ------|---------|-----------------------------------|----------
//   0   | (0,0,0) | ρ  (density)                      | s=0 (conserved)
//   1   | (1,0,0) | jx (x-momentum)                   | s=0 (conserved)
//   2   | (0,1,0) | jy (y-momentum)                   | s=0 (conserved)
//   3   | (0,0,1) | jz (z-momentum)                   | s=0 (conserved)
//   4   | (2,0,0) | σxx = ξx²-1/3 (normal stress)     | s=1/τ (viscosity)
//   5   | (0,2,0) | σyy = ξy²-1/3 (normal stress)     | s=1/τ (viscosity)
//   6   | (0,0,2) | σzz = ξz²-1/3 (normal stress)     | s=1/τ (viscosity)
//   7   | (1,1,0) | σxy = ξxξy    (shear stress)      | s=1/τ (viscosity)
//   8   | (1,0,1) | σxz = ξxξz    (shear stress)      | s=1/τ (viscosity)
//   9   | (0,1,1) | σyz = ξyξz    (shear stress)      | s=1/τ (viscosity)
//  10   | (2,1,0) | qx(y) = (ξx²-1/3)ξy              | s_q (energy flux)
//  11   | (2,0,1) | qx(z) = (ξx²-1/3)ξz              | s_q
//  12   | (1,2,0) | qy(x) = ξx(ξy²-1/3)              | s_q
//  13   | (0,2,1) | qy(z) = (ξy²-1/3)ξz              | s_q
//  14   | (1,0,2) | qz(x) = ξx(ξz²-1/3)              | s_q
//  15   | (0,1,2) | qz(y) = ξy(ξz²-1/3)              | s_q
//  16   | (1,1,1) | ξxξyξz                            | s_16
//  17   | (2,2,0) | (ξx²-1/3)(ξy²-1/3)               | s_ghost
//  18   | (2,0,2) | (ξx²-1/3)(ξz²-1/3)               | s_ghost
//  19   | (0,2,2) | (ξy²-1/3)(ξz²-1/3)               | s_ghost
//  20   | (2,1,1) | (ξx²-1/3)ξyξz                    | s_ghost
//  21   | (1,2,1) | ξx(ξy²-1/3)ξz                    | s_ghost
//  22   | (1,1,2) | ξxξy(ξz²-1/3)                    | s_ghost
//  23   | (2,2,1) | (ξx²-1/3)(ξy²-1/3)ξz             | s_ghost
//  24   | (2,1,2) | (ξx²-1/3)ξy(ξz²-1/3)             | s_ghost
//  25   | (1,2,2) | ξx(ξy²-1/3)(ξz²-1/3)             | s_ghost
//  26   | (2,2,2) | (ξx²-1/3)(ξy²-1/3)(ξz²-1/3)      | s_ghost

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

// 1D polynomial norms squared: ||p0||²=1, ||p1||²=1/3, ||p2||²=2/9
static const double poly_norm2[3] = { 1.0, 1.0/3.0, 2.0/9.0 };

// 1D orthogonal polynomials evaluated at ξ ∈ {-1, 0, 1}
// p0(ξ) = 1,  p1(ξ) = ξ,  p2(ξ) = ξ² - 1/3
static double poly1D(int order, int xi) {
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

    // Build M[n][α] = p_a(ξx_α) × p_b(ξy_α) × p_c(ξz_α)
    for (int n = 0; n < NQ; n++) {
        int a = MRT27_abc[n][0];
        int b = MRT27_abc[n][1];
        int c = MRT27_abc[n][2];
        for (int alpha = 0; alpha < NQ; alpha++) {
            M_out[n][alpha] = poly1D(a, D3Q27_ex[alpha])
                            * poly1D(b, D3Q27_ey[alpha])
                            * poly1D(c, D3Q27_ez[alpha]);
        }
    }

    // Build M_inv[α][n] = w_α × M[n][α] / ||m_n||²_w
    // where ||m_n||²_w = ||p_a||² × ||p_b||² × ||p_c||²
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
// Verify M × M_inv = I (host-side unit test)
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
// Setup relaxation rates S_diag[27]
//
// Strategy: TRT with magic parameter Λ = 1/4 (Ginzburg 2008)
//
//   s⁺ = s_visc = 1/τ       (physical viscosity, determined by Re)無因次化鬆弛時間的倒數
//   s⁻ = 1 / (Λ/(τ⁺-0.5) + 0.5)   (ghost mode damping, auto-adaptive)
//
// Why Λ = 1/4?
//   - Bounce-back wall location error = 0 (exact halfway)
//   - Critical for wall-bounded flows (Periodic Hill)
//   - s⁻ adapts to Re automatically: never exceeds 2.0
//
// Previous: Suga et al. (2015) fixed rates (S[23-25]=1.98 → |G|=0.98)
// Current:  TRT Λ=1/4 adaptive  (s⁻ ≈ 1.20 @Re100 → |G|=0.20)
//           Ghost decay: 100× faster, no sign-alternating oscillation
//
// Moment groups:
//   0-3  (conserved): s = 0
//   4-9  (stress):    s = s⁺ = s_visc  (viscosity)
//   10-26 (ghost):    s = s⁻            (TRT Λ=1/4)//讓高階矩的鬆弛絕對不超過2
// ============================================================================
static void SetupD3Q27_Relaxation(double S_diag[NQ], double s_visc)
{
    // ---- s⁺: physical viscosity rate (unchanged) ----
    double s_plus = s_visc;

    // ---- s⁻: ghost mode rate from TRT magic parameter Λ = 1/4 ----
    //   Λ = (τ⁺ - 1/2)(τ⁻ - 1/2)
    //   τ⁺ = 1/s⁺,  τ⁻ = Λ/(τ⁺ - 1/2) + 1/2
    //   s⁻ = 1/τ⁻
    double tau_plus  = 1.0 / s_plus;
    double Lambda    = 0.25;  // magic parameter
    double tau_minus = Lambda / (tau_plus - 0.5) + 0.5;
    double s_minus   = 1.0 / tau_minus;

    // Conserved moments (density + momentum): s = 0
    S_diag[0]  = 0.0;
    S_diag[1]  = 0.0;
    S_diag[2]  = 0.0;
    S_diag[3]  = 0.0;

    // Stress tensor moments: s = s⁺ (determines kinematic viscosity ν)
    S_diag[4]  = s_plus;   // σxx
    S_diag[5]  = s_plus;   // σyy
    S_diag[6]  = s_plus;   // σzz
    S_diag[7]  = s_plus;   // σxy
    S_diag[8]  = s_plus;   // σxz
    S_diag[9]  = s_plus;   // σyz

    // All ghost moments (3rd–6th order): s = s⁻ (TRT Λ=1/4)
    for (int n = 10; n < NQ; n++)
        S_diag[n] = s_minus;
}

#endif // MRT_MATRIX_D3Q27_FILE
