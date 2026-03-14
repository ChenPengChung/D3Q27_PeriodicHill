#ifndef MRT_CM_SHIFT_OPERATOR_H
#define MRT_CM_SHIFT_OPERATOR_H

// ============================================================================
// MRT-CM Shift Operator for d'Humières D3Q19 Moment Basis
// ============================================================================
//
// Derived via exact symbolic computation:
//   T_dH(u) = A · T_poly(u) · A^{-1}
// where:
//   A = M · N^{-1}  (d'Humières → polynomial basis conversion)
//   T_poly(u) = binomial shift in natural polynomial basis
//   N = polynomial basis evaluated at D3Q19 velocities
//
// Forward shift: κ = T(u) · m   (raw → central moments)
// Inverse shift: m = T^{-1}(u) · κ  (central → raw moments)
//   T^{-1}(u) is obtained by replacing u → -u in T(u), i.e. T(u)·T(-u) = I
//
// d'Humières moment ordering (matching MRT_Matrix.h):
//   m0  = ρ       (density,     conserved)
//   m1  = e       (energy,      2nd order)
//   m2  = ε       (energy²,     4th order effective)
//   m3  = jx      (x-momentum,  conserved)
//   m4  = qx      (energy flux, 3rd order effective)
//   m5  = jy      (y-momentum,  conserved)
//   m6  = qy      (energy flux, 3rd order effective)
//   m7  = jz      (z-momentum,  conserved)
//   m8  = qz      (energy flux, 3rd order effective)
//   m9  = 3pxx    (normal stress diagonal, 2nd order)
//   m10 = 3πxx    (stress ghost,  4th order effective)
//   m11 = pww     (normal stress off-diag, 2nd order)
//   m12 = πww     (stress ghost, 4th order effective)
//   m13 = pxy     (shear stress, 2nd order)
//   m14 = pyz     (shear stress, 2nd order)
//   m15 = pxz     (shear stress, 2nd order)
//   m16 = mx      (3rd order kinetic)
//   m17 = my      (3rd order kinetic)
//   m18 = mz      (3rd order kinetic)
//
// Reference: Chávez-Modena et al., J. Comput. Sci. (2018, 2020)
//            Dubois et al., Comput. Math. Appl. (2015)
// ============================================================================

// ============================================================================
// Forward shift: raw moments m[0..18] → central moments κ[0..18]
// κ = T(u) · m
// ============================================================================
__device__ __forceinline__ void raw_to_central_dH(
    const double m[19],   // input:  raw moments (d'Humières basis)
    double ux, double uy, double uz,  // local macroscopic velocity
    double k[19]          // output: central moments (d'Humières basis)
) {
    // Precompute velocity products
    const double ux2 = ux * ux;
    const double uy2 = uy * uy;
    const double uz2 = uz * uz;
    const double u2  = ux2 + uy2 + uz2;  // |u|²
    const double uxuy = ux * uy;
    const double uxuz = ux * uz;
    const double uyuz = uy * uz;

    // ---- Conserved moments: unchanged ----
    // κ0 = m0
    k[0] = m[0];

    // κ3 = m3  (+ ux*m0, but since m3 = ρ*ux by definition, shift cancels)
    // Actually the shift gives: κ3 = -ux*m0 + m3 = -ux*ρ + ρ*ux = 0
    // But we keep the formal expression for non-equilibrium accuracy:
    k[3] = -ux * m[0] + m[3];
    k[5] = -uy * m[0] + m[5];
    k[7] = -uz * m[0] + m[7];

    // ---- 2nd order: energy ----
    // κ1 = 19*|u|²*m0 + m1 - 38*ux*m3 - 38*uy*m5 - 38*uz*m7
    k[1] = 19.0 * u2 * m[0] + m[1]
          - 38.0 * ux * m[3] - 38.0 * uy * m[5] - 38.0 * uz * m[7];

    // ---- 2nd order: diagonal stress ----
    // κ9 = (2*ux²-uy²-uz²)*m0 - 4*ux*m3 + 2*uy*m5 + 2*uz*m7 + m9
    k[9] = (2.0*ux2 - uy2 - uz2) * m[0]
          - 4.0*ux * m[3] + 2.0*uy * m[5] + 2.0*uz * m[7] + m[9];

    // κ11 = (uy²-uz²)*m0 - 2*uy*m5 + 2*uz*m7 + m11
    k[11] = (uy2 - uz2) * m[0]
           - 2.0*uy * m[5] + 2.0*uz * m[7] + m[11];

    // ---- 2nd order: off-diagonal (shear) stress ----
    // κ13 = ux*uy*m0 - uy*m3 - ux*m5 + m13
    k[13] = uxuy * m[0] - uy * m[3] - ux * m[5] + m[13];

    // κ14 = uy*uz*m0 - uz*m5 - uy*m7 + m14
    k[14] = uyuz * m[0] - uz * m[5] - uy * m[7] + m[14];

    // κ15 = ux*uz*m0 - uz*m3 - ux*m7 + m15
    k[15] = uxuz * m[0] - uz * m[3] - ux * m[7] + m[15];

    // ---- Energy flux (effective 3rd order) ----
    // Exact: T[4,0] = -5*ux*uy²-5*ux*uz²-24*ux/19
    k[4] = (-5.0*ux*uy2 - 5.0*ux*uz2 - (24.0/19.0)*ux) * m[0]
          - (10.0/57.0) * ux * m[1]
          + (5.0*(uy2+uz2)) * m[3] + m[4]
          + 10.0*uxuy * m[5] + 10.0*uxuz * m[7]
          + (5.0/3.0)*ux * m[9]
          - 10.0*uy * m[13] - 10.0*uz * m[15];

    // Exact: T[6,0] = -5*ux²*uy-5*uy*uz²-24*uy/19
    k[6] = (-5.0*ux2*uy - 5.0*uy*uz2 - (24.0/19.0)*uy) * m[0]
          - (10.0/57.0) * uy * m[1]
          + 10.0*uxuy * m[3] + (5.0*(ux2+uz2)) * m[5] + m[6]
          + 10.0*uyuz * m[7]
          - (5.0/6.0)*uy * m[9] + (5.0/2.0)*uy * m[11]
          - 10.0*ux * m[13] - 10.0*uz * m[14];

    // Exact: T[8,0] = -5*ux²*uz-5*uy²*uz-24*uz/19
    k[8] = (-5.0*ux2*uz - 5.0*uy2*uz - (24.0/19.0)*uz) * m[0]
          - (10.0/57.0) * uz * m[1]
          + 10.0*uxuz * m[3] + 10.0*uyuz * m[5]
          + (5.0*(ux2+uy2)) * m[7] + m[8]
          - (5.0/6.0)*uz * m[9] - (5.0/2.0)*uz * m[11]
          - 10.0*uy * m[14] - 10.0*ux * m[15];

    // ---- Higher-order energy (4th order effective) ----
    // κ2 = (21*ux²uy²+21*ux²uz²+116*ux²/19+21*uy²uz²+116*uy²/19+116*uz²/19)*m0
    //    + (14/19)*|u|²*m1 + m2
    //    + (-2*ux*(105*uy²+105*uz²+4)/5)*m3 + (-42*ux/5)*m4
    //    + (-2*uy*(105*ux²+105*uz²+4)/5)*m5 + (-42*uy/5)*m6
    //    + (-2*uz*(105*ux²+105*uy²+4)/5)*m7 + (-42*uz/5)*m8
    //    + (-7*ux²+7*uy²/2+7*uz²/2)*m9
    //    + (-21*uy²/2+21*uz²/2)*m11
    //    + 84*ux*uy*m13 + 84*uy*uz*m14 + 84*ux*uz*m15
    // Exact: T[2,3] = -42*ux*uy²-42*ux*uz²-8*ux/5
    k[2] = (21.0*(ux2*uy2 + ux2*uz2 + uy2*uz2) + (116.0/19.0)*u2) * m[0]
          + (14.0/19.0) * u2 * m[1] + m[2]
          + (-42.0*ux*uy2 - 42.0*ux*uz2 - (8.0/5.0)*ux) * m[3] - (42.0/5.0)*ux * m[4]
          + (-42.0*ux2*uy - 42.0*uy*uz2 - (8.0/5.0)*uy) * m[5] - (42.0/5.0)*uy * m[6]
          + (-42.0*ux2*uz - 42.0*uy2*uz - (8.0/5.0)*uz) * m[7] - (42.0/5.0)*uz * m[8]
          + (-7.0*ux2 + 3.5*uy2 + 3.5*uz2) * m[9]
          + (-10.5*uy2 + 10.5*uz2) * m[11]
          + 84.0*uxuy * m[13] + 84.0*uyuz * m[14] + 84.0*uxuz * m[15];

    // ---- Stress ghost (4th order effective) ----
    // κ10 = (...complex...)*m0 + (2ux²/19-uy²/19-uz²/19)*m1
    //     + (2*ux*(-15uy²-15uz²+8)/5)*m3 + (-6ux/5)*m4
    //     + (2*uy*(-15ux²+30uz²-4)/5)*m5 + (3uy/5)*m6
    //     + (2*uz*(-15ux²+30uy²-4)/5)*m7 + (3uz/5)*m8
    //     + (-ux²+2uy²+2uz²)*m9 + m10 + (3uy²-3uz²)*m11
    //     + 12*uxuy*m13 - 24*uyuz*m14 + 12*uxuz*m15
    //     + 9*uy*m17 - 9*uz*m18
    // Exact: T[10,3] = -6*ux*uy²-6*ux*uz²+16*ux/5
    //        T[10,5] = -6*ux²*uy+12*uy*uz²-8*uy/5
    //        T[10,7] = -6*ux²*uz+12*uy²*uz-8*uz/5
    k[10] = (3.0*(ux2*uy2 + ux2*uz2) - 6.0*uy2*uz2
            - (16.0/19.0)*ux2 + (8.0/19.0)*uy2 + (8.0/19.0)*uz2) * m[0]
           + ((2.0*ux2 - uy2 - uz2)/19.0) * m[1]
           + (-6.0*ux*uy2 - 6.0*ux*uz2 + (16.0/5.0)*ux) * m[3] - (6.0/5.0)*ux * m[4]
           + (-6.0*ux2*uy + 12.0*uy*uz2 - (8.0/5.0)*uy) * m[5] + (3.0/5.0)*uy * m[6]
           + (-6.0*ux2*uz + 12.0*uy2*uz - (8.0/5.0)*uz) * m[7] + (3.0/5.0)*uz * m[8]
           + (-ux2 + 2.0*uy2 + 2.0*uz2) * m[9] + m[10]
           + (3.0*uy2 - 3.0*uz2) * m[11]
           + 12.0*uxuy * m[13] - 24.0*uyuz * m[14] + 12.0*uxuz * m[15]
           + 9.0*uy * m[17] - 9.0*uz * m[18];

    // κ12 = (3*ux²uy²-3*ux²uz²-8*uy²/19+8*uz²/19)*m0 + (uy²/19-uz²/19)*m1
    //     - 6*ux*(uy²-uz²)*m3
    //     + (2*uy*(4-15*ux²)/5)*m5 + (-3*uy/5)*m6
    //     + (2*uz*(15*ux²-4)/5)*m7 + (3*uz/5)*m8
    //     + (uy²-uz²)*m9 + 3*ux²*m11 + m12
    //     + 12*uxuy*m13 - 12*uxuz*m15
    //     - 6*ux*m16 + 3*uy*m17 + 3*uz*m18
    k[12] = (3.0*ux2*(uy2-uz2) + (8.0/19.0)*(uz2-uy2)) * m[0]
           + (uy2-uz2)/19.0 * m[1]
           - 6.0*ux*(uy2-uz2) * m[3]
           + (2.0/5.0)*uy*(4.0-15.0*ux2) * m[5] - (3.0/5.0)*uy * m[6]
           + (2.0/5.0)*uz*(15.0*ux2-4.0) * m[7] + (3.0/5.0)*uz * m[8]
           + (uy2-uz2) * m[9] + 3.0*ux2 * m[11] + m[12]
           + 12.0*uxuy * m[13] - 12.0*uxuz * m[15]
           - 6.0*ux * m[16] + 3.0*uy * m[17] + 3.0*uz * m[18];

    // ---- 3rd order kinetic moments ----
    // κ16 = ux*(-uy²+uz²)*m0 + (uy²-uz²)*m3 + 2*uxuy*m5 + (-2*uxuz)*m7
    //     - ux*m11 - 2*uy*m13 + 2*uz*m15 + m16
    k[16] = ux*(uz2-uy2) * m[0]
           + (uy2-uz2) * m[3] + 2.0*uxuy * m[5] - 2.0*uxuz * m[7]
           - ux * m[11] - 2.0*uy * m[13] + 2.0*uz * m[15] + m[16];

    // κ17 = uy*(ux²-uz²)*m0 + (-2*uxuy)*m3 + (-ux²+uz²)*m5 + 2*uyuz*m7
    //     + (uy/2)*m9 + (uy/2)*m11 + 2*ux*m13 - 2*uz*m14 + m17
    k[17] = uy*(ux2-uz2) * m[0]
           - 2.0*uxuy * m[3] + (uz2-ux2) * m[5] + 2.0*uyuz * m[7]
           + 0.5*uy * m[9] + 0.5*uy * m[11]
           + 2.0*ux * m[13] - 2.0*uz * m[14] + m[17];

    // κ18 = uz*(-ux²+uy²)*m0 + 2*uxuz*m3 + (-2*uyuz)*m5 + (ux²-uy²)*m7
    //     + (-uz/2)*m9 + (uz/2)*m11 + 2*uy*m14 - 2*ux*m15 + m18
    k[18] = uz*(uy2-ux2) * m[0]
           + 2.0*uxuz * m[3] - 2.0*uyuz * m[5] + (ux2-uy2) * m[7]
           - 0.5*uz * m[9] + 0.5*uz * m[11]
           + 2.0*uy * m[14] - 2.0*ux * m[15] + m[18];
}


// ============================================================================
// Inverse shift: central moments κ[0..18] → raw moments m[0..18]
// m = T^{-1}(u) · κ  =  T(-u) · κ
// (obtained by replacing u → -u in the forward shift, i.e., flip sign of all
//  odd-power-of-u terms)
// ============================================================================
__device__ __forceinline__ void central_to_raw_dH(
    const double k[19],   // input:  central moments (d'Humières basis)
    double ux, double uy, double uz,  // local macroscopic velocity
    double m[19]          // output: raw moments (d'Humières basis)
) {
    // Precompute velocity products
    const double ux2 = ux * ux;
    const double uy2 = uy * uy;
    const double uz2 = uz * uz;
    const double u2  = ux2 + uy2 + uz2;
    const double uxuy = ux * uy;
    const double uxuz = ux * uz;
    const double uyuz = uy * uz;

    // ---- Conserved moments ----
    m[0] = k[0];
    m[3] = ux * k[0] + k[3];
    m[5] = uy * k[0] + k[5];
    m[7] = uz * k[0] + k[7];

    // ---- 2nd order: energy ----
    m[1] = 19.0 * u2 * k[0] + k[1]
          + 38.0 * ux * k[3] + 38.0 * uy * k[5] + 38.0 * uz * k[7];

    // ---- 2nd order: diagonal stress ----
    m[9] = (2.0*ux2 - uy2 - uz2) * k[0]
          + 4.0*ux * k[3] - 2.0*uy * k[5] - 2.0*uz * k[7] + k[9];

    m[11] = (uy2 - uz2) * k[0]
           + 2.0*uy * k[5] - 2.0*uz * k[7] + k[11];

    // ---- 2nd order: shear stress ----
    m[13] = uxuy * k[0] + uy * k[3] + ux * k[5] + k[13];
    m[14] = uyuz * k[0] + uz * k[5] + uy * k[7] + k[14];
    m[15] = uxuz * k[0] + uz * k[3] + ux * k[7] + k[15];

    // ---- Energy flux ----
    m[4] = ux * (95.0*(uy2+uz2) + 24.0) / 19.0 * k[0]
          + (10.0/57.0) * ux * k[1]
          + (5.0*(uy2+uz2)) * k[3] + k[4]
          + 10.0*uxuy * k[5] + 10.0*uxuz * k[7]
          - (5.0/3.0)*ux * k[9]
          + 10.0*uy * k[13] + 10.0*uz * k[15];

    m[6] = uy * (95.0*(ux2+uz2) + 24.0) / 19.0 * k[0]
          + (10.0/57.0) * uy * k[1]
          + 10.0*uxuy * k[3] + (5.0*(ux2+uz2)) * k[5] + k[6]
          + 10.0*uyuz * k[7]
          + (5.0/6.0)*uy * k[9] - (5.0/2.0)*uy * k[11]
          + 10.0*ux * k[13] + 10.0*uz * k[14];

    m[8] = uz * (95.0*(ux2+uy2) + 24.0) / 19.0 * k[0]
          + (10.0/57.0) * uz * k[1]
          + 10.0*uxuz * k[3] + 10.0*uyuz * k[5]
          + (5.0*(ux2+uy2)) * k[7] + k[8]
          + (5.0/6.0)*uz * k[9] + (5.0/2.0)*uz * k[11]
          + 10.0*uy * k[14] + 10.0*ux * k[15];

    // ---- Higher-order energy ----
    m[2] = (21.0*(ux2*uy2 + ux2*uz2 + uy2*uz2) + (116.0/19.0)*u2) * k[0]
          + (14.0/19.0) * u2 * k[1] + k[2]
          + (2.0/5.0)*ux*(105.0*(uy2+uz2) + 4.0) * k[3] + (42.0/5.0)*ux * k[4]
          + (2.0/5.0)*uy*(105.0*(ux2+uz2) + 4.0) * k[5] + (42.0/5.0)*uy * k[6]
          + (2.0/5.0)*uz*(105.0*(ux2+uy2) + 4.0) * k[7] + (42.0/5.0)*uz * k[8]
          + (-7.0*ux2 + 3.5*uy2 + 3.5*uz2) * k[9]
          + (-10.5*uy2 + 10.5*uz2) * k[11]
          + 84.0*uxuy * k[13] + 84.0*uyuz * k[14] + 84.0*uxuz * k[15];

    // ---- Stress ghost ----
    m[10] = (3.0*(ux2*uy2 + ux2*uz2) - 6.0*uy2*uz2
            - (16.0/19.0)*ux2 + (8.0/19.0)*uy2 + (8.0/19.0)*uz2) * k[0]
           + ((2.0*ux2 - uy2 - uz2)/19.0) * k[1]
           + (2.0/5.0)*ux*(15.0*(uy2+uz2) - 8.0) * k[3] + (6.0/5.0)*ux * k[4]
           + (2.0/5.0)*uy*(15.0*ux2 - 30.0*uz2 + 4.0) * k[5] - (3.0/5.0)*uy * k[6]
           + (2.0/5.0)*uz*(15.0*ux2 - 30.0*uy2 + 4.0) * k[7] - (3.0/5.0)*uz * k[8]
           + (-ux2 + 2.0*uy2 + 2.0*uz2) * k[9] + k[10]
           + (3.0*uy2 - 3.0*uz2) * k[11]
           + 12.0*uxuy * k[13] - 24.0*uyuz * k[14] + 12.0*uxuz * k[15]
           - 9.0*uy * k[17] + 9.0*uz * k[18];

    m[12] = (3.0*ux2*(uy2-uz2) + (8.0/19.0)*(uz2-uy2)) * k[0]
           + (uy2-uz2)/19.0 * k[1]
           + 6.0*ux*(uy2-uz2) * k[3]
           + (2.0/5.0)*uy*(15.0*ux2-4.0) * k[5] + (3.0/5.0)*uy * k[6]
           + (2.0/5.0)*uz*(4.0-15.0*ux2) * k[7] - (3.0/5.0)*uz * k[8]
           + (uy2-uz2) * k[9] + 3.0*ux2 * k[11] + k[12]
           + 12.0*uxuy * k[13] - 12.0*uxuz * k[15]
           + 6.0*ux * k[16] - 3.0*uy * k[17] - 3.0*uz * k[18];

    // ---- 3rd order kinetic ----
    m[16] = ux*(uy2-uz2) * k[0]
           + (uy2-uz2) * k[3] + 2.0*uxuy * k[5] - 2.0*uxuz * k[7]
           + ux * k[11] + 2.0*uy * k[13] - 2.0*uz * k[15] + k[16];

    m[17] = uy*(uz2-ux2) * k[0]
           - 2.0*uxuy * k[3] + (uz2-ux2) * k[5] + 2.0*uyuz * k[7]
           - 0.5*uy * k[9] - 0.5*uy * k[11]
           - 2.0*ux * k[13] + 2.0*uz * k[14] + k[17];

    m[18] = uz*(ux2-uy2) * k[0]
           + 2.0*uxuz * k[3] - 2.0*uyuz * k[5] + (ux2-uy2) * k[7]
           + 0.5*uz * k[9] - 0.5*uz * k[11]
           - 2.0*uy * k[14] + 2.0*ux * k[15] + k[18];
}

// ============================================================================
// Dubois Continuous Polynomial Extension
// ============================================================================
// In the Dubois (2015) continuous polynomial version, x⁴ is NOT reduced to x²
// on the D3Q19 lattice. This introduces "ghost degrees of freedom" that cause:
//   - T_dubois(u)·T_dubois(-u) ≠ I  (approximate inverse, O(Ma²) error)
//   - Non-stress modes (rows 2,4,6,8,10,12) differ from lattice-reduced version
//   - Stress modes (rows 9,11,13,14,15) are IDENTICAL → same viscosity
//
// Implementation: T_dubois(u) = T_lattice(u) + ΔT(u)
// where ΔT only has nonzero entries in rows {2,4,6,8,10,12}
// and columns {0,1,3,5,7,9,11} (at most 7 moments per row).
//
// Correction polynomials: δ(x) = x²(x²-1), γ(x) = x(x²-1)
// These vanish on the lattice (|e_i|∈{0,1}) but NOT at shifted points (e_i-u).
// ============================================================================

#if USE_DUBOIS_CONTINUOUS

// ============================================================================
// Forward shift (Dubois continuous): raw → central moments
// κ = T_dubois(u) · m = T_lattice(u) · m + ΔT(u) · m
// ============================================================================
__device__ __forceinline__ void raw_to_central_dH_dubois(
    const double m[19],
    double ux, double uy, double uz,
    double k[19]
) {
    // First: compute lattice-reduced version (all 19 rows)
    raw_to_central_dH(m, ux, uy, uz, k);

    // Then: add corrections to rows {2,4,6,8,10,12}
    // These corrections come from ΔP = P_continuous - P_lattice
    // where δ(x) = x²(x²-1), γ(x) = x(x²-1)

    const double ux2 = ux * ux;
    const double uy2 = uy * uy;
    const double uz2 = uz * uz;
    const double ux3 = ux2 * ux;
    const double uy3 = uy2 * uy;
    const double uz3 = uz2 * uz;
    const double ux4 = ux2 * ux2;
    const double uy4 = uy2 * uy2;
    const double uz4 = uz2 * uz2;

    // Row 2 (ε, higher-order energy):
    // ΔT[2,0]*m0 + ΔT[2,1]*m1 + ΔT[2,3]*m3 + ΔT[2,5]*m5 + ΔT[2,7]*m7
    //   + ΔT[2,9]*m9 + ΔT[2,11]*m11
    k[2] += (10.5*ux4 + (861.0/38.0)*ux2 + 10.5*uy4 + (861.0/38.0)*uy2
           + 10.5*uz4 + (861.0/38.0)*uz2) * m[0]
          + (21.0/19.0) * (ux2 + uy2 + uz2) * m[1]
          + (-42.0*ux3 - 21.0*ux) * m[3]
          + (-42.0*uy3 - 21.0*uy) * m[5]
          + (-42.0*uz3 - 21.0*uz) * m[7]
          + (21.0*ux2 - 10.5*uy2 - 10.5*uz2) * m[9]
          + (31.5*uy2 - 31.5*uz2) * m[11];

    // Row 4 (qx, energy flux x):
    // ΔT[4,0]*m0 + ΔT[4,1]*m1 + ΔT[4,3]*m3 + ΔT[4,9]*m9
    k[4] += (-5.0*ux3 - (55.0/19.0)*ux) * m[0]
          + (-5.0/19.0) * ux * m[1]
          + 15.0*ux2 * m[3]
          + (-5.0*ux) * m[9];

    // Row 6 (qy, energy flux y):
    // ΔT[6,0]*m0 + ΔT[6,1]*m1 + ΔT[6,5]*m5 + ΔT[6,9]*m9 + ΔT[6,11]*m11
    k[6] += (-5.0*uy3 - (55.0/19.0)*uy) * m[0]
          + (-5.0/19.0) * uy * m[1]
          + 15.0*uy2 * m[5]
          + 2.5*uy * m[9]
          + (-7.5*uy) * m[11];

    // Row 8 (qz, energy flux z):
    // ΔT[8,0]*m0 + ΔT[8,1]*m1 + ΔT[8,7]*m7 + ΔT[8,9]*m9 + ΔT[8,11]*m11
    k[8] += (-5.0*uz3 - (55.0/19.0)*uz) * m[0]
          + (-5.0/19.0) * uz * m[1]
          + 15.0*uz2 * m[7]
          + 2.5*uz * m[9]
          + 7.5*uz * m[11];

    // Row 10 (3πxx, stress ghost diagonal):
    // ΔT[10,0]*m0 + ΔT[10,1]*m1 + ΔT[10,3]*m3 + ΔT[10,5]*m5 + ΔT[10,7]*m7
    //   + ΔT[10,9]*m9 + ΔT[10,11]*m11
    k[10] += (6.0*ux4 + (246.0/19.0)*ux2 - 3.0*uy4 - (123.0/19.0)*uy2
            - 3.0*uz4 - (123.0/19.0)*uz2) * m[0]
           + (12.0*ux2 - 6.0*uy2 - 6.0*uz2) / 19.0 * m[1]
           + (-24.0*ux3 - 12.0*ux) * m[3]
           + (12.0*uy3 + 6.0*uy) * m[5]
           + (12.0*uz3 + 6.0*uz) * m[7]
           + (12.0*ux2 + 3.0*uy2 + 3.0*uz2) * m[9]
           + (-9.0*uy2 + 9.0*uz2) * m[11];

    // Row 12 (πww, stress ghost off-diagonal):
    // ΔT[12,0]*m0 + ΔT[12,1]*m1 + ΔT[12,5]*m5 + ΔT[12,7]*m7
    //   + ΔT[12,9]*m9 + ΔT[12,11]*m11
    k[12] += (3.0*uy4 + (123.0/19.0)*uy2 - 3.0*uz4 - (123.0/19.0)*uz2) * m[0]
           + (6.0*uy2 - 6.0*uz2) / 19.0 * m[1]
           + (-12.0*uy3 - 6.0*uy) * m[5]
           + (12.0*uz3 + 6.0*uz) * m[7]
           + (-3.0*uy2 + 3.0*uz2) * m[9]
           + (9.0*uy2 + 9.0*uz2) * m[11];
}

// ============================================================================
// Inverse shift (Dubois continuous): central → raw moments
// m = T_dubois(-u) · κ  (NOTE: T_dubois(-u) ≠ T_dubois⁻¹(u), O(Ma²) error)
// Same correction structure with sign flip on odd-power terms
// ============================================================================
__device__ __forceinline__ void central_to_raw_dH_dubois(
    const double k[19],
    double ux, double uy, double uz,
    double m[19]
) {
    // First: compute lattice-reduced inverse (all 19 rows)
    central_to_raw_dH(k, ux, uy, uz, m);

    // Then: add corrections for T(-u) on rows {2,4,6,8,10,12}
    // For T(-u): even powers stay same, odd powers flip sign vs T(u)

    const double ux2 = ux * ux;
    const double uy2 = uy * uy;
    const double uz2 = uz * uz;
    const double ux3 = ux2 * ux;
    const double uy3 = uy2 * uy;
    const double uz3 = uz2 * uz;
    const double ux4 = ux2 * ux2;
    const double uy4 = uy2 * uy2;
    const double uz4 = uz2 * uz2;

    // Row 2: even terms same, odd terms flip sign
    m[2] += (10.5*ux4 + (861.0/38.0)*ux2 + 10.5*uy4 + (861.0/38.0)*uy2
           + 10.5*uz4 + (861.0/38.0)*uz2) * k[0]
          + (21.0/19.0) * (ux2 + uy2 + uz2) * k[1]
          + (42.0*ux3 + 21.0*ux) * k[3]   // sign flipped
          + (42.0*uy3 + 21.0*uy) * k[5]   // sign flipped
          + (42.0*uz3 + 21.0*uz) * k[7]   // sign flipped
          + (21.0*ux2 - 10.5*uy2 - 10.5*uz2) * k[9]
          + (31.5*uy2 - 31.5*uz2) * k[11];

    // Row 4: odd overall in ux → all terms flip sign
    m[4] += (5.0*ux3 + (55.0/19.0)*ux) * k[0]     // sign flipped
          + (5.0/19.0) * ux * k[1]                   // sign flipped
          + 15.0*ux2 * k[3]                           // even in ux → same
          + 5.0*ux * k[9];                            // sign flipped

    // Row 6: odd overall in uy → all terms flip sign
    m[6] += (5.0*uy3 + (55.0/19.0)*uy) * k[0]
          + (5.0/19.0) * uy * k[1]
          + 15.0*uy2 * k[5]
          + (-2.5*uy) * k[9]
          + 7.5*uy * k[11];

    // Row 8: odd overall in uz → all terms flip sign
    m[8] += (5.0*uz3 + (55.0/19.0)*uz) * k[0]
          + (5.0/19.0) * uz * k[1]
          + 15.0*uz2 * k[7]
          + (-2.5*uz) * k[9]
          + (-7.5*uz) * k[11];

    // Row 10: even overall → signs of odd-ux/uy/uz terms flip
    m[10] += (6.0*ux4 + (246.0/19.0)*ux2 - 3.0*uy4 - (123.0/19.0)*uy2
            - 3.0*uz4 - (123.0/19.0)*uz2) * k[0]
           + (12.0*ux2 - 6.0*uy2 - 6.0*uz2) / 19.0 * k[1]
           + (24.0*ux3 + 12.0*ux) * k[3]   // sign flipped
           + (-12.0*uy3 - 6.0*uy) * k[5]   // sign flipped
           + (-12.0*uz3 - 6.0*uz) * k[7]   // sign flipped
           + (12.0*ux2 + 3.0*uy2 + 3.0*uz2) * k[9]
           + (-9.0*uy2 + 9.0*uz2) * k[11];

    // Row 12: even overall → signs of odd-uy/uz terms flip
    m[12] += (3.0*uy4 + (123.0/19.0)*uy2 - 3.0*uz4 - (123.0/19.0)*uz2) * k[0]
           + (6.0*uy2 - 6.0*uz2) / 19.0 * k[1]
           + (12.0*uy3 + 6.0*uy) * k[5]    // sign flipped
           + (-12.0*uz3 - 6.0*uz) * k[7]   // sign flipped
           + (-3.0*uy2 + 3.0*uz2) * k[9]
           + (9.0*uy2 + 9.0*uz2) * k[11];
}

#endif // USE_DUBOIS_CONTINUOUS

#endif // MRT_CM_SHIFT_OPERATOR_H
