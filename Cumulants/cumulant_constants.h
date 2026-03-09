// ================================================================
// cumulant_constants.h
// D3Q27 Cumulant Collision Constants
//
// Derived for THIS project's D3Q27 velocity ordering
// (MRT_Matrix_D3Q27.h: rest=0, faces=1-6, xy-edges=7-10,
//  xz-edges=11-14, yz-edges=15-18, corners=19-26)
//
// Source: OpenLB cum.h (GPL v2+) — constants re-derived for our ordering
// Reference: Geier et al., Comp. Math. Appl. 70(4), 507-547, 2015
// ================================================================
#ifndef CUMULANT_CONSTANTS_H
#define CUMULANT_CONSTANTS_H

// ================================================================
// Chimera triplet indices for our D3Q27 velocity ordering
// ================================================================
// 27 passes = 9 z-triplets + 9 y-triplets + 9 x-triplets
//
// Each row {a, b, c} groups three populations (or intermediate values):
//   Forward:  a = negative component, b = zero component, c = positive component
//   After transform: a → κ₀ (sum), b → κ₁ (1st moment), c → κ₂ (2nd moment)
//
// Z-sweep: group by (ex,ey), order by ez = {-1, 0, +1}
// Y-sweep: group by (ex, z-order), order by ey = {-1, 0, +1}
// X-sweep: group by (y-order, z-order), order by ex = {-1, 0, +1}
// ================================================================
__constant__ int CUM_IDX[27][3] = {
    // ---- z-direction (passes 0-8): group by (ex,ey), sweep ez ----
    //  (ex,ey)     ez=-1  ez=0  ez=+1
    //  (0,  0) :
    { 6,  0,  5},
    //  (1,  0) :
    {13,  1, 11},
    //  (-1, 0) :
    {14,  2, 12},
    //  (0,  1) :
    {17,  3, 15},
    //  (0, -1) :
    {18,  4, 16},
    //  (1,  1) :
    {23,  7, 19},
    //  (-1, 1) :
    {24,  8, 20},
    //  (1, -1) :
    {25,  9, 21},
    //  (-1,-1) :
    {26, 10, 22},

    // ---- y-direction (passes 9-17): group by (ex, z-order), sweep ey ----
    //  (ex,zord)   ey=-1  ey=0  ey=+1
    //  (0, 0) :
    {18,  6, 17},
    //  (0, 1) :
    { 4,  0,  3},
    //  (0, 2) :
    {16,  5, 15},
    //  (1, 0) :
    {25, 13, 23},
    //  (1, 1) :
    { 9,  1,  7},
    //  (1, 2) :
    {21, 11, 19},
    //  (-1, 0):
    {26, 14, 24},
    //  (-1, 1):
    {10,  2,  8},
    //  (-1, 2):
    {22, 12, 20},

    // ---- x-direction (passes 18-26): group by (y-order, z-order), sweep ex ----
    //  (yord,zord) ex=-1  ex=0  ex=+1
    //  (0, 0) :
    {26, 18, 25},
    //  (1, 0) :
    {14,  6, 13},
    //  (2, 0) :
    {24, 17, 23},
    //  (0, 1) :
    {10,  4,  9},
    //  (1, 1) :
    { 2,  0,  1},
    //  (2, 1) :
    { 8,  3,  7},
    //  (0, 2) :
    {22, 16, 21},
    //  (1, 2) :
    {12,  5, 11},
    //  (2, 2) :
    {20, 15, 19}
};

// ================================================================
// K constants for well-conditioned Chimera transform
// ================================================================
// Computed by applying the Chimera forward transform to the D3Q27
// weight array W[27] with u=0. K[p] = κ₀ (sum of triplet) at pass p,
// accounting for cascading through z → y → x sweeps.
//
// These correct for the well-conditioned formulation f̄ = f - w.
// ================================================================
__constant__ double CUM_K[27] = {
    // z-sweep (passes 0-8):
    //   K = sum of 3 weights in each z-triplet
    4.0/9.0,      // pass 0:  (0,0) group, W[6]+W[0]+W[5] = 2/27+8/27+2/27
    1.0/9.0,      // pass 1:  (1,0) group
    1.0/9.0,      // pass 2:  (-1,0) group
    1.0/9.0,      // pass 3:  (0,1) group
    1.0/9.0,      // pass 4:  (0,-1) group
    1.0/36.0,     // pass 5:  (1,1) group
    1.0/36.0,     // pass 6:  (-1,1) group
    1.0/36.0,     // pass 7:  (1,-1) group
    1.0/36.0,     // pass 8:  (-1,-1) group

    // y-sweep (passes 9-17):
    //   K = sum of 3 z-transformed values in each y-triplet
    2.0/3.0,      // pass 9:  (0,zord=0) group
    0.0,          // pass 10: (0,zord=1) group  (all zero after z-sweep)
    2.0/9.0,      // pass 11: (0,zord=2) group
    1.0/6.0,      // pass 12: (1,zord=0) group
    0.0,          // pass 13: (1,zord=1) group
    1.0/18.0,     // pass 14: (1,zord=2) group
    1.0/6.0,      // pass 15: (-1,zord=0) group
    0.0,          // pass 16: (-1,zord=1) group
    1.0/18.0,     // pass 17: (-1,zord=2) group

    // x-sweep (passes 18-26):
    //   K = sum of 3 y-transformed values in each x-triplet
    1.0,          // pass 18: (yord=0,zord=0) → total weight sum = 1
    0.0,          // pass 19: (yord=1,zord=0)
    1.0/3.0,      // pass 20: (yord=2,zord=0)
    0.0,          // pass 21: (yord=0,zord=1)
    0.0,          // pass 22: (yord=1,zord=1)
    0.0,          // pass 23: (yord=2,zord=1)
    1.0/3.0,      // pass 24: (yord=0,zord=2)
    0.0,          // pass 25: (yord=1,zord=2)
    1.0/9.0       // pass 26: (yord=2,zord=2)
};

// ================================================================
// 27-element moment array index aliases
// ================================================================
// After the full Chimera forward transform (z→y→x), array position [p]
// holds central moment κ_{αβγ} where α=x-order, β=y-order, γ=z-order.
//
// Naming: I_{αβγ} where a=order 0, b=order 1, c=order 2
//   maaa = κ₀₀₀ = δρ (density deviation)
//   mbaa = κ₁₀₀ = 1st moment x (after well-conditioning + Chimera)
//   mabb = κ₀₁₁ = off-diagonal stress yz
//   mcaa = κ₂₀₀ = diagonal stress xx
//   mbbb = κ₁₁₁ = 3rd order xyz
//   mccc = κ₂₂₂ = 6th order
//
// Derived from our velocity ordering by tracing through the Chimera sweeps.
// ================================================================

// 0th order
#define I_aaa 26    // κ₀₀₀ = δρ

// 1st order
#define I_baa 18    // κ₁₀₀ (x-momentum)
#define I_aba 14    // κ₀₁₀ (y-momentum)
#define I_aab 10    // κ₀₀₁ (z-momentum)

// 2nd order diagonal
#define I_caa 25    // κ₂₀₀ (xx-stress)
#define I_aca 24    // κ₀₂₀ (yy-stress)
#define I_aac 22    // κ₀₀₂ (zz-stress)

// 2nd order off-diagonal
#define I_bba  6    // κ₁₁₀ (xy-stress)
#define I_bab  4    // κ₁₀₁ (xz-stress)
#define I_abb  2    // κ₀₁₁ (yz-stress)

// 3rd order
#define I_bbb  0    // κ₁₁₁
#define I_cba 13    // κ₂₁₀
#define I_bca 17    // κ₁₂₀
#define I_cab  9    // κ₂₀₁
#define I_acb  8    // κ₀₂₁
#define I_bac 16    // κ₁₀₂
#define I_abc 12    // κ₀₁₂

// 4th order
#define I_cbb  1    // κ₂₁₁
#define I_bcb  3    // κ₁₂₁
#define I_bbc  5    // κ₁₁₂
#define I_cca 23    // κ₂₂₀
#define I_cac 21    // κ₂₀₂
#define I_acc 20    // κ₀₂₂
#define I_ccb  7    // κ₂₂₁

// 5th order
#define I_bcc 15    // κ₁₂₂
#define I_cbc 11    // κ₂₁₂

// 6th order
#define I_ccc 19    // κ₂₂₂

#endif // CUMULANT_CONSTANTS_H
