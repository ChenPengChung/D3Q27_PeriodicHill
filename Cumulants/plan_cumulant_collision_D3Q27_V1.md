# Plan: Standalone D3Q27 Cumulant Collision Operator for GILBM

## Design Goal — A Black-Box Collision Module

Design the Cumulant collision operator as a **self-contained CUDA device function** with the following interface:

```
═══════════════════════════════════════════════════════════════
                    BLACK-BOX INTERFACE
═══════════════════════════════════════════════════════════════

  INPUT (5 items):
  ┌─────────────────────────────────────────────────────────┐
  │ 1. f_streamed[27]  — post-streaming distribution fns    │
  │ 2. feq[27]         — equilibrium distribution fns       │
  │ 3. dt_global       — LBM global time step (= minSize)  │
  │ 4. F_body[3]       — macroscopic body force (Fx,Fy,Fz) │
  │ 5. omega            — relaxation rate (from nu_LB)      │
  └─────────────────────────────────────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │   Cumulant Collision    │
             │   (5 internal stages)   │
             └─────────────────────────┘
                           │
                           ▼
  OUTPUT:
  ┌─────────────────────────────────────────────────────────┐
  │ f_post[27]  — post-collision distribution functions      │
  │ rho         — density (computed internally)              │
  │ u[3]        — velocity with half-force correction        │
  └─────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
```

---

## Internal Architecture — 5 Stages

Based on the OpenLB implementation (Geier et al. 2015, Appendix J),
the Cumulant collision proceeds in **5 sequential stages**:

```
 f_streamed[27]
      │
      ▼
 ┌──────────────────────────────────────────────────┐
 │  STAGE 0: Macroscopic Quantities                 │
 │  ─────────────────────────────────────────────    │
 │  Compute ρ, u, v, w from f_streamed[27]          │
 │  Apply half-force correction to velocity:        │
 │    u_corr = u + 0.5*F/ρ*dt                       │
 │  Compute δρ = ρ - 1, 1/ρ                         │
 │  Convert to well-conditioned: f̄ = f - w_α        │
 └──────────────────────────────────────────────────┘
      │
      ▼
 ┌──────────────────────────────────────────────────┐
 │  STAGE 1: Forward Central Moment Transformation  │
 │  (Chimera transform: z → y → x)                  │
 │  ─────────────────────────────────────────────    │
 │  f̄_ijk → κ_αβγ (27 central moments)              │
 │                                                   │
 │  For each direction d = z, y, x:                  │
 │    For each triplet (a, b, c) in velocityIndices: │
 │      κ₀ = f₁ + f₀ + f₋₁           (Eq. J.4/7/10)│
 │      κ₁ = (f₁-f₋₁) - u·(κ₀+K)    (Eq. J.5/8/11)│
 │      κ₂ = (f₁+f₋₁) - 2u(f₁-f₋₁)               │
 │           + u²·(κ₀+K)              (Eq. J.6/9/12)│
 │                                                   │
 │  K constants from lattice weights (cum.h K<3,27>) │
 └──────────────────────────────────────────────────┘
      │
      ▼  27 central moments: κ_αβγ
 ┌──────────────────────────────────────────────────┐
 │  STAGE 2: Central Moments → Cumulants            │
 │  ─────────────────────────────────────────────    │
 │  Orders 0–3: C_αβγ = κ_αβγ (identical)           │
 │  Order 4:    subtract products of lower orders    │
 │  Order 5:    subtract products of lower orders    │
 │  Order 6:    subtract all lower-order products    │
 │                                                   │
 │  ★ Key formulas (well-conditioned version):       │
 │                                                   │
 │  4th order (Eq. J.16):                            │
 │    C₂₁₁ = κ₂₁₁ - [(κ₂₀₀+⅓)κ₀₁₁               │
 │            + 2κ₁₁₀κ₁₀₁] / ρ                     │
 │                                                   │
 │  4th order (Eq. J.17):                            │
 │    C₂₂₀ = κ₂₂₀ - [(κ₂₀₀κ₀₂₀ + 2κ₁₁₀²)        │
 │            + ⅓(κ₂₀₀+κ₀₂₀)] / ρ + δρ/(9ρ)       │
 │                                                   │
 │  5th order (Eq. J.18):                            │
 │    C₁₂₂ = κ₁₂₂ - [(κ₀₀₂κ₁₂₀ + κ₀₂₀κ₁₀₂       │
 │            + 4κ₀₁₁κ₁₁₁ + 2(κ₁₀₁κ₀₂₁            │
 │            + κ₁₁₀κ₀₁₂)) + ⅓(κ₁₂₀+κ₁₀₂)] / ρ   │
 │                                                   │
 │  6th order (Eq. J.19): [see full expression]      │
 └──────────────────────────────────────────────────┘
      │
      ▼  27 cumulants: C_αβγ
 ┌──────────────────────────────────────────────────┐
 │  STAGE 3: Relaxation (Collision)                  │
 │  C_αβγ → C*_αβγ                                  │
 │  ─────────────────────────────────────────────    │
 │  Each order relaxes with its own rate toward 0:   │
 │                                                   │
 │  ┌─────────┬────────────┬──────────────────────┐ │
 │  │  Order  │  ω symbol  │  Cumulants affected  │ │
 │  ├─────────┼────────────┼──────────────────────┤ │
 │  │ 0th     │  (conserv) │  ρ (no relaxation)   │ │
 │  │ 1st     │  (conserv) │  j_x,j_y,j_z (none) │ │
 │  │ 2nd off │  ω₁=omega  │  C₁₁₀,C₁₀₁,C₀₁₁   │ │
 │  │ 2nd diag│  ω₁,ω₂    │  trace + deviatoric  │ │
 │  │ 3rd sym │  ω₃        │  C₂₁₀+C₀₁₂, etc.   │ │
 │  │ 3rd anti│  ω₄        │  C₂₁₀-C₀₁₂, etc.   │ │
 │  │ 3rd     │  ω₄(=ω₅)  │  C₁₁₁                │ │
 │  │ 4th     │  ω₆(=ω₇=ω₈)│ C₂₂₀,C₂₀₂,C₀₂₂   │ │
 │  │         │            │  C₂₁₁,C₁₂₁,C₁₁₂    │ │
 │  │ 5th     │  ω₉        │  C₂₂₁,C₂₁₂,C₁₂₂   │ │
 │  │ 6th     │  ω₁₀       │  C₂₂₂               │ │
 │  └─────────┴────────────┴──────────────────────┘ │
 │                                                   │
 │  ★ Only ω₁ affects Navier-Stokes viscosity:      │
 │    ν = cs²(1/ω₁ - 1/2)δt                         │
 │  ★ All others (ω₂–ω₁₀) set to 1 by default      │
 │    (full relaxation to equilibrium)               │
 │                                                   │
 │  ★ 2nd order diagonal requires special treatment: │
 │    Decompose into orthogonal modes:               │
 │      trace  = C₂₀₀+C₀₂₀+C₀₀₂  → relax with ω₂  │
 │      dev1   = C₂₀₀-C₀₂₀        → relax with ω₁  │
 │      dev2   = C₂₀₀-C₀₀₂        → relax with ω₁  │
 │    Then reconstruct:                              │
 │      C*₂₀₀ = ⅓(dev1+dev2+trace)                  │
 │      C*₀₂₀ = ⅓(-2·dev1+dev2+trace)               │
 │      C*₀₀₂ = ⅓(dev1-2·dev2+trace)                │
 └──────────────────────────────────────────────────┘
      │
      ▼  27 post-collision cumulants: C*_αβγ
 ┌──────────────────────────────────────────────────┐
 │  STAGE 4: Cumulants → Central Moments             │
 │  (Inverse of Stage 2)                             │
 │  ─────────────────────────────────────────────    │
 │  Add back the lower-order product terms:          │
 │                                                   │
 │  4th order (Eq. J.16 inverse):                    │
 │    κ*₂₁₁ = C*₂₁₁ + [(κ*₂₀₀+⅓)κ*₀₁₁            │
 │             + 2κ*₁₁₀κ*₁₀₁] / ρ                  │
 │  (and all permutations)                           │
 │                                                   │
 │  ★ Uses POST-collision 2nd & 3rd order moments    │
 │    (already relaxed in Stage 3)                   │
 │                                                   │
 │  ★ Force correction applied here:                 │
 │    κ*₁₀₀ = -κ₁₀₀  (Eq. 85)                      │
 │    κ*₀₁₀ = -κ₀₁₀  (Eq. 86)                      │
 │    κ*₀₀₁ = -κ₀₀₁  (Eq. 87)                      │
 │    (sign flip ensures time-symmetric forcing)     │
 └──────────────────────────────────────────────────┘
      │
      ▼  27 post-collision central moments: κ*_αβγ
 ┌──────────────────────────────────────────────────┐
 │  STAGE 5: Backward Central Moment Transformation  │
 │  (Inverse Chimera: x → y → z)                    │
 │  ─────────────────────────────────────────────    │
 │  κ*_αβγ → f̄*_ijk (well-conditioned populations)  │
 │                                                   │
 │  For each direction d = x, y, z:                  │
 │    For each triplet (a, b, c):                    │
 │      f̄₋₁ = [(κ₀+K)(u²-u)+κ₁(2u-1)+κ₂]/2       │
 │                                   (Eq. J.21/24/27)│
 │      f̄₀  = κ₀(1-u²)-2uκ₁-κ₂    (Eq. J.20/23/26)│
 │      f̄₊₁ = [(κ₀+K)(u²+u)+κ₁(2u+1)+κ₂]/2       │
 │                                   (Eq. J.22/25/28)│
 │                                                   │
 │  Finally: f*_α = f̄*_α + w_α  (restore weights)   │
 └──────────────────────────────────────────────────┘
      │
      ▼
 f_post[27]  (output)
```

---

## Complete Data Flow Diagram

```
 Inputs                    Internal                    Output
 ──────                    ────────                    ──────

 f_streamed[27] ──┐
                   │     ┌──────────┐
                   ├────►│ STAGE 0  │──► ρ, u[3], δρ, 1/ρ
 F_body[3] ───────┤     │ Macro +  │──► u_corrected[3] (half-force)
                   │     │ f̄ = f-w  │──► f̄[27] (well-conditioned)
 omega ───────────┤     └──────────┘
                   │          │
 dt_global ───────┤          ▼
                   │     ┌──────────┐
                   │     │ STAGE 1  │──► κ[27] (central moments)
                   │     │ Chimera  │    via z→y→x sweep
                   │     │ Forward  │
                   │     └──────────┘
                   │          │
                   │          ▼
                   │     ┌──────────┐
                   │     │ STAGE 2  │──► C[27] (cumulants)
                   │     │ κ → C    │    subtract lower-order products
                   │     └──────────┘
                   │          │
                   │          ▼
                   │     ┌──────────┐
                   └────►│ STAGE 3  │──► C*[27] (post-collision)
                         │ Relax    │    ω₁ from input omega
                         │          │    ω₂–ω₁₀ = 1 (default)
                         └──────────┘
                              │
                              ▼
                         ┌──────────┐
                         │ STAGE 4  │──► κ*[27] (post-coll. moments)
                         │ C* → κ*  │    + force sign flip (Eq.85-87)
                         └──────────┘
                              │
                              ▼
                         ┌──────────┐
                         │ STAGE 5  │──► f_post[27]
                         │ Chimera  │
                         │ Backward │
                         └──────────┘
```

---

## CUDA Device Function Signature

```cpp
// ====================================================================
// Standalone Cumulant Collision for D3Q27
// ====================================================================
// Reference: Geier, M. et al., "The cumulant lattice Boltzmann
//            equation in three dimensions: Theory and validation."
//            Comp. Math. Appl. 70(4), 507–547, 2015.
//
// This is the well-conditioned formulation (Appendix J).
// ====================================================================

__device__ void cumulant_collision_D3Q27(
    // ===== INPUTS =====
    const double  f_in[27],    // post-streaming distribution functions
    const double  omega,       // primary relaxation rate (shear viscosity)
                               // ν = cs²(1/omega - 0.5)*dt
    const double  Fx,          // body force x-component
    const double  Fy,          // body force y-component
    const double  Fz,          // body force z-component
    const double  dt,          // global time step (dt_global)

    // ===== OUTPUTS =====
    double        f_out[27],   // post-collision distribution functions
    double&       rho_out,     // computed density
    double&       ux_out,      // corrected velocity x (with half-force)
    double&       uy_out,      // corrected velocity y
    double&       uz_out       // corrected velocity z
);
```

**Note on feq[27]:**
In the Cumulant method, the equilibrium distribution functions `feq[27]` are **NOT explicitly needed** as a separate input. The cumulant relaxation drives each cumulant toward its equilibrium value (which is 0 for all orders ≥ 2) directly in cumulant space. The equilibrium information is embedded implicitly:
- 2nd order diagonal equilibrium → $\rho$ (trace mode)
- All other equilibria → 0

This is a fundamental advantage over BGK/MRT: no need to compute $f^{eq}_\alpha$ explicitly.

---

## Detailed Variable Naming Convention

### Central Moment Notation (from OpenLB code)

The 27 central moments $\kappa_{\alpha\beta\gamma}$ where $\alpha,\beta,\gamma \in \{0,1,2\}$ are stored as a flat array `m[27]` and accessed via structured binding:

```
Index mapping: a→0, b→1, c→2

Notation:  m_XYZ  where X,Y,Z ∈ {a,b,c}
           a = order 0 in that dimension
           b = order 1
           c = order 2

Examples:
  maaa = κ₀₀₀ = ρ-1 (= δρ in well-conditioned form)
  mbaa = κ₁₀₀ = first moment x
  maba = κ₀₁₀ = first moment y
  maab = κ₀₀₁ = first moment z
  mcaa = κ₂₀₀ = second moment xx
  mabb = κ₀₁₁ = second moment yz (off-diagonal)
  mbba = κ₁₁₀ = second moment xy
  mbbb = κ₁₁₁ = third moment xyz
  ...
  mccc = κ₂₂₂ = sixth-order moment
```

### Full 27-element structured binding order:

```cpp
auto [mbbb, mabb, mbab, mbba, maab, macb, maba, mabc, mbaa,
      mbac, maaa, maac, maca, macc, mcbb, mbcb, mbbc, mccb,
      mcab, mcbc, mcba, mbcc, mbca, mccc, mcca, mcac, mcaa]
     = moments;
```

This ordering is dictated by the `velocityIndices<3,27>` array in `cum.h`.

---

## Constant Arrays Required

### 1. D3Q27 Lattice Weights (w_α)

```cpp
__constant__ double w27[27] = {
    // Rest (1): 8/27
    8.0/27.0,
    // Face (6): 2/27
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0,
    // Edge (12): 1/54
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    // Corner (8): 1/216
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};
```

### 2. K Constants (from cum.h, K<3,27>)

These are precomputed from lattice weights for the well-conditioned Chimera transform:

```cpp
__constant__ double K27[27] = {
    1.0,                          // K[0]
    0.0,    1.0/3.0,              // K[1], K[2]
    0.0,    0.0,    0.0,  1.0/3.0,// K[3..6]
    0.0,    1.0/9.0,              // K[7], K[8]
    1.0/6.0,                      // K[9]
    0.0,    1.0/18.0,             // K[10], K[11]
    2.0/3.0,                      // K[12]
    0.0,    2.0/9.0,              // K[13], K[14]
    1.0/6.0,                      // K[15]
    0.0,    1.0/18.0,             // K[16], K[17]
    1.0/36.0,                     // K[18]
    1.0/9.0,                      // K[19]
    1.0/36.0,                     // K[20]
    1.0/9.0,                      // K[21]
    4.0/9.0,                      // K[22]
    1.0/9.0,                      // K[23]
    1.0/36.0,                     // K[24]
    1.0/9.0,                      // K[25]
    1.0/36.0                      // K[26]
};
```

### 3. Velocity Indices (Chimera triplet mapping)

```cpp
// velocityIndices[27][3]: for each of 9 passes × 3 directions,
// maps (a, b, c) triplet indices into the 27-element moment array.
// Direction order: z(i=2), y(i=1), x(i=0), 9 triplets each.
__constant__ int velIdx[27][3] = {
    // --- z-direction (i=2): 9 triplets ---
    {10,  8, 26}, {12, 22, 24}, { 6,  3, 20},
    { 4,  2, 18}, { 1,  0, 14}, { 5, 15, 17},
    {11,  9, 25}, { 7, 16, 19}, {13, 21, 23},
    // --- y-direction (i=1): 9 triplets ---
    {10,  6, 12}, { 4,  1,  5}, {11,  7, 13},
    { 8,  3, 22}, { 2,  0, 15}, { 9, 16, 21},
    {26, 20, 24}, {18, 14, 17}, {25, 19, 23},
    // --- x-direction (i=0): 9 triplets ---
    {10,  4, 11}, { 6,  1,  7}, {12,  5, 13},
    { 8,  2,  9}, { 3,  0, 16}, {22, 15, 21},
    {26, 18, 25}, {20, 14, 19}, {24, 17, 23}
};
```

---

## Relaxation Rate Configuration

```
═══════════════════════════════════════════════════════════════
  Relaxation Rate Table (Geier et al. 2015, Eq. 55–80)
═══════════════════════════════════════════════════════════════

  Code var  │ Paper  │ Default │ Controls              │ Eq.
  ──────────┼────────┼─────────┼───────────────────────┼──────
  omega     │ ω₁     │ (input) │ Shear viscosity ν     │ 55–62
  omega2    │ ω₂     │ 1.0     │ Bulk viscosity        │ 63
  omega3    │ ω₃     │ 1.0     │ 3rd order symmetric   │ 64–66
  omega4    │ ω₄,ω₅  │ 1.0     │ 3rd order antisym     │ 67–70
  omega6    │ ω₆–ω₈  │ 1.0     │ 4th order             │ 71–76
  omega7    │ ω₉     │ 1.0     │ 5th order             │ 77–79
  omega10   │ ω₁₀    │ 1.0     │ 6th order             │ 80

  ★ ONLY omega (= ω₁) affects the physical viscosity.
  ★ Setting all others to 1.0 = full relaxation to equilibrium.
  ★ All rates must be in range (0, 2) for stability.
  ★ Galilean correction terms in Eq. 61–62 vanish when ω₂ = 1.

═══════════════════════════════════════════════════════════════
```

---

## Force Treatment — Time-Symmetric Splitting

The body force is incorporated via **half-force** (Guo-style, time-symmetric):

```
Before collision:
  u_corrected = u_raw + 0.5 * F / ρ * dt
  (use u_corrected in all Chimera transforms)

After collision (Stage 4, before backward Chimera):
  κ*₁₀₀ = -κ₁₀₀     (Eq. 85)
  κ*₀₁₀ = -κ₀₁₀     (Eq. 86)
  κ*₀₀₁ = -κ₀₀₁     (Eq. 87)
  (sign flip of 1st-order central moments = apply second half-force)
```

This ensures **second-order temporal accuracy** without explicitly constructing the Guo forcing term $F_\alpha$. It is simpler than the MRT forcing formulation.

---

## Integration with GILBM Evolution Kernel

```
═══════════════════════════════════════════════════════════════
  GILBM Evolution with Cumulant Collision (per lattice node)
═══════════════════════════════════════════════════════════════

  // Step 1: Interpolation + Streaming
  for (alpha = 0; alpha < 27; alpha++) {
      // Compute departure point from displacement arrays
      // 7-point Lagrange interpolation
      f_streamed[alpha] = lagrange_interpolate(...);
  }

  // Step 2: Cumulant Collision (THE BLACK BOX)
  double f_post[27];
  double rho, ux, uy, uz;

  cumulant_collision_D3Q27(
      f_streamed,          // input: post-streaming f
      omega_global,        // input: 1/(3*nu_LB + 0.5)
      F_x_body,            // input: streamwise body force
      0.0,                 // input: Fy (usually 0)
      0.0,                 // input: Fz (usually 0)
      dt_global,           // input: time step
      f_post,              // output: post-collision f
      rho, ux, uy, uz      // output: macroscopic quantities
  );

  // Step 3: Write back to global memory
  for (alpha = 0; alpha < 27; alpha++) {
      f[alpha * total_nodes + idx] = f_post[alpha];
  }
  rho_d[idx] = rho;
  ux_d[idx] = ux;
  uy_d[idx] = uy;
  uz_d[idx] = uz;

═══════════════════════════════════════════════════════════════
```

---

## Advantages of Cumulant over MRT and BGK

```
┌─────────────────┬────────┬─────────┬───────────┐
│ Feature         │ BGK    │ MRT     │ Cumulant  │
├─────────────────┼────────┼─────────┼───────────┤
│ Relaxation      │ Single │ 27 diag │ 10 groups │
│ parameters      │ (1/τ)  │ rates   │ (ω₁–ω₁₀) │
├─────────────────┼────────┼─────────┼───────────┤
│ Galilean        │ ✗ No   │ ✗ No    │ ✓ Yes     │
│ invariance      │        │         │ (built-in)│
├─────────────────┼────────┼─────────┼───────────┤
│ Need M matrix   │ No     │ Yes     │ No        │
│ (27×27)         │        │ (hard-  │ (Chimera  │
│                 │        │  coded) │  in-place)│
├─────────────────┼────────┼─────────┼───────────┤
│ Need feq[27]    │ Yes    │ Yes     │ No        │
│ explicitly      │        │(m_eq)   │(implicit) │
├─────────────────┼────────┼─────────┼───────────┤
│ Stability       │ Low    │ Medium  │ High      │
├─────────────────┼────────┼─────────┼───────────┤
│ Forcing         │ Guo    │ M⁻¹(I- │ Half-force│
│ treatment       │ F_α    │ S/2)MF  │ + sign    │
│                 │        │        │  flip     │
├─────────────────┼────────┼─────────┼───────────┤
│ Register usage  │ Low    │ High   │ Medium    │
│ (per thread)    │ ~30    │ ~80+   │ ~50       │
├─────────────────┼────────┼─────────┼───────────┤
│ FLOPs/node      │ ~250   │ ~1500  │ ~800      │
│ (estimated)     │        │(M*M⁻¹) │           │
└─────────────────┴────────┴─────────┴───────────┘
```

---

## Implementation Phases

```
Phase 1: Standalone Verification (CPU)
─────────────────────────────────────
  - Implement cumulant_collision_D3Q27() as a pure C++ function
  - Unit test: zero velocity → f_post = feq (all cumulants → 0)
  - Unit test: uniform flow → conserved quantities unchanged
  - Unit test: omega = 2.0 → identical to BGK with same omega
  - Compare against OpenLB reference output if available

Phase 2: CUDA Kernel Integration
─────────────────────────────────────
  - Port to __device__ function
  - Store K27, velIdx, w27 in __constant__ memory
  - Benchmark register usage: target < 64 registers/thread
  - Verify numerical equivalence with CPU version

Phase 3: GILBM Coupling
─────────────────────────────────────
  - Replace BGK collision in evolution_gilbm.h
  - Connect to existing displacement/interpolation pipeline
  - Connect to adaptive force controller (Part A)
  - Short run (1000 steps): stability check
  - Long run: benchmark against ERCOFTAC data

Phase 4: Tuning (Optional)
─────────────────────────────────────
  - Experiment with ω₂ ≠ 1 (bulk viscosity tuning)
  - Experiment with ω₃, ω₄ for improved stability
  - Profile GPU occupancy and optimize if needed
```

---

## Questions Before Implementation

```
Q1: Do you want the standalone function to accept feq[27]
    as input anyway (for compatibility), or strictly use the
    cumulant-native interface (no feq needed)?

Q2: Should ω₂–ω₁₀ be hardcoded as 1.0, or passed as
    additional parameters for future tunability?

Q3: For GILBM coupling: is the well-conditioned form
    (f̄ = f - w) handled inside the collision function,
    or should the caller subtract/add weights?

Q4: The OpenLB code uses structured bindings (C++17).
    Does your CUDA compiler (nvcc with sm_60) support C++17?
    If not, we need to use explicit indexing instead.

Q5: Preferred file structure:
    (A) Single header: cumulant_collision.h
    (B) Split: cum_constants.h + cum_chimera.h + cum_collision.h
```

---

## References

1. Geier M., Schönherr M., Pasquali A., Krafczyk M.,
   "The cumulant lattice Boltzmann equation in three dimensions:
    Theory and validation," *Comp. Math. Appl.* 70(4), 507–547, 2015.
2. OpenLB — Open Source Lattice Boltzmann Code,
   collisionCUM.h, cum.h, cumulantDynamics.h (GPL v2+).
3. Geier M., Greiner A., Korvink J.G.,
   "Cascaded digital lattice Boltzmann automata for high Reynolds
    number flow," *Phys. Rev. E* 73, 066705, 2006.
