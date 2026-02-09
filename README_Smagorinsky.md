# Smagorinsky LES Subgrid Model Implementation

## Overview

This document describes the implementation of the Smagorinsky Large Eddy Simulation (LES) subgrid-scale model for the D3Q19 MRT Lattice Boltzmann Method, following the procedure outlined in the referenced paper.

## Motivation

At high Reynolds numbers (Re ≥ 700), the base relaxation time τ approaches the stability limit (τ → 0.5), causing numerical instability:
- τ = 0.6833 is very close to 0.5
- ν = (τ - 0.5) / 3 × dt becomes very small
- Re = U × L / ν requires small ν for high Re

The Smagorinsky model adds **eddy viscosity** in regions of high strain rate, effectively increasing τ locally to maintain stability.

---

## Mathematical Formulation

### Step 1: Nonequilibrium Stress Tensor

The nonequilibrium part of the stress tensor is computed from the distribution functions:

$$\Pi_{\alpha\beta} = \sum_{i} e_{i\alpha} e_{i\beta} (f_i - f_i^{eq})$$

where:
- $f_i$ is the distribution function
- $f_i^{eq}$ is the equilibrium distribution function
- $e_{i\alpha}$ is the lattice velocity component in direction $\alpha$

For D3Q19 lattice, the stress tensor components are:

| Component | Formula |
|-----------|---------|
| $\Pi_{xx}$ | $\sum_i e_{ix}^2 \cdot f_i^{neq}$ for directions with $e_x \neq 0$ |
| $\Pi_{yy}$ | $\sum_i e_{iy}^2 \cdot f_i^{neq}$ for directions with $e_y \neq 0$ |
| $\Pi_{zz}$ | $\sum_i e_{iz}^2 \cdot f_i^{neq}$ for directions with $e_z \neq 0$ |
| $\Pi_{xy}$ | $\sum_i e_{ix} e_{iy} \cdot f_i^{neq}$ for directions with $e_x e_y \neq 0$ |
| $\Pi_{xz}$ | $\sum_i e_{ix} e_{iz} \cdot f_i^{neq}$ for directions with $e_x e_z \neq 0$ |
| $\Pi_{yz}$ | $\sum_i e_{iy} e_{iz} \cdot f_i^{neq}$ for directions with $e_y e_z \neq 0$ |

### Step 2: Second Invariant of Strain Rate Tensor

$$Q = \Pi_{\alpha\beta} \Pi_{\alpha\beta} = \Pi_{xx}^2 + \Pi_{yy}^2 + \Pi_{zz}^2 + 2(\Pi_{xy}^2 + \Pi_{xz}^2 + \Pi_{yz}^2)$$

### Step 3: Strain Rate Magnitude

The strain rate magnitude $|\bar{S}|$ is computed using the analytical solution:

$$|\bar{S}| = \frac{\sqrt{\nu_0^2 + 18 C_s^2 \Delta^2 \sqrt{Q}} - \nu_0}{6 C_s^2 \Delta^2}$$

where:
- $\nu_0$ = base kinematic viscosity = $(τ - 0.5) / 3 \times dt$
- $C_s$ = Smagorinsky constant (typically 0.1 - 0.2)
- $\Delta$ = filter width (grid spacing in lattice units)

### Step 4: Total Viscosity and Relaxation Time

$$\nu_{total} = \nu_0 + (C_s \Delta)^2 |\bar{S}|$$

$$\tau_{total} = 3 \frac{\nu_{total}}{dt} + 0.5$$

The viscosity-related MRT relaxation rates are then updated:

$$s_9 = s_{11} = s_{13} = s_{14} = s_{15} = \frac{1}{\tau_{total}}$$

---

## Implementation Details

### Configuration Parameters (`variables.h`)

```cpp
#define SMAGORINSKY  1          // 1=enable LES, 0=disable
#define C_Smag       0.1        // Smagorinsky constant (typical: 0.1-0.2)
#define DELTA        (1.0)      // Filter width in lattice units
```

### D3Q19 Lattice Velocities

| Direction | $e_x$ | $e_y$ | $e_z$ |
|-----------|-------|-------|-------|
| f0  | 0  | 0  | 0  |
| f1  | 1  | 0  | 0  |
| f2  | -1 | 0  | 0  |
| f3  | 0  | 1  | 0  |
| f4  | 0  | -1 | 0  |
| f5  | 0  | 0  | 1  |
| f6  | 0  | 0  | -1 |
| f7  | 1  | 1  | 0  |
| f8  | -1 | 1  | 0  |
| f9  | 1  | -1 | 0  |
| f10 | -1 | -1 | 0  |
| f11 | 1  | 0  | 1  |
| f12 | -1 | 0  | 1  |
| f13 | 1  | 0  | -1 |
| f14 | -1 | 0  | -1 |
| f15 | 0  | 1  | 1  |
| f16 | 0  | -1 | 1  |
| f17 | 0  | 1  | -1 |
| f18 | 0  | -1 | -1 |

### Stress Tensor Implementation (`evolution.h`)

Based on the lattice velocities, the stress tensor components are:

```cpp
// Pi_xx: directions with ex^2 = 1
Pi_xx = fneq1 + fneq2 + fneq7 + fneq8 + fneq9 + fneq10 + fneq11 + fneq12 + fneq13 + fneq14;

// Pi_yy: directions with ey^2 = 1
Pi_yy = fneq3 + fneq4 + fneq7 + fneq8 + fneq9 + fneq10 + fneq15 + fneq16 + fneq17 + fneq18;

// Pi_zz: directions with ez^2 = 1
Pi_zz = fneq5 + fneq6 + fneq11 + fneq12 + fneq13 + fneq14 + fneq15 + fneq16 + fneq17 + fneq18;

// Pi_xy: directions with ex*ey = ±1
Pi_xy = fneq7 - fneq8 - fneq9 + fneq10;  // (+1,+1), (-1,+1), (+1,-1), (-1,-1)

// Pi_xz: directions with ex*ez = ±1
Pi_xz = fneq11 - fneq12 - fneq13 + fneq14;

// Pi_yz: directions with ey*ez = ±1
Pi_yz = fneq15 - fneq16 - fneq17 + fneq18;
```

### Complete Algorithm Flow

```
1. Stream & gather f_in from neighboring nodes
2. Compute macroscopic quantities (ρ, u, v, w)
3. Compute equilibrium distributions f_eq
4. [LES] Calculate f_neq = f_in - f_eq
5. [LES] Compute stress tensor Π_αβ
6. [LES] Compute Q = Π_αβ Π_αβ
7. [LES] Compute |S| from analytical formula
8. [LES] Update τ_total and relaxation rates s9,s11,s13,s14,s15
9. Apply MRT collision with updated relaxation matrix
10. Store f_new
```

---

## Modified Files

| File | Changes |
|------|---------|
| `variables.h` | Added `SMAGORINSKY`, `C_Smag`, `DELTA` parameters |
| `evolution.h` | Added Smagorinsky calculation in both `stream_collide_Buffer` and `stream_collide` kernels |

---

## Usage

### Enable/Disable LES
```cpp
#define SMAGORINSKY  1   // Enable
#define SMAGORINSKY  0   // Disable (use base τ)
```

### Tuning Smagorinsky Constant

| $C_s$ Value | Effect |
|-------------|--------|
| 0.1 | Standard value, minimal dissipation |
| 0.15 | Moderate dissipation, better stability |
| 0.2 | Higher dissipation, maximum stability |

**Recommendation**: Start with `C_Smag = 0.1`. If simulation still diverges or oscillates, increase to 0.15 or 0.2.

---

## Stability Safeguard

A minimum τ threshold is enforced to prevent instability:

```cpp
if (tau_total < 0.505) tau_total = 0.505;
```

This ensures τ never approaches the theoretical stability limit of 0.5.

---

## References

The implementation follows the Smagorinsky LES procedure for LBM as described in lattice Boltzmann turbulence modeling literature, where the eddy viscosity is computed locally from the nonequilibrium stress tensor rather than velocity gradients.

Key formula source:
$$|\bar{S}| = \frac{\sqrt{\nu_0^2 + 18 C_s^2 \Delta^2 \sqrt{Q}} - \nu_0}{6 C_s^2 \Delta^2}$$

This analytical form allows direct computation of the strain rate magnitude from the second invariant Q of the nonequilibrium stress tensor.
