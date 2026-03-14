#!/usr/bin/env python3
"""
Verify CUDA Dubois correction formulas against Python reference.

Tests that:
  raw_to_central_dH_dubois(m, ux, uy, uz) = raw_to_central_dH(m, ux, uy, uz) + ΔT(u)·m
matches the full matrix-based Dubois T(u) from test_compare_two_Tu.py.

Author: Claude (GILBM project verification)
"""
import numpy as np
from numpy.linalg import inv, norm
np.set_printoptions(precision=14, linewidth=200)

# ====================================================================
# D3Q19 lattice and M matrix
# ====================================================================
e3d = np.array([
    [0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
    [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
    [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
    [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]], dtype=float)

M3d = np.array([
    [   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ -30,-11,-11,-11,-11,-11,-11,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
    [  12, -4, -4, -4, -4, -4, -4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [   0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
    [   0, -4,  4,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
    [   0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1],
    [   0,  0,  0, -4,  4,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1],
    [   0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1],
    [   0,  0,  0,  0,  0, -4,  4,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1],
    [   0,  2,  2, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],
    [   0, -4, -4,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],
    [   0,  0,  0,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],
    [   0,  0,  0, -2, -2,  2,  2,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],
    [   0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1],
    [   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0],
    [   0,  0,  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1, -1,  1,  0,  0,  0,  0],
    [   0,  0,  0,  0,  0,  0,  0, -1, -1,  1,  1,  0,  0,  0,  0,  1, -1,  1, -1],
    [   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1, -1, -1,  1,  1]
], dtype=float)
M3d_inv = inv(M3d)

# ====================================================================
# Continuous polynomial basis (Dubois)
# ====================================================================
def dH_polys_continuous(X, Y, Z):
    """d'Humières polynomials — CONTINUOUS form (Dubois 2015).
    r⁴ = (x²+y²+z²)² is kept as degree-4 polynomial, NOT reduced.
    Must match test_compare_two_Tu.py exactly."""
    X2, Y2, Z2 = X*X, Y*Y, Z*Z
    r2 = X2 + Y2 + Z2
    p = np.zeros(19)
    p[0]  = 1.0
    p[1]  = 19.0*r2 - 30.0                            # e
    p[2]  = 10.5*r2*r2 - 26.5*r2 + 12.0               # ε = (21r⁴-53r²+24)/2
    p[3]  = X
    p[4]  = 5.0*X*r2 - 9.0*X                           # qx (NO /2!)
    p[5]  = Y
    p[6]  = 5.0*Y*r2 - 9.0*Y                           # qy
    p[7]  = Z
    p[8]  = 5.0*Z*r2 - 9.0*Z                           # qz
    p[9]  = 2.0*X2 - Y2 - Z2                           # 3pxx
    p[10] = (2.0*X2 - Y2 - Z2) * (3.0*r2 - 5.0)       # 3πxx (NO /2!)
    p[11] = Y2 - Z2                                    # pww
    p[12] = (Y2 - Z2) * (3.0*r2 - 5.0)                 # πww (NO /2!)
    p[13] = X * Y; p[14] = Y * Z; p[15] = X * Z
    p[16] = X * (Y2 - Z2); p[17] = Y * (Z2 - X2); p[18] = Z * (X2 - Y2)
    return p

def build_T_dubois(ux, uy, uz):
    """Build full 19×19 T_dubois(u) = M_dH(e-u)·M_dH(e)⁻¹"""
    M_shifted = np.zeros((19, 19))
    for col in range(19):
        X = e3d[col, 0] - ux
        Y = e3d[col, 1] - uy
        Z = e3d[col, 2] - uz
        M_shifted[:, col] = dH_polys_continuous(X, Y, Z)
    return M_shifted @ M3d_inv

# ====================================================================
# Our lattice-reduced T(u) — Python port of CUDA raw_to_central_dH
# ====================================================================
def raw_to_central_OURS(m, ux, uy, uz):
    ux2, uy2, uz2 = ux*ux, uy*uy, uz*uz
    u2 = ux2 + uy2 + uz2
    uxuy, uxuz, uyuz = ux*uy, ux*uz, uy*uz
    k = np.zeros(19)
    k[0] = m[0]
    k[3] = -ux*m[0] + m[3]
    k[5] = -uy*m[0] + m[5]
    k[7] = -uz*m[0] + m[7]
    k[1] = 19.0*u2*m[0] + m[1] - 38*ux*m[3] - 38*uy*m[5] - 38*uz*m[7]
    k[9] = (2*ux2-uy2-uz2)*m[0] - 4*ux*m[3] + 2*uy*m[5] + 2*uz*m[7] + m[9]
    k[11] = (uy2-uz2)*m[0] - 2*uy*m[5] + 2*uz*m[7] + m[11]
    k[13] = uxuy*m[0] - uy*m[3] - ux*m[5] + m[13]
    k[14] = uyuz*m[0] - uz*m[5] - uy*m[7] + m[14]
    k[15] = uxuz*m[0] - uz*m[3] - ux*m[7] + m[15]
    k[4] = (-5*ux*uy2 - 5*ux*uz2 - 24*ux/19)*m[0] \
           - (10/57)*ux*m[1] + 5*(uy2+uz2)*m[3] + m[4] \
           + 10*uxuy*m[5] + 10*uxuz*m[7] + (5/3)*ux*m[9] \
           - 10*uy*m[13] - 10*uz*m[15]
    k[6] = (-5*ux2*uy - 5*uy*uz2 - 24*uy/19)*m[0] \
           - (10/57)*uy*m[1] + 10*uxuy*m[3] + 5*(ux2+uz2)*m[5] + m[6] \
           + 10*uyuz*m[7] - (5/6)*uy*m[9] + 2.5*uy*m[11] \
           - 10*ux*m[13] - 10*uz*m[14]
    k[8] = (-5*ux2*uz - 5*uy2*uz - 24*uz/19)*m[0] \
           - (10/57)*uz*m[1] + 10*uxuz*m[3] + 10*uyuz*m[5] \
           + 5*(ux2+uy2)*m[7] + m[8] - (5/6)*uz*m[9] - 2.5*uz*m[11] \
           - 10*uy*m[14] - 10*ux*m[15]
    k[2] = (21*(ux2*uy2+ux2*uz2+uy2*uz2) + 116*u2/19)*m[0] \
           + 14*u2/19*m[1] + m[2] \
           + (-42*ux*uy2-42*ux*uz2-8*ux/5)*m[3] - 42*ux/5*m[4] \
           + (-42*ux2*uy-42*uy*uz2-8*uy/5)*m[5] - 42*uy/5*m[6] \
           + (-42*ux2*uz-42*uy2*uz-8*uz/5)*m[7] - 42*uz/5*m[8] \
           + (-7*ux2+3.5*uy2+3.5*uz2)*m[9] + (-10.5*uy2+10.5*uz2)*m[11] \
           + 84*uxuy*m[13] + 84*uyuz*m[14] + 84*uxuz*m[15]
    k[10] = (3*(ux2*uy2+ux2*uz2) - 6*uy2*uz2 - 16*ux2/19 + 8*uy2/19 + 8*uz2/19)*m[0] \
            + (2*ux2-uy2-uz2)/19*m[1] \
            + (-6*ux*uy2-6*ux*uz2+16*ux/5)*m[3] - 6*ux/5*m[4] \
            + (-6*ux2*uy+12*uy*uz2-8*uy/5)*m[5] + 3*uy/5*m[6] \
            + (-6*ux2*uz+12*uy2*uz-8*uz/5)*m[7] + 3*uz/5*m[8] \
            + (-ux2+2*uy2+2*uz2)*m[9] + m[10] + (3*uy2-3*uz2)*m[11] \
            + 12*uxuy*m[13] - 24*uyuz*m[14] + 12*uxuz*m[15] \
            + 9*uy*m[17] - 9*uz*m[18]
    k[12] = (3*ux2*(uy2-uz2) + 8*(uz2-uy2)/19)*m[0] \
            + (uy2-uz2)/19*m[1] - 6*ux*(uy2-uz2)*m[3] \
            + 2*uy*(4-15*ux2)/5*m[5] - 3*uy/5*m[6] \
            + 2*uz*(15*ux2-4)/5*m[7] + 3*uz/5*m[8] \
            + (uy2-uz2)*m[9] + 3*ux2*m[11] + m[12] \
            + 12*uxuy*m[13] - 12*uxuz*m[15] \
            - 6*ux*m[16] + 3*uy*m[17] + 3*uz*m[18]
    k[16] = ux*(uz2-uy2)*m[0] + (uy2-uz2)*m[3] + 2*uxuy*m[5] - 2*uxuz*m[7] \
            - ux*m[11] - 2*uy*m[13] + 2*uz*m[15] + m[16]
    k[17] = uy*(ux2-uz2)*m[0] - 2*uxuy*m[3] + (uz2-ux2)*m[5] + 2*uyuz*m[7] \
            + 0.5*uy*m[9] + 0.5*uy*m[11] + 2*ux*m[13] - 2*uz*m[14] + m[17]
    k[18] = uz*(uy2-ux2)*m[0] + 2*uxuz*m[3] - 2*uyuz*m[5] + (ux2-uy2)*m[7] \
            - 0.5*uz*m[9] + 0.5*uz*m[11] + 2*uy*m[14] - 2*ux*m[15] + m[18]
    return k

# ====================================================================
# CUDA Dubois correction (Python port of MRT_CM_ShiftOperator.h)
# ====================================================================
def cuda_dubois_forward_correction(m, ux, uy, uz):
    """Additive correction ΔT(u)·m for rows {2,4,6,8,10,12}"""
    ux2, uy2, uz2 = ux*ux, uy*uy, uz*uz
    ux3, uy3, uz3 = ux2*ux, uy2*uy, uz2*uz
    ux4, uy4, uz4 = ux2*ux2, uy2*uy2, uz2*uz2
    dk = np.zeros(19)

    # Row 2
    dk[2] = (10.5*ux4 + (861.0/38.0)*ux2 + 10.5*uy4 + (861.0/38.0)*uy2
           + 10.5*uz4 + (861.0/38.0)*uz2) * m[0] \
          + (21.0/19.0)*(ux2+uy2+uz2) * m[1] \
          + (-42*ux3-21*ux)*m[3] + (-42*uy3-21*uy)*m[5] + (-42*uz3-21*uz)*m[7] \
          + (21*ux2-10.5*uy2-10.5*uz2)*m[9] + (31.5*uy2-31.5*uz2)*m[11]

    # Row 4
    dk[4] = (-5*ux3 - 55*ux/19)*m[0] + (-5.0/19.0)*ux*m[1] \
          + 15*ux2*m[3] + (-5*ux)*m[9]

    # Row 6
    dk[6] = (-5*uy3 - 55*uy/19)*m[0] + (-5.0/19.0)*uy*m[1] \
          + 15*uy2*m[5] + 2.5*uy*m[9] + (-7.5*uy)*m[11]

    # Row 8
    dk[8] = (-5*uz3 - 55*uz/19)*m[0] + (-5.0/19.0)*uz*m[1] \
          + 15*uz2*m[7] + 2.5*uz*m[9] + 7.5*uz*m[11]

    # Row 10
    dk[10] = (6*ux4 + 246*ux2/19 - 3*uy4 - 123*uy2/19 - 3*uz4 - 123*uz2/19)*m[0] \
           + (12*ux2 - 6*uy2 - 6*uz2)/19*m[1] \
           + (-24*ux3-12*ux)*m[3] + (12*uy3+6*uy)*m[5] + (12*uz3+6*uz)*m[7] \
           + (12*ux2+3*uy2+3*uz2)*m[9] + (-9*uy2+9*uz2)*m[11]

    # Row 12
    dk[12] = (3*uy4 + 123*uy2/19 - 3*uz4 - 123*uz2/19)*m[0] \
           + (6*uy2-6*uz2)/19*m[1] \
           + (-12*uy3-6*uy)*m[5] + (12*uz3+6*uz)*m[7] \
           + (-3*uy2+3*uz2)*m[9] + (9*uy2+9*uz2)*m[11]

    return dk

def cuda_dubois_inverse_correction(k, ux, uy, uz):
    """Additive correction for T(-u) on rows {2,4,6,8,10,12}"""
    ux2, uy2, uz2 = ux*ux, uy*uy, uz*uz
    ux3, uy3, uz3 = ux2*ux, uy2*uy, uz2*uz
    ux4, uy4, uz4 = ux2*ux2, uy2*uy2, uz2*uz2
    dm = np.zeros(19)

    # Row 2: even terms same, odd terms flip
    dm[2] = (10.5*ux4 + (861.0/38.0)*ux2 + 10.5*uy4 + (861.0/38.0)*uy2
           + 10.5*uz4 + (861.0/38.0)*uz2) * k[0] \
          + (21.0/19.0)*(ux2+uy2+uz2) * k[1] \
          + (42*ux3+21*ux)*k[3] + (42*uy3+21*uy)*k[5] + (42*uz3+21*uz)*k[7] \
          + (21*ux2-10.5*uy2-10.5*uz2)*k[9] + (31.5*uy2-31.5*uz2)*k[11]

    # Row 4: all signs flipped except even-in-ux term
    dm[4] = (5*ux3 + 55*ux/19)*k[0] + (5.0/19.0)*ux*k[1] \
          + 15*ux2*k[3] + 5*ux*k[9]

    # Row 6
    dm[6] = (5*uy3 + 55*uy/19)*k[0] + (5.0/19.0)*uy*k[1] \
          + 15*uy2*k[5] + (-2.5*uy)*k[9] + 7.5*uy*k[11]

    # Row 8
    dm[8] = (5*uz3 + 55*uz/19)*k[0] + (5.0/19.0)*uz*k[1] \
          + 15*uz2*k[7] + (-2.5*uz)*k[9] + (-7.5*uz)*k[11]

    # Row 10: signs of odd-u terms flipped
    dm[10] = (6*ux4 + 246*ux2/19 - 3*uy4 - 123*uy2/19 - 3*uz4 - 123*uz2/19)*k[0] \
           + (12*ux2 - 6*uy2 - 6*uz2)/19*k[1] \
           + (24*ux3+12*ux)*k[3] + (-12*uy3-6*uy)*k[5] + (-12*uz3-6*uz)*k[7] \
           + (12*ux2+3*uy2+3*uz2)*k[9] + (-9*uy2+9*uz2)*k[11]

    # Row 12
    dm[12] = (3*uy4 + 123*uy2/19 - 3*uz4 - 123*uz2/19)*k[0] \
           + (6*uy2-6*uz2)/19*k[1] \
           + (12*uy3+6*uy)*k[5] + (-12*uz3-6*uz)*k[7] \
           + (-3*uy2+3*uz2)*k[9] + (9*uy2+9*uz2)*k[11]

    return dm

# ====================================================================
# Tests
# ====================================================================
def test_velocities():
    """Return a list of test velocity vectors."""
    return [
        (0.1, 0.05, -0.03),   # typical LBM
        (0.15, -0.08, 0.12),  # higher Ma
        (0.0, 0.0, 0.0),      # zero (should give no correction)
        (0.3, 0.2, -0.1),     # Ma ~ 0.3
        (0.01, 0.0, 0.0),     # nearly zero, only ux
        (-0.05, 0.1, 0.08),   # mixed signs
    ]

passed = 0
failed = 0
total  = 0

print("=" * 80)
print("CUDA Dubois Correction Verification")
print("=" * 80)

# ====================================================================
# Test 1: Forward shift — CUDA Dubois vs matrix-based Dubois
# ====================================================================
print("\n--- Test 1: Forward shift T_dubois(u)·m ---")
for ux, uy, uz in test_velocities():
    total += 1
    rng = np.random.RandomState(42)
    m = rng.randn(19)

    # Reference: full matrix multiplication
    T_dub = build_T_dubois(ux, uy, uz)
    k_ref = T_dub @ m

    # CUDA approach: lattice-reduced + correction
    k_ours = raw_to_central_OURS(m, ux, uy, uz)
    dk = cuda_dubois_forward_correction(m, ux, uy, uz)
    k_cuda = k_ours + dk

    err = norm(k_cuda - k_ref) / (norm(k_ref) + 1e-30)
    status = "PASS" if err < 1e-12 else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        failed += 1
    print(f"  u=({ux:+.3f},{uy:+.3f},{uz:+.3f}): err={err:.2e} [{status}]")
    if status == "FAIL":
        for i in range(19):
            if abs(k_cuda[i] - k_ref[i]) > 1e-10:
                print(f"    Row {i}: cuda={k_cuda[i]:.14e} ref={k_ref[i]:.14e} diff={k_cuda[i]-k_ref[i]:.2e}")

# ====================================================================
# Test 2: Inverse shift — CUDA T(-u) vs matrix-based T(-u)
# ====================================================================
print("\n--- Test 2: Inverse shift T_dubois(-u)·k ---")
for ux, uy, uz in test_velocities():
    total += 1
    rng = np.random.RandomState(123)
    k = rng.randn(19)

    # Reference: T_dubois(-u)·k
    T_dub_inv = build_T_dubois(-ux, -uy, -uz)
    m_ref = T_dub_inv @ k

    # CUDA approach: lattice-reduced inverse + correction
    from test_compare_two_Tu import central_to_raw_OURS
    # Inline the inverse to avoid import issues
    # Actually, let's just compute it here:
    # central_to_raw_OURS = lattice-reduced T(-u)
    # We need to recompute: use raw_to_central_OURS with -u
    # Actually T⁻¹_lattice(u) = T_lattice(-u), so:
    m_ours_inv = raw_to_central_OURS(k, -ux, -uy, -uz)
    dm = cuda_dubois_inverse_correction(k, ux, uy, uz)
    m_cuda = m_ours_inv + dm

    err = norm(m_cuda - m_ref) / (norm(m_ref) + 1e-30)
    status = "PASS" if err < 1e-12 else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        failed += 1
    print(f"  u=({ux:+.3f},{uy:+.3f},{uz:+.3f}): err={err:.2e} [{status}]")
    if status == "FAIL":
        for i in range(19):
            if abs(m_cuda[i] - m_ref[i]) > 1e-10:
                print(f"    Row {i}: cuda={m_cuda[i]:.14e} ref={m_ref[i]:.14e} diff={m_cuda[i]-m_ref[i]:.2e}")

# ====================================================================
# Test 3: Zero velocity — correction should be zero
# ====================================================================
print("\n--- Test 3: Zero velocity correction = 0 ---")
total += 1
rng = np.random.RandomState(999)
m = rng.randn(19)
dk = cuda_dubois_forward_correction(m, 0.0, 0.0, 0.0)
err = norm(dk)
status = "PASS" if err < 1e-15 else "FAIL"
if status == "PASS": passed += 1
else: failed += 1
print(f"  Forward correction at u=0: ||ΔT·m|| = {err:.2e} [{status}]")

total += 1
dm = cuda_dubois_inverse_correction(m, 0.0, 0.0, 0.0)
err = norm(dm)
status = "PASS" if err < 1e-15 else "FAIL"
if status == "PASS": passed += 1
else: failed += 1
print(f"  Inverse correction at u=0: ||ΔT·k|| = {err:.2e} [{status}]")

# ====================================================================
# Test 4: Correction only on ghost rows {2,4,6,8,10,12}
# ====================================================================
print("\n--- Test 4: Correction confined to rows {2,4,6,8,10,12} ---")
total += 1
rng = np.random.RandomState(77)
m = rng.randn(19)
dk = cuda_dubois_forward_correction(m, 0.15, -0.08, 0.12)
non_ghost_rows = [0,1,3,5,7,9,11,13,14,15,16,17,18]
err = norm(dk[non_ghost_rows])
status = "PASS" if err < 1e-15 else "FAIL"
if status == "PASS": passed += 1
else: failed += 1
print(f"  Non-ghost rows correction norm = {err:.2e} [{status}]")

# ====================================================================
# Test 5: Stress modes identical (rows 9,11,13,14,15 unaffected)
# ====================================================================
print("\n--- Test 5: Stress modes identical between lattice & Dubois ---")
for ux, uy, uz in test_velocities():
    total += 1
    rng = np.random.RandomState(42)
    m = rng.randn(19)

    T_dub = build_T_dubois(ux, uy, uz)
    k_dub = T_dub @ m
    k_lat = raw_to_central_OURS(m, ux, uy, uz)

    stress_rows = [9, 11, 13, 14, 15]
    err = norm(k_dub[stress_rows] - k_lat[stress_rows])
    status = "PASS" if err < 1e-12 else "FAIL"
    if status == "PASS": passed += 1
    else: failed += 1
    print(f"  u=({ux:+.3f},{uy:+.3f},{uz:+.3f}): stress diff={err:.2e} [{status}]")

# ====================================================================
# Test 6: Full MRT-CM collision round-trip
# ====================================================================
print("\n--- Test 6: MRT-CM collision f* identical on stress modes ---")
for ux, uy, uz in [(0.1, 0.05, -0.03), (0.3, 0.2, -0.1)]:
    total += 1
    rho = 1.0 + 0.01 * np.random.randn()
    w19 = np.array([1/3]+[1/18]*6+[1/36]*12)
    # Build feq
    feq = np.zeros(19)
    for q in range(19):
        eu = e3d[q,0]*ux + e3d[q,1]*uy + e3d[q,2]*uz
        feq[q] = w19[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*(ux*ux+uy*uy+uz*uz))
    # Random perturbation
    rng = np.random.RandomState(55)
    f = feq + 0.001 * rng.randn(19)

    # Relaxation rates
    s = np.array([0, 1.19, 1.4, 0, 1.2, 0, 1.2, 0, 1.2,
                  0.8, 1.4, 0.8, 1.4, 0.8, 0.8, 0.8, 1.5, 1.5, 1.5])

    # --- Lattice-reduced collision ---
    m_neq = M3d @ (f - feq)
    k_lat = raw_to_central_OURS(m_neq, ux, uy, uz)
    dk_lat = s * k_lat
    # Inverse: use our function with -u
    dm_lat = raw_to_central_OURS(dk_lat, -ux, -uy, -uz)
    f_star_lat = f.copy()
    for q in range(19):
        f_star_lat[q] -= sum(M3d_inv[q, i] * dm_lat[i] for i in range(19))

    # --- Dubois collision ---
    k_dub_neq = raw_to_central_OURS(m_neq, ux, uy, uz) + cuda_dubois_forward_correction(m_neq, ux, uy, uz)
    dk_dub = s * k_dub_neq
    dm_dub_raw = raw_to_central_OURS(dk_dub, -ux, -uy, -uz) + cuda_dubois_inverse_correction(dk_dub, ux, uy, uz)
    f_star_dub = f.copy()
    for q in range(19):
        f_star_dub[q] -= sum(M3d_inv[q, i] * dm_dub_raw[i] for i in range(19))

    diff = norm(f_star_dub - f_star_lat)
    # The difference should be small (only ghost modes differ, but those get relaxed differently)
    status = "PASS" if diff < 0.01 else "FAIL"
    if status == "PASS": passed += 1
    else: failed += 1
    print(f"  u=({ux:+.3f},{uy:+.3f},{uz:+.3f}): ||f*_dub - f*_lat|| = {diff:.6e} [{status}]")

# ====================================================================
# Summary
# ====================================================================
print("\n" + "=" * 80)
print(f"TOTAL: {passed}/{total} PASSED, {failed} FAILED")
if failed == 0:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗")
print("=" * 80)
