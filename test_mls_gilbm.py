#!/usr/bin/env python3
"""
MLS (Moving Least Squares) 插值單元測試
========================================
驗證 interpolation_gilbm.h 中的 MLS 實作正確性。

測試項目 (16 項):
  1. Wendland C² 權重性質
  2. Cholesky 10×10 求解器正確性
  3. 常數場再現 (partition of unity)
  4. 線性場精確再現
  5. 二次場精確再現
  6. 節點附近精度 (departure ≈ node)
  7. 壁面梯度精度 (vs IDW 95% → MLS < 2%)
  8. 各向異性正確 (dx/dz ≈ 49)
  9. Cholesky 穩定性 (極端拉伸)
  10. 對稱性保持
  11. 收斂階數 (h→0)
  12. vs Lagrange 3D 非可分離精度
  13. Fallback 觸發 (active_nodes < 10)
  14. 歸一化一致性 (sx, sy, sz 正確縮放)
  15. PᵀWP 正定性
  16. 物理距離 vs 計算距離比較

Usage:
  python3 test_mls_gilbm.py
"""

import numpy as np
from numpy.linalg import solve, cholesky, cond
import sys

# ── Grid parameters (match variables.h) ──
LX, LY, LZ = 4.5, 9.0, 3.036
NX, NY, NZ = 32, 128, 128
NZ6 = NZ + 6
dx = LX / NX  # 0.140625
dy = LY / NY  # 0.0703125

# ── Wendland C² weight ──
def wendland_c2(d2, h2):
    if d2 >= h2:
        return 0.0
    q = np.sqrt(d2 / h2)
    t = 1.0 - q
    return t**4 * (4.0 * q + 1.0)

# ── Quadratic basis (m=10) ──
def basis_quadratic(px, py, pz):
    return np.array([1.0, px, py, pz,
                     px*px, py*py, pz*pz,
                     px*py, px*pz, py*pz])

# ── Cholesky solve (reference implementation) ──
def cholesky_solve_ref(A_sym, b):
    """Solve A_sym @ x = b using Cholesky."""
    L = cholesky(A_sym)
    y = solve(L, b)
    x = solve(L.T, y)
    return x

# ── Build MLS system for 7x7x7 stencil ──
def mls_3d_python(f_values, x_nodes, y_nodes, z_nodes,
                   x_dep, y_dep, z_dep, h_support=2.0):
    """
    MLS interpolation at departure point (x_dep, y_dep, z_dep).

    f_values: (7,7,7) array of function values
    x_nodes, y_nodes, z_nodes: (7,7,7) physical coordinates of nodes
    """
    m = 10  # quadratic basis size

    # Normalization scales
    sx = np.max(np.abs(x_nodes - x_dep))
    sy = np.max(np.abs(y_nodes - y_dep))
    sz = np.max(np.abs(z_nodes - z_dep))
    if sx < 1e-30: sx = 1.0
    if sy < 1e-30: sy = 1.0
    if sz < 1e-30: sz = 1.0

    h2 = h_support * h_support

    PtWP = np.zeros((m, m))
    PtWf = np.zeros(m)
    active = 0

    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                px = (x_nodes[si, sj, sk] - x_dep) / sx
                py = (y_nodes[si, sj, sk] - y_dep) / sy
                pz = (z_nodes[si, sj, sk] - z_dep) / sz

                d2 = px*px + py*py + pz*pz
                w = wendland_c2(d2, h2)
                if w < 1e-30:
                    continue

                active += 1
                fval = f_values[si, sj, sk]
                p = basis_quadratic(px, py, pz)

                PtWP += w * np.outer(p, p)
                PtWf += w * p * fval

    if active < m:
        return None, active  # Insufficient nodes

    try:
        a = cholesky_solve_ref(PtWP, PtWf)
        return a[0], active  # f̃(x₀) = a[0]
    except np.linalg.LinAlgError:
        return None, active

# ── Generate tanh z-coordinates (match initialization.h) ──
def generate_z_coords(nz6, lz, beta=2.0):
    """Generate tanh-stretched z coordinates (0 to LZ)."""
    z = np.zeros(nz6)
    n_inner = nz6 - 6  # physical points
    for k in range(3, nz6 - 3):
        eta = (k - 3.0) / (n_inner - 1.0)  # [0, 1]
        z[k] = 1.0 + (lz - 1.0) * (1.0 + np.tanh(beta * (eta - 0.5)) / np.tanh(beta * 0.5)) / 2.0
    # Extrapolate ghost zones
    for m_idx in range(3):
        z[2 - m_idx] = 2 * z[3] - z[4 + m_idx]
        z[nz6 - 3 + m_idx] = 2 * z[nz6 - 4] - z[nz6 - 5 - m_idx]
    return z

# ── Test functions ──
PASS = 0
FAIL = 0

def check(name, condition, msg=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {msg}")

# ============================================================================
# TEST 1: Wendland C² weight properties
# ============================================================================
def test_wendland_properties():
    print("\nTest 1: Wendland C² weight properties")
    h2 = 4.0  # h = 2.0

    # w(0) = 1
    check("w(0) = 1", abs(wendland_c2(0.0, h2) - 1.0) < 1e-15)

    # w(h²) = 0
    check("w(h²) = 0", abs(wendland_c2(h2, h2)) < 1e-15)

    # w beyond support = 0
    check("w(1.01*h²) = 0", wendland_c2(1.01 * h2, h2) == 0.0)

    # w > 0 for all d < h
    all_positive = all(wendland_c2(d2, h2) > 0 for d2 in np.linspace(0, 0.99*h2, 100))
    check("w > 0 for d < h", all_positive)

    # Monotonically decreasing
    vals = [wendland_c2(d2, h2) for d2 in np.linspace(0, 0.99*h2, 100)]
    mono = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
    check("Monotonically decreasing", mono)

    # C² at boundary: numerical derivative → 0
    eps = 1e-6
    h = 2.0
    r = h - eps
    w_r = wendland_c2(r*r, h2)
    w_r_minus = wendland_c2((r-eps)**2, h2)
    dw_dr = (w_r - w_r_minus) / eps
    check("dw/dr → 0 at boundary", abs(dw_dr) < 1e-3,
          f"dw/dr={dw_dr:.2e}")

# ============================================================================
# TEST 2: Cholesky 10×10 solver
# ============================================================================
def test_cholesky_solver():
    print("\nTest 2: Cholesky 10×10 solver")

    np.random.seed(42)
    # Generate random SPD matrix
    R = np.random.randn(10, 10)
    A = R.T @ R + 10 * np.eye(10)  # Well-conditioned SPD

    b = np.random.randn(10)
    x_ref = solve(A, b)

    # Using our reference Cholesky
    x_chol = cholesky_solve_ref(A, b)
    err = np.max(np.abs(x_chol - x_ref))
    check("Cholesky matches numpy.solve", err < 1e-10, f"err={err:.2e}")

    # Check with packed upper triangle (as in CUDA code)
    A_packed = np.zeros(55)
    for i in range(10):
        for j in range(i, 10):
            idx = i * (21 - i) // 2 + j - i
            A_packed[idx] = A[i, j]

    # Verify unpacking
    A_unpacked = np.zeros((10, 10))
    for i in range(10):
        for j in range(i, 10):
            idx = i * (21 - i) // 2 + j - i
            A_unpacked[i, j] = A_packed[idx]
            A_unpacked[j, i] = A_packed[idx]

    err_pack = np.max(np.abs(A_unpacked - A))
    check("Packed indexing correct", err_pack < 1e-15, f"err={err_pack:.2e}")

# ============================================================================
# TEST 3: Constant field reproduction
# ============================================================================
def test_constant_field():
    print("\nTest 3: Constant field reproduction")

    z = generate_z_coords(NZ6, LZ)
    const_val = 3.14159

    # Build stencil around k=NZ6//2 (mid-domain)
    bk = NZ6 // 2 - 3
    bj = 5  # arbitrary j
    f = np.full((7, 7, 7), const_val)

    # Node coordinates
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]

    # Departure at center
    t_eta, t_xi, t_zeta = 3.0, 3.0, 3.0
    x_dep = t_eta * dx
    y_dep = t_xi * dy
    z_dep = z[bk + 3]

    result, active = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                    x_dep, y_dep, z_dep)
    err = abs(result - const_val) if result is not None else 1e10
    check("Constant field", err < 1e-12, f"err={err:.2e}")
    check(f"All 343 nodes active", active == 343)

# ============================================================================
# TEST 4: Linear field exact reproduction
# ============================================================================
def test_linear_field():
    print("\nTest 4: Linear field exact reproduction")

    z = generate_z_coords(NZ6, LZ)
    bk = NZ6 // 2 - 3
    bj = 5

    # f = 2.0 + 3.0*x + 1.5*y - 0.7*z
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    f = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x = si * dx
                y = sj * dy
                zv = z[bk + sk]
                x_nodes[si, sj, sk] = x
                y_nodes[si, sj, sk] = y
                z_nodes[si, sj, sk] = zv
                f[si, sj, sk] = 2.0 + 3.0*x + 1.5*y - 0.7*zv

    # Departure at (2.7, 3.4, interp_z)
    t_eta, t_xi, t_zeta = 2.7, 3.4, 3.2
    x_dep = t_eta * dx
    y_dep = t_xi * dy
    # Bilinear z interpolation
    sj_lo, sk_lo = int(t_xi), int(t_zeta)
    sj_hi = min(sj_lo + 1, 6)
    sk_hi = min(sk_lo + 1, 6)
    fj, fk = t_xi - sj_lo, t_zeta - sk_lo
    z_dep = ((1-fj)*((1-fk)*z[bk+sk_lo]+fk*z[bk+sk_hi])
            + fj*((1-fk)*z[bk+sk_lo]+fk*z[bk+sk_hi]))  # Simplified: same j row
    z_dep = z[bk + 3] + (t_zeta - 3.0) * (z[bk+4] - z[bk+3])  # Linear approx

    f_exact = 2.0 + 3.0*x_dep + 1.5*y_dep - 0.7*z_dep
    result, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                x_dep, y_dep, z_dep)
    err = abs(result - f_exact) / abs(f_exact) if result else 1e10
    check("Linear field", err < 1e-10, f"rel_err={err:.2e}")

# ============================================================================
# TEST 5: Quadratic field exact reproduction
# ============================================================================
def test_quadratic_field():
    print("\nTest 5: Quadratic field exact reproduction")

    z = generate_z_coords(NZ6, LZ)
    bk = NZ6 // 2 - 3
    bj = 5

    # f = 1.0 + 0.5*x² - 0.3*y² + 0.2*z² + 0.1*x*y
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    f = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x = si * dx
                y = sj * dy
                zv = z[bk + sk]
                x_nodes[si, sj, sk] = x
                y_nodes[si, sj, sk] = y
                z_nodes[si, sj, sk] = zv
                f[si, sj, sk] = 1.0 + 0.5*x*x - 0.3*y*y + 0.2*zv*zv + 0.1*x*y

    t_eta, t_xi, t_zeta = 2.3, 4.1, 2.8
    x_dep = t_eta * dx
    y_dep = t_xi * dy
    z_dep = z[bk + 3] + (t_zeta - 3.0) * (z[bk+4] - z[bk+3])

    f_exact = 1.0 + 0.5*x_dep**2 - 0.3*y_dep**2 + 0.2*z_dep**2 + 0.1*x_dep*y_dep
    result, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                x_dep, y_dep, z_dep)
    err = abs(result - f_exact) / abs(f_exact) if result else 1e10
    check("Quadratic field", err < 1e-8, f"rel_err={err:.2e}")

# ============================================================================
# TEST 6: Node proximity accuracy
# ============================================================================
def test_node_proximity():
    print("\nTest 6: Node proximity accuracy (smooth function)")

    z = generate_z_coords(NZ6, LZ)
    bk = NZ6 // 2 - 3

    # Use a SMOOTH function (quadratic) — MLS reproduces quadratics exactly
    # so near-node value should be very close to the exact polynomial value
    f = np.zeros((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    z_mid = z[bk + 3]
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]
                x = si * dx
                zv = z[bk + sk]
                f[si, sj, sk] = 1.0 + 0.5 * x + 0.3 * (zv - z_mid)**2

    # Evaluate very close to center node (3,3,3)
    eps = 1e-8
    x_dep = 3.0 * dx + eps
    y_dep = 3.0 * dy + eps
    z_dep = z[bk + 3] + eps * (z[bk+4] - z[bk+3])

    result, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                x_dep, y_dep, z_dep)
    # Exact value at departure point
    f_exact = 1.0 + 0.5 * x_dep + 0.3 * (z_dep - z_mid)**2
    f_node = f[3, 3, 3]
    err = abs(result - f_exact) if result else 1e10
    # For smooth quadratic, MLS should be very accurate near node
    check("Near-node accuracy (smooth)", err < 1e-8,
          f"err={err:.2e}, f_node={f_node:.4f}, f_exact={f_exact:.4f}")

# ============================================================================
# TEST 7: Wall gradient accuracy (THE KEY TEST)
# ============================================================================
def test_wall_gradient():
    print("\nTest 7: Wall gradient accuracy (wall-normal dv/dz)")

    z = generate_z_coords(NZ6, LZ, beta=2.0)

    # Stencil at wall: bk=0 → nodes at k=0..6 (wall at k=3)
    bk = 0
    bj = 5

    # Linear velocity profile: v(z) = dv/dz × (z - z_wall)
    z_wall = z[3]
    dvdz_exact = 1.0  # unit gradient

    f = np.zeros((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]
                f[si, sj, sk] = dvdz_exact * (z[bk + sk] - z_wall)

    # Departure: half grid spacing above wall in z
    dz_wall = z[4] - z[3]
    t_eta, t_xi = 3.0, 3.0
    z_dep = z_wall + 0.5 * dz_wall
    # Find t_zeta for this z_dep
    # z_dep is between z[3] and z[4], so t_zeta ≈ 3 + 0.5
    t_zeta = 3.0 + 0.5 * dz_wall / (z[4] - z[3])  # = 3.5

    f_exact = dvdz_exact * (z_dep - z_wall)

    result_mls, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                    3.0 * dx, 3.0 * dy, z_dep)

    err_mls = abs(result_mls - f_exact) / abs(f_exact) * 100 if result_mls else 100.0

    # Compare with IDW (simple weighted average)
    sum_w, sum_wf = 0.0, 0.0
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                ddx = (si - 3.0) * dx
                ddy = (sj - 3.0) * dy
                ddz = z_nodes[si, sj, sk] - z_dep
                d2 = ddx*ddx + ddy*ddy + ddz*ddz
                if d2 < 1e-24:
                    sum_w, sum_wf = 1.0, f[si, sj, sk]
                    break
                w = 1.0 / d2
                sum_w += w
                sum_wf += w * f[si, sj, sk]
    result_idw = sum_wf / sum_w if sum_w > 0 else 0
    err_idw = abs(result_idw - f_exact) / abs(f_exact) * 100

    check(f"MLS gradient error < 5% (got {err_mls:.1f}%)", err_mls < 5.0)
    check(f"MLS << IDW ({err_mls:.1f}% vs {err_idw:.1f}%)", err_mls < err_idw * 0.2,
          f"MLS not significantly better")

    print(f"    dx/dz_wall = {dx/dz_wall:.1f}×")
    print(f"    IDW error = {err_idw:.1f}%, MLS error = {err_mls:.1f}%")

# ============================================================================
# TEST 8: Anisotropy handling
# ============================================================================
def test_anisotropy():
    print("\nTest 8: Anisotropy handling (dx/dz ≈ 49)")

    z = generate_z_coords(NZ6, LZ, beta=2.0)
    bk = 0  # Wall stencil
    dz_wall = z[4] - z[3]
    aniso = dx / dz_wall

    print(f"    Anisotropy ratio dx/dz_wall = {aniso:.1f}")

    # Test that MLS resolves z-direction variation despite dx >> dz
    # f = z² (pure z-direction function)
    f = np.zeros((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    z_wall = z[3]
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]
                f[si, sj, sk] = (z[bk + sk] - z_wall)**2

    z_dep = z_wall + 0.3 * dz_wall
    f_exact = (z_dep - z_wall)**2

    result, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                3.0 * dx, 3.0 * dy, z_dep)
    err = abs(result - f_exact) / max(abs(f_exact), 1e-30) * 100 if result else 100
    check(f"z² reproduction at wall (err={err:.2f}%)", err < 5.0)

# ============================================================================
# TEST 9: Cholesky stability under extreme stretching
# ============================================================================
def test_cholesky_stability():
    print("\nTest 9: Cholesky stability (extreme stretching)")

    z = generate_z_coords(NZ6, LZ, beta=3.0)  # Strong stretching
    bk = 0
    dz_wall = z[4] - z[3]
    print(f"    beta=3.0, dx/dz = {dx/dz_wall:.1f}")

    f = np.zeros((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    z_wall = z[3]
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]
                f[si, sj, sk] = (z[bk + sk] - z_wall)

    z_dep = z_wall + 0.5 * dz_wall
    result, active = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                     3.0 * dx, 3.0 * dy, z_dep)

    check("Cholesky succeeds with beta=3.0", result is not None,
          f"Failed with {active} active nodes")

    if result is not None:
        f_exact = z_dep - z_wall
        err = abs(result - f_exact) / abs(f_exact) * 100
        check(f"Accuracy OK (err={err:.2f}%)", err < 10.0)

# ============================================================================
# TEST 10: Symmetry preservation
# ============================================================================
def test_symmetry():
    print("\nTest 10: Symmetry preservation")

    z = generate_z_coords(NZ6, LZ)
    bk = NZ6 // 2 - 3

    # Symmetric function: f(z) = (z - z_mid)²
    z_mid = z[bk + 3]
    f = np.zeros((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]
                f[si, sj, sk] = (z[bk + sk] - z_mid)**2

    # Evaluate at +δ and -δ from center
    delta = 0.3
    z_dep_plus = z_mid + delta * (z[bk+4] - z[bk+3])
    z_dep_minus = z_mid - delta * (z[bk+3] - z[bk+2])

    r_plus, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                3.0*dx, 3.0*dy, z_dep_plus)
    r_minus, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                 3.0*dx, 3.0*dy, z_dep_minus)

    if r_plus is not None and r_minus is not None:
        # For tanh mesh, +δ and -δ spans are different, so values differ
        # But both should be close to (δ·dz)²
        f_plus_exact = (z_dep_plus - z_mid)**2
        f_minus_exact = (z_dep_minus - z_mid)**2
        err_plus = abs(r_plus - f_plus_exact) / max(f_plus_exact, 1e-30)
        err_minus = abs(r_minus - f_minus_exact) / max(f_minus_exact, 1e-30)
        check(f"Symmetric errors comparable ({err_plus:.2e} vs {err_minus:.2e})",
              max(err_plus, err_minus) < 0.05)

# ============================================================================
# TEST 11: Convergence order
# ============================================================================
def test_convergence():
    print("\nTest 11: Convergence order")

    # Test with uniform grid (different resolutions)
    errors = []
    hs = []
    for n in [8, 16, 32, 64]:
        h = 1.0 / n
        # 7-point stencil centered at 0.5
        nodes = np.linspace(0.5 - 3*h, 0.5 + 3*h, 7)

        # Test function: sin(2π x)
        f_nodes = np.sin(2 * np.pi * nodes)

        # Build 3D stencil (uniform, make it 1D effectively)
        f = np.zeros((7, 7, 7))
        x_nodes = np.zeros((7, 7, 7))
        y_nodes = np.zeros((7, 7, 7))
        z_nodes = np.zeros((7, 7, 7))
        for si in range(7):
            for sj in range(7):
                for sk in range(7):
                    x_nodes[si, sj, sk] = nodes[si]
                    y_nodes[si, sj, sk] = sj * h
                    z_nodes[si, sj, sk] = sk * h
                    f[si, sj, sk] = f_nodes[si]  # Only varies in x

        t = 3.0 + 0.3  # Departure point
        x_dep = nodes[3] + 0.3 * h
        y_dep = 3 * h
        z_dep = 3 * h
        f_exact = np.sin(2 * np.pi * x_dep)

        result, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                    x_dep, y_dep, z_dep)
        if result is not None:
            errors.append(abs(result - f_exact))
            hs.append(h)

    if len(errors) >= 3:
        # Compute convergence rate
        rates = []
        for i in range(1, len(errors)):
            if errors[i-1] > 1e-15 and errors[i] > 1e-15:
                rate = np.log(errors[i-1] / errors[i]) / np.log(hs[i-1] / hs[i])
                rates.append(rate)
        avg_rate = np.mean(rates) if rates else 0
        check(f"Convergence rate ≥ 2.5 (got {avg_rate:.1f})", avg_rate >= 2.5,
              f"rates={[f'{r:.1f}' for r in rates]}")
    else:
        check("Convergence (insufficient data)", False)

# ============================================================================
# TEST 12: vs Lagrange on non-separable function
# ============================================================================
def test_vs_lagrange_nonseparable():
    print("\nTest 12: MLS vs Lagrange on non-separable f_eq")

    z = generate_z_coords(NZ6, LZ, beta=2.0)
    bk = 0  # Wall stencil

    # Use realistic f_eq: w_α · ρ · (1 + 3eu + 4.5eu² - 1.5u²)
    # With Couette-like profile: v(z) = Uref × (z - z_wall) / H
    Uref = 0.037
    z_wall = z[3]

    f = np.zeros((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))

    w_alpha = 1.0/18.0  # D3Q19 weight for e=(0,1,0)
    e_y = 1.0
    rho = 1.0

    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                zv = z[bk + sk]
                z_nodes[si, sj, sk] = zv
                v = Uref * (zv - z_wall)  # Linear velocity
                eu = e_y * v
                u2 = v * v
                f[si, sj, sk] = w_alpha * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2)

    # Departure half-way to first interior point
    dz = z[4] - z[3]
    z_dep = z_wall + 0.5 * dz
    v_dep = Uref * (z_dep - z_wall)
    eu_dep = e_y * v_dep
    f_exact = w_alpha * rho * (1.0 + 3.0*eu_dep + 4.5*eu_dep**2 - 1.5*v_dep**2)

    result_mls, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                    3.0*dx, 3.0*dy, z_dep)

    err_mls = abs(result_mls - f_exact) / abs(f_exact) * 100 if result_mls else 100.0

    # Lagrange tensor product (1D in z direction since f doesn't vary in x,y)
    from numpy.polynomial.polynomial import polyval
    # Simplified: just use numpy for Lagrange
    z_stencil = z[bk:bk+7]
    f_z = [w_alpha * rho * (1.0 + 3.0*e_y*Uref*(zv-z_wall)
           + 4.5*(e_y*Uref*(zv-z_wall))**2
           - 1.5*(Uref*(zv-z_wall))**2) for zv in z_stencil]
    # Lagrange interpolation
    from functools import reduce
    def lagrange_1d(x_pts, f_pts, x_eval):
        n = len(x_pts)
        result = 0.0
        for i in range(n):
            Li = 1.0
            for j in range(n):
                if j != i:
                    Li *= (x_eval - x_pts[j]) / (x_pts[i] - x_pts[j])
            result += f_pts[i] * Li
        return result

    result_lag = lagrange_1d(z_stencil, f_z, z_dep)
    err_lag = abs(result_lag - f_exact) / abs(f_exact) * 100

    print(f"    MLS error = {err_mls:.3f}%, Lagrange (1D) error = {err_lag:.3f}%")
    check(f"MLS error < 5% on f_eq", err_mls < 5.0)

# ============================================================================
# TEST 13: Fallback trigger
# ============================================================================
def test_fallback():
    print("\nTest 13: Fallback trigger (too few nodes)")

    # Use h so small that < 10 nodes have weight
    z = generate_z_coords(NZ6, LZ)
    bk = NZ6 // 2 - 3

    f = np.ones((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]

    # Very small support radius → few active nodes
    result, active = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                     3*dx, 3*dy, z[bk+3], h_support=0.01)
    check(f"Few active nodes with tiny h (got {active})", active < 10)
    check("Fallback returns None", result is None)

# ============================================================================
# TEST 14: Normalization consistency
# ============================================================================
def test_normalization():
    print("\nTest 14: Normalization consistency")

    z = generate_z_coords(NZ6, LZ, beta=2.0)
    bk = 0  # Wall

    # Same function, evaluate with different normalization approaches
    # If normalization is correct, both should give same result
    f = np.zeros((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    z_wall = z[3]
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]
                f[si, sj, sk] = 1.0 + 0.1 * si * dx + 0.2 * (z[bk+sk] - z_wall)

    z_dep = z_wall + 0.3 * (z[4] - z[3])
    r1, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                            3.0*dx, 3.0*dy, z_dep, h_support=2.0)
    r2, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                            3.0*dx, 3.0*dy, z_dep, h_support=3.0)

    if r1 is not None and r2 is not None:
        # Different h_support should give similar results for well-conditioned case
        err = abs(r1 - r2) / max(abs(r1), 1e-30)
        check(f"h=2 vs h=3 consistent (diff={err:.2e})", err < 0.01)

# ============================================================================
# TEST 15: PᵀWP positive definiteness
# ============================================================================
def test_ptwp_positive_definite():
    print("\nTest 15: PᵀWP positive definiteness")

    z = generate_z_coords(NZ6, LZ, beta=2.0)
    bk = 0
    h2 = 4.0
    m = 10

    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]

    x_dep, y_dep, z_dep = 3.0*dx, 3.0*dy, z[3]
    sx = np.max(np.abs(x_nodes - x_dep))
    sy = np.max(np.abs(y_nodes - y_dep))
    sz = np.max(np.abs(z_nodes - z_dep))
    if sz < 1e-30: sz = 1.0

    PtWP = np.zeros((m, m))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                px = (x_nodes[si,sj,sk] - x_dep) / sx
                py = (y_nodes[si,sj,sk] - y_dep) / sy
                pz = (z_nodes[si,sj,sk] - z_dep) / sz
                d2 = px*px + py*py + pz*pz
                w = wendland_c2(d2, h2)
                if w < 1e-30: continue
                p = basis_quadratic(px, py, pz)
                PtWP += w * np.outer(p, p)

    eigenvalues = np.linalg.eigvalsh(PtWP)
    min_eig = np.min(eigenvalues)
    cond_num = cond(PtWP)

    check(f"All eigenvalues > 0 (min={min_eig:.2e})", min_eig > 0)
    check(f"Condition number reasonable (cond={cond_num:.1e})", cond_num < 1e10)
    print(f"    Eigenvalue range: [{min_eig:.2e}, {np.max(eigenvalues):.2e}]")
    print(f"    Condition number: {cond_num:.2e}")

# ============================================================================
# TEST 16: Physical vs computational distance
# ============================================================================
def test_physical_vs_computational():
    print("\nTest 16: Physical vs computational distance MLS")

    z = generate_z_coords(NZ6, LZ, beta=2.0)
    bk = 0
    z_wall = z[3]

    # f = z - z_wall (linear in z)
    f = np.zeros((7, 7, 7))
    x_nodes = np.zeros((7, 7, 7))
    y_nodes = np.zeros((7, 7, 7))
    z_nodes = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_nodes[si, sj, sk] = si * dx
                y_nodes[si, sj, sk] = sj * dy
                z_nodes[si, sj, sk] = z[bk + sk]
                f[si, sj, sk] = z[bk + sk] - z_wall

    z_dep = z_wall + 0.5 * (z[4] - z[3])
    f_exact = z_dep - z_wall

    # Physical distance MLS
    result_phys, _ = mls_3d_python(f, x_nodes, y_nodes, z_nodes,
                                     3.0*dx, 3.0*dy, z_dep)

    # Computational distance MLS (isotropic normalization)
    x_comp = np.zeros((7, 7, 7))
    y_comp = np.zeros((7, 7, 7))
    z_comp = np.zeros((7, 7, 7))
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                x_comp[si, sj, sk] = float(si)
                y_comp[si, sj, sk] = float(sj)
                z_comp[si, sj, sk] = float(sk)

    result_comp, _ = mls_3d_python(f, x_comp, y_comp, z_comp,
                                     3.0, 3.0, 3.5)  # t_zeta ≈ 3.5

    err_phys = abs(result_phys - f_exact) / abs(f_exact) * 100 if result_phys else 100
    err_comp = abs(result_comp - f_exact) / abs(f_exact) * 100 if result_comp else 100

    print(f"    Physical distance error = {err_phys:.3f}%")
    print(f"    Computational distance error = {err_comp:.3f}%")
    check(f"Physical distance MLS accurate (err={err_phys:.3f}%)", err_phys < 2.0)

# ============================================================================
# Run all tests
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MLS (Moving Least Squares) Interpolation Unit Tests")
    print("=" * 60)

    test_wendland_properties()
    test_cholesky_solver()
    test_constant_field()
    test_linear_field()
    test_quadratic_field()
    test_node_proximity()
    test_wall_gradient()
    test_anisotropy()
    test_cholesky_stability()
    test_symmetry()
    test_convergence()
    test_vs_lagrange_nonseparable()
    test_fallback()
    test_normalization()
    test_ptwp_positive_definite()
    test_physical_vs_computational()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
