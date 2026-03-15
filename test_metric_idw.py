#!/usr/bin/env python3
"""
10 項嚴謹單元測試: Direct physical-coordinate IDW 3D interpolation
==================================================================
驗證 idw_3d_interpolate() 使用直接 z_d 查表計算物理空間距離的正確性。

方法:
  ds² = dx²·Δη² + dy²·Δξ² + (z_node - z_dep)²
  z_node: 從 z_d[j,k] 直接讀取 (精確)
  z_dep:  雙線性插值 z_d at departure point (CFL<1, 近似誤差極小)
  dx, dy: 均勻方向, 精確

測試:
  1. 均勻 z 網格: z_dep 雙線性精確還原
  2. 節點精確還原 (departure = node → exact)
  3. 權重歸一化 (常數場精確還原)
  4. 單調性保證 (結果 ∈ [min, max])
  5. 各向異性: 壁面 ζ-鄰點主導 (tanh stretching)
  6. 直接 z_d 精度 vs 度量張量線性化精度
  7. z_dep 雙線性插值精度 (vs Jacobian)
  8. p=4 冪次正確行為
  9. 數值穩定性 (極端 stretching)
  10. 山丘斜面: z(j,k) 同時隨 j,k 變化
"""

import numpy as np
import sys

# ================================================================
# 模擬參數 (from variables.h)
# ================================================================
LX, LY, LZ = 4.5, 9.0, 3.036
NX, NY, NZ = 32, 128, 128
NZ6 = NZ + 6
dx_phys = LX / NX
dy_phys = LY / NY


def generate_tanh_z(nz6, beta=2.0, z_bot=1.0, z_top=None):
    """生成 tanh 壁面加密 z 座標 (模擬 initialization.h)"""
    if z_top is None:
        z_top = LZ
    eta = np.linspace(0, 1, nz6)
    z_tanh = 0.5 * (1.0 + np.tanh(beta * (2 * eta - 1)) / np.tanh(beta))
    return z_bot + z_tanh * (z_top - z_bot)


def generate_hill_z(ny_local, nz6, beta=2.0):
    """生成 2D z(j,k) 含山丘地形 (z 隨 j 和 k 都變化)"""
    z_2d = np.zeros((ny_local, nz6))
    for jj in range(ny_local):
        y_frac = jj / max(ny_local - 1, 1)
        z_bot = 1.0 + 0.3 * np.sin(2 * np.pi * y_frac)
        z_2d[jj, :] = generate_tanh_z(nz6, beta, z_bot, LZ)
    return z_2d


def bilinear_interp_z(z_2d, t_xi, t_zeta, bj, bk):
    """雙線性插值 z_dep, 完全對應 CUDA 代碼"""
    sj_lo = int(t_xi)
    sk_lo = int(t_zeta)
    sj_hi = min(sj_lo + 1, 6)
    sk_hi = min(sk_lo + 1, 6)
    fj = t_xi - sj_lo
    fk = t_zeta - sk_lo
    return ((1 - fj) * ((1 - fk) * z_2d[bj + sj_lo, bk + sk_lo]
                        + fk * z_2d[bj + sj_lo, bk + sk_hi])
            + fj * ((1 - fk) * z_2d[bj + sj_hi, bk + sk_lo]
                    + fk * z_2d[bj + sj_hi, bk + sk_hi]))


def idw_3d_direct(f_3d, t_eta, t_xi, t_zeta, p, z_dep, z_2d, bj, bk):
    """
    Python 版 idw_3d_interpolate (direct z_d lookup).
    f_3d: (7,7,7) stencil values.
    z_2d: z_d[j, k] physical z-coordinates.
    """
    eps = 1e-24
    dx2 = dx_phys ** 2
    dy2 = dy_phys ** 2
    sum_w = 0.0
    sum_wf = 0.0

    for si in range(7):
        d_eta = t_eta - si
        dx2_term = dx2 * d_eta * d_eta
        for sj in range(7):
            d_xi = t_xi - sj
            dy2_term = dy2 * d_xi * d_xi
            gj = bj + sj
            for sk in range(7):
                gk = bk + sk
                dz = z_2d[gj, gk] - z_dep
                d2 = dx2_term + dy2_term + dz * dz

                f_val = f_3d[si, sj, sk]
                if d2 < eps:
                    return f_val

                if p == 2.0:
                    w = 1.0 / d2
                elif p == 4.0:
                    w = 1.0 / (d2 * d2)
                else:
                    w = 1.0 / d2 ** (p * 0.5)

                sum_w += w
                sum_wf += w * f_val

    return sum_wf / sum_w


def idw_3d_isotropic(f_3d, t_eta, t_xi, t_zeta, p):
    """舊版 isotropic IDW (計算空間距離)"""
    eps = 1e-24
    sum_w = 0.0
    sum_wf = 0.0
    for si in range(7):
        for sj in range(7):
            for sk in range(7):
                d2 = (t_eta - si)**2 + (t_xi - sj)**2 + (t_zeta - sk)**2
                f_val = f_3d[si, sj, sk]
                if d2 < eps:
                    return f_val
                w = 1.0 / d2 if p == 2.0 else 1.0 / d2**(p*0.5)
                sum_w += w
                sum_wf += w * f_val
    return sum_wf / sum_w


# ================================================================
n_pass = 0
n_fail = 0

def check(name, condition, detail=""):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f"  ✓ Test {n_pass+n_fail:2d}: {name}")
    else:
        n_fail += 1
        print(f"  ✗ Test {n_pass+n_fail:2d}: {name}  ← FAIL{' ('+detail+')' if detail else ''}")


ny_loc, nz6_loc = 20, NZ6
np.random.seed(42)
f_rand = np.random.uniform(0.9, 1.1, (7, 7, 7))

# ================================================================
# Test 1: 均勻 z 網格 — z_dep bilinear 精確
# ================================================================
print("\n=== Test 1: Uniform z-grid: bilinear z_dep exact ===")
dz_const = 0.05
z_unif = np.zeros((ny_loc, nz6_loc))
for jj in range(ny_loc):
    for kk in range(nz6_loc):
        z_unif[jj, kk] = 1.0 + kk * dz_const

bj, bk = 5, 30
t_xi, t_zeta = 2.8, 3.5
z_dep_bl = bilinear_interp_z(z_unif, t_xi, t_zeta, bj, bk)
z_dep_exact = z_unif[bj, bk] + t_zeta * dz_const
err = abs(z_dep_bl - z_dep_exact)
check("Uniform z: bilinear z_dep == z_base + t_zeta*dz", err < 1e-12,
      f"err={err:.2e}")

# ================================================================
# Test 2: 節點精確還原
# ================================================================
print("\n=== Test 2: Node-exact reproduction ===")
z_tanh = generate_hill_z(ny_loc, nz6_loc)
bj2, bk2 = 5, 10
all_exact = True
for si, sj, sk in [(0,0,0), (3,3,3), (6,6,6), (2,5,1)]:
    z_dep_node = z_tanh[bj2+sj, bk2+sk]
    result = idw_3d_direct(f_rand, float(si), float(sj), float(sk),
                           2.0, z_dep_node, z_tanh, bj2, bk2)
    expected = f_rand[si, sj, sk]
    if abs(result - expected) > 1e-14:
        all_exact = False
        print(f"    FAIL at ({si},{sj},{sk}): {result:.15e} vs {expected:.15e}")
check("Departure on node → exact (4 test points)", all_exact)

# ================================================================
# Test 3: 常數場精確還原
# ================================================================
print("\n=== Test 3: Constant field reproduction ===")
f_const = np.full((7, 7, 7), 2.71828)
errs = []
for te, tx, tz in [(3.1, 3.2, 3.3), (0.5, 5.5, 1.2), (5.9, 0.1, 4.8)]:
    z_dep_c = bilinear_interp_z(z_tanh, tx, tz, bj2, bk2)
    result = idw_3d_direct(f_const, te, tx, tz, 2.0, z_dep_c, z_tanh, bj2, bk2)
    errs.append(abs(result - 2.71828))
check("Constant field: 3 points exact", max(errs) < 1e-14,
      f"max_err={max(errs):.2e}")

# ================================================================
# Test 4: 單調性
# ================================================================
print("\n=== Test 4: Monotonicity (100 random points) ===")
np.random.seed(123)
f_wide = np.random.uniform(-5, 10, (7, 7, 7))
f_min, f_max = f_wide.min(), f_wide.max()
all_mono = True
for _ in range(100):
    te = np.random.uniform(0, 6)
    tx = np.random.uniform(0, 6)
    tz = np.random.uniform(0, 6)
    z_dep_m = bilinear_interp_z(z_tanh, tx, tz, bj2, bk2)
    result = idw_3d_direct(f_wide, te, tx, tz, 2.0, z_dep_m, z_tanh, bj2, bk2)
    if result < f_min - 1e-14 or result > f_max + 1e-14:
        all_mono = False
        break
check("Result ∈ [min, max] for 100 random points", all_mono)

# ================================================================
# Test 5: 壁面 ζ-鄰點主導
# ================================================================
print("\n=== Test 5: Wall: ζ-neighbor dominates ===")
z_wall = generate_tanh_z(nz6_loc, beta=2.5, z_bot=1.0)
z_2d_flat = np.tile(z_wall, (ny_loc, 1))
bk_wall = 3

f_test = np.ones((7, 7, 7))
f_test[3, 3, 4] = 2.0  # ζ-neighbor
f_test[4, 3, 3] = 2.0  # η-neighbor

z_dep_w = bilinear_interp_z(z_2d_flat, 3.0, 3.5, bj2, bk_wall)
result_direct = idw_3d_direct(f_test, 3.0, 3.0, 3.5, 2.0, z_dep_w, z_2d_flat, bj2, bk_wall)
result_iso = idw_3d_isotropic(f_test, 3.0, 3.0, 3.5, 2.0)
check("Direct IDW: ζ-neighbor more weight than isotropic",
      abs(result_direct - 2.0) < abs(result_iso - 2.0),
      f"direct→{result_direct:.6f}, iso→{result_iso:.6f}")

dz_wall_local = z_wall[bk_wall + 4] - z_wall[bk_wall + 3]
ratio = dx_phys / dz_wall_local
check("dx/dz at wall >> 1", ratio > 5, f"ratio={ratio:.1f}")

# ================================================================
# Test 6: Direct z_d 精度 vs 度量張量線性化
# ================================================================
print("\n=== Test 6: Direct z_d accuracy vs metric linearization ===")
k_center = 6
bk_test = k_center - 3
z_tanh_1d = generate_tanh_z(nz6_loc, beta=2.0)
dk_dz_center = 1.0 / ((z_tanh_1d[k_center+1] - z_tanh_1d[k_center-1]) / 2.0)

max_err_metric = 0.0
for sk in range(7):
    z_exact = z_tanh_1d[bk_test + sk]
    z_metric = z_tanh_1d[k_center] + (sk - 3) / dk_dz_center
    max_err_metric = max(max_err_metric, abs(z_metric - z_exact))

check("Direct z_d: zero node error; metric has non-zero error",
      max_err_metric > 1e-6,
      f"metric_max_err={max_err_metric:.3e}")

# ================================================================
# Test 7: z_dep 雙線性 vs Jacobian 精度
# ================================================================
print("\n=== Test 7: z_dep bilinear vs Jacobian ===")
z_tanh_2d = np.tile(z_tanh_1d, (ny_loc, 1))
k_dep = bk_test + 3.5
z_dep_true = 0.5 * (z_tanh_1d[int(k_dep)] + z_tanh_1d[int(k_dep)+1])
z_dep_bilin = bilinear_interp_z(z_tanh_2d, 3.0, 3.5, 5, bk_test)
z_dep_jacob = z_tanh_1d[k_center] + 0.5 / dk_dz_center

err_bilin = abs(z_dep_bilin - z_dep_true)
err_jacob = abs(z_dep_jacob - z_dep_true)
check("z_dep: bilinear ≤ Jacobian error",
      err_bilin <= err_jacob + 1e-15,
      f"bilinear={err_bilin:.3e}, jacobian={err_jacob:.3e}")

# ================================================================
# Test 8: p=4 冪次
# ================================================================
print("\n=== Test 8: Power p=4 ===")
np.random.seed(77)
f_p4 = np.random.uniform(0.5, 1.5, (7, 7, 7))
z_dep_p4 = bilinear_interp_z(z_tanh_2d, 3.1, 3.4, 5, bk_test)

result_p2 = idw_3d_direct(f_p4, 3.2, 3.1, 3.4, 2.0, z_dep_p4, z_tanh_2d, 5, bk_test)
result_p4 = idw_3d_direct(f_p4, 3.2, 3.1, 3.4, 4.0, z_dep_p4, z_tanh_2d, 5, bk_test)
nearest = f_p4[3, 3, 3]
check("p=4 closer to nearest neighbor than p=2",
      abs(result_p4 - nearest) < abs(result_p2 - nearest),
      f"p2={result_p2:.8f}, p4={result_p4:.8f}, near={nearest:.8f}")

all_mono4 = True
for _ in range(50):
    te = np.random.uniform(0, 6)
    tx = np.random.uniform(0, 6)
    tz = np.random.uniform(0, 6)
    zd = bilinear_interp_z(z_tanh_2d, tx, tz, 5, bk_test)
    r = idw_3d_direct(f_p4, te, tx, tz, 4.0, zd, z_tanh_2d, 5, bk_test)
    if r < f_p4.min() - 1e-14 or r > f_p4.max() + 1e-14:
        all_mono4 = False
        break
check("p=4 monotonicity: 50 random points", all_mono4)

# ================================================================
# Test 9: 極端 stretching 穩定性
# ================================================================
print("\n=== Test 9: Extreme stretching stability ===")
np.random.seed(999)
f_stab = np.random.uniform(0.99, 1.01, (7, 7, 7))
all_stable = True
for beta, label in [(1.0, "mild"), (2.5, "moderate"), (4.0, "extreme"), (5.0, "ultra")]:
    z_ext = generate_tanh_z(nz6_loc, beta)
    z_ext_2d = np.tile(z_ext, (ny_loc, 1))
    z_dep_ext = bilinear_interp_z(z_ext_2d, 3.5, 3.5, 5, 3)
    result = idw_3d_direct(f_stab, 3.5, 3.5, 3.5, 2.0, z_dep_ext, z_ext_2d, 5, 3)
    if not np.isfinite(result) or result < f_stab.min()-1e-10 or result > f_stab.max()+1e-10:
        all_stable = False
        print(f"    FAIL: {label} (beta={beta}): result={result}")
check("Stable for beta ∈ {1, 2.5, 4, 5}", all_stable)

# ================================================================
# Test 10: 山丘斜面 — z(j,k) 隨 j,k 都變化
# ================================================================
print("\n=== Test 10: Hill slope: z(j,k) varies with j and k ===")
z_hill = generate_hill_z(ny_loc, nz6_loc, beta=2.0)
bj_h, bk_h = 3, 10

z_vary_j = z_hill[bj_h, bk_h+3] != z_hill[bj_h+3, bk_h+3]
check("z(j,k) varies with j (hill geometry)", z_vary_j)

z_flat_2d = np.tile(z_hill[bj_h+3, :], (ny_loc, 1))
np.random.seed(55)
f_hill = np.random.uniform(0.8, 1.2, (7, 7, 7))
z_dep_hill = bilinear_interp_z(z_hill, 3.3, 3.7, bj_h, bk_h)
z_dep_flat = bilinear_interp_z(z_flat_2d, 3.3, 3.7, bj_h, bk_h)
result_hill = idw_3d_direct(f_hill, 3.0, 3.3, 3.7, 2.0, z_dep_hill, z_hill, bj_h, bk_h)
result_flat = idw_3d_direct(f_hill, 3.0, 3.3, 3.7, 2.0, z_dep_flat, z_flat_2d, bj_h, bk_h)
diff = abs(result_hill - result_flat)
check("Hill slope ≠ flat result", diff > 1e-6, f"diff={diff:.3e}")

# ================================================================
print("\n" + "=" * 60)
total = n_pass + n_fail
print(f"Results: {n_pass}/{total} PASSED, {n_fail}/{total} FAILED")
if n_fail == 0:
    print("All tests PASSED ✓")
else:
    print(f"WARNING: {n_fail} test(s) FAILED ✗")
print("=" * 60)
sys.exit(0 if n_fail == 0 else 1)
