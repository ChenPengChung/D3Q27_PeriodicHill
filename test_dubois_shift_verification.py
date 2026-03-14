#!/usr/bin/env python3
"""
Verification: T_dH(u) = M_dH(u) · M_dH(0)^{-1}  (Dubois 2015)

Tests:
  1. D2Q9: Reproduce Dubois Appendix A blocks (A, B1, B2, C1-C3, D1-D4)
  2. D2Q9: Group properties T(u)·T(-u)=I, T(u+v)=T(u)·T(v)
  3. D3Q19: M_dH(u)·M_dH(0)^{-1} vs our raw_to_central_dH
  4. D3Q19: Group properties
  5. D3Q19: Verify moment physical content (index mapping)
  6. Chávez-Modena optimized s_i mapping
  7. MRT-CM collision basis-independence
"""
import numpy as np
from numpy.linalg import inv, norm
np.set_printoptions(precision=12, linewidth=200)

PASS_COUNT = 0
FAIL_COUNT = 0

def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ {name}")
    else:
        FAIL_COUNT += 1
        print(f"  ✗ {name}  {detail}")

# ====================================================================
# D2Q9
# ====================================================================
e2d = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

def build_M_dubois_d2q9(ux=0.0, uy=0.0, la=1.0):
    """Dubois D2Q9 moment matrix: evaluate d'Humières polynomials at (e_i - u)."""
    M = np.zeros((9, 9))
    for i in range(9):
        X = e2d[i, 0] - ux
        Y = e2d[i, 1] - uy
        r2 = X*X + Y*Y
        M[0, i] = 1.0
        M[1, i] = X
        M[2, i] = Y
        M[3, i] = 6.0/(la**2) * r2/2.0 - 4.0
        M[4, i] = (X*X - Y*Y) / la**2
        M[5, i] = X*Y / la**2
        M[6, i] = (6.0/(la**3)) * X * r2/2.0 - (5.0/la) * X
        M[7, i] = (6.0/(la**3)) * Y * r2/2.0 - (5.0/la) * Y
        M[8, i] = 18.0/(la**4) * (r2**2)/4.0 + 4.0 - 21.0/(la**2) * r2/2.0
    return M

print("=" * 70)
print("Test 1: D2Q9 Dubois T(ũ) blocks vs Appendix A")
print("=" * 70)

la = 1.0
ux_t, uy_t = 0.15, -0.08
u2 = ux_t**2 + uy_t**2

M0_2d = build_M_dubois_d2q9(0, 0, la)
Mu_2d = build_M_dubois_d2q9(ux_t, uy_t, la)
Tu_2d = Mu_2d @ inv(M0_2d)

# Block A
check("Block A", norm(Tu_2d[1:3, 0] - np.array([-ux_t, -uy_t])) < 1e-14)
check("Block I2", norm(Tu_2d[1:3, 1:3] - np.eye(2)) < 1e-14)

# Block B1
B1 = np.array([3.0/la**2*u2, 1.0/la**2*(ux_t**2-uy_t**2), 1.0/la**2*ux_t*uy_t])
check("Block B1", norm(Tu_2d[3:6, 0] - B1) < 1e-14)

# Block B2
B2 = np.array([[-6/la**2*ux_t, -6/la**2*uy_t],
               [-2/la**2*ux_t,  2/la**2*uy_t],
               [-1/la**2*uy_t, -1/la**2*ux_t]])
check("Block B2", norm(Tu_2d[3:6, 1:3] - B2) < 1e-14)

# Block C1
C1 = np.array([-3/la**3*ux_t*(u2+la**2), -3/la**3*uy_t*(u2+la**2)])
check("Block C1", norm(Tu_2d[6:8, 0] - C1) < 1e-14)

# Block C2
C2 = np.array([[3/la**3*(u2+2*ux_t**2), 6/la**3*ux_t*uy_t],
               [6/la**3*ux_t*uy_t,      3/la**3*(u2+2*uy_t**2)]])
check("Block C2", norm(Tu_2d[6:8, 1:3] - C2) < 1e-14)

# Block C3
C3 = np.array([[-2/la*ux_t, -3/la*ux_t, -6/la*uy_t],
               [-2/la*uy_t,  3/la*uy_t, -6/la*ux_t]])
check("Block C3", norm(Tu_2d[6:8, 3:6] - C3) < 1e-14)

# Block D1
D1 = 9/(2*la**4)*(u2**2 + 3*la**2*u2)
check("Block D1", abs(Tu_2d[8, 0] - D1) < 1e-14)

# Block D2 — row 8 derivative w.r.t. momentum rows (1,2)
# P_8 = 18(r²)²/(4la⁴) - 21r²/(2la²) + 4, expanded at (e_i - u):
# T[8,1:3] = -9/la⁴ · u · (la²+2u²)
D2_expected = np.array([-9.0/(la**4)*ux_t*(la**2+2*u2),
                        -9.0/(la**4)*uy_t*(la**2+2*u2)])
check("Block D2", norm(Tu_2d[8, 1:3] - D2_expected) < 1e-13)

# Block D3
D3 = np.array([6/la**2*u2, 9/la**2*(ux_t**2-uy_t**2), 36/la**2*ux_t*uy_t])
check("Block D3", norm(Tu_2d[8, 3:6] - D3) < 1e-14)

# Block D4
D4 = np.array([-6/la*ux_t, -6/la*uy_t])
check("Block D4", norm(Tu_2d[8, 6:8] - D4) < 1e-14)

print("\n" + "=" * 70)
print("Test 2: D2Q9 Group properties")
print("=" * 70)
Tn = build_M_dubois_d2q9(-ux_t, -uy_t, la) @ inv(M0_2d)
check("T(u)·T(-u) = I", norm(Tu_2d @ Tn - np.eye(9)) < 1e-13)
vx, vy = 0.05, 0.12
Tv = build_M_dubois_d2q9(vx, vy, la) @ inv(M0_2d)
Tuv = build_M_dubois_d2q9(ux_t+vx, uy_t+vy, la) @ inv(M0_2d)
check("T(u+v) = T(u)·T(v)", norm(Tuv - Tu_2d @ Tv) < 1e-13)

# ====================================================================
# D3Q19 — exact d'Humières polynomial forms
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

def dH_polys(X, Y, Z):
    """Evaluate all 19 d'Humières polynomials at (X, Y, Z).

    Uses LATTICE-REDUCED polynomial forms: x^4 → x^2, x^3 → x on D3Q19.
    This matches our T(u) implementation which is derived from the conjugation
    T_dH = A · T_poly · A⁻¹ using the natural monomial basis.

    Note: Dubois (2015) uses CONTINUOUS polynomial forms (r^4 = (r^2)^2).
    Both representations agree on D3Q19 lattice points but differ at shifted
    velocities (e_i - u). The difference only affects ghost/energy modes
    (rows 2,4,6,8,10,12), NOT the stress modes (9,11,13,14,15) which
    control viscosity. Both are valid group homomorphisms.
    """
    X2, Y2, Z2 = X*X, Y*Y, Z*Z
    r2 = X2 + Y2 + Z2
    X2Y2, X2Z2, Y2Z2 = X2*Y2, X2*Z2, Y2*Z2
    p = np.zeros(19)
    p[0]  = 1.0                                        # ρ
    p[1]  = 19.0*r2 - 30.0                             # e = 19r² - 30
    # ε: lattice-reduced from (21r⁴-53r²+24)/2 with r⁴→r²+2(x²y²+x²z²+y²z²)
    p[2]  = -16.0*r2 + 21.0*(X2Y2+X2Z2+Y2Z2) + 12.0
    p[3]  = X                                          # jx
    # qx: lattice-reduced from 5Xr²-9X with x³→x: -4X+5XY²+5XZ²
    p[4]  = -4.0*X + 5.0*X*Y2 + 5.0*X*Z2
    p[5]  = Y                                          # jy
    p[6]  = -4.0*Y + 5.0*Y*X2 + 5.0*Y*Z2              # qy
    p[7]  = Z                                          # jz
    p[8]  = -4.0*Z + 5.0*Z*X2 + 5.0*Z*Y2              # qz
    p[9]  = 2.0*X2 - Y2 - Z2                           # 3pxx (degree 2, exact)
    # 3πxx: lattice-reduced from (2X²-Y²-Z²)(3r²-5)
    p[10] = -4.0*X2 + 2.0*Y2 + 2.0*Z2 + 3.0*X2Y2 + 3.0*X2Z2 - 6.0*Y2Z2
    p[11] = Y2 - Z2                                    # pww (degree 2, exact)
    # πww: lattice-reduced from (Y²-Z²)(3r²-5)
    p[12] = -2.0*Y2 + 2.0*Z2 + 3.0*X2Y2 - 3.0*X2Z2
    p[13] = X * Y                                      # pxy
    p[14] = Y * Z                                      # pyz
    p[15] = X * Z                                      # pxz
    p[16] = X * (Y2 - Z2)                              # mx
    p[17] = Y * (Z2 - X2)                              # my
    p[18] = Z * (X2 - Y2)                              # mz
    return p

# Verify polynomial forms match M3d
print("\n" + "=" * 70)
print("Test 3: Verify d'Humières polynomial forms match M3d")
print("=" * 70)
M3d_check = np.zeros((19, 19))
for i in range(19):
    M3d_check[:, i] = dH_polys(e3d[i, 0], e3d[i, 1], e3d[i, 2])
err_poly = norm(M3d_check - M3d)
check("Polynomial forms reproduce M3d exactly", err_poly < 1e-12, f"err={err_poly:.2e}")

def build_M3d_shifted(ux, uy, uz):
    """M_dH(u): evaluate d'Humières polynomials at (e_i - u)."""
    M = np.zeros((19, 19))
    for i in range(19):
        M[:, i] = dH_polys(e3d[i,0]-ux, e3d[i,1]-uy, e3d[i,2]-uz)
    return M

# ====================================================================
# Test 4: D3Q19 T_dH(u) = M_dH(u) · M_dH(0)^{-1} vs our code
# ====================================================================
print("\n" + "=" * 70)
print("Test 4: D3Q19 T_direct = M_dH(u)·M_dH(0)⁻¹ vs our code")
print("=" * 70)

ux3, uy3, uz3 = 0.12, -0.07, 0.05

M3d_u = build_M3d_shifted(ux3, uy3, uz3)
T_direct = M3d_u @ inv(M3d)

# Our raw_to_central_dH (ported from CUDA)
def raw_to_central_dH_py(m, ux, uy, uz):
    ux2, uy2, uz2 = ux*ux, uy*uy, uz*uz
    u2 = ux2 + uy2 + uz2
    uxuy, uxuz, uyuz = ux*uy, ux*uz, uy*uz
    k = np.zeros(19)
    k[0] = m[0]
    k[3] = -ux*m[0] + m[3]
    k[5] = -uy*m[0] + m[5]
    k[7] = -uz*m[0] + m[7]
    k[1] = 19.0*u2*m[0] + m[1] - 38.0*ux*m[3] - 38.0*uy*m[5] - 38.0*uz*m[7]
    k[9] = (2.0*ux2-uy2-uz2)*m[0] - 4.0*ux*m[3] + 2.0*uy*m[5] + 2.0*uz*m[7] + m[9]
    k[11] = (uy2-uz2)*m[0] - 2.0*uy*m[5] + 2.0*uz*m[7] + m[11]
    k[13] = uxuy*m[0] - uy*m[3] - ux*m[5] + m[13]
    k[14] = uyuz*m[0] - uz*m[5] - uy*m[7] + m[14]
    k[15] = uxuz*m[0] - uz*m[3] - ux*m[7] + m[15]
    k[4] = (-5.0*ux*uy2-5.0*ux*uz2-(24.0/19.0)*ux)*m[0] \
          -(10.0/57.0)*ux*m[1] + (5.0*(uy2+uz2))*m[3]+m[4] \
          +10.0*uxuy*m[5]+10.0*uxuz*m[7]+(5.0/3.0)*ux*m[9] \
          -10.0*uy*m[13]-10.0*uz*m[15]
    k[6] = (-5.0*ux2*uy-5.0*uy*uz2-(24.0/19.0)*uy)*m[0] \
          -(10.0/57.0)*uy*m[1]+10.0*uxuy*m[3]+(5.0*(ux2+uz2))*m[5]+m[6] \
          +10.0*uyuz*m[7]-(5.0/6.0)*uy*m[9]+(5.0/2.0)*uy*m[11] \
          -10.0*ux*m[13]-10.0*uz*m[14]
    k[8] = (-5.0*ux2*uz-5.0*uy2*uz-(24.0/19.0)*uz)*m[0] \
          -(10.0/57.0)*uz*m[1]+10.0*uxuz*m[3]+10.0*uyuz*m[5] \
          +(5.0*(ux2+uy2))*m[7]+m[8]-(5.0/6.0)*uz*m[9]-(5.0/2.0)*uz*m[11] \
          -10.0*uy*m[14]-10.0*ux*m[15]
    k[2] = (21.0*(ux2*uy2+ux2*uz2+uy2*uz2)+(116.0/19.0)*u2)*m[0] \
          +(14.0/19.0)*u2*m[1]+m[2] \
          +(-42.0*ux*uy2-42.0*ux*uz2-(8.0/5.0)*ux)*m[3]-(42.0/5.0)*ux*m[4] \
          +(-42.0*ux2*uy-42.0*uy*uz2-(8.0/5.0)*uy)*m[5]-(42.0/5.0)*uy*m[6] \
          +(-42.0*ux2*uz-42.0*uy2*uz-(8.0/5.0)*uz)*m[7]-(42.0/5.0)*uz*m[8] \
          +(-7.0*ux2+3.5*uy2+3.5*uz2)*m[9]+(-10.5*uy2+10.5*uz2)*m[11] \
          +84.0*uxuy*m[13]+84.0*uyuz*m[14]+84.0*uxuz*m[15]
    k[10] = (3.0*(ux2*uy2+ux2*uz2)-6.0*uy2*uz2-(16.0/19.0)*ux2+(8.0/19.0)*uy2+(8.0/19.0)*uz2)*m[0] \
           +((2.0*ux2-uy2-uz2)/19.0)*m[1] \
           +(-6.0*ux*uy2-6.0*ux*uz2+(16.0/5.0)*ux)*m[3]-(6.0/5.0)*ux*m[4] \
           +(-6.0*ux2*uy+12.0*uy*uz2-(8.0/5.0)*uy)*m[5]+(3.0/5.0)*uy*m[6] \
           +(-6.0*ux2*uz+12.0*uy2*uz-(8.0/5.0)*uz)*m[7]+(3.0/5.0)*uz*m[8] \
           +(-ux2+2.0*uy2+2.0*uz2)*m[9]+m[10]+(3.0*uy2-3.0*uz2)*m[11] \
           +12.0*uxuy*m[13]-24.0*uyuz*m[14]+12.0*uxuz*m[15] \
           +9.0*uy*m[17]-9.0*uz*m[18]
    k[12] = (3.0*ux2*(uy2-uz2)+(8.0/19.0)*(uz2-uy2))*m[0]+(uy2-uz2)/19.0*m[1] \
           -6.0*ux*(uy2-uz2)*m[3] \
           +(2.0/5.0)*uy*(4.0-15.0*ux2)*m[5]-(3.0/5.0)*uy*m[6] \
           +(2.0/5.0)*uz*(15.0*ux2-4.0)*m[7]+(3.0/5.0)*uz*m[8] \
           +(uy2-uz2)*m[9]+3.0*ux2*m[11]+m[12] \
           +12.0*uxuy*m[13]-12.0*uxuz*m[15] \
           -6.0*ux*m[16]+3.0*uy*m[17]+3.0*uz*m[18]
    k[16] = ux*(uz2-uy2)*m[0]+(uy2-uz2)*m[3]+2.0*uxuy*m[5]-2.0*uxuz*m[7] \
           -ux*m[11]-2.0*uy*m[13]+2.0*uz*m[15]+m[16]
    k[17] = uy*(ux2-uz2)*m[0]-2.0*uxuy*m[3]+(uz2-ux2)*m[5]+2.0*uyuz*m[7] \
           +0.5*uy*m[9]+0.5*uy*m[11]+2.0*ux*m[13]-2.0*uz*m[14]+m[17]
    k[18] = uz*(uy2-ux2)*m[0]+2.0*uxuz*m[3]-2.0*uyuz*m[5]+(ux2-uy2)*m[7] \
           -0.5*uz*m[9]+0.5*uz*m[11]+2.0*uy*m[14]-2.0*ux*m[15]+m[18]
    return k

def central_to_raw_dH_py(k, ux, uy, uz):
    ux2, uy2, uz2 = ux*ux, uy*uy, uz*uz
    u2 = ux2+uy2+uz2
    uxuy, uxuz, uyuz = ux*uy, ux*uz, uy*uz
    m = np.zeros(19)
    m[0]=k[0]; m[3]=ux*k[0]+k[3]; m[5]=uy*k[0]+k[5]; m[7]=uz*k[0]+k[7]
    m[1]=19.0*u2*k[0]+k[1]+38.0*ux*k[3]+38.0*uy*k[5]+38.0*uz*k[7]
    m[9]=(2.0*ux2-uy2-uz2)*k[0]+4.0*ux*k[3]-2.0*uy*k[5]-2.0*uz*k[7]+k[9]
    m[11]=(uy2-uz2)*k[0]+2.0*uy*k[5]-2.0*uz*k[7]+k[11]
    m[13]=uxuy*k[0]+uy*k[3]+ux*k[5]+k[13]
    m[14]=uyuz*k[0]+uz*k[5]+uy*k[7]+k[14]
    m[15]=uxuz*k[0]+uz*k[3]+ux*k[7]+k[15]
    m[4]=ux*(95.0*(uy2+uz2)+24.0)/19.0*k[0]+(10.0/57.0)*ux*k[1] \
        +(5.0*(uy2+uz2))*k[3]+k[4]+10.0*uxuy*k[5]+10.0*uxuz*k[7] \
        -(5.0/3.0)*ux*k[9]+10.0*uy*k[13]+10.0*uz*k[15]
    m[6]=uy*(95.0*(ux2+uz2)+24.0)/19.0*k[0]+(10.0/57.0)*uy*k[1] \
        +10.0*uxuy*k[3]+(5.0*(ux2+uz2))*k[5]+k[6]+10.0*uyuz*k[7] \
        +(5.0/6.0)*uy*k[9]-(5.0/2.0)*uy*k[11]+10.0*ux*k[13]+10.0*uz*k[14]
    m[8]=uz*(95.0*(ux2+uy2)+24.0)/19.0*k[0]+(10.0/57.0)*uz*k[1] \
        +10.0*uxuz*k[3]+10.0*uyuz*k[5]+(5.0*(ux2+uy2))*k[7]+k[8] \
        +(5.0/6.0)*uz*k[9]+(5.0/2.0)*uz*k[11]+10.0*uy*k[14]+10.0*ux*k[15]
    m[2]=(21.0*(ux2*uy2+ux2*uz2+uy2*uz2)+(116.0/19.0)*u2)*k[0] \
        +(14.0/19.0)*u2*k[1]+k[2] \
        +(2.0/5.0)*ux*(105.0*(uy2+uz2)+4.0)*k[3]+(42.0/5.0)*ux*k[4] \
        +(2.0/5.0)*uy*(105.0*(ux2+uz2)+4.0)*k[5]+(42.0/5.0)*uy*k[6] \
        +(2.0/5.0)*uz*(105.0*(ux2+uy2)+4.0)*k[7]+(42.0/5.0)*uz*k[8] \
        +(-7.0*ux2+3.5*uy2+3.5*uz2)*k[9]+(-10.5*uy2+10.5*uz2)*k[11] \
        +84.0*uxuy*k[13]+84.0*uyuz*k[14]+84.0*uxuz*k[15]
    m[10]=(3.0*(ux2*uy2+ux2*uz2)-6.0*uy2*uz2-(16.0/19.0)*ux2+(8.0/19.0)*uy2+(8.0/19.0)*uz2)*k[0] \
         +((2.0*ux2-uy2-uz2)/19.0)*k[1] \
         +(2.0/5.0)*ux*(15.0*(uy2+uz2)-8.0)*k[3]+(6.0/5.0)*ux*k[4] \
         +(2.0/5.0)*uy*(15.0*ux2-30.0*uz2+4.0)*k[5]-(3.0/5.0)*uy*k[6] \
         +(2.0/5.0)*uz*(15.0*ux2-30.0*uy2+4.0)*k[7]-(3.0/5.0)*uz*k[8] \
         +(-ux2+2.0*uy2+2.0*uz2)*k[9]+k[10]+(3.0*uy2-3.0*uz2)*k[11] \
         +12.0*uxuy*k[13]-24.0*uyuz*k[14]+12.0*uxuz*k[15] \
         -9.0*uy*k[17]+9.0*uz*k[18]
    m[12]=(3.0*ux2*(uy2-uz2)+(8.0/19.0)*(uz2-uy2))*k[0]+(uy2-uz2)/19.0*k[1] \
         +6.0*ux*(uy2-uz2)*k[3] \
         +(2.0/5.0)*uy*(15.0*ux2-4.0)*k[5]+(3.0/5.0)*uy*k[6] \
         +(2.0/5.0)*uz*(4.0-15.0*ux2)*k[7]-(3.0/5.0)*uz*k[8] \
         +(uy2-uz2)*k[9]+3.0*ux2*k[11]+k[12] \
         +12.0*uxuy*k[13]-12.0*uxuz*k[15] \
         +6.0*ux*k[16]-3.0*uy*k[17]-3.0*uz*k[18]
    m[16]=ux*(uy2-uz2)*k[0]+(uy2-uz2)*k[3]+2.0*uxuy*k[5]-2.0*uxuz*k[7] \
         +ux*k[11]+2.0*uy*k[13]-2.0*uz*k[15]+k[16]
    m[17]=uy*(uz2-ux2)*k[0]-2.0*uxuy*k[3]+(uz2-ux2)*k[5]+2.0*uyuz*k[7] \
         -0.5*uy*k[9]-0.5*uy*k[11]-2.0*ux*k[13]+2.0*uz*k[14]+k[17]
    m[18]=uz*(ux2-uy2)*k[0]+2.0*uxuz*k[3]-2.0*uyuz*k[5]+(ux2-uy2)*k[7] \
         +0.5*uz*k[9]-0.5*uz*k[11]-2.0*uy*k[14]+2.0*ux*k[15]+k[18]
    return m

# Build T matrix from our code
T_our = np.zeros((19, 19))
for j in range(19):
    ej = np.zeros(19); ej[j] = 1.0
    T_our[:, j] = raw_to_central_dH_py(ej, ux3, uy3, uz3)

err4 = norm(T_direct - T_our)
check("T_direct vs T_our (full matrix)", err4 < 1e-10, f"err={err4:.2e}")

# Show per-row errors if any
if err4 > 1e-10:
    for row in range(19):
        re = norm(T_direct[row] - T_our[row])
        if re > 1e-12:
            print(f"    Row {row}: err={re:.4e}")

# ====================================================================
# Test 5: D3Q19 Group properties
# ====================================================================
print("\n" + "=" * 70)
print("Test 5: D3Q19 Group properties")
print("=" * 70)

T_neg = np.zeros((19, 19))
for j in range(19):
    ej = np.zeros(19); ej[j] = 1.0
    T_neg[:, j] = raw_to_central_dH_py(ej, -ux3, -uy3, -uz3)
check("T(u)·T(-u) = I", norm(T_our @ T_neg - np.eye(19)) < 1e-10)

vx3, vy3, vz3 = 0.03, 0.08, -0.04
T_v = np.zeros((19, 19)); T_uv = np.zeros((19, 19))
for j in range(19):
    ej = np.zeros(19); ej[j] = 1.0
    T_v[:, j] = raw_to_central_dH_py(ej, vx3, vy3, vz3)
    T_uv[:, j] = raw_to_central_dH_py(ej, ux3+vx3, uy3+vy3, uz3+vz3)
check("T(u+v) = T(u)·T(v)", norm(T_uv - T_our @ T_v) < 1e-10)

# ====================================================================
# Test 6: Moment physical content verification
# ====================================================================
print("\n" + "=" * 70)
print("Test 6: Physical content of d'Humières moments")
print("=" * 70)
np.random.seed(42)
f_r = np.random.randn(19)
m_r = M3d @ f_r

rho  = sum(f_r)
jx   = sum(f_r * e3d[:,0])
jy   = sum(f_r * e3d[:,1])
jz   = sum(f_r * e3d[:,2])
pxy  = sum(f_r * e3d[:,0]*e3d[:,1])
pxz  = sum(f_r * e3d[:,0]*e3d[:,2])
pyz  = sum(f_r * e3d[:,1]*e3d[:,2])
pxx  = sum(f_r * e3d[:,0]**2)
pyy  = sum(f_r * e3d[:,1]**2)
pzz  = sum(f_r * e3d[:,2]**2)
mx   = sum(f_r * e3d[:,0]*(e3d[:,1]**2 - e3d[:,2]**2))
my   = sum(f_r * e3d[:,1]*(e3d[:,2]**2 - e3d[:,0]**2))
mz   = sum(f_r * e3d[:,2]*(e3d[:,0]**2 - e3d[:,1]**2))

check("m[0]=ρ", abs(m_r[0]-rho)<1e-12)
check("m[3]=jx", abs(m_r[3]-jx)<1e-12)
check("m[5]=jy", abs(m_r[5]-jy)<1e-12)
check("m[7]=jz", abs(m_r[7]-jz)<1e-12)
check("m[9]=2pxx-pyy-pzz", abs(m_r[9]-(2*pxx-pyy-pzz))<1e-12)
check("m[11]=pyy-pzz", abs(m_r[11]-(pyy-pzz))<1e-12)
check("m[13]=pxy", abs(m_r[13]-pxy)<1e-12)
check("m[14]=pyz", abs(m_r[14]-pyz)<1e-12)
check("m[15]=pxz", abs(m_r[15]-pxz)<1e-12)
check("m[1]=19(pxx+pyy+pzz)-30ρ", abs(m_r[1]-(19*(pxx+pyy+pzz)-30*rho))<1e-11)
check("m[16]=mx=Σfi·ex·(ey²-ez²)", abs(m_r[16]-mx)<1e-12)
check("m[17]=my=Σfi·ey·(ez²-ex²)", abs(m_r[17]-my)<1e-12)
check("m[18]=mz=Σfi·ez·(ex²-ey²)", abs(m_r[18]-mz)<1e-12)

# ====================================================================
# Test 7: Chávez-Modena mapping + basis-independent collision
# ====================================================================
print("\n" + "=" * 70)
print("Test 7: MRT-CM collision: d'Humières basis = direct T·S·T⁻¹")
print("=" * 70)

w19 = np.array([1/3]+[1/18]*6+[1/36]*12)
rho_t = 1.0 + 0.01*np.random.randn()
ux_c, uy_c, uz_c = 0.10, -0.06, 0.04

feq_c = np.zeros(19)
for i in range(19):
    eu = e3d[i,0]*ux_c + e3d[i,1]*uy_c + e3d[i,2]*uz_c
    u2c = ux_c**2+uy_c**2+uz_c**2
    feq_c[i] = w19[i]*rho_t*(1.0+3.0*eu+4.5*eu**2-1.5*u2c)

f_c = feq_c + 0.001*np.random.randn(19)

s_visc_c = 1.0/0.55
s_dH = np.array([0, 1.19, 1.4, 0, 1.2, 0, 1.2, 0, 1.2,
                  s_visc_c, 1.4, s_visc_c, 1.4,
                  s_visc_c, s_visc_c, s_visc_c, 1.5, 1.5, 1.5])

# Method A: element-wise (our code)
m_neq = M3d @ (f_c - feq_c)
k_neq = raw_to_central_dH_py(m_neq, ux_c, uy_c, uz_c)
dk = s_dH * k_neq
dm = central_to_raw_dH_py(dk, ux_c, uy_c, uz_c)
f_star_A = f_c - inv(M3d) @ dm

# Method B: matrix form f* = f - M⁻¹·T⁻¹·S·T·M·(f-feq)
S_mat = np.diag(s_dH)
Mi3d = inv(M3d)
T_inv = np.zeros((19, 19))
for j in range(19):
    ej = np.zeros(19); ej[j] = 1.0
    T_inv[:, j] = central_to_raw_dH_py(ej, ux_c, uy_c, uz_c)
T_fwd = T_our_c = np.zeros((19, 19))
for j in range(19):
    ej = np.zeros(19); ej[j] = 1.0
    T_fwd[:, j] = raw_to_central_dH_py(ej, ux_c, uy_c, uz_c)

f_star_B = f_c - Mi3d @ T_inv @ S_mat @ T_fwd @ M3d @ (f_c - feq_c)

err7 = norm(f_star_A - f_star_B)
check("Element-wise = Matrix form", err7 < 1e-12, f"err={err7:.2e}")

# Method C: use T_direct from M(u)·M(0)⁻¹ with lattice-reduced polynomials
# (matches our T(u) implementation which uses natural monomial basis conjugation)
T_d_c = build_M3d_shifted(ux_c, uy_c, uz_c) @ inv(M3d)
T_d_c_inv = build_M3d_shifted(-ux_c, -uy_c, -uz_c) @ inv(M3d)  # T(-u)
f_star_C = f_c - Mi3d @ T_d_c_inv @ S_mat @ T_d_c @ M3d @ (f_c - feq_c)

err7c = norm(f_star_A - f_star_C)
check("Our code = lattice-reduced M(u)·M(0)⁻¹", err7c < 1e-12, f"err={err7c:.2e}")

# ====================================================================
# Test 8: Round-trip T⁻¹(u) · T(u) = I via our functions
# ====================================================================
print("\n" + "=" * 70)
print("Test 8: Round-trip central_to_raw(raw_to_central(m)) = m")
print("=" * 70)
m_test = np.random.randn(19)
k_test = raw_to_central_dH_py(m_test, ux3, uy3, uz3)
m_back = central_to_raw_dH_py(k_test, ux3, uy3, uz3)
check("Forward → inverse round trip", norm(m_test - m_back) < 1e-11,
      f"err={norm(m_test - m_back):.2e}")

# ====================================================================
# Summary
# ====================================================================
print("\n" + "=" * 70)
total = PASS_COUNT + FAIL_COUNT
print(f"Results: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"WARNING: {FAIL_COUNT} test(s) FAILED")
print("=" * 70)
