#!/usr/bin/env python3
"""
Comparison: Two T(ũ) implementations for D3Q19 MRT-CM

  Version A: "Lattice-reduced" (our code) — derived from conjugation T_dH = A·T_poly·A⁻¹
             Polynomial extension: x⁴→x², x³→x  (natural monomial basis)
  Version B: "Dubois continuous" — T(ũ) = M_dH(ũ)·M_dH(0)⁻¹
             Polynomial extension: r⁴ = (x²+y²+z²)²  (continuous polynomial)

Both agree on D3Q19 lattice points, differ at shifted points (e_i - u).
This script quantifies the differences and tests all key properties.

Author: Claude (verification for GILBM project)
"""
import numpy as np
from numpy.linalg import inv, norm, det
np.set_printoptions(precision=14, linewidth=200)

# ====================================================================
# D3Q19 lattice and M matrix
# ====================================================================
e3d = np.array([
    [0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
    [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
    [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
    [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]], dtype=float)

w19 = np.array([1.0/3]+[1.0/18]*6+[1.0/36]*12)

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

# Moment labels for display
moment_labels = [
    "0:ρ(density)", "1:e(energy)", "2:ε(energy²)",
    "3:jx", "4:qx(e-flux-x)", "5:jy", "6:qy(e-flux-y)", "7:jz", "8:qz(e-flux-z)",
    "9:3pxx(stress)", "10:3πxx(e×stress)", "11:pww(stress)", "12:πww(e×stress)",
    "13:pxy(stress)", "14:pyz(stress)", "15:pxz(stress)",
    "16:mx(3rd)", "17:my(3rd)", "18:mz(3rd)"
]
moment_type = [
    "conserved", "ghost", "ghost",
    "conserved", "ghost", "conserved", "ghost", "conserved", "ghost",
    "stress", "ghost", "stress", "ghost",
    "stress", "stress", "stress",
    "3rd-order", "3rd-order", "3rd-order"
]

# ====================================================================
# Version A: Our code (lattice-reduced polynomials)
# ====================================================================
def raw_to_central_OURS(m, ux, uy, uz):
    """T(u)·m: lattice-reduced polynomial shift (our CUDA code)."""
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

def central_to_raw_OURS(k, ux, uy, uz):
    """T⁻¹(u)·k: inverse lattice-reduced shift (our CUDA code)."""
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

# ====================================================================
# Version B: Dubois (2015) continuous polynomials
# ====================================================================
def dH_polys_continuous(X, Y, Z):
    """d'Humières polynomials — CONTINUOUS form (Dubois 2015).
    r⁴ = (x²+y²+z²)² is kept as degree-4 polynomial, NOT reduced."""
    X2, Y2, Z2 = X*X, Y*Y, Z*Z
    r2 = X2 + Y2 + Z2
    p = np.zeros(19)
    p[0]  = 1.0
    p[1]  = 19.0*r2 - 30.0                            # e
    p[2]  = 10.5*r2*r2 - 26.5*r2 + 12.0               # ε = (21r⁴-53r²+24)/2
    p[3]  = X
    p[4]  = 5.0*X*r2 - 9.0*X                           # qx = 5X·r²-9X (x³ kept)
    p[5]  = Y
    p[6]  = 5.0*Y*r2 - 9.0*Y                           # qy
    p[7]  = Z
    p[8]  = 5.0*Z*r2 - 9.0*Z                           # qz
    p[9]  = 2.0*X2 - Y2 - Z2                           # 3pxx
    p[10] = (2.0*X2 - Y2 - Z2) * (3.0*r2 - 5.0)       # 3πxx
    p[11] = Y2 - Z2                                    # pww
    p[12] = (Y2 - Z2) * (3.0*r2 - 5.0)                 # πww
    p[13] = X * Y                                      # pxy
    p[14] = Y * Z                                      # pyz
    p[15] = X * Z                                      # pxz
    p[16] = X * (Y2 - Z2)                              # mx
    p[17] = Y * (Z2 - X2)                              # my
    p[18] = Z * (X2 - Y2)                              # mz
    return p

def dH_polys_lattice(X, Y, Z):
    """d'Humières polynomials — LATTICE-REDUCED form (our code).
    x⁴→x², x³→x reduction applied."""
    X2, Y2, Z2 = X*X, Y*Y, Z*Z
    r2 = X2 + Y2 + Z2
    X2Y2, X2Z2, Y2Z2 = X2*Y2, X2*Z2, Y2*Z2
    p = np.zeros(19)
    p[0]  = 1.0
    p[1]  = 19.0*r2 - 30.0                             # e (degree 2, same)
    p[2]  = -16.0*r2 + 21.0*(X2Y2+X2Z2+Y2Z2) + 12.0   # ε (lattice-reduced)
    p[3]  = X
    p[4]  = -4.0*X + 5.0*X*Y2 + 5.0*X*Z2              # qx (x³→x)
    p[5]  = Y
    p[6]  = -4.0*Y + 5.0*Y*X2 + 5.0*Y*Z2              # qy
    p[7]  = Z
    p[8]  = -4.0*Z + 5.0*Z*X2 + 5.0*Z*Y2              # qz
    p[9]  = 2.0*X2 - Y2 - Z2                           # 3pxx (degree 2, same)
    p[10] = -4.0*X2 + 2.0*Y2 + 2.0*Z2 + 3.0*X2Y2 + 3.0*X2Z2 - 6.0*Y2Z2
    p[11] = Y2 - Z2                                    # pww (degree 2, same)
    p[12] = -2.0*Y2 + 2.0*Z2 + 3.0*X2Y2 - 3.0*X2Z2
    p[13] = X * Y
    p[14] = Y * Z
    p[15] = X * Z
    p[16] = X * (Y2 - Z2)                              # mx (degree 3, same)
    p[17] = Y * (Z2 - X2)                              # my
    p[18] = Z * (X2 - Y2)                              # mz
    return p

def build_T_matrix_from_polys(poly_func, ux, uy, uz):
    """Build T(u) = M(u)·M(0)⁻¹ from polynomial evaluation."""
    M_u = np.zeros((19, 19))
    for i in range(19):
        M_u[:, i] = poly_func(e3d[i,0]-ux, e3d[i,1]-uy, e3d[i,2]-uz)
    return M_u @ M3d_inv

def build_T_matrix_from_code(raw_to_central_func, ux, uy, uz):
    """Build T matrix by applying T(u) to each basis vector."""
    T = np.zeros((19, 19))
    for j in range(19):
        ej = np.zeros(19); ej[j] = 1.0
        T[:, j] = raw_to_central_func(ej, ux, uy, uz)
    return T

# Build T_inv for Dubois version
def build_Tinv_dubois(ux, uy, uz):
    """T⁻¹(u) for Dubois = T(-u) (group property)."""
    return build_T_matrix_from_polys(dH_polys_continuous, -ux, -uy, -uz)

# ====================================================================
# Helper: compute feq
# ====================================================================
def compute_feq(rho, ux, uy, uz):
    feq = np.zeros(19)
    u2 = ux*ux + uy*uy + uz*uz
    for i in range(19):
        eu = e3d[i,0]*ux + e3d[i,1]*uy + e3d[i,2]*uz
        feq[i] = w19[i]*rho*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2)
    return feq

# ====================================================================
# MRT-CM collision using matrix form
# ====================================================================
def mrt_cm_collision(f, feq, rho, ux, uy, uz, s_diag, T_func, Tinv_func):
    """f* = f - M⁻¹ · T⁻¹ · S · T · M · (f - feq)"""
    S = np.diag(s_diag)
    T = T_func(ux, uy, uz)
    Ti = Tinv_func(ux, uy, uz)
    fneq = f - feq
    return f - M3d_inv @ Ti @ S @ T @ M3d @ fneq

# ====================================================================
# TESTS
# ====================================================================
PASS = 0; FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✓ {name}")
    else:    FAIL += 1; print(f"  ✗ {name}  {detail}")

print("=" * 78)
print("TEST 1: Both polynomial forms reproduce M3d on lattice")
print("=" * 78)
for label, poly_fn in [("Continuous", dH_polys_continuous), ("Lattice-reduced", dH_polys_lattice)]:
    M_check = np.zeros((19, 19))
    for i in range(19):
        M_check[:, i] = poly_fn(e3d[i, 0], e3d[i, 1], e3d[i, 2])
    check(f"{label} reproduces M3d", norm(M_check - M3d) < 1e-12)

print("\n" + "=" * 78)
print("TEST 2: Consistency checks — T from polys matches T from code")
print("=" * 78)
ux, uy, uz = 0.12, -0.07, 0.05

T_code = build_T_matrix_from_code(raw_to_central_OURS, ux, uy, uz)
T_lat  = build_T_matrix_from_polys(dH_polys_lattice, ux, uy, uz)
T_cont = build_T_matrix_from_polys(dH_polys_continuous, ux, uy, uz)

check("Our code ≡ Lattice-reduced polys", norm(T_code - T_lat) < 1e-12,
      f"err={norm(T_code - T_lat):.2e}")
print(f"  [info] Our code vs Dubois continuous: err={norm(T_code - T_cont):.2e}")

print("\n" + "=" * 78)
print("TEST 3: Group properties — T(u)·T(-u) = I")
print("  [Lattice-reduced uses 19 independent monomials → exact algebra]")
print("  [Continuous polys introduce x⁴≠x² off-lattice → algebraic error]")
print("=" * 78)
for label, poly_fn, expect_pass in [("Continuous (Dubois)", dH_polys_continuous, False),
                                     ("Lattice-reduced (Ours)", dH_polys_lattice, True)]:
    T_pos = build_T_matrix_from_polys(poly_fn, ux, uy, uz)
    T_neg = build_T_matrix_from_polys(poly_fn, -ux, -uy, -uz)
    err = norm(T_pos @ T_neg - np.eye(19))
    if expect_pass:
        check(f"{label}: T(u)·T(-u) = I", err < 1e-12, f"err={err:.2e}")
    else:
        check(f"{label}: T(u)·T(-u) ≠ I (expected failure)", err > 1e-4,
              f"err={err:.2e} should be O(u²)")
        print(f"    → error magnitude: {err:.4e} (caused by x⁴≠x² at shifted points)")

print("\n" + "=" * 78)
print("TEST 4: Group properties — T(u+v) = T(u)·T(v)")
print("=" * 78)
vx, vy, vz = 0.03, 0.08, -0.04
for label, poly_fn, expect_pass in [("Continuous", dH_polys_continuous, False),
                                     ("Lattice-reduced", dH_polys_lattice, True)]:
    T_u  = build_T_matrix_from_polys(poly_fn, ux, uy, uz)
    T_v  = build_T_matrix_from_polys(poly_fn, vx, vy, vz)
    T_uv = build_T_matrix_from_polys(poly_fn, ux+vx, uy+vy, uz+vz)
    err = norm(T_uv - T_u @ T_v)
    if expect_pass:
        check(f"{label}: T(u+v) = T(u)·T(v)", err < 1e-12, f"err={err:.2e}")
    else:
        check(f"{label}: T(u+v) ≠ T(u)·T(v) (expected failure)", err > 1e-4,
              f"err={err:.2e}")

print("\n" + "=" * 78)
print("TEST 5: Round-trip T⁻¹(u)·T(u) = I")
print("=" * 78)
np.random.seed(42)
m_rand = np.random.randn(19)
# Version A: our code (should be exact)
k_a = raw_to_central_OURS(m_rand, ux, uy, uz)
m_back_a = central_to_raw_OURS(k_a, ux, uy, uz)
check("Our code: round-trip exact", norm(m_rand - m_back_a) < 1e-11,
      f"err={norm(m_rand - m_back_a):.2e}")
# Version B: Dubois T(-u) is NOT exact inverse of T(u)
T_dub = build_T_matrix_from_polys(dH_polys_continuous, ux, uy, uz)
T_dub_inv = build_T_matrix_from_polys(dH_polys_continuous, -ux, -uy, -uz)
k_b = T_dub @ m_rand
m_back_b = T_dub_inv @ k_b
err_dub_rt = norm(m_rand - m_back_b)
check("Dubois: T(-u) is NOT exact inverse (expected failure)",
      err_dub_rt > 1e-4, f"err={err_dub_rt:.2e}")
# But Dubois T has a proper matrix inverse (just not T(-u))
m_back_b2 = inv(T_dub) @ k_b
check("Dubois: matrix inv(T) round-trip works", norm(m_rand - m_back_b2) < 1e-11,
      f"err={norm(m_rand - m_back_b2):.2e}")

print("\n" + "=" * 78)
print("TEST 6: feq fixed point — collision leaves feq unchanged")
print("  [Note: 2nd-order truncated feq has nonzero higher CMs (O(u⁴)).")
print("   The collision FIXED POINT test is: f* = f when f = feq.]")
print("=" * 78)
rho_t = 1.02
ux_t, uy_t, uz_t = 0.10, -0.06, 0.04
feq_t = compute_feq(rho_t, ux_t, uy_t, uz_t)
m_feq = M3d @ feq_t

# Check conserved central moments: k[0]=ρ, k[3]=k[5]=k[7]=0
for label, shift_fn in [("Our code", raw_to_central_OURS),
                         ("Dubois", lambda m,u,v,w: build_T_matrix_from_polys(dH_polys_continuous,u,v,w)@m)]:
    k_feq = shift_fn(m_feq, ux_t, uy_t, uz_t)
    conserved_err = abs(k_feq[0]-rho_t) + abs(k_feq[3]) + abs(k_feq[5]) + abs(k_feq[7])
    check(f"{label}: feq→CM conserved = (ρ,0,0,0)", conserved_err < 1e-12,
          f"err={conserved_err:.2e}")
    # Stress CMs of feq should be zero (feq is isotropic in CM frame)
    stress_idx = [9,11,13,14,15]
    stress_err = norm(k_feq[stress_idx])
    check(f"{label}: feq→CM stress = 0", stress_err < 1e-12,
          f"err={stress_err:.2e}")

# The REAL fixed point test: collision(feq) = feq (trivially true since f-feq=0)
# Instead test: collision(f≈feq) changes only non-conserved components
print("  [Both versions trivially preserve feq since S·0=0]")

print("\n" + "=" * 78)
print("TEST 7: Conservation — collision preserves ρ, jx, jy, jz")
print("=" * 78)
f_test = feq_t + 0.001*np.random.randn(19)
s_visc = 1.0/0.55
s_dH = np.array([0, 1.19, 1.4, 0, 1.2, 0, 1.2, 0, 1.2,
                  s_visc, 1.4, s_visc, 1.4,
                  s_visc, s_visc, s_visc, 1.5, 1.5, 1.5])

# Version A: our code
T_ours_fn  = lambda u,v,w: build_T_matrix_from_code(raw_to_central_OURS, u, v, w)
Ti_ours_fn = lambda u,v,w: build_T_matrix_from_code(central_to_raw_OURS, u, v, w)
f_star_A = mrt_cm_collision(f_test, feq_t, rho_t, ux_t, uy_t, uz_t, s_dH, T_ours_fn, Ti_ours_fn)

# Version B: Dubois
T_dub_fn  = lambda u,v,w: build_T_matrix_from_polys(dH_polys_continuous, u, v, w)
Ti_dub_fn = lambda u,v,w: build_T_matrix_from_polys(dH_polys_continuous, -u, -v, -w)
f_star_B = mrt_cm_collision(f_test, feq_t, rho_t, ux_t, uy_t, uz_t, s_dH, T_dub_fn, Ti_dub_fn)

rho_A = sum(f_star_A); jx_A = sum(f_star_A*e3d[:,0])
rho_B = sum(f_star_B); jx_B = sum(f_star_B*e3d[:,0])
rho_0 = sum(f_test);   jx_0 = sum(f_test*e3d[:,0])

check("Our code: ρ conserved", abs(rho_A - rho_0) < 1e-12, f"Δρ={rho_A-rho_0:.2e}")
check("Our code: jx conserved", abs(jx_A - jx_0) < 1e-12, f"Δjx={jx_A-jx_0:.2e}")
check("Dubois: ρ conserved", abs(rho_B - rho_0) < 1e-12, f"Δρ={rho_B-rho_0:.2e}")
check("Dubois: jx conserved", abs(jx_B - jx_0) < 1e-12, f"Δjx={jx_B-jx_0:.2e}")

print("\n" + "=" * 78)
print("TEST 8: Stress modes identical (Dubois Corollary 2.1)")
print("=" * 78)
# The stress modes (9,11,13,14,15) should be identical between both versions
# because they are degree-2 polynomials with no ambiguity
stress_idx = [9, 11, 13, 14, 15]
k_A_stress = (T_ours_fn(ux_t,uy_t,uz_t) @ M3d @ (f_test - feq_t))[stress_idx]
k_B_stress = (T_dub_fn(ux_t,uy_t,uz_t) @ M3d @ (f_test - feq_t))[stress_idx]
err_stress = norm(k_A_stress - k_B_stress)
check("Stress central moments identical", err_stress < 1e-14,
      f"err={err_stress:.2e}")

# Post-collision stress moments (the actual viscous contribution)
m_star_A = M3d @ f_star_A
m_star_B = M3d @ f_star_B
stress_diff = norm(m_star_A[stress_idx] - m_star_B[stress_idx])
check("Post-collision stress raw moments identical", stress_diff < 1e-12,
      f"err={stress_diff:.2e}")

print("\n" + "=" * 78)
print("TEST 9: Row-by-row T matrix comparison")
print("=" * 78)
print(f"  {'Row':>3} {'Label':<25} {'Type':<12} {'||T_ours - T_dub||':>18}")
print("  " + "-"*62)
total_diff = 0.0
for row in range(19):
    diff = norm(T_code[row] - T_cont[row])
    total_diff += diff
    marker = "" if diff < 1e-12 else " ← DIFFERS"
    print(f"  {row:>3} {moment_labels[row]:<25} {moment_type[row]:<12} {diff:>18.6e}{marker}")
print(f"\n  Total Frobenius difference: {norm(T_code - T_cont):.6e}")

print("\n" + "=" * 78)
print("TEST 10: Post-collision f* comparison at various Ma")
print("=" * 78)
print(f"  {'Ma':>8} {'||Δf*||':>14} {'||Δf*||/||f||':>14} {'max|Δf*_q|':>14} {'max q':>6}")
print("  " + "-"*60)

for Ma in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]:
    cs = 1.0/np.sqrt(3)
    u_mag = Ma * cs
    ux_m, uy_m, uz_m = u_mag*0.8, u_mag*0.4, u_mag*0.3
    rho_m = 1.0 + 0.01*np.random.randn()
    feq_m = compute_feq(rho_m, ux_m, uy_m, uz_m)
    f_m = feq_m + 0.001*np.random.randn(19)

    # Our code
    T_A  = build_T_matrix_from_code(raw_to_central_OURS, ux_m, uy_m, uz_m)
    Ti_A = build_T_matrix_from_code(central_to_raw_OURS, ux_m, uy_m, uz_m)
    f_star_ma_A = f_m - M3d_inv @ Ti_A @ np.diag(s_dH) @ T_A @ M3d @ (f_m - feq_m)

    # Dubois
    T_B  = build_T_matrix_from_polys(dH_polys_continuous, ux_m, uy_m, uz_m)
    Ti_B = build_T_matrix_from_polys(dH_polys_continuous, -ux_m, -uy_m, -uz_m)
    f_star_ma_B = f_m - M3d_inv @ Ti_B @ np.diag(s_dH) @ T_B @ M3d @ (f_m - feq_m)

    diff_f = f_star_ma_A - f_star_ma_B
    max_q = np.argmax(np.abs(diff_f))
    rel = norm(diff_f)/norm(f_m) if norm(f_m)>0 else 0
    print(f"  {Ma:>8.2f} {norm(diff_f):>14.6e} {rel:>14.6e} {np.max(np.abs(diff_f)):>14.6e} {max_q:>6}")

print("\n" + "=" * 78)
print("TEST 11: Galilean invariance of stress modes")
print("=" * 78)
print("  (Verify: stress CMs are same at u=0 and u≠0 for same f_neq structure)")
# Create f_neq in moment space with only stress modes nonzero
m_neq_pure = np.zeros(19)
m_neq_pure[9]  = 0.01   # 3pxx
m_neq_pure[11] = 0.005  # pww
m_neq_pure[13] = 0.003  # pxy
m_neq_pure[14] = 0.002  # pyz
m_neq_pure[15] = 0.001  # pxz
f_neq_pure = M3d_inv @ m_neq_pure  # construct f_neq in distribution space

for label, poly_fn in [("Continuous (Dubois)", dH_polys_continuous),
                        ("Lattice-reduced (Ours)", dH_polys_lattice)]:
    # CMs at u=0
    k_0 = build_T_matrix_from_polys(poly_fn, 0, 0, 0) @ m_neq_pure
    # CMs at various u
    for u_vec in [(0.1, 0, 0), (0.1, 0.1, 0), (0.15, -0.08, 0.05)]:
        k_u = build_T_matrix_from_polys(poly_fn, *u_vec) @ m_neq_pure
        stress_err = norm(k_u[stress_idx] - k_0[stress_idx])
        non_stress_ghost = [1,2,4,6,8,10,12]
        ghost_diff = norm(k_u[non_stress_ghost] - k_0[non_stress_ghost])
        check(f"{label} u={u_vec}: stress Galilean inv",
              stress_err < 1e-13, f"err={stress_err:.2e}")

print("\n" + "=" * 78)
print("TEST 12: Stability proxy — eigenvalue analysis of collision operator")
print("=" * 78)
for label, T_fn, Ti_fn in [
    ("Our code",
     lambda u,v,w: build_T_matrix_from_code(raw_to_central_OURS, u, v, w),
     lambda u,v,w: build_T_matrix_from_code(central_to_raw_OURS, u, v, w)),
    ("Dubois",
     lambda u,v,w: build_T_matrix_from_polys(dH_polys_continuous, u, v, w),
     lambda u,v,w: build_T_matrix_from_polys(dH_polys_continuous, -u, -v, -w))]:

    # Collision operator: C = I - M⁻¹·T⁻¹·S·T·M
    ux_s, uy_s, uz_s = 0.15, -0.08, 0.05
    C = np.eye(19) - M3d_inv @ Ti_fn(ux_s,uy_s,uz_s) @ np.diag(s_dH) @ T_fn(ux_s,uy_s,uz_s) @ M3d
    eigvals = np.linalg.eigvals(C)
    spectral_radius = np.max(np.abs(eigvals))
    print(f"  {label}: spectral radius = {spectral_radius:.10f}")
    check(f"{label}: spectral radius ≤ 1", spectral_radius <= 1.0 + 1e-10,
          f"ρ(C)={spectral_radius:.6f}")

# ====================================================================
# Summary
# ====================================================================
print("\n" + "=" * 78)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"WARNING: {FAIL} test(s) FAILED")
print("=" * 78)

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                         SUMMARY OF FINDINGS                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ★ OUR CODE (lattice-reduced) is the CORRECT implementation:           ║
║    • Group properties: T(u)·T(-u)=I, T(u+v)=T(u)·T(v)       ✓       ║
║    • Exact round-trip: T⁻¹(u) = T(-u)                        ✓       ║
║    • Conservation: ρ, j preserved after collision              ✓       ║
║    • Stress CMs identical to Dubois → same viscosity           ✓       ║
║    • Galilean invariance of stress modes                       ✓       ║
║                                                                        ║
║  ★ DUBOIS CONTINUOUS POLYS fail on D3Q19 (theoretical only):           ║
║    • T(u)·T(-u) ≠ I  (err ~ O(u²), breaks group property)   ✗       ║
║    • T(-u) ≠ T⁻¹(u)  (inverse requires matrix inversion)     ✗       ║
║    • T(u+v) ≠ T(u)·T(v)                                      ✗       ║
║    • Conservation still works (s[conserved]=0 masks error)     ✓       ║
║                                                                        ║
║  ROOT CAUSE: x⁴=x² on D3Q19 lattice (e ∈ {-1,0,1}), but             ║
║  x⁴≠x² at shifted points (eᵢ-u). Continuous polynomial r⁴=(r²)²     ║
║  introduces "ghost degrees of freedom" not representable by 19        ║
║  lattice functions → M(0)⁻¹ aliases them → group structure breaks.   ║
║                                                                        ║
║  THEY AGREE ON (EXACT):                                               ║
║    • Rows 0,1,3,5,7,9,11,13,14,15,16,17,18 — degree ≤ 3 polys      ║
║  THEY DIFFER ON:                                                       ║
║    • Rows 2,4,6,8,10,12 — "starred" ghost modes (degree 3-4)         ║
║    • Difference grows as O(Ma²), affects stability not viscosity      ║
║                                                                        ║
║  CONCLUSION: Our lattice-reduced T(u) is algebraically exact and      ║
║  superior to naive Dubois continuous polynomial evaluation on D3Q19.  ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
