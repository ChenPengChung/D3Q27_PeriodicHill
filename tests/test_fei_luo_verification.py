#!/usr/bin/env python3
"""
Verification of MRT-CM (Central Moment MRT) against Fei & Luo (2017),
"Consistent forcing scheme in the cascaded lattice Boltzmann method",
Phys. Rev. E 96, 053307.

This test suite validates that our D3Q19 MRT-CM implementation follows the
exact same mathematical framework as Fei & Luo's D2Q9 CLBM, generalized
to three dimensions.

Key correspondence (Fei & Luo ↔ Our code):
  Fei & Luo N        ↔  Our T(u)         [raw → central shift]
  Fei & Luo N⁻¹      ↔  Our T⁻¹(u)       [central → raw shift]
  Fei & Luo M         ↔  Our M (d'Humières 19×19)
  Fei & Luo Eq.(7)    ↔  f* = f - M⁻¹·T⁻¹·S·T·M·(f-feq)
  Fei & Luo Eq.(8)    ↔  Central moment equilibrium from continuous MB

10 Tests:
  1. D2Q9 N matrix matches Fei & Luo Eq.(6) exactly
  2. D2Q9 N⁻¹ matrix matches Fei & Luo Eq.(A3) exactly
  3. D2Q9 N·N⁻¹ = I (round-trip)
  4. D2Q9: N=I ⟹ CLBM degrades to MRT-RM (Fei & Luo key result)
  5. D2Q9: Central moment equilibrium = Fei & Luo Eq.(8)
  6. D3Q19: T(u)·T(-u) = I (round-trip, same structure as D2Q9)
  7. D3Q19: T(u+v) = T(u)·T(v) (group morphism, Fei & Luo framework)
  8. D3Q19: N=I degeneracy → CLBM = MRT-RM (generalized Fei & Luo)
  9. D3Q19: Central moment feq matches continuous Maxwell-Boltzmann
 10. D3Q19: CLBM collision Eq.(7)+(9) = our MRT-CM collision
"""
import numpy as np
import sys

# ============================================================================
# D2Q9 definitions (matching Fei & Luo exactly)
# ============================================================================
# Fei & Luo Eq.(1): e_ix, e_iy for D2Q9 (i=0..8)
e2d = np.array([
    [0, 0],   # 0: rest
    [1, 0],   # 1
    [0, 1],   # 2
    [-1, 0],  # 3
    [0, -1],  # 4
    [1, 1],   # 5
    [-1, 1],  # 6
    [-1, -1], # 7
    [1, -1],  # 8
], dtype=float)

W2d = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Fei & Luo Eq.(5): Transformation matrix M for D2Q9
M2d = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0,-1, 0, 1,-1,-1, 1],
    [0, 0, 1, 0,-1, 1, 1,-1,-1],
    [0, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 1,-1, 1,-1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1,-1, 1,-1],
    [0, 0, 0, 0, 0, 1, 1,-1,-1],
    [0, 0, 0, 0, 0, 1,-1,-1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
], dtype=float)

M2d_inv = np.linalg.inv(M2d)

def compute_feq_2d(rho, ux, uy):
    """D2Q9 equilibrium: Fei & Luo Eq.(3) with c=1, c_s²=1/3"""
    feq = np.zeros(9)
    u2 = ux**2 + uy**2
    for i in range(9):
        eu = e2d[i, 0]*ux + e2d[i, 1]*uy
        feq[i] = W2d[i] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u2)
    return feq

def build_N_fei_luo(ux, uy):
    """
    Build Fei & Luo Eq.(6): the shift matrix N (9×9).
    N maps raw moments T to central moments T̃: |T̃⟩ = N|T⟩

    Fei & Luo raw moment ordering (Eq.3):
    T = [k00, k10, k01, k20+k02, k20-k02, k11, k21, k12, k22]
    """
    N = np.eye(9)
    ux2 = ux**2
    uy2 = uy**2

    # Row 0: k̃00 = k00 (density, conserved)
    # N[0,:] = [1, 0, 0, 0, 0, 0, 0, 0, 0]  (already identity)

    # Row 1: k̃10 = -ux*k00 + k10
    N[1, 0] = -ux
    N[1, 1] = 1.0

    # Row 2: k̃01 = -uy*k00 + k01
    N[2, 0] = -uy
    N[2, 2] = 1.0

    # Row 3: k̃(20+02) = (ux²+uy²)*k00 - 2ux*k10 - 2uy*k01 + k(20+02)
    N[3, 0] = ux2 + uy2
    N[3, 1] = -2.0*ux
    N[3, 2] = -2.0*uy
    N[3, 3] = 1.0

    # Row 4: k̃(20-02) = (ux²-uy²)*k00 - 2ux*k10 + 2uy*k01 + k(20-02)
    N[4, 0] = ux2 - uy2
    N[4, 1] = -2.0*ux
    N[4, 2] = 2.0*uy
    N[4, 4] = 1.0

    # Row 5: k̃11 = ux*uy*k00 - uy*k10 - ux*k01 + k11
    N[5, 0] = ux*uy
    N[5, 1] = -uy
    N[5, 2] = -ux
    N[5, 5] = 1.0

    # Row 6: k̃21 = -ux²*uy*k00 + 2ux*uy*k10 + ux²*k01
    #        - uy/2*k(20+02) - uy/2*k(20-02) - 2ux*k11 + k21
    N[6, 0] = -ux2*uy
    N[6, 1] = 2.0*ux*uy
    N[6, 2] = ux2
    N[6, 3] = -uy/2.0
    N[6, 4] = -uy/2.0
    N[6, 5] = -2.0*ux
    N[6, 6] = 1.0

    # Row 7: k̃12 = -ux*uy²*k00 + uy²*k10 + 2ux*uy*k01
    #        - ux/2*k(20+02) + ux/2*k(20-02) - 2uy*k11 + k12
    # NOTE: Fei & Luo Eq.(6) has signs of [3],[4] swapped (typo in paper).
    #   Correct derivation: -ux·k02 = -ux·(k20+k02 - (k20-k02))/2
    #                       = -ux/2·m[3] + ux/2·m[4]
    N[7, 0] = -ux*uy2
    N[7, 1] = uy2
    N[7, 2] = 2.0*ux*uy
    N[7, 3] = -ux/2.0    # paper has +ux/2 (typo)
    N[7, 4] = ux/2.0     # paper has -ux/2 (typo)
    N[7, 5] = -2.0*uy
    N[7, 7] = 1.0

    # Row 8: k̃22 = ux²*uy²*k00 - 2ux*uy²*k10 - 2ux²*uy*k01
    #        + (ux²/2+uy²/2)*k(20+02) + (uy²/2-ux²/2)*k(20-02)
    #        + 4ux*uy*k11 - 2uy*k21 - 2ux*k12 + k22
    N[8, 0] = ux2*uy2
    N[8, 1] = -2.0*ux*uy2
    N[8, 2] = -2.0*ux2*uy
    N[8, 3] = ux2/2.0 + uy2/2.0
    N[8, 4] = uy2/2.0 - ux2/2.0
    N[8, 5] = 4.0*ux*uy
    N[8, 6] = -2.0*uy
    N[8, 7] = -2.0*ux
    N[8, 8] = 1.0

    return N

def build_Ninv_fei_luo(ux, uy):
    """
    Build Fei & Luo Eq.(A3): the inverse shift matrix N⁻¹ (9×9).
    N⁻¹ maps central moments T̃ back to raw moments T: |T⟩ = N⁻¹|T̃⟩
    Obtained by replacing u → -u in N (flip odd-power signs).
    """
    Ni = np.eye(9)
    ux2 = ux**2
    uy2 = uy**2

    # Row 0
    # Ni[0,:] = identity

    # Row 1: ux*k̃00 + k̃10
    Ni[1, 0] = ux
    Ni[1, 1] = 1.0

    # Row 2: uy*k̃00 + k̃01
    Ni[2, 0] = uy
    Ni[2, 2] = 1.0

    # Row 3
    Ni[3, 0] = ux2 + uy2
    Ni[3, 1] = 2.0*ux
    Ni[3, 2] = 2.0*uy
    Ni[3, 3] = 1.0

    # Row 4
    Ni[4, 0] = ux2 - uy2
    Ni[4, 1] = 2.0*ux
    Ni[4, 2] = -2.0*uy
    Ni[4, 4] = 1.0

    # Row 5
    Ni[5, 0] = ux*uy
    Ni[5, 1] = uy
    Ni[5, 2] = ux
    Ni[5, 5] = 1.0

    # Row 6
    Ni[6, 0] = ux2*uy
    Ni[6, 1] = 2.0*ux*uy
    Ni[6, 2] = ux2
    Ni[6, 3] = uy/2.0
    Ni[6, 4] = uy/2.0
    Ni[6, 5] = 2.0*ux
    Ni[6, 6] = 1.0

    # Row 7: N⁻¹ = N(-u), so replace u → -u in N
    # N[7,3] = -ux/2 → N⁻¹[7,3] = +ux/2
    # N[7,4] = +ux/2 → N⁻¹[7,4] = -ux/2
    # NOTE: Fei & Luo Eq.(A3) has same sign typo as Eq.(6) for this row
    Ni[7, 0] = ux*uy2
    Ni[7, 1] = uy2
    Ni[7, 2] = 2.0*ux*uy
    Ni[7, 3] = ux/2.0     # paper has -ux/2 (typo, consistent with Eq.6 typo)
    Ni[7, 4] = -ux/2.0    # paper has +ux/2 (typo)
    Ni[7, 5] = 2.0*uy
    Ni[7, 7] = 1.0

    # Row 8
    Ni[8, 0] = ux2*uy2
    Ni[8, 1] = 2.0*ux*uy2
    Ni[8, 2] = 2.0*ux2*uy
    Ni[8, 3] = ux2/2.0 + uy2/2.0
    Ni[8, 4] = uy2/2.0 - ux2/2.0  # even-power term, same as N[8,4]
    Ni[8, 5] = 4.0*ux*uy
    Ni[8, 6] = 2.0*uy
    Ni[8, 7] = 2.0*ux
    Ni[8, 8] = 1.0

    return Ni


# ============================================================================
# D3Q19 definitions (matching our GILBM code)
# ============================================================================
e3d = np.array([
    [0,0,0],
    [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
    [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
    [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
    [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]
], dtype=float)

W3d = np.array([
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36
])

# d'Humières M matrix (19×19)
M3d = np.array([
    [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [-30,-11,-11,-11,-11,-11,-11,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
    [ 12, -4, -4, -4, -4, -4, -4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
    [  0, -4,  4,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
    [  0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1],
    [  0,  0,  0, -4,  4,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1],
    [  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1],
    [  0,  0,  0,  0,  0, -4,  4,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1],
    [  0,  2,  2, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],
    [  0, -4, -4,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],
    [  0,  0,  0,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],
    [  0,  0,  0, -2, -2,  2,  2,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1, -1,  1,  0,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0, -1, -1,  1,  1,  0,  0,  0,  0,  1, -1,  1, -1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1, -1, -1,  1,  1]
], dtype=float)

M3d_inv = np.linalg.inv(M3d)

# Relaxation rates (d'Humières standard)
s_rates_3d = np.array([0.0, 1.19, 1.4, 0.0, 1.2, 0.0, 1.2, 0.0, 1.2,
                        0.0, 1.4, 0.0, 1.4, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5])
stress_modes_3d = [9, 11, 13, 14, 15]

def compute_feq_3d(rho, ux, uy, uz):
    feq = np.zeros(19)
    u2 = ux**2 + uy**2 + uz**2
    for q in range(19):
        eu = e3d[q, 0]*ux + e3d[q, 1]*uy + e3d[q, 2]*uz
        feq[q] = W3d[q] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u2)
    return feq

# --- Python port of raw_to_central_dH and central_to_raw_dH ---
# (exact copy from test_mrt_cm_comprehensive.py)
def raw_to_central_py(m, ux, uy, uz):
    ux2=ux*ux; uy2=uy*uy; uz2=uz*uz; u2=ux2+uy2+uz2
    uxuy=ux*uy; uxuz=ux*uz; uyuz=uy*uz
    k = np.zeros(19)
    k[0]=m[0]
    k[3]=-ux*m[0]+m[3]; k[5]=-uy*m[0]+m[5]; k[7]=-uz*m[0]+m[7]
    k[1]=19.0*u2*m[0]+m[1]-38.0*ux*m[3]-38.0*uy*m[5]-38.0*uz*m[7]
    k[9]=(2.0*ux2-uy2-uz2)*m[0]-4.0*ux*m[3]+2.0*uy*m[5]+2.0*uz*m[7]+m[9]
    k[11]=(uy2-uz2)*m[0]-2.0*uy*m[5]+2.0*uz*m[7]+m[11]
    k[13]=uxuy*m[0]-uy*m[3]-ux*m[5]+m[13]
    k[14]=uyuz*m[0]-uz*m[5]-uy*m[7]+m[14]
    k[15]=uxuz*m[0]-uz*m[3]-ux*m[7]+m[15]
    k[4]=(-5.0*ux*uy2-5.0*ux*uz2-(24.0/19.0)*ux)*m[0]-(10.0/57.0)*ux*m[1]+(5.0*(uy2+uz2))*m[3]+m[4]+10.0*uxuy*m[5]+10.0*uxuz*m[7]+(5.0/3.0)*ux*m[9]-10.0*uy*m[13]-10.0*uz*m[15]
    k[6]=(-5.0*ux2*uy-5.0*uy*uz2-(24.0/19.0)*uy)*m[0]-(10.0/57.0)*uy*m[1]+10.0*uxuy*m[3]+(5.0*(ux2+uz2))*m[5]+m[6]+10.0*uyuz*m[7]-(5.0/6.0)*uy*m[9]+(5.0/2.0)*uy*m[11]-10.0*ux*m[13]-10.0*uz*m[14]
    k[8]=(-5.0*ux2*uz-5.0*uy2*uz-(24.0/19.0)*uz)*m[0]-(10.0/57.0)*uz*m[1]+10.0*uxuz*m[3]+10.0*uyuz*m[5]+(5.0*(ux2+uy2))*m[7]+m[8]-(5.0/6.0)*uz*m[9]-(5.0/2.0)*uz*m[11]-10.0*uy*m[14]-10.0*ux*m[15]
    k[2]=(21.0*(ux2*uy2+ux2*uz2+uy2*uz2)+(116.0/19.0)*u2)*m[0]+(14.0/19.0)*u2*m[1]+m[2]+(-42.0*ux*uy2-42.0*ux*uz2-(8.0/5.0)*ux)*m[3]-(42.0/5.0)*ux*m[4]+(-42.0*ux2*uy-42.0*uy*uz2-(8.0/5.0)*uy)*m[5]-(42.0/5.0)*uy*m[6]+(-42.0*ux2*uz-42.0*uy2*uz-(8.0/5.0)*uz)*m[7]-(42.0/5.0)*uz*m[8]+(-7.0*ux2+3.5*uy2+3.5*uz2)*m[9]+(-10.5*uy2+10.5*uz2)*m[11]+84.0*uxuy*m[13]+84.0*uyuz*m[14]+84.0*uxuz*m[15]
    k[10]=(3.0*(ux2*uy2+ux2*uz2)-6.0*uy2*uz2-(16.0/19.0)*ux2+(8.0/19.0)*uy2+(8.0/19.0)*uz2)*m[0]+((2.0*ux2-uy2-uz2)/19.0)*m[1]+(-6.0*ux*uy2-6.0*ux*uz2+(16.0/5.0)*ux)*m[3]-(6.0/5.0)*ux*m[4]+(-6.0*ux2*uy+12.0*uy*uz2-(8.0/5.0)*uy)*m[5]+(3.0/5.0)*uy*m[6]+(-6.0*ux2*uz+12.0*uy2*uz-(8.0/5.0)*uz)*m[7]+(3.0/5.0)*uz*m[8]+(-ux2+2.0*uy2+2.0*uz2)*m[9]+m[10]+(3.0*uy2-3.0*uz2)*m[11]+12.0*uxuy*m[13]-24.0*uyuz*m[14]+12.0*uxuz*m[15]+9.0*uy*m[17]-9.0*uz*m[18]
    k[12]=(3.0*ux2*(uy2-uz2)+(8.0/19.0)*(uz2-uy2))*m[0]+(uy2-uz2)/19.0*m[1]-6.0*ux*(uy2-uz2)*m[3]+(2.0/5.0)*uy*(4.0-15.0*ux2)*m[5]-(3.0/5.0)*uy*m[6]+(2.0/5.0)*uz*(15.0*ux2-4.0)*m[7]+(3.0/5.0)*uz*m[8]+(uy2-uz2)*m[9]+3.0*ux2*m[11]+m[12]+12.0*uxuy*m[13]-12.0*uxuz*m[15]-6.0*ux*m[16]+3.0*uy*m[17]+3.0*uz*m[18]
    k[16]=ux*(uz2-uy2)*m[0]+(uy2-uz2)*m[3]+2.0*uxuy*m[5]-2.0*uxuz*m[7]-ux*m[11]-2.0*uy*m[13]+2.0*uz*m[15]+m[16]
    k[17]=uy*(ux2-uz2)*m[0]-2.0*uxuy*m[3]+(uz2-ux2)*m[5]+2.0*uyuz*m[7]+0.5*uy*m[9]+0.5*uy*m[11]+2.0*ux*m[13]-2.0*uz*m[14]+m[17]
    k[18]=uz*(uy2-ux2)*m[0]+2.0*uxuz*m[3]-2.0*uyuz*m[5]+(ux2-uy2)*m[7]-0.5*uz*m[9]+0.5*uz*m[11]+2.0*uy*m[14]-2.0*ux*m[15]+m[18]
    return k

def central_to_raw_py(k, ux, uy, uz):
    ux2=ux*ux; uy2=uy*uy; uz2=uz*uz; u2=ux2+uy2+uz2
    uxuy=ux*uy; uxuz=ux*uz; uyuz=uy*uz
    m = np.zeros(19)
    m[0]=k[0]; m[3]=ux*k[0]+k[3]; m[5]=uy*k[0]+k[5]; m[7]=uz*k[0]+k[7]
    m[1]=19.0*u2*k[0]+k[1]+38.0*ux*k[3]+38.0*uy*k[5]+38.0*uz*k[7]
    m[9]=(2.0*ux2-uy2-uz2)*k[0]+4.0*ux*k[3]-2.0*uy*k[5]-2.0*uz*k[7]+k[9]
    m[11]=(uy2-uz2)*k[0]+2.0*uy*k[5]-2.0*uz*k[7]+k[11]
    m[13]=uxuy*k[0]+uy*k[3]+ux*k[5]+k[13]
    m[14]=uyuz*k[0]+uz*k[5]+uy*k[7]+k[14]
    m[15]=uxuz*k[0]+uz*k[3]+ux*k[7]+k[15]
    m[4]=ux*(95.0*(uy2+uz2)+24.0)/19.0*k[0]+(10.0/57.0)*ux*k[1]+(5.0*(uy2+uz2))*k[3]+k[4]+10.0*uxuy*k[5]+10.0*uxuz*k[7]-(5.0/3.0)*ux*k[9]+10.0*uy*k[13]+10.0*uz*k[15]
    m[6]=uy*(95.0*(ux2+uz2)+24.0)/19.0*k[0]+(10.0/57.0)*uy*k[1]+10.0*uxuy*k[3]+(5.0*(ux2+uz2))*k[5]+k[6]+10.0*uyuz*k[7]+(5.0/6.0)*uy*k[9]-(5.0/2.0)*uy*k[11]+10.0*ux*k[13]+10.0*uz*k[14]
    m[8]=uz*(95.0*(ux2+uy2)+24.0)/19.0*k[0]+(10.0/57.0)*uz*k[1]+10.0*uxuz*k[3]+10.0*uyuz*k[5]+(5.0*(ux2+uy2))*k[7]+k[8]+(5.0/6.0)*uz*k[9]+(5.0/2.0)*uz*k[11]+10.0*uy*k[14]+10.0*ux*k[15]
    m[2]=(21.0*(ux2*uy2+ux2*uz2+uy2*uz2)+(116.0/19.0)*u2)*k[0]+(14.0/19.0)*u2*k[1]+k[2]+(2.0/5.0)*ux*(105.0*(uy2+uz2)+4.0)*k[3]+(42.0/5.0)*ux*k[4]+(2.0/5.0)*uy*(105.0*(ux2+uz2)+4.0)*k[5]+(42.0/5.0)*uy*k[6]+(2.0/5.0)*uz*(105.0*(ux2+uy2)+4.0)*k[7]+(42.0/5.0)*uz*k[8]+(-7.0*ux2+3.5*uy2+3.5*uz2)*k[9]+(-10.5*uy2+10.5*uz2)*k[11]+84.0*uxuy*k[13]+84.0*uyuz*k[14]+84.0*uxuz*k[15]
    m[10]=(3.0*(ux2*uy2+ux2*uz2)-6.0*uy2*uz2-(16.0/19.0)*ux2+(8.0/19.0)*uy2+(8.0/19.0)*uz2)*k[0]+((2.0*ux2-uy2-uz2)/19.0)*k[1]+(2.0/5.0)*ux*(15.0*(uy2+uz2)-8.0)*k[3]+(6.0/5.0)*ux*k[4]+(2.0/5.0)*uy*(15.0*ux2-30.0*uz2+4.0)*k[5]-(3.0/5.0)*uy*k[6]+(2.0/5.0)*uz*(15.0*ux2-30.0*uy2+4.0)*k[7]-(3.0/5.0)*uz*k[8]+(-ux2+2.0*uy2+2.0*uz2)*k[9]+k[10]+(3.0*uy2-3.0*uz2)*k[11]+12.0*uxuy*k[13]-24.0*uyuz*k[14]+12.0*uxuz*k[15]-9.0*uy*k[17]+9.0*uz*k[18]
    m[12]=(3.0*ux2*(uy2-uz2)+(8.0/19.0)*(uz2-uy2))*k[0]+(uy2-uz2)/19.0*k[1]+6.0*ux*(uy2-uz2)*k[3]+(2.0/5.0)*uy*(15.0*ux2-4.0)*k[5]+(3.0/5.0)*uy*k[6]+(2.0/5.0)*uz*(4.0-15.0*ux2)*k[7]-(3.0/5.0)*uz*k[8]+(uy2-uz2)*k[9]+3.0*ux2*k[11]+k[12]+12.0*uxuy*k[13]-12.0*uxuz*k[15]+6.0*ux*k[16]-3.0*uy*k[17]-3.0*uz*k[18]
    m[16]=ux*(uy2-uz2)*k[0]+(uy2-uz2)*k[3]+2.0*uxuy*k[5]-2.0*uxuz*k[7]+ux*k[11]+2.0*uy*k[13]-2.0*uz*k[15]+k[16]
    m[17]=uy*(uz2-ux2)*k[0]-2.0*uxuy*k[3]+(uz2-ux2)*k[5]+2.0*uyuz*k[7]-0.5*uy*k[9]-0.5*uy*k[11]-2.0*ux*k[13]+2.0*uz*k[14]+k[17]
    m[18]=uz*(ux2-uy2)*k[0]+2.0*uxuz*k[3]-2.0*uyuz*k[5]+(ux2-uy2)*k[7]+0.5*uz*k[9]-0.5*uz*k[11]-2.0*uy*k[14]+2.0*ux*k[15]+k[18]
    return m

def build_T3d_matrix(ux, uy, uz):
    T = np.zeros((19, 19))
    for col in range(19):
        m_in = np.zeros(19); m_in[col] = 1.0
        T[:, col] = raw_to_central_py(m_in, ux, uy, uz)
    return T

def build_T3d_inv_matrix(ux, uy, uz):
    Ti = np.zeros((19, 19))
    for col in range(19):
        k_in = np.zeros(19); k_in[col] = 1.0
        Ti[:, col] = central_to_raw_py(k_in, ux, uy, uz)
    return Ti

def mrt_rm_3d(f, feq, s_visc):
    m_neq = M3d @ (f - feq)
    S = np.diag(s_rates_3d.copy())
    for idx in stress_modes_3d:
        S[idx, idx] = s_visc
    return f - M3d_inv @ (S @ m_neq)

def mrt_cm_3d(f, feq, s_visc, ux, uy, uz):
    m_neq = M3d @ (f - feq)
    k_neq = raw_to_central_py(m_neq, ux, uy, uz)
    S = np.diag(s_rates_3d.copy())
    for idx in stress_modes_3d:
        S[idx, idx] = s_visc
    dm = central_to_raw_py(S @ k_neq, ux, uy, uz)
    return f - M3d_inv @ dm


# ============================================================================
# TEST INFRASTRUCTURE
# ============================================================================
n_pass = 0
n_total = 0
n_fail = 0

def check(cond, msg, err_val):
    global n_pass, n_total, n_fail
    n_total += 1
    if cond:
        n_pass += 1
        print(f"    ✓ {msg} (err = {err_val:.2e})")
    else:
        n_fail += 1
        print(f"    ✗ FAIL: {msg} (err = {err_val:.2e})")

rng = np.random.default_rng(2024)


# ============================================================================
# TEST 1: D2Q9 N matrix matches Fei & Luo Eq.(6) exactly
# ============================================================================
print("=" * 72)
print("TEST 1: D2Q9 N matrix matches Fei & Luo Eq.(6)")
print("  Verify our build_N_fei_luo() against brute-force central moment")
print("  definition k̃_mn = Σ f_i (e_ix - ux)^m (e_iy - uy)^n")
print("=" * 72)

ux_t, uy_t = 0.12, -0.08

# Ground truth: compute N by definition
# raw moments T = M·f, central moments T̃ = N·T = N·M·f
# Central moments directly: k̃_mn = Σ f_i (e_ix - ux)^m (e_iy - uy)^n
# This means T̃ = M(u)·f where M(u) evaluates at shifted velocities
# So N = M(u)·M⁻¹

# Build M(u): same polynomial structure as M but at (e - u)
def build_M2d_shifted(ux, uy):
    """M evaluated at (e_i - u), i.e. the Fei & Luo moment basis at shifted velocities."""
    Ms = np.zeros((9, 9))
    for j in range(9):
        vx = e2d[j, 0] - ux
        vy = e2d[j, 1] - uy
        # Fei & Luo moment ordering: k00, k10, k01, k20+k02, k20-k02, k11, k21, k12, k22
        Ms[0, j] = 1.0                  # k00
        Ms[1, j] = vx                   # k10
        Ms[2, j] = vy                   # k01
        Ms[3, j] = vx**2 + vy**2        # k20 + k02
        Ms[4, j] = vx**2 - vy**2        # k20 - k02
        Ms[5, j] = vx * vy              # k11
        Ms[6, j] = vx**2 * vy           # k21
        Ms[7, j] = vx * vy**2           # k12
        Ms[8, j] = vx**2 * vy**2        # k22
    return Ms

# Build M(0): unshifted (same polynomial structure, velocities = e_i)
M2d_poly_0 = build_M2d_shifted(0.0, 0.0)
M2d_poly_u = build_M2d_shifted(ux_t, uy_t)

# Ground truth: N_gt = M(u)·M(0)⁻¹
N_gt = M2d_poly_u @ np.linalg.inv(M2d_poly_0)

# Our implementation
N_ours = build_N_fei_luo(ux_t, uy_t)

err1 = np.max(np.abs(N_ours - N_gt))
check(err1 < 1e-13, "N matches Eq.(6) via brute-force M(u)·M(0)⁻¹", err1)


# ============================================================================
# TEST 2: D2Q9 N⁻¹ matches Fei & Luo Eq.(A3) exactly
# ============================================================================
print()
print("=" * 72)
print("TEST 2: D2Q9 N⁻¹ matches Fei & Luo Eq.(A3)")
print("  Verify build_Ninv_fei_luo() = N(-u) = inverse of N(u)")
print("=" * 72)

Ni_ours = build_Ninv_fei_luo(ux_t, uy_t)
Ni_gt = np.linalg.inv(N_ours)  # numerical inverse of our N

err2 = np.max(np.abs(Ni_ours - Ni_gt))
check(err2 < 1e-13, "N⁻¹(u) = numpy.inv(N(u))", err2)


# ============================================================================
# TEST 3: D2Q9 N·N⁻¹ = I (round-trip consistency)
# ============================================================================
print()
print("=" * 72)
print("TEST 3: D2Q9 round-trip N(u)·N⁻¹(u) = I")
print("  Also verify N(u)·N(-u) = I (Fei & Luo: flip sign ↔ inversion)")
print("=" * 72)

prod1 = N_ours @ Ni_ours
err3a = np.max(np.abs(prod1 - np.eye(9)))
check(err3a < 1e-13, "N(u) · N⁻¹(u) = I", err3a)

N_neg = build_N_fei_luo(-ux_t, -uy_t)
prod2 = N_ours @ N_neg
err3b = np.max(np.abs(prod2 - np.eye(9)))
check(err3b < 1e-13, "N(u) · N(-u) = I (sign-flip inversion)", err3b)


# ============================================================================
# TEST 4: D2Q9 N=I degeneracy → CLBM = MRT-RM
# (Fei & Luo key result: "when the shift matrix is a unit matrix,
#  the CLBM degrades into an MRT LBM")
# ============================================================================
print()
print("=" * 72)
print("TEST 4: D2Q9 N=I → CLBM degrades to MRT-RM (Fei & Luo key result)")
print("  When u=0, N(0) = I, so collision in central moments = raw moments")
print("=" * 72)

N_zero = build_N_fei_luo(0.0, 0.0)
err4a = np.max(np.abs(N_zero - np.eye(9)))
check(err4a < 1e-15, "N(u=0) = I (shift matrix is identity at rest)", err4a)

# Full collision comparison at u=0
rho4 = 1.0
feq4 = compute_feq_2d(rho4, 0.0, 0.0)
f4 = feq4 + 0.005 * rng.standard_normal(9)
f4 *= rho4 / np.sum(f4)

# D2Q9 relaxation
s2d = np.array([0.0, 0.0, 0.0, 1.1, 1.5, 1.5, 1.2, 1.2, 1.4])
S2d = np.diag(s2d)

# MRT-RM: f* = f - M⁻¹·S·M·(f-feq)
f_rm_2d = f4 - M2d_inv @ (S2d @ (M2d @ (f4 - feq4)))

# CLBM at u=0: f* = f - M⁻¹·N⁻¹·S·N·M·(f-feq) = f - M⁻¹·I·S·I·M·(f-feq)
N0 = build_N_fei_luo(0.0, 0.0)
N0i = build_Ninv_fei_luo(0.0, 0.0)
f_cm_2d = f4 - M2d_inv @ (N0i @ (S2d @ (N0 @ (M2d @ (f4 - feq4)))))

err4b = np.max(np.abs(f_rm_2d - f_cm_2d))
check(err4b < 1e-15, "CLBM(u=0) = MRT-RM collision (identical)", err4b)


# ============================================================================
# TEST 5: D2Q9 Central moment equilibrium = Fei & Luo Eq.(8)
# T̃_eq = [ρ, 0, 0, 2ρc_s², 0, 0, 0, 0, ρc_s⁴]
# Note: Eq.(8) uses continuous MB equilibrium. Standard LBM feq matches
# continuous MB up to 2nd order; 3rd+ order have O(u²) residual.
# ============================================================================
print()
print("=" * 72)
print("TEST 5: D2Q9 central moment feq vs Fei & Luo Eq.(8)")
print("  T̃_eq = [ρ, 0, 0, 2ρc_s², 0, 0, 0, 0, ρc_s⁴]")
print("  Low-order (0-2nd) exact; higher-order ≈ O(Ma²) deviation")
print("=" * 72)

rho5 = 1.3
ux5, uy5 = 0.15, -0.10
cs2 = 1.0/3.0  # c_s² = 1/3

feq5 = compute_feq_2d(rho5, ux5, uy5)

# Central moments: T̃_eq = N · M · feq
N5 = build_N_fei_luo(ux5, uy5)
T_eq = N5 @ (M2d @ feq5)

# Fei & Luo Eq.(8): continuous MB
T_eq_analytic = np.array([rho5, 0, 0, 2*rho5*cs2, 0, 0, 0, 0, rho5*cs2**2])

# Low-order (0-5): ρ, jx, jy, k20+k02, k20-k02, k11
err5_low = np.max(np.abs(T_eq[:6] - T_eq_analytic[:6]))
check(err5_low < 1e-14, "Low-order κ_eq (0-2nd) match Eq.(8) exactly", err5_low)

# Higher-order (6-8): k21, k12, k22 have O(u²) deviation from cont. MB
err5_high = np.max(np.abs(T_eq[6:] - T_eq_analytic[6:]))
Ma2 = ux5**2 + uy5**2  # ~ 0.0325
check(err5_high < Ma2, f"High-order κ_eq deviation < Ma² ≈ {Ma2:.4f}", err5_high)


# ============================================================================
# TEST 6: D3Q19 T(u)·T(-u) = I (round-trip)
# Same mathematical structure as D2Q9 (Fei & Luo framework generalized)
# ============================================================================
print()
print("=" * 72)
print("TEST 6: D3Q19 T(u)·T(-u) = I (round-trip)")
print("  Same structure as D2Q9 N(u)·N(-u)=I, generalized to 3D")
print("=" * 72)

u6 = 0.1 * rng.standard_normal(3)
T6 = build_T3d_matrix(*u6)
T6_neg = build_T3d_matrix(*(-u6))
prod6 = T6 @ T6_neg
err6 = np.max(np.abs(prod6 - np.eye(19)))
check(err6 < 1e-12, "T(u)·T(-u) = I₁₉ₓ₁₉", err6)


# ============================================================================
# TEST 7: D3Q19 T(u+v) = T(u)·T(v) (group morphism)
# Fei & Luo framework: shift operators form a group under composition
# ============================================================================
print()
print("=" * 72)
print("TEST 7: D3Q19 T(u+v) = T(u)·T(v) (group morphism)")
print("  Fei & Luo: shift operators form a group (Dubois Prop. 1.1)")
print("=" * 72)

u7a = 0.1 * rng.standard_normal(3)
u7b = 0.1 * rng.standard_normal(3)
T7a = build_T3d_matrix(*u7a)
T7b = build_T3d_matrix(*u7b)
T7ab = build_T3d_matrix(*(u7a + u7b))
err7 = np.max(np.abs(T7ab - T7a @ T7b))
check(err7 < 1e-12, "T(u+v) = T(u)·T(v)", err7)


# ============================================================================
# TEST 8: D3Q19 N=I degeneracy → CLBM = MRT-RM
# (Fei & Luo generalized to 3D: "when shift matrix is unit matrix...")
# ============================================================================
print()
print("=" * 72)
print("TEST 8: D3Q19 N=I degeneracy → CLBM = MRT-RM")
print("  Fei & Luo: 'CLBM degrades into MRT LBM' when N=I (u=0)")
print("=" * 72)

# Verify T(0) = I
T_zero = build_T3d_matrix(0.0, 0.0, 0.0)
err8a = np.max(np.abs(T_zero - np.eye(19)))
check(err8a < 1e-15, "T(u=0) = I₁₉ₓ₁₉ (shift is identity at rest)", err8a)

# Full collision comparison at u=0
rho8 = 1.0
feq8 = compute_feq_3d(rho8, 0.0, 0.0, 0.0)
f8 = feq8 + 0.005 * rng.standard_normal(19)
f8 *= rho8 / np.sum(f8)

s_visc8 = 1.0 / 0.55
f_rm8 = mrt_rm_3d(f8.copy(), feq8, s_visc8)
f_cm8 = mrt_cm_3d(f8.copy(), feq8, s_visc8, 0.0, 0.0, 0.0)
err8b = np.max(np.abs(f_rm8 - f_cm8))
check(err8b < 1e-14, "CLBM(u=0) = MRT-RM collision (3D identical)", err8b)


# ============================================================================
# TEST 9: D3Q19 Central moment feq: low-order modes velocity-independent
# Generalized Fei & Luo Eq.(8) to D3Q19:
#   Standard LBM feq is O(u²) accurate, so ONLY 0th-2nd order central
#   moments are velocity-independent (Galilean invariant).
#   Higher-order modes (3rd, 4th) have O(u²) residual velocity dependence.
# ============================================================================
print()
print("=" * 72)
print("TEST 9: D3Q19 central moment feq: low-order velocity-independence")
print("  LBM feq is O(u²), so 0th-2nd order κ_eq are velocity-invariant")
print("  Higher orders have O(u²) residual (expected, not a bug)")
print("=" * 72)

rho9 = 1.2
u9a = np.array([0.08, -0.05, 0.03])
u9b = np.array([-0.12, 0.10, -0.07])

feq9a = compute_feq_3d(rho9, *u9a)
feq9b = compute_feq_3d(rho9, *u9b)

m_feq9a = M3d @ feq9a
m_feq9b = M3d @ feq9b
k_feq9a = raw_to_central_py(m_feq9a, *u9a)
k_feq9b = raw_to_central_py(m_feq9b, *u9b)

# 0th order: density (conserved)
check(abs(k_feq9a[0] - rho9) < 1e-14, f"κ₀ = ρ = {rho9}", abs(k_feq9a[0] - rho9))

# 1st order: momentum (conserved → κ = 0 in central moment space)
mom_modes = [3, 5, 7]
err9_mom = max(abs(k_feq9a[i]) for i in mom_modes)
check(err9_mom < 1e-14, "κ₃,κ₅,κ₇ = 0 (central momentum = 0)", err9_mom)

# 2nd order stress modes: velocity-independent between u_a and u_b
#   d'Humières modes 9,11,13,14,15 are 2nd order → should match
stress_2nd = [9, 11, 13, 14, 15]
err9_stress = max(abs(k_feq9a[i] - k_feq9b[i]) for i in stress_2nd)
check(err9_stress < 1e-14,
      "κ_stress(u₁) = κ_stress(u₂) (2nd order invariant)", err9_stress)

# 2nd order energy mode (m1): also velocity-independent
err9_energy = abs(k_feq9a[1] - k_feq9b[1])
check(err9_energy < 1e-13,
      "κ_energy(u₁) = κ_energy(u₂) (2nd order invariant)", err9_energy)


# ============================================================================
# TEST 10: D3Q19 CLBM Eq.(7)+(9) = our MRT-CM collision
# Full Fei & Luo collision pipeline vs our implementation
#   Eq.(7): |T̃*⟩ = (I-S)·N·M·f + S·N·M·feq  (central moment collision)
#   Eq.(9): |f*⟩ = M⁻¹·N⁻¹·|T̃*⟩             (back to distribution space)
# ============================================================================
print()
print("=" * 72)
print("TEST 10: D3Q19 CLBM collision Eq.(7)+(9) = our MRT-CM")
print("  Full Fei & Luo pipeline: collision in central moments → f*")
print("=" * 72)

rho10 = 1.15
ux10, uy10, uz10 = 0.06, -0.04, 0.03
s_visc10 = 1.0 / 0.6

feq10 = compute_feq_3d(rho10, ux10, uy10, uz10)
f10 = feq10 + 0.008 * rng.standard_normal(19)
f10 *= rho10 / np.sum(f10)

# Build full S matrix
S10 = np.diag(s_rates_3d.copy())
for idx in stress_modes_3d:
    S10[idx, idx] = s_visc10

# --- Method A: Fei & Luo Eq.(7)+(9) explicitly ---
T10 = build_T3d_matrix(ux10, uy10, uz10)
T10i = build_T3d_inv_matrix(ux10, uy10, uz10)

# Eq.(7): |T̃*⟩ = (I-S)·N·M·f + S·N·M·feq
#        = (I-S)·T·M·f + S·T·M·feq
T_tilde_f = T10 @ M3d @ f10           # central moments of f
T_tilde_feq = T10 @ M3d @ feq10       # central moments of feq
T_tilde_star = (np.eye(19) - S10) @ T_tilde_f + S10 @ T_tilde_feq

# Eq.(9): |f*⟩ = M⁻¹·N⁻¹·|T̃*⟩
f_star_fei_luo = M3d_inv @ (T10i @ T_tilde_star)

# --- Method B: our mrt_cm_3d ---
f_star_ours = mrt_cm_3d(f10.copy(), feq10, s_visc10, ux10, uy10, uz10)

err10 = np.max(np.abs(f_star_fei_luo - f_star_ours))
check(err10 < 1e-13, "Fei & Luo Eq.(7)+(9) = our MRT-CM collision", err10)

# Also verify algebraically: the above should equal f - M⁻¹·T⁻¹·S·T·M·(f-feq)
f_star_algebra = f10 - M3d_inv @ (T10i @ (S10 @ (T10 @ (M3d @ (f10 - feq10)))))
err10b = np.max(np.abs(f_star_fei_luo - f_star_algebra))
check(err10b < 1e-13, "Expanded form: f - M⁻¹T⁻¹STM(f-feq)", err10b)


# ============================================================================
# SUMMARY
# ============================================================================
print()
print("=" * 72)
print(f"SUMMARY: {n_pass}/{n_total} checks passed across 10 tests")
print("=" * 72)
if n_fail == 0:
    print("  >>> ALL CHECKS PASSED <<<")
    print()
    print("  Correspondence verified:")
    print("    Fei & Luo N (D2Q9)    ↔  Our T(u) (D3Q19)  [shift matrix]")
    print("    Fei & Luo N⁻¹         ↔  Our T⁻¹(u)         [inverse shift]")
    print("    Fei & Luo Eq.(7)+(9)  ↔  f* = f - M⁻¹T⁻¹STM(f-feq)")
    print("    Fei & Luo Eq.(8)      ↔  velocity-independent κ_eq")
    print("    N=I → CLBM=MRT-RM    (both D2Q9 and D3Q19)")
else:
    print(f"  >>> {n_fail} CHECKS FAILED <<<")

sys.exit(0 if n_fail == 0 else 1)
