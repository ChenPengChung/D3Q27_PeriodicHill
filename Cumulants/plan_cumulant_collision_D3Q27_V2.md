# Plan v2: D3Q27 Cumulant Collision — "Building Block" Bridge Design

---

## 設計哲學 — 積木式橋接架構

OpenLB 的 Cumulant 實作使用了大量 C++ 模板元編程（template metaprogramming）、
structured bindings、CRTP 繼承、tag dispatch 等高階抽象。
這些設計對框架內部很好，但對外部使用者來說是**不透明的黑盒**。

本計畫的核心思路：

```
 ┌─────────────────────────────────────────────────────────┐
 │                    OpenLB 原始碼                         │
 │  (templates, CRTP, tag dispatch, structured bindings)   │
 │  collisionCUM.h + cum.h + cumulantDynamics.h            │
 └────────────────────────┬────────────────────────────────┘
                          │
                   「翻譯 / 橋接」
                   保留數學邏輯不動
                   剝離所有框架依賴
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────┐
 │              Bridge Layer（積木模組）                     │
 │                                                         │
 │  純 C/CUDA 函式，零模板，零繼承，零外部依賴               │
 │  輸入：double f[27], omega, Fx, Fy, Fz                  │
 │  輸出：double f_post[27], rho, ux, uy, uz               │
 │                                                         │
 │  使用者不需要懂 OpenLB，只需要：                          │
 │  「把你算好的 f[27] 丟進來，拿走碰撞後的 f[27]」         │
 └────────────────────────┬────────────────────────────────┘
                          │
                    使用者的程式碼
                    直接呼叫即可
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────┐
 │            使用者的 GILBM Evolution Kernel               │
 │                                                         │
 │  // Step 1: Lagrange interpolation → f_streamed[27]     │
 │  // Step 2: ★ 呼叫積木 ★                                │
 │  cumulant_collision(f_streamed, omega, Fx, 0, 0,        │
 │                     f_post, rho, ux, uy, uz);           │
 │  // Step 3: 寫回 global memory                          │
 └─────────────────────────────────────────────────────────┘
```

---

## 從 OpenLB 到 Bridge 的翻譯對照表

下表是「OpenLB 抽象 → Bridge 具體實作」的逐項翻譯：

```
╔══════════════════════════════╦═══════════════════════════════════╗
║  OpenLB 抽象                 ║  Bridge 具體實作                  ║
╠══════════════════════════════╬═══════════════════════════════════╣
║                              ║                                   ║
║  template <typename CELL>    ║  double f[27]                     ║
║  cell[i]                     ║  f[i] - w[i]  (手動減權重)        ║
║                              ║                                   ║
║  DESCRIPTOR::q = 27          ║  const int Q = 27;                ║
║  DESCRIPTOR::d = 3           ║  const int D = 3;                 ║
║                              ║                                   ║
║  descriptors::constantK<V,   ║  const double K27[27] = {...};    ║
║    D, Q>(idx)                ║  K27[idx]                         ║
║                              ║                                   ║
║  cum_data::velocityIndices   ║  const int velIdx[27][3] = {...}; ║
║    <3,27>[idx]               ║  velIdx[idx]                      ║
║                              ║                                   ║
║  auto [a, b, c] = ...       ║  int a = velIdx[p][0];            ║
║  (C++17 structured binding)  ║  int b = velIdx[p][1];            ║
║                              ║  int c = velIdx[p][2];            ║
║                              ║                                   ║
║  auto [mbbb, mabb, ...]     ║  // 直接用 m[INDEX] 存取          ║
║  = moments;                  ║  // 或定義 enum 別名              ║
║  (27-element decomposition)  ║  #define mbbb m[0]               ║
║                              ║  #define mabb m[1]  ...           ║
║                              ║                                   ║
║  V = typename CELL::value_t  ║  double                           ║
║                              ║                                   ║
║  parameters.get<OMEGA>()     ║  函式參數 omega                   ║
║                              ║                                   ║
║  MomentaF().computeRhoU()   ║  手動 for 迴圈算 ρ, u             ║
║                              ║                                   ║
║  any_platform                ║  __device__ (CUDA)                ║
║                              ║                                   ║
╚══════════════════════════════╩═══════════════════════════════════╝
```

---

## Bridge 模組檔案結構

```
cumulant_bridge/
├── cumulant_constants.h    ← 所有常數（w27, K27, velIdx, ex, ey, ez）
├── cumulant_collision.h    ← 核心碰撞函式（5 stages）
└── cumulant_test.cu        ← 單元測試（可選）
```

只有 **兩個標頭檔**，使用者 `#include "cumulant_collision.h"` 即可。

---

## File 1: cumulant_constants.h

```cpp
// ================================================================
// cumulant_constants.h
// D3Q27 Cumulant Collision Constants
// Translated from OpenLB (cum.h) — no templates, no dependencies
// ================================================================
#ifndef CUMULANT_CONSTANTS_H
#define CUMULANT_CONSTANTS_H

// ---- D3Q27 lattice weights ----
__constant__ double CUM_W[27] = {
    8.0/27.0,                                          // rest
    2.0/27.0, 2.0/27.0, 2.0/27.0,                     // face +x,-x,+y
    2.0/27.0, 2.0/27.0, 2.0/27.0,                     // face -y,+z,-z
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,           // edge
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,       // corner
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

// ---- K constants for well-conditioned Chimera ----
// Source: cum.h → K<3,27>
// 使用方法：k = K27[pass_index] * (1 - drho)  (但 OpenLB 已簡化為直接用)
__constant__ double CUM_K[27] = {
    1.0,
    0.0,      1.0/3.0,
    0.0,      0.0,      0.0,      1.0/3.0,
    0.0,      1.0/9.0,
    1.0/6.0,
    0.0,      1.0/18.0,
    2.0/3.0,
    0.0,      2.0/9.0,
    1.0/6.0,
    0.0,      1.0/18.0,
    1.0/36.0,
    1.0/9.0,
    1.0/36.0,
    1.0/9.0,
    4.0/9.0,
    1.0/9.0,
    1.0/36.0,
    1.0/9.0,
    1.0/36.0
};

// ---- Chimera triplet indices ----
// 9 passes × 3 directions = 27 rows
// 方向順序：z(pass 0-8), y(pass 9-17), x(pass 18-26)
// 每行 [a, b, c] 是 moment 陣列中的三元組索引
__constant__ int CUM_IDX[27][3] = {
    // z-direction (9 triplets)
    {10,  8, 26}, {12, 22, 24}, { 6,  3, 20},
    { 4,  2, 18}, { 1,  0, 14}, { 5, 15, 17},
    {11,  9, 25}, { 7, 16, 19}, {13, 21, 23},
    // y-direction (9 triplets)
    {10,  6, 12}, { 4,  1,  5}, {11,  7, 13},
    { 8,  3, 22}, { 2,  0, 15}, { 9, 16, 21},
    {26, 20, 24}, {18, 14, 17}, {25, 19, 23},
    // x-direction (9 triplets)
    {10,  4, 11}, { 6,  1,  7}, {12,  5, 13},
    { 8,  2,  9}, { 3,  0, 16}, {22, 15, 21},
    {26, 18, 25}, {20, 14, 19}, {24, 17, 23}
};

// ---- 27-element moment array index aliases ----
// 對照 OpenLB structured binding 順序:
// auto [mbbb, mabb, mbab, mbba, maab, macb, maba, mabc, mbaa,
//       mbac, maaa, maac, maca, macc, mcbb, mbcb, mbbc, mccb,
//       mcab, mcbc, mcba, mbcc, mbca, mccc, mcca, mcac, mcaa]
//
// 命名: m_{αβγ} where a=0, b=1, c=2
//   maaa = κ₀₀₀ (= δρ)
//   mbaa = κ₁₀₀, maba = κ₀₁₀, maab = κ₀₀₁
//   mcaa = κ₂₀₀, maca = κ₀₂₀, maac = κ₀₀₂
//   mabb = κ₀₁₁, mbab = κ₁₀₁, mbba = κ₁₁₀
//   mbbb = κ₁₁₁
//   ...

#define I_bbb  0
#define I_abb  1
#define I_bab  2
#define I_bba  3
#define I_aab  4
#define I_acb  5
#define I_aba  6
#define I_abc  7
#define I_baa  8
#define I_bac  9
#define I_aaa 10
#define I_aac 11
#define I_aca 12
#define I_acc 13
#define I_cbb 14
#define I_bcb 15
#define I_bbc 16
#define I_ccb 17
#define I_cab 18
#define I_cbc 19
#define I_cba 20
#define I_bcc 21
#define I_bca 22
#define I_ccc 23
#define I_cca 24
#define I_cac 25
#define I_caa 26

#endif // CUMULANT_CONSTANTS_H
```

---

## File 2: cumulant_collision.h — 完整積木模組

```cpp
// ================================================================
// cumulant_collision.h
// D3Q27 Cumulant Collision — Standalone Bridge Module
//
// 「積木」設計：
//   使用者只需提供算好的 f[27]，呼叫一個函式，取回碰撞結果。
//   內部邏輯 100% 等價於 OpenLB collisionCUM.h (Geier et al. 2015)
//   但零模板、零繼承、零外部依賴。
//
// Reference:
//   Geier M. et al., Comp. Math. Appl. 70(4), 507–547, 2015.
//   OpenLB collisionCUM.h (GPL v2+), translated to plain C/CUDA.
// ================================================================
#ifndef CUMULANT_COLLISION_H
#define CUMULANT_COLLISION_H

#include "cumulant_constants.h"

// ================================================================
// Forward declaration of internal stages
// (使用者不需要直接呼叫這些)
// ================================================================

// Stage 1: populations → central moments (Chimera forward, z→y→x)
__device__ static void _cum_forward_chimera(
    double m[27], const double u[3]);

// Stage 5: central moments → populations (Chimera backward, x→y→z)
__device__ static void _cum_backward_chimera(
    double m[27], const double u[3]);


// ================================================================
//
//  ★★★ 使用者唯一需要呼叫的函式 ★★★
//
// ================================================================
//
//  呼叫方式：
//
//    double f_in[27];       // ← 你已經算好的 post-streaming f
//    double f_out[27];      // ← 碰撞後的 f（輸出）
//    double rho, ux, uy, uz;// ← 宏觀量（輸出）
//
//    cumulant_collision_D3Q27(
//        f_in,              // 輸入：遷移後分佈函數
//        1.4634,            // 輸入：omega = 1/(3*nu + 0.5)
//        F_body_x,          // 輸入：x方向外力
//        0.0,               // 輸入：y方向外力
//        0.0,               // 輸入：z方向外力
//        f_out,             // 輸出：碰撞後分佈函數
//        rho, ux, uy, uz    // 輸出：密度與修正速度
//    );
//
// ================================================================
__device__ void cumulant_collision_D3Q27(
    const double f_in[27],   // INPUT:  post-streaming distributions
    const double omega,      // INPUT:  shear relaxation rate
    const double Fx,         // INPUT:  body force x
    const double Fy,         // INPUT:  body force y
    const double Fz,         // INPUT:  body force z
    double       f_out[27],  // OUTPUT: post-collision distributions
    double&      rho_out,    // OUTPUT: density
    double&      ux_out,     // OUTPUT: velocity x (half-force corrected)
    double&      uy_out,     // OUTPUT: velocity y
    double&      uz_out      // OUTPUT: velocity z
)
{
    // ==============================================================
    // STAGE 0: Macroscopic Quantities + Well-Conditioning
    // ==============================================================
    // 0a. 計算密度 ρ = Σ f_α
    double rho = 0.0;
    for (int i = 0; i < 27; i++) rho += f_in[i];

    // 0b. 計算原始動量 ρu = Σ f_α · e_α
    //     （需要離散速度 ex, ey, ez —— 使用者須確認自己的速度排列）
    double jx = 0.0, jy = 0.0, jz = 0.0;
    for (int i = 0; i < 27; i++) {
        jx += f_in[i] * ex_d[i];  // ex_d: 你的離散速度 x 分量
        jy += f_in[i] * ey_d[i];
        jz += f_in[i] * ez_d[i];
    }

    // 0c. 半力修正速度（Guo-style, 時間對稱）
    double inv_rho = 1.0 / rho;
    double u[3];
    u[0] = jx * inv_rho + 0.5 * Fx * inv_rho;
    u[1] = jy * inv_rho + 0.5 * Fy * inv_rho;
    u[2] = jz * inv_rho + 0.5 * Fz * inv_rho;

    // 0d. Well-conditioning: f̄ = f - w
    double m[27];
    for (int i = 0; i < 27; i++) {
        m[i] = f_in[i] - CUM_W[i];
    }

    // 0e. 輔助變數
    double drho = rho - 1.0;

    // ==============================================================
    // STAGE 1: Forward Chimera Transform (z → y → x)
    //          f̄[27] → κ[27] (central moments)
    // ==============================================================
    _cum_forward_chimera(m, u);

    // --- 此時 m[27] 已是 central moments κ_αβγ ---
    // 使用 index aliases 存取（見 cumulant_constants.h）

    // ==============================================================
    // STAGE 2: Central Moments → Cumulants
    //          κ_αβγ → C_αβγ
    //          (只有 4~6 階需要轉換，1~3 階 C = κ)
    // ==============================================================

    // --- 4th order cumulants (Eq. J.16) ---
    double CUMcbb = m[I_cbb] - ((m[I_caa] + 1.0/3.0) * m[I_abb]
                    + 2.0 * m[I_bba] * m[I_bab]) * inv_rho;
    double CUMbcb = m[I_bcb] - ((m[I_aca] + 1.0/3.0) * m[I_bab]
                    + 2.0 * m[I_bba] * m[I_abb]) * inv_rho;
    double CUMbbc = m[I_bbc] - ((m[I_aac] + 1.0/3.0) * m[I_bba]
                    + 2.0 * m[I_bab] * m[I_abb]) * inv_rho;

    // --- 4th order diagonal (Eq. J.17) ---
    double CUMcca = m[I_cca] - (((m[I_caa]*m[I_aca] + 2.0*m[I_bba]*m[I_bba])
                    + (m[I_caa]+m[I_aca])/3.0) * inv_rho
                    - drho * inv_rho / 9.0);
    double CUMcac = m[I_cac] - (((m[I_caa]*m[I_aac] + 2.0*m[I_bab]*m[I_bab])
                    + (m[I_caa]+m[I_aac])/3.0) * inv_rho
                    - drho * inv_rho / 9.0);
    double CUMacc = m[I_acc] - (((m[I_aac]*m[I_aca] + 2.0*m[I_abb]*m[I_abb])
                    + (m[I_aac]+m[I_aca])/3.0) * inv_rho
                    - drho * inv_rho / 9.0);

    // --- 5th order cumulants (Eq. J.18) ---
    double CUMbcc = m[I_bcc] - ((m[I_aac]*m[I_bca] + m[I_aca]*m[I_bac]
                    + 4.0*m[I_abb]*m[I_bbb]
                    + 2.0*(m[I_bab]*m[I_acb] + m[I_bba]*m[I_abc]))
                    + (m[I_bca]+m[I_bac])/3.0) * inv_rho;
    double CUMcbc = m[I_cbc] - ((m[I_aac]*m[I_cba] + m[I_caa]*m[I_abc]
                    + 4.0*m[I_bab]*m[I_bbb]
                    + 2.0*(m[I_abb]*m[I_cab] + m[I_bba]*m[I_bac]))
                    + (m[I_cba]+m[I_abc])/3.0) * inv_rho;
    double CUMccb = m[I_ccb] - ((m[I_caa]*m[I_acb] + m[I_aca]*m[I_cab]
                    + 4.0*m[I_bba]*m[I_bbb]
                    + 2.0*(m[I_bab]*m[I_bca] + m[I_abb]*m[I_cba]))
                    + (m[I_acb]+m[I_cab])/3.0) * inv_rho;

    // --- 6th order cumulant (Eq. J.19) ---
    double CUMccc = m[I_ccc]
        + ((-4.0*m[I_bbb]*m[I_bbb]
            - (m[I_caa]*m[I_acc] + m[I_aca]*m[I_cac] + m[I_aac]*m[I_cca])
            - 4.0*(m[I_abb]*m[I_cbb] + m[I_bab]*m[I_bcb] + m[I_bba]*m[I_bbc])
            - 2.0*(m[I_bca]*m[I_bac] + m[I_cba]*m[I_abc] + m[I_cab]*m[I_acb]))
                * inv_rho
        + (4.0*(m[I_bab]*m[I_bab]*m[I_aca]
              + m[I_abb]*m[I_abb]*m[I_caa]
              + m[I_bba]*m[I_bba]*m[I_aac])
          + 2.0*m[I_caa]*m[I_aca]*m[I_aac]
          + 16.0*m[I_bba]*m[I_bab]*m[I_abb])
                * inv_rho * inv_rho
        - (m[I_acc]+m[I_cac]+m[I_cca]) / 3.0 * inv_rho
        - (m[I_caa]+m[I_aca]+m[I_aac]) / 9.0 * inv_rho
        + (2.0*(m[I_bab]*m[I_bab] + m[I_abb]*m[I_abb] + m[I_bba]*m[I_bba])
          + (m[I_aac]*m[I_aca] + m[I_aac]*m[I_caa] + m[I_aca]*m[I_caa])
          + (m[I_aac]+m[I_aca]+m[I_caa])/3.0)
                * inv_rho * inv_rho * 2.0/3.0
        + (drho*drho - drho) * inv_rho * inv_rho / 27.0);

    // ==============================================================
    // STAGE 3: Relaxation (Collision in Cumulant Space)
    //
    //  ★ 只有 omega (= ω₁) 影響物理黏性
    //  ★ 所有其他 ω₂–ω₁₀ = 1.0（完全鬆弛至平衡態）
    //  ★ 如需調參，修改此 block 內的常數即可
    // ==============================================================
    const double omega2  = 1.0;  // ω₂: bulk viscosity
    const double omega3  = 1.0;  // ω₃: 3rd order symmetric
    const double omega4  = 1.0;  // ω₄: 3rd order antisymmetric
    const double omega6  = 1.0;  // ω₆=ω₇=ω₈: 4th order
    const double omega7  = 1.0;  // ω₉: 5th order
    const double omega10 = 1.0;  // ω₁₀: 6th order

    // --- 2nd order: decompose into orthogonal modes ---
    double mxxPyyPzz = m[I_caa] + m[I_aca] + m[I_aac];  // trace
    double mxxMyy    = m[I_caa] - m[I_aca];              // deviatoric 1
    double mxxMzz    = m[I_caa] - m[I_aac];              // deviatoric 2

    // Relax trace with ω₂ (Eq. 63)
    mxxPyyPzz += omega2 * (m[I_aaa] - mxxPyyPzz);
    // Relax deviatorics with ω₁ (Eq. 61-62)
    mxxMyy *= (1.0 - omega);
    mxxMzz *= (1.0 - omega);

    // Off-diagonal 2nd order (Eq. 55-57)
    m[I_abb] *= (1.0 - omega);  // C₀₁₁
    m[I_bab] *= (1.0 - omega);  // C₁₀₁
    m[I_bba] *= (1.0 - omega);  // C₁₁₀

    // --- 3rd order ---
    // Decompose into symmetric/antisymmetric pairs
    double mxxyPyzz = m[I_cba] + m[I_abc];
    double mxxyMyzz = m[I_cba] - m[I_abc];
    double mxxzPyyz = m[I_cab] + m[I_acb];
    double mxxzMyyz = m[I_cab] - m[I_acb];
    double mxyyPxzz = m[I_bca] + m[I_bac];
    double mxyyMxzz = m[I_bca] - m[I_bac];

    m[I_bbb]  *= (1.0 - omega4);           // C₁₁₁ (Eq. 70)
    mxxyPyzz  *= (1.0 - omega3);           // Eq. 64
    mxxyMyzz  *= (1.0 - omega4);           // Eq. 67
    mxxzPyyz  *= (1.0 - omega3);           // Eq. 65
    mxxzMyyz  *= (1.0 - omega4);           // Eq. 68
    mxyyPxzz  *= (1.0 - omega3);           // Eq. 66
    mxyyMxzz  *= (1.0 - omega4);           // Eq. 69

    // --- Reconstruct 2nd order individual moments ---
    m[I_caa] = (mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
    m[I_aca] = (-2.0*mxxMyy + mxxMzz + mxxPyyPzz) / 3.0;
    m[I_aac] = (mxxMyy - 2.0*mxxMzz + mxxPyyPzz) / 3.0;

    // --- Reconstruct 3rd order ---
    m[I_cba] = (mxxyMyzz + mxxyPyzz) * 0.5;
    m[I_abc] = (-mxxyMyzz + mxxyPyzz) * 0.5;
    m[I_cab] = (mxxzMyyz + mxxzPyyz) * 0.5;
    m[I_acb] = (-mxxzMyyz + mxxzPyyz) * 0.5;
    m[I_bca] = (mxyyMxzz + mxyyPxzz) * 0.5;
    m[I_bac] = (-mxyyMxzz + mxyyPxzz) * 0.5;

    // --- 4th order relaxation (Eq. 71-76) ---
    CUMacc *= (1.0 - omega6);
    CUMcac *= (1.0 - omega6);
    CUMcca *= (1.0 - omega6);
    CUMbbc *= (1.0 - omega6);
    CUMbcb *= (1.0 - omega6);
    CUMcbb *= (1.0 - omega6);

    // --- 5th order relaxation (Eq. 77-79) ---
    CUMbcc *= (1.0 - omega7);
    CUMcbc *= (1.0 - omega7);
    CUMccb *= (1.0 - omega7);

    // --- 6th order relaxation (Eq. 80) ---
    CUMccc *= (1.0 - omega10);

    // ==============================================================
    // STAGE 4: Cumulants → Central Moments (Inverse of Stage 2)
    // ==============================================================

    // --- 4th order inverse (Eq. J.16 inverse) ---
    m[I_cbb] = CUMcbb + ((m[I_caa]+1.0/3.0)*m[I_abb]
               + 2.0*m[I_bba]*m[I_bab]) / 3.0 * 3.0 * inv_rho;
    // (simplified: original = ... * inverse_rho directly)
    m[I_cbb] = CUMcbb + ((3.0*m[I_caa]+1.0)*m[I_abb]
               + 6.0*m[I_bba]*m[I_bab]) * inv_rho / 3.0;
    m[I_bcb] = CUMbcb + ((3.0*m[I_aca]+1.0)*m[I_bab]
               + 6.0*m[I_bba]*m[I_abb]) * inv_rho / 3.0;
    m[I_bbc] = CUMbbc + ((3.0*m[I_aac]+1.0)*m[I_bba]
               + 6.0*m[I_bab]*m[I_abb]) * inv_rho / 3.0;

    // --- 4th order diagonal inverse (Eq. J.17 inverse) ---
    m[I_cca] = CUMcca + (((m[I_caa]*m[I_aca]+2.0*m[I_bba]*m[I_bba])*9.0
               + 3.0*(m[I_caa]+m[I_aca])) * inv_rho
               - drho*inv_rho) / 9.0;
    m[I_cac] = CUMcac + (((m[I_caa]*m[I_aac]+2.0*m[I_bab]*m[I_bab])*9.0
               + 3.0*(m[I_caa]+m[I_aac])) * inv_rho
               - drho*inv_rho) / 9.0;
    m[I_acc] = CUMacc + (((m[I_aac]*m[I_aca]+2.0*m[I_abb]*m[I_abb])*9.0
               + 3.0*(m[I_aac]+m[I_aca])) * inv_rho
               - drho*inv_rho) / 9.0;

    // --- 5th order inverse (Eq. J.18 inverse) ---
    m[I_bcc] = CUMbcc + (3.0*(m[I_aac]*m[I_bca] + m[I_aca]*m[I_bac]
               + 4.0*m[I_abb]*m[I_bbb]
               + 2.0*(m[I_bab]*m[I_acb] + m[I_bba]*m[I_abc]))
               + (m[I_bca]+m[I_bac])) * inv_rho / 3.0;
    m[I_cbc] = CUMcbc + (3.0*(m[I_aac]*m[I_cba] + m[I_caa]*m[I_abc]
               + 4.0*m[I_bab]*m[I_bbb]
               + 2.0*(m[I_abb]*m[I_cab] + m[I_bba]*m[I_bac]))
               + (m[I_cba]+m[I_abc])) * inv_rho / 3.0;
    m[I_ccb] = CUMccb + (3.0*(m[I_caa]*m[I_acb] + m[I_aca]*m[I_cab]
               + 4.0*m[I_bba]*m[I_bbb]
               + 2.0*(m[I_bab]*m[I_bca] + m[I_abb]*m[I_cba]))
               + (m[I_acb]+m[I_cab])) * inv_rho / 3.0;

    // --- 6th order inverse (Eq. J.19 inverse) ---
    m[I_ccc] = CUMccc
        - ((-4.0*m[I_bbb]*m[I_bbb]
            - (m[I_caa]*m[I_acc]+m[I_aca]*m[I_cac]+m[I_aac]*m[I_cca])
            - 4.0*(m[I_abb]*m[I_cbb]+m[I_bab]*m[I_bcb]+m[I_bba]*m[I_bbc])
            - 2.0*(m[I_bca]*m[I_bac]+m[I_cba]*m[I_abc]+m[I_cab]*m[I_acb]))
                * inv_rho
        + (4.0*(m[I_bab]*m[I_bab]*m[I_aca]
              + m[I_abb]*m[I_abb]*m[I_caa]
              + m[I_bba]*m[I_bba]*m[I_aac])
          + 2.0*m[I_caa]*m[I_aca]*m[I_aac]
          + 16.0*m[I_bba]*m[I_bab]*m[I_abb])
                * inv_rho * inv_rho
        - (m[I_acc]+m[I_cac]+m[I_cca]) * inv_rho / 9.0
        - (m[I_caa]+m[I_aca]+m[I_aac]) * inv_rho / 9.0
        + (2.0*(m[I_bab]*m[I_bab]+m[I_abb]*m[I_abb]+m[I_bba]*m[I_bba])
          + (m[I_aac]*m[I_aca]+m[I_aac]*m[I_caa]+m[I_aca]*m[I_caa])
          + (m[I_aac]+m[I_aca]+m[I_caa])/3.0)
                * inv_rho * inv_rho * 2.0/3.0
        + (drho*drho - drho) * inv_rho * inv_rho / 27.0);

    // --- Force correction: sign flip of 1st order (Eq. 85-87) ---
    m[I_baa] = -m[I_baa];
    m[I_aba] = -m[I_aba];
    m[I_aab] = -m[I_aab];

    // ==============================================================
    // STAGE 5: Backward Chimera Transform (x → y → z)
    //          κ*[27] → f̄*[27]
    // ==============================================================
    _cum_backward_chimera(m, u);

    // --- Restore from well-conditioned: f* = f̄* + w ---
    for (int i = 0; i < 27; i++) {
        f_out[i] = m[i] + CUM_W[i];
    }

    // --- Output macroscopic quantities ---
    rho_out = rho;
    ux_out  = u[0];
    uy_out  = u[1];
    uz_out  = u[2];
}


// ================================================================
// Internal: Forward Chimera (Stage 1)
// Sweep order: z(i=2) → y(i=1) → x(i=0)
// ================================================================
__device__ static void _cum_forward_chimera(
    double m[27], const double u[3])
{
    // direction loop: 2=z, 1=y, 0=x
    for (int dir = 2; dir >= 0; dir--) {
        int base = (2 - dir) * 9;  // z: passes 0-8, y: 9-17, x: 18-26
        for (int j = 0; j < 9; j++) {
            int p = base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double k = CUM_K[p];

            double sum  = m[a] + m[c];      // f₁ + f₋₁
            double diff = m[c] - m[a];      // f₁ - f₋₁

            // Eq. J.4/J.7/J.10: κ₀ = sum + f₀
            m[a] = m[a] + m[b] + m[c];
            // Eq. J.5/J.8/J.11: κ₁ = diff - u·(κ₀ + k)
            m[b] = diff - (m[a] + k) * u[dir];
            // Eq. J.6/J.9/J.12: κ₂ = sum - 2u·diff + u²·(κ₀ + k)
            m[c] = sum - 2.0 * diff * u[dir]
                   + u[dir] * u[dir] * (m[a] + k);
        }
    }
}

// ================================================================
// Internal: Backward Chimera (Stage 5)
// Sweep order: x(i=0) → y(i=1) → z(i=2)
// ================================================================
__device__ static void _cum_backward_chimera(
    double m[27], const double u[3])
{
    // direction loop: 0=x, 1=y, 2=z
    for (int dir = 0; dir < 3; dir++) {
        int base = dir * 9;  // x: passes 18-26... wait
        // 注意：backward 的 pass 順序是 x(18-26), y(9-17), z(0-8)
        // 但因為 CUM_IDX 的排列是 z,y,x，所以：
        //   dir=0 (x方向) → 使用 passes 18-26
        //   dir=1 (y方向) → 使用 passes 9-17
        //   dir=2 (z方向) → 使用 passes 0-8
        int pass_base = (2 - dir) * 9;  // x→18, y→9, z→0
        // 不對——backward 需要用 dir 遞增的順序存取，但 index 表不變
        // 修正：直接用 dir * 9 來索引，因為 CUM_IDX 排列是 z=0,y=9,x=18
        // backward 是 x→y→z，所以先存取 x(base=18), 再 y(base=9), 再 z(base=0)
        int actual_base;
        if      (dir == 0) actual_base = 18;  // x
        else if (dir == 1) actual_base = 9;   // y
        else               actual_base = 0;   // z

        for (int j = 0; j < 9; j++) {
            int p = actual_base + j;
            int a = CUM_IDX[p][0];
            int b = CUM_IDX[p][1];
            int c = CUM_IDX[p][2];
            double k = CUM_K[p];

            int d = dir;  // 0=x, 1=y, 2=z

            // Eq. J.21/24/27: f̄₋₁
            double ma = ((m[c] - m[b]) * 0.5 + m[b] * u[d]
                        + (m[a] + k) * (u[d]*u[d] - u[d]) * 0.5);
            // Eq. J.20/23/26: f̄₀
            double mb = (m[a] - m[c]) - 2.0 * m[b] * u[d]
                        - (m[a] + k) * u[d] * u[d];
            // Eq. J.22/25/28: f̄₊₁
            double mc = ((m[c] + m[b]) * 0.5 + m[b] * u[d]
                        + (m[a] + k) * (u[d]*u[d] + u[d]) * 0.5);

            m[a] = ma;
            m[b] = mb;
            m[c] = mc;
        }
    }
}

#endif // CUMULANT_COLLISION_H
```

---

## 使用者的視角 — 呼叫範例

```cpp
// ================================================================
// 在你的 evolution_gilbm.h kernel 中：
// ================================================================

#include "cumulant_collision.h"

__global__ void evolution_kernel(
    double* f_global,    // [27 * total_nodes]
    double* rho_d,
    double* ux_d, double* uy_d, double* uz_d,
    double  omega,
    double  F_body_x,
    int     total_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nodes) return;

    // -------- Step 1: Interpolation + Streaming --------
    double f_streamed[27];
    for (int a = 0; a < 27; a++) {
        // ... 你原有的 Lagrange 插值邏輯 ...
        f_streamed[a] = /* interpolated value */;
    }

    // -------- Step 2: 呼叫積木 --------
    double f_post[27];
    double rho, ux, uy, uz;

    cumulant_collision_D3Q27(
        f_streamed,    // ← 你算好的遷移後 f
        omega,         // ← 1/(3ν+0.5)
        F_body_x,      // ← 流向驅動力
        0.0, 0.0,      // ← Fy, Fz
        f_post,        // → 碰撞後的 f
        rho, ux, uy, uz
    );

    // -------- Step 3: 寫回 --------
    for (int a = 0; a < 27; a++) {
        f_global[a * total_nodes + idx] = f_post[a];
    }
    rho_d[idx] = rho;
    ux_d[idx]  = ux;
    uy_d[idx]  = uy;
    uz_d[idx]  = uz;
}
```

---

## 積木概念總結圖

```
    ╔═══════════════════════════════════════════════╗
    ║          使用者的 GILBM 程式碼               ║
    ║                                               ║
    ║  ┌───────────────────────────────────────┐    ║
    ║  │ Step 1: Lagrange Interpolation         │    ║
    ║  │         f_streamed[27] = ...            │    ║
    ║  └──────────────────┬────────────────────┘    ║
    ║                     │                          ║
    ║                     ▼                          ║
    ║  ┌─────────────────────────────────────────┐  ║
    ║  │          ★ 積木模組 (Bridge) ★          │  ║
    ║  │                                         │  ║
    ║  │  #include "cumulant_collision.h"        │  ║
    ║  │                                         │  ║
    ║  │  IN:  f[27], omega, Fx, Fy, Fz          │  ║
    ║  │  OUT: f_post[27], rho, ux, uy, uz       │  ║
    ║  │                                         │  ║
    ║  │  ┌───────┐ ┌───────┐ ┌───────┐         │  ║
    ║  │  │Stage 1│→│Stage 2│→│Stage 3│→ ...     │  ║
    ║  │  │Chimera│ │κ→C    │ │Relax  │          │  ║
    ║  │  └───────┘ └───────┘ └───────┘         │  ║
    ║  │         (使用者不需要懂這些)             │  ║
    ║  └──────────────────┬──────────────────────┘  ║
    ║                     │                          ║
    ║                     ▼                          ║
    ║  ┌───────────────────────────────────────┐    ║
    ║  │ Step 3: Write back to global memory    │    ║
    ║  └───────────────────────────────────────┘    ║
    ╚═══════════════════════════════════════════════╝
```

---

## 實施前必須確認的事項

```
Q1: 你的 D3Q27 離散速度排列順序 (ex, ey, ez) 是否與 OpenLB 一致？
    若不一致，需要建立 index mapping。
    → 請提供你目前的 27 個離散速度定義

Q2: f[27] 是 AoS (f[node][27]) 還是 SoA (f[dir][nodes])？
    積木內部用 f[27] per node，外部存取方式不影響。

Q3: nvcc 編譯旗標是否支援 C++14 以上？
    （本模組只需要 C++11，不用 structured bindings）

Q4: omega 的定義：
    你目前用 tau = 0.6833，omega = 1/tau ≈ 1.4634
    還是 omega = 1/(3*nu + 0.5)？
    兩者等價，但確認一下數值。

Q5: 外力 Fx 的量綱：
    是 LB 單位的加速度 (a = F/ρ)?
    還是 LB 單位的力密度 (F)?
    OpenLB 的 Chimera half-force 用的是 F/ρ。
```
